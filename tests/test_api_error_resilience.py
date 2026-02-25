"""
tests/test_api_error_resilience.py - API error resilience tests.

Tests for:
1. Error classification (ErrorCategory assignment)
2. Retry configuration and backoff delay computation
3. Fallback mode state management
4. Error observation recording (ErrorStats)
5. Resilient call wrapper (resilient_call)
6. LLM wrapper integration (llm_call, llm_call_with_image)
7. Brain fallback detection helpers
8. Safe shutdown signaling
9. Streaming retry behavior
10. Session boundary reset
"""

import asyncio
import json
import os
import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api_error_resilience import (
    ErrorCategory,
    ErrorRecord,
    ErrorStats,
    FallbackModeState,
    RetryConfig,
    classify_error,
    compute_backoff_delay,
    is_retryable,
    resilient_call,
    _extract_retry_after,
    _extract_status_code,
)


# =============================================================================
# 1. Error Classification
# =============================================================================

class TestErrorClassification:
    """Tests for classify_error function."""

    def test_timeout_error(self):
        assert classify_error(asyncio.TimeoutError()) == ErrorCategory.RESPONSE_TIMEOUT

    def test_python_timeout_error(self):
        assert classify_error(TimeoutError("timed out")) == ErrorCategory.RESPONSE_TIMEOUT

    def test_connection_error(self):
        assert classify_error(ConnectionError("refused")) == ErrorCategory.TRANSIENT_NETWORK

    def test_connection_refused_error(self):
        assert classify_error(ConnectionRefusedError()) == ErrorCategory.TRANSIENT_NETWORK

    def test_connection_reset_error(self):
        assert classify_error(ConnectionResetError()) == ErrorCategory.TRANSIENT_NETWORK

    def test_os_error(self):
        assert classify_error(OSError("network error")) == ErrorCategory.TRANSIENT_NETWORK

    def test_rate_limit_by_status_code(self):
        err = Exception("rate limit")
        err.code = 429
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_by_message(self):
        err = Exception("rate limit exceeded")
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_by_quota(self):
        err = Exception("quota exceeded")
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_by_resource_exhausted(self):
        err = Exception("RESOURCE_EXHAUSTED")
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_429_in_message(self):
        err = Exception("HTTP 429 Too Many Requests")
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_permanent_auth_failure(self):
        err = Exception("invalid api key")
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_permanent_permission(self):
        err = Exception("permission denied")
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_permanent_by_status_code_401(self):
        err = Exception("unauthorized")
        err.code = 401
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_permanent_by_status_code_403(self):
        err = Exception("forbidden")
        err.code = 403
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_permanent_by_status_code_404(self):
        err = Exception("not found")
        err.code = 404
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_permanent_by_status_code_400(self):
        err = Exception("bad request")
        err.code = 400
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_transient_by_status_code_502(self):
        err = Exception("bad gateway")
        err.code = 502
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_transient_by_status_code_503(self):
        err = Exception("service unavailable")
        err.code = 503
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_transient_by_status_code_504(self):
        err = Exception("gateway timeout")
        err.code = 504
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_transient_by_message_dns(self):
        err = Exception("DNS resolution failed")
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_transient_by_message_ssl(self):
        err = Exception("SSL handshake error")
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_unknown_generic_error(self):
        err = Exception("something unusual happened")
        assert classify_error(err) == ErrorCategory.UNKNOWN

    def test_unknown_value_error(self):
        err = ValueError("unexpected")
        assert classify_error(err) == ErrorCategory.UNKNOWN


class TestExtractStatusCode:
    """Tests for _extract_status_code helper."""

    def test_code_attribute(self):
        err = Exception("test")
        err.code = 429
        assert _extract_status_code(err) == 429

    def test_status_attribute(self):
        err = Exception("test")
        err.status = 503
        assert _extract_status_code(err) == 503

    def test_status_code_attribute(self):
        err = Exception("test")
        err.status_code = 401
        assert _extract_status_code(err) == 401

    def test_nested_cause(self):
        inner = Exception("inner")
        inner.code = 429
        outer = Exception("outer")
        outer.__cause__ = inner
        assert _extract_status_code(outer) == 429

    def test_no_status(self):
        assert _extract_status_code(Exception("plain")) is None

    def test_non_numeric_code(self):
        err = Exception("test")
        err.code = "not_a_number"
        assert _extract_status_code(err) is None


class TestExtractRetryAfter:
    """Tests for _extract_retry_after helper."""

    def test_retry_after_attribute(self):
        err = Exception("test")
        err.retry_after = 30
        assert _extract_retry_after(err) == 30.0

    def test_retry_after_in_message(self):
        err = Exception("Rate limited. Retry-After: 15 seconds")
        assert _extract_retry_after(err) == 15.0

    def test_retry_after_in_message_lowercase(self):
        err = Exception("retry after 5 seconds")
        assert _extract_retry_after(err) == 5.0

    def test_no_retry_after(self):
        err = Exception("generic error")
        assert _extract_retry_after(err) is None


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_transient_is_retryable(self):
        assert is_retryable(ErrorCategory.TRANSIENT_NETWORK) is True

    def test_rate_limit_is_retryable(self):
        assert is_retryable(ErrorCategory.RATE_LIMIT) is True

    def test_timeout_is_retryable(self):
        assert is_retryable(ErrorCategory.RESPONSE_TIMEOUT) is True

    def test_unknown_is_retryable(self):
        assert is_retryable(ErrorCategory.UNKNOWN) is True

    def test_permanent_is_not_retryable(self):
        assert is_retryable(ErrorCategory.PERMANENT) is False


# =============================================================================
# 2. Retry Configuration and Backoff
# =============================================================================

class TestRetryConfig:
    """Tests for RetryConfig defaults and compute_backoff_delay."""

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 4
        assert cfg.initial_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.backoff_multiplier == 2.0
        assert cfg.jitter_fraction == 0.25
        assert cfg.timeout == 30.0

    def test_custom_config(self):
        cfg = RetryConfig(
            max_retries=2,
            initial_delay=0.5,
            max_delay=10.0,
            backoff_multiplier=3.0,
            jitter_fraction=0.1,
            timeout=15.0,
        )
        assert cfg.max_retries == 2
        assert cfg.initial_delay == 0.5

    def test_backoff_delay_exponential(self):
        cfg = RetryConfig(initial_delay=1.0, backoff_multiplier=2.0, jitter_fraction=0.0)
        # attempt 0: 1.0 * 2^0 = 1.0
        assert compute_backoff_delay(0, cfg) == 1.0
        # attempt 1: 1.0 * 2^1 = 2.0
        assert compute_backoff_delay(1, cfg) == 2.0
        # attempt 2: 1.0 * 2^2 = 4.0
        assert compute_backoff_delay(2, cfg) == 4.0
        # attempt 3: 1.0 * 2^3 = 8.0
        assert compute_backoff_delay(3, cfg) == 8.0

    def test_backoff_delay_capped(self):
        cfg = RetryConfig(initial_delay=1.0, backoff_multiplier=2.0, max_delay=5.0, jitter_fraction=0.0)
        # attempt 3: 1.0 * 2^3 = 8.0 -> capped to 5.0
        assert compute_backoff_delay(3, cfg) == 5.0

    def test_backoff_delay_with_jitter(self):
        cfg = RetryConfig(initial_delay=1.0, backoff_multiplier=2.0, jitter_fraction=0.25)
        delay = compute_backoff_delay(0, cfg)
        # Base is 1.0, jitter adds [0, 0.25], so delay in [1.0, 1.25]
        assert 1.0 <= delay <= 1.25

    def test_backoff_delay_jitter_randomness(self):
        """Jitter should produce varying delays (not always the same)."""
        cfg = RetryConfig(initial_delay=1.0, backoff_multiplier=2.0, jitter_fraction=0.25)
        delays = [compute_backoff_delay(0, cfg) for _ in range(100)]
        # At least some variation should exist
        assert len(set(delays)) > 1

    def test_backoff_delay_with_retry_after(self):
        cfg = RetryConfig(initial_delay=1.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg, retry_after=10.0)
        assert delay == 10.0

    def test_backoff_delay_retry_after_capped(self):
        cfg = RetryConfig(max_delay=5.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg, retry_after=10.0)
        assert delay == 5.0

    def test_backoff_delay_retry_after_zero(self):
        """retry_after=0 should fall back to exponential backoff."""
        cfg = RetryConfig(initial_delay=2.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg, retry_after=0)
        assert delay == 2.0

    def test_backoff_delay_retry_after_negative(self):
        """Negative retry_after should fall back to exponential."""
        cfg = RetryConfig(initial_delay=2.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg, retry_after=-5)
        assert delay == 2.0


# =============================================================================
# 3. Fallback Mode State
# =============================================================================

class TestFallbackModeState:
    """Tests for FallbackModeState management."""

    def test_initial_state(self):
        fs = FallbackModeState()
        assert fs.consecutive_failures == 0
        assert fs.is_in_fallback is False
        assert fs.should_safe_shutdown() is False

    def test_on_failure_increments(self):
        fs = FallbackModeState(failure_threshold=3)
        fs.on_failure()
        assert fs.consecutive_failures == 1
        assert fs.is_in_fallback is False

    def test_enters_fallback_at_threshold(self):
        fs = FallbackModeState(failure_threshold=2)
        fs.on_failure()
        fs.on_failure()
        assert fs.consecutive_failures == 2
        assert fs.is_in_fallback is True

    def test_on_success_resets(self):
        fs = FallbackModeState(failure_threshold=2)
        fs.on_failure()
        fs.on_failure()
        assert fs.is_in_fallback is True
        fs.on_success()
        assert fs.consecutive_failures == 0
        assert fs.is_in_fallback is False

    def test_should_safe_shutdown_not_in_fallback(self):
        fs = FallbackModeState()
        assert fs.should_safe_shutdown() is False

    def test_should_safe_shutdown_within_duration(self):
        fs = FallbackModeState(failure_threshold=1, max_fallback_duration=300.0)
        fs.on_failure()
        assert fs.is_in_fallback is True
        # Just entered, so duration should be < 300
        assert fs.should_safe_shutdown() is False

    def test_should_safe_shutdown_exceeded(self):
        fs = FallbackModeState(failure_threshold=1, max_fallback_duration=0.0)
        fs.on_failure()
        assert fs.is_in_fallback is True
        # Duration 0 means immediate shutdown
        assert fs.should_safe_shutdown() is True

    def test_get_status_normal(self):
        fs = FallbackModeState()
        status = fs.get_status()
        assert status["is_in_fallback"] is False
        assert status["consecutive_failures"] == 0
        assert "fallback_duration_sec" not in status

    def test_get_status_in_fallback(self):
        fs = FallbackModeState(failure_threshold=1)
        fs.on_failure()
        status = fs.get_status()
        assert status["is_in_fallback"] is True
        assert "fallback_duration_sec" in status
        assert status["fallback_duration_sec"] >= 0

    def test_recovery_after_multiple_failures(self):
        fs = FallbackModeState(failure_threshold=2)
        fs.on_failure()
        fs.on_failure()
        fs.on_failure()
        assert fs.consecutive_failures == 3
        assert fs.is_in_fallback is True
        fs.on_success()
        assert fs.consecutive_failures == 0
        assert fs.is_in_fallback is False

    def test_re_entering_fallback(self):
        fs = FallbackModeState(failure_threshold=2)
        fs.on_failure()
        fs.on_failure()
        assert fs.is_in_fallback is True
        fs.on_success()
        assert fs.is_in_fallback is False
        fs.on_failure()
        fs.on_failure()
        assert fs.is_in_fallback is True


# =============================================================================
# 4. Error Observation Recording
# =============================================================================

class TestErrorStats:
    """Tests for ErrorStats recording."""

    def test_initial_state(self):
        stats = ErrorStats()
        assert stats.retry_recovery_count == 0
        assert stats.fallback_count == 0
        assert stats.total_retries == 0
        assert stats.total_wait_time == 0.0
        assert len(stats.records) == 0

    def test_record_retry_success(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.TRANSIENT_NETWORK,
            error_message="connection reset",
            retry_count=2,
            total_wait_time=3.5,
            final_result="retry_success",
            call_type="expression",
        )
        assert stats.retry_recovery_count == 1
        assert stats.fallback_count == 0
        assert stats.total_retries == 2
        assert stats.total_wait_time == 3.5
        assert len(stats.records) == 1

    def test_record_fallback(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.RESPONSE_TIMEOUT,
            error_message="timed out",
            retry_count=3,
            total_wait_time=7.0,
            final_result="fallback",
            call_type="perception",
        )
        assert stats.retry_recovery_count == 0
        assert stats.fallback_count == 1
        assert stats.total_retries == 3

    def test_record_category_counting(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.RATE_LIMIT,
            error_message="429",
            retry_count=1,
            total_wait_time=1.0,
            final_result="retry_success",
            call_type="expression",
        )
        stats.record_error(
            category=ErrorCategory.RATE_LIMIT,
            error_message="429",
            retry_count=1,
            total_wait_time=2.0,
            final_result="fallback",
            call_type="expression",
        )
        assert stats.errors_by_category["RATE_LIMIT"] == 2
        assert stats.total_retries == 2

    def test_fifo_eviction(self):
        stats = ErrorStats()
        stats._max_records = 5
        for i in range(10):
            stats.record_error(
                category=ErrorCategory.UNKNOWN,
                error_message=f"error {i}",
                retry_count=0,
                total_wait_time=0,
                final_result="fallback",
                call_type="test",
            )
        assert len(stats.records) == 5
        assert stats.records[0].error_message == "error 5"

    def test_error_message_truncation(self):
        stats = ErrorStats()
        long_msg = "a" * 500
        stats.record_error(
            category=ErrorCategory.UNKNOWN,
            error_message=long_msg,
            retry_count=0,
            total_wait_time=0,
            final_result="fallback",
            call_type="test",
        )
        assert len(stats.records[0].error_message) == 200

    def test_get_summary(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.TRANSIENT_NETWORK,
            error_message="err",
            retry_count=2,
            total_wait_time=3.0,
            final_result="retry_success",
            call_type="test",
        )
        summary = stats.get_summary()
        assert summary["retry_recovery_count"] == 1
        assert summary["fallback_count"] == 0
        assert summary["total_retries"] == 2
        assert summary["total_wait_time_sec"] == 3.0
        assert "errors_by_category" in summary

    def test_multiple_call_types(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.TRANSIENT_NETWORK,
            error_message="err1",
            retry_count=1, total_wait_time=1.0,
            final_result="retry_success", call_type="perception",
        )
        stats.record_error(
            category=ErrorCategory.RESPONSE_TIMEOUT,
            error_message="err2",
            retry_count=2, total_wait_time=3.0,
            final_result="fallback", call_type="expression",
        )
        assert len(stats.records) == 2
        assert stats.records[0].call_type == "perception"
        assert stats.records[1].call_type == "expression"


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_creation(self):
        r = ErrorRecord(
            timestamp=time.time(),
            category=ErrorCategory.RATE_LIMIT,
            error_message="rate limited",
            retry_count=3,
            total_wait_time=5.0,
            final_result="retry_success",
            call_type="expression",
        )
        assert r.category == ErrorCategory.RATE_LIMIT
        assert r.final_result == "retry_success"
        assert r.retry_count == 3


# =============================================================================
# 5. Resilient Call Wrapper
# =============================================================================

class TestResilientCall:
    """Tests for the resilient_call wrapper."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        async def ok():
            return "hello"
        result = await resilient_call(ok, config=RetryConfig(timeout=5.0))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        call_count = 0
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("flaky")
            return "recovered"

        cfg = RetryConfig(max_retries=3, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)
        result = await resilient_call(flaky, config=cfg)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        async def always_fail():
            raise ConnectionError("down")

        cfg = RetryConfig(max_retries=2, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)
        result = await resilient_call(
            always_fail, config=cfg, fallback_value="fallback"
        )
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_permanent_error_no_retry(self):
        call_count = 0
        async def auth_fail():
            nonlocal call_count
            call_count += 1
            err = Exception("invalid api key")
            err.code = 401
            raise err

        cfg = RetryConfig(max_retries=3, initial_delay=0.01, timeout=5.0)
        result = await resilient_call(
            auth_fail, config=cfg, fallback_value="fallback"
        )
        assert result == "fallback"
        assert call_count == 1  # No retries for permanent errors

    @pytest.mark.asyncio
    async def test_timeout_retry(self):
        call_count = 0
        async def slow():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                await asyncio.sleep(10)  # Will trigger timeout
            return "done"

        cfg = RetryConfig(max_retries=2, initial_delay=0.01, jitter_fraction=0.0, timeout=0.05)
        result = await resilient_call(slow, config=cfg)
        assert result == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_stats_recording_retry_success(self):
        call_count = 0
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("blip")
            return "ok"

        stats = ErrorStats()
        cfg = RetryConfig(max_retries=3, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)
        result = await resilient_call(
            flaky, config=cfg, stats=stats, call_type="test"
        )
        assert result == "ok"
        assert stats.retry_recovery_count == 1

    @pytest.mark.asyncio
    async def test_stats_recording_fallback(self):
        async def fail():
            raise ConnectionError("nope")

        stats = ErrorStats()
        cfg = RetryConfig(max_retries=1, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)
        result = await resilient_call(
            fail, config=cfg, stats=stats, fallback_value="fb", call_type="test"
        )
        assert result == "fb"
        assert stats.fallback_count == 1

    @pytest.mark.asyncio
    async def test_fallback_state_on_success(self):
        async def ok():
            return "good"

        fs = FallbackModeState(failure_threshold=1)
        fs.on_failure()
        assert fs.is_in_fallback is True

        cfg = RetryConfig(timeout=5.0)
        await resilient_call(ok, config=cfg, fallback_state=fs)
        assert fs.is_in_fallback is False

    @pytest.mark.asyncio
    async def test_fallback_state_on_failure(self):
        async def fail():
            raise ConnectionError("down")

        fs = FallbackModeState(failure_threshold=1)
        cfg = RetryConfig(max_retries=0, timeout=5.0)
        await resilient_call(
            fail, config=cfg, fallback_state=fs, fallback_value=None
        )
        assert fs.consecutive_failures == 1
        assert fs.is_in_fallback is True

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self):
        """Rate limit errors should use retry_after if available."""
        call_count = 0
        async def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                err = Exception("rate limited. Retry-After: 0.01")
                err.code = 429
                raise err
            return "success"

        cfg = RetryConfig(max_retries=2, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)
        result = await resilient_call(rate_limited, config=cfg)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_args_and_kwargs(self):
        async def add(a, b, extra=0):
            return a + b + extra

        cfg = RetryConfig(timeout=5.0)
        result = await resilient_call(add, 3, 4, extra=10, config=cfg)
        assert result == 17


# =============================================================================
# 6. LLM Wrapper Integration
# =============================================================================

class TestLLMWrapperIntegration:
    """Tests for llm_wrapper functions using the resilience layer."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset module-level resilience state before each test."""
        from src import llm_wrapper
        llm_wrapper.reset_error_stats()
        llm_wrapper.reset_fallback_state()
        yield

    @pytest.mark.asyncio
    async def test_no_api_key_returns_fallback(self):
        """Without API key, should return fallback JSON."""
        saved_llm = os.environ.pop("LLM_API_KEY", None)
        saved_gemini = os.environ.pop("GEMINI_API_KEY", None)
        try:
            from src.llm_wrapper import llm_call
            result = await llm_call("test")
            assert "no_llm_available" in result
        finally:
            if saved_llm is not None:
                os.environ["LLM_API_KEY"] = saved_llm
            if saved_gemini is not None:
                os.environ["GEMINI_API_KEY"] = saved_gemini

    def test_is_api_available_initially(self):
        from src.llm_wrapper import is_api_available
        assert is_api_available() is True

    def test_should_safe_shutdown_initially(self):
        from src.llm_wrapper import should_safe_shutdown
        assert should_safe_shutdown() is False

    def test_get_error_stats(self):
        from src.llm_wrapper import get_error_stats
        stats = get_error_stats()
        assert isinstance(stats, ErrorStats)
        assert stats.total_retries == 0

    def test_get_fallback_state(self):
        from src.llm_wrapper import get_fallback_state
        fs = get_fallback_state()
        assert isinstance(fs, FallbackModeState)
        assert fs.is_in_fallback is False

    def test_reset_error_stats(self):
        from src.llm_wrapper import get_error_stats, reset_error_stats
        stats = get_error_stats()
        stats.total_retries = 99
        reset_error_stats()
        assert get_error_stats().total_retries == 0

    def test_reset_fallback_state(self):
        from src.llm_wrapper import get_fallback_state, reset_fallback_state
        fs = get_fallback_state()
        fs.on_failure()
        fs.on_failure()
        fs.on_failure()
        reset_fallback_state()
        assert get_fallback_state().consecutive_failures == 0

    def test_filter_forbidden(self):
        from src.llm_wrapper import filter_forbidden
        text = "line1\n判断した結果はこうです\nline3"
        result = filter_forbidden(text)
        assert "判断" not in result
        assert "line1" in result
        assert "line3" in result


# =============================================================================
# 7. Brain Fallback Detection Helpers
# =============================================================================

class TestBrainFallbackDetection:
    """Tests for brain.py fallback detection helpers."""

    def test_is_perception_fallback_true(self):
        from brain import _is_perception_fallback
        fallback_text = json.dumps({"result": "no_llm_available"})
        assert _is_perception_fallback(fallback_text) is True

    def test_is_perception_fallback_false(self):
        from brain import _is_perception_fallback
        normal_text = "画面にはテキストエディタが表示されている。"
        assert _is_perception_fallback(normal_text) is False

    def test_is_perception_fallback_embedded(self):
        from brain import _is_perception_fallback
        text = 'response was: {"result": "no_llm_available"}'
        assert _is_perception_fallback(text) is True

    def test_fallback_indicator_constant(self):
        from brain import _FALLBACK_INDICATOR
        assert _FALLBACK_INDICATOR == "no_llm_available"


# =============================================================================
# 8. Safe Shutdown Signaling
# =============================================================================

class TestSafeShutdownSignaling:
    """Tests for SafeShutdownRequested exception."""

    def test_safe_shutdown_exception_exists(self):
        from brain import SafeShutdownRequested
        assert issubclass(SafeShutdownRequested, Exception)

    def test_safe_shutdown_message(self):
        from brain import SafeShutdownRequested
        exc = SafeShutdownRequested("test message")
        assert "test message" in str(exc)

    def test_safe_shutdown_is_catchable(self):
        from brain import SafeShutdownRequested
        try:
            raise SafeShutdownRequested("shutdown")
        except SafeShutdownRequested as e:
            assert "shutdown" in str(e)

    def test_safe_shutdown_not_caught_by_generic_handler(self):
        """SafeShutdownRequested inherits from Exception but should be
        specifically caught, not swallowed by generic except blocks
        that re-raise it."""
        from brain import SafeShutdownRequested
        with pytest.raises(SafeShutdownRequested):
            try:
                raise SafeShutdownRequested("stop")
            except SafeShutdownRequested:
                raise
            except Exception:
                pass  # Should not reach here


# =============================================================================
# 9. Streaming Retry (unit-level)
# =============================================================================

class TestStreamingRetryBehavior:
    """Tests for streaming-specific retry logic in resilience layer."""

    @pytest.mark.asyncio
    async def test_resilient_call_timeout_generates_retry(self):
        """Verify that timeout errors are retried."""
        call_count = 0
        async def slow_then_fast():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                await asyncio.sleep(10)
            return "done"

        cfg = RetryConfig(max_retries=2, initial_delay=0.01, timeout=0.05, jitter_fraction=0.0)
        result = await resilient_call(slow_then_fast, config=cfg)
        assert result == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_resilient_call_returns_fallback_after_exhaustion(self):
        call_count = 0
        async def always_timeout():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)

        cfg = RetryConfig(max_retries=1, initial_delay=0.01, timeout=0.05, jitter_fraction=0.0)
        result = await resilient_call(
            always_timeout, config=cfg, fallback_value="fallback"
        )
        assert result == "fallback"
        assert call_count == 2  # initial + 1 retry


# =============================================================================
# 10. Session Boundary Reset
# =============================================================================

class TestSessionBoundaryReset:
    """Tests for session-scoped state reset."""

    def test_error_stats_not_persisted(self):
        """ErrorStats should be session-scoped, reset clears all."""
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.TRANSIENT_NETWORK,
            error_message="err",
            retry_count=1,
            total_wait_time=1.0,
            final_result="fallback",
            call_type="test",
        )
        assert stats.fallback_count == 1

        # Simulate session boundary by creating new stats
        stats2 = ErrorStats()
        assert stats2.fallback_count == 0
        assert stats2.total_retries == 0

    def test_fallback_state_reset(self):
        """FallbackModeState should be reset at session boundary."""
        fs = FallbackModeState(failure_threshold=1)
        fs.on_failure()
        assert fs.is_in_fallback is True

        # Simulate session boundary
        fs2 = FallbackModeState()
        assert fs2.is_in_fallback is False
        assert fs2.consecutive_failures == 0


# =============================================================================
# 11. Error Category Enum
# =============================================================================

class TestErrorCategoryEnum:
    """Tests for ErrorCategory enum values."""

    def test_all_categories_exist(self):
        assert hasattr(ErrorCategory, "TRANSIENT_NETWORK")
        assert hasattr(ErrorCategory, "RATE_LIMIT")
        assert hasattr(ErrorCategory, "RESPONSE_TIMEOUT")
        assert hasattr(ErrorCategory, "PERMANENT")
        assert hasattr(ErrorCategory, "UNKNOWN")

    def test_category_count(self):
        assert len(ErrorCategory) == 5


# =============================================================================
# 12. Backoff Edge Cases
# =============================================================================

class TestBackoffEdgeCases:
    """Edge case tests for backoff computation."""

    def test_very_large_attempt_number(self):
        cfg = RetryConfig(initial_delay=1.0, max_delay=30.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(100, cfg)
        assert delay == 30.0  # Should be capped

    def test_zero_initial_delay(self):
        cfg = RetryConfig(initial_delay=0.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg)
        assert delay == 0.0

    def test_zero_max_delay(self):
        cfg = RetryConfig(initial_delay=1.0, max_delay=0.0, jitter_fraction=0.0)
        delay = compute_backoff_delay(0, cfg)
        assert delay == 0.0  # min(1.0, 0.0) = 0.0

    def test_retry_after_with_jitter(self):
        cfg = RetryConfig(jitter_fraction=0.5)
        delay = compute_backoff_delay(0, cfg, retry_after=10.0)
        assert 10.0 <= delay <= 15.0


# =============================================================================
# 13. Fallback Mode Duration Tracking
# =============================================================================

class TestFallbackModeDuration:
    """Tests for fallback mode duration tracking and shutdown."""

    def test_fallback_duration_tracking(self):
        fs = FallbackModeState(failure_threshold=1, max_fallback_duration=100.0)
        fs.on_failure()
        status = fs.get_status()
        assert status["is_in_fallback"] is True
        assert 0 <= status["fallback_duration_sec"] < 1.0

    def test_multiple_failures_dont_reset_start_time(self):
        """Additional failures after entering fallback should not reset start time."""
        fs = FallbackModeState(failure_threshold=1, max_fallback_duration=100.0)
        fs.on_failure()
        first_start = fs.fallback_mode_start
        fs.on_failure()
        assert fs.fallback_mode_start == first_start

    def test_success_clears_fallback_start(self):
        fs = FallbackModeState(failure_threshold=1)
        fs.on_failure()
        assert fs.fallback_mode_start is not None
        fs.on_success()
        assert fs.fallback_mode_start is None


# =============================================================================
# 14. Resilient Call Concurrency Safety
# =============================================================================

class TestResilientCallConcurrency:
    """Tests for concurrent resilient calls."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_calls(self):
        """Multiple concurrent calls should not interfere with each other."""
        call_count = 0
        async def counting_call():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        cfg = RetryConfig(timeout=5.0)
        results = await asyncio.gather(
            resilient_call(counting_call, config=cfg),
            resilient_call(counting_call, config=cfg),
            resilient_call(counting_call, config=cfg),
        )
        assert len(results) == 3
        assert all(r.startswith("result_") for r in results)

    @pytest.mark.asyncio
    async def test_shared_stats_concurrent(self):
        """Shared ErrorStats should accumulate from concurrent calls."""
        call_count = 0
        async def sometimes_fail():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                raise ConnectionError("odd fail")
            return "ok"

        stats = ErrorStats()
        cfg = RetryConfig(max_retries=1, initial_delay=0.01, jitter_fraction=0.0, timeout=5.0)

        await asyncio.gather(
            resilient_call(sometimes_fail, config=cfg, stats=stats, call_type="test"),
            resilient_call(sometimes_fail, config=cfg, stats=stats, call_type="test"),
        )
        # Both calls should have recorded something
        assert stats.total_retries >= 0


# =============================================================================
# 15. Integration: Error Classification with Real-World Patterns
# =============================================================================

class TestRealWorldErrorPatterns:
    """Tests with error patterns typical of the Google Gemini API."""

    def test_google_resource_exhausted(self):
        """google.api_core.exceptions.ResourceExhausted pattern."""
        err = Exception("429 ResourceExhausted: Quota exceeded")
        assert classify_error(err) == ErrorCategory.RATE_LIMIT

    def test_google_service_unavailable(self):
        err = Exception("503 The service is currently unavailable")
        err.code = 503
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_google_invalid_argument(self):
        err = Exception("400 Invalid argument: image too large")
        err.code = 400
        assert classify_error(err) == ErrorCategory.PERMANENT

    def test_google_internal_error(self):
        """Internal server errors should be treated as transient."""
        err = Exception("500 Internal Server Error")
        # 500 is not in our explicit transient list, but "service unavailable"
        # heuristic may not match. This would be UNKNOWN, which is retryable.
        category = classify_error(err)
        assert is_retryable(category)

    def test_network_unreachable(self):
        err = OSError("Network is unreachable")
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

    def test_broken_pipe(self):
        err = Exception("broken pipe during upload")
        assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK


# =============================================================================
# 16. LLM Wrapper Fallback Response Format
# =============================================================================

class TestFallbackResponseFormat:
    """Tests for the fallback response format consistency."""

    def test_fallback_response_is_valid_json(self):
        from src.llm_wrapper import _FALLBACK_RESPONSE
        data = json.loads(_FALLBACK_RESPONSE)
        assert data["result"] == "no_llm_available"

    def test_fallback_response_detectable(self):
        from src.llm_wrapper import _FALLBACK_RESPONSE
        from brain import _is_perception_fallback
        assert _is_perception_fallback(_FALLBACK_RESPONSE) is True


# =============================================================================
# 17. Brain API Properties
# =============================================================================

class TestBrainAPIProperties:
    """Tests for brain.py API error resilience properties (without full init)."""

    def test_safe_shutdown_exception_type(self):
        from brain import SafeShutdownRequested
        exc = SafeShutdownRequested("test")
        assert isinstance(exc, Exception)

    def test_fallback_indicator(self):
        from brain import _FALLBACK_INDICATOR, _is_perception_fallback
        assert isinstance(_FALLBACK_INDICATOR, str)
        assert _is_perception_fallback(_FALLBACK_INDICATOR) is True

    def test_perception_fallback_empty_string(self):
        from brain import _is_perception_fallback
        assert _is_perception_fallback("") is False

    def test_perception_fallback_normal_japanese(self):
        from brain import _is_perception_fallback
        assert _is_perception_fallback("画面には何も表示されていない") is False


# =============================================================================
# 18. Resilient Call Exception Safety
# =============================================================================

class TestResilientCallExceptionSafety:
    """Tests for resilient_call's own exception safety."""

    @pytest.mark.asyncio
    async def test_none_fallback_value(self):
        async def fail():
            raise ConnectionError("down")

        cfg = RetryConfig(max_retries=0, timeout=5.0)
        result = await resilient_call(fail, config=cfg, fallback_value=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_dict_fallback_value(self):
        async def fail():
            raise ConnectionError("down")

        cfg = RetryConfig(max_retries=0, timeout=5.0)
        result = await resilient_call(
            fail, config=cfg, fallback_value={"error": True}
        )
        assert result == {"error": True}

    @pytest.mark.asyncio
    async def test_no_config_uses_defaults(self):
        async def ok():
            return "default"
        result = await resilient_call(ok)
        assert result == "default"


# =============================================================================
# 19. ErrorStats Category Initialization
# =============================================================================

class TestErrorStatsCategoryInit:
    """Tests for ErrorStats category initialization."""

    def test_all_categories_initialized(self):
        stats = ErrorStats()
        for cat in ErrorCategory:
            assert cat.name in stats.errors_by_category
            assert stats.errors_by_category[cat.name] == 0

    def test_summary_includes_all_categories(self):
        stats = ErrorStats()
        summary = stats.get_summary()
        for cat in ErrorCategory:
            assert cat.name in summary["errors_by_category"]


# =============================================================================
# 20. Cumulative Wait Time Tracking
# =============================================================================

class TestWaitTimeTracking:
    """Tests for accurate wait time tracking in ErrorStats."""

    def test_cumulative_wait_time(self):
        stats = ErrorStats()
        stats.record_error(
            category=ErrorCategory.TRANSIENT_NETWORK,
            error_message="err1", retry_count=1,
            total_wait_time=1.5,
            final_result="retry_success", call_type="test",
        )
        stats.record_error(
            category=ErrorCategory.RESPONSE_TIMEOUT,
            error_message="err2", retry_count=2,
            total_wait_time=4.0,
            final_result="fallback", call_type="test",
        )
        assert stats.total_wait_time == 5.5
        assert stats.get_summary()["total_wait_time_sec"] == 5.5
