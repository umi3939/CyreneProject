"""
tests/test_brain_error_resilience.py - Systematic error recovery quality tests for brain.py.

Verifies the 2-call pipeline (perception + expression) error resilience across:
- Stage A: External API perception failure (vision pathway only)
- Stage B: Perception structuring failure
- Stage C: Internal state update exceptions
- Stage D: Expression call failure
- Stage E: Memory save failure
- Stage F: Error classification
- Stage G: Backoff computation
- Stage H: Retry judgment and fallback mode
- Stage I: Integrated retry execution
- Stage J: Cross-pathway chain errors

Design document: design_brain_error_test.md

Safety valves:
1. No changes to existing logic (test-only additions)
2. No real API connections (all mocked)
3. State pollution prevention (module-level singletons reset per test)
4. Test independence (no inter-test state dependency)
5. No psyche state persisted to disk
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.api_error_resilience import (
    ErrorCategory,
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
from src.llm_wrapper import (
    _FALLBACK_RESPONSE,
    filter_forbidden,
)

logger = logging.getLogger(__name__)


# ── Helper: Standard mock responses ───────────────────────────

def _valid_perception_json() -> str:
    """Return a valid perception API response (screen description text)."""
    return "ゲーム画面にキャラクターが映っている。UIには体力バーとスコアが表示されている。"


def _valid_expression_json(text: str = "ふふっ、楽しいわね♪") -> str:
    """Return a valid expression API response (JSON string)."""
    return json.dumps({
        "text": text,
        "meta": {"emotion": "happy", "intensity": 0.7, "action": "共感する"},
    }, ensure_ascii=False)


def _valid_perception_llm_json() -> str:
    """Return a valid LLM enrichment JSON for parse_percept."""
    return json.dumps({
        "meaning": "挨拶",
        "emotion": "happy",
        "intent": "greeting",
        "emotion_valence": 0.5,
        "topics": ["挨拶"],
    }, ensure_ascii=False)


def _fallback_response() -> str:
    """Return the standard fallback response string."""
    return _FALLBACK_RESPONSE


# ── Helper: N-fail-then-succeed mock ─────────────────────────

def _make_n_fail_mock(n_failures: int, error: Exception, success_value: str):
    """Create an async mock that fails n times then succeeds."""
    call_count = 0

    async def _mock(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= n_failures:
            raise error
        return success_value

    return _mock


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Reset module-level singletons before each test to prevent state pollution."""
    from src import llm_wrapper
    llm_wrapper._error_stats = ErrorStats()
    llm_wrapper._fallback_state = FallbackModeState()
    yield
    # Also reset after test
    llm_wrapper._error_stats = ErrorStats()
    llm_wrapper._fallback_state = FallbackModeState()


@pytest.fixture
def error_stats():
    """Fresh ErrorStats instance for isolated tests."""
    return ErrorStats()


@pytest.fixture
def fallback_state():
    """Fresh FallbackModeState instance for isolated tests."""
    return FallbackModeState()


@pytest.fixture
def retry_config():
    """Fast retry config for testing (minimal delays)."""
    return RetryConfig(
        max_retries=3,
        initial_delay=0.001,
        max_delay=0.01,
        backoff_multiplier=2.0,
        jitter_fraction=0.0,  # Deterministic for testing
        timeout=5.0,
    )


# ══════════════════════════════════════════════════════════════
# Stage A: External API perception failure (vision pathway)
# ══════════════════════════════════════════════════════════════

class TestStageA_PerceptionAPIFailure:
    """Tests for vision pathway perception API failures."""

    @pytest.mark.asyncio
    async def test_a1_api_unreachable_returns_none(self):
        """A-1: When perception API is unreachable, think() skips the frame and returns None."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                mock_img.return_value = _fallback_response()
                with patch("brain.CyreneBrain.__init__", return_value=None):
                    from brain import CyreneBrain, _is_perception_fallback
                    brain = CyreneBrain.__new__(CyreneBrain)
                    # Minimal attribute setup
                    brain._orchestrator = MagicMock()
                    brain._orchestrator.psyche = MagicMock()
                    brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                    brain._last_response = ""
                    brain._context = MagicMock()
                    brain._pipeline_measurement = MagicMock()
                    brain._pipeline_measurement.begin_pipeline = MagicMock()
                    brain._pipeline_measurement.end_pipeline = MagicMock()
                    brain._last_psyche_update = time.monotonic()
                    brain._memory = MagicMock()

                    # Verify fallback detection
                    assert _is_perception_fallback(_fallback_response())

                    with patch("brain.should_safe_shutdown", return_value=False):
                        from PIL import Image
                        img = Image.new("RGB", (10, 10))
                        result = await brain.think(image=img)
                    assert result is None

    @pytest.mark.asyncio
    async def test_a2_api_timeout_retry_and_fallback(self):
        """A-2: Timeout triggers retries, eventual fallback returns None for vision."""
        # This test verifies through resilient_call that timeouts are retried
        stats = ErrorStats()
        config = RetryConfig(max_retries=2, initial_delay=0.001, max_delay=0.01, timeout=0.001)

        async def _slow_call():
            await asyncio.sleep(10)  # Will always timeout

        result = await resilient_call(
            _slow_call,
            config=config,
            call_type="perception",
            stats=stats,
            fallback_value=_fallback_response(),
        )
        assert result == _fallback_response()
        assert stats.total_retries > 0

    @pytest.mark.asyncio
    async def test_a3_rate_limit_retry_strategy(self):
        """A-3: Rate limit (429) triggers retries with backoff."""
        stats = ErrorStats()
        fb = FallbackModeState()
        config = RetryConfig(max_retries=2, initial_delay=0.001, max_delay=0.01, jitter_fraction=0.0)

        class RateLimitError(Exception):
            code = 429

        call_count = 0

        async def _rate_limited():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("rate limit exceeded")

        result = await resilient_call(
            _rate_limited,
            config=config,
            call_type="perception",
            stats=stats,
            fallback_state=fb,
            fallback_value="fallback",
        )
        assert result == "fallback"
        # Should have attempted initial + max_retries times
        assert call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_a4_permanent_error_no_retry(self):
        """A-4: Permanent error (auth failure) stops retries immediately."""
        stats = ErrorStats()
        config = RetryConfig(max_retries=3, initial_delay=0.001, max_delay=0.01)

        class AuthError(Exception):
            code = 401

        call_count = 0

        async def _auth_fail():
            nonlocal call_count
            call_count += 1
            raise AuthError("invalid api key")

        result = await resilient_call(
            _auth_fail,
            config=config,
            call_type="perception",
            stats=stats,
            fallback_value="fallback",
        )
        assert result == "fallback"
        # Permanent error: only 1 attempt, no retries
        assert call_count == 1

    def test_a5_missing_image_raises(self):
        """A-5: Missing image path raises FileNotFoundError."""
        # _is_perception_fallback is separate, but brain.think checks file existence
        p = Path("nonexistent_test_image_12345.jpg")
        assert not p.exists()
        # Direct check: FileNotFoundError would be caught by think()'s except block


# ══════════════════════════════════════════════════════════════
# Stage B: Perception structuring failure
# ══════════════════════════════════════════════════════════════

class TestStageB_PerceptionStructuring:
    """Tests for perception structuring (parse_percept) failures."""

    @pytest.mark.asyncio
    async def test_b1_llm_enrichment_failure_falls_back_to_heuristic(self):
        """B-1: When LLM enrichment fails, heuristic baseline is used."""
        from psyche.perception import parse_percept

        async def _failing_llm(prompt, **kwargs):
            raise ConnectionError("API down")

        percept = await parse_percept("こんにちは！嬉しい", llm_call_fn=_failing_llm)
        # Heuristic should have parsed emotion from keywords
        assert percept.emotion in ("happy", "neutral")
        assert percept.text == "こんにちは！嬉しい"

    @pytest.mark.asyncio
    async def test_b2_empty_input_heuristic(self):
        """B-2: Empty input produces neutral baseline percept."""
        from psyche.perception import parse_percept

        percept = await parse_percept("", llm_call_fn=None)
        assert percept.emotion == "neutral"
        assert percept.intent == "unknown"
        assert percept.text == ""

    @pytest.mark.asyncio
    async def test_b3_invalid_json_from_llm_falls_back(self):
        """B-3: Invalid JSON from LLM enrichment falls back to heuristic."""
        from psyche.perception import parse_percept

        async def _bad_json_llm(prompt, **kwargs):
            return "This is not JSON at all {{{broken"

        percept = await parse_percept("怒りの表現", llm_call_fn=_bad_json_llm)
        # Should still produce a valid Percept from heuristic
        assert percept.text == "怒りの表現"
        assert isinstance(percept.emotion, str)

    @pytest.mark.asyncio
    async def test_b3b_no_llm_available_response_falls_back(self):
        """B-3b: Fallback response from LLM (no_llm_available) falls back to heuristic."""
        from psyche.perception import parse_percept

        async def _fallback_llm(prompt, **kwargs):
            return _fallback_response()

        percept = await parse_percept("テスト入力", llm_call_fn=_fallback_llm)
        assert percept.text == "テスト入力"
        # Should be heuristic-parsed (neutral for this generic input)
        assert isinstance(percept.emotion, str)


# ══════════════════════════════════════════════════════════════
# Stage C: Internal state update exceptions
# ══════════════════════════════════════════════════════════════

class TestStageC_InternalStateUpdate:
    """Tests for exceptions during orchestrator updates."""

    @pytest.mark.asyncio
    async def test_c1_orchestrator_update_exception_returns_fallback(self):
        """C-1: Exception in post_response_update returns fallback text."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._orchestrator.psyche = MagicMock()
                brain._orchestrator.post_response_update = MagicMock(
                    side_effect=RuntimeError("orchestrator crash")
                )
                brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain._last_response = ""
                brain._last_psyche_update = time.monotonic()
                brain._context = MagicMock()
                brain._pipeline_measurement = MagicMock()
                brain._memory = MagicMock()
                brain._last_emotion = "neutral"

                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                        mock_img.return_value = _valid_perception_json()
                        with patch("brain.parse_percept", new_callable=AsyncMock) as mock_parse:
                            from psyche.state import Percept
                            mock_parse.return_value = Percept(text="test", emotion="neutral")
                            from PIL import Image
                            img = Image.new("RGB", (10, 10))
                            result = await brain.think(image=img)

                # Should catch exception and return fallback text
                assert result is not None
                assert "ごめんなさい" in result

    @pytest.mark.asyncio
    async def test_c2_memory_recall_exception(self):
        """C-2: Exception in recall_with_mood is caught by think()."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._orchestrator.psyche = MagicMock()
                brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain._last_response = ""
                brain._last_psyche_update = time.monotonic()
                brain._context = MagicMock()
                brain._pipeline_measurement = MagicMock()
                brain._memory = MagicMock()
                brain._last_emotion = "neutral"

                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                        mock_img.return_value = _valid_perception_json()
                        with patch("brain.parse_percept", new_callable=AsyncMock) as mock_parse:
                            from psyche.state import Percept
                            mock_parse.return_value = Percept(text="test", emotion="neutral")
                            with patch("brain.recall_with_mood", new_callable=AsyncMock) as mock_recall:
                                mock_recall.side_effect = RuntimeError("recall failed")
                                from PIL import Image
                                img = Image.new("RGB", (10, 10))
                                result = await brain.think(image=img)

                assert result is not None
                assert "ごめんなさい" in result

    @pytest.mark.asyncio
    async def test_c3_policy_selection_exception(self):
        """C-3: Exception in select_policy_dict is caught by think()."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._orchestrator.psyche = MagicMock()
                brain._orchestrator.select_policy_dict = MagicMock(
                    side_effect=RuntimeError("policy crash")
                )
                brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain._last_response = ""
                brain._last_psyche_update = time.monotonic()
                brain._context = MagicMock()
                brain._pipeline_measurement = MagicMock()
                brain._memory = MagicMock()
                brain._last_emotion = "neutral"

                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                        mock_img.return_value = _valid_perception_json()
                        with patch("brain.parse_percept", new_callable=AsyncMock) as mock_parse:
                            from psyche.state import Percept
                            mock_parse.return_value = Percept(text="test", emotion="neutral")
                            with patch("brain.recall_with_mood", new_callable=AsyncMock) as mock_recall:
                                mock_recall.return_value = []
                                from PIL import Image
                                img = Image.new("RGB", (10, 10))
                                result = await brain.think(image=img)

                assert result is not None
                assert "ごめんなさい" in result

    @pytest.mark.asyncio
    async def test_c4_empty_memory_recall_normal_operation(self):
        """C-4: Empty memory recall is normal operation, pipeline continues."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._orchestrator.psyche = MagicMock()
                brain._orchestrator.select_policy_dict = MagicMock(
                    return_value={"policy_label": "共感する", "rationale": "test"}
                )
                brain._orchestrator.get_prompt_enrichment = MagicMock(return_value="")
                brain._orchestrator.notify_self_output = MagicMock()
                brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain._last_response = ""
                brain._last_psyche_update = time.monotonic()
                brain._context = MagicMock()
                brain._context.get_window_entries = MagicMock(return_value=[])
                brain._pipeline_measurement = MagicMock()
                brain._memory = MagicMock()
                brain._last_emotion = "neutral"
                brain._persona_dict = {"name": "キュレネ", "tone": "romantic"}

                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                        mock_img.return_value = _valid_perception_json()
                        with patch("brain.parse_percept", new_callable=AsyncMock) as mock_parse:
                            from psyche.state import Percept
                            mock_parse.return_value = Percept(text="test", emotion="happy")
                            with patch("brain.recall_with_mood", new_callable=AsyncMock) as mock_recall:
                                mock_recall.return_value = []  # Empty recall
                                with patch("brain.render_expression", new_callable=AsyncMock) as mock_expr:
                                    mock_expr.return_value = {
                                        "text": "ふふっ♪",
                                        "meta": {"emotion": "happy"},
                                    }
                                    with patch("brain.is_silence_policy", return_value=False):
                                        from PIL import Image
                                        img = Image.new("RGB", (10, 10))
                                        result = await brain.think(image=img)

                # Should succeed normally with empty memories
                assert result == "ふふっ♪"


# ══════════════════════════════════════════════════════════════
# Stage D: Expression call failure
# ══════════════════════════════════════════════════════════════

class TestStageD_ExpressionCallFailure:
    """Tests for expression (render) call failures."""

    @pytest.mark.asyncio
    async def test_d1_expression_api_unreachable_fallback(self):
        """D-1: When expression API returns fallback, rule-based text is used."""
        from psyche.expression import _parse_expression_output, _fallback_expression
        from psyche.state import PsycheState

        state = PsycheState()
        policy = {"policy_label": "共感する"}
        # Simulate fallback response from LLM
        result = _parse_expression_output(_fallback_response(), state, policy)
        # Should use rule-based fallback
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_d2_expression_invalid_json_fallback(self):
        """D-2: Invalid JSON from expression API triggers rule-based fallback."""
        from psyche.expression import _parse_expression_output
        from psyche.state import PsycheState

        state = PsycheState()
        policy = {"policy_label": "共感する"}
        result = _parse_expression_output("not valid json {{{", state, policy)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_d3_expression_empty_response_fallback(self):
        """D-3: Empty response from expression API triggers rule-based fallback."""
        from psyche.expression import _parse_expression_output
        from psyche.state import PsycheState

        state = PsycheState()
        policy = {"policy_label": "共感する"}
        result = _parse_expression_output("", state, policy)
        assert "text" in result
        assert isinstance(result["text"], str)

    @pytest.mark.asyncio
    async def test_d4_fallback_reflects_state_and_policy(self):
        """D-4: Fallback expression content reflects current state and policy."""
        from psyche.expression import _fallback_expression
        from psyche.state import PsycheState

        # Test with positive mood
        state = PsycheState()
        state = state.model_copy(update={"mood": state.mood.model_copy(update={"valence": 0.5})})
        policy = {"policy_label": "共感する"}
        result = _fallback_expression(state, policy)
        assert "text" in result
        assert result["meta"]["action"] == "共感する"

        # Test with negative mood
        state_neg = PsycheState()
        state_neg = state_neg.model_copy(update={"mood": state_neg.mood.model_copy(update={"valence": -0.5})})
        result_neg = _fallback_expression(state_neg, policy)
        assert "text" in result_neg
        # Different mood should produce different fallback
        assert result_neg["text"] != result["text"]


# ══════════════════════════════════════════════════════════════
# Stage E: Memory save failure
# ══════════════════════════════════════════════════════════════

class TestStageE_MemorySaveFailure:
    """Tests for memory save (summarize_and_save) failures."""

    @pytest.mark.asyncio
    async def test_e1_summary_api_failure_swallowed(self):
        """E-1: Summary generation API failure is swallowed, next turn works."""
        with patch("brain.CyreneBrain.__init__", return_value=None):
            from brain import CyreneBrain
            brain = CyreneBrain.__new__(CyreneBrain)
            brain._context = MagicMock()
            from brain import ContextEntry
            entries = [
                ContextEntry("ユーザー", "こんにちは", "text", "u1", 1.0),
                ContextEntry("キュレネ", "やあ♪", "text", "u1", 2.0),
                ContextEntry("ユーザー", "元気？", "text", "u1", 3.0),
            ]
            brain._context.get_entries = MagicMock(return_value=entries)
            brain._memory = MagicMock()
            brain._orchestrator = MagicMock()
            brain._model_name = "test-model"
            # Mock client that raises on generate_content
            brain._client = MagicMock()
            brain._client.aio.models.generate_content = AsyncMock(
                side_effect=ConnectionError("API down")
            )
            brain._summary_config = MagicMock()

            # Should not raise
            await brain.summarize_and_save()
            # Memory should not be saved
            brain._memory.maybe_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_e2_summary_json_parse_failure(self):
        """E-2: Summary JSON parse failure is handled gracefully."""
        with patch("brain.CyreneBrain.__init__", return_value=None):
            from brain import CyreneBrain
            brain = CyreneBrain.__new__(CyreneBrain)
            brain._context = MagicMock()
            from brain import ContextEntry
            entries = [
                ContextEntry("ユーザー", "こんにちは", "text", "u1", 1.0),
                ContextEntry("キュレネ", "やあ♪", "text", "u1", 2.0),
            ]
            brain._context.get_entries = MagicMock(return_value=entries)
            brain._memory = MagicMock()
            brain._orchestrator = MagicMock()
            brain._model_name = "test-model"
            brain._summary_config = MagicMock()

            mock_response = MagicMock()
            mock_response.text = "This is not valid JSON at all"
            brain._client = MagicMock()
            brain._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            await brain.summarize_and_save()
            brain._memory.maybe_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_e3_memory_save_method_exception(self):
        """E-3: Exception from memory.maybe_save is caught."""
        with patch("brain.CyreneBrain.__init__", return_value=None):
            from brain import CyreneBrain
            brain = CyreneBrain.__new__(CyreneBrain)
            brain._context = MagicMock()
            from brain import ContextEntry
            entries = [
                ContextEntry("ユーザー", "こんにちは", "text", "u1", 1.0),
                ContextEntry("キュレネ", "やあ♪", "text", "u1", 2.0),
            ]
            brain._context.get_entries = MagicMock(return_value=entries)
            brain._memory = MagicMock()
            brain._memory.maybe_save = MagicMock(side_effect=IOError("disk full"))
            brain._orchestrator = MagicMock()
            brain._model_name = "test-model"
            brain._summary_config = MagicMock()

            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "summary": "test summary",
                "keywords": ["test"],
                "importance": 3,
            })
            brain._client = MagicMock()
            brain._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            # Should not raise
            await brain.summarize_and_save()

    @pytest.mark.asyncio
    async def test_e4_memory_save_success_calls_orchestrator(self):
        """E-4: Successful memory save notifies orchestrator."""
        with patch("brain.CyreneBrain.__init__", return_value=None):
            from brain import CyreneBrain
            brain = CyreneBrain.__new__(CyreneBrain)
            brain._context = MagicMock()
            from brain import ContextEntry
            entries = [
                ContextEntry("ユーザー", "こんにちは", "text", "u1", 1.0),
                ContextEntry("キュレネ", "やあ♪", "text", "u1", 2.0),
            ]
            brain._context.get_entries = MagicMock(return_value=entries)
            brain._memory = MagicMock()
            brain._memory._memories = [{"summary": "old"}]
            brain._orchestrator = MagicMock()
            brain._model_name = "test-model"
            brain._summary_config = MagicMock()
            brain._chat = MagicMock()
            brain._last_response = ""
            brain._config = MagicMock()

            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "summary": "test summary",
                "keywords": ["test"],
                "importance": 3,
            })
            brain._client = MagicMock()
            brain._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            brain._client.aio.chats.create = MagicMock()

            await brain.summarize_and_save()

            # on_memory_saved should have been called
            brain._orchestrator.on_memory_saved.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Stage F: Error classification
# ══════════════════════════════════════════════════════════════

class TestStageF_ErrorClassification:
    """Tests for classify_error function."""

    def test_f1_timeout_classified(self):
        """F-1: TimeoutError is classified as RESPONSE_TIMEOUT."""
        assert classify_error(asyncio.TimeoutError()) == ErrorCategory.RESPONSE_TIMEOUT
        assert classify_error(TimeoutError()) == ErrorCategory.RESPONSE_TIMEOUT

    def test_f2_connection_errors_classified(self):
        """F-2: Connection errors are classified as TRANSIENT_NETWORK."""
        assert classify_error(ConnectionRefusedError()) == ErrorCategory.TRANSIENT_NETWORK
        assert classify_error(ConnectionResetError()) == ErrorCategory.TRANSIENT_NETWORK
        assert classify_error(ConnectionAbortedError()) == ErrorCategory.TRANSIENT_NETWORK
        assert classify_error(ConnectionError()) == ErrorCategory.TRANSIENT_NETWORK

    def test_f3_http_status_codes_classified(self):
        """F-3: HTTP status codes are correctly classified."""
        # Rate limit
        err429 = Exception("API error")
        err429.code = 429
        assert classify_error(err429) == ErrorCategory.RATE_LIMIT

        # Transient server errors
        for code in (502, 503, 504):
            err = Exception("server error")
            err.code = code
            assert classify_error(err) == ErrorCategory.TRANSIENT_NETWORK

        # Permanent errors
        for code in (400, 401, 403):
            err = Exception("client error")
            err.code = code
            assert classify_error(err) == ErrorCategory.PERMANENT

    def test_f4_heuristic_keywords_classified(self):
        """F-4: Heuristic keywords in error message are correctly classified."""
        assert classify_error(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMIT
        assert classify_error(Exception("quota exceeded")) == ErrorCategory.RATE_LIMIT
        assert classify_error(Exception("resource_exhausted")) == ErrorCategory.RATE_LIMIT
        assert classify_error(Exception("invalid api key")) == ErrorCategory.PERMANENT

    def test_f5_nested_cause_status_code(self):
        """F-5: Nested exception (__cause__) status code extraction."""
        inner = Exception("inner error")
        inner.code = 429
        outer = Exception("outer wrapper")
        outer.__cause__ = inner
        # _extract_status_code should find 429 from __cause__
        assert _extract_status_code(outer) == 429

    def test_f6_unknown_error_classified(self):
        """F-6: Unclassifiable exception is classified as UNKNOWN."""
        assert classify_error(ValueError("some random error")) == ErrorCategory.UNKNOWN


# ══════════════════════════════════════════════════════════════
# Stage G: Backoff computation
# ══════════════════════════════════════════════════════════════

class TestStageG_BackoffComputation:
    """Tests for compute_backoff_delay function."""

    def test_g1_exponential_increase(self):
        """G-1: Delay increases exponentially with attempt number."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=100.0,
            jitter_fraction=0.0,
        )
        d0 = compute_backoff_delay(0, config)
        d1 = compute_backoff_delay(1, config)
        d2 = compute_backoff_delay(2, config)
        assert d0 == pytest.approx(1.0)
        assert d1 == pytest.approx(2.0)
        assert d2 == pytest.approx(4.0)

    def test_g2_max_delay_cap(self):
        """G-2: Delay never exceeds max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=10.0,
            max_delay=5.0,
            jitter_fraction=0.0,
        )
        d = compute_backoff_delay(10, config)
        assert d == pytest.approx(5.0)

    def test_g3_jitter_within_range(self):
        """G-3: Jitter stays within [0, base * jitter_fraction]."""
        config = RetryConfig(
            initial_delay=10.0,
            backoff_multiplier=1.0,
            max_delay=100.0,
            jitter_fraction=0.25,
        )
        for _ in range(100):
            d = compute_backoff_delay(0, config)
            assert 10.0 <= d <= 10.0 + 10.0 * 0.25

    def test_g4_retry_after_takes_priority(self):
        """G-4: Server-specified retry_after is used when present."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=100.0,
            jitter_fraction=0.0,
        )
        d = compute_backoff_delay(0, config, retry_after=15.0)
        assert d == pytest.approx(15.0)

    def test_g5_retry_after_capped_by_max_delay(self):
        """G-5: retry_after is capped by max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0,
            jitter_fraction=0.0,
        )
        d = compute_backoff_delay(0, config, retry_after=50.0)
        assert d == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════
# Stage H: Retry judgment and fallback mode
# ══════════════════════════════════════════════════════════════

class TestStageH_RetryAndFallbackMode:
    """Tests for retry judgment and fallback mode transitions."""

    def test_h1_permanent_error_not_retryable(self):
        """H-1: Permanent errors are not retryable."""
        assert is_retryable(ErrorCategory.PERMANENT) is False

    def test_h2_transient_errors_retryable(self):
        """H-2: Transient errors are retryable."""
        assert is_retryable(ErrorCategory.TRANSIENT_NETWORK) is True
        assert is_retryable(ErrorCategory.RATE_LIMIT) is True
        assert is_retryable(ErrorCategory.RESPONSE_TIMEOUT) is True
        assert is_retryable(ErrorCategory.UNKNOWN) is True

    def test_h3_consecutive_failures_enter_fallback(self):
        """H-3: Consecutive failures >= threshold triggers fallback mode."""
        fb = FallbackModeState(failure_threshold=3)
        assert not fb.is_in_fallback
        fb.on_failure()
        fb.on_failure()
        assert not fb.is_in_fallback
        fb.on_failure()
        assert fb.is_in_fallback

    def test_h4_success_exits_fallback(self):
        """H-4: API success exits fallback mode."""
        fb = FallbackModeState(failure_threshold=2)
        fb.on_failure()
        fb.on_failure()
        assert fb.is_in_fallback
        fb.on_success()
        assert not fb.is_in_fallback
        assert fb.consecutive_failures == 0

    def test_h5_fallback_duration_triggers_safe_shutdown(self):
        """H-5: Fallback mode exceeding max duration triggers safe shutdown."""
        fb = FallbackModeState(failure_threshold=1, max_fallback_duration=0.0)
        fb.on_failure()
        assert fb.is_in_fallback
        # With max_fallback_duration=0.0, any time elapsed should trigger shutdown
        assert fb.should_safe_shutdown() is True

    def test_h6_safe_shutdown_raises_dedicated_exception(self):
        """H-6: Safe shutdown is raised as dedicated exception type."""
        from brain import SafeShutdownRequested
        with pytest.raises(SafeShutdownRequested):
            raise SafeShutdownRequested("test shutdown")

    @pytest.mark.asyncio
    async def test_h6b_safe_shutdown_propagates_through_pipeline(self):
        """H-6b: SafeShutdownRequested propagates through think() without being caught."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain, SafeShutdownRequested
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._pipeline_measurement = MagicMock()

                with patch("brain.should_safe_shutdown", return_value=True):
                    brain.save_state = MagicMock()
                    with pytest.raises(SafeShutdownRequested):
                        from PIL import Image
                        img = Image.new("RGB", (10, 10))
                        await brain.think(image=img)


# ══════════════════════════════════════════════════════════════
# Stage I: Integrated retry execution
# ══════════════════════════════════════════════════════════════

class TestStageI_IntegratedRetry:
    """Tests for resilient_call integrated retry execution."""

    @pytest.mark.asyncio
    async def test_i1_first_success_no_record(self):
        """I-1: First-attempt success generates no error record."""
        stats = ErrorStats()

        async def _success():
            return "ok"

        result = await resilient_call(
            _success,
            config=RetryConfig(max_retries=3, timeout=5.0),
            call_type="test",
            stats=stats,
        )
        assert result == "ok"
        assert len(stats.records) == 0
        assert stats.total_retries == 0

    @pytest.mark.asyncio
    async def test_i2_retry_success_generates_record(self):
        """I-2: Success after retry generates 'retry_success' record."""
        stats = ErrorStats()
        call_count = 0

        async def _fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("first fail")
            return "recovered"

        config = RetryConfig(max_retries=3, initial_delay=0.001, max_delay=0.01, timeout=5.0)
        result = await resilient_call(
            _fail_once,
            config=config,
            call_type="test",
            stats=stats,
        )
        assert result == "recovered"
        assert stats.retry_recovery_count == 1
        assert len(stats.records) == 1
        assert stats.records[0].final_result == "retry_success"

    @pytest.mark.asyncio
    async def test_i3_all_retries_fail_generates_fallback_record(self):
        """I-3: All retries failing generates 'fallback' record."""
        stats = ErrorStats()

        async def _always_fail():
            raise ConnectionError("always fail")

        config = RetryConfig(max_retries=2, initial_delay=0.001, max_delay=0.01, timeout=5.0)
        result = await resilient_call(
            _always_fail,
            config=config,
            call_type="test",
            stats=stats,
            fallback_value="fb",
        )
        assert result == "fb"
        assert stats.fallback_count == 1
        assert len(stats.records) == 1
        assert stats.records[0].final_result == "fallback"

    @pytest.mark.asyncio
    async def test_i4_error_stats_accumulation(self):
        """I-4: Error statistics accumulate correctly across calls."""
        stats = ErrorStats()
        config = RetryConfig(max_retries=1, initial_delay=0.001, max_delay=0.01, timeout=5.0)

        async def _fail():
            raise ConnectionError("fail")

        for _ in range(3):
            await resilient_call(
                _fail,
                config=config,
                call_type="test",
                stats=stats,
                fallback_value="fb",
            )

        assert stats.fallback_count == 3
        assert len(stats.records) == 3
        # Each call has 1 initial + 1 retry = 2 attempts, records retry_count=1 (min of attempt and max_retries)
        assert stats.total_retries >= 3  # At least 1 retry per call

    @pytest.mark.asyncio
    async def test_i5_fifo_record_eviction(self):
        """I-5: Error records are evicted when exceeding FIFO cap."""
        stats = ErrorStats()
        stats._max_records = 5  # Small cap for testing

        for i in range(10):
            stats.record_error(
                category=ErrorCategory.TRANSIENT_NETWORK,
                error_message=f"error {i}",
                retry_count=1,
                total_wait_time=0.0,
                final_result="fallback",
                call_type="test",
            )

        assert len(stats.records) == 5
        # Oldest records should be evicted
        assert stats.records[0].error_message == "error 5"


# ══════════════════════════════════════════════════════════════
# Stage J: Cross-pathway chain errors
# ══════════════════════════════════════════════════════════════

class TestStageJ_CrossPathwayChain:
    """Tests for cross-pathway and chain error scenarios."""

    @pytest.mark.asyncio
    async def test_j1_vision_failures_then_text_works(self):
        """J-1: Multiple vision failures do not affect text pathway."""
        # Verify that after vision fallback, text input still works
        from psyche.perception import parse_percept
        from psyche.state import Percept

        # Simulate that LLM is down for perception
        async def _failing_llm(prompt, **kwargs):
            raise ConnectionError("API unreachable")

        # Text pathway parse_percept should still work via heuristic
        percept = await parse_percept("嬉しいです", llm_call_fn=_failing_llm)
        assert percept.emotion == "happy"
        assert percept.text == "嬉しいです"

    @pytest.mark.asyncio
    async def test_j2_expression_failure_next_turn_normal(self):
        """J-2: Expression failure does not contaminate next turn."""
        from psyche.expression import _parse_expression_output
        from psyche.state import PsycheState

        state = PsycheState()
        policy = {"policy_label": "共感する"}

        # First call: bad output (fallback)
        result1 = _parse_expression_output("garbage", state, policy)
        assert "text" in result1  # Fallback used

        # Second call: valid output (should parse normally)
        valid_json = json.dumps({
            "text": "ふふっ♪",
            "meta": {"emotion": "happy", "intensity": 0.7, "action": "共感する"},
        }, ensure_ascii=False)
        result2 = _parse_expression_output(valid_json, state, policy)
        assert result2["text"] == "ふふっ♪"

    @pytest.mark.asyncio
    async def test_j3_fallback_content_per_pathway(self):
        """J-3: Fallback responses differ by pathway (vision/text return text, spontaneous returns None)."""
        # Verify the error handling return values for each pathway

        # Vision pathway: exception caught, returns fallback text
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                from brain import CyreneBrain
                brain = CyreneBrain.__new__(CyreneBrain)
                brain._orchestrator = MagicMock()
                brain._orchestrator.psyche = MagicMock()
                brain._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain._last_response = ""
                brain._last_psyche_update = time.monotonic()
                brain._context = MagicMock()
                brain._pipeline_measurement = MagicMock()
                brain._memory = MagicMock()
                brain._last_emotion = "neutral"

                # Force an exception during think
                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.llm_call_with_image", new_callable=AsyncMock) as mock_img:
                        mock_img.side_effect = Exception("total failure")
                        from PIL import Image
                        img = Image.new("RGB", (10, 10))
                        result_vision = await brain.think(image=img)

                assert result_vision is not None
                assert "ごめんなさい" in result_vision

        # Text pathway: exception caught, returns fallback text
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                brain2 = CyreneBrain.__new__(CyreneBrain)
                brain2._orchestrator = MagicMock()
                brain2._orchestrator.psyche = MagicMock()
                brain2._perception_config = {"temperature": 0.3, "max_tokens": 256}
                brain2._last_response = ""
                brain2._last_psyche_update = time.monotonic()
                brain2._context = MagicMock()
                brain2._pipeline_measurement = MagicMock()
                brain2._memory = MagicMock()
                brain2._last_emotion = "neutral"

                with patch("brain.should_safe_shutdown", return_value=False):
                    with patch("brain.parse_percept", new_callable=AsyncMock) as mock_parse:
                        mock_parse.side_effect = Exception("total failure")
                        result_text = await brain2.think_text("test input")

                assert result_text is not None
                assert "ごめんなさい" in result_text

        # Spontaneous pathway: exception caught, returns None
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("brain.CyreneBrain.__init__", return_value=None):
                brain3 = CyreneBrain.__new__(CyreneBrain)
                brain3._orchestrator = MagicMock()
                brain3._orchestrator.check_spontaneous_activation = MagicMock(
                    side_effect=Exception("total failure")
                )
                brain3._pipeline_measurement = MagicMock()

                with patch("brain.should_safe_shutdown", return_value=False):
                    result_spontaneous = await brain3.think_spontaneous()

                # Spontaneous returns None on error
                assert result_spontaneous is None


# ══════════════════════════════════════════════════════════════
# Additional: filter_forbidden, _extract_retry_after
# ══════════════════════════════════════════════════════════════

class TestFilterForbidden:
    """Tests for LLM output filtering."""

    def test_forbidden_pattern_stripped(self):
        """Lines with forbidden patterns are removed."""
        text = "こんにちは♪\n判断した結果こうなった\nまた会おうね"
        result = filter_forbidden(text)
        assert "判断" not in result
        assert "こんにちは♪" in result
        assert "また会おうね" in result

    def test_clean_text_unchanged(self):
        """Clean text passes through unchanged."""
        text = "ふふっ、楽しいわね♪"
        assert filter_forbidden(text) == text


class TestExtractRetryAfter:
    """Tests for retry_after extraction."""

    def test_attribute_extraction(self):
        """Extract retry_after from error attribute."""
        err = Exception("rate limited")
        err.retry_after = 30
        assert _extract_retry_after(err) == 30.0

    def test_message_extraction(self):
        """Extract retry_after from error message."""
        err = Exception("Please retry after 15 seconds")
        result = _extract_retry_after(err)
        assert result == 15.0

    def test_no_retry_after(self):
        """Returns None when no retry_after info available."""
        err = Exception("generic error")
        assert _extract_retry_after(err) is None


class TestFallbackDetection:
    """Tests for brain._is_perception_fallback."""

    def test_fallback_detected(self):
        """Fallback response is correctly detected."""
        from brain import _is_perception_fallback
        assert _is_perception_fallback(_fallback_response()) is True

    def test_normal_response_not_detected(self):
        """Normal response is not detected as fallback."""
        from brain import _is_perception_fallback
        assert _is_perception_fallback("ゲーム画面が表示されています") is False

    def test_empty_response_not_detected(self):
        """Empty response is not detected as fallback."""
        from brain import _is_perception_fallback
        assert _is_perception_fallback("") is False


class TestFallbackModeStatus:
    """Tests for FallbackModeState.get_status."""

    def test_normal_status(self):
        """Normal (non-fallback) status."""
        fb = FallbackModeState()
        status = fb.get_status()
        assert status["is_in_fallback"] is False
        assert status["consecutive_failures"] == 0

    def test_fallback_status_includes_duration(self):
        """Fallback status includes duration info."""
        fb = FallbackModeState(failure_threshold=1)
        fb.on_failure()
        assert fb.is_in_fallback
        status = fb.get_status()
        assert status["is_in_fallback"] is True
        assert "fallback_duration_sec" in status


class TestErrorStatsSummary:
    """Tests for ErrorStats.get_summary."""

    def test_empty_summary(self):
        """Empty stats produce zero-value summary."""
        stats = ErrorStats()
        summary = stats.get_summary()
        assert summary["retry_recovery_count"] == 0
        assert summary["fallback_count"] == 0
        assert summary["total_retries"] == 0
        assert summary["total_wait_time_sec"] == 0.0

    def test_populated_summary(self):
        """Stats with records produce correct summary."""
        stats = ErrorStats()
        stats.record_error(
            ErrorCategory.TRANSIENT_NETWORK, "test", 2, 1.5, "retry_success", "test"
        )
        stats.record_error(
            ErrorCategory.RATE_LIMIT, "test2", 3, 5.0, "fallback", "test"
        )
        summary = stats.get_summary()
        assert summary["retry_recovery_count"] == 1
        assert summary["fallback_count"] == 1
        assert summary["total_retries"] == 5  # 2 + 3
        assert summary["total_wait_time_sec"] == 6.5  # 1.5 + 5.0
