"""
src/api_error_resilience.py - API error resilience layer.

Provides exponential backoff retry, error classification, rate limit handling,
fallback mode management, and error observation recording.

This module exists entirely within the external API call layer. It does not
access, modify, or reference any psyche internal state. All retry logic is
encapsulated and invisible to callers.

Architecture rule: This module NEVER affects psyche judgment, state, or actions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Error Classification ──────────────────────────────────────

class ErrorCategory(Enum):
    """Classification of API call errors."""
    TRANSIENT_NETWORK = auto()     # Connection timeout, refused, DNS failure
    RATE_LIMIT = auto()            # API rate limit (429)
    RESPONSE_TIMEOUT = auto()      # Call did not complete within timeout
    PERMANENT = auto()             # Auth failure, invalid request, version mismatch
    UNKNOWN = auto()               # Unclassifiable exception


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an exception into an error category.

    Classification is based on exception type and HTTP status code
    (when available). The classification result determines retry strategy.

    Args:
        error: The exception to classify.

    Returns:
        The error category.
    """
    # Timeout errors
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return ErrorCategory.RESPONSE_TIMEOUT

    # Connection errors (transient network issues)
    if isinstance(error, (
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        ConnectionAbortedError,
        OSError,
    )):
        return ErrorCategory.TRANSIENT_NETWORK

    # Check for HTTP status codes in google-genai exceptions
    error_str = str(error).lower()
    status_code = _extract_status_code(error)

    if status_code is not None:
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if status_code in (408, 502, 503, 504):
            return ErrorCategory.TRANSIENT_NETWORK
        if status_code in (400, 401, 403, 404, 405):
            return ErrorCategory.PERMANENT

    # Heuristic: check error message for rate limit indicators
    if "rate" in error_str and "limit" in error_str:
        return ErrorCategory.RATE_LIMIT
    if "quota" in error_str or "resource_exhausted" in error_str or "resourceexhausted" in error_str:
        return ErrorCategory.RATE_LIMIT
    if "429" in error_str:
        return ErrorCategory.RATE_LIMIT

    # Heuristic: check for permanent errors
    if any(kw in error_str for kw in (
        "invalid api key", "api_key_invalid", "authentication",
        "permission", "forbidden", "not found", "invalid argument",
        "invalid_argument",
    )):
        return ErrorCategory.PERMANENT

    # Heuristic: transient network indicators
    if any(kw in error_str for kw in (
        "connection", "timeout", "dns", "unreachable",
        "reset by peer", "broken pipe", "ssl",
        "temporarily unavailable", "service unavailable",
    )):
        return ErrorCategory.TRANSIENT_NETWORK

    return ErrorCategory.UNKNOWN


def _extract_status_code(error: Exception) -> Optional[int]:
    """Try to extract HTTP status code from an exception.

    Handles google-genai ClientError/ServerError which store status in
    various attributes.
    """
    # google.genai errors often have a .code or .status attribute
    for attr in ("code", "status", "status_code", "grpc_status_code"):
        val = getattr(error, attr, None)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # Check nested __cause__
    if error.__cause__ is not None:
        return _extract_status_code(error.__cause__)

    return None


def _extract_retry_after(error: Exception) -> Optional[float]:
    """Try to extract Retry-After value from a rate limit error.

    Some API errors include a recommended wait time in headers or message.
    """
    # Check for retry_after attribute
    val = getattr(error, "retry_after", None)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass

    # Parse from error message: "retry after X seconds" or "Retry-After: X"
    import re
    msg = str(error)
    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def is_retryable(category: ErrorCategory) -> bool:
    """Determine if an error category should be retried.

    Permanent errors are never retried. All other categories are retried
    up to the configured maximum attempts.
    """
    return category != ErrorCategory.PERMANENT


# ── Retry Configuration ───────────────────────────────────────

@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry.

    All values are configurable. Defaults provide reasonable behavior
    for real-time API calls.

    Attributes:
        max_retries: Maximum number of retry attempts (excluding the initial attempt).
        initial_delay: Base delay in seconds for the first retry.
        max_delay: Maximum delay cap in seconds.
        backoff_multiplier: Factor by which delay increases each retry.
        jitter_fraction: Maximum fraction of computed delay to add as random jitter.
        timeout: Per-call timeout in seconds.
    """
    max_retries: int = 4
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25
    timeout: float = 30.0


def compute_backoff_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[float] = None,
) -> float:
    """Compute the delay before the next retry attempt.

    Uses exponential backoff with jitter. If retry_after is provided
    (from a rate limit response), it takes priority but is still capped
    by max_delay.

    Args:
        attempt: Zero-based retry attempt number (0 = first retry).
        config: Retry configuration.
        retry_after: Optional server-suggested wait time in seconds.

    Returns:
        Delay in seconds before the next retry.
    """
    if retry_after is not None and retry_after > 0:
        base = min(retry_after, config.max_delay)
    else:
        base = config.initial_delay * (config.backoff_multiplier ** attempt)
        base = min(base, config.max_delay)

    # Add jitter: random value in [0, base * jitter_fraction]
    jitter = random.uniform(0, base * config.jitter_fraction)
    return base + jitter


# ── Error Observation Record ──────────────────────────────────

@dataclass
class ErrorRecord:
    """A single API error event record (READ-ONLY observation)."""
    timestamp: float
    category: ErrorCategory
    error_message: str
    retry_count: int
    total_wait_time: float
    final_result: str   # "success" | "retry_success" | "fallback"
    call_type: str      # "perception" | "expression" | "text" | "streaming"


@dataclass
class ErrorStats:
    """Session-cumulative API error statistics (READ-ONLY observation).

    This record is never referenced by psyche internal processing.
    It exists purely for external monitoring and debugging.
    All values are reset at session boundary (not persisted).
    """
    errors_by_category: dict[str, int] = field(default_factory=lambda: {
        c.name: 0 for c in ErrorCategory
    })
    retry_recovery_count: int = 0
    fallback_count: int = 0
    total_retries: int = 0
    total_wait_time: float = 0.0
    records: list[ErrorRecord] = field(default_factory=list)
    _max_records: int = 200  # FIFO cap for record storage

    def record_error(
        self,
        category: ErrorCategory,
        error_message: str,
        retry_count: int,
        total_wait_time: float,
        final_result: str,
        call_type: str,
    ) -> None:
        """Record an error event."""
        self.errors_by_category[category.name] = (
            self.errors_by_category.get(category.name, 0) + 1
        )
        self.total_retries += retry_count
        self.total_wait_time += total_wait_time

        if final_result == "retry_success":
            self.retry_recovery_count += 1
        elif final_result == "fallback":
            self.fallback_count += 1

        record = ErrorRecord(
            timestamp=time.time(),
            category=category,
            error_message=error_message[:200],  # Truncate for safety
            retry_count=retry_count,
            total_wait_time=total_wait_time,
            final_result=final_result,
            call_type=call_type,
        )
        self.records.append(record)

        # FIFO eviction
        while len(self.records) > self._max_records:
            self.records.pop(0)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/observation."""
        return {
            "errors_by_category": dict(self.errors_by_category),
            "retry_recovery_count": self.retry_recovery_count,
            "fallback_count": self.fallback_count,
            "total_retries": self.total_retries,
            "total_wait_time_sec": round(self.total_wait_time, 2),
        }


# ── Fallback Mode ─────────────────────────────────────────────

@dataclass
class FallbackModeState:
    """Tracks API reachability state for fallback mode management.

    When the API is unreachable (consecutive failures exceed threshold),
    the system enters fallback mode. The fallback mode has a maximum
    duration; exceeding it triggers a safe shutdown request.

    This state is never referenced by psyche processing.
    """
    consecutive_failures: int = 0
    fallback_mode_start: Optional[float] = None
    # Configurable thresholds
    failure_threshold: int = 3
    max_fallback_duration: float = 300.0  # 5 minutes

    @property
    def is_in_fallback(self) -> bool:
        """Whether the system is currently in fallback mode."""
        return self.fallback_mode_start is not None

    def on_success(self) -> None:
        """Called when an API call succeeds. Resets consecutive failures."""
        if self.is_in_fallback:
            elapsed = time.monotonic() - self.fallback_mode_start if self.fallback_mode_start else 0
            logger.info(
                "API recovered from fallback mode (was in fallback for %.1f sec)",
                elapsed,
            )
        self.consecutive_failures = 0
        self.fallback_mode_start = None

    def on_failure(self) -> None:
        """Called when an API call fails after all retries."""
        self.consecutive_failures += 1
        if (
            self.consecutive_failures >= self.failure_threshold
            and self.fallback_mode_start is None
        ):
            self.fallback_mode_start = time.monotonic()
            logger.warning(
                "Entering fallback mode after %d consecutive failures",
                self.consecutive_failures,
            )

    def should_safe_shutdown(self) -> bool:
        """Check if fallback mode duration exceeds the maximum.

        Returns True if the system should perform a safe shutdown
        (persist internal state and exit main loop).
        """
        if self.fallback_mode_start is None:
            return False
        elapsed = time.monotonic() - self.fallback_mode_start
        return elapsed >= self.max_fallback_duration

    def get_status(self) -> dict[str, Any]:
        """Return current fallback mode status for observation."""
        result: dict[str, Any] = {
            "is_in_fallback": self.is_in_fallback,
            "consecutive_failures": self.consecutive_failures,
        }
        if self.fallback_mode_start is not None:
            result["fallback_duration_sec"] = round(
                time.monotonic() - self.fallback_mode_start, 1
            )
        return result


# ── Resilient Call Wrapper ────────────────────────────────────

async def resilient_call(
    call_fn: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    call_type: str = "unknown",
    stats: Optional[ErrorStats] = None,
    fallback_state: Optional[FallbackModeState] = None,
    fallback_value: Any = None,
    **kwargs: Any,
) -> Any:
    """Execute an async callable with exponential backoff retry.

    This is the core resilient wrapper. It classifies errors, applies
    appropriate retry strategies, records observations, and manages
    fallback state.

    The retry logic is fully encapsulated. Callers see only the final
    result (success or fallback value) and are unaware of retry internals.

    Args:
        call_fn: Async callable to execute.
        *args: Positional arguments to pass to call_fn.
        config: Retry configuration. Uses defaults if None.
        call_type: Label for observation recording.
        stats: Error statistics recorder (optional).
        fallback_state: Fallback mode tracker (optional).
        fallback_value: Value to return when all retries fail.
        **kwargs: Keyword arguments to pass to call_fn.

    Returns:
        The result of call_fn on success, or fallback_value on final failure.
    """
    cfg = config or RetryConfig()
    last_error: Optional[Exception] = None
    total_wait = 0.0

    for attempt in range(cfg.max_retries + 1):  # initial + retries
        try:
            result = await asyncio.wait_for(
                call_fn(*args, **kwargs),
                timeout=cfg.timeout,
            )

            # Success
            if fallback_state is not None:
                fallback_state.on_success()

            if attempt > 0 and stats is not None:
                # Recovered via retry
                stats.record_error(
                    category=classify_error(last_error) if last_error else ErrorCategory.UNKNOWN,
                    error_message=str(last_error) if last_error else "",
                    retry_count=attempt,
                    total_wait_time=total_wait,
                    final_result="retry_success",
                    call_type=call_type,
                )
                logger.info(
                    "API call recovered after %d retries (total wait: %.1fs)",
                    attempt, total_wait,
                )

            return result

        except Exception as e:
            last_error = e
            category = classify_error(e)

            logger.warning(
                "API call error [%s] (attempt %d/%d, category=%s): %s",
                call_type, attempt + 1, cfg.max_retries + 1,
                category.name, str(e)[:200],
            )

            # Non-retryable: stop immediately
            if not is_retryable(category):
                logger.error(
                    "Permanent error, skipping retries: %s", str(e)[:200],
                )
                break

            # Last attempt: no more retries
            if attempt >= cfg.max_retries:
                break

            # Compute and apply backoff delay
            retry_after = _extract_retry_after(e) if category == ErrorCategory.RATE_LIMIT else None
            delay = compute_backoff_delay(attempt, cfg, retry_after=retry_after)
            total_wait += delay

            logger.info(
                "Retrying in %.2fs (attempt %d/%d)",
                delay, attempt + 2, cfg.max_retries + 1,
            )
            await asyncio.sleep(delay)

    # All retries exhausted or permanent error
    if fallback_state is not None:
        fallback_state.on_failure()

    if stats is not None and last_error is not None:
        stats.record_error(
            category=classify_error(last_error),
            error_message=str(last_error),
            retry_count=min(attempt, cfg.max_retries) if 'attempt' in dir() else 0,
            total_wait_time=total_wait,
            final_result="fallback",
            call_type=call_type,
        )

    logger.error(
        "API call failed after all retries [%s]: %s",
        call_type, str(last_error)[:200] if last_error else "unknown",
    )

    return fallback_value
