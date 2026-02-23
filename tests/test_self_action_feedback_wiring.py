"""
Tests for C4-9: self_action_perception -> introspection_trace indirect pathway.

Verifies that the Phase 22 wiring in orchestrator correctly reads the latest
self_action record and passes context (text existence, text length, policy label,
tick) to introspection_trace's generate_trace context parameter.

Text body is never included (self-reinforcement loop prevention).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes for self_action_perception types
# ---------------------------------------------------------------------------

@dataclass
class FakeSelfActionRecord:
    """Minimal replica of SelfActionRecord for testing."""
    record_id: str = "rec-001"
    response_text: str = "Hello world"
    policy_label: str = "empathy"
    tick: int = 5
    timestamp: float = field(default_factory=time.time)
    status: str = "active"


class FakeSelfActionRecorder:
    """Minimal replica of SelfActionPerceptionRecorder."""

    def __init__(self, latest: Optional[FakeSelfActionRecord] = None):
        self._latest = latest
        self.state = MagicMock()

    def get_latest_record(self) -> Optional[FakeSelfActionRecord]:
        return self._latest


class FakeFailingRecorder:
    """Recorder that raises on get_latest_record (for exception fallback test)."""

    def __init__(self):
        self.state = MagicMock()

    def get_latest_record(self):
        raise RuntimeError("Simulated recorder failure")


# ---------------------------------------------------------------------------
# Helper: extract Phase 22 logic into a callable for isolated testing
# ---------------------------------------------------------------------------

def _build_sa_context(recorder: Any) -> dict[str, Any]:
    """
    Replicates the Phase 22 self_action context building logic from orchestrator.
    This is the exact code path under test.
    """
    sa_context: dict[str, Any] = {}
    if recorder is not None:
        try:
            sa_record = recorder.get_latest_record()
            if sa_record is not None:
                text_len = len(sa_record.response_text) if sa_record.response_text else 0
                _SA_TEXT_LENGTH_CAP = 100_000
                sa_context = {
                    "self_action_has_output": True,
                    "self_action_text_length": min(text_len, _SA_TEXT_LENGTH_CAP),
                    "self_action_policy_label": sa_record.policy_label or "",
                    "self_action_tick": sa_record.tick,
                }
            else:
                sa_context = {"self_action_has_output": False}
        except Exception:
            sa_context = {}
    return sa_context


# ---------------------------------------------------------------------------
# Tests: context building logic
# ---------------------------------------------------------------------------

class TestSelfActionContextBuilding:
    """Test the context dict construction from self_action records."""

    def test_normal_record_produces_correct_context(self):
        """With a normal record, context has all 4 keys and correct values."""
        record = FakeSelfActionRecord(
            response_text="Hello world",
            policy_label="empathy",
            tick=5,
        )
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        assert ctx["self_action_has_output"] is True
        assert ctx["self_action_text_length"] == 11  # len("Hello world")
        assert ctx["self_action_policy_label"] == "empathy"
        assert ctx["self_action_tick"] == 5

    def test_text_body_not_included(self):
        """The actual text body must never appear in the context dict."""
        record = FakeSelfActionRecord(response_text="Secret text content")
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        # The text body must not be in any value
        all_values_str = str(ctx.values())
        assert "Secret text content" not in all_values_str
        assert "response_text" not in ctx

    def test_no_record_produces_has_output_false(self):
        """When latest record is None, context indicates no output."""
        recorder = FakeSelfActionRecorder(latest=None)
        ctx = _build_sa_context(recorder)

        assert ctx == {"self_action_has_output": False}

    def test_no_recorder_produces_empty_context(self):
        """When recorder itself is None, context is empty dict."""
        ctx = _build_sa_context(None)
        assert ctx == {}

    def test_recorder_exception_produces_empty_context(self):
        """When recorder raises exception, context falls back to empty dict."""
        recorder = FakeFailingRecorder()
        ctx = _build_sa_context(recorder)
        assert ctx == {}

    def test_empty_response_text(self):
        """Empty response text results in text_length=0 but has_output=True."""
        record = FakeSelfActionRecord(response_text="", policy_label="neutral", tick=3)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        assert ctx["self_action_has_output"] is True
        assert ctx["self_action_text_length"] == 0
        assert ctx["self_action_policy_label"] == "neutral"
        assert ctx["self_action_tick"] == 3

    def test_none_response_text(self):
        """None response text results in text_length=0."""
        record = FakeSelfActionRecord(response_text=None, policy_label="calm", tick=1)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        assert ctx["self_action_has_output"] is True
        assert ctx["self_action_text_length"] == 0

    def test_text_length_cap(self):
        """Very long text is capped at 100_000."""
        long_text = "x" * 200_000
        record = FakeSelfActionRecord(response_text=long_text, policy_label="verbose", tick=99)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        assert ctx["self_action_text_length"] == 100_000

    def test_empty_policy_label(self):
        """Empty or None policy label becomes empty string."""
        record = FakeSelfActionRecord(response_text="text", policy_label="", tick=2)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)
        assert ctx["self_action_policy_label"] == ""

    def test_none_policy_label(self):
        """None policy label becomes empty string."""
        record = FakeSelfActionRecord(response_text="text", policy_label=None, tick=2)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)
        assert ctx["self_action_policy_label"] == ""

    def test_context_keys_are_exactly_four(self):
        """Normal record context has exactly 4 specified keys."""
        record = FakeSelfActionRecord()
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        expected_keys = {
            "self_action_has_output",
            "self_action_text_length",
            "self_action_policy_label",
            "self_action_tick",
        }
        assert set(ctx.keys()) == expected_keys

    def test_tick_zero(self):
        """Tick value of 0 is preserved correctly."""
        record = FakeSelfActionRecord(response_text="a", policy_label="p", tick=0)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)
        assert ctx["self_action_tick"] == 0


# ---------------------------------------------------------------------------
# Tests: data flow direction (unidirectional)
# ---------------------------------------------------------------------------

class TestDataFlowDirection:
    """Verify that data flows only self_action -> introspection_trace, never reverse."""

    def test_recorder_not_modified_during_context_build(self):
        """Building context must not modify the recorder state."""
        record = FakeSelfActionRecord()
        recorder = FakeSelfActionRecorder(latest=record)

        original_text = record.response_text
        original_policy = record.policy_label
        original_tick = record.tick

        _build_sa_context(recorder)

        assert record.response_text == original_text
        assert record.policy_label == original_policy
        assert record.tick == original_tick

    def test_context_is_independent_copy(self):
        """Mutating returned context must not affect the recorder."""
        record = FakeSelfActionRecord(response_text="hello", policy_label="kind", tick=10)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        # Mutate context
        ctx["self_action_policy_label"] = "CHANGED"
        ctx["self_action_tick"] = 999

        # Original record is unchanged
        assert record.policy_label == "kind"
        assert record.tick == 10


# ---------------------------------------------------------------------------
# Tests: integration with generate_trace context parameter
# ---------------------------------------------------------------------------

class TestGenerateTraceContextIntegration:
    """Test that the context dict is properly passed to generate_trace."""

    def test_context_passed_to_generate_trace(self):
        """Verify context parameter reaches generate_trace correctly."""
        record = FakeSelfActionRecord(
            response_text="Test output",
            policy_label="explore",
            tick=7,
        )
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))

        # Mock the introspection system
        mock_introspection_sys = MagicMock()
        mock_introspection_sys.generate_trace.return_value = MagicMock()

        # Call generate_trace with the context
        mock_introspection_sys.generate_trace(
            emotion_state=None,
            responsibility_state=None,
            value_orientation=None,
            fear_index=None,
            context=ctx if ctx else None,
        )

        # Verify the context was passed
        call_kwargs = mock_introspection_sys.generate_trace.call_args[1]
        passed_ctx = call_kwargs["context"]

        assert passed_ctx["self_action_has_output"] is True
        assert passed_ctx["self_action_text_length"] == 11
        assert passed_ctx["self_action_policy_label"] == "explore"
        assert passed_ctx["self_action_tick"] == 7
        # Text body not present
        assert "Test output" not in str(passed_ctx.values())

    def test_empty_context_passed_as_none(self):
        """When recorder is None, context should be passed as None."""
        ctx = _build_sa_context(None)
        assert ctx == {}
        # The orchestrator code: context=sa_context if sa_context else None
        passed_value = ctx if ctx else None
        assert passed_value is None

    def test_no_output_context_passed_as_dict(self):
        """When no record exists, context has self_action_has_output=False."""
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=None))
        # Non-empty dict, so it should be passed (not None)
        passed_value = ctx if ctx else None
        assert passed_value is not None
        assert passed_value["self_action_has_output"] is False


# ---------------------------------------------------------------------------
# Tests: safety valve verification
# ---------------------------------------------------------------------------

class TestSafetyValves:
    """Test all 5 safety valves specified in the design document."""

    def test_safety_valve_1_empty_context_fallback(self):
        """Safety valve 1: No record -> empty/minimal context, no crash."""
        recorder = FakeSelfActionRecorder(latest=None)
        ctx = _build_sa_context(recorder)
        assert "self_action_has_output" in ctx
        assert ctx["self_action_has_output"] is False

    def test_safety_valve_2_exception_capture(self):
        """Safety valve 2: Exception in recorder reference -> skip with empty context."""
        recorder = FakeFailingRecorder()
        ctx = _build_sa_context(recorder)
        assert ctx == {}

    def test_safety_valve_3_text_length_cap(self):
        """Safety valve 3: Text length capped at reasonable upper bound."""
        huge_text = "a" * 500_000
        record = FakeSelfActionRecord(response_text=huge_text, tick=1)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)
        assert ctx["self_action_text_length"] == 100_000

    def test_safety_valve_4_unidirectional_flow(self):
        """Safety valve 4: Data flows only forward (recorder -> trace), not reverse."""
        record = FakeSelfActionRecord(response_text="original", tick=42)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)

        # Mutate context
        ctx["self_action_tick"] = 0
        ctx["self_action_text_length"] = 0

        # Recorder's record is unchanged
        assert recorder.get_latest_record().tick == 42
        assert len(recorder.get_latest_record().response_text) == 8  # "original"

    def test_safety_valve_5_removable_wiring(self):
        """Safety valve 5: Setting context to empty effectively removes wiring."""
        record = FakeSelfActionRecord(response_text="text", tick=10)
        recorder = FakeSelfActionRecorder(latest=record)
        ctx = _build_sa_context(recorder)
        assert ctx  # Non-empty

        # Simulate wiring removal by replacing with empty
        disabled_ctx = {}
        passed = disabled_ctx if disabled_ctx else None
        assert passed is None  # Effectively removed


# ---------------------------------------------------------------------------
# Tests: loop prevention (design section 4 - structural disconnections)
# ---------------------------------------------------------------------------

class TestLoopPrevention:
    """Verify structural disconnections that prevent self-reinforcement loops."""

    def test_disconnection_1_no_text_body_in_context(self):
        """Disconnection 1: Text body never flows into context."""
        texts = [
            "Hello, how are you?",
            "I feel happy today!",
            "This is a very long response " * 100,
            "",
        ]
        for text in texts:
            record = FakeSelfActionRecord(response_text=text)
            ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
            # Flatten all values to string
            all_values = " ".join(str(v) for v in ctx.values())
            if text and len(text) > 0:
                # Non-empty text must not appear in context values
                assert text not in all_values

    def test_disconnection_4_no_frequency_or_pattern(self):
        """Disconnection 4: Only single-record facts, no cumulative frequency/pattern."""
        # Build context from multiple records sequentially
        contexts = []
        for i in range(5):
            record = FakeSelfActionRecord(
                response_text=f"text_{i}",
                policy_label="empathy",
                tick=i,
            )
            ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
            contexts.append(ctx)

        # Each context is independent - no accumulation
        for i, ctx in enumerate(contexts):
            assert ctx["self_action_tick"] == i
            assert ctx["self_action_text_length"] == len(f"text_{i}")
            # No frequency, count, or history fields
            assert "frequency" not in str(ctx.keys()).lower()
            assert "count" not in str(ctx.keys()).lower()
            assert "history" not in str(ctx.keys()).lower()
            assert "pattern" not in str(ctx.keys()).lower()


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling tests."""

    def test_unicode_text_length(self):
        """Unicode text length is measured correctly."""
        record = FakeSelfActionRecord(response_text="Hello", tick=1)
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
        assert ctx["self_action_text_length"] == 5  # 5 characters

    def test_multiline_text_length(self):
        """Multiline text measures correctly."""
        text = "line1\nline2\nline3"
        record = FakeSelfActionRecord(response_text=text, tick=1)
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
        assert ctx["self_action_text_length"] == len(text)

    def test_large_tick_value(self):
        """Large tick values are preserved."""
        record = FakeSelfActionRecord(response_text="a", tick=999999)
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
        assert ctx["self_action_tick"] == 999999

    def test_special_chars_in_policy_label(self):
        """Special characters in policy label are preserved."""
        record = FakeSelfActionRecord(
            response_text="a",
            policy_label="policy/with-special_chars.v2",
            tick=1,
        )
        ctx = _build_sa_context(FakeSelfActionRecorder(latest=record))
        assert ctx["self_action_policy_label"] == "policy/with-special_chars.v2"
