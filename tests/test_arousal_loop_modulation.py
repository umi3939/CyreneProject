"""
test_arousal_loop_modulation.py - Tests for arousal-based loop interval modulation.

Tests cover:
A. Basic modulation behavior (coefficient computation, interval scaling)
B. Modulation bandwidth enforcement (+-10% max)
C. Safety valve: monotonic arousal change detection
D. Safety valve: hysteresis recovery (reversal counting)
E. Safety valve: interaction with normal modulation
F. Arousal value clamping (out-of-range inputs)
G. Effect record FIFO behavior
H. Diagnostics snapshot correctness
I. Non-interference with existing 3-path control
J. Edge cases and long-term stability

All tests use explicit values (no real-time dependency).
No imports from psyche modules or any module other than loop_interval_controller.
"""

import pytest

from loop_interval_controller import (
    LoopIntervalController,
    ArousalModulationState,
    ArousalModulationRecord,
    _AROUSAL_MOD_MAX_RATIO,
    _AROUSAL_NEUTRAL,
    _AROUSAL_HISTORY_SIZE,
    _AROUSAL_MONOTONIC_THRESHOLD,
    _AROUSAL_HYSTERESIS_REVERSALS,
    _AROUSAL_EFFECT_RECORD_SIZE,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def ctrl():
    """Standard controller with default settings."""
    return LoopIntervalController()


@pytest.fixture
def fast_ctrl():
    """Controller with short intervals for testing."""
    return LoopIntervalController(
        capture_stages=[0.01, 0.02, 0.05, 0.1, 0.2],
        spontaneous_stages=[0.0, 0.5, 1.0, 2.0],
        text_consecutive_limit=3,
        base_idle_threshold=1.0,
    )


# =========================================================================
# A. Basic Modulation Behavior
# =========================================================================

class TestBasicModulation:
    """Verify basic arousal modulation coefficient and interval scaling."""

    def test_neutral_arousal_no_modulation(self, ctrl):
        """Arousal at neutral (0.5) produces coefficient 1.0 (no change)."""
        result = ctrl.apply_arousal_modulation(1.0, _AROUSAL_NEUTRAL)
        assert result == pytest.approx(1.0)

    def test_high_arousal_shortens_interval(self, ctrl):
        """Arousal above neutral produces interval shorter than base."""
        base = 2.0
        result = ctrl.apply_arousal_modulation(base, 0.8)
        assert result < base

    def test_low_arousal_lengthens_interval(self, ctrl):
        """Arousal below neutral produces interval longer than base."""
        base = 2.0
        result = ctrl.apply_arousal_modulation(base, 0.2)
        assert result > base

    def test_max_arousal_coefficient(self, ctrl):
        """Arousal at 1.0 produces minimum coefficient (1.0 - MAX_RATIO)."""
        base = 1.0
        result = ctrl.apply_arousal_modulation(base, 1.0)
        expected = base * (1.0 - _AROUSAL_MOD_MAX_RATIO)
        assert result == pytest.approx(expected)

    def test_min_arousal_coefficient(self, ctrl):
        """Arousal at 0.0 produces maximum coefficient (1.0 + MAX_RATIO)."""
        base = 1.0
        result = ctrl.apply_arousal_modulation(base, 0.0)
        expected = base * (1.0 + _AROUSAL_MOD_MAX_RATIO)
        assert result == pytest.approx(expected)

    def test_modulation_scales_with_base_interval(self, ctrl):
        """Modulation is proportional to base interval."""
        arousal = 0.8
        result_1 = ctrl.apply_arousal_modulation(1.0, arousal)
        # Reset for clean state
        ctrl2 = LoopIntervalController()
        result_10 = ctrl2.apply_arousal_modulation(10.0, arousal)
        assert result_10 / result_1 == pytest.approx(10.0, rel=0.01)

    def test_coefficient_linear_with_arousal(self, ctrl):
        """Coefficient changes linearly with arousal deviation from neutral."""
        base = 1.0
        # Arousal 0.75 (deviation +0.25 from neutral)
        r1 = ctrl.apply_arousal_modulation(base, 0.75)
        ctrl2 = LoopIntervalController()
        # Arousal 1.0 (deviation +0.5 from neutral)
        r2 = ctrl2.apply_arousal_modulation(base, 1.0)
        # The modulation effect at 0.75 should be half of that at 1.0
        mod1 = base - r1
        mod2 = base - r2
        assert mod1 == pytest.approx(mod2 / 2.0, rel=0.01)


# =========================================================================
# B. Modulation Bandwidth Enforcement
# =========================================================================

class TestBandwidthEnforcement:
    """Verify modulation never exceeds +-10% of base interval."""

    def test_max_increase_is_10_percent(self, ctrl):
        """Lowest arousal produces at most 10% increase."""
        base = 5.0
        result = ctrl.apply_arousal_modulation(base, 0.0)
        assert result <= base * (1.0 + _AROUSAL_MOD_MAX_RATIO) + 1e-10
        assert result >= base  # Must be >= base for low arousal

    def test_max_decrease_is_10_percent(self, ctrl):
        """Highest arousal produces at most 10% decrease."""
        base = 5.0
        result = ctrl.apply_arousal_modulation(base, 1.0)
        assert result >= base * (1.0 - _AROUSAL_MOD_MAX_RATIO) - 1e-10
        assert result <= base  # Must be <= base for high arousal

    def test_various_bases_bandwidth(self, ctrl):
        """Bandwidth enforcement holds for various base intervals."""
        bases = [0.01, 0.1, 0.5, 1.0, 5.0, 30.0, 100.0]
        for base in bases:
            c = LoopIntervalController()
            for arousal in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = c.apply_arousal_modulation(base, arousal)
                assert result >= base * (1.0 - _AROUSAL_MOD_MAX_RATIO) - 1e-10
                assert result <= base * (1.0 + _AROUSAL_MOD_MAX_RATIO) + 1e-10

    def test_zero_base_interval(self, ctrl):
        """Zero base interval stays zero regardless of arousal."""
        result = ctrl.apply_arousal_modulation(0.0, 1.0)
        assert result == 0.0

    def test_negative_base_interval_handled(self, ctrl):
        """Negative base interval (invalid) still returns finite value."""
        result = ctrl.apply_arousal_modulation(-1.0, 0.5)
        assert result == pytest.approx(-1.0)  # coefficient 1.0 * -1.0


# =========================================================================
# C. Safety Valve: Monotonic Arousal Change Detection
# =========================================================================

class TestSafetyValveMonotonic:
    """Verify safety valve activates on monotonic arousal changes."""

    def test_monotonic_increasing_triggers_valve(self, ctrl):
        """Consecutive increasing arousal values trigger the safety valve."""
        base = 1.0
        # Need THRESHOLD + 1 values for THRESHOLD consecutive increases
        # THRESHOLD = 4, so need 5 values (4 diffs all positive)
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(values) == _AROUSAL_MONOTONIC_THRESHOLD + 1

        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        assert ctrl.arousal_state.safety_valve_active is True

    def test_monotonic_decreasing_triggers_valve(self, ctrl):
        """Consecutive decreasing arousal values trigger the safety valve."""
        base = 1.0
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        assert len(values) == _AROUSAL_MONOTONIC_THRESHOLD + 1

        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        assert ctrl.arousal_state.safety_valve_active is True

    def test_non_monotonic_does_not_trigger(self, ctrl):
        """Non-monotonic sequence does not trigger the safety valve."""
        base = 1.0
        values = [0.2, 0.4, 0.3, 0.5, 0.4]  # oscillating

        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        assert ctrl.arousal_state.safety_valve_active is False

    def test_monotonic_with_equal_values_does_not_trigger(self, ctrl):
        """Constant (equal) values do not count as monotonic change."""
        base = 1.0
        for _ in range(_AROUSAL_MONOTONIC_THRESHOLD + 2):
            ctrl.apply_arousal_modulation(base, 0.5)

        assert ctrl.arousal_state.safety_valve_active is False

    def test_safety_valve_disables_modulation(self, ctrl):
        """When safety valve is active, modulation returns base interval."""
        base = 2.0
        # Trigger safety valve
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        assert ctrl.arousal_state.safety_valve_active is True

        # Next call should return base interval unchanged
        result = ctrl.apply_arousal_modulation(base, 0.9)
        assert result == base

    def test_fewer_than_threshold_does_not_trigger(self, ctrl):
        """Fewer than THRESHOLD consecutive same-direction changes: no trigger."""
        base = 1.0
        # Only 3 consecutive increases (threshold is 4)
        values = [0.1, 0.2, 0.3, 0.4]
        assert len(values) == _AROUSAL_MONOTONIC_THRESHOLD  # 4 values = 3 diffs

        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        assert ctrl.arousal_state.safety_valve_active is False

    def test_safety_valve_triggers_on_exact_threshold(self, ctrl):
        """Safety valve triggers at exactly THRESHOLD consecutive changes."""
        base = 1.0
        # Start with a non-directional value, then THRESHOLD+1 monotonic
        ctrl.apply_arousal_modulation(base, 0.5)  # seed
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            ctrl.apply_arousal_modulation(base, v)

        # After 0.5 seed + 5 monotonic values: last 5 diffs check
        # The deque has [0.5, 0.1, 0.2, 0.3, 0.4, 0.5] but maxlen=5
        # so it keeps [0.1, 0.2, 0.3, 0.4, 0.5] — 4 diffs all positive
        assert ctrl.arousal_state.safety_valve_active is True


# =========================================================================
# D. Safety Valve: Hysteresis Recovery
# =========================================================================

class TestSafetyValveHysteresis:
    """Verify hysteresis-based recovery from safety valve activation."""

    def _trigger_safety_valve(self, ctrl):
        """Helper to trigger the safety valve via monotonic increase."""
        base = 1.0
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(base, v)
        assert ctrl.arousal_state.safety_valve_active is True

    def test_single_reversal_not_enough(self, ctrl):
        """One direction reversal is not enough to release safety valve."""
        self._trigger_safety_valve(ctrl)
        # One reversal: up then down
        ctrl.apply_arousal_modulation(1.0, 0.6)  # continue up
        ctrl.apply_arousal_modulation(1.0, 0.4)  # reversal (down)
        if _AROUSAL_HYSTERESIS_REVERSALS > 1:
            assert ctrl.arousal_state.safety_valve_active is True

    def test_enough_reversals_releases_valve(self, ctrl):
        """Required number of reversals releases the safety valve."""
        self._trigger_safety_valve(ctrl)
        # Provide alternating values to create reversals
        # Each pair (up, down) creates one reversal
        for i in range(_AROUSAL_HYSTERESIS_REVERSALS + 1):
            ctrl.apply_arousal_modulation(1.0, 0.7)  # up
            ctrl.apply_arousal_modulation(1.0, 0.3)  # down (reversal)

        assert ctrl.arousal_state.safety_valve_active is False

    def test_modulation_resumes_after_release(self, ctrl):
        """After safety valve release, modulation works again."""
        self._trigger_safety_valve(ctrl)
        # Release: provide enough reversals
        for i in range(_AROUSAL_HYSTERESIS_REVERSALS + 1):
            ctrl.apply_arousal_modulation(1.0, 0.8)
            ctrl.apply_arousal_modulation(1.0, 0.2)

        assert ctrl.arousal_state.safety_valve_active is False

        # Now modulation should work
        result = ctrl.apply_arousal_modulation(2.0, 0.9)
        assert result < 2.0  # High arousal shortens interval

    def test_reversal_count_resets_after_release(self, ctrl):
        """Reversal count resets to 0 after safety valve release."""
        self._trigger_safety_valve(ctrl)
        # Release
        for i in range(_AROUSAL_HYSTERESIS_REVERSALS + 1):
            ctrl.apply_arousal_modulation(1.0, 0.8)
            ctrl.apply_arousal_modulation(1.0, 0.2)

        assert ctrl.arousal_state.reversal_count == 0

    def test_same_direction_does_not_count_as_reversal(self, ctrl):
        """Continued same-direction changes do not count as reversals."""
        self._trigger_safety_valve(ctrl)
        # Continue increasing (same direction, no reversal)
        for v in [0.6, 0.7, 0.8, 0.9]:
            ctrl.apply_arousal_modulation(1.0, v)

        assert ctrl.arousal_state.safety_valve_active is True
        assert ctrl.arousal_state.reversal_count == 0


# =========================================================================
# E. Safety Valve Interaction with Normal Modulation
# =========================================================================

class TestSafetyValveInteraction:
    """Test interactions between safety valve and normal modulation flow."""

    def test_valve_then_monotonic_again(self, ctrl):
        """Safety valve can re-trigger after release if monotonic resumes."""
        # First trigger
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(1.0, v)
        assert ctrl.arousal_state.safety_valve_active is True

        # Release
        for _ in range(_AROUSAL_HYSTERESIS_REVERSALS + 1):
            ctrl.apply_arousal_modulation(1.0, 0.8)
            ctrl.apply_arousal_modulation(1.0, 0.2)
        assert ctrl.arousal_state.safety_valve_active is False

        # Trigger again with new monotonic sequence
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(1.0, v)
        assert ctrl.arousal_state.safety_valve_active is True

    def test_valve_effect_log_records_safety_valve_state(self, ctrl):
        """Effect log correctly records safety_valve_active for each call."""
        base = 1.0
        # Normal modulation
        ctrl.apply_arousal_modulation(base, 0.5)
        assert ctrl.arousal_effect_log[-1].safety_valve_active is False

        # Trigger safety valve
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(base, v)
        # The last call (0.5) should have triggered the valve
        last_record = ctrl.arousal_effect_log[-1]
        assert last_record.safety_valve_active is True
        assert last_record.modulation_coefficient == 1.0

    def test_3path_control_unaffected_by_arousal_valve(self, ctrl):
        """3-path control continues normally when arousal valve is active."""
        # Trigger arousal safety valve
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(1.0, v)
        assert ctrl.arousal_state.safety_valve_active is True

        # 3-path control should still work normally
        t = 1000.0
        ctrl.on_capture_result(frame_present=False, now=t)
        assert ctrl.capture_state.current_stage == 1
        ctrl.on_capture_result(frame_present=True, now=t + 1)
        assert ctrl.capture_state.current_stage == 0

    def test_on_any_activity_does_not_reset_arousal_state(self, ctrl):
        """on_any_activity resets 3-path states but NOT arousal state."""
        # Build some arousal history
        for v in [0.3, 0.5, 0.7]:
            ctrl.apply_arousal_modulation(1.0, v)

        history_before = list(ctrl.arousal_state.arousal_history)
        ctrl.on_any_activity(now=1000.0)

        # Arousal history should be unchanged
        assert list(ctrl.arousal_state.arousal_history) == history_before
        # But 3-path states should be reset
        assert ctrl.capture_state.current_stage == 0


# =========================================================================
# F. Arousal Value Clamping
# =========================================================================

class TestArousalClamping:
    """Verify out-of-range arousal values are clamped."""

    def test_negative_arousal_clamped_to_zero(self, ctrl):
        """Negative arousal is treated as 0.0."""
        result = ctrl.apply_arousal_modulation(1.0, -0.5)
        expected = ctrl._compute_arousal_coefficient(0.0) * 1.0
        # Due to clamping internally, the stored value should be 0.0
        assert ctrl.arousal_state.arousal_history[-1] == 0.0

    def test_arousal_above_one_clamped(self, ctrl):
        """Arousal above 1.0 is treated as 1.0."""
        result = ctrl.apply_arousal_modulation(1.0, 1.5)
        expected = ctrl._compute_arousal_coefficient(1.0) * 1.0
        assert ctrl.arousal_state.arousal_history[-1] == 1.0
        assert result == pytest.approx(expected)

    def test_large_negative_arousal(self, ctrl):
        """Very large negative arousal clamped to 0.0."""
        result = ctrl.apply_arousal_modulation(2.0, -100.0)
        expected = 2.0 * (1.0 + _AROUSAL_MOD_MAX_RATIO)
        assert result == pytest.approx(expected)

    def test_large_positive_arousal(self, ctrl):
        """Very large positive arousal clamped to 1.0."""
        result = ctrl.apply_arousal_modulation(2.0, 100.0)
        expected = 2.0 * (1.0 - _AROUSAL_MOD_MAX_RATIO)
        assert result == pytest.approx(expected)

    def test_nan_like_edge_values(self, ctrl):
        """Boundary values (exactly 0.0, exactly 1.0) work correctly."""
        r0 = ctrl.apply_arousal_modulation(1.0, 0.0)
        assert r0 == pytest.approx(1.0 + _AROUSAL_MOD_MAX_RATIO)

        ctrl2 = LoopIntervalController()
        r1 = ctrl2.apply_arousal_modulation(1.0, 1.0)
        assert r1 == pytest.approx(1.0 - _AROUSAL_MOD_MAX_RATIO)


# =========================================================================
# G. Effect Record FIFO Behavior
# =========================================================================

class TestEffectRecordFIFO:
    """Verify effect record FIFO capacity and behavior."""

    def test_effect_log_starts_empty(self, ctrl):
        """Effect log is empty initially."""
        assert len(ctrl.arousal_effect_log) == 0

    def test_effect_log_grows_with_calls(self, ctrl):
        """Each apply_arousal_modulation adds one record."""
        for i in range(5):
            ctrl.apply_arousal_modulation(1.0, 0.5)
        assert len(ctrl.arousal_effect_log) == 5

    def test_effect_log_caps_at_max_size(self, ctrl):
        """Effect log does not exceed _AROUSAL_EFFECT_RECORD_SIZE."""
        for i in range(_AROUSAL_EFFECT_RECORD_SIZE + 10):
            ctrl.apply_arousal_modulation(1.0, 0.3 + (i % 3) * 0.1)
        assert len(ctrl.arousal_effect_log) == _AROUSAL_EFFECT_RECORD_SIZE

    def test_oldest_records_dropped(self, ctrl):
        """Oldest records are dropped when FIFO is full."""
        # Fill with known arousal values
        for i in range(_AROUSAL_EFFECT_RECORD_SIZE):
            ctrl.apply_arousal_modulation(1.0, 0.1)

        # Add one more with distinct value
        ctrl.apply_arousal_modulation(1.0, 0.9)

        # Oldest (first) should now be the second original 0.1 value
        assert ctrl.arousal_effect_log[0].arousal_value == 0.1
        # Newest should be 0.9
        assert ctrl.arousal_effect_log[-1].arousal_value == 0.9

    def test_record_fields_populated(self, ctrl):
        """Each record has all expected fields populated."""
        ctrl.apply_arousal_modulation(2.0, 0.7)
        rec = ctrl.arousal_effect_log[-1]
        assert rec.arousal_value == pytest.approx(0.7)
        assert rec.base_interval == pytest.approx(2.0)
        assert rec.effective_interval != 0.0
        assert rec.modulation_coefficient != 0.0
        assert isinstance(rec.safety_valve_active, bool)


# =========================================================================
# H. Diagnostics Snapshot Correctness
# =========================================================================

class TestDiagnostics:
    """Verify diagnostics snapshots are correct and complete."""

    def test_arousal_diagnostics_initial(self, ctrl):
        """Initial arousal diagnostics have correct structure."""
        diag = ctrl.get_arousal_diagnostics()
        assert diag["arousal_history"] == []
        assert diag["safety_valve_active"] is False
        assert diag["reversal_count"] == 0
        assert diag["effect_log_size"] == 0
        assert diag["recent_effects"] == []

    def test_arousal_diagnostics_after_modulation(self, ctrl):
        """Diagnostics reflect correct state after modulation calls."""
        ctrl.apply_arousal_modulation(1.0, 0.6)
        ctrl.apply_arousal_modulation(1.0, 0.4)

        diag = ctrl.get_arousal_diagnostics()
        assert len(diag["arousal_history"]) == 2
        assert diag["arousal_history"][0] == pytest.approx(0.6)
        assert diag["arousal_history"][1] == pytest.approx(0.4)
        assert diag["effect_log_size"] == 2
        assert len(diag["recent_effects"]) == 2

    def test_arousal_diagnostics_recent_effects_capped_at_5(self, ctrl):
        """recent_effects in diagnostics shows at most 5 entries."""
        for i in range(10):
            ctrl.apply_arousal_modulation(1.0, 0.3 + (i % 4) * 0.1)

        diag = ctrl.get_arousal_diagnostics()
        assert len(diag["recent_effects"]) <= 5

    def test_general_diagnostics_includes_arousal(self, ctrl):
        """get_diagnostics includes arousal_modulation section."""
        diag = ctrl.get_diagnostics()
        assert "arousal_modulation" in diag
        am = diag["arousal_modulation"]
        assert "safety_valve_active" in am
        assert "history_length" in am
        assert "effect_log_size" in am

    def test_diagnostics_safety_valve_state(self, ctrl):
        """Diagnostics correctly reflect safety valve state."""
        # Before trigger
        assert ctrl.get_diagnostics()["arousal_modulation"]["safety_valve_active"] is False

        # Trigger
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ctrl.apply_arousal_modulation(1.0, v)

        assert ctrl.get_diagnostics()["arousal_modulation"]["safety_valve_active"] is True

    def test_diagnostics_effect_record_fields(self, ctrl):
        """Effect records in diagnostics have all required fields."""
        ctrl.apply_arousal_modulation(1.5, 0.7)
        diag = ctrl.get_arousal_diagnostics()
        effect = diag["recent_effects"][0]
        assert "arousal" in effect
        assert "coefficient" in effect
        assert "base_interval" in effect
        assert "effective_interval" in effect
        assert "safety_valve_active" in effect


# =========================================================================
# I. Non-Interference with Existing 3-Path Control
# =========================================================================

class TestNonInterference:
    """Verify arousal modulation does not interfere with 3-path control."""

    def test_arousal_does_not_change_capture_state(self, ctrl):
        """apply_arousal_modulation does not modify capture state."""
        t = 1000.0
        ctrl.on_capture_result(frame_present=False, now=t)
        stage_before = ctrl.capture_state.current_stage

        ctrl.apply_arousal_modulation(1.0, 0.9)
        assert ctrl.capture_state.current_stage == stage_before

    def test_arousal_does_not_change_spontaneous_state(self, ctrl):
        """apply_arousal_modulation does not modify spontaneous state."""
        t = 1000.0
        ctrl.on_spontaneous_result(activated=False, now=t)
        stage_before = ctrl.spontaneous_state.additional_wait_stage

        ctrl.apply_arousal_modulation(1.0, 0.9)
        assert ctrl.spontaneous_state.additional_wait_stage == stage_before

    def test_arousal_does_not_change_text_state(self, ctrl):
        """apply_arousal_modulation does not modify text coordination state."""
        ctrl.on_text_processed()
        count_before = ctrl.text_capture_coord.consecutive_text_count

        ctrl.apply_arousal_modulation(1.0, 0.9)
        assert ctrl.text_capture_coord.consecutive_text_count == count_before

    def test_arousal_does_not_change_telemetry(self, ctrl):
        """apply_arousal_modulation does not modify telemetry counters."""
        t = 1000.0
        ctrl.on_capture_result(frame_present=True, now=t)
        tel_before = ctrl.telemetry.to_dict()

        ctrl.apply_arousal_modulation(1.0, 0.9)
        tel_after = ctrl.telemetry.to_dict()
        assert tel_before == tel_after

    def test_3path_does_not_change_arousal_state(self, ctrl):
        """3-path control methods do not modify arousal modulation state."""
        # Build arousal state
        ctrl.apply_arousal_modulation(1.0, 0.6)
        ctrl.apply_arousal_modulation(1.0, 0.4)
        history_before = list(ctrl.arousal_state.arousal_history)
        log_size_before = len(ctrl.arousal_effect_log)

        # Exercise all 3-path methods
        t = 1000.0
        ctrl.on_capture_result(frame_present=True, now=t)
        ctrl.on_spontaneous_result(activated=True, now=t + 1)
        ctrl.on_text_processed()
        ctrl.on_any_activity(now=t + 2)

        assert list(ctrl.arousal_state.arousal_history) == history_before
        assert len(ctrl.arousal_effect_log) == log_size_before

    def test_two_controllers_same_3path_behavior(self):
        """Two controllers behave identically on 3-path control
        regardless of arousal modulation on one of them."""
        ctrl1 = LoopIntervalController(
            capture_stages=[0.1, 0.5, 1.0],
        )
        ctrl2 = LoopIntervalController(
            capture_stages=[0.1, 0.5, 1.0],
        )

        # Apply arousal modulation only on ctrl2
        ctrl2.apply_arousal_modulation(1.0, 0.9)
        ctrl2.apply_arousal_modulation(1.0, 0.1)

        t = 1000.0
        events = [
            ("capture", False),
            ("capture", True),
            ("capture", False),
        ]
        for i, (etype, val) in enumerate(events):
            ctrl1.on_capture_result(frame_present=val, now=t + i)
            ctrl2.on_capture_result(frame_present=val, now=t + i)
            assert ctrl1.capture_state.current_stage == ctrl2.capture_state.current_stage


# =========================================================================
# J. Edge Cases and Long-Term Stability
# =========================================================================

class TestEdgeCasesAndStability:
    """Test edge cases and long-term stability of arousal modulation."""

    def test_arousal_history_fifo_caps_at_max(self, ctrl):
        """Arousal history deque does not exceed _AROUSAL_HISTORY_SIZE."""
        for i in range(_AROUSAL_HISTORY_SIZE + 10):
            ctrl.apply_arousal_modulation(1.0, (i % 10) / 10.0)
        assert len(ctrl.arousal_state.arousal_history) == _AROUSAL_HISTORY_SIZE

    def test_500_calls_all_in_range(self, ctrl):
        """500 modulation calls all produce intervals within +-10%."""
        base = 3.0
        min_allowed = base * (1.0 - _AROUSAL_MOD_MAX_RATIO) - 1e-10
        max_allowed = base * (1.0 + _AROUSAL_MOD_MAX_RATIO) + 1e-10

        for i in range(500):
            arousal = (i % 100) / 100.0
            result = ctrl.apply_arousal_modulation(base, arousal)
            # When safety valve is active, result == base
            # When not, result is within +-10%
            assert min_allowed <= result <= max_allowed, (
                f"Call {i}: arousal={arousal}, result={result}"
            )

    def test_safety_valve_trigger_release_cycle_50_times(self, ctrl):
        """Safety valve can trigger and release 50 times without issues."""
        for cycle in range(50):
            # Trigger: monotonic increase
            for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
                ctrl.apply_arousal_modulation(1.0, v)
            assert ctrl.arousal_state.safety_valve_active is True, f"Cycle {cycle}: not triggered"

            # Release: enough reversals
            for _ in range(_AROUSAL_HYSTERESIS_REVERSALS + 1):
                ctrl.apply_arousal_modulation(1.0, 0.8)
                ctrl.apply_arousal_modulation(1.0, 0.2)
            assert ctrl.arousal_state.safety_valve_active is False, f"Cycle {cycle}: not released"

    def test_constant_arousal_no_modulation_drift(self, ctrl):
        """Constant arousal value produces same effective interval every time."""
        base = 1.0
        arousal = 0.7
        results = []
        for _ in range(100):
            r = ctrl.apply_arousal_modulation(base, arousal)
            results.append(r)

        # All results should be the same (no drift)
        for r in results:
            assert r == pytest.approx(results[0])

    def test_alternating_extreme_arousal_stable(self, ctrl):
        """Rapidly alternating between 0.0 and 1.0 stays stable."""
        base = 2.0
        min_allowed = base * (1.0 - _AROUSAL_MOD_MAX_RATIO) - 1e-10
        max_allowed = base * (1.0 + _AROUSAL_MOD_MAX_RATIO) + 1e-10

        for i in range(200):
            arousal = 0.0 if i % 2 == 0 else 1.0
            result = ctrl.apply_arousal_modulation(base, arousal)
            assert min_allowed <= result <= max_allowed

    def test_arousal_modulation_state_reset(self):
        """ArousalModulationState.reset() clears all fields."""
        state = ArousalModulationState()
        state.arousal_history.append(0.5)
        state.arousal_history.append(0.7)
        state.safety_valve_active = True
        state.reversal_count = 3

        state.reset()
        assert len(state.arousal_history) == 0
        assert state.safety_valve_active is False
        assert state.reversal_count == 0

    def test_arousal_record_defaults(self):
        """ArousalModulationRecord defaults are sensible."""
        rec = ArousalModulationRecord()
        assert rec.arousal_value == 0.0
        assert rec.modulation_coefficient == 1.0
        assert rec.base_interval == 0.0
        assert rec.effective_interval == 0.0
        assert rec.safety_valve_active is False

    def test_mixed_3path_and_arousal_long_sequence(self, ctrl):
        """Long mixed sequence of 3-path events and arousal modulation."""
        t = 1000.0
        for i in range(300):
            t_now = t + i * 0.5
            arousal = 0.3 + 0.4 * ((i % 7) / 6.0)  # Varies 0.3 to 0.7

            # Mix events
            if i % 5 == 0:
                ctrl.on_text_processed()
            elif i % 5 == 1:
                ctrl.on_capture_result(frame_present=False, now=t_now)
            elif i % 5 == 2:
                ctrl.on_capture_result(frame_present=True, now=t_now)
                ctrl.on_any_activity(now=t_now)
            elif i % 5 == 3:
                ctrl.on_spontaneous_result(activated=False, now=t_now)

            # Apply arousal modulation
            base = ctrl.get_current_capture_interval()
            result = ctrl.apply_arousal_modulation(base, arousal)

            # Always within bounds
            min_allowed = base * (1.0 - _AROUSAL_MOD_MAX_RATIO) - 1e-10
            max_allowed = base * (1.0 + _AROUSAL_MOD_MAX_RATIO) + 1e-10
            assert min_allowed <= result <= max_allowed, (
                f"Iteration {i}: base={base}, arousal={arousal}, result={result}"
            )

    def test_compute_coefficient_symmetry(self, ctrl):
        """Coefficient for arousal=0.25 and arousal=0.75 are symmetric around 1.0."""
        c_low = ctrl._compute_arousal_coefficient(0.25)
        c_high = ctrl._compute_arousal_coefficient(0.75)
        # c_low should be 1.0 + delta, c_high should be 1.0 - delta
        assert c_low == pytest.approx(2.0 - c_high)

    def test_compute_coefficient_at_neutral(self, ctrl):
        """Coefficient at neutral is exactly 1.0."""
        c = ctrl._compute_arousal_coefficient(_AROUSAL_NEUTRAL)
        assert c == pytest.approx(1.0)

    def test_detect_monotonic_insufficient_history(self, ctrl):
        """_detect_monotonic_arousal returns False with insufficient history."""
        # Empty history
        assert ctrl._detect_monotonic_arousal() is False
        # Partial history
        ctrl.arousal_state.arousal_history.append(0.1)
        ctrl.arousal_state.arousal_history.append(0.2)
        assert ctrl._detect_monotonic_arousal() is False

    def test_detect_reversal_insufficient_history(self, ctrl):
        """_detect_reversal returns False with insufficient history."""
        assert ctrl._detect_reversal() is False
        ctrl.arousal_state.arousal_history.append(0.5)
        assert ctrl._detect_reversal() is False
        ctrl.arousal_state.arousal_history.append(0.6)
        assert ctrl._detect_reversal() is False
