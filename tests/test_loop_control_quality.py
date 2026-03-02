"""
test_loop_control_quality.py - Extended quality tests for the 3-path loop controller.

Tests cover:
A. Long-term sequence stability (hundreds of events)
B. 3-path chain transition patterns
C. Boundary value cycle tests (max -> reset -> max cycles)
D. Time regression resilience (backward timestamps)
E. High-frequency toggle at scale (hundreds of rapid alternations)
F. Forced capture full state verification
G. Telemetry counter correctness and overflow safety

All tests use explicit timestamps only (no real-time dependency).
No imports from psyche modules or any module other than loop_interval_controller.
"""

import pytest

from loop_interval_controller import (
    LoopIntervalController,
    TelemetryCounters,
    _TELEMETRY_COUNTER_MAX,
    CAPTURE_INTERVAL_STAGES,
    SPONTANEOUS_WAIT_STAGES,
    TEXT_CONSECUTIVE_LIMIT,
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


@pytest.fixture
def simple_ctrl():
    """Controller with minimal stages for boundary testing."""
    return LoopIntervalController(
        capture_stages=[0.1, 0.5, 1.0],
        spontaneous_stages=[0.0, 5.0],
        text_consecutive_limit=2,
        base_idle_threshold=10.0,
    )


# =========================================================================
# A. Long-term Sequence Stability (hundreds of events)
# =========================================================================

class TestLongTermSequenceStability:
    """Verify that stage values remain within defined ranges over hundreds of events."""

    def test_capture_500_no_change_stays_in_range(self, ctrl):
        """500 consecutive no-change captures keep stage within defined range."""
        max_stage = len(ctrl.capture_stages) - 1
        max_interval = ctrl.capture_stages[-1]
        t = 1000.0

        for i in range(500):
            ctrl.on_capture_result(frame_present=False, now=t + i * 0.5)
            assert 0 <= ctrl.capture_state.current_stage <= max_stage
            assert ctrl.get_current_capture_interval() <= max_interval
            assert ctrl.capture_state.consecutive_no_change == i + 1

    def test_capture_500_alternating_stays_in_range(self, ctrl):
        """500 alternating frame_present True/False keeps stage within range."""
        max_stage = len(ctrl.capture_stages) - 1
        t = 1000.0

        for i in range(500):
            frame_present = (i % 3 != 0)  # True, True, False, True, True, False...
            ctrl.on_capture_result(frame_present=frame_present, now=t + i * 0.1)
            assert 0 <= ctrl.capture_state.current_stage <= max_stage

    def test_spontaneous_500_not_activated_stays_in_range(self, ctrl):
        """500 consecutive not-activated results keep wait stage in range."""
        max_stage = len(ctrl.spontaneous_stages) - 1
        max_threshold = ctrl.base_idle_threshold + ctrl.spontaneous_stages[-1]
        t = 1000.0

        for i in range(500):
            ctrl.on_spontaneous_result(activated=False, now=t + i * 10)
            assert 0 <= ctrl.spontaneous_state.additional_wait_stage <= max_stage
            assert ctrl.get_effective_idle_threshold() <= max_threshold

    def test_mixed_pathway_500_events(self, ctrl):
        """500 events across all 3 pathways keep all states in range."""
        max_capture_stage = len(ctrl.capture_stages) - 1
        max_spontaneous_stage = len(ctrl.spontaneous_stages) - 1
        t = 1000.0

        for i in range(500):
            t_now = t + i * 0.5
            pathway = i % 5

            if pathway == 0:
                ctrl.on_text_processed()
                ctrl.check_and_apply_text_capture_reset(now=t_now)
            elif pathway == 1:
                ctrl.on_capture_result(frame_present=False, now=t_now)
            elif pathway == 2:
                ctrl.on_capture_result(frame_present=True, now=t_now)
                ctrl.on_any_activity(now=t_now)
            elif pathway == 3:
                ctrl.on_spontaneous_result(activated=False, now=t_now)
            else:
                ctrl.on_spontaneous_result(activated=True, now=t_now)
                ctrl.on_any_activity(now=t_now)

            # All stages in range after every event
            assert 0 <= ctrl.capture_state.current_stage <= max_capture_stage
            assert 0 <= ctrl.spontaneous_state.additional_wait_stage <= max_spontaneous_stage
            assert ctrl.text_capture_coord.consecutive_text_count >= 0

    def test_text_consecutive_count_never_negative(self, ctrl):
        """Consecutive text count never goes below 0 over long sequences."""
        t = 1000.0
        for i in range(300):
            t_now = t + i * 0.1
            if i % 7 == 0:
                ctrl.on_text_processed()
            elif i % 11 == 0:
                ctrl.on_capture_after_text(now=t_now)
            elif i % 13 == 0:
                ctrl.on_any_activity(now=t_now)
            assert ctrl.text_capture_coord.consecutive_text_count >= 0


# =========================================================================
# B. 3-Path Chain Transition Patterns
# =========================================================================

class TestThreePathChainTransitions:
    """Test sequences where activity on one path affects other paths."""

    def test_text_then_capture_forced_then_spontaneous_suppressed(self, simple_ctrl):
        """Text input -> forced capture -> spontaneous check suppressed in sequence."""
        t = 1000.0

        # Text input x2 (reaches limit of 2)
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

        # Next loop: apply text reset, then forced capture
        simple_ctrl.check_and_apply_text_capture_reset(now=t + 1)
        assert simple_ctrl.capture_state.current_stage == 0

        # Forced capture: change detected
        simple_ctrl.on_capture_result(frame_present=True, now=t + 1.1)
        simple_ctrl.on_capture_after_text(now=t + 1.1)
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

        # Activity resets all
        simple_ctrl.on_any_activity(now=t + 1.2)
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0

        # Spontaneous check: idle not enough yet (just had activity)
        simple_ctrl._spontaneous_state.last_check_time = t + 1.2
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=5.0, now=t + 3.0,
        )

    def test_capture_change_resets_all_then_text_then_spontaneous(self, simple_ctrl):
        """Capture (change) -> full reset -> text -> spontaneous in chain."""
        t = 1000.0

        # Advance capture and spontaneous states
        simple_ctrl.on_capture_result(frame_present=False, now=t)
        simple_ctrl.on_capture_result(frame_present=False, now=t + 1)
        simple_ctrl.on_spontaneous_result(activated=False, now=t + 2)
        assert simple_ctrl.capture_state.current_stage == 2
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 1

        # Capture with change -> reset all
        simple_ctrl.on_capture_result(frame_present=True, now=t + 3)
        simple_ctrl.on_any_activity(now=t + 3)
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

        # Text input
        simple_ctrl.on_text_processed()
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 1

        # Next loop: text reset applied
        simple_ctrl.check_and_apply_text_capture_reset(now=t + 4)
        assert simple_ctrl.capture_state.current_stage == 0

        # Spontaneous check (enough idle)
        simple_ctrl._spontaneous_state.last_check_time = t + 3
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=10.0, now=t + 14,
        )

    def test_spontaneous_activated_resets_all_then_text_then_capture(self, simple_ctrl):
        """Spontaneous activation -> reset all -> text -> capture forced chain."""
        t = 1000.0

        # Advance states
        simple_ctrl.on_capture_result(frame_present=False, now=t)
        simple_ctrl.on_spontaneous_result(activated=False, now=t)
        simple_ctrl.on_text_processed()
        assert simple_ctrl.capture_state.current_stage == 1
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 1
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 1

        # Spontaneous activation -> reset all
        simple_ctrl.on_spontaneous_result(activated=True, now=t + 10)
        simple_ctrl.on_any_activity(now=t + 10)
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

        # Text x2 -> forced capture
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

        # Forced capture
        simple_ctrl.on_capture_result(frame_present=False, now=t + 11)
        simple_ctrl.on_capture_after_text(now=t + 11)
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0
        assert simple_ctrl.capture_state.current_stage == 1  # no-change advances

    def test_multiple_resets_in_sequence(self, simple_ctrl):
        """Multiple on_any_activity calls in sequence are idempotent."""
        t = 1000.0

        # Advance states
        simple_ctrl.on_capture_result(frame_present=False, now=t)
        simple_ctrl.on_spontaneous_result(activated=False, now=t)
        simple_ctrl.on_text_processed()

        # Multiple resets
        for i in range(5):
            simple_ctrl.on_any_activity(now=t + 1 + i * 0.1)
            assert simple_ctrl.capture_state.current_stage == 0
            assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
            assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

    def test_chain_text_capture_text_capture_text(self, simple_ctrl):
        """Alternating text -> capture 5 times maintains proper coordination."""
        t = 1000.0

        for cycle in range(5):
            base_t = t + cycle * 10

            # Text processed
            simple_ctrl.on_text_processed()
            # Next tick: reset applied
            simple_ctrl.check_and_apply_text_capture_reset(now=base_t + 1)
            assert simple_ctrl.capture_state.current_stage == 0

            # Capture (no change)
            simple_ctrl.on_capture_result(frame_present=False, now=base_t + 2)
            assert simple_ctrl.capture_state.current_stage == 1

    def test_text_burst_to_forced_capture_full_cycle(self, fast_ctrl):
        """Text burst to limit -> forced capture -> verify all state fields."""
        t = 1000.0

        # Text x3 (limit=3)
        for _ in range(3):
            fast_ctrl.on_text_processed()
        assert fast_ctrl.should_force_capture_after_text()
        assert fast_ctrl.text_capture_coord.consecutive_text_count == 3

        # Apply text-capture reset before forced capture
        fast_ctrl.check_and_apply_text_capture_reset(now=t)
        assert fast_ctrl.capture_state.current_stage == 0

        # Forced capture (change detected)
        fast_ctrl.on_capture_result(frame_present=True, now=t + 0.1)
        fast_ctrl.on_capture_after_text(now=t + 0.1)

        # Verify full state
        assert fast_ctrl.capture_state.current_stage == 0
        assert fast_ctrl.capture_state.consecutive_no_change == 0
        assert fast_ctrl.text_capture_coord.consecutive_text_count == 0
        assert not fast_ctrl.should_force_capture_after_text()


# =========================================================================
# C. Boundary Value Cycle Tests
# =========================================================================

class TestBoundaryValueCycles:
    """Test max -> reset -> max cycles for repeated boundary traversal."""

    def test_capture_stage_max_reset_cycle_10_times(self, simple_ctrl):
        """Repeatedly reaching max capture stage then resetting 10 times."""
        max_stage = len(simple_ctrl.capture_stages) - 1
        t = 1000.0

        for cycle in range(10):
            base_t = t + cycle * 100

            # Advance to max stage
            for i in range(max_stage + 5):  # +5 to verify clamping
                simple_ctrl.on_capture_result(
                    frame_present=False, now=base_t + i,
                )
            assert simple_ctrl.capture_state.current_stage == max_stage

            # Reset via frame detected
            simple_ctrl.on_capture_result(
                frame_present=True, now=base_t + 50,
            )
            assert simple_ctrl.capture_state.current_stage == 0
            assert simple_ctrl.capture_state.consecutive_no_change == 0

    def test_spontaneous_stage_max_reset_cycle_10_times(self, simple_ctrl):
        """Repeatedly reaching max spontaneous stage then resetting 10 times."""
        max_stage = len(simple_ctrl.spontaneous_stages) - 1
        t = 1000.0

        for cycle in range(10):
            base_t = t + cycle * 100

            # Advance to max stage
            for i in range(max_stage + 5):
                simple_ctrl.on_spontaneous_result(
                    activated=False, now=base_t + i * 10,
                )
            assert simple_ctrl.spontaneous_state.additional_wait_stage == max_stage

            # Reset via activation
            simple_ctrl.on_spontaneous_result(
                activated=True, now=base_t + 60,
            )
            assert simple_ctrl.spontaneous_state.additional_wait_stage == 0

    def test_text_count_limit_reset_cycle_10_times(self, simple_ctrl):
        """Repeatedly reaching text limit then resetting via capture 10 times."""
        limit = simple_ctrl.text_consecutive_limit
        t = 1000.0

        for cycle in range(10):
            # Reach limit
            for _ in range(limit):
                simple_ctrl.on_text_processed()
            assert simple_ctrl.should_force_capture_after_text()
            assert simple_ctrl.text_capture_coord.consecutive_text_count == limit

            # Reset via capture after text
            simple_ctrl.on_capture_after_text(now=t + cycle * 10)
            assert simple_ctrl.text_capture_coord.consecutive_text_count == 0
            assert not simple_ctrl.should_force_capture_after_text()

    def test_all_states_max_then_on_any_activity_cycle(self, simple_ctrl):
        """All states at max, then on_any_activity resets, repeated 10 times."""
        max_capture = len(simple_ctrl.capture_stages) - 1
        max_spontaneous = len(simple_ctrl.spontaneous_stages) - 1
        limit = simple_ctrl.text_consecutive_limit
        t = 1000.0

        for cycle in range(10):
            base_t = t + cycle * 100

            # Advance all to max
            for i in range(max_capture + 2):
                simple_ctrl.on_capture_result(
                    frame_present=False, now=base_t + i,
                )
            for i in range(max_spontaneous + 2):
                simple_ctrl.on_spontaneous_result(
                    activated=False, now=base_t + 20 + i * 10,
                )
            for _ in range(limit + 2):
                simple_ctrl.on_text_processed()

            assert simple_ctrl.capture_state.current_stage == max_capture
            assert simple_ctrl.spontaneous_state.additional_wait_stage == max_spontaneous
            assert simple_ctrl.text_capture_coord.consecutive_text_count >= limit

            # Reset all
            simple_ctrl.on_any_activity(now=base_t + 50)
            assert simple_ctrl.capture_state.current_stage == 0
            assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
            assert simple_ctrl.text_capture_coord.consecutive_text_count == 0


# =========================================================================
# D. Time Regression Resilience
# =========================================================================

class TestTimeRegressionResilience:
    """Ensure backward timestamps do not crash the controller."""

    def test_capture_result_backward_time_no_crash(self, ctrl):
        """on_capture_result with backward timestamps does not raise."""
        ctrl.on_capture_result(frame_present=False, now=1000.0)
        ctrl.on_capture_result(frame_present=False, now=500.0)
        ctrl.on_capture_result(frame_present=True, now=200.0)
        # Should not crash; stage values remain valid
        assert 0 <= ctrl.capture_state.current_stage <= len(ctrl.capture_stages) - 1

    def test_spontaneous_result_backward_time_no_crash(self, ctrl):
        """on_spontaneous_result with backward timestamps does not raise."""
        ctrl.on_spontaneous_result(activated=False, now=1000.0)
        ctrl.on_spontaneous_result(activated=True, now=500.0)
        ctrl.on_spontaneous_result(activated=False, now=200.0)
        assert 0 <= ctrl.spontaneous_state.additional_wait_stage <= len(ctrl.spontaneous_stages) - 1

    def test_on_any_activity_backward_time_no_crash(self, ctrl):
        """on_any_activity with backward timestamps does not raise."""
        ctrl.on_any_activity(now=1000.0)
        ctrl.on_any_activity(now=500.0)
        # Should not crash; states should be reset
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.spontaneous_state.additional_wait_stage == 0

    def test_should_attempt_capture_backward_no_crash(self, ctrl):
        """should_attempt_capture with backward time does not crash."""
        ctrl.on_capture_result(frame_present=False, now=1000.0)
        # Backward time: elapsed will be negative
        result = ctrl.should_attempt_capture(now=500.0)
        # Should return False (negative elapsed)
        assert result is False

    def test_should_check_spontaneous_backward_no_crash(self, ctrl):
        """should_check_spontaneous with backward time does not crash."""
        ctrl.on_spontaneous_result(activated=False, now=1000.0)
        result = ctrl.should_check_spontaneous(idle_seconds=100.0, now=500.0)
        # Should not crash; result may be False due to negative time_since_last_check
        assert isinstance(result, bool)

    def test_check_and_apply_backward_no_crash(self, ctrl):
        """check_and_apply_text_capture_reset with backward time does not crash."""
        ctrl.on_text_processed()
        result = ctrl.check_and_apply_text_capture_reset(now=0.0)
        assert result is True
        assert ctrl.capture_state.current_stage == 0

    def test_on_capture_after_text_backward_no_crash(self, ctrl):
        """on_capture_after_text with backward time does not crash."""
        ctrl.on_text_processed()
        ctrl.on_text_processed()
        ctrl.on_capture_after_text(now=0.0)
        assert ctrl.text_capture_coord.consecutive_text_count == 0

    def test_mixed_backward_forward_sequence(self, ctrl):
        """Mix of backward and forward timestamps maintains valid state."""
        times = [100.0, 50.0, 200.0, 10.0, 300.0, 150.0, 400.0]
        max_stage = len(ctrl.capture_stages) - 1

        for t in times:
            ctrl.on_capture_result(frame_present=False, now=t)
            assert 0 <= ctrl.capture_state.current_stage <= max_stage

        for t in times:
            ctrl.on_spontaneous_result(activated=False, now=t)
            assert 0 <= ctrl.spontaneous_state.additional_wait_stage <= len(ctrl.spontaneous_stages) - 1


# =========================================================================
# E. High-Frequency Toggle at Scale
# =========================================================================

class TestHighFrequencyToggle:
    """Test rapid True/False alternation over hundreds of iterations."""

    def test_capture_toggle_500_iterations(self, ctrl):
        """500 rapid capture result alternations stay in valid range."""
        max_stage = len(ctrl.capture_stages) - 1
        t = 1000.0

        for i in range(500):
            frame_present = (i % 2 == 0)
            ctrl.on_capture_result(frame_present=frame_present, now=t + i * 0.01)
            assert 0 <= ctrl.capture_state.current_stage <= max_stage

        # After 500 iterations ending on odd (False), stage should be at most 1
        # because True resets to 0 and False advances by 1
        assert ctrl.capture_state.current_stage <= 1

    def test_spontaneous_toggle_500_iterations(self, ctrl):
        """500 rapid spontaneous result alternations stay in valid range."""
        max_stage = len(ctrl.spontaneous_stages) - 1
        t = 1000.0

        for i in range(500):
            activated = (i % 2 == 0)
            ctrl.on_spontaneous_result(activated=activated, now=t + i * 0.01)
            assert 0 <= ctrl.spontaneous_state.additional_wait_stage <= max_stage

        # Ending on odd (False), stage should be 1
        assert ctrl.spontaneous_state.additional_wait_stage <= 1

    def test_capture_toggle_every_3rd_500_iterations(self, ctrl):
        """500 iterations where every 3rd capture has change: stage bounded."""
        max_stage = len(ctrl.capture_stages) - 1
        t = 1000.0

        for i in range(500):
            frame_present = (i % 3 == 0)
            ctrl.on_capture_result(frame_present=frame_present, now=t + i * 0.1)
            assert 0 <= ctrl.capture_state.current_stage <= max_stage

        # After pattern ..FFF.FFF.. with resets every 3rd, max advance is 2
        assert ctrl.capture_state.current_stage <= 2

    def test_on_any_activity_rapid_300_iterations(self, ctrl):
        """300 rapid on_any_activity calls keep everything at base."""
        t = 1000.0

        for i in range(300):
            # Advance a bit first
            ctrl.on_capture_result(frame_present=False, now=t + i * 2)
            ctrl.on_any_activity(now=t + i * 2 + 1)

            assert ctrl.capture_state.current_stage == 0
            assert ctrl.spontaneous_state.additional_wait_stage == 0
            assert ctrl.text_capture_coord.consecutive_text_count == 0

    def test_text_process_and_reset_rapid_300_iterations(self, ctrl):
        """300 rapid text process -> capture after text cycles."""
        t = 1000.0

        for i in range(300):
            ctrl.on_text_processed()
            ctrl.on_capture_after_text(now=t + i)
            assert ctrl.text_capture_coord.consecutive_text_count == 0


# =========================================================================
# F. Forced Capture Full State Verification
# =========================================================================

class TestForcedCaptureFullState:
    """After text consecutive limit -> forced capture, verify ALL state fields."""

    def test_full_state_after_forced_capture_default(self, ctrl):
        """Default controller: reach text limit, forced capture, check all fields."""
        t = 1000.0

        # Advance capture to some stage first
        for i in range(3):
            ctrl.on_capture_result(frame_present=False, now=t + i)
        assert ctrl.capture_state.current_stage == 3

        # Advance spontaneous
        ctrl.on_spontaneous_result(activated=False, now=t + 5)
        assert ctrl.spontaneous_state.additional_wait_stage == 1

        # Reach text limit
        for _ in range(TEXT_CONSECUTIVE_LIMIT):
            ctrl.on_text_processed()
        assert ctrl.should_force_capture_after_text()

        # Apply text-capture reset (simulating start of next loop)
        ctrl.check_and_apply_text_capture_reset(now=t + 10)

        # Verify capture state after text reset
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.capture_state.consecutive_no_change == 0

        # Forced capture happens (no change)
        ctrl.on_capture_result(frame_present=False, now=t + 10.1)
        ctrl.on_capture_after_text(now=t + 10.1)

        # Verify all fields
        assert ctrl.capture_state.current_stage == 1  # advanced by no-change
        assert ctrl.capture_state.consecutive_no_change == 1
        assert ctrl.text_capture_coord.consecutive_text_count == 0
        assert not ctrl.should_force_capture_after_text()
        # Spontaneous should be unchanged (not affected by text/capture)
        assert ctrl.spontaneous_state.additional_wait_stage == 1

    def test_full_state_after_forced_capture_with_change(self, simple_ctrl):
        """Forced capture with frame change: all states verified."""
        t = 1000.0

        # Reach text limit (2)
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

        # Apply text reset
        simple_ctrl.check_and_apply_text_capture_reset(now=t)

        # Forced capture: change detected
        simple_ctrl.on_capture_result(frame_present=True, now=t + 0.1)
        simple_ctrl.on_capture_after_text(now=t + 0.1)

        # Activity resets all
        simple_ctrl.on_any_activity(now=t + 0.2)

        # All states at base
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.capture_state.consecutive_no_change == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0
        assert not simple_ctrl.should_force_capture_after_text()
        assert simple_ctrl.text_capture_coord.text_just_processed is False

    def test_diagnostics_after_forced_capture(self, simple_ctrl):
        """Diagnostics reflect correct state after forced capture scenario."""
        t = 1000.0

        # Reach text limit, forced capture (no change)
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        simple_ctrl.check_and_apply_text_capture_reset(now=t)
        simple_ctrl.on_capture_result(frame_present=False, now=t + 0.1)
        simple_ctrl.on_capture_after_text(now=t + 0.1)

        diag = simple_ctrl.get_diagnostics()
        assert diag["capture"]["current_stage"] == 1
        assert diag["text_capture_coord"]["consecutive_text_count"] == 0
        assert diag["text_capture_coord"]["force_capture_needed"] is False


# =========================================================================
# G. Telemetry Counter Correctness and Overflow Safety
# =========================================================================

class TestTelemetryCounters:
    """Test telemetry counters are recorded accurately and safely."""

    def test_initial_counters_zero(self, ctrl):
        """All telemetry counters start at zero."""
        t = ctrl.telemetry
        assert t.text_processed_count == 0
        assert t.capture_attempt_count == 0
        assert t.capture_change_detected_count == 0
        assert t.spontaneous_check_count == 0
        assert t.spontaneous_activated_count == 0
        assert t.all_pathway_reset_count == 0

    def test_text_processing_increments_counter(self, ctrl):
        """on_text_processed increments text_processed_count."""
        for i in range(7):
            ctrl.on_text_processed()
        assert ctrl.telemetry.text_processed_count == 7

    def test_capture_result_increments_counters(self, ctrl):
        """on_capture_result increments attempt count and conditionally change count."""
        t = 1000.0
        ctrl.on_capture_result(frame_present=False, now=t)
        ctrl.on_capture_result(frame_present=True, now=t + 1)
        ctrl.on_capture_result(frame_present=False, now=t + 2)
        ctrl.on_capture_result(frame_present=True, now=t + 3)
        ctrl.on_capture_result(frame_present=True, now=t + 4)

        assert ctrl.telemetry.capture_attempt_count == 5
        assert ctrl.telemetry.capture_change_detected_count == 3

    def test_spontaneous_result_increments_counters(self, ctrl):
        """on_spontaneous_result increments check count and conditionally activated count."""
        t = 1000.0
        ctrl.on_spontaneous_result(activated=False, now=t)
        ctrl.on_spontaneous_result(activated=True, now=t + 1)
        ctrl.on_spontaneous_result(activated=False, now=t + 2)

        assert ctrl.telemetry.spontaneous_check_count == 3
        assert ctrl.telemetry.spontaneous_activated_count == 1

    def test_on_any_activity_increments_reset_counter(self, ctrl):
        """on_any_activity increments all_pathway_reset_count."""
        t = 1000.0
        for i in range(4):
            ctrl.on_any_activity(now=t + i)
        assert ctrl.telemetry.all_pathway_reset_count == 4

    def test_counters_in_diagnostics(self, ctrl):
        """Telemetry counters appear in diagnostics snapshot."""
        t = 1000.0
        ctrl.on_text_processed()
        ctrl.on_capture_result(frame_present=True, now=t)
        ctrl.on_spontaneous_result(activated=True, now=t + 1)
        ctrl.on_any_activity(now=t + 2)

        diag = ctrl.get_diagnostics()
        assert "telemetry" in diag
        tel = diag["telemetry"]
        assert tel["text_processed_count"] == 1
        assert tel["capture_attempt_count"] == 1
        assert tel["capture_change_detected_count"] == 1
        assert tel["spontaneous_check_count"] == 1
        assert tel["spontaneous_activated_count"] == 1
        assert tel["all_pathway_reset_count"] == 1

    def test_counters_do_not_affect_control_logic(self, ctrl):
        """Telemetry counters have no influence on stage transitions."""
        t = 1000.0
        # Record many events to accumulate counters
        for i in range(100):
            ctrl.on_capture_result(frame_present=False, now=t + i)

        # Counter accumulated
        assert ctrl.telemetry.capture_attempt_count == 100

        # Reset via frame detected: stage goes to 0 regardless of counter value
        ctrl.on_capture_result(frame_present=True, now=t + 200)
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.telemetry.capture_attempt_count == 101

    def test_counter_reset(self):
        """TelemetryCounters.reset() zeros all counters."""
        tc = TelemetryCounters(
            text_processed_count=100,
            capture_attempt_count=200,
            capture_change_detected_count=50,
            spontaneous_check_count=30,
            spontaneous_activated_count=5,
            all_pathway_reset_count=10,
        )
        tc.reset()
        assert tc.text_processed_count == 0
        assert tc.capture_attempt_count == 0
        assert tc.capture_change_detected_count == 0
        assert tc.spontaneous_check_count == 0
        assert tc.spontaneous_activated_count == 0
        assert tc.all_pathway_reset_count == 0

    def test_counter_to_dict(self):
        """TelemetryCounters.to_dict() returns correct dictionary."""
        tc = TelemetryCounters(
            text_processed_count=10,
            capture_attempt_count=20,
            capture_change_detected_count=5,
            spontaneous_check_count=3,
            spontaneous_activated_count=1,
            all_pathway_reset_count=2,
        )
        d = tc.to_dict()
        assert d == {
            "text_processed_count": 10,
            "capture_attempt_count": 20,
            "capture_change_detected_count": 5,
            "spontaneous_check_count": 3,
            "spontaneous_activated_count": 1,
            "all_pathway_reset_count": 2,
        }

    def test_counter_overflow_halves_all(self):
        """When any counter exceeds max, all counters are halved."""
        tc = TelemetryCounters(
            text_processed_count=_TELEMETRY_COUNTER_MAX + 1,
            capture_attempt_count=1000,
            capture_change_detected_count=500,
            spontaneous_check_count=100,
            spontaneous_activated_count=50,
            all_pathway_reset_count=10,
        )
        tc._check_overflow()

        # All halved
        assert tc.text_processed_count == (_TELEMETRY_COUNTER_MAX + 1) // 2
        assert tc.capture_attempt_count == 500
        assert tc.capture_change_detected_count == 250
        assert tc.spontaneous_check_count == 50
        assert tc.spontaneous_activated_count == 25
        assert tc.all_pathway_reset_count == 5

    def test_counter_overflow_via_controller(self, ctrl):
        """Overflow protection works through controller methods."""
        # Directly set counter near max
        ctrl._telemetry.capture_attempt_count = _TELEMETRY_COUNTER_MAX
        ctrl._telemetry.text_processed_count = 1000

        # This should trigger overflow check
        ctrl.on_capture_result(frame_present=False, now=1000.0)

        # After increment + overflow check, counters should be halved
        assert ctrl.telemetry.capture_attempt_count < _TELEMETRY_COUNTER_MAX
        assert ctrl.telemetry.text_processed_count == 500

    def test_counter_preserves_ratio_after_halving(self):
        """Halving preserves approximate ratios between counters."""
        tc = TelemetryCounters(
            text_processed_count=_TELEMETRY_COUNTER_MAX + 1,
            capture_attempt_count=2000,
            spontaneous_check_count=1000,
        )
        # Ratios: capture:spontaneous = 2:1
        tc._check_overflow()
        # After halving: 1000:500 = 2:1 (preserved)
        assert tc.capture_attempt_count == 1000
        assert tc.spontaneous_check_count == 500

    def test_mixed_events_telemetry_accuracy(self, ctrl):
        """Complex mixed event sequence produces accurate telemetry."""
        t = 1000.0

        # 10 text inputs
        for _ in range(10):
            ctrl.on_text_processed()

        # 20 captures (12 no-change, 8 change)
        for i in range(20):
            frame = i % 5 < 3  # True, True, True, False, False pattern
            ctrl.on_capture_result(frame_present=frame, now=t + i)

        # 5 spontaneous checks (2 activated, 3 not)
        for i in range(5):
            activated = i < 2
            ctrl.on_spontaneous_result(activated=activated, now=t + 50 + i)

        # 3 full resets
        for i in range(3):
            ctrl.on_any_activity(now=t + 100 + i)

        tel = ctrl.telemetry
        assert tel.text_processed_count == 10
        assert tel.capture_attempt_count == 20
        assert tel.capture_change_detected_count == 12  # 3 out of every 5 pattern
        assert tel.spontaneous_check_count == 5
        assert tel.spontaneous_activated_count == 2
        assert tel.all_pathway_reset_count == 3


# =========================================================================
# H. Telemetry Non-Interference Verification
# =========================================================================

class TestTelemetryNonInterference:
    """Verify telemetry counters never influence control decisions."""

    def test_should_attempt_capture_ignores_telemetry(self, ctrl):
        """should_attempt_capture result is independent of telemetry values."""
        t = 1000.0
        ctrl._capture_state.last_capture_time = t

        # Record result without telemetry
        result_before = ctrl.should_attempt_capture(now=t + 0.05)

        # Artificially inflate telemetry
        ctrl._telemetry.capture_attempt_count = 999999
        ctrl._telemetry.capture_change_detected_count = 500000

        # Same query gives same result
        result_after = ctrl.should_attempt_capture(now=t + 0.05)
        assert result_before == result_after

    def test_should_check_spontaneous_ignores_telemetry(self, ctrl):
        """should_check_spontaneous result is independent of telemetry values."""
        t = 1000.0
        ctrl._spontaneous_state.last_check_time = t - 100

        result_before = ctrl.should_check_spontaneous(
            idle_seconds=50.0, now=t,
        )

        # Inflate telemetry
        ctrl._telemetry.spontaneous_check_count = 999999
        ctrl._telemetry.spontaneous_activated_count = 500000

        result_after = ctrl.should_check_spontaneous(
            idle_seconds=50.0, now=t,
        )
        assert result_before == result_after

    def test_should_force_capture_ignores_telemetry(self, ctrl):
        """should_force_capture_after_text is independent of telemetry values."""
        result_before = ctrl.should_force_capture_after_text()

        ctrl._telemetry.text_processed_count = 999999
        result_after = ctrl.should_force_capture_after_text()
        assert result_before == result_after

    def test_stage_transitions_identical_with_different_telemetry(self):
        """Two controllers with different telemetry produce identical stage transitions."""
        ctrl1 = LoopIntervalController(
            capture_stages=[0.1, 0.5, 1.0],
            spontaneous_stages=[0.0, 5.0],
            text_consecutive_limit=3,
            base_idle_threshold=10.0,
        )
        ctrl2 = LoopIntervalController(
            capture_stages=[0.1, 0.5, 1.0],
            spontaneous_stages=[0.0, 5.0],
            text_consecutive_limit=3,
            base_idle_threshold=10.0,
        )

        # Inflate ctrl2 telemetry
        ctrl2._telemetry.capture_attempt_count = 1000000
        ctrl2._telemetry.text_processed_count = 500000

        t = 1000.0
        events = [
            ("capture", False),
            ("capture", False),
            ("text", None),
            ("capture", True),
            ("spontaneous", False),
            ("spontaneous", True),
            ("reset", None),
        ]

        for i, (event_type, value) in enumerate(events):
            t_now = t + i
            if event_type == "capture":
                ctrl1.on_capture_result(frame_present=value, now=t_now)
                ctrl2.on_capture_result(frame_present=value, now=t_now)
            elif event_type == "text":
                ctrl1.on_text_processed()
                ctrl2.on_text_processed()
            elif event_type == "spontaneous":
                ctrl1.on_spontaneous_result(activated=value, now=t_now)
                ctrl2.on_spontaneous_result(activated=value, now=t_now)
            elif event_type == "reset":
                ctrl1.on_any_activity(now=t_now)
                ctrl2.on_any_activity(now=t_now)

            # States should be identical
            assert ctrl1.capture_state.current_stage == ctrl2.capture_state.current_stage
            assert ctrl1.spontaneous_state.additional_wait_stage == ctrl2.spontaneous_state.additional_wait_stage
            assert ctrl1.text_capture_coord.consecutive_text_count == ctrl2.text_capture_coord.consecutive_text_count
