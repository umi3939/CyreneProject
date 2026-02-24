"""
test_main_control_loop.py - Tests for the main loop 3-path interval controller.

Tests cover:
A. Screen capture interval adaptive variation
B. Spontaneous activation check interval result-linked variation
C. Text input - screen capture coordination
D. Cross-pathway reset (safety valve)
E. Diagnostics snapshot
F. Edge cases and boundary conditions
"""

import time
import pytest

from loop_interval_controller import (
    LoopIntervalController,
    CaptureIntervalState,
    SpontaneousIntervalState,
    TextCaptureCoordState,
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
    """Controller with very short intervals for fast testing."""
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
# A. Screen Capture Interval - Adaptive Variation
# =========================================================================

class TestCaptureIntervalAdaptive:
    """Test A: Capture interval extends on no-change, resets on change."""

    def test_initial_stage_is_zero(self, ctrl):
        """At creation, capture stage starts at 0 (base interval)."""
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.capture_state.consecutive_no_change == 0

    def test_capture_interval_at_base(self, ctrl):
        """Base interval is the first entry in CAPTURE_INTERVAL_STAGES."""
        assert ctrl.get_current_capture_interval() == CAPTURE_INTERVAL_STAGES[0]

    def test_no_change_advances_stage(self, ctrl):
        """Each 'no change' result advances the capture stage by 1."""
        now = time.monotonic()
        ctrl.on_capture_result(frame_present=False, now=now)
        assert ctrl.capture_state.current_stage == 1
        assert ctrl.capture_state.consecutive_no_change == 1

        ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert ctrl.capture_state.current_stage == 2
        assert ctrl.capture_state.consecutive_no_change == 2

    def test_no_change_stage_clamped_at_max(self, simple_ctrl):
        """Stage does not exceed the number of defined stages."""
        now = time.monotonic()
        max_stage = len(simple_ctrl.capture_stages) - 1

        # Advance past max
        for i in range(10):
            simple_ctrl.on_capture_result(frame_present=False, now=now + i)

        assert simple_ctrl.capture_state.current_stage == max_stage
        assert simple_ctrl.get_current_capture_interval() == simple_ctrl.capture_stages[-1]

    def test_change_detected_resets_stage(self, ctrl):
        """When a frame is captured (change detected), stage resets to 0."""
        now = time.monotonic()
        # Advance stage
        ctrl.on_capture_result(frame_present=False, now=now)
        ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert ctrl.capture_state.current_stage == 2

        # Change detected
        ctrl.on_capture_result(frame_present=True, now=now + 2)
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.capture_state.consecutive_no_change == 0

    def test_should_attempt_capture_timing(self, simple_ctrl):
        """should_attempt_capture respects the current interval stage."""
        now = time.monotonic()
        simple_ctrl._capture_state.last_capture_time = now

        # At stage 0, interval is 0.1s
        assert simple_ctrl.get_current_capture_interval() == 0.1
        # Too soon
        assert not simple_ctrl.should_attempt_capture(now=now + 0.05)
        # Enough time (slightly past threshold to avoid float precision issues)
        assert simple_ctrl.should_attempt_capture(now=now + 0.11)
        assert simple_ctrl.should_attempt_capture(now=now + 1.0)

    def test_should_attempt_capture_at_higher_stage(self, simple_ctrl):
        """At higher stages, more time must pass before next capture."""
        now = time.monotonic()
        # Advance to stage 2 (interval = 1.0)
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 0.5)
        assert simple_ctrl.capture_state.current_stage == 2
        assert simple_ctrl.get_current_capture_interval() == 1.0

        last_time = simple_ctrl.capture_state.last_capture_time
        # 0.5s since last capture: not yet
        assert not simple_ctrl.should_attempt_capture(now=last_time + 0.5)
        # 1.0s since last capture: ready
        assert simple_ctrl.should_attempt_capture(now=last_time + 1.0)

    def test_interval_increases_monotonically(self, ctrl):
        """Each successive no-change increases the interval."""
        now = time.monotonic()
        prev_interval = ctrl.get_current_capture_interval()
        for i in range(len(CAPTURE_INTERVAL_STAGES) - 1):
            ctrl.on_capture_result(frame_present=False, now=now + i)
            new_interval = ctrl.get_current_capture_interval()
            assert new_interval >= prev_interval
            prev_interval = new_interval

    def test_upper_bound_never_exceeded(self, ctrl):
        """The capture interval never exceeds the last stage value."""
        now = time.monotonic()
        max_interval = CAPTURE_INTERVAL_STAGES[-1]
        for i in range(100):
            ctrl.on_capture_result(frame_present=False, now=now + i)
            assert ctrl.get_current_capture_interval() <= max_interval

    def test_text_processing_resets_capture_on_next_loop(self, simple_ctrl):
        """After text processing, next loop resets capture interval."""
        now = time.monotonic()
        # Advance capture stage
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert simple_ctrl.capture_state.current_stage == 2

        # Process text
        simple_ctrl.on_text_processed()

        # Next loop: check_and_apply should reset
        reset_applied = simple_ctrl.check_and_apply_text_capture_reset(
            now=now + 2,
        )
        assert reset_applied
        assert simple_ctrl.capture_state.current_stage == 0

    def test_capture_last_time_updates(self, ctrl):
        """Last capture time updates on each capture result."""
        t1 = time.monotonic()
        ctrl.on_capture_result(frame_present=True, now=t1)
        assert ctrl.capture_state.last_capture_time == t1

        t2 = t1 + 5.0
        ctrl.on_capture_result(frame_present=False, now=t2)
        assert ctrl.capture_state.last_capture_time == t2


# =========================================================================
# B. Spontaneous Activation Check Interval - Result-Linked
# =========================================================================

class TestSpontaneousIntervalLinked:
    """Test B: Spontaneous check interval varies based on check result."""

    def test_initial_state(self, ctrl):
        """Initial spontaneous state: stage 0, no prior result."""
        assert ctrl.spontaneous_state.additional_wait_stage == 0
        assert ctrl.spontaneous_state.last_check_result_activated is False

    def test_effective_threshold_at_base(self, ctrl):
        """At base stage, effective threshold equals base idle threshold."""
        assert ctrl.get_effective_idle_threshold() == ctrl.base_idle_threshold

    def test_not_activated_increases_wait(self, ctrl):
        """When check returns 'not activated', additional wait increases."""
        now = time.monotonic()
        ctrl.on_spontaneous_result(activated=False, now=now)
        assert ctrl.spontaneous_state.additional_wait_stage == 1
        # Effective threshold should now be higher
        assert ctrl.get_effective_idle_threshold() > ctrl.base_idle_threshold

    def test_activated_resets_wait(self, ctrl):
        """When check returns 'activated', wait resets to stage 0."""
        now = time.monotonic()
        # Advance stage
        ctrl.on_spontaneous_result(activated=False, now=now)
        ctrl.on_spontaneous_result(activated=False, now=now + 1)
        assert ctrl.spontaneous_state.additional_wait_stage == 2

        # Activation occurs
        ctrl.on_spontaneous_result(activated=True, now=now + 2)
        assert ctrl.spontaneous_state.additional_wait_stage == 0
        assert ctrl.get_effective_idle_threshold() == ctrl.base_idle_threshold

    def test_wait_stage_clamped_at_max(self, simple_ctrl):
        """Additional wait stage doesn't exceed max stages."""
        now = time.monotonic()
        max_stage = len(simple_ctrl.spontaneous_stages) - 1

        for i in range(20):
            simple_ctrl.on_spontaneous_result(activated=False, now=now + i)

        assert simple_ctrl.spontaneous_state.additional_wait_stage == max_stage

    def test_should_check_spontaneous_base(self, simple_ctrl):
        """At base stage, check triggers when idle exceeds base threshold."""
        now = time.monotonic()
        simple_ctrl._spontaneous_state.last_check_time = now - 100

        # Idle < threshold
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=5.0, now=now,
        )
        # Idle >= threshold (10.0)
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=10.0, now=now,
        )

    def test_should_check_spontaneous_with_additional_wait(self, simple_ctrl):
        """After a 'not activated' result, effective threshold increases."""
        now = time.monotonic()
        simple_ctrl._spontaneous_state.last_check_time = now - 100

        # Advance to stage 1 (additional wait = 5.0)
        simple_ctrl.on_spontaneous_result(activated=False, now=now)

        # Effective threshold = 10.0 + 5.0 = 15.0
        assert simple_ctrl.get_effective_idle_threshold() == 15.0

        # Idle = 12.0 < 15.0: not yet
        # Need to reset last_check_time so the time-since-last-check passes too
        simple_ctrl._spontaneous_state.last_check_time = now
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=12.0, now=now + 10,
        )

        # Idle = 15.0: should check
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=15.0, now=now + 10,
        )

    def test_effective_threshold_upper_bound(self, ctrl):
        """Effective threshold has an upper bound from stage definitions."""
        now = time.monotonic()
        max_additional = SPONTANEOUS_WAIT_STAGES[-1]

        for i in range(100):
            ctrl.on_spontaneous_result(activated=False, now=now + i)

        assert ctrl.get_effective_idle_threshold() == (
            ctrl.base_idle_threshold + max_additional
        )

    def test_last_check_time_updates(self, ctrl):
        """Last check time updates on each spontaneous result."""
        t1 = time.monotonic()
        ctrl.on_spontaneous_result(activated=True, now=t1)
        assert ctrl.spontaneous_state.last_check_time == t1

        t2 = t1 + 10.0
        ctrl.on_spontaneous_result(activated=False, now=t2)
        assert ctrl.spontaneous_state.last_check_time == t2

    def test_should_check_respects_recheck_interval(self, simple_ctrl):
        """After 'not activated', minimum recheck interval is enforced."""
        now = time.monotonic()
        # Stage 0, check once
        simple_ctrl.on_spontaneous_result(activated=False, now=now)
        # Now at stage 1, additional_wait = 5.0

        # Immediately after (0.1s), even if idle is enough, recheck too soon
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=20.0, now=now + 0.1,
        )

        # After additional_wait time passes
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=20.0, now=now + 5.0,
        )


# =========================================================================
# C. Text Input - Screen Capture Coordination
# =========================================================================

class TestTextCaptureCoordination:
    """Test C: Text input processing coordinates with screen capture."""

    def test_initial_coord_state(self, ctrl):
        """Initially no text is processed, no consecutive count."""
        assert ctrl.text_capture_coord.text_just_processed is False
        assert ctrl.text_capture_coord.consecutive_text_count == 0

    def test_on_text_processed_sets_flag(self, ctrl):
        """Processing text sets the flag for next-loop reset."""
        ctrl.on_text_processed()
        assert ctrl.text_capture_coord.text_just_processed is True
        assert ctrl.text_capture_coord.consecutive_text_count == 1

    def test_consecutive_text_increments(self, ctrl):
        """Each text processing increments consecutive count."""
        ctrl.on_text_processed()
        ctrl.on_text_processed()
        ctrl.on_text_processed()
        assert ctrl.text_capture_coord.consecutive_text_count == 3

    def test_force_capture_below_limit(self, simple_ctrl):
        """Below the limit, force capture is not triggered."""
        # limit is 2
        simple_ctrl.on_text_processed()
        assert not simple_ctrl.should_force_capture_after_text()

    def test_force_capture_at_limit(self, simple_ctrl):
        """At the limit, force capture is triggered."""
        # limit is 2
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

    def test_force_capture_above_limit(self, simple_ctrl):
        """Above the limit, force capture remains triggered."""
        for _ in range(5):
            simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

    def test_on_capture_after_text_resets_counter(self, simple_ctrl):
        """After a forced capture, the consecutive counter resets."""
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

        now = time.monotonic()
        simple_ctrl.on_capture_after_text(now=now)
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0
        assert not simple_ctrl.should_force_capture_after_text()

    def test_check_and_apply_reset_clears_flag(self, simple_ctrl):
        """check_and_apply_text_capture_reset clears the flag and resets stage."""
        now = time.monotonic()
        # Advance capture stage
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert simple_ctrl.capture_state.current_stage == 2

        # Process text
        simple_ctrl.on_text_processed()
        assert simple_ctrl.text_capture_coord.text_just_processed is True

        # Apply reset
        result = simple_ctrl.check_and_apply_text_capture_reset(now=now + 2)
        assert result is True
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.text_capture_coord.text_just_processed is False

    def test_check_and_apply_no_reset_when_no_text(self, simple_ctrl):
        """If no text was processed, no reset is applied."""
        result = simple_ctrl.check_and_apply_text_capture_reset()
        assert result is False

    def test_text_flag_consumed_once(self, ctrl):
        """The text_just_processed flag is consumed on first check_and_apply."""
        ctrl.on_text_processed()

        now = time.monotonic()
        assert ctrl.check_and_apply_text_capture_reset(now=now)
        # Second call: already consumed
        assert not ctrl.check_and_apply_text_capture_reset(now=now + 1)

    def test_text_processing_does_not_directly_reset_capture(self, ctrl):
        """on_text_processed only sets flag; actual reset is deferred."""
        now = time.monotonic()
        ctrl.on_capture_result(frame_present=False, now=now)
        old_stage = ctrl.capture_state.current_stage

        ctrl.on_text_processed()
        # Stage should NOT change yet
        assert ctrl.capture_state.current_stage == old_stage


# =========================================================================
# D. Cross-Pathway Reset (Safety Valve)
# =========================================================================

class TestCrossPathwayReset:
    """Test D: Activity on any pathway resets all interval states."""

    def test_on_any_activity_resets_capture(self, simple_ctrl):
        """on_any_activity resets capture state to base."""
        now = time.monotonic()
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert simple_ctrl.capture_state.current_stage == 2

        simple_ctrl.on_any_activity(now=now + 2)
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.capture_state.consecutive_no_change == 0

    def test_on_any_activity_resets_spontaneous(self, simple_ctrl):
        """on_any_activity resets spontaneous state to base."""
        now = time.monotonic()
        simple_ctrl.on_spontaneous_result(activated=False, now=now)
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 1

        simple_ctrl.on_any_activity(now=now + 1)
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0

    def test_on_any_activity_resets_text_coord(self, simple_ctrl):
        """on_any_activity resets text-capture coordination."""
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 2

        now = time.monotonic()
        simple_ctrl.on_any_activity(now=now)
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0
        assert simple_ctrl.text_capture_coord.text_just_processed is False

    def test_on_any_activity_resets_all_simultaneously(self, simple_ctrl):
        """All three states reset atomically on any activity."""
        now = time.monotonic()
        # Advance all states
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_spontaneous_result(activated=False, now=now)
        simple_ctrl.on_text_processed()

        assert simple_ctrl.capture_state.current_stage > 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage > 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count > 0

        simple_ctrl.on_any_activity(now=now + 1)

        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

    def test_on_any_activity_updates_times(self, ctrl):
        """on_any_activity updates the last capture and check times."""
        t = time.monotonic() + 100
        ctrl.on_any_activity(now=t)
        assert ctrl.capture_state.last_capture_time == t
        assert ctrl.spontaneous_state.last_check_time == t


# =========================================================================
# E. Diagnostics Snapshot
# =========================================================================

class TestDiagnostics:
    """Test E: Diagnostic snapshot returns correct state."""

    def test_diagnostics_initial(self, ctrl):
        """Initial diagnostics reflect base state."""
        diag = ctrl.get_diagnostics()
        assert diag["capture"]["current_stage"] == 0
        assert diag["capture"]["current_interval"] == CAPTURE_INTERVAL_STAGES[0]
        assert diag["capture"]["consecutive_no_change"] == 0
        assert diag["spontaneous"]["additional_wait_stage"] == 0
        assert diag["spontaneous"]["effective_threshold"] == ctrl.base_idle_threshold
        assert diag["spontaneous"]["last_result_activated"] is False
        assert diag["text_capture_coord"]["text_just_processed"] is False
        assert diag["text_capture_coord"]["consecutive_text_count"] == 0
        assert diag["text_capture_coord"]["force_capture_needed"] is False

    def test_diagnostics_after_changes(self, simple_ctrl):
        """Diagnostics reflect state changes accurately."""
        now = time.monotonic()
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_spontaneous_result(activated=False, now=now)
        simple_ctrl.on_text_processed()
        simple_ctrl.on_text_processed()

        diag = simple_ctrl.get_diagnostics()
        assert diag["capture"]["current_stage"] == 1
        assert diag["capture"]["consecutive_no_change"] == 1
        assert diag["spontaneous"]["additional_wait_stage"] == 1
        assert diag["text_capture_coord"]["consecutive_text_count"] == 2
        assert diag["text_capture_coord"]["force_capture_needed"] is True

    def test_diagnostics_returns_dict(self, ctrl):
        """Diagnostics returns a dictionary with expected keys."""
        diag = ctrl.get_diagnostics()
        assert isinstance(diag, dict)
        assert "capture" in diag
        assert "spontaneous" in diag
        assert "text_capture_coord" in diag


# =========================================================================
# F. Edge Cases and Boundary Conditions
# =========================================================================

class TestEdgeCases:
    """Test F: Boundary conditions and edge cases."""

    def test_empty_capture_stages_gets_default(self):
        """Empty capture stages list gets a default single stage."""
        ctrl = LoopIntervalController(capture_stages=[])
        assert len(ctrl.capture_stages) == 1
        assert ctrl.capture_stages[0] == 0.1

    def test_empty_spontaneous_stages_gets_default(self):
        """Empty spontaneous stages list gets a default single stage."""
        ctrl = LoopIntervalController(spontaneous_stages=[])
        assert len(ctrl.spontaneous_stages) == 1
        assert ctrl.spontaneous_stages[0] == 0.0

    def test_text_limit_below_one_gets_clamped(self):
        """Text consecutive limit < 1 is clamped to 1."""
        ctrl = LoopIntervalController(text_consecutive_limit=0)
        assert ctrl.text_consecutive_limit == 1

        ctrl2 = LoopIntervalController(text_consecutive_limit=-5)
        assert ctrl2.text_consecutive_limit == 1

    def test_single_stage_capture(self):
        """With a single capture stage, it stays at that stage."""
        ctrl = LoopIntervalController(capture_stages=[0.5])
        assert ctrl.get_current_capture_interval() == 0.5

        now = time.monotonic()
        ctrl.on_capture_result(frame_present=False, now=now)
        # Still at stage 0 (max is 0)
        assert ctrl.capture_state.current_stage == 0
        assert ctrl.get_current_capture_interval() == 0.5

    def test_single_stage_spontaneous(self):
        """With a single spontaneous stage, wait stays constant."""
        ctrl = LoopIntervalController(
            spontaneous_stages=[3.0],
            base_idle_threshold=10.0,
        )
        assert ctrl.get_effective_idle_threshold() == 13.0

        now = time.monotonic()
        ctrl.on_spontaneous_result(activated=False, now=now)
        # Still at stage 0
        assert ctrl.spontaneous_state.additional_wait_stage == 0
        assert ctrl.get_effective_idle_threshold() == 13.0

    def test_rapid_toggle_capture_results(self, ctrl):
        """Rapidly toggling between frame_present=True/False is stable."""
        now = time.monotonic()
        for i in range(50):
            frame_present = (i % 2 == 0)
            ctrl.on_capture_result(frame_present=frame_present, now=now + i * 0.1)

        # After ending on False (i=49 is odd -> False), stage should be 1
        # After ending on True (i=48 is even -> True then i=49 False -> stage 1)
        assert ctrl.capture_state.current_stage <= 1

    def test_rapid_toggle_spontaneous_results(self, ctrl):
        """Rapidly toggling spontaneous results is stable."""
        now = time.monotonic()
        for i in range(50):
            activated = (i % 2 == 0)
            ctrl.on_spontaneous_result(activated=activated, now=now + i)

        # After ending on False (i=49), stage should be 1
        assert ctrl.spontaneous_state.additional_wait_stage <= 1

    def test_no_monotonic_time_dependency_when_explicit(self, ctrl):
        """When explicit time is passed, time.monotonic() is not used."""
        # This tests that the controller works with explicit timestamps
        ctrl.on_capture_result(frame_present=False, now=1000.0)
        assert ctrl.capture_state.last_capture_time == 1000.0

        ctrl.on_spontaneous_result(activated=True, now=2000.0)
        assert ctrl.spontaneous_state.last_check_time == 2000.0

    def test_default_now_uses_monotonic(self, ctrl):
        """When no explicit time is passed, time.monotonic() is used."""
        before = time.monotonic()
        ctrl.on_capture_result(frame_present=True)
        after = time.monotonic()
        assert before <= ctrl.capture_state.last_capture_time <= after

    def test_properties_return_copies(self, ctrl):
        """Property accessors return copies, not references to internal state."""
        stages = ctrl.capture_stages
        stages.append(999.0)
        # Internal list should not be modified
        assert 999.0 not in ctrl.capture_stages

        sp_stages = ctrl.spontaneous_stages
        sp_stages.append(999.0)
        assert 999.0 not in ctrl.spontaneous_stages


# =========================================================================
# G. CaptureIntervalState Dataclass
# =========================================================================

class TestCaptureIntervalState:
    """Test the CaptureIntervalState dataclass directly."""

    def test_default_values(self):
        state = CaptureIntervalState()
        assert state.current_stage == 0
        assert state.last_capture_time == 0.0
        assert state.consecutive_no_change == 0

    def test_reset_with_time(self):
        state = CaptureIntervalState(
            current_stage=3, consecutive_no_change=5,
        )
        state.reset(now=100.0)
        assert state.current_stage == 0
        assert state.consecutive_no_change == 0
        assert state.last_capture_time == 100.0

    def test_reset_without_time(self):
        state = CaptureIntervalState(
            current_stage=3, last_capture_time=50.0,
            consecutive_no_change=5,
        )
        state.reset()
        assert state.current_stage == 0
        assert state.consecutive_no_change == 0
        # Time should not change when now=None
        assert state.last_capture_time == 50.0


# =========================================================================
# H. SpontaneousIntervalState Dataclass
# =========================================================================

class TestSpontaneousIntervalState:
    """Test the SpontaneousIntervalState dataclass directly."""

    def test_default_values(self):
        state = SpontaneousIntervalState()
        assert state.last_check_result_activated is False
        assert state.last_check_time == 0.0
        assert state.additional_wait_stage == 0

    def test_reset_with_time(self):
        state = SpontaneousIntervalState(
            additional_wait_stage=3,
            last_check_result_activated=True,
        )
        state.reset(now=200.0)
        assert state.additional_wait_stage == 0
        assert state.last_check_result_activated is False
        assert state.last_check_time == 200.0

    def test_reset_without_time(self):
        state = SpontaneousIntervalState(
            additional_wait_stage=3,
            last_check_time=50.0,
        )
        state.reset()
        assert state.additional_wait_stage == 0
        assert state.last_check_time == 50.0


# =========================================================================
# I. TextCaptureCoordState Dataclass
# =========================================================================

class TestTextCaptureCoordState:
    """Test the TextCaptureCoordState dataclass directly."""

    def test_default_values(self):
        state = TextCaptureCoordState()
        assert state.text_just_processed is False
        assert state.consecutive_text_count == 0

    def test_reset(self):
        state = TextCaptureCoordState(
            text_just_processed=True, consecutive_text_count=5,
        )
        state.reset()
        assert state.text_just_processed is False
        assert state.consecutive_text_count == 0


# =========================================================================
# J. Integration Scenarios
# =========================================================================

class TestIntegrationScenarios:
    """Test multi-step scenarios simulating real main loop behavior."""

    def test_scenario_idle_to_active_to_idle(self, simple_ctrl):
        """Simulate: idle period -> screen change -> idle period."""
        now = time.monotonic()

        # Phase 1: No change for several captures -> interval extends
        for i in range(3):
            simple_ctrl.on_capture_result(
                frame_present=False, now=now + i * 2,
            )
        max_stage = len(simple_ctrl.capture_stages) - 1
        assert simple_ctrl.capture_state.current_stage == max_stage

        # Phase 2: Screen change detected -> interval resets
        simple_ctrl.on_capture_result(frame_present=True, now=now + 10)
        assert simple_ctrl.capture_state.current_stage == 0

        # Phase 3: Activity -> all reset
        simple_ctrl.on_any_activity(now=now + 11)
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0

    def test_scenario_text_burst_then_capture(self, simple_ctrl):
        """Simulate: multiple text inputs followed by forced capture."""
        # limit is 2
        simple_ctrl.on_text_processed()
        assert not simple_ctrl.should_force_capture_after_text()

        simple_ctrl.on_text_processed()
        assert simple_ctrl.should_force_capture_after_text()

        # Forced capture occurs
        now = time.monotonic()
        simple_ctrl.on_capture_after_text(now=now)
        assert not simple_ctrl.should_force_capture_after_text()
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

    def test_scenario_spontaneous_backoff_then_activate(self, simple_ctrl):
        """Simulate: multiple failed spontaneous checks then activation."""
        now = time.monotonic()

        # Several non-activations -> wait increases
        for i in range(2):
            simple_ctrl.on_spontaneous_result(
                activated=False, now=now + i * 10,
            )

        max_stage = len(simple_ctrl.spontaneous_stages) - 1
        assert simple_ctrl.spontaneous_state.additional_wait_stage == max_stage

        # Activation occurs -> wait resets
        simple_ctrl.on_spontaneous_result(activated=True, now=now + 30)
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.spontaneous_state.last_check_result_activated is True

    def test_scenario_text_then_screen_reset_next_loop(self, simple_ctrl):
        """Simulate: text input -> next loop capture interval reset."""
        now = time.monotonic()

        # Advance capture interval
        simple_ctrl.on_capture_result(frame_present=False, now=now)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 1)
        assert simple_ctrl.capture_state.current_stage == 2

        # Text processed (simulating loop N)
        simple_ctrl.on_text_processed()

        # Next loop (N+1): apply reset
        simple_ctrl.check_and_apply_text_capture_reset(now=now + 2)
        assert simple_ctrl.capture_state.current_stage == 0

    def test_scenario_mixed_pathway_activity(self, simple_ctrl):
        """Simulate: alternating between text and screen pathways."""
        now = time.monotonic()

        # Loop 1: text input
        simple_ctrl.on_text_processed()
        simple_ctrl.on_any_activity(now=now)

        # Loop 2: reset from text, then capture (no change)
        simple_ctrl.check_and_apply_text_capture_reset(now=now + 0.5)
        simple_ctrl.on_capture_result(frame_present=False, now=now + 0.5)
        assert simple_ctrl.capture_state.current_stage == 1

        # Loop 3: capture (change found)
        simple_ctrl.on_capture_result(frame_present=True, now=now + 1.0)
        assert simple_ctrl.capture_state.current_stage == 0
        simple_ctrl.on_any_activity(now=now + 1.0)

        # Everything reset
        assert simple_ctrl.capture_state.current_stage == 0
        assert simple_ctrl.spontaneous_state.additional_wait_stage == 0
        assert simple_ctrl.text_capture_coord.consecutive_text_count == 0

    def test_scenario_all_pathways_quiet(self, simple_ctrl):
        """Simulate: no input on any pathway -> spontaneous eventually checked."""
        now = time.monotonic()
        simple_ctrl._spontaneous_state.last_check_time = now

        # Not enough idle time
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=5.0, now=now + 5,
        )

        # Enough idle time (threshold=10)
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=10.0, now=now + 15,
        )

        # Check result: not activated -> wait increases
        simple_ctrl.on_spontaneous_result(activated=False, now=now + 15)

        # Next check requires more idle time
        assert not simple_ctrl.should_check_spontaneous(
            idle_seconds=12.0, now=now + 20,
        )
        assert simple_ctrl.should_check_spontaneous(
            idle_seconds=15.0, now=now + 25,
        )


# =========================================================================
# K. Default Constants Validation
# =========================================================================

class TestDefaultConstants:
    """Test that default constants are sensible."""

    def test_capture_stages_ascending(self):
        """Capture interval stages are in ascending order."""
        for i in range(len(CAPTURE_INTERVAL_STAGES) - 1):
            assert CAPTURE_INTERVAL_STAGES[i] <= CAPTURE_INTERVAL_STAGES[i + 1]

    def test_capture_stages_positive(self):
        """All capture stages are positive."""
        for stage in CAPTURE_INTERVAL_STAGES:
            assert stage > 0

    def test_spontaneous_stages_ascending(self):
        """Spontaneous wait stages are in ascending order."""
        for i in range(len(SPONTANEOUS_WAIT_STAGES) - 1):
            assert SPONTANEOUS_WAIT_STAGES[i] <= SPONTANEOUS_WAIT_STAGES[i + 1]

    def test_spontaneous_stages_non_negative(self):
        """All spontaneous stages are non-negative."""
        for stage in SPONTANEOUS_WAIT_STAGES:
            assert stage >= 0

    def test_text_consecutive_limit_positive(self):
        """Text consecutive limit is positive."""
        assert TEXT_CONSECUTIVE_LIMIT > 0

    def test_capture_stages_has_multiple(self):
        """Default capture stages has multiple entries for gradual extension."""
        assert len(CAPTURE_INTERVAL_STAGES) >= 3

    def test_spontaneous_stages_starts_at_zero(self):
        """Default spontaneous stages starts at 0 (no additional wait)."""
        assert SPONTANEOUS_WAIT_STAGES[0] == 0.0


# =========================================================================
# L. Constructor Customization
# =========================================================================

class TestConstructorCustomization:
    """Test custom constructor parameters."""

    def test_custom_capture_stages(self):
        ctrl = LoopIntervalController(capture_stages=[1.0, 2.0, 4.0])
        assert ctrl.capture_stages == [1.0, 2.0, 4.0]
        assert ctrl.get_current_capture_interval() == 1.0

    def test_custom_spontaneous_stages(self):
        ctrl = LoopIntervalController(spontaneous_stages=[0.0, 10.0, 20.0])
        assert ctrl.spontaneous_stages == [0.0, 10.0, 20.0]

    def test_custom_text_limit(self):
        ctrl = LoopIntervalController(text_consecutive_limit=10)
        assert ctrl.text_consecutive_limit == 10

    def test_custom_base_idle_threshold(self):
        ctrl = LoopIntervalController(base_idle_threshold=60.0)
        assert ctrl.base_idle_threshold == 60.0
        assert ctrl.get_effective_idle_threshold() == 60.0

    def test_custom_stages_are_independent_copies(self):
        """Modifying the original list doesn't affect the controller."""
        stages = [1.0, 2.0, 3.0]
        ctrl = LoopIntervalController(capture_stages=stages)
        stages.append(99.0)
        assert len(ctrl.capture_stages) == 3

    def test_all_defaults(self):
        """Default constructor uses module-level constants."""
        ctrl = LoopIntervalController()
        assert ctrl.capture_stages == list(CAPTURE_INTERVAL_STAGES)
        assert ctrl.spontaneous_stages == list(SPONTANEOUS_WAIT_STAGES)
        assert ctrl.text_consecutive_limit == TEXT_CONSECUTIVE_LIMIT
        assert ctrl.base_idle_threshold == 30.0
