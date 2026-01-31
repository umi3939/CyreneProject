"""
tests/test_dynamics.py - Tests for emotional dynamics (peak & rebound)

Verifies:
1. Phase transitions (normal → peak → rebound → normal)
2. Configurable thresholds and durations
3. Decay modifiers per phase
4. Persistence through snapshot
5. No hardcoded values affecting behavior
"""

import time
from pathlib import Path

import pytest

from psyche.dynamics import (
    DynamicsPhase,
    DynamicsConfig,
    DynamicsState,
    create_dynamics_state,
    update_dynamics,
    check_peak_trigger,
    check_rebound_trigger,
    check_normal_trigger,
    enter_peak,
    enter_rebound,
    enter_normal,
    get_decay_modifier,
    get_intensity_modifier,
    get_dynamics_summary,
    apply_dynamics_to_decay,
)
from psyche.snapshot import (
    Snapshot,
    create_default_snapshot,
    validate_snapshot,
)
from psyche.persistence import PersistenceManager


class TestDynamicsState:
    """Tests for DynamicsState structure."""

    def test_create_default_state(self):
        """Default state starts in NORMAL phase."""
        state = create_dynamics_state()

        assert state.phase == DynamicsPhase.NORMAL
        assert state.peak_emotion == ""
        assert state.peak_intensity == 0.0
        assert state.accumulated_intensity == 0.0

    def test_state_serialization(self):
        """State survives serialization roundtrip."""
        state = create_dynamics_state()
        state = enter_peak(state, "joy", 0.8)

        data = state.to_dict()
        restored = DynamicsState.from_dict(data)

        assert restored.phase == DynamicsPhase.PEAK
        assert restored.peak_emotion == "joy"
        assert restored.peak_intensity == 0.8

    def test_phase_enum_values(self):
        """Phase enum has expected values."""
        assert DynamicsPhase.NORMAL.value == "normal"
        assert DynamicsPhase.PEAK.value == "peak"
        assert DynamicsPhase.REBOUND.value == "rebound"


class TestPhaseTransitions:
    """Tests for phase transition logic."""

    def test_normal_to_peak_on_intensity(self):
        """High intensity triggers peak transition."""
        config = DynamicsConfig(peak_intensity_threshold=0.7)
        state = create_dynamics_state(config)

        emotions = {"joy": 0.8, "sorrow": 0.1}
        should_peak, peak_emo, peak_int = check_peak_trigger(state, emotions)

        assert should_peak is True
        assert peak_emo == "joy"
        assert peak_int == 0.8

    def test_normal_to_peak_on_accumulation(self):
        """High accumulation triggers peak transition."""
        config = DynamicsConfig(
            peak_intensity_threshold=1.0,  # Won't trigger on intensity
            peak_accumulation_threshold=1.5,
        )
        state = create_dynamics_state(config)
        state.accumulated_intensity = 2.0  # Exceeds threshold

        emotions = {"joy": 0.5}
        should_peak, _, _ = check_peak_trigger(state, emotions)

        assert should_peak is True

    def test_no_peak_below_threshold(self):
        """Low intensity doesn't trigger peak."""
        config = DynamicsConfig(peak_intensity_threshold=0.9)
        state = create_dynamics_state(config)

        emotions = {"joy": 0.5, "sorrow": 0.3}
        should_peak, _, _ = check_peak_trigger(state, emotions)

        assert should_peak is False

    def test_peak_to_rebound_on_turns(self):
        """Peak transitions to rebound after configured turns."""
        config = DynamicsConfig(peak_duration_turns=3)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)
        state.phase_turn_count = 3

        assert check_rebound_trigger(state) is True

    def test_peak_to_rebound_on_time(self):
        """Peak transitions to rebound after configured time."""
        config = DynamicsConfig(peak_duration_seconds=1.0)
        state = create_dynamics_state(config)

        past_time = time.time() - 2.0  # 2 seconds ago
        state = enter_peak(state, "joy", 0.8, current_time=past_time)

        assert check_rebound_trigger(state) is True

    def test_rebound_to_normal_on_turns(self):
        """Rebound transitions to normal after configured turns."""
        config = DynamicsConfig(rebound_duration_turns=2)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)
        state = enter_rebound(state)
        state.phase_turn_count = 2

        assert check_normal_trigger(state) is True

    def test_full_cycle(self):
        """Complete cycle: normal → peak → rebound → normal."""
        config = DynamicsConfig(
            peak_intensity_threshold=0.7,
            peak_duration_turns=2,
            rebound_duration_turns=2,
        )
        state = create_dynamics_state(config)

        # Start in normal
        assert state.phase == DynamicsPhase.NORMAL

        # High intensity triggers peak
        emotions = {"joy": 0.9}
        state = update_dynamics(state, emotions)
        assert state.phase == DynamicsPhase.PEAK

        # Wait for peak duration
        state = update_dynamics(state, {"joy": 0.5})  # Turn 2
        assert state.phase == DynamicsPhase.PEAK

        state = update_dynamics(state, {"joy": 0.4})  # Turn 3 - should transition
        assert state.phase == DynamicsPhase.REBOUND

        # Wait for rebound duration
        state = update_dynamics(state, {"joy": 0.3})  # Turn 2 of rebound
        assert state.phase == DynamicsPhase.REBOUND

        state = update_dynamics(state, {"joy": 0.2})  # Turn 3 - should transition
        assert state.phase == DynamicsPhase.NORMAL


class TestDecayModifiers:
    """Tests for decay rate modification."""

    def test_normal_phase_no_modifier(self):
        """Normal phase has no decay modification."""
        state = create_dynamics_state()
        assert state.phase == DynamicsPhase.NORMAL

        modifier = get_decay_modifier(state)
        assert modifier == 1.0

    def test_peak_phase_modifier(self):
        """Peak phase applies configured decay modifier."""
        config = DynamicsConfig(peak_decay_multiplier=0.5)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)

        modifier = get_decay_modifier(state)
        assert modifier == 0.5

    def test_rebound_phase_modifier(self):
        """Rebound phase applies configured decay modifier."""
        config = DynamicsConfig(rebound_decay_multiplier=2.0)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)
        state = enter_rebound(state)

        modifier = get_decay_modifier(state)
        assert modifier == 2.0

    def test_apply_dynamics_to_decay(self):
        """Decay rate is correctly modified by dynamics."""
        config = DynamicsConfig(rebound_decay_multiplier=2.0)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)
        state = enter_rebound(state)

        base_rate = 0.95
        modified_rate = apply_dynamics_to_decay(base_rate, state)

        # With multiplier 2.0, decay should be faster (lower rate)
        assert modified_rate < base_rate


class TestIntensityModifiers:
    """Tests for intensity boost/dampening."""

    def test_peak_intensity_boost(self):
        """Peak phase provides intensity boost."""
        config = DynamicsConfig(peak_intensity_boost=0.2)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)

        boost, dampening = get_intensity_modifier(state)
        assert boost == 0.2
        assert dampening == 0.0

    def test_rebound_dampening(self):
        """Rebound phase provides dampening."""
        config = DynamicsConfig(rebound_dampening=0.3)
        state = create_dynamics_state(config)
        state = enter_peak(state, "joy", 0.8)
        state = enter_rebound(state)

        boost, dampening = get_intensity_modifier(state)
        assert boost == 0.0
        assert dampening == 0.3

    def test_normal_no_modifiers(self):
        """Normal phase has no intensity modifiers."""
        state = create_dynamics_state()

        boost, dampening = get_intensity_modifier(state)
        assert boost == 0.0
        assert dampening == 0.0


class TestAccumulation:
    """Tests for intensity accumulation tracking."""

    def test_accumulation_updates(self):
        """Intensity accumulates over updates."""
        # Use high thresholds to prevent accidental peak trigger
        config = DynamicsConfig(
            peak_intensity_threshold=10.0,
            peak_accumulation_threshold=10.0,
        )
        state = create_dynamics_state(config)

        state = update_dynamics(state, {"joy": 0.5})
        assert state.accumulated_intensity > 0
        assert state.phase == DynamicsPhase.NORMAL

        prev_accum = state.accumulated_intensity
        state = update_dynamics(state, {"joy": 0.5})
        assert state.accumulated_intensity > prev_accum
        assert state.phase == DynamicsPhase.NORMAL

    def test_accumulation_resets_on_peak(self):
        """Accumulation resets when entering peak."""
        config = DynamicsConfig(
            peak_intensity_threshold=0.9,
            peak_accumulation_threshold=10.0,  # High so only intensity triggers
        )
        state = create_dynamics_state(config)

        # Build up accumulation (won't trigger peak due to high threshold)
        state = update_dynamics(state, {"joy": 0.5})
        state = update_dynamics(state, {"joy": 0.5})
        assert state.phase == DynamicsPhase.NORMAL
        assert state.accumulated_intensity > 0

        # Trigger peak via intensity
        state = update_dynamics(state, {"joy": 0.95})
        assert state.phase == DynamicsPhase.PEAK
        assert state.accumulated_intensity == 0.0


class TestCustomTriggerFunctions:
    """Tests for injectable trigger functions."""

    def test_custom_peak_trigger(self):
        """Custom peak trigger function is used."""
        def custom_trigger(state, context):
            # Always trigger if any emotion > 0.3
            emotions = context.get("emotions", {})
            return any(v > 0.3 for v in emotions.values())

        config = DynamicsConfig(
            peak_intensity_threshold=1.0,  # Would never trigger normally
            peak_trigger_function=custom_trigger,
        )
        state = create_dynamics_state(config)

        emotions = {"joy": 0.4}  # Below normal threshold but above custom
        should_peak, _, _ = check_peak_trigger(state, emotions)

        assert should_peak is True


class TestSnapshotIntegration:
    """Tests for dynamics persistence in snapshots."""

    def test_snapshot_includes_dynamics(self):
        """Snapshot contains dynamics state."""
        snapshot = create_default_snapshot()

        assert hasattr(snapshot, "dynamics")
        assert snapshot.dynamics.phase == DynamicsPhase.NORMAL

    def test_dynamics_survives_snapshot_roundtrip(self):
        """Dynamics state survives serialization."""
        snapshot = create_default_snapshot()

        # Modify dynamics
        snapshot.dynamics = enter_peak(snapshot.dynamics, "anger", 0.7)

        # Serialize and deserialize
        data = snapshot.to_dict()
        restored = Snapshot.from_dict(data)

        assert restored.dynamics.phase == DynamicsPhase.PEAK
        assert restored.dynamics.peak_emotion == "anger"
        assert restored.dynamics.peak_intensity == 0.7

    def test_dynamics_persists_to_disk(self, tmp_path: Path):
        """Dynamics state survives disk persistence."""
        mgr = PersistenceManager(directory=tmp_path)

        # Create snapshot with peak state
        snapshot = create_default_snapshot()
        snapshot.dynamics = enter_peak(snapshot.dynamics, "fear", 0.6)
        snapshot.dynamics = enter_rebound(snapshot.dynamics)
        mgr.save(snapshot)

        # Load and verify
        loaded = mgr.load()
        assert loaded.dynamics.phase == DynamicsPhase.REBOUND
        assert loaded.dynamics.peak_emotion == "fear"

    def test_snapshot_validates_dynamics(self):
        """Snapshot validation includes dynamics check."""
        snapshot = create_default_snapshot()

        is_valid, issues = validate_snapshot(snapshot)
        assert is_valid is True

    def test_v1_snapshot_gets_default_dynamics(self):
        """Loading v1 snapshot (without dynamics) creates default."""
        # Simulate v1 snapshot data (no dynamics field)
        v1_data = {
            "version": 1,
            "user_id": "test",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "psyche": {},
            "loop": {"memory": {}, "config": {}},
            "responsibility": {},
            # No "dynamics" field
        }

        snapshot = Snapshot.from_dict(v1_data)

        assert snapshot is not None
        assert snapshot.dynamics.phase == DynamicsPhase.NORMAL


class TestDiagnostics:
    """Tests for diagnostic functions."""

    def test_summary_normal(self):
        """Summary describes normal state."""
        state = create_dynamics_state()
        summary = get_dynamics_summary(state)

        assert "NORMAL" in summary

    def test_summary_peak(self):
        """Summary describes peak state."""
        state = create_dynamics_state()
        state = enter_peak(state, "joy", 0.8)
        summary = get_dynamics_summary(state)

        assert "PEAK" in summary
        assert "joy" in summary

    def test_summary_rebound(self):
        """Summary describes rebound state."""
        state = create_dynamics_state()
        state = enter_peak(state, "joy", 0.8)
        state = enter_rebound(state)
        summary = get_dynamics_summary(state)

        assert "REBOUND" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
