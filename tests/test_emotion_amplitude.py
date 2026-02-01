"""
tests/test_emotion_amplitude.py - Tests for emotion amplitude expansion

Verifies:
1. Amplitude scales delta (change amount), not absolute value
2. Amplitude preserves sign (direction) of emotion updates
3. Amplitude is temporary and decays over time
4. No new emotion types are added
5. Integration with reaction system
"""

import pytest

from psyche.emotion_amplitude import (
    AmplitudeState,
    AmplitudeConfig,
    apply_amplitude_to_delta,
    apply_amplitude_to_emotion_deltas,
    update_amplitude,
    decay_amplitude,
    compute_amplitude_from_dynamics,
    compute_amplitude_from_residue,
    create_amplitude_state,
    get_amplitude_summary,
)
from psyche.reaction import react
from psyche.state import PsycheState, Percept, EmotionVector
from psyche.dynamics import create_dynamics_state, enter_peak, enter_rebound


class TestApplyAmplitudeToDelta:
    """Tests for apply_amplitude_to_delta function."""

    def test_neutral_amplitude_no_change(self):
        """Amplitude of 1.0 does not change delta."""
        delta = apply_amplitude_to_delta(0.2, amplitude=1.0)
        assert delta == pytest.approx(0.2)

    def test_amplify_positive_delta(self):
        """Amplitude > 1 increases positive delta."""
        delta = apply_amplitude_to_delta(0.2, amplitude=1.5)
        assert delta == pytest.approx(0.3)

    def test_amplify_negative_delta(self):
        """Amplitude > 1 increases magnitude of negative delta."""
        delta = apply_amplitude_to_delta(-0.2, amplitude=1.5)
        assert delta == pytest.approx(-0.3)

    def test_dampen_positive_delta(self):
        """Amplitude < 1 decreases positive delta."""
        delta = apply_amplitude_to_delta(0.2, amplitude=0.5)
        assert delta == pytest.approx(0.1)

    def test_dampen_negative_delta(self):
        """Amplitude < 1 decreases magnitude of negative delta."""
        delta = apply_amplitude_to_delta(-0.2, amplitude=0.5)
        assert delta == pytest.approx(-0.1)

    def test_preserves_sign_positive(self):
        """Positive delta remains positive regardless of amplitude."""
        delta = apply_amplitude_to_delta(0.1, amplitude=2.0)
        assert delta > 0

    def test_preserves_sign_negative(self):
        """Negative delta remains negative regardless of amplitude."""
        delta = apply_amplitude_to_delta(-0.1, amplitude=2.0)
        assert delta < 0

    def test_zero_delta_unchanged(self):
        """Zero delta remains zero."""
        delta = apply_amplitude_to_delta(0.0, amplitude=2.0)
        assert delta == 0.0

    def test_amplitude_clamped_to_max(self):
        """Amplitude is clamped to max."""
        config = AmplitudeConfig(max_amplitude=1.5)
        delta = apply_amplitude_to_delta(0.2, amplitude=3.0, config=config)
        assert delta == pytest.approx(0.3)  # 0.2 * 1.5

    def test_amplitude_clamped_to_min(self):
        """Amplitude is clamped to min."""
        config = AmplitudeConfig(min_amplitude=0.5)
        delta = apply_amplitude_to_delta(0.2, amplitude=0.1, config=config)
        assert delta == pytest.approx(0.1)  # 0.2 * 0.5


class TestApplyAmplitudeToEmotionDeltas:
    """Tests for apply_amplitude_to_emotion_deltas function."""

    def test_apply_to_multiple_deltas(self):
        """Amplitude is applied to all deltas."""
        deltas = {"joy": 0.2, "sorrow": 0.1, "anger": -0.1}
        result = apply_amplitude_to_emotion_deltas(deltas, amplitude=2.0)

        assert result["joy"] == pytest.approx(0.4)
        assert result["sorrow"] == pytest.approx(0.2)
        assert result["anger"] == pytest.approx(-0.2)

    def test_preserves_all_signs(self):
        """All signs are preserved."""
        deltas = {"joy": 0.1, "anger": -0.2}
        result = apply_amplitude_to_emotion_deltas(deltas, amplitude=1.5)

        assert result["joy"] > 0
        assert result["anger"] < 0


class TestAmplitudeState:
    """Tests for AmplitudeState structure."""

    def test_create_default_state(self):
        """Default state has neutral amplitude."""
        state = create_amplitude_state()

        assert state.current_amplitude == 1.0
        assert state.accumulated_boost == 0.0
        assert state.update_count == 0

    def test_state_serialization(self):
        """State survives serialization roundtrip."""
        state = AmplitudeState(
            current_amplitude=1.5,
            accumulated_boost=0.3,
            update_count=5,
        )

        data = state.to_dict()
        restored = AmplitudeState.from_dict(data)

        assert restored.current_amplitude == state.current_amplitude
        assert restored.accumulated_boost == state.accumulated_boost
        assert restored.update_count == state.update_count


class TestUpdateAmplitude:
    """Tests for update_amplitude function."""

    def test_update_increases_amplitude(self):
        """Positive intensity increases amplitude."""
        state = create_amplitude_state()
        updated = update_amplitude(state, intensity_factor=0.5)

        assert updated.current_amplitude > state.current_amplitude

    def test_update_increments_count(self):
        """Update increments update count."""
        state = create_amplitude_state()
        updated = update_amplitude(state, intensity_factor=0.1)

        assert updated.update_count == state.update_count + 1

    def test_update_accumulates_boost(self):
        """Update accumulates boost."""
        state = create_amplitude_state()
        updated = update_amplitude(state, intensity_factor=0.5)

        assert updated.accumulated_boost > state.accumulated_boost

    def test_amplitude_clamped_to_max(self):
        """Amplitude is clamped to max after update."""
        config = AmplitudeConfig(max_amplitude=1.5)
        state = AmplitudeState(current_amplitude=1.4, config=config)
        updated = update_amplitude(state, intensity_factor=10.0)

        assert updated.current_amplitude <= 1.5


class TestDecayAmplitude:
    """Tests for decay_amplitude function."""

    def test_decay_toward_base(self):
        """Amplitude decays toward base value."""
        config = AmplitudeConfig(base_amplitude=1.0, decay_rate=0.5)
        state = AmplitudeState(current_amplitude=2.0, config=config)

        decayed = decay_amplitude(state, delta_time=1.0)

        assert decayed.current_amplitude < state.current_amplitude
        assert decayed.current_amplitude > 1.0  # Still above base

    def test_decay_from_below_base(self):
        """Amplitude below base decays upward."""
        config = AmplitudeConfig(base_amplitude=1.0, decay_rate=0.5)
        state = AmplitudeState(current_amplitude=0.5, config=config)

        decayed = decay_amplitude(state, delta_time=1.0)

        assert decayed.current_amplitude > state.current_amplitude
        assert decayed.current_amplitude < 1.0  # Still below base

    def test_decay_accumulated_boost(self):
        """Accumulated boost also decays."""
        config = AmplitudeConfig(decay_rate=0.5)
        state = AmplitudeState(accumulated_boost=1.0, config=config)

        decayed = decay_amplitude(state, delta_time=1.0)

        assert decayed.accumulated_boost < state.accumulated_boost

    def test_decay_over_time_returns_to_base(self):
        """Long decay returns amplitude to base."""
        config = AmplitudeConfig(base_amplitude=1.0, decay_rate=0.5)
        state = AmplitudeState(current_amplitude=2.0, config=config)

        # Decay over many steps
        for _ in range(20):
            state = decay_amplitude(state, delta_time=1.0)

        assert state.current_amplitude == pytest.approx(1.0, abs=0.01)


class TestComputeAmplitudeFromDynamics:
    """Tests for compute_amplitude_from_dynamics function."""

    def test_normal_phase_returns_base(self):
        """Normal phase returns base amplitude."""
        dynamics = create_dynamics_state()

        amp = compute_amplitude_from_dynamics(dynamics, base_amplitude=1.0)

        assert amp == pytest.approx(1.0)

    def test_peak_phase_boosts_amplitude(self):
        """Peak phase boosts amplitude."""
        dynamics = create_dynamics_state()
        dynamics = enter_peak(dynamics, "joy", 0.8)

        amp = compute_amplitude_from_dynamics(
            dynamics, base_amplitude=1.0, peak_boost=0.3
        )

        assert amp == pytest.approx(1.3)

    def test_rebound_phase_reduces_amplitude(self):
        """Rebound phase reduces amplitude."""
        dynamics = create_dynamics_state()
        dynamics = enter_peak(dynamics, "joy", 0.8)
        dynamics = enter_rebound(dynamics)

        amp = compute_amplitude_from_dynamics(
            dynamics, base_amplitude=1.0, rebound_reduction=0.2
        )

        assert amp == pytest.approx(0.8)

    def test_none_dynamics_returns_base(self):
        """None dynamics returns base amplitude."""
        amp = compute_amplitude_from_dynamics(None, base_amplitude=1.0)

        assert amp == pytest.approx(1.0)


class TestComputeAmplitudeFromResidue:
    """Tests for compute_amplitude_from_residue function."""

    def test_none_memory_returns_base(self):
        """None memory returns base amplitude."""
        amp = compute_amplitude_from_residue(None, base_amplitude=1.0)

        assert amp == pytest.approx(1.0)


class TestReactionWithAmplitude:
    """Tests for amplitude integration with reaction system."""

    def test_neutral_amplitude_unchanged(self):
        """Amplitude 1.0 does not change reaction."""
        state = PsycheState()
        percept = Percept(emotion="happy", emotion_valence=0.5)

        result_no_amp = react(percept, state, amplitude_modifier=1.0)
        result_with_amp = react(percept, state, amplitude_modifier=1.0)

        assert result_no_amp.emotions.joy == result_with_amp.emotions.joy

    def test_amplified_reaction_larger(self):
        """Amplified reaction produces larger emotion change."""
        state = PsycheState()
        percept = Percept(emotion="happy", emotion_valence=0.5)

        result_normal = react(percept, state, amplitude_modifier=1.0)
        result_amplified = react(percept, state, amplitude_modifier=2.0)

        assert result_amplified.emotions.joy > result_normal.emotions.joy

    def test_dampened_reaction_smaller(self):
        """Dampened reaction produces smaller emotion change."""
        state = PsycheState()
        percept = Percept(emotion="happy", emotion_valence=0.5)

        result_normal = react(percept, state, amplitude_modifier=1.0)
        result_dampened = react(percept, state, amplitude_modifier=0.5)

        assert result_dampened.emotions.joy < result_normal.emotions.joy

    def test_amplitude_preserves_emotion_type(self):
        """Amplitude does not change which emotion is affected."""
        state = PsycheState()
        percept = Percept(emotion="sad", emotion_valence=-0.5)

        result = react(percept, state, amplitude_modifier=2.0)

        # Sorrow should increase, not other emotions
        assert result.emotions.sorrow > state.emotions.sorrow

    def test_amplitude_with_negative_valence(self):
        """Amplitude works correctly with negative valence."""
        state = PsycheState()
        percept = Percept(emotion="neutral", emotion_valence=-0.8)

        result_normal = react(percept, state, amplitude_modifier=1.0)
        result_amplified = react(percept, state, amplitude_modifier=1.5)

        # Negative emotions should be more affected
        assert result_amplified.emotions.sorrow > result_normal.emotions.sorrow


class TestAmplitudeSummary:
    """Tests for get_amplitude_summary function."""

    def test_summary_neutral(self):
        """Neutral amplitude shows neutral."""
        state = AmplitudeState(current_amplitude=1.0)
        summary = get_amplitude_summary(state)

        assert "neutral" in summary
        assert "1.00" in summary

    def test_summary_amplified(self):
        """Amplified state shows amplified."""
        state = AmplitudeState(current_amplitude=1.5)
        summary = get_amplitude_summary(state)

        assert "amplified" in summary

    def test_summary_dampened(self):
        """Dampened state shows dampened."""
        state = AmplitudeState(current_amplitude=0.7)
        summary = get_amplitude_summary(state)

        assert "dampened" in summary


class TestDesignConstraints:
    """Tests verifying design constraints are met."""

    def test_no_new_emotion_types(self):
        """Amplitude does not add new emotion types."""
        state = PsycheState()
        percept = Percept(emotion="happy", emotion_valence=0.5)

        result = react(percept, state, amplitude_modifier=2.0)

        # Same emotion fields exist
        original_fields = set(state.emotions.as_dict().keys())
        result_fields = set(result.emotions.as_dict().keys())
        assert original_fields == result_fields

    def test_amplitude_is_temporary(self):
        """Amplitude decays over time (not permanent)."""
        config = AmplitudeConfig(decay_rate=0.5)
        state = AmplitudeState(current_amplitude=2.0, config=config)

        # After decay, amplitude is closer to base
        decayed = decay_amplitude(state, delta_time=1.0)
        assert decayed.current_amplitude < state.current_amplitude

        # After more decay, even closer
        decayed2 = decay_amplitude(decayed, delta_time=1.0)
        assert decayed2.current_amplitude < decayed.current_amplitude

    def test_sign_always_preserved(self):
        """Sign is always preserved regardless of amplitude."""
        for amp in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]:
            pos_delta = apply_amplitude_to_delta(0.3, amp)
            neg_delta = apply_amplitude_to_delta(-0.3, amp)

            assert pos_delta > 0, f"Positive sign lost at amplitude {amp}"
            assert neg_delta < 0, f"Negative sign lost at amplitude {amp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
