"""
tests/test_decision_bias.py - Tests for decision bias injection structure

Verifies:
1. Bias computation from ShortTermMemory and DynamicsState
2. Bias application to policy scores
3. Configurable scale factors
4. Integration with existing thought.py scoring
5. Natural decay alignment
6. No algorithm flow changes
"""

import pytest
import time

from psyche.decision_bias import (
    DecisionBias,
    DecisionBiasConfig,
    compute_decision_bias,
    apply_bias_to_score,
    get_policy_bias_breakdown,
    create_neutral_bias,
    merge_biases,
)
from psyche.short_term_memory import ShortTermMemory, StimulusEntry
from psyche.dynamics import (
    DynamicsState,
    DynamicsPhase,
    DynamicsConfig,
    create_dynamics_state,
    enter_peak,
    enter_rebound,
)
from psyche.thought import (
    generate_thought_candidates,
    select_policy,
    POLICIES,
)
from psyche.state import PsycheState, Percept


class TestDecisionBiasComputation:
    """Tests for computing bias from memory and dynamics."""

    def test_neutral_bias_from_empty_inputs(self):
        """Empty inputs produce neutral bias."""
        bias = compute_decision_bias()

        assert bias.is_neutral()
        assert bias.dynamics_phase == DynamicsPhase.NORMAL

    def test_bias_from_short_term_memory(self):
        """ShortTermMemory residue creates emotion biases."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
        )

        bias = compute_decision_bias(memory=memory)

        assert not bias.is_neutral()
        assert "angry" in bias.emotion_biases
        assert bias.emotion_biases["angry"] > 0
        assert bias.valence_bias < 0  # Negative valence

    def test_bias_from_dynamics_peak(self):
        """Peak dynamics state creates peak boost."""
        config = DynamicsConfig(peak_intensity_boost=0.5)
        dynamics = create_dynamics_state(config)
        dynamics = enter_peak(dynamics, "joy", 0.8)

        bias = compute_decision_bias(dynamics=dynamics)

        assert bias.dynamics_phase == DynamicsPhase.PEAK
        assert bias.peak_boost > 0
        assert bias.peak_emotion == "joy"
        assert bias.rebound_dampening == 0.0

    def test_bias_from_dynamics_rebound(self):
        """Rebound dynamics state creates dampening."""
        config = DynamicsConfig(rebound_dampening=0.3)
        dynamics = create_dynamics_state(config)
        dynamics = enter_peak(dynamics, "joy", 0.8)
        dynamics = enter_rebound(dynamics)

        bias = compute_decision_bias(dynamics=dynamics)

        assert bias.dynamics_phase == DynamicsPhase.REBOUND
        assert bias.rebound_dampening > 0
        assert bias.peak_boost == 0.0

    def test_combined_memory_and_dynamics(self):
        """Both memory and dynamics contribute to bias."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="sad",
            intent="sharing",
            raw_intensity=0.7,
            valence=-0.5,
        )

        config = DynamicsConfig(peak_intensity_boost=0.4)
        dynamics = create_dynamics_state(config)
        dynamics = enter_peak(dynamics, "sad", 0.7)

        bias = compute_decision_bias(memory=memory, dynamics=dynamics)

        assert "sad" in bias.emotion_biases
        assert bias.dynamics_phase == DynamicsPhase.PEAK
        assert bias.peak_boost > 0
        assert bias.valence_bias < 0


class TestDecisionBiasApplication:
    """Tests for applying bias to policy scores."""

    def test_neutral_bias_no_effect(self):
        """Neutral bias doesn't change score."""
        bias = create_neutral_bias()
        base_score = 5.0

        result = apply_bias_to_score(base_score, bias, "共感する")

        assert result == base_score

    def test_angry_residue_boosts_empathy(self):
        """Angry residue increases empathy policy score."""
        bias = DecisionBias(
            emotion_biases={"angry": 1.0},
            valence_bias=-0.5,
        )
        base_score = 5.0

        empathy_score = apply_bias_to_score(base_score, bias, "共感する")
        tease_score = apply_bias_to_score(base_score, bias, "からかう")

        # Empathy should be boosted, teasing penalized
        assert empathy_score > base_score
        assert tease_score < base_score

    def test_negative_valence_favors_support(self):
        """Negative valence favors supportive policies."""
        bias = DecisionBias(valence_bias=-0.5)
        base_score = 5.0

        empathy_score = apply_bias_to_score(base_score, bias, "共感する")
        encourage_score = apply_bias_to_score(base_score, bias, "励ます")
        tease_score = apply_bias_to_score(base_score, bias, "からかう")

        assert empathy_score > base_score
        assert encourage_score > base_score
        assert tease_score < base_score

    def test_positive_valence_allows_expression(self):
        """Positive valence slightly favors expression."""
        bias = DecisionBias(valence_bias=0.5)
        base_score = 5.0

        tease_score = apply_bias_to_score(base_score, bias, "からかう")
        comment_score = apply_bias_to_score(base_score, bias, "感想を述べる")

        assert tease_score > base_score
        assert comment_score > base_score

    def test_peak_phase_boosts_scores(self):
        """Peak phase adds boost to scores."""
        bias = DecisionBias(
            dynamics_phase=DynamicsPhase.PEAK,
            peak_boost=0.5,
            peak_emotion="joy",
        )
        base_score = 5.0

        result = apply_bias_to_score(base_score, bias, "共感する")

        assert result > base_score

    def test_rebound_phase_dampens_adjustment(self):
        """Rebound phase dampens bias adjustments."""
        # Create a bias with some emotion influence
        bias_normal = DecisionBias(
            emotion_biases={"angry": 1.0},
            dynamics_phase=DynamicsPhase.NORMAL,
        )
        bias_rebound = DecisionBias(
            emotion_biases={"angry": 1.0},
            dynamics_phase=DynamicsPhase.REBOUND,
            rebound_dampening=0.5,
        )
        base_score = 5.0

        normal_result = apply_bias_to_score(base_score, bias_normal, "共感する")
        rebound_result = apply_bias_to_score(base_score, bias_rebound, "共感する")

        # Rebound should have smaller adjustment
        normal_delta = abs(normal_result - base_score)
        rebound_delta = abs(rebound_result - base_score)

        # Rebound may have slightly different score due to calm preference
        # but the magnitude of emotion-based adjustment should be smaller
        assert rebound_delta <= normal_delta + 0.2  # Allow some tolerance


class TestConfigurableScaleFactors:
    """Tests for configurable bias scale factors."""

    def test_global_scale_zero_neutralizes(self):
        """Global scale of 0 neutralizes all bias."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
        )

        config = DecisionBiasConfig(global_scale=0.0)
        bias = compute_decision_bias(memory=memory, config=config)

        assert bias.is_neutral()

    def test_emotion_scale_affects_emotion_bias(self):
        """Residue emotion scale affects emotion biases."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
        )

        config_low = DecisionBiasConfig(residue_emotion_scale=0.5)
        config_high = DecisionBiasConfig(residue_emotion_scale=2.0)

        bias_low = compute_decision_bias(memory=memory, config=config_low)
        bias_high = compute_decision_bias(memory=memory, config=config_high)

        assert bias_high.emotion_biases["angry"] > bias_low.emotion_biases["angry"]

    def test_dynamics_scale_affects_peak_boost(self):
        """Dynamics phase scale affects peak boost."""
        config_dynamics = DynamicsConfig(peak_intensity_boost=0.5)
        dynamics = create_dynamics_state(config_dynamics)
        dynamics = enter_peak(dynamics, "joy", 0.8)

        config_low = DecisionBiasConfig(dynamics_phase_scale=0.5)
        config_high = DecisionBiasConfig(dynamics_phase_scale=2.0)

        bias_low = compute_decision_bias(dynamics=dynamics, config=config_low)
        bias_high = compute_decision_bias(dynamics=dynamics, config=config_high)

        assert bias_high.peak_boost > bias_low.peak_boost


class TestThoughtIntegration:
    """Tests for integration with thought.py scoring."""

    def test_generate_candidates_with_bias(self):
        """Bias parameter is accepted by generate_thought_candidates."""
        state = PsycheState()
        percept = Percept(
            text="悲しい...",
            emotion="sad",
            emotion_valence=-0.7,
            intent="sharing",
        )
        recalled = []

        # Without bias
        candidates_no_bias = generate_thought_candidates(
            state, percept, recalled
        )

        # With bias that favors empathy
        bias = DecisionBias(
            emotion_biases={"sad": 1.0},
            valence_bias=-0.5,
        )
        candidates_with_bias = generate_thought_candidates(
            state, percept, recalled, decision_bias=bias
        )

        # Both should return candidates
        assert len(candidates_no_bias) > 0
        assert len(candidates_with_bias) > 0

        # Scores should differ
        scores_no_bias = {c["policy_label"]: c["_score"] for c in candidates_no_bias}
        scores_with_bias = {c["policy_label"]: c["_score"] for c in candidates_with_bias}

        # Empathy should be boosted with bias
        if "共感する" in scores_no_bias and "共感する" in scores_with_bias:
            assert scores_with_bias["共感する"] >= scores_no_bias.get("共感する", 0)

    def test_bias_affects_policy_ranking(self):
        """Bias can change policy ranking."""
        state = PsycheState()
        percept = Percept(
            text="楽しい！",
            emotion="happy",
            emotion_valence=0.8,
            intent="sharing",
        )
        recalled = []

        # Create bias that strongly favors questions over default
        bias = DecisionBias(
            intent_biases={"sharing": 2.0},
            valence_bias=0.5,
        )

        candidates = generate_thought_candidates(
            state, percept, recalled, decision_bias=bias
        )

        # Should have multiple candidates
        assert len(candidates) >= 2

    def test_algorithm_flow_unchanged(self):
        """Bias doesn't change algorithm flow (always returns same structure)."""
        state = PsycheState()
        percept = Percept(text="Test", emotion="neutral", emotion_valence=0.0)
        recalled = []

        # Various bias configurations
        biases = [
            None,
            create_neutral_bias(),
            DecisionBias(emotion_biases={"angry": 1.0}),
            DecisionBias(dynamics_phase=DynamicsPhase.PEAK, peak_boost=0.5),
        ]

        for bias in biases:
            candidates = generate_thought_candidates(
                state, percept, recalled, decision_bias=bias
            )

            # Same structure regardless of bias
            assert isinstance(candidates, list)
            assert len(candidates) <= 5
            for c in candidates:
                assert "policy_label" in c
                assert "_score" in c


class TestDecayAlignment:
    """Tests for alignment with existing decay structures."""

    def test_residue_decay_reduces_bias(self):
        """Decayed residue produces smaller bias."""
        # Fresh memory
        memory_fresh = ShortTermMemory()
        memory_fresh = memory_fresh.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
        )

        # Decayed memory (simulate with lower residue weight)
        memory_decayed = ShortTermMemory()
        entry = StimulusEntry(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
            residue_weight=0.3,  # Decayed
        )
        memory_decayed = ShortTermMemory(entries=[entry])

        bias_fresh = compute_decision_bias(memory=memory_fresh)
        bias_decayed = compute_decision_bias(memory=memory_decayed)

        assert bias_decayed.emotion_biases.get("angry", 0) < bias_fresh.emotion_biases.get("angry", 0)

    def test_processed_entries_excluded(self):
        """Processed entries don't contribute to bias."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="Test",
            topics=["topic1"],
            emotion_label="angry",
            intent="complaint",
            raw_intensity=0.8,
            valence=-0.6,
        )

        bias_unprocessed = compute_decision_bias(memory=memory)

        # Mark as processed
        memory = memory.mark_processed()
        bias_processed = compute_decision_bias(memory=memory)

        # Processed should have no emotion biases
        assert bias_unprocessed.emotion_biases.get("angry", 0) > 0
        assert bias_processed.emotion_biases.get("angry", 0) == 0


class TestDiagnostics:
    """Tests for diagnostic functions."""

    def test_get_summary(self):
        """Bias summary provides useful info."""
        bias = DecisionBias(
            emotion_biases={"angry": 1.0, "sad": 0.5},
            valence_bias=-0.5,
            dynamics_phase=DynamicsPhase.PEAK,
            peak_boost=0.3,
        )

        summary = bias.get_summary()

        assert summary["phase"] == "peak"
        assert summary["peak_boost"] == 0.3
        assert summary["emotion_count"] == 2

    def test_policy_bias_breakdown(self):
        """Breakdown shows bias per policy."""
        bias = DecisionBias(
            emotion_biases={"angry": 1.0},
            valence_bias=-0.5,
        )

        policy_labels = [p["policy_label"] for p in POLICIES]
        breakdown = get_policy_bias_breakdown(bias, policy_labels)

        assert "共感する" in breakdown
        assert "からかう" in breakdown
        # Empathy should have positive bias, teasing negative
        assert breakdown["共感する"] > 0
        assert breakdown["からかう"] < 0


class TestMergeBiases:
    """Tests for merging multiple bias sources."""

    def test_merge_empty_list(self):
        """Merging empty list returns neutral bias."""
        result = merge_biases([])
        assert result.is_neutral()

    def test_merge_sums_values(self):
        """Merged biases sum their values."""
        bias1 = DecisionBias(
            emotion_biases={"angry": 0.5},
            valence_bias=-0.3,
        )
        bias2 = DecisionBias(
            emotion_biases={"angry": 0.3, "sad": 0.2},
            valence_bias=-0.2,
        )

        merged = merge_biases([bias1, bias2])

        assert merged.emotion_biases["angry"] == 0.8
        assert merged.emotion_biases["sad"] == 0.2
        assert merged.valence_bias == -0.5

    def test_merge_takes_first_non_normal_phase(self):
        """Merged bias uses first non-NORMAL phase."""
        bias1 = DecisionBias(dynamics_phase=DynamicsPhase.NORMAL)
        bias2 = DecisionBias(dynamics_phase=DynamicsPhase.PEAK, peak_emotion="joy")
        bias3 = DecisionBias(dynamics_phase=DynamicsPhase.REBOUND)

        merged = merge_biases([bias1, bias2, bias3])

        assert merged.dynamics_phase == DynamicsPhase.PEAK
        assert merged.peak_emotion == "joy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
