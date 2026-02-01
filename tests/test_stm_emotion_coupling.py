"""
tests/test_stm_emotion_coupling.py - Tests for STM-Emotion Coupling

Verifies:
1. Read-only access to STM (no modification)
2. Persistence effect (slower decay for STM-supported emotions)
3. Re-activation (boost existing emotions when context is continuous)
4. Accumulation (repeated emotions stack within context)
5. Multi-emotion compatibility (no single emotion reduction)
6. No new emotion generation from STM
"""

import pytest
import time

from psyche.state import EmotionVector
from psyche.short_term_memory import ShortTermMemory, StimulusEntry
from psyche.stm_emotion_coupling import (
    STMEmotionCouplingConfig,
    EmotionCouplingData,
    CouplingInfluence,
    compute_coupling_influence,
    compute_decay_modifier_from_stm,
    apply_persistence_modifier,
    apply_reactivation,
    apply_reactivation_to_existing,
    apply_accumulation,
    apply_stm_coupling,
    get_coupling_summary,
    get_emotion_persistence_breakdown,
    create_coupling_config,
    to_dict,
    from_dict,
)


def create_test_stm_with_entries(
    entries: list[tuple[str, float, float]],  # (emotion_label, intensity, weight)
    continuity: float = 0.5,
) -> ShortTermMemory:
    """Create a test STM with specified entries."""
    stm = ShortTermMemory(
        context_continuity_score=continuity,
        current_context_topics=["test_topic"],
    )

    stimulus_entries = []
    for emotion_label, intensity, weight in entries:
        entry = StimulusEntry(
            source_text="test",
            topics=["test_topic"],
            emotion_label=emotion_label,
            intent="test",
            raw_intensity=intensity,
            valence=0.5 if emotion_label in ["happy", "joy"] else -0.5,
            residue_weight=weight,
            processed=False,
        )
        stimulus_entries.append(entry)

    return ShortTermMemory(
        entries=stimulus_entries,
        context_continuity_score=continuity,
        current_context_topics=["test_topic"],
    )


class TestReadOnlySTMAccess:
    """Tests verifying STM is accessed read-only."""

    def test_compute_influence_does_not_modify_stm(self):
        """Computing influence does not modify STM."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        original_entries = len(stm.entries)
        original_weight = stm.entries[0].residue_weight

        _ = compute_coupling_influence(stm)

        assert len(stm.entries) == original_entries
        assert stm.entries[0].residue_weight == original_weight

    def test_apply_coupling_does_not_modify_stm(self):
        """Applying coupling does not modify STM."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        emotions = EmotionVector(joy=0.5)
        original_continuity = stm.context_continuity_score

        _, _ = apply_stm_coupling(emotions, stm)

        assert stm.context_continuity_score == original_continuity


class TestComputeCouplingInfluence:
    """Tests for compute_coupling_influence function."""

    def test_empty_stm_returns_zero_influence(self):
        """Empty STM produces no influence."""
        stm = ShortTermMemory()

        influence = compute_coupling_influence(stm)

        assert influence.active_entry_count == 0
        for data in influence.emotion_data.values():
            assert data.persistence_support == 0.0

    def test_single_entry_creates_influence(self):
        """Single STM entry creates appropriate influence."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])

        influence = compute_coupling_influence(stm)

        assert influence.active_entry_count == 1
        assert influence.emotion_data["joy"].persistence_support > 0

    def test_multiple_entries_same_emotion(self):
        """Multiple entries for same emotion stack."""
        stm = create_test_stm_with_entries([
            ("happy", 0.3, 0.8),
            ("happy", 0.4, 0.7),
        ])

        influence = compute_coupling_influence(stm)

        assert influence.emotion_data["joy"].supporting_entry_count == 2
        assert influence.emotion_data["joy"].accumulation_weight > 0

    def test_multiple_emotions_independent(self):
        """Different emotions have independent influence."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("sad", 0.4, 0.7),
        ])

        influence = compute_coupling_influence(stm)

        # Both emotions should have independent support
        assert influence.emotion_data["joy"].supporting_entry_count == 1
        assert influence.emotion_data["sorrow"].supporting_entry_count == 1

    def test_context_continuity_detected(self):
        """Context continuity is correctly detected."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)

        influence = compute_coupling_influence(stm, config=config)

        assert influence.is_continuous is True
        assert influence.context_continuity == 0.5


class TestPersistenceEffect:
    """Tests for persistence (slower decay) effect."""

    def test_stm_support_slows_decay(self):
        """Emotions with STM support decay slower."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        joy_modifier = compute_decay_modifier_from_stm("joy", influence)
        sorrow_modifier = compute_decay_modifier_from_stm("sorrow", influence)

        # Joy has STM support, so lower modifier (slower decay)
        assert joy_modifier < sorrow_modifier

    def test_no_support_normal_decay(self):
        """Emotions without STM support decay normally."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        # Anger has no STM support
        anger_modifier = compute_decay_modifier_from_stm("anger", influence)

        assert anger_modifier == 1.0  # Normal decay

    def test_apply_persistence_modifier(self):
        """Persistence modifier actually slows decay."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        emotions = EmotionVector(joy=0.5, sorrow=0.5)

        # Apply decay with STM influence
        decayed = apply_persistence_modifier(
            emotions,
            base_decay_rate=0.9,
            delta_time=1.0,
            influence=influence,
        )

        # Joy should decay slower (higher value remaining)
        assert decayed.joy > decayed.sorrow

    def test_persistence_does_not_affect_other_emotions(self):
        """Persistence for one emotion doesn't affect others."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        emotions = EmotionVector(joy=0.5, sorrow=0.5, anger=0.5)

        decayed = apply_persistence_modifier(
            emotions,
            base_decay_rate=0.9,
            delta_time=1.0,
            influence=influence,
        )

        # Sorrow and anger should decay at same rate (no STM support)
        assert decayed.sorrow == pytest.approx(decayed.anger, rel=0.01)


class TestReactivationEffect:
    """Tests for re-activation effect."""

    def test_reactivation_boosts_existing_emotion(self):
        """Re-activation boosts existing emotions."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
            reactivation_boost_base=0.1,
        )
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        reactivated = apply_reactivation(emotions, influence)

        assert reactivated.joy > emotions.joy

    def test_reactivation_requires_continuous_context(self):
        """Re-activation only happens with continuous context."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.1)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
        )
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        reactivated = apply_reactivation(emotions, influence)

        # No boost because context is not continuous
        assert reactivated.joy == emotions.joy

    def test_reactivation_does_not_create_new_emotions(self):
        """Re-activation does not create emotions that don't exist."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.0)  # Joy is zero

        reactivated = apply_reactivation_to_existing(emotions, influence)

        # Joy should remain zero (not created from STM)
        assert reactivated.joy == 0.0

    def test_reactivation_independent_per_emotion(self):
        """Re-activation is independent for each emotion."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("sad", 0.4, 0.7),
        ], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3, sorrow=0.3)

        reactivated = apply_reactivation(emotions, influence)

        # Both should be boosted independently
        assert reactivated.joy > emotions.joy
        assert reactivated.sorrow > emotions.sorrow


class TestAccumulationEffect:
    """Tests for accumulation (stacking) effect."""

    def test_accumulation_increases_emotion(self):
        """Accumulation increases emotion value."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("happy", 0.4, 0.7),
        ], continuity=0.5)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
            accumulation_rate=0.2,
        )
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        accumulated = apply_accumulation(emotions, influence)

        assert accumulated.joy > emotions.joy

    def test_accumulation_requires_continuous_context(self):
        """Accumulation only happens with continuous context."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("happy", 0.4, 0.7),
        ], continuity=0.1)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        accumulated = apply_accumulation(emotions, influence)

        # No accumulation because context not continuous
        assert accumulated.joy == emotions.joy

    def test_accumulation_capped(self):
        """Accumulation is capped at maximum."""
        stm = create_test_stm_with_entries([
            ("happy", 1.0, 1.0),
            ("happy", 1.0, 1.0),
            ("happy", 1.0, 1.0),
            ("happy", 1.0, 1.0),
            ("happy", 1.0, 1.0),
        ], continuity=0.9)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
            accumulation_rate=0.5,
            accumulation_cap=0.2,
        )
        influence = compute_coupling_influence(stm, config=config)

        # Accumulation weight should be capped
        assert influence.emotion_data["joy"].accumulation_weight <= 0.2

    def test_accumulation_does_not_create_new_emotions(self):
        """Accumulation does not create emotions from nothing."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("happy", 0.4, 0.7),
        ], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.0)  # Joy is zero

        accumulated = apply_accumulation(emotions, influence)

        # Joy should remain zero
        assert accumulated.joy == 0.0


class TestCombinedApplication:
    """Tests for apply_stm_coupling combined function."""

    def test_all_effects_applied(self):
        """All three effects are applied together."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("happy", 0.3, 0.6),
        ], continuity=0.5)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
            reactivation_boost_base=0.05,
            accumulation_rate=0.1,
        )

        emotions = EmotionVector(joy=0.4, sorrow=0.4)

        result, influence = apply_stm_coupling(
            emotions, stm, config=config
        )

        # Joy should be higher than sorrow (persistence + reactivation + accumulation)
        assert result.joy > result.sorrow

    def test_can_disable_individual_effects(self):
        """Individual effects can be disabled."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        emotions = EmotionVector(joy=0.5)

        # Only persistence
        result1, _ = apply_stm_coupling(
            emotions, stm,
            apply_persistence=True,
            apply_reactivation_effect=False,
            apply_accumulation_effect=False,
        )

        # All effects
        result2, _ = apply_stm_coupling(
            emotions, stm,
            apply_persistence=True,
            apply_reactivation_effect=True,
            apply_accumulation_effect=True,
        )

        # Result with all effects should have higher joy
        assert result2.joy >= result1.joy

    def test_returns_influence_for_inspection(self):
        """Function returns influence for debugging."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        emotions = EmotionVector(joy=0.5)

        _, influence = apply_stm_coupling(emotions, stm)

        assert isinstance(influence, CouplingInfluence)
        assert influence.active_entry_count == 1


class TestMultiEmotionCompatibility:
    """Tests verifying multi-emotion compatibility."""

    def test_multiple_emotions_coexist(self):
        """Multiple emotions can be influenced simultaneously."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("scared", 0.4, 0.7),
            ("loving", 0.3, 0.6),
        ], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)

        emotions = EmotionVector(joy=0.3, fear=0.3, love=0.3)

        result, _ = apply_stm_coupling(emotions, stm, config=config)

        # All three emotions should still exist
        assert result.joy > 0
        assert result.fear > 0
        assert result.love > 0

    def test_no_single_emotion_reduction(self):
        """STM does not reduce emotions to single dominant one."""
        stm = create_test_stm_with_entries([
            ("happy", 0.8, 0.9),  # Strong joy support
        ], continuity=0.5)

        emotions = EmotionVector(joy=0.3, sorrow=0.5, anger=0.4)

        result, _ = apply_stm_coupling(emotions, stm)

        # Sorrow and anger should still exist (not reduced by joy support)
        assert result.sorrow > 0
        assert result.anger > 0

    def test_independent_decay_preserved(self):
        """Each emotion's decay is independent."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        breakdown = get_emotion_persistence_breakdown(influence)

        # Joy has support (lower modifier), others don't
        assert breakdown["joy"] < breakdown["sorrow"]
        assert breakdown["sorrow"] == breakdown["anger"]


class TestNoNewEmotionGeneration:
    """Tests ensuring STM does not generate new emotions."""

    def test_zero_emotion_stays_zero(self):
        """Zero emotions are not increased by STM."""
        stm = create_test_stm_with_entries([
            ("happy", 1.0, 1.0),  # Very strong joy in STM
        ], continuity=0.9)
        config = STMEmotionCouplingConfig(
            reactivation_continuity_threshold=0.3,
            reactivation_boost_base=0.5,  # High boost
        )

        emotions = EmotionVector(joy=0.0)  # Joy is zero

        result, _ = apply_stm_coupling(emotions, stm, config=config)

        # Joy should remain zero - STM doesn't create emotions
        assert result.joy == 0.0

    def test_very_small_emotion_below_threshold(self):
        """Emotions below existence threshold are not boosted."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)

        emotions = EmotionVector(joy=0.005)  # Below threshold

        result = apply_reactivation_to_existing(
            emotions, compute_coupling_influence(stm, config=config),
            existence_threshold=0.01,
        )

        # Should not be boosted (below threshold)
        assert result.joy == emotions.joy


class TestDiscontinuityBehavior:
    """Tests for behavior when context is discontinuous."""

    def test_no_reactivation_on_discontinuity(self):
        """No re-activation when context is discontinuous."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.1)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        result = apply_reactivation(emotions, influence)

        assert result.joy == emotions.joy

    def test_no_accumulation_on_discontinuity(self):
        """No accumulation when context is discontinuous."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("happy", 0.4, 0.7),
        ], continuity=0.1)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        emotions = EmotionVector(joy=0.3)

        result = apply_accumulation(emotions, influence)

        assert result.joy == emotions.joy

    def test_persistence_still_applies(self):
        """Persistence effect still applies even with discontinuity."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.1)
        influence = compute_coupling_influence(stm)

        # Persistence is NOT context-dependent (it's based on STM entries)
        modifier = compute_decay_modifier_from_stm("joy", influence)

        assert modifier < 1.0  # Still has persistence effect


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict(self):
        """Config can be serialized to dict."""
        config = create_coupling_config(
            persistence_factor=0.8,
            reactivation_boost=0.1,
        )

        data = to_dict(config)

        assert data["persistence_factor_base"] == 0.8
        assert data["reactivation_boost_base"] == 0.1

    def test_from_dict(self):
        """Config can be deserialized from dict."""
        data = {
            "persistence_factor_base": 0.6,
            "accumulation_rate": 0.15,
        }

        config = from_dict(data)

        assert config.persistence_factor_base == 0.6
        assert config.accumulation_rate == 0.15

    def test_roundtrip(self):
        """Config survives roundtrip serialization."""
        original = create_coupling_config(
            persistence_factor=0.75,
            reactivation_boost=0.08,
            accumulation_rate=0.12,
        )

        data = to_dict(original)
        restored = from_dict(data)

        assert restored.persistence_factor_base == original.persistence_factor_base
        assert restored.reactivation_boost_base == original.reactivation_boost_base
        assert restored.accumulation_rate == original.accumulation_rate


class TestSummaryAndDiagnostics:
    """Tests for summary and diagnostic functions."""

    def test_coupling_summary_empty(self):
        """Summary for empty STM."""
        stm = ShortTermMemory()
        influence = compute_coupling_influence(stm)

        summary = get_coupling_summary(influence)

        assert "No active STM entries" in summary

    def test_coupling_summary_with_entries(self):
        """Summary shows entry count and supported emotions."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)], continuity=0.5)
        config = STMEmotionCouplingConfig(reactivation_continuity_threshold=0.3)
        influence = compute_coupling_influence(stm, config=config)

        summary = get_coupling_summary(influence)

        assert "STM entries: 1" in summary
        assert "CONTINUOUS" in summary

    def test_persistence_breakdown(self):
        """Persistence breakdown shows per-emotion modifiers."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        breakdown = get_emotion_persistence_breakdown(influence)

        assert "joy" in breakdown
        assert "sorrow" in breakdown
        assert breakdown["joy"] < breakdown["sorrow"]


class TestDesignConstraints:
    """Tests verifying design document constraints are met."""

    def test_stm_affects_persistence_not_strength(self):
        """STM affects persistence ease, not direct strength."""
        stm = create_test_stm_with_entries([("happy", 0.5, 0.8)])
        influence = compute_coupling_influence(stm)

        # The coupling data contains persistence/reactivation/accumulation
        # NOT direct emotion value changes
        data = influence.emotion_data["joy"]

        assert hasattr(data, "persistence_support")
        assert hasattr(data, "reactivation_potential")
        assert hasattr(data, "accumulation_weight")

    def test_stm_does_not_generate_emotions(self):
        """STM does not generate new emotions."""
        stm = create_test_stm_with_entries([
            ("happy", 1.0, 1.0),
            ("sad", 1.0, 1.0),
        ], continuity=0.9)

        # Start with all emotions at zero
        emotions = EmotionVector()

        result, _ = apply_stm_coupling(emotions, stm)

        # All emotions should still be at default (not generated)
        emo_dict = result.as_dict()
        for value in emo_dict.values():
            assert value == 0.0

    def test_compatible_with_multi_emotion(self):
        """STM coupling is compatible with multi-emotion system."""
        stm = create_test_stm_with_entries([
            ("happy", 0.5, 0.8),
            ("scared", 0.4, 0.7),
        ], continuity=0.5)

        # All emotions active
        emotions = EmotionVector(
            joy=0.5, sorrow=0.4, anger=0.3, fear=0.35,
            surprise=0.2, love=0.3, fun=0.25,
        )

        result, _ = apply_stm_coupling(emotions, stm)

        # All 7 emotions should still be present
        emo_dict = result.as_dict()
        non_zero = sum(1 for v in emo_dict.values() if v > 0)
        assert non_zero == 7

    def test_influence_is_temporary(self):
        """STM influence weakens as entries decay."""
        # High weight entry
        stm1 = create_test_stm_with_entries([("happy", 0.5, 0.9)])
        influence1 = compute_coupling_influence(stm1)

        # Lower weight entry (simulating decay)
        stm2 = create_test_stm_with_entries([("happy", 0.5, 0.2)])
        influence2 = compute_coupling_influence(stm2)

        # Lower weight = less persistence support
        assert (influence1.emotion_data["joy"].persistence_support >
                influence2.emotion_data["joy"].persistence_support)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
