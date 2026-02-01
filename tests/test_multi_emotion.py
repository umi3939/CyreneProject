"""
tests/test_multi_emotion.py - Tests for multi-emotion reference system

Verifies:
1. Emotions coexist without cancellation or normalization
2. Independent decay per emotion type
3. No single "dominant" emotion reduction
4. Multiple emotions can be active simultaneously
5. Read-only reference functions
"""

import pytest

from psyche.state import EmotionVector
from psyche.multi_emotion import (
    EmotionDecayConfig,
    MultiEmotionConfig,
    get_active_emotions,
    get_all_emotions,
    get_coexisting_pairs,
    has_conflicting_emotions,
    get_emotion_intensity,
    get_emotion_spread,
    apply_independent_decay,
    apply_independent_update,
    set_emotions_independently,
    reference_emotions_for_judgment,
    reference_emotion_by_name,
    reference_multiple_emotions,
    get_emotion_vector_summary,
    create_multi_emotion_config,
    to_dict,
    from_dict,
)


class TestEmotionCoexistence:
    """Tests verifying emotions coexist without cancellation."""

    def test_joy_and_sorrow_coexist(self):
        """Joy and sorrow can both be active simultaneously."""
        emotions = EmotionVector(joy=0.8, sorrow=0.6)
        active = get_active_emotions(emotions)

        assert "joy" in active
        assert "sorrow" in active
        assert active["joy"] == pytest.approx(0.8)
        assert active["sorrow"] == pytest.approx(0.6)

    def test_joy_and_fear_coexist(self):
        """Joy and fear can both be active simultaneously."""
        emotions = EmotionVector(joy=0.7, fear=0.5)
        active = get_active_emotions(emotions)

        assert "joy" in active
        assert "fear" in active

    def test_love_and_anger_coexist(self):
        """Love and anger can both be active simultaneously."""
        emotions = EmotionVector(love=0.9, anger=0.4)
        active = get_active_emotions(emotions)

        assert "love" in active
        assert "anger" in active

    def test_all_emotions_can_be_active(self):
        """All seven emotions can be active at once."""
        emotions = EmotionVector(
            joy=0.5, sorrow=0.4, anger=0.3, fear=0.35,
            surprise=0.6, love=0.7, fun=0.45
        )
        active = get_active_emotions(emotions)

        assert len(active) == 7

    def test_no_normalization_after_update(self):
        """Increasing one emotion does NOT decrease others."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5, anger=0.3)

        # Increase joy
        updated = apply_independent_update(emotions, {"joy": 0.3})

        # Joy increased
        assert updated.joy == pytest.approx(0.8)
        # Sorrow and anger remain UNCHANGED
        assert updated.sorrow == pytest.approx(0.5)
        assert updated.anger == pytest.approx(0.3)

    def test_decreasing_one_does_not_increase_others(self):
        """Decreasing one emotion does NOT increase others."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5)

        updated = apply_independent_update(emotions, {"joy": -0.3})

        assert updated.joy == pytest.approx(0.2)
        # Sorrow remains UNCHANGED (not increased)
        assert updated.sorrow == pytest.approx(0.5)


class TestIndependentDecay:
    """Tests for per-emotion independent decay."""

    def test_different_decay_rates(self):
        """Different emotions decay at different rates."""
        emotions = EmotionVector(
            joy=0.5, sorrow=0.5, surprise=0.5, love=0.5
        )

        # Apply decay for 10 seconds
        decayed = apply_independent_decay(emotions, delta_time=10.0)

        # Surprise decays fastest (0.15 rate)
        # Love decays slowest (0.01 rate)
        # Sorrow decays slower than joy (0.03 vs 0.05)
        assert decayed.surprise < decayed.joy < decayed.sorrow < decayed.love

    def test_decay_is_independent(self):
        """Each emotion decays independently, not affecting others."""
        emotions = EmotionVector(joy=0.8, sorrow=0.2)

        decayed = apply_independent_decay(emotions, delta_time=5.0)

        # Both should have decayed, but independently
        assert decayed.joy < emotions.joy
        assert decayed.sorrow < emotions.sorrow

        # The decay of joy does NOT affect sorrow's decay
        # (no compensating increase in sorrow)

    def test_decay_respects_minimum(self):
        """Decayed values don't go below 0."""
        emotions = EmotionVector(joy=0.01)

        # Very long decay
        decayed = apply_independent_decay(emotions, delta_time=100.0)

        assert decayed.joy >= 0.0

    def test_decay_preserves_inactive_emotions(self):
        """Zero emotions remain zero after decay."""
        emotions = EmotionVector(joy=0.5, sorrow=0.0)

        decayed = apply_independent_decay(emotions, delta_time=5.0)

        assert decayed.sorrow == 0.0

    def test_custom_decay_rates(self):
        """Custom decay rates can be configured."""
        config = create_multi_emotion_config(
            emotion_decay_rates={"joy": 0.5, "sorrow": 0.1}
        )
        emotions = EmotionVector(joy=0.5, sorrow=0.5)

        decayed = apply_independent_decay(emotions, delta_time=1.0, config=config)

        # Joy should decay faster
        assert decayed.joy < decayed.sorrow


class TestActiveEmotions:
    """Tests for get_active_emotions function."""

    def test_threshold_filtering(self):
        """Only emotions above threshold are returned."""
        emotions = EmotionVector(joy=0.5, sorrow=0.05, anger=0.15)
        config = MultiEmotionConfig(active_threshold=0.1)

        active = get_active_emotions(emotions, config)

        assert "joy" in active
        assert "anger" in active
        assert "sorrow" not in active  # Below threshold

    def test_returns_all_above_threshold(self):
        """Returns ALL emotions above threshold, not just one."""
        emotions = EmotionVector(joy=0.5, love=0.4, fun=0.3)

        active = get_active_emotions(emotions)

        # All three should be returned, not just the highest
        assert len(active) == 3

    def test_empty_when_all_below_threshold(self):
        """Returns empty dict when all emotions below threshold."""
        emotions = EmotionVector(joy=0.05, sorrow=0.02)
        config = MultiEmotionConfig(active_threshold=0.1)

        active = get_active_emotions(emotions, config)

        assert len(active) == 0

    def test_does_not_modify_original(self):
        """get_active_emotions is read-only."""
        emotions = EmotionVector(joy=0.5, sorrow=0.4)
        original_joy = emotions.joy

        _ = get_active_emotions(emotions)

        assert emotions.joy == original_joy


class TestCoexistingPairs:
    """Tests for get_coexisting_pairs function."""

    def test_returns_pairs_above_coexistence_threshold(self):
        """Returns pairs where both emotions exceed coexistence threshold."""
        emotions = EmotionVector(joy=0.5, sorrow=0.4, anger=0.3)
        config = MultiEmotionConfig(coexistence_threshold=0.25)

        pairs = get_coexisting_pairs(emotions, config)

        # joy-sorrow, joy-anger, sorrow-anger all qualify
        assert len(pairs) == 3

    def test_no_pairs_when_only_one_active(self):
        """Returns empty when only one emotion above threshold."""
        emotions = EmotionVector(joy=0.5, sorrow=0.05)
        config = MultiEmotionConfig(coexistence_threshold=0.1)

        pairs = get_coexisting_pairs(emotions, config)

        assert len(pairs) == 0

    def test_pair_contains_both_values(self):
        """Each pair contains both emotion names and values."""
        emotions = EmotionVector(joy=0.5, sorrow=0.4)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        pairs = get_coexisting_pairs(emotions, config)

        assert len(pairs) == 1
        name_a, name_b, val_a, val_b = pairs[0]
        assert {name_a, name_b} == {"joy", "sorrow"}


class TestConflictingEmotions:
    """Tests for has_conflicting_emotions function."""

    def test_joy_sorrow_conflict(self):
        """Joy and sorrow are considered conflicting."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        assert has_conflicting_emotions(emotions, config) is True

    def test_joy_fear_conflict(self):
        """Joy and fear are considered conflicting."""
        emotions = EmotionVector(joy=0.5, fear=0.5)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        assert has_conflicting_emotions(emotions, config) is True

    def test_love_anger_conflict(self):
        """Love and anger are considered conflicting."""
        emotions = EmotionVector(love=0.5, anger=0.5)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        assert has_conflicting_emotions(emotions, config) is True

    def test_no_conflict_when_below_threshold(self):
        """No conflict when one emotion is below threshold."""
        emotions = EmotionVector(joy=0.5, sorrow=0.1)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        assert has_conflicting_emotions(emotions, config) is False

    def test_non_opposing_emotions_not_conflict(self):
        """Non-opposing emotions don't count as conflict."""
        emotions = EmotionVector(joy=0.5, fun=0.5, love=0.5)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        assert has_conflicting_emotions(emotions, config) is False


class TestEmotionMetrics:
    """Tests for emotion intensity and spread calculations."""

    def test_intensity_is_sum(self):
        """Intensity is the sum of all emotion values."""
        emotions = EmotionVector(joy=0.3, sorrow=0.2, anger=0.1)

        intensity = get_emotion_intensity(emotions)

        assert intensity == pytest.approx(0.6)

    def test_spread_counts_active(self):
        """Spread counts emotions above minimal threshold (0.05)."""
        emotions = EmotionVector(joy=0.5, sorrow=0.1, anger=0.03)

        spread = get_emotion_spread(emotions)

        assert spread == 2  # joy and sorrow, not anger

    def test_spread_with_all_active(self):
        """Spread can count all 7 emotions."""
        emotions = EmotionVector(
            joy=0.5, sorrow=0.4, anger=0.3, fear=0.35,
            surprise=0.6, love=0.7, fun=0.45
        )

        spread = get_emotion_spread(emotions)

        assert spread == 7


class TestIndependentUpdate:
    """Tests for apply_independent_update function."""

    def test_update_single_emotion(self):
        """Can update a single emotion."""
        emotions = EmotionVector(joy=0.3)

        updated = apply_independent_update(emotions, {"joy": 0.2})

        assert updated.joy == pytest.approx(0.5)

    def test_update_multiple_emotions(self):
        """Can update multiple emotions at once."""
        emotions = EmotionVector(joy=0.3, sorrow=0.2)

        updated = apply_independent_update(emotions, {"joy": 0.1, "sorrow": 0.1})

        assert updated.joy == pytest.approx(0.4)
        assert updated.sorrow == pytest.approx(0.3)

    def test_update_does_not_affect_others(self):
        """Updating one emotion does NOT affect others."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5, anger=0.5)

        updated = apply_independent_update(emotions, {"joy": 0.3})

        # Only joy changes
        assert updated.joy == pytest.approx(0.8)
        # Others remain UNCHANGED
        assert updated.sorrow == pytest.approx(0.5)
        assert updated.anger == pytest.approx(0.5)

    def test_update_clamped_to_max(self):
        """Updated values clamped to 1.0 max."""
        emotions = EmotionVector(joy=0.8)

        updated = apply_independent_update(emotions, {"joy": 0.5})

        assert updated.joy == pytest.approx(1.0)

    def test_update_clamped_to_min(self):
        """Updated values clamped to 0.0 min."""
        emotions = EmotionVector(joy=0.2)

        updated = apply_independent_update(emotions, {"joy": -0.5})

        assert updated.joy == pytest.approx(0.0)

    def test_negative_delta_works(self):
        """Negative deltas decrease emotion values."""
        emotions = EmotionVector(joy=0.5)

        updated = apply_independent_update(emotions, {"joy": -0.2})

        assert updated.joy == pytest.approx(0.3)


class TestSetEmotionsIndependently:
    """Tests for set_emotions_independently function."""

    def test_set_single_emotion(self):
        """Can set a single emotion to exact value."""
        emotions = EmotionVector(joy=0.3)

        updated = set_emotions_independently(emotions, {"joy": 0.7})

        assert updated.joy == pytest.approx(0.7)

    def test_set_does_not_affect_others(self):
        """Setting one emotion does NOT affect others."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5)

        updated = set_emotions_independently(emotions, {"joy": 0.9})

        assert updated.joy == pytest.approx(0.9)
        assert updated.sorrow == pytest.approx(0.5)  # Unchanged

    def test_set_multiple_emotions(self):
        """Can set multiple emotions at once."""
        emotions = EmotionVector(joy=0.3, sorrow=0.3)

        updated = set_emotions_independently(emotions, {"joy": 0.8, "sorrow": 0.1})

        assert updated.joy == pytest.approx(0.8)
        assert updated.sorrow == pytest.approx(0.1)


class TestReferenceForJudgment:
    """Tests for reference_emotions_for_judgment function."""

    def test_returns_all_required_fields(self):
        """Returns all required reference data."""
        emotions = EmotionVector(joy=0.5, sorrow=0.3)

        ref = reference_emotions_for_judgment(emotions)

        assert "all_emotions" in ref
        assert "active_emotions" in ref
        assert "active_count" in ref
        assert "total_intensity" in ref
        assert "has_conflict" in ref
        assert "coexisting_pairs" in ref

    def test_read_only_does_not_modify(self):
        """Reference is read-only, does not modify state."""
        emotions = EmotionVector(joy=0.5, sorrow=0.3)
        original_joy = emotions.joy

        _ = reference_emotions_for_judgment(emotions)

        assert emotions.joy == original_joy

    def test_shows_multiple_active(self):
        """Reference correctly shows multiple active emotions."""
        emotions = EmotionVector(joy=0.5, sorrow=0.4, anger=0.3)

        ref = reference_emotions_for_judgment(emotions)

        assert ref["active_count"] == 3


class TestReferenceFunctions:
    """Tests for individual reference functions."""

    def test_reference_by_name(self):
        """Can reference a specific emotion by name."""
        emotions = EmotionVector(joy=0.7)

        value = reference_emotion_by_name(emotions, "joy")

        assert value == pytest.approx(0.7)

    def test_reference_by_name_not_found(self):
        """Returns 0.0 for unknown emotion name."""
        emotions = EmotionVector(joy=0.7)

        value = reference_emotion_by_name(emotions, "unknown")

        assert value == 0.0

    def test_reference_multiple(self):
        """Can reference multiple emotions at once."""
        emotions = EmotionVector(joy=0.7, sorrow=0.3, anger=0.5)

        values = reference_multiple_emotions(emotions, ["joy", "anger"])

        assert values["joy"] == pytest.approx(0.7)
        assert values["anger"] == pytest.approx(0.5)
        assert "sorrow" not in values


class TestSummary:
    """Tests for get_emotion_vector_summary function."""

    def test_summary_shows_all_active(self):
        """Summary includes all active emotions."""
        emotions = EmotionVector(joy=0.5, sorrow=0.4, anger=0.3)

        summary = get_emotion_vector_summary(emotions)

        assert "joy" in summary
        assert "sorrow" in summary
        assert "anger" in summary

    def test_summary_shows_conflict(self):
        """Summary shows conflict marker when present."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5)
        config = MultiEmotionConfig(coexistence_threshold=0.2)

        summary = get_emotion_vector_summary(emotions, config)

        assert "CONFLICT" in summary

    def test_summary_calm_when_no_active(self):
        """Summary shows calm when no active emotions."""
        emotions = EmotionVector(joy=0.01, sorrow=0.02)
        config = MultiEmotionConfig(active_threshold=0.1)

        summary = get_emotion_vector_summary(emotions, config)

        assert "calm" in summary


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict_serialization(self):
        """Config can be serialized to dict."""
        config = create_multi_emotion_config(
            active_threshold=0.2,
            coexistence_threshold=0.3,
        )

        data = to_dict(config)

        assert data["active_threshold"] == 0.2
        assert data["coexistence_threshold"] == 0.3

    def test_from_dict_deserialization(self):
        """Config can be deserialized from dict."""
        data = {
            "active_threshold": 0.25,
            "coexistence_threshold": 0.35,
            "decay_config": {
                "base_decay_rate": 0.1,
            }
        }

        config = from_dict(data)

        assert config.active_threshold == 0.25
        assert config.coexistence_threshold == 0.35
        assert config.decay_config.base_decay_rate == 0.1

    def test_roundtrip_serialization(self):
        """Config survives serialization roundtrip."""
        original = create_multi_emotion_config(
            active_threshold=0.15,
            coexistence_threshold=0.25,
            base_decay_rate=0.08,
        )

        data = to_dict(original)
        restored = from_dict(data)

        assert restored.active_threshold == original.active_threshold
        assert restored.coexistence_threshold == original.coexistence_threshold
        assert restored.decay_config.base_decay_rate == original.decay_config.base_decay_rate


class TestDesignConstraints:
    """Tests verifying design document constraints are met."""

    def test_no_normalization_ever(self):
        """Emotions are NEVER normalized (sum to 1)."""
        emotions = EmotionVector(
            joy=1.0, sorrow=1.0, anger=1.0, fear=1.0,
            surprise=1.0, love=1.0, fun=1.0
        )

        # All at max is valid
        all_emo = get_all_emotions(emotions)
        total = sum(all_emo.values())

        # Total can exceed 1.0 (7.0 in this case)
        assert total == pytest.approx(7.0)

    def test_no_cancellation(self):
        """Joy does NOT cancel or reduce sorrow."""
        emotions = EmotionVector(joy=0.5, sorrow=0.5)

        # Even with high joy, sorrow remains
        updated = apply_independent_update(emotions, {"joy": 0.5})

        assert updated.sorrow == pytest.approx(0.5)  # Unchanged

    def test_no_averaging(self):
        """Emotions are NOT averaged or merged."""
        emotions = EmotionVector(joy=0.8, sorrow=0.2, anger=0.6)

        active = get_active_emotions(emotions)

        # Each emotion keeps its distinct value
        assert active["joy"] == pytest.approx(0.8)
        assert active["sorrow"] == pytest.approx(0.2)
        assert active["anger"] == pytest.approx(0.6)

    def test_no_single_dominant(self):
        """Multiple emotions are returned, not just the highest."""
        emotions = EmotionVector(joy=0.9, sorrow=0.5, anger=0.3, fear=0.2)

        active = get_active_emotions(emotions)

        # All 4 returned, not just joy
        assert len(active) == 4

    def test_decay_independent_per_emotion(self):
        """Each emotion has its own decay rate (not shared)."""
        config = MultiEmotionConfig()

        # Different emotions have different decay rates
        decay_rates = config.decay_config.emotion_decay_rates

        assert decay_rates["surprise"] != decay_rates["love"]
        assert decay_rates["anger"] != decay_rates["sorrow"]

    def test_emotions_can_swap_dominance(self):
        """Over time, different emotions can become more prominent."""
        emotions = EmotionVector(surprise=0.5, love=0.5)

        # After decay, love should be higher (decays slower)
        decayed = apply_independent_decay(emotions, delta_time=10.0)

        assert decayed.love > decayed.surprise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
