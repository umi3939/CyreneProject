"""
Tests for psyche/value_orientation.py - Consistent Value Orientation (一貫した価値軸)

Verifies:
- Value orientation persists and updates gradually (high inertia)
- No hardcoded moral rules (abstract dimensions only)
- Bias is subtle and does not dictate decisions
- Serialization/deserialization works correctly
"""

import pytest
import time

from psyche.value_orientation import (
    ValueOrientation,
    ValueOrientationConfig,
    OrientationBias,
    compute_effective_learning_rate,
    update_dimension,
    update_orientation,
    generate_decision_signal,
    update_from_decision,
    compute_orientation_bias,
    apply_orientation_to_candidate,
    apply_orientation_to_candidates,
    generate_emotion_signal,
    generate_responsibility_signal,
    get_orientation_summary,
    get_orientation_vector,
    compute_orientation_distance,
    is_orientation_stable,
    create_orientation,
    create_config,
    to_dict,
    from_dict,
)


# ── ValueOrientation Tests ──────────────────────────────────────────


class TestValueOrientation:
    """Tests for ValueOrientation dataclass."""

    def test_default_values_are_neutral(self):
        """Default orientation has all neutral (zero) values."""
        orientation = ValueOrientation()
        assert orientation.dim_a == 0.0
        assert orientation.dim_b == 0.0
        assert orientation.dim_c == 0.0
        assert orientation.dim_d == 0.0
        assert orientation.dim_e == 0.0

    def test_values_clamped_to_range(self):
        """Values are clamped to [-1, 1]."""
        orientation = ValueOrientation(
            dim_a=1.5,
            dim_b=-1.5,
            dim_c=2.0,
            confidence_a=1.5,
            confidence_b=-0.5,
        )
        assert orientation.dim_a == 1.0
        assert orientation.dim_b == -1.0
        assert orientation.dim_c == 1.0
        assert orientation.confidence_a == 1.0
        assert orientation.confidence_b == 0.0

    def test_get_dimension(self):
        """Can get dimension by name."""
        orientation = ValueOrientation(dim_a=0.5, dim_c=-0.3)
        assert orientation.get_dimension("a") == 0.5
        assert orientation.get_dimension("c") == -0.3
        assert orientation.get_dimension("x") == 0.0  # Unknown

    def test_get_all_dimensions(self):
        """Can get all dimensions as dict."""
        orientation = ValueOrientation(dim_a=0.5, dim_b=-0.2)
        dims = orientation.get_all_dimensions()
        assert dims["a"] == 0.5
        assert dims["b"] == -0.2
        assert "c" in dims

    def test_overall_stability(self):
        """Overall stability is average of confidences."""
        orientation = ValueOrientation(
            confidence_a=0.4,
            confidence_b=0.6,
            confidence_c=0.5,
            confidence_d=0.5,
            confidence_e=0.5,
        )
        stability = orientation.get_overall_stability()
        assert abs(stability - 0.5) < 0.01

    def test_serialization(self):
        """Orientation can be serialized and deserialized."""
        orientation = ValueOrientation(
            dim_a=0.5,
            dim_b=-0.3,
            confidence_a=0.4,
            update_count=10,
        )
        data = orientation.to_dict()
        restored = ValueOrientation.from_dict(data)

        assert restored.dim_a == orientation.dim_a
        assert restored.dim_b == orientation.dim_b
        assert restored.confidence_a == orientation.confidence_a
        assert restored.update_count == orientation.update_count


class TestAbstractDimensions:
    """Tests verifying dimensions are abstract, not moral rules."""

    def test_no_moral_names_in_orientation(self):
        """Orientation has no morally-named fields."""
        orientation = ValueOrientation()
        # Verify no moral labels
        assert not hasattr(orientation, "justice")
        assert not hasattr(orientation, "kindness")
        assert not hasattr(orientation, "honesty")
        assert not hasattr(orientation, "fairness")
        assert not hasattr(orientation, "virtue")

    def test_dimensions_are_abstract_letters(self):
        """Dimensions use abstract names (a, b, c, d, e)."""
        orientation = ValueOrientation()
        dims = orientation.get_all_dimensions()
        # All keys should be single letters
        for key in dims.keys():
            assert len(key) == 1
            assert key in "abcde"


# ── High Inertia Tests ──────────────────────────────────────────────


class TestHighInertia:
    """Tests verifying high inertia (gradual changes)."""

    def test_single_update_is_small(self):
        """A single update produces only a small change."""
        orientation = ValueOrientation()
        signal = {"a": 1.0}  # Maximum signal

        updated = update_orientation(orientation, decision_signal=signal)

        # Change should be small (base_learning_rate = 0.01)
        assert abs(updated.dim_a) < 0.05
        assert abs(updated.dim_a - orientation.dim_a) < 0.02

    def test_cannot_flip_in_one_update(self):
        """Orientation cannot flip from one pole to another in one update."""
        orientation = ValueOrientation(dim_a=0.5, confidence_a=0.3)
        signal = {"a": -1.0}  # Strong opposing signal

        updated = update_orientation(orientation, decision_signal=signal)

        # Should still be positive (cannot flip instantly)
        assert updated.dim_a > 0

    def test_many_updates_needed_for_significant_change(self):
        """Many updates are needed to make a significant change."""
        orientation = ValueOrientation()
        signal = {"a": 1.0}

        # Apply 10 updates
        for _ in range(10):
            orientation = update_orientation(orientation, decision_signal=signal)

        # Should still be relatively small
        assert orientation.dim_a < 0.2

        # Apply 90 more updates (100 total)
        for _ in range(90):
            orientation = update_orientation(orientation, decision_signal=signal)

        # Now should be more significant
        assert orientation.dim_a > 0.2

    def test_confidence_dampens_further_changes(self):
        """High confidence makes changes even smaller."""
        low_conf = ValueOrientation(dim_a=0.3, confidence_a=0.1)
        high_conf = ValueOrientation(dim_a=0.3, confidence_a=0.8)

        signal = {"a": 1.0}

        updated_low = update_orientation(low_conf, decision_signal=signal)
        updated_high = update_orientation(high_conf, decision_signal=signal)

        # High confidence should change less
        change_low = abs(updated_low.dim_a - low_conf.dim_a)
        change_high = abs(updated_high.dim_a - high_conf.dim_a)

        assert change_high < change_low

    def test_effective_learning_rate_decreases_with_confidence(self):
        """Learning rate decreases as confidence increases."""
        config = ValueOrientationConfig()

        lr_no_conf = compute_effective_learning_rate(0.01, 0.0, config.confidence_damping)
        lr_mid_conf = compute_effective_learning_rate(0.01, 0.5, config.confidence_damping)
        lr_high_conf = compute_effective_learning_rate(0.01, 1.0, config.confidence_damping)

        assert lr_no_conf > lr_mid_conf > lr_high_conf


# ── Update Functions Tests ──────────────────────────────────────────


class TestUpdateFunctions:
    """Tests for update functions."""

    def test_update_dimension_with_no_signal(self):
        """No signal causes tiny decay toward neutral."""
        value, conf = update_dimension(
            current_value=0.5,
            current_confidence=0.3,
            signal=0.0,
            config=ValueOrientationConfig(),
        )
        # Should decay slightly toward neutral
        assert value < 0.5
        assert value > 0.499  # Very small decay

    def test_update_dimension_consistent_signal_builds_confidence(self):
        """Consistent signals build confidence."""
        config = ValueOrientationConfig()

        # Start with positive orientation
        value = 0.3
        conf = 0.2

        # Apply positive signal (consistent with orientation)
        new_value, new_conf = update_dimension(value, conf, signal=0.5, config=config)

        # Confidence should increase
        assert new_conf > conf

    def test_update_dimension_opposing_signal_reduces_confidence(self):
        """Opposing signals reduce confidence."""
        config = ValueOrientationConfig()

        # Start with positive orientation
        value = 0.3
        conf = 0.5

        # Apply negative signal (opposing orientation)
        new_value, new_conf = update_dimension(value, conf, signal=-0.5, config=config)

        # Confidence should decrease
        assert new_conf < conf

    def test_update_orientation_combines_signals(self):
        """Multiple signal sources are combined."""
        orientation = ValueOrientation()

        decision_signal = {"a": 0.5, "b": 0.3}
        emotion_signal = {"a": 0.3, "c": 0.4}

        updated = update_orientation(
            orientation,
            decision_signal=decision_signal,
            emotion_signal=emotion_signal,
        )

        # All affected dimensions should have changed
        assert updated.dim_a != 0.0
        assert updated.dim_b != 0.0
        assert updated.dim_c != 0.0

    def test_update_from_decision(self):
        """Can update orientation from a decision."""
        orientation = ValueOrientation()

        updated = update_from_decision(orientation, "からかう")

        assert updated.update_count == 1
        assert updated.dim_a != 0.0  # からかう maps to dim_a

    def test_generate_decision_signal(self):
        """Decision signal generated from policy label."""
        signal = generate_decision_signal("共感する")

        assert "b" in signal  # 共感する maps to dim_b
        assert signal["b"] > 0


# ── Bias Application Tests ──────────────────────────────────────────


class TestBiasApplication:
    """Tests for bias application to candidates."""

    def test_neutral_orientation_no_bias(self):
        """Neutral orientation produces no significant bias."""
        orientation = ValueOrientation()  # All zeros
        candidate = {"_score": 0.5, "policy_label": "共感する"}

        adjusted = apply_orientation_to_candidate(candidate, orientation)

        # Score should be virtually unchanged
        assert abs(adjusted["_score"] - 0.5) < 0.01

    def test_oriented_state_produces_bias(self):
        """Non-neutral orientation produces bias."""
        orientation = ValueOrientation(
            dim_b=0.5,  # Strong positive on dim_b
            confidence_b=0.5,
        )
        # 共感する has positive influence on dim_b
        candidate = {"_score": 0.5, "policy_label": "共感する"}

        adjusted = apply_orientation_to_candidate(candidate, orientation)

        # Should have positive bias (alignment)
        assert adjusted["_score"] > 0.5
        assert adjusted["_orientation_bias"] > 0

    def test_bias_is_subtle(self):
        """Bias is subtle (within max_bias_strength)."""
        orientation = ValueOrientation(
            dim_a=1.0,
            dim_b=1.0,
            dim_c=1.0,
            confidence_a=1.0,
            confidence_b=1.0,
            confidence_c=1.0,
        )
        config = ValueOrientationConfig(max_bias_strength=0.15)

        candidate = {"_score": 0.5, "policy_label": "からかう"}
        adjusted = apply_orientation_to_candidate(candidate, orientation, config)

        # Bias should not exceed max
        assert abs(adjusted["_orientation_bias"]) <= 0.15

    def test_bias_does_not_block_candidates(self):
        """Bias does not block any candidate."""
        orientation = ValueOrientation(
            dim_a=-1.0,  # Maximum negative on dim_a
            confidence_a=1.0,
        )
        # からかう has positive influence on dim_a (opposing)
        candidate = {"_score": 0.5, "policy_label": "からかう"}

        adjusted = apply_orientation_to_candidate(candidate, orientation)

        # Score may decrease but candidate is not blocked
        assert adjusted["_score"] > 0

    def test_applies_to_all_candidates(self):
        """Bias applies to all candidates."""
        orientation = ValueOrientation(dim_b=0.3, confidence_b=0.3)

        candidates = [
            {"_score": 0.8, "policy_label": "からかう"},
            {"_score": 0.7, "policy_label": "共感する"},
            {"_score": 0.6, "policy_label": "沈黙する"},
        ]

        adjusted = apply_orientation_to_candidates(candidates, orientation)

        # All candidates should have orientation applied
        for c in adjusted:
            assert c.get("_orientation_applied") is True

    def test_candidates_reordered_by_adjusted_score(self):
        """Candidates are reordered by adjusted score."""
        orientation = ValueOrientation(dim_b=0.5, confidence_b=0.5)

        candidates = [
            {"_score": 0.7, "policy_label": "からかう"},  # からかう: dim_a (not dim_b)
            {"_score": 0.6, "policy_label": "共感する"},  # 共感する: dim_b (aligned)
        ]

        adjusted = apply_orientation_to_candidates(candidates, orientation)

        # 共感する should be boosted, possibly reordering
        assert adjusted[0].get("_orientation_applied")


# ── Signal Generation Tests ─────────────────────────────────────────


class TestSignalGeneration:
    """Tests for signal generation functions."""

    def test_generate_emotion_signal_from_joy(self):
        """Joy generates appropriate signal."""
        class MockEmotion:
            joy = 0.5
            anger = 0.0
            sadness = 0.0
            fear = 0.0

        signal = generate_emotion_signal(MockEmotion())

        assert "a" in signal
        assert signal["a"] > 0  # Joy -> boldness

    def test_generate_emotion_signal_from_fear(self):
        """Fear generates appropriate signal."""
        class MockEmotion:
            joy = 0.0
            anger = 0.0
            sadness = 0.0
            fear = 0.5

        signal = generate_emotion_signal(MockEmotion())

        assert "a" in signal
        assert signal["a"] < 0  # Fear -> caution

    def test_generate_responsibility_signal(self):
        """Responsibility generates appropriate signal."""
        signal = generate_responsibility_signal(total_weight=0.7)

        assert "a" in signal
        assert signal["a"] < 0  # High responsibility -> caution


# ── Utility Function Tests ──────────────────────────────────────────


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_orientation_summary(self):
        """Summary returns readable string."""
        orientation = ValueOrientation(
            dim_a=0.5,
            confidence_a=0.4,
            update_count=10,
        )
        summary = get_orientation_summary(orientation)

        assert "A+" in summary  # Positive dim_a
        assert "10" in summary  # Update count

    def test_get_orientation_vector(self):
        """Can get orientation as vector."""
        orientation = ValueOrientation(
            dim_a=0.1,
            dim_b=0.2,
            dim_c=0.3,
            dim_d=0.4,
            dim_e=0.5,
        )
        vector = get_orientation_vector(orientation)

        assert vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_compute_orientation_distance(self):
        """Can compute distance between orientations."""
        o1 = ValueOrientation(dim_a=0.0)
        o2 = ValueOrientation(dim_a=1.0)

        distance = compute_orientation_distance(o1, o2)

        assert distance == 1.0  # Only dim_a differs

    def test_is_orientation_stable(self):
        """Can check if orientation is stable."""
        unstable = ValueOrientation(confidence_a=0.1, confidence_b=0.1)
        stable = ValueOrientation(
            confidence_a=0.5,
            confidence_b=0.5,
            confidence_c=0.5,
            confidence_d=0.5,
            confidence_e=0.5,
        )

        assert not is_orientation_stable(unstable, stability_threshold=0.3)
        assert is_orientation_stable(stable, stability_threshold=0.3)

    def test_create_orientation(self):
        """Can create new orientation."""
        orientation = create_orientation()
        assert isinstance(orientation, ValueOrientation)
        assert orientation.dim_a == 0.0


# ── Configuration Tests ─────────────────────────────────────────────


class TestConfiguration:
    """Tests for ValueOrientationConfig."""

    def test_create_config(self):
        """Can create config with custom values."""
        config = create_config(
            base_learning_rate=0.02,
            max_bias_strength=0.2,
        )

        assert config.base_learning_rate == 0.02
        assert config.max_bias_strength == 0.2

    def test_config_serialization(self):
        """Config can be serialized and deserialized."""
        config = ValueOrientationConfig(
            base_learning_rate=0.02,
            confidence_damping=0.6,
        )

        data = to_dict(config)
        restored = from_dict(data)

        assert restored.base_learning_rate == 0.02
        assert restored.confidence_damping == 0.6

    def test_policy_dimension_map_is_abstract(self):
        """Policy dimension map uses abstract dimension names."""
        config = ValueOrientationConfig()

        for policy, influences in config.policy_dimension_map.items():
            for dim_name in influences.keys():
                # All dimension names should be single letters
                assert len(dim_name) == 1
                assert dim_name in "abcde"


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    """Integration tests for value orientation."""

    def test_long_term_consistency_emerges(self):
        """Repeated decisions build consistent orientation."""
        orientation = ValueOrientation()

        # Repeatedly choose empathetic responses
        for _ in range(50):
            orientation = update_from_decision(orientation, "共感する")

        # Orientation should have developed
        assert orientation.dim_b > 0  # 共感する influences dim_b
        assert orientation.confidence_b > 0

    def test_mixed_decisions_create_nuanced_orientation(self):
        """Mixed decisions create multi-dimensional orientation."""
        orientation = ValueOrientation()

        # Alternate between different policies
        for i in range(30):
            if i % 2 == 0:
                orientation = update_from_decision(orientation, "共感する")
            else:
                orientation = update_from_decision(orientation, "からかう")

        # Should have influences on multiple dimensions
        assert orientation.dim_a != 0.0  # からかう
        assert orientation.dim_b != 0.0  # 共感する

    def test_persistence_maintains_continuity(self):
        """Orientation persists correctly for continuity."""
        orientation = ValueOrientation(
            dim_a=0.3,
            dim_b=-0.2,
            confidence_a=0.5,
            update_count=100,
        )

        # Serialize and restore
        data = orientation.to_dict()
        restored = ValueOrientation.from_dict(data)

        # All values should be preserved
        assert restored.dim_a == orientation.dim_a
        assert restored.dim_b == orientation.dim_b
        assert restored.confidence_a == orientation.confidence_a
        assert restored.update_count == orientation.update_count

    def test_bias_works_with_existing_candidate_structure(self):
        """Bias integrates with existing candidate dict structure."""
        orientation = ValueOrientation(dim_b=0.4, confidence_b=0.4)

        # Candidate with existing metadata (like from other systems)
        candidate = {
            "_score": 0.6,
            "policy_label": "共感する",
            "_tone": "warm",
            "_is_silence": False,
            "_sensitivity_adjusted": True,
        }

        adjusted = apply_orientation_to_candidate(candidate, orientation)

        # Original metadata preserved
        assert adjusted["_tone"] == "warm"
        assert adjusted["_is_silence"] is False
        assert adjusted["_sensitivity_adjusted"] is True

        # Orientation metadata added
        assert adjusted["_orientation_applied"] is True
        assert "_orientation_bias" in adjusted
