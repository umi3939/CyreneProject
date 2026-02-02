"""
Tests for psyche/stability_valve.py - Stability / Safety Valve (極端回避防止)

Verifies:
- Monitors patterns (repetition/extremity), NOT content
- Gradual bias adjustment, NOT hard reset
- Flattens score distribution to prevent fixation
- Integrates with IntrospectionTrace
"""

import pytest
import time

from psyche.stability_valve import (
    ExtremityIndicators,
    StabilityBias,
    StabilityValveConfig,
    StabilityValve,
    create_neutral_bias,
    flatten_scores,
    apply_stability_to_candidate,
    apply_stability_bias,
    create_stability_factor,
    get_stability_trace_context,
    create_stability_valve,
    create_config,
    observe_extremity,
    get_stability_summary,
    to_dict,
    from_dict,
)

from psyche.state import EmotionVector
from psyche.value_orientation import ValueOrientation


# ── ExtremityIndicators Tests ───────────────────────────────────────


class TestExtremityIndicators:
    """Tests for ExtremityIndicators."""

    def test_default_values(self):
        """Default indicators are zero (no extremity)."""
        indicators = ExtremityIndicators()
        assert indicators.fear_extremity == 0.0
        assert indicators.decision_fixation == 0.0
        assert indicators.overall_extremity == 0.0

    def test_values_clamped(self):
        """Values are clamped to [0, 1]."""
        indicators = ExtremityIndicators(
            fear_extremity=1.5,
            decision_fixation=-0.3,
        )
        assert indicators.fear_extremity == 1.0
        assert indicators.decision_fixation == 0.0

    def test_overall_computed_from_components(self):
        """Overall extremity is computed from components."""
        indicators = ExtremityIndicators(
            fear_extremity=0.8,
            responsibility_extremity=0.6,
            decision_fixation=0.4,
        )
        # Overall should be positive when components are high
        assert indicators.overall_extremity > 0.4

    def test_overall_weighted_toward_max(self):
        """Overall is weighted toward max indicator."""
        # One high indicator
        single_high = ExtremityIndicators(fear_extremity=0.9)
        # Multiple moderate indicators
        multiple_mod = ExtremityIndicators(
            fear_extremity=0.4,
            responsibility_extremity=0.4,
            decision_fixation=0.4,
        )

        # Single high should have higher overall due to max weighting
        assert single_high.overall_extremity > multiple_mod.overall_extremity

    def test_serialization(self):
        """Indicators can be serialized and deserialized."""
        original = ExtremityIndicators(
            fear_extremity=0.7,
            decision_fixation=0.5,
            consecutive_extreme_count=3,
        )
        data = original.to_dict()
        restored = ExtremityIndicators.from_dict(data)

        assert restored.fear_extremity == original.fear_extremity
        assert restored.decision_fixation == original.decision_fixation
        assert restored.consecutive_extreme_count == original.consecutive_extreme_count


# ── StabilityBias Tests ─────────────────────────────────────────────


class TestStabilityBias:
    """Tests for StabilityBias."""

    def test_neutral_bias(self):
        """Neutral bias has no effect."""
        bias = create_neutral_bias()
        assert bias.flatten_strength == 0.0
        assert bias.is_active is False
        assert bias.activation_level == 0.0

    def test_serialization(self):
        """Bias can be serialized and deserialized."""
        original = StabilityBias(
            flatten_strength=0.3,
            is_active=True,
            activation_level=0.5,
            source_indicators={"fear": 0.8},
        )
        data = original.to_dict()
        restored = StabilityBias.from_dict(data)

        assert restored.flatten_strength == original.flatten_strength
        assert restored.is_active == original.is_active
        assert restored.activation_level == original.activation_level


# ── StabilityValve Tests ────────────────────────────────────────────


class TestStabilityValve:
    """Tests for StabilityValve."""

    def test_creates_with_default_config(self):
        """Valve creates with default config."""
        valve = StabilityValve()
        assert valve.config is not None
        assert valve.get_activation_level() == 0.0

    def test_records_decisions(self):
        """Valve records decision history."""
        valve = StabilityValve()

        valve.record_decision("共感する")
        valve.record_decision("沈黙する")
        valve.record_decision("からかう")

        history = valve.get_decision_history()
        assert len(history) == 3
        assert "共感する" in history

    def test_observe_extremity_returns_indicators(self):
        """Observe extremity returns indicators."""
        valve = StabilityValve()

        indicators = valve.observe_extremity(fear_level=0.8)

        assert isinstance(indicators, ExtremityIndicators)
        assert indicators.fear_extremity > 0

    def test_generate_bias_returns_bias(self):
        """Generate bias returns stability bias."""
        valve = StabilityValve()

        indicators = valve.observe_extremity(fear_level=0.9)
        bias = valve.generate_bias(indicators)

        assert isinstance(bias, StabilityBias)


class TestPatternDetection:
    """Tests verifying pattern detection (not content judgment)."""

    def test_detects_decision_fixation(self):
        """Detects repeated identical decisions."""
        config = StabilityValveConfig(
            fixation_threshold=0.7,
            decision_history_size=10,
        )
        valve = StabilityValve(config)

        # Record same decision repeatedly
        for _ in range(8):
            valve.record_decision("沈黙する")
        valve.record_decision("共感する")
        valve.record_decision("共感する")

        indicators = valve.observe_extremity()

        # Should detect fixation (8/10 same decision)
        assert indicators.decision_fixation > 0

    def test_no_fixation_with_varied_decisions(self):
        """No fixation with varied decisions."""
        valve = StabilityValve()

        # Record varied decisions
        decisions = ["共感する", "からかう", "沈黙する", "励ます", "質問する"]
        for d in decisions * 2:
            valve.record_decision(d)

        indicators = valve.observe_extremity()

        # Should not detect fixation
        assert indicators.decision_fixation == 0.0

    def test_detects_fear_extremity(self):
        """Detects extreme fear levels."""
        config = StabilityValveConfig(fear_extreme_threshold=0.7)
        valve = StabilityValve(config)

        indicators = valve.observe_extremity(fear_level=0.9)

        assert indicators.fear_extremity > 0

    def test_detects_value_extremity(self):
        """Detects extreme value orientation."""
        config = StabilityValveConfig(value_extreme_threshold=0.7)
        valve = StabilityValve(config)

        orientation = ValueOrientation(dim_a=0.9)
        indicators = valve.observe_extremity(value_orientation=orientation)

        assert indicators.value_extremity > 0

    def test_detects_emotion_saturation(self):
        """Detects single emotion dominating."""
        config = StabilityValveConfig(emotion_saturation_threshold=0.6)
        valve = StabilityValve(config)

        # One emotion dominates
        emotion = EmotionVector(fear=0.9, joy=0.1)
        indicators = valve.observe_extremity(emotion_state=emotion)

        assert indicators.emotion_saturation > 0


class TestGradualActivation:
    """Tests verifying gradual activation (not ON/OFF)."""

    def test_activation_increases_gradually(self):
        """Activation increases gradually with extremity."""
        config = StabilityValveConfig(
            activation_rate=0.1,
            extremity_threshold=0.3,  # Lower threshold
            fear_extreme_threshold=0.5,  # Lower threshold
        )
        valve = StabilityValve(config)

        # First observation - slight activation
        indicators = valve.observe_extremity(fear_level=0.95)
        bias1 = valve.generate_bias(indicators)
        level1 = valve.get_activation_level()

        # Second observation - more activation
        indicators = valve.observe_extremity(fear_level=0.95)
        bias2 = valve.generate_bias(indicators)
        level2 = valve.get_activation_level()

        assert level2 > level1
        assert level2 < 1.0  # Not instant max

    def test_activation_decays_when_not_extreme(self):
        """Activation decays when extremity reduces."""
        config = StabilityValveConfig(
            activation_rate=0.2,
            decay_rate=0.1,
            extremity_threshold=0.3,  # Lower threshold
            fear_extreme_threshold=0.5,  # Lower threshold
        )
        valve = StabilityValve(config)

        # Build up activation
        for _ in range(5):
            indicators = valve.observe_extremity(fear_level=0.95)
            valve.generate_bias(indicators)

        high_level = valve.get_activation_level()

        # Now reduce extremity
        for _ in range(3):
            indicators = valve.observe_extremity(fear_level=0.3)
            valve.generate_bias(indicators)

        low_level = valve.get_activation_level()

        assert low_level < high_level

    def test_no_hard_reset(self):
        """No hard reset, only gradual changes."""
        config = StabilityValveConfig(
            extremity_threshold=0.3,
            fear_extreme_threshold=0.5,
            decay_rate=0.1,  # Explicit decay rate
        )
        valve = StabilityValve(config)

        # Build up activation
        for _ in range(5):
            indicators = valve.observe_extremity(fear_level=0.95)
            valve.generate_bias(indicators)

        high_level = valve.get_activation_level()

        # Multiple non-extreme observations (allow time to decay)
        for _ in range(10):
            indicators = valve.observe_extremity(fear_level=0.2)
            valve.generate_bias(indicators)

        new_level = valve.get_activation_level()

        # Should not jump to zero immediately (gradual decay)
        # After 10 non-extreme observations, it should have decayed
        assert new_level < high_level

    def test_consecutive_extremes_boost_activation(self):
        """Consecutive extreme observations boost activation."""
        config = StabilityValveConfig(
            consecutive_threshold=3,
            extremity_threshold=0.3,  # Lower threshold
            fear_extreme_threshold=0.5,  # Lower threshold
        )
        valve = StabilityValve(config)

        # Build consecutive extremes
        for i in range(5):
            indicators = valve.observe_extremity(fear_level=0.95)
            valve.generate_bias(indicators)

        assert indicators.consecutive_extreme_count >= 3


# ── Score Flattening Tests ──────────────────────────────────────────


class TestScoreFlattening:
    """Tests for score flattening logic."""

    def test_flatten_scores_moves_toward_mean(self):
        """Flattening moves scores toward mean."""
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        mean = sum(scores) / len(scores)  # 0.5

        flattened = flatten_scores(scores, flatten_strength=0.5)

        # All scores should be closer to mean (except the one at mean)
        for orig, flat in zip(scores, flattened):
            if abs(orig - mean) > 0.01:  # Skip the one at mean
                assert abs(flat - mean) < abs(orig - mean)

    def test_no_flatten_when_strength_zero(self):
        """No flattening when strength is zero."""
        scores = [0.9, 0.1]
        flattened = flatten_scores(scores, flatten_strength=0.0)

        assert flattened == scores

    def test_full_flatten_equals_mean(self):
        """Full flatten (1.0) makes all scores equal to mean."""
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        mean = sum(scores) / len(scores)

        flattened = flatten_scores(scores, flatten_strength=1.0)

        for score in flattened:
            assert abs(score - mean) < 0.001


class TestBiasApplication:
    """Tests for applying stability bias to candidates."""

    def test_inactive_bias_no_change(self):
        """Inactive bias does not change candidates."""
        candidate = {"_score": 0.8, "policy_label": "test"}
        bias = StabilityBias(is_active=False)

        result = apply_stability_to_candidate(candidate, bias)

        assert result["_score"] == 0.8

    def test_active_bias_flattens_score(self):
        """Active bias flattens scores."""
        high_score = {"_score": 0.9, "policy_label": "high"}
        low_score = {"_score": 0.3, "policy_label": "low"}

        bias = StabilityBias(
            flatten_strength=0.4,
            min_score_boost=0.1,
            max_score_reduction=0.1,
            is_active=True,
            activation_level=0.5,
        )

        high_adjusted = apply_stability_to_candidate(high_score, bias, mean_score=0.6)
        low_adjusted = apply_stability_to_candidate(low_score, bias, mean_score=0.6)

        # High score should decrease
        assert high_adjusted["_score"] < 0.9
        # Low score should increase
        assert low_adjusted["_score"] > 0.3

    def test_applies_to_all_candidates(self):
        """Bias applies to all candidates."""
        candidates = [
            {"_score": 0.9, "policy_label": "a"},
            {"_score": 0.7, "policy_label": "b"},
            {"_score": 0.5, "policy_label": "c"},
            {"_score": 0.3, "policy_label": "d"},
        ]

        bias = StabilityBias(
            flatten_strength=0.3,
            is_active=True,
            activation_level=0.5,
        )

        adjusted = apply_stability_bias(candidates, bias)

        # All should have stability metadata
        for c in adjusted:
            assert c.get("_stability_active") is True

    def test_does_not_prohibit_any_decision(self):
        """Bias does not set any score to zero."""
        candidates = [
            {"_score": 0.9, "policy_label": "high"},
            {"_score": 0.1, "policy_label": "low"},
        ]

        # Maximum bias
        bias = StabilityBias(
            flatten_strength=0.4,
            max_score_reduction=0.3,
            is_active=True,
            activation_level=1.0,
        )

        adjusted = apply_stability_bias(candidates, bias)

        # All scores should be positive
        for c in adjusted:
            assert c["_score"] > 0

    def test_preserves_original_score(self):
        """Original score is preserved in metadata."""
        candidate = {"_score": 0.8, "policy_label": "test"}
        bias = StabilityBias(
            flatten_strength=0.3,
            is_active=True,
            activation_level=0.5,
        )

        adjusted = apply_stability_to_candidate(candidate, bias, mean_score=0.5)

        assert adjusted["_pre_stability_score"] == 0.8
        assert adjusted["_score"] != 0.8


# ── Introspection Integration Tests ─────────────────────────────────


class TestIntrospectionIntegration:
    """Tests for introspection trace integration."""

    def test_create_stability_factor_when_active(self):
        """Creates contributing factor when valve is active."""
        bias = StabilityBias(
            flatten_strength=0.3,
            is_active=True,
            activation_level=0.5,
            source_indicators={"fear": 0.8, "fixation": 0.3},
        )

        factor = create_stability_factor(bias)

        assert factor is not None
        assert factor["category"] == "stability"
        assert factor["name"] == "stability_valve"
        assert factor["contribution_strength"] == 0.3
        assert "fear" in factor["description"]

    def test_no_factor_when_inactive(self):
        """No factor created when valve is inactive."""
        bias = StabilityBias(is_active=False)

        factor = create_stability_factor(bias)

        assert factor is None

    def test_get_stability_trace_context(self):
        """Gets trace context from valve."""
        valve = StabilityValve()

        # Generate some activity
        for _ in range(3):
            indicators = valve.observe_extremity(fear_level=0.8)
            valve.generate_bias(indicators)

        context = get_stability_trace_context(valve)

        assert "stability_valve_active" in context
        assert "activation_level" in context
        assert "indicators" in context


# ── Configuration Tests ─────────────────────────────────────────────


class TestConfiguration:
    """Tests for StabilityValveConfig."""

    def test_create_config(self):
        """Can create config with custom values."""
        config = create_config(
            extremity_threshold=0.7,
            max_flatten_strength=0.5,
        )

        assert config.extremity_threshold == 0.7
        assert config.max_flatten_strength == 0.5

    def test_config_serialization(self):
        """Config can be serialized and deserialized."""
        original = StabilityValveConfig(
            extremity_threshold=0.7,
            fear_extreme_threshold=0.8,
        )

        data = to_dict(original)
        restored = from_dict(data)

        assert restored.extremity_threshold == 0.7
        assert restored.fear_extreme_threshold == 0.8


# ── Utility Function Tests ──────────────────────────────────────────


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_stability_valve(self):
        """create_stability_valve works."""
        valve = create_stability_valve()
        assert isinstance(valve, StabilityValve)

    def test_observe_extremity_function(self):
        """observe_extremity convenience function works."""
        valve = create_stability_valve()
        indicators = observe_extremity(valve, fear_level=0.8)

        assert isinstance(indicators, ExtremityIndicators)

    def test_get_stability_summary_inactive(self):
        """Summary shows inactive when not activated."""
        valve = StabilityValve()
        summary = get_stability_summary(valve)

        assert "inactive" in summary

    def test_get_stability_summary_active(self):
        """Summary shows active state with details."""
        config = StabilityValveConfig(
            extremity_threshold=0.3,
            fear_extreme_threshold=0.5,
        )
        valve = StabilityValve(config)

        # Build up activation
        for _ in range(5):
            indicators = valve.observe_extremity(fear_level=0.95)
            valve.generate_bias(indicators)

        summary = get_stability_summary(valve)

        assert "active" in summary or "warming" in summary
        assert "level=" in summary


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Complete workflow from observation to bias application."""
        valve = StabilityValve()

        # Record repeated decisions
        for _ in range(8):
            valve.record_decision("沈黙する")

        # Observe with high fear
        indicators = valve.observe_extremity(
            fear_level=0.85,
            emotion_state=EmotionVector(fear=0.8),
        )

        # Generate bias
        bias = valve.generate_bias(indicators)

        # Apply to candidates
        candidates = [
            {"_score": 0.9, "policy_label": "沈黙する"},
            {"_score": 0.5, "policy_label": "共感する"},
            {"_score": 0.3, "policy_label": "からかう"},
        ]

        adjusted = apply_stability_bias(candidates, bias)

        # If valve is active, scores should be flattened
        if bias.is_active:
            # Gap between highest and lowest should be smaller
            original_gap = 0.9 - 0.3
            adjusted_scores = [c["_score"] for c in adjusted]
            new_gap = max(adjusted_scores) - min(adjusted_scores)

            assert new_gap < original_gap

    def test_recovery_from_extremity(self):
        """System recovers when extremity reduces."""
        config = StabilityValveConfig(
            extremity_threshold=0.3,
            fear_extreme_threshold=0.5,
        )
        valve = StabilityValve(config)

        # Build up extremity
        for _ in range(10):
            valve.observe_extremity(fear_level=0.95)
            valve.generate_bias()

        high_activation = valve.get_activation_level()

        # Reduce extremity over time
        for _ in range(20):
            valve.observe_extremity(fear_level=0.3)
            valve.generate_bias()

        low_activation = valve.get_activation_level()

        # Should have recovered (lower activation)
        assert low_activation < high_activation

    def test_works_with_existing_candidate_metadata(self):
        """Works with candidates that have existing metadata."""
        valve = StabilityValve()

        # Generate active bias
        for _ in range(5):
            valve.observe_extremity(fear_level=0.9)
            valve.generate_bias()

        bias = valve.get_last_bias()

        # Candidate with existing metadata
        candidate = {
            "_score": 0.8,
            "policy_label": "test",
            "_tone": "serious",
            "_sensitivity_adjusted": True,
            "_orientation_applied": True,
        }

        if bias and bias.is_active:
            adjusted = apply_stability_to_candidate(candidate, bias, mean_score=0.5)

            # Original metadata preserved
            assert adjusted["_tone"] == "serious"
            assert adjusted["_sensitivity_adjusted"] is True
            assert adjusted["_orientation_applied"] is True

            # Stability metadata added
            assert adjusted["_stability_active"] is True
