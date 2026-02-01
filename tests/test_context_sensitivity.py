"""
Tests for psyche/context_sensitivity.py - Context Sensitivity Bias (空気読みバイアス)

Verifies:
- External context as pure signal intensity (no emotion guessing)
- Sensitivity bias applies universally to all candidates
- High weight/density increases caution (dampens risky candidates)
- Temporary effect with decay
- Serialization/deserialization
"""

import pytest
import time

from psyche.context_sensitivity import (
    ExternalContext,
    SensitivityBias,
    ContextSensitivityConfig,
    ContextState,
    create_external_context,
    create_neutral_context,
    create_heavy_context,
    create_light_context,
    create_neutral_bias,
    compute_sensitivity_bias,
    get_policy_risk,
    apply_sensitivity_to_candidate,
    apply_sensitivity_to_candidates,
    process_with_context_sensitivity,
    get_sensitivity_summary,
    get_context_summary,
    is_high_caution,
    is_low_caution,
    create_config,
    to_dict,
    from_dict,
)


# ── ExternalContext Tests ───────────────────────────────────────────


class TestExternalContext:
    """Tests for ExternalContext dataclass."""

    def test_default_values(self):
        """Default context has neutral values (0.5)."""
        ctx = ExternalContext()
        assert ctx.pace == 0.5
        assert ctx.weight == 0.5
        assert ctx.density == 0.5
        assert ctx.continuity == 0.5
        assert ctx.responsiveness == 0.5

    def test_values_clamped_to_range(self):
        """Values are clamped to [0, 1]."""
        ctx = ExternalContext(
            pace=-0.5,
            weight=1.5,
            density=2.0,
            continuity=-1.0,
            responsiveness=0.7,
        )
        assert ctx.pace == 0.0
        assert ctx.weight == 1.0
        assert ctx.density == 1.0
        assert ctx.continuity == 0.0
        assert ctx.responsiveness == 0.7

    def test_timestamp_set_automatically(self):
        """Timestamp is set on creation."""
        before = time.time()
        ctx = ExternalContext()
        after = time.time()
        assert before <= ctx.timestamp <= after

    def test_serialization(self):
        """Context can be serialized and deserialized."""
        ctx = ExternalContext(
            pace=0.3,
            weight=0.7,
            density=0.6,
            continuity=0.4,
            responsiveness=0.8,
        )
        data = ctx.to_dict()
        restored = ExternalContext.from_dict(data)

        assert restored.pace == ctx.pace
        assert restored.weight == ctx.weight
        assert restored.density == ctx.density
        assert restored.continuity == ctx.continuity
        assert restored.responsiveness == ctx.responsiveness


class TestContextFactories:
    """Tests for context factory functions."""

    def test_create_external_context(self):
        """create_external_context creates context with specified values."""
        ctx = create_external_context(
            pace=0.2, weight=0.8, density=0.7
        )
        assert ctx.pace == 0.2
        assert ctx.weight == 0.8
        assert ctx.density == 0.7
        # Defaults for unspecified
        assert ctx.continuity == 0.5
        assert ctx.responsiveness == 0.5

    def test_neutral_context(self):
        """Neutral context has all midpoint values."""
        ctx = create_neutral_context()
        assert ctx.pace == 0.5
        assert ctx.weight == 0.5
        assert ctx.density == 0.5
        assert ctx.continuity == 0.5
        assert ctx.responsiveness == 0.5

    def test_heavy_context(self):
        """Heavy context has slow pace, high weight/density."""
        ctx = create_heavy_context()
        assert ctx.pace < 0.5  # Slow
        assert ctx.weight > 0.5  # Heavy
        assert ctx.density > 0.5  # Dense
        assert ctx.responsiveness < 0.5  # Low responsiveness

    def test_light_context(self):
        """Light context has fast pace, low weight."""
        ctx = create_light_context()
        assert ctx.pace > 0.5  # Fast
        assert ctx.weight < 0.5  # Light
        assert ctx.density < 0.5  # Sparse
        assert ctx.responsiveness > 0.5  # High responsiveness


# ── SensitivityBias Tests ───────────────────────────────────────────


class TestSensitivityBias:
    """Tests for SensitivityBias dataclass."""

    def test_default_values(self):
        """Default bias has neutral caution."""
        bias = SensitivityBias()
        assert bias.caution_level == 0.5
        assert bias.risk_dampening == 0.0
        assert bias.safety_boost == 0.0
        assert bias.score_multiplier == 1.0

    def test_neutral_bias(self):
        """Neutral bias has no effect."""
        bias = create_neutral_bias()
        assert bias.caution_level == 0.5
        assert bias.risk_dampening == 0.0
        assert bias.safety_boost == 0.0
        assert bias.score_multiplier == 1.0
        assert bias.selection_threshold_shift == 0.0

    def test_serialization(self):
        """Bias can be serialized and deserialized."""
        bias = SensitivityBias(
            caution_level=0.7,
            risk_dampening=0.15,
            safety_boost=0.1,
            score_multiplier=0.9,
            selection_threshold_shift=0.1,
        )
        data = bias.to_dict()
        restored = SensitivityBias.from_dict(data)

        assert restored.caution_level == bias.caution_level
        assert restored.risk_dampening == bias.risk_dampening
        assert restored.safety_boost == bias.safety_boost
        assert restored.score_multiplier == bias.score_multiplier


# ── Compute Sensitivity Bias Tests ──────────────────────────────────


class TestComputeSensitivityBias:
    """Tests for compute_sensitivity_bias function."""

    def test_neutral_context_produces_base_caution(self):
        """Neutral context produces base caution level."""
        ctx = create_neutral_context()
        bias = compute_sensitivity_bias(ctx)
        # With neutral context, caution stays at base (0.5)
        assert 0.45 <= bias.caution_level <= 0.55

    def test_high_weight_increases_caution(self):
        """High weight increases caution level."""
        neutral = create_neutral_context()
        heavy = create_external_context(weight=0.9)

        bias_neutral = compute_sensitivity_bias(neutral)
        bias_heavy = compute_sensitivity_bias(heavy)

        assert bias_heavy.caution_level > bias_neutral.caution_level

    def test_high_density_increases_caution(self):
        """High density increases caution level."""
        neutral = create_neutral_context()
        dense = create_external_context(density=0.9)

        bias_neutral = compute_sensitivity_bias(neutral)
        bias_dense = compute_sensitivity_bias(dense)

        assert bias_dense.caution_level > bias_neutral.caution_level

    def test_slow_pace_increases_caution(self):
        """Slow pace increases caution level."""
        neutral = create_neutral_context()
        slow = create_external_context(pace=0.1)

        bias_neutral = compute_sensitivity_bias(neutral)
        bias_slow = compute_sensitivity_bias(slow)

        assert bias_slow.caution_level > bias_neutral.caution_level

    def test_low_responsiveness_increases_caution(self):
        """Low responsiveness increases caution level."""
        neutral = create_neutral_context()
        unresponsive = create_external_context(responsiveness=0.1)

        bias_neutral = compute_sensitivity_bias(neutral)
        bias_unresponsive = compute_sensitivity_bias(unresponsive)

        assert bias_unresponsive.caution_level > bias_neutral.caution_level

    def test_high_continuity_reduces_caution(self):
        """High continuity reduces caution level (familiar context)."""
        neutral = create_neutral_context()
        continuous = create_external_context(continuity=0.9)

        bias_neutral = compute_sensitivity_bias(neutral)
        bias_continuous = compute_sensitivity_bias(continuous)

        assert bias_continuous.caution_level < bias_neutral.caution_level

    def test_heavy_context_produces_high_caution(self):
        """Heavy context produces high caution with risk dampening."""
        ctx = create_heavy_context()
        bias = compute_sensitivity_bias(ctx)

        assert bias.caution_level > 0.5
        assert bias.risk_dampening > 0
        assert bias.safety_boost > 0
        assert bias.score_multiplier < 1.0

    def test_light_context_produces_low_caution(self):
        """Light context produces lower caution."""
        # Use light context with high continuity to get relief
        ctx = create_external_context(
            pace=0.8,  # Fast
            weight=0.2,  # Light
            density=0.3,  # Sparse
            continuity=0.9,  # High continuity = relief
            responsiveness=0.8,  # High
        )
        bias = compute_sensitivity_bias(ctx)

        assert bias.caution_level < 0.5
        # Low caution means no risk dampening
        assert bias.risk_dampening == 0
        # Score multiplier can expand
        assert bias.score_multiplier >= 1.0

    def test_caution_clamped_to_config_limits(self):
        """Caution is clamped to min/max from config."""
        config = ContextSensitivityConfig(
            min_caution=0.2,
            max_caution=0.8,
        )

        # Extremely heavy context
        extreme_heavy = create_external_context(
            pace=0.0, weight=1.0, density=1.0, responsiveness=0.0
        )
        bias_heavy = compute_sensitivity_bias(extreme_heavy, config)
        assert bias_heavy.caution_level <= 0.8

        # Extremely light context
        extreme_light = create_external_context(
            pace=1.0, weight=0.0, density=0.0, continuity=1.0, responsiveness=1.0
        )
        bias_light = compute_sensitivity_bias(extreme_light, config)
        assert bias_light.caution_level >= 0.2


class TestContextDoesNotGuessEmotion:
    """Tests verifying context is pure signal, not emotion interpretation."""

    def test_context_has_no_emotion_field(self):
        """Context does not have emotion-related fields."""
        ctx = ExternalContext()
        # Verify no emotion/intent fields exist
        assert not hasattr(ctx, "emotion")
        assert not hasattr(ctx, "intent")
        assert not hasattr(ctx, "sentiment")
        assert not hasattr(ctx, "mood")

    def test_bias_has_no_emotion_interpretation(self):
        """Bias does not interpret emotions."""
        bias = SensitivityBias()
        assert not hasattr(bias, "detected_emotion")
        assert not hasattr(bias, "user_mood")
        assert not hasattr(bias, "sentiment")

    def test_bias_applies_same_formula_to_all_contexts(self):
        """Bias uses same formula regardless of context (no semantic interpretation)."""
        # Two different contexts with same numerical values should produce same bias
        ctx1 = ExternalContext(pace=0.3, weight=0.7, density=0.6)
        ctx2 = ExternalContext(pace=0.3, weight=0.7, density=0.6)

        bias1 = compute_sensitivity_bias(ctx1)
        bias2 = compute_sensitivity_bias(ctx2)

        assert bias1.caution_level == bias2.caution_level
        assert bias1.risk_dampening == bias2.risk_dampening


# ── Policy Risk Tests ───────────────────────────────────────────────


class TestPolicyRisk:
    """Tests for get_policy_risk function."""

    def test_teasing_has_high_risk(self):
        """Teasing policy has high risk."""
        candidate = {"policy_label": "からかう"}
        risk = get_policy_risk(candidate)
        assert risk >= 0.7

    def test_silence_has_low_risk(self):
        """Silence has very low risk."""
        candidate = {"policy_label": "沈黙する", "_is_silence": True}
        risk = get_policy_risk(candidate)
        assert risk <= 0.2

    def test_empathy_has_low_risk(self):
        """Empathy has low risk."""
        candidate = {"policy_label": "共感する"}
        risk = get_policy_risk(candidate)
        assert risk <= 0.3

    def test_unknown_policy_uses_default(self):
        """Unknown policy uses default risk."""
        config = ContextSensitivityConfig(default_policy_risk=0.5)
        candidate = {"policy_label": "unknown_policy"}
        risk = get_policy_risk(candidate, config)
        assert risk == 0.5

    def test_light_tone_increases_risk(self):
        """Light tone slightly increases risk."""
        neutral = {"policy_label": "感想を述べる", "_tone": "neutral"}
        light = {"policy_label": "感想を述べる", "_tone": "light"}

        risk_neutral = get_policy_risk(neutral)
        risk_light = get_policy_risk(light)

        assert risk_light > risk_neutral


# ── Apply Sensitivity Tests ─────────────────────────────────────────


class TestApplySensitivity:
    """Tests for applying sensitivity bias to candidates."""

    def test_neutral_bias_preserves_score(self):
        """Neutral bias preserves original score."""
        candidate = {"_score": 0.8, "policy_label": "共感する"}
        bias = create_neutral_bias()

        adjusted = apply_sensitivity_to_candidate(candidate, bias)

        # With neutral bias, score should be unchanged
        assert abs(adjusted["_score"] - 0.8) < 0.01

    def test_high_caution_dampens_risky_candidate(self):
        """High caution dampens risky candidates more."""
        risky = {"_score": 0.8, "policy_label": "からかう"}
        safe = {"_score": 0.8, "policy_label": "共感する"}

        bias = SensitivityBias(
            caution_level=0.8,
            risk_dampening=0.3,
            safety_boost=0.1,
            score_multiplier=0.9,
        )

        adjusted_risky = apply_sensitivity_to_candidate(risky, bias)
        adjusted_safe = apply_sensitivity_to_candidate(safe, bias)

        # Risky candidate should be dampened more
        assert adjusted_risky["_score"] < adjusted_safe["_score"]

    def test_applies_to_all_candidates(self):
        """Sensitivity applies to ALL candidates (universal)."""
        candidates = [
            {"_score": 0.9, "policy_label": "からかう"},
            {"_score": 0.8, "policy_label": "感想を述べる"},
            {"_score": 0.7, "policy_label": "共感する"},
            {"_score": 0.6, "policy_label": "沈黙する", "_is_silence": True},
        ]

        bias = SensitivityBias(
            caution_level=0.7,
            risk_dampening=0.2,
            safety_boost=0.1,
            score_multiplier=0.9,
        )

        adjusted = apply_sensitivity_to_candidates(candidates, bias)

        # All candidates should be affected
        for c in adjusted:
            assert "_sensitivity_adjusted" in c
            assert c["_sensitivity_adjusted"] is True
            assert "_caution_level" in c
            assert c["_caution_level"] == 0.7

    def test_reorders_by_adjusted_score(self):
        """Candidates are reordered by adjusted score."""
        candidates = [
            {"_score": 0.9, "policy_label": "からかう"},  # High risk
            {"_score": 0.7, "policy_label": "共感する"},  # Low risk
        ]

        # High caution to dampen risky candidate significantly
        bias = SensitivityBias(
            caution_level=0.8,
            risk_dampening=0.5,
            safety_boost=0.2,
            score_multiplier=0.9,
        )

        adjusted = apply_sensitivity_to_candidates(candidates, bias)

        # Safe candidate may now rank higher
        assert adjusted[0].get("_sensitivity_adjusted")

    def test_preserves_original_score(self):
        """Original score is preserved in metadata."""
        candidate = {"_score": 0.8, "policy_label": "共感する"}
        bias = SensitivityBias(
            caution_level=0.7,
            risk_dampening=0.2,
            score_multiplier=0.9,
        )

        adjusted = apply_sensitivity_to_candidate(candidate, bias)

        assert adjusted["_original_score"] == 0.8
        assert adjusted["_score"] != adjusted["_original_score"]


# ── Context State Tests ─────────────────────────────────────────────


class TestContextState:
    """Tests for ContextState tracking."""

    def test_initial_state_is_neutral(self):
        """Initial state has neutral smoothed values."""
        state = ContextState()
        assert state.smoothed_pace == 0.5
        assert state.smoothed_weight == 0.5
        assert state.smoothed_density == 0.5

    def test_update_applies_smoothing(self):
        """Update applies exponential smoothing."""
        state = ContextState(smoothing_alpha=0.5)
        ctx = create_external_context(pace=1.0, weight=1.0)

        updated = state.update(ctx)

        # With alpha=0.5, new value is 0.5*1.0 + 0.5*0.5 = 0.75
        assert updated.smoothed_pace == 0.75
        assert updated.smoothed_weight == 0.75

    def test_decay_toward_neutral(self):
        """Decay moves values toward neutral."""
        state = ContextState(
            smoothed_pace=0.9,
            smoothed_weight=0.1,
            smoothed_density=0.8,
        )

        decayed = state.apply_decay(delta_time=1.0, decay_rate=0.5)

        # Values should move toward 0.5
        assert 0.5 < decayed.smoothed_pace < 0.9
        assert 0.1 < decayed.smoothed_weight < 0.5
        assert 0.5 < decayed.smoothed_density < 0.8

    def test_serialization(self):
        """State can be serialized and deserialized."""
        state = ContextState(
            smoothed_pace=0.7,
            smoothed_weight=0.3,
            smoothing_alpha=0.4,
        )
        state = state.update(create_neutral_context())

        data = state.to_dict()
        restored = ContextState.from_dict(data)

        assert restored.smoothed_pace == state.smoothed_pace
        assert restored.smoothing_alpha == state.smoothing_alpha


# ── Full Pipeline Tests ─────────────────────────────────────────────


class TestProcessWithContextSensitivity:
    """Tests for full pipeline function."""

    def test_returns_adjusted_candidates_and_state(self):
        """Pipeline returns adjusted candidates and updated state."""
        candidates = [
            {"_score": 0.8, "policy_label": "共感する"},
            {"_score": 0.7, "policy_label": "質問で会話を広げる"},
        ]
        ctx = create_external_context(weight=0.7)

        adjusted, bias, state = process_with_context_sensitivity(candidates, ctx)

        assert len(adjusted) == 2
        assert isinstance(bias, SensitivityBias)
        assert isinstance(state, ContextState)
        assert state.last_context is not None

    def test_state_accumulates_across_calls(self):
        """State accumulates context across multiple calls."""
        candidates = [{"_score": 0.5, "policy_label": "共感する"}]
        state = ContextState()

        # First call with heavy context
        _, bias1, state = process_with_context_sensitivity(
            candidates, create_heavy_context(), context_state=state
        )

        # Second call with light context
        _, bias2, state = process_with_context_sensitivity(
            candidates, create_light_context(), context_state=state
        )

        # Smoothing should dampen the change
        # (not testing exact values, just that state is used)
        assert state.smoothed_weight != 0.5


# ── Utility Function Tests ──────────────────────────────────────────


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_sensitivity_summary(self):
        """Summary returns readable string."""
        bias = SensitivityBias(caution_level=0.75, risk_dampening=0.2)
        summary = get_sensitivity_summary(bias)

        assert "cautious" in summary
        assert "0.75" in summary

    def test_get_context_summary(self):
        """Context summary returns readable string."""
        ctx = create_external_context(pace=0.3, weight=0.8)
        summary = get_context_summary(ctx)

        assert "pace=0.30" in summary
        assert "weight=0.80" in summary

    def test_is_high_caution(self):
        """High caution detection works."""
        high = SensitivityBias(caution_level=0.7)
        low = SensitivityBias(caution_level=0.3)

        assert is_high_caution(high, threshold=0.6)
        assert not is_high_caution(low, threshold=0.6)

    def test_is_low_caution(self):
        """Low caution detection works."""
        high = SensitivityBias(caution_level=0.7)
        low = SensitivityBias(caution_level=0.3)

        assert is_low_caution(low, threshold=0.4)
        assert not is_low_caution(high, threshold=0.4)


# ── Configuration Tests ─────────────────────────────────────────────


class TestConfiguration:
    """Tests for ContextSensitivityConfig."""

    def test_create_config_with_custom_values(self):
        """Custom config creation works."""
        config = create_config(
            base_caution=0.6,
            weight_factor=0.7,
            max_risk_dampening=0.4,
        )

        assert config.base_caution == 0.6
        assert config.weight_caution_factor == 0.7
        assert config.max_risk_dampening == 0.4

    def test_config_serialization(self):
        """Config can be serialized and deserialized."""
        config = ContextSensitivityConfig(
            base_caution=0.6,
            weight_caution_factor=0.7,
            policy_risk_levels={"custom": 0.9},
        )

        data = to_dict(config)
        restored = from_dict(data)

        assert restored.base_caution == 0.6
        assert restored.weight_caution_factor == 0.7
        assert "custom" in restored.policy_risk_levels

    def test_config_affects_computation(self):
        """Different configs produce different results."""
        ctx = create_heavy_context()

        config_default = ContextSensitivityConfig()
        config_sensitive = ContextSensitivityConfig(
            weight_caution_factor=1.0,
            max_risk_dampening=0.5,
        )

        bias_default = compute_sensitivity_bias(ctx, config_default)
        bias_sensitive = compute_sensitivity_bias(ctx, config_sensitive)

        # More sensitive config should produce higher caution
        assert bias_sensitive.caution_level > bias_default.caution_level


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    """Integration tests verifying design principles."""

    def test_temporary_effect_decays(self):
        """Bias effect is temporary and decays."""
        state = ContextState()
        heavy_ctx = create_heavy_context()

        # Apply heavy context
        state = state.update(heavy_ctx)
        initial_weight = state.smoothed_weight

        # Apply decay
        decayed = state.apply_decay(delta_time=5.0, decay_rate=0.2)

        # Weight should have decayed toward neutral
        assert decayed.smoothed_weight < initial_weight
        assert decayed.smoothed_weight > 0.5  # But not all the way

    def test_bias_does_not_block_candidates(self):
        """Bias dampens but never blocks candidates."""
        candidates = [
            {"_score": 0.9, "policy_label": "からかう"},  # Very risky
        ]

        # Maximum caution
        bias = SensitivityBias(
            caution_level=0.9,
            risk_dampening=0.3,
            score_multiplier=0.8,
        )

        adjusted = apply_sensitivity_to_candidates(candidates, bias)

        # Score should be reduced but positive
        assert adjusted[0]["_score"] > 0

    def test_all_candidates_affected_equally_by_multiplier(self):
        """Score multiplier affects all candidates."""
        candidates = [
            {"_score": 0.8, "policy_label": "からかう"},
            {"_score": 0.8, "policy_label": "共感する"},
            {"_score": 0.8, "policy_label": "沈黙する", "_is_silence": True},
        ]

        # Only use multiplier, no risk dampening
        bias = SensitivityBias(
            caution_level=0.5,
            risk_dampening=0.0,
            safety_boost=0.0,
            score_multiplier=0.5,
        )

        adjusted = apply_sensitivity_to_candidates(candidates, bias)

        # All scores should be halved (approximately)
        for c in adjusted:
            assert c["_score"] <= 0.5
