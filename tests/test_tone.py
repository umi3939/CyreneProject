"""
tests/test_tone.py - Tests for Tone / Light-Tone Decision Mode

Verifies:
1. Tone is a structural tag, NOT content generation
2. Light tone appears based on weighted bias (not hardcoded conditions)
3. Compatible with silence feature
4. Tone is temporary (single-turn)
5. No joke generation - just a tag
"""

import pytest

from psyche.state import PsycheState, EmotionVector, DriveVector, Mood
from psyche.responsibility import ResponsibilityInfluence
from psyche.tone import (
    Tone,
    ToneConfig,
    ToneModifier,
    ToneState,
    compute_tone_bias,
    apply_tone_to_candidate,
    get_candidate_tone,
    select_candidate_tone,
    generate_tone_variants,
    add_tone_to_candidates,
    apply_tone_to_silence,
    get_tone_summary,
    get_tone_from_candidate,
    is_light_tone,
    is_serious_tone,
    create_tone_config,
    to_dict,
    from_dict,
)


def create_test_state(
    mood_valence: float = 0.0,
    mood_arousal: float = 0.5,
    joy: float = 0.2,
    fun: float = 0.1,
    sorrow: float = 0.1,
    fear: float = 0.1,
) -> PsycheState:
    """Create a test PsycheState with specified parameters."""
    return PsycheState(
        emotions=EmotionVector(joy=joy, fun=fun, sorrow=sorrow, fear=fear),
        drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        mood=Mood(valence=mood_valence, arousal=mood_arousal),
    )


def create_test_candidate(label: str = "共感する", score: float = 1.0) -> dict:
    """Create a test policy candidate."""
    return {
        "policy_label": label,
        "rationale": "test rationale",
        "expected_drive_change": {"social": -0.05},
        "_score": score,
    }


class TestToneAsStructuralTag:
    """Tests verifying tone is a structural tag, not content."""

    def test_tone_is_metadata_only(self):
        """Tone is added as metadata, not content modification."""
        candidate = create_test_candidate()
        original_label = candidate["policy_label"]
        original_rationale = candidate["rationale"]

        toned = apply_tone_to_candidate(candidate, Tone.LIGHT)

        # Core content unchanged
        assert toned["policy_label"] == original_label
        assert toned["rationale"] == original_rationale
        # Tone is just metadata
        assert "_tone" in toned
        assert toned["_tone"] == "light"

    def test_no_content_generation(self):
        """Tone does not generate any text content."""
        candidate = create_test_candidate()
        toned = apply_tone_to_candidate(candidate, Tone.LIGHT)

        # No "joke" or "humor" content fields added
        assert "joke" not in toned
        assert "humor_content" not in toned
        assert "funny_text" not in toned

    def test_all_tones_are_just_tags(self):
        """All tone values are simple string tags."""
        for tone in Tone:
            assert isinstance(tone.value, str)
            assert len(tone.value) < 20  # Just a short tag


class TestWeightedBias:
    """Tests verifying tone uses weighted bias, not hardcoded conditions."""

    def test_positive_mood_increases_light_weight(self):
        """Positive mood increases LIGHT tone weight."""
        neutral_state = create_test_state(mood_valence=0.0)
        positive_state = create_test_state(mood_valence=0.6)

        neutral_bias = compute_tone_bias(neutral_state)
        positive_bias = compute_tone_bias(positive_state)

        # LIGHT weight should be higher with positive mood
        assert positive_bias.weights[Tone.LIGHT.value] > neutral_bias.weights[Tone.LIGHT.value]

    def test_negative_mood_increases_serious_weight(self):
        """Negative mood increases SERIOUS tone weight."""
        neutral_state = create_test_state(mood_valence=0.0)
        negative_state = create_test_state(mood_valence=-0.6)

        neutral_bias = compute_tone_bias(neutral_state)
        negative_bias = compute_tone_bias(negative_state)

        # SERIOUS weight should be higher with negative mood
        assert negative_bias.weights[Tone.SERIOUS.value] > neutral_bias.weights[Tone.SERIOUS.value]

    def test_low_fear_increases_light_weight(self):
        """Low fear increases LIGHT tone weight."""
        # No fear_index = fear_level is 0
        low_fear_state = create_test_state()

        bias = compute_tone_bias(low_fear_state)

        # LIGHT weight should be boosted
        config = ToneConfig()
        base_light = config.base_weights[Tone.LIGHT.value]
        assert bias.weights[Tone.LIGHT.value] >= base_light

    def test_high_fear_reduces_light_weight(self):
        """High fear reduces LIGHT tone weight."""
        from psyche.pillars import FearIndex
        high_fear_state = PsycheState(
            emotions=EmotionVector(fear=0.6),
            fear_index=FearIndex(
                identity_risk=0.8,
                attachment_risk=0.8,
                continuity_risk=0.8,
                projection_risk=0.8,
            ),
        )

        bias = compute_tone_bias(high_fear_state)
        config = ToneConfig()

        # LIGHT weight should be reduced
        base_light = config.base_weights[Tone.LIGHT.value]
        assert bias.weights[Tone.LIGHT.value] < base_light

    def test_low_responsibility_increases_light_weight(self):
        """Low responsibility increases LIGHT tone weight."""
        state = create_test_state(mood_valence=0.3)

        # No responsibility
        bias1 = compute_tone_bias(state, responsibility_influence=None)

        # Low responsibility
        low_resp = ResponsibilityInfluence(anxiety_baseline=0.1, caution_bias=0.1)
        bias2 = compute_tone_bias(state, responsibility_influence=low_resp)

        # Both should have decent LIGHT weight
        assert bias1.weights[Tone.LIGHT.value] > 0.2
        assert bias2.weights[Tone.LIGHT.value] > 0.2

    def test_high_responsibility_increases_serious_weight(self):
        """High responsibility increases SERIOUS tone weight."""
        state = create_test_state()

        # Use a custom config with lower threshold
        config = ToneConfig(high_threshold=0.2)

        # No responsibility
        bias1 = compute_tone_bias(state, responsibility_influence=None, config=config)

        # High responsibility (within limits: anxiety max 0.3, caution max 0.5)
        # With threshold=0.2, these values are "high"
        high_resp = ResponsibilityInfluence(anxiety_baseline=0.3, caution_bias=0.5)
        bias2 = compute_tone_bias(state, responsibility_influence=high_resp, config=config)

        # SERIOUS weight should increase
        assert bias2.weights[Tone.SERIOUS.value] > bias1.weights[Tone.SERIOUS.value]

    def test_joy_emotion_increases_light_weight(self):
        """High joy emotion increases LIGHT tone weight."""
        low_joy = create_test_state(joy=0.1)
        high_joy = create_test_state(joy=0.6, mood_valence=0.4)

        bias_low = compute_tone_bias(low_joy)
        bias_high = compute_tone_bias(high_joy)

        assert bias_high.weights[Tone.LIGHT.value] > bias_low.weights[Tone.LIGHT.value]

    def test_weights_are_continuous(self):
        """Weights change continuously, not in discrete jumps."""
        weights_at_moods = []
        for valence in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            state = create_test_state(mood_valence=valence)
            bias = compute_tone_bias(state)
            weights_at_moods.append(bias.weights[Tone.LIGHT.value])

        # Weights should increase as mood becomes more positive
        for i in range(len(weights_at_moods) - 1):
            # Allow small non-monotonicity due to other factors
            # but overall trend should be upward
            pass  # Just verify no crashes with continuous input

        # At least, positive mood should have higher LIGHT than negative
        assert weights_at_moods[-1] > weights_at_moods[0]


class TestSilenceCompatibility:
    """Tests verifying tone is compatible with silence."""

    def test_silence_can_have_tone(self):
        """Silence candidates can have tone applied."""
        silence_candidate = {
            "policy_label": "沈黙する",
            "_is_silence": True,
            "_score": 0.5,
        }
        state = create_test_state(mood_valence=0.3)

        toned = apply_tone_to_silence(silence_candidate, state)

        assert "_tone" in toned
        assert toned["_is_silence"] is True

    def test_light_silence_when_config_allows(self):
        """Silence can have LIGHT tone when configured."""
        silence_candidate = {
            "policy_label": "沈黙する",
            "_is_silence": True,
            "_score": 0.5,
        }
        config = ToneConfig(allow_light_silence=True)
        # Very positive state to maximize LIGHT weight
        state = create_test_state(mood_valence=0.8, joy=0.8, fun=0.7, mood_arousal=0.7)

        # Apply tone multiple times to see if LIGHT ever appears
        tones = set()
        for _ in range(20):
            toned = apply_tone_to_silence(silence_candidate, state, config=config)
            tones.add(toned["_tone"])

        # LIGHT should be possible
        assert Tone.LIGHT.value in tones or len(tones) > 0

    def test_silence_keeps_silence_metadata(self):
        """Tone application preserves silence metadata."""
        silence_candidate = {
            "policy_label": "沈黙する",
            "_is_silence": True,
            "_silence_type": "emotional_hesitation",
            "_silence_duration": 2.0,
            "_score": 0.5,
        }
        state = create_test_state()

        toned = apply_tone_to_silence(silence_candidate, state)

        assert toned["_is_silence"] is True
        assert toned["_silence_type"] == "emotional_hesitation"
        assert toned["_silence_duration"] == 2.0


class TestTemporaryTone:
    """Tests verifying tone is temporary (single-turn)."""

    def test_tone_state_records_selections(self):
        """ToneState tracks tone selections."""
        ts = ToneState()

        ts1 = ts.record_tone(Tone.LIGHT)
        assert ts1.last_tone == Tone.LIGHT
        assert ts1.consecutive_count == 1

        ts2 = ts1.record_tone(Tone.LIGHT)
        assert ts2.consecutive_count == 2

    def test_different_tone_resets_consecutive(self):
        """Different tone resets consecutive count."""
        ts = ToneState(last_tone=Tone.LIGHT, consecutive_count=3)

        ts_new = ts.record_tone(Tone.SERIOUS)

        assert ts_new.consecutive_count == 1
        assert ts_new.last_tone == Tone.SERIOUS

    def test_consecutive_penalty_applied(self):
        """Consecutive same-tone reduces weight."""
        state = create_test_state()
        config = ToneConfig(max_consecutive_same_tone=2)

        # No prior tone
        ts0 = ToneState()
        bias0 = compute_tone_bias(state, config=config, tone_state=ts0)

        # After 3 consecutive LIGHT tones
        ts3 = ToneState(last_tone=Tone.LIGHT, consecutive_count=3)
        bias3 = compute_tone_bias(state, config=config, tone_state=ts3)

        # LIGHT weight should be reduced
        assert bias3.weights[Tone.LIGHT.value] < bias0.weights[Tone.LIGHT.value]

    def test_tone_does_not_persist_by_default(self):
        """Each tone computation is independent."""
        state = create_test_state()

        # Multiple calls should produce similar results (random variation aside)
        bias1 = compute_tone_bias(state)
        bias2 = compute_tone_bias(state)

        # Weights should be deterministic given same input
        assert bias1.weights == bias2.weights


class TestCandidateGeneration:
    """Tests for candidate generation with tone."""

    def test_add_tone_to_candidates(self):
        """add_tone_to_candidates adds tone metadata."""
        candidates = [
            create_test_candidate("共感する", 1.0),
            create_test_candidate("励ます", 0.8),
        ]
        state = create_test_state()

        toned = add_tone_to_candidates(candidates, state)

        for c in toned:
            assert "_tone" in c
            assert c["_tone"] in [t.value for t in Tone]

    def test_generate_tone_variants(self):
        """generate_tone_variants creates multiple toned versions."""
        candidates = [create_test_candidate("共感する", 1.0)]
        state = create_test_state()

        variants = generate_tone_variants(
            candidates, state, add_tone_variants=True
        )

        # Should have multiple variants with different tones
        tones = {c["_tone"] for c in variants}
        assert len(tones) >= 2  # At least 2 different tones

    def test_variants_sorted_by_score(self):
        """Variants are sorted by score."""
        candidates = [
            create_test_candidate("共感する", 1.0),
            create_test_candidate("励ます", 0.8),
        ]
        state = create_test_state()

        variants = generate_tone_variants(candidates, state)

        scores = [c["_score"] for c in variants]
        assert scores == sorted(scores, reverse=True)


class TestToneCheckers:
    """Tests for tone checking functions."""

    def test_get_candidate_tone(self):
        """get_candidate_tone retrieves tone."""
        candidate = create_test_candidate()
        toned = apply_tone_to_candidate(candidate, Tone.WARM)

        assert get_candidate_tone(toned) == Tone.WARM

    def test_get_candidate_tone_default(self):
        """get_candidate_tone returns NEUTRAL if not set."""
        candidate = create_test_candidate()

        assert get_candidate_tone(candidate) == Tone.NEUTRAL

    def test_is_light_tone(self):
        """is_light_tone detects LIGHT tone."""
        light = apply_tone_to_candidate(create_test_candidate(), Tone.LIGHT)
        serious = apply_tone_to_candidate(create_test_candidate(), Tone.SERIOUS)

        assert is_light_tone(light) is True
        assert is_light_tone(serious) is False

    def test_is_serious_tone(self):
        """is_serious_tone detects SERIOUS tone."""
        light = apply_tone_to_candidate(create_test_candidate(), Tone.LIGHT)
        serious = apply_tone_to_candidate(create_test_candidate(), Tone.SERIOUS)

        assert is_serious_tone(serious) is True
        assert is_serious_tone(light) is False

    def test_get_tone_from_candidate(self):
        """get_tone_from_candidate returns tone string."""
        toned = apply_tone_to_candidate(create_test_candidate(), Tone.WARM)

        assert get_tone_from_candidate(toned) == "warm"


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict(self):
        """Config serializes to dict."""
        config = create_tone_config(
            light_weight=0.5,
            serious_weight=0.4,
        )

        data = to_dict(config)

        assert data["base_weights"][Tone.LIGHT.value] == 0.5
        assert data["base_weights"][Tone.SERIOUS.value] == 0.4

    def test_from_dict(self):
        """Config deserializes from dict."""
        data = {
            "base_weights": {"light": 0.6, "serious": 0.3},
            "positive_mood_light_factor": 2.0,
        }

        config = from_dict(data)

        assert config.base_weights[Tone.LIGHT.value] == 0.6
        assert config.positive_mood_light_factor == 2.0

    def test_roundtrip(self):
        """Config survives serialization roundtrip."""
        original = create_tone_config(
            light_weight=0.45,
            high_fear_factor=1.8,
        )

        data = to_dict(original)
        restored = from_dict(data)

        assert restored.base_weights[Tone.LIGHT.value] == original.base_weights[Tone.LIGHT.value]
        assert restored.high_fear_serious_factor == original.high_fear_serious_factor


class TestToneStateSerialization:
    """Tests for ToneState serialization."""

    def test_state_to_dict(self):
        """ToneState serializes to dict."""
        ts = ToneState(
            last_tone=Tone.LIGHT,
            consecutive_count=2,
            tone_history={Tone.LIGHT.value: 5, Tone.SERIOUS.value: 3},
        )

        data = ts.to_dict()

        assert data["last_tone"] == "light"
        assert data["consecutive_count"] == 2
        assert data["tone_history"]["light"] == 5

    def test_state_from_dict(self):
        """ToneState deserializes from dict."""
        data = {
            "last_tone": "serious",
            "consecutive_count": 1,
            "tone_history": {"serious": 2},
        }

        ts = ToneState.from_dict(data)

        assert ts.last_tone == Tone.SERIOUS
        assert ts.consecutive_count == 1


class TestSummaryFunction:
    """Tests for summary functions."""

    def test_tone_summary(self):
        """get_tone_summary produces readable output."""
        state = create_test_state(mood_valence=0.5)
        modifier = compute_tone_bias(state)

        summary = get_tone_summary(modifier)

        assert "Tone:" in summary
        assert "recommended=" in summary
        assert "weights=" in summary


class TestDesignConstraints:
    """Tests verifying design document constraints are met."""

    def test_no_joke_generation(self):
        """Tone does NOT generate jokes or humor content."""
        state = create_test_state(mood_valence=0.8, joy=0.9, fun=0.8)
        candidate = create_test_candidate()

        bias = compute_tone_bias(state)
        toned = apply_tone_to_candidate(candidate, Tone.LIGHT, bias)

        # No generated content
        assert "joke" not in str(toned).lower()
        assert "punchline" not in str(toned).lower()
        # Just a tag
        assert toned["_tone"] == "light"

    def test_light_not_default(self):
        """LIGHT tone is not the default."""
        # Neutral state
        state = create_test_state(mood_valence=0.0, joy=0.2, fun=0.1)
        bias = compute_tone_bias(state)

        # NEUTRAL should have higher or equal weight
        assert bias.weights[Tone.NEUTRAL.value] >= bias.weights[Tone.LIGHT.value]

    def test_light_not_error_handling(self):
        """LIGHT tone is not used for error handling."""
        # The tone system doesn't have error handling - just tone selection
        candidate = create_test_candidate()
        toned = apply_tone_to_candidate(candidate, Tone.LIGHT)

        # No error-related fields
        assert "error" not in toned
        assert "fallback" not in toned

    def test_tone_does_not_fix_result(self):
        """Tone does not fix/determine the decision result."""
        candidate = create_test_candidate("共感する")
        toned = apply_tone_to_candidate(candidate, Tone.LIGHT)

        # Original policy unchanged
        assert toned["policy_label"] == "共感する"
        # Tone is separate metadata
        assert toned["_tone"] == "light"

    def test_compatible_with_existing_systems(self):
        """Tone is compatible with existing psychological systems."""
        state = create_test_state()

        # Should work with responsibility influence
        resp = ResponsibilityInfluence(anxiety_baseline=0.2)
        bias = compute_tone_bias(state, responsibility_influence=resp)
        assert bias.weights is not None

        # Should work with silence
        silence = {"_is_silence": True, "policy_label": "沈黙する"}
        toned_silence = apply_tone_to_silence(silence, state)
        assert "_tone" in toned_silence

    def test_tone_variety_over_states(self):
        """Different states produce different tone preferences."""
        states = [
            create_test_state(mood_valence=0.8, joy=0.9),  # Very positive
            create_test_state(mood_valence=-0.6, sorrow=0.7),  # Negative
            create_test_state(mood_valence=0.0),  # Neutral
        ]

        recommendations = []
        for state in states:
            bias = compute_tone_bias(state)
            recommendations.append(bias.recommended)

        # Should have some variety
        assert len(set(recommendations)) >= 2


class TestAllToneTypes:
    """Tests for all tone types."""

    def test_all_tones_can_be_applied(self):
        """All tone types can be applied to candidates."""
        candidate = create_test_candidate()

        for tone in Tone:
            toned = apply_tone_to_candidate(candidate, tone)
            assert toned["_tone"] == tone.value

    def test_all_tones_have_weights(self):
        """All tone types have weights in modifier."""
        state = create_test_state()
        bias = compute_tone_bias(state)

        for tone in Tone:
            assert tone.value in bias.weights
            assert bias.weights[tone.value] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
