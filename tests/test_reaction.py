"""
tests/test_reaction.py - Comprehensive tests for psyche/reaction.py

Verifies:
1.  Neutral percept -> only decay applied
2.  Happy percept -> joy increases
3.  Sad percept -> sorrow increases
4.  Each emotion mapping in _EMOTION_MAP
5.  Positive valence secondary effects
6.  Negative valence secondary effects
7.  Decay rate over time (delta_time=1, 2, 5)
8.  Drive updates (social, curiosity, expression)
9.  Mood drift direction
10. Responsibility influence: anxiety baseline raises fear/sorrow
11. Responsibility influence: fear amplification
12. Responsibility influence: mood penalty
13. Amplitude modifier > 1 amplifies emotion change
14. Amplitude modifier < 1 dampens emotion change
15. All values clamped [0, 1] for emotions/drives, [-1, 1] for valence
16. Immutability: original state not modified
"""

import copy

import pytest

from psyche.pillars import FearIndex
from psyche.reaction import (
    DECAY_RATE,
    _EMOTION_MAP,
    _VALENCE_NEGATIVE,
    _VALENCE_POSITIVE,
    _apply_responsibility_emotion_influence,
    _clamp,
    react,
)
from psyche.responsibility import ResponsibilityInfluence
from psyche.state import DriveVector, EmotionVector, Mood, Percept, PsycheState


# ── Helpers ───────────────────────────────────────────────────

def _zero_emotion_state(**overrides) -> PsycheState:
    """Create a PsycheState with all emotions at 0.0 and default drives/mood."""
    emo_kw = {k: 0.0 for k in EmotionVector.model_fields}
    emo_kw.update(overrides.pop("emotions", {}))
    return PsycheState(
        emotions=EmotionVector(**emo_kw),
        drives=overrides.pop("drives", DriveVector(social=0.5, curiosity=0.5, expression=0.5)),
        mood=overrides.pop("mood", Mood(valence=0.0, arousal=0.3)),
        **overrides,
    )


def _neutral_percept(**overrides) -> Percept:
    """Create a neutral percept with optional overrides."""
    kw = dict(text="", meaning="", emotion="neutral", intent="unknown",
              topics=[], sentiment=0.0, emotion_valence=0.0)
    kw.update(overrides)
    return Percept(**kw)


# ── Test _clamp ───────────────────────────────────────────────

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_lower_bound(self):
        assert _clamp(-0.1) == 0.0

    def test_above_upper_bound(self):
        assert _clamp(1.5) == 1.0

    def test_at_lower_bound(self):
        assert _clamp(0.0) == 0.0

    def test_at_upper_bound(self):
        assert _clamp(1.0) == 1.0

    def test_custom_bounds(self):
        assert _clamp(0.5, lo=-1.0, hi=1.0) == 0.5
        assert _clamp(-2.0, lo=-1.0, hi=1.0) == -1.0
        assert _clamp(2.0, lo=-1.0, hi=1.0) == 1.0


# ── Test 1: Neutral percept -> only decay ─────────────────────

class TestNeutralPercept:
    """When emotion is 'neutral', no direct emotion stimulus is applied.
    Only time-based decay should affect existing emotion values."""

    def test_neutral_percept_decay_only(self):
        state = _zero_emotion_state(emotions={"joy": 0.5, "sorrow": 0.3})
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        expected_decay = DECAY_RATE ** 1.0
        assert result.emotions.joy == pytest.approx(0.5 * expected_decay, abs=1e-6)
        assert result.emotions.sorrow == pytest.approx(0.3 * expected_decay, abs=1e-6)

    def test_neutral_percept_no_direct_stimulus(self):
        """No emotion field should increase from a neutral percept on zero-emotion state."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        for field in EmotionVector.model_fields:
            assert getattr(result.emotions, field) == pytest.approx(0.0, abs=1e-9)


# ── Test 2: Happy percept -> joy increases ────────────────────

class TestHappyPercept:
    def test_joy_increases_from_happy(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="happy")
        result = react(percept, state, delta_time=1.0)

        # joy should have been set by base_delta=0.2, then decayed
        expected = (0.0 + 0.2) * (DECAY_RATE ** 1.0)
        assert result.emotions.joy == pytest.approx(expected, abs=1e-4)
        assert result.emotions.joy > 0.0

    def test_joy_accumulates_from_prior_joy(self):
        state = _zero_emotion_state(emotions={"joy": 0.3})
        percept = _neutral_percept(emotion="happy")
        result = react(percept, state, delta_time=1.0)

        expected = (0.3 + 0.2) * (DECAY_RATE ** 1.0)
        assert result.emotions.joy == pytest.approx(expected, abs=1e-4)


# ── Test 3: Sad percept -> sorrow increases ───────────────────

class TestSadPercept:
    def test_sorrow_increases_from_sad(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="sad")
        result = react(percept, state, delta_time=1.0)

        expected = (0.0 + 0.2) * (DECAY_RATE ** 1.0)
        assert result.emotions.sorrow == pytest.approx(expected, abs=1e-4)
        assert result.emotions.sorrow > 0.0


# ── Test 4: All _EMOTION_MAP entries ──────────────────────────

class TestEmotionMapping:
    """Each percept emotion label maps to the correct EmotionVector field."""

    @pytest.mark.parametrize("label,field", [
        ("happy", "joy"),
        ("sad", "sorrow"),
        ("angry", "anger"),
        ("surprised", "surprise"),
        ("scared", "fear"),
        ("loving", "love"),
        ("teasing", "fun"),
    ])
    def test_emotion_label_maps_to_field(self, label, field):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion=label)
        result = react(percept, state, delta_time=1.0)

        # The mapped field should be non-zero
        value = getattr(result.emotions, field)
        assert value > 0.0, f"Expected {field} > 0 for percept emotion '{label}'"

        # All other emotions (except the mapped one) should remain at 0 (after decay of 0)
        for other_field in EmotionVector.model_fields:
            if other_field != field:
                assert getattr(result.emotions, other_field) == pytest.approx(0.0, abs=1e-9), (
                    f"Expected {other_field} == 0 but got {getattr(result.emotions, other_field)}"
                )

    def test_neutral_maps_to_empty_string(self):
        assert _EMOTION_MAP["neutral"] == ""

    def test_all_map_keys_present(self):
        expected_keys = {"happy", "sad", "angry", "surprised", "scared", "loving", "teasing", "neutral"}
        assert set(_EMOTION_MAP.keys()) == expected_keys


# ── Test 5: Positive valence secondary effects ────────────────

class TestPositiveValence:
    def test_positive_valence_boosts_joy_love_fun(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=0.8)
        result = react(percept, state, delta_time=1.0)

        # Positive valence should increase joy, love, fun via secondary effects
        decay = DECAY_RATE ** 1.0
        for field, weight in _VALENCE_POSITIVE.items():
            expected = (0.8 * weight) * decay
            value = getattr(result.emotions, field)
            assert value == pytest.approx(expected, abs=1e-4), (
                f"{field}: expected {expected:.4f}, got {value:.4f}"
            )

    def test_positive_valence_does_not_affect_negative_emotions(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=0.5)
        result = react(percept, state, delta_time=1.0)

        # sorrow, anger, fear should remain at 0 (only positive valence map applied)
        assert result.emotions.sorrow == pytest.approx(0.0, abs=1e-9)
        assert result.emotions.anger == pytest.approx(0.0, abs=1e-9)
        assert result.emotions.fear == pytest.approx(0.0, abs=1e-9)

    def test_zero_valence_no_secondary(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=0.0)
        result = react(percept, state, delta_time=1.0)

        for field in EmotionVector.model_fields:
            assert getattr(result.emotions, field) == pytest.approx(0.0, abs=1e-9)


# ── Test 6: Negative valence secondary effects ────────────────

class TestNegativeValence:
    def test_negative_valence_boosts_sorrow_anger_fear(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=-0.8)
        result = react(percept, state, delta_time=1.0)

        decay = DECAY_RATE ** 1.0
        for field, weight in _VALENCE_NEGATIVE.items():
            expected = (0.8 * weight) * decay
            value = getattr(result.emotions, field)
            assert value == pytest.approx(expected, abs=1e-4), (
                f"{field}: expected {expected:.4f}, got {value:.4f}"
            )

    def test_negative_valence_does_not_affect_positive_emotions(self):
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=-0.5)
        result = react(percept, state, delta_time=1.0)

        assert result.emotions.joy == pytest.approx(0.0, abs=1e-9)
        assert result.emotions.love == pytest.approx(0.0, abs=1e-9)
        assert result.emotions.fun == pytest.approx(0.0, abs=1e-9)


# ── Test 7: Decay rate over time ──────────────────────────────

class TestDecayRate:
    @pytest.mark.parametrize("dt", [1.0, 2.0, 5.0])
    def test_decay_exponential_over_time(self, dt):
        """Emotions decay as DECAY_RATE ** delta_time."""
        state = _zero_emotion_state(emotions={"joy": 0.8, "anger": 0.6})
        percept = _neutral_percept()
        result = react(percept, state, delta_time=dt)

        decay = DECAY_RATE ** dt
        assert result.emotions.joy == pytest.approx(0.8 * decay, abs=1e-4)
        assert result.emotions.anger == pytest.approx(0.6 * decay, abs=1e-4)

    def test_longer_decay_produces_smaller_values(self):
        state = _zero_emotion_state(emotions={"joy": 0.8})
        percept = _neutral_percept()
        r1 = react(percept, state, delta_time=1.0)
        r2 = react(percept, state, delta_time=5.0)

        assert r2.emotions.joy < r1.emotions.joy

    def test_decay_approaches_zero(self):
        state = _zero_emotion_state(emotions={"joy": 0.5})
        percept = _neutral_percept()
        result = react(percept, state, delta_time=100.0)

        # 0.95^100 ~ 0.0059, so 0.5 * 0.0059 ~ 0.003
        assert result.emotions.joy < 0.01


# ── Test 8: Drive updates (state-dependent dynamics) ──────────

class TestDriveUpdates:
    def test_social_drive_changes_from_neutral(self):
        """Social drive changes via state-dependent dynamics on neutral input."""
        state = _zero_emotion_state(drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # Drive should change (not remain exactly 0.5) due to time_passage section
        # The exact value is state-dependent, but social should decrease
        # due to the default time_passage section (social_raw = 0.02*1 - 0.12 < 0)
        assert result.drives.social < 0.5

    def test_curiosity_changes_with_intent(self):
        """Curiosity: changes differently based on intent."""
        state = _zero_emotion_state(drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5))
        percept_greeting = _neutral_percept(intent="greeting")
        percept_sharing = _neutral_percept(intent="sharing")
        result_greeting = react(percept_greeting, state, delta_time=1.0)
        result_sharing = react(percept_sharing, state, delta_time=1.0)

        # Sharing intent should reduce curiosity more than greeting
        assert result_sharing.drives.curiosity < result_greeting.drives.curiosity

    def test_curiosity_decreases_on_sharing(self):
        """Curiosity decreases when intent is 'sharing' (information satisfies curiosity)."""
        state = _zero_emotion_state(drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5))
        percept = _neutral_percept(intent="sharing")
        result = react(percept, state, delta_time=1.0)

        # Curiosity should decrease due to sharing intent in time_passage section
        assert result.drives.curiosity < 0.5

    def test_curiosity_decreases_on_question(self):
        """Curiosity decreases when intent is 'question' (information satisfies curiosity)."""
        state = _zero_emotion_state(drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5))
        percept = _neutral_percept(intent="question")
        result = react(percept, state, delta_time=1.0)

        # Curiosity should decrease due to question intent in time_passage section
        assert result.drives.curiosity < 0.5

    def test_expression_drive_affected_by_emotion(self):
        """Expression drive is influenced by emotion intensity via state-dependent dynamics."""
        state_low_emo = _zero_emotion_state(
            emotions={"joy": 0.1},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        state_high_emo = _zero_emotion_state(
            emotions={"joy": 0.8},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        percept = _neutral_percept()
        result_low = react(percept, state_low_emo, delta_time=1.0)
        result_high = react(percept, state_high_emo, delta_time=1.0)

        # Higher emotion should lead to higher expression drive
        assert result_high.drives.expression > result_low.drives.expression

    def test_social_drive_clamps_at_boundaries(self):
        """Social drive stays within [0, 1] even with extreme deltas."""
        state = _zero_emotion_state(drives=DriveVector(social=0.01, curiosity=0.5, expression=0.5))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # Drive values must be clamped to [0, 1]
        assert 0.0 <= result.drives.social <= 1.0


# ── Test 9: Mood drift direction ──────────────────────────────

class TestMoodDrift:
    def test_positive_emotions_drift_valence_up(self):
        """Joy pushes mood valence toward positive."""
        state = _zero_emotion_state(emotions={"joy": 0.8}, mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        assert result.mood.valence > 0.0

    def test_negative_emotions_drift_valence_down(self):
        """Sorrow pushes mood valence toward negative."""
        state = _zero_emotion_state(emotions={"sorrow": 0.8}, mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        assert result.mood.valence < 0.0

    def test_arousal_tracks_max_emotion(self):
        """Arousal should move toward max emotion intensity."""
        state = _zero_emotion_state(emotions={"joy": 0.9}, mood=Mood(valence=0.0, arousal=0.0))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # Arousal should increase (was 0, max_emo is high)
        assert result.mood.arousal > 0.0

    def test_mood_ema_inertia(self):
        """Mood changes slowly (EMA with alpha=0.1)."""
        state = _zero_emotion_state(emotions={"joy": 1.0}, mood=Mood(valence=0.0, arousal=0.0))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # With alpha=0.1, valence should move only 10% toward instant_valence
        # instant_valence = (joy + love + fun - sorrow - anger - fear) / 3
        # joy after decay = 1.0 * 0.95 = 0.95
        # instant_valence = 0.95 / 3 ~ 0.3167
        # new_valence = 0.0 + 0.1 * (0.3167 - 0.0) = 0.03167
        assert result.mood.valence < 0.1  # Only a small fraction of the instant value


# ── Test 10: Responsibility influence - anxiety baseline ──────

class TestResponsibilityAnxiety:
    def test_anxiety_raises_fear_baseline(self):
        """anxiety_baseline > 0 sets fear floor to anxiety * 0.5."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        # Fear should be at least anxiety * 0.5 = 0.1
        assert result.emotions.fear >= 0.1 - 1e-6

    def test_anxiety_raises_sorrow_baseline(self):
        """anxiety_baseline > 0 sets sorrow floor to anxiety * 0.3."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        # Sorrow should be at least anxiety * 0.3 = 0.06
        assert result.emotions.sorrow >= 0.06 - 1e-6

    def test_anxiety_zero_no_effect(self):
        """When anxiety_baseline is 0, no effect on fear/sorrow."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.0, fear_amplification=0.0)
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        assert result.emotions.fear == pytest.approx(0.0, abs=1e-9)
        assert result.emotions.sorrow == pytest.approx(0.0, abs=1e-9)

    def test_anxiety_does_not_lower_existing_fear(self):
        """Anxiety baseline is a floor (max), not additive. High fear stays high."""
        state = _zero_emotion_state(emotions={"fear": 0.8})
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.1, fear_amplification=0.0)
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        # fear after decay = 0.8 * 0.95 = 0.76, which is above anxiety*0.5=0.05
        assert result.emotions.fear >= 0.7


# ── Test 11: Responsibility influence - fear amplification ────

class TestResponsibilityFearAmplification:
    def test_fear_amplification_when_fear_above_threshold(self):
        """When fear > 0.1, fear_amplification adds fear_amp * 0.2."""
        # Start with enough fear to be above 0.1 after decay
        state = _zero_emotion_state(emotions={"fear": 0.3})
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(
            anxiety_baseline=0.0,
            fear_amplification=0.5,
        )
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        # fear after decay = 0.3 * 0.95 = 0.285 (> 0.1)
        # fear_amplification: fear += 0.5 * 0.2 = 0.1 -> fear = 0.385
        fear_decayed = 0.3 * (DECAY_RATE ** 1.0)
        expected_fear = fear_decayed + 0.5 * 0.2
        assert result.emotions.fear == pytest.approx(expected_fear, abs=1e-3)

    def test_fear_amplification_no_effect_when_fear_low(self):
        """When fear <= 0.1, fear_amplification has no effect."""
        # Start with fear that decays below 0.1
        state = _zero_emotion_state(emotions={"fear": 0.05})
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(
            anxiety_baseline=0.0,
            fear_amplification=0.5,
        )
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        # fear after decay = 0.05 * 0.95 = 0.0475 (< 0.1), no amplification
        expected = 0.05 * (DECAY_RATE ** 1.0)
        assert result.emotions.fear == pytest.approx(expected, abs=1e-4)

    def test_fear_amplification_zero_has_no_effect(self):
        """Zero fear_amplification does not change fear."""
        state = _zero_emotion_state(emotions={"fear": 0.5})
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(
            anxiety_baseline=0.0,
            fear_amplification=0.0,
        )
        result = react(percept, state, delta_time=1.0, responsibility_influence=influence)

        expected = 0.5 * (DECAY_RATE ** 1.0)
        assert result.emotions.fear == pytest.approx(expected, abs=1e-4)


# ── Test 12: Responsibility influence - mood penalty ──────────

class TestResponsibilityMoodPenalty:
    def test_anxiety_reduces_mood_valence(self):
        """Mood valence is reduced by anxiety_baseline * 0.3."""
        state = _zero_emotion_state(mood=Mood(valence=0.5, arousal=0.3))
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)

        result_with = react(percept, state, delta_time=1.0, responsibility_influence=influence)
        result_without = react(percept, state, delta_time=1.0)

        # The mood penalty is anxiety_baseline * 0.3 = 0.06
        penalty = 0.2 * 0.3
        assert result_with.mood.valence == pytest.approx(
            result_without.mood.valence - penalty, abs=0.05
        )

    def test_no_responsibility_no_mood_penalty(self):
        """Without responsibility_influence, no penalty applied."""
        state = _zero_emotion_state(mood=Mood(valence=0.5, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # Mood should drift via EMA toward neutral but without penalty
        # With all emotions at 0, instant_valence = 0
        # new_valence = 0.5 + 0.1 * (0 - 0.5) = 0.45
        assert result.mood.valence == pytest.approx(0.45, abs=1e-4)


# ── Test 13: Amplitude modifier > 1 amplifies emotion ────────

class TestAmplitudeModifierHigh:
    def test_high_amplitude_increases_emotion_delta(self):
        """amplitude_modifier > 1 makes emotion changes larger."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="happy")

        result_normal = react(percept, state, delta_time=1.0, amplitude_modifier=1.0)
        result_amplified = react(percept, state, delta_time=1.0, amplitude_modifier=1.5)

        assert result_amplified.emotions.joy > result_normal.emotions.joy

    def test_high_amplitude_scales_valence_effects(self):
        """amplitude_modifier > 1 also amplifies valence-based secondary effects."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=0.8)

        result_normal = react(percept, state, delta_time=1.0, amplitude_modifier=1.0)
        result_amplified = react(percept, state, delta_time=1.0, amplitude_modifier=1.5)

        assert result_amplified.emotions.joy > result_normal.emotions.joy
        assert result_amplified.emotions.love > result_normal.emotions.love
        assert result_amplified.emotions.fun > result_normal.emotions.fun


# ── Test 14: Amplitude modifier < 1 dampens emotion ──────────

class TestAmplitudeModifierLow:
    def test_low_amplitude_decreases_emotion_delta(self):
        """amplitude_modifier < 1 makes emotion changes smaller."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="happy")

        result_normal = react(percept, state, delta_time=1.0, amplitude_modifier=1.0)
        result_dampened = react(percept, state, delta_time=1.0, amplitude_modifier=0.7)

        assert result_dampened.emotions.joy < result_normal.emotions.joy

    def test_low_amplitude_scales_valence_effects(self):
        """amplitude_modifier < 1 also dampens valence-based secondary effects."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion_valence=-0.8)

        result_normal = react(percept, state, delta_time=1.0, amplitude_modifier=1.0)
        result_dampened = react(percept, state, delta_time=1.0, amplitude_modifier=0.7)

        assert result_dampened.emotions.sorrow < result_normal.emotions.sorrow
        assert result_dampened.emotions.anger < result_normal.emotions.anger
        assert result_dampened.emotions.fear < result_normal.emotions.fear


# ── Test 15: Clamping bounds ──────────────────────────────────

class TestClamping:
    def test_emotions_clamped_0_to_1(self):
        """All emotion values stay within [0, 1] even with extreme inputs."""
        # Start near max and add more
        state = _zero_emotion_state(emotions={"joy": 0.95})
        percept = _neutral_percept(emotion="happy", emotion_valence=1.0)
        result = react(percept, state, delta_time=1.0, amplitude_modifier=2.0)

        for field in EmotionVector.model_fields:
            value = getattr(result.emotions, field)
            assert 0.0 <= value <= 1.0, f"{field} = {value} out of [0, 1]"

    def test_drives_clamped_0_to_1(self):
        """All drive values stay within [0, 1]."""
        state = _zero_emotion_state(drives=DriveVector(social=0.0, curiosity=0.0, expression=0.0))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        for field in ["social", "curiosity", "expression"]:
            value = getattr(result.drives, field)
            assert 0.0 <= value <= 1.0, f"{field} = {value} out of [0, 1]"

    def test_valence_clamped_neg1_to_1(self):
        """Mood valence stays within [-1, 1]."""
        # Push valence way down with strong negative emotions + responsibility
        state = _zero_emotion_state(
            emotions={"sorrow": 1.0, "anger": 1.0, "fear": 1.0},
            mood=Mood(valence=-0.9, arousal=0.3),
        )
        percept = _neutral_percept(emotion="sad", emotion_valence=-1.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.3, fear_amplification=0.5)
        result = react(percept, state, delta_time=1.0,
                       responsibility_influence=influence, amplitude_modifier=2.0)

        assert -1.0 <= result.mood.valence <= 1.0

    def test_arousal_clamped_0_to_1(self):
        """Mood arousal stays within [0, 1]."""
        state = _zero_emotion_state(
            emotions={"joy": 1.0},
            mood=Mood(valence=0.0, arousal=0.99),
        )
        percept = _neutral_percept(emotion="happy", emotion_valence=1.0)
        result = react(percept, state, delta_time=1.0, amplitude_modifier=2.0)

        assert 0.0 <= result.mood.arousal <= 1.0


# ── Test 16: Immutability - original state not modified ───────

class TestImmutability:
    def test_original_state_unchanged(self):
        """react() must not mutate the input PsycheState."""
        state = _zero_emotion_state(
            emotions={"joy": 0.5, "sorrow": 0.3},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.0, arousal=0.3),
        )
        original_emotions = state.emotions.as_dict().copy()
        original_drives = state.drives.model_dump().copy()
        original_mood_valence = state.mood.valence
        original_mood_arousal = state.mood.arousal

        percept = _neutral_percept(emotion="happy", emotion_valence=0.5)
        _ = react(percept, state, delta_time=1.0)

        # Verify original is unchanged
        assert state.emotions.as_dict() == original_emotions
        assert state.drives.model_dump() == original_drives
        assert state.mood.valence == original_mood_valence
        assert state.mood.arousal == original_mood_arousal

    def test_original_state_unchanged_with_responsibility(self):
        """react() with responsibility does not mutate the input state."""
        state = _zero_emotion_state(emotions={"fear": 0.5})
        original_fear = state.emotions.fear

        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.3)
        _ = react(_neutral_percept(), state, delta_time=1.0, responsibility_influence=influence)

        assert state.emotions.fear == original_fear

    def test_returned_state_is_different_object(self):
        """react() returns a new PsycheState, not the same reference."""
        state = _zero_emotion_state()
        result = react(_neutral_percept(), state, delta_time=1.0)

        assert result is not state


# ── Test _apply_responsibility_emotion_influence directly ─────

class TestApplyResponsibilityEmotionInfluence:
    def test_anxiety_baseline_sets_fear_floor(self):
        """Direct test: anxiety_baseline raises fear minimum to anxiety * 0.5."""
        emotions = EmotionVector(fear=0.0, sorrow=0.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        assert result.fear == pytest.approx(0.1, abs=1e-6)  # 0.2 * 0.5

    def test_anxiety_baseline_sets_sorrow_floor(self):
        """Direct test: anxiety_baseline raises sorrow minimum to anxiety * 0.3."""
        emotions = EmotionVector(fear=0.0, sorrow=0.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        assert result.sorrow == pytest.approx(0.06, abs=1e-6)  # 0.2 * 0.3

    def test_fear_amplification_adds_to_fear_above_threshold(self):
        """Direct test: fear_amp > 0 and fear > 0.1 adds fear_amp * 0.2."""
        emotions = EmotionVector(fear=0.3)
        influence = ResponsibilityInfluence(anxiety_baseline=0.0, fear_amplification=0.4)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        # fear = 0.3 > 0.1, so fear += 0.4 * 0.2 = 0.08 -> 0.38
        assert result.fear == pytest.approx(0.38, abs=1e-6)

    def test_fear_amplification_no_effect_below_threshold(self):
        """Direct test: fear <= 0.1 means no amplification."""
        emotions = EmotionVector(fear=0.05)
        influence = ResponsibilityInfluence(anxiety_baseline=0.0, fear_amplification=0.4)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        assert result.fear == pytest.approx(0.05, abs=1e-6)

    def test_combined_anxiety_and_amplification(self):
        """Direct test: anxiety sets floor, then amplification increases if above threshold."""
        emotions = EmotionVector(fear=0.0, sorrow=0.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.3, fear_amplification=0.5)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        # anxiety sets fear floor: max(0.0, 0.3*0.5) = 0.15 (> 0.1)
        # then amplification: 0.15 + 0.5*0.2 = 0.25
        assert result.fear == pytest.approx(0.25, abs=1e-6)
        # sorrow floor: max(0.0, 0.3*0.3) = 0.09
        assert result.sorrow == pytest.approx(0.09, abs=1e-6)

    def test_does_not_modify_other_emotions(self):
        """Responsibility influence only touches fear and sorrow."""
        emotions = EmotionVector(joy=0.5, anger=0.3, surprise=0.2, love=0.4, fun=0.1)
        influence = ResponsibilityInfluence(anxiety_baseline=0.3, fear_amplification=0.5)
        result = _apply_responsibility_emotion_influence(emotions, influence)

        assert result.joy == pytest.approx(0.5, abs=1e-9)
        assert result.anger == pytest.approx(0.3, abs=1e-9)
        assert result.surprise == pytest.approx(0.2, abs=1e-9)
        assert result.love == pytest.approx(0.4, abs=1e-9)
        assert result.fun == pytest.approx(0.1, abs=1e-9)

    def test_immutability_of_input(self):
        """Input EmotionVector must not be mutated."""
        emotions = EmotionVector(fear=0.0, sorrow=0.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)
        _ = _apply_responsibility_emotion_influence(emotions, influence)

        # Original should be unchanged
        assert emotions.fear == pytest.approx(0.0, abs=1e-9)
        assert emotions.sorrow == pytest.approx(0.0, abs=1e-9)


# ── Combined / integration tests ─────────────────────────────

class TestCombinedBehavior:
    def test_happy_percept_with_positive_valence(self):
        """Happy emotion + positive valence should stack on joy."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="happy", emotion_valence=0.8)
        result = react(percept, state, delta_time=1.0)

        # joy gets base_delta=0.2 from happy + v*0.15 from valence = 0.2 + 0.12 = 0.32
        # Then decayed: 0.32 * 0.95 = 0.304
        decay = DECAY_RATE ** 1.0
        expected_joy = (0.2 + 0.8 * 0.15) * decay
        assert result.emotions.joy == pytest.approx(expected_joy, abs=1e-4)

    def test_sad_percept_with_negative_valence(self):
        """Sad emotion + negative valence should stack on sorrow."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="sad", emotion_valence=-0.6)
        result = react(percept, state, delta_time=1.0)

        decay = DECAY_RATE ** 1.0
        expected_sorrow = (0.2 + 0.6 * 0.10) * decay
        assert result.emotions.sorrow == pytest.approx(expected_sorrow, abs=1e-4)

    def test_multiple_reactions_accumulate(self):
        """Applying react() multiple times should accumulate effects."""
        state = _zero_emotion_state()
        percept = _neutral_percept(emotion="happy")

        state1 = react(percept, state, delta_time=1.0)
        state2 = react(percept, state1, delta_time=1.0)

        assert state2.emotions.joy > state1.emotions.joy

    def test_fear_index_boosts_fear_emotion(self):
        """When fear_index is present and value > 0.2, fear/sorrow get boosted."""
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        # fi.value = 0.5*0.3 + 0.5*0.3 + 0.5*0.2 + 0.5*0.2 = 0.5
        state = _zero_emotion_state(fear_index=fi, emotions={"fear": 0.1})
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)

        # Fear should be boosted beyond simple decay
        simple_decay = 0.1 * (DECAY_RATE ** 1.0)
        assert result.emotions.fear > simple_decay

    def test_no_responsibility_influence_by_default(self):
        """Without passing responsibility_influence, behavior is unchanged."""
        state = _zero_emotion_state(emotions={"joy": 0.5})
        percept = _neutral_percept()

        result = react(percept, state, delta_time=1.0)
        # Should simply decay
        assert result.emotions.joy == pytest.approx(0.5 * (DECAY_RATE ** 1.0), abs=1e-4)

    def test_full_pipeline_with_all_features(self):
        """Smoke test: react with emotion, valence, responsibility, amplitude."""
        fi = FearIndex(identity_risk=0.3, attachment_risk=0.3,
                       continuity_risk=0.2, projection_risk=0.2)
        state = _zero_emotion_state(
            emotions={"joy": 0.3, "fear": 0.2},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.1, arousal=0.3),
            fear_index=fi,
        )
        percept = _neutral_percept(emotion="happy", emotion_valence=0.5, intent="sharing")
        influence = ResponsibilityInfluence(anxiety_baseline=0.1, fear_amplification=0.2)
        result = react(
            percept, state, delta_time=2.0,
            responsibility_influence=influence,
            amplitude_modifier=1.3,
        )

        # Just verify the result is a valid PsycheState with sane values
        assert isinstance(result, PsycheState)
        for field in EmotionVector.model_fields:
            val = getattr(result.emotions, field)
            assert 0.0 <= val <= 1.0, f"emotion {field} = {val}"
        for field in ["social", "curiosity", "expression"]:
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0, f"drive {field} = {val}"
        assert -1.0 <= result.mood.valence <= 1.0
        assert 0.0 <= result.mood.arousal <= 1.0
