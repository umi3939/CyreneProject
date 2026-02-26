"""
tests/test_drive_dynamics_state_dependent.py - Tests for drive dynamics state-dependent computation

Verifies:
1. DriveContextInputs construction and defaults
2. Section 1: Emotion-drive coupling (断面1)
3. Section 2: Drive interaction (断面2)
4. Section 3: Goal hierarchy presence (断面3)
5. Section 4: Time passage (断面4)
6. Section 5: Arousal-drive (断面5)
7. Composite: compute_state_dependent_drive_changes
8. Safety valves (安全弁 1-6)
9. Integration with react() function
10. State-dependency verification (different states produce different results)
11. Backward compatibility (react without drive_context)
"""

import pytest

from psyche.reaction import (
    DriveContextInputs,
    _SECTION_BAND,
    _TOTAL_CHANGE_LIMIT,
    _clamp,
    _compute_arousal_drive,
    _compute_drive_interaction,
    _compute_emotion_drive_coupling,
    _compute_goal_hierarchy,
    _compute_time_passage,
    compute_state_dependent_drive_changes,
    react,
)
from psyche.responsibility import ResponsibilityInfluence
from psyche.state import DriveVector, EmotionVector, Mood, Percept, PsycheState


# ── Helpers ───────────────────────────────────────────────────

def _default_ctx(**overrides) -> DriveContextInputs:
    """Create a DriveContextInputs with reasonable defaults."""
    kw = dict(
        emotions={"joy": 0.0, "anger": 0.0, "sorrow": 0.0, "fear": 0.0,
                  "surprise": 0.0, "love": 0.0, "fun": 0.0},
        mood_valence=0.0,
        mood_arousal=0.3,
        drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        delta_time=1.0,
        percept_intent="unknown",
        percept_emotion="neutral",
        percept_valence=0.0,
        fear_level=0.0,
    )
    kw.update(overrides)
    return DriveContextInputs(**kw)


def _zero_emotion_state(**overrides) -> PsycheState:
    emo_kw = {k: 0.0 for k in EmotionVector.model_fields}
    emo_kw.update(overrides.pop("emotions", {}))
    return PsycheState(
        emotions=EmotionVector(**emo_kw),
        drives=overrides.pop("drives", DriveVector(social=0.5, curiosity=0.5, expression=0.5)),
        mood=overrides.pop("mood", Mood(valence=0.0, arousal=0.3)),
        **overrides,
    )


def _neutral_percept(**overrides) -> Percept:
    kw = dict(text="", meaning="", emotion="neutral", intent="unknown",
              topics=[], sentiment=0.0, emotion_valence=0.0)
    kw.update(overrides)
    return Percept(**kw)


# =============================================================================
# Test 1: DriveContextInputs construction and defaults
# =============================================================================

class TestDriveContextInputs:
    def test_default_construction(self):
        """Default DriveContextInputs has all optional fields as None or defaults."""
        ctx = DriveContextInputs()
        assert ctx.emotions is None
        assert ctx.mood_valence is None
        assert ctx.mood_arousal is None
        assert ctx.drives is None
        assert ctx.has_transient_goal is False
        assert ctx.persistent_commitment_count == 0
        assert ctx.has_scoped_goal is False
        assert ctx.time_density_label is None
        assert ctx.delta_time == 1.0
        assert ctx.percept_intent is None
        assert ctx.percept_emotion is None
        assert ctx.percept_valence is None
        assert ctx.fear_level == 0.0

    def test_full_construction(self):
        """DriveContextInputs with all fields set."""
        ctx = _default_ctx(
            has_transient_goal=True,
            persistent_commitment_count=2,
            has_scoped_goal=True,
            time_density_label="dense",
            fear_level=0.3,
        )
        assert ctx.has_transient_goal is True
        assert ctx.persistent_commitment_count == 2
        assert ctx.has_scoped_goal is True
        assert ctx.time_density_label == "dense"
        assert ctx.fear_level == 0.3


# =============================================================================
# Test 2: Section 1 - Emotion-drive coupling
# =============================================================================

class TestEmotionDriveCoupling:
    def test_none_emotions_returns_zero(self):
        """When emotions are None, all contributions are zero (safety valve 5)."""
        ctx = DriveContextInputs()
        result = _compute_emotion_drive_coupling(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_positive_emotions_affect_social(self):
        """Positive emotions should contribute positively to social drive."""
        ctx = _default_ctx(emotions={"joy": 0.8, "love": 0.5, "fun": 0.3,
                                     "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                     "surprise": 0.0})
        result = _compute_emotion_drive_coupling(ctx)
        assert result["social"] > 0.0

    def test_negative_emotions_reduce_social(self):
        """Negative emotions should reduce social drive contribution."""
        ctx = _default_ctx(emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                                     "sorrow": 0.8, "anger": 0.5, "fear": 0.3,
                                     "surprise": 0.0})
        result = _compute_emotion_drive_coupling(ctx)
        assert result["social"] < 0.0

    def test_surprise_boosts_curiosity(self):
        """Surprise should boost curiosity contribution."""
        ctx_no_surprise = _default_ctx(emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                                                  "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                                  "surprise": 0.0})
        ctx_surprise = _default_ctx(emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                                               "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                               "surprise": 0.8})
        r_no = _compute_emotion_drive_coupling(ctx_no_surprise)
        r_yes = _compute_emotion_drive_coupling(ctx_surprise)
        assert r_yes["curiosity"] > r_no["curiosity"]

    def test_high_emotion_boosts_expression(self):
        """High max emotion should boost expression contribution."""
        ctx_low = _default_ctx(emotions={"joy": 0.1, "love": 0.0, "fun": 0.0,
                                          "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                          "surprise": 0.0})
        ctx_high = _default_ctx(emotions={"joy": 0.9, "love": 0.0, "fun": 0.0,
                                           "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                           "surprise": 0.0})
        r_low = _compute_emotion_drive_coupling(ctx_low)
        r_high = _compute_emotion_drive_coupling(ctx_high)
        assert r_high["expression"] > r_low["expression"]

    def test_negative_mood_dampens_social_recovery(self):
        """Negative mood valence should dampen social recovery from positive emotions."""
        # Use moderate emotions so raw value stays below band ceiling (0.06)
        # positive_sum=0.5 -> social_raw=0.05, below band limit
        ctx_pos_mood = _default_ctx(
            emotions={"joy": 0.3, "love": 0.1, "fun": 0.1,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            mood_valence=0.5)
        ctx_neg_mood = _default_ctx(
            emotions={"joy": 0.3, "love": 0.1, "fun": 0.1,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            mood_valence=-0.8)
        r_pos = _compute_emotion_drive_coupling(ctx_pos_mood)
        r_neg = _compute_emotion_drive_coupling(ctx_neg_mood)
        # Negative mood should produce a lower (or dampened) social contribution
        assert r_neg["social"] < r_pos["social"]

    def test_arousal_amplifies_curiosity(self):
        """Higher arousal should amplify curiosity effects from surprise."""
        ctx_low_arousal = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.5},
            mood_arousal=0.1)
        ctx_high_arousal = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.5},
            mood_arousal=0.9)
        r_low = _compute_emotion_drive_coupling(ctx_low_arousal)
        r_high = _compute_emotion_drive_coupling(ctx_high_arousal)
        assert r_high["curiosity"] > r_low["curiosity"]

    def test_fear_suppresses_expression(self):
        """Fear should suppress expression contribution."""
        ctx_no_fear = _default_ctx(emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                                              "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                              "surprise": 0.0})
        ctx_fear = _default_ctx(emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                                           "sorrow": 0.0, "anger": 0.0, "fear": 0.8,
                                           "surprise": 0.0})
        r_no = _compute_emotion_drive_coupling(ctx_no_fear)
        r_fear = _compute_emotion_drive_coupling(ctx_fear)
        assert r_fear["expression"] < r_no["expression"]

    def test_clamped_to_band(self):
        """All values must be within the section band limits (safety valve 1)."""
        ctx = _default_ctx(emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                      "surprise": 1.0},
                           mood_arousal=1.0)
        result = _compute_emotion_drive_coupling(ctx)
        band = _SECTION_BAND["emotion_drive_coupling"]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis]


# =============================================================================
# Test 3: Section 2 - Drive interaction
# =============================================================================

class TestDriveInteraction:
    def test_none_drives_returns_zero(self):
        """When drives are None, all contributions are zero (safety valve 5)."""
        ctx = DriveContextInputs()
        result = _compute_drive_interaction(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_high_curiosity_boosts_social(self):
        """High curiosity should slightly boost social drive."""
        ctx_low = _default_ctx(drives={"social": 0.5, "curiosity": 0.3, "expression": 0.5})
        ctx_high = _default_ctx(drives={"social": 0.5, "curiosity": 0.9, "expression": 0.5})
        r_low = _compute_drive_interaction(ctx_low)
        r_high = _compute_drive_interaction(ctx_high)
        assert r_high["social"] > r_low["social"]

    def test_high_expression_reduces_curiosity(self):
        """High expression drive should slightly reduce curiosity."""
        ctx_low = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.3})
        ctx_high = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.9})
        r_low = _compute_drive_interaction(ctx_low)
        r_high = _compute_drive_interaction(ctx_high)
        assert r_high["curiosity"] < r_low["curiosity"]

    def test_high_social_boosts_expression(self):
        """High social drive should slightly boost expression."""
        ctx_low = _default_ctx(drives={"social": 0.3, "curiosity": 0.5, "expression": 0.5})
        ctx_high = _default_ctx(drives={"social": 0.9, "curiosity": 0.5, "expression": 0.5})
        r_low = _compute_drive_interaction(ctx_low)
        r_high = _compute_drive_interaction(ctx_high)
        assert r_high["expression"] > r_low["expression"]

    def test_band_narrower_than_direct_input(self):
        """Drive interaction band must be narrower than direct input bands (safety valve 4)."""
        interaction_band = _SECTION_BAND["drive_interaction"]
        for section in ("emotion_drive_coupling", "time_passage"):
            direct_band = _SECTION_BAND[section]
            for axis in ("social", "curiosity", "expression"):
                assert interaction_band[axis] < direct_band[axis], (
                    f"Interaction band {axis}={interaction_band[axis]} "
                    f"not less than {section} band {axis}={direct_band[axis]}"
                )

    def test_clamped_to_band(self):
        """All values must be within the section band limits."""
        ctx = _default_ctx(drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0})
        result = _compute_drive_interaction(ctx)
        band = _SECTION_BAND["drive_interaction"]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis]

    def test_negative_mood_weakens_interaction(self):
        """Negative mood should weaken the curiosity->social interaction."""
        ctx_pos = _default_ctx(drives={"social": 0.5, "curiosity": 0.9, "expression": 0.5},
                               mood_valence=0.5)
        ctx_neg = _default_ctx(drives={"social": 0.5, "curiosity": 0.9, "expression": 0.5},
                               mood_valence=-0.8)
        r_pos = _compute_drive_interaction(ctx_pos)
        r_neg = _compute_drive_interaction(ctx_neg)
        assert abs(r_neg["social"]) < abs(r_pos["social"])


# =============================================================================
# Test 4: Section 3 - Goal hierarchy presence
# =============================================================================

class TestGoalHierarchy:
    def test_no_goals_slight_decrease(self):
        """Without any goals, curiosity and expression should slightly decrease."""
        ctx = _default_ctx()
        result = _compute_goal_hierarchy(ctx)
        assert result["curiosity"] < 0.0
        assert result["expression"] < 0.0
        assert result["social"] == 0.0

    def test_transient_goal_boosts_curiosity(self):
        """Having a transient goal should boost curiosity."""
        ctx_no = _default_ctx()
        ctx_yes = _default_ctx(has_transient_goal=True)
        r_no = _compute_goal_hierarchy(ctx_no)
        r_yes = _compute_goal_hierarchy(ctx_yes)
        assert r_yes["curiosity"] > r_no["curiosity"]

    def test_persistent_commitment_boosts_social(self):
        """Having persistent commitments should boost social drive."""
        ctx_no = _default_ctx()
        ctx_yes = _default_ctx(persistent_commitment_count=2)
        r_no = _compute_goal_hierarchy(ctx_no)
        r_yes = _compute_goal_hierarchy(ctx_yes)
        assert r_yes["social"] > r_no["social"]

    def test_more_goals_stronger_effect(self):
        """More goals should produce a stronger effect (up to cap)."""
        ctx_1 = _default_ctx(has_transient_goal=True)
        ctx_3 = _default_ctx(has_transient_goal=True, persistent_commitment_count=2)
        r_1 = _compute_goal_hierarchy(ctx_1)
        r_3 = _compute_goal_hierarchy(ctx_3)
        assert r_3["curiosity"] > r_1["curiosity"]

    def test_goal_count_capped(self):
        """Effect from goals should cap at 3."""
        ctx_3 = _default_ctx(has_transient_goal=True, persistent_commitment_count=2)
        ctx_5 = _default_ctx(has_transient_goal=True, persistent_commitment_count=3,
                             has_scoped_goal=True)
        r_3 = _compute_goal_hierarchy(ctx_3)
        r_5 = _compute_goal_hierarchy(ctx_5)
        # Both should hit the cap (factor = 1.0) so same curiosity contribution
        assert r_3["curiosity"] == pytest.approx(r_5["curiosity"], abs=1e-6)

    def test_clamped_to_band(self):
        """All values must be within the section band limits."""
        ctx = _default_ctx(has_transient_goal=True, persistent_commitment_count=5,
                           has_scoped_goal=True)
        result = _compute_goal_hierarchy(ctx)
        band = _SECTION_BAND["goal_hierarchy"]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis]


# =============================================================================
# Test 5: Section 4 - Time passage
# =============================================================================

class TestTimePassage:
    def test_normal_density_social_decrease(self):
        """With normal density, social drive should decrease (time + conversation)."""
        ctx = _default_ctx(time_density_label=None)
        result = _compute_time_passage(ctx)
        # social_raw = 0.02*1 - 0.12 = -0.10 -> clamped to band
        assert result["social"] < 0.0

    def test_sparse_density_social_increase(self):
        """With sparse density (long interval), social drive should increase (loneliness)."""
        ctx = _default_ctx(time_density_label="sparse", delta_time=2.0)
        result = _compute_time_passage(ctx)
        assert result["social"] > 0.0

    def test_dense_density_social_decrease(self):
        """With dense density (short interval), social should decrease (satisfaction)."""
        ctx = _default_ctx(time_density_label="dense", delta_time=1.0)
        result = _compute_time_passage(ctx)
        assert result["social"] < 0.0

    def test_sharing_reduces_curiosity(self):
        """Sharing intent should reduce curiosity."""
        ctx_unknown = _default_ctx(percept_intent="unknown")
        ctx_sharing = _default_ctx(percept_intent="sharing")
        r_unknown = _compute_time_passage(ctx_unknown)
        r_sharing = _compute_time_passage(ctx_sharing)
        assert r_sharing["curiosity"] < r_unknown["curiosity"]

    def test_question_reduces_curiosity(self):
        """Question intent should reduce curiosity."""
        ctx_unknown = _default_ctx(percept_intent="unknown")
        ctx_question = _default_ctx(percept_intent="question")
        r_unknown = _compute_time_passage(ctx_unknown)
        r_question = _compute_time_passage(ctx_question)
        assert r_question["curiosity"] < r_unknown["curiosity"]

    def test_high_emotion_boosts_expression(self):
        """High max emotion should boost expression via time passage."""
        ctx_low = _default_ctx(emotions={"joy": 0.1, "love": 0.0, "fun": 0.0,
                                          "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                          "surprise": 0.0})
        ctx_high = _default_ctx(emotions={"joy": 0.9, "love": 0.0, "fun": 0.0,
                                           "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                           "surprise": 0.0})
        r_low = _compute_time_passage(ctx_low)
        r_high = _compute_time_passage(ctx_high)
        assert r_high["expression"] > r_low["expression"]

    def test_expression_time_decay(self):
        """Expression drive should decay over time."""
        ctx = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.8},
                           delta_time=5.0)
        result = _compute_time_passage(ctx)
        # High expression + long dt should produce negative expression delta
        assert result["expression"] < 0.0

    def test_clamped_to_band(self):
        """All values must be within the section band limits."""
        ctx = _default_ctx(delta_time=10.0, time_density_label="sparse",
                           emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                                     "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                     "surprise": 0.0})
        result = _compute_time_passage(ctx)
        band = _SECTION_BAND["time_passage"]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis]


# =============================================================================
# Test 6: Section 5 - Arousal-drive
# =============================================================================

class TestArousalDrive:
    def test_neutral_arousal_returns_zero(self):
        """Mid-range arousal (0.3-0.6) should produce near-zero effects."""
        ctx = _default_ctx(mood_arousal=0.45)
        result = _compute_arousal_drive(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == pytest.approx(0.0, abs=1e-6)

    def test_high_arousal_positive_effect(self):
        """High arousal (>0.6) should produce positive drive effects."""
        ctx = _default_ctx(mood_arousal=0.9)
        result = _compute_arousal_drive(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] > 0.0

    def test_low_arousal_negative_effect(self):
        """Low arousal (<0.3) should produce negative drive effects."""
        ctx = _default_ctx(mood_arousal=0.1)
        result = _compute_arousal_drive(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] < 0.0

    def test_fear_suppresses_arousal_effect(self):
        """High fear should suppress the arousal->drive amplification."""
        # Use moderate arousal so raw values stay below band ceiling (0.04)
        # arousal=0.65 -> scale=(0.65-0.6)*0.5=0.025, within band
        ctx_no_fear = _default_ctx(mood_arousal=0.65, fear_level=0.0)
        ctx_fear = _default_ctx(mood_arousal=0.65, fear_level=0.8)
        r_no = _compute_arousal_drive(ctx_no_fear)
        r_fear = _compute_arousal_drive(ctx_fear)
        for axis in ("social", "curiosity", "expression"):
            assert abs(r_fear[axis]) < abs(r_no[axis])

    def test_none_arousal_uses_default(self):
        """When mood_arousal is None, default 0.3 is used (neutral range)."""
        ctx = DriveContextInputs()
        result = _compute_arousal_drive(ctx)
        # 0.3 is at boundary -> scale = 0.0
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == pytest.approx(0.0, abs=1e-6)

    def test_clamped_to_band(self):
        """All values must be within the section band limits."""
        ctx = _default_ctx(mood_arousal=1.0)
        result = _compute_arousal_drive(ctx)
        band = _SECTION_BAND["arousal_drive"]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis]


# =============================================================================
# Test 7: Composite - compute_state_dependent_drive_changes
# =============================================================================

class TestCompositeComputation:
    def test_all_none_returns_near_zero(self):
        """With all None inputs, the composite result should be near zero."""
        ctx = DriveContextInputs()
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            # Only time_passage has non-zero for default (normal density path)
            assert -_TOTAL_CHANGE_LIMIT <= result[axis] <= _TOTAL_CHANGE_LIMIT

    def test_different_states_produce_different_results(self):
        """Different internal states should produce different drive changes."""
        ctx_a = _default_ctx(
            emotions={"joy": 0.8, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            mood_valence=0.5,
            mood_arousal=0.7)
        ctx_b = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.8, "anger": 0.5, "fear": 0.3, "surprise": 0.0},
            mood_valence=-0.5,
            mood_arousal=0.2)
        r_a = compute_state_dependent_drive_changes(ctx_a)
        r_b = compute_state_dependent_drive_changes(ctx_b)
        # At least one axis should differ
        differs = any(abs(r_a[ax] - r_b[ax]) > 0.001 for ax in ("social", "curiosity", "expression"))
        assert differs, "Different states should produce different drive changes"

    def test_total_clamped(self):
        """Composite result should be clamped to _TOTAL_CHANGE_LIMIT."""
        # Create extreme inputs to try to exceed limit
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 1.0},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            time_density_label="sparse",
            delta_time=10.0,
        )
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert -_TOTAL_CHANGE_LIMIT <= result[axis] <= _TOTAL_CHANGE_LIMIT

    def test_returns_all_three_axes(self):
        """Result must contain all three drive axes."""
        ctx = _default_ctx()
        result = compute_state_dependent_drive_changes(ctx)
        assert "social" in result
        assert "curiosity" in result
        assert "expression" in result

    def test_pure_function(self):
        """Same input should produce same output (pure function, safety valve 6)."""
        ctx = _default_ctx(
            emotions={"joy": 0.5, "love": 0.3, "fun": 0.1,
                      "sorrow": 0.1, "anger": 0.0, "fear": 0.0, "surprise": 0.2},
            mood_valence=0.2,
            mood_arousal=0.5)
        r1 = compute_state_dependent_drive_changes(ctx)
        r2 = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert r1[axis] == r2[axis]


# =============================================================================
# Test 8: Safety valves
# =============================================================================

class TestSafetyValves:
    def test_sv1_section_band_limits(self):
        """Safety valve 1: each section's output must be within its band."""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 1.0, "anger": 1.0, "fear": 1.0, "surprise": 1.0},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
        )
        sections = {
            "emotion_drive_coupling": _compute_emotion_drive_coupling(ctx),
            "drive_interaction": _compute_drive_interaction(ctx),
            "goal_hierarchy": _compute_goal_hierarchy(ctx),
            "time_passage": _compute_time_passage(ctx),
            "arousal_drive": _compute_arousal_drive(ctx),
        }
        for section_name, result in sections.items():
            band = _SECTION_BAND[section_name]
            for axis in ("social", "curiosity", "expression"):
                assert -band[axis] <= result[axis] <= band[axis], (
                    f"{section_name}.{axis}={result[axis]} outside band +-{band[axis]}"
                )

    def test_sv2_total_change_limit(self):
        """Safety valve 2: composite output must be within total change limit."""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 1.0},
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
        )
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT + 1e-9

    def test_sv3_drive_clamp_after_application(self):
        """Safety valve 3: drive values must be clamped to [0, 1] after application."""
        state = _zero_emotion_state(drives=DriveVector(social=0.01, curiosity=0.99, expression=0.5))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        for field in ("social", "curiosity", "expression"):
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0, f"drive {field} = {val} out of [0, 1]"

    def test_sv4_interaction_band_narrower(self):
        """Safety valve 4: drive_interaction band < direct input bands."""
        ib = _SECTION_BAND["drive_interaction"]
        for section in ("emotion_drive_coupling", "time_passage", "goal_hierarchy"):
            db = _SECTION_BAND[section]
            for axis in ("social", "curiosity", "expression"):
                assert ib[axis] <= db[axis], (
                    f"Interaction {axis}={ib[axis]} not <= {section} {axis}={db[axis]}"
                )

    def test_sv5_missing_inputs_neutral(self):
        """Safety valve 5: missing inputs produce neutral (zero) contributions."""
        ctx_empty = DriveContextInputs()
        r_coupling = _compute_emotion_drive_coupling(ctx_empty)
        r_interaction = _compute_drive_interaction(ctx_empty)
        assert r_coupling == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}
        assert r_interaction == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_sv6_no_state_accumulation(self):
        """Safety valve 6: calling compute twice with same input gives same result."""
        ctx = _default_ctx(
            emotions={"joy": 0.5, "love": 0.3, "fun": 0.1,
                      "sorrow": 0.1, "anger": 0.0, "fear": 0.0, "surprise": 0.2})
        r1 = compute_state_dependent_drive_changes(ctx)
        r2 = compute_state_dependent_drive_changes(ctx)
        assert r1 == r2

    def test_total_limit_matches_old_max(self):
        """Total change limit should be at same level as old fixed max (~0.15)."""
        assert _TOTAL_CHANGE_LIMIT == pytest.approx(0.15, abs=0.01)

    def test_section_bands_within_vo_limit(self):
        """All section band values should be <= value_orientation max_bias_strength (0.15)."""
        for section_name, band in _SECTION_BAND.items():
            for axis, val in band.items():
                assert val <= 0.15, (
                    f"{section_name}.{axis}={val} exceeds VO max_bias_strength 0.15"
                )


# =============================================================================
# Test 9: Integration with react()
# =============================================================================

class TestReactIntegration:
    def test_react_with_default_context(self):
        """react() should work without explicit drive_context."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        assert isinstance(result, PsycheState)
        for field in ("social", "curiosity", "expression"):
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0

    def test_react_with_explicit_context(self):
        """react() should use the provided drive_context."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        ctx = DriveContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=2,
            time_density_label="sparse",
        )
        result = react(percept, state, delta_time=1.0, drive_context=ctx)
        assert isinstance(result, PsycheState)

    def test_react_context_gets_populated(self):
        """react() should populate missing context fields from state/percept."""
        state = _zero_emotion_state(
            emotions={"joy": 0.5},
            mood=Mood(valence=0.3, arousal=0.6),
        )
        percept = _neutral_percept(intent="sharing", emotion="happy", emotion_valence=0.5)
        ctx = DriveContextInputs(
            has_transient_goal=True,
        )
        result = react(percept, state, delta_time=2.0, drive_context=ctx)
        # Verify context was updated by checking result is valid
        assert isinstance(result, PsycheState)

    def test_react_with_responsibility_and_context(self):
        """react() with both responsibility and drive_context should work."""
        state = _zero_emotion_state(
            emotions={"fear": 0.3},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        percept = _neutral_percept(emotion="sad", emotion_valence=-0.5)
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.3)
        ctx = DriveContextInputs(
            has_transient_goal=True,
            time_density_label="dense",
        )
        result = react(percept, state, delta_time=1.0,
                       responsibility_influence=influence,
                       drive_context=ctx)
        assert isinstance(result, PsycheState)
        for field in EmotionVector.model_fields:
            val = getattr(result.emotions, field)
            assert 0.0 <= val <= 1.0
        for field in ("social", "curiosity", "expression"):
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0

    def test_react_immutability_with_context(self):
        """react() with drive_context must not mutate the input state."""
        state = _zero_emotion_state(
            emotions={"joy": 0.5},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        original_drives = state.drives.model_dump().copy()
        original_emotions = state.emotions.as_dict().copy()
        ctx = DriveContextInputs(has_transient_goal=True)
        percept = _neutral_percept(emotion="happy")
        _ = react(percept, state, delta_time=1.0, drive_context=ctx)
        assert state.drives.model_dump() == original_drives
        assert state.emotions.as_dict() == original_emotions


# =============================================================================
# Test 10: State-dependency verification
# =============================================================================

class TestStateDependency:
    def test_different_emotions_different_drives(self):
        """Different emotional states should lead to different drive changes."""
        state_happy = _zero_emotion_state(emotions={"joy": 0.8})
        state_sad = _zero_emotion_state(emotions={"sorrow": 0.8})
        percept = _neutral_percept()
        r_happy = react(percept, state_happy, delta_time=1.0)
        r_sad = react(percept, state_sad, delta_time=1.0)
        # At least one drive axis should differ
        differs = any(
            abs(getattr(r_happy.drives, ax) - getattr(r_sad.drives, ax)) > 0.001
            for ax in ("social", "curiosity", "expression")
        )
        assert differs, "Different emotions should produce different drive results"

    def test_different_moods_different_drives(self):
        """Different mood states should lead to different drive changes."""
        state_pos = _zero_emotion_state(
            emotions={"joy": 0.5},
            mood=Mood(valence=0.8, arousal=0.8))
        state_neg = _zero_emotion_state(
            emotions={"joy": 0.5},
            mood=Mood(valence=-0.8, arousal=0.1))
        percept = _neutral_percept()
        r_pos = react(percept, state_pos, delta_time=1.0)
        r_neg = react(percept, state_neg, delta_time=1.0)
        differs = any(
            abs(getattr(r_pos.drives, ax) - getattr(r_neg.drives, ax)) > 0.001
            for ax in ("social", "curiosity", "expression")
        )
        assert differs, "Different moods should produce different drive results"

    def test_different_goals_different_drives(self):
        """Goal presence should affect drive changes."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        ctx_no_goals = DriveContextInputs()
        ctx_goals = DriveContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=2,
        )
        r_no = react(percept, state, delta_time=1.0, drive_context=ctx_no_goals)
        r_yes = react(percept, state, delta_time=1.0, drive_context=ctx_goals)
        differs = any(
            abs(getattr(r_no.drives, ax) - getattr(r_yes.drives, ax)) > 0.001
            for ax in ("social", "curiosity", "expression")
        )
        assert differs, "Goal presence should affect drive results"

    def test_different_time_densities_different_drives(self):
        """Different time densities should produce different drive changes."""
        state = _zero_emotion_state()
        percept = _neutral_percept()
        ctx_sparse = DriveContextInputs(time_density_label="sparse")
        ctx_dense = DriveContextInputs(time_density_label="dense")
        r_sparse = react(percept, state, delta_time=1.0, drive_context=ctx_sparse)
        r_dense = react(percept, state, delta_time=1.0, drive_context=ctx_dense)
        # Social should differ: sparse increases, dense decreases
        assert r_sparse.drives.social != pytest.approx(r_dense.drives.social, abs=0.001)


# =============================================================================
# Test 11: Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_react_without_context_works(self):
        """react() without drive_context should still produce valid results."""
        state = _zero_emotion_state(
            emotions={"joy": 0.5, "sorrow": 0.2},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.1, arousal=0.4),
        )
        percept = _neutral_percept(emotion="happy", emotion_valence=0.5, intent="sharing")
        result = react(percept, state, delta_time=2.0)
        assert isinstance(result, PsycheState)
        assert result is not state
        # Emotions should still update correctly
        assert result.emotions.joy > 0.0
        # Drives should be within valid range
        for field in ("social", "curiosity", "expression"):
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0

    def test_react_with_all_old_params_works(self):
        """react() with all original parameters should still work."""
        from psyche.pillars import FearIndex
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
        assert isinstance(result, PsycheState)
        for field in EmotionVector.model_fields:
            val = getattr(result.emotions, field)
            assert 0.0 <= val <= 1.0
        for field in ("social", "curiosity", "expression"):
            val = getattr(result.drives, field)
            assert 0.0 <= val <= 1.0
        assert -1.0 <= result.mood.valence <= 1.0
        assert 0.0 <= result.mood.arousal <= 1.0

    def test_reaction_with_stm_still_works(self):
        """react_with_stm wrapper should work with updated react()."""
        from psyche.reaction_with_stm import react_with_stm
        from psyche.short_term_loop import create_loop_state
        state = _zero_emotion_state(
            emotions={"joy": 0.4},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        loop_state = create_loop_state()
        percept = _neutral_percept(emotion="happy")
        new_psyche, new_loop, loop_result = react_with_stm(
            percept=percept,
            psyche_state=state,
            loop_state=loop_state,
            delta_time=1.0,
        )
        assert isinstance(new_psyche, PsycheState)
        for field in ("social", "curiosity", "expression"):
            val = getattr(new_psyche.drives, field)
            assert 0.0 <= val <= 1.0


# =============================================================================
# Test 12: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_zero_delta_time(self):
        """Delta time of 0 should produce minimal changes."""
        ctx = _default_ctx(delta_time=0.0)
        result = compute_state_dependent_drive_changes(ctx)
        # With dt=0, time_passage section has minimal contributions
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_very_large_delta_time(self):
        """Very large delta time should be bounded by section bands."""
        ctx = _default_ctx(delta_time=1000.0, time_density_label="sparse")
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_all_emotions_at_max(self):
        """All emotions at 1.0 should be handled correctly."""
        ctx = _default_ctx(emotions={"joy": 1.0, "anger": 1.0, "sorrow": 1.0,
                                      "fear": 1.0, "surprise": 1.0, "love": 1.0,
                                      "fun": 1.0})
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_all_drives_at_zero(self):
        """All drives at 0.0 should be handled correctly."""
        ctx = _default_ctx(drives={"social": 0.0, "curiosity": 0.0, "expression": 0.0})
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_all_drives_at_max(self):
        """All drives at 1.0 should be handled correctly."""
        ctx = _default_ctx(drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0})
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_extreme_negative_mood(self):
        """Extreme negative mood should be handled correctly."""
        ctx = _default_ctx(mood_valence=-1.0, mood_arousal=0.0)
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_extreme_positive_mood(self):
        """Extreme positive mood should be handled correctly."""
        ctx = _default_ctx(mood_valence=1.0, mood_arousal=1.0)
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT

    def test_high_fear_level(self):
        """High fear level should suppress arousal effects."""
        ctx = _default_ctx(mood_arousal=0.9, fear_level=0.9)
        result = _compute_arousal_drive(ctx)
        # Should still be positive but suppressed
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] >= 0.0
            assert result[axis] <= _SECTION_BAND["arousal_drive"][axis]
