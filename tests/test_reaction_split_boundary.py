"""
tests/test_reaction_split_boundary.py - Reaction Split Boundary Tests

3-layer test structure for verifying the structural integrity of the reaction module split:
  Layer 1: Re-export identity verification (reaction.py -> reaction_drive_dynamics.py / reaction_mood_update.py)
  Layer 2: Independent operation tests for each split-off module
  Layer 3: Call-path integration tests through orchestrator_1tick_phases context builders

This file tests ONLY the structural properties that emerged from the split.
It does NOT duplicate existing test_reaction.py coverage (emotion stimulus, decay, fear, etc.).
No psyche logic changes. No quality/desirability judgments.
"""

import pytest

# ── Direct imports from split-off modules ──
from psyche.reaction_drive_dynamics import (
    DriveContextInputs as DirectDriveContextInputs,
    _SECTION_BAND as DirectSectionBand,
    _TOTAL_CHANGE_LIMIT as DirectTotalChangeLimit,
    _compute_emotion_drive_coupling as direct_emotion_drive_coupling,
    _compute_drive_interaction as direct_drive_interaction,
    _compute_goal_hierarchy as direct_goal_hierarchy,
    _compute_time_passage as direct_time_passage,
    _compute_arousal_drive as direct_arousal_drive,
    _compute_behavioral_diversity as direct_behavioral_diversity,
    _compute_internal_contradiction as direct_internal_contradiction,
    _compute_result_diversity_return as direct_result_diversity_return,
    compute_state_dependent_drive_changes as direct_compute_drive_changes,
)

from psyche.reaction_mood_update import (
    MoodContextInputs as DirectMoodContextInputs,
    _MOOD_BAND as DirectMoodBand,
    _TRACKING_SPEED_MIN as DirectTrackingSpeedMin,
    _TRACKING_SPEED_MAX as DirectTrackingSpeedMax,
    _MOOD_DELTA_LIMIT as DirectMoodDeltaLimit,
    _derive_mood_targets as direct_derive_mood_targets,
    _derive_tracking_speeds as direct_derive_tracking_speeds,
    compute_autonomous_mood as direct_compute_mood,
)

# ── Re-exported imports from reaction.py ──
from psyche.reaction import (
    DriveContextInputs as ReexportedDriveContextInputs,
    _SECTION_BAND as ReexportedSectionBand,
    _TOTAL_CHANGE_LIMIT as ReexportedTotalChangeLimit,
    _compute_emotion_drive_coupling as reexported_emotion_drive_coupling,
    _compute_drive_interaction as reexported_drive_interaction,
    _compute_goal_hierarchy as reexported_goal_hierarchy,
    _compute_time_passage as reexported_time_passage,
    _compute_arousal_drive as reexported_arousal_drive,
    _compute_behavioral_diversity as reexported_behavioral_diversity,
    _compute_internal_contradiction as reexported_internal_contradiction,
    _compute_result_diversity_return as reexported_result_diversity_return,
    compute_state_dependent_drive_changes as reexported_compute_drive_changes,
    MoodContextInputs as ReexportedMoodContextInputs,
    _MOOD_BAND as ReexportedMoodBand,
    _TRACKING_SPEED_MIN as ReexportedTrackingSpeedMin,
    _TRACKING_SPEED_MAX as ReexportedTrackingSpeedMax,
    _MOOD_DELTA_LIMIT as ReexportedMoodDeltaLimit,
    _derive_mood_targets as reexported_derive_mood_targets,
    _derive_tracking_speeds as reexported_derive_tracking_speeds,
    compute_autonomous_mood as reexported_compute_mood,
    react,
)

from psyche.state import DriveVector, EmotionVector, Mood, Percept, PsycheState


# ── Helpers ──────────────────────────────────────────────────────

def _default_emotions() -> dict[str, float]:
    """Return a neutral emotion dict with all fields at 0.0."""
    return {k: 0.0 for k in EmotionVector.model_fields}


def _mid_emotions() -> dict[str, float]:
    """Return emotion dict with all fields at 0.5."""
    return {k: 0.5 for k in EmotionVector.model_fields}


def _default_drives() -> dict[str, float]:
    """Return default drive dict with all fields at 0.5."""
    return {"social": 0.5, "curiosity": 0.5, "expression": 0.5}


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


# =============================================================================
# Layer 1: Re-export Identity Verification
# =============================================================================

class TestLayer1DriveReexportIdentity:
    """Verify that all drive dynamics elements re-exported by reaction.py
    are the same objects as those in reaction_drive_dynamics.py."""

    def test_drive_context_inputs_identity(self):
        assert ReexportedDriveContextInputs is DirectDriveContextInputs

    def test_section_band_identity(self):
        assert ReexportedSectionBand is DirectSectionBand

    def test_total_change_limit_identity(self):
        assert ReexportedTotalChangeLimit is DirectTotalChangeLimit

    def test_compute_emotion_drive_coupling_identity(self):
        assert reexported_emotion_drive_coupling is direct_emotion_drive_coupling

    def test_compute_drive_interaction_identity(self):
        assert reexported_drive_interaction is direct_drive_interaction

    def test_compute_goal_hierarchy_identity(self):
        assert reexported_goal_hierarchy is direct_goal_hierarchy

    def test_compute_time_passage_identity(self):
        assert reexported_time_passage is direct_time_passage

    def test_compute_arousal_drive_identity(self):
        assert reexported_arousal_drive is direct_arousal_drive

    def test_compute_behavioral_diversity_identity(self):
        assert reexported_behavioral_diversity is direct_behavioral_diversity

    def test_compute_internal_contradiction_identity(self):
        assert reexported_internal_contradiction is direct_internal_contradiction

    def test_compute_result_diversity_return_identity(self):
        assert reexported_result_diversity_return is direct_result_diversity_return

    def test_compute_state_dependent_drive_changes_identity(self):
        assert reexported_compute_drive_changes is direct_compute_drive_changes


class TestLayer1MoodReexportIdentity:
    """Verify that all mood update elements re-exported by reaction.py
    are the same objects as those in reaction_mood_update.py."""

    def test_mood_context_inputs_identity(self):
        assert ReexportedMoodContextInputs is DirectMoodContextInputs

    def test_mood_band_identity(self):
        assert ReexportedMoodBand is DirectMoodBand

    def test_tracking_speed_min_identity(self):
        assert ReexportedTrackingSpeedMin is DirectTrackingSpeedMin

    def test_tracking_speed_max_identity(self):
        assert ReexportedTrackingSpeedMax is DirectTrackingSpeedMax

    def test_mood_delta_limit_identity(self):
        assert ReexportedMoodDeltaLimit is DirectMoodDeltaLimit

    def test_derive_mood_targets_identity(self):
        assert reexported_derive_mood_targets is direct_derive_mood_targets

    def test_derive_tracking_speeds_identity(self):
        assert reexported_derive_tracking_speeds is direct_derive_tracking_speeds

    def test_compute_autonomous_mood_identity(self):
        assert reexported_compute_mood is direct_compute_mood


class TestLayer1CompleteCoverage:
    """Verify completeness: all public elements from split modules
    are re-exported by reaction.py."""

    def test_drive_dynamics_all_public_reexported(self):
        """Every public name in reaction_drive_dynamics should be importable from reaction."""
        import psyche.reaction_drive_dynamics as dd_mod
        import psyche.reaction as reaction_mod

        public_names = [
            n for n in dir(dd_mod)
            if not n.startswith('__') and not n.startswith('_') or n.startswith('_') and not n.startswith('__')
        ]
        # Filter to the actual exported names (dataclass, constants, functions)
        expected = {
            'DriveContextInputs', '_SECTION_BAND', '_TOTAL_CHANGE_LIMIT',
            '_compute_emotion_drive_coupling', '_compute_drive_interaction',
            '_compute_goal_hierarchy', '_compute_time_passage',
            '_compute_arousal_drive', '_compute_behavioral_diversity',
            '_compute_internal_contradiction', '_compute_result_diversity_return',
            'compute_state_dependent_drive_changes',
        }
        for name in expected:
            assert hasattr(reaction_mod, name), f"{name} not re-exported by reaction.py"
            assert getattr(reaction_mod, name) is getattr(dd_mod, name), (
                f"{name} re-exported but not same object"
            )

    def test_mood_update_all_public_reexported(self):
        """Every public name in reaction_mood_update should be importable from reaction."""
        import psyche.reaction_mood_update as mu_mod
        import psyche.reaction as reaction_mod

        expected = {
            'MoodContextInputs', '_MOOD_BAND', '_TRACKING_SPEED_MIN',
            '_TRACKING_SPEED_MAX', '_MOOD_DELTA_LIMIT',
            '_derive_mood_targets', '_derive_tracking_speeds',
            'compute_autonomous_mood',
        }
        for name in expected:
            assert hasattr(reaction_mod, name), f"{name} not re-exported by reaction.py"
            assert getattr(reaction_mod, name) is getattr(mu_mod, name), (
                f"{name} re-exported but not same object"
            )

    def test_no_circular_import_between_split_modules(self):
        """Verify that reaction_drive_dynamics and reaction_mood_update
        do not import from each other."""
        import psyche.reaction_drive_dynamics as dd_mod
        import psyche.reaction_mood_update as mu_mod
        import inspect

        dd_source = inspect.getsource(dd_mod)
        mu_source = inspect.getsource(mu_mod)

        assert "reaction_mood_update" not in dd_source, (
            "reaction_drive_dynamics imports from reaction_mood_update (circular)"
        )
        assert "reaction_drive_dynamics" not in mu_source, (
            "reaction_mood_update imports from reaction_drive_dynamics (circular)"
        )


# =============================================================================
# Layer 2: Independent Operation Tests - Drive Dynamics
# =============================================================================

class TestLayer2DriveContextInputsDefaults:
    """Verify DriveContextInputs default values (directly from reaction_drive_dynamics)."""

    def test_all_optional_fields_default_none_or_neutral(self):
        ctx = DirectDriveContextInputs()
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
        assert ctx.behavioral_diversity_stage_value is None
        assert ctx.contradiction_count is None
        assert ctx.result_diversity_section_key_level is None
        assert ctx.result_diversity_selection_label_level is None
        assert ctx.result_diversity_candidate_variance_level is None
        assert ctx.drive_total_limit_multiplier is None


class TestLayer2DriveSectionsZeroInput:
    """All 8 section functions return zero contributions when input is absent."""

    def test_section1_emotion_drive_coupling_no_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_emotion_drive_coupling(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_section2_drive_interaction_no_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_drive_interaction(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_section3_goal_hierarchy_no_goals(self):
        ctx = DirectDriveContextInputs()
        result = direct_goal_hierarchy(ctx)
        # With no goals, curiosity and expression get -0.01 each (goal absence effect)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectSectionBand["goal_hierarchy"][axis]

    def test_section4_time_passage_default(self):
        ctx = DirectDriveContextInputs()
        result = direct_time_passage(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectSectionBand["time_passage"][axis]

    def test_section5_arousal_drive_default(self):
        ctx = DirectDriveContextInputs()
        result = direct_arousal_drive(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectSectionBand["arousal_drive"][axis]

    def test_section6_behavioral_diversity_no_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_behavioral_diversity(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_section7_internal_contradiction_no_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_internal_contradiction(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_section8_result_diversity_no_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_result_diversity_return(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}


class TestLayer2DriveSectionsBandLimits:
    """All 8 section functions respect their band limits even with extreme inputs."""

    def _assert_within_band(self, result: dict, band_key: str):
        band = DirectSectionBand[band_key]
        for axis in ("social", "curiosity", "expression"):
            assert -band[axis] <= result[axis] <= band[axis], (
                f"{band_key} {axis}: {result[axis]} outside [-{band[axis]}, {band[axis]}]"
            )

    def test_section1_band_limit_extreme_emotions(self):
        ctx = DirectDriveContextInputs(
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            mood_valence=1.0,
            mood_arousal=1.0,
        )
        result = direct_emotion_drive_coupling(ctx)
        self._assert_within_band(result, "emotion_drive_coupling")

    def test_section1_band_limit_extreme_negative(self):
        ctx = DirectDriveContextInputs(
            emotions={"joy": 0.0, "sorrow": 1.0, "anger": 1.0, "fear": 1.0,
                       "surprise": 0.0, "love": 0.0, "fun": 0.0},
            mood_valence=-1.0,
            mood_arousal=0.0,
        )
        result = direct_emotion_drive_coupling(ctx)
        self._assert_within_band(result, "emotion_drive_coupling")

    def test_section2_band_limit_extreme_drives(self):
        ctx = DirectDriveContextInputs(
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            mood_valence=1.0,
        )
        result = direct_drive_interaction(ctx)
        self._assert_within_band(result, "drive_interaction")

    def test_section2_band_limit_zero_drives(self):
        ctx = DirectDriveContextInputs(
            drives={"social": 0.0, "curiosity": 0.0, "expression": 0.0},
            mood_valence=-1.0,
        )
        result = direct_drive_interaction(ctx)
        self._assert_within_band(result, "drive_interaction")

    def test_section3_band_limit_max_goals(self):
        ctx = DirectDriveContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=5,
            has_scoped_goal=True,
        )
        result = direct_goal_hierarchy(ctx)
        self._assert_within_band(result, "goal_hierarchy")

    def test_section4_band_limit_large_delta_time(self):
        ctx = DirectDriveContextInputs(
            delta_time=100.0,
            time_density_label="sparse",
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            percept_intent="sharing",
        )
        result = direct_time_passage(ctx)
        self._assert_within_band(result, "time_passage")

    def test_section5_band_limit_max_arousal(self):
        ctx = DirectDriveContextInputs(
            mood_arousal=1.0,
            fear_level=0.0,
        )
        result = direct_arousal_drive(ctx)
        self._assert_within_band(result, "arousal_drive")

    def test_section5_band_limit_min_arousal_max_fear(self):
        ctx = DirectDriveContextInputs(
            mood_arousal=0.0,
            fear_level=1.0,
        )
        result = direct_arousal_drive(ctx)
        self._assert_within_band(result, "arousal_drive")

    def test_section6_band_limit_max_diversity(self):
        ctx = DirectDriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=1.0,
        )
        result = direct_behavioral_diversity(ctx)
        self._assert_within_band(result, "behavioral_diversity")

    def test_section7_band_limit_max_contradictions(self):
        ctx = DirectDriveContextInputs(
            contradiction_count=100,
            mood_arousal=1.0,
            mood_valence=1.0,
        )
        result = direct_internal_contradiction(ctx)
        self._assert_within_band(result, "internal_contradiction")

    def test_section8_band_limit_max_diversity_values(self):
        ctx = DirectDriveContextInputs(
            result_diversity_section_key_level="level_16_plus",
            result_diversity_selection_label_level="level_16_plus",
            result_diversity_candidate_variance_level="high",
        )
        result = direct_result_diversity_return(ctx)
        self._assert_within_band(result, "result_diversity_return")


class TestLayer2DriveSectionsMidValues:
    """Test section functions with intermediate/mixed inputs."""

    def test_section1_mid_emotions(self):
        ctx = DirectDriveContextInputs(
            emotions=_mid_emotions(),
            mood_valence=0.0,
            mood_arousal=0.5,
        )
        result = direct_emotion_drive_coupling(ctx)
        # Should produce non-zero values for at least some axes
        has_nonzero = any(abs(v) > 1e-9 for v in result.values())
        assert has_nonzero

    def test_section2_mid_drives(self):
        ctx = DirectDriveContextInputs(
            drives=_default_drives(),
            mood_valence=0.0,
        )
        result = direct_drive_interaction(ctx)
        # All drives at 0.5 -> (drv - 0.5) = 0 -> expect near-zero
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) < 0.01

    def test_section3_single_goal(self):
        ctx = DirectDriveContextInputs(has_transient_goal=True)
        result = direct_goal_hierarchy(ctx)
        # With one goal, curiosity and expression should get positive boost
        assert result["curiosity"] > 0.0
        assert result["expression"] > 0.0

    def test_section4_dense_input(self):
        ctx = DirectDriveContextInputs(
            time_density_label="dense",
            delta_time=0.5,
        )
        result = direct_time_passage(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert isinstance(result[axis], float)

    def test_section5_mid_arousal(self):
        ctx = DirectDriveContextInputs(
            mood_arousal=0.5,
            fear_level=0.0,
        )
        result = direct_arousal_drive(ctx)
        # Mid arousal (0.3-0.6) -> scale = 0.0 -> all axes near zero
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) < 0.01

    def test_section6_mid_stage(self):
        ctx = DirectDriveContextInputs(
            behavioral_diversity_stage_value="level_6_10",
            mood_valence=0.0,
        )
        result = direct_behavioral_diversity(ctx)
        # Non-zero contribution with mid-stage
        has_nonzero = any(abs(v) > 1e-9 for v in result.values())
        assert has_nonzero

    def test_section7_mid_contradiction(self):
        ctx = DirectDriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.7,
            mood_valence=0.2,
        )
        result = direct_internal_contradiction(ctx)
        has_nonzero = any(abs(v) > 1e-9 for v in result.values())
        assert has_nonzero

    def test_section8_partial_inputs(self):
        ctx = DirectDriveContextInputs(
            result_diversity_section_key_level="level_6_10",
        )
        result = direct_result_diversity_return(ctx)
        # With partial input (only one of three), should still produce non-zero
        has_nonzero = any(abs(v) > 1e-9 for v in result.values())
        assert has_nonzero


class TestLayer2DriveComposite:
    """Test the composite function compute_state_dependent_drive_changes."""

    def test_all_sections_zero_input(self):
        ctx = DirectDriveContextInputs()
        result = direct_compute_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectTotalChangeLimit

    def test_all_sections_max_input(self):
        ctx = DirectDriveContextInputs(
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            delta_time=1.0,
            time_density_label="dense",
            percept_intent="sharing",
            fear_level=0.0,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=6,
            result_diversity_section_key_level="level_16_plus",
            result_diversity_selection_label_level="level_16_plus",
            result_diversity_candidate_variance_level="high",
        )
        result = direct_compute_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectTotalChangeLimit + 1e-9

    def test_mixed_sections_composite_limit(self):
        ctx = DirectDriveContextInputs(
            emotions=_mid_emotions(),
            mood_valence=-0.5,
            mood_arousal=0.8,
            drives={"social": 0.2, "curiosity": 0.8, "expression": 0.3},
            has_transient_goal=True,
            delta_time=2.0,
            time_density_label="sparse",
            percept_intent="question",
            fear_level=0.5,
        )
        result = direct_compute_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectTotalChangeLimit + 1e-9

    def test_multiplier_expands_limit(self):
        ctx = DirectDriveContextInputs(
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            delta_time=1.0,
            drive_total_limit_multiplier=2.0,
        )
        result = direct_compute_drive_changes(ctx)
        expanded_limit = DirectTotalChangeLimit * 2.0
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= expanded_limit + 1e-9

    def test_multiplier_none_uses_default_limit(self):
        ctx = DirectDriveContextInputs(
            emotions=_mid_emotions(),
            mood_valence=0.3,
            drive_total_limit_multiplier=None,
        )
        result = direct_compute_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= DirectTotalChangeLimit + 1e-9


class TestLayer2DriveCoefficients:
    """Verify coefficient registry reading is correctly wired."""

    def test_section_band_has_all_8_sections(self):
        expected_sections = {
            "emotion_drive_coupling", "drive_interaction", "goal_hierarchy",
            "time_passage", "arousal_drive", "behavioral_diversity",
            "internal_contradiction", "result_diversity_return",
        }
        assert set(DirectSectionBand.keys()) == expected_sections

    def test_section_band_all_three_axes(self):
        for section_name, section_band in DirectSectionBand.items():
            for axis in ("social", "curiosity", "expression"):
                assert axis in section_band, f"{section_name} missing {axis}"
                assert isinstance(section_band[axis], (int, float)), (
                    f"{section_name}.{axis} is not numeric"
                )
                assert section_band[axis] > 0, f"{section_name}.{axis} must be positive"

    def test_total_change_limit_positive(self):
        assert DirectTotalChangeLimit > 0


# =============================================================================
# Layer 2: Independent Operation Tests - Mood Update
# =============================================================================

class TestLayer2MoodContextInputsDefaults:
    """Verify MoodContextInputs default values (directly from reaction_mood_update)."""

    def test_all_fields_default_values(self):
        ctx = DirectMoodContextInputs()
        assert ctx.emotions is None
        assert ctx.drives is None
        assert ctx.current_valence == 0.0
        assert ctx.current_arousal == 0.3
        assert ctx.fear_level == 0.0
        assert ctx.has_transient_goal is False
        assert ctx.persistent_commitment_count == 0
        assert ctx.has_scoped_goal is False
        assert ctx.responsibility_anxiety == 0.0
        assert ctx.time_density_label is None
        assert ctx.delta_time == 1.0
        assert ctx.emotion_return_tracking_speed_modulation_valence is None
        assert ctx.emotion_return_tracking_speed_modulation_arousal is None


class TestLayer2MoodTargets:
    """Test _derive_mood_targets with various inputs."""

    def test_no_input_returns_zero_targets(self):
        ctx = DirectMoodContextInputs()
        tv, ta = direct_derive_mood_targets(ctx)
        assert tv == 0.0
        assert ta == 0.0

    def test_positive_emotions_positive_targets(self):
        ctx = DirectMoodContextInputs(
            emotions={"joy": 0.8, "love": 0.5, "fun": 0.5,
                       "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
        )
        tv, ta = direct_derive_mood_targets(ctx)
        # Positive emotions should push valence target positive
        assert tv > 0.0

    def test_negative_emotions_negative_targets(self):
        ctx = DirectMoodContextInputs(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                       "sorrow": 0.8, "anger": 0.5, "fear": 0.5, "surprise": 0.0},
        )
        tv, ta = direct_derive_mood_targets(ctx)
        assert tv < 0.0

    def test_drive_contribution_to_targets(self):
        ctx_high = DirectMoodContextInputs(
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
        )
        ctx_low = DirectMoodContextInputs(
            drives={"social": 0.0, "curiosity": 0.0, "expression": 0.0},
        )
        tv_high, _ = direct_derive_mood_targets(ctx_high)
        tv_low, _ = direct_derive_mood_targets(ctx_low)
        # Higher drives should produce higher valence target
        assert tv_high > tv_low

    def test_goal_presence_adds_positive_contribution(self):
        ctx_goals = DirectMoodContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=2,
        )
        ctx_no_goals = DirectMoodContextInputs()
        tv_goals, ta_goals = direct_derive_mood_targets(ctx_goals)
        tv_no, ta_no = direct_derive_mood_targets(ctx_no_goals)
        assert tv_goals > tv_no
        assert ta_goals > ta_no

    def test_fear_only_affects_arousal(self):
        ctx = DirectMoodContextInputs(fear_level=0.8)
        tv, ta = direct_derive_mood_targets(ctx)
        assert tv == 0.0  # Fear does not contribute to valence target
        assert ta > 0.0  # Fear contributes to arousal target

    def test_band_limits_respected(self):
        ctx = DirectMoodContextInputs(
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=5,
            has_scoped_goal=True,
            fear_level=1.0,
        )
        tv, ta = direct_derive_mood_targets(ctx)
        # Target values should be bounded by the sum of all band limits
        max_v = sum(DirectMoodBand[src]["valence"] for src in DirectMoodBand)
        max_a = sum(DirectMoodBand[src]["arousal"] for src in DirectMoodBand)
        assert abs(tv) <= max_v + 1e-9
        assert abs(ta) <= max_a + 1e-9


class TestLayer2TrackingSpeeds:
    """Test _derive_tracking_speeds with various inputs."""

    def test_default_returns_within_band(self):
        ctx = DirectMoodContextInputs()
        vs, as_ = direct_derive_tracking_speeds(ctx)
        assert DirectTrackingSpeedMin <= vs <= DirectTrackingSpeedMax
        assert DirectTrackingSpeedMin <= as_ <= DirectTrackingSpeedMax

    def test_high_arousal_increases_valence_speed(self):
        ctx_high = DirectMoodContextInputs(current_arousal=0.9)
        ctx_low = DirectMoodContextInputs(current_arousal=0.1)
        vs_high, _ = direct_derive_tracking_speeds(ctx_high)
        vs_low, _ = direct_derive_tracking_speeds(ctx_low)
        assert vs_high > vs_low

    def test_high_fear_increases_arousal_speed(self):
        ctx_high = DirectMoodContextInputs(fear_level=0.8)
        ctx_low = DirectMoodContextInputs(fear_level=0.0)
        _, as_high = direct_derive_tracking_speeds(ctx_high)
        _, as_low = direct_derive_tracking_speeds(ctx_low)
        assert as_high > as_low

    def test_sparse_density_slows_speeds(self):
        ctx_sparse = DirectMoodContextInputs(time_density_label="sparse")
        ctx_normal = DirectMoodContextInputs(time_density_label="normal")
        vs_sparse, as_sparse = direct_derive_tracking_speeds(ctx_sparse)
        vs_normal, as_normal = direct_derive_tracking_speeds(ctx_normal)
        assert vs_sparse < vs_normal
        assert as_sparse < as_normal

    def test_emotion_return_modulation_adds_speed(self):
        ctx_mod = DirectMoodContextInputs(
            emotion_return_tracking_speed_modulation_valence=0.05,
            emotion_return_tracking_speed_modulation_arousal=0.05,
        )
        ctx_no_mod = DirectMoodContextInputs()
        vs_mod, as_mod = direct_derive_tracking_speeds(ctx_mod)
        vs_no, as_no = direct_derive_tracking_speeds(ctx_no_mod)
        assert vs_mod > vs_no
        assert as_mod > as_no

    def test_speeds_always_within_band(self):
        # Extreme values
        ctx = DirectMoodContextInputs(
            current_arousal=1.0,
            fear_level=1.0,
            emotions={k: 1.0 for k in EmotionVector.model_fields},
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            time_density_label="dense",
            emotion_return_tracking_speed_modulation_valence=1.0,
            emotion_return_tracking_speed_modulation_arousal=1.0,
        )
        vs, as_ = direct_derive_tracking_speeds(ctx)
        assert DirectTrackingSpeedMin <= vs <= DirectTrackingSpeedMax
        assert DirectTrackingSpeedMin <= as_ <= DirectTrackingSpeedMax


class TestLayer2MoodPipeline:
    """Test compute_autonomous_mood 3-stage pipeline."""

    def test_pipeline_no_input_near_current(self):
        ctx = DirectMoodContextInputs(
            current_valence=0.3,
            current_arousal=0.5,
        )
        nv, na = direct_compute_mood(ctx)
        # With no inputs, targets are zero, so mood should move toward zero
        # But with slow tracking, it should stay near current values
        assert abs(nv - 0.3) < 0.1
        assert abs(na - 0.5) < 0.1

    def test_pipeline_valence_moves_toward_target(self):
        ctx = DirectMoodContextInputs(
            emotions={"joy": 1.0, "love": 0.5, "fun": 0.5,
                       "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            current_valence=-0.5,
            current_arousal=0.3,
        )
        nv, _ = direct_compute_mood(ctx)
        # With strong positive emotions and negative current valence,
        # valence should increase (move toward positive target)
        assert nv > -0.5

    def test_pipeline_arousal_moves_toward_target(self):
        ctx = DirectMoodContextInputs(
            emotions={k: 0.0 for k in EmotionVector.model_fields},
            current_valence=0.0,
            current_arousal=0.8,
        )
        _, na = direct_compute_mood(ctx)
        # With zero emotions, arousal target is low, so arousal should decrease
        assert na < 0.8

    def test_pipeline_delta_limit_applied(self):
        ctx = DirectMoodContextInputs(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                       "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            current_valence=-1.0,
            current_arousal=0.0,
        )
        nv, na = direct_compute_mood(ctx)
        # Change should not exceed delta limit
        assert abs(nv - (-1.0)) <= DirectMoodDeltaLimit + 1e-9
        assert abs(na - 0.0) <= DirectMoodDeltaLimit + 1e-9

    def test_pipeline_valence_arousal_independent(self):
        # Valence and arousal should be updated independently
        ctx1 = DirectMoodContextInputs(
            emotions={"joy": 1.0, "love": 0.0, "fun": 0.0,
                       "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            current_valence=0.0,
            current_arousal=0.5,
        )
        ctx2 = DirectMoodContextInputs(
            emotions={"joy": 1.0, "love": 0.0, "fun": 0.0,
                       "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            current_valence=0.5,  # Different starting valence
            current_arousal=0.5,
        )
        nv1, na1 = direct_compute_mood(ctx1)
        nv2, na2 = direct_compute_mood(ctx2)
        # Arousal should be the same since only valence starting point differs
        # (Same target and same speed for arousal in both)
        assert abs(na1 - na2) < 1e-9


class TestLayer2MoodCoefficients:
    """Verify mood coefficient registry reading is correctly wired."""

    def test_mood_band_has_all_sources(self):
        expected = {"emotion", "drive", "goal", "fear"}
        assert set(DirectMoodBand.keys()) == expected

    def test_mood_band_has_valence_arousal(self):
        for source, band in DirectMoodBand.items():
            assert "valence" in band, f"{source} missing valence"
            assert "arousal" in band, f"{source} missing arousal"
            assert isinstance(band["valence"], (int, float))
            assert isinstance(band["arousal"], (int, float))

    def test_tracking_speed_bounds_valid(self):
        assert DirectTrackingSpeedMin > 0
        assert DirectTrackingSpeedMax > DirectTrackingSpeedMin

    def test_mood_delta_limit_positive(self):
        assert DirectMoodDeltaLimit > 0


# =============================================================================
# Layer 3: Call-Path Integration Tests
# =============================================================================

class TestLayer3ContextBuilders:
    """Test build_drive_context and build_mood_context from orchestrator_1tick_phases.

    Uses mock orchestrator objects with minimal required attributes."""

    def _make_mock_orch(self):
        """Create a minimal mock with the attributes needed by context builders."""
        class MockOrch:
            pass
        orch = MockOrch()

        # PsycheState (fear_level is a read-only property derived from fear_index)
        orch._psyche = PsycheState(
            emotions=EmotionVector(**_default_emotions()),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.0, arousal=0.3),
        )

        # Transient goal manager (mock)
        class MockTGMgr:
            class state:
                active_goal = None
        orch._transient_goal_mgr = MockTGMgr()

        # Persistent commitment (mock)
        class MockPC:
            class _state:
                items = []
        orch._persistent_commitment = MockPC()

        # Scoped goal system (mock)
        class MockSGS:
            has_active_scope = False
        orch._scoped_goal_sys = MockSGS()

        # Temporal cognition (mock)
        class MockTC:
            def get_snapshot(self):
                return {"activity_density": "normal"}
        orch._temporal_cognition = MockTC()

        # Behavioral diversity (mock)
        class MockBD:
            latest_record = None
        orch._behavioral_diversity_state = MockBD()

        # Contradiction processor (mock)
        class MockCP:
            class state:
                previous_contradictions = []
        orch._contradiction_processor = MockCP()

        # Memory emotion return (mock)
        class MockMER:
            def get_tracking_speed_modulation(self, **kwargs):
                return 0.0, 0.0
        orch._memory_emotion_return = MockMER()

        # Responsibility manager (mock)
        class MockRM:
            def get_influence(self, user_id):
                return None
        orch._responsibility_mgr = MockRM()

        # Exp multiplier (optional)
        orch._exp_drive_total_limit_multiplier = None

        return orch

    def test_build_drive_context_returns_correct_type(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = self._make_mock_orch()
        ctx = build_drive_context(orch)
        assert isinstance(ctx, DirectDriveContextInputs)

    def test_build_mood_context_returns_correct_type(self):
        from psyche.orchestrator_1tick_phases import build_mood_context
        orch = self._make_mock_orch()
        ctx = build_mood_context(orch)
        assert isinstance(ctx, DirectMoodContextInputs)

    def test_drive_context_goal_fields(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = self._make_mock_orch()
        # Set up goal presence
        orch._transient_goal_mgr.state.active_goal = "some_goal"
        orch._scoped_goal_sys.has_active_scope = True
        ctx = build_drive_context(orch)
        assert ctx.has_transient_goal is True
        assert ctx.has_scoped_goal is True

    def test_mood_context_fear_level(self):
        from psyche.orchestrator_1tick_phases import build_mood_context
        from psyche.pillars import FearIndex
        orch = self._make_mock_orch()
        # Set fear_index to get a non-zero fear_level (read-only property)
        orch._psyche = PsycheState(
            emotions=EmotionVector(**_default_emotions()),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.0, arousal=0.3),
            fear_index=FearIndex(
                identity_risk=0.3, attachment_risk=0.2,
                continuity_risk=0.1, projection_risk=0.1,
            ),
        )
        ctx = build_mood_context(orch)
        assert isinstance(ctx.fear_level, float)
        assert ctx.fear_level > 0.0

    def test_drive_context_temporal_label(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = self._make_mock_orch()
        ctx = build_drive_context(orch)
        assert ctx.time_density_label == "normal"

    def test_mood_context_valence_arousal_from_state(self):
        from psyche.orchestrator_1tick_phases import build_mood_context
        orch = self._make_mock_orch()
        orch._psyche = PsycheState(
            emotions=EmotionVector(**_default_emotions()),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.7, arousal=0.6),
        )
        ctx = build_mood_context(orch)
        assert ctx.current_valence == 0.7
        assert ctx.current_arousal == 0.6

    def test_drive_context_exp_multiplier_none(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = self._make_mock_orch()
        ctx = build_drive_context(orch)
        assert ctx.drive_total_limit_multiplier is None

    def test_drive_context_exp_multiplier_set(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = self._make_mock_orch()
        orch._exp_drive_total_limit_multiplier = 1.5
        ctx = build_drive_context(orch)
        assert ctx.drive_total_limit_multiplier == 1.5


class TestLayer3ReactCallPathEquivalence:
    """Test that react() produces equivalent output with and without
    explicitly provided context (context None -> internal auto-construction)."""

    def test_react_with_none_contexts_produces_valid_output(self):
        state = _zero_emotion_state()
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0,
                       drive_context=None, mood_context=None)
        assert isinstance(result, PsycheState)
        assert 0.0 <= result.emotions.joy <= 1.0
        assert -1.0 <= result.mood.valence <= 1.0
        assert 0.0 <= result.mood.arousal <= 1.0

    def test_react_with_explicit_contexts_produces_valid_output(self):
        state = _zero_emotion_state(emotions={"joy": 0.3})
        percept = _neutral_percept()
        drive_ctx = DirectDriveContextInputs(
            emotions=state.emotions.as_dict(),
            mood_valence=state.mood.valence,
            mood_arousal=state.mood.arousal,
            drives=state.drives.model_dump(),
            delta_time=1.0,
            percept_intent=percept.intent,
            percept_emotion=percept.emotion,
            percept_valence=percept.emotion_valence,
            fear_level=state.fear_level,
        )
        mood_ctx = DirectMoodContextInputs(
            emotions=state.emotions.as_dict(),
            drives=state.drives.as_dict(),
            current_valence=state.mood.valence,
            current_arousal=state.mood.arousal,
            fear_level=state.fear_level,
            delta_time=1.0,
        )
        result = react(percept, state, delta_time=1.0,
                       drive_context=drive_ctx, mood_context=mood_ctx)
        assert isinstance(result, PsycheState)

    def test_react_context_none_vs_explicit_equivalent(self):
        """Verify that None context (auto-constructed) and explicitly matching
        context produce the same output."""
        state = _zero_emotion_state(emotions={"joy": 0.3, "sorrow": 0.1})
        percept = _neutral_percept()

        # react with None contexts (auto-construction path)
        result_auto = react(percept, state, delta_time=1.0,
                           drive_context=None, mood_context=None)

        # Build contexts matching auto-construction logic
        emo = state.emotions.as_dict()
        drive_ctx = DirectDriveContextInputs(
            emotions=emo,
            mood_valence=state.mood.valence,
            mood_arousal=state.mood.arousal,
            drives=state.drives.model_dump(),
            delta_time=1.0,
            percept_intent=percept.intent,
            percept_emotion=percept.emotion,
            percept_valence=percept.emotion_valence,
            fear_level=state.fear_level,
        )

        # For mood context, we need to match the auto-constructed version
        # Note: react() updates emotions and drives before building mood context
        # So we need the post-emotion-update values
        # This test verifies that explicitly passing contexts gets overwritten
        # to current tick values inside react() anyway
        mood_ctx = DirectMoodContextInputs(
            delta_time=1.0,
        )
        result_explicit = react(percept, state, delta_time=1.0,
                               drive_context=drive_ctx, mood_context=mood_ctx)

        # Both should produce PsycheState with same structure
        assert isinstance(result_auto, PsycheState)
        assert isinstance(result_explicit, PsycheState)
        # Drive values should be equivalent
        for axis in ("social", "curiosity", "expression"):
            assert abs(getattr(result_auto.drives, axis) -
                       getattr(result_explicit.drives, axis)) < 1e-9

    def test_react_all_argument_patterns(self):
        """Test react with all combinations of responsibility/amplitude/contexts."""
        from psyche.responsibility import ResponsibilityInfluence

        state = _zero_emotion_state(emotions={"joy": 0.5})
        percept = _neutral_percept(emotion="happy")

        # Pattern 1: all defaults
        r1 = react(percept, state, delta_time=1.0)
        assert isinstance(r1, PsycheState)

        # Pattern 2: with responsibility
        resp = ResponsibilityInfluence(anxiety_baseline=0.3, fear_amplification=0.1)
        r2 = react(percept, state, delta_time=1.0,
                  responsibility_influence=resp)
        assert isinstance(r2, PsycheState)

        # Pattern 3: with amplitude
        r3 = react(percept, state, delta_time=1.0,
                  amplitude_modifier=1.5)
        assert isinstance(r3, PsycheState)

        # Pattern 4: with explicit contexts
        drive_ctx = DirectDriveContextInputs()
        mood_ctx = DirectMoodContextInputs()
        r4 = react(percept, state, delta_time=1.0,
                  drive_context=drive_ctx, mood_context=mood_ctx)
        assert isinstance(r4, PsycheState)

        # Pattern 5: everything together
        r5 = react(percept, state, delta_time=1.0,
                  responsibility_influence=resp,
                  amplitude_modifier=0.8,
                  drive_context=DirectDriveContextInputs(),
                  mood_context=DirectMoodContextInputs())
        assert isinstance(r5, PsycheState)


class TestLayer3ContextToSplitModuleFlow:
    """Verify that contexts built by orchestrator flow correctly to split modules."""

    def test_drive_context_accepted_by_compute_drive_changes(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = TestLayer3ContextBuilders()._make_mock_orch()
        ctx = build_drive_context(orch)
        # Should be accepted by the drive dynamics function without error
        result = direct_compute_drive_changes(ctx)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_mood_context_accepted_by_compute_autonomous_mood(self):
        from psyche.orchestrator_1tick_phases import build_mood_context
        orch = TestLayer3ContextBuilders()._make_mock_orch()
        ctx = build_mood_context(orch)
        # Should be accepted by the mood update function without error
        nv, na = direct_compute_mood(ctx)
        assert isinstance(nv, float)
        assert isinstance(na, float)

    def test_drive_context_with_diversity_data(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = TestLayer3ContextBuilders()._make_mock_orch()

        # Set up behavioral diversity mock
        class MockRecord:
            section_key_type_count_level = "level_6_10"
            policy_label_type_count_level = "level_1_5"
            candidate_size_dispersion_level = "moderate"
        orch._behavioral_diversity_state.latest_record = MockRecord()

        ctx = build_drive_context(orch)
        assert ctx.behavioral_diversity_stage_value == "level_6_10"
        assert ctx.result_diversity_section_key_level == "level_6_10"
        assert ctx.result_diversity_selection_label_level == "level_1_5"
        assert ctx.result_diversity_candidate_variance_level == "moderate"

        # Verify these values are accepted by the function
        result = direct_compute_drive_changes(ctx)
        assert isinstance(result, dict)

    def test_drive_context_with_contradiction_data(self):
        from psyche.orchestrator_1tick_phases import build_drive_context
        orch = TestLayer3ContextBuilders()._make_mock_orch()

        # Set up contradiction mock
        orch._contradiction_processor.state.previous_contradictions = [
            "c1", "c2", "c3"
        ]

        ctx = build_drive_context(orch)
        assert ctx.contradiction_count == 3

        result = direct_compute_drive_changes(ctx)
        assert isinstance(result, dict)

    def test_mood_context_with_emotion_return_modulation(self):
        from psyche.orchestrator_1tick_phases import build_mood_context
        orch = TestLayer3ContextBuilders()._make_mock_orch()

        # Set up emotion return with non-zero modulation
        class MockMER:
            def get_tracking_speed_modulation(self, **kwargs):
                return 0.03, 0.02
        orch._memory_emotion_return = MockMER()

        ctx = build_mood_context(orch)
        assert ctx.emotion_return_tracking_speed_modulation_valence == 0.03
        assert ctx.emotion_return_tracking_speed_modulation_arousal == 0.02

        nv, na = direct_compute_mood(ctx)
        assert isinstance(nv, float)
        assert isinstance(na, float)
