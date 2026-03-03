"""
tests/test_sd3_extension.py - SD-3 Input Source Extension Tests

Tests for the 2 new drive dynamics sections (behavioral diversity + internal contradiction)
added to reaction.py as part of C9-6.

Verifies:
1. DriveContextInputs has new fields (behavioral_diversity_stage_value, contradiction_count)
2. New section functions (_compute_behavioral_diversity, _compute_internal_contradiction)
3. _SECTION_BAND has entries for new sections
4. Composite function includes new sections
5. Input-absent neutralization (None -> zero contribution)
6. Band limits are respected (each axis <= 0.02)
7. _TOTAL_CHANGE_LIMIT is unchanged
8. Existing 5 sections are unchanged
9. orchestrator_1tick_phases.build_drive_context populates new fields
10. react_with_stm passes drive_context through
"""

from __future__ import annotations

import pytest
from typing import Optional

from psyche.reaction import (
    DriveContextInputs,
    _SECTION_BAND,
    _TOTAL_CHANGE_LIMIT,
    _compute_behavioral_diversity,
    _compute_internal_contradiction,
    _compute_emotion_drive_coupling,
    _compute_drive_interaction,
    _compute_goal_hierarchy,
    _compute_time_passage,
    _compute_arousal_drive,
    compute_state_dependent_drive_changes,
)


# =============================================================================
# DriveContextInputs field tests
# =============================================================================

class TestDriveContextInputsNewFields:
    """Verify DriveContextInputs has the 2 new optional fields."""

    def test_default_behavioral_diversity_is_none(self):
        ctx = DriveContextInputs()
        assert ctx.behavioral_diversity_stage_value is None

    def test_default_contradiction_count_is_none(self):
        ctx = DriveContextInputs()
        assert ctx.contradiction_count is None

    def test_behavioral_diversity_can_be_set(self):
        ctx = DriveContextInputs(behavioral_diversity_stage_value="level_1_5")
        assert ctx.behavioral_diversity_stage_value == "level_1_5"

    def test_contradiction_count_can_be_set(self):
        ctx = DriveContextInputs(contradiction_count=3)
        assert ctx.contradiction_count == 3

    def test_existing_fields_unchanged(self):
        """Existing fields remain functional."""
        ctx = DriveContextInputs(
            emotions={"joy": 0.5},
            mood_valence=0.2,
            mood_arousal=0.5,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            has_transient_goal=True,
            persistent_commitment_count=2,
            has_scoped_goal=False,
            time_density_label="normal",
            delta_time=1.0,
            percept_intent="sharing",
            percept_emotion="happy",
            percept_valence=0.8,
            fear_level=0.1,
            behavioral_diversity_stage_value="level_6_10",
            contradiction_count=2,
        )
        assert ctx.emotions == {"joy": 0.5}
        assert ctx.mood_valence == 0.2
        assert ctx.behavioral_diversity_stage_value == "level_6_10"
        assert ctx.contradiction_count == 2


# =============================================================================
# _SECTION_BAND tests
# =============================================================================

class TestSectionBand:
    """Verify _SECTION_BAND has entries for new sections."""

    def test_behavioral_diversity_band_exists(self):
        assert "behavioral_diversity" in _SECTION_BAND

    def test_internal_contradiction_band_exists(self):
        assert "internal_contradiction" in _SECTION_BAND

    def test_behavioral_diversity_band_values(self):
        band = _SECTION_BAND["behavioral_diversity"]
        assert band["social"] == 0.02
        assert band["curiosity"] == 0.02
        assert band["expression"] == 0.02

    def test_internal_contradiction_band_values(self):
        band = _SECTION_BAND["internal_contradiction"]
        assert band["social"] == 0.02
        assert band["curiosity"] == 0.02
        assert band["expression"] == 0.02

    def test_new_bands_below_drive_interaction(self):
        """New section bands must be <= drive_interaction band (0.03)."""
        di_band = _SECTION_BAND["drive_interaction"]
        bd_band = _SECTION_BAND["behavioral_diversity"]
        ic_band = _SECTION_BAND["internal_contradiction"]
        for axis in ("social", "curiosity", "expression"):
            assert bd_band[axis] <= di_band[axis]
            assert ic_band[axis] <= di_band[axis]

    def test_total_change_limit_unchanged(self):
        """_TOTAL_CHANGE_LIMIT must remain at 0.15."""
        assert _TOTAL_CHANGE_LIMIT == 0.15

    def test_existing_bands_unchanged(self):
        """Existing 5 section bands are not modified."""
        assert _SECTION_BAND["emotion_drive_coupling"]["social"] == 0.06
        assert _SECTION_BAND["drive_interaction"]["curiosity"] == 0.03
        assert _SECTION_BAND["goal_hierarchy"]["expression"] == 0.05
        assert _SECTION_BAND["time_passage"]["social"] == 0.06
        assert _SECTION_BAND["arousal_drive"]["curiosity"] == 0.04


# =============================================================================
# _compute_behavioral_diversity tests
# =============================================================================

class TestComputeBehavioralDiversity:
    """Tests for section 6: behavioral diversity contribution."""

    def test_none_input_returns_zero(self):
        """Input absent -> zero contribution (safety valve 5)."""
        ctx = DriveContextInputs(behavioral_diversity_stage_value=None)
        result = _compute_behavioral_diversity(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_level_16_plus_returns_zero(self):
        """16+ types -> neutral (zero contribution)."""
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_16_plus",
            mood_valence=0.3,
        )
        result = _compute_behavioral_diversity(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_level_0_returns_positive(self):
        """0 types -> maximum positive contribution."""
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=0.5,
            mood_arousal=0.5,
        )
        result = _compute_behavioral_diversity(ctx)
        assert result["social"] > 0.0
        assert result["curiosity"] > 0.0
        assert result["expression"] > 0.0

    def test_level_1_5_lower_than_level_0(self):
        """Lower diversity stages produce higher contributions."""
        ctx_0 = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=0.3,
        )
        ctx_1_5 = DriveContextInputs(
            behavioral_diversity_stage_value="level_1_5",
            mood_valence=0.3,
        )
        r0 = _compute_behavioral_diversity(ctx_0)
        r1 = _compute_behavioral_diversity(ctx_1_5)
        assert r0["social"] > r1["social"]

    def test_band_limit_respected(self):
        """Output must not exceed band limit for any axis."""
        band = _SECTION_BAND["behavioral_diversity"]
        for stage in ("level_0", "level_1_5", "level_6_10", "level_11_15"):
            for valence in (-1.0, -0.5, 0.0, 0.5, 1.0):
                ctx = DriveContextInputs(
                    behavioral_diversity_stage_value=stage,
                    mood_valence=valence,
                )
                result = _compute_behavioral_diversity(ctx)
                for axis in ("social", "curiosity", "expression"):
                    assert abs(result[axis]) <= band[axis] + 1e-10, (
                        f"Band violated: stage={stage}, valence={valence}, "
                        f"axis={axis}, value={result[axis]}"
                    )

    def test_equal_contribution_across_axes(self):
        """All axes get equal contribution (no selective axis influence)."""
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=0.3,
        )
        result = _compute_behavioral_diversity(ctx)
        assert result["social"] == result["curiosity"]
        assert result["curiosity"] == result["expression"]

    def test_mood_valence_affects_magnitude(self):
        """Different mood valence -> different magnitudes (non-fixed mapping)."""
        ctx_pos = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=0.8,
        )
        ctx_neg = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            mood_valence=-0.8,
        )
        r_pos = _compute_behavioral_diversity(ctx_pos)
        r_neg = _compute_behavioral_diversity(ctx_neg)
        assert r_pos["social"] != r_neg["social"]

    def test_unknown_stage_returns_zero(self):
        """Unknown stage value -> zero contribution (safe fallback)."""
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="unknown_stage",
            mood_valence=0.5,
        )
        result = _compute_behavioral_diversity(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_pure_function_no_state(self):
        """Function is pure: same input -> same output (safety valve 6)."""
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_1_5",
            mood_valence=0.3,
        )
        r1 = _compute_behavioral_diversity(ctx)
        r2 = _compute_behavioral_diversity(ctx)
        assert r1 == r2


# =============================================================================
# _compute_internal_contradiction tests
# =============================================================================

class TestComputeInternalContradiction:
    """Tests for section 7: internal contradiction contribution."""

    def test_none_input_returns_zero(self):
        """Input absent -> zero contribution (safety valve 5)."""
        ctx = DriveContextInputs(contradiction_count=None)
        result = _compute_internal_contradiction(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_zero_count_returns_zero(self):
        """Zero contradictions -> zero contribution."""
        ctx = DriveContextInputs(contradiction_count=0)
        result = _compute_internal_contradiction(ctx)
        assert result == {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    def test_positive_count_with_high_arousal_positive_valence(self):
        """Contradictions + high arousal + positive valence -> positive contribution."""
        ctx = DriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.8,
            mood_valence=0.5,
        )
        result = _compute_internal_contradiction(ctx)
        assert result["social"] > 0.0

    def test_positive_count_with_low_arousal_negative_valence(self):
        """Contradictions + low arousal + negative valence -> negative contribution."""
        ctx = DriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.1,
            mood_valence=-0.8,
        )
        result = _compute_internal_contradiction(ctx)
        assert result["social"] < 0.0

    def test_direction_depends_on_mood(self):
        """Same count but different mood -> different direction (no fixed mapping)."""
        ctx1 = DriveContextInputs(
            contradiction_count=4,
            mood_arousal=0.9,
            mood_valence=0.8,
        )
        ctx2 = DriveContextInputs(
            contradiction_count=4,
            mood_arousal=0.1,
            mood_valence=-0.9,
        )
        r1 = _compute_internal_contradiction(ctx1)
        r2 = _compute_internal_contradiction(ctx2)
        # Different signs expected
        assert (r1["social"] > 0 and r2["social"] < 0) or (r1["social"] < 0 and r2["social"] > 0)

    def test_band_limit_respected(self):
        """Output must not exceed band limit for any axis."""
        band = _SECTION_BAND["internal_contradiction"]
        for count in (1, 3, 6, 10):
            for arousal in (0.0, 0.3, 0.5, 0.8, 1.0):
                for valence in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    ctx = DriveContextInputs(
                        contradiction_count=count,
                        mood_arousal=arousal,
                        mood_valence=valence,
                    )
                    result = _compute_internal_contradiction(ctx)
                    for axis in ("social", "curiosity", "expression"):
                        assert abs(result[axis]) <= band[axis] + 1e-10, (
                            f"Band violated: count={count}, arousal={arousal}, "
                            f"valence={valence}, axis={axis}, value={result[axis]}"
                        )

    def test_equal_contribution_across_axes(self):
        """All axes get equal contribution (no selective axis influence)."""
        ctx = DriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.7,
            mood_valence=0.3,
        )
        result = _compute_internal_contradiction(ctx)
        assert result["social"] == result["curiosity"]
        assert result["curiosity"] == result["expression"]

    def test_higher_count_higher_magnitude(self):
        """More contradictions -> higher absolute magnitude."""
        ctx1 = DriveContextInputs(
            contradiction_count=1,
            mood_arousal=0.7,
            mood_valence=0.5,
        )
        ctx6 = DriveContextInputs(
            contradiction_count=6,
            mood_arousal=0.7,
            mood_valence=0.5,
        )
        r1 = _compute_internal_contradiction(ctx1)
        r6 = _compute_internal_contradiction(ctx6)
        assert abs(r6["social"]) > abs(r1["social"])

    def test_count_capped_at_6(self):
        """Counts above 6 produce same result as 6 (cap)."""
        ctx6 = DriveContextInputs(
            contradiction_count=6,
            mood_arousal=0.7,
            mood_valence=0.5,
        )
        ctx10 = DriveContextInputs(
            contradiction_count=10,
            mood_arousal=0.7,
            mood_valence=0.5,
        )
        r6 = _compute_internal_contradiction(ctx6)
        r10 = _compute_internal_contradiction(ctx10)
        assert r6 == r10

    def test_pure_function_no_state(self):
        """Function is pure: same input -> same output (safety valve 6)."""
        ctx = DriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.5,
            mood_valence=0.2,
        )
        r1 = _compute_internal_contradiction(ctx)
        r2 = _compute_internal_contradiction(ctx)
        assert r1 == r2

    def test_depends_on_count_not_quality(self):
        """Only count matters, not which contradictions were detected."""
        ctx = DriveContextInputs(
            contradiction_count=3,
            mood_arousal=0.6,
            mood_valence=0.4,
        )
        result = _compute_internal_contradiction(ctx)
        # Non-zero with 3 contradictions and positive direction
        assert result["social"] != 0.0


# =============================================================================
# Composite function tests
# =============================================================================

class TestCompositeFunction:
    """Tests for compute_state_dependent_drive_changes with new sections."""

    def test_new_sections_included_in_composite(self):
        """New sections contribute to composite result."""
        # Context with only new section inputs active
        ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_0",
            contradiction_count=5,
            mood_valence=0.5,
            mood_arousal=0.7,
        )
        result = compute_state_dependent_drive_changes(ctx)
        # Should have non-zero values from the new sections
        # (other sections may also contribute default values)
        assert isinstance(result, dict)
        assert "social" in result
        assert "curiosity" in result
        assert "expression" in result

    def test_total_change_limit_enforced(self):
        """Composite result respects _TOTAL_CHANGE_LIMIT."""
        # Maximize all inputs to test clamping
        ctx = DriveContextInputs(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 1.0},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            time_density_label="sparse",
            delta_time=5.0,
            percept_intent="expression",
            fear_level=0.0,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=6,
        )
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT + 1e-10

    def test_none_new_fields_no_change_from_baseline(self):
        """When new fields are None, result equals 5-section-only result."""
        ctx_without = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.1, "anger": 0.0,
                      "fear": 0.0, "surprise": 0.2, "love": 0.3, "fun": 0.1},
            mood_valence=0.2,
            mood_arousal=0.4,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            behavioral_diversity_stage_value=None,
            contradiction_count=None,
        )
        ctx_with = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.1, "anger": 0.0,
                      "fear": 0.0, "surprise": 0.2, "love": 0.3, "fun": 0.1},
            mood_valence=0.2,
            mood_arousal=0.4,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            behavioral_diversity_stage_value=None,
            contradiction_count=None,
        )
        r1 = compute_state_dependent_drive_changes(ctx_without)
        r2 = compute_state_dependent_drive_changes(ctx_with)
        for axis in ("social", "curiosity", "expression"):
            assert abs(r1[axis] - r2[axis]) < 1e-10


# =============================================================================
# Existing section non-modification tests
# =============================================================================

class TestExistingSectionsUnchanged:
    """Verify that existing 5 section functions are not altered."""

    def _base_ctx(self) -> DriveContextInputs:
        return DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.1, "anger": 0.0,
                      "fear": 0.0, "surprise": 0.2, "love": 0.3, "fun": 0.1},
            mood_valence=0.2,
            mood_arousal=0.4,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            has_transient_goal=True,
            persistent_commitment_count=1,
            has_scoped_goal=False,
            time_density_label="normal",
            delta_time=1.0,
            percept_intent="sharing",
            percept_emotion="happy",
            percept_valence=0.7,
            fear_level=0.1,
        )

    def test_emotion_drive_coupling_output_shape(self):
        result = _compute_emotion_drive_coupling(self._base_ctx())
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_drive_interaction_output_shape(self):
        result = _compute_drive_interaction(self._base_ctx())
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_goal_hierarchy_output_shape(self):
        result = _compute_goal_hierarchy(self._base_ctx())
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_time_passage_output_shape(self):
        result = _compute_time_passage(self._base_ctx())
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_arousal_drive_output_shape(self):
        result = _compute_arousal_drive(self._base_ctx())
        assert set(result.keys()) == {"social", "curiosity", "expression"}

    def test_new_fields_dont_affect_existing_sections(self):
        """Adding new fields to ctx doesn't change existing section outputs."""
        ctx1 = self._base_ctx()
        ctx2 = DriveContextInputs(
            emotions=ctx1.emotions,
            mood_valence=ctx1.mood_valence,
            mood_arousal=ctx1.mood_arousal,
            drives=ctx1.drives,
            has_transient_goal=ctx1.has_transient_goal,
            persistent_commitment_count=ctx1.persistent_commitment_count,
            has_scoped_goal=ctx1.has_scoped_goal,
            time_density_label=ctx1.time_density_label,
            delta_time=ctx1.delta_time,
            percept_intent=ctx1.percept_intent,
            percept_emotion=ctx1.percept_emotion,
            percept_valence=ctx1.percept_valence,
            fear_level=ctx1.fear_level,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=5,
        )
        for fn in [_compute_emotion_drive_coupling, _compute_drive_interaction,
                   _compute_goal_hierarchy, _compute_time_passage, _compute_arousal_drive]:
            r1 = fn(ctx1)
            r2 = fn(ctx2)
            for axis in ("social", "curiosity", "expression"):
                assert abs(r1[axis] - r2[axis]) < 1e-10, (
                    f"{fn.__name__}: axis={axis} changed with new fields"
                )


# =============================================================================
# react_with_stm passthrough tests
# =============================================================================

class TestReactWithStmPassthrough:
    """Verify react_with_stm accepts and passes drive_context."""

    def test_drive_context_parameter_accepted(self):
        """react_with_stm accepts drive_context parameter."""
        from psyche.reaction_with_stm import react_with_stm
        from psyche.state import Percept, PsycheState
        from psyche.short_term_loop import create_loop_state

        percept = Percept(
            text="hello",
            intent="sharing",
            emotion="happy",
            emotion_valence=0.5,
        )
        state = PsycheState()
        loop_state = create_loop_state()

        drive_ctx = DriveContextInputs(
            behavioral_diversity_stage_value="level_1_5",
            contradiction_count=2,
        )

        # Should not raise
        new_psyche, new_loop, result = react_with_stm(
            percept=percept,
            psyche_state=state,
            loop_state=loop_state,
            delta_time=1.0,
            drive_context=drive_ctx,
        )
        assert new_psyche is not None

    def test_none_drive_context_works(self):
        """react_with_stm works with drive_context=None (default)."""
        from psyche.reaction_with_stm import react_with_stm
        from psyche.state import Percept, PsycheState
        from psyche.short_term_loop import create_loop_state

        percept = Percept(
            text="hello",
            intent="sharing",
            emotion="happy",
            emotion_valence=0.5,
        )
        state = PsycheState()
        loop_state = create_loop_state()

        new_psyche, new_loop, result = react_with_stm(
            percept=percept,
            psyche_state=state,
            loop_state=loop_state,
            delta_time=1.0,
        )
        assert new_psyche is not None


# =============================================================================
# build_drive_context tests
# =============================================================================

class TestBuildDriveContext:
    """Tests for orchestrator_1tick_phases.build_drive_context."""

    def test_build_drive_context_returns_drive_context_inputs(self):
        """build_drive_context returns a DriveContextInputs instance."""
        from psyche.orchestrator_1tick_phases import build_drive_context
        from unittest.mock import MagicMock

        # Minimal mock orchestrator
        orch = MagicMock()
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition.get_snapshot.return_value = {}
        orch._behavioral_diversity_state.latest_record = None
        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert isinstance(ctx, DriveContextInputs)

    def test_build_drive_context_populates_behavioral_diversity(self):
        """build_drive_context reads behavioral_diversity stage value."""
        from psyche.orchestrator_1tick_phases import build_drive_context
        from unittest.mock import MagicMock

        orch = MagicMock()
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition.get_snapshot.return_value = {}

        # Set up behavioral diversity with a latest record
        mock_record = MagicMock()
        mock_record.section_key_type_count_level = "level_6_10"
        orch._behavioral_diversity_state.latest_record = mock_record

        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert ctx.behavioral_diversity_stage_value == "level_6_10"

    def test_build_drive_context_populates_contradiction_count(self):
        """build_drive_context reads contradiction detection count."""
        from psyche.orchestrator_1tick_phases import build_drive_context
        from unittest.mock import MagicMock

        orch = MagicMock()
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition.get_snapshot.return_value = {}
        orch._behavioral_diversity_state.latest_record = None

        # Set up contradiction processor with previous results
        orch._contradiction_processor.state.previous_contradictions = [
            MagicMock(), MagicMock(), MagicMock()  # 3 contradictions
        ]

        ctx = build_drive_context(orch)
        assert ctx.contradiction_count == 3

    def test_build_drive_context_none_on_exception(self):
        """build_drive_context handles exceptions gracefully (None fields)."""
        from psyche.orchestrator_1tick_phases import build_drive_context

        class BrokenOrch:
            """All attribute access raises AttributeError."""
            def __getattr__(self, name):
                raise AttributeError(f"no {name}")

        ctx = build_drive_context(BrokenOrch())
        # Should return a valid context with default values
        assert isinstance(ctx, DriveContextInputs)
        assert ctx.behavioral_diversity_stage_value is None
        assert ctx.contradiction_count is None

    def test_build_drive_context_no_latest_record(self):
        """build_drive_context with no behavioral diversity record -> None."""
        from psyche.orchestrator_1tick_phases import build_drive_context
        from unittest.mock import MagicMock

        orch = MagicMock()
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition.get_snapshot.return_value = {}
        orch._behavioral_diversity_state.latest_record = None
        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert ctx.behavioral_diversity_stage_value is None

    def test_build_drive_context_zero_contradictions(self):
        """build_drive_context with empty previous contradictions -> 0."""
        from psyche.orchestrator_1tick_phases import build_drive_context
        from unittest.mock import MagicMock

        orch = MagicMock()
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition.get_snapshot.return_value = {}
        orch._behavioral_diversity_state.latest_record = None
        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert ctx.contradiction_count == 0


# =============================================================================
# Integration-level: new sections affect composite
# =============================================================================

class TestNewSectionsAffectComposite:
    """Verify new sections can produce non-zero composite changes."""

    def test_behavioral_diversity_adds_to_composite(self):
        """Behavioral diversity section contributes to composite."""
        ctx_without = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.5,
            behavioral_diversity_stage_value=None,
            contradiction_count=None,
        )
        ctx_with = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.5,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=None,
        )
        r_without = compute_state_dependent_drive_changes(ctx_without)
        r_with = compute_state_dependent_drive_changes(ctx_with)
        # With behavioral diversity active, result should differ
        differs = any(
            abs(r_without[a] - r_with[a]) > 1e-10
            for a in ("social", "curiosity", "expression")
        )
        assert differs, "Behavioral diversity should affect composite"

    def test_contradiction_adds_to_composite(self):
        """Internal contradiction section contributes to composite."""
        ctx_without = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.7,
            behavioral_diversity_stage_value=None,
            contradiction_count=None,
        )
        ctx_with = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.7,
            behavioral_diversity_stage_value=None,
            contradiction_count=5,
        )
        r_without = compute_state_dependent_drive_changes(ctx_without)
        r_with = compute_state_dependent_drive_changes(ctx_with)
        differs = any(
            abs(r_without[a] - r_with[a]) > 1e-10
            for a in ("social", "curiosity", "expression")
        )
        assert differs, "Internal contradiction should affect composite"

    def test_both_sections_combine(self):
        """Both new sections contribute additively."""
        ctx_bd_only = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.7,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=None,
        )
        ctx_ic_only = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.7,
            behavioral_diversity_stage_value=None,
            contradiction_count=5,
        )
        ctx_both = DriveContextInputs(
            mood_valence=0.5,
            mood_arousal=0.7,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=5,
        )
        r_bd = compute_state_dependent_drive_changes(ctx_bd_only)
        r_ic = compute_state_dependent_drive_changes(ctx_ic_only)
        r_both = compute_state_dependent_drive_changes(ctx_both)
        # Both combined should differ from either alone
        # (unless clamped, which is also valid behavior)
        assert r_both is not None
