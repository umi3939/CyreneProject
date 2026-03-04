"""
Tests for C9-4: result diversity return pathway.

Verifies that behavioral_diversity_description stage values correctly
flow into drive bandwidth via _compute_result_diversity_return.
"""

import pytest

from psyche.reaction import (
    DriveContextInputs,
    _compute_result_diversity_return,
    _SECTION_BAND,
    _TOTAL_CHANGE_LIMIT,
    compute_state_dependent_drive_changes,
)


# =============================================================================
# Stage value → scale conversion tests
# =============================================================================

class TestTypeCountLevelScale:
    """Each TypeCountLevel stage value maps to the correct scale."""

    def test_level_0_maps_to_zero(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_0",
        )
        result = _compute_result_diversity_return(ctx)
        # level_0 → scale 0.0 → raw 0.0 → all axes zero
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_level_1_5_maps_to_025(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_1_5",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.25 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_level_6_10_maps_to_050(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_6_10",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.5 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_level_11_15_maps_to_075(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_11_15",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.75 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_level_16_plus_maps_to_100(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 1.0 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10


class TestSelectionLabelLevelScale:
    """Selection label level uses the same TypeCountLevel scale."""

    def test_selection_label_level_0(self):
        ctx = DriveContextInputs(
            result_diversity_selection_label_level="level_0",
        )
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_selection_label_level_16_plus(self):
        ctx = DriveContextInputs(
            result_diversity_selection_label_level="level_16_plus",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 1.0 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10


class TestCandidateVarianceLevelScale:
    """DispersionLevel stage values map to correct scale."""

    def test_empty_maps_to_zero(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="empty",
        )
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_uniform_maps_to_025(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="uniform",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.25 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_low_maps_to_050(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="low",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.5 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_moderate_maps_to_075(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="moderate",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.75 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_high_maps_to_100(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="high",
        )
        result = _compute_result_diversity_return(ctx)
        expected = 1.0 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10


# =============================================================================
# None input → zero contribution tests
# =============================================================================

class TestNoneInputNeutralization:
    """Input absence → zero contribution (safety valve 5)."""

    def test_all_none_returns_zero(self):
        ctx = DriveContextInputs()  # All diversity fields default to None
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_explicit_all_none(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level=None,
            result_diversity_selection_label_level=None,
            result_diversity_candidate_variance_level=None,
        )
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_one_none_two_present(self):
        """Two present inputs are averaged, None is excluded."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",  # 1.0
            result_diversity_selection_label_level="level_0",    # 0.0
            result_diversity_candidate_variance_level=None,
        )
        result = _compute_result_diversity_return(ctx)
        expected = ((1.0 + 0.0) / 2) * 0.03  # avg of 2 values
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_two_none_one_present(self):
        """Single present input is used alone."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level=None,
            result_diversity_selection_label_level=None,
            result_diversity_candidate_variance_level="high",  # 1.0
        )
        result = _compute_result_diversity_return(ctx)
        expected = 1.0 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10


# =============================================================================
# Equal averaging of 3 stage values
# =============================================================================

class TestEqualAveraging:
    """3 stage values are averaged equally (no weighting)."""

    def test_three_equal_values(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_6_10",     # 0.5
            result_diversity_selection_label_level="level_6_10", # 0.5
            result_diversity_candidate_variance_level="low",     # 0.5
        )
        result = _compute_result_diversity_return(ctx)
        expected = 0.5 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_three_different_values(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_0",        # 0.0
            result_diversity_selection_label_level="level_6_10", # 0.5
            result_diversity_candidate_variance_level="high",    # 1.0
        )
        result = _compute_result_diversity_return(ctx)
        expected = ((0.0 + 0.5 + 1.0) / 3) * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_all_maximum(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",  # 1.0
            result_diversity_selection_label_level="level_16_plus",  # 1.0
            result_diversity_candidate_variance_level="high",    # 1.0
        )
        result = _compute_result_diversity_return(ctx)
        expected = 1.0 * 0.03
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis] - expected) < 1e-10

    def test_all_minimum(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_0",   # 0.0
            result_diversity_selection_label_level="level_0",  # 0.0
            result_diversity_candidate_variance_level="empty",  # 0.0
        )
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_equal_distribution_across_axes(self):
        """All 3 axes receive the same amount (no axis-specific weighting)."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_11_15",  # 0.75
            result_diversity_selection_label_level="level_1_5",  # 0.25
            result_diversity_candidate_variance_level="moderate",  # 0.75
        )
        result = _compute_result_diversity_return(ctx)
        # All axes must be identical
        assert result["social"] == result["curiosity"]
        assert result["curiosity"] == result["expression"]


# =============================================================================
# Band limit compliance
# =============================================================================

class TestBandLimitCompliance:
    """Band limits are respected for all inputs."""

    def test_band_entry_exists(self):
        assert "result_diversity_return" in _SECTION_BAND

    def test_band_values_at_most_003(self):
        band = _SECTION_BAND["result_diversity_return"]
        for axis in ("social", "curiosity", "expression"):
            assert band[axis] <= 0.03

    def test_max_output_within_band(self):
        """Maximum possible output does not exceed band limits."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",  # 1.0
            result_diversity_selection_label_level="level_16_plus",  # 1.0
            result_diversity_candidate_variance_level="high",    # 1.0
        )
        result = _compute_result_diversity_return(ctx)
        band = _SECTION_BAND["result_diversity_return"]
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= band[axis] + 1e-10

    def test_band_at_most_drive_interaction(self):
        """Band is at most equal to drive_interaction band (design constraint)."""
        di_band = _SECTION_BAND["drive_interaction"]
        rd_band = _SECTION_BAND["result_diversity_return"]
        for axis in ("social", "curiosity", "expression"):
            assert rd_band[axis] <= di_band[axis]


# =============================================================================
# _TOTAL_CHANGE_LIMIT unchanged
# =============================================================================

class TestTotalChangeLimitUnchanged:
    """_TOTAL_CHANGE_LIMIT remains at 0.15."""

    def test_total_change_limit_value(self):
        assert _TOTAL_CHANGE_LIMIT == 0.15


# =============================================================================
# Pure function (no state)
# =============================================================================

class TestPureFunction:
    """_compute_result_diversity_return is a pure function with no state."""

    def test_same_input_same_output(self):
        """Calling with same inputs yields identical outputs."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_6_10",
            result_diversity_selection_label_level="level_1_5",
            result_diversity_candidate_variance_level="moderate",
        )
        result1 = _compute_result_diversity_return(ctx)
        result2 = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result1[axis] == result2[axis]

    def test_no_side_effects_on_ctx(self):
        """Input context is not modified."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_11_15",
            result_diversity_selection_label_level="level_6_10",
            result_diversity_candidate_variance_level="low",
        )
        original_section = ctx.result_diversity_section_key_level
        original_selection = ctx.result_diversity_selection_label_level
        original_variance = ctx.result_diversity_candidate_variance_level
        _compute_result_diversity_return(ctx)
        assert ctx.result_diversity_section_key_level == original_section
        assert ctx.result_diversity_selection_label_level == original_selection
        assert ctx.result_diversity_candidate_variance_level == original_variance

    def test_repeated_calls_no_accumulation(self):
        """Multiple calls do not accumulate or change results."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",
            result_diversity_selection_label_level="level_16_plus",
            result_diversity_candidate_variance_level="high",
        )
        results = [_compute_result_diversity_return(ctx) for _ in range(10)]
        for r in results:
            assert r == results[0]


# =============================================================================
# SD-5 input source separation
# =============================================================================

class TestSD5InputSourceSeparation:
    """Result diversity return uses different data sources from SD-5."""

    def test_inputs_are_stage_values_not_policy_data(self):
        """This section uses behavioral diversity stage values,
        not the expected_drive_change / selection snapshot that SD-5 uses."""
        ctx = DriveContextInputs(
            result_diversity_section_key_level="level_6_10",
            result_diversity_selection_label_level="level_1_5",
            result_diversity_candidate_variance_level="moderate",
        )
        result = _compute_result_diversity_return(ctx)
        # Verify the function produces output from stage values alone
        # (no reference to emotions, drives, policy data, etc.)
        assert any(result[a] != 0.0 for a in ("social", "curiosity", "expression"))

    def test_emotion_fields_do_not_affect_result(self):
        """Emotion fields on DriveContextInputs do not change this section's output."""
        ctx_no_emo = DriveContextInputs(
            result_diversity_section_key_level="level_6_10",
            result_diversity_selection_label_level="level_1_5",
            result_diversity_candidate_variance_level="low",
        )
        ctx_with_emo = DriveContextInputs(
            emotions={"joy": 1.0, "sorrow": 0.5, "anger": 0.3,
                       "surprise": 0.0, "fear": 0.8, "love": 0.2, "fun": 0.1},
            mood_valence=0.5,
            mood_arousal=0.8,
            result_diversity_section_key_level="level_6_10",
            result_diversity_selection_label_level="level_1_5",
            result_diversity_candidate_variance_level="low",
        )
        r1 = _compute_result_diversity_return(ctx_no_emo)
        r2 = _compute_result_diversity_return(ctx_with_emo)
        for axis in ("social", "curiosity", "expression"):
            assert r1[axis] == r2[axis]

    def test_drive_fields_do_not_affect_result(self):
        """Drive fields on DriveContextInputs do not change this section's output."""
        ctx_no_drv = DriveContextInputs(
            result_diversity_section_key_level="level_11_15",
            result_diversity_selection_label_level="level_6_10",
            result_diversity_candidate_variance_level="high",
        )
        ctx_with_drv = DriveContextInputs(
            drives={"social": 0.9, "curiosity": 0.1, "expression": 0.7},
            result_diversity_section_key_level="level_11_15",
            result_diversity_selection_label_level="level_6_10",
            result_diversity_candidate_variance_level="high",
        )
        r1 = _compute_result_diversity_return(ctx_no_drv)
        r2 = _compute_result_diversity_return(ctx_with_drv)
        for axis in ("social", "curiosity", "expression"):
            assert r1[axis] == r2[axis]


# =============================================================================
# Integration: included in compute_state_dependent_drive_changes
# =============================================================================

class TestIntegrationInCompute:
    """_compute_result_diversity_return is included in the composite function."""

    def test_section_included_in_sections_list(self):
        """The new section contributes to the total drive changes."""
        ctx_without = DriveContextInputs()
        ctx_with = DriveContextInputs(
            result_diversity_section_key_level="level_16_plus",
            result_diversity_selection_label_level="level_16_plus",
            result_diversity_candidate_variance_level="high",
        )
        total_without = compute_state_dependent_drive_changes(ctx_without)
        total_with = compute_state_dependent_drive_changes(ctx_with)
        # With maximum diversity input, at least one axis should differ
        diffs = [abs(total_with[a] - total_without[a]) for a in ("social", "curiosity", "expression")]
        assert max(diffs) > 0.0

    def test_total_still_clamped_by_limit(self):
        """Even with maximum input, total change is still within _TOTAL_CHANGE_LIMIT."""
        ctx = DriveContextInputs(
            emotions={"joy": 1.0, "sorrow": 1.0, "anger": 1.0,
                       "surprise": 1.0, "fear": 1.0, "love": 1.0, "fun": 1.0},
            mood_valence=1.0,
            mood_arousal=1.0,
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            delta_time=10.0,
            fear_level=1.0,
            behavioral_diversity_stage_value="level_0",
            contradiction_count=6,
            result_diversity_section_key_level="level_16_plus",
            result_diversity_selection_label_level="level_16_plus",
            result_diversity_candidate_variance_level="high",
        )
        total = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(total[axis]) <= _TOTAL_CHANGE_LIMIT + 1e-10


# =============================================================================
# Unknown stage value handling
# =============================================================================

class TestUnknownStageValue:
    """Unknown stage values default to scale 0.0."""

    def test_unknown_type_count_level(self):
        ctx = DriveContextInputs(
            result_diversity_section_key_level="unknown_level",
        )
        result = _compute_result_diversity_return(ctx)
        # unknown maps to 0.0 scale
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0

    def test_unknown_dispersion_level(self):
        ctx = DriveContextInputs(
            result_diversity_candidate_variance_level="unknown_dispersion",
        )
        result = _compute_result_diversity_return(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert result[axis] == 0.0
