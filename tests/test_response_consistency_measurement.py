"""
tests/test_response_consistency_measurement.py - 応答一貫性の経時的計測テスト

Measurement Mode 4: Probe-based Response Consistency (C13-2)
"""

from __future__ import annotations

import math

import pytest

from tools.long_term_sim import (
    INPUT_PATTERNS,
    SCENARIOS,
    STATE_VECTOR_DIM,
    _compute_binned_entropy,
    _compute_effective_dimensionality,
    _compute_lag1_autocorrelation,
    _compute_split_half_stationarity,
    _compute_transition_frequency,
    _compute_variance,
    _dimension_labels,
    _generate_probe_positions,
    run_probe_consistency_measurement,
)


# ── Probe Position Generation ───────────────────────────────────


class TestGenerateProbePositions:
    """_generate_probe_positions equidistant probe placement."""

    def test_basic_interval(self):
        """Generates positions at regular intervals."""
        positions = _generate_probe_positions(30, 10, 1)
        assert 9 in positions  # 0-based: interval 10 -> index 9
        assert 19 in positions
        assert 29 in positions
        assert len(positions) == 3

    def test_minimum_interval_enforced(self):
        """Interval below 3 is raised to 3."""
        positions = _generate_probe_positions(10, 1, 1)
        # effective_interval = 3, so positions at 2, 5, 8
        assert all(p >= 2 for p in positions)
        for i in range(len(positions) - 1):
            assert positions[i + 1] - positions[i] >= 3

    def test_empty_on_zero_turns(self):
        """Returns empty list for zero background turns."""
        assert _generate_probe_positions(0, 10, 1) == []

    def test_empty_on_zero_interval(self):
        """Returns empty list for zero interval."""
        assert _generate_probe_positions(10, 0, 1) == []

    def test_empty_on_zero_probe_types(self):
        """Returns empty list for zero probe types."""
        assert _generate_probe_positions(10, 5, 0) == []

    def test_ratio_safety_valve(self):
        """Total probes must not exceed half of background turns."""
        # 10 background turns, interval=3, 3 probe types
        # Without limit: positions at 2,5,8 -> 9 probes > 10//2=5
        positions = _generate_probe_positions(10, 3, 3)
        total_probes = len(positions) * 3
        assert total_probes <= 10 // 2

    def test_ratio_safety_valve_large(self):
        """Ratio check with many probe types."""
        positions = _generate_probe_positions(20, 3, 5)
        total_probes = len(positions) * 5
        assert total_probes <= 20 // 2

    def test_sorted_positions(self):
        """Positions are always sorted."""
        positions = _generate_probe_positions(50, 7, 2)
        assert positions == sorted(positions)

    def test_no_positions_beyond_turns(self):
        """No position is >= total_background_turns."""
        positions = _generate_probe_positions(25, 8, 1)
        for p in positions:
            assert p < 25

    def test_single_probe_type(self):
        """Single probe type: simpler ratio check."""
        positions = _generate_probe_positions(50, 10, 1)
        assert len(positions) <= 50 // 2

    def test_large_interval(self):
        """Interval larger than total turns gives empty list."""
        positions = _generate_probe_positions(5, 20, 1)
        assert positions == []


# ── Run Probe Consistency Measurement ─────────────────────────


class TestRunProbeConsistencyBasic:
    """Basic execution and output structure of run_probe_consistency_measurement."""

    def test_basic_execution(self):
        """Runs with smoke scenario and default probe."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        assert "metadata" in result
        assert result["metadata"]["mode"] == "probe_consistency"
        assert "turn_sequence" in result
        assert "probe_records" in result
        assert "probe_trajectory_features" in result

    def test_metadata_fields(self):
        """Metadata contains all required fields."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        meta = result["metadata"]
        assert meta["mode"] == "probe_consistency"
        assert meta["scenario"] == "smoke"
        assert meta["background_turns"] == len(SCENARIOS["smoke"])
        assert meta["probe_pattern_keys"] == ["neutral"]
        assert "probe_positions" in meta
        assert "total_probes" in meta
        assert "total_turns_including_probes" in meta
        assert "delta_time" in meta
        assert "user_id" in meta
        assert "started_at" in meta
        assert "finished_at" in meta
        assert meta["dimension_labels"] == _dimension_labels()

    def test_total_turns_accounting(self):
        """Total turns = background turns + total probe turns."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        meta = result["metadata"]
        # Each probe position inserts len(probe_pattern_keys) probe turns
        expected_probes = len(meta["probe_positions"]) * len(meta["probe_pattern_keys"])
        assert meta["total_probes"] == expected_probes
        assert meta["total_turns_including_probes"] == (
            meta["background_turns"] + expected_probes
        )

    def test_custom_sequence(self):
        """Works with custom_sequence instead of scenario_name."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        assert result["metadata"]["scenario"] == "custom"
        assert result["metadata"]["background_turns"] == 15


class TestRunProbeConsistencyValidation:
    """Input validation for run_probe_consistency_measurement."""

    def test_invalid_scenario(self):
        """Raises ValueError for unknown scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_probe_consistency_measurement(scenario_name="nonexistent_scenario")

    def test_invalid_probe_pattern(self):
        """Raises ValueError for invalid probe pattern key."""
        with pytest.raises(ValueError, match="Invalid probe pattern key"):
            run_probe_consistency_measurement(
                scenario_name="smoke",
                probe_pattern_keys=["nonexistent_pattern"],
            )

    def test_invalid_background_pattern(self):
        """Raises ValueError for invalid background pattern key."""
        with pytest.raises(ValueError, match="Invalid pattern key"):
            run_probe_consistency_measurement(
                custom_sequence=["invalid_key"],
                probe_pattern_keys=["neutral"],
            )

    def test_no_scenario_or_sequence(self):
        """Raises ValueError when neither scenario nor sequence provided."""
        with pytest.raises(ValueError, match="Either scenario_name or custom_sequence"):
            run_probe_consistency_measurement(
                probe_pattern_keys=["neutral"],
            )


class TestProbeRecordStructure:
    """Structure of individual probe records."""

    def test_probe_record_fields(self):
        """Each probe record has all required fields."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        if not result["probe_records"]:
            pytest.skip("No probes generated for this scenario/interval")
        record = result["probe_records"][0]
        assert "turn" in record
        assert record["type"] == "probe"
        assert "probe_pattern" in record
        assert "background_index" in record
        assert "background_turns_elapsed" in record
        assert "tick" in record
        assert "state_before" in record
        assert "state_after" in record
        assert "state_diff" in record
        assert "policy_label" in record
        assert "policy_score" in record

    def test_state_vectors_are_12dim(self):
        """State vectors (before, after, diff) have 12 dimensions."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        if not result["probe_records"]:
            pytest.skip("No probes generated")
        record = result["probe_records"][0]
        assert len(record["state_before"]) == STATE_VECTOR_DIM
        assert len(record["state_after"]) == STATE_VECTOR_DIM
        assert len(record["state_diff"]) == STATE_VECTOR_DIM

    def test_state_diff_is_after_minus_before(self):
        """state_diff[i] == state_after[i] - state_before[i]."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        if not result["probe_records"]:
            pytest.skip("No probes generated")
        record = result["probe_records"][0]
        for i in range(STATE_VECTOR_DIM):
            expected = round(record["state_after"][i] - record["state_before"][i], 6)
            assert abs(record["state_diff"][i] - expected) < 1e-5

    def test_probe_pattern_matches_key(self):
        """Each probe record has the correct probe_pattern key."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_pattern_keys=["neutral"],
            probe_interval=3,
        )
        for record in result["probe_records"]:
            assert record["probe_pattern"] == "neutral"

    def test_background_turns_elapsed_increases(self):
        """background_turns_elapsed is monotonically increasing."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        records = result["probe_records"]
        if len(records) < 2:
            pytest.skip("Too few probes")
        for i in range(len(records) - 1):
            assert records[i + 1]["background_turns_elapsed"] > records[i]["background_turns_elapsed"]


class TestMultipleProbeTypes:
    """Multiple probe types in a single run."""

    def test_two_probe_types(self):
        """Records probes for each probe type at each position."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative"] * 15,
            probe_pattern_keys=["neutral", "positive"],
            probe_interval=5,
        )
        # Each probe position should have one record per probe type
        positions = result["metadata"]["probe_positions"]
        expected_total = len(positions) * 2
        assert result["metadata"]["total_probes"] == expected_total

        # Check that both probe types appear
        probe_types = set(r["probe_pattern"] for r in result["probe_records"])
        assert "neutral" in probe_types
        assert "positive" in probe_types

    def test_max_probe_types_safety_valve(self):
        """Probe types are limited by max_probe_types."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral", "positive", "negative", "confused", "angry", "loving"],
            probe_interval=10,
            max_probe_types=3,
        )
        # Only first 3 probe types should be used
        probe_types = set(r["probe_pattern"] for r in result["probe_records"])
        assert len(probe_types) <= 3

    def test_trajectory_features_per_probe_type(self):
        """Trajectory features are computed separately for each probe type."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative"] * 15,
            probe_pattern_keys=["neutral", "positive"],
            probe_interval=5,
        )
        features = result["probe_trajectory_features"]
        assert "neutral" in features
        assert "positive" in features


class TestProbePositionControl:
    """Custom probe positions vs auto-generated positions."""

    def test_custom_positions(self):
        """Custom probe_positions overrides probe_interval."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_positions=[4, 9, 14, 19, 24],
        )
        # Check that probes were inserted at or near requested positions
        bg_indices = [r["background_index"] for r in result["probe_records"]]
        for pos in bg_indices:
            assert pos in [4, 9, 14, 19, 24]

    def test_custom_positions_out_of_range_filtered(self):
        """Positions >= total_background are silently filtered."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_positions=[2, 7, 100, 200],
        )
        bg_indices = [r["background_index"] for r in result["probe_records"]]
        assert 100 not in bg_indices
        assert 200 not in bg_indices

    def test_custom_positions_deduplicated(self):
        """Duplicate custom positions are deduplicated."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_positions=[2, 2, 7, 7],
        )
        # At most 2 distinct positions
        bg_indices = [r["background_index"] for r in result["probe_records"]]
        assert len(set(bg_indices)) <= 2


class TestProbeTrajectoryFeatures:
    """Trajectory statistical features computed on probe response differences."""

    def _get_features_with_sufficient_probes(self):
        """Helper: run a scenario that generates enough probes for analysis."""
        return run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral", "confused", "angry"] * 8,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )

    def test_per_dimension_features_present(self):
        """Per-dimension features are computed for each state vector dimension."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes for feature analysis")
        labels = _dimension_labels()
        for label in labels:
            assert label in features["per_dimension"]

    def test_variance_computed(self):
        """Variance is computed per dimension."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        for label in _dimension_labels():
            assert "variance" in features["per_dimension"][label]
            assert isinstance(features["per_dimension"][label]["variance"], float)

    def test_autocorrelation_computed(self):
        """Lag-1 autocorrelation is computed per dimension."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        for label in _dimension_labels():
            assert "lag1_autocorrelation" in features["per_dimension"][label]

    def test_entropy_computed_with_multiple_bins(self):
        """Entropy is computed with default bin parameters."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        label = _dimension_labels()[0]
        entropy = features["per_dimension"][label]["entropy"]
        assert "bins_5" in entropy
        assert "bins_10" in entropy
        assert "bins_20" in entropy

    def test_stationarity_computed(self):
        """Split-half stationarity is computed per dimension."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        label = _dimension_labels()[0]
        stat = features["per_dimension"][label]["stationarity"]
        assert "mean_diff" in stat
        assert "variance_diff" in stat

    def test_effective_dimensionality_present(self):
        """Effective dimensionality is computed for probe diff vectors."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        assert "effective_dimensionality" in features
        assert isinstance(features["effective_dimensionality"], float)

    def test_policy_transitions_present(self):
        """Policy label transitions are computed."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        assert "policy_transitions" in features
        assert "transition_count" in features["policy_transitions"]
        assert "unique_labels" in features["policy_transitions"]

    def test_change_from_first_probe_present(self):
        """Change from first probe is computed for each probe point."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        assert "change_from_first_probe" in features
        changes = features["change_from_first_probe"]
        assert len(changes) >= 2

    def test_change_from_first_probe_first_entry_is_zero(self):
        """First probe's change from itself should be zero."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        first = features["change_from_first_probe"][0]
        assert first["probe_index"] == 0
        assert first["euclidean_distance"] == 0.0

    def test_change_from_first_has_per_dimension(self):
        """Each change_from_first entry has per_dimension breakdown."""
        result = self._get_features_with_sufficient_probes()
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        for entry in features["change_from_first_probe"]:
            assert "per_dimension" in entry
            assert "euclidean_distance" in entry
            assert "background_turns_elapsed" in entry
            assert "probe_index" in entry

    def test_custom_entropy_bins(self):
        """Custom entropy bin parameters are used."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral", "confused", "angry"] * 8,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
            entropy_bin_params=[3, 7],
        )
        features = result["probe_trajectory_features"].get("neutral", {})
        if features.get("insufficient_data"):
            pytest.skip("Not enough probes")
        label = _dimension_labels()[0]
        entropy = features["per_dimension"][label]["entropy"]
        assert "bins_3" in entropy
        assert "bins_7" in entropy


class TestInsufficientProbeData:
    """Behavior when too few probes for trajectory features."""

    def test_single_probe_insufficient(self):
        """Single probe point marked as insufficient data."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"],
            probe_pattern_keys=["neutral"],
            probe_positions=[2],
        )
        features = result["probe_trajectory_features"].get("neutral", {})
        assert features.get("insufficient_data") is True or features.get("num_probes", 0) <= 1


class TestTurnSequenceIntegrity:
    """Integrity of the combined turn sequence (background + probes)."""

    def test_turn_numbers_are_sequential(self):
        """Turn numbers in turn_sequence are 1, 2, 3, ... N."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        turns = result["turn_sequence"]
        for i, t in enumerate(turns):
            assert t["turn"] == i + 1

    def test_background_and_probe_types_present(self):
        """Turn sequence contains both 'background' and 'probe' types."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        types = set(t["type"] for t in result["turn_sequence"])
        assert "background" in types
        if result["probe_records"]:
            assert "probe" in types

    def test_total_turns_match_sequence_length(self):
        """Total turns in metadata matches turn_sequence length."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 5,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        assert result["metadata"]["total_turns_including_probes"] == len(result["turn_sequence"])

    def test_background_turns_count_matches(self):
        """Number of background turns matches input sequence length."""
        seq = ["positive", "negative", "neutral"] * 5
        result = run_probe_consistency_measurement(
            custom_sequence=seq,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        bg_count = sum(1 for t in result["turn_sequence"] if t["type"] == "background")
        assert bg_count == len(seq)


class TestProbeExperienceAccumulation:
    """Probes are processed as normal turns (experience accumulates)."""

    def test_probes_change_state(self):
        """Probe input causes state change (state_diff is not all zeros)."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral", "confused", "angry"] * 8,
            probe_pattern_keys=["positive"],
            probe_interval=5,
        )
        if not result["probe_records"]:
            pytest.skip("No probes generated")
        # At least some probes should cause non-zero state changes
        any_nonzero = False
        for record in result["probe_records"]:
            if any(abs(d) > 1e-10 for d in record["state_diff"]):
                any_nonzero = True
                break
        assert any_nonzero, "All probe diffs are zero - probes should affect state"

    def test_successive_probes_differ(self):
        """State before successive probes of the same type differs (experience accumulated)."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral", "confused", "angry"] * 8,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        records = [r for r in result["probe_records"] if r["probe_pattern"] == "neutral"]
        if len(records) < 2:
            pytest.skip("Too few probes")
        # State vectors before successive probes should differ
        # (since background turns and probes accumulate experience)
        first_before = records[0]["state_before"]
        last_before = records[-1]["state_before"]
        dist = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(first_before, last_before)
        ))
        assert dist > 1e-6, "Pre-probe states should differ as experience accumulates"


class TestNoSaveLoadEvacuation:
    """Safety valve 6: No save/load state evacuation around probes."""

    def test_probe_state_is_continuous(self):
        """State after probe is the starting state for next background turn.

        This verifies that probes are not wrapped in save/load (which
        would undo their effect on state).
        """
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        # The turn sequence should show continuous ticks
        # (tick count should be monotonically increasing)
        ticks = [t["tick"] for t in result["turn_sequence"]]
        for i in range(len(ticks) - 1):
            assert ticks[i + 1] >= ticks[i], "Ticks should be monotonically increasing"


class TestProbeRatioSafetyValve:
    """Safety valve 4: Probes must not exceed half of background turns."""

    def test_probe_ratio_respected_with_interval(self):
        """Auto-generated probes respect the ratio limit."""
        seq = ["positive", "negative"] * 5  # 10 background turns
        result = run_probe_consistency_measurement(
            custom_sequence=seq,
            probe_pattern_keys=["neutral", "positive", "negative"],
            probe_interval=3,
        )
        total_probes = result["metadata"]["total_probes"]
        assert total_probes <= len(seq) // 2

    def test_probe_ratio_respected_with_positions(self):
        """Custom positions are trimmed to respect ratio limit."""
        seq = ["positive", "negative"] * 5  # 10 background turns
        result = run_probe_consistency_measurement(
            custom_sequence=seq,
            probe_pattern_keys=["neutral", "positive", "negative"],
            probe_positions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        total_probes = result["metadata"]["total_probes"]
        assert total_probes <= len(seq) // 2


class TestCLIProbeArgs:
    """CLI argument parsing for probe consistency mode."""

    def test_parser_has_probe_consistency_arg(self):
        """Parser recognizes --probe-consistency."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--probe-consistency", "smoke"])
        assert args.probe_consistency == "smoke"

    def test_parser_has_probe_patterns_arg(self):
        """Parser recognizes --probe-patterns."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--probe-consistency", "smoke",
            "--probe-patterns", "neutral", "positive",
        ])
        assert args.probe_patterns == ["neutral", "positive"]

    def test_parser_has_probe_interval_arg(self):
        """Parser recognizes --probe-interval."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--probe-consistency", "smoke",
            "--probe-interval", "15",
        ])
        assert args.probe_interval == 15

    def test_parser_has_max_probe_types_arg(self):
        """Parser recognizes --max-probe-types."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--probe-consistency", "smoke",
            "--max-probe-types", "3",
        ])
        assert args.max_probe_types == 3

    def test_parser_defaults(self):
        """Default values for probe args."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--probe-consistency", "smoke"])
        assert args.probe_patterns is None
        assert args.probe_interval == 10
        assert args.max_probe_types == 5


class TestDifferentScenarios:
    """Probe consistency measurement works with various scenarios."""

    def test_mixed_scenario(self):
        """Works with the 'mixed' scenario."""
        result = run_probe_consistency_measurement(
            scenario_name="mixed",
            probe_pattern_keys=["neutral"],
            probe_interval=10,
        )
        assert result["metadata"]["scenario"] == "mixed"
        assert len(result["probe_records"]) > 0

    def test_gradual_recovery_scenario(self):
        """Works with the 'gradual_recovery' scenario."""
        result = run_probe_consistency_measurement(
            scenario_name="gradual_recovery",
            probe_pattern_keys=["neutral"],
            probe_interval=10,
        )
        assert result["metadata"]["scenario"] == "gradual_recovery"
        assert len(result["probe_records"]) > 0

    def test_neutral_baseline_scenario(self):
        """Works with the 'neutral_baseline' scenario."""
        result = run_probe_consistency_measurement(
            scenario_name="neutral_baseline",
            probe_pattern_keys=["positive"],
            probe_interval=10,
        )
        assert result["metadata"]["scenario"] == "neutral_baseline"
        assert len(result["probe_records"]) > 0


class TestProbeDoesNotModifyPsycheLogic:
    """Verify probes use the normal processing interface."""

    def test_probe_has_policy_label(self):
        """Each probe record has a policy_label from normal selection."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        for record in result["probe_records"]:
            assert isinstance(record["policy_label"], str)

    def test_probe_has_policy_score(self):
        """Each probe record has a numeric policy_score."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        for record in result["probe_records"]:
            assert isinstance(record["policy_score"], float)


class TestDefaultProbePattern:
    """Default probe pattern is 'neutral' when not specified."""

    def test_default_probe_is_neutral(self):
        """When probe_pattern_keys is None, defaults to ['neutral']."""
        result = run_probe_consistency_measurement(
            scenario_name="smoke",
            probe_interval=3,
        )
        assert result["metadata"]["probe_pattern_keys"] == ["neutral"]
        for record in result["probe_records"]:
            assert record["probe_pattern"] == "neutral"


class TestNoEvaluationJudgment:
    """Output contains no evaluation judgments."""

    def test_no_good_bad_labels(self):
        """Result structure contains no evaluation labels."""
        result = run_probe_consistency_measurement(
            custom_sequence=["positive", "negative", "neutral"] * 10,
            probe_pattern_keys=["neutral"],
            probe_interval=5,
        )
        # Recursively check all string values for evaluation labels
        forbidden = ["good", "bad", "normal", "abnormal", "healthy", "unhealthy"]
        result_str = str(result)
        for word in forbidden:
            # Only check in keys, not in values that might be input text
            for key in _flatten_keys(result):
                assert word not in key.lower(), f"Evaluation label '{word}' found in key: {key}"


def _flatten_keys(d, prefix=""):
    """Helper to extract all keys from nested dict."""
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            keys.extend(_flatten_keys(v, full_key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            keys.extend(_flatten_keys(v, f"{prefix}[{i}]"))
    return keys
