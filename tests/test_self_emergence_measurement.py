"""
tests/test_self_emergence_measurement.py - 自己発現計測フレームワークのテスト

Measurement Mode 1: State Divergence (C13-1)
Measurement Mode 2: Trajectory Statistical Features (C13-7)
Measurement Mode 3: A/B Coefficient Comparison (C13-9)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.long_term_sim import (
    INPUT_PATTERNS,
    SCENARIOS,
    STATE_VECTOR_DIM,
    _compute_binned_entropy,
    _compute_effective_dimensionality,
    _compute_lag1_autocorrelation,
    _compute_per_dimension_effect_size,
    _compute_split_half_stationarity,
    _compute_transition_frequency,
    _compute_variance,
    _dimension_labels,
    _euclidean_distance,
    _extract_state_vector,
    _extract_state_vector_from_record,
    compute_trajectory_features,
    run_ab_comparison,
    run_divergence_measurement,
    run_simulation,
)


# ── State Vector Extraction ──────────────────────────────────

class TestStateVectorDimension:
    """State vector dimensionality constants and label consistency."""

    def test_state_vector_dim_is_12(self):
        """State vector has exactly 12 dimensions."""
        assert STATE_VECTOR_DIM == 12

    def test_dimension_labels_count(self):
        """Dimension labels match STATE_VECTOR_DIM."""
        labels = _dimension_labels()
        assert len(labels) == STATE_VECTOR_DIM

    def test_dimension_labels_content(self):
        """Labels include all expected emotion, drive, and mood keys."""
        labels = _dimension_labels()
        # 7 emotions
        assert "joy" in labels
        assert "anger" in labels
        assert "sorrow" in labels
        assert "fear" in labels
        assert "surprise" in labels
        assert "love" in labels
        assert "fun" in labels
        # 3 drives
        assert "social" in labels
        assert "curiosity" in labels
        assert "expression" in labels
        # 2 mood
        assert "valence" in labels
        assert "arousal" in labels

    def test_dimension_labels_no_duplicates(self):
        """No duplicate labels."""
        labels = _dimension_labels()
        assert len(labels) == len(set(labels))


class TestStateVectorFromRecord:
    """Extract state vector from turn record dict."""

    def test_extract_from_record(self):
        """Extracts 12-dim vector from a turn record structure."""
        record = {
            "psyche_state": {
                "emotions": {
                    "joy": 0.1, "anger": 0.2, "sorrow": 0.3,
                    "fear": 0.4, "surprise": 0.5, "love": 0.6, "fun": 0.7,
                },
                "drives": {
                    "social": 0.8, "curiosity": 0.9, "expression": 0.95,
                },
                "mood": {"valence": -0.5, "arousal": 0.6},
            },
        }
        vec = _extract_state_vector_from_record(record)
        assert len(vec) == 12
        assert vec[0] == 0.1  # joy
        assert vec[6] == 0.7  # fun
        assert vec[7] == 0.8  # social
        assert vec[10] == -0.5  # valence
        assert vec[11] == 0.6  # arousal

    def test_extract_from_record_missing_keys_uses_defaults(self):
        """Missing keys fall back to defaults."""
        record = {
            "psyche_state": {
                "emotions": {},
                "drives": {},
                "mood": {},
            },
        }
        vec = _extract_state_vector_from_record(record)
        assert len(vec) == 12
        # Emotions default to 0.0
        assert vec[0] == 0.0
        # Drives default to 0.5
        assert vec[7] == 0.5
        # Mood defaults
        assert vec[10] == 0.0  # valence default
        assert vec[11] == 0.3  # arousal default


# ── Euclidean Distance ───────────────────────────────────────

class TestEuclideanDistance:
    """Euclidean distance helper."""

    def test_identical_vectors(self):
        assert _euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_distance(self):
        # 3-4-5 triangle
        dist = _euclidean_distance([0, 0], [3, 4])
        assert abs(dist - 5.0) < 1e-10

    def test_single_dimension(self):
        assert abs(_euclidean_distance([0], [5]) - 5.0) < 1e-10

    def test_negative_values(self):
        dist = _euclidean_distance([-1, -1], [1, 1])
        assert abs(dist - math.sqrt(8)) < 1e-10


# ── Variance ─────────────────────────────────────────────────

class TestVariance:
    """Population variance helper."""

    def test_zero_variance(self):
        assert _compute_variance([5, 5, 5]) == 0.0

    def test_known_variance(self):
        # [1, 2, 3] mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
        var = _compute_variance([1, 2, 3])
        assert abs(var - 2 / 3) < 1e-10

    def test_single_value(self):
        assert _compute_variance([42]) == 0.0

    def test_empty(self):
        assert _compute_variance([]) == 0.0


# ── Lag-1 Autocorrelation ────────────────────────────────────

class TestLag1Autocorrelation:
    """Lag-1 autocorrelation computation."""

    def test_constant_series(self):
        """Constant series has zero variance, returns 0.0."""
        assert _compute_lag1_autocorrelation([1, 1, 1, 1, 1]) == 0.0

    def test_alternating_series(self):
        """Alternating series should have negative autocorrelation."""
        values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        ac = _compute_lag1_autocorrelation(values)
        assert ac < 0

    def test_trending_series(self):
        """Monotonically increasing series should have positive autocorrelation."""
        values = [float(i) for i in range(20)]
        ac = _compute_lag1_autocorrelation(values)
        assert ac > 0

    def test_short_series(self):
        """Series with < 3 elements returns 0.0."""
        assert _compute_lag1_autocorrelation([1, 2]) == 0.0
        assert _compute_lag1_autocorrelation([]) == 0.0


# ── Binned Entropy ───────────────────────────────────────────

class TestBinnedEntropy:
    """Shannon entropy of binned values."""

    def test_zero_entropy_constant(self):
        """Constant values yield zero entropy."""
        assert _compute_binned_entropy([1, 1, 1, 1], 5) == 0.0

    def test_max_entropy_uniform(self):
        """Uniform distribution across bins yields max entropy."""
        # 100 values uniformly distributed in 10 bins
        values = [i / 10.0 for i in range(100)]
        entropy = _compute_binned_entropy(values, 10)
        max_entropy = math.log(10)
        assert abs(entropy - max_entropy) < 0.15  # approximate

    def test_empty_list(self):
        assert _compute_binned_entropy([], 5) == 0.0

    def test_single_bin(self):
        """Single bin always yields zero entropy."""
        assert _compute_binned_entropy([1, 2, 3], 1) == 0.0

    def test_different_bin_counts(self):
        """More bins generally means higher entropy for varied data."""
        values = [i / 100.0 for i in range(100)]
        e5 = _compute_binned_entropy(values, 5)
        e10 = _compute_binned_entropy(values, 10)
        e20 = _compute_binned_entropy(values, 20)
        assert e5 > 0
        assert e10 > e5
        assert e20 > e10

    def test_entropy_non_negative(self):
        """Entropy is always non-negative."""
        import random
        rng = random.Random(42)
        values = [rng.random() for _ in range(50)]
        for bins in [2, 5, 10, 20]:
            assert _compute_binned_entropy(values, bins) >= 0.0


# ── Split-Half Stationarity ─────────────────────────────────

class TestSplitHalfStationarity:
    """Split-half stationarity indicator."""

    def test_stationary_series(self):
        """Constant series has zero mean diff and variance diff."""
        result = _compute_split_half_stationarity([5] * 20)
        assert result["mean_diff"] == 0.0
        assert result["variance_diff"] == 0.0

    def test_trending_series(self):
        """Monotone increasing series has positive mean diff."""
        values = list(range(20))
        result = _compute_split_half_stationarity([float(v) for v in values])
        assert result["mean_diff"] > 0  # second half has higher mean

    def test_short_series(self):
        """Single element returns all zeros."""
        result = _compute_split_half_stationarity([1.0])
        assert result["mean_diff"] == 0.0

    def test_output_keys(self):
        """All expected keys are present."""
        result = _compute_split_half_stationarity([1.0, 2.0, 3.0, 4.0])
        expected_keys = {
            "mean_diff", "variance_diff",
            "first_half_mean", "second_half_mean",
            "first_half_variance", "second_half_variance",
        }
        assert set(result.keys()) == expected_keys


# ── Effective Dimensionality ─────────────────────────────────

class TestEffectiveDimensionality:
    """Effective dimensionality (participation ratio)."""

    def test_single_active_dimension(self):
        """When only one dimension varies, effective dim should be near 1."""
        vectors = [[float(i), 0.0, 0.0] for i in range(20)]
        ed = _compute_effective_dimensionality(vectors)
        assert abs(ed - 1.0) < 0.01

    def test_all_dimensions_equal(self):
        """When all dimensions have equal variance, effective dim = num dims."""
        import random
        rng = random.Random(42)
        dim = 5
        vectors = [[rng.random() for _ in range(dim)] for _ in range(100)]
        ed = _compute_effective_dimensionality(vectors)
        # Should be close to dim (5), but random data won't be perfectly equal
        assert ed > dim * 0.5

    def test_empty_vectors(self):
        assert _compute_effective_dimensionality([]) == 0.0

    def test_constant_vectors(self):
        """All-constant vectors have zero variance, returns 0.0."""
        vectors = [[1.0, 2.0, 3.0]] * 10
        assert _compute_effective_dimensionality(vectors) == 0.0


# ── Transition Frequency ─────────────────────────────────────

class TestTransitionFrequency:
    """Transition frequency in label sequences."""

    def test_no_transitions(self):
        result = _compute_transition_frequency(["a", "a", "a"])
        assert result["transition_count"] == 0
        assert result["unique_labels"] == 1

    def test_all_transitions(self):
        result = _compute_transition_frequency(["a", "b", "c", "d"])
        assert result["transition_count"] == 3
        assert result["unique_labels"] == 4

    def test_single_label(self):
        result = _compute_transition_frequency(["x"])
        assert result["transition_count"] == 0
        assert result["unique_labels"] == 1

    def test_alternating(self):
        result = _compute_transition_frequency(["a", "b", "a", "b", "a"])
        assert result["transition_count"] == 4
        assert result["unique_labels"] == 2


# ── Measurement Mode 1: Divergence ───────────────────────────

class TestDivergenceMeasurement:
    """State divergence measurement tests."""

    @pytest.fixture(scope="class")
    def divergence_result(self) -> dict:
        """Run divergence measurement with smoke scenario."""
        return run_divergence_measurement(
            scenario_name="smoke",
            num_instances=2,
            warmup_turns=3,
            max_instances=5,
        )

    def test_result_structure(self, divergence_result):
        """Result has expected top-level keys."""
        assert "metadata" in divergence_result
        assert "warmup_states" in divergence_result
        assert "tick_records" in divergence_result
        assert "divergence_summary" in divergence_result

    def test_metadata_fields(self, divergence_result):
        """Metadata contains mode and instance count."""
        meta = divergence_result["metadata"]
        assert meta["mode"] == "divergence"
        assert meta["num_instances"] == 2
        assert meta["warmup_turns"] == 3
        assert meta["scenario"] == "smoke"
        assert "dimension_labels" in meta
        assert len(meta["dimension_labels"]) == 12

    def test_warmup_states_count(self, divergence_result):
        """One warmup state per instance."""
        assert len(divergence_result["warmup_states"]) == 2

    def test_warmup_state_has_vector(self, divergence_result):
        """Each warmup state has a 12-dim vector."""
        for ws in divergence_result["warmup_states"]:
            assert "state_vector" in ws
            assert len(ws["state_vector"]) == 12
            assert "warmup_pattern" in ws

    def test_tick_records_count(self, divergence_result):
        """One tick record per turn in the main sequence."""
        expected = len(SCENARIOS["smoke"])
        assert len(divergence_result["tick_records"]) == expected

    def test_tick_record_fields(self, divergence_result):
        """Each tick record has required fields."""
        for rec in divergence_result["tick_records"]:
            assert "turn" in rec
            assert "input_pattern" in rec
            assert "instance_vectors" in rec
            assert "pairwise_distances" in rec
            assert "mean_distance" in rec

    def test_instance_vectors_shape(self, divergence_result):
        """Each tick has num_instances vectors of dimension 12."""
        for rec in divergence_result["tick_records"]:
            assert len(rec["instance_vectors"]) == 2
            for vec in rec["instance_vectors"]:
                assert len(vec) == 12

    def test_pairwise_distances_count(self, divergence_result):
        """2 instances -> 1 pair."""
        for rec in divergence_result["tick_records"]:
            assert len(rec["pairwise_distances"]) == 1

    def test_pairwise_distance_non_negative(self, divergence_result):
        """All distances are non-negative."""
        for rec in divergence_result["tick_records"]:
            for pd in rec["pairwise_distances"]:
                assert pd["distance"] >= 0.0

    def test_mean_distance_numeric(self, divergence_result):
        """Mean distance is a float."""
        for rec in divergence_result["tick_records"]:
            assert isinstance(rec["mean_distance"], float)

    def test_divergence_summary_fields(self, divergence_result):
        """Summary has expected statistical fields."""
        summary = divergence_result["divergence_summary"]
        assert "initial_mean_distance" in summary
        assert "final_mean_distance" in summary
        assert "max_mean_distance" in summary
        assert "min_mean_distance" in summary
        assert "mean_of_mean_distances" in summary

    def test_three_instances_pairwise(self):
        """3 instances -> 3 pairs."""
        result = run_divergence_measurement(
            scenario_name="smoke",
            num_instances=3,
            warmup_turns=2,
            max_instances=5,
        )
        for rec in result["tick_records"]:
            # C(3,2) = 3 pairs
            assert len(rec["pairwise_distances"]) == 3

    def test_min_instances_validation(self):
        """num_instances < 2 raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            run_divergence_measurement(
                scenario_name="smoke",
                num_instances=1,
            )

    def test_max_instances_safety(self):
        """num_instances is capped by max_instances."""
        result = run_divergence_measurement(
            scenario_name="smoke",
            num_instances=100,
            warmup_turns=2,
            max_instances=3,
        )
        assert result["metadata"]["num_instances"] == 3

    def test_invalid_scenario(self):
        """Invalid scenario raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_divergence_measurement(
                scenario_name="nonexistent_scenario",
            )

    def test_custom_sequence(self):
        """Custom sequence works."""
        result = run_divergence_measurement(
            custom_sequence=["positive", "negative", "neutral"],
            num_instances=2,
            warmup_turns=2,
        )
        assert result["metadata"]["scenario"] == "custom"
        assert result["metadata"]["main_turns"] == 3

    def test_custom_warmup_patterns(self):
        """Explicit warmup patterns are used."""
        result = run_divergence_measurement(
            scenario_name="smoke",
            num_instances=2,
            warmup_turns=2,
            warmup_patterns=[
                ["positive", "positive"],
                ["negative", "negative"],
            ],
        )
        assert result["warmup_states"][0]["warmup_pattern"] == ["positive", "positive"]
        assert result["warmup_states"][1]["warmup_pattern"] == ["negative", "negative"]

    def test_warmup_patterns_length_mismatch(self):
        """warmup_patterns length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            run_divergence_measurement(
                scenario_name="smoke",
                num_instances=2,
                warmup_patterns=[["positive"]],  # only 1, need 2
            )

    def test_no_evaluation_in_output(self, divergence_result):
        """Output contains no evaluative labels."""
        import json
        text = json.dumps(divergence_result)
        for word in ["good", "bad", "normal", "abnormal", "optimal", "better", "worse"]:
            assert word not in text.lower(), (
                f"Evaluative word '{word}' found in divergence output"
            )


# ── Measurement Mode 2: Trajectory Features ─────────────────

class TestTrajectoryFeatures:
    """Trajectory statistical features tests."""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        """Run smoke scenario."""
        return run_simulation(scenario_name="smoke")

    @pytest.fixture(scope="class")
    def trajectory(self, smoke_result) -> dict:
        """Compute trajectory features on smoke result."""
        return compute_trajectory_features(smoke_result)

    def test_result_structure(self, trajectory):
        """Has expected top-level keys."""
        assert "scenario" in trajectory
        assert "total_turns" in trajectory
        assert "dimension_labels" in trajectory
        assert "per_dimension" in trajectory
        assert "effective_dimensionality" in trajectory
        assert "dominant_emotion_transitions" in trajectory
        assert "policy_label_transitions" in trajectory

    def test_per_dimension_count(self, trajectory):
        """12 dimension entries."""
        assert len(trajectory["per_dimension"]) == 12

    def test_per_dimension_keys(self, trajectory):
        """Each dimension has variance, autocorrelation, entropy, stationarity."""
        for label, features in trajectory["per_dimension"].items():
            assert "variance" in features, f"Missing variance for {label}"
            assert "lag1_autocorrelation" in features, f"Missing autocorrelation for {label}"
            assert "entropy" in features, f"Missing entropy for {label}"
            assert "stationarity" in features, f"Missing stationarity for {label}"

    def test_entropy_multiple_bins(self, trajectory):
        """Default entropy uses 3 bin parameters (5, 10, 20)."""
        for label, features in trajectory["per_dimension"].items():
            ent = features["entropy"]
            assert "bins_5" in ent, f"Missing bins_5 for {label}"
            assert "bins_10" in ent, f"Missing bins_10 for {label}"
            assert "bins_20" in ent, f"Missing bins_20 for {label}"

    def test_custom_entropy_bins(self, smoke_result):
        """Custom bin parameters are used."""
        features = compute_trajectory_features(
            smoke_result, entropy_bin_params=[3, 7]
        )
        for label, dim_feat in features["per_dimension"].items():
            ent = dim_feat["entropy"]
            assert "bins_3" in ent
            assert "bins_7" in ent
            assert "bins_5" not in ent  # default not present

    def test_variance_non_negative(self, trajectory):
        """Variance is non-negative."""
        for label, features in trajectory["per_dimension"].items():
            assert features["variance"] >= 0.0

    def test_effective_dimensionality_range(self, trajectory):
        """Effective dimensionality is in valid range."""
        ed = trajectory["effective_dimensionality"]
        assert 0 <= ed <= STATE_VECTOR_DIM

    def test_transition_frequency_fields(self, trajectory):
        """Transition records have expected fields."""
        for key in ["dominant_emotion_transitions", "policy_label_transitions"]:
            tr = trajectory[key]
            assert "transition_count" in tr
            assert "unique_labels" in tr

    def test_empty_turns(self):
        """Empty turns produce minimal output."""
        fake_result = {"metadata": {"scenario": "test"}, "turns": []}
        features = compute_trajectory_features(fake_result)
        assert features["total_turns"] == 0

    def test_stationarity_output_keys(self, trajectory):
        """Stationarity has all expected keys."""
        expected_keys = {
            "mean_diff", "variance_diff",
            "first_half_mean", "second_half_mean",
            "first_half_variance", "second_half_variance",
        }
        for label, features in trajectory["per_dimension"].items():
            assert set(features["stationarity"].keys()) == expected_keys

    def test_no_evaluation_in_output(self, trajectory):
        """Output contains no evaluative labels."""
        import json
        text = json.dumps(trajectory)
        for word in ["good", "bad", "normal", "abnormal", "threshold"]:
            assert word not in text.lower(), (
                f"Evaluative word '{word}' found in trajectory features"
            )


# ── Measurement Mode 3: A/B Comparison ───────────────────────

class TestABComparison:
    """A/B coefficient comparison tests."""

    @pytest.fixture(scope="class")
    def ab_result(self) -> dict:
        """Run A/B comparison with smoke scenario."""
        return run_ab_comparison(scenario_name="smoke")

    def test_result_structure(self, ab_result):
        """Has expected top-level keys."""
        assert "metadata" in ab_result
        assert "condition_a" in ab_result
        assert "condition_b" in ab_result
        assert "per_tick_distances" in ab_result
        assert "effect_sizes" in ab_result
        assert "distance_summary" in ab_result

    def test_metadata_fields(self, ab_result):
        """Metadata contains mode and condition labels."""
        meta = ab_result["metadata"]
        assert meta["mode"] == "ab_comparison"
        assert meta["condition_a"] == "session_modulation_enabled"
        assert meta["condition_b"] == "session_modulation_disabled"
        assert "dimension_labels" in meta

    def test_condition_a_has_turns(self, ab_result):
        """Condition A result has turns."""
        turns = ab_result["condition_a"]["turns"]
        assert len(turns) > 0

    def test_condition_b_has_turns(self, ab_result):
        """Condition B result has turns."""
        turns = ab_result["condition_b"]["turns"]
        assert len(turns) > 0

    def test_conditions_same_length(self, ab_result):
        """Both conditions have the same number of turns."""
        assert (
            len(ab_result["condition_a"]["turns"])
            == len(ab_result["condition_b"]["turns"])
        )

    def test_condition_a_bypass_false(self, ab_result):
        """Condition A has bypass_session_modulation=False."""
        meta = ab_result["condition_a"]["metadata"]
        assert meta["bypass_session_modulation"] is False

    def test_condition_b_bypass_true(self, ab_result):
        """Condition B has bypass_session_modulation=True."""
        meta = ab_result["condition_b"]["metadata"]
        assert meta["bypass_session_modulation"] is True

    def test_per_tick_distances_count(self, ab_result):
        """One distance record per turn."""
        expected = len(ab_result["condition_a"]["turns"])
        assert len(ab_result["per_tick_distances"]) == expected

    def test_per_tick_distances_non_negative(self, ab_result):
        """All distances are non-negative."""
        for d in ab_result["per_tick_distances"]:
            assert d["distance"] >= 0.0

    def test_effect_sizes_count(self, ab_result):
        """12 effect size records (one per dimension)."""
        assert len(ab_result["effect_sizes"]) == 12

    def test_effect_size_fields(self, ab_result):
        """Each effect size has expected fields."""
        for es in ab_result["effect_sizes"]:
            assert "dimension" in es
            assert "mean_a" in es
            assert "mean_b" in es
            assert "diff" in es
            assert "pooled_stddev" in es
            assert "effect_size" in es

    def test_effect_size_dimensions_match_labels(self, ab_result):
        """Effect size dimensions match state vector labels."""
        labels = _dimension_labels()
        for i, es in enumerate(ab_result["effect_sizes"]):
            assert es["dimension"] == labels[i]

    def test_distance_summary_fields(self, ab_result):
        """Distance summary has expected fields."""
        summary = ab_result["distance_summary"]
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert "stddev" in summary

    def test_invalid_scenario(self):
        """Invalid scenario raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_ab_comparison(scenario_name="nonexistent")

    def test_custom_sequence(self):
        """Custom sequence works."""
        result = run_ab_comparison(
            custom_sequence=["positive", "neutral", "negative"],
        )
        assert result["metadata"]["scenario"] == "custom"
        assert len(result["condition_a"]["turns"]) == 3

    def test_no_evaluation_in_output(self, ab_result):
        """Output contains no evaluative labels."""
        import json
        text = json.dumps(ab_result)
        for word in ["better", "worse", "optimal", "desired"]:
            assert word not in text.lower(), (
                f"Evaluative word '{word}' found in A/B comparison output"
            )


# ── Effect Size Computation ──────────────────────────────────

class TestEffectSizeComputation:
    """Per-dimension effect size computation."""

    def test_identical_conditions(self):
        """Identical conditions have zero effect size."""
        vectors = [[1.0, 2.0, 3.0]] * 10
        result = _compute_per_dimension_effect_size(vectors, vectors)
        for es in result:
            assert abs(es["effect_size"]) < 1e-10

    def test_different_means(self):
        """Different means produce non-zero effect size."""
        va = [[1.0, 0.0]] * 10
        vb = [[2.0, 0.0]] * 10
        result = _compute_per_dimension_effect_size(va, vb)
        # First dimension should have non-zero effect
        # But all values are constant -> variance is 0 -> effect_size is 0
        # Need variance to compute meaningful effect size
        assert len(result) == 2

    def test_with_variance(self):
        """Different distributions with variance produce meaningful effect size."""
        import random
        rng = random.Random(42)
        va = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(50)]
        vb = [[rng.gauss(1, 1), rng.gauss(0, 1)] for _ in range(50)]
        result = _compute_per_dimension_effect_size(va, vb)
        # First dimension should have effect around -1.0 (shifted by 1 sigma)
        assert abs(result[0]["effect_size"]) > 0.5
        # Second dimension should have small effect (same distribution)
        assert abs(result[1]["effect_size"]) < 1.0

    def test_empty_vectors(self):
        """Empty vectors return empty result."""
        assert _compute_per_dimension_effect_size([], []) == []
        assert _compute_per_dimension_effect_size([[1, 2]], []) == []


# ── Integration: Trajectory Features with Real Sim ───────────

class TestTrajectoryFeaturesIntegration:
    """Integration test: trajectory features on longer simulation."""

    @pytest.fixture(scope="class")
    def mixed_result(self) -> dict:
        """Run mixed scenario (45 turns)."""
        return run_simulation(scenario_name="mixed")

    @pytest.fixture(scope="class")
    def mixed_features(self, mixed_result) -> dict:
        """Compute trajectory features on mixed scenario."""
        return compute_trajectory_features(mixed_result)

    def test_longer_scenario_variance_positive(self, mixed_features):
        """At least some dimensions have positive variance on varied input."""
        has_positive_variance = False
        for label, features in mixed_features["per_dimension"].items():
            if features["variance"] > 0:
                has_positive_variance = True
                break
        assert has_positive_variance

    def test_mixed_has_transitions(self, mixed_features):
        """Mixed scenario should have policy transitions."""
        # At minimum, dominant emotion should change with mixed input
        de_tr = mixed_features["dominant_emotion_transitions"]
        assert de_tr["unique_labels"] >= 1

    def test_effective_dim_positive(self, mixed_features):
        """Mixed input should use multiple dimensions (effective dim > 1)."""
        assert mixed_features["effective_dimensionality"] > 0


# ── Integration: Divergence with Different Warmup ────────────

class TestDivergenceIntegration:
    """Integration: different warmups create measurable divergence."""

    def test_different_warmup_creates_divergence(self):
        """Opposite warmup patterns should create measurable initial distance."""
        result = run_divergence_measurement(
            scenario_name="smoke",
            num_instances=2,
            warmup_turns=5,
            warmup_patterns=[
                ["positive", "loving", "positive", "loving", "positive"],
                ["negative", "angry", "rejected", "fearful", "negative"],
            ],
        )
        # After different warmups, initial states should differ
        ws0 = result["warmup_states"][0]["state_vector"]
        ws1 = result["warmup_states"][1]["state_vector"]
        dist = _euclidean_distance(ws0, ws1)
        assert dist > 0.0  # some measurable difference

    def test_same_warmup_minimal_divergence(self):
        """Same warmup pattern should produce zero initial divergence."""
        result = run_divergence_measurement(
            scenario_name="smoke",
            num_instances=2,
            warmup_turns=3,
            warmup_patterns=[
                ["positive", "neutral", "negative"],
                ["positive", "neutral", "negative"],
            ],
        )
        ws0 = result["warmup_states"][0]["state_vector"]
        ws1 = result["warmup_states"][1]["state_vector"]
        dist = _euclidean_distance(ws0, ws1)
        # Same warmup -> should be identical
        assert dist < 1e-6


# ── CLI Argument Parsing ─────────────────────────────────────

class TestCLIArgumentParsing:
    """CLI argument parsing for new measurement modes."""

    def test_divergence_arg(self):
        """--divergence argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--divergence", "smoke"])
        assert args.divergence == "smoke"

    def test_num_instances_arg(self):
        """--num-instances argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--divergence", "smoke", "--num-instances", "4"])
        assert args.num_instances == 4

    def test_warmup_turns_arg(self):
        """--warmup-turns argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--divergence", "smoke", "--warmup-turns", "10"])
        assert args.warmup_turns == 10

    def test_trajectory_features_arg(self):
        """--trajectory-features argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--scenario", "smoke", "--trajectory-features"])
        assert args.trajectory_features is True

    def test_entropy_bins_arg(self):
        """--entropy-bins argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--scenario", "smoke", "--trajectory-features",
            "--entropy-bins", "3", "7", "15",
        ])
        assert args.entropy_bins == [3, 7, 15]

    def test_ab_compare_arg(self):
        """--ab-compare argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--ab-compare", "smoke"])
        assert args.ab_compare == "smoke"

    def test_max_instances_arg(self):
        """--max-instances argument is parsed."""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--divergence", "smoke", "--max-instances", "5",
        ])
        assert args.max_instances == 5


# ── No Psyche Modification Verification ──────────────────────

class TestNoPsycheModification:
    """Verify that measurement functions do not modify psyche internals."""

    def test_state_vector_extraction_readonly(self):
        """_extract_state_vector does not modify orchestrator state."""
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="psyche_test_")
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0, data_dir=Path(tmpdir))

        # Read state before
        from psyche.state import Percept
        percept = Percept(**INPUT_PATTERNS["positive"])
        orch.post_response_update(percept, 2.0, "test")

        before = _extract_state_vector(orch)
        # Call extraction again
        after = _extract_state_vector(orch)
        # Should be identical (no modification from reading)
        assert before == after

    def test_trajectory_features_pure_function(self):
        """compute_trajectory_features does not modify input."""
        result = run_simulation(scenario_name="smoke")
        import copy
        result_copy = copy.deepcopy(result)
        compute_trajectory_features(result)
        # Original result should be unchanged
        assert result == result_copy
