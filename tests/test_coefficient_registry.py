"""
tests/test_coefficient_registry.py - Tests for coefficient_registry module.

Tests:
- File absent: all defaults used
- Individual constant missing: defaults for that constant
- File values loaded correctly
- Read-only enforcement (no write path)
- Default values match previous hardcoded values exactly
- Enrichment non-exposure
- Category/key access patterns
- Reset and reload
"""

import json
import os
import tempfile

import pytest

from psyche import coefficient_registry


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before and after each test."""
    coefficient_registry.reset()
    yield
    coefficient_registry.reset()


def _write_json(path: str, data: dict) -> None:
    """Helper to write JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# =============================================================================
# Test: File absent -> all defaults
# =============================================================================

class TestFileAbsent:
    """When coefficient file does not exist, all defaults are used."""

    def test_load_nonexistent_file_uses_defaults(self):
        coefficient_registry.load("/nonexistent/path/coefficients.json")
        defaults = coefficient_registry.get_defaults()

        for category in defaults:
            result = coefficient_registry.get(category)
            assert result == defaults[category], (
                f"Category '{category}' should match defaults when file is absent"
            )

    def test_auto_init_without_explicit_load(self):
        """get() should work even without explicit load()."""
        result = coefficient_registry.get("drive_dynamics", "total_change_limit")
        assert result == 0.15

    def test_all_categories_accessible(self):
        coefficient_registry.load("/nonexistent/path.json")
        expected_categories = [
            "drive_dynamics", "mood_autonomy", "policy_selection",
            "value_orientation", "fluctuation", "experience_intensity",
            "emotion_processing", "perception",
        ]
        for cat in expected_categories:
            result = coefficient_registry.get(cat)
            assert isinstance(result, dict), f"Category '{cat}' should be a dict"


# =============================================================================
# Test: Individual constant missing -> default for that constant
# =============================================================================

class TestPartialFile:
    """When file has some but not all constants, missing ones use defaults."""

    def test_partial_category(self, tmp_path):
        """File with only some categories -> missing categories use defaults."""
        partial = {
            "drive_dynamics": {
                "total_change_limit": 0.20,
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, partial)
        coefficient_registry.load(path)

        # Changed value
        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.20

        # Missing key in present category -> default
        section_band = coefficient_registry.get("drive_dynamics", "section_band")
        defaults = coefficient_registry.get_defaults()
        assert section_band == defaults["drive_dynamics"]["section_band"]

        # Missing category entirely -> all defaults
        assert coefficient_registry.get("mood_autonomy") == defaults["mood_autonomy"]

    def test_partial_nested_values(self, tmp_path):
        """File with partial nested dict -> missing nested keys use defaults."""
        partial = {
            "mood_autonomy": {
                "tracking_speed_min": 0.05,
                # tracking_speed_max, mood_delta_limit, mood_band not specified
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, partial)
        coefficient_registry.load(path)

        result = coefficient_registry.get("mood_autonomy")
        assert result["tracking_speed_min"] == 0.05
        # Missing keys should be defaults
        defaults = coefficient_registry.get_defaults()
        assert result["tracking_speed_max"] == defaults["mood_autonomy"]["tracking_speed_max"]
        assert result["mood_delta_limit"] == defaults["mood_autonomy"]["mood_delta_limit"]

    def test_empty_file(self, tmp_path):
        """Empty JSON object -> all defaults."""
        path = str(tmp_path / "coefficients.json")
        _write_json(path, {})
        coefficient_registry.load(path)

        defaults = coefficient_registry.get_defaults()
        for category in defaults:
            assert coefficient_registry.get(category) == defaults[category]


# =============================================================================
# Test: File values loaded correctly
# =============================================================================

class TestFileLoading:
    """File values are correctly loaded and accessible."""

    def test_custom_values_loaded(self, tmp_path):
        """Custom values from file should be used."""
        custom = {
            "drive_dynamics": {
                "total_change_limit": 0.25,
                "section_band": {
                    "emotion_drive_coupling": {"social": 0.10, "curiosity": 0.10, "expression": 0.10},
                },
            },
            "fluctuation": {
                "amplitude_cap": 0.10,
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, custom)
        coefficient_registry.load(path)

        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.25
        section_band = coefficient_registry.get("drive_dynamics", "section_band")
        assert section_band["emotion_drive_coupling"]["social"] == 0.10
        assert coefficient_registry.get("fluctuation", "amplitude_cap") == 0.10

    def test_invalid_json_falls_back_to_defaults(self, tmp_path):
        """Invalid JSON file -> all defaults."""
        path = str(tmp_path / "coefficients.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        coefficient_registry.load(path)

        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("drive_dynamics") == defaults["drive_dynamics"]

    def test_non_dict_json_falls_back_to_defaults(self, tmp_path):
        """JSON file with non-dict content -> all defaults."""
        path = str(tmp_path / "coefficients.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        coefficient_registry.load(path)

        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("drive_dynamics") == defaults["drive_dynamics"]

    def test_get_entire_category(self):
        """get(category) returns the entire category dict."""
        coefficient_registry.load("/nonexistent.json")
        result = coefficient_registry.get("value_orientation")
        assert isinstance(result, dict)
        assert "base_learning_rate" in result
        assert "max_bias_strength" in result

    def test_get_specific_key(self):
        """get(category, key) returns a specific value."""
        coefficient_registry.load("/nonexistent.json")
        result = coefficient_registry.get("policy_selection", "score_section_band")
        assert result == 1.5

    def test_unknown_category_raises(self):
        """get() with unknown category raises KeyError."""
        coefficient_registry.load("/nonexistent.json")
        with pytest.raises(KeyError, match="Unknown coefficient category"):
            coefficient_registry.get("nonexistent_category")

    def test_unknown_key_raises(self):
        """get() with unknown key raises KeyError."""
        coefficient_registry.load("/nonexistent.json")
        with pytest.raises(KeyError, match="Unknown coefficient key"):
            coefficient_registry.get("drive_dynamics", "nonexistent_key")


# =============================================================================
# Test: Read-only (no write path)
# =============================================================================

class TestReadOnly:
    """The registry is read-only after loading."""

    def test_returned_dict_mutation_does_not_affect_registry(self):
        """Mutating the returned dict should not affect the internal registry."""
        coefficient_registry.load("/nonexistent.json")

        # Get and mutate
        result = coefficient_registry.get("drive_dynamics")
        result["total_change_limit"] = 999.0
        result["section_band"]["emotion_drive_coupling"]["social"] = 999.0

        # Re-get should still show original values
        fresh = coefficient_registry.get("drive_dynamics")
        assert fresh["total_change_limit"] == 0.15
        assert fresh["section_band"]["emotion_drive_coupling"]["social"] == 0.06

    def test_no_set_method_exists(self):
        """There should be no set/update/write method on the module."""
        public_attrs = [a for a in dir(coefficient_registry) if not a.startswith("_")]
        write_names = {"set", "update", "write", "save", "put", "modify"}
        for attr in public_attrs:
            assert attr not in write_names, (
                f"Module should not have write method '{attr}'"
            )

    def test_defaults_copy_independence(self):
        """get_defaults() returns an independent copy."""
        d1 = coefficient_registry.get_defaults()
        d2 = coefficient_registry.get_defaults()
        d1["drive_dynamics"]["total_change_limit"] = 999.0
        assert d2["drive_dynamics"]["total_change_limit"] == 0.15


# =============================================================================
# Test: Default values match previous hardcoded values exactly
# =============================================================================

class TestDefaultsMatchHardcoded:
    """All default values must be identical to the previous hardcoded values."""

    def test_drive_dynamics_section_band(self):
        defaults = coefficient_registry.get_defaults()
        sb = defaults["drive_dynamics"]["section_band"]
        assert sb["emotion_drive_coupling"] == {"social": 0.06, "curiosity": 0.06, "expression": 0.06}
        assert sb["drive_interaction"] == {"social": 0.03, "curiosity": 0.03, "expression": 0.03}
        assert sb["goal_hierarchy"] == {"social": 0.05, "curiosity": 0.05, "expression": 0.05}
        assert sb["time_passage"] == {"social": 0.06, "curiosity": 0.06, "expression": 0.06}
        assert sb["arousal_drive"] == {"social": 0.04, "curiosity": 0.04, "expression": 0.04}
        assert sb["behavioral_diversity"] == {"social": 0.02, "curiosity": 0.02, "expression": 0.02}
        assert sb["internal_contradiction"] == {"social": 0.02, "curiosity": 0.02, "expression": 0.02}
        assert sb["result_diversity_return"] == {"social": 0.03, "curiosity": 0.03, "expression": 0.03}

    def test_drive_dynamics_total_change_limit(self):
        assert coefficient_registry.get_defaults()["drive_dynamics"]["total_change_limit"] == 0.15

    def test_mood_band(self):
        defaults = coefficient_registry.get_defaults()
        mb = defaults["mood_autonomy"]["mood_band"]
        assert mb["emotion"] == {"valence": 0.12, "arousal": 0.10}
        assert mb["drive"] == {"valence": 0.05, "arousal": 0.04}
        assert mb["goal"] == {"valence": 0.03, "arousal": 0.02}
        assert mb["fear"] == {"valence": 0.00, "arousal": 0.06}

    def test_mood_tracking_speeds(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["mood_autonomy"]["tracking_speed_min"] == 0.03
        assert defaults["mood_autonomy"]["tracking_speed_max"] == 0.25

    def test_mood_delta_limit(self):
        assert coefficient_registry.get_defaults()["mood_autonomy"]["mood_delta_limit"] == 0.15

    def test_score_section_band(self):
        assert coefficient_registry.get_defaults()["policy_selection"]["score_section_band"] == 1.5

    def test_value_orientation_constants(self):
        defaults = coefficient_registry.get_defaults()
        vo = defaults["value_orientation"]
        assert vo["base_learning_rate"] == 0.01
        assert vo["confidence_damping"] == 0.5
        assert vo["confidence_growth_rate"] == 0.005
        assert vo["confidence_decay_rate"] == 0.001
        assert vo["max_bias_strength"] == 0.15
        assert vo["min_dimension_threshold"] == 0.1
        assert vo["confidence_bias_amplifier"] == 0.5
        assert vo["neutral_decay_rate"] == 0.0001

    def test_fluctuation_constants(self):
        defaults = coefficient_registry.get_defaults()
        fl = defaults["fluctuation"]
        assert fl["amplitude_cap"] == 0.12
        assert fl["amplitude_floor"] == 0.005

    def test_experience_intensity_constants(self):
        defaults = coefficient_registry.get_defaults()
        ei = defaults["experience_intensity"]
        assert ei["bandwidth_max_multiplier"] == 4.0
        assert ei["bandwidth_max_delta_per_dim"] == 0.08
        assert ei["bandwidth_cooldown_ticks"] == 3
        assert ei["cooldown_min_ticks"] == 2
        assert ei["drive_limit_multiplier_max"] == 1.3
        assert ei["score_band_addition_max"] == 0.5

    def test_emotion_processing_constants(self):
        defaults = coefficient_registry.get_defaults()
        ep = defaults["emotion_processing"]
        assert ep["decay_rate"] == 0.95
        assert ep["stimulus_base_delta"] == 0.2
        assert ep["valence_positive"] == {"joy": 0.15, "love": 0.05, "fun": 0.05}
        assert ep["valence_negative"] == {"sorrow": 0.10, "anger": 0.05, "fear": 0.05}

    def test_perception_constants(self):
        defaults = coefficient_registry.get_defaults()
        pc = defaults["perception"]
        assert pc["bias_bandwidth"] == 0.04
        assert pc["bias_coefficient"] == 0.1


# =============================================================================
# Test: Enrichment non-exposure
# =============================================================================

class TestEnrichmentNonExposure:
    """The coefficient registry must not be exposed in enrichment."""

    def test_no_enrichment_method(self):
        """Module should not have any enrichment-related methods."""
        public_attrs = [a for a in dir(coefficient_registry) if not a.startswith("_")]
        enrichment_names = {
            "get_enrichment", "to_enrichment", "enrichment_entry",
            "get_prompt_enrichment", "enrichment",
        }
        for attr in public_attrs:
            assert attr not in enrichment_names, (
                f"Module should not have enrichment method '{attr}'"
            )


# =============================================================================
# Test: Reset functionality
# =============================================================================

class TestReset:
    """Reset clears the registry state."""

    def test_reset_allows_reload(self, tmp_path):
        """After reset, a different file can be loaded."""
        # Load with custom value
        custom = {"drive_dynamics": {"total_change_limit": 0.30}}
        path = str(tmp_path / "coefficients.json")
        _write_json(path, custom)
        coefficient_registry.load(path)
        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.30

        # Reset and load with defaults
        coefficient_registry.reset()
        coefficient_registry.load("/nonexistent.json")
        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.15


# =============================================================================
# Test: JSON file matches defaults
# =============================================================================

class TestJsonFileContent:
    """The provided JSON file should match all defaults."""

    def test_json_file_matches_defaults(self):
        """data/coefficients.json should contain all default values."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "data", "coefficients.json")

        if not os.path.isfile(json_path):
            pytest.skip("coefficients.json not found")

        with open(json_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)

        defaults = coefficient_registry.get_defaults()

        # Load from file and verify equivalence
        coefficient_registry.load(json_path)
        for category in defaults:
            assert coefficient_registry.get(category) == defaults[category], (
                f"Category '{category}' from JSON file should match defaults"
            )


# =============================================================================
# Test: Module-level integration (import-time constants)
# =============================================================================

class TestModuleIntegration:
    """Verify that modules that import from coefficient_registry get correct values."""

    def test_reaction_constants_accessible(self):
        """reaction.py should have its constants from the registry."""
        from psyche import reaction
        # These should be the default values
        assert reaction._TOTAL_CHANGE_LIMIT == 0.15
        assert reaction.DECAY_RATE == 0.95
        assert reaction._SECTION_BAND["emotion_drive_coupling"]["social"] == 0.06

    def test_thought_constant_accessible(self):
        """thought.py should have its constant from the registry."""
        from psyche import thought
        assert thought._SCORE_SECTION_BAND == 1.5

    def test_scoring_fluctuation_defaults(self):
        """scoring_fluctuation.py config defaults from the registry."""
        from psyche.scoring_fluctuation import ScoringFluctuationConfig
        config = ScoringFluctuationConfig()
        assert config.amplitude_cap == 0.12
        assert config.amplitude_floor == 0.005

    def test_value_orientation_defaults(self):
        """value_orientation.py config defaults from the registry."""
        from psyche.value_orientation import ValueOrientationConfig
        config = ValueOrientationConfig()
        assert config.base_learning_rate == 0.01
        assert config.max_bias_strength == 0.15
        assert config.confidence_damping == 0.5
        assert config.confidence_growth_rate == 0.005
        assert config.confidence_decay_rate == 0.001
        assert config.min_dimension_threshold == 0.1
        assert config.confidence_bias_amplifier == 0.5
        assert config.neutral_decay_rate == 0.0001

    def test_perception_constants_accessible(self):
        """perception.py should have its constants from the registry (deferred load)."""
        from psyche import perception
        # Constants are lazily loaded; trigger via accessor
        bandwidth, coeff = perception._get_perception_coeffs()
        assert bandwidth == 0.04
        assert coeff == 0.1
        # After first call, module-level cache is populated
        assert perception._BIAS_BANDWIDTH == 0.04
        assert perception._BIAS_COEFFICIENT == 0.1
