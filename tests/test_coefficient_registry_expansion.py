"""
tests/test_coefficient_registry_expansion.py - Tests for coefficient registry expansion

Tests for the externalization of memory_emotion_return and
other_hypothesis_emotion_return constants to the coefficient registry.

Verifies:
- Each constant returns the same value via coefficient registry as the
  previous hardcoded default
- Fallback mechanism works when coefficients.json is absent
- Partial override resilience (overriding some constants preserves defaults
  for the rest)
- Existing tests continue to pass (equivalence verification)
"""

import json
import os

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
# Test: memory_emotion_return defaults match hardcoded values
# =============================================================================

class TestMemoryEmotionReturnDefaults:
    """All memory_emotion_return defaults must be identical to the previous
    hardcoded values in MemoryEmotionReturnConfig."""

    def test_per_candidate_max_delta(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["per_candidate_max_delta"] == 0.03

    def test_total_max_delta(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["total_max_delta"] == 0.15

    def test_rumination_threshold(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["rumination_threshold"] == 2

    def test_rumination_decay_factor(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["rumination_decay_factor"] == 0.5

    def test_low_arousal_threshold(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["low_arousal_threshold"] == 0.2

    def test_low_arousal_scale(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["low_arousal_scale"] == 0.3

    def test_convergence_scale(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["convergence_scale"] == 0.5

    def test_direction_freshness_decay(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["direction_freshness_decay"] == 0.8

    def test_tracking_speed_modulation_ratio_cap(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["tracking_speed_modulation_ratio_cap"] == 0.10

    def test_tracking_speed_modulation_scale(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["memory_emotion_return"]["tracking_speed_modulation_scale"] == 0.02

    def test_all_10_constants_present(self):
        defaults = coefficient_registry.get_defaults()
        mer = defaults["memory_emotion_return"]
        expected_keys = {
            "per_candidate_max_delta",
            "total_max_delta",
            "rumination_threshold",
            "rumination_decay_factor",
            "low_arousal_threshold",
            "low_arousal_scale",
            "convergence_scale",
            "direction_freshness_decay",
            "tracking_speed_modulation_ratio_cap",
            "tracking_speed_modulation_scale",
        }
        assert set(mer.keys()) == expected_keys

    def test_category_accessible_via_get(self):
        """get("memory_emotion_return") returns the full category dict."""
        result = coefficient_registry.get("memory_emotion_return")
        assert isinstance(result, dict)
        assert result["per_candidate_max_delta"] == 0.03
        assert result["total_max_delta"] == 0.15

    def test_specific_key_accessible_via_get(self):
        """get("memory_emotion_return", key) returns specific values."""
        assert coefficient_registry.get("memory_emotion_return", "per_candidate_max_delta") == 0.03
        assert coefficient_registry.get("memory_emotion_return", "rumination_threshold") == 2
        assert coefficient_registry.get("memory_emotion_return", "direction_freshness_decay") == 0.8


# =============================================================================
# Test: other_hypothesis_emotion_return defaults match hardcoded values
# =============================================================================

class TestOtherHypothesisEmotionReturnDefaults:
    """All other_hypothesis_emotion_return defaults must be identical to the
    previous hardcoded values in OtherHypothesisEmotionReturnConfig."""

    def test_per_candidate_max_delta(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["per_candidate_max_delta"] == 0.02

    def test_total_max_delta(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["total_max_delta"] == 0.07

    def test_rumination_threshold(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["rumination_threshold"] == 2

    def test_rumination_decay_factor(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["rumination_decay_factor"] == 0.5

    def test_low_arousal_threshold(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["low_arousal_threshold"] == 0.2

    def test_low_arousal_scale(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["low_arousal_scale"] == 0.3

    def test_convergence_scale(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["convergence_scale"] == 0.5

    def test_combined_max_delta(self):
        defaults = coefficient_registry.get_defaults()
        assert defaults["other_hypothesis_emotion_return"]["combined_max_delta"] == 0.15

    def test_all_8_constants_present(self):
        defaults = coefficient_registry.get_defaults()
        oher = defaults["other_hypothesis_emotion_return"]
        expected_keys = {
            "per_candidate_max_delta",
            "total_max_delta",
            "rumination_threshold",
            "rumination_decay_factor",
            "low_arousal_threshold",
            "low_arousal_scale",
            "convergence_scale",
            "combined_max_delta",
        }
        assert set(oher.keys()) == expected_keys

    def test_category_accessible_via_get(self):
        """get("other_hypothesis_emotion_return") returns the full category dict."""
        result = coefficient_registry.get("other_hypothesis_emotion_return")
        assert isinstance(result, dict)
        assert result["per_candidate_max_delta"] == 0.02
        assert result["combined_max_delta"] == 0.15

    def test_specific_key_accessible_via_get(self):
        """get("other_hypothesis_emotion_return", key) returns specific values."""
        assert coefficient_registry.get("other_hypothesis_emotion_return", "per_candidate_max_delta") == 0.02
        assert coefficient_registry.get("other_hypothesis_emotion_return", "total_max_delta") == 0.07
        assert coefficient_registry.get("other_hypothesis_emotion_return", "combined_max_delta") == 0.15


# =============================================================================
# Test: Fallback mechanism (no coefficients.json)
# =============================================================================

class TestFallbackMechanism:
    """When coefficients.json is absent, all defaults are used."""

    def test_memory_emotion_return_fallback(self):
        """Loading a nonexistent file still provides correct MER defaults."""
        coefficient_registry.load("/nonexistent/path/coefficients.json")
        defaults = coefficient_registry.get_defaults()
        result = coefficient_registry.get("memory_emotion_return")
        assert result == defaults["memory_emotion_return"]

    def test_other_hypothesis_emotion_return_fallback(self):
        """Loading a nonexistent file still provides correct OHER defaults."""
        coefficient_registry.load("/nonexistent/path/coefficients.json")
        defaults = coefficient_registry.get_defaults()
        result = coefficient_registry.get("other_hypothesis_emotion_return")
        assert result == defaults["other_hypothesis_emotion_return"]

    def test_auto_init_memory_emotion_return(self):
        """get() should work without explicit load() for new categories."""
        result = coefficient_registry.get("memory_emotion_return", "per_candidate_max_delta")
        assert result == 0.03

    def test_auto_init_other_hypothesis_emotion_return(self):
        """get() should work without explicit load() for new categories."""
        result = coefficient_registry.get("other_hypothesis_emotion_return", "per_candidate_max_delta")
        assert result == 0.02

    def test_all_10_categories_accessible(self):
        """All 10 categories (8 existing + 2 new) should be accessible."""
        coefficient_registry.load("/nonexistent.json")
        expected_categories = [
            "drive_dynamics", "mood_autonomy", "policy_selection",
            "value_orientation", "fluctuation", "experience_intensity",
            "emotion_processing", "perception",
            "memory_emotion_return", "other_hypothesis_emotion_return",
        ]
        for cat in expected_categories:
            result = coefficient_registry.get(cat)
            assert isinstance(result, dict), f"Category '{cat}' should be a dict"


# =============================================================================
# Test: Partial override resilience
# =============================================================================

class TestPartialOverrideResilience:
    """When only some constants are overridden, the rest use defaults."""

    def test_partial_memory_emotion_return(self, tmp_path):
        """Override one MER constant; others remain default."""
        partial = {
            "memory_emotion_return": {
                "per_candidate_max_delta": 0.05,
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, partial)
        coefficient_registry.load(path)

        # Changed value
        assert coefficient_registry.get("memory_emotion_return", "per_candidate_max_delta") == 0.05

        # Unchanged values should be defaults
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("memory_emotion_return", "total_max_delta") == defaults["memory_emotion_return"]["total_max_delta"]
        assert coefficient_registry.get("memory_emotion_return", "rumination_threshold") == defaults["memory_emotion_return"]["rumination_threshold"]
        assert coefficient_registry.get("memory_emotion_return", "direction_freshness_decay") == defaults["memory_emotion_return"]["direction_freshness_decay"]
        assert coefficient_registry.get("memory_emotion_return", "tracking_speed_modulation_scale") == defaults["memory_emotion_return"]["tracking_speed_modulation_scale"]

    def test_partial_other_hypothesis_emotion_return(self, tmp_path):
        """Override one OHER constant; others remain default."""
        partial = {
            "other_hypothesis_emotion_return": {
                "combined_max_delta": 0.20,
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, partial)
        coefficient_registry.load(path)

        # Changed value
        assert coefficient_registry.get("other_hypothesis_emotion_return", "combined_max_delta") == 0.20

        # Unchanged values should be defaults
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("other_hypothesis_emotion_return", "per_candidate_max_delta") == defaults["other_hypothesis_emotion_return"]["per_candidate_max_delta"]
        assert coefficient_registry.get("other_hypothesis_emotion_return", "total_max_delta") == defaults["other_hypothesis_emotion_return"]["total_max_delta"]
        assert coefficient_registry.get("other_hypothesis_emotion_return", "convergence_scale") == defaults["other_hypothesis_emotion_return"]["convergence_scale"]

    def test_mixed_override_new_and_existing(self, tmp_path):
        """Override constants in both new and existing categories."""
        partial = {
            "memory_emotion_return": {
                "low_arousal_scale": 0.5,
            },
            "drive_dynamics": {
                "total_change_limit": 0.20,
            },
        }
        path = str(tmp_path / "coefficients.json")
        _write_json(path, partial)
        coefficient_registry.load(path)

        # Changed values
        assert coefficient_registry.get("memory_emotion_return", "low_arousal_scale") == 0.5
        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.20

        # Unchanged values should be defaults
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("memory_emotion_return", "per_candidate_max_delta") == defaults["memory_emotion_return"]["per_candidate_max_delta"]
        assert coefficient_registry.get("drive_dynamics", "section_band") == defaults["drive_dynamics"]["section_band"]

    def test_empty_file_all_defaults(self, tmp_path):
        """Empty JSON object -> all defaults for new categories too."""
        path = str(tmp_path / "coefficients.json")
        _write_json(path, {})
        coefficient_registry.load(path)

        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("memory_emotion_return") == defaults["memory_emotion_return"]
        assert coefficient_registry.get("other_hypothesis_emotion_return") == defaults["other_hypothesis_emotion_return"]


# =============================================================================
# Test: Module-level Config integration
# =============================================================================

class TestConfigIntegration:
    """Verify that Config dataclasses in the modules get correct values
    from the coefficient registry."""

    def test_memory_emotion_return_config_defaults(self):
        """MemoryEmotionReturnConfig defaults should match registry values."""
        from psyche.memory_emotion_return import MemoryEmotionReturnConfig
        config = MemoryEmotionReturnConfig()
        assert config.per_candidate_max_delta == 0.03
        assert config.total_max_delta == 0.15
        assert config.rumination_threshold == 2
        assert config.rumination_decay_factor == 0.5
        assert config.low_arousal_threshold == 0.2
        assert config.low_arousal_scale == 0.3
        assert config.convergence_scale == 0.5
        assert config.direction_freshness_decay == 0.8
        assert config.tracking_speed_modulation_ratio_cap == 0.10
        assert config.tracking_speed_modulation_scale == 0.02
        # Non-externalized constant should still have its value
        assert config.history_window_size == 50

    def test_other_hypothesis_emotion_return_config_defaults(self):
        """OtherHypothesisEmotionReturnConfig defaults should match registry values."""
        from psyche.other_hypothesis_emotion_return import OtherHypothesisEmotionReturnConfig
        config = OtherHypothesisEmotionReturnConfig()
        assert config.per_candidate_max_delta == 0.02
        assert config.total_max_delta == 0.07
        assert config.rumination_threshold == 2
        assert config.rumination_decay_factor == 0.5
        assert config.low_arousal_threshold == 0.2
        assert config.low_arousal_scale == 0.3
        assert config.convergence_scale == 0.5
        assert config.combined_max_delta == 0.15
        # Non-externalized constant should still have its value
        assert config.history_window_size == 50

    def test_memory_emotion_return_config_explicit_override(self):
        """Explicitly passed values should override registry defaults."""
        from psyche.memory_emotion_return import MemoryEmotionReturnConfig
        config = MemoryEmotionReturnConfig(per_candidate_max_delta=0.05)
        assert config.per_candidate_max_delta == 0.05
        # Other values remain registry defaults
        assert config.total_max_delta == 0.15
        assert config.convergence_scale == 0.5

    def test_other_hypothesis_emotion_return_config_explicit_override(self):
        """Explicitly passed values should override registry defaults."""
        from psyche.other_hypothesis_emotion_return import OtherHypothesisEmotionReturnConfig
        config = OtherHypothesisEmotionReturnConfig(combined_max_delta=0.20)
        assert config.combined_max_delta == 0.20
        # Other values remain registry defaults
        assert config.per_candidate_max_delta == 0.02
        assert config.total_max_delta == 0.07


# =============================================================================
# Test: Read-only enforcement for new categories
# =============================================================================

class TestReadOnlyNewCategories:
    """Mutating returned dicts should not affect the internal registry."""

    def test_memory_emotion_return_mutation_safe(self):
        result = coefficient_registry.get("memory_emotion_return")
        result["per_candidate_max_delta"] = 999.0
        fresh = coefficient_registry.get("memory_emotion_return")
        assert fresh["per_candidate_max_delta"] == 0.03

    def test_other_hypothesis_emotion_return_mutation_safe(self):
        result = coefficient_registry.get("other_hypothesis_emotion_return")
        result["combined_max_delta"] = 999.0
        fresh = coefficient_registry.get("other_hypothesis_emotion_return")
        assert fresh["combined_max_delta"] == 0.15


# =============================================================================
# Test: JSON file matches defaults for new categories
# =============================================================================

class TestJsonFileNewCategories:
    """The provided JSON file should contain the new category values."""

    def test_json_file_matches_defaults_memory_emotion_return(self):
        """data/coefficients.json memory_emotion_return should match defaults."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "data", "coefficients.json")

        if not os.path.isfile(json_path):
            pytest.skip("coefficients.json not found")

        coefficient_registry.load(json_path)
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("memory_emotion_return") == defaults["memory_emotion_return"]

    def test_json_file_matches_defaults_other_hypothesis_emotion_return(self):
        """data/coefficients.json other_hypothesis_emotion_return should match defaults."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "data", "coefficients.json")

        if not os.path.isfile(json_path):
            pytest.skip("coefficients.json not found")

        coefficient_registry.load(json_path)
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("other_hypothesis_emotion_return") == defaults["other_hypothesis_emotion_return"]


# =============================================================================
# Test: Factory function equivalence
# =============================================================================

class TestFactoryEquivalence:
    """Processor created via factory function should have correct config."""

    def test_memory_emotion_return_factory(self):
        from psyche.memory_emotion_return import create_memory_emotion_return
        processor = create_memory_emotion_return()
        config = processor._config
        assert config.per_candidate_max_delta == 0.03
        assert config.total_max_delta == 0.15
        assert config.rumination_threshold == 2
        assert config.rumination_decay_factor == 0.5
        assert config.low_arousal_threshold == 0.2
        assert config.low_arousal_scale == 0.3
        assert config.convergence_scale == 0.5
        assert config.direction_freshness_decay == 0.8
        assert config.tracking_speed_modulation_ratio_cap == 0.10
        assert config.tracking_speed_modulation_scale == 0.02

    def test_other_hypothesis_emotion_return_factory(self):
        from psyche.other_hypothesis_emotion_return import create_other_hypothesis_emotion_return
        processor = create_other_hypothesis_emotion_return()
        config = processor._config
        assert config.per_candidate_max_delta == 0.02
        assert config.total_max_delta == 0.07
        assert config.rumination_threshold == 2
        assert config.rumination_decay_factor == 0.5
        assert config.low_arousal_threshold == 0.2
        assert config.low_arousal_scale == 0.3
        assert config.convergence_scale == 0.5
        assert config.combined_max_delta == 0.15
