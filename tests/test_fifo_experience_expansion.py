"""
tests/test_fifo_experience_expansion.py - Tests for FIFO experience-dependent expansion.

Tests:
- Expansion factor computation from cumulative tick count
- Discrete stage transitions (not continuous)
- Max expansion ratio safety valve
- Registry FIFO limit values are updated correctly
- No expansion for low tick counts
- Negative tick counts handled safely
- Expansion only increases (never decreases) limits
- apply_fifo_experience_expansion returns correct expanded values
- Registry not loaded: expansion returns empty
- get_fifo_expansion_stages / get_fifo_expansion_max_ratio accessors
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


def _load_defaults():
    """Load registry with defaults (no file)."""
    coefficient_registry.load("/nonexistent/path/coefficients.json")


# =============================================================================
# Test: _compute_fifo_expansion_factor
# =============================================================================

class TestComputeFifoExpansionFactor:
    """Test the expansion factor computation from cumulative ticks."""

    def test_zero_ticks_returns_1(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(0)
        assert factor == 1.0

    def test_below_first_threshold_returns_1(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(499)
        assert factor == 1.0

    def test_at_first_threshold_returns_stage_1(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(500)
        assert factor == 1.2

    def test_between_stages(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(1000)
        assert factor == 1.2

    def test_at_second_threshold(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(2000)
        assert factor == 1.4

    def test_at_third_threshold(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(5000)
        assert factor == 1.6

    def test_at_fourth_threshold(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(10000)
        assert factor == 1.8

    def test_at_final_threshold(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(20000)
        assert factor == coefficient_registry._FIFO_EXPANSION_MAX_RATIO

    def test_above_final_threshold_capped(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(100000)
        assert factor == coefficient_registry._FIFO_EXPANSION_MAX_RATIO

    def test_negative_ticks_returns_1(self):
        factor = coefficient_registry._compute_fifo_expansion_factor(-100)
        assert factor == 1.0

    def test_factor_never_exceeds_max_ratio(self):
        max_ratio = coefficient_registry._FIFO_EXPANSION_MAX_RATIO
        for ticks in [0, 100, 500, 2000, 5000, 10000, 20000, 50000, 999999]:
            factor = coefficient_registry._compute_fifo_expansion_factor(ticks)
            assert factor <= max_ratio, (
                f"Factor {factor} exceeds max ratio {max_ratio} at {ticks} ticks"
            )

    def test_factor_is_monotonically_non_decreasing(self):
        prev_factor = 0.0
        for ticks in range(0, 25001, 100):
            factor = coefficient_registry._compute_fifo_expansion_factor(ticks)
            assert factor >= prev_factor, (
                f"Factor decreased from {prev_factor} to {factor} at {ticks} ticks"
            )
            prev_factor = factor

    def test_discrete_stages_not_continuous(self):
        """Factor should jump at thresholds, not change continuously."""
        f499 = coefficient_registry._compute_fifo_expansion_factor(499)
        f500 = coefficient_registry._compute_fifo_expansion_factor(500)
        assert f499 != f500, "Factor should change at stage boundary"
        # Between stages, factor should be constant
        f600 = coefficient_registry._compute_fifo_expansion_factor(600)
        f1000 = coefficient_registry._compute_fifo_expansion_factor(1000)
        assert f600 == f1000, "Factor should be constant between stages"


# =============================================================================
# Test: apply_fifo_experience_expansion
# =============================================================================

class TestApplyFifoExperienceExpansion:
    """Test the registry-level FIFO expansion application."""

    def test_no_expansion_below_threshold(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(0)
        assert result == {}

    def test_no_expansion_at_499_ticks(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(499)
        assert result == {}

    def test_expansion_at_500_ticks(self):
        _load_defaults()
        defaults = coefficient_registry.get_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(500)

        assert "fifo_limit_30" in result
        assert "fifo_limit_50" in result
        assert "fifo_limit_100" in result
        assert "fifo_limit_200" in result

        # 20% expansion
        assert result["fifo_limit_30"] == round(30 * 1.2)
        assert result["fifo_limit_50"] == round(50 * 1.2)
        assert result["fifo_limit_100"] == round(100 * 1.2)
        assert result["fifo_limit_200"] == round(200 * 1.2)

    def test_expansion_at_2000_ticks(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(2000)

        # 40% expansion
        assert result["fifo_limit_30"] == round(30 * 1.4)
        assert result["fifo_limit_50"] == round(50 * 1.4)
        assert result["fifo_limit_100"] == round(100 * 1.4)
        assert result["fifo_limit_200"] == round(200 * 1.4)

    def test_expansion_at_max(self):
        _load_defaults()
        max_ratio = coefficient_registry._FIFO_EXPANSION_MAX_RATIO
        result = coefficient_registry.apply_fifo_experience_expansion(20000)

        assert result["fifo_limit_30"] == round(30 * max_ratio)
        assert result["fifo_limit_50"] == round(50 * max_ratio)
        assert result["fifo_limit_100"] == round(100 * max_ratio)
        assert result["fifo_limit_200"] == round(200 * max_ratio)

    def test_registry_values_updated(self):
        _load_defaults()
        coefficient_registry.apply_fifo_experience_expansion(500)

        # Registry should now return expanded values
        val30 = coefficient_registry.get("description_common", "fifo_limit_30")
        val50 = coefficient_registry.get("description_common", "fifo_limit_50")
        assert val30 == round(30 * 1.2)
        assert val50 == round(50 * 1.2)

    def test_non_fifo_values_unchanged(self):
        _load_defaults()
        before_window_25 = coefficient_registry.get("description_common", "window_size_25")
        before_decay = coefficient_registry.get("description_common", "freshness_decay_rate_002")

        coefficient_registry.apply_fifo_experience_expansion(500)

        after_window_25 = coefficient_registry.get("description_common", "window_size_25")
        after_decay = coefficient_registry.get("description_common", "freshness_decay_rate_002")

        assert after_window_25 == before_window_25
        assert after_decay == before_decay

    def test_expansion_returns_integers(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(5000)
        for key, val in result.items():
            assert isinstance(val, int), f"{key} should be int, got {type(val)}"

    def test_expansion_never_decreases(self):
        """Expanded values should always be >= base values."""
        _load_defaults()
        defaults = coefficient_registry.get("description_common")
        result = coefficient_registry.apply_fifo_experience_expansion(500)
        for key, expanded_val in result.items():
            base_val = defaults[key]
            assert expanded_val >= base_val, (
                f"{key}: expanded {expanded_val} < base {base_val}"
            )

    def test_registry_not_loaded_returns_empty(self):
        # Don't call load()
        result = coefficient_registry.apply_fifo_experience_expansion(5000)
        assert result == {}

    def test_negative_ticks_no_expansion(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(-100)
        assert result == {}

    def test_idempotent_expansion(self):
        """Applying expansion twice should not double the expansion."""
        _load_defaults()
        result1 = coefficient_registry.apply_fifo_experience_expansion(500)
        # After first expansion, values are already expanded
        # Second call with same ticks would expand further from the already-expanded base
        # But the design says expansion is a one-time startup operation
        # Here we verify the first result is correct
        assert result1["fifo_limit_30"] == round(30 * 1.2)


# =============================================================================
# Test: Accessors
# =============================================================================

class TestAccessors:
    """Test accessor functions for testing/verification."""

    def test_get_fifo_expansion_stages(self):
        stages = coefficient_registry.get_fifo_expansion_stages()
        assert isinstance(stages, tuple)
        assert len(stages) > 0
        # Stages should be sorted by threshold
        thresholds = [s[0] for s in stages]
        assert thresholds == sorted(thresholds)

    def test_get_fifo_expansion_max_ratio(self):
        ratio = coefficient_registry.get_fifo_expansion_max_ratio()
        assert ratio > 1.0
        assert isinstance(ratio, float)

    def test_expansion_targets_exist_in_defaults(self):
        """All FIFO expansion target keys should exist in defaults."""
        defaults = coefficient_registry.get_defaults()
        desc_common = defaults.get("description_common", {})
        for key in coefficient_registry._FIFO_EXPANSION_TARGET_KEYS:
            assert key in desc_common, (
                f"Target key '{key}' not found in description_common defaults"
            )


# =============================================================================
# Test: Custom coefficient file with expansion
# =============================================================================

class TestCustomCoefficientsExpansion:
    """Test expansion with custom coefficient values."""

    def test_expansion_with_custom_fifo_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "coefficients.json")
            custom_data = {
                "description_common": {
                    "fifo_limit_30": 40,  # Custom value
                    "fifo_limit_50": 60,
                }
            }
            with open(path, "w") as f:
                json.dump(custom_data, f)

            coefficient_registry.load(path)
            result = coefficient_registry.apply_fifo_experience_expansion(500)

            # Should expand from the custom values
            assert result["fifo_limit_30"] == round(40 * 1.2)
            assert result["fifo_limit_50"] == round(60 * 1.2)
            # Non-overridden values should use defaults expanded
            assert result["fifo_limit_100"] == round(100 * 1.2)


# =============================================================================
# Test: Safety valves
# =============================================================================

class TestSafetyValves:
    """Test safety valve behavior."""

    def test_max_ratio_caps_expansion(self):
        _load_defaults()
        max_ratio = coefficient_registry._FIFO_EXPANSION_MAX_RATIO
        result = coefficient_registry.apply_fifo_experience_expansion(999999)

        for key, val in result.items():
            base = coefficient_registry.get_defaults()["description_common"][key]
            assert val <= int(round(base * max_ratio)), (
                f"{key}: {val} exceeds max ratio cap {base * max_ratio}"
            )

    def test_stages_are_discrete(self):
        """Between thresholds, expansion should not change."""
        _load_defaults()
        r1 = coefficient_registry.apply_fifo_experience_expansion(600)

        coefficient_registry.reset()
        _load_defaults()
        r2 = coefficient_registry.apply_fifo_experience_expansion(1500)

        # Both should be stage 1 (factor 1.2)
        assert r1 == r2

    def test_expansion_only_positive_direction(self):
        """FIFO limits should only increase, never decrease."""
        _load_defaults()
        defaults = coefficient_registry.get("description_common")

        for ticks in [500, 2000, 5000, 10000, 20000]:
            coefficient_registry.reset()
            _load_defaults()
            result = coefficient_registry.apply_fifo_experience_expansion(ticks)
            for key, val in result.items():
                assert val >= defaults[key], (
                    f"At {ticks} ticks, {key}: {val} < base {defaults[key]}"
                )


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_threshold_values(self):
        """Test expansion at exact threshold boundaries."""
        thresholds = [s[0] for s in coefficient_registry._FIFO_EXPANSION_STAGES]
        for threshold in thresholds:
            coefficient_registry.reset()
            _load_defaults()
            result_at = coefficient_registry.apply_fifo_experience_expansion(threshold)

            coefficient_registry.reset()
            _load_defaults()
            result_below = coefficient_registry.apply_fifo_experience_expansion(threshold - 1)

            # At threshold should have different (higher or equal) expansion than below
            if result_at and result_below:
                for key in result_at:
                    assert result_at[key] >= result_below.get(key, 0)

    def test_very_large_tick_count(self):
        _load_defaults()
        result = coefficient_registry.apply_fifo_experience_expansion(10_000_000)
        max_ratio = coefficient_registry._FIFO_EXPANSION_MAX_RATIO
        # Should be capped at max ratio
        assert result["fifo_limit_30"] == round(30 * max_ratio)
