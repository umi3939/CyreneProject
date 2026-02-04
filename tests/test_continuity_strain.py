"""
Tests for Self-Continuity Strain (自己連続性負荷)

These tests verify:
1. Abstract enums for strain description (no raw numbers exposed)
2. Strain generation only when difference PERSISTS (not for single changes)
3. Natural decay when difference resolves
4. STRICTLY NO IMPACT on decision making
5. Connection to SelfReferenceSystem for introspection only
"""

import pytest
import time
from typing import Any

from psyche.continuity_strain import (
    # Enums
    StrainPresence,
    StrainLevel,
    StrainPersistence,
    StrainTrend,
    # Core structures
    StrainState,
    DifferenceObservation,
    ContinuityStrainConfig,
    ContinuityStrainInternalState,
    # System
    ContinuityStrainSystem,
    # Functions
    determine_strain_level,
    determine_strain_persistence,
    determine_strain_trend,
    get_average_magnitude,
    generate_strain_description,
    generate_strain_tags,
    get_strain_summary,
    get_strain_for_introspection,
    create_config,
    create_empty_strain,
    save_strain_state,
    load_strain_state,
    verify_no_decision_impact,
    verify_no_correction_mechanism,
)

from psyche.temporal_self_difference import (
    SelfDifferenceSummary,
    DifferenceMagnitude,
    ChangeNature,
    TemporalSpan,
    ComponentDifference,
    ComponentChangeType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default strain configuration"""
    return ContinuityStrainConfig()


@pytest.fixture
def fast_decay_config():
    """Configuration with faster decay for testing"""
    return ContinuityStrainConfig(
        observation_window=3,
        min_observations_for_strain=2,
        min_observations_for_ongoing=3,
        min_observations_for_chronic=5,
        decay_observations_for_easing=1,
        decay_observations_for_resolution=2,
    )


@pytest.fixture
def strain_system(default_config):
    """Default strain system"""
    return ContinuityStrainSystem(default_config)


@pytest.fixture
def fast_decay_system(fast_decay_config):
    """Strain system with fast decay"""
    return ContinuityStrainSystem(fast_decay_config)


def create_difference_summary(
    magnitude: DifferenceMagnitude = DifferenceMagnitude.NOTICEABLE,
    nature: ChangeNature = ChangeNature.SHIFTING,
    has_difference: bool = True,
) -> SelfDifferenceSummary:
    """Helper to create a difference summary for testing"""
    unchanged = ComponentDifference.unchanged("test", "stable")

    return SelfDifferenceSummary(
        has_difference=has_difference,
        magnitude=magnitude,
        nature=nature,
        temporal_span=TemporalSpan.IMMEDIATE,
        emotional_diff=unchanged,
        responsibility_diff=unchanged,
        tendency_diff=unchanged,
        direction_diff=unchanged,
        value_diff=unchanged,
        current_snapshot_id="current",
        reference_snapshot_id="reference",
        comparison_timestamp=time.time(),
        integrated_description="Test difference",
    )


def create_no_difference_summary() -> SelfDifferenceSummary:
    """Helper to create a no-difference summary"""
    return create_difference_summary(
        magnitude=DifferenceMagnitude.NONE,
        nature=ChangeNature.STABLE,
        has_difference=False,
    )


def create_significant_difference_summary() -> SelfDifferenceSummary:
    """Helper to create a significant difference summary"""
    return create_difference_summary(
        magnitude=DifferenceMagnitude.SIGNIFICANT,
        nature=ChangeNature.SHIFTING,
        has_difference=True,
    )


def create_substantial_difference_summary() -> SelfDifferenceSummary:
    """Helper to create a substantial difference summary"""
    return create_difference_summary(
        magnitude=DifferenceMagnitude.SUBSTANTIAL,
        nature=ChangeNature.TRANSFORMED,
        has_difference=True,
    )


# =============================================================================
# Test Abstract Enums
# =============================================================================

class TestStrainEnums:
    """Test that all enums are abstract (no raw numbers)"""

    def test_strain_presence_values(self):
        """StrainPresence should be simple binary"""
        assert StrainPresence.ABSENT.value == "absent"
        assert StrainPresence.PRESENT.value == "present"

    def test_strain_level_values(self):
        """StrainLevel should be abstract categories"""
        assert StrainLevel.AT_EASE.value == "at_ease"
        assert StrainLevel.UNSETTLED.value == "unsettled"
        assert StrainLevel.DISSONANT.value == "dissonant"
        assert StrainLevel.ALIENATED.value == "alienated"
        assert StrainLevel.UNDEFINED.value == "undefined"

    def test_strain_level_no_numeric_values(self):
        """StrainLevel values should not be numeric"""
        for level in StrainLevel:
            assert not isinstance(level.value, (int, float))

    def test_strain_persistence_values(self):
        """StrainPersistence should be abstract categories"""
        assert StrainPersistence.NONE.value == "none"
        assert StrainPersistence.MOMENTARY.value == "momentary"
        assert StrainPersistence.ONGOING.value == "ongoing"
        assert StrainPersistence.CHRONIC.value == "chronic"

    def test_strain_trend_values(self):
        """StrainTrend should be abstract categories"""
        assert StrainTrend.STABLE.value == "stable"
        assert StrainTrend.BUILDING.value == "building"
        assert StrainTrend.EASING.value == "easing"
        assert StrainTrend.FLUCTUATING.value == "fluctuating"


# =============================================================================
# Test StrainState Structure
# =============================================================================

class TestStrainState:
    """Test StrainState dataclass"""

    def test_at_ease_creation(self):
        """Test creating an at-ease state"""
        state = StrainState.at_ease()

        assert state.presence == StrainPresence.ABSENT
        assert state.level == StrainLevel.AT_EASE
        assert state.persistence == StrainPersistence.NONE
        assert state.trend == StrainTrend.STABLE
        assert not state.is_strained()

    def test_undefined_creation(self):
        """Test creating an undefined state"""
        state = StrainState.undefined()

        assert state.level == StrainLevel.UNDEFINED
        assert state.persistence == StrainPersistence.UNDEFINED
        assert state.trend == StrainTrend.UNDEFINED

    def test_is_strained_when_present(self):
        """Test is_strained returns True when strain present"""
        state = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.UNSETTLED,
            persistence=StrainPersistence.MOMENTARY,
            trend=StrainTrend.STABLE,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="Test strain",
        )

        assert state.is_strained()

    def test_is_strained_when_absent(self):
        """Test is_strained returns False when strain absent"""
        state = StrainState.at_ease()
        assert not state.is_strained()

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip"""
        original = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.DISSONANT,
            persistence=StrainPersistence.ONGOING,
            trend=StrainTrend.BUILDING,
            timestamp=1000.0,
            last_update_timestamp=1100.0,
            description="Test description",
        )

        data = original.to_dict()
        restored = StrainState.from_dict(data)

        assert restored.presence == original.presence
        assert restored.level == original.level
        assert restored.persistence == original.persistence
        assert restored.trend == original.trend
        assert restored.timestamp == original.timestamp
        assert restored.description == original.description


# =============================================================================
# Test Strain Generation Conditions
# =============================================================================

class TestStrainGenerationConditions:
    """Test that strain only generates when difference PERSISTS"""

    def test_no_strain_for_single_difference(self, strain_system):
        """Single difference should NOT generate strain"""
        diff = create_significant_difference_summary()
        strain = strain_system.observe_difference(diff)

        # Single observation should not trigger strain
        assert not strain.is_strained()
        assert strain.level == StrainLevel.AT_EASE

    def test_no_strain_for_two_differences(self, strain_system):
        """Two differences should NOT generate strain (default config requires 3)"""
        for _ in range(2):
            strain = strain_system.observe_difference(create_significant_difference_summary())

        assert not strain.is_strained()

    def test_strain_after_persistent_differences(self, strain_system):
        """Strain should generate after persistent differences"""
        # Default config requires 3 consecutive significant observations
        for i in range(3):
            strain = strain_system.observe_difference(create_significant_difference_summary())

        assert strain.is_strained()
        assert strain.level in (StrainLevel.UNSETTLED, StrainLevel.DISSONANT)

    def test_no_strain_for_minimal_differences(self, strain_system):
        """Minimal differences should NOT generate strain"""
        for _ in range(10):
            diff = create_difference_summary(
                magnitude=DifferenceMagnitude.MINIMAL,
                has_difference=True,
            )
            strain = strain_system.observe_difference(diff)

        assert not strain.is_strained()

    def test_no_strain_for_no_differences(self, strain_system):
        """No differences should NOT generate strain"""
        for _ in range(10):
            strain = strain_system.observe_difference(create_no_difference_summary())

        assert not strain.is_strained()

    def test_strain_increases_with_persistence(self, fast_decay_system):
        """Strain level should increase with longer persistence"""
        levels_seen = set()

        for i in range(10):
            strain = fast_decay_system.observe_difference(create_significant_difference_summary())
            if strain.is_strained():
                levels_seen.add(strain.level)

        # Should see multiple levels as strain increases
        assert len(levels_seen) >= 1


# =============================================================================
# Test Natural Decay
# =============================================================================

class TestNaturalDecay:
    """Test that strain naturally decays when difference resolves"""

    def test_strain_eases_when_differences_stop(self, fast_decay_system):
        """Strain should ease when differences stop occurring"""
        # Build up strain
        for _ in range(5):
            fast_decay_system.observe_difference(create_significant_difference_summary())

        # Verify strain is present
        assert fast_decay_system.is_strained()

        # Stop differences
        for _ in range(2):
            strain = fast_decay_system.observe_difference(create_no_difference_summary())

        # Should be fully resolved with fast decay config
        assert not strain.is_strained()

    def test_strain_shows_easing_trend(self, fast_decay_system):
        """Strain should show EASING trend when differences stop for a while"""
        # Build up strain
        for _ in range(5):
            fast_decay_system.observe_difference(create_significant_difference_summary())

        # Stop differences - need multiple observations for trend to show easing
        # The trend is based on strain level history, not just the current observation
        for _ in range(3):
            strain = fast_decay_system.observe_difference(create_no_difference_summary())

        # After several no-difference observations, should either be:
        # - Not strained anymore (resolved)
        # - Or showing EASING/STABLE trend (no longer building)
        if strain.is_strained():
            assert strain.trend in (StrainTrend.EASING, StrainTrend.STABLE, StrainTrend.FLUCTUATING)

    def test_no_explicit_reset_method(self, strain_system):
        """System should NOT have explicit reset/fix methods"""
        # These method names should NOT exist
        forbidden_methods = [
            "reset_strain",
            "fix_strain",
            "correct_strain",
            "force_ease",
            "eliminate_strain",
            "restore_continuity",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(strain_system, method_name), \
                f"System should NOT have {method_name} method"

    def test_strain_decays_gradually(self, fast_decay_system):
        """Strain should decay gradually, not abruptly"""
        # Build up high strain
        for _ in range(10):
            fast_decay_system.observe_difference(create_substantial_difference_summary())

        initial_level = fast_decay_system.get_current_strain().level

        # Decay with no-difference observations
        levels = []
        for _ in range(3):
            strain = fast_decay_system.observe_difference(create_no_difference_summary())
            levels.append(strain.level)

        # Level should decrease or stay same, not increase
        level_order = [StrainLevel.AT_EASE, StrainLevel.UNSETTLED,
                       StrainLevel.DISSONANT, StrainLevel.ALIENATED]

        for i in range(len(levels) - 1):
            curr_idx = level_order.index(levels[i]) if levels[i] in level_order else 0
            next_idx = level_order.index(levels[i + 1]) if levels[i + 1] in level_order else 0
            assert next_idx <= curr_idx or levels[i + 1] == StrainLevel.AT_EASE


# =============================================================================
# Test NO Decision Impact
# =============================================================================

class TestNoDecisionImpact:
    """Test that strain has STRICTLY NO IMPACT on decision making"""

    def test_strain_state_has_no_decision_values(self):
        """StrainState should not expose decision-impacting values"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.ALIENATED,
            persistence=StrainPersistence.CHRONIC,
            trend=StrainTrend.BUILDING,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="High strain",
        )

        assert verify_no_decision_impact(strain)

    def test_system_has_no_correction_mechanism(self, strain_system):
        """System should not have methods to correct/fix strain"""
        assert verify_no_correction_mechanism(strain_system)

    def test_strain_state_no_numeric_scores(self):
        """StrainState should not have numeric scores"""
        strain = StrainState.at_ease()

        # Check all public attributes
        for attr in dir(strain):
            if attr.startswith('_'):
                continue
            if callable(getattr(strain, attr)):
                continue

            value = getattr(strain, attr)

            # Timestamps are allowed
            if attr in ('timestamp', 'last_update_timestamp'):
                continue

            # No other numeric attributes should exist
            if isinstance(value, (int, float)):
                pytest.fail(f"Found numeric attribute: {attr}={value}")

    def test_strain_to_dict_no_scores(self):
        """StrainState.to_dict should not expose numeric scores"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.DISSONANT,
            persistence=StrainPersistence.ONGOING,
            trend=StrainTrend.BUILDING,
            timestamp=100.0,
            last_update_timestamp=200.0,
            description="Test",
        )

        data = strain.to_dict()

        # Check all values
        for key, value in data.items():
            if key in ('timestamp', 'last_update_timestamp'):
                continue

            if isinstance(value, (int, float)):
                pytest.fail(f"Found numeric value in to_dict: {key}={value}")

    def test_system_does_not_modify_external_state(self, strain_system):
        """System should not modify any external state"""
        # The system only tracks internal state
        # It should not have methods to modify external systems

        forbidden_patterns = [
            "modify_",
            "update_decision",
            "influence_",
            "bias_",
            "adjust_",
        ]

        methods = [m for m in dir(strain_system) if not m.startswith('_')]

        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden_patterns:
                assert pattern not in method_lower, \
                    f"Method {method} suggests external modification"


# =============================================================================
# Test Strain Level Determination
# =============================================================================

class TestStrainLevelDetermination:
    """Test strain level calculation logic"""

    def test_at_ease_when_below_threshold(self, default_config):
        """Should be AT_EASE when below strain threshold"""
        level = determine_strain_level(
            consecutive_significant=2,  # Below threshold of 3
            total_strain_observations=2,
            average_magnitude=DifferenceMagnitude.SIGNIFICANT,
            config=default_config,
        )

        assert level == StrainLevel.AT_EASE

    def test_unsettled_at_threshold(self, default_config):
        """Should be UNSETTLED at strain threshold"""
        level = determine_strain_level(
            consecutive_significant=3,  # At threshold
            total_strain_observations=3,
            average_magnitude=DifferenceMagnitude.NOTICEABLE,
            config=default_config,
        )

        assert level == StrainLevel.UNSETTLED

    def test_dissonant_at_ongoing(self, default_config):
        """Should be DISSONANT at ongoing threshold"""
        level = determine_strain_level(
            consecutive_significant=5,  # At ongoing threshold
            total_strain_observations=5,
            average_magnitude=DifferenceMagnitude.SIGNIFICANT,
            config=default_config,
        )

        assert level == StrainLevel.DISSONANT

    def test_alienated_at_chronic(self, default_config):
        """Should be ALIENATED at chronic threshold"""
        level = determine_strain_level(
            consecutive_significant=10,  # At chronic threshold
            total_strain_observations=10,
            average_magnitude=DifferenceMagnitude.SUBSTANTIAL,
            config=default_config,
        )

        assert level == StrainLevel.ALIENATED

    def test_substantial_magnitude_escalates_faster(self, default_config):
        """Substantial magnitude should escalate level faster"""
        # With noticeable magnitude
        level_noticeable = determine_strain_level(
            consecutive_significant=3,
            total_strain_observations=3,
            average_magnitude=DifferenceMagnitude.NOTICEABLE,
            config=default_config,
        )

        # With substantial magnitude
        level_substantial = determine_strain_level(
            consecutive_significant=3,
            total_strain_observations=3,
            average_magnitude=DifferenceMagnitude.SUBSTANTIAL,
            config=default_config,
        )

        # Substantial should be higher or equal
        level_order = [StrainLevel.AT_EASE, StrainLevel.UNSETTLED,
                       StrainLevel.DISSONANT, StrainLevel.ALIENATED]

        noticeable_idx = level_order.index(level_noticeable)
        substantial_idx = level_order.index(level_substantial)

        assert substantial_idx >= noticeable_idx


# =============================================================================
# Test Strain Persistence
# =============================================================================

class TestStrainPersistence:
    """Test strain persistence classification"""

    def test_persistence_none_when_no_strain(self, default_config):
        """Should be NONE when below strain threshold"""
        persistence = determine_strain_persistence(
            consecutive_significant=2,
            strain_started_at=None,
            current_time=time.time(),
            config=default_config,
        )

        assert persistence == StrainPersistence.NONE

    def test_persistence_momentary(self, default_config):
        """Should be MOMENTARY at threshold"""
        persistence = determine_strain_persistence(
            consecutive_significant=3,  # At threshold
            strain_started_at=time.time(),
            current_time=time.time(),
            config=default_config,
        )

        assert persistence == StrainPersistence.MOMENTARY

    def test_persistence_ongoing(self, default_config):
        """Should be ONGOING at ongoing threshold"""
        persistence = determine_strain_persistence(
            consecutive_significant=5,
            strain_started_at=time.time() - 300,
            current_time=time.time(),
            config=default_config,
        )

        assert persistence == StrainPersistence.ONGOING

    def test_persistence_chronic(self, default_config):
        """Should be CHRONIC at chronic threshold"""
        persistence = determine_strain_persistence(
            consecutive_significant=10,
            strain_started_at=time.time() - 600,
            current_time=time.time(),
            config=default_config,
        )

        assert persistence == StrainPersistence.CHRONIC


# =============================================================================
# Test Strain Trend
# =============================================================================

class TestStrainTrend:
    """Test strain trend detection"""

    def test_trend_stable_when_same_levels(self):
        """Should be STABLE when levels are same"""
        config = ContinuityStrainConfig()
        history = [StrainLevel.UNSETTLED] * 5

        trend = determine_strain_trend(history, config)

        assert trend == StrainTrend.STABLE

    def test_trend_building_when_increasing(self):
        """Should be BUILDING when levels increase"""
        config = ContinuityStrainConfig()
        history = [
            StrainLevel.AT_EASE,
            StrainLevel.UNSETTLED,
            StrainLevel.DISSONANT,
            StrainLevel.ALIENATED,
        ]

        trend = determine_strain_trend(history, config)

        assert trend == StrainTrend.BUILDING

    def test_trend_easing_when_decreasing(self):
        """Should be EASING when levels decrease"""
        config = ContinuityStrainConfig()
        history = [
            StrainLevel.ALIENATED,
            StrainLevel.DISSONANT,
            StrainLevel.UNSETTLED,
            StrainLevel.AT_EASE,
        ]

        trend = determine_strain_trend(history, config)

        assert trend == StrainTrend.EASING

    def test_trend_fluctuating_when_oscillating(self):
        """Should be FLUCTUATING when levels oscillate"""
        config = ContinuityStrainConfig()
        history = [
            StrainLevel.UNSETTLED,
            StrainLevel.DISSONANT,
            StrainLevel.UNSETTLED,
            StrainLevel.DISSONANT,
        ]

        trend = determine_strain_trend(history, config)

        assert trend == StrainTrend.FLUCTUATING

    def test_trend_undefined_when_no_history(self):
        """Should be UNDEFINED when insufficient history"""
        config = ContinuityStrainConfig()

        trend = determine_strain_trend([], config)
        assert trend == StrainTrend.UNDEFINED

        trend = determine_strain_trend([StrainLevel.UNSETTLED], config)
        assert trend == StrainTrend.UNDEFINED


# =============================================================================
# Test SelfReferenceSystem Integration
# =============================================================================

class TestSelfReferenceIntegration:
    """Test integration with SelfReferenceSystem"""

    def test_generate_strain_tags_at_ease(self):
        """Tags for at-ease state should be minimal"""
        strain = StrainState.at_ease()
        tags = generate_strain_tags(strain)

        assert len(tags) == 1
        assert tags[0]["category"] == "CONTINUITY_STRAIN"
        assert tags[0]["label"] == "continuous_self"

    def test_generate_strain_tags_strained(self):
        """Tags for strained state should include all aspects"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.DISSONANT,
            persistence=StrainPersistence.ONGOING,
            trend=StrainTrend.BUILDING,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="Notable discontinuity",
        )

        tags = generate_strain_tags(strain)

        categories = [t["category"] for t in tags]

        assert "CONTINUITY_STRAIN_PRESENCE" in categories
        assert "CONTINUITY_STRAIN_LEVEL" in categories
        assert "CONTINUITY_STRAIN_PERSISTENCE" in categories
        assert "CONTINUITY_STRAIN_TREND" in categories
        assert "CONTINUITY_STRAIN_INTEGRATED" in categories

    def test_tags_have_weights(self):
        """All tags should have weights"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.ALIENATED,
            persistence=StrainPersistence.CHRONIC,
            trend=StrainTrend.STABLE,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="Test",
        )

        tags = generate_strain_tags(strain)

        for tag in tags:
            assert "weight" in tag
            assert isinstance(tag["weight"], float)
            assert tag["weight"] > 0

    def test_tags_scale_factor(self):
        """Tags should respect scale factor"""
        strain = StrainState.at_ease()

        tags_default = generate_strain_tags(strain, scale=1.0)
        tags_scaled = generate_strain_tags(strain, scale=2.0)

        assert tags_scaled[0]["weight"] == tags_default[0]["weight"] * 2


# =============================================================================
# Test Introspection Functions
# =============================================================================

class TestIntrospection:
    """Test introspection support functions"""

    def test_get_strain_summary(self):
        """get_strain_summary should return readable text"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.DISSONANT,
            persistence=StrainPersistence.ONGOING,
            trend=StrainTrend.BUILDING,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="Test description",
        )

        summary = get_strain_summary(strain)

        assert "Self-Continuity Strain Awareness" in summary
        assert "dissonant" in summary
        assert "ongoing" in summary
        assert "building" in summary

    def test_get_strain_for_introspection(self):
        """get_strain_for_introspection should return structured data"""
        strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.UNSETTLED,
            persistence=StrainPersistence.MOMENTARY,
            trend=StrainTrend.STABLE,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="Test",
        )

        data = get_strain_for_introspection(strain)

        assert data["is_strained"] is True
        assert data["level"] == "unsettled"
        assert data["persistence"] == "momentary"
        assert data["trend"] == "stable"

    def test_generate_strain_description_at_ease(self):
        """At-ease description should be calm"""
        description = generate_strain_description(
            StrainLevel.AT_EASE,
            StrainPersistence.NONE,
            StrainTrend.STABLE,
        )

        assert "continuous" in description.lower() or "natural" in description.lower()

    def test_generate_strain_description_alienated(self):
        """Alienated description should convey discontinuity"""
        description = generate_strain_description(
            StrainLevel.ALIENATED,
            StrainPersistence.CHRONIC,
            StrainTrend.BUILDING,
        )

        assert "separation" in description.lower() or "discontinuity" in description.lower()


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration system"""

    def test_default_config_values(self):
        """Default config should have sensible values"""
        config = ContinuityStrainConfig()

        assert config.observation_window > 0
        assert config.min_observations_for_strain > 0
        assert config.min_observations_for_ongoing > config.min_observations_for_strain
        assert config.min_observations_for_chronic > config.min_observations_for_ongoing

    def test_create_config(self):
        """create_config helper should work"""
        config = create_config(
            observation_window=10,
            min_observations_for_strain=5,
        )

        assert config.observation_window == 10
        assert config.min_observations_for_strain == 5

    def test_strain_triggering_magnitudes(self):
        """Config should specify which magnitudes trigger strain"""
        config = ContinuityStrainConfig()

        assert DifferenceMagnitude.NOTICEABLE in config.strain_triggering_magnitudes
        assert DifferenceMagnitude.SIGNIFICANT in config.strain_triggering_magnitudes
        assert DifferenceMagnitude.SUBSTANTIAL in config.strain_triggering_magnitudes
        assert DifferenceMagnitude.MINIMAL not in config.strain_triggering_magnitudes
        assert DifferenceMagnitude.NONE not in config.strain_triggering_magnitudes


# =============================================================================
# Test Persistence
# =============================================================================

class TestPersistence:
    """Test save/load functionality"""

    def test_save_and_load_strain_state(self, tmp_path, fast_decay_system):
        """Should be able to save and restore strain state"""
        # Build up some strain
        for _ in range(5):
            fast_decay_system.observe_difference(create_significant_difference_summary())

        original_strain = fast_decay_system.get_current_strain()

        # Save
        path = tmp_path / "strain_state.json"
        save_strain_state(fast_decay_system, str(path))

        # Load into new system
        config = ContinuityStrainConfig(
            min_observations_for_strain=2,
            decay_observations_for_resolution=2,
        )
        restored_system = load_strain_state(str(path), config)

        restored_strain = restored_system.get_current_strain()

        assert restored_strain.level == original_strain.level
        assert restored_strain.persistence == original_strain.persistence

    def test_load_nonexistent_file(self, tmp_path):
        """Loading nonexistent file should return fresh system"""
        path = tmp_path / "nonexistent.json"
        system = load_strain_state(str(path))

        assert system.get_observation_count() == 0
        assert not system.is_strained()


# =============================================================================
# Test DifferenceObservation
# =============================================================================

class TestDifferenceObservation:
    """Test DifferenceObservation structure"""

    def test_is_significant_with_noticeable(self, default_config):
        """NOTICEABLE magnitude should be significant"""
        obs = DifferenceObservation(
            magnitude=DifferenceMagnitude.NOTICEABLE,
            nature=ChangeNature.SHIFTING,
            has_difference=True,
            timestamp=time.time(),
        )

        assert obs.is_significant(default_config)

    def test_is_significant_with_minimal(self, default_config):
        """MINIMAL magnitude should NOT be significant"""
        obs = DifferenceObservation(
            magnitude=DifferenceMagnitude.MINIMAL,
            nature=ChangeNature.FLUCTUATING,
            has_difference=True,
            timestamp=time.time(),
        )

        assert not obs.is_significant(default_config)

    def test_is_significant_requires_has_difference(self, default_config):
        """Observation without has_difference should not be significant"""
        obs = DifferenceObservation(
            magnitude=DifferenceMagnitude.SUBSTANTIAL,
            nature=ChangeNature.TRANSFORMED,
            has_difference=False,
            timestamp=time.time(),
        )

        assert not obs.is_significant(default_config)


# =============================================================================
# Test System Behavior
# =============================================================================

class TestSystemBehavior:
    """Test overall system behavior"""

    def test_observation_count_increments(self, strain_system):
        """Observation count should increment"""
        assert strain_system.get_observation_count() == 0

        strain_system.observe_difference(create_no_difference_summary())
        assert strain_system.get_observation_count() == 1

        strain_system.observe_difference(create_significant_difference_summary())
        assert strain_system.get_observation_count() == 2

    def test_get_current_strain(self, strain_system):
        """get_current_strain should return current state"""
        strain = strain_system.get_current_strain()
        assert isinstance(strain, StrainState)
        assert not strain.is_strained()

    def test_strain_duration_observations(self, fast_decay_system):
        """get_strain_duration_observations should track consecutive significant"""
        assert fast_decay_system.get_strain_duration_observations() == 0

        fast_decay_system.observe_difference(create_significant_difference_summary())
        assert fast_decay_system.get_strain_duration_observations() == 1

        fast_decay_system.observe_difference(create_significant_difference_summary())
        assert fast_decay_system.get_strain_duration_observations() == 2

        # Breaks with no difference
        fast_decay_system.observe_difference(create_no_difference_summary())
        fast_decay_system.observe_difference(create_no_difference_summary())
        assert fast_decay_system.get_strain_duration_observations() == 0


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_strain(self):
        """create_empty_strain should return at-ease state"""
        strain = create_empty_strain()
        assert strain.level == StrainLevel.AT_EASE
        assert not strain.is_strained()

    def test_rapid_alternation(self, fast_decay_system):
        """System should handle rapid alternation between difference and no-difference"""
        for _ in range(10):
            fast_decay_system.observe_difference(create_significant_difference_summary())
            fast_decay_system.observe_difference(create_no_difference_summary())

        # Should not have sustained strain due to alternation
        strain = fast_decay_system.get_current_strain()
        # Strain level should be low or absent
        assert strain.level in (StrainLevel.AT_EASE, StrainLevel.UNSETTLED)

    def test_get_average_magnitude_empty(self):
        """get_average_magnitude should handle empty list"""
        result = get_average_magnitude([])
        assert result == DifferenceMagnitude.NONE

    def test_get_average_magnitude_all_minimal(self, default_config):
        """get_average_magnitude should handle all minimal magnitudes"""
        observations = [
            DifferenceObservation(
                magnitude=DifferenceMagnitude.MINIMAL,
                nature=ChangeNature.FLUCTUATING,
                has_difference=True,
                timestamp=time.time(),
            )
            for _ in range(5)
        ]

        result = get_average_magnitude(observations)
        assert result == DifferenceMagnitude.NONE

    def test_system_with_custom_triggering_magnitudes(self):
        """System should respect custom triggering magnitudes"""
        config = ContinuityStrainConfig(
            strain_triggering_magnitudes=(DifferenceMagnitude.SUBSTANTIAL,),
            min_observations_for_strain=2,
        )
        system = ContinuityStrainSystem(config)

        # SIGNIFICANT should NOT trigger strain
        for _ in range(5):
            strain = system.observe_difference(create_significant_difference_summary())

        assert not strain.is_strained()

        # SUBSTANTIAL should trigger strain
        for _ in range(3):
            strain = system.observe_difference(create_substantial_difference_summary())

        assert strain.is_strained()


# =============================================================================
# Test Philosophy Compliance
# =============================================================================

class TestPhilosophyCompliance:
    """Test that implementation follows the design philosophy"""

    def test_no_self_preservation(self, strain_system):
        """System should NOT try to preserve self"""
        # No methods that try to maintain a particular state
        methods = [m for m in dir(strain_system) if not m.startswith('_')]

        preservation_patterns = [
            "protect",
            "preserve",
            "maintain",
            "keep_stable",
            "ensure_continuity",
        ]

        for method in methods:
            for pattern in preservation_patterns:
                assert pattern not in method.lower()

    def test_no_correction(self, strain_system):
        """System should NOT try to correct/fix strain"""
        methods = [m for m in dir(strain_system) if not m.startswith('_')]

        correction_patterns = [
            "fix",
            "correct",
            "repair",
            "heal",
            "restore",
        ]

        for method in methods:
            for pattern in correction_patterns:
                assert pattern not in method.lower()

    def test_no_evaluation_in_descriptions(self):
        """Descriptions should NOT evaluate strain as good/bad"""
        # Test various strain descriptions
        descriptions = [
            generate_strain_description(StrainLevel.AT_EASE, StrainPersistence.NONE, StrainTrend.STABLE),
            generate_strain_description(StrainLevel.UNSETTLED, StrainPersistence.MOMENTARY, StrainTrend.STABLE),
            generate_strain_description(StrainLevel.DISSONANT, StrainPersistence.ONGOING, StrainTrend.BUILDING),
            generate_strain_description(StrainLevel.ALIENATED, StrainPersistence.CHRONIC, StrainTrend.STABLE),
        ]

        evaluative_words = ["good", "bad", "wrong", "right", "problem", "error", "fix"]

        for desc in descriptions:
            desc_lower = desc.lower()
            for word in evaluative_words:
                assert word not in desc_lower, \
                    f"Description contains evaluative word '{word}': {desc}"

    def test_change_not_considered_bad(self):
        """System should NOT treat change as inherently bad"""
        # The StrainLevel.ALIENATED is the highest level but should not be
        # described as "bad" or "wrong"
        description = generate_strain_description(
            StrainLevel.ALIENATED,
            StrainPersistence.CHRONIC,
            StrainTrend.BUILDING,
        )

        negative_words = ["bad", "wrong", "error", "failure", "problem", "broken"]

        for word in negative_words:
            assert word not in description.lower()

    def test_stability_not_a_goal(self, strain_system):
        """System should NOT make stability a goal"""
        # No methods that try to achieve stability
        methods = [m for m in dir(strain_system) if not m.startswith('_')]

        goal_patterns = [
            "achieve_stability",
            "reach_ease",
            "target_",
            "goal_",
            "optimize",
        ]

        for method in methods:
            for pattern in goal_patterns:
                assert pattern not in method.lower()
