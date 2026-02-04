"""
Tests for Temporal Self-Difference Awareness (自己モデル差分認知)

These tests verify:
1. Abstract enums and categorization
2. Component difference detection
3. SelfDifferenceSummary structure
4. TemporalSelfDifferenceSystem
5. NO IMPACT on decision making
6. NO JUDGMENT (good/bad)
7. Natural shrinking of differences when states converge
8. SelfReferenceSystem integration
9. IntrospectionTrace integration
"""

import pytest
import time
from dataclasses import dataclass

from psyche.temporal_self_difference import (
    # Enums
    DifferenceMagnitude,
    ChangeNature,
    ComponentChangeType,
    TemporalSpan,
    # Structures
    ComponentDifference,
    SelfDifferenceSummary,
    TemporalDifferenceConfig,
    TemporalDifferenceState,
    TemporalSelfDifferenceSystem,
    # Comparison functions
    compare_emotional_state,
    compare_responsibility_state,
    compare_tendency_state,
    compare_direction_state,
    compare_value_state,
    determine_magnitude,
    determine_nature,
    # Integration functions
    generate_difference_tags,
    get_difference_summary,
    get_difference_for_introspection,
    # Convenience
    create_config,
    create_empty_summary,
)

from psyche.self_model import (
    SelfStateView,
    EmotionalStateView,
    ResponsibilityStateView,
    TendencyStateView,
    DirectionStateView,
    ValueStateView,
    EmotionalSpread,
    EmotionalIntensity,
    EmotionalHarmony,
    BurdenLevel,
    BurdenDistribution,
    BurdenTrend,
    HabitPresence,
    HabitCharacter,
    DirectionClarity,
    DirectionConvergence,
    ValueStability,
    ValueClarity,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def calm_emotional_view():
    """Create a calm emotional state view"""
    return EmotionalStateView(
        spread=EmotionalSpread.FOCUSED,
        intensity=EmotionalIntensity.CALM,
        harmony=EmotionalHarmony.HARMONIOUS,
        active_emotion_count=2,
        has_coexisting_pairs=False,
        description="Calm emotional state",
    )


@pytest.fixture
def intense_emotional_view():
    """Create an intense emotional state view"""
    return EmotionalStateView(
        spread=EmotionalSpread.DIFFUSE,
        intensity=EmotionalIntensity.INTENSE,
        harmony=EmotionalHarmony.CONFLICTED,
        active_emotion_count=5,
        has_coexisting_pairs=True,
        description="Intense emotional state",
    )


@pytest.fixture
def light_responsibility_view():
    """Create a light responsibility view"""
    return ResponsibilityStateView(
        burden_level=BurdenLevel.LIGHT,
        distribution=BurdenDistribution.DISTRIBUTED,
        trend=BurdenTrend.STABLE,
        has_pending_decisions=False,
        description="Light responsibility",
    )


@pytest.fixture
def heavy_responsibility_view():
    """Create a heavy responsibility view"""
    return ResponsibilityStateView(
        burden_level=BurdenLevel.HEAVY,
        distribution=BurdenDistribution.CONCENTRATED,
        trend=BurdenTrend.ACCUMULATING,
        has_pending_decisions=True,
        description="Heavy responsibility",
    )


@pytest.fixture
def no_habit_tendency_view():
    """Create a no-habit tendency view"""
    return TendencyStateView(
        presence=HabitPresence.NONE,
        character=HabitCharacter.EXPLORATORY,
        tendency_count=0,
        has_strong_habits=False,
        has_fading_habits=False,
        description="No habits",
    )


@pytest.fixture
def established_tendency_view():
    """Create an established tendency view"""
    return TendencyStateView(
        presence=HabitPresence.ESTABLISHED,
        character=HabitCharacter.HABITUAL,
        tendency_count=5,
        has_strong_habits=True,
        has_fading_habits=False,
        description="Established habits",
    )


@pytest.fixture
def unclear_direction_view():
    """Create an unclear direction view"""
    return DirectionStateView(
        clarity=DirectionClarity.UNCLEAR,
        convergence=DirectionConvergence.SCATTERED,
        vector_count=2,
        has_dominant_direction=False,
        description="Unclear direction",
    )


@pytest.fixture
def clear_direction_view():
    """Create a clear direction view"""
    return DirectionStateView(
        clarity=DirectionClarity.CLEAR,
        convergence=DirectionConvergence.CONVERGENT,
        vector_count=3,
        has_dominant_direction=True,
        description="Clear direction",
    )


@pytest.fixture
def unstable_value_view():
    """Create an unstable value view"""
    return ValueStateView(
        stability=ValueStability.UNSTABLE,
        clarity=ValueClarity.EMERGING,
        has_strong_orientations=False,
        is_recently_changed=True,
        description="Unstable values",
    )


@pytest.fixture
def anchored_value_view():
    """Create an anchored value view"""
    return ValueStateView(
        stability=ValueStability.ANCHORED,
        clarity=ValueClarity.DEFINED,
        has_strong_orientations=True,
        is_recently_changed=False,
        description="Anchored values",
    )


def create_self_state_view(
    emotional: EmotionalStateView,
    responsibility: ResponsibilityStateView,
    tendency: TendencyStateView,
    direction: DirectionStateView,
    value: ValueStateView,
    snapshot_id: str = "test_snapshot",
) -> SelfStateView:
    """Helper to create a SelfStateView"""
    return SelfStateView(
        emotional=emotional,
        responsibility=responsibility,
        tendency=tendency,
        direction=direction,
        value=value,
        timestamp=time.time(),
        snapshot_id=snapshot_id,
        is_complete=True,
        integrated_description="Test self state",
    )


@pytest.fixture
def calm_self_state(
    calm_emotional_view,
    light_responsibility_view,
    no_habit_tendency_view,
    unclear_direction_view,
    unstable_value_view,
):
    """Create a calm/light self state"""
    return create_self_state_view(
        calm_emotional_view,
        light_responsibility_view,
        no_habit_tendency_view,
        unclear_direction_view,
        unstable_value_view,
        "calm_state",
    )


@pytest.fixture
def intense_self_state(
    intense_emotional_view,
    heavy_responsibility_view,
    established_tendency_view,
    clear_direction_view,
    anchored_value_view,
):
    """Create an intense/heavy self state"""
    return create_self_state_view(
        intense_emotional_view,
        heavy_responsibility_view,
        established_tendency_view,
        clear_direction_view,
        anchored_value_view,
        "intense_state",
    )


@pytest.fixture
def temporal_system():
    """Create a TemporalSelfDifferenceSystem"""
    return TemporalSelfDifferenceSystem()


# =============================================================================
# Tests for Abstract Enums
# =============================================================================

class TestAbstractEnums:
    """Test that all enums provide proper abstraction"""

    def test_difference_magnitude_values(self):
        """DifferenceMagnitude has expected values"""
        assert DifferenceMagnitude.NONE.value == "none"
        assert DifferenceMagnitude.MINIMAL.value == "minimal"
        assert DifferenceMagnitude.NOTICEABLE.value == "noticeable"
        assert DifferenceMagnitude.SIGNIFICANT.value == "significant"
        assert DifferenceMagnitude.SUBSTANTIAL.value == "substantial"

    def test_change_nature_values(self):
        """ChangeNature has expected values"""
        assert ChangeNature.STABLE.value == "stable"
        assert ChangeNature.FLUCTUATING.value == "fluctuating"
        assert ChangeNature.SHIFTING.value == "shifting"
        assert ChangeNature.TRANSFORMED.value == "transformed"
        assert ChangeNature.RETURNING.value == "returning"

    def test_component_change_type_values(self):
        """ComponentChangeType has expected values"""
        assert ComponentChangeType.UNCHANGED.value == "unchanged"
        assert ComponentChangeType.INTENSIFIED.value == "intensified"
        assert ComponentChangeType.SOFTENED.value == "softened"
        assert ComponentChangeType.SHIFTED.value == "shifted"

    def test_temporal_span_values(self):
        """TemporalSpan has expected values"""
        assert TemporalSpan.IMMEDIATE.value == "immediate"
        assert TemporalSpan.SHORT_TERM.value == "short_term"
        assert TemporalSpan.MEDIUM_TERM.value == "medium_term"
        assert TemporalSpan.LONG_TERM.value == "long_term"

    def test_all_enums_are_strings(self):
        """All enum values are strings, not numbers"""
        all_enums = [
            DifferenceMagnitude,
            ChangeNature,
            ComponentChangeType,
            TemporalSpan,
        ]

        for enum_class in all_enums:
            for member in enum_class:
                assert isinstance(member.value, str), \
                    f"{enum_class.__name__}.{member.name} should have string value"


# =============================================================================
# Tests for Component Difference
# =============================================================================

class TestComponentDifference:
    """Test ComponentDifference structure"""

    def test_unchanged_creation(self):
        """Unchanged component difference can be created"""
        diff = ComponentDifference.unchanged("emotional", "calm")
        assert diff.component_name == "emotional"
        assert diff.change_type == ComponentChangeType.UNCHANGED
        assert diff.from_state == "calm"
        assert diff.to_state == "calm"

    def test_undefined_creation(self):
        """Undefined component difference can be created"""
        diff = ComponentDifference.undefined("tendency")
        assert diff.component_name == "tendency"
        assert diff.change_type == ComponentChangeType.UNDEFINED
        assert "undefined" in diff.from_state
        assert "undefined" in diff.to_state

    def test_difference_is_frozen(self):
        """ComponentDifference is immutable"""
        diff = ComponentDifference.unchanged("emotional", "calm")
        with pytest.raises(Exception):
            diff.change_type = ComponentChangeType.SHIFTED


# =============================================================================
# Tests for Component Comparison Functions
# =============================================================================

class TestComponentComparison:
    """Test component comparison functions"""

    def test_emotional_unchanged(self, calm_self_state):
        """Same emotional state returns unchanged"""
        diff = compare_emotional_state(calm_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.UNCHANGED

    def test_emotional_intensified(self, calm_self_state, intense_self_state):
        """Intensity increase returns intensified"""
        diff = compare_emotional_state(intense_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.INTENSIFIED

    def test_emotional_softened(self, calm_self_state, intense_self_state):
        """Intensity decrease returns softened"""
        diff = compare_emotional_state(calm_self_state, intense_self_state)
        assert diff.change_type == ComponentChangeType.SOFTENED

    def test_responsibility_unchanged(self, calm_self_state):
        """Same responsibility state returns unchanged"""
        diff = compare_responsibility_state(calm_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.UNCHANGED

    def test_responsibility_intensified(self, calm_self_state, intense_self_state):
        """Burden increase returns intensified"""
        diff = compare_responsibility_state(intense_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.INTENSIFIED

    def test_tendency_unchanged(self, calm_self_state):
        """Same tendency state returns unchanged"""
        diff = compare_tendency_state(calm_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.UNCHANGED

    def test_tendency_intensified(self, calm_self_state, intense_self_state):
        """Habit establishment returns intensified"""
        diff = compare_tendency_state(intense_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.INTENSIFIED

    def test_direction_unchanged(self, calm_self_state):
        """Same direction state returns unchanged"""
        diff = compare_direction_state(calm_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.UNCHANGED

    def test_direction_clarified(self, calm_self_state, intense_self_state):
        """Clarity increase returns intensified"""
        diff = compare_direction_state(intense_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.INTENSIFIED

    def test_value_unchanged(self, calm_self_state):
        """Same value state returns unchanged"""
        diff = compare_value_state(calm_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.UNCHANGED

    def test_value_stabilized(self, calm_self_state, intense_self_state):
        """Stability increase returns intensified"""
        diff = compare_value_state(intense_self_state, calm_self_state)
        assert diff.change_type == ComponentChangeType.INTENSIFIED


# =============================================================================
# Tests for SelfDifferenceSummary
# =============================================================================

class TestSelfDifferenceSummary:
    """Test SelfDifferenceSummary structure"""

    def test_empty_summary(self):
        """Empty summary can be created"""
        summary = create_empty_summary()
        assert not summary.has_difference
        assert summary.magnitude == DifferenceMagnitude.NONE

    def test_get_changed_components(self, temporal_system, calm_self_state, intense_self_state):
        """Changed components are listed"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        changed = summary.get_changed_components()
        # All components should be changed between calm and intense
        assert len(changed) > 0

    def test_get_unchanged_components(self, temporal_system, calm_self_state):
        """Unchanged components are listed"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(calm_self_state, calm_self_state)

        unchanged = summary.get_unchanged_components()
        assert len(unchanged) == 5  # All components unchanged

    def test_serialization_roundtrip(self, temporal_system, calm_self_state, intense_self_state):
        """Summary can be serialized and deserialized"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        # Serialize
        data = summary.to_dict()
        assert isinstance(data, dict)

        # Deserialize
        restored = SelfDifferenceSummary.from_dict(data)

        assert restored.has_difference == summary.has_difference
        assert restored.magnitude == summary.magnitude
        assert restored.nature == summary.nature


# =============================================================================
# Tests for TemporalSelfDifferenceSystem
# =============================================================================

class TestTemporalSelfDifferenceSystem:
    """Test TemporalSelfDifferenceSystem"""

    def test_record_snapshot(self, temporal_system, calm_self_state):
        """Snapshots can be recorded"""
        assert temporal_system.get_history_size() == 0

        temporal_system.record_snapshot(calm_self_state)
        assert temporal_system.get_history_size() == 1

    def test_compare_immediate(self, temporal_system, calm_self_state, intense_self_state):
        """Immediate comparison works"""
        temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_immediate(intense_self_state)

        assert summary is not None
        assert summary.has_difference

    def test_compare_with_no_history(self, temporal_system, calm_self_state):
        """Comparison with no history returns None"""
        summary = temporal_system.compare_immediate(calm_self_state)
        assert summary is None

    def test_no_difference_same_state(self, temporal_system, calm_self_state):
        """Same state shows no difference"""
        temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_immediate(calm_self_state)

        assert summary is not None
        assert not summary.has_difference
        assert summary.magnitude == DifferenceMagnitude.NONE

    def test_substantial_difference(self, temporal_system, calm_self_state, intense_self_state):
        """Large change shows substantial difference"""
        temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_immediate(intense_self_state)

        assert summary.has_difference
        assert summary.magnitude in (
            DifferenceMagnitude.SIGNIFICANT,
            DifferenceMagnitude.SUBSTANTIAL,
        )

    def test_compare_short_term(self, temporal_system, calm_self_state, intense_self_state):
        """Short-term comparison works"""
        # Add multiple snapshots
        for i in range(5):
            temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_short_term(intense_self_state)

        assert summary is not None
        assert summary.has_difference

    def test_compare_medium_term(self, temporal_system, calm_self_state, intense_self_state):
        """Medium-term comparison works"""
        # Add many snapshots
        for i in range(20):
            temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_medium_term(intense_self_state)

        assert summary is not None

    def test_history_limit(self, temporal_system, calm_self_state):
        """History size is limited"""
        config = TemporalDifferenceConfig(max_history_size=10)
        system = TemporalSelfDifferenceSystem(config)

        for i in range(20):
            system.record_snapshot(calm_self_state)

        assert system.get_history_size() == 10

    def test_comparison_count(self, temporal_system, calm_self_state, intense_self_state):
        """Comparison count increments"""
        temporal_system.record_snapshot(calm_self_state)

        assert temporal_system.get_comparison_count() == 0

        temporal_system.compare_immediate(intense_self_state)
        assert temporal_system.get_comparison_count() == 1

        temporal_system.compare_immediate(intense_self_state)
        assert temporal_system.get_comparison_count() == 2

    def test_clear_history(self, temporal_system, calm_self_state):
        """History can be cleared"""
        temporal_system.record_snapshot(calm_self_state)
        temporal_system.record_snapshot(calm_self_state)

        assert temporal_system.get_history_size() == 2

        temporal_system.clear_history()
        assert temporal_system.get_history_size() == 0


# =============================================================================
# Tests for NO Decision Impact
# =============================================================================

class TestNoDecisionImpact:
    """Test that difference awareness does NOT affect decisions"""

    def test_no_bias_modification_interface(self, temporal_system):
        """System has no bias modification methods"""
        assert not hasattr(temporal_system, "apply_bias")
        assert not hasattr(temporal_system, "modify_decision")
        assert not hasattr(temporal_system, "adjust_score")

    def test_summary_has_no_decision_methods(self, temporal_system, calm_self_state, intense_self_state):
        """SelfDifferenceSummary has no decision-affecting methods"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        assert not hasattr(summary, "apply_to_decision")
        assert not hasattr(summary, "compute_bias")
        assert not hasattr(summary, "modify_policy")

    def test_tags_have_low_weight(self, temporal_system, calm_self_state, intense_self_state):
        """Generated tags have low weight for awareness only"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        tags = generate_difference_tags(summary)

        for tag in tags:
            assert tag["weight"] <= 0.15, \
                f"Tag {tag['label']} has weight {tag['weight']} > 0.15"


# =============================================================================
# Tests for NO Judgment
# =============================================================================

class TestNoJudgment:
    """Test that system makes NO evaluative judgments"""

    def test_no_good_bad_in_enums(self):
        """Enums don't contain good/bad values"""
        for magnitude in DifferenceMagnitude:
            assert "good" not in magnitude.value.lower()
            assert "bad" not in magnitude.value.lower()

        for nature in ChangeNature:
            assert "good" not in nature.value.lower()
            assert "bad" not in nature.value.lower()
            assert "progress" not in nature.value.lower()
            assert "regression" not in nature.value.lower()

    def test_no_evaluation_in_description(self, temporal_system, calm_self_state, intense_self_state):
        """Descriptions don't contain evaluative terms"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        desc = summary.integrated_description.lower()

        assert "good" not in desc
        assert "bad" not in desc
        assert "better" not in desc
        assert "worse" not in desc
        assert "progress" not in desc
        assert "regression" not in desc

    def test_no_should_statements(self, temporal_system, calm_self_state, intense_self_state):
        """Descriptions don't contain prescriptive statements"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        desc = summary.integrated_description.lower()

        assert "should" not in desc
        assert "must" not in desc
        assert "need to" not in desc


# =============================================================================
# Tests for Natural Shrinking of Differences
# =============================================================================

class TestNaturalShrinking:
    """Test that differences naturally shrink when states converge"""

    def test_difference_shrinks_on_convergence(self, temporal_system, calm_self_state, intense_self_state):
        """Difference shrinks when states become similar again"""
        # Start with calm
        temporal_system.record_snapshot(calm_self_state)

        # Change to intense - large difference
        temporal_system.record_snapshot(intense_self_state)
        summary1 = temporal_system.compare_with_reference(intense_self_state, -2)
        assert summary1.has_difference

        # Return to calm - difference should be less from immediate
        temporal_system.record_snapshot(calm_self_state)
        summary2 = temporal_system.compare_immediate(calm_self_state)

        # Compare with original calm - should show no/minimal difference
        summary3 = temporal_system.compare_with_reference(calm_self_state, -3)
        assert summary3.magnitude in (DifferenceMagnitude.NONE, DifferenceMagnitude.MINIMAL)

    def test_no_explicit_correction_operation(self, temporal_system):
        """System has no explicit correction/recovery methods"""
        assert not hasattr(temporal_system, "correct")
        assert not hasattr(temporal_system, "recover")
        assert not hasattr(temporal_system, "fix")
        assert not hasattr(temporal_system, "reset_difference")


# =============================================================================
# Tests for SelfReferenceSystem Integration
# =============================================================================

class TestSelfReferenceIntegration:
    """Test integration with SelfReferenceSystem"""

    def test_generate_tags(self, temporal_system, calm_self_state, intense_self_state):
        """Tags are generated for SelfReferenceSystem"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        tags = generate_difference_tags(summary)

        assert isinstance(tags, list)
        assert len(tags) > 0

        for tag in tags:
            assert "category" in tag
            assert "label" in tag
            assert "description" in tag
            assert "weight" in tag

    def test_tag_categories(self, temporal_system, calm_self_state, intense_self_state):
        """Tags have appropriate categories"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        tags = generate_difference_tags(summary)
        categories = {tag["category"] for tag in tags}

        # Should have difference-related categories
        assert any("SELF_DIFFERENCE" in cat for cat in categories)

    def test_no_difference_tag(self, temporal_system, calm_self_state):
        """No-difference state generates appropriate tag"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(calm_self_state, calm_self_state)

        tags = generate_difference_tags(summary)

        assert len(tags) == 1
        assert tags[0]["label"] == "self_consistent"


# =============================================================================
# Tests for IntrospectionTrace Integration
# =============================================================================

class TestIntrospectionIntegration:
    """Test integration with IntrospectionTrace"""

    def test_get_introspection_data(self, temporal_system, calm_self_state, intense_self_state):
        """Introspection data is generated"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        data = get_difference_for_introspection(summary)

        assert isinstance(data, dict)
        assert "has_difference" in data
        assert "magnitude" in data
        assert "nature" in data
        assert "changed_components" in data
        assert "integrated_description" in data

    def test_introspection_data_has_abstract_values(self, temporal_system, calm_self_state, intense_self_state):
        """Introspection data has abstract (not numeric) values"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        data = get_difference_for_introspection(summary)

        assert isinstance(data["magnitude"], str)
        assert isinstance(data["nature"], str)

    def test_get_summary_string(self, temporal_system, calm_self_state, intense_self_state):
        """Summary string is human-readable"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        summary_str = get_difference_summary(summary)

        assert isinstance(summary_str, str)
        assert len(summary_str) > 50
        assert "Self-Difference Awareness" in summary_str


# =============================================================================
# Tests for Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration options"""

    def test_default_config(self):
        """Default config has reasonable values"""
        config = TemporalDifferenceConfig()

        assert config.max_history_size > 0
        assert config.immediate_window > 0
        assert config.short_term_window > config.immediate_window
        assert config.medium_term_window > config.short_term_window

    def test_custom_config(self):
        """Custom config can be created"""
        config = create_config(
            max_history_size=100,
            immediate_window=5,
        )

        assert config.max_history_size == 100
        assert config.immediate_window == 5


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_compare_with_empty_history(self, temporal_system, calm_self_state):
        """Comparing with empty history returns None"""
        summary = temporal_system.compare_immediate(calm_self_state)
        assert summary is None

    def test_compare_with_invalid_index(self, temporal_system, calm_self_state):
        """Comparing with invalid index returns None"""
        temporal_system.record_snapshot(calm_self_state)

        summary = temporal_system.compare_with_reference(calm_self_state, -100)
        assert summary is None

    def test_undefined_component_handling(self):
        """Undefined components are handled gracefully"""
        undefined_emotional = EmotionalStateView.undefined()
        defined_emotional = EmotionalStateView(
            spread=EmotionalSpread.FOCUSED,
            intensity=EmotionalIntensity.CALM,
            harmony=EmotionalHarmony.HARMONIOUS,
            active_emotion_count=1,
            has_coexisting_pairs=False,
            description="Defined",
        )

        state1 = create_self_state_view(
            undefined_emotional,
            ResponsibilityStateView.undefined(),
            TendencyStateView.undefined(),
            DirectionStateView.undefined(),
            ValueStateView.undefined(),
            "state1",
        )

        state2 = create_self_state_view(
            defined_emotional,
            ResponsibilityStateView.undefined(),
            TendencyStateView.undefined(),
            DirectionStateView.undefined(),
            ValueStateView.undefined(),
            "state2",
        )

        # Should not raise
        diff = compare_emotional_state(state1, state2)
        assert diff.change_type == ComponentChangeType.UNDEFINED


# =============================================================================
# Tests for Philosophy Compliance
# =============================================================================

class TestPhilosophyCompliance:
    """Test that implementation follows design philosophy"""

    def test_difference_is_awareness_not_problem(self, temporal_system, calm_self_state, intense_self_state):
        """Difference is for awareness, not a problem to solve"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        # Should not have problem-solving related attributes
        assert not hasattr(summary, "solution")
        assert not hasattr(summary, "fix")
        assert not hasattr(summary, "correction")

    def test_self_not_protected(self, temporal_system):
        """System does not try to protect self"""
        assert not hasattr(temporal_system, "protect")
        assert not hasattr(temporal_system, "preserve")
        assert not hasattr(temporal_system, "maintain_identity")

    def test_action_remains_free(self, temporal_system, calm_self_state, intense_self_state):
        """Summary does not constrain future actions"""
        temporal_system.record_snapshot(calm_self_state)
        summary = temporal_system._compare(intense_self_state, calm_self_state)

        assert not hasattr(summary, "recommended_action")
        assert not hasattr(summary, "forbidden_action")
        assert not hasattr(summary, "required_behavior")


# =============================================================================
# Import Test
# =============================================================================

def test_import_from_psyche():
    """All exports are importable from psyche package"""
    from psyche import (
        DifferenceMagnitude,
        ChangeNature,
        ComponentChangeType,
        TemporalSpan,
        ComponentDifference,
        SelfDifferenceSummary,
        TemporalDifferenceConfig,
        TemporalDifferenceState,
        TemporalSelfDifferenceSystem,
        compare_emotional_diff,
        compare_responsibility_diff,
        compare_tendency_diff,
        compare_direction_diff,
        compare_value_diff,
        determine_magnitude,
        determine_nature,
        generate_difference_tags,
        get_difference_summary,
        get_difference_for_introspection,
        save_difference_history,
        load_difference_history,
        create_difference_config,
        create_empty_difference_summary,
    )

    # Verify they're the right types
    assert issubclass(TemporalSelfDifferenceSystem, object)
    assert callable(generate_difference_tags)
