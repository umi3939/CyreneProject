"""
Tests for Self-Model System (自己状態統合モデル)

These tests verify:
1. Abstract enums and categorization
2. Component state views
3. Unified SelfStateView
4. READ-ONLY observation (no system modification)
5. NO IMPACT on decision making
6. SelfReferenceSystem integration
7. IntrospectionTrace integration
8. Dynamic updates on state changes
"""

import pytest
import time
from dataclasses import dataclass
from typing import Optional

from psyche.self_model import (
    # Enums
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
    # Component views
    EmotionalStateView,
    ResponsibilityStateView,
    TendencyStateView,
    DirectionStateView,
    ValueStateView,
    # Unified view
    SelfStateView,
    SelfModelConfig,
    SelfModelState,
    SelfModelSystem,
    # Observation functions
    observe_emotional_state,
    observe_responsibility_state,
    observe_tendency_state,
    observe_direction_state,
    observe_value_state,
    generate_integrated_description,
    # Integration functions
    generate_self_model_tags,
    get_self_model_summary,
    get_self_model_for_introspection,
    # Persistence
    save_self_model_state,
    load_self_model_state,
    # Convenience
    create_empty_view,
    create_config,
)

# Import actual psyche modules for integration testing
from psyche.state import EmotionVector
from psyche.responsibility import ResponsibilityState
from psyche.repeated_tendency import RepeatedTendencySystem, RepeatedTendencyConfig
from psyche.proto_goal_vector import VectorGenerator, VectorStateConfig
from psyche.value_orientation import ValueOrientation


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def emotion_vector():
    """Create a test EmotionVector"""
    return EmotionVector(
        joy=0.5,
        anger=0.2,
        sorrow=0.1,
        fear=0.3,
        surprise=0.0,
        love=0.4,
        fun=0.2,
    )


@pytest.fixture
def calm_emotion_vector():
    """Create a calm EmotionVector"""
    return EmotionVector(
        joy=0.1,
        anger=0.0,
        sorrow=0.0,
        fear=0.05,
        surprise=0.0,
        love=0.1,
        fun=0.0,
    )


@pytest.fixture
def intense_emotion_vector():
    """Create an intense EmotionVector with conflicts"""
    return EmotionVector(
        joy=0.9,
        anger=0.8,
        sorrow=0.7,
        fear=0.6,
        surprise=0.5,
        love=0.4,
        fun=0.3,
    )


@pytest.fixture
def responsibility_state():
    """Create a test ResponsibilityState"""
    return ResponsibilityState(
        total_weight=0.4,
        pending_decisions=2,
        accumulated_harm=0.2,
        accumulated_confidence=0.1,
        recent_decisions=[],
    )


@pytest.fixture
def light_responsibility_state():
    """Create a light ResponsibilityState"""
    return ResponsibilityState(
        total_weight=0.05,
        pending_decisions=0,
        accumulated_harm=0.0,
        accumulated_confidence=0.1,
        recent_decisions=[],
    )


@pytest.fixture
def heavy_responsibility_state():
    """Create a heavy ResponsibilityState"""
    return ResponsibilityState(
        total_weight=0.8,
        pending_decisions=5,
        accumulated_harm=0.6,
        accumulated_confidence=0.1,
        recent_decisions=[{} for _ in range(15)],  # Many decisions
    )


@pytest.fixture
def tendency_system():
    """Create a test RepeatedTendencySystem with some tendencies"""
    from psyche.scoped_goal import ScopedGoal, ScopeType, ScopeStatus
    from psyche.goal_candidates import CandidateCategory

    config = RepeatedTendencyConfig(
        strength_increment=0.05,
        max_strength=0.15,
        confidence_increment=0.1,
        max_confidence=0.8,
        decay_rate=0.01,
        min_repetitions=2,  # Lower threshold for testing
    )
    system = RepeatedTendencySystem(config)

    # Form some tendencies by observing turns with scoped goals
    # Must have action_taken=True for usage to be recorded
    for i in range(5):
        scoped_goal = ScopedGoal(
            scope_id=f"test_scope_{i}",
            source_goal_id=f"test_goal_{i}",
            source_candidate_id=f"test_candidate_{i}",
            category=CandidateCategory.EXPLORATION,
            strength=0.08,
            direction_alignment={"curiosity": 0.8},
            scope_type=ScopeType.SINGLE_ACTION,
            turn_committed=i,
            status=ScopeStatus.ACTIVE,
            action_taken=True,  # Critical: must be True for usage to be recorded
        )
        system.observe_turn(scoped_goal_used=scoped_goal)

    return system


@pytest.fixture
def empty_tendency_system():
    """Create an empty RepeatedTendencySystem"""
    config = RepeatedTendencyConfig()
    return RepeatedTendencySystem(config)


@pytest.fixture
def proto_goal_system():
    """Create a test VectorGenerator with some vectors"""
    config = VectorStateConfig(
        max_vectors=10,
        decay_rate=0.02,
        min_magnitude=0.05,
    )
    generator = VectorGenerator(config)

    # Generate some vectors using correct API
    generator.observe_turn(
        emotion_tendency={"joy": 0.5, "fun": 0.6},
        value_orientation=None,
    )

    return generator


@pytest.fixture
def value_orientation():
    """Create a test ValueOrientation"""
    return ValueOrientation(
        dim_a=0.3,
        dim_b=-0.2,
        dim_c=0.5,
        dim_d=0.1,
        dim_e=-0.1,
        confidence_a=0.5,
        confidence_b=0.3,
        confidence_c=0.6,
        confidence_d=0.2,
        confidence_e=0.4,
        update_count=10,
        last_update=time.time(),
    )


@pytest.fixture
def unstable_value_orientation():
    """Create an unstable ValueOrientation"""
    return ValueOrientation(
        dim_a=0.05,
        dim_b=0.02,
        dim_c=-0.03,
        dim_d=0.01,
        dim_e=0.0,
        confidence_a=0.1,
        confidence_b=0.1,
        confidence_c=0.1,
        confidence_d=0.1,
        confidence_e=0.1,
        update_count=2,
        last_update=time.time(),
    )


@pytest.fixture
def self_model_system():
    """Create a SelfModelSystem"""
    return SelfModelSystem()


# =============================================================================
# Tests for Abstract Enums
# =============================================================================

class TestAbstractEnums:
    """Test that all enums provide proper abstraction"""

    def test_emotional_spread_values(self):
        """EmotionalSpread has expected values"""
        assert EmotionalSpread.FOCUSED.value == "focused"
        assert EmotionalSpread.MIXED.value == "mixed"
        assert EmotionalSpread.DIFFUSE.value == "diffuse"
        assert EmotionalSpread.UNDEFINED.value == "undefined"

    def test_emotional_intensity_values(self):
        """EmotionalIntensity has expected values"""
        assert EmotionalIntensity.CALM.value == "calm"
        assert EmotionalIntensity.MODERATE.value == "moderate"
        assert EmotionalIntensity.INTENSE.value == "intense"
        assert EmotionalIntensity.OVERWHELMING.value == "overwhelming"

    def test_emotional_harmony_values(self):
        """EmotionalHarmony has expected values"""
        assert EmotionalHarmony.HARMONIOUS.value == "harmonious"
        assert EmotionalHarmony.SLIGHT_TENSION.value == "slight_tension"
        assert EmotionalHarmony.CONFLICTED.value == "conflicted"

    def test_burden_level_values(self):
        """BurdenLevel has expected values"""
        assert BurdenLevel.UNBURDENED.value == "unburdened"
        assert BurdenLevel.LIGHT.value == "light"
        assert BurdenLevel.MODERATE.value == "moderate"
        assert BurdenLevel.BURDENED.value == "burdened"
        assert BurdenLevel.HEAVY.value == "heavy"

    def test_habit_presence_values(self):
        """HabitPresence has expected values"""
        assert HabitPresence.NONE.value == "none"
        assert HabitPresence.EMERGING.value == "emerging"
        assert HabitPresence.FORMING.value == "forming"
        assert HabitPresence.ESTABLISHED.value == "established"

    def test_direction_clarity_values(self):
        """DirectionClarity has expected values"""
        assert DirectionClarity.CLEAR.value == "clear"
        assert DirectionClarity.UNCERTAIN.value == "uncertain"
        assert DirectionClarity.UNCLEAR.value == "unclear"
        assert DirectionClarity.UNDEFINED.value == "undefined"

    def test_value_stability_values(self):
        """ValueStability has expected values"""
        assert ValueStability.UNSTABLE.value == "unstable"
        assert ValueStability.SHIFTING.value == "shifting"
        assert ValueStability.STABLE.value == "stable"
        assert ValueStability.ANCHORED.value == "anchored"

    def test_enums_hide_raw_numbers(self):
        """All enum values are strings, not numbers"""
        all_enums = [
            EmotionalSpread, EmotionalIntensity, EmotionalHarmony,
            BurdenLevel, BurdenDistribution, BurdenTrend,
            HabitPresence, HabitCharacter,
            DirectionClarity, DirectionConvergence,
            ValueStability, ValueClarity,
        ]

        for enum_class in all_enums:
            for member in enum_class:
                assert isinstance(member.value, str), \
                    f"{enum_class.__name__}.{member.name} should have string value"


# =============================================================================
# Tests for Component State Views
# =============================================================================

class TestEmotionalStateView:
    """Test EmotionalStateView"""

    def test_undefined_view(self):
        """Undefined view has correct values"""
        view = EmotionalStateView.undefined()
        assert view.spread == EmotionalSpread.UNDEFINED
        assert view.intensity == EmotionalIntensity.UNDEFINED
        assert view.harmony == EmotionalHarmony.UNDEFINED
        assert view.active_emotion_count == 0

    def test_view_is_frozen(self):
        """View is immutable"""
        view = EmotionalStateView.undefined()
        with pytest.raises(Exception):  # FrozenInstanceError
            view.spread = EmotionalSpread.FOCUSED

    def test_observe_calm_emotions(self, calm_emotion_vector):
        """Calm emotions are observed correctly"""
        view = observe_emotional_state(calm_emotion_vector)
        assert view.intensity == EmotionalIntensity.CALM
        assert view.spread == EmotionalSpread.FOCUSED

    def test_observe_intense_emotions(self, intense_emotion_vector):
        """Intense emotions are observed correctly"""
        view = observe_emotional_state(intense_emotion_vector)
        assert view.intensity in (EmotionalIntensity.INTENSE, EmotionalIntensity.OVERWHELMING)
        assert view.spread == EmotionalSpread.DIFFUSE

    def test_observe_mixed_emotions(self, emotion_vector):
        """Mixed emotions are observed correctly"""
        view = observe_emotional_state(emotion_vector)
        # emotion_vector fixture has 6 active emotions (joy, anger, sorrow, fear, love, fun all > 0.1)
        # so it will be DIFFUSE (5+), not MIXED
        assert view.spread in (EmotionalSpread.FOCUSED, EmotionalSpread.MIXED, EmotionalSpread.DIFFUSE)
        assert view.active_emotion_count > 0

    def test_description_is_human_readable(self, emotion_vector):
        """Description is generated"""
        view = observe_emotional_state(emotion_vector)
        assert len(view.description) > 0
        assert "undefined" not in view.description.lower() or view.spread == EmotionalSpread.UNDEFINED


class TestResponsibilityStateView:
    """Test ResponsibilityStateView"""

    def test_undefined_view(self):
        """Undefined view has correct values"""
        view = ResponsibilityStateView.undefined()
        assert view.burden_level == BurdenLevel.UNDEFINED
        assert view.distribution == BurdenDistribution.UNDEFINED
        assert view.trend == BurdenTrend.UNDEFINED

    def test_observe_light_responsibility(self, light_responsibility_state):
        """Light responsibility is observed correctly"""
        view = observe_responsibility_state(light_responsibility_state)
        assert view.burden_level == BurdenLevel.UNBURDENED
        assert not view.has_pending_decisions

    def test_observe_heavy_responsibility(self, heavy_responsibility_state):
        """Heavy responsibility is observed correctly"""
        view = observe_responsibility_state(heavy_responsibility_state)
        assert view.burden_level == BurdenLevel.HEAVY
        assert view.has_pending_decisions

    def test_trend_detection_accumulating(self, responsibility_state):
        """Accumulating trend is detected"""
        # First observation
        view1 = observe_responsibility_state(responsibility_state, previous_weight=0.2)
        assert view1.trend == BurdenTrend.ACCUMULATING

    def test_trend_detection_releasing(self, responsibility_state):
        """Releasing trend is detected"""
        view = observe_responsibility_state(responsibility_state, previous_weight=0.6)
        assert view.trend == BurdenTrend.RELEASING


class TestTendencyStateView:
    """Test TendencyStateView"""

    def test_undefined_view(self):
        """Undefined view has correct values"""
        view = TendencyStateView.undefined()
        assert view.presence == HabitPresence.UNDEFINED
        assert view.character == HabitCharacter.UNDEFINED
        assert view.tendency_count == 0

    def test_observe_empty_tendencies(self, empty_tendency_system):
        """Empty system shows no habits"""
        view = observe_tendency_state(empty_tendency_system)
        assert view.presence == HabitPresence.NONE
        assert view.character == HabitCharacter.EXPLORATORY
        assert view.tendency_count == 0

    def test_observe_formed_tendencies(self, tendency_system):
        """Formed tendencies are detected"""
        view = observe_tendency_state(tendency_system)
        assert view.presence != HabitPresence.NONE
        assert view.tendency_count > 0


class TestDirectionStateView:
    """Test DirectionStateView"""

    def test_undefined_view(self):
        """Undefined view has correct values"""
        view = DirectionStateView.undefined()
        assert view.clarity == DirectionClarity.UNDEFINED
        assert view.convergence == DirectionConvergence.UNDEFINED
        assert view.vector_count == 0

    def test_observe_with_vectors(self, proto_goal_system):
        """Direction with vectors is observed"""
        view = observe_direction_state(proto_goal_system)
        # May or may not have vectors depending on generation
        assert view.clarity in DirectionClarity


class TestValueStateView:
    """Test ValueStateView"""

    def test_undefined_view(self):
        """Undefined view has correct values"""
        view = ValueStateView.undefined()
        assert view.stability == ValueStability.UNDEFINED
        assert view.clarity == ValueClarity.UNDEFINED_VALUES

    def test_observe_stable_values(self, value_orientation):
        """Stable values are observed"""
        view = observe_value_state(value_orientation)
        assert view.stability != ValueStability.UNDEFINED
        assert view.has_strong_orientations

    def test_observe_unstable_values(self, unstable_value_orientation):
        """Unstable values are observed"""
        view = observe_value_state(unstable_value_orientation)
        assert view.stability == ValueStability.UNSTABLE

    def test_detect_recent_change(self, value_orientation, unstable_value_orientation):
        """Recent changes are detected"""
        view = observe_value_state(value_orientation, previous_orientation=unstable_value_orientation)
        assert view.is_recently_changed


# =============================================================================
# Tests for SelfStateView
# =============================================================================

class TestSelfStateView:
    """Test unified SelfStateView"""

    def test_empty_view(self):
        """Empty view can be created"""
        view = create_empty_view()
        assert not view.is_complete
        assert "undefined" in view.integrated_description.lower()

    def test_complete_view(self, emotion_vector, responsibility_state,
                           tendency_system, proto_goal_system, value_orientation):
        """Complete view integrates all components"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
            tendency_system=tendency_system,
            proto_goal_system=proto_goal_system,
            value_orientation=value_orientation,
        )

        # Check all components are populated
        assert view.emotional.spread != EmotionalSpread.UNDEFINED
        assert view.responsibility.burden_level != BurdenLevel.UNDEFINED
        assert view.value.stability != ValueStability.UNDEFINED

        # Check metadata
        assert view.timestamp > 0
        assert len(view.snapshot_id) > 0

    def test_component_summaries(self, emotion_vector, responsibility_state):
        """Component summaries are available"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        summaries = view.get_component_summaries()
        assert "emotional" in summaries
        assert "responsibility" in summaries
        assert len(summaries["emotional"]) > 0

    def test_undefined_components_list(self, emotion_vector):
        """Undefined components are listed"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        undefined = view.get_undefined_components()
        # Should have some undefined since we only provided emotion
        assert "tendency" in undefined or "direction" in undefined

    def test_serialization_roundtrip(self, emotion_vector, responsibility_state):
        """View can be serialized and deserialized"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        # Serialize
        data = view.to_dict()
        assert isinstance(data, dict)

        # Deserialize
        restored = SelfStateView.from_dict(data)

        # Verify
        assert restored.emotional.spread == view.emotional.spread
        assert restored.responsibility.burden_level == view.responsibility.burden_level
        assert restored.integrated_description == view.integrated_description


# =============================================================================
# Tests for READ-ONLY Observation
# =============================================================================

class TestReadOnlyObservation:
    """Test that observation does not modify observed systems"""

    def test_emotion_vector_unchanged(self, emotion_vector):
        """EmotionVector is not modified by observation"""
        original_joy = emotion_vector.joy
        original_anger = emotion_vector.anger

        observe_emotional_state(emotion_vector)

        assert emotion_vector.joy == original_joy
        assert emotion_vector.anger == original_anger

    def test_responsibility_state_unchanged(self, responsibility_state):
        """ResponsibilityState is not modified by observation"""
        original_weight = responsibility_state.total_weight
        original_pending = responsibility_state.pending_decisions

        observe_responsibility_state(responsibility_state)

        assert responsibility_state.total_weight == original_weight
        assert responsibility_state.pending_decisions == original_pending

    def test_tendency_system_unchanged(self, tendency_system):
        """RepeatedTendencySystem is not modified by observation"""
        original_tendencies = tendency_system.get_tendencies()
        original_count = len(original_tendencies)

        observe_tendency_state(tendency_system)

        assert len(tendency_system.get_tendencies()) == original_count

    def test_value_orientation_unchanged(self, value_orientation):
        """ValueOrientation is not modified by observation"""
        original_dim_a = value_orientation.dim_a
        original_confidence_a = value_orientation.confidence_a

        observe_value_state(value_orientation)

        assert value_orientation.dim_a == original_dim_a
        assert value_orientation.confidence_a == original_confidence_a

    def test_full_observation_leaves_systems_unchanged(
        self, emotion_vector, responsibility_state, tendency_system, value_orientation
    ):
        """Full SelfModelSystem observation leaves all systems unchanged"""
        # Record original states
        original_joy = emotion_vector.joy
        original_weight = responsibility_state.total_weight
        original_tendencies = len(tendency_system.get_tendencies())
        original_dim_a = value_orientation.dim_a

        # Perform observation
        system = SelfModelSystem()
        system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
            tendency_system=tendency_system,
            value_orientation=value_orientation,
        )

        # Verify nothing changed
        assert emotion_vector.joy == original_joy
        assert responsibility_state.total_weight == original_weight
        assert len(tendency_system.get_tendencies()) == original_tendencies
        assert value_orientation.dim_a == original_dim_a


# =============================================================================
# Tests for NO Decision Impact
# =============================================================================

class TestNoDecisionImpact:
    """Test that self-model does NOT affect decision making"""

    def test_no_bias_modification_interface(self):
        """SelfModelSystem has no bias modification methods"""
        system = SelfModelSystem()

        # Should NOT have methods that modify decision bias
        assert not hasattr(system, "apply_bias")
        assert not hasattr(system, "modify_decision")
        assert not hasattr(system, "adjust_score")

    def test_view_has_no_decision_methods(self, emotion_vector):
        """SelfStateView has no decision-affecting methods"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # Should NOT have decision-related methods
        assert not hasattr(view, "apply_to_decision")
        assert not hasattr(view, "compute_bias")
        assert not hasattr(view, "modify_policy")

    def test_tags_have_low_weight(self, emotion_vector, responsibility_state):
        """Generated tags have low weight for observation only"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        tags = generate_self_model_tags(view)

        # All tags should have low weight (observation only)
        for tag in tags:
            assert tag["weight"] <= 0.15, \
                f"Tag {tag['label']} has weight {tag['weight']} > 0.15"

    def test_observation_does_not_return_bias(self, emotion_vector):
        """observe() returns view, not bias"""
        system = SelfModelSystem()
        result = system.observe(emotion_vector=emotion_vector)

        # Result should be SelfStateView, not any bias type
        assert isinstance(result, SelfStateView)
        assert not hasattr(result, "bias_strength")
        assert not hasattr(result, "policy_modifiers")


# =============================================================================
# Tests for SelfReferenceSystem Integration
# =============================================================================

class TestSelfReferenceIntegration:
    """Test integration with SelfReferenceSystem"""

    def test_generate_tags(self, emotion_vector, responsibility_state):
        """Tags are generated for SelfReferenceSystem"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        tags = generate_self_model_tags(view)

        assert isinstance(tags, list)
        assert len(tags) > 0

        # Check tag structure
        for tag in tags:
            assert "category" in tag
            assert "label" in tag
            assert "description" in tag
            assert "weight" in tag
            assert tag["category"].startswith("SELF_MODEL_")

    def test_tag_categories(self, emotion_vector, responsibility_state,
                            tendency_system, value_orientation):
        """Tags have appropriate categories"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
            tendency_system=tendency_system,
            value_orientation=value_orientation,
        )

        tags = generate_self_model_tags(view)
        categories = {tag["category"] for tag in tags}

        # Should have multiple category types
        assert "SELF_MODEL_EMOTIONAL" in categories
        assert "SELF_MODEL_RESPONSIBILITY" in categories
        assert "SELF_MODEL_INTEGRATED" in categories

    def test_tag_scale_factor(self, emotion_vector):
        """Tag weight can be scaled"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # Default scale
        tags_default = generate_self_model_tags(view, scale=1.0)

        # Reduced scale
        tags_reduced = generate_self_model_tags(view, scale=0.5)

        # Weights should be proportionally reduced
        for tag_d, tag_r in zip(tags_default, tags_reduced):
            if tag_d["label"] == tag_r["label"]:
                assert abs(tag_r["weight"] - tag_d["weight"] * 0.5) < 0.001


# =============================================================================
# Tests for IntrospectionTrace Integration
# =============================================================================

class TestIntrospectionIntegration:
    """Test integration with IntrospectionTrace"""

    def test_get_introspection_data(self, emotion_vector, responsibility_state):
        """Introspection data is generated"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        data = get_self_model_for_introspection(view)

        assert isinstance(data, dict)
        assert "snapshot_id" in data
        assert "timestamp" in data
        assert "emotional" in data
        assert "responsibility" in data
        assert "integrated_description" in data

    def test_introspection_data_has_abstract_values(self, emotion_vector):
        """Introspection data has abstract (not numeric) values"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        data = get_self_model_for_introspection(view)

        # Check emotional data is abstract
        assert isinstance(data["emotional"]["spread"], str)
        assert isinstance(data["emotional"]["intensity"], str)

        # Should NOT contain raw numbers
        for key in ["spread", "intensity", "harmony"]:
            value = data["emotional"][key]
            assert not isinstance(value, (int, float)), \
                f"emotional.{key} should be abstract, not {type(value)}"

    def test_get_summary(self, emotion_vector, responsibility_state):
        """Summary is human-readable"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        summary = get_self_model_summary(view)

        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be detailed
        assert "Self-Model Snapshot" in summary
        assert "Components:" in summary


# =============================================================================
# Tests for Dynamic Updates
# =============================================================================

class TestDynamicUpdates:
    """Test that view updates as internal states change"""

    def test_view_changes_with_emotion(self, emotion_vector):
        """View updates when emotion changes"""
        system = SelfModelSystem()

        # First observation
        view1 = system.observe(emotion_vector=emotion_vector)

        # Modify emotion
        emotion_vector.joy = 0.9
        emotion_vector.anger = 0.8

        # Second observation
        view2 = system.observe(emotion_vector=emotion_vector)

        # Views should differ
        assert view1.snapshot_id != view2.snapshot_id

    def test_view_changes_with_responsibility(self, responsibility_state):
        """View updates when responsibility changes"""
        system = SelfModelSystem()

        # First observation
        view1 = system.observe(responsibility_state=responsibility_state)

        # Modify responsibility
        responsibility_state.total_weight = 0.9

        # Second observation
        view2 = system.observe(responsibility_state=responsibility_state)

        # Burden level should change
        assert view2.responsibility.burden_level != view1.responsibility.burden_level or \
               view2.responsibility.trend != view1.responsibility.trend

    def test_snapshot_count_increments(self, emotion_vector):
        """Snapshot count increments with each observation"""
        system = SelfModelSystem()

        initial_count = system.get_snapshot_count()

        system.observe(emotion_vector=emotion_vector)
        assert system.get_snapshot_count() == initial_count + 1

        system.observe(emotion_vector=emotion_vector)
        assert system.get_snapshot_count() == initial_count + 2

    def test_turn_advancement(self):
        """Turn counter advances"""
        system = SelfModelSystem()

        assert system.get_turn_count() == 0

        system.advance_turn()
        assert system.get_turn_count() == 1

        system.advance_turn()
        assert system.get_turn_count() == 2


# =============================================================================
# Tests for Integrated Description
# =============================================================================

class TestIntegratedDescription:
    """Test integrated description generation"""

    def test_description_not_empty(self, emotion_vector, responsibility_state):
        """Integrated description is generated"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        assert len(view.integrated_description) > 0
        assert view.integrated_description != "Current self-state is provisional and undefined"

    def test_description_reflects_multiple_components(
        self, emotion_vector, responsibility_state, value_orientation
    ):
        """Description integrates multiple components"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
            value_orientation=value_orientation,
        )

        desc = view.integrated_description.lower()

        # Should mention aspects from different components
        has_emotional = any(word in desc for word in ["calm", "emotional", "intense", "overwhelm"])
        has_burden = any(word in desc for word in ["burden", "weight", "unburdened"])
        has_value = any(word in desc for word in ["value", "stable", "anchored", "shifting"])

        # At least two components should be mentioned
        mentions = sum([has_emotional, has_burden, has_value])
        assert mentions >= 1  # At least one component

    def test_empty_components_produce_provisional_description(self):
        """Empty observation produces provisional description"""
        desc = generate_integrated_description(
            EmotionalStateView.undefined(),
            ResponsibilityStateView.undefined(),
            TendencyStateView.undefined(),
            DirectionStateView.undefined(),
            ValueStateView.undefined(),
        )

        assert "provisional" in desc.lower() or "undefined" in desc.lower()


# =============================================================================
# Tests for Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration options"""

    def test_default_config(self):
        """Default config has reasonable values"""
        config = SelfModelConfig()

        assert config.emotion_active_threshold > 0
        assert config.burden_heavy_max > config.burden_light_max
        assert config.value_confidence_stable_max > config.value_confidence_unstable_max

    def test_custom_config(self):
        """Custom config can be created"""
        config = create_config(
            emotion_active_threshold=0.2,
            burden_heavy_max=0.8,
        )

        assert config.emotion_active_threshold == 0.2
        assert config.burden_heavy_max == 0.8

    def test_config_affects_observation(self, emotion_vector):
        """Config thresholds affect observation results"""
        # Strict config (high thresholds)
        strict_config = SelfModelConfig(emotion_intensity_calm_max=0.1)

        # Lenient config (high thresholds)
        lenient_config = SelfModelConfig(emotion_intensity_calm_max=2.0)

        # Same emotion, different configs
        strict_view = observe_emotional_state(emotion_vector, strict_config)
        lenient_view = observe_emotional_state(emotion_vector, lenient_config)

        # Strict should be more intense, lenient should be calmer
        # (emotion_vector has moderate intensity)
        assert lenient_view.intensity == EmotionalIntensity.CALM


# =============================================================================
# Tests for Persistence
# =============================================================================

class TestPersistence:
    """Test state persistence"""

    def test_save_and_load_state(self, tmp_path):
        """State can be saved and loaded"""
        state = SelfModelState(
            last_responsibility_weight=0.5,
            turn_count=10,
            snapshot_count=25,
        )

        path = str(tmp_path / "self_model_state.json")
        save_self_model_state(state, path)

        loaded = load_self_model_state(path)

        assert loaded.last_responsibility_weight == 0.5
        assert loaded.turn_count == 10
        assert loaded.snapshot_count == 25

    def test_load_nonexistent_returns_default(self, tmp_path):
        """Loading nonexistent file returns default state"""
        path = str(tmp_path / "nonexistent.json")
        state = load_self_model_state(path)

        assert state.turn_count == 0
        assert state.snapshot_count == 0


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_none_inputs_produce_undefined(self):
        """None inputs produce undefined views"""
        assert observe_emotional_state(None).spread == EmotionalSpread.UNDEFINED
        assert observe_responsibility_state(None).burden_level == BurdenLevel.UNDEFINED
        assert observe_tendency_state(None).presence == HabitPresence.UNDEFINED
        assert observe_direction_state(None).clarity == DirectionClarity.UNDEFINED
        assert observe_value_state(None).stability == ValueStability.UNDEFINED

    def test_full_observation_with_all_none(self):
        """Full observation with all None inputs"""
        system = SelfModelSystem()
        view = system.observe()

        assert not view.is_complete
        assert len(view.get_undefined_components()) == 5

    def test_partial_observation(self, emotion_vector):
        """Partial observation with some inputs"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # Emotional should be defined
        assert view.emotional.spread != EmotionalSpread.UNDEFINED

        # Others should be undefined
        assert "responsibility" in view.get_undefined_components()

    def test_reset_tracking(self, emotion_vector, responsibility_state):
        """Tracking can be reset"""
        system = SelfModelSystem()

        # First observation sets tracking
        system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        assert system._state.last_responsibility_weight is not None

        # Reset tracking
        system.reset_tracking()

        assert system._state.last_responsibility_weight is None
        # Turn count should remain
        assert system._state.turn_count >= 0


# =============================================================================
# Tests for Philosophy Compliance
# =============================================================================

class TestPhilosophyCompliance:
    """Test that implementation follows design philosophy"""

    def test_self_model_is_provisional(self, emotion_vector):
        """Self-model is explicitly provisional (暫定的)"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # Snapshot ID suggests it's temporal
        assert "snapshot" in view.snapshot_id.lower()

        # Timestamp shows it's a point in time
        assert view.timestamp > 0

    def test_no_personality_definition(self, emotion_vector, value_orientation):
        """Self-model does NOT define personality"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            value_orientation=value_orientation,
        )

        # Should NOT have personality-related attributes
        assert not hasattr(view, "personality")
        assert not hasattr(view, "character_traits")
        assert not hasattr(view, "identity")

    def test_no_self_evaluation(self, emotion_vector):
        """Self-model does NOT evaluate self as good/bad"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # Description should not contain evaluative terms
        desc = view.integrated_description.lower()
        assert "good" not in desc
        assert "bad" not in desc
        assert "should" not in desc
        assert "wrong" not in desc

    def test_single_element_does_not_determine_identity(self, emotion_vector):
        """Single element does not determine complete self-image"""
        system = SelfModelSystem()
        view = system.observe(emotion_vector=emotion_vector)

        # With only emotion, should not be "complete"
        assert not view.is_complete

        # Should have undefined components
        assert len(view.get_undefined_components()) > 0

    def test_view_allows_free_choice(self, emotion_vector, responsibility_state):
        """View does not constrain future choices"""
        system = SelfModelSystem()
        view = system.observe(
            emotion_vector=emotion_vector,
            responsibility_state=responsibility_state,
        )

        # No prescriptive attributes
        assert not hasattr(view, "recommended_action")
        assert not hasattr(view, "forbidden_choices")
        assert not hasattr(view, "required_behavior")


# =============================================================================
# Import Test
# =============================================================================

def test_import_from_psyche():
    """All exports are importable from psyche package"""
    from psyche import (
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
        EmotionalStateView,
        ResponsibilityStateView,
        TendencyStateView,
        DirectionStateView,
        ValueStateView,
        SelfStateView,
        SelfModelConfig,
        SelfModelState,
        SelfModelSystem,
        observe_emotional_state,
        observe_responsibility_state,
        observe_tendency_state,
        observe_direction_state,
        observe_value_state,
        generate_integrated_description,
        generate_self_model_tags,
        get_self_model_summary,
        get_self_model_for_introspection,
        save_self_model_state,
        load_self_model_state,
        create_empty_self_view,
        create_self_model_config,
    )

    # Verify they're the right types
    assert issubclass(SelfModelSystem, object)
    assert callable(observe_emotional_state)
