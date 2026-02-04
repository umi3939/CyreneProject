"""
Tests for Provisional Self-Image Integration (自己像統合システム)

These tests verify:
1. Abstract enums for self-image description (impressions, senses, atmospheres)
2. ProvisionalSelfImage is always provisional/temporary
3. Image changes with state changes
4. Contradictions are allowed to coexist
5. STRICTLY NO IMPACT on decision making
6. Connection to SelfReferenceSystem for introspection only
"""

import pytest
import time
from typing import Any
from dataclasses import dataclass

from psyche.self_image_integration import (
    # Enums
    EmotionalTone,
    TendencyHint,
    StabilityFeeling,
    ChangePresence,
    ContinuityFeeling,
    OverallImpression,
    # Core structures
    ImageAspect,
    ProvisionalSelfImage,
    SelfImageConfig,
    # System
    SelfImageIntegrationSystem,
    # Integration functions
    generate_self_image_tags,
    get_self_image_summary,
    get_self_image_for_introspection,
    # Convenience
    create_config,
    create_empty_image,
    # Verification
    verify_no_decision_impact,
    verify_provisional_nature,
    # Private integration functions for direct testing
    _integrate_emotional_tone,
    _integrate_tendency_hint,
    _integrate_stability_feeling,
    _integrate_change_presence,
    _integrate_continuity_feeling,
    _determine_overall_impression,
    _detect_contradictions,
)

# Import types we need to create test fixtures
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

from psyche.tendency_awareness import (
    TendencyAwareness,
    TendencyAwarenessItem,
    StrengthLevel,
    DurationLevel,
    ConfidenceLevel,
    AwarenessType,
)

from psyche.temporal_self_difference import (
    SelfDifferenceSummary,
    DifferenceMagnitude,
    ChangeNature,
    TemporalSpan,
    ComponentDifference,
    ComponentChangeType,
)

from psyche.continuity_strain import (
    StrainState,
    StrainPresence,
    StrainLevel,
    StrainPersistence,
    StrainTrend,
)

from psyche.goal_candidates import CandidateCategory


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default self-image configuration"""
    return SelfImageConfig()


@pytest.fixture
def image_system(default_config):
    """Default self-image integration system"""
    return SelfImageIntegrationSystem(default_config)


@pytest.fixture
def calm_emotional_view():
    """Create a calm emotional state view"""
    return EmotionalStateView(
        spread=EmotionalSpread.FOCUSED,
        intensity=EmotionalIntensity.CALM,
        harmony=EmotionalHarmony.HARMONIOUS,
        active_emotion_count=1,
        has_coexisting_pairs=False,
        description="Calm emotional state",
    )


@pytest.fixture
def intense_emotional_view():
    """Create an intense emotional state view"""
    return EmotionalStateView(
        spread=EmotionalSpread.FOCUSED,
        intensity=EmotionalIntensity.OVERWHELMING,
        harmony=EmotionalHarmony.HARMONIOUS,
        active_emotion_count=1,
        has_coexisting_pairs=False,
        description="Intense emotional state",
    )


@pytest.fixture
def mixed_emotional_view():
    """Create a mixed emotional state view"""
    return EmotionalStateView(
        spread=EmotionalSpread.MIXED,
        intensity=EmotionalIntensity.MODERATE,
        harmony=EmotionalHarmony.CONFLICTED,
        active_emotion_count=3,
        has_coexisting_pairs=True,
        description="Mixed emotional state",
    )


@pytest.fixture
def stable_value_view():
    """Create a stable value state view"""
    return ValueStateView(
        stability=ValueStability.ANCHORED,
        clarity=ValueClarity.DEFINED,
        has_strong_orientations=True,
        is_recently_changed=False,
        description="Stable values",
    )


@pytest.fixture
def unstable_value_view():
    """Create an unstable value state view"""
    return ValueStateView(
        stability=ValueStability.UNSTABLE,
        clarity=ValueClarity.EMERGING,
        has_strong_orientations=False,
        is_recently_changed=True,
        description="Unstable values",
    )


def create_self_state_view(
    emotional: EmotionalStateView,
    value: ValueStateView,
    tendency_presence: HabitPresence = HabitPresence.NONE,
) -> SelfStateView:
    """Helper to create a SelfStateView with specified components"""
    return SelfStateView(
        emotional=emotional,
        responsibility=ResponsibilityStateView(
            burden_level=BurdenLevel.LIGHT,
            distribution=BurdenDistribution.DISTRIBUTED,
            trend=BurdenTrend.STABLE,
            has_pending_decisions=False,
            description="Light burden",
        ),
        tendency=TendencyStateView(
            presence=tendency_presence,
            character=HabitCharacter.EXPLORATORY,
            tendency_count=0,
            has_strong_habits=False,
            has_fading_habits=False,
            description="No strong tendencies",
        ),
        direction=DirectionStateView(
            clarity=DirectionClarity.CLEAR,
            convergence=DirectionConvergence.CONVERGENT,
            vector_count=2,
            has_dominant_direction=True,
            description="Clear direction",
        ),
        value=value,
        timestamp=time.time(),
        snapshot_id="test-snapshot",
        is_complete=True,
        integrated_description="Test self state",
    )


@pytest.fixture
def calm_self_state(calm_emotional_view, stable_value_view):
    """Create a calm, stable self state"""
    return create_self_state_view(calm_emotional_view, stable_value_view)


@pytest.fixture
def turbulent_self_state(intense_emotional_view, unstable_value_view):
    """Create a turbulent self state"""
    return create_self_state_view(intense_emotional_view, unstable_value_view)


@pytest.fixture
def tendency_awareness_strong():
    """Create strong tendency awareness"""
    return TendencyAwareness(
        items=[
            TendencyAwarenessItem(
                awareness_type=AwarenessType.STRONG_HABIT,
                category=CandidateCategory.EXPLORATION,
                strength_level=StrengthLevel.STRONG,
                duration_level=DurationLevel.ESTABLISHED,
                confidence_level=ConfidenceLevel.ESTABLISHED,
                description="Strong exploration tendency",
            )
        ],
        has_awareness=True,
        dominant_category=CandidateCategory.EXPLORATION,
        overall_strength=StrengthLevel.STRONG,
    )


@pytest.fixture
def tendency_awareness_none():
    """Create no tendency awareness"""
    return TendencyAwareness(
        items=[],
        has_awareness=False,
        dominant_category=None,
        overall_strength=StrengthLevel.NONE,
    )


@pytest.fixture
def no_difference_summary():
    """Create a no-difference summary"""
    unchanged = ComponentDifference.unchanged("test", "stable")
    return SelfDifferenceSummary(
        has_difference=False,
        magnitude=DifferenceMagnitude.NONE,
        nature=ChangeNature.STABLE,
        temporal_span=TemporalSpan.IMMEDIATE,
        emotional_diff=unchanged,
        responsibility_diff=unchanged,
        tendency_diff=unchanged,
        direction_diff=unchanged,
        value_diff=unchanged,
        current_snapshot_id="current",
        reference_snapshot_id="reference",
        comparison_timestamp=time.time(),
        integrated_description="No difference",
    )


@pytest.fixture
def significant_difference_summary():
    """Create a significant difference summary"""
    changed = ComponentDifference(
        component_name="emotional",
        change_type=ComponentChangeType.SHIFTED,
        from_state="calm",
        to_state="intense",
        description="Emotional state shifted",
    )
    unchanged = ComponentDifference.unchanged("test", "stable")
    return SelfDifferenceSummary(
        has_difference=True,
        magnitude=DifferenceMagnitude.SIGNIFICANT,
        nature=ChangeNature.SHIFTING,
        temporal_span=TemporalSpan.SHORT_TERM,
        emotional_diff=changed,
        responsibility_diff=unchanged,
        tendency_diff=unchanged,
        direction_diff=unchanged,
        value_diff=unchanged,
        current_snapshot_id="current",
        reference_snapshot_id="reference",
        comparison_timestamp=time.time(),
        integrated_description="Significant change",
    )


@pytest.fixture
def no_strain_state():
    """Create an at-ease strain state"""
    return StrainState.at_ease()


@pytest.fixture
def high_strain_state():
    """Create a high strain state"""
    return StrainState(
        presence=StrainPresence.PRESENT,
        level=StrainLevel.ALIENATED,
        persistence=StrainPersistence.ONGOING,
        trend=StrainTrend.STABLE,
        timestamp=time.time(),
        last_update_timestamp=time.time(),
        description="High strain present",
    )


# =============================================================================
# Test Abstract Enums
# =============================================================================

class TestAbstractEnums:
    """Test that all enums are abstract (impressions, not definitive)"""

    def test_emotional_tone_values(self):
        """EmotionalTone should be impressions, not definitive emotions"""
        # Values should be general "tones", not specific emotions
        assert EmotionalTone.CALM.value == "calm"
        assert EmotionalTone.STIRRED.value == "stirred"
        assert EmotionalTone.MIXED.value == "mixed"
        assert EmotionalTone.INTENSE.value == "intense"
        assert EmotionalTone.MUTED.value == "muted"

    def test_tendency_hint_values(self):
        """TendencyHint should be hints, not definitive habits"""
        # Values should be tentative, not certain
        assert TendencyHint.NONE_APPARENT.value == "none_apparent"
        assert TendencyHint.SLIGHT_INCLINATION.value == "slight_inclination"
        assert TendencyHint.FORMING_PATTERN.value == "forming_pattern"
        assert TendencyHint.ESTABLISHED_WAY.value == "established_way"

    def test_stability_feeling_values(self):
        """StabilityFeeling should be feelings, not measurements"""
        assert StabilityFeeling.GROUNDED.value == "grounded"
        assert StabilityFeeling.MOSTLY_SETTLED.value == "mostly_settled"
        assert StabilityFeeling.WAVERING.value == "wavering"
        assert StabilityFeeling.TURBULENT.value == "turbulent"

    def test_change_presence_values(self):
        """ChangePresence should be sensed presence, not measured amount"""
        assert ChangePresence.NO_CHANGE_SENSED.value == "no_change_sensed"
        assert ChangePresence.SUBTLE_SHIFT.value == "subtle_shift"
        assert ChangePresence.NOTICEABLE_CHANGE.value == "noticeable_change"
        assert ChangePresence.SIGNIFICANT_SHIFT.value == "significant_shift"

    def test_continuity_feeling_values(self):
        """ContinuityFeeling should be feelings, not scores"""
        assert ContinuityFeeling.CONTINUOUS.value == "continuous"
        assert ContinuityFeeling.MOSTLY_FAMILIAR.value == "mostly_familiar"
        assert ContinuityFeeling.SOMEWHAT_DIFFERENT.value == "somewhat_different"
        assert ContinuityFeeling.DISCONNECTED.value == "disconnected"

    def test_overall_impression_values(self):
        """OverallImpression should be general impressions"""
        assert OverallImpression.SETTLED.value == "settled"
        assert OverallImpression.ACTIVE.value == "active"
        assert OverallImpression.TRANSITIONAL.value == "transitional"
        assert OverallImpression.UNCERTAIN.value == "uncertain"
        assert OverallImpression.CONFLICTED.value == "conflicted"

    def test_no_numeric_values_in_enums(self):
        """All enum values should be strings, not numbers"""
        for enum_class in [EmotionalTone, TendencyHint, StabilityFeeling,
                          ChangePresence, ContinuityFeeling, OverallImpression]:
            for member in enum_class:
                assert isinstance(member.value, str), \
                    f"{enum_class.__name__}.{member.name} has non-string value"


# =============================================================================
# Test ProvisionalSelfImage Structure
# =============================================================================

class TestProvisionalSelfImage:
    """Test ProvisionalSelfImage dataclass"""

    def test_undefined_creation(self):
        """Test creating an undefined image"""
        image = ProvisionalSelfImage.undefined()

        assert image.emotional_tone == EmotionalTone.UNDEFINED
        assert image.overall_impression == OverallImpression.UNDEFINED
        assert not image.is_complete

    def test_has_contradictions_false_when_empty(self):
        """has_contradictions should return False when no contradictions"""
        image = ProvisionalSelfImage.undefined()
        assert not image.has_contradictions()

    def test_has_contradictions_true_when_present(self):
        """has_contradictions should return True when contradictions exist"""
        image = ProvisionalSelfImage(
            emotional_tone=EmotionalTone.CALM,
            tendency_hint=TendencyHint.NONE_APPARENT,
            stability_feeling=StabilityFeeling.GROUNDED,
            change_presence=ChangePresence.NO_CHANGE_SENSED,
            continuity_feeling=ContinuityFeeling.CONTINUOUS,
            overall_impression=OverallImpression.SETTLED,
            emotional_aspect=ImageAspect("emotional", "calm", "Calm"),
            tendency_aspect=ImageAspect("tendency", "none", "None"),
            stability_aspect=ImageAspect("stability", "grounded", "Grounded"),
            change_aspect=ImageAspect("change", "none", "None"),
            continuity_aspect=ImageAspect("continuity", "continuous", "Continuous"),
            contradictions=("Some contradiction",),
            integrated_description="Test",
            timestamp=time.time(),
            is_complete=True,
        )

        assert image.has_contradictions()

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip"""
        original = ProvisionalSelfImage(
            emotional_tone=EmotionalTone.STIRRED,
            tendency_hint=TendencyHint.FORMING_PATTERN,
            stability_feeling=StabilityFeeling.WAVERING,
            change_presence=ChangePresence.NOTICEABLE_CHANGE,
            continuity_feeling=ContinuityFeeling.SOMEWHAT_DIFFERENT,
            overall_impression=OverallImpression.TRANSITIONAL,
            emotional_aspect=ImageAspect("emotional", "stirred", "Something stirring"),
            tendency_aspect=ImageAspect("tendency", "forming", "Pattern forming"),
            stability_aspect=ImageAspect("stability", "wavering", "Some wavering"),
            change_aspect=ImageAspect("change", "noticeable", "Change occurred"),
            continuity_aspect=ImageAspect("continuity", "different", "Somewhat different"),
            contradictions=("Tension A", "Tension B"),
            integrated_description="Test description",
            timestamp=1000.0,
            is_complete=True,
        )

        data = original.to_dict()
        restored = ProvisionalSelfImage.from_dict(data)

        assert restored.emotional_tone == original.emotional_tone
        assert restored.overall_impression == original.overall_impression
        assert restored.contradictions == original.contradictions
        assert restored.integrated_description == original.integrated_description

    def test_get_all_aspects(self):
        """Test getting all aspects"""
        image = ProvisionalSelfImage.undefined()
        aspects = image.get_all_aspects()

        assert len(aspects) == 5
        assert all(isinstance(a, ImageAspect) for a in aspects)


# =============================================================================
# Test Image Generation
# =============================================================================

class TestImageGeneration:
    """Test self-image generation from inputs"""

    def test_generate_with_no_inputs(self, image_system):
        """Should generate undefined image when no inputs"""
        image = image_system.generate_image()

        assert not image.is_complete
        assert image.overall_impression == OverallImpression.UNDEFINED

    def test_generate_with_calm_state(self, image_system, calm_self_state, no_strain_state):
        """Should generate settled image for calm inputs"""
        image = image_system.generate_image(
            self_state_view=calm_self_state,
            strain_state=no_strain_state,
        )

        assert image.emotional_tone == EmotionalTone.CALM
        assert image.stability_feeling == StabilityFeeling.GROUNDED

    def test_generate_with_turbulent_state(self, image_system, turbulent_self_state, high_strain_state):
        """Should generate turbulent image for turbulent inputs"""
        image = image_system.generate_image(
            self_state_view=turbulent_self_state,
            strain_state=high_strain_state,
        )

        assert image.emotional_tone == EmotionalTone.INTENSE
        assert image.stability_feeling == StabilityFeeling.TURBULENT

    def test_generate_with_strong_tendency(self, image_system, calm_self_state, tendency_awareness_strong):
        """Should detect tendency hints from awareness"""
        image = image_system.generate_image(
            self_state_view=calm_self_state,
            tendency_awareness=tendency_awareness_strong,
        )

        assert image.tendency_hint == TendencyHint.ESTABLISHED_WAY

    def test_generate_with_significant_change(self, image_system, calm_self_state, significant_difference_summary):
        """Should detect change presence from difference summary"""
        image = image_system.generate_image(
            self_state_view=calm_self_state,
            difference_summary=significant_difference_summary,
        )

        assert image.change_presence == ChangePresence.SIGNIFICANT_SHIFT

    def test_generate_with_no_change(self, image_system, calm_self_state, no_difference_summary):
        """Should detect no change from no-difference summary"""
        image = image_system.generate_image(
            self_state_view=calm_self_state,
            difference_summary=no_difference_summary,
        )

        assert image.change_presence == ChangePresence.NO_CHANGE_SENSED

    def test_generation_count_increments(self, image_system, calm_self_state):
        """Generation count should increment"""
        assert image_system.get_generation_count() == 0

        image_system.generate_image(self_state_view=calm_self_state)
        assert image_system.get_generation_count() == 1

        image_system.generate_image(self_state_view=calm_self_state)
        assert image_system.get_generation_count() == 2


# =============================================================================
# Test Image Changes with State
# =============================================================================

class TestImageChangesWithState:
    """Test that image changes when state changes"""

    def test_image_changes_with_emotional_change(self, image_system, calm_emotional_view,
                                                   intense_emotional_view, stable_value_view):
        """Image should change when emotional state changes"""
        calm_state = create_self_state_view(calm_emotional_view, stable_value_view)
        intense_state = create_self_state_view(intense_emotional_view, stable_value_view)

        image1 = image_system.generate_image(self_state_view=calm_state)
        image2 = image_system.generate_image(self_state_view=intense_state)

        assert image1.emotional_tone != image2.emotional_tone
        assert image1.emotional_tone == EmotionalTone.CALM
        assert image2.emotional_tone == EmotionalTone.INTENSE

    def test_image_changes_with_stability_change(self, image_system, calm_emotional_view,
                                                   stable_value_view, unstable_value_view):
        """Image should change when stability changes"""
        stable_state = create_self_state_view(calm_emotional_view, stable_value_view)
        unstable_state = create_self_state_view(calm_emotional_view, unstable_value_view)

        image1 = image_system.generate_image(self_state_view=stable_state)
        image2 = image_system.generate_image(self_state_view=unstable_state)

        assert image1.stability_feeling != image2.stability_feeling

    def test_image_changes_with_strain_change(self, image_system, calm_self_state,
                                                no_strain_state, high_strain_state):
        """Image should change when strain changes"""
        image1 = image_system.generate_image(
            self_state_view=calm_self_state,
            strain_state=no_strain_state,
        )
        image2 = image_system.generate_image(
            self_state_view=calm_self_state,
            strain_state=high_strain_state,
        )

        # Strain affects both stability and continuity feelings
        assert image1.continuity_feeling != image2.continuity_feeling


# =============================================================================
# Test Contradictions
# =============================================================================

class TestContradictions:
    """Test that contradictions are allowed to coexist"""

    def test_contradictions_detected(self):
        """Contradictions should be detected"""
        contradictions = _detect_contradictions(
            emotional_tone=EmotionalTone.CALM,
            stability_feeling=StabilityFeeling.TURBULENT,
            change_presence=ChangePresence.NO_CHANGE_SENSED,
            continuity_feeling=ContinuityFeeling.CONTINUOUS,
        )

        assert len(contradictions) > 0
        assert any("calm" in c.lower() and "turbulent" in c.lower() for c in contradictions)

    def test_no_contradictions_for_consistent_state(self):
        """No contradictions for consistent state"""
        contradictions = _detect_contradictions(
            emotional_tone=EmotionalTone.CALM,
            stability_feeling=StabilityFeeling.GROUNDED,
            change_presence=ChangePresence.NO_CHANGE_SENSED,
            continuity_feeling=ContinuityFeeling.CONTINUOUS,
        )

        assert len(contradictions) == 0

    def test_contradictions_preserved_in_image(self, image_system):
        """Contradictions should be preserved in the image"""
        # Create conflicting inputs
        calm_emotional = EmotionalStateView(
            spread=EmotionalSpread.FOCUSED,
            intensity=EmotionalIntensity.CALM,
            harmony=EmotionalHarmony.HARMONIOUS,
            active_emotion_count=1,
            has_coexisting_pairs=False,
            description="Calm",
        )

        unstable_value = ValueStateView(
            stability=ValueStability.UNSTABLE,
            clarity=ValueClarity.EMERGING,
            has_strong_orientations=False,
            is_recently_changed=True,
            description="Unstable",
        )

        conflicting_state = create_self_state_view(calm_emotional, unstable_value)

        # Also add high strain (turbulent stability)
        high_strain = StrainState(
            presence=StrainPresence.PRESENT,
            level=StrainLevel.ALIENATED,
            persistence=StrainPersistence.ONGOING,
            trend=StrainTrend.STABLE,
            timestamp=time.time(),
            last_update_timestamp=time.time(),
            description="High strain",
        )

        image = image_system.generate_image(
            self_state_view=conflicting_state,
            strain_state=high_strain,
        )

        # Should have contradictions (calm but turbulent)
        assert image.has_contradictions()


# =============================================================================
# Test NO Decision Impact
# =============================================================================

class TestNoDecisionImpact:
    """Test that image has STRICTLY NO IMPACT on decision making"""

    def test_image_has_no_decision_values(self, image_system, calm_self_state):
        """ProvisionalSelfImage should not expose decision-impacting values"""
        image = image_system.generate_image(self_state_view=calm_self_state)
        assert verify_no_decision_impact(image)

    def test_system_has_provisional_nature(self, image_system):
        """System should maintain provisional nature"""
        assert verify_provisional_nature(image_system)

    def test_image_no_numeric_scores(self, image_system, calm_self_state):
        """Image should not have numeric scores"""
        image = image_system.generate_image(self_state_view=calm_self_state)

        # Check all public attributes
        for attr in dir(image):
            if attr.startswith('_'):
                continue
            if callable(getattr(image, attr)):
                continue

            value = getattr(image, attr)

            # Timestamps are allowed
            if attr == 'timestamp':
                continue

            # Booleans are allowed (is_complete, etc.)
            if isinstance(value, bool):
                continue

            # No other numeric attributes should exist (besides bool)
            if isinstance(value, (int, float)):
                pytest.fail(f"Found numeric attribute: {attr}={value}")

    def test_system_does_not_modify_external_state(self, image_system):
        """System should not modify any external state"""
        forbidden_patterns = [
            "modify_",
            "update_decision",
            "influence_",
            "bias_",
            "adjust_",
            "apply_to_",
        ]

        methods = [m for m in dir(image_system) if not m.startswith('_')]

        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden_patterns:
                assert pattern not in method_lower, \
                    f"Method {method} suggests external modification"


# =============================================================================
# Test Provisional Nature
# =============================================================================

class TestProvisionalNature:
    """Test that the image is always provisional"""

    def test_image_is_regenerated_each_cycle(self, image_system, calm_self_state):
        """Each generation should create a new image"""
        image1 = image_system.generate_image(self_state_view=calm_self_state)
        # Small delay to ensure different timestamp
        time.sleep(0.001)
        image2 = image_system.generate_image(self_state_view=calm_self_state)

        # Timestamps should be different (even if very close)
        # The key point is that each call generates a NEW image
        assert image1 is not image2  # Different objects
        # Generation count should have increased
        assert image_system.get_generation_count() == 2

    def test_no_persistent_storage_methods(self, image_system):
        """System should not have methods to permanently store the image"""
        forbidden_methods = [
            "save_permanent",
            "fix_image",
            "store_identity",
            "persist_personality",
            "lock_",
        ]

        methods = [m for m in dir(image_system) if not m.startswith('_')]

        for method_name in forbidden_methods:
            assert not any(method_name in m.lower() for m in methods)

    def test_image_is_frozen_dataclass(self):
        """ProvisionalSelfImage should be frozen (immutable)"""
        image = ProvisionalSelfImage.undefined()

        with pytest.raises(Exception):  # FrozenInstanceError
            image.emotional_tone = EmotionalTone.CALM


# =============================================================================
# Test SelfReferenceSystem Integration
# =============================================================================

class TestSelfReferenceIntegration:
    """Test integration with SelfReferenceSystem"""

    def test_generate_tags_incomplete_image(self):
        """Tags for incomplete image should be minimal"""
        image = ProvisionalSelfImage.undefined()
        tags = generate_self_image_tags(image)

        assert len(tags) == 1
        assert tags[0]["category"] == "SELF_IMAGE"
        assert tags[0]["label"] == "image_incomplete"

    def test_generate_tags_complete_image(self, image_system, calm_self_state,
                                           no_strain_state, no_difference_summary):
        """Tags for complete image should include all aspects"""
        # Provide all inputs to get a complete image
        image = image_system.generate_image(
            self_state_view=calm_self_state,
            strain_state=no_strain_state,
            difference_summary=no_difference_summary,
        )
        tags = generate_self_image_tags(image)

        categories = [t["category"] for t in tags]

        assert "SELF_IMAGE_OVERALL" in categories
        assert "SELF_IMAGE_EMOTIONAL" in categories
        assert "SELF_IMAGE_STABILITY" in categories
        assert "SELF_IMAGE_INTEGRATED" in categories

    def test_tags_have_weights(self, image_system, calm_self_state):
        """All tags should have weights"""
        image = image_system.generate_image(self_state_view=calm_self_state)
        tags = generate_self_image_tags(image)

        for tag in tags:
            assert "weight" in tag
            assert isinstance(tag["weight"], float)
            assert tag["weight"] > 0

    def test_tags_scale_factor(self):
        """Tags should respect scale factor"""
        image = ProvisionalSelfImage.undefined()

        tags_default = generate_self_image_tags(image, scale=1.0)
        tags_scaled = generate_self_image_tags(image, scale=2.0)

        assert tags_scaled[0]["weight"] == tags_default[0]["weight"] * 2

    def test_tags_include_contradiction_info(self):
        """Tags should include contradiction info when present"""
        image = ProvisionalSelfImage(
            emotional_tone=EmotionalTone.CALM,
            tendency_hint=TendencyHint.NONE_APPARENT,
            stability_feeling=StabilityFeeling.GROUNDED,
            change_presence=ChangePresence.NO_CHANGE_SENSED,
            continuity_feeling=ContinuityFeeling.CONTINUOUS,
            overall_impression=OverallImpression.SETTLED,
            emotional_aspect=ImageAspect("emotional", "calm", "Calm"),
            tendency_aspect=ImageAspect("tendency", "none", "None"),
            stability_aspect=ImageAspect("stability", "grounded", "Grounded"),
            change_aspect=ImageAspect("change", "none", "None"),
            continuity_aspect=ImageAspect("continuity", "continuous", "Continuous"),
            contradictions=("Some contradiction",),
            integrated_description="Test",
            timestamp=time.time(),
            is_complete=True,
        )

        tags = generate_self_image_tags(image)
        categories = [t["category"] for t in tags]

        assert "SELF_IMAGE_TENSION" in categories


# =============================================================================
# Test Introspection Functions
# =============================================================================

class TestIntrospection:
    """Test introspection support functions"""

    def test_get_self_image_summary(self, image_system, calm_self_state):
        """get_self_image_summary should return readable text"""
        image = image_system.generate_image(self_state_view=calm_self_state)
        summary = get_self_image_summary(image)

        assert "Provisional Self-Image" in summary
        assert "Overall Impression" in summary
        assert "Emotional Tone" in summary

    def test_get_self_image_for_introspection(self, image_system, calm_self_state):
        """get_self_image_for_introspection should return structured data"""
        image = image_system.generate_image(self_state_view=calm_self_state)
        data = get_self_image_for_introspection(image)

        assert "overall_impression" in data
        assert "emotional_tone" in data
        assert "integrated_description" in data
        assert "is_complete" in data


# =============================================================================
# Test Tentative Language
# =============================================================================

class TestTentativeLanguage:
    """Test that descriptions use 'appears to be' language"""

    def test_integrated_description_uses_appears(self, image_system, calm_self_state):
        """Integrated description should use 'appears' language"""
        image = image_system.generate_image(self_state_view=calm_self_state)

        # Should use "appears" not definitive "is"
        desc = image.integrated_description.lower()
        assert "appears" in desc

    def test_aspect_descriptions_tentative(self, image_system, calm_self_state):
        """Aspect descriptions should be tentative"""
        image = image_system.generate_image(self_state_view=calm_self_state)

        for aspect in image.get_all_aspects():
            # Descriptions should not be definitive personality statements
            desc = aspect.description.lower()
            # Should not contain definitive personality words
            definitive_words = ["always", "never", "definitely", "certainly"]
            for word in definitive_words:
                assert word not in desc, f"Found definitive word '{word}' in: {aspect.description}"


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration system"""

    def test_default_config_values(self):
        """Default config should have sensible values"""
        config = SelfImageConfig()

        assert config.include_detailed_descriptions is True
        assert config.detect_contradictions is True
        assert config.use_tentative_language is True

    def test_create_config(self):
        """create_config helper should work"""
        config = create_config(
            include_detailed_descriptions=False,
            detect_contradictions=False,
        )

        assert config.include_detailed_descriptions is False
        assert config.detect_contradictions is False


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_image(self):
        """create_empty_image should return undefined image"""
        image = create_empty_image()
        assert image.overall_impression == OverallImpression.UNDEFINED
        assert not image.is_complete

    def test_partial_inputs(self, image_system, calm_self_state):
        """Should handle partial inputs gracefully"""
        # Only self_state_view
        image = image_system.generate_image(self_state_view=calm_self_state)
        assert image.emotional_tone != EmotionalTone.UNDEFINED

    def test_none_inputs_handled(self, image_system):
        """Should handle all None inputs"""
        image = image_system.generate_image(
            self_state_view=None,
            tendency_awareness=None,
            difference_summary=None,
            strain_state=None,
        )

        assert not image.is_complete

    def test_get_last_image(self, image_system, calm_self_state):
        """get_last_image should return last generated image"""
        assert image_system.get_last_image() is None

        image1 = image_system.generate_image(self_state_view=calm_self_state)
        assert image_system.get_last_image() == image1

        image2 = image_system.generate_image(self_state_view=calm_self_state)
        assert image_system.get_last_image() == image2


# =============================================================================
# Test Philosophy Compliance
# =============================================================================

class TestPhilosophyCompliance:
    """Test that implementation follows the design philosophy"""

    def test_no_personality_definition(self, image_system):
        """System should NOT define personality"""
        methods = [m for m in dir(image_system) if not m.startswith('_')]

        forbidden_patterns = [
            "define_personality",
            "set_identity",
            "establish_character",
            "create_ego",
        ]

        for method in methods:
            for pattern in forbidden_patterns:
                assert pattern not in method.lower()

    def test_no_self_evaluation(self, image_system, calm_self_state):
        """Image should NOT evaluate self"""
        image = image_system.generate_image(self_state_view=calm_self_state)

        evaluative_words = ["good", "bad", "right", "wrong", "should", "must"]

        for aspect in image.get_all_aspects():
            desc = aspect.description.lower()
            for word in evaluative_words:
                assert word not in desc, f"Found evaluative word '{word}'"

    def test_no_consistency_enforcement(self, image_system):
        """System should NOT enforce consistency"""
        methods = [m for m in dir(image_system) if not m.startswith('_')]

        forbidden_patterns = [
            "enforce_consistency",
            "resolve_contradiction",
            "fix_conflict",
            "harmonize",
        ]

        for method in methods:
            for pattern in forbidden_patterns:
                assert pattern not in method.lower()

    def test_contradictions_not_resolved(self):
        """Contradictions should be noted, not resolved"""
        contradictions = _detect_contradictions(
            emotional_tone=EmotionalTone.CALM,
            stability_feeling=StabilityFeeling.TURBULENT,
            change_presence=ChangePresence.NO_CHANGE_SENSED,
            continuity_feeling=ContinuityFeeling.CONTINUOUS,
        )

        # Should detect but not try to resolve
        assert len(contradictions) > 0


# =============================================================================
# Test Integration Logic Functions
# =============================================================================

class TestIntegrationLogicFunctions:
    """Test individual integration logic functions"""

    def test_integrate_emotional_tone_calm(self, calm_emotional_view, stable_value_view):
        """Calm emotional state should produce calm tone"""
        state = create_self_state_view(calm_emotional_view, stable_value_view)
        tone, aspect = _integrate_emotional_tone(state)

        assert tone == EmotionalTone.CALM

    def test_integrate_emotional_tone_none(self):
        """None input should produce undefined tone"""
        tone, aspect = _integrate_emotional_tone(None)

        assert tone == EmotionalTone.UNDEFINED

    def test_integrate_tendency_hint_strong(self, tendency_awareness_strong):
        """Strong awareness should produce established way hint"""
        hint, aspect = _integrate_tendency_hint(None, tendency_awareness_strong)

        assert hint == TendencyHint.ESTABLISHED_WAY

    def test_integrate_stability_with_strain(self, calm_self_state, high_strain_state):
        """High strain should override value stability"""
        feeling, aspect = _integrate_stability_feeling(calm_self_state, high_strain_state)

        assert feeling == StabilityFeeling.TURBULENT

    def test_integrate_change_none(self):
        """None difference should produce undefined"""
        presence, aspect = _integrate_change_presence(None)

        assert presence == ChangePresence.UNDEFINED

    def test_determine_overall_impression_settled(self):
        """Calm + grounded + continuous should be settled"""
        impression = _determine_overall_impression(
            EmotionalTone.CALM,
            TendencyHint.NONE_APPARENT,
            StabilityFeeling.GROUNDED,
            ChangePresence.NO_CHANGE_SENSED,
            ContinuityFeeling.CONTINUOUS,
        )

        assert impression == OverallImpression.SETTLED

    def test_determine_overall_impression_conflicted(self):
        """Mixed emotions should produce conflicted"""
        impression = _determine_overall_impression(
            EmotionalTone.MIXED,
            TendencyHint.NONE_APPARENT,
            StabilityFeeling.GROUNDED,
            ChangePresence.NO_CHANGE_SENSED,
            ContinuityFeeling.CONTINUOUS,
        )

        assert impression == OverallImpression.CONFLICTED

    def test_determine_overall_impression_transitional(self):
        """Significant change should produce transitional"""
        impression = _determine_overall_impression(
            EmotionalTone.CALM,
            TendencyHint.NONE_APPARENT,
            StabilityFeeling.GROUNDED,
            ChangePresence.SIGNIFICANT_SHIFT,
            ContinuityFeeling.CONTINUOUS,
        )

        assert impression == OverallImpression.TRANSITIONAL
