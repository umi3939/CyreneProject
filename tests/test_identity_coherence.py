"""
Tests for Identity Coherence Awareness (自己同一性の揺らぎ認知)

These tests verify that the IdentityCoherenceSystem:
1. Detects overlapping shifts from multiple observation systems
2. Generates coherence states based on shift overlap (NOT single factors)
3. Maintains STRICTLY NO IMPACT on decision making
4. Does NOT define identity or "true self"
5. Has NO self-preservation mechanisms
6. Regenerates state every cycle
"""

import pytest
import time
from typing import Optional

from psyche.identity_coherence import (
    # Enums
    CoherenceLevel,
    ShiftSource,
    OverlapIntensity,
    CoherenceTrend,
    # Data structures
    DetectedShift,
    ShiftOverlap,
    IdentityCoherenceState,
    IdentityCoherenceConfig,
    # System
    IdentityCoherenceSystem,
    # Detection functions
    detect_temporal_difference_shift,
    detect_tendency_change_shift,
    detect_continuity_strain_shift,
    detect_value_instability_shift,
    detect_self_image_flux_shift,
    detect_emotional_turbulence_shift,
    # Level determination
    determine_overlap_intensity,
    determine_coherence_level,
    determine_coherence_trend,
    # Integration
    generate_coherence_tags,
    get_coherence_summary,
    get_coherence_for_introspection,
    # Convenience
    create_config,
    create_empty_state,
    # Verification
    verify_no_decision_impact,
    verify_no_self_preservation,
    verify_no_identity_definition,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockSelfDifferenceSummary:
    """Mock SelfDifferenceSummary for testing"""

    def __init__(
        self,
        has_difference: bool = False,
        magnitude: str = "none",
        nature: str = "stable",
    ):
        from psyche.temporal_self_difference import DifferenceMagnitude, ChangeNature
        self.has_difference = has_difference
        self.magnitude = DifferenceMagnitude(magnitude)
        self.nature = ChangeNature(nature)


class MockTendencyAwarenessItem:
    """Mock TendencyAwarenessItem for testing"""

    def __init__(self, awareness_type: str = "slight_bias"):
        from psyche.tendency_awareness import AwarenessType
        self.awareness_type = AwarenessType(awareness_type)


class MockTendencyAwareness:
    """Mock TendencyAwareness for testing"""

    def __init__(
        self,
        has_awareness: bool = False,
        overall_strength: str = "none",
        items: list = None,
    ):
        from psyche.tendency_awareness import StrengthLevel
        self.has_awareness = has_awareness
        self.overall_strength = StrengthLevel(overall_strength)
        self.items = items if items is not None else []


class MockStrainState:
    """Mock StrainState for testing"""

    def __init__(
        self,
        _is_strained: bool = False,
        level: str = "at_ease",
        persistence: str = "none",
    ):
        from psyche.continuity_strain import StrainLevel, StrainPersistence
        self._is_strained = _is_strained
        self.level = StrainLevel(level)
        self.persistence = StrainPersistence(persistence)

    def is_strained(self):
        return self._is_strained


class MockValueOrientation:
    """Mock ValueOrientation for testing"""

    def __init__(self, _overall_stability: float = 0.5):
        self._overall_stability = _overall_stability

    def get_overall_stability(self):
        return self._overall_stability


class MockProvisionalSelfImage:
    """Mock ProvisionalSelfImage for testing"""

    def __init__(
        self,
        overall_impression: str = "settled",
        stability_feeling: str = "grounded",
        continuity_feeling: str = "continuous",
        emotional_tone: str = "calm",
        _has_contradictions: bool = False,
    ):
        from psyche.self_image_integration import (
            OverallImpression, StabilityFeeling, ContinuityFeeling, EmotionalTone
        )
        self.overall_impression = OverallImpression(overall_impression)
        self.stability_feeling = StabilityFeeling(stability_feeling)
        self.continuity_feeling = ContinuityFeeling(continuity_feeling)
        self.emotional_tone = EmotionalTone(emotional_tone)
        self._has_contradictions = _has_contradictions

    def has_contradictions(self):
        return self._has_contradictions


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def system():
    """Create a fresh IdentityCoherenceSystem"""
    return IdentityCoherenceSystem()


@pytest.fixture
def config():
    """Create a default config"""
    return IdentityCoherenceConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing"""
    return create_config(
        min_shifts_for_slightly_shifting=2,
        min_shifts_for_unsettled=3,
        min_shifts_for_disconnected=4,
    )


# =============================================================================
# Test: Basic Structures
# =============================================================================

class TestCoherenceEnums:
    """Test coherence-related enums"""

    def test_coherence_level_values(self):
        """All coherence levels should have the expected values"""
        assert CoherenceLevel.STABLE.value == "stable"
        assert CoherenceLevel.SLIGHTLY_SHIFTING.value == "slightly_shifting"
        assert CoherenceLevel.UNSETTLED.value == "unsettled"
        assert CoherenceLevel.DISCONNECTED.value == "disconnected"
        assert CoherenceLevel.UNDEFINED.value == "undefined"

    def test_shift_source_values(self):
        """All shift sources should have the expected values"""
        assert ShiftSource.TEMPORAL_DIFFERENCE.value == "temporal_difference"
        assert ShiftSource.TENDENCY_CHANGE.value == "tendency_change"
        assert ShiftSource.CONTINUITY_STRAIN.value == "continuity_strain"
        assert ShiftSource.VALUE_INSTABILITY.value == "value_instability"
        assert ShiftSource.SELF_IMAGE_FLUX.value == "self_image_flux"
        assert ShiftSource.EMOTIONAL_TURBULENCE.value == "emotional_turbulence"

    def test_overlap_intensity_values(self):
        """All overlap intensities should have the expected values"""
        assert OverlapIntensity.NONE.value == "none"
        assert OverlapIntensity.MINIMAL.value == "minimal"
        assert OverlapIntensity.MODERATE.value == "moderate"
        assert OverlapIntensity.SIGNIFICANT.value == "significant"

    def test_coherence_trend_values(self):
        """All coherence trends should have the expected values"""
        assert CoherenceTrend.STABLE.value == "stable"
        assert CoherenceTrend.CONVERGING.value == "converging"
        assert CoherenceTrend.DIVERGING.value == "diverging"
        assert CoherenceTrend.FLUCTUATING.value == "fluctuating"


class TestDetectedShift:
    """Test DetectedShift structure"""

    def test_create_active_shift(self):
        """Active shift should have is_active=True"""
        shift = DetectedShift(
            source=ShiftSource.TEMPORAL_DIFFERENCE,
            is_active=True,
            description="Test shift",
        )
        assert shift.is_active
        assert shift.source == ShiftSource.TEMPORAL_DIFFERENCE

    def test_create_inactive_shift(self):
        """Inactive factory method should create is_active=False"""
        shift = DetectedShift.inactive(ShiftSource.CONTINUITY_STRAIN)
        assert not shift.is_active
        assert shift.source == ShiftSource.CONTINUITY_STRAIN


class TestShiftOverlap:
    """Test ShiftOverlap structure"""

    def test_empty_overlap(self):
        """Empty overlap should have no active shifts"""
        overlap = ShiftOverlap.none()
        assert overlap.active_count == 0
        assert overlap.intensity == OverlapIntensity.NONE
        assert len(overlap.get_active_shifts()) == 0

    def test_overlap_with_active_shifts(self):
        """Overlap with active shifts should report them correctly"""
        shifts = (
            DetectedShift(ShiftSource.TEMPORAL_DIFFERENCE, True, "desc1"),
            DetectedShift(ShiftSource.CONTINUITY_STRAIN, True, "desc2"),
            DetectedShift(ShiftSource.VALUE_INSTABILITY, False, "desc3"),
        )
        overlap = ShiftOverlap(
            detected_shifts=shifts,
            active_count=2,
            intensity=OverlapIntensity.MODERATE,
            overlap_description="Test",
        )
        assert len(overlap.get_active_shifts()) == 2
        assert ShiftSource.TEMPORAL_DIFFERENCE in overlap.get_active_sources()
        assert ShiftSource.CONTINUITY_STRAIN in overlap.get_active_sources()
        assert ShiftSource.VALUE_INSTABILITY not in overlap.get_active_sources()


class TestIdentityCoherenceState:
    """Test IdentityCoherenceState structure"""

    def test_stable_factory(self):
        """Stable factory should create stable state"""
        state = IdentityCoherenceState.stable()
        assert state.level == CoherenceLevel.STABLE
        assert state.is_coherent()
        assert not state.is_incoherent()

    def test_undefined_factory(self):
        """Undefined factory should create undefined state"""
        state = IdentityCoherenceState.undefined()
        assert state.level == CoherenceLevel.UNDEFINED

    def test_is_coherent(self):
        """is_coherent should return True for stable/slightly_shifting"""
        stable_state = IdentityCoherenceState.stable()
        assert stable_state.is_coherent()

    def test_is_incoherent(self):
        """is_incoherent should return True for unsettled/disconnected"""
        state = IdentityCoherenceState(
            level=CoherenceLevel.UNSETTLED,
            shift_overlap=ShiftOverlap.none(),
            trend=CoherenceTrend.STABLE,
            timestamp=time.time(),
            generation_count=1,
            description="Test",
        )
        assert state.is_incoherent()
        assert not state.is_coherent()

    def test_serialization(self):
        """State should serialize and deserialize correctly"""
        state = IdentityCoherenceState.stable()
        data = state.to_dict()
        restored = IdentityCoherenceState.from_dict(data)
        assert restored.level == state.level
        assert restored.description == state.description


# =============================================================================
# Test: Shift Detection Functions
# =============================================================================

class TestTemporalDifferenceShiftDetection:
    """Test detection of shifts from temporal difference"""

    def test_no_difference_no_shift(self):
        """No difference should not cause a shift"""
        summary = MockSelfDifferenceSummary(
            has_difference=False,
            magnitude="none",
            nature="stable",
        )
        shift = detect_temporal_difference_shift(summary)
        assert not shift.is_active

    def test_minimal_difference_no_shift(self):
        """Minimal difference should not cause a shift"""
        summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="minimal",
            nature="fluctuating",
        )
        shift = detect_temporal_difference_shift(summary)
        assert not shift.is_active

    def test_significant_shifting_causes_shift(self):
        """Significant + shifting should cause a shift"""
        summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        shift = detect_temporal_difference_shift(summary)
        assert shift.is_active
        assert shift.source == ShiftSource.TEMPORAL_DIFFERENCE

    def test_substantial_transformed_causes_shift(self):
        """Substantial + transformed should cause a shift"""
        summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="substantial",
            nature="transformed",
        )
        shift = detect_temporal_difference_shift(summary)
        assert shift.is_active

    def test_fluctuating_nature_no_shift(self):
        """Fluctuating nature should not cause shift (temporary)"""
        summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="noticeable",
            nature="fluctuating",
        )
        shift = detect_temporal_difference_shift(summary)
        assert not shift.is_active


class TestContinuityStrainShiftDetection:
    """Test detection of shifts from continuity strain"""

    def test_no_strain_no_shift(self):
        """No strain should not cause a shift"""
        strain = MockStrainState(_is_strained=False)
        shift = detect_continuity_strain_shift(strain)
        assert not shift.is_active

    def test_momentary_strain_no_shift(self):
        """Momentary strain should not cause a shift"""
        strain = MockStrainState(
            _is_strained=True,
            level="unsettled",
            persistence="momentary",
        )
        shift = detect_continuity_strain_shift(strain)
        assert not shift.is_active

    def test_ongoing_dissonant_causes_shift(self):
        """Ongoing + dissonant should cause a shift"""
        strain = MockStrainState(
            _is_strained=True,
            level="dissonant",
            persistence="ongoing",
        )
        shift = detect_continuity_strain_shift(strain)
        assert shift.is_active

    def test_chronic_alienated_causes_shift(self):
        """Chronic + alienated should cause a shift"""
        strain = MockStrainState(
            _is_strained=True,
            level="alienated",
            persistence="chronic",
        )
        shift = detect_continuity_strain_shift(strain)
        assert shift.is_active


class TestValueInstabilityShiftDetection:
    """Test detection of shifts from value instability"""

    def test_stable_values_no_shift(self):
        """Stable values should not cause a shift"""
        config = IdentityCoherenceConfig()
        orientation = MockValueOrientation(_overall_stability=0.5)
        shift = detect_value_instability_shift(orientation, config)
        assert not shift.is_active

    def test_unstable_values_causes_shift(self):
        """Unstable values should cause a shift"""
        config = IdentityCoherenceConfig(value_stability_threshold=0.3)
        orientation = MockValueOrientation(_overall_stability=0.1)
        shift = detect_value_instability_shift(orientation, config)
        assert shift.is_active


class TestSelfImageFluxShiftDetection:
    """Test detection of shifts from self-image flux"""

    def test_settled_image_no_shift(self):
        """Settled self-image should not cause a shift"""
        image = MockProvisionalSelfImage(
            overall_impression="settled",
            stability_feeling="grounded",
            continuity_feeling="continuous",
        )
        shift = detect_self_image_flux_shift(image)
        assert not shift.is_active

    def test_transitional_wavering_causes_shift(self):
        """Transitional + wavering should cause a shift"""
        image = MockProvisionalSelfImage(
            overall_impression="transitional",
            stability_feeling="wavering",
            continuity_feeling="continuous",
        )
        shift = detect_self_image_flux_shift(image)
        assert shift.is_active

    def test_single_indicator_no_shift(self):
        """Single indicator should not cause a shift"""
        image = MockProvisionalSelfImage(
            overall_impression="transitional",
            stability_feeling="grounded",
            continuity_feeling="continuous",
        )
        shift = detect_self_image_flux_shift(image)
        assert not shift.is_active


class TestEmotionalTurbulenceShiftDetection:
    """Test detection of shifts from emotional turbulence"""

    def test_calm_emotion_no_shift(self):
        """Calm emotion should not cause a shift"""
        image = MockProvisionalSelfImage(
            emotional_tone="calm",
            stability_feeling="grounded",
        )
        shift = detect_emotional_turbulence_shift(image)
        assert not shift.is_active

    def test_intense_but_stable_no_shift(self):
        """Intense emotion alone should NOT cause a shift (temporary)"""
        image = MockProvisionalSelfImage(
            emotional_tone="intense",
            stability_feeling="grounded",
        )
        shift = detect_emotional_turbulence_shift(image)
        assert not shift.is_active  # Key: temporary emotions don't cause shifts

    def test_intense_and_turbulent_causes_shift(self):
        """Intense emotion + turbulent stability should cause a shift"""
        image = MockProvisionalSelfImage(
            emotional_tone="intense",
            stability_feeling="turbulent",
        )
        shift = detect_emotional_turbulence_shift(image)
        assert shift.is_active


# =============================================================================
# Test: Coherence Level Determination
# =============================================================================

class TestCoherenceLevelDetermination:
    """Test determination of coherence level from shift overlap"""

    def test_no_shifts_stable(self, config):
        """Zero shifts should result in STABLE"""
        level = determine_coherence_level(0, config)
        assert level == CoherenceLevel.STABLE

    def test_single_shift_stable(self, config):
        """Single shift should NOT affect coherence (per design doc)"""
        level = determine_coherence_level(1, config)
        assert level == CoherenceLevel.STABLE

    def test_two_shifts_slightly_shifting(self, config):
        """Two shifts should result in SLIGHTLY_SHIFTING"""
        level = determine_coherence_level(2, config)
        assert level == CoherenceLevel.SLIGHTLY_SHIFTING

    def test_three_shifts_unsettled(self, config):
        """Three shifts should result in UNSETTLED"""
        level = determine_coherence_level(3, config)
        assert level == CoherenceLevel.UNSETTLED

    def test_four_shifts_disconnected(self, config):
        """Four or more shifts should result in DISCONNECTED"""
        level = determine_coherence_level(4, config)
        assert level == CoherenceLevel.DISCONNECTED


class TestOverlapIntensityDetermination:
    """Test determination of overlap intensity"""

    def test_no_shifts_none(self, config):
        """Zero shifts should have NONE intensity"""
        intensity = determine_overlap_intensity(0, config)
        assert intensity == OverlapIntensity.NONE

    def test_one_shift_minimal(self, config):
        """One shift should have MINIMAL intensity"""
        intensity = determine_overlap_intensity(1, config)
        assert intensity == OverlapIntensity.MINIMAL

    def test_two_shifts_moderate(self, config):
        """Two shifts should have MODERATE intensity"""
        intensity = determine_overlap_intensity(2, config)
        assert intensity == OverlapIntensity.MODERATE

    def test_three_shifts_significant(self, config):
        """Three+ shifts should have SIGNIFICANT intensity"""
        intensity = determine_overlap_intensity(3, config)
        assert intensity == OverlapIntensity.SIGNIFICANT


# =============================================================================
# Test: System Integration
# =============================================================================

class TestIdentityCoherenceSystem:
    """Test the main IdentityCoherenceSystem"""

    def test_create_system(self, system):
        """System should be creatable"""
        assert system.get_generation_count() == 0
        assert system.get_last_state() is None

    def test_generate_with_no_inputs(self, system):
        """No inputs should produce undefined state"""
        state = system.generate_state()
        assert state.level == CoherenceLevel.UNDEFINED

    def test_generate_with_stable_inputs(self, system):
        """Stable inputs should produce stable state"""
        self_image = MockProvisionalSelfImage()
        strain_state = MockStrainState()
        value_orientation = MockValueOrientation()

        state = system.generate_state(
            self_image=self_image,
            strain_state=strain_state,
            value_orientation=value_orientation,
        )
        assert state.level == CoherenceLevel.STABLE
        assert state.is_coherent()

    def test_generate_with_overlapping_shifts(self, system):
        """Overlapping shifts should affect coherence"""
        # Create inputs with multiple active shifts
        diff_summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        strain_state = MockStrainState(
            _is_strained=True,
            level="dissonant",
            persistence="ongoing",
        )
        value_orientation = MockValueOrientation(_overall_stability=0.1)

        state = system.generate_state(
            difference_summary=diff_summary,
            strain_state=strain_state,
            value_orientation=value_orientation,
        )

        # Should have multiple shifts and affect coherence
        assert state.shift_overlap.active_count >= 2
        assert state.level != CoherenceLevel.STABLE

    def test_single_shift_does_not_affect_coherence(self, system):
        """A SINGLE shift should NOT affect coherence (per design doc)"""
        # Only one active shift source
        diff_summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )

        state = system.generate_state(
            difference_summary=diff_summary,
        )

        # Should still be STABLE because only one shift
        assert state.level == CoherenceLevel.STABLE
        assert state.shift_overlap.active_count == 1

    def test_generation_count_increments(self, system):
        """Generation count should increment with each call"""
        assert system.get_generation_count() == 0

        system.generate_state()
        assert system.get_generation_count() == 1

        system.generate_state()
        assert system.get_generation_count() == 2

    def test_state_is_regenerated_each_time(self, system):
        """State should be regenerated fresh each call"""
        state1 = system.generate_state()
        state2 = system.generate_state()

        assert state1 is not state2
        assert state1.generation_count != state2.generation_count

    def test_last_state_is_tracked(self, system):
        """Last state should be accessible"""
        state = system.generate_state()
        assert system.get_last_state() is state


class TestCoherenceTrendDetection:
    """Test trend detection over multiple observations"""

    def test_stable_trend(self):
        """Consistent levels should produce STABLE trend"""
        history = [CoherenceLevel.STABLE] * 5
        config = IdentityCoherenceConfig()
        trend = determine_coherence_trend(history, config)
        assert trend == CoherenceTrend.STABLE

    def test_diverging_trend(self):
        """Increasing levels should produce DIVERGING trend"""
        history = [
            CoherenceLevel.STABLE,
            CoherenceLevel.SLIGHTLY_SHIFTING,
            CoherenceLevel.UNSETTLED,
        ]
        config = IdentityCoherenceConfig()
        trend = determine_coherence_trend(history, config)
        assert trend == CoherenceTrend.DIVERGING

    def test_converging_trend(self):
        """Decreasing levels should produce CONVERGING trend"""
        history = [
            CoherenceLevel.UNSETTLED,
            CoherenceLevel.SLIGHTLY_SHIFTING,
            CoherenceLevel.STABLE,
        ]
        config = IdentityCoherenceConfig()
        trend = determine_coherence_trend(history, config)
        assert trend == CoherenceTrend.CONVERGING

    def test_insufficient_history(self):
        """Insufficient history should produce UNDEFINED trend"""
        history = [CoherenceLevel.STABLE]
        config = IdentityCoherenceConfig()
        trend = determine_coherence_trend(history, config)
        assert trend == CoherenceTrend.UNDEFINED


# =============================================================================
# Test: No Decision Impact (Critical Constraint)
# =============================================================================

class TestNoDecisionImpact:
    """Test that the system has NO impact on decision making"""

    def test_state_has_no_decision_impact(self):
        """State should pass no-decision-impact verification"""
        state = IdentityCoherenceState.stable()
        assert verify_no_decision_impact(state)

    def test_incoherent_state_has_no_decision_impact(self):
        """Even incoherent state should have no decision impact"""
        state = IdentityCoherenceState(
            level=CoherenceLevel.DISCONNECTED,
            shift_overlap=ShiftOverlap.none(),
            trend=CoherenceTrend.DIVERGING,
            timestamp=time.time(),
            generation_count=1,
            description="Test",
        )
        assert verify_no_decision_impact(state)

    def test_state_contains_only_abstract_values(self):
        """State should only contain enums, strings, timestamps"""
        state = IdentityCoherenceState.stable()
        data = state.to_dict()

        # Check all values are abstract (strings, numbers for metadata only)
        assert isinstance(data["level"], str)
        assert isinstance(data["description"], str)
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["generation_count"], int)

    def test_tags_are_for_introspection_only(self):
        """Generated tags should be for introspection only"""
        state = IdentityCoherenceState.stable()
        tags = generate_coherence_tags(state)

        for tag in tags:
            # All tags should be categorized as identity coherence (introspection)
            assert "IDENTITY_COHERENCE" in tag["category"]
            # Weights should be small (not decision-driving)
            assert tag["weight"] <= 0.15


# =============================================================================
# Test: No Self-Preservation (Critical Constraint)
# =============================================================================

class TestNoSelfPreservation:
    """Test that the system has NO self-preservation mechanisms"""

    def test_system_has_no_preservation_methods(self, system):
        """System should pass no-self-preservation verification"""
        assert verify_no_self_preservation(system)

    def test_no_fix_methods(self, system):
        """System should have no 'fix' methods"""
        methods = [m for m in dir(system) if not m.startswith('_')]
        fix_methods = [m for m in methods if 'fix' in m.lower()]
        assert len(fix_methods) == 0

    def test_no_repair_methods(self, system):
        """System should have no 'repair' methods"""
        methods = [m for m in dir(system) if not m.startswith('_')]
        repair_methods = [m for m in methods if 'repair' in m.lower()]
        assert len(repair_methods) == 0

    def test_no_restore_methods(self, system):
        """System should have no 'restore' methods"""
        methods = [m for m in dir(system) if not m.startswith('_')]
        restore_methods = [m for m in methods if 'restore' in m.lower()]
        assert len(restore_methods) == 0

    def test_no_protect_methods(self, system):
        """System should have no 'protect' methods"""
        methods = [m for m in dir(system) if not m.startswith('_')]
        protect_methods = [m for m in methods if 'protect' in m.lower()]
        assert len(protect_methods) == 0


# =============================================================================
# Test: No Identity Definition (Critical Constraint)
# =============================================================================

class TestNoIdentityDefinition:
    """Test that the system does NOT define identity"""

    def test_stable_state_no_identity_definition(self):
        """Stable state should not define identity"""
        state = IdentityCoherenceState.stable()
        assert verify_no_identity_definition(state)

    def test_incoherent_state_no_identity_definition(self):
        """Even incoherent state should not define identity"""
        state = IdentityCoherenceState(
            level=CoherenceLevel.DISCONNECTED,
            shift_overlap=ShiftOverlap.none(),
            trend=CoherenceTrend.DIVERGING,
            timestamp=time.time(),
            generation_count=1,
            description="Self feels disconnected from earlier self.",
        )
        assert verify_no_identity_definition(state)

    def test_description_does_not_define_true_self(self):
        """Description should not reference 'true self'"""
        state = IdentityCoherenceState.stable()
        assert "true self" not in state.description.lower()
        assert "real self" not in state.description.lower()

    def test_no_should_be_statements(self, system):
        """Generated descriptions should not say what self 'should be'"""
        # Generate various states
        stable_state = system.generate_state()

        diff_summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        strain_state = MockStrainState(
            _is_strained=True,
            level="dissonant",
            persistence="ongoing",
        )

        incoherent_state = system.generate_state(
            difference_summary=diff_summary,
            strain_state=strain_state,
        )

        # Check descriptions
        for state in [stable_state, incoherent_state]:
            assert "should be" not in state.description.lower()
            assert "must be" not in state.description.lower()


# =============================================================================
# Test: State Regeneration (Critical Constraint)
# =============================================================================

class TestStateRegeneration:
    """Test that state is regenerated each cycle"""

    def test_state_is_not_cached(self, system):
        """State should not be cached between calls"""
        state1 = system.generate_state()
        state2 = system.generate_state()

        # Different objects
        assert state1 is not state2

    def test_state_responds_to_input_changes(self, system):
        """State should respond to input changes"""
        # First: stable inputs
        state1 = system.generate_state(
            strain_state=MockStrainState(),
        )
        assert state1.level == CoherenceLevel.STABLE

        # Second: unstable inputs
        diff_summary = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        strain_state = MockStrainState(
            _is_strained=True,
            level="dissonant",
            persistence="ongoing",
        )

        state2 = system.generate_state(
            difference_summary=diff_summary,
            strain_state=strain_state,
        )

        # State should have changed
        assert state2.shift_overlap.active_count > state1.shift_overlap.active_count

    def test_no_fix_operations_exist(self, system):
        """There should be no operations to 'fix' or 'maintain' coherence"""
        methods = [m for m in dir(system) if not m.startswith('_') and callable(getattr(system, m))]

        fix_patterns = ['fix', 'repair', 'restore', 'maintain', 'force']
        for method in methods:
            for pattern in fix_patterns:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"


# =============================================================================
# Test: Integration with SelfReferenceSystem
# =============================================================================

class TestSelfReferenceIntegration:
    """Test integration with SelfReferenceSystem"""

    def test_generate_tags_for_stable_state(self):
        """Stable state should generate appropriate tags"""
        state = IdentityCoherenceState.stable()
        tags = generate_coherence_tags(state)

        assert len(tags) == 1
        assert tags[0]["label"] == "coherent_self"

    def test_generate_tags_for_incoherent_state(self):
        """Incoherent state should generate multiple tags"""
        shifts = (
            DetectedShift(ShiftSource.TEMPORAL_DIFFERENCE, True, "desc1"),
            DetectedShift(ShiftSource.CONTINUITY_STRAIN, True, "desc2"),
        )
        overlap = ShiftOverlap(
            detected_shifts=shifts,
            active_count=2,
            intensity=OverlapIntensity.MODERATE,
            overlap_description="Test overlap",
        )
        state = IdentityCoherenceState(
            level=CoherenceLevel.SLIGHTLY_SHIFTING,
            shift_overlap=overlap,
            trend=CoherenceTrend.DIVERGING,
            timestamp=time.time(),
            generation_count=1,
            description="Test",
        )

        tags = generate_coherence_tags(state)

        # Should have level, overlap, trend, shift, and integrated tags
        assert len(tags) > 1
        categories = [t["category"] for t in tags]
        assert any("LEVEL" in c for c in categories)
        assert any("OVERLAP" in c for c in categories)

    def test_get_coherence_summary(self):
        """Summary should be human-readable"""
        state = IdentityCoherenceState.stable()
        summary = get_coherence_summary(state)

        assert "Identity Coherence" in summary
        assert state.level.value in summary

    def test_get_coherence_for_introspection(self):
        """Introspection data should be structured correctly"""
        state = IdentityCoherenceState.stable()
        data = get_coherence_for_introspection(state)

        assert "is_coherent" in data
        assert "level" in data
        assert "description" in data
        assert data["is_coherent"] is True


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration options"""

    def test_default_config(self):
        """Default config should have sensible values"""
        config = IdentityCoherenceConfig()
        assert config.min_shifts_for_slightly_shifting == 2
        assert config.min_shifts_for_unsettled == 3
        assert config.min_shifts_for_disconnected == 4

    def test_custom_config(self):
        """Custom config should be usable"""
        config = create_config(
            min_shifts_for_slightly_shifting=3,
            min_shifts_for_unsettled=4,
            min_shifts_for_disconnected=5,
        )
        assert config.min_shifts_for_slightly_shifting == 3
        assert config.min_shifts_for_unsettled == 4
        assert config.min_shifts_for_disconnected == 5

    def test_config_affects_level_determination(self):
        """Config should affect level determination"""
        default_config = IdentityCoherenceConfig()
        strict_config = create_config(
            min_shifts_for_slightly_shifting=4,
            min_shifts_for_unsettled=5,
        )

        # With 2 shifts: default should be SLIGHTLY_SHIFTING
        assert determine_coherence_level(2, default_config) == CoherenceLevel.SLIGHTLY_SHIFTING

        # With 2 shifts: strict should still be STABLE
        assert determine_coherence_level(2, strict_config) == CoherenceLevel.STABLE


# =============================================================================
# Test: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_empty_state(self):
        """create_empty_state should create stable state"""
        state = create_empty_state()
        assert state.level == CoherenceLevel.STABLE
        assert state.is_coherent()

    def test_create_config(self):
        """create_config should create valid config"""
        config = create_config()
        assert isinstance(config, IdentityCoherenceConfig)
