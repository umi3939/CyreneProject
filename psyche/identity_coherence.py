"""
Identity Coherence Awareness (自己同一性の揺らぎ認知)

This module provides awareness of whether the self feels continuous or shifting.
It observes multiple observation systems to detect "overlap of shifts" and
generates a sense of identity coherence.

CRITICAL DESIGN PRINCIPLE:
- This is NOT self-defense or self-repair
- This is NOT defining "true self"
- STRICTLY NO IMPACT on decision making
- Just observes whether current self feels continuous with earlier self
- Regenerated every cycle - never fixed

Philosophy:
自己とは固定された定義ではなく、
連続性の中で「まだ自分だと感じられるか」という感覚である。

本機能は、自己防衛でも、自己修復でもない。
ましてや「正しい自分」を定義するものでもない。

ただ、「今の自分は、さっきまでの自分と
ちゃんと同じ場所に立っているように感じるか」
という感覚を観測するだけである。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# =============================================================================
# Abstract Enums for Coherence Description
# These describe the FEELING without judgment or evaluation
# =============================================================================

class CoherenceLevel(Enum):
    """
    Level of identity coherence felt.

    NOT evaluative (no good/bad):
    - STABLE: Self feels continuous, standing in the same place
    - SLIGHTLY_SHIFTING: Something feels a bit off
    - UNSETTLED: Difficult to grasp sense of self
    - DISCONNECTED: Feels distant from earlier self

    These are NOT judgments. All states are valid.
    """
    STABLE = "stable"                    # Self feels continuous
    SLIGHTLY_SHIFTING = "slightly_shifting"  # Slight sense of displacement
    UNSETTLED = "unsettled"              # Difficult to grasp self-identity
    DISCONNECTED = "disconnected"        # Feels separate from earlier self
    UNDEFINED = "undefined"              # Cannot determine


class ShiftSource(Enum):
    """
    Source of detected shift in identity coherence.

    These are observational - they indicate WHAT is shifting,
    not WHY or whether it's good/bad.
    """
    TEMPORAL_DIFFERENCE = "temporal_difference"  # Self-state has changed over time
    TENDENCY_CHANGE = "tendency_change"          # Behavioral tendencies are shifting
    CONTINUITY_STRAIN = "continuity_strain"      # Strain from persistent difference
    VALUE_INSTABILITY = "value_instability"      # Value orientation is unstable
    SELF_IMAGE_FLUX = "self_image_flux"          # Self-image is in flux
    EMOTIONAL_TURBULENCE = "emotional_turbulence"  # Emotional state is turbulent


class OverlapIntensity(Enum):
    """
    Intensity of shift overlap.

    Coherence is affected when MULTIPLE shifts overlap,
    not by single factors alone.
    """
    NONE = "none"                # No significant overlap
    MINIMAL = "minimal"          # One or two minor shifts
    MODERATE = "moderate"        # Multiple shifts overlapping
    SIGNIFICANT = "significant"  # Strong overlap of multiple shifts
    UNDEFINED = "undefined"


class CoherenceTrend(Enum):
    """
    Direction the coherence state is moving.
    NOT evaluative - just observational.
    """
    STABLE = "stable"            # Coherence level not changing
    CONVERGING = "converging"    # Shifts are settling down
    DIVERGING = "diverging"      # More shifts appearing
    FLUCTUATING = "fluctuating"  # Oscillating state
    UNDEFINED = "undefined"


# =============================================================================
# Shift Detection Structures
# =============================================================================

@dataclass(frozen=True)
class DetectedShift:
    """
    A single detected shift affecting identity coherence.

    This is observational data - no judgment or evaluation.
    """
    source: ShiftSource
    is_active: bool
    description: str

    @classmethod
    def inactive(cls, source: ShiftSource) -> DetectedShift:
        """Create an inactive shift (no shift detected from this source)"""
        return cls(
            source=source,
            is_active=False,
            description=f"No shift detected from {source.value}",
        )


@dataclass(frozen=True)
class ShiftOverlap:
    """
    Summary of overlapping shifts.

    Identity coherence is affected by the OVERLAP of multiple shifts,
    not by single factors alone.
    """
    detected_shifts: tuple[DetectedShift, ...]
    active_count: int
    intensity: OverlapIntensity
    overlap_description: str

    @classmethod
    def none(cls) -> ShiftOverlap:
        """Create an empty overlap (no shifts)"""
        return cls(
            detected_shifts=(),
            active_count=0,
            intensity=OverlapIntensity.NONE,
            overlap_description="No overlapping shifts detected.",
        )

    def get_active_shifts(self) -> list[DetectedShift]:
        """Get list of active shifts"""
        return [s for s in self.detected_shifts if s.is_active]

    def get_active_sources(self) -> list[ShiftSource]:
        """Get list of active shift sources"""
        return [s.source for s in self.detected_shifts if s.is_active]


# =============================================================================
# Identity Coherence State Structure
# =============================================================================

@dataclass(frozen=True)
class IdentityCoherenceState:
    """
    The current state of identity coherence awareness.

    This structure:
    - Does NOT contain raw numbers or scores (externally)
    - Does NOT evaluate coherence as good/bad
    - Is for awareness/introspection only
    - Has NO influence on decisions

    This represents:
    「今の自分は、さっきまでの自分とちゃんと同じ場所に立っているか」
    という感覚
    """
    level: CoherenceLevel
    shift_overlap: ShiftOverlap
    trend: CoherenceTrend

    # Metadata
    timestamp: float
    generation_count: int

    # Human-readable description (for introspection only)
    description: str

    @classmethod
    def stable(cls, timestamp: Optional[float] = None, generation_count: int = 0) -> IdentityCoherenceState:
        """Create a stable state (self feels continuous)"""
        ts = timestamp or time.time()
        return cls(
            level=CoherenceLevel.STABLE,
            shift_overlap=ShiftOverlap.none(),
            trend=CoherenceTrend.STABLE,
            timestamp=ts,
            generation_count=generation_count,
            description="Self feels continuous and coherent.",
        )

    @classmethod
    def undefined(cls, timestamp: Optional[float] = None, generation_count: int = 0) -> IdentityCoherenceState:
        """Create an undefined state"""
        ts = timestamp or time.time()
        return cls(
            level=CoherenceLevel.UNDEFINED,
            shift_overlap=ShiftOverlap.none(),
            trend=CoherenceTrend.UNDEFINED,
            timestamp=ts,
            generation_count=generation_count,
            description="Identity coherence state cannot be determined.",
        )

    def is_coherent(self) -> bool:
        """Check if identity feels coherent (stable or only slightly shifting)"""
        return self.level in (CoherenceLevel.STABLE, CoherenceLevel.SLIGHTLY_SHIFTING)

    def is_incoherent(self) -> bool:
        """Check if identity feels incoherent (unsettled or disconnected)"""
        return self.level in (CoherenceLevel.UNSETTLED, CoherenceLevel.DISCONNECTED)

    def has_active_shifts(self) -> bool:
        """Check if there are any active shifts"""
        return self.shift_overlap.active_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "level": self.level.value,
            "shift_overlap": {
                "detected_shifts": [
                    {
                        "source": s.source.value,
                        "is_active": s.is_active,
                        "description": s.description,
                    }
                    for s in self.shift_overlap.detected_shifts
                ],
                "active_count": self.shift_overlap.active_count,
                "intensity": self.shift_overlap.intensity.value,
                "overlap_description": self.shift_overlap.overlap_description,
            },
            "trend": self.trend.value,
            "timestamp": self.timestamp,
            "generation_count": self.generation_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityCoherenceState:
        """Create from dictionary"""
        overlap_data = data.get("shift_overlap", {})
        detected_shifts = tuple(
            DetectedShift(
                source=ShiftSource(s["source"]),
                is_active=s["is_active"],
                description=s["description"],
            )
            for s in overlap_data.get("detected_shifts", [])
        )

        shift_overlap = ShiftOverlap(
            detected_shifts=detected_shifts,
            active_count=overlap_data.get("active_count", 0),
            intensity=OverlapIntensity(overlap_data.get("intensity", "none")),
            overlap_description=overlap_data.get("overlap_description", ""),
        )

        return cls(
            level=CoherenceLevel(data["level"]),
            shift_overlap=shift_overlap,
            trend=CoherenceTrend(data.get("trend", "undefined")),
            timestamp=data["timestamp"],
            generation_count=data.get("generation_count", 0),
            description=data["description"],
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class IdentityCoherenceConfig:
    """
    Configuration for identity coherence detection.

    These parameters determine when shifts are considered significant
    enough to affect coherence. They do NOT affect decisions.
    """
    # Minimum number of active shifts required for each level
    # Single shifts do NOT affect coherence (per design document)
    min_shifts_for_slightly_shifting: int = 2
    min_shifts_for_unsettled: int = 3
    min_shifts_for_disconnected: int = 4

    # Value stability threshold (from ValueOrientation.get_overall_stability())
    # Below this threshold, value orientation is considered "unstable"
    value_stability_threshold: float = 0.3

    # Thresholds for overlap intensity
    minimal_overlap_count: int = 1
    moderate_overlap_count: int = 2
    significant_overlap_count: int = 3

    # Trend detection window (number of states to consider)
    trend_lookback: int = 5


# =============================================================================
# Shift Detection Functions
# =============================================================================

def detect_temporal_difference_shift(
    difference_summary: Optional[Any],  # SelfDifferenceSummary
) -> DetectedShift:
    """
    Detect shift from temporal self-difference.

    A shift is detected when difference is PERSISTENT, not just momentary.
    """
    if difference_summary is None:
        return DetectedShift.inactive(ShiftSource.TEMPORAL_DIFFERENCE)

    from .temporal_self_difference import DifferenceMagnitude, ChangeNature

    # Only persistent differences cause shifts, not momentary fluctuations
    if not difference_summary.has_difference:
        return DetectedShift.inactive(ShiftSource.TEMPORAL_DIFFERENCE)

    # Check magnitude - only significant+ differences
    significant_magnitudes = (
        DifferenceMagnitude.NOTICEABLE,
        DifferenceMagnitude.SIGNIFICANT,
        DifferenceMagnitude.SUBSTANTIAL,
    )

    if difference_summary.magnitude not in significant_magnitudes:
        return DetectedShift.inactive(ShiftSource.TEMPORAL_DIFFERENCE)

    # Check nature - only shifting/transformed, not stable/fluctuating
    shifting_natures = (
        ChangeNature.SHIFTING,
        ChangeNature.TRANSFORMED,
    )

    if difference_summary.nature not in shifting_natures:
        return DetectedShift.inactive(ShiftSource.TEMPORAL_DIFFERENCE)

    return DetectedShift(
        source=ShiftSource.TEMPORAL_DIFFERENCE,
        is_active=True,
        description=f"Self-state is {difference_summary.nature.value} with {difference_summary.magnitude.value} magnitude.",
    )


def detect_tendency_change_shift(
    tendency_awareness: Optional[Any],  # TendencyAwareness
) -> DetectedShift:
    """
    Detect shift from tendency changes.

    A shift is detected when tendencies are actively forming or fading.
    """
    if tendency_awareness is None:
        return DetectedShift.inactive(ShiftSource.TENDENCY_CHANGE)

    if not tendency_awareness.has_awareness:
        return DetectedShift.inactive(ShiftSource.TENDENCY_CHANGE)

    from .tendency_awareness import AwarenessType, StrengthLevel

    # Check for actively changing tendencies
    changing_types = (
        AwarenessType.HABIT_FORMING,
        AwarenessType.FADING_HABIT,
    )

    active_changes = [
        item for item in tendency_awareness.items
        if item.awareness_type in changing_types
    ]

    if not active_changes:
        return DetectedShift.inactive(ShiftSource.TENDENCY_CHANGE)

    # Only count if overall strength is at least moderate
    if tendency_awareness.overall_strength not in (StrengthLevel.MODERATE, StrengthLevel.STRONG):
        return DetectedShift.inactive(ShiftSource.TENDENCY_CHANGE)

    return DetectedShift(
        source=ShiftSource.TENDENCY_CHANGE,
        is_active=True,
        description=f"Tendencies are actively changing ({len(active_changes)} shifts detected).",
    )


def detect_continuity_strain_shift(
    strain_state: Optional[Any],  # StrainState
) -> DetectedShift:
    """
    Detect shift from continuity strain.

    A shift is detected when strain is PERSISTENT, not just momentary.
    """
    if strain_state is None:
        return DetectedShift.inactive(ShiftSource.CONTINUITY_STRAIN)

    if not strain_state.is_strained():
        return DetectedShift.inactive(ShiftSource.CONTINUITY_STRAIN)

    from .continuity_strain import StrainLevel, StrainPersistence

    # Only ongoing/chronic strain counts, not momentary
    significant_persistence = (
        StrainPersistence.ONGOING,
        StrainPersistence.CHRONIC,
    )

    if strain_state.persistence not in significant_persistence:
        return DetectedShift.inactive(ShiftSource.CONTINUITY_STRAIN)

    # Only dissonant/alienated levels count
    significant_levels = (
        StrainLevel.DISSONANT,
        StrainLevel.ALIENATED,
    )

    if strain_state.level not in significant_levels:
        return DetectedShift.inactive(ShiftSource.CONTINUITY_STRAIN)

    return DetectedShift(
        source=ShiftSource.CONTINUITY_STRAIN,
        is_active=True,
        description=f"Continuity strain is {strain_state.level.value} and {strain_state.persistence.value}.",
    )


def detect_value_instability_shift(
    value_orientation: Optional[Any],  # ValueOrientation
    config: IdentityCoherenceConfig,
) -> DetectedShift:
    """
    Detect shift from value orientation instability.

    A shift is detected when value orientation has low overall stability.
    """
    if value_orientation is None:
        return DetectedShift.inactive(ShiftSource.VALUE_INSTABILITY)

    overall_stability = value_orientation.get_overall_stability()

    if overall_stability >= config.value_stability_threshold:
        return DetectedShift.inactive(ShiftSource.VALUE_INSTABILITY)

    return DetectedShift(
        source=ShiftSource.VALUE_INSTABILITY,
        is_active=True,
        description="Value orientation is not yet stable.",
    )


def detect_self_image_flux_shift(
    self_image: Optional[Any],  # ProvisionalSelfImage
) -> DetectedShift:
    """
    Detect shift from self-image flux.

    A shift is detected when the self-image shows significant instability.
    """
    if self_image is None:
        return DetectedShift.inactive(ShiftSource.SELF_IMAGE_FLUX)

    from .self_image_integration import (
        OverallImpression,
        StabilityFeeling,
        ContinuityFeeling,
    )

    # Check overall impression
    flux_impressions = (
        OverallImpression.TRANSITIONAL,
        OverallImpression.CONFLICTED,
    )

    # Check stability feeling
    unstable_feelings = (
        StabilityFeeling.WAVERING,
        StabilityFeeling.TURBULENT,
    )

    # Check continuity feeling
    disconnected_feelings = (
        ContinuityFeeling.SOMEWHAT_DIFFERENT,
        ContinuityFeeling.DISCONNECTED,
    )

    # Count indicators of flux
    flux_indicators = 0
    descriptions = []

    if self_image.overall_impression in flux_impressions:
        flux_indicators += 1
        descriptions.append(f"overall impression is {self_image.overall_impression.value}")

    if self_image.stability_feeling in unstable_feelings:
        flux_indicators += 1
        descriptions.append(f"stability feeling is {self_image.stability_feeling.value}")

    if self_image.continuity_feeling in disconnected_feelings:
        flux_indicators += 1
        descriptions.append(f"continuity feeling is {self_image.continuity_feeling.value}")

    if self_image.has_contradictions():
        flux_indicators += 1
        descriptions.append("self-image contains contradictions")

    # Need at least 2 indicators for flux
    if flux_indicators < 2:
        return DetectedShift.inactive(ShiftSource.SELF_IMAGE_FLUX)

    return DetectedShift(
        source=ShiftSource.SELF_IMAGE_FLUX,
        is_active=True,
        description=f"Self-image is in flux: {'; '.join(descriptions)}.",
    )


def detect_emotional_turbulence_shift(
    self_image: Optional[Any],  # ProvisionalSelfImage
) -> DetectedShift:
    """
    Detect shift from emotional turbulence.

    NOTE: Temporary emotional changes do NOT cause shifts.
    Only persistent turbulence combined with instability counts.
    """
    if self_image is None:
        return DetectedShift.inactive(ShiftSource.EMOTIONAL_TURBULENCE)

    from .self_image_integration import (
        EmotionalTone,
        StabilityFeeling,
    )

    # Only turbulent states with instability count
    turbulent_tones = (
        EmotionalTone.INTENSE,
        EmotionalTone.MIXED,
    )

    # Must have BOTH turbulent emotion AND instability
    if self_image.emotional_tone not in turbulent_tones:
        return DetectedShift.inactive(ShiftSource.EMOTIONAL_TURBULENCE)

    if self_image.stability_feeling not in (StabilityFeeling.WAVERING, StabilityFeeling.TURBULENT):
        # Intense emotion alone doesn't cause shift
        return DetectedShift.inactive(ShiftSource.EMOTIONAL_TURBULENCE)

    return DetectedShift(
        source=ShiftSource.EMOTIONAL_TURBULENCE,
        is_active=True,
        description=f"Emotional state is {self_image.emotional_tone.value} with {self_image.stability_feeling.value} stability.",
    )


# =============================================================================
# Coherence Level Determination
# =============================================================================

def determine_overlap_intensity(
    active_count: int,
    config: IdentityCoherenceConfig,
) -> OverlapIntensity:
    """Determine the intensity of shift overlap."""
    if active_count == 0:
        return OverlapIntensity.NONE
    elif active_count < config.moderate_overlap_count:
        return OverlapIntensity.MINIMAL
    elif active_count < config.significant_overlap_count:
        return OverlapIntensity.MODERATE
    else:
        return OverlapIntensity.SIGNIFICANT


def determine_coherence_level(
    active_count: int,
    config: IdentityCoherenceConfig,
) -> CoherenceLevel:
    """
    Determine coherence level from active shift count.

    Key principle: Single shifts do NOT affect coherence.
    Only OVERLAP of multiple shifts affects coherence.
    """
    if active_count < config.min_shifts_for_slightly_shifting:
        return CoherenceLevel.STABLE
    elif active_count < config.min_shifts_for_unsettled:
        return CoherenceLevel.SLIGHTLY_SHIFTING
    elif active_count < config.min_shifts_for_disconnected:
        return CoherenceLevel.UNSETTLED
    else:
        return CoherenceLevel.DISCONNECTED


def generate_overlap_description(
    active_shifts: list[DetectedShift],
    intensity: OverlapIntensity,
) -> str:
    """Generate a human-readable description of the shift overlap."""
    if not active_shifts:
        return "No overlapping shifts detected."

    if len(active_shifts) == 1:
        return f"Single shift detected: {active_shifts[0].description}"

    sources = [s.source.value.replace("_", " ") for s in active_shifts]

    if intensity == OverlapIntensity.MINIMAL:
        return f"Minor overlap in: {', '.join(sources)}."
    elif intensity == OverlapIntensity.MODERATE:
        return f"Moderate overlap across: {', '.join(sources)}."
    else:
        return f"Significant overlap across multiple aspects: {', '.join(sources)}."


def generate_coherence_description(
    level: CoherenceLevel,
    overlap: ShiftOverlap,
    trend: CoherenceTrend,
) -> str:
    """
    Generate a human-readable description of the coherence state.

    This is for introspection only - NOT evaluative.
    """
    if level == CoherenceLevel.STABLE:
        return "Self feels continuous and coherent."

    # Level descriptions (non-evaluative)
    level_descriptions = {
        CoherenceLevel.SLIGHTLY_SHIFTING: "There is a slight sense of displacement from usual self",
        CoherenceLevel.UNSETTLED: "It is difficult to grasp the sense of self-identity",
        CoherenceLevel.DISCONNECTED: "There is a sense of distance from earlier self",
    }

    description = level_descriptions.get(level, "Identity coherence state unclear")

    # Add source information
    if overlap.active_count > 0:
        sources = [s.source.value.replace("_", " ") for s in overlap.get_active_shifts()]
        description += f", arising from shifts in: {', '.join(sources)}"

    # Add trend qualifier
    if trend == CoherenceTrend.CONVERGING:
        description += ". This feeling is settling down"
    elif trend == CoherenceTrend.DIVERGING:
        description += ". This feeling is intensifying"
    elif trend == CoherenceTrend.FLUCTUATING:
        description += ". This feeling fluctuates"

    return description + "."


def determine_coherence_trend(
    level_history: list[CoherenceLevel],
    config: IdentityCoherenceConfig,
) -> CoherenceTrend:
    """Determine the trend of coherence over recent observations."""
    if len(level_history) < 2:
        return CoherenceTrend.UNDEFINED

    # Get recent levels for comparison
    recent = level_history[-config.trend_lookback:]

    if len(recent) < 2:
        return CoherenceTrend.STABLE

    # Map levels to numeric values for trend detection
    level_values = {
        CoherenceLevel.STABLE: 0,
        CoherenceLevel.SLIGHTLY_SHIFTING: 1,
        CoherenceLevel.UNSETTLED: 2,
        CoherenceLevel.DISCONNECTED: 3,
        CoherenceLevel.UNDEFINED: -1,
    }

    values = [level_values.get(lvl, -1) for lvl in recent if lvl != CoherenceLevel.UNDEFINED]

    if len(values) < 2:
        return CoherenceTrend.STABLE

    # Check for consistent direction
    increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    if all(v == values[0] for v in values):
        return CoherenceTrend.STABLE
    elif decreasing and values[-1] < values[0]:
        return CoherenceTrend.CONVERGING  # Getting more stable
    elif increasing and values[-1] > values[0]:
        return CoherenceTrend.DIVERGING   # Getting less stable
    else:
        return CoherenceTrend.FLUCTUATING


# =============================================================================
# Identity Coherence System
# =============================================================================

class IdentityCoherenceSystem:
    """
    Identity Coherence Awareness System (自己同一性の揺らぎ認知)

    Observes multiple self-observation systems to detect "overlap of shifts"
    and generates awareness of identity coherence state.

    CRITICAL CONSTRAINTS:
    - For AWARENESS only, not control
    - STRICTLY NO IMPACT on decision making
    - NOT self-defense or self-repair
    - NOT defining "true self"
    - State naturally resolves when changes converge
    - NO explicit fix/recover/maintain operations
    """

    def __init__(self, config: Optional[IdentityCoherenceConfig] = None):
        """Initialize with optional configuration"""
        self._config = config or IdentityCoherenceConfig()
        self._generation_count: int = 0
        self._level_history: list[CoherenceLevel] = []
        self._last_state: Optional[IdentityCoherenceState] = None

    def generate_state(
        self,
        self_image: Optional[Any] = None,           # ProvisionalSelfImage
        difference_summary: Optional[Any] = None,   # SelfDifferenceSummary
        strain_state: Optional[Any] = None,         # StrainState
        tendency_awareness: Optional[Any] = None,   # TendencyAwareness
        value_orientation: Optional[Any] = None,    # ValueOrientation
    ) -> IdentityCoherenceState:
        """
        Generate the current identity coherence state.

        This is the main entry point. The state is generated fresh
        every time - it is NEVER cached or fixed.

        Args:
            self_image: Current ProvisionalSelfImage (from SelfImageIntegrationSystem)
            difference_summary: Current SelfDifferenceSummary (from TemporalSelfDifferenceSystem)
            strain_state: Current StrainState (from ContinuityStrainSystem)
            tendency_awareness: Current TendencyAwareness (from observe_tendencies)
            value_orientation: Current ValueOrientation

        Returns:
            IdentityCoherenceState representing the current sense of coherence
        """
        current_time = time.time()
        self._generation_count += 1

        # Check if any input is available
        if all(x is None for x in [self_image, difference_summary, strain_state,
                                    tendency_awareness, value_orientation]):
            state = IdentityCoherenceState.undefined(current_time, self._generation_count)
            self._last_state = state
            return state

        # Detect shifts from each source
        detected_shifts = [
            detect_temporal_difference_shift(difference_summary),
            detect_tendency_change_shift(tendency_awareness),
            detect_continuity_strain_shift(strain_state),
            detect_value_instability_shift(value_orientation, self._config),
            detect_self_image_flux_shift(self_image),
            detect_emotional_turbulence_shift(self_image),
        ]

        # Count active shifts
        active_shifts = [s for s in detected_shifts if s.is_active]
        active_count = len(active_shifts)

        # Determine overlap intensity
        intensity = determine_overlap_intensity(active_count, self._config)

        # Generate overlap description
        overlap_description = generate_overlap_description(active_shifts, intensity)

        # Create shift overlap structure
        shift_overlap = ShiftOverlap(
            detected_shifts=tuple(detected_shifts),
            active_count=active_count,
            intensity=intensity,
            overlap_description=overlap_description,
        )

        # Determine coherence level
        level = determine_coherence_level(active_count, self._config)

        # Update level history for trend detection
        self._level_history.append(level)
        if len(self._level_history) > self._config.trend_lookback * 2:
            self._level_history = self._level_history[-self._config.trend_lookback * 2:]

        # Determine trend
        trend = determine_coherence_trend(self._level_history, self._config)

        # Generate description
        description = generate_coherence_description(level, shift_overlap, trend)

        # Create state
        state = IdentityCoherenceState(
            level=level,
            shift_overlap=shift_overlap,
            trend=trend,
            timestamp=current_time,
            generation_count=self._generation_count,
            description=description,
        )

        self._last_state = state
        return state

    def get_last_state(self) -> Optional[IdentityCoherenceState]:
        """
        Get the last generated state (for reference only).

        NOTE: This should NOT be treated as "current" -
        always regenerate for accurate view.
        """
        return self._last_state

    def get_generation_count(self) -> int:
        """Get how many states have been generated"""
        return self._generation_count


# =============================================================================
# Integration with SelfReferenceSystem
# =============================================================================

def generate_coherence_tags(
    state: IdentityCoherenceState,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from IdentityCoherenceState for SelfReferenceSystem integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.

    Args:
        state: IdentityCoherenceState to convert
        scale: Weight scaling factor (default 1.0)

    Returns:
        List of tag dictionaries
    """
    tags = []

    if state.level == CoherenceLevel.STABLE:
        tags.append({
            "category": "IDENTITY_COHERENCE",
            "label": "coherent_self",
            "description": "Self feels continuous and coherent",
            "weight": 0.05 * scale,
        })
        return tags

    if state.level == CoherenceLevel.UNDEFINED:
        tags.append({
            "category": "IDENTITY_COHERENCE",
            "label": "coherence_undefined",
            "description": "Identity coherence state cannot be determined",
            "weight": 0.05 * scale,
        })
        return tags

    # Level tag
    tags.append({
        "category": "IDENTITY_COHERENCE_LEVEL",
        "label": f"coherence_{state.level.value}",
        "description": f"Identity coherence level: {state.level.value}",
        "weight": 0.1 * scale,
    })

    # Shift overlap tag
    if state.shift_overlap.active_count > 0:
        tags.append({
            "category": "IDENTITY_COHERENCE_OVERLAP",
            "label": f"overlap_{state.shift_overlap.intensity.value}",
            "description": state.shift_overlap.overlap_description,
            "weight": 0.08 * scale,
        })

        # Individual shift source tags
        for shift in state.shift_overlap.get_active_shifts():
            tags.append({
                "category": "IDENTITY_COHERENCE_SHIFT",
                "label": f"shift_{shift.source.value}",
                "description": shift.description,
                "weight": 0.05 * scale,
            })

    # Trend tag
    if state.trend not in (CoherenceTrend.STABLE, CoherenceTrend.UNDEFINED):
        tags.append({
            "category": "IDENTITY_COHERENCE_TREND",
            "label": f"trend_{state.trend.value}",
            "description": f"Coherence trend: {state.trend.value}",
            "weight": 0.05 * scale,
        })

    # Integrated description tag
    tags.append({
        "category": "IDENTITY_COHERENCE_INTEGRATED",
        "label": "coherence_awareness",
        "description": state.description,
        "weight": 0.1 * scale,
    })

    return tags


def get_coherence_summary(state: IdentityCoherenceState) -> str:
    """
    Get a human-readable summary of the coherence state.

    For introspection/logging only.
    """
    lines = [
        "=== Identity Coherence Awareness ===",
        f"Level: {state.level.value}",
        f"Trend: {state.trend.value}",
        "",
        "Shift Overlap:",
        f"  Active Shifts: {state.shift_overlap.active_count}",
        f"  Intensity: {state.shift_overlap.intensity.value}",
        f"  Description: {state.shift_overlap.overlap_description}",
        "",
    ]

    if state.has_active_shifts():
        lines.append("Active Shift Sources:")
        for shift in state.shift_overlap.get_active_shifts():
            lines.append(f"  - {shift.source.value}: {shift.description}")
        lines.append("")

    lines.append(f"Integrated: {state.description}")

    return "\n".join(lines)


def get_coherence_for_introspection(state: IdentityCoherenceState) -> dict[str, Any]:
    """
    Get structured coherence data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    return {
        "is_coherent": state.is_coherent(),
        "is_incoherent": state.is_incoherent(),
        "level": state.level.value,
        "trend": state.trend.value,
        "active_shift_count": state.shift_overlap.active_count,
        "overlap_intensity": state.shift_overlap.intensity.value,
        "active_shift_sources": [s.value for s in state.shift_overlap.get_active_sources()],
        "description": state.description,
        "generation_count": state.generation_count,
        "timestamp": state.timestamp,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config(
    min_shifts_for_slightly_shifting: int = 2,
    min_shifts_for_unsettled: int = 3,
    min_shifts_for_disconnected: int = 4,
    value_stability_threshold: float = 0.3,
    trend_lookback: int = 5,
) -> IdentityCoherenceConfig:
    """Create a custom configuration"""
    return IdentityCoherenceConfig(
        min_shifts_for_slightly_shifting=min_shifts_for_slightly_shifting,
        min_shifts_for_unsettled=min_shifts_for_unsettled,
        min_shifts_for_disconnected=min_shifts_for_disconnected,
        value_stability_threshold=value_stability_threshold,
        trend_lookback=trend_lookback,
    )


def create_empty_state() -> IdentityCoherenceState:
    """Create a stable state (coherent)"""
    return IdentityCoherenceState.stable()


# =============================================================================
# Impact Verification (Test Support)
# =============================================================================

def verify_no_decision_impact(state: IdentityCoherenceState) -> bool:
    """
    Verify that the coherence state has no decision-impacting values.

    This is for testing to ensure the module maintains its design constraint
    of having STRICTLY NO IMPACT on decision making.

    Returns True if the state is safe (no decision impact).
    """
    # The IdentityCoherenceState contains only:
    # - Enums (abstract categories, not numeric scores)
    # - Frozen dataclasses with enums and strings
    # - Timestamps (for tracking, not influence)
    # - Description (for introspection only)
    # - Generation count (metadata)

    public_attrs = [attr for attr in dir(state) if not attr.startswith('_')]

    for attr in public_attrs:
        if callable(getattr(state, attr)):
            continue

        value = getattr(state, attr)

        # Timestamps and counts are allowed
        if attr in ('timestamp', 'generation_count'):
            continue

        # Strings are allowed (description)
        if isinstance(value, str):
            continue

        # Enums are allowed
        if isinstance(value, Enum):
            continue

        # ShiftOverlap is allowed (contains only enums and strings)
        if isinstance(value, ShiftOverlap):
            continue

        # Anything else that looks like a decision-influencing number is not allowed
        if isinstance(value, (int, float)) and attr not in ('timestamp', 'generation_count'):
            return False

    return True


def verify_no_self_preservation(system: IdentityCoherenceSystem) -> bool:
    """
    Verify that the system has no self-preservation mechanisms.

    This checks that there are no methods that would:
    - Fix or repair coherence
    - Protect identity from change
    - Define "true self" to return to

    Returns True if the system is safe (observation only).
    """
    forbidden_patterns = [
        "fix",
        "repair",
        "restore",
        "protect",
        "defend",
        "preserve",
        "maintain",
        "correct",
        "reset_to_normal",
        "force_coherence",
        "true_self",
        "real_self",
    ]

    methods = [m for m in dir(system) if not m.startswith('_') and callable(getattr(system, m))]

    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden_patterns:
            if pattern in method_lower:
                return False

    return True


def verify_no_identity_definition(state: IdentityCoherenceState) -> bool:
    """
    Verify that the state does not define identity.

    This checks that the description does not contain
    definitive identity statements.
    """
    forbidden_phrases = [
        "true self",
        "real self",
        "correct identity",
        "proper self",
        "should be",
        "must be",
        "need to return",
        "restore to",
    ]

    description_lower = state.description.lower()

    for phrase in forbidden_phrases:
        if phrase in description_lower:
            return False

    return True
