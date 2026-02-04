"""
Provisional Self-Image Integration (自己像統合システム)

This module provides a unified, provisional "image" of the current self
by integrating multiple observation systems.

CRITICAL DESIGN PRINCIPLE:
- This is NOT a personality definition or identity
- This is NOT an evaluation or judgment
- This is just "how the current self appears to be" as a temporary image
- STRICTLY NO IMPACT on decision making
- Regenerated every cycle - never fixed or saved

Philosophy:
自己とは、定義された属性でも、固定された性格でもない。
それは感情・傾向・価値・記憶・時間差分といった
複数の観測結果が「今の私はこう見えている」という像として、
束ねられている状態である。

本機能は自己を決定しない。評価しない。正当化しない。
ただ、現在の自己の"見え方"を一時的な像として立ち上げる。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# =============================================================================
# Abstract Enums for Self-Image Description
# These describe "impressions" and "senses", NOT definitive traits
# =============================================================================

class EmotionalTone(Enum):
    """
    Abstract emotional tone of the current self-image.

    NOT a single emotion - represents the overall "color" or "atmosphere"
    of the emotional state. Uses impression language, not definitive terms.
    """
    CALM = "calm"                    # Appears calm, settled
    STIRRED = "stirred"              # Something is moving emotionally
    MIXED = "mixed"                  # Multiple feelings coexist
    INTENSE = "intense"              # Strong emotional presence
    MUTED = "muted"                  # Subdued, dampened feelings
    UNDEFINED = "undefined"          # Cannot determine


class TendencyHint(Enum):
    """
    Hints/signs of tendencies in the current self-image.

    NOT definitive habits - represents vague "signs" that something
    might be becoming habitual. Uses tentative language.
    """
    NONE_APPARENT = "none_apparent"       # No noticeable tendency signs
    SLIGHT_INCLINATION = "slight_inclination"  # Maybe a slight lean
    FORMING_PATTERN = "forming_pattern"   # Something might be forming
    ESTABLISHED_WAY = "established_way"   # Seems to have a way
    UNDEFINED = "undefined"


class StabilityFeeling(Enum):
    """
    Sense of stability/fluctuation in the current self-image.

    NOT a measurement - represents a "feeling" of how settled
    or unsettled the self appears to be.
    """
    GROUNDED = "grounded"            # Feels stable, rooted
    MOSTLY_SETTLED = "mostly_settled"  # Generally stable with minor shifts
    WAVERING = "wavering"            # Some fluctuation sensed
    TURBULENT = "turbulent"          # Significant instability felt
    UNDEFINED = "undefined"


class ChangePresence(Enum):
    """
    Whether recent change is sensed in the self-image.

    NOT a measurement of how much - just whether change
    "seems to be present" in the current view.
    """
    NO_CHANGE_SENSED = "no_change_sensed"      # Self seems consistent
    SUBTLE_SHIFT = "subtle_shift"              # Something small may have changed
    NOTICEABLE_CHANGE = "noticeable_change"    # Change is apparent
    SIGNIFICANT_SHIFT = "significant_shift"    # Major change is sensed
    UNDEFINED = "undefined"


class ContinuityFeeling(Enum):
    """
    Feeling of continuity with past self.

    NOT a score - represents the subjective "feeling" of whether
    the current self feels connected to earlier states.
    """
    CONTINUOUS = "continuous"         # Feels connected to past self
    MOSTLY_FAMILIAR = "mostly_familiar"  # Generally recognizable
    SOMEWHAT_DIFFERENT = "somewhat_different"  # Some unfamiliarity
    DISCONNECTED = "disconnected"     # Feels separate from earlier self
    UNDEFINED = "undefined"


class OverallImpression(Enum):
    """
    Overall impression of the current self-image.

    This is the most abstract level - just a general "sense"
    of how the self appears at this moment. NOT evaluative.
    """
    SETTLED = "settled"               # Self appears at ease
    ACTIVE = "active"                 # Self appears engaged/moving
    TRANSITIONAL = "transitional"     # Self appears in flux
    UNCERTAIN = "uncertain"           # Self appears unclear
    CONFLICTED = "conflicted"         # Self appears to have tensions
    UNDEFINED = "undefined"


# =============================================================================
# Provisional Self-Image Structure
# =============================================================================

@dataclass(frozen=True)
class ImageAspect:
    """
    A single aspect of the self-image with its description.

    Uses "appears to be" language, not "is" language.
    """
    aspect_name: str
    impression: str  # Abstract category value
    description: str  # "appears to be..." description

    @classmethod
    def undefined(cls, name: str) -> ImageAspect:
        """Create an undefined aspect"""
        return cls(
            aspect_name=name,
            impression="undefined",
            description=f"The {name} cannot be determined at this moment.",
        )


@dataclass(frozen=True)
class ProvisionalSelfImage:
    """
    A provisional, temporary image of the current self.

    CRITICAL CONSTRAINTS:
    - This is ALWAYS provisional (暫定的)
    - NEVER treated as permanent self-definition
    - NEVER used for judgment or decision making
    - Only for introspection and self-description
    - Regenerated every cycle
    - Contradictions are allowed to coexist

    This represents: 「今の私はこう見えている」
    """
    # Core aspects of the image
    emotional_tone: EmotionalTone
    tendency_hint: TendencyHint
    stability_feeling: StabilityFeeling
    change_presence: ChangePresence
    continuity_feeling: ContinuityFeeling
    overall_impression: OverallImpression

    # Detailed aspects (for richer description)
    emotional_aspect: ImageAspect
    tendency_aspect: ImageAspect
    stability_aspect: ImageAspect
    change_aspect: ImageAspect
    continuity_aspect: ImageAspect

    # Contradictions present (allowed to coexist)
    contradictions: tuple[str, ...]

    # Integrated description using "appears to be" language
    integrated_description: str

    # Metadata
    timestamp: float
    is_complete: bool  # True if all inputs were available

    @classmethod
    def undefined(cls, timestamp: Optional[float] = None) -> ProvisionalSelfImage:
        """Create an undefined/empty self-image"""
        ts = timestamp or time.time()
        undefined_aspect = ImageAspect.undefined("self")

        return cls(
            emotional_tone=EmotionalTone.UNDEFINED,
            tendency_hint=TendencyHint.UNDEFINED,
            stability_feeling=StabilityFeeling.UNDEFINED,
            change_presence=ChangePresence.UNDEFINED,
            continuity_feeling=ContinuityFeeling.UNDEFINED,
            overall_impression=OverallImpression.UNDEFINED,
            emotional_aspect=undefined_aspect,
            tendency_aspect=undefined_aspect,
            stability_aspect=undefined_aspect,
            change_aspect=undefined_aspect,
            continuity_aspect=undefined_aspect,
            contradictions=(),
            integrated_description="The current self-image cannot be determined.",
            timestamp=ts,
            is_complete=False,
        )

    def has_contradictions(self) -> bool:
        """Check if the image contains contradictions"""
        return len(self.contradictions) > 0

    def get_all_aspects(self) -> list[ImageAspect]:
        """Get all aspects as a list"""
        return [
            self.emotional_aspect,
            self.tendency_aspect,
            self.stability_aspect,
            self.change_aspect,
            self.continuity_aspect,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "emotional_tone": self.emotional_tone.value,
            "tendency_hint": self.tendency_hint.value,
            "stability_feeling": self.stability_feeling.value,
            "change_presence": self.change_presence.value,
            "continuity_feeling": self.continuity_feeling.value,
            "overall_impression": self.overall_impression.value,
            "emotional_aspect": {
                "aspect_name": self.emotional_aspect.aspect_name,
                "impression": self.emotional_aspect.impression,
                "description": self.emotional_aspect.description,
            },
            "tendency_aspect": {
                "aspect_name": self.tendency_aspect.aspect_name,
                "impression": self.tendency_aspect.impression,
                "description": self.tendency_aspect.description,
            },
            "stability_aspect": {
                "aspect_name": self.stability_aspect.aspect_name,
                "impression": self.stability_aspect.impression,
                "description": self.stability_aspect.description,
            },
            "change_aspect": {
                "aspect_name": self.change_aspect.aspect_name,
                "impression": self.change_aspect.impression,
                "description": self.change_aspect.description,
            },
            "continuity_aspect": {
                "aspect_name": self.continuity_aspect.aspect_name,
                "impression": self.continuity_aspect.impression,
                "description": self.continuity_aspect.description,
            },
            "contradictions": list(self.contradictions),
            "integrated_description": self.integrated_description,
            "timestamp": self.timestamp,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvisionalSelfImage:
        """Create from dictionary"""
        return cls(
            emotional_tone=EmotionalTone(data["emotional_tone"]),
            tendency_hint=TendencyHint(data["tendency_hint"]),
            stability_feeling=StabilityFeeling(data["stability_feeling"]),
            change_presence=ChangePresence(data["change_presence"]),
            continuity_feeling=ContinuityFeeling(data["continuity_feeling"]),
            overall_impression=OverallImpression(data["overall_impression"]),
            emotional_aspect=ImageAspect(
                aspect_name=data["emotional_aspect"]["aspect_name"],
                impression=data["emotional_aspect"]["impression"],
                description=data["emotional_aspect"]["description"],
            ),
            tendency_aspect=ImageAspect(
                aspect_name=data["tendency_aspect"]["aspect_name"],
                impression=data["tendency_aspect"]["impression"],
                description=data["tendency_aspect"]["description"],
            ),
            stability_aspect=ImageAspect(
                aspect_name=data["stability_aspect"]["aspect_name"],
                impression=data["stability_aspect"]["impression"],
                description=data["stability_aspect"]["description"],
            ),
            change_aspect=ImageAspect(
                aspect_name=data["change_aspect"]["aspect_name"],
                impression=data["change_aspect"]["impression"],
                description=data["change_aspect"]["description"],
            ),
            continuity_aspect=ImageAspect(
                aspect_name=data["continuity_aspect"]["aspect_name"],
                impression=data["continuity_aspect"]["impression"],
                description=data["continuity_aspect"]["description"],
            ),
            contradictions=tuple(data.get("contradictions", [])),
            integrated_description=data["integrated_description"],
            timestamp=data["timestamp"],
            is_complete=data["is_complete"],
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SelfImageConfig:
    """
    Configuration for self-image integration.

    These parameters affect how observations are translated into
    image aspects. They do NOT affect decisions.
    """
    # Whether to include detailed descriptions
    include_detailed_descriptions: bool = True

    # Whether to detect and report contradictions
    detect_contradictions: bool = True

    # Language style for descriptions
    use_tentative_language: bool = True  # "appears to be" vs "is"


# =============================================================================
# Integration Logic - Emotional Tone
# =============================================================================

def _integrate_emotional_tone(
    self_state_view: Optional[Any],  # SelfStateView
) -> tuple[EmotionalTone, ImageAspect]:
    """
    Integrate emotional observations into an emotional tone.

    Uses self_state_view.emotional to determine the overall
    emotional "color" of the self-image.
    """
    if self_state_view is None:
        return EmotionalTone.UNDEFINED, ImageAspect.undefined("emotional")

    emotional = self_state_view.emotional

    # Import enums for comparison
    from .self_model import EmotionalIntensity, EmotionalSpread, EmotionalHarmony

    if emotional.spread == EmotionalSpread.UNDEFINED:
        return EmotionalTone.UNDEFINED, ImageAspect.undefined("emotional")

    # Determine tone based on intensity and spread
    if emotional.intensity == EmotionalIntensity.CALM:
        tone = EmotionalTone.CALM
        desc = "The emotional state appears calm and settled."
    elif emotional.intensity == EmotionalIntensity.OVERWHELMING:
        tone = EmotionalTone.INTENSE
        desc = "There appears to be strong emotional presence."
    elif emotional.spread in (EmotionalSpread.MIXED, EmotionalSpread.DIFFUSE):
        tone = EmotionalTone.MIXED
        desc = "Multiple feelings appear to coexist."
    elif emotional.harmony == EmotionalHarmony.CONFLICTED:
        tone = EmotionalTone.MIXED
        desc = "There appears to be some emotional tension."
    elif emotional.intensity == EmotionalIntensity.MODERATE:
        tone = EmotionalTone.STIRRED
        desc = "Something appears to be moving emotionally."
    else:
        tone = EmotionalTone.MUTED
        desc = "The emotional state appears subdued."

    aspect = ImageAspect(
        aspect_name="emotional",
        impression=tone.value,
        description=desc,
    )

    return tone, aspect


# =============================================================================
# Integration Logic - Tendency Hint
# =============================================================================

def _integrate_tendency_hint(
    self_state_view: Optional[Any],  # SelfStateView
    tendency_awareness: Optional[Any],  # TendencyAwareness
) -> tuple[TendencyHint, ImageAspect]:
    """
    Integrate tendency observations into a tendency hint.

    Uses both self_state_view.tendency and tendency_awareness
    to determine signs of habitual patterns.
    """
    from .self_model import HabitPresence
    from .tendency_awareness import StrengthLevel

    # Check tendency awareness first (more detailed)
    if tendency_awareness is not None and tendency_awareness.has_awareness:
        overall = tendency_awareness.overall_strength

        if overall == StrengthLevel.STRONG:
            hint = TendencyHint.ESTABLISHED_WAY
            desc = "There appears to be an established way of doing things."
        elif overall == StrengthLevel.MODERATE:
            hint = TendencyHint.FORMING_PATTERN
            desc = "Something appears to be forming into a pattern."
        elif overall == StrengthLevel.SLIGHT:
            hint = TendencyHint.SLIGHT_INCLINATION
            desc = "There might be a slight inclination in certain directions."
        else:
            hint = TendencyHint.NONE_APPARENT
            desc = "No particular tendency appears at this moment."

        aspect = ImageAspect(
            aspect_name="tendency",
            impression=hint.value,
            description=desc,
        )
        return hint, aspect

    # Fall back to self_state_view.tendency
    if self_state_view is not None:
        tendency = self_state_view.tendency

        if tendency.presence == HabitPresence.UNDEFINED:
            return TendencyHint.UNDEFINED, ImageAspect.undefined("tendency")

        if tendency.presence == HabitPresence.ESTABLISHED:
            hint = TendencyHint.ESTABLISHED_WAY
            desc = "There appears to be an established way of doing things."
        elif tendency.presence == HabitPresence.FORMING:
            hint = TendencyHint.FORMING_PATTERN
            desc = "Something appears to be forming into a pattern."
        elif tendency.presence == HabitPresence.EMERGING:
            hint = TendencyHint.SLIGHT_INCLINATION
            desc = "There might be a slight inclination emerging."
        else:
            hint = TendencyHint.NONE_APPARENT
            desc = "No particular tendency appears at this moment."

        aspect = ImageAspect(
            aspect_name="tendency",
            impression=hint.value,
            description=desc,
        )
        return hint, aspect

    return TendencyHint.UNDEFINED, ImageAspect.undefined("tendency")


# =============================================================================
# Integration Logic - Stability Feeling
# =============================================================================

def _integrate_stability_feeling(
    self_state_view: Optional[Any],  # SelfStateView
    strain_state: Optional[Any],  # StrainState
) -> tuple[StabilityFeeling, ImageAspect]:
    """
    Integrate stability observations into a stability feeling.

    Uses value stability from self_state_view and strain from
    strain_state to determine how "grounded" the self appears.
    """
    from .self_model import ValueStability
    from .continuity_strain import StrainLevel

    # Check strain state first
    strain_contribution = None
    if strain_state is not None and strain_state.is_strained():
        if strain_state.level == StrainLevel.ALIENATED:
            strain_contribution = StabilityFeeling.TURBULENT
        elif strain_state.level == StrainLevel.DISSONANT:
            strain_contribution = StabilityFeeling.WAVERING
        elif strain_state.level == StrainLevel.UNSETTLED:
            strain_contribution = StabilityFeeling.MOSTLY_SETTLED

    # Check value stability
    value_contribution = None
    if self_state_view is not None:
        value = self_state_view.value

        if value.stability != ValueStability.UNDEFINED:
            if value.stability == ValueStability.ANCHORED:
                value_contribution = StabilityFeeling.GROUNDED
            elif value.stability == ValueStability.STABLE:
                value_contribution = StabilityFeeling.MOSTLY_SETTLED
            elif value.stability == ValueStability.SHIFTING:
                value_contribution = StabilityFeeling.WAVERING
            else:  # UNSTABLE
                value_contribution = StabilityFeeling.TURBULENT

    # Combine contributions (strain takes precedence if present)
    if strain_contribution is not None:
        feeling = strain_contribution
    elif value_contribution is not None:
        feeling = value_contribution
    else:
        return StabilityFeeling.UNDEFINED, ImageAspect.undefined("stability")

    # Generate description
    descriptions = {
        StabilityFeeling.GROUNDED: "The self appears stable and grounded.",
        StabilityFeeling.MOSTLY_SETTLED: "The self appears generally settled with minor fluctuations.",
        StabilityFeeling.WAVERING: "There appears to be some wavering in the sense of self.",
        StabilityFeeling.TURBULENT: "The self appears to be in a turbulent state.",
    }

    aspect = ImageAspect(
        aspect_name="stability",
        impression=feeling.value,
        description=descriptions.get(feeling, "Stability cannot be determined."),
    )

    return feeling, aspect


# =============================================================================
# Integration Logic - Change Presence
# =============================================================================

def _integrate_change_presence(
    difference_summary: Optional[Any],  # SelfDifferenceSummary
) -> tuple[ChangePresence, ImageAspect]:
    """
    Integrate temporal difference into a change presence sense.

    Uses SelfDifferenceSummary to determine whether recent
    change is sensed in the self-image.
    """
    if difference_summary is None:
        return ChangePresence.UNDEFINED, ImageAspect.undefined("change")

    from .temporal_self_difference import DifferenceMagnitude

    if not difference_summary.has_difference:
        presence = ChangePresence.NO_CHANGE_SENSED
        desc = "The self appears consistent with recent memory."
    elif difference_summary.magnitude == DifferenceMagnitude.MINIMAL:
        presence = ChangePresence.SUBTLE_SHIFT
        desc = "There might be a subtle shift from before."
    elif difference_summary.magnitude == DifferenceMagnitude.NOTICEABLE:
        presence = ChangePresence.NOTICEABLE_CHANGE
        desc = "Some change appears to have occurred."
    elif difference_summary.magnitude in (DifferenceMagnitude.SIGNIFICANT,
                                           DifferenceMagnitude.SUBSTANTIAL):
        presence = ChangePresence.SIGNIFICANT_SHIFT
        desc = "A significant shift appears to have occurred."
    else:
        presence = ChangePresence.NO_CHANGE_SENSED
        desc = "The self appears consistent with recent memory."

    aspect = ImageAspect(
        aspect_name="change",
        impression=presence.value,
        description=desc,
    )

    return presence, aspect


# =============================================================================
# Integration Logic - Continuity Feeling
# =============================================================================

def _integrate_continuity_feeling(
    strain_state: Optional[Any],  # StrainState
    difference_summary: Optional[Any],  # SelfDifferenceSummary
) -> tuple[ContinuityFeeling, ImageAspect]:
    """
    Integrate continuity observations into a continuity feeling.

    Uses strain_state and difference_summary to determine the
    feeling of connection to past self.
    """
    from .continuity_strain import StrainLevel
    from .temporal_self_difference import DifferenceMagnitude

    # Primary: strain state
    if strain_state is not None and strain_state.is_strained():
        if strain_state.level == StrainLevel.ALIENATED:
            feeling = ContinuityFeeling.DISCONNECTED
            desc = "There appears to be a sense of disconnection from earlier self."
        elif strain_state.level == StrainLevel.DISSONANT:
            feeling = ContinuityFeeling.SOMEWHAT_DIFFERENT
            desc = "The self appears somewhat different from before."
        else:  # UNSETTLED
            feeling = ContinuityFeeling.MOSTLY_FAMILIAR
            desc = "The self appears mostly familiar, with some unfamiliar notes."

        aspect = ImageAspect(
            aspect_name="continuity",
            impression=feeling.value,
            description=desc,
        )
        return feeling, aspect

    # Secondary: difference summary
    if difference_summary is not None and difference_summary.has_difference:
        if difference_summary.magnitude in (DifferenceMagnitude.SIGNIFICANT,
                                             DifferenceMagnitude.SUBSTANTIAL):
            feeling = ContinuityFeeling.SOMEWHAT_DIFFERENT
            desc = "The self appears to have changed noticeably."
        else:
            feeling = ContinuityFeeling.MOSTLY_FAMILIAR
            desc = "The self appears familiar with minor variations."

        aspect = ImageAspect(
            aspect_name="continuity",
            impression=feeling.value,
            description=desc,
        )
        return feeling, aspect

    # Default: continuous
    if strain_state is not None or difference_summary is not None:
        feeling = ContinuityFeeling.CONTINUOUS
        desc = "The self appears continuous with its earlier states."

        aspect = ImageAspect(
            aspect_name="continuity",
            impression=feeling.value,
            description=desc,
        )
        return feeling, aspect

    return ContinuityFeeling.UNDEFINED, ImageAspect.undefined("continuity")


# =============================================================================
# Integration Logic - Overall Impression
# =============================================================================

def _determine_overall_impression(
    emotional_tone: EmotionalTone,
    tendency_hint: TendencyHint,
    stability_feeling: StabilityFeeling,
    change_presence: ChangePresence,
    continuity_feeling: ContinuityFeeling,
) -> OverallImpression:
    """
    Determine the overall impression from individual aspects.

    This is the most abstract level - just a general "sense"
    of how the self appears at this moment.
    """
    # Check for undefined
    undefined_count = sum([
        emotional_tone == EmotionalTone.UNDEFINED,
        stability_feeling == StabilityFeeling.UNDEFINED,
        continuity_feeling == ContinuityFeeling.UNDEFINED,
    ])

    if undefined_count >= 2:
        return OverallImpression.UNDEFINED

    # Check for conflicted state
    if (emotional_tone == EmotionalTone.MIXED or
        continuity_feeling == ContinuityFeeling.DISCONNECTED):
        return OverallImpression.CONFLICTED

    # Check for transitional state
    if (change_presence in (ChangePresence.NOTICEABLE_CHANGE, ChangePresence.SIGNIFICANT_SHIFT) or
        stability_feeling == StabilityFeeling.TURBULENT):
        return OverallImpression.TRANSITIONAL

    # Check for uncertain state
    if (stability_feeling == StabilityFeeling.WAVERING or
        continuity_feeling == ContinuityFeeling.SOMEWHAT_DIFFERENT):
        return OverallImpression.UNCERTAIN

    # Check for active state
    if (emotional_tone in (EmotionalTone.STIRRED, EmotionalTone.INTENSE) or
        tendency_hint in (TendencyHint.FORMING_PATTERN, TendencyHint.ESTABLISHED_WAY)):
        return OverallImpression.ACTIVE

    # Default to settled
    return OverallImpression.SETTLED


# =============================================================================
# Contradiction Detection
# =============================================================================

def _detect_contradictions(
    emotional_tone: EmotionalTone,
    stability_feeling: StabilityFeeling,
    change_presence: ChangePresence,
    continuity_feeling: ContinuityFeeling,
) -> list[str]:
    """
    Detect contradictions in the self-image.

    Contradictions are NOT errors - they are allowed to coexist.
    This just notes their presence for awareness.
    """
    contradictions = []

    # Calm but turbulent
    if emotional_tone == EmotionalTone.CALM and stability_feeling == StabilityFeeling.TURBULENT:
        contradictions.append("Appears emotionally calm yet internally turbulent.")

    # No change but disconnected
    if change_presence == ChangePresence.NO_CHANGE_SENSED and continuity_feeling == ContinuityFeeling.DISCONNECTED:
        contradictions.append("No change is sensed yet there is a feeling of disconnection.")

    # Intense but grounded
    if emotional_tone == EmotionalTone.INTENSE and stability_feeling == StabilityFeeling.GROUNDED:
        contradictions.append("Intense emotions yet appears grounded.")

    # Significant change but continuous
    if change_presence == ChangePresence.SIGNIFICANT_SHIFT and continuity_feeling == ContinuityFeeling.CONTINUOUS:
        contradictions.append("Significant change occurred yet continuity is felt.")

    return contradictions


# =============================================================================
# Integrated Description Generation
# =============================================================================

def _generate_integrated_description(
    emotional_tone: EmotionalTone,
    tendency_hint: TendencyHint,
    stability_feeling: StabilityFeeling,
    change_presence: ChangePresence,
    continuity_feeling: ContinuityFeeling,
    overall_impression: OverallImpression,
    contradictions: list[str],
) -> str:
    """
    Generate an integrated description of the self-image.

    Uses "appears to be" language throughout - never definitive "is".
    """
    if overall_impression == OverallImpression.UNDEFINED:
        return "The current self-image cannot be clearly determined."

    parts = []

    # Overall impression opener
    impression_openers = {
        OverallImpression.SETTLED: "The self appears to be in a settled state",
        OverallImpression.ACTIVE: "The self appears to be active and engaged",
        OverallImpression.TRANSITIONAL: "The self appears to be in a transitional phase",
        OverallImpression.UNCERTAIN: "The self appears somewhat uncertain",
        OverallImpression.CONFLICTED: "The self appears to hold some tensions",
    }
    parts.append(impression_openers.get(overall_impression, "The current self"))

    # Emotional qualifier
    if emotional_tone not in (EmotionalTone.UNDEFINED, EmotionalTone.MUTED):
        emotional_qualifiers = {
            EmotionalTone.CALM: "with a calm emotional tone",
            EmotionalTone.STIRRED: "with something stirring emotionally",
            EmotionalTone.MIXED: "with mixed feelings present",
            EmotionalTone.INTENSE: "with strong emotional presence",
        }
        qualifier = emotional_qualifiers.get(emotional_tone)
        if qualifier:
            parts.append(qualifier)

    # Change note
    if change_presence in (ChangePresence.NOTICEABLE_CHANGE, ChangePresence.SIGNIFICANT_SHIFT):
        parts.append("and some change appears to have occurred")

    # Tendency note
    if tendency_hint in (TendencyHint.FORMING_PATTERN, TendencyHint.ESTABLISHED_WAY):
        parts.append("while certain patterns seem to be present")

    # Build sentence
    description = ", ".join(parts) + "."

    # Add contradiction note if present
    if contradictions:
        description += " However, some tensions exist: " + " ".join(contradictions)

    return description


# =============================================================================
# Self-Image Integration System
# =============================================================================

class SelfImageIntegrationSystem:
    """
    Provisional Self-Image Integration System (自己像統合システム)

    Integrates multiple observation systems into a unified, provisional
    self-image that represents "how the current self appears to be".

    CRITICAL CONSTRAINTS:
    - For INTROSPECTION only, not for control
    - STRICTLY NO IMPACT on decision making
    - The image is ALWAYS provisional - never fixed
    - Regenerated every cycle
    - Contradictions are allowed to coexist
    - Does NOT define ego, personality, or identity
    """

    def __init__(self, config: Optional[SelfImageConfig] = None):
        """Initialize with optional configuration"""
        self._config = config or SelfImageConfig()
        self._last_image: Optional[ProvisionalSelfImage] = None
        self._generation_count: int = 0

    def generate_image(
        self,
        self_state_view: Optional[Any] = None,  # SelfStateView
        tendency_awareness: Optional[Any] = None,  # TendencyAwareness
        difference_summary: Optional[Any] = None,  # SelfDifferenceSummary
        strain_state: Optional[Any] = None,  # StrainState
    ) -> ProvisionalSelfImage:
        """
        Generate a provisional self-image from current observations.

        This is the main entry point. The image is generated fresh
        every time - it is NEVER cached or fixed.

        Args:
            self_state_view: Current SelfStateView (from SelfModelSystem)
            tendency_awareness: Current TendencyAwareness (from observe_tendencies)
            difference_summary: Current SelfDifferenceSummary (from TemporalSelfDifferenceSystem)
            strain_state: Current StrainState (from ContinuityStrainSystem)

        Returns:
            ProvisionalSelfImage representing the current "appearance" of self
        """
        current_time = time.time()
        self._generation_count += 1

        # Check if any input is available
        if all(x is None for x in [self_state_view, tendency_awareness, difference_summary, strain_state]):
            image = ProvisionalSelfImage.undefined(current_time)
            self._last_image = image
            return image

        # Integrate each aspect
        emotional_tone, emotional_aspect = _integrate_emotional_tone(self_state_view)
        tendency_hint, tendency_aspect = _integrate_tendency_hint(self_state_view, tendency_awareness)
        stability_feeling, stability_aspect = _integrate_stability_feeling(self_state_view, strain_state)
        change_presence, change_aspect = _integrate_change_presence(difference_summary)
        continuity_feeling, continuity_aspect = _integrate_continuity_feeling(strain_state, difference_summary)

        # Determine overall impression
        overall_impression = _determine_overall_impression(
            emotional_tone,
            tendency_hint,
            stability_feeling,
            change_presence,
            continuity_feeling,
        )

        # Detect contradictions (allowed to coexist)
        contradictions = []
        if self._config.detect_contradictions:
            contradictions = _detect_contradictions(
                emotional_tone,
                stability_feeling,
                change_presence,
                continuity_feeling,
            )

        # Generate integrated description
        integrated_description = _generate_integrated_description(
            emotional_tone,
            tendency_hint,
            stability_feeling,
            change_presence,
            continuity_feeling,
            overall_impression,
            contradictions,
        )

        # Determine completeness
        is_complete = (
            emotional_tone != EmotionalTone.UNDEFINED and
            stability_feeling != StabilityFeeling.UNDEFINED and
            continuity_feeling != ContinuityFeeling.UNDEFINED
        )

        # Create the image
        image = ProvisionalSelfImage(
            emotional_tone=emotional_tone,
            tendency_hint=tendency_hint,
            stability_feeling=stability_feeling,
            change_presence=change_presence,
            continuity_feeling=continuity_feeling,
            overall_impression=overall_impression,
            emotional_aspect=emotional_aspect,
            tendency_aspect=tendency_aspect,
            stability_aspect=stability_aspect,
            change_aspect=change_aspect,
            continuity_aspect=continuity_aspect,
            contradictions=tuple(contradictions),
            integrated_description=integrated_description,
            timestamp=current_time,
            is_complete=is_complete,
        )

        self._last_image = image
        return image

    def get_last_image(self) -> Optional[ProvisionalSelfImage]:
        """
        Get the last generated image (for reference only).

        NOTE: This should NOT be treated as "current" -
        always regenerate for accurate view.
        """
        return self._last_image

    def get_generation_count(self) -> int:
        """Get how many images have been generated"""
        return self._generation_count


# =============================================================================
# Integration with SelfReferenceSystem
# =============================================================================

def generate_self_image_tags(
    image: ProvisionalSelfImage,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from ProvisionalSelfImage for SelfReferenceSystem integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.

    Args:
        image: ProvisionalSelfImage to convert
        scale: Weight scaling factor (default 1.0)

    Returns:
        List of tag dictionaries
    """
    tags = []

    if not image.is_complete:
        tags.append({
            "category": "SELF_IMAGE",
            "label": "image_incomplete",
            "description": "Current self-image is incomplete",
            "weight": 0.05 * scale,
        })
        return tags

    # Overall impression tag
    tags.append({
        "category": "SELF_IMAGE_OVERALL",
        "label": f"impression_{image.overall_impression.value}",
        "description": f"Overall impression: {image.overall_impression.value}",
        "weight": 0.1 * scale,
    })

    # Emotional tone tag
    if image.emotional_tone != EmotionalTone.UNDEFINED:
        tags.append({
            "category": "SELF_IMAGE_EMOTIONAL",
            "label": f"tone_{image.emotional_tone.value}",
            "description": image.emotional_aspect.description,
            "weight": 0.08 * scale,
        })

    # Stability feeling tag
    if image.stability_feeling != StabilityFeeling.UNDEFINED:
        tags.append({
            "category": "SELF_IMAGE_STABILITY",
            "label": f"feeling_{image.stability_feeling.value}",
            "description": image.stability_aspect.description,
            "weight": 0.08 * scale,
        })

    # Change presence tag
    if image.change_presence != ChangePresence.UNDEFINED:
        tags.append({
            "category": "SELF_IMAGE_CHANGE",
            "label": f"change_{image.change_presence.value}",
            "description": image.change_aspect.description,
            "weight": 0.05 * scale,
        })

    # Continuity feeling tag
    if image.continuity_feeling != ContinuityFeeling.UNDEFINED:
        tags.append({
            "category": "SELF_IMAGE_CONTINUITY",
            "label": f"continuity_{image.continuity_feeling.value}",
            "description": image.continuity_aspect.description,
            "weight": 0.08 * scale,
        })

    # Contradiction tag (if present)
    if image.has_contradictions():
        tags.append({
            "category": "SELF_IMAGE_TENSION",
            "label": "has_contradictions",
            "description": f"Self-image contains tensions: {'; '.join(image.contradictions)}",
            "weight": 0.05 * scale,
        })

    # Integrated description tag
    tags.append({
        "category": "SELF_IMAGE_INTEGRATED",
        "label": "self_appearance",
        "description": image.integrated_description,
        "weight": 0.1 * scale,
    })

    return tags


def get_self_image_summary(image: ProvisionalSelfImage) -> str:
    """
    Get a human-readable summary of the self-image.

    For introspection/logging only.
    """
    lines = [
        "=== Provisional Self-Image ===",
        f"Overall Impression: {image.overall_impression.value}",
        f"Emotional Tone: {image.emotional_tone.value}",
        f"Tendency Hint: {image.tendency_hint.value}",
        f"Stability Feeling: {image.stability_feeling.value}",
        f"Change Presence: {image.change_presence.value}",
        f"Continuity Feeling: {image.continuity_feeling.value}",
        "",
        "Aspects:",
        f"  Emotional: {image.emotional_aspect.description}",
        f"  Tendency: {image.tendency_aspect.description}",
        f"  Stability: {image.stability_aspect.description}",
        f"  Change: {image.change_aspect.description}",
        f"  Continuity: {image.continuity_aspect.description}",
        "",
        f"Complete: {image.is_complete}",
    ]

    if image.has_contradictions():
        lines.append("")
        lines.append("Contradictions (allowed to coexist):")
        for c in image.contradictions:
            lines.append(f"  - {c}")

    lines.append("")
    lines.append(f"Integrated: {image.integrated_description}")

    return "\n".join(lines)


def get_self_image_for_introspection(image: ProvisionalSelfImage) -> dict[str, Any]:
    """
    Get structured self-image data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    return {
        "overall_impression": image.overall_impression.value,
        "emotional_tone": image.emotional_tone.value,
        "tendency_hint": image.tendency_hint.value,
        "stability_feeling": image.stability_feeling.value,
        "change_presence": image.change_presence.value,
        "continuity_feeling": image.continuity_feeling.value,
        "has_contradictions": image.has_contradictions(),
        "contradictions": list(image.contradictions),
        "integrated_description": image.integrated_description,
        "is_complete": image.is_complete,
        "timestamp": image.timestamp,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config(
    include_detailed_descriptions: bool = True,
    detect_contradictions: bool = True,
    use_tentative_language: bool = True,
) -> SelfImageConfig:
    """Create a custom configuration"""
    return SelfImageConfig(
        include_detailed_descriptions=include_detailed_descriptions,
        detect_contradictions=detect_contradictions,
        use_tentative_language=use_tentative_language,
    )


def create_empty_image() -> ProvisionalSelfImage:
    """Create an undefined/empty self-image"""
    return ProvisionalSelfImage.undefined()


# =============================================================================
# Impact Verification (Test Support)
# =============================================================================

def verify_no_decision_impact(image: ProvisionalSelfImage) -> bool:
    """
    Verify that the self-image has no decision-impacting values.

    This is for testing to ensure the module maintains its design constraint
    of having STRICTLY NO IMPACT on decision making.

    Returns True if the image is safe (no decision impact).
    """
    # The ProvisionalSelfImage contains only:
    # - Enums (abstract categories, not numeric scores)
    # - Strings (descriptions)
    # - Timestamps (for tracking, not influence)
    # - Boolean flags (for completeness checking)

    public_attrs = [attr for attr in dir(image) if not attr.startswith('_')]

    for attr in public_attrs:
        if callable(getattr(image, attr)):
            continue

        value = getattr(image, attr)

        # Timestamps are allowed
        if attr == 'timestamp':
            continue

        # Booleans are allowed
        if isinstance(value, bool):
            continue

        # Strings are allowed
        if isinstance(value, str):
            continue

        # Tuples of strings are allowed (contradictions)
        if isinstance(value, tuple) and all(isinstance(x, str) for x in value):
            continue

        # Enums are allowed
        if isinstance(value, Enum):
            continue

        # ImageAspect is allowed
        if isinstance(value, ImageAspect):
            continue

        # Anything else that looks like a decision-influencing number is not allowed
        if isinstance(value, (int, float)) and attr != 'timestamp':
            return False

    return True


def verify_provisional_nature(system: SelfImageIntegrationSystem) -> bool:
    """
    Verify that the system maintains provisional nature.

    This checks that there are no methods that would:
    - Fix or save the self-image permanently
    - Use the image to make decisions
    - Modify other systems based on the image

    Returns True if the system is safe (observation only).
    """
    forbidden_patterns = [
        "fix",
        "save_permanent",
        "decide",
        "choose",
        "select",
        "apply_to_decision",
        "modify_",
        "influence_",
    ]

    methods = [m for m in dir(system) if not m.startswith('_') and callable(getattr(system, m))]

    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden_patterns:
            if pattern in method_lower:
                return False

    return True
