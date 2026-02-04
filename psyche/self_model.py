"""
Self-Model System (自己状態統合モデル)

This module provides a unified view of internal psychological state.
It observes existing systems READ-ONLY and generates abstract descriptions
that can be used for introspection and self-description.

CRITICAL DESIGN PRINCIPLE:
- This is a MIRROR, not a controller
- READ-ONLY observation of other systems
- STRICTLY NO IMPACT on decision making
- Numbers are hidden; only abstract descriptions are exposed
- Self-model is always a provisional snapshot, never definitive

Philosophy:
自我とは、判断や意思ではない。
それは「今の自分がどういう状態にあるかを、
ひとつの像として把握できている状態」である。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# =============================================================================
# Abstract Enums for State Description
# These replace raw numbers with categorical descriptions
# =============================================================================

class EmotionalSpread(Enum):
    """How many emotions are simultaneously active"""
    FOCUSED = "focused"        # 1-2 active emotions
    MIXED = "mixed"            # 3-4 active emotions
    DIFFUSE = "diffuse"        # 5+ active emotions
    UNDEFINED = "undefined"    # No emotion data available


class EmotionalIntensity(Enum):
    """Overall intensity of emotional state"""
    CALM = "calm"              # Total intensity < 0.3
    MODERATE = "moderate"      # Total intensity 0.3-0.7
    INTENSE = "intense"        # Total intensity 0.7-1.5
    OVERWHELMING = "overwhelming"  # Total intensity > 1.5
    UNDEFINED = "undefined"


class EmotionalHarmony(Enum):
    """Whether emotions are in conflict or harmony"""
    HARMONIOUS = "harmonious"      # No conflicting emotions
    SLIGHT_TENSION = "slight_tension"  # Minor conflicts present
    CONFLICTED = "conflicted"      # Significant opposing emotions
    UNDEFINED = "undefined"


class BurdenLevel(Enum):
    """Level of responsibility burden"""
    UNBURDENED = "unburdened"  # weight < 0.1
    LIGHT = "light"            # weight 0.1-0.3
    MODERATE = "moderate"      # weight 0.3-0.5
    BURDENED = "burdened"      # weight 0.5-0.7
    HEAVY = "heavy"            # weight > 0.7
    UNDEFINED = "undefined"


class BurdenDistribution(Enum):
    """How responsibility is distributed across units"""
    CONCENTRATED = "concentrated"  # Few units, high weight each
    DISTRIBUTED = "distributed"    # Many units, balanced weight
    SCATTERED = "scattered"        # Many units, uneven weight
    UNDEFINED = "undefined"


class BurdenTrend(Enum):
    """Direction of responsibility change"""
    ACCUMULATING = "accumulating"  # Increasing harm, pending decisions
    STABLE = "stable"              # Little change
    RELEASING = "releasing"        # Decreasing, resolved decisions
    UNDEFINED = "undefined"


class HabitPresence(Enum):
    """Presence and stage of habitual tendencies"""
    NONE = "none"              # No tendencies detected
    EMERGING = "emerging"      # Tendencies starting to form
    FORMING = "forming"        # Tendencies becoming established
    ESTABLISHED = "established"  # Strong habitual patterns
    UNDEFINED = "undefined"


class HabitCharacter(Enum):
    """Character of habitual tendencies"""
    EXPLORATORY = "exploratory"    # Varied, non-repetitive
    SLIGHT_PATTERN = "slight_pattern"  # Mild repetition
    HABITUAL = "habitual"          # Clear habitual behavior
    RIGID = "rigid"                # Strongly fixed patterns
    UNDEFINED = "undefined"


class DirectionClarity(Enum):
    """How clear the proto-goal direction is"""
    CLEAR = "clear"            # Strong, consistent direction
    UNCERTAIN = "uncertain"    # Direction exists but weak
    UNCLEAR = "unclear"        # Multiple conflicting directions
    UNDEFINED = "undefined"    # No direction data


class DirectionConvergence(Enum):
    """Whether multiple directions are converging or diverging"""
    CONVERGENT = "convergent"  # Vectors point similarly
    PARALLEL = "parallel"      # Vectors coexist independently
    DIVERGENT = "divergent"    # Vectors point differently
    SCATTERED = "scattered"    # No coherent pattern
    UNDEFINED = "undefined"


class ValueStability(Enum):
    """Stability of value orientations"""
    UNSTABLE = "unstable"      # Low confidence, frequent changes
    SHIFTING = "shifting"      # Moderate confidence, some changes
    STABLE = "stable"          # High confidence, rare changes
    ANCHORED = "anchored"      # Very high confidence, settled
    UNDEFINED = "undefined"


class ValueClarity(Enum):
    """How clearly defined value orientations are"""
    UNDEFINED_VALUES = "undefined"  # No meaningful orientation
    EMERGING = "emerging"      # Beginning to form
    FORMING = "forming"        # Becoming clearer
    DEFINED = "defined"        # Clear value orientations
    # Note: Using UNDEFINED_VALUES to avoid conflict with Python's None


# =============================================================================
# Component State Views
# Each captures one aspect of internal state in abstract terms
# =============================================================================

@dataclass(frozen=True)
class EmotionalStateView:
    """
    Abstract view of emotional state.
    Does NOT singularize emotions - captures the overall pattern.
    """
    spread: EmotionalSpread
    intensity: EmotionalIntensity
    harmony: EmotionalHarmony
    active_emotion_count: int  # Count only, not which ones
    has_coexisting_pairs: bool
    description: str  # Human-readable description

    @classmethod
    def undefined(cls) -> EmotionalStateView:
        """Create an undefined emotional state view"""
        return cls(
            spread=EmotionalSpread.UNDEFINED,
            intensity=EmotionalIntensity.UNDEFINED,
            harmony=EmotionalHarmony.UNDEFINED,
            active_emotion_count=0,
            has_coexisting_pairs=False,
            description="Emotional state is undefined"
        )


@dataclass(frozen=True)
class ResponsibilityStateView:
    """
    Abstract view of responsibility burden.
    Captures weight presence and distribution pattern.
    """
    burden_level: BurdenLevel
    distribution: BurdenDistribution
    trend: BurdenTrend
    has_pending_decisions: bool
    description: str

    @classmethod
    def undefined(cls) -> ResponsibilityStateView:
        """Create an undefined responsibility state view"""
        return cls(
            burden_level=BurdenLevel.UNDEFINED,
            distribution=BurdenDistribution.UNDEFINED,
            trend=BurdenTrend.UNDEFINED,
            has_pending_decisions=False,
            description="Responsibility state is undefined"
        )


@dataclass(frozen=True)
class TendencyStateView:
    """
    Abstract view of habitual tendencies.
    Captures existence and stage of habit formation.
    """
    presence: HabitPresence
    character: HabitCharacter
    tendency_count: int  # Count only, not details
    has_strong_habits: bool
    has_fading_habits: bool
    description: str

    @classmethod
    def undefined(cls) -> TendencyStateView:
        """Create an undefined tendency state view"""
        return cls(
            presence=HabitPresence.UNDEFINED,
            character=HabitCharacter.UNDEFINED,
            tendency_count=0,
            has_strong_habits=False,
            has_fading_habits=False,
            description="Tendency state is undefined"
        )


@dataclass(frozen=True)
class DirectionStateView:
    """
    Abstract view of proto-goal direction.
    Captures convergence vs divergence of internal vectors.
    """
    clarity: DirectionClarity
    convergence: DirectionConvergence
    vector_count: int  # Count only
    has_dominant_direction: bool
    description: str

    @classmethod
    def undefined(cls) -> DirectionStateView:
        """Create an undefined direction state view"""
        return cls(
            clarity=DirectionClarity.UNDEFINED,
            convergence=DirectionConvergence.UNDEFINED,
            vector_count=0,
            has_dominant_direction=False,
            description="Direction state is undefined"
        )


@dataclass(frozen=True)
class ValueStateView:
    """
    Abstract view of value orientation.
    Captures stability and clarity without exposing actual values.
    """
    stability: ValueStability
    clarity: ValueClarity
    has_strong_orientations: bool
    is_recently_changed: bool
    description: str

    @classmethod
    def undefined(cls) -> ValueStateView:
        """Create an undefined value state view"""
        return cls(
            stability=ValueStability.UNDEFINED,
            clarity=ValueClarity.UNDEFINED_VALUES,
            has_strong_orientations=False,
            is_recently_changed=False,
            description="Value orientation is undefined"
        )


# =============================================================================
# Unified Self-State View
# =============================================================================

@dataclass(frozen=True)
class SelfStateView:
    """
    Unified view of self-state.

    This is NOT a personality, belief, or identity.
    It is simply "how the current self appears" as a snapshot.

    This view:
    - Is always provisional (暫定的)
    - Does not determine self-image from single elements
    - Cannot be used for judgment or decision making
    - Is only for introspection, logging, and self-description
    """
    # Component views
    emotional: EmotionalStateView
    responsibility: ResponsibilityStateView
    tendency: TendencyStateView
    direction: DirectionStateView
    value: ValueStateView

    # Metadata
    timestamp: float
    snapshot_id: str
    is_complete: bool  # True if all components have data

    # Integrated description
    integrated_description: str

    def get_component_summaries(self) -> dict[str, str]:
        """Get all component descriptions as a dictionary"""
        return {
            "emotional": self.emotional.description,
            "responsibility": self.responsibility.description,
            "tendency": self.tendency.description,
            "direction": self.direction.description,
            "value": self.value.description,
        }

    def get_undefined_components(self) -> list[str]:
        """List components that are undefined"""
        undefined = []
        if self.emotional.spread == EmotionalSpread.UNDEFINED:
            undefined.append("emotional")
        if self.responsibility.burden_level == BurdenLevel.UNDEFINED:
            undefined.append("responsibility")
        if self.tendency.presence == HabitPresence.UNDEFINED:
            undefined.append("tendency")
        if self.direction.clarity == DirectionClarity.UNDEFINED:
            undefined.append("direction")
        if self.value.stability == ValueStability.UNDEFINED:
            undefined.append("value")
        return undefined

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "emotional": {
                "spread": self.emotional.spread.value,
                "intensity": self.emotional.intensity.value,
                "harmony": self.emotional.harmony.value,
                "active_emotion_count": self.emotional.active_emotion_count,
                "has_coexisting_pairs": self.emotional.has_coexisting_pairs,
                "description": self.emotional.description,
            },
            "responsibility": {
                "burden_level": self.responsibility.burden_level.value,
                "distribution": self.responsibility.distribution.value,
                "trend": self.responsibility.trend.value,
                "has_pending_decisions": self.responsibility.has_pending_decisions,
                "description": self.responsibility.description,
            },
            "tendency": {
                "presence": self.tendency.presence.value,
                "character": self.tendency.character.value,
                "tendency_count": self.tendency.tendency_count,
                "has_strong_habits": self.tendency.has_strong_habits,
                "has_fading_habits": self.tendency.has_fading_habits,
                "description": self.tendency.description,
            },
            "direction": {
                "clarity": self.direction.clarity.value,
                "convergence": self.direction.convergence.value,
                "vector_count": self.direction.vector_count,
                "has_dominant_direction": self.direction.has_dominant_direction,
                "description": self.direction.description,
            },
            "value": {
                "stability": self.value.stability.value,
                "clarity": self.value.clarity.value,
                "has_strong_orientations": self.value.has_strong_orientations,
                "is_recently_changed": self.value.is_recently_changed,
                "description": self.value.description,
            },
            "timestamp": self.timestamp,
            "snapshot_id": self.snapshot_id,
            "is_complete": self.is_complete,
            "integrated_description": self.integrated_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelfStateView:
        """Create from dictionary"""
        return cls(
            emotional=EmotionalStateView(
                spread=EmotionalSpread(data["emotional"]["spread"]),
                intensity=EmotionalIntensity(data["emotional"]["intensity"]),
                harmony=EmotionalHarmony(data["emotional"]["harmony"]),
                active_emotion_count=data["emotional"]["active_emotion_count"],
                has_coexisting_pairs=data["emotional"]["has_coexisting_pairs"],
                description=data["emotional"]["description"],
            ),
            responsibility=ResponsibilityStateView(
                burden_level=BurdenLevel(data["responsibility"]["burden_level"]),
                distribution=BurdenDistribution(data["responsibility"]["distribution"]),
                trend=BurdenTrend(data["responsibility"]["trend"]),
                has_pending_decisions=data["responsibility"]["has_pending_decisions"],
                description=data["responsibility"]["description"],
            ),
            tendency=TendencyStateView(
                presence=HabitPresence(data["tendency"]["presence"]),
                character=HabitCharacter(data["tendency"]["character"]),
                tendency_count=data["tendency"]["tendency_count"],
                has_strong_habits=data["tendency"]["has_strong_habits"],
                has_fading_habits=data["tendency"]["has_fading_habits"],
                description=data["tendency"]["description"],
            ),
            direction=DirectionStateView(
                clarity=DirectionClarity(data["direction"]["clarity"]),
                convergence=DirectionConvergence(data["direction"]["convergence"]),
                vector_count=data["direction"]["vector_count"],
                has_dominant_direction=data["direction"]["has_dominant_direction"],
                description=data["direction"]["description"],
            ),
            value=ValueStateView(
                stability=ValueStability(data["value"]["stability"]),
                clarity=ValueClarity(data["value"]["clarity"]),
                has_strong_orientations=data["value"]["has_strong_orientations"],
                is_recently_changed=data["value"]["is_recently_changed"],
                description=data["value"]["description"],
            ),
            timestamp=data["timestamp"],
            snapshot_id=data["snapshot_id"],
            is_complete=data["is_complete"],
            integrated_description=data["integrated_description"],
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SelfModelConfig:
    """
    Configuration for self-model observation.

    These thresholds determine how raw values map to abstract categories.
    They do NOT affect decision making in any way.
    """
    # Emotional thresholds
    emotion_active_threshold: float = 0.1
    emotion_intensity_calm_max: float = 0.3
    emotion_intensity_moderate_max: float = 0.7
    emotion_intensity_intense_max: float = 1.5

    # Responsibility thresholds
    burden_light_max: float = 0.1
    burden_moderate_max: float = 0.3
    burden_burdened_max: float = 0.5
    burden_heavy_max: float = 0.7

    # Tendency thresholds
    tendency_emerging_min_count: int = 1
    tendency_forming_min_count: int = 2
    tendency_established_min_count: int = 3
    tendency_strong_threshold: float = 0.1  # Strength threshold

    # Direction thresholds
    direction_clear_min_magnitude: float = 0.3
    direction_convergence_similarity: float = 0.7

    # Value thresholds
    value_confidence_unstable_max: float = 0.2
    value_confidence_shifting_max: float = 0.4
    value_confidence_stable_max: float = 0.7
    value_strong_orientation_threshold: float = 0.3
    value_recent_change_turns: int = 10


# =============================================================================
# Observation Functions
# These READ-ONLY observe existing systems and generate abstract views
# =============================================================================

def observe_emotional_state(
    emotion_vector: Any,
    config: SelfModelConfig = SelfModelConfig()
) -> EmotionalStateView:
    """
    Observe emotional state and generate abstract view.

    Args:
        emotion_vector: EmotionVector instance (READ-ONLY access)
        config: Configuration for thresholds

    Returns:
        Abstract EmotionalStateView
    """
    if emotion_vector is None:
        return EmotionalStateView.undefined()

    try:
        # Import multi_emotion functions for READ-ONLY observation
        from .multi_emotion import (
            get_active_emotions,
            get_emotion_intensity,
            has_conflicting_emotions,
            get_coexisting_pairs,
            MultiEmotionConfig,
        )

        # Create config for observation
        me_config = MultiEmotionConfig(active_threshold=config.emotion_active_threshold)

        # Get active emotions (READ-ONLY)
        active_emotions = get_active_emotions(emotion_vector, me_config)
        active_count = len(active_emotions)

        # Determine spread
        if active_count == 0:
            spread = EmotionalSpread.FOCUSED
        elif active_count <= 2:
            spread = EmotionalSpread.FOCUSED
        elif active_count <= 4:
            spread = EmotionalSpread.MIXED
        else:
            spread = EmotionalSpread.DIFFUSE

        # Get intensity (READ-ONLY)
        total_intensity = get_emotion_intensity(emotion_vector)
        if total_intensity < config.emotion_intensity_calm_max:
            intensity = EmotionalIntensity.CALM
        elif total_intensity < config.emotion_intensity_moderate_max:
            intensity = EmotionalIntensity.MODERATE
        elif total_intensity < config.emotion_intensity_intense_max:
            intensity = EmotionalIntensity.INTENSE
        else:
            intensity = EmotionalIntensity.OVERWHELMING

        # Check harmony (READ-ONLY)
        has_conflicts = has_conflicting_emotions(emotion_vector, me_config)
        coexisting_pairs = get_coexisting_pairs(emotion_vector, me_config)
        has_coexisting = len(coexisting_pairs) > 0

        if not has_conflicts:
            harmony = EmotionalHarmony.HARMONIOUS
        elif len(coexisting_pairs) <= 1:
            harmony = EmotionalHarmony.SLIGHT_TENSION
        else:
            harmony = EmotionalHarmony.CONFLICTED

        # Generate description
        description = _generate_emotional_description(spread, intensity, harmony)

        return EmotionalStateView(
            spread=spread,
            intensity=intensity,
            harmony=harmony,
            active_emotion_count=active_count,
            has_coexisting_pairs=has_coexisting,
            description=description,
        )

    except Exception:
        return EmotionalStateView.undefined()


def _generate_emotional_description(
    spread: EmotionalSpread,
    intensity: EmotionalIntensity,
    harmony: EmotionalHarmony
) -> str:
    """Generate human-readable emotional description"""
    parts = []

    if intensity == EmotionalIntensity.CALM:
        parts.append("Emotionally calm")
    elif intensity == EmotionalIntensity.MODERATE:
        parts.append("Emotionally active")
    elif intensity == EmotionalIntensity.INTENSE:
        parts.append("Emotionally intense")
    elif intensity == EmotionalIntensity.OVERWHELMING:
        parts.append("Emotionally overwhelmed")

    if spread == EmotionalSpread.FOCUSED:
        parts.append("with focused feeling")
    elif spread == EmotionalSpread.MIXED:
        parts.append("with mixed feelings")
    elif spread == EmotionalSpread.DIFFUSE:
        parts.append("with diffuse feelings")

    if harmony == EmotionalHarmony.CONFLICTED:
        parts.append("and inner conflict")
    elif harmony == EmotionalHarmony.SLIGHT_TENSION:
        parts.append("and slight tension")

    return ", ".join(parts) if parts else "Emotional state observed"


def observe_responsibility_state(
    responsibility_state: Any,
    previous_weight: Optional[float] = None,
    config: SelfModelConfig = SelfModelConfig()
) -> ResponsibilityStateView:
    """
    Observe responsibility state and generate abstract view.

    Args:
        responsibility_state: ResponsibilityState instance (READ-ONLY)
        previous_weight: Previous total weight for trend detection
        config: Configuration for thresholds

    Returns:
        Abstract ResponsibilityStateView
    """
    if responsibility_state is None:
        return ResponsibilityStateView.undefined()

    try:
        # Get burden level (READ-ONLY)
        weight = responsibility_state.total_weight
        if weight < config.burden_light_max:
            burden_level = BurdenLevel.UNBURDENED
        elif weight < config.burden_moderate_max:
            burden_level = BurdenLevel.LIGHT
        elif weight < config.burden_burdened_max:
            burden_level = BurdenLevel.MODERATE
        elif weight < config.burden_heavy_max:
            burden_level = BurdenLevel.BURDENED
        else:
            burden_level = BurdenLevel.HEAVY

        # Determine distribution based on decision count and weight
        pending = responsibility_state.pending_decisions
        recent_count = len(responsibility_state.recent_decisions)

        if recent_count == 0:
            distribution = BurdenDistribution.UNDEFINED
        elif recent_count <= 3 and weight > 0.3:
            distribution = BurdenDistribution.CONCENTRATED
        elif recent_count > 10:
            distribution = BurdenDistribution.SCATTERED
        else:
            distribution = BurdenDistribution.DISTRIBUTED

        # Determine trend
        if previous_weight is not None:
            weight_delta = weight - previous_weight
            if weight_delta > 0.05:
                trend = BurdenTrend.ACCUMULATING
            elif weight_delta < -0.05:
                trend = BurdenTrend.RELEASING
            else:
                trend = BurdenTrend.STABLE
        else:
            # Infer from harm vs confidence
            if responsibility_state.accumulated_harm > responsibility_state.accumulated_confidence:
                trend = BurdenTrend.ACCUMULATING
            elif responsibility_state.accumulated_confidence > responsibility_state.accumulated_harm:
                trend = BurdenTrend.RELEASING
            else:
                trend = BurdenTrend.STABLE

        # Generate description
        description = _generate_responsibility_description(
            burden_level, distribution, trend, pending > 0
        )

        return ResponsibilityStateView(
            burden_level=burden_level,
            distribution=distribution,
            trend=trend,
            has_pending_decisions=pending > 0,
            description=description,
        )

    except Exception:
        return ResponsibilityStateView.undefined()


def _generate_responsibility_description(
    burden_level: BurdenLevel,
    distribution: BurdenDistribution,
    trend: BurdenTrend,
    has_pending: bool
) -> str:
    """Generate human-readable responsibility description"""
    parts = []

    if burden_level == BurdenLevel.UNBURDENED:
        parts.append("Unburdened")
    elif burden_level == BurdenLevel.LIGHT:
        parts.append("Lightly burdened")
    elif burden_level == BurdenLevel.MODERATE:
        parts.append("Moderately burdened")
    elif burden_level == BurdenLevel.BURDENED:
        parts.append("Burdened")
    elif burden_level == BurdenLevel.HEAVY:
        parts.append("Heavily burdened")

    if distribution == BurdenDistribution.CONCENTRATED:
        parts.append("with concentrated weight")
    elif distribution == BurdenDistribution.SCATTERED:
        parts.append("with scattered concerns")

    if trend == BurdenTrend.ACCUMULATING:
        parts.append("and accumulating")
    elif trend == BurdenTrend.RELEASING:
        parts.append("and releasing")

    if has_pending:
        parts.append("with pending matters")

    return ", ".join(parts) if parts else "Responsibility state observed"


def observe_tendency_state(
    tendency_system: Any,
    tendency_awareness: Any = None,
    config: SelfModelConfig = SelfModelConfig()
) -> TendencyStateView:
    """
    Observe tendency state and generate abstract view.

    Args:
        tendency_system: RepeatedTendencySystem instance (READ-ONLY)
        tendency_awareness: TendencyAwareness instance if available (READ-ONLY)
        config: Configuration for thresholds

    Returns:
        Abstract TendencyStateView
    """
    if tendency_system is None:
        return TendencyStateView.undefined()

    try:
        # Get tendencies (READ-ONLY)
        tendencies = tendency_system.get_tendencies()
        tendency_count = len(tendencies)

        # Determine presence
        if tendency_count == 0:
            presence = HabitPresence.NONE
        elif tendency_count < config.tendency_forming_min_count:
            presence = HabitPresence.EMERGING
        elif tendency_count < config.tendency_established_min_count:
            presence = HabitPresence.FORMING
        else:
            presence = HabitPresence.ESTABLISHED

        # Check for strong and fading habits
        has_strong = False
        has_fading = False
        strong_count = 0

        for t in tendencies:
            if t.strength >= config.tendency_strong_threshold:
                has_strong = True
                strong_count += 1
            if t.consecutive_misses > 3:
                has_fading = True

        # Determine character
        if tendency_count == 0:
            character = HabitCharacter.EXPLORATORY
        elif strong_count == 0:
            character = HabitCharacter.SLIGHT_PATTERN
        elif strong_count <= tendency_count // 2:
            character = HabitCharacter.HABITUAL
        else:
            character = HabitCharacter.RIGID

        # Use awareness if available for richer description
        if tendency_awareness is not None and tendency_awareness.has_awareness:
            # Incorporate awareness insights
            if tendency_awareness.overall_strength.value == "strong":
                character = HabitCharacter.RIGID
            elif tendency_awareness.overall_strength.value == "moderate":
                character = HabitCharacter.HABITUAL

        # Generate description
        description = _generate_tendency_description(
            presence, character, has_strong, has_fading
        )

        return TendencyStateView(
            presence=presence,
            character=character,
            tendency_count=tendency_count,
            has_strong_habits=has_strong,
            has_fading_habits=has_fading,
            description=description,
        )

    except Exception:
        return TendencyStateView.undefined()


def _generate_tendency_description(
    presence: HabitPresence,
    character: HabitCharacter,
    has_strong: bool,
    has_fading: bool
) -> str:
    """Generate human-readable tendency description"""
    parts = []

    if presence == HabitPresence.NONE:
        return "No habitual patterns"
    elif presence == HabitPresence.EMERGING:
        parts.append("Patterns emerging")
    elif presence == HabitPresence.FORMING:
        parts.append("Patterns forming")
    elif presence == HabitPresence.ESTABLISHED:
        parts.append("Established patterns")

    if character == HabitCharacter.EXPLORATORY:
        parts.append("with exploratory behavior")
    elif character == HabitCharacter.SLIGHT_PATTERN:
        parts.append("with slight repetition")
    elif character == HabitCharacter.HABITUAL:
        parts.append("with habitual tendencies")
    elif character == HabitCharacter.RIGID:
        parts.append("with rigid habits")

    if has_fading:
        parts.append("and some fading")

    return ", ".join(parts) if parts else "Tendency state observed"


def observe_direction_state(
    proto_goal_system: Any,
    config: SelfModelConfig = SelfModelConfig()
) -> DirectionStateView:
    """
    Observe proto-goal direction and generate abstract view.

    Args:
        proto_goal_system: ProtoGoalVectorSystem instance (READ-ONLY)
        config: Configuration for thresholds

    Returns:
        Abstract DirectionStateView
    """
    if proto_goal_system is None:
        return DirectionStateView.undefined()

    try:
        # Get vectors (READ-ONLY)
        vectors = proto_goal_system.get_vectors()
        vector_count = len(vectors)

        if vector_count == 0:
            return DirectionStateView(
                clarity=DirectionClarity.UNDEFINED,
                convergence=DirectionConvergence.UNDEFINED,
                vector_count=0,
                has_dominant_direction=False,
                description="No direction vectors present",
            )

        # Get strongest vectors for analysis
        strongest = proto_goal_system.get_strongest_vectors(3)

        # Determine clarity based on strongest magnitude
        max_magnitude = max(v.magnitude for v in vectors) if vectors else 0
        if max_magnitude >= config.direction_clear_min_magnitude:
            clarity = DirectionClarity.CLEAR
        elif max_magnitude >= config.direction_clear_min_magnitude / 2:
            clarity = DirectionClarity.UNCERTAIN
        else:
            clarity = DirectionClarity.UNCLEAR

        # Determine convergence by comparing top vectors
        if len(strongest) <= 1:
            convergence = DirectionConvergence.UNDEFINED
            has_dominant = len(strongest) == 1
        else:
            # Calculate similarity between top vectors
            similarities = []
            for i, v1 in enumerate(strongest):
                for v2 in strongest[i+1:]:
                    sim = _calculate_direction_similarity(v1.direction, v2.direction)
                    similarities.append(sim)

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0

            if avg_similarity >= config.direction_convergence_similarity:
                convergence = DirectionConvergence.CONVERGENT
            elif avg_similarity >= config.direction_convergence_similarity / 2:
                convergence = DirectionConvergence.PARALLEL
            elif avg_similarity >= 0:
                convergence = DirectionConvergence.DIVERGENT
            else:
                convergence = DirectionConvergence.SCATTERED

            # Dominant if top vector is significantly stronger
            has_dominant = (
                len(strongest) >= 2 and
                strongest[0].magnitude > strongest[1].magnitude * 1.5
            )

        # Generate description
        description = _generate_direction_description(
            clarity, convergence, has_dominant
        )

        return DirectionStateView(
            clarity=clarity,
            convergence=convergence,
            vector_count=vector_count,
            has_dominant_direction=has_dominant,
            description=description,
        )

    except Exception:
        return DirectionStateView.undefined()


def _calculate_direction_similarity(dir1: dict, dir2: dict) -> float:
    """Calculate cosine similarity between two direction dictionaries"""
    if not dir1 or not dir2:
        return 0.0

    # Get common keys
    all_keys = set(dir1.keys()) | set(dir2.keys())
    if not all_keys:
        return 0.0

    # Calculate dot product and magnitudes
    dot_product = 0.0
    mag1 = 0.0
    mag2 = 0.0

    for key in all_keys:
        v1 = dir1.get(key, 0.0)
        v2 = dir2.get(key, 0.0)
        dot_product += v1 * v2
        mag1 += v1 * v1
        mag2 += v2 * v2

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / ((mag1 ** 0.5) * (mag2 ** 0.5))


def _generate_direction_description(
    clarity: DirectionClarity,
    convergence: DirectionConvergence,
    has_dominant: bool
) -> str:
    """Generate human-readable direction description"""
    parts = []

    if clarity == DirectionClarity.CLEAR:
        parts.append("Clear direction")
    elif clarity == DirectionClarity.UNCERTAIN:
        parts.append("Uncertain direction")
    elif clarity == DirectionClarity.UNCLEAR:
        parts.append("Unclear direction")
    else:
        return "Direction undefined"

    if convergence == DirectionConvergence.CONVERGENT:
        parts.append("with converging tendencies")
    elif convergence == DirectionConvergence.PARALLEL:
        parts.append("with parallel tendencies")
    elif convergence == DirectionConvergence.DIVERGENT:
        parts.append("with diverging tendencies")
    elif convergence == DirectionConvergence.SCATTERED:
        parts.append("with scattered tendencies")

    if has_dominant:
        parts.append("and dominant path")

    return ", ".join(parts) if parts else "Direction state observed"


def observe_value_state(
    value_orientation: Any,
    previous_orientation: Any = None,
    turn_count: int = 0,
    config: SelfModelConfig = SelfModelConfig()
) -> ValueStateView:
    """
    Observe value orientation and generate abstract view.

    Args:
        value_orientation: ValueOrientation instance (READ-ONLY)
        previous_orientation: Previous ValueOrientation for change detection
        turn_count: Current turn for recency calculation
        config: Configuration for thresholds

    Returns:
        Abstract ValueStateView
    """
    if value_orientation is None:
        return ValueStateView.undefined()

    try:
        # Get confidence levels (READ-ONLY)
        confidences = [
            value_orientation.confidence_a,
            value_orientation.confidence_b,
            value_orientation.confidence_c,
            value_orientation.confidence_d,
            value_orientation.confidence_e,
        ]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)

        # Determine stability
        if avg_confidence < config.value_confidence_unstable_max:
            stability = ValueStability.UNSTABLE
        elif avg_confidence < config.value_confidence_shifting_max:
            stability = ValueStability.SHIFTING
        elif avg_confidence < config.value_confidence_stable_max:
            stability = ValueStability.STABLE
        else:
            stability = ValueStability.ANCHORED

        # Determine clarity based on dimension values
        dimensions = [
            abs(value_orientation.dim_a),
            abs(value_orientation.dim_b),
            abs(value_orientation.dim_c),
            abs(value_orientation.dim_d),
            abs(value_orientation.dim_e),
        ]
        max_dim = max(dimensions)

        if max_dim < 0.1:
            clarity = ValueClarity.UNDEFINED_VALUES
        elif max_dim < 0.2:
            clarity = ValueClarity.EMERGING
        elif max_dim < 0.4:
            clarity = ValueClarity.FORMING
        else:
            clarity = ValueClarity.DEFINED

        # Check for strong orientations
        has_strong = max_dim >= config.value_strong_orientation_threshold

        # Check for recent changes
        is_recently_changed = False
        if previous_orientation is not None:
            # Calculate distance
            prev_dims = [
                previous_orientation.dim_a,
                previous_orientation.dim_b,
                previous_orientation.dim_c,
                previous_orientation.dim_d,
                previous_orientation.dim_e,
            ]
            curr_dims = [
                value_orientation.dim_a,
                value_orientation.dim_b,
                value_orientation.dim_c,
                value_orientation.dim_d,
                value_orientation.dim_e,
            ]
            distance = sum((c - p) ** 2 for c, p in zip(curr_dims, prev_dims)) ** 0.5
            is_recently_changed = distance > 0.1

        # Generate description
        description = _generate_value_description(
            stability, clarity, has_strong, is_recently_changed
        )

        return ValueStateView(
            stability=stability,
            clarity=clarity,
            has_strong_orientations=has_strong,
            is_recently_changed=is_recently_changed,
            description=description,
        )

    except Exception:
        return ValueStateView.undefined()


def _generate_value_description(
    stability: ValueStability,
    clarity: ValueClarity,
    has_strong: bool,
    is_recently_changed: bool
) -> str:
    """Generate human-readable value description"""
    parts = []

    if stability == ValueStability.UNSTABLE:
        parts.append("Values unstable")
    elif stability == ValueStability.SHIFTING:
        parts.append("Values shifting")
    elif stability == ValueStability.STABLE:
        parts.append("Values stable")
    elif stability == ValueStability.ANCHORED:
        parts.append("Values anchored")
    else:
        return "Value orientation undefined"

    if clarity == ValueClarity.EMERGING:
        parts.append("and emerging")
    elif clarity == ValueClarity.FORMING:
        parts.append("and forming")
    elif clarity == ValueClarity.DEFINED:
        parts.append("and defined")

    if is_recently_changed:
        parts.append("with recent change")

    return ", ".join(parts) if parts else "Value state observed"


# =============================================================================
# Integrated Description Generation
# =============================================================================

def generate_integrated_description(
    emotional: EmotionalStateView,
    responsibility: ResponsibilityStateView,
    tendency: TendencyStateView,
    direction: DirectionStateView,
    value: ValueStateView
) -> str:
    """
    Generate an integrated description of self-state.

    This creates a holistic view without determining identity
    from any single component.

    Note: This description is for self-reference only,
    NOT for decision making or justification.
    """
    parts = []

    # Emotional foundation
    if emotional.intensity != EmotionalIntensity.UNDEFINED:
        if emotional.intensity == EmotionalIntensity.CALM:
            parts.append("In a calm state")
        elif emotional.intensity == EmotionalIntensity.OVERWHELMING:
            parts.append("In an overwhelmed state")
        else:
            parts.append("Emotionally present")

        if emotional.harmony == EmotionalHarmony.CONFLICTED:
            parts.append("with inner conflict")

    # Responsibility layer
    if responsibility.burden_level != BurdenLevel.UNDEFINED:
        if responsibility.burden_level in (BurdenLevel.BURDENED, BurdenLevel.HEAVY):
            parts.append("carrying significant burden")
        elif responsibility.burden_level == BurdenLevel.UNBURDENED:
            parts.append("unburdened")

        if responsibility.trend == BurdenTrend.ACCUMULATING:
            parts.append("and accumulating weight")

    # Tendency layer
    if tendency.presence != HabitPresence.UNDEFINED:
        if tendency.presence == HabitPresence.ESTABLISHED:
            if tendency.character == HabitCharacter.RIGID:
                parts.append("with established rigid patterns")
            else:
                parts.append("with established patterns")
        elif tendency.presence != HabitPresence.NONE:
            parts.append("with forming tendencies")

    # Direction layer
    if direction.clarity != DirectionClarity.UNDEFINED:
        if direction.clarity == DirectionClarity.CLEAR:
            if direction.has_dominant_direction:
                parts.append("oriented toward clear direction")
            else:
                parts.append("with clear general direction")
        elif direction.clarity == DirectionClarity.UNCLEAR:
            parts.append("without clear direction")

    # Value foundation
    if value.stability != ValueStability.UNDEFINED:
        if value.stability == ValueStability.ANCHORED:
            parts.append("grounded in stable values")
        elif value.stability == ValueStability.UNSTABLE:
            parts.append("with shifting values")

        if value.is_recently_changed:
            parts.append("experiencing value change")

    if not parts:
        return "Current self-state is provisional and undefined"

    # Join with appropriate connectors
    result = parts[0]
    for part in parts[1:]:
        if part.startswith("with") or part.startswith("and"):
            result += " " + part
        else:
            result += ", " + part

    return result + "."


# =============================================================================
# Self-Model System
# =============================================================================

@dataclass
class SelfModelState:
    """
    Internal state of the Self-Model system.

    Tracks observation history for trend detection.
    This state itself is NOT part of decision making.
    """
    last_responsibility_weight: Optional[float] = None
    last_value_orientation: Any = None
    turn_count: int = 0
    snapshot_count: int = 0


class SelfModelSystem:
    """
    Self-Model System (自己状態統合モデル)

    Provides a unified, abstract view of internal psychological state.

    CRITICAL CONSTRAINTS:
    - READ-ONLY observation of other systems
    - STRICTLY NO IMPACT on decision making
    - Numbers are abstracted into categories
    - Self-model is always provisional

    This system generates "how the current self appears" without:
    - Defining personality or beliefs
    - Evaluating self as good/bad
    - Providing reasons or justifications
    - Modifying any other system's state
    """

    def __init__(self, config: Optional[SelfModelConfig] = None):
        """Initialize with optional configuration"""
        self._config = config or SelfModelConfig()
        self._state = SelfModelState()

    def observe(
        self,
        emotion_vector: Any = None,
        responsibility_state: Any = None,
        tendency_system: Any = None,
        tendency_awareness: Any = None,
        proto_goal_system: Any = None,
        value_orientation: Any = None,
    ) -> SelfStateView:
        """
        Generate a unified self-state view by observing all systems.

        This is the main entry point for generating self-model snapshots.
        All observations are READ-ONLY.

        Args:
            emotion_vector: EmotionVector instance
            responsibility_state: ResponsibilityState instance
            tendency_system: RepeatedTendencySystem instance
            tendency_awareness: TendencyAwareness instance (optional)
            proto_goal_system: ProtoGoalVectorSystem instance
            value_orientation: ValueOrientation instance

        Returns:
            SelfStateView representing current self-state
        """
        # Observe each component
        emotional = observe_emotional_state(emotion_vector, self._config)

        responsibility = observe_responsibility_state(
            responsibility_state,
            self._state.last_responsibility_weight,
            self._config
        )

        tendency = observe_tendency_state(
            tendency_system,
            tendency_awareness,
            self._config
        )

        direction = observe_direction_state(proto_goal_system, self._config)

        value = observe_value_state(
            value_orientation,
            self._state.last_value_orientation,
            self._state.turn_count,
            self._config
        )

        # Generate integrated description
        integrated = generate_integrated_description(
            emotional, responsibility, tendency, direction, value
        )

        # Check completeness
        undefined = []
        if emotional.spread == EmotionalSpread.UNDEFINED:
            undefined.append("emotional")
        if responsibility.burden_level == BurdenLevel.UNDEFINED:
            undefined.append("responsibility")
        if tendency.presence == HabitPresence.UNDEFINED:
            undefined.append("tendency")
        if direction.clarity == DirectionClarity.UNDEFINED:
            undefined.append("direction")
        if value.stability == ValueStability.UNDEFINED:
            undefined.append("value")

        is_complete = len(undefined) == 0

        # Update internal state for next observation
        if responsibility_state is not None:
            self._state.last_responsibility_weight = responsibility_state.total_weight
        if value_orientation is not None:
            self._state.last_value_orientation = value_orientation

        self._state.snapshot_count += 1

        # Create snapshot
        snapshot = SelfStateView(
            emotional=emotional,
            responsibility=responsibility,
            tendency=tendency,
            direction=direction,
            value=value,
            timestamp=time.time(),
            snapshot_id=f"self_snapshot_{self._state.snapshot_count}",
            is_complete=is_complete,
            integrated_description=integrated,
        )

        return snapshot

    def advance_turn(self) -> None:
        """Advance internal turn counter"""
        self._state.turn_count += 1

    def get_turn_count(self) -> int:
        """Get current turn count"""
        return self._state.turn_count

    def get_snapshot_count(self) -> int:
        """Get total snapshots generated"""
        return self._state.snapshot_count

    def reset_tracking(self) -> None:
        """Reset trend tracking state (but not turn count)"""
        self._state.last_responsibility_weight = None
        self._state.last_value_orientation = None


# =============================================================================
# Integration with SelfReferenceSystem
# =============================================================================

def generate_self_model_tags(
    view: SelfStateView,
    scale: float = 1.0
) -> list[dict[str, Any]]:
    """
    Generate tags from SelfStateView for SelfReferenceSystem integration.

    These tags can be used for introspection logging and self-description,
    but MUST NOT be used for decision making.

    Args:
        view: SelfStateView to convert
        scale: Weight scaling factor (default 1.0)

    Returns:
        List of tag dictionaries compatible with SelfReferenceSystem
    """
    tags = []

    # Emotional tags
    if view.emotional.intensity != EmotionalIntensity.UNDEFINED:
        tags.append({
            "category": "SELF_MODEL_EMOTIONAL",
            "label": f"emotional_{view.emotional.intensity.value}",
            "description": view.emotional.description,
            "weight": 0.1 * scale,  # Low weight - observation only
        })

        if view.emotional.harmony == EmotionalHarmony.CONFLICTED:
            tags.append({
                "category": "SELF_MODEL_EMOTIONAL",
                "label": "emotional_conflict",
                "description": "Inner emotional conflict present",
                "weight": 0.05 * scale,
            })

    # Responsibility tags
    if view.responsibility.burden_level != BurdenLevel.UNDEFINED:
        tags.append({
            "category": "SELF_MODEL_RESPONSIBILITY",
            "label": f"burden_{view.responsibility.burden_level.value}",
            "description": view.responsibility.description,
            "weight": 0.1 * scale,
        })

        if view.responsibility.trend == BurdenTrend.ACCUMULATING:
            tags.append({
                "category": "SELF_MODEL_RESPONSIBILITY",
                "label": "burden_accumulating",
                "description": "Responsibility accumulating",
                "weight": 0.05 * scale,
            })

    # Tendency tags
    if view.tendency.presence != HabitPresence.UNDEFINED:
        tags.append({
            "category": "SELF_MODEL_TENDENCY",
            "label": f"habit_{view.tendency.presence.value}",
            "description": view.tendency.description,
            "weight": 0.1 * scale,
        })

        if view.tendency.character == HabitCharacter.RIGID:
            tags.append({
                "category": "SELF_MODEL_TENDENCY",
                "label": "habit_rigid",
                "description": "Rigid habitual patterns",
                "weight": 0.05 * scale,
            })

    # Direction tags
    if view.direction.clarity != DirectionClarity.UNDEFINED:
        tags.append({
            "category": "SELF_MODEL_DIRECTION",
            "label": f"direction_{view.direction.clarity.value}",
            "description": view.direction.description,
            "weight": 0.1 * scale,
        })

        if view.direction.has_dominant_direction:
            tags.append({
                "category": "SELF_MODEL_DIRECTION",
                "label": "direction_dominant",
                "description": "Dominant direction present",
                "weight": 0.05 * scale,
            })

    # Value tags
    if view.value.stability != ValueStability.UNDEFINED:
        tags.append({
            "category": "SELF_MODEL_VALUE",
            "label": f"value_{view.value.stability.value}",
            "description": view.value.description,
            "weight": 0.1 * scale,
        })

        if view.value.is_recently_changed:
            tags.append({
                "category": "SELF_MODEL_VALUE",
                "label": "value_changing",
                "description": "Values recently changed",
                "weight": 0.05 * scale,
            })

    # Integrated tag
    tags.append({
        "category": "SELF_MODEL_INTEGRATED",
        "label": "self_state",
        "description": view.integrated_description,
        "weight": 0.15 * scale,
    })

    return tags


def get_self_model_summary(view: SelfStateView) -> str:
    """
    Get a human-readable summary of self-state for logging.

    This is for introspection/logging only, NOT for decision input.
    """
    lines = [
        "=== Self-Model Snapshot ===",
        f"ID: {view.snapshot_id}",
        f"Complete: {view.is_complete}",
        "",
        "Components:",
        f"  Emotional: {view.emotional.description}",
        f"  Responsibility: {view.responsibility.description}",
        f"  Tendency: {view.tendency.description}",
        f"  Direction: {view.direction.description}",
        f"  Value: {view.value.description}",
        "",
        "Integrated View:",
        f"  {view.integrated_description}",
    ]

    undefined = view.get_undefined_components()
    if undefined:
        lines.append("")
        lines.append(f"Undefined: {', '.join(undefined)}")

    return "\n".join(lines)


def get_self_model_for_introspection(view: SelfStateView) -> dict[str, Any]:
    """
    Get structured self-model data for IntrospectionTrace integration.

    This provides abstract state information for post-hoc analysis.
    MUST NOT be used as input to decision-making systems.
    """
    return {
        "snapshot_id": view.snapshot_id,
        "timestamp": view.timestamp,
        "is_complete": view.is_complete,
        "emotional": {
            "spread": view.emotional.spread.value,
            "intensity": view.emotional.intensity.value,
            "harmony": view.emotional.harmony.value,
        },
        "responsibility": {
            "burden_level": view.responsibility.burden_level.value,
            "distribution": view.responsibility.distribution.value,
            "trend": view.responsibility.trend.value,
        },
        "tendency": {
            "presence": view.tendency.presence.value,
            "character": view.tendency.character.value,
        },
        "direction": {
            "clarity": view.direction.clarity.value,
            "convergence": view.direction.convergence.value,
        },
        "value": {
            "stability": view.value.stability.value,
            "clarity": view.value.clarity.value,
        },
        "integrated_description": view.integrated_description,
    }


# =============================================================================
# Persistence Support (Optional)
# =============================================================================

def save_self_model_state(state: SelfModelState, path: str) -> None:
    """
    Save self-model tracking state to file.

    Note: The SelfStateView itself is not persisted as it's always
    a fresh snapshot. Only tracking state for trend detection is saved.
    """
    import json

    data = {
        "last_responsibility_weight": state.last_responsibility_weight,
        "turn_count": state.turn_count,
        "snapshot_count": state.snapshot_count,
        # Note: last_value_orientation requires custom serialization
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_self_model_state(path: str) -> SelfModelState:
    """
    Load self-model tracking state from file.
    """
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SelfModelState(
            last_responsibility_weight=data.get("last_responsibility_weight"),
            last_value_orientation=None,  # Must be restored separately
            turn_count=data.get("turn_count", 0),
            snapshot_count=data.get("snapshot_count", 0),
        )
    except FileNotFoundError:
        return SelfModelState()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_empty_view() -> SelfStateView:
    """Create an empty/undefined SelfStateView"""
    return SelfStateView(
        emotional=EmotionalStateView.undefined(),
        responsibility=ResponsibilityStateView.undefined(),
        tendency=TendencyStateView.undefined(),
        direction=DirectionStateView.undefined(),
        value=ValueStateView.undefined(),
        timestamp=time.time(),
        snapshot_id="empty_snapshot",
        is_complete=False,
        integrated_description="Self-state is currently undefined",
    )


def create_config(
    emotion_active_threshold: float = 0.1,
    burden_heavy_max: float = 0.7,
    tendency_strong_threshold: float = 0.1,
    direction_clear_min_magnitude: float = 0.3,
    value_confidence_stable_max: float = 0.7,
) -> SelfModelConfig:
    """Create a custom configuration with common overrides"""
    return SelfModelConfig(
        emotion_active_threshold=emotion_active_threshold,
        burden_heavy_max=burden_heavy_max,
        tendency_strong_threshold=tendency_strong_threshold,
        direction_clear_min_magnitude=direction_clear_min_magnitude,
        value_confidence_stable_max=value_confidence_stable_max,
    )
