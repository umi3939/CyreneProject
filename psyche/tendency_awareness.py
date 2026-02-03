"""
tendency_awareness.py - Self-Awareness of Repeated Tendency (反復傾向の自己認知)

This module provides PURE OBSERVATION of RepeatedTendency for self-description.
It converts raw numerical tendencies into ABSTRACT human-like concepts.

【思想】
人は行動の反復によって性質を獲得するが、
それ以上に重要なのは
「自分がどうなりつつあるかを知ってしまう」ことである。

Key design principles:
- PURE OBSERVATION: Does NOT feedback into Decision Making
- ABSTRACT CONCEPTS: No raw numbers exposed (Low/Medium/High)
- FOR SELF-DESCRIPTION ONLY: Connects to SelfReferenceSystem
- TEMPORARY AWARENESS: Fades when tendency weakens
- NO EVALUATION: Does not judge, fix, or modify tendencies

Usage::

    from psyche.tendency_awareness import (
        TendencyAwareness,
        observe_tendencies,
        generate_awareness_tags,
        StrengthLevel,
        DurationLevel,
    )

    # Observe tendencies (read-only)
    awareness = observe_tendencies(repeated_tendency_system)

    # Generate self-reference tags for SelfReferenceSystem
    tags = generate_awareness_tags(awareness)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .goal_candidates import CandidateCategory
from .repeated_tendency import RepeatedTendencySystem, Tendency


# ── Abstract Concept Enums (No Raw Numbers) ─────────────────────


class StrengthLevel(Enum):
    """
    Abstract strength level for tendency awareness.

    These are FUZZY human-like concepts, not precise measurements.
    Used for self-description, NOT for decision making.
    """
    NONE = "none"              # No noticeable tendency
    SLIGHT = "slight"          # Barely noticeable bias
    MODERATE = "moderate"      # Noticeable habit forming
    STRONG = "strong"          # Clear established habit


class DurationLevel(Enum):
    """
    Abstract duration level for tendency awareness.

    Represents how long the tendency has been present.
    Uses fuzzy concepts, not precise turn counts.
    """
    RECENT = "recent"          # Just started forming
    ESTABLISHED = "established"  # Has been present for a while
    PERSISTENT = "persistent"  # Long-standing tendency


class ConfidenceLevel(Enum):
    """
    Abstract confidence level for tendency awareness.

    Represents how established/reinforced the tendency is.
    """
    UNCERTAIN = "uncertain"    # Few reinforcements
    FORMING = "forming"        # Some reinforcements
    ESTABLISHED = "established"  # Many reinforcements


class AwarenessType(Enum):
    """
    Type of awareness about a tendency.

    These are the kinds of "noticing" that can occur.
    """
    HABIT_FORMING = "habit_forming"      # "I seem to be doing this a lot"
    SLIGHT_BIAS = "slight_bias"          # "I might have a slight preference"
    STRONG_HABIT = "strong_habit"        # "This has become a habit"
    FADING_HABIT = "fading_habit"        # "I used to do this more"


# ── Awareness Data Structures ───────────────────────────────────


@dataclass
class TendencyAwarenessItem:
    """
    A single awareness about a tendency.

    This is a TEMPORARY self-perception, not a permanent definition.
    Contains only ABSTRACT concepts, no raw numbers.
    """
    # What kind of awareness this is
    awareness_type: AwarenessType = AwarenessType.SLIGHT_BIAS

    # Abstract category (from CandidateCategory)
    category: CandidateCategory = CandidateCategory.EXPLORATION

    # Abstract levels (fuzzy, not precise)
    strength_level: StrengthLevel = StrengthLevel.NONE
    duration_level: DurationLevel = DurationLevel.RECENT
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN

    # Human-readable description for self-reference
    description: str = ""

    # Internal reference (not exposed to decisions)
    _tendency_id: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization (without internal refs)."""
        return {
            "awareness_type": self.awareness_type.value,
            "category": self.category.value,
            "strength_level": self.strength_level.value,
            "duration_level": self.duration_level.value,
            "confidence_level": self.confidence_level.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TendencyAwarenessItem:
        """Create from dict."""
        return cls(
            awareness_type=AwarenessType(data.get("awareness_type", "slight_bias")),
            category=CandidateCategory(data.get("category", "exploration")),
            strength_level=StrengthLevel(data.get("strength_level", "none")),
            duration_level=DurationLevel(data.get("duration_level", "recent")),
            confidence_level=ConfidenceLevel(data.get("confidence_level", "uncertain")),
            description=data.get("description", ""),
        )


@dataclass
class TendencyAwareness:
    """
    Collection of awareness items about current tendencies.

    This is PURE OBSERVATION data for self-description.
    Does NOT affect decision making in any way.
    """
    # Active awareness items
    items: list[TendencyAwarenessItem] = field(default_factory=list)

    # Whether any notable tendency exists
    has_awareness: bool = False

    # Summary for self-reference (no numbers)
    dominant_category: Optional[CandidateCategory] = None
    overall_strength: StrengthLevel = StrengthLevel.NONE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "items": [item.to_dict() for item in self.items],
            "has_awareness": self.has_awareness,
            "dominant_category": self.dominant_category.value if self.dominant_category else None,
            "overall_strength": self.overall_strength.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TendencyAwareness:
        """Create from dict."""
        items = [TendencyAwarenessItem.from_dict(i) for i in data.get("items", [])]
        dominant = data.get("dominant_category")
        return cls(
            items=items,
            has_awareness=data.get("has_awareness", False),
            dominant_category=CandidateCategory(dominant) if dominant else None,
            overall_strength=StrengthLevel(data.get("overall_strength", "none")),
        )

    def get_by_category(self, category: CandidateCategory) -> list[TendencyAwarenessItem]:
        """Get awareness items for a specific category."""
        return [item for item in self.items if item.category == category]

    def get_strongest(self) -> Optional[TendencyAwarenessItem]:
        """Get the strongest awareness item (if any)."""
        if not self.items:
            return None

        # Order by strength level
        strength_order = {
            StrengthLevel.STRONG: 3,
            StrengthLevel.MODERATE: 2,
            StrengthLevel.SLIGHT: 1,
            StrengthLevel.NONE: 0,
        }
        return max(self.items, key=lambda x: strength_order.get(x.strength_level, 0))


# ── Configuration ───────────────────────────────────────────────


@dataclass
class AwarenessConfig:
    """
    Configuration for tendency awareness generation.

    Defines thresholds for when awareness is generated.
    All values are abstract - no hardcoded meanings.
    """
    # Minimum tendency strength to generate awareness (0-1)
    # Below this, tendency is not "noticed"
    min_strength_for_awareness: float = 0.03

    # Thresholds for strength levels (abstract boundaries)
    slight_threshold: float = 0.03
    moderate_threshold: float = 0.08
    strong_threshold: float = 0.12

    # Thresholds for duration levels (turn counts)
    recent_max_turns: int = 10
    established_max_turns: int = 30

    # Thresholds for confidence levels (reinforcement counts)
    forming_min_reinforcements: int = 3
    established_min_reinforcements: int = 6

    # Threshold for "fading" detection (consecutive misses)
    fading_miss_threshold: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_strength_for_awareness": self.min_strength_for_awareness,
            "slight_threshold": self.slight_threshold,
            "moderate_threshold": self.moderate_threshold,
            "strong_threshold": self.strong_threshold,
            "recent_max_turns": self.recent_max_turns,
            "established_max_turns": self.established_max_turns,
            "forming_min_reinforcements": self.forming_min_reinforcements,
            "established_min_reinforcements": self.established_min_reinforcements,
            "fading_miss_threshold": self.fading_miss_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AwarenessConfig:
        return cls(
            min_strength_for_awareness=data.get("min_strength_for_awareness", 0.03),
            slight_threshold=data.get("slight_threshold", 0.03),
            moderate_threshold=data.get("moderate_threshold", 0.08),
            strong_threshold=data.get("strong_threshold", 0.12),
            recent_max_turns=data.get("recent_max_turns", 10),
            established_max_turns=data.get("established_max_turns", 30),
            forming_min_reinforcements=data.get("forming_min_reinforcements", 3),
            established_min_reinforcements=data.get("established_min_reinforcements", 6),
            fading_miss_threshold=data.get("fading_miss_threshold", 5),
        )


# ── Core Observation Functions (PURE - No Side Effects) ─────────


def _classify_strength(strength: float, config: AwarenessConfig) -> StrengthLevel:
    """
    Convert raw strength to abstract level.

    This is an internal function - the result is abstract, not numeric.
    """
    if strength >= config.strong_threshold:
        return StrengthLevel.STRONG
    elif strength >= config.moderate_threshold:
        return StrengthLevel.MODERATE
    elif strength >= config.slight_threshold:
        return StrengthLevel.SLIGHT
    else:
        return StrengthLevel.NONE


def _classify_duration(
    first_formed_turn: int,
    current_turn: int,
    config: AwarenessConfig,
) -> DurationLevel:
    """
    Convert turn count to abstract duration level.

    This is an internal function - the result is abstract, not numeric.
    """
    age = current_turn - first_formed_turn

    if age <= config.recent_max_turns:
        return DurationLevel.RECENT
    elif age <= config.established_max_turns:
        return DurationLevel.ESTABLISHED
    else:
        return DurationLevel.PERSISTENT


def _classify_confidence(
    reinforcement_count: int,
    config: AwarenessConfig,
) -> ConfidenceLevel:
    """
    Convert reinforcement count to abstract confidence level.

    This is an internal function - the result is abstract, not numeric.
    """
    if reinforcement_count >= config.established_min_reinforcements:
        return ConfidenceLevel.ESTABLISHED
    elif reinforcement_count >= config.forming_min_reinforcements:
        return ConfidenceLevel.FORMING
    else:
        return ConfidenceLevel.UNCERTAIN


def _determine_awareness_type(
    strength_level: StrengthLevel,
    confidence_level: ConfidenceLevel,
    consecutive_misses: int,
    config: AwarenessConfig,
) -> AwarenessType:
    """
    Determine what type of awareness to generate.

    Based on abstract levels, not raw numbers.
    """
    # Check for fading first
    if consecutive_misses >= config.fading_miss_threshold:
        return AwarenessType.FADING_HABIT

    # Strong + Established = Strong Habit
    if (strength_level == StrengthLevel.STRONG and
            confidence_level == ConfidenceLevel.ESTABLISHED):
        return AwarenessType.STRONG_HABIT

    # Moderate or forming = Habit Forming
    if (strength_level == StrengthLevel.MODERATE or
            confidence_level == ConfidenceLevel.FORMING):
        return AwarenessType.HABIT_FORMING

    # Default = Slight Bias
    return AwarenessType.SLIGHT_BIAS


def _generate_description(
    awareness_type: AwarenessType,
    category: CandidateCategory,
    strength_level: StrengthLevel,
) -> str:
    """
    Generate human-readable description for self-reference.

    These are vague, self-aware descriptions, not precise statements.
    """
    category_descriptions = {
        CandidateCategory.APPROACH: "engaging with situations",
        CandidateCategory.AVOIDANCE: "stepping back from situations",
        CandidateCategory.CONNECTION: "seeking connection",
        CandidateCategory.ISOLATION: "preferring solitude",
        CandidateCategory.EXPRESSION: "expressing myself",
        CandidateCategory.ABSORPTION: "taking things in",
        CandidateCategory.EXPLORATION: "exploring new things",
        CandidateCategory.MAINTENANCE: "maintaining the status quo",
    }

    category_desc = category_descriptions.get(category, "certain behaviors")

    if awareness_type == AwarenessType.STRONG_HABIT:
        return f"I notice I've developed a habit of {category_desc}"
    elif awareness_type == AwarenessType.HABIT_FORMING:
        return f"I seem to be {category_desc} more often lately"
    elif awareness_type == AwarenessType.FADING_HABIT:
        return f"I used to {category_desc} more, but less so recently"
    else:  # SLIGHT_BIAS
        return f"I might have a slight tendency toward {category_desc}"


def observe_tendency(
    tendency: Tendency,
    current_turn: int,
    config: Optional[AwarenessConfig] = None,
) -> Optional[TendencyAwarenessItem]:
    """
    Observe a single tendency and generate awareness (if notable).

    This is a PURE OBSERVATION function:
    - READ-ONLY: Does not modify the tendency
    - NO FEEDBACK: Result is for self-description only
    - ABSTRACT OUTPUT: No raw numbers in the result

    Args:
        tendency: The Tendency to observe
        current_turn: Current turn number for duration calculation
        config: Optional configuration

    Returns:
        TendencyAwarenessItem if tendency is notable, None otherwise
    """
    cfg = config or AwarenessConfig()

    # Check if tendency is strong enough to notice
    if tendency.strength < cfg.min_strength_for_awareness:
        return None

    # Convert to abstract levels
    strength_level = _classify_strength(tendency.strength, cfg)
    duration_level = _classify_duration(
        tendency.first_formed_turn,
        current_turn,
        cfg,
    )
    confidence_level = _classify_confidence(
        tendency.total_reinforcements,
        cfg,
    )

    # Skip if strength is NONE (below awareness threshold)
    if strength_level == StrengthLevel.NONE:
        return None

    # Determine awareness type
    awareness_type = _determine_awareness_type(
        strength_level,
        confidence_level,
        tendency.consecutive_misses,
        cfg,
    )

    # Generate description
    description = _generate_description(
        awareness_type,
        tendency.pattern.category,
        strength_level,
    )

    return TendencyAwarenessItem(
        awareness_type=awareness_type,
        category=tendency.pattern.category,
        strength_level=strength_level,
        duration_level=duration_level,
        confidence_level=confidence_level,
        description=description,
        _tendency_id=tendency.tendency_id,
    )


def observe_tendencies(
    system: RepeatedTendencySystem,
    config: Optional[AwarenessConfig] = None,
) -> TendencyAwareness:
    """
    Observe all tendencies in a system and generate awareness.

    This is the main entry point for tendency awareness.

    PURE OBSERVATION:
    - Does NOT modify the RepeatedTendencySystem
    - Does NOT affect decision making
    - Output is for SELF-DESCRIPTION only

    Args:
        system: The RepeatedTendencySystem to observe
        config: Optional configuration

    Returns:
        TendencyAwareness containing abstract awareness of tendencies
    """
    cfg = config or AwarenessConfig()

    # Get current turn from system state
    current_turn = system.state.turn_count

    # Observe each tendency
    items: list[TendencyAwarenessItem] = []
    for tendency in system.get_tendencies():
        item = observe_tendency(tendency, current_turn, cfg)
        if item is not None:
            items.append(item)

    # Determine overall state
    has_awareness = len(items) > 0
    dominant_category = None
    overall_strength = StrengthLevel.NONE

    if items:
        # Find dominant category (strongest tendency)
        strongest = max(
            items,
            key=lambda x: {
                StrengthLevel.STRONG: 3,
                StrengthLevel.MODERATE: 2,
                StrengthLevel.SLIGHT: 1,
                StrengthLevel.NONE: 0,
            }.get(x.strength_level, 0),
        )
        dominant_category = strongest.category
        overall_strength = strongest.strength_level

    return TendencyAwareness(
        items=items,
        has_awareness=has_awareness,
        dominant_category=dominant_category,
        overall_strength=overall_strength,
    )


# ── Self-Reference Integration ──────────────────────────────────


def generate_awareness_tags(
    awareness: TendencyAwareness,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate self-reference tags from tendency awareness.

    These tags are for the SelfReferenceSystem to use in self-description.
    They contain ABSTRACT concepts only, no raw numbers.

    IMPORTANT: These tags do NOT affect decision making.
    They are purely for self-description and introspection.

    Args:
        awareness: TendencyAwareness from observe_tendencies()
        scale: Optional weight scale for tags

    Returns:
        List of tag dicts compatible with SelfReferenceSystem
    """
    if not awareness.has_awareness:
        return []

    tags = []

    for item in awareness.items:
        # Create tag with abstract label (no numbers)
        tag = {
            "category": "tendency",  # SelfTagCategory.TENDENCY
            "label": f"tendency_{item.awareness_type.value}_{item.category.value}",
            "source_value": 0.0,  # Intentionally hidden - use abstract level
            "weight": scale,
            # Additional abstract metadata for self-description
            "metadata": {
                "strength": item.strength_level.value,
                "duration": item.duration_level.value,
                "confidence": item.confidence_level.value,
                "description": item.description,
            },
        }
        tags.append(tag)

    # Add overall awareness tag
    if awareness.dominant_category:
        tags.append({
            "category": "tendency",
            "label": f"tendency_overall_{awareness.overall_strength.value}",
            "source_value": 0.0,
            "weight": scale,
            "metadata": {
                "dominant_category": awareness.dominant_category.value,
            },
        })

    return tags


def get_awareness_summary(awareness: TendencyAwareness) -> str:
    """
    Get human-readable summary of tendency awareness.

    For logging and debugging purposes.
    Contains ABSTRACT descriptions only.
    """
    if not awareness.has_awareness:
        return "No notable tendencies noticed"

    parts = []
    for item in awareness.items:
        parts.append(f"  - {item.description} ({item.strength_level.value})")

    return "Current self-awareness about tendencies:\n" + "\n".join(parts)


def get_awareness_for_introspection(awareness: TendencyAwareness) -> dict[str, Any]:
    """
    Get awareness data formatted for introspection trace.

    This provides ABSTRACT data for internal logging,
    without exposing raw numerical values.

    Returns:
        Dict suitable for introspection trace logging
    """
    if not awareness.has_awareness:
        return {
            "has_tendency_awareness": False,
            "observation_note": "No tendencies noticed at this time",
        }

    return {
        "has_tendency_awareness": True,
        "awareness_count": len(awareness.items),
        "overall_strength": awareness.overall_strength.value,
        "dominant_category": (
            awareness.dominant_category.value if awareness.dominant_category else None
        ),
        "descriptions": [item.description for item in awareness.items],
        "observation_note": (
            "These are temporary self-perceptions about behavioral patterns. "
            "They do NOT affect decision making - they are for self-description only."
        ),
    }


# ── Factory Functions ───────────────────────────────────────────


def create_config(
    min_strength_for_awareness: float = 0.03,
    slight_threshold: float = 0.03,
    moderate_threshold: float = 0.08,
    strong_threshold: float = 0.12,
    recent_max_turns: int = 10,
    established_max_turns: int = 30,
    forming_min_reinforcements: int = 3,
    established_min_reinforcements: int = 6,
    fading_miss_threshold: int = 5,
) -> AwarenessConfig:
    """Factory function to create AwarenessConfig."""
    return AwarenessConfig(
        min_strength_for_awareness=min_strength_for_awareness,
        slight_threshold=slight_threshold,
        moderate_threshold=moderate_threshold,
        strong_threshold=strong_threshold,
        recent_max_turns=recent_max_turns,
        established_max_turns=established_max_turns,
        forming_min_reinforcements=forming_min_reinforcements,
        established_min_reinforcements=established_min_reinforcements,
        fading_miss_threshold=fading_miss_threshold,
    )


def create_empty_awareness() -> TendencyAwareness:
    """Create an empty TendencyAwareness (no tendencies noticed)."""
    return TendencyAwareness()
