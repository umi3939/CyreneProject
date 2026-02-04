"""
Temporal Self-Difference Awareness (自己モデル差分認知)

This module provides awareness of how the self-state has changed over time.
It compares current SelfStateView with past snapshots and generates abstract
descriptions of the differences.

CRITICAL DESIGN PRINCIPLE:
- This is for AWARENESS only, not for control or correction
- STRICTLY NO IMPACT on decision making
- NO JUDGMENT (good/bad, progress/regression)
- Differences are not problems to solve, just facts to know
- Differences naturally shrink when states converge again

Philosophy:
自己とは固定された定義ではなく、
時間をまたいで連続しているという「感覚」によってのみ成立する。

「昨日の自分と、今の自分が少し違う」という事実を、
評価も解釈もせず、ただ知ってしまうための構造である。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque
import time

from .self_model import (
    SelfStateView,
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
# Abstract Enums for Difference Description
# These describe the NATURE of change without judgment
# =============================================================================

class DifferenceMagnitude(Enum):
    """
    Magnitude of difference between self-states.
    NOT a score - just an abstract category.
    """
    NONE = "none"                # No meaningful difference
    MINIMAL = "minimal"          # Very small fluctuation (noise)
    NOTICEABLE = "noticeable"    # Detectable but small change
    SIGNIFICANT = "significant"  # Clear change
    SUBSTANTIAL = "substantial"  # Large change
    UNDEFINED = "undefined"      # Cannot determine (missing data)


class ChangeNature(Enum):
    """
    Nature/character of the change.
    NOT evaluative (no good/bad) - just descriptive.
    """
    STABLE = "stable"              # Little to no change over time
    FLUCTUATING = "fluctuating"    # Small back-and-forth changes
    SHIFTING = "shifting"          # Gradual movement in a direction
    TRANSFORMED = "transformed"    # Substantial change has occurred
    RETURNING = "returning"        # Moving back toward previous state
    UNDEFINED = "undefined"        # Cannot determine


class ComponentChangeType(Enum):
    """
    Type of change for a specific component.
    """
    UNCHANGED = "unchanged"        # Same category
    INTENSIFIED = "intensified"    # Moved to stronger/more category
    SOFTENED = "softened"          # Moved to lighter/less category
    SHIFTED = "shifted"            # Changed to different type
    UNDEFINED = "undefined"        # One or both undefined


class TemporalSpan(Enum):
    """
    Time span being compared.
    """
    IMMEDIATE = "immediate"        # Very recent (last few snapshots)
    SHORT_TERM = "short_term"      # Recent (several turns)
    MEDIUM_TERM = "medium_term"    # Extended period
    LONG_TERM = "long_term"        # Long period
    UNDEFINED = "undefined"


# =============================================================================
# Component Difference Structures
# =============================================================================

@dataclass(frozen=True)
class ComponentDifference:
    """
    Difference in a single component between two self-states.
    """
    component_name: str
    change_type: ComponentChangeType
    from_state: str  # Abstract description of previous state
    to_state: str    # Abstract description of current state
    description: str

    @classmethod
    def unchanged(cls, name: str, state: str) -> ComponentDifference:
        """Create an unchanged component difference"""
        return cls(
            component_name=name,
            change_type=ComponentChangeType.UNCHANGED,
            from_state=state,
            to_state=state,
            description=f"{name} remains {state}",
        )

    @classmethod
    def undefined(cls, name: str) -> ComponentDifference:
        """Create an undefined component difference"""
        return cls(
            component_name=name,
            change_type=ComponentChangeType.UNDEFINED,
            from_state="undefined",
            to_state="undefined",
            description=f"{name} state is undefined",
        )


# =============================================================================
# Self-Difference Summary Structure
# =============================================================================

@dataclass(frozen=True)
class SelfDifferenceSummary:
    """
    Summary of differences between current and past self-state.

    This structure:
    - Does NOT contain raw numbers or scores
    - Does NOT evaluate changes as good/bad
    - Is for awareness/introspection only
    - Has NO influence on decisions
    """
    # Overall difference characteristics
    has_difference: bool
    magnitude: DifferenceMagnitude
    nature: ChangeNature
    temporal_span: TemporalSpan

    # Component-level differences
    emotional_diff: ComponentDifference
    responsibility_diff: ComponentDifference
    tendency_diff: ComponentDifference
    direction_diff: ComponentDifference
    value_diff: ComponentDifference

    # Metadata
    current_snapshot_id: str
    reference_snapshot_id: str
    comparison_timestamp: float

    # Human-readable description
    integrated_description: str

    def get_changed_components(self) -> list[str]:
        """List components that have changed"""
        changed = []
        if self.emotional_diff.change_type != ComponentChangeType.UNCHANGED:
            changed.append("emotional")
        if self.responsibility_diff.change_type != ComponentChangeType.UNCHANGED:
            changed.append("responsibility")
        if self.tendency_diff.change_type != ComponentChangeType.UNCHANGED:
            changed.append("tendency")
        if self.direction_diff.change_type != ComponentChangeType.UNCHANGED:
            changed.append("direction")
        if self.value_diff.change_type != ComponentChangeType.UNCHANGED:
            changed.append("value")
        return changed

    def get_unchanged_components(self) -> list[str]:
        """List components that have NOT changed"""
        unchanged = []
        if self.emotional_diff.change_type == ComponentChangeType.UNCHANGED:
            unchanged.append("emotional")
        if self.responsibility_diff.change_type == ComponentChangeType.UNCHANGED:
            unchanged.append("responsibility")
        if self.tendency_diff.change_type == ComponentChangeType.UNCHANGED:
            unchanged.append("tendency")
        if self.direction_diff.change_type == ComponentChangeType.UNCHANGED:
            unchanged.append("direction")
        if self.value_diff.change_type == ComponentChangeType.UNCHANGED:
            unchanged.append("value")
        return unchanged

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "has_difference": self.has_difference,
            "magnitude": self.magnitude.value,
            "nature": self.nature.value,
            "temporal_span": self.temporal_span.value,
            "emotional_diff": {
                "component_name": self.emotional_diff.component_name,
                "change_type": self.emotional_diff.change_type.value,
                "from_state": self.emotional_diff.from_state,
                "to_state": self.emotional_diff.to_state,
                "description": self.emotional_diff.description,
            },
            "responsibility_diff": {
                "component_name": self.responsibility_diff.component_name,
                "change_type": self.responsibility_diff.change_type.value,
                "from_state": self.responsibility_diff.from_state,
                "to_state": self.responsibility_diff.to_state,
                "description": self.responsibility_diff.description,
            },
            "tendency_diff": {
                "component_name": self.tendency_diff.component_name,
                "change_type": self.tendency_diff.change_type.value,
                "from_state": self.tendency_diff.from_state,
                "to_state": self.tendency_diff.to_state,
                "description": self.tendency_diff.description,
            },
            "direction_diff": {
                "component_name": self.direction_diff.component_name,
                "change_type": self.direction_diff.change_type.value,
                "from_state": self.direction_diff.from_state,
                "to_state": self.direction_diff.to_state,
                "description": self.direction_diff.description,
            },
            "value_diff": {
                "component_name": self.value_diff.component_name,
                "change_type": self.value_diff.change_type.value,
                "from_state": self.value_diff.from_state,
                "to_state": self.value_diff.to_state,
                "description": self.value_diff.description,
            },
            "current_snapshot_id": self.current_snapshot_id,
            "reference_snapshot_id": self.reference_snapshot_id,
            "comparison_timestamp": self.comparison_timestamp,
            "integrated_description": self.integrated_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelfDifferenceSummary:
        """Create from dictionary"""
        return cls(
            has_difference=data["has_difference"],
            magnitude=DifferenceMagnitude(data["magnitude"]),
            nature=ChangeNature(data["nature"]),
            temporal_span=TemporalSpan(data["temporal_span"]),
            emotional_diff=ComponentDifference(
                component_name=data["emotional_diff"]["component_name"],
                change_type=ComponentChangeType(data["emotional_diff"]["change_type"]),
                from_state=data["emotional_diff"]["from_state"],
                to_state=data["emotional_diff"]["to_state"],
                description=data["emotional_diff"]["description"],
            ),
            responsibility_diff=ComponentDifference(
                component_name=data["responsibility_diff"]["component_name"],
                change_type=ComponentChangeType(data["responsibility_diff"]["change_type"]),
                from_state=data["responsibility_diff"]["from_state"],
                to_state=data["responsibility_diff"]["to_state"],
                description=data["responsibility_diff"]["description"],
            ),
            tendency_diff=ComponentDifference(
                component_name=data["tendency_diff"]["component_name"],
                change_type=ComponentChangeType(data["tendency_diff"]["change_type"]),
                from_state=data["tendency_diff"]["from_state"],
                to_state=data["tendency_diff"]["to_state"],
                description=data["tendency_diff"]["description"],
            ),
            direction_diff=ComponentDifference(
                component_name=data["direction_diff"]["component_name"],
                change_type=ComponentChangeType(data["direction_diff"]["change_type"]),
                from_state=data["direction_diff"]["from_state"],
                to_state=data["direction_diff"]["to_state"],
                description=data["direction_diff"]["description"],
            ),
            value_diff=ComponentDifference(
                component_name=data["value_diff"]["component_name"],
                change_type=ComponentChangeType(data["value_diff"]["change_type"]),
                from_state=data["value_diff"]["from_state"],
                to_state=data["value_diff"]["to_state"],
                description=data["value_diff"]["description"],
            ),
            current_snapshot_id=data["current_snapshot_id"],
            reference_snapshot_id=data["reference_snapshot_id"],
            comparison_timestamp=data["comparison_timestamp"],
            integrated_description=data["integrated_description"],
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TemporalDifferenceConfig:
    """
    Configuration for temporal difference detection.

    These thresholds determine when changes are considered "different enough"
    to generate a difference summary. They do NOT affect decisions.
    """
    # History settings
    max_history_size: int = 50  # Maximum snapshots to keep
    immediate_window: int = 3   # Snapshots for immediate comparison
    short_term_window: int = 10  # Snapshots for short-term comparison
    medium_term_window: int = 25  # Snapshots for medium-term comparison

    # Minimum changes required to register as "different"
    min_component_changes_for_noticeable: int = 1
    min_component_changes_for_significant: int = 2
    min_component_changes_for_substantial: int = 3

    # Fluctuation detection
    fluctuation_lookback: int = 5  # How many snapshots to check for oscillation


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_emotional_state(
    current: SelfStateView,
    reference: SelfStateView,
) -> ComponentDifference:
    """Compare emotional component between two states"""
    curr = current.emotional
    ref = reference.emotional

    # Handle undefined cases
    if curr.spread == EmotionalSpread.UNDEFINED or ref.spread == EmotionalSpread.UNDEFINED:
        return ComponentDifference.undefined("emotional")

    # Check for changes
    spread_changed = curr.spread != ref.spread
    intensity_changed = curr.intensity != ref.intensity
    harmony_changed = curr.harmony != ref.harmony

    if not spread_changed and not intensity_changed and not harmony_changed:
        return ComponentDifference.unchanged(
            "emotional",
            f"{curr.intensity.value} intensity, {curr.spread.value}",
        )

    # Determine change type
    from_state = f"{ref.intensity.value}, {ref.spread.value}, {ref.harmony.value}"
    to_state = f"{curr.intensity.value}, {curr.spread.value}, {curr.harmony.value}"

    # Check if intensified or softened based on intensity
    intensity_order = [
        EmotionalIntensity.CALM,
        EmotionalIntensity.MODERATE,
        EmotionalIntensity.INTENSE,
        EmotionalIntensity.OVERWHELMING,
    ]

    try:
        curr_idx = intensity_order.index(curr.intensity)
        ref_idx = intensity_order.index(ref.intensity)

        if curr_idx > ref_idx:
            change_type = ComponentChangeType.INTENSIFIED
        elif curr_idx < ref_idx:
            change_type = ComponentChangeType.SOFTENED
        else:
            change_type = ComponentChangeType.SHIFTED
    except ValueError:
        change_type = ComponentChangeType.SHIFTED

    description = _generate_emotional_change_description(
        ref, curr, change_type, spread_changed, harmony_changed
    )

    return ComponentDifference(
        component_name="emotional",
        change_type=change_type,
        from_state=from_state,
        to_state=to_state,
        description=description,
    )


def _generate_emotional_change_description(
    ref_state,
    curr_state,
    change_type: ComponentChangeType,
    spread_changed: bool,
    harmony_changed: bool,
) -> str:
    """Generate description for emotional change"""
    parts = []

    if change_type == ComponentChangeType.INTENSIFIED:
        parts.append(f"Emotional intensity increased from {ref_state.intensity.value} to {curr_state.intensity.value}")
    elif change_type == ComponentChangeType.SOFTENED:
        parts.append(f"Emotional intensity decreased from {ref_state.intensity.value} to {curr_state.intensity.value}")

    if spread_changed:
        parts.append(f"spread changed from {ref_state.spread.value} to {curr_state.spread.value}")

    if harmony_changed:
        parts.append(f"harmony shifted from {ref_state.harmony.value} to {curr_state.harmony.value}")

    return "; ".join(parts) if parts else "Emotional state shifted"


def compare_responsibility_state(
    current: SelfStateView,
    reference: SelfStateView,
) -> ComponentDifference:
    """Compare responsibility component between two states"""
    curr = current.responsibility
    ref = reference.responsibility

    # Handle undefined cases
    if curr.burden_level == BurdenLevel.UNDEFINED or ref.burden_level == BurdenLevel.UNDEFINED:
        return ComponentDifference.undefined("responsibility")

    # Check for changes
    burden_changed = curr.burden_level != ref.burden_level
    distribution_changed = curr.distribution != ref.distribution
    trend_changed = curr.trend != ref.trend

    if not burden_changed and not distribution_changed and not trend_changed:
        return ComponentDifference.unchanged(
            "responsibility",
            f"{curr.burden_level.value} burden",
        )

    from_state = f"{ref.burden_level.value}, {ref.distribution.value}"
    to_state = f"{curr.burden_level.value}, {curr.distribution.value}"

    # Determine change type based on burden level
    burden_order = [
        BurdenLevel.UNBURDENED,
        BurdenLevel.LIGHT,
        BurdenLevel.MODERATE,
        BurdenLevel.BURDENED,
        BurdenLevel.HEAVY,
    ]

    try:
        curr_idx = burden_order.index(curr.burden_level)
        ref_idx = burden_order.index(ref.burden_level)

        if curr_idx > ref_idx:
            change_type = ComponentChangeType.INTENSIFIED
        elif curr_idx < ref_idx:
            change_type = ComponentChangeType.SOFTENED
        else:
            change_type = ComponentChangeType.SHIFTED
    except ValueError:
        change_type = ComponentChangeType.SHIFTED

    description = _generate_responsibility_change_description(
        ref, curr, change_type, distribution_changed, trend_changed
    )

    return ComponentDifference(
        component_name="responsibility",
        change_type=change_type,
        from_state=from_state,
        to_state=to_state,
        description=description,
    )


def _generate_responsibility_change_description(
    ref_state,
    curr_state,
    change_type: ComponentChangeType,
    distribution_changed: bool,
    trend_changed: bool,
) -> str:
    """Generate description for responsibility change"""
    parts = []

    if change_type == ComponentChangeType.INTENSIFIED:
        parts.append(f"Burden increased from {ref_state.burden_level.value} to {curr_state.burden_level.value}")
    elif change_type == ComponentChangeType.SOFTENED:
        parts.append(f"Burden decreased from {ref_state.burden_level.value} to {curr_state.burden_level.value}")

    if distribution_changed:
        parts.append(f"distribution shifted from {ref_state.distribution.value} to {curr_state.distribution.value}")

    if trend_changed:
        parts.append(f"trend changed from {ref_state.trend.value} to {curr_state.trend.value}")

    return "; ".join(parts) if parts else "Responsibility state shifted"


def compare_tendency_state(
    current: SelfStateView,
    reference: SelfStateView,
) -> ComponentDifference:
    """Compare tendency component between two states"""
    curr = current.tendency
    ref = reference.tendency

    # Handle undefined cases
    if curr.presence == HabitPresence.UNDEFINED or ref.presence == HabitPresence.UNDEFINED:
        return ComponentDifference.undefined("tendency")

    # Check for changes
    presence_changed = curr.presence != ref.presence
    character_changed = curr.character != ref.character

    if not presence_changed and not character_changed:
        return ComponentDifference.unchanged(
            "tendency",
            f"{curr.presence.value} presence, {curr.character.value}",
        )

    from_state = f"{ref.presence.value}, {ref.character.value}"
    to_state = f"{curr.presence.value}, {curr.character.value}"

    # Determine change type based on presence
    presence_order = [
        HabitPresence.NONE,
        HabitPresence.EMERGING,
        HabitPresence.FORMING,
        HabitPresence.ESTABLISHED,
    ]

    try:
        curr_idx = presence_order.index(curr.presence)
        ref_idx = presence_order.index(ref.presence)

        if curr_idx > ref_idx:
            change_type = ComponentChangeType.INTENSIFIED
        elif curr_idx < ref_idx:
            change_type = ComponentChangeType.SOFTENED
        else:
            change_type = ComponentChangeType.SHIFTED
    except ValueError:
        change_type = ComponentChangeType.SHIFTED

    description = _generate_tendency_change_description(
        ref, curr, change_type, character_changed
    )

    return ComponentDifference(
        component_name="tendency",
        change_type=change_type,
        from_state=from_state,
        to_state=to_state,
        description=description,
    )


def _generate_tendency_change_description(
    ref_state,
    curr_state,
    change_type: ComponentChangeType,
    character_changed: bool,
) -> str:
    """Generate description for tendency change"""
    parts = []

    if change_type == ComponentChangeType.INTENSIFIED:
        parts.append(f"Habits strengthened from {ref_state.presence.value} to {curr_state.presence.value}")
    elif change_type == ComponentChangeType.SOFTENED:
        parts.append(f"Habits weakened from {ref_state.presence.value} to {curr_state.presence.value}")

    if character_changed:
        parts.append(f"character shifted from {ref_state.character.value} to {curr_state.character.value}")

    return "; ".join(parts) if parts else "Tendency state shifted"


def compare_direction_state(
    current: SelfStateView,
    reference: SelfStateView,
) -> ComponentDifference:
    """Compare direction component between two states"""
    curr = current.direction
    ref = reference.direction

    # Handle undefined cases
    if curr.clarity == DirectionClarity.UNDEFINED or ref.clarity == DirectionClarity.UNDEFINED:
        return ComponentDifference.undefined("direction")

    # Check for changes
    clarity_changed = curr.clarity != ref.clarity
    convergence_changed = curr.convergence != ref.convergence

    if not clarity_changed and not convergence_changed:
        return ComponentDifference.unchanged(
            "direction",
            f"{curr.clarity.value} clarity, {curr.convergence.value}",
        )

    from_state = f"{ref.clarity.value}, {ref.convergence.value}"
    to_state = f"{curr.clarity.value}, {curr.convergence.value}"

    # Determine change type based on clarity
    clarity_order = [
        DirectionClarity.UNCLEAR,
        DirectionClarity.UNCERTAIN,
        DirectionClarity.CLEAR,
    ]

    try:
        curr_idx = clarity_order.index(curr.clarity)
        ref_idx = clarity_order.index(ref.clarity)

        if curr_idx > ref_idx:
            change_type = ComponentChangeType.INTENSIFIED
        elif curr_idx < ref_idx:
            change_type = ComponentChangeType.SOFTENED
        else:
            change_type = ComponentChangeType.SHIFTED
    except ValueError:
        change_type = ComponentChangeType.SHIFTED

    description = _generate_direction_change_description(
        ref, curr, change_type, convergence_changed
    )

    return ComponentDifference(
        component_name="direction",
        change_type=change_type,
        from_state=from_state,
        to_state=to_state,
        description=description,
    )


def _generate_direction_change_description(
    ref_state,
    curr_state,
    change_type: ComponentChangeType,
    convergence_changed: bool,
) -> str:
    """Generate description for direction change"""
    parts = []

    if change_type == ComponentChangeType.INTENSIFIED:
        parts.append(f"Direction clarified from {ref_state.clarity.value} to {curr_state.clarity.value}")
    elif change_type == ComponentChangeType.SOFTENED:
        parts.append(f"Direction became less clear from {ref_state.clarity.value} to {curr_state.clarity.value}")

    if convergence_changed:
        parts.append(f"convergence shifted from {ref_state.convergence.value} to {curr_state.convergence.value}")

    return "; ".join(parts) if parts else "Direction state shifted"


def compare_value_state(
    current: SelfStateView,
    reference: SelfStateView,
) -> ComponentDifference:
    """Compare value component between two states"""
    curr = current.value
    ref = reference.value

    # Handle undefined cases
    if curr.stability == ValueStability.UNDEFINED or ref.stability == ValueStability.UNDEFINED:
        return ComponentDifference.undefined("value")

    # Check for changes
    stability_changed = curr.stability != ref.stability
    clarity_changed = curr.clarity != ref.clarity

    if not stability_changed and not clarity_changed:
        return ComponentDifference.unchanged(
            "value",
            f"{curr.stability.value} stability, {curr.clarity.value}",
        )

    from_state = f"{ref.stability.value}, {ref.clarity.value}"
    to_state = f"{curr.stability.value}, {curr.clarity.value}"

    # Determine change type based on stability
    stability_order = [
        ValueStability.UNSTABLE,
        ValueStability.SHIFTING,
        ValueStability.STABLE,
        ValueStability.ANCHORED,
    ]

    try:
        curr_idx = stability_order.index(curr.stability)
        ref_idx = stability_order.index(ref.stability)

        if curr_idx > ref_idx:
            change_type = ComponentChangeType.INTENSIFIED
        elif curr_idx < ref_idx:
            change_type = ComponentChangeType.SOFTENED
        else:
            change_type = ComponentChangeType.SHIFTED
    except ValueError:
        change_type = ComponentChangeType.SHIFTED

    description = _generate_value_change_description(
        ref, curr, change_type, clarity_changed
    )

    return ComponentDifference(
        component_name="value",
        change_type=change_type,
        from_state=from_state,
        to_state=to_state,
        description=description,
    )


def _generate_value_change_description(
    ref_state,
    curr_state,
    change_type: ComponentChangeType,
    clarity_changed: bool,
) -> str:
    """Generate description for value change"""
    parts = []

    if change_type == ComponentChangeType.INTENSIFIED:
        parts.append(f"Values stabilized from {ref_state.stability.value} to {curr_state.stability.value}")
    elif change_type == ComponentChangeType.SOFTENED:
        parts.append(f"Values became less stable from {ref_state.stability.value} to {curr_state.stability.value}")

    if clarity_changed:
        parts.append(f"clarity shifted from {ref_state.clarity.value} to {curr_state.clarity.value}")

    return "; ".join(parts) if parts else "Value state shifted"


# =============================================================================
# Magnitude and Nature Determination
# =============================================================================

def determine_magnitude(
    emotional_diff: ComponentDifference,
    responsibility_diff: ComponentDifference,
    tendency_diff: ComponentDifference,
    direction_diff: ComponentDifference,
    value_diff: ComponentDifference,
    config: TemporalDifferenceConfig,
) -> DifferenceMagnitude:
    """
    Determine overall magnitude of difference.

    This counts how many components changed and categorizes
    the magnitude WITHOUT assigning numerical scores.
    """
    changed_count = 0
    diffs = [emotional_diff, responsibility_diff, tendency_diff, direction_diff, value_diff]

    for diff in diffs:
        if diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed_count += 1

    if changed_count == 0:
        return DifferenceMagnitude.NONE
    elif changed_count < config.min_component_changes_for_noticeable:
        return DifferenceMagnitude.MINIMAL
    elif changed_count < config.min_component_changes_for_significant:
        return DifferenceMagnitude.NOTICEABLE
    elif changed_count < config.min_component_changes_for_substantial:
        return DifferenceMagnitude.SIGNIFICANT
    else:
        return DifferenceMagnitude.SUBSTANTIAL


def determine_nature(
    current: SelfStateView,
    reference: SelfStateView,
    history: list[SelfStateView],
    config: TemporalDifferenceConfig,
) -> ChangeNature:
    """
    Determine the nature of change over time.

    This analyzes patterns in the history to determine if change is:
    - STABLE: Little change
    - FLUCTUATING: Oscillating back and forth
    - SHIFTING: Moving in a consistent direction
    - TRANSFORMED: Substantial change has occurred
    - RETURNING: Moving back toward an earlier state
    """
    if not history or len(history) < 2:
        return ChangeNature.UNDEFINED

    # Count component changes across recent history
    recent_changes = []
    for i in range(min(config.fluctuation_lookback, len(history) - 1)):
        if i + 1 < len(history):
            prev = history[-(i + 2)]
            curr = history[-(i + 1)]
            changes = _count_component_changes(prev, curr)
            recent_changes.append(changes)

    if not recent_changes:
        return ChangeNature.STABLE

    avg_changes = sum(recent_changes) / len(recent_changes)

    # Check for stability
    if avg_changes < 0.5:
        return ChangeNature.STABLE

    # Check for fluctuation (high variance in changes)
    if len(recent_changes) >= 3:
        variance = sum((c - avg_changes) ** 2 for c in recent_changes) / len(recent_changes)
        if variance > 1.0 and avg_changes < 2:
            return ChangeNature.FLUCTUATING

    # Check if returning to earlier state
    if len(history) >= 5:
        earlier_state = history[-5]
        earlier_distance = _count_component_changes(earlier_state, current)
        current_distance = _count_component_changes(reference, current)

        if earlier_distance < current_distance:
            return ChangeNature.RETURNING

    # Check for substantial transformation
    if avg_changes >= 3:
        return ChangeNature.TRANSFORMED

    # Default to shifting
    return ChangeNature.SHIFTING


def _count_component_changes(state1: SelfStateView, state2: SelfStateView) -> int:
    """Count number of component changes between two states"""
    changes = 0

    if state1.emotional.spread != state2.emotional.spread:
        changes += 1
    if state1.emotional.intensity != state2.emotional.intensity:
        changes += 1
    if state1.responsibility.burden_level != state2.responsibility.burden_level:
        changes += 1
    if state1.tendency.presence != state2.tendency.presence:
        changes += 1
    if state1.direction.clarity != state2.direction.clarity:
        changes += 1
    if state1.value.stability != state2.value.stability:
        changes += 1

    return changes


def determine_temporal_span(
    current_timestamp: float,
    reference_timestamp: float,
    config: TemporalDifferenceConfig,
) -> TemporalSpan:
    """
    Determine the temporal span of comparison.
    Based on number of snapshots rather than actual time.
    """
    # This is simplified - in actual use, we track by snapshot count
    time_diff = current_timestamp - reference_timestamp

    if time_diff < 60:  # Less than 1 minute
        return TemporalSpan.IMMEDIATE
    elif time_diff < 300:  # Less than 5 minutes
        return TemporalSpan.SHORT_TERM
    elif time_diff < 1800:  # Less than 30 minutes
        return TemporalSpan.MEDIUM_TERM
    else:
        return TemporalSpan.LONG_TERM


# =============================================================================
# Integrated Description Generation
# =============================================================================

def generate_integrated_description(
    magnitude: DifferenceMagnitude,
    nature: ChangeNature,
    changed_components: list[str],
) -> str:
    """
    Generate an integrated description of the self-difference.

    This is for awareness only - NO evaluation or judgment.
    """
    if magnitude == DifferenceMagnitude.NONE:
        return "Self-state remains consistent."

    if magnitude == DifferenceMagnitude.MINIMAL:
        return "Self-state shows minor fluctuation."

    # Build description based on nature
    nature_desc = {
        ChangeNature.STABLE: "relatively stable",
        ChangeNature.FLUCTUATING: "fluctuating",
        ChangeNature.SHIFTING: "gradually shifting",
        ChangeNature.TRANSFORMED: "notably different",
        ChangeNature.RETURNING: "returning toward previous state",
    }

    nature_str = nature_desc.get(nature, "changing")

    # Build component list
    if len(changed_components) == 1:
        component_str = f"in {changed_components[0]}"
    elif len(changed_components) == 2:
        component_str = f"in {changed_components[0]} and {changed_components[1]}"
    else:
        component_str = f"across {len(changed_components)} aspects"

    return f"Self-state is {nature_str} {component_str}."


# =============================================================================
# Temporal Self-Difference System
# =============================================================================

@dataclass
class TemporalDifferenceState:
    """
    Internal state of the temporal difference system.
    """
    history: deque = field(default_factory=lambda: deque(maxlen=50))
    comparison_count: int = 0
    last_significant_change_id: Optional[str] = None


class TemporalSelfDifferenceSystem:
    """
    Temporal Self-Difference Awareness System (自己モデル差分認知)

    Compares current self-state with past snapshots and generates
    abstract awareness of change.

    CRITICAL CONSTRAINTS:
    - For AWARENESS only, not control
    - STRICTLY NO IMPACT on decision making
    - NO JUDGMENT (good/bad)
    - Differences naturally shrink when states converge
    """

    def __init__(self, config: Optional[TemporalDifferenceConfig] = None):
        """Initialize with optional configuration"""
        self._config = config or TemporalDifferenceConfig()
        self._state = TemporalDifferenceState(
            history=deque(maxlen=self._config.max_history_size)
        )

    def record_snapshot(self, view: SelfStateView) -> None:
        """
        Record a self-state snapshot in history.

        This does NOT generate a difference - it just stores
        the snapshot for future comparison.
        """
        self._state.history.append(view)

    def compare_with_reference(
        self,
        current: SelfStateView,
        reference_index: int = -1,
    ) -> Optional[SelfDifferenceSummary]:
        """
        Compare current state with a reference from history.

        Args:
            current: Current SelfStateView
            reference_index: Index into history (-1 = most recent, -5 = 5 snapshots ago)

        Returns:
            SelfDifferenceSummary or None if no reference available
        """
        if not self._state.history:
            return None

        # Get reference snapshot
        try:
            actual_index = reference_index
            if actual_index < 0:
                actual_index = len(self._state.history) + reference_index
            if actual_index < 0 or actual_index >= len(self._state.history):
                return None

            reference = self._state.history[actual_index]
        except (IndexError, KeyError):
            return None

        return self._compare(current, reference)

    def compare_immediate(self, current: SelfStateView) -> Optional[SelfDifferenceSummary]:
        """Compare with most recent snapshot"""
        return self.compare_with_reference(current, -1)

    def compare_short_term(self, current: SelfStateView) -> Optional[SelfDifferenceSummary]:
        """Compare with short-term reference"""
        window = min(self._config.short_term_window, len(self._state.history))
        if window < 1:
            return None
        return self.compare_with_reference(current, -window)

    def compare_medium_term(self, current: SelfStateView) -> Optional[SelfDifferenceSummary]:
        """Compare with medium-term reference"""
        window = min(self._config.medium_term_window, len(self._state.history))
        if window < 1:
            return None
        return self.compare_with_reference(current, -window)

    def _compare(
        self,
        current: SelfStateView,
        reference: SelfStateView,
    ) -> SelfDifferenceSummary:
        """
        Internal comparison logic.

        This generates the difference summary without any
        judgment or impact on decisions.
        """
        self._state.comparison_count += 1

        # Compare each component
        emotional_diff = compare_emotional_state(current, reference)
        responsibility_diff = compare_responsibility_state(current, reference)
        tendency_diff = compare_tendency_state(current, reference)
        direction_diff = compare_direction_state(current, reference)
        value_diff = compare_value_state(current, reference)

        # Determine overall magnitude
        magnitude = determine_magnitude(
            emotional_diff,
            responsibility_diff,
            tendency_diff,
            direction_diff,
            value_diff,
            self._config,
        )

        # Determine nature of change
        nature = determine_nature(
            current,
            reference,
            list(self._state.history),
            self._config,
        )

        # Determine temporal span
        temporal_span = determine_temporal_span(
            current.timestamp,
            reference.timestamp,
            self._config,
        )

        # Get changed components
        changed = []
        if emotional_diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed.append("emotional")
        if responsibility_diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed.append("responsibility")
        if tendency_diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed.append("tendency")
        if direction_diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed.append("direction")
        if value_diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            changed.append("value")

        # Generate description
        description = generate_integrated_description(magnitude, nature, changed)

        # Track significant changes
        if magnitude in (DifferenceMagnitude.SIGNIFICANT, DifferenceMagnitude.SUBSTANTIAL):
            self._state.last_significant_change_id = current.snapshot_id

        return SelfDifferenceSummary(
            has_difference=magnitude != DifferenceMagnitude.NONE,
            magnitude=magnitude,
            nature=nature,
            temporal_span=temporal_span,
            emotional_diff=emotional_diff,
            responsibility_diff=responsibility_diff,
            tendency_diff=tendency_diff,
            direction_diff=direction_diff,
            value_diff=value_diff,
            current_snapshot_id=current.snapshot_id,
            reference_snapshot_id=reference.snapshot_id,
            comparison_timestamp=time.time(),
            integrated_description=description,
        )

    def get_history_size(self) -> int:
        """Get current history size"""
        return len(self._state.history)

    def get_comparison_count(self) -> int:
        """Get total comparisons made"""
        return self._state.comparison_count

    def get_last_significant_change_id(self) -> Optional[str]:
        """Get ID of last snapshot with significant change"""
        return self._state.last_significant_change_id

    def clear_history(self) -> None:
        """Clear snapshot history"""
        self._state.history.clear()
        self._state.last_significant_change_id = None


# =============================================================================
# Integration with SelfReferenceSystem
# =============================================================================

def generate_difference_tags(
    summary: SelfDifferenceSummary,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from SelfDifferenceSummary for SelfReferenceSystem integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.

    Args:
        summary: SelfDifferenceSummary to convert
        scale: Weight scaling factor (default 1.0)

    Returns:
        List of tag dictionaries
    """
    tags = []

    # Only generate tags if there is a difference
    if not summary.has_difference:
        tags.append({
            "category": "SELF_DIFFERENCE",
            "label": "self_consistent",
            "description": "Self-state remains consistent over time",
            "weight": 0.05 * scale,
        })
        return tags

    # Magnitude tag
    tags.append({
        "category": "SELF_DIFFERENCE_MAGNITUDE",
        "label": f"magnitude_{summary.magnitude.value}",
        "description": f"Self-change magnitude: {summary.magnitude.value}",
        "weight": 0.1 * scale,
    })

    # Nature tag
    tags.append({
        "category": "SELF_DIFFERENCE_NATURE",
        "label": f"nature_{summary.nature.value}",
        "description": f"Change nature: {summary.nature.value}",
        "weight": 0.1 * scale,
    })

    # Component-specific tags (only for changed components)
    for diff in [summary.emotional_diff, summary.responsibility_diff,
                 summary.tendency_diff, summary.direction_diff, summary.value_diff]:
        if diff.change_type not in (ComponentChangeType.UNCHANGED, ComponentChangeType.UNDEFINED):
            tags.append({
                "category": f"SELF_DIFFERENCE_{diff.component_name.upper()}",
                "label": f"{diff.component_name}_{diff.change_type.value}",
                "description": diff.description,
                "weight": 0.05 * scale,
            })

    # Integrated tag
    tags.append({
        "category": "SELF_DIFFERENCE_INTEGRATED",
        "label": "self_change_awareness",
        "description": summary.integrated_description,
        "weight": 0.1 * scale,
    })

    return tags


def get_difference_summary(summary: SelfDifferenceSummary) -> str:
    """
    Get a human-readable summary of the difference.

    For introspection/logging only.
    """
    lines = [
        "=== Self-Difference Awareness ===",
        f"Has Difference: {summary.has_difference}",
        f"Magnitude: {summary.magnitude.value}",
        f"Nature: {summary.nature.value}",
        f"Temporal Span: {summary.temporal_span.value}",
        "",
        "Component Changes:",
        f"  Emotional: {summary.emotional_diff.description}",
        f"  Responsibility: {summary.responsibility_diff.description}",
        f"  Tendency: {summary.tendency_diff.description}",
        f"  Direction: {summary.direction_diff.description}",
        f"  Value: {summary.value_diff.description}",
        "",
        f"Integrated: {summary.integrated_description}",
    ]

    return "\n".join(lines)


def get_difference_for_introspection(summary: SelfDifferenceSummary) -> dict[str, Any]:
    """
    Get structured difference data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    return {
        "has_difference": summary.has_difference,
        "magnitude": summary.magnitude.value,
        "nature": summary.nature.value,
        "temporal_span": summary.temporal_span.value,
        "changed_components": summary.get_changed_components(),
        "unchanged_components": summary.get_unchanged_components(),
        "integrated_description": summary.integrated_description,
        "comparison_timestamp": summary.comparison_timestamp,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config(
    max_history_size: int = 50,
    immediate_window: int = 3,
    short_term_window: int = 10,
    medium_term_window: int = 25,
) -> TemporalDifferenceConfig:
    """Create a custom configuration"""
    return TemporalDifferenceConfig(
        max_history_size=max_history_size,
        immediate_window=immediate_window,
        short_term_window=short_term_window,
        medium_term_window=medium_term_window,
    )


def create_empty_summary() -> SelfDifferenceSummary:
    """Create an empty/no-difference summary"""
    undefined_diff = ComponentDifference.undefined("undefined")
    return SelfDifferenceSummary(
        has_difference=False,
        magnitude=DifferenceMagnitude.NONE,
        nature=ChangeNature.STABLE,
        temporal_span=TemporalSpan.UNDEFINED,
        emotional_diff=undefined_diff,
        responsibility_diff=undefined_diff,
        tendency_diff=undefined_diff,
        direction_diff=undefined_diff,
        value_diff=undefined_diff,
        current_snapshot_id="",
        reference_snapshot_id="",
        comparison_timestamp=time.time(),
        integrated_description="No difference data available",
    )


# =============================================================================
# Persistence Support (Optional)
# =============================================================================

def save_difference_history(
    system: TemporalSelfDifferenceSystem,
    path: str,
) -> None:
    """
    Save difference system history to file.

    Note: Only saves serializable snapshot data.
    """
    import json

    history_data = [view.to_dict() for view in system._state.history]

    data = {
        "history": history_data,
        "comparison_count": system._state.comparison_count,
        "last_significant_change_id": system._state.last_significant_change_id,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_difference_history(
    path: str,
    config: Optional[TemporalDifferenceConfig] = None,
) -> TemporalSelfDifferenceSystem:
    """
    Load difference system history from file.
    """
    import json
    from .self_model import SelfStateView

    system = TemporalSelfDifferenceSystem(config)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for view_data in data.get("history", []):
            view = SelfStateView.from_dict(view_data)
            system._state.history.append(view)

        system._state.comparison_count = data.get("comparison_count", 0)
        system._state.last_significant_change_id = data.get("last_significant_change_id")

    except FileNotFoundError:
        pass

    return system
