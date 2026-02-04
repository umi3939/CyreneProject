"""
Self-Continuity Strain (自己連続性負荷)

This module provides awareness of strain/unease when self-difference persists over time.
It monitors the TemporalSelfDifferenceSystem and generates abstract descriptions of
the "strain" felt when the self continues to be different from its earlier state.

CRITICAL DESIGN PRINCIPLE:
- This is for AWARENESS only, not for control or correction
- STRICTLY NO IMPACT on decision making
- NOT self-preservation - just the "feeling of unease"
- Strain naturally decays when difference resolves
- NO explicit reset/recover/fix operations

Philosophy:
自己とは、定義されるものではなく、
連続しているときにだけ「自然さ」として感じられるものである。

変化が急すぎたり、説明できない形で続いたとき、
そこには評価ではない「違和感」が生まれる。

本機能は、修正や制御を行わずに
この違和感を内部状態として保持するための構造である。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque
import time

from .temporal_self_difference import (
    SelfDifferenceSummary,
    DifferenceMagnitude,
    ChangeNature,
    TemporalSelfDifferenceSystem,
)


# =============================================================================
# Abstract Enums for Strain Description
# These describe the FEELING without judgment or evaluation
# =============================================================================

class StrainPresence(Enum):
    """
    Whether strain is present or absent.
    Simple binary indicator - no nuance needed here.
    """
    ABSENT = "absent"      # No strain felt
    PRESENT = "present"    # Strain is felt


class StrainLevel(Enum):
    """
    Level of strain felt from discontinuity.
    NOT a score - just an abstract category of the "unease".

    These are NOT evaluative (no good/bad):
    - AT_EASE: Natural, settled feeling
    - UNSETTLED: Something feels a bit off
    - DISSONANT: Notable unease about self-continuity
    - ALIENATED: Strong sense of discontinuity
    """
    AT_EASE = "at_ease"         # No strain, self feels continuous
    UNSETTLED = "unsettled"     # Mild strain, slight unease
    DISSONANT = "dissonant"     # Moderate strain, notable unease
    ALIENATED = "alienated"     # High strain, strong discontinuity feeling
    UNDEFINED = "undefined"     # Cannot determine


class StrainPersistence(Enum):
    """
    How long the strain has been present.
    NOT a score - just temporal classification.
    """
    NONE = "none"              # No strain to track
    MOMENTARY = "momentary"    # Just appeared
    ONGOING = "ongoing"        # Has been present for a while
    CHRONIC = "chronic"        # Persisted for extended period
    UNDEFINED = "undefined"    # Cannot determine


class StrainTrend(Enum):
    """
    Direction the strain is moving.
    NOT evaluative - just observational.
    """
    STABLE = "stable"          # Strain level not changing
    BUILDING = "building"      # Strain is increasing
    EASING = "easing"          # Strain is decreasing
    FLUCTUATING = "fluctuating"  # Strain is oscillating
    UNDEFINED = "undefined"


# =============================================================================
# Strain State Structure
# =============================================================================

@dataclass(frozen=True)
class StrainState:
    """
    The current strain state from self-continuity awareness.

    This structure:
    - Does NOT contain raw numbers or scores (externally)
    - Does NOT evaluate strain as good/bad
    - Is for awareness/introspection only
    - Has NO influence on decisions

    This represents: 「このままでは落ち着かない」という感触
    """
    presence: StrainPresence
    level: StrainLevel
    persistence: StrainPersistence
    trend: StrainTrend

    # Metadata
    timestamp: float
    last_update_timestamp: float

    # Human-readable description (for introspection only)
    description: str

    @classmethod
    def at_ease(cls, timestamp: Optional[float] = None) -> StrainState:
        """Create an at-ease state (no strain)"""
        ts = timestamp or time.time()
        return cls(
            presence=StrainPresence.ABSENT,
            level=StrainLevel.AT_EASE,
            persistence=StrainPersistence.NONE,
            trend=StrainTrend.STABLE,
            timestamp=ts,
            last_update_timestamp=ts,
            description="Self feels continuous and natural.",
        )

    @classmethod
    def undefined(cls, timestamp: Optional[float] = None) -> StrainState:
        """Create an undefined state"""
        ts = timestamp or time.time()
        return cls(
            presence=StrainPresence.ABSENT,
            level=StrainLevel.UNDEFINED,
            persistence=StrainPersistence.UNDEFINED,
            trend=StrainTrend.UNDEFINED,
            timestamp=ts,
            last_update_timestamp=ts,
            description="Strain state cannot be determined.",
        )

    def is_strained(self) -> bool:
        """Check if there is any strain present"""
        return self.presence == StrainPresence.PRESENT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "presence": self.presence.value,
            "level": self.level.value,
            "persistence": self.persistence.value,
            "trend": self.trend.value,
            "timestamp": self.timestamp,
            "last_update_timestamp": self.last_update_timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrainState:
        """Create from dictionary"""
        return cls(
            presence=StrainPresence(data["presence"]),
            level=StrainLevel(data["level"]),
            persistence=StrainPersistence(data["persistence"]),
            trend=StrainTrend(data["trend"]),
            timestamp=data["timestamp"],
            last_update_timestamp=data["last_update_timestamp"],
            description=data["description"],
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ContinuityStrainConfig:
    """
    Configuration for continuity strain detection.

    These parameters determine when differences are considered
    "persistent enough" to generate strain. They do NOT affect decisions.
    """
    # Observation window settings
    observation_window: int = 5  # How many difference observations to consider

    # Persistence thresholds (in observation counts)
    min_observations_for_strain: int = 3  # Minimum consecutive significant diffs for strain
    min_observations_for_ongoing: int = 5  # Observations for "ongoing" persistence
    min_observations_for_chronic: int = 10  # Observations for "chronic" persistence

    # Magnitude thresholds for strain generation
    # Only these magnitude levels can cause strain
    strain_triggering_magnitudes: tuple[DifferenceMagnitude, ...] = (
        DifferenceMagnitude.NOTICEABLE,
        DifferenceMagnitude.SIGNIFICANT,
        DifferenceMagnitude.SUBSTANTIAL,
    )

    # Decay settings (in observation counts without difference)
    decay_observations_for_easing: int = 2  # Start easing after N non-significant observations
    decay_observations_for_resolution: int = 4  # Full resolution after N non-significant observations

    # Trend detection
    trend_lookback: int = 4  # Observations to consider for trend detection


# =============================================================================
# Internal State Tracking
# =============================================================================

@dataclass
class DifferenceObservation:
    """Single observation of self-difference"""
    magnitude: DifferenceMagnitude
    nature: ChangeNature
    has_difference: bool
    timestamp: float

    def is_significant(self, config: ContinuityStrainConfig) -> bool:
        """Check if this observation is significant enough to cause/maintain strain"""
        return (
            self.has_difference and
            self.magnitude in config.strain_triggering_magnitudes
        )


@dataclass
class ContinuityStrainInternalState:
    """
    Internal state for tracking strain evolution.

    This tracks the raw data needed to determine strain,
    but the raw data is NOT exposed externally.
    """
    # Observation history
    observations: deque = field(default_factory=lambda: deque(maxlen=20))

    # Current strain state
    current_strain: StrainState = field(default_factory=StrainState.at_ease)

    # Tracking variables (internal, not exposed)
    consecutive_significant_count: int = 0
    consecutive_insignificant_count: int = 0
    strain_started_at: Optional[float] = None
    total_strain_observations: int = 0

    # For trend detection
    strain_level_history: deque = field(default_factory=lambda: deque(maxlen=10))


# =============================================================================
# Strain Level Determination
# =============================================================================

def determine_strain_level(
    consecutive_significant: int,
    total_strain_observations: int,
    average_magnitude: DifferenceMagnitude,
    config: ContinuityStrainConfig,
) -> StrainLevel:
    """
    Determine the strain level from observation data.

    This maps raw tracking data to abstract strain levels.
    The raw data is NOT exposed.
    """
    if consecutive_significant < config.min_observations_for_strain:
        return StrainLevel.AT_EASE

    # Base level from consecutive observations
    if consecutive_significant >= config.min_observations_for_chronic:
        base_level = StrainLevel.ALIENATED
    elif consecutive_significant >= config.min_observations_for_ongoing:
        base_level = StrainLevel.DISSONANT
    else:
        base_level = StrainLevel.UNSETTLED

    # Adjust based on average magnitude
    if average_magnitude == DifferenceMagnitude.SUBSTANTIAL:
        # Substantial differences may escalate faster
        if base_level == StrainLevel.UNSETTLED:
            base_level = StrainLevel.DISSONANT
        elif base_level == StrainLevel.DISSONANT:
            base_level = StrainLevel.ALIENATED

    return base_level


def determine_strain_persistence(
    consecutive_significant: int,
    strain_started_at: Optional[float],
    current_time: float,
    config: ContinuityStrainConfig,
) -> StrainPersistence:
    """
    Determine how long strain has persisted.
    """
    if consecutive_significant < config.min_observations_for_strain:
        return StrainPersistence.NONE

    if consecutive_significant >= config.min_observations_for_chronic:
        return StrainPersistence.CHRONIC
    elif consecutive_significant >= config.min_observations_for_ongoing:
        return StrainPersistence.ONGOING
    else:
        return StrainPersistence.MOMENTARY


def determine_strain_trend(
    strain_level_history: list[StrainLevel],
    config: ContinuityStrainConfig,
) -> StrainTrend:
    """
    Determine the trend of strain over recent observations.
    """
    if len(strain_level_history) < 2:
        return StrainTrend.UNDEFINED

    # Get recent levels for comparison
    recent = list(strain_level_history)[-config.trend_lookback:]

    if len(recent) < 2:
        return StrainTrend.STABLE

    # Map levels to numeric values for trend detection
    level_values = {
        StrainLevel.AT_EASE: 0,
        StrainLevel.UNSETTLED: 1,
        StrainLevel.DISSONANT: 2,
        StrainLevel.ALIENATED: 3,
        StrainLevel.UNDEFINED: -1,
    }

    values = [level_values.get(lvl, -1) for lvl in recent if lvl != StrainLevel.UNDEFINED]

    if len(values) < 2:
        return StrainTrend.STABLE

    # Check for consistent direction
    increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    if all(v == values[0] for v in values):
        return StrainTrend.STABLE
    elif increasing and values[-1] > values[0]:
        return StrainTrend.BUILDING
    elif decreasing and values[-1] < values[0]:
        return StrainTrend.EASING
    else:
        return StrainTrend.FLUCTUATING


def get_average_magnitude(observations: list[DifferenceObservation]) -> DifferenceMagnitude:
    """
    Get the average magnitude from observations.
    Returns the most common significant magnitude.
    """
    significant_magnitudes = [
        obs.magnitude for obs in observations
        if obs.has_difference and obs.magnitude not in (
            DifferenceMagnitude.NONE,
            DifferenceMagnitude.MINIMAL,
            DifferenceMagnitude.UNDEFINED,
        )
    ]

    if not significant_magnitudes:
        return DifferenceMagnitude.NONE

    # Return the highest occurring magnitude
    from collections import Counter
    counts = Counter(significant_magnitudes)
    return counts.most_common(1)[0][0]


def generate_strain_description(
    level: StrainLevel,
    persistence: StrainPersistence,
    trend: StrainTrend,
) -> str:
    """
    Generate a human-readable description of the strain state.

    This is for introspection only - NOT evaluative.
    """
    if level == StrainLevel.AT_EASE:
        return "Self feels continuous and natural."

    # Level descriptions (non-evaluative)
    level_descriptions = {
        StrainLevel.UNSETTLED: "There is a slight sense of unease about self-continuity",
        StrainLevel.DISSONANT: "There is a notable feeling of discontinuity",
        StrainLevel.ALIENATED: "There is a strong sense of separation from earlier self",
    }

    description = level_descriptions.get(level, "Self-continuity state unclear")

    # Add persistence qualifier
    if persistence == StrainPersistence.CHRONIC:
        description += " that has persisted for some time"
    elif persistence == StrainPersistence.ONGOING:
        description += " that has been present"
    elif persistence == StrainPersistence.MOMENTARY:
        description += " that recently appeared"

    # Add trend qualifier
    if trend == StrainTrend.BUILDING:
        description += ", and this feeling is intensifying"
    elif trend == StrainTrend.EASING:
        description += ", though this feeling is easing"
    elif trend == StrainTrend.FLUCTUATING:
        description += ", with this feeling fluctuating"

    return description + "."


# =============================================================================
# Continuity Strain System
# =============================================================================

class ContinuityStrainSystem:
    """
    Self-Continuity Strain Awareness System (自己連続性負荷)

    Monitors self-difference over time and generates awareness
    of the "strain" or "unease" when differences persist.

    CRITICAL CONSTRAINTS:
    - For AWARENESS only, not control
    - STRICTLY NO IMPACT on decision making
    - NOT self-preservation - just "unease" awareness
    - Strain naturally decays when difference resolves
    - NO explicit reset/recover/fix operations
    """

    def __init__(self, config: Optional[ContinuityStrainConfig] = None):
        """Initialize with optional configuration"""
        self._config = config or ContinuityStrainConfig()
        self._state = ContinuityStrainInternalState(
            observations=deque(maxlen=self._config.observation_window * 4),
            strain_level_history=deque(maxlen=self._config.trend_lookback * 2),
        )

    def observe_difference(
        self,
        difference_summary: SelfDifferenceSummary,
    ) -> StrainState:
        """
        Observe a self-difference and update strain state.

        This is the main entry point for processing difference observations.

        Args:
            difference_summary: SelfDifferenceSummary from TemporalSelfDifferenceSystem

        Returns:
            Updated StrainState (for awareness only)
        """
        current_time = time.time()

        # Create observation record
        observation = DifferenceObservation(
            magnitude=difference_summary.magnitude,
            nature=difference_summary.nature,
            has_difference=difference_summary.has_difference,
            timestamp=current_time,
        )

        # Add to history
        self._state.observations.append(observation)

        # Update consecutive counts
        if observation.is_significant(self._config):
            self._state.consecutive_significant_count += 1
            self._state.consecutive_insignificant_count = 0

            # Track when strain started
            if self._state.strain_started_at is None:
                self._state.strain_started_at = current_time

            self._state.total_strain_observations += 1
        else:
            self._state.consecutive_insignificant_count += 1

            # Natural decay: reduce consecutive significant count
            if self._state.consecutive_insignificant_count >= self._config.decay_observations_for_resolution:
                # Full resolution
                self._state.consecutive_significant_count = 0
                self._state.strain_started_at = None
            elif self._state.consecutive_insignificant_count >= self._config.decay_observations_for_easing:
                # Start easing
                self._state.consecutive_significant_count = max(
                    0,
                    self._state.consecutive_significant_count - 1
                )

        # Determine new strain state
        new_strain = self._calculate_strain_state(current_time)

        # Track strain level history for trend detection
        self._state.strain_level_history.append(new_strain.level)

        # Update current strain
        self._state.current_strain = new_strain

        return new_strain

    def observe_from_system(
        self,
        temporal_system: TemporalSelfDifferenceSystem,
        current_view,  # SelfStateView - imported from self_model
    ) -> StrainState:
        """
        Convenience method to observe directly from TemporalSelfDifferenceSystem.

        This performs an immediate comparison and processes the result.

        Args:
            temporal_system: The TemporalSelfDifferenceSystem to query
            current_view: Current SelfStateView for comparison

        Returns:
            Updated StrainState
        """
        # Get immediate difference
        diff = temporal_system.compare_immediate(current_view)

        if diff is None:
            # No comparison possible, maintain current state
            return self._state.current_strain

        return self.observe_difference(diff)

    def _calculate_strain_state(self, current_time: float) -> StrainState:
        """
        Calculate the current strain state from internal tracking data.
        """
        # Check if strain should be present
        has_strain = (
            self._state.consecutive_significant_count >=
            self._config.min_observations_for_strain
        )

        if not has_strain:
            # Check if we're in decay phase
            if self._state.consecutive_insignificant_count > 0 and self._state.current_strain.is_strained():
                # Still decaying but not fully resolved
                previous = self._state.current_strain

                # Determine if still has residual strain
                if self._state.consecutive_insignificant_count < self._config.decay_observations_for_resolution:
                    # Still has some residual strain
                    return StrainState(
                        presence=StrainPresence.PRESENT,
                        level=StrainLevel.UNSETTLED,  # Decay to lowest level
                        persistence=previous.persistence,
                        trend=StrainTrend.EASING,
                        timestamp=previous.timestamp,
                        last_update_timestamp=current_time,
                        description="Self-continuity strain is easing.",
                    )

            # No strain
            return StrainState.at_ease(current_time)

        # Calculate strain components
        recent_observations = list(self._state.observations)[-self._config.observation_window:]
        average_magnitude = get_average_magnitude(recent_observations)

        level = determine_strain_level(
            self._state.consecutive_significant_count,
            self._state.total_strain_observations,
            average_magnitude,
            self._config,
        )

        persistence = determine_strain_persistence(
            self._state.consecutive_significant_count,
            self._state.strain_started_at,
            current_time,
            self._config,
        )

        trend = determine_strain_trend(
            list(self._state.strain_level_history),
            self._config,
        )

        description = generate_strain_description(level, persistence, trend)

        return StrainState(
            presence=StrainPresence.PRESENT,
            level=level,
            persistence=persistence,
            trend=trend,
            timestamp=self._state.strain_started_at or current_time,
            last_update_timestamp=current_time,
            description=description,
        )

    def get_current_strain(self) -> StrainState:
        """
        Get the current strain state.

        This is for introspection only - MUST NOT be used for decisions.
        """
        return self._state.current_strain

    def get_observation_count(self) -> int:
        """Get total number of observations processed"""
        return len(self._state.observations)

    def is_strained(self) -> bool:
        """Check if currently experiencing strain"""
        return self._state.current_strain.is_strained()

    def get_strain_duration_observations(self) -> int:
        """Get how many consecutive observations have shown significant difference"""
        return self._state.consecutive_significant_count


# =============================================================================
# Integration with SelfReferenceSystem
# =============================================================================

def generate_strain_tags(
    strain: StrainState,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from StrainState for SelfReferenceSystem integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.

    Args:
        strain: StrainState to convert
        scale: Weight scaling factor (default 1.0)

    Returns:
        List of tag dictionaries
    """
    tags = []

    if not strain.is_strained():
        tags.append({
            "category": "CONTINUITY_STRAIN",
            "label": "continuous_self",
            "description": "Self feels continuous and natural",
            "weight": 0.05 * scale,
        })
        return tags

    # Presence tag
    tags.append({
        "category": "CONTINUITY_STRAIN_PRESENCE",
        "label": "strain_present",
        "description": "Self-continuity strain is present",
        "weight": 0.1 * scale,
    })

    # Level tag
    tags.append({
        "category": "CONTINUITY_STRAIN_LEVEL",
        "label": f"strain_{strain.level.value}",
        "description": f"Strain level: {strain.level.value}",
        "weight": 0.1 * scale,
    })

    # Persistence tag
    if strain.persistence != StrainPersistence.NONE:
        tags.append({
            "category": "CONTINUITY_STRAIN_PERSISTENCE",
            "label": f"persistence_{strain.persistence.value}",
            "description": f"Strain persistence: {strain.persistence.value}",
            "weight": 0.05 * scale,
        })

    # Trend tag
    if strain.trend not in (StrainTrend.STABLE, StrainTrend.UNDEFINED):
        tags.append({
            "category": "CONTINUITY_STRAIN_TREND",
            "label": f"trend_{strain.trend.value}",
            "description": f"Strain trend: {strain.trend.value}",
            "weight": 0.05 * scale,
        })

    # Integrated description tag
    tags.append({
        "category": "CONTINUITY_STRAIN_INTEGRATED",
        "label": "strain_awareness",
        "description": strain.description,
        "weight": 0.1 * scale,
    })

    return tags


def get_strain_summary(strain: StrainState) -> str:
    """
    Get a human-readable summary of the strain state.

    For introspection/logging only.
    """
    lines = [
        "=== Self-Continuity Strain Awareness ===",
        f"Presence: {strain.presence.value}",
        f"Level: {strain.level.value}",
        f"Persistence: {strain.persistence.value}",
        f"Trend: {strain.trend.value}",
        "",
        f"Description: {strain.description}",
    ]

    return "\n".join(lines)


def get_strain_for_introspection(strain: StrainState) -> dict[str, Any]:
    """
    Get structured strain data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    return {
        "is_strained": strain.is_strained(),
        "presence": strain.presence.value,
        "level": strain.level.value,
        "persistence": strain.persistence.value,
        "trend": strain.trend.value,
        "description": strain.description,
        "last_update_timestamp": strain.last_update_timestamp,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config(
    observation_window: int = 5,
    min_observations_for_strain: int = 3,
    min_observations_for_ongoing: int = 5,
    min_observations_for_chronic: int = 10,
    decay_observations_for_easing: int = 2,
    decay_observations_for_resolution: int = 4,
) -> ContinuityStrainConfig:
    """Create a custom configuration"""
    return ContinuityStrainConfig(
        observation_window=observation_window,
        min_observations_for_strain=min_observations_for_strain,
        min_observations_for_ongoing=min_observations_for_ongoing,
        min_observations_for_chronic=min_observations_for_chronic,
        decay_observations_for_easing=decay_observations_for_easing,
        decay_observations_for_resolution=decay_observations_for_resolution,
    )


def create_empty_strain() -> StrainState:
    """Create an at-ease state (no strain)"""
    return StrainState.at_ease()


# =============================================================================
# Persistence Support (Optional)
# =============================================================================

def save_strain_state(
    system: ContinuityStrainSystem,
    path: str,
) -> None:
    """
    Save strain system state to file.

    Saves the current strain state and observation history.
    """
    import json

    observations_data = [
        {
            "magnitude": obs.magnitude.value,
            "nature": obs.nature.value,
            "has_difference": obs.has_difference,
            "timestamp": obs.timestamp,
        }
        for obs in system._state.observations
    ]

    strain_history_data = [level.value for level in system._state.strain_level_history]

    data = {
        "observations": observations_data,
        "current_strain": system._state.current_strain.to_dict(),
        "consecutive_significant_count": system._state.consecutive_significant_count,
        "consecutive_insignificant_count": system._state.consecutive_insignificant_count,
        "strain_started_at": system._state.strain_started_at,
        "total_strain_observations": system._state.total_strain_observations,
        "strain_level_history": strain_history_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_strain_state(
    path: str,
    config: Optional[ContinuityStrainConfig] = None,
) -> ContinuityStrainSystem:
    """
    Load strain system state from file.
    """
    import json

    system = ContinuityStrainSystem(config)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Restore observations
        for obs_data in data.get("observations", []):
            obs = DifferenceObservation(
                magnitude=DifferenceMagnitude(obs_data["magnitude"]),
                nature=ChangeNature(obs_data["nature"]),
                has_difference=obs_data["has_difference"],
                timestamp=obs_data["timestamp"],
            )
            system._state.observations.append(obs)

        # Restore current strain
        if "current_strain" in data:
            system._state.current_strain = StrainState.from_dict(data["current_strain"])

        # Restore tracking variables
        system._state.consecutive_significant_count = data.get("consecutive_significant_count", 0)
        system._state.consecutive_insignificant_count = data.get("consecutive_insignificant_count", 0)
        system._state.strain_started_at = data.get("strain_started_at")
        system._state.total_strain_observations = data.get("total_strain_observations", 0)

        # Restore strain level history
        for level_value in data.get("strain_level_history", []):
            system._state.strain_level_history.append(StrainLevel(level_value))

    except FileNotFoundError:
        pass

    return system


# =============================================================================
# Impact Verification (Test Support)
# =============================================================================

def verify_no_decision_impact(strain: StrainState) -> bool:
    """
    Verify that the strain state has no decision-impacting values.

    This is for testing to ensure the module maintains its design constraint
    of having STRICTLY NO IMPACT on decision making.

    Returns True if the strain state is safe (no decision impact).
    """
    # The StrainState contains only:
    # - Enums (abstract categories, not numeric scores)
    # - Timestamps (for tracking, not influence)
    # - Description (for introspection only)

    # Check that no numeric scores are exposed
    public_attrs = [attr for attr in dir(strain) if not attr.startswith('_')]

    for attr in public_attrs:
        if callable(getattr(strain, attr)):
            continue

        value = getattr(strain, attr)

        # Timestamps are allowed
        if attr in ('timestamp', 'last_update_timestamp'):
            continue

        # Strings are allowed (description)
        if isinstance(value, str):
            continue

        # Enums are allowed
        if isinstance(value, Enum):
            continue

        # Anything else that looks like a decision-influencing number is not allowed
        if isinstance(value, (int, float)) and attr not in ('timestamp', 'last_update_timestamp'):
            return False

    return True


def verify_no_correction_mechanism(system: ContinuityStrainSystem) -> bool:
    """
    Verify that the system has no correction mechanisms.

    This checks that there are no methods that would:
    - Force the system back to a "good" state
    - Reset strain to correct it
    - Modify other systems to fix discontinuity

    Returns True if the system is safe (observation only).
    """
    # List of method names that would indicate correction mechanisms
    forbidden_patterns = [
        "fix",
        "correct",
        "repair",
        "reset_to_normal",
        "force_ease",
        "eliminate_strain",
        "restore_continuity",
        "stabilize",
    ]

    methods = [m for m in dir(system) if not m.startswith('_') and callable(getattr(system, m))]

    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden_patterns:
            if pattern in method_lower:
                return False

    return True
