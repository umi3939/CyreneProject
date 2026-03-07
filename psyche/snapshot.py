"""
psyche/snapshot.py - Unified Snapshot Structure for Persistence

Combines all psychological state components into a single persistable unit:
- PsycheState (emotions, drives, mood, fear pillars)
- ShortTermMemory (recent stimuli, context)
- ResponsibilityState (decision burden)
- DynamicsState (peak/rebound emotional phases)

Design principles:
- All-or-nothing persistence (no partial saves/restores)
- No automatic normalization or correction on load
- Structure validation without value interpretation
- Version tracking for future compatibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .state import PsycheState
from .short_term_memory import ShortTermMemory
from .short_term_loop import LoopState, LoopConfig
from .responsibility import ResponsibilityState, create_default_state as create_default_responsibility
from .dynamics import DynamicsState, create_dynamics_state

# Current snapshot version for compatibility checks
# Version 2: Added DynamicsState for peak/rebound emotional phases
SNAPSHOT_VERSION = 2


@dataclass
class Snapshot:
    """
    Complete psychological state snapshot for persistence.

    This structure contains everything needed to restore the AI's
    psychological continuity across restarts:
    - Emotional state and mood
    - Fear/loss pillar states
    - Short-term memory and context
    - Responsibility burden from past decisions
    - Emotional dynamics (peak/rebound phases)

    The snapshot is treated as an atomic unit - either fully saved/restored
    or not at all.
    """

    # Core psychological state
    psyche: PsycheState = field(default_factory=PsycheState)

    # Short-term memory loop state
    loop: LoopState = field(default_factory=LoopState)

    # Responsibility state (decision burden)
    responsibility: ResponsibilityState = field(default_factory=create_default_responsibility)

    # Emotional dynamics state (peak/rebound phases)
    dynamics: DynamicsState = field(default_factory=create_dynamics_state)

    # Metadata
    version: int = SNAPSHOT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # Optional user/session identifier
    user_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize snapshot to a plain dict for JSON persistence.

        All nested structures are converted to dicts recursively.
        """
        return {
            "version": self.version,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "psyche": self.psyche.to_dict(),
            "loop": self.loop.to_dict(),
            "responsibility": self.responsibility.model_dump(),
            "dynamics": self.dynamics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["Snapshot"]:
        """
        Reconstruct a Snapshot from a plain dict.

        Returns None if the data is invalid or incompatible.
        Validation is structural only - values are not corrected.
        """
        # Version check
        version = data.get("version")
        if version is None or not isinstance(version, int):
            return None

        if version > SNAPSHOT_VERSION:
            # Future version - cannot safely load
            return None

        # Required fields check (dynamics is optional for v1 compatibility)
        required = ["psyche", "loop", "responsibility"]
        for key in required:
            if key not in data or not isinstance(data[key], dict):
                return None

        try:
            # Reconstruct each component
            psyche = PsycheState.from_dict(data["psyche"])
            loop = LoopState.from_dict(data["loop"])

            # Responsibility uses Pydantic
            from .responsibility import from_dict as resp_from_dict
            responsibility = resp_from_dict(data["responsibility"])

            # Dynamics (optional for backward compatibility with v1)
            if "dynamics" in data and isinstance(data["dynamics"], dict):
                dynamics = DynamicsState.from_dict(data["dynamics"])
            else:
                dynamics = create_dynamics_state()

            return cls(
                psyche=psyche,
                loop=loop,
                responsibility=responsibility,
                dynamics=dynamics,
                version=version,
                user_id=data.get("user_id", "default"),
                created_at=data.get("created_at", datetime.now().isoformat(timespec="seconds")),
                updated_at=data.get("updated_at", datetime.now().isoformat(timespec="seconds")),
            )

        except Exception as e:
            # Any reconstruction error means invalid data
            logger.debug("Snapshot.from_dict reconstruction failed: %s", e)
            return None

    def update_timestamp(self) -> "Snapshot":
        """Return a new snapshot with updated timestamp."""
        return Snapshot(
            psyche=self.psyche,
            loop=self.loop,
            responsibility=self.responsibility,
            dynamics=self.dynamics,
            version=self.version,
            user_id=self.user_id,
            created_at=self.created_at,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )


def create_default_snapshot(user_id: str = "default") -> Snapshot:
    """Create a fresh snapshot with default values."""
    now = datetime.now().isoformat(timespec="seconds")
    return Snapshot(
        psyche=PsycheState(),
        loop=LoopState(
            memory=ShortTermMemory(),
            config=LoopConfig(),
        ),
        responsibility=create_default_responsibility(),
        dynamics=create_dynamics_state(),
        version=SNAPSHOT_VERSION,
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )


def validate_snapshot(snapshot: Snapshot) -> tuple[bool, list[str]]:
    """
    Validate a snapshot's structural integrity.

    Returns (is_valid, list_of_issues).
    Does NOT validate or correct values - only structure.
    """
    issues: list[str] = []

    # Check version
    if snapshot.version > SNAPSHOT_VERSION:
        issues.append(f"Future version {snapshot.version} > {SNAPSHOT_VERSION}")

    # Check psyche has required fields
    try:
        emo = snapshot.psyche.emotions.as_dict()
        required_emotions = {"joy", "anger", "sorrow", "fear", "surprise", "love", "fun"}
        missing = required_emotions - set(emo.keys())
        if missing:
            issues.append(f"Missing emotions: {missing}")
    except Exception as e:
        issues.append(f"Invalid psyche.emotions: {e}")

    # Check drives
    try:
        drv = snapshot.psyche.drives.as_dict()
        required_drives = {"social", "curiosity", "expression"}
        missing = required_drives - set(drv.keys())
        if missing:
            issues.append(f"Missing drives: {missing}")
    except Exception as e:
        issues.append(f"Invalid psyche.drives: {e}")

    # Check mood
    try:
        if not (-1.0 <= snapshot.psyche.mood.valence <= 1.0):
            issues.append(f"Mood valence out of range: {snapshot.psyche.mood.valence}")
        if not (0.0 <= snapshot.psyche.mood.arousal <= 1.0):
            issues.append(f"Mood arousal out of range: {snapshot.psyche.mood.arousal}")
    except Exception as e:
        issues.append(f"Invalid psyche.mood: {e}")

    # Check loop state
    try:
        if snapshot.loop.memory is None:
            issues.append("Loop memory is None")
        if snapshot.loop.config is None:
            issues.append("Loop config is None")
    except Exception as e:
        issues.append(f"Invalid loop state: {e}")

    # Check responsibility
    try:
        if snapshot.responsibility.total_weight < 0:
            issues.append(f"Negative responsibility weight: {snapshot.responsibility.total_weight}")
    except Exception as e:
        issues.append(f"Invalid responsibility: {e}")

    # Check dynamics
    try:
        from .dynamics import DynamicsPhase
        if snapshot.dynamics.phase not in DynamicsPhase:
            issues.append(f"Invalid dynamics phase: {snapshot.dynamics.phase}")
    except Exception as e:
        issues.append(f"Invalid dynamics: {e}")

    return (len(issues) == 0, issues)
