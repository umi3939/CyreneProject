"""
scoped_goal.py - Scoped Goal Commitment (行動スコープ限定の目的コミット)

This module implements EPHEMERAL goal commitment that lives for exactly
one action/turn and is then automatically wiped. It creates a temporary
"focus" from a TransientGoal that biases decisions without persisting.

Key design principles:
- EPHEMERAL ONLY: ScopedGoal is NEVER persisted to disk
- Lives for ONE action/turn, then destroyed
- NO SUCCESS/FAILURE judgment - just records that attempt was made
- Lightweight responsibility that is dispersion/sublimation eligible

Flow: TransientGoal → ScopedGoal → Decision Bias → Lightweight Responsibility

【思想】
目的は恒久的に決定されるものではなく、
「この行動だけは、これを意識して行う」という
限定的・一時的なコミットとして発生する。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .transient_goal import ActiveGoal, TransientGoalManager, GoalBias
from .goal_candidates import CandidateCategory


class ScopeType(Enum):
    """Defines the lifespan of a scoped goal."""
    SINGLE_ACTION = "single_action"  # Expires after one action
    SINGLE_TURN = "single_turn"      # Expires after one turn


class ScopeStatus(Enum):
    """Status of the scoped goal lifecycle."""
    ACTIVE = "active"        # Currently in effect
    USED = "used"            # Action taken under this scope
    EXPIRED = "expired"      # Scope ended, awaiting cleanup
    CLEARED = "cleared"      # Fully cleaned up


@dataclass
class ScopedGoal:
    """
    A temporary, EPHEMERAL goal commitment for a single action/turn.

    This is NOT persisted. It lives in memory only and is automatically
    destroyed after use. It represents "right now, I'm focusing on this"
    without any permanent commitment.

    Attributes:
        scope_id: Unique identifier for this scope instance
        source_goal_id: ID of the TransientGoal this was created from
        source_candidate_id: ID of the original GoalCandidate
        category: Category from the source goal
        strength: Influence strength on decisions (weak, configurable)
        direction_alignment: Direction info for bias calculation
        scope_type: Whether this is single_action or single_turn
        committed_at: Timestamp when this scope was created
        turn_committed: Turn number when committed
        status: Current lifecycle status
        action_taken: Whether an action was taken under this scope
    """
    scope_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_goal_id: str = ""
    source_candidate_id: str = ""
    category: CandidateCategory = CandidateCategory.EXPLORATION
    strength: float = 0.0
    direction_alignment: dict[str, float] = field(default_factory=dict)
    scope_type: ScopeType = ScopeType.SINGLE_TURN
    committed_at: float = field(default_factory=time.time)
    turn_committed: int = 0
    status: ScopeStatus = ScopeStatus.ACTIVE
    action_taken: bool = False

    # NOT serializable - this is intentionally ephemeral
    def __post_init__(self):
        """Mark that this object should never be serialized."""
        self._ephemeral = True


@dataclass
class ScopedBias:
    """
    A bias to apply to decision candidates from a scoped goal.

    Similar to GoalBias but specifically for the ephemeral scope.
    """
    is_active: bool = False
    scope_id: str = ""
    category: CandidateCategory = CandidateCategory.EXPLORATION
    bias_strength: float = 0.0
    direction_alignment: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """For logging/tracing only - NOT for persistence."""
        return {
            "is_active": self.is_active,
            "scope_id": self.scope_id,
            "category": self.category.value,
            "bias_strength": self.bias_strength,
        }


@dataclass
class ScopedResponsibility:
    """
    A lightweight responsibility entry from acting under a scoped goal.

    This is:
    - NOT evaluated for success/failure
    - Eligible for immediate dispersion/sublimation
    - Very light weight and high distance
    """
    responsibility_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_scope_id: str = ""
    source_goal_id: str = ""
    category: CandidateCategory = CandidateCategory.EXPLORATION
    weight: float = 0.05        # Very light (lighter than transient_goal)
    distance: float = 0.9       # Very far
    created_at: float = field(default_factory=time.time)
    # Explicitly mark: no success/failure judgment
    success: None = None
    failure: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "responsibility_id": self.responsibility_id,
            "source_scope_id": self.source_scope_id,
            "source_goal_id": self.source_goal_id,
            "category": self.category.value,
            "weight": self.weight,
            "distance": self.distance,
            "created_at": self.created_at,
            "is_evaluated": False,  # NEVER evaluated
            "is_failure": False,    # NEVER a failure
            "sublimation_eligible": True,
        }


@dataclass
class ScopedGoalConfig:
    """
    Configuration for the scoped goal system.

    Attributes:
        base_strength: Base strength for scoped goal bias
        strength_variance: Random variance in strength
        max_strength: Maximum allowed strength (keeps influence weak)
        commit_probability: Probability of committing when transient goal exists
        responsibility_weight: Weight for scoped responsibility (very light)
        responsibility_distance: Distance for scoped responsibility (very far)
    """
    base_strength: float = 0.08
    strength_variance: float = 0.03
    max_strength: float = 0.10
    commit_probability: float = 0.6
    responsibility_weight: float = 0.05
    responsibility_distance: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_strength": self.base_strength,
            "strength_variance": self.strength_variance,
            "max_strength": self.max_strength,
            "commit_probability": self.commit_probability,
            "responsibility_weight": self.responsibility_weight,
            "responsibility_distance": self.responsibility_distance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScopedGoalConfig:
        return cls(
            base_strength=data.get("base_strength", 0.08),
            strength_variance=data.get("strength_variance", 0.03),
            max_strength=data.get("max_strength", 0.10),
            commit_probability=data.get("commit_probability", 0.6),
            responsibility_weight=data.get("responsibility_weight", 0.05),
            responsibility_distance=data.get("responsibility_distance", 0.9),
        )


class ScopedGoalSystem:
    """
    Manages EPHEMERAL scoped goal commitments.

    This system creates temporary focus from TransientGoals, applies
    bias to decisions, records lightweight responsibility, and then
    automatically clears the scope.

    IMPORTANT: Nothing in this system is persisted. On restart,
    there is no scoped goal - by design.
    """

    def __init__(self, config: Optional[ScopedGoalConfig] = None):
        self._config = config or ScopedGoalConfig()
        self._current_scope: Optional[ScopedGoal] = None
        self._pending_responsibilities: list[ScopedResponsibility] = []
        self._turn_count: int = 0
        self._total_scopes_created: int = 0
        self._total_actions_under_scope: int = 0

    @property
    def config(self) -> ScopedGoalConfig:
        """Read-only access to configuration."""
        return self._config

    @property
    def has_active_scope(self) -> bool:
        """Check if there is currently an active scope."""
        return (
            self._current_scope is not None and
            self._current_scope.status == ScopeStatus.ACTIVE
        )

    @property
    def current_scope(self) -> Optional[ScopedGoal]:
        """Get the current scope (may be None or non-active)."""
        return self._current_scope

    def begin_turn(
        self,
        transient_manager: Optional[TransientGoalManager] = None,
        active_goal: Optional[ActiveGoal] = None,
    ) -> Optional[ScopedGoal]:
        """
        Begin a new turn, potentially creating a scoped goal.

        This should be called at the start of each turn. It will:
        1. Clear any expired scope from the previous turn
        2. Potentially create a new scoped goal from the transient goal

        Args:
            transient_manager: The TransientGoalManager to get active goal from
            active_goal: Alternatively, pass the active goal directly

        Returns:
            The newly created ScopedGoal, or None if not created
        """
        self._turn_count += 1

        # Step 1: Clear any expired scope
        self._clear_expired_scope()

        # Step 2: Get active goal
        goal = active_goal
        if goal is None and transient_manager is not None:
            goal = transient_manager.active_goal

        if goal is None:
            return None

        # Step 3: Probabilistic commitment
        return self._attempt_commit(goal)

    def _attempt_commit(self, goal: ActiveGoal) -> Optional[ScopedGoal]:
        """Attempt to create a scoped goal from the transient goal."""
        import random

        # Probabilistic check
        if random.random() > self._config.commit_probability:
            return None

        # Already have an active scope
        if self.has_active_scope:
            return None

        # Calculate strength with variance
        strength = self._config.base_strength + random.uniform(
            -self._config.strength_variance,
            self._config.strength_variance,
        )
        strength = max(0.01, min(self._config.max_strength, strength))

        # Create scoped goal
        scope = ScopedGoal(
            source_goal_id=goal.goal_id,
            source_candidate_id=goal.candidate_id,
            category=goal.candidate_category,
            strength=strength,
            direction_alignment=goal.direction_alignment.copy(),
            scope_type=ScopeType.SINGLE_TURN,
            turn_committed=self._turn_count,
            status=ScopeStatus.ACTIVE,
        )

        self._current_scope = scope
        self._total_scopes_created += 1

        return scope

    def get_bias(self) -> ScopedBias:
        """
        Get the current scoped bias for decision making.

        Returns an inactive bias if no active scope exists.
        """
        if not self.has_active_scope:
            return ScopedBias(is_active=False)

        scope = self._current_scope
        assert scope is not None

        return ScopedBias(
            is_active=True,
            scope_id=scope.scope_id,
            category=scope.category,
            bias_strength=scope.strength,
            direction_alignment=scope.direction_alignment.copy(),
        )

    def record_action(self, decision_info: Optional[dict[str, Any]] = None) -> Optional[ScopedResponsibility]:
        """
        Record that an action was taken under the current scope.

        This:
        1. Marks the scope as USED
        2. Creates a lightweight responsibility entry
        3. Does NOT evaluate success/failure

        Args:
            decision_info: Optional info about the decision (for logging only)

        Returns:
            The created ScopedResponsibility, or None if no active scope
        """
        if not self.has_active_scope:
            return None

        scope = self._current_scope
        assert scope is not None

        # Mark scope as used
        scope.action_taken = True
        scope.status = ScopeStatus.USED
        self._total_actions_under_scope += 1

        # Create lightweight responsibility
        responsibility = ScopedResponsibility(
            source_scope_id=scope.scope_id,
            source_goal_id=scope.source_goal_id,
            category=scope.category,
            weight=self._config.responsibility_weight,
            distance=self._config.responsibility_distance,
        )

        self._pending_responsibilities.append(responsibility)

        return responsibility

    def end_turn(self) -> None:
        """
        End the current turn, expiring any active scope.

        This should be called at the end of each turn. It marks
        the scope as expired, ready for cleanup on next turn.
        """
        if self._current_scope is not None:
            if self._current_scope.status in (ScopeStatus.ACTIVE, ScopeStatus.USED):
                self._current_scope.status = ScopeStatus.EXPIRED

    def _clear_expired_scope(self) -> None:
        """Clear any expired scope from memory."""
        if self._current_scope is not None:
            if self._current_scope.status == ScopeStatus.EXPIRED:
                self._current_scope.status = ScopeStatus.CLEARED
                self._current_scope = None

    def get_pending_responsibilities(self) -> list[ScopedResponsibility]:
        """Get pending responsibilities (copies for observation)."""
        return [
            ScopedResponsibility(
                responsibility_id=r.responsibility_id,
                source_scope_id=r.source_scope_id,
                source_goal_id=r.source_goal_id,
                category=r.category,
                weight=r.weight,
                distance=r.distance,
                created_at=r.created_at,
            )
            for r in self._pending_responsibilities
        ]

    def clear_responsibilities(self) -> list[ScopedResponsibility]:
        """
        Clear and return pending responsibilities.

        Called by the responsibility system after integrating them.
        """
        result = self._pending_responsibilities
        self._pending_responsibilities = []
        return result

    def get_total_pending_weight(self) -> float:
        """Get total weight of pending responsibilities."""
        return sum(r.weight for r in self._pending_responsibilities)


def create_system(
    config: Optional[ScopedGoalConfig] = None,
) -> ScopedGoalSystem:
    """Factory function to create a ScopedGoalSystem."""
    return ScopedGoalSystem(config=config)


def create_config(
    base_strength: float = 0.08,
    strength_variance: float = 0.03,
    max_strength: float = 0.10,
    commit_probability: float = 0.6,
    responsibility_weight: float = 0.05,
    responsibility_distance: float = 0.9,
) -> ScopedGoalConfig:
    """Factory function to create a ScopedGoalConfig."""
    return ScopedGoalConfig(
        base_strength=base_strength,
        strength_variance=strength_variance,
        max_strength=max_strength,
        commit_probability=commit_probability,
        responsibility_weight=responsibility_weight,
        responsibility_distance=responsibility_distance,
    )


def apply_scoped_bias_to_candidate(
    candidate: dict[str, Any],
    bias: ScopedBias,
) -> dict[str, Any]:
    """
    Apply scoped goal bias to a single decision candidate.

    The effect is very subtle - even weaker than transient goal bias.
    This is a momentary focus, not a strong influence.

    Args:
        candidate: A decision candidate with "score"
        bias: The scoped bias to apply

    Returns:
        Modified candidate (copy) with adjusted score
    """
    if not bias.is_active or bias.bias_strength < 0.001:
        return candidate

    result = candidate.copy()
    original_score = result.get("score", 0.5)

    # Calculate alignment
    alignment = _calculate_scope_alignment(candidate, bias)

    # Apply very subtle bias
    adjustment = bias.bias_strength * alignment
    new_score = original_score * (1.0 + adjustment)

    # Keep in bounds
    new_score = max(0.0, min(1.0, new_score))

    result["score"] = new_score
    result["_scoped_bias_applied"] = {
        "scope_id": bias.scope_id,
        "adjustment": round(adjustment, 4),
    }

    return result


def apply_scoped_bias_to_candidates(
    candidates: list[dict[str, Any]],
    bias: ScopedBias,
) -> list[dict[str, Any]]:
    """Apply scoped bias to multiple decision candidates."""
    return [apply_scoped_bias_to_candidate(c, bias) for c in candidates]


def _calculate_scope_alignment(
    candidate: dict[str, Any],
    bias: ScopedBias,
) -> float:
    """
    Calculate alignment between candidate and scoped goal.

    Returns value from -1.0 to 1.0.
    """
    policy = candidate.get("policy", "")
    category = bias.category

    # Category-based hints (keys must match thought.py POLICIES policy_label)
    category_policy_affinity = {
        CandidateCategory.APPROACH: ["共感する", "励ます", "質問で会話を広げる", "提案する"],
        CandidateCategory.AVOIDANCE: ["黙って聞く", "見守る", "話題を変える", "確認する"],
        CandidateCategory.CONNECTION: ["共感する", "励ます", "質問で会話を広げる", "同意する"],
        CandidateCategory.ISOLATION: ["黙って聞く", "見守る", "話題を変える"],
        CandidateCategory.EXPRESSION: ["感想を述べる", "冗談を言う", "からかう", "自分の経験を話す"],
        CandidateCategory.ABSORPTION: ["黙って聞く", "見守る", "確認する"],
        CandidateCategory.EXPLORATION: ["質問で会話を広げる", "確認する", "提案する"],
        CandidateCategory.MAINTENANCE: ["同意する", "感想を述べる", "確認する"],
    }

    affinity_keywords = category_policy_affinity.get(category, [])
    alignment = 0.0
    policy_lower = policy.lower()

    for keyword in affinity_keywords:
        if keyword in policy_lower:
            alignment = 0.4  # Slightly weaker than transient
            break

    # Direction alignment
    if bias.direction_alignment and "direction" in candidate:
        cand_direction = candidate.get("direction", {})
        if cand_direction:
            dot = sum(
                bias.direction_alignment.get(k, 0.0) * cand_direction.get(k, 0.0)
                for k in set(bias.direction_alignment.keys()) | set(cand_direction.keys())
            )
            alignment = max(alignment, min(1.0, dot))

    return alignment


def get_scoped_goal_summary(system: ScopedGoalSystem) -> dict[str, Any]:
    """
    Get a summary of the scoped goal system state.

    For observation and debugging purposes.
    """
    scope = system.current_scope

    return {
        "has_active_scope": system.has_active_scope,
        "current_scope": {
            "scope_id": scope.scope_id,
            "source_goal_id": scope.source_goal_id,
            "category": scope.category.value,
            "strength": round(scope.strength, 4),
            "status": scope.status.value,
            "action_taken": scope.action_taken,
            "turn_committed": scope.turn_committed,
        } if scope else None,
        "pending_responsibility_count": len(system._pending_responsibilities),
        "pending_responsibility_weight": round(system.get_total_pending_weight(), 4),
        "turn_count": system._turn_count,
        "total_scopes_created": system._total_scopes_created,
        "total_actions_under_scope": system._total_actions_under_scope,
    }


# For introspection trace integration
def create_scope_context_for_trace(
    system: ScopedGoalSystem,
) -> Optional[dict[str, Any]]:
    """
    Create a context dict for introspection trace logging.

    Records that a scoped goal was active during this decision.
    """
    if not system.has_active_scope:
        return None

    scope = system.current_scope
    assert scope is not None

    bias = system.get_bias()

    return {
        "scoped_goal": {
            "scope_id": scope.scope_id,
            "source_goal_id": scope.source_goal_id,
            "category": scope.category.value,
            "strength": round(scope.strength, 4),
            "action_taken": scope.action_taken,
        },
        "bias_strength": round(bias.bias_strength, 4),
        "observation_note": "This scoped goal was a momentary focus. "
                          "It is EPHEMERAL and will not persist.",
    }


# For responsibility system integration
def get_responsibilities_for_integration(
    system: ScopedGoalSystem,
) -> list[dict[str, Any]]:
    """
    Get pending responsibilities in a format for the responsibility system.

    These are very lightweight and always eligible for sublimation.
    NO success/failure evaluation.
    """
    return [
        {
            "source": "scoped_goal",
            "source_id": r.source_scope_id,
            "source_goal_id": r.source_goal_id,
            "category": r.category.value,
            "weight": r.weight,
            "distance": r.distance,
            "is_evaluated": False,      # NEVER evaluated
            "is_failure": False,        # NEVER a failure
            "is_success": False,        # NEVER a success either
            "sublimation_eligible": True,
            "dispersion_eligible": True,
        }
        for r in system._pending_responsibilities
    ]


def to_dict(config: ScopedGoalConfig) -> dict[str, Any]:
    """Convert config to dictionary."""
    return config.to_dict()


def from_dict(data: dict[str, Any]) -> ScopedGoalConfig:
    """Create config from dictionary."""
    return ScopedGoalConfig.from_dict(data)


# Complete flow helper
def execute_scoped_decision_flow(
    system: ScopedGoalSystem,
    candidates: list[dict[str, Any]],
    select_fn: Optional[callable] = None,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], Optional[ScopedResponsibility]]:
    """
    Execute the complete scoped goal decision flow.

    This is a helper that:
    1. Gets the current scoped bias
    2. Applies it to candidates
    3. Optionally selects using provided function
    4. Records the action if selection was made

    Args:
        system: The ScopedGoalSystem
        candidates: Decision candidates
        select_fn: Optional function to select from biased candidates

    Returns:
        Tuple of (biased_candidates, selected_candidate, responsibility)
    """
    # Get and apply bias
    bias = system.get_bias()
    biased_candidates = apply_scoped_bias_to_candidates(candidates, bias)

    # Select if function provided
    selected = None
    if select_fn is not None and biased_candidates:
        selected = select_fn(biased_candidates)

    # Record action if selection was made
    responsibility = None
    if selected is not None:
        responsibility = system.record_action(decision_info=selected)

    return biased_candidates, selected, responsibility
