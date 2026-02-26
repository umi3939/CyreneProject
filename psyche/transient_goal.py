"""
transient_goal.py - Transient Goal Selection (一時的目的選択)

This module implements temporary goal selection where at most ONE goal
can be "active" at a time, providing a WEAK bias to decisions and
generating LIGHT responsibility.

Key design principles:
- Selection is TEMPORARY, not permanent commitment
- Only 0 or 1 active goal at any time
- Influence on decisions is SUBTLE (small bias, not hard filter)
- Responsibility is LIGHT (high distance, low weight)
- Release/switch does NOT create "failure" - natural sublimation
- Integrates with introspection trace and long-term dynamics

【思想】
人は複数の目的候補を同時に抱えたまま、
状況に応じて「今回はこれを意識する」という
一時的な選択を行う。
この選択は恒久的な意志ではなく、
撤回可能で、失われても自己否定を生まない。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .goal_candidates import GoalCandidate, CandidateCategory


class GoalReleaseReason(Enum):
    """Reason why an active goal was released."""
    NATURAL_DECAY = "natural_decay"       # Faded over time
    SWITCHED = "switched"                  # Replaced by another goal
    MANUAL_RELEASE = "manual_release"      # Explicitly released
    CANDIDATE_FADED = "candidate_faded"    # Source candidate no longer exists
    CONTEXT_CHANGE = "context_change"      # External conditions changed


@dataclass
class ActiveGoal:
    """
    A temporarily selected goal - at most one exists at any time.

    This represents "what I'm currently focusing on" without
    permanent commitment. It can be released or switched freely.

    Attributes:
        goal_id: Unique identifier for this active goal instance
        candidate_id: ID of the source GoalCandidate
        candidate_category: Category from the source candidate
        selection_strength: How strongly this goal is selected (0.0 to 1.0)
        direction_alignment: Direction from source candidate for bias calculation
        selected_at: Timestamp when this goal was selected
        initial_strength: Original strength at selection time
        decay_rate: How quickly strength decays per turn
    """
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    candidate_id: str = ""
    candidate_category: CandidateCategory = CandidateCategory.EXPLORATION
    selection_strength: float = 0.0
    direction_alignment: dict[str, float] = field(default_factory=dict)
    selected_at: float = field(default_factory=time.time)
    initial_strength: float = 0.0
    decay_rate: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "candidate_id": self.candidate_id,
            "candidate_category": self.candidate_category.value,
            "selection_strength": self.selection_strength,
            "direction_alignment": self.direction_alignment.copy(),
            "selected_at": self.selected_at,
            "initial_strength": self.initial_strength,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveGoal:
        category_str = data.get("candidate_category", "exploration")
        try:
            category = CandidateCategory(category_str)
        except ValueError:
            category = CandidateCategory.EXPLORATION

        return cls(
            goal_id=data.get("goal_id", str(uuid.uuid4())[:8]),
            candidate_id=data.get("candidate_id", ""),
            candidate_category=category,
            selection_strength=data.get("selection_strength", 0.0),
            direction_alignment=data.get("direction_alignment", {}),
            selected_at=data.get("selected_at", time.time()),
            initial_strength=data.get("initial_strength", 0.0),
            decay_rate=data.get("decay_rate", 0.02),
        )


@dataclass
class GoalBias:
    """
    A WEAK bias to apply to decision candidates.

    This bias is SUBTLE - it nudges scores slightly but
    never forces or filters decisions.
    """
    is_active: bool = False
    goal_id: str = ""
    candidate_category: CandidateCategory = CandidateCategory.EXPLORATION
    bias_strength: float = 0.0  # 0.0 to ~0.15 typically
    direction_alignment: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_active": self.is_active,
            "goal_id": self.goal_id,
            "candidate_category": self.candidate_category.value,
            "bias_strength": self.bias_strength,
            "direction_alignment": self.direction_alignment.copy(),
        }


@dataclass
class LightResponsibility:
    """
    A LIGHT responsibility entry generated from goal selection.

    This responsibility is intentionally weak:
    - High distance (feels remote)
    - Low weight (minimal burden)
    - Short time slice (quick dispersion)
    """
    responsibility_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_goal_id: str = ""
    weight: float = 0.1          # Intentionally LOW
    distance: float = 0.8        # Intentionally HIGH (far)
    time_slice_turns: int = 10   # Intentionally SHORT
    created_at: float = field(default_factory=time.time)
    release_reason: Optional[GoalReleaseReason] = None
    sublimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "responsibility_id": self.responsibility_id,
            "source_goal_id": self.source_goal_id,
            "weight": self.weight,
            "distance": self.distance,
            "time_slice_turns": self.time_slice_turns,
            "created_at": self.created_at,
            "release_reason": self.release_reason.value if self.release_reason else None,
            "sublimated": self.sublimated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightResponsibility:
        release_reason = None
        if data.get("release_reason"):
            try:
                release_reason = GoalReleaseReason(data["release_reason"])
            except ValueError:
                pass

        return cls(
            responsibility_id=data.get("responsibility_id", str(uuid.uuid4())[:8]),
            source_goal_id=data.get("source_goal_id", ""),
            weight=data.get("weight", 0.1),
            distance=data.get("distance", 0.8),
            time_slice_turns=data.get("time_slice_turns", 10),
            created_at=data.get("created_at", time.time()),
            release_reason=release_reason,
            sublimated=data.get("sublimated", False),
        )


@dataclass
class TransientGoalConfig:
    """
    Configuration for the transient goal system.

    Attributes:
        min_selection_strength: Minimum strength to select a goal
        decay_rate: How quickly active goal strength decays per turn
        min_active_strength: Minimum strength before goal auto-releases
        bias_multiplier: How much selection strength affects decision bias
        max_bias: Maximum bias that can be applied (keeps influence subtle)
        responsibility_weight: Weight for light responsibility (low)
        responsibility_distance: Distance for light responsibility (high)
        responsibility_time_slice: Time slice turns for responsibility
        auto_select_threshold: Candidate intensity threshold for auto-selection
        auto_select_probability: Probability of auto-selecting a strong candidate
    """
    min_selection_strength: float = 0.3
    decay_rate: float = 0.02
    min_active_strength: float = 0.1
    bias_multiplier: float = 0.15
    max_bias: float = 0.12
    responsibility_weight: float = 0.1
    responsibility_distance: float = 0.8
    responsibility_time_slice: int = 10
    auto_select_threshold: float = 0.7
    auto_select_probability: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_selection_strength": self.min_selection_strength,
            "decay_rate": self.decay_rate,
            "min_active_strength": self.min_active_strength,
            "bias_multiplier": self.bias_multiplier,
            "max_bias": self.max_bias,
            "responsibility_weight": self.responsibility_weight,
            "responsibility_distance": self.responsibility_distance,
            "responsibility_time_slice": self.responsibility_time_slice,
            "auto_select_threshold": self.auto_select_threshold,
            "auto_select_probability": self.auto_select_probability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransientGoalConfig:
        return cls(
            min_selection_strength=data.get("min_selection_strength", 0.3),
            decay_rate=data.get("decay_rate", 0.02),
            min_active_strength=data.get("min_active_strength", 0.1),
            bias_multiplier=data.get("bias_multiplier", 0.15),
            max_bias=data.get("max_bias", 0.12),
            responsibility_weight=data.get("responsibility_weight", 0.1),
            responsibility_distance=data.get("responsibility_distance", 0.8),
            responsibility_time_slice=data.get("responsibility_time_slice", 10),
            auto_select_threshold=data.get("auto_select_threshold", 0.7),
            auto_select_probability=data.get("auto_select_probability", 0.3),
        )


@dataclass
class TransientGoalState:
    """
    State for the transient goal system.

    Contains at most ONE active goal at any time.
    """
    active_goal: Optional[ActiveGoal] = None
    config: TransientGoalConfig = field(default_factory=TransientGoalConfig)
    pending_responsibilities: list[LightResponsibility] = field(default_factory=list)
    turn_count: int = 0
    total_goals_selected: int = 0
    total_goals_released: int = 0
    total_natural_decays: int = 0
    total_switches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
            "config": self.config.to_dict(),
            "pending_responsibilities": [r.to_dict() for r in self.pending_responsibilities],
            "turn_count": self.turn_count,
            "total_goals_selected": self.total_goals_selected,
            "total_goals_released": self.total_goals_released,
            "total_natural_decays": self.total_natural_decays,
            "total_switches": self.total_switches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransientGoalState:
        active_goal = None
        if data.get("active_goal"):
            active_goal = ActiveGoal.from_dict(data["active_goal"])

        config_data = data.get("config", {})
        config = TransientGoalConfig.from_dict(config_data) if config_data else TransientGoalConfig()

        responsibilities = [
            LightResponsibility.from_dict(r)
            for r in data.get("pending_responsibilities", [])
        ]

        return cls(
            active_goal=active_goal,
            config=config,
            pending_responsibilities=responsibilities,
            turn_count=data.get("turn_count", 0),
            total_goals_selected=data.get("total_goals_selected", 0),
            total_goals_released=data.get("total_goals_released", 0),
            total_natural_decays=data.get("total_natural_decays", 0),
            total_switches=data.get("total_switches", 0),
        )


class TransientGoalManager:
    """
    Manages transient goal selection and its effects.

    At most ONE goal can be active at any time. The active goal
    provides a WEAK bias to decisions and generates LIGHT responsibility.
    """

    def __init__(self, config: Optional[TransientGoalConfig] = None):
        self._config = config or TransientGoalConfig()
        self._state = TransientGoalState(config=self._config)

    @property
    def state(self) -> TransientGoalState:
        """Read-only access to current state."""
        return self._state

    @property
    def config(self) -> TransientGoalConfig:
        """Read-only access to configuration."""
        return self._config

    @property
    def has_active_goal(self) -> bool:
        """Check if there is currently an active goal."""
        return self._state.active_goal is not None

    @property
    def active_goal(self) -> Optional[ActiveGoal]:
        """Get the current active goal (or None)."""
        return self._state.active_goal

    def observe_turn(
        self,
        available_candidates: Optional[list[GoalCandidate]] = None,
    ) -> Optional[GoalReleaseReason]:
        """
        Observe a turn: decay active goal, potentially auto-select.

        Args:
            available_candidates: Current goal candidates for auto-selection

        Returns:
            Release reason if a goal was released this turn, None otherwise
        """
        self._state.turn_count += 1
        release_reason = None

        # Step 1: Decay active goal
        if self._state.active_goal:
            self._state.active_goal.selection_strength *= (1.0 - self._config.decay_rate)

            # Check if goal should be released
            if self._state.active_goal.selection_strength < self._config.min_active_strength:
                self._release_goal(GoalReleaseReason.NATURAL_DECAY)
                release_reason = GoalReleaseReason.NATURAL_DECAY

        # Step 2: Process pending responsibilities (disperse over time)
        self._process_responsibilities()

        # Step 3: Potentially auto-select if no active goal
        if not self._state.active_goal and available_candidates:
            self._attempt_auto_select(available_candidates)

        return release_reason

    def select_goal(
        self,
        candidate: GoalCandidate,
        strength: Optional[float] = None,
    ) -> ActiveGoal:
        """
        Select a goal candidate as the active goal.

        If another goal is already active, it will be switched
        (released with SWITCHED reason, not failure).

        Args:
            candidate: The GoalCandidate to select
            strength: Selection strength (defaults to candidate intensity)

        Returns:
            The newly created ActiveGoal
        """
        # Release existing goal if any (switch, not failure)
        if self._state.active_goal:
            self._release_goal(GoalReleaseReason.SWITCHED)
            self._state.total_switches += 1

        # Determine strength
        selection_strength = strength if strength is not None else candidate.intensity
        selection_strength = max(
            self._config.min_selection_strength,
            min(1.0, selection_strength),
        )

        # Create active goal
        active = ActiveGoal(
            candidate_id=candidate.candidate_id,
            candidate_category=candidate.category,
            selection_strength=selection_strength,
            direction_alignment=candidate.direction_expression.copy(),
            initial_strength=selection_strength,
            decay_rate=self._config.decay_rate,
        )

        self._state.active_goal = active
        self._state.total_goals_selected += 1

        # Generate light responsibility for selection
        self._create_selection_responsibility(active)

        return active

    def release_goal(
        self,
        reason: GoalReleaseReason = GoalReleaseReason.MANUAL_RELEASE,
    ) -> Optional[LightResponsibility]:
        """
        Manually release the active goal.

        This is NOT a failure - the responsibility is sublimated naturally.

        Returns:
            The responsibility created from release (for observation)
        """
        if not self._state.active_goal:
            return None

        return self._release_goal(reason)

    def _release_goal(self, reason: GoalReleaseReason) -> LightResponsibility:
        """Internal method to release the active goal."""
        goal = self._state.active_goal
        assert goal is not None

        # Create light responsibility for the release
        # NOT a failure - just natural transition
        responsibility = LightResponsibility(
            source_goal_id=goal.goal_id,
            weight=self._config.responsibility_weight * 0.5,  # Even lighter on release
            distance=self._config.responsibility_distance + 0.1,  # Even more distant
            time_slice_turns=self._config.responsibility_time_slice,
            release_reason=reason,
        )

        self._state.pending_responsibilities.append(responsibility)
        self._state.active_goal = None
        self._state.total_goals_released += 1

        if reason == GoalReleaseReason.NATURAL_DECAY:
            self._state.total_natural_decays += 1

        return responsibility

    def _create_selection_responsibility(self, goal: ActiveGoal) -> None:
        """Create light responsibility for selecting a goal."""
        responsibility = LightResponsibility(
            source_goal_id=goal.goal_id,
            weight=self._config.responsibility_weight,
            distance=self._config.responsibility_distance,
            time_slice_turns=self._config.responsibility_time_slice,
        )
        self._state.pending_responsibilities.append(responsibility)

    def _process_responsibilities(self) -> None:
        """Process pending responsibilities - disperse over time."""
        remaining = []
        for resp in self._state.pending_responsibilities:
            resp.time_slice_turns -= 1
            if resp.time_slice_turns <= 0:
                resp.sublimated = True
            else:
                # Gradually reduce weight (natural sublimation)
                resp.weight *= 0.9
                resp.distance = min(1.0, resp.distance + 0.02)
                if resp.weight > 0.01:
                    remaining.append(resp)

        self._state.pending_responsibilities = remaining

    def _attempt_auto_select(self, candidates: list[GoalCandidate]) -> None:
        """Attempt to automatically select a goal from strong candidates."""
        import random

        # Filter candidates above threshold
        strong = [
            c for c in candidates
            if c.intensity >= self._config.auto_select_threshold
        ]

        if not strong:
            return

        # Probabilistic selection
        if random.random() > self._config.auto_select_probability:
            return

        # Select the strongest candidate
        best = max(strong, key=lambda c: c.intensity)
        self.select_goal(best)

    def get_bias(self) -> GoalBias:
        """
        Get the current goal bias for decision making.

        Returns a WEAK bias that can be applied to decision candidates.
        The bias is intentionally subtle.
        """
        if not self._state.active_goal:
            return GoalBias(is_active=False)

        goal = self._state.active_goal

        # Calculate bias strength (intentionally capped to stay subtle)
        bias_strength = goal.selection_strength * self._config.bias_multiplier
        bias_strength = min(self._config.max_bias, bias_strength)

        return GoalBias(
            is_active=True,
            goal_id=goal.goal_id,
            candidate_category=goal.candidate_category,
            bias_strength=bias_strength,
            direction_alignment=goal.direction_alignment.copy(),
        )

    def get_pending_responsibility_weight(self) -> float:
        """Get total weight of pending (unsublimated) responsibilities."""
        return sum(
            r.weight for r in self._state.pending_responsibilities
            if not r.sublimated
        )

    def save_to_file(self, file_path: str) -> None:
        """Save current state to a JSON file."""
        data = self._state.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str) -> bool:
        """
        Load state from a JSON file.

        Returns:
            True if an active goal was loaded
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._state = TransientGoalState.from_dict(data)
        self._config = self._state.config
        return self._state.active_goal is not None


def create_manager(
    config: Optional[TransientGoalConfig] = None,
) -> TransientGoalManager:
    """Factory function to create a TransientGoalManager."""
    return TransientGoalManager(config=config)


def create_config(
    min_selection_strength: float = 0.3,
    decay_rate: float = 0.02,
    min_active_strength: float = 0.1,
    bias_multiplier: float = 0.15,
    max_bias: float = 0.12,
    responsibility_weight: float = 0.1,
    responsibility_distance: float = 0.8,
    responsibility_time_slice: int = 10,
    auto_select_threshold: float = 0.7,
    auto_select_probability: float = 0.3,
) -> TransientGoalConfig:
    """Factory function to create a TransientGoalConfig."""
    return TransientGoalConfig(
        min_selection_strength=min_selection_strength,
        decay_rate=decay_rate,
        min_active_strength=min_active_strength,
        bias_multiplier=bias_multiplier,
        max_bias=max_bias,
        responsibility_weight=responsibility_weight,
        responsibility_distance=responsibility_distance,
        responsibility_time_slice=responsibility_time_slice,
        auto_select_threshold=auto_select_threshold,
        auto_select_probability=auto_select_probability,
    )


def apply_goal_bias_to_candidate(
    candidate: dict[str, Any],
    bias: GoalBias,
) -> dict[str, Any]:
    """
    Apply goal bias to a single decision candidate.

    The effect is SUBTLE - a small score adjustment based on
    alignment between the candidate and the active goal direction.

    Args:
        candidate: A decision candidate with "score" and optionally "policy"
        bias: The goal bias to apply

    Returns:
        Modified candidate (copy) with adjusted score
    """
    if not bias.is_active or bias.bias_strength < 0.001:
        return candidate

    result = candidate.copy()
    original_score = result.get("score", 0.5)

    # Calculate alignment bonus
    # This is intentionally weak - just a nudge
    alignment = _calculate_alignment(candidate, bias)

    # Apply subtle bias: score * (1 + bias_strength * alignment)
    # alignment ranges from -1 to 1, bias_strength is capped at ~0.12
    # So maximum effect is roughly ±12% score change
    adjustment = bias.bias_strength * alignment
    new_score = original_score * (1.0 + adjustment)

    # Keep score in reasonable bounds
    new_score = max(0.0, min(1.0, new_score))

    result["score"] = new_score
    result["_goal_bias_applied"] = {
        "goal_id": bias.goal_id,
        "alignment": round(alignment, 3),
        "adjustment": round(adjustment, 4),
    }

    return result


def apply_goal_bias_to_candidates(
    candidates: list[dict[str, Any]],
    bias: GoalBias,
) -> list[dict[str, Any]]:
    """
    Apply goal bias to multiple decision candidates.

    The effect is UNIFORM across all candidates - no filtering
    or hard selection, just subtle score adjustments.
    """
    return [apply_goal_bias_to_candidate(c, bias) for c in candidates]


def _calculate_alignment(
    candidate: dict[str, Any],
    bias: GoalBias,
) -> float:
    """
    Calculate how well a candidate aligns with the goal direction.

    Returns a value from -1.0 (opposed) to 1.0 (aligned).
    """
    # Simple heuristic based on category matching
    policy = candidate.get("policy", "")
    category = bias.candidate_category

    # Category-based alignment hints (keys must match thought.py POLICIES policy_label)
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

    # Check if policy matches affinity
    alignment = 0.0
    policy_lower = policy.lower()

    for keyword in affinity_keywords:
        if keyword in policy_lower:
            alignment = 0.5
            break

    # Check direction alignment if candidate has direction info
    if bias.direction_alignment and "direction" in candidate:
        cand_direction = candidate.get("direction", {})
        if cand_direction:
            # Cosine-like similarity
            dot = sum(
                bias.direction_alignment.get(k, 0.0) * cand_direction.get(k, 0.0)
                for k in set(bias.direction_alignment.keys()) | set(cand_direction.keys())
            )
            alignment = max(alignment, min(1.0, dot))

    return alignment


def get_transient_goal_summary(manager: TransientGoalManager) -> dict[str, Any]:
    """
    Get a summary of the transient goal state.

    For observation and debugging purposes.
    """
    state = manager.state
    active = state.active_goal

    return {
        "has_active_goal": active is not None,
        "active_goal": {
            "goal_id": active.goal_id,
            "candidate_id": active.candidate_id,
            "category": active.candidate_category.value,
            "strength": round(active.selection_strength, 3),
            "initial_strength": round(active.initial_strength, 3),
            "age_seconds": round(time.time() - active.selected_at, 1),
        } if active else None,
        "pending_responsibility_weight": round(
            manager.get_pending_responsibility_weight(), 3
        ),
        "pending_responsibility_count": len(state.pending_responsibilities),
        "turn_count": state.turn_count,
        "total_selected": state.total_goals_selected,
        "total_released": state.total_goals_released,
        "total_natural_decays": state.total_natural_decays,
        "total_switches": state.total_switches,
    }


# For introspection trace integration
def create_goal_context_for_trace(
    manager: TransientGoalManager,
) -> Optional[dict[str, Any]]:
    """
    Create a context dict for introspection trace logging.

    Records which goal (if any) was active during the decision,
    and its influence strength.
    """
    if not manager.has_active_goal:
        return None

    goal = manager.active_goal
    assert goal is not None

    bias = manager.get_bias()

    return {
        "active_goal": {
            "goal_id": goal.goal_id,
            "candidate_id": goal.candidate_id,
            "category": goal.candidate_category.value,
            "selection_strength": round(goal.selection_strength, 3),
            "bias_strength": round(bias.bias_strength, 4),
        },
        "pending_responsibility_weight": round(
            manager.get_pending_responsibility_weight(), 3
        ),
        "observation_note": "This goal was temporarily active during decision. "
                          "It provided a subtle bias, not a hard filter.",
    }


# For long-term dynamics integration
def create_goal_stats_for_dynamics(
    manager: TransientGoalManager,
) -> dict[str, Any]:
    """
    Create stats for long-term dynamics logging.

    Returns aggregated statistics suitable for window-based observation.
    """
    state = manager.state
    active = state.active_goal

    return {
        "is_active": active is not None,
        "active_category": active.candidate_category.value if active else None,
        "active_strength": round(active.selection_strength, 3) if active else 0.0,
        "pending_responsibility_weight": round(
            manager.get_pending_responsibility_weight(), 3
        ),
        "total_selected": state.total_goals_selected,
        "total_released": state.total_goals_released,
    }


# For responsibility system integration
def get_responsibilities_for_dispersion(
    manager: TransientGoalManager,
) -> list[dict[str, Any]]:
    """
    Get pending responsibilities in a format suitable for
    the responsibility dispersion system.

    These are LIGHT responsibilities - already high distance, low weight.
    They should be dispersed/sublimated naturally, never treated as failures.
    """
    return [
        {
            "source": "transient_goal",
            "source_id": r.source_goal_id,
            "weight": r.weight,
            "distance": r.distance,
            "remaining_turns": r.time_slice_turns,
            "release_reason": r.release_reason.value if r.release_reason else None,
            "is_failure": False,  # NEVER a failure
            "sublimation_eligible": True,
        }
        for r in manager.state.pending_responsibilities
        if not r.sublimated
    ]


def to_dict(config: TransientGoalConfig) -> dict[str, Any]:
    """Convert config to dictionary."""
    return config.to_dict()


def from_dict(data: dict[str, Any]) -> TransientGoalConfig:
    """Create config from dictionary."""
    return TransientGoalConfig.from_dict(data)
