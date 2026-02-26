"""
repeated_tendency.py - Repeated Scoped-Goal Tendency (反復傾向の形成)

This module implements MEDIUM-TERM tendency formation from repeated
scoped goal usage. It creates "habit" or "inertia" - NOT personality.

Key design principles:
- This is "habit/inertia", NOT "personality" or "belief"
- Influence is WEAK - creates a "slope" (easier), not a "wall" (must)
- MEDIUM-TERM: between ValueOrientation (long) and TransientGoal (short)
- No success/failure evaluation
- Natural decay when repetition stops

【思想】
人は一度の選択で自分を決めない。
しかし、同じような選択を何度も繰り返すことで、
「気づけば、こういう動きをする存在になっている」
という傾向が静かに形成される。
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .goal_candidates import CandidateCategory
from .scoped_goal import ScopedGoal


@dataclass
class TendencyPattern:
    """
    A pattern that identifies similar scoped goals.

    Used to group goals that are "similar enough" to count
    as repetition of the same tendency.
    """
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: CandidateCategory = CandidateCategory.EXPLORATION
    direction_signature: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "category": self.category.value,
            "direction_signature": self.direction_signature.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TendencyPattern:
        category_str = data.get("category", "exploration")
        try:
            category = CandidateCategory(category_str)
        except ValueError:
            category = CandidateCategory.EXPLORATION

        return cls(
            pattern_id=data.get("pattern_id", str(uuid.uuid4())[:8]),
            category=category,
            direction_signature=data.get("direction_signature", {}),
        )


@dataclass
class UsageRecord:
    """
    A record of a single scoped goal usage.

    Used to track history for repetition detection.
    """
    turn: int
    timestamp: float
    pattern: TendencyPattern
    scope_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "timestamp": self.timestamp,
            "pattern": self.pattern.to_dict(),
            "scope_id": self.scope_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        pattern_data = data.get("pattern", {})
        pattern = TendencyPattern.from_dict(pattern_data) if pattern_data else TendencyPattern()

        return cls(
            turn=data.get("turn", 0),
            timestamp=data.get("timestamp", time.time()),
            pattern=pattern,
            scope_id=data.get("scope_id", ""),
        )


@dataclass
class Tendency:
    """
    A formed tendency from repeated scoped goal usage.

    This is "habit" or "inertia" - NOT personality or belief.
    It creates a slight slope toward certain choices but never
    forces or blocks any decision.

    Attributes:
        tendency_id: Unique identifier
        pattern: The pattern this tendency is associated with
        strength: Current strength (0.0 to max_strength, weak by design)
        confidence: How established this tendency is (repetition count based)
        last_used_turn: Turn when this tendency was last reinforced
        first_formed_turn: Turn when this tendency first emerged
        total_reinforcements: Total number of times reinforced
        consecutive_misses: Turns since last reinforcement
    """
    tendency_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern: TendencyPattern = field(default_factory=TendencyPattern)
    strength: float = 0.0
    confidence: float = 0.0
    last_used_turn: int = 0
    first_formed_turn: int = 0
    total_reinforcements: int = 0
    consecutive_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tendency_id": self.tendency_id,
            "pattern": self.pattern.to_dict(),
            "strength": self.strength,
            "confidence": self.confidence,
            "last_used_turn": self.last_used_turn,
            "first_formed_turn": self.first_formed_turn,
            "total_reinforcements": self.total_reinforcements,
            "consecutive_misses": self.consecutive_misses,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tendency:
        pattern_data = data.get("pattern", {})
        pattern = TendencyPattern.from_dict(pattern_data) if pattern_data else TendencyPattern()

        return cls(
            tendency_id=data.get("tendency_id", str(uuid.uuid4())[:8]),
            pattern=pattern,
            strength=data.get("strength", 0.0),
            confidence=data.get("confidence", 0.0),
            last_used_turn=data.get("last_used_turn", 0),
            first_formed_turn=data.get("first_formed_turn", 0),
            total_reinforcements=data.get("total_reinforcements", 0),
            consecutive_misses=data.get("consecutive_misses", 0),
        )


@dataclass
class TendencyBias:
    """
    A WEAK bias from tendencies to apply to decision candidates.

    This creates a "slope" - makes some choices slightly easier -
    but never creates a "wall" that blocks other choices.
    """
    has_bias: bool = False
    biases: dict[str, float] = field(default_factory=dict)  # category -> bias
    strongest_category: Optional[CandidateCategory] = None
    strongest_bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_bias": self.has_bias,
            "biases": self.biases.copy(),
            "strongest_category": self.strongest_category.value if self.strongest_category else None,
            "strongest_bias": self.strongest_bias,
        }


@dataclass
class RepeatedTendencyConfig:
    """
    Configuration for the repeated tendency system.

    All values are intentionally weak to ensure tendencies
    create slopes, not walls.

    Attributes:
        history_window: Number of recent usages to keep
        min_repetitions: Minimum repetitions to form a tendency
        recency_window: Turns within which usages count as "recent"
        strength_increment: How much strength increases per reinforcement
        max_strength: Maximum allowed strength (keeps influence weak)
        confidence_increment: How much confidence increases per reinforcement
        max_confidence: Maximum confidence level
        decay_rate: Strength decay per turn of non-use
        miss_decay_multiplier: Extra decay when consecutive misses occur
        similarity_threshold: How similar patterns must be to match
        bias_multiplier: How much tendency strength affects decision bias
        max_bias: Maximum bias that can be applied
    """
    history_window: int = 50
    min_repetitions: int = 3
    recency_window: int = 20
    strength_increment: float = 0.02
    max_strength: float = 0.15
    confidence_increment: float = 0.05
    max_confidence: float = 0.8
    decay_rate: float = 0.01
    miss_decay_multiplier: float = 1.5
    similarity_threshold: float = 0.7
    bias_multiplier: float = 0.08
    max_bias: float = 0.06

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_window": self.history_window,
            "min_repetitions": self.min_repetitions,
            "recency_window": self.recency_window,
            "strength_increment": self.strength_increment,
            "max_strength": self.max_strength,
            "confidence_increment": self.confidence_increment,
            "max_confidence": self.max_confidence,
            "decay_rate": self.decay_rate,
            "miss_decay_multiplier": self.miss_decay_multiplier,
            "similarity_threshold": self.similarity_threshold,
            "bias_multiplier": self.bias_multiplier,
            "max_bias": self.max_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepeatedTendencyConfig:
        return cls(
            history_window=data.get("history_window", 50),
            min_repetitions=data.get("min_repetitions", 3),
            recency_window=data.get("recency_window", 20),
            strength_increment=data.get("strength_increment", 0.02),
            max_strength=data.get("max_strength", 0.15),
            confidence_increment=data.get("confidence_increment", 0.05),
            max_confidence=data.get("max_confidence", 0.8),
            decay_rate=data.get("decay_rate", 0.01),
            miss_decay_multiplier=data.get("miss_decay_multiplier", 1.5),
            similarity_threshold=data.get("similarity_threshold", 0.7),
            bias_multiplier=data.get("bias_multiplier", 0.08),
            max_bias=data.get("max_bias", 0.06),
        )


@dataclass
class RepeatedTendencyState:
    """
    State for the repeated tendency system.

    Persistence is optional - tendencies naturally decay anyway.
    """
    tendencies: list[Tendency] = field(default_factory=list)
    usage_history: list[UsageRecord] = field(default_factory=list)
    config: RepeatedTendencyConfig = field(default_factory=RepeatedTendencyConfig)
    turn_count: int = 0
    total_tendencies_formed: int = 0
    total_tendencies_faded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tendencies": [t.to_dict() for t in self.tendencies],
            "usage_history": [u.to_dict() for u in self.usage_history],
            "config": self.config.to_dict(),
            "turn_count": self.turn_count,
            "total_tendencies_formed": self.total_tendencies_formed,
            "total_tendencies_faded": self.total_tendencies_faded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepeatedTendencyState:
        tendencies = [Tendency.from_dict(t) for t in data.get("tendencies", [])]
        history = [UsageRecord.from_dict(u) for u in data.get("usage_history", [])]
        config_data = data.get("config", {})
        config = RepeatedTendencyConfig.from_dict(config_data) if config_data else RepeatedTendencyConfig()

        return cls(
            tendencies=tendencies,
            usage_history=history,
            config=config,
            turn_count=data.get("turn_count", 0),
            total_tendencies_formed=data.get("total_tendencies_formed", 0),
            total_tendencies_faded=data.get("total_tendencies_faded", 0),
        )


class RepeatedTendencySystem:
    """
    Tracks scoped goal usage and forms MEDIUM-TERM tendencies.

    This creates "habit" or "inertia" - NOT personality or rules.
    Tendencies make certain choices slightly easier but never
    block or force any decision.
    """

    def __init__(self, config: Optional[RepeatedTendencyConfig] = None):
        self._config = config or RepeatedTendencyConfig()
        self._state = RepeatedTendencyState(config=self._config)

    @property
    def config(self) -> RepeatedTendencyConfig:
        """Read-only access to configuration."""
        return self._config

    @property
    def state(self) -> RepeatedTendencyState:
        """Read-only access to state."""
        return self._state

    def observe_turn(
        self,
        scoped_goal_used: Optional[ScopedGoal] = None,
    ) -> Optional[Tendency]:
        """
        Observe a turn, optionally recording scoped goal usage.

        This method:
        1. Records usage if a scoped goal was used
        2. Detects repetition and forms/reinforces tendencies
        3. Applies decay to existing tendencies
        4. Removes faded tendencies

        Args:
            scoped_goal_used: The ScopedGoal that was used this turn (if any)

        Returns:
            Tendency that was formed or reinforced, or None
        """
        self._state.turn_count += 1
        affected_tendency = None

        # Step 1: Record usage if scoped goal was used
        if scoped_goal_used is not None and scoped_goal_used.action_taken:
            pattern = self._extract_pattern(scoped_goal_used)
            record = UsageRecord(
                turn=self._state.turn_count,
                timestamp=time.time(),
                pattern=pattern,
                scope_id=scoped_goal_used.scope_id,
            )
            self._add_usage_record(record)

            # Step 2: Check for repetition and form/reinforce tendency
            affected_tendency = self._process_repetition(pattern)

        # Step 3: Apply decay to all tendencies
        self._apply_decay(scoped_goal_used)

        # Step 4: Remove faded tendencies
        self._prune_faded_tendencies()

        return affected_tendency

    def _extract_pattern(self, scope: ScopedGoal) -> TendencyPattern:
        """Extract a pattern from a scoped goal for matching."""
        # Simplify direction to key dimensions
        simplified_direction = {}
        for key, value in scope.direction_alignment.items():
            if abs(value) > 0.2:  # Only significant dimensions
                # Round to reduce noise
                simplified_direction[key] = round(value, 1)

        return TendencyPattern(
            category=scope.category,
            direction_signature=simplified_direction,
        )

    def _add_usage_record(self, record: UsageRecord) -> None:
        """Add a usage record to history, maintaining window size."""
        self._state.usage_history.append(record)

        # Trim to window size
        while len(self._state.usage_history) > self._config.history_window:
            self._state.usage_history.pop(0)

    def _process_repetition(self, pattern: TendencyPattern) -> Optional[Tendency]:
        """Check for repetition and form/reinforce tendency."""
        # Find existing tendency for this pattern
        existing = self._find_matching_tendency(pattern)

        # Count recent similar usages
        recent_count = self._count_recent_similar(pattern)

        if existing:
            # Reinforce existing tendency
            self._reinforce_tendency(existing)
            return existing
        elif recent_count >= self._config.min_repetitions:
            # Form new tendency
            return self._form_tendency(pattern)

        return None

    def _find_matching_tendency(self, pattern: TendencyPattern) -> Optional[Tendency]:
        """Find an existing tendency that matches the pattern."""
        for tendency in self._state.tendencies:
            if self._patterns_similar(tendency.pattern, pattern):
                return tendency
        return None

    def _patterns_similar(self, p1: TendencyPattern, p2: TendencyPattern) -> bool:
        """Check if two patterns are similar enough to be the same tendency."""
        # Must be same category
        if p1.category != p2.category:
            return False

        # Check direction similarity
        similarity = self._compute_direction_similarity(
            p1.direction_signature,
            p2.direction_signature,
        )

        return similarity >= self._config.similarity_threshold

    def _compute_direction_similarity(
        self,
        d1: dict[str, float],
        d2: dict[str, float],
    ) -> float:
        """Compute similarity between two direction signatures."""
        if not d1 and not d2:
            return 1.0
        if not d1 or not d2:
            return 0.5  # Partial match for empty vs non-empty

        all_keys = set(d1.keys()) | set(d2.keys())
        if not all_keys:
            return 1.0

        # Compute cosine-like similarity
        dot = sum(d1.get(k, 0.0) * d2.get(k, 0.0) for k in all_keys)
        mag1 = sum(v ** 2 for v in d1.values()) ** 0.5
        mag2 = sum(v ** 2 for v in d2.values()) ** 0.5

        if mag1 < 0.001 or mag2 < 0.001:
            return 0.5

        return max(0.0, min(1.0, dot / (mag1 * mag2)))

    def _count_recent_similar(self, pattern: TendencyPattern) -> int:
        """Count recent usages similar to the pattern."""
        current_turn = self._state.turn_count
        count = 0

        for record in reversed(self._state.usage_history):
            # Only count within recency window
            if current_turn - record.turn > self._config.recency_window:
                break

            if self._patterns_similar(record.pattern, pattern):
                count += 1

        return count

    def _form_tendency(self, pattern: TendencyPattern) -> Tendency:
        """Form a new tendency from a pattern."""
        tendency = Tendency(
            pattern=pattern,
            strength=self._config.strength_increment,
            confidence=self._config.confidence_increment,
            last_used_turn=self._state.turn_count,
            first_formed_turn=self._state.turn_count,
            total_reinforcements=1,
            consecutive_misses=0,
        )

        self._state.tendencies.append(tendency)
        self._state.total_tendencies_formed += 1

        return tendency

    def _reinforce_tendency(self, tendency: Tendency) -> None:
        """Reinforce an existing tendency."""
        tendency.strength = min(
            self._config.max_strength,
            tendency.strength + self._config.strength_increment,
        )
        tendency.confidence = min(
            self._config.max_confidence,
            tendency.confidence + self._config.confidence_increment,
        )
        tendency.last_used_turn = self._state.turn_count
        tendency.total_reinforcements += 1
        tendency.consecutive_misses = 0

    def _apply_decay(self, used_scope: Optional[ScopedGoal]) -> None:
        """Apply decay to all tendencies."""
        used_pattern = None
        if used_scope is not None and used_scope.action_taken:
            used_pattern = self._extract_pattern(used_scope)

        for tendency in self._state.tendencies:
            # Check if this tendency was used
            was_used = (
                used_pattern is not None and
                self._patterns_similar(tendency.pattern, used_pattern)
            )

            if not was_used:
                # Apply decay
                decay = self._config.decay_rate

                # Extra decay for consecutive misses
                if tendency.consecutive_misses > 0:
                    decay *= self._config.miss_decay_multiplier

                tendency.strength = max(0.0, tendency.strength - decay)
                tendency.consecutive_misses += 1

    def _prune_faded_tendencies(self) -> None:
        """Remove tendencies that have faded to nothing."""
        original_count = len(self._state.tendencies)

        self._state.tendencies = [
            t for t in self._state.tendencies
            if t.strength > 0.001
        ]

        faded = original_count - len(self._state.tendencies)
        self._state.total_tendencies_faded += faded

    def get_bias(self) -> TendencyBias:
        """
        Get the current tendency bias for decision making.

        Returns a WEAK bias that creates a "slope" toward certain
        choices but never blocks any option.
        """
        if not self._state.tendencies:
            return TendencyBias(has_bias=False)

        # Aggregate biases by category
        category_biases: dict[str, float] = {}

        for tendency in self._state.tendencies:
            category = tendency.pattern.category.value

            # Calculate bias from strength and confidence
            bias = tendency.strength * tendency.confidence * self._config.bias_multiplier

            # Cap at max_bias
            bias = min(self._config.max_bias, bias)

            # Aggregate (take max for each category)
            if category not in category_biases:
                category_biases[category] = bias
            else:
                category_biases[category] = max(category_biases[category], bias)

        if not category_biases:
            return TendencyBias(has_bias=False)

        # Find strongest
        strongest_cat = max(category_biases, key=category_biases.get)
        strongest_bias = category_biases[strongest_cat]

        return TendencyBias(
            has_bias=True,
            biases=category_biases,
            strongest_category=CandidateCategory(strongest_cat),
            strongest_bias=strongest_bias,
        )

    def get_tendencies(self) -> list[Tendency]:
        """Get a copy of current tendencies (for observation)."""
        return [
            Tendency.from_dict(t.to_dict())
            for t in self._state.tendencies
        ]

    def get_tendency_for_category(
        self,
        category: CandidateCategory,
    ) -> Optional[Tendency]:
        """Get the strongest tendency for a specific category."""
        matching = [
            t for t in self._state.tendencies
            if t.pattern.category == category
        ]

        if not matching:
            return None

        return max(matching, key=lambda t: t.strength)

    def save_to_file(self, file_path: str) -> None:
        """Save current state to a JSON file (optional persistence)."""
        data = self._state.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str) -> int:
        """
        Load state from a JSON file.

        Returns:
            Number of tendencies loaded
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._state = RepeatedTendencyState.from_dict(data)
        self._config = self._state.config
        return len(self._state.tendencies)


def create_system(
    config: Optional[RepeatedTendencyConfig] = None,
) -> RepeatedTendencySystem:
    """Factory function to create a RepeatedTendencySystem."""
    return RepeatedTendencySystem(config=config)


def create_config(
    history_window: int = 50,
    min_repetitions: int = 3,
    recency_window: int = 20,
    strength_increment: float = 0.02,
    max_strength: float = 0.15,
    confidence_increment: float = 0.05,
    max_confidence: float = 0.8,
    decay_rate: float = 0.01,
    miss_decay_multiplier: float = 1.5,
    similarity_threshold: float = 0.7,
    bias_multiplier: float = 0.08,
    max_bias: float = 0.06,
) -> RepeatedTendencyConfig:
    """Factory function to create a RepeatedTendencyConfig."""
    return RepeatedTendencyConfig(
        history_window=history_window,
        min_repetitions=min_repetitions,
        recency_window=recency_window,
        strength_increment=strength_increment,
        max_strength=max_strength,
        confidence_increment=confidence_increment,
        max_confidence=max_confidence,
        decay_rate=decay_rate,
        miss_decay_multiplier=miss_decay_multiplier,
        similarity_threshold=similarity_threshold,
        bias_multiplier=bias_multiplier,
        max_bias=max_bias,
    )


def apply_tendency_bias_to_candidate(
    candidate: dict[str, Any],
    bias: TendencyBias,
) -> dict[str, Any]:
    """
    Apply tendency bias to a single decision candidate.

    This creates a "slope" - makes aligned choices slightly easier -
    but NEVER blocks or forces any choice.

    Args:
        candidate: A decision candidate with "score"
        bias: The tendency bias to apply

    Returns:
        Modified candidate (copy) with adjusted score
    """
    if not bias.has_bias:
        return candidate

    result = candidate.copy()
    original_score = result.get("score", 0.5)

    # Get candidate's alignment with tendencies
    alignment = _calculate_tendency_alignment(candidate, bias)

    # Apply very subtle bias
    # This is weaker than scoped_goal or transient_goal
    adjustment = alignment * bias.strongest_bias
    new_score = original_score * (1.0 + adjustment)

    # Keep in bounds
    new_score = max(0.0, min(1.0, new_score))

    result["score"] = new_score

    if abs(adjustment) > 0.001:
        result["_tendency_bias_applied"] = {
            "alignment": round(alignment, 3),
            "adjustment": round(adjustment, 4),
        }

    return result


def apply_tendency_bias_to_candidates(
    candidates: list[dict[str, Any]],
    bias: TendencyBias,
) -> list[dict[str, Any]]:
    """Apply tendency bias to multiple decision candidates."""
    return [apply_tendency_bias_to_candidate(c, bias) for c in candidates]


def _calculate_tendency_alignment(
    candidate: dict[str, Any],
    bias: TendencyBias,
) -> float:
    """
    Calculate how well a candidate aligns with formed tendencies.

    Returns value from -0.5 to 1.0 (asymmetric - habit helps more than hurts).
    """
    policy = candidate.get("policy", "")

    # Category-based alignment (keys must match thought.py POLICIES policy_label)
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

    alignment = 0.0
    policy_lower = policy.lower()

    for category, bias_value in bias.biases.items():
        try:
            cat = CandidateCategory(category)
        except ValueError:
            continue

        affinity_keywords = category_policy_affinity.get(cat, [])

        for keyword in affinity_keywords:
            if keyword in policy_lower:
                # Positive alignment scaled by bias strength
                alignment += bias_value / bias.strongest_bias if bias.strongest_bias > 0 else 0
                break

    # Normalize and cap
    alignment = min(1.0, max(-0.5, alignment))

    return alignment


def get_tendency_summary(system: RepeatedTendencySystem) -> dict[str, Any]:
    """
    Get a summary of the tendency system state.

    For observation and debugging purposes.
    """
    state = system.state
    tendencies = system.get_tendencies()

    category_strengths: dict[str, float] = {}
    for t in tendencies:
        cat = t.pattern.category.value
        if cat not in category_strengths:
            category_strengths[cat] = t.strength
        else:
            category_strengths[cat] = max(category_strengths[cat], t.strength)

    return {
        "tendency_count": len(tendencies),
        "total_formed": state.total_tendencies_formed,
        "total_faded": state.total_tendencies_faded,
        "turn_count": state.turn_count,
        "history_size": len(state.usage_history),
        "category_strengths": category_strengths,
        "strongest_tendency": max(
            ({"category": t.pattern.category.value, "strength": t.strength}
             for t in tendencies),
            key=lambda x: x["strength"],
            default=None,
        ),
        "tendencies": [
            {
                "id": t.tendency_id,
                "category": t.pattern.category.value,
                "strength": round(t.strength, 4),
                "confidence": round(t.confidence, 3),
                "reinforcements": t.total_reinforcements,
                "consecutive_misses": t.consecutive_misses,
            }
            for t in sorted(tendencies, key=lambda x: x.strength, reverse=True)
        ],
    }


# For introspection trace integration
def create_tendency_context_for_trace(
    system: RepeatedTendencySystem,
) -> Optional[dict[str, Any]]:
    """
    Create a context dict for introspection trace logging.

    Records what tendencies existed during the decision.
    """
    tendencies = system.get_tendencies()

    if not tendencies:
        return None

    bias = system.get_bias()

    return {
        "tendency_count": len(tendencies),
        "strongest_category": bias.strongest_category.value if bias.strongest_category else None,
        "strongest_bias": round(bias.strongest_bias, 4),
        "active_tendencies": [
            {
                "category": t.pattern.category.value,
                "strength": round(t.strength, 4),
            }
            for t in tendencies[:3]  # Top 3
        ],
        "observation_note": "These tendencies are habits/inertia, NOT personality. "
                          "They create a slight slope but never force decisions.",
    }


# For long-term dynamics integration
def create_tendency_stats_for_dynamics(
    system: RepeatedTendencySystem,
) -> dict[str, Any]:
    """
    Create stats for long-term dynamics logging.

    Returns aggregated statistics suitable for window-based observation.
    """
    state = system.state
    tendencies = system.get_tendencies()

    return {
        "tendency_count": len(tendencies),
        "total_strength": sum(t.strength for t in tendencies),
        "average_strength": (
            sum(t.strength for t in tendencies) / len(tendencies)
            if tendencies else 0.0
        ),
        "average_confidence": (
            sum(t.confidence for t in tendencies) / len(tendencies)
            if tendencies else 0.0
        ),
        "total_formed": state.total_tendencies_formed,
        "total_faded": state.total_tendencies_faded,
    }


def to_dict(config: RepeatedTendencyConfig) -> dict[str, Any]:
    """Convert config to dictionary."""
    return config.to_dict()


def from_dict(data: dict[str, Any]) -> RepeatedTendencyConfig:
    """Create config from dictionary."""
    return RepeatedTendencyConfig.from_dict(data)
