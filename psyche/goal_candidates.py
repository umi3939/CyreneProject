"""
goal_candidates.py - Proto-Goal → Goal Candidates (自発的目的候補生成)

This module projects ProtoGoalVectors into more concrete (but still abstract)
"Candidate Concepts" - daydreams or hypotheses that are never selected or acted upon.

Key design principles:
- Candidates are HYPOTHESES, not decisions
- Multiple conflicting candidates can coexist (e.g., "Be Quiet" AND "Speak Up")
- NO selection mechanism - this system generates options but NEVER chooses
- NO impact on decisions or actions - these are "daydreams" only
- Candidates decay and disappear if underlying vectors fade

【思想】
人は方向性を感じ始めたあと、
いくつかの「こうかもしれない」という目的像を思い浮かべる。
この段階では、まだ選ばず、比べず、行動もしない。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .proto_goal_vector import ProtoGoalVector, VectorGenerator


class CandidateCategory(Enum):
    """Abstract categories for goal candidates (no semantic meaning)."""
    APPROACH = "approach"       # Tendency toward something
    AVOIDANCE = "avoidance"     # Tendency away from something
    MAINTENANCE = "maintenance" # Tendency to preserve current state
    EXPLORATION = "exploration" # Tendency to seek novelty
    CONNECTION = "connection"   # Tendency toward others
    ISOLATION = "isolation"     # Tendency toward solitude
    EXPRESSION = "expression"   # Tendency to output/create
    ABSORPTION = "absorption"   # Tendency to receive/learn


# Shared affinity mapping: CandidateCategory -> policy labels (thought.py POLICIES)
CATEGORY_POLICY_AFFINITY: dict[CandidateCategory, list[str]] = {
    CandidateCategory.APPROACH: ["共感する", "励ます", "質問で会話を広げる", "提案する"],
    CandidateCategory.AVOIDANCE: ["黙って聞く", "見守る", "話題を変える", "確認する"],
    CandidateCategory.CONNECTION: ["共感する", "励ます", "質問で会話を広げる", "同意する"],
    CandidateCategory.ISOLATION: ["黙って聞く", "見守る", "話題を変える"],
    CandidateCategory.EXPRESSION: ["感想を述べる", "冗談を言う", "からかう", "自分の経験を話す"],
    CandidateCategory.ABSORPTION: ["黙って聞く", "見守る", "確認する"],
    CandidateCategory.EXPLORATION: ["質問で会話を広げる", "確認する", "提案する"],
    CandidateCategory.MAINTENANCE: ["同意する", "感想を述べる", "確認する"],
}


@dataclass
class CandidateSource:
    """
    Links a goal candidate to its source vectors.

    A candidate can emerge from multiple vectors, representing
    a convergence of different internal tendencies.
    """
    vector_ids: list[str] = field(default_factory=list)
    contribution_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_ids": self.vector_ids.copy(),
            "contribution_weights": self.contribution_weights.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSource:
        return cls(
            vector_ids=data.get("vector_ids", []),
            contribution_weights=data.get("contribution_weights", {}),
        )


@dataclass
class GoalCandidate:
    """
    A single goal candidate - a "daydream" or "hypothesis".

    NOT a goal, NOT a plan, NOT an intention.
    Just a projected possibility that exists without selection.

    Attributes:
        candidate_id: Unique identifier
        category: Abstract category (no semantic meaning)
        direction_expression: Abstract direction (dict of dimensions)
        intensity: How strongly this candidate is present (0.0 to 1.0)
        source: Links to source ProtoGoalVectors
        created_at: Timestamp of creation
        last_reinforced_at: Timestamp of last reinforcement
        reinforcement_count: Number of reinforcements
    """
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: CandidateCategory = CandidateCategory.EXPLORATION
    direction_expression: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    source: CandidateSource = field(default_factory=CandidateSource)
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    reinforcement_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "category": self.category.value,
            "direction_expression": self.direction_expression.copy(),
            "intensity": self.intensity,
            "source": self.source.to_dict(),
            "created_at": self.created_at,
            "last_reinforced_at": self.last_reinforced_at,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalCandidate:
        source_data = data.get("source", {})
        if isinstance(source_data, dict):
            source = CandidateSource.from_dict(source_data)
        else:
            source = CandidateSource()

        category_str = data.get("category", "exploration")
        try:
            category = CandidateCategory(category_str)
        except ValueError:
            category = CandidateCategory.EXPLORATION

        return cls(
            candidate_id=data.get("candidate_id", str(uuid.uuid4())[:8]),
            category=category,
            direction_expression=data.get("direction_expression", {}),
            intensity=data.get("intensity", 0.0),
            source=source,
            created_at=data.get("created_at", time.time()),
            last_reinforced_at=data.get("last_reinforced_at", time.time()),
            reinforcement_count=data.get("reinforcement_count", 1),
        )


@dataclass
class CandidateStateConfig:
    """
    Configuration for the GoalCandidate system.

    Attributes:
        max_candidates: Maximum number of candidates to maintain
        decay_rate: How quickly candidates decay per turn
        min_intensity: Minimum intensity before candidate is removed
        generation_probability: Base probability of generating candidates
        vector_threshold: Minimum vector magnitude to trigger generation
        similarity_threshold: How similar candidates must be to merge
        merge_enabled: Whether to merge similar candidates
        reinforcement_boost: Intensity boost on reinforcement
        auto_save_interval: Turns between auto-saves (0 = disabled)
    """
    max_candidates: int = 15
    decay_rate: float = 0.03
    min_intensity: float = 0.05
    generation_probability: float = 0.4
    vector_threshold: float = 0.3
    similarity_threshold: float = 0.8
    merge_enabled: bool = True
    reinforcement_boost: float = 0.15
    auto_save_interval: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "decay_rate": self.decay_rate,
            "min_intensity": self.min_intensity,
            "generation_probability": self.generation_probability,
            "vector_threshold": self.vector_threshold,
            "similarity_threshold": self.similarity_threshold,
            "merge_enabled": self.merge_enabled,
            "reinforcement_boost": self.reinforcement_boost,
            "auto_save_interval": self.auto_save_interval,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateStateConfig:
        return cls(
            max_candidates=data.get("max_candidates", 15),
            decay_rate=data.get("decay_rate", 0.03),
            min_intensity=data.get("min_intensity", 0.05),
            generation_probability=data.get("generation_probability", 0.4),
            vector_threshold=data.get("vector_threshold", 0.3),
            similarity_threshold=data.get("similarity_threshold", 0.8),
            merge_enabled=data.get("merge_enabled", True),
            reinforcement_boost=data.get("reinforcement_boost", 0.15),
            auto_save_interval=data.get("auto_save_interval", 50),
        )


@dataclass
class CandidateState:
    """
    Persistent state holding multiple goal candidates.

    These are "daydreams" - they exist but never influence decisions.
    """
    candidates: list[GoalCandidate] = field(default_factory=list)
    config: CandidateStateConfig = field(default_factory=CandidateStateConfig)
    turn_count: int = 0
    total_candidates_generated: int = 0
    total_candidates_decayed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "config": self.config.to_dict(),
            "turn_count": self.turn_count,
            "total_candidates_generated": self.total_candidates_generated,
            "total_candidates_decayed": self.total_candidates_decayed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateState:
        config_data = data.get("config", {})
        if isinstance(config_data, dict):
            config = CandidateStateConfig.from_dict(config_data)
        else:
            config = CandidateStateConfig()

        candidates = [
            GoalCandidate.from_dict(c)
            for c in data.get("candidates", [])
        ]

        return cls(
            candidates=candidates,
            config=config,
            turn_count=data.get("turn_count", 0),
            total_candidates_generated=data.get("total_candidates_generated", 0),
            total_candidates_decayed=data.get("total_candidates_decayed", 0),
        )


class CandidateGenerator:
    """
    Generates goal candidates from ProtoGoalVectors.

    This generator projects abstract direction vectors into more
    concrete (but still abstract) candidate concepts.

    IMPORTANT: This system NEVER selects candidates.
    Candidates are daydreams that exist without action.
    """

    def __init__(self, config: Optional[CandidateStateConfig] = None):
        self._config = config or CandidateStateConfig()
        self._state = CandidateState(config=self._config)
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._save_path: Optional[str] = None

    @property
    def state(self) -> CandidateState:
        """Read-only access to current state."""
        return self._state

    @property
    def config(self) -> CandidateStateConfig:
        """Read-only access to configuration."""
        return self._config

    def observe_vectors(
        self,
        vectors: list[ProtoGoalVector],
    ) -> list[GoalCandidate]:
        """
        Observe current vectors and potentially generate candidates.

        This method:
        1. Applies decay to existing candidates
        2. Removes candidates with faded source vectors
        3. Removes candidates below minimum intensity
        4. Potentially generates new candidates from vectors
        5. Potentially reinforces similar existing candidates

        Args:
            vectors: Current list of ProtoGoalVectors

        Returns:
            List of newly generated candidates (may be empty)
        """
        with self._lock:
            self._state.turn_count += 1

            # Step 1: Apply decay to all candidates
            self._apply_decay()

            # Step 2: Remove candidates whose source vectors have faded
            self._prune_orphaned_candidates(vectors)

            # Step 3: Remove candidates below minimum intensity
            self._prune_weak_candidates()

            # Step 4: Attempt to generate new candidates
            new_candidates = self._attempt_generation(vectors)

            # Step 5: Handle auto-save if configured
            if (self._config.auto_save_interval > 0 and
                self._state.turn_count % self._config.auto_save_interval == 0):
                self._trigger_auto_save()

            return new_candidates

    def _apply_decay(self) -> None:
        """Apply decay to all candidates."""
        for candidate in self._state.candidates:
            candidate.intensity *= (1.0 - self._config.decay_rate)

    def _prune_orphaned_candidates(self, vectors: list[ProtoGoalVector]) -> None:
        """Remove candidates whose source vectors no longer exist or are too weak."""
        active_vector_ids = {v.vector_id for v in vectors if v.magnitude >= self._config.vector_threshold}

        original_count = len(self._state.candidates)
        surviving = []

        for candidate in self._state.candidates:
            # Check if at least one source vector is still active
            has_active_source = any(
                vid in active_vector_ids
                for vid in candidate.source.vector_ids
            )
            if has_active_source:
                surviving.append(candidate)

        self._state.candidates = surviving
        removed = original_count - len(surviving)
        self._state.total_candidates_decayed += removed

    def _prune_weak_candidates(self) -> None:
        """Remove candidates below minimum intensity."""
        original_count = len(self._state.candidates)
        self._state.candidates = [
            c for c in self._state.candidates
            if c.intensity >= self._config.min_intensity
        ]
        removed = original_count - len(self._state.candidates)
        self._state.total_candidates_decayed += removed

    def _attempt_generation(
        self,
        vectors: list[ProtoGoalVector],
    ) -> list[GoalCandidate]:
        """
        Attempt to generate new candidates from vectors.

        Multiple candidates can be generated from a single vector,
        and candidates can emerge from multiple vectors.
        """
        import random

        new_candidates: list[GoalCandidate] = []

        # Filter vectors above threshold
        strong_vectors = [
            v for v in vectors
            if v.magnitude >= self._config.vector_threshold
        ]

        if not strong_vectors:
            return new_candidates

        # Probabilistic generation for each vector
        for vector in strong_vectors:
            if random.random() > self._config.generation_probability:
                continue

            # Generate one or more candidates from this vector
            num_candidates = self._determine_candidate_count(vector)

            for _ in range(num_candidates):
                candidate = self._create_candidate_from_vector(vector)

                # Check for similar existing candidate
                similar = self._find_similar_candidate(candidate)

                if similar and self._config.merge_enabled:
                    # Reinforce existing instead of creating new
                    self._reinforce_candidate(similar, candidate)
                else:
                    self._state.candidates.append(candidate)
                    self._state.total_candidates_generated += 1
                    new_candidates.append(candidate)

        # Try to generate from vector combinations
        if len(strong_vectors) >= 2 and random.random() < self._config.generation_probability * 0.5:
            combined = self._create_combined_candidate(strong_vectors[:3])
            if combined:
                similar = self._find_similar_candidate(combined)
                if similar and self._config.merge_enabled:
                    self._reinforce_candidate(similar, combined)
                else:
                    self._state.candidates.append(combined)
                    self._state.total_candidates_generated += 1
                    new_candidates.append(combined)

        # Enforce max candidates limit
        self._enforce_max_candidates()

        return new_candidates

    def _determine_candidate_count(self, vector: ProtoGoalVector) -> int:
        """Determine how many candidates to generate from a vector."""
        import random

        # Higher magnitude vectors can produce more candidates
        if vector.magnitude > 0.8:
            return random.choice([1, 1, 2])
        elif vector.magnitude > 0.5:
            return random.choice([1, 1, 1, 2])
        else:
            return 1

    def _create_candidate_from_vector(
        self,
        vector: ProtoGoalVector,
    ) -> GoalCandidate:
        """Create a new candidate from a single vector."""
        import random

        # Determine category based on vector direction
        category = self._infer_category(vector.direction)

        # Project direction into candidate expression
        expression = self._project_direction(vector.direction)

        # Intensity based on vector magnitude with some randomness
        intensity = vector.magnitude * random.uniform(0.7, 1.0)
        intensity = max(self._config.min_intensity, min(1.0, intensity))

        return GoalCandidate(
            category=category,
            direction_expression=expression,
            intensity=intensity,
            source=CandidateSource(
                vector_ids=[vector.vector_id],
                contribution_weights={vector.vector_id: 1.0},
            ),
        )

    def _create_combined_candidate(
        self,
        vectors: list[ProtoGoalVector],
    ) -> Optional[GoalCandidate]:
        """Create a candidate from multiple vectors."""
        if not vectors:
            return None

        import random

        # Combine directions
        combined_direction: dict[str, float] = {}
        total_magnitude = sum(v.magnitude for v in vectors)

        for vector in vectors:
            weight = vector.magnitude / total_magnitude if total_magnitude > 0 else 1.0 / len(vectors)
            for key, val in vector.direction.items():
                combined_direction[key] = combined_direction.get(key, 0.0) + val * weight

        # Infer category from combined direction
        category = self._infer_category(combined_direction)

        # Project into expression
        expression = self._project_direction(combined_direction)

        # Combined intensity
        intensity = (total_magnitude / len(vectors)) * random.uniform(0.6, 0.9)
        intensity = max(self._config.min_intensity, min(1.0, intensity))

        # Build source
        vector_ids = [v.vector_id for v in vectors]
        weights = {
            v.vector_id: v.magnitude / total_magnitude if total_magnitude > 0 else 1.0 / len(vectors)
            for v in vectors
        }

        return GoalCandidate(
            category=category,
            direction_expression=expression,
            intensity=intensity,
            source=CandidateSource(
                vector_ids=vector_ids,
                contribution_weights=weights,
            ),
        )

    def _infer_category(self, direction: dict[str, float]) -> CandidateCategory:
        """Infer a category from direction dimensions."""
        import random

        # Simple heuristic based on direction characteristics
        positive_sum = sum(v for v in direction.values() if v > 0)
        negative_sum = sum(v for v in direction.values() if v < 0)
        total = abs(positive_sum) + abs(negative_sum)

        if total < 0.01:
            return random.choice(list(CandidateCategory))

        positive_ratio = positive_sum / total if total > 0 else 0.5

        # Map to categories based on overall tendency
        categories_by_tendency = [
            (CandidateCategory.APPROACH, positive_ratio > 0.6),
            (CandidateCategory.AVOIDANCE, positive_ratio < 0.4),
            (CandidateCategory.EXPLORATION, 0.4 <= positive_ratio <= 0.6 and len(direction) > 2),
            (CandidateCategory.MAINTENANCE, len(direction) <= 1),
        ]

        matching = [cat for cat, matches in categories_by_tendency if matches]

        if matching:
            return random.choice(matching)

        # Check for specific dimension patterns
        has_emotion = any("emotion" in k for k in direction.keys())
        has_dim = any("dim_" in k for k in direction.keys())

        if has_emotion:
            return random.choice([
                CandidateCategory.EXPRESSION,
                CandidateCategory.CONNECTION,
            ])
        elif has_dim:
            return random.choice([
                CandidateCategory.EXPLORATION,
                CandidateCategory.ABSORPTION,
            ])

        return random.choice(list(CandidateCategory))

    def _project_direction(self, direction: dict[str, float]) -> dict[str, float]:
        """Project vector direction into candidate expression."""
        import random

        expression = {}

        for key, value in direction.items():
            # Add some noise to make candidates distinct
            noise = random.uniform(-0.1, 0.1)
            projected = value + noise
            projected = max(-1.0, min(1.0, projected))

            # Rename dimensions to candidate space
            if key.startswith("dim_"):
                new_key = f"goal_{key}"
            elif key.startswith("emotion_"):
                new_key = f"affective_{key[8:]}"
            else:
                new_key = f"abstract_{key}"

            expression[new_key] = projected

        return expression

    def _find_similar_candidate(
        self,
        candidate: GoalCandidate,
    ) -> Optional[GoalCandidate]:
        """Find an existing candidate with similar expression."""
        for existing in self._state.candidates:
            # Must be same category
            if existing.category != candidate.category:
                continue

            # Check direction similarity
            similarity = self._compute_similarity(
                existing.direction_expression,
                candidate.direction_expression,
            )
            if similarity >= self._config.similarity_threshold:
                return existing

        return None

    def _compute_similarity(
        self,
        expr1: dict[str, float],
        expr2: dict[str, float],
    ) -> float:
        """Compute similarity between two expressions."""
        all_keys = set(expr1.keys()) | set(expr2.keys())
        if not all_keys:
            return 0.0

        dot_product = sum(
            expr1.get(k, 0.0) * expr2.get(k, 0.0)
            for k in all_keys
        )

        mag1 = sum(v ** 2 for v in expr1.values()) ** 0.5
        mag2 = sum(v ** 2 for v in expr2.values()) ** 0.5

        if mag1 < 0.001 or mag2 < 0.001:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _reinforce_candidate(
        self,
        existing: GoalCandidate,
        new: GoalCandidate,
    ) -> None:
        """Reinforce an existing candidate with new signal."""
        # Blend expressions
        all_keys = set(existing.direction_expression.keys()) | set(new.direction_expression.keys())
        blended = {}
        for key in all_keys:
            old_val = existing.direction_expression.get(key, 0.0)
            new_val = new.direction_expression.get(key, 0.0)
            blended[key] = old_val * 0.7 + new_val * 0.3

        existing.direction_expression = blended

        # Boost intensity
        existing.intensity = min(
            1.0,
            existing.intensity + self._config.reinforcement_boost * new.intensity,
        )

        # Merge source vectors
        for vid in new.source.vector_ids:
            if vid not in existing.source.vector_ids:
                existing.source.vector_ids.append(vid)

        for vid, weight in new.source.contribution_weights.items():
            old_weight = existing.source.contribution_weights.get(vid, 0.0)
            existing.source.contribution_weights[vid] = max(old_weight, weight)

        existing.last_reinforced_at = time.time()
        existing.reinforcement_count += 1

    def _enforce_max_candidates(self) -> None:
        """Remove oldest candidates if we exceed max limit."""
        while len(self._state.candidates) > self._config.max_candidates:
            # Remove weakest (lowest intensity)
            weakest = min(self._state.candidates, key=lambda c: c.intensity)
            self._state.candidates.remove(weakest)
            self._state.total_candidates_decayed += 1

    def get_candidates(self) -> list[GoalCandidate]:
        """
        Get a READ-ONLY copy of current candidates.

        These candidates are "daydreams" - they can be observed
        but MUST NOT be used to select or influence decisions.
        """
        with self._lock:
            return [
                GoalCandidate.from_dict(c.to_dict())
                for c in self._state.candidates
            ]

    def get_candidate_by_id(self, candidate_id: str) -> Optional[GoalCandidate]:
        """Get a specific candidate by ID (read-only copy)."""
        with self._lock:
            for c in self._state.candidates:
                if c.candidate_id == candidate_id:
                    return GoalCandidate.from_dict(c.to_dict())
        return None

    def get_candidates_by_category(
        self,
        category: CandidateCategory,
    ) -> list[GoalCandidate]:
        """Get all candidates of a specific category (read-only copies)."""
        with self._lock:
            return [
                GoalCandidate.from_dict(c.to_dict())
                for c in self._state.candidates
                if c.category == category
            ]

    def get_strongest_candidates(self, n: int = 3) -> list[GoalCandidate]:
        """Get the N strongest candidates by intensity (read-only copies)."""
        with self._lock:
            sorted_candidates = sorted(
                self._state.candidates,
                key=lambda c: c.intensity,
                reverse=True,
            )
            return [
                GoalCandidate.from_dict(c.to_dict())
                for c in sorted_candidates[:n]
            ]

    def get_conflicting_pairs(self) -> list[tuple[GoalCandidate, GoalCandidate]]:
        """
        Find pairs of candidates that represent conflicting tendencies.

        This is purely observational - conflicts are allowed and expected.
        No resolution is performed or suggested.
        """
        with self._lock:
            # Define opposing category pairs
            opposing_pairs = [
                (CandidateCategory.APPROACH, CandidateCategory.AVOIDANCE),
                (CandidateCategory.CONNECTION, CandidateCategory.ISOLATION),
                (CandidateCategory.EXPRESSION, CandidateCategory.ABSORPTION),
                (CandidateCategory.EXPLORATION, CandidateCategory.MAINTENANCE),
            ]

            conflicts = []
            for cat1, cat2 in opposing_pairs:
                candidates1 = [c for c in self._state.candidates if c.category == cat1]
                candidates2 = [c for c in self._state.candidates if c.category == cat2]

                for c1 in candidates1:
                    for c2 in candidates2:
                        conflicts.append((
                            GoalCandidate.from_dict(c1.to_dict()),
                            GoalCandidate.from_dict(c2.to_dict()),
                        ))

            return conflicts

    def save_to_file(self, file_path: str) -> None:
        """Save current state to a JSON file."""
        with self._lock:
            data = self._state.to_dict()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._save_path = file_path

    def load_from_file(self, file_path: str) -> int:
        """
        Load state from a JSON file.

        Returns:
            Number of candidates loaded
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._state = CandidateState.from_dict(data)
            self._config = self._state.config
            self._save_path = file_path
            return len(self._state.candidates)

    def _trigger_auto_save(self) -> None:
        """Trigger async auto-save if path is configured."""
        if self._save_path is None:
            return

        if self._save_thread is not None and self._save_thread.is_alive():
            return

        data = self._state.to_dict()
        path = self._save_path

        def save():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        self._save_thread = threading.Thread(target=save, daemon=True)
        self._save_thread.start()

    def set_save_path(self, file_path: str) -> None:
        """Set the path for auto-saves."""
        self._save_path = file_path


def create_candidate_generator(
    config: Optional[CandidateStateConfig] = None,
) -> CandidateGenerator:
    """Factory function to create a CandidateGenerator."""
    return CandidateGenerator(config=config)


def create_config(
    max_candidates: int = 15,
    decay_rate: float = 0.03,
    min_intensity: float = 0.05,
    generation_probability: float = 0.4,
    vector_threshold: float = 0.3,
    similarity_threshold: float = 0.8,
    merge_enabled: bool = True,
    reinforcement_boost: float = 0.15,
    auto_save_interval: int = 50,
) -> CandidateStateConfig:
    """Factory function to create a CandidateStateConfig."""
    return CandidateStateConfig(
        max_candidates=max_candidates,
        decay_rate=decay_rate,
        min_intensity=min_intensity,
        generation_probability=generation_probability,
        vector_threshold=vector_threshold,
        similarity_threshold=similarity_threshold,
        merge_enabled=merge_enabled,
        reinforcement_boost=reinforcement_boost,
        auto_save_interval=auto_save_interval,
    )


def get_candidate_summary(generator: CandidateGenerator) -> dict[str, Any]:
    """
    Get a summary of the current candidate state.

    This is for observation/debugging purposes only.
    The summary MUST NOT be used to select or prioritize candidates.
    """
    state = generator.state
    candidates = generator.get_candidates()

    category_counts: dict[str, int] = {}
    for c in candidates:
        cat = c.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "candidate_count": len(candidates),
        "total_generated": state.total_candidates_generated,
        "total_decayed": state.total_candidates_decayed,
        "turn_count": state.turn_count,
        "strongest_intensity": max((c.intensity for c in candidates), default=0.0),
        "average_intensity": (
            sum(c.intensity for c in candidates) / len(candidates)
            if candidates else 0.0
        ),
        "category_distribution": category_counts,
        "conflict_count": len(generator.get_conflicting_pairs()),
        "candidates": [
            {
                "id": c.candidate_id,
                "category": c.category.value,
                "intensity": round(c.intensity, 3),
                "source_count": len(c.source.vector_ids),
                "reinforcements": c.reinforcement_count,
            }
            for c in sorted(candidates, key=lambda x: x.intensity, reverse=True)
        ],
    }


def candidates_to_json(candidates: list[GoalCandidate]) -> str:
    """Serialize candidates to JSON string."""
    return json.dumps(
        [c.to_dict() for c in candidates],
        indent=2,
        ensure_ascii=False,
    )


def candidates_from_json(json_str: str) -> list[GoalCandidate]:
    """Deserialize candidates from JSON string."""
    data = json.loads(json_str)
    return [GoalCandidate.from_dict(c) for c in data]


def to_dict(config: CandidateStateConfig) -> dict[str, Any]:
    """Convert config to dictionary."""
    return config.to_dict()


def from_dict(data: dict[str, Any]) -> CandidateStateConfig:
    """Create config from dictionary."""
    return CandidateStateConfig.from_dict(data)


# For introspection trace integration (read-only observation)
def create_candidate_context_for_trace(
    generator: CandidateGenerator,
) -> Optional[dict[str, Any]]:
    """
    Create a context dict for introspection trace logging.

    This allows the introspection system to note which goal
    candidates existed at a given moment, without implying
    any selection or preference.

    The context is purely informational "daydream" data.
    """
    candidates = generator.get_strongest_candidates(5)

    if not candidates:
        return None

    return {
        "goal_candidates": [
            {
                "candidate_id": c.candidate_id,
                "category": c.category.value,
                "intensity": round(c.intensity, 3),
                "source_vector_count": len(c.source.vector_ids),
            }
            for c in candidates
        ],
        "conflict_count": len(generator.get_conflicting_pairs()),
        "observation_note": "These candidates are daydreams/hypotheses. "
                          "They are NOT selected, prioritized, or acted upon.",
    }


# For long-term dynamics integration (observation only)
def create_candidate_stats_for_dynamics(
    generator: CandidateGenerator,
) -> dict[str, Any]:
    """
    Create stats for long-term dynamics logging.

    Returns aggregated statistics suitable for window-based
    observation without identifying specific candidates.
    """
    candidates = generator.get_candidates()

    category_counts = {}
    category_intensities: dict[str, list[float]] = {}

    for c in candidates:
        cat = c.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if cat not in category_intensities:
            category_intensities[cat] = []
        category_intensities[cat].append(c.intensity)

    return {
        "total_count": len(candidates),
        "category_counts": category_counts,
        "category_avg_intensity": {
            cat: sum(vals) / len(vals)
            for cat, vals in category_intensities.items()
            if vals
        },
        "conflict_count": len(generator.get_conflicting_pairs()),
        "max_intensity": max((c.intensity for c in candidates), default=0.0),
    }
