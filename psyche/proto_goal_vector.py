"""
proto_goal_vector.py - Proto-Goal Direction Vector (自発的方向ベクトル生成)

This module implements abstract "direction vectors" that represent vague
tendencies like "I feel like going this way" - NOT goals, intentions, or tasks.

Key design principles:
- Vectors DO NOT influence decisions, emotions, or responsibility
- Vectors are "ghost data" - observed but never acting
- No achievement/failure/evaluation concepts
- Multiple vectors can coexist simultaneously
- Vectors naturally decay if not reinforced
- Persistence across sessions with decay allowance

【思想】
人は突然「目的」を持つのではなく、
価値・感情・経験の積み重なりから
「こうなりたい気がする」「この方向が落ち着く」
という曖昧な方向性を感じ始める。
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class VectorSourceType(Enum):
    """Source category for vector generation."""
    VALUE_ORIENTATION = "value_orientation"
    INTROSPECTION_PATTERN = "introspection_pattern"
    RESPONSIBILITY_PATTERN = "responsibility_pattern"
    EMOTION_TENDENCY = "emotion_tendency"
    COMBINED = "combined"


@dataclass
class VectorSource:
    """
    Origin information for a proto-goal vector.

    Records where this vector emerged from without
    implying causation or justification.
    """
    source_type: VectorSourceType
    reference_id: str = ""  # Optional ID for tracing (e.g., trace_id)
    description: str = ""   # Human-readable note (for debugging only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "reference_id": self.reference_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorSource:
        return cls(
            source_type=VectorSourceType(data.get("source_type", "combined")),
            reference_id=data.get("reference_id", ""),
            description=data.get("description", ""),
        )


@dataclass
class ProtoGoalVector:
    """
    A single abstract direction vector.

    NOT a goal, NOT an intention, NOT a task.
    Just a direction with magnitude that represents
    an accumulated internal tendency.

    Attributes:
        vector_id: Unique identifier for this vector
        direction: Abstract direction in N-dimensional space (no semantic meaning)
        magnitude: Strength of this tendency (0.0 to 1.0)
        source: Where this vector emerged from
        created_at: Timestamp when this vector was created
        last_reinforced_at: Timestamp when this vector was last reinforced
        reinforcement_count: Number of times this vector has been reinforced
    """
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    direction: dict[str, float] = field(default_factory=dict)
    magnitude: float = 0.0
    source: VectorSource = field(default_factory=lambda: VectorSource(VectorSourceType.COMBINED))
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    reinforcement_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "direction": self.direction.copy(),
            "magnitude": self.magnitude,
            "source": self.source.to_dict(),
            "created_at": self.created_at,
            "last_reinforced_at": self.last_reinforced_at,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtoGoalVector:
        source_data = data.get("source", {})
        if isinstance(source_data, dict):
            source = VectorSource.from_dict(source_data)
        else:
            source = VectorSource(VectorSourceType.COMBINED)

        return cls(
            vector_id=data.get("vector_id", str(uuid.uuid4())[:8]),
            direction=data.get("direction", {}),
            magnitude=data.get("magnitude", 0.0),
            source=source,
            created_at=data.get("created_at", time.time()),
            last_reinforced_at=data.get("last_reinforced_at", time.time()),
            reinforcement_count=data.get("reinforcement_count", 1),
        )


@dataclass
class VectorStateConfig:
    """
    Configuration for the VectorState system.

    Attributes:
        max_vectors: Maximum number of vectors to store (oldest removed first)
        decay_rate: How quickly vectors decay per turn (0.0 to 1.0)
        min_magnitude: Minimum magnitude before vector is removed
        similarity_threshold: How similar directions must be to merge
        merge_enabled: Whether to merge similar vectors
        generation_probability: Base probability of generating a new vector
        reinforcement_boost: How much magnitude increases on reinforcement
        auto_save_interval: Turns between auto-saves (0 = disabled)
    """
    max_vectors: int = 10
    decay_rate: float = 0.02
    min_magnitude: float = 0.05
    similarity_threshold: float = 0.85
    merge_enabled: bool = True
    generation_probability: float = 0.3
    reinforcement_boost: float = 0.1
    auto_save_interval: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_vectors": self.max_vectors,
            "decay_rate": self.decay_rate,
            "min_magnitude": self.min_magnitude,
            "similarity_threshold": self.similarity_threshold,
            "merge_enabled": self.merge_enabled,
            "generation_probability": self.generation_probability,
            "reinforcement_boost": self.reinforcement_boost,
            "auto_save_interval": self.auto_save_interval,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorStateConfig:
        return cls(
            max_vectors=data.get("max_vectors", 10),
            decay_rate=data.get("decay_rate", 0.02),
            min_magnitude=data.get("min_magnitude", 0.05),
            similarity_threshold=data.get("similarity_threshold", 0.85),
            merge_enabled=data.get("merge_enabled", True),
            generation_probability=data.get("generation_probability", 0.3),
            reinforcement_boost=data.get("reinforcement_boost", 0.1),
            auto_save_interval=data.get("auto_save_interval", 50),
        )


@dataclass
class VectorState:
    """
    Persistent state holding multiple proto-goal vectors.

    This is the "ghost data" container - it holds vectors
    that represent internal tendencies but NEVER influence
    decisions, emotions, or responsibility.
    """
    vectors: list[ProtoGoalVector] = field(default_factory=list)
    config: VectorStateConfig = field(default_factory=VectorStateConfig)
    turn_count: int = 0
    total_vectors_generated: int = 0
    total_vectors_decayed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vectors": [v.to_dict() for v in self.vectors],
            "config": self.config.to_dict(),
            "turn_count": self.turn_count,
            "total_vectors_generated": self.total_vectors_generated,
            "total_vectors_decayed": self.total_vectors_decayed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorState:
        config_data = data.get("config", {})
        if isinstance(config_data, dict):
            config = VectorStateConfig.from_dict(config_data)
        else:
            config = VectorStateConfig()

        vectors = [
            ProtoGoalVector.from_dict(v)
            for v in data.get("vectors", [])
        ]

        return cls(
            vectors=vectors,
            config=config,
            turn_count=data.get("turn_count", 0),
            total_vectors_generated=data.get("total_vectors_generated", 0),
            total_vectors_decayed=data.get("total_vectors_decayed", 0),
        )


class VectorGenerator:
    """
    Generates proto-goal vectors from accumulated internal states.

    This generator observes value orientation, introspection patterns,
    and responsibility patterns to produce abstract direction vectors.

    IMPORTANT: Generation is probabilistic and does NOT guarantee
    any specific outcome. Vectors are tendencies, not goals.
    """

    def __init__(self, config: Optional[VectorStateConfig] = None):
        self._config = config or VectorStateConfig()
        self._state = VectorState(config=self._config)
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._save_path: Optional[str] = None

    @property
    def state(self) -> VectorState:
        """Read-only access to current state."""
        return self._state

    @property
    def config(self) -> VectorStateConfig:
        """Read-only access to configuration."""
        return self._config

    def observe_turn(
        self,
        value_orientation: Optional[Any] = None,
        introspection_trace: Optional[Any] = None,
        responsibility_pattern: Optional[dict[str, Any]] = None,
        emotion_tendency: Optional[dict[str, float]] = None,
    ) -> Optional[ProtoGoalVector]:
        """
        Observe a turn and potentially generate a new vector.

        This method:
        1. Applies decay to existing vectors
        2. Removes vectors below minimum magnitude
        3. Potentially generates a new vector based on inputs
        4. Potentially reinforces existing similar vectors

        Args:
            value_orientation: Current ValueOrientation state
            introspection_trace: Recent TraceLog or pattern summary
            responsibility_pattern: Summary of responsibility trends
            emotion_tendency: Recent emotion distribution/bias

        Returns:
            New ProtoGoalVector if one was generated, None otherwise
        """
        with self._lock:
            self._state.turn_count += 1

            # Step 1: Apply decay to all vectors
            self._apply_decay()

            # Step 2: Remove vectors below minimum magnitude
            self._prune_weak_vectors()

            # Step 3: Attempt to generate a new vector
            new_vector = self._attempt_generation(
                value_orientation=value_orientation,
                introspection_trace=introspection_trace,
                responsibility_pattern=responsibility_pattern,
                emotion_tendency=emotion_tendency,
            )

            # Step 4: Handle auto-save if configured
            if (self._config.auto_save_interval > 0 and
                self._state.turn_count % self._config.auto_save_interval == 0):
                self._trigger_auto_save()

            return new_vector

    def _apply_decay(self) -> None:
        """Apply decay to all vectors."""
        for vector in self._state.vectors:
            vector.magnitude *= (1.0 - self._config.decay_rate)

    def _prune_weak_vectors(self) -> None:
        """Remove vectors below minimum magnitude."""
        original_count = len(self._state.vectors)
        self._state.vectors = [
            v for v in self._state.vectors
            if v.magnitude >= self._config.min_magnitude
        ]
        removed = original_count - len(self._state.vectors)
        self._state.total_vectors_decayed += removed

    def _attempt_generation(
        self,
        value_orientation: Optional[Any] = None,
        introspection_trace: Optional[Any] = None,
        responsibility_pattern: Optional[dict[str, Any]] = None,
        emotion_tendency: Optional[dict[str, float]] = None,
    ) -> Optional[ProtoGoalVector]:
        """
        Attempt to generate a new vector from inputs.

        Generation is probabilistic and based on the strength
        of signals from various sources.
        """
        # Collect potential direction signals
        signals: list[tuple[dict[str, float], VectorSourceType, str]] = []

        # Signal from value orientation
        if value_orientation is not None:
            signal = self._extract_value_signal(value_orientation)
            if signal:
                signals.append((
                    signal,
                    VectorSourceType.VALUE_ORIENTATION,
                    getattr(value_orientation, "update_count", ""),
                ))

        # Signal from introspection trace
        if introspection_trace is not None:
            signal = self._extract_introspection_signal(introspection_trace)
            if signal:
                ref_id = ""
                if hasattr(introspection_trace, "trace_id"):
                    ref_id = introspection_trace.trace_id
                signals.append((
                    signal,
                    VectorSourceType.INTROSPECTION_PATTERN,
                    ref_id,
                ))

        # Signal from responsibility pattern
        if responsibility_pattern is not None:
            signal = self._extract_responsibility_signal(responsibility_pattern)
            if signal:
                signals.append((
                    signal,
                    VectorSourceType.RESPONSIBILITY_PATTERN,
                    responsibility_pattern.get("pattern_id", ""),
                ))

        # Signal from emotion tendency
        if emotion_tendency is not None:
            signal = self._extract_emotion_signal(emotion_tendency)
            if signal:
                signals.append((
                    signal,
                    VectorSourceType.EMOTION_TENDENCY,
                    "",
                ))

        if not signals:
            return None

        # Probabilistic generation check
        if random.random() > self._config.generation_probability:
            return None

        # Select a signal (weighted by magnitude)
        total_magnitude = sum(
            sum(abs(v) for v in s[0].values())
            for s in signals
        )
        if total_magnitude < 0.01:
            return None

        # Pick a random signal weighted by magnitude
        r = random.random() * total_magnitude
        cumulative = 0.0
        selected_signal = signals[0]
        for signal_data in signals:
            signal_mag = sum(abs(v) for v in signal_data[0].values())
            cumulative += signal_mag
            if cumulative >= r:
                selected_signal = signal_data
                break

        direction, source_type, ref_id = selected_signal

        # Calculate magnitude from signal strength
        magnitude = min(1.0, sum(abs(v) for v in direction.values()) / len(direction))
        magnitude = max(self._config.min_magnitude, magnitude)

        # Check for similar existing vectors
        similar_vector = self._find_similar_vector(direction)

        if similar_vector and self._config.merge_enabled:
            # Reinforce existing vector instead of creating new
            self._reinforce_vector(similar_vector, direction, magnitude)
            return None

        # Create new vector
        new_vector = ProtoGoalVector(
            direction=direction,
            magnitude=magnitude,
            source=VectorSource(
                source_type=source_type,
                reference_id=str(ref_id),
            ),
        )

        self._state.vectors.append(new_vector)
        self._state.total_vectors_generated += 1

        # Enforce max vectors limit
        self._enforce_max_vectors()

        return new_vector

    def _extract_value_signal(self, orientation: Any) -> Optional[dict[str, float]]:
        """Extract direction signal from value orientation."""
        signal = {}

        # Check for dimension attributes
        for dim in ["dim_a", "dim_b", "dim_c", "dim_d", "dim_e"]:
            if hasattr(orientation, dim):
                val = getattr(orientation, dim, 0.0)
                if abs(val) > 0.2:  # Only include significant biases
                    signal[dim] = val

        return signal if signal else None

    def _extract_introspection_signal(self, trace: Any) -> Optional[dict[str, float]]:
        """Extract direction signal from introspection trace."""
        signal = {}

        # Check for emotion snapshot
        if hasattr(trace, "emotion_snapshot"):
            snapshot = trace.emotion_snapshot
            if hasattr(snapshot, "dominant_emotion") and snapshot.dominant_emotion:
                signal[f"emotion_{snapshot.dominant_emotion}"] = getattr(
                    snapshot, "intensity", 0.5
                )

        # Check for contributing factors
        if hasattr(trace, "contributing_factors"):
            for factor in trace.contributing_factors[:3]:  # Limit to top 3
                if hasattr(factor, "category") and hasattr(factor, "weight"):
                    cat = factor.category
                    if hasattr(cat, "value"):
                        cat = cat.value
                    signal[f"factor_{cat}"] = factor.weight

        return signal if signal else None

    def _extract_responsibility_signal(
        self, pattern: dict[str, Any]
    ) -> Optional[dict[str, float]]:
        """Extract direction signal from responsibility pattern."""
        signal = {}

        # Look for common pattern keys
        for key in ["sublimation_tendency", "dispersion_bias", "distance_preference"]:
            if key in pattern:
                val = pattern[key]
                if isinstance(val, (int, float)) and abs(val) > 0.1:
                    signal[key] = float(val)

        return signal if signal else None

    def _extract_emotion_signal(
        self, tendency: dict[str, float]
    ) -> Optional[dict[str, float]]:
        """Extract direction signal from emotion tendency."""
        signal = {}

        for emotion, value in tendency.items():
            if abs(value) > 0.2:  # Only include significant tendencies
                signal[f"emotion_{emotion}"] = value

        return signal if signal else None

    def _find_similar_vector(self, direction: dict[str, float]) -> Optional[ProtoGoalVector]:
        """Find an existing vector with similar direction."""
        for vector in self._state.vectors:
            similarity = self._compute_similarity(vector.direction, direction)
            if similarity >= self._config.similarity_threshold:
                return vector
        return None

    def _compute_similarity(
        self, dir1: dict[str, float], dir2: dict[str, float]
    ) -> float:
        """Compute cosine similarity between two direction vectors."""
        all_keys = set(dir1.keys()) | set(dir2.keys())
        if not all_keys:
            return 0.0

        dot_product = sum(
            dir1.get(k, 0.0) * dir2.get(k, 0.0)
            for k in all_keys
        )

        mag1 = sum(v ** 2 for v in dir1.values()) ** 0.5
        mag2 = sum(v ** 2 for v in dir2.values()) ** 0.5

        if mag1 < 0.001 or mag2 < 0.001:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _reinforce_vector(
        self,
        vector: ProtoGoalVector,
        new_direction: dict[str, float],
        new_magnitude: float,
    ) -> None:
        """Reinforce an existing vector with new signal."""
        # Blend directions
        all_keys = set(vector.direction.keys()) | set(new_direction.keys())
        blended = {}
        for key in all_keys:
            old_val = vector.direction.get(key, 0.0)
            new_val = new_direction.get(key, 0.0)
            # Weight toward existing direction
            blended[key] = old_val * 0.7 + new_val * 0.3

        vector.direction = blended

        # Boost magnitude
        vector.magnitude = min(
            1.0,
            vector.magnitude + self._config.reinforcement_boost * new_magnitude
        )

        vector.last_reinforced_at = time.time()
        vector.reinforcement_count += 1

    def _enforce_max_vectors(self) -> None:
        """Remove oldest vectors if we exceed max limit."""
        while len(self._state.vectors) > self._config.max_vectors:
            # Remove oldest (first created)
            oldest = min(self._state.vectors, key=lambda v: v.created_at)
            self._state.vectors.remove(oldest)
            self._state.total_vectors_decayed += 1

    def get_vectors(self) -> list[ProtoGoalVector]:
        """
        Get a READ-ONLY copy of current vectors.

        These vectors are "ghost data" - they can be observed
        but MUST NOT be used to influence decisions.
        """
        with self._lock:
            return [
                ProtoGoalVector.from_dict(v.to_dict())
                for v in self._state.vectors
            ]

    def get_vector_by_id(self, vector_id: str) -> Optional[ProtoGoalVector]:
        """Get a specific vector by ID (read-only copy)."""
        with self._lock:
            for v in self._state.vectors:
                if v.vector_id == vector_id:
                    return ProtoGoalVector.from_dict(v.to_dict())
        return None

    def get_strongest_vectors(self, n: int = 3) -> list[ProtoGoalVector]:
        """Get the N strongest vectors by magnitude (read-only copies)."""
        with self._lock:
            sorted_vectors = sorted(
                self._state.vectors,
                key=lambda v: v.magnitude,
                reverse=True,
            )
            return [
                ProtoGoalVector.from_dict(v.to_dict())
                for v in sorted_vectors[:n]
            ]

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
            Number of vectors loaded
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._state = VectorState.from_dict(data)
            self._config = self._state.config
            self._save_path = file_path
            return len(self._state.vectors)

    def _trigger_auto_save(self) -> None:
        """Trigger async auto-save if path is configured."""
        if self._save_path is None:
            return

        # Check if a save is already in progress
        if self._save_thread is not None and self._save_thread.is_alive():
            return

        # Copy data for thread-safe saving
        data = self._state.to_dict()
        path = self._save_path

        def save():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Silent fail for auto-save

        self._save_thread = threading.Thread(target=save, daemon=True)
        self._save_thread.start()

    def set_save_path(self, file_path: str) -> None:
        """Set the path for auto-saves."""
        self._save_path = file_path


def create_vector_generator(
    config: Optional[VectorStateConfig] = None,
) -> VectorGenerator:
    """Factory function to create a VectorGenerator."""
    return VectorGenerator(config=config)


def create_config(
    max_vectors: int = 10,
    decay_rate: float = 0.02,
    min_magnitude: float = 0.05,
    similarity_threshold: float = 0.85,
    merge_enabled: bool = True,
    generation_probability: float = 0.3,
    reinforcement_boost: float = 0.1,
    auto_save_interval: int = 50,
) -> VectorStateConfig:
    """Factory function to create a VectorStateConfig."""
    return VectorStateConfig(
        max_vectors=max_vectors,
        decay_rate=decay_rate,
        min_magnitude=min_magnitude,
        similarity_threshold=similarity_threshold,
        merge_enabled=merge_enabled,
        generation_probability=generation_probability,
        reinforcement_boost=reinforcement_boost,
        auto_save_interval=auto_save_interval,
    )


def get_vector_summary(generator: VectorGenerator) -> dict[str, Any]:
    """
    Get a summary of the current vector state.

    This is for observation/debugging purposes only.
    The summary MUST NOT be used to influence decisions.
    """
    state = generator.state
    vectors = generator.get_vectors()

    return {
        "vector_count": len(vectors),
        "total_generated": state.total_vectors_generated,
        "total_decayed": state.total_vectors_decayed,
        "turn_count": state.turn_count,
        "strongest_magnitude": max((v.magnitude for v in vectors), default=0.0),
        "average_magnitude": (
            sum(v.magnitude for v in vectors) / len(vectors)
            if vectors else 0.0
        ),
        "source_distribution": _count_sources(vectors),
        "vectors": [
            {
                "id": v.vector_id,
                "magnitude": round(v.magnitude, 3),
                "source": v.source.source_type.value,
                "reinforcements": v.reinforcement_count,
            }
            for v in sorted(vectors, key=lambda x: x.magnitude, reverse=True)
        ],
    }


def _count_sources(vectors: list[ProtoGoalVector]) -> dict[str, int]:
    """Count vectors by source type."""
    counts: dict[str, int] = {}
    for v in vectors:
        source = v.source.source_type.value
        counts[source] = counts.get(source, 0) + 1
    return counts


def vectors_to_json(vectors: list[ProtoGoalVector]) -> str:
    """Serialize vectors to JSON string."""
    return json.dumps(
        [v.to_dict() for v in vectors],
        indent=2,
        ensure_ascii=False,
    )


def vectors_from_json(json_str: str) -> list[ProtoGoalVector]:
    """Deserialize vectors from JSON string."""
    data = json.loads(json_str)
    return [ProtoGoalVector.from_dict(v) for v in data]


def to_dict(config: VectorStateConfig) -> dict[str, Any]:
    """Convert config to dictionary."""
    return config.to_dict()


def from_dict(data: dict[str, Any]) -> VectorStateConfig:
    """Create config from dictionary."""
    return VectorStateConfig.from_dict(data)


# For introspection trace integration (read-only observation)
def create_vector_context_for_trace(
    generator: VectorGenerator,
) -> Optional[dict[str, Any]]:
    """
    Create a context dict for introspection trace logging.

    This allows the introspection system to note that certain
    direction vectors exist, but does NOT imply any influence.

    The context is purely informational "ghost data".
    """
    vectors = generator.get_strongest_vectors(3)

    if not vectors:
        return None

    return {
        "proto_goal_vectors": [
            {
                "vector_id": v.vector_id,
                "magnitude": round(v.magnitude, 3),
                "source_type": v.source.source_type.value,
                "direction_dimensions": len(v.direction),
            }
            for v in vectors
        ],
        "observation_note": "These vectors represent accumulated tendencies. "
                          "They do NOT influence decisions or carry responsibility.",
    }
