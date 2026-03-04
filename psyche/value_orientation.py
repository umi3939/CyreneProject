"""
psyche/value_orientation.py - Consistent Value Orientation (一貫した価値軸)

Implements a persistent value orientation state that subtly biases
decision candidates without dictating outcomes.

Design principles (from design_value_orientation.md):
- Value orientation is a "weight bias" not moral rules
- Uses abstract dimensions (not named morals like "Justice")
- High inertia: changes are gradual and cumulative
- Does NOT force any specific decision
- Persists across sessions for continuity

Usage::

    from psyche.value_orientation import (
        ValueOrientation,
        ValueOrientationConfig,
        update_orientation,
        apply_orientation_to_candidates,
    )

    # Create or load orientation
    orientation = ValueOrientation()

    # Update based on decisions/states (very gradual)
    orientation = update_orientation(
        orientation,
        decision_signal={"dim_a": 0.1, "dim_b": -0.05},
    )

    # Apply as subtle bias to candidates
    adjusted = apply_orientation_to_candidates(candidates, orientation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time
import math

from . import coefficient_registry


# ── Value Orientation State ─────────────────────────────────────────


@dataclass
class ValueOrientation:
    """
    Persistent value orientation state.

    Represents tendencies in decision-making that persist across sessions.
    Uses abstract dimensions (A, B, C, D, E) without fixed meanings.

    Each dimension is in range [-1.0, 1.0]:
    - Negative: tendency toward one pole
    - Zero: neutral/balanced
    - Positive: tendency toward opposite pole

    The dimensions have NO hardcoded meaning. They are abstract axes
    that emerge from patterns of decisions and experiences.
    """

    # Abstract value dimensions (no fixed meanings)
    # Each in range [-1.0, 1.0]
    dim_a: float = 0.0  # e.g., could emerge as cautious vs bold
    dim_b: float = 0.0  # e.g., could emerge as individual vs collective
    dim_c: float = 0.0  # e.g., could emerge as immediate vs long-term
    dim_d: float = 0.0  # e.g., could emerge as concrete vs abstract
    dim_e: float = 0.0  # e.g., could emerge as stability vs change

    # Confidence/stability of each dimension (how established it is)
    # Higher = more resistant to change
    confidence_a: float = 0.0
    confidence_b: float = 0.0
    confidence_c: float = 0.0
    confidence_d: float = 0.0
    confidence_e: float = 0.0

    # Total update count (for tracking how developed the orientation is)
    update_count: int = 0

    # Timestamp of last update
    last_update: float = field(default_factory=time.time)

    # Version for schema evolution
    version: int = 1

    def __post_init__(self):
        """Clamp all values to valid ranges."""
        self.dim_a = max(-1.0, min(1.0, self.dim_a))
        self.dim_b = max(-1.0, min(1.0, self.dim_b))
        self.dim_c = max(-1.0, min(1.0, self.dim_c))
        self.dim_d = max(-1.0, min(1.0, self.dim_d))
        self.dim_e = max(-1.0, min(1.0, self.dim_e))

        self.confidence_a = max(0.0, min(1.0, self.confidence_a))
        self.confidence_b = max(0.0, min(1.0, self.confidence_b))
        self.confidence_c = max(0.0, min(1.0, self.confidence_c))
        self.confidence_d = max(0.0, min(1.0, self.confidence_d))
        self.confidence_e = max(0.0, min(1.0, self.confidence_e))

    def get_dimension(self, name: str) -> float:
        """Get dimension value by name."""
        return getattr(self, f"dim_{name}", 0.0)

    def get_confidence(self, name: str) -> float:
        """Get confidence value by name."""
        return getattr(self, f"confidence_{name}", 0.0)

    def get_all_dimensions(self) -> dict[str, float]:
        """Get all dimension values."""
        return {
            "a": self.dim_a,
            "b": self.dim_b,
            "c": self.dim_c,
            "d": self.dim_d,
            "e": self.dim_e,
        }

    def get_all_confidences(self) -> dict[str, float]:
        """Get all confidence values."""
        return {
            "a": self.confidence_a,
            "b": self.confidence_b,
            "c": self.confidence_c,
            "d": self.confidence_d,
            "e": self.confidence_e,
        }

    def get_overall_stability(self) -> float:
        """Get overall stability (average confidence)."""
        confidences = self.get_all_confidences()
        return sum(confidences.values()) / len(confidences)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "dim_a": self.dim_a,
            "dim_b": self.dim_b,
            "dim_c": self.dim_c,
            "dim_d": self.dim_d,
            "dim_e": self.dim_e,
            "confidence_a": self.confidence_a,
            "confidence_b": self.confidence_b,
            "confidence_c": self.confidence_c,
            "confidence_d": self.confidence_d,
            "confidence_e": self.confidence_e,
            "update_count": self.update_count,
            "last_update": self.last_update,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueOrientation":
        """Deserialize from dict."""
        return cls(
            dim_a=data.get("dim_a", 0.0),
            dim_b=data.get("dim_b", 0.0),
            dim_c=data.get("dim_c", 0.0),
            dim_d=data.get("dim_d", 0.0),
            dim_e=data.get("dim_e", 0.0),
            confidence_a=data.get("confidence_a", 0.0),
            confidence_b=data.get("confidence_b", 0.0),
            confidence_c=data.get("confidence_c", 0.0),
            confidence_d=data.get("confidence_d", 0.0),
            confidence_e=data.get("confidence_e", 0.0),
            update_count=data.get("update_count", 0),
            last_update=data.get("last_update", time.time()),
            version=data.get("version", 1),
        )


# ── Configuration ───────────────────────────────────────────────────


def _vo_defaults() -> dict[str, Any]:
    """Load value orientation defaults from coefficient registry."""
    return coefficient_registry.get("value_orientation")


@dataclass
class ValueOrientationConfig:
    """
    Configuration for value orientation behavior.

    Key principle: HIGH INERTIA
    - base_learning_rate is very small (0.01 default)
    - confidence dampens changes further
    - Single events cannot flip orientation
    """

    # Base learning rate for dimension updates (VERY SMALL for inertia)
    base_learning_rate: float = field(default_factory=lambda: _vo_defaults()["base_learning_rate"])

    # How much confidence dampens learning (higher = more stable)
    confidence_damping: float = field(default_factory=lambda: _vo_defaults()["confidence_damping"])

    # Rate at which confidence increases with consistent signals
    confidence_growth_rate: float = field(default_factory=lambda: _vo_defaults()["confidence_growth_rate"])

    # Rate at which confidence decays without reinforcement
    confidence_decay_rate: float = field(default_factory=lambda: _vo_defaults()["confidence_decay_rate"])

    # Maximum bias strength applied to candidates
    max_bias_strength: float = field(default_factory=lambda: _vo_defaults()["max_bias_strength"])

    # Minimum absolute dimension value to apply bias
    min_dimension_threshold: float = field(default_factory=lambda: _vo_defaults()["min_dimension_threshold"])

    # How much confidence amplifies bias application
    confidence_bias_amplifier: float = field(default_factory=lambda: _vo_defaults()["confidence_bias_amplifier"])

    # Decay rate for orientation toward neutral (very slow)
    neutral_decay_rate: float = field(default_factory=lambda: _vo_defaults()["neutral_decay_rate"])

    # Policy dimension mappings (abstract, not moral)
    # Maps policy labels to dimension influences
    # Positive = increases dimension, Negative = decreases
    policy_dimension_map: dict[str, dict[str, float]] = field(default_factory=lambda: {
        # These are EXAMPLES of how policies might map to abstract dimensions
        # The actual meaning emerges from usage patterns
        "からかう": {"a": 0.3, "c": 0.2},       # Teasing: bold, immediate
        "共感する": {"b": 0.2, "d": -0.1},      # Empathy: collective, concrete
        "励ます": {"b": 0.2, "e": 0.1},         # Encourage: collective, change
        "質問で会話を広げる": {"d": 0.2},       # Questions: abstract
        "話題を変える": {"e": 0.2, "c": 0.1},   # Topic change: change, immediate
        "感想を述べる": {"a": 0.1},             # Opinion: slightly bold
        "沈黙する": {"a": -0.2, "e": -0.1},     # Silence: cautious, stability
        # ── 追加9件: ポリシー候補構造の動的化 ──
        "黙って聞く": {"a": -0.2, "e": -0.1},
        "自分の経験を話す": {"a": 0.2, "d": 0.1},
        "確認する": {"d": 0.2, "b": 0.1},
        "冗談を言う": {"a": 0.3, "c": 0.2},
        "謝る": {"a": -0.3, "b": 0.2},
        "提案する": {"a": 0.1, "e": 0.2},
        "見守る": {"a": -0.1, "e": -0.2},
        "同意する": {"b": 0.3, "a": -0.1},
        "反論する": {"a": 0.3, "b": -0.2},
    })

    # Default influence for unmapped policies
    default_policy_influence: dict[str, float] = field(default_factory=dict)


# ── Update Functions ────────────────────────────────────────────────


def compute_effective_learning_rate(
    base_rate: float,
    confidence: float,
    damping: float,
) -> float:
    """
    Compute effective learning rate with confidence damping.

    Higher confidence = lower learning rate (more stable).
    """
    damping_factor = 1.0 - (confidence * damping)
    return base_rate * max(0.1, damping_factor)


def update_dimension(
    current_value: float,
    current_confidence: float,
    signal: float,
    config: ValueOrientationConfig,
) -> tuple[float, float]:
    """
    Update a single dimension based on a signal.

    Returns (new_value, new_confidence).

    Key properties:
    - High inertia: small changes only
    - Confidence dampens changes
    - Consistent signals build confidence
    """
    if abs(signal) < 0.001:
        # No significant signal, apply tiny decay toward neutral
        decay = config.neutral_decay_rate
        new_value = current_value * (1.0 - decay)
        new_confidence = max(0.0, current_confidence - config.confidence_decay_rate)
        return (new_value, new_confidence)

    # Compute effective learning rate (dampened by confidence)
    lr = compute_effective_learning_rate(
        config.base_learning_rate,
        current_confidence,
        config.confidence_damping,
    )

    # Apply update (very small due to low learning rate)
    delta = signal * lr
    new_value = current_value + delta

    # Clamp to valid range
    new_value = max(-1.0, min(1.0, new_value))

    # Update confidence based on signal consistency
    # If signal aligns with current direction, increase confidence
    # If signal opposes, decrease confidence
    if (signal > 0 and current_value > 0) or (signal < 0 and current_value < 0):
        # Consistent signal: grow confidence
        new_confidence = current_confidence + config.confidence_growth_rate
    elif abs(current_value) < 0.1:
        # Near neutral: small confidence growth for any signal
        new_confidence = current_confidence + config.confidence_growth_rate * 0.5
    else:
        # Opposing signal: decay confidence
        new_confidence = current_confidence - config.confidence_decay_rate * 2

    new_confidence = max(0.0, min(1.0, new_confidence))

    return (new_value, new_confidence)


def update_orientation(
    orientation: ValueOrientation,
    decision_signal: Optional[dict[str, float]] = None,
    emotion_signal: Optional[dict[str, float]] = None,
    responsibility_signal: Optional[dict[str, float]] = None,
    config: Optional[ValueOrientationConfig] = None,
) -> ValueOrientation:
    """
    Update value orientation based on various signals.

    All updates are GRADUAL due to high inertia.
    A single event cannot flip the orientation.

    Args:
        orientation: Current orientation state
        decision_signal: Signal from recent decisions (dim_name -> strength)
        emotion_signal: Signal from emotion patterns
        responsibility_signal: Signal from responsibility patterns
        config: Configuration

    Returns:
        New ValueOrientation (original is not mutated)
    """
    cfg = config or ValueOrientationConfig()

    # Combine all signals (weighted equally for now)
    combined_signal: dict[str, float] = {}

    for signal in [decision_signal, emotion_signal, responsibility_signal]:
        if signal:
            for dim, value in signal.items():
                # Normalize dimension names (strip "dim_" prefix if present)
                dim_key = dim.replace("dim_", "")
                if dim_key in combined_signal:
                    combined_signal[dim_key] += value
                else:
                    combined_signal[dim_key] = value

    # Average combined signals if multiple sources
    signal_count = sum(1 for s in [decision_signal, emotion_signal, responsibility_signal] if s)
    if signal_count > 1:
        for key in combined_signal:
            combined_signal[key] /= signal_count

    # Update each dimension
    new_dim_a, new_conf_a = update_dimension(
        orientation.dim_a,
        orientation.confidence_a,
        combined_signal.get("a", 0.0),
        cfg,
    )
    new_dim_b, new_conf_b = update_dimension(
        orientation.dim_b,
        orientation.confidence_b,
        combined_signal.get("b", 0.0),
        cfg,
    )
    new_dim_c, new_conf_c = update_dimension(
        orientation.dim_c,
        orientation.confidence_c,
        combined_signal.get("c", 0.0),
        cfg,
    )
    new_dim_d, new_conf_d = update_dimension(
        orientation.dim_d,
        orientation.confidence_d,
        combined_signal.get("d", 0.0),
        cfg,
    )
    new_dim_e, new_conf_e = update_dimension(
        orientation.dim_e,
        orientation.confidence_e,
        combined_signal.get("e", 0.0),
        cfg,
    )

    return ValueOrientation(
        dim_a=new_dim_a,
        dim_b=new_dim_b,
        dim_c=new_dim_c,
        dim_d=new_dim_d,
        dim_e=new_dim_e,
        confidence_a=new_conf_a,
        confidence_b=new_conf_b,
        confidence_c=new_conf_c,
        confidence_d=new_conf_d,
        confidence_e=new_conf_e,
        update_count=orientation.update_count + 1,
        last_update=time.time(),
        version=orientation.version,
    )


def generate_decision_signal(
    policy_label: str,
    config: Optional[ValueOrientationConfig] = None,
) -> dict[str, float]:
    """
    Generate a decision signal from a chosen policy.

    Maps policy labels to dimension influences.
    """
    cfg = config or ValueOrientationConfig()

    if policy_label in cfg.policy_dimension_map:
        return cfg.policy_dimension_map[policy_label].copy()
    else:
        return cfg.default_policy_influence.copy()


def update_from_decision(
    orientation: ValueOrientation,
    policy_label: str,
    config: Optional[ValueOrientationConfig] = None,
) -> ValueOrientation:
    """
    Convenience function to update orientation from a decision.

    Args:
        orientation: Current orientation
        policy_label: Label of the chosen policy
        config: Configuration

    Returns:
        Updated orientation
    """
    cfg = config or ValueOrientationConfig()
    signal = generate_decision_signal(policy_label, cfg)

    if not signal:
        return orientation

    return update_orientation(orientation, decision_signal=signal, config=cfg)


# ── Bias Application ────────────────────────────────────────────────


@dataclass
class OrientationBias:
    """
    Computed bias from value orientation.

    This bias is SUBTLE and does NOT dictate decisions.
    """

    # Per-dimension bias contributions
    dimension_biases: dict[str, float] = field(default_factory=dict)

    # Overall bias strength (for scaling)
    overall_strength: float = 0.0

    # Summary of orientation state
    orientation_summary: dict[str, float] = field(default_factory=dict)


def compute_orientation_bias(
    orientation: ValueOrientation,
    policy_label: str,
    config: Optional[ValueOrientationConfig] = None,
) -> OrientationBias:
    """
    Compute bias for a specific policy based on orientation.

    The bias represents how much the orientation "prefers" this policy.
    It is SUBTLE and does not force selection.
    """
    cfg = config or ValueOrientationConfig()

    # Get policy's dimension influences
    policy_influences = cfg.policy_dimension_map.get(
        policy_label,
        cfg.default_policy_influence,
    )

    if not policy_influences:
        return OrientationBias(
            dimension_biases={},
            overall_strength=0.0,
            orientation_summary=orientation.get_all_dimensions(),
        )

    dimension_biases: dict[str, float] = {}
    total_bias = 0.0

    for dim_name, influence in policy_influences.items():
        dim_value = orientation.get_dimension(dim_name)
        confidence = orientation.get_confidence(dim_name)

        # Skip dimensions below threshold
        if abs(dim_value) < cfg.min_dimension_threshold:
            continue

        # Compute alignment: positive if orientation and influence agree
        alignment = dim_value * influence

        # Scale by confidence
        confidence_factor = 1.0 + (confidence * cfg.confidence_bias_amplifier)

        # Compute dimension bias
        dim_bias = alignment * confidence_factor

        dimension_biases[dim_name] = dim_bias
        total_bias += dim_bias

    # Scale to max bias strength
    if abs(total_bias) > cfg.max_bias_strength:
        scale = cfg.max_bias_strength / abs(total_bias)
        total_bias *= scale
        for key in dimension_biases:
            dimension_biases[key] *= scale

    return OrientationBias(
        dimension_biases=dimension_biases,
        overall_strength=total_bias,
        orientation_summary=orientation.get_all_dimensions(),
    )


def apply_orientation_to_candidate(
    candidate: dict[str, Any],
    orientation: ValueOrientation,
    config: Optional[ValueOrientationConfig] = None,
) -> dict[str, Any]:
    """
    Apply orientation bias to a single candidate.

    The bias is ADDITIVE and SUBTLE.
    It does NOT block or force any candidate.
    """
    cfg = config or ValueOrientationConfig()

    result = candidate.copy()
    original_score = candidate.get("_score", 0.0)

    policy_label = candidate.get("policy_label", "")

    # Compute bias for this policy
    bias = compute_orientation_bias(orientation, policy_label, cfg)

    # Apply bias additively (subtle)
    adjusted_score = original_score + bias.overall_strength

    # Store metadata
    result["_score"] = round(adjusted_score, 4)
    result["_pre_orientation_score"] = original_score
    result["_orientation_bias"] = round(bias.overall_strength, 4)
    result["_orientation_applied"] = True

    return result


def apply_orientation_to_candidates(
    candidates: list[dict[str, Any]],
    orientation: ValueOrientation,
    config: Optional[ValueOrientationConfig] = None,
) -> list[dict[str, Any]]:
    """
    Apply orientation bias to all candidates.

    Bias applies to ALL candidates (universal).
    The list is re-sorted by adjusted scores.
    """
    adjusted = [
        apply_orientation_to_candidate(c, orientation, config)
        for c in candidates
    ]

    # Re-sort by adjusted score
    adjusted.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return adjusted


# ── Signal Generation from State ────────────────────────────────────


def generate_emotion_signal(
    emotion_vector: Any,  # EmotionVector
) -> dict[str, float]:
    """
    Generate orientation signal from emotion patterns.

    Maps emotion tendencies to abstract dimensions.
    This is one possible mapping - not hardcoded morals.
    """
    signal: dict[str, float] = {}

    # Extract emotion values if available
    joy = getattr(emotion_vector, "joy", 0.0)
    anger = getattr(emotion_vector, "anger", 0.0)
    sadness = getattr(emotion_vector, "sadness", 0.0)
    fear = getattr(emotion_vector, "fear", 0.0)

    # Map to dimensions (abstract associations)
    # Joy -> slightly toward boldness (dim_a)
    if joy > 0.3:
        signal["a"] = joy * 0.1

    # Anger -> toward immediacy (dim_c)
    if anger > 0.3:
        signal["c"] = anger * 0.15

    # Sadness -> toward stability (dim_e negative)
    if sadness > 0.3:
        signal["e"] = -sadness * 0.1

    # Fear -> toward caution (dim_a negative)
    if fear > 0.3:
        signal["a"] = signal.get("a", 0.0) - fear * 0.15

    return signal


def generate_responsibility_signal(
    total_weight: float,
    sublimation_count: int = 0,
) -> dict[str, float]:
    """
    Generate orientation signal from responsibility patterns.

    High responsibility load may influence certain dimensions.
    """
    signal: dict[str, float] = {}

    # High responsibility -> toward caution (dim_a negative)
    if total_weight > 0.5:
        signal["a"] = -0.1 * min(1.0, total_weight)

    # Sublimation success -> toward long-term (dim_c negative)
    if sublimation_count > 0:
        signal["c"] = -0.05 * min(1.0, sublimation_count / 5.0)

    return signal


# ── Utility Functions ───────────────────────────────────────────────


def get_orientation_summary(orientation: ValueOrientation) -> str:
    """Get human-readable summary of orientation."""
    dims = orientation.get_all_dimensions()
    confs = orientation.get_all_confidences()

    active_dims = []
    for name, value in dims.items():
        if abs(value) >= 0.1:
            conf = confs.get(name, 0.0)
            direction = "+" if value > 0 else "-"
            active_dims.append(f"{name.upper()}{direction}({abs(value):.2f}, conf={conf:.2f})")

    if not active_dims:
        return f"Orientation: neutral (updates={orientation.update_count})"

    return f"Orientation: {', '.join(active_dims)} (updates={orientation.update_count})"


def get_orientation_vector(orientation: ValueOrientation) -> list[float]:
    """Get orientation as a vector (for distance calculations)."""
    return [
        orientation.dim_a,
        orientation.dim_b,
        orientation.dim_c,
        orientation.dim_d,
        orientation.dim_e,
    ]


def compute_orientation_distance(
    orientation1: ValueOrientation,
    orientation2: ValueOrientation,
) -> float:
    """Compute Euclidean distance between two orientations."""
    v1 = get_orientation_vector(orientation1)
    v2 = get_orientation_vector(orientation2)

    squared_diff = sum((a - b) ** 2 for a, b in zip(v1, v2))
    return math.sqrt(squared_diff)


def is_orientation_stable(
    orientation: ValueOrientation,
    stability_threshold: float = 0.3,
) -> bool:
    """Check if orientation has reached a stable state."""
    return orientation.get_overall_stability() >= stability_threshold


def create_orientation() -> ValueOrientation:
    """Create a new neutral orientation."""
    return ValueOrientation()


def create_config(
    base_learning_rate: float = 0.01,
    max_bias_strength: float = 0.15,
) -> ValueOrientationConfig:
    """Create a configuration with custom parameters."""
    return ValueOrientationConfig(
        base_learning_rate=base_learning_rate,
        max_bias_strength=max_bias_strength,
    )


def to_dict(config: ValueOrientationConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "base_learning_rate": config.base_learning_rate,
        "confidence_damping": config.confidence_damping,
        "confidence_growth_rate": config.confidence_growth_rate,
        "confidence_decay_rate": config.confidence_decay_rate,
        "max_bias_strength": config.max_bias_strength,
        "min_dimension_threshold": config.min_dimension_threshold,
        "confidence_bias_amplifier": config.confidence_bias_amplifier,
        "neutral_decay_rate": config.neutral_decay_rate,
        "policy_dimension_map": {
            k: v.copy() for k, v in config.policy_dimension_map.items()
        },
        "default_policy_influence": config.default_policy_influence.copy(),
    }


def from_dict(data: dict[str, Any]) -> ValueOrientationConfig:
    """Deserialize config from dict."""
    config = ValueOrientationConfig()

    config.base_learning_rate = data.get("base_learning_rate", config.base_learning_rate)
    config.confidence_damping = data.get("confidence_damping", config.confidence_damping)
    config.confidence_growth_rate = data.get("confidence_growth_rate", config.confidence_growth_rate)
    config.confidence_decay_rate = data.get("confidence_decay_rate", config.confidence_decay_rate)
    config.max_bias_strength = data.get("max_bias_strength", config.max_bias_strength)
    config.min_dimension_threshold = data.get("min_dimension_threshold", config.min_dimension_threshold)
    config.confidence_bias_amplifier = data.get("confidence_bias_amplifier", config.confidence_bias_amplifier)
    config.neutral_decay_rate = data.get("neutral_decay_rate", config.neutral_decay_rate)

    if "policy_dimension_map" in data:
        config.policy_dimension_map = {
            k: v.copy() for k, v in data["policy_dimension_map"].items()
        }

    if "default_policy_influence" in data:
        config.default_policy_influence = data["default_policy_influence"].copy()

    return config
