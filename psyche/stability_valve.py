"""
psyche/stability_valve.py - Stability / Safety Valve (極端回避防止)

Implements a structural anti-stuck mechanism that monitors system extremes
and flattens score distributions to prevent fixation.

Design principles (from design_stability_valve.md):
- Does NOT judge content, only patterns (repetition/extremity)
- Does NOT force hard reset, uses gradual bias adjustment
- Does NOT prohibit or force any specific decision
- Gradual, continuous activation (not ON/OFF switch)
- Integrates with IntrospectionTrace for logging

Usage::

    from psyche.stability_valve import (
        StabilityValve,
        ExtremityIndicators,
        StabilityBias,
        observe_extremity,
        apply_stability_bias,
    )

    # Create valve
    valve = StabilityValve()

    # Record decision patterns
    valve.record_decision("沈黙する")

    # Observe extremity indicators
    indicators = valve.observe_extremity(
        fear_level=0.8,
        responsibility_weight=0.7,
    )

    # Generate and apply stability bias
    bias = valve.generate_bias(indicators)
    adjusted = apply_stability_bias(candidates, bias)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import time
import math


# ── Extremity Indicators ────────────────────────────────────────────


@dataclass
class ExtremityIndicators:
    """
    Observed indicators of system extremity.

    These are PATTERNS, not judgments of content.
    High values indicate structural extremity, not "wrongness".
    """

    # Fear/loss extremity (0.0 = normal, 1.0 = maxed out)
    fear_extremity: float = 0.0

    # Responsibility concentration (0.0 = distributed, 1.0 = concentrated)
    responsibility_extremity: float = 0.0

    # Decision pattern fixation (0.0 = varied, 1.0 = single pattern)
    decision_fixation: float = 0.0

    # Value orientation extremity (0.0 = balanced, 1.0 = extreme bias)
    value_extremity: float = 0.0

    # Emotion saturation (0.0 = varied, 1.0 = single emotion dominates)
    emotion_saturation: float = 0.0

    # Overall extremity score (composite)
    overall_extremity: float = 0.0

    # Count of consecutive extreme observations
    consecutive_extreme_count: int = 0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Clamp all values and compute overall."""
        self.fear_extremity = max(0.0, min(1.0, self.fear_extremity))
        self.responsibility_extremity = max(0.0, min(1.0, self.responsibility_extremity))
        self.decision_fixation = max(0.0, min(1.0, self.decision_fixation))
        self.value_extremity = max(0.0, min(1.0, self.value_extremity))
        self.emotion_saturation = max(0.0, min(1.0, self.emotion_saturation))

        # Compute overall as weighted average
        if self.overall_extremity == 0.0:
            self.overall_extremity = self._compute_overall()

    def _compute_overall(self) -> float:
        """Compute overall extremity from components."""
        components = [
            self.fear_extremity,
            self.responsibility_extremity,
            self.decision_fixation,
            self.value_extremity,
            self.emotion_saturation,
        ]
        # Use max-weighted average (worst indicator matters most)
        if not components:
            return 0.0
        max_val = max(components)
        avg_val = sum(components) / len(components)
        # Blend: 60% max, 40% average
        return 0.6 * max_val + 0.4 * avg_val

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "fear_extremity": round(self.fear_extremity, 4),
            "responsibility_extremity": round(self.responsibility_extremity, 4),
            "decision_fixation": round(self.decision_fixation, 4),
            "value_extremity": round(self.value_extremity, 4),
            "emotion_saturation": round(self.emotion_saturation, 4),
            "overall_extremity": round(self.overall_extremity, 4),
            "consecutive_extreme_count": self.consecutive_extreme_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtremityIndicators":
        """Deserialize from dict."""
        return cls(
            fear_extremity=data.get("fear_extremity", 0.0),
            responsibility_extremity=data.get("responsibility_extremity", 0.0),
            decision_fixation=data.get("decision_fixation", 0.0),
            value_extremity=data.get("value_extremity", 0.0),
            emotion_saturation=data.get("emotion_saturation", 0.0),
            overall_extremity=data.get("overall_extremity", 0.0),
            consecutive_extreme_count=data.get("consecutive_extreme_count", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# ── Stability Bias ──────────────────────────────────────────────────


@dataclass
class StabilityBias:
    """
    Bias that flattens score distributions to prevent fixation.

    This bias does NOT:
    - Prohibit any decision
    - Force any specific choice
    - Judge content as good/bad

    It only makes alternatives more likely by flattening scores.
    """

    # How much to flatten the score distribution (0.0 = no effect, 1.0 = full flatten)
    flatten_strength: float = 0.0

    # Minimum score boost for low-scoring candidates
    min_score_boost: float = 0.0

    # Maximum score reduction for high-scoring candidates
    max_score_reduction: float = 0.0

    # Whether valve is currently active
    is_active: bool = False

    # Activation level (continuous, not binary)
    activation_level: float = 0.0

    # Source indicators that triggered this bias
    source_indicators: dict[str, float] = field(default_factory=dict)

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "flatten_strength": round(self.flatten_strength, 4),
            "min_score_boost": round(self.min_score_boost, 4),
            "max_score_reduction": round(self.max_score_reduction, 4),
            "is_active": self.is_active,
            "activation_level": round(self.activation_level, 4),
            "source_indicators": {k: round(v, 4) for k, v in self.source_indicators.items()},
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StabilityBias":
        """Deserialize from dict."""
        return cls(
            flatten_strength=data.get("flatten_strength", 0.0),
            min_score_boost=data.get("min_score_boost", 0.0),
            max_score_reduction=data.get("max_score_reduction", 0.0),
            is_active=data.get("is_active", False),
            activation_level=data.get("activation_level", 0.0),
            source_indicators=data.get("source_indicators", {}),
            timestamp=data.get("timestamp", time.time()),
        )


def create_neutral_bias() -> StabilityBias:
    """Create a neutral bias (no effect)."""
    return StabilityBias()


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class StabilityValveConfig:
    """
    Configuration for stability valve behavior.

    All thresholds are configurable, not hardcoded.
    """

    # Threshold for considering an indicator "extreme"
    extremity_threshold: float = 0.6

    # How many consecutive extreme observations before activation
    consecutive_threshold: int = 3

    # Maximum flatten strength
    max_flatten_strength: float = 0.4

    # How fast activation increases when extreme
    activation_rate: float = 0.1

    # How fast activation decays when not extreme
    decay_rate: float = 0.05

    # Decision history window size
    decision_history_size: int = 10

    # Fixation threshold (same decision ratio to trigger)
    fixation_threshold: float = 0.7

    # Fear level considered extreme
    fear_extreme_threshold: float = 0.7

    # Responsibility weight considered extreme
    responsibility_extreme_threshold: float = 0.8

    # Value orientation absolute value considered extreme
    value_extreme_threshold: float = 0.7

    # Emotion dominance ratio considered saturated
    emotion_saturation_threshold: float = 0.6


# ── Stability Valve ─────────────────────────────────────────────────


class StabilityValve:
    """
    Structural anti-stuck mechanism.

    Monitors system extremes and generates stabilizing bias
    that flattens score distributions.

    Key properties:
    - Does NOT judge content, only patterns
    - Gradual activation (not ON/OFF)
    - Does NOT prohibit any decision
    """

    def __init__(self, config: Optional[StabilityValveConfig] = None):
        self.config = config or StabilityValveConfig()
        self._decision_history: deque[str] = deque(maxlen=self.config.decision_history_size)
        self._activation_level: float = 0.0
        self._consecutive_extreme: int = 0
        self._last_indicators: Optional[ExtremityIndicators] = None
        self._last_bias: Optional[StabilityBias] = None
        self._observation_count: int = 0

    def record_decision(self, policy_label: str) -> None:
        """
        Record a decision for pattern tracking.

        This does NOT judge the decision, only tracks patterns.
        """
        self._decision_history.append(policy_label)

    def observe_extremity(
        self,
        fear_level: float = 0.0,
        responsibility_weight: float = 0.0,
        value_orientation: Optional[Any] = None,
        emotion_state: Optional[Any] = None,
        fear_index: Optional[Any] = None,
    ) -> ExtremityIndicators:
        """
        Observe current extremity indicators.

        This is READ-ONLY observation, no modifications.
        """
        self._observation_count += 1
        cfg = self.config

        # ── Fear extremity ──
        fear_val = fear_level
        if fear_index is not None:
            fear_val = max(fear_val, getattr(fear_index, "value", 0.0))

        fear_extremity = 0.0
        if fear_val > cfg.fear_extreme_threshold:
            fear_extremity = (fear_val - cfg.fear_extreme_threshold) / (1.0 - cfg.fear_extreme_threshold)

        # ── Responsibility extremity ──
        resp_extremity = 0.0
        if responsibility_weight > cfg.responsibility_extreme_threshold:
            resp_extremity = (responsibility_weight - cfg.responsibility_extreme_threshold) / (1.0 - cfg.responsibility_extreme_threshold)

        # ── Decision fixation ──
        decision_fixation = self._compute_decision_fixation()

        # ── Value orientation extremity ──
        value_extremity = 0.0
        if value_orientation is not None:
            dims = [
                abs(getattr(value_orientation, f"dim_{d}", 0.0))
                for d in "abcde"
            ]
            max_dim = max(dims) if dims else 0.0
            if max_dim > cfg.value_extreme_threshold:
                value_extremity = (max_dim - cfg.value_extreme_threshold) / (1.0 - cfg.value_extreme_threshold)

        # ── Emotion saturation ──
        emotion_saturation = 0.0
        if emotion_state is not None:
            emotion_saturation = self._compute_emotion_saturation(emotion_state)

        # ── Build indicators ──
        indicators = ExtremityIndicators(
            fear_extremity=fear_extremity,
            responsibility_extremity=resp_extremity,
            decision_fixation=decision_fixation,
            value_extremity=value_extremity,
            emotion_saturation=emotion_saturation,
            consecutive_extreme_count=self._consecutive_extreme,
        )

        # Update consecutive count
        if indicators.overall_extremity > cfg.extremity_threshold:
            self._consecutive_extreme += 1
        else:
            self._consecutive_extreme = max(0, self._consecutive_extreme - 1)

        indicators.consecutive_extreme_count = self._consecutive_extreme
        self._last_indicators = indicators

        return indicators

    def _compute_decision_fixation(self) -> float:
        """Compute decision fixation from history."""
        if len(self._decision_history) < 3:
            return 0.0

        history = list(self._decision_history)
        total = len(history)

        # Count most common decision
        from collections import Counter
        counts = Counter(history)
        most_common_count = counts.most_common(1)[0][1]

        ratio = most_common_count / total

        if ratio > self.config.fixation_threshold:
            return (ratio - self.config.fixation_threshold) / (1.0 - self.config.fixation_threshold)

        return 0.0

    def _compute_emotion_saturation(self, emotion_state: Any) -> float:
        """Compute emotion saturation (one emotion dominates)."""
        emotions = [
            getattr(emotion_state, "joy", 0.0),
            getattr(emotion_state, "anger", 0.0),
            getattr(emotion_state, "sadness", 0.0),
            getattr(emotion_state, "fear", 0.0),
            getattr(emotion_state, "disgust", 0.0),
            getattr(emotion_state, "surprise", 0.0),
        ]

        total = sum(emotions)
        if total < 0.1:
            return 0.0

        max_emotion = max(emotions)
        dominance = max_emotion / total if total > 0 else 0.0

        if dominance > self.config.emotion_saturation_threshold:
            return (dominance - self.config.emotion_saturation_threshold) / (1.0 - self.config.emotion_saturation_threshold)

        return 0.0

    def generate_bias(
        self,
        indicators: Optional[ExtremityIndicators] = None,
    ) -> StabilityBias:
        """
        Generate stability bias based on extremity indicators.

        The bias flattens score distributions, making alternatives more likely.
        It does NOT prohibit or force any specific decision.
        """
        cfg = self.config
        ind = indicators or self._last_indicators or ExtremityIndicators()

        # Update activation level (gradual, not ON/OFF)
        if ind.overall_extremity > cfg.extremity_threshold:
            # Increase activation
            self._activation_level = min(
                1.0,
                self._activation_level + cfg.activation_rate * ind.overall_extremity
            )
        else:
            # Decay activation
            self._activation_level = max(
                0.0,
                self._activation_level - cfg.decay_rate
            )

        # Boost activation if consecutive extremes
        if ind.consecutive_extreme_count >= cfg.consecutive_threshold:
            consecutive_boost = 0.1 * (ind.consecutive_extreme_count - cfg.consecutive_threshold + 1)
            self._activation_level = min(1.0, self._activation_level + consecutive_boost)

        # Compute bias strength
        is_active = self._activation_level > 0.01
        flatten_strength = self._activation_level * cfg.max_flatten_strength

        # Compute boost/reduction amounts
        min_score_boost = flatten_strength * 0.5  # Boost low scores
        max_score_reduction = flatten_strength * 0.3  # Reduce high scores

        # Build source indicators
        source_indicators = {
            "fear": ind.fear_extremity,
            "responsibility": ind.responsibility_extremity,
            "decision_fixation": ind.decision_fixation,
            "value": ind.value_extremity,
            "emotion_saturation": ind.emotion_saturation,
        }

        bias = StabilityBias(
            flatten_strength=flatten_strength,
            min_score_boost=min_score_boost,
            max_score_reduction=max_score_reduction,
            is_active=is_active,
            activation_level=self._activation_level,
            source_indicators=source_indicators,
        )

        self._last_bias = bias
        return bias

    def get_activation_level(self) -> float:
        """Get current activation level."""
        return self._activation_level

    def get_last_indicators(self) -> Optional[ExtremityIndicators]:
        """Get last observed indicators."""
        return self._last_indicators

    def get_last_bias(self) -> Optional[StabilityBias]:
        """Get last generated bias."""
        return self._last_bias

    def reset_activation(self) -> None:
        """Reset activation level (use sparingly)."""
        self._activation_level = 0.0
        self._consecutive_extreme = 0

    def get_decision_history(self) -> list[str]:
        """Get copy of decision history."""
        return list(self._decision_history)

    def clear_history(self) -> None:
        """Clear decision history."""
        self._decision_history.clear()


# ── Bias Application ────────────────────────────────────────────────


def flatten_scores(
    scores: list[float],
    flatten_strength: float,
) -> list[float]:
    """
    Flatten a list of scores toward the mean.

    Higher flatten_strength = more flattening.
    """
    if not scores or flatten_strength <= 0:
        return scores

    mean = sum(scores) / len(scores)

    flattened = []
    for score in scores:
        # Move toward mean by flatten_strength
        new_score = score + (mean - score) * flatten_strength
        flattened.append(new_score)

    return flattened


def apply_stability_to_candidate(
    candidate: dict[str, Any],
    bias: StabilityBias,
    mean_score: float = 0.5,
) -> dict[str, Any]:
    """
    Apply stability bias to a single candidate.

    Flattens score toward mean, making alternatives more likely.
    """
    if not bias.is_active:
        return candidate

    result = candidate.copy()
    original_score = candidate.get("_score", 0.0)

    # Flatten toward mean
    adjustment = (mean_score - original_score) * bias.flatten_strength

    # Apply boost for low scores
    if original_score < mean_score:
        adjustment += bias.min_score_boost * (1.0 - original_score / mean_score)

    # Apply reduction for high scores
    if original_score > mean_score and mean_score > 0:
        adjustment -= bias.max_score_reduction * (original_score - mean_score) / (1.0 - mean_score + 0.01)

    new_score = original_score + adjustment

    # Store metadata
    result["_score"] = round(max(0.0, new_score), 4)
    result["_pre_stability_score"] = original_score
    result["_stability_adjustment"] = round(adjustment, 4)
    result["_stability_active"] = True
    result["_stability_activation"] = round(bias.activation_level, 4)

    return result


def apply_stability_bias(
    candidates: list[dict[str, Any]],
    bias: StabilityBias,
) -> list[dict[str, Any]]:
    """
    Apply stability bias to all candidates.

    Flattens score distribution, making alternatives more likely.
    Does NOT prohibit or force any specific decision.
    """
    if not bias.is_active or not candidates:
        return candidates

    # Compute mean score
    scores = [c.get("_score", 0.0) for c in candidates]
    mean_score = sum(scores) / len(scores) if scores else 0.5

    # Apply to each candidate
    adjusted = [
        apply_stability_to_candidate(c, bias, mean_score)
        for c in candidates
    ]

    # Re-sort by adjusted score
    adjusted.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return adjusted


# ── Introspection Integration ───────────────────────────────────────


def create_stability_factor(
    bias: StabilityBias,
) -> Optional[dict[str, Any]]:
    """
    Create a contributing factor for introspection trace.

    Returns None if valve is not active.
    """
    if not bias.is_active:
        return None

    # Find dominant source
    dominant_source = max(bias.source_indicators.items(), key=lambda x: x[1]) if bias.source_indicators else ("unknown", 0.0)

    return {
        "factor_id": f"stability_{int(time.time() * 1000) % 100000}",
        "category": "stability",
        "name": "stability_valve",
        "observed_value": bias.activation_level,
        "threshold": 0.01,
        "direction": "dampening",
        "contribution_strength": bias.flatten_strength,
        "description": (
            f"Stability valve active (level={bias.activation_level:.2f}) "
            f"due to {dominant_source[0]} extremity ({dominant_source[1]:.2f}). "
            f"Score distribution flattened by {bias.flatten_strength:.2f}."
        ),
    }


def get_stability_trace_context(
    valve: StabilityValve,
) -> dict[str, Any]:
    """
    Get context information for introspection trace.
    """
    indicators = valve.get_last_indicators()
    bias = valve.get_last_bias()

    return {
        "stability_valve_active": bias.is_active if bias else False,
        "activation_level": valve.get_activation_level(),
        "indicators": indicators.to_dict() if indicators else None,
        "bias": bias.to_dict() if bias else None,
        "decision_history_length": len(valve.get_decision_history()),
    }


# ── Convenience Functions ───────────────────────────────────────────


def create_stability_valve(
    config: Optional[StabilityValveConfig] = None,
) -> StabilityValve:
    """Create a new stability valve."""
    return StabilityValve(config)


def create_config(
    extremity_threshold: float = 0.6,
    max_flatten_strength: float = 0.4,
    decision_history_size: int = 10,
) -> StabilityValveConfig:
    """Create a configuration with custom parameters."""
    return StabilityValveConfig(
        extremity_threshold=extremity_threshold,
        max_flatten_strength=max_flatten_strength,
        decision_history_size=decision_history_size,
    )


def observe_extremity(
    valve: StabilityValve,
    fear_level: float = 0.0,
    responsibility_weight: float = 0.0,
    value_orientation: Optional[Any] = None,
    emotion_state: Optional[Any] = None,
) -> ExtremityIndicators:
    """Convenience function to observe extremity."""
    return valve.observe_extremity(
        fear_level=fear_level,
        responsibility_weight=responsibility_weight,
        value_orientation=value_orientation,
        emotion_state=emotion_state,
    )


def get_stability_summary(valve: StabilityValve) -> str:
    """Get human-readable summary of stability valve state."""
    activation = valve.get_activation_level()
    indicators = valve.get_last_indicators()

    if activation < 0.01:
        return "Stability Valve: inactive"

    status = "active" if activation > 0.1 else "warming"

    details = []
    if indicators:
        if indicators.fear_extremity > 0.1:
            details.append(f"fear={indicators.fear_extremity:.2f}")
        if indicators.decision_fixation > 0.1:
            details.append(f"fixation={indicators.decision_fixation:.2f}")
        if indicators.value_extremity > 0.1:
            details.append(f"value={indicators.value_extremity:.2f}")

    detail_str = f" ({', '.join(details)})" if details else ""

    return f"Stability Valve: {status} level={activation:.2f}{detail_str}"


def to_dict(config: StabilityValveConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "extremity_threshold": config.extremity_threshold,
        "consecutive_threshold": config.consecutive_threshold,
        "max_flatten_strength": config.max_flatten_strength,
        "activation_rate": config.activation_rate,
        "decay_rate": config.decay_rate,
        "decision_history_size": config.decision_history_size,
        "fixation_threshold": config.fixation_threshold,
        "fear_extreme_threshold": config.fear_extreme_threshold,
        "responsibility_extreme_threshold": config.responsibility_extreme_threshold,
        "value_extreme_threshold": config.value_extreme_threshold,
        "emotion_saturation_threshold": config.emotion_saturation_threshold,
    }


def from_dict(data: dict[str, Any]) -> StabilityValveConfig:
    """Deserialize config from dict."""
    return StabilityValveConfig(
        extremity_threshold=data.get("extremity_threshold", 0.6),
        consecutive_threshold=data.get("consecutive_threshold", 3),
        max_flatten_strength=data.get("max_flatten_strength", 0.4),
        activation_rate=data.get("activation_rate", 0.1),
        decay_rate=data.get("decay_rate", 0.05),
        decision_history_size=data.get("decision_history_size", 10),
        fixation_threshold=data.get("fixation_threshold", 0.7),
        fear_extreme_threshold=data.get("fear_extreme_threshold", 0.7),
        responsibility_extreme_threshold=data.get("responsibility_extreme_threshold", 0.8),
        value_extreme_threshold=data.get("value_extreme_threshold", 0.7),
        emotion_saturation_threshold=data.get("emotion_saturation_threshold", 0.6),
    )
