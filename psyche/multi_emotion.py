"""
psyche/multi_emotion.py - Multi-Emotion Reference (複数感情参照)

Provides structures and functions for referencing multiple emotions simultaneously.
Emotions coexist without canceling, normalizing, or averaging each other.

Design principles (from design_multi_emotion.md):
- Emotions are NOT single-valued; multiple emotions coexist in parallel
- Each emotion has independent value, decay, and amplitude
- Emotions do NOT cancel each other (joy does NOT reduce sorrow)
- No normalization, averaging, or merging of emotions
- Judgment/reaction can reference multiple emotions without prioritization

Usage::

    from psyche.multi_emotion import (
        MultiEmotionConfig,
        get_active_emotions,
        get_coexisting_pairs,
        apply_independent_decay,
        get_emotion_vector_summary,
    )

    # Get all active emotions (above threshold)
    active = get_active_emotions(emotion_vector)

    # Get emotions that coexist (e.g., joy AND fear simultaneously)
    pairs = get_coexisting_pairs(emotion_vector)

    # Apply decay with per-emotion rates
    decayed = apply_independent_decay(emotion_vector, config, delta_time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .state import EmotionVector


@dataclass
class EmotionDecayConfig:
    """
    Per-emotion decay configuration.

    Each emotion can have its own decay rate, allowing some emotions
    to persist longer than others without normalization.
    """

    # Base decay rate (per second, 0.0-1.0 where 1.0 = instant decay)
    base_decay_rate: float = 0.05

    # Per-emotion decay rates (emotion_name -> rate)
    # If not specified, uses base_decay_rate
    emotion_decay_rates: dict[str, float] = field(default_factory=lambda: {
        "joy": 0.05,
        "anger": 0.08,      # Anger decays slightly faster
        "sorrow": 0.03,     # Sorrow persists longer
        "fear": 0.06,
        "surprise": 0.15,   # Surprise decays fastest
        "love": 0.01,       # Love persists longest
        "fun": 0.07,
    })

    # Minimum emotion value (below this, consider emotion "inactive")
    inactive_threshold: float = 0.05

    def get_decay_rate(self, emotion: str) -> float:
        """Get decay rate for a specific emotion."""
        return self.emotion_decay_rates.get(emotion, self.base_decay_rate)


@dataclass
class MultiEmotionConfig:
    """
    Configuration for multi-emotion reference behavior.

    All parameters are externally configurable.
    No hardcoded emotion meanings or evaluations.
    """

    # Threshold for considering an emotion "active"
    active_threshold: float = 0.1

    # Threshold for considering emotions "coexisting"
    coexistence_threshold: float = 0.15

    # Decay configuration
    decay_config: EmotionDecayConfig = field(default_factory=EmotionDecayConfig)

    # Whether to allow per-emotion amplitude (future extension)
    per_emotion_amplitude: bool = False


# ── Core Functions ────────────────────────────────────────────────


def get_active_emotions(
    emotions: EmotionVector,
    config: Optional[MultiEmotionConfig] = None,
) -> dict[str, float]:
    """
    Get all emotions that are currently active (above threshold).

    This function returns MULTIPLE emotions; it does NOT reduce to
    a single "dominant" emotion. Emotions coexist.

    Args:
        emotions: Current EmotionVector
        config: Optional configuration

    Returns:
        Dict of emotion_name -> value for all active emotions
    """
    cfg = config or MultiEmotionConfig()
    emo_dict = emotions.as_dict()

    return {
        name: value
        for name, value in emo_dict.items()
        if value >= cfg.active_threshold
    }


def get_all_emotions(emotions: EmotionVector) -> dict[str, float]:
    """
    Get all emotions with their current values.

    No filtering, no normalization, no reduction.

    Args:
        emotions: Current EmotionVector

    Returns:
        Dict of all emotion_name -> value pairs
    """
    return emotions.as_dict()


def get_coexisting_pairs(
    emotions: EmotionVector,
    config: Optional[MultiEmotionConfig] = None,
) -> list[tuple[str, str, float, float]]:
    """
    Get pairs of emotions that are both active simultaneously.

    This explicitly shows that emotions can coexist (e.g., joy AND fear).
    No cancellation, no averaging.

    Args:
        emotions: Current EmotionVector
        config: Optional configuration

    Returns:
        List of (emotion_a, emotion_b, value_a, value_b) tuples
    """
    cfg = config or MultiEmotionConfig()
    active = get_active_emotions(emotions, cfg)

    if len(active) < 2:
        return []

    pairs = []
    emotion_names = list(active.keys())

    for i, name_a in enumerate(emotion_names):
        for name_b in emotion_names[i + 1:]:
            # Both must be above coexistence threshold
            if (active[name_a] >= cfg.coexistence_threshold and
                active[name_b] >= cfg.coexistence_threshold):
                pairs.append((name_a, name_b, active[name_a], active[name_b]))

    return pairs


def has_conflicting_emotions(
    emotions: EmotionVector,
    config: Optional[MultiEmotionConfig] = None,
) -> bool:
    """
    Check if typically "opposing" emotions coexist.

    This is NOT for normalization; it's for detecting internal conflict
    that can affect judgment (e.g., "hesitation", "ambivalence").

    Opposing pairs (structural, not evaluative):
    - joy / sorrow
    - joy / fear
    - love / anger

    Args:
        emotions: Current EmotionVector
        config: Optional configuration

    Returns:
        True if opposing emotions are both active
    """
    cfg = config or MultiEmotionConfig()
    emo = emotions.as_dict()
    threshold = cfg.coexistence_threshold

    opposing_pairs = [
        ("joy", "sorrow"),
        ("joy", "fear"),
        ("love", "anger"),
    ]

    for a, b in opposing_pairs:
        if emo.get(a, 0) >= threshold and emo.get(b, 0) >= threshold:
            return True

    return False


def get_emotion_intensity(emotions: EmotionVector) -> float:
    """
    Get overall emotional intensity (sum of all emotions).

    This is NOT a normalization or averaging; it's the total
    emotional activation level.

    Args:
        emotions: Current EmotionVector

    Returns:
        Sum of all emotion values
    """
    return sum(emotions.as_dict().values())


def get_emotion_spread(emotions: EmotionVector) -> int:
    """
    Get the number of distinct active emotions.

    Higher spread = more emotions active simultaneously.
    This indicates emotional complexity, not a problem to solve.

    Args:
        emotions: Current EmotionVector

    Returns:
        Count of emotions above minimal threshold (0.05)
    """
    return sum(1 for v in emotions.as_dict().values() if v >= 0.05)


# ── Independent Decay ─────────────────────────────────────────────


def apply_independent_decay(
    emotions: EmotionVector,
    delta_time: float,
    config: Optional[MultiEmotionConfig] = None,
) -> EmotionVector:
    """
    Apply decay to each emotion INDEPENDENTLY.

    Each emotion decays at its own rate. No emotion's decay
    affects another emotion. No normalization.

    Args:
        emotions: Current EmotionVector
        delta_time: Time elapsed in seconds
        config: Optional configuration with per-emotion decay rates

    Returns:
        New EmotionVector with independently decayed values
    """
    cfg = config or MultiEmotionConfig()
    decay_cfg = cfg.decay_config

    emo = emotions.as_dict()
    new_emo = {}

    for emotion_name, value in emo.items():
        # Get decay rate for this specific emotion
        decay_rate = decay_cfg.get_decay_rate(emotion_name)

        # Exponential decay: value * (1 - rate) ^ time
        decay_factor = (1.0 - decay_rate) ** delta_time
        new_value = value * decay_factor

        # Clamp to valid range
        new_emo[emotion_name] = max(0.0, min(1.0, new_value))

    return EmotionVector(**new_emo)


def apply_independent_update(
    emotions: EmotionVector,
    updates: dict[str, float],
) -> EmotionVector:
    """
    Apply updates to specific emotions WITHOUT affecting others.

    This function explicitly does NOT normalize or reduce other emotions.
    If joy increases, sorrow is NOT decreased.

    Args:
        emotions: Current EmotionVector
        updates: Dict of emotion_name -> delta (change amount)

    Returns:
        New EmotionVector with updates applied independently
    """
    emo = emotions.as_dict()

    for emotion_name, delta in updates.items():
        if emotion_name in emo:
            new_value = emo[emotion_name] + delta
            emo[emotion_name] = max(0.0, min(1.0, new_value))

    return EmotionVector(**emo)


def set_emotions_independently(
    emotions: EmotionVector,
    values: dict[str, float],
) -> EmotionVector:
    """
    Set specific emotion values WITHOUT affecting others.

    Emotions not in 'values' remain unchanged.
    No normalization or balancing.

    Args:
        emotions: Current EmotionVector
        values: Dict of emotion_name -> new_value

    Returns:
        New EmotionVector with specified values set
    """
    emo = emotions.as_dict()

    for emotion_name, new_value in values.items():
        if emotion_name in emo:
            emo[emotion_name] = max(0.0, min(1.0, new_value))

    return EmotionVector(**emo)


# ── Reference Functions (Read-Only) ───────────────────────────────


def reference_emotions_for_judgment(
    emotions: EmotionVector,
    config: Optional[MultiEmotionConfig] = None,
) -> dict[str, Any]:
    """
    Reference multiple emotions for judgment/decision processes.

    This function provides a READ-ONLY view of emotions.
    It does NOT modify emotions or determine outcomes.
    It does NOT prioritize any emotion over another.

    Args:
        emotions: Current EmotionVector
        config: Optional configuration

    Returns:
        Dict with emotion reference data for external use
    """
    cfg = config or MultiEmotionConfig()
    active = get_active_emotions(emotions, cfg)
    all_emo = get_all_emotions(emotions)

    return {
        "all_emotions": all_emo,
        "active_emotions": active,
        "active_count": len(active),
        "total_intensity": get_emotion_intensity(emotions),
        "has_conflict": has_conflicting_emotions(emotions, cfg),
        "coexisting_pairs": get_coexisting_pairs(emotions, cfg),
    }


def reference_emotion_by_name(
    emotions: EmotionVector,
    emotion_name: str,
) -> float:
    """
    Reference a specific emotion's current value.

    This is a simple accessor; no computation or transformation.

    Args:
        emotions: Current EmotionVector
        emotion_name: Name of the emotion to reference

    Returns:
        Current value of the emotion (0.0 if not found)
    """
    return emotions.as_dict().get(emotion_name, 0.0)


def reference_multiple_emotions(
    emotions: EmotionVector,
    emotion_names: list[str],
) -> dict[str, float]:
    """
    Reference multiple specific emotions by name.

    No averaging, no aggregation. Just the raw values.

    Args:
        emotions: Current EmotionVector
        emotion_names: List of emotion names to reference

    Returns:
        Dict of emotion_name -> value for requested emotions
    """
    emo = emotions.as_dict()
    return {name: emo.get(name, 0.0) for name in emotion_names}


# ── Utility Functions ─────────────────────────────────────────────


def get_emotion_vector_summary(
    emotions: EmotionVector,
    config: Optional[MultiEmotionConfig] = None,
) -> str:
    """
    Get a human-readable summary of the emotion state.

    Shows ALL active emotions, not just the dominant one.

    Args:
        emotions: Current EmotionVector
        config: Optional configuration

    Returns:
        String summary
    """
    cfg = config or MultiEmotionConfig()
    active = get_active_emotions(emotions, cfg)

    if not active:
        return "EMOTIONS: calm (no active emotions)"

    parts = [f"{name}={value:.2f}" for name, value in sorted(active.items(), key=lambda x: -x[1])]

    conflict_marker = " [CONFLICT]" if has_conflicting_emotions(emotions, cfg) else ""

    return f"EMOTIONS: {', '.join(parts)}{conflict_marker}"


def create_multi_emotion_config(
    active_threshold: float = 0.1,
    coexistence_threshold: float = 0.15,
    base_decay_rate: float = 0.05,
    emotion_decay_rates: Optional[dict[str, float]] = None,
) -> MultiEmotionConfig:
    """Create a multi-emotion configuration with custom parameters."""
    decay_config = EmotionDecayConfig(
        base_decay_rate=base_decay_rate,
    )
    if emotion_decay_rates:
        decay_config.emotion_decay_rates.update(emotion_decay_rates)

    return MultiEmotionConfig(
        active_threshold=active_threshold,
        coexistence_threshold=coexistence_threshold,
        decay_config=decay_config,
    )


def to_dict(config: MultiEmotionConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "active_threshold": config.active_threshold,
        "coexistence_threshold": config.coexistence_threshold,
        "decay_config": {
            "base_decay_rate": config.decay_config.base_decay_rate,
            "emotion_decay_rates": config.decay_config.emotion_decay_rates.copy(),
            "inactive_threshold": config.decay_config.inactive_threshold,
        },
    }


def from_dict(data: dict[str, Any]) -> MultiEmotionConfig:
    """Deserialize config from dict."""
    decay_data = data.get("decay_config", {})
    decay_config = EmotionDecayConfig(
        base_decay_rate=decay_data.get("base_decay_rate", 0.05),
        emotion_decay_rates=decay_data.get("emotion_decay_rates", {}),
        inactive_threshold=decay_data.get("inactive_threshold", 0.05),
    )

    return MultiEmotionConfig(
        active_threshold=data.get("active_threshold", 0.1),
        coexistence_threshold=data.get("coexistence_threshold", 0.15),
        decay_config=decay_config,
    )
