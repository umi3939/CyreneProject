"""
psyche/emotion_amplitude.py - Emotion Amplitude Expansion (感情振れ幅拡張)

Introduces an amplitude modifier that scales emotion change amounts (deltas)
without changing the direction or adding new emotion types.

Design principles (from design_emotion_amplitude.md):
- Amplitude scales the DELTA (change amount), not the absolute value
- Amplitude does NOT change the sign/direction of emotion updates
- Amplitude is TEMPORARY and decays over time
- No new emotion categories are added
- No hardcoded values; all parameters are configurable
- Acts as a modifier layer, not a replacement for emotion logic

Usage::

    from psyche.emotion_amplitude import (
        AmplitudeState,
        AmplitudeConfig,
        apply_amplitude_to_delta,
        update_amplitude,
        decay_amplitude,
    )

    # Create amplitude state
    amp_state = AmplitudeState()

    # Apply amplitude to emotion delta
    scaled_delta = apply_amplitude_to_delta(
        base_delta=0.2,
        amplitude=amp_state.current_amplitude,
    )

    # Update amplitude based on context
    amp_state = update_amplitude(amp_state, intensity_factor=0.5)

    # Decay amplitude over time
    amp_state = decay_amplitude(amp_state, delta_time=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AmplitudeConfig:
    """
    Configuration for emotion amplitude behavior.

    All parameters are externally configurable.
    No hardcoded thresholds or meanings.
    """

    # Base amplitude (1.0 = no change, >1 = amplify, <1 = dampen)
    base_amplitude: float = 1.0

    # Minimum and maximum amplitude bounds
    min_amplitude: float = 0.5
    max_amplitude: float = 2.0

    # Decay rate per second (amplitude approaches base over time)
    decay_rate: float = 0.1

    # Scale factor for amplitude increases
    increase_scale: float = 1.0

    # Custom amplitude calculation function (injectable)
    amplitude_function: Optional[Callable[[float, float], float]] = None


@dataclass
class AmplitudeState:
    """
    Current amplitude state.

    Tracks the current amplitude modifier and its temporal behavior.
    Amplitude is temporary and decays toward base over time.
    """

    # Current amplitude modifier (1.0 = neutral)
    current_amplitude: float = 1.0

    # Accumulated amplitude from recent updates
    accumulated_boost: float = 0.0

    # Turn counter for tracking updates
    update_count: int = 0

    # Configuration
    config: AmplitudeConfig = field(default_factory=AmplitudeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_amplitude": self.current_amplitude,
            "accumulated_boost": self.accumulated_boost,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: Optional[AmplitudeConfig] = None,
    ) -> "AmplitudeState":
        return cls(
            current_amplitude=data.get("current_amplitude", 1.0),
            accumulated_boost=data.get("accumulated_boost", 0.0),
            update_count=data.get("update_count", 0),
            config=config or AmplitudeConfig(),
        )


def apply_amplitude_to_delta(
    base_delta: float,
    amplitude: float,
    config: Optional[AmplitudeConfig] = None,
) -> float:
    """
    Apply amplitude modifier to an emotion delta (change amount).

    This function:
    - Scales the magnitude of the delta by the amplitude
    - PRESERVES the sign (direction) of the delta
    - Does NOT change the emotion type or add new types

    Args:
        base_delta: The original change amount (can be positive or negative)
        amplitude: The amplitude modifier (1.0 = no change)
        config: Optional configuration for bounds

    Returns:
        Scaled delta with preserved sign
    """
    cfg = config or AmplitudeConfig()

    # Clamp amplitude to valid range
    clamped_amplitude = max(cfg.min_amplitude, min(cfg.max_amplitude, amplitude))

    # Apply amplitude to delta magnitude while preserving sign
    # sign * |delta| * amplitude
    if base_delta == 0:
        return 0.0

    sign = 1.0 if base_delta > 0 else -1.0
    magnitude = abs(base_delta)
    scaled_magnitude = magnitude * clamped_amplitude

    return sign * scaled_magnitude


def apply_amplitude_to_emotion_deltas(
    deltas: dict[str, float],
    amplitude: float,
    config: Optional[AmplitudeConfig] = None,
) -> dict[str, float]:
    """
    Apply amplitude modifier to multiple emotion deltas.

    Args:
        deltas: Dict mapping emotion names to their deltas
        amplitude: The amplitude modifier (1.0 = no change)
        config: Optional configuration

    Returns:
        Dict with scaled deltas, signs preserved
    """
    return {
        emotion: apply_amplitude_to_delta(delta, amplitude, config)
        for emotion, delta in deltas.items()
    }


def update_amplitude(
    state: AmplitudeState,
    intensity_factor: float = 0.0,
    emotion_intensity: float = 0.0,
    context_boost: float = 0.0,
    config: Optional[AmplitudeConfig] = None,
) -> AmplitudeState:
    """
    Update amplitude based on current context.

    Amplitude can be increased by:
    - intensity_factor: Direct intensity input
    - emotion_intensity: Current emotional intensity
    - context_boost: External context modifier

    Args:
        state: Current amplitude state
        intensity_factor: Direct intensity contribution
        emotion_intensity: Current max emotion value
        context_boost: Additional context-based boost
        config: Optional configuration override

    Returns:
        New AmplitudeState with updated amplitude
    """
    cfg = config or state.config

    # Use custom function if provided
    if cfg.amplitude_function is not None:
        new_amplitude = cfg.amplitude_function(state.current_amplitude, intensity_factor)
    else:
        # Default: accumulate boost toward amplitude
        total_boost = (
            intensity_factor * cfg.increase_scale +
            emotion_intensity * cfg.increase_scale * 0.5 +
            context_boost
        )

        # Accumulated boost affects amplitude
        new_accumulated = state.accumulated_boost + total_boost

        # Convert accumulated boost to amplitude adjustment
        # Higher accumulated boost = higher amplitude
        adjustment = new_accumulated * 0.1
        new_amplitude = cfg.base_amplitude + adjustment

    # Clamp to valid range
    new_amplitude = max(cfg.min_amplitude, min(cfg.max_amplitude, new_amplitude))

    return AmplitudeState(
        current_amplitude=new_amplitude,
        accumulated_boost=state.accumulated_boost + intensity_factor * cfg.increase_scale,
        update_count=state.update_count + 1,
        config=cfg,
    )


def decay_amplitude(
    state: AmplitudeState,
    delta_time: float = 1.0,
    config: Optional[AmplitudeConfig] = None,
) -> AmplitudeState:
    """
    Decay amplitude toward base value over time.

    Amplitude naturally returns to base_amplitude, ensuring
    that amplitude effects are temporary, not permanent.

    Args:
        state: Current amplitude state
        delta_time: Time elapsed in seconds
        config: Optional configuration override

    Returns:
        New AmplitudeState with decayed amplitude
    """
    cfg = config or state.config

    # Exponential decay toward base amplitude
    decay_factor = (1.0 - cfg.decay_rate) ** delta_time

    # Current deviation from base
    deviation = state.current_amplitude - cfg.base_amplitude

    # Decay the deviation
    new_deviation = deviation * decay_factor
    new_amplitude = cfg.base_amplitude + new_deviation

    # Also decay accumulated boost
    new_accumulated = state.accumulated_boost * decay_factor

    # Clamp to valid range
    new_amplitude = max(cfg.min_amplitude, min(cfg.max_amplitude, new_amplitude))

    return AmplitudeState(
        current_amplitude=new_amplitude,
        accumulated_boost=new_accumulated,
        update_count=state.update_count,
        config=cfg,
    )


def compute_amplitude_from_dynamics(
    dynamics_state: Any,
    base_amplitude: float = 1.0,
    peak_boost: float = 0.3,
    rebound_reduction: float = 0.2,
) -> float:
    """
    Compute amplitude modifier from dynamics state (peak/rebound phases).

    During peak phase: amplitude is boosted
    During rebound phase: amplitude is reduced

    Args:
        dynamics_state: DynamicsState instance
        base_amplitude: Base amplitude value
        peak_boost: Additional amplitude during peak
        rebound_reduction: Amplitude reduction during rebound

    Returns:
        Computed amplitude value
    """
    if dynamics_state is None:
        return base_amplitude

    phase = getattr(dynamics_state, "phase", None)
    if phase is None:
        return base_amplitude

    phase_value = phase.value if hasattr(phase, "value") else str(phase)

    if phase_value == "peak":
        return base_amplitude + peak_boost
    elif phase_value == "rebound":
        return base_amplitude - rebound_reduction
    else:
        return base_amplitude


def compute_amplitude_from_residue(
    short_term_memory: Any,
    base_amplitude: float = 1.0,
    residue_scale: float = 0.5,
) -> float:
    """
    Compute amplitude modifier from short-term memory residue.

    Higher residue intensity = higher amplitude (emotions react more strongly)

    Args:
        short_term_memory: ShortTermMemory instance
        base_amplitude: Base amplitude value
        residue_scale: How much residue affects amplitude

    Returns:
        Computed amplitude value
    """
    if short_term_memory is None:
        return base_amplitude

    # Compute residue intensity
    residue_intensity = 0.0
    if hasattr(short_term_memory, "get_unprocessed_residue"):
        for entry in short_term_memory.get_unprocessed_residue():
            residue_intensity += getattr(entry, "residue_weight", 0) * getattr(entry, "raw_intensity", 0)

    return base_amplitude + residue_intensity * residue_scale


def create_amplitude_state(
    config: Optional[AmplitudeConfig] = None,
) -> AmplitudeState:
    """Create initial amplitude state."""
    return AmplitudeState(config=config or AmplitudeConfig())


def get_amplitude_summary(state: AmplitudeState) -> str:
    """Get human-readable summary of amplitude state."""
    amp = state.current_amplitude
    if amp > 1.1:
        level = "amplified"
    elif amp < 0.9:
        level = "dampened"
    else:
        level = "neutral"

    return f"AMPLITUDE: {amp:.2f} ({level}), boost={state.accumulated_boost:.2f}"
