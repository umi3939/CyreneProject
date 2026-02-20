"""
psyche/dynamics.py - Emotional Dynamics: Peak & Rebound System

Implements temporal emotional dynamics where:
- Emotions can enter a temporary "peak" state from accumulation
- Peak states transition to "rebound" phase with accelerated convergence
- Both phases resolve over time, preventing permanent extremes

This is a MODIFICATION LAYER on top of base emotion updates.
It does not replace core emotion logic, only modifies decay/convergence behavior.

Design principles:
- No hardcoded thresholds, durations, or intensities
- Phase transitions are pathway-only (conditions configured externally)
- All parameters are configurable via DynamicsConfig
- State survives persistence via to_dict/from_dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import time


class DynamicsPhase(Enum):
    """Current phase of emotional dynamics."""
    NORMAL = "normal"       # Standard emotion processing
    PEAK = "peak"           # Elevated emotional state
    REBOUND = "rebound"     # Post-peak convergence phase


@dataclass
class DynamicsConfig:
    """
    Configuration for emotional dynamics behavior.

    All parameters are externally configurable.
    Defaults are neutral (no effect) to avoid hardcoding behavior.
    """

    # Peak detection thresholds (configurable)
    peak_intensity_threshold: float = 1.0  # Emotion intensity to trigger peak
    peak_accumulation_threshold: float = 1.0  # Residue accumulation to trigger peak

    # Phase durations (in seconds, 0 = use turn count instead)
    peak_duration_seconds: float = 0.0
    rebound_duration_seconds: float = 0.0

    # Phase durations (in turns, 0 = use time instead)
    peak_duration_turns: int = 0
    rebound_duration_turns: int = 0

    # Decay modifiers during phases
    peak_decay_multiplier: float = 1.0      # Decay rate multiplier during peak
    rebound_decay_multiplier: float = 1.0   # Decay rate multiplier during rebound

    # Intensity modifiers
    peak_intensity_boost: float = 0.0       # Added to emotion changes during peak
    rebound_dampening: float = 0.0          # Reduces emotion changes during rebound

    # Custom transition functions (injectable)
    peak_trigger_function: Optional[Callable[["DynamicsState", dict], bool]] = None
    rebound_trigger_function: Optional[Callable[["DynamicsState"], bool]] = None


@dataclass
class DynamicsState:
    """
    Current state of emotional dynamics.

    Tracks which phase we're in and progress through it.
    Kept separate from PsycheState to maintain clean separation.
    """

    # Current phase
    phase: DynamicsPhase = DynamicsPhase.NORMAL

    # Phase entry tracking
    phase_entered_at: float = field(default_factory=time.time)
    phase_turn_count: int = 0

    # Peak state details (which emotion peaked, intensity)
    peak_emotion: str = ""
    peak_intensity: float = 0.0

    # Accumulation tracking (for peak detection)
    accumulated_intensity: float = 0.0
    intensity_history: list[float] = field(default_factory=list)
    max_history_length: int = 10

    # Configuration
    config: DynamicsConfig = field(default_factory=DynamicsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "phase": self.phase.value,
            "phase_entered_at": self.phase_entered_at,
            "phase_turn_count": self.phase_turn_count,
            "peak_emotion": self.peak_emotion,
            "peak_intensity": self.peak_intensity,
            "accumulated_intensity": self.accumulated_intensity,
            "intensity_history": self.intensity_history,
            "max_history_length": self.max_history_length,
            # Config is not persisted - it's set at runtime
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicsState":
        """Deserialize from persistence."""
        phase_str = data.get("phase", "normal")
        try:
            phase = DynamicsPhase(phase_str)
        except ValueError:
            phase = DynamicsPhase.NORMAL

        return cls(
            phase=phase,
            phase_entered_at=data.get("phase_entered_at", time.time()),
            phase_turn_count=data.get("phase_turn_count", 0),
            peak_emotion=data.get("peak_emotion", ""),
            peak_intensity=data.get("peak_intensity", 0.0),
            accumulated_intensity=data.get("accumulated_intensity", 0.0),
            intensity_history=data.get("intensity_history", []),
            max_history_length=data.get("max_history_length", 10),
        )


# ── Phase Transitions ──────────────────────────────────────────

def check_peak_trigger(
    state: DynamicsState,
    current_emotions: dict[str, float],
    residue_intensity: float = 0.0,
) -> tuple[bool, str, float]:
    """
    Check if conditions are met to enter peak phase.

    Returns:
        Tuple of (should_trigger, peak_emotion, peak_intensity)

    This provides the PATHWAY for peak transition.
    Actual trigger conditions depend on config thresholds.
    """
    # Use custom function if provided
    if state.config.peak_trigger_function is not None:
        context = {
            "emotions": current_emotions,
            "residue_intensity": residue_intensity,
        }
        if state.config.peak_trigger_function(state, context):
            # Find dominant emotion
            if current_emotions:
                peak_emo = max(current_emotions, key=current_emotions.get)
                return True, peak_emo, current_emotions[peak_emo]
            return True, "", 0.0
        return False, "", 0.0

    # Default logic: check intensity thresholds
    if not current_emotions:
        return False, "", 0.0

    max_emotion = max(current_emotions, key=current_emotions.get)
    max_intensity = current_emotions[max_emotion]

    # Check direct intensity threshold
    if max_intensity >= state.config.peak_intensity_threshold:
        return True, max_emotion, max_intensity

    # Check accumulation threshold
    if state.accumulated_intensity >= state.config.peak_accumulation_threshold:
        return True, max_emotion, max_intensity

    # Check residue-based trigger
    if residue_intensity >= state.config.peak_accumulation_threshold:
        return True, max_emotion, max_intensity

    return False, "", 0.0


def check_rebound_trigger(state: DynamicsState, current_time: Optional[float] = None) -> bool:
    """
    Check if peak phase should transition to rebound.

    Returns True if rebound should begin.
    """
    if state.phase != DynamicsPhase.PEAK:
        return False

    # Use custom function if provided
    if state.config.rebound_trigger_function is not None:
        return state.config.rebound_trigger_function(state)

    now = current_time if current_time is not None else time.time()

    # Check time-based duration
    if state.config.peak_duration_seconds > 0:
        elapsed = now - state.phase_entered_at
        if elapsed >= state.config.peak_duration_seconds:
            return True

    # Check turn-based duration
    if state.config.peak_duration_turns > 0:
        if state.phase_turn_count >= state.config.peak_duration_turns:
            return True

    return False


def check_normal_trigger(state: DynamicsState, current_time: Optional[float] = None) -> bool:
    """
    Check if rebound phase should transition back to normal.

    Returns True if normal state should resume.
    """
    if state.phase != DynamicsPhase.REBOUND:
        return False

    now = current_time if current_time is not None else time.time()

    # Check time-based duration
    if state.config.rebound_duration_seconds > 0:
        elapsed = now - state.phase_entered_at
        if elapsed >= state.config.rebound_duration_seconds:
            return True

    # Check turn-based duration
    if state.config.rebound_duration_turns > 0:
        if state.phase_turn_count >= state.config.rebound_duration_turns:
            return True

    return False


# ── State Transitions ──────────────────────────────────────────

def enter_peak(
    state: DynamicsState,
    peak_emotion: str,
    peak_intensity: float,
    current_time: Optional[float] = None,
) -> DynamicsState:
    """Transition to peak phase."""
    now = current_time if current_time is not None else time.time()

    return DynamicsState(
        phase=DynamicsPhase.PEAK,
        phase_entered_at=now,
        phase_turn_count=0,
        peak_emotion=peak_emotion,
        peak_intensity=peak_intensity,
        accumulated_intensity=0.0,  # Reset accumulation
        intensity_history=[],
        max_history_length=state.max_history_length,
        config=state.config,
    )


def enter_rebound(
    state: DynamicsState,
    current_time: Optional[float] = None,
) -> DynamicsState:
    """Transition from peak to rebound phase."""
    now = current_time if current_time is not None else time.time()

    return DynamicsState(
        phase=DynamicsPhase.REBOUND,
        phase_entered_at=now,
        phase_turn_count=0,
        peak_emotion=state.peak_emotion,  # Preserve for reference
        peak_intensity=state.peak_intensity,
        accumulated_intensity=0.0,
        intensity_history=[],
        max_history_length=state.max_history_length,
        config=state.config,
    )


def enter_normal(
    state: DynamicsState,
    current_time: Optional[float] = None,
) -> DynamicsState:
    """Transition back to normal phase."""
    now = current_time if current_time is not None else time.time()

    return DynamicsState(
        phase=DynamicsPhase.NORMAL,
        phase_entered_at=now,
        phase_turn_count=0,
        peak_emotion="",
        peak_intensity=0.0,
        accumulated_intensity=0.0,
        intensity_history=[],
        max_history_length=state.max_history_length,
        config=state.config,
    )


def increment_turn(state: DynamicsState) -> DynamicsState:
    """Increment turn counter for current phase."""
    return DynamicsState(
        phase=state.phase,
        phase_entered_at=state.phase_entered_at,
        phase_turn_count=state.phase_turn_count + 1,
        peak_emotion=state.peak_emotion,
        peak_intensity=state.peak_intensity,
        accumulated_intensity=state.accumulated_intensity,
        intensity_history=state.intensity_history,
        max_history_length=state.max_history_length,
        config=state.config,
    )


def update_accumulation(
    state: DynamicsState,
    intensity: float,
) -> DynamicsState:
    """Update intensity accumulation for peak detection."""
    # Add to history
    new_history = state.intensity_history.copy()
    new_history.append(intensity)
    if len(new_history) > state.max_history_length:
        new_history = new_history[-state.max_history_length:]

    # Calculate accumulated intensity (sum of recent history)
    accumulated = sum(new_history)

    return DynamicsState(
        phase=state.phase,
        phase_entered_at=state.phase_entered_at,
        phase_turn_count=state.phase_turn_count,
        peak_emotion=state.peak_emotion,
        peak_intensity=state.peak_intensity,
        accumulated_intensity=accumulated,
        intensity_history=new_history,
        max_history_length=state.max_history_length,
        config=state.config,
    )


# ── Decay Modification ─────────────────────────────────────────

def get_decay_modifier(state: DynamicsState) -> float:
    """
    Get decay rate modifier based on current phase.

    Returns a multiplier for the decay rate:
    - < 1.0: Slower decay (emotions persist longer)
    - = 1.0: Normal decay
    - > 1.0: Faster decay (emotions converge quicker)
    """
    if state.phase == DynamicsPhase.PEAK:
        return state.config.peak_decay_multiplier
    elif state.phase == DynamicsPhase.REBOUND:
        return state.config.rebound_decay_multiplier
    else:
        return 1.0


def get_intensity_modifier(state: DynamicsState) -> tuple[float, float]:
    """
    Get intensity modifiers based on current phase.

    Returns (boost, dampening) tuple:
    - boost: Added to emotion changes (positive in peak)
    - dampening: Subtracted from emotion changes (positive in rebound)
    """
    if state.phase == DynamicsPhase.PEAK:
        return state.config.peak_intensity_boost, 0.0
    elif state.phase == DynamicsPhase.REBOUND:
        return 0.0, state.config.rebound_dampening
    else:
        return 0.0, 0.0


# ── Main Update Function ───────────────────────────────────────

def update_dynamics(
    state: DynamicsState,
    current_emotions: dict[str, float],
    residue_intensity: float = 0.0,
    current_time: Optional[float] = None,
) -> DynamicsState:
    """
    Update dynamics state based on current emotions and residue.

    This is the main entry point for the dynamics system.
    Call this once per turn BEFORE applying decay modifiers.

    Flow:
    1. Update accumulation tracking
    2. Check phase transitions (normal→peak→rebound→normal)
    3. Increment turn counter
    4. Return updated state

    The caller should then use get_decay_modifier() and get_intensity_modifier()
    to adjust emotion processing.
    """
    now = current_time if current_time is not None else time.time()

    # Calculate current max intensity
    max_intensity = max(current_emotions.values()) if current_emotions else 0.0

    # Update accumulation
    state = update_accumulation(state, max_intensity)

    # Phase transitions
    if state.phase == DynamicsPhase.NORMAL:
        # Check for peak trigger
        should_peak, peak_emo, peak_int = check_peak_trigger(
            state, current_emotions, residue_intensity
        )
        if should_peak:
            state = enter_peak(state, peak_emo, peak_int, now)

    elif state.phase == DynamicsPhase.PEAK:
        # Check for rebound trigger
        if check_rebound_trigger(state, now):
            state = enter_rebound(state, now)

    elif state.phase == DynamicsPhase.REBOUND:
        # Check for normal trigger
        if check_normal_trigger(state, now):
            state = enter_normal(state, now)

    # Increment turn counter
    state = increment_turn(state)

    return state


# ── Convenience Functions ──────────────────────────────────────

def create_dynamics_state(config: Optional[DynamicsConfig] = None) -> DynamicsState:
    """Create a fresh dynamics state with optional configuration."""
    return DynamicsState(config=config or DynamicsConfig())


def get_dynamics_summary(state: DynamicsState) -> str:
    """Get human-readable summary of dynamics state."""
    phase_str = state.phase.value
    if state.phase == DynamicsPhase.PEAK:
        return f"PEAK ({state.peak_emotion}, intensity={state.peak_intensity:.2f}, turn={state.phase_turn_count})"
    elif state.phase == DynamicsPhase.REBOUND:
        return f"REBOUND (post-{state.peak_emotion}, turn={state.phase_turn_count})"
    else:
        return f"NORMAL (accumulated={state.accumulated_intensity:.2f})"


def apply_dynamics_to_decay(
    base_decay_rate: float,
    dynamics_state: DynamicsState,
) -> float:
    """
    Apply dynamics modifier to a decay rate.

    Args:
        base_decay_rate: Original decay rate (e.g., 0.95)
        dynamics_state: Current dynamics state

    Returns:
        Modified decay rate
    """
    modifier = get_decay_modifier(dynamics_state)

    if modifier == 1.0:
        return base_decay_rate

    # Adjust decay rate: higher modifier = faster decay = lower rate
    # decay_rate^modifier shifts the curve
    return base_decay_rate ** modifier
