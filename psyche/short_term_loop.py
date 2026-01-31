"""
psyche/short_term_loop.py - Short-Term Emotional Loop Pipeline

Orchestrates the complete short-term emotional loop:
1. Receive stimulus and add to short-term memory
2. Detect context continuity
3. Compute residue influence
4. Apply influence to emotion update (via reaction.py)
5. Apply decay to short-term memory

This module defines the STRUCTURE and PIPELINE only.
No specific reaction values, weights, or thresholds are hardcoded.

Pipeline flow:
    [Input] --> [Add Stimulus] --> [Continuity Check] --> [Residue Computation]
                                                              |
                                                              v
    [Decayed Memory] <-- [Decay] <-- [Mark Processed] <-- [Emotion Update]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
import time

from .short_term_memory import (
    ShortTermMemory,
    StimulusEntry,
    ResidueInfluence,
    compute_residue_influence,
)
from .state import EmotionVector, Percept, PsycheState


# ── Pipeline Configuration ─────────────────────────────────────

@dataclass
class LoopConfig:
    """
    Configuration for the short-term emotional loop.

    All parameters are externally configurable.
    No hardcoded values - defaults are neutral (1.0 multipliers, 0.0 thresholds).
    """

    # Continuity detection
    continuity_threshold: float = 0.0  # Overlap score threshold for "continuous"

    # Residue influence
    residue_scale: float = 1.0  # Multiplier for residue influence

    # Decay parameters
    decay_rate: float = 1.0  # Base decay rate (passed to decay function)
    min_residue_weight: float = 0.001  # Minimum weight before removal

    # Accumulation
    max_accumulation_entries: int = 10  # Max entries in short-term memory

    # Custom functions (injectable)
    continuity_function: Optional[Callable[[float], bool]] = None
    decay_function: Optional[Callable[[float, float], float]] = None
    influence_function: Optional[Callable[[ResidueInfluence, EmotionVector], EmotionVector]] = None


# ── Pipeline State ─────────────────────────────────────────────

@dataclass
class LoopState:
    """
    Complete state of the short-term emotional loop.

    Kept separate from PsycheState to maintain clear separation.
    """

    memory: ShortTermMemory = field(default_factory=ShortTermMemory)
    config: LoopConfig = field(default_factory=LoopConfig)

    # Track if update has been applied this turn
    updated_this_turn: bool = False

    # Timestamp of last loop execution
    last_loop_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "memory": self.memory.to_dict(),
            "config": {
                "continuity_threshold": self.config.continuity_threshold,
                "residue_scale": self.config.residue_scale,
                "decay_rate": self.config.decay_rate,
                "min_residue_weight": self.config.min_residue_weight,
                "max_accumulation_entries": self.config.max_accumulation_entries,
            },
            "updated_this_turn": self.updated_this_turn,
            "last_loop_time": self.last_loop_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopState":
        """Deserialize from dict."""
        config_data = d.get("config", {})
        config = LoopConfig(
            continuity_threshold=config_data.get("continuity_threshold", 0.0),
            residue_scale=config_data.get("residue_scale", 1.0),
            decay_rate=config_data.get("decay_rate", 1.0),
            min_residue_weight=config_data.get("min_residue_weight", 0.001),
            max_accumulation_entries=config_data.get("max_accumulation_entries", 10),
        )
        return cls(
            memory=ShortTermMemory.from_dict(d.get("memory", {})),
            config=config,
            updated_this_turn=d.get("updated_this_turn", False),
            last_loop_time=d.get("last_loop_time", time.time()),
        )


# ── Pipeline Execution ─────────────────────────────────────────

@dataclass
class LoopResult:
    """Result of executing one iteration of the short-term loop."""

    # Updated loop state
    loop_state: LoopState

    # Computed residue influence (for emotion update phase)
    residue_influence: ResidueInfluence

    # Whether context was detected as continuous
    is_continuous: bool

    # Delta emotions suggested by residue (not applied, just computed)
    suggested_emotion_delta: dict[str, float] = field(default_factory=dict)


def detect_continuity(
    memory: ShortTermMemory,
    new_topics: list[str],
    config: LoopConfig,
) -> bool:
    """
    Detect whether new input is contextually continuous with memory.

    Returns True if continuous, False if discontinuous.
    Uses config.continuity_function if provided, otherwise uses threshold.
    """
    overlap = memory.compute_context_overlap(new_topics)

    if config.continuity_function is not None:
        return config.continuity_function(overlap)

    # Default: compare overlap to threshold
    return overlap >= config.continuity_threshold


def execute_loop_phase1_stimulus(
    loop_state: LoopState,
    percept: Percept,
) -> LoopState:
    """
    Phase 1: Add stimulus to short-term memory.

    Extracts stimulus information from Percept and stores it.
    Does not interpret or process the stimulus.
    """
    memory = loop_state.memory.add_stimulus(
        source_text=percept.text,
        topics=percept.topics,
        emotion_label=percept.emotion,
        intent=percept.intent,
        raw_intensity=abs(percept.emotion_valence),
        valence=percept.emotion_valence,
    )

    return LoopState(
        memory=memory,
        config=loop_state.config,
        updated_this_turn=False,
        last_loop_time=loop_state.last_loop_time,
    )


def execute_loop_phase2_continuity(
    loop_state: LoopState,
    percept: Percept,
) -> tuple[LoopState, bool]:
    """
    Phase 2: Detect and update context continuity.

    Returns:
        Tuple of (updated_loop_state, is_continuous)
    """
    is_continuous = detect_continuity(
        loop_state.memory,
        percept.topics,
        loop_state.config,
    )

    # Update context based on continuity
    if is_continuous:
        memory = loop_state.memory.update_context(percept.topics, is_continuous=True)
    else:
        # On discontinuity, decide whether to reset or just update
        # This is configurable behavior - by default we just update context
        memory = loop_state.memory.update_context(percept.topics, is_continuous=False)

    return LoopState(
        memory=memory,
        config=loop_state.config,
        updated_this_turn=loop_state.updated_this_turn,
        last_loop_time=loop_state.last_loop_time,
    ), is_continuous


def execute_loop_phase3_residue(
    loop_state: LoopState,
) -> ResidueInfluence:
    """
    Phase 3: Compute residue influence from short-term memory.

    Returns ResidueInfluence structure for use in emotion update.
    """
    return compute_residue_influence(
        loop_state.memory,
        scale_factor=loop_state.config.residue_scale,
    )


def execute_loop_phase4_mark_processed(
    loop_state: LoopState,
) -> LoopState:
    """
    Phase 4: Mark all entries as processed after emotion update.

    This prevents double-processing of stimuli.
    """
    memory = loop_state.memory.mark_processed()

    return LoopState(
        memory=memory,
        config=loop_state.config,
        updated_this_turn=True,
        last_loop_time=loop_state.last_loop_time,
    )


def execute_loop_phase5_decay(
    loop_state: LoopState,
    current_time: Optional[float] = None,
) -> LoopState:
    """
    Phase 5: Apply decay to short-term memory.

    This is the FINAL phase of the loop, executed once per turn.
    """
    # Prepare decay function if custom one provided
    decay_func = loop_state.config.decay_function
    if decay_func is None:
        # Create default decay function using config rate
        rate = loop_state.config.decay_rate

        def decay_func(weight: float, elapsed: float) -> float:
            # Clamp elapsed to prevent overflow with extreme values
            clamped_elapsed = min(elapsed, 100.0)
            try:
                return weight * (rate ** clamped_elapsed)
            except OverflowError:
                return 0.0

    # Update memory scale factors for decay
    memory = loop_state.memory
    memory.scale_factors["decay_base"] = loop_state.config.decay_rate
    memory.scale_factors["min_residue_weight"] = loop_state.config.min_residue_weight

    # Apply decay
    decayed_memory = memory.apply_decay(
        decay_function=decay_func,
        current_time=current_time,
    )

    now = current_time if current_time is not None else time.time()

    return LoopState(
        memory=decayed_memory,
        config=loop_state.config,
        updated_this_turn=False,  # Reset for next turn
        last_loop_time=now,
    )


def execute_full_loop(
    loop_state: LoopState,
    percept: Percept,
    current_time: Optional[float] = None,
) -> LoopResult:
    """
    Execute the complete short-term emotional loop.

    Pipeline:
    1. Add stimulus to memory
    2. Detect context continuity
    3. Compute residue influence
    4. Mark entries as processed
    5. Apply decay

    Note: This does NOT apply the influence to emotions.
    The caller (reaction.py integration) is responsible for that.
    This maintains clear separation of concerns.

    Returns:
        LoopResult containing updated state and computed influence.
    """
    # Guard: only one update per turn
    if loop_state.updated_this_turn:
        # Return current state without modification
        return LoopResult(
            loop_state=loop_state,
            residue_influence=ResidueInfluence(),
            is_continuous=True,
            suggested_emotion_delta={},
        )

    # Phase 1: Add stimulus
    state1 = execute_loop_phase1_stimulus(loop_state, percept)

    # Phase 2: Continuity detection
    state2, is_continuous = execute_loop_phase2_continuity(state1, percept)

    # Phase 3: Compute residue influence
    residue = execute_loop_phase3_residue(state2)

    # Phase 4: Mark processed
    state4 = execute_loop_phase4_mark_processed(state2)

    # Phase 5: Decay
    state5 = execute_loop_phase5_decay(state4, current_time)

    return LoopResult(
        loop_state=state5,
        residue_influence=residue,
        is_continuous=is_continuous,
        suggested_emotion_delta=residue.emotion_influences,
    )


# ── Integration Interface ──────────────────────────────────────

class ResidueApplicator(Protocol):
    """Protocol for applying residue influence to emotions."""

    def __call__(
        self,
        emotions: EmotionVector,
        influence: ResidueInfluence,
    ) -> EmotionVector:
        """Apply residue influence to emotions and return new EmotionVector."""
        ...


def create_default_applicator(
    emotion_mapping: Optional[dict[str, str]] = None,
) -> ResidueApplicator:
    """
    Create a default residue applicator.

    This provides the PATHWAY for influence without fixing specific values.
    The actual impact depends on:
    - The scale factor in ResidueInfluence
    - The raw intensities in short-term memory
    - Any external configuration

    Args:
        emotion_mapping: Maps stimulus emotion labels to EmotionVector fields.
                         If None, uses identity mapping (label = field name).
    """
    mapping = emotion_mapping or {}

    def applicator(
        emotions: EmotionVector,
        influence: ResidueInfluence,
    ) -> EmotionVector:
        emo = emotions.as_dict()

        for label, amount in influence.emotion_influences.items():
            # Map label to emotion field
            field = mapping.get(label, label)
            if field in emo:
                # Apply scaled influence
                delta = amount * influence.scale
                emo[field] = max(0.0, min(1.0, emo[field] + delta))

        return EmotionVector(**emo)

    return applicator


# ── Convenience Functions ──────────────────────────────────────

def create_loop_state(
    config: Optional[LoopConfig] = None,
) -> LoopState:
    """Create a fresh loop state with optional configuration."""
    return LoopState(
        memory=ShortTermMemory(
            max_entries=config.max_accumulation_entries if config else 10,
            scale_factors={
                "residue_influence": config.residue_scale if config else 1.0,
                "decay_base": config.decay_rate if config else 1.0,
                "min_residue_weight": config.min_residue_weight if config else 0.001,
            },
        ),
        config=config or LoopConfig(),
    )


def get_loop_diagnostics(loop_state: LoopState) -> dict[str, Any]:
    """Get diagnostic information about the loop state."""
    memory = loop_state.memory
    return {
        "entry_count": len(memory.entries),
        "unprocessed_count": len(memory.get_unprocessed_residue()),
        "current_topics": memory.current_context_topics,
        "continuity_score": memory.context_continuity_score,
        "last_update": memory.last_update_time,
        "config": {
            "continuity_threshold": loop_state.config.continuity_threshold,
            "residue_scale": loop_state.config.residue_scale,
            "decay_rate": loop_state.config.decay_rate,
        },
    }
