"""
psyche/reaction_with_stm.py - Reaction with Short-Term Memory Integration

Extends the base reaction module to include the short-term emotional loop.
Maintains the existing reaction pipeline while adding:
- Short-term memory residue influence
- Context continuity handling
- Time-based decay

This module DOES NOT modify reaction.py - it wraps it.
Existing loss/fear related logic is preserved unchanged.

Pipeline position of short-term memory influence:
    [Percept] --> [STM Loop] --> [Residue Influence]
                                       |
                                       v
    [react()] --> [Emotion Stimulus] --> [+ Residue] --> [Decay] --> [Drives/Mood]

The residue influence is applied AFTER percept stimulus but BEFORE time decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

from .reaction import react, DECAY_RATE, _EMOTION_MAP
from .responsibility import ResponsibilityInfluence
from .state import EmotionVector, Percept, PsycheState
from .short_term_loop import (
    LoopConfig,
    LoopResult,
    LoopState,
    ResidueInfluence,
    create_loop_state,
    execute_full_loop,
    get_loop_diagnostics,
)


# ── Emotion Label Mapping ──────────────────────────────────────

# Maps short-term memory emotion labels to EmotionVector fields
# Using same mapping as reaction.py for consistency
STM_EMOTION_MAP: dict[str, str] = {
    "happy": "joy",
    "sad": "sorrow",
    "angry": "anger",
    "surprised": "surprise",
    "scared": "fear",
    "loving": "love",
    "teasing": "fun",
    "neutral": "",
    # Direct mappings for internal labels
    "joy": "joy",
    "sorrow": "sorrow",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "love": "love",
    "fun": "fun",
}


# ── Combined State ─────────────────────────────────────────────

@dataclass
class CombinedReactionState:
    """
    Combined state for reaction with short-term memory.

    Keeps PsycheState and LoopState as separate entities
    as required by the design constraint.
    """

    psyche: PsycheState
    loop: LoopState

    def to_dict(self) -> dict[str, Any]:
        return {
            "psyche": self.psyche.to_dict(),
            "loop": self.loop.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CombinedReactionState":
        return cls(
            psyche=PsycheState.from_dict(d.get("psyche", {})),
            loop=LoopState.from_dict(d.get("loop", {})),
        )


# ── Residue Application ────────────────────────────────────────

def apply_residue_to_emotions(
    emotions: EmotionVector,
    influence: ResidueInfluence,
    scale_override: Optional[float] = None,
) -> EmotionVector:
    """
    Apply short-term memory residue influence to emotions.

    This is the PATHWAY where residue affects emotions.
    The actual magnitude depends on:
    - influence.scale (from loop config)
    - scale_override (if provided, overrides influence.scale)
    - The raw values in influence.emotion_influences

    No hardcoded weights - everything flows from configuration.
    """
    emo = emotions.as_dict()
    scale = scale_override if scale_override is not None else influence.scale

    for label, raw_amount in influence.emotion_influences.items():
        # Map label to emotion field
        field = STM_EMOTION_MAP.get(label, label)
        if field and field in emo:
            # Apply scaled influence
            delta = raw_amount * scale
            emo[field] = max(0.0, min(1.0, emo[field] + delta))

    return EmotionVector(**emo)


# ── Extended Reaction ──────────────────────────────────────────

def react_with_stm(
    percept: Percept,
    psyche_state: PsycheState,
    loop_state: LoopState,
    delta_time: float = 1.0,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    residue_scale_override: Optional[float] = None,
    current_time: Optional[float] = None,
) -> tuple[PsycheState, LoopState, LoopResult]:
    """
    Extended reaction that includes short-term memory loop.

    This function:
    1. Executes the short-term memory loop (add stimulus, continuity, residue)
    2. Calls base react() for standard emotion processing
    3. Applies residue influence to emotions
    4. Returns updated states

    The residue influence is applied AFTER base emotion updates
    but the loop decay happens at the END of the pipeline.

    Args:
        percept: Interpreted stimulus.
        psyche_state: Current psychological state.
        loop_state: Current short-term memory loop state.
        delta_time: Seconds elapsed since last update.
        responsibility_influence: Optional responsibility influence.
        residue_scale_override: Optional override for residue influence scale.
        current_time: Optional current timestamp.

    Returns:
        Tuple of (new_psyche_state, new_loop_state, loop_result)
    """
    now = current_time if current_time is not None else time.time()

    # ── Step 1: Execute short-term memory loop ──
    loop_result = execute_full_loop(
        loop_state=loop_state,
        percept=percept,
        current_time=now,
    )

    # ── Step 2: Base reaction (existing logic unchanged) ──
    base_new_state = react(
        percept=percept,
        state=psyche_state,
        delta_time=delta_time,
        responsibility_influence=responsibility_influence,
    )

    # ── Step 3: Apply residue influence ──
    # This is where short-term memory affects emotions
    if loop_result.residue_influence.total_intensity > 0:
        modified_emotions = apply_residue_to_emotions(
            emotions=base_new_state.emotions,
            influence=loop_result.residue_influence,
            scale_override=residue_scale_override,
        )

        # Create new state with modified emotions
        final_state = PsycheState(
            emotions=modified_emotions,
            drives=base_new_state.drives,
            mood=base_new_state.mood,
            identity=base_new_state.identity,
            attachment=base_new_state.attachment,
            continuity=base_new_state.continuity,
            projection=base_new_state.projection,
            fear_index=base_new_state.fear_index,
            loss_aversion=base_new_state.loss_aversion,
            last_updated=base_new_state.last_updated,
        )
    else:
        final_state = base_new_state

    return final_state, loop_result.loop_state, loop_result


# ── Convenience Wrapper ────────────────────────────────────────

def create_combined_state(
    psyche: Optional[PsycheState] = None,
    loop_config: Optional[LoopConfig] = None,
) -> CombinedReactionState:
    """Create a fresh combined state."""
    return CombinedReactionState(
        psyche=psyche or PsycheState(),
        loop=create_loop_state(loop_config),
    )


def react_combined(
    percept: Percept,
    combined_state: CombinedReactionState,
    delta_time: float = 1.0,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    residue_scale_override: Optional[float] = None,
    current_time: Optional[float] = None,
) -> tuple[CombinedReactionState, LoopResult]:
    """
    Convenience wrapper for react_with_stm using combined state.

    Returns:
        Tuple of (new_combined_state, loop_result)
    """
    new_psyche, new_loop, loop_result = react_with_stm(
        percept=percept,
        psyche_state=combined_state.psyche,
        loop_state=combined_state.loop,
        delta_time=delta_time,
        responsibility_influence=responsibility_influence,
        residue_scale_override=residue_scale_override,
        current_time=current_time,
    )

    return CombinedReactionState(psyche=new_psyche, loop=new_loop), loop_result


# ── Diagnostics ────────────────────────────────────────────────

def get_stm_diagnostics(
    loop_state: LoopState,
    include_entries: bool = False,
) -> dict[str, Any]:
    """
    Get diagnostic information about the short-term memory state.

    Useful for debugging and understanding the loop behavior.
    """
    base_diag = get_loop_diagnostics(loop_state)

    if include_entries:
        base_diag["entries"] = [
            {
                "emotion": e.emotion_label,
                "valence": e.valence,
                "weight": e.residue_weight,
                "processed": e.processed,
                "topics": e.topics,
            }
            for e in loop_state.memory.entries
        ]

    return base_diag


def summarize_residue_influence(influence: ResidueInfluence) -> str:
    """
    Create human-readable summary of residue influence.

    Useful for logging and debugging.
    """
    if influence.total_intensity == 0:
        return "No residue influence"

    parts = []
    for label, amount in sorted(
        influence.emotion_influences.items(),
        key=lambda x: -x[1],
    ):
        if amount > 0.01:
            parts.append(f"{label}:{amount:.2f}")

    scaled = f"(scale={influence.scale:.2f})"
    continuity = f"continuity={influence.continuity:.2f}"

    return f"Residue: {', '.join(parts)} {scaled} {continuity}"
