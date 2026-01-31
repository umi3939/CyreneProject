"""
psyche - Artificial Mind Package for Cyrene.

Pipeline: perception → reaction → memory_link → thought → expression

Architecture: **Local Brain** handles all state/decisions.
Gemini is voice-only (parse_percept auxiliary + render_expression).

Pillar system: identity / attachment / continuity / projection → fear
Responsibility system: decisions → outcomes → psychological burden
Short-term memory loop: stimulus → residue → emotion influence → decay
"""

from .state import DriveVector, EmotionVector, Mood, Percept, PsycheState

from .pillars import (
    AttachmentState,
    ContinuityState,
    FearIndex,
    IdentityState,
    ProjectionState,
)

from .perception import parse_percept
from .reaction import react
from .memory_link import recall_with_mood
from .thought import generate_thought_candidates, select_policy
from .expression import render_expression

# Alias for backward compatibility
recall_by_mood = recall_with_mood

from .fear import compute_fear_index, fear_drive_boost, fear_emotion_boost

from .responsibility import (
    DecisionRecord,
    ResponsibilityState,
    ResponsibilityInfluence,
    record_decision,
    evaluate_outcome,
    apply_decay,
    get_influence,
    create_default_state as create_default_responsibility_state,
    to_dict as responsibility_to_dict,
    from_dict as responsibility_from_dict,
)
from .responsibility_manager import ResponsibilityManager

# Short-term memory loop (短期感情ループ)
from .short_term_memory import (
    ShortTermMemory,
    StimulusEntry,
    ResidueInfluence,
    compute_residue_influence,
)
from .short_term_loop import (
    LoopConfig,
    LoopState,
    LoopResult,
    create_loop_state,
    execute_full_loop,
    get_loop_diagnostics,
)
from .reaction_with_stm import (
    CombinedReactionState,
    react_with_stm,
    react_combined,
    create_combined_state,
    get_stm_diagnostics,
    summarize_residue_influence,
)

from . import attachment_manager
from . import continuity_manager
from . import identity_manager
from . import projection_manager

__all__ = [
    # Data models
    "PsycheState", "Percept", "EmotionVector", "DriveVector", "Mood",
    # Pillar states
    "IdentityState", "AttachmentState", "ContinuityState", "ProjectionState", "FearIndex",
    # Pipeline functions
    "parse_percept", "react", "recall_with_mood", "recall_by_mood",
    "generate_thought_candidates", "select_policy", "render_expression",
    # Fear functions
    "compute_fear_index", "fear_emotion_boost", "fear_drive_boost",
    # Responsibility (psychological burden from past decisions)
    "DecisionRecord", "ResponsibilityState", "ResponsibilityInfluence",
    "record_decision", "evaluate_outcome", "apply_decay", "get_influence",
    "create_default_responsibility_state", "responsibility_to_dict", "responsibility_from_dict",
    "ResponsibilityManager",
    # Short-term memory loop (短期感情ループ)
    "ShortTermMemory", "StimulusEntry", "ResidueInfluence", "compute_residue_influence",
    "LoopConfig", "LoopState", "LoopResult",
    "create_loop_state", "execute_full_loop", "get_loop_diagnostics",
    "CombinedReactionState", "react_with_stm", "react_combined",
    "create_combined_state", "get_stm_diagnostics", "summarize_residue_influence",
    # Pillar managers
    "attachment_manager", "continuity_manager", "identity_manager", "projection_manager",
]
