"""
psyche - Artificial Mind Package for Cyrene.

Pipeline: perception → reaction → memory_link → thought → expression

Architecture: **Local Brain** handles all state/decisions.
Gemini is voice-only (parse_percept auxiliary + render_expression).

Pillar system: identity / attachment / continuity / projection → fear
Responsibility system: decisions → outcomes → psychological burden
Short-term memory loop: stimulus → residue → emotion influence → decay
Persistence: atomic snapshot save/restore for continuity across restarts
Dynamics: peak & rebound emotional phases for non-monotonic emotion curves
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

# Persistence (永続化)
from .snapshot import (
    Snapshot,
    create_default_snapshot,
    validate_snapshot,
    SNAPSHOT_VERSION,
)
from .persistence import (
    PersistenceManager,
    save_snapshot,
    load_snapshot,
    restore_or_create,
    create_persistence_hooks,
)

# Dynamics (感情ピークと反動)
from .dynamics import (
    DynamicsPhase,
    DynamicsConfig,
    DynamicsState,
    create_dynamics_state,
    update_dynamics,
    get_decay_modifier,
    get_intensity_modifier,
    get_dynamics_summary,
    apply_dynamics_to_decay,
)

# Decision Bias (短期記憶由来の判断バイアス)
from .decision_bias import (
    DecisionBias,
    DecisionBiasConfig,
    compute_decision_bias,
    apply_bias_to_score,
    get_policy_bias_breakdown,
    create_neutral_bias,
    merge_biases,
)

# Self-Reference Loop (自己参照ループ)
from .self_reference import (
    SelfTag,
    SelfTagCategory,
    SelfReferenceConfig,
    SelfReferenceState,
    acquire_self_reference_targets,
    summarize_state,
    generate_self_tags,
    execute_self_reference,
    apply_self_tags_to_bias,
    get_self_reference_summary,
    create_self_reference_state,
)

# Responsibility Dispersion & Sublimation (責任の発散・昇華)
from .responsibility_dispersion import (
    ResponsibilityUnit,
    DispersionPlan,
    SublimationPath,
    TimeSlice,
    TimeDistributionPlan,
    DispersionState,
    DispersionConfig,
    AuditEntry,
    AuditEventType,
    DispersionStrategy,
    ConservationViolationError,
    create_responsibility_unit,
    disperse_responsibility,
    sublimate_responsibility,
    distribute_over_time,
    adjust_distance,
    merge_responsibilities,
    get_audit_trail,
    get_unit_by_id,
    get_active_units,
    get_total_active_weight,
    get_lineage,
    verify_state_conservation,
    create_dispersion_state,
    get_dispersion_summary,
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
    # Persistence (永続化)
    "Snapshot", "create_default_snapshot", "validate_snapshot", "SNAPSHOT_VERSION",
    "PersistenceManager", "save_snapshot", "load_snapshot", "restore_or_create",
    "create_persistence_hooks",
    # Dynamics (感情ピークと反動)
    "DynamicsPhase", "DynamicsConfig", "DynamicsState",
    "create_dynamics_state", "update_dynamics",
    "get_decay_modifier", "get_intensity_modifier",
    "get_dynamics_summary", "apply_dynamics_to_decay",
    # Decision Bias (短期記憶由来の判断バイアス)
    "DecisionBias", "DecisionBiasConfig",
    "compute_decision_bias", "apply_bias_to_score",
    "get_policy_bias_breakdown", "create_neutral_bias", "merge_biases",
    # Self-Reference Loop (自己参照ループ)
    "SelfTag", "SelfTagCategory", "SelfReferenceConfig", "SelfReferenceState",
    "acquire_self_reference_targets", "summarize_state", "generate_self_tags",
    "execute_self_reference", "apply_self_tags_to_bias",
    "get_self_reference_summary", "create_self_reference_state",
    # Responsibility Dispersion & Sublimation (責任の発散・昇華)
    "ResponsibilityUnit", "DispersionPlan", "SublimationPath",
    "TimeSlice", "TimeDistributionPlan", "DispersionState", "DispersionConfig",
    "AuditEntry", "AuditEventType", "DispersionStrategy", "ConservationViolationError",
    "create_responsibility_unit", "disperse_responsibility", "sublimate_responsibility",
    "distribute_over_time", "adjust_distance", "merge_responsibilities",
    "get_audit_trail", "get_unit_by_id", "get_active_units", "get_total_active_weight",
    "get_lineage", "verify_state_conservation", "create_dispersion_state",
    "get_dispersion_summary",
    # Pillar managers
    "attachment_manager", "continuity_manager", "identity_manager", "projection_manager",
]
