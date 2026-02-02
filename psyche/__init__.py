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
    # Responsibility distribution summary (自己参照による責任要約)
    ResponsibilityDistributionSummary,
    summarize_responsibility_units,
    generate_responsibility_distribution_tags,
)

# Emotion Amplitude Expansion (感情振れ幅拡張)
from .emotion_amplitude import (
    AmplitudeState,
    AmplitudeConfig,
    apply_amplitude_to_delta,
    apply_amplitude_to_emotion_deltas,
    update_amplitude,
    decay_amplitude,
    compute_amplitude_from_dynamics,
    compute_amplitude_from_residue,
    create_amplitude_state,
    get_amplitude_summary,
)

# Multi-Emotion Reference (複数感情参照)
from .multi_emotion import (
    EmotionDecayConfig,
    MultiEmotionConfig,
    get_active_emotions,
    get_all_emotions,
    get_coexisting_pairs,
    has_conflicting_emotions,
    get_emotion_intensity,
    get_emotion_spread,
    apply_independent_decay,
    apply_independent_update,
    set_emotions_independently,
    reference_emotions_for_judgment,
    reference_emotion_by_name,
    reference_multiple_emotions,
    get_emotion_vector_summary,
    create_multi_emotion_config,
    to_dict as multi_emotion_config_to_dict,
    from_dict as multi_emotion_config_from_dict,
)

# STM-Emotion Coupling (短期記憶の感情連動)
from .stm_emotion_coupling import (
    STMEmotionCouplingConfig,
    EmotionCouplingData,
    CouplingInfluence,
    compute_coupling_influence,
    compute_decay_modifier_from_stm,
    apply_persistence_modifier,
    apply_reactivation,
    apply_reactivation_to_existing,
    apply_accumulation,
    apply_stm_coupling,
    get_coupling_summary,
    get_emotion_persistence_breakdown,
    create_coupling_config,
    to_dict as stm_coupling_config_to_dict,
    from_dict as stm_coupling_config_from_dict,
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

# Silence / Hesitation as Choice (沈黙・躊躇の選択)
from .silence_hesitation import (
    SilenceType,
    SilenceConfig,
    SilenceCandidate,
    SilenceResult,
    SilenceState,
    generate_silence_candidate,
    evaluate_silence_score,
    create_silence_result,
    create_speech_result,
    silence_candidate_to_policy,
    is_silence_policy,
    is_silence_result,
    get_silence_duration,
    generate_candidates_with_silence,
    get_silence_summary,
    create_silence_config,
    to_dict as silence_config_to_dict,
    from_dict as silence_config_from_dict,
)

# Tone / Light-Tone Mode (冗談・軽口モード)
from .tone import (
    Tone,
    ToneConfig,
    ToneModifier,
    ToneState,
    compute_tone_bias,
    apply_tone_to_candidate,
    get_candidate_tone,
    select_candidate_tone,
    generate_tone_variants,
    add_tone_to_candidates,
    apply_tone_to_silence,
    get_tone_summary,
    get_tone_from_candidate,
    is_light_tone,
    is_serious_tone,
    create_tone_config,
    to_dict as tone_config_to_dict,
    from_dict as tone_config_from_dict,
)

# Long-Term Dynamics (長期挙動ログ)
from .long_term_dynamics import (
    EmotionStats,
    DecisionStats,
    ValueOrientationStats,
    ResponsibilityStats,
    StabilityValveStats,
    WindowStats,
    LongTermEntry,
    DynamicsObserverConfig,
    DynamicsObserver,
    create_observer,
    create_config as create_observer_config,
    get_observer_summary,
    entries_to_json,
    entries_from_json,
)

# Stability Valve (極端回避防止)
from .stability_valve import (
    ExtremityIndicators,
    StabilityBias,
    StabilityValveConfig,
    StabilityValve,
    create_neutral_bias as create_neutral_stability_bias,
    flatten_scores,
    apply_stability_to_candidate,
    apply_stability_bias,
    create_stability_factor,
    get_stability_trace_context,
    create_stability_valve,
    create_config as create_stability_config,
    observe_extremity,
    get_stability_summary,
    to_dict as stability_config_to_dict,
    from_dict as stability_config_from_dict,
)

# Introspection Trace (内省ログ生成)
from .introspection_trace import (
    InfluenceDirection,
    FactorCategory,
    OutcomeType,
    ContributingFactor,
    EmotionSnapshot,
    ResponsibilitySnapshot,
    ValueOrientationSnapshot,
    DecisionSnapshot,
    TraceLog,
    IntrospectionConfig,
    IntrospectionSystem,
    generate_trace,
    create_introspection_system,
    create_config as create_introspection_config,
    get_trace_summary,
    traces_to_json,
    traces_from_json,
)

# Value Orientation (一貫した価値軸)
from .value_orientation import (
    ValueOrientation,
    ValueOrientationConfig,
    OrientationBias,
    compute_effective_learning_rate,
    update_dimension,
    update_orientation,
    generate_decision_signal,
    update_from_decision,
    compute_orientation_bias,
    apply_orientation_to_candidate,
    apply_orientation_to_candidates,
    generate_emotion_signal,
    generate_responsibility_signal,
    get_orientation_summary,
    get_orientation_vector,
    compute_orientation_distance,
    is_orientation_stable,
    create_orientation,
    create_config as create_orientation_config,
    to_dict as orientation_config_to_dict,
    from_dict as orientation_config_from_dict,
)

# Context Sensitivity (空気読みバイアス)
from .context_sensitivity import (
    ExternalContext,
    SensitivityBias,
    ContextSensitivityConfig,
    ContextState,
    create_external_context,
    create_neutral_context,
    create_heavy_context,
    create_light_context,
    create_neutral_bias as create_neutral_sensitivity_bias,
    compute_sensitivity_bias,
    get_policy_risk,
    apply_sensitivity_to_candidate,
    apply_sensitivity_to_candidates,
    process_with_context_sensitivity,
    get_sensitivity_summary,
    get_context_summary,
    is_high_caution,
    is_low_caution,
    create_config as create_sensitivity_config,
    to_dict as sensitivity_config_to_dict,
    from_dict as sensitivity_config_from_dict,
)

# Proto-Goal Direction Vector (自発的方向ベクトル)
from .proto_goal_vector import (
    VectorSourceType,
    VectorSource,
    ProtoGoalVector,
    VectorStateConfig,
    VectorState,
    VectorGenerator,
    create_vector_generator,
    create_config as create_vector_config,
    get_vector_summary,
    vectors_to_json,
    vectors_from_json,
    create_vector_context_for_trace,
    to_dict as vector_config_to_dict,
    from_dict as vector_config_from_dict,
)

# Goal Candidates (自発的目的候補)
from .goal_candidates import (
    CandidateCategory,
    CandidateSource,
    GoalCandidate,
    CandidateStateConfig,
    CandidateState,
    CandidateGenerator,
    create_candidate_generator,
    create_config as create_candidate_config,
    get_candidate_summary,
    candidates_to_json,
    candidates_from_json,
    create_candidate_context_for_trace,
    create_candidate_stats_for_dynamics,
    to_dict as candidate_config_to_dict,
    from_dict as candidate_config_from_dict,
)

# Transient Goal Selection (一時的目的選択)
from .transient_goal import (
    GoalReleaseReason,
    ActiveGoal,
    GoalBias,
    LightResponsibility,
    TransientGoalConfig,
    TransientGoalState,
    TransientGoalManager,
    create_manager as create_transient_goal_manager,
    create_config as create_transient_goal_config,
    apply_goal_bias_to_candidate,
    apply_goal_bias_to_candidates,
    get_transient_goal_summary,
    create_goal_context_for_trace,
    create_goal_stats_for_dynamics,
    get_responsibilities_for_dispersion,
    to_dict as transient_goal_config_to_dict,
    from_dict as transient_goal_config_from_dict,
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
    # Responsibility distribution summary (自己参照による責任要約)
    "ResponsibilityDistributionSummary", "summarize_responsibility_units",
    "generate_responsibility_distribution_tags",
    # Emotion Amplitude Expansion (感情振れ幅拡張)
    "AmplitudeState", "AmplitudeConfig",
    "apply_amplitude_to_delta", "apply_amplitude_to_emotion_deltas",
    "update_amplitude", "decay_amplitude",
    "compute_amplitude_from_dynamics", "compute_amplitude_from_residue",
    "create_amplitude_state", "get_amplitude_summary",
    # Responsibility Dispersion & Sublimation (責任の発散・昇華)
    "ResponsibilityUnit", "DispersionPlan", "SublimationPath",
    "TimeSlice", "TimeDistributionPlan", "DispersionState", "DispersionConfig",
    "AuditEntry", "AuditEventType", "DispersionStrategy", "ConservationViolationError",
    "create_responsibility_unit", "disperse_responsibility", "sublimate_responsibility",
    "distribute_over_time", "adjust_distance", "merge_responsibilities",
    "get_audit_trail", "get_unit_by_id", "get_active_units", "get_total_active_weight",
    "get_lineage", "verify_state_conservation", "create_dispersion_state",
    "get_dispersion_summary",
    # Multi-Emotion Reference (複数感情参照)
    "EmotionDecayConfig", "MultiEmotionConfig",
    "get_active_emotions", "get_all_emotions", "get_coexisting_pairs",
    "has_conflicting_emotions", "get_emotion_intensity", "get_emotion_spread",
    "apply_independent_decay", "apply_independent_update", "set_emotions_independently",
    "reference_emotions_for_judgment", "reference_emotion_by_name", "reference_multiple_emotions",
    "get_emotion_vector_summary", "create_multi_emotion_config",
    "multi_emotion_config_to_dict", "multi_emotion_config_from_dict",
    # STM-Emotion Coupling (短期記憶の感情連動)
    "STMEmotionCouplingConfig", "EmotionCouplingData", "CouplingInfluence",
    "compute_coupling_influence", "compute_decay_modifier_from_stm",
    "apply_persistence_modifier", "apply_reactivation", "apply_reactivation_to_existing",
    "apply_accumulation", "apply_stm_coupling",
    "get_coupling_summary", "get_emotion_persistence_breakdown",
    "create_coupling_config", "stm_coupling_config_to_dict", "stm_coupling_config_from_dict",
    # Silence / Hesitation as Choice (沈黙・躊躇の選択)
    "SilenceType", "SilenceConfig", "SilenceCandidate", "SilenceResult", "SilenceState",
    "generate_silence_candidate", "evaluate_silence_score",
    "create_silence_result", "create_speech_result",
    "silence_candidate_to_policy", "is_silence_policy", "is_silence_result",
    "get_silence_duration", "generate_candidates_with_silence",
    "get_silence_summary", "create_silence_config",
    "silence_config_to_dict", "silence_config_from_dict",
    # Tone / Light-Tone Mode (冗談・軽口モード)
    "Tone", "ToneConfig", "ToneModifier", "ToneState",
    "compute_tone_bias", "apply_tone_to_candidate", "get_candidate_tone",
    "select_candidate_tone", "generate_tone_variants", "add_tone_to_candidates",
    "apply_tone_to_silence", "get_tone_summary", "get_tone_from_candidate",
    "is_light_tone", "is_serious_tone", "create_tone_config",
    "tone_config_to_dict", "tone_config_from_dict",
    # Pillar managers
    "attachment_manager", "continuity_manager", "identity_manager", "projection_manager",
    # Long-Term Dynamics (長期挙動ログ)
    "EmotionStats", "DecisionStats", "ValueOrientationStats",
    "ResponsibilityStats", "StabilityValveStats", "WindowStats", "LongTermEntry",
    "DynamicsObserverConfig", "DynamicsObserver",
    "create_observer", "create_observer_config", "get_observer_summary",
    "entries_to_json", "entries_from_json",
    # Stability Valve (極端回避防止)
    "ExtremityIndicators", "StabilityBias", "StabilityValveConfig", "StabilityValve",
    "create_neutral_stability_bias", "flatten_scores",
    "apply_stability_to_candidate", "apply_stability_bias",
    "create_stability_factor", "get_stability_trace_context",
    "create_stability_valve", "create_stability_config", "observe_extremity",
    "get_stability_summary", "stability_config_to_dict", "stability_config_from_dict",
    # Introspection Trace (内省ログ生成)
    "InfluenceDirection", "FactorCategory", "OutcomeType",
    "ContributingFactor", "EmotionSnapshot", "ResponsibilitySnapshot",
    "ValueOrientationSnapshot", "DecisionSnapshot", "TraceLog",
    "IntrospectionConfig", "IntrospectionSystem",
    "generate_trace", "create_introspection_system", "create_introspection_config",
    "get_trace_summary", "traces_to_json", "traces_from_json",
    # Value Orientation (一貫した価値軸)
    "ValueOrientation", "ValueOrientationConfig", "OrientationBias",
    "compute_effective_learning_rate", "update_dimension", "update_orientation",
    "generate_decision_signal", "update_from_decision",
    "compute_orientation_bias", "apply_orientation_to_candidate", "apply_orientation_to_candidates",
    "generate_emotion_signal", "generate_responsibility_signal",
    "get_orientation_summary", "get_orientation_vector", "compute_orientation_distance",
    "is_orientation_stable", "create_orientation", "create_orientation_config",
    "orientation_config_to_dict", "orientation_config_from_dict",
    # Context Sensitivity (空気読みバイアス)
    "ExternalContext", "SensitivityBias", "ContextSensitivityConfig", "ContextState",
    "create_external_context", "create_neutral_context", "create_heavy_context", "create_light_context",
    "create_neutral_sensitivity_bias", "compute_sensitivity_bias",
    "get_policy_risk", "apply_sensitivity_to_candidate", "apply_sensitivity_to_candidates",
    "process_with_context_sensitivity", "get_sensitivity_summary", "get_context_summary",
    "is_high_caution", "is_low_caution", "create_sensitivity_config",
    "sensitivity_config_to_dict", "sensitivity_config_from_dict",
    # Proto-Goal Direction Vector (自発的方向ベクトル)
    "VectorSourceType", "VectorSource", "ProtoGoalVector",
    "VectorStateConfig", "VectorState", "VectorGenerator",
    "create_vector_generator", "create_vector_config", "get_vector_summary",
    "vectors_to_json", "vectors_from_json", "create_vector_context_for_trace",
    "vector_config_to_dict", "vector_config_from_dict",
    # Goal Candidates (自発的目的候補)
    "CandidateCategory", "CandidateSource", "GoalCandidate",
    "CandidateStateConfig", "CandidateState", "CandidateGenerator",
    "create_candidate_generator", "create_candidate_config", "get_candidate_summary",
    "candidates_to_json", "candidates_from_json",
    "create_candidate_context_for_trace", "create_candidate_stats_for_dynamics",
    "candidate_config_to_dict", "candidate_config_from_dict",
    # Transient Goal Selection (一時的目的選択)
    "GoalReleaseReason", "ActiveGoal", "GoalBias", "LightResponsibility",
    "TransientGoalConfig", "TransientGoalState", "TransientGoalManager",
    "create_transient_goal_manager", "create_transient_goal_config",
    "apply_goal_bias_to_candidate", "apply_goal_bias_to_candidates",
    "get_transient_goal_summary", "create_goal_context_for_trace",
    "create_goal_stats_for_dynamics", "get_responsibilities_for_dispersion",
    "transient_goal_config_to_dict", "transient_goal_config_from_dict",
]
