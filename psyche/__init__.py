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

# Scoped Goal Commitment (行動スコープ限定の目的コミット)
from .scoped_goal import (
    ScopeType,
    ScopeStatus,
    ScopedGoal,
    ScopedBias,
    ScopedResponsibility,
    ScopedGoalConfig,
    ScopedGoalSystem,
    create_system as create_scoped_goal_system,
    create_config as create_scoped_goal_config,
    apply_scoped_bias_to_candidate,
    apply_scoped_bias_to_candidates,
    get_scoped_goal_summary,
    create_scope_context_for_trace,
    get_responsibilities_for_integration,
    execute_scoped_decision_flow,
    to_dict as scoped_goal_config_to_dict,
    from_dict as scoped_goal_config_from_dict,
)

# Repeated Tendency (反復傾向の形成)
from .repeated_tendency import (
    TendencyPattern,
    UsageRecord,
    Tendency,
    TendencyBias,
    RepeatedTendencyConfig,
    RepeatedTendencyState,
    RepeatedTendencySystem,
    create_system as create_tendency_system,
    create_config as create_tendency_config,
    apply_tendency_bias_to_candidate,
    apply_tendency_bias_to_candidates,
    get_tendency_summary,
    create_tendency_context_for_trace,
    create_tendency_stats_for_dynamics,
    to_dict as tendency_config_to_dict,
    from_dict as tendency_config_from_dict,
)

# Tendency Awareness (反復傾向の自己認知)
from .tendency_awareness import (
    StrengthLevel,
    DurationLevel,
    ConfidenceLevel,
    AwarenessType,
    TendencyAwarenessItem,
    TendencyAwareness,
    AwarenessConfig,
    observe_tendency,
    observe_tendencies,
    generate_awareness_tags,
    get_awareness_summary,
    get_awareness_for_introspection,
    create_config as create_awareness_config,
    create_empty_awareness,
)

# Self-Model (自己状態統合モデル)
from .self_model import (
    # Abstract enums
    EmotionalSpread,
    EmotionalIntensity,
    EmotionalHarmony,
    BurdenLevel,
    BurdenDistribution,
    BurdenTrend,
    HabitPresence,
    HabitCharacter,
    DirectionClarity,
    DirectionConvergence,
    ValueStability,
    ValueClarity,
    # Component views
    EmotionalStateView,
    ResponsibilityStateView,
    TendencyStateView,
    DirectionStateView,
    ValueStateView,
    # Unified view
    SelfStateView,
    SelfModelConfig,
    SelfModelState,
    SelfModelSystem,
    # Observation functions
    observe_emotional_state,
    observe_responsibility_state,
    observe_tendency_state,
    observe_direction_state,
    observe_value_state,
    generate_integrated_description,
    # Integration functions
    generate_self_model_tags,
    get_self_model_summary,
    get_self_model_for_introspection,
    # Persistence
    save_self_model_state,
    load_self_model_state,
    # Convenience
    create_empty_view as create_empty_self_view,
    create_config as create_self_model_config,
)

# Temporal Self-Difference (自己モデル差分認知)
from .temporal_self_difference import (
    # Abstract enums
    DifferenceMagnitude,
    ChangeNature,
    ComponentChangeType,
    TemporalSpan,
    # Structures
    ComponentDifference,
    SelfDifferenceSummary,
    TemporalDifferenceConfig,
    TemporalDifferenceState,
    TemporalSelfDifferenceSystem,
    # Comparison functions
    compare_emotional_state as compare_emotional_diff,
    compare_responsibility_state as compare_responsibility_diff,
    compare_tendency_state as compare_tendency_diff,
    compare_direction_state as compare_direction_diff,
    compare_value_state as compare_value_diff,
    determine_magnitude,
    determine_nature,
    # Integration functions
    generate_difference_tags,
    get_difference_summary,
    get_difference_for_introspection,
    # Persistence
    save_difference_history,
    load_difference_history,
    # Convenience
    create_config as create_difference_config,
    create_empty_summary as create_empty_difference_summary,
)

# Continuity Strain (自己連続性負荷)
from .continuity_strain import (
    # Abstract enums
    StrainPresence,
    StrainLevel,
    StrainPersistence,
    StrainTrend,
    # Structures
    StrainState,
    DifferenceObservation,
    ContinuityStrainConfig,
    ContinuityStrainInternalState,
    ContinuityStrainSystem,
    # Determination functions
    determine_strain_level,
    determine_strain_persistence,
    determine_strain_trend,
    get_average_magnitude,
    generate_strain_description,
    # Integration functions
    generate_strain_tags,
    get_strain_summary,
    get_strain_for_introspection,
    # Persistence
    save_strain_state,
    load_strain_state,
    # Convenience
    create_config as create_strain_config,
    create_empty_strain,
    # Verification (test support)
    verify_no_decision_impact,
    verify_no_correction_mechanism,
)

# Self-Image Integration (自己像統合システム)
from .self_image_integration import (
    # Abstract enums
    EmotionalTone,
    TendencyHint,
    StabilityFeeling,
    ChangePresence,
    ContinuityFeeling,
    OverallImpression,
    # Structures
    ImageAspect,
    ProvisionalSelfImage,
    SelfImageConfig,
    SelfImageIntegrationSystem,
    # Integration functions
    generate_self_image_tags,
    get_self_image_summary,
    get_self_image_for_introspection,
    # Convenience
    create_config as create_self_image_config,
    create_empty_image,
    # Verification (test support)
    verify_no_decision_impact as verify_image_no_decision_impact,
    verify_provisional_nature,
)

# Identity Coherence Awareness (自己同一性の揺らぎ認知)
from .identity_coherence import (
    # Abstract enums
    CoherenceLevel,
    ShiftSource,
    OverlapIntensity,
    CoherenceTrend,
    # Structures
    DetectedShift,
    ShiftOverlap,
    IdentityCoherenceState,
    IdentityCoherenceConfig,
    IdentityCoherenceSystem,
    # Detection functions
    detect_temporal_difference_shift,
    detect_tendency_change_shift,
    detect_continuity_strain_shift,
    detect_value_instability_shift,
    detect_self_image_flux_shift,
    detect_emotional_turbulence_shift,
    # Determination functions
    determine_overlap_intensity,
    determine_coherence_level,
    determine_coherence_trend,
    # Integration functions
    generate_coherence_tags,
    get_coherence_summary,
    get_coherence_for_introspection,
    # Convenience
    create_config as create_coherence_config,
    create_empty_state as create_empty_coherence_state,
    # Verification (test support)
    verify_no_decision_impact as verify_coherence_no_decision_impact,
    verify_no_self_preservation,
    verify_no_identity_definition,
)

# Self-Narrative Formation (自己物語形成)
from .self_narrative import (
    # Abstract enums
    FragmentType,
    LinkType,
    VividnessLevel,
    NarrativeCoherence,
    NarrativeTrend,
    # Structures
    NarrativeFragment,
    FragmentLink,
    CoherenceInfo,
    NarrativeState,
    SelfNarrativeConfig,
    SelfNarrativeSystem,
    # Classification functions
    classify_emotion_observation,
    classify_memory_observation,
    classify_tendency_observation,
    classify_difference_observation,
    classify_context_observation,
    # Processing functions
    check_for_rewrites,
    summarize_fragments,
    compute_coherence,
    determine_narrative_trend,
    generate_narrative_description,
    # Integration functions
    observe_from_chain,
    generate_narrative_tags,
    get_narrative_summary,
    get_narrative_for_introspection,
    # Convenience
    create_config as create_narrative_config,
    create_empty_state as create_empty_narrative_state,
    # Verification (test support)
    verify_no_decision_impact as verify_narrative_no_decision_impact,
    verify_no_identity_definition as verify_narrative_no_identity_definition,
    verify_no_goal_generation,
    verify_read_only_principle,
)

# Episodic Memory (エピソード記憶 - 自伝的記憶)
from .episodic_memory import (
    # Abstract enums
    EpisodeType,
    ImportanceLevel,
    DecayState,
    EpisodeLinkType,
    SearchMode,
    # Structures
    EmotionalCompanion,
    SelfObservationCompanion,
    EpisodeLink,
    EpisodeEntry,
    EpisodeStore,
    EpisodicMemoryConfig,
    EpisodicMemorySystem,
    # Helper functions
    determine_decay_state,
    classify_episode_type,
    compute_importance,
    compute_topic_overlap,
    compute_emotional_similarity,
    compute_temporal_proximity,
    generate_episode_summary,
    # Integration functions
    record_from_chain as record_episode_from_chain,
    generate_episodic_memory_tags,
    get_episodic_memory_summary,
    get_episodic_memory_for_introspection,
    # Persistence
    save_episodic_memory,
    load_episodic_memory,
    # Convenience
    create_config as create_episodic_memory_config,
    create_empty_store as create_empty_episode_store,
    create_system as create_episodic_memory_system,
    # Verification (test support)
    verify_no_decision_impact as verify_episodic_no_decision_impact,
    verify_no_goal_generation as verify_episodic_no_goal_generation,
    verify_read_only_principle as verify_episodic_read_only_principle,
    verify_no_value_modification,
)

# Introspection Consumption Layer (内省の消費層)
from .introspection_consumption import (
    # Abstract enums
    FragmentSourceType,
    BundleCoherence,
    FragmentFreshness,
    # Structures
    IntrospectionFragment,
    FragmentBundle,
    ConsumptionRecord,
    ConsumptionStore,
    ConsumptionLayerConfig,
    IntrospectionConsumptionSystem,
    # Helper functions
    determine_freshness,
    extract_from_introspection,
    extract_from_narrative,
    extract_from_coherence,
    extract_from_tendency,
    extract_from_episodic,
    compute_fragment_relevance,
    determine_bundle_coherence,
    generate_bundles,
    # Integration functions
    consume_from_chain as consume_introspection_from_chain,
    generate_consumption_tags,
    get_consumption_summary,
    get_consumption_for_introspection,
    # Persistence
    save_consumption_state,
    load_consumption_state,
    # Convenience
    create_config as create_consumption_config,
    create_empty_store as create_empty_consumption_store,
    create_system as create_consumption_system,
    # Verification (test support)
    verify_no_decision_impact as verify_consumption_no_decision_impact,
    verify_no_goal_generation as verify_consumption_no_goal_generation,
    verify_read_only_principle as verify_consumption_read_only_principle,
    verify_no_value_modification as verify_consumption_no_value_modification,
)

# Expectation Formation (予期・期待の形成)
from .expectation_formation import (
    # Abstract enums
    ExpectationSourceType,
    ExpectationBasis,
    ExpectationStrength,
    ExpectationFreshness,
    # Structures
    EvidenceLink,
    ExpectationCandidate,
    ExpectationStore,
    ExpectationFormationConfig,
    ExpectationFormationSystem,
    # Helper functions
    determine_freshness_level as determine_expectation_freshness_level,
    determine_strength_level,
    extract_from_tendency as extract_expectation_from_tendency,
    extract_from_difference as extract_expectation_from_difference,
    extract_from_narrative as extract_expectation_from_narrative,
    compute_evidence_strength,
    detect_competitions,
    determine_expectation_basis,
    generate_expectation_description,
    # Integration functions
    form_from_chain as form_expectations_from_chain,
    generate_expectation_tags,
    get_expectation_summary,
    get_expectation_for_introspection,
    # Persistence
    save_expectation_state,
    load_expectation_state,
    # Convenience
    create_config as create_expectation_config,
    create_empty_store as create_empty_expectation_store,
    create_system as create_expectation_system,
    # Verification (test support)
    verify_no_decision_impact as verify_expectation_no_decision_impact,
    verify_no_goal_generation as verify_expectation_no_goal_generation,
    verify_read_only_principle as verify_expectation_read_only_principle,
    verify_no_value_modification as verify_expectation_no_value_modification,
)

# Other Agent Model (他者モデル)
from .other_agent_model import (
    # Abstract enums
    ObservationSourceType,
    InferenceBasis,
    HypothesisStrength,
    HypothesisFreshness,
    # Structures
    ObservationLink,
    OtherStateHypothesis,
    SelfOtherBoundary,
    OtherModelStore,
    OtherAgentModelConfig,
    OtherAgentModelSystem,
    # Helper functions
    determine_freshness_level as determine_hypothesis_freshness_level,
    determine_strength_level as determine_hypothesis_strength_level,
    extract_from_external_context,
    extract_from_reaction_log,
    extract_from_self_contrast,
    compute_observation_strength,
    detect_hypothesis_competitions,
    determine_inference_basis,
    generate_hypothesis_description,
    compute_self_other_boundary,
    # Integration functions
    observe_from_chain as observe_other_from_chain,
    generate_other_model_tags,
    get_other_model_summary,
    get_other_model_for_introspection,
    # Persistence
    save_other_model_state,
    load_other_model_state,
    # Convenience
    create_config as create_other_model_config,
    create_empty_store as create_empty_other_model_store,
    create_system as create_other_model_system,
    # Verification (test support)
    verify_no_decision_impact as verify_other_model_no_decision_impact,
    verify_no_goal_generation as verify_other_model_no_goal_generation,
    verify_read_only_principle as verify_other_model_read_only_principle,
    verify_no_value_modification as verify_other_model_no_value_modification,
    verify_no_intent_assertion,
)

# Emotional Memory Binding (感情記憶の紐づけ)
from .emotional_memory_binding import (
    # Abstract enums
    BindingSourceType,
    TraceAffinity,
    TraceStrength,
    TraceFreshness,
    # Structures
    EmotionalTrace,
    BindingLink,
    MemoryBinding,
    BindingStore,
    EmotionalMemoryBindingConfig,
    EmotionalMemoryBindingSystem,
    # Helper functions
    determine_freshness_level as determine_binding_freshness_level,
    determine_strength_level as determine_binding_strength_level,
    generate_memory_key,
    extract_from_stm as extract_binding_from_stm,
    extract_from_emotion_state as extract_binding_from_emotion_state,
    extract_from_recalled_memories as extract_binding_from_recalled_memories,
    extract_from_episodes as extract_binding_from_episodes,
    compute_binding_strength,
    detect_trace_coexistence,
    compute_emotional_accompaniment,
    # Integration functions
    bind_from_chain,
    generate_binding_tags,
    get_binding_summary,
    get_binding_for_introspection,
    # Persistence
    save_binding_state,
    load_binding_state,
    # Convenience
    create_config as create_binding_config,
    create_empty_store as create_empty_binding_store,
    create_system as create_binding_system,
    # Verification (test support)
    verify_no_decision_impact as verify_binding_no_decision_impact,
    verify_no_goal_generation as verify_binding_no_goal_generation,
    verify_read_only_principle as verify_binding_read_only_principle,
    verify_no_value_modification as verify_binding_no_value_modification,
    verify_no_emotion_evaluation,
)

# Intrinsic Motivation (自発的内的動機)
from .intrinsic_motivation import (
    # Abstract enums
    MotiveSourceType,
    MotiveAffinity,
    MotiveStrength,
    MotiveFreshness,
    # Structures
    MotiveImpulse,
    MotiveLink,
    MotiveEntry,
    MotiveStore,
    IntrinsicMotivationConfig,
    IntrinsicMotivationSystem,
    # Helper functions
    determine_freshness_level as determine_motive_freshness_level,
    determine_strength_level as determine_motive_strength_level,
    generate_motive_key,
    extract_from_emotion_state as extract_motive_from_emotion_state,
    extract_from_tendencies as extract_motive_from_tendencies,
    extract_from_goal_vectors as extract_motive_from_goal_vectors,
    extract_from_goal_candidates as extract_motive_from_goal_candidates,
    compute_motive_strength,
    detect_motive_coexistence,
    compute_motive_overlay,
    # Integration functions
    sense_from_chain as sense_motives_from_chain,
    generate_motive_tags,
    get_motive_summary,
    get_motive_for_introspection,
    # Persistence
    save_motive_state,
    load_motive_state,
    # Convenience
    create_config as create_motive_config,
    create_empty_store as create_empty_motive_store,
    create_system as create_motive_system,
    # Verification (test support)
    verify_no_decision_impact as verify_motive_no_decision_impact,
    verify_no_goal_generation as verify_motive_no_goal_generation,
    verify_read_only_principle as verify_motive_read_only_principle,
    verify_no_value_modification as verify_motive_no_value_modification,
    verify_no_motivation_prescription,
)

# Policy Candidate Expansion (ポリシー候補拡張)
from .policy_candidate_expansion import (
    CrossSection,
    PolicyAxis,
    InputFragment,
    ExpandedCandidate,
    HistoryEntry,
    SuppressionEntry,
    CompetitionEntry,
    ExpansionConfig,
    ExpansionState,
    CrossSectionInputs,
    PolicyCandidateExpander,
    create_expander as create_policy_expander,
    create_config as create_expansion_config,
    extract_all_fragments,
    get_expansion_summary,
    get_expansion_summary_text,
)

# Memory System Integration (記憶系統統合)
from .memory_system_integration import (
    MemorySource,
    TemporalPhase,
    UnifiedMemoryUnit,
    DuplicateEntry,
    ConflictEntry,
    ReferenceHistoryEntry,
    IntegrationContext,
    IntegrationConfig,
    IntegrationState,
    IntegrationResult,
    MemorySystemIntegrator,
    create_integrator as create_memory_integrator,
    create_config as create_integration_config,
    get_integration_summary,
    get_integration_summary_text,
    normalize_episodic,
    normalize_long_term,
    normalize_bindings,
    detect_duplicates as detect_memory_duplicates,
    detect_conflicts,
    check_conflict_health,
)

# Other Model Real Feed (他者モデルリアルフィード統合)
from .other_model_real_feed import (
    ObservationFragmentType,
    FragmentFreshness as RealFeedFragmentFreshness,
    AlignmentStatus,
    ConflictStatus,
    ObservationFragment,
    ObservationUnit,
    ConflictRecord,
    FeedHistoryEntry,
    HoldbackEntry,
    RealFeedConfig,
    RealFeedState,
    FeedResult,
    extract_speech_reaction,
    extract_response_interval,
    extract_topic_transition,
    extract_emotional_tone,
    extract_continued_engagement,
    extract_rejection_acceptance,
    extract_context_alignment,
    extract_recent_history,
    normalize_fragments,
    align_units,
    detect_feed_duplicates,
    detect_feed_conflicts,
    apply_freshness,
    suppress_recent_series,
    ensure_type_diversity,
    check_convergence,
    check_stagnation,
    RealFeedProcessor,
    create_real_feed_processor,
    enhance_context_with_feed,
    get_real_feed_summary,
)

# Other Model Input Supply (他者モデル入力供給)
from .other_model_input_supply import (
    SupplyEntry,
    ContextSnapshot,
    ReactionBufferEntry,
    InputSupplyState,
    create_input_supply,
    update_from_percept as update_input_supply,
    decay_buffer,
    supply_context,
    supply_reaction_log,
    get_input_supply_summary,
)

# Text Dialogue Input (テキスト対話入力経路)
from .text_dialogue_input import (
    InputRouteType,
    InputFreshness,
    NormalizationStatus,
    ContextLinkStatus,
    DuplicateStatus,
    RouteConflictStatus,
    InputUnit,
    ContextLink,
    DuplicateRecord,
    RouteConflict,
    ReceiveHistoryEntry,
    SuppressionHistoryEntry,
    DecayHistoryEntry,
    TextDialogueConfig,
    TextDialogueState,
    HandoffResult,
    receive_input,
    normalize_unit,
    attach_context,
    align_to_percept_format,
    detect_duplicates as detect_text_duplicates,
    prepare_handoff,
    apply_freshness_decay,
    decay_receive_history,
    decay_suppression_history,
    suppress_recent_adoption,
    check_empty_streak,
    detect_single_route_dominance,
    ensure_format_diversity,
    restore_multi_route,
    filter_circular_reference,
    TextDialogueProcessor,
    merge_with_percept,
    get_text_dialogue_summary,
    create_text_dialogue_processor,
)

# Spontaneous Activation (自発性の追加)
from .spontaneous_activation import (
    ActivationSourceType,
    ActivationFreshness,
    CandidateStatus,
    SuppressionMode,
    ConflictResolution,
    ActivationFragment,
    ActivationCandidate,
    ActivationRationale,
    SuppressionEntry as SpontaneousSuppressionEntry,
    StandbyEntry,
    ConflictHistoryEntry as SpontaneousConflictHistoryEntry,
    UnadoptedHistoryEntry,
    ContinuousActivationEntry,
    SpontaneousDecayEntry,
    SpontaneousConfig,
    SpontaneousState,
    ActivationResult,
    extract_intrinsic_motivation,
    extract_direction_vector,
    extract_unfinished_intent,
    extract_memory_echo,
    extract_emotional_transition,
    extract_responsibility as extract_activation_responsibility,
    extract_recent_action,
    extract_external_input_absence,
    extract_all_fragments as extract_all_activation_fragments,
    form_candidates,
    align_conditions,
    resolve_conflicts,
    check_activation_feasibility,
    suppress_consecutive_series,
    apply_overdense_cooldown,
    restore_candidate_diversity,
    apply_freshness_decay as apply_spontaneous_freshness_decay,
    decay_unadopted_history,
    SpontaneousActivationProcessor,
    get_spontaneous_summary,
    create_spontaneous_processor,
)

# Self-Action Perception (自己行動知覚)
from .self_action_perception import (
    RecordStatus,
    SelfActionRecord,
    SelfActionPerceptionState,
    SelfActionPerceptionConfig,
    SelfActionPerceptionRecorder,
    get_self_action_summary,
    create_self_action_perception_recorder,
)

# Intent-Action Gap (意図-行動間の乖離認知)
from .intent_action_gap import (
    GapRecord,
    IntentActionGapState,
    IntentActionGapConfig,
    IntentActionGapRecorder,
    get_gap_summary,
    create_intent_action_gap_recorder,
)

# Temporal Cognition (時間認知構造)
from .temporal_cognition import (
    DensityLevel,
    ElapsedRecord,
    TemporalCognitionState,
    TemporalCognitionConfig,
    TemporalCognitionProcessor,
    get_temporal_summary,
    create_temporal_cognition,
    SECTION_ORDER,
    SECTION_ACTIVITY_DENSITY,
    SECTION_MEMORY_INTERVAL,
    SECTION_EMOTION_FREQUENCY,
    SECTION_NARRATIVE_INTERVAL,
    SECTION_EXTERNAL_INPUT_INTERVAL,
    SECTION_OVERALL_ELAPSED,
    SECTION_LABELS,
    DENSITY_LABELS,
)

# Multi-Path Recall (記憶の多経路想起)
from .multi_path_recall import (
    RecallPathLabel,
    RecallCandidate,
    PathStatistics,
    EmotionSnapshot as RecallEmotionSnapshot,
    ContextSnapshot as RecallContextSnapshot,
    TemporalSnapshot as RecallTemporalSnapshot,
    MultiPathRecallState,
    MultiPathRecallConfig,
    MultiPathRecallProcessor,
    get_recall_summary,
    create_multi_path_recall,
)

# Introspection Cross-Section (内省断面間の横断的記述)
from .introspection_cross_section import (
    CrossSectionSnapshot,
    IntrospectionCrossSectionState,
    IntrospectionCrossSectionConfig,
    IntrospectionCrossSectionProcessor,
    create_introspection_cross_section,
    SECTION_ORDER as INTROSPECTION_CS_SECTION_ORDER,
    SECTION_LABELS as INTROSPECTION_CS_SECTION_LABELS,
)

# Introspection Longitudinal View (内省の時間的縦断参照)
from .introspection_longitudinal_view import (
    LongitudinalViewConfig,
    TimePointEntry,
    SectionTimeline,
    LongitudinalView,
    IntrospectionLongitudinalViewProcessor,
    create_introspection_longitudinal_view,
    get_enrichment_data as get_longitudinal_enrichment_data,
    get_longitudinal_view,
    get_section_timeline,
)

# Perceptual Context (知覚入力の内部文脈化)
from .perceptual_context import (
    PerceptualSummary,
    PerceptualContextState,
    PerceptualContextConfig,
    PerceptualContextProcessor,
    create_perceptual_context,
    get_perceptual_context_summary,
    SECTION_ORDER as PERCEPTUAL_CTX_SECTION_ORDER,
    SECTION_LABELS as PERCEPTUAL_CTX_SECTION_LABELS,
)

# Value Orientation Validation (価値方向性実運用検証)
from .value_orientation_validation import (
    ObservationSourceType as ValidationObservationSourceType,
    ObservationFreshness as ValidationObservationFreshness,
    DifferentialType,
    ValidationStatus,
    ObservationRecord as ValidationObservationRecord,
    ValidationDescriptionUnit,
    DifferentialEntry,
    TimeSeriesEntry as ValidationTimeSeriesEntry,
    ValidationInputs,
    ValidationState,
    ValidationResult,
    ValidationConfig,
    ValueOrientationValidator,
    get_validation_summary,
    create_validation_processor,
)

# Memory Forgetting and Fixation (記憶の忘却と固定化)
from .memory_forgetting_fixation import (
    ObservationSourceType as ForgettingObservationSourceType,
    ForgettingStage,
    FixationLevel,
    SeriesStatus,
    MemorySeriesRecord,
    ForgettingCandidate,
    FixationSign,
    ForgettingFixationInputs,
    ForgettingFixationState,
    ForgettingFixationResult,
    ForgettingFixationConfig,
    MemoryForgettingFixationProcessor,
    get_forgetting_fixation_summary,
    create_forgetting_fixation_processor,
)

# Spontaneous Recall (記憶の自発的想起 - 非参照型想起)
from .spontaneous_recall import (
    SpontaneousRecallPathLabel,
    SpontaneousRecallCandidate,
    SpontaneousRecallPathStatistics,
    InternalEmotionSnapshot as SpontaneousRecallEmotionSnapshot,
    InternalStateCrossSections,
    SpontaneousRecallState,
    SpontaneousRecallConfig,
    SpontaneousRecallProcessor,
    extract_cross_sections as extract_recall_cross_sections,
    get_spontaneous_recall_summary,
    create_spontaneous_recall,
)

# Action Result Observation (行動-結果の観測と蓄積)
from .action_result_observation import (
    ObservationSection,
    FreshnessStage as ActionResultFreshnessStage,
    PairStatus,
    ConvergenceLevel as ActionResultConvergenceLevel,
    SectionDescription,
    ActionDescription,
    ResultDescription,
    ContextAttribution,
    ActionResultPair,
    SectionWeightRecord,
    ConvergenceRecord as ActionResultConvergenceRecord,
    ActionResultInputs,
    ActionResultObservationState,
    ActionResultObservationResult,
    ActionResultConfig,
    ActionResultObservationProcessor,
    get_action_result_summary,
    create_action_result_processor,
)

# Other Model Dialogue Learning (他者観測の長期蓄積と仮説補助)
from .other_model_dialogue_learning import (
    InputSection as DialogueLearningInputSection,
    FreshnessStage as DialogueLearningFreshnessStage,
    EntryStatus as DialogueLearningEntryStatus,
    PatternType,
    ConvergenceLevel as DialogueLearningConvergenceLevel,
    AccumulationEntry,
    PatternRecord,
    HypothesisMaterial,
    ConvergenceRecord as DialogueLearningConvergenceRecord,
    DialogueLearningInputs,
    DialogueLearningState,
    DialogueLearningResult,
    DialogueLearningConfig,
    DialogueLearningProcessor,
    get_dialogue_learning_summary,
    create_dialogue_learning_processor,
)

# Meta-Emotion Cognition (メタ感情認知と変動候補生成)
from .meta_emotion_cognition import (
    InputSection as MetaEmotionInputSection,
    FreshnessStage as MetaEmotionFreshnessStage,
    RecordStatus as MetaEmotionRecordStatus,
    ConvergenceLevel as MetaEmotionConvergenceLevel,
    TransitionFeature,
    SustainedPattern,
    VariationCandidate,
    CognitionRecord,
    ConvergenceRecord as MetaEmotionConvergenceRecord,
    MetaEmotionInputs,
    MetaEmotionState,
    MetaEmotionResult,
    MetaEmotionConfig,
    MetaEmotionProcessor,
    get_meta_emotion_summary,
    create_meta_emotion_processor,
)

from . import attachment_manager
from . import continuity_manager
from . import identity_manager
from . import projection_manager

# Internal Contradiction Description (内部状態の矛盾並置の構造的記述)
from .internal_contradiction_description import (
    ContradictionRecord,
    ContradictionInputs,
    ContradictionState,
    ContradictionResult,
    ContradictionConfig,
    InternalContradictionProcessor,
    get_contradiction_summary,
    create_contradiction_processor,
    PAIR_DEFINITIONS as CONTRADICTION_PAIR_DEFINITIONS,
    PAIR_LABELS as CONTRADICTION_PAIR_LABELS,
)

# Selection Attribution (選択帰属)
from .selection_attribution import (
    SelectionRecord,
    SelectionAttributionState,
    SelectionAttributionConfig,
    SelectionAttributionRecorder,
    create_selection_attribution_recorder,
    get_selection_attribution_summary,
)

# Reference Frequency Description (参照頻度の構造的記述)
from .reference_frequency_description import (
    ReferenceFrequencyConfig,
    ReferenceFrequencyState,
    ReferenceSnapshot,
    VariationDescription,
    process_reference_frequency,
    collect_reference_counts,
    create_reference_frequency_state,
    create_reference_frequency_config,
    get_latest_snapshot,
    get_snapshot_history,
    get_latest_variation,
    get_reference_summary,
)

# Persistent Commitment (持続的取り組み保持構造)
from .persistent_commitment import (
    PersistentCommitmentConfig,
    PersistentCommitmentState,
    PersistentCommitmentProcessor,
    CommitmentItem,
    CognitionRecord,
    CommitmentCrossSectionInputs,
    create_persistent_commitment_processor,
    get_commitment_summary,
)

# Stabilization Description (安定化の構造的記述)
from .stabilization_description import (
    StabilizationDescriptionConfig,
    StabilizationDescriptionState,
    StabilizationRecord,
    process_stabilization_description,
    read_signal_sources,
    read_diff_reference,
    compose_record as compose_stabilization_record,
    accumulate_record as accumulate_stabilization_record,
    get_latest_record as get_latest_stabilization_record,
    get_record_history as get_stabilization_record_history,
    get_stabilization_summary,
    create_stabilization_description_state,
    create_stabilization_description_config,
    save_state as save_stabilization_state,
    load_state as load_stabilization_state,
    SIGNAL_EMOTION,
    SIGNAL_STM_ENTRIES,
    SIGNAL_TRANSIENT_GOAL,
    SIGNAL_PERSISTENT_COMMITMENT,
    SIGNAL_SPONTANEOUS_CANDIDATE,
    SIGNAL_EXTERNAL_INPUT,
    ALL_SIGNAL_KEYS,
)

# Behavioral Diversity Description (行動多様性の構造的記述)
from .behavioral_diversity_description import (
    BehavioralDiversityConfig,
    BehavioralDiversityState,
    DiversityRecord,
    TypeCountLevel,
    DispersionLevel,
    determine_type_count_level,
    determine_dispersion_level,
    process_behavioral_diversity,
    read_section_key_types,
    read_policy_label_types,
    read_candidate_size_types,
    compose_record as compose_diversity_record,
    accumulate_record as accumulate_diversity_record,
    get_latest_record as get_latest_diversity_record,
    get_record_history as get_diversity_record_history,
    get_diversity_summary,
    create_behavioral_diversity_state,
    create_behavioral_diversity_config,
    save_state as save_behavioral_diversity_state,
    load_state as load_behavioral_diversity_state,
)

# Interaction Accumulation (相互作用の蓄積記述)
from .interaction_accumulation import (
    AdjacentPair,
    BufferEntry,
    InteractionAccumulationState,
    InteractionAccumulationConfig,
    InteractionAccumulationProcessor,
    get_interaction_summary,
    create_interaction_accumulation_processor,
)

# Emotional Backdrop Cognition (感情基調の持続認知)
from .emotional_backdrop_cognition import (
    InputSection as BackdropInputSection,
    FreshnessStage as BackdropFreshnessStage,
    ConvergenceLevel as BackdropConvergenceLevel,
    WindowEntry as BackdropWindowEntry,
    CompositionRecord as BackdropCompositionRecord,
    ConvergenceRecord as BackdropConvergenceRecord,
    BackdropInputs,
    BackdropState,
    BackdropResult,
    BackdropConfig,
    EmotionalBackdropProcessor,
    get_backdrop_summary,
    create_emotional_backdrop_processor,
)

# Situational Self-Presentation (状況依存的自己呈示の認知)
from .situational_self_presentation import (
    RecordFreshness as PresentationRecordFreshness,
    TypeCountLevel as PresentationTypeCountLevel,
    ConvergenceLevel as PresentationConvergenceLevel,
    PresentationRecord,
    CompositionDescription,
    ConvergenceRecord as PresentationConvergenceRecord,
    SituationalSelfPresentationState,
    SituationalSelfPresentationConfig,
    SituationalSelfPresentationProcessor,
    determine_type_count_level as determine_presentation_type_count_level,
    get_presentation_summary,
    create_situational_self_presentation_processor,
)

# Drive Variation Description (駆動の変動記述)
from .drive_variation_description import (
    InputSection as DriveVariationInputSection,
    FreshnessStage as DriveVariationFreshnessStage,
    ConvergenceLevel as DriveVariationConvergenceLevel,
    WindowEntry as DriveVariationWindowEntry,
    CompositionRecord as DriveVariationCompositionRecord,
    DecayRecord as DriveVariationDecayRecord,
    ConvergenceRecord as DriveVariationConvergenceRecord,
    DriveVariationInputs,
    DriveVariationState,
    DriveVariationResult,
    DriveVariationConfig,
    DriveVariationProcessor,
    get_drive_variation_summary,
    create_drive_variation_processor,
)

# Scoring Fluctuation (スコアリングの構造的揺らぎ)
from .scoring_fluctuation import (
    ScoringFluctuationConfig,
    apply_scoring_fluctuation,
    extract_emotion_variability,
    extract_stm_variability,
    extract_drive_variability,
    extract_elapsed_variability,
    compose_variability,
    limit_amplitude,
    generate_per_policy_fluctuations,
    apply_fluctuations_to_candidates,
    extract_stm_info,
    get_fluctuation_summary,
    create_fluctuation_config,
)

# Expectation Lifecycle Description (予期の成立・消失の事後記述)
from .expectation_lifecycle_description import (
    TransitionType as LifecycleTransitionType,
    RecordFreshness as LifecycleRecordFreshness,
    ConvergenceLevel as LifecycleConvergenceLevel,
    TransitionRecord as LifecycleTransitionRecord,
    LifecycleView,
    ConvergenceRecord as LifecycleConvergenceRecord,
    SnapshotEntry as LifecycleSnapshotEntry,
    ExpectationLifecycleState,
    ExpectationLifecycleConfig,
    ExpectationLifecycleDescriptionProcessor,
    get_lifecycle_summary,
    create_expectation_lifecycle_processor,
)

# Input Pathway Balance (入力経路間の均衡記述)
from .input_pathway_balance import (
    InputPathwayBalanceConfig,
    InputPathwayBalanceState,
    UsageFact,
    PathwaySnapshot,
    PathwayVariation,
    UsageLevel,
    BiasLevel,
    PATHWAY_TEXT,
    PATHWAY_SCREEN,
    PATHWAY_SPONTANEOUS,
    ALL_PATHWAYS,
    determine_usage_level,
    determine_bias_level,
    compute_bias_value,
    collect_usage_fact,
    read_text_dialogue_usage,
    read_spontaneous_usage,
    compose_snapshot,
    derive_variation,
    process_input_pathway_balance,
    get_latest_snapshot as get_latest_pathway_snapshot,
    get_snapshot_history as get_pathway_snapshot_history,
    get_latest_variation as get_latest_pathway_variation,
    get_pathway_balance_summary,
    get_enrichment_text as get_pathway_balance_enrichment_text,
    save_state as save_input_pathway_balance_state,
    load_state as load_input_pathway_balance_state,
    create_input_pathway_balance_state,
    create_input_pathway_balance_config,
)

# Responsibility Temporal Trace (責任の時間的推移記述)
from .responsibility_temporal_trace import (
    VariationLevel,
    ResponsibilitySnapshot,
    ResponsibilityTemporalTraceState,
    ResponsibilityTemporalTraceConfig,
    ResponsibilityTemporalTraceProcessor,
    get_trace_summary,
    create_responsibility_temporal_trace,
    SECTION_ORDER as RESPONSIBILITY_TRACE_SECTION_ORDER,
    SECTION_TOTAL_WEIGHT_VARIATION,
    SECTION_PENDING_DECISIONS_RETENTION,
    SECTION_HARM_VARIATION,
    SECTION_CONFIDENCE_VARIATION,
    SECTION_DISPERSION_ACTIVITY_DENSITY,
    SECTION_LABELS as RESPONSIBILITY_TRACE_SECTION_LABELS,
    VARIATION_LABELS,
)

# Other Boundary Accumulation (他者境界の多相蓄積)
from .other_boundary_accumulation import (
    FreshnessStage as BoundaryFreshnessStage,
    DivergenceLevel,
    ConvergenceLevel as BoundaryConvergenceLevel,
    BoundaryRecord,
    ConvergenceRecord as BoundaryConvergenceRecord,
    OtherBoundaryAccumulationState,
    BoundaryAccumulationResult,
    OtherBoundaryAccumulationConfig,
    OtherBoundaryAccumulationProcessor,
    determine_divergence_level,
    get_boundary_accumulation_summary,
    create_other_boundary_accumulation_processor,
)

# Emotion cooccurrence description (感情間の共起記述)
from .emotion_cooccurrence_description import (
    FreshnessStage as CooccurrenceFreshnessStage,
    DiversityLevel as CooccurrenceDiversityLevel,
    CooccurrencePair,
    CooccurrenceRecord,
    CooccurrenceState,
    CooccurrenceResult,
    CooccurrenceConfig,
    EmotionCooccurrenceDescriptionProcessor,
    get_cooccurrence_summary,
    create_cooccurrence_processor,
)

# Attention Distribution Description (注意配分の構造的記述)
from .attention_distribution_description import (
    AttentionDistributionConfig,
    AttentionDistributionState,
    AttentionSnapshot,
    AttentionVariation,
    QuantityLevel,
    ConcentrationLevel,
    SOURCE_PERCEPTION,
    SOURCE_TEXT_INPUT,
    SOURCE_SPONTANEOUS,
    SIGNAL_EMOTION as ATTENTION_SIGNAL_EMOTION,
    SIGNAL_MEMORY as ATTENTION_SIGNAL_MEMORY,
    SIGNAL_MOTIVATION as ATTENTION_SIGNAL_MOTIVATION,
    SIGNAL_GOAL as ATTENTION_SIGNAL_GOAL,
    SIGNAL_RESPONSIBILITY as ATTENTION_SIGNAL_RESPONSIBILITY,
    ALL_SOURCE_KEYS,
    determine_quantity_level,
    determine_concentration_level,
    compute_concentration as compute_attention_concentration,
    collect_source_quantities,
    compose_snapshot as compose_attention_snapshot,
    derive_variation as derive_attention_variation,
    process_attention_distribution,
    get_latest_snapshot as get_latest_attention_snapshot,
    get_snapshot_history as get_attention_snapshot_history,
    get_latest_variation as get_latest_attention_variation,
    get_attention_distribution_summary,
    get_enrichment_text as get_attention_distribution_enrichment_text,
    save_state as save_attention_distribution_state,
    load_state as load_attention_distribution_state,
    create_attention_distribution_state,
    create_attention_distribution_config,
)

# Forgetting-Recall Balance (忘却と想起の均衡記述)
from .forgetting_recall_balance import (
    ForgettingRecallBalanceConfig,
    ForgettingRecallBalanceState,
    ForgettingSectionSnapshot,
    ExternalRecallSectionSnapshot,
    SpontaneousRecallSectionSnapshot,
    JuxtapositionEntry,
    extract_forgetting_section,
    extract_external_recall_section,
    extract_spontaneous_recall_section,
    compose_juxtaposition,
    accumulate_entry,
    process_forgetting_recall_balance,
    get_recent_entries as get_frb_recent_entries,
    get_history as get_frb_history,
    get_balance_summary as get_frb_balance_summary,
    get_enrichment_text as get_frb_enrichment_text,
    save_state as save_frb_state,
    load_state as load_frb_state,
    create_forgetting_recall_balance_state,
    create_forgetting_recall_balance_config,
)

# Orchestrator (全モジュール統合管理)
from .orchestrator import PsycheOrchestrator

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
    # Scoped Goal Commitment (行動スコープ限定の目的コミット)
    "ScopeType", "ScopeStatus", "ScopedGoal", "ScopedBias", "ScopedResponsibility",
    "ScopedGoalConfig", "ScopedGoalSystem",
    "create_scoped_goal_system", "create_scoped_goal_config",
    "apply_scoped_bias_to_candidate", "apply_scoped_bias_to_candidates",
    "get_scoped_goal_summary", "create_scope_context_for_trace",
    "get_responsibilities_for_integration", "execute_scoped_decision_flow",
    "scoped_goal_config_to_dict", "scoped_goal_config_from_dict",
    # Repeated Tendency (反復傾向の形成)
    "TendencyPattern", "UsageRecord", "Tendency", "TendencyBias",
    "RepeatedTendencyConfig", "RepeatedTendencyState", "RepeatedTendencySystem",
    "create_tendency_system", "create_tendency_config",
    "apply_tendency_bias_to_candidate", "apply_tendency_bias_to_candidates",
    "get_tendency_summary", "create_tendency_context_for_trace",
    "create_tendency_stats_for_dynamics",
    "tendency_config_to_dict", "tendency_config_from_dict",
    # Tendency Awareness (反復傾向の自己認知)
    "StrengthLevel", "DurationLevel", "ConfidenceLevel", "AwarenessType",
    "TendencyAwarenessItem", "TendencyAwareness", "AwarenessConfig",
    "observe_tendency", "observe_tendencies",
    "generate_awareness_tags", "get_awareness_summary", "get_awareness_for_introspection",
    "create_awareness_config", "create_empty_awareness",
    # Self-Model (自己状態統合モデル)
    "EmotionalSpread", "EmotionalIntensity", "EmotionalHarmony",
    "BurdenLevel", "BurdenDistribution", "BurdenTrend",
    "HabitPresence", "HabitCharacter",
    "DirectionClarity", "DirectionConvergence",
    "ValueStability", "ValueClarity",
    "EmotionalStateView", "ResponsibilityStateView", "TendencyStateView",
    "DirectionStateView", "ValueStateView",
    "SelfStateView", "SelfModelConfig", "SelfModelState", "SelfModelSystem",
    "observe_emotional_state", "observe_responsibility_state",
    "observe_tendency_state", "observe_direction_state", "observe_value_state",
    "generate_integrated_description",
    "generate_self_model_tags", "get_self_model_summary", "get_self_model_for_introspection",
    "save_self_model_state", "load_self_model_state",
    "create_empty_self_view", "create_self_model_config",
    # Temporal Self-Difference (自己モデル差分認知)
    "DifferenceMagnitude", "ChangeNature", "ComponentChangeType", "TemporalSpan",
    "ComponentDifference", "SelfDifferenceSummary",
    "TemporalDifferenceConfig", "TemporalDifferenceState", "TemporalSelfDifferenceSystem",
    "compare_emotional_diff", "compare_responsibility_diff",
    "compare_tendency_diff", "compare_direction_diff", "compare_value_diff",
    "determine_magnitude", "determine_nature",
    "generate_difference_tags", "get_difference_summary", "get_difference_for_introspection",
    "save_difference_history", "load_difference_history",
    "create_difference_config", "create_empty_difference_summary",
    # Continuity Strain (自己連続性負荷)
    "StrainPresence", "StrainLevel", "StrainPersistence", "StrainTrend",
    "StrainState", "DifferenceObservation",
    "ContinuityStrainConfig", "ContinuityStrainInternalState", "ContinuityStrainSystem",
    "determine_strain_level", "determine_strain_persistence", "determine_strain_trend",
    "get_average_magnitude", "generate_strain_description",
    "generate_strain_tags", "get_strain_summary", "get_strain_for_introspection",
    "save_strain_state", "load_strain_state",
    "create_strain_config", "create_empty_strain",
    "verify_no_decision_impact", "verify_no_correction_mechanism",
    # Self-Image Integration (自己像統合システム)
    "EmotionalTone", "TendencyHint", "StabilityFeeling",
    "ChangePresence", "ContinuityFeeling", "OverallImpression",
    "ImageAspect", "ProvisionalSelfImage", "SelfImageConfig", "SelfImageIntegrationSystem",
    "generate_self_image_tags", "get_self_image_summary", "get_self_image_for_introspection",
    "create_self_image_config", "create_empty_image",
    "verify_image_no_decision_impact", "verify_provisional_nature",
    # Identity Coherence Awareness (自己同一性の揺らぎ認知)
    "CoherenceLevel", "ShiftSource", "OverlapIntensity", "CoherenceTrend",
    "DetectedShift", "ShiftOverlap", "IdentityCoherenceState",
    "IdentityCoherenceConfig", "IdentityCoherenceSystem",
    "detect_temporal_difference_shift", "detect_tendency_change_shift",
    "detect_continuity_strain_shift", "detect_value_instability_shift",
    "detect_self_image_flux_shift", "detect_emotional_turbulence_shift",
    "determine_overlap_intensity", "determine_coherence_level", "determine_coherence_trend",
    "generate_coherence_tags", "get_coherence_summary", "get_coherence_for_introspection",
    "create_coherence_config", "create_empty_coherence_state",
    "verify_coherence_no_decision_impact", "verify_no_self_preservation",
    "verify_no_identity_definition",
    # Self-Narrative Formation (自己物語形成)
    "FragmentType", "LinkType", "VividnessLevel",
    "NarrativeCoherence", "NarrativeTrend",
    "NarrativeFragment", "FragmentLink", "CoherenceInfo",
    "NarrativeState", "SelfNarrativeConfig", "SelfNarrativeSystem",
    "classify_emotion_observation", "classify_memory_observation",
    "classify_tendency_observation", "classify_difference_observation",
    "classify_context_observation",
    "check_for_rewrites", "summarize_fragments",
    "compute_coherence", "determine_narrative_trend",
    "generate_narrative_description",
    "observe_from_chain",
    "generate_narrative_tags", "get_narrative_summary",
    "get_narrative_for_introspection",
    "create_narrative_config", "create_empty_narrative_state",
    "verify_narrative_no_decision_impact", "verify_narrative_no_identity_definition",
    "verify_no_goal_generation", "verify_read_only_principle",
    # Episodic Memory (エピソード記憶 - 自伝的記憶)
    "EpisodeType", "ImportanceLevel", "DecayState", "EpisodeLinkType", "SearchMode",
    "EmotionalCompanion", "SelfObservationCompanion", "EpisodeLink",
    "EpisodeEntry", "EpisodeStore", "EpisodicMemoryConfig", "EpisodicMemorySystem",
    "determine_decay_state", "classify_episode_type", "compute_importance",
    "compute_topic_overlap", "compute_emotional_similarity", "compute_temporal_proximity",
    "generate_episode_summary",
    "record_episode_from_chain",
    "generate_episodic_memory_tags", "get_episodic_memory_summary",
    "get_episodic_memory_for_introspection",
    "save_episodic_memory", "load_episodic_memory",
    "create_episodic_memory_config", "create_empty_episode_store", "create_episodic_memory_system",
    "verify_episodic_no_decision_impact", "verify_episodic_no_goal_generation",
    "verify_episodic_read_only_principle", "verify_no_value_modification",
    # Introspection Consumption Layer (内省の消費層)
    "FragmentSourceType", "BundleCoherence", "FragmentFreshness",
    "IntrospectionFragment", "FragmentBundle", "ConsumptionRecord",
    "ConsumptionStore", "ConsumptionLayerConfig", "IntrospectionConsumptionSystem",
    "determine_freshness",
    "extract_from_introspection", "extract_from_narrative",
    "extract_from_coherence", "extract_from_tendency", "extract_from_episodic",
    "compute_fragment_relevance", "determine_bundle_coherence", "generate_bundles",
    "consume_introspection_from_chain",
    "generate_consumption_tags", "get_consumption_summary",
    "get_consumption_for_introspection",
    "save_consumption_state", "load_consumption_state",
    "create_consumption_config", "create_empty_consumption_store", "create_consumption_system",
    "verify_consumption_no_decision_impact", "verify_consumption_no_goal_generation",
    "verify_consumption_read_only_principle", "verify_consumption_no_value_modification",
    # Expectation Formation (予期・期待の形成)
    "ExpectationSourceType", "ExpectationBasis", "ExpectationStrength", "ExpectationFreshness",
    "EvidenceLink", "ExpectationCandidate", "ExpectationStore",
    "ExpectationFormationConfig", "ExpectationFormationSystem",
    "determine_expectation_freshness_level", "determine_strength_level",
    "extract_expectation_from_tendency", "extract_expectation_from_difference",
    "extract_expectation_from_narrative",
    "compute_evidence_strength", "detect_competitions",
    "determine_expectation_basis", "generate_expectation_description",
    "form_expectations_from_chain",
    "generate_expectation_tags", "get_expectation_summary",
    "get_expectation_for_introspection",
    "save_expectation_state", "load_expectation_state",
    "create_expectation_config", "create_empty_expectation_store", "create_expectation_system",
    "verify_expectation_no_decision_impact", "verify_expectation_no_goal_generation",
    "verify_expectation_read_only_principle", "verify_expectation_no_value_modification",
    # Other Agent Model (他者モデル)
    "ObservationSourceType", "InferenceBasis", "HypothesisStrength", "HypothesisFreshness",
    "ObservationLink", "OtherStateHypothesis", "SelfOtherBoundary",
    "OtherModelStore", "OtherAgentModelConfig", "OtherAgentModelSystem",
    "determine_hypothesis_freshness_level", "determine_hypothesis_strength_level",
    "extract_from_external_context", "extract_from_reaction_log", "extract_from_self_contrast",
    "compute_observation_strength", "detect_hypothesis_competitions",
    "determine_inference_basis", "generate_hypothesis_description",
    "compute_self_other_boundary",
    "observe_other_from_chain",
    "generate_other_model_tags", "get_other_model_summary",
    "get_other_model_for_introspection",
    "save_other_model_state", "load_other_model_state",
    "create_other_model_config", "create_empty_other_model_store", "create_other_model_system",
    "verify_other_model_no_decision_impact", "verify_other_model_no_goal_generation",
    "verify_other_model_read_only_principle", "verify_other_model_no_value_modification",
    "verify_no_intent_assertion",
    # Emotional Memory Binding (感情記憶の紐づけ)
    "BindingSourceType", "TraceAffinity", "TraceStrength", "TraceFreshness",
    "EmotionalTrace", "BindingLink", "MemoryBinding",
    "BindingStore", "EmotionalMemoryBindingConfig", "EmotionalMemoryBindingSystem",
    "determine_binding_freshness_level", "determine_binding_strength_level",
    "generate_memory_key",
    "extract_binding_from_stm", "extract_binding_from_emotion_state",
    "extract_binding_from_recalled_memories", "extract_binding_from_episodes",
    "compute_binding_strength", "detect_trace_coexistence",
    "compute_emotional_accompaniment",
    "bind_from_chain",
    "generate_binding_tags", "get_binding_summary",
    "get_binding_for_introspection",
    "save_binding_state", "load_binding_state",
    "create_binding_config", "create_empty_binding_store", "create_binding_system",
    "verify_binding_no_decision_impact", "verify_binding_no_goal_generation",
    "verify_binding_read_only_principle", "verify_binding_no_value_modification",
    "verify_no_emotion_evaluation",
    # Intrinsic Motivation (自発的内的動機)
    "MotiveSourceType", "MotiveAffinity", "MotiveStrength", "MotiveFreshness",
    "MotiveImpulse", "MotiveLink", "MotiveEntry",
    "MotiveStore", "IntrinsicMotivationConfig", "IntrinsicMotivationSystem",
    "determine_motive_freshness_level", "determine_motive_strength_level",
    "generate_motive_key",
    "extract_motive_from_emotion_state", "extract_motive_from_tendencies",
    "extract_motive_from_goal_vectors", "extract_motive_from_goal_candidates",
    "compute_motive_strength", "detect_motive_coexistence",
    "compute_motive_overlay",
    "sense_motives_from_chain",
    "generate_motive_tags", "get_motive_summary",
    "get_motive_for_introspection",
    "save_motive_state", "load_motive_state",
    "create_motive_config", "create_empty_motive_store", "create_motive_system",
    "verify_motive_no_decision_impact", "verify_motive_no_goal_generation",
    "verify_motive_read_only_principle", "verify_motive_no_value_modification",
    "verify_no_motivation_prescription",
    # Memory System Integration (記憶系統統合)
    "MemorySource", "TemporalPhase",
    "UnifiedMemoryUnit", "DuplicateEntry", "ConflictEntry", "ReferenceHistoryEntry",
    "IntegrationContext", "IntegrationConfig", "IntegrationState", "IntegrationResult",
    "MemorySystemIntegrator",
    "create_memory_integrator", "create_integration_config",
    "get_integration_summary", "get_integration_summary_text",
    "normalize_episodic", "normalize_long_term", "normalize_bindings",
    "detect_memory_duplicates", "detect_conflicts", "check_conflict_health",
    # Other Model Real Feed (他者モデルリアルフィード統合)
    "ObservationFragmentType", "RealFeedFragmentFreshness",
    "AlignmentStatus", "ConflictStatus",
    "ObservationFragment", "ObservationUnit", "ConflictRecord",
    "FeedHistoryEntry", "HoldbackEntry",
    "RealFeedConfig", "RealFeedState", "FeedResult",
    "extract_speech_reaction", "extract_response_interval",
    "extract_topic_transition", "extract_emotional_tone",
    "extract_continued_engagement", "extract_rejection_acceptance",
    "extract_context_alignment", "extract_recent_history",
    "normalize_fragments", "align_units",
    "detect_feed_duplicates", "detect_feed_conflicts",
    "apply_freshness", "suppress_recent_series",
    "ensure_type_diversity", "check_convergence", "check_stagnation",
    "RealFeedProcessor", "create_real_feed_processor",
    "enhance_context_with_feed", "get_real_feed_summary",
    # Other Model Input Supply (他者モデル入力供給)
    "SupplyEntry", "ContextSnapshot", "ReactionBufferEntry", "InputSupplyState",
    "create_input_supply", "update_input_supply",
    "decay_buffer", "supply_context", "supply_reaction_log",
    "get_input_supply_summary",
    # Policy Candidate Expansion (ポリシー候補拡張)
    "CrossSection", "PolicyAxis", "InputFragment",
    "ExpandedCandidate", "HistoryEntry", "SuppressionEntry", "CompetitionEntry",
    "ExpansionConfig", "ExpansionState", "CrossSectionInputs",
    "PolicyCandidateExpander",
    "create_policy_expander", "create_expansion_config",
    "extract_all_fragments", "get_expansion_summary", "get_expansion_summary_text",
    # Text Dialogue Input (テキスト対話入力経路)
    "InputRouteType", "InputFreshness", "NormalizationStatus",
    "ContextLinkStatus", "DuplicateStatus", "RouteConflictStatus",
    "InputUnit", "ContextLink", "DuplicateRecord", "RouteConflict",
    "ReceiveHistoryEntry", "SuppressionHistoryEntry", "DecayHistoryEntry",
    "TextDialogueConfig", "TextDialogueState", "HandoffResult",
    "receive_input", "normalize_unit", "attach_context",
    "align_to_percept_format", "detect_text_duplicates", "prepare_handoff",
    "apply_freshness_decay", "decay_receive_history", "decay_suppression_history",
    "suppress_recent_adoption", "check_empty_streak",
    "detect_single_route_dominance", "ensure_format_diversity",
    "restore_multi_route", "filter_circular_reference",
    "TextDialogueProcessor", "merge_with_percept",
    "get_text_dialogue_summary", "create_text_dialogue_processor",
    # Spontaneous Activation (自発性の追加)
    "ActivationSourceType", "ActivationFreshness", "CandidateStatus",
    "SuppressionMode", "ConflictResolution",
    "ActivationFragment", "ActivationCandidate", "ActivationRationale",
    "SpontaneousSuppressionEntry", "StandbyEntry",
    "SpontaneousConflictHistoryEntry", "UnadoptedHistoryEntry",
    "ContinuousActivationEntry", "SpontaneousDecayEntry",
    "SpontaneousConfig", "SpontaneousState", "ActivationResult",
    "extract_intrinsic_motivation", "extract_direction_vector",
    "extract_unfinished_intent", "extract_memory_echo",
    "extract_emotional_transition", "extract_activation_responsibility",
    "extract_recent_action", "extract_external_input_absence",
    "extract_all_activation_fragments",
    "form_candidates", "align_conditions", "resolve_conflicts",
    "check_activation_feasibility",
    "suppress_consecutive_series", "apply_overdense_cooldown",
    "restore_candidate_diversity",
    "apply_spontaneous_freshness_decay", "decay_unadopted_history",
    "SpontaneousActivationProcessor",
    "get_spontaneous_summary", "create_spontaneous_processor",
    # Self-Action Perception (自己行動知覚)
    "RecordStatus", "SelfActionRecord",
    "SelfActionPerceptionState", "SelfActionPerceptionConfig",
    "SelfActionPerceptionRecorder",
    "get_self_action_summary", "create_self_action_perception_recorder",
    # Intent-Action Gap (意図-行動間の乖離認知)
    "GapRecord", "IntentActionGapState", "IntentActionGapConfig",
    "IntentActionGapRecorder",
    "get_gap_summary", "create_intent_action_gap_recorder",
    # Temporal Cognition (時間認知構造)
    "DensityLevel", "ElapsedRecord",
    "TemporalCognitionState", "TemporalCognitionConfig",
    "TemporalCognitionProcessor",
    "get_temporal_summary", "create_temporal_cognition",
    "SECTION_ORDER",
    "SECTION_ACTIVITY_DENSITY", "SECTION_MEMORY_INTERVAL",
    "SECTION_EMOTION_FREQUENCY", "SECTION_NARRATIVE_INTERVAL",
    "SECTION_EXTERNAL_INPUT_INTERVAL", "SECTION_OVERALL_ELAPSED",
    "SECTION_LABELS", "DENSITY_LABELS",
    # Multi-Path Recall (記憶の多経路想起)
    "RecallPathLabel", "RecallCandidate", "PathStatistics",
    "RecallEmotionSnapshot", "RecallContextSnapshot", "RecallTemporalSnapshot",
    "MultiPathRecallState", "MultiPathRecallConfig", "MultiPathRecallProcessor",
    "get_recall_summary", "create_multi_path_recall",
    # Introspection Cross-Section (内省断面間の横断的記述)
    "CrossSectionSnapshot",
    "IntrospectionCrossSectionState", "IntrospectionCrossSectionConfig",
    "IntrospectionCrossSectionProcessor",
    "create_introspection_cross_section",
    "INTROSPECTION_CS_SECTION_ORDER", "INTROSPECTION_CS_SECTION_LABELS",
    # Perceptual Context (知覚入力の内部文脈化)
    "PerceptualSummary",
    "PerceptualContextState", "PerceptualContextConfig",
    "PerceptualContextProcessor",
    "create_perceptual_context", "get_perceptual_context_summary",
    "PERCEPTUAL_CTX_SECTION_ORDER", "PERCEPTUAL_CTX_SECTION_LABELS",
    # Value Orientation Validation (価値方向性実運用検証)
    "ValidationObservationSourceType", "ValidationObservationFreshness",
    "DifferentialType", "ValidationStatus",
    "ValidationObservationRecord", "ValidationDescriptionUnit",
    "DifferentialEntry", "ValidationTimeSeriesEntry",
    "ValidationInputs", "ValidationState", "ValidationResult", "ValidationConfig",
    "ValueOrientationValidator",
    "get_validation_summary", "create_validation_processor",
    # Memory Forgetting and Fixation (記憶の忘却と固定化)
    "ForgettingObservationSourceType",
    "ForgettingStage", "FixationLevel", "SeriesStatus",
    "MemorySeriesRecord", "ForgettingCandidate", "FixationSign",
    "ForgettingFixationInputs", "ForgettingFixationState",
    "ForgettingFixationResult", "ForgettingFixationConfig",
    "MemoryForgettingFixationProcessor",
    "get_forgetting_fixation_summary", "create_forgetting_fixation_processor",
    # Spontaneous Recall (記憶の自発的想起 - 非参照型想起)
    "SpontaneousRecallPathLabel", "SpontaneousRecallCandidate",
    "SpontaneousRecallPathStatistics", "SpontaneousRecallEmotionSnapshot",
    "InternalStateCrossSections", "SpontaneousRecallState",
    "SpontaneousRecallConfig", "SpontaneousRecallProcessor",
    "extract_recall_cross_sections", "get_spontaneous_recall_summary",
    "create_spontaneous_recall",
    # Action Result Observation (行動-結果の観測と蓄積)
    "ObservationSection", "ActionResultFreshnessStage",
    "PairStatus", "ActionResultConvergenceLevel",
    "SectionDescription", "ActionDescription", "ResultDescription", "ContextAttribution",
    "ActionResultPair", "SectionWeightRecord", "ActionResultConvergenceRecord",
    "ActionResultInputs", "ActionResultObservationState",
    "ActionResultObservationResult", "ActionResultConfig",
    "ActionResultObservationProcessor",
    "get_action_result_summary", "create_action_result_processor",
    # Other Model Dialogue Learning (他者観測の長期蓄積と仮説補助)
    "DialogueLearningInputSection", "DialogueLearningFreshnessStage",
    "DialogueLearningEntryStatus", "PatternType", "DialogueLearningConvergenceLevel",
    "AccumulationEntry", "PatternRecord", "HypothesisMaterial",
    "DialogueLearningConvergenceRecord",
    "DialogueLearningInputs", "DialogueLearningState",
    "DialogueLearningResult", "DialogueLearningConfig",
    "DialogueLearningProcessor",
    "get_dialogue_learning_summary", "create_dialogue_learning_processor",
    # Meta-Emotion Cognition (メタ感情認知と変動候補生成)
    "MetaEmotionInputSection", "MetaEmotionFreshnessStage",
    "MetaEmotionRecordStatus", "MetaEmotionConvergenceLevel",
    "TransitionFeature", "SustainedPattern", "VariationCandidate",
    "CognitionRecord", "MetaEmotionConvergenceRecord",
    "MetaEmotionInputs", "MetaEmotionState",
    "MetaEmotionResult", "MetaEmotionConfig",
    "MetaEmotionProcessor",
    "get_meta_emotion_summary", "create_meta_emotion_processor",
    # Scoring Fluctuation (スコアリングの構造的揺らぎ)
    "ScoringFluctuationConfig",
    "apply_scoring_fluctuation",
    "extract_emotion_variability", "extract_stm_variability",
    "extract_drive_variability", "extract_elapsed_variability",
    "compose_variability", "limit_amplitude",
    "generate_per_policy_fluctuations", "apply_fluctuations_to_candidates",
    "extract_stm_info", "get_fluctuation_summary", "create_fluctuation_config",
    # Selection Attribution (選択帰属)
    "SelectionRecord", "SelectionAttributionState", "SelectionAttributionConfig",
    "SelectionAttributionRecorder",
    "create_selection_attribution_recorder", "get_selection_attribution_summary",
    # Reference Frequency Description (参照頻度の構造的記述)
    "ReferenceFrequencyConfig", "ReferenceFrequencyState",
    "ReferenceSnapshot", "VariationDescription",
    "process_reference_frequency", "collect_reference_counts",
    "create_reference_frequency_state", "create_reference_frequency_config",
    "get_latest_snapshot", "get_snapshot_history",
    "get_latest_variation", "get_reference_summary",
    # Persistent Commitment (持続的取り組み保持構造)
    "PersistentCommitmentConfig", "PersistentCommitmentState",
    "PersistentCommitmentProcessor", "CommitmentItem", "CognitionRecord",
    "CommitmentCrossSectionInputs",
    "create_persistent_commitment_processor", "get_commitment_summary",
    # Stabilization Description (安定化の構造的記述)
    "StabilizationDescriptionConfig", "StabilizationDescriptionState",
    "StabilizationRecord",
    "process_stabilization_description",
    "read_signal_sources", "read_diff_reference",
    "compose_stabilization_record", "accumulate_stabilization_record",
    "get_latest_stabilization_record", "get_stabilization_record_history",
    "get_stabilization_summary",
    "create_stabilization_description_state", "create_stabilization_description_config",
    "save_stabilization_state", "load_stabilization_state",
    "SIGNAL_EMOTION", "SIGNAL_STM_ENTRIES", "SIGNAL_TRANSIENT_GOAL",
    "SIGNAL_PERSISTENT_COMMITMENT", "SIGNAL_SPONTANEOUS_CANDIDATE",
    "SIGNAL_EXTERNAL_INPUT", "ALL_SIGNAL_KEYS",
    # Behavioral Diversity Description (行動多様性の構造的記述)
    "BehavioralDiversityConfig", "BehavioralDiversityState",
    "DiversityRecord", "TypeCountLevel", "DispersionLevel",
    "determine_type_count_level", "determine_dispersion_level",
    "process_behavioral_diversity",
    "read_section_key_types", "read_policy_label_types", "read_candidate_size_types",
    "compose_diversity_record", "accumulate_diversity_record",
    "get_latest_diversity_record", "get_diversity_record_history",
    "get_diversity_summary",
    "create_behavioral_diversity_state", "create_behavioral_diversity_config",
    "save_behavioral_diversity_state", "load_behavioral_diversity_state",
    # Internal Contradiction Description (内部状態の矛盾並置の構造的記述)
    "ContradictionRecord", "ContradictionInputs",
    "ContradictionState", "ContradictionResult", "ContradictionConfig",
    "InternalContradictionProcessor",
    "get_contradiction_summary", "create_contradiction_processor",
    "CONTRADICTION_PAIR_DEFINITIONS", "CONTRADICTION_PAIR_LABELS",
    # Interaction Accumulation (相互作用の蓄積記述)
    "AdjacentPair", "BufferEntry",
    "InteractionAccumulationState", "InteractionAccumulationConfig",
    "InteractionAccumulationProcessor",
    "get_interaction_summary", "create_interaction_accumulation_processor",
    # Emotional Backdrop Cognition (感情基調の持続認知)
    "BackdropInputSection", "BackdropFreshnessStage", "BackdropConvergenceLevel",
    "BackdropWindowEntry", "BackdropCompositionRecord", "BackdropConvergenceRecord",
    "BackdropInputs", "BackdropState", "BackdropResult", "BackdropConfig",
    "EmotionalBackdropProcessor",
    "get_backdrop_summary", "create_emotional_backdrop_processor",
    # Situational Self-Presentation (状況依存的自己呈示の認知)
    "PresentationRecordFreshness", "PresentationTypeCountLevel", "PresentationConvergenceLevel",
    "PresentationRecord", "CompositionDescription", "PresentationConvergenceRecord",
    "SituationalSelfPresentationState", "SituationalSelfPresentationConfig",
    "SituationalSelfPresentationProcessor",
    "determine_presentation_type_count_level",
    "get_presentation_summary", "create_situational_self_presentation_processor",
    # Drive Variation Description (駆動の変動記述)
    "DriveVariationInputSection", "DriveVariationFreshnessStage", "DriveVariationConvergenceLevel",
    "DriveVariationWindowEntry", "DriveVariationCompositionRecord",
    "DriveVariationDecayRecord", "DriveVariationConvergenceRecord",
    "DriveVariationInputs", "DriveVariationState", "DriveVariationResult", "DriveVariationConfig",
    "DriveVariationProcessor",
    "get_drive_variation_summary", "create_drive_variation_processor",
    # Expectation Lifecycle Description (予期の成立・消失の事後記述)
    "LifecycleTransitionType", "LifecycleRecordFreshness", "LifecycleConvergenceLevel",
    "LifecycleTransitionRecord", "LifecycleView",
    "LifecycleConvergenceRecord", "LifecycleSnapshotEntry",
    "ExpectationLifecycleState", "ExpectationLifecycleConfig",
    "ExpectationLifecycleDescriptionProcessor",
    "get_lifecycle_summary", "create_expectation_lifecycle_processor",
    # Input Pathway Balance (入力経路間の均衡記述)
    "InputPathwayBalanceConfig", "InputPathwayBalanceState",
    "UsageFact", "PathwaySnapshot", "PathwayVariation",
    "UsageLevel", "BiasLevel",
    "PATHWAY_TEXT", "PATHWAY_SCREEN", "PATHWAY_SPONTANEOUS", "ALL_PATHWAYS",
    "determine_usage_level", "determine_bias_level", "compute_bias_value",
    "collect_usage_fact", "read_text_dialogue_usage", "read_spontaneous_usage",
    "compose_snapshot", "derive_variation",
    "process_input_pathway_balance",
    "get_latest_pathway_snapshot", "get_pathway_snapshot_history",
    "get_latest_pathway_variation", "get_pathway_balance_summary",
    "get_pathway_balance_enrichment_text",
    "save_input_pathway_balance_state", "load_input_pathway_balance_state",
    "create_input_pathway_balance_state", "create_input_pathway_balance_config",
    # Responsibility Temporal Trace (責任の時間的推移記述)
    "VariationLevel",
    "ResponsibilitySnapshot",
    "ResponsibilityTemporalTraceState", "ResponsibilityTemporalTraceConfig",
    "ResponsibilityTemporalTraceProcessor",
    "get_trace_summary", "create_responsibility_temporal_trace",
    "RESPONSIBILITY_TRACE_SECTION_ORDER",
    "SECTION_TOTAL_WEIGHT_VARIATION", "SECTION_PENDING_DECISIONS_RETENTION",
    "SECTION_HARM_VARIATION", "SECTION_CONFIDENCE_VARIATION",
    "SECTION_DISPERSION_ACTIVITY_DENSITY",
    "RESPONSIBILITY_TRACE_SECTION_LABELS", "VARIATION_LABELS",
    # Other Boundary Accumulation (他者境界の多相蓄積)
    "BoundaryFreshnessStage", "DivergenceLevel", "BoundaryConvergenceLevel",
    "BoundaryRecord", "BoundaryConvergenceRecord",
    "OtherBoundaryAccumulationState", "BoundaryAccumulationResult",
    "OtherBoundaryAccumulationConfig", "OtherBoundaryAccumulationProcessor",
    "determine_divergence_level",
    "get_boundary_accumulation_summary", "create_other_boundary_accumulation_processor",
    # Emotion cooccurrence description (感情間の共起記述)
    "CooccurrenceFreshnessStage", "CooccurrenceDiversityLevel",
    "CooccurrencePair", "CooccurrenceRecord",
    "CooccurrenceState", "CooccurrenceResult", "CooccurrenceConfig",
    "EmotionCooccurrenceDescriptionProcessor",
    "get_cooccurrence_summary", "create_cooccurrence_processor",
    # Attention Distribution Description (注意配分の構造的記述)
    "AttentionDistributionConfig", "AttentionDistributionState",
    "AttentionSnapshot", "AttentionVariation",
    "QuantityLevel", "ConcentrationLevel",
    "SOURCE_PERCEPTION", "SOURCE_TEXT_INPUT", "SOURCE_SPONTANEOUS",
    "ATTENTION_SIGNAL_EMOTION", "ATTENTION_SIGNAL_MEMORY",
    "ATTENTION_SIGNAL_MOTIVATION", "ATTENTION_SIGNAL_GOAL",
    "ATTENTION_SIGNAL_RESPONSIBILITY", "ALL_SOURCE_KEYS",
    "determine_quantity_level", "determine_concentration_level",
    "compute_attention_concentration", "collect_source_quantities",
    "compose_attention_snapshot", "derive_attention_variation",
    "process_attention_distribution",
    "get_latest_attention_snapshot", "get_attention_snapshot_history",
    "get_latest_attention_variation", "get_attention_distribution_summary",
    "get_attention_distribution_enrichment_text",
    "save_attention_distribution_state", "load_attention_distribution_state",
    "create_attention_distribution_state", "create_attention_distribution_config",
    # Forgetting-Recall Balance (忘却と想起の均衡記述)
    "ForgettingRecallBalanceConfig", "ForgettingRecallBalanceState",
    "ForgettingSectionSnapshot", "ExternalRecallSectionSnapshot",
    "SpontaneousRecallSectionSnapshot", "JuxtapositionEntry",
    "extract_forgetting_section", "extract_external_recall_section",
    "extract_spontaneous_recall_section",
    "compose_juxtaposition", "accumulate_entry",
    "process_forgetting_recall_balance",
    "get_frb_recent_entries", "get_frb_history",
    "get_frb_balance_summary", "get_frb_enrichment_text",
    "save_frb_state", "load_frb_state",
    "create_forgetting_recall_balance_state", "create_forgetting_recall_balance_config",
    # Orchestrator (全モジュール統合管理)
    "PsycheOrchestrator",
]
