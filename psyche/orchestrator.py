"""
psyche/orchestrator.py - PsycheOrchestrator: 全モジュール統合管理

全 psyche モジュールの初期化・毎ティック実行・プロンプト生成・永続化を一元管理する。
brain.py からは本クラスのみを参照すればよく、個別モジュールへの直接依存を排除する。

実行モデル:
- 毎ティック: react_with_stm, dynamics, attachment, responsibility, self_reference,
              repeated_tendency, fear_recompute
- 3ティック毎: tendency_awareness → self_model → goals → intrinsic_motivation
- 5ティック毎: temporal_diff → strain → self_image → coherence → narrative →
               episodic → binding → memory_integration → introspection →
               consumption → expectation → other_model → value_orientation →
               value_orientation_validation
- 10ティック毎: stability_valve → long_term_dynamics → snapshot
- プロンプト生成前: thought → decision_bias → tone → context_sensitivity →
                    silence_hesitation → stability_valve (bias application)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .state import PsycheState, Percept, EmotionVector
from .pillars import (
    AttachmentState,
    ContinuityState,
    FearIndex,
    IdentityState,
    ProjectionState,
)

# Core reaction pipeline
from .reaction_with_stm import react_with_stm
from .short_term_loop import LoopState, create_loop_state

# Dynamics
from .dynamics import (
    DynamicsState,
    create_dynamics_state,
    update_dynamics,
    get_dynamics_summary,
)

# Emotion amplitude
from .emotion_amplitude import (
    AmplitudeState,
    create_amplitude_state,
    update_amplitude,
    decay_amplitude,
)

# Multi-emotion
from .multi_emotion import (
    MultiEmotionConfig,
    apply_independent_decay,
    get_active_emotions,
)

# STM-Emotion coupling
from .stm_emotion_coupling import (
    STMEmotionCouplingConfig,
    CouplingInfluence,
    apply_stm_coupling,
    get_coupling_summary,
)

# Pillar managers
from . import attachment_manager, identity_manager, continuity_manager, projection_manager

# Fear
from .fear import compute_fear_index

# Responsibility
from .responsibility_manager import ResponsibilityManager
from .responsibility import ResponsibilityInfluence, get_influence as get_resp_influence

# Self-reference
from .self_reference import (
    SelfReferenceConfig,
    SelfReferenceState,
    execute_self_reference,
)

# Decision bias
from .decision_bias import DecisionBias, DecisionBiasConfig, compute_decision_bias

# Thought / policy
from .thought import generate_thought_candidates, select_policy

# Tone
from .tone import (
    ToneConfig,
    ToneModifier,
    compute_tone_bias,
)

# Context sensitivity
from .context_sensitivity import (
    ExternalContext,
    ContextSensitivityConfig,
    ContextState,
    SensitivityBias,
    compute_sensitivity_bias,
    apply_sensitivity_to_candidates,
)

# Silence / Hesitation
from .silence_hesitation import (
    SilenceConfig,
    generate_candidates_with_silence,
)

# Stability valve
from .stability_valve import (
    StabilityValve,
    StabilityBias,
    apply_stability_bias,
    create_stability_valve,
)

# Value orientation
from .value_orientation import (
    ValueOrientation,
    create_orientation,
    update_orientation,
    update_from_decision,
    generate_emotion_signal,
    generate_responsibility_signal,
    apply_orientation_to_candidates,
    get_orientation_summary,
)

# Value orientation validation
from .value_orientation_validation import (
    ValueOrientationValidator,
    ValidationState as VOValidationState,
    ValidationInputs as VOValidationInputs,
    ValidationResult as VOValidationResult,
    create_validation_processor as create_vo_validator,
    get_validation_summary as get_vo_validation_summary,
)

# Repeated tendency
from .repeated_tendency import RepeatedTendencySystem, RepeatedTendencyState
from .repeated_tendency import create_system as create_tendency_system

# Tendency awareness
from .tendency_awareness import (
    TendencyAwareness,
    observe_tendencies,
    get_awareness_summary,
)

# Self-model
from .self_model import (
    SelfModelSystem,
    SelfStateView,
    get_self_model_summary,
)

# Temporal self-difference
from .temporal_self_difference import (
    TemporalSelfDifferenceSystem,
    SelfDifferenceSummary,
    get_difference_summary,
)

# Continuity strain
from .continuity_strain import (
    ContinuityStrainSystem,
    StrainState,
    get_strain_summary,
)

# Self-image integration
from .self_image_integration import (
    SelfImageIntegrationSystem,
    ProvisionalSelfImage,
    get_self_image_summary,
)

# Identity coherence
from .identity_coherence import (
    IdentityCoherenceSystem,
    IdentityCoherenceState,
    get_coherence_summary,
)

# Self-narrative
from .self_narrative import (
    SelfNarrativeSystem,
    NarrativeState,
    observe_from_chain as observe_narrative_from_chain,
    get_narrative_summary,
)

# Episodic memory
from .episodic_memory import (
    EpisodicMemorySystem,
    EpisodeStore,
    record_from_chain as record_episode_from_chain,
    get_episodic_memory_summary,
)

# Emotional memory binding
from .emotional_memory_binding import (
    EmotionalMemoryBindingSystem,
    BindingStore,
    bind_from_chain,
    get_binding_summary,
)

# Introspection trace
from .introspection_trace import (
    IntrospectionSystem,
    TraceLog,
    get_trace_summary,
)

# Introspection consumption
from .introspection_consumption import (
    IntrospectionConsumptionSystem,
    ConsumptionStore,
    consume_from_chain as consume_introspection_from_chain,
    get_consumption_summary,
)

# Expectation formation
from .expectation_formation import (
    ExpectationFormationSystem,
    ExpectationStore,
    form_from_chain as form_expectations_from_chain,
    get_expectation_summary,
)

# Intrinsic motivation
from .intrinsic_motivation import (
    IntrinsicMotivationSystem,
    MotiveStore,
    sense_from_chain as sense_motives_from_chain,
    get_motive_summary,
)

# Other agent model
from .other_agent_model import (
    OtherAgentModelSystem,
    OtherModelStore,
    observe_from_chain as observe_other_from_chain,
    get_other_model_summary,
)

# Other model input supply
from .other_model_input_supply import (
    InputSupplyState,
    create_input_supply,
    update_from_percept as update_input_supply,
    decay_buffer,
    supply_context,
    supply_reaction_log,
    get_input_supply_summary,
)

# Proto-goal vector
from .proto_goal_vector import (
    VectorGenerator,
    VectorState,
    get_vector_summary,
)

# Goal candidates
from .goal_candidates import (
    CandidateGenerator,
    CandidateState,
    get_candidate_summary,
)

# Transient goal
from .transient_goal import (
    TransientGoalManager,
    TransientGoalState,
    get_transient_goal_summary,
)

# Scoped goal
from .scoped_goal import (
    ScopedGoalSystem,
    get_scoped_goal_summary,
)

# Long-term dynamics
from .long_term_dynamics import (
    DynamicsObserver,
    create_observer as create_dynamics_observer,
    get_observer_summary,
)

# Dispersion
from .responsibility_dispersion import (
    DispersionState,
    create_dispersion_state,
    to_dict as dispersion_to_dict,
    from_dict as dispersion_from_dict,
    get_dispersion_summary,
    get_active_units as get_dispersion_active_units,
    get_total_active_weight as get_dispersion_active_weight,
)

# Policy candidate expansion
from .policy_candidate_expansion import (
    PolicyCandidateExpander,
    ExpansionState,
    ExpansionConfig,
    CrossSectionInputs,
    create_expander as create_policy_expander,
    get_expansion_summary_text,
)

# Memory system integration
from .memory_system_integration import (
    MemorySystemIntegrator,
    IntegrationState,
    IntegrationContext,
    IntegrationResult,
    create_integrator as create_memory_integrator,
    get_integration_summary_text,
)

# Other model real feed
from .other_model_real_feed import (
    RealFeedProcessor,
    RealFeedState,
    FeedResult,
    create_real_feed_processor,
    enhance_context_with_feed,
    get_real_feed_summary,
)
from .text_dialogue_input import (
    TextDialogueProcessor,
    TextDialogueState,
    HandoffResult as TextHandoffResult,
    create_text_dialogue_processor,
    get_text_dialogue_summary,
)

# Memory forgetting & fixation
from .memory_forgetting_fixation import (
    MemoryForgettingFixationProcessor,
    ForgettingFixationState,
    ForgettingFixationInputs,
    ForgettingFixationResult,
    create_forgetting_fixation_processor,
    get_forgetting_fixation_summary,
)

# Action-result observation
from .action_result_observation import (
    ActionResultObservationProcessor,
    ActionResultObservationState,
    ActionResultInputs,
    ActionResultObservationResult,
    create_action_result_processor,
    get_action_result_summary,
)

# Spontaneous activation
from .spontaneous_activation import (
    SpontaneousActivationProcessor,
    SpontaneousState,
    ActivationResult as SpontaneousResult,
    create_spontaneous_processor,
    get_spontaneous_summary,
)

# Other model dialogue learning
from .other_model_dialogue_learning import (
    DialogueLearningProcessor,
    DialogueLearningState,
    DialogueLearningInputs,
    DialogueLearningResult,
    create_dialogue_learning_processor,
    get_dialogue_learning_summary,
)

# Meta-emotion cognition
from .meta_emotion_cognition import (
    MetaEmotionProcessor,
    MetaEmotionState,
    MetaEmotionInputs,
    MetaEmotionResult,
    create_meta_emotion_processor,
    get_meta_emotion_summary,
)

# Self-action perception
from .self_action_perception import (
    SelfActionPerceptionRecorder,
    SelfActionPerceptionState,
    create_self_action_perception_recorder,
    get_self_action_summary,
)

# Situational self-presentation (状況依存的自己呈示の認知)
from .situational_self_presentation import (
    SituationalSelfPresentationProcessor,
    SituationalSelfPresentationState,
    create_situational_self_presentation_processor,
    get_presentation_summary,
)

# Intent-action gap (意図-行動間の乖離認知)
from .intent_action_gap import (
    IntentActionGapRecorder,
    IntentActionGapState,
    create_intent_action_gap_recorder,
    get_gap_summary,
)

# Temporal cognition (時間認知構造)
from .temporal_cognition import (
    TemporalCognitionProcessor,
    TemporalCognitionState,
    create_temporal_cognition,
)

# Multi-path recall (記憶の多経路想起)
from .multi_path_recall import (
    MultiPathRecallProcessor,
    MultiPathRecallState,
    EmotionSnapshot as RecallEmotionSnapshot,
    ContextSnapshot as RecallContextSnapshot,
    TemporalSnapshot as RecallTemporalSnapshot,
    create_multi_path_recall,
)

# Spontaneous recall (記憶の自発的想起 - 非参照型想起)
from .spontaneous_recall import (
    SpontaneousRecallProcessor,
    SpontaneousRecallState,
    InternalEmotionSnapshot as SpontaneousRecallEmotionSnapshot,
    create_spontaneous_recall,
)

# Introspection cross-section (内省断面間の横断的記述)
from .introspection_cross_section import (
    IntrospectionCrossSectionProcessor,
    IntrospectionCrossSectionState,
    create_introspection_cross_section,
    SECTION_SELF_MODEL as ICS_SELF_MODEL,
    SECTION_TEMPORAL_SELF_DIFFERENCE as ICS_TEMPORAL_DIFF,
    SECTION_IDENTITY_COHERENCE as ICS_IDENTITY_COHERENCE,
    SECTION_SELF_NARRATIVE as ICS_SELF_NARRATIVE,
    SECTION_INTROSPECTION_CONSUMPTION as ICS_INTROSPECTION_CONSUMPTION,
    SECTION_META_EMOTION_COGNITION as ICS_META_EMOTION,
)

# Introspection longitudinal view (内省の時間的縦断参照)
from .introspection_longitudinal_view import (
    IntrospectionLongitudinalViewProcessor,
    create_introspection_longitudinal_view,
)

# Perceptual context (知覚入力の内部文脈化)
from .perceptual_context import (
    PerceptualContextProcessor,
    create_perceptual_context,
)

# Scoring fluctuation (スコアリングの構造的揺らぎ)
from .scoring_fluctuation import (
    ScoringFluctuationConfig,
    apply_scoring_fluctuation,
    extract_stm_info,
    get_fluctuation_summary,
    create_fluctuation_config,
)

# Selection attribution (選択帰属)
from .selection_attribution import (
    SelectionAttributionRecorder,
    SelectionAttributionState,
    create_selection_attribution_recorder,
    get_selection_attribution_summary,
)

# Reference frequency description (参照頻度の構造的記述)
from .reference_frequency_description import (
    ReferenceFrequencyConfig,
    ReferenceFrequencyState,
    process_reference_frequency,
    create_reference_frequency_state,
    get_reference_summary,
)

# Persistent commitment (持続的取り組み保持構造)
from .persistent_commitment import (
    PersistentCommitmentProcessor,
    PersistentCommitmentState,
    PersistentCommitmentConfig,
    CommitmentCrossSectionInputs,
    create_persistent_commitment_processor,
    get_commitment_summary,
)

# Internal contradiction description (内部状態の矛盾並置の構造的記述)
from .internal_contradiction_description import (
    InternalContradictionProcessor,
    ContradictionInputs,
    ContradictionResult,
    create_contradiction_processor,
)

# Interaction accumulation (相互作用の蓄積記述)
from .interaction_accumulation import (
    InteractionAccumulationProcessor,
    InteractionAccumulationState,
    create_interaction_accumulation_processor,
    get_interaction_summary,
)

# Emotional backdrop cognition (感情基調の持続認知)
from .emotional_backdrop_cognition import (
    EmotionalBackdropProcessor,
    BackdropInputs,
    BackdropResult,
    BackdropState,
    create_emotional_backdrop_processor,
    get_backdrop_summary,
)

# Expectation lifecycle description (予期の成立・消失の事後記述)
from .expectation_lifecycle_description import (
    ExpectationLifecycleDescriptionProcessor,
    ExpectationLifecycleState,
    create_expectation_lifecycle_processor,
    get_lifecycle_summary,
)

# Goal hierarchy propagation (目的階層間の隣接状態変化記述)
from .goal_hierarchy_propagation import (
    GoalHierarchyPropagationProcessor,
    GoalHierarchyPropagationState,
    create_goal_hierarchy_propagation_processor,
)

# Drive variation description (駆動の変動記述)
from .drive_variation_description import (
    DriveVariationProcessor,
    DriveVariationInputs,
    DriveVariationResult,
    DriveVariationState,
    create_drive_variation_processor,
    get_drive_variation_summary,
)

# Stabilization description (安定化の構造的記述)
from .stabilization_description import (
    StabilizationDescriptionConfig,
    StabilizationDescriptionState,
    process_stabilization_description,
    create_stabilization_description_state,
    get_stabilization_summary,
)

# Behavioral diversity description (行動多様性の構造的記述)
from .behavioral_diversity_description import (
    BehavioralDiversityConfig,
    BehavioralDiversityState,
    process_behavioral_diversity,
    create_behavioral_diversity_state,
    get_diversity_summary,
)

# Input pathway balance (入力経路間の均衡記述)
from .input_pathway_balance import (
    InputPathwayBalanceConfig,
    InputPathwayBalanceState,
    process_input_pathway_balance,
    get_enrichment_text as get_pathway_balance_enrichment_text,
    save_state as save_input_pathway_balance_state,
    load_state as load_input_pathway_balance_state,
    create_input_pathway_balance_state,
    PATHWAY_TEXT,
    PATHWAY_SCREEN,
    PATHWAY_SPONTANEOUS,
)

# Responsibility temporal trace (責任の時間的推移記述)
from .responsibility_temporal_trace import (
    ResponsibilityTemporalTraceProcessor,
    ResponsibilityTemporalTraceState,
    create_responsibility_temporal_trace,
    get_trace_summary as get_responsibility_trace_summary,
)

# Emotion cooccurrence description (感情間の共起記述)
from .emotion_cooccurrence_description import (
    EmotionCooccurrenceDescriptionProcessor,
    CooccurrenceState,
    CooccurrenceResult,
    create_cooccurrence_processor,
    get_cooccurrence_summary,
)

# Other boundary accumulation (他者境界の多相蓄積)
from .other_boundary_accumulation import (
    OtherBoundaryAccumulationProcessor,
    OtherBoundaryAccumulationState,
    BoundaryAccumulationResult,
    create_other_boundary_accumulation_processor,
    get_boundary_accumulation_summary,
)

# Hypothesis-observation pairing (他者モデル仮説の事後検証経路)
from .hypothesis_observation_pairing import (
    HypothesisObservationPairingProcessor,
    HypothesisObservationPairingState as HOPairingState,
    create_hypothesis_observation_pairing_processor,
    get_hypothesis_observation_pairing_summary,
)

# Forgetting-recall balance (忘却と想起の均衡記述)
from .forgetting_recall_balance import (
    ForgettingRecallBalanceConfig,
    ForgettingRecallBalanceState,
    process_forgetting_recall_balance,
    get_enrichment_text as get_frb_enrichment_text,
    save_state as save_frb_state,
    load_state as load_frb_state,
    create_forgetting_recall_balance_state,
)

# Attention distribution description (注意配分の構造的記述)
from .attention_distribution_description import (
    AttentionDistributionConfig as AttDistConfig,
    AttentionDistributionState as AttDistState,
    process_attention_distribution,
    get_enrichment_text as get_att_dist_enrichment_text,
    save_state as save_att_dist_state,
    load_state as load_att_dist_state,
    create_attention_distribution_state,
)


# Enrichment compression (enrichment-to-prompt パイプライン効率化)
from .enrichment_compression import (
    build_compressed_enrichment,
    detect_item_changed,
    apply_item_granularity,
    ORIGINAL_FOOTER,
)

logger = logging.getLogger(__name__)


# ── Orchestrator ──────────────────────────────────────────────────


class PsycheOrchestrator:
    """全 psyche モジュールの統合管理クラス。

    brain.py からはこのクラスのみを参照し、個別モジュールを直接使わない。
    """

    # ── Initialization ────────────────────────────────────────────

    def __init__(
        self,
        memory_count: int = 0,
        data_dir: Optional[Path] = None,
    ):
        """全サブシステムを初期化する。

        Args:
            memory_count: 長期記憶の件数（ContinuityState 初期化用）
            data_dir: データ永続化ディレクトリ（デフォルト: data/）
        """
        self._data_dir = data_dir or (Path(__file__).parent.parent / "data")
        self._tick_count: int = 0

        # ── Core psyche state ──
        identity = IdentityState(
            core_traits=["romantic", "sweet", "playful", "caring", "confident"],
            trait_confidence={
                "romantic": 0.9, "sweet": 0.9, "playful": 0.8,
                "caring": 0.9, "confident": 0.8,
            },
        )
        attachment = AttachmentState()
        continuity = ContinuityState(memory_count=memory_count)
        projection = ProjectionState(
            goals=[{
                "id": "engage",
                "description": "対話相手と関わる",
                "progress": 0.1,
                "status": "active",
            }],
        )
        fear = compute_fear_index(
            identity_risk=identity_manager.calc_identity_risk(identity),
            attachment_risk=attachment_manager.calc_attachment_risk(attachment),
            continuity_risk=continuity_manager.calc_continuity_risk(
                memory_count=continuity.memory_count,
            ),
            projection_risk=projection_manager.calc_projection_risk(projection),
        )
        self._psyche = PsycheState(
            identity=identity,
            attachment=attachment,
            continuity=continuity,
            projection=projection,
            fear_index=fear,
        )

        # ── STM loop state ──
        self._loop_state = create_loop_state()

        # ── Dynamics (peak / rebound) ──
        self._dynamics = create_dynamics_state()

        # ── Responsibility ──
        self._responsibility_mgr = ResponsibilityManager(
            filepath=self._data_dir / "responsibility.json",
        )

        # ── Self-reference ──
        self._self_ref_config = SelfReferenceConfig()
        self._self_ref_state: Optional[SelfReferenceState] = None

        # ── Decision bias ──
        self._decision_bias_config = DecisionBiasConfig()

        # ── Tone ──
        self._tone_config = ToneConfig()

        # ── Context sensitivity ──
        self._ctx_sensitivity_config = ContextSensitivityConfig()
        self._ctx_state = ContextState()

        # ── Silence ──
        self._silence_config = SilenceConfig()

        # ── Stability valve ──
        self._stability_valve = create_stability_valve()

        # ── Value orientation ──
        self._value_orientation = create_orientation()

        # ── Repeated tendency ──
        self._tendency_sys = create_tendency_system()

        # ── Self-model chain ──
        self._self_model_sys = SelfModelSystem()
        self._last_self_view: Optional[SelfStateView] = None
        self._tendency_awareness: Optional[TendencyAwareness] = None

        # ── Temporal self-difference ──
        self._temporal_diff_sys = TemporalSelfDifferenceSystem()
        self._last_diff_summary: Optional[SelfDifferenceSummary] = None

        # ── Continuity strain ──
        self._strain_sys = ContinuityStrainSystem()
        self._last_strain: Optional[StrainState] = None

        # ── Self-image integration ──
        self._self_image_sys = SelfImageIntegrationSystem()
        self._last_self_image: Optional[ProvisionalSelfImage] = None

        # ── Identity coherence ──
        self._coherence_sys = IdentityCoherenceSystem()
        self._last_coherence: Optional[IdentityCoherenceState] = None

        # ── Self-narrative ──
        self._narrative_sys = SelfNarrativeSystem()
        self._last_narrative: Optional[NarrativeState] = None

        # ── Episodic memory ──
        self._episodic_sys = EpisodicMemorySystem()
        self._last_episodes: Optional[EpisodeStore] = None

        # ── Emotional memory binding ──
        self._binding_sys = EmotionalMemoryBindingSystem()
        self._last_bindings: Optional[BindingStore] = None

        # ── Introspection trace ──
        self._introspection_sys = IntrospectionSystem()
        self._last_trace: Optional[TraceLog] = None

        # ── Introspection consumption ──
        self._consumption_sys = IntrospectionConsumptionSystem()
        self._last_consumption: Optional[ConsumptionStore] = None

        # ── Expectation formation ──
        self._expectation_sys = ExpectationFormationSystem()
        self._last_expectations: Optional[ExpectationStore] = None

        # ── Intrinsic motivation ──
        self._motivation_sys = IntrinsicMotivationSystem()
        self._last_motives: Optional[MotiveStore] = None

        # ── Other agent model ──
        self._other_model_sys = OtherAgentModelSystem()
        self._last_other_model: Optional[OtherModelStore] = None

        # ── Other model input supply ──
        self._input_supply = create_input_supply()
        self._last_percept: Optional[Percept] = None

        # ── Recalled memories (brain.py から供給) ──
        self._last_recalled_memories: Optional[list[dict]] = None

        # ── Proto-goal vector ──
        self._vector_gen = VectorGenerator()

        # ── Goal candidates ──
        self._candidate_gen = CandidateGenerator()

        # ── Transient goal ──
        self._transient_goal_mgr = TransientGoalManager()

        # ── Scoped goal ──
        self._scoped_goal_sys = ScopedGoalSystem()

        # ── Long-term dynamics observer ──
        self._dynamics_observer = create_dynamics_observer(
            log_file_path=str(self._data_dir / "long_term_dynamics.json"),
        )

        # ── Dispersion state ──
        self._dispersion_state = create_dispersion_state()

        # ── Emotion amplitude ──
        self._amplitude_state = create_amplitude_state()

        # ── Multi-emotion config ──
        self._multi_emotion_config = MultiEmotionConfig()

        # ── STM-Emotion coupling ──
        self._stm_coupling_config = STMEmotionCouplingConfig()
        self._last_coupling: Optional[CouplingInfluence] = None

        # ── Policy candidate expansion ──
        self._policy_expander = create_policy_expander()

        # ── Memory system integration ──
        self._memory_integrator = create_memory_integrator()

        # ── Other model real feed ──
        self._real_feed_processor = create_real_feed_processor()
        self._last_feed_result: Optional[FeedResult] = None
        self._last_integration_result: Optional[IntegrationResult] = None

        # ── Text dialogue input ──
        self._text_dialogue_processor = create_text_dialogue_processor()
        self._last_text_handoff: Optional[TextHandoffResult] = None

        # ── Spontaneous activation ──
        self._spontaneous_processor = create_spontaneous_processor()
        self._last_activation_result: Optional[SpontaneousResult] = None

        # ── Value orientation validation ──
        self._vo_validator = create_vo_validator()
        self._last_vo_validation: Optional[VOValidationResult] = None
        self._state_for_vo_validation_last_time: float = 0.0

        # ── Memory forgetting & fixation ──
        self._forgetting_fixation_processor = create_forgetting_fixation_processor()
        self._last_forgetting_fixation: Optional[ForgettingFixationResult] = None

        # ── Action-result observation ──
        self._action_result_observer = create_action_result_processor()
        self._last_action_result: Optional[ActionResultObservationResult] = None
        self._last_selected_policy_label: str = ""
        self._last_selected_policy_axis: str = ""
        self._last_emotion_for_action_result: dict[str, float] = {}
        self._expectation_action_diff_log: list[dict] = []

        # ── Other model dialogue learning ──
        self._dialogue_learning_processor = create_dialogue_learning_processor()
        self._last_dialogue_learning: Optional[DialogueLearningResult] = None

        # ── Meta-emotion cognition ──
        self._meta_emotion_processor = create_meta_emotion_processor()
        self._last_meta_emotion: Optional[MetaEmotionResult] = None

        # ── Self-action perception ──
        self._self_action_recorder = create_self_action_perception_recorder()

        # ── Situational self-presentation (状況依存的自己呈示の認知) ──
        self._situational_self_presentation = create_situational_self_presentation_processor()

        # ── Intent-action gap (意図-行動間の乖離認知) ──
        self._intent_action_gap_recorder = create_intent_action_gap_recorder()

        # ── Temporal cognition (時間認知構造) ──
        self._temporal_cognition = create_temporal_cognition()

        # ── Multi-path recall (記憶の多経路想起) ──
        self._multi_path_recall = create_multi_path_recall()

        # ── Spontaneous recall (記憶の自発的想起 - 非参照型想起) ──
        self._spontaneous_recall = create_spontaneous_recall()

        # ── Introspection cross-section (内省断面間の横断的記述) ──
        self._introspection_cross_section = create_introspection_cross_section()

        # ── Introspection longitudinal view (内省の時間的縦断参照) ──
        # 独自の永続的内部状態を保持しない薄い変換層。
        # 横断的記述のスナップショットウィンドウを唯一の入力源とする。
        self._introspection_longitudinal_view = create_introspection_longitudinal_view()

        # ── Perceptual context (知覚入力の内部文脈化) ──
        self._perceptual_context = create_perceptual_context()

        # ── Scoring fluctuation (スコアリングの構造的揺らぎ) ──
        self._fluctuation_config = ScoringFluctuationConfig()
        self._last_fluctuation_select_time: float = 0.0

        # ── Selection attribution (選択帰属) ──
        self._selection_attribution_recorder = create_selection_attribution_recorder()

        # ── Reference frequency description (参照頻度の構造的記述) ──
        self._reference_frequency_state = create_reference_frequency_state()
        self._reference_frequency_config = ReferenceFrequencyConfig()

        # ── Persistent commitment (持続的取り組み保持構造) ──
        self._persistent_commitment = create_persistent_commitment_processor()

        # ── Internal contradiction description (内部状態の矛盾並置の構造的記述) ──
        self._contradiction_processor = create_contradiction_processor()
        self._last_contradiction_result: Optional[ContradictionResult] = None

        # ── Stabilization description (安定化の構造的記述) ──
        self._stabilization_desc_state = create_stabilization_description_state()
        self._stabilization_desc_config = StabilizationDescriptionConfig()

        # ── Behavioral diversity description (行動多様性の構造的記述) ──
        self._behavioral_diversity_state = create_behavioral_diversity_state()
        self._behavioral_diversity_config = BehavioralDiversityConfig()

        # ── Input pathway balance (入力経路間の均衡記述) ──
        self._input_pathway_balance_state = create_input_pathway_balance_state()
        self._input_pathway_balance_config = InputPathwayBalanceConfig()

        # ── Interaction accumulation (相互作用の蓄積記述) ──
        self._interaction_accumulation = create_interaction_accumulation_processor()

        # ── Emotional backdrop cognition (感情基調の持続認知) ──
        self._emotional_backdrop_processor = create_emotional_backdrop_processor()
        self._last_backdrop_result: Optional[BackdropResult] = None

        # ── Drive variation description (駆動の変動記述) ──
        self._drive_variation_processor = create_drive_variation_processor()
        self._last_drive_variation_result: Optional[DriveVariationResult] = None

        # ── Expectation lifecycle description (予期の成立・消失の事後記述) ──
        self._expectation_lifecycle_processor = create_expectation_lifecycle_processor()

        # ── Responsibility temporal trace (責任の時間的推移記述) ──
        self._responsibility_temporal_trace = create_responsibility_temporal_trace()

        # ── Emotion cooccurrence description (感情間の共起記述) ──
        self._emotion_cooccurrence_processor = create_cooccurrence_processor()
        self._last_cooccurrence_result: Optional[CooccurrenceResult] = None

        # ── Other boundary accumulation (他者境界の多相蓄積) ──
        self._other_boundary_accumulation = create_other_boundary_accumulation_processor()
        self._last_boundary_accumulation: Optional[BoundaryAccumulationResult] = None

        # ── Hypothesis-observation pairing (他者モデル仮説の事後検証経路) ──
        self._hypothesis_observation_pairing = create_hypothesis_observation_pairing_processor()

        # ── Forgetting-recall balance (忘却と想起の均衡記述) ──
        self._frb_state = create_forgetting_recall_balance_state()
        self._frb_config = ForgettingRecallBalanceConfig()

        # ── Attention distribution description (注意配分の構造的記述) ──
        self._att_dist_state = create_attention_distribution_state()
        self._att_dist_config = AttDistConfig()

        # ── Goal hierarchy propagation (目的階層間の隣接状態変化記述) ──
        self._goal_hierarchy_propagation = create_goal_hierarchy_propagation_processor()

        # ── Enrichment compression (前回テキストキャッシュ、save/load対象外) ──
        self._enrichment_prev_cache: dict[str, str] = {}

        # ── Phase 30-35 cached results (for enrichment) ──
        self._last_decision_bias: Optional[DecisionBias] = None
        self._last_tone_mod: Optional[ToneModifier] = None
        self._last_sensitivity_bias: Optional[SensitivityBias] = None
        self._last_has_silence: bool = False

        logger.info(
            "PsycheOrchestrator initialized: fear=%.2f, dominant=%s, "
            "fields=49",
            fear.value, fear.dominant_fear,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def psyche(self) -> PsycheState:
        """Current core psyche state (read-only)."""
        return self._psyche

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def fear_level(self) -> float:
        return self._psyche.fear_level

    def set_recalled_memories(self, memories: Optional[list[dict]]) -> None:
        """brain.py から recall_with_mood の結果を受け取る。

        次回の Phase 21 (emotional_memory_binding) で使用される。
        """
        self._last_recalled_memories = memories

    def notify_self_output(
        self,
        response_text: str,
        policy_label: str = "",
    ) -> None:
        """brain.py から代弁コール完了後の出力テキストを受け取る。

        自己行動知覚モジュールへの通知インターフェース。
        set_recalled_memories と同型の接続構造。
        沈黙選択時（出力テキストなし）は呼び出さない。
        """
        if not response_text:
            return
        try:
            self._self_action_recorder.receive_response(
                response_text=response_text,
                policy_label=policy_label,
                tick=self._tick_count,
            )
        except Exception as e:
            logger.debug("Self-action perception notification failed: %s", e)

    def process_text_input(
        self,
        text: str,
        sender_id: str = "",
        conversation_id: str = "",
    ) -> Optional[TextHandoffResult]:
        """テキスト対話入力を処理する（brain.pyから呼出）。

        既存の画面知覚経路と同列で統合前段へ接続する。
        """
        from .text_dialogue_input import InputRouteType
        try:
            self._last_text_handoff = self._text_dialogue_processor.process(
                text=text,
                route_type=InputRouteType.TEXT,
                sender_id=sender_id,
                conversation_id=conversation_id,
                existing_percept=self._last_percept,
                tick_count=self._tick_count,
            )
            return self._last_text_handoff
        except Exception as e:
            logger.debug("Text dialogue input processing failed: %s", e)
            return None

    def check_spontaneous_activation(self) -> Optional[SpontaneousResult]:
        """外部入力なし時の自発起動チェック。brain.pyから呼出。

        起動候補情報のみを返し、判断・評価・行動決定は行わない。
        """
        try:
            result = self._spontaneous_processor.process(
                psyche=self._psyche,
                dynamics=self._dynamics,
                stm=self._loop_state.memory if self._loop_state else None,
                memories=self._last_recalled_memories,
                recent_actions=None,
                has_external_input=False,
                tick_count=self._tick_count,
            )
            self._last_activation_result = result
            return result
        except Exception as e:
            logger.debug("Spontaneous activation check failed: %s", e)
            return None

    # ── Phase 1-7: Every-tick update ──────────────────────────────

    def _run_every_tick(
        self,
        percept: Percept,
        delta_time: float,
        user_id: str,
    ) -> None:
        """毎ティック実行するフェーズ (Phase 1-7)."""

        # Preserve percept for 5-tick phase input supply
        self._last_percept = percept

        # Phase 1: react_with_stm — 感情更新 + STM残留
        resp_influence = self._responsibility_mgr.get_influence(user_id)
        new_psyche, new_loop, loop_result = react_with_stm(
            percept=percept,
            psyche_state=self._psyche,
            loop_state=self._loop_state,
            delta_time=delta_time,
            responsibility_influence=resp_influence,
        )
        self._psyche = new_psyche
        self._loop_state = new_loop

        # Phase 2: dynamics — ピーク/リバウンド判定
        emo_dict = self._psyche.emotions.as_dict()
        residue_total = loop_result.residue_influence.total_intensity
        self._dynamics = update_dynamics(
            state=self._dynamics,
            current_emotions=emo_dict,
            residue_intensity=residue_total,
        )

        # Phase 2a: emotion_amplitude — dynamics相による振幅計算
        max_emo = max(emo_dict.values()) if emo_dict else 0.0
        self._amplitude_state = update_amplitude(
            self._amplitude_state,
            intensity_factor=residue_total,
            emotion_intensity=max_emo,
        )
        self._amplitude_state = decay_amplitude(self._amplitude_state, delta_time)

        # Phase 2b: multi_emotion — 感情別独立減衰
        try:
            decayed = apply_independent_decay(
                self._psyche.emotions,
                delta_time=delta_time,
                config=self._multi_emotion_config,
            )
            self._psyche = self._psyche.model_copy(update={"emotions": decayed})
        except Exception as e:
            logger.debug("Multi-emotion decay skipped: %s", e)

        # Phase 2c: stm_emotion_coupling — STM残留→再活性化・蓄積
        if self._loop_state and self._loop_state.memory:
            try:
                coupled, self._last_coupling = apply_stm_coupling(
                    emotions=self._psyche.emotions,
                    stm=self._loop_state.memory,
                    delta_time=delta_time,
                    config=self._stm_coupling_config,
                    apply_persistence=False,  # multi_emotion handles decay
                )
                self._psyche = self._psyche.model_copy(update={"emotions": coupled})
            except Exception as e:
                logger.debug("STM-emotion coupling skipped: %s", e)

        # Phase 3: attachment — 対話相手ボンド更新
        if self._psyche.attachment is not None:
            valence = abs(percept.emotion_valence)
            event_type = "positive" if percept.emotion_valence >= 0 else "negative"
            self._psyche = self._psyche.model_copy(update={
                "attachment": attachment_manager.update_bond(
                    self._psyche.attachment, user_id, event_type, valence,
                ),
            })

        # Phase 4: responsibility — 判断記録
        if percept.intent and percept.intent != "expression":
            policy_label = percept.intent
        else:
            policy_label = percept.emotion or "neutral"
        try:
            self._responsibility_mgr.record_decision(
                user_id=user_id,
                policy={"policy_label": policy_label},
                context={"emotion": percept.emotion, "text": percept.text[:100]},
            )
        except Exception as e:
            logger.debug("Responsibility record skipped: %s", e)

        # Phase 5: self_reference — 自己参照サマリ
        try:
            self._self_ref_state = execute_self_reference(
                psyche_state=self._psyche,
                responsibility_state=self._responsibility_mgr.get_state(user_id),
                short_term_memory=self._loop_state.memory if self._loop_state else None,
                dynamics_state=self._dynamics,
                dispersion_state=self._dispersion_state,
            )
        except Exception as e:
            logger.debug("Self-reference skipped: %s", e)

        # Phase 6: repeated_tendency — 傾向観測
        scoped = (
            self._scoped_goal_sys._current_scope
            if self._scoped_goal_sys.has_active_scope
            else None
        )
        try:
            self._tendency_sys.observe_turn(scoped_goal_used=scoped)
        except Exception as e:
            logger.debug("Tendency observation skipped: %s", e)

        # Phase 7: fear recompute — 4柱リスク再計算
        self._recompute_fear()

        # Phase 7a: action_result_observation — 行動記録を構成バッファへ
        # 構成バッファへの行動記録は、ポリシー選択完了後（毎ティック処理の末尾付近）に行う。
        # 結果記述の結合は低頻度処理帯（Phase 26c）に行い、同一周期内での即時構成を禁止する。
        if self._last_selected_policy_label:
            try:
                # 入力経路の判定（Phase 7e と同じロジック）
                current_pathway_label = ""
                if percept.text:
                    current_pathway_label = "text"
                elif bool(percept.intent and percept.intent != "expression"):
                    current_pathway_label = "screen"

                record_inputs = ActionResultInputs(
                    selected_policy_label=self._last_selected_policy_label,
                    selected_policy_axis=self._last_selected_policy_axis,
                    current_tick=self._tick_count,
                    input_pathway_label=current_pathway_label,
                )
                self._action_result_observer.record_action(record_inputs)
            except Exception as e:
                logger.debug("Action-result record skipped: %s", e)

        # Phase 7b: temporal_cognition — 経過記録の蓄積（毎ティック）
        # 既存のティック数ベース処理を変更しない。出力は参照情報のみ。
        # 入力経路の判定: text/screenが明確に特定できる場合のみ記録
        current_pathway_for_tc = ""
        if percept.text:
            current_pathway_for_tc = "text"
        elif percept.intent and percept.intent != "expression":
            current_pathway_for_tc = "screen"
        try:
            self._temporal_cognition.accumulate_elapsed(
                tick=self._tick_count,
                delta_time=delta_time,
                timestamp=time.time(),
                current_pathway=current_pathway_for_tc,
            )
        except Exception as e:
            logger.debug("Temporal cognition accumulate skipped: %s", e)

        # Phase 7c: perceptual_context — 知覚サマリの蓄積（毎ティック）
        # Perceptの4要素(emotion, intent, topics, emotion_valence)を蓄積する。
        # 出力は参照情報としてのみ流れ、判断・評価を含まない。
        try:
            self._perceptual_context.accumulate_summary(
                emotion=percept.emotion or "neutral",
                intent=percept.intent or "unknown",
                topics=list(getattr(percept, 'topics', ()) or ()),
                emotion_valence=percept.emotion_valence,
                tick=self._tick_count,
            )
        except Exception as e:
            logger.debug("Perceptual context accumulate skipped: %s", e)

        # Phase 7d: situational_self_presentation — 相手別自己出力記録の蓄積（毎ティック）
        # 自己行動知覚の受領処理の直後に、本機能の蓄積処理を呼び出す。
        # 自己行動知覚の記録をREAD-ONLYで参照し、相手識別情報と対にして蓄積する。
        # パターン抽出禁止。マッピング形成禁止。ポリシー選択経路遮断。
        try:
            latest_record = self._self_action_recorder.get_latest_record()
            if latest_record is not None and user_id:
                self._situational_self_presentation.receive_and_accumulate(
                    user_id=user_id,
                    response_text=latest_record.response_text,
                    policy_label=latest_record.policy_label,
                    tick=self._tick_count,
                )
                # 構成記述の生成
                self._situational_self_presentation.generate_compositions(
                    current_tick=self._tick_count,
                )
        except Exception as e:
            logger.debug("Situational self-presentation skipped: %s", e)

        # Phase 7e: input_pathway_balance — 入力経路間均衡記述（毎ティック）
        # 3経路の使用事実が確定した後に実行する。
        # 安全弁3: 経路選択遮断。安全弁4: 判断層遮断。
        # 本機能の出力は経路選択・判断バイアス計算に接続されない。
        try:
            # 現在のサイクルで使用された経路を判定
            current_pathway = ""
            if percept.text:
                current_pathway = PATHWAY_TEXT
            has_screen = bool(percept.intent and percept.intent != "expression")

            self._input_pathway_balance_state = process_input_pathway_balance(
                self._input_pathway_balance_state,
                current_pathway=current_pathway,
                text_dialogue_state=(
                    self._text_dialogue_processor.state
                    if self._text_dialogue_processor else None
                ),
                spontaneous_state=(
                    self._spontaneous_processor.state
                    if self._spontaneous_processor else None
                ),
                has_screen_input=has_screen,
                config=self._input_pathway_balance_config,
            )
        except Exception as e:
            logger.debug("Input pathway balance skipped: %s", e)

        # Phase 7f: attention_distribution_description — 注意配分の構造的記述（毎ティック）
        # 各処理Phaseの実行有無を事後的に読み取り、断面として記述する。
        # 安全弁5: 帯域制御経路の遮断。本機能の出力は帯域配分制御に接続されない。
        # 安全弁7: 出力経路不拡張。
        try:
            # 各入力源・内部信号の量的指標を既存状態からREAD-ONLYで参照
            _has_perception = bool(percept.text and percept.intent != "expression")
            _has_text = bool(percept.text)
            _has_spontaneous = (
                self._last_activation_result is not None
                and bool(getattr(self._last_activation_result, "candidates", None))
            )
            _perception_count = 0
            if _has_perception and percept.text:
                _perception_count = max(1, len(percept.text.split()) // 5)

            self._att_dist_state = process_attention_distribution(
                self._att_dist_state,
                emotion_state=self._psyche.emotions,
                memory_state=self._last_bindings,
                motivation_state=self._last_motives,
                transient_goal_state=self._transient_goal_mgr._state if self._transient_goal_mgr else None,
                scoped_goal_state=self._scoped_goal_sys,
                responsibility_state=self._dispersion_state,
                text_dialogue_state=(
                    self._text_dialogue_processor.state
                    if self._text_dialogue_processor else None
                ),
                spontaneous_state=(
                    self._spontaneous_processor.state
                    if self._spontaneous_processor else None
                ),
                has_perception_input=_has_perception,
                has_text_input=_has_text,
                has_spontaneous_activation=_has_spontaneous,
                perception_element_count=_perception_count,
                config=self._att_dist_config,
            )
        except Exception as e:
            logger.debug("Attention distribution description skipped: %s", e)

        logger.debug(
            "Tick %d every-tick: emotion=%s, mood=%.2f, fear=%.2f, "
            "dynamics=%s (accumulated=%.2f)",
            self._tick_count,
            percept.emotion or "neutral",
            self._psyche.mood.valence,
            self._psyche.fear_level,
            get_dynamics_summary(self._dynamics),
            self._amplitude_state.current_amplitude,
        )

    # ── Phase 8-14: Every 3 ticks ────────────────────────────────

    def _run_every_3_ticks(self, user_id: str) -> None:
        """3ティック毎の自己モデル + 目標フェーズ (Phase 8-14)."""

        # Phase 8: tendency_awareness
        try:
            self._tendency_awareness = observe_tendencies(self._tendency_sys)
        except Exception as e:
            logger.debug("Tendency awareness skipped: %s", e)

        # Phase 9: self_model — 統合自己ビュー
        try:
            resp_state = self._responsibility_mgr.get_state(user_id)
            self._last_self_view = self._self_model_sys.observe(
                emotion_vector=self._psyche.emotions,
                responsibility_state=resp_state,
                tendency_system=self._tendency_sys,
                tendency_awareness=self._tendency_awareness,
                proto_goal_system=self._vector_gen,
                value_orientation=self._value_orientation,
            )
        except Exception as e:
            logger.debug("Self-model observe skipped: %s", e)

        # Phase 10: proto_goal_vector — 方向ベクトル更新
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            self._vector_gen.observe_turn(
                value_orientation=self._value_orientation,
                introspection_trace=self._last_trace,
                responsibility_pattern={
                    "caution": resp_influence.caution_bias,
                    "empathy": resp_influence.empathy_bias,
                },
                emotion_tendency=self._psyche.emotions.as_dict(),
            )
        except Exception as e:
            logger.debug("Proto-goal vector skipped: %s", e)

        # Phase 11: goal_candidates — 目標候補生成
        try:
            vectors = self._vector_gen.state.vectors
            self._candidate_gen.observe_vectors(vectors)
        except Exception as e:
            logger.debug("Goal candidates skipped: %s", e)

        # Phase 12: transient_goal — 一時目標選択
        try:
            candidates_list = self._candidate_gen.state.candidates
            self._transient_goal_mgr.observe_turn(
                available_candidates=candidates_list,
            )
        except Exception as e:
            logger.debug("Transient goal skipped: %s", e)

        # Phase 12b: persistent_commitment — 持続的取り組み保持
        # transient_goal からの昇格チェック + 保持項目の減衰・解除・資源競合
        try:
            self._run_persistent_commitment()
        except Exception as e:
            logger.debug("Persistent commitment skipped: %s", e)

        # Phase 13: scoped_goal — スコープ目標コミット
        try:
            self._scoped_goal_sys.begin_turn(
                transient_manager=self._transient_goal_mgr,
                active_goal=self._transient_goal_mgr.state.active_goal,
            )
        except Exception as e:
            logger.debug("Scoped goal skipped: %s", e)

        # Phase 14: intrinsic_motivation — 内的動機感知
        try:
            self._last_motives = sense_motives_from_chain(
                system=self._motivation_sys,
                emotion=self._psyche.emotions,
                mood=self._psyche.mood,
                tendencies=self._tendency_sys.state.tendencies if self._tendency_sys else None,
                vectors=self._vector_gen.state.vectors if self._vector_gen else None,
                candidates=self._candidate_gen.state.candidates if self._candidate_gen else None,
            )
        except Exception as e:
            logger.debug("Intrinsic motivation skipped: %s", e)

        # Phase 14b: meta_emotion_cognition — メタ感情認知と変動候補生成
        # 3ティック毎の低頻度処理帯に配置（設計書の指示通り）。
        # 感情処理パイプラインのパラメータを一切変更しない（READ-ONLY参照のみ）。
        try:
            me_inputs = self._build_meta_emotion_inputs()
            self._last_meta_emotion = self._meta_emotion_processor.tick(me_inputs)
        except Exception as e:
            logger.debug("Meta-emotion cognition skipped: %s", e)

        # Phase 14c: temporal_cognition — 多断面特徴量の記述（3ティック毎）
        # 各モジュールからの統計参照はREAD-ONLYアクセサのみ使用。
        # 出力が他モジュールの処理パラメータを変更する経路を作らない。
        try:
            # エピソード記憶のタイムスタンプ（READ-ONLY）
            ep_timestamps: list[float] = []
            try:
                if self._last_episodes is not None and self._last_episodes.has_episodes:
                    ep_timestamps = [ep.timestamp for ep in self._last_episodes.episodes]
            except Exception:
                pass

            # 感情変動回数: dynamics の phase_turn_count を参照（READ-ONLY）
            emotion_change_count = 0
            try:
                if self._dynamics is not None:
                    emotion_change_count = self._dynamics.phase_turn_count
            except Exception:
                pass

            # 自己物語断片のタイムスタンプ（READ-ONLY）
            narr_timestamps: list[float] = []
            try:
                if self._last_narrative is not None and self._last_narrative.has_fragments:
                    narr_timestamps = [f.timestamp for f in self._last_narrative.fragments]
            except Exception:
                pass

            # 帯域別キャッシュ鮮度情報の算出（READ-ONLY）
            # 各帯域の最終発火ティックを_tick_countとの差分で算出。
            # 帯域の発火条件: every_tick=毎回, every_3=% 3 == 0, every_5=% 5 == 0, every_10=% 10 == 0
            band_freshness_info: dict[str, int] = {}
            try:
                tc = self._tick_count
                # every_tick: 毎ティック発火なので経過は常に0
                band_freshness_info["every_tick"] = 0
                # every_3: 最後に % 3 == 0 だったティックからの経過
                band_freshness_info["every_3"] = tc % 3
                # every_5: 最後に % 5 == 0 だったティックからの経過
                band_freshness_info["every_5"] = tc % 5
                # every_10: 最後に % 10 == 0 だったティックからの経過
                band_freshness_info["every_10"] = tc % 10
            except Exception:
                band_freshness_info = {}

            self._temporal_cognition.describe_features(
                episodic_timestamps=ep_timestamps or None,
                emotion_change_count=emotion_change_count,
                narrative_timestamps=narr_timestamps or None,
                band_freshness=band_freshness_info or None,
            )
        except Exception as e:
            logger.debug("Temporal cognition describe skipped: %s", e)

        # Phase 14d: introspection_cross_section — 内省断面のスナップショット構成（3ティック毎）
        # orchestratorが持つ6モジュールのキャッシュ出力を束ねて渡す。
        # 横断的記述 = 並置（juxtaposition）であり、統合（integration）ではない。
        # 出力は参照情報としてのみ流れる。
        try:
            ics_module_outputs = {
                ICS_SELF_MODEL: (
                    get_self_model_summary(self._last_self_view)
                    if self._last_self_view else None
                ),
                ICS_TEMPORAL_DIFF: (
                    get_difference_summary(self._last_diff_summary)
                    if self._last_diff_summary else None
                ),
                ICS_IDENTITY_COHERENCE: (
                    get_coherence_summary(self._last_coherence)
                    if self._last_coherence else None
                ),
                ICS_SELF_NARRATIVE: (
                    get_narrative_summary(self._last_narrative)
                    if self._last_narrative else None
                ),
                ICS_INTROSPECTION_CONSUMPTION: (
                    get_consumption_summary(self._last_consumption)
                    if self._last_consumption else None
                ),
                ICS_META_EMOTION: (
                    get_meta_emotion_summary(self._meta_emotion_processor.state)
                    if self._last_meta_emotion else None
                ),
            }
            self._introspection_cross_section.process(
                module_outputs=ics_module_outputs,
                tick=self._tick_count,
            )
        except Exception as e:
            logger.debug("Introspection cross-section skipped: %s", e)

        # Phase 14e: perceptual_context — 知覚推移特徴量の記述（3ティック毎）
        # ウィンドウ内の既存データに基づいて4断面の段階値を記述する。
        # 出力は参照情報としてのみ流れる。
        try:
            self._perceptual_context.describe_features()
        except Exception as e:
            logger.debug("Perceptual context describe skipped: %s", e)

        # Phase 14f: internal_contradiction_description — 内部状態の矛盾並置の構造的記述（3ティック毎）
        # 複数の内省系モジュールの出力を READ-ONLY で参照し、数値的に反対方向を同時に
        # 示している断面対を検出し、解消せず、評価せず、そのまま対として記述する。
        # 矛盾を解消しない。矛盾に優先度を付けない。矛盾を評価しない。パターンを抽出しない。
        # 出力は参照情報としてのみ流れる。判断・行動・責任の各処理系統に接続しない。
        try:
            contradiction_inputs = self._build_contradiction_inputs()
            self._last_contradiction_result = self._contradiction_processor.process(
                contradiction_inputs
            )
        except Exception as e:
            logger.debug("Internal contradiction description skipped: %s", e)

        # Phase 14g: emotional_backdrop_cognition — 感情基調の持続認知（3ティック毎）
        # 感情状態の時系列をより広い時間窓で観測し、窓内の構成を等価に記述する。
        # 感情処理パイプラインのパラメータを一切変更しない（READ-ONLY参照のみ）。
        # 出力は参照情報としてのみ流れる。判断・行動・責任の各処理系統に接続しない。
        try:
            backdrop_inputs = self._build_backdrop_inputs()
            self._last_backdrop_result = self._emotional_backdrop_processor.tick(backdrop_inputs)
        except Exception as e:
            logger.debug("Emotional backdrop cognition skipped: %s", e)

        # Phase 14h: introspection_longitudinal_view — 内省の時間的縦断参照（3ティック毎）
        # 横断的記述のスナップショットウィンドウを唯一の入力源として、
        # 断面別の時系列並置に変換する薄い変換層。
        # 独自の永続的内部状態を保持しない。
        # 出力は参照情報としてのみ流れる。判断・行動・責任の各処理系統に接続しない。
        # introspection_cross_section（Phase 14d）の後に配置する。
        try:
            ilv_snapshots = self._introspection_cross_section.get_snapshot_window()
            self._introspection_longitudinal_view.process(ilv_snapshots)
        except Exception as e:
            logger.debug("Introspection longitudinal view skipped: %s", e)

        # Phase 14i: drive_variation_description — 駆動の変動記述（3ティック毎）
        # 駆動ベクトルの時間的推移をスライディングウィンドウで収集し、
        # 窓内の構成を等価に列挙する記述構造。
        # 駆動値・反応処理パラメータ・動機生成・ポリシー候補・感情パイプラインのパラメータを一切変更しない（READ-ONLY参照のみ）。
        # 出力は参照情報としてのみ流れる。判断・行動・責任の各処理系統に接続しない。
        try:
            dv_inputs = self._build_drive_variation_inputs()
            self._last_drive_variation_result = self._drive_variation_processor.tick(dv_inputs)
        except Exception as e:
            logger.debug("Drive variation description skipped: %s", e)

        # Phase 14j: emotion_cooccurrence_description — 感情間の共起記述（3ティック毎）
        # 毎ティックにおいて同時に閾値以上で存在した感情の組み合わせを、
        # 事実としてFIFOに等価蓄積する。
        # 感情処理パイプラインのパラメータ（減衰率、振幅、ムード、連動設定）を一切変更しない（READ-ONLY参照のみ）。
        # 出力は参照情報としてのみ流れる。判断・行動・責任・ポリシー選択に接続しない。
        # 設計書: 実行位置は「感情処理パイプライン完了後」かつ「enrichment構築前」。
        try:
            emo_values = self._psyche.emotions.as_dict()
            self._last_cooccurrence_result = self._emotion_cooccurrence_processor.tick(emo_values)
        except Exception as e:
            logger.debug("Emotion cooccurrence description skipped: %s", e)

        logger.debug(
            "Tick %d every-3: self_model=%s, goals=%d vectors, motives=%s",
            self._tick_count,
            "ok" if self._last_self_view else "none",
            len(self._vector_gen.state.vectors) if self._vector_gen else 0,
            "ok" if self._last_motives else "none",
        )

    # ── Phase 15-26: Every 5 ticks ───────────────────────────────

    def _run_every_5_ticks(self, user_id: str) -> None:
        """5ティック毎の自己連続性 + 記憶 + 内省フェーズ (Phase 15-26)."""

        # Phase 15: temporal_self_difference — 過去/現在の自己差分
        try:
            if self._last_self_view is not None:
                self._temporal_diff_sys.record_snapshot(self._last_self_view)
                self._last_diff_summary = self._temporal_diff_sys.compare_with_reference(
                    current=self._last_self_view,
                    reference_index=-1,
                )
        except Exception as e:
            logger.debug("Temporal diff skipped: %s", e)

        # Phase 15b: stabilization_description — 安定化の構造的記述
        # temporal_self_differenceの更新完了後に配置する（設計書の指示通り）。
        # enrichmentへの直接露出を行わない（安全弁3）。
        # 忘却パイプラインとの経路遮断（安全弁4）。
        # 出力先は内省系構造への参照情報のみに限定する（安全弁5）。
        try:
            # 断面1用: 6信号源のREAD-ONLY読み取り
            _sd_emo_intensity = 0.0
            try:
                _sd_emo_intensity = max(
                    self._psyche.emotions.as_dict().values()
                ) if self._psyche.emotions else 0.0
            except Exception:
                pass

            _sd_stm_count = 0
            try:
                if self._loop_state and self._loop_state.memory:
                    _sd_stm_count = len(self._loop_state.memory.entries)
            except Exception:
                pass

            _sd_transient_active = False
            try:
                _sd_transient_active = self._transient_goal_mgr.state.active_goal is not None
            except Exception:
                pass

            _sd_commitment_count = 0
            try:
                _sd_commitment_count = len([
                    it for it in self._persistent_commitment.state.items
                    if not it.released
                ])
            except Exception:
                pass

            _sd_spontaneous = False
            try:
                if self._last_activation_result is not None:
                    _sd_spontaneous = bool(self._last_activation_result.candidates)
            except Exception:
                pass

            _sd_has_external = self._last_percept is not None

            self._stabilization_desc_state = process_stabilization_description(
                self._stabilization_desc_state,
                emotion_intensity=_sd_emo_intensity,
                stm_entry_count=_sd_stm_count,
                transient_goal_active=_sd_transient_active,
                persistent_commitment_unreleased_count=_sd_commitment_count,
                spontaneous_candidate_exists=_sd_spontaneous,
                has_external_input=_sd_has_external,
                diff_summary=self._last_diff_summary,
                tick=self._tick_count,
                config=self._stabilization_desc_config,
            )
        except Exception as e:
            logger.debug("Stabilization description skipped: %s", e)

        # Phase 16: continuity_strain — 連続性負荷判定
        try:
            if self._last_diff_summary is not None:
                self._last_strain = self._strain_sys.observe_difference(
                    self._last_diff_summary,
                )
        except Exception as e:
            logger.debug("Continuity strain skipped: %s", e)

        # Phase 17: self_image_integration — 暫定自己像生成
        try:
            self._last_self_image = self._self_image_sys.generate_image(
                self_state_view=self._last_self_view,
                tendency_awareness=self._tendency_awareness,
                difference_summary=self._last_diff_summary,
                strain_state=self._last_strain,
            )
        except Exception as e:
            logger.debug("Self-image skipped: %s", e)

        # Phase 18: identity_coherence — 一貫性評価
        try:
            self._last_coherence = self._coherence_sys.generate_state(
                self_image=self._last_self_image,
                difference_summary=self._last_diff_summary,
                strain_state=self._last_strain,
                tendency_awareness=self._tendency_awareness,
                value_orientation=self._value_orientation,
            )
        except Exception as e:
            logger.debug("Identity coherence skipped: %s", e)

        # Phase 19: self_narrative — 自己ナラティブ断片追加
        try:
            ctx_for_narrative = supply_context(self._input_supply)
            self._last_narrative = observe_narrative_from_chain(
                system=self._narrative_sys,
                emotional_state=self._psyche.emotions,
                short_term_memory=self._loop_state.memory if self._loop_state else None,
                tendency_awareness=self._tendency_awareness,
                difference_summary=self._last_diff_summary,
                external_context=ctx_for_narrative,
            )
        except Exception as e:
            logger.debug("Self-narrative skipped: %s", e)

        # Phase 20: episodic_memory — エピソード記録
        try:
            ctx_for_episode = supply_context(self._input_supply)
            self._last_episodes = record_episode_from_chain(
                system=self._episodic_sys,
                emotional_state=self._psyche.emotions,
                short_term_memory=self._loop_state.memory if self._loop_state else None,
                tendency_awareness=self._tendency_awareness,
                difference_summary=self._last_diff_summary,
                coherence_state=self._last_coherence,
                narrative_state=self._last_narrative,
                external_context=ctx_for_episode,
            )
        except Exception as e:
            logger.debug("Episodic memory skipped: %s", e)

        # Phase 21: emotional_memory_binding — 感情記憶紐づけ
        try:
            self._last_bindings = bind_from_chain(
                system=self._binding_sys,
                stm=self._loop_state.memory if self._loop_state else None,
                emotion=self._psyche.emotions,
                mood=self._psyche.mood,
                memories=self._last_recalled_memories,
                episodes=self._last_episodes,
            )
        except Exception as e:
            logger.debug("Emotional binding skipped: %s", e)

        # Phase 21b: memory_system_integration — 記憶系統統合
        try:
            int_ctx = IntegrationContext(
                emotions={
                    k: getattr(self._psyche.emotions, k, 0.0)
                    for k in ["joy", "sadness", "anger", "fear",
                              "surprise", "disgust", "trust"]
                },
                mood_valence=self._psyche.mood.valence,
                percept_topics=list(
                    getattr(self._last_percept, 'topics', ()) or ()
                ) if self._last_percept else [],
                percept_text=getattr(self._last_percept, 'text', '')
                if self._last_percept else '',
                percept_intent=getattr(self._last_percept, 'intent', 'unknown')
                if self._last_percept else 'unknown',
                current_time=time.time(),
                tick_count=self._tick_count,
            )
            # [H] action_result → memory_system_integration 新系統認識
            # 行動-結果対を新たな系統として認識可能にする。
            # 二重記録の消去は行わず、同一経験の異なる視点として並立保持する。
            ar_pair_dicts: Optional[list[dict]] = None
            if self._action_result_observer is not None:
                try:
                    active = self._action_result_observer.get_active_pairs()
                    if active:
                        ar_pair_dicts = [p.to_dict() for p in active]
                except Exception:
                    pass
            self._last_integration_result = self._memory_integrator.integrate(
                episodes=self._last_episodes,
                long_term_memories=self._last_recalled_memories,
                bindings=self._last_bindings,
                context=int_ctx,
                action_result_pairs=ar_pair_dicts,
            )
        except Exception as e:
            logger.debug("Memory integration skipped: %s", e)

        # Phase 21c: memory_forgetting_fixation — 記憶の忘却と固定化
        try:
            ff_inputs = self._build_forgetting_fixation_inputs()
            self._last_forgetting_fixation = (
                self._forgetting_fixation_processor.process(ff_inputs)
            )
        except Exception as e:
            logger.debug("Memory forgetting/fixation skipped: %s", e)

        # Phase 21d: multi_path_recall — 記憶の多経路想起
        try:
            # unified_units: memory_system_integrationの結果候補
            recall_units: list = []
            if self._last_integration_result is not None:
                recall_units = list(self._last_integration_result.candidates)

            # emotion_snapshot
            emo_snap = RecallEmotionSnapshot()
            try:
                active_emos = get_active_emotions(self._psyche.emotions)
                dominant_label = ""
                dominant_intensity = 0.0
                for emo_name, emo_val in active_emos:
                    if emo_val > dominant_intensity:
                        dominant_intensity = emo_val
                        dominant_label = emo_name
                emo_snap = RecallEmotionSnapshot(
                    emotions={name: val for name, val in active_emos},
                    mood_valence=self._psyche.emotions.mood,
                    dominant_emotion=dominant_label,
                )
            except Exception:
                pass

            # context_snapshot
            ctx_snap = RecallContextSnapshot(current_time=time.time())
            try:
                if self._last_percept is not None:
                    ctx_snap = RecallContextSnapshot(
                        topics=list(getattr(self._last_percept, "topics", []) or []),
                        percept_text=getattr(self._last_percept, "text", "") or "",
                        current_time=time.time(),
                    )
            except Exception:
                pass

            # temporal_snapshot
            temp_snap = RecallTemporalSnapshot(tick_count=self._tick_count)
            try:
                if self._temporal_cognition is not None:
                    tc_data = self._temporal_cognition.get_enrichment_data()
                    temp_snap = RecallTemporalSnapshot(
                        snapshot=tc_data.get("sections", {}),
                        tick_count=self._tick_count,
                    )
            except Exception:
                pass

            # binding_store
            binding_store = self._last_bindings

            # forgetting_state
            forgetting_state = None
            if self._forgetting_fixation_processor is not None:
                forgetting_state = self._forgetting_fixation_processor.state

            self._multi_path_recall.recall_all_paths(
                unified_units=recall_units,
                binding_store=binding_store,
                forgetting_state=forgetting_state,
                emotion_snapshot=emo_snap,
                context_snapshot=ctx_snap,
                temporal_snapshot=temp_snap,
            )
        except Exception as e:
            logger.debug("Multi-path recall skipped: %s", e)

        # Phase 21e: spontaneous_recall — 記憶の自発的想起（非参照型想起）
        # 外部入力(Percept)は一切参照しない。内部状態変動のみを契機とする。
        # 出力はenrichment参照情報のみ。判断・行動・感情への逆流経路なし。
        try:
            sr_units: list = []
            if self._last_integration_result is not None:
                sr_units = list(self._last_integration_result.candidates)

            # emotion_snapshot (現在の感情断面)
            sr_emo_snap = SpontaneousRecallEmotionSnapshot()
            try:
                sr_active_emos = get_active_emotions(self._psyche.emotions)
                sr_dominant_label = ""
                sr_dominant_intensity = 0.0
                for emo_name, emo_val in sr_active_emos:
                    if emo_val > sr_dominant_intensity:
                        sr_dominant_intensity = emo_val
                        sr_dominant_label = emo_name
                sr_emo_snap = SpontaneousRecallEmotionSnapshot(
                    emotions={name: val for name, val in sr_active_emos},
                    mood_valence=self._psyche.emotions.mood,
                    dominant_emotion=sr_dominant_label,
                )
            except Exception:
                pass

            # binding_store
            sr_binding_store = self._last_bindings

            # forgetting_state
            sr_forgetting_state = None
            if self._forgetting_fixation_processor is not None:
                sr_forgetting_state = self._forgetting_fixation_processor.state

            # motive_store (内的動機)
            sr_motive_store = self._last_motives

            # strain_state (連続性の揺らぎ)
            sr_strain_state = self._last_strain

            # direction_vectors (方向ベクトル)
            sr_direction_vectors = None
            try:
                if self._vector_gen is not None:
                    sr_direction_vectors = self._vector_gen.state
            except Exception:
                pass

            # temporal_snapshot (時間認知)
            sr_temporal_snapshot = None
            try:
                if self._temporal_cognition is not None:
                    sr_temporal_snapshot = self._temporal_cognition.get_enrichment_data()
            except Exception:
                pass

            self._spontaneous_recall.process(
                unified_units=sr_units,
                binding_store=sr_binding_store,
                forgetting_state=sr_forgetting_state,
                emotion_snapshot=sr_emo_snap,
                prev_emotion_snapshot=None,  # processor uses stored prev state
                motive_store=sr_motive_store,
                strain_state=sr_strain_state,
                direction_vectors=sr_direction_vectors,
                temporal_snapshot=sr_temporal_snapshot,
            )
        except Exception as e:
            logger.debug("Spontaneous recall skipped: %s", e)

        # Phase 21f: forgetting_recall_balance — 忘却と想起の均衡記述
        # 忘却側1構造・想起側2構造の状態を読み取り専用で参照し、
        # 3断面（忘却断面・外部トリガー型想起断面・自発的想起断面）を等価に並置記述する。
        # 処理タイミング: 忘却処理(21c)と想起処理(21d, 21e)の双方が完了した後。
        # 出力は情報としてのみ流れ、忘却処理・想起処理・判断系に直接作用しない。
        try:
            frb_forgetting_state = None
            if self._forgetting_fixation_processor is not None:
                frb_forgetting_state = self._forgetting_fixation_processor.state

            frb_forgetting_result = self._last_forgetting_fixation

            frb_multi_path_recall_state = None
            if self._multi_path_recall is not None:
                frb_multi_path_recall_state = self._multi_path_recall.state

            frb_spontaneous_recall_state = None
            if self._spontaneous_recall is not None:
                frb_spontaneous_recall_state = self._spontaneous_recall.state

            self._frb_state = process_forgetting_recall_balance(
                self._frb_state,
                forgetting_state=frb_forgetting_state,
                forgetting_result=frb_forgetting_result,
                multi_path_recall_state=frb_multi_path_recall_state,
                spontaneous_recall_state=frb_spontaneous_recall_state,
                config=self._frb_config,
            )
        except Exception as e:
            logger.debug("Forgetting-recall balance skipped: %s", e)

        # Phase 22: introspection_trace — 内省ログ生成
        # self_action_perception → introspection_trace 間接経路:
        # 自己行動知覚の直近記録からテキスト存在事実・長さ・ポリシーラベル・ティックのみを
        # context パラメータとして渡す。テキスト本文は含めない（自己強化ループ遮断）。
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            # --- self_action context 構築 ---
            sa_context: dict[str, Any] = {}
            if self._self_action_recorder is not None:
                try:
                    sa_record = self._self_action_recorder.get_latest_record()
                    if sa_record is not None:
                        text_len = len(sa_record.response_text) if sa_record.response_text else 0
                        # 安全弁: テキスト長上限（異常値切り詰め）
                        _SA_TEXT_LENGTH_CAP = 100_000
                        sa_context = {
                            "self_action_has_output": True,
                            "self_action_text_length": min(text_len, _SA_TEXT_LENGTH_CAP),
                            "self_action_policy_label": sa_record.policy_label or "",
                            "self_action_tick": sa_record.tick,
                        }
                    else:
                        sa_context = {"self_action_has_output": False}
                except Exception:
                    # 安全弁: 参照失敗時は空文脈フォールバック
                    sa_context = {}
            self._last_trace = self._introspection_sys.generate_trace(
                emotion_state=self._psyche.emotions,
                responsibility_state=resp_influence,
                value_orientation=self._value_orientation,
                fear_index=self._psyche.fear_index,
                context=sa_context if sa_context else None,
            )
        except Exception as e:
            logger.debug("Introspection trace skipped: %s", e)

        # Phase 23: introspection_consumption — 内省消費・再構成
        try:
            trace_summary = get_trace_summary(self._last_trace) if self._last_trace else None
            self._last_consumption = consume_introspection_from_chain(
                system=self._consumption_sys,
                introspection_summary=trace_summary,
                narrative_state=self._last_narrative,
                coherence_state=self._last_coherence,
                tendency_awareness=self._tendency_awareness,
                episodic_store=self._last_episodes,
            )
        except Exception as e:
            logger.debug("Introspection consumption skipped: %s", e)

        # Phase 24: expectation_formation — 期待形成
        try:
            tendency_bias = (
                self._tendency_sys.state.tendencies
                if self._tendency_sys else None
            )
            self._last_expectations = form_expectations_from_chain(
                system=self._expectation_sys,
                tendency_bias=tendency_bias,
                difference_summary=self._last_diff_summary,
                narrative_state=self._last_narrative,
            )
        except Exception as e:
            logger.debug("Expectation formation skipped: %s", e)

        # Phase 24b: reference_frequency_description — 参照頻度の構造的記述
        # 各記憶系構造の参照回数更新が完了した後に実行する。
        # enrichmentへの直接露出を行わない（設計制約: 安全弁5）。
        # 忘却パイプラインとの経路遮断（設計制約）。
        # 想起経路選択への影響遮断（設計制約）。
        # 出力先は内省系構造への参照情報のみに限定する。
        try:
            self._reference_frequency_state = process_reference_frequency(
                self._reference_frequency_state,
                episodic_store=self._last_episodes,
                binding_store=self._last_bindings,
                consumption_store=self._last_consumption,
                expectation_store=self._last_expectations,
                motive_store=self._last_motives,
                narrative_state=self._last_narrative,
                other_model_store=self._last_other_model,
                self_reference_state=self._self_ref_state,
                action_result_state=(
                    self._action_result_observer.state
                    if self._action_result_observer else None
                ),
                dialogue_learning_state=(
                    self._dialogue_learning_processor.state
                    if self._dialogue_learning_processor else None
                ),
                forgetting_state=(
                    self._forgetting_fixation_processor.state
                    if self._forgetting_fixation_processor else None
                ),
                multi_path_recall_state=(
                    self._multi_path_recall.state
                    if self._multi_path_recall else None
                ),
                spontaneous_recall_state=(
                    self._spontaneous_recall.state
                    if self._spontaneous_recall else None
                ),
                config=self._reference_frequency_config,
            )
        except Exception as e:
            logger.debug("Reference frequency description skipped: %s", e)

        # Phase 25a: other_model_real_feed — 実対話由来の観測断片抽出・正規化
        try:
            self._last_feed_result = self._real_feed_processor.process(
                percept=self._last_percept,
                stm=self._loop_state.memory if self._loop_state else None,
                psyche=self._psyche,
                dynamics=self._dynamics,
                recalled_memories=self._last_recalled_memories,
                integration_result=self._last_integration_result,
                tick_count=self._tick_count,
            )
            # [D] action_result → other_model_real_feed 時系列隣接記録供給
            # 「この行動の後にこの他者反応が観測された」という時系列的隣接の記録を供給。
            # 因果帰属は行わない。他者の反応は他者自身の内部状態にも依存する。
            if self._action_result_observer is not None:
                try:
                    from .other_model_real_feed import (
                        ObservationFragment as RealFeedFragment,
                        ObservationFragmentType as RealFeedFragmentType,
                    )
                    active_pairs = self._action_result_observer.get_active_pairs()
                    external_frags = []
                    for pair in active_pairs[:3]:  # 直近3対のみ
                        # 他者観測断面があれば断片として供給
                        if pair.result and pair.result.sections:
                            other_secs = [
                                s for s in pair.result.sections
                                if s.section == "other_observation"
                            ]
                            for sec in other_secs[:1]:
                                frag = RealFeedFragment(
                                    type=RealFeedFragmentType.RECENT_HISTORY,
                                    source_description=(
                                        f"action_result: {pair.action.policy_label} "
                                        f"-> {sec.description[:40]}"
                                    ),
                                    value=max(0.0, min(1.0, sec.value)),
                                    text_hint=f"temporal_adjacency:{pair.pair_id[:8]}",
                                )
                                external_frags.append(frag)
                    if external_frags:
                        self._real_feed_processor.inject_external_fragments(external_frags)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Real feed processing skipped: %s", e)

        # Phase 25b: text_dialogue_input — テキスト対話入力経路処理
        # (orchestrator外部からtext入力が供給された場合のみ実行)
        # Processor自体は常時利用可能。外部からprocess()を直接呼ぶ形式。

        # Phase 25c: other_model_dialogue_learning — 他者観測の長期蓄積と仮説補助
        # Phase 25a(real_feed)の結果とaction_result_observerの他者断面を入力とし、
        # 8段パイプラインを実行する。蓄積→仮説更新→ポリシー選択が同一ティック内で
        # 連鎖しないよう、Phase 25(other_agent_model)の前に配置する。
        try:
            dl_inputs = self._build_dialogue_learning_inputs(user_id)
            self._last_dialogue_learning = self._dialogue_learning_processor.tick(dl_inputs)
        except Exception as e:
            logger.debug("Dialogue learning skipped: %s", e)

        # Phase 25d: interaction_accumulation — 相互作用の蓄積記述
        # 自己行動知覚の記録と他者モデルリアルフィードの観測記録を時系列的隣接関係として
        # 対構成し蓄積する。自己行動知覚の記録更新と他者モデルリアルフィードの観測更新が
        # 完了した後に実行する。因果帰属を行わない。パターン抽出を行わない。
        # 出力はenrichmentへの等価列挙と内省系構造へのREAD-ONLY参照のみ。
        try:
            ia_self_records = []
            if self._self_action_recorder is not None:
                ia_self_records = self._self_action_recorder.get_reference_history()
            ia_other_units = []
            if self._real_feed_processor is not None:
                ia_other_units = list(self._real_feed_processor.state.units)
            self._interaction_accumulation.process(
                self_records=ia_self_records,
                other_units=ia_other_units,
                current_tick=self._tick_count,
            )
        except Exception as e:
            logger.debug("Interaction accumulation skipped: %s", e)

        # Phase 25e: other_boundary_accumulation — 他者境界の多相蓄積
        # other_agent_modelが前ティックで生成したSelfOtherBoundaryのリストを
        # READ-ONLY参照し、相手別に蓄積する。境界の乖離度を制御・調整・最適化しない。
        # 蓄積された推移からパターンを抽出しない。判断・行動選択に接続しない。
        try:
            oba_boundaries = []
            if self._last_other_model is not None:
                oba_boundaries = list(self._last_other_model.boundaries)
            # 最新の境界情報（あれば1件）を蓄積対象とする
            oba_boundary = oba_boundaries[-1] if oba_boundaries else None
            self._last_boundary_accumulation = self._other_boundary_accumulation.tick(
                boundary=oba_boundary,
                user_id=user_id,
                current_tick=self._tick_count,
            )
        except Exception as e:
            logger.debug("Other boundary accumulation skipped: %s", e)

        # Phase 25f: hypothesis_observation_pairing — 他者モデル仮説の事後検証経路
        # other_agent_modelから仮説群をREAD-ONLYで取得し、
        # other_model_real_feedの最新結果から観測断片をREAD-ONLYで取得し、
        # 時間的隣接に基づいて仮説-観測の隣接対を構成・蓄積する。
        # 仮説の正誤判定を行わない。整合性を算出しない。パターン抽出を行わない。
        # 出力はenrichmentへの等価列挙とREAD-ONLY参照のみ。
        try:
            hop_hypothesis_source = self._last_other_model if self._last_other_model is not None else None
            hop_observation_source = self._real_feed_processor if self._real_feed_processor is not None else None
            self._hypothesis_observation_pairing.process(
                hypothesis_source=hop_hypothesis_source,
                observation_source=hop_observation_source,
                user_id_source=user_id,
                current_cycle=self._tick_count,
            )
        except Exception as e:
            logger.debug("Hypothesis-observation pairing skipped: %s", e)

        # Phase 25: other_agent_model — 他者モデル仮説更新 (入力供給経由)
        try:
            # 入力供給更新: percept / STM / dynamics / psyche から計算
            self._input_supply = update_input_supply(
                state=self._input_supply,
                percept=self._last_percept,
                stm=self._loop_state.memory if self._loop_state else None,
                dynamics=self._dynamics,
                psyche=self._psyche,
            )
            self._input_supply = decay_buffer(self._input_supply, time.time())

            # 供給: context snapshot + reaction log
            ctx = supply_context(self._input_supply)

            # リアルフィードで context を差分調整
            if self._last_feed_result is not None:
                ctx = enhance_context_with_feed(ctx, self._last_feed_result)

            # [G] dialogue_learning → input_supply 長期蓄積補足
            # 蓄積記述の概要を、既存の文脈生成処理を上書きするのではなく、
            # 長期蓄積由来の補足情報として追加する。
            if self._dialogue_learning_processor is not None:
                try:
                    dl_data = self._dialogue_learning_processor.get_enrichment_data()
                    dl_active = dl_data.get("active_count", 0)
                    dl_rep = dl_data.get("repetition_active", 0)
                    if dl_active > 0:
                        # 長期蓄積の存在を密度・連続性に微弱に反映
                        density_supplement = min(0.1, dl_active * 0.01)
                        continuity_supplement = min(0.05, dl_rep * 0.01)
                        ctx.density = min(1.0, ctx.density + density_supplement)
                        ctx.continuity = min(1.0, ctx.continuity + continuity_supplement)
                except Exception:
                    pass

            # [F] dialogue_learning → other_agent_model 仮説重み分布
            # 蓄積記述から、特定の対話文脈において仮説候補間の相対的重みが
            # どのように分布するかを記述した情報を提供する。
            # 分布は常に複数の仮説候補を含み、単一候補に収束しない。
            # contextに仮説材料情報を付加（duck typing属性として追加）
            if self._dialogue_learning_processor is not None and self._last_dialogue_learning is not None:
                try:
                    dl_data = self._dialogue_learning_processor.get_enrichment_data()
                    materials = dl_data.get("material_count", 0)
                    supply_str = dl_data.get("supply_strength", 0.0)
                    if materials > 0:
                        # 仮説材料分布を参照情報として付加
                        # weight: 蓄積由来情報の相対的比重を微弱に上方修正
                        weight_boost = min(0.1, materials * 0.005) * supply_str
                        ctx.weight = min(1.0, ctx.weight + weight_boost)
                except Exception:
                    pass

            rlog = supply_reaction_log(self._input_supply)

            self._last_other_model = observe_other_from_chain(
                system=self._other_model_sys,
                external_context=ctx,
                reaction_log=rlog,
                self_state=self._last_self_view,
            )
        except Exception as e:
            logger.debug("Other agent model skipped: %s", e)

        # Phase 26: value_orientation — 価値指向更新（遅い変化）
        try:
            emo_signal = generate_emotion_signal(self._psyche.emotions)
            resp_influence_for_vo = self._responsibility_mgr.get_influence(user_id)
            resp_signal = generate_responsibility_signal(
                total_weight=resp_influence_for_vo.caution_bias,
            )
            # [A] action_result → value_orientation シグナル供給
            # 蓄積された行動-結果対の情報を微弱なシグナルとして供給。
            # シグナル強度は既存シグナル（emo_signal, resp_signal）を超えない上限を持つ。
            ar_signal: Optional[dict[str, float]] = None
            if self._action_result_observer is not None and self._last_action_result is not None:
                try:
                    ar_data = self._action_result_observer.get_enrichment_data()
                    ar_pattern = ar_data.get("pattern_distribution", {})
                    supply_strength = ar_data.get("signal_supply_strength", 1.0)
                    if ar_pattern and supply_strength > 0.0:
                        # パターン分布から微弱なシグナルを生成
                        # 既存シグナルの強度上限: emo/respシグナルの最大値を超えない
                        existing_max = 0.0
                        for sig in [emo_signal, resp_signal]:
                            if sig:
                                for v in sig.values():
                                    existing_max = max(existing_max, abs(v))
                        cap = max(0.01, existing_max * 0.5)  # 既存の50%以下に制限
                        ar_signal = {}
                        total = sum(ar_pattern.values()) or 1
                        # パターン多様性をdim_a-e方向のシグナルに変換
                        diversity = len(ar_pattern) / max(1, total)
                        base_strength = diversity * supply_strength * cap
                        for i, dim in enumerate(["a", "b", "c", "d", "e"]):
                            # 微弱な双方向的情報。最適化方向を指示しない
                            ar_signal[dim] = base_strength * (0.5 - (i * 0.1))
                except Exception:
                    pass

            self._value_orientation = update_orientation(
                orientation=self._value_orientation,
                emotion_signal=emo_signal if emo_signal else None,
                responsibility_signal=resp_signal if resp_signal else None,
            )
            # action_result由来シグナルを分離供給（3番目のシグナルとして追加）
            # update_orientationは3信号を平均するため、別途供給して影響を微弱に保つ
            if ar_signal:
                self._value_orientation = update_orientation(
                    orientation=self._value_orientation,
                    decision_signal=ar_signal,
                )
        except Exception as e:
            logger.debug("Value orientation skipped: %s", e)

        # Phase 26b: value_orientation_validation — 価値方向性の実運用検証
        try:
            vo_inputs = self._build_vo_validation_inputs(user_id)
            self._last_vo_validation = self._vo_validator.process(vo_inputs)
        except Exception as e:
            logger.debug("Value orientation validation skipped: %s", e)

        # Phase 26c: action_result_observation — 行動-結果対の処理
        # 結果記述の結合は低頻度処理帯で行い、同一周期内での即時構成を禁止する。
        try:
            ar_inputs = self._build_action_result_inputs(user_id)
            self._last_action_result = self._action_result_observer.process(ar_inputs)
        except Exception as e:
            logger.debug("Action-result observation skipped: %s", e)

        # Phase 26c2: behavioral_diversity_description — 行動多様性の構造的記述
        # 行動結果観測構造（Phase 26c）と選択帰属構造の更新完了後に配置する。
        # enrichmentへの直接露出を行わない（安全弁3）。
        # 忘却パイプラインとの経路遮断（安全弁4）。
        # 想起経路との経路遮断（安全弁5）。
        # 出力先は内省系構造への参照情報のみに限定する（安全弁7）。
        # 既存モジュールの安全弁を一切緩和しない（安全弁8）。
        try:
            self._behavioral_diversity_state = process_behavioral_diversity(
                self._behavioral_diversity_state,
                action_result_state=self._action_result_observer,
                selection_attribution_state=self._selection_attribution_recorder,
                tick=self._tick_count,
                config=self._behavioral_diversity_config,
            )
        except Exception as e:
            logger.debug("Behavioral diversity description skipped: %s", e)

        # Phase 26d: [C] action_result → expectation_formation 差分照合
        # 蓄積された行動-結果対と予期情報との差分を記録する。
        # 照合結果は「差分の認知」として記録にとどめる。
        # 予期が「当たった/外れた」という評価はしない。
        # 照合結果を判断系に直接接続しない。
        try:
            if (self._last_expectations is not None
                    and self._action_result_observer is not None):
                active_pairs = self._action_result_observer.get_active_pairs()
                expectations = getattr(self._last_expectations, 'expectations', ())
                if active_pairs and expectations:
                    # 差分記録: 予期の内容と行動-結果対のパターンの多断面並存記録
                    diff_records = []
                    for exp in expectations:
                        exp_desc = getattr(exp, 'description', '')
                        exp_source = getattr(exp, 'source_type', None)
                        source_val = exp_source.value if exp_source else 'unknown'
                        exp_basis_raw = getattr(exp, 'basis', None)
                        basis_val = exp_basis_raw.value if exp_basis_raw else 'unknown'
                        exp_strength_raw = getattr(exp, 'strength', None)
                        if isinstance(exp_strength_raw, (int, float)):
                            strength_val = float(exp_strength_raw)
                        elif hasattr(exp_strength_raw, 'value'):
                            strength_val = exp_strength_raw.value
                        else:
                            strength_val = 'unknown'
                        for pair in active_pairs[:5]:  # 直近5対のみ照合
                            pair_pattern = pair.pattern_key or ''
                            # 結果断面キー一覧
                            result_section_keys = []
                            try:
                                for sec in pair.result.sections:
                                    sec_key = getattr(sec, 'section', '')
                                    if sec_key:
                                        result_section_keys.append(sec_key)
                            except Exception:
                                pass
                            diff_records.append({
                                "expectation": {
                                    "description": exp_desc[:60],
                                    "source_type": source_val,
                                    "basis": basis_val,
                                    "strength": strength_val,
                                },
                                "action": {
                                    "policy_label": pair.action.policy_label,
                                    "pattern_key": pair_pattern,
                                },
                                "result_sections": result_section_keys,
                                "context": {
                                    "tick": self._tick_count,
                                },
                            })
                    # 差分記録を保持（上限付き）
                    if not hasattr(self, '_expectation_action_diff_log'):
                        self._expectation_action_diff_log: list[dict] = []
                    self._expectation_action_diff_log.extend(diff_records)
                    self._expectation_action_diff_log = self._expectation_action_diff_log[-50:]
        except Exception as e:
            logger.debug("Expectation-action diff skipped: %s", e)

        # Phase 26e: intent_action_gap — 意図-行動間の乖離認知
        # 自己行動知覚(notify_self_output)の後の帯で、自己行動記録の最新記録と
        # ポリシー選択情報を入力として処理を呼び出す。
        # 乖離記録→ポリシー選択(Phase 30-35)への接続禁止。
        # 乖離記録→行動-結果観測への接続禁止。
        # 乖離記録→予期形成への接続禁止。
        try:
            latest_action = self._self_action_recorder.get_latest_record()
            if latest_action is not None:
                self._intent_action_gap_recorder.process_action_record(
                    response_text=latest_action.response_text,
                    policy_label=latest_action.policy_label,
                    tick=latest_action.tick,
                    context_info=self._last_selected_policy_axis or "",
                )
        except Exception as e:
            logger.debug("Intent-action gap skipped: %s", e)

        # Phase 26f: expectation_lifecycle_description — 予期の成立・消失の事後記述
        # 予期形成が保持する予期集合のスナップショットを前回と比較し、
        # 生成・消失・修正・強度変化・鮮度変化の遷移を検出して記録する。
        # 予期形成への書き込み経路を持たない（READ-ONLY）。
        # 遷移記録→ポリシー選択・バイアス適用・スコアリングへの接続禁止。
        # 遷移記録→予期形成パラメータへの書き込み禁止。
        # 因果帰属禁止、的中率等の統計量算出禁止。
        try:
            self._expectation_lifecycle_processor.process(self._last_expectations)
        except Exception as e:
            logger.debug("Expectation lifecycle description skipped: %s", e)

        # Phase 26g: responsibility_temporal_trace — 責任の時間的推移記述
        # 責任管理構造と責任分散構造からREAD-ONLYで値を読み取り、
        # スナップショットとして時系列蓄積し、変動度合いを段階値で記述する。
        # 責任管理構造・責任分散構造への書き込み経路を一切持たない（READ-ONLY）。
        # 段階値を判断バイアス計算・方針選択・安定弁に接続しない。
        # パターン抽出禁止、統計量算出禁止、方向性記述排除。
        try:
            resp_summary = self._responsibility_mgr.get_summary(user_id)
            disp_active_weight = 0.0
            disp_active_count = 0
            disp_transformation_count = 0
            if self._dispersion_state is not None:
                disp_active_weight = get_dispersion_active_weight(self._dispersion_state)
                disp_active_count = len(get_dispersion_active_units(self._dispersion_state))
                disp_transformation_count = self._dispersion_state.transformation_count

            self._responsibility_temporal_trace.record_snapshot(
                tick=self._tick_count,
                total_weight=resp_summary["total_weight"],
                pending_decisions=resp_summary["pending_decisions"],
                accumulated_harm=resp_summary["accumulated_harm"],
                accumulated_confidence=resp_summary["accumulated_confidence"],
                dispersion_active_weight=disp_active_weight,
                dispersion_active_count=disp_active_count,
                dispersion_transformation_count=disp_transformation_count,
            )
            self._responsibility_temporal_trace.describe_variation()
        except Exception as e:
            logger.debug("Responsibility temporal trace skipped: %s", e)

        # Phase 26h: goal_hierarchy_propagation — 目的階層間の隣接状態変化記述
        # 3層（transient_goal, persistent_commitment, value_orientation）の状態が
        # すべて確定した後のタイミングでスナップショット取得と変化検出を行う。
        # 3層からの読み取りはREAD-ONLYであり、各層の公開済みアクセサのみを使用する。
        # enrichmentへの直接露出を遮断する（内省系構造からのREAD-ONLY参照のみ）。
        # 3層への書き戻し経路なし。ポリシー候補生成・バイアス適用・スコアリングに入力されない。
        try:
            # 第1層: transient_goal のスナップショット情報
            tg_data: dict = {}
            try:
                active_goal = self._transient_goal_mgr.state.active_goal
                if active_goal is not None:
                    tg_data = {
                        "has_active": True,
                        "category": active_goal.candidate_category.value,
                        "direction_signature": dict(active_goal.direction_alignment),
                        "strength": active_goal.selection_strength,
                    }
                else:
                    tg_data = {"has_active": False}
            except Exception:
                pass

            # 第2層: persistent_commitment のスナップショット情報
            pc_data: dict = {}
            try:
                pc = self._persistent_commitment
                active_items = [it for it in pc.state.items if not it.released]
                items_list = [
                    {
                        "item_id": it.item_id,
                        "category": it.category,
                        "strength": it.strength,
                    }
                    for it in active_items
                ]
                recent_types = [
                    r.record_type for r in pc.state.cognition_records[-3:]
                ] if pc.state.cognition_records else []
                pc_data = {
                    "items": items_list,
                    "recent_cognition_types": recent_types,
                }
            except Exception:
                pass

            # 第3層: value_orientation のスナップショット情報
            vo_data: dict = {}
            try:
                vo = self._value_orientation
                vo_data = {
                    "dimensions": vo.get_all_dimensions(),
                    "confidences": vo.get_all_confidences(),
                    "update_count": vo.update_count,
                }
            except Exception:
                pass

            self._goal_hierarchy_propagation.process(
                transient_goal_data=tg_data if tg_data else None,
                persistent_commitment_data=pc_data if pc_data else None,
                value_orientation_data=vo_data if vo_data else None,
            )
        except Exception as e:
            logger.debug("Goal hierarchy propagation skipped: %s", e)

        logger.debug(
            "Tick %d every-5: diff=%s, strain=%s, coherence=%s, "
            "episodes=%s, expectations=%s",
            self._tick_count,
            "ok" if self._last_diff_summary else "none",
            "ok" if self._last_strain else "none",
            "ok" if self._last_coherence else "none",
            "ok" if self._last_episodes else "none",
            "ok" if self._last_expectations else "none",
        )

    # ── Phase 27-29: Every 10 ticks ──────────────────────────────

    def _run_every_10_ticks(self, user_id: str) -> None:
        """10ティック毎の安定性 + ロギング + スナップショット (Phase 27-29)."""

        # Phase 27: stability_valve — 極端偏り検出
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            self._stability_valve.observe_extremity(
                fear_level=self._psyche.fear_level,
                responsibility_weight=resp_influence.caution_bias,
                value_orientation=self._value_orientation,
                emotion_state=self._psyche.emotions,
            )
        except Exception as e:
            logger.debug("Stability valve skipped: %s", e)

        # Phase 28: long_term_dynamics — 長期行動ログ
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            self._dynamics_observer.record_turn(
                emotion_state=self._psyche.emotions,
                value_orientation=self._value_orientation,
                responsibility_weight=resp_influence.caution_bias,
                responsibility_caution=resp_influence.caution_bias,
            )
        except Exception as e:
            logger.debug("Long-term dynamics skipped: %s", e)

        # Phase 29: snapshot — 永続化は save() で明示的に行う
        logger.debug(
            "Tick %d every-10: stability_valve observed, dynamics logged",
            self._tick_count,
        )

    # ── Main tick entry point ─────────────────────────────────────

    def post_response_update(
        self,
        percept: Percept,
        delta_time: float,
        user_id: str = "viewer",
    ) -> None:
        """応答後の状態更新。brain.py の _update_psyche を置き換える。

        Args:
            percept: 知覚入力（Gemini応答からのエモーションタグ等）
            delta_time: 前回更新からの経過秒
            user_id: 対話相手ID
        """
        self._tick_count += 1

        # Notify spontaneous processor of external input
        self._spontaneous_processor.notify_external_input()

        # Notify temporal cognition of external input arrival
        try:
            self._temporal_cognition.notify_external_input(timestamp=time.time())
        except Exception as e:
            logger.debug("Temporal cognition external input notify skipped: %s", e)

        # Phase 1-7: every tick
        self._run_every_tick(percept, delta_time, user_id)

        # Phase 8-14: every 3 ticks
        if self._tick_count % 3 == 0:
            self._run_every_3_ticks(user_id)

        # Phase 15-26: every 5 ticks
        if self._tick_count % 5 == 0:
            self._run_every_5_ticks(user_id)

        # Phase 27-29: every 10 ticks
        if self._tick_count % 10 == 0:
            self._run_every_10_ticks(user_id)

    # ── Expectation diff accessor ────────────────────────────────

    def get_expectation_diff_summary(self) -> dict:
        """予期差分記録の読み取り専用サマリを返す。

        情報の変換・評価・選別を行わない。
        記録されている内容をそのまま構造化して返す。
        """
        log = self._expectation_action_diff_log
        total = len(log)
        recent = log[-5:] if total > 0 else []
        # 断面別記述キー一覧（どの断面で差分が記述されているか）
        section_keys: set[str] = set()
        for rec in log:
            for key in rec.get("result_sections", []):
                section_keys.add(key)
        return {
            "total_count": total,
            "recent_records": recent,
            "section_keys": sorted(section_keys),
        }

    # ── Prompt enrichment ─────────────────────────────────────────

    def _collect_enrichment_items(
        self, user_id: str = "viewer",
    ) -> list[dict]:
        """enrichment項目を(ラベル, テキスト)ペアとしてセクション別に収集する。

        各モジュールの出力メソッドは一切変更せず、テキストを読み取り専用で取得する。
        戻り値はbuild_compressed_enrichment()の入力形式に合わせた辞書リスト。

        Returns:
            list of {"header": str, "items": list[tuple[str, str]]}
        """
        p = self._psyche
        sections_data: list[dict] = []

        # ── 【心理状態（内面）】 ──
        psyche_items: list[tuple[str, str]] = []
        psyche_items.append(("感情", f"感情: {p.emotion_summary()}"))
        psyche_items.append((
            "ムード",
            f"ムード: valence={p.mood.valence:.2f}, arousal={p.mood.arousal:.2f}",
        ))
        psyche_items.append((
            "ドライブ",
            f"ドライブ: social={p.drives.social:.2f}, "
            f"curiosity={p.drives.curiosity:.2f}, "
            f"expression={p.drives.expression:.2f}",
        ))
        psyche_items.append(("恐怖", p.fear_summary()))
        if p.dominant_emotion_value > 0.3:
            psyche_items.append((
                "支配的感情",
                f"支配的感情: {p.dominant_emotion} "
                f"({p.dominant_emotion_value:.2f})",
            ))
        # #1 responsibility
        try:
            resp_summary = self._responsibility_mgr.get_summary(user_id)
            psyche_items.append((
                "責任",
                f"責任: weight={resp_summary['total_weight']:.2f}, "
                f"harm={resp_summary['accumulated_harm']:.2f}, "
                f"caution={resp_summary['influence']['caution_bias']:.2f}, "
                f"empathy={resp_summary['influence']['empathy_bias']:.2f}",
            ))
        except Exception:
            pass
        # #2 responsibility_dispersion
        if self._dispersion_state is not None:
            disp_summary = get_dispersion_summary(self._dispersion_state)
            if disp_summary:
                psyche_items.append(("責任拡散", f"責任拡散: {disp_summary}"))
        # #42 responsibility_temporal_trace
        if self._responsibility_temporal_trace is not None:
            try:
                rtt_data = self._responsibility_temporal_trace.get_enrichment_data()
                rtt_text = rtt_data.get("summary_text", "")
                if rtt_text and "待機中" not in rtt_text:
                    psyche_items.append(("責任推移", f"責任推移: {rtt_text}"))
            except Exception:
                pass
        # #6 stability_valve
        try:
            valve_bias = self._stability_valve.generate_bias()
            if valve_bias.is_active:
                psyche_items.append((
                    "安定弁",
                    f"安定弁: active, level={valve_bias.activation_level:.2f}",
                ))
        except Exception:
            pass
        # #10 stm_emotion_coupling
        if self._last_coupling is not None:
            coupling_str = get_coupling_summary(self._last_coupling)
            if coupling_str:
                psyche_items.append(("感情連動", f"感情連動: {coupling_str}"))
        if psyche_items:
            sections_data.append({
                "header": "【心理状態（内面）】",
                "items": psyche_items,
            })

        # ── 【自己認識】 ──
        self_items: list[tuple[str, str]] = []
        if self._last_self_image is not None:
            summary = get_self_image_summary(self._last_self_image)
            if summary:
                self_items.append(("自己像", f"自己像: {summary}"))
        if self._last_coherence is not None:
            coh_summary = get_coherence_summary(self._last_coherence)
            if coh_summary:
                self_items.append(("一貫性", f"一貫性: {coh_summary}"))
        if self._tendency_awareness is not None:
            awareness_summary = get_awareness_summary(self._tendency_awareness)
            if awareness_summary:
                self_items.append(("傾向", f"傾向: {awareness_summary}"))
        if self._last_diff_summary is not None:
            diff_summary = get_difference_summary(self._last_diff_summary)
            if diff_summary:
                self_items.append(("変化", f"変化: {diff_summary}"))
        if self._last_strain is not None:
            strain_summary = get_strain_summary(self._last_strain)
            if strain_summary:
                self_items.append(("連続性緊張", f"連続性緊張: {strain_summary}"))
        if self._last_narrative is not None:
            narr_summary = get_narrative_summary(self._last_narrative)
            if narr_summary:
                self_items.append(("自己語り", f"自己語り: {narr_summary}"))
        # #8 long_term_dynamics
        if self._dynamics_observer is not None:
            ltd_summary = get_observer_summary(self._dynamics_observer)
            if ltd_summary:
                self_items.append(("長期傾向", f"長期傾向: {ltd_summary}"))
        # #27 temporal_cognition
        if self._temporal_cognition is not None:
            try:
                tc_data = self._temporal_cognition.get_enrichment_data()
                tc_text = tc_data.get("summary_text", "")
                if tc_text and "待機中" not in tc_text:
                    self_items.append(("時間認知", f"時間認知: {tc_text}"))
            except Exception:
                pass
        # #30 perceptual_context
        if self._perceptual_context is not None:
            try:
                pc_text = self._perceptual_context.get_enrichment_text()
                if pc_text and "待機中" not in pc_text:
                    self_items.append(("知覚推移", f"知覚推移: {pc_text}"))
            except Exception:
                pass
        # #37 situational_self_presentation
        if self._situational_self_presentation is not None:
            try:
                ssp_data = self._situational_self_presentation.get_enrichment_data(
                    user_id=user_id,
                )
                ssp_text = ssp_data.get("summary_text", "")
                if ssp_text and "待機中" not in ssp_text:
                    self_items.append(("自己呈示", f"自己呈示: {ssp_text}"))
            except Exception:
                pass
        # #46 attention_distribution_description
        if self._att_dist_state is not None:
            try:
                att_text = get_att_dist_enrichment_text(self._att_dist_state)
                if att_text and "待機中" not in att_text:
                    self_items.append(("注意配分", f"注意配分: {att_text}"))
            except Exception:
                pass
        if self_items:
            sections_data.append({
                "header": "【自己認識】",
                "items": self_items,
            })

        # ── 【動機・目標】 ──
        motive_items: list[tuple[str, str]] = []
        if self._last_motives is not None:
            motive_summary = get_motive_summary(self._last_motives)
            if motive_summary:
                motive_items.append(("動機", f"動機: {motive_summary}"))
        if self._candidate_gen is not None:
            try:
                cand_summary = get_candidate_summary(self._candidate_gen)
                if cand_summary:
                    total = cand_summary.get("total_candidates", 0)
                    if total > 0:
                        motive_items.append(("目標候補", f"目標候補: {total}件"))
            except Exception:
                pass
        if self._last_expectations is not None:
            exp_summary = get_expectation_summary(self._last_expectations)
            if exp_summary:
                motive_items.append(("期待", f"期待: {exp_summary}"))
        # #3 scoped_goal
        if self._scoped_goal_sys is not None:
            sg_summary = get_scoped_goal_summary(self._scoped_goal_sys)
            if sg_summary and sg_summary.get("has_active_scope"):
                scope = sg_summary.get("current_scope", {})
                motive_items.append((
                    "スコープ目標",
                    f"スコープ目標: {scope.get('category', '?')} "
                    f"(strength={scope.get('strength', 0):.2f})",
                ))
        # #4 transient_goal
        if self._transient_goal_mgr is not None:
            tg_summary = get_transient_goal_summary(self._transient_goal_mgr)
            if tg_summary and tg_summary.get("has_active_goal"):
                goal = tg_summary.get("active_goal", {})
                motive_items.append((
                    "一時目標",
                    f"一時目標: {goal.get('category', '?')} "
                    f"(strength={goal.get('strength', 0):.2f})",
                ))
        # #31 persistent_commitment
        if self._persistent_commitment is not None:
            try:
                pc_text = get_commitment_summary(
                    self._persistent_commitment.state
                )
                if pc_text and "待機中" not in pc_text:
                    motive_items.append(("持続保持", f"持続保持: {pc_text}"))
            except Exception:
                pass
        # #5 proto_goal_vector
        if self._vector_gen is not None:
            vec_summary = get_vector_summary(self._vector_gen)
            if vec_summary and vec_summary.get("vector_count", 0) > 0:
                motive_items.append((
                    "方向ベクトル",
                    f"方向ベクトル: {vec_summary['vector_count']}本, "
                    f"最強={vec_summary['strongest_magnitude']:.2f}",
                ))
        # #14 policy_candidate_expansion
        if self._policy_expander is not None:
            try:
                exp_text = get_expansion_summary_text(self._policy_expander)
                if exp_text:
                    motive_items.append(("候補拡張", f"候補拡張: {exp_text}"))
            except Exception:
                pass
        # #18 spontaneous_activation
        if self._spontaneous_processor is not None:
            try:
                sp_text = get_spontaneous_summary(
                    self._spontaneous_processor.state
                )
                if sp_text and "待機中" not in sp_text:
                    motive_items.append(("自発起動", f"自発起動: {sp_text}"))
            except Exception:
                pass
        # #39 drive_variation_description
        if self._drive_variation_processor is not None:
            try:
                dv_data = self._drive_variation_processor.get_enrichment_data()
                dv_text = dv_data.get("summary_text", "")
                if dv_text and "待機中" not in dv_text:
                    motive_items.append(("駆動変動", f"駆動変動: {dv_text}"))
            except Exception:
                pass
        # #40 expectation_lifecycle_description
        if self._expectation_lifecycle_processor is not None:
            try:
                el_data = self._expectation_lifecycle_processor.get_enrichment_data()
                el_text = el_data.get("summary_text", "")
                if el_text and "待機中" not in el_text:
                    motive_items.append(("予期ライフサイクル", f"予期ライフサイクル: {el_text}"))
            except Exception:
                pass
        if motive_items:
            sections_data.append({
                "header": "【動機・目標】",
                "items": motive_items,
            })

        # ── 【記憶・内省】 ──
        memory_items: list[tuple[str, str]] = []
        if self._last_episodes is not None:
            ep_summary = get_episodic_memory_summary(self._last_episodes)
            if ep_summary:
                memory_items.append(("エピソード記憶", f"エピソード記憶: {ep_summary}"))
        if self._last_bindings is not None:
            bind_summary = get_binding_summary(self._last_bindings)
            if bind_summary:
                memory_items.append(("感情結合", f"感情結合: {bind_summary}"))
        if self._last_consumption is not None:
            cons_summary = get_consumption_summary(self._last_consumption)
            if cons_summary:
                memory_items.append(("内省消費", f"内省消費: {cons_summary}"))
        if self._last_other_model is not None:
            other_summary = get_other_model_summary(self._last_other_model)
            if other_summary:
                memory_items.append(("他者モデル", f"他者モデル: {other_summary}"))
        # #7 introspection_trace
        if self._last_trace is not None:
            trace_str = get_trace_summary(self._last_trace)
            if trace_str:
                memory_items.append(("内省", f"内省: {trace_str}"))
        # #15 memory_system_integration
        if self._memory_integrator is not None:
            try:
                int_text = get_integration_summary_text(self._memory_integrator)
                if int_text:
                    memory_items.append(("記憶統合", f"記憶統合: {int_text}"))
            except Exception:
                pass
        # #16 other_model_real_feed
        if self._real_feed_processor is not None:
            try:
                feed_text = get_real_feed_summary(self._real_feed_processor)
                if feed_text and "inactive" not in feed_text:
                    memory_items.append(("観測フィード", f"観測フィード: {feed_text}"))
            except Exception:
                pass
        # #17 text_dialogue_input
        if self._text_dialogue_processor is not None:
            try:
                tdi_text = get_text_dialogue_summary(
                    self._text_dialogue_processor.state
                )
                if tdi_text and "待機中" not in tdi_text:
                    memory_items.append(("入力経路", f"入力経路: {tdi_text}"))
            except Exception:
                pass
        # #20 memory_forgetting_fixation
        if self._forgetting_fixation_processor is not None:
            try:
                ff_text = get_forgetting_fixation_summary(
                    self._forgetting_fixation_processor.state
                )
                if ff_text and "待機中" not in ff_text:
                    memory_items.append(("記憶流動", f"記憶流動: {ff_text}"))
            except Exception:
                pass
        # #21 action_result_observation
        if self._action_result_observer is not None:
            try:
                ar_data = self._action_result_observer.get_enrichment_data()
                ar_text = ar_data.get("summary_text", "")
                if ar_text and "待機中" not in ar_text:
                    memory_items.append(("行動-結果", f"行動-結果: {ar_text}"))
            except Exception:
                pass
        # #22 other_model_dialogue_learning
        if self._dialogue_learning_processor is not None:
            try:
                dl_data = self._dialogue_learning_processor.get_enrichment_data()
                dl_text = dl_data.get("summary_text", "")
                if dl_text and "待機中" not in dl_text:
                    memory_items.append(("他者蓄積", f"他者蓄積: {dl_text}"))
            except Exception:
                pass
        # #44 other_boundary_accumulation
        if self._other_boundary_accumulation is not None:
            try:
                oba_data = self._other_boundary_accumulation.get_enrichment_data(
                    user_id=user_id,
                )
                oba_text = oba_data.get("summary_text", "")
                if oba_text and "待機中" not in oba_text:
                    memory_items.append(("境界蓄積", f"境界蓄積: {oba_text}"))
            except Exception:
                pass
        # #48 hypothesis_observation_pairing
        if self._hypothesis_observation_pairing is not None:
            try:
                hop_data = self._hypothesis_observation_pairing.get_enrichment_data()
                hop_text = hop_data.get("summary_text", "")
                if hop_text and "待機中" not in hop_text:
                    memory_items.append(("仮説-観測対", f"仮説-観測対: {hop_text}"))
            except Exception:
                pass
        # #23 meta_emotion_cognition
        if self._meta_emotion_processor is not None:
            try:
                me_data = self._meta_emotion_processor.get_enrichment_data()
                me_text = me_data.get("summary_text", "")
                if me_text and "待機中" not in me_text:
                    memory_items.append(("メタ感情", f"メタ感情: {me_text}"))
            except Exception:
                pass
        # #24 self_action_perception
        if self._self_action_recorder is not None:
            try:
                sa_data = self._self_action_recorder.get_enrichment_data()
                sa_text = sa_data.get("summary_text", "")
                if sa_text and "待機中" not in sa_text:
                    memory_items.append(("自己行動", f"自己行動: {sa_text}"))
            except Exception:
                pass
        # #25 expectation_action_diff
        try:
            diff_summary = self.get_expectation_diff_summary()
            diff_total = diff_summary["total_count"]
            if diff_total > 0:
                diff_parts = [f"予期差分記録: {diff_total}件"]
                for rec in diff_summary["recent_records"]:
                    exp_info = rec.get("expectation", {})
                    act_info = rec.get("action", {})
                    desc = exp_info.get("description", "")[:30]
                    pattern = act_info.get("pattern_key", "")
                    diff_parts.append(
                        f"  [{desc}] - [{pattern}]"
                    )
                memory_items.append(("予期差分記録", "\n".join(diff_parts)))
        except Exception:
            pass
        # #26 intent_action_gap
        if self._intent_action_gap_recorder is not None:
            try:
                gap_data = self._intent_action_gap_recorder.get_enrichment_data()
                gap_text = gap_data.get("summary_text", "")
                gap_lines: list[str] = []
                if gap_text and "待機中" not in gap_text:
                    gap_lines.append(f"意図-行動対: {gap_text}")
                recent_entries = gap_data.get("recent_entries", [])
                if recent_entries:
                    for entry in recent_entries:
                        label = entry.get("policy_label", "")
                        snippet = entry.get("text_snippet", "")[:40]
                        if label or snippet:
                            gap_lines.append(
                                f"  [{label}] - [{snippet}]"
                            )
                if gap_lines:
                    memory_items.append(("意図-行動対", "\n".join(gap_lines)))
            except Exception:
                pass
        # #28 multi_path_recall
        if self._multi_path_recall is not None:
            try:
                mpr_data = self._multi_path_recall.get_enrichment_data()
                mpr_text = mpr_data.get("summary_text", "")
                mpr_lines: list[str] = []
                if mpr_text and "待機中" not in mpr_text:
                    mpr_lines.append(f"多経路想起: {mpr_text}")
                mpr_entries = mpr_data.get("entries", [])
                if mpr_entries:
                    for entry in mpr_entries:
                        path_label = entry.get("path", "")
                        summary = entry.get("summary", "")[:40]
                        if path_label or summary:
                            mpr_lines.append(
                                f"  [{path_label}] {summary}"
                            )
                if mpr_lines:
                    memory_items.append(("多経路想起", "\n".join(mpr_lines)))
            except Exception:
                pass
        # #29 introspection_cross_section
        if self._introspection_cross_section is not None:
            try:
                ics_text = self._introspection_cross_section.get_enrichment_text()
                if ics_text and "待機中" not in ics_text:
                    memory_items.append(("内省横断", f"内省横断: {ics_text}"))
            except Exception:
                pass
        # #38 introspection_longitudinal_view
        if self._introspection_longitudinal_view is not None:
            try:
                ilv_snaps = self._introspection_cross_section.get_snapshot_window()
                ilv_data = self._introspection_longitudinal_view.get_enrichment_data(ilv_snaps)
                ilv_text = ilv_data.get("summary_text", "")
                if ilv_text and "待機中" not in ilv_text:
                    memory_items.append(("内省縦断", f"内省縦断: {ilv_text}"))
            except Exception:
                pass
        # #31 selection_attribution
        if self._selection_attribution_recorder is not None:
            try:
                sa_data = self._selection_attribution_recorder.get_enrichment_data()
                sa_text = sa_data.get("summary_text", "")
                sa_lines: list[str] = []
                if sa_text and "待機中" not in sa_text:
                    sa_lines.append(f"選択帰属: {sa_text}")
                recent_entries = sa_data.get("recent_entries", [])
                if recent_entries:
                    for entry in recent_entries:
                        label = entry.get("selected_policy_label", "")
                        count = entry.get("candidate_count", 0)
                        if label:
                            sa_lines.append(
                                f"  [{label}] 候補{count}件"
                            )
                if sa_lines:
                    memory_items.append(("選択帰属", "\n".join(sa_lines)))
            except Exception:
                pass
        # #33 spontaneous_recall
        if self._spontaneous_recall is not None:
            try:
                sr_data = self._spontaneous_recall.get_enrichment_data()
                sr_text = sr_data.get("summary_text", "")
                sr_lines: list[str] = []
                if sr_text and "待機中" not in sr_text:
                    sr_lines.append(f"自発想起: {sr_text}")
                sr_entries = sr_data.get("entries", [])
                if sr_entries:
                    for entry in sr_entries:
                        path_label = entry.get("path", "")
                        summary = entry.get("summary", "")[:40]
                        if path_label or summary:
                            sr_lines.append(
                                f"  [{path_label}] {summary}"
                            )
                if sr_lines:
                    memory_items.append(("自発想起", "\n".join(sr_lines)))
            except Exception:
                pass
        # #34 internal_contradiction_description
        if self._contradiction_processor is not None:
            try:
                ic_text = self._contradiction_processor.get_enrichment_text()
                if ic_text and "待機中" not in ic_text:
                    memory_items.append(("矛盾並置", f"矛盾並置: {ic_text}"))
            except Exception:
                pass
        # #35 interaction_accumulation
        if self._interaction_accumulation is not None:
            try:
                ia_data = self._interaction_accumulation.get_enrichment_data()
                ia_text = ia_data.get("summary_text", "")
                ia_lines: list[str] = []
                if ia_text and "待機中" not in ia_text:
                    ia_lines.append(f"相互作用: {ia_text}")
                ia_entries = ia_data.get("entries", [])
                if ia_entries:
                    for entry in ia_entries:
                        self_preview = entry.get("self_text", "")[:40]
                        other_preview = entry.get("other_reaction", "")[:40]
                        policy = entry.get("self_policy", "")
                        if self_preview or other_preview:
                            ia_lines.append(
                                f"  [{policy}] {self_preview} → {other_preview}"
                            )
                if ia_lines:
                    memory_items.append(("相互作用", "\n".join(ia_lines)))
            except Exception:
                pass
        # #36 emotional_backdrop_cognition
        if self._emotional_backdrop_processor is not None:
            try:
                eb_data = self._emotional_backdrop_processor.get_enrichment_data()
                eb_text = eb_data.get("summary_text", "")
                if eb_text and "待機中" not in eb_text:
                    memory_items.append(("感情基調", f"感情基調: {eb_text}"))
            except Exception:
                pass
        # #43 emotion_cooccurrence_description
        if self._emotion_cooccurrence_processor is not None:
            try:
                ec_data = self._emotion_cooccurrence_processor.get_enrichment_data()
                ec_text = ec_data.get("summary_text", "")
                ec_lines: list[str] = []
                if ec_text and "待機中" not in ec_text:
                    ec_lines.append(f"感情共起: {ec_text}")
                ec_entries = ec_data.get("entries", [])
                if ec_entries:
                    for entry in ec_entries:
                        if entry.get("no_cooccurrence"):
                            ec_lines.append(
                                f"  [tick={entry.get('tick', '?')}] 共起なし"
                            )
                        else:
                            pairs = entry.get("pairs", [])
                            pair_strs = [
                                f"{p['a']}={p['va']:.3f},{p['b']}={p['vb']:.3f}"
                                for p in pairs
                            ]
                            if pair_strs:
                                ec_lines.append(
                                    f"  [tick={entry.get('tick', '?')}] {'; '.join(pair_strs)}"
                                )
                if ec_lines:
                    memory_items.append(("感情共起", "\n".join(ec_lines)))
            except Exception:
                pass
        # #41 input_pathway_balance
        if self._input_pathway_balance_state is not None:
            try:
                ipb_text = get_pathway_balance_enrichment_text(
                    self._input_pathway_balance_state
                )
                if ipb_text and "待機中" not in ipb_text:
                    memory_items.append(("経路均衡", f"経路均衡: {ipb_text}"))
            except Exception:
                pass
        # #45 forgetting_recall_balance
        if self._frb_state is not None:
            try:
                frb_text = get_frb_enrichment_text(self._frb_state)
                if frb_text and "待機中" not in frb_text:
                    memory_items.append(("忘却想起均衡", f"忘却想起均衡: {frb_text}"))
            except Exception:
                pass
        if memory_items:
            sections_data.append({
                "header": "【記憶・内省】",
                "items": memory_items,
            })

        # ── 【判断傾向】 ── (Phase 30-35 cached)
        bias_items: list[tuple[str, str]] = []
        # #9 decision_bias
        if self._last_decision_bias is not None:
            db = self._last_decision_bias
            bias_items.append((
                "判断バイアス",
                f"判断バイアス: phase={db.dynamics_phase.value}, "
                f"valence={db.valence_bias:.2f}",
            ))
        # #11 tone modifier
        if self._last_tone_mod is not None:
            bias_items.append((
                "トーン推奨",
                f"トーン推奨: {self._last_tone_mod.recommended.value}",
            ))
        # #12 context_sensitivity
        if self._last_sensitivity_bias is not None:
            sb = self._last_sensitivity_bias
            if sb.caution_level > 0.5:
                bias_items.append((
                    "空気読み",
                    f"空気読み: caution={sb.caution_level:.2f}",
                ))
        # #13 silence/hesitation
        if self._last_has_silence:
            bias_items.append(("沈黙傾向", "沈黙傾向: あり"))
        # #19 value_orientation_validation
        if self._vo_validator is not None:
            try:
                vo_text = get_vo_validation_summary(self._vo_validator.state)
                if vo_text and "待機中" not in vo_text:
                    bias_items.append(("価値検証", f"価値検証: {vo_text}"))
            except Exception:
                pass
        # #20 内部-外部間の張力サマリー (READ-ONLY参照のみ)
        try:
            tension_parts: list[str] = []
            # (a) persistent_commitment の方向的張力
            if self._persistent_commitment is not None:
                pc_active = [
                    it for it in self._persistent_commitment.state.items
                    if not it.released
                ]
                if pc_active:
                    tension_parts.append(f"保持方向{len(pc_active)}件の方向的バイアスあり")
            # (b) context_sensitivity の慎重度
            if (self._last_sensitivity_bias is not None
                    and self._last_sensitivity_bias.caution_level > 0.5):
                tension_parts.append("外部文脈の慎重度が高い")
            # (c) value_orientation の最強方向バイアス (vo_validation未記述時のみ)
            vo_already_shown = (
                self._vo_validator is not None
                and any(lbl == "価値検証" for lbl, _ in bias_items)
            )
            if self._value_orientation is not None and not vo_already_shown:
                dims = self._value_orientation.get_all_dimensions()
                if dims:
                    max_dim = max(dims.values(), key=abs)
                    if abs(max_dim) > 0.1:
                        tension_parts.append("価値軸の傾斜方向あり")
            if tension_parts:
                bias_items.append((
                    "内部-外部間の張力",
                    "内部-外部間の張力: " + " / ".join(tension_parts),
                ))
        except Exception:
            pass
        if bias_items:
            sections_data.append({
                "header": "【判断傾向】",
                "items": bias_items,
            })

        return sections_data

    def get_prompt_enrichment(self, user_id: str = "viewer") -> str:
        """Gemini プロンプト用の心理状態テキストを生成する。

        enrichment項目を収集後、圧縮パイプライン（第1段: 変動度算出、
        第2段: 粒度選択、第3段: フォーマット圧縮）を適用する。
        圧縮結果のフィードバック経路は構造的に遮断されている。

        brain.py の _format_psyche_for_prompt を置き換える。

        Args:
            user_id: 責任状態取得用のユーザーID
        """
        # 項目収集（既存ロジックの構造を維持）
        sections_data = self._collect_enrichment_items(user_id)

        footer = ORIGINAL_FOOTER

        # 圧縮パイプライン実行
        compressed_text, new_cache, _ratio = build_compressed_enrichment(
            sections_data=sections_data,
            prev_cache=self._enrichment_prev_cache,
            footer=footer,
        )

        # キャッシュ更新（1ティック分のみ保持）
        self._enrichment_prev_cache = new_cache

        return compressed_text

    # ── Policy suggestions ────────────────────────────────────────

    def _generate_final_candidates(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        user_id: str,
    ) -> tuple[list[dict], Any]:
        """Phase 30-35: 候補生成+バイアス適用。candidatesとtone_modを返す。"""
        resp_influence = self._responsibility_mgr.get_influence(user_id)

        # Phase 31: decision_bias — STM→判断バイアス計算
        decision_bias = compute_decision_bias(
            memory=self._loop_state.memory if self._loop_state else None,
            dynamics=self._dynamics,
            config=self._decision_bias_config,
        )

        # Phase 30直前: cross_section_inputs計算（thought.py + expansion両方で使用）
        try:
            cs_inputs = self._build_cross_section_inputs(
                percept, recalled_memories, resp_influence, user_id,
            )
        except Exception:
            cs_inputs = None

        # Phase 30: thought — 候補ポリシー生成
        # extended_inputs を構築してthought.pyに渡す
        extended_inputs = None
        if cs_inputs is not None:
            extended_inputs = {
                "self_image_stability": cs_inputs.self_image_stability,
                "coherence_level": cs_inputs.coherence_level,
                "strain_level": cs_inputs.strain_level,
                "narrative_coherence": cs_inputs.narrative_coherence,
                "tendency_count": cs_inputs.tendency_count,
                "dominant_tendency": cs_inputs.dominant_tendency,
                "tendency_strength": cs_inputs.tendency_strength,
                "other_count": cs_inputs.other_model_count,
                "boundary_clarity": cs_inputs.other_boundary_clarity,
                "has_active_goal": cs_inputs.has_active_goal,
                "goal_strength": cs_inputs.goal_strength,
                "motive_count": cs_inputs.motive_count,
                "me_supply_strength": cs_inputs.meta_emotion_supply_strength,
            }

        candidates = generate_thought_candidates(
            state=self._psyche,
            percept=percept,
            recalled=recalled_memories,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
            extended_inputs=extended_inputs,
        )

        # Phase 30b: policy_candidate_expansion — 内面反映候補拡張
        # cs_inputsを再利用（既にtry/exceptの外で計算済み）
        if cs_inputs is not None:
            try:
                expanded = self._policy_expander.expand_candidates(
                    base_candidates=candidates,
                    inputs=cs_inputs,
                )
                candidates.extend(expanded)
                # 拡張後にスコア降順で再ソート
                candidates.sort(key=lambda c: c.get("_score", 0), reverse=True)
            except Exception as e:
                logger.debug("Policy expansion skipped: %s", e)

        # Phase 32: tone — トーン修飾子計算
        tone_mod = compute_tone_bias(
            state=self._psyche,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
            config=self._tone_config,
        )

        # Phase 33: context_sensitivity — 空気読みバイアス (input_supply経由)
        ctx_snapshot = supply_context(self._input_supply)
        ext_ctx = ExternalContext(
            pace=ctx_snapshot.pace,
            weight=ctx_snapshot.weight,
            density=ctx_snapshot.density,
            continuity=ctx_snapshot.continuity,
            responsiveness=ctx_snapshot.responsiveness,
        )
        sensitivity_bias = compute_sensitivity_bias(
            context=ext_ctx,
            config=self._ctx_sensitivity_config,
            context_state=self._ctx_state,
        )
        candidates = apply_sensitivity_to_candidates(
            candidates=candidates,
            bias=sensitivity_bias,
        )

        # Phase 34: silence_hesitation — 沈黙候補生成
        candidates = generate_candidates_with_silence(
            state=self._psyche,
            percept=percept,
            recalled=recalled_memories,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
            silence_config=self._silence_config,
            base_candidates=candidates,
        )

        # Phase 35: stability_valve — 安定化バイアス適用
        try:
            stability_bias = self._stability_valve.generate_bias()
            candidates = apply_stability_bias(
                candidates=candidates,
                bias=stability_bias,
            )
        except Exception:
            pass

        # Phase 35b: value_orientation — 価値軸バイアス適用
        try:
            candidates = apply_orientation_to_candidates(
                candidates=candidates,
                orientation=self._value_orientation,
            )
        except Exception:
            pass

        # Phase 35b2: persistent_commitment — 持続的取り組み保持バイアス適用
        # value_orientation の後、scoring_fluctuation の前に適用
        try:
            candidates = self._persistent_commitment.apply_bias_to_candidates(
                candidates=candidates,
            )
        except Exception:
            pass

        # Phase 35c: scoring_fluctuation — スコアリングの構造的揺らぎ
        # 全バイアス適用完了後の最後の加算層として、内部状態由来の揺らぎを適用
        try:
            now = time.time()
            elapsed = now - self._last_fluctuation_select_time if self._last_fluctuation_select_time > 0 else 0.0
            self._last_fluctuation_select_time = now

            p = self._psyche
            candidates = apply_scoring_fluctuation(
                candidates=candidates,
                emotions=p.emotions.as_dict() if p.emotions else {},
                drives=p.drives.as_dict() if p.drives else {},
                stm=self._loop_state.memory if self._loop_state else None,
                elapsed_seconds=elapsed,
                config=self._fluctuation_config,
            )
        except Exception as e:
            logger.debug("Scoring fluctuation skipped: %s", e)

        # Cache Phase 30-35 results for enrichment
        self._last_decision_bias = decision_bias
        self._last_tone_mod = tone_mod
        self._last_sensitivity_bias = sensitivity_bias
        self._last_has_silence = any(
            c.get("policy_label") == "silence" for c in candidates
        )

        return candidates, tone_mod

    def _build_cross_section_inputs(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        resp_influence: Optional[ResponsibilityInfluence],
        user_id: str,
    ) -> CrossSectionInputs:
        """各断面の参照データを CrossSectionInputs に集約する。"""
        p = self._psyche
        emotions = p.emotions.as_dict() if p.emotions else {}

        # 傾向断面
        tendency_count = 0
        dominant_tendency = ""
        tendency_strength = 0.0
        if self._tendency_sys is not None:
            tendencies = self._tendency_sys.state.tendencies
            tendency_count = len(tendencies)
            if tendencies:
                strongest = max(tendencies, key=lambda t: t.strength)
                dominant_tendency = strongest.pattern.value if hasattr(strongest, 'pattern') else ""
                tendency_strength = strongest.strength

        # 責任断面
        caution = 0.0
        empathy = 0.0
        if resp_influence is not None:
            caution = resp_influence.caution_bias
            empathy = resp_influence.empathy_bias

        # 自己観測断面
        self_image_stability = 0.5
        coherence_level = 0.5
        strain_level = 0.0
        narrative_coherence = 0.5
        if self._last_self_image is not None:
            self_image_stability = getattr(self._last_self_image, 'stability', 0.5)
        if self._last_coherence is not None:
            coherence_level = getattr(self._last_coherence, 'overall_level', 0.5)
            if hasattr(coherence_level, 'value'):
                # Enum → float mapping
                level_map = {"high": 0.8, "moderate": 0.5, "low": 0.2, "unstable": 0.1}
                coherence_level = level_map.get(str(coherence_level), 0.5)
        if self._last_strain is not None:
            strain_level = getattr(self._last_strain, 'level', 0.0)
            if hasattr(strain_level, 'value'):
                level_map = {"none": 0.0, "low": 0.2, "moderate": 0.5, "high": 0.8, "critical": 1.0}
                strain_level = level_map.get(str(strain_level), 0.0)
        if self._last_narrative is not None:
            narrative_coherence = getattr(self._last_narrative, 'coherence', 0.5)
            if hasattr(narrative_coherence, 'level'):
                narrative_coherence = getattr(narrative_coherence, 'level', 0.5)

        # 他者推定断面
        other_count = 0
        boundary_clarity = 0.5
        if self._last_other_model is not None:
            hypotheses = getattr(self._last_other_model, 'hypotheses', [])
            other_count = len(hypotheses) if hypotheses else 0
            boundary = getattr(self._last_other_model, 'boundary', None)
            if boundary is not None:
                boundary_clarity = getattr(boundary, 'clarity', 0.5)

        # 目的断面
        has_active_goal = False
        goal_strength = 0.0
        if self._transient_goal_mgr is not None:
            active = getattr(self._transient_goal_mgr, 'active_goal', None)
            if active is None:
                active = getattr(self._transient_goal_mgr.state, 'active_goal', None)
            if active is not None:
                has_active_goal = True
                goal_strength = getattr(active, 'strength', 0.5)

        motive_count = 0
        if self._last_motives is not None:
            motives = getattr(self._last_motives, 'entries', [])
            motive_count = len(motives) if motives else 0

        expectation_count = 0
        if self._last_expectations is not None:
            expectations = getattr(self._last_expectations, 'candidates', [])
            expectation_count = len(expectations) if expectations else 0

        vector_count = 0
        if self._vector_gen is not None:
            vectors = getattr(self._vector_gen.state, 'vectors', [])
            vector_count = len(vectors) if vectors else 0

        # [B] action_result → policy_candidate_expansion 参照情報供給
        # 蓄積された行動-結果対の情報を候補生成時の参照情報として提供する。
        # 「スコア加算」ではなく「こういう経験がある」という情報形式。
        ar_active_count = 0
        ar_pattern_dist: dict[str, int] = {}
        ar_convergence_warning = False
        if self._action_result_observer is not None:
            try:
                ar_data = self._action_result_observer.get_enrichment_data()
                ar_active_count = ar_data.get("active_count", 0)
                ar_pattern_dist = ar_data.get("pattern_distribution", {})
                ar_convergence_warning = ar_data.get("pattern_convergence_warning", False)
            except Exception:
                pass

        # [E] meta_emotion → policy_candidate_expansion 変動候補断面供給
        # 変動候補群をポリシー候補拡張モジュールの新たな断面として供給する。
        # 既存の感情断面とは別の断面として、感情の推移特徴に基づく変動可能性の情報。
        me_candidate_count = 0
        me_pattern_count = 0
        me_supply_strength = 0.0
        if self._meta_emotion_processor is not None:
            try:
                me_data = self._meta_emotion_processor.get_enrichment_data()
                me_candidate_count = me_data.get("candidate_count", 0)
                me_pattern_count = me_data.get("active_pattern_count", 0)
                me_supply_strength = me_data.get("supply_strength", 0.0)
            except Exception:
                pass

        return CrossSectionInputs(
            emotions=emotions,
            mood_valence=p.mood.valence,
            mood_arousal=p.mood.arousal,
            recalled_count=len(recalled_memories) if recalled_memories else 0,
            has_emotional_bindings=self._last_bindings is not None,
            episode_count=len(getattr(self._last_episodes, 'entries', [])) if self._last_episodes else 0,
            tendency_count=tendency_count,
            dominant_tendency=dominant_tendency,
            tendency_strength=tendency_strength,
            caution_bias=caution,
            empathy_bias=empathy,
            dispersion_active=self._dispersion_state is not None,
            percept_intent=percept.intent,
            percept_valence=percept.emotion_valence,
            percept_text_length=len(percept.text),
            self_image_stability=self_image_stability if isinstance(self_image_stability, (int, float)) else 0.5,
            coherence_level=coherence_level if isinstance(coherence_level, (int, float)) else 0.5,
            strain_level=strain_level if isinstance(strain_level, (int, float)) else 0.0,
            narrative_coherence=narrative_coherence if isinstance(narrative_coherence, (int, float)) else 0.5,
            other_model_count=other_count,
            other_boundary_clarity=boundary_clarity if isinstance(boundary_clarity, (int, float)) else 0.5,
            has_active_goal=has_active_goal,
            goal_strength=goal_strength,
            motive_count=motive_count,
            expectation_count=expectation_count,
            vector_count=vector_count,
            action_result_active_count=ar_active_count,
            action_result_pattern_distribution=ar_pattern_dist,
            action_result_convergence_warning=ar_convergence_warning,
            meta_emotion_candidate_count=me_candidate_count,
            meta_emotion_pattern_count=me_pattern_count,
            meta_emotion_supply_strength=me_supply_strength,
        )

    def _build_vo_validation_inputs(self, user_id: str) -> VOValidationInputs:
        """価値方向性検証に必要な8断面の入力を構築する。"""
        p = self._psyche

        # 1. 価値方向性断面
        orientation_dims = self._value_orientation.get_all_dimensions()
        orientation_confs = self._value_orientation.get_all_confidences()

        # 2. 行動候補断面（直近のPhase 30-35キャッシュから）
        candidate_count = 0
        top_label = ""
        top_score = 0.0
        candidate_diversity = 0.0

        # 3. 選択履歴断面（repeated_tendency から）
        recent_selections: list[str] = []
        selection_consistency = 0.0
        if self._tendency_sys is not None:
            tendencies = self._tendency_sys.state.tendencies
            for t in tendencies[:10]:
                pattern = getattr(t, 'pattern', None)
                if pattern is not None:
                    recent_selections.append(
                        pattern.value if hasattr(pattern, 'value') else str(pattern)
                    )
            if tendencies:
                strengths = [t.strength for t in tendencies]
                if len(strengths) > 1:
                    avg = sum(strengths) / len(strengths)
                    variance = sum((s - avg) ** 2 for s in strengths) / len(strengths)
                    selection_consistency = max(0.0, 1.0 - variance)
                else:
                    selection_consistency = strengths[0] if strengths else 0.0

        # 4. 文脈断面
        ctx_pace = 0.0
        ctx_density = 0.0
        ctx_continuity = 0.0
        try:
            ctx_snapshot = supply_context(self._input_supply)
            ctx_pace = ctx_snapshot.pace
            ctx_density = ctx_snapshot.density
            ctx_continuity = ctx_snapshot.continuity
        except Exception:
            pass

        # 5. 感情推移断面
        emotions = p.emotions.as_dict() if p.emotions else {}

        # 6. 記憶参照断面
        recalled_count = len(self._last_recalled_memories) if self._last_recalled_memories else 0
        has_bindings = self._last_bindings is not None
        episode_count = 0
        if self._last_episodes is not None:
            entries = getattr(self._last_episodes, 'entries', [])
            episode_count = len(entries) if entries else 0

        # 7. 責任断面
        caution = 0.0
        empathy = 0.0
        resp_weight = 0.0
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            caution = resp_influence.caution_bias
            empathy = resp_influence.empathy_bias
            resp_weight = caution
        except Exception:
            pass

        # 8. 時間経過断面
        elapsed = 0.0
        if self._state_for_vo_validation_last_time > 0:
            elapsed = time.time() - self._state_for_vo_validation_last_time
        self._state_for_vo_validation_last_time = time.time()

        return VOValidationInputs(
            orientation_dimensions=orientation_dims,
            orientation_confidences=orientation_confs,
            orientation_update_count=self._value_orientation.update_count,
            candidate_count=candidate_count,
            top_candidate_label=top_label,
            top_candidate_score=top_score,
            candidate_diversity=candidate_diversity,
            recent_selections=recent_selections,
            selection_consistency=selection_consistency,
            context_pace=ctx_pace,
            context_density=ctx_density,
            context_continuity=ctx_continuity,
            emotion_valence=p.mood.valence,
            emotion_arousal=p.mood.arousal,
            emotions=emotions,
            recalled_count=recalled_count,
            has_bindings=has_bindings,
            episode_count=episode_count,
            caution_bias=caution,
            empathy_bias=empathy,
            responsibility_weight=resp_weight,
            tick_count=self._tick_count,
            elapsed_since_last=elapsed,
        )

    def _build_forgetting_fixation_inputs(self) -> ForgettingFixationInputs:
        """記憶の忘却と固定化に必要な8断面の入力を構築する。"""
        p = self._psyche

        # 1. 記憶参照頻度断面 / 2. 再利用間隔断面 — エピソード・結合・長期
        ep_entries: list[dict] = []
        if self._last_episodes is not None:
            entries = getattr(self._last_episodes, 'entries', [])
            if entries:
                for e in entries:
                    ep_entries.append({
                        "id": getattr(e, 'id', getattr(e, 'episode_id', '')),
                        "emotional_valence": getattr(e, 'emotional_valence', 0.0),
                    })

        bind_entries: list[dict] = []
        if self._last_bindings is not None:
            traces = getattr(self._last_bindings, 'traces', [])
            if traces:
                for t in traces:
                    bind_entries.append({
                        "id": getattr(t, 'id', getattr(t, 'binding_id', '')),
                        "freshness": getattr(t, 'freshness', 0.0),
                    })

        lt_entries: list[dict] = []
        if self._last_recalled_memories:
            for m in self._last_recalled_memories:
                mid = m.get("id", m.get("memory_id", ""))
                if mid:
                    lt_entries.append({
                        "id": mid,
                        "emotional_valence": m.get("emotional_valence", 0.0),
                    })

        # 3. 時系列断面
        tick = self._tick_count

        # 4. 競合系列断面
        active_count = 0
        dominant_id = ""
        if self._forgetting_fixation_processor:
            state = self._forgetting_fixation_processor.state
            active_count = len(state.series_index)
            if state.series_index:
                dominant = max(state.series_index, key=lambda s: s.reference_count)
                dominant_id = dominant.source_id

        # 5. 感情連結断面
        bind_count = len(bind_entries)
        avg_freshness = 0.0
        if bind_entries:
            avg_freshness = sum(b.get("freshness", 0.0) for b in bind_entries) / len(bind_entries)

        # 6. 文脈連結断面
        ctx_continuity = 0.0
        ctx_density = 0.0
        try:
            ctx_snapshot = supply_context(self._input_supply)
            ctx_continuity = ctx_snapshot.continuity
            ctx_density = ctx_snapshot.density
        except Exception:
            pass

        # 7. 保護状態断面 — scoped_goalのメモリIDを保護
        protected: list[str] = []
        if self._scoped_goal_sys is not None:
            goal_mem = getattr(self._scoped_goal_sys, 'source_memory_id', '')
            if goal_mem:
                protected.append(goal_mem)

        # 8. 固定化兆候断面 — 直近で繰り返し参照されたID
        repeated_ids: list[str] = []
        if self._forgetting_fixation_processor:
            rh = self._forgetting_fixation_processor.state.reference_history
            if len(rh) >= 5:
                from collections import Counter
                recent = [r.get("source_id", "") for r in rh[-10:]]
                counts = Counter(recent)
                repeated_ids = [sid for sid, cnt in counts.items() if cnt >= 3 and sid]

        invisible_alt = 0
        if self._forgetting_fixation_processor:
            invisible_alt = sum(
                1 for s in self._forgetting_fixation_processor.state.series_index
                if s.forgetting_stage == "invisible"
            )

        # [I] action_result → memory_forgetting_fixation 忘却対象登録
        # 行動-結果対の鮮度状態を既存の記憶鮮度構造と互換性のある形式で供給する。
        ar_ff_entries: list[dict] = []
        if self._action_result_observer is not None:
            try:
                ar_freshness_info = self._action_result_observer.get_freshness_compatible_info()
                for info in ar_freshness_info:
                    ar_ff_entries.append({
                        "id": info.get("id", ""),
                        "freshness": info.get("freshness", 0.0),
                        "freshness_stage": info.get("freshness_stage", "active"),
                        "status": info.get("status", ""),
                    })
            except Exception:
                pass

        return ForgettingFixationInputs(
            episode_entries=ep_entries,
            binding_entries=bind_entries,
            long_term_entries=lt_entries,
            action_result_entries=ar_ff_entries,
            reuse_history=dict(self._forgetting_fixation_processor.state.reuse_history)
            if self._forgetting_fixation_processor else {},
            tick_count=tick,
            elapsed_since_last=0.0,
            active_series_count=active_count,
            dominant_series_id=dominant_id,
            emotion_valence=p.mood.valence,
            emotion_arousal=p.mood.arousal,
            binding_count=bind_count,
            average_binding_freshness=avg_freshness,
            context_continuity=ctx_continuity,
            context_density=ctx_density,
            protected_ids=protected,
            repeated_reference_ids=repeated_ids,
            invisible_alternative_count=invisible_alt,
        )

    def _build_action_result_inputs(self, user_id: str) -> ActionResultInputs:
        """行動-結果観測に必要な8断面の入力を構築する。"""
        p = self._psyche

        # 1. 直近行動断面 — select_policy_dict で記録されたラベル
        policy_label = self._last_selected_policy_label
        policy_axis = self._last_selected_policy_axis

        # 2. 外部反応断面 — 直近 percept の emotion valence を外部反応として使用
        ext_change = 0.0
        ext_desc = ""
        if self._last_percept is not None:
            ext_change = abs(getattr(self._last_percept, 'emotion_valence', 0.0))
            ext_desc = getattr(self._last_percept, 'emotion', '') or ""

        # 3. 内部状態変化断面 — dynamics & motivation
        internal_delta = 0.0
        if self._dynamics is not None:
            internal_delta = abs(getattr(self._dynamics, 'accumulated_intensity', 0.0))
        motivation_delta = 0.0
        if self._last_motives is not None:
            motives_list = getattr(self._last_motives, 'motives', [])
            if motives_list:
                motivation_delta = sum(
                    getattr(m, 'strength', 0.0) for m in motives_list
                ) / len(motives_list)
        direction_delta = 0.0
        if self._value_orientation is not None:
            axes = getattr(self._value_orientation, 'axes', {})
            if axes:
                direction_delta = sum(abs(v) for v in axes.values()) / len(axes)

        # 4. 感情推移断面 — 前回保持の感情 vs 現在
        emotion_after = p.emotions.as_dict() if p.emotions else {}
        emotion_before = dict(self._last_emotion_for_action_result)
        # 次回用に現在の感情を保持
        self._last_emotion_for_action_result = dict(emotion_after)

        # 5. 文脈断面
        context_summary = ""
        dialogue_state = ""
        environment_tags: list[str] = []
        if self._last_percept is not None:
            context_summary = getattr(self._last_percept, 'text', '')[:100]
            dialogue_state = getattr(self._last_percept, 'intent', 'unknown') or 'unknown'
            environment_tags = list(
                getattr(self._last_percept, 'topics', ()) or ()
            )

        # 6. 時間経過断面
        ticks_since = self._tick_count

        # 7. 他者観測断面
        other_change = 0.0
        other_desc = ""
        if self._last_feed_result is not None:
            fragments = getattr(self._last_feed_result, 'fragments', [])
            if fragments:
                other_change = sum(
                    getattr(f, 'confidence', 0.0) for f in fragments
                ) / len(fragments)
                other_desc = getattr(fragments[0], 'description', '') if fragments else ""

        # 8. 記憶参照断面
        ref_ids: list[str] = []
        if self._last_recalled_memories:
            ref_ids = [
                m.get("id", m.get("memory_id", ""))
                for m in self._last_recalled_memories
                if m.get("id") or m.get("memory_id")
            ]

        # 9. テキスト断面（自己行動知覚から供給）
        sa_output_text = ""
        if self._self_action_recorder is not None:
            sa_output_text = self._self_action_recorder.get_text_for_action_result()

        return ActionResultInputs(
            selected_policy_label=policy_label,
            selected_policy_axis=policy_axis,
            selection_context_summary=context_summary,
            action_tick=self._tick_count,
            external_response_change=ext_change,
            external_response_description=ext_desc,
            internal_state_delta=internal_delta,
            motivation_delta=motivation_delta,
            direction_delta=direction_delta,
            emotion_before=emotion_before,
            emotion_after=emotion_after,
            context_summary=context_summary,
            dialogue_state=dialogue_state,
            environment_tags=environment_tags,
            ticks_since_action=ticks_since,
            elapsed_seconds=0.0,
            other_reaction_change=other_change,
            other_reaction_description=other_desc,
            referenced_memory_ids=ref_ids,
            referenced_memory_count=len(ref_ids),
            output_text=sa_output_text,
            current_tick=self._tick_count,
        )

    def _build_dialogue_learning_inputs(self, user_id: str) -> DialogueLearningInputs:
        """他者観測の長期蓄積に必要な8断面の入力を構築する。

        Phase 25a(real_feed)の結果とaction_result_observerの他者断面を入力として使用。
        因果帰属は行わない。時系列的隣接の記録のみ。
        """
        # 1. 短期観測断片集合 — real_feed の観測ユニットを変換
        short_term_fragments: list[dict] = []
        if self._last_feed_result is not None:
            for unit in getattr(self._last_feed_result, "units", []):
                short_term_fragments.append({
                    "type": "|".join(getattr(unit, "source_types", [])) or "observation",
                    "description": getattr(unit, "description", ""),
                    "value": getattr(unit, "value", 0.5),
                    "confidence": 0.5 + 0.5 * getattr(unit, "value", 0.5),
                    "text_hint": "",
                })

        # 2. 行動-結果対の他者観測断面 — action_result_observer の enrichment から
        action_result_other: list[dict] = []
        if self._action_result_observer is not None:
            try:
                ar_data = self._action_result_observer.get_enrichment_data()
                sec_dist = ar_data.get("section_distribution", {})
                if "other_observation" in sec_dist:
                    action_result_other.append({
                        "observation_type": "other_observation",
                        "description": "action-result other observation",
                        "value": sec_dist["other_observation"] * 0.1,
                        "confidence": 0.5,
                    })
            except Exception:
                pass

        # 3. 対話文脈断面
        context_summary = ""
        dialogue_state = ""
        topic = ""
        if self._last_percept is not None:
            context_summary = getattr(self._last_percept, "text", "")[:100]
            dialogue_state = getattr(self._last_percept, "intent", "unknown") or "unknown"
            topics = getattr(self._last_percept, "topics", []) or []
            topic = topics[0] if topics else ""

        # 4. 相手識別断面
        # user_id は orchestrator から引き渡される

        # 5. 感情トーン断面
        emotion_tone = ""
        emotion_value = 0.0
        if self._last_percept is not None:
            emotion_tone = getattr(self._last_percept, "emotion", "neutral") or "neutral"
            emotion_value = getattr(self._last_percept, "emotion_valence", 0.0)

        # 6. 反応間隔断面
        response_interval = 0.0

        # 7. 話題遷移断面
        topic_changed = False
        previous_topic = ""

        # 8. 蓄積鮮度断面
        dl_state = self._dialogue_learning_processor.state
        existing_count = len(dl_state.entries)
        avg_freshness = 0.0
        if dl_state.entries:
            avg_freshness = sum(e.freshness for e in dl_state.entries) / len(dl_state.entries)

        return DialogueLearningInputs(
            short_term_fragments=short_term_fragments,
            action_result_other_observations=action_result_other,
            context_summary=context_summary,
            dialogue_state=dialogue_state,
            topic=topic,
            user_id=user_id,
            emotion_tone=emotion_tone,
            emotion_value=emotion_value,
            response_interval_seconds=response_interval,
            topic_changed=topic_changed,
            previous_topic=previous_topic,
            existing_entry_count=existing_count,
            average_freshness=avg_freshness,
            current_tick=self._tick_count,
        )

    def _run_persistent_commitment(self) -> None:
        """Phase 12b: 持続的取り組み保持の昇格チェック + ティック処理。

        transient_goal からの昇格が唯一の生成経路。
        昇格はコピーであり移動ではない（transient_goal側は変更しない）。
        """
        p = self._psyche
        tg_mgr = self._transient_goal_mgr
        pc = self._persistent_commitment

        # ── 昇格チェック ──
        # transient_goal にアクティブな目標があれば昇格候補として評価
        active_goal = tg_mgr.state.active_goal
        if active_goal is not None:
            # 維持ティック数: turn_countからの近似
            maintained_ticks = tg_mgr.state.turn_count
            pc.try_promote(
                goal_id=active_goal.goal_id,
                category=active_goal.candidate_category.value,
                direction_signature=dict(active_goal.direction_alignment),
                remaining_strength=active_goal.selection_strength,
                maintained_ticks=maintained_ticks,
                current_tick=self._tick_count,
            )

        # ── 8断面入力の構築 ──
        emo = p.emotions.as_dict() if p.emotions else {}
        drives = p.drives.as_dict() if p.drives else {}

        # 断面1: 支配的感情と覚醒度の変化量
        arousal_delta = abs(p.mood.arousal - 0.5) if p.mood else 0.0

        # 断面2: 文脈連続性の断裂度
        context_disruption = 0.0
        if self._last_coupling is not None:
            context_disruption = 1.0 - getattr(
                self._last_coupling, 'coupling_strength', 0.5
            )

        # 断面3: 内的推進力の変動幅
        drive_variability = 0.0
        if drives:
            vals = list(drives.values())
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                drive_variability = sum(abs(v - mean) for v in vals) / len(vals)

        # 断面4: 現在の注目と保持項目の方向的距離
        transient_dir_dist = 0.0
        if active_goal is not None and pc.state.items:
            active_items = [it for it in pc.state.items if not it.released]
            if active_items:
                from .persistent_commitment import _direction_similarity
                avg_sim = sum(
                    _direction_similarity(
                        active_goal.direction_alignment,
                        it.direction_signature,
                    )
                    for it in active_items
                ) / len(active_items)
                transient_dir_dist = max(0.0, 1.0 - avg_sim)

        # 断面5: 長期傾斜との整合度の変化量
        orientation_delta = 0.0

        # 断面6: 競合する候補の出現状態
        competing_intensity = 0.0
        if self._candidate_gen is not None:
            try:
                cands = self._candidate_gen.state.candidates
                if cands:
                    competing_intensity = max(c.intensity for c in cands) if cands else 0.0
            except Exception:
                pass

        # 断面7: 責任容量の圧迫度
        resp_pressure = 0.0
        try:
            resp_summary = self._responsibility_mgr.get_summary("viewer")
            resp_pressure = min(1.0, resp_summary.get("total_weight", 0.0))
        except Exception:
            pass

        # 断面8: 非決定性由来の変動量
        scoring_fluct = 0.0
        if self._last_fluctuation_select_time > 0:
            scoring_fluct = min(1.0, (time.time() - self._last_fluctuation_select_time) / 300.0)

        inputs = CommitmentCrossSectionInputs(
            dominant_emotion=p.dominant_emotion if p.dominant_emotion else "",
            arousal_delta=arousal_delta,
            context_disruption=context_disruption,
            drive_variability=drive_variability,
            transient_direction_distance=transient_dir_dist,
            orientation_alignment_delta=orientation_delta,
            competing_candidate_intensity=competing_intensity,
            responsibility_pressure=resp_pressure,
            scoring_fluctuation_amount=scoring_fluct,
        )

        # ── ティック処理 ──
        pc.tick(inputs, current_tick=self._tick_count)

    def _build_contradiction_inputs(self) -> ContradictionInputs:
        """矛盾並置に必要な入力を構築する。

        全てREAD-ONLY参照。いかなるモジュールの内部状態にも書き込まない。
        判断・行動・責任の各処理系統に接続しない。
        """
        # 1. 自己モデルの統合ビュー（感情側面の出力）
        sm_intensity = 0.0
        sm_spread = 0.0
        sm_conflict = False
        if self._last_self_view is not None:
            emo_view = self._last_self_view.emotional
            spread_map = {"focused": 0.2, "mixed": 0.5, "diffuse": 0.8, "undefined": 0.0}
            sm_spread = spread_map.get(emo_view.spread.value, 0.0)
            intensity_map = {"calm": 0.15, "moderate": 0.5, "intense": 0.85, "overwhelming": 1.0, "undefined": 0.0}
            sm_intensity = intensity_map.get(emo_view.intensity.value, 0.0)
            sm_conflict = emo_view.has_coexisting_pairs

        # 2. メタ感情認知の変動候補列挙
        me_change_speed = 0.0
        me_dominant_stability = 0.0
        if self._last_meta_emotion is not None:
            me_state = self._meta_emotion_processor.state
            if me_state.cognition_history:
                latest = me_state.cognition_history[-1]
                me_change_speed = getattr(latest, "change_speed", 0.0)
                me_dominant_stability = getattr(latest, "dominant_stability", 0.0)

        # 3. 自己像統合の暫定的自己像
        si_stability = 0.5
        si_continuity = 0.5
        si_emotional_tone = 0.5
        if self._last_self_image is not None:
            stability_map = {"grounded": 1.0, "mostly_settled": 0.7, "shifting": 0.3, "turbulent": 0.0}
            si_stability = stability_map.get(self._last_self_image.stability_feeling.value, 0.5)
            continuity_map = {"continuous": 1.0, "mostly_familiar": 0.7, "fading": 0.3, "disconnected": 0.0}
            si_continuity = continuity_map.get(self._last_self_image.continuity_feeling.value, 0.5)
            tone_map = {"calm": 1.0, "stirred": 0.6, "intense": 0.2}
            si_emotional_tone = tone_map.get(self._last_self_image.emotional_tone.value, 0.5)

        # 4. 同一性揺らぎ認知の揺らぎ状態
        ic_active_shifts = 0
        ic_level = 0.0
        if self._last_coherence is not None:
            coherence_map = {"stable": 0.0, "slightly_shifting": 0.3, "unsettled": 0.6, "disconnected": 1.0}
            ic_level = coherence_map.get(self._last_coherence.level.value, 0.0)
            if hasattr(self._last_coherence, "shift_overlap") and self._last_coherence.shift_overlap is not None:
                ic_active_shifts = self._last_coherence.shift_overlap.active_count

        # 5. 時間的自己差分の差分規模
        td_magnitude = 0.0
        if self._last_diff_summary is not None:
            magnitude_map = {"negligible": 0.0, "noticeable": 0.3, "significant": 0.6, "substantial": 1.0}
            td_magnitude = magnitude_map.get(self._last_diff_summary.magnitude.value, 0.0)

        # 6. 連続性負荷の負荷水準
        cs_level = 0.0
        if self._last_strain is not None:
            strain_map = {"at_ease": 0.0, "unsettled": 0.3, "dissonant": 0.6, "alienated": 1.0}
            cs_level = strain_map.get(self._last_strain.level.value, 0.0)

        # 7. 内省断面横断記述のスナップショット（6断面の数値化）
        cross_section_values: dict[str, float] = {}
        if self._introspection_cross_section is not None:
            latest_snap = self._introspection_cross_section.get_latest_snapshot()
            if latest_snap and "sections" in latest_snap:
                # 断面テキストの長さから相対的な活性度を推定
                sections = latest_snap["sections"]
                if sections:
                    max_len = max(len(str(v)) for v in sections.values()) if sections.values() else 1
                    if max_len > 0:
                        for key, val in sections.items():
                            cross_section_values[key] = min(len(str(val)) / max(max_len, 1), 1.0)

        # 8. 安定化の構造的記述
        stab_signal_count = 0
        stab_diff_degree = 0.0
        if self._stabilization_desc_state is not None:
            try:
                latest_rec = None
                if hasattr(self._stabilization_desc_state, "record_window") and self._stabilization_desc_state.record_window:
                    latest_rec = self._stabilization_desc_state.record_window[-1]
                if latest_rec is not None:
                    stab_signal_count = getattr(latest_rec, "active_signal_count", 0)
                    stab_diff_degree = getattr(latest_rec, "diff_degree", 0.0)
            except Exception:
                pass

        return ContradictionInputs(
            self_model_emotion_intensity=sm_intensity,
            self_model_emotion_spread=sm_spread,
            self_model_emotion_conflict=sm_conflict,
            meta_emotion_change_speed=me_change_speed,
            meta_emotion_dominant_stability=me_dominant_stability,
            self_image_stability=si_stability,
            self_image_continuity=si_continuity,
            self_image_emotional_tone=si_emotional_tone,
            identity_coherence_active_shifts=ic_active_shifts,
            identity_coherence_level=ic_level,
            temporal_diff_magnitude=td_magnitude,
            continuity_strain_level=cs_level,
            cross_section_values=cross_section_values,
            stabilization_signal_count=stab_signal_count,
            stabilization_diff_degree=stab_diff_degree,
            current_tick=self._tick_count,
        )

    def _build_meta_emotion_inputs(self) -> MetaEmotionInputs:
        """メタ感情認知に必要な8断面の入力を構築する。

        すべてREAD-ONLY参照。感情処理パイプラインのパラメータを変更しない。
        """
        # 1. 感情状態断面（感情ベクトル、ムード — READ-ONLY）
        emotion_values = self._psyche.emotions.as_dict()
        mood_valence = self._psyche.mood.valence
        mood_arousal = self._psyche.mood.arousal

        # 2. ダイナミクス相断面（READ-ONLY）
        dynamics_phase = self._dynamics.phase.value if self._dynamics else "normal"
        dynamics_peak_intensity = self._dynamics.peak_intensity if self._dynamics else 0.0
        dynamics_accumulated_intensity = self._dynamics.accumulated_intensity if self._dynamics else 0.0

        # 3. STM-感情連動結果断面（READ-ONLY）
        coupling_continuity = 0.0
        coupling_active_entries = 0
        if self._last_coupling is not None:
            coupling_continuity = self._last_coupling.context_continuity
            coupling_active_entries = self._last_coupling.active_entry_count

        # 4. 自己モデル感情記述断面（READ-ONLY）
        # EmotionalSpread/EmotionalIntensity are enums; convert to numeric
        self_model_spread = 0.0
        self_model_intensity = 0.0
        self_model_conflict = False
        if self._last_self_view is not None:
            emo_view = self._last_self_view.emotional
            # spread: FOCUSED=0.2, MIXED=0.5, DIFFUSE=0.8, UNDEFINED=0.0
            spread_map = {"focused": 0.2, "mixed": 0.5, "diffuse": 0.8, "undefined": 0.0}
            self_model_spread = spread_map.get(emo_view.spread.value, 0.0)
            # intensity: CALM=0.15, MODERATE=0.5, INTENSE=0.85, OVERWHELMING=1.0, UNDEFINED=0.0
            intensity_map = {"calm": 0.15, "moderate": 0.5, "intense": 0.85, "overwhelming": 1.0, "undefined": 0.0}
            self_model_intensity = intensity_map.get(emo_view.intensity.value, 0.0)
            self_model_conflict = emo_view.has_coexisting_pairs

        # 5. 振幅状態断面（READ-ONLY）
        amplitude_value = self._amplitude_state.current_amplitude if self._amplitude_state else 1.0
        amplitude_boost = self._amplitude_state.accumulated_boost if self._amplitude_state else 0.0

        # 6. 対話文脈断面
        context_summary = ""
        dialogue_state = ""
        if self._last_percept is not None:
            context_summary = getattr(self._last_percept, "text", "")[:100]
            dialogue_state = getattr(self._last_percept, "intent", "unknown") or "unknown"

        # 7. 記憶参照断面
        referenced_memory_count = 0
        if self._last_recalled_memories is not None:
            referenced_memory_count = len(self._last_recalled_memories)

        # 8. 蓄積鮮度断面
        me_state = self._meta_emotion_processor.state
        existing_record_count = len(me_state.cognition_history)
        average_freshness = 0.0
        if me_state.cognition_history:
            average_freshness = sum(
                r.freshness for r in me_state.cognition_history
            ) / len(me_state.cognition_history)

        return MetaEmotionInputs(
            emotion_values=emotion_values,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
            dynamics_phase=dynamics_phase,
            dynamics_peak_intensity=dynamics_peak_intensity,
            dynamics_accumulated_intensity=dynamics_accumulated_intensity,
            coupling_continuity=coupling_continuity,
            coupling_active_entries=coupling_active_entries,
            self_model_spread=self_model_spread,
            self_model_intensity=self_model_intensity,
            self_model_conflict=self_model_conflict,
            amplitude_value=amplitude_value,
            amplitude_boost=amplitude_boost,
            context_summary=context_summary,
            dialogue_state=dialogue_state,
            referenced_memory_count=referenced_memory_count,
            existing_record_count=existing_record_count,
            average_freshness=average_freshness,
            current_tick=self._tick_count,
        )

    def _build_backdrop_inputs(self) -> BackdropInputs:
        """感情基調認知に必要な8断面の入力を構築する。

        すべてREAD-ONLY参照。感情処理パイプラインのパラメータを変更しない。
        """
        # 1. 感情状態断面（感情ベクトル — READ-ONLY）
        emotion_values = self._psyche.emotions.as_dict()

        # 2. ムード断面（READ-ONLY）
        mood_valence = self._psyche.mood.valence
        mood_arousal = self._psyche.mood.arousal

        # 3. ダイナミクス相断面（READ-ONLY）
        dynamics_phase = self._dynamics.phase.value if self._dynamics else "normal"

        # 4. 振幅断面（READ-ONLY）
        amplitude_value = self._amplitude_state.current_amplitude if self._amplitude_state else 1.0

        # 5. メタ感情認知断面（推移パターン特徴量のみ — 持続パターン検出結果・変動候補は参照しない）
        meta_change_speed = 0.0
        meta_dominant_stability = 0.0
        if self._last_meta_emotion is not None:
            try:
                # TransitionFeature の直近値を参照
                me_state = self._meta_emotion_processor.state
                if me_state.cognition_history:
                    last_rec = me_state.cognition_history[-1]
                    if last_rec.transition_features:
                        last_tf = last_rec.transition_features[-1]
                        meta_change_speed = last_tf.change_speed
                        meta_dominant_stability = last_tf.dominant_stability
            except Exception:
                pass

        # 6. 蓄積鮮度断面（自己参照）
        bd_state = self._emotional_backdrop_processor.state
        existing_record_count = len(bd_state.composition_records)
        average_freshness = 0.0
        if bd_state.composition_records:
            average_freshness = sum(
                r.freshness for r in bd_state.composition_records
            ) / len(bd_state.composition_records)

        # 7. 対話経過断面
        dialogue_elapsed_ticks = self._tick_count

        # 8. 時間認知断面（利用可能な場合のみ）
        temporal_elapsed_description = ""
        try:
            if self._temporal_cognition is not None:
                tc_state = self._temporal_cognition.state
                if tc_state.elapsed_records:
                    last_elapsed = tc_state.elapsed_records[-1]
                    temporal_elapsed_description = last_elapsed.description or ""
            # 空の場合は空文字のまま
        except Exception:
            pass

        return BackdropInputs(
            emotion_values=emotion_values,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
            dynamics_phase=dynamics_phase,
            amplitude_value=amplitude_value,
            meta_emotion_change_speed=meta_change_speed,
            meta_emotion_dominant_stability=meta_dominant_stability,
            existing_record_count=existing_record_count,
            average_freshness=average_freshness,
            dialogue_elapsed_ticks=dialogue_elapsed_ticks,
            temporal_elapsed_description=temporal_elapsed_description,
            current_tick=self._tick_count,
        )

    def _build_drive_variation_inputs(self) -> DriveVariationInputs:
        """駆動の変動記述に必要な8断面の入力を構築する。

        すべてREAD-ONLY参照。駆動値・反応処理パラメータ・動機生成・
        ポリシー候補・感情パイプラインのパラメータを変更しない。
        """
        # 1. 駆動状態断面（駆動ベクトル全次元の値 — READ-ONLY）
        drive_values = self._psyche.drives.as_dict()

        # 2. 感情基調認知断面（窓サイズと低変動性警告のみ — READ-ONLY）
        backdrop_window_size = 0
        backdrop_low_variability = False
        if self._last_backdrop_result is not None:
            backdrop_window_size = self._last_backdrop_result.window_size
            backdrop_low_variability = self._last_backdrop_result.low_variability_warning

        # 3. メタ感情認知断面（推移特徴量の変化速度と支配安定度のみ — READ-ONLY）
        meta_change_speed = 0.0
        meta_dominant_stability = 0.0
        if self._last_meta_emotion is not None:
            try:
                me_state = self._meta_emotion_processor.state
                if me_state.cognition_history:
                    last_rec = me_state.cognition_history[-1]
                    if last_rec.transition_features:
                        last_tf = last_rec.transition_features[-1]
                        meta_change_speed = last_tf.change_speed
                        meta_dominant_stability = last_tf.dominant_stability
            except Exception:
                pass

        # 4. 蓄積鮮度断面（自己参照）
        dv_state = self._drive_variation_processor.state
        existing_record_count = len(dv_state.composition_records)
        average_freshness = 0.0
        if dv_state.composition_records:
            average_freshness = sum(
                r.freshness for r in dv_state.composition_records
            ) / len(dv_state.composition_records)

        # 5. 対話経過断面
        dialogue_elapsed_ticks = self._tick_count

        # 6. 時間認知断面（利用可能な場合のみ）
        temporal_elapsed_description = ""
        try:
            if self._temporal_cognition is not None:
                tc_state = self._temporal_cognition.state
                if tc_state.elapsed_records:
                    last_elapsed = tc_state.elapsed_records[-1]
                    temporal_elapsed_description = last_elapsed.description or ""
        except Exception:
            pass

        # 7. ムード断面（READ-ONLY）
        mood_valence = self._psyche.mood.valence
        mood_arousal = self._psyche.mood.arousal

        # 8. 反応結果断面（直近の反応処理が駆動に対して行った更新の有無）
        # 反応処理はpost_response_update内で毎回実行されるため、
        # 3ティック周期帯では前回の反応が存在する
        reaction_updated_drives = True  # 反応処理が実行されたティックかどうか

        return DriveVariationInputs(
            drive_values=drive_values,
            backdrop_window_size=backdrop_window_size,
            backdrop_low_variability=backdrop_low_variability,
            meta_emotion_change_speed=meta_change_speed,
            meta_emotion_dominant_stability=meta_dominant_stability,
            existing_record_count=existing_record_count,
            average_freshness=average_freshness,
            dialogue_elapsed_ticks=dialogue_elapsed_ticks,
            temporal_elapsed_description=temporal_elapsed_description,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
            reaction_updated_drives=reaction_updated_drives,
            current_tick=self._tick_count,
        )

    def select_policy_dict(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        user_id: str = "viewer",
    ) -> dict[str, Any]:
        """最終選択されたポリシーをdictで返す（expression.py用）。"""
        candidates, _ = self._generate_final_candidates(percept, recalled_memories, user_id)
        resp_influence = self._responsibility_mgr.get_influence(user_id)
        policy = select_policy(candidates, self._psyche, resp_influence)

        # 選択結果を価値軸にフィードバック（超高慣性の微小更新）
        try:
            policy_label = policy.get("policy_label", "")
            if policy_label:
                self._value_orientation = update_from_decision(
                    orientation=self._value_orientation,
                    policy_label=policy_label,
                )
        except Exception:
            pass

        # 選択されたポリシーを記録（Phase 7a で構成バッファに記録するため）
        self._last_selected_policy_label = policy.get("policy_label", "")
        self._last_selected_policy_axis = policy.get("_axis", "")

        # 選択帰属: 選択事実を記録（select_policy_dict直後）
        try:
            candidate_labels = [
                c.get("policy_label", "") for c in candidates
            ]
            # バイアス源の名前リストを収集（スコア・重み・方向性は記録しない）
            bias_labels: list[str] = []
            if self._last_decision_bias is not None:
                bias_labels.append("decision_bias")
            if self._last_sensitivity_bias is not None:
                bias_labels.append("context_sensitivity")
            try:
                if self._stability_valve is not None:
                    bias_labels.append("stability_valve")
            except Exception:
                pass
            if self._value_orientation is not None:
                bias_labels.append("value_orientation")
            if self._persistent_commitment is not None:
                bias_labels.append("persistent_commitment")
            # scoring_fluctuation は常に適用される
            bias_labels.append("scoring_fluctuation")
            if self._last_has_silence:
                bias_labels.append("silence_hesitation")

            self._selection_attribution_recorder.record_selection(
                selected_policy_label=self._last_selected_policy_label,
                candidate_labels=candidate_labels,
                tick=self._tick_count,
                bias_source_labels=bias_labels,
            )
        except Exception:
            pass

        return policy

    def get_policy_suggestions(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        user_id: str = "viewer",
    ) -> str:
        """行動方針候補を生成し、プロンプト用テキストとして返す。

        Phase 30-35 に相当する処理を一括実行。

        Args:
            percept: 知覚入力
            recalled_memories: 記憶検索結果
            user_id: 対話相手ID

        Returns:
            【行動方針候補】セクションのテキスト
        """
        candidates, tone_mod = self._generate_final_candidates(
            percept, recalled_memories, user_id,
        )

        # Format output
        if not candidates:
            return ""

        lines = ["【行動方針候補】"]
        for i, c in enumerate(candidates[:3], 1):
            label = c.get("policy_label", "?")
            score = c.get("_score", 0.0)
            tone_str = ""
            if tone_mod and tone_mod.recommended:
                tone_str = f" [{tone_mod.recommended.value}]"
            lines.append(f"{i}. {label} (score={score:.2f}){tone_str}")

        lines.append("")
        lines.append(
            "この内面状態と行動方針候補を参考に、自然な反応をしてください。"
        )
        lines.append("候補に従う義務はありません。あくまで参考です。")
        return "\n".join(lines)

    # ── Fear recompute ────────────────────────────────────────────

    def _recompute_fear(self) -> None:
        """4柱からfear_indexを再計算。"""
        p = self._psyche
        fear = compute_fear_index(
            identity_risk=(
                identity_manager.calc_identity_risk(p.identity)
                if p.identity else 0.0
            ),
            attachment_risk=(
                attachment_manager.calc_attachment_risk(p.attachment)
                if p.attachment else 0.7
            ),
            continuity_risk=continuity_manager.calc_continuity_risk(
                memory_count=p.continuity.memory_count if p.continuity else 0,
            ),
            projection_risk=(
                projection_manager.calc_projection_risk(p.projection)
                if p.projection else 0.7
            ),
        )
        self._psyche = self._psyche.model_copy(update={"fear_index": fear})

    # ── Memory saved callback ─────────────────────────────────────

    def on_memory_saved(
        self,
        summary: str,
        keywords: list[str],
        memory_count: int,
    ) -> None:
        """記憶保存時のコールバック。brain.py の summarize_and_save から呼ばれる。

        Args:
            summary: 保存された要約テキスト
            keywords: キーワード
            memory_count: 保存後の全記憶件数
        """
        if self._psyche.continuity is not None:
            self._psyche = self._psyche.model_copy(update={
                "continuity": self._psyche.continuity.model_copy(
                    update={"memory_count": memory_count},
                ),
            })
            self._recompute_fear()
            logger.debug(
                "Memory saved callback: count=%d, fear=%.2f",
                memory_count, self._psyche.fear_level,
            )

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """全状態を永続化する。

        Args:
            path: 保存先パス（デフォルト: data/psyche_snapshot.json）
        """
        save_path = path or (self._data_dir / "psyche_snapshot.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 42,
            "tick_count": self._tick_count,
            "psyche": self._psyche.to_dict(),
            "loop_state": self._loop_state.to_dict() if self._loop_state else {},
            "dynamics": self._dynamics.to_dict() if self._dynamics else {},
            "amplitude": self._amplitude_state.to_dict() if self._amplitude_state else {},
            "value_orientation": self._value_orientation.to_dict() if self._value_orientation else {},
            "self_ref_state": self._self_ref_state.to_dict() if self._self_ref_state else {},
            "last_self_view": self._last_self_view.to_dict() if self._last_self_view else {},
            "tendency_awareness": self._tendency_awareness.to_dict() if self._tendency_awareness else {},
            "last_diff_summary": self._last_diff_summary.to_dict() if self._last_diff_summary else {},
            "last_strain": self._last_strain.to_dict() if self._last_strain else {},
            "last_self_image": self._last_self_image.to_dict() if self._last_self_image else {},
            "last_coherence": self._last_coherence.to_dict() if self._last_coherence else {},
            "last_narrative": self._last_narrative.to_dict() if self._last_narrative else {},
            "last_episodes": self._last_episodes.to_dict() if self._last_episodes else {},
            "last_bindings": self._last_bindings.to_dict() if self._last_bindings else {},
            "last_trace": self._last_trace.to_dict() if self._last_trace else {},
            "last_consumption": self._last_consumption.to_dict() if self._last_consumption else {},
            "last_expectations": self._last_expectations.to_dict() if self._last_expectations else {},
            "last_motives": self._last_motives.to_dict() if self._last_motives else {},
            "last_other_model": self._last_other_model.to_dict() if self._last_other_model else {},
            "input_supply": self._input_supply.to_dict() if self._input_supply else {},
            # Version 5 fields
            "tendency_state": self._tendency_sys.state.to_dict(),
            "vector_state": self._vector_gen.state.to_dict(),
            "candidate_state": self._candidate_gen.state.to_dict(),
            "transient_goal_state": self._transient_goal_mgr.state.to_dict(),
            "stability_valve": self._stability_valve.to_dict(),
            # Version 6 fields
            "dispersion_state": dispersion_to_dict(self._dispersion_state),
            "context_sensitivity_state": self._ctx_state.to_dict(),
            "last_coupling": self._last_coupling.to_dict() if self._last_coupling else {},
            # Version 7 fields
            "policy_expansion_state": self._policy_expander.state.to_dict() if self._policy_expander else {},
            # Version 8 fields
            "memory_integration_state": self._memory_integrator.state.to_dict() if self._memory_integrator else {},
            # Version 9 fields
            "real_feed_state": self._real_feed_processor.state.to_dict() if self._real_feed_processor else {},
            # Version 10 fields
            "text_dialogue_state": self._text_dialogue_processor.state.to_dict() if self._text_dialogue_processor else {},
            # Version 11 fields
            "spontaneous_state": self._spontaneous_processor.state.to_dict() if self._spontaneous_processor else {},
            # Version 12 fields
            "vo_validation_state": self._vo_validator.state.to_dict() if self._vo_validator else {},
            # Version 13 fields
            "forgetting_fixation_state": self._forgetting_fixation_processor.state.to_dict() if self._forgetting_fixation_processor else {},
            # Version 14 fields
            "action_result_state": self._action_result_observer.state.to_dict() if self._action_result_observer else {},
            # Version 15 fields
            "dialogue_learning_state": self._dialogue_learning_processor.state.to_dict() if self._dialogue_learning_processor else {},
            # Version 16 fields
            "meta_emotion_state": self._meta_emotion_processor.state.to_dict() if self._meta_emotion_processor else {},
            # Version 17 fields
            "self_action_perception_state": self._self_action_recorder.state.to_dict() if self._self_action_recorder else {},
            # Version 18 fields
            "expectation_action_diff_log": self._expectation_action_diff_log,
            # Version 19 fields
            "intent_action_gap_state": self._intent_action_gap_recorder.state.to_dict() if self._intent_action_gap_recorder else {},
            # Version 20 fields
            "temporal_cognition_state": self._temporal_cognition.state.to_dict() if self._temporal_cognition else {},
            # Version 21 fields
            "multi_path_recall_state": self._multi_path_recall.state.to_dict() if self._multi_path_recall else {},
            # Version 22 fields
            "introspection_cross_section_state": self._introspection_cross_section.save() if self._introspection_cross_section else {},
            "perceptual_context_state": self._perceptual_context.save() if self._perceptual_context else {},
            # Version 23 fields
            "selection_attribution_state": self._selection_attribution_recorder.state.to_dict() if self._selection_attribution_recorder else {},
            # Version 24 fields
            "reference_frequency_state": self._reference_frequency_state.to_dict() if self._reference_frequency_state else {},
            # Version 25 fields
            "persistent_commitment_state": self._persistent_commitment.state.to_dict() if self._persistent_commitment else {},
            # Version 26 fields
            "stabilization_description_state": self._stabilization_desc_state.to_dict() if self._stabilization_desc_state else {},
            # Version 27 fields
            "behavioral_diversity_state": self._behavioral_diversity_state.to_dict() if self._behavioral_diversity_state else {},
            # Version 28 fields
            "spontaneous_recall_state": self._spontaneous_recall.state.to_dict() if self._spontaneous_recall else {},
            # Version 29 fields
            "internal_contradiction_state": self._contradiction_processor.save() if self._contradiction_processor else {},
            # Version 30 fields
            "interaction_accumulation_state": self._interaction_accumulation.state.to_dict() if self._interaction_accumulation else {},
            # Version 31 fields
            "emotional_backdrop_state": self._emotional_backdrop_processor.save() if self._emotional_backdrop_processor else {},
            # Version 32 fields
            "situational_self_presentation_state": self._situational_self_presentation.state.to_dict() if self._situational_self_presentation else {},
            # Version 33 fields
            "drive_variation_state": self._drive_variation_processor.save() if self._drive_variation_processor else {},
            # Version 34 fields
            "expectation_lifecycle_state": self._expectation_lifecycle_processor.state.to_dict() if self._expectation_lifecycle_processor else {},
            # Version 35 fields
            "input_pathway_balance_state": save_input_pathway_balance_state(self._input_pathway_balance_state) if self._input_pathway_balance_state else {},
            # Version 36 fields
            "responsibility_temporal_trace_state": self._responsibility_temporal_trace.save() if self._responsibility_temporal_trace else {},
            # Version 37 fields
            "emotion_cooccurrence_state": self._emotion_cooccurrence_processor.save() if self._emotion_cooccurrence_processor else {},
            # Version 38 fields
            "other_boundary_accumulation_state": self._other_boundary_accumulation.state.to_dict() if self._other_boundary_accumulation else {},
            # Version 39 fields
            "forgetting_recall_balance_state": save_frb_state(self._frb_state) if self._frb_state else {},
            # Version 40 fields
            "attention_distribution_state": save_att_dist_state(self._att_dist_state) if self._att_dist_state else {},
            # Version 41 fields
            "goal_hierarchy_propagation_state": self._goal_hierarchy_propagation.state.to_dict() if self._goal_hierarchy_propagation else {},
            # Version 42 fields
            "hypothesis_observation_pairing_state": self._hypothesis_observation_pairing.state.to_dict() if self._hypothesis_observation_pairing else {},
        }

        save_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Psyche state saved to %s", save_path)

    def load(self, path: Optional[Path] = None) -> bool:
        """永続化された状態を復元する。

        Args:
            path: 読込先パス

        Returns:
            True if loaded successfully, False otherwise.
        """
        load_path = path or (self._data_dir / "psyche_snapshot.json")
        if not load_path.exists():
            logger.info("No snapshot found at %s", load_path)
            return False

        try:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            if "psyche" in data:
                self._psyche = PsycheState.from_dict(data["psyche"])
            if "loop_state" in data:
                self._loop_state = LoopState.from_dict(data["loop_state"])
            if "dynamics" in data and data["dynamics"]:
                self._dynamics = DynamicsState.from_dict(data["dynamics"])
            if "tick_count" in data:
                self._tick_count = data["tick_count"]

            # Version 4+ fields
            if data.get("amplitude"):
                self._amplitude_state = AmplitudeState.from_dict(data["amplitude"])
            if data.get("value_orientation"):
                self._value_orientation = ValueOrientation.from_dict(data["value_orientation"])
            if data.get("self_ref_state"):
                self._self_ref_state = SelfReferenceState.from_dict(data["self_ref_state"])
            if data.get("last_self_view"):
                self._last_self_view = SelfStateView.from_dict(data["last_self_view"])
            if data.get("tendency_awareness"):
                self._tendency_awareness = TendencyAwareness.from_dict(data["tendency_awareness"])
            if data.get("last_diff_summary"):
                self._last_diff_summary = SelfDifferenceSummary.from_dict(data["last_diff_summary"])
            if data.get("last_strain"):
                self._last_strain = StrainState.from_dict(data["last_strain"])
            if data.get("last_self_image"):
                self._last_self_image = ProvisionalSelfImage.from_dict(data["last_self_image"])
            if data.get("last_coherence"):
                self._last_coherence = IdentityCoherenceState.from_dict(data["last_coherence"])
            if data.get("last_narrative"):
                self._last_narrative = NarrativeState.from_dict(data["last_narrative"])
            if data.get("last_episodes"):
                self._last_episodes = EpisodeStore.from_dict(data["last_episodes"])
            if data.get("last_bindings"):
                self._last_bindings = BindingStore.from_dict(data["last_bindings"])
            if data.get("last_trace"):
                self._last_trace = TraceLog.from_dict(data["last_trace"])
            if data.get("last_consumption"):
                self._last_consumption = ConsumptionStore.from_dict(data["last_consumption"])
            if data.get("last_expectations"):
                self._last_expectations = ExpectationStore.from_dict(data["last_expectations"])
            if data.get("last_motives"):
                self._last_motives = MotiveStore.from_dict(data["last_motives"])
            if data.get("last_other_model"):
                self._last_other_model = OtherModelStore.from_dict(data["last_other_model"])
            if data.get("input_supply"):
                self._input_supply = InputSupplyState.from_dict(data["input_supply"])

            # Version 5+ fields
            if data.get("tendency_state"):
                self._tendency_sys._state = RepeatedTendencyState.from_dict(data["tendency_state"])
            if data.get("vector_state"):
                self._vector_gen._state = VectorState.from_dict(data["vector_state"])
            if data.get("candidate_state"):
                self._candidate_gen._state = CandidateState.from_dict(data["candidate_state"])
            if data.get("transient_goal_state"):
                self._transient_goal_mgr._state = TransientGoalState.from_dict(data["transient_goal_state"])
            if data.get("stability_valve"):
                self._stability_valve = StabilityValve.from_dict(data["stability_valve"])

            # Version 6+ fields
            if data.get("dispersion_state"):
                self._dispersion_state = dispersion_from_dict(data["dispersion_state"])
            if data.get("context_sensitivity_state"):
                self._ctx_state = ContextState.from_dict(data["context_sensitivity_state"])
            if data.get("last_coupling"):
                self._last_coupling = CouplingInfluence.from_dict(data["last_coupling"])

            # Version 7+ fields
            if data.get("policy_expansion_state"):
                self._policy_expander._state = ExpansionState.from_dict(data["policy_expansion_state"])

            # Version 8+ fields
            if data.get("memory_integration_state"):
                self._memory_integrator._state = IntegrationState.from_dict(data["memory_integration_state"])

            # Version 9+ fields
            if data.get("real_feed_state"):
                self._real_feed_processor._state = RealFeedState.from_dict(data["real_feed_state"])

            # Version 10+ fields
            if data.get("text_dialogue_state"):
                self._text_dialogue_processor._state = TextDialogueState.from_dict(data["text_dialogue_state"])

            # Version 11+ fields
            if data.get("spontaneous_state"):
                self._spontaneous_processor._state = SpontaneousState.from_dict(data["spontaneous_state"])

            # Version 12+ fields
            if data.get("vo_validation_state"):
                self._vo_validator._state = VOValidationState.from_dict(data["vo_validation_state"])

            # Version 13+ fields
            if data.get("forgetting_fixation_state"):
                self._forgetting_fixation_processor._state = ForgettingFixationState.from_dict(data["forgetting_fixation_state"])

            # Version 14+ fields
            if data.get("action_result_state"):
                self._action_result_observer._state = ActionResultObservationState.from_dict(data["action_result_state"])

            # Version 15+ fields
            if data.get("dialogue_learning_state"):
                self._dialogue_learning_processor._state = DialogueLearningState.from_dict(data["dialogue_learning_state"])

            # Version 16+ fields
            if data.get("meta_emotion_state"):
                self._meta_emotion_processor.state = MetaEmotionState.from_dict(data["meta_emotion_state"])
                self._meta_emotion_processor.state.apply_session_decay()

            # Version 17+ fields
            if data.get("self_action_perception_state"):
                self._self_action_recorder.state = SelfActionPerceptionState.from_dict(data["self_action_perception_state"])

            # Version 18+ fields
            if data.get("expectation_action_diff_log"):
                self._expectation_action_diff_log = data["expectation_action_diff_log"]

            # Version 19+ fields
            if data.get("intent_action_gap_state"):
                self._intent_action_gap_recorder.state = IntentActionGapState.from_dict(data["intent_action_gap_state"])

            # Version 20+ fields
            if data.get("temporal_cognition_state"):
                self._temporal_cognition.state = TemporalCognitionState.from_dict(data["temporal_cognition_state"])

            # Version 21+ fields
            if data.get("multi_path_recall_state"):
                self._multi_path_recall.state = MultiPathRecallState.from_dict(data["multi_path_recall_state"])

            # Version 22+ fields
            if data.get("introspection_cross_section_state"):
                self._introspection_cross_section.load(data["introspection_cross_section_state"])
            if data.get("perceptual_context_state"):
                self._perceptual_context.load(data["perceptual_context_state"])

            # Version 23+ fields
            if data.get("selection_attribution_state"):
                self._selection_attribution_recorder.state = SelectionAttributionState.from_dict(data["selection_attribution_state"])

            # Version 24+ fields
            if data.get("reference_frequency_state"):
                self._reference_frequency_state = ReferenceFrequencyState.from_dict(data["reference_frequency_state"])

            # Version 25+ fields
            if data.get("persistent_commitment_state"):
                self._persistent_commitment._state = PersistentCommitmentState.from_dict(data["persistent_commitment_state"])
                self._persistent_commitment.validate_on_load()

            # Version 26+ fields
            if data.get("stabilization_description_state"):
                self._stabilization_desc_state = StabilizationDescriptionState.from_dict(data["stabilization_description_state"])

            # Version 27+ fields
            if data.get("behavioral_diversity_state"):
                self._behavioral_diversity_state = BehavioralDiversityState.from_dict(data["behavioral_diversity_state"])

            # Version 28+ fields
            if data.get("spontaneous_recall_state"):
                self._spontaneous_recall.state = SpontaneousRecallState.from_dict(data["spontaneous_recall_state"])

            # Version 29+ fields
            if data.get("internal_contradiction_state"):
                self._contradiction_processor.load(data["internal_contradiction_state"])

            # Version 30+ fields
            if data.get("interaction_accumulation_state"):
                self._interaction_accumulation.state = InteractionAccumulationState.from_dict(data["interaction_accumulation_state"])

            # Version 31+ fields
            if data.get("emotional_backdrop_state"):
                self._emotional_backdrop_processor.load(data["emotional_backdrop_state"])
                self._emotional_backdrop_processor.state.apply_session_decay()

            # Version 32+ fields
            if data.get("situational_self_presentation_state"):
                self._situational_self_presentation.state = SituationalSelfPresentationState.from_dict(data["situational_self_presentation_state"])
                self._situational_self_presentation.state.apply_session_decay()

            # Version 33+ fields
            if data.get("drive_variation_state"):
                self._drive_variation_processor.load(data["drive_variation_state"])
                self._drive_variation_processor.state.apply_session_decay()

            # Version 34+ fields
            if data.get("expectation_lifecycle_state"):
                self._expectation_lifecycle_processor.state = ExpectationLifecycleState.from_dict(data["expectation_lifecycle_state"])

            # Version 35+ fields
            if data.get("input_pathway_balance_state"):
                self._input_pathway_balance_state = load_input_pathway_balance_state(data["input_pathway_balance_state"])

            # Version 36+ fields
            if data.get("responsibility_temporal_trace_state"):
                self._responsibility_temporal_trace.load(data["responsibility_temporal_trace_state"])

            # Version 37+ fields
            if data.get("emotion_cooccurrence_state"):
                self._emotion_cooccurrence_processor.load(data["emotion_cooccurrence_state"])
                self._emotion_cooccurrence_processor.state.apply_session_decay()

            # Version 38+ fields
            if data.get("other_boundary_accumulation_state"):
                self._other_boundary_accumulation._state = OtherBoundaryAccumulationState.from_dict(data["other_boundary_accumulation_state"])
                self._other_boundary_accumulation.state.apply_session_decay()

            # Version 39+ fields
            if data.get("forgetting_recall_balance_state"):
                self._frb_state = load_frb_state(data["forgetting_recall_balance_state"])

            # Version 40+ fields
            if data.get("attention_distribution_state"):
                self._att_dist_state = load_att_dist_state(data["attention_distribution_state"])

            # Version 41+ fields
            if data.get("goal_hierarchy_propagation_state"):
                self._goal_hierarchy_propagation.state = GoalHierarchyPropagationState.from_dict(data["goal_hierarchy_propagation_state"])

            # Version 42+ fields
            if data.get("hypothesis_observation_pairing_state"):
                self._hypothesis_observation_pairing._state = HOPairingState.from_dict(data["hypothesis_observation_pairing_state"])

            logger.info("Psyche state loaded from %s (v%d, tick=%d)",
                        load_path, data.get("version", 0), self._tick_count)
            return True
        except Exception as e:
            logger.error("Failed to load snapshot: %s", e)
            return False
