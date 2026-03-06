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
import os
import time
from pathlib import Path
from typing import Any, Optional

from .state import PsycheState, Percept, EmotionVector, DriveVector
from .pillars import (
    AttachmentState,
    ContinuityState,
    FearIndex,
    IdentityState,
    ProjectionState,
)

# Core reaction pipeline
from .reaction import MoodContextInputs
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

# Memory emotion return (記憶想起から感情への帰還経路)
from .memory_emotion_return import (
    MemoryEmotionReturnProcessor,
    MemoryEmotionReturnState,
    create_memory_emotion_return,
)

# Other hypothesis emotion return (他者仮説由来の感情帰還経路)
from .other_hypothesis_emotion_return import (
    OtherHypothesisEmotionReturnProcessor,
    OtherHypothesisEmotionReturnState,
    create_other_hypothesis_emotion_return,
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
    get_latest_snapshot as get_ref_freq_latest_snapshot,
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
    normalize_section_items,
    prepend_freshness_annotation,
    ORIGINAL_FOOTER,
)

# Phase execution engine (段階3: 帯域横断の宣言的実行エンジン)
# Band列挙値はエンジンモジュール経由で取得
from .phase_execution_engine import PhaseExecutionEngine, Band

# 1-tick band phases (物理的分離: Phase 1-7)
from .orchestrator_1tick_phases import run_every_tick as _run_1tick_phases
from .orchestrator_1tick_phases import apply_return_aggregate_cap as _apply_return_aggregate_cap

# 5-tick band phases (物理的分離: Phase 15-26)
from .orchestrator_5tick_phases import run_every_5_ticks as _run_5tick_phases

# Enrichment generation (物理的分離: enrichment生成ロジック)
from .orchestrator_enrichment import get_prompt_enrichment as _get_prompt_enrichment

# Persistence helpers (save/load構造圧縮)
from .persistence_helpers import (
    FieldDef,
    SaveInterface,
    LoadInterface,
    SemanticGroup,
    CURRENT_VERSION,
    save_fields,
    load_fields,
    get_fields_by_group,
)

# Execution monitor (実運用向けログ・モニタリング基盤)
# READ-ONLY観測のみ。内部の判断・行動・選択に一切介入しない。
# 永続化対象外。save/loadフィールド追加なし。
from tools.execution_monitor import ExecutionMonitor, BandTimer, read_orchestrator_fields
from tools.policy_selection_log import PolicySelectionLog, create_policy_selection_log

# Return pathway monitor (帰還経路の動作検証と相互干渉検出)
# READ-ONLY観測のみ。内部の判断・行動・選択に一切介入しない。
# 永続化対象外。save/loadフィールド追加なし。enrichment非接続。
from tools.return_pathway_monitor import (
    ReturnPathwayMonitor,
    PATHWAY_A as _RPM_PATHWAY_A,
    PATHWAY_B as _RPM_PATHWAY_B,
    PATHWAY_C as _RPM_PATHWAY_C,
)

# Save/load warmup (復帰時キャッシュ再導出)
# 独自の内部状態を保持しない。永続化フィールド追加なし。
from .save_load_warmup import (
    execute_warmup,
    execute_session_recovery_check,
    compute_session_difference_scalar,
)

logger = logging.getLogger(__name__)


# ── フィールド定義リスト (save/load 構造圧縮) ─────────────────────
#
# 永続化対象の全フィールドを宣言的に定義する。
# 各エントリは保存・復元インターフェース種別、型、セマンティックグループを含む。
# FIELD_DEFINITIONS はクラス定義より前に置き、外部からも参照可能にする。
#
# 注: 外部関数参照 (save_func/load_func) はモジュールレベルのインポート済み関数を使用。
#     遅延参照が必要な型は None にし、_build_field_definitions() で構築する。

def _build_field_definitions() -> list[FieldDef]:
    """フィールド定義リストを構築する。

    型参照がモジュールレベルのインポートに依存するため、関数内で構築する。
    """
    SG = SemanticGroup
    SI = SaveInterface
    LI = LoadInterface

    return [
        # ── コア状態 (CORE) ──────────────────────────────────────
        FieldDef(
            key="psyche", attr_path="_psyche",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=PsycheState, version=1, group=SG.CORE,
            nullable_check=False,
        ),
        FieldDef(
            key="loop_state", attr_path="_loop_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=LoopState, version=1, group=SG.CORE,
        ),
        FieldDef(
            key="dynamics", attr_path="_dynamics",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=DynamicsState, version=1, group=SG.CORE,
        ),
        FieldDef(
            key="amplitude", attr_path="_amplitude_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=AmplitudeState, version=4, group=SG.CORE,
        ),
        FieldDef(
            key="value_orientation", attr_path="_value_orientation",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ValueOrientation, version=4, group=SG.CORE,
        ),
        FieldDef(
            key="tendency_state", attr_path="_tendency_sys",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=RepeatedTendencyState, version=5, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="vector_state", attr_path="_vector_gen",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=VectorState, version=5, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="candidate_state", attr_path="_candidate_gen",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=CandidateState, version=5, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="transient_goal_state", attr_path="_transient_goal_mgr",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=TransientGoalState, version=5, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="stability_valve", attr_path="_stability_valve",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=StabilityValve, version=5, group=SG.CORE,
        ),
        FieldDef(
            key="dispersion_state", attr_path="_dispersion_state",
            save_interface=SI.EXTERNAL_SAVE, load_interface=LI.EXTERNAL_LOAD,
            save_func=dispersion_to_dict, load_func=dispersion_from_dict,
            version=6, group=SG.CORE,
        ),
        FieldDef(
            key="context_sensitivity_state", attr_path="_ctx_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ContextState, version=6, group=SG.CORE,
        ),
        FieldDef(
            key="last_coupling", attr_path="_last_coupling",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=CouplingInfluence, version=6, group=SG.CORE,
        ),
        FieldDef(
            key="policy_expansion_state", attr_path="_policy_expander",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=ExpansionState, version=7, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="spontaneous_state", attr_path="_spontaneous_processor",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=SpontaneousState, version=11, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="vo_validation_state", attr_path="_vo_validator",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=VOValidationState, version=12, group=SG.CORE,
            state_sub_attr="state",
        ),
        FieldDef(
            key="persistent_commitment_state", attr_path="_persistent_commitment",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=PersistentCommitmentState, version=25, group=SG.CORE,
            state_sub_attr="state", validate_on_load=True,
        ),

        # ── 自己認識 (SELF_RECOGNITION) ──────────────────────────
        FieldDef(
            key="self_ref_state", attr_path="_self_ref_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=SelfReferenceState, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_self_view", attr_path="_last_self_view",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=SelfStateView, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="tendency_awareness", attr_path="_tendency_awareness",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=TendencyAwareness, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_diff_summary", attr_path="_last_diff_summary",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=SelfDifferenceSummary, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_strain", attr_path="_last_strain",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=StrainState, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_self_image", attr_path="_last_self_image",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ProvisionalSelfImage, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_coherence", attr_path="_last_coherence",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=IdentityCoherenceState, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_narrative", attr_path="_last_narrative",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=NarrativeState, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_trace", attr_path="_last_trace",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=TraceLog, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_consumption", attr_path="_last_consumption",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ConsumptionStore, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_expectations", attr_path="_last_expectations",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ExpectationStore, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="last_motives", attr_path="_last_motives",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=MotiveStore, version=4, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="self_action_perception_state", attr_path="_self_action_recorder",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=SelfActionPerceptionState, version=17, group=SG.SELF_RECOGNITION,
            state_sub_attr="state",
        ),
        FieldDef(
            key="intent_action_gap_state", attr_path="_intent_action_gap_recorder",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=IntentActionGapState, version=19, group=SG.SELF_RECOGNITION,
            state_sub_attr="state",
        ),
        FieldDef(
            key="introspection_cross_section_state", attr_path="_introspection_cross_section",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=22, group=SG.SELF_RECOGNITION,
        ),
        FieldDef(
            key="selection_attribution_state", attr_path="_selection_attribution_recorder",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=SelectionAttributionState, version=23, group=SG.SELF_RECOGNITION,
            state_sub_attr="state",
        ),

        # ── 記憶 (MEMORY) ────────────────────────────────────────
        FieldDef(
            key="last_episodes", attr_path="_last_episodes",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=EpisodeStore, version=4, group=SG.MEMORY,
        ),
        FieldDef(
            key="last_bindings", attr_path="_last_bindings",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=BindingStore, version=4, group=SG.MEMORY,
        ),
        FieldDef(
            key="memory_integration_state", attr_path="_memory_integrator",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=IntegrationState, version=8, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="forgetting_fixation_state", attr_path="_forgetting_fixation_processor",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=ForgettingFixationState, version=13, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="action_result_state", attr_path="_action_result_observer",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=ActionResultObservationState, version=14, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="expectation_action_diff_log", attr_path="_expectation_action_diff_log",
            save_interface=SI.RAW, load_interface=LI.RAW,
            version=18, group=SG.MEMORY,
            nullable_check=False,
        ),
        FieldDef(
            key="multi_path_recall_state", attr_path="_multi_path_recall",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=MultiPathRecallState, version=21, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="reference_frequency_state", attr_path="_reference_frequency_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=ReferenceFrequencyState, version=24, group=SG.MEMORY,
        ),
        FieldDef(
            key="spontaneous_recall_state", attr_path="_spontaneous_recall",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=SpontaneousRecallState, version=28, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="memory_emotion_return_state", attr_path="_memory_emotion_return",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=MemoryEmotionReturnState, version=43, group=SG.MEMORY,
            state_sub_attr="state",
        ),
        FieldDef(
            key="other_hypothesis_emotion_return_state",
            attr_path="_other_hypothesis_emotion_return",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=OtherHypothesisEmotionReturnState, version=44,
            group=SG.MEMORY,
            state_sub_attr="state",
        ),

        # ── 他者モデル (OTHER_MODEL) ─────────────────────────────
        FieldDef(
            key="last_other_model", attr_path="_last_other_model",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=OtherModelStore, version=4, group=SG.OTHER_MODEL,
        ),
        FieldDef(
            key="input_supply", attr_path="_input_supply",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=InputSupplyState, version=4, group=SG.OTHER_MODEL,
        ),
        FieldDef(
            key="real_feed_state", attr_path="_real_feed_processor",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=RealFeedState, version=9, group=SG.OTHER_MODEL,
            state_sub_attr="state",
        ),
        FieldDef(
            key="text_dialogue_state", attr_path="_text_dialogue_processor",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=TextDialogueState, version=10, group=SG.OTHER_MODEL,
            state_sub_attr="state",
        ),
        FieldDef(
            key="dialogue_learning_state", attr_path="_dialogue_learning_processor",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=DialogueLearningState, version=15, group=SG.OTHER_MODEL,
            state_sub_attr="state",
        ),
        FieldDef(
            key="other_boundary_accumulation_state", attr_path="_other_boundary_accumulation",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=OtherBoundaryAccumulationState, version=38, group=SG.OTHER_MODEL,
            state_sub_attr="state", session_decay=True,
        ),
        FieldDef(
            key="hypothesis_observation_pairing_state", attr_path="_hypothesis_observation_pairing",
            save_interface=SI.TO_DICT, load_interface=LI.PRIVATE_STATE_ATTR,
            load_type=HOPairingState, version=42, group=SG.OTHER_MODEL,
            state_sub_attr="state",
        ),

        # ── 記述・認知 (DESCRIPTION_COGNITION) ───────────────────
        FieldDef(
            key="meta_emotion_state", attr_path="_meta_emotion_processor",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=MetaEmotionState, version=16, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state", session_decay=True,
        ),
        FieldDef(
            key="temporal_cognition_state", attr_path="_temporal_cognition",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=TemporalCognitionState, version=20, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state",
        ),
        FieldDef(
            key="perceptual_context_state", attr_path="_perceptual_context",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=22, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="stabilization_description_state", attr_path="_stabilization_desc_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=StabilizationDescriptionState, version=26, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="behavioral_diversity_state", attr_path="_behavioral_diversity_state",
            save_interface=SI.TO_DICT, load_interface=LI.DIRECT_ASSIGN,
            load_type=BehavioralDiversityState, version=27, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="internal_contradiction_state", attr_path="_contradiction_processor",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=29, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="interaction_accumulation_state", attr_path="_interaction_accumulation",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=InteractionAccumulationState, version=30, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state",
        ),
        FieldDef(
            key="emotional_backdrop_state", attr_path="_emotional_backdrop_processor",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=31, group=SG.DESCRIPTION_COGNITION,
            session_decay=True,
        ),
        FieldDef(
            key="situational_self_presentation_state", attr_path="_situational_self_presentation",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=SituationalSelfPresentationState, version=32, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state", session_decay=True,
        ),
        FieldDef(
            key="drive_variation_state", attr_path="_drive_variation_processor",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=33, group=SG.DESCRIPTION_COGNITION,
            session_decay=True,
        ),
        FieldDef(
            key="expectation_lifecycle_state", attr_path="_expectation_lifecycle_processor",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=ExpectationLifecycleState, version=34, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state",
        ),
        FieldDef(
            key="input_pathway_balance_state", attr_path="_input_pathway_balance_state",
            save_interface=SI.EXTERNAL_SAVE, load_interface=LI.EXTERNAL_LOAD,
            save_func=save_input_pathway_balance_state,
            load_func=load_input_pathway_balance_state,
            version=35, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="responsibility_temporal_trace_state", attr_path="_responsibility_temporal_trace",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=36, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="emotion_cooccurrence_state", attr_path="_emotion_cooccurrence_processor",
            save_interface=SI.SAVE_METHOD, load_interface=LI.LOAD_METHOD,
            version=37, group=SG.DESCRIPTION_COGNITION,
            session_decay=True,
        ),
        FieldDef(
            key="forgetting_recall_balance_state", attr_path="_frb_state",
            save_interface=SI.EXTERNAL_SAVE, load_interface=LI.EXTERNAL_LOAD,
            save_func=save_frb_state, load_func=load_frb_state,
            version=39, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="attention_distribution_state", attr_path="_att_dist_state",
            save_interface=SI.EXTERNAL_SAVE, load_interface=LI.EXTERNAL_LOAD,
            save_func=save_att_dist_state, load_func=load_att_dist_state,
            version=40, group=SG.DESCRIPTION_COGNITION,
        ),
        FieldDef(
            key="goal_hierarchy_propagation_state", attr_path="_goal_hierarchy_propagation",
            save_interface=SI.TO_DICT, load_interface=LI.STATE_ATTR,
            load_type=GoalHierarchyPropagationState, version=41, group=SG.DESCRIPTION_COGNITION,
            state_sub_attr="state",
        ),
    ]


# モジュールロード時にフィールド定義を構築
FIELD_DEFINITIONS: list[FieldDef] = _build_field_definitions()


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
        # 毎ティック帯域の経過時間一時保存（同一ティック内での引数受渡し用、永続化対象外）
        self._last_delta_time: float = 0.0
        # Phase 1 → Phase 2/2a 間の loop_result 受渡し用一時属性（永続化対象外）
        self._last_loop_result: Any = None

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

        # ── Memory emotion return (記憶想起から感情への帰還経路) ──
        self._memory_emotion_return = create_memory_emotion_return()

        # ── Other hypothesis emotion return (他者仮説由来の感情帰還経路) ──
        self._other_hypothesis_emotion_return = create_other_hypothesis_emotion_return()

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

        # ── Enrichment empty skip tracker (セッション境界で消失、save/load対象外) ──
        from .enrichment_compression import EmptySkipTracker
        self._enrichment_empty_skip_tracker = EmptySkipTracker()

        # ── Session boundary freshness (save/load対象外、毎セッション再算出) ──
        self._session_gap_seconds: Optional[float] = None
        self._session_resume_tick: Optional[int] = None

        # ── Session difference (セッション間差分記述) ──
        # 前回復元時の辞書データ（次回保存時の差分算出に使用）
        self._session_prev_snapshot: Optional[dict[str, Any]] = None
        # 直近の保存時に算出された差分スカラー値（復元時に読み取り）
        self._session_diff_scalar: Optional[float] = None

        # ── Phase 30-35 cached results (for enrichment) ──
        self._last_decision_bias: Optional[DecisionBias] = None
        self._last_tone_mod: Optional[ToneModifier] = None
        self._last_sensitivity_bias: Optional[SensitivityBias] = None
        self._last_has_silence: bool = False

        # ── Phase execution engine (段階4: 毎ティック帯域拡大) ──
        # save/load非影響・enrichment非接続。永続化しない。
        # 対象帯域: 毎ティック帯域（段階4で追加）+ 3ティック帯域（段階3）+ 10ティック帯域（段階2）
        self._phase_engine = PhaseExecutionEngine()
        # 10ティック帯域ハンドラ（段階2から継続）
        self._phase_engine.register_handler("27", self._phase27_handler)
        self._phase_engine.register_handler("28", self._phase28_handler)
        self._phase_engine.register_handler("29", self._phase29_handler)
        # 3ティック帯域ハンドラ（段階3で追加: 17 Phase）
        self._phase_engine.register_handler("8", self._phase8_handler)
        self._phase_engine.register_handler("9", self._phase9_handler)
        self._phase_engine.register_handler("10", self._phase10_handler)
        self._phase_engine.register_handler("11", self._phase11_handler)
        self._phase_engine.register_handler("12", self._phase12_handler)
        self._phase_engine.register_handler("12b", self._phase12b_handler)
        self._phase_engine.register_handler("13", self._phase13_handler)
        self._phase_engine.register_handler("14", self._phase14_handler)
        self._phase_engine.register_handler("14b", self._phase14b_handler)
        self._phase_engine.register_handler("14c", self._phase14c_handler)
        self._phase_engine.register_handler("14d", self._phase14d_handler)
        self._phase_engine.register_handler("14e", self._phase14e_handler)
        self._phase_engine.register_handler("14f", self._phase14f_handler)
        self._phase_engine.register_handler("14g", self._phase14g_handler)
        self._phase_engine.register_handler("14h", self._phase14h_handler)
        self._phase_engine.register_handler("14i", self._phase14i_handler)
        self._phase_engine.register_handler("14j", self._phase14j_handler)
        # 毎ティック帯域ハンドラ（段階4で追加: 16 Phase）
        self._phase_engine.register_handler("1", self._phase1_handler)
        self._phase_engine.register_handler("2", self._phase2_handler)
        self._phase_engine.register_handler("2a", self._phase2a_handler)
        self._phase_engine.register_handler("2b", self._phase2b_handler)
        self._phase_engine.register_handler("2c", self._phase2c_handler)
        self._phase_engine.register_handler("3", self._phase3_handler)
        self._phase_engine.register_handler("4", self._phase4_handler)
        self._phase_engine.register_handler("5", self._phase5_handler)
        self._phase_engine.register_handler("6", self._phase6_handler)
        self._phase_engine.register_handler("7", self._phase7_handler)
        self._phase_engine.register_handler("7a", self._phase7a_handler)
        self._phase_engine.register_handler("7b", self._phase7b_handler)
        self._phase_engine.register_handler("7c", self._phase7c_handler)
        self._phase_engine.register_handler("7d", self._phase7d_handler)
        self._phase_engine.register_handler("7e", self._phase7e_handler)
        self._phase_engine.register_handler("7f", self._phase7f_handler)

        # ── Execution monitor (実運用向けログ・モニタリング基盤) ──
        # 永続化対象外。save/loadフィールド追加なし。
        # 内部の判断・行動・選択に一切介入しない(READ-ONLY観測のみ)。
        self._execution_monitor = ExecutionMonitor()

        # ── Policy selection log (ポリシー選択ログ) ──
        # 永続化対象外。save/loadフィールド追加なし。enrichment非接続。
        # 方針選択処理への逆流経路を持たない(READ-ONLY観測のみ)。
        self._policy_selection_log = create_policy_selection_log()

        # ── Return pathway monitor (帰還経路の動作検証と相互干渉検出) ──
        # 永続化対象外。save/loadフィールド追加なし。enrichment非接続。
        # 帰還経路の処理ロジックに一切変更を加えない(READ-ONLY観測のみ)。
        # セッション境界で全内部状態が消失する。
        self._return_pathway_monitor = ReturnPathwayMonitor()

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
    # 処理実行コードは psyche/orchestrator_1tick_phases.py に物理的に分離済み。
    # 実行エンジンが有効な場合は宣言的定義に基づいて駆動し、
    # 無効な場合は既存の手続き的コード（分離先の統括関数）にフォールバックする。
    # ムード自律更新用コンテキスト構築関数も分離先に移動済み。

    def _run_every_tick(
        self,
        percept: Percept,
        delta_time: float,
        user_id: str,
    ) -> None:
        """毎ティック実行するフェーズ (Phase 1-7f).

        実行エンジンが有効な場合は宣言的定義に基づいて駆動し、
        無効な場合は既存の手続き的コードにフォールバックする。
        外部から見た呼び出しインターフェースは不変。
        """
        # 知覚入力と経過時間を一時属性に保存（ハンドラから標準シグネチャ経由で取得）
        self._last_percept = percept
        self._last_delta_time = delta_time

        if self._phase_engine.is_band_enabled(Band.EVERY_TICK):
            # 宣言的実行エンジンによる駆動
            self._phase_engine.execute_band(self, user_id, band=Band.EVERY_TICK)
        else:
            # フォールバック: 既存の手続き的コード
            _run_1tick_phases(self, percept, delta_time, user_id)

        # 帯域末尾のデバッグログ出力（エンジン/フォールバック共通）
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

    # ── 毎ティック帯域ハンドラ（Phase 1-7f: 16 Phase） ────────────
    # 各ハンドラは標準シグネチャ（self + _orchestrator + user_id）で呼ばれる。
    # 知覚入力と経過時間は一時属性（_last_percept, _last_delta_time）から取得する。
    # 処理内容は orchestrator_1tick_phases.py の既存コードと等価。

    def _phase1_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 1: react_with_stm — 感情更新 + STM残留 + ムード自律更新。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        from .orchestrator_1tick_phases import build_mood_context, build_drive_context
        percept = self._last_percept
        delta_time = self._last_delta_time
        resp_influence = self._responsibility_mgr.get_influence(user_id)
        mood_ctx = build_mood_context(self, resp_influence)
        drive_ctx = build_drive_context(self)
        new_psyche, new_loop, self._last_loop_result = react_with_stm(
            percept=percept,
            psyche_state=self._psyche,
            loop_state=self._loop_state,
            delta_time=delta_time,
            responsibility_influence=resp_influence,
            mood_context=mood_ctx,
            drive_context=drive_ctx,
        )
        self._psyche = new_psyche
        self._loop_state = new_loop

    def _phase2_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 2: dynamics — ピーク/リバウンド判定。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        emo_dict = self._psyche.emotions.as_dict()
        residue_total = self._last_loop_result.residue_influence.total_intensity
        self._dynamics = update_dynamics(
            state=self._dynamics,
            current_emotions=emo_dict,
            residue_intensity=residue_total,
        )

    def _phase2a_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 2a: emotion_amplitude — dynamics相による振幅計算。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        delta_time = self._last_delta_time
        emo_dict = self._psyche.emotions.as_dict()
        residue_total = self._last_loop_result.residue_influence.total_intensity
        max_emo = max(emo_dict.values()) if emo_dict else 0.0
        self._amplitude_state = update_amplitude(
            self._amplitude_state,
            intensity_factor=residue_total,
            emotion_intensity=max_emo,
        )
        self._amplitude_state = decay_amplitude(self._amplitude_state, delta_time)

    def _phase2b_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 2b: multi_emotion — 感情別独立減衰。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        delta_time = self._last_delta_time
        decayed = apply_independent_decay(
            self._psyche.emotions,
            delta_time=delta_time,
            config=self._multi_emotion_config,
        )
        self._psyche = self._psyche.model_copy(update={"emotions": decayed})

    def _phase2c_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 2c: stm_emotion_coupling — STM残留→再活性化・蓄積。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        delta_time = self._last_delta_time
        if self._loop_state and self._loop_state.memory:
            coupled, self._last_coupling = apply_stm_coupling(
                emotions=self._psyche.emotions,
                stm=self._loop_state.memory,
                delta_time=delta_time,
                config=self._stm_coupling_config,
                apply_persistence=False,
            )
            self._psyche = self._psyche.model_copy(update={"emotions": coupled})

    def _phase3_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 3: attachment — 対話相手ボンド更新。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
        if self._psyche.attachment is not None:
            valence = abs(percept.emotion_valence)
            event_type = "positive" if percept.emotion_valence >= 0 else "negative"
            self._psyche = self._psyche.model_copy(update={
                "attachment": attachment_manager.update_bond(
                    self._psyche.attachment, user_id, event_type, valence,
                ),
            })

    def _phase4_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 4: responsibility — 判断記録。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
        if percept.intent and percept.intent != "expression":
            policy_label = percept.intent
        else:
            policy_label = percept.emotion or "neutral"
        self._responsibility_mgr.record_decision(
            user_id=user_id,
            policy={"policy_label": policy_label},
            context={"emotion": percept.emotion, "text": percept.text[:100]},
        )

    def _phase5_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 5: self_reference — 自己参照サマリ。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._self_ref_state = execute_self_reference(
            psyche_state=self._psyche,
            responsibility_state=self._responsibility_mgr.get_state(user_id),
            short_term_memory=self._loop_state.memory if self._loop_state else None,
            dynamics_state=self._dynamics,
            dispersion_state=self._dispersion_state,
        )

    def _phase6_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 6: repeated_tendency — 傾向観測。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        scoped = (
            self._scoped_goal_sys._current_scope
            if self._scoped_goal_sys.has_active_scope
            else None
        )
        self._tendency_sys.observe_turn(scoped_goal_used=scoped)

    def _phase7_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7: fear recompute — 4柱リスク再計算。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._recompute_fear()

    def _phase7a_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7a: action_result_observation — 行動記録→構成バッファ。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
        if self._last_selected_policy_label:
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

    def _phase7b_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7b: temporal_cognition — 経過記録の蓄積（毎ティック）。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
        delta_time = self._last_delta_time
        current_pathway_for_tc = ""
        if percept.text:
            current_pathway_for_tc = "text"
        elif percept.intent and percept.intent != "expression":
            current_pathway_for_tc = "screen"
        self._temporal_cognition.accumulate_elapsed(
            tick=self._tick_count,
            delta_time=delta_time,
            timestamp=time.time(),
            current_pathway=current_pathway_for_tc,
        )

    def _phase7c_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7c: perceptual_context — 知覚サマリの蓄積（毎ティック）。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
        self._perceptual_context.accumulate_summary(
            emotion=percept.emotion or "neutral",
            intent=percept.intent or "unknown",
            topics=list(getattr(percept, 'topics', ()) or ()),
            emotion_valence=percept.emotion_valence,
            tick=self._tick_count,
        )

    def _phase7d_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7d: situational_self_presentation — 相手別自己出力記録蓄積。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        latest_record = self._self_action_recorder.get_latest_record()
        if latest_record is not None and user_id:
            self._situational_self_presentation.receive_and_accumulate(
                user_id=user_id,
                response_text=latest_record.response_text,
                policy_label=latest_record.policy_label,
                tick=self._tick_count,
            )
            self._situational_self_presentation.generate_compositions(
                current_tick=self._tick_count,
            )

    def _phase7e_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7e: input_pathway_balance — 入力経路間均衡記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
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

    def _phase7f_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 7f: attention_distribution_description — 注意配分の構造的記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        percept = self._last_percept
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

    # ── Phase 8-14: Every 3 ticks ────────────────────────────────

    def _run_every_3_ticks(self, user_id: str) -> None:
        """3ティック毎の自己モデル + 目標フェーズ (Phase 8-14).

        実行エンジンが有効な場合は宣言的定義に基づいて駆動し、
        無効な場合は既存の手続き的コードにフォールバックする。
        外部から見た呼び出しインターフェースは不変。
        """
        if self._phase_engine.is_band_enabled(Band.EVERY_3_TICKS):
            # 宣言的実行エンジンによる駆動
            self._phase_engine.execute_band(self, user_id, band=Band.EVERY_3_TICKS)
        else:
            # フォールバック: 既存の手続き的コード
            self._run_every_3_ticks_fallback(user_id)

        # 帯域末尾のデバッグログ出力（エンジン/フォールバック共通）
        logger.debug(
            "Tick %d every-3: self_model=%s, goals=%d vectors, motives=%s",
            self._tick_count,
            "ok" if self._last_self_view else "none",
            len(self._vector_gen.state.vectors) if self._vector_gen else 0,
            "ok" if self._last_motives else "none",
        )

    def _run_every_3_ticks_fallback(self, user_id: str) -> None:
        """3ティック帯域のフォールバック実行（既存手続き的コード）.

        実行エンジン無効化時に使用される。
        実行エンジンの安定性が確認された後の段階で除去を検討する（本設計の範囲外）。
        """

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
        try:
            me_inputs = self._build_meta_emotion_inputs()
            self._last_meta_emotion = self._meta_emotion_processor.tick(me_inputs)
        except Exception as e:
            logger.debug("Meta-emotion cognition skipped: %s", e)

        # Phase 14c: temporal_cognition — 多断面特徴量の記述（3ティック毎）
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
            band_freshness_info: dict[str, int] = {}
            try:
                tc = self._tick_count
                band_freshness_info["every_tick"] = 0
                band_freshness_info["every_3"] = tc % 3
                band_freshness_info["every_5"] = tc % 5
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
        try:
            self._perceptual_context.describe_features()
        except Exception as e:
            logger.debug("Perceptual context describe skipped: %s", e)

        # Phase 14f: internal_contradiction_description — 内部状態の矛盾並置の構造的記述（3ティック毎）
        try:
            contradiction_inputs = self._build_contradiction_inputs()
            self._last_contradiction_result = self._contradiction_processor.process(
                contradiction_inputs
            )
        except Exception as e:
            logger.debug("Internal contradiction description skipped: %s", e)

        # Phase 14g: emotional_backdrop_cognition — 感情基調の持続認知（3ティック毎）
        try:
            backdrop_inputs = self._build_backdrop_inputs()
            self._last_backdrop_result = self._emotional_backdrop_processor.tick(backdrop_inputs)
        except Exception as e:
            logger.debug("Emotional backdrop cognition skipped: %s", e)

        # Phase 14h: introspection_longitudinal_view — 内省の時間的縦断参照（3ティック毎）
        try:
            ilv_snapshots = self._introspection_cross_section.get_snapshot_window()
            self._introspection_longitudinal_view.process(ilv_snapshots)
        except Exception as e:
            logger.debug("Introspection longitudinal view skipped: %s", e)

        # Phase 14i: drive_variation_description — 駆動の変動記述（3ティック毎）
        try:
            dv_inputs = self._build_drive_variation_inputs()
            self._last_drive_variation_result = self._drive_variation_processor.tick(dv_inputs)
        except Exception as e:
            logger.debug("Drive variation description skipped: %s", e)

        # Phase 14j: emotion_cooccurrence_description — 感情間の共起記述（3ティック毎）
        try:
            emo_values = self._psyche.emotions.as_dict()
            self._last_cooccurrence_result = self._emotion_cooccurrence_processor.tick(emo_values)
        except Exception as e:
            logger.debug("Emotion cooccurrence description skipped: %s", e)

    # ── Phase 15-26: Every 5 ticks ───────────────────────────────
    # 処理実行コードは psyche/orchestrator_5tick_phases.py に物理的に分離済み。
    # 本メソッドは分離先の統括関数への委譲のみを行う。

    def _run_every_5_ticks(self, user_id: str) -> None:
        """5ティック毎の自己連続性 + 記憶 + 内省フェーズ (Phase 15-26)."""
        _run_5tick_phases(self, user_id)

    # ── Phase 8-14j: 3ティック帯域ハンドラ ────────────────────────
    # 実行エンジンから呼び出される個別Phase処理関数。
    # 処理の「中身」は既存の手続き的コードをそのまま保持する。
    # 統合管理構造インスタンスと追加引数を受け取る共通インターフェース。

    def _phase8_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 8: tendency_awareness — 傾向認知。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._tendency_awareness = observe_tendencies(self._tendency_sys)

    def _phase9_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 9: self_model — 統合自己ビュー。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        resp_state = self._responsibility_mgr.get_state(user_id)
        self._last_self_view = self._self_model_sys.observe(
            emotion_vector=self._psyche.emotions,
            responsibility_state=resp_state,
            tendency_system=self._tendency_sys,
            tendency_awareness=self._tendency_awareness,
            proto_goal_system=self._vector_gen,
            value_orientation=self._value_orientation,
        )

    def _phase10_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 10: proto_goal_vector — 方向ベクトル更新。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
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

    def _phase11_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 11: goal_candidates — 目標候補生成。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        vectors = self._vector_gen.state.vectors
        self._candidate_gen.observe_vectors(vectors)

    def _phase12_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 12: transient_goal — 一時目標選択。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        candidates_list = self._candidate_gen.state.candidates
        self._transient_goal_mgr.observe_turn(
            available_candidates=candidates_list,
        )

    def _phase12b_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 12b: persistent_commitment — 持続的取り組み保持。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._run_persistent_commitment()

    def _phase13_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 13: scoped_goal — スコープ目標コミット。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._scoped_goal_sys.begin_turn(
            transient_manager=self._transient_goal_mgr,
            active_goal=self._transient_goal_mgr.state.active_goal,
        )

    def _phase14_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14: intrinsic_motivation — 内的動機感知。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._last_motives = sense_motives_from_chain(
            system=self._motivation_sys,
            emotion=self._psyche.emotions,
            mood=self._psyche.mood,
            tendencies=self._tendency_sys.state.tendencies if self._tendency_sys else None,
            vectors=self._vector_gen.state.vectors if self._vector_gen else None,
            candidates=self._candidate_gen.state.candidates if self._candidate_gen else None,
        )

    def _phase14b_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14b: meta_emotion_cognition — メタ感情認知と変動候補生成。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        me_inputs = self._build_meta_emotion_inputs()
        self._last_meta_emotion = self._meta_emotion_processor.tick(me_inputs)

    def _phase14c_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14c: temporal_cognition — 多断面特徴量の記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
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
        band_freshness_info: dict[str, int] = {}
        try:
            tc = self._tick_count
            band_freshness_info["every_tick"] = 0
            band_freshness_info["every_3"] = tc % 3
            band_freshness_info["every_5"] = tc % 5
            band_freshness_info["every_10"] = tc % 10
        except Exception:
            band_freshness_info = {}

        self._temporal_cognition.describe_features(
            episodic_timestamps=ep_timestamps or None,
            emotion_change_count=emotion_change_count,
            narrative_timestamps=narr_timestamps or None,
            band_freshness=band_freshness_info or None,
        )

    def _phase14d_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14d: introspection_cross_section — 内省断面のスナップショット構成。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
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

    def _phase14e_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14e: perceptual_context — 知覚推移特徴量の記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        self._perceptual_context.describe_features()

    def _phase14f_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14f: internal_contradiction_description — 矛盾並置記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        contradiction_inputs = self._build_contradiction_inputs()
        self._last_contradiction_result = self._contradiction_processor.process(
            contradiction_inputs
        )

    def _phase14g_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14g: emotional_backdrop_cognition — 感情基調の持続認知。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        backdrop_inputs = self._build_backdrop_inputs()
        self._last_backdrop_result = self._emotional_backdrop_processor.tick(backdrop_inputs)

    def _phase14h_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14h: introspection_longitudinal_view — 内省の時間的縦断参照。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        ilv_snapshots = self._introspection_cross_section.get_snapshot_window()
        self._introspection_longitudinal_view.process(ilv_snapshots)

    def _phase14i_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14i: drive_variation_description — 駆動の変動記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        dv_inputs = self._build_drive_variation_inputs()
        self._last_drive_variation_result = self._drive_variation_processor.tick(dv_inputs)

    def _phase14j_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 14j: emotion_cooccurrence_description — 感情間の共起記述。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        emo_values = self._psyche.emotions.as_dict()
        self._last_cooccurrence_result = self._emotion_cooccurrence_processor.tick(emo_values)

    # ── Phase 27-29: Every 10 ticks ──────────────────────────────

    # Phase処理関数: 実行エンジンから呼び出される個別Phase処理
    # 処理の「中身」は既存の手続き的コードをそのまま保持する。

    def _phase27_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 27: stability_valve — 極端偏り検出。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        resp_influence = self._responsibility_mgr.get_influence(user_id)
        self._stability_valve.observe_extremity(
            fear_level=self._psyche.fear_level,
            responsibility_weight=resp_influence.caution_bias,
            value_orientation=self._value_orientation,
            emotion_state=self._psyche.emotions,
        )

    def _phase28_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 28: long_term_dynamics — 長期行動ログ。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        resp_influence = self._responsibility_mgr.get_influence(user_id)
        self._dynamics_observer.record_turn(
            emotion_state=self._psyche.emotions,
            value_orientation=self._value_orientation,
            responsibility_weight=resp_influence.caution_bias,
            responsibility_caution=resp_influence.caution_bias,
        )

    def _phase29_handler(self, _orchestrator: Any, user_id: str) -> None:
        """Phase 29: snapshot — 永続化は save() で明示的に行う。

        実行エンジンから呼び出される処理関数。
        処理内容は既存の手続き的コードと等価。
        """
        logger.debug(
            "Tick %d every-10: stability_valve observed, dynamics logged",
            self._tick_count,
        )

    def _run_every_10_ticks(self, user_id: str) -> None:
        """10ティック毎の安定性 + ロギング + スナップショット (Phase 27-29).

        実行エンジンが有効な場合は宣言的定義に基づいて駆動し、
        無効な場合は既存の手続き的コードにフォールバックする。
        外部から見た呼び出しインターフェースは不変。
        """
        if self._phase_engine.enabled:
            # 宣言的実行エンジンによる駆動
            self._phase_engine.execute_band(self, user_id)
        else:
            # フォールバック: 既存の手続き的コード
            self._run_every_10_ticks_fallback(user_id)

    def _run_every_10_ticks_fallback(self, user_id: str) -> None:
        """10ティック帯域のフォールバック実行（既存手続き的コード）.

        実行エンジン無効化時に使用される。
        実行エンジンの安定性が確認された後の段階で除去を検討する（本設計の範囲外）。
        """
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
        with BandTimer(self._execution_monitor, "every_tick"):
            self._run_every_tick(percept, delta_time, user_id)

        # Phase 8-14: every 3 ticks
        if self._tick_count % 3 == 0:
            with BandTimer(self._execution_monitor, "every_3_ticks"):
                self._run_every_3_ticks(user_id)

        # Phase 15-26: every 5 ticks
        if self._tick_count % 5 == 0:
            with BandTimer(self._execution_monitor, "every_5_ticks"):
                self._run_every_5_ticks(user_id)

        # Phase 27-29: every 10 ticks
        if self._tick_count % 10 == 0:
            with BandTimer(self._execution_monitor, "every_10_ticks"):
                self._run_every_10_ticks(user_id)

        # ── Execution monitor: サイクル完了記録 + スナップショット ──
        # 計測点自体が例外を投げた場合、元の処理に影響を与えない(安全弁1)
        try:
            self._execution_monitor.record_cycle_complete(self._tick_count)
            self._execution_monitor.maybe_emit_snapshot(
                self._tick_count,
                lambda: read_orchestrator_fields(self),
            )
        except Exception:
            pass

        # ── Return pathway: 合算帯域上限 ──
        # 全帰還経路の変動が内部状態に適用された後、finalize_tickの直前に配置。
        # 種類別(感情/ドライブ/ムード追従速度)に独立した合算上限を判定し、
        # 超過時は全経路に等比率の比例縮小を適用する。
        # 計測失敗時は帰還量をそのまま保持する(安全弁パターン踏襲)。
        try:
            _apply_return_aggregate_cap(self)
        except Exception:
            pass

        # ── Return pathway monitor: サイクルサマリー ──
        # ティック処理完了時に、そのティック内で発火した帰還経路の合算記述を記録。
        # 計測点自体が例外を投げた場合、元の処理に影響を与えない(安全弁パターン踏襲)。
        try:
            self._return_pathway_monitor.finalize_tick(self._tick_count)
        except Exception:
            pass

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
    # enrichment生成ロジックはorchestrator_enrichment.pyに物理的に分離済み。
    # 以下は外部インターフェースを維持するための委譲メソッドのみ。

    def get_prompt_enrichment(self, user_id: str = "viewer") -> str:
        """Gemini プロンプト用の心理状態テキストを生成する。

        enrichment項目を収集後、空状態統一→圧縮パイプライン→鮮度注釈を適用する。
        圧縮結果のフィードバック経路は構造的に遮断されている。

        brain.py の _format_psyche_for_prompt を置き換える。

        Args:
            user_id: 責任状態取得用のユーザーID
        """
        return _get_prompt_enrichment(self, user_id)

    # ── Policy suggestions ────────────────────────────────────────

    def _generate_final_candidates(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        user_id: str,
    ) -> tuple[list[dict], Any]:
        """Phase 30-35: 候補生成+バイアス適用。candidatesとtone_modを返す。"""
        resp_influence = self._responsibility_mgr.get_influence(user_id)

        candidates, decision_bias = self._gen_candidates_preprocessing(
            percept, recalled_memories, user_id, resp_influence,
        )
        candidates, tone_mod, sensitivity_bias = self._gen_candidates_decoration(
            candidates, percept, recalled_memories, resp_influence, decision_bias,
        )
        candidates = self._gen_candidates_bias_application(candidates)

        # Cache Phase 30-35 results for enrichment
        self._last_decision_bias = decision_bias
        self._last_tone_mod = tone_mod
        self._last_sensitivity_bias = sensitivity_bias
        self._last_has_silence = any(
            c.get("policy_label") == "silence" for c in candidates
        )

        return candidates, tone_mod

    # ── Candidate generation group: Preprocessing (Phase 31, 30, 30b) ─

    def _gen_candidates_preprocessing(
        self,
        percept: Percept,
        recalled_memories: list[dict],
        user_id: str,
        resp_influence: Any,
    ) -> tuple[list[dict], Any]:
        """Phase 31, 30, 30b: 判断バイアス・候補生成・候補拡張."""

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

        # Phase 26-EXP 帯域拡大: 対象Bの加算量を参照（非永続属性）
        _score_band_add = getattr(self, '_exp_score_band_addition', None)

        candidates = generate_thought_candidates(
            state=self._psyche,
            percept=percept,
            recalled=recalled_memories,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
            extended_inputs=extended_inputs,
            collect_breakdown=self._policy_selection_log.enabled,
            score_section_band_addition=_score_band_add,
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

        return candidates, decision_bias

    # ── Candidate generation group: Decoration / Integration (Phase 32-34)

    def _gen_candidates_decoration(
        self,
        candidates: list[dict],
        percept: Percept,
        recalled_memories: list[dict],
        resp_influence: Any,
        decision_bias: Any,
    ) -> tuple[list[dict], Any, Any]:
        """Phase 32-34: トーン・空気読み・沈黙."""

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

        return candidates, tone_mod, sensitivity_bias

    # ── Candidate generation group: Bias Application (Phase 35-35c) ───

    def _gen_candidates_bias_application(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """Phase 35-35c: 安定化・価値軸・持続的取り組み・スコアリング揺らぎ."""

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
        # 第5入力源: 参照頻度記述構造の偏在度（READ-ONLY）
        try:
            now = time.time()
            elapsed = now - self._last_fluctuation_select_time if self._last_fluctuation_select_time > 0 else 0.0
            self._last_fluctuation_select_time = now

            # 参照偏在度の取得（参照頻度記述構造からREAD-ONLY）
            ref_imbalance: Optional[float] = None
            try:
                ref_snap = get_ref_freq_latest_snapshot(self._reference_frequency_state)
                if ref_snap is not None:
                    ref_imbalance = ref_snap.structural_bias
            except Exception:
                pass

            p = self._psyche
            candidates = apply_scoring_fluctuation(
                candidates=candidates,
                emotions=p.emotions.as_dict() if p.emotions else {},
                drives=p.drives.as_dict() if p.drives else {},
                stm=self._loop_state.memory if self._loop_state else None,
                elapsed_seconds=elapsed,
                config=self._fluctuation_config,
                reference_imbalance=ref_imbalance,
            )
        except Exception as e:
            logger.debug("Scoring fluctuation skipped: %s", e)

        return candidates

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

        # ポリシー選択ログ: スコア内訳を外部分析ツールに記録
        # 方針選択処理の呼び出し直後に返却値からスコア内訳を抽出して渡す
        try:
            self._policy_selection_log.record(
                tick=self._tick_count,
                selected_label=self._last_selected_policy_label,
                candidates=candidates,
                selected_count=len(candidates),
            )
        except Exception:
            pass

        # ── expected_drive_change の実適用 ──
        # ポリシー選択直後にドライブ帰還を適用する（1選択イベントにつき1回）
        try:
            self._apply_expected_drive_change(policy)
        except Exception:
            pass

        # ── 選択結果から感情への帰還 ──
        # ドライブ帰還の直後に感情帰還を適用する（1選択イベントにつき1回）
        try:
            self._apply_selection_emotion_return(policy, len(candidates))
        except Exception:
            pass

        return policy

    def _apply_expected_drive_change(self, policy: dict[str, Any]) -> None:
        """選択されたポリシーの expected_drive_change をドライブに実適用する。

        安全弁5種:
        1. 帰還量の軸別上限: 各軸の変化量を max_bias_strength 以下にクランプ
        2. ドライブ有効範囲のクランプ: 0.0-1.0
        3. 適用回数制限: select_policy_dict 直後の1回のみ（構造的に保証）
        4. 宣言値の不変性保証: コピーして使用
        5. 正の帰還の禁止: 正の値は適用しない
        """
        edc = policy.get("expected_drive_change")
        if not edc or not isinstance(edc, dict):
            return

        # 安全弁4: 宣言値をコピーして使用（元の定義を変更しない）
        changes = dict(edc)

        # 帰還量の上限: ValueOrientationConfig.max_bias_strength と同値
        # （価値方向性の最大影響幅を超えない保証）
        max_change = 0.15

        current_drives = self._psyche.drives.as_dict()
        new_drives = dict(current_drives)

        for axis in ("social", "curiosity", "expression"):
            delta = changes.get(axis, 0.0)
            if not isinstance(delta, (int, float)):
                continue

            # 安全弁5: 正の帰還の禁止
            if delta > 0.0:
                continue

            # 安全弁1: 帰還量の軸別上限クランプ
            if abs(delta) > max_change:
                delta = -max_change  # delta は負なので符号を保持

            # 加算
            new_val = current_drives.get(axis, 0.5) + delta

            # 安全弁2: ドライブ有効範囲のクランプ (0.0-1.0)
            new_drives[axis] = max(0.0, min(1.0, new_val))

        # ドライブ更新を反映
        updated_drive_vector = DriveVector(**new_drives)
        self._psyche = self._psyche.model_copy(
            update={"drives": updated_drive_vector}
        )

    def _apply_selection_emotion_return(
        self, policy: dict[str, Any], candidate_count: int,
    ) -> None:
        """選択結果から感情ベクトルへの帰還経路。

        3段パイプライン:
          段階1: 帰還方向の導出（ドライブ対象軸 + 選択時内部状態から間接導出）
          段階2: 帰還量の導出（距離比例・候補数スケーリング・arousalスケーリング）
          段階3: 適用と安全弁

        安全弁6種:
          1. 帰還帯域の上限制限: 各感情次元の帰還量をドライブ帰還上限(0.15)の半分以下にクランプ
          2. 合計帰還量の上限: 全次元合計に上限を設ける
          3. 距離比例による自動抑制: 境界付近で帰還量が自動縮小
          4. 正帰還ループの3重遮断: 帯域制限 + 非固定性 + 距離比例
          5. 適用回数の構造的制限: select_policy_dict内で1回のみ呼ばれる
          6. 有効範囲クランプ: 適用後の感情値を0.0-1.0にクランプ

        独自の永続的状態を保持しない（純粋関数として構成）。
        """
        edc = policy.get("expected_drive_change")
        if not edc or not isinstance(edc, dict):
            return

        # ── 定数 ──
        # 安全弁1: ドライブ帰還の軸別上限(0.15)の半分以下
        MAX_PER_DIM = 0.075
        # 安全弁2: 全次元合計の上限（ドライブ帰還合計上限 0.15*3=0.45 より小さい）
        MAX_TOTAL = 0.30

        # 感情7次元
        EMOTION_DIMS = ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun")

        # 選択時の内部状態スナップショット（読み取り専用）
        current_emotions = self._psyche.emotions.as_dict()
        mood_valence = self._psyche.mood.valence
        mood_arousal = self._psyche.mood.arousal
        fear_level = self._psyche.fear_level
        current_drives = self._psyche.drives.as_dict()

        # ドライブ対象軸の取得
        drive_target = policy.get("drive_target", "")
        if not drive_target:
            # drive_target がない場合は expected_drive_change から推定
            # 最も大きな変化量の軸をドライブ対象とする
            max_abs = 0.0
            for axis, val in edc.items():
                if isinstance(val, (int, float)) and abs(val) > max_abs:
                    max_abs = abs(val)
                    drive_target = axis

        if not drive_target:
            return

        # ドライブの現在値を取得
        drive_val = current_drives.get(drive_target, 0.5)
        # ドライブの宣言変化量（充足方向かどうかの判定に使用）
        drive_change = edc.get(drive_target, 0.0)
        if not isinstance(drive_change, (int, float)):
            drive_change = 0.0

        # ── 段階1: 帰還方向の導出 ──
        # 方針ラベルを直接参照しない。ドライブ対象軸と選択時内部状態から間接導出。
        # ドライブの充足が生じる場合（drive_change < 0）、関連感情次元に微弱な変化。
        # 方向は mood_valence と fear_level と感情の偏りから都度導出。
        directions: dict[str, float] = {}
        for dim in EMOTION_DIMS:
            emo_val = current_emotions.get(dim, 0.0)

            # 方向は状態断面に依存して都度異なる
            # 基本方向: mood_valence の符号とドライブ充足方向から導出
            if drive_change >= 0.0:
                # ドライブ充足なし → 帰還方向は中立に近い
                directions[dim] = 0.0
                continue

            # ドライブ充足がある場合: 感情偏りとムード正負から方向を導出
            # 感情値が高い次元ほど帰還方向がその次元に向かいやすい（状態依存）
            # fear_level が高い場合は fear/sorrow 次元に正方向、joy/fun に負方向
            if dim in ("fear", "sorrow"):
                direction = fear_level * 0.5 - mood_valence * 0.3 - emo_val * 0.2
            elif dim in ("joy", "fun", "love"):
                direction = mood_valence * 0.5 - fear_level * 0.3 + emo_val * 0.2
            elif dim == "anger":
                direction = -mood_valence * 0.3 + (1.0 - drive_val) * 0.2 - emo_val * 0.1
            elif dim == "surprise":
                direction = abs(drive_change) * 0.5 - emo_val * 0.3
            else:
                direction = 0.0

            directions[dim] = direction

        # ── 段階2: 帰還量の導出 ──
        deltas: dict[str, float] = {}
        total_abs = 0.0

        for dim in EMOTION_DIMS:
            direction = directions.get(dim, 0.0)
            if abs(direction) < 1e-9:
                deltas[dim] = 0.0
                continue

            emo_val = current_emotions.get(dim, 0.0)
            sign = 1.0 if direction > 0 else -1.0

            # 安全弁3: 距離比例による自動抑制
            # 境界値(1.0 or 0.0)との距離に比例して帰還量を縮小
            if sign > 0:
                distance = 1.0 - emo_val  # 上限との距離
            else:
                distance = emo_val  # 下限との距離
            distance = max(distance, 0.0)

            # 基本帰還量: |direction| * |drive_change| * 距離比例
            raw_amount = abs(direction) * abs(drive_change) * distance

            # 候補数スケーリング: 候補が少ないほど帰還量が抑制される
            # candidate_count=1 → scale=0.2, =3 → 0.6, =5+ → 1.0
            if candidate_count <= 0:
                count_scale = 0.0
            else:
                count_scale = min(1.0, candidate_count / 5.0)
            raw_amount *= count_scale

            # arousalスケーリング: 低覚醒時は帰還量が抑制される
            arousal_scale = max(0.1, mood_arousal)
            raw_amount *= arousal_scale

            # 安全弁1: 帰還帯域の上限制限（各次元）
            raw_amount = min(raw_amount, MAX_PER_DIM)

            deltas[dim] = sign * raw_amount
            total_abs += abs(deltas[dim])

        # 安全弁2: 合計帰還量の上限
        if total_abs > MAX_TOTAL:
            scale = MAX_TOTAL / total_abs
            for dim in EMOTION_DIMS:
                deltas[dim] *= scale

        # ── 段階3: 適用 ──
        new_emotions = dict(current_emotions)
        any_changed = False
        for dim in EMOTION_DIMS:
            delta = deltas.get(dim, 0.0)
            if abs(delta) < 1e-9:
                continue
            new_val = current_emotions.get(dim, 0.0) + delta
            # 安全弁6: 有効範囲クランプ (0.0-1.0)
            new_emotions[dim] = max(0.0, min(1.0, new_val))
            any_changed = True

        if any_changed:
            updated_emotions = EmotionVector(**new_emotions)
            self._psyche = self._psyche.model_copy(
                update={"emotions": updated_emotions}
            )

        # ── 帰還経路B: 発火通知 ──
        # 処理完了後に通知する(処理前や処理中ではない)。
        # 通知失敗時は例外を捕捉してスキップする(安全弁パターン踏襲)。
        try:
            # deltas から実際に適用された非ゼロの変動のみを抽出
            applied_deltas = {
                dim: v for dim, v in deltas.items() if abs(v) >= 1e-9
            }
            if applied_deltas:
                self._return_pathway_monitor.record_firing(
                    pathway_id=_RPM_PATHWAY_B,
                    tick_number=self._tick_count,
                    emotion_deltas=applied_deltas,
                )
        except Exception:
            pass

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
    #
    # save/load は FIELD_DEFINITIONS (モジュールレベル定義) に基づく
    # 共通ヘルパー save_fields / load_fields を使用する。
    # tick_count は特殊フィールドとして直接処理する。

    def save(self, path: Optional[Path] = None) -> None:
        """全状態を永続化する。

        FIELD_DEFINITIONS に基づく共通ヘルパーで全フィールドを保存する。

        Args:
            path: 保存先パス（デフォルト: data/psyche_snapshot.json）
        """
        save_path = path or (self._data_dir / "psyche_snapshot.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "version": CURRENT_VERSION,
            "save_timestamp": time.time(),
            "tick_count": self._tick_count,
        }
        data.update(save_fields(self, FIELD_DEFINITIONS))

        # セッション間差分: 前回スナップショットがあれば差分スカラーを算出
        if self._session_prev_snapshot is not None:
            try:
                diff_scalar = compute_session_difference_scalar(
                    self._session_prev_snapshot, data,
                )
                data["session_diff_scalar"] = diff_scalar
                self._session_diff_scalar = diff_scalar
            except Exception as e:
                logger.debug("Session diff computation failed: %s", e)
        # 差分フィールドが不明の場合は含めない（設計書の仕様）

        # Atomic write: tmp file + os.replace() to prevent corruption on crash
        tmp_path = save_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(save_path))
        logger.info("Psyche state saved to %s", save_path)

    def load(self, path: Optional[Path] = None) -> bool:
        """永続化された状態を復元する。

        FIELD_DEFINITIONS に基づく共通ヘルパーで全フィールドを復元する。
        マイグレーション定義に該当しないフィールドは「存在すれば読み込む」
        フォールバックで処理される（安全弁1）。

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

            # tick_count は特殊フィールド: 直接処理
            if "tick_count" in data:
                self._tick_count = data["tick_count"]

            # 全フィールドを共通ヘルパーで復元
            load_fields(self, FIELD_DEFINITIONS, data)

            # Session boundary freshness: compute gap from saved timestamp
            saved_ts = data.get("save_timestamp")
            if saved_ts is not None:
                self._session_gap_seconds = time.time() - saved_ts
                self._session_resume_tick = self._tick_count
            else:
                # No timestamp in snapshot (pre-existing data) — no annotation
                self._session_gap_seconds = None
                self._session_resume_tick = None

            # セッション間差分: 保存辞書から差分スカラー値を読み取り、
            # 辞書データを前回スナップショットとして保持
            self._session_diff_scalar = data.get("session_diff_scalar")
            self._session_prev_snapshot = dict(data)

            # Cache warmup: 復元済みモジュール蓄積状態から中間キャッシュを再導出
            # セッション境界鮮度注釈の計算後、最初のティック実行前に実行する。
            # 独自の内部状態を保持しない(1回実行のみ)。
            warmup_results = execute_warmup(self)

            # Session recovery check: 復元フィールド間の数値的整合性を検証
            # ウォームアップ完了後、最初のティック実行前に1回のみ実行する。
            # 警告のみ（修復・進行阻止なし）。内部状態への書き込みなし。
            execute_session_recovery_check(self, warmup_results)

            logger.info("Psyche state loaded from %s (v%d, tick=%d)",
                        load_path, data.get("version", 0), self._tick_count)
            return True
        except Exception as e:
            logger.error("Failed to load snapshot: %s", e)
            return False
