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
from dataclasses import dataclass, field
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
from .reaction_with_stm import react_with_stm, CombinedReactionState
from .short_term_loop import LoopState, create_loop_state

# Dynamics
from .dynamics import (
    DynamicsState,
    create_dynamics_state,
    update_dynamics,
    get_decay_modifier,
    get_dynamics_summary,
)

# Emotion amplitude
from .emotion_amplitude import (
    AmplitudeState,
    create_amplitude_state,
    update_amplitude,
    decay_amplitude,
    compute_amplitude_from_dynamics,
    apply_amplitude_to_emotion_deltas,
)

# Multi-emotion
from .multi_emotion import (
    MultiEmotionConfig,
    apply_independent_decay,
    get_active_emotions,
    has_conflicting_emotions,
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
    add_tone_to_candidates,
)

# Context sensitivity
from .context_sensitivity import (
    ExternalContext,
    ContextSensitivityConfig,
    ContextState,
    SensitivityBias,
    compute_sensitivity_bias,
    apply_sensitivity_to_candidates,
    create_neutral_context,
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
    ProtoGoalVector,
    get_vector_summary,
)

# Goal candidates
from .goal_candidates import (
    CandidateGenerator,
    CandidateState,
    GoalCandidate,
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
    MetaEmotionConfig,
    create_meta_emotion_processor,
    get_meta_emotion_summary,
)

# Self-action perception
from .self_action_perception import (
    SelfActionPerceptionRecorder,
    SelfActionPerceptionState,
    SelfActionPerceptionConfig,
    SelfActionRecord,
    create_self_action_perception_recorder,
    get_self_action_summary,
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

        # ── Intent-action gap (意図-行動間の乖離認知) ──
        self._intent_action_gap_recorder = create_intent_action_gap_recorder()

        # ── Temporal cognition (時間認知構造) ──
        self._temporal_cognition = create_temporal_cognition()

        # ── Phase 30-35 cached results (for enrichment) ──
        self._last_decision_bias: Optional[DecisionBias] = None
        self._last_tone_mod: Optional[ToneModifier] = None
        self._last_sensitivity_bias: Optional[SensitivityBias] = None
        self._last_has_silence: bool = False

        logger.info(
            "PsycheOrchestrator initialized: fear=%.2f, dominant=%s, "
            "systems=45",
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
        amp_from_dynamics = compute_amplitude_from_dynamics(self._dynamics)
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
                record_inputs = ActionResultInputs(
                    selected_policy_label=self._last_selected_policy_label,
                    selected_policy_axis=self._last_selected_policy_axis,
                    current_tick=self._tick_count,
                )
                self._action_result_observer.record_action(record_inputs)
            except Exception as e:
                logger.debug("Action-result record skipped: %s", e)

        # Phase 7b: temporal_cognition — 経過記録の蓄積（毎ティック）
        # 既存のティック数ベース処理を変更しない。出力は参照情報のみ。
        try:
            self._temporal_cognition.accumulate_elapsed(
                tick=self._tick_count,
                delta_time=delta_time,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.debug("Temporal cognition accumulate skipped: %s", e)

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

            self._temporal_cognition.describe_features(
                episodic_timestamps=ep_timestamps or None,
                emotion_change_count=emotion_change_count,
                narrative_timestamps=narr_timestamps or None,
            )
        except Exception as e:
            logger.debug("Temporal cognition describe skipped: %s", e)

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

        # Phase 22: introspection_trace — 内省ログ生成
        try:
            resp_influence = self._responsibility_mgr.get_influence(user_id)
            self._last_trace = self._introspection_sys.generate_trace(
                emotion_state=self._psyche.emotions,
                responsibility_state=resp_influence,
                value_orientation=self._value_orientation,
                fear_index=self._psyche.fear_index,
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
                    dl_non_rep = dl_data.get("non_repetition_active", 0)
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
                    rep_active = dl_data.get("repetition_active", 0)
                    non_rep_active = dl_data.get("non_repetition_active", 0)
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

    def get_prompt_enrichment(self, user_id: str = "viewer") -> str:
        """Gemini プロンプト用の心理状態テキストを生成する。

        brain.py の _format_psyche_for_prompt を置き換える。

        Args:
            user_id: 責任状態取得用のユーザーID
        """
        p = self._psyche
        sections: list[str] = []

        # ── 【心理状態（内面）】 ──
        psyche_lines = [
            "【心理状態（内面）】",
            f"感情: {p.emotion_summary()}",
            f"ムード: valence={p.mood.valence:.2f}, arousal={p.mood.arousal:.2f}",
            f"ドライブ: social={p.drives.social:.2f}, "
            f"curiosity={p.drives.curiosity:.2f}, "
            f"expression={p.drives.expression:.2f}",
            p.fear_summary(),
        ]
        if p.dominant_emotion_value > 0.3:
            psyche_lines.append(
                f"支配的感情: {p.dominant_emotion} "
                f"({p.dominant_emotion_value:.2f})"
            )
        # #1 responsibility
        try:
            resp_summary = self._responsibility_mgr.get_summary(user_id)
            psyche_lines.append(
                f"責任: weight={resp_summary['total_weight']:.2f}, "
                f"harm={resp_summary['accumulated_harm']:.2f}, "
                f"caution={resp_summary['influence']['caution_bias']:.2f}, "
                f"empathy={resp_summary['influence']['empathy_bias']:.2f}"
            )
        except Exception:
            pass
        # #2 responsibility_dispersion
        if self._dispersion_state is not None:
            disp_summary = get_dispersion_summary(self._dispersion_state)
            if disp_summary:
                psyche_lines.append(f"責任拡散: {disp_summary}")
        # #6 stability_valve
        try:
            valve_bias = self._stability_valve.generate_bias()
            if valve_bias.is_active:
                psyche_lines.append(
                    f"安定弁: active, level={valve_bias.activation_level:.2f}"
                )
        except Exception:
            pass
        # #10 stm_emotion_coupling
        if self._last_coupling is not None:
            coupling_str = get_coupling_summary(self._last_coupling)
            if coupling_str:
                psyche_lines.append(f"感情連動: {coupling_str}")
        sections.append("\n".join(psyche_lines))

        # ── 【自己認識】 ──
        self_lines = ["【自己認識】"]
        if self._last_self_image is not None:
            summary = get_self_image_summary(self._last_self_image)
            if summary:
                self_lines.append(f"自己像: {summary}")
        if self._last_coherence is not None:
            coh_summary = get_coherence_summary(self._last_coherence)
            if coh_summary:
                self_lines.append(f"一貫性: {coh_summary}")
        if self._tendency_awareness is not None:
            awareness_summary = get_awareness_summary(self._tendency_awareness)
            if awareness_summary:
                self_lines.append(f"傾向: {awareness_summary}")
        if self._last_diff_summary is not None:
            diff_summary = get_difference_summary(self._last_diff_summary)
            if diff_summary:
                self_lines.append(f"変化: {diff_summary}")
        if self._last_strain is not None:
            strain_summary = get_strain_summary(self._last_strain)
            if strain_summary:
                self_lines.append(f"連続性緊張: {strain_summary}")
        if self._last_narrative is not None:
            narr_summary = get_narrative_summary(self._last_narrative)
            if narr_summary:
                self_lines.append(f"自己語り: {narr_summary}")
        # #8 long_term_dynamics
        if self._dynamics_observer is not None:
            ltd_summary = get_observer_summary(self._dynamics_observer)
            if ltd_summary:
                self_lines.append(f"長期傾向: {ltd_summary}")
        # #27 temporal_cognition — 時間的特徴量の等価列挙（強調禁止）
        if self._temporal_cognition is not None:
            try:
                tc_data = self._temporal_cognition.get_enrichment_data()
                tc_text = tc_data.get("summary_text", "")
                if tc_text and "待機中" not in tc_text:
                    self_lines.append(f"時間認知: {tc_text}")
            except Exception:
                pass
        if len(self_lines) > 1:
            sections.append("\n".join(self_lines))

        # ── 【動機・目標】 ──
        motive_lines = ["【動機・目標】"]
        if self._last_motives is not None:
            motive_summary = get_motive_summary(self._last_motives)
            if motive_summary:
                motive_lines.append(f"動機: {motive_summary}")
        if self._candidate_gen is not None:
            try:
                cand_summary = get_candidate_summary(self._candidate_gen)
                if cand_summary:
                    total = cand_summary.get("total_candidates", 0)
                    if total > 0:
                        motive_lines.append(f"目標候補: {total}件")
            except Exception:
                pass
        if self._last_expectations is not None:
            exp_summary = get_expectation_summary(self._last_expectations)
            if exp_summary:
                motive_lines.append(f"期待: {exp_summary}")
        # #3 scoped_goal
        if self._scoped_goal_sys is not None:
            sg_summary = get_scoped_goal_summary(self._scoped_goal_sys)
            if sg_summary and sg_summary.get("has_active_scope"):
                scope = sg_summary.get("current_scope", {})
                motive_lines.append(
                    f"スコープ目標: {scope.get('category', '?')} "
                    f"(strength={scope.get('strength', 0):.2f})"
                )
        # #4 transient_goal
        if self._transient_goal_mgr is not None:
            tg_summary = get_transient_goal_summary(self._transient_goal_mgr)
            if tg_summary and tg_summary.get("has_active_goal"):
                goal = tg_summary.get("active_goal", {})
                motive_lines.append(
                    f"一時目標: {goal.get('category', '?')} "
                    f"(strength={goal.get('strength', 0):.2f})"
                )
        # #5 proto_goal_vector
        if self._vector_gen is not None:
            vec_summary = get_vector_summary(self._vector_gen)
            if vec_summary and vec_summary.get("vector_count", 0) > 0:
                motive_lines.append(
                    f"方向ベクトル: {vec_summary['vector_count']}本, "
                    f"最強={vec_summary['strongest_magnitude']:.2f}"
                )
        # #14 policy_candidate_expansion
        if self._policy_expander is not None:
            try:
                exp_text = get_expansion_summary_text(self._policy_expander)
                if exp_text:
                    motive_lines.append(f"候補拡張: {exp_text}")
            except Exception:
                pass
        # #18 spontaneous_activation
        if self._spontaneous_processor is not None:
            try:
                sp_text = get_spontaneous_summary(
                    self._spontaneous_processor.state
                )
                if sp_text and "待機中" not in sp_text:
                    motive_lines.append(f"自発起動: {sp_text}")
            except Exception:
                pass
        if len(motive_lines) > 1:
            sections.append("\n".join(motive_lines))

        # ── 【記憶・内省】 ──
        memory_lines = ["【記憶・内省】"]
        if self._last_episodes is not None:
            ep_summary = get_episodic_memory_summary(self._last_episodes)
            if ep_summary:
                memory_lines.append(f"エピソード記憶: {ep_summary}")
        if self._last_bindings is not None:
            bind_summary = get_binding_summary(self._last_bindings)
            if bind_summary:
                memory_lines.append(f"感情結合: {bind_summary}")
        if self._last_consumption is not None:
            cons_summary = get_consumption_summary(self._last_consumption)
            if cons_summary:
                memory_lines.append(f"内省消費: {cons_summary}")
        if self._last_other_model is not None:
            other_summary = get_other_model_summary(self._last_other_model)
            if other_summary:
                memory_lines.append(f"他者モデル: {other_summary}")
        # #7 introspection_trace
        if self._last_trace is not None:
            trace_str = get_trace_summary(self._last_trace)
            if trace_str:
                memory_lines.append(f"内省: {trace_str}")
        # #15 memory_system_integration
        if self._memory_integrator is not None:
            try:
                int_text = get_integration_summary_text(self._memory_integrator)
                if int_text:
                    memory_lines.append(f"記憶統合: {int_text}")
            except Exception:
                pass
        # #16 other_model_real_feed
        if self._real_feed_processor is not None:
            try:
                feed_text = get_real_feed_summary(self._real_feed_processor)
                if feed_text and "inactive" not in feed_text:
                    memory_lines.append(f"観測フィード: {feed_text}")
            except Exception:
                pass
        # #17 text_dialogue_input
        if self._text_dialogue_processor is not None:
            try:
                tdi_text = get_text_dialogue_summary(
                    self._text_dialogue_processor.state
                )
                if tdi_text and "待機中" not in tdi_text:
                    memory_lines.append(f"入力経路: {tdi_text}")
            except Exception:
                pass
        # #20 memory_forgetting_fixation
        if self._forgetting_fixation_processor is not None:
            try:
                ff_text = get_forgetting_fixation_summary(
                    self._forgetting_fixation_processor.state
                )
                if ff_text and "待機中" not in ff_text:
                    memory_lines.append(f"記憶流動: {ff_text}")
            except Exception:
                pass
        # #21 action_result_observation
        if self._action_result_observer is not None:
            try:
                ar_data = self._action_result_observer.get_enrichment_data()
                ar_text = ar_data.get("summary_text", "")
                if ar_text and "待機中" not in ar_text:
                    memory_lines.append(f"行動-結果: {ar_text}")
            except Exception:
                pass
        # #22 other_model_dialogue_learning
        if self._dialogue_learning_processor is not None:
            try:
                dl_data = self._dialogue_learning_processor.get_enrichment_data()
                dl_text = dl_data.get("summary_text", "")
                if dl_text and "待機中" not in dl_text:
                    memory_lines.append(f"他者蓄積: {dl_text}")
            except Exception:
                pass
        # #23 meta_emotion_cognition
        if self._meta_emotion_processor is not None:
            try:
                me_data = self._meta_emotion_processor.get_enrichment_data()
                me_text = me_data.get("summary_text", "")
                if me_text and "待機中" not in me_text:
                    memory_lines.append(f"メタ感情: {me_text}")
            except Exception:
                pass
        # #24 self_action_perception
        if self._self_action_recorder is not None:
            try:
                sa_data = self._self_action_recorder.get_enrichment_data()
                sa_text = sa_data.get("summary_text", "")
                if sa_text and "待機中" not in sa_text:
                    memory_lines.append(f"自己行動: {sa_text}")
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
                memory_lines.append("\n".join(diff_parts))
        except Exception:
            pass
        # #26 intent_action_gap — 意図-行動対の等価列挙（強調禁止）
        if self._intent_action_gap_recorder is not None:
            try:
                gap_data = self._intent_action_gap_recorder.get_enrichment_data()
                gap_text = gap_data.get("summary_text", "")
                if gap_text and "待機中" not in gap_text:
                    memory_lines.append(f"意図-行動対: {gap_text}")
                # 直近記録をポリシーラベルとテキスト断片の対として等価に列挙
                recent_entries = gap_data.get("recent_entries", [])
                if recent_entries:
                    for entry in recent_entries:
                        label = entry.get("policy_label", "")
                        snippet = entry.get("text_snippet", "")[:40]
                        if label or snippet:
                            memory_lines.append(
                                f"  [{label}] - [{snippet}]"
                            )
            except Exception:
                pass
        if len(memory_lines) > 1:
            sections.append("\n".join(memory_lines))

        # ── 【判断傾向】 ── (Phase 30-35 cached)
        bias_lines = ["【判断傾向】"]
        # #9 decision_bias
        if self._last_decision_bias is not None:
            db = self._last_decision_bias
            bias_lines.append(
                f"判断バイアス: phase={db.dynamics_phase.value}, "
                f"valence={db.valence_bias:.2f}"
            )
        # #11 tone modifier
        if self._last_tone_mod is not None:
            bias_lines.append(
                f"トーン推奨: {self._last_tone_mod.recommended.value}"
            )
        # #12 context_sensitivity
        if self._last_sensitivity_bias is not None:
            sb = self._last_sensitivity_bias
            if sb.caution_level > 0.5:
                bias_lines.append(
                    f"空気読み: caution={sb.caution_level:.2f}"
                )
        # #13 silence/hesitation
        if self._last_has_silence:
            bias_lines.append("沈黙傾向: あり")
        # #19 value_orientation_validation
        if self._vo_validator is not None:
            try:
                vo_text = get_vo_validation_summary(self._vo_validator.state)
                if vo_text and "待機中" not in vo_text:
                    bias_lines.append(f"価値検証: {vo_text}")
            except Exception:
                pass
        if len(bias_lines) > 1:
            sections.append("\n".join(bias_lines))

        # Footer
        sections.append(
            "この内面状態を自然に反映した反応をしてください。"
            "機械的に読み上げないこと。"
        )

        return "\n\n".join(sections)

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

        # Phase 30: thought — 候補ポリシー生成
        candidates = generate_thought_candidates(
            state=self._psyche,
            percept=percept,
            recalled=recalled_memories,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
        )

        # Phase 30b: policy_candidate_expansion — 内面反映候補拡張
        try:
            cs_inputs = self._build_cross_section_inputs(
                percept, recalled_memories, resp_influence, user_id,
            )
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
            "version": 20,
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

            logger.info("Psyche state loaded from %s (v%d, tick=%d)",
                        load_path, data.get("version", 0), self._tick_count)
            return True
        except Exception as e:
            logger.error("Failed to load snapshot: %s", e)
            return False
