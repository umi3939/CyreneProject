"""
psyche/orchestrator.py - PsycheOrchestrator: 全モジュール統合管理

全 psyche モジュールの初期化・毎ティック実行・プロンプト生成・永続化を一元管理する。
brain.py からは本クラスのみを参照すればよく、個別モジュールへの直接依存を排除する。

実行モデル:
- 毎ティック: react_with_stm, dynamics, attachment, responsibility, self_reference,
              repeated_tendency, fear_recompute
- 3ティック毎: tendency_awareness → self_model → goals → intrinsic_motivation
- 5ティック毎: temporal_diff → strain → self_image → coherence → narrative →
               episodic → binding → introspection → consumption → expectation →
               other_model → value_orientation
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
    get_orientation_summary,
)

# Repeated tendency
from .repeated_tendency import RepeatedTendencySystem
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

# Proto-goal vector
from .proto_goal_vector import (
    VectorGenerator,
    ProtoGoalVector,
    get_vector_summary,
)

# Goal candidates
from .goal_candidates import (
    CandidateGenerator,
    GoalCandidate,
    get_candidate_summary,
)

# Transient goal
from .transient_goal import (
    TransientGoalManager,
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

# Snapshot / Persistence
from .snapshot import Snapshot

# Dispersion
from .responsibility_dispersion import DispersionState, create_dispersion_state


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
                "id": "entertain",
                "description": "視聴者を楽しませる",
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

        logger.info(
            "PsycheOrchestrator initialized: fear=%.2f, dominant=%s, "
            "systems=35",
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

    # ── Phase 1-7: Every-tick update ──────────────────────────────

    def _run_every_tick(
        self,
        percept: Percept,
        delta_time: float,
        user_id: str,
    ) -> None:
        """毎ティック実行するフェーズ (Phase 1-7)."""

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

        # Phase 3: attachment — 視聴者ボンド更新
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

        logger.info(
            "Tick %d every-tick: emotion=%s, mood=%.2f, fear=%.2f, "
            "dynamics=%s",
            self._tick_count,
            percept.emotion or "neutral",
            self._psyche.mood.valence,
            self._psyche.fear_level,
            get_dynamics_summary(self._dynamics),
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

        logger.info(
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
            self._last_narrative = observe_narrative_from_chain(
                system=self._narrative_sys,
                emotional_state=self._psyche.emotions,
                short_term_memory=self._loop_state.memory if self._loop_state else None,
                tendency_awareness=self._tendency_awareness,
                difference_summary=self._last_diff_summary,
                external_context=None,
            )
        except Exception as e:
            logger.debug("Self-narrative skipped: %s", e)

        # Phase 20: episodic_memory — エピソード記録
        try:
            self._last_episodes = record_episode_from_chain(
                system=self._episodic_sys,
                emotional_state=self._psyche.emotions,
                short_term_memory=self._loop_state.memory if self._loop_state else None,
                tendency_awareness=self._tendency_awareness,
                difference_summary=self._last_diff_summary,
                coherence_state=self._last_coherence,
                narrative_state=self._last_narrative,
                external_context=None,
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
                memories=None,
                episodes=self._last_episodes,
            )
        except Exception as e:
            logger.debug("Emotional binding skipped: %s", e)

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

        # Phase 25: other_agent_model — 他者モデル仮説更新
        try:
            self._last_other_model = observe_other_from_chain(
                system=self._other_model_sys,
                external_context=None,
                reaction_log=None,
                self_state=self._last_self_view,
            )
        except Exception as e:
            logger.debug("Other agent model skipped: %s", e)

        # Phase 26: value_orientation — 価値指向更新（遅い変化）
        try:
            self._value_orientation = update_orientation(
                orientation=self._value_orientation,
                signal_type="emotion",
                signal_value=self._psyche.mood.valence,
            )
        except Exception as e:
            logger.debug("Value orientation skipped: %s", e)

        logger.info(
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
        logger.info(
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
            user_id: 視聴者ID
        """
        self._tick_count += 1

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

    # ── Prompt enrichment ─────────────────────────────────────────

    def get_prompt_enrichment(self) -> str:
        """Gemini プロンプト用の心理状態テキストを生成する。

        brain.py の _format_psyche_for_prompt を置き換える。
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
        if len(motive_lines) > 1:
            sections.append("\n".join(motive_lines))

        # Footer
        sections.append(
            "この内面状態を自然に反映した反応をしてください。"
            "機械的に読み上げないこと。"
        )

        return "\n\n".join(sections)

    # ── Policy suggestions ────────────────────────────────────────

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
            user_id: 視聴者ID

        Returns:
            【行動方針候補】セクションのテキスト
        """
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

        # Phase 32: tone — トーン修飾子計算
        tone_mod = compute_tone_bias(
            state=self._psyche,
            responsibility_influence=resp_influence,
            decision_bias=decision_bias,
            config=self._tone_config,
        )

        # Phase 33: context_sensitivity — 空気読みバイアス
        ext_ctx = create_neutral_context()
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
            logger.info(
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
            "version": 3,
            "tick_count": self._tick_count,
            "psyche": self._psyche.to_dict(),
            "loop_state": self._loop_state.to_dict() if self._loop_state else {},
            "dynamics": self._dynamics.to_dict() if self._dynamics else {},
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
            logger.info("Psyche state loaded from %s (tick=%d)", load_path, self._tick_count)
            return True
        except Exception as e:
            logger.error("Failed to load snapshot: %s", e)
            return False
