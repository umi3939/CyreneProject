"""
psyche/orchestrator_5tick_phases.py - 5ティック帯域のPhase実行コード

orchestrator.pyから物理的に分離された5ティック帯域（Phase 15-26）の処理実行コード。
ロジックは分離前と完全に同一であり、いかなる処理ロジックの変更も行っていない。

各関数は統合管理構造（PsycheOrchestrator）のインスタンスを第一引数として受け取り、
そのインスタンスの既存属性に対してのみ読み書きする。
新しいデータ構造・新しい状態変数・新しいキャッシュは追加しない。

グループ構成:
- グループ1: 自己差分/自己像（Phase 15-18）
- グループ2: ナラティブ/記憶（Phase 19-21f）
- グループ3: 内省/期待（Phase 22-24b）
- グループ4: 他者モデル/相互作用（Phase 25a-25f, 25）
- グループ5: 価値指向/責任（Phase 26-26h）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import PsycheOrchestrator

# ── orchestrator.py から参照されるモジュールレベルインポート ──────────
# これらは orchestrator.py のトップレベルで既にインポートされているが、
# 分離先ファイルでも直接使用するため、同一のインポートを行う。

from .other_model_input_supply import (
    supply_context,
    update_from_percept as update_input_supply,
    decay_buffer,
    supply_reaction_log,
)

from .self_narrative import (
    observe_from_chain as observe_narrative_from_chain,
)

from .episodic_memory import (
    record_from_chain as record_episode_from_chain,
)

from .emotional_memory_binding import (
    bind_from_chain,
)

from .memory_system_integration import (
    IntegrationContext,
)

from .multi_path_recall import (
    EmotionSnapshot as RecallEmotionSnapshot,
    ContextSnapshot as RecallContextSnapshot,
    TemporalSnapshot as RecallTemporalSnapshot,
)

from .spontaneous_recall import (
    InternalEmotionSnapshot as SpontaneousRecallEmotionSnapshot,
)

from .multi_emotion import (
    get_active_emotions,
)

from .introspection_trace import (
    get_trace_summary,
)

from .introspection_consumption import (
    consume_from_chain as consume_introspection_from_chain,
)

from .expectation_formation import (
    form_from_chain as form_expectations_from_chain,
)

from .reference_frequency_description import (
    process_reference_frequency,
)

from .other_model_real_feed import (
    enhance_context_with_feed,
)

from .other_agent_model import (
    observe_from_chain as observe_other_from_chain,
)

from .value_orientation import (
    update_orientation,
    generate_emotion_signal,
    generate_responsibility_signal,
)

from .stabilization_description import (
    process_stabilization_description,
)

from .behavioral_diversity_description import (
    process_behavioral_diversity,
)

from .forgetting_recall_balance import (
    process_forgetting_recall_balance,
)

from .responsibility_dispersion import (
    get_active_units as get_dispersion_active_units,
    get_total_active_weight as get_dispersion_active_weight,
)

logger = logging.getLogger(__name__)


# ── 統括関数 ─────────────────────────────────────────────────────

def run_every_5_ticks(orch: PsycheOrchestrator, user_id: str) -> None:
    """5ティック毎の自己連続性 + 記憶 + 内省フェーズ (Phase 15-26)."""
    _run_5t_self_diff_image(orch, user_id)
    _run_5t_narrative_memory(orch, user_id)
    _run_5t_introspection_expectation(orch, user_id)
    _run_5t_other_model_interaction(orch, user_id)
    _run_5t_value_responsibility(orch, user_id)

    logger.debug(
        "Tick %d every-5: diff=%s, strain=%s, coherence=%s, "
        "episodes=%s, expectations=%s",
        orch._tick_count,
        "ok" if orch._last_diff_summary else "none",
        "ok" if orch._last_strain else "none",
        "ok" if orch._last_coherence else "none",
        "ok" if orch._last_episodes else "none",
        "ok" if orch._last_expectations else "none",
    )


# ── 5-tick group: Self-diff / Self-image (Phase 15-18) ────────

def _run_5t_self_diff_image(orch: PsycheOrchestrator, user_id: str) -> None:
    """Phase 15-18: 自己差分・安定化記述・連続性負荷・自己像・一貫性."""

    # Phase 15: temporal_self_difference — 過去/現在の自己差分
    try:
        if orch._last_self_view is not None:
            orch._temporal_diff_sys.record_snapshot(orch._last_self_view)
            orch._last_diff_summary = orch._temporal_diff_sys.compare_with_reference(
                current=orch._last_self_view,
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
                orch._psyche.emotions.as_dict().values()
            ) if orch._psyche.emotions else 0.0
        except Exception:
            pass

        _sd_stm_count = 0
        try:
            if orch._loop_state and orch._loop_state.memory:
                _sd_stm_count = len(orch._loop_state.memory.entries)
        except Exception:
            pass

        _sd_transient_active = False
        try:
            _sd_transient_active = orch._transient_goal_mgr.state.active_goal is not None
        except Exception:
            pass

        _sd_commitment_count = 0
        try:
            _sd_commitment_count = len([
                it for it in orch._persistent_commitment.state.items
                if not it.released
            ])
        except Exception:
            pass

        _sd_spontaneous = False
        try:
            if orch._last_activation_result is not None:
                _sd_spontaneous = bool(orch._last_activation_result.candidates)
        except Exception:
            pass

        _sd_has_external = orch._last_percept is not None

        orch._stabilization_desc_state = process_stabilization_description(
            orch._stabilization_desc_state,
            emotion_intensity=_sd_emo_intensity,
            stm_entry_count=_sd_stm_count,
            transient_goal_active=_sd_transient_active,
            persistent_commitment_unreleased_count=_sd_commitment_count,
            spontaneous_candidate_exists=_sd_spontaneous,
            has_external_input=_sd_has_external,
            diff_summary=orch._last_diff_summary,
            tick=orch._tick_count,
            config=orch._stabilization_desc_config,
        )
    except Exception as e:
        logger.debug("Stabilization description skipped: %s", e)

    # Phase 16: continuity_strain — 連続性負荷判定
    try:
        if orch._last_diff_summary is not None:
            orch._last_strain = orch._strain_sys.observe_difference(
                orch._last_diff_summary,
            )
    except Exception as e:
        logger.debug("Continuity strain skipped: %s", e)

    # Phase 17: self_image_integration — 暫定自己像生成
    try:
        orch._last_self_image = orch._self_image_sys.generate_image(
            self_state_view=orch._last_self_view,
            tendency_awareness=orch._tendency_awareness,
            difference_summary=orch._last_diff_summary,
            strain_state=orch._last_strain,
        )
    except Exception as e:
        logger.debug("Self-image skipped: %s", e)

    # Phase 18: identity_coherence — 一貫性評価
    try:
        orch._last_coherence = orch._coherence_sys.generate_state(
            self_image=orch._last_self_image,
            difference_summary=orch._last_diff_summary,
            strain_state=orch._last_strain,
            tendency_awareness=orch._tendency_awareness,
            value_orientation=orch._value_orientation,
        )
    except Exception as e:
        logger.debug("Identity coherence skipped: %s", e)


# ── 5-tick group: Narrative / Memory (Phase 19-21f) ─────────

def _run_5t_narrative_memory(orch: PsycheOrchestrator, user_id: str) -> None:
    """Phase 19-21f: ナラティブ・エピソード・記憶統合・忘却・想起."""

    # Phase 19: self_narrative — 自己ナラティブ断片追加
    try:
        ctx_for_narrative = supply_context(orch._input_supply)
        orch._last_narrative = observe_narrative_from_chain(
            system=orch._narrative_sys,
            emotional_state=orch._psyche.emotions,
            short_term_memory=orch._loop_state.memory if orch._loop_state else None,
            tendency_awareness=orch._tendency_awareness,
            difference_summary=orch._last_diff_summary,
            external_context=ctx_for_narrative,
        )
    except Exception as e:
        logger.debug("Self-narrative skipped: %s", e)

    # Phase 20: episodic_memory — エピソード記録
    try:
        ctx_for_episode = supply_context(orch._input_supply)
        orch._last_episodes = record_episode_from_chain(
            system=orch._episodic_sys,
            emotional_state=orch._psyche.emotions,
            short_term_memory=orch._loop_state.memory if orch._loop_state else None,
            tendency_awareness=orch._tendency_awareness,
            difference_summary=orch._last_diff_summary,
            coherence_state=orch._last_coherence,
            narrative_state=orch._last_narrative,
            external_context=ctx_for_episode,
        )
    except Exception as e:
        logger.debug("Episodic memory skipped: %s", e)

    # Phase 21: emotional_memory_binding — 感情記憶紐づけ
    try:
        orch._last_bindings = bind_from_chain(
            system=orch._binding_sys,
            stm=orch._loop_state.memory if orch._loop_state else None,
            emotion=orch._psyche.emotions,
            mood=orch._psyche.mood,
            memories=orch._last_recalled_memories,
            episodes=orch._last_episodes,
        )
    except Exception as e:
        logger.debug("Emotional binding skipped: %s", e)

    # Phase 21b: memory_system_integration — 記憶系統統合
    try:
        int_ctx = IntegrationContext(
            emotions={
                k: getattr(orch._psyche.emotions, k, 0.0)
                for k in ["joy", "sadness", "anger", "fear",
                          "surprise", "disgust", "trust"]
            },
            mood_valence=orch._psyche.mood.valence,
            percept_topics=list(
                getattr(orch._last_percept, 'topics', ()) or ()
            ) if orch._last_percept else [],
            percept_text=getattr(orch._last_percept, 'text', '')
            if orch._last_percept else '',
            percept_intent=getattr(orch._last_percept, 'intent', 'unknown')
            if orch._last_percept else 'unknown',
            current_time=time.time(),
            tick_count=orch._tick_count,
        )
        # [H] action_result → memory_system_integration 新系統認識
        # 行動-結果対を新たな系統として認識可能にする。
        # 二重記録の消去は行わず、同一経験の異なる視点として並立保持する。
        ar_pair_dicts: Optional[list[dict]] = None
        if orch._action_result_observer is not None:
            try:
                active = orch._action_result_observer.get_active_pairs()
                if active:
                    ar_pair_dicts = [p.to_dict() for p in active]
            except Exception:
                pass
        orch._last_integration_result = orch._memory_integrator.integrate(
            episodes=orch._last_episodes,
            long_term_memories=orch._last_recalled_memories,
            bindings=orch._last_bindings,
            context=int_ctx,
            action_result_pairs=ar_pair_dicts,
        )
    except Exception as e:
        logger.debug("Memory integration skipped: %s", e)

    # Phase 21c: memory_forgetting_fixation — 記憶の忘却と固定化
    try:
        ff_inputs = orch._build_forgetting_fixation_inputs()
        orch._last_forgetting_fixation = (
            orch._forgetting_fixation_processor.process(ff_inputs)
        )
    except Exception as e:
        logger.debug("Memory forgetting/fixation skipped: %s", e)

    # Phase 21d: multi_path_recall — 記憶の多経路想起
    try:
        # unified_units: memory_system_integrationの結果候補
        recall_units: list = []
        if orch._last_integration_result is not None:
            recall_units = list(orch._last_integration_result.candidates)

        # emotion_snapshot
        emo_snap = RecallEmotionSnapshot()
        try:
            active_emos = get_active_emotions(orch._psyche.emotions)
            dominant_label = ""
            dominant_intensity = 0.0
            for emo_name, emo_val in active_emos:
                if emo_val > dominant_intensity:
                    dominant_intensity = emo_val
                    dominant_label = emo_name
            emo_snap = RecallEmotionSnapshot(
                emotions={name: val for name, val in active_emos},
                mood_valence=orch._psyche.emotions.mood,
                dominant_emotion=dominant_label,
            )
        except Exception:
            pass

        # context_snapshot
        ctx_snap = RecallContextSnapshot(current_time=time.time())
        try:
            if orch._last_percept is not None:
                ctx_snap = RecallContextSnapshot(
                    topics=list(getattr(orch._last_percept, "topics", []) or []),
                    percept_text=getattr(orch._last_percept, "text", "") or "",
                    current_time=time.time(),
                )
        except Exception:
            pass

        # temporal_snapshot
        temp_snap = RecallTemporalSnapshot(tick_count=orch._tick_count)
        try:
            if orch._temporal_cognition is not None:
                tc_data = orch._temporal_cognition.get_enrichment_data()
                temp_snap = RecallTemporalSnapshot(
                    snapshot=tc_data.get("sections", {}),
                    tick_count=orch._tick_count,
                )
        except Exception:
            pass

        # binding_store
        binding_store = orch._last_bindings

        # forgetting_state
        forgetting_state = None
        if orch._forgetting_fixation_processor is not None:
            forgetting_state = orch._forgetting_fixation_processor.state

        orch._multi_path_recall.recall_all_paths(
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
        if orch._last_integration_result is not None:
            sr_units = list(orch._last_integration_result.candidates)

        # emotion_snapshot (現在の感情断面)
        sr_emo_snap = SpontaneousRecallEmotionSnapshot()
        try:
            sr_active_emos = get_active_emotions(orch._psyche.emotions)
            sr_dominant_label = ""
            sr_dominant_intensity = 0.0
            for emo_name, emo_val in sr_active_emos:
                if emo_val > sr_dominant_intensity:
                    sr_dominant_intensity = emo_val
                    sr_dominant_label = emo_name
            sr_emo_snap = SpontaneousRecallEmotionSnapshot(
                emotions={name: val for name, val in sr_active_emos},
                mood_valence=orch._psyche.emotions.mood,
                dominant_emotion=sr_dominant_label,
            )
        except Exception:
            pass

        # binding_store
        sr_binding_store = orch._last_bindings

        # forgetting_state
        sr_forgetting_state = None
        if orch._forgetting_fixation_processor is not None:
            sr_forgetting_state = orch._forgetting_fixation_processor.state

        # motive_store (内的動機)
        sr_motive_store = orch._last_motives

        # strain_state (連続性の揺らぎ)
        sr_strain_state = orch._last_strain

        # direction_vectors (方向ベクトル)
        sr_direction_vectors = None
        try:
            if orch._vector_gen is not None:
                sr_direction_vectors = orch._vector_gen.state
        except Exception:
            pass

        # temporal_snapshot (時間認知)
        sr_temporal_snapshot = None
        try:
            if orch._temporal_cognition is not None:
                sr_temporal_snapshot = orch._temporal_cognition.get_enrichment_data()
        except Exception:
            pass

        orch._spontaneous_recall.process(
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
        if orch._forgetting_fixation_processor is not None:
            frb_forgetting_state = orch._forgetting_fixation_processor.state

        frb_forgetting_result = orch._last_forgetting_fixation

        frb_multi_path_recall_state = None
        if orch._multi_path_recall is not None:
            frb_multi_path_recall_state = orch._multi_path_recall.state

        frb_spontaneous_recall_state = None
        if orch._spontaneous_recall is not None:
            frb_spontaneous_recall_state = orch._spontaneous_recall.state

        orch._frb_state = process_forgetting_recall_balance(
            orch._frb_state,
            forgetting_state=frb_forgetting_state,
            forgetting_result=frb_forgetting_result,
            multi_path_recall_state=frb_multi_path_recall_state,
            spontaneous_recall_state=frb_spontaneous_recall_state,
            config=orch._frb_config,
        )
    except Exception as e:
        logger.debug("Forgetting-recall balance skipped: %s", e)


# ── 5-tick group: Introspection / Expectation (Phase 22-24b) ─

def _run_5t_introspection_expectation(orch: PsycheOrchestrator, user_id: str) -> None:
    """Phase 22-24b: 内省・消費・期待形成・参照頻度."""

    # Phase 22: introspection_trace — 内省ログ生成
    # self_action_perception → introspection_trace 間接経路:
    # 自己行動知覚の直近記録からテキスト存在事実・長さ・ポリシーラベル・ティックのみを
    # context パラメータとして渡す。テキスト本文は含めない（自己強化ループ遮断）。
    try:
        resp_influence = orch._responsibility_mgr.get_influence(user_id)
        # --- self_action context 構築 ---
        sa_context: dict[str, Any] = {}
        if orch._self_action_recorder is not None:
            try:
                sa_record = orch._self_action_recorder.get_latest_record()
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
        orch._last_trace = orch._introspection_sys.generate_trace(
            emotion_state=orch._psyche.emotions,
            responsibility_state=resp_influence,
            value_orientation=orch._value_orientation,
            fear_index=orch._psyche.fear_index,
            context=sa_context if sa_context else None,
        )
    except Exception as e:
        logger.debug("Introspection trace skipped: %s", e)

    # Phase 23: introspection_consumption — 内省消費・再構成
    try:
        trace_summary = get_trace_summary(orch._last_trace) if orch._last_trace else None
        orch._last_consumption = consume_introspection_from_chain(
            system=orch._consumption_sys,
            introspection_summary=trace_summary,
            narrative_state=orch._last_narrative,
            coherence_state=orch._last_coherence,
            tendency_awareness=orch._tendency_awareness,
            episodic_store=orch._last_episodes,
        )
    except Exception as e:
        logger.debug("Introspection consumption skipped: %s", e)

    # Phase 24: expectation_formation — 期待形成
    try:
        tendency_bias = (
            orch._tendency_sys.state.tendencies
            if orch._tendency_sys else None
        )
        orch._last_expectations = form_expectations_from_chain(
            system=orch._expectation_sys,
            tendency_bias=tendency_bias,
            difference_summary=orch._last_diff_summary,
            narrative_state=orch._last_narrative,
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
        orch._reference_frequency_state = process_reference_frequency(
            orch._reference_frequency_state,
            episodic_store=orch._last_episodes,
            binding_store=orch._last_bindings,
            consumption_store=orch._last_consumption,
            expectation_store=orch._last_expectations,
            motive_store=orch._last_motives,
            narrative_state=orch._last_narrative,
            other_model_store=orch._last_other_model,
            self_reference_state=orch._self_ref_state,
            action_result_state=(
                orch._action_result_observer.state
                if orch._action_result_observer else None
            ),
            dialogue_learning_state=(
                orch._dialogue_learning_processor.state
                if orch._dialogue_learning_processor else None
            ),
            forgetting_state=(
                orch._forgetting_fixation_processor.state
                if orch._forgetting_fixation_processor else None
            ),
            multi_path_recall_state=(
                orch._multi_path_recall.state
                if orch._multi_path_recall else None
            ),
            spontaneous_recall_state=(
                orch._spontaneous_recall.state
                if orch._spontaneous_recall else None
            ),
            config=orch._reference_frequency_config,
        )
    except Exception as e:
        logger.debug("Reference frequency description skipped: %s", e)


# ── 5-tick group: Other-model / Interaction (Phase 25a-25f, 25)

def _run_5t_other_model_interaction(orch: PsycheOrchestrator, user_id: str) -> None:
    """Phase 25a-25f, 25: 他者モデル・対話学習・相互作用・境界・仮説."""

    # Phase 25a: other_model_real_feed — 実対話由来の観測断片抽出・正規化
    try:
        orch._last_feed_result = orch._real_feed_processor.process(
            percept=orch._last_percept,
            stm=orch._loop_state.memory if orch._loop_state else None,
            psyche=orch._psyche,
            dynamics=orch._dynamics,
            recalled_memories=orch._last_recalled_memories,
            integration_result=orch._last_integration_result,
            tick_count=orch._tick_count,
        )
        # [D] action_result → other_model_real_feed 時系列隣接記録供給
        # 「この行動の後にこの他者反応が観測された」という時系列的隣接の記録を供給。
        # 因果帰属は行わない。他者の反応は他者自身の内部状態にも依存する。
        if orch._action_result_observer is not None:
            try:
                from .other_model_real_feed import (
                    ObservationFragment as RealFeedFragment,
                    ObservationFragmentType as RealFeedFragmentType,
                )
                active_pairs = orch._action_result_observer.get_active_pairs()
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
                    orch._real_feed_processor.inject_external_fragments(external_frags)
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
        dl_inputs = orch._build_dialogue_learning_inputs(user_id)
        orch._last_dialogue_learning = orch._dialogue_learning_processor.tick(dl_inputs)
    except Exception as e:
        logger.debug("Dialogue learning skipped: %s", e)

    # Phase 25d: interaction_accumulation — 相互作用の蓄積記述
    # 自己行動知覚の記録と他者モデルリアルフィードの観測記録を時系列的隣接関係として
    # 対構成し蓄積する。自己行動知覚の記録更新と他者モデルリアルフィードの観測更新が
    # 完了した後に実行する。因果帰属を行わない。パターン抽出を行わない。
    # 出力はenrichmentへの等価列挙と内省系構造へのREAD-ONLY参照のみ。
    try:
        ia_self_records = []
        if orch._self_action_recorder is not None:
            ia_self_records = orch._self_action_recorder.get_reference_history()
        ia_other_units = []
        if orch._real_feed_processor is not None:
            ia_other_units = list(orch._real_feed_processor.state.units)
        orch._interaction_accumulation.process(
            self_records=ia_self_records,
            other_units=ia_other_units,
            current_tick=orch._tick_count,
        )
    except Exception as e:
        logger.debug("Interaction accumulation skipped: %s", e)

    # Phase 25e: other_boundary_accumulation — 他者境界の多相蓄積
    # other_agent_modelが前ティックで生成したSelfOtherBoundaryのリストを
    # READ-ONLY参照し、相手別に蓄積する。境界の乖離度を制御・調整・最適化しない。
    # 蓄積された推移からパターンを抽出しない。判断・行動選択に接続しない。
    try:
        oba_boundaries = []
        if orch._last_other_model is not None:
            oba_boundaries = list(orch._last_other_model.boundaries)
        # 最新の境界情報（あれば1件）を蓄積対象とする
        oba_boundary = oba_boundaries[-1] if oba_boundaries else None
        orch._last_boundary_accumulation = orch._other_boundary_accumulation.tick(
            boundary=oba_boundary,
            user_id=user_id,
            current_tick=orch._tick_count,
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
        hop_hypothesis_source = orch._last_other_model if orch._last_other_model is not None else None
        hop_observation_source = orch._real_feed_processor if orch._real_feed_processor is not None else None
        orch._hypothesis_observation_pairing.process(
            hypothesis_source=hop_hypothesis_source,
            observation_source=hop_observation_source,
            user_id_source=user_id,
            current_cycle=orch._tick_count,
        )
    except Exception as e:
        logger.debug("Hypothesis-observation pairing skipped: %s", e)

    # Phase 25: other_agent_model — 他者モデル仮説更新 (入力供給経由)
    try:
        # 入力供給更新: percept / STM / dynamics / psyche から計算
        orch._input_supply = update_input_supply(
            state=orch._input_supply,
            percept=orch._last_percept,
            stm=orch._loop_state.memory if orch._loop_state else None,
            dynamics=orch._dynamics,
            psyche=orch._psyche,
        )
        orch._input_supply = decay_buffer(orch._input_supply, time.time())

        # 供給: context snapshot + reaction log
        ctx = supply_context(orch._input_supply)

        # リアルフィードで context を差分調整
        if orch._last_feed_result is not None:
            ctx = enhance_context_with_feed(ctx, orch._last_feed_result)

        # [G] dialogue_learning → input_supply 長期蓄積補足
        # 蓄積記述の概要を、既存の文脈生成処理を上書きするのではなく、
        # 長期蓄積由来の補足情報として追加する。
        if orch._dialogue_learning_processor is not None:
            try:
                dl_data = orch._dialogue_learning_processor.get_enrichment_data()
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
        if orch._dialogue_learning_processor is not None and orch._last_dialogue_learning is not None:
            try:
                dl_data = orch._dialogue_learning_processor.get_enrichment_data()
                materials = dl_data.get("material_count", 0)
                supply_str = dl_data.get("supply_strength", 0.0)
                if materials > 0:
                    # 仮説材料分布を参照情報として付加
                    # weight: 蓄積由来情報の相対的比重を微弱に上方修正
                    weight_boost = min(0.1, materials * 0.005) * supply_str
                    ctx.weight = min(1.0, ctx.weight + weight_boost)
            except Exception:
                pass

        rlog = supply_reaction_log(orch._input_supply)

        orch._last_other_model = observe_other_from_chain(
            system=orch._other_model_sys,
            external_context=ctx,
            reaction_log=rlog,
            self_state=orch._last_self_view,
        )
    except Exception as e:
        logger.debug("Other agent model skipped: %s", e)


# ── 5-tick group: Value-orientation / Responsibility (Phase 26-26h)

def _run_5t_value_responsibility(orch: PsycheOrchestrator, user_id: str) -> None:
    """Phase 26-26h: 価値指向・行動結果・多様性・乖離・ライフサイクル・責任・階層."""

    # Phase 26: value_orientation — 価値指向更新（遅い変化）
    try:
        emo_signal = generate_emotion_signal(orch._psyche.emotions)
        resp_influence_for_vo = orch._responsibility_mgr.get_influence(user_id)
        resp_signal = generate_responsibility_signal(
            total_weight=resp_influence_for_vo.caution_bias,
        )
        # [A] action_result → value_orientation シグナル供給
        # 蓄積された行動-結果対の情報を微弱なシグナルとして供給。
        # シグナル強度は既存シグナル（emo_signal, resp_signal）を超えない上限を持つ。
        ar_signal: Optional[dict[str, float]] = None
        if orch._action_result_observer is not None and orch._last_action_result is not None:
            try:
                ar_data = orch._action_result_observer.get_enrichment_data()
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

        orch._value_orientation = update_orientation(
            orientation=orch._value_orientation,
            emotion_signal=emo_signal if emo_signal else None,
            responsibility_signal=resp_signal if resp_signal else None,
        )
        # action_result由来シグナルを分離供給（3番目のシグナルとして追加）
        # update_orientationは3信号を平均するため、別途供給して影響を微弱に保つ
        if ar_signal:
            orch._value_orientation = update_orientation(
                orientation=orch._value_orientation,
                decision_signal=ar_signal,
            )
    except Exception as e:
        logger.debug("Value orientation skipped: %s", e)

    # Phase 26b: value_orientation_validation — 価値方向性の実運用検証
    try:
        vo_inputs = orch._build_vo_validation_inputs(user_id)
        orch._last_vo_validation = orch._vo_validator.process(vo_inputs)
    except Exception as e:
        logger.debug("Value orientation validation skipped: %s", e)

    # Phase 26c: action_result_observation — 行動-結果対の処理
    # 結果記述の結合は低頻度処理帯で行い、同一周期内での即時構成を禁止する。
    try:
        ar_inputs = orch._build_action_result_inputs(user_id)
        orch._last_action_result = orch._action_result_observer.process(ar_inputs)
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
        orch._behavioral_diversity_state = process_behavioral_diversity(
            orch._behavioral_diversity_state,
            action_result_state=orch._action_result_observer,
            selection_attribution_state=orch._selection_attribution_recorder,
            tick=orch._tick_count,
            config=orch._behavioral_diversity_config,
        )
    except Exception as e:
        logger.debug("Behavioral diversity description skipped: %s", e)

    # Phase 26d: [C] action_result → expectation_formation 差分照合
    # 蓄積された行動-結果対と予期情報との差分を記録する。
    # 照合結果は「差分の認知」として記録にとどめる。
    # 予期が「当たった/外れた」という評価はしない。
    # 照合結果を判断系に直接接続しない。
    try:
        if (orch._last_expectations is not None
                and orch._action_result_observer is not None):
            active_pairs = orch._action_result_observer.get_active_pairs()
            expectations = getattr(orch._last_expectations, 'expectations', ())
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
                                "tick": orch._tick_count,
                            },
                        })
                # 差分記録を保持（上限付き）
                if not hasattr(orch, '_expectation_action_diff_log'):
                    orch._expectation_action_diff_log: list[dict] = []
                orch._expectation_action_diff_log.extend(diff_records)
                orch._expectation_action_diff_log = orch._expectation_action_diff_log[-50:]
    except Exception as e:
        logger.debug("Expectation-action diff skipped: %s", e)

    # Phase 26e: intent_action_gap — 意図-行動間の乖離認知
    # 自己行動知覚(notify_self_output)の後の帯で、自己行動記録の最新記録と
    # ポリシー選択情報を入力として処理を呼び出す。
    # 乖離記録→ポリシー選択(Phase 30-35)への接続禁止。
    # 乖離記録→行動-結果観測への接続禁止。
    # 乖離記録→予期形成への接続禁止。
    try:
        latest_action = orch._self_action_recorder.get_latest_record()
        if latest_action is not None:
            orch._intent_action_gap_recorder.process_action_record(
                response_text=latest_action.response_text,
                policy_label=latest_action.policy_label,
                tick=latest_action.tick,
                context_info=orch._last_selected_policy_axis or "",
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
        orch._expectation_lifecycle_processor.process(orch._last_expectations)
    except Exception as e:
        logger.debug("Expectation lifecycle description skipped: %s", e)

    # Phase 26g: responsibility_temporal_trace — 責任の時間的推移記述
    # 責任管理構造と責任分散構造からREAD-ONLYで値を読み取り、
    # スナップショットとして時系列蓄積し、変動度合いを段階値で記述する。
    # 責任管理構造・責任分散構造への書き込み経路を一切持たない（READ-ONLY）。
    # 段階値を判断バイアス計算・方針選択・安定弁に接続しない。
    # パターン抽出禁止、統計量算出禁止、方向性記述排除。
    try:
        resp_summary = orch._responsibility_mgr.get_summary(user_id)
        disp_active_weight = 0.0
        disp_active_count = 0
        disp_transformation_count = 0
        if orch._dispersion_state is not None:
            disp_active_weight = get_dispersion_active_weight(orch._dispersion_state)
            disp_active_count = len(get_dispersion_active_units(orch._dispersion_state))
            disp_transformation_count = orch._dispersion_state.transformation_count

        orch._responsibility_temporal_trace.record_snapshot(
            tick=orch._tick_count,
            total_weight=resp_summary["total_weight"],
            pending_decisions=resp_summary["pending_decisions"],
            accumulated_harm=resp_summary["accumulated_harm"],
            accumulated_confidence=resp_summary["accumulated_confidence"],
            dispersion_active_weight=disp_active_weight,
            dispersion_active_count=disp_active_count,
            dispersion_transformation_count=disp_transformation_count,
        )
        orch._responsibility_temporal_trace.describe_variation()
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
            active_goal = orch._transient_goal_mgr.state.active_goal
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
            pc = orch._persistent_commitment
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
            vo = orch._value_orientation
            vo_data = {
                "dimensions": vo.get_all_dimensions(),
                "confidences": vo.get_all_confidences(),
                "update_count": vo.update_count,
            }
        except Exception:
            pass

        orch._goal_hierarchy_propagation.process(
            transient_goal_data=tg_data if tg_data else None,
            persistent_commitment_data=pc_data if pc_data else None,
            value_orientation_data=vo_data if vo_data else None,
        )
    except Exception as e:
        logger.debug("Goal hierarchy propagation skipped: %s", e)
