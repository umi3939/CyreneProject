"""
psyche/orchestrator_1tick_phases.py - 毎ティック帯域のPhase実行コード

orchestrator.pyから物理的に分離された毎ティック帯域（Phase 1-7系列）の処理実行コード。
ロジックは分離前と完全に同一であり、いかなる処理ロジックの変更も行っていない。

各関数は統合管理構造（PsycheOrchestrator）のインスタンスを第一引数として受け取り、
そのインスタンスの既存属性に対してのみ読み書きする。
新しいデータ構造・新しい状態変数・新しいキャッシュは追加しない。

グループ構成:
- グループ1: 感情コア処理（Phase 1-2c）— 感情更新→ダイナミクス→振幅→独立減衰→STM連携
- グループ2: 対話・責任・自己参照（Phase 3-6）— ボンド更新→責任記録→自己参照→傾向観測
- グループ3: 恐怖・観測・記述（Phase 7-7f）— 恐怖再計算→行動結果記録→時間認知→知覚文脈→自己呈示→経路均衡→注意配分
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

from .state import Percept

from .reaction import MoodContextInputs, DriveContextInputs
from .reaction_with_stm import react_with_stm

from .dynamics import (
    update_dynamics,
    get_dynamics_summary,
)

from .emotion_amplitude import (
    update_amplitude,
    decay_amplitude,
)

from .multi_emotion import (
    apply_independent_decay,
)

from .stm_emotion_coupling import (
    apply_stm_coupling,
)

from . import attachment_manager

from .self_reference import (
    execute_self_reference,
)

from .action_result_observation import (
    ActionResultInputs,
)

from .input_pathway_balance import (
    process_input_pathway_balance,
    PATHWAY_TEXT,
)

from .attention_distribution_description import (
    process_attention_distribution,
)

logger = logging.getLogger(__name__)


# ── 統括関数 ─────────────────────────────────────────────────────

def run_every_tick(
    orch: PsycheOrchestrator,
    percept: Percept,
    delta_time: float,
    user_id: str,
) -> None:
    """毎ティック帯域の統括実行関数 (Phase 1-7f).

    統合管理構造インスタンスと知覚入力と経過時間と対話相手IDを受け取り、
    内部のグループ関数を順序通り呼び出す。
    """
    # Preserve percept for 5-tick phase input supply
    orch._last_percept = percept

    _run_1t_emotion_core(orch, percept, delta_time, user_id)
    _run_1t_dialogue_responsibility_self(orch, percept, user_id)
    _run_1t_fear_observation_description(orch, percept, delta_time, user_id)

    logger.debug(
        "Tick %d every-tick: emotion=%s, mood=%.2f, fear=%.2f, "
        "dynamics=%s (accumulated=%.2f)",
        orch._tick_count,
        percept.emotion or "neutral",
        orch._psyche.mood.valence,
        orch._psyche.fear_level,
        get_dynamics_summary(orch._dynamics),
        orch._amplitude_state.current_amplitude,
    )


# ── ヘルパー関数 ─────────────────────────────────────────────────

def build_mood_context(
    orch: PsycheOrchestrator,
    resp_influence: Optional[Any] = None,
) -> MoodContextInputs:
    """ムード自律更新のための入力コンテキストを構築する。

    各入力源は利用可能な場合のみ設定され、
    利用不能な場合はデフォルト値（寄与ゼロ）のまま残される。
    """
    ctx = MoodContextInputs(
        emotions=orch._psyche.emotions.as_dict(),
        drives=orch._psyche.drives.as_dict(),
        current_valence=orch._psyche.mood.valence,
        current_arousal=orch._psyche.mood.arousal,
        fear_level=orch._psyche.fear_level,
    )

    # 責任影響
    if resp_influence is not None:
        ctx.responsibility_anxiety = getattr(
            resp_influence, "anxiety_baseline", 0.0
        )

    # 目的階層の存在情報
    try:
        tg_state = orch._transient_goal_mgr.state
        ctx.has_transient_goal = tg_state.active_goal is not None
    except Exception:
        pass

    try:
        active_items = [
            it for it in orch._persistent_commitment._state.items
            if not it.released
        ]
        ctx.persistent_commitment_count = len(active_items)
    except Exception:
        pass

    try:
        ctx.has_scoped_goal = orch._scoped_goal_sys.has_active_scope
    except Exception:
        pass

    # 時間認知の段階値
    try:
        snapshot = orch._temporal_cognition.get_snapshot()
        activity = snapshot.get("activity_density")
        if activity:
            ctx.time_density_label = activity
    except Exception:
        pass

    # 感情帰還方向連続性由来の追従速度変調量を注入
    try:
        v_mod, a_mod = orch._memory_emotion_return.get_tracking_speed_modulation(
            current_tracking_speed_valence=0.10,
            current_tracking_speed_arousal=0.10,
        )
        if v_mod > 0.0:
            ctx.emotion_return_tracking_speed_modulation_valence = v_mod
        if a_mod > 0.0:
            ctx.emotion_return_tracking_speed_modulation_arousal = a_mod
    except Exception:
        pass

    return ctx


def build_drive_context(
    orch: PsycheOrchestrator,
) -> DriveContextInputs:
    """ドライブ変動係数導出のための入力コンテキストを構築する。

    行動多様性記述構造と内部矛盾並置記述構造の最新状態をREAD-ONLYで参照し、
    コンテキストフィールドに注入する。
    各入力源は利用可能な場合のみ設定され、
    利用不能な場合はNone（安全弁5: 寄与ゼロ）のまま残される。
    """
    ctx = DriveContextInputs()

    # 目的階層の存在情報
    try:
        tg_state = orch._transient_goal_mgr.state
        ctx.has_transient_goal = tg_state.active_goal is not None
    except Exception:
        pass

    try:
        active_items = [
            it for it in orch._persistent_commitment._state.items
            if not it.released
        ]
        ctx.persistent_commitment_count = len(active_items)
    except Exception:
        pass

    try:
        ctx.has_scoped_goal = orch._scoped_goal_sys.has_active_scope
    except Exception:
        pass

    # 時間認知の段階値
    try:
        snapshot = orch._temporal_cognition.get_snapshot()
        activity = snapshot.get("activity_density")
        if activity:
            ctx.time_density_label = activity
    except Exception:
        pass

    # 断面6: 行動多様性記述の結果断面キー種類数の段階値（READ-ONLY参照）
    try:
        latest_record = orch._behavioral_diversity_state.latest_record
        if latest_record is not None:
            ctx.behavioral_diversity_stage_value = (
                latest_record.section_key_type_count_level
            )
    except Exception:
        pass

    # 断面8: 行動多様性記述の3つの段階値（結果多様性帰還経路用, READ-ONLY参照）
    try:
        latest_record = orch._behavioral_diversity_state.latest_record
        if latest_record is not None:
            ctx.result_diversity_section_key_level = (
                latest_record.section_key_type_count_level
            )
            ctx.result_diversity_selection_label_level = (
                latest_record.policy_label_type_count_level
            )
            ctx.result_diversity_candidate_variance_level = (
                latest_record.candidate_size_dispersion_level
            )
    except Exception:
        pass

    # 断面7: 内部矛盾並置記述の直前サイクル矛盾対件数（READ-ONLY参照）
    try:
        prev = orch._contradiction_processor.state.previous_contradictions
        ctx.contradiction_count = len(prev)
    except Exception:
        pass

    return ctx


# ── 1-tick group 1: Emotion core (Phase 1-2c) ───────────────────

def _run_1t_emotion_core(
    orch: PsycheOrchestrator,
    percept: Percept,
    delta_time: float,
    user_id: str,
) -> None:
    """Phase 1-2c: 感情更新→ダイナミクス→振幅→独立減衰→STM連携."""

    # Phase 1: react_with_stm — 感情更新 + STM残留 + ムード自律更新
    resp_influence = orch._responsibility_mgr.get_influence(user_id)

    # Construct mood context for autonomous mood update
    mood_ctx = build_mood_context(orch, resp_influence)

    # Construct drive context for state-dependent drive dynamics
    drive_ctx = build_drive_context(orch)

    new_psyche, new_loop, loop_result = react_with_stm(
        percept=percept,
        psyche_state=orch._psyche,
        loop_state=orch._loop_state,
        delta_time=delta_time,
        responsibility_influence=resp_influence,
        mood_context=mood_ctx,
        drive_context=drive_ctx,
    )
    orch._psyche = new_psyche
    orch._loop_state = new_loop

    # Phase 2: dynamics — ピーク/リバウンド判定
    emo_dict = orch._psyche.emotions.as_dict()
    residue_total = loop_result.residue_influence.total_intensity
    orch._dynamics = update_dynamics(
        state=orch._dynamics,
        current_emotions=emo_dict,
        residue_intensity=residue_total,
    )

    # Phase 2a: emotion_amplitude — dynamics相による振幅計算
    max_emo = max(emo_dict.values()) if emo_dict else 0.0
    orch._amplitude_state = update_amplitude(
        orch._amplitude_state,
        intensity_factor=residue_total,
        emotion_intensity=max_emo,
    )
    orch._amplitude_state = decay_amplitude(orch._amplitude_state, delta_time)

    # Phase 2b: multi_emotion — 感情別独立減衰
    try:
        decayed = apply_independent_decay(
            orch._psyche.emotions,
            delta_time=delta_time,
            config=orch._multi_emotion_config,
        )
        orch._psyche = orch._psyche.model_copy(update={"emotions": decayed})
    except Exception as e:
        logger.debug("Multi-emotion decay skipped: %s", e)

    # Phase 2c: stm_emotion_coupling — STM残留→再活性化・蓄積
    if orch._loop_state and orch._loop_state.memory:
        try:
            coupled, orch._last_coupling = apply_stm_coupling(
                emotions=orch._psyche.emotions,
                stm=orch._loop_state.memory,
                delta_time=delta_time,
                config=orch._stm_coupling_config,
                apply_persistence=False,  # multi_emotion handles decay
            )
            orch._psyche = orch._psyche.model_copy(update={"emotions": coupled})
        except Exception as e:
            logger.debug("STM-emotion coupling skipped: %s", e)


# ── 1-tick group 2: Dialogue / Responsibility / Self-reference (Phase 3-6) ──

def _run_1t_dialogue_responsibility_self(
    orch: PsycheOrchestrator,
    percept: Percept,
    user_id: str,
) -> None:
    """Phase 3-6: ボンド更新→責任記録→自己参照→傾向観測."""

    # Phase 3: attachment — 対話相手ボンド更新
    if orch._psyche.attachment is not None:
        valence = abs(percept.emotion_valence)
        event_type = "positive" if percept.emotion_valence >= 0 else "negative"
        orch._psyche = orch._psyche.model_copy(update={
            "attachment": attachment_manager.update_bond(
                orch._psyche.attachment, user_id, event_type, valence,
            ),
        })

    # Phase 4: responsibility — 判断記録
    if percept.intent and percept.intent != "expression":
        policy_label = percept.intent
    else:
        policy_label = percept.emotion or "neutral"
    try:
        orch._responsibility_mgr.record_decision(
            user_id=user_id,
            policy={"policy_label": policy_label},
            context={"emotion": percept.emotion, "text": percept.text[:100]},
        )
    except Exception as e:
        logger.debug("Responsibility record skipped: %s", e)

    # Phase 5: self_reference — 自己参照サマリ
    try:
        orch._self_ref_state = execute_self_reference(
            psyche_state=orch._psyche,
            responsibility_state=orch._responsibility_mgr.get_state(user_id),
            short_term_memory=orch._loop_state.memory if orch._loop_state else None,
            dynamics_state=orch._dynamics,
            dispersion_state=orch._dispersion_state,
        )
    except Exception as e:
        logger.debug("Self-reference skipped: %s", e)

    # Phase 6: repeated_tendency — 傾向観測
    scoped = (
        orch._scoped_goal_sys._current_scope
        if orch._scoped_goal_sys.has_active_scope
        else None
    )
    try:
        orch._tendency_sys.observe_turn(scoped_goal_used=scoped)
    except Exception as e:
        logger.debug("Tendency observation skipped: %s", e)


# ── 1-tick group 3: Fear / Observation / Description (Phase 7-7f) ──

def _run_1t_fear_observation_description(
    orch: PsycheOrchestrator,
    percept: Percept,
    delta_time: float,
    user_id: str,
) -> None:
    """Phase 7-7f: 恐怖再計算→行動結果記録→時間認知→知覚文脈→自己呈示→経路均衡→注意配分."""

    # Phase 7: fear recompute — 4柱リスク再計算
    orch._recompute_fear()

    # Phase 7a: action_result_observation — 行動記録を構成バッファへ
    # 構成バッファへの行動記録は、ポリシー選択完了後（毎ティック処理の末尾付近）に行う。
    # 結果記述の結合は低頻度処理帯（Phase 26c）に行い、同一周期内での即時構成を禁止する。
    if orch._last_selected_policy_label:
        try:
            # 入力経路の判定（Phase 7e と同じロジック）
            current_pathway_label = ""
            if percept.text:
                current_pathway_label = "text"
            elif bool(percept.intent and percept.intent != "expression"):
                current_pathway_label = "screen"

            record_inputs = ActionResultInputs(
                selected_policy_label=orch._last_selected_policy_label,
                selected_policy_axis=orch._last_selected_policy_axis,
                current_tick=orch._tick_count,
                input_pathway_label=current_pathway_label,
            )
            orch._action_result_observer.record_action(record_inputs)
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
        orch._temporal_cognition.accumulate_elapsed(
            tick=orch._tick_count,
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
        orch._perceptual_context.accumulate_summary(
            emotion=percept.emotion or "neutral",
            intent=percept.intent or "unknown",
            topics=list(getattr(percept, 'topics', ()) or ()),
            emotion_valence=percept.emotion_valence,
            tick=orch._tick_count,
        )
    except Exception as e:
        logger.debug("Perceptual context accumulate skipped: %s", e)

    # Phase 7d: situational_self_presentation — 相手別自己出力記録の蓄積（毎ティック）
    # 自己行動知覚の受領処理の直後に、本機能の蓄積処理を呼び出す。
    # 自己行動知覚の記録をREAD-ONLYで参照し、相手識別情報と対にして蓄積する。
    # パターン抽出禁止。マッピング形成禁止。ポリシー選択経路遮断。
    try:
        latest_record = orch._self_action_recorder.get_latest_record()
        if latest_record is not None and user_id:
            orch._situational_self_presentation.receive_and_accumulate(
                user_id=user_id,
                response_text=latest_record.response_text,
                policy_label=latest_record.policy_label,
                tick=orch._tick_count,
            )
            # 構成記述の生成
            orch._situational_self_presentation.generate_compositions(
                current_tick=orch._tick_count,
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

        orch._input_pathway_balance_state = process_input_pathway_balance(
            orch._input_pathway_balance_state,
            current_pathway=current_pathway,
            text_dialogue_state=(
                orch._text_dialogue_processor.state
                if orch._text_dialogue_processor else None
            ),
            spontaneous_state=(
                orch._spontaneous_processor.state
                if orch._spontaneous_processor else None
            ),
            has_screen_input=has_screen,
            config=orch._input_pathway_balance_config,
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
            orch._last_activation_result is not None
            and bool(getattr(orch._last_activation_result, "candidates", None))
        )
        _perception_count = 0
        if _has_perception and percept.text:
            _perception_count = max(1, len(percept.text.split()) // 5)

        orch._att_dist_state = process_attention_distribution(
            orch._att_dist_state,
            emotion_state=orch._psyche.emotions,
            memory_state=orch._last_bindings,
            motivation_state=orch._last_motives,
            transient_goal_state=orch._transient_goal_mgr._state if orch._transient_goal_mgr else None,
            scoped_goal_state=orch._scoped_goal_sys,
            responsibility_state=orch._dispersion_state,
            text_dialogue_state=(
                orch._text_dialogue_processor.state
                if orch._text_dialogue_processor else None
            ),
            spontaneous_state=(
                orch._spontaneous_processor.state
                if orch._spontaneous_processor else None
            ),
            has_perception_input=_has_perception,
            has_text_input=_has_text,
            has_spontaneous_activation=_has_spontaneous,
            perception_element_count=_perception_count,
            config=orch._att_dist_config,
        )
    except Exception as e:
        logger.debug("Attention distribution description skipped: %s", e)
