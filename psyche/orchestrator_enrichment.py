"""
psyche/orchestrator_enrichment.py - enrichment生成ロジックの物理的分離

orchestrator.pyから物理的に分離されたenrichment生成（48項目・5セクション）の処理コード。
ロジックは分離前と完全に同一であり、いかなる処理ロジックの変更も行っていない。

各関数は統合管理構造（PsycheOrchestrator）のインスタンスを第一引数として受け取り、
そのインスタンスの既存属性に対してのみ読み書きする。
新しいデータ構造・新しい状態変数・新しいキャッシュは追加しない。

セクション構成:
- セクション1: 心理状態（内面）— 10項目
- セクション2: 自己認識 — 12項目
- セクション3: 動機・目標 — 11項目
- セクション4: 記憶・内省 — 28項目
- セクション5: 判断傾向 — 6項目
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import PsycheOrchestrator

# ── orchestrator.py から参照されるモジュールレベルインポート ──────────
# これらは orchestrator.py のトップレベルで既にインポートされているが、
# 分離先ファイルでも直接使用するため、同一のインポートを行う。

from .responsibility_dispersion import (
    get_dispersion_summary,
)

from .stm_emotion_coupling import (
    get_coupling_summary,
)

from .self_image_integration import (
    get_self_image_summary,
)

from .identity_coherence import (
    get_coherence_summary,
)

from .tendency_awareness import (
    get_awareness_summary,
)

from .temporal_self_difference import (
    get_difference_summary,
)

from .continuity_strain import (
    get_strain_summary,
)

from .self_narrative import (
    get_narrative_summary,
)

from .long_term_dynamics import (
    get_observer_summary,
)

from .attention_distribution_description import (
    get_enrichment_text as get_att_dist_enrichment_text,
)

from .intrinsic_motivation import (
    get_motive_summary,
)

from .goal_candidates import (
    get_candidate_summary,
)

from .expectation_formation import (
    get_expectation_summary,
)

from .scoped_goal import (
    get_scoped_goal_summary,
)

from .transient_goal import (
    get_transient_goal_summary,
)

from .persistent_commitment import (
    get_commitment_summary,
)

from .proto_goal_vector import (
    get_vector_summary,
)

from .policy_candidate_expansion import (
    get_expansion_summary_text,
)

from .spontaneous_activation import (
    get_spontaneous_summary,
)

from .episodic_memory import (
    get_episodic_memory_summary,
)

from .emotional_memory_binding import (
    get_binding_summary,
)

from .introspection_consumption import (
    get_consumption_summary,
)

from .other_agent_model import (
    get_other_model_summary,
)

from .introspection_trace import (
    get_trace_summary,
)

from .memory_system_integration import (
    get_integration_summary_text,
)

from .other_model_real_feed import (
    get_real_feed_summary,
)

from .text_dialogue_input import (
    get_text_dialogue_summary,
)

from .memory_forgetting_fixation import (
    get_forgetting_fixation_summary,
)

from .input_pathway_balance import (
    get_enrichment_text as get_pathway_balance_enrichment_text,
)

from .forgetting_recall_balance import (
    get_enrichment_text as get_frb_enrichment_text,
)

from .value_orientation_validation import (
    get_validation_summary as get_vo_validation_summary,
)

from .enrichment_compression import (
    build_compressed_enrichment,
    normalize_section_items,
    prepend_freshness_annotation,
    ORIGINAL_FOOTER,
)

logger = logging.getLogger(__name__)


# ── セクション1: 心理状態（内面）──────────────────────────────────────

def _collect_enrichment_psyche(
    orch: "PsycheOrchestrator", user_id: str,
) -> list[tuple[str, str]]:
    """enrichment: 心理状態（内面）セクションの項目を収集する。"""
    p = orch._psyche
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
        resp_summary = orch._responsibility_mgr.get_summary(user_id)
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
    if orch._dispersion_state is not None:
        disp_summary = get_dispersion_summary(orch._dispersion_state)
        if disp_summary:
            psyche_items.append(("責任拡散", f"責任拡散: {disp_summary}"))
    # #42 responsibility_temporal_trace
    if orch._responsibility_temporal_trace is not None:
        try:
            rtt_data = orch._responsibility_temporal_trace.get_enrichment_data()
            rtt_text = rtt_data.get("summary_text", "")
            if rtt_text and "待機中" not in rtt_text:
                psyche_items.append(("責任推移", f"責任推移: {rtt_text}"))
        except Exception:
            pass
    # #6 stability_valve
    try:
        valve_bias = orch._stability_valve.generate_bias()
        if valve_bias.is_active:
            psyche_items.append((
                "安定弁",
                f"安定弁: active, level={valve_bias.activation_level:.2f}",
            ))
    except Exception:
        pass
    # #10 stm_emotion_coupling
    if orch._last_coupling is not None:
        coupling_str = get_coupling_summary(orch._last_coupling)
        if coupling_str:
            psyche_items.append(("感情連動", f"感情連動: {coupling_str}"))
    return psyche_items


# ── セクション2: 自己認識 ──────────────────────────────────────────

def _collect_enrichment_self(
    orch: "PsycheOrchestrator", user_id: str,
) -> list[tuple[str, str]]:
    """enrichment: 自己認識セクションの項目を収集する。"""
    self_items: list[tuple[str, str]] = []
    if orch._last_self_image is not None:
        summary = get_self_image_summary(orch._last_self_image)
        if summary:
            self_items.append(("自己像", f"自己像: {summary}"))
    if orch._last_coherence is not None:
        coh_summary = get_coherence_summary(orch._last_coherence)
        if coh_summary:
            self_items.append(("一貫性", f"一貫性: {coh_summary}"))
    if orch._tendency_awareness is not None:
        awareness_summary = get_awareness_summary(orch._tendency_awareness)
        if awareness_summary:
            self_items.append(("傾向", f"傾向: {awareness_summary}"))
    if orch._last_diff_summary is not None:
        diff_summary = get_difference_summary(orch._last_diff_summary)
        if diff_summary:
            self_items.append(("変化", f"変化: {diff_summary}"))
    if orch._last_strain is not None:
        strain_summary = get_strain_summary(orch._last_strain)
        if strain_summary:
            self_items.append(("連続性緊張", f"連続性緊張: {strain_summary}"))
    if orch._last_narrative is not None:
        narr_summary = get_narrative_summary(orch._last_narrative)
        if narr_summary:
            self_items.append(("自己語り", f"自己語り: {narr_summary}"))
    # #8 long_term_dynamics
    if orch._dynamics_observer is not None:
        ltd_summary = get_observer_summary(orch._dynamics_observer)
        if ltd_summary:
            self_items.append(("長期傾向", f"長期傾向: {ltd_summary}"))
    # #27 temporal_cognition
    if orch._temporal_cognition is not None:
        try:
            tc_data = orch._temporal_cognition.get_enrichment_data()
            tc_text = tc_data.get("summary_text", "")
            if tc_text and "待機中" not in tc_text:
                self_items.append(("時間認知", f"時間認知: {tc_text}"))
        except Exception:
            pass
    # #30 perceptual_context
    if orch._perceptual_context is not None:
        try:
            pc_text = orch._perceptual_context.get_enrichment_text()
            if pc_text and "待機中" not in pc_text:
                self_items.append(("知覚推移", f"知覚推移: {pc_text}"))
        except Exception:
            pass
    # #37 situational_self_presentation
    if orch._situational_self_presentation is not None:
        try:
            ssp_data = orch._situational_self_presentation.get_enrichment_data(
                user_id=user_id,
            )
            ssp_text = ssp_data.get("summary_text", "")
            if ssp_text and "待機中" not in ssp_text:
                self_items.append(("自己呈示", f"自己呈示: {ssp_text}"))
        except Exception:
            pass
    # #46 attention_distribution_description
    if orch._att_dist_state is not None:
        try:
            att_text = get_att_dist_enrichment_text(orch._att_dist_state)
            if att_text and "待機中" not in att_text:
                self_items.append(("注意配分", f"注意配分: {att_text}"))
        except Exception:
            pass
    return self_items


# ── セクション3: 動機・目標 ──────────────────────────────────────────

def _collect_enrichment_motive(
    orch: "PsycheOrchestrator",
) -> list[tuple[str, str]]:
    """enrichment: 動機・目標セクションの項目を収集する。"""
    motive_items: list[tuple[str, str]] = []
    if orch._last_motives is not None:
        motive_summary = get_motive_summary(orch._last_motives)
        if motive_summary:
            motive_items.append(("動機", f"動機: {motive_summary}"))
    if orch._candidate_gen is not None:
        try:
            cand_summary = get_candidate_summary(orch._candidate_gen)
            if cand_summary:
                total = cand_summary.get("total_candidates", 0)
                if total > 0:
                    motive_items.append(("目標候補", f"目標候補: {total}件"))
        except Exception:
            pass
    if orch._last_expectations is not None:
        exp_summary = get_expectation_summary(orch._last_expectations)
        if exp_summary:
            motive_items.append(("期待", f"期待: {exp_summary}"))
    # #3 scoped_goal
    if orch._scoped_goal_sys is not None:
        sg_summary = get_scoped_goal_summary(orch._scoped_goal_sys)
        if sg_summary and sg_summary.get("has_active_scope"):
            scope = sg_summary.get("current_scope", {})
            motive_items.append((
                "スコープ目標",
                f"スコープ目標: {scope.get('category', '?')} "
                f"(strength={scope.get('strength', 0):.2f})",
            ))
    # #4 transient_goal
    if orch._transient_goal_mgr is not None:
        tg_summary = get_transient_goal_summary(orch._transient_goal_mgr)
        if tg_summary and tg_summary.get("has_active_goal"):
            goal = tg_summary.get("active_goal", {})
            motive_items.append((
                "一時目標",
                f"一時目標: {goal.get('category', '?')} "
                f"(strength={goal.get('strength', 0):.2f})",
            ))
    # #31 persistent_commitment
    if orch._persistent_commitment is not None:
        try:
            pc_text = get_commitment_summary(
                orch._persistent_commitment.state
            )
            if pc_text and "待機中" not in pc_text:
                motive_items.append(("持続保持", f"持続保持: {pc_text}"))
        except Exception:
            pass
    # #5 proto_goal_vector
    if orch._vector_gen is not None:
        vec_summary = get_vector_summary(orch._vector_gen)
        if vec_summary and vec_summary.get("vector_count", 0) > 0:
            motive_items.append((
                "方向ベクトル",
                f"方向ベクトル: {vec_summary['vector_count']}本, "
                f"最強={vec_summary['strongest_magnitude']:.2f}",
            ))
    # #14 policy_candidate_expansion
    if orch._policy_expander is not None:
        try:
            exp_text = get_expansion_summary_text(orch._policy_expander)
            if exp_text:
                motive_items.append(("候補拡張", f"候補拡張: {exp_text}"))
        except Exception:
            pass
    # #18 spontaneous_activation
    if orch._spontaneous_processor is not None:
        try:
            sp_text = get_spontaneous_summary(
                orch._spontaneous_processor.state
            )
            if sp_text and "待機中" not in sp_text:
                motive_items.append(("自発起動", f"自発起動: {sp_text}"))
        except Exception:
            pass
    # #39 drive_variation_description
    if orch._drive_variation_processor is not None:
        try:
            dv_data = orch._drive_variation_processor.get_enrichment_data()
            dv_text = dv_data.get("summary_text", "")
            if dv_text and "待機中" not in dv_text:
                motive_items.append(("駆動変動", f"駆動変動: {dv_text}"))
        except Exception:
            pass
    # #40 expectation_lifecycle_description
    if orch._expectation_lifecycle_processor is not None:
        try:
            el_data = orch._expectation_lifecycle_processor.get_enrichment_data()
            el_text = el_data.get("summary_text", "")
            if el_text and "待機中" not in el_text:
                motive_items.append(("予期ライフサイクル", f"予期ライフサイクル: {el_text}"))
        except Exception:
            pass
    return motive_items


# ── セクション4: 記憶・内省 ──────────────────────────────────────────

def _collect_enrichment_memory(
    orch: "PsycheOrchestrator", user_id: str,
) -> list[tuple[str, str]]:
    """enrichment: 記憶・内省セクションの項目を収集する。"""
    memory_items: list[tuple[str, str]] = []
    if orch._last_episodes is not None:
        ep_summary = get_episodic_memory_summary(orch._last_episodes)
        if ep_summary:
            memory_items.append(("エピソード記憶", f"エピソード記憶: {ep_summary}"))
    if orch._last_bindings is not None:
        bind_summary = get_binding_summary(orch._last_bindings)
        if bind_summary:
            memory_items.append(("感情結合", f"感情結合: {bind_summary}"))
    if orch._last_consumption is not None:
        cons_summary = get_consumption_summary(orch._last_consumption)
        if cons_summary:
            memory_items.append(("内省消費", f"内省消費: {cons_summary}"))
    if orch._last_other_model is not None:
        other_summary = get_other_model_summary(orch._last_other_model)
        if other_summary:
            memory_items.append(("他者モデル", f"他者モデル: {other_summary}"))
    # #7 introspection_trace
    if orch._last_trace is not None:
        trace_str = get_trace_summary(orch._last_trace)
        if trace_str:
            memory_items.append(("内省", f"内省: {trace_str}"))
    # #15 memory_system_integration
    if orch._memory_integrator is not None:
        try:
            int_text = get_integration_summary_text(orch._memory_integrator)
            if int_text:
                memory_items.append(("記憶統合", f"記憶統合: {int_text}"))
        except Exception:
            pass
    # #16 other_model_real_feed
    if orch._real_feed_processor is not None:
        try:
            feed_text = get_real_feed_summary(orch._real_feed_processor)
            if feed_text and "inactive" not in feed_text:
                memory_items.append(("観測フィード", f"観測フィード: {feed_text}"))
        except Exception:
            pass
    # #17 text_dialogue_input
    if orch._text_dialogue_processor is not None:
        try:
            tdi_text = get_text_dialogue_summary(
                orch._text_dialogue_processor.state
            )
            if tdi_text and "待機中" not in tdi_text:
                memory_items.append(("入力経路", f"入力経路: {tdi_text}"))
        except Exception:
            pass
    # #20 memory_forgetting_fixation
    if orch._forgetting_fixation_processor is not None:
        try:
            ff_text = get_forgetting_fixation_summary(
                orch._forgetting_fixation_processor.state
            )
            if ff_text and "待機中" not in ff_text:
                memory_items.append(("記憶流動", f"記憶流動: {ff_text}"))
        except Exception:
            pass
    # #21 action_result_observation
    if orch._action_result_observer is not None:
        try:
            ar_data = orch._action_result_observer.get_enrichment_data()
            ar_text = ar_data.get("summary_text", "")
            if ar_text and "待機中" not in ar_text:
                memory_items.append(("行動-結果", f"行動-結果: {ar_text}"))
        except Exception:
            pass
    # #22 other_model_dialogue_learning
    if orch._dialogue_learning_processor is not None:
        try:
            dl_data = orch._dialogue_learning_processor.get_enrichment_data()
            dl_text = dl_data.get("summary_text", "")
            if dl_text and "待機中" not in dl_text:
                memory_items.append(("他者蓄積", f"他者蓄積: {dl_text}"))
        except Exception:
            pass
    # #44 other_boundary_accumulation
    if orch._other_boundary_accumulation is not None:
        try:
            oba_data = orch._other_boundary_accumulation.get_enrichment_data(
                user_id=user_id,
            )
            oba_text = oba_data.get("summary_text", "")
            if oba_text and "待機中" not in oba_text:
                memory_items.append(("境界蓄積", f"境界蓄積: {oba_text}"))
        except Exception:
            pass
    # #48 hypothesis_observation_pairing
    if orch._hypothesis_observation_pairing is not None:
        try:
            hop_data = orch._hypothesis_observation_pairing.get_enrichment_data()
            hop_text = hop_data.get("summary_text", "")
            if hop_text and "待機中" not in hop_text:
                memory_items.append(("仮説-観測対", f"仮説-観測対: {hop_text}"))
        except Exception:
            pass
    # #23 meta_emotion_cognition
    if orch._meta_emotion_processor is not None:
        try:
            me_data = orch._meta_emotion_processor.get_enrichment_data()
            me_text = me_data.get("summary_text", "")
            if me_text and "待機中" not in me_text:
                memory_items.append(("メタ感情", f"メタ感情: {me_text}"))
        except Exception:
            pass
    # #24 self_action_perception
    if orch._self_action_recorder is not None:
        try:
            sa_data = orch._self_action_recorder.get_enrichment_data()
            sa_text = sa_data.get("summary_text", "")
            if sa_text and "待機中" not in sa_text:
                memory_items.append(("自己行動", f"自己行動: {sa_text}"))
        except Exception:
            pass
    # #25 expectation_action_diff
    try:
        diff_summary = orch.get_expectation_diff_summary()
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
    if orch._intent_action_gap_recorder is not None:
        try:
            gap_data = orch._intent_action_gap_recorder.get_enrichment_data()
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
    if orch._multi_path_recall is not None:
        try:
            mpr_data = orch._multi_path_recall.get_enrichment_data()
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
    if orch._introspection_cross_section is not None:
        try:
            ics_text = orch._introspection_cross_section.get_enrichment_text()
            if ics_text and "待機中" not in ics_text:
                memory_items.append(("内省横断", f"内省横断: {ics_text}"))
        except Exception:
            pass
    # #38 introspection_longitudinal_view
    if orch._introspection_longitudinal_view is not None:
        try:
            ilv_snaps = orch._introspection_cross_section.get_snapshot_window()
            ilv_data = orch._introspection_longitudinal_view.get_enrichment_data(ilv_snaps)
            ilv_text = ilv_data.get("summary_text", "")
            if ilv_text and "待機中" not in ilv_text:
                memory_items.append(("内省縦断", f"内省縦断: {ilv_text}"))
        except Exception:
            pass
    # #31 selection_attribution
    if orch._selection_attribution_recorder is not None:
        try:
            sa_data = orch._selection_attribution_recorder.get_enrichment_data()
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
    if orch._spontaneous_recall is not None:
        try:
            sr_data = orch._spontaneous_recall.get_enrichment_data()
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
    if orch._contradiction_processor is not None:
        try:
            ic_text = orch._contradiction_processor.get_enrichment_text()
            if ic_text and "待機中" not in ic_text:
                memory_items.append(("矛盾並置", f"矛盾並置: {ic_text}"))
        except Exception:
            pass
    # #35 interaction_accumulation
    if orch._interaction_accumulation is not None:
        try:
            ia_data = orch._interaction_accumulation.get_enrichment_data()
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
    if orch._emotional_backdrop_processor is not None:
        try:
            eb_data = orch._emotional_backdrop_processor.get_enrichment_data()
            eb_text = eb_data.get("summary_text", "")
            if eb_text and "待機中" not in eb_text:
                memory_items.append(("感情基調", f"感情基調: {eb_text}"))
        except Exception:
            pass
    # #43 emotion_cooccurrence_description
    if orch._emotion_cooccurrence_processor is not None:
        try:
            ec_data = orch._emotion_cooccurrence_processor.get_enrichment_data()
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
    if orch._input_pathway_balance_state is not None:
        try:
            ipb_text = get_pathway_balance_enrichment_text(
                orch._input_pathway_balance_state
            )
            if ipb_text and "待機中" not in ipb_text:
                memory_items.append(("経路均衡", f"経路均衡: {ipb_text}"))
        except Exception:
            pass
    # #45 forgetting_recall_balance
    if orch._frb_state is not None:
        try:
            frb_text = get_frb_enrichment_text(orch._frb_state)
            if frb_text and "待機中" not in frb_text:
                memory_items.append(("忘却想起均衡", f"忘却想起均衡: {frb_text}"))
        except Exception:
            pass
    return memory_items


# ── セクション5: 判断傾向 ──────────────────────────────────────────

def _collect_enrichment_bias(
    orch: "PsycheOrchestrator",
) -> list[tuple[str, str]]:
    """enrichment: 判断傾向セクションの項目を収集する。"""
    bias_items: list[tuple[str, str]] = []
    # #9 decision_bias
    if orch._last_decision_bias is not None:
        db = orch._last_decision_bias
        bias_items.append((
            "判断バイアス",
            f"判断バイアス: phase={db.dynamics_phase.value}, "
            f"valence={db.valence_bias:.2f}",
        ))
    # #11 tone modifier
    if orch._last_tone_mod is not None:
        bias_items.append((
            "トーン推奨",
            f"トーン推奨: {orch._last_tone_mod.recommended.value}",
        ))
    # #12 context_sensitivity
    if orch._last_sensitivity_bias is not None:
        sb = orch._last_sensitivity_bias
        if sb.caution_level > 0.5:
            bias_items.append((
                "空気読み",
                f"空気読み: caution={sb.caution_level:.2f}",
            ))
    # #13 silence/hesitation
    if orch._last_has_silence:
        bias_items.append(("沈黙傾向", "沈黙傾向: あり"))
    # #19 value_orientation_validation
    if orch._vo_validator is not None:
        try:
            vo_text = get_vo_validation_summary(orch._vo_validator.state)
            if vo_text and "待機中" not in vo_text:
                bias_items.append(("価値検証", f"価値検証: {vo_text}"))
        except Exception:
            pass
    # #20 内部-外部間の張力サマリー (READ-ONLY参照のみ)
    try:
        tension_parts: list[str] = []
        # (a) persistent_commitment の方向的張力
        if orch._persistent_commitment is not None:
            pc_active = [
                it for it in orch._persistent_commitment.state.items
                if not it.released
            ]
            if pc_active:
                tension_parts.append(f"保持方向{len(pc_active)}件の方向的バイアスあり")
        # (b) context_sensitivity の慎重度
        if (orch._last_sensitivity_bias is not None
                and orch._last_sensitivity_bias.caution_level > 0.5):
            tension_parts.append("外部文脈の慎重度が高い")
        # (c) value_orientation の最強方向バイアス (vo_validation未記述時のみ)
        vo_already_shown = (
            orch._vo_validator is not None
            and any(lbl == "価値検証" for lbl, _ in bias_items)
        )
        if orch._value_orientation is not None and not vo_already_shown:
            dims = orch._value_orientation.get_all_dimensions()
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
    return bias_items


# ── 項目統合 ──────────────────────────────────────────────────────

def _collect_enrichment_items(
    orch: "PsycheOrchestrator", user_id: str = "viewer",
) -> list[dict]:
    """enrichment項目を(ラベル, テキスト)ペアとしてセクション別に収集する。

    各モジュールの出力メソッドは一切変更せず、テキストを読み取り専用で取得する。
    戻り値はbuild_compressed_enrichment()の入力形式に合わせた辞書リスト。

    Returns:
        list of {"header": str, "items": list[tuple[str, str]]}
    """
    sections_data: list[dict] = []

    # ── 【心理状態（内面）】 ──
    psyche_items = _collect_enrichment_psyche(orch, user_id)
    if psyche_items:
        sections_data.append({
            "header": "【心理状態（内面）】",
            "items": psyche_items,
        })

    # ── 【自己認識】 ──
    self_items = _collect_enrichment_self(orch, user_id)
    if self_items:
        sections_data.append({
            "header": "【自己認識】",
            "items": self_items,
        })

    # ── 【動機・目標】 ──
    motive_items = _collect_enrichment_motive(orch)
    if motive_items:
        sections_data.append({
            "header": "【動機・目標】",
            "items": motive_items,
        })

    # ── 【記憶・内省】 ──
    memory_items = _collect_enrichment_memory(orch, user_id)
    if memory_items:
        sections_data.append({
            "header": "【記憶・内省】",
            "items": memory_items,
        })

    # ── 【判断傾向】 ── (Phase 30-35 cached)
    bias_items = _collect_enrichment_bias(orch)
    if bias_items:
        sections_data.append({
            "header": "【判断傾向】",
            "items": bias_items,
        })

    return sections_data


# ── 公開関数 ──────────────────────────────────────────────────────

def get_prompt_enrichment(
    orch: "PsycheOrchestrator", user_id: str = "viewer",
) -> str:
    """Gemini プロンプト用の心理状態テキストを生成する。

    enrichment項目を収集後、空状態統一→圧縮パイプライン→鮮度注釈を適用する。
    圧縮結果のフィードバック経路は構造的に遮断されている。

    brain.py の _format_psyche_for_prompt を置き換える。

    Args:
        orch: PsycheOrchestrator インスタンス
        user_id: 責任状態取得用のユーザーID
    """
    # 項目収集（既存ロジックの構造を維持）
    sections_data = _collect_enrichment_items(orch, user_id)

    # 起動品質A: 空状態記述の統一（各項目テキストの表層置換）
    for section in sections_data:
        section["items"] = normalize_section_items(section["items"])

    footer = ORIGINAL_FOOTER

    # 圧縮パイプライン実行
    compressed_text, new_cache, _ratio = build_compressed_enrichment(
        sections_data=sections_data,
        prev_cache=orch._enrichment_prev_cache,
        footer=footer,
    )

    # ── Execution monitor: 圧縮比記録 (READ-ONLY観測) ──
    try:
        _before_chars = sum(
            len(t) for sec in sections_data for _, t in sec.get("items", [])
        )
        _after_chars = len(compressed_text)
        orch._execution_monitor.record_compression(
            _before_chars, _after_chars, _ratio,
        )
    except Exception:
        pass  # 安全弁1

    # ── Execution monitor: enrichment分布記録 (READ-ONLY観測) ──
    try:
        orch._execution_monitor.record_enrichment_distribution(
            tick_count=orch._tick_count,
            sections_data=sections_data,
            compressed_text=compressed_text,
        )
    except Exception:
        pass  # 安全弁1

    # キャッシュ更新（1ティック分のみ保持）
    orch._enrichment_prev_cache = new_cache

    # 起動品質B: セッション境界の鮮度注釈
    compressed_text = prepend_freshness_annotation(
        enrichment_text=compressed_text,
        session_gap_seconds=orch._session_gap_seconds,
        session_resume_tick=orch._session_resume_tick,
        current_tick=orch._tick_count,
    )

    return compressed_text
