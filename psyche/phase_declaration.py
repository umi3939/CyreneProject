"""
psyche/phase_declaration.py - orchestrator Phase実行順序の宣言的定義

統合管理構造(orchestrator)が実行する全処理を、宣言的データ構造として外部化する。
実行制御フロー自体は変更しない。既存の手続き的コードとは独立に存在し、
テスト・検証・ドキュメント生成時にのみ参照される。

実行時には一切参照されない（安全弁1: 実行非介入保証）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ── 帯域定義 ──────────────────────────────────────────────────


class Band(Enum):
    """6帯域の列挙型。"""
    EVERY_TICK = "every_tick"
    EVERY_3_TICKS = "every_3_ticks"
    EVERY_5_TICKS = "every_5_ticks"
    EVERY_10_TICKS = "every_10_ticks"
    CANDIDATE_GENERATION = "candidate_generation"
    POST_SELECTION = "post_selection"


@dataclass(frozen=True)
class BandDefinition:
    """帯域定義レコード。"""
    band: Band
    execution_condition: str
    execution_method: str
    phase_ids: tuple[str, ...]  # 所属Phase識別子を実行順に列挙


# ── Phase定義レコード ──────────────────────────────────────────


@dataclass(frozen=True)
class PhaseDefinition:
    """処理単位(Phase)の宣言的定義レコード。

    全属性は不変(frozen)。実行時に変化する内部状態を持たない。
    """
    phase_id: str                           # 一意の識別子 (例: "1", "7a", "14b")
    display_name: str                       # 処理の短い説明
    band: Band                              # 帯域所属
    band_order: int                         # 同一帯域内の実行順序(0始まり連番)
    modules: tuple[str, ...]                # 対応モジュール名
    reads: tuple[str, ...]                  # 読み取り中間状態
    writes: tuple[str, ...]                 # 書き込み中間状態
    persisted_fields: tuple[str, ...]       # 永続化対象フィールド名
    enrichment_items: tuple[str, ...]       # 供給するenrichment項目番号
    method_name: str                        # 統合管理構造のメソッド名
    error_absorbed: bool                    # 個別のエラー吸収境界で囲まれているか


# ── 全Phase一覧 ──────────────────────────────────────────────
# 設計書 §3.3 の全Phase定義をそのまま宣言的データ構造として定義する。


# --- 帯域: EVERY_TICK (_run_every_tick) ---

PHASE_1 = PhaseDefinition(
    phase_id="1",
    display_name="感情更新+STM残留",
    band=Band.EVERY_TICK,
    band_order=0,
    modules=("react_with_stm",),
    reads=(
        "_psyche", "_loop_state",
        # Cycle 9-10: build_drive_context reads
        "_behavioral_diversity_state",
        "_contradiction_processor",
        "_transient_goal_mgr", "_persistent_commitment", "_scoped_goal_sys",
        "_temporal_cognition",
        # Cycle 9-10: build_mood_context reads
        "_memory_emotion_return",
    ),
    writes=("_psyche", "_loop_state"),
    persisted_fields=("psyche", "loop_state"),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=False,
)

PHASE_2 = PhaseDefinition(
    phase_id="2",
    display_name="ピーク/リバウンド判定",
    band=Band.EVERY_TICK,
    band_order=1,
    modules=("dynamics",),
    reads=("_psyche", "_loop_state"),
    writes=("_dynamics",),
    persisted_fields=("dynamics",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=False,
)

PHASE_2A = PhaseDefinition(
    phase_id="2a",
    display_name="dynamics相振幅計算",
    band=Band.EVERY_TICK,
    band_order=2,
    modules=("emotion_amplitude",),
    reads=("_amplitude_state", "_dynamics"),
    writes=("_amplitude_state",),
    persisted_fields=("amplitude",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=False,
)

PHASE_2B = PhaseDefinition(
    phase_id="2b",
    display_name="感情別独立減衰",
    band=Band.EVERY_TICK,
    band_order=3,
    modules=("multi_emotion",),
    reads=("_psyche",),
    writes=("_psyche",),
    persisted_fields=("psyche",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_2C = PhaseDefinition(
    phase_id="2c",
    display_name="STM残留→再活性化",
    band=Band.EVERY_TICK,
    band_order=4,
    modules=("stm_emotion_coupling",),
    reads=("_psyche", "_loop_state"),
    writes=("_psyche", "_last_coupling"),
    persisted_fields=("psyche", "last_coupling"),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_3 = PhaseDefinition(
    phase_id="3",
    display_name="対話相手ボンド更新",
    band=Band.EVERY_TICK,
    band_order=5,
    modules=("attachment_manager",),
    reads=("_psyche", "_last_percept"),
    writes=("_psyche",),
    persisted_fields=("psyche",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=False,
)

PHASE_4 = PhaseDefinition(
    phase_id="4",
    display_name="判断記録",
    band=Band.EVERY_TICK,
    band_order=6,
    modules=("responsibility_manager",),
    reads=("_last_percept",),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_5 = PhaseDefinition(
    phase_id="5",
    display_name="自己参照サマリ",
    band=Band.EVERY_TICK,
    band_order=7,
    modules=("self_reference",),
    reads=("_psyche", "_loop_state", "_dynamics", "_dispersion_state"),
    writes=("_self_ref_state",),
    persisted_fields=("self_ref_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_6 = PhaseDefinition(
    phase_id="6",
    display_name="傾向観測",
    band=Band.EVERY_TICK,
    band_order=8,
    modules=("repeated_tendency",),
    reads=("_scoped_goal_sys",),
    writes=("_tendency_sys",),
    persisted_fields=("tendency_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7 = PhaseDefinition(
    phase_id="7",
    display_name="4柱リスク再計算",
    band=Band.EVERY_TICK,
    band_order=9,
    modules=("fear",),
    reads=("_psyche",),
    writes=("_psyche",),
    persisted_fields=("psyche",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=False,
)

PHASE_7A = PhaseDefinition(
    phase_id="7a",
    display_name="行動記録→構成バッファ",
    band=Band.EVERY_TICK,
    band_order=10,
    modules=("action_result_observation",),
    reads=("_last_selected_policy_label", "_last_selected_policy_axis", "_last_percept"),
    writes=(),
    persisted_fields=("action_result_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7B = PhaseDefinition(
    phase_id="7b",
    display_name="経過記録の蓄積",
    band=Band.EVERY_TICK,
    band_order=11,
    modules=("temporal_cognition",),
    reads=("_last_percept",),
    writes=(),
    persisted_fields=("temporal_cognition_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7C = PhaseDefinition(
    phase_id="7c",
    display_name="知覚サマリの蓄積",
    band=Band.EVERY_TICK,
    band_order=12,
    modules=("perceptual_context",),
    reads=("_last_percept",),
    writes=(),
    persisted_fields=("perceptual_context_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7D = PhaseDefinition(
    phase_id="7d",
    display_name="相手別自己出力記録蓄積",
    band=Band.EVERY_TICK,
    band_order=13,
    modules=("situational_self_presentation",),
    reads=("_self_action_recorder",),
    writes=(),
    persisted_fields=("situational_self_presentation_state",),
    enrichment_items=(),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7E = PhaseDefinition(
    phase_id="7e",
    display_name="入力経路間均衡記述",
    band=Band.EVERY_TICK,
    band_order=14,
    modules=("input_pathway_balance",),
    reads=("_last_percept", "_text_dialogue_processor", "_spontaneous_processor"),
    writes=("_input_pathway_balance_state",),
    persisted_fields=("input_pathway_balance_state",),
    enrichment_items=("41",),
    method_name="_run_every_tick",
    error_absorbed=True,
)

PHASE_7F = PhaseDefinition(
    phase_id="7f",
    display_name="注意配分の構造的記述",
    band=Band.EVERY_TICK,
    band_order=15,
    modules=("attention_distribution_description",),
    reads=(
        "_psyche", "_last_bindings", "_last_motives",
        "_transient_goal_mgr", "_scoped_goal_sys", "_dispersion_state",
        "_text_dialogue_processor", "_spontaneous_processor",
        "_last_percept", "_last_activation_result",
    ),
    writes=("_att_dist_state",),
    persisted_fields=("attention_distribution_state",),
    enrichment_items=("46",),
    method_name="_run_every_tick",
    error_absorbed=True,
)


# --- 帯域: EVERY_3_TICKS (_run_every_3_ticks) ---

PHASE_8 = PhaseDefinition(
    phase_id="8",
    display_name="傾向認知",
    band=Band.EVERY_3_TICKS,
    band_order=0,
    modules=("tendency_awareness",),
    reads=("_tendency_sys",),
    writes=("_tendency_awareness",),
    persisted_fields=("tendency_awareness",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_9 = PhaseDefinition(
    phase_id="9",
    display_name="統合自己ビュー",
    band=Band.EVERY_3_TICKS,
    band_order=1,
    modules=("self_model",),
    reads=("_psyche", "_tendency_sys", "_tendency_awareness", "_vector_gen", "_value_orientation"),
    writes=("_last_self_view",),
    persisted_fields=("last_self_view",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_10 = PhaseDefinition(
    phase_id="10",
    display_name="方向ベクトル更新",
    band=Band.EVERY_3_TICKS,
    band_order=2,
    modules=("proto_goal_vector",),
    reads=("_value_orientation", "_last_trace", "_psyche"),
    writes=("_vector_gen",),
    persisted_fields=("vector_state",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_11 = PhaseDefinition(
    phase_id="11",
    display_name="目標候補生成",
    band=Band.EVERY_3_TICKS,
    band_order=3,
    modules=("goal_candidates",),
    reads=("_vector_gen",),
    writes=("_candidate_gen",),
    persisted_fields=("candidate_state",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_12 = PhaseDefinition(
    phase_id="12",
    display_name="一時目標選択",
    band=Band.EVERY_3_TICKS,
    band_order=4,
    modules=("transient_goal",),
    reads=("_candidate_gen",),
    writes=("_transient_goal_mgr",),
    persisted_fields=("transient_goal_state",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_12B = PhaseDefinition(
    phase_id="12b",
    display_name="持続的取り組み保持",
    band=Band.EVERY_3_TICKS,
    band_order=5,
    modules=("persistent_commitment",),
    reads=("_transient_goal_mgr", "_value_orientation", "_psyche"),
    writes=("_persistent_commitment",),
    persisted_fields=("persistent_commitment_state",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_13 = PhaseDefinition(
    phase_id="13",
    display_name="スコープ目標コミット",
    band=Band.EVERY_3_TICKS,
    band_order=6,
    modules=("scoped_goal",),
    reads=("_transient_goal_mgr",),
    writes=("_scoped_goal_sys",),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14 = PhaseDefinition(
    phase_id="14",
    display_name="内的動機感知",
    band=Band.EVERY_3_TICKS,
    band_order=7,
    modules=("intrinsic_motivation",),
    reads=("_psyche", "_tendency_sys", "_vector_gen", "_candidate_gen"),
    writes=("_last_motives",),
    persisted_fields=("last_motives",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14B = PhaseDefinition(
    phase_id="14b",
    display_name="メタ感情認知",
    band=Band.EVERY_3_TICKS,
    band_order=8,
    modules=("meta_emotion_cognition",),
    reads=("_psyche", "_dynamics", "_loop_state", "_last_motives"),
    writes=("_last_meta_emotion",),
    persisted_fields=("meta_emotion_state",),
    enrichment_items=(),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14C = PhaseDefinition(
    phase_id="14c",
    display_name="時間認知:多断面記述",
    band=Band.EVERY_3_TICKS,
    band_order=9,
    modules=("temporal_cognition",),
    reads=("_last_episodes", "_dynamics", "_last_narrative"),
    writes=(),
    persisted_fields=("temporal_cognition_state",),
    enrichment_items=("27",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14D = PhaseDefinition(
    phase_id="14d",
    display_name="内省断面:横断的記述",
    band=Band.EVERY_3_TICKS,
    band_order=10,
    modules=("introspection_cross_section",),
    reads=(
        "_last_self_view", "_last_diff_summary", "_last_coherence",
        "_last_narrative", "_last_consumption", "_meta_emotion_processor",
    ),
    writes=(),
    persisted_fields=("introspection_cross_section_state",),
    enrichment_items=("29",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14E = PhaseDefinition(
    phase_id="14e",
    display_name="知覚推移特徴量記述",
    band=Band.EVERY_3_TICKS,
    band_order=11,
    modules=("perceptual_context",),
    reads=(),
    writes=(),
    persisted_fields=("perceptual_context_state",),
    enrichment_items=("30",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14F = PhaseDefinition(
    phase_id="14f",
    display_name="矛盾並置記述",
    band=Band.EVERY_3_TICKS,
    band_order=12,
    modules=("internal_contradiction_description",),
    reads=(),
    writes=("_last_contradiction_result",),
    persisted_fields=("internal_contradiction_state",),
    enrichment_items=("34",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14G = PhaseDefinition(
    phase_id="14g",
    display_name="感情基調の持続認知",
    band=Band.EVERY_3_TICKS,
    band_order=13,
    modules=("emotional_backdrop_cognition",),
    reads=("_psyche", "_dynamics", "_loop_state", "_last_motives"),
    writes=("_last_backdrop_result",),
    persisted_fields=("emotional_backdrop_state",),
    enrichment_items=("36",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14H = PhaseDefinition(
    phase_id="14h",
    display_name="内省:時間的縦断参照",
    band=Band.EVERY_3_TICKS,
    band_order=14,
    modules=("introspection_longitudinal_view",),
    reads=("_introspection_cross_section",),
    writes=(),
    persisted_fields=(),
    enrichment_items=("38",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14I = PhaseDefinition(
    phase_id="14i",
    display_name="駆動の変動記述",
    band=Band.EVERY_3_TICKS,
    band_order=15,
    modules=("drive_variation_description",),
    reads=(),
    writes=("_last_drive_variation_result",),
    persisted_fields=("drive_variation_state",),
    enrichment_items=("39",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)

PHASE_14J = PhaseDefinition(
    phase_id="14j",
    display_name="感情間の共起記述",
    band=Band.EVERY_3_TICKS,
    band_order=16,
    modules=("emotion_cooccurrence_description",),
    reads=("_psyche",),
    writes=("_last_cooccurrence_result",),
    persisted_fields=("emotion_cooccurrence_state",),
    enrichment_items=("43",),
    method_name="_run_every_3_ticks",
    error_absorbed=True,
)


# --- 帯域: EVERY_5_TICKS (_run_every_5_ticks) ---

PHASE_15 = PhaseDefinition(
    phase_id="15",
    display_name="自己差分",
    band=Band.EVERY_5_TICKS,
    band_order=0,
    modules=("temporal_self_difference",),
    reads=("_last_self_view",),
    writes=("_last_diff_summary",),
    persisted_fields=("last_diff_summary",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_15B = PhaseDefinition(
    phase_id="15b",
    display_name="安定化記述",
    band=Band.EVERY_5_TICKS,
    band_order=1,
    modules=("stabilization_description",),
    reads=(
        "_psyche", "_loop_state", "_transient_goal_mgr",
        "_persistent_commitment", "_last_activation_result",
        "_last_percept", "_last_diff_summary",
    ),
    writes=("_stabilization_desc_state",),
    persisted_fields=("stabilization_description_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_16 = PhaseDefinition(
    phase_id="16",
    display_name="連続性負荷",
    band=Band.EVERY_5_TICKS,
    band_order=2,
    modules=("continuity_strain",),
    reads=("_last_diff_summary",),
    writes=("_last_strain",),
    persisted_fields=("last_strain",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_17 = PhaseDefinition(
    phase_id="17",
    display_name="暫定自己像",
    band=Band.EVERY_5_TICKS,
    band_order=3,
    modules=("self_image_integration",),
    reads=("_last_self_view", "_tendency_awareness", "_last_diff_summary", "_last_strain"),
    writes=("_last_self_image",),
    persisted_fields=("last_self_image",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_18 = PhaseDefinition(
    phase_id="18",
    display_name="一貫性評価",
    band=Band.EVERY_5_TICKS,
    band_order=4,
    modules=("identity_coherence",),
    reads=(
        "_last_self_image", "_last_diff_summary", "_last_strain",
        "_tendency_awareness", "_value_orientation",
    ),
    writes=("_last_coherence",),
    persisted_fields=("last_coherence",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_19 = PhaseDefinition(
    phase_id="19",
    display_name="自己ナラティブ",
    band=Band.EVERY_5_TICKS,
    band_order=5,
    modules=("self_narrative",),
    reads=("_psyche", "_loop_state", "_tendency_awareness", "_last_diff_summary", "_input_supply"),
    writes=("_last_narrative",),
    persisted_fields=("last_narrative",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_20 = PhaseDefinition(
    phase_id="20",
    display_name="エピソード記録",
    band=Band.EVERY_5_TICKS,
    band_order=6,
    modules=("episodic_memory",),
    reads=(
        "_psyche", "_loop_state", "_tendency_awareness",
        "_last_diff_summary", "_last_coherence", "_last_narrative",
        "_input_supply",
    ),
    writes=("_last_episodes",),
    persisted_fields=("last_episodes",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21 = PhaseDefinition(
    phase_id="21",
    display_name="感情記憶紐づけ",
    band=Band.EVERY_5_TICKS,
    band_order=7,
    modules=("emotional_memory_binding",),
    reads=("_loop_state", "_psyche", "_last_recalled_memories", "_last_episodes"),
    writes=("_last_bindings",),
    persisted_fields=("last_bindings",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21B = PhaseDefinition(
    phase_id="21b",
    display_name="記憶系統統合",
    band=Band.EVERY_5_TICKS,
    band_order=8,
    modules=("memory_system_integration",),
    reads=(
        "_last_episodes", "_last_recalled_memories", "_last_bindings",
        "_last_percept", "_action_result_observer",
    ),
    writes=("_last_integration_result",),
    persisted_fields=("memory_integration_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21C = PhaseDefinition(
    phase_id="21c",
    display_name="忘却と固定化",
    band=Band.EVERY_5_TICKS,
    band_order=9,
    modules=("memory_forgetting_fixation",),
    reads=(),
    writes=("_last_forgetting_fixation",),
    persisted_fields=("forgetting_fixation_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21D = PhaseDefinition(
    phase_id="21d",
    display_name="多経路想起",
    band=Band.EVERY_5_TICKS,
    band_order=10,
    modules=("multi_path_recall",),
    reads=(
        "_last_integration_result", "_last_bindings",
        "_forgetting_fixation_processor", "_psyche",
        "_last_percept", "_temporal_cognition",
    ),
    writes=(),
    persisted_fields=("multi_path_recall_state",),
    enrichment_items=("28",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21E = PhaseDefinition(
    phase_id="21e",
    display_name="自発的想起",
    band=Band.EVERY_5_TICKS,
    band_order=11,
    modules=("spontaneous_recall",),
    reads=(
        "_last_integration_result", "_forgetting_fixation_processor",
        "_psyche", "_last_motives", "_last_strain",
        "_vector_gen", "_temporal_cognition", "_last_bindings",
    ),
    writes=(),
    persisted_fields=("spontaneous_recall_state",),
    enrichment_items=("33",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21F = PhaseDefinition(
    phase_id="21f",
    display_name="忘却-想起均衡記述",
    band=Band.EVERY_5_TICKS,
    band_order=12,
    modules=("forgetting_recall_balance",),
    reads=("_forgetting_fixation_processor", "_multi_path_recall", "_spontaneous_recall"),
    writes=("_frb_state",),
    persisted_fields=("forgetting_recall_balance_state",),
    enrichment_items=("45",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21G = PhaseDefinition(
    phase_id="21g",
    display_name="記憶想起→感情帰還",
    band=Band.EVERY_5_TICKS,
    band_order=13,
    modules=("memory_emotion_return",),
    reads=(
        "_memory_emotion_return", "_multi_path_recall", "_spontaneous_recall",
        "_last_bindings", "_psyche",
    ),
    writes=("_psyche",),
    persisted_fields=("memory_emotion_return_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_21H = PhaseDefinition(
    phase_id="21h",
    display_name="他者仮説→感情帰還",
    band=Band.EVERY_5_TICKS,
    band_order=14,
    modules=("other_hypothesis_emotion_return",),
    reads=(
        "_other_hypothesis_emotion_return", "_other_model_sys",
        "_psyche",
    ),
    writes=("_psyche",),
    persisted_fields=("other_hypothesis_emotion_return_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_22 = PhaseDefinition(
    phase_id="22",
    display_name="内省ログ生成",
    band=Band.EVERY_5_TICKS,
    band_order=15,
    modules=("introspection_trace",),
    reads=("_psyche", "_value_orientation"),
    writes=("_last_trace",),
    persisted_fields=("last_trace",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_23 = PhaseDefinition(
    phase_id="23",
    display_name="内省消費・再構成",
    band=Band.EVERY_5_TICKS,
    band_order=16,
    modules=("introspection_consumption",),
    reads=("_last_trace", "_last_narrative", "_last_coherence", "_tendency_awareness", "_last_episodes"),
    writes=("_last_consumption",),
    persisted_fields=("last_consumption",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_24 = PhaseDefinition(
    phase_id="24",
    display_name="期待形成",
    band=Band.EVERY_5_TICKS,
    band_order=17,
    modules=("expectation_formation",),
    reads=("_tendency_sys", "_last_diff_summary", "_last_narrative"),
    writes=("_last_expectations",),
    persisted_fields=("last_expectations",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_24B = PhaseDefinition(
    phase_id="24b",
    display_name="参照頻度記述",
    band=Band.EVERY_5_TICKS,
    band_order=18,
    modules=("reference_frequency_description",),
    reads=(
        "_last_episodes", "_last_bindings", "_last_consumption",
        "_last_expectations", "_last_motives", "_last_narrative",
        "_last_other_model", "_self_ref_state",
        "_action_result_observer", "_dialogue_learning_processor",
        "_forgetting_fixation_processor", "_multi_path_recall",
        "_spontaneous_recall",
    ),
    writes=("_reference_frequency_state",),
    persisted_fields=("reference_frequency_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25A = PhaseDefinition(
    phase_id="25a",
    display_name="実対話観測抽出",
    band=Band.EVERY_5_TICKS,
    band_order=19,
    modules=("other_model_real_feed",),
    reads=(
        "_last_percept", "_loop_state", "_psyche", "_dynamics",
        "_last_recalled_memories", "_last_integration_result",
        "_action_result_observer",
    ),
    writes=("_last_feed_result",),
    persisted_fields=("real_feed_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25C = PhaseDefinition(
    phase_id="25c",
    display_name="他者対話学習",
    band=Band.EVERY_5_TICKS,
    band_order=20,
    modules=("other_model_dialogue_learning",),
    reads=("_last_feed_result", "_action_result_observer"),
    writes=("_last_dialogue_learning",),
    persisted_fields=("dialogue_learning_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25D = PhaseDefinition(
    phase_id="25d",
    display_name="相互作用蓄積",
    band=Band.EVERY_5_TICKS,
    band_order=21,
    modules=("interaction_accumulation",),
    reads=("_self_action_recorder", "_real_feed_processor"),
    writes=(),
    persisted_fields=("interaction_accumulation_state",),
    enrichment_items=("35",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25E = PhaseDefinition(
    phase_id="25e",
    display_name="他者境界蓄積",
    band=Band.EVERY_5_TICKS,
    band_order=22,
    modules=("other_boundary_accumulation",),
    reads=("_last_other_model",),
    writes=("_last_boundary_accumulation",),
    persisted_fields=("other_boundary_accumulation_state",),
    enrichment_items=("44",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25F = PhaseDefinition(
    phase_id="25f",
    display_name="仮説-観測対構成",
    band=Band.EVERY_5_TICKS,
    band_order=23,
    modules=("hypothesis_observation_pairing",),
    reads=("_last_other_model", "_real_feed_processor"),
    writes=(),
    persisted_fields=("hypothesis_observation_pairing_state",),
    enrichment_items=("48",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_25 = PhaseDefinition(
    phase_id="25",
    display_name="他者モデル仮説更新",
    band=Band.EVERY_5_TICKS,
    band_order=24,
    modules=("other_agent_model",),
    reads=(
        "_last_percept", "_loop_state", "_dynamics", "_psyche",
        "_last_feed_result", "_dialogue_learning_processor",
        "_last_self_view",
    ),
    writes=("_last_other_model", "_input_supply"),
    persisted_fields=("last_other_model", "input_supply"),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26 = PhaseDefinition(
    phase_id="26",
    display_name="価値指向更新",
    band=Band.EVERY_5_TICKS,
    band_order=25,
    modules=("value_orientation",),
    reads=("_psyche", "_action_result_observer"),
    writes=("_value_orientation",),
    persisted_fields=("value_orientation",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26_EXP = PhaseDefinition(
    phase_id="26-exp",
    display_name="経験強度帯域拡大",
    band=Band.EVERY_5_TICKS,
    band_order=26,
    modules=("value_orientation",),
    reads=("_psyche", "_value_orientation", "_last_episodes"),
    writes=("_value_orientation",),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26B = PhaseDefinition(
    phase_id="26b",
    display_name="価値方向性検証",
    band=Band.EVERY_5_TICKS,
    band_order=27,
    modules=("value_orientation_validation",),
    reads=(),
    writes=("_last_vo_validation",),
    persisted_fields=("vo_validation_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26C = PhaseDefinition(
    phase_id="26c",
    display_name="行動-結果対処理",
    band=Band.EVERY_5_TICKS,
    band_order=28,
    modules=("action_result_observation",),
    reads=(),
    writes=("_last_action_result",),
    persisted_fields=("action_result_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26C2 = PhaseDefinition(
    phase_id="26c2",
    display_name="行動多様性記述",
    band=Band.EVERY_5_TICKS,
    band_order=29,
    modules=("behavioral_diversity_description",),
    reads=("_action_result_observer", "_selection_attribution_recorder"),
    writes=("_behavioral_diversity_state",),
    persisted_fields=("behavioral_diversity_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26D = PhaseDefinition(
    phase_id="26d",
    display_name="予期差分照合",
    band=Band.EVERY_5_TICKS,
    band_order=30,
    modules=(),
    reads=("_last_expectations", "_action_result_observer"),
    writes=("_expectation_action_diff_log",),
    persisted_fields=("expectation_action_diff_log",),
    enrichment_items=("25",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26E = PhaseDefinition(
    phase_id="26e",
    display_name="意図-行動乖離認知",
    band=Band.EVERY_5_TICKS,
    band_order=31,
    modules=("intent_action_gap",),
    reads=("_self_action_recorder", "_last_selected_policy_axis"),
    writes=(),
    persisted_fields=("intent_action_gap_state",),
    enrichment_items=("26",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26F = PhaseDefinition(
    phase_id="26f",
    display_name="予期成立・消失記述",
    band=Band.EVERY_5_TICKS,
    band_order=32,
    modules=("expectation_lifecycle_description",),
    reads=("_last_expectations",),
    writes=(),
    persisted_fields=("expectation_lifecycle_state",),
    enrichment_items=("40",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26G = PhaseDefinition(
    phase_id="26g",
    display_name="責任推移記述",
    band=Band.EVERY_5_TICKS,
    band_order=33,
    modules=("responsibility_temporal_trace",),
    reads=("_responsibility_mgr", "_dispersion_state"),
    writes=(),
    persisted_fields=("responsibility_temporal_trace_state",),
    enrichment_items=("42",),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)

PHASE_26H = PhaseDefinition(
    phase_id="26h",
    display_name="目的階層変化記述",
    band=Band.EVERY_5_TICKS,
    band_order=34,
    modules=("goal_hierarchy_propagation",),
    reads=("_transient_goal_mgr", "_persistent_commitment", "_value_orientation"),
    writes=(),
    persisted_fields=("goal_hierarchy_propagation_state",),
    enrichment_items=(),
    method_name="_run_every_5_ticks",
    error_absorbed=True,
)


# --- 帯域: EVERY_10_TICKS (_run_every_10_ticks) ---

PHASE_27 = PhaseDefinition(
    phase_id="27",
    display_name="極端偏り検出",
    band=Band.EVERY_10_TICKS,
    band_order=0,
    modules=("stability_valve",),
    reads=("_psyche", "_value_orientation"),
    writes=("_stability_valve",),
    persisted_fields=("stability_valve",),
    enrichment_items=(),
    method_name="_run_every_10_ticks",
    error_absorbed=True,
)

PHASE_28 = PhaseDefinition(
    phase_id="28",
    display_name="長期行動ログ",
    band=Band.EVERY_10_TICKS,
    band_order=1,
    modules=("long_term_dynamics",),
    reads=("_psyche", "_value_orientation"),
    writes=("_dynamics_observer",),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_run_every_10_ticks",
    error_absorbed=True,
)

PHASE_29 = PhaseDefinition(
    phase_id="29",
    display_name="スナップショット",
    band=Band.EVERY_10_TICKS,
    band_order=2,
    modules=(),
    reads=(),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_run_every_10_ticks",
    error_absorbed=False,
)


# --- 帯域: CANDIDATE_GENERATION (_generate_final_candidates) ---

PHASE_31 = PhaseDefinition(
    phase_id="31",
    display_name="判断バイアス計算",
    band=Band.CANDIDATE_GENERATION,
    band_order=0,
    modules=("decision_bias",),
    reads=("_loop_state", "_dynamics"),
    writes=("_last_decision_bias",),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=False,
)

PHASE_30 = PhaseDefinition(
    phase_id="30",
    display_name="候補ポリシー生成",
    band=Band.CANDIDATE_GENERATION,
    band_order=1,
    modules=("thought",),
    reads=("_psyche", "_last_percept"),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=False,
)

PHASE_30B = PhaseDefinition(
    phase_id="30b",
    display_name="候補拡張",
    band=Band.CANDIDATE_GENERATION,
    band_order=2,
    modules=("policy_candidate_expansion",),
    reads=(),
    writes=(),
    persisted_fields=("policy_expansion_state",),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=True,
)

PHASE_32 = PhaseDefinition(
    phase_id="32",
    display_name="トーン修飾子",
    band=Band.CANDIDATE_GENERATION,
    band_order=3,
    modules=("tone",),
    reads=("_psyche",),
    writes=("_last_tone_mod",),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=False,
)

PHASE_33 = PhaseDefinition(
    phase_id="33",
    display_name="空気読みバイアス",
    band=Band.CANDIDATE_GENERATION,
    band_order=4,
    modules=("context_sensitivity",),
    reads=("_input_supply",),
    writes=("_last_sensitivity_bias",),
    persisted_fields=("context_sensitivity_state",),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=False,
)

PHASE_34 = PhaseDefinition(
    phase_id="34",
    display_name="沈黙候補生成",
    band=Band.CANDIDATE_GENERATION,
    band_order=5,
    modules=("silence_hesitation",),
    reads=("_psyche", "_last_percept"),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=False,
)

PHASE_35 = PhaseDefinition(
    phase_id="35",
    display_name="安定化バイアス適用",
    band=Band.CANDIDATE_GENERATION,
    band_order=6,
    modules=("stability_valve",),
    reads=("_stability_valve",),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=True,
)

PHASE_35B = PhaseDefinition(
    phase_id="35b",
    display_name="価値軸バイアス適用",
    band=Band.CANDIDATE_GENERATION,
    band_order=7,
    modules=("value_orientation",),
    reads=("_value_orientation",),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=True,
)

PHASE_35B2 = PhaseDefinition(
    phase_id="35b2",
    display_name="持続的取り組みバイアス",
    band=Band.CANDIDATE_GENERATION,
    band_order=8,
    modules=("persistent_commitment",),
    reads=("_persistent_commitment",),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=True,
)

PHASE_35C = PhaseDefinition(
    phase_id="35c",
    display_name="スコアリング揺らぎ",
    band=Band.CANDIDATE_GENERATION,
    band_order=9,
    modules=("scoring_fluctuation",),
    reads=("_psyche", "_loop_state"),
    writes=(),
    persisted_fields=(),
    enrichment_items=(),
    method_name="_generate_final_candidates",
    error_absorbed=True,
)


# --- 帯域: POST_SELECTION (select_policy_dict内、選択後) ---

PHASE_PS1 = PhaseDefinition(
    phase_id="ps-1",
    display_name="価値軸フィードバック",
    band=Band.POST_SELECTION,
    band_order=0,
    modules=("value_orientation",),
    reads=(),
    writes=("_value_orientation",),
    persisted_fields=("value_orientation",),
    enrichment_items=(),
    method_name="select_policy_dict",
    error_absorbed=True,
)

PHASE_PS2 = PhaseDefinition(
    phase_id="ps-2",
    display_name="選択帰属記録",
    band=Band.POST_SELECTION,
    band_order=1,
    modules=("selection_attribution",),
    reads=(
        "_last_decision_bias", "_last_sensitivity_bias",
        "_stability_valve", "_value_orientation",
        "_persistent_commitment",
    ),
    writes=(),
    persisted_fields=("selection_attribution_state",),
    enrichment_items=("31",),
    method_name="select_policy_dict",
    error_absorbed=True,
)


# ── 全Phase定義の集約 ────────────────────────────────────────

ALL_PHASES: tuple[PhaseDefinition, ...] = (
    # EVERY_TICK
    PHASE_1, PHASE_2, PHASE_2A, PHASE_2B, PHASE_2C,
    PHASE_3, PHASE_4, PHASE_5, PHASE_6, PHASE_7,
    PHASE_7A, PHASE_7B, PHASE_7C, PHASE_7D, PHASE_7E, PHASE_7F,
    # EVERY_3_TICKS
    PHASE_8, PHASE_9, PHASE_10, PHASE_11, PHASE_12, PHASE_12B,
    PHASE_13, PHASE_14, PHASE_14B, PHASE_14C, PHASE_14D, PHASE_14E,
    PHASE_14F, PHASE_14G, PHASE_14H, PHASE_14I, PHASE_14J,
    # EVERY_5_TICKS
    PHASE_15, PHASE_15B, PHASE_16, PHASE_17, PHASE_18, PHASE_19, PHASE_20,
    PHASE_21, PHASE_21B, PHASE_21C, PHASE_21D, PHASE_21E, PHASE_21F,
    PHASE_21G, PHASE_21H,
    PHASE_22, PHASE_23, PHASE_24, PHASE_24B,
    PHASE_25A, PHASE_25C, PHASE_25D, PHASE_25E, PHASE_25F, PHASE_25,
    PHASE_26, PHASE_26_EXP, PHASE_26B, PHASE_26C, PHASE_26C2, PHASE_26D, PHASE_26E,
    PHASE_26F, PHASE_26G, PHASE_26H,
    # EVERY_10_TICKS
    PHASE_27, PHASE_28, PHASE_29,
    # CANDIDATE_GENERATION
    PHASE_31, PHASE_30, PHASE_30B, PHASE_32, PHASE_33, PHASE_34,
    PHASE_35, PHASE_35B, PHASE_35B2, PHASE_35C,
    # POST_SELECTION
    PHASE_PS1, PHASE_PS2,
)

# Phase IDでの索引 (導出データ)
PHASE_BY_ID: dict[str, PhaseDefinition] = {p.phase_id: p for p in ALL_PHASES}


# ── 帯域定義の集約 ────────────────────────────────────────────

def _phases_for_band(band: Band) -> tuple[str, ...]:
    """指定帯域に所属するPhase IDを帯域内順序で返す。"""
    matching = sorted(
        (p for p in ALL_PHASES if p.band == band),
        key=lambda p: p.band_order,
    )
    return tuple(p.phase_id for p in matching)


BAND_EVERY_TICK = BandDefinition(
    band=Band.EVERY_TICK,
    execution_condition="every_tick",
    execution_method="_run_every_tick",
    phase_ids=_phases_for_band(Band.EVERY_TICK),
)

BAND_EVERY_3_TICKS = BandDefinition(
    band=Band.EVERY_3_TICKS,
    execution_condition="tick % 3 == 0",
    execution_method="_run_every_3_ticks",
    phase_ids=_phases_for_band(Band.EVERY_3_TICKS),
)

BAND_EVERY_5_TICKS = BandDefinition(
    band=Band.EVERY_5_TICKS,
    execution_condition="tick % 5 == 0",
    execution_method="_run_every_5_ticks",
    phase_ids=_phases_for_band(Band.EVERY_5_TICKS),
)

BAND_EVERY_10_TICKS = BandDefinition(
    band=Band.EVERY_10_TICKS,
    execution_condition="tick % 10 == 0",
    execution_method="_run_every_10_ticks",
    phase_ids=_phases_for_band(Band.EVERY_10_TICKS),
)

BAND_CANDIDATE_GENERATION = BandDefinition(
    band=Band.CANDIDATE_GENERATION,
    execution_condition="on_candidate_generation",
    execution_method="_generate_final_candidates",
    phase_ids=_phases_for_band(Band.CANDIDATE_GENERATION),
)

BAND_POST_SELECTION = BandDefinition(
    band=Band.POST_SELECTION,
    execution_condition="on_post_selection",
    execution_method="select_policy_dict",
    phase_ids=_phases_for_band(Band.POST_SELECTION),
)

ALL_BANDS: tuple[BandDefinition, ...] = (
    BAND_EVERY_TICK,
    BAND_EVERY_3_TICKS,
    BAND_EVERY_5_TICKS,
    BAND_EVERY_10_TICKS,
    BAND_CANDIDATE_GENERATION,
    BAND_POST_SELECTION,
)


# ── 永続化フィールド一覧 (導出データ) ──────────────────────────

def get_all_persisted_fields() -> set[str]:
    """全Phase定義から永続化対象フィールドの集合を導出する。"""
    fields: set[str] = set()
    for p in ALL_PHASES:
        fields.update(p.persisted_fields)
    return fields


# ── enrichment項目一覧 (導出データ) ──────────────────────────

def get_all_enrichment_items() -> set[str]:
    """全Phase定義からenrichment項目番号の集合を導出する。"""
    items: set[str] = set()
    for p in ALL_PHASES:
        items.update(p.enrichment_items)
    return items


# ── データ依存グラフ (導出データ) ──────────────────────────────

@dataclass(frozen=True)
class DataDependency:
    """Phase間のデータ依存を表現するレコード。"""
    consumer_phase_id: str      # 読み取り側Phase
    producer_phase_id: str      # 書き込み側Phase
    intermediate_state: str     # 介在する中間状態変数名
    same_band: bool             # 同一帯域内の依存か


def compute_data_dependencies() -> tuple[DataDependency, ...]:
    """全Phase定義から前方依存関係を機械的に導出する。

    処理Aが書き込む中間状態を処理Bが読み取る場合、B→Aの前方依存が存在する。

    同一帯域内の依存: 生産者(writer)のband_orderが消費者(reader)より小さい場合のみ
    記録する。同一帯域内で消費者が先に実行される場合、消費者は前回ティック末尾の
    状態を参照しており、当該ティック内の生産者への依存ではない。

    帯域間の依存: 異なる帯域間では実行タイミングが異なるため、全組み合わせを記録する。

    Returns:
        データ依存レコードのタプル
    """
    # 書き込み側の索引を構築: 中間状態名 → 書き込むPhase ID群
    writers: dict[str, list[str]] = {}
    for p in ALL_PHASES:
        for w in p.writes:
            writers.setdefault(w, []).append(p.phase_id)

    deps: list[DataDependency] = []
    for p in ALL_PHASES:
        for r in p.reads:
            if r in writers:
                for writer_id in writers[r]:
                    if writer_id == p.phase_id:
                        continue  # 自己参照はスキップ
                    writer_phase = PHASE_BY_ID[writer_id]
                    same_band = (p.band == writer_phase.band)

                    # 同一帯域内: 生産者が消費者より先に実行される場合のみ
                    if same_band and writer_phase.band_order >= p.band_order:
                        continue

                    deps.append(DataDependency(
                        consumer_phase_id=p.phase_id,
                        producer_phase_id=writer_id,
                        intermediate_state=r,
                        same_band=same_band,
                    ))

    return tuple(deps)


def get_intra_band_dependencies() -> dict[Band, list[DataDependency]]:
    """同一帯域内のデータ依存を帯域別に返す。"""
    all_deps = compute_data_dependencies()
    result: dict[Band, list[DataDependency]] = {b: [] for b in Band}
    for d in all_deps:
        if d.same_band:
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            result[consumer.band].append(d)
    return result


def get_cross_band_dependencies() -> list[DataDependency]:
    """帯域間のデータ依存を返す。"""
    return [d for d in compute_data_dependencies() if not d.same_band]
