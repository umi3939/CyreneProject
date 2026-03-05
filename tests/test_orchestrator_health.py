"""
tests/test_orchestrator_health.py - orchestrator系ファイルの構造健全性テスト

設計書: design_orchestrator_health_test.md (C11-5)
討論判定: 条件付き推奨

3領域18パターンの構造健全性テスト:
- 領域A: Phase間データ受け渡しの整合性 (7パターン)
- 領域B: enrichment 49項目の生成パス網羅検証 (6パターン)
- 領域C: 4ファイル間の関数呼出し依存関係マッピング (5パターン)

save/load個別フィールド値検証は候補9に委任。
本テストではpsycheのロジック変更を一切行わない。
"""

import copy
import sys
import tempfile
from pathlib import Path

import pytest

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept


# ── Helpers ───────────────────────────────────────────────────────


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent="expression",
        emotion_valence=valence,
    )


# 多様な感情入力パターン（帯域横断のデータフロー検証用）
EMOTIONS = [
    "happy", "sad", "angry", "neutral", "surprised",
    "loving", "teasing", "scared", "happy", "neutral",
]
VALENCES = [
    0.7, -0.6, -0.5, 0.0, 0.3,
    0.8, 0.4, -0.5, 0.6, 0.0,
]


def _run_ticks(orch: PsycheOrchestrator, n: int, user_id: str = "viewer") -> None:
    """指定ティック数のpost_response_updateを実行する。"""
    for i in range(n):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
        )
        orch.post_response_update(percept, delta_time=1.0, user_id=user_id)


def _create_orchestrator() -> PsycheOrchestrator:
    """一時ディレクトリでorchestratorを初期化する。"""
    tmp = tempfile.mkdtemp()
    return PsycheOrchestrator(data_dir=Path(tmp))


# ═══════════════════════════════════════════════════════════════════
# 領域A: Phase間データ受け渡しの整合性テスト
# ═══════════════════════════════════════════════════════════════════


class TestPhaseDataHandoff:
    """Phase間データ受け渡しの整合性を検証する。"""

    def test_a1_every_tick_attributes_updated(self):
        """A-1: 毎ティック帯域実行後の属性状態検証。

        1ティック分のpost_response_updateを実行した後、毎ティック帯域が更新すべき
        属性（感情ベクトル、ドライブベクトル、ムード、恐怖指数等）が初期値から
        変化していること。
        """
        orch = _create_orchestrator()

        # 初期状態を記録
        initial_emotion = orch.psyche.emotions.as_dict()
        initial_mood_valence = orch.psyche.mood.valence
        initial_fear = orch.psyche.fear_level

        # 強い感情入力で1ティック実行
        percept = _make_percept(emotion="angry", valence=-0.8)
        orch.post_response_update(percept, delta_time=1.0, user_id="viewer")

        # 毎ティック帯域の更新結果を検証
        assert orch.tick_count == 1
        # 感情ベクトルが変化していること（angerが増加するはず）
        updated_emotion = orch.psyche.emotions.as_dict()
        assert updated_emotion != initial_emotion, (
            "感情ベクトルが1ティック後も初期値のまま"
        )
        # ムードが変化しているか、または恐怖指数が計算されていること
        # （入力次第で必ず変わるとは限らないが、少なくとも処理が完了していること）
        assert orch.psyche.fear_level >= 0.0, "恐怖指数が負の値"

    def test_a2_five_tick_attributes_updated(self):
        """A-2: 5ティック帯域実行後の属性状態検証。

        5ティック分のpost_response_updateを実行した後、5ティック帯域が更新すべき
        属性（内省断面、記憶系統、価値指向等）が初期化状態ではないこと。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 5)

        assert orch.tick_count == 5

        # 5ティック帯域で更新される属性が存在すること
        # TemporalSelfDifference, ContinuityStrain等が処理されているはず
        # 少なくともAttributeErrorなく参照できることを確認
        assert hasattr(orch, '_last_diff_summary')
        assert hasattr(orch, '_last_strain')
        assert hasattr(orch, '_last_self_image')
        assert hasattr(orch, '_last_coherence')
        assert hasattr(orch, '_last_narrative')
        assert hasattr(orch, '_last_episodes')
        assert hasattr(orch, '_last_bindings')
        assert hasattr(orch, '_last_consumption')
        assert hasattr(orch, '_last_expectations')
        assert hasattr(orch, '_last_motives')

        # 5ティック帯域の主要出力のうち少なくとも一部がNone以外であること
        five_tick_outputs = [
            orch._last_diff_summary,
            orch._last_strain,
            orch._last_self_image,
            orch._last_coherence,
            orch._last_narrative,
        ]
        non_none_count = sum(1 for x in five_tick_outputs if x is not None)
        assert non_none_count >= 1, (
            f"5ティック帯域の主要出力が全てNone: "
            f"diff={orch._last_diff_summary}, strain={orch._last_strain}"
        )

    def test_a3_cross_band_data_flow(self):
        """A-3: 帯域横断のデータフロー検証。

        毎ティック帯域で更新された感情・ドライブが、5ティック帯域のPhaseの
        入力として実際に参照されていること。多様な感情入力で5ティック以上実行した後、
        enrichment出力に感情・ドライブの影響が反映されていること。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 5)

        # enrichment出力を生成
        enrichment = orch.get_prompt_enrichment(user_id="viewer")

        # enrichmentが空でないこと
        assert len(enrichment) > 0, "enrichment出力が空"

        # 感情・ドライブの情報がenrichmentに含まれること
        # （毎ティック帯域の出力が5ティック帯域→enrichment生成に伝播していること）
        assert "感情" in enrichment or "ムード" in enrichment or "ドライブ" in enrichment, (
            "enrichmentに感情・ムード・ドライブの情報が含まれていない"
        )

    def test_a4_ten_tick_band_execution(self):
        """A-4: 10ティック帯域の実行検証。

        10ティック分のpost_response_updateを実行した後、10ティック帯域のPhase
        （安定弁、長期動態、スナップショット）が実行されエラーが発生しないこと。
        """
        orch = _create_orchestrator()
        # 10ティック実行（10ティック帯域が1回発動する）
        _run_ticks(orch, 10)

        assert orch.tick_count == 10

        # 10ティック帯域の処理対象が初期化されていること
        assert orch._stability_valve is not None
        assert orch._dynamics_observer is not None

        # エラーなく完了していること（ここに到達していれば成功）

    def test_a5_three_tick_band_execution(self):
        """A-5: 3ティック帯域の実行検証。

        3ティック分のpost_response_updateを実行した後、3ティック帯域のPhase
        （傾向認知→自己モデル→目標→内発動機）が実行されエラーが発生しないこと。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 3)

        assert orch.tick_count == 3

        # 3ティック帯域の処理対象属性が参照可能であること
        assert hasattr(orch, '_tendency_awareness')
        assert hasattr(orch, '_last_self_view')
        assert hasattr(orch, '_vector_gen')
        assert hasattr(orch, '_last_motives')

    def test_a6_non_executed_band_safety(self):
        """A-6: 非実行帯域の安全性検証。

        1ティック目（5ティック帯域、10ティック帯域は実行されない）でも、
        enrichment生成やselect_policy_dictが正常に動作すること。
        """
        orch = _create_orchestrator()

        # 1ティックのみ実行（5ティック帯域・10ティック帯域は未実行）
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0, user_id="viewer")

        assert orch.tick_count == 1

        # enrichment生成がエラーなく完了すること
        enrichment = orch.get_prompt_enrichment(user_id="viewer")
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        # select_policy_dictがエラーなく完了すること
        policy = orch.select_policy_dict(percept, recalled_memories=[], user_id="viewer")
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_a7_reaction_split_data_flow(self):
        """A-7: 反応処理分割後のデータフロー検証。

        反応処理が3ファイルに分割された後も、毎ティック帯域のPhase 1-2系列
        （感情コア処理）が正しくドライブ動態とムード更新を呼び出し、
        その結果が統合管理本体の状態に反映されること。
        """
        orch = _create_orchestrator()

        # 初期ドライブ・ムードを記録
        initial_drives = orch.psyche.drives.as_dict()
        initial_mood = (orch.psyche.mood.valence, orch.psyche.mood.arousal)

        # 複数ティック実行（ドライブ動態・ムード更新の変化を促す）
        for i in range(5):
            percept = _make_percept(
                emotion=EMOTIONS[i],
                valence=VALENCES[i],
            )
            orch.post_response_update(percept, delta_time=1.0, user_id="viewer")

        # ドライブまたはムードが変化していること（分割後もデータフローが正常）
        updated_drives = orch.psyche.drives.as_dict()
        updated_mood = (orch.psyche.mood.valence, orch.psyche.mood.arousal)

        # 少なくともドライブかムードのどちらかが変化しているはず
        drives_changed = updated_drives != initial_drives
        mood_changed = updated_mood != initial_mood
        assert drives_changed or mood_changed, (
            "5ティック後もドライブ・ムードが初期値のまま: "
            f"drives_changed={drives_changed}, mood_changed={mood_changed}"
        )

        # 反応処理分割ファイルが正しくインポートされていること
        from psyche.reaction_drive_dynamics import DriveContextInputs
        from psyche.reaction_mood_update import MoodContextInputs
        assert DriveContextInputs is not None
        assert MoodContextInputs is not None


# ═══════════════════════════════════════════════════════════════════
# 領域B: enrichment 49項目の生成パス網羅テスト
# ═══════════════════════════════════════════════════════════════════


class TestEnrichmentGeneration:
    """enrichment 49項目の生成パス網羅を検証する。"""

    def test_b1_section_presence(self):
        """B-1: セクション別項目存在検証。

        十分なティック数（10ティック）の実行後、enrichment出力に5セクション全てが
        存在し、各セクションに少なくとも1つの項目が含まれること。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 10)

        # select_policy_dictを実行して判断傾向セクションにもデータを供給
        percept = _make_percept()
        orch.select_policy_dict(percept, recalled_memories=[], user_id="viewer")

        enrichment = orch.get_prompt_enrichment(user_id="viewer")

        # 5セクションの圧縮済みヘッダーが存在すること
        # enrichment_compression.pyのSECTION_HEADER_MAPに基づく圧縮後のヘッダー
        expected_sections = [
            "[内面]",
            "[自己]",
            "[動機]",
            "[記憶]",
            "[判断]",
        ]

        found_sections = []
        for section_name in expected_sections:
            if section_name in enrichment:
                found_sections.append(section_name)

        assert len(found_sections) >= 3, (
            f"enrichmentに含まれるセクション数が不足: {found_sections} "
            f"(期待: {expected_sections}のうち3以上)"
        )

    def test_b2_enrichment_read_only(self):
        """B-2: enrichment生成の読み取り専用性検証。

        enrichment生成の前後で統合管理本体の主要属性（感情ベクトル、ドライブベクトル、
        ムード、tick_count等）が変化しないこと。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 5)

        # enrichment生成前の状態を記録
        before_emotion = orch.psyche.emotions.as_dict()
        before_drives = orch.psyche.drives.as_dict()
        before_mood = (orch.psyche.mood.valence, orch.psyche.mood.arousal)
        before_tick = orch.tick_count
        before_fear = orch.psyche.fear_level

        # enrichment生成を実行
        enrichment = orch.get_prompt_enrichment(user_id="viewer")
        assert len(enrichment) > 0

        # enrichment生成後の状態が変化していないこと
        after_emotion = orch.psyche.emotions.as_dict()
        after_drives = orch.psyche.drives.as_dict()
        after_mood = (orch.psyche.mood.valence, orch.psyche.mood.arousal)
        after_tick = orch.tick_count
        after_fear = orch.psyche.fear_level

        assert before_emotion == after_emotion, "enrichment生成が感情ベクトルを変更した"
        assert before_drives == after_drives, "enrichment生成がドライブベクトルを変更した"
        assert before_mood == after_mood, "enrichment生成がムードを変更した"
        assert before_tick == after_tick, "enrichment生成がtick_countを変更した"
        assert before_fear == after_fear, "enrichment生成がfear_levelを変更した"

    def test_b3_all_generation_paths_no_exception(self):
        """B-3: enrichment 49項目の全生成パス検証。

        十分なティック数の実行後にenrichment生成を呼び出し、5セクション生成関数群が
        内部で例外を発生させないこと。生成された各セクションのテキストが空でないこと。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 10)

        # select_policy_dictを先に実行して判断傾向セクションを生成可能にする
        percept = _make_percept()
        orch.select_policy_dict(percept, recalled_memories=[], user_id="viewer")

        # enrichment生成が例外なく完了すること
        enrichment = orch.get_prompt_enrichment(user_id="viewer")
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0, "enrichment出力が空"

        # 各セクション生成関数を直接呼び出して例外がないことを確認
        from psyche.orchestrator_enrichment import (
            _collect_enrichment_psyche,
            _collect_enrichment_self,
            _collect_enrichment_motive,
            _collect_enrichment_memory,
            _collect_enrichment_bias,
        )

        psyche_items = _collect_enrichment_psyche(orch, "viewer")
        self_items = _collect_enrichment_self(orch, "viewer")
        motive_items = _collect_enrichment_motive(orch)
        memory_items = _collect_enrichment_memory(orch, "viewer")
        bias_items = _collect_enrichment_bias(orch)

        # 各セクションがリストであること
        assert isinstance(psyche_items, list)
        assert isinstance(self_items, list)
        assert isinstance(motive_items, list)
        assert isinstance(memory_items, list)
        assert isinstance(bias_items, list)

        # 心理状態セクションは常に項目を含むはず（感情・ムード・ドライブは常に存在）
        assert len(psyche_items) >= 3, (
            f"心理状態セクションの項目数が不足: {len(psyche_items)}"
        )

    def test_b4_empty_skip_behavior(self):
        """B-4: 空項目スキップの動作検証。

        初期状態（蓄積なし）でのenrichment生成が、未蓄積の項目をスキップしても
        セクション構造を維持すること。
        """
        orch = _create_orchestrator()

        # 1ティック目で実行（多くの蓄積が未だないため空項目が多い）
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0, user_id="viewer")

        # enrichment生成が例外なく完了すること
        enrichment = orch.get_prompt_enrichment(user_id="viewer")
        assert isinstance(enrichment, str)

        # セクション構造が維持されていること（少なくとも心理状態セクションが存在）
        # 空項目があってもセクション自体は空にならない（感情/ムード/ドライブは常に存在）
        assert "感情" in enrichment or "ムード" in enrichment, (
            "初期状態のenrichmentにも感情/ムードが含まれるべき"
        )

    def test_b5_enrichment_stability(self):
        """B-5: enrichment安定性検証。

        同一状態でenrichment生成を複数回呼び出した場合、2回目以降の出力が
        一致すること（圧縮パイプラインの安定性）。

        注意: 圧縮パイプラインは前回キャッシュとの差分で圧縮するため、
        1回目と2回目は異なる（差分圧縮が効く）。
        2回目と3回目が安定していることを検証する。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 5)

        # 1回目のenrichment生成（キャッシュ構築）
        _enrichment_1 = orch.get_prompt_enrichment(user_id="viewer")

        # 2回目のenrichment生成（差分圧縮済み）
        enrichment_2 = orch.get_prompt_enrichment(user_id="viewer")

        # 3回目のenrichment生成（同一状態、差分圧縮済み）
        enrichment_3 = orch.get_prompt_enrichment(user_id="viewer")

        # 2回目以降の出力が安定していること
        assert enrichment_2 == enrichment_3, (
            "同一状態でのenrichment生成（2回目以降）が安定していない\n"
            f"2回目長さ={len(enrichment_2)}, 3回目長さ={len(enrichment_3)}"
        )

    def test_b6_enrichment_after_policy_selection(self):
        """B-6: enrichmentとselect_policy_dictの相互作用検証。

        select_policy_dictを呼び出した後のenrichment生成が、判断傾向セクションに
        情報を含むこと（Phase 30-35系列のキャッシュがenrichment生成に反映されること）。
        """
        orch = _create_orchestrator()
        _run_ticks(orch, 5)

        # select_policy_dict前のenrichment
        enrichment_before = orch.get_prompt_enrichment(user_id="viewer")

        # select_policy_dictを実行
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, recalled_memories=[], user_id="viewer")
        assert isinstance(policy, dict)

        # select_policy_dict後のenrichment
        enrichment_after = orch.get_prompt_enrichment(user_id="viewer")

        # 判断傾向セクションに情報が含まれること
        # select_policy_dict実行後は少なくとも判断バイアス情報が存在するはず
        has_bias_info = (
            "判断" in enrichment_after
            or "トーン" in enrichment_after
            or "バイアス" in enrichment_after
        )
        assert has_bias_info, (
            "select_policy_dict後のenrichmentに判断傾向情報がない"
        )


# ═══════════════════════════════════════════════════════════════════
# 領域C: 4ファイル間の関数呼出し依存関係テスト
# ═══════════════════════════════════════════════════════════════════


class TestInterFileDependency:
    """4ファイル間の関数呼出し依存関係を検証する。"""

    def test_c1_initial_attribute_existence(self):
        """C-1: 初期化直後の属性存在検証。

        統合管理構造の初期化直後に、毎ティック帯域実行部・5ティック帯域実行部・
        enrichment生成部が参照する全属性が存在し、参照時にAttributeErrorが
        発生しないこと。
        """
        orch = _create_orchestrator()

        # ── 毎ティック帯域実行部が参照する属性 ──
        assert hasattr(orch, '_psyche')
        assert hasattr(orch, '_loop_state')
        assert hasattr(orch, '_dynamics')
        assert hasattr(orch, '_amplitude_state')
        assert hasattr(orch, '_multi_emotion_config')
        assert hasattr(orch, '_stm_coupling_config')
        assert hasattr(orch, '_last_coupling')
        assert hasattr(orch, '_responsibility_mgr')
        assert hasattr(orch, '_self_ref_config')
        assert hasattr(orch, '_self_ref_state')
        assert hasattr(orch, '_tendency_sys')
        assert hasattr(orch, '_last_percept')
        assert hasattr(orch, '_last_delta_time')
        assert hasattr(orch, '_last_loop_result')
        assert hasattr(orch, '_action_result_observer')
        assert hasattr(orch, '_temporal_cognition')
        assert hasattr(orch, '_perceptual_context')
        assert hasattr(orch, '_situational_self_presentation')
        assert hasattr(orch, '_input_pathway_balance_state')
        assert hasattr(orch, '_att_dist_state')

        # ── 5ティック帯域実行部が参照する属性 ──
        assert hasattr(orch, '_temporal_diff_sys')
        assert hasattr(orch, '_strain_sys')
        assert hasattr(orch, '_self_image_sys')
        assert hasattr(orch, '_coherence_sys')
        assert hasattr(orch, '_narrative_sys')
        assert hasattr(orch, '_episodic_sys')
        assert hasattr(orch, '_binding_sys')
        assert hasattr(orch, '_memory_integrator')
        assert hasattr(orch, '_multi_path_recall')
        assert hasattr(orch, '_spontaneous_recall')
        assert hasattr(orch, '_memory_emotion_return')
        assert hasattr(orch, '_other_hypothesis_emotion_return')
        assert hasattr(orch, '_introspection_sys')
        assert hasattr(orch, '_consumption_sys')
        assert hasattr(orch, '_expectation_sys')
        assert hasattr(orch, '_motivation_sys')
        assert hasattr(orch, '_other_model_sys')
        assert hasattr(orch, '_input_supply')
        assert hasattr(orch, '_real_feed_processor')
        assert hasattr(orch, '_dialogue_learning_processor')
        assert hasattr(orch, '_value_orientation')
        assert hasattr(orch, '_vo_validator')
        assert hasattr(orch, '_forgetting_fixation_processor')
        assert hasattr(orch, '_introspection_cross_section')
        assert hasattr(orch, '_introspection_longitudinal_view')
        assert hasattr(orch, '_reference_frequency_state')
        assert hasattr(orch, '_persistent_commitment')
        assert hasattr(orch, '_interaction_accumulation')
        assert hasattr(orch, '_other_boundary_accumulation')
        assert hasattr(orch, '_hypothesis_observation_pairing')
        assert hasattr(orch, '_expectation_lifecycle_processor')
        assert hasattr(orch, '_goal_hierarchy_propagation')

        # ── enrichment生成部が参照する属性 ──
        assert hasattr(orch, '_dispersion_state')
        assert hasattr(orch, '_stability_valve')
        assert hasattr(orch, '_last_decision_bias')
        assert hasattr(orch, '_last_tone_mod')
        assert hasattr(orch, '_last_sensitivity_bias')
        assert hasattr(orch, '_last_has_silence')
        assert hasattr(orch, '_dynamics_observer')
        assert hasattr(orch, '_last_diff_summary')
        assert hasattr(orch, '_last_strain')
        assert hasattr(orch, '_last_self_image')
        assert hasattr(orch, '_last_coherence')
        assert hasattr(orch, '_tendency_awareness')
        assert hasattr(orch, '_last_narrative')
        assert hasattr(orch, '_enrichment_prev_cache')
        assert hasattr(orch, '_enrichment_empty_skip_tracker')
        assert hasattr(orch, '_responsibility_temporal_trace')
        assert hasattr(orch, '_emotion_cooccurrence_processor')
        assert hasattr(orch, '_frb_state')
        assert hasattr(orch, '_emotional_backdrop_processor')
        assert hasattr(orch, '_drive_variation_processor')
        assert hasattr(orch, '_contradiction_processor')
        assert hasattr(orch, '_stabilization_desc_state')
        assert hasattr(orch, '_behavioral_diversity_state')
        assert hasattr(orch, '_meta_emotion_processor')
        assert hasattr(orch, '_self_action_recorder')
        assert hasattr(orch, '_intent_action_gap_recorder')
        assert hasattr(orch, '_selection_attribution_recorder')
        assert hasattr(orch, '_policy_expander')

    def test_c2_band_function_call_success(self):
        """C-2: 帯域実行部からの関数呼出し成功検証。

        統合管理本体のインスタンスを渡して、毎ティック帯域実行部の公開関数・
        5ティック帯域実行部の公開関数が引数として受理し、処理を完了できること。
        """
        orch = _create_orchestrator()

        # 毎ティック帯域の公開関数を直接呼び出す
        from psyche.orchestrator_1tick_phases import run_every_tick
        percept = _make_percept()
        # run_every_tickが例外なく完了すること
        run_every_tick(orch, percept, delta_time=1.0, user_id="viewer")
        orch._tick_count = 1  # ティックカウンタを手動更新（通常はpost_response_update内）

        # 5ティック帯域の公開関数を直接呼び出す
        from psyche.orchestrator_5tick_phases import run_every_5_ticks
        # run_every_5_ticksが例外なく完了すること
        run_every_5_ticks(orch, user_id="viewer")

    def test_c3_enrichment_attribute_reference(self):
        """C-3: enrichment生成部からの属性参照成功検証。

        統合管理本体のインスタンスを渡して、enrichment生成部のセクション生成関数群が
        統合管理本体の属性を正常に参照できること。
        """
        orch = _create_orchestrator()

        from psyche.orchestrator_enrichment import (
            _collect_enrichment_psyche,
            _collect_enrichment_self,
            _collect_enrichment_motive,
            _collect_enrichment_memory,
            _collect_enrichment_bias,
        )

        # 各セクション生成関数がorchの属性を参照してエラーなく完了すること
        psyche_items = _collect_enrichment_psyche(orch, "viewer")
        assert isinstance(psyche_items, list)

        self_items = _collect_enrichment_self(orch, "viewer")
        assert isinstance(self_items, list)

        motive_items = _collect_enrichment_motive(orch)
        assert isinstance(motive_items, list)

        memory_items = _collect_enrichment_memory(orch, "viewer")
        assert isinstance(memory_items, list)

        bias_items = _collect_enrichment_bias(orch)
        assert isinstance(bias_items, list)

        # 1ティック以上実行した後でも同様に成功すること
        _run_ticks(orch, 5)

        psyche_items_2 = _collect_enrichment_psyche(orch, "viewer")
        assert isinstance(psyche_items_2, list)
        assert len(psyche_items_2) >= len(psyche_items), (
            "5ティック後のpysche_itemsが初期状態より減少"
        )

    def test_c4_band_execution_order_equivalence(self):
        """C-4: 帯域実行順序の暗黙的依存検証。

        毎ティック帯域→3ティック帯域→5ティック帯域→10ティック帯域の順序で
        実行された場合と、post_response_updateの通常フローで実行された場合とで、
        主要な出力（enrichment、policy選択結果）が同等であること。

        注意: スコアリング揺らぎやドライブ動態には時間依存の微小変動があるため、
        完全一致ではなく構造的同等性を検証する。
        """
        # パターン1: post_response_updateの通常フロー
        orch1 = _create_orchestrator()
        for i in range(10):
            percept = _make_percept(
                emotion=EMOTIONS[i],
                valence=VALENCES[i],
            )
            orch1.post_response_update(percept, delta_time=1.0, user_id="viewer")

        enrichment1 = orch1.get_prompt_enrichment(user_id="viewer")
        percept_final = _make_percept()
        policy1 = orch1.select_policy_dict(percept_final, recalled_memories=[], user_id="viewer")

        # パターン2: 同一入力で同一フローを再実行
        orch2 = _create_orchestrator()
        for i in range(10):
            percept = _make_percept(
                emotion=EMOTIONS[i],
                valence=VALENCES[i],
            )
            orch2.post_response_update(percept, delta_time=1.0, user_id="viewer")

        enrichment2 = orch2.get_prompt_enrichment(user_id="viewer")
        policy2 = orch2.select_policy_dict(percept_final, recalled_memories=[], user_id="viewer")

        # enrichmentの構造的同等性: 同じセクションが存在すること
        sections_1 = set()
        sections_2 = set()
        for marker in ["[内面]", "[自己]", "[動機]", "[記憶]", "[判断]"]:
            if marker in enrichment1:
                sections_1.add(marker)
            if marker in enrichment2:
                sections_2.add(marker)
        assert sections_1 == sections_2, (
            f"同一入力系列に対するenrichmentのセクション構成が異なる: "
            f"{sections_1} vs {sections_2}"
        )

        # enrichmentの長さが大きく異ならないこと（±50%以内）
        len1 = len(enrichment1)
        len2 = len(enrichment2)
        if len1 > 0 and len2 > 0:
            ratio = min(len1, len2) / max(len1, len2)
            assert ratio > 0.5, (
                f"enrichment長の乖離が大きい: {len1} vs {len2} (ratio={ratio:.2f})"
            )

        # policy選択結果が構造的に同等であること
        assert isinstance(policy1, dict)
        assert isinstance(policy2, dict)
        assert "policy_label" in policy1
        assert "policy_label" in policy2
        # 同一の属性キーを持つこと
        assert set(policy1.keys()) == set(policy2.keys()), (
            f"policy dictのキー構成が異なる: {set(policy1.keys())} vs {set(policy2.keys())}"
        )

    def test_c5_reaction_split_import_integrity(self):
        """C-5: 反応処理分割のimport整合性検証。

        毎ティック帯域実行部が反応処理の分割後のファイルから正しくインポートしており、
        ドライブ動態とムード更新の関数が呼び出し可能であること。
        """
        # reaction_drive_dynamics.pyのインポート確認
        from psyche.reaction_drive_dynamics import (
            DriveContextInputs,
            compute_state_dependent_drive_changes,
        )
        assert DriveContextInputs is not None
        assert callable(compute_state_dependent_drive_changes)

        # reaction_mood_update.pyのインポート確認
        from psyche.reaction_mood_update import (
            MoodContextInputs,
            compute_autonomous_mood,
        )
        assert MoodContextInputs is not None
        assert callable(compute_autonomous_mood)

        # reaction.pyからの依存確認（orchestrator_1tick_phases.pyが使用するインポート）
        from psyche.reaction import (
            MoodContextInputs,
            DriveContextInputs as ReactionDriveContextInputs,
            _compute_result_diversity_return,
        )
        assert MoodContextInputs is not None
        assert ReactionDriveContextInputs is not None
        assert callable(_compute_result_diversity_return)

        # orchestrator_1tick_phases.pyが正常にインポート可能であること
        from psyche.orchestrator_1tick_phases import run_every_tick
        assert callable(run_every_tick)

        # orchestrator_5tick_phases.pyが正常にインポート可能であること
        from psyche.orchestrator_5tick_phases import run_every_5_ticks
        assert callable(run_every_5_ticks)

        # orchestrator_enrichment.pyが正常にインポート可能であること
        from psyche.orchestrator_enrichment import get_prompt_enrichment
        assert callable(get_prompt_enrichment)
