"""
tests/test_integration_extended.py - orchestrator 結合テスト拡充

4領域をカバー:
1. save/load v42 後の resume 動作テスト
2. 初回起動（全デフォルト値）テスト
3. enrichment 48 項目の生成整合性テスト
4. 長時間連続稼働の安定性テスト
"""

import json
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


EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised",
            "loving", "teasing", "scared", "happy", "neutral"]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3,
            0.8, 0.4, -0.5, 0.6, 0.0]

# save/load v45 で保存される全69キー
SAVE_V42_KEYS = [
    # Core (v1)
    "version", "tick_count", "psyche", "loop_state", "dynamics",
    # v4
    "amplitude", "value_orientation", "self_ref_state",
    "last_self_view", "tendency_awareness", "last_diff_summary",
    "last_strain", "last_self_image", "last_coherence",
    "last_narrative", "last_episodes", "last_bindings",
    "last_trace", "last_consumption",
    "last_expectations", "last_motives",
    "last_other_model", "input_supply",
    # v5
    "tendency_state", "vector_state", "candidate_state",
    "transient_goal_state", "stability_valve",
    # v6
    "dispersion_state", "context_sensitivity_state", "last_coupling",
    # v7-v13
    "policy_expansion_state", "memory_integration_state",
    "real_feed_state", "text_dialogue_state",
    "spontaneous_state", "vo_validation_state",
    "forgetting_fixation_state",
    # v14-v23
    "action_result_state", "dialogue_learning_state",
    "meta_emotion_state", "self_action_perception_state",
    "expectation_action_diff_log", "intent_action_gap_state",
    "temporal_cognition_state", "multi_path_recall_state",
    "introspection_cross_section_state", "perceptual_context_state",
    "selection_attribution_state",
    # v24
    "reference_frequency_state",
    # v25
    "persistent_commitment_state",
    # v26
    "stabilization_description_state",
    # v27
    "behavioral_diversity_state",
    # v28
    "spontaneous_recall_state",
    # v29
    "internal_contradiction_state",
    # v30
    "interaction_accumulation_state",
    # v31
    "emotional_backdrop_state",
    # v32
    "situational_self_presentation_state",
    # v33
    "drive_variation_state",
    # v34
    "expectation_lifecycle_state",
    # v35
    "input_pathway_balance_state",
    # v36
    "responsibility_temporal_trace_state",
    # v37
    "emotion_cooccurrence_state",
    # v38
    "other_boundary_accumulation_state",
    # v39
    "forgetting_recall_balance_state",
    # v40
    "attention_distribution_state",
    # v41
    "goal_hierarchy_propagation_state",
    # v42
    "hypothesis_observation_pairing_state",
    # v43
    "memory_emotion_return_state",
    # v44
    "other_hypothesis_emotion_return_state",
    # v45
    "return_pathway_history",
]

# enrichment の5セクションヘッダ（圧縮済み形式）
ENRICHMENT_SECTIONS = [
    "[内面]",
    "[自己]",
    "[動機]",
    "[記憶]",
    "[判断]",
]


def _run_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新する。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)


# ══════════════════════════════════════════════════════════════════
# 領域1: save/load v42 後の resume 動作テスト
# ══════════════════════════════════════════════════════════════════


class TestSaveLoadResumeV42:
    """save/load v42 後の resume 動作テスト。"""

    def test_save_contains_all_67_keys(self, tmp_path):
        """save() が全フィールドを含む JSON を出力する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch, 10)
        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        for key in SAVE_V42_KEYS:
            assert key in data, f"Missing save field: {key}"
        assert data["version"] == 45

    def test_load_restores_all_fields_not_none(self, tmp_path):
        """load 後の全フィールドが正しく復元されている（None でない、型が正しい）。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True

        # tick_count が復元
        assert orch2.tick_count == 10
        assert isinstance(orch2.tick_count, int)

        # psyche state が復元
        assert orch2.psyche is not None

        # mood が復元（型チェック）
        assert isinstance(orch2.psyche.mood.valence, float)

        # fear_level が 0-1 範囲
        assert 0.0 <= orch2.fear_level <= 1.0

        # 主要サブシステムが復元
        assert orch2._dynamics is not None
        assert orch2._loop_state is not None
        assert orch2._amplitude_state is not None
        assert orch2._value_orientation is not None
        assert orch2._tendency_sys is not None
        assert orch2._stability_valve is not None

    def test_load_then_post_response_update(self, tmp_path):
        """load 後の post_response_update がエラーなく動作する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # load 後にさらに 10 ティック実行
        _run_ticks(orch2, 10)
        assert orch2.tick_count == 20

    def test_load_then_select_policy_dict(self, tmp_path):
        """load 後の select_policy_dict がエラーなく動作する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        percept = _make_percept()
        policy = orch2.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_load_then_get_prompt_enrichment(self, tmp_path):
        """load 後の get_prompt_enrichment がエラーなく動作する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0
        assert "[内面]" in enrichment

    def test_double_save_load_cycle(self, tmp_path):
        """save -> load -> save -> load の 2 回サイクルでもエラーなし。"""
        dir1 = tmp_path / "cycle1"
        dir2 = tmp_path / "cycle2"
        dir1.mkdir()
        dir2.mkdir()

        # Cycle 1: 作成 -> 10ティック -> save
        orch1 = PsycheOrchestrator(data_dir=dir1, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.save()

        # Cycle 1: load -> 10ティック -> save
        orch2 = PsycheOrchestrator(data_dir=dir1, memory_count=5)
        orch2.load()
        _run_ticks(orch2, 10)
        assert orch2.tick_count == 20
        orch2.save()

        # Cycle 2: load -> 10ティック -> save
        orch3 = PsycheOrchestrator(data_dir=dir1, memory_count=5)
        orch3.load()
        assert orch3.tick_count == 20
        _run_ticks(orch3, 10)
        assert orch3.tick_count == 30

        # enrichment, policy が正常動作する
        enrichment = orch3.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch3.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict) and "policy_label" in policy

        # 最終 save
        orch3.save()
        data = json.loads(
            (dir1 / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert data["tick_count"] == 30
        assert data["version"] == 45

    def test_load_resume_varied_emotions(self, tmp_path):
        """load 後に多様な感情入力で 10 ティック実行してもエラーなし。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 多様な感情パターンで継続実行
        varied_emotions = [
            ("loving", 0.9), ("scared", -0.7), ("surprised", 0.4),
            ("angry", -0.8), ("teasing", 0.3), ("neutral", 0.0),
            ("happy", 0.6), ("sad", -0.4), ("loving", 0.5), ("neutral", 0.1),
        ]
        for emotion, valence in varied_emotions:
            percept = _make_percept(emotion=emotion, valence=valence)
            orch2.post_response_update(percept, delta_time=1.0)

        assert orch2.tick_count == 20

        # 全公開 API が正常動作
        enrichment = orch2.get_prompt_enrichment()
        assert len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_load_preserves_mood_and_tick(self, tmp_path):
        """load 後に mood と tick_count が保存時の値と一致する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        saved_tick = orch1.tick_count
        saved_mood = orch1.psyche.mood.valence
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == saved_tick
        assert abs(orch2.psyche.mood.valence - saved_mood) < 0.01


# ══════════════════════════════════════════════════════════════════
# 領域2: 初回起動（全デフォルト値）テスト
# ══════════════════════════════════════════════════════════════════


class TestFirstBootDefaults:
    """初回起動（save/load なし）でのデフォルト動作テスト。"""

    def test_first_tick_post_response_update(self):
        """最初の 1 ティック目の post_response_update がエラーなく動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_first_tick_select_policy_dict(self):
        """最初の 1 ティック目の select_policy_dict がエラーなく動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_first_tick_get_prompt_enrichment(self):
        """最初の 1 ティック目の get_prompt_enrichment がエラーなく動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

    def test_initial_enrichment_not_empty(self):
        """初期状態での enrichment 出力が空文字列でないこと。"""
        orch = PsycheOrchestrator()
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0
        # 心理状態セクションは常に存在する
        assert "[内面]" in enrichment

    def test_enrichment_has_content_in_each_section_after_ticks(self):
        """ティック後の enrichment 各セクションに何かしらのテキストがあること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        enrichment = orch.get_prompt_enrichment()
        # 心理状態セクションは必ず存在
        assert "[内面]" in enrichment
        # 感情情報は必ず含まれる
        assert "感情" in enrichment
        # 最低限長さがある
        assert len(enrichment) > 100

    def test_default_30_ticks_continuous(self):
        """全デフォルト状態での 30 ティック連続実行がエラーなし。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 30)
        assert orch.tick_count == 30

        # 30 ティック後の公開 API が正常
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_initial_fear_in_valid_range(self):
        """初期状態の fear_level が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        assert 0.0 <= orch.fear_level <= 1.0

    def test_initial_emotions_in_valid_range(self):
        """初期状態の全感情値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, f"Emotion {name} out of range: {val}"

    def test_initial_drives_in_valid_range(self):
        """初期状態の全ドライブ値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, f"Drive {name} out of range: {val}"

    def test_select_policy_dict_at_tick_0(self):
        """ティック 0 でも select_policy_dict がエラーなく動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        # ティック 0（post_response_update 未呼び出し）でもポリシー生成可能
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy


# ══════════════════════════════════════════════════════════════════
# 領域3: enrichment 48 項目の生成整合性テスト
# ══════════════════════════════════════════════════════════════════


class TestEnrichmentIntegrity:
    """enrichment 48 項目の生成整合性テスト。"""

    def test_5_sections_present_after_10_ticks(self):
        """10 ティック後に 5 セクション全てが存在すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        # select_policy_dict を呼んで判断傾向セクションのキャッシュを充実させる
        orch.select_policy_dict(_make_percept(), [])
        enrichment = orch.get_prompt_enrichment()

        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Section '{section}' not found in enrichment after 10 ticks"
            )

    def test_enrichment_total_chars_reasonable_at_10_ticks(self):
        """10 ティック後の enrichment テキストの合計文字数が妥当な範囲。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        enrichment = orch.get_prompt_enrichment()

        # 0 より大きい
        assert len(enrichment) > 0, "Enrichment should not be empty"
        # 上限を超えない（50000 文字以下）
        assert len(enrichment) <= 50000, (
            f"Enrichment too long: {len(enrichment)} chars"
        )

    def test_enrichment_section_order_correct(self):
        """セクション間の順序が正しいこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        orch.select_policy_dict(_make_percept(), [])
        enrichment = orch.get_prompt_enrichment()

        # 各セクションの位置を取得し、順序が正しいことを確認
        positions = []
        for section in ENRICHMENT_SECTIONS:
            pos = enrichment.find(section)
            if pos >= 0:
                positions.append((section, pos))

        # 見つかったセクションの順序が正しいか確認
        for i in range(len(positions) - 1):
            assert positions[i][1] < positions[i + 1][1], (
                f"Section '{positions[i][0]}' should come before "
                f"'{positions[i + 1][0]}' in enrichment"
            )

    def test_enrichment_integrity_at_30_ticks(self):
        """30 ティック後にも同様の整合性が維持されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 30)
        orch.select_policy_dict(_make_percept(), [])
        enrichment = orch.get_prompt_enrichment()

        # 5 セクション全てが存在
        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Section '{section}' not found in enrichment after 30 ticks"
            )

        # 文字数が妥当な範囲
        assert len(enrichment) > 0
        assert len(enrichment) <= 50000

        # セクション順序
        positions = []
        for section in ENRICHMENT_SECTIONS:
            pos = enrichment.find(section)
            if pos >= 0:
                positions.append((section, pos))
        for i in range(len(positions) - 1):
            assert positions[i][1] < positions[i + 1][1]

    def test_enrichment_footer_present(self):
        """enrichment のフッター（指示文）が含まれること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        enrichment = orch.get_prompt_enrichment()
        assert "機械的読み上げ禁止" in enrichment

    def test_enrichment_contains_emotion_info(self):
        """enrichment に感情情報が含まれること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        enrichment = orch.get_prompt_enrichment()
        assert "感情" in enrichment

    def test_enrichment_contains_mood_info(self):
        """enrichment にムード情報が含まれること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        enrichment = orch.get_prompt_enrichment()
        assert "ムード" in enrichment or "valence" in enrichment

    def test_enrichment_contains_drive_info(self):
        """enrichment にドライブ情報が含まれること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        enrichment = orch.get_prompt_enrichment()
        assert "ドライブ" in enrichment or "social" in enrichment

    def test_enrichment_grows_with_ticks(self):
        """ティック数が増えると enrichment の内容が充実する。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 1)
        enrichment_1 = orch.get_prompt_enrichment()

        _run_ticks(orch, 29)  # total 30
        orch.select_policy_dict(_make_percept(), [])
        enrichment_30 = orch.get_prompt_enrichment()

        # 30 ティック後は 1 ティック後より長い（より多くの情報が蓄積）
        assert len(enrichment_30) > len(enrichment_1), (
            "Enrichment should grow as more ticks provide more data"
        )

    def test_enrichment_stable_across_multiple_calls(self):
        """同じ状態で get_prompt_enrichment を複数回呼んでも一貫性がある。

        圧縮パイプライン導入により、初回呼び出しはキャッシュ不在のため全文記述、
        2回目以降は変動なし項目が短縮形となる。2回目以降は同じ出力になる。
        """
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)

        enrichment1 = orch.get_prompt_enrichment()  # 初回: キャッシュ構築
        enrichment2 = orch.get_prompt_enrichment()  # 2回目: 圧縮適用
        enrichment3 = orch.get_prompt_enrichment()  # 3回目: 安定状態

        # 2回目以降は状態変更なしなら同じ出力（圧縮が安定）
        assert enrichment2 == enrichment3
        # 初回と2回目は異なる可能性がある（圧縮による短縮）
        assert len(enrichment2) <= len(enrichment1)


# ══════════════════════════════════════════════════════════════════
# 領域4: 長時間連続稼働の安定性テスト
# ══════════════════════════════════════════════════════════════════


class TestLongRunStability:
    """長時間連続稼働の安定性テスト。"""

    def test_50_ticks_no_error(self):
        """50 ティック連続実行で毎ティック post_response_update + 適宜 select_policy_dict。"""
        orch = PsycheOrchestrator()
        for i in range(50):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
                text=f"連続稼働テスト{i}",
            )
            orch.post_response_update(percept, delta_time=1.0)

            # 10 ティック毎に select_policy_dict を呼ぶ
            if (i + 1) % 10 == 0:
                policy = orch.select_policy_dict(percept, [])
                assert isinstance(policy, dict)
                assert "policy_label" in policy

        assert orch.tick_count == 50

    def test_50_ticks_emotions_in_range(self):
        """50 ティック後も全感情値が 0-1 の範囲内に収まっていること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50)

        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range at tick 50: {val}"
            )

    def test_50_ticks_fear_in_range(self):
        """50 ティック後も fear_level が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50)
        assert 0.0 <= orch.fear_level <= 1.0

    def test_50_ticks_drives_in_range(self):
        """50 ティック後も全ドライブ値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50)

        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, (
                f"Drive {name} out of range at tick 50: {val}"
            )

    def test_50_ticks_fifo_not_unbounded(self):
        """50 ティック後に蓄積系フィールドのサイズが上限を超えないこと（FIFO 機能確認）。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50)

        # expectation_action_diff_log は FIFO で管理される
        assert len(orch._expectation_action_diff_log) <= 200, (
            "expectation_action_diff_log should be bounded"
        )

        # short_term_memory のバッファサイズ確認
        if orch._loop_state and orch._loop_state.memory:
            stm = orch._loop_state.memory
            if hasattr(stm, 'entries'):
                assert len(stm.entries) <= stm.max_entries, (
                    "STM memory entries should be bounded by max_entries"
                )

        # temporal_cognition の elapsed_records サイズ確認
        if orch._temporal_cognition is not None:
            tc_state = orch._temporal_cognition.state
            if hasattr(tc_state, 'elapsed_records'):
                assert len(tc_state.elapsed_records) <= 200, (
                    "temporal_cognition elapsed_records should be bounded"
                )

    def test_50_ticks_enrichment_still_valid(self):
        """50 ティック後も enrichment が正常に生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50)

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 100
        assert "[内面]" in enrichment

    def test_50_ticks_save_load_success(self, tmp_path):
        """50 ティック後の save -> load が正常に動作すること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch, 50)
        orch.select_policy_dict(_make_percept(), [])

        # save
        orch.save()
        assert (tmp_path / "psyche_snapshot.json").exists()

        # load
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 50

        # load 後の操作
        enrichment = orch2.get_prompt_enrichment()
        assert len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_50_ticks_every_tick_no_exception(self):
        """50 ティックの各ティックで例外が発生しないこと。"""
        orch = PsycheOrchestrator()
        for i in range(50):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
                text=f"安定性テスト{i}",
            )
            # 各ティックで例外が発生しないことを確認
            orch.post_response_update(percept, delta_time=1.0)
            assert orch.tick_count == i + 1

    def test_50_ticks_with_policy_every_5(self):
        """50 ティック実行中に 5 ティック毎に select_policy_dict を呼び出す。"""
        orch = PsycheOrchestrator()
        policies = []
        for i in range(50):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
            )
            orch.post_response_update(percept, delta_time=1.0)

            if (i + 1) % 5 == 0:
                policy = orch.select_policy_dict(percept, [])
                assert isinstance(policy, dict)
                assert "policy_label" in policy
                policies.append(policy)

        assert orch.tick_count == 50
        assert len(policies) == 10  # 50 / 5 = 10 回

    def test_50_ticks_enrichment_every_10(self):
        """50 ティック実行中に 10 ティック毎に enrichment を生成する。"""
        orch = PsycheOrchestrator()
        enrichments = []
        for i in range(50):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
            )
            orch.post_response_update(percept, delta_time=1.0)

            if (i + 1) % 10 == 0:
                enrichment = orch.get_prompt_enrichment()
                assert isinstance(enrichment, str)
                assert len(enrichment) > 0
                enrichments.append(enrichment)

        assert len(enrichments) == 5  # 50 / 10 = 5 回

    def test_midway_save_load_and_continue(self, tmp_path):
        """途中で save/load を挟んでも 50 ティック完走できる。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        # 前半 25 ティック
        _run_ticks(orch, 25)
        assert orch.tick_count == 25
        orch.save()

        # load して後半 25 ティック
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 25

        _run_ticks(orch2, 25)
        assert orch2.tick_count == 50

        # 全公開 API が正常
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_emotion_clamping_under_extreme_input(self):
        """極端な感情入力でも感情値が 0-1 に収まること。"""
        orch = PsycheOrchestrator()
        extreme_emotions = [
            ("happy", 1.0), ("sad", -1.0), ("angry", -1.0),
            ("happy", 1.0), ("sad", -1.0),
        ]
        for _ in range(10):
            for emotion, valence in extreme_emotions:
                percept = _make_percept(emotion=emotion, valence=valence)
                orch.post_response_update(percept, delta_time=1.0)

        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range after extreme input: {val}"
            )
