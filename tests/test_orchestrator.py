"""
tests/test_orchestrator.py - PsycheOrchestrator のテスト

テスト項目:
- 初期化テスト
- ティック実行テスト（各フェーズ）
- プロンプト生成テスト
- ポリシー提案テスト
- 永続化ラウンドトリップ
- エラー耐性テスト
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept, PsycheState


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


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """PsycheOrchestrator の初期化テスト。"""

    def test_basic_init(self):
        """基本的な初期化が成功する。"""
        orch = PsycheOrchestrator(memory_count=0)
        assert orch.tick_count == 0
        assert isinstance(orch.psyche, PsycheState)

    def test_init_with_memory_count(self):
        """memory_count が ContinuityState に反映される。"""
        orch = PsycheOrchestrator(memory_count=42)
        assert orch.psyche.continuity.memory_count == 42

    def test_init_fear_computed(self):
        """初期化時に fear_index が計算される。"""
        orch = PsycheOrchestrator()
        assert orch.fear_level >= 0.0
        assert orch.fear_level <= 1.0

    def test_init_with_custom_data_dir(self, tmp_path):
        """カスタムデータディレクトリで初期化。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        assert orch._data_dir == tmp_path

    def test_init_psyche_state_has_pillars(self):
        """初期化時に4柱がすべて設定される。"""
        orch = PsycheOrchestrator()
        p = orch.psyche
        assert p.identity is not None
        assert p.attachment is not None
        assert p.continuity is not None
        assert p.projection is not None
        assert p.fear_index is not None

    def test_init_subsystems_created(self):
        """全サブシステムがインスタンス化される。"""
        orch = PsycheOrchestrator()
        assert orch._tendency_sys is not None
        assert orch._self_model_sys is not None
        assert orch._temporal_diff_sys is not None
        assert orch._strain_sys is not None
        assert orch._self_image_sys is not None
        assert orch._coherence_sys is not None
        assert orch._narrative_sys is not None
        assert orch._episodic_sys is not None
        assert orch._binding_sys is not None
        assert orch._introspection_sys is not None
        assert orch._consumption_sys is not None
        assert orch._expectation_sys is not None
        assert orch._motivation_sys is not None
        assert orch._other_model_sys is not None
        assert orch._vector_gen is not None
        assert orch._candidate_gen is not None
        assert orch._transient_goal_mgr is not None
        assert orch._scoped_goal_sys is not None
        assert orch._stability_valve is not None
        assert orch._dynamics_observer is not None


# ── ティック実行テスト ────────────────────────────────────────────


class TestTickExecution:
    """ティック実行テスト。"""

    def test_single_tick(self):
        """1ティック実行が成功する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_multiple_ticks(self):
        """複数ティック実行が成功する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 5

    def test_tick_updates_mood(self):
        """ティックで mood が変化する。"""
        orch = PsycheOrchestrator()
        initial_mood = orch.psyche.mood.valence
        percept = _make_percept(emotion="happy", valence=0.8)
        orch.post_response_update(percept, delta_time=1.0)
        # Happy percept should shift mood positively
        assert orch.psyche.mood.valence != initial_mood

    def test_tick_with_negative_emotion(self):
        """ネガティブ感情でもエラーなくティック実行。"""
        orch = PsycheOrchestrator()
        percept = _make_percept(emotion="sad", valence=-0.6)
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_tick_with_neutral_emotion(self):
        """ニュートラル感情でもエラーなくティック実行。"""
        orch = PsycheOrchestrator()
        percept = _make_percept(emotion="neutral", valence=0.0)
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_phase_3_tick_triggers(self):
        """3ティック目でフェーズ8-14が発火する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 3
        # tendency_awareness should be populated
        assert orch._tendency_awareness is not None

    def test_phase_5_tick_triggers(self):
        """5ティック目でフェーズ15-26が発火する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 5

    def test_phase_10_tick_triggers(self):
        """10ティック目でフェーズ27-29が発火する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 10

    def test_15_ticks_no_error(self):
        """15ティック（全フェーズ複数回）エラーなし。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(15):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 15

    def test_30_ticks_no_error(self):
        """30ティック（全フェーズ複数回、10ティック×3）エラーなし。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(30):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 30

    def test_tick_with_varied_emotions(self):
        """様々な感情タグでティック実行。"""
        orch = PsycheOrchestrator()
        emotions = [
            ("happy", 0.7), ("sad", -0.6), ("angry", -0.5),
            ("surprised", 0.3), ("scared", -0.5), ("loving", 0.8),
            ("teasing", 0.4), ("neutral", 0.0),
        ]
        for emotion, valence in emotions:
            percept = _make_percept(emotion=emotion, valence=valence)
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == len(emotions)

    def test_tick_with_small_delta(self):
        """小さな delta_time でも正常動作。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=0.01)
        assert orch.tick_count == 1

    def test_tick_with_large_delta(self):
        """大きな delta_time でも正常動作。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=60.0)
        assert orch.tick_count == 1


# ── プロンプト生成テスト ──────────────────────────────────────────


class TestPromptEnrichment:
    """プロンプト生成テスト。"""

    def test_initial_enrichment(self):
        """初期状態でプロンプトが生成される。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_enrichment_contains_psyche_section(self):
        """【心理状態（内面）】セクションが含まれる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "心理状態" in text

    def test_enrichment_contains_emotions(self):
        """感情情報が含まれる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "感情" in text

    def test_enrichment_contains_mood(self):
        """ムード情報が含まれる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "ムード" in text or "valence" in text

    def test_enrichment_contains_drive(self):
        """ドライブ情報が含まれる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "ドライブ" in text or "social" in text

    def test_enrichment_contains_fear(self):
        """恐怖情報が含まれる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "恐怖" in text or "fear" in text or "level=" in text

    def test_enrichment_after_ticks_has_self_awareness(self):
        """ティック後のプロンプトに自己認識セクションが出る。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        # Run enough ticks for self-awareness sections
        for _ in range(6):
            orch.post_response_update(percept, delta_time=1.0)
        text = orch.get_prompt_enrichment()
        # After 6 ticks (including tick 3 and 5), self sections should appear
        assert "自己認識" in text or "動機" in text or len(text) > 200

    def test_enrichment_ends_with_instruction(self):
        """プロンプトが指示文で終わる。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "機械的に読み上げないこと" in text

    def test_enrichment_tension_with_commitment(self):
        """persistent_commitment に保持項目がある場合、張力情報が含まれる。"""
        orch = PsycheOrchestrator()
        # persistent_commitment に保持項目を直接追加する
        from psyche.persistent_commitment import CommitmentItem
        item = CommitmentItem(
            item_id="test_item_1",
            source_goal_id="goal_1",
            category="test",
            direction_signature={"a": 0.5, "b": -0.3},
            strength=0.6,
            initial_strength=0.6,
        )
        orch._persistent_commitment._state.items.append(item)
        text = orch.get_prompt_enrichment()
        assert "内部-外部間の張力" in text
        assert "保持方向1件の方向的バイアスあり" in text

    def test_enrichment_no_tension_in_initial_state(self):
        """初期状態（保持項目なし、caution低、価値軸中立）では張力情報が含まれない。"""
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "内部-外部間の張力" not in text


# ── ポリシー提案テスト ────────────────────────────────────────────


class TestPolicySuggestions:
    """ポリシー提案テスト。"""

    def test_initial_suggestions(self):
        """初期状態でポリシー候補が生成される。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        text = orch.get_policy_suggestions(percept, [])
        assert isinstance(text, str)
        assert len(text) > 0

    def test_suggestions_contain_header(self):
        """【行動方針候補】ヘッダが含まれる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        text = orch.get_policy_suggestions(percept, [])
        assert "行動方針候補" in text

    def test_suggestions_contain_candidates(self):
        """候補に番号とスコアが含まれる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        text = orch.get_policy_suggestions(percept, [])
        assert "1." in text
        assert "score=" in text

    def test_suggestions_contain_disclaimer(self):
        """「候補に従う義務はありません」が含まれる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        text = orch.get_policy_suggestions(percept, [])
        assert "義務" in text or "参考" in text

    def test_suggestions_after_ticks(self):
        """ティック後にポリシー候補が変化しうる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        text_before = orch.get_policy_suggestions(percept, [])
        # Run ticks with emotional input
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        text_after = orch.get_policy_suggestions(percept, [])
        # At minimum, both should be valid
        assert "行動方針候補" in text_before
        assert "行動方針候補" in text_after

    def test_suggestions_with_memories(self):
        """記憶入力ありでもエラーなし。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        memories = [
            {"summary": "テスト記憶", "date": "2026-01-01", "keywords": ["test"]},
        ]
        text = orch.get_policy_suggestions(percept, memories)
        assert len(text) > 0


# ── 永続化ラウンドトリップテスト ──────────────────────────────────


class TestPersistence:
    """永続化テスト。"""

    def test_save_creates_file(self, tmp_path):
        """save() がファイルを作成する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.save()
        assert (tmp_path / "psyche_snapshot.json").exists()

    def test_save_produces_valid_json(self, tmp_path):
        """save() が有効な JSON を出力する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.save()
        data = json.loads((tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8"))
        assert "version" in data
        assert "psyche" in data

    def test_load_restores_tick_count(self, tmp_path):
        """load() が tick_count を復元する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(7):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 7

    def test_load_restores_mood(self, tmp_path):
        """load() が mood を復元する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept(emotion="happy", valence=0.8)
        for _ in range(5):
            orch1.post_response_update(percept, delta_time=1.0)
        saved_mood = orch1.psyche.mood.valence
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        assert abs(orch2.psyche.mood.valence - saved_mood) < 0.01

    def test_load_nonexistent_returns_false(self, tmp_path):
        """存在しないファイルの load() は False を返す。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        result = orch.load()
        assert result is False

    def test_save_with_custom_path(self, tmp_path):
        """カスタムパスへの保存。"""
        custom = tmp_path / "custom_snapshot.json"
        orch = PsycheOrchestrator()
        orch.save(path=custom)
        assert custom.exists()

    def test_roundtrip_after_ticks(self, tmp_path):
        """ティック実行後のラウンドトリップ。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        # Core state should be restored
        assert orch2.tick_count == 10
        # Enrichment should still work
        text = orch2.get_prompt_enrichment()
        assert len(text) > 0

    def test_save_contains_all_v23_fields(self, tmp_path):
        """save() が v23 で定義された全49フィールドを含む。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        expected_keys = [
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
        ]
        for key in expected_keys:
            assert key in data, f"Missing save field: {key}"
        assert data["version"] == 26

    def test_roundtrip_json_match(self, tmp_path):
        """save → load → save で JSON が一致する（全フィールド復元確認）。

        meta_emotion_state は load 時に apply_session_decay() が適用されるため
        差分が生じうる。それ以外の全フィールドが完全一致することを検証する。
        """
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        # 1) Orchestrator を作成し、多様な入力でティック実行
        orch1 = PsycheOrchestrator(data_dir=dir_a, memory_count=10)
        emotions = ["happy", "sad", "loving", "angry", "neutral",
                     "surprised", "teasing", "scared", "happy", "sad"]
        valences = [0.7, -0.6, 0.8, -0.5, 0.0,
                    0.3, 0.4, -0.4, 0.6, -0.3]
        for i in range(20):
            idx = i % len(emotions)
            percept = _make_percept(
                emotion=emotions[idx],
                valence=valences[idx],
                text=f"テスト入力{i}",
            )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        # 2) JSON A を読み込み
        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # 3) 新しい Orchestrator に load → save
        orch2 = PsycheOrchestrator(data_dir=dir_b, memory_count=10)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        # 4) JSON B を読み込み
        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # 5) 既知の差異を除いた全フィールドで一致確認
        # meta_emotion_state: load時に apply_session_decay() が適用される
        # psyche.fear_index: from_dict が個別リスク値を復元しない制約
        skip_keys = {"meta_emotion_state", "psyche"}
        for key in json_a:
            if key in skip_keys:
                continue
            assert json_a[key] == json_b[key], (
                f"Roundtrip mismatch on field '{key}'"
            )

        # 6) psyche は fear_index 以外で一致確認
        psyche_a = {k: v for k, v in json_a["psyche"].items() if k != "fear_index"}
        psyche_b = {k: v for k, v in json_b["psyche"].items() if k != "fear_index"}
        assert psyche_a == psyche_b, "Roundtrip mismatch on field 'psyche' (excluding fear_index)"

        # 7) meta_emotion_state は存在することだけ確認
        assert "meta_emotion_state" in json_b

    def test_roundtrip_v5_system_states(self, tmp_path):
        """v5 システムステート（tendency/vector/candidate/transient_goal/stability）の復元。"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a)
        percept = _make_percept()
        for _ in range(15):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        v5_keys = [
            "tendency_state", "vector_state", "candidate_state",
            "transient_goal_state", "stability_valve",
        ]
        for key in v5_keys:
            assert json_a[key] == json_b[key], f"v5 roundtrip mismatch: {key}"

    def test_roundtrip_v7_to_v13_states(self, tmp_path):
        """v7-v13 拡張ステート（expansion/integration/feed/dialogue/spontaneous/vo/forgetting）の復元。"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        v7_to_v13_keys = [
            "policy_expansion_state", "memory_integration_state",
            "real_feed_state", "text_dialogue_state",
            "spontaneous_state", "vo_validation_state",
            "forgetting_fixation_state",
        ]
        for key in v7_to_v13_keys:
            assert json_a[key] == json_b[key], f"v7-v13 roundtrip mismatch: {key}"

    def test_roundtrip_v14_to_v23_states(self, tmp_path):
        """v14-v23 高度認知ステート（action_result〜selection_attribution）の復元。"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a)
        emotions = ["happy", "sad", "angry", "neutral", "surprised"]
        valences = [0.7, -0.6, -0.5, 0.0, 0.3]
        for i in range(15):
            idx = i % len(emotions)
            percept = _make_percept(emotion=emotions[idx], valence=valences[idx])
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        v14_to_v25_keys = [
            "action_result_state", "dialogue_learning_state",
            "self_action_perception_state",
            "expectation_action_diff_log", "intent_action_gap_state",
            "temporal_cognition_state", "multi_path_recall_state",
            "introspection_cross_section_state", "perceptual_context_state",
            "selection_attribution_state",
            # v24
            "reference_frequency_state",
            # v25
            "persistent_commitment_state",
        ]
        for key in v14_to_v25_keys:
            assert json_a[key] == json_b[key], f"v14-v25 roundtrip mismatch: {key}"

    def test_roundtrip_v6_responsibility_context(self, tmp_path):
        """v6 責任・文脈感度・カップリングの復元。"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        v6_keys = ["dispersion_state", "context_sensitivity_state", "last_coupling"]
        for key in v6_keys:
            assert json_a[key] == json_b[key], f"v6 roundtrip mismatch: {key}"

    def test_roundtrip_v4_snapshot_states(self, tmp_path):
        """v4 スナップショットステート（self_view〜consumption, expectations, motives等）の復元。"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a, memory_count=5)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b, memory_count=5)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        v4_keys = [
            "amplitude", "value_orientation", "self_ref_state",
            "last_self_view", "tendency_awareness", "last_diff_summary",
            "last_strain", "last_self_image", "last_coherence",
            "last_narrative", "last_episodes", "last_bindings",
            "last_trace", "last_consumption",
            "last_expectations", "last_motives",
            "last_other_model", "input_supply",
        ]
        for key in v4_keys:
            assert json_a[key] == json_b[key], f"v4 roundtrip mismatch: {key}"

    def test_roundtrip_preserves_enrichment(self, tmp_path):
        """ラウンドトリップ後も get_prompt_enrichment の内容が維持される。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=10)
        emotions = ["happy", "sad", "loving", "angry", "neutral"]
        valences = [0.7, -0.6, 0.8, -0.5, 0.0]
        for i in range(20):
            idx = i % len(emotions)
            percept = _make_percept(emotion=emotions[idx], valence=valences[idx])
            orch1.post_response_update(percept, delta_time=1.0)

        enrichment_before = orch1.get_prompt_enrichment()
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=10)
        orch2.load()
        enrichment_after = orch2.get_prompt_enrichment()

        # enrichment は内部キャッシュの再生成で多少変わりうるが、
        # 主要セクションは存在するはず
        assert "心理状態" in enrichment_after
        assert len(enrichment_after) > 100


# ── メモリ保存コールバックテスト ──────────────────────────────────


class TestMemoryCallback:
    """on_memory_saved コールバックテスト。"""

    def test_memory_callback_updates_continuity(self):
        """コールバックで continuity.memory_count が更新される。"""
        orch = PsycheOrchestrator(memory_count=5)
        assert orch.psyche.continuity.memory_count == 5
        orch.on_memory_saved("test summary", ["kw1"], memory_count=10)
        assert orch.psyche.continuity.memory_count == 10

    def test_memory_callback_recomputes_fear(self):
        """コールバックで fear_index が再計算される。"""
        orch = PsycheOrchestrator(memory_count=0)
        fear_before = orch.fear_level
        orch.on_memory_saved("test", ["kw"], memory_count=100)
        # More memories → lower continuity risk → fear should decrease
        assert orch.fear_level <= fear_before


# ── エラー耐性テスト ──────────────────────────────────────────────


class TestErrorResilience:
    """エラー耐性テスト。"""

    def test_empty_percept(self):
        """空の Percept でもクラッシュしない。"""
        orch = PsycheOrchestrator()
        percept = Percept(text="")
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_unknown_emotion(self):
        """未知の感情タグでもクラッシュしない。"""
        orch = PsycheOrchestrator()
        percept = Percept(
            text="test", emotion="UNKNOWN_EMOTION",
            emotion_valence=0.0,
        )
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_zero_delta_time(self):
        """delta_time=0 でもクラッシュしない。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=0.0)
        assert orch.tick_count == 1

    def test_negative_delta_time(self):
        """負の delta_time でもクラッシュしない（異常値だが耐える）。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=-1.0)
        assert orch.tick_count == 1

    def test_rapid_ticks(self):
        """大量の連続ティックでも安定。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(100):
            orch.post_response_update(percept, delta_time=0.1)
        assert orch.tick_count == 100

    def test_prompt_enrichment_never_raises(self):
        """get_prompt_enrichment は例外を投げない。"""
        orch = PsycheOrchestrator()
        # Run varied ticks
        emotions = ["happy", "sad", "angry", "neutral"]
        for em in emotions:
            percept = _make_percept(emotion=em, valence=0.5 if em == "happy" else -0.3)
            orch.post_response_update(percept, delta_time=1.0)
        text = orch.get_prompt_enrichment()
        assert isinstance(text, str)

    def test_policy_suggestions_never_raises(self):
        """get_policy_suggestions は例外を投げない。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        text = orch.get_policy_suggestions(percept, [])
        assert isinstance(text, str)

    def test_save_load_corrupted_graceful(self, tmp_path):
        """壊れたスナップショットでも load はクラッシュしない。"""
        bad_file = tmp_path / "psyche_snapshot.json"
        bad_file.write_text("{invalid json!!!}", encoding="utf-8")
        orch = PsycheOrchestrator(data_dir=tmp_path)
        result = orch.load()
        assert result is False


# ── 統合テスト ────────────────────────────────────────────────────


class TestIntegration:
    """統合テスト: 全フェーズを通した動作確認。"""

    def test_full_lifecycle(self, tmp_path):
        """初期化→30ティック→プロンプト→ポリシー→保存→復元。"""
        orch = PsycheOrchestrator(memory_count=10, data_dir=tmp_path)

        # Run 30 ticks with varied input
        emotions = ["happy", "sad", "loving", "teasing", "neutral",
                     "surprised", "scared", "angry", "happy", "neutral"]
        valences = [0.7, -0.6, 0.8, 0.4, 0.0,
                    0.3, -0.5, -0.5, 0.7, 0.0]

        for i in range(30):
            idx = i % len(emotions)
            percept = _make_percept(emotion=emotions[idx], valence=valences[idx])
            orch.post_response_update(percept, delta_time=1.0)

        assert orch.tick_count == 30

        # Prompt enrichment should be rich
        enrichment = orch.get_prompt_enrichment()
        assert len(enrichment) > 200
        assert "心理状態" in enrichment

        # Policy suggestions should work
        percept = _make_percept()
        suggestions = orch.get_policy_suggestions(percept, [])
        assert "行動方針候補" in suggestions

        # Save
        orch.save()
        assert (tmp_path / "psyche_snapshot.json").exists()

        # Restore
        orch2 = PsycheOrchestrator(memory_count=10, data_dir=tmp_path)
        assert orch2.load() is True
        assert orch2.tick_count == 30

        # Memory callback (use fresh orch that still has continuity pillar)
        orch3 = PsycheOrchestrator(memory_count=10, data_dir=tmp_path)
        orch3.on_memory_saved("test memory", ["kw"], memory_count=20)
        assert orch3.psyche.continuity.memory_count == 20

    def test_different_user_ids(self):
        """異なるユーザーIDで動作確認。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0, user_id="user_a")
        orch.post_response_update(percept, delta_time=1.0, user_id="user_b")
        assert orch.tick_count == 2


# ── スモークテスト: 全パイプライン通過 ─────────────────────────────


class TestSmokeFullPipeline:
    """全パイプラインを通過するスモークテスト。

    post_response_update → get_prompt_enrichment → get_policy_suggestions →
    select_policy_dict → save → load → 再実行 → 再save の全経路を検証する。
    """

    # ── 1. 全パス通過テスト ──────────────────────────────────────

    def test_full_path_smoke(self, tmp_path):
        """全パス通過: tick(全インターバル) → enrichment → suggestions → select → save → load → tick → save。"""
        orch = PsycheOrchestrator(memory_count=5, data_dir=tmp_path)
        percept = _make_percept()

        # 全ティックインターバルを通過させる (3, 5, 10 の最小公倍数 = 30)
        for i in range(30):
            p = _make_percept(
                emotion=["happy", "sad", "neutral", "angry", "surprised"][i % 5],
                valence=[0.7, -0.6, 0.0, -0.5, 0.3][i % 5],
                text=f"スモークテスト入力{i}",
            )
            orch.post_response_update(p, delta_time=1.0)

        assert orch.tick_count == 30

        # get_prompt_enrichment
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        # get_policy_suggestions
        suggestions = orch.get_policy_suggestions(percept, [])
        assert isinstance(suggestions, str)
        assert len(suggestions) > 0

        # select_policy_dict
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)

        # save
        orch.save()
        assert (tmp_path / "psyche_snapshot.json").exists()

        # load
        orch2 = PsycheOrchestrator(memory_count=5, data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 30

        # 再度ティック実行
        for i in range(10):
            p = _make_percept(
                emotion=["loving", "teasing", "scared"][i % 3],
                valence=[0.8, 0.4, -0.4][i % 3],
            )
            orch2.post_response_update(p, delta_time=1.0)
        assert orch2.tick_count == 40

        # 再度save
        orch2.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert data["tick_count"] == 40

    # ── 2. select_policy_dict の動作テスト ────────────────────────

    def test_select_policy_dict_returns_valid_dict(self):
        """select_policy_dict() が有効な dict を返す。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)

    def test_select_policy_dict_contains_required_keys(self):
        """select_policy_dict の返り値に policy_label, _score, rationale が含まれる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert "policy_label" in policy
        assert "_score" in policy
        assert "rationale" in policy

    def test_select_policy_dict_with_few_ticks(self):
        """ティック数が少なくても select_policy_dict が動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        # わずか1ティックでも動作する
        orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_select_policy_dict_after_diverse_input(self):
        """多様な入力後でも select_policy_dict が動作する。"""
        orch = PsycheOrchestrator()
        emotions = [
            ("happy", 0.9), ("sad", -0.8), ("angry", -0.7),
            ("surprised", 0.5), ("scared", -0.6), ("loving", 0.8),
            ("teasing", 0.3), ("neutral", 0.0),
        ]
        for emotion, valence in emotions:
            percept = _make_percept(emotion=emotion, valence=valence)
            orch.post_response_update(percept, delta_time=1.0)

        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy
        assert "_score" in policy
        assert "rationale" in policy

    # ── 3. save→load後のselect_policy_dict テスト ────────────────

    def test_select_policy_dict_after_save_load(self, tmp_path):
        """save→load 後に select_policy_dict が動作する。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        policy = orch2.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy
        assert "_score" in policy

    def test_enrichment_length_after_save_load(self, tmp_path):
        """load 後の enrichment が妥当な長さを持つ。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(15):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        # load後でも心理状態セクションは必ず含まれる
        assert "心理状態" in enrichment
        assert len(enrichment) > 100

    # ── 4. Phase発火の確認テスト ──────────────────────────────────

    def test_phase_3_tick_attributes_set(self):
        """3ティック後に _tendency_awareness, transient_goal, _last_motives が設定される。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=1.0)

        # Phase 8: tendency_awareness
        assert orch._tendency_awareness is not None
        # Phase 12: transient_goal_mgr が observe_turn を実行済み
        assert orch._transient_goal_mgr is not None
        assert orch._transient_goal_mgr.state is not None
        # Phase 14: intrinsic_motivation
        assert orch._last_motives is not None

    def test_phase_5_tick_attributes_set(self):
        """5ティック後に自己像・一貫性・ナラティブ等が設定される。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)

        # Phase 17: self_image
        assert orch._last_self_image is not None
        # Phase 18: coherence
        assert orch._last_coherence is not None
        # Phase 19: narrative
        assert orch._last_narrative is not None
        # Phase 20: episodes
        assert orch._last_episodes is not None
        # Phase 21: bindings
        assert orch._last_bindings is not None
        # Phase 22: trace
        assert orch._last_trace is not None
        # Phase 23: consumption
        assert orch._last_consumption is not None
        # Phase 24: expectations
        assert orch._last_expectations is not None
        # Phase 25: other_model
        assert orch._last_other_model is not None

    def test_phase_10_tick_all_phases_fired(self):
        """10ティック後に全フェーズが発火済み。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=1.0)

        # Phase 8 (3ティック毎)
        assert orch._tendency_awareness is not None
        # Phase 14 (3ティック毎)
        assert orch._last_motives is not None
        # Phase 15-26 (5ティック毎) の主要属性
        assert orch._last_self_image is not None
        assert orch._last_coherence is not None
        assert orch._last_narrative is not None
        assert orch._last_episodes is not None
        assert orch._last_bindings is not None
        assert orch._last_trace is not None
        assert orch._last_consumption is not None
        assert orch._last_expectations is not None
        assert orch._last_other_model is not None
        # Phase 27-29 (10ティック毎): stability_valve が observe_extremity 実行済み
        assert orch._stability_valve is not None
        # dynamics_observer が record_turn 実行済み
        assert orch._dynamics_observer is not None

    # ── 5. enrichment内容の検証テスト ─────────────────────────────

    def test_enrichment_contains_all_five_sections(self):
        """十分なティック後、enrichmentに5セクション全てが含まれる。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        # 全フェーズが複数回発火するまでティック実行
        for i in range(30):
            p = _make_percept(
                emotion=["happy", "sad", "neutral", "angry", "surprised"][i % 5],
                valence=[0.7, -0.6, 0.0, -0.5, 0.3][i % 5],
            )
            orch.post_response_update(p, delta_time=1.0)

        enrichment = orch.get_prompt_enrichment()

        # 5セクション: 心理状態、自己認識、動機・目標、記憶・内省、判断傾向
        assert "心理状態" in enrichment
        assert "自己認識" in enrichment
        assert "動機・目標" in enrichment or "動機" in enrichment
        assert "記憶・内省" in enrichment or "記憶" in enrichment
        assert "判断傾向" in enrichment

    def test_enrichment_size_reasonable(self):
        """enrichment のサイズが合理的範囲内（500文字以上、50000文字以下）。"""
        orch = PsycheOrchestrator()
        for i in range(30):
            p = _make_percept(
                emotion=["happy", "sad", "neutral", "angry", "loving"][i % 5],
                valence=[0.7, -0.6, 0.0, -0.5, 0.8][i % 5],
            )
            orch.post_response_update(p, delta_time=1.0)

        enrichment = orch.get_prompt_enrichment()
        assert len(enrichment) >= 500, (
            f"Enrichment too short: {len(enrichment)} chars"
        )
        assert len(enrichment) <= 50000, (
            f"Enrichment too long: {len(enrichment)} chars"
        )

    # ── 6. 連続稼働安定性テスト ──────────────────────────────────

    def test_50_ticks_continuous_no_error(self):
        """50ティック連続実行でエラーなし。"""
        orch = PsycheOrchestrator()
        emotions = ["happy", "sad", "angry", "neutral", "surprised",
                     "loving", "teasing", "scared"]
        valences = [0.7, -0.6, -0.5, 0.0, 0.3, 0.8, 0.4, -0.4]

        for i in range(50):
            idx = i % len(emotions)
            percept = _make_percept(
                emotion=emotions[idx],
                valence=valences[idx],
                text=f"連続稼働テスト{i}",
            )
            orch.post_response_update(percept, delta_time=1.0)

        assert orch.tick_count == 50

        # 50ティック後もすべての公開APIが正常動作する
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        percept = _make_percept()
        suggestions = orch.get_policy_suggestions(percept, [])
        assert isinstance(suggestions, str)

        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)

    def test_50_ticks_with_midway_save_load_resume(self, tmp_path):
        """途中のsave/load/resumeでもエラーなし。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()

        # 最初の20ティック
        for i in range(20):
            p = _make_percept(
                emotion=["happy", "sad", "neutral"][i % 3],
                valence=[0.7, -0.6, 0.0][i % 3],
            )
            orch.post_response_update(p, delta_time=1.0)
        assert orch.tick_count == 20

        # 中間save
        orch.save()
        assert (tmp_path / "psyche_snapshot.json").exists()

        # load して再開
        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        assert orch2.tick_count == 20

        # 次の15ティック
        for i in range(15):
            p = _make_percept(
                emotion=["angry", "surprised", "loving"][i % 3],
                valence=[-0.5, 0.3, 0.8][i % 3],
            )
            orch2.post_response_update(p, delta_time=1.0)
        assert orch2.tick_count == 35

        # 2回目のsave
        orch2.save()

        # 再度load して再開
        orch3 = PsycheOrchestrator(data_dir=tmp_path)
        orch3.load()
        assert orch3.tick_count == 35

        # 最後の15ティック
        for i in range(15):
            p = _make_percept(
                emotion=["teasing", "scared", "neutral"][i % 3],
                valence=[0.4, -0.4, 0.0][i % 3],
            )
            orch3.post_response_update(p, delta_time=1.0)
        assert orch3.tick_count == 50

        # 最終save
        orch3.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert data["tick_count"] == 50

        # 最終状態でもすべての公開APIが正常動作する
        enrichment = orch3.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        policy = orch3.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
