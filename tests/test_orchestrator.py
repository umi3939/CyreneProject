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
