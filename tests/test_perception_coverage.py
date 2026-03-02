"""
tests/test_perception_coverage.py - PerceptionCoverageMeasurement のテスト

設計書: design_perception_coverage.md

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定、環境変数制御)
- 第1段: 知覚結果の分類記録テスト
- 第2段: セッション内の累積集計テスト
- 第3段: サマリの読み取り専用提供テスト
- FIFOバッファテスト(上限、自然消失)
- 安全弁テスト(計測失敗時の無視、無効時の完全スキップ)
- 永続化非対象テスト
- psyche非参照テスト
- 辞書非変更テスト
- ログ出力テスト
- 入力テキスト非保持テスト
- PerceptionCoverageRecordテスト
"""

import json
import logging
import os
import time
from unittest.mock import patch

import pytest

from tools.pipeline_measurement import (
    PerceptionCoverageMeasurement,
    PerceptionCoverageRecord,
    _DEFAULT_COVERAGE_BUFFER_MAX,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_coverage(
    enabled: bool = True,
    buffer_max: int = 200,
) -> PerceptionCoverageMeasurement:
    """テスト用のPerceptionCoverageMeasurementを生成する。"""
    return PerceptionCoverageMeasurement(enabled=enabled, buffer_max=buffer_max)


# ── PerceptionCoverageRecord テスト ───────────────────────────────


class TestPerceptionCoverageRecord:
    """PerceptionCoverageRecordの単体テスト。"""

    def test_init(self):
        """初期状態。"""
        r = PerceptionCoverageRecord(
            emotion_is_neutral=True,
            intent_is_unknown=False,
            keyword_hit=True,
            llm_used=False,
        )
        assert r.emotion_is_neutral is True
        assert r.intent_is_unknown is False
        assert r.keyword_hit is True
        assert r.llm_used is False
        assert r.timestamp > 0

    def test_to_dict(self):
        """辞書表現の構造。"""
        r = PerceptionCoverageRecord(
            emotion_is_neutral=False,
            intent_is_unknown=True,
            keyword_hit=False,
            llm_used=True,
        )
        d = r.to_dict()
        assert d["emotion_is_neutral"] is False
        assert d["intent_is_unknown"] is True
        assert d["keyword_hit"] is False
        assert d["llm_used"] is True
        assert "timestamp" in d

    def test_slots(self):
        """__slots__により任意の属性追加が禁止される。"""
        r = PerceptionCoverageRecord(
            emotion_is_neutral=True,
            intent_is_unknown=True,
            keyword_hit=False,
            llm_used=False,
        )
        with pytest.raises(AttributeError):
            r.extra_field = "should fail"

    def test_no_text_stored(self):
        """入力テキストが保持されない(安全弁7)。"""
        r = PerceptionCoverageRecord(
            emotion_is_neutral=True,
            intent_is_unknown=True,
            keyword_hit=False,
            llm_used=False,
        )
        assert not hasattr(r, "text")
        assert not hasattr(r, "input_text")
        assert not hasattr(r, "user_text")
        d = r.to_dict()
        assert "text" not in d
        assert "input_text" not in d
        assert "user_text" not in d


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """PerceptionCoverageMeasurementの初期化テスト。"""

    def test_default_init_disabled(self):
        """デフォルトでは計測は無効(CYRENE_MONITOR未設定)。"""
        with patch.dict(os.environ, {}, clear=False):
            if "CYRENE_MONITOR" in os.environ:
                del os.environ["CYRENE_MONITOR"]
            m = PerceptionCoverageMeasurement()
            assert m.enabled is False

    def test_enabled_via_constructor(self):
        """コンストラクタで明示的に有効化。"""
        m = _make_coverage(enabled=True)
        assert m.enabled is True

    def test_disabled_via_constructor(self):
        """コンストラクタで明示的に無効化。"""
        m = _make_coverage(enabled=False)
        assert m.enabled is False

    def test_enabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=1で有効化(安全弁6)。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PerceptionCoverageMeasurement()
            assert m.enabled is True

    def test_disabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=0で無効化(安全弁6)。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            m = PerceptionCoverageMeasurement()
            assert m.enabled is False

    def test_constructor_overrides_env(self):
        """コンストラクタの指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PerceptionCoverageMeasurement(enabled=False)
            assert m.enabled is False

    def test_initial_counters_zero(self):
        """初期状態で全カウンタがゼロ。"""
        m = _make_coverage()
        assert m.total_count == 0
        assert m.neutral_count == 0
        assert m.unknown_count == 0
        assert m.keyword_hit_count == 0
        assert m.llm_used_count == 0
        assert m.buffer_size == 0

    def test_buffer_max_applied(self):
        """FIFOバッファの上限が適用される(安全弁2)。"""
        m = _make_coverage(buffer_max=5)
        for _ in range(10):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.buffer_size == 5

    def test_buffer_max_minimum(self):
        """FIFOバッファの上限は最低1。"""
        m = _make_coverage(buffer_max=0)
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.buffer_size == 1

    def test_default_buffer_max(self):
        """デフォルトのバッファ上限。"""
        assert _DEFAULT_COVERAGE_BUFFER_MAX == 200


# ── 第1段: 知覚結果の分類記録テスト ─────────────────────────────


class TestClassificationRecording:
    """知覚結果の分類記録。"""

    def test_neutral_emotion_detected(self):
        """感情ラベルneutralが正しく分類される。"""
        m = _make_coverage()
        m.record_perception("neutral", "greeting", [], False)
        assert m.neutral_count == 1
        assert m.total_count == 1

    def test_non_neutral_emotion_detected(self):
        """感情ラベルneutral以外が正しく分類される。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.neutral_count == 0
        assert m.total_count == 1

    def test_unknown_intent_detected(self):
        """意図ラベルunknownが正しく分類される。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        assert m.unknown_count == 1

    def test_non_unknown_intent_detected(self):
        """意図ラベルunknown以外が正しく分類される。"""
        m = _make_coverage()
        m.record_perception("neutral", "question", [], False)
        assert m.unknown_count == 0

    def test_keyword_hit_with_topics(self):
        """話題リストが空でない場合、キーワード一致ありと分類される。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい", "楽しい"], False)
        assert m.keyword_hit_count == 1

    def test_no_keyword_hit_with_empty_topics(self):
        """話題リストが空の場合、キーワード一致なしと分類される。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        assert m.keyword_hit_count == 0

    def test_llm_used_detected(self):
        """LLM利用が正しく記録される。"""
        m = _make_coverage()
        m.record_perception("happy", "sharing", ["嬉しい"], True)
        assert m.llm_used_count == 1

    def test_llm_not_used_detected(self):
        """LLM非利用が正しく記録される。"""
        m = _make_coverage()
        m.record_perception("happy", "sharing", ["嬉しい"], False)
        assert m.llm_used_count == 0

    def test_all_classified_perception(self):
        """全てが分類された知覚結果(neutral=False, unknown=False, hit=True)。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.neutral_count == 0
        assert m.unknown_count == 0
        assert m.keyword_hit_count == 1
        assert m.llm_used_count == 0

    def test_all_unclassified_perception(self):
        """全てが未分類の知覚結果(neutral=True, unknown=True, hit=False)。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        assert m.neutral_count == 1
        assert m.unknown_count == 1
        assert m.keyword_hit_count == 0

    def test_record_added_to_fifo_buffer(self):
        """分類記録がFIFOバッファに追加される。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.buffer_size == 1
        summary = m.get_summary()
        latest = summary["latest_record"]
        assert latest is not None
        assert latest["emotion_is_neutral"] is False
        assert latest["intent_is_unknown"] is False
        assert latest["keyword_hit"] is True
        assert latest["llm_used"] is False

    def test_multiple_records(self):
        """複数回の記録が正しく累積される。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        m.record_perception("sad", "sharing", ["悲しい"], True)
        m.record_perception("neutral", "question", [], False)
        assert m.total_count == 4
        assert m.neutral_count == 2
        assert m.unknown_count == 1
        assert m.keyword_hit_count == 2
        assert m.llm_used_count == 1


# ── 第2段: セッション内の累積集計テスト ──────────────────────────


class TestCumulativeCounters:
    """セッション累積カウンタの検証。"""

    def test_counters_monotonic_increase(self):
        """カウンタは単調増加する。"""
        m = _make_coverage()
        for i in range(10):
            m.record_perception("neutral", "unknown", [], False)
            assert m.total_count == i + 1
            assert m.neutral_count == i + 1
            assert m.unknown_count == i + 1

    def test_counters_independent(self):
        """各カウンタは独立して加算される。"""
        m = _make_coverage()
        # Only neutral, not unknown, no keyword hit
        m.record_perception("neutral", "greeting", [], False)
        assert m.neutral_count == 1
        assert m.unknown_count == 0
        assert m.keyword_hit_count == 0

        # Only unknown, not neutral, with keyword hit
        m.record_perception("happy", "unknown", ["嬉しい"], False)
        assert m.neutral_count == 1
        assert m.unknown_count == 1
        assert m.keyword_hit_count == 1

    def test_large_count_accumulation(self):
        """大量の知覚結果でカウンタが正しく累積される。"""
        m = _make_coverage(buffer_max=50)
        for _ in range(500):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.total_count == 500
        assert m.neutral_count == 0
        assert m.unknown_count == 0
        assert m.keyword_hit_count == 500
        assert m.llm_used_count == 0
        # FIFOバッファは上限で制限される
        assert m.buffer_size == 50


# ── 第3段: サマリの読み取り専用提供テスト ────────────────────────


class TestReadOnlySummary:
    """get_summary()の構造。"""

    def test_get_summary_empty(self):
        """空状態でget_summaryが正しい構造を返す。"""
        m = _make_coverage()
        s = m.get_summary()
        assert s["total_count"] == 0
        assert s["neutral_count"] == 0
        assert s["unknown_count"] == 0
        assert s["keyword_hit_count"] == 0
        assert s["llm_used_count"] == 0
        assert s["buffer_size"] == 0
        assert s["latest_record"] is None

    def test_get_summary_with_data(self):
        """データがある状態でget_summaryが正しい値を返す。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("happy", "greeting", ["嬉しい"], True)
        s = m.get_summary()
        assert s["total_count"] == 2
        assert s["neutral_count"] == 1
        assert s["unknown_count"] == 1
        assert s["keyword_hit_count"] == 1
        assert s["llm_used_count"] == 1
        assert s["buffer_size"] == 2
        assert s["latest_record"] is not None
        assert s["latest_record"]["emotion_is_neutral"] is False

    def test_get_summary_returns_copy(self):
        """get_summaryは読み取り専用コピーを返す。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        s1 = m.get_summary()
        s2 = m.get_summary()
        assert s1 is not s2
        assert s1 == s2

    def test_latest_record_is_most_recent(self):
        """latest_recordは最新の記録を返す。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("happy", "greeting", ["嬉しい"], True)
        s = m.get_summary()
        assert s["latest_record"]["emotion_is_neutral"] is False
        assert s["latest_record"]["keyword_hit"] is True
        assert s["latest_record"]["llm_used"] is True


# ── FIFOバッファテスト ────────────────────────────────────────────


class TestFIFOBuffer:
    """FIFOバッファの上限と自然消失。"""

    def test_fifo_eviction(self):
        """上限を超えると最古の記録が自然消失する(安全弁2)。"""
        m = _make_coverage(buffer_max=3)
        for i in range(5):
            emotion = "neutral" if i < 2 else "happy"
            m.record_perception(emotion, "greeting", ["嬉しい"] if emotion == "happy" else [], False)
        assert m.buffer_size == 3
        # 最古の2件(neutral)が消失し、残りは最新3件
        s = m.get_summary()
        # Total count still tracks all 5
        assert s["total_count"] == 5

    def test_fifo_buffer_independence_from_counters(self):
        """FIFOバッファの消失はカウンタに影響しない。"""
        m = _make_coverage(buffer_max=2)
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        m.record_perception("sad", "sharing", ["悲しい"], True)
        # Buffer: only last 2
        assert m.buffer_size == 2
        # Counters: all 3
        assert m.total_count == 3
        assert m.neutral_count == 1
        assert m.unknown_count == 1
        assert m.keyword_hit_count == 2
        assert m.llm_used_count == 1


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作確認。"""

    def test_disabled_record_perception_noop(self):
        """無効時にrecord_perceptionは何もしない(安全弁6)。"""
        m = _make_coverage(enabled=False)
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.total_count == 0
        assert m.buffer_size == 0

    def test_disabled_emit_coverage_summary_noop(self):
        """無効時にemit_coverage_summaryは何もしない。"""
        m = _make_coverage(enabled=False)
        m.emit_coverage_summary()  # Should not raise

    def test_fifo_buffer_limit(self):
        """FIFOバッファの上限が正しく適用される(安全弁2)。"""
        m = _make_coverage(buffer_max=3)
        for _ in range(10):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.buffer_size == 3

    def test_exception_safety_in_record(self):
        """record_perception内部で予期せぬ状況でもエラーにならない(安全弁3)。"""
        m = _make_coverage()
        # Even with unusual inputs, should not raise
        m.record_perception("", "", [], False)
        assert m.total_count == 1

    def test_no_text_in_buffer(self):
        """FIFOバッファに入力テキストが保持されない(安全弁7)。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        s = m.get_summary()
        latest = s["latest_record"]
        assert "text" not in latest
        assert "input_text" not in latest
        assert "user_text" not in latest


# ── 永続化非対象テスト ───────────────────────────────────────────


class TestNoPersistence:
    """永続化対象外であることの確認(安全弁1)。"""

    def test_no_save_load_methods(self):
        """PerceptionCoverageMeasurementにsave/loadメソッドが存在しない。"""
        m = _make_coverage()
        assert not hasattr(m, "save")
        assert not hasattr(m, "load")
        assert not hasattr(m, "to_dict")
        assert not hasattr(m, "from_dict")

    def test_session_boundary_clears_state(self):
        """新しいインスタンスは空の状態で開始する。"""
        m1 = _make_coverage()
        m1.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m1.total_count == 1

        m2 = _make_coverage()
        assert m2.total_count == 0
        assert m2.buffer_size == 0


# ── psyche非参照テスト ───────────────────────────────────────────


class TestPsycheIsolation:
    """psycheモジュールとの構造的分離の確認(安全弁4)。"""

    def test_no_psyche_imports(self):
        """pipeline_measurement.pyがpsycheモジュールをインポートしない。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        assert "from psyche" not in source
        assert "import psyche" not in source

    def test_no_enrichment_methods(self):
        """enrichmentに関連するメソッドが存在しない。"""
        m = _make_coverage()
        assert not hasattr(m, "get_enrichment")
        assert not hasattr(m, "enrichment")
        assert not hasattr(m, "get_prompt_enrichment")

    def test_no_orchestrator_dependency(self):
        """orchestratorへの依存がない(importレベル)。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "orchestrator" not in line.lower()


# ── 辞書非変更テスト ───────────────────────────────────────────


class TestDictionaryReadOnly:
    """辞書の追加・削除・変更を行わないことの確認(安全弁5)。"""

    def test_no_dictionary_modification_methods(self):
        """辞書を変更するメソッドが存在しない。"""
        m = _make_coverage()
        assert not hasattr(m, "add_keyword")
        assert not hasattr(m, "remove_keyword")
        assert not hasattr(m, "update_dictionary")
        assert not hasattr(m, "modify_dictionary")

    def test_perception_dictionary_unchanged_after_recording(self):
        """record_perception後に知覚辞書が変更されていない。"""
        from psyche.perception import _EMOTION_KEYWORDS, _INTENT_KEYWORDS
        original_emotion_size = len(_EMOTION_KEYWORDS)
        original_intent_size = len(_INTENT_KEYWORDS)

        m = _make_coverage()
        for _ in range(100):
            m.record_perception("happy", "greeting", ["嬉しい"], False)

        assert len(_EMOTION_KEYWORDS) == original_emotion_size
        assert len(_INTENT_KEYWORDS) == original_intent_size


# ── ログ出力テスト ───────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のフォーマット検証。"""

    def test_perception_coverage_log_format(self, caplog):
        """perception_coverageログのJSON構造。"""
        m = _make_coverage()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "perception_coverage" in msg and "session_summary" not in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "perception_coverage"
                assert "timestamp" in data
                assert data["emotion_is_neutral"] is False
                assert data["intent_is_unknown"] is False
                assert data["keyword_hit"] is True
                assert data["llm_used"] is False
                break
        assert found, "perception_coverage log not found"

    def test_coverage_summary_log_format(self, caplog):
        """perception_coverage_session_summaryログのJSON構造。"""
        m = _make_coverage()
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("happy", "greeting", ["嬉しい"], True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_coverage_summary()
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "perception_coverage_session_summary" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "perception_coverage_session_summary"
                assert data["total_count"] == 2
                assert data["neutral_count"] == 1
                assert data["unknown_count"] == 1
                assert data["keyword_hit_count"] == 1
                assert data["llm_used_count"] == 1
                assert "buffer_size" in data
                break
        assert found, "perception_coverage_session_summary log not found"

    def test_no_log_when_disabled(self, caplog):
        """無効時にはログが出力されない。"""
        m = _make_coverage(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
            m.emit_coverage_summary()
        for record in caplog.records:
            assert "perception_coverage" not in record.getMessage()


# ── 統合テスト ───────────────────────────────────────────────────


class TestIntegration:
    """PerceptionCoverageMeasurementの統合テスト。"""

    def test_realistic_session_simulation(self):
        """現実的なセッションのシミュレーション。"""
        m = _make_coverage()

        # Typical session: mix of classified and unclassified
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("sad", "sharing", ["悲しい", "泣"], False)
        m.record_perception("neutral", "question", [], True)
        m.record_perception("angry", "complaint", ["怒", "ムカ"], False)
        m.record_perception("neutral", "unknown", [], True)
        m.record_perception("happy", "gratitude", ["ありがとう"], False)
        m.record_perception("surprised", "unknown", ["びっくり"], False)
        m.record_perception("neutral", "unknown", [], False)
        m.record_perception("loving", "sharing", ["好き"], False)

        s = m.get_summary()
        assert s["total_count"] == 10
        assert s["neutral_count"] == 4
        assert s["unknown_count"] == 4
        assert s["keyword_hit_count"] == 6
        assert s["llm_used_count"] == 2

    def test_coexistence_with_pipeline_measurement(self):
        """PipelineMeasurementとの共存(同一セッション内)。"""
        from tools.pipeline_measurement import PipelineMeasurement, PATHWAY_TEXT
        pm = PipelineMeasurement(enabled=True)
        cm = _make_coverage()

        # Both can operate independently
        pm.begin_pipeline(PATHWAY_TEXT)
        cm.record_perception("happy", "greeting", ["嬉しい"], False)
        pm.end_pipeline()

        assert pm.record_count == 1
        assert cm.total_count == 1

    def test_session_lifecycle(self):
        """セッションのライフサイクル: 計測→サマリ出力。"""
        m = _make_coverage()

        # Multiple perceptions
        for _ in range(5):
            m.record_perception("happy", "greeting", ["嬉しい"], False)
        for _ in range(3):
            m.record_perception("neutral", "unknown", [], True)

        summary = m.get_summary()
        assert summary["total_count"] == 8
        assert summary["neutral_count"] == 3
        assert summary["unknown_count"] == 3
        assert summary["keyword_hit_count"] == 5
        assert summary["llm_used_count"] == 3
        assert summary["buffer_size"] == 8


# ── エッジケーステスト ───────────────────────────────────────────


class TestEdgeCases:
    """エッジケースの処理。"""

    def test_empty_emotion_string(self):
        """空文字列の感情ラベル。"""
        m = _make_coverage()
        m.record_perception("", "greeting", [], False)
        assert m.neutral_count == 0  # "" != "neutral"
        assert m.total_count == 1

    def test_empty_intent_string(self):
        """空文字列の意図ラベル。"""
        m = _make_coverage()
        m.record_perception("happy", "", ["嬉しい"], False)
        assert m.unknown_count == 0  # "" != "unknown"
        assert m.total_count == 1

    def test_single_topic(self):
        """話題リストに1要素の場合。"""
        m = _make_coverage()
        m.record_perception("happy", "greeting", ["嬉しい"], False)
        assert m.keyword_hit_count == 1

    def test_many_topics(self):
        """話題リストに多数の要素がある場合。"""
        m = _make_coverage()
        topics = ["嬉しい", "楽しい", "幸せ", "やったー", "最高"]
        m.record_perception("happy", "sharing", topics, False)
        assert m.keyword_hit_count == 1  # Still 1 (hit is boolean)

    def test_custom_emotion_label(self):
        """辞書にない感情ラベル(LLM由来)。"""
        m = _make_coverage()
        m.record_perception("proud", "sharing", [], True)
        assert m.neutral_count == 0
        assert m.llm_used_count == 1

    def test_custom_intent_label(self):
        """辞書にない意図ラベル(LLM由来)。"""
        m = _make_coverage()
        m.record_perception("happy", "monologue", ["嬉しい"], True)
        assert m.unknown_count == 0
        assert m.llm_used_count == 1

    def test_rapid_recording(self):
        """高速な連続記録でもエラーにならない。"""
        m = _make_coverage()
        for _ in range(1000):
            m.record_perception("neutral", "unknown", [], False)
        assert m.total_count == 1000
