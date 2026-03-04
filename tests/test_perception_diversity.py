"""
tests/test_perception_diversity.py - PerceptionDiversityMeasurement のテスト

設計書: design_perception_diversity.md

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定、環境変数制御)
- 断面記録テスト(感情ラベル・意図ラベル・話題数・入力長・感情価・keyword_hit)
- FIFOバッファテスト(上限、自然消失)
- ラベル種類のセッション累積テスト(感情ラベル・意図ラベル)
- 入力長の累積情報テスト(最小・最大・平均)
- セッションサマリ出力テスト
- 読み取り専用アクセサテスト
- 安全弁テスト(記録失敗時の安全な無視、無効時の完全スキップ)
- 永続化非対象テスト
- psyche非参照テスト(enrichment・orchestratorへの非依存)
- 入力テキスト非保持テスト(テキスト長のみ)
- perception.pyフックテスト(parse_percept経由の記録)
- ログ出力テスト(JSON構造)
- エッジケーステスト
"""

import json
import logging
import os
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.pipeline_measurement import (
    PerceptionDiversityRecord,
    PerceptionDiversityMeasurement,
    _DEFAULT_DIVERSITY_BUFFER_MAX,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_diversity(
    enabled: bool = True,
    buffer_max: int = 200,
) -> PerceptionDiversityMeasurement:
    """テスト用のPerceptionDiversityMeasurementを生成する。"""
    return PerceptionDiversityMeasurement(enabled=enabled, buffer_max=buffer_max)


def _record_sample(
    m: PerceptionDiversityMeasurement,
    emotion: str = "happy",
    intent: str = "greeting",
    topic_count: int = 2,
    input_length: int = 10,
    emotion_valence: float = 0.6,
    keyword_hit: bool = True,
) -> None:
    """テスト用のサンプル記録を行う。"""
    m.record_perception_diversity(
        emotion_label=emotion,
        intent_label=intent,
        topic_count=topic_count,
        input_length=input_length,
        emotion_valence=emotion_valence,
        keyword_hit=keyword_hit,
    )


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """PerceptionDiversityMeasurementの初期化テスト。"""

    def test_default_init_disabled(self):
        """デフォルトでは計測は無効(CYRENE_MONITOR未設定)。"""
        with patch.dict(os.environ, {}, clear=False):
            if "CYRENE_MONITOR" in os.environ:
                del os.environ["CYRENE_MONITOR"]
            m = PerceptionDiversityMeasurement()
            assert m.enabled is False

    def test_enabled_via_constructor(self):
        """コンストラクタで明示的に有効化。"""
        m = _make_diversity(enabled=True)
        assert m.enabled is True

    def test_disabled_via_constructor(self):
        """コンストラクタで明示的に無効化。"""
        m = _make_diversity(enabled=False)
        assert m.enabled is False

    def test_enabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=1で有効化(安全弁5)。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PerceptionDiversityMeasurement()
            assert m.enabled is True

    def test_disabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=0で無効化(安全弁5)。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            m = PerceptionDiversityMeasurement()
            assert m.enabled is False

    def test_constructor_overrides_env(self):
        """コンストラクタの指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PerceptionDiversityMeasurement(enabled=False)
            assert m.enabled is False

    def test_initial_state_empty(self):
        """初期状態で全カウンタが空。"""
        m = _make_diversity()
        assert m.buffer_size == 0
        assert m.emotion_label_counts == {}
        assert m.intent_label_counts == {}
        assert m.total_count == 0

    def test_buffer_max_applied(self):
        """FIFOバッファの上限が適用される。"""
        m = _make_diversity(buffer_max=5)
        for i in range(10):
            _record_sample(m, emotion=f"emo_{i}")
        assert m.buffer_size == 5

    def test_buffer_max_minimum(self):
        """FIFOバッファの上限は最低1。"""
        m = _make_diversity(buffer_max=0)
        _record_sample(m)
        assert m.buffer_size == 1

    def test_default_buffer_max(self):
        """デフォルトのFIFOバッファ上限が200。"""
        assert _DEFAULT_DIVERSITY_BUFFER_MAX == 200


# ── PerceptionDiversityRecord テスト ─────────────────────────────


class TestPerceptionDiversityRecord:
    """PerceptionDiversityRecordの単体テスト。"""

    def test_init(self):
        """初期状態。"""
        r = PerceptionDiversityRecord(
            emotion_label="happy",
            intent_label="greeting",
            topic_count=3,
            input_length=20,
            emotion_valence=0.7,
            keyword_hit=True,
        )
        assert r.emotion_label == "happy"
        assert r.intent_label == "greeting"
        assert r.topic_count == 3
        assert r.input_length == 20
        assert r.emotion_valence == 0.7
        assert r.keyword_hit is True
        assert r.timestamp > 0

    def test_to_dict(self):
        """辞書表現の構造。"""
        r = PerceptionDiversityRecord(
            emotion_label="sad",
            intent_label="sharing",
            topic_count=1,
            input_length=15,
            emotion_valence=-0.5,
            keyword_hit=False,
        )
        d = r.to_dict()
        assert d["emotion_label"] == "sad"
        assert d["intent_label"] == "sharing"
        assert d["topic_count"] == 1
        assert d["input_length"] == 15
        assert d["emotion_valence"] == -0.5
        assert d["keyword_hit"] is False
        assert "timestamp" in d

    def test_to_dict_valence_rounding(self):
        """感情価が4桁に丸められる。"""
        r = PerceptionDiversityRecord(
            emotion_label="neutral",
            intent_label="unknown",
            topic_count=0,
            input_length=5,
            emotion_valence=0.123456789,
            keyword_hit=False,
        )
        d = r.to_dict()
        assert d["emotion_valence"] == 0.1235

    def test_slots_prevent_arbitrary_attrs(self):
        """__slots__で任意の属性追加が防がれる。"""
        r = PerceptionDiversityRecord(
            emotion_label="happy",
            intent_label="greeting",
            topic_count=0,
            input_length=5,
            emotion_valence=0.0,
            keyword_hit=False,
        )
        with pytest.raises(AttributeError):
            r.arbitrary_field = "test"


# ── 断面記録テスト ────────────────────────────────────────────────


class TestDiversityRecording:
    """知覚結果の断面記録。"""

    def test_basic_record(self):
        """基本的な断面記録が正しく行われる。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy", intent="greeting", topic_count=2,
                       input_length=10, emotion_valence=0.6, keyword_hit=True)
        assert m.buffer_size == 1
        assert m.total_count == 1

    def test_multiple_records(self):
        """複数の断面記録が正しく蓄積される。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy", intent="greeting")
        _record_sample(m, emotion="sad", intent="sharing")
        _record_sample(m, emotion="angry", intent="complaint")
        assert m.buffer_size == 3
        assert m.total_count == 3

    def test_record_preserves_all_fields(self):
        """記録された断面が全フィールドを保持する。"""
        m = _make_diversity()
        _record_sample(m, emotion="surprised", intent="question",
                       topic_count=5, input_length=42, emotion_valence=0.3,
                       keyword_hit=True)
        summary = m.get_summary()
        latest = summary["latest_record"]
        assert latest["emotion_label"] == "surprised"
        assert latest["intent_label"] == "question"
        assert latest["topic_count"] == 5
        assert latest["input_length"] == 42
        assert latest["emotion_valence"] == 0.3
        assert latest["keyword_hit"] is True

    def test_disabled_recording_noop(self):
        """無効時に記録は行われない。"""
        m = _make_diversity(enabled=False)
        _record_sample(m)
        assert m.buffer_size == 0
        assert m.total_count == 0

    def test_neutral_emotion_recorded(self):
        """neutral感情ラベルが正しく記録される。"""
        m = _make_diversity()
        _record_sample(m, emotion="neutral", keyword_hit=False)
        assert m.emotion_label_counts["neutral"] == 1


# ── FIFOバッファテスト ────────────────────────────────────────────


class TestFIFOBuffer:
    """FIFOバッファの上限と自然消失。"""

    def test_fifo_eviction(self):
        """上限を超えると最古の記録が自然消失する(安全弁1)。"""
        m = _make_diversity(buffer_max=3)
        for i in range(5):
            _record_sample(m, emotion=f"emo_{i}", input_length=i * 10)
        assert m.buffer_size == 3
        # 最新3件のみ残る
        summary = m.get_summary()
        latest = summary["latest_record"]
        assert latest["input_length"] == 40

    def test_cumulative_counters_not_affected_by_eviction(self):
        """FIFOバッファの消失は累積カウンタに影響しない。"""
        m = _make_diversity(buffer_max=3)
        for i in range(5):
            _record_sample(m, emotion="happy")
        assert m.buffer_size == 3
        assert m.emotion_label_counts["happy"] == 5
        assert m.total_count == 5


# ── ラベル種類のセッション累積テスト ──────────────────────────────


class TestLabelCumulative:
    """ラベル種類のセッション累積カウンタ。"""

    def test_emotion_label_counting(self):
        """感情ラベルの出現回数が正しく累積される。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy")
        _record_sample(m, emotion="happy")
        _record_sample(m, emotion="sad")
        _record_sample(m, emotion="angry")
        _record_sample(m, emotion="happy")
        counts = m.emotion_label_counts
        assert counts["happy"] == 3
        assert counts["sad"] == 1
        assert counts["angry"] == 1

    def test_intent_label_counting(self):
        """意図ラベルの出現回数が正しく累積される。"""
        m = _make_diversity()
        _record_sample(m, intent="greeting")
        _record_sample(m, intent="question")
        _record_sample(m, intent="greeting")
        _record_sample(m, intent="sharing")
        counts = m.intent_label_counts
        assert counts["greeting"] == 2
        assert counts["question"] == 1
        assert counts["sharing"] == 1

    def test_label_counts_return_copy(self):
        """累積カウンタプロパティは読み取り専用コピーを返す。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy", intent="greeting")
        c1 = m.emotion_label_counts
        c2 = m.emotion_label_counts
        assert c1 is not c2
        assert c1 == c2
        i1 = m.intent_label_counts
        i2 = m.intent_label_counts
        assert i1 is not i2
        assert i1 == i2

    def test_multiple_unique_labels(self):
        """多数のユニークなラベルが正しく記録される。"""
        m = _make_diversity()
        emotions = ["happy", "sad", "angry", "surprised", "scared",
                     "loving", "teasing", "neutral", "confused", "disappointed"]
        for emo in emotions:
            _record_sample(m, emotion=emo)
        assert len(m.emotion_label_counts) == 10
        for emo in emotions:
            assert m.emotion_label_counts[emo] == 1


# ── 入力長の累積情報テスト ────────────────────────────────────────


class TestInputLengthCumulative:
    """入力長の累積情報。"""

    def test_single_record(self):
        """1件の記録で最小・最大・平均が一致する。"""
        m = _make_diversity()
        _record_sample(m, input_length=25)
        summary = m.get_summary()
        assert summary["input_length_min"] == 25
        assert summary["input_length_max"] == 25
        assert summary["input_length_avg"] == 25.0

    def test_multiple_records(self):
        """複数記録で最小・最大・平均が正しく算出される。"""
        m = _make_diversity()
        _record_sample(m, input_length=10)
        _record_sample(m, input_length=30)
        _record_sample(m, input_length=20)
        summary = m.get_summary()
        assert summary["input_length_min"] == 10
        assert summary["input_length_max"] == 30
        assert summary["input_length_avg"] == 20.0

    def test_zero_length_input(self):
        """入力長0が正しく記録される。"""
        m = _make_diversity()
        _record_sample(m, input_length=0)
        summary = m.get_summary()
        assert summary["input_length_min"] == 0
        assert summary["input_length_max"] == 0

    def test_empty_state_returns_none(self):
        """記録なしの状態でmin/maxがNone。"""
        m = _make_diversity()
        summary = m.get_summary()
        assert summary["input_length_min"] is None
        assert summary["input_length_max"] is None
        assert summary["input_length_avg"] == 0.0


# ── セッションサマリ出力テスト ────────────────────────────────────


class TestDiversitySummary:
    """セッションサマリの出力。"""

    def test_emit_diversity_summary(self, caplog):
        """セッションサマリがログに出力される。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy", intent="greeting", input_length=10)
        _record_sample(m, emotion="sad", intent="sharing", input_length=20)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_diversity_summary()

        found = False
        for record in caplog.records:
            if "perception_diversity_session_summary" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["total_count"] == 2
                assert data["emotion_label_counts"]["happy"] == 1
                assert data["emotion_label_counts"]["sad"] == 1
                assert data["intent_label_counts"]["greeting"] == 1
                assert data["intent_label_counts"]["sharing"] == 1
                assert data["emotion_label_unique_count"] == 2
                assert data["intent_label_unique_count"] == 2
                assert data["input_length_min"] == 10
                assert data["input_length_max"] == 20
                assert data["input_length_avg"] == 15.0
                break
        assert found, "perception_diversity_session_summary log not found"

    def test_emit_summary_when_disabled(self, caplog):
        """無効時にはサマリは出力されない。"""
        m = _make_diversity(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_diversity_summary()
        for record in caplog.records:
            assert "perception_diversity_session_summary" not in record.getMessage()

    def test_emit_summary_empty_state(self, caplog):
        """記録なしでもサマリ出力がエラーにならない。"""
        m = _make_diversity()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_diversity_summary()
        found = False
        for record in caplog.records:
            if "perception_diversity_session_summary" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["total_count"] == 0
                break
        assert found


# ── 読み取り専用アクセサテスト ────────────────────────────────────


class TestReadOnlyAccessor:
    """get_summary()の構造。"""

    def test_get_summary_empty(self):
        """空状態でget_summaryが正しい構造を返す。"""
        m = _make_diversity()
        s = m.get_summary()
        assert s["total_count"] == 0
        assert s["emotion_label_counts"] == {}
        assert s["intent_label_counts"] == {}
        assert s["emotion_label_unique_count"] == 0
        assert s["intent_label_unique_count"] == 0
        assert s["input_length_min"] is None
        assert s["input_length_max"] is None
        assert s["input_length_avg"] == 0.0
        assert s["buffer_size"] == 0
        assert s["latest_record"] is None

    def test_get_summary_with_data(self):
        """データがある状態でget_summaryが正しい値を返す。"""
        m = _make_diversity()
        _record_sample(m, emotion="happy", intent="greeting", input_length=15)
        s = m.get_summary()
        assert s["total_count"] == 1
        assert s["emotion_label_counts"]["happy"] == 1
        assert s["intent_label_counts"]["greeting"] == 1
        assert s["emotion_label_unique_count"] == 1
        assert s["intent_label_unique_count"] == 1
        assert s["input_length_min"] == 15
        assert s["input_length_max"] == 15
        assert s["buffer_size"] == 1
        assert s["latest_record"] is not None
        assert s["latest_record"]["emotion_label"] == "happy"

    def test_get_summary_returns_copy(self):
        """get_summaryは読み取り専用コピーを返す。"""
        m = _make_diversity()
        _record_sample(m)
        s1 = m.get_summary()
        s2 = m.get_summary()
        assert s1 is not s2


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作確認。"""

    def test_disabled_recording_noop(self):
        """無効時に記録は完全にスキップされる。"""
        m = _make_diversity(enabled=False)
        _record_sample(m)
        assert m.buffer_size == 0
        assert m.total_count == 0
        assert m.emotion_label_counts == {}
        assert m.intent_label_counts == {}

    def test_fifo_buffer_limit(self):
        """FIFOバッファの上限が正しく適用される(安全弁1)。"""
        m = _make_diversity(buffer_max=3)
        for _ in range(10):
            _record_sample(m)
        assert m.buffer_size == 3

    def test_recording_exception_ignored(self):
        """記録処理の例外が安全に無視される(安全弁2)。"""
        m = _make_diversity()
        # _emotion_label_countsの__getitem__を壊して例外を発生させる
        m._emotion_label_counts = None  # type: ignore
        # 例外が発生しても外部には伝播しない
        _record_sample(m)  # Should not raise

    def test_input_text_not_stored(self):
        """入力テキストの全文が記録に含まれない(安全弁3)。"""
        m = _make_diversity()
        _record_sample(m, input_length=100)
        summary = m.get_summary()
        latest = summary["latest_record"]
        # input_lengthは記録されるがテキスト内容は記録されない
        assert "text" not in latest
        assert "input_text" not in latest
        assert "user_text" not in latest
        assert latest["input_length"] == 100

    def test_no_psyche_feedback(self):
        """記録データがpsycheに帰還しない(安全弁4)。"""
        m = _make_diversity()
        _record_sample(m)
        # enrichmentメソッドが存在しない
        assert not hasattr(m, "get_enrichment")
        assert not hasattr(m, "enrichment")
        assert not hasattr(m, "get_prompt_enrichment")


# ── 永続化非対象テスト ───────────────────────────────────────────


class TestNoPersistence:
    """永続化対象外であることの確認(安全弁6)。"""

    def test_no_save_load_methods(self):
        """PerceptionDiversityMeasurementにsave/loadメソッドが存在しない。"""
        m = _make_diversity()
        assert not hasattr(m, "save")
        assert not hasattr(m, "load")
        assert not hasattr(m, "to_dict")
        assert not hasattr(m, "from_dict")

    def test_session_boundary_clears_state(self):
        """新しいインスタンスは空の状態で開始する。"""
        m1 = _make_diversity()
        _record_sample(m1)
        assert m1.total_count == 1

        m2 = _make_diversity()
        assert m2.total_count == 0
        assert m2.buffer_size == 0
        assert m2.emotion_label_counts == {}


# ── psyche非参照テスト ───────────────────────────────────────────


class TestPsycheIsolation:
    """psycheモジュールとの構造的分離の確認。"""

    def test_no_psyche_imports(self):
        """pipeline_measurement.pyがpsycheモジュールをインポートしない。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        # コメント・docstringを除いた実際のimport文を検査
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "psyche" not in line

    def test_no_enrichment_methods(self):
        """enrichmentに関連するメソッドが存在しない。"""
        m = _make_diversity()
        assert not hasattr(m, "get_enrichment")
        assert not hasattr(m, "enrichment")
        assert not hasattr(m, "get_prompt_enrichment")

    def test_no_orchestrator_dependency(self):
        """orchestratorへの依存がない。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "orchestrator" not in line.lower()


# ── ログ出力テスト ───────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のフォーマット検証。"""

    def test_record_log_format(self, caplog):
        """perception_diversityログのJSON構造。"""
        m = _make_diversity()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            _record_sample(m, emotion="happy", intent="greeting",
                          topic_count=3, input_length=20,
                          emotion_valence=0.7, keyword_hit=True)
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "perception_diversity" in msg and "session_summary" not in msg:
                data = json.loads(msg)
                assert data["type"] == "perception_diversity"
                assert "timestamp" in data
                assert data["emotion_label"] == "happy"
                assert data["intent_label"] == "greeting"
                assert data["topic_count"] == 3
                assert data["input_length"] == 20
                assert data["emotion_valence"] == 0.7
                assert data["keyword_hit"] is True
                found = True
                break
        assert found, "perception_diversity log not found"

    def test_summary_log_format(self, caplog):
        """perception_diversity_session_summaryログのJSON構造。"""
        m = _make_diversity()
        _record_sample(m)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_diversity_summary()
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "perception_diversity_session_summary" in msg:
                data = json.loads(msg)
                assert data["type"] == "perception_diversity_session_summary"
                assert "timestamp" in data
                assert "total_count" in data
                assert "emotion_label_counts" in data
                assert "intent_label_counts" in data
                assert "emotion_label_unique_count" in data
                assert "intent_label_unique_count" in data
                assert "input_length_min" in data
                assert "input_length_max" in data
                assert "input_length_avg" in data
                assert "buffer_size" in data
                found = True
                break
        assert found, "perception_diversity_session_summary log not found"

    def test_disabled_no_log(self, caplog):
        """無効時にログ出力がない。"""
        m = _make_diversity(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            _record_sample(m)
        for record in caplog.records:
            assert "perception_diversity" not in record.getMessage()


# ── perception.pyフックテスト ────────────────────────────────────


class TestPerceptionHook:
    """perception.pyのparse_perceptからのフック呼び出し。"""

    @pytest.mark.asyncio
    async def test_parse_percept_with_diversity_recorder(self):
        """parse_perceptにdiversity_recorderを渡すと記録される。"""
        from psyche.perception import parse_percept

        m = _make_diversity()
        percept = await parse_percept("嬉しいニュースだね", diversity_recorder=m)
        # 知覚結果は正常に返される
        assert percept is not None
        assert percept.emotion == "happy"
        # 多様性記録が行われた
        assert m.total_count == 1
        assert m.emotion_label_counts.get("happy", 0) == 1

    @pytest.mark.asyncio
    async def test_parse_percept_without_diversity_recorder(self):
        """diversity_recorder=Noneでも正常に動作する。"""
        from psyche.perception import parse_percept

        percept = await parse_percept("こんにちは")
        assert percept is not None
        # 何もクラッシュしない

    @pytest.mark.asyncio
    async def test_parse_percept_diversity_recorder_default_none(self):
        """diversity_recorderのデフォルト値がNone。"""
        from psyche.perception import parse_percept

        percept = await parse_percept("テスト")
        assert percept is not None

    @pytest.mark.asyncio
    async def test_recorder_does_not_affect_percept(self):
        """recorderの有無がperceptの結果に影響しない。"""
        from psyche.perception import parse_percept

        percept_without = await parse_percept("嬉しい")
        m = _make_diversity()
        percept_with = await parse_percept("嬉しい", diversity_recorder=m)
        assert percept_without.emotion == percept_with.emotion
        assert percept_without.intent == percept_with.intent
        assert percept_without.emotion_valence == percept_with.emotion_valence

    @pytest.mark.asyncio
    async def test_recorder_exception_does_not_break_percept(self):
        """recorderが例外を発生させてもperceptは正常に返される。"""
        from psyche.perception import parse_percept

        broken_recorder = MagicMock()
        broken_recorder.record_perception_diversity.side_effect = RuntimeError("test")

        percept = await parse_percept("嬉しい", diversity_recorder=broken_recorder)
        assert percept is not None
        assert percept.emotion == "happy"

    @pytest.mark.asyncio
    async def test_recorder_captures_correct_input_length(self):
        """recorderに正しいテキスト長が記録される。"""
        from psyche.perception import parse_percept

        m = _make_diversity()
        text = "こんにちは、元気ですか？"
        await parse_percept(text, diversity_recorder=m)
        summary = m.get_summary()
        assert summary["latest_record"]["input_length"] == len(text)

    @pytest.mark.asyncio
    async def test_recorder_captures_correct_topic_count(self):
        """recorderに正しい話題数が記録される。"""
        from psyche.perception import parse_percept

        m = _make_diversity()
        # "嬉しい" は emotion keyword -> topics に含まれる
        await parse_percept("嬉しい", diversity_recorder=m)
        summary = m.get_summary()
        assert summary["latest_record"]["topic_count"] >= 1
        assert summary["latest_record"]["keyword_hit"] is True

    @pytest.mark.asyncio
    async def test_recorder_neutral_no_keywords(self):
        """キーワードなし入力でneutral/unknown/keyword_hit=Falseが記録される。"""
        from psyche.perception import parse_percept

        m = _make_diversity()
        await parse_percept("abc", diversity_recorder=m)
        summary = m.get_summary()
        latest = summary["latest_record"]
        assert latest["emotion_label"] == "neutral"
        assert latest["intent_label"] == "unknown"
        assert latest["keyword_hit"] is False
        assert latest["topic_count"] == 0


# ── エッジケーステスト ───────────────────────────────────────────


class TestEdgeCases:
    """エッジケースの処理。"""

    def test_empty_emotion_label(self):
        """空の感情ラベルが記録可能。"""
        m = _make_diversity()
        _record_sample(m, emotion="")
        assert m.emotion_label_counts[""] == 1

    def test_empty_intent_label(self):
        """空の意図ラベルが記録可能。"""
        m = _make_diversity()
        _record_sample(m, intent="")
        assert m.intent_label_counts[""] == 1

    def test_negative_emotion_valence(self):
        """負の感情価が正しく記録される。"""
        m = _make_diversity()
        _record_sample(m, emotion_valence=-0.8)
        summary = m.get_summary()
        assert summary["latest_record"]["emotion_valence"] == -0.8

    def test_zero_topic_count(self):
        """話題数0が正しく記録される。"""
        m = _make_diversity()
        _record_sample(m, topic_count=0)
        summary = m.get_summary()
        assert summary["latest_record"]["topic_count"] == 0

    def test_large_input_length(self):
        """非常に長い入力長が正しく記録される。"""
        m = _make_diversity()
        _record_sample(m, input_length=100000)
        summary = m.get_summary()
        assert summary["input_length_max"] == 100000

    def test_large_number_of_records(self):
        """大量の記録でメモリが制限される(安全弁1)。"""
        m = _make_diversity(buffer_max=50)
        for i in range(500):
            _record_sample(m, emotion=f"emo_{i % 10}", input_length=i)
        assert m.buffer_size == 50
        assert m.total_count == 500
        assert len(m.emotion_label_counts) == 10

    def test_concurrent_emotion_and_intent_variety(self):
        """多種の感情・意図ラベルが正しく累積される。"""
        m = _make_diversity()
        combos = [
            ("happy", "greeting"),
            ("sad", "sharing"),
            ("angry", "complaint"),
            ("surprised", "question"),
            ("neutral", "unknown"),
            ("happy", "question"),
            ("sad", "farewell"),
        ]
        for emo, intent in combos:
            _record_sample(m, emotion=emo, intent=intent)
        assert m.total_count == 7
        assert len(m.emotion_label_counts) == 5
        assert len(m.intent_label_counts) == 6
        assert m.emotion_label_counts["happy"] == 2
        assert m.emotion_label_counts["sad"] == 2
        assert m.intent_label_counts["question"] == 2


# ── 既存のPerceptionCoverageMeasurementとの非重複テスト ───────────


class TestNonOverlapWithCoverage:
    """既存のPerceptionCoverageMeasurementとの重複回避。"""

    def test_different_record_types(self):
        """DiversityRecordとCoverageRecordは異なるフィールドを持つ。"""
        from tools.pipeline_measurement import PerceptionCoverageRecord

        div_record = PerceptionDiversityRecord(
            emotion_label="happy", intent_label="greeting",
            topic_count=2, input_length=10,
            emotion_valence=0.6, keyword_hit=True,
        )
        cov_record = PerceptionCoverageRecord(
            emotion_is_neutral=False, intent_is_unknown=False,
            keyword_hit=True, llm_used=False,
        )
        div_dict = div_record.to_dict()
        cov_dict = cov_record.to_dict()
        # DiversityRecordはラベル文字列そのものを持つ
        assert "emotion_label" in div_dict
        assert "intent_label" in div_dict
        # CoverageRecordは真偽値の分類のみを持つ
        assert "emotion_is_neutral" in cov_dict
        assert "intent_is_unknown" in cov_dict
        # 重複フィールドはkeyword_hitとtimestampのみ(設計上許容)

    def test_different_log_types(self, caplog):
        """DiversityとCoverageのログタイプ名が異なる。"""
        from tools.pipeline_measurement import PerceptionCoverageMeasurement

        div_m = _make_diversity()
        cov_m = PerceptionCoverageMeasurement(enabled=True)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            _record_sample(div_m)
            cov_m.record_perception(
                emotion="happy", intent="greeting",
                topics=["嬉しい"], llm_used=False,
            )

        log_types = set()
        for record in caplog.records:
            msg = record.getMessage()
            try:
                data = json.loads(msg)
                log_types.add(data.get("type"))
            except (json.JSONDecodeError, AttributeError):
                pass
        assert "perception_diversity" in log_types
        assert "perception_coverage" in log_types
