"""
tests/test_pipeline_measurement.py - PipelineMeasurement のテスト

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定、環境変数制御)
- パイプライン開始/完了テスト
- フェーズ区間記録テスト
- FIFOバッファテスト(上限、自然消失)
- セッション累積カウンタテスト(経路別、フェーズ別)
- 入力経路別テスト(vision/text/internal)
- セッションサマリテスト
- 読み取り専用アクセサテスト
- PhaseTimerテスト(正常、例外時、無効時)
- 安全弁テスト(計測失敗時の無視、無効時の完全スキップ)
- 永続化非対象テスト
- psyche非参照テスト
"""

import json
import logging
import os
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.pipeline_measurement import (
    PipelineMeasurement,
    PipelineRecord,
    PhaseTimer,
    PHASE_PERCEPTION_API,
    PHASE_PERCEPTION_PARSE,
    PHASE_PSYCHE_UPDATE,
    PHASE_MEMORY_RECALL,
    PHASE_POLICY_SELECT,
    PHASE_EXPRESSION_API,
    PHASE_PIPELINE_TOTAL,
    PATHWAY_VISION,
    PATHWAY_TEXT,
    PATHWAY_INTERNAL,
    _DEFAULT_PIPELINE_BUFFER_MAX,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_measurement(
    enabled: bool = True,
    buffer_max: int = 200,
) -> PipelineMeasurement:
    """テスト用のPipelineMeasurementを生成する。"""
    return PipelineMeasurement(enabled=enabled, buffer_max=buffer_max)


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """PipelineMeasurementの初期化テスト。"""

    def test_default_init_disabled(self):
        """デフォルトでは計測は無効(CYRENE_MONITOR未設定)。"""
        with patch.dict(os.environ, {}, clear=False):
            if "CYRENE_MONITOR" in os.environ:
                del os.environ["CYRENE_MONITOR"]
            m = PipelineMeasurement()
            assert m.enabled is False

    def test_enabled_via_constructor(self):
        """コンストラクタで明示的に有効化。"""
        m = _make_measurement(enabled=True)
        assert m.enabled is True

    def test_disabled_via_constructor(self):
        """コンストラクタで明示的に無効化。"""
        m = _make_measurement(enabled=False)
        assert m.enabled is False

    def test_enabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=1で有効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PipelineMeasurement()
            assert m.enabled is True

    def test_disabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=0で無効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            m = PipelineMeasurement()
            assert m.enabled is False

    def test_constructor_overrides_env(self):
        """コンストラクタの指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = PipelineMeasurement(enabled=False)
            assert m.enabled is False

    def test_initial_counters_empty(self):
        """初期状態で全カウンタが空。"""
        m = _make_measurement()
        assert m.pathway_count == {}
        assert m.pathway_phase_cumulative == {}
        assert m.pathway_total_cumulative == {}
        assert m.record_count == 0

    def test_buffer_max_applied(self):
        """FIFOバッファの上限が適用される。"""
        m = _make_measurement(buffer_max=5)
        # 5件を超えて追加すると最古が消失
        for i in range(10):
            m.begin_pipeline(PATHWAY_VISION)
            m.end_pipeline()
        assert m.record_count == 5

    def test_buffer_max_minimum(self):
        """FIFOバッファの上限は最低1。"""
        m = _make_measurement(buffer_max=0)
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        assert m.record_count == 1


# ── PipelineRecord テスト ─────────────────────────────────────────


class TestPipelineRecord:
    """PipelineRecordの単体テスト。"""

    def test_init(self):
        """初期状態。"""
        r = PipelineRecord(PATHWAY_VISION)
        assert r.pathway == PATHWAY_VISION
        assert r.phase_times == {}
        assert r.total_time == 0.0
        assert r.timestamp > 0

    def test_to_dict(self):
        """辞書表現の構造。"""
        r = PipelineRecord(PATHWAY_TEXT)
        r.phase_times = {PHASE_PERCEPTION_PARSE: 0.01, PHASE_PSYCHE_UPDATE: 0.05}
        r.total_time = 0.06
        d = r.to_dict()
        assert d["pathway"] == PATHWAY_TEXT
        assert PHASE_PERCEPTION_PARSE in d["phase_times"]
        assert PHASE_PSYCHE_UPDATE in d["phase_times"]
        assert d["total_time"] == 0.06
        assert "timestamp" in d


# ── パイプライン開始/完了テスト ────────────────────────────────────


class TestPipelineLifecycle:
    """パイプライン開始→完了の基本フロー。"""

    def test_basic_begin_end(self):
        """開始→完了でFIFOバッファに記録が追加される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        time.sleep(0.001)
        m.end_pipeline()
        assert m.record_count == 1
        records = m.buffer
        assert records[0]["pathway"] == PATHWAY_VISION
        assert records[0]["total_time"] > 0

    def test_end_without_begin(self):
        """begin_pipelineなしでend_pipelineを呼んでもエラーにならない。"""
        m = _make_measurement()
        m.end_pipeline()  # Should not raise
        assert m.record_count == 0

    def test_multiple_pipelines(self):
        """複数パイプライン実行が正しく記録される。"""
        m = _make_measurement()
        for pathway in [PATHWAY_VISION, PATHWAY_TEXT, PATHWAY_INTERNAL]:
            m.begin_pipeline(pathway)
            m.end_pipeline()
        assert m.record_count == 3
        pathways = [r["pathway"] for r in m.buffer]
        assert pathways == [PATHWAY_VISION, PATHWAY_TEXT, PATHWAY_INTERNAL]

    def test_current_record_cleared_after_end(self):
        """end_pipeline後に現在の記録がクリアされる。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        # 再度endを呼んでも新しい記録は追加されない
        m.end_pipeline()
        assert m.record_count == 1


# ── フェーズ区間記録テスト ────────────────────────────────────────


class TestPhaseRecording:
    """各フェーズ区間の経過時間記録。"""

    def test_record_phase(self):
        """フェーズ区間の経過時間が記録される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.123)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.045)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.067)
        m.end_pipeline()
        record = m.buffer[0]
        assert record["phase_times"][PHASE_PERCEPTION_API] == pytest.approx(0.123, abs=1e-6)
        assert record["phase_times"][PHASE_PERCEPTION_PARSE] == pytest.approx(0.045, abs=1e-6)
        assert record["phase_times"][PHASE_PSYCHE_UPDATE] == pytest.approx(0.067, abs=1e-6)

    def test_record_phase_without_begin(self):
        """begin_pipelineなしでrecord_phaseを呼んでもエラーにならない。"""
        m = _make_measurement()
        m.record_phase(PHASE_PERCEPTION_API, 0.1)  # Should not raise

    def test_all_phase_identifiers(self):
        """全フェーズ識別子が記録可能。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        all_phases = [
            PHASE_PERCEPTION_API,
            PHASE_PERCEPTION_PARSE,
            PHASE_PSYCHE_UPDATE,
            PHASE_MEMORY_RECALL,
            PHASE_POLICY_SELECT,
            PHASE_EXPRESSION_API,
        ]
        for i, phase in enumerate(all_phases):
            m.record_phase(phase, 0.01 * (i + 1))
        m.end_pipeline()
        record = m.buffer[0]
        for phase in all_phases:
            assert phase in record["phase_times"]

    def test_pipeline_total_auto_added(self):
        """end_pipeline時にPIPELINE_TOTALが自動追加される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.01)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PIPELINE_TOTAL in record["phase_times"]
        assert record["total_time"] > 0

    def test_missing_phases_not_recorded(self):
        """存在しないフェーズ区間は記録に含まれない。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_TEXT)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.01)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API not in record["phase_times"]
        assert PHASE_EXPRESSION_API not in record["phase_times"]


# ── FIFOバッファテスト ────────────────────────────────────────────


class TestFIFOBuffer:
    """FIFOバッファの上限と自然消失。"""

    def test_fifo_eviction(self):
        """上限を超えると最古の記録が自然消失する。"""
        m = _make_measurement(buffer_max=3)
        for i in range(5):
            m.begin_pipeline(f"pathway_{i}")
            m.end_pipeline()
        assert m.record_count == 3
        pathways = [r["pathway"] for r in m.buffer]
        assert pathways == ["pathway_2", "pathway_3", "pathway_4"]

    def test_buffer_returns_copy(self):
        """bufferプロパティは読み取り専用コピーを返す。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        buf1 = m.buffer
        buf2 = m.buffer
        assert buf1 is not buf2
        assert buf1 == buf2


# ── セッション累積カウンタテスト ──────────────────────────────────


class TestSessionCumulative:
    """セッション累積カウンタの更新。"""

    def test_pathway_count(self):
        """経路別のパイプライン実行回数が正しく累積される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_TEXT)
        m.end_pipeline()
        assert m.pathway_count[PATHWAY_VISION] == 2
        assert m.pathway_count[PATHWAY_TEXT] == 1

    def test_pathway_total_cumulative(self):
        """経路別のパイプライン全体累積経過時間が正しく累積される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        time.sleep(0.001)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_VISION)
        time.sleep(0.001)
        m.end_pipeline()
        assert m.pathway_total_cumulative[PATHWAY_VISION] > 0

    def test_pathway_phase_cumulative(self):
        """経路別・フェーズ別の累積経過時間が正しく累積される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.05)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.2)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.03)
        m.end_pipeline()
        cum = m.pathway_phase_cumulative[PATHWAY_VISION]
        assert cum[PHASE_PERCEPTION_API] == pytest.approx(0.3, abs=1e-6)
        assert cum[PHASE_PSYCHE_UPDATE] == pytest.approx(0.08, abs=1e-6)

    def test_cumulative_across_pathways(self):
        """異なる経路の累積は独立して保持される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_TEXT)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.05)
        m.end_pipeline()
        assert PATHWAY_VISION in m.pathway_phase_cumulative
        assert PATHWAY_TEXT in m.pathway_phase_cumulative
        assert PHASE_PERCEPTION_API in m.pathway_phase_cumulative[PATHWAY_VISION]
        assert PHASE_PERCEPTION_PARSE in m.pathway_phase_cumulative[PATHWAY_TEXT]
        assert PHASE_PERCEPTION_API not in m.pathway_phase_cumulative[PATHWAY_TEXT]

    def test_cumulative_returns_copy(self):
        """累積カウンタプロパティは読み取り専用コピーを返す。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        c1 = m.pathway_count
        c2 = m.pathway_count
        assert c1 is not c2


# ── 入力経路別テスト ──────────────────────────────────────────────


class TestPathwaySpecific:
    """入力経路ごとの特性。"""

    def test_vision_pathway_has_perception_api(self):
        """画面知覚経路ではperception_apiフェーズが記録可能。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.5)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.02)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.05)
        m.record_phase(PHASE_MEMORY_RECALL, 0.01)
        m.record_phase(PHASE_POLICY_SELECT, 0.005)
        m.record_phase(PHASE_EXPRESSION_API, 0.3)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API in record["phase_times"]

    def test_text_pathway_no_perception_api(self):
        """テキスト経路ではperception_apiフェーズが存在しない。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_TEXT)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.02)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.05)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API not in record["phase_times"]

    def test_internal_pathway_no_perception_api(self):
        """自発起動経路ではperception_apiフェーズが存在しない。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_INTERNAL)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.05)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API not in record["phase_times"]

    def test_pathways_separate_in_cumulative(self):
        """各経路の累積は完全に独立。"""
        m = _make_measurement()
        for pathway in [PATHWAY_VISION, PATHWAY_TEXT, PATHWAY_INTERNAL]:
            m.begin_pipeline(pathway)
            m.record_phase(PHASE_PSYCHE_UPDATE, 0.01)
            m.end_pipeline()
        assert len(m.pathway_count) == 3
        for pathway in [PATHWAY_VISION, PATHWAY_TEXT, PATHWAY_INTERNAL]:
            assert m.pathway_count[pathway] == 1


# ── セッションサマリテスト ────────────────────────────────────────


class TestSessionSummary:
    """セッションサマリの出力。"""

    def test_emit_pipeline_summary(self, caplog):
        """セッションサマリがログに出力される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        m.end_pipeline()
        m.begin_pipeline(PATHWAY_TEXT)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.05)
        m.end_pipeline()

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_pipeline_summary()

        # ログに pipeline_session_summary タイプが含まれる
        found = False
        for record in caplog.records:
            if "pipeline_session_summary" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["pathway_counts"][PATHWAY_VISION] == 1
                assert data["pathway_counts"][PATHWAY_TEXT] == 1
                break
        assert found, "pipeline_session_summary log not found"

    def test_emit_summary_when_disabled(self, caplog):
        """無効時にはサマリは出力されない。"""
        m = _make_measurement(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_pipeline_summary()
        for record in caplog.records:
            assert "pipeline_session_summary" not in record.getMessage()


# ── 読み取り専用アクセサテスト ────────────────────────────────────


class TestReadOnlyAccessor:
    """get_summary()の構造。"""

    def test_get_summary_empty(self):
        """空状態でget_summaryが正しい構造を返す。"""
        m = _make_measurement()
        s = m.get_summary()
        assert s["pathway_counts"] == {}
        assert s["pathway_total_cumulative"] == {}
        assert s["pathway_phase_cumulative"] == {}
        assert s["buffer_size"] == 0
        assert s["latest_record"] is None

    def test_get_summary_with_data(self):
        """データがある状態でget_summaryが正しい値を返す。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        m.end_pipeline()
        s = m.get_summary()
        assert s["pathway_counts"][PATHWAY_VISION] == 1
        assert s["buffer_size"] == 1
        assert s["latest_record"] is not None
        assert s["latest_record"]["pathway"] == PATHWAY_VISION

    def test_get_summary_returns_copy(self):
        """get_summaryは読み取り専用コピーを返す。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        s1 = m.get_summary()
        s2 = m.get_summary()
        assert s1 is not s2


# ── PhaseTimer テスト ─────────────────────────────────────────────


class TestPhaseTimer:
    """PhaseTimerコンテキストマネージャのテスト。"""

    def test_basic_timing(self):
        """基本的なwith文での計測。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        with PhaseTimer(m, PHASE_PERCEPTION_API):
            time.sleep(0.005)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API in record["phase_times"]
        assert record["phase_times"][PHASE_PERCEPTION_API] >= 0.001

    def test_exception_in_phase_propagates(self):
        """with文内で例外が発生した場合、例外が再送出される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        with pytest.raises(ValueError):
            with PhaseTimer(m, PHASE_PERCEPTION_API):
                raise ValueError("test error")
        # 計測は記録される(例外前まで)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PERCEPTION_API in record["phase_times"]

    def test_disabled_measurement_skips(self):
        """計測が無効な場合、PhaseTimerは何もしない。"""
        m = _make_measurement(enabled=False)
        m.begin_pipeline(PATHWAY_VISION)
        with PhaseTimer(m, PHASE_PERCEPTION_API):
            time.sleep(0.001)
        m.end_pipeline()
        assert m.record_count == 0

    def test_none_measurement(self):
        """measurement=Noneの場合、PhaseTimerは安全に何もしない。"""
        with PhaseTimer(None, PHASE_PERCEPTION_API):
            time.sleep(0.001)
        # No exception raised

    def test_multiple_phase_timers(self):
        """複数のPhaseTimerが独立して計測される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        with PhaseTimer(m, PHASE_PERCEPTION_API):
            time.sleep(0.002)
        with PhaseTimer(m, PHASE_PERCEPTION_PARSE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            time.sleep(0.001)
        m.end_pipeline()
        record = m.buffer[0]
        assert len(record["phase_times"]) >= 4  # 3 phases + pipeline_total

    def test_nested_phase_timers(self):
        """ネストされたPhaseTimerが独立して計測される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            with PhaseTimer(m, PHASE_MEMORY_RECALL):
                time.sleep(0.001)
        m.end_pipeline()
        record = m.buffer[0]
        assert PHASE_PSYCHE_UPDATE in record["phase_times"]
        assert PHASE_MEMORY_RECALL in record["phase_times"]


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作確認。"""

    def test_disabled_begin_pipeline_noop(self):
        """無効時にbegin_pipelineは何もしない。"""
        m = _make_measurement(enabled=False)
        m.begin_pipeline(PATHWAY_VISION)
        # _current_recordがNoneのまま(実質検証不可だが例外なし確認)

    def test_disabled_end_pipeline_noop(self):
        """無効時にend_pipelineは何もしない。"""
        m = _make_measurement(enabled=False)
        m.end_pipeline()
        assert m.record_count == 0

    def test_disabled_record_phase_noop(self):
        """無効時にrecord_phaseは何もしない。"""
        m = _make_measurement(enabled=False)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        # No exception raised

    def test_exception_in_record_phase(self):
        """record_phase内部で例外が発生しても安全に無視される。"""
        m = _make_measurement()
        # current_recordがNoneの状態でrecord_phaseを呼ぶ
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        # No exception raised

    def test_fifo_buffer_limit(self):
        """FIFOバッファの上限が正しく適用される(安全弁2)。"""
        m = _make_measurement(buffer_max=3)
        for _ in range(10):
            m.begin_pipeline(PATHWAY_VISION)
            m.end_pipeline()
        assert m.record_count == 3

    def test_log_output_on_complete(self, caplog):
        """パイプライン完了時にログが出力される(安全弁3と連携)。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.end_pipeline()
        found = False
        for record in caplog.records:
            if "pipeline_complete" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["pathway"] == PATHWAY_VISION
                break
        assert found, "pipeline_complete log not found"


# ── 永続化非対象テスト ───────────────────────────────────────────


class TestNoPersistence:
    """永続化対象外であることの確認(安全弁4)。"""

    def test_no_save_load_methods(self):
        """PipelineMeasurementにsave/loadメソッドが存在しない。"""
        m = _make_measurement()
        assert not hasattr(m, "save")
        assert not hasattr(m, "load")
        assert not hasattr(m, "to_dict")
        assert not hasattr(m, "from_dict")

    def test_session_boundary_clears_state(self):
        """新しいインスタンスは空の状態で開始する。"""
        m1 = _make_measurement()
        m1.begin_pipeline(PATHWAY_VISION)
        m1.end_pipeline()
        assert m1.record_count == 1

        m2 = _make_measurement()
        assert m2.record_count == 0
        assert m2.pathway_count == {}


# ── psyche非参照テスト ───────────────────────────────────────────


class TestPsycheIsolation:
    """psycheモジュールとの構造的分離の確認(安全弁6)。"""

    def test_no_psyche_imports(self):
        """pipeline_measurement.pyがpsycheモジュールをインポートしない。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        assert "from psyche" not in source
        assert "import psyche" not in source

    def test_no_enrichment_methods(self):
        """enrichmentに関連するメソッドが存在しない。"""
        m = _make_measurement()
        assert not hasattr(m, "get_enrichment")
        assert not hasattr(m, "enrichment")
        assert not hasattr(m, "get_prompt_enrichment")

    def test_no_orchestrator_dependency(self):
        """orchestratorへの依存がない(importレベル)。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        # コメント・docstringを除いた実際のimport文にorchestratorが含まれないことを確認
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "orchestrator" not in line.lower()


# ── ログ出力テスト ───────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のフォーマット検証。"""

    def test_pipeline_complete_log_format(self, caplog):
        """pipeline_completeログのJSON構造。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.123)
        m.record_phase(PHASE_EXPRESSION_API, 0.456)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.end_pipeline()
        for record in caplog.records:
            msg = record.getMessage()
            if "pipeline_complete" in msg:
                data = json.loads(msg)
                assert data["type"] == "pipeline_complete"
                assert "timestamp" in data
                assert data["pathway"] == PATHWAY_VISION
                assert "phase_times" in data
                assert "total_time" in data
                break

    def test_session_summary_log_format(self, caplog):
        """pipeline_session_summaryログのJSON構造。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_pipeline_summary()
        for record in caplog.records:
            msg = record.getMessage()
            if "pipeline_session_summary" in msg:
                data = json.loads(msg)
                assert data["type"] == "pipeline_session_summary"
                assert "pathway_counts" in data
                assert "pathway_total_cumulative" in data
                assert "pathway_phase_cumulative" in data
                assert "buffer_size" in data
                break


# ── フェーズ識別子定数テスト ──────────────────────────────────────


class TestPhaseConstants:
    """フェーズ区間識別子定数の存在確認。"""

    def test_all_phase_constants_defined(self):
        """設計書で定義された全フェーズ区間の識別子が定義されている。"""
        assert PHASE_PERCEPTION_API == "perception_api"
        assert PHASE_PERCEPTION_PARSE == "perception_parse"
        assert PHASE_PSYCHE_UPDATE == "psyche_update"
        assert PHASE_MEMORY_RECALL == "memory_recall"
        assert PHASE_POLICY_SELECT == "policy_select"
        assert PHASE_EXPRESSION_API == "expression_api"
        assert PHASE_PIPELINE_TOTAL == "pipeline_total"

    def test_all_pathway_constants_defined(self):
        """設計書で定義された全入力経路の識別子が定義されている。"""
        assert PATHWAY_VISION == "vision"
        assert PATHWAY_TEXT == "text"
        assert PATHWAY_INTERNAL == "internal"


# ── 統合テスト ───────────────────────────────────────────────────


class TestIntegration:
    """パイプライン計測の統合テスト。"""

    def test_full_vision_pipeline_simulation(self):
        """画面知覚パイプラインの全フェーズ計測シミュレーション。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)

        with PhaseTimer(m, PHASE_PERCEPTION_API):
            time.sleep(0.002)
        with PhaseTimer(m, PHASE_PERCEPTION_PARSE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_MEMORY_RECALL):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_POLICY_SELECT):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_EXPRESSION_API):
            time.sleep(0.002)

        m.end_pipeline()

        assert m.record_count == 1
        record = m.buffer[0]
        assert record["pathway"] == PATHWAY_VISION
        assert len(record["phase_times"]) == 7  # 6 phases + pipeline_total
        assert record["total_time"] > 0

    def test_full_text_pipeline_simulation(self):
        """テキスト対話パイプラインの計測シミュレーション。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_TEXT)

        with PhaseTimer(m, PHASE_PERCEPTION_PARSE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_MEMORY_RECALL):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_POLICY_SELECT):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_EXPRESSION_API):
            time.sleep(0.001)

        m.end_pipeline()

        assert m.record_count == 1
        record = m.buffer[0]
        assert record["pathway"] == PATHWAY_TEXT
        # text pathway: no perception_api
        assert PHASE_PERCEPTION_API not in record["phase_times"]
        assert len(record["phase_times"]) == 6  # 5 phases + pipeline_total

    def test_full_internal_pipeline_simulation(self):
        """自発起動パイプラインの計測シミュレーション。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_INTERNAL)

        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_MEMORY_RECALL):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_POLICY_SELECT):
            time.sleep(0.001)
        with PhaseTimer(m, PHASE_EXPRESSION_API):
            time.sleep(0.001)

        m.end_pipeline()

        assert m.record_count == 1
        record = m.buffer[0]
        assert record["pathway"] == PATHWAY_INTERNAL
        assert PHASE_PERCEPTION_API not in record["phase_times"]
        assert PHASE_PERCEPTION_PARSE not in record["phase_times"]

    def test_mixed_pathway_session(self):
        """混合経路のセッションで累積が正しく分離される。"""
        m = _make_measurement()

        # Vision pipeline
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.5)
        m.record_phase(PHASE_EXPRESSION_API, 0.3)
        m.end_pipeline()

        # Text pipeline
        m.begin_pipeline(PATHWAY_TEXT)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.02)
        m.record_phase(PHASE_EXPRESSION_API, 0.2)
        m.end_pipeline()

        # Internal pipeline
        m.begin_pipeline(PATHWAY_INTERNAL)
        m.record_phase(PHASE_PSYCHE_UPDATE, 0.05)
        m.end_pipeline()

        assert m.pathway_count == {
            PATHWAY_VISION: 1,
            PATHWAY_TEXT: 1,
            PATHWAY_INTERNAL: 1,
        }
        assert PATHWAY_VISION in m.pathway_phase_cumulative
        assert PATHWAY_TEXT in m.pathway_phase_cumulative
        assert PATHWAY_INTERNAL in m.pathway_phase_cumulative

    def test_session_lifecycle(self):
        """セッションのライフサイクル: 計測→サマリ出力。"""
        m = _make_measurement()

        # Multiple pipelines
        for _ in range(5):
            m.begin_pipeline(PATHWAY_VISION)
            with PhaseTimer(m, PHASE_PERCEPTION_API):
                pass  # simulate fast operation
            m.end_pipeline()

        for _ in range(3):
            m.begin_pipeline(PATHWAY_TEXT)
            with PhaseTimer(m, PHASE_PERCEPTION_PARSE):
                pass
            m.end_pipeline()

        # Summary
        summary = m.get_summary()
        assert summary["pathway_counts"][PATHWAY_VISION] == 5
        assert summary["pathway_counts"][PATHWAY_TEXT] == 3
        assert summary["buffer_size"] == 8

    def test_silence_pipeline_no_expression(self):
        """沈黙時(代弁コールなし)のパイプライン計測。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        with PhaseTimer(m, PHASE_PERCEPTION_API):
            pass
        with PhaseTimer(m, PHASE_PERCEPTION_PARSE):
            pass
        with PhaseTimer(m, PHASE_PSYCHE_UPDATE):
            pass
        with PhaseTimer(m, PHASE_MEMORY_RECALL):
            pass
        with PhaseTimer(m, PHASE_POLICY_SELECT):
            pass
        # No expression phase (silence chosen)
        m.end_pipeline()

        record = m.buffer[0]
        assert PHASE_EXPRESSION_API not in record["phase_times"]
        assert record["total_time"] > 0


# ── エッジケーステスト ───────────────────────────────────────────


class TestEdgeCases:
    """エッジケースの処理。"""

    def test_rapid_begin_end(self):
        """非常に短いパイプラインでも記録される。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.end_pipeline()
        assert m.record_count == 1
        assert m.buffer[0]["total_time"] >= 0

    def test_double_begin_overwrites(self):
        """begin_pipelineを2回呼ぶと最新が使われる。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_API, 0.1)
        m.begin_pipeline(PATHWAY_TEXT)  # overwrite
        m.end_pipeline()
        assert m.record_count == 1
        assert m.buffer[0]["pathway"] == PATHWAY_TEXT
        # Previous phases are lost
        assert PHASE_PERCEPTION_API not in m.buffer[0]["phase_times"]

    def test_custom_pathway_name(self):
        """未定義の経路名でも記録可能(拡張性)。"""
        m = _make_measurement()
        m.begin_pipeline("custom_pathway")
        m.end_pipeline()
        assert m.buffer[0]["pathway"] == "custom_pathway"

    def test_zero_elapsed_time(self):
        """0秒の経過時間が記録可能。"""
        m = _make_measurement()
        m.begin_pipeline(PATHWAY_VISION)
        m.record_phase(PHASE_PERCEPTION_PARSE, 0.0)
        m.end_pipeline()
        record = m.buffer[0]
        assert record["phase_times"][PHASE_PERCEPTION_PARSE] == 0.0

    def test_large_number_of_pipelines(self):
        """大量のパイプライン実行でメモリが制限される(安全弁2)。"""
        m = _make_measurement(buffer_max=50)
        for i in range(500):
            m.begin_pipeline(PATHWAY_VISION)
            m.record_phase(PHASE_PSYCHE_UPDATE, 0.001)
            m.end_pipeline()
        assert m.record_count == 50
        assert m.pathway_count[PATHWAY_VISION] == 500
