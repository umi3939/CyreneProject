"""
tests/test_phase_profiler.py - Phase単位実行時間プロファイリングのテスト

tools/phase_profiler.py のテスト。
設計書: design_phase_profiling.md

テスト対象:
- PhaseProfiler: Phase単位実行時間プロファイリング本体
- PhaseProfileTimer: Phase計測コンテキストマネージャ
- EnrichmentProfileTimer: enrichment生成計測コンテキストマネージャ
- PhaseExecutionEngine との統合(プロファイラ設定・計測点挿入)
"""

from __future__ import annotations

import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.phase_profiler import (
    PhaseProfiler,
    PhaseProfileTimer,
    PhaseTimingRecord,
    TickProfile,
    EnrichmentProfileTimer,
    _DEFAULT_TICK_BUFFER_MAX,
)


# ── テスト用ヘルパー ──────────────────────────────────────────────


def make_enabled_profiler(**kwargs) -> PhaseProfiler:
    """有効化されたPhaseProfilerを生成する。"""
    return PhaseProfiler(enabled=True, **kwargs)


def make_disabled_profiler(**kwargs) -> PhaseProfiler:
    """無効化されたPhaseProfilerを生成する。"""
    return PhaseProfiler(enabled=False, **kwargs)


# ── PhaseTimingRecord テスト ──────────────────────────────────────


class TestPhaseTimingRecord:
    """PhaseTimingRecordの基本テスト。"""

    def test_creation(self):
        """レコードが正しく生成されること。"""
        rec = PhaseTimingRecord(tick=1, band_name="every_tick", phase_name="phase_1", elapsed=0.001)
        assert rec.tick == 1
        assert rec.band_name == "every_tick"
        assert rec.phase_name == "phase_1"
        assert rec.elapsed == 0.001

    def test_to_dict(self):
        """辞書表現が正しいこと。"""
        rec = PhaseTimingRecord(tick=5, band_name="every_3_ticks", phase_name="phase_8", elapsed=0.0123456)
        d = rec.to_dict()
        assert d["tick"] == 5
        assert d["band_name"] == "every_3_ticks"
        assert d["phase_name"] == "phase_8"
        assert d["elapsed"] == round(0.0123456, 6)

    def test_zero_elapsed(self):
        """経過時間0のレコードが正しく生成されること。"""
        rec = PhaseTimingRecord(tick=0, band_name="b", phase_name="p", elapsed=0.0)
        assert rec.elapsed == 0.0
        assert rec.to_dict()["elapsed"] == 0.0


# ── TickProfile テスト ────────────────────────────────────────────


class TestTickProfile:
    """TickProfileの基本テスト。"""

    def test_creation(self):
        """TickProfileが正しく生成されること。"""
        tp = TickProfile(tick=10)
        assert tp.tick == 10
        assert tp.records == []
        assert tp.enrichment_elapsed is None
        assert tp.timestamp > 0

    def test_to_dict_empty(self):
        """空のTickProfileの辞書表現。"""
        tp = TickProfile(tick=1)
        d = tp.to_dict()
        assert d["tick"] == 1
        assert d["records"] == []
        assert "enrichment_elapsed" not in d

    def test_to_dict_with_records(self):
        """レコード付きTickProfileの辞書表現。"""
        tp = TickProfile(tick=2)
        tp.records.append(
            PhaseTimingRecord(tick=2, band_name="b", phase_name="p1", elapsed=0.001)
        )
        tp.records.append(
            PhaseTimingRecord(tick=2, band_name="b", phase_name="p2", elapsed=0.002)
        )
        d = tp.to_dict()
        assert len(d["records"]) == 2
        assert d["records"][0]["phase_name"] == "p1"
        assert d["records"][1]["phase_name"] == "p2"

    def test_to_dict_with_enrichment(self):
        """enrichment時間付きTickProfileの辞書表現。"""
        tp = TickProfile(tick=3)
        tp.enrichment_elapsed = 0.005
        d = tp.to_dict()
        assert d["enrichment_elapsed"] == round(0.005, 6)


# ── PhaseProfiler 基本テスト ──────────────────────────────────────


class TestPhaseProfilerBasic:
    """PhaseProfilerの基本的な有効/無効テスト。"""

    def test_explicit_enabled(self):
        """明示的に有効化されること。"""
        p = PhaseProfiler(enabled=True)
        assert p.enabled is True

    def test_explicit_disabled(self):
        """明示的に無効化されること。"""
        p = PhaseProfiler(enabled=False)
        assert p.enabled is False

    def test_env_var_enabled(self):
        """環境変数CYRENE_MONITOR=1で有効化されること。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            p = PhaseProfiler()
            assert p.enabled is True

    def test_env_var_disabled(self):
        """環境変数CYRENE_MONITOR=0で無効化されること。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            p = PhaseProfiler()
            assert p.enabled is False

    def test_env_var_unset(self):
        """環境変数未設定で無効化されること。"""
        env = os.environ.copy()
        env.pop("CYRENE_MONITOR", None)
        with patch.dict(os.environ, env, clear=True):
            p = PhaseProfiler()
            assert p.enabled is False

    def test_initial_state(self):
        """初期状態が空であること。"""
        p = make_enabled_profiler()
        assert p.total_ticks_profiled == 0
        assert p.phase_cumulative_time == {}
        assert p.phase_call_count == {}
        assert p.band_cumulative_time == {}
        assert p.enrichment_cumulative_time == 0.0
        assert p.enrichment_call_count == 0

    def test_tick_buffer_max_minimum(self):
        """FIFOバッファ上限の最低値が1であること(安全弁2)。"""
        p = PhaseProfiler(enabled=True, tick_buffer_max=0)
        # dequeのmaxlenが1以上であることを確認
        assert p._tick_buffer.maxlen >= 1

    def test_tick_buffer_max_negative(self):
        """負の値でもFIFOバッファ上限が1以上になること(安全弁2)。"""
        p = PhaseProfiler(enabled=True, tick_buffer_max=-10)
        assert p._tick_buffer.maxlen >= 1

    def test_default_tick_buffer_max(self):
        """デフォルトのFIFOバッファ上限。"""
        p = make_enabled_profiler()
        assert p._tick_buffer.maxlen == _DEFAULT_TICK_BUFFER_MAX


# ── PhaseProfiler ティック計測テスト ──────────────────────────────


class TestPhaseProfilerTick:
    """ティック開始/完了の計測テスト。"""

    def test_begin_end_tick(self):
        """ティックの開始と完了が正しく動作すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.end_tick()
        assert p.total_ticks_profiled == 1
        assert len(p._tick_buffer) == 1

    def test_multiple_ticks(self):
        """複数ティックの計測が正しく蓄積されること。"""
        p = make_enabled_profiler()
        for i in range(5):
            p.begin_tick(i)
            p.end_tick()
        assert p.total_ticks_profiled == 5
        assert len(p._tick_buffer) == 5

    def test_fifo_eviction(self):
        """FIFO上限到達時に最古の記録が消失すること(安全弁2)。"""
        p = PhaseProfiler(enabled=True, tick_buffer_max=3)
        for i in range(5):
            p.begin_tick(i)
            p.end_tick()
        assert p.total_ticks_profiled == 5
        assert len(p._tick_buffer) == 3
        # 最古のtick=0, tick=1が消失し、tick=2, 3, 4が残る
        ticks = [tp.tick for tp in p._tick_buffer]
        assert ticks == [2, 3, 4]

    def test_begin_tick_flushes_previous(self):
        """begin_tickが前回の未完了プロファイルをFIFOに移動すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        # end_tickを呼ばずにbegin_tick
        p.begin_tick(2)
        # tick=1がFIFOに移動している
        assert len(p._tick_buffer) == 1
        assert p._tick_buffer[0].tick == 1

    def test_disabled_begin_end_tick(self):
        """無効化時にティック計測が行われないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)
        p.end_tick()
        assert p.total_ticks_profiled == 0
        assert len(p._tick_buffer) == 0

    def test_end_tick_without_begin(self):
        """begin_tickなしのend_tickが安全にスキップされること。"""
        p = make_enabled_profiler()
        p.end_tick()  # no-op
        assert p.total_ticks_profiled == 0


# ── PhaseProfiler Phase記録テスト ────────────────────────────────


class TestPhaseProfilerRecordPhase:
    """Phase実行時間の記録テスト。"""

    def test_record_phase(self):
        """Phase実行時間が正しく記録されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        p.record_phase("every_tick", "phase_2", 0.002, 1)
        p.end_tick()

        assert p.phase_cumulative_time["phase_1"] == 0.001
        assert p.phase_cumulative_time["phase_2"] == 0.002
        assert p.phase_call_count["phase_1"] == 1
        assert p.phase_call_count["phase_2"] == 1
        assert p.band_cumulative_time["every_tick"] == pytest.approx(0.003)

    def test_record_phase_cumulative(self):
        """複数ティックにまたがるPhase記録が累積されること。"""
        p = make_enabled_profiler()
        for i in range(3):
            p.begin_tick(i)
            p.record_phase("every_tick", "phase_1", 0.001, i)
            p.end_tick()

        assert p.phase_cumulative_time["phase_1"] == pytest.approx(0.003)
        assert p.phase_call_count["phase_1"] == 3
        assert p.band_cumulative_time["every_tick"] == pytest.approx(0.003)

    def test_record_phase_multiple_bands(self):
        """異なる帯域のPhase記録が独立して累積されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        p.record_phase("every_3_ticks", "phase_8", 0.002, 1)
        p.end_tick()

        assert p.band_cumulative_time["every_tick"] == 0.001
        assert p.band_cumulative_time["every_3_ticks"] == 0.002

    def test_record_phase_disabled(self):
        """無効化時にPhase記録が行われないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        p.end_tick()
        assert p.phase_cumulative_time == {}

    def test_record_phase_in_current_profile(self):
        """Phase記録が現在のTickProfileに追加されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        # end_tick前にプロファイルのrecordsを確認
        assert len(p._current_tick_profile.records) == 1
        assert p._current_tick_profile.records[0].phase_name == "phase_1"

    def test_record_phase_without_begin_tick(self):
        """begin_tickなしのrecord_phaseが累積カウンタには記録されること。"""
        p = make_enabled_profiler()
        # begin_tickを呼ばずに直接record_phase
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        # 累積カウンタには記録される(current_tick_profileはNoneなので追加されない)
        assert p.phase_cumulative_time["phase_1"] == 0.001
        assert p.phase_call_count["phase_1"] == 1


# ── PhaseProfiler enrichment記録テスト ───────────────────────────


class TestPhaseProfilerRecordEnrichment:
    """enrichment生成時間の記録テスト。"""

    def test_record_enrichment(self):
        """enrichment生成時間が正しく記録されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_enrichment(0.005)
        p.end_tick()

        assert p.enrichment_cumulative_time == 0.005
        assert p.enrichment_call_count == 1

    def test_record_enrichment_cumulative(self):
        """enrichment記録が累積されること。"""
        p = make_enabled_profiler()
        for i in range(3):
            p.begin_tick(i)
            p.record_enrichment(0.01)
            p.end_tick()

        assert p.enrichment_cumulative_time == pytest.approx(0.03)
        assert p.enrichment_call_count == 3

    def test_record_enrichment_in_tick_profile(self):
        """enrichment記録がTickProfileに記録されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_enrichment(0.005)
        assert p._current_tick_profile.enrichment_elapsed == 0.005

    def test_record_enrichment_disabled(self):
        """無効化時にenrichment記録が行われないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)
        p.record_enrichment(0.005)
        p.end_tick()
        assert p.enrichment_cumulative_time == 0.0
        assert p.enrichment_call_count == 0


# ── PhaseProfiler サマリ・アクセサテスト ──────────────────────────


class TestPhaseProfilerSummary:
    """サマリと読み取り専用アクセサのテスト。"""

    def test_get_summary_empty(self):
        """空の状態でのサマリが正しいこと。"""
        p = make_enabled_profiler()
        summary = p.get_summary()
        assert summary["total_ticks_profiled"] == 0
        assert summary["phase_cumulative_time"] == {}
        assert summary["phase_call_count"] == {}
        assert summary["phase_avg_time"] == {}
        assert summary["band_cumulative_time"] == {}
        assert summary["enrichment_cumulative_time"] == 0.0
        assert summary["enrichment_call_count"] == 0
        assert summary["enrichment_avg_time"] == 0.0
        assert summary["tick_buffer_size"] == 0
        assert summary["latest_tick_profile"] is None

    def test_get_summary_with_data(self):
        """データ蓄積後のサマリが正しいこと。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.002, 1)
        p.record_phase("every_tick", "phase_2", 0.003, 1)
        p.record_enrichment(0.01)
        p.end_tick()

        p.begin_tick(2)
        p.record_phase("every_tick", "phase_1", 0.004, 2)
        p.record_enrichment(0.02)
        p.end_tick()

        summary = p.get_summary()
        assert summary["total_ticks_profiled"] == 2
        assert summary["phase_avg_time"]["phase_1"] == pytest.approx(0.003, abs=1e-6)
        assert summary["phase_avg_time"]["phase_2"] == pytest.approx(0.003, abs=1e-6)
        assert summary["enrichment_avg_time"] == pytest.approx(0.015, abs=1e-6)
        assert summary["tick_buffer_size"] == 2

    def test_get_latest_tick_profile_empty(self):
        """空の状態でlatest_tick_profileがNoneであること。"""
        p = make_enabled_profiler()
        assert p.get_latest_tick_profile() is None

    def test_get_latest_tick_profile_with_data(self):
        """データ蓄積後のlatest_tick_profileが正しいこと。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("b", "p1", 0.001, 1)
        p.end_tick()
        p.begin_tick(2)
        p.record_phase("b", "p2", 0.002, 2)
        p.end_tick()

        latest = p.get_latest_tick_profile()
        assert latest is not None
        assert latest["tick"] == 2
        assert len(latest["records"]) == 1
        assert latest["records"][0]["phase_name"] == "p2"

    def test_read_only_copies(self):
        """読み取り専用プロパティが元データと独立していること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("b", "p1", 0.001, 1)
        p.end_tick()

        # 返却された辞書を変更しても内部状態に影響しない
        cum = p.phase_cumulative_time
        cum["p1"] = 999.0
        assert p.phase_cumulative_time["p1"] == 0.001

        count = p.phase_call_count
        count["p1"] = 999
        assert p.phase_call_count["p1"] == 1


# ── PhaseProfiler セッションサマリ出力テスト ──────────────────────


class TestPhaseProfilerSessionSummary:
    """セッションサマリ出力のテスト。"""

    def test_emit_session_summary_enabled(self):
        """有効時にセッションサマリがログ出力されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("b", "p1", 0.001, 1)
        p.end_tick()

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.emit_session_summary()
            assert mock_debug.called
            # JSON形式の出力を検証
            call_args = mock_debug.call_args[0][0]
            data = json.loads(call_args)
            assert data["type"] == "phase_profiling_session_summary"
            assert data["total_ticks_profiled"] == 1

    def test_emit_session_summary_disabled(self):
        """無効時にセッションサマリが出力されないこと。"""
        p = make_disabled_profiler()
        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.emit_session_summary()
            assert not mock_debug.called


# ── PhaseProfiler ティックプロファイル出力テスト ──────────────────


class TestPhaseProfilerTickLog:
    """ティックプロファイルのログ出力テスト。"""

    def test_tick_profile_logged(self):
        """ティック完了時にプロファイルがログ出力されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        p.record_phase("every_3_ticks", "phase_8", 0.002, 1)
        p.record_enrichment(0.005)

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.end_tick()
            assert mock_debug.called
            call_args = mock_debug.call_args[0][0]
            data = json.loads(call_args)
            assert data["type"] == "phase_profiling_tick"
            assert data["tick"] == 1
            assert "every_tick" in data["band_breakdown"]
            assert "every_3_ticks" in data["band_breakdown"]
            assert data["enrichment_elapsed"] == round(0.005, 6)

    def test_tick_profile_band_totals(self):
        """帯域別合計が正しく計算されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "p1", 0.001, 1)
        p.record_phase("every_tick", "p2", 0.002, 1)

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.end_tick()
            call_args = mock_debug.call_args[0][0]
            data = json.loads(call_args)
            assert data["band_totals"]["every_tick"] == pytest.approx(0.003, abs=1e-6)


# ── PhaseProfileTimer テスト ──────────────────────────────────────


class TestPhaseProfileTimer:
    """PhaseProfileTimerコンテキストマネージャのテスト。"""

    def test_basic_timing(self):
        """基本的な計測が動作すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with PhaseProfileTimer(p, "every_tick", "phase_1", 1):
            time.sleep(0.001)

        assert p.phase_cumulative_time.get("phase_1", 0.0) > 0
        assert p.phase_call_count.get("phase_1", 0) == 1

    def test_disabled_no_timing(self):
        """無効化時に計測が行われないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)

        with PhaseProfileTimer(p, "every_tick", "phase_1", 1):
            pass

        assert p.phase_cumulative_time == {}

    def test_none_profiler(self):
        """プロファイラがNoneの場合に安全にスキップされること。"""
        with PhaseProfileTimer(None, "b", "p", 1):
            pass
        # 例外が発生しないことを確認

    def test_exception_in_handler_propagates(self):
        """ハンドラ内の例外が伝播すること(計測は中断しない)。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with pytest.raises(ValueError):
            with PhaseProfileTimer(p, "b", "p", 1):
                raise ValueError("test error")

        # 例外が発生しても計測は記録される
        assert p.phase_cumulative_time.get("p", 0.0) > 0

    def test_multiple_phases_in_sequence(self):
        """複数Phase連続計測が正しく動作すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with PhaseProfileTimer(p, "b", "p1", 1):
            time.sleep(0.001)
        with PhaseProfileTimer(p, "b", "p2", 1):
            time.sleep(0.001)

        assert p.phase_call_count.get("p1") == 1
        assert p.phase_call_count.get("p2") == 1
        assert p.phase_cumulative_time.get("p1", 0) > 0
        assert p.phase_cumulative_time.get("p2", 0) > 0


# ── EnrichmentProfileTimer テスト ─────────────────────────────────


class TestEnrichmentProfileTimer:
    """EnrichmentProfileTimerコンテキストマネージャのテスト。"""

    def test_basic_timing(self):
        """基本的なenrichment計測が動作すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with EnrichmentProfileTimer(p):
            time.sleep(0.001)

        assert p.enrichment_cumulative_time > 0
        assert p.enrichment_call_count == 1

    def test_disabled_no_timing(self):
        """無効化時に計測が行われないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)

        with EnrichmentProfileTimer(p):
            pass

        assert p.enrichment_cumulative_time == 0.0
        assert p.enrichment_call_count == 0

    def test_none_profiler(self):
        """プロファイラがNoneの場合に安全にスキップされること。"""
        with EnrichmentProfileTimer(None):
            pass

    def test_exception_propagates(self):
        """ブロック内の例外が伝播すること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with pytest.raises(RuntimeError):
            with EnrichmentProfileTimer(p):
                raise RuntimeError("test")

        # 例外が発生しても計測は記録される
        assert p.enrichment_cumulative_time > 0

    def test_enrichment_in_tick_profile(self):
        """enrichment計測がTickProfileに記録されること。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        with EnrichmentProfileTimer(p):
            time.sleep(0.001)

        p.end_tick()
        latest = p.get_latest_tick_profile()
        assert latest is not None
        assert latest.get("enrichment_elapsed", 0) > 0


# ── PhaseExecutionEngine 統合テスト ───────────────────────────────


class TestPhaseExecutionEngineIntegration:
    """PhaseExecutionEngineとPhaseProfilerの統合テスト。"""

    def test_set_profiler(self):
        """プロファイラの設定が正しく動作すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        engine = PhaseExecutionEngine()
        profiler = make_enabled_profiler()
        engine.set_profiler(profiler)
        assert engine.profiler is profiler

    def test_set_profiler_none(self):
        """プロファイラのNone設定が正しく動作すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        engine = PhaseExecutionEngine()
        engine.set_profiler(None)
        assert engine.profiler is None

    def test_execute_band_with_profiler(self):
        """プロファイラ付きでの帯域実行がPhase時間を記録すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        from psyche.phase_declaration import Band

        engine = PhaseExecutionEngine()
        profiler = make_enabled_profiler()
        engine.set_profiler(profiler)

        # 10ティック帯域のPhaseを登録
        phase_ids = engine._band_phase_order[Band.EVERY_10_TICKS]
        for pid in phase_ids:
            engine.register_handler(pid, lambda orch, uid: None)

        # ダミーオーケストレータ
        mock_orch = MagicMock()
        mock_orch._tick_count = 10

        profiler.begin_tick(10)
        engine.execute_band(mock_orch, "test_user")
        profiler.end_tick()

        # 各Phaseの計測が記録されていること
        for pid in phase_ids:
            assert pid in profiler.phase_call_count
            assert profiler.phase_call_count[pid] == 1
            assert profiler.phase_cumulative_time.get(pid, 0.0) >= 0.0

    def test_execute_band_without_profiler(self):
        """プロファイラなしでの帯域実行が通常通り動作すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        from psyche.phase_declaration import Band

        engine = PhaseExecutionEngine()
        # profilerを設定しない

        phase_ids = engine._band_phase_order[Band.EVERY_10_TICKS]
        call_log = []
        for pid in phase_ids:
            engine.register_handler(
                pid, lambda orch, uid, p=pid: call_log.append(p)
            )

        mock_orch = MagicMock()
        mock_orch._tick_count = 10

        log = engine.execute_band(mock_orch, "test_user")
        # 全Phaseが実行されたこと
        assert len(call_log) == len(phase_ids)

    def test_execute_band_profiler_disabled(self):
        """プロファイラが無効化されている場合、計測せず通常実行すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        from psyche.phase_declaration import Band

        engine = PhaseExecutionEngine()
        profiler = make_disabled_profiler()
        engine.set_profiler(profiler)

        phase_ids = engine._band_phase_order[Band.EVERY_10_TICKS]
        call_log = []
        for pid in phase_ids:
            engine.register_handler(
                pid, lambda orch, uid, p=pid: call_log.append(p)
            )

        mock_orch = MagicMock()
        mock_orch._tick_count = 10

        engine.execute_band(mock_orch, "test_user")
        # 全Phaseが実行されたこと
        assert len(call_log) == len(phase_ids)
        # 計測は行われていないこと
        assert profiler.phase_cumulative_time == {}

    def test_execute_band_error_absorbed_with_profiler(self):
        """エラー吸収Phaseでの例外がプロファイラに影響しないこと。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        from psyche.phase_declaration import Band, PHASE_BY_ID

        engine = PhaseExecutionEngine()
        profiler = make_enabled_profiler()
        engine.set_profiler(profiler)

        phase_ids = engine._band_phase_order[Band.EVERY_10_TICKS]
        # 最初のPhaseで例外を発生させる
        first_pid = phase_ids[0]

        def failing_handler(orch, uid):
            raise RuntimeError("test error")

        engine.register_handler(first_pid, failing_handler)
        # 残りは正常ハンドラ
        for pid in phase_ids[1:]:
            engine.register_handler(pid, lambda orch, uid: None)

        mock_orch = MagicMock()
        mock_orch._tick_count = 10

        profiler.begin_tick(10)
        log = engine.execute_band(mock_orch, "test_user")
        profiler.end_tick()

        # エラー吸収Phaseの場合、FAILEDとして記録される
        phase_def = PHASE_BY_ID[first_pid]
        if phase_def.error_absorbed:
            assert log.phase_results[first_pid] == "failed"
        # 後続Phaseは実行されること
        for pid in phase_ids[1:]:
            assert log.phase_results[pid] == "success"

    def test_profiler_records_correct_band_name(self):
        """プロファイラが正しい帯域名で記録すること。"""
        from psyche.phase_execution_engine import PhaseExecutionEngine
        from psyche.phase_declaration import Band

        engine = PhaseExecutionEngine()
        profiler = make_enabled_profiler()
        engine.set_profiler(profiler)

        # 毎ティック帯域のPhaseを登録
        phase_ids = engine._band_phase_order[Band.EVERY_TICK]
        for pid in phase_ids:
            engine.register_handler(pid, lambda orch, uid: None)

        mock_orch = MagicMock()
        mock_orch._tick_count = 1

        profiler.begin_tick(1)
        engine.execute_band(mock_orch, "test", band=Band.EVERY_TICK)
        profiler.end_tick()

        # 帯域名が"every_tick"であること
        assert "every_tick" in profiler.band_cumulative_time


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作テスト。"""

    def test_safety_valve_1_record_phase_exception(self):
        """record_phase内の例外が安全に無視されること(安全弁1)。"""
        p = make_enabled_profiler()
        # 内部状態を壊して例外を誘発
        p._phase_cumulative_time = None  # type: ignore
        # 例外が外に漏れないこと
        p.record_phase("b", "p", 0.001, 1)

    def test_safety_valve_1_record_enrichment_exception(self):
        """record_enrichment内の例外が安全に無視されること(安全弁1)。"""
        p = make_enabled_profiler()
        p._enrichment_cumulative_time = None  # type: ignore
        p.record_enrichment(0.001)

    def test_safety_valve_1_begin_tick_exception(self):
        """begin_tick内の例外が安全に無視されること(安全弁1)。"""
        p = make_enabled_profiler()
        p._tick_buffer = None  # type: ignore
        p.begin_tick(1)  # 例外が外に漏れないこと

    def test_safety_valve_1_end_tick_exception(self):
        """end_tick内の例外が安全に無視されること(安全弁1)。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p._tick_buffer = None  # type: ignore
        p.end_tick()  # 例外が外に漏れないこと

    def test_safety_valve_1_emit_session_summary_exception(self):
        """emit_session_summary内の例外が安全に無視されること(安全弁1)。"""
        p = make_enabled_profiler()
        p._phase_cumulative_time = None  # type: ignore
        p.emit_session_summary()  # 例外が外に漏れないこと

    def test_safety_valve_1_get_summary_exception(self):
        """get_summary内の例外が安全に無視されること。"""
        p = make_enabled_profiler()
        p._phase_cumulative_time = None  # type: ignore
        summary = p.get_summary()
        assert summary == {}

    def test_safety_valve_2_fifo_upper_bound(self):
        """FIFOバッファが上限を超えないこと(安全弁2)。"""
        max_size = 5
        p = PhaseProfiler(enabled=True, tick_buffer_max=max_size)
        for i in range(20):
            p.begin_tick(i)
            p.end_tick()
        assert len(p._tick_buffer) == max_size

    def test_safety_valve_3_env_var_complete_disable(self):
        """環境変数で完全無効化されること(安全弁3)。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            p = PhaseProfiler()
            p.begin_tick(1)
            p.record_phase("b", "p", 0.001, 1)
            p.record_enrichment(0.001)
            p.end_tick()
            assert p.total_ticks_profiled == 0
            assert p.phase_cumulative_time == {}
            assert p.enrichment_cumulative_time == 0.0

    def test_safety_valve_4_no_persistence(self):
        """save/loadフィールドが存在しないこと(安全弁4)。"""
        p = make_enabled_profiler()
        # to_dict/from_dictメソッドが存在しないことを確認
        assert not hasattr(p, 'to_dict')
        assert not hasattr(p, 'from_dict')
        assert not hasattr(p, 'save')
        assert not hasattr(p, 'load')

    def test_safety_valve_6_disabled_no_time_call(self):
        """無効化時にtime.perf_counterが呼ばれないこと(安全弁6)。"""
        p = make_disabled_profiler()
        with patch('time.perf_counter') as mock_perf:
            with PhaseProfileTimer(p, "b", "p", 1):
                pass
            mock_perf.assert_not_called()

    def test_safety_valve_6_none_profiler_no_time_call(self):
        """プロファイラNone時にtime.perf_counterが呼ばれないこと(安全弁6)。"""
        with patch('time.perf_counter') as mock_perf:
            with PhaseProfileTimer(None, "b", "p", 1):
                pass
            mock_perf.assert_not_called()


# ── ログ出力フォーマットテスト ────────────────────────────────────


class TestLogFormat:
    """ログ出力のJSON形式テスト。"""

    def test_tick_profile_json_format(self):
        """ティックプロファイルのJSON出力形式が正しいこと。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("every_tick", "phase_1", 0.001, 1)
        p.record_phase("every_tick", "phase_2", 0.002, 1)
        p.record_phase("every_3_ticks", "phase_8", 0.003, 1)
        p.record_enrichment(0.005)

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.end_tick()

            call_args = mock_debug.call_args[0][0]
            data = json.loads(call_args)

            assert data["type"] == "phase_profiling_tick"
            assert data["tick"] == 1
            assert "band_breakdown" in data
            assert "band_totals" in data
            assert "enrichment_elapsed" in data

            # 帯域別内訳
            assert len(data["band_breakdown"]["every_tick"]) == 2
            assert len(data["band_breakdown"]["every_3_ticks"]) == 1

    def test_session_summary_json_format(self):
        """セッションサマリのJSON出力形式が正しいこと。"""
        p = make_enabled_profiler()
        p.begin_tick(1)
        p.record_phase("b", "p1", 0.001, 1)
        p.record_enrichment(0.01)
        p.end_tick()

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.emit_session_summary()

            call_args = mock_debug.call_args[0][0]
            data = json.loads(call_args)

            assert data["type"] == "phase_profiling_session_summary"
            assert "total_ticks_profiled" in data
            assert "phase_cumulative_time" in data
            assert "phase_call_count" in data
            assert "phase_avg_time" in data
            assert "band_cumulative_time" in data
            assert "enrichment_cumulative_time" in data
            assert "enrichment_call_count" in data
            assert "enrichment_avg_time" in data

    def test_disabled_no_log_output(self):
        """無効化時にログが出力されないこと。"""
        p = make_disabled_profiler()
        p.begin_tick(1)
        p.record_phase("b", "p", 0.001, 1)
        p.end_tick()

        with patch.object(logging.getLogger("cyrene.monitor"), "debug") as mock_debug:
            p.emit_session_summary()
            mock_debug.assert_not_called()


# ── 複合シナリオテスト ────────────────────────────────────────────


class TestComplexScenarios:
    """複合的な使用シナリオのテスト。"""

    def test_full_tick_scenario(self):
        """完全なティックシナリオ: 4帯域 + enrichment。"""
        p = make_enabled_profiler()
        p.begin_tick(10)

        # 毎ティック帯域
        p.record_phase("every_tick", "phase_1", 0.001, 10)
        p.record_phase("every_tick", "phase_2", 0.002, 10)
        p.record_phase("every_tick", "phase_3", 0.001, 10)

        # 3ティック帯域
        p.record_phase("every_3_ticks", "phase_8", 0.003, 10)
        p.record_phase("every_3_ticks", "phase_9", 0.002, 10)

        # 5ティック帯域
        p.record_phase("every_5_ticks", "phase_15", 0.004, 10)

        # 10ティック帯域
        p.record_phase("every_10_ticks", "phase_27", 0.005, 10)

        # enrichment生成
        p.record_enrichment(0.008)

        p.end_tick()

        # 検証
        assert p.total_ticks_profiled == 1
        assert len(p.phase_cumulative_time) == 7
        assert len(p.band_cumulative_time) == 4
        assert p.enrichment_call_count == 1

        summary = p.get_summary()
        assert summary["total_ticks_profiled"] == 1
        assert summary["tick_buffer_size"] == 1

    def test_long_running_session(self):
        """長時間セッション: 50ティック。"""
        p = PhaseProfiler(enabled=True, tick_buffer_max=20)

        for i in range(50):
            p.begin_tick(i)
            p.record_phase("every_tick", "p1", 0.001, i)
            p.record_phase("every_tick", "p2", 0.002, i)
            if i % 3 == 0:
                p.record_phase("every_3_ticks", "p8", 0.003, i)
            if i % 5 == 0:
                p.record_phase("every_5_ticks", "p15", 0.004, i)
                p.record_enrichment(0.01)
            if i % 10 == 0:
                p.record_phase("every_10_ticks", "p27", 0.005, i)
            p.end_tick()

        assert p.total_ticks_profiled == 50
        assert len(p._tick_buffer) == 20  # FIFOに20件のみ
        assert p.phase_call_count["p1"] == 50
        assert p.phase_call_count["p2"] == 50
        assert p.phase_call_count["p8"] == 17  # 0,3,6,...,48
        assert p.phase_call_count["p15"] == 10  # 0,5,10,...,45
        assert p.enrichment_call_count == 10
        assert p.phase_call_count["p27"] == 5  # 0,10,20,30,40

    def test_interleaved_timer_and_direct_record(self):
        """コンテキストマネージャとdirect recordの混合使用。"""
        p = make_enabled_profiler()
        p.begin_tick(1)

        # コンテキストマネージャで計測
        with PhaseProfileTimer(p, "every_tick", "p1", 1):
            time.sleep(0.001)

        # 直接記録
        p.record_phase("every_tick", "p2", 0.005, 1)

        # enrichmentコンテキストマネージャ
        with EnrichmentProfileTimer(p):
            time.sleep(0.001)

        p.end_tick()

        assert p.phase_call_count["p1"] == 1
        assert p.phase_call_count["p2"] == 1
        assert p.enrichment_call_count == 1
        latest = p.get_latest_tick_profile()
        assert latest is not None
        assert len(latest["records"]) == 2

    def test_session_summary_after_multiple_ticks(self):
        """複数ティック後のセッションサマリが正しいこと。"""
        p = make_enabled_profiler()
        for i in range(10):
            p.begin_tick(i)
            p.record_phase("b", "p1", 0.001 * (i + 1), i)
            p.record_enrichment(0.002)
            p.end_tick()

        summary = p.get_summary()
        assert summary["total_ticks_profiled"] == 10
        # p1の合計: 0.001 + 0.002 + ... + 0.010 = 0.055
        assert summary["phase_cumulative_time"]["p1"] == pytest.approx(0.055, abs=1e-6)
        assert summary["phase_avg_time"]["p1"] == pytest.approx(0.0055, abs=1e-6)
        assert summary["enrichment_cumulative_time"] == pytest.approx(0.02, abs=1e-6)
        assert summary["enrichment_avg_time"] == pytest.approx(0.002, abs=1e-6)
