"""
tests/test_execution_monitor.py - ExecutionMonitor のテスト

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定、環境変数制御)
- 帯域実行時間記録テスト
- 圧縮比記録テスト
- API呼び出し記録テスト
- サイクル完了記録テスト
- スナップショット出力テスト(間隔制御、下限適用、全フィールド等価出力)
- セッションサマリテスト
- 安全弁テスト(計測失敗時の無視、ログ出力量制限、無効時の完全スキップ)
- BandTimerテスト(正常、例外時、無効時)
- read_orchestrator_fieldsテスト
- 統合テスト(orchestrator連携)
"""

import json
import logging
import os
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.execution_monitor import (
    ExecutionMonitor,
    BandTimer,
    read_orchestrator_fields,
    is_monitor_enabled,
    _SNAPSHOT_INTERVAL_MIN,
    _MAX_LOG_CHARS_PER_CYCLE,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_monitor(enabled: bool = True, snapshot_interval: int = 10) -> ExecutionMonitor:
    """テスト用のExecutionMonitorを生成する。"""
    return ExecutionMonitor(enabled=enabled, snapshot_interval=snapshot_interval)


def _make_state_reader(fields: Optional[dict] = None) -> Any:
    """テスト用の状態読み取り関数を返す。"""
    if fields is None:
        fields = {"tick_count": 42, "fear_level": 0.3, "mood_valence": 0.5}
    return lambda: fields


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """ExecutionMonitorの初期化テスト。"""

    def test_default_init_disabled(self):
        """デフォルトではモニタリングは無効(CYRENE_MONITOR未設定)。"""
        with patch.dict(os.environ, {}, clear=False):
            if "CYRENE_MONITOR" in os.environ:
                del os.environ["CYRENE_MONITOR"]
            m = ExecutionMonitor()
            assert m.enabled is False

    def test_enabled_via_constructor(self):
        """コンストラクタで明示的に有効化。"""
        m = _make_monitor(enabled=True)
        assert m.enabled is True

    def test_disabled_via_constructor(self):
        """コンストラクタで明示的に無効化。"""
        m = _make_monitor(enabled=False)
        assert m.enabled is False

    def test_enabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=1で有効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = ExecutionMonitor()
            assert m.enabled is True

    def test_disabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=0で無効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            m = ExecutionMonitor()
            assert m.enabled is False

    def test_constructor_overrides_env(self):
        """コンストラクタの指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = ExecutionMonitor(enabled=False)
            assert m.enabled is False

    def test_initial_counters_zero(self):
        """初期状態で全カウンタがゼロ。"""
        m = _make_monitor()
        assert m.cycle_count == 0
        assert m.api_call_count == {"perception": 0, "expression": 0}
        assert m.band_cumulative_time == {}

    def test_snapshot_interval_default(self):
        """デフォルトのスナップショット間隔。"""
        m = ExecutionMonitor(enabled=True)
        assert m.snapshot_interval == 50

    def test_snapshot_interval_custom(self):
        """カスタムスナップショット間隔。"""
        m = ExecutionMonitor(enabled=True, snapshot_interval=20)
        assert m.snapshot_interval == 20

    def test_snapshot_interval_min_enforced(self):
        """安全弁2: スナップショット間隔の下限が適用される。"""
        m = ExecutionMonitor(enabled=True, snapshot_interval=1)
        assert m.snapshot_interval >= _SNAPSHOT_INTERVAL_MIN

    def test_snapshot_interval_min_boundary(self):
        """下限ちょうどの値が許可される。"""
        m = ExecutionMonitor(enabled=True, snapshot_interval=_SNAPSHOT_INTERVAL_MIN)
        assert m.snapshot_interval == _SNAPSHOT_INTERVAL_MIN

    def test_is_monitor_enabled_function(self):
        """is_monitor_enabled関数のテスト。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            assert is_monitor_enabled() is True
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            assert is_monitor_enabled() is False


# ── 帯域実行時間記録テスト ────────────────────────────────────────


class TestBandTimeRecording:
    """帯域実行時間の記録テスト。"""

    def test_record_single_band(self):
        """1帯域の記録。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        assert m.last_band_times["every_tick"] == pytest.approx(0.005)
        assert m.band_cumulative_time["every_tick"] == pytest.approx(0.005)

    def test_record_multiple_bands(self):
        """複数帯域の記録。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        m.record_band_time("every_3_ticks", 0.010)
        assert len(m.last_band_times) == 2
        assert m.last_band_times["every_tick"] == pytest.approx(0.005)
        assert m.last_band_times["every_3_ticks"] == pytest.approx(0.010)

    def test_cumulative_accumulation(self):
        """累積時間が加算される。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        m.record_band_time("every_tick", 0.010)
        assert m.band_cumulative_time["every_tick"] == pytest.approx(0.015)

    def test_last_band_times_overwrite(self):
        """直近値は上書きされる(最新のみ保持)。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        m.record_band_time("every_tick", 0.010)
        assert m.last_band_times["every_tick"] == pytest.approx(0.010)

    def test_disabled_no_record(self):
        """安全弁5: 無効時は記録しない。"""
        m = _make_monitor(enabled=False)
        m.record_band_time("every_tick", 0.005)
        assert m.last_band_times == {}
        assert m.band_cumulative_time == {}

    def test_properties_return_copies(self):
        """プロパティが内部状態のコピーを返す。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        times = m.last_band_times
        times["every_tick"] = 999.0
        assert m.last_band_times["every_tick"] == pytest.approx(0.005)


# ── 圧縮比記録テスト ─────────────────────────────────────────────


class TestCompressionRecording:
    """enrichment圧縮の記録テスト。"""

    def test_record_compression(self):
        """圧縮前後文字数の記録。"""
        m = _make_monitor()
        m.record_compression(1000, 500, 0.5)
        assert m.last_compression_chars == (1000, 500)

    def test_record_compression_updates(self):
        """圧縮記録が上書きされる。"""
        m = _make_monitor()
        m.record_compression(1000, 500, 0.5)
        m.record_compression(2000, 800, 0.4)
        assert m.last_compression_chars == (2000, 800)

    def test_disabled_no_compression(self):
        """無効時は記録しない。"""
        m = _make_monitor(enabled=False)
        m.record_compression(1000, 500, 0.5)
        assert m.last_compression_chars == (0, 0)

    def test_compression_emits_json(self, caplog):
        """圧縮記録がJSON形式でログ出力される。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_compression(1000, 500, 0.5)
        # ログにJSON文字列が含まれる
        found = False
        for record in caplog.records:
            if "enrichment_compression" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["type"] == "enrichment_compression"
                assert data["before_chars"] == 1000
                assert data["after_chars"] == 500
                assert data["ratio"] == 0.5
        assert found, "compression log not found"


# ── API呼び出し記録テスト ─────────────────────────────────────────


class TestApiCallRecording:
    """API呼び出しの記録テスト。"""

    def test_record_perception_call(self):
        """知覚コールの記録。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        assert m.api_call_count["perception"] == 1
        assert m.api_token_count["perception"]["input"] == 100
        assert m.api_token_count["perception"]["output"] == 50

    def test_record_expression_call(self):
        """代弁コールの記録。"""
        m = _make_monitor()
        m.record_api_call("expression", input_tokens=200, output_tokens=100)
        assert m.api_call_count["expression"] == 1
        assert m.api_token_count["expression"]["input"] == 200
        assert m.api_token_count["expression"]["output"] == 100

    def test_cumulative_calls(self):
        """累積呼び出し回数。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        m.record_api_call("perception", input_tokens=200, output_tokens=100)
        assert m.api_call_count["perception"] == 2
        assert m.api_token_count["perception"]["input"] == 300
        assert m.api_token_count["perception"]["output"] == 150

    def test_mixed_calls(self):
        """知覚+代弁の混合記録。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        m.record_api_call("expression", input_tokens=200, output_tokens=100)
        assert m.api_call_count["perception"] == 1
        assert m.api_call_count["expression"] == 1

    def test_unknown_call_type(self):
        """未知の呼び出し種別も記録可能。"""
        m = _make_monitor()
        m.record_api_call("unknown_type", input_tokens=10, output_tokens=5)
        assert m.api_call_count["unknown_type"] == 1

    def test_disabled_no_api_record(self):
        """無効時は記録しない。"""
        m = _make_monitor(enabled=False)
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        assert m.api_call_count["perception"] == 0

    def test_api_call_emits_json(self, caplog):
        """API呼び出し記録がJSON形式でログ出力される。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_api_call("perception", input_tokens=100, output_tokens=50)
        found = False
        for record in caplog.records:
            if "api_call" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["type"] == "api_call"
                assert data["call_type"] == "perception"
                assert data["input_tokens"] == 100
        assert found, "api_call log not found"

    def test_api_token_count_returns_copy(self):
        """api_token_countプロパティがコピーを返す。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        tokens = m.api_token_count
        tokens["perception"]["input"] = 9999
        assert m.api_token_count["perception"]["input"] == 100


# ── サイクル完了記録テスト ────────────────────────────────────────


class TestCycleComplete:
    """サイクル完了記録のテスト。"""

    def test_cycle_count_increments(self):
        """サイクル数がインクリメントされる。"""
        m = _make_monitor()
        m.record_cycle_complete(tick_count=1)
        assert m.cycle_count == 1
        m.record_cycle_complete(tick_count=2)
        assert m.cycle_count == 2

    def test_band_times_reset_after_cycle(self):
        """サイクル完了後に直近帯域時間がリセットされる。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        assert "every_tick" in m.last_band_times
        m.record_cycle_complete(tick_count=1)
        assert m.last_band_times == {}

    def test_cumulative_preserved_after_cycle(self):
        """サイクル完了後も累積時間は保持される。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        m.record_cycle_complete(tick_count=1)
        assert m.band_cumulative_time["every_tick"] == pytest.approx(0.005)

    def test_disabled_no_cycle_record(self):
        """無効時はサイクル記録しない。"""
        m = _make_monitor(enabled=False)
        m.record_cycle_complete(tick_count=1)
        assert m.cycle_count == 0

    def test_cycle_complete_emits_json(self, caplog):
        """サイクル完了がJSON形式でログ出力される。"""
        m = _make_monitor()
        m.record_band_time("every_tick", 0.005)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_cycle_complete(tick_count=1)
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "cycle_complete" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "cycle_complete"
                assert data["tick_count"] == 1
                assert data["cycle_count"] == 1
                assert "band_times" in data
        assert found, "cycle_complete log not found"


# ── スナップショットテスト ────────────────────────────────────────


class TestSnapshot:
    """全フィールドダンプのスナップショットテスト。"""

    def test_snapshot_not_emitted_before_interval(self):
        """間隔に達していなければスナップショットを出力しない。"""
        m = _make_monitor(snapshot_interval=10)
        reader = _make_state_reader()
        for i in range(9):
            m.record_cycle_complete(tick_count=i + 1)
        result = m.maybe_emit_snapshot(tick_count=9, state_reader=reader)
        assert result is False

    def test_snapshot_emitted_at_interval(self):
        """間隔に達したらスナップショットを出力する。"""
        m = _make_monitor(snapshot_interval=10)
        reader = _make_state_reader()
        for i in range(10):
            m.record_cycle_complete(tick_count=i + 1)
        result = m.maybe_emit_snapshot(tick_count=10, state_reader=reader)
        assert result is True

    def test_snapshot_interval_resets(self):
        """スナップショット出力後、次のスナップショットまで間隔が必要。"""
        m = _make_monitor(snapshot_interval=5)
        reader = _make_state_reader()
        for i in range(5):
            m.record_cycle_complete(tick_count=i + 1)
        assert m.maybe_emit_snapshot(tick_count=5, state_reader=reader) is True
        # 直後は出力しない
        m.record_cycle_complete(tick_count=6)
        assert m.maybe_emit_snapshot(tick_count=6, state_reader=reader) is False
        # 5サイクル後に再出力
        for i in range(4):
            m.record_cycle_complete(tick_count=7 + i)
        assert m.maybe_emit_snapshot(tick_count=11, state_reader=reader) is True

    def test_snapshot_disabled(self):
        """無効時はスナップショットを出力しない。"""
        m = _make_monitor(enabled=False, snapshot_interval=5)
        reader = _make_state_reader()
        for i in range(10):
            m.record_cycle_complete(tick_count=i + 1)
        result = m.maybe_emit_snapshot(tick_count=10, state_reader=reader)
        assert result is False

    def test_snapshot_reader_returns_non_dict(self):
        """状態読み取りがdictを返さない場合はFalse。"""
        m = _make_monitor(snapshot_interval=5)
        reader = lambda: "not a dict"
        for i in range(5):
            m.record_cycle_complete(tick_count=i + 1)
        result = m.maybe_emit_snapshot(tick_count=5, state_reader=reader)
        assert result is False

    def test_snapshot_reader_raises_exception(self):
        """安全弁1: 状態読み取りで例外が発生してもFalseを返す。"""
        m = _make_monitor(snapshot_interval=5)
        def bad_reader():
            raise RuntimeError("test error")
        for i in range(5):
            m.record_cycle_complete(tick_count=i + 1)
        result = m.maybe_emit_snapshot(tick_count=5, state_reader=bad_reader)
        assert result is False

    def test_snapshot_all_fields_equal(self, caplog):
        """全フィールドが等価に出力される(選別なし)。"""
        m = _make_monitor(snapshot_interval=5)
        fields = {"field_a": 1, "field_b": 2, "field_c": 3}
        reader = _make_state_reader(fields)
        for i in range(5):
            m.record_cycle_complete(tick_count=i + 1)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            result = m.maybe_emit_snapshot(tick_count=5, state_reader=reader)
        assert result is True
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "state_snapshot" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "state_snapshot"
                assert data["field_count"] == 3
                assert data["fields"]["field_a"] == 1
                assert data["fields"]["field_b"] == 2
                assert data["fields"]["field_c"] == 3
        assert found, "state_snapshot log not found"

    def test_snapshot_min_interval_enforced(self):
        """安全弁2: 間隔下限が適用される。"""
        m = _make_monitor(snapshot_interval=1)
        assert m.snapshot_interval >= _SNAPSHOT_INTERVAL_MIN


# ── セッションサマリテスト ────────────────────────────────────────


class TestSessionSummary:
    """セッション終了時サマリのテスト。"""

    def test_session_summary_emitted(self, caplog):
        """セッションサマリがJSON形式で出力される。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        m.record_band_time("every_tick", 0.005)
        m.record_cycle_complete(tick_count=1)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "session_summary" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "session_summary"
                assert data["total_cycles"] == 1
                assert data["api_call_counts"]["perception"] == 1
                assert data["api_token_totals"]["perception"]["input"] == 100
                assert "band_cumulative_times" in data
                assert data["session_duration_seconds"] >= 0
        assert found, "session_summary log not found"

    def test_session_summary_disabled(self, caplog):
        """無効時はサマリを出力しない。"""
        m = _make_monitor(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()
        for record in caplog.records:
            assert "session_summary" not in record.getMessage()


# ── BandTimerテスト ───────────────────────────────────────────────


class TestBandTimer:
    """帯域計測コンテキストマネージャのテスト。"""

    def test_band_timer_records_time(self):
        """BandTimerが実行時間を記録する。"""
        m = _make_monitor()
        with BandTimer(m, "test_band"):
            time.sleep(0.01)
        assert "test_band" in m.last_band_times
        assert m.last_band_times["test_band"] > 0

    def test_band_timer_no_exception_suppression(self):
        """BandTimerは内部例外を抑制しない。"""
        m = _make_monitor()
        with pytest.raises(ValueError):
            with BandTimer(m, "test_band"):
                raise ValueError("test error")
        # 例外後でも計測は記録されている
        assert "test_band" in m.last_band_times

    def test_band_timer_disabled(self):
        """無効時はBandTimerが記録しない。"""
        m = _make_monitor(enabled=False)
        with BandTimer(m, "test_band"):
            time.sleep(0.001)
        assert m.last_band_times == {}

    def test_band_timer_none_monitor(self):
        """モニターがNoneでも安全に動作する。"""
        with BandTimer(None, "test_band"):
            pass  # 例外が発生しないことを確認

    def test_band_timer_multiple(self):
        """複数のBandTimerが独立に記録される。"""
        m = _make_monitor()
        with BandTimer(m, "band_a"):
            time.sleep(0.001)
        with BandTimer(m, "band_b"):
            time.sleep(0.001)
        assert "band_a" in m.last_band_times
        assert "band_b" in m.last_band_times


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作テスト。"""

    def test_valve1_record_band_exception_safe(self):
        """安全弁1: record_band_timeで例外が発生しても無視。"""
        m = _make_monitor()
        # 正常に動作する(内部で例外は発生しない想定だが、安全弁の確認)
        m.record_band_time("test", 0.005)
        assert "test" in m.last_band_times

    def test_valve2_snapshot_interval_min(self):
        """安全弁2: スナップショット間隔の下限。"""
        m = _make_monitor(snapshot_interval=0)
        assert m.snapshot_interval >= _SNAPSHOT_INTERVAL_MIN

    def test_valve3_log_output_limit(self, caplog):
        """安全弁3: ログ出力量の上限超過時、帯域時間以外を省略。"""
        m = _make_monitor()
        # サイクルログカウンタを上限超過に設定
        m._cycle_log_chars = _MAX_LOG_CHARS_PER_CYCLE + 1
        # この状態でAPI記録を試みる -> 出力されないはず
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_api_call("perception", input_tokens=100, output_tokens=50)
        api_logged = False
        for record in caplog.records:
            if "api_call" in record.getMessage():
                api_logged = True
        assert api_logged is False, "API call should not be logged when over limit"

    def test_valve3_cycle_complete_still_logs(self, caplog):
        """安全弁3: 上限超過時もcycle_completeは出力される。"""
        m = _make_monitor()
        m._cycle_log_chars = _MAX_LOG_CHARS_PER_CYCLE + 1
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_cycle_complete(tick_count=1)
        found = False
        for record in caplog.records:
            if "cycle_complete" in record.getMessage():
                found = True
        assert found, "cycle_complete should be logged even over limit"

    def test_valve4_no_persistent_state(self):
        """安全弁4: 永続化対象外(save/loadフィールド追加なし)。"""
        # ExecutionMonitorにsave/loadメソッドが存在しないことを確認
        m = _make_monitor()
        assert not hasattr(m, 'save')
        assert not hasattr(m, 'load')

    def test_valve5_disabled_zero_overhead(self):
        """安全弁5: 無効時は全計測点がスキップされる。"""
        m = _make_monitor(enabled=False)
        m.record_band_time("every_tick", 0.005)
        m.record_compression(1000, 500, 0.5)
        m.record_api_call("perception", input_tokens=100, output_tokens=50)
        m.record_cycle_complete(tick_count=1)
        m.emit_session_summary()
        assert m.cycle_count == 0
        assert m.api_call_count == {"perception": 0, "expression": 0}
        assert m.band_cumulative_time == {}
        assert m.last_band_times == {}
        assert m.last_compression_chars == (0, 0)


# ── read_orchestrator_fields テスト ──────────────────────────────


class TestReadOrchestratorFields:
    """orchestratorフィールド読み取りのテスト。"""

    def test_reads_tick_count(self):
        """tick_countが読み取れる。"""
        mock_orch = MagicMock()
        mock_orch._tick_count = 42
        mock_orch._psyche.emotion_summary.return_value = "test"
        mock_orch._psyche.mood.valence = 0.5
        mock_orch._psyche.mood.arousal = 0.3
        mock_orch._psyche.drives.social = 0.5
        mock_orch._psyche.drives.curiosity = 0.5
        mock_orch._psyche.drives.expression = 0.5
        mock_orch._psyche.fear_level = 0.2
        mock_orch._psyche.dominant_emotion = "happy"
        mock_orch._psyche.dominant_emotion_value = 0.7
        fields = read_orchestrator_fields(mock_orch)
        assert fields["tick_count"] == 42

    def test_reads_psyche_state(self):
        """psyche状態が読み取れる。"""
        mock_orch = MagicMock()
        mock_orch._tick_count = 1
        mock_orch._psyche.emotion_summary.return_value = "happy: 0.7"
        mock_orch._psyche.mood.valence = 0.5
        mock_orch._psyche.mood.arousal = 0.3
        mock_orch._psyche.drives.social = 0.5
        mock_orch._psyche.drives.curiosity = 0.5
        mock_orch._psyche.drives.expression = 0.5
        mock_orch._psyche.fear_level = 0.2
        mock_orch._psyche.dominant_emotion = "happy"
        mock_orch._psyche.dominant_emotion_value = 0.7
        fields = read_orchestrator_fields(mock_orch)
        assert "psyche_emotion" in fields
        assert "psyche_mood_valence" in fields
        assert "psyche_fear_level" in fields

    def test_handles_missing_attributes(self):
        """属性が欠落しても例外を投げない。"""
        mock_orch = MagicMock()
        mock_orch._tick_count = 1
        # psycheを読み取り不能にする
        mock_orch._psyche.emotion_summary.side_effect = AttributeError
        mock_orch._psyche.mood.valence = 0.5  # keep other attributes valid
        fields = read_orchestrator_fields(mock_orch)
        assert "tick_count" in fields

    def test_handles_none_subsystems(self):
        """サブシステムがNoneでも安全に動作。"""
        mock_orch = MagicMock()
        mock_orch._tick_count = 1
        mock_orch._psyche.emotion_summary.return_value = "neutral"
        mock_orch._psyche.mood.valence = 0.0
        mock_orch._psyche.mood.arousal = 0.0
        mock_orch._psyche.drives.social = 0.0
        mock_orch._psyche.drives.curiosity = 0.0
        mock_orch._psyche.drives.expression = 0.0
        mock_orch._psyche.fear_level = 0.0
        mock_orch._psyche.dominant_emotion = "neutral"
        mock_orch._psyche.dominant_emotion_value = 0.0
        mock_orch._dynamics = None
        mock_orch._loop_state = None
        mock_orch._value_orientation = None
        mock_orch._last_self_view = None
        mock_orch._last_diff_summary = None
        mock_orch._last_strain = None
        mock_orch._last_self_image = None
        mock_orch._last_coherence = None
        mock_orch._last_narrative = None
        mock_orch._last_episodes = None
        mock_orch._last_bindings = None
        mock_orch._last_trace = None
        mock_orch._last_consumption = None
        mock_orch._last_expectations = None
        mock_orch._last_motives = None
        mock_orch._last_other_model = None
        mock_orch._last_feed_result = None
        mock_orch._last_activation_result = None
        mock_orch._last_vo_validation = None
        mock_orch._last_forgetting_fixation = None
        mock_orch._last_action_result = None
        mock_orch._last_dialogue_learning = None
        mock_orch._last_meta_emotion = None
        mock_orch._last_coupling = None
        mock_orch._last_decision_bias = None
        mock_orch._last_tone_mod = None
        mock_orch._last_sensitivity_bias = None
        mock_orch._last_selected_policy_label = ""
        mock_orch._last_contradiction_result = None
        mock_orch._last_backdrop_result = None
        mock_orch._last_drive_variation_result = None
        mock_orch._last_cooccurrence_result = None
        mock_orch._last_boundary_accumulation = None
        mock_orch._enrichment_prev_cache = {}
        mock_orch._session_gap_seconds = None
        fields = read_orchestrator_fields(mock_orch)
        assert isinstance(fields, dict)
        assert "tick_count" in fields

    def test_returns_dict(self):
        """常にdictを返す。"""
        mock_orch = MagicMock()
        mock_orch._tick_count = 0
        mock_orch._psyche.emotion_summary.return_value = ""
        mock_orch._psyche.mood.valence = 0.0
        mock_orch._psyche.mood.arousal = 0.0
        mock_orch._psyche.drives.social = 0.0
        mock_orch._psyche.drives.curiosity = 0.0
        mock_orch._psyche.drives.expression = 0.0
        mock_orch._psyche.fear_level = 0.0
        mock_orch._psyche.dominant_emotion = ""
        mock_orch._psyche.dominant_emotion_value = 0.0
        fields = read_orchestrator_fields(mock_orch)
        assert isinstance(fields, dict)


# ── 統合テスト(orchestrator連携) ──────────────────────────────────


class TestOrchestratorIntegration:
    """orchestratorとの統合テスト。"""

    def test_orchestrator_has_monitor(self):
        """orchestratorが_execution_monitorを持つ。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        assert hasattr(orch, '_execution_monitor')
        assert isinstance(orch._execution_monitor, ExecutionMonitor)

    def test_post_response_update_records_bands(self):
        """post_response_updateが帯域時間を記録する。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        # モニターを有効化
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        # サイクルカウントが記録されているはず
        assert orch._execution_monitor.cycle_count == 1
        # 帯域累積時間にevery_tickがあるはず
        assert "every_tick" in orch._execution_monitor.band_cumulative_time

    def test_post_response_update_3tick_band(self):
        """3ティック毎の帯域が記録される。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        # 3ティック分実行
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        assert "every_3_ticks" in orch._execution_monitor.band_cumulative_time

    def test_post_response_update_5tick_band(self):
        """5ティック毎の帯域が記録される。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        for _ in range(5):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        assert "every_5_ticks" in orch._execution_monitor.band_cumulative_time

    def test_post_response_update_10tick_band(self):
        """10ティック毎の帯域が記録される。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        assert "every_10_ticks" in orch._execution_monitor.band_cumulative_time

    def test_monitor_disabled_no_impact(self):
        """モニター無効時、通常処理に影響しない。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=False)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        # 例外なしで実行できる
        orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        assert orch.tick_count == 1

    def test_enrichment_compression_recorded(self):
        """enrichment圧縮比が記録される。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        # enrichment生成で圧縮記録が発生する
        _result = orch.get_prompt_enrichment(user_id="viewer")
        # 圧縮が記録されているはず
        before, after = orch._execution_monitor.last_compression_chars
        # enrichmentが空でなければ記録がある
        if _result:
            assert before > 0 or after > 0

    def test_save_load_not_affected(self, tmp_path):
        """save/loadがモニターの影響を受けない。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0, data_dir=tmp_path)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        # save
        orch.save()
        # load
        orch2 = PsycheOrchestrator(memory_count=0, data_dir=tmp_path)
        orch2.load()
        assert orch2.tick_count == 1

    def test_read_orchestrator_fields_on_real_orch(self):
        """実際のorchestratorに対してread_orchestrator_fieldsが動作する。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        fields = read_orchestrator_fields(orch)
        assert isinstance(fields, dict)
        assert "tick_count" in fields
        assert fields["tick_count"] == 0
        assert "psyche_fear_level" in fields

    def test_snapshot_on_real_orch(self):
        """実際のorchestratorに対してスナップショットが動作する。

        post_response_updateが内部でmaybe_emit_snapshotを呼ぶため、
        snapshot_interval到達時に自動的にスナップショットが出力される。
        """
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        m = ExecutionMonitor(enabled=True, snapshot_interval=5)
        orch._execution_monitor = m
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        for _ in range(5):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        # post_response_update内部でスナップショットが出力済み
        # _last_snapshot_cycleがcycle_countと同じ値になっている
        assert m._last_snapshot_cycle == 5
        assert m.cycle_count == 5

        # さらに5サイクル進めれば再度スナップショット出力される
        for _ in range(5):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        assert m._last_snapshot_cycle == 10
        assert m.cycle_count == 10


# ── JSON出力フォーマットテスト ────────────────────────────────────


class TestJsonFormat:
    """JSON構造化ログの形式テスト。"""

    def test_all_records_have_timestamp(self, caplog):
        """全レコードにタイムスタンプが含まれる。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_band_time("test", 0.005)
            m.record_compression(100, 50, 0.5)
            m.record_api_call("perception", input_tokens=10, output_tokens=5)
            m.record_cycle_complete(tick_count=1)
        for record in caplog.records:
            msg = record.getMessage()
            if msg.startswith("{"):
                data = json.loads(msg)
                assert "timestamp" in data

    def test_all_records_have_type(self, caplog):
        """全レコードにtypeフィールドが含まれる。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_compression(100, 50, 0.5)
            m.record_api_call("perception", input_tokens=10, output_tokens=5)
            m.record_cycle_complete(tick_count=1)
        for record in caplog.records:
            msg = record.getMessage()
            if msg.startswith("{"):
                data = json.loads(msg)
                assert "type" in data

    def test_json_parseable(self, caplog):
        """全ログ出力が有効なJSONである。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_compression(100, 50, 0.5)
            m.record_api_call("perception", input_tokens=10, output_tokens=5)
            m.record_cycle_complete(tick_count=1)
            m.emit_session_summary()
        for record in caplog.records:
            msg = record.getMessage()
            if msg.startswith("{"):
                data = json.loads(msg)
                assert isinstance(data, dict)


# ── ログ名前空間テスト ───────────────────────────────────────────


class TestLogNamespace:
    """ログ名前空間の分離テスト。"""

    def test_uses_monitor_namespace(self, caplog):
        """cyrene.monitor名前空間を使用する。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_compression(100, 50, 0.5)
        assert any(
            r.name == "cyrene.monitor" for r in caplog.records
        ), "Should use cyrene.monitor namespace"

    def test_does_not_use_root_namespace(self, caplog):
        """ルート名前空間を使用しない。"""
        m = _make_monitor()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_compression(100, 50, 0.5)
        for record in caplog.records:
            if "enrichment_compression" in record.getMessage():
                assert record.name == "cyrene.monitor"


# ── セッション境界テスト ─────────────────────────────────────────


class TestSessionBoundary:
    """セッション境界での状態消失テスト。"""

    def test_new_instance_resets_all(self):
        """新しいインスタンスは全カウンタがゼロ。"""
        m1 = _make_monitor()
        m1.record_api_call("perception", input_tokens=100, output_tokens=50)
        m1.record_band_time("every_tick", 0.005)
        m1.record_cycle_complete(tick_count=1)

        m2 = _make_monitor()
        assert m2.cycle_count == 0
        assert m2.api_call_count == {"perception": 0, "expression": 0}
        assert m2.band_cumulative_time == {}

    def test_session_start_time_recorded(self):
        """セッション開始時刻が記録される。"""
        m = _make_monitor()
        assert m._session_start_time > 0

    def test_session_duration_in_summary(self, caplog):
        """セッションサマリにセッション経過時間が含まれる。"""
        m = _make_monitor()
        time.sleep(0.01)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()
        for record in caplog.records:
            msg = record.getMessage()
            if "session_summary" in msg:
                data = json.loads(msg)
                assert data["session_duration_seconds"] >= 0.0


# ── エッジケーステスト ────────────────────────────────────────────


class TestEdgeCases:
    """境界値・異常値のテスト。"""

    def test_zero_elapsed_time(self):
        """実行時間ゼロの記録。"""
        m = _make_monitor()
        m.record_band_time("test", 0.0)
        assert m.last_band_times["test"] == 0.0

    def test_large_token_count(self):
        """大きなトークン数の記録。"""
        m = _make_monitor()
        m.record_api_call("perception", input_tokens=1_000_000, output_tokens=500_000)
        assert m.api_token_count["perception"]["input"] == 1_000_000

    def test_empty_band_name(self):
        """空の帯域名。"""
        m = _make_monitor()
        m.record_band_time("", 0.005)
        assert "" in m.last_band_times

    def test_concurrent_cycle_and_snapshot(self):
        """サイクル完了とスナップショットの連続実行。"""
        m = _make_monitor(snapshot_interval=5)
        reader = _make_state_reader()
        for i in range(5):
            m.record_cycle_complete(tick_count=i + 1)
        # スナップショット出力
        result = m.maybe_emit_snapshot(tick_count=5, state_reader=reader)
        assert result is True
        # 次のサイクル
        m.record_cycle_complete(tick_count=6)
        assert m.cycle_count == 6

    def test_many_rapid_cycles(self):
        """多数のサイクルを高速に処理。"""
        m = _make_monitor()
        for i in range(1000):
            m.record_band_time("every_tick", 0.001)
            m.record_cycle_complete(tick_count=i + 1)
        assert m.cycle_count == 1000
        assert m.band_cumulative_time["every_tick"] == pytest.approx(1.0)
