"""
tests/test_anomaly_detection.py - 動態停止検出と警告フレームワークのテスト

tools/anomaly_detection.py のテスト。
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# テスト対象
from tools.anomaly_detection import (
    SIGNAL_DRIVE,
    SIGNAL_EMOTION,
    SIGNAL_ENRICHMENT_VARIATION,
    SIGNAL_RETURN_PATHWAY,
    AnomalyDetector,
    _Snapshot,
    collect_snapshot_from_orchestrator,
)


# ── テスト用ヘルパー ──────────────────────────────────────────────


def _make_emotion(
    joy: float = 0.0,
    anger: float = 0.0,
    sorrow: float = 0.0,
    fear: float = 0.0,
    surprise: float = 0.0,
    love: float = 0.0,
    fun: float = 0.0,
) -> dict[str, float]:
    """テスト用の感情ベクトル辞書を作成する。"""
    return {
        "joy": joy,
        "anger": anger,
        "sorrow": sorrow,
        "fear": fear,
        "surprise": surprise,
        "love": love,
        "fun": fun,
    }


def _make_drive(
    social: float = 0.5,
    curiosity: float = 0.5,
    expression: float = 0.5,
) -> dict[str, float]:
    """テスト用の駆動ベクトル辞書を作成する。"""
    return {
        "social": social,
        "curiosity": curiosity,
        "expression": expression,
    }


def _fill_detector_with_identical(
    detector: AnomalyDetector,
    count: int,
    emotion: Optional[dict[str, float]] = None,
    drive: Optional[dict[str, float]] = None,
    pathway_fired: bool = False,
    enrichment_changed: int = 0,
) -> None:
    """検出器に同一の値のスナップショットを指定回数分蓄積する。"""
    if emotion is None:
        emotion = _make_emotion()
    if drive is None:
        drive = _make_drive()
    for i in range(count):
        detector.record_snapshot(
            tick_number=i,
            emotion_values=emotion,
            drive_values=drive,
            return_pathway_fired=pathway_fired,
            enrichment_variation_count=enrichment_changed,
        )


# ── Snapshot テスト ──────────────────────────────────────────────


class TestSnapshot:
    """_Snapshotの基本テスト。"""

    def test_creation(self):
        """スナップショットが正しく生成される。"""
        s = _Snapshot(
            tick_number=10,
            emotion_values=_make_emotion(joy=0.5),
            drive_values=_make_drive(social=0.8),
            return_pathway_fired=True,
            enrichment_variation_count=3,
        )
        assert s.tick_number == 10
        assert s.emotion_values["joy"] == 0.5
        assert s.drive_values["social"] == 0.8
        assert s.return_pathway_fired is True
        assert s.enrichment_variation_count == 3
        assert s.timestamp > 0

    def test_to_dict(self):
        """to_dictが全フィールドを含む辞書を返す。"""
        s = _Snapshot(
            tick_number=5,
            emotion_values=_make_emotion(),
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        d = s.to_dict()
        assert "tick_number" in d
        assert "timestamp" in d
        assert "emotion_values" in d
        assert "drive_values" in d
        assert "return_pathway_fired" in d
        assert "enrichment_variation_count" in d

    def test_values_are_copies(self):
        """渡した辞書の変更がスナップショットに影響しない。"""
        emotion = _make_emotion(joy=0.3)
        s = _Snapshot(
            tick_number=1,
            emotion_values=emotion,
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        emotion["joy"] = 0.9
        assert s.emotion_values["joy"] == 0.3


# ── AnomalyDetector 初期化テスト ──────────────────────────────────


class TestAnomalyDetectorInit:
    """AnomalyDetectorの初期化テスト。"""

    def test_default_init(self):
        """デフォルト初期化の検証。"""
        d = AnomalyDetector(enabled=True)
        assert d.enabled is True
        assert d.buffer_size == 0
        assert d.buffer_max == 30
        assert d.snapshot_count == 0
        # 全信号種別が非停止状態
        for flag in d.stall_flags.values():
            assert flag is False
        # 全カウンタがゼロ
        for count in d.stall_detected_count.values():
            assert count == 0
        for count in d.stall_resolved_count.values():
            assert count == 0

    def test_custom_buffer_max(self):
        """バッファ上限をカスタム値で初期化できる。"""
        d = AnomalyDetector(buffer_max=10, enabled=True)
        assert d.buffer_max == 10

    def test_buffer_max_minimum(self):
        """バッファ上限に0以下を指定しても1になる。"""
        d = AnomalyDetector(buffer_max=0, enabled=True)
        assert d.buffer_max == 1
        d2 = AnomalyDetector(buffer_max=-5, enabled=True)
        assert d2.buffer_max == 1

    def test_disabled_by_default_without_env(self):
        """環境変数なしではデフォルトで無効。"""
        with patch.dict(os.environ, {}, clear=True):
            d = AnomalyDetector()
            assert d.enabled is False

    def test_enabled_by_env(self):
        """CYRENE_MONITOR=1で有効化される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            d = AnomalyDetector()
            assert d.enabled is True

    def test_explicit_enabled_overrides_env(self):
        """明示的なenabled指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            d = AnomalyDetector(enabled=True)
            assert d.enabled is True
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            d = AnomalyDetector(enabled=False)
            assert d.enabled is False


# ── スナップショット蓄積テスト ──────────────────────────────────


class TestSnapshotAccumulation:
    """第1段: スナップショットFIFOバッファの蓄積テスト。"""

    def test_single_snapshot(self):
        """1件のスナップショットが蓄積される。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        d.record_snapshot(
            tick_number=1,
            emotion_values=_make_emotion(),
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        assert d.buffer_size == 1
        assert d.snapshot_count == 1

    def test_fifo_overflow(self):
        """バッファ上限を超えると最古のエントリが消失する。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        for i in range(5):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(joy=float(i) / 10),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.buffer_size == 3
        assert d.snapshot_count == 5

    def test_disabled_no_accumulation(self):
        """無効時はスナップショットが蓄積されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=False)
        d.record_snapshot(
            tick_number=1,
            emotion_values=_make_emotion(),
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        assert d.buffer_size == 0
        assert d.snapshot_count == 0


# ── 動態停止検出テスト ──────────────────────────────────────────


class TestStallDetection:
    """第2段: 変化率ゼロの検出テスト。"""

    def test_emotion_stall_detected(self):
        """感情ベクトルの全次元が同一値で停止が検出される。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        _fill_detector_with_identical(d, 5, emotion=_make_emotion(joy=0.0))
        assert d.stall_flags[SIGNAL_EMOTION] is True

    def test_emotion_no_stall_when_varying(self):
        """感情ベクトルが変化している場合は停止が検出されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        for i in range(5):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(joy=float(i) / 10),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.stall_flags[SIGNAL_EMOTION] is False

    def test_drive_stall_detected(self):
        """駆動ベクトルの全次元が同一値で停止が検出される。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        _fill_detector_with_identical(d, 5, drive=_make_drive(social=0.5))
        assert d.stall_flags[SIGNAL_DRIVE] is True

    def test_drive_no_stall_when_varying(self):
        """駆動ベクトルが変化している場合は停止が検出されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        for i in range(5):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(),
                drive_values=_make_drive(curiosity=float(i) / 10),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.stall_flags[SIGNAL_DRIVE] is False

    def test_return_pathway_stall_detected(self):
        """帰還経路が全て発火ゼロで停止が検出される。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        _fill_detector_with_identical(d, 5, pathway_fired=False)
        assert d.stall_flags[SIGNAL_RETURN_PATHWAY] is True

    def test_return_pathway_no_stall_when_fired(self):
        """帰還経路の発火があれば停止が検出されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        for i in range(4):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        # 5件目で発火あり
        d.record_snapshot(
            tick_number=4,
            emotion_values=_make_emotion(),
            drive_values=_make_drive(),
            return_pathway_fired=True,
            enrichment_variation_count=0,
        )
        assert d.stall_flags[SIGNAL_RETURN_PATHWAY] is False

    def test_enrichment_stall_detected(self):
        """enrichment変動がゼロ継続で停止が検出される。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        _fill_detector_with_identical(d, 5, enrichment_changed=0)
        assert d.stall_flags[SIGNAL_ENRICHMENT_VARIATION] is True

    def test_enrichment_no_stall_when_varying(self):
        """enrichment変動がある場合は停止が検出されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        for i in range(5):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=i,  # 変動あり
            )
        assert d.stall_flags[SIGNAL_ENRICHMENT_VARIATION] is False

    def test_no_detection_before_buffer_full(self):
        """バッファが満杯でない場合は検出が行われない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        _fill_detector_with_identical(d, 4)  # 4件(5未満)
        # バッファが満杯でないので検出は行われない
        for flag in d.stall_flags.values():
            assert flag is False

    def test_all_signals_can_stall_simultaneously(self):
        """全信号種別が同時に停止できる。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(
            d, 3,
            emotion=_make_emotion(),
            drive=_make_drive(),
            pathway_fired=False,
            enrichment_changed=0,
        )
        assert d.stall_flags[SIGNAL_EMOTION] is True
        assert d.stall_flags[SIGNAL_DRIVE] is True
        assert d.stall_flags[SIGNAL_RETURN_PATHWAY] is True
        assert d.stall_flags[SIGNAL_ENRICHMENT_VARIATION] is True

    def test_non_zero_constant_emotion_is_stall(self):
        """全次元がゼロでなくても同一値の継続は停止として検出される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(
            d, 3,
            emotion=_make_emotion(joy=0.7, anger=0.3, fun=0.2),
        )
        assert d.stall_flags[SIGNAL_EMOTION] is True

    def test_partial_change_breaks_stall(self):
        """1次元でも変化すれば停止は検出されない。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        base = _make_emotion(joy=0.5)
        d.record_snapshot(0, base, _make_drive(), False, 0)
        d.record_snapshot(1, base, _make_drive(), False, 0)
        changed = _make_emotion(joy=0.5, anger=0.01)  # angerだけ変化
        d.record_snapshot(2, changed, _make_drive(), False, 0)
        assert d.stall_flags[SIGNAL_EMOTION] is False


# ── 遷移検出とログ出力テスト ──────────────────────────────────


class TestAlertEmission:
    """第3段: 警告の出力テスト。"""

    def test_stall_detected_log_emitted(self, caplog):
        """停止状態への遷移時に検出ログが出力される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            _fill_detector_with_identical(d, 3)

        stall_logs = [
            r for r in caplog.records
            if "dynamics_stall_detected" in r.getMessage()
        ]
        assert len(stall_logs) > 0

    def test_stall_resolved_log_emitted(self, caplog):
        """停止状態の解消時に解消ログが出力される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        # まず停止状態にする
        _fill_detector_with_identical(d, 3)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            caplog.clear()
            # 感情を変化させて停止を解消
            d.record_snapshot(
                tick_number=10,
                emotion_values=_make_emotion(joy=0.9),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )

        resolved_logs = [
            r for r in caplog.records
            if "dynamics_stall_resolved" in r.getMessage()
        ]
        # 感情の停止が解消されたログが出る
        assert len(resolved_logs) >= 1

    def test_no_repeated_stall_log(self, caplog):
        """停止状態が継続中は同一の警告を出力しない(安全弁8)。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            # バッファを満たして停止検出(3件)
            _fill_detector_with_identical(d, 3)
            first_count = len([
                r for r in caplog.records
                if "dynamics_stall_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ])

            # さらに同一値のスナップショットを追加(停止継続中)
            caplog.clear()
            d.record_snapshot(
                tick_number=10,
                emotion_values=_make_emotion(),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
            repeated_logs = [
                r for r in caplog.records
                if "dynamics_stall_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ]
            # 継続中は感情についての検出ログは出力されない
            assert len(repeated_logs) == 0

    def test_stall_log_contains_signal_info(self, caplog):
        """検出ログに信号種別と停止情報が含まれる。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            _fill_detector_with_identical(d, 3)

        stall_logs = [
            r for r in caplog.records
            if "dynamics_stall_detected" in r.getMessage()
        ]
        assert len(stall_logs) > 0
        for log_record in stall_logs:
            msg = log_record.getMessage()
            parsed = json.loads(msg)
            assert "signal" in parsed
            assert "tick_number" in parsed
            assert "consecutive_snapshots" in parsed
            assert parsed["type"] == "dynamics_stall_detected"

    def test_stall_value_included_for_emotion(self, caplog):
        """感情停止の検出ログにstall_valueが含まれる。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            _fill_detector_with_identical(
                d, 3, emotion=_make_emotion(joy=0.4, anger=0.1)
            )

        emotion_stall_logs = [
            r for r in caplog.records
            if "dynamics_stall_detected" in r.getMessage()
            and SIGNAL_EMOTION in r.getMessage()
        ]
        assert len(emotion_stall_logs) >= 1
        parsed = json.loads(emotion_stall_logs[0].getMessage())
        assert "stall_value" in parsed
        assert parsed["stall_value"]["joy"] == 0.4

    def test_resolved_log_format(self, caplog):
        """解消ログのフォーマットが正しい。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            caplog.clear()
            d.record_snapshot(
                tick_number=10,
                emotion_values=_make_emotion(joy=0.9),
                drive_values=_make_drive(social=0.9),
                return_pathway_fired=True,
                enrichment_variation_count=5,
            )

        resolved_logs = [
            r for r in caplog.records
            if "dynamics_stall_resolved" in r.getMessage()
        ]
        for log_record in resolved_logs:
            parsed = json.loads(log_record.getMessage())
            assert parsed["type"] == "dynamics_stall_resolved"
            assert "signal" in parsed
            assert "tick_number" in parsed


# ── カウンタテスト ──────────────────────────────────────────────


class TestCumulativeCounters:
    """セッション累積カウンタのテスト。"""

    def test_stall_detected_count_increments(self):
        """停止検出で累積カウンタが増加する。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)
        assert d.stall_detected_count[SIGNAL_EMOTION] == 1
        assert d.stall_detected_count[SIGNAL_DRIVE] == 1

    def test_stall_resolved_count_increments(self):
        """停止解消で累積カウンタが増加する。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)
        # 解消
        d.record_snapshot(
            tick_number=10,
            emotion_values=_make_emotion(joy=0.9),
            drive_values=_make_drive(social=0.9),
            return_pathway_fired=True,
            enrichment_variation_count=5,
        )
        assert d.stall_resolved_count[SIGNAL_EMOTION] == 1
        assert d.stall_resolved_count[SIGNAL_DRIVE] == 1

    def test_multiple_stall_resolve_cycles(self):
        """停止→解消→停止を繰り返すとカウンタが累積する。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)

        # 第1サイクル: 停止
        _fill_detector_with_identical(d, 3)
        assert d.stall_detected_count[SIGNAL_EMOTION] == 1

        # 解消
        d.record_snapshot(
            tick_number=10,
            emotion_values=_make_emotion(joy=0.9),
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        assert d.stall_resolved_count[SIGNAL_EMOTION] == 1

        # 第2サイクル: 再び停止
        _fill_detector_with_identical(d, 3, emotion=_make_emotion(anger=0.5))
        assert d.stall_detected_count[SIGNAL_EMOTION] == 2

    def test_snapshot_count_tracks_total(self):
        """全体のスナップショット取得回数が正しく追跡される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        for i in range(10):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(joy=float(i) / 10),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.snapshot_count == 10


# ── セッションサマリテスト ──────────────────────────────────────


class TestSessionSummary:
    """セッションサマリ出力のテスト。"""

    def test_session_summary_content(self):
        """セッションサマリが正しい内容を含む。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)
        summary = d.emit_session_summary()
        assert summary is not None
        assert summary["type"] == "anomaly_detection_session_summary"
        assert summary["snapshot_count"] == 3
        assert summary["buffer_max"] == 3
        assert "stall_detected_counts" in summary
        assert "stall_resolved_counts" in summary
        assert "current_stall_flags" in summary

    def test_session_summary_disabled(self):
        """無効時はセッションサマリがNoneを返す。"""
        d = AnomalyDetector(buffer_max=3, enabled=False)
        summary = d.emit_session_summary()
        assert summary is None

    def test_session_summary_log_output(self, caplog):
        """セッションサマリがログに出力される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            d.emit_session_summary()

        summary_logs = [
            r for r in caplog.records
            if "anomaly_detection_session_summary" in r.getMessage()
        ]
        assert len(summary_logs) == 1


# ── 読み取り専用アクセサテスト ──────────────────────────────────


class TestGetSummary:
    """get_summary()読み取り専用アクセサのテスト。"""

    def test_initial_summary(self):
        """初期状態のサマリが正しい。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        s = d.get_summary()
        assert s["snapshot_count"] == 0
        assert s["buffer_size"] == 0
        assert s["buffer_max"] == 5
        assert s["latest_snapshot"] is None

    def test_summary_after_snapshots(self):
        """スナップショット蓄積後のサマリが正しい。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)
        s = d.get_summary()
        assert s["snapshot_count"] == 3
        assert s["buffer_size"] == 3
        assert s["latest_snapshot"] is not None
        assert s["latest_snapshot"]["tick_number"] == 2

    def test_summary_returns_copies(self):
        """サマリが返す辞書がコピーであること。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        s1 = d.get_summary()
        s2 = d.get_summary()
        assert s1 is not s2
        assert s1["stall_detected_counts"] is not s2["stall_detected_counts"]


# ── プロパティの読み取り専用テスト ──────────────────────────────


class TestPropertyReadonly:
    """プロパティが読み取り専用コピーを返すことのテスト。"""

    def test_stall_flags_readonly(self):
        """stall_flagsの変更が内部状態に影響しない。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        flags = d.stall_flags
        flags[SIGNAL_EMOTION] = True
        assert d.stall_flags[SIGNAL_EMOTION] is False

    def test_stall_detected_count_readonly(self):
        """stall_detected_countの変更が内部状態に影響しない。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        counts = d.stall_detected_count
        counts[SIGNAL_EMOTION] = 99
        assert d.stall_detected_count[SIGNAL_EMOTION] == 0

    def test_stall_resolved_count_readonly(self):
        """stall_resolved_countの変更が内部状態に影響しない。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        counts = d.stall_resolved_count
        counts[SIGNAL_EMOTION] = 99
        assert d.stall_resolved_count[SIGNAL_EMOTION] == 0


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁のテスト。"""

    def test_exception_in_snapshot_does_not_propagate(self):
        """スナップショット処理中の例外が伝播しない(安全弁1)。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        # 不正な型のemotion_valuesを渡す
        d.record_snapshot(
            tick_number=1,
            emotion_values=None,  # type: ignore
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        # 例外が伝播しないことを確認(テストが到達すればOK)

    def test_disabled_skips_all(self):
        """無効時は全ての処理がスキップされる(安全弁2)。"""
        d = AnomalyDetector(buffer_max=3, enabled=False)
        _fill_detector_with_identical(d, 5)
        assert d.buffer_size == 0
        assert d.snapshot_count == 0
        for flag in d.stall_flags.values():
            assert flag is False
        summary = d.emit_session_summary()
        assert summary is None

    def test_no_state_modification_methods(self):
        """状態を変更するメソッドが存在しない(安全弁3)。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        # 公開メソッドが読み取り or 記録のみであることを確認
        public_methods = [m for m in dir(d) if not m.startswith("_")]
        # 修復系メソッドがないことを確認
        assert "reset" not in public_methods
        assert "repair" not in public_methods
        assert "fix" not in public_methods
        assert "modify" not in public_methods
        assert "set_state" not in public_methods
        assert "update_state" not in public_methods

    def test_fifo_buffer_bounded(self):
        """FIFOバッファがサイズ上限で制御されている(安全弁7)。"""
        d = AnomalyDetector(buffer_max=5, enabled=True)
        for i in range(100):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(joy=float(i) / 100),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.buffer_size <= 5

    def test_log_redundancy_suppression(self, caplog):
        """ログ出力の冗長抑制が機能する(安全弁8)。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            # 停止を検出(3件)
            _fill_detector_with_identical(d, 3)
            stall_count = len([
                r for r in caplog.records
                if "dynamics_stall_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ])

            # さらに10件の同一スナップショット(停止継続)
            for i in range(10):
                d.record_snapshot(
                    tick_number=10 + i,
                    emotion_values=_make_emotion(),
                    drive_values=_make_drive(),
                    return_pathway_fired=False,
                    enrichment_variation_count=0,
                )

            total_stall_count = len([
                r for r in caplog.records
                if "dynamics_stall_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ])
            # 感情の停止検出ログは最初の1回のみ
            assert total_stall_count == stall_count


# ── collect_snapshot_from_orchestrator テスト ─────────────────────


class TestCollectSnapshot:
    """collect_snapshot_from_orchestrator()のテスト。"""

    def _make_mock_orchestrator(
        self,
        tick: int = 10,
        emotion: Optional[dict[str, float]] = None,
        drive: Optional[dict[str, float]] = None,
    ) -> Any:
        """テスト用のモックオーケストレータを作成する。"""
        if emotion is None:
            emotion = _make_emotion(joy=0.3)
        if drive is None:
            drive = _make_drive(social=0.6)

        mock = MagicMock()
        mock._tick_count = tick
        mock._psyche.emotions.as_dict.return_value = emotion
        mock._psyche.drives.as_dict.return_value = drive
        return mock

    def test_basic_collection(self):
        """基本的なスナップショット収集が機能する。"""
        orch = self._make_mock_orchestrator(tick=10)
        result = collect_snapshot_from_orchestrator(orch)
        assert result is not None
        assert result["tick_number"] == 10
        assert result["emotion_values"]["joy"] == 0.3
        assert result["drive_values"]["social"] == 0.6
        assert result["return_pathway_fired"] is False
        assert result["enrichment_variation_count"] == 0

    def test_with_direct_pathway_fired(self):
        """直接指定の帰還経路発火有無が反映される。"""
        orch = self._make_mock_orchestrator()
        result = collect_snapshot_from_orchestrator(
            orch, last_tick_had_pathway_firing=True
        )
        assert result is not None
        assert result["return_pathway_fired"] is True

    def test_with_direct_enrichment_changed(self):
        """直接指定のenrichment変動数が反映される。"""
        orch = self._make_mock_orchestrator()
        result = collect_snapshot_from_orchestrator(
            orch, last_tick_enrichment_changed=7
        )
        assert result is not None
        assert result["enrichment_variation_count"] == 7

    def test_with_return_pathway_monitor(self):
        """ReturnPathwayMonitorからの帰還経路情報が使用される。"""
        orch = self._make_mock_orchestrator()
        rpm = MagicMock()
        rpm.last_tick_record = {"some": "data"}  # 発火あり
        result = collect_snapshot_from_orchestrator(
            orch, return_pathway_monitor=rpm
        )
        assert result is not None
        assert result["return_pathway_fired"] is True

    def test_with_return_pathway_monitor_no_fire(self):
        """ReturnPathwayMonitorで発火なしの場合。"""
        orch = self._make_mock_orchestrator()
        rpm = MagicMock()
        rpm.last_tick_record = None  # 発火なし
        result = collect_snapshot_from_orchestrator(
            orch, return_pathway_monitor=rpm
        )
        assert result is not None
        assert result["return_pathway_fired"] is False

    def test_with_enrichment_distribution_monitor(self):
        """EnrichmentDistributionMonitorからの変動情報が使用される。"""
        orch = self._make_mock_orchestrator()
        edm = MagicMock()
        edm.get_distribution_summary.return_value = {
            "latest_entry": {"total_changed": 5}
        }
        result = collect_snapshot_from_orchestrator(
            orch, enrichment_distribution_monitor=edm
        )
        assert result is not None
        assert result["enrichment_variation_count"] == 5

    def test_direct_values_override_monitors(self):
        """直接指定の値がモニターからの値より優先される。"""
        orch = self._make_mock_orchestrator()
        rpm = MagicMock()
        rpm.last_tick_record = {"some": "data"}  # 発火あり
        edm = MagicMock()
        edm.get_distribution_summary.return_value = {
            "latest_entry": {"total_changed": 5}
        }

        result = collect_snapshot_from_orchestrator(
            orch,
            return_pathway_monitor=rpm,
            enrichment_distribution_monitor=edm,
            last_tick_had_pathway_firing=False,  # 直接指定で上書き
            last_tick_enrichment_changed=0,  # 直接指定で上書き
        )
        assert result is not None
        assert result["return_pathway_fired"] is False
        assert result["enrichment_variation_count"] == 0

    def test_orchestrator_read_failure_returns_none(self):
        """オーケストレータ読み取り失敗時にNoneを返す。"""
        mock = MagicMock()
        mock._tick_count = 10
        mock._psyche.emotions.as_dict.side_effect = RuntimeError("read error")
        result = collect_snapshot_from_orchestrator(mock)
        assert result is None

    def test_monitor_read_failure_safe(self):
        """モニター読み取り失敗時も正常に動作する。"""
        orch = self._make_mock_orchestrator()
        rpm = MagicMock()
        rpm.last_tick_record = property(lambda self: None)
        type(rpm).last_tick_record = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("read error"))
        )

        result = collect_snapshot_from_orchestrator(
            orch, return_pathway_monitor=rpm
        )
        assert result is not None
        assert result["return_pathway_fired"] is False


# ── エンドツーエンドテスト ──────────────────────────────────────


class TestEndToEnd:
    """エンドツーエンドのシナリオテスト。"""

    def test_full_lifecycle(self, caplog):
        """停止検出→継続→解消→再停止のフルライフサイクル。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            # Phase 1: 同一値でバッファを満たす(停止検出)
            for i in range(3):
                d.record_snapshot(
                    tick_number=i,
                    emotion_values=_make_emotion(joy=0.3),
                    drive_values=_make_drive(),
                    return_pathway_fired=False,
                    enrichment_variation_count=0,
                )
            assert d.stall_flags[SIGNAL_EMOTION] is True
            assert d.stall_detected_count[SIGNAL_EMOTION] == 1

            # Phase 2: 停止継続(同一値をもう1件追加)
            d.record_snapshot(
                tick_number=3,
                emotion_values=_make_emotion(joy=0.3),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
            assert d.stall_flags[SIGNAL_EMOTION] is True
            assert d.stall_detected_count[SIGNAL_EMOTION] == 1  # 変化なし

            # Phase 3: 解消(感情値が変化)
            d.record_snapshot(
                tick_number=4,
                emotion_values=_make_emotion(joy=0.8),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
            assert d.stall_flags[SIGNAL_EMOTION] is False
            assert d.stall_resolved_count[SIGNAL_EMOTION] == 1

            # Phase 4: 再び停止
            for i in range(3):
                d.record_snapshot(
                    tick_number=5 + i,
                    emotion_values=_make_emotion(anger=0.5),
                    drive_values=_make_drive(),
                    return_pathway_fired=False,
                    enrichment_variation_count=0,
                )
            assert d.stall_flags[SIGNAL_EMOTION] is True
            assert d.stall_detected_count[SIGNAL_EMOTION] == 2

        # セッションサマリ
        summary = d.emit_session_summary()
        assert summary is not None
        assert summary["stall_detected_counts"][SIGNAL_EMOTION] == 2
        assert summary["stall_resolved_counts"][SIGNAL_EMOTION] == 1

    def test_independent_signal_tracking(self):
        """各信号種別が独立して追跡される。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)

        # 感情は変化するが、駆動は固定
        for i in range(3):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(joy=float(i) / 10),
                drive_values=_make_drive(social=0.5),  # 固定
                return_pathway_fired=True,  # 発火あり
                enrichment_variation_count=i + 1,  # 変動あり
            )

        assert d.stall_flags[SIGNAL_EMOTION] is False
        assert d.stall_flags[SIGNAL_DRIVE] is True
        assert d.stall_flags[SIGNAL_RETURN_PATHWAY] is False
        assert d.stall_flags[SIGNAL_ENRICHMENT_VARIATION] is False

    def test_collect_and_record_integration(self):
        """collect_snapshot_from_orchestratorとrecord_snapshotの統合テスト。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)

        orch = MagicMock()
        orch._tick_count = 1
        orch._psyche.emotions.as_dict.return_value = _make_emotion(joy=0.3)
        orch._psyche.drives.as_dict.return_value = _make_drive(social=0.5)

        for i in range(3):
            orch._tick_count = i
            snapshot_data = collect_snapshot_from_orchestrator(
                orch,
                last_tick_had_pathway_firing=False,
                last_tick_enrichment_changed=0,
            )
            assert snapshot_data is not None
            d.record_snapshot(**snapshot_data)

        assert d.buffer_size == 3
        assert d.stall_flags[SIGNAL_EMOTION] is True

    def test_large_buffer_performance(self):
        """大きなバッファサイズでも正常に動作する。"""
        d = AnomalyDetector(buffer_max=100, enabled=True)
        for i in range(100):
            d.record_snapshot(
                tick_number=i,
                emotion_values=_make_emotion(),
                drive_values=_make_drive(),
                return_pathway_fired=False,
                enrichment_variation_count=0,
            )
        assert d.buffer_size == 100
        assert d.stall_flags[SIGNAL_EMOTION] is True

    def test_structural_isolation(self):
        """構造的分離の検証: 出力メソッドは全てREAD-ONLY。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        _fill_detector_with_identical(d, 3)

        # get_summaryはREAD-ONLY
        summary1 = d.get_summary()
        summary2 = d.get_summary()
        assert summary1["stall_detected_counts"] == summary2["stall_detected_counts"]

        # emit_session_summaryはログ出力のみで内部状態を変更しない
        pre_flags = d.stall_flags
        d.emit_session_summary()
        post_flags = d.stall_flags
        assert pre_flags == post_flags

    def test_no_save_load_interface(self):
        """save/loadインタフェースが存在しない。"""
        d = AnomalyDetector(buffer_max=3, enabled=True)
        assert not hasattr(d, "save")
        assert not hasattr(d, "load")
        assert not hasattr(d, "to_dict")
        assert not hasattr(d, "from_dict")
