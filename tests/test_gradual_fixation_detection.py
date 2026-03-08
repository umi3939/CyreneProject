"""
tests/test_gradual_fixation_detection.py - 緩やかな固定化進行検出のテスト

tools/anomaly_detection.py の緩やかな固定化進行検出拡張のテスト。
設計書: design_gradual_fixation_detection.md
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import pytest

from tools.anomaly_detection import (
    SIGNAL_DRIVE,
    SIGNAL_EMOTION,
    SIGNAL_ENRICHMENT_VARIATION,
    SIGNAL_RETURN_PATHWAY,
    AnomalyDetector,
    _GRADUAL_CONVERGENCE_SIGNALS,
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


def _record(
    d: AnomalyDetector,
    tick: int,
    emotion: Optional[dict[str, float]] = None,
    drive: Optional[dict[str, float]] = None,
    pathway_fired: bool = False,
    enrichment_changed: int = 0,
) -> None:
    """ヘルパー: 1件のスナップショットを記録する。"""
    if emotion is None:
        emotion = _make_emotion()
    if drive is None:
        drive = _make_drive()
    d.record_snapshot(
        tick_number=tick,
        emotion_values=emotion,
        drive_values=drive,
        return_pathway_fired=pathway_fired,
        enrichment_variation_count=enrichment_changed,
    )


# ── 初期化テスト ──────────────────────────────────────────────


class TestGradualConvergenceInit:
    """緩やかな固定化検出の初期化テスト。"""

    def test_default_sub_window_count(self):
        """デフォルトのサブウィンドウ分割数が5である。"""
        d = AnomalyDetector(enabled=True)
        assert d.sub_window_count == 5

    def test_custom_sub_window_count(self):
        """カスタムのサブウィンドウ分割数が設定できる。"""
        d = AnomalyDetector(enabled=True, sub_window_count=3)
        assert d.sub_window_count == 3

    def test_sub_window_count_minimum(self):
        """サブウィンドウ分割数に1以下を指定しても2になる。"""
        d = AnomalyDetector(enabled=True, sub_window_count=1)
        assert d.sub_window_count == 2
        d2 = AnomalyDetector(enabled=True, sub_window_count=0)
        assert d2.sub_window_count == 2
        d3 = AnomalyDetector(enabled=True, sub_window_count=-5)
        assert d3.sub_window_count == 2

    def test_initial_convergence_flags_all_false(self):
        """初期状態では全ての緩やかな収束フラグがFalse。"""
        d = AnomalyDetector(enabled=True)
        flags = d.gradual_convergence_flags
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert flags[signal] is False

    def test_initial_convergence_counters_all_zero(self):
        """初期状態では全ての緩やかな収束カウンタがゼロ。"""
        d = AnomalyDetector(enabled=True)
        detected = d.gradual_convergence_detected_count
        resolved = d.gradual_convergence_resolved_count
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert detected[signal] == 0
            assert resolved[signal] == 0

    def test_initial_variance_series_empty(self):
        """初期状態では全ての散らばり列が空。"""
        d = AnomalyDetector(enabled=True)
        series = d.latest_variance_series
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert series[signal] == []

    def test_convergence_signals_subset_of_all(self):
        """収束検出対象信号が感情・駆動のみである。"""
        assert _GRADUAL_CONVERGENCE_SIGNALS == frozenset({SIGNAL_EMOTION, SIGNAL_DRIVE})
        assert SIGNAL_RETURN_PATHWAY not in _GRADUAL_CONVERGENCE_SIGNALS
        assert SIGNAL_ENRICHMENT_VARIATION not in _GRADUAL_CONVERGENCE_SIGNALS


# ── 単調減少判定テスト ──────────────────────────────────────────


class TestMonotonicallyDecreasing:
    """_is_monotonically_decreasing のテスト。"""

    def test_strictly_decreasing(self):
        """厳密に減少する列は単調減少と判定される。"""
        assert AnomalyDetector._is_monotonically_decreasing([5.0, 3.0, 1.0]) is True

    def test_non_increasing(self):
        """等しい要素を含む非増加列も単調減少と判定される。"""
        assert AnomalyDetector._is_monotonically_decreasing([5.0, 5.0, 3.0]) is True
        assert AnomalyDetector._is_monotonically_decreasing([5.0, 3.0, 3.0]) is True

    def test_all_same_nonzero(self):
        """全要素が同一の非ゼロ値は単調減少と判定される。"""
        assert AnomalyDetector._is_monotonically_decreasing([3.0, 3.0, 3.0]) is True

    def test_all_zero(self):
        """全要素がゼロの場合は単調減少と判定されない。"""
        assert AnomalyDetector._is_monotonically_decreasing([0.0, 0.0, 0.0]) is False

    def test_increasing(self):
        """増加する列は単調減少ではない。"""
        assert AnomalyDetector._is_monotonically_decreasing([1.0, 3.0, 5.0]) is False

    def test_non_monotone(self):
        """途中で増加する列は単調減少ではない。"""
        assert AnomalyDetector._is_monotonically_decreasing([5.0, 2.0, 4.0]) is False

    def test_single_element(self):
        """1要素の列は判定不能でFalse。"""
        assert AnomalyDetector._is_monotonically_decreasing([5.0]) is False

    def test_empty_list(self):
        """空の列は判定不能でFalse。"""
        assert AnomalyDetector._is_monotonically_decreasing([]) is False

    def test_two_elements_decreasing(self):
        """2要素の減少列は単調減少。"""
        assert AnomalyDetector._is_monotonically_decreasing([5.0, 3.0]) is True

    def test_two_elements_increasing(self):
        """2要素の増加列は単調減少でない。"""
        assert AnomalyDetector._is_monotonically_decreasing([3.0, 5.0]) is False

    def test_very_small_decrease(self):
        """微小な減少でも単調減少と判定される。"""
        assert AnomalyDetector._is_monotonically_decreasing(
            [0.001, 0.0009, 0.0008]
        ) is True


# ── バッファ不足時のスキップテスト ────────────────────────────────


class TestBufferInsufficiency:
    """バッファ不足時のスキップ(安全弁8)のテスト。"""

    def test_no_detection_when_buffer_too_small(self):
        """バッファがサブウィンドウ分割に不十分な場合は検出されない。"""
        # sub_window_count=3なので、最低6件必要(各サブウィンドウに2件)
        d = AnomalyDetector(buffer_max=5, enabled=True, sub_window_count=3)
        for i in range(5):
            _record(d, i, emotion=_make_emotion(joy=0.5 - i * 0.1))
        # バッファは5件で満杯だが、3サブウィンドウ×2=6件に足りない
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert d.gradual_convergence_flags[signal] is False

    def test_detection_when_buffer_sufficient(self):
        """バッファが十分にある場合は検出が実行される。"""
        # sub_window_count=3, buffer_max=6: 各サブウィンドウに2件
        d = AnomalyDetector(buffer_max=6, enabled=True, sub_window_count=3)
        # 分散が減少するパターン: 先頭は大きく、末尾は小さく
        # サブウィンドウ1: [joy=0.0, joy=0.5] → 分散大
        # サブウィンドウ2: [joy=0.2, joy=0.3] → 分散中
        # サブウィンドウ3: [joy=0.25, joy=0.25] → 分散ゼロ
        emotions = [
            _make_emotion(joy=0.0),
            _make_emotion(joy=0.5),
            _make_emotion(joy=0.2),
            _make_emotion(joy=0.3),
            _make_emotion(joy=0.25),
            _make_emotion(joy=0.25),
        ]
        for i, e in enumerate(emotions):
            _record(d, i, emotion=e)

        # 感情の緩やかな収束が検出されるはず
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

    def test_large_sub_window_count_exceeds_buffer(self):
        """サブウィンドウ数がバッファより大きい場合は検出されない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True, sub_window_count=10)
        for i in range(5):
            _record(d, i, emotion=_make_emotion(joy=0.5 - i * 0.1))
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert d.gradual_convergence_flags[signal] is False


# ── 分散の単調減少検出テスト ──────────────────────────────────


class TestVarianceDecreaseDetection:
    """分散の単調減少検出のテスト。"""

    def test_emotion_convergence_detected(self):
        """感情ベクトルの散らばりの単調減少が検出される。"""
        # 3サブウィンドウ×4件=12件
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # サブウィンドウ1: 大きな分散(joy: 0.0, 0.8, 0.0, 0.8)
        for i in range(4):
            _record(d, i, emotion=_make_emotion(joy=0.8 * (i % 2)))

        # サブウィンドウ2: 中程度の分散(joy: 0.3, 0.5, 0.3, 0.5)
        for i in range(4):
            _record(d, 4 + i, emotion=_make_emotion(joy=0.3 + 0.2 * (i % 2)))

        # サブウィンドウ3: 小さな分散(joy: 0.4, 0.41, 0.4, 0.41)
        for i in range(4):
            _record(d, 8 + i, emotion=_make_emotion(joy=0.4 + 0.01 * (i % 2)))

        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

    def test_drive_convergence_detected(self):
        """駆動ベクトルの散らばりの単調減少が検出される。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # サブウィンドウ1: 大きな分散
        for i in range(4):
            _record(d, i, drive=_make_drive(social=0.8 * (i % 2)))

        # サブウィンドウ2: 中程度の分散
        for i in range(4):
            _record(d, 4 + i, drive=_make_drive(social=0.3 + 0.2 * (i % 2)))

        # サブウィンドウ3: 小さな分散
        for i in range(4):
            _record(d, 8 + i, drive=_make_drive(social=0.4 + 0.01 * (i % 2)))

        assert d.gradual_convergence_flags[SIGNAL_DRIVE] is True

    def test_no_convergence_when_variance_increases(self):
        """分散が増加する場合は収束が検出されない。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # サブウィンドウ1: 小さな分散
        for i in range(4):
            _record(d, i, emotion=_make_emotion(joy=0.4 + 0.01 * (i % 2)))

        # サブウィンドウ2: 中程度の分散
        for i in range(4):
            _record(d, 4 + i, emotion=_make_emotion(joy=0.3 + 0.2 * (i % 2)))

        # サブウィンドウ3: 大きな分散
        for i in range(4):
            _record(d, 8 + i, emotion=_make_emotion(joy=0.8 * (i % 2)))

        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False

    def test_no_convergence_when_constant_variance(self):
        """全サブウィンドウの分散が同一で非ゼロの場合はTrue(非増加)。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 全サブウィンドウで同じパターン
        for sw in range(3):
            for i in range(4):
                _record(
                    d, sw * 4 + i,
                    emotion=_make_emotion(joy=0.3 + 0.2 * (i % 2))
                )

        # 全てのサブウィンドウで同じ分散→非増加→True
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

    def test_all_zero_variance_not_convergence(self):
        """全サブウィンドウの分散がゼロの場合は収束と判定されない。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 全サブウィンドウで同一値
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=0.5))

        # 全て同一値→各サブウィンドウの分散はゼロ→全ゼロ→False
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False

    def test_independent_signal_tracking(self):
        """感情と駆動の収束検出が独立して動作する。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 感情は収束、駆動は発散
        for sw in range(3):
            for i in range(4):
                # 感情: 分散が減少
                e_spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(joy=0.5 + e_spread * (1 if i % 2 == 0 else -1))

                # 駆動: 分散が増加
                d_spread = 0.05 + sw * 0.15
                drive = _make_drive(social=0.5 + d_spread * (1 if i % 2 == 0 else -1))

                _record(d, sw * 4 + i, emotion=emotion, drive=drive)

        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True
        assert d.gradual_convergence_flags[SIGNAL_DRIVE] is False


# ── 遷移検出とログ出力テスト ──────────────────────────────────


class TestGradualConvergenceAlerts:
    """緩やかな収束の遷移検出とログ出力テスト。"""

    def _make_converging_detector(self) -> AnomalyDetector:
        """収束が検出された状態のdetectorを作成するヘルパー。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        return d

    def test_convergence_detected_log_emitted(self, caplog):
        """収束進行への遷移時に検出ログが出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            d = self._make_converging_detector()

        convergence_logs = [
            r for r in caplog.records
            if "gradual_convergence_detected" in r.getMessage()
        ]
        assert len(convergence_logs) > 0

    def test_convergence_detected_log_format(self, caplog):
        """収束検出ログのフォーマットが正しい。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            d = self._make_converging_detector()

        convergence_logs = [
            r for r in caplog.records
            if "gradual_convergence_detected" in r.getMessage()
        ]
        for log_record in convergence_logs:
            parsed = json.loads(log_record.getMessage())
            assert parsed["type"] == "gradual_convergence_detected"
            assert "signal" in parsed
            assert "tick_number" in parsed
            assert "sub_window_count" in parsed
            assert "variance_series" in parsed
            assert isinstance(parsed["variance_series"], list)

    def test_convergence_resolved_log_emitted(self, caplog):
        """収束解消時に解消ログが出力される。"""
        d = self._make_converging_detector()
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            caplog.clear()
            # 分散が増加するパターンを注入して収束を解消
            # サブウィンドウ1: 小さい分散, サブウィンドウ2: 中, サブウィンドウ3: 大
            for sw in range(3):
                for i in range(4):
                    spread = 0.05 + sw * 0.2
                    emotion = _make_emotion(
                        joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                    )
                    _record(d, 100 + sw * 4 + i, emotion=emotion)

        resolved_logs = [
            r for r in caplog.records
            if "gradual_convergence_resolved" in r.getMessage()
        ]
        assert len(resolved_logs) >= 1

    def test_convergence_resolved_log_format(self, caplog):
        """解消ログのフォーマットが正しい。"""
        d = self._make_converging_detector()

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            caplog.clear()
            # 分散が増加するパターンを注入
            for sw in range(3):
                for i in range(4):
                    spread = 0.05 + sw * 0.2
                    emotion = _make_emotion(
                        joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                    )
                    _record(d, 100 + sw * 4 + i, emotion=emotion)

        resolved_logs = [
            r for r in caplog.records
            if "gradual_convergence_resolved" in r.getMessage()
        ]
        for log_record in resolved_logs:
            parsed = json.loads(log_record.getMessage())
            assert parsed["type"] == "gradual_convergence_resolved"
            assert "signal" in parsed
            assert "tick_number" in parsed

    def test_no_repeated_convergence_log(self, caplog):
        """収束継続中は同一のログを出力しない(冗長抑制)。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            d = self._make_converging_detector()
            first_count = len([
                r for r in caplog.records
                if "gradual_convergence_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ])

            # 収束継続中にさらに収束パターンを追加
            caplog.clear()
            for i in range(4):
                _record(d, 20 + i, emotion=_make_emotion(joy=0.5 + 0.001 * (i % 2)))

            repeated_logs = [
                r for r in caplog.records
                if "gradual_convergence_detected" in r.getMessage()
                and SIGNAL_EMOTION in r.getMessage()
            ]
            # 継続中はログが出力されない
            assert len(repeated_logs) == 0


# ── カウンタテスト ──────────────────────────────────────────────


class TestGradualConvergenceCounters:
    """緩やかな収束のカウンタテスト。"""

    def test_detected_count_increments(self):
        """収束検出でカウンタが増加する。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_detected_count[SIGNAL_EMOTION] == 1

    def test_resolved_count_increments(self):
        """収束解消でカウンタが増加する。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        # 収束を発生させる
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

        # 解消: 分散が増加するパターン
        for sw in range(3):
            for i in range(4):
                spread = 0.05 + sw * 0.2
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, 100 + sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_resolved_count[SIGNAL_EMOTION] >= 1

    def test_multiple_detect_resolve_cycles(self):
        """検出→解消→検出でカウンタが累積する。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 第1サイクル: 収束(分散が減少)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_detected_count[SIGNAL_EMOTION] == 1

        # 解消(分散が増加)
        for sw in range(3):
            for i in range(4):
                spread = 0.05 + sw * 0.2
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, 100 + sw * 4 + i, emotion=emotion)

        # 第2サイクル: 再び収束(分散が減少)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.3 - sw * 0.1)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, 200 + sw * 4 + i, emotion=emotion)

        assert d.gradual_convergence_detected_count[SIGNAL_EMOTION] >= 2


# ── 散らばり列テスト ──────────────────────────────────────────────


class TestVarianceSeries:
    """最新の散らばり列のテスト。"""

    def test_variance_series_populated(self):
        """検出実行後に散らばり列が記録される。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        series = d.latest_variance_series
        assert len(series[SIGNAL_EMOTION]) == 3

    def test_variance_series_length_equals_sub_window_count(self):
        """散らばり列の長さがサブウィンドウ分割数と一致する。"""
        for swc in [2, 3, 4]:
            d = AnomalyDetector(buffer_max=swc * 4, enabled=True, sub_window_count=swc)
            for i in range(swc * 4):
                _record(d, i, emotion=_make_emotion(joy=float(i) / (swc * 4)))
            series = d.latest_variance_series
            assert len(series[SIGNAL_EMOTION]) == swc

    def test_variance_series_readonly(self):
        """散らばり列が読み取り専用コピーであること。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        s1 = d.latest_variance_series
        s2 = d.latest_variance_series
        assert s1 is not s2
        assert s1[SIGNAL_EMOTION] is not s2[SIGNAL_EMOTION]

    def test_variance_series_values_are_nonnegative(self):
        """散らばり列の全要素が非負であること。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        series = d.latest_variance_series
        for val in series[SIGNAL_EMOTION]:
            assert val >= 0.0

    def test_variance_series_in_get_summary(self):
        """get_summaryに散らばり列が含まれる。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        summary = d.get_summary()
        assert "latest_variance_series" in summary
        assert SIGNAL_EMOTION in summary["latest_variance_series"]


# ── セッションサマリテスト ──────────────────────────────────────


class TestGradualConvergenceSessionSummary:
    """緩やかな収束のセッションサマリテスト。"""

    def test_session_summary_includes_gradual_convergence(self):
        """セッションサマリに緩やかな収束情報が含まれる。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        summary = d.emit_session_summary()
        assert summary is not None
        assert "sub_window_count" in summary
        assert "gradual_convergence_detected_counts" in summary
        assert "gradual_convergence_resolved_counts" in summary
        assert "current_gradual_convergence_flags" in summary

    def test_session_summary_gradual_convergence_values(self):
        """セッションサマリの緩やかな収束情報が正しい値を含む。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        # 収束を発生させる
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)

        summary = d.emit_session_summary()
        assert summary is not None
        assert summary["sub_window_count"] == 3
        assert summary["gradual_convergence_detected_counts"][SIGNAL_EMOTION] == 1


# ── get_summary テスト ──────────────────────────────────────────


class TestGradualConvergenceGetSummary:
    """get_summaryの緩やかな収束情報テスト。"""

    def test_get_summary_includes_gradual_convergence(self):
        """get_summaryに緩やかな収束情報が含まれる。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        summary = d.get_summary()
        assert "sub_window_count" in summary
        assert "gradual_convergence_detected_counts" in summary
        assert "gradual_convergence_resolved_counts" in summary
        assert "current_gradual_convergence_flags" in summary
        assert "latest_variance_series" in summary

    def test_get_summary_returns_copies(self):
        """get_summaryが返す辞書がコピーであること。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        s1 = d.get_summary()
        s2 = d.get_summary()
        assert s1["gradual_convergence_detected_counts"] is not \
            s2["gradual_convergence_detected_counts"]
        assert s1["latest_variance_series"] is not s2["latest_variance_series"]


# ── プロパティの読み取り専用テスト ──────────────────────────────


class TestGradualConvergencePropertyReadonly:
    """緩やかな収束のプロパティが読み取り専用コピーを返すことのテスト。"""

    def test_convergence_flags_readonly(self):
        """gradual_convergence_flagsの変更が内部状態に影響しない。"""
        d = AnomalyDetector(enabled=True)
        flags = d.gradual_convergence_flags
        flags[SIGNAL_EMOTION] = True
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False

    def test_convergence_detected_count_readonly(self):
        """gradual_convergence_detected_countの変更が内部状態に影響しない。"""
        d = AnomalyDetector(enabled=True)
        counts = d.gradual_convergence_detected_count
        counts[SIGNAL_EMOTION] = 99
        assert d.gradual_convergence_detected_count[SIGNAL_EMOTION] == 0

    def test_convergence_resolved_count_readonly(self):
        """gradual_convergence_resolved_countの変更が内部状態に影響しない。"""
        d = AnomalyDetector(enabled=True)
        counts = d.gradual_convergence_resolved_count
        counts[SIGNAL_EMOTION] = 99
        assert d.gradual_convergence_resolved_count[SIGNAL_EMOTION] == 0

    def test_variance_series_readonly(self):
        """latest_variance_seriesの変更が内部状態に影響しない。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        s = d.latest_variance_series
        s[SIGNAL_EMOTION] = [999.0]
        assert d.latest_variance_series[SIGNAL_EMOTION] != [999.0]


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestGradualConvergenceSafetyValves:
    """緩やかな固定化検出の安全弁テスト。"""

    def test_disabled_skips_convergence_detection(self):
        """無効時は緩やかな収束検出が行われない(安全弁2)。"""
        d = AnomalyDetector(buffer_max=12, enabled=False, sub_window_count=3)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        for signal in _GRADUAL_CONVERGENCE_SIGNALS:
            assert d.gradual_convergence_flags[signal] is False

    def test_no_psyche_state_modification(self):
        """検出結果がpsycheの状態を変更しない(安全弁5)。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        # 修復・介入メソッドが存在しない
        public_methods = [m for m in dir(d) if not m.startswith("_")]
        assert "repair_convergence" not in public_methods
        assert "fix_convergence" not in public_methods
        assert "reset_convergence" not in public_methods
        assert "modify_convergence" not in public_methods

    def test_no_save_load_for_convergence(self):
        """緩やかな収束の状態にsave/loadインタフェースがない(安全弁6)。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        assert not hasattr(d, "save")
        assert not hasattr(d, "load")
        assert not hasattr(d, "to_dict")
        assert not hasattr(d, "from_dict")

    def test_exception_in_convergence_does_not_propagate(self):
        """収束検出中の例外が伝播しない(安全弁1)。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        # 正常なスナップショットを蓄積
        for i in range(11):
            _record(d, i, emotion=_make_emotion(joy=float(i) / 12))
        # 不正な型のemotion_valuesを持つスナップショットを手動注入しても
        # 安全弁1で捕捉される(record_snapshotが例外を握りつぶす)
        d.record_snapshot(
            tick_number=11,
            emotion_values=None,  # type: ignore
            drive_values=_make_drive(),
            return_pathway_fired=False,
            enrichment_variation_count=0,
        )
        # テストが到達すればOK(例外が伝播していない)

    def test_convergence_flag_auto_resolves(self):
        """収束フラグは散らばりの回復により自動解消される(可逆性)。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 収束を発生させる
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True

        # 散らばりを増加させて解消(分散が増加するパターン)
        for sw in range(3):
            for i in range(4):
                spread = 0.05 + sw * 0.2
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, 100 + sw * 4 + i, emotion=emotion)
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False


# ── エンドツーエンドテスト ──────────────────────────────────────


class TestGradualConvergenceEndToEnd:
    """緩やかな収束のエンドツーエンドシナリオテスト。"""

    def test_full_lifecycle(self, caplog):
        """収束検出→継続→解消→再収束のフルライフサイクル。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.anomaly_detection"):
            # Phase 1: 分散が減少→収束検出
            for sw in range(3):
                for i in range(4):
                    spread = max(0, 0.4 - sw * 0.15)
                    emotion = _make_emotion(
                        joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                    )
                    _record(d, sw * 4 + i, emotion=emotion)
            assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True
            assert d.gradual_convergence_detected_count[SIGNAL_EMOTION] == 1

            # Phase 2: 分散を増加させて解消
            for sw in range(3):
                for i in range(4):
                    spread = 0.05 + sw * 0.2
                    emotion = _make_emotion(
                        joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                    )
                    _record(d, 100 + sw * 4 + i, emotion=emotion)
            assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False
            assert d.gradual_convergence_resolved_count[SIGNAL_EMOTION] >= 1

        # セッションサマリの確認
        summary = d.emit_session_summary()
        assert summary is not None
        assert summary["gradual_convergence_detected_counts"][SIGNAL_EMOTION] >= 1

    def test_coexistence_with_stall_detection(self):
        """動態停止検出と緩やかな収束検出が独立して共存する。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        # 全て同一値→停止も検出、分散は全ゼロ→収束は非検出
        for i in range(12):
            _record(d, i, emotion=_make_emotion(joy=0.5))

        assert d.stall_flags[SIGNAL_EMOTION] is True
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is False

    def test_convergence_without_stall(self):
        """値は変化しているが分散が減少→収束検出、停止は非検出。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)

        for sw in range(3):
            for i in range(4):
                spread = max(0.001, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)

        assert d.stall_flags[SIGNAL_EMOTION] is False  # 値は変化している
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True  # 分散は減少

    def test_structural_isolation(self):
        """構造的分離の検証: 収束検出はREAD-ONLYで介入しない。"""
        d = AnomalyDetector(buffer_max=12, enabled=True, sub_window_count=3)
        for sw in range(3):
            for i in range(4):
                spread = max(0, 0.4 - sw * 0.15)
                emotion = _make_emotion(
                    joy=0.5 + spread * (1 if i % 2 == 0 else -1)
                )
                _record(d, sw * 4 + i, emotion=emotion)

        # get_summaryはREAD-ONLY
        summary1 = d.get_summary()
        summary2 = d.get_summary()
        assert summary1["gradual_convergence_detected_counts"] == \
            summary2["gradual_convergence_detected_counts"]

        # emit_session_summaryは内部状態を変更しない
        pre_flags = d.gradual_convergence_flags
        d.emit_session_summary()
        post_flags = d.gradual_convergence_flags
        assert pre_flags == post_flags

    def test_existing_stall_detection_unaffected(self):
        """既存の停止検出が緩やかな収束検出の追加で影響を受けない。"""
        d = AnomalyDetector(buffer_max=5, enabled=True, sub_window_count=3)

        # 既存の停止検出テストと同様のシナリオ
        for i in range(5):
            _record(d, i, emotion=_make_emotion())
        assert d.stall_flags[SIGNAL_EMOTION] is True

        # 解消
        _record(d, 10, emotion=_make_emotion(joy=0.9))
        assert d.stall_flags[SIGNAL_EMOTION] is False

    def test_multi_dimension_variance(self):
        """複数次元の分散が合算されることの検証。"""
        d = AnomalyDetector(buffer_max=6, enabled=True, sub_window_count=3)

        # サブウィンドウ1: joyとangerが大きく変動
        _record(d, 0, emotion=_make_emotion(joy=0.0, anger=0.0))
        _record(d, 1, emotion=_make_emotion(joy=0.8, anger=0.8))

        # サブウィンドウ2: joyとangerが中程度に変動
        _record(d, 2, emotion=_make_emotion(joy=0.3, anger=0.3))
        _record(d, 3, emotion=_make_emotion(joy=0.5, anger=0.5))

        # サブウィンドウ3: joyとangerがほぼ変動なし
        _record(d, 4, emotion=_make_emotion(joy=0.4, anger=0.4))
        _record(d, 5, emotion=_make_emotion(joy=0.41, anger=0.41))

        # 全次元の分散の合計が単調減少
        assert d.gradual_convergence_flags[SIGNAL_EMOTION] is True
        # 散らばり列が3要素
        series = d.latest_variance_series
        assert len(series[SIGNAL_EMOTION]) == 3
        # 各要素が非負
        for val in series[SIGNAL_EMOTION]:
            assert val >= 0.0
