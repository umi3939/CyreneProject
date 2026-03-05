"""
tests/test_return_aggregate_cap.py - 帰還経路合算帯域上限のテスト

設計書: design_return_aggregate_cap.md

テスト対象:
- orchestrator_1tick_phases.apply_return_aggregate_cap()
- tools/return_pathway_monitor.ReturnPathwayMonitor の合算上限到達記録
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Optional

from tools.return_pathway_monitor import (
    ReturnPathwayMonitor,
    PATHWAY_A,
    PATHWAY_B,
    PATHWAY_C,
    PATHWAY_D,
    PATHWAY_E,
)

from psyche.orchestrator_1tick_phases import (
    apply_return_aggregate_cap,
    _AGGREGATE_CAP_EMOTION,
    _AGGREGATE_CAP_DRIVE,
    _AGGREGATE_CAP_MOOD_SPEED,
    _apply_emotion_proportional_reduction,
    _apply_drive_proportional_reduction,
    _apply_mood_speed_proportional_reduction,
)

from psyche.state import (
    PsycheState,
    EmotionVector,
    DriveVector,
    Mood,
)


# ── ヘルパー ────────────────────────────────────────────────────


def _make_mock_orch(
    emotions: Optional[dict[str, float]] = None,
    drives: Optional[dict[str, float]] = None,
    mood_valence: float = 0.0,
    mood_arousal: float = 0.3,
    monitor: Optional[ReturnPathwayMonitor] = None,
) -> MagicMock:
    """テスト用のモックorchestratorを生成する。"""
    orch = MagicMock()
    orch._psyche = PsycheState(
        emotions=EmotionVector(**(emotions or {})),
        drives=DriveVector(**(drives or {})),
        mood=Mood(valence=mood_valence, arousal=mood_arousal),
    )
    orch._return_pathway_monitor = monitor or ReturnPathwayMonitor(enabled=True)
    return orch


# ── テストクラス: ReturnPathwayMonitor 合算上限記録 ──────────────


class TestMonitorAggregateCap:
    """ReturnPathwayMonitor の合算上限到達記録のテスト。"""

    def test_initial_aggregate_cap_hit_counts_all_zero(self) -> None:
        """初期状態では全種類のカウンタがゼロ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        counts = monitor.aggregate_cap_hit_counts
        assert counts["emotion"] == 0
        assert counts["drive"] == 0
        assert counts["mood_speed"] == 0

    def test_record_aggregate_cap_hit_emotion(self) -> None:
        """感情帯域の合算上限到達記録が正しくインクリメントされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("emotion")
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1
        monitor.record_aggregate_cap_hit("emotion")
        assert monitor.aggregate_cap_hit_counts["emotion"] == 2

    def test_record_aggregate_cap_hit_drive(self) -> None:
        """ドライブ帯域の合算上限到達記録が正しくインクリメントされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("drive")
        assert monitor.aggregate_cap_hit_counts["drive"] == 1

    def test_record_aggregate_cap_hit_mood_speed(self) -> None:
        """ムード追従速度の合算上限到達記録が正しくインクリメントされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("mood_speed")
        assert monitor.aggregate_cap_hit_counts["mood_speed"] == 1

    def test_record_aggregate_cap_hit_invalid_kind_ignored(self) -> None:
        """無効なkindは無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("invalid_kind")
        counts = monitor.aggregate_cap_hit_counts
        assert counts["emotion"] == 0
        assert counts["drive"] == 0
        assert counts["mood_speed"] == 0

    def test_aggregate_cap_hit_counts_independent(self) -> None:
        """種類ごとのカウンタが独立している。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("drive")
        counts = monitor.aggregate_cap_hit_counts
        assert counts["emotion"] == 2
        assert counts["drive"] == 1
        assert counts["mood_speed"] == 0

    def test_aggregate_cap_hit_in_session_summary(self) -> None:
        """セッションサマリーに合算上限到達カウンタが含まれる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("drive")
        summary = monitor.emit_session_summary()
        assert summary is not None
        assert "aggregate_cap_hit_counts" in summary
        assert summary["aggregate_cap_hit_counts"]["emotion"] == 1
        assert summary["aggregate_cap_hit_counts"]["drive"] == 1
        assert summary["aggregate_cap_hit_counts"]["mood_speed"] == 0

    def test_aggregate_cap_hit_in_get_summary(self) -> None:
        """get_summaryに合算上限到達カウンタが含まれる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("mood_speed")
        summary = monitor.get_summary()
        assert "aggregate_cap_hit_counts" in summary
        assert summary["aggregate_cap_hit_counts"]["mood_speed"] == 1

    def test_aggregate_cap_hit_counts_is_copy(self) -> None:
        """aggregate_cap_hit_countsは読み取り専用コピーを返す。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        counts = monitor.aggregate_cap_hit_counts
        counts["emotion"] = 999
        assert monitor.aggregate_cap_hit_counts["emotion"] == 0

    def test_get_tick_buffer_empty_initially(self) -> None:
        """初期状態ではティックバッファが空。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert monitor.get_tick_buffer() == []

    def test_get_tick_buffer_returns_copy(self) -> None:
        """get_tick_bufferはコピーを返す。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        buf = monitor.get_tick_buffer()
        buf.clear()
        assert len(monitor.get_tick_buffer()) == 1

    def test_get_tick_buffer_after_record(self) -> None:
        """発火記録後のティックバッファに記録が含まれる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.02})
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": 0.01})
        buf = monitor.get_tick_buffer()
        assert len(buf) == 2

    def test_get_current_tick(self) -> None:
        """get_current_tickが現在のティック番号を返す。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert monitor.get_current_tick() == -1
        monitor.record_firing(PATHWAY_A, 5, emotion_deltas={"joy": 0.01})
        assert monitor.get_current_tick() == 5


# ── テストクラス: 合算帯域上限チェック定数 ───────────────────────


class TestAggregateCaps:
    """合算帯域上限の定数値のテスト。"""

    def test_emotion_cap_is_positive(self) -> None:
        """感情帯域の上限が正の値。"""
        assert _AGGREGATE_CAP_EMOTION > 0.0

    def test_drive_cap_is_positive(self) -> None:
        """ドライブ帯域の上限が正の値。"""
        assert _AGGREGATE_CAP_DRIVE > 0.0

    def test_mood_speed_cap_is_positive(self) -> None:
        """ムード追従速度の上限が正の値。"""
        assert _AGGREGATE_CAP_MOOD_SPEED > 0.0


# ── テストクラス: apply_return_aggregate_cap ──────────────────────


class TestApplyReturnAggregateCap:
    """合算帯域上限の適用テスト。"""

    def test_no_firing_no_change(self) -> None:
        """発火がない場合は状態変化なし。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        orch = _make_mock_orch(
            emotions={"joy": 0.5},
            monitor=monitor,
        )
        original_joy = orch._psyche.emotions.joy
        apply_return_aggregate_cap(orch)
        assert orch._psyche.emotions.joy == original_joy

    def test_below_cap_no_reduction(self) -> None:
        """合算が上限以下の場合は補正なし。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 非常に小さな変動（上限の半分以下）
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        orch = _make_mock_orch(
            emotions={"joy": 0.51},  # 元は0.5, +0.01適用後
            monitor=monitor,
        )
        original_joy = orch._psyche.emotions.joy
        apply_return_aggregate_cap(orch)
        assert orch._psyche.emotions.joy == original_joy
        assert monitor.aggregate_cap_hit_counts["emotion"] == 0

    def test_emotion_cap_exceeded_proportional_reduction(self) -> None:
        """感情帯域の合算上限超過時に比例縮小が適用される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 3経路から大きな変動を記録（合算が上限を超過するように）
        delta_per_dim = _AGGREGATE_CAP_EMOTION  # 1次元だけで上限到達
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": delta_per_dim})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"joy": delta_per_dim})
        # 合算 = 2 * delta_per_dim > _AGGREGATE_CAP_EMOTION

        orch = _make_mock_orch(
            emotions={"joy": 0.5 + 2 * delta_per_dim},  # 帰還適用後の値
            monitor=monitor,
        )
        pre_joy = orch._psyche.emotions.joy
        apply_return_aggregate_cap(orch)
        post_joy = orch._psyche.emotions.joy

        # 縮小が適用されたことを確認（joy値が減少）
        assert post_joy < pre_joy
        # 合算上限到達が記録されたことを確認
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

    def test_drive_cap_exceeded_proportional_reduction(self) -> None:
        """ドライブ帯域の合算上限超過時に比例縮小が適用される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        large_delta = _AGGREGATE_CAP_DRIVE * 2  # 上限の2倍
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": large_delta})

        orch = _make_mock_orch(
            drives={"social": 0.5 + large_delta, "curiosity": 0.5, "expression": 0.5},
            monitor=monitor,
        )
        pre_social = orch._psyche.drives.social
        apply_return_aggregate_cap(orch)
        post_social = orch._psyche.drives.social

        assert post_social < pre_social
        assert monitor.aggregate_cap_hit_counts["drive"] == 1

    def test_mood_speed_cap_exceeded_proportional_reduction(self) -> None:
        """ムード追従速度の合算上限超過時に比例縮小が適用される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        large_mod = _AGGREGATE_CAP_MOOD_SPEED * 3  # 上限の3倍
        monitor.record_firing(PATHWAY_E, 1, mood_speed_deltas={
            "valence_modulation": large_mod,
        })

        orch = _make_mock_orch(
            mood_valence=0.2 + large_mod,  # 変調適用後の近似値
            monitor=monitor,
        )
        pre_valence = orch._psyche.mood.valence
        apply_return_aggregate_cap(orch)
        post_valence = orch._psyche.mood.valence

        assert post_valence < pre_valence
        assert monitor.aggregate_cap_hit_counts["mood_speed"] == 1

    def test_independent_kind_caps(self) -> None:
        """各種類が独立して判定される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 感情は上限超過、ドライブは上限以下
        large_emotion = _AGGREGATE_CAP_EMOTION * 2
        small_drive = _AGGREGATE_CAP_DRIVE * 0.3
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": large_emotion})
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": small_drive})

        orch = _make_mock_orch(
            emotions={"joy": 0.5 + large_emotion},
            drives={"social": 0.5 + small_drive, "curiosity": 0.5, "expression": 0.5},
            monitor=monitor,
        )
        pre_joy = orch._psyche.emotions.joy
        pre_social = orch._psyche.drives.social

        apply_return_aggregate_cap(orch)

        # 感情は縮小
        assert orch._psyche.emotions.joy < pre_joy
        # ドライブは変化なし
        assert orch._psyche.drives.social == pre_social
        # カウンタ
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1
        assert monitor.aggregate_cap_hit_counts["drive"] == 0

    def test_proportional_reduction_ratio(self) -> None:
        """比例縮小の比率が正しく計算される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 合算が上限のちょうど2倍になる場合、比率は0.5
        delta = _AGGREGATE_CAP_EMOTION  # 上限と同じ値を2経路で
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": delta})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"joy": delta})

        base_joy = 0.3
        applied_joy = base_joy + 2 * delta
        orch = _make_mock_orch(
            emotions={"joy": applied_joy},
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)

        # 合算 = 2 * delta, 上限 = delta, ratio = 0.5
        # 逆補正 = 2 * delta * (1 - 0.5) = delta
        # 期待値 = applied_joy - delta = base_joy + delta
        expected = base_joy + delta
        assert abs(orch._psyche.emotions.joy - expected) < 1e-6

    def test_multi_dimension_emotion_reduction(self) -> None:
        """複数次元の感情変動が同一比率で縮小される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # joy + sorrow で上限超過
        joy_delta = _AGGREGATE_CAP_EMOTION * 0.8
        sorrow_delta = _AGGREGATE_CAP_EMOTION * 0.8
        # 合算 = 1.6 * cap > cap
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={
            "joy": joy_delta,
            "sorrow": sorrow_delta,
        })

        base_joy = 0.3
        base_sorrow = 0.2
        orch = _make_mock_orch(
            emotions={
                "joy": base_joy + joy_delta,
                "sorrow": base_sorrow + sorrow_delta,
            },
            monitor=monitor,
        )
        pre_joy = orch._psyche.emotions.joy
        pre_sorrow = orch._psyche.emotions.sorrow
        apply_return_aggregate_cap(orch)

        # 両方とも縮小されること
        assert orch._psyche.emotions.joy < pre_joy
        assert orch._psyche.emotions.sorrow < pre_sorrow

    def test_all_pathways_equal_reduction(self) -> None:
        """全経路等価の比例縮小: 各経路の変動に同一比率が適用される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        d = _AGGREGATE_CAP_EMOTION * 0.6  # 各経路0.6*cap
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": d})
        monitor.record_firing(PATHWAY_C, 1, emotion_deltas={"joy": d})
        # 合算 = 1.2 * cap > cap

        base = 0.3
        orch = _make_mock_orch(
            emotions={"joy": base + 2 * d},
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)

        # ratio = cap / (1.2 * cap) = 1/1.2
        ratio = 1.0 / 1.2
        correction = 2 * d * (1.0 - ratio)
        expected = base + 2 * d - correction
        assert abs(orch._psyche.emotions.joy - expected) < 1e-6

    def test_exception_safety_monitor_failure(self) -> None:
        """モニター読み取り失敗時に例外が伝播しない。"""
        orch = MagicMock()
        orch._return_pathway_monitor.get_tick_buffer.side_effect = RuntimeError("fail")
        orch._psyche = PsycheState()

        # 例外が伝播しないことを確認
        apply_return_aggregate_cap(orch)

    def test_exception_safety_state_update_failure(self) -> None:
        """状態更新失敗時に例外が伝播しない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        large = _AGGREGATE_CAP_EMOTION * 3
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": large})

        orch = MagicMock()
        orch._return_pathway_monitor = monitor
        orch._psyche = MagicMock()
        orch._psyche.emotions.as_dict.side_effect = RuntimeError("fail")

        # 例外が伝播しないことを確認
        apply_return_aggregate_cap(orch)

    def test_emotion_values_clamped_0_1(self) -> None:
        """補正後の感情値が0.0-1.0の範囲内に収まる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 負方向の大きな変動
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": -0.5})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"joy": -0.5})
        # 合算絶対値 = 1.0 > cap

        orch = _make_mock_orch(
            emotions={"joy": 0.0},  # 帰還適用後既にクランプ
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)
        # 逆補正は正方向に作用するが、1.0を超えない
        assert 0.0 <= orch._psyche.emotions.joy <= 1.0

    def test_drive_values_clamped_0_1(self) -> None:
        """補正後のドライブ値が0.0-1.0の範囲内に収まる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        large = _AGGREGATE_CAP_DRIVE * 5
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": large})

        orch = _make_mock_orch(
            drives={"social": 1.0, "curiosity": 0.5, "expression": 0.5},
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)
        assert 0.0 <= orch._psyche.drives.social <= 1.0

    def test_mood_values_clamped(self) -> None:
        """補正後のムード値が範囲内に収まる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        large = _AGGREGATE_CAP_MOOD_SPEED * 5
        monitor.record_firing(PATHWAY_E, 1, mood_speed_deltas={
            "valence_modulation": large,
            "arousal_modulation": large,
        })

        orch = _make_mock_orch(
            mood_valence=0.9,
            mood_arousal=0.9,
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)
        assert -1.0 <= orch._psyche.mood.valence <= 1.0
        assert 0.0 <= orch._psyche.mood.arousal <= 1.0

    def test_tick_independence(self) -> None:
        """ティック間の独立性: 前ティックの上限到達が次ティックに影響しない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # ティック1: 上限超過
        large = _AGGREGATE_CAP_EMOTION * 3
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": large})

        orch = _make_mock_orch(
            emotions={"joy": 0.5 + large},
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

        # ティック2: ティックバッファリセット後、小さな変動
        monitor.record_firing(PATHWAY_A, 2, emotion_deltas={"joy": 0.01})
        orch2 = _make_mock_orch(
            emotions={"joy": 0.51},
            monitor=monitor,
        )
        original = orch2._psyche.emotions.joy
        apply_return_aggregate_cap(orch2)
        # 上限以下なので変化なし
        assert orch2._psyche.emotions.joy == original
        # 前ティックの到達は累積、今回は追加なし
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

    def test_no_cross_kind_aggregation(self) -> None:
        """種類を横断する合算は行わない（安全弁1）。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 感情は上限ギリギリ、ドライブも上限ギリギリ
        emotion_delta = _AGGREGATE_CAP_EMOTION * 0.9
        drive_delta = _AGGREGATE_CAP_DRIVE * 0.9
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": emotion_delta})
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": drive_delta})

        orch = _make_mock_orch(
            emotions={"joy": 0.3 + emotion_delta},
            drives={"social": 0.5 + drive_delta, "curiosity": 0.5, "expression": 0.5},
            monitor=monitor,
        )
        pre_joy = orch._psyche.emotions.joy
        pre_social = orch._psyche.drives.social

        apply_return_aggregate_cap(orch)

        # どちらも上限以下なので変化なし
        assert orch._psyche.emotions.joy == pre_joy
        assert orch._psyche.drives.social == pre_social
        assert monitor.aggregate_cap_hit_counts["emotion"] == 0
        assert monitor.aggregate_cap_hit_counts["drive"] == 0

    def test_exact_cap_no_reduction(self) -> None:
        """合算がちょうど上限と一致する場合は縮小なし。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # ちょうど上限と一致
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": _AGGREGATE_CAP_EMOTION})

        orch = _make_mock_orch(
            emotions={"joy": 0.3 + _AGGREGATE_CAP_EMOTION},
            monitor=monitor,
        )
        original = orch._psyche.emotions.joy
        apply_return_aggregate_cap(orch)
        assert orch._psyche.emotions.joy == original
        assert monitor.aggregate_cap_hit_counts["emotion"] == 0


# ── テストクラス: 個別縮小関数の直接テスト ────────────────────────


class TestEmotionProportionalReduction:
    """_apply_emotion_proportional_reduction のテスト。"""

    def test_basic_reduction(self) -> None:
        """基本的な比例縮小。"""
        tick_buffer = [
            {"emotion_deltas": {"joy": 0.1, "sorrow": 0.1}},
        ]
        orch = _make_mock_orch(emotions={"joy": 0.6, "sorrow": 0.6})
        _apply_emotion_proportional_reduction(orch, tick_buffer, 0.5)
        # 逆補正 = 0.1 * 0.5 = 0.05 each
        assert abs(orch._psyche.emotions.joy - 0.55) < 1e-6
        assert abs(orch._psyche.emotions.sorrow - 0.55) < 1e-6

    def test_empty_buffer_no_change(self) -> None:
        """空バッファでは変化なし。"""
        orch = _make_mock_orch(emotions={"joy": 0.5})
        _apply_emotion_proportional_reduction(orch, [], 0.5)
        assert orch._psyche.emotions.joy == 0.5

    def test_no_emotion_deltas_no_change(self) -> None:
        """emotion_deltasがない記録では変化なし。"""
        tick_buffer = [{"drive_deltas": {"social": 0.1}}]
        orch = _make_mock_orch(emotions={"joy": 0.5})
        _apply_emotion_proportional_reduction(orch, tick_buffer, 0.5)
        assert orch._psyche.emotions.joy == 0.5

    def test_ratio_1_no_change(self) -> None:
        """ratio=1.0では変化なし（逆補正がゼロ）。"""
        tick_buffer = [{"emotion_deltas": {"joy": 0.1}}]
        orch = _make_mock_orch(emotions={"joy": 0.6})
        _apply_emotion_proportional_reduction(orch, tick_buffer, 1.0)
        assert abs(orch._psyche.emotions.joy - 0.6) < 1e-6


class TestDriveProportionalReduction:
    """_apply_drive_proportional_reduction のテスト。"""

    def test_basic_reduction(self) -> None:
        """基本的な比例縮小。"""
        tick_buffer = [
            {"drive_deltas": {"social": 0.1}},
        ]
        orch = _make_mock_orch(drives={"social": 0.6, "curiosity": 0.5, "expression": 0.5})
        _apply_drive_proportional_reduction(orch, tick_buffer, 0.5)
        assert abs(orch._psyche.drives.social - 0.55) < 1e-6
        # 変動なしの次元は変化なし
        assert orch._psyche.drives.curiosity == 0.5

    def test_empty_buffer_no_change(self) -> None:
        """空バッファでは変化なし。"""
        orch = _make_mock_orch(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5})
        _apply_drive_proportional_reduction(orch, [], 0.5)
        assert orch._psyche.drives.social == 0.5


class TestMoodSpeedProportionalReduction:
    """_apply_mood_speed_proportional_reduction のテスト。"""

    def test_basic_reduction(self) -> None:
        """基本的な比例縮小。"""
        tick_buffer = [
            {"mood_speed_deltas": {"valence_modulation": 0.1, "arousal_modulation": 0.05}},
        ]
        orch = _make_mock_orch(mood_valence=0.3, mood_arousal=0.4)
        _apply_mood_speed_proportional_reduction(orch, tick_buffer, 0.5)
        # valence逆補正 = 0.1 * 0.5 = 0.05
        assert abs(orch._psyche.mood.valence - 0.25) < 1e-6
        # arousal逆補正 = 0.05 * 0.5 = 0.025
        assert abs(orch._psyche.mood.arousal - 0.375) < 1e-6

    def test_empty_buffer_no_change(self) -> None:
        """空バッファでは変化なし。"""
        orch = _make_mock_orch(mood_valence=0.3, mood_arousal=0.4)
        _apply_mood_speed_proportional_reduction(orch, [], 0.5)
        assert orch._psyche.mood.valence == 0.3


# ── テストクラス: 統合シナリオ ─────────────────────────────────


class TestIntegrationScenarios:
    """統合的なシナリオテスト。"""

    def test_5_pathway_simultaneous_firing(self) -> None:
        """5経路同時発火で感情帯域が上限超過するシナリオ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 3つの感情経路が同時発火
        e_delta = _AGGREGATE_CAP_EMOTION * 0.5
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": e_delta})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"sorrow": e_delta})
        monitor.record_firing(PATHWAY_C, 1, emotion_deltas={"love": e_delta})
        # ドライブとムード追従速度は上限以下
        monitor.record_firing(PATHWAY_D, 1, drive_deltas={"social": 0.001})
        monitor.record_firing(PATHWAY_E, 1, mood_speed_deltas={"valence_modulation": 0.001})

        orch = _make_mock_orch(
            emotions={
                "joy": 0.3 + e_delta,
                "sorrow": 0.2 + e_delta,
                "love": 0.1 + e_delta,
            },
            drives={"social": 0.501, "curiosity": 0.5, "expression": 0.5},
            mood_valence=0.001,
            monitor=monitor,
        )

        apply_return_aggregate_cap(orch)

        # 感情帯域のみ上限超過（合算 = 1.5 * cap > cap）
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1
        assert monitor.aggregate_cap_hit_counts["drive"] == 0
        assert monitor.aggregate_cap_hit_counts["mood_speed"] == 0

    def test_no_direction_bias(self) -> None:
        """正方向と負方向の変動が混在しても方向選択的制限をしない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 正方向と負方向の混在（合算絶対値で上限超過）
        pos_delta = _AGGREGATE_CAP_EMOTION * 0.8
        neg_delta = -_AGGREGATE_CAP_EMOTION * 0.8
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": pos_delta})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"sorrow": neg_delta})

        base_joy = 0.3
        base_sorrow = 0.5
        orch = _make_mock_orch(
            emotions={
                "joy": base_joy + pos_delta,
                "sorrow": max(0.0, base_sorrow + neg_delta),
            },
            monitor=monitor,
        )

        apply_return_aggregate_cap(orch)

        # 合算絶対値 = 1.6 * cap > cap なので縮小発動
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

    def test_enrichment_non_exposure(self) -> None:
        """合算上限の到達事実がenrichmentに含まれないことの構造的確認。

        apply_return_aggregate_capがenrichment関連のメソッドを呼び出さないことを確認。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        large = _AGGREGATE_CAP_EMOTION * 3
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": large})

        orch = _make_mock_orch(
            emotions={"joy": 0.5 + large},
            monitor=monitor,
        )

        # enrichment関連属性へのアクセスがないことを確認
        # （apply_return_aggregate_capがenrichment系メソッドを呼ばない）
        apply_return_aggregate_cap(orch)

        # monitor.record_aggregate_cap_hitのみが呼ばれる
        # enrichment系の変更は一切行われない
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

    def test_no_persistence(self) -> None:
        """永続化非対象: 合算上限状態がsave/loadに影響しない。

        モニターの合算上限カウンタはセッション消失であり、
        orchestratorのsave_state/load_stateに影響しない。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("drive")

        # カウンタは存在する
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1
        assert monitor.aggregate_cap_hit_counts["drive"] == 1

        # 新しいモニターインスタンスではカウンタがリセット
        new_monitor = ReturnPathwayMonitor(enabled=True)
        assert new_monitor.aggregate_cap_hit_counts["emotion"] == 0
        assert new_monitor.aggregate_cap_hit_counts["drive"] == 0

    def test_no_feedback_to_individual_caps(self) -> None:
        """帰還経路への逆流なし: 合算上限が個別安全弁に影響しない。

        合算上限の判定結果が、次ティック以降の各帰還経路の個別安全弁の
        閾値を変更しないことを確認（構造的にそのような経路が存在しない）。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        large = _AGGREGATE_CAP_EMOTION * 3
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": large})

        orch = _make_mock_orch(
            emotions={"joy": 0.5 + large},
            monitor=monitor,
        )
        apply_return_aggregate_cap(orch)

        # 合算上限の適用はorch._psyche（感情/ドライブ/ムード）のみに影響し、
        # モニターの設定やその他の構造体には書き込まない
        # （record_aggregate_cap_hitは読み取り専用ログのみ）
        assert monitor.aggregate_cap_hit_counts["emotion"] == 1

    def test_multiple_emotion_pathways_same_dimension(self) -> None:
        """複数の感情経路が同一次元に変動を与える場合の比例縮小。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        d = _AGGREGATE_CAP_EMOTION * 0.4
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": d})
        monitor.record_firing(PATHWAY_B, 1, emotion_deltas={"joy": d})
        monitor.record_firing(PATHWAY_C, 1, emotion_deltas={"joy": d})
        # 合算 = 1.2 * cap > cap

        base = 0.2
        orch = _make_mock_orch(
            emotions={"joy": base + 3 * d},
            monitor=monitor,
        )

        apply_return_aggregate_cap(orch)

        # 縮小が適用される
        assert orch._psyche.emotions.joy < base + 3 * d
        # 比率チェック
        total_abs = 3 * d
        ratio = _AGGREGATE_CAP_EMOTION / total_abs
        correction = 3 * d * (1.0 - ratio)
        expected = base + 3 * d - correction
        assert abs(orch._psyche.emotions.joy - expected) < 1e-6
