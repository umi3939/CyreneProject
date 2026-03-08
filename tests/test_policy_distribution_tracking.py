"""
tests/test_policy_distribution_tracking.py - ポリシー選択分布の経時的変化追跡のテスト

tools/policy_distribution_tracking.py のテスト。
設計書 design_policy_distribution_tracking.md に基づき、
3つの処理（スナップショット蓄積・文脈別分布・推移記述）、
安全弁7種、構造的分離を検証する。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools.policy_distribution_tracking import (
    PolicyDistributionTracking,
    PolicyDistributionTrackingConfig,
    DistributionSnapshot,
    DistributionTransition,
    ContextDistribution,
    build_context_key,
    _discretize_drive_level,
    _discretize_valence,
    _discretize_arousal,
    _compute_concentration_level,
    _compute_stability_level,
    create_policy_distribution_tracking,
)


# ── テストデータ生成ヘルパー ────────────────────────────────────────


def _make_entry(
    tick: int,
    selected_label: str,
) -> dict[str, Any]:
    """テスト用のポリシー選択ログエントリを生成する。"""
    return {
        "tick": tick,
        "timestamp": time.time(),
        "selected_label": selected_label,
        "candidates": [
            {
                "policy_label": selected_label,
                "score": 2.0,
                "score_breakdown": {
                    "drive_goal_match": 0.8,
                    "fear_bias": 0.3,
                },
            },
        ],
        "candidate_count": 1,
        "selected_count": 1,
    }


def _make_entries(
    count: int,
    label_pattern: list[str] | None = None,
    start_tick: int = 1,
) -> list[dict[str, Any]]:
    """複数のテスト用エントリを一括生成する。"""
    if label_pattern is None:
        label_pattern = ["共感する", "質問で会話を広げる", "からかう", "感想を述べる", "励ます"]
    entries = []
    for i in range(count):
        label = label_pattern[i % len(label_pattern)]
        entries.append(_make_entry(tick=start_tick + i, selected_label=label))
    return entries


# ═══════════════════════════════════════════════════════════════════
# 離散化関数のテスト
# ═══════════════════════════════════════════════════════════════════


class TestDiscretization:
    """段階値離散化関数のテスト。"""

    def test_drive_level_low(self) -> None:
        assert _discretize_drive_level(0.0) == "low"
        assert _discretize_drive_level(0.1) == "low"
        assert _discretize_drive_level(0.32) == "low"

    def test_drive_level_mid(self) -> None:
        assert _discretize_drive_level(0.33) == "mid"
        assert _discretize_drive_level(0.5) == "mid"
        assert _discretize_drive_level(0.66) == "mid"

    def test_drive_level_high(self) -> None:
        assert _discretize_drive_level(0.67) == "high"
        assert _discretize_drive_level(0.9) == "high"
        assert _discretize_drive_level(1.0) == "high"

    def test_valence_negative(self) -> None:
        assert _discretize_valence(-0.5) == "negative"
        assert _discretize_valence(-0.21) == "negative"

    def test_valence_neutral(self) -> None:
        assert _discretize_valence(-0.2) == "neutral"
        assert _discretize_valence(0.0) == "neutral"
        assert _discretize_valence(0.2) == "neutral"

    def test_valence_positive(self) -> None:
        assert _discretize_valence(0.21) == "positive"
        assert _discretize_valence(0.5) == "positive"

    def test_arousal_low(self) -> None:
        assert _discretize_arousal(0.0) == "low"
        assert _discretize_arousal(0.32) == "low"

    def test_arousal_mid(self) -> None:
        assert _discretize_arousal(0.33) == "mid"
        assert _discretize_arousal(0.5) == "mid"

    def test_arousal_high(self) -> None:
        assert _discretize_arousal(0.67) == "high"
        assert _discretize_arousal(1.0) == "high"


class TestBuildContextKey:
    """文脈条件キー構築のテスト。"""

    def test_default_key(self) -> None:
        key = build_context_key()
        assert key == "drive=low|valence=neutral|arousal=low"

    def test_all_high(self) -> None:
        key = build_context_key(max_drive=0.9, valence=0.5, arousal=0.8)
        assert key == "drive=high|valence=positive|arousal=high"

    def test_mixed(self) -> None:
        key = build_context_key(max_drive=0.5, valence=-0.5, arousal=0.5)
        assert key == "drive=mid|valence=negative|arousal=mid"


# ═══════════════════════════════════════════════════════════════════
# 集中度算出のテスト
# ═══════════════════════════════════════════════════════════════════


class TestConcentrationLevel:
    """集中度段階値算出のテスト。"""

    def test_empty(self) -> None:
        assert _compute_concentration_level({}) == "none"

    def test_zero_total(self) -> None:
        assert _compute_concentration_level({"A": 0}) == "none"

    def test_low_concentration(self) -> None:
        # 4種が均等 -> 最大比率 0.25 < 0.3
        counts = {"A": 5, "B": 5, "C": 5, "D": 5}
        assert _compute_concentration_level(counts) == "low"

    def test_mid_concentration(self) -> None:
        # 最大比率 ~0.4
        counts = {"A": 4, "B": 3, "C": 2, "D": 1}
        assert _compute_concentration_level(counts) == "mid"

    def test_high_concentration(self) -> None:
        # 最大比率 ~0.6
        counts = {"A": 6, "B": 2, "C": 1, "D": 1}
        assert _compute_concentration_level(counts) == "high"

    def test_very_high_concentration(self) -> None:
        # 最大比率 0.8
        counts = {"A": 8, "B": 1, "C": 1}
        assert _compute_concentration_level(counts) == "very_high"

    def test_single_label(self) -> None:
        # 1種のみ -> 比率1.0 -> very_high
        counts = {"A": 10}
        assert _compute_concentration_level(counts) == "very_high"


# ═══════════════════════════════════════════════════════════════════
# 安定度算出のテスト
# ═══════════════════════════════════════════════════════════════════


class TestStabilityLevel:
    """時間的安定度段階値算出のテスト。"""

    def test_empty(self) -> None:
        assert _compute_stability_level([]) == "none"

    def test_stable(self) -> None:
        # 変化量が小さい
        transitions = [
            DistributionTransition(
                from_tick=0, to_tick=50,
                label_ratio_deltas={"A": 0.01, "B": -0.01}
            ),
            DistributionTransition(
                from_tick=50, to_tick=100,
                label_ratio_deltas={"A": 0.02, "B": -0.02}
            ),
        ]
        assert _compute_stability_level(transitions) == "stable"

    def test_moderate(self) -> None:
        # 変化量が中程度
        transitions = [
            DistributionTransition(
                from_tick=0, to_tick=50,
                label_ratio_deltas={"A": 0.1, "B": -0.1}
            ),
        ]
        assert _compute_stability_level(transitions) == "moderate"

    def test_volatile(self) -> None:
        # 変化量が大きい
        transitions = [
            DistributionTransition(
                from_tick=0, to_tick=50,
                label_ratio_deltas={"A": 0.3, "B": -0.3}
            ),
        ]
        assert _compute_stability_level(transitions) == "volatile"

    def test_empty_deltas(self) -> None:
        transitions = [
            DistributionTransition(
                from_tick=0, to_tick=50,
                label_ratio_deltas={}
            ),
        ]
        assert _compute_stability_level(transitions) == "none"


# ═══════════════════════════════════════════════════════════════════
# 設定のテスト
# ═══════════════════════════════════════════════════════════════════


class TestConfig:
    """設定パラメータのテスト。"""

    def test_default_config(self) -> None:
        config = PolicyDistributionTrackingConfig()
        assert config.snapshot_interval == 50
        assert config.max_snapshots == 30
        assert config.max_transitions == 29
        assert config.max_context_keys == 27
        assert config.max_context_entries == 200

    def test_custom_config(self) -> None:
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=100,
            max_snapshots=50,
            max_transitions=49,
            max_context_keys=10,
            max_context_entries=300,
        )
        assert config.snapshot_interval == 100
        assert config.max_snapshots == 50

    def test_invalid_config_reset(self) -> None:
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=0,
            max_snapshots=-1,
            max_transitions=0,
            max_context_keys=-5,
            max_context_entries=0,
        )
        assert config.snapshot_interval == 50
        assert config.max_snapshots == 30
        assert config.max_transitions == 29
        assert config.max_context_keys == 27
        assert config.max_context_entries == 200


# ═══════════════════════════════════════════════════════════════════
# データ構造のテスト
# ═══════════════════════════════════════════════════════════════════


class TestDistributionSnapshot:
    """分布スナップショットのテスト。"""

    def test_to_dict(self) -> None:
        snapshot = DistributionSnapshot(
            tick=100,
            label_counts={"A": 5, "B": 3},
            label_ratios={"A": 0.625, "B": 0.375},
            concentration_level="high",
            record_count=8,
        )
        d = snapshot.to_dict()
        assert d["tick"] == 100
        assert d["label_counts"] == {"A": 5, "B": 3}
        assert d["concentration_level"] == "high"
        assert d["record_count"] == 8
        assert abs(d["label_ratios"]["A"] - 0.625) < 1e-6

    def test_default_values(self) -> None:
        snapshot = DistributionSnapshot()
        d = snapshot.to_dict()
        assert d["tick"] == 0
        assert d["label_counts"] == {}
        assert d["concentration_level"] == "none"


class TestDistributionTransition:
    """推移記録のテスト。"""

    def test_to_dict(self) -> None:
        transition = DistributionTransition(
            from_tick=50,
            to_tick=100,
            label_ratio_deltas={"A": 0.1, "B": -0.1},
        )
        d = transition.to_dict()
        assert d["from_tick"] == 50
        assert d["to_tick"] == 100
        assert abs(d["label_ratio_deltas"]["A"] - 0.1) < 1e-6


class TestContextDistribution:
    """文脈別選択分布のテスト。"""

    def test_to_dict(self) -> None:
        dist = ContextDistribution(
            context_key="drive=mid|valence=neutral|arousal=mid",
            label_counts={"A": 6, "B": 4},
            total_count=10,
        )
        d = dist.to_dict()
        assert d["context_key"] == "drive=mid|valence=neutral|arousal=mid"
        assert d["label_counts"] == {"A": 6, "B": 4}
        assert d["total_count"] == 10
        assert abs(d["label_ratios"]["A"] - 0.6) < 1e-6
        assert abs(d["label_ratios"]["B"] - 0.4) < 1e-6

    def test_zero_total(self) -> None:
        dist = ContextDistribution(context_key="test", total_count=0)
        d = dist.to_dict()
        assert d["total_count"] == 0


# ═══════════════════════════════════════════════════════════════════
# 安全弁4: 環境変数による完全無効化のテスト
# ═══════════════════════════════════════════════════════════════════


class TestDisabledMode:
    """無効時の動作テスト（安全弁4）。"""

    def test_disabled_by_default(self) -> None:
        # CYRENE_MONITOR未設定時は無効
        env = os.environ.copy()
        env.pop("CYRENE_MONITOR", None)
        os.environ.clear()
        os.environ.update(env)
        tracker = PolicyDistributionTracking()
        assert not tracker.enabled

    def test_disabled_explicit(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        assert not tracker.enabled

    def test_disabled_record_snapshot_returns_none(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        entries = _make_entries(10)
        result = tracker.record_snapshot(tick=100, entries=entries)
        assert result is None
        assert tracker.snapshot_count == 0

    def test_disabled_record_context_returns_none(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        result = tracker.record_context_entry("共感する", 0.5, 0.0, 0.5)
        assert result is None

    def test_disabled_report(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        report = tracker.generate_report()
        assert report["enabled"] is False
        assert report["type"] == "policy_distribution_tracking_report"

    def test_disabled_track_from_log(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        mock_log = MagicMock()
        result = tracker.track_from_log(mock_log, tick=100)
        assert result is None

    def test_disabled_extend_analysis_report(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        base_report = {"type": "policy_selection_analysis_report"}
        result = tracker.extend_analysis_report(base_report)
        assert result["distribution_tracking"]["enabled"] is False


class TestEnabledMode:
    """有効時の基本動作テスト。"""

    def test_enabled_explicit(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        assert tracker.enabled

    def test_initial_state(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        assert tracker.snapshot_count == 0
        assert tracker.transition_count == 0
        assert tracker.context_key_count == 0


# ═══════════════════════════════════════════════════════════════════
# 処理A: 定期的な分布スナップショットの蓄積のテスト
# ═══════════════════════════════════════════════════════════════════


class TestRecordSnapshot:
    """処理A: スナップショット蓄積のテスト。"""

    def test_first_snapshot(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        entries = _make_entries(10)
        result = tracker.record_snapshot(tick=50, entries=entries)
        assert result is not None
        assert result.tick == 50
        assert result.record_count == 10
        assert tracker.snapshot_count == 1

    def test_label_counts(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        entries = _make_entries(10, label_pattern=["A", "A", "B"])
        result = tracker.record_snapshot(tick=50, entries=entries)
        assert result is not None
        # 10 entries: A=4, A=3+1=... pattern [A,A,B,A,A,B,A,A,B,A]
        # pattern index: 0=A, 1=A, 2=B, 3=A, 4=A, 5=B, 6=A, 7=A, 8=B, 9=A
        assert result.label_counts.get("A", 0) == 7
        assert result.label_counts.get("B", 0) == 3

    def test_label_ratios(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        entries = _make_entries(4, label_pattern=["A", "B"])
        result = tracker.record_snapshot(tick=50, entries=entries)
        assert result is not None
        assert abs(result.label_ratios["A"] - 0.5) < 1e-6
        assert abs(result.label_ratios["B"] - 0.5) < 1e-6

    def test_concentration_level(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        # All same label -> very_high
        entries = _make_entries(10, label_pattern=["A"])
        result = tracker.record_snapshot(tick=50, entries=entries)
        assert result is not None
        assert result.concentration_level == "very_high"

    def test_interval_check(self) -> None:
        """一定ティック間隔未満では記録されないことを確認。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=50)
        tracker = PolicyDistributionTracking(config=config, enabled=True)
        entries = _make_entries(10)

        # 最初のスナップショットは常に記録される
        r1 = tracker.record_snapshot(tick=50, entries=entries)
        assert r1 is not None

        # 間隔未満では記録されない
        r2 = tracker.record_snapshot(tick=70, entries=entries)
        assert r2 is None

        # 間隔に達したら記録される
        r3 = tracker.record_snapshot(tick=100, entries=entries)
        assert r3 is not None
        assert tracker.snapshot_count == 2

    def test_empty_entries(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        result = tracker.record_snapshot(tick=50, entries=[])
        assert result is None

    def test_transition_auto_generation(self) -> None:
        """2つ目のスナップショットで推移記録が自動生成されることを確認。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries1 = _make_entries(10, label_pattern=["A"])
        tracker.record_snapshot(tick=1, entries=entries1)
        assert tracker.transition_count == 0

        entries2 = _make_entries(10, label_pattern=["B"])
        tracker.record_snapshot(tick=2, entries=entries2)
        assert tracker.transition_count == 1

    def test_snapshot_fifo(self) -> None:
        """安全弁3: FIFO上限テスト。"""
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=1,
            max_snapshots=3,
        )
        tracker = PolicyDistributionTracking(config=config, enabled=True)
        entries = _make_entries(5)

        for i in range(5):
            tracker.record_snapshot(tick=i + 1, entries=entries)

        assert tracker.snapshot_count == 3
        snapshots = tracker.get_snapshots()
        assert snapshots[0]["tick"] == 3
        assert snapshots[2]["tick"] == 5


# ═══════════════════════════════════════════════════════════════════
# 処理B: 文脈別選択分布のテスト
# ═══════════════════════════════════════════════════════════════════


class TestContextDistributionRecording:
    """処理B: 文脈別選択分布の構成テスト。"""

    def test_basic_recording(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        key = tracker.record_context_entry("共感する", 0.5, 0.0, 0.5)
        assert key is not None
        assert key == "drive=mid|valence=neutral|arousal=mid"
        assert tracker.context_key_count == 1

    def test_same_context_accumulates(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)
        tracker.record_context_entry("B", 0.5, 0.0, 0.5)

        dists = tracker.get_context_distributions()
        key = "drive=mid|valence=neutral|arousal=mid"
        assert dists[key]["label_counts"]["A"] == 2
        assert dists[key]["label_counts"]["B"] == 1
        assert dists[key]["total_count"] == 3

    def test_different_contexts_separate(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        tracker.record_context_entry("A", 0.1, 0.0, 0.1)  # low/neutral/low
        tracker.record_context_entry("B", 0.9, 0.5, 0.9)  # high/positive/high
        assert tracker.context_key_count == 2

    def test_empty_label_ignored(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        result = tracker.record_context_entry("", 0.5, 0.0, 0.5)
        assert result is None

    def test_context_key_limit(self) -> None:
        """安全弁6: 文脈条件種類数の上限テスト。"""
        config = PolicyDistributionTrackingConfig(max_context_keys=3)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        # 3種類まで登録可能
        tracker.record_context_entry("A", 0.1, 0.0, 0.1)
        tracker.record_context_entry("B", 0.5, 0.0, 0.5)
        tracker.record_context_entry("C", 0.9, 0.0, 0.9)
        assert tracker.context_key_count == 3

        # 4種類目は拒否される
        result = tracker.record_context_entry("D", 0.1, 0.5, 0.1)
        assert result is None
        assert tracker.context_key_count == 3

    def test_context_entry_fifo(self) -> None:
        """安全弁3: 各文脈条件のエントリ数FIFO上限テスト。"""
        config = PolicyDistributionTrackingConfig(max_context_entries=5)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        # 5件まで蓄積
        for _ in range(5):
            tracker.record_context_entry("A", 0.5, 0.0, 0.5)

        dists = tracker.get_context_distributions()
        key = "drive=mid|valence=neutral|arousal=mid"
        assert dists[key]["total_count"] == 5

        # 6件目で古い記録が押し出される
        tracker.record_context_entry("B", 0.5, 0.0, 0.5)
        dists = tracker.get_context_distributions()
        assert dists[key]["total_count"] == 5

    def test_context_ratios(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        for _ in range(3):
            tracker.record_context_entry("A", 0.5, 0.0, 0.5)
        tracker.record_context_entry("B", 0.5, 0.0, 0.5)

        dists = tracker.get_context_distributions()
        key = "drive=mid|valence=neutral|arousal=mid"
        assert abs(dists[key]["label_ratios"]["A"] - 0.75) < 1e-6
        assert abs(dists[key]["label_ratios"]["B"] - 0.25) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# 処理C: 推移記録のテスト
# ═══════════════════════════════════════════════════════════════════


class TestTransitions:
    """処理C: 分布推移の数値的記述テスト。"""

    def test_transition_deltas(self) -> None:
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries1 = _make_entries(10, label_pattern=["A", "B"])
        tracker.record_snapshot(tick=1, entries=entries1)

        entries2 = _make_entries(10, label_pattern=["A"])
        tracker.record_snapshot(tick=2, entries=entries2)

        transitions = tracker.get_transitions()
        assert len(transitions) == 1
        t = transitions[0]
        assert t["from_tick"] == 1
        assert t["to_tick"] == 2
        # First: A=5, B=5 -> ratio A=0.5, B=0.5
        # Second: A=10 -> ratio A=1.0
        # Delta: A=+0.5, B=-0.5
        assert abs(t["label_ratio_deltas"]["A"] - 0.5) < 1e-6
        assert abs(t["label_ratio_deltas"]["B"] - (-0.5)) < 1e-6

    def test_transition_fifo(self) -> None:
        """安全弁3: 推移記録のFIFO上限テスト。"""
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=1,
            max_transitions=3,
            max_snapshots=10,
        )
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(5)
        for i in range(6):
            tracker.record_snapshot(tick=i + 1, entries=entries)

        # 6 snapshots -> 5 transitions, but FIFO=3
        assert tracker.transition_count == 3

    def test_stability_level_calculation(self) -> None:
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        # Same entries -> stable
        entries = _make_entries(10, label_pattern=["A", "B"])
        tracker.record_snapshot(tick=1, entries=entries)
        tracker.record_snapshot(tick=2, entries=entries)
        tracker.record_snapshot(tick=3, entries=entries)

        level = tracker.get_stability_level()
        assert level == "stable"

    def test_no_transitions_stability(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        assert tracker.get_stability_level() == "none"


# ═══════════════════════════════════════════════════════════════════
# レポート生成のテスト
# ═══════════════════════════════════════════════════════════════════


class TestReport:
    """レポート生成のテスト。"""

    def test_report_structure(self) -> None:
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(10, label_pattern=["A", "B"])
        tracker.record_snapshot(tick=1, entries=entries)
        tracker.record_snapshot(tick=2, entries=entries)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)

        report = tracker.generate_report()
        assert report["type"] == "policy_distribution_tracking_report"
        assert report["enabled"] is True
        assert "timestamp" in report
        assert report["snapshot_count"] == 2
        assert report["transition_count"] == 1
        assert report["context_key_count"] == 1
        assert "snapshots" in report
        assert "transitions" in report
        assert "context_distributions" in report
        assert "stability_level" in report

    def test_report_disabled(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        report = tracker.generate_report()
        assert report["enabled"] is False
        assert "snapshots" not in report


# ═══════════════════════════════════════════════════════════════════
# 便利メソッドのテスト
# ═══════════════════════════════════════════════════════════════════


class TestTrackFromLog:
    """PolicySelectionLogからの直接追跡テスト。"""

    def test_track_from_log_basic(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.return_value = _make_entries(10)

        result = tracker.track_from_log(mock_log, tick=50)
        assert result is not None
        assert tracker.snapshot_count == 1
        mock_log.get_entries.assert_called_once()

    def test_track_from_log_exception(self) -> None:
        tracker = PolicyDistributionTracking(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.side_effect = RuntimeError("test error")

        result = tracker.track_from_log(mock_log, tick=50)
        assert result is None

    def test_track_from_log_disabled(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        mock_log = MagicMock()
        result = tracker.track_from_log(mock_log, tick=50)
        assert result is None
        mock_log.get_entries.assert_not_called()


class TestExtendAnalysisReport:
    """既存分析基盤レポートへの統合テスト。"""

    def test_extend_report(self) -> None:
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(10)
        tracker.record_snapshot(tick=1, entries=entries)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)

        base_report = {
            "type": "policy_selection_analysis_report",
            "interval_summaries": [],
        }

        result = tracker.extend_analysis_report(base_report)

        # 既存のレポート構造が破壊されていないこと
        assert result["type"] == "policy_selection_analysis_report"
        assert "interval_summaries" in result

        # 追跡結果が追加フィールドとして含まれていること
        assert "distribution_tracking" in result
        dt = result["distribution_tracking"]
        assert dt["enabled"] is True
        assert dt["snapshot_count"] == 1
        assert dt["context_key_count"] == 1

    def test_extend_report_disabled(self) -> None:
        tracker = PolicyDistributionTracking(enabled=False)
        base_report = {"type": "policy_selection_analysis_report"}
        result = tracker.extend_analysis_report(base_report)
        assert result["distribution_tracking"]["enabled"] is False


# ═══════════════════════════════════════════════════════════════════
# ファクトリ関数のテスト
# ═══════════════════════════════════════════════════════════════════


class TestFactory:
    """ファクトリ関数のテスト。"""

    def test_create_default(self) -> None:
        tracker = create_policy_distribution_tracking(enabled=True)
        assert tracker.enabled
        assert tracker.config.snapshot_interval == 50

    def test_create_with_config(self) -> None:
        config = PolicyDistributionTrackingConfig(snapshot_interval=100)
        tracker = create_policy_distribution_tracking(config=config, enabled=True)
        assert tracker.config.snapshot_interval == 100

    def test_create_disabled(self) -> None:
        tracker = create_policy_distribution_tracking(enabled=False)
        assert not tracker.enabled


# ═══════════════════════════════════════════════════════════════════
# 安全弁の検証テスト
# ═══════════════════════════════════════════════════════════════════


class TestSafetyValves:
    """安全弁の検証テスト。"""

    def test_safety_1_no_evaluative_vocabulary(self) -> None:
        """安全弁1: 出力に評価的語彙が含まれないこと。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(10, label_pattern=["A"])
        tracker.record_snapshot(tick=1, entries=entries)

        report = tracker.generate_report()
        report_str = json.dumps(report, ensure_ascii=False)

        evaluative_words = [
            "偏りすぎ", "多様性不足", "望ましい", "目標分布",
            "should", "recommend", "ideal", "optimal",
        ]
        for word in evaluative_words:
            assert word not in report_str

    def test_safety_2_no_recommendations(self) -> None:
        """安全弁2: 推奨が含まれないこと。"""
        tracker = PolicyDistributionTracking(enabled=True)
        report = tracker.generate_report()
        report_str = json.dumps(report, ensure_ascii=False)
        assert "推奨" not in report_str
        assert "recommendation" not in report_str.lower()

    def test_safety_3_fifo_limits(self) -> None:
        """安全弁3: FIFO上限が全てのデータ構造に適用されること。"""
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=1,
            max_snapshots=5,
            max_transitions=4,
            max_context_keys=2,
            max_context_entries=3,
        )
        tracker = PolicyDistributionTracking(config=config, enabled=True)
        entries = _make_entries(5)

        for i in range(10):
            tracker.record_snapshot(tick=i + 1, entries=entries)
        assert tracker.snapshot_count <= 5
        assert tracker.transition_count <= 4

        tracker.record_context_entry("A", 0.1, 0.0, 0.1)
        tracker.record_context_entry("B", 0.5, 0.0, 0.5)
        result = tracker.record_context_entry("C", 0.9, 0.0, 0.9)
        assert result is None
        assert tracker.context_key_count <= 2

    def test_safety_5_session_boundary(self) -> None:
        """安全弁5: セッション境界で全データが消失すること。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)
        entries = _make_entries(10)
        tracker.record_snapshot(tick=1, entries=entries)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)

        assert tracker.snapshot_count == 1
        assert tracker.context_key_count == 1

        # 新しいインスタンスを作成（セッション境界を模倣）
        tracker2 = PolicyDistributionTracking(config=config, enabled=True)
        assert tracker2.snapshot_count == 0
        assert tracker2.context_key_count == 0
        assert tracker2.transition_count == 0

    def test_safety_7_state_independence(self) -> None:
        """安全弁7: 各スナップショットが過去の結果に依存しないこと。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        # 異なるパターンで2つのスナップショットを作成
        entries1 = _make_entries(10, label_pattern=["A"])
        s1 = tracker.record_snapshot(tick=1, entries=entries1)

        entries2 = _make_entries(10, label_pattern=["B"])
        s2 = tracker.record_snapshot(tick=2, entries=entries2)

        # s2はs1に依存していないことを確認
        assert s2 is not None
        assert s2.label_counts.get("B", 0) == 10
        assert s2.label_counts.get("A", 0) == 0
        assert abs(s2.label_ratios["B"] - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# 構造的分離の検証テスト
# ═══════════════════════════════════════════════════════════════════


class TestStructuralSeparation:
    """構造的分離の検証テスト。"""

    def test_no_enrichment_output(self) -> None:
        """prompt enrichment への出力経路を持たないこと。"""
        tracker = PolicyDistributionTracking(enabled=True)
        # enrichment出力を生成するメソッドが存在しないこと
        public_methods = [
            m for m in dir(tracker)
            if not m.startswith("_") and callable(getattr(tracker, m))
        ]
        enrichment_methods = [
            m for m in public_methods
            if "enrichment" in m.lower()
        ]
        assert enrichment_methods == [], \
            f"Enrichment-related methods found: {enrichment_methods}"

    def test_no_policy_modification(self) -> None:
        """方針選択処理を変更する機能を持たないこと。"""
        tracker = PolicyDistributionTracking(enabled=True)
        # 全てのパブリックメソッドが読み取り専用であることを確認
        public_methods = [
            m for m in dir(tracker)
            if not m.startswith("_") and callable(getattr(tracker, m))
        ]
        # 読み取りと記録のみのメソッド名を期待
        expected_prefixes = ["get_", "record_", "track_", "generate_", "extend_"]
        for method in public_methods:
            if method in ("enabled", "config"):
                continue
            assert any(method.startswith(p) or method.endswith("_count") for p in expected_prefixes), \
                f"Unexpected public method: {method}"

    def test_log_readonly(self) -> None:
        """ポリシー選択ログ基盤への書き込みが行われないこと。"""
        tracker = PolicyDistributionTracking(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.return_value = _make_entries(10)

        tracker.track_from_log(mock_log, tick=50)

        # get_entries以外のメソッドが呼ばれていないこと
        assert mock_log.get_entries.call_count == 1
        # record等の書き込みメソッドが呼ばれていないこと
        mock_log.record.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# 統合テスト
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    """統合テスト。"""

    def test_full_workflow(self) -> None:
        """完全なワークフロー: 記録→文脈別分類→レポート生成。"""
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=1,
            max_snapshots=10,
        )
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        # 3回分のスナップショットを記録
        for i in range(3):
            if i == 0:
                entries = _make_entries(20, label_pattern=["A", "B"])
            elif i == 1:
                entries = _make_entries(20, label_pattern=["A", "A", "B"])
            else:
                entries = _make_entries(20, label_pattern=["A"])
            tracker.record_snapshot(tick=(i + 1), entries=entries)

        # 文脈別記録を追加
        for _ in range(5):
            tracker.record_context_entry("A", 0.5, 0.0, 0.5)
        for _ in range(3):
            tracker.record_context_entry("B", 0.1, -0.5, 0.1)

        # レポート生成
        report = tracker.generate_report()
        assert report["snapshot_count"] == 3
        assert report["transition_count"] == 2
        assert report["context_key_count"] == 2
        assert report["stability_level"] in ["none", "stable", "moderate", "volatile"]

        # スナップショットの内容を検証
        snapshots = report["snapshots"]
        assert len(snapshots) == 3
        assert snapshots[0]["tick"] == 1
        assert snapshots[2]["tick"] == 3

    def test_with_policy_selection_analysis(self) -> None:
        """PolicySelectionAnalysisとの統合テスト。"""
        from tools.policy_selection_analysis import (
            PolicySelectionAnalysis,
            PolicySelectionAnalysisConfig,
        )

        config_tracking = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config_tracking, enabled=True)

        config_analysis = PolicySelectionAnalysisConfig(interval_size=10)
        analysis = PolicySelectionAnalysis(config=config_analysis, enabled=True)

        entries = _make_entries(30)
        analysis.analyze_intervals(entries)
        tracker.record_snapshot(tick=30, entries=entries)

        # 分析レポートを生成し、追跡結果を統合
        analysis_report = analysis.generate_report()
        combined = tracker.extend_analysis_report(analysis_report)

        # 既存のレポート構造が維持されていること
        assert "interval_summaries" in combined
        assert "interval_transitions" in combined

        # 追跡結果が追加されていること
        assert "distribution_tracking" in combined
        dt = combined["distribution_tracking"]
        assert dt["enabled"] is True
        assert dt["snapshot_count"] == 1

    def test_multiple_sessions_independent(self) -> None:
        """複数セッション間でデータが独立していることを確認。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)

        # Session 1
        t1 = PolicyDistributionTracking(config=config, enabled=True)
        entries = _make_entries(10, label_pattern=["A"])
        t1.record_snapshot(tick=1, entries=entries)
        t1.record_context_entry("A", 0.5, 0.0, 0.5)
        assert t1.snapshot_count == 1
        assert t1.context_key_count == 1

        # Session 2 (new instance = new session)
        t2 = PolicyDistributionTracking(config=config, enabled=True)
        assert t2.snapshot_count == 0
        assert t2.context_key_count == 0
        assert t2.transition_count == 0

    def test_high_volume(self) -> None:
        """大量データでもFIFO上限に収まることを確認。"""
        config = PolicyDistributionTrackingConfig(
            snapshot_interval=1,
            max_snapshots=10,
            max_transitions=9,
            max_context_keys=5,
            max_context_entries=50,
        )
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(100)
        for i in range(100):
            tracker.record_snapshot(tick=i + 1, entries=entries)

        assert tracker.snapshot_count <= 10
        assert tracker.transition_count <= 9

        for i in range(100):
            tracker.record_context_entry(
                f"label_{i % 3}",
                (i % 3) * 0.4,
                0.0,
                0.5,
            )
        assert tracker.context_key_count <= 5

    def test_report_json_serializable(self) -> None:
        """レポートがJSON直列化可能であることを確認。"""
        config = PolicyDistributionTrackingConfig(snapshot_interval=1)
        tracker = PolicyDistributionTracking(config=config, enabled=True)

        entries = _make_entries(10)
        tracker.record_snapshot(tick=1, entries=entries)
        tracker.record_snapshot(tick=2, entries=entries)
        tracker.record_context_entry("A", 0.5, 0.0, 0.5)

        report = tracker.generate_report()
        text = json.dumps(report, ensure_ascii=False, default=str)
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["type"] == "policy_distribution_tracking_report"
