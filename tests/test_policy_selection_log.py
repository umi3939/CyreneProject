"""
tests/test_policy_selection_log.py - ポリシー選択ログのテスト

tools/policy_selection_log.py の全機能を検証する。

テスト対象:
- PolicySelectionLogConfig: 設定パラメータのバリデーション
- ScoreLogEntry: ログエントリのデータ構造
- AggregationCache: 窓内集計キャッシュのデータ構造
- PolicySelectionLog: 蓄積・集計・出力の本体クラス
- create_policy_selection_log: ファクトリ関数
- 安全弁: FIFO上限、無効時の動作、事実記述限定
- 構造的分離: enrichment非接続、逆流防止
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from unittest.mock import patch

import pytest

# テスト対象のモジュールをインポート
from tools.policy_selection_log import (
    AggregationCache,
    PolicySelectionLog,
    PolicySelectionLogConfig,
    ScoreLogEntry,
    _is_monitor_enabled,
    create_policy_selection_log,
)


# ── テスト用ヘルパー関数 ──────────────────────────────────────────


def _make_candidates(
    labels: list[str],
    scores: list[float],
    breakdowns: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """テスト用の候補リストを作成するヘルパー。

    Args:
        labels: ポリシーラベルのリスト
        scores: スコアのリスト
        breakdowns: 断面別寄与量の辞書のリスト（省略可）

    Returns:
        候補辞書のリスト
    """
    # ラベルとスコアの数が一致することを前提とする
    result = []
    for i, (label, score) in enumerate(zip(labels, scores)):
        # 候補辞書を作成
        cand: dict[str, Any] = {
            "policy_label": label,
            "_score": score,
        }
        # 内訳が指定されている場合は追加
        if breakdowns is not None and i < len(breakdowns):
            cand["_score_breakdown"] = breakdowns[i]
        result.append(cand)
    return result


def _make_log(enabled: bool = True, max_entries: int = 500, window: int = 50) -> PolicySelectionLog:
    """テスト用のPolicySelectionLogインスタンスを作成するヘルパー。"""
    config = PolicySelectionLogConfig(
        max_log_entries=max_entries,
        aggregation_window=window,
    )
    return PolicySelectionLog(config=config, enabled=enabled)


# ── PolicySelectionLogConfig テスト ────────────────────────────────


class TestPolicySelectionLogConfig:
    """PolicySelectionLogConfig のテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値が正しいことを検証する。"""
        config = PolicySelectionLogConfig()
        # デフォルトのFIFO上限は500
        assert config.max_log_entries == 500
        # デフォルトの集計窓サイズは50
        assert config.aggregation_window == 50

    def test_custom_values(self) -> None:
        """カスタム値が正しく設定されることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=100, aggregation_window=20)
        assert config.max_log_entries == 100
        assert config.aggregation_window == 20

    def test_invalid_max_entries_reset(self) -> None:
        """max_log_entries が0以下の場合、500にリセットされることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=0)
        assert config.max_log_entries == 500

    def test_invalid_negative_max_entries(self) -> None:
        """max_log_entries が負の場合、500にリセットされることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=-10)
        assert config.max_log_entries == 500

    def test_invalid_window_reset(self) -> None:
        """aggregation_window が0以下の場合、50にリセットされることを検証する。"""
        config = PolicySelectionLogConfig(aggregation_window=0)
        assert config.aggregation_window == 50

    def test_window_clamped_to_max_entries(self) -> None:
        """aggregation_window が max_log_entries を超えた場合、
        max_log_entries に合わせられることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=10, aggregation_window=20)
        assert config.aggregation_window == 10


# ── ScoreLogEntry テスト ──────────────────────────────────────────


class TestScoreLogEntry:
    """ScoreLogEntry のテスト。"""

    def test_creation(self) -> None:
        """エントリが正しく生成されることを検証する。"""
        entry = ScoreLogEntry(
            tick=1,
            timestamp=1234567890.0,
            selected_label="共感する",
            candidates=[{"policy_label": "共感する", "score": 5.0}],
            candidate_count=3,
            selected_count=3,
        )
        assert entry.tick == 1
        assert entry.timestamp == 1234567890.0
        assert entry.selected_label == "共感する"
        assert entry.candidate_count == 3
        assert entry.selected_count == 3

    def test_to_dict(self) -> None:
        """to_dict() が正しい辞書を返すことを検証する。"""
        candidates = [
            {"policy_label": "共感する", "score": 5.0, "score_breakdown": {"fear_bias": 2.0}},
        ]
        entry = ScoreLogEntry(
            tick=10,
            timestamp=100.5,
            selected_label="共感する",
            candidates=candidates,
            candidate_count=5,
            selected_count=3,
        )
        d = entry.to_dict()
        assert d["tick"] == 10
        assert d["timestamp"] == 100.5
        assert d["selected_label"] == "共感する"
        assert d["candidates"] == candidates
        assert d["candidate_count"] == 5
        assert d["selected_count"] == 3


# ── AggregationCache テスト ───────────────────────────────────────


class TestAggregationCache:
    """AggregationCache のテスト。"""

    def test_empty_defaults(self) -> None:
        """空の集計結果のデフォルト値を検証する。"""
        cache = AggregationCache()
        assert cache.label_selection_counts == {}
        assert cache.section_contribution_totals == {}
        assert cache.section_contribution_variances == {}
        assert cache.top_gap_history == []
        assert cache.max_selection_reached_count == 0
        assert cache.window_size == 0

    def test_to_dict(self) -> None:
        """to_dict() が正しいフォーマットで辞書を返すことを検証する。"""
        cache = AggregationCache(
            label_selection_counts={"共感する": 3, "からかう": 1},
            section_contribution_totals={"fear_bias": 1.123456789},
            section_contribution_variances={"fear_bias": 0.567890123},
            top_gap_history=[1.23456],
            max_selection_reached_count=2,
            window_size=4,
        )
        d = cache.to_dict()
        # ラベル別選択回数
        assert d["label_selection_counts"] == {"共感する": 3, "からかう": 1}
        # 寄与量合計は小数点以下6桁に丸められる
        assert d["section_contribution_totals"]["fear_bias"] == round(1.123456789, 6)
        # 分散も小数点以下6桁に丸められる
        assert d["section_contribution_variances"]["fear_bias"] == round(0.567890123, 6)
        # 差分推移は小数点以下4桁に丸められる
        assert d["top_gap_history"] == [round(1.23456, 4)]
        assert d["max_selection_reached_count"] == 2
        assert d["window_size"] == 4


# ── _is_monitor_enabled テスト ────────────────────────────────────


class TestIsMonitorEnabled:
    """環境変数によるモニタリング有効/無効の判定テスト。"""

    def test_disabled_by_default(self) -> None:
        """デフォルトでは無効であることを検証する。"""
        with patch.dict(os.environ, {}, clear=True):
            # 環境変数が未設定なら無効
            assert _is_monitor_enabled() is False

    def test_disabled_when_zero(self) -> None:
        """CYRENE_MONITOR=0 のとき無効であることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}, clear=True):
            assert _is_monitor_enabled() is False

    def test_enabled_when_one(self) -> None:
        """CYRENE_MONITOR=1 のとき有効であることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}, clear=True):
            assert _is_monitor_enabled() is True

    def test_disabled_for_other_values(self) -> None:
        """CYRENE_MONITOR が "1" 以外の値のとき無効であることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "true"}, clear=True):
            assert _is_monitor_enabled() is False


# ── PolicySelectionLog テスト ─────────────────────────────────────


class TestPolicySelectionLogInit:
    """PolicySelectionLog の初期化テスト。"""

    def test_default_init(self) -> None:
        """デフォルト初期化で正しい状態になることを検証する。"""
        log = PolicySelectionLog(enabled=True)
        assert log.enabled is True
        assert log.entry_count == 0
        assert log.config.max_log_entries == 500

    def test_disabled_init(self) -> None:
        """無効状態で初期化されることを検証する。"""
        log = PolicySelectionLog(enabled=False)
        assert log.enabled is False
        assert log.entry_count == 0

    def test_custom_config(self) -> None:
        """カスタム設定で初期化されることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=10, aggregation_window=5)
        log = PolicySelectionLog(config=config, enabled=True)
        assert log.config.max_log_entries == 10
        assert log.config.aggregation_window == 5

    def test_env_disabled(self) -> None:
        """環境変数で無効化されることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}, clear=True):
            log = PolicySelectionLog()
            assert log.enabled is False

    def test_env_enabled(self) -> None:
        """環境変数で有効化されることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}, clear=True):
            log = PolicySelectionLog()
            assert log.enabled is True

    def test_explicit_enabled_overrides_env(self) -> None:
        """明示的な enabled 指定が環境変数より優先されることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}, clear=True):
            log = PolicySelectionLog(enabled=True)
            assert log.enabled is True


class TestPolicySelectionLogRecord:
    """PolicySelectionLog.record() のテスト。"""

    def test_basic_record(self) -> None:
        """基本的な記録が正しく蓄積されることを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(
            labels=["共感する", "からかう"],
            scores=[5.0, 3.0],
        )
        log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=2)
        # 1件記録された
        assert log.entry_count == 1

    def test_record_with_breakdown(self) -> None:
        """断面別寄与量付きの記録が正しく蓄積されることを検証する。"""
        log = _make_log(enabled=True)
        breakdowns = [
            {"drive_goal_match": 1.0, "fear_bias": 2.0},
            {"drive_goal_match": 0.5, "fear_bias": 0.0},
        ]
        candidates = _make_candidates(
            labels=["共感する", "からかう"],
            scores=[5.0, 3.0],
            breakdowns=breakdowns,
        )
        log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=2)
        # エントリを取得して内訳が含まれていることを確認
        entries = log.get_entries()
        assert len(entries) == 1
        cands = entries[0]["candidates"]
        assert "score_breakdown" in cands[0]
        assert cands[0]["score_breakdown"]["drive_goal_match"] == 1.0

    def test_record_without_breakdown(self) -> None:
        """断面別寄与量がない候補でもエラーなく記録されることを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(
            labels=["共感する"],
            scores=[5.0],
        )
        log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=1)
        entries = log.get_entries()
        assert len(entries) == 1
        # 内訳フィールドは存在しないことを確認
        assert "score_breakdown" not in entries[0]["candidates"][0]

    def test_disabled_record_ignored(self) -> None:
        """無効時はrecordが何もしないことを検証する。"""
        log = _make_log(enabled=False)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=1)
        # 無効なので記録されない
        assert log.entry_count == 0

    def test_multiple_records(self) -> None:
        """複数回の記録が正しく蓄積されることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(10):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)
        assert log.entry_count == 10

    def test_fifo_limit(self) -> None:
        """FIFO上限を超えた場合に古い記録が押し出されることを検証する（安全弁4）。"""
        # FIFO上限を5件に設定
        log = _make_log(enabled=True, max_entries=5)
        for i in range(10):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)
        # 上限5件なので5件だけ残る
        assert log.entry_count == 5
        # 最古の記録はtick=5（0〜4は押し出された）
        entries = log.get_entries()
        assert entries[0]["tick"] == 5
        # 最新の記録はtick=9
        assert entries[-1]["tick"] == 9

    def test_empty_candidates(self) -> None:
        """候補リストが空でもエラーなく記録されることを検証する。"""
        log = _make_log(enabled=True)
        log.record(tick=1, selected_label="", candidates=[], selected_count=0)
        assert log.entry_count == 1

    def test_candidate_count_accurate(self) -> None:
        """candidate_count が候補の実際の数を反映することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(
            labels=["共感する", "からかう", "質問で会話を広げる"],
            scores=[5.0, 3.0, 2.0],
        )
        log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=3)
        entries = log.get_entries()
        assert entries[0]["candidate_count"] == 3


class TestPolicySelectionLogAggregation:
    """窓内集計のテスト。"""

    def test_empty_aggregation(self) -> None:
        """記録が空の場合の集計結果を検証する。"""
        log = _make_log(enabled=True)
        agg = log.get_aggregation()
        assert agg.label_selection_counts == {}
        assert agg.section_contribution_totals == {}
        assert agg.window_size == 0

    def test_label_selection_counts(self) -> None:
        """ポリシーラベル別の選択回数が正しく集計されることを検証する。"""
        log = _make_log(enabled=True)
        # "共感する" を3回、"からかう" を2回選択する
        for _ in range(3):
            candidates = _make_candidates(labels=["共感する"], scores=[5.0])
            log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        for _ in range(2):
            candidates = _make_candidates(labels=["からかう"], scores=[4.0])
            log.record(tick=0, selected_label="からかう", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        assert agg.label_selection_counts["共感する"] == 3
        assert agg.label_selection_counts["からかう"] == 2

    def test_section_contribution_totals(self) -> None:
        """断面別の寄与量合計が正しく集計されることを検証する。"""
        log = _make_log(enabled=True)
        # 2件の記録を追加。各記録に1候補、各候補に断面別寄与量がある
        for i in range(2):
            breakdowns = [{"drive_goal_match": 1.5, "fear_bias": -0.5}]
            candidates = _make_candidates(
                labels=["共感する"], scores=[5.0], breakdowns=breakdowns,
            )
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        # 1.5 * 2 = 3.0
        assert abs(agg.section_contribution_totals["drive_goal_match"] - 3.0) < 1e-6
        # -0.5 * 2 = -1.0
        assert abs(agg.section_contribution_totals["fear_bias"] - (-1.0)) < 1e-6

    def test_section_contribution_variances(self) -> None:
        """断面別の寄与量の分散が正しく計算されることを検証する。"""
        log = _make_log(enabled=True)
        # 異なる寄与量で2件の記録を追加
        breakdowns1 = [{"fear_bias": 2.0}]
        breakdowns2 = [{"fear_bias": 4.0}]
        candidates1 = _make_candidates(labels=["共感する"], scores=[5.0], breakdowns=breakdowns1)
        candidates2 = _make_candidates(labels=["共感する"], scores=[5.0], breakdowns=breakdowns2)
        log.record(tick=0, selected_label="共感する", candidates=candidates1, selected_count=1)
        log.record(tick=1, selected_label="共感する", candidates=candidates2, selected_count=1)

        agg = log.get_aggregation()
        # 平均 = (2.0 + 4.0) / 2 = 3.0
        # 分散 = ((2-3)^2 + (4-3)^2) / 2 = (1 + 1) / 2 = 1.0
        assert abs(agg.section_contribution_variances["fear_bias"] - 1.0) < 1e-6

    def test_single_value_variance_is_zero(self) -> None:
        """値が1つだけの場合は分散が0になることを検証する。"""
        log = _make_log(enabled=True)
        breakdowns = [{"fear_bias": 3.0}]
        candidates = _make_candidates(labels=["共感する"], scores=[5.0], breakdowns=breakdowns)
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        assert agg.section_contribution_variances["fear_bias"] == 0.0

    def test_top_gap_history(self) -> None:
        """1位と2位のスコア差分が正しく記録されることを検証する。"""
        log = _make_log(enabled=True)
        # 1位=5.0, 2位=3.0 -> gap=2.0
        candidates = _make_candidates(labels=["共感する", "からかう"], scores=[5.0, 3.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=2)

        agg = log.get_aggregation()
        assert len(agg.top_gap_history) == 1
        assert abs(agg.top_gap_history[0] - 2.0) < 1e-6

    def test_top_gap_single_candidate(self) -> None:
        """候補が1つだけの場合、そのスコアがgapとして記録されることを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        assert len(agg.top_gap_history) == 1
        assert abs(agg.top_gap_history[0] - 5.0) < 1e-6

    def test_max_selection_reached_count(self) -> None:
        """選出数が上限(5件)に到達した回数が正しくカウントされることを検証する。"""
        log = _make_log(enabled=True)
        # selected_count=5 が2回
        for _ in range(2):
            candidates = _make_candidates(labels=["共感する"], scores=[5.0])
            log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=5)
        # selected_count=3 が3回
        for _ in range(3):
            candidates = _make_candidates(labels=["共感する"], scores=[5.0])
            log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=3)

        agg = log.get_aggregation()
        # selected_count >= 5 は2回
        assert agg.max_selection_reached_count == 2

    def test_window_limits_aggregation(self) -> None:
        """集計が窓サイズ内の記録のみを対象とすることを検証する。"""
        # 窓サイズを3に設定
        log = _make_log(enabled=True, window=3)
        # 5件記録する
        labels = ["共感する", "からかう", "質問で会話を広げる", "共感する", "共感する"]
        for i, label in enumerate(labels):
            candidates = _make_candidates(labels=[label], scores=[5.0])
            log.record(tick=i, selected_label=label, candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        # 窓サイズ3なので直近3件のみ集計
        assert agg.window_size == 3
        assert agg.label_selection_counts.get("共感する", 0) == 2
        assert agg.label_selection_counts.get("質問で会話を広げる", 0) == 1
        # "からかう" は窓外なのでカウントされない
        assert agg.label_selection_counts.get("からかう", 0) == 0

    def test_cache_invalidation(self) -> None:
        """新しい記録が追加された後にキャッシュが再計算されることを検証する。"""
        log = _make_log(enabled=True)
        # 1件記録
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        agg1 = log.get_aggregation()
        assert agg1.label_selection_counts["共感する"] == 1

        # 追加で1件記録
        candidates2 = _make_candidates(labels=["からかう"], scores=[4.0])
        log.record(tick=1, selected_label="からかう", candidates=candidates2, selected_count=1)
        agg2 = log.get_aggregation()
        # キャッシュが再計算されているので両方カウントされている
        assert agg2.label_selection_counts["共感する"] == 1
        assert agg2.label_selection_counts["からかう"] == 1

    def test_cache_reuse(self) -> None:
        """記録追加なしで複数回呼んだ場合、同じキャッシュが返ることを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        agg1 = log.get_aggregation()
        agg2 = log.get_aggregation()
        # 同じオブジェクトが返される
        assert agg1 is agg2

    def test_aggregation_no_breakdown(self) -> None:
        """内訳がない候補で集計がエラーなく動くことを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する", "からかう"], scores=[5.0, 3.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=2)

        agg = log.get_aggregation()
        # 内訳がないので断面別集計は空
        assert agg.section_contribution_totals == {}
        assert agg.section_contribution_variances == {}


class TestPolicySelectionLogReport:
    """emit_report() のテスト。"""

    def test_empty_report(self) -> None:
        """記録が空の場合のレポートを検証する。"""
        log = _make_log(enabled=True)
        report = log.emit_report()
        assert report["type"] == "policy_selection_report"
        assert report["total_entries"] == 0
        assert "aggregation" in report
        assert report["aggregation"]["window_size"] == 0

    def test_report_with_data(self) -> None:
        """データがある場合のレポートを検証する。"""
        log = _make_log(enabled=True)
        breakdowns = [{"drive_goal_match": 1.0}]
        candidates = _make_candidates(labels=["共感する"], scores=[5.0], breakdowns=breakdowns)
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        report = log.emit_report()
        assert report["type"] == "policy_selection_report"
        assert report["total_entries"] == 1
        assert report["aggregation"]["window_size"] == 1

    def test_report_has_no_evaluative_judgment(self) -> None:
        """レポートに評価的判断が含まれないことを検証する（安全弁3）。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        report = log.emit_report()
        # レポート全体をJSON文字列に変換して検索
        report_str = json.dumps(report, ensure_ascii=False)
        # 評価的な語彙が含まれていないことを確認
        forbidden_words = ["偏りすぎ", "適切", "改善", "警告", "異常", "正常", "推奨"]
        for word in forbidden_words:
            assert word not in report_str

    def test_report_disabled_no_log_output(self) -> None:
        """無効時でもレポートは辞書として返されることを検証する。"""
        log = _make_log(enabled=False)
        report = log.emit_report()
        assert report["type"] == "policy_selection_report"


class TestPolicySelectionLogGetEntries:
    """get_entries() のテスト。"""

    def test_get_all_entries(self) -> None:
        """全件取得ができることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(5):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        entries = log.get_entries()
        assert len(entries) == 5
        # 時系列順（tick=0が最初）
        assert entries[0]["tick"] == 0
        assert entries[-1]["tick"] == 4

    def test_get_last_n_entries(self) -> None:
        """直近N件の取得ができることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(10):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        entries = log.get_entries(last_n=3)
        assert len(entries) == 3
        # 直近3件はtick=7, 8, 9
        assert entries[0]["tick"] == 7
        assert entries[-1]["tick"] == 9

    def test_get_entries_last_n_larger_than_total(self) -> None:
        """last_n が蓄積件数より大きい場合、全件が返ることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(3):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        entries = log.get_entries(last_n=100)
        assert len(entries) == 3

    def test_get_entries_none_returns_all(self) -> None:
        """last_n=None で全件が返ることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(5):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        entries = log.get_entries(last_n=None)
        assert len(entries) == 5

    def test_get_entries_zero_returns_all(self) -> None:
        """last_n=0 で全件が返ることを検証する。"""
        log = _make_log(enabled=True)
        for i in range(5):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        entries = log.get_entries(last_n=0)
        assert len(entries) == 5


# ── create_policy_selection_log テスト ────────────────────────────


class TestCreatePolicySelectionLog:
    """ファクトリ関数のテスト。"""

    def test_default_creation(self) -> None:
        """デフォルトパラメータで正しくインスタンスが生成されることを検証する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}, clear=True):
            log = create_policy_selection_log()
            assert isinstance(log, PolicySelectionLog)
            assert log.enabled is True

    def test_custom_creation(self) -> None:
        """カスタムパラメータで正しくインスタンスが生成されることを検証する。"""
        config = PolicySelectionLogConfig(max_log_entries=100, aggregation_window=10)
        log = create_policy_selection_log(config=config, enabled=True)
        assert log.config.max_log_entries == 100
        assert log.enabled is True

    def test_explicit_disabled(self) -> None:
        """enabled=False で無効なインスタンスが生成されることを検証する。"""
        log = create_policy_selection_log(enabled=False)
        assert log.enabled is False


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作テスト。"""

    def test_valve1_no_enrichment_function(self) -> None:
        """安全弁1: enrichment出力を生成する関数が存在しないことを検証する。"""
        log = _make_log(enabled=True)
        # enrichment関連のメソッドが存在しないことを確認
        assert not hasattr(log, "get_enrichment")
        assert not hasattr(log, "get_prompt_enrichment")
        assert not hasattr(log, "enrichment")

    def test_valve2_no_save_load(self) -> None:
        """安全弁2: save/load関連の関数が存在しないことを検証する。"""
        log = _make_log(enabled=True)
        # save/load関連のメソッドが存在しないことを確認
        assert not hasattr(log, "save")
        assert not hasattr(log, "load")
        assert not hasattr(log, "to_dict")
        assert not hasattr(log, "from_dict")

    def test_valve3_fact_only_no_judgment(self) -> None:
        """安全弁3: 集計結果が数値的事実のみであることを検証する。"""
        log = _make_log(enabled=True)
        breakdowns = [{"fear_bias": 5.0, "drive_goal_match": 2.0}]
        candidates = _make_candidates(labels=["共感する"], scores=[7.0], breakdowns=breakdowns)
        for i in range(10):
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        agg_dict = agg.to_dict()
        # 辞書のキーが想定通りであることを確認
        expected_keys = {
            "label_selection_counts",
            "section_contribution_totals",
            "section_contribution_variances",
            "top_gap_history",
            "max_selection_reached_count",
            "window_size",
        }
        assert set(agg_dict.keys()) == expected_keys
        # 全ての値が数値またはリスト/辞書であり、文字列の判断文を含まない
        for key, value in agg_dict.items():
            assert not isinstance(value, str), f"{key} should not be a string judgment"

    def test_valve4_fifo_limit_enforced(self) -> None:
        """安全弁4: FIFO上限が正しく適用されることを検証する。"""
        log = _make_log(enabled=True, max_entries=3)
        for i in range(10):
            candidates = _make_candidates(labels=["共感する"], scores=[float(i)])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)
        # 上限3件
        assert log.entry_count == 3
        entries = log.get_entries()
        # 最古はtick=7, 最新はtick=9
        assert entries[0]["tick"] == 7
        assert entries[-1]["tick"] == 9

    def test_valve5_disabled_skips_all(self) -> None:
        """安全弁5: 無効時は蓄積・集計・出力が全て省略されることを検証する。"""
        log = _make_log(enabled=False)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        # 10件記録を試行
        for i in range(10):
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=1)
        # 全て無視される
        assert log.entry_count == 0
        # 集計は空
        agg = log.get_aggregation()
        assert agg.window_size == 0


# ── 構造的分離テスト ─────────────────────────────────────────────


class TestStructuralSeparation:
    """設計書で定義された構造的分離の検証。"""

    def test_no_enrichment_output(self) -> None:
        """prompt enrichment への接続経路がないことを検証する。"""
        log = _make_log(enabled=True)
        # enrichment関連の属性・メソッドが一切存在しない
        enrichment_names = [
            "get_enrichment", "get_prompt_enrichment", "enrichment_text",
            "build_enrichment", "_enrichment", "enrichment_section",
        ]
        for name in enrichment_names:
            assert not hasattr(log, name), f"Unexpected enrichment attribute: {name}"

    def test_no_reverse_flow(self) -> None:
        """方針選択処理への逆流経路がないことを検証する。"""
        log = _make_log(enabled=True)
        # 方針選択処理への入力を生成するメソッドが存在しない
        reverse_names = [
            "apply_bias", "modify_candidates", "inject_input",
            "set_bias", "update_scores", "adjust_scoring",
        ]
        for name in reverse_names:
            assert not hasattr(log, name), f"Unexpected reverse-flow method: {name}"

    def test_no_persistence(self) -> None:
        """永続化対象外であることを検証する。"""
        log = _make_log(enabled=True)
        # save/load/to_dict/from_dict が存在しない
        persistence_names = ["save", "load", "to_dict", "from_dict", "state"]
        for name in persistence_names:
            assert not hasattr(log, name), f"Unexpected persistence attribute: {name}"


# ── 複合シナリオテスト ───────────────────────────────────────────


class TestComplexScenarios:
    """複合的なシナリオのテスト。"""

    def test_mixed_breakdowns(self) -> None:
        """一部の候補にのみ内訳がある場合の集計を検証する。"""
        log = _make_log(enabled=True)
        # 候補1には内訳あり、候補2には内訳なし
        candidates = [
            {"policy_label": "共感する", "_score": 5.0, "_score_breakdown": {"fear_bias": 2.0}},
            {"policy_label": "からかう", "_score": 3.0},  # 内訳なし
        ]
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=2)

        agg = log.get_aggregation()
        # 内訳がある候補のみ集計される
        assert "fear_bias" in agg.section_contribution_totals

    def test_long_running_session(self) -> None:
        """長時間セッション（多数の記録）でもFIFOが正しく機能することを検証する。"""
        log = _make_log(enabled=True, max_entries=100)
        for i in range(500):
            label = f"policy_{i % 5}"
            breakdowns = [{"section_a": float(i % 3), "section_b": float(i % 7)}]
            candidates = _make_candidates(labels=[label], scores=[float(i)], breakdowns=breakdowns)
            log.record(tick=i, selected_label=label, candidates=candidates, selected_count=1)

        # FIFO上限100件
        assert log.entry_count == 100
        # 最古の記録はtick=400
        entries = log.get_entries()
        assert entries[0]["tick"] == 400

        # 集計も正常に動作する
        agg = log.get_aggregation()
        assert agg.window_size > 0

    def test_aggregation_with_many_sections(self) -> None:
        """多数の断面がある場合の集計を検証する。"""
        log = _make_log(enabled=True)
        # 11個の断面（thought.pyの実際の断面数に相当）
        breakdown = {
            "drive_goal_match": 1.2,
            "fear_bias": -0.5,
            "mood_alignment": 0.8,
            "percept_intent_match": 1.5,
            "percept_emotion_valence": 0.3,
            "attachment_risk_reaction": 0.0,
            "identity_risk_reaction": 0.0,
            "memory_context": 0.3,
            "responsibility_influence": 0.6,
            "stm_decision_bias": 0.1,
            "extended_input_reaction": 0.4,
        }
        candidates = _make_candidates(
            labels=["共感する"], scores=[5.0], breakdowns=[breakdown],
        )
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        # 全11断面の寄与量が集計されている
        assert len(agg.section_contribution_totals) == 11

    def test_repeated_same_policy_detection(self) -> None:
        """同じポリシーが連続選択される偏りを数値的に検出できることを検証する。"""
        log = _make_log(enabled=True, window=10)
        # 全て"共感する"を選択
        for i in range(10):
            candidates = _make_candidates(labels=["共感する", "からかう"], scores=[5.0, 2.0])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=2)

        agg = log.get_aggregation()
        # "共感する"が10回選択されている（事実の記述）
        assert agg.label_selection_counts["共感する"] == 10
        # "からかう"は選択されていない
        assert agg.label_selection_counts.get("からかう", 0) == 0

    def test_diverse_policy_selection(self) -> None:
        """多様なポリシーが選択される場合の集計を検証する。"""
        log = _make_log(enabled=True, window=10)
        labels = ["共感する", "からかう", "質問で会話を広げる", "話題を変える", "感想を述べる"]
        for i, label in enumerate(labels * 2):  # 各2回
            candidates = _make_candidates(labels=[label], scores=[5.0])
            log.record(tick=i, selected_label=label, candidates=candidates, selected_count=1)

        agg = log.get_aggregation()
        assert agg.window_size == 10
        for label in labels:
            assert agg.label_selection_counts[label] == 2

    def test_score_gap_trend(self) -> None:
        """スコア差の推移が記録されることを検証する。"""
        log = _make_log(enabled=True, window=5)
        # スコア差が次第に小さくなるシナリオ
        gaps = [(10.0, 2.0), (10.0, 5.0), (10.0, 8.0), (10.0, 9.0), (10.0, 9.5)]
        for i, (s1, s2) in enumerate(gaps):
            candidates = _make_candidates(labels=["共感する", "からかう"], scores=[s1, s2])
            log.record(tick=i, selected_label="共感する", candidates=candidates, selected_count=2)

        agg = log.get_aggregation()
        expected_gaps = [8.0, 5.0, 2.0, 1.0, 0.5]
        for actual, expected in zip(agg.top_gap_history, expected_gaps):
            assert abs(actual - expected) < 1e-6


# ── ログ出力テスト ───────────────────────────────────────────────


class TestLogOutput:
    """ログストリームへの出力テスト。"""

    def test_record_emits_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """record()がログストリームにJSON出力することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=1)

        # ログが出力されている
        assert len(caplog.records) > 0
        # ログはJSON形式
        record_text = caplog.records[0].message
        parsed = json.loads(record_text)
        assert parsed["type"] == "policy_selection_log"
        assert parsed["selected_label"] == "共感する"

    def test_report_emits_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """emit_report()がログストリームにJSON出力することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            report = log.emit_report()

        # ログが出力されている
        found_report = False
        for rec in caplog.records:
            try:
                parsed = json.loads(rec.message)
                if parsed.get("type") == "policy_selection_report":
                    found_report = True
            except (json.JSONDecodeError, AttributeError):
                pass
        assert found_report

    def test_disabled_no_log_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """無効時はログが出力されないことを検証する。"""
        log = _make_log(enabled=False)
        candidates = _make_candidates(labels=["共感する"], scores=[5.0])
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            log.record(tick=1, selected_label="共感する", candidates=candidates, selected_count=1)

        # ログが出力されていない
        assert len(caplog.records) == 0


# ── エッジケーステスト ───────────────────────────────────────────


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_negative_scores(self) -> None:
        """負のスコアでもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する", "からかう"], scores=[-1.0, -3.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=2)
        assert log.entry_count == 1
        agg = log.get_aggregation()
        assert agg.top_gap_history[0] == 2.0  # -1.0 - (-3.0) = 2.0

    def test_zero_scores(self) -> None:
        """スコアが全て0でもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する", "からかう"], scores=[0.0, 0.0])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=2)
        assert log.entry_count == 1
        agg = log.get_aggregation()
        assert agg.top_gap_history[0] == 0.0

    def test_large_score_values(self) -> None:
        """非常に大きいスコア値でもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=["共感する"], scores=[1e10])
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        assert log.entry_count == 1

    def test_unicode_labels(self) -> None:
        """日本語ラベルが正しく処理されることを検証する。"""
        log = _make_log(enabled=True)
        labels = ["共感する", "質問で会話を広げる", "距離を詰める", "内面を振り返る"]
        for i, label in enumerate(labels):
            candidates = _make_candidates(labels=[label], scores=[5.0])
            log.record(tick=i, selected_label=label, candidates=candidates, selected_count=1)

        entries = log.get_entries()
        assert len(entries) == 4
        agg = log.get_aggregation()
        for label in labels:
            assert agg.label_selection_counts[label] == 1

    def test_empty_string_label(self) -> None:
        """空文字列のラベルでもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = _make_candidates(labels=[""], scores=[5.0])
        log.record(tick=0, selected_label="", candidates=candidates, selected_count=1)
        assert log.entry_count == 1

    def test_breakdown_with_non_numeric_values(self) -> None:
        """内訳に非数値が含まれていてもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        # 文字列が混ざった内訳
        candidates = [
            {
                "policy_label": "共感する",
                "_score": 5.0,
                "_score_breakdown": {"fear_bias": 2.0, "invalid": "not_a_number"},
            },
        ]
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        assert log.entry_count == 1
        # 集計時に非数値はスキップされる
        agg = log.get_aggregation()
        assert "fear_bias" in agg.section_contribution_totals
        # "invalid" は非数値なので集計されない
        assert "invalid" not in agg.section_contribution_totals

    def test_breakdown_is_none(self) -> None:
        """内訳がNoneの場合でもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = [
            {"policy_label": "共感する", "_score": 5.0, "_score_breakdown": None},
        ]
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        assert log.entry_count == 1
        entries = log.get_entries()
        # Noneの場合は内訳フィールドが付帯されない
        assert "score_breakdown" not in entries[0]["candidates"][0]

    def test_breakdown_is_not_dict(self) -> None:
        """内訳が辞書でない場合でもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        candidates = [
            {"policy_label": "共感する", "_score": 5.0, "_score_breakdown": [1, 2, 3]},
        ]
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=1)
        assert log.entry_count == 1
        entries = log.get_entries()
        # リスト型は辞書ではないので内訳フィールドが付帯されない
        assert "score_breakdown" not in entries[0]["candidates"][0]

    def test_many_candidates(self) -> None:
        """多数の候補（15件: thought.pyの全ポリシー相当）でもエラーなく動作することを検証する。"""
        log = _make_log(enabled=True)
        labels = [
            "共感する", "質問で会話を広げる", "からかう", "話題を変える", "感想を述べる",
            "励ます", "黙って聞く", "自分の経験を話す", "確認する", "冗談を言う",
            "謝る", "提案する", "見守る", "同意する", "反論する",
        ]
        scores = [float(15 - i) for i in range(15)]
        breakdowns = [{"drive_goal_match": float(i)} for i in range(15)]
        candidates = _make_candidates(labels=labels, scores=scores, breakdowns=breakdowns)
        log.record(tick=0, selected_label="共感する", candidates=candidates, selected_count=5)

        entries = log.get_entries()
        assert entries[0]["candidate_count"] == 15
        assert entries[0]["selected_count"] == 5

    def test_record_exception_safety(self) -> None:
        """record()内で例外が発生しても安全に無視されることを検証する。"""
        log = _make_log(enabled=True)
        # time.time() がエラーを起こしても安全に処理される
        with patch("tools.policy_selection_log.time.time", side_effect=RuntimeError("test")):
            # 例外が外に漏れない
            log.record(tick=0, selected_label="test", candidates=[], selected_count=0)
        # エントリは追加されない（例外で中断された）
        assert log.entry_count == 0
