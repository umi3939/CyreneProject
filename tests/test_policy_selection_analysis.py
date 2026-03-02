"""
tests/test_policy_selection_analysis.py - 動的ポリシー選択の長期効果検証ツールのテスト

tools/policy_selection_analysis.py のテスト。
設計書 design_policy_selection_analysis.md に基づき、
4段パイプライン処理・安全弁・構造的分離を検証する。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools.policy_selection_analysis import (
    PolicySelectionAnalysis,
    PolicySelectionAnalysisConfig,
    IntervalSummary,
    IntervalTransition,
    ScenarioSummary,
    _compute_section_stats,
    _compute_label_counts,
    create_policy_selection_analysis,
)


# ── テストデータ生成ヘルパー ────────────────────────────────────────


def _make_entry(
    tick: int,
    selected_label: str,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """テスト用のポリシー選択ログエントリを生成する。"""
    if candidates is None:
        candidates = [
            {
                "policy_label": selected_label,
                "score": 2.0,
                "score_breakdown": {
                    "drive_goal_match": 0.8,
                    "fear_bias": 0.3,
                    "mood_alignment": 0.5,
                    "percept_intent_match": 0.2,
                    "percept_emotion_valence": 0.1,
                    "responsibility_influence": 0.1,
                },
            },
            {
                "policy_label": "質問で会話を広げる",
                "score": 1.0,
                "score_breakdown": {
                    "drive_goal_match": 0.4,
                    "fear_bias": 0.1,
                    "mood_alignment": 0.2,
                    "percept_intent_match": 0.1,
                    "percept_emotion_valence": 0.1,
                    "responsibility_influence": 0.1,
                },
            },
        ]
    return {
        "tick": tick,
        "timestamp": time.time(),
        "selected_label": selected_label,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "selected_count": min(len(candidates), 3),
    }


def _make_entries(
    count: int,
    label_pattern: list[str] | None = None,
) -> list[dict[str, Any]]:
    """複数のテスト用エントリを一括生成する。"""
    if label_pattern is None:
        label_pattern = ["共感する", "質問で会話を広げる", "からかう", "感想を述べる", "励ます"]
    entries = []
    for i in range(count):
        label = label_pattern[i % len(label_pattern)]
        entries.append(_make_entry(tick=i + 1, selected_label=label))
    return entries


def _make_sim_result(
    scenario_name: str,
    turns: int = 50,
    label_pattern: list[str] | None = None,
) -> dict[str, Any]:
    """テスト用の長期シミュレーション結果を生成する。"""
    if label_pattern is None:
        label_pattern = ["共感する", "質問で会話を広げる", "からかう"]
    turn_records = []
    for i in range(turns):
        label = label_pattern[i % len(label_pattern)]
        turn_records.append({
            "turn": i + 1,
            "tick": i + 1,
            "policy": {
                "policy_label": label,
                "score": 2.0,
                "rationale": "test",
            },
        })
    return {
        "metadata": {
            "scenario": scenario_name,
            "total_turns": turns,
        },
        "turns": turn_records,
    }


# ── PolicySelectionAnalysisConfig テスト ────────────────────────────


class TestConfig:
    """設定のバリデーションをテストする。"""

    def test_default_config(self) -> None:
        """デフォルト設定が正しい値を持つことを確認する。"""
        config = PolicySelectionAnalysisConfig()
        assert config.interval_size == 50
        assert config.max_intervals == 20

    def test_custom_config(self) -> None:
        """カスタム設定が受け入れられることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=100, max_intervals=10)
        assert config.interval_size == 100
        assert config.max_intervals == 10

    def test_invalid_interval_size_reset(self) -> None:
        """不正なinterval_sizeがデフォルトに戻されることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=0)
        assert config.interval_size == 50

    def test_invalid_max_intervals_reset(self) -> None:
        """不正なmax_intervalsがデフォルトに戻されることを確認する。"""
        config = PolicySelectionAnalysisConfig(max_intervals=-1)
        assert config.max_intervals == 20

    def test_negative_values_reset(self) -> None:
        """負の値がデフォルトに戻されることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=-10, max_intervals=-5)
        assert config.interval_size == 50
        assert config.max_intervals == 20


# ── 環境変数制御テスト ────────────────────────────────────────────


class TestEnvironmentControl:
    """環境変数による有効/無効制御をテストする。"""

    def test_enabled_by_explicit_flag(self) -> None:
        """明示的なenabledフラグで有効化できることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        assert analyzer.enabled is True

    def test_disabled_by_explicit_flag(self) -> None:
        """明示的なenabledフラグで無効化できることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        assert analyzer.enabled is False

    def test_disabled_returns_empty_on_analyze(self) -> None:
        """無効時はanalyze_intervalsが空リストを返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        entries = _make_entries(100)
        result = analyzer.analyze_intervals(entries)
        assert result == []

    def test_disabled_returns_empty_on_transitions(self) -> None:
        """無効時はcompute_transitionsが空リストを返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        result = analyzer.compute_transitions()
        assert result == []

    def test_disabled_returns_empty_on_compare(self) -> None:
        """無効時はcompare_scenariosが空辞書を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        result = analyzer.compare_scenarios()
        assert result == {}

    def test_disabled_report_has_enabled_false(self) -> None:
        """無効時のレポートにenabled=Falseが含まれることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        report = analyzer.generate_report()
        assert report["enabled"] is False
        assert report["type"] == "policy_selection_analysis_report"

    def test_enabled_by_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数CYRENE_MONITOR=1で有効化されることを確認する。"""
        monkeypatch.setenv("CYRENE_MONITOR", "1")
        analyzer = PolicySelectionAnalysis()
        assert analyzer.enabled is True

    def test_disabled_by_default_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数未設定時に無効化されることを確認する。"""
        monkeypatch.delenv("CYRENE_MONITOR", raising=False)
        analyzer = PolicySelectionAnalysis()
        assert analyzer.enabled is False


# ── 内部ヘルパーテスト ──────────────────────────────────────────


class TestHelpers:
    """内部ヘルパー関数をテストする。"""

    def test_compute_label_counts_empty(self) -> None:
        """空リストで空辞書を返すことを確認する。"""
        result = _compute_label_counts([])
        assert result == {}

    def test_compute_label_counts_basic(self) -> None:
        """基本的なラベルカウントが正しいことを確認する。"""
        entries = [
            {"selected_label": "共感する"},
            {"selected_label": "共感する"},
            {"selected_label": "からかう"},
        ]
        result = _compute_label_counts(entries)
        assert result["共感する"] == 2
        assert result["からかう"] == 1

    def test_compute_label_counts_missing_label(self) -> None:
        """selected_labelが空のエントリがスキップされることを確認する。"""
        entries = [
            {"selected_label": "共感する"},
            {"selected_label": ""},
        ]
        result = _compute_label_counts(entries)
        assert result == {"共感する": 1}

    def test_compute_section_stats_empty(self) -> None:
        """空リストで空タプルを返すことを確認する。"""
        totals, variances = _compute_section_stats([])
        assert totals == {}
        assert variances == {}

    def test_compute_section_stats_basic(self) -> None:
        """基本的な断面統計が正しいことを確認する。"""
        entries = [
            {
                "candidates": [
                    {"score_breakdown": {"drive_goal_match": 1.0, "fear_bias": 0.5}},
                    {"score_breakdown": {"drive_goal_match": 0.5, "fear_bias": 0.3}},
                ]
            },
        ]
        totals, variances = _compute_section_stats(entries)
        assert abs(totals["drive_goal_match"] - 1.5) < 1e-6
        assert abs(totals["fear_bias"] - 0.8) < 1e-6

    def test_compute_section_stats_no_breakdown(self) -> None:
        """breakdownがないエントリがスキップされることを確認する。"""
        entries = [
            {"candidates": [{"score": 1.0}]},
        ]
        totals, variances = _compute_section_stats(entries)
        assert totals == {}
        assert variances == {}

    def test_compute_section_stats_variance(self) -> None:
        """分散が正しく算出されることを確認する。"""
        entries = [
            {"candidates": [{"score_breakdown": {"x": 1.0}}]},
            {"candidates": [{"score_breakdown": {"x": 3.0}}]},
        ]
        totals, variances = _compute_section_stats(entries)
        # 平均 = 2.0, 分散 = ((1-2)^2 + (3-2)^2) / 2 = 1.0
        assert abs(variances["x"] - 1.0) < 1e-6

    def test_compute_section_stats_single_value(self) -> None:
        """値が1つの場合の分散が0になることを確認する。"""
        entries = [
            {"candidates": [{"score_breakdown": {"x": 5.0}}]},
        ]
        totals, variances = _compute_section_stats(entries)
        assert variances["x"] == 0.0


# ── 第1段: 時間区間分割テスト ──────────────────────────────────────


class TestAnalyzeIntervals:
    """第1段パイプライン（時間区間分割）をテストする。"""

    def test_basic_interval_division(self) -> None:
        """基本的な区間分割が正しく行われることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        summaries = analyzer.analyze_intervals(entries, interval_size=25)
        assert len(summaries) == 4
        assert summaries[0].interval_index == 0
        assert summaries[3].interval_index == 3
        assert summaries[0].entry_count == 25

    def test_interval_tick_ranges(self) -> None:
        """区間のティック範囲が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        summaries = analyzer.analyze_intervals(entries, interval_size=50)
        assert summaries[0].tick_start == 1
        assert summaries[0].tick_end == 50
        assert summaries[1].tick_start == 51
        assert summaries[1].tick_end == 100

    def test_uneven_division(self) -> None:
        """エントリ数が区間サイズで割り切れない場合の処理を確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(73)
        summaries = analyzer.analyze_intervals(entries, interval_size=25)
        # 25 + 25 + 23 = 73
        assert len(summaries) == 3
        assert summaries[0].entry_count == 25
        assert summaries[1].entry_count == 25
        assert summaries[2].entry_count == 23

    def test_label_counts_per_interval(self) -> None:
        """区間ごとのラベル別選択回数が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        # 全て「共感する」の10件
        entries = _make_entries(10, label_pattern=["共感する"])
        summaries = analyzer.analyze_intervals(entries, interval_size=5)
        assert summaries[0].label_counts["共感する"] == 5
        assert summaries[1].label_counts["共感する"] == 5

    def test_section_totals_per_interval(self) -> None:
        """区間ごとの断面別寄与量合計が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(10)
        summaries = analyzer.analyze_intervals(entries, interval_size=10)
        assert len(summaries) == 1
        # 断面合計が存在することを確認する
        assert "drive_goal_match" in summaries[0].section_totals

    def test_empty_entries(self) -> None:
        """空リストで空リストが返ることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        result = analyzer.analyze_intervals([])
        assert result == []

    def test_single_entry(self) -> None:
        """1件のエントリで1区間が生成されることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(1)
        summaries = analyzer.analyze_intervals(entries, interval_size=50)
        assert len(summaries) == 1
        assert summaries[0].entry_count == 1

    def test_fifo_limit(self) -> None:
        """FIFO上限で古い区間が押し出されることを確認する。（安全弁3）"""
        config = PolicySelectionAnalysisConfig(interval_size=5, max_intervals=3)
        analyzer = PolicySelectionAnalysis(config=config, enabled=True)
        entries = _make_entries(25)
        summaries = analyzer.analyze_intervals(entries, interval_size=5)
        # 25 / 5 = 5区間だが、max_intervals=3で直近3区間のみ残る
        assert len(summaries) == 3
        # 最後の3区間（index 2, 3, 4）が残っているはず
        assert summaries[0].interval_index == 2
        assert summaries[-1].interval_index == 4

    def test_uses_config_interval_size(self) -> None:
        """interval_size未指定時は設定値が使われることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=10)
        analyzer = PolicySelectionAnalysis(config=config, enabled=True)
        entries = _make_entries(30)
        summaries = analyzer.analyze_intervals(entries)
        assert len(summaries) == 3

    def test_invalid_interval_size_uses_config(self) -> None:
        """不正なinterval_size指定時は設定値にフォールバックすることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=10)
        analyzer = PolicySelectionAnalysis(config=config, enabled=True)
        entries = _make_entries(30)
        summaries = analyzer.analyze_intervals(entries, interval_size=0)
        assert len(summaries) == 3

    def test_interval_summary_to_dict(self) -> None:
        """IntervalSummaryのto_dict変換が正しいことを確認する。"""
        summary = IntervalSummary(
            interval_index=0,
            tick_start=1,
            tick_end=50,
            entry_count=50,
            label_counts={"共感する": 30, "からかう": 20},
            section_totals={"drive_goal_match": 12.3456789},
            section_variances={"drive_goal_match": 0.1234567},
        )
        d = summary.to_dict()
        assert d["interval_index"] == 0
        assert d["label_counts"]["共感する"] == 30
        # 小数点以下6桁に丸められること
        assert d["section_totals"]["drive_goal_match"] == round(12.3456789, 6)
        assert d["section_variances"]["drive_goal_match"] == round(0.1234567, 6)


# ── 第2段: 区間間推移テスト ────────────────────────────────────────


class TestComputeTransitions:
    """第2段パイプライン（区間間推移の記述）をテストする。"""

    def test_basic_transitions(self) -> None:
        """基本的な推移計算が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        # 前半: 全て「共感する」、後半: 全て「からかう」
        entries = _make_entries(10, label_pattern=["共感する"]) + \
                  _make_entries(10, label_pattern=["からかう"])
        # tick番号を再設定
        for i, e in enumerate(entries):
            e["tick"] = i + 1
        summaries = analyzer.analyze_intervals(entries, interval_size=10)
        transitions = analyzer.compute_transitions()
        assert len(transitions) == 1
        t = transitions[0]
        assert t.from_interval == 0
        assert t.to_interval == 1
        # 共感するが10→0で変化量-10
        assert t.label_count_deltas.get("共感する", 0) == -10
        # からかうが0→10で変化量+10
        assert t.label_count_deltas.get("からかう", 0) == 10

    def test_no_transition_with_single_interval(self) -> None:
        """区間が1つの場合は推移がないことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(5)
        analyzer.analyze_intervals(entries, interval_size=10)
        transitions = analyzer.compute_transitions()
        assert len(transitions) == 0

    def test_multiple_transitions(self) -> None:
        """複数区間間の推移が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(30)
        analyzer.analyze_intervals(entries, interval_size=10)
        transitions = analyzer.compute_transitions()
        assert len(transitions) == 2
        assert transitions[0].from_interval == 0
        assert transitions[0].to_interval == 1
        assert transitions[1].from_interval == 1
        assert transitions[1].to_interval == 2

    def test_transition_section_deltas(self) -> None:
        """断面別寄与量の変化量が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(20)
        summaries = analyzer.analyze_intervals(entries, interval_size=10)
        transitions = analyzer.compute_transitions(summaries)
        assert len(transitions) == 1
        # 断面変化量が辞書として存在することを確認
        assert isinstance(transitions[0].section_total_deltas, dict)

    def test_transition_with_explicit_summaries(self) -> None:
        """明示的にsummariesを渡した場合の動作を確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        s1 = IntervalSummary(interval_index=0, label_counts={"共感する": 5})
        s2 = IntervalSummary(interval_index=1, label_counts={"共感する": 3, "からかう": 2})
        transitions = analyzer.compute_transitions([s1, s2])
        assert len(transitions) == 1
        assert transitions[0].label_count_deltas["共感する"] == -2
        assert transitions[0].label_count_deltas["からかう"] == 2

    def test_transition_to_dict(self) -> None:
        """IntervalTransitionのto_dict変換が正しいことを確認する。"""
        t = IntervalTransition(
            from_interval=0,
            to_interval=1,
            label_count_deltas={"共感する": -5},
            section_total_deltas={"drive_goal_match": 1.2345678},
        )
        d = t.to_dict()
        assert d["from_interval"] == 0
        assert d["to_interval"] == 1
        assert d["label_count_deltas"]["共感する"] == -5
        assert d["section_total_deltas"]["drive_goal_match"] == round(1.2345678, 6)


# ── 第3段: シナリオ間比較テスト ──────────────────────────────────


class TestScenarioComparison:
    """第3段パイプライン（シナリオ間比較）をテストする。"""

    def test_register_single_scenario(self) -> None:
        """シナリオ登録が正しく行われることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(50)
        summary = analyzer.register_scenario("test_scenario", entries)
        assert summary.scenario_name == "test_scenario"
        assert summary.total_entries == 50
        assert analyzer.scenario_count == 1

    def test_register_overwrites_existing(self) -> None:
        """同名シナリオの再登録が上書きされることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries1 = _make_entries(50)
        entries2 = _make_entries(30)
        analyzer.register_scenario("test", entries1)
        assert analyzer.scenario_count == 1
        summary = analyzer.register_scenario("test", entries2)
        assert analyzer.scenario_count == 1
        assert summary.total_entries == 30

    def test_compare_returns_empty_for_single_scenario(self) -> None:
        """1件のみの場合はcompare_scenariosが空を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(50)
        analyzer.register_scenario("only_one", entries)
        result = analyzer.compare_scenarios()
        assert result == {}

    def test_compare_two_scenarios(self) -> None:
        """2シナリオの比較が正しい構造を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries_a = _make_entries(50, label_pattern=["共感する"])
        entries_b = _make_entries(50, label_pattern=["からかう"])
        analyzer.register_scenario("scenario_a", entries_a)
        analyzer.register_scenario("scenario_b", entries_b)
        result = analyzer.compare_scenarios()
        assert "scenario_a" in result
        assert "scenario_b" in result
        assert result["scenario_a"]["label_counts"]["共感する"] == 50
        assert result["scenario_b"]["label_counts"]["からかう"] == 50

    def test_scenario_summary_to_dict(self) -> None:
        """ScenarioSummaryのto_dict変換が正しいことを確認する。"""
        summary = ScenarioSummary(
            scenario_name="test",
            total_entries=100,
            label_counts={"共感する": 60, "からかう": 40},
            section_totals={"drive_goal_match": 12.3456789},
            section_variances={"drive_goal_match": 0.1234567},
        )
        d = summary.to_dict()
        assert d["scenario_name"] == "test"
        assert d["total_entries"] == 100
        assert d["section_totals"]["drive_goal_match"] == round(12.3456789, 6)

    def test_scenario_label_counts(self) -> None:
        """シナリオ登録時のラベルカウントが正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(
            15, label_pattern=["共感する", "からかう", "励ます"]
        )
        summary = analyzer.register_scenario("test", entries)
        assert summary.label_counts["共感する"] == 5
        assert summary.label_counts["からかう"] == 5
        assert summary.label_counts["励ます"] == 5


# ── 第4段: 構造化出力テスト ────────────────────────────────────────


class TestGenerateReport:
    """第4段パイプライン（構造化出力の生成）をテストする。"""

    def test_report_structure(self) -> None:
        """レポートが正しい構造を持つことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        assert report["type"] == "policy_selection_analysis_report"
        assert report["enabled"] is True
        assert "timestamp" in report
        assert "interval_summaries" in report
        assert "interval_transitions" in report
        assert "scenario_comparison" in report
        assert "interval_count" in report
        assert "scenario_count" in report

    def test_report_interval_count(self) -> None:
        """レポートの区間数が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        assert report["interval_count"] == 4
        assert len(report["interval_summaries"]) == 4

    def test_report_transitions_count(self) -> None:
        """レポートの推移数が区間数-1であることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        assert len(report["interval_transitions"]) == 3

    def test_report_with_scenarios(self) -> None:
        """シナリオ比較が含まれるレポートの構造を確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(50)
        analyzer.analyze_intervals(entries, interval_size=25)
        analyzer.register_scenario("s1", _make_entries(30))
        analyzer.register_scenario("s2", _make_entries(40))
        report = analyzer.generate_report()
        assert report["scenario_count"] == 2
        assert "s1" in report["scenario_comparison"]
        assert "s2" in report["scenario_comparison"]

    def test_report_without_scenarios(self) -> None:
        """シナリオ未登録時は比較が空であることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(50)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        assert report["scenario_comparison"] == {}

    def test_report_is_json_serializable(self) -> None:
        """レポートがJSON直列化可能であることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        analyzer.register_scenario("s1", _make_entries(30))
        analyzer.register_scenario("s2", _make_entries(40))
        report = analyzer.generate_report()
        # JSON直列化が例外を出さないことを確認
        json_text = json.dumps(report, ensure_ascii=False)
        assert isinstance(json_text, str)

    def test_empty_report(self) -> None:
        """分析前のレポートが空の構造を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        report = analyzer.generate_report()
        assert report["interval_count"] == 0
        assert report["scenario_count"] == 0
        assert report["interval_summaries"] == []
        assert report["interval_transitions"] == []


# ── ログ基盤連携テスト ──────────────────────────────────────────


class TestAnalyzeFromLog:
    """PolicySelectionLogインスタンスからの直接分析をテストする。"""

    def test_analyze_from_mock_log(self) -> None:
        """モックのPolicySelectionLogから分析できることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.return_value = _make_entries(100)
        report = analyzer.analyze_from_log(mock_log, interval_size=25)
        assert report["interval_count"] == 4
        mock_log.get_entries.assert_called_once()

    def test_analyze_from_log_read_only(self) -> None:
        """ログ基盤への書き込みメソッドが呼ばれないことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.return_value = _make_entries(50)
        analyzer.analyze_from_log(mock_log, interval_size=25)
        # record メソッドが呼ばれていないことを確認（書き込み遮断）
        mock_log.record.assert_not_called()

    def test_analyze_from_log_exception_handling(self) -> None:
        """ログ取得が例外を出した場合でも安全に空レポートを返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        mock_log = MagicMock()
        mock_log.get_entries.side_effect = RuntimeError("test error")
        report = analyzer.analyze_from_log(mock_log)
        assert report["type"] == "policy_selection_analysis_report"
        assert report["interval_count"] == 0

    def test_analyze_from_log_disabled(self) -> None:
        """無効時はログを読み取らないことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        mock_log = MagicMock()
        report = analyzer.analyze_from_log(mock_log)
        mock_log.get_entries.assert_not_called()
        assert report["enabled"] is False


# ── シミュレーション結果連携テスト ──────────────────────────────────


class TestAnalyzeFromSimulation:
    """長期シミュレーション結果からの分析をテストする。"""

    def test_analyze_from_sim_result(self) -> None:
        """シミュレーション結果から正しくシナリオ登録されることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        sim_result = _make_sim_result("test_scenario", turns=60)
        summary = analyzer.analyze_from_simulation(sim_result)
        assert summary.scenario_name == "test_scenario"
        assert summary.total_entries == 60

    def test_analyze_from_sim_with_name_override(self) -> None:
        """シナリオ名の明示的指定が機能することを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        sim_result = _make_sim_result("original_name", turns=30)
        summary = analyzer.analyze_from_simulation(sim_result, scenario_name="override")
        assert summary.scenario_name == "override"

    def test_analyze_from_sim_label_distribution(self) -> None:
        """シミュレーション結果のラベル分布が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        sim_result = _make_sim_result(
            "test", turns=30, label_pattern=["共感する", "からかう", "励ます"]
        )
        summary = analyzer.analyze_from_simulation(sim_result)
        assert summary.label_counts["共感する"] == 10
        assert summary.label_counts["からかう"] == 10
        assert summary.label_counts["励ます"] == 10

    def test_compare_multiple_sim_results(self) -> None:
        """複数シミュレーション結果の比較が正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        sim_a = _make_sim_result("stable", turns=50, label_pattern=["共感する"])
        sim_b = _make_sim_result("mixed", turns=50, label_pattern=["からかう", "励ます"])
        analyzer.analyze_from_simulation(sim_a)
        analyzer.analyze_from_simulation(sim_b)
        comparison = analyzer.compare_scenarios()
        assert "stable" in comparison
        assert "mixed" in comparison

    def test_analyze_from_sim_empty_turns(self) -> None:
        """空のターンリストでも正常に処理されることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        sim_result = {"metadata": {"scenario": "empty"}, "turns": []}
        summary = analyzer.analyze_from_simulation(sim_result)
        assert summary.total_entries == 0
        assert summary.label_counts == {}

    def test_analyze_from_sim_disabled(self) -> None:
        """無効時はシミュレーション分析をスキップすることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        sim_result = _make_sim_result("test", turns=30)
        summary = analyzer.analyze_from_simulation(sim_result)
        assert summary.scenario_name == "test"
        assert summary.total_entries == 0


# ── アクセサテスト ──────────────────────────────────────────────


class TestAccessors:
    """読み取り専用アクセサをテストする。"""

    def test_get_interval_summaries(self) -> None:
        """区間別集計結果のアクセサが辞書形式を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(50)
        analyzer.analyze_intervals(entries, interval_size=25)
        summaries = analyzer.get_interval_summaries()
        assert len(summaries) == 2
        assert isinstance(summaries[0], dict)
        assert "interval_index" in summaries[0]

    def test_get_scenario_summaries(self) -> None:
        """シナリオ別集計結果のアクセサが辞書形式を返すことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        analyzer.register_scenario("s1", _make_entries(30))
        summaries = analyzer.get_scenario_summaries()
        assert "s1" in summaries
        assert isinstance(summaries["s1"], dict)
        assert "scenario_name" in summaries["s1"]

    def test_interval_count_property(self) -> None:
        """interval_countプロパティが正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        assert analyzer.interval_count == 0
        analyzer.analyze_intervals(_make_entries(50), interval_size=25)
        assert analyzer.interval_count == 2

    def test_scenario_count_property(self) -> None:
        """scenario_countプロパティが正しいことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        assert analyzer.scenario_count == 0
        analyzer.register_scenario("s1", _make_entries(30))
        assert analyzer.scenario_count == 1


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """設計書記載の安全弁を検証する。"""

    def test_no_evaluative_language_in_report(self) -> None:
        """レポート出力に評価的語彙が含まれないことを確認する。（安全弁1）"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        report_text = json.dumps(report, ensure_ascii=False)
        # 評価的語彙が含まれていないことを確認
        evaluative_words = [
            "不足", "過剰", "問題", "改善", "推奨", "すべき",
            "望ましい", "異常", "正常", "偏り", "支配的",
        ]
        for word in evaluative_words:
            assert word not in report_text, f"Report contains evaluative word: {word}"

    def test_no_recommendation_in_output(self) -> None:
        """出力に推奨が含まれないことを確認する。（安全弁2）"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(100)
        analyzer.analyze_intervals(entries, interval_size=25)
        report = analyzer.generate_report()
        # 推奨に関連するキーが存在しないことを確認
        assert "recommendation" not in report
        assert "suggestion" not in report

    def test_fifo_upper_limit(self) -> None:
        """FIFO上限が機能することを確認する。（安全弁3）"""
        config = PolicySelectionAnalysisConfig(interval_size=5, max_intervals=2)
        analyzer = PolicySelectionAnalysis(config=config, enabled=True)
        entries = _make_entries(25)  # 5区間生成
        summaries = analyzer.analyze_intervals(entries, interval_size=5)
        assert len(summaries) == 2  # max_intervals=2で制限

    def test_disabled_skips_all_processing(self) -> None:
        """無効時は全処理がスキップされることを確認する。（安全弁4）"""
        analyzer = PolicySelectionAnalysis(enabled=False)
        entries = _make_entries(100)
        assert analyzer.analyze_intervals(entries) == []
        assert analyzer.compute_transitions() == []
        assert analyzer.compare_scenarios() == {}
        report = analyzer.generate_report()
        assert report["enabled"] is False

    def test_analysis_state_independence(self) -> None:
        """過去の分析結果が将来の分析に影響しないことを確認する。（安全弁5）"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries1 = _make_entries(50, label_pattern=["共感する"])
        entries2 = _make_entries(50, label_pattern=["からかう"])

        # 1回目の分析
        summaries1 = analyzer.analyze_intervals(entries1, interval_size=25)
        # 2回目の分析（完全に独立）
        summaries2 = analyzer.analyze_intervals(entries2, interval_size=25)

        # 2回目の結果が1回目の影響を受けていないことを確認
        assert summaries2[0].label_counts.get("共感する", 0) == 0
        assert summaries2[0].label_counts.get("からかう", 0) == 25


# ── 構造的分離テスト ────────────────────────────────────────────


class TestStructuralSeparation:
    """psycheモジュールからの構造的分離を検証する。"""

    def test_no_psyche_import(self) -> None:
        """psycheモジュールをインポートしていないことを確認する。"""
        import tools.policy_selection_analysis as module
        source = open(module.__file__, "r", encoding="utf-8").read()
        # psyche直接インポートがないことを確認
        assert "from psyche" not in source
        assert "import psyche" not in source

    def test_no_enrichment_output(self) -> None:
        """enrichment出力メソッドが存在しないことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        # enrichmentに関連するメソッドが存在しないことを確認
        assert not hasattr(analyzer, "get_enrichment")
        assert not hasattr(analyzer, "get_prompt_enrichment")
        assert not hasattr(analyzer, "to_enrichment")

    def test_no_save_load_methods(self) -> None:
        """save/loadメソッドが存在しないことを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        assert not hasattr(analyzer, "save")
        assert not hasattr(analyzer, "load")
        assert not hasattr(analyzer, "to_dict")
        assert not hasattr(analyzer, "from_dict")

    def test_no_state_mutation_on_analysis(self) -> None:
        """分析実行がポリシー選択ログの状態を変更しないことを確認する。"""
        from tools.policy_selection_log import PolicySelectionLog
        log = PolicySelectionLog(enabled=True)
        # いくつかのエントリを記録する
        for i in range(10):
            log.record(
                tick=i + 1,
                selected_label="共感する",
                candidates=[{"policy_label": "共感する", "_score": 2.0}],
                selected_count=3,
            )
        original_count = log.entry_count

        # 分析を実行する
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = log.get_entries()
        analyzer.analyze_intervals(entries, interval_size=5)

        # ログの状態が変わっていないことを確認
        assert log.entry_count == original_count


# ── ファクトリ関数テスト ────────────────────────────────────────


class TestFactory:
    """ファクトリ関数をテストする。"""

    def test_create_with_defaults(self) -> None:
        """デフォルト設定でインスタンスが作成されることを確認する。"""
        analyzer = create_policy_selection_analysis(enabled=True)
        assert isinstance(analyzer, PolicySelectionAnalysis)
        assert analyzer.enabled is True

    def test_create_with_custom_config(self) -> None:
        """カスタム設定でインスタンスが作成されることを確認する。"""
        config = PolicySelectionAnalysisConfig(interval_size=100, max_intervals=5)
        analyzer = create_policy_selection_analysis(config=config, enabled=True)
        assert analyzer.config.interval_size == 100
        assert analyzer.config.max_intervals == 5

    def test_create_with_env_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数未設定時にデフォルト無効で作成されることを確認する。"""
        monkeypatch.delenv("CYRENE_MONITOR", raising=False)
        analyzer = create_policy_selection_analysis()
        assert analyzer.enabled is False


# ── 統合テスト ──────────────────────────────────────────────────


class TestIntegration:
    """全4段パイプラインの統合テスト。"""

    def test_full_pipeline(self) -> None:
        """全段のパイプラインが正しく連動することを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)

        # 第1段: 200エントリを4区間に分割
        entries = _make_entries(200)
        summaries = analyzer.analyze_intervals(entries, interval_size=50)
        assert len(summaries) == 4

        # 第2段: 3つの推移を算出
        transitions = analyzer.compute_transitions()
        assert len(transitions) == 3

        # 第3段: 2シナリオを登録して比較
        analyzer.register_scenario("stable", _make_entries(100, ["共感する"]))
        analyzer.register_scenario("varied", _make_entries(100))
        comparison = analyzer.compare_scenarios()
        assert len(comparison) == 2

        # 第4段: レポート生成
        report = analyzer.generate_report()
        assert report["interval_count"] == 4
        assert report["scenario_count"] == 2
        assert len(report["interval_transitions"]) == 3

    def test_full_pipeline_with_sim_results(self) -> None:
        """シミュレーション結果を含む全段パイプラインを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)

        # 2つのシミュレーション結果を登録
        sim_a = _make_sim_result("stable", turns=50, label_pattern=["共感する"])
        sim_b = _make_sim_result("mixed", turns=50, label_pattern=["からかう", "励ます"])
        analyzer.analyze_from_simulation(sim_a)
        analyzer.analyze_from_simulation(sim_b)

        # 比較が可能であることを確認
        comparison = analyzer.compare_scenarios()
        assert "stable" in comparison
        assert "mixed" in comparison

        # レポートにシナリオ情報が含まれることを確認
        report = analyzer.generate_report()
        assert report["scenario_count"] == 2

    def test_repeated_analysis_independence(self) -> None:
        """繰り返し分析しても結果が互いに独立であることを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)

        # 1回目: 全て同じラベル
        entries1 = _make_entries(50, label_pattern=["共感する"])
        summaries1 = analyzer.analyze_intervals(entries1, interval_size=25)
        report1 = analyzer.generate_report()

        # 2回目: 全て異なるラベル
        entries2 = _make_entries(50, label_pattern=["からかう"])
        summaries2 = analyzer.analyze_intervals(entries2, interval_size=25)
        report2 = analyzer.generate_report()

        # 2回目のレポートに1回目のデータが混入していないこと
        for summary in report2["interval_summaries"]:
            assert "共感する" not in summary["label_counts"]
            assert summary["label_counts"].get("からかう", 0) > 0

    def test_large_dataset(self) -> None:
        """大量データ（500件）でも正常に動作することを確認する。"""
        analyzer = PolicySelectionAnalysis(enabled=True)
        entries = _make_entries(500)
        summaries = analyzer.analyze_intervals(entries, interval_size=50)
        assert len(summaries) == 10
        transitions = analyzer.compute_transitions()
        assert len(transitions) == 9
        report = analyzer.generate_report()
        assert report["interval_count"] == 10
