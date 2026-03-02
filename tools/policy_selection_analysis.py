"""
tools/policy_selection_analysis.py - 動的ポリシー選択の長期効果検証ツール

方針選択処理のスコア内訳（断面別寄与量）を長期にわたって集計し、
時間区間分割・区間間推移・シナリオ間比較の3段で構造化して記述する。

設計書: design_policy_selection_analysis.md

本機能の構造的分離:
- psycheモジュール群には配置しない。外部ツール群(tools/)に配置する。
- 方針選択処理の入力（状態・知覚・記憶・責任影響・判断バイアス・拡張入力）を一切変更しない
- 方針選択処理の出力（候補リスト・選択結果）を一切変更しない
- ポリシー選択ログ基盤の蓄積データを読み取るのみであり、書き込まない
- prompt enrichment への出力経路を構造的に持たない
- 方針選択処理への逆流経路を構造的に持たない
- 全内部状態はセッション境界で消失する（永続化対象外）
- save/loadの対象フィールドに一切追加しない
- 本機能の有効/無効がpsycheの動作に影響しない（完全に透過的）

安全弁:
1. 規範的判断の構造的排除: 数値的事実のみ出力。評価的語彙を含めない
2. 推奨生成の禁止: 分析結果に基づく推奨を生成する仕組みを持たない
3. FIFO上限による蓄積量制御: 区間別集計結果のリストにFIFO上限
4. 環境変数による完全無効化: CYRENE_MONITOR=1 で有効化
5. 分析結果の状態非依存性: 過去の分析結果が将来の分析に影響しない
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# 既存の実行時観測基盤(execution_monitor.py)と同じログ名前空間を使用する
monitor_logger = logging.getLogger("cyrene.monitor")


# ── 環境変数制御 ──────────────────────────────────────────────────


def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    安全弁4: 無効時は全ての処理をスキップし、空の結果を返す。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── 設定 ──────────────────────────────────────────────────────────


@dataclass
class PolicySelectionAnalysisConfig:
    """ポリシー選択分析の設定パラメータ。

    Attributes:
        interval_size: 1区間あたりのティック数。
            ポリシー選択ログをこのサイズで等分割する。
        max_intervals: 区間別集計結果のFIFO上限（安全弁3）。
            この数を超えると古い区間の集計が押し出される。
    """
    # 1区間あたりのティック数（デフォルト50件で1区間）
    interval_size: int = 50
    # 区間別集計結果のFIFO上限（安全弁3）
    max_intervals: int = 20

    def __post_init__(self) -> None:
        """設定値のバリデーション。不正な値はデフォルトに戻す。"""
        if self.interval_size < 1:
            self.interval_size = 50
        if self.max_intervals < 1:
            self.max_intervals = 20


# ── 区間別集計データ ──────────────────────────────────────────────


@dataclass
class IntervalSummary:
    """1つの時間区間に対する集計結果。

    安全弁1: 数値的事実のみを格納する。評価的判断を含めない。

    Attributes:
        interval_index: 区間の通し番号（0始まり）
        tick_start: この区間の最初のティック番号
        tick_end: この区間の最後のティック番号
        entry_count: この区間に含まれるログエントリの件数
        label_counts: ポリシーラベル別の選択回数
        section_totals: 断面別の寄与量合計
        section_variances: 断面別の寄与量分散
    """
    interval_index: int = 0
    tick_start: int = 0
    tick_end: int = 0
    entry_count: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    section_totals: dict[str, float] = field(default_factory=dict)
    section_variances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """集計結果を辞書形式に変換する。"""
        return {
            "interval_index": self.interval_index,
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
            "entry_count": self.entry_count,
            "label_counts": dict(self.label_counts),
            "section_totals": {
                k: round(v, 6) for k, v in self.section_totals.items()
            },
            "section_variances": {
                k: round(v, 6) for k, v in self.section_variances.items()
            },
        }


# ── 区間間推移データ ──────────────────────────────────────────────


@dataclass
class IntervalTransition:
    """隣接する2区間間の変化量を記述する。

    安全弁1: 数値的事実のみ。「なぜ変化したか」は記述しない。

    Attributes:
        from_interval: 変化元の区間番号
        to_interval: 変化先の区間番号
        label_count_deltas: ポリシーラベル別の選択回数変化量
        section_total_deltas: 断面別の寄与量合計変化量
    """
    from_interval: int = 0
    to_interval: int = 0
    label_count_deltas: dict[str, int] = field(default_factory=dict)
    section_total_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """推移データを辞書形式に変換する。"""
        return {
            "from_interval": self.from_interval,
            "to_interval": self.to_interval,
            "label_count_deltas": dict(self.label_count_deltas),
            "section_total_deltas": {
                k: round(v, 6) for k, v in self.section_total_deltas.items()
            },
        }


# ── シナリオ別集計データ ──────────────────────────────────────────


@dataclass
class ScenarioSummary:
    """1つのシナリオに対するポリシー選択分布と断面別寄与量分布。

    安全弁1: 数値的事実のみ。シナリオ間の優劣判断を含めない。

    Attributes:
        scenario_name: シナリオの名前
        total_entries: シナリオ内の全ログエントリ数
        label_counts: ポリシーラベル別の選択回数
        section_totals: 断面別の寄与量合計
        section_variances: 断面別の寄与量分散
    """
    scenario_name: str = ""
    total_entries: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    section_totals: dict[str, float] = field(default_factory=dict)
    section_variances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """シナリオ集計を辞書形式に変換する。"""
        return {
            "scenario_name": self.scenario_name,
            "total_entries": self.total_entries,
            "label_counts": dict(self.label_counts),
            "section_totals": {
                k: round(v, 6) for k, v in self.section_totals.items()
            },
            "section_variances": {
                k: round(v, 6) for k, v in self.section_variances.items()
            },
        }


# ── 内部ヘルパー ────────────────────────────────────────────────


def _compute_section_stats(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """ログエントリ群から断面別の寄与量合計と分散を算出する。

    Args:
        entries: ログエントリの辞書リスト。各エントリは candidates を含む。

    Returns:
        (section_totals, section_variances) のタプル
    """
    section_sums: dict[str, float] = defaultdict(float)
    section_values: dict[str, list[float]] = defaultdict(list)

    for entry in entries:
        for cand in entry.get("candidates", []):
            breakdown = cand.get("score_breakdown")
            if not breakdown or not isinstance(breakdown, dict):
                continue
            for section_name, contrib in breakdown.items():
                if isinstance(contrib, (int, float)):
                    section_sums[section_name] += contrib
                    section_values[section_name].append(contrib)

    section_variances: dict[str, float] = {}
    for section_name, values in section_values.items():
        if len(values) < 2:
            section_variances[section_name] = 0.0
            continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        section_variances[section_name] = variance

    return dict(section_sums), section_variances


def _compute_label_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    """ログエントリ群からポリシーラベル別選択回数を算出する。

    Args:
        entries: ログエントリの辞書リスト。各エントリは selected_label を含む。

    Returns:
        ポリシーラベル別選択回数の辞書
    """
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        label = entry.get("selected_label", "")
        if label:
            counts[label] += 1
    return dict(counts)


# ── PolicySelectionAnalysis 本体 ──────────────────────────────────


class PolicySelectionAnalysis:
    """ポリシー選択の長期効果検証を行う本体クラス。

    セッション開始時にインスタンスを生成し、セッション終了時に破棄する。
    全内部状態はインスタンス破棄時に消失する（永続化対象外）。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない（READ-ONLY観測のみ）
    - prompt enrichment への出力経路を持たない
    - 方針選択処理への逆流経路を持たない
    - ポリシー選択ログ基盤への書き込みを行わない
    """

    def __init__(
        self,
        config: Optional[PolicySelectionAnalysisConfig] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            config: 設定。Noneの場合はデフォルト設定を使用する。
            enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。
        """
        self._config = config or PolicySelectionAnalysisConfig()
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # 区間別集計結果のリスト（安全弁3: FIFO上限で古い結果を押し出す）
        self._interval_summaries: list[IntervalSummary] = []

        # シナリオ別集計結果の辞書（シナリオ名をキー）
        self._scenario_summaries: dict[str, ScenarioSummary] = {}

    @property
    def enabled(self) -> bool:
        """分析が有効かどうかを返す。"""
        return self._enabled

    @property
    def config(self) -> PolicySelectionAnalysisConfig:
        """設定を読み取り専用で返す。"""
        return self._config

    @property
    def interval_count(self) -> int:
        """現在保持している区間別集計結果の件数を返す。"""
        return len(self._interval_summaries)

    @property
    def scenario_count(self) -> int:
        """現在保持しているシナリオ別集計結果の件数を返す。"""
        return len(self._scenario_summaries)

    # ── 第1段: 時間区間分割 ───────────────────────────────────────

    def analyze_intervals(
        self,
        entries: list[dict[str, Any]],
        interval_size: Optional[int] = None,
    ) -> list[IntervalSummary]:
        """ポリシー選択ログを時間区間に分割し、区間ごとの集計を行う。

        第1段パイプライン処理。蓄積されたログエントリを等分割し、
        区間ごとにポリシーラベル別選択回数と断面別寄与量の合計・分散を算出する。

        安全弁5: 各分析は入力データのみから決定論的に算出される。
        過去の分析結果が将来の分析に影響しない。

        Args:
            entries: ポリシー選択ログのエントリ辞書リスト。
                PolicySelectionLog.get_entries() の戻り値を想定する。
            interval_size: 区間あたりのエントリ数。
                Noneの場合は設定値を使用する。

        Returns:
            区間別集計結果のリスト
        """
        if not self._enabled:
            return []

        if not entries:
            return []

        size = interval_size if interval_size is not None else self._config.interval_size
        if size < 1:
            size = self._config.interval_size

        summaries: list[IntervalSummary] = []

        # エントリを等分割する
        total = len(entries)
        interval_idx = 0
        for start in range(0, total, size):
            end = min(start + size, total)
            chunk = entries[start:end]
            if not chunk:
                continue

            # 区間の最初と最後のティック番号を取得する
            tick_start = chunk[0].get("tick", 0)
            tick_end = chunk[-1].get("tick", 0)

            # ラベル別選択回数を算出する
            label_counts = _compute_label_counts(chunk)

            # 断面別寄与量の合計・分散を算出する
            section_totals, section_variances = _compute_section_stats(chunk)

            summary = IntervalSummary(
                interval_index=interval_idx,
                tick_start=tick_start,
                tick_end=tick_end,
                entry_count=len(chunk),
                label_counts=label_counts,
                section_totals=section_totals,
                section_variances=section_variances,
            )
            summaries.append(summary)
            interval_idx += 1

        # 内部状態として保持する（安全弁3: FIFO上限）
        self._interval_summaries = summaries[-self._config.max_intervals:]

        return list(self._interval_summaries)

    # ── 第2段: 区間間推移の記述 ──────────────────────────────────

    def compute_transitions(
        self,
        summaries: Optional[list[IntervalSummary]] = None,
    ) -> list[IntervalTransition]:
        """隣接する区間間のポリシー選択傾向の変化量を算出する。

        第2段パイプライン処理。隣接する区間間で、
        ポリシーラベル別選択回数の変化量と断面別寄与量の変化量を算出する。

        安全弁1: 因果帰属は行わない（「なぜ変化したか」は記述しない）。

        Args:
            summaries: 区間別集計結果のリスト。
                Noneの場合は内部保持のものを使用する。

        Returns:
            区間間推移データのリスト
        """
        if not self._enabled:
            return []

        intervals = summaries if summaries is not None else self._interval_summaries
        if len(intervals) < 2:
            return []

        transitions: list[IntervalTransition] = []

        for i in range(len(intervals) - 1):
            prev = intervals[i]
            curr = intervals[i + 1]

            # ラベル別選択回数の変化量
            all_labels = set(prev.label_counts.keys()) | set(curr.label_counts.keys())
            label_deltas: dict[str, int] = {}
            for label in all_labels:
                prev_count = prev.label_counts.get(label, 0)
                curr_count = curr.label_counts.get(label, 0)
                label_deltas[label] = curr_count - prev_count

            # 断面別寄与量合計の変化量
            all_sections = set(prev.section_totals.keys()) | set(curr.section_totals.keys())
            section_deltas: dict[str, float] = {}
            for section in all_sections:
                prev_total = prev.section_totals.get(section, 0.0)
                curr_total = curr.section_totals.get(section, 0.0)
                section_deltas[section] = curr_total - prev_total

            transition = IntervalTransition(
                from_interval=prev.interval_index,
                to_interval=curr.interval_index,
                label_count_deltas=label_deltas,
                section_total_deltas=section_deltas,
            )
            transitions.append(transition)

        return transitions

    # ── 第3段: シナリオ間比較の記述 ──────────────────────────────

    def register_scenario(
        self,
        scenario_name: str,
        entries: list[dict[str, Any]],
    ) -> ScenarioSummary:
        """シナリオの実行結果からポリシー選択分布を集計して登録する。

        第3段パイプライン処理。シナリオごとのポリシー選択分布と
        断面別寄与量分布を並置記述用に蓄積する。

        安全弁1: シナリオ間の優劣判断は行わない。

        Args:
            scenario_name: シナリオ名
            entries: このシナリオのポリシー選択ログエントリリスト。

        Returns:
            シナリオ集計結果
        """
        if not self._enabled:
            return ScenarioSummary(scenario_name=scenario_name)

        label_counts = _compute_label_counts(entries)
        section_totals, section_variances = _compute_section_stats(entries)

        summary = ScenarioSummary(
            scenario_name=scenario_name,
            total_entries=len(entries),
            label_counts=label_counts,
            section_totals=section_totals,
            section_variances=section_variances,
        )

        # シナリオ実行ごとに上書きされる
        self._scenario_summaries[scenario_name] = summary

        return summary

    def compare_scenarios(self) -> dict[str, Any]:
        """登録済みの全シナリオのポリシー選択分布を並置して返す。

        安全弁1: シナリオ間の優劣判断を行わない。事実の並置のみ。
        安全弁2: 推奨を生成しない。

        Returns:
            シナリオ名をキーとしたシナリオ集計結果の辞書。
            2件未満の場合は空辞書を返す。
        """
        if not self._enabled:
            return {}

        if len(self._scenario_summaries) < 2:
            return {}

        return {
            name: summary.to_dict()
            for name, summary in self._scenario_summaries.items()
        }

    # ── 第4段: 構造化出力の生成 ──────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """全分析結果を構造化辞書として整理し返却する。

        第4段パイプライン処理。区間別集計・区間間推移・シナリオ間比較の
        結果を1つの構造化辞書にまとめる。

        安全弁1: 評価的語彙を出力に含めない。事実記述のみ。

        Returns:
            構造化されたレポート辞書
        """
        if not self._enabled:
            return {
                "type": "policy_selection_analysis_report",
                "enabled": False,
                "timestamp": time.time(),
            }

        # 区間間推移を算出
        transitions = self.compute_transitions()

        report: dict[str, Any] = {
            "type": "policy_selection_analysis_report",
            "enabled": True,
            "timestamp": time.time(),
            "interval_summaries": [s.to_dict() for s in self._interval_summaries],
            "interval_transitions": [t.to_dict() for t in transitions],
            "scenario_comparison": self.compare_scenarios(),
            "interval_count": len(self._interval_summaries),
            "scenario_count": len(self._scenario_summaries),
        }

        # ログストリームにJSON形式で出力する
        try:
            text = json.dumps(report, ensure_ascii=False, default=str)
            monitor_logger.debug(text)
        except Exception:
            pass

        return report

    # ── 便利メソッド: ログ基盤からの直接分析 ──────────────────────

    def analyze_from_log(
        self,
        log_instance: Any,
        interval_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """PolicySelectionLog インスタンスから直接ログを読み取り分析する。

        読み取り専用アクセサ(get_entries)のみを使用する。
        ポリシー選択ログ基盤の内部状態を変更しない。

        Args:
            log_instance: PolicySelectionLog のインスタンス。
                get_entries() メソッドを持つこと。
            interval_size: 区間あたりのエントリ数（オプション）。

        Returns:
            generate_report() と同じ構造の辞書
        """
        if not self._enabled:
            return self.generate_report()

        try:
            entries = log_instance.get_entries()
        except Exception:
            return self.generate_report()

        self.analyze_intervals(entries, interval_size=interval_size)
        return self.generate_report()

    def analyze_from_simulation(
        self,
        sim_result: dict[str, Any],
        scenario_name: Optional[str] = None,
    ) -> ScenarioSummary:
        """長期シミュレーション結果からポリシー選択情報を抽出し登録する。

        シミュレーション結果の時系列記録からポリシー選択ログエントリ相当の
        情報を構築し、シナリオとして登録する。

        読み取り専用で参照する。シミュレーションの実行制御には関与しない。

        Args:
            sim_result: long_term_sim.run_simulation() の戻り値。
            scenario_name: シナリオ名。Noneの場合はメタデータから取得する。

        Returns:
            シナリオ集計結果
        """
        # メタデータからシナリオ名を取得する
        name = scenario_name
        if name is None:
            metadata = sim_result.get("metadata", {})
            name = metadata.get("scenario", "unknown")

        if not self._enabled:
            return ScenarioSummary(scenario_name=name)

        # シミュレーション結果のターン記録からポリシー選択情報を抽出する
        turns = sim_result.get("turns", [])
        entries: list[dict[str, Any]] = []

        for turn in turns:
            policy_info = turn.get("policy", {})
            tick = turn.get("tick", 0)
            selected_label = policy_info.get("policy_label", "")

            # シミュレーション結果にはcandidatesの詳細(breakdown)がないため
            # 選択されたポリシーとスコアのみの簡易エントリを作成する
            entry: dict[str, Any] = {
                "tick": tick,
                "timestamp": 0.0,
                "selected_label": selected_label,
                "candidates": [],
                "candidate_count": 0,
                "selected_count": 0,
            }
            entries.append(entry)

        return self.register_scenario(name, entries)

    # ── アクセサ ────────────────────────────────────────────────

    def get_interval_summaries(self) -> list[dict[str, Any]]:
        """区間別集計結果を辞書形式で返す。読み取り専用アクセサ。"""
        return [s.to_dict() for s in self._interval_summaries]

    def get_scenario_summaries(self) -> dict[str, dict[str, Any]]:
        """シナリオ別集計結果を辞書形式で返す。読み取り専用アクセサ。"""
        return {
            name: summary.to_dict()
            for name, summary in self._scenario_summaries.items()
        }


# ── ファクトリ関数 ────────────────────────────────────────────────


def create_policy_selection_analysis(
    config: Optional[PolicySelectionAnalysisConfig] = None,
    enabled: Optional[bool] = None,
) -> PolicySelectionAnalysis:
    """PolicySelectionAnalysis のファクトリ関数。

    Args:
        config: 設定。Noneの場合はデフォルト設定を使用する。
        enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。

    Returns:
        PolicySelectionAnalysis インスタンス
    """
    return PolicySelectionAnalysis(config=config, enabled=enabled)
