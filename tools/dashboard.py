"""
tools/dashboard.py - 開発者向け統合ダッシュボード（CLIベース）

設計書: design_dashboard.md

分散した複数の実行時観測ツールの出力を一箇所で閲覧する手段を提供する。
各ツールの既存の読み取り専用アクセサのみを参照し、新しいデータ収集を一切行わない。

本機能の構造的分離:
- 各ツールの読み取り専用アクセサのみを参照する一方向参照
- psycheの状態変数に書き込むメソッドが存在しない
- enrichmentの項目として追加するメソッドが存在しない
- 各ツールインスタンスの書き込みメソッドを呼び出さない
- CLIからの手動起動のみであり、自動実行経路が存在しない
- orchestratorのPhase処理に組み込まれない
- save/loadの対象フィールドに追加しない
- 独自の内部状態を保持しない

安全弁:
1. enrichment経路の構造的遮断: enrichment出力を生成する関数・メソッドを持たない
2. psyche状態非接続: psycheの状態変数に書き込むメソッドが存在しない
3. 永続化の対象外: save/loadのフィールドに追加しない
4. 各ツール読み取り時の例外安全: 1つのツールの読み取り失敗が他のセクションの
   表示を阻害しない
5. 環境変数による完全無効化: 既存のモニタリング基盤と同一の環境変数制御に従う
6. 評価的語彙の除去: 数値と事実ラベルのみ
7. 修復機能の構造的排除: 状態変更・修復・リセットを行うメソッドが存在しない
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


# ── セクション識別子 ──────────────────────────────────────────────

SECTION_SESSION = "session"
SECTION_PIPELINE = "pipeline"
SECTION_BAND = "band"
SECTION_POLICY = "policy"
SECTION_EXPRESSION = "expression"
SECTION_PATHWAY = "pathway"
SECTION_ENRICHMENT = "enrichment"
SECTION_ANOMALY = "anomaly"

ALL_SECTIONS = (
    SECTION_SESSION,
    SECTION_PIPELINE,
    SECTION_BAND,
    SECTION_POLICY,
    SECTION_EXPRESSION,
    SECTION_PATHWAY,
    SECTION_ENRICHMENT,
    SECTION_ANOMALY,
)


# ── Dashboard ─────────────────────────────────────────────────────


class Dashboard:
    """統合ダッシュボード。

    各ツールインスタンスの読み取り専用アクセサのみを参照し、
    セクション別に構造化した情報をCLI向けテキストまたはJSON形式で出力する。

    独自の内部状態を保持しない。各ツールインスタンスへの参照（読み取り専用）のみ。

    いずれのインスタンスに対しても、書き込みメソッド
    (record_*, begin_*, end_*等)を呼び出さない。
    """

    def __init__(
        self,
        return_pathway_monitor: Any = None,
        execution_monitor: Any = None,
        pipeline_measurement: Any = None,
        policy_selection_log: Any = None,
        expression_quality: Any = None,
        anomaly_detector: Any = None,
    ) -> None:
        """初期化。

        各ツールインスタンスを引数として受け取る。
        ツールインスタンスを自力で探索・生成しない。

        Args:
            return_pathway_monitor: ReturnPathwayMonitorインスタンス（読み取り専用）
            execution_monitor: ExecutionMonitorインスタンス（読み取り専用）
            pipeline_measurement: PipelineMeasurementインスタンス（読み取り専用）
            policy_selection_log: PolicySelectionLogインスタンス（読み取り専用）
            expression_quality: ExpressionQualityVerificationインスタンス（読み取り専用）
            anomaly_detector: AnomalyDetectorインスタンス（読み取り専用）
        """
        self._return_pathway_monitor = return_pathway_monitor
        self._execution_monitor = execution_monitor
        self._pipeline_measurement = pipeline_measurement
        self._policy_selection_log = policy_selection_log
        self._expression_quality = expression_quality
        self._anomaly_detector = anomaly_detector

    # ── セクション別データ収集 ─────────────────────────────────────

    def _collect_session(self) -> dict[str, Any]:
        """セッション概要を収集する。"""
        if self._execution_monitor is None:
            return {"status": "not_connected"}
        try:
            return {
                "cycle_count": self._execution_monitor.cycle_count,
                "api_call_count": self._execution_monitor.api_call_count,
                "api_token_count": self._execution_monitor.api_token_count,
            }
        except Exception:
            return {"status": "read_error"}

    def _collect_pipeline(self) -> dict[str, Any]:
        """パイプライン計測を収集する。"""
        if self._pipeline_measurement is None:
            return {"status": "not_connected"}
        try:
            summary = self._pipeline_measurement.get_summary()
            # 経路別平均時間を算出
            pathway_counts = summary.get("pathway_counts", {})
            pathway_totals = summary.get("pathway_total_cumulative", {})
            pathway_avg: dict[str, float] = {}
            for pathway, total in pathway_totals.items():
                count = pathway_counts.get(pathway, 0)
                if count > 0:
                    pathway_avg[pathway] = round(total / count, 6)
            return {
                "pathway_counts": pathway_counts,
                "pathway_avg_time": pathway_avg,
                "pathway_phase_cumulative": summary.get(
                    "pathway_phase_cumulative", {}
                ),
                "buffer_size": summary.get("buffer_size", 0),
            }
        except Exception:
            return {"status": "read_error"}

    def _collect_band(self) -> dict[str, Any]:
        """帯域別実行時間を収集する。"""
        if self._execution_monitor is None:
            return {"status": "not_connected"}
        try:
            return {
                "band_cumulative_time": (
                    self._execution_monitor.band_cumulative_time
                ),
            }
        except Exception:
            return {"status": "read_error"}

    def _collect_policy(self) -> dict[str, Any]:
        """方針選択分布を収集する。"""
        if self._policy_selection_log is None:
            return {"status": "not_connected"}
        try:
            agg = self._policy_selection_log.get_aggregation()
            return agg.to_dict()
        except Exception:
            return {"status": "read_error"}

    def _collect_expression(self) -> dict[str, Any]:
        """代弁品質を収集する。"""
        if self._expression_quality is None:
            return {"status": "not_connected"}
        try:
            return self._expression_quality.get_summary()
        except Exception:
            return {"status": "read_error"}

    def _collect_pathway(self) -> dict[str, Any]:
        """帰還経路を収集する。

        セッション内カウンタと永続化用カウンタの両方を収集し、
        「全セッション累計」と「現セッション分」の対比を可能にする。
        """
        if self._return_pathway_monitor is None:
            return {"status": "not_connected"}
        try:
            summary = self._return_pathway_monitor.get_summary()
            # 永続化用データも取得（累計値を含む）
            try:
                persistence_data = self._return_pathway_monitor.get_persistence_data()
                summary["persistence_data"] = persistence_data
            except Exception:
                pass
            return summary
        except Exception:
            return {"status": "read_error"}

    def _collect_enrichment(self) -> dict[str, Any]:
        """enrichment分布を収集する。"""
        if self._execution_monitor is None:
            return {"status": "not_connected"}
        try:
            result: dict[str, Any] = {}

            # enrichment分布サマリ
            try:
                dist_monitor = self._execution_monitor.enrichment_distribution
                dist_summary = dist_monitor.get_distribution_summary()
                if not isinstance(dist_summary, dict):
                    raise TypeError("expected dict")
                latest = dist_summary.get("latest_entry")
                if isinstance(latest, dict):
                    result["total_items"] = int(
                        latest.get("total_items", 0)
                    )
                    result["total_non_empty"] = int(
                        latest.get("total_non_empty", 0)
                    )
                    result["total_changed"] = int(
                        latest.get("total_changed", 0)
                    )
                else:
                    result["total_items"] = 0
                    result["total_non_empty"] = 0
                    result["total_changed"] = 0
                obs = dist_summary.get("observation_count", 0)
                result["observation_count"] = int(obs) if isinstance(
                    obs, (int, float)
                ) else 0
            except Exception:
                result["distribution"] = "read_error"

            # 圧縮前後文字数
            try:
                before, after = (
                    self._execution_monitor.last_compression_chars
                )
                result["compression_before_chars"] = int(before)
                result["compression_after_chars"] = int(after)
            except Exception:
                result["compression"] = "read_error"

            # enrichment有効性サマリ
            try:
                eff_analyzer = (
                    self._execution_monitor.enrichment_effectiveness
                )
                eff_summary = eff_analyzer.compute_summary()
                if isinstance(eff_summary, dict) and eff_summary:
                    ti = eff_summary.get("total_items", 0)
                    tc = eff_summary.get("total_chars_cumulative", 0)
                    result["effectiveness_total_items"] = int(ti) if isinstance(
                        ti, (int, float)
                    ) else 0
                    result["effectiveness_total_chars"] = int(tc) if isinstance(
                        tc, (int, float)
                    ) else 0
            except Exception:
                result["effectiveness"] = "read_error"

            return result
        except Exception:
            return {"status": "read_error"}

    def _collect_anomaly(self) -> dict[str, Any]:
        """動態停止検出を収集する。"""
        if self._anomaly_detector is None:
            return {"status": "not_connected"}
        try:
            return self._anomaly_detector.get_summary()
        except Exception:
            return {"status": "read_error"}

    # ── 全セクション収集 ──────────────────────────────────────────

    _COLLECTORS = {
        SECTION_SESSION: "_collect_session",
        SECTION_PIPELINE: "_collect_pipeline",
        SECTION_BAND: "_collect_band",
        SECTION_POLICY: "_collect_policy",
        SECTION_EXPRESSION: "_collect_expression",
        SECTION_PATHWAY: "_collect_pathway",
        SECTION_ENRICHMENT: "_collect_enrichment",
        SECTION_ANOMALY: "_collect_anomaly",
    }

    def collect(
        self, sections: Optional[list[str]] = None
    ) -> dict[str, dict[str, Any]]:
        """指定セクションの情報を収集する。

        各ツールアクセサの呼び出しは独立しており、
        1つのツールの読み取り失敗が他のツールの表示に影響しない。

        Args:
            sections: 表示するセクション識別子のリスト。
                Noneの場合は全セクションを収集する。

        Returns:
            セクション名→データ辞書のマッピング。
        """
        target_sections = sections if sections else list(ALL_SECTIONS)
        result: dict[str, dict[str, Any]] = {}

        for section_id in target_sections:
            if section_id not in self._COLLECTORS:
                continue
            collector_name = self._COLLECTORS[section_id]
            collector = getattr(self, collector_name, None)
            if collector is None:
                continue
            try:
                result[section_id] = collector()
            except Exception:
                # 安全弁4: 例外安全
                result[section_id] = {"status": "read_error"}

        return result

    # ── テキストフォーマット ──────────────────────────────────────

    def format_text(
        self, sections: Optional[list[str]] = None
    ) -> str:
        """収集した情報をCLI向けプレーンテキストに変換する。

        数値は丸めて読みやすくする。
        動態停止検出で現在停止状態にある信号がある場合は、
        該当セクションに「[停止中]」の事実記述マーカーを付加する。

        安全弁6: 評価的語彙を使用しない。数値と事実ラベルのみ。

        Args:
            sections: 表示するセクション識別子のリスト。

        Returns:
            CLI向けプレーンテキスト。
        """
        data = self.collect(sections)
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("Cyrene Dashboard")
        lines.append("=" * 60)

        # session
        if SECTION_SESSION in data:
            lines.append("")
            lines.append("--- Session ---")
            d = data[SECTION_SESSION]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                lines.append(f"  Cycles: {d.get('cycle_count', 0)}")
                api_calls = d.get("api_call_count", {})
                lines.append(
                    f"  API calls: perception={api_calls.get('perception', 0)}"
                    f", expression={api_calls.get('expression', 0)}"
                )
                tokens = d.get("api_token_count", {})
                for call_type, tok in tokens.items():
                    if isinstance(tok, dict):
                        lines.append(
                            f"  Tokens ({call_type}):"
                            f" in={tok.get('input', 0)},"
                            f" out={tok.get('output', 0)}"
                        )

        # pipeline
        if SECTION_PIPELINE in data:
            lines.append("")
            lines.append("--- Pipeline ---")
            d = data[SECTION_PIPELINE]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                counts = d.get("pathway_counts", {})
                avgs = d.get("pathway_avg_time", {})
                for pathway in sorted(counts.keys()):
                    count = counts[pathway]
                    avg = avgs.get(pathway, 0.0)
                    lines.append(
                        f"  {pathway}: count={count},"
                        f" avg={round(avg * 1000, 2)}ms"
                    )
                lines.append(
                    f"  Buffer: {d.get('buffer_size', 0)}"
                )

        # band
        if SECTION_BAND in data:
            lines.append("")
            lines.append("--- Band Times ---")
            d = data[SECTION_BAND]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                band_times = d.get("band_cumulative_time", {})
                for band_name in sorted(band_times.keys()):
                    t = band_times[band_name]
                    lines.append(
                        f"  {band_name}: {round(t, 4)}s"
                    )

        # policy
        if SECTION_POLICY in data:
            lines.append("")
            lines.append("--- Policy Selection ---")
            d = data[SECTION_POLICY]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                label_counts = d.get("label_selection_counts", {})
                if label_counts:
                    lines.append("  Label counts:")
                    for label in sorted(
                        label_counts.keys(),
                        key=lambda k: label_counts[k],
                        reverse=True,
                    ):
                        lines.append(
                            f"    {label}: {label_counts[label]}"
                        )
                contrib = d.get("section_contribution_totals", {})
                if contrib:
                    lines.append("  Section contributions:")
                    for sec in sorted(contrib.keys()):
                        lines.append(
                            f"    {sec}: {round(contrib[sec], 4)}"
                        )
                lines.append(
                    f"  Window: {d.get('window_size', 0)}"
                )
                lines.append(
                    f"  Max selection reached:"
                    f" {d.get('max_selection_reached_count', 0)}"
                )

        # expression
        if SECTION_EXPRESSION in data:
            lines.append("")
            lines.append("--- Expression Quality ---")
            d = data[SECTION_EXPRESSION]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                lines.append(
                    f"  Records: {d.get('record_count', 0)}"
                )
                lines.append(
                    f"  Fallbacks: {d.get('fallback_count', 0)}"
                )
                lines.append(
                    f"  Buffer: {d.get('buffer_size', 0)}"
                )

        # pathway
        if SECTION_PATHWAY in data:
            lines.append("")
            lines.append("--- Return Pathways ---")
            d = data[SECTION_PATHWAY]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                fire_counts = d.get("pathway_fire_counts", {})
                for pathway_id in sorted(fire_counts.keys()):
                    lines.append(
                        f"  {pathway_id}: {fire_counts[pathway_id]}"
                    )
                lines.append(
                    f"  Concurrent (2+):"
                    f" {d.get('concurrent_2plus_count', 0)}"
                )
                lines.append(
                    f"  Concurrent (3+):"
                    f" {d.get('concurrent_3plus_count', 0)}"
                )
                lines.append(
                    f"  Concurrent (4+):"
                    f" {d.get('concurrent_4plus_count', 0)}"
                )
                lines.append(
                    f"  Concurrent (5):"
                    f" {d.get('concurrent_5_count', 0)}"
                )
                # 合算帯域上限到達カウンタ
                cap_hits = d.get("aggregate_cap_hit_counts", {})
                if cap_hits:
                    lines.append("  Cap hits:")
                    for kind in sorted(cap_hits.keys()):
                        lines.append(
                            f"    {kind}: {cap_hits[kind]}"
                        )

        # enrichment
        if SECTION_ENRICHMENT in data:
            lines.append("")
            lines.append("--- Enrichment ---")
            d = data[SECTION_ENRICHMENT]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                lines.append(
                    f"  Items: {d.get('total_items', 0)}"
                )
                total_items = d.get("total_items", 0)
                non_empty = d.get("total_non_empty", 0)
                if total_items > 0:
                    rate = round(non_empty / total_items * 100, 1)
                    lines.append(f"  Non-empty rate: {rate}%")
                else:
                    lines.append("  Non-empty rate: -")
                lines.append(
                    f"  Changed: {d.get('total_changed', 0)}"
                )
                lines.append(
                    f"  Compression:"
                    f" {d.get('compression_before_chars', 0)}"
                    f" -> {d.get('compression_after_chars', 0)}"
                )

        # anomaly
        if SECTION_ANOMALY in data:
            lines.append("")
            lines.append("--- Dynamics Stall ---")
            d = data[SECTION_ANOMALY]
            if d.get("status") == "not_connected":
                lines.append("  (not connected)")
            elif d.get("status") == "read_error":
                lines.append("  (read error)")
            else:
                lines.append(
                    f"  Snapshots: {d.get('snapshot_count', 0)}"
                )
                lines.append(
                    f"  Buffer: {d.get('buffer_size', 0)}"
                    f"/{d.get('buffer_max', 0)}"
                )
                stall_flags = d.get("current_stall_flags", {})
                detected = d.get("stall_detected_counts", {})
                resolved = d.get("stall_resolved_counts", {})
                for signal in sorted(stall_flags.keys()):
                    is_stalled = stall_flags[signal]
                    det = detected.get(signal, 0)
                    res = resolved.get(signal, 0)
                    marker = " [stalled]" if is_stalled else ""
                    lines.append(
                        f"  {signal}:"
                        f" detected={det},"
                        f" resolved={res}{marker}"
                    )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── JSON出力 ──────────────────────────────────────────────────

    def format_json(
        self, sections: Optional[list[str]] = None
    ) -> str:
        """収集した情報をJSON形式で返す。

        外部ツール（スプレッドシート、スクリプト等）での後処理を可能にする。

        Args:
            sections: 表示するセクション識別子のリスト。

        Returns:
            JSON文字列。
        """
        data = self.collect(sections)
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    # ── CLI出力 ───────────────────────────────────────────────────

    def print_dashboard(
        self,
        sections: Optional[list[str]] = None,
        as_json: bool = False,
        file: Any = None,
    ) -> None:
        """ダッシュボードをCLI標準出力に表示する。

        Args:
            sections: 表示するセクション識別子のリスト。
            as_json: JSON形式で出力するかどうか。
            file: 出力先。Noneの場合はsys.stdout。
        """
        output_file = file if file is not None else sys.stdout
        if as_json:
            text = self.format_json(sections)
        else:
            text = self.format_text(sections)
        print(text, file=output_file)


# ── CLIエントリポイント ───────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    """CLIからの起動エントリポイント。

    引数なしで全セクション表示。
    --json フラグでJSON形式出力。
    セクション名を指定して絞り込み可能。

    実運用時はbrain.pyやorchestratorが保持するツールインスタンスを
    渡す形式とする。CLIからの単体起動時はインスタンスなし（全セクション
    「未接続」表示）となる。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Cyrene integrated dashboard (CLI)"
    )
    parser.add_argument(
        "sections",
        nargs="*",
        default=[],
        help=(
            "Sections to display. If empty, all sections are shown. "
            "Choices: " + ", ".join(ALL_SECTIONS)
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output in JSON format.",
    )

    args = parser.parse_args(argv)

    sections = args.sections if args.sections else None

    # CLIからの単体起動: インスタンスなし
    dashboard = Dashboard()
    dashboard.print_dashboard(sections=sections, as_json=args.as_json)


if __name__ == "__main__":
    main()
