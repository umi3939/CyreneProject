"""
tools/phase_dependency_validator.py - Phase間データ依存関係の宣言的可視化

Phase宣言定義(phase_declaration.py)を読み取り専用で参照し、
依存関係の構造的検証と要約を生成する純粋関数ツール。

psycheパッケージの外部に配置され、orchestratorの処理フローには一切組み込まれない。
Phase宣言定義の変更は行わない。save/loadの対象外。enrichmentに接続しない。
統合管理構造の実行時に呼び出される経路は存在しない。

Usage::

    # Pythonから呼び出す場合
    from tools.phase_dependency_validator import validate_dependencies
    result = validate_dependencies()

    # コマンドラインから呼び出す場合
    python -m tools.phase_dependency_validator
    python -m tools.phase_dependency_validator --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psyche.phase_declaration import (
    ALL_PHASES,
    ALL_BANDS,
    Band,
    BandDefinition,
    PhaseDefinition,
    PHASE_BY_ID,
    compute_data_dependencies,
    DataDependency,
)


# ── 外部入力の宣言的列挙 ────────────────────────────────────────
# 統合管理構造が外部イベント（知覚入力、テキスト入力、自発起動）により
# 設定する中間状態変数。「未生産読み取り」の検出から除外するために使用する。
# 不変の定数として保持し、実行時に変化しない。

EXTERNAL_INPUT_VARIABLES: frozenset[str] = frozenset({
    # 知覚入力により設定される変数
    "_last_percept",
    # 自発起動により設定される変数
    "_last_activation_result",
    # 初期化時に設定されるモジュールインスタンス参照
    "_psyche",
    "_loop_state",
    "_text_dialogue_processor",
    "_spontaneous_processor",
    "_last_recalled_memories",
    "_responsibility_mgr",
    "_dispersion_state",
    "_action_result_observer",
    "_dialogue_learning_processor",
    "_forgetting_fixation_processor",
    "_multi_path_recall",
    "_spontaneous_recall",
    "_real_feed_processor",
    "_self_action_recorder",
    "_last_other_model",
    "_input_supply",
    "_meta_emotion_processor",
    "_introspection_cross_section",
    "_temporal_cognition",
    # Cycle 9-10: 新規追加モジュールインスタンス参照
    "_memory_emotion_return",
    "_other_hypothesis_emotion_return",
    "_other_model_sys",
    "_behavioral_diversity_state",
    "_contradiction_processor",
    "_return_pathway_monitor",
    "_selection_attribution_recorder",
    # 選択結果として設定される変数
    "_last_selected_policy_label",
    "_last_selected_policy_axis",
})


# ── 検証結果レコード ────────────────────────────────────────────


@dataclass(frozen=True)
class UnreferencedWrite:
    """未参照書き込み: 書き込まれるが他のどのPhaseからも読み取られない変数。"""
    phase_id: str
    variable: str


@dataclass(frozen=True)
class UnproducedRead:
    """未生産読み取り: 読み取られるがどのPhaseからも書き込まれない変数。
    外部入力として既知のものは除外済み。"""
    phase_id: str
    variable: str


@dataclass(frozen=True)
class IntraBandOrderViolation:
    """同一帯域内順序不整合: 読み取り側が書き込み側より先に実行される。"""
    band: str
    reader_phase_id: str
    writer_phase_id: str
    variable: str
    reader_order: int
    writer_order: int


@dataclass(frozen=True)
class CrossBandImplicitAssumption:
    """帯域間暗黙的前提: 異なる帯域間の依存。"""
    reader_band: str
    writer_band: str
    reader_phase_id: str
    writer_phase_id: str
    variable: str


@dataclass(frozen=True)
class ValidationResult:
    """第一層: 検証結果の集約レコード。"""
    unreferenced_writes: tuple[UnreferencedWrite, ...]
    unproduced_reads: tuple[UnproducedRead, ...]
    intra_band_order_violations: tuple[IntraBandOrderViolation, ...]
    cross_band_implicit_assumptions: tuple[CrossBandImplicitAssumption, ...]


@dataclass(frozen=True)
class BandDependencyEdge:
    """帯域内依存マップの辺。"""
    producer_phase_id: str
    consumer_phase_id: str
    variable: str


@dataclass(frozen=True)
class CrossBandDependencyEntry:
    """帯域間依存マップのエントリ。"""
    producer_band: str
    consumer_band: str
    producer_phase_id: str
    consumer_phase_id: str
    variable: str


@dataclass(frozen=True)
class OverallStatistics:
    """全体統計。"""
    total_phases: int
    total_variables: int
    total_dependency_edges: int
    intra_band_edges: int
    cross_band_edges: int
    unreferenced_writes_count: int
    unproduced_reads_count: int
    order_violations_count: int


@dataclass(frozen=True)
class StructuralSummary:
    """第二層: 構造的要約の集約レコード。"""
    intra_band_maps: dict[str, list[dict[str, str]]]
    cross_band_entries: list[dict[str, str]]
    statistics: OverallStatistics


@dataclass(frozen=True)
class DependencyReport:
    """検証・要約の統合レポート。"""
    validation: ValidationResult
    summary: StructuralSummary


# ── 第一層: 既存依存宣言の監査 ──────────────────────────────────


def _collect_all_written_variables() -> dict[str, list[str]]:
    """全Phaseの書き込み変数を収集する。

    Returns:
        変数名 -> 書き込むPhase IDリスト
    """
    writers: dict[str, list[str]] = {}
    for p in ALL_PHASES:
        for w in p.writes:
            writers.setdefault(w, []).append(p.phase_id)
    return writers


def _collect_all_read_variables() -> dict[str, list[str]]:
    """全Phaseの読み取り変数を収集する。

    Returns:
        変数名 -> 読み取るPhase IDリスト
    """
    readers: dict[str, list[str]] = {}
    for p in ALL_PHASES:
        for r in p.reads:
            readers.setdefault(r, []).append(p.phase_id)
    return readers


def _detect_unreferenced_writes(
    writers: dict[str, list[str]],
    readers: dict[str, list[str]],
) -> tuple[UnreferencedWrite, ...]:
    """(a) 書き込み先の未参照検出。

    ある処理単位が書き込む中間状態変数が、他のどの処理単位からも
    読み取られていない場合、「未参照」として報告する。
    """
    results: list[UnreferencedWrite] = []
    for variable, writer_ids in sorted(writers.items()):
        if variable not in readers:
            for pid in sorted(writer_ids):
                results.append(UnreferencedWrite(
                    phase_id=pid,
                    variable=variable,
                ))
    return tuple(results)


def _detect_unproduced_reads(
    writers: dict[str, list[str]],
    readers: dict[str, list[str]],
) -> tuple[UnproducedRead, ...]:
    """(b) 読み取り元の未生産検出。

    ある処理単位が読み取る中間状態変数が、他のどの処理単位からも
    書き込まれていない場合、外部入力として既知のものを除外した残りを
    「宣言漏れ候補」として報告する。
    """
    results: list[UnproducedRead] = []
    for variable, reader_ids in sorted(readers.items()):
        if variable not in writers and variable not in EXTERNAL_INPUT_VARIABLES:
            for pid in sorted(reader_ids):
                results.append(UnproducedRead(
                    phase_id=pid,
                    variable=variable,
                ))
    return tuple(results)


def _detect_intra_band_order_violations() -> tuple[IntraBandOrderViolation, ...]:
    """(c) 同一帯域内順序整合性検証。

    同一帯域内で、処理単位Aが書き込む中間状態変数を処理単位Bが読み取る場合、
    Aのband_orderがBより小さくなければならない。
    この条件が成立しない場合を「同一帯域内順序不整合」として報告する。
    """
    results: list[IntraBandOrderViolation] = []

    # 帯域ごとにグループ化
    band_phases: dict[Band, list[PhaseDefinition]] = {}
    for p in ALL_PHASES:
        band_phases.setdefault(p.band, []).append(p)

    for band, phases in sorted(band_phases.items(), key=lambda x: x[0].value):
        # 帯域内の書き込み変数索引
        band_writers: dict[str, list[PhaseDefinition]] = {}
        for p in phases:
            for w in p.writes:
                band_writers.setdefault(w, []).append(p)

        # 帯域内の読み取りチェック
        for reader in phases:
            for r in reader.reads:
                if r in band_writers:
                    for writer in band_writers[r]:
                        if writer.phase_id == reader.phase_id:
                            continue  # 自己参照はスキップ
                        # 書き込み側のband_orderが読み取り側以上 = 不整合
                        if writer.band_order >= reader.band_order:
                            results.append(IntraBandOrderViolation(
                                band=band.value,
                                reader_phase_id=reader.phase_id,
                                writer_phase_id=writer.phase_id,
                                variable=r,
                                reader_order=reader.band_order,
                                writer_order=writer.band_order,
                            ))

    return tuple(results)


def _enumerate_cross_band_implicit_assumptions() -> tuple[CrossBandImplicitAssumption, ...]:
    """(d) 帯域間暗黙的前提の一覧化。

    異なる帯域間の依存を明示的に列挙する。
    """
    deps = compute_data_dependencies()
    results: list[CrossBandImplicitAssumption] = []
    for d in deps:
        if not d.same_band:
            reader = PHASE_BY_ID[d.consumer_phase_id]
            writer = PHASE_BY_ID[d.producer_phase_id]
            results.append(CrossBandImplicitAssumption(
                reader_band=reader.band.value,
                writer_band=writer.band.value,
                reader_phase_id=d.consumer_phase_id,
                writer_phase_id=d.producer_phase_id,
                variable=d.intermediate_state,
            ))
    return tuple(results)


def _run_validation() -> ValidationResult:
    """第一層の全検査を実行する。"""
    writers = _collect_all_written_variables()
    readers = _collect_all_read_variables()

    return ValidationResult(
        unreferenced_writes=_detect_unreferenced_writes(writers, readers),
        unproduced_reads=_detect_unproduced_reads(writers, readers),
        intra_band_order_violations=_detect_intra_band_order_violations(),
        cross_band_implicit_assumptions=_enumerate_cross_band_implicit_assumptions(),
    )


# ── 第二層: 依存関係の構造的要約 ────────────────────────────────


def _build_intra_band_maps(
    deps: tuple[DataDependency, ...],
) -> dict[str, list[dict[str, str]]]:
    """(e) 帯域別依存マップ。

    各帯域内の処理単位間の依存連鎖を、中間状態変数名をラベルとする
    隣接リスト形式で表現する。
    """
    result: dict[str, list[dict[str, str]]] = {}
    for band in Band:
        result[band.value] = []

    for d in deps:
        if d.same_band:
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            result[consumer.band.value].append({
                "producer": d.producer_phase_id,
                "consumer": d.consumer_phase_id,
                "variable": d.intermediate_state,
            })

    return result


def _build_cross_band_entries(
    deps: tuple[DataDependency, ...],
) -> list[dict[str, str]]:
    """(f) 帯域間依存マップ。

    帯域をまたぐ依存を列挙する。
    """
    entries: list[dict[str, str]] = []
    for d in deps:
        if not d.same_band:
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            producer = PHASE_BY_ID[d.producer_phase_id]
            entries.append({
                "producer_band": producer.band.value,
                "consumer_band": consumer.band.value,
                "producer": d.producer_phase_id,
                "consumer": d.consumer_phase_id,
                "variable": d.intermediate_state,
            })
    return entries


def _compute_statistics(
    validation: ValidationResult,
    deps: tuple[DataDependency, ...],
) -> OverallStatistics:
    """(g) 全体統計。"""
    # 中間状態変数の総数
    all_variables: set[str] = set()
    for p in ALL_PHASES:
        all_variables.update(p.reads)
        all_variables.update(p.writes)

    intra_count = sum(1 for d in deps if d.same_band)
    cross_count = sum(1 for d in deps if not d.same_band)

    return OverallStatistics(
        total_phases=len(ALL_PHASES),
        total_variables=len(all_variables),
        total_dependency_edges=len(deps),
        intra_band_edges=intra_count,
        cross_band_edges=cross_count,
        unreferenced_writes_count=len(validation.unreferenced_writes),
        unproduced_reads_count=len(validation.unproduced_reads),
        order_violations_count=len(validation.intra_band_order_violations),
    )


def _build_summary(
    validation: ValidationResult,
    deps: tuple[DataDependency, ...],
) -> StructuralSummary:
    """第二層の構造的要約を生成する。"""
    return StructuralSummary(
        intra_band_maps=_build_intra_band_maps(deps),
        cross_band_entries=_build_cross_band_entries(deps),
        statistics=_compute_statistics(validation, deps),
    )


# ── 公開API ────────────────────────────────────────────────────


def validate_dependencies() -> DependencyReport:
    """Phase間データ依存関係の検証と構造的要約を生成する。

    Phase宣言定義を入力として、検証結果と構造的要約を返す純粋関数。
    内部状態を持たない。実行のたびに同一入力に対して同一出力を返す。

    Returns:
        DependencyReport: 検証結果と構造的要約を含むレポート
    """
    validation = _run_validation()
    deps = compute_data_dependencies()
    summary = _build_summary(validation, deps)
    return DependencyReport(validation=validation, summary=summary)


def report_to_dict(report: DependencyReport) -> dict[str, Any]:
    """DependencyReportをJSON直列化可能な辞書に変換する。

    Args:
        report: DependencyReportインスタンス

    Returns:
        JSON直列化可能な辞書
    """
    v = report.validation
    s = report.summary

    return {
        "validation": {
            "unreferenced_writes": [
                {"phase_id": uw.phase_id, "variable": uw.variable}
                for uw in v.unreferenced_writes
            ],
            "unproduced_reads": [
                {"phase_id": ur.phase_id, "variable": ur.variable}
                for ur in v.unproduced_reads
            ],
            "intra_band_order_violations": [
                {
                    "band": iv.band,
                    "reader_phase_id": iv.reader_phase_id,
                    "writer_phase_id": iv.writer_phase_id,
                    "variable": iv.variable,
                    "reader_order": iv.reader_order,
                    "writer_order": iv.writer_order,
                }
                for iv in v.intra_band_order_violations
            ],
            "cross_band_implicit_assumptions": [
                {
                    "reader_band": ca.reader_band,
                    "writer_band": ca.writer_band,
                    "reader_phase_id": ca.reader_phase_id,
                    "writer_phase_id": ca.writer_phase_id,
                    "variable": ca.variable,
                }
                for ca in v.cross_band_implicit_assumptions
            ],
        },
        "summary": {
            "intra_band_maps": s.intra_band_maps,
            "cross_band_entries": s.cross_band_entries,
            "statistics": {
                "total_phases": s.statistics.total_phases,
                "total_variables": s.statistics.total_variables,
                "total_dependency_edges": s.statistics.total_dependency_edges,
                "intra_band_edges": s.statistics.intra_band_edges,
                "cross_band_edges": s.statistics.cross_band_edges,
                "unreferenced_writes_count": s.statistics.unreferenced_writes_count,
                "unproduced_reads_count": s.statistics.unproduced_reads_count,
                "order_violations_count": s.statistics.order_violations_count,
            },
        },
    }


# ── コマンドライン呼び出し ────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """コマンドラインから直接呼び出す場合のエントリポイント。

    Phase宣言定義を読み取り、依存関係の検証結果と構造的要約を出力する。
    """
    parser = argparse.ArgumentParser(
        description="Phase間データ依存関係の宣言的可視化: "
                    "Phase宣言定義の依存関係を検証し、構造的要約を生成する",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="検証結果の出力先ファイルパス（省略時は標準出力）",
    )

    args = parser.parse_args(argv)

    report = validate_dependencies()
    result_dict = report_to_dict(report)
    output_text = json.dumps(result_dict, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Results written to {output_path}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
