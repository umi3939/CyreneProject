"""
tools/persistence_integrity.py - 永続化整合性検査（外部ツール）

永続化された辞書データの構造的劣化を検出する。
psycheパッケージの外部に配置され、orchestratorの処理フローには一切組み込まれない。

5種類の汎用検証パターンを適用し、検出された事実を文字列リストとして返却する。
修復・補正・正規化は一切行わない。

Usage::

    # Pythonから呼び出す場合
    from tools.persistence_integrity import check_integrity
    result = check_integrity(save_dict)

    # コマンドラインから呼び出す場合
    python -m tools.persistence_integrity data/psyche_snapshot.json
    python -m tools.persistence_integrity data/psyche_snapshot.json --output report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


# ── 検証パターン定義 ─────────────────────────────────────────

# パターン1: 参照先存在確認
# 各要素は (参照元フィールドパス, 識別子抽出方法, 参照先フィールドパス) の三つ組
# 識別子抽出方法:
#   "list_field:<field_name>" - リスト内辞書の指定フィールドから識別子を抽出
#   "list_of_strings" - リスト要素自体が識別子
#   "keys" - 辞書のキーが識別子
REFERENCE_EXISTENCE_PATTERNS: list[dict[str, str]] = [
    {
        "source_path": "last_episodes.links",
        "id_extraction": "list_field:from_episode_id",
        "target_path": "last_episodes.episodes",
        "target_id_field": "episode_id",
        "description": "エピソードリンクの参照元エピソードID",
    },
    {
        "source_path": "last_episodes.links",
        "id_extraction": "list_field:to_episode_id",
        "target_path": "last_episodes.episodes",
        "target_id_field": "episode_id",
        "description": "エピソードリンクの参照先エピソードID",
    },
]

# パターン2: 蓄積構造の構造的制約確認
# 各要素は (フィールドパス, 上限件数) の二つ組
ACCUMULATION_LIMIT_PATTERNS: list[dict[str, Any]] = [
    {
        "field_path": "last_episodes.episodes",
        "limit": 200,
        "description": "エピソード蓄積上限",
    },
    {
        "field_path": "dynamics.intensity_history",
        "limit": 10,
        "description": "感情力学の強度履歴上限",
    },
    {
        "field_path": "loop_state.memory.entries",
        "limit": 10,
        "description": "ループ状態の記憶エントリ上限",
    },
    {
        "field_path": "action_result_state.pairs",
        "limit": 200,
        "description": "行動結果観測ペア蓄積上限",
    },
    {
        "field_path": "action_result_state.composition_buffer",
        "limit": 30,
        "description": "行動結果観測合成バッファ上限",
    },
    {
        "field_path": "action_result_state.recovery_candidates",
        "limit": 50,
        "description": "行動結果観測復帰候補上限",
    },
    {
        "field_path": "action_result_state.decay_history",
        "limit": 100,
        "description": "行動結果観測減衰履歴上限",
    },
    {
        "field_path": "action_result_state.section_description_history",
        "limit": 100,
        "description": "行動結果観測断面記述履歴上限",
    },
    {
        "field_path": "action_result_state.convergence_records",
        "limit": 50,
        "description": "行動結果観測収束記録上限",
    },
    {
        "field_path": "action_result_state.section_weight_history",
        "limit": 100,
        "description": "行動結果観測断面重み履歴上限",
    },
    {
        "field_path": "emotion_cooccurrence_state.records",
        "limit": 50,
        "description": "感情共起記録上限",
    },
    {
        "field_path": "expectation_lifecycle_state.transition_records",
        "limit": 200,
        "description": "期待ライフサイクル遷移記録上限",
    },
    {
        "field_path": "interaction_accumulation_state.pairs",
        "limit": 100,
        "description": "相互作用蓄積ペア上限",
    },
    {
        "field_path": "goal_hierarchy_propagation_state.adjacency_records",
        "limit": 200,
        "description": "目標階層伝搬隣接記録上限",
    },
    {
        "field_path": "goal_hierarchy_propagation_state.convergence_records",
        "limit": 30,
        "description": "目標階層伝搬収束記録上限",
    },
    {
        "field_path": "expectation_lifecycle_state.convergence_records",
        "limit": 30,
        "description": "期待ライフサイクル収束記録上限",
    },
]

# パターン3: 時刻順序の矛盾確認
# 各要素は (先行フィールドパス, 後続フィールドパス) の二つ組
# 先行の時刻値が後続の時刻値より大きい場合に矛盾とする
TIMESTAMP_ORDER_PATTERNS: list[dict[str, str]] = [
    {
        "earlier_path": "dynamics.phase_entered_at",
        "later_path": "loop_state.last_loop_time",
        "description": "力学フェーズ開始時刻とループ最終時刻",
    },
]

# パターン4: 必須フィールド存在確認
# 復元後に空であってはならないフィールドのリスト
REQUIRED_FIELD_PATTERNS: list[dict[str, str]] = [
    {"field_path": "version", "description": "バージョン番号"},
    {"field_path": "tick_count", "description": "ティックカウント"},
    {"field_path": "psyche", "description": "心理状態"},
    {"field_path": "psyche.emotions", "description": "感情ベクトル"},
    {"field_path": "psyche.drives", "description": "駆動ベクトル"},
    {"field_path": "psyche.mood", "description": "気分"},
]

# パターン5: バージョン整合確認
# バージョン番号と、そのバージョンで存在すべきフィールド名の対応
VERSION_FIELD_PATTERNS: dict[int, list[str]] = {
    1: [
        "version", "tick_count", "psyche", "loop_state", "dynamics",
    ],
    5: [
        "tendency_state", "vector_state", "candidate_state",
        "transient_goal_state", "stability_valve",
    ],
    6: [
        "dispersion_state", "context_sensitivity_state", "last_coupling",
    ],
    7: ["policy_expansion_state"],
    8: ["memory_integration_state"],
    9: ["real_feed_state"],
    10: ["text_dialogue_state"],
    11: ["spontaneous_state"],
    12: ["vo_validation_state"],
    13: ["forgetting_fixation_state"],
    14: ["action_result_state"],
    15: ["dialogue_learning_state"],
    16: ["meta_emotion_state"],
    17: ["self_action_perception_state"],
    18: ["expectation_action_diff_log"],
    19: ["intent_action_gap_state"],
    20: ["temporal_cognition_state"],
    21: ["multi_path_recall_state"],
    22: ["introspection_cross_section_state", "perceptual_context_state"],
    23: ["selection_attribution_state"],
    24: ["reference_frequency_state"],
    25: ["persistent_commitment_state"],
    26: ["stabilization_description_state"],
    27: ["behavioral_diversity_state"],
    28: ["spontaneous_recall_state"],
    29: ["internal_contradiction_state"],
    30: ["interaction_accumulation_state"],
    31: ["emotional_backdrop_state"],
    32: ["situational_self_presentation_state"],
    33: ["drive_variation_state"],
    34: ["expectation_lifecycle_state"],
    35: ["input_pathway_balance_state"],
    36: ["responsibility_temporal_trace_state"],
    37: ["emotion_cooccurrence_state"],
    38: ["other_boundary_accumulation_state"],
    39: ["forgetting_recall_balance_state"],
    40: ["attention_distribution_state"],
    41: ["goal_hierarchy_propagation_state"],
    42: ["hypothesis_observation_pairing_state"],
}


# ── ユーティリティ ────────────────────────────────────────────

def _resolve_path(data: dict[str, Any], path: str) -> Any:
    """ドット区切りのパスを辿り、辞書から値を取得する。

    パスの途中でキーが見つからない場合やNoneに到達した場合は _MISSING を返す。
    """
    current: Any = data
    for key in path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return _MISSING
    return current


class _MissingSentinel:
    """パス解決失敗を示すセンチネル値。"""
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def _extract_ids(
    data: Any,
    extraction_method: str,
) -> list[str]:
    """指定された抽出方法に従って識別子リストを取得する。"""
    if isinstance(data, _MissingSentinel) or data is None:
        return []

    if extraction_method.startswith("list_field:"):
        field_name = extraction_method[len("list_field:"):]
        if not isinstance(data, list):
            return []
        ids = []
        for item in data:
            if isinstance(item, dict) and field_name in item:
                val = item[field_name]
                if isinstance(val, str):
                    ids.append(val)
        return ids

    if extraction_method == "list_of_strings":
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, str)]

    if extraction_method == "keys":
        if not isinstance(data, dict):
            return []
        return list(data.keys())

    return []


def _extract_target_ids(
    data: Any,
    id_field: str,
) -> set[str]:
    """参照先のリストからIDの集合を取得する。"""
    if isinstance(data, _MissingSentinel) or data is None:
        return set()

    if not isinstance(data, list):
        return set()

    ids: set[str] = set()
    for item in data:
        if isinstance(item, dict) and id_field in item:
            val = item[id_field]
            if isinstance(val, str):
                ids.add(val)
    return ids


def _is_empty_value(value: Any) -> bool:
    """値が空(None, 空辞書, 空リスト)であるかを判定する。"""
    if value is None:
        return True
    if isinstance(value, _MissingSentinel):
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _is_numeric(value: Any) -> bool:
    """値が数値（int, float）であるかを判定する。"""
    return isinstance(value, (int, float))


# ── 検証パターン適用関数 ──────────────────────────────────────

def _check_reference_existence(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """パターン1: 参照先存在確認を実行する。"""
    findings: list[dict[str, str]] = []

    for pattern in REFERENCE_EXISTENCE_PATTERNS:
        source_data = _resolve_path(data, pattern["source_path"])
        if isinstance(source_data, _MissingSentinel):
            # 参照元フィールド自体が存在しない場合はスキップ
            continue

        target_data = _resolve_path(data, pattern["target_path"])
        target_ids = _extract_target_ids(
            target_data,
            pattern["target_id_field"],
        )

        source_ids = _extract_ids(source_data, pattern["id_extraction"])

        for sid in source_ids:
            if sid not in target_ids:
                findings.append({
                    "pattern": "reference_existence",
                    "field_path": pattern["source_path"],
                    "fact": (
                        f"{pattern['description']}: "
                        f"'{pattern['source_path']}'内の識別子'{sid}'が"
                        f"'{pattern['target_path']}'に存在しない"
                    ),
                })

    return findings


def _check_accumulation_limits(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """パターン2: 蓄積構造の構造的制約確認を実行する。"""
    findings: list[dict[str, str]] = []

    for pattern in ACCUMULATION_LIMIT_PATTERNS:
        field_data = _resolve_path(data, pattern["field_path"])
        if isinstance(field_data, _MissingSentinel):
            continue

        if isinstance(field_data, list):
            count = len(field_data)
            limit = pattern["limit"]
            if count > limit:
                findings.append({
                    "pattern": "accumulation_limit",
                    "field_path": pattern["field_path"],
                    "fact": (
                        f"{pattern['description']}: "
                        f"'{pattern['field_path']}'の件数({count})が"
                        f"上限({limit})を超過している"
                    ),
                })

    return findings


def _check_timestamp_order(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """パターン3: 時刻順序の矛盾確認を実行する。"""
    findings: list[dict[str, str]] = []

    for pattern in TIMESTAMP_ORDER_PATTERNS:
        earlier_val = _resolve_path(data, pattern["earlier_path"])
        later_val = _resolve_path(data, pattern["later_path"])

        if isinstance(earlier_val, _MissingSentinel):
            continue
        if isinstance(later_val, _MissingSentinel):
            continue

        if not _is_numeric(earlier_val) or not _is_numeric(later_val):
            # 数値でない場合は文字列として比較を試みる
            if isinstance(earlier_val, str) and isinstance(later_val, str):
                if earlier_val > later_val:
                    findings.append({
                        "pattern": "timestamp_order",
                        "field_path": f"{pattern['earlier_path']} -> {pattern['later_path']}",
                        "fact": (
                            f"{pattern['description']}: "
                            f"先行時刻'{pattern['earlier_path']}'({earlier_val})が"
                            f"後続時刻'{pattern['later_path']}'({later_val})より新しい"
                        ),
                    })
            continue

        if earlier_val > later_val:
            findings.append({
                "pattern": "timestamp_order",
                "field_path": f"{pattern['earlier_path']} -> {pattern['later_path']}",
                "fact": (
                    f"{pattern['description']}: "
                    f"先行時刻'{pattern['earlier_path']}'({earlier_val})が"
                    f"後続時刻'{pattern['later_path']}'({later_val})より新しい"
                ),
            })

    return findings


def _check_required_fields(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """パターン4: 必須フィールド存在確認を実行する。"""
    findings: list[dict[str, str]] = []

    for pattern in REQUIRED_FIELD_PATTERNS:
        value = _resolve_path(data, pattern["field_path"])
        if _is_empty_value(value):
            findings.append({
                "pattern": "required_field",
                "field_path": pattern["field_path"],
                "fact": (
                    f"{pattern['description']}: "
                    f"必須フィールド'{pattern['field_path']}'が"
                    f"空または存在しない"
                ),
            })

    return findings


def _check_version_fields(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """パターン5: バージョン整合確認を実行する。"""
    findings: list[dict[str, str]] = []

    version = data.get("version")
    if version is None or not isinstance(version, (int, float)):
        findings.append({
            "pattern": "version_consistency",
            "field_path": "version",
            "fact": "バージョン番号が存在しないか数値でない",
        })
        return findings

    version_int = int(version)

    for ver, expected_fields in sorted(VERSION_FIELD_PATTERNS.items()):
        if version_int >= ver:
            for field_name in expected_fields:
                if field_name not in data:
                    findings.append({
                        "pattern": "version_consistency",
                        "field_path": field_name,
                        "fact": (
                            f"バージョン{version_int}ではフィールド'{field_name}'"
                            f"が存在すべきだが見つからない"
                            f"（バージョン{ver}で追加）"
                        ),
                    })

    return findings


# ── メイン検証関数 ────────────────────────────────────────────

def check_integrity(save_dict: dict[str, Any]) -> dict[str, Any]:
    """永続化辞書の構造的整合性を検証する。

    入力辞書の複製を操作し、元の辞書は変更しない。
    状態を保持しない。毎回辞書を受け取り結果を返却して終了する。
    修復・補正・正規化は一切行わない。

    Args:
        save_dict: orchestratorのsave()が出力する辞書構造。
                   この辞書は複製されるため元データは変更されない。

    Returns:
        検証結果を含む辞書。以下の構造を持つ:
        - basic_info: 基本情報（バージョン、フィールド総数、パターン適用数）
        - findings: パターン別検出結果リスト
        - summary: パターン種別ごとの検出件数
    """
    # 入力辞書の複製を操作し、元のsave辞書を変更しない
    data = copy.deepcopy(save_dict)

    # 基本情報の収集
    version = data.get("version", None)
    top_level_field_count = len(data)

    # 各検証パターンを順に適用
    all_findings: list[dict[str, str]] = []

    ref_findings = _check_reference_existence(data)
    all_findings.extend(ref_findings)

    acc_findings = _check_accumulation_limits(data)
    all_findings.extend(acc_findings)

    ts_findings = _check_timestamp_order(data)
    all_findings.extend(ts_findings)

    req_findings = _check_required_fields(data)
    all_findings.extend(req_findings)

    ver_findings = _check_version_fields(data)
    all_findings.extend(ver_findings)

    # パターン適用数の集計
    pattern_count = (
        len(REFERENCE_EXISTENCE_PATTERNS)
        + len(ACCUMULATION_LIMIT_PATTERNS)
        + len(TIMESTAMP_ORDER_PATTERNS)
        + len(REQUIRED_FIELD_PATTERNS)
        + 1  # バージョン整合確認は1パターン
    )

    # パターン種別ごとの検出件数集計
    summary: dict[str, int] = {
        "reference_existence": len(ref_findings),
        "accumulation_limit": len(acc_findings),
        "timestamp_order": len(ts_findings),
        "required_field": len(req_findings),
        "version_consistency": len(ver_findings),
    }

    return {
        "basic_info": {
            "version": version,
            "top_level_field_count": top_level_field_count,
            "pattern_count": pattern_count,
        },
        "findings": all_findings,
        "summary": summary,
        "total_findings": len(all_findings),
    }


# ── コマンドライン呼び出し ────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """コマンドラインから直接呼び出す場合のエントリポイント。

    永続化ファイルのパスを引数に受け取り、検証結果を出力する。
    """
    parser = argparse.ArgumentParser(
        description="永続化整合性検査: 永続化辞書の構造的劣化を検出する",
    )
    parser.add_argument(
        "snapshot_path",
        type=str,
        help="永続化ファイル（psyche_snapshot.json等）のパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="検証結果の出力先ファイルパス（省略時は標準出力）",
    )

    args = parser.parse_args(argv)

    snapshot_path = Path(args.snapshot_path)
    if not snapshot_path.exists():
        print(f"Error: File not found: {snapshot_path}", file=sys.stderr)
        return 1

    try:
        save_dict = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(save_dict, dict):
        print("Error: Top-level structure is not a dictionary", file=sys.stderr)
        return 1

    result = check_integrity(save_dict)

    output_text = json.dumps(result, ensure_ascii=False, indent=2)

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
