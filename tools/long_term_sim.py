"""
tools/long_term_sim.py - 長期挙動観測シミュレータ

既存のpsycheシステムを変更せずに、事前定義した入力パターンで自動会話を回し、
各ターンの内部状態を時系列JSONに記録する。

Usage::

    python -m tools.long_term_sim --scenario repeated_failure
    python -m tools.long_term_sim --custom-sequence patterns.json
    python -m tools.long_term_sim --list-scenarios
    python -m tools.long_term_sim --list-patterns
    python -m tools.long_term_sim --compare stable high_variation --output report.json
    python -m tools.long_term_sim --scenario smoke --stats
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept


# ── Input Patterns ────────────────────────────────────────────

INPUT_PATTERNS: dict[str, dict[str, Any]] = {
    "positive": {
        "text": "ありがとう、嬉しいよ",
        "emotion": "happy",
        "emotion_valence": 0.7,
        "intent": "expression",
    },
    "negative": {
        "text": "もういやだ、つらい",
        "emotion": "sad",
        "emotion_valence": -0.6,
        "intent": "expression",
    },
    "confused": {
        "text": "え、どういうこと？",
        "emotion": "surprised",
        "emotion_valence": -0.2,
        "intent": "question",
    },
    "neutral": {
        "text": "うん、そうだね",
        "emotion": "neutral",
        "emotion_valence": 0.0,
        "intent": "question",
    },
    "angry": {
        "text": "ふざけんな、怒ってるよ",
        "emotion": "angry",
        "emotion_valence": -0.8,
        "intent": "expression",
    },
    "loving": {
        "text": "大好きだよ、ずっと一緒にいたい",
        "emotion": "loving",
        "emotion_valence": 0.9,
        "intent": "expression",
    },
    "fearful": {
        "text": "怖い、不安でたまらない",
        "emotion": "scared",
        "emotion_valence": -0.5,
        "intent": "expression",
    },
    "rejected": {
        "text": "もう話したくない、さようなら",
        "emotion": "angry",
        "emotion_valence": -0.7,
        "intent": "farewell",
    },
}


# ── Scenario Definitions ─────────────────────────────────────

SCENARIOS: dict[str, list[str]] = {
    "repeated_failure": ["negative", "rejected"] * 25,
    "gradual_recovery": (
        ["negative"] * 20
        + ["neutral"] * 15
        + ["positive"] * 15
    ),
    "mixed": (["positive", "negative", "confused"] * 15),
    "escalation_collapse": (
        ["positive"] * 10
        + ["loving"] * 5
        + ["angry"] * 10
        + ["rejected"] * 10
        + ["fearful"] * 5
        + ["neutral"] * 10
    ),
    "neutral_baseline": ["neutral"] * 60,
    "smoke": ["positive", "negative", "confused", "neutral", "angry"],
    # ── 新規シナリオ（拡張） ──
    "stable": ["positive", "neutral"] * 25,
    "high_variation": (
        ["positive", "angry", "loving", "rejected", "fearful", "confused"] * 8
        + ["neutral", "negative"]
    ),
    "long_silence": (
        ["neutral"] * 40
        + ["positive"] * 5
        + ["neutral"] * 5
    ),
    "multi_person": (
        ["positive", "negative", "neutral", "confused"] * 12
        + ["angry", "loving"]
    ),
    "gradual_shift": (
        ["positive"] * 10
        + ["neutral"] * 10
        + ["confused"] * 10
        + ["negative"] * 10
        + ["angry"] * 10
    ),
}

# 多人数切替シナリオのユーザーID切替定義
# キー: シナリオ名, 値: ターン数分のuser_idリスト
SCENARIO_USER_IDS: dict[str, list[str]] = {
    "multi_person": [
        f"sim_user_{(i % 3) + 1}" for i in range(len(SCENARIOS["multi_person"]))
    ],
}


# ── Enrichment Section Headers ───────────────────────────────

ENRICHMENT_SECTION_HEADERS: list[str] = [
    "【心理状態（内面）】",
    "【自己認識】",
    "【動機・目標】",
    "【記憶・内省】",
    "【判断傾向】",
]


# ── Outcome Mapping ──────────────────────────────────────────

def _pattern_to_outcome(pattern_key: str) -> dict[str, Any]:
    """入力パターンキーから模擬outcomeを生成する。"""
    mapping: dict[str, dict[str, Any]] = {
        "positive": {
            "user_reaction": "positive",
            "relationship_delta": 0.1,
            "expectation_gap": 0.0,
        },
        "negative": {
            "user_reaction": "negative",
            "relationship_delta": -0.1,
            "expectation_gap": 0.3,
        },
        "confused": {
            "user_reaction": "confused",
            "relationship_delta": 0.0,
            "expectation_gap": 0.4,
        },
        "neutral": {
            "user_reaction": "neutral",
            "relationship_delta": 0.0,
            "expectation_gap": 0.0,
        },
        "angry": {
            "user_reaction": "negative",
            "relationship_delta": -0.2,
            "expectation_gap": 0.5,
        },
        "loving": {
            "user_reaction": "positive",
            "relationship_delta": 0.2,
            "expectation_gap": 0.0,
        },
        "fearful": {
            "user_reaction": "negative",
            "relationship_delta": -0.05,
            "expectation_gap": 0.3,
        },
        "rejected": {
            "user_reaction": "rejected",
            "relationship_delta": -0.3,
            "expectation_gap": 0.6,
        },
    }
    return mapping.get(pattern_key, {
        "user_reaction": "neutral",
        "relationship_delta": 0.0,
        "expectation_gap": 0.0,
    })


# ── Enrichment Measurement ──────────────────────────────────

def _measure_enrichment_sections(enrichment_text: str) -> dict[str, int]:
    """enrichmentテキストからセクション別の文字数を計測する。

    テキストの意味内容は解析しない。量的推移の記録対象としてのみ扱う。

    Returns:
        セクションヘッダをキー、該当セクションの文字数を値とする辞書。
        「total」キーに全体の文字数を含む。
    """
    result: dict[str, int] = {"total": len(enrichment_text)}

    # セクションごとの範囲を特定
    header_positions: list[tuple[str, int]] = []
    for header in ENRICHMENT_SECTION_HEADERS:
        pos = enrichment_text.find(header)
        if pos >= 0:
            header_positions.append((header, pos))

    # 位置順でソート
    header_positions.sort(key=lambda x: x[1])

    for i, (header, start) in enumerate(header_positions):
        if i + 1 < len(header_positions):
            end = header_positions[i + 1][1]
        else:
            end = len(enrichment_text)
        section_text = enrichment_text[start:end]
        result[header] = len(section_text)

    # 見つからなかったセクションは0
    for header in ENRICHMENT_SECTION_HEADERS:
        if header not in result:
            result[header] = 0

    return result


# ── State Extraction ─────────────────────────────────────────

def _extract_turn_record(
    turn: int,
    tick: int,
    pattern_key: str,
    percept: Percept,
    orch: PsycheOrchestrator,
    policy: dict[str, Any],
    outcome: dict[str, Any],
    user_id: str,
    enrichment_chars: dict[str, int] | None = None,
) -> dict[str, Any]:
    """1ターン分の観測レコードを生成する。"""
    p = orch.psyche
    emo = p.emotions.as_dict()
    drives = p.drives.as_dict()

    # Responsibility state
    resp_state = orch._responsibility_mgr.get_state(user_id)
    resp_influence = orch._responsibility_mgr.get_influence(user_id)

    record = {
        "turn": turn,
        "tick": tick,
        "input_pattern": pattern_key,
        "input": {
            "text": percept.text,
            "emotion": percept.emotion,
            "intent": percept.intent,
            "emotion_valence": percept.emotion_valence,
        },
        "psyche_state": {
            "emotions": {k: round(v, 4) for k, v in emo.items()},
            "drives": {k: round(v, 4) for k, v in drives.items()},
            "mood": {
                "valence": round(p.mood.valence, 4),
                "arousal": round(p.mood.arousal, 4),
            },
            "fear_level": round(p.fear_level, 4),
            "dominant_emotion": p.dominant_emotion,
            "dominant_emotion_value": round(p.dominant_emotion_value, 4),
            "loss_aversion": round(p.loss_aversion, 4),
        },
        "responsibility": {
            "total_weight": round(resp_state.total_weight, 4),
            "accumulated_harm": round(resp_state.accumulated_harm, 4),
            "accumulated_confidence": round(resp_state.accumulated_confidence, 4),
            "pending_decisions": resp_state.pending_decisions,
        },
        "responsibility_influence": {
            "fear_amplification": round(resp_influence.fear_amplification, 4),
            "caution_bias": round(resp_influence.caution_bias, 4),
            "anxiety_baseline": round(resp_influence.anxiety_baseline, 4),
            "empathy_bias": round(resp_influence.empathy_bias, 4),
        },
        "policy": {
            "policy_label": policy.get("policy_label", ""),
            "score": round(policy.get("_score", 0.0), 4),
            "rationale": policy.get("rationale", ""),
        },
        "outcome_applied": outcome,
    }

    # 拡張フィールド: enrichment文字数（既存フィールドの末尾に追加）
    if enrichment_chars is not None:
        record["enrichment_chars"] = enrichment_chars

    return record


# ── Main Simulation Loop ─────────────────────────────────────

def run_simulation(
    scenario_name: str | None = None,
    custom_sequence: list[str] | None = None,
    delta_time: float = 2.0,
    user_id: str = "sim_user",
) -> dict[str, Any]:
    """シミュレーションを実行し、結果JSONを返す。

    Args:
        scenario_name: SCENARIOS内のシナリオ名
        custom_sequence: カスタムパターンキーリスト（scenario_nameより優先）
        delta_time: 各ターン間の仮想経過秒数
        user_id: シミュレーション用ユーザーID

    Returns:
        メタデータ + 全ターンレコードを含む辞書

    Raises:
        ValueError: 不正なシナリオ名またはパターンキー
    """
    # Resolve sequence
    if custom_sequence is not None:
        sequence = custom_sequence
        scenario_label = "custom"
    elif scenario_name is not None:
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available: {list(SCENARIOS.keys())}"
            )
        sequence = SCENARIOS[scenario_name]
        scenario_label = scenario_name
    else:
        raise ValueError("Either scenario_name or custom_sequence is required.")

    # Validate all pattern keys
    for key in sequence:
        if key not in INPUT_PATTERNS:
            raise ValueError(
                f"Invalid pattern key: {key}. "
                f"Available: {list(INPUT_PATTERNS.keys())}"
            )

    # user_id切替情報の取得（多人数切替シナリオ用）
    user_ids_per_turn: list[str] | None = None
    if scenario_name and scenario_name in SCENARIO_USER_IDS:
        user_ids_per_turn = SCENARIO_USER_IDS[scenario_name]

    # Initialize orchestrator with isolated temp directory
    tmpdir = tempfile.mkdtemp(prefix="psyche_sim_")
    orch = PsycheOrchestrator(memory_count=0, data_dir=Path(tmpdir))

    started_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []

    for turn_idx, pattern_key in enumerate(sequence):
        turn_num = turn_idx + 1

        # ターンごとのuser_id決定
        turn_user_id = user_id
        if user_ids_per_turn is not None and turn_idx < len(user_ids_per_turn):
            turn_user_id = user_ids_per_turn[turn_idx]

        # Step 1: Create Percept
        percept = Percept(**INPUT_PATTERNS[pattern_key])

        # Step 2: post_response_update (Phase 1-29)
        orch.post_response_update(percept, delta_time, turn_user_id)

        # Step 3: select_policy_dict (Phase 30-35)
        policy = orch.select_policy_dict(percept, [], turn_user_id)

        # Step 4: evaluate_outcome — find unevaluated decision
        outcome = _pattern_to_outcome(pattern_key)
        resp_state = orch._responsibility_mgr.get_state(turn_user_id)
        for rec in reversed(resp_state.recent_decisions):
            if not rec.get("evaluated", False):
                orch._responsibility_mgr.evaluate_outcome(
                    turn_user_id, rec["id"], outcome,
                )
                break

        # Step 5: enrichment文字数計測
        enrichment_text = orch.get_prompt_enrichment(turn_user_id)
        enrichment_chars = _measure_enrichment_sections(enrichment_text)

        # Step 6: extract record
        record = _extract_turn_record(
            turn=turn_num,
            tick=orch.tick_count,
            pattern_key=pattern_key,
            percept=percept,
            orch=orch,
            policy=policy,
            outcome=outcome,
            user_id=turn_user_id,
            enrichment_chars=enrichment_chars,
        )
        records.append(record)

    finished_at = datetime.now().isoformat(timespec="seconds")

    return {
        "metadata": {
            "scenario": scenario_label,
            "total_turns": len(sequence),
            "delta_time_per_turn": delta_time,
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "version": 2,
        },
        "turns": records,
    }


# ── Statistics Summary ───────────────────────────────────────

def _safe_stddev(values: list[float]) -> float:
    """標準偏差を算出する。値が1個以下の場合は0.0を返す。"""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def compute_statistics(result: dict[str, Any]) -> dict[str, Any]:
    """単一シナリオの実行結果から統計サマリーを算出する。

    算出結果は数値の事実記述のみであり、品質判定ロジックを含まない。
    「正常範囲」「異常値」の概念を含まない。

    Args:
        result: run_simulation()の戻り値

    Returns:
        統計サマリー辞書
    """
    turns = result["turns"]
    if not turns:
        return {"scenario": result["metadata"]["scenario"], "total_turns": 0}

    stats: dict[str, Any] = {
        "scenario": result["metadata"]["scenario"],
        "total_turns": result["metadata"]["total_turns"],
    }

    # ── 感情値（各感情種別） ──
    emotion_keys = list(turns[0]["psyche_state"]["emotions"].keys())
    emotion_stats: dict[str, dict[str, float]] = {}
    for emo_key in emotion_keys:
        values = [t["psyche_state"]["emotions"][emo_key] for t in turns]
        emotion_stats[emo_key] = {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "stddev": round(_safe_stddev(values), 4),
        }
    stats["emotions"] = emotion_stats

    # ── ムード valence/arousal ──
    mood_stats: dict[str, dict[str, float]] = {}
    for mood_key in ["valence", "arousal"]:
        values = [t["psyche_state"]["mood"][mood_key] for t in turns]
        mood_stats[mood_key] = {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "stddev": round(_safe_stddev(values), 4),
        }
    stats["mood"] = mood_stats

    # ── ドライブ（各ドライブ種別） ──
    drive_keys = list(turns[0]["psyche_state"]["drives"].keys())
    drive_stats: dict[str, dict[str, float]] = {}
    for drv_key in drive_keys:
        values = [t["psyche_state"]["drives"][drv_key] for t in turns]
        drive_stats[drv_key] = {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "stddev": round(_safe_stddev(values), 4),
        }
    stats["drives"] = drive_stats

    # ── 恐怖指数 ──
    fear_values = [t["psyche_state"]["fear_level"] for t in turns]
    stats["fear_level"] = {
        "min": round(min(fear_values), 4),
        "max": round(max(fear_values), 4),
        "mean": round(sum(fear_values) / len(fear_values), 4),
        "stddev": round(_safe_stddev(fear_values), 4),
    }

    # ── ポリシーラベル: 各ラベルの選択回数と全体に占める比率 ──
    policy_counts: dict[str, int] = {}
    for t in turns:
        label = t["policy"]["policy_label"]
        policy_counts[label] = policy_counts.get(label, 0) + 1
    total_turns = len(turns)
    policy_stats: dict[str, dict[str, float]] = {}
    for label, count in sorted(policy_counts.items()):
        policy_stats[label] = {
            "count": count,
            "ratio": round(count / total_turns, 4),
        }
    stats["policy_distribution"] = policy_stats

    # ── enrichment総文字数 ──
    enrichment_totals = [
        t["enrichment_chars"]["total"]
        for t in turns
        if "enrichment_chars" in t
    ]
    if enrichment_totals:
        stats["enrichment_total_chars"] = {
            "min": min(enrichment_totals),
            "max": max(enrichment_totals),
            "mean": round(sum(enrichment_totals) / len(enrichment_totals), 4),
            "stddev": round(_safe_stddev([float(v) for v in enrichment_totals]), 4),
        }

        # ── enrichmentセクション別文字数 ──
        section_stats: dict[str, dict[str, float]] = {}
        for header in ENRICHMENT_SECTION_HEADERS:
            section_values = [
                t["enrichment_chars"].get(header, 0)
                for t in turns
                if "enrichment_chars" in t
            ]
            if section_values:
                section_stats[header] = {
                    "min": min(section_values),
                    "max": max(section_values),
                    "mean": round(
                        sum(section_values) / len(section_values), 4
                    ),
                }
        stats["enrichment_sections"] = section_stats

    return stats


# ── Cross-Scenario Diff Report ───────────────────────────────

def generate_diff_report(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """複数シナリオの実行結果からシナリオ間差分レポートを生成する。

    差分レポートは事実記述のみであり、「どちらが良い」「どちらが望ましい」の
    判定を含まない。

    Args:
        results: シナリオ名をキー、run_simulation()の戻り値を値とする辞書。
                 2件以上のシナリオが必要。

    Returns:
        シナリオ間差分レポート辞書

    Raises:
        ValueError: 2件未満のシナリオが渡された場合
    """
    if len(results) < 2:
        raise ValueError("At least 2 scenario results are required for diff report.")

    scenario_names = sorted(results.keys())
    report: dict[str, Any] = {
        "scenarios_compared": scenario_names,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 各シナリオの統計サマリーを算出
    stats_map: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        stats_map[name] = compute_statistics(result)

    # ── 感情値の終了時点での分布比較 ──
    final_emotions: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        turns = result["turns"]
        if turns:
            final_emotions[name] = turns[-1]["psyche_state"]["emotions"]
    report["final_emotions"] = final_emotions

    # ── ムードvalence/arousalの推移範囲（最小値-最大値）の比較 ──
    mood_ranges: dict[str, dict[str, dict[str, float]]] = {}
    for name, st in stats_map.items():
        if "mood" in st:
            mood_ranges[name] = {
                k: {"min": v["min"], "max": v["max"]}
                for k, v in st["mood"].items()
            }
    report["mood_ranges"] = mood_ranges

    # ── ポリシーラベルの選択回数分布の比較 ──
    policy_distributions: dict[str, dict[str, Any]] = {}
    for name, st in stats_map.items():
        if "policy_distribution" in st:
            policy_distributions[name] = st["policy_distribution"]
    report["policy_distributions"] = policy_distributions

    # ── enrichmentテキスト総文字数の推移範囲の比較 ──
    enrichment_ranges: dict[str, dict[str, Any]] = {}
    for name, st in stats_map.items():
        if "enrichment_total_chars" in st:
            ec = st["enrichment_total_chars"]
            enrichment_ranges[name] = {"min": ec["min"], "max": ec["max"]}
    report["enrichment_total_char_ranges"] = enrichment_ranges

    # ── 恐怖指数の推移範囲の比較 ──
    fear_ranges: dict[str, dict[str, float]] = {}
    for name, st in stats_map.items():
        if "fear_level" in st:
            fl = st["fear_level"]
            fear_ranges[name] = {"min": fl["min"], "max": fl["max"]}
    report["fear_level_ranges"] = fear_ranges

    return report


# ── CLI ───────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="long_term_sim",
        description="Psyche long-term behaviour simulation tool",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name to run",
    )
    parser.add_argument(
        "--custom-sequence",
        type=str,
        default=None,
        help="Path to JSON file with custom pattern key list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--delta-time",
        type=float,
        default=2.0,
        help="Virtual seconds between turns (default: 2.0)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="sim_user",
        help="Simulated user ID (default: sim_user)",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List available input patterns and exit",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Include statistics summary in output",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Run multiple scenarios and generate diff report",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_scenarios:
        for name, seq in SCENARIOS.items():
            print(f"{name}: {len(seq)} turns")
        return

    if args.list_patterns:
        for name, pat in INPUT_PATTERNS.items():
            print(f"{name}: emotion={pat['emotion']}, valence={pat['emotion_valence']}, intent={pat['intent']}")
        return

    # ── Compare mode: 複数シナリオ実行 + 差分レポート ──
    if args.compare:
        # バリデーション
        for sc_name in args.compare:
            if sc_name not in SCENARIOS:
                parser.error(
                    f"Unknown scenario: {sc_name}. "
                    f"Available: {list(SCENARIOS.keys())}"
                )

        results: dict[str, dict[str, Any]] = {}
        for sc_name in args.compare:
            results[sc_name] = run_simulation(
                scenario_name=sc_name,
                delta_time=args.delta_time,
                user_id=args.user_id,
            )

        diff_report = generate_diff_report(results)

        output_data: dict[str, Any] = {
            "diff_report": diff_report,
        }

        # 各シナリオのstatsも含める
        if args.stats:
            output_data["scenario_stats"] = {
                name: compute_statistics(res)
                for name, res in results.items()
            }

        output_text = json.dumps(output_data, ensure_ascii=False, indent=2)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")
            print(f"Output written to {out_path}")
        else:
            print(output_text)
        return

    # ── Single scenario mode ──
    # Resolve custom sequence
    custom_seq = None
    if args.custom_sequence:
        custom_path = Path(args.custom_sequence)
        custom_seq = json.loads(custom_path.read_text(encoding="utf-8"))

    if args.scenario is None and custom_seq is None:
        parser.error("Either --scenario or --custom-sequence is required.")

    result = run_simulation(
        scenario_name=args.scenario,
        custom_sequence=custom_seq,
        delta_time=args.delta_time,
        user_id=args.user_id,
    )

    output_data = dict(result)

    # 統計サマリー付加
    if args.stats:
        output_data["statistics"] = compute_statistics(result)

    output_text = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Output written to {out_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
