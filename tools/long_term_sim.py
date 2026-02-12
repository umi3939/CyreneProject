"""
tools/long_term_sim.py - 長期挙動観測シミュレータ

既存のpsycheシステムを変更せずに、事前定義した入力パターンで自動会話を回し、
各ターンの内部状態を時系列JSONに記録する。

Usage::

    python -m tools.long_term_sim --scenario repeated_failure
    python -m tools.long_term_sim --custom-sequence patterns.json
    python -m tools.long_term_sim --list-scenarios
    python -m tools.long_term_sim --list-patterns
"""

from __future__ import annotations

import argparse
import json
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
}


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
) -> dict[str, Any]:
    """1ターン分の観測レコードを生成する。"""
    p = orch.psyche
    emo = p.emotions.as_dict()
    drives = p.drives.as_dict()

    # Responsibility state
    resp_state = orch._responsibility_mgr.get_state(user_id)
    resp_influence = orch._responsibility_mgr.get_influence(user_id)

    return {
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

    # Initialize orchestrator with isolated temp directory
    tmpdir = tempfile.mkdtemp(prefix="psyche_sim_")
    orch = PsycheOrchestrator(memory_count=0, data_dir=Path(tmpdir))

    started_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []

    for turn_idx, pattern_key in enumerate(sequence):
        turn_num = turn_idx + 1

        # Step 1: Create Percept
        percept = Percept(**INPUT_PATTERNS[pattern_key])

        # Step 2: post_response_update (Phase 1-29)
        orch.post_response_update(percept, delta_time, user_id)

        # Step 3: select_policy_dict (Phase 30-35)
        policy = orch.select_policy_dict(percept, [], user_id)

        # Step 4: evaluate_outcome — find unevaluated decision
        outcome = _pattern_to_outcome(pattern_key)
        resp_state = orch._responsibility_mgr.get_state(user_id)
        for rec in reversed(resp_state.recent_decisions):
            if not rec.get("evaluated", False):
                orch._responsibility_mgr.evaluate_outcome(
                    user_id, rec["id"], outcome,
                )
                break

        # Step 5: extract record
        record = _extract_turn_record(
            turn=turn_num,
            tick=orch.tick_count,
            pattern_key=pattern_key,
            percept=percept,
            orch=orch,
            policy=policy,
            outcome=outcome,
            user_id=user_id,
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
            "version": 1,
        },
        "turns": records,
    }


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

    output_text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Output written to {out_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
