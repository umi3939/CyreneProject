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
    python -m tools.long_term_sim --divergence smoke --num-instances 3 --warmup-turns 5
    python -m tools.long_term_sim --ab-compare smoke
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
from psyche.orchestrator_5tick_phases import _derive_dynamic_cooldown
from tools.return_pathway_monitor import (
    PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E,
)


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
    # ── 帰還経路相互作用分析用 ──
    "high_positive": {
        "text": "本当に最高！ありがとう、大好き！",
        "emotion": "happy",
        "emotion_valence": 0.9,
        "intent": "expression",
    },
    "high_negative": {
        "text": "最悪だ、もう耐えられない、許せない",
        "emotion": "angry",
        "emotion_valence": -0.9,
        "intent": "expression",
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
    # ── Cycle 9 動態検証シナリオ ──
    "c9_high_arousal": (
        ["loving", "angry", "loving", "angry", "fearful"] * 10
    ),
    "c9_monotone": (
        ["neutral"] * 50
    ),
    "c9_abrupt_shift": (
        ["positive", "loving"] * 13
        + ["negative", "angry", "rejected", "fearful"] * 6
    ),
    # ── 帰還経路相互作用分析シナリオ ──
    "return_interaction_simultaneous": (
        ["high_positive", "loving", "high_negative", "angry", "fearful"] * 10
    ),
    "return_interaction_neutral": (
        ["neutral"] * 50
    ),
    "return_interaction_residual": (
        ["high_positive", "high_negative", "loving", "angry", "fearful"] * 5
        + ["neutral"] * 25
    ),
}

# 多人数切替シナリオのユーザーID切替定義
# キー: シナリオ名, 値: ターン数分のuser_idリスト
SCENARIO_USER_IDS: dict[str, list[str]] = {
    "multi_person": [
        f"sim_user_{(i % 3) + 1}" for i in range(len(SCENARIOS["multi_person"]))
    ],
}


# ── Return Pathway Identifiers ───────────────────────────────

RETURN_PATHWAY_IDS: list[str] = [PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E]


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
        "high_positive": {
            "user_reaction": "positive",
            "relationship_delta": 0.2,
            "expectation_gap": 0.0,
        },
        "high_negative": {
            "user_reaction": "negative",
            "relationship_delta": -0.2,
            "expectation_gap": 0.5,
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


# ── Return Pathway Reading ──────────────────────────────────

def _read_return_pathway_tick(orch: PsycheOrchestrator) -> dict[str, Any]:
    """帰還経路モニターから現在ティックの発火情報を読み取る。

    処理結果の読み取りのみであり、帰還経路の処理には影響しない。
    読み取り失敗時は空のレコードを返す（安全弁1: 安全な無視）。

    Returns:
        帰還経路の発火情報辞書。発火がない場合は空の構造を返す。
    """
    try:
        monitor = orch._return_pathway_monitor
        last = monitor.last_tick_record
        if last is not None:
            return {
                "fired_pathways": last.get("fired_pathways", []),
                "fire_count": last.get("fire_count", 0),
                "combined_deltas": {
                    k: round(v, 6)
                    for k, v in last.get("combined_deltas", {}).items()
                    if isinstance(v, (int, float))
                },
            }
    except Exception:
        pass
    # 発火なしまたは読み取り失敗
    return {
        "fired_pathways": [],
        "fire_count": 0,
        "combined_deltas": {},
    }


# ── Cycle 9 Dynamics Reading ──────────────────────────────

def _read_cycle9_dynamics(
    orch: PsycheOrchestrator,
    drives_before: dict[str, float],
) -> dict[str, Any]:
    """Cycle 9固有の動態観測項目を読み取る。

    全てのCycle 9固有読み取りをtry-exceptで囲み、読み取り失敗時は
    フォールバック値を返す（安全弁1: 安全な無視）。
    psycheの状態を変更する呼び出しを含まない（READ-ONLY）。

    Args:
        orch: オーケストレータ
        drives_before: ターン処理前のドライブベクトル

    Returns:
        cycle9_dynamics辞書（4サブフィールド）
    """
    result: dict[str, Any] = {}

    # ── drive_dynamics: 合成後の各軸変動量、合成後総変動量 ──
    try:
        drives_after = orch._psyche.drives.as_dict()
        per_axis: dict[str, float] = {}
        total_variation = 0.0
        for axis in drives_after:
            before_val = drives_before.get(axis, 0.5)
            delta = drives_after[axis] - before_val
            per_axis[axis] = round(delta, 6)
            total_variation += abs(delta)
        result["drive_dynamics"] = {
            "per_axis_delta": per_axis,
            "total_variation": round(total_variation, 6),
        }
    except Exception:
        result["drive_dynamics"] = {
            "per_axis_delta": {},
            "total_variation": 0.0,
        }

    # ── exp_bandwidth: Phase 26-EXP帯域拡大 ──
    try:
        last_tick = getattr(orch, '_exp_bandwidth_last_tick', None)
        drive_mult = getattr(orch, '_exp_drive_total_limit_multiplier', None)
        score_add = getattr(orch, '_exp_score_band_addition', None)
        fired = (drive_mult is not None and drive_mult > 1.0) or (
            score_add is not None and score_add > 0.0
        )
        result["exp_bandwidth"] = {
            "fired": fired,
            "last_applied_tick": last_tick if last_tick is not None else -1,
            "drive_limit_multiplier": round(drive_mult, 6) if drive_mult is not None else None,
            "score_band_addition": round(score_add, 6) if score_add is not None else None,
        }
    except Exception:
        result["exp_bandwidth"] = {
            "fired": False,
            "last_applied_tick": -1,
            "drive_limit_multiplier": None,
            "score_band_addition": None,
        }

    # ── dynamic_cooldown: 冷却期間の動的導出 ──
    try:
        arousal = orch._psyche.mood.arousal
        # 駆動変動幅: _exp_prev_drives から算出
        current_drives = orch._psyche.drives.as_dict()
        prev_drives = getattr(orch, '_exp_prev_drives', None)
        drive_variation = 0.0
        if prev_drives is not None:
            diffs = []
            for axis in current_drives:
                prev_val = prev_drives.get(axis, 0.5)
                diffs.append(abs(current_drives[axis] - prev_val))
            if diffs:
                drive_variation = max(diffs)
        cooldown_ticks = _derive_dynamic_cooldown(arousal, drive_variation)
        result["dynamic_cooldown"] = {
            "cooldown_ticks": cooldown_ticks,
            "arousal_input": round(arousal, 6),
            "drive_variation_input": round(drive_variation, 6),
        }
    except Exception:
        result["dynamic_cooldown"] = {
            "cooldown_ticks": 0,
            "arousal_input": 0.0,
            "drive_variation_input": 0.0,
        }

    # ── emotion_return_tracking: 感情帰還方向追跡 ──
    try:
        mer = orch._memory_emotion_return
        state = mer._state
        pos_count = state.direction_consecutive_count_positive
        neg_count = state.direction_consecutive_count_negative
        v_mod, a_mod = mer.get_tracking_speed_modulation()
        result["emotion_return_tracking"] = {
            "positive_consecutive_count": round(pos_count, 6),
            "negative_consecutive_count": round(neg_count, 6),
            "valence_modulation": round(v_mod, 6),
            "arousal_modulation": round(a_mod, 6),
        }
    except Exception:
        result["emotion_return_tracking"] = {
            "positive_consecutive_count": 0.0,
            "negative_consecutive_count": 0.0,
            "valence_modulation": 0.0,
            "arousal_modulation": 0.0,
        }

    return result


# ── Return Interaction Analysis ─────────────────────────────

# 帰還経路相互作用分析対象のシナリオ名セット
_RETURN_INTERACTION_SCENARIOS = frozenset({
    "return_interaction_simultaneous",
    "return_interaction_neutral",
    "return_interaction_residual",
})


def _read_return_interaction_tick(
    orch: PsycheOrchestrator,
    emotions_before: dict[str, float],
    drives_before: dict[str, float],
    mood_before: dict[str, float],
) -> dict[str, Any]:
    """帰還経路相互作用分析用のティック単位記録を読み取る。

    各ティックにおける帰還経路の発火状態と、感情・ドライブ・ムードの
    変動量を対にして記録する。READ-ONLY操作のみ。

    Args:
        orch: オーケストレータ
        emotions_before: ターン処理前の感情ベクトル
        drives_before: ターン処理前のドライブベクトル
        mood_before: ターン処理前のムード(valence, arousal)

    Returns:
        帰還経路相互作用記録辞書
    """
    result: dict[str, Any] = {}

    # 帰還経路発火状態の読み取り
    try:
        monitor = orch._return_pathway_monitor
        last = monitor.last_tick_record
        if last is not None:
            result["fired_pathways"] = last.get("fired_pathways", [])
            result["fire_count"] = last.get("fire_count", 0)
            # 種類別の合算変動値を記録
            result["combined_emotion_deltas"] = {
                k: round(v, 6)
                for k, v in last.get("combined_emotion_deltas", {}).items()
                if isinstance(v, (int, float))
            }
            result["combined_drive_deltas"] = {
                k: round(v, 6)
                for k, v in last.get("combined_drive_deltas", {}).items()
                if isinstance(v, (int, float))
            }
            result["combined_mood_speed_deltas"] = {
                k: round(v, 6)
                for k, v in last.get("combined_mood_speed_deltas", {}).items()
                if isinstance(v, (int, float))
            }
        else:
            result["fired_pathways"] = []
            result["fire_count"] = 0
            result["combined_emotion_deltas"] = {}
            result["combined_drive_deltas"] = {}
            result["combined_mood_speed_deltas"] = {}
    except Exception:
        result["fired_pathways"] = []
        result["fire_count"] = 0
        result["combined_emotion_deltas"] = {}
        result["combined_drive_deltas"] = {}
        result["combined_mood_speed_deltas"] = {}

    # 状態変動量の算出(処理前後の差分)
    try:
        emotions_after = orch._psyche.emotions.as_dict()
        emotion_deltas: dict[str, float] = {}
        for k in emotions_after:
            before_val = emotions_before.get(k, 0.0)
            emotion_deltas[k] = round(emotions_after[k] - before_val, 6)
        result["emotion_variation"] = emotion_deltas
        result["emotion_total_variation"] = round(
            sum(abs(v) for v in emotion_deltas.values()), 6
        )
    except Exception:
        result["emotion_variation"] = {}
        result["emotion_total_variation"] = 0.0

    try:
        drives_after = orch._psyche.drives.as_dict()
        drive_deltas: dict[str, float] = {}
        for k in drives_after:
            before_val = drives_before.get(k, 0.5)
            drive_deltas[k] = round(drives_after[k] - before_val, 6)
        result["drive_variation"] = drive_deltas
        result["drive_total_variation"] = round(
            sum(abs(v) for v in drive_deltas.values()), 6
        )
    except Exception:
        result["drive_variation"] = {}
        result["drive_total_variation"] = 0.0

    try:
        mood_after_valence = orch._psyche.mood.valence
        mood_after_arousal = orch._psyche.mood.arousal
        result["mood_variation"] = {
            "valence": round(mood_after_valence - mood_before.get("valence", 0.0), 6),
            "arousal": round(mood_after_arousal - mood_before.get("arousal", 0.5), 6),
        }
        result["mood_total_variation"] = round(
            abs(result["mood_variation"]["valence"])
            + abs(result["mood_variation"]["arousal"]),
            6,
        )
    except Exception:
        result["mood_variation"] = {"valence": 0.0, "arousal": 0.0}
        result["mood_total_variation"] = 0.0

    return result


def _compute_return_interaction_analysis(
    turns: list[dict[str, Any]],
    scenario_label: str,
) -> dict[str, Any]:
    """帰還経路相互作用分析シナリオの結果から分析情報を生成する。

    事実記述のみ。優劣判定・最適化提案を含まない。

    Args:
        turns: ターンレコードのリスト(return_interactionフィールド含む)
        scenario_label: シナリオ名

    Returns:
        帰還経路相互作用分析辞書
    """
    analysis: dict[str, Any] = {"scenario": scenario_label}

    # return_interactionフィールドがあるターンのみ
    ri_turns = [t for t in turns if "return_interaction" in t]
    if not ri_turns:
        return analysis

    # 同時発火ティックの特定と変動量比較
    simultaneous_ticks: list[dict[str, Any]] = []
    non_simultaneous_ticks: list[dict[str, Any]] = []

    for t in ri_turns:
        ri = t["return_interaction"]
        entry = {
            "turn": t["turn"],
            "fire_count": ri["fire_count"],
            "fired_pathways": ri["fired_pathways"],
            "emotion_total_variation": ri["emotion_total_variation"],
            "drive_total_variation": ri["drive_total_variation"],
            "mood_total_variation": ri["mood_total_variation"],
        }
        if ri["fire_count"] >= 2:
            simultaneous_ticks.append(entry)
        else:
            non_simultaneous_ticks.append(entry)

    analysis["simultaneous_fire_count"] = len(simultaneous_ticks)
    analysis["non_simultaneous_fire_count"] = len(non_simultaneous_ticks)

    # 同時発火ティックの変動量統計
    if simultaneous_ticks:
        emo_vars = [e["emotion_total_variation"] for e in simultaneous_ticks]
        drv_vars = [e["drive_total_variation"] for e in simultaneous_ticks]
        mood_vars = [e["mood_total_variation"] for e in simultaneous_ticks]
        analysis["simultaneous_variation"] = {
            "emotion_total": {
                "min": round(min(emo_vars), 6),
                "max": round(max(emo_vars), 6),
                "mean": round(sum(emo_vars) / len(emo_vars), 6),
            },
            "drive_total": {
                "min": round(min(drv_vars), 6),
                "max": round(max(drv_vars), 6),
                "mean": round(sum(drv_vars) / len(drv_vars), 6),
            },
            "mood_total": {
                "min": round(min(mood_vars), 6),
                "max": round(max(mood_vars), 6),
                "mean": round(sum(mood_vars) / len(mood_vars), 6),
            },
        }

    # 非同時発火ティックの変動量統計
    if non_simultaneous_ticks:
        emo_vars = [e["emotion_total_variation"] for e in non_simultaneous_ticks]
        drv_vars = [e["drive_total_variation"] for e in non_simultaneous_ticks]
        mood_vars = [e["mood_total_variation"] for e in non_simultaneous_ticks]
        analysis["non_simultaneous_variation"] = {
            "emotion_total": {
                "min": round(min(emo_vars), 6),
                "max": round(max(emo_vars), 6),
                "mean": round(sum(emo_vars) / len(emo_vars), 6),
            },
            "drive_total": {
                "min": round(min(drv_vars), 6),
                "max": round(max(drv_vars), 6),
                "mean": round(sum(drv_vars) / len(drv_vars), 6),
            },
            "mood_total": {
                "min": round(min(mood_vars), 6),
                "max": round(max(mood_vars), 6),
                "mean": round(sum(mood_vars) / len(mood_vars), 6),
            },
        }

    # シナリオ3(残響分析): 入力パターンの切替点を検出し、残響期間を記録
    if scenario_label == "return_interaction_residual":
        # 切替点: 最後の非neutralターンの次のターン
        neutral_start_turn: int | None = None
        for t in ri_turns:
            if t["input_pattern"] == "neutral" and neutral_start_turn is None:
                # 前のターンが非neutralであれば切替点
                idx = ri_turns.index(t)
                if idx > 0 and ri_turns[idx - 1]["input_pattern"] != "neutral":
                    neutral_start_turn = t["turn"]

        if neutral_start_turn is not None:
            analysis["neutral_start_turn"] = neutral_start_turn
            # 残響期間: 中立入力開始後、変動量が一定水準以下に収束するまでのティック数
            # 水準: emotion_total_variation + drive_total_variation + mood_total_variation < 0.01
            threshold = 0.01
            residual_ticks = 0
            converged = False
            for t in ri_turns:
                if t["turn"] < neutral_start_turn:
                    continue
                ri = t["return_interaction"]
                total_var = (
                    ri["emotion_total_variation"]
                    + ri["drive_total_variation"]
                    + ri["mood_total_variation"]
                )
                if total_var < threshold:
                    converged = True
                    break
                residual_ticks += 1

            analysis["residual_ticks"] = residual_ticks
            analysis["converged"] = converged

            # 残響期間中の各ティックの変動量推移
            residual_variations: list[dict[str, float]] = []
            for t in ri_turns:
                if t["turn"] < neutral_start_turn:
                    continue
                ri = t["return_interaction"]
                residual_variations.append({
                    "turn": t["turn"],
                    "emotion_total": ri["emotion_total_variation"],
                    "drive_total": ri["drive_total_variation"],
                    "mood_total": ri["mood_total_variation"],
                })
            analysis["residual_variation_series"] = residual_variations

    # 帰還経路別の発火頻度(全シナリオ共通)
    pathway_counts: dict[str, int] = {}
    for t in ri_turns:
        ri = t["return_interaction"]
        for pid in ri["fired_pathways"]:
            pathway_counts[pid] = pathway_counts.get(pid, 0) + 1
    total_ri_turns = len(ri_turns)
    analysis["pathway_fire_counts"] = pathway_counts
    analysis["pathway_fire_ratios"] = {
        pid: round(count / total_ri_turns, 6)
        for pid, count in pathway_counts.items()
    }

    return analysis


# ── Intermediate Snapshot ──────────────────────────────────

def _compute_snapshot_positions(total_turns: int) -> list[int]:
    """中間スナップショットの記録位置を機械的に等間隔で決定する。

    全ターン数に対する一定割合の地点を返す。
    短いシナリオ（10ターン以下）ではスナップショットを省略する。

    Args:
        total_turns: シナリオの全ターン数

    Returns:
        スナップショットを記録するターン番号のリスト（1-indexed）
    """
    if total_turns <= 10:
        return []
    # 25%, 50%, 75% の3地点
    positions = []
    for frac in [0.25, 0.50, 0.75]:
        pos = int(total_turns * frac)
        if pos >= 1 and pos <= total_turns:
            positions.append(pos)
    return sorted(set(positions))


def _extract_snapshot(orch: PsycheOrchestrator, user_id: str) -> dict[str, Any]:
    """中間スナップショットとして全内部状態のサマリーを読み取る。

    読み取りのみであり、内部状態の変更を伴わない。

    Returns:
        スナップショット辞書
    """
    p = orch.psyche
    snapshot: dict[str, Any] = {
        "emotions": {k: round(v, 4) for k, v in p.emotions.as_dict().items()},
        "drives": {k: round(v, 4) for k, v in p.drives.as_dict().items()},
        "mood": {
            "valence": round(p.mood.valence, 4),
            "arousal": round(p.mood.arousal, 4),
        },
        "fear_level": round(p.fear_level, 4),
        "dominant_emotion": p.dominant_emotion,
    }
    # 帰還経路の累積情報をスナップショットに含める
    try:
        monitor = orch._return_pathway_monitor
        snapshot["return_pathway_cumulative"] = dict(monitor.pathway_fire_counts)
    except Exception:
        snapshot["return_pathway_cumulative"] = {}
    return snapshot


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
    return_pathway: dict[str, Any] | None = None,
    cycle9_dynamics: dict[str, Any] | None = None,
    return_interaction: dict[str, Any] | None = None,
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

    # 拡張フィールド: 帰還経路発火情報（処理Aに対応）
    if return_pathway is not None:
        record["return_pathway"] = return_pathway

    # 拡張フィールド: Cycle 9動態観測（既存フィールドの末尾に追加）
    if cycle9_dynamics is not None:
        record["cycle9_dynamics"] = cycle9_dynamics

    # 拡張フィールド: 帰還経路相互作用分析（処理完了後の末尾追加）
    if return_interaction is not None:
        record["return_interaction"] = return_interaction

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
    sim_memory_count = 0  # シミュレーション中の記憶保存カウンタ

    # 帰還経路相互作用分析の対象シナリオかどうか
    is_interaction_scenario = scenario_label in _RETURN_INTERACTION_SCENARIOS

    # 中間スナップショット位置の決定（処理B: 機械的に等間隔で決定）
    total_turns = len(sequence)
    snapshot_positions = _compute_snapshot_positions(total_turns)
    snapshots: list[dict[str, Any]] = []

    for turn_idx, pattern_key in enumerate(sequence):
        turn_num = turn_idx + 1

        # ターンごとのuser_id決定
        turn_user_id = user_id
        if user_ids_per_turn is not None and turn_idx < len(user_ids_per_turn):
            turn_user_id = user_ids_per_turn[turn_idx]

        # Step 0: 前値の記録（READ-ONLY）
        try:
            drives_before = {
                k: v for k, v in orch._psyche.drives.as_dict().items()
            }
        except Exception:
            drives_before = {}

        # 帰還経路相互作用分析用の前値記録
        emotions_before: dict[str, float] = {}
        mood_before: dict[str, float] = {}
        if is_interaction_scenario:
            try:
                emotions_before = {
                    k: v for k, v in orch._psyche.emotions.as_dict().items()
                }
            except Exception:
                pass
            try:
                mood_before = {
                    "valence": orch._psyche.mood.valence,
                    "arousal": orch._psyche.mood.arousal,
                }
            except Exception:
                pass

        # Step 1: Create Percept
        percept = Percept(**INPUT_PATTERNS[pattern_key])

        # Step 2: post_response_update (Phase 1-29)
        # Phase 3 (attachment bond update) と Phase 7 (_recompute_fear) が内部で実行される
        orch.post_response_update(percept, delta_time, turn_user_id)

        # Step 2a: 記憶保存の模擬 — 感情的に顕著な入力で on_memory_saved() を呼ぶ
        # 中立入力では記憶保存を発火しない（ティック数の単純反映ではない）
        if abs(percept.emotion_valence) > 0.3:
            sim_memory_count += 1
            orch.on_memory_saved(
                summary=percept.text,
                keywords=[percept.emotion or "unknown"],
                memory_count=sim_memory_count,
            )

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

        # Step 5a: 帰還経路発火情報の読み取り（処理A）
        # 帰還経路の処理が完了した後に読み取る。処理自体には影響しない。
        return_pathway = _read_return_pathway_tick(orch)

        # Step 5b: Cycle 9動態観測項目の読み取り（ターン処理完了後の末尾追加）
        cycle9_dynamics = _read_cycle9_dynamics(orch, drives_before)

        # Step 5c: 帰還経路相互作用分析の読み取り（対象シナリオのみ）
        return_interaction: dict[str, Any] | None = None
        if is_interaction_scenario:
            return_interaction = _read_return_interaction_tick(
                orch, emotions_before, drives_before, mood_before,
            )

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
            return_pathway=return_pathway,
            cycle9_dynamics=cycle9_dynamics,
            return_interaction=return_interaction,
        )
        records.append(record)

        # Step 7: 中間スナップショット記録（処理B）
        # ターン処理の後に行う読み取りのみであり、内部状態の変更を伴わない。
        if turn_num in snapshot_positions:
            snap = _extract_snapshot(orch, turn_user_id)
            snap["turn"] = turn_num
            snap["tick"] = orch.tick_count
            snapshots.append(snap)

    finished_at = datetime.now().isoformat(timespec="seconds")

    # 帰還経路の累積サマリー（リアルタイム観測構造がある場合、任意で含める）
    return_pathway_summary: dict[str, Any] = {}
    try:
        monitor = orch._return_pathway_monitor
        return_pathway_summary = monitor.get_summary()
    except Exception:
        pass

    result: dict[str, Any] = {
        "metadata": {
            "scenario": scenario_label,
            "total_turns": total_turns,
            "delta_time_per_turn": delta_time,
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "version": 3,
        },
        "turns": records,
    }

    # 中間スナップショットの追加（存在する場合のみ）
    if snapshots:
        result["snapshots"] = snapshots

    # 帰還経路セッションサマリーの追加
    if return_pathway_summary:
        result["return_pathway_summary"] = return_pathway_summary

    # 帰還経路相互作用分析の追加（対象シナリオのみ）
    if is_interaction_scenario:
        result["return_interaction_analysis"] = _compute_return_interaction_analysis(
            records, scenario_label,
        )

    return result


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

    # ── 帰還経路統計（処理C） ──
    # 各帰還経路の発火回数、発火ターン比率、同時発火回数、累積帯域変動量。
    # 統計量は事実記述のみであり、「期待値」「正常範囲」の概念を含まない。
    turns_with_rp = [t for t in turns if "return_pathway" in t]
    if turns_with_rp:
        rp_stats: dict[str, Any] = {}

        # 各帰還経路の発火回数と発火ターン比率
        pathway_fire_counts: dict[str, int] = {pid: 0 for pid in RETURN_PATHWAY_IDS}
        concurrent_2plus = 0
        cumulative_deltas: dict[str, float] = {}

        for t in turns_with_rp:
            rp = t["return_pathway"]
            fired = rp.get("fired_pathways", [])
            fire_count = rp.get("fire_count", 0)

            for pid in fired:
                if pid in pathway_fire_counts:
                    pathway_fire_counts[pid] += 1

            if fire_count >= 2:
                concurrent_2plus += 1

            for dim, delta in rp.get("combined_deltas", {}).items():
                if isinstance(delta, (int, float)):
                    cumulative_deltas[dim] = cumulative_deltas.get(dim, 0.0) + delta

        total_rp_turns = len(turns_with_rp)
        rp_stats["pathway_fire_counts"] = pathway_fire_counts
        rp_stats["pathway_fire_ratios"] = {
            pid: round(count / total_rp_turns, 4)
            for pid, count in pathway_fire_counts.items()
        }
        rp_stats["concurrent_2plus_count"] = concurrent_2plus
        rp_stats["cumulative_deltas"] = {
            dim: round(v, 6) for dim, v in cumulative_deltas.items()
        }

        stats["return_pathway"] = rp_stats

    # 帰還経路セッションサマリー（結果全体に含まれている場合）
    if "return_pathway_summary" in result:
        stats["return_pathway_session"] = result["return_pathway_summary"]

    # ── Cycle 9 動態統計 ──
    # cycle9_dynamicsフィールドが存在するターンのみから算出する。
    # 品質判定を含まない事実記述のみ。
    turns_with_c9 = [t for t in turns if "cycle9_dynamics" in t]
    if turns_with_c9:
        c9_stats: dict[str, Any] = {}

        # SD-3 合成後ドライブ変動量の分布
        total_variations = [
            t["cycle9_dynamics"]["drive_dynamics"]["total_variation"]
            for t in turns_with_c9
        ]
        c9_stats["drive_total_variation"] = {
            "min": round(min(total_variations), 6),
            "max": round(max(total_variations), 6),
            "mean": round(sum(total_variations) / len(total_variations), 6),
            "stddev": round(_safe_stddev(total_variations), 6),
        }

        # 各ドライブ軸の変動量分布
        axis_keys: set[str] = set()
        for t in turns_with_c9:
            axis_keys.update(t["cycle9_dynamics"]["drive_dynamics"]["per_axis_delta"].keys())
        drive_axis_stats: dict[str, dict[str, float]] = {}
        for axis in sorted(axis_keys):
            axis_vals = [
                t["cycle9_dynamics"]["drive_dynamics"]["per_axis_delta"].get(axis, 0.0)
                for t in turns_with_c9
            ]
            drive_axis_stats[axis] = {
                "min": round(min(axis_vals), 6),
                "max": round(max(axis_vals), 6),
                "mean": round(sum(axis_vals) / len(axis_vals), 6),
                "stddev": round(_safe_stddev(axis_vals), 6),
            }
        c9_stats["drive_per_axis_variation"] = drive_axis_stats

        # Phase 26-EXP 帯域拡大の発動ターン比率、経験強度係数の分布
        fired_count = sum(
            1 for t in turns_with_c9
            if t["cycle9_dynamics"]["exp_bandwidth"]["fired"]
        )
        c9_stats["exp_bandwidth_fire_ratio"] = round(
            fired_count / len(turns_with_c9), 6
        )
        drive_mults = [
            t["cycle9_dynamics"]["exp_bandwidth"]["drive_limit_multiplier"]
            for t in turns_with_c9
            if t["cycle9_dynamics"]["exp_bandwidth"]["drive_limit_multiplier"] is not None
        ]
        if drive_mults:
            c9_stats["exp_drive_limit_multiplier"] = {
                "min": round(min(drive_mults), 6),
                "max": round(max(drive_mults), 6),
                "mean": round(sum(drive_mults) / len(drive_mults), 6),
                "stddev": round(_safe_stddev(drive_mults), 6),
            }
        score_adds = [
            t["cycle9_dynamics"]["exp_bandwidth"]["score_band_addition"]
            for t in turns_with_c9
            if t["cycle9_dynamics"]["exp_bandwidth"]["score_band_addition"] is not None
        ]
        if score_adds:
            c9_stats["exp_score_band_addition"] = {
                "min": round(min(score_adds), 6),
                "max": round(max(score_adds), 6),
                "mean": round(sum(score_adds) / len(score_adds), 6),
                "stddev": round(_safe_stddev(score_adds), 6),
            }

        # 冷却動的期間の分布
        cooldown_vals = [
            t["cycle9_dynamics"]["dynamic_cooldown"]["cooldown_ticks"]
            for t in turns_with_c9
        ]
        c9_stats["dynamic_cooldown_ticks"] = {
            "min": min(cooldown_vals),
            "max": max(cooldown_vals),
            "mean": round(sum(cooldown_vals) / len(cooldown_vals), 4),
            "stddev": round(_safe_stddev([float(v) for v in cooldown_vals]), 4),
        }

        # 追従速度変調量の分布
        v_mods = [
            t["cycle9_dynamics"]["emotion_return_tracking"]["valence_modulation"]
            for t in turns_with_c9
        ]
        a_mods = [
            t["cycle9_dynamics"]["emotion_return_tracking"]["arousal_modulation"]
            for t in turns_with_c9
        ]
        c9_stats["tracking_speed_modulation"] = {
            "valence": {
                "min": round(min(v_mods), 6),
                "max": round(max(v_mods), 6),
                "mean": round(sum(v_mods) / len(v_mods), 6),
                "stddev": round(_safe_stddev(v_mods), 6),
            },
            "arousal": {
                "min": round(min(a_mods), 6),
                "max": round(max(a_mods), 6),
                "mean": round(sum(a_mods) / len(a_mods), 6),
                "stddev": round(_safe_stddev(a_mods), 6),
            },
        }

        stats["cycle9_dynamics"] = c9_stats

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

    # ── 帰還経路の動作状況比較（処理D） ──
    # シナリオ間で帰還経路の発火回数がどう異なるかの並列記述。
    # 「どちらが良い」「どちらが望ましい」の判定を含まない。
    rp_fire_counts: dict[str, dict[str, int]] = {}
    rp_cumulative_deltas: dict[str, dict[str, float]] = {}
    for name, st in stats_map.items():
        if "return_pathway" in st:
            rp = st["return_pathway"]
            rp_fire_counts[name] = rp.get("pathway_fire_counts", {})
            rp_cumulative_deltas[name] = rp.get("cumulative_deltas", {})
    if rp_fire_counts:
        report["return_pathway_fire_counts"] = rp_fire_counts
    if rp_cumulative_deltas:
        report["return_pathway_cumulative_deltas"] = rp_cumulative_deltas

    # ── Cycle 9 動態の並列比較 ──
    # シナリオ間のCycle 9固有統計の並列記述。品質判定を含まない。
    c9_comparisons: dict[str, dict[str, Any]] = {}
    for name, st in stats_map.items():
        if "cycle9_dynamics" in st:
            c9_comparisons[name] = st["cycle9_dynamics"]
    if c9_comparisons:
        report["cycle9_dynamics"] = c9_comparisons

    return report


# ── State Vector Extraction (12-dimensional) ─────────────────

# Dimension order for state vectors: 7 emotions + 3 drives + 2 mood
_STATE_VECTOR_EMOTION_KEYS = ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun")
_STATE_VECTOR_DRIVE_KEYS = ("social", "curiosity", "expression")
_STATE_VECTOR_MOOD_KEYS = ("valence", "arousal")

# Total number of dimensions in the state vector
STATE_VECTOR_DIM = (
    len(_STATE_VECTOR_EMOTION_KEYS)
    + len(_STATE_VECTOR_DRIVE_KEYS)
    + len(_STATE_VECTOR_MOOD_KEYS)
)


def _extract_state_vector(orch: PsycheOrchestrator) -> list[float]:
    """Extract a 12-dimensional state vector from the orchestrator.

    Order: joy, anger, sorrow, fear, surprise, love, fun,
           social, curiosity, expression, valence, arousal.

    READ-ONLY operation. No state modification.

    Returns:
        List of 12 float values.
    """
    p = orch.psyche
    emo = p.emotions.as_dict()
    drv = p.drives.as_dict()
    vec: list[float] = []
    for k in _STATE_VECTOR_EMOTION_KEYS:
        vec.append(emo.get(k, 0.0))
    for k in _STATE_VECTOR_DRIVE_KEYS:
        vec.append(drv.get(k, 0.5))
    vec.append(p.mood.valence)
    vec.append(p.mood.arousal)
    return vec


def _extract_state_vector_from_record(record: dict[str, Any]) -> list[float]:
    """Extract a 12-dimensional state vector from a turn record dict.

    Used for trajectory analysis on already-recorded data.

    Returns:
        List of 12 float values.
    """
    ps = record["psyche_state"]
    emo = ps["emotions"]
    drv = ps["drives"]
    mood = ps["mood"]
    vec: list[float] = []
    for k in _STATE_VECTOR_EMOTION_KEYS:
        vec.append(emo.get(k, 0.0))
    for k in _STATE_VECTOR_DRIVE_KEYS:
        vec.append(drv.get(k, 0.5))
    vec.append(mood.get("valence", 0.0))
    vec.append(mood.get("arousal", 0.3))
    return vec


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors of equal length."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _dimension_labels() -> list[str]:
    """Return the ordered labels for the 12 state vector dimensions."""
    labels: list[str] = []
    labels.extend(_STATE_VECTOR_EMOTION_KEYS)
    labels.extend(_STATE_VECTOR_DRIVE_KEYS)
    labels.extend(_STATE_VECTOR_MOOD_KEYS)
    return labels


# ── Measurement Mode 1: State Divergence ─────────────────────

def run_divergence_measurement(
    scenario_name: str | None = None,
    custom_sequence: list[str] | None = None,
    num_instances: int = 3,
    warmup_turns: int = 5,
    warmup_patterns: list[list[str]] | None = None,
    delta_time: float = 2.0,
    user_id: str = "sim_user",
    max_instances: int = 10,
) -> dict[str, Any]:
    """Run state divergence measurement with multiple orchestrator instances.

    Creates multiple PsycheOrchestrator instances with different initial
    conditions (via different warmup input sequences), then feeds the same
    input sequence to all instances and records pairwise Euclidean distances
    at each tick.

    Args:
        scenario_name: SCENARIOS name for the main input sequence.
        custom_sequence: Custom pattern key list (overrides scenario_name).
        num_instances: Number of orchestrator instances to create.
        warmup_turns: Number of warmup turns per instance. Each instance
            gets a different warmup pattern to create initial divergence.
        warmup_patterns: Optional explicit warmup pattern lists per instance.
            If None, auto-generated from available patterns.
        delta_time: Virtual seconds between turns.
        user_id: Simulated user ID.
        max_instances: Upper bound for num_instances (safety valve 6).

    Returns:
        Measurement result dict with metadata, per-tick distance records,
        and summary statistics.

    Raises:
        ValueError: Invalid scenario or pattern keys.
    """
    # Safety valve 6: instance count upper bound
    num_instances = min(num_instances, max_instances)
    if num_instances < 2:
        raise ValueError("num_instances must be at least 2.")

    # Resolve main sequence
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

    # Validate pattern keys
    for key in sequence:
        if key not in INPUT_PATTERNS:
            raise ValueError(f"Invalid pattern key: {key}")

    # Generate warmup patterns if not provided
    if warmup_patterns is None:
        available_keys = list(INPUT_PATTERNS.keys())
        warmup_patterns = []
        for i in range(num_instances):
            # Rotate through available patterns to create different initial states
            offset = i * 2  # offset to ensure diversity
            wp = []
            for j in range(warmup_turns):
                idx = (offset + j) % len(available_keys)
                wp.append(available_keys[idx])
            warmup_patterns.append(wp)
    else:
        if len(warmup_patterns) != num_instances:
            raise ValueError(
                f"warmup_patterns length ({len(warmup_patterns)}) "
                f"must match num_instances ({num_instances})."
            )
        for i, wp in enumerate(warmup_patterns):
            for key in wp:
                if key not in INPUT_PATTERNS:
                    raise ValueError(
                        f"Invalid warmup pattern key: {key} (instance {i})"
                    )

    started_at = datetime.now().isoformat(timespec="seconds")

    # Create isolated instances
    instances: list[PsycheOrchestrator] = []
    tmpdirs: list[str] = []
    for _ in range(num_instances):
        tmpdir = tempfile.mkdtemp(prefix="psyche_div_")
        tmpdirs.append(tmpdir)
        orch = PsycheOrchestrator(memory_count=0, data_dir=Path(tmpdir))
        instances.append(orch)

    # Phase 1: Warmup — each instance gets its own warmup sequence
    warmup_records: list[dict[str, Any]] = []
    for inst_idx, orch in enumerate(instances):
        wp = warmup_patterns[inst_idx]
        for pattern_key in wp:
            percept = Percept(**INPUT_PATTERNS[pattern_key])
            orch.post_response_update(percept, delta_time, user_id)
            policy = orch.select_policy_dict(percept, [], user_id)
            outcome = _pattern_to_outcome(pattern_key)
            resp_state = orch._responsibility_mgr.get_state(user_id)
            for rec in reversed(resp_state.recent_decisions):
                if not rec.get("evaluated", False):
                    orch._responsibility_mgr.evaluate_outcome(
                        user_id, rec["id"], outcome,
                    )
                    break

        # Record post-warmup state vector
        warmup_records.append({
            "instance": inst_idx,
            "warmup_pattern": wp,
            "state_vector": _extract_state_vector(orch),
        })

    # Phase 2: Main sequence — same inputs to all instances
    tick_records: list[dict[str, Any]] = []
    for turn_idx, pattern_key in enumerate(sequence):
        turn_num = turn_idx + 1
        percept = Percept(**INPUT_PATTERNS[pattern_key])
        outcome = _pattern_to_outcome(pattern_key)

        # Process each instance
        tick_vectors: list[list[float]] = []
        for orch in instances:
            orch.post_response_update(percept, delta_time, user_id)
            policy = orch.select_policy_dict(percept, [], user_id)
            resp_state = orch._responsibility_mgr.get_state(user_id)
            for rec in reversed(resp_state.recent_decisions):
                if not rec.get("evaluated", False):
                    orch._responsibility_mgr.evaluate_outcome(
                        user_id, rec["id"], outcome,
                    )
                    break
            tick_vectors.append(_extract_state_vector(orch))

        # Compute pairwise distances
        pairwise_distances: list[dict[str, Any]] = []
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dist = _euclidean_distance(tick_vectors[i], tick_vectors[j])
                pairwise_distances.append({
                    "pair": [i, j],
                    "distance": round(dist, 6),
                })

        # Mean distance across all pairs
        all_dists = [d["distance"] for d in pairwise_distances]
        mean_distance = round(sum(all_dists) / len(all_dists), 6) if all_dists else 0.0

        tick_record: dict[str, Any] = {
            "turn": turn_num,
            "input_pattern": pattern_key,
            "instance_vectors": [
                [round(v, 6) for v in vec] for vec in tick_vectors
            ],
            "pairwise_distances": pairwise_distances,
            "mean_distance": mean_distance,
        }
        tick_records.append(tick_record)

    finished_at = datetime.now().isoformat(timespec="seconds")

    # Compute divergence trajectory summary
    mean_distances = [r["mean_distance"] for r in tick_records]
    divergence_summary: dict[str, Any] = {}
    if mean_distances:
        divergence_summary["initial_mean_distance"] = mean_distances[0]
        divergence_summary["final_mean_distance"] = mean_distances[-1]
        divergence_summary["max_mean_distance"] = round(max(mean_distances), 6)
        divergence_summary["min_mean_distance"] = round(min(mean_distances), 6)
        divergence_summary["mean_of_mean_distances"] = round(
            sum(mean_distances) / len(mean_distances), 6
        )
        if len(mean_distances) >= 2:
            divergence_summary["distance_change"] = round(
                mean_distances[-1] - mean_distances[0], 6
            )

    return {
        "metadata": {
            "mode": "divergence",
            "scenario": scenario_label,
            "num_instances": num_instances,
            "warmup_turns": warmup_turns,
            "main_turns": len(sequence),
            "delta_time": delta_time,
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "dimension_labels": _dimension_labels(),
        },
        "warmup_states": warmup_records,
        "tick_records": tick_records,
        "divergence_summary": divergence_summary,
    }


# ── Measurement Mode 2: Trajectory Statistical Features ──────

def _compute_variance(values: list[float]) -> float:
    """Compute population variance."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _compute_lag1_autocorrelation(values: list[float]) -> float:
    """Compute lag-1 autocorrelation of a time series.

    Returns 0.0 if the series has fewer than 3 elements or zero variance.
    """
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var < 1e-15:
        return 0.0
    cov = sum(
        (values[i] - mean) * (values[i + 1] - mean)
        for i in range(n - 1)
    ) / (n - 1)
    return cov / var


def _compute_binned_entropy(values: list[float], num_bins: int) -> float:
    """Compute Shannon entropy of binned values.

    Divides the value range into equal-width bins and computes entropy.
    Returns 0.0 if all values are identical or list is empty.

    Args:
        values: Numeric time series.
        num_bins: Number of equal-width bins.

    Returns:
        Entropy in nats (natural logarithm base).
    """
    n = len(values)
    if n == 0 or num_bins < 1:
        return 0.0
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-15:
        return 0.0
    # Bin width
    bin_width = (v_max - v_min) / num_bins
    # Count per bin
    counts = [0] * num_bins
    for v in values:
        idx = int((v - v_min) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    # Shannon entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    return entropy


def _compute_split_half_stationarity(
    values: list[float],
) -> dict[str, float]:
    """Compute split-half stationarity indicators.

    Splits the series into first half and second half, and records the
    difference in mean and variance between the two halves.

    No binary "stationary/non-stationary" judgment is made.

    Returns:
        Dict with mean_diff, variance_diff, first_half_mean,
        second_half_mean, first_half_variance, second_half_variance.
    """
    n = len(values)
    if n < 2:
        return {
            "mean_diff": 0.0,
            "variance_diff": 0.0,
            "first_half_mean": 0.0,
            "second_half_mean": 0.0,
            "first_half_variance": 0.0,
            "second_half_variance": 0.0,
        }
    mid = n // 2
    first_half = values[:mid]
    second_half = values[mid:]

    fh_mean = sum(first_half) / len(first_half)
    sh_mean = sum(second_half) / len(second_half)
    fh_var = _compute_variance(first_half)
    sh_var = _compute_variance(second_half)

    return {
        "mean_diff": round(sh_mean - fh_mean, 6),
        "variance_diff": round(sh_var - fh_var, 6),
        "first_half_mean": round(fh_mean, 6),
        "second_half_mean": round(sh_mean, 6),
        "first_half_variance": round(fh_var, 6),
        "second_half_variance": round(sh_var, 6),
    }


def _compute_effective_dimensionality(
    vectors: list[list[float]],
) -> float:
    """Compute effective dimensionality from variance contribution ratios.

    Uses the proportion of variance explained by each dimension to compute
    a participation ratio, which corresponds to the effective number of
    dimensions being used.

    Effective dimensionality = (sum of variances)^2 / sum of (variances^2)
    This is the participation ratio of the variance distribution.

    Returns:
        Effective dimensionality (float, range 1.0 to STATE_VECTOR_DIM).
        Returns 0.0 if input is empty.
    """
    if not vectors:
        return 0.0
    n = len(vectors)
    dim = len(vectors[0])
    if n < 2 or dim == 0:
        return 0.0

    # Per-dimension variance
    variances: list[float] = []
    for d in range(dim):
        vals = [v[d] for v in vectors]
        variances.append(_compute_variance(vals))

    total_var = sum(variances)
    if total_var < 1e-15:
        return 0.0

    # Participation ratio
    sum_var_sq = sum(v ** 2 for v in variances)
    if sum_var_sq < 1e-15:
        return 0.0

    return (total_var ** 2) / sum_var_sq


def _compute_transition_frequency(
    labels: list[str],
) -> dict[str, Any]:
    """Count the number of transitions (label changes) in a sequence.

    Returns:
        Dict with transition_count and unique_labels.
    """
    if len(labels) < 2:
        return {"transition_count": 0, "unique_labels": len(set(labels))}
    transitions = sum(
        1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1]
    )
    return {
        "transition_count": transitions,
        "unique_labels": len(set(labels)),
    }


def compute_trajectory_features(
    result: dict[str, Any],
    entropy_bin_params: list[int] | None = None,
) -> dict[str, Any]:
    """Compute statistical features of the state trajectory.

    This is a pure function on existing turn record data. No additional
    state is held. Extends the existing statistics with structural
    features of the trajectory.

    No evaluation judgments (good/bad, normal/abnormal) are included.

    Args:
        result: Output from run_simulation().
        entropy_bin_params: List of bin counts for entropy calculation.
            If None, defaults to [5, 10, 20].

    Returns:
        Trajectory feature dict.
    """
    if entropy_bin_params is None:
        entropy_bin_params = [5, 10, 20]

    turns = result.get("turns", [])
    if not turns:
        return {
            "scenario": result.get("metadata", {}).get("scenario", "unknown"),
            "total_turns": 0,
        }

    # Extract state vectors from all turns
    vectors = [_extract_state_vector_from_record(t) for t in turns]
    labels = _dimension_labels()

    features: dict[str, Any] = {
        "scenario": result["metadata"]["scenario"],
        "total_turns": len(turns),
        "dimension_labels": labels,
    }

    # Per-dimension features
    per_dim: dict[str, dict[str, Any]] = {}
    for d, label in enumerate(labels):
        dim_values = [v[d] for v in vectors]
        dim_features: dict[str, Any] = {}

        # Variance
        dim_features["variance"] = round(_compute_variance(dim_values), 6)

        # Lag-1 autocorrelation
        dim_features["lag1_autocorrelation"] = round(
            _compute_lag1_autocorrelation(dim_values), 6
        )

        # Binned entropy (multiple bin parameters)
        dim_features["entropy"] = {}
        for num_bins in entropy_bin_params:
            dim_features["entropy"][f"bins_{num_bins}"] = round(
                _compute_binned_entropy(dim_values, num_bins), 6
            )

        # Split-half stationarity
        dim_features["stationarity"] = _compute_split_half_stationarity(dim_values)

        per_dim[label] = dim_features

    features["per_dimension"] = per_dim

    # Full trajectory features
    features["effective_dimensionality"] = round(
        _compute_effective_dimensionality(vectors), 6
    )

    # Transition frequency: dominant emotion and policy label
    dominant_emotions = [t["psyche_state"]["dominant_emotion"] for t in turns]
    features["dominant_emotion_transitions"] = _compute_transition_frequency(
        dominant_emotions
    )

    policy_labels = [t["policy"]["policy_label"] for t in turns]
    features["policy_label_transitions"] = _compute_transition_frequency(
        policy_labels
    )

    return features


# ── Measurement Mode 3: A/B Coefficient Comparison ───────────

def _run_sim_for_ab(
    sequence: list[str],
    delta_time: float,
    user_id: str,
    bypass_session_modulation: bool = False,
) -> dict[str, Any]:
    """Run a single simulation for A/B comparison.

    When bypass_session_modulation is True, the session boundary
    coefficient modulations are skipped by not calling load() on
    the orchestrator instance (cold start without modulations).

    When False, a save+load cycle is performed to trigger session
    boundary modulations (apply_fifo_experience_expansion,
    apply_contradiction_amplitude_modulation, session_decay).

    Args:
        sequence: Pattern key list.
        delta_time: Virtual seconds between turns.
        user_id: Simulated user ID.
        bypass_session_modulation: If True, skip session boundary
            coefficient modulation.

    Returns:
        Simulation result dict (same structure as run_simulation).
    """
    # Phase 1: Create instance with some state (warmup)
    tmpdir = tempfile.mkdtemp(prefix="psyche_ab_")
    data_dir = Path(tmpdir)
    orch = PsycheOrchestrator(memory_count=0, data_dir=data_dir)

    # Minimal warmup to establish non-trivial state for session boundary effects
    warmup_keys = ["positive", "negative", "neutral", "confused", "angry"]
    for key in warmup_keys:
        percept = Percept(**INPUT_PATTERNS[key])
        orch.post_response_update(percept, delta_time, user_id)
        policy = orch.select_policy_dict(percept, [], user_id)
        outcome = _pattern_to_outcome(key)
        resp_state = orch._responsibility_mgr.get_state(user_id)
        for rec in reversed(resp_state.recent_decisions):
            if not rec.get("evaluated", False):
                orch._responsibility_mgr.evaluate_outcome(
                    user_id, rec["id"], outcome,
                )
                break

    if not bypass_session_modulation:
        # Condition A: save and load to trigger session boundary modulations
        orch.save()
        orch2 = PsycheOrchestrator(memory_count=0, data_dir=data_dir)
        orch2.load()
        orch = orch2
    # Condition B (bypass_session_modulation=True): continue with the
    # same instance — no save/load, so no session boundary modulations.

    # Phase 2: Run main sequence
    started_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    sim_memory_count = 0

    for turn_idx, pattern_key in enumerate(sequence):
        turn_num = turn_idx + 1
        percept = Percept(**INPUT_PATTERNS[pattern_key])
        orch.post_response_update(percept, delta_time, user_id)

        if abs(percept.emotion_valence) > 0.3:
            sim_memory_count += 1
            orch.on_memory_saved(
                summary=percept.text,
                keywords=[percept.emotion or "unknown"],
                memory_count=sim_memory_count,
            )

        policy = orch.select_policy_dict(percept, [], user_id)
        outcome = _pattern_to_outcome(pattern_key)
        resp_state = orch._responsibility_mgr.get_state(user_id)
        for rec in reversed(resp_state.recent_decisions):
            if not rec.get("evaluated", False):
                orch._responsibility_mgr.evaluate_outcome(
                    user_id, rec["id"], outcome,
                )
                break

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
            "scenario": "ab_condition",
            "total_turns": len(sequence),
            "delta_time_per_turn": delta_time,
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "bypass_session_modulation": bypass_session_modulation,
            "version": 3,
        },
        "turns": records,
    }


def _compute_per_dimension_effect_size(
    vectors_a: list[list[float]],
    vectors_b: list[list[float]],
) -> list[dict[str, Any]]:
    """Compute standardized effect size per dimension for A/B comparison.

    For each dimension, computes:
    - mean_a, mean_b: Mean of each condition
    - diff: mean_a - mean_b
    - pooled_stddev: sqrt((var_a + var_b) / 2)
    - effect_size: diff / pooled_stddev (Cohen's d equivalent)

    No thresholds or evaluation ("small/medium/large") are applied.

    Returns:
        List of per-dimension effect size records.
    """
    if not vectors_a or not vectors_b:
        return []

    dim = len(vectors_a[0])
    labels = _dimension_labels()
    results: list[dict[str, Any]] = []

    for d in range(dim):
        vals_a = [v[d] for v in vectors_a]
        vals_b = [v[d] for v in vectors_b]

        mean_a = sum(vals_a) / len(vals_a)
        mean_b = sum(vals_b) / len(vals_b)
        var_a = _compute_variance(vals_a)
        var_b = _compute_variance(vals_b)
        pooled_var = (var_a + var_b) / 2.0
        pooled_sd = math.sqrt(pooled_var) if pooled_var > 1e-15 else 0.0

        diff = mean_a - mean_b
        effect_size = diff / pooled_sd if pooled_sd > 1e-15 else 0.0

        results.append({
            "dimension": labels[d] if d < len(labels) else f"dim_{d}",
            "mean_a": round(mean_a, 6),
            "mean_b": round(mean_b, 6),
            "diff": round(diff, 6),
            "pooled_stddev": round(pooled_sd, 6),
            "effect_size": round(effect_size, 6),
        })

    return results


def run_ab_comparison(
    scenario_name: str | None = None,
    custom_sequence: list[str] | None = None,
    delta_time: float = 2.0,
    user_id: str = "sim_user",
) -> dict[str, Any]:
    """Run A/B comparison of session boundary coefficient modulation.

    Condition A: Session boundary modulations enabled (save + load cycle).
    Condition B: Session boundary modulations disabled (no save/load).

    Both conditions start from the same warmup state and receive the same
    input sequence.

    No evaluation judgment (which condition is "better") is made.

    Args:
        scenario_name: SCENARIOS name for the input sequence.
        custom_sequence: Custom pattern key list (overrides scenario_name).
        delta_time: Virtual seconds between turns.
        user_id: Simulated user ID.

    Returns:
        A/B comparison result dict.

    Raises:
        ValueError: Invalid scenario or pattern keys.
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

    for key in sequence:
        if key not in INPUT_PATTERNS:
            raise ValueError(f"Invalid pattern key: {key}")

    started_at = datetime.now().isoformat(timespec="seconds")

    # Run both conditions
    result_a = _run_sim_for_ab(
        sequence, delta_time, user_id,
        bypass_session_modulation=False,
    )
    result_b = _run_sim_for_ab(
        sequence, delta_time, user_id,
        bypass_session_modulation=True,
    )

    # Extract state vectors
    vectors_a = [
        _extract_state_vector_from_record(t) for t in result_a["turns"]
    ]
    vectors_b = [
        _extract_state_vector_from_record(t) for t in result_b["turns"]
    ]

    # Per-tick distance between A and B
    per_tick_distances: list[dict[str, Any]] = []
    for idx in range(min(len(vectors_a), len(vectors_b))):
        dist = _euclidean_distance(vectors_a[idx], vectors_b[idx])
        per_tick_distances.append({
            "turn": idx + 1,
            "distance": round(dist, 6),
        })

    # Per-dimension effect sizes
    effect_sizes = _compute_per_dimension_effect_size(vectors_a, vectors_b)

    # Summary statistics of distances
    all_dists = [d["distance"] for d in per_tick_distances]
    distance_summary: dict[str, Any] = {}
    if all_dists:
        distance_summary["min"] = round(min(all_dists), 6)
        distance_summary["max"] = round(max(all_dists), 6)
        distance_summary["mean"] = round(sum(all_dists) / len(all_dists), 6)
        distance_summary["stddev"] = round(_safe_stddev(all_dists), 6)

    finished_at = datetime.now().isoformat(timespec="seconds")

    return {
        "metadata": {
            "mode": "ab_comparison",
            "scenario": scenario_label,
            "total_turns": len(sequence),
            "delta_time": delta_time,
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "condition_a": "session_modulation_enabled",
            "condition_b": "session_modulation_disabled",
            "dimension_labels": _dimension_labels(),
        },
        "condition_a": result_a,
        "condition_b": result_b,
        "per_tick_distances": per_tick_distances,
        "effect_sizes": effect_sizes,
        "distance_summary": distance_summary,
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
    # ── Measurement mode arguments ──
    parser.add_argument(
        "--divergence",
        type=str,
        default=None,
        help="Run state divergence measurement for given scenario",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=3,
        help="Number of instances for divergence measurement (default: 3)",
    )
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=5,
        help="Number of warmup turns per instance for divergence (default: 5)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Upper bound for num-instances (default: 10)",
    )
    parser.add_argument(
        "--trajectory-features",
        action="store_true",
        help="Include trajectory statistical features in output",
    )
    parser.add_argument(
        "--entropy-bins",
        nargs="+",
        type=int,
        default=None,
        help="Entropy bin parameters (default: 5 10 20)",
    )
    parser.add_argument(
        "--ab-compare",
        type=str,
        default=None,
        help="Run A/B comparison (session modulation enabled vs disabled)",
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

    # ── Divergence measurement mode ──
    if args.divergence:
        div_result = run_divergence_measurement(
            scenario_name=args.divergence,
            num_instances=args.num_instances,
            warmup_turns=args.warmup_turns,
            delta_time=args.delta_time,
            user_id=args.user_id,
            max_instances=args.max_instances,
        )
        output_text = json.dumps(div_result, ensure_ascii=False, indent=2)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")
            print(f"Output written to {out_path}")
        else:
            print(output_text)
        return

    # ── A/B comparison mode ──
    if args.ab_compare:
        ab_result = run_ab_comparison(
            scenario_name=args.ab_compare,
            delta_time=args.delta_time,
            user_id=args.user_id,
        )
        output_text = json.dumps(ab_result, ensure_ascii=False, indent=2)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")
            print(f"Output written to {out_path}")
        else:
            print(output_text)
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

    # 軌跡統計特徴量付加
    if args.trajectory_features:
        output_data["trajectory_features"] = compute_trajectory_features(
            result, entropy_bin_params=args.entropy_bins,
        )

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
