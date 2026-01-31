"""
psyche/thought.py - Policy Decision Module (LOCAL ONLY).

Generates and scores response policy candidates using **only local logic**.
No LLM calls.  Drives, fear_index, mood, percept, and **responsibility** determine the policy.

責任（Responsibility）の影響:
- caution_bias: 判断時の慎重さ → からかう等のリスキーな選択を抑制
- empathy_bias: 共感へのバイアス → 寄り添う選択を促進

Usage::

    candidates = generate_thought_candidates(state, percept, recalled, responsibility_influence)
    policy = select_policy(candidates, state, responsibility_influence)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .state import Percept, PsycheState
from .responsibility import ResponsibilityInfluence

logger = logging.getLogger(__name__)

# ── Policy definitions ─────────────────────────────────────────

POLICIES: list[dict[str, Any]] = [
    {
        "policy_label": "共感する",
        "rationale_template": "相手の気持ちに寄り添う",
        "drive_target": "social",
        "expected_drive_change": {"social": -0.08, "curiosity": -0.02, "expression": -0.02},
    },
    {
        "policy_label": "質問で会話を広げる",
        "rationale_template": "好奇心を満たし、会話を続ける",
        "drive_target": "curiosity",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.10, "expression": -0.01},
    },
    {
        "policy_label": "からかう",
        "rationale_template": "楽しさや親密さを表現する",
        "drive_target": "expression",
        "expected_drive_change": {"social": -0.05, "curiosity": -0.01, "expression": -0.08},
    },
    {
        "policy_label": "話題を変える",
        "rationale_template": "新しい方向に会話を導く",
        "drive_target": "curiosity",
        "expected_drive_change": {"social": -0.02, "curiosity": -0.08, "expression": -0.03},
    },
    {
        "policy_label": "感想を述べる",
        "rationale_template": "自分の考えや感情を表現する",
        "drive_target": "expression",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.02, "expression": -0.10},
    },
    {
        "policy_label": "励ます",
        "rationale_template": "相手を元気づける",
        "drive_target": "social",
        "expected_drive_change": {"social": -0.10, "curiosity": -0.01, "expression": -0.03},
    },
]

# ── Fallback text per policy ──────────────────────────────────

_FALLBACK_TEXT: dict[str, str] = {
    "共感する": "...あなたの気持ち、わかるわ",
    "質問で会話を広げる": "ねえ、もっと聞かせて？",
    "からかう": "ふふっ、面白いわね♪",
    "話題を変える": "そういえば、ほかに何かあった？",
    "感想を述べる": "あたしはね...そう思うの",
    "励ます": "大丈夫、あたしがそばにいるから♪",
}


# ── Public API (LOCAL ONLY — no LLM) ──────────────────────────

def generate_thought_candidates(
    state: PsycheState,
    percept: Percept,
    recalled: list[dict],
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
) -> list[dict]:
    """Generate response policy candidates from local state analysis.

    **No LLM calls.**  All logic is deterministic from state + percept.
    責任の影響がある場合、判断に慎重さと共感バイアスが加わる。

    Args:
        state: Current psychological state.
        percept: Interpreted input stimulus.
        recalled: Retrieved memories.
        responsibility_influence: Optional responsibility influence on decisions.

    Returns a list of candidate dicts, each with:
      - policy_label, rationale, expected_drive_change, text
    """
    candidates: list[dict] = []

    for policy_def in POLICIES:
        label = policy_def["policy_label"]
        score = _score_candidate(policy_def, state, percept, recalled, responsibility_influence)
        candidates.append({
            "policy_label": label,
            "rationale": policy_def["rationale_template"],
            "expected_drive_change": dict(policy_def["expected_drive_change"]),
            "text": _FALLBACK_TEXT.get(label, "..."),
            "_score": score,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    # Return top 3 candidates
    return candidates[:3]


def select_policy(
    candidates: list[dict],
    state: PsycheState,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
) -> dict:
    """Select the best policy from candidates.  LOCAL scoring only.

    責任の影響がある場合、最終選択にも慎重さが反映される。

    Returns the top-scoring candidate dict.
    """
    if not candidates:
        return {
            "policy_label": "共感する",
            "rationale": "デフォルト方針",
            "expected_drive_change": {"social": -0.05, "curiosity": -0.02, "expression": -0.02},
            "text": _FALLBACK_TEXT["共感する"],
        }

    # Already sorted by generate_thought_candidates
    best = candidates[0]

    # 責任による最終チェック: 非常に慎重な状態ではリスキーな選択を再考
    if responsibility_influence and responsibility_influence.caution_bias > 0.3:
        if best["policy_label"] == "からかう" and len(candidates) > 1:
            # からかうは避けて次善策を選ぶ
            logger.info(
                "Responsibility caution override: %s → %s (caution=%.2f)",
                best["policy_label"], candidates[1]["policy_label"],
                responsibility_influence.caution_bias
            )
            best = candidates[1]

    logger.info("Policy selected (LOCAL): %s (score=%.2f)", best["policy_label"], best.get("_score", 0))
    return best


# ── Internal scoring (deterministic) ──────────────────────────

def _score_candidate(
    policy_def: dict,
    state: PsycheState,
    percept: Percept,
    recalled: list[dict],
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
) -> float:
    """Score a candidate policy against current state.  Pure function.

    責任の影響がある場合:
    - caution_bias: リスキーな選択（からかう）にペナルティ
    - empathy_bias: 共感・励ましにボーナス
    """
    score = 0.0
    drives = state.drives
    fear = state.fear_level
    mood_val = state.mood.valence
    label = policy_def["policy_label"]

    # 1. Drive satisfaction: high drive + policy satisfies it → bonus
    target_drive = policy_def["drive_target"]
    drive_val = getattr(drives, target_drive, 0.5)
    if drive_val > 0.6:
        score += drive_val * 2.0
    elif drive_val > 0.4:
        score += drive_val * 1.0

    # 2. Fear bias: high fear → prefer 共感/励ます, penalise からかう
    if fear > 0.3:
        if label in ("共感する", "励ます"):
            score += fear * 3.0
        elif label == "からかう":
            score -= fear * 2.0

    # 3. Mood alignment
    if mood_val < -0.3:
        # Negative mood → prefer empathy/encouragement
        if label in ("共感する", "励ます"):
            score += abs(mood_val) * 2.0
        elif label == "からかう":
            score -= 1.0
    elif mood_val > 0.3:
        # Positive mood → teasing/sharing is fine
        if label in ("からかう", "感想を述べる"):
            score += mood_val * 1.5

    # 4. Percept intent matching
    intent = percept.intent
    if intent == "question":
        if label == "質問で会話を広げる":
            score += 1.5
    elif intent in ("sharing", "complaint"):
        if label == "共感する":
            score += 1.5
    elif intent == "greeting":
        if label in ("感想を述べる", "質問で会話を広げる"):
            score += 1.0
    elif intent == "joke":
        if label == "からかう":
            score += 2.0

    # 5. Percept valence
    v = percept.emotion_valence
    if v < -0.3 and label in ("共感する", "励ます"):
        score += abs(v) * 1.5
    elif v > 0.3 and label in ("からかう", "感想を述べる"):
        score += v * 1.0

    # 6. Attachment fear bonus
    if state.fear_index and state.fear_index.attachment_risk > 0.4:
        if label in ("共感する", "質問で会話を広げる"):
            score += state.fear_index.attachment_risk * 2.0

    # 7. Identity fear → self-expression
    if state.fear_index and state.fear_index.identity_risk > 0.4:
        if label == "感想を述べる":
            score += state.fear_index.identity_risk * 1.5

    # 8. Memory context: if recalled memories relate, slight boost to empathy
    if recalled:
        score += 0.3

    # 9. 責任（Responsibility）による影響
    # 過去の判断の重みが、現在の選択を歪める
    if responsibility_influence:
        # 慎重さバイアス: リスキーな選択にペナルティ
        caution = responsibility_influence.caution_bias
        if caution > 0.1:
            if label == "からかう":
                score -= caution * 4.0  # からかうは傷つけるリスクがある
            elif label == "話題を変える":
                score -= caution * 1.5  # 逃げと見なされるリスク

        # 共感バイアス: 寄り添う選択にボーナス
        empathy = responsibility_influence.empathy_bias
        if empathy > 0.1:
            if label in ("共感する", "励ます"):
                score += empathy * 3.0  # 傷つけないよう寄り添う
            elif label == "質問で会話を広げる":
                score += empathy * 1.5  # 相手に寄り添う姿勢

        # 不安ベースライン: 全体的に慎重な選択を促す
        anxiety = responsibility_influence.anxiety_baseline
        if anxiety > 0.1:
            if label in ("共感する", "励ます", "質問で会話を広げる"):
                score += anxiety * 2.0

    return score
