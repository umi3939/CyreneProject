"""
psyche/thought.py - Policy Decision Module (LOCAL ONLY).

Generates and scores response policy candidates using **only local logic**.
No LLM calls.  Drives, fear_index, mood, percept, and **responsibility** determine the policy.

責任（Responsibility）の影響:
- caution_bias: 判断時の慎重さ → 特定ポリシーへのスコアペナルティ
- empathy_bias: 共感へのバイアス → 特定ポリシーへのスコアボーナス

短期記憶・ダイナミクス由来のバイアス:
- DecisionBias: 短期記憶の余韻とピーク/反動状態が判断スコアに影響
- バイアスは一時的で、時間経過により自然に減衰する

Usage::

    candidates = generate_thought_candidates(state, percept, recalled, responsibility_influence)
    policy = select_policy(candidates, state, responsibility_influence)

    # With decision bias from STM/dynamics:
    candidates = generate_thought_candidates(
        state, percept, recalled, responsibility_influence, decision_bias
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .state import Percept, PsycheState
from .responsibility import ResponsibilityInfluence
from .decision_bias import DecisionBias, apply_bias_to_score

logger = logging.getLogger(__name__)

# ── スコアリング断面別帯域上限 ──
# 全断面で同一の上限値を使用（断面間の均衡化）
_SCORE_SECTION_BAND: float = 1.5

# ── Policy definitions ─────────────────────────────────────────

POLICIES: list[dict[str, Any]] = [
    {
        "policy_label": "共感する",
        "rationale_template": "相手の気持ちに寄り添う",
        "drive_target": "social",
        "expected_drive_change": {"social": -0.08, "curiosity": -0.01, "expression": -0.02},
    },
    {
        "policy_label": "質問で会話を広げる",
        "rationale_template": "好奇心を満たし、会話を続ける",
        "drive_target": "curiosity",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.06, "expression": -0.01},
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
        "expected_drive_change": {"social": -0.02, "curiosity": -0.05, "expression": -0.03},
    },
    {
        "policy_label": "感想を述べる",
        "rationale_template": "自分の考えや感情を表現する",
        "drive_target": "expression",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.01, "expression": -0.10},
    },
    {
        "policy_label": "励ます",
        "rationale_template": "相手を元気づける",
        "drive_target": "social",
        "expected_drive_change": {"social": -0.10, "curiosity": -0.01, "expression": -0.03},
    },
    # ── 追加9件: ポリシー候補構造の動的化 ──
    {
        "policy_label": "黙って聞く",
        "rationale_template": "相手の発話を受け止める",
        "drive_target": "safety",
        "expected_drive_change": {"social": -0.04, "curiosity": -0.01, "expression": -0.01},
    },
    {
        "policy_label": "自分の経験を話す",
        "rationale_template": "自身の記憶を参照して提示する",
        "drive_target": "expression",
        "expected_drive_change": {"social": -0.04, "curiosity": -0.01, "expression": -0.09},
    },
    {
        "policy_label": "確認する",
        "rationale_template": "相手の意図の理解を照合する",
        "drive_target": "safety",
        "expected_drive_change": {"social": -0.05, "curiosity": -0.03, "expression": -0.02},
    },
    {
        "policy_label": "冗談を言う",
        "rationale_template": "場の空気を変える発話を試みる",
        "drive_target": "expression",
        "expected_drive_change": {"social": -0.04, "curiosity": -0.01, "expression": -0.09},
    },
    {
        "policy_label": "謝る",
        "rationale_template": "過去の行動の不適切さを認める",
        "drive_target": "safety",
        "expected_drive_change": {"social": -0.07, "curiosity": -0.01, "expression": -0.04},
    },
    {
        "policy_label": "提案する",
        "rationale_template": "相手の状況に対して選択肢を提示する",
        "drive_target": "curiosity",
        "expected_drive_change": {"social": -0.04, "curiosity": -0.05, "expression": -0.03},
    },
    {
        "policy_label": "見守る",
        "rationale_template": "状況の推移を能動的に観察しつつ介入しない",
        "drive_target": "autonomy",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.01, "expression": -0.01},
    },
    {
        "policy_label": "同意する",
        "rationale_template": "相手の見解に同調を示す",
        "drive_target": "social",
        "expected_drive_change": {"social": -0.07, "curiosity": -0.01, "expression": -0.03},
    },
    {
        "policy_label": "反論する",
        "rationale_template": "相手の見解に対して異なる視点を提示する",
        "drive_target": "autonomy",
        "expected_drive_change": {"social": -0.03, "curiosity": -0.01, "expression": -0.07},
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
    # ── 追加9件 ──
    "黙って聞く": "...うん、聞いてるよ",
    "自分の経験を話す": "...あたしもね、こういうことがあってさ",
    "確認する": "...ちょっと確認なんだけど、こういうこと？",
    "冗談を言う": "...ふふ、ちょっと思いついちゃった",
    "謝る": "...ごめんね、さっきのは良くなかったかも",
    "提案する": "...こういうのはどう？",
    "見守る": "...うん、見てるからね",
    "同意する": "...そうだよね、あたしもそう思う",
    "反論する": "...でもさ、こういう見方もあると思うの",
}


# ── Public API (LOCAL ONLY — no LLM) ──────────────────────────

def generate_thought_candidates(
    state: PsycheState,
    percept: Percept,
    recalled: list[dict],
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    extended_inputs: Optional[dict] = None,
    collect_breakdown: bool = False,
    score_section_band_addition: Optional[float] = None,
) -> list[dict]:
    """Generate response policy candidates from local state analysis.

    **No LLM calls.**  All logic is deterministic from state + percept.
    責任の影響がある場合、判断に慎重さと共感バイアスが加わる。
    短期記憶/ダイナミクス由来のバイアスがある場合、スコアに微調整が加わる。
    extended_inputsがある場合、自己観測・傾向・他者推定・目的断面の
    情報がスコアリングに追加反映される。

    collect_breakdown=True の場合、各候補に "_score_breakdown" フィールドを
    オプショナルに付帯する。このフィールドは断面名→寄与量の辞書であり、
    返却値の基本構造には影響しない。

    Args:
        state: Current psychological state.
        percept: Interpreted input stimulus.
        recalled: Retrieved memories.
        responsibility_influence: Optional responsibility influence on decisions.
        decision_bias: Optional bias from short-term memory and dynamics.
        extended_inputs: Optional dict of cross-section inputs for extended scoring.
        collect_breakdown: If True, attach per-section score breakdown to each candidate.

    Returns a list of candidate dicts, each with:
      - policy_label, rationale, expected_drive_change, text
    """
    candidates: list[dict] = []

    for policy_def in POLICIES:
        label = policy_def["policy_label"]
        result = _score_candidate(
            policy_def, state, percept, recalled,
            responsibility_influence, decision_bias, extended_inputs,
            collect_breakdown=collect_breakdown,
            score_section_band_addition=score_section_band_addition,
        )
        if collect_breakdown:
            score, breakdown = result
        else:
            score = result
            breakdown = None
        entry: dict = {
            "policy_label": label,
            "rationale": policy_def["rationale_template"],
            "expected_drive_change": dict(policy_def["expected_drive_change"]),
            "text": _FALLBACK_TEXT.get(label, "..."),
            "_score": score,
        }
        if breakdown is not None:
            entry["_score_breakdown"] = breakdown
        candidates.append(entry)

    # Sort by score descending
    candidates.sort(key=lambda c: c["_score"], reverse=True)

    # ── 安全弁: スコア差過大時の非線形圧縮 ──
    if len(candidates) >= 2:
        gap = candidates[0]["_score"] - candidates[1]["_score"]
        if gap > 1.0:
            # compression_factor: gapが大きいほど圧縮が強い (1/(1+gap*0.3))
            compression_factor = 1.0 / (1.0 + gap * 0.3)
            compressed_score = candidates[1]["_score"] + gap * compression_factor
            candidates[0]["_score"] = compressed_score
            # 再ソート
            candidates.sort(key=lambda c: c["_score"], reverse=True)

    # ── 動的選出: スコア差に基づいて3-5件 ──
    MIN_SELECT = 3
    MAX_SELECT = 5
    if len(candidates) <= MIN_SELECT:
        return candidates

    top_score = candidates[0]["_score"]
    selected = candidates[:MIN_SELECT]
    for i in range(MIN_SELECT, min(MAX_SELECT, len(candidates))):
        gap = top_score - candidates[i]["_score"]
        # スコア差がtop_scoreの30%以内なら追加
        if top_score > 0 and gap / max(top_score, 0.01) < 0.3:
            selected.append(candidates[i])
        else:
            break

    return selected


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

    logger.debug("Policy selected (LOCAL): %s (score=%.2f)", best["policy_label"], best.get("_score", 0))
    return best


# ── Internal scoring (deterministic) ──────────────────────────

def _score_candidate(
    policy_def: dict,
    state: PsycheState,
    percept: Percept,
    recalled: list[dict],
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    extended_inputs: Optional[dict] = None,
    collect_breakdown: bool = False,
    score_section_band_addition: Optional[float] = None,
) -> float | tuple[float, dict[str, float]]:
    """Score a candidate policy against current state.  Pure function.

    責任の影響がある場合:
    - caution_bias: 特定ポリシーにスコアペナルティ
    - empathy_bias: 特定ポリシーにスコアボーナス

    短期記憶/ダイナミクス由来のバイアスがある場合:
    - 直近の感情の余韻がスコアに影響
    - ピーク/反動状態が選択傾向を微調整

    extended_inputsがある場合:
    - 自己観測・傾向・他者推定・目的断面の情報がスコアリングに追加反映

    collect_breakdown=True の場合、(score, breakdown_dict) のタプルを返す。
    breakdown_dict は断面名→寄与量の辞書。
    """
    score = 0.0
    # 断面別寄与量の一時辞書（collect_breakdown=True の場合のみ使用）
    bd: dict[str, float] = {} if collect_breakdown else {}
    drives = state.drives
    fear = state.fear_level
    mood_val = state.mood.valence
    label = policy_def["policy_label"]

    # 帯域制限ヘルパー: 各断面の寄与を均一上限でクランプ
    # Phase 26-EXP 帯域拡大: 加算量が指定されている場合は一時的に上限を拡大
    effective_band = _SCORE_SECTION_BAND
    if score_section_band_addition is not None:
        effective_band = _SCORE_SECTION_BAND + score_section_band_addition

    def _clamp_section(v: float) -> float:
        return max(-effective_band, min(effective_band, v))

    # 1. Drive satisfaction: high drive + policy satisfies it → bonus
    target_drive = policy_def["drive_target"]
    if target_drive in ("social", "curiosity", "expression"):
        # 既存3軸: 既存のまま
        drive_val = getattr(drives, target_drive, 0.5)
    elif target_drive == "safety":
        # 新軸 safety: fear + caution_bias + (1 - self_image_stability) の平均値
        caution_bias = 0.0
        if responsibility_influence:
            caution_bias = responsibility_influence.caution_bias
        self_image_stability = 0.5
        if extended_inputs:
            self_image_stability = extended_inputs.get("self_image_stability", 0.5)
        drive_val = (fear + caution_bias + (1.0 - self_image_stability)) / 3.0
    elif target_drive == "autonomy":
        # 新軸 autonomy: tendency_strength + goal_strength + coherence_level の平均値
        t_str = 0.0
        g_str = 0.0
        c_lvl = 0.5
        if extended_inputs:
            t_str = extended_inputs.get("tendency_strength", 0.0)
            g_str = extended_inputs.get("goal_strength", 0.0)
            c_lvl = extended_inputs.get("coherence_level", 0.5)
        drive_val = (t_str + g_str + c_lvl) / 3.0
    else:
        drive_val = 0.5

    drive_contrib = 0.0
    if drive_val > 0.6:
        drive_contrib = drive_val * 2.0
    elif drive_val > 0.4:
        drive_contrib = drive_val * 1.0
    drive_contrib = _clamp_section(drive_contrib)
    score += drive_contrib
    if collect_breakdown:
        bd["drive_goal_match"] = drive_contrib

    # 2. Fear bias: high fear → prefer 共感/励ます, penalise からかう
    fear_contrib = 0.0
    if fear > 0.3:
        if label in ("共感する", "励ます"):
            fear_contrib = fear * 3.0
        elif label == "からかう":
            fear_contrib = -(fear * 2.0)
    fear_contrib = _clamp_section(fear_contrib)
    score += fear_contrib
    if collect_breakdown:
        bd["fear_bias"] = fear_contrib

    # 3. Mood alignment
    mood_contrib = 0.0
    if mood_val < -0.3:
        # Negative mood → prefer empathy/encouragement
        if label in ("共感する", "励ます"):
            mood_contrib = abs(mood_val) * 2.0
        elif label == "からかう":
            mood_contrib = -1.0
    elif mood_val > 0.3:
        # Positive mood → teasing/sharing is fine
        if label in ("からかう", "感想を述べる"):
            mood_contrib = mood_val * 1.5
    mood_contrib = _clamp_section(mood_contrib)
    score += mood_contrib
    if collect_breakdown:
        bd["mood_alignment"] = mood_contrib

    # 4. Percept intent matching
    intent_contrib = 0.0
    intent = percept.intent
    if intent == "question":
        if label == "質問で会話を広げる":
            intent_contrib = 1.5
    elif intent in ("sharing", "complaint"):
        if label == "共感する":
            intent_contrib = 1.5
    elif intent == "greeting":
        if label in ("感想を述べる", "質問で会話を広げる"):
            intent_contrib = 1.0
    elif intent == "joke":
        if label == "からかう":
            intent_contrib = 2.0
    elif intent == "expression":
        # 表現入力: 複数ポリシーに等価な微小寄与
        if label in ("感想を述べる", "共感する", "自分の経験を話す"):
            intent_contrib = 0.3
    intent_contrib = _clamp_section(intent_contrib)
    score += intent_contrib
    if collect_breakdown:
        bd["percept_intent_match"] = intent_contrib

    # 5. Percept valence
    valence_contrib = 0.0
    v = percept.emotion_valence
    if v < -0.3 and label in ("共感する", "励ます"):
        valence_contrib = abs(v) * 1.5
    elif v > 0.3 and label in ("からかう", "感想を述べる"):
        valence_contrib = v * 1.0
    valence_contrib = _clamp_section(valence_contrib)
    score += valence_contrib
    if collect_breakdown:
        bd["percept_emotion_valence"] = valence_contrib

    # 6. Attachment fear bonus
    attach_contrib = 0.0
    if state.fear_index and state.fear_index.attachment_risk > 0.4:
        if label in ("共感する", "質問で会話を広げる"):
            attach_contrib = state.fear_index.attachment_risk * 2.0
    attach_contrib = _clamp_section(attach_contrib)
    score += attach_contrib
    if collect_breakdown:
        bd["attachment_risk_reaction"] = attach_contrib

    # 7. Identity fear → self-expression
    identity_contrib = 0.0
    if state.fear_index and state.fear_index.identity_risk > 0.4:
        if label == "感想を述べる":
            identity_contrib = state.fear_index.identity_risk * 1.5
    identity_contrib = _clamp_section(identity_contrib)
    score += identity_contrib
    if collect_breakdown:
        bd["identity_risk_reaction"] = identity_contrib

    # 8. Memory context: if recalled memories relate, slight boost to empathy
    memory_contrib = 0.0
    if recalled:
        memory_contrib = 0.3
    score += memory_contrib
    if collect_breakdown:
        bd["memory_context"] = memory_contrib

    # 9. 責任（Responsibility）による影響
    # 過去の判断の重みが、現在の選択を歪める
    resp_contrib = 0.0
    if responsibility_influence:
        # 慎重さバイアス: リスキーな選択にペナルティ
        caution = responsibility_influence.caution_bias
        if caution > 0.1:
            if label == "からかう":
                resp_contrib -= caution * 4.0  # caution_biasが高い状態でのペナルティ
            elif label == "話題を変える":
                resp_contrib -= caution * 1.5  # 逃げと見なされるリスク

        # 共感バイアス: 寄り添う選択にボーナス
        empathy = responsibility_influence.empathy_bias
        if empathy > 0.1:
            if label in ("共感する", "励ます"):
                resp_contrib += empathy * 3.0  # empathy_biasが高い状態でのボーナス
            elif label == "質問で会話を広げる":
                resp_contrib += empathy * 1.5  # 相手に寄り添う姿勢

        # 不安ベースライン: 全体的に慎重な選択を促す
        anxiety = responsibility_influence.anxiety_baseline
        if anxiety > 0.1:
            if label in ("共感する", "励ます", "質問で会話を広げる"):
                resp_contrib += anxiety * 2.0
    resp_contrib = _clamp_section(resp_contrib)
    score += resp_contrib
    if collect_breakdown:
        bd["responsibility_influence"] = resp_contrib

    # 10. 短期記憶/ダイナミクス由来のバイアス（DecisionBias）
    # 直近の体験や感情の余韻によって判断が微妙に傾く
    # バイアスは一時的で、時間経過により自然に減衰する
    stm_bias_contrib = 0.0
    if decision_bias is not None:
        score_before = score
        score = apply_bias_to_score(score, decision_bias, label, policy_def)
        stm_bias_contrib = score - score_before
    if collect_breakdown:
        bd["stm_decision_bias"] = stm_bias_contrib

    # ── 追加条件 #11-16: 内部状態断面の反映 ──
    extended_contrib = 0.0

    # 11. 自己像負荷反応
    if extended_inputs:
        stability = extended_inputs.get("self_image_stability", 0.5)
        strain = extended_inputs.get("strain_level", 0.0)
        if stability < 0.4 or strain > 0.5:
            if label in ("黙って聞く", "確認する", "謝る"):
                extended_contrib += (1.0 - stability + strain) * 1.5

    # 12. 傾向親和性
    if extended_inputs:
        t_strength = extended_inputs.get("tendency_strength", 0.0)
        if t_strength > 0.3:
            if label in ("自分の経験を話す", "感想を述べる"):
                extended_contrib += t_strength * 1.5

    # 13. 他者境界反応
    if extended_inputs:
        o_count = extended_inputs.get("other_count", 0)
        b_clarity = extended_inputs.get("boundary_clarity", 0.5)
        if o_count > 0 and b_clarity < 0.5:
            if label == "確認する":
                extended_contrib += (1.0 - b_clarity) * 2.0
            elif label in ("見守る", "反論する"):
                extended_contrib += (1.0 - b_clarity) * 1.0

    # 14. 目的指向親和性
    if extended_inputs:
        has_goal = extended_inputs.get("has_active_goal", False)
        g_strength = extended_inputs.get("goal_strength", 0.0)
        m_count = extended_inputs.get("motive_count", 0)
        if has_goal and g_strength > 0.3:
            if label in ("提案する", "見守る"):
                extended_contrib += g_strength * 1.5
        if m_count > 2:
            if label in ("提案する", "質問で会話を広げる"):
                extended_contrib += min(m_count * 0.2, 1.0)

    # 15. 一貫性水準反応
    if extended_inputs:
        coherence = extended_inputs.get("coherence_level", 0.5)
        narr_coh = extended_inputs.get("narrative_coherence", 0.5)
        avg_coherence = (coherence + narr_coh) / 2
        if avg_coherence > 0.6:
            if label in ("自分の経験を話す", "感想を述べる", "反論する"):
                extended_contrib += avg_coherence * 1.0
        elif avg_coherence < 0.3:
            if label in ("黙って聞く", "確認する", "同意する"):
                extended_contrib += (1.0 - avg_coherence) * 1.0

    # 16. メタ感情供給反応
    if extended_inputs:
        me_supply = extended_inputs.get("me_supply_strength", 0.0)
        if me_supply > 0.3:
            if label in ("自分の経験を話す", "冗談を言う"):
                extended_contrib += me_supply * 1.5

    extended_contrib = _clamp_section(extended_contrib)
    score += extended_contrib
    if collect_breakdown:
        bd["extended_input_reaction"] = extended_contrib

    if collect_breakdown:
        return score, bd
    return score
