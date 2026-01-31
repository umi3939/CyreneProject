"""
psyche/responsibility.py - 責任（Responsibility）心理的重み機構

責任とは「自分が選択した行動が、他者や関係性に影響を与え、
その結果を引き受け続ける状態」を意味する。

責任は：
- 判断（Policy）を確定した瞬間に発生する
- 結果を観測してから評価される
- 時間をまたいで残留・蓄積・減衰する
- 心理状態に間接的に影響を与える

外部からは以下の機能呼び出しのみ可能：
- record_decision(): 判断を記録する
- evaluate_outcome(): 結果を観測して責任を評価する
- get_influence(): 心理状態への影響を取得する

Usage::

    from psyche.responsibility import (
        record_decision,
        evaluate_outcome,
        get_influence,
        ResponsibilityState,
    )

    # 判断時に記録
    decision_id = record_decision(state, policy, context)

    # 結果観測後に評価
    evaluate_outcome(state, decision_id, outcome)

    # 心理への影響を取得
    influence = get_influence(state)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Data Models ────────────────────────────────────────────────

class DecisionRecord(BaseModel):
    """不変の決定記録（Decision Record）

    判断を確定した瞬間に作成され、変更されない。
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # 判断内容
    policy_label: str = ""
    policy_rationale: str = ""

    # コンテキスト
    target_partner: str = ""  # 影響を与える相手
    emotional_state: str = ""  # 判断時の感情状態
    fear_level: float = 0.0  # 判断時の恐怖レベル

    # 重要度（1-5）
    importance: int = 3

    # 評価済みフラグと責任量
    evaluated: bool = False
    responsibility_delta: float = 0.0

    def fingerprint(self) -> str:
        """判断の一意なフィンガープリント"""
        data = f"{self.timestamp}:{self.policy_label}:{self.target_partner}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ResponsibilityState(BaseModel):
    """責任の心理的重み状態

    蓄積された責任の総量と、それが心理に与える影響を保持する。
    """
    # 責任の総量（0.0 - 1.0）
    total_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    # 未評価の判断数
    pending_decisions: int = Field(default=0, ge=0)

    # 過去の判断による傷（0.0 - 1.0）
    accumulated_harm: float = Field(default=0.0, ge=0.0, le=1.0)

    # 成功体験による自信（0.0 - 1.0）
    accumulated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # 最後の更新時刻
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # 決定履歴（直近N件のみ保持）
    recent_decisions: list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class ResponsibilityInfluence(BaseModel):
    """責任が心理状態に与える影響

    これを通じて他の機能に間接的に作用する。
    """
    # 喪失への恐れの増幅（0.0 - 0.5）
    fear_amplification: float = Field(default=0.0, ge=0.0, le=0.5)

    # 判断時の慎重さバイアス（0.0 - 0.5）
    caution_bias: float = Field(default=0.0, ge=0.0, le=0.5)

    # 不安・恐怖系感情のベースライン上昇（0.0 - 0.3）
    anxiety_baseline: float = Field(default=0.0, ge=0.0, le=0.3)

    # 共感・寄り添い方針へのバイアス（0.0 - 0.5）
    empathy_bias: float = Field(default=0.0, ge=0.0, le=0.5)


# ── Constants ──────────────────────────────────────────────────

# 責任の減衰率（1時間あたり）
DECAY_RATE_PER_HOUR: float = 0.02

# 責任の上限・下限
MAX_TOTAL_WEIGHT: float = 1.0
MIN_TOTAL_WEIGHT: float = 0.0

# 蓄積の上限
MAX_ACCUMULATED_HARM: float = 1.0
MAX_ACCUMULATED_CONFIDENCE: float = 1.0

# 影響の係数
FEAR_AMPLIFICATION_FACTOR: float = 0.4
CAUTION_BIAS_FACTOR: float = 0.3
ANXIETY_BASELINE_FACTOR: float = 0.2
EMPATHY_BIAS_FACTOR: float = 0.35

# 直近の決定履歴の保持数
MAX_RECENT_DECISIONS: int = 20


# ── Helper Functions ───────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _calculate_importance(policy: dict, context: dict) -> int:
    """判断の重要度を計算（1-5）"""
    importance = 3

    # 感情的に強い判断は重要度が高い
    policy_label = policy.get("policy_label", "")
    if policy_label in ("励ます", "共感する"):
        importance = max(importance, 4)
    elif policy_label == "からかう":
        importance = max(importance, 4)

    # 恐怖が高い状態での判断は重い
    fear_level = context.get("fear_level", 0.0)
    if fear_level > 0.5:
        importance = min(5, importance + 1)

    # 愛着対象への判断は重い
    if context.get("involves_attachment", False):
        importance = min(5, importance + 1)

    return importance


# ── Public API ─────────────────────────────────────────────────

def record_decision(
    resp_state: ResponsibilityState,
    policy: dict[str, Any],
    context: dict[str, Any],
) -> tuple[ResponsibilityState, str]:
    """判断を不変の決定記録として記録する。

    判断（Policy）を確定した瞬間に呼び出す。
    この時点で責任が発生するが、まだ重みは適用されない。

    Args:
        resp_state: 現在の責任状態
        policy: 確定した方針 (policy_label, rationale 等)
        context: 判断時のコンテキスト (partner, emotions, fear_level 等)

    Returns:
        (新しいResponsibilityState, decision_id)
    """
    # 決定記録を作成
    record = DecisionRecord(
        policy_label=policy.get("policy_label", "unknown"),
        policy_rationale=policy.get("rationale", ""),
        target_partner=context.get("target_partner", "unknown"),
        emotional_state=context.get("emotional_state", "neutral"),
        fear_level=context.get("fear_level", 0.0),
        importance=_calculate_importance(policy, context),
        evaluated=False,
        responsibility_delta=0.0,
    )

    # 状態を更新
    recent = list(resp_state.recent_decisions)
    recent.append(record.model_dump())

    # 履歴の上限を維持
    if len(recent) > MAX_RECENT_DECISIONS:
        recent = recent[-MAX_RECENT_DECISIONS:]

    new_state = ResponsibilityState(
        total_weight=resp_state.total_weight,
        pending_decisions=resp_state.pending_decisions + 1,
        accumulated_harm=resp_state.accumulated_harm,
        accumulated_confidence=resp_state.accumulated_confidence,
        last_updated=datetime.now().isoformat(timespec="seconds"),
        recent_decisions=recent,
    )

    return new_state, record.id


def evaluate_outcome(
    resp_state: ResponsibilityState,
    decision_id: str,
    outcome: dict[str, Any],
) -> ResponsibilityState:
    """結果を観測して責任を評価する。

    ユーザーの反応や関係性の変化を観測した後に呼び出す。
    ここで初めて責任の重みが計算され、適用される。

    Args:
        resp_state: 現在の責任状態
        decision_id: 評価対象の決定ID
        outcome: 観測された結果
            - user_reaction: "positive" | "negative" | "neutral" | "confused" | "rejected"
            - relationship_delta: float (-1.0 ~ 1.0)
            - expectation_gap: float (0.0 ~ 1.0) 期待との乖離

    Returns:
        新しいResponsibilityState
    """
    # 該当する決定記録を探す
    recent = list(resp_state.recent_decisions)
    target_idx = None
    target_record = None

    for i, rec in enumerate(recent):
        if rec.get("id") == decision_id and not rec.get("evaluated", False):
            target_idx = i
            target_record = rec
            break

    if target_record is None:
        # 見つからない場合は状態をそのまま返す
        return resp_state

    # 責任の増減量を計算
    delta = _calculate_responsibility_delta(target_record, outcome)

    # 記録を更新
    recent[target_idx]["evaluated"] = True
    recent[target_idx]["responsibility_delta"] = delta

    # 結果に応じて蓄積を更新
    harm_delta = 0.0
    confidence_delta = 0.0

    user_reaction = outcome.get("user_reaction", "neutral")
    if user_reaction in ("negative", "rejected"):
        harm_delta = abs(delta) * 0.5
    elif user_reaction == "positive":
        confidence_delta = abs(delta) * 0.3

    # 新しい状態を計算
    new_total = _clamp(
        resp_state.total_weight + delta,
        MIN_TOTAL_WEIGHT,
        MAX_TOTAL_WEIGHT,
    )
    new_harm = _clamp(
        resp_state.accumulated_harm + harm_delta,
        0.0,
        MAX_ACCUMULATED_HARM,
    )
    new_confidence = _clamp(
        resp_state.accumulated_confidence + confidence_delta,
        0.0,
        MAX_ACCUMULATED_CONFIDENCE,
    )

    return ResponsibilityState(
        total_weight=new_total,
        pending_decisions=max(0, resp_state.pending_decisions - 1),
        accumulated_harm=new_harm,
        accumulated_confidence=new_confidence,
        last_updated=datetime.now().isoformat(timespec="seconds"),
        recent_decisions=recent,
    )


def _calculate_responsibility_delta(
    record: dict,
    outcome: dict,
) -> float:
    """責任の増減量を計算する。

    正の値 = 責任が増える（傷つけた可能性）
    負の値 = 責任が減る（良い結果を生んだ）
    """
    delta = 0.0
    importance = record.get("importance", 3) / 5.0

    # ユーザーの反応による影響
    user_reaction = outcome.get("user_reaction", "neutral")
    reaction_weight = {
        "positive": -0.05,
        "neutral": 0.0,
        "confused": 0.03,
        "negative": 0.08,
        "rejected": 0.12,
    }.get(user_reaction, 0.0)

    delta += reaction_weight * importance

    # 関係性の変化による影響
    rel_delta = outcome.get("relationship_delta", 0.0)
    if rel_delta < 0:
        delta += abs(rel_delta) * 0.1 * importance
    else:
        delta -= rel_delta * 0.05 * importance

    # 期待との乖離による影響
    gap = outcome.get("expectation_gap", 0.0)
    delta += gap * 0.05 * importance

    return _clamp(delta, -0.1, 0.15)


def apply_decay(
    resp_state: ResponsibilityState,
    hours_elapsed: float,
) -> ResponsibilityState:
    """時間経過による自然減衰を適用する。

    責任は時間とともに少しずつ軽くなるが、
    蓄積された傷は減衰が遅い。

    Args:
        resp_state: 現在の責任状態
        hours_elapsed: 経過時間（時間）

    Returns:
        減衰後のResponsibilityState
    """
    hours = min(hours_elapsed, 168.0)  # 最大1週間

    # 総量は比較的早く減衰
    decay_factor = (1.0 - DECAY_RATE_PER_HOUR) ** hours
    new_total = resp_state.total_weight * decay_factor

    # 蓄積された傷はゆっくり減衰
    harm_decay = (1.0 - DECAY_RATE_PER_HOUR * 0.3) ** hours
    new_harm = resp_state.accumulated_harm * harm_decay

    # 自信もゆっくり減衰
    conf_decay = (1.0 - DECAY_RATE_PER_HOUR * 0.5) ** hours
    new_confidence = resp_state.accumulated_confidence * conf_decay

    return ResponsibilityState(
        total_weight=_clamp(new_total, MIN_TOTAL_WEIGHT, MAX_TOTAL_WEIGHT),
        pending_decisions=resp_state.pending_decisions,
        accumulated_harm=_clamp(new_harm, 0.0, MAX_ACCUMULATED_HARM),
        accumulated_confidence=_clamp(new_confidence, 0.0, MAX_ACCUMULATED_CONFIDENCE),
        last_updated=datetime.now().isoformat(timespec="seconds"),
        recent_decisions=resp_state.recent_decisions,
    )


def get_influence(resp_state: ResponsibilityState) -> ResponsibilityInfluence:
    """責任が心理状態に与える影響を取得する。

    この影響は他の機能（感情、判断、恐怖）に間接的に作用する。
    責任は他の機能を支配せず、あくまで「重み」として作用する。

    Args:
        resp_state: 現在の責任状態

    Returns:
        ResponsibilityInfluence
    """
    total = resp_state.total_weight
    harm = resp_state.accumulated_harm
    confidence = resp_state.accumulated_confidence

    # 傷と自信の差分が影響に作用
    net_burden = harm - confidence * 0.5

    # 喪失への恐れの増幅
    # 責任が重いほど、失うことを恐れる
    fear_amp = _clamp(
        total * FEAR_AMPLIFICATION_FACTOR + net_burden * 0.2,
        0.0,
        0.5,
    )

    # 判断時の慎重さバイアス
    # 傷が多いほど慎重になる
    caution = _clamp(
        harm * CAUTION_BIAS_FACTOR + total * 0.1,
        0.0,
        0.5,
    )

    # 不安・恐怖のベースライン上昇
    # 蓄積された責任が不安を生む
    anxiety = _clamp(
        net_burden * ANXIETY_BASELINE_FACTOR,
        0.0,
        0.3,
    )

    # 共感・寄り添いへのバイアス
    # 責任を感じているほど、傷つけないよう寄り添う
    empathy = _clamp(
        total * EMPATHY_BIAS_FACTOR + harm * 0.2,
        0.0,
        0.5,
    )

    return ResponsibilityInfluence(
        fear_amplification=round(fear_amp, 4),
        caution_bias=round(caution, 4),
        anxiety_baseline=round(anxiety, 4),
        empathy_bias=round(empathy, 4),
    )


def get_recent_decisions(
    resp_state: ResponsibilityState,
    evaluated_only: bool = False,
) -> list[dict]:
    """直近の決定記録を取得する。

    Args:
        resp_state: 責任状態
        evaluated_only: True の場合、評価済みの記録のみ返す

    Returns:
        決定記録のリスト
    """
    if evaluated_only:
        return [r for r in resp_state.recent_decisions if r.get("evaluated", False)]
    return list(resp_state.recent_decisions)


def create_default_state() -> ResponsibilityState:
    """初期状態を作成する。

    初回起動やデータ欠損時に使用。
    """
    return ResponsibilityState()


def to_dict(resp_state: ResponsibilityState) -> dict[str, Any]:
    """永続化用に辞書に変換する。"""
    return resp_state.model_dump()


def from_dict(data: dict[str, Any]) -> ResponsibilityState:
    """辞書から復元する。データ欠損時はデフォルト値で補完。"""
    try:
        return ResponsibilityState(**data)
    except Exception:
        return create_default_state()
