"""
psyche/other_hypothesis_emotion_return.py - 他者仮説由来の感情帯域追加

他者モデルの活性仮説に含まれる感情関連語を検出し、
現在の内部状態に基づいて感情ベクトルへの微弱な帰還量を導出する経路を開く。

設計原則 (design_other_hypothesis_emotion_return.md 準拠):
- 他者の意図・感情・状態を断定しない
- 仮説を「確定情報」として扱わない
- 帰還量を通じて他者モデルの仮説を修正・強化・固定しない
- 帰還量を通じて判断・行動・方針を確定させない
- 帰還の「望ましさ」「適切さ」を判定しない
- 特定の仮説内容が特定の感情を「必ず」引き起こすマッピングを定義しない
- 帰還量の蓄積・永続化を行わない（帰還事実記録のFIFO蓄積のみ許容）
- 帰還経路を通じて他者モデルの仮説生成・減衰・競合処理に影響を与えない
- 帰還経路を通じて記憶系・忘却系に影響を与えない
- enrichmentへの直接露出を行わない

4段パイプライン:
1. 仮説群からの感情関連語の構造的抽出
2. 帰還量の導出
3. 安全弁の適用
4. 帰還の適用と事実記録

安全弁 (7種):
1. 候補別帰還量上限: 1仮説からの帰還量を各感情軸ごとに上限で切り捨て
2. 合成後総帰還量上限: 全候補合算後の総量を価値方向性の最大バイアス強度以下に制限
3. 記憶帰還経路との合算帯域上限: 両経路合算が最大バイアス強度を超えないこと
4. ルーミネーション減衰: 同一仮説からの連続帰還を収束的に縮小
5. 同一ティック循環遮断: 処理順序の固定による即時循環防止
6. 感情値有効範囲クランプ: 帰還後の感情値が0.0-1.0の範囲内
7. enrichment非露出: 帰還の事実をenrichmentに含めない

Usage::

    processor = create_other_hypothesis_emotion_return()
    result = processor.process(
        active_hypotheses=hypotheses,
        current_emotions=emotion_dict,
        mood_valence=0.1,
        mood_arousal=0.4,
        max_bias_strength=0.15,
        memory_return_deltas={"joy": 0.01},
        tick_number=42,
    )
    # result.emotion_deltas -> {"joy": 0.005, "fear": -0.002, ...}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Emotion Keyword Dictionary
# =============================================================================

# Each emotion axis maps to positive-direction keywords and negative-direction keywords.
# Hypothesis description text is lowercased and checked for substring matches.
# All 7 axes are defined equivalently — no structural bias toward any axis.

EMOTION_KEYWORD_DICT: dict[str, dict[str, list[str]]] = {
    "joy": {
        "positive": [
            "happy", "pleased", "glad", "cheerful", "joyful", "delighted",
            "satisfied", "content", "enjoying", "excited", "positive",
            "enthusiastic", "engaged", "energetic",
        ],
        "negative": [
            "unhappy", "displeased", "dissatisfied", "joyless", "bored",
            "uninterested", "apathetic", "indifferent",
        ],
    },
    "anger": {
        "positive": [
            "angry", "frustrated", "irritated", "annoyed", "hostile",
            "aggressive", "tense", "confrontational", "agitated",
        ],
        "negative": [
            "calm", "peaceful", "relaxed", "composed", "patient",
            "tolerant", "accepting",
        ],
    },
    "sorrow": {
        "positive": [
            "sad", "sorrowful", "grieving", "melancholy", "despondent",
            "dejected", "distressed", "heavy", "mournful",
        ],
        "negative": [
            "uplifted", "relieved", "lightened", "comforted",
            "reassured", "encouraged",
        ],
    },
    "fear": {
        "positive": [
            "afraid", "fearful", "anxious", "nervous", "worried",
            "apprehensive", "uncertain", "threatened", "uneasy",
        ],
        "negative": [
            "confident", "secure", "assured", "safe", "comfortable",
            "trusting", "bold",
        ],
    },
    "surprise": {
        "positive": [
            "surprised", "unexpected", "astonished", "amazed",
            "startled", "shocked", "sudden",
        ],
        "negative": [
            "expected", "predictable", "unsurprised", "routine",
            "familiar", "anticipated",
        ],
    },
    "love": {
        "positive": [
            "affectionate", "caring", "warm", "tender", "attached",
            "connected", "bonded", "empathetic", "sympathetic",
        ],
        "negative": [
            "distant", "detached", "cold", "disengaged", "withdrawn",
            "aloof", "indifferent",
        ],
    },
    "fun": {
        "positive": [
            "playful", "humorous", "amusing", "funny", "lighthearted",
            "witty", "entertaining", "lively",
        ],
        "negative": [
            "serious", "somber", "solemn", "grave", "dull",
            "monotonous", "flat",
        ],
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HypothesisReturnRecord:
    """帰還事実の1レコード。

    仮説識別子、照合された感情ラベル群、帰還量、
    帰還時のムード方向、覚醒度水準、ルーミネーション減衰の適用有無を記録する。
    記録はFIFO方式で蓄積される。
    記録は判断系・行動系・目的系への経路を持たない。
    """
    hypothesis_id: str = ""
    emotion_labels: list[str] = field(default_factory=list)
    emotion_deltas: dict[str, float] = field(default_factory=dict)
    mood_direction: float = 0.0
    arousal_level: float = 0.0
    rumination_decay_applied: bool = False
    timestamp: float = field(default_factory=time.time)
    tick_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "emotion_labels": list(self.emotion_labels),
            "emotion_deltas": dict(self.emotion_deltas),
            "mood_direction": self.mood_direction,
            "arousal_level": self.arousal_level,
            "rumination_decay_applied": self.rumination_decay_applied,
            "timestamp": self.timestamp,
            "tick_number": self.tick_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisReturnRecord":
        return cls(
            hypothesis_id=data.get("hypothesis_id", ""),
            emotion_labels=list(data.get("emotion_labels", [])),
            emotion_deltas=dict(data.get("emotion_deltas", {})),
            mood_direction=data.get("mood_direction", 0.0),
            arousal_level=data.get("arousal_level", 0.0),
            rumination_decay_applied=data.get("rumination_decay_applied", False),
            timestamp=data.get("timestamp", 0.0),
            tick_number=data.get("tick_number", 0),
        )


@dataclass
class HypothesisReturnResult:
    """帰還処理の結果。

    emotion_deltas は各感情軸ごとの加算量。
    反応処理が生成する形式と同一の加算量として出力。
    """
    emotion_deltas: dict[str, float] = field(default_factory=dict)
    records_created: int = 0
    total_hypotheses_processed: int = 0
    hypotheses_with_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_deltas": dict(self.emotion_deltas),
            "records_created": self.records_created,
            "total_hypotheses_processed": self.total_hypotheses_processed,
            "hypotheses_with_matches": self.hypotheses_with_matches,
        }


# =============================================================================
# State
# =============================================================================

@dataclass
class OtherHypothesisEmotionReturnState:
    """帰還処理の永続化可能な状態。"""

    # 帰還事実履歴（FIFOリスト）
    return_history: list[HypothesisReturnRecord] = field(default_factory=list)

    # 前回適用ティック番号（同一ティック内での二重適用防止）
    last_applied_tick: int = -1

    # サイクルカウンタ（記述用、判定に使用しない）
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_history": [r.to_dict() for r in self.return_history],
            "last_applied_tick": self.last_applied_tick,
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OtherHypothesisEmotionReturnState":
        return cls(
            return_history=[
                HypothesisReturnRecord.from_dict(r)
                for r in data.get("return_history", [])
            ],
            last_applied_tick=data.get("last_applied_tick", -1),
            cycle_count=data.get("cycle_count", 0),
        )


# =============================================================================
# Configuration
# =============================================================================

def _oher_defaults() -> dict[str, Any]:
    """Load other hypothesis emotion return defaults from coefficient registry."""
    return coefficient_registry.get("other_hypothesis_emotion_return")


@dataclass
class OtherHypothesisEmotionReturnConfig:
    """設定。"""

    # 帰還事実履歴のFIFOウィンドウサイズ（ローカルバッファ管理、外部化対象外）
    history_window_size: int = 50

    # 候補別帰還量上限（各感情軸ごと）(安全弁1)
    # 記憶帰還経路の per_candidate_max_delta=0.03 以下
    per_candidate_max_delta: float = field(default_factory=lambda: _oher_defaults()["per_candidate_max_delta"])

    # 合成後総帰還量上限（各感情軸ごと）(安全弁2)
    # 記憶帰還経路の total_max_delta=0.15 の半分以下
    total_max_delta: float = field(default_factory=lambda: _oher_defaults()["total_max_delta"])

    # ルーミネーション減衰: 履歴内の同一仮説出現回数の閾値（安全弁4）
    rumination_threshold: int = field(default_factory=lambda: _oher_defaults()["rumination_threshold"])

    # ルーミネーション減衰率: 出現回数に応じた減衰
    rumination_decay_factor: float = field(default_factory=lambda: _oher_defaults()["rumination_decay_factor"])

    # 覚醒度による帰還量鈍化の閾値
    low_arousal_threshold: float = field(default_factory=lambda: _oher_defaults()["low_arousal_threshold"])

    # 覚醒度鈍化係数（低覚醒時にかかる係数）
    low_arousal_scale: float = field(default_factory=lambda: _oher_defaults()["low_arousal_scale"])

    # 既存感情値による収束係数（高い感情への帰還は縮小する）
    convergence_scale: float = field(default_factory=lambda: _oher_defaults()["convergence_scale"])

    # 合算帯域上限（安全弁3: 記憶帰還経路との合算）
    # 本経路の帯域は記憶帰還経路より狭い
    combined_max_delta: float = field(default_factory=lambda: _oher_defaults()["combined_max_delta"])


# =============================================================================
# Stage 1: 仮説群からの感情関連語の構造的抽出
# =============================================================================

def _extract_emotion_labels(
    hypotheses: list[Any],
    keyword_dict: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """活性仮説の記述テキストから感情関連語との照合により、
    各仮説に対応する感情ラベル群を導出する。

    テキストの意味解釈を行わない。
    キーワードの部分文字列照合のみで実行し、文脈理解や推論を含まない。
    照合に失敗した仮説（感情関連語を含まない仮説）は帰還対象から除外する。

    Returns:
        list of dicts with: hypothesis_id, description, strength, freshness,
        competing_count, matches (list of (emotion_axis, direction))
    """
    results: list[dict[str, Any]] = []

    for hyp in hypotheses:
        hypothesis_id = getattr(hyp, "hypothesis_id", "")
        description = getattr(hyp, "description", "")
        strength = getattr(hyp, "strength", 0.0)
        freshness = getattr(hyp, "freshness", 0.0)
        competing_ids = getattr(hyp, "competing_ids", ())

        if not description or not isinstance(description, str):
            continue

        text_lower = description.lower()
        matches: list[tuple[str, str]] = []  # (emotion_axis, direction)

        for emotion_axis, direction_keywords in keyword_dict.items():
            for direction, keywords in direction_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append((emotion_axis, direction))
                        break  # one match per direction per axis is enough

        if matches:
            results.append({
                "hypothesis_id": hypothesis_id,
                "description": description,
                "strength": float(strength) if isinstance(strength, (int, float)) else 0.0,
                "freshness": float(freshness) if isinstance(freshness, (int, float)) else 0.0,
                "competing_count": len(competing_ids) if competing_ids else 0,
                "matches": matches,
            })

    return results


# =============================================================================
# Stage 2: 帰還量の導出
# =============================================================================

def _derive_return_amounts(
    matched_entries: list[dict[str, Any]],
    current_emotions: dict[str, float],
    mood_valence: float,
    mood_arousal: float,
    config: OtherHypothesisEmotionReturnConfig,
) -> list[dict[str, Any]]:
    """各候補の感情ラベルから、感情ベクトルへの帰還量を都度の内部状態から導出する。

    帰還量は以下の要素の組み合わせから算出される:
    - 仮説の強度（強度が高いほど帰還量が大きい）
    - 仮説の鮮度（鮮度が低いほど帰還量が縮小する）
    - 仮説の競合状態（競合相手が多い仮説ほど帰還量が縮小する）
    - 現在の感情ベクトルにおける該当感情の既存値（既に高い感情への帰還は収束的に縮小する）
    - 現在のムードの覚醒度（低覚醒時は帰還量全体が鈍化する）

    帰還量の方向（正負）は照合された感情ラベルの方向から導出されるが、
    ムードの感情価との組み合わせで都度変動する。

    Returns:
        list of dicts with: hypothesis_id, deltas (dict: label -> float)
    """
    results: list[dict[str, Any]] = []

    for entry in matched_entries:
        hypothesis_id = entry["hypothesis_id"]
        strength = entry["strength"]
        freshness = entry["freshness"]
        competing_count = entry["competing_count"]
        matches = entry["matches"]

        deltas: dict[str, float] = {}

        for emotion_axis, direction in matches:
            # Base amount from hypothesis strength
            base_amount = strength * 0.08

            # Freshness scaling: lower freshness -> smaller return
            freshness_factor = max(0.0, freshness)
            base_amount *= freshness_factor

            # Competition reduction: more competitors -> smaller return
            if competing_count > 0:
                competition_factor = 1.0 / (1.0 + competing_count * 0.3)
                base_amount *= competition_factor

            # Direction from keyword match
            sign = 1.0 if direction == "positive" else -1.0

            # Mood-valence interaction: when mood and direction are aligned,
            # return is slightly stronger; when opposed, slightly weaker
            mood_alignment = 1.0
            if sign > 0 and mood_valence > 0:
                mood_alignment = 1.0 + abs(mood_valence) * 0.15
            elif sign > 0 and mood_valence < 0:
                mood_alignment = 1.0 - abs(mood_valence) * 0.15
            elif sign < 0 and mood_valence < 0:
                mood_alignment = 1.0 + abs(mood_valence) * 0.15
            elif sign < 0 and mood_valence > 0:
                mood_alignment = 1.0 - abs(mood_valence) * 0.15

            base_amount *= mood_alignment

            # Convergence: already-high emotions get less return
            current_val = current_emotions.get(emotion_axis, 0.0)
            if sign > 0 and current_val > 0.3:
                convergence = 1.0 - (current_val - 0.3) * config.convergence_scale
                base_amount *= max(0.1, convergence)

            # Arousal scaling: low arousal dampens return
            if mood_arousal < config.low_arousal_threshold:
                arousal_factor = config.low_arousal_scale + (
                    (1.0 - config.low_arousal_scale) *
                    (mood_arousal / max(0.01, config.low_arousal_threshold))
                )
                base_amount *= arousal_factor

            delta = sign * base_amount

            # Accumulate per-axis (multiple matches may affect the same axis)
            if emotion_axis in deltas:
                deltas[emotion_axis] += delta
            else:
                deltas[emotion_axis] = delta

        if deltas:
            results.append({
                "hypothesis_id": hypothesis_id,
                "deltas": deltas,
            })

    return results


# =============================================================================
# Stage 3: 安全弁の適用
# =============================================================================

def _apply_safety_valves(
    derived_entries: list[dict[str, Any]],
    return_history: list[HypothesisReturnRecord],
    config: OtherHypothesisEmotionReturnConfig,
    max_bias_strength: float,
    memory_return_deltas: Optional[dict[str, float]] = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """安全弁を段階2で導出された帰還量に適用する。

    安全弁1: 候補別帰還量上限
    安全弁2: 合成後総帰還量上限
    安全弁3: 記憶帰還経路との合算帯域上限
    安全弁4: ルーミネーション減衰

    Returns:
        (modified entries with rumination_applied flag, total_deltas dict)
    """
    # Count occurrences of each hypothesis_id in recent history (for rumination)
    history_counts: dict[str, int] = {}
    for rec in return_history:
        hid = rec.hypothesis_id
        if hid:
            history_counts[hid] = history_counts.get(hid, 0) + 1

    # Effective total max: min of config total_max and max_bias_strength
    effective_total_max = min(config.total_max_delta, max_bias_strength)

    modified_entries: list[dict[str, Any]] = []

    for entry in derived_entries:
        hypothesis_id = entry["hypothesis_id"]
        deltas = dict(entry["deltas"])
        rumination_applied = False

        # Safety valve 4: Rumination decay
        occurrence_count = history_counts.get(hypothesis_id, 0)
        if occurrence_count >= config.rumination_threshold:
            decay_multiplier = max(
                0.0,
                1.0 - (occurrence_count - config.rumination_threshold + 1) *
                config.rumination_decay_factor
            )
            for label in deltas:
                deltas[label] *= decay_multiplier
            rumination_applied = True

        # Safety valve 1: Per-candidate max delta (per axis)
        for label in deltas:
            if deltas[label] > config.per_candidate_max_delta:
                deltas[label] = config.per_candidate_max_delta
            elif deltas[label] < -config.per_candidate_max_delta:
                deltas[label] = -config.per_candidate_max_delta

        modified_entries.append({
            "hypothesis_id": hypothesis_id,
            "deltas": deltas,
            "rumination_applied": rumination_applied,
        })

    # Safety valve 2: Total delta per axis (sum across all candidates)
    total_deltas: dict[str, float] = {}
    for entry in modified_entries:
        for label, delta in entry["deltas"].items():
            if label in total_deltas:
                total_deltas[label] += delta
            else:
                total_deltas[label] = delta

    # Clamp total deltas to effective_total_max
    for label in total_deltas:
        if total_deltas[label] > effective_total_max:
            total_deltas[label] = effective_total_max
        elif total_deltas[label] < -effective_total_max:
            total_deltas[label] = -effective_total_max

    # Safety valve 3: Combined bandwidth with memory return path
    if memory_return_deltas:
        combined_max = min(config.combined_max_delta, max_bias_strength)
        for label in total_deltas:
            mem_delta = memory_return_deltas.get(label, 0.0)
            combined = abs(total_deltas[label]) + abs(mem_delta)
            if combined > combined_max:
                # Reduce this path's contribution to stay within combined limit
                allowed = max(0.0, combined_max - abs(mem_delta))
                if total_deltas[label] > 0:
                    total_deltas[label] = min(total_deltas[label], allowed)
                else:
                    total_deltas[label] = max(total_deltas[label], -allowed)

    return modified_entries, total_deltas


# =============================================================================
# Stage 4: 帰還の適用と事実記録
# =============================================================================

def _apply_and_record(
    total_deltas: dict[str, float],
    modified_entries: list[dict[str, Any]],
    current_emotions: dict[str, float],
    mood_valence: float,
    mood_arousal: float,
    tick_number: int,
) -> tuple[dict[str, float], list[HypothesisReturnRecord]]:
    """安全弁を通過した帰還量を出力し、帰還の事実を記録する。

    Safety valve 6: Emotion value range clamp (0.0 - 1.0)

    出力は感情ベクトルの変化量のみ。本モジュールは感情ベクトルを直接書き換えない。

    Returns:
        (clamped_deltas, new_records)
    """
    # Clamp deltas so that applied emotions remain in [0.0, 1.0]
    clamped_deltas: dict[str, float] = {}
    for label, delta in total_deltas.items():
        current_val = current_emotions.get(label, 0.0)
        new_val = current_val + delta
        clamped_new = _clamp(new_val, 0.0, 1.0)
        actual_delta = clamped_new - current_val
        if abs(actual_delta) > 1e-6:
            clamped_deltas[label] = actual_delta

    # Create records for each contributing candidate
    new_records: list[HypothesisReturnRecord] = []
    now = time.time()
    for entry in modified_entries:
        deltas = entry["deltas"]
        has_nonzero = any(abs(v) > 1e-6 for v in deltas.values())
        if has_nonzero:
            record = HypothesisReturnRecord(
                hypothesis_id=entry["hypothesis_id"],
                emotion_labels=list(deltas.keys()),
                emotion_deltas=dict(deltas),
                mood_direction=mood_valence,
                arousal_level=mood_arousal,
                rumination_decay_applied=entry.get("rumination_applied", False),
                timestamp=now,
                tick_number=tick_number,
            )
            new_records.append(record)

    return clamped_deltas, new_records


# =============================================================================
# Processor
# =============================================================================

class OtherHypothesisEmotionReturnProcessor:
    """他者仮説由来の感情帰還プロセッサ。

    4段パイプライン:
    1. 仮説群からの感情関連語の構造的抽出
    2. 帰還量の導出
    3. 安全弁の適用
    4. 帰還の適用と事実記録

    出力は感情ベクトルの変化量のみ。
    帰還事実記録はFIFO蓄積のみ。
    判断・行動・評価を直接引き起こさない。
    enrichmentへの直接露出を行わない。
    他者モデルへの書き込み経路を持たない。
    記憶系・忘却系への経路を持たない。
    """

    def __init__(self, config: Optional[OtherHypothesisEmotionReturnConfig] = None):
        self._config = config or OtherHypothesisEmotionReturnConfig()
        self._state = OtherHypothesisEmotionReturnState()

    @property
    def state(self) -> OtherHypothesisEmotionReturnState:
        return self._state

    @state.setter
    def state(self, value: OtherHypothesisEmotionReturnState) -> None:
        self._state = value

    def process(
        self,
        active_hypotheses: Optional[list[Any]] = None,
        current_emotions: Optional[dict[str, float]] = None,
        mood_valence: float = 0.0,
        mood_arousal: float = 0.3,
        max_bias_strength: float = 0.15,
        memory_return_deltas: Optional[dict[str, float]] = None,
        tick_number: int = 0,
    ) -> HypothesisReturnResult:
        """4段パイプラインの帰還処理を実行する。

        Args:
            active_hypotheses: 活性仮説リスト (READ-ONLY, 他者モデルから取得)
            current_emotions: 現在の感情ベクトル (READ-ONLY, label -> value)
            mood_valence: 現在のムード感情価
            mood_arousal: 現在のムード覚醒度
            max_bias_strength: 価値方向性の最大バイアス強度
            memory_return_deltas: 記憶帰還経路の帰還量 (合算帯域制約用, READ-ONLY)
            tick_number: 現在のティック番号

        Returns:
            HypothesisReturnResult with emotion_deltas (label -> delta)
        """
        # Safety valve 5: Same-tick prevention
        if tick_number == self._state.last_applied_tick and tick_number >= 0:
            return HypothesisReturnResult()

        self._state.cycle_count += 1

        hypotheses = active_hypotheses or []
        emotions = current_emotions or {}

        # No hypotheses -> no return
        if not hypotheses:
            self._state.last_applied_tick = tick_number
            return HypothesisReturnResult()

        # === Stage 1: Extract emotion labels from hypotheses ===
        matched = _extract_emotion_labels(hypotheses, EMOTION_KEYWORD_DICT)

        if not matched:
            self._state.last_applied_tick = tick_number
            return HypothesisReturnResult(
                total_hypotheses_processed=len(hypotheses),
            )

        # === Stage 2: Derive return amounts ===
        derived = _derive_return_amounts(
            matched, emotions, mood_valence, mood_arousal, self._config,
        )

        if not derived:
            self._state.last_applied_tick = tick_number
            return HypothesisReturnResult(
                total_hypotheses_processed=len(hypotheses),
                hypotheses_with_matches=len(matched),
            )

        # === Stage 3: Apply safety valves ===
        modified_entries, total_deltas = _apply_safety_valves(
            derived, self._state.return_history, self._config,
            max_bias_strength, memory_return_deltas,
        )

        if not total_deltas:
            self._state.last_applied_tick = tick_number
            return HypothesisReturnResult(
                total_hypotheses_processed=len(hypotheses),
                hypotheses_with_matches=len(matched),
            )

        # === Stage 4: Apply and record ===
        clamped_deltas, new_records = _apply_and_record(
            total_deltas, modified_entries, emotions,
            mood_valence, mood_arousal, tick_number,
        )

        # Update state: add new records to history
        self._state.return_history.extend(new_records)

        # FIFO trimming
        window = self._config.history_window_size
        if len(self._state.return_history) > window:
            self._state.return_history = self._state.return_history[-window:]

        # Record last applied tick
        self._state.last_applied_tick = tick_number

        logger.debug(
            "Other hypothesis emotion return: cycle=%d, tick=%d, "
            "hypotheses=%d, matched=%d, deltas=%s, records=%d",
            self._state.cycle_count, tick_number,
            len(hypotheses), len(matched),
            {k: round(v, 4) for k, v in clamped_deltas.items()},
            len(new_records),
        )

        return HypothesisReturnResult(
            emotion_deltas=clamped_deltas,
            records_created=len(new_records),
            total_hypotheses_processed=len(hypotheses),
            hypotheses_with_matches=len(matched),
        )

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。判断系への経路なし。"""
        st = self._state
        return {
            "cycle_count": st.cycle_count,
            "history_length": len(st.return_history),
            "last_applied_tick": st.last_applied_tick,
        }


# =============================================================================
# Factory
# =============================================================================

def create_other_hypothesis_emotion_return(
    config: Optional[OtherHypothesisEmotionReturnConfig] = None,
) -> OtherHypothesisEmotionReturnProcessor:
    """OtherHypothesisEmotionReturnProcessor のファクトリ関数。"""
    return OtherHypothesisEmotionReturnProcessor(config=config)
