"""
psyche/memory_emotion_return.py - 記憶想起から感情への帰還経路

想起された記憶に付随する感情痕跡を読み取り、現在の感情ベクトルへの
帰還量を都度の内部状態から導出して適用する経路を開く。

設計原則 (design_memory_emotion_return.md 準拠):
- 特定の記憶が特定の感情を「必ず」引き起こすマッピングを定義しない
- 想起された記憶の内容をテキストとして解釈しない
- 記憶の「重要度」「有用性」「正誤」を判定しない
- 帰還量の方向（正負）を固定しない
- 帰還経路を通じて記憶の想起条件を変更しない（想起モジュールへの逆流なし）
- 帰還経路を通じて忘却処理に影響を与えない
- 帰還処理の結果に基づいて判断・行動・方針を確定しない
- 感情帰還の「望ましさ」「適切さ」を判定しない
- 記憶の繰り返し想起による感情の無限増幅を許容しない
- enrichmentへの直接露出を行わない

4段パイプライン:
1. 想起候補の感情痕跡収集
2. 帰還量の導出
3. 安全弁の適用
4. 帰還の適用と事実記録

安全弁:
1. 候補別帰還量上限: 1候補からの帰還量を各感情軸ごとに上限で切り捨て
2. 合成後総帰還量上限: 全候補合算後の総量を価値方向性の最大バイアス強度以下に制限
3. ルーミネーション減衰: 同一記憶からの連続帰還を収束的に縮小
4. 同一ティック循環遮断: 処理順序の固定によるティック内正帰還ループの構造的排除
5. 感情値有効範囲クランプ: 帰還後の感情値が0.0-1.0の範囲内であることの保証
6. 既存感情減衰との協調: 帰還で上昇した感情は既存の指数減衰により時間とともに基線復帰
7. enrichment非露出: 帰還の事実をenrichmentに含めない

Usage::

    processor = create_memory_emotion_return()
    result = processor.process(
        multi_path_candidates=mpr_candidates,
        spontaneous_candidates=sr_candidates,
        binding_store=binding_store,
        current_emotions=emotion_dict,
        mood_valence=0.1,
        mood_arousal=0.4,
        max_bias_strength=0.15,
        tick_number=42,
    )
    # result.emotion_deltas -> {"joy": 0.02, "sorrow": -0.01, ...}
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
# Data Structures
# =============================================================================

@dataclass
class ReturnRecord:
    """帰還事実の1レコード。

    帰還元の記憶識別子、想起系統ラベル、帰還された感情ラベルと量、
    帰還時の内部状態断面、ルーミネーション減衰の適用有無を記録する。
    記録はFIFO方式で蓄積され、ルーミネーション減衰の判定に使用される。
    記録は判断系・行動系・目的系への経路を持たない。
    """
    unit_id: str = ""
    recall_system_label: str = ""  # "multi_path" or "spontaneous"
    emotion_labels: list[str] = field(default_factory=list)
    emotion_deltas: dict[str, float] = field(default_factory=dict)
    mood_direction: float = 0.0
    arousal_level: float = 0.0
    rumination_decay_applied: bool = False
    timestamp: float = field(default_factory=time.time)
    tick_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "recall_system_label": self.recall_system_label,
            "emotion_labels": list(self.emotion_labels),
            "emotion_deltas": dict(self.emotion_deltas),
            "mood_direction": self.mood_direction,
            "arousal_level": self.arousal_level,
            "rumination_decay_applied": self.rumination_decay_applied,
            "timestamp": self.timestamp,
            "tick_number": self.tick_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReturnRecord":
        return cls(
            unit_id=data.get("unit_id", ""),
            recall_system_label=data.get("recall_system_label", ""),
            emotion_labels=list(data.get("emotion_labels", [])),
            emotion_deltas=dict(data.get("emotion_deltas", {})),
            mood_direction=data.get("mood_direction", 0.0),
            arousal_level=data.get("arousal_level", 0.0),
            rumination_decay_applied=data.get("rumination_decay_applied", False),
            timestamp=data.get("timestamp", 0.0),
            tick_number=data.get("tick_number", 0),
        )


@dataclass
class ReturnResult:
    """帰還処理の結果。

    emotion_deltas は各感情軸ごとの加算量。
    反応処理が生成する形式と同一の加算量として出力。
    """
    emotion_deltas: dict[str, float] = field(default_factory=dict)
    records_created: int = 0
    total_candidates_processed: int = 0
    candidates_with_traces: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_deltas": dict(self.emotion_deltas),
            "records_created": self.records_created,
            "total_candidates_processed": self.total_candidates_processed,
            "candidates_with_traces": self.candidates_with_traces,
        }


# =============================================================================
# State
# =============================================================================

@dataclass
class MemoryEmotionReturnState:
    """帰還処理の永続化可能な状態。"""

    # 帰還事実履歴（FIFOリスト）
    return_history: list[ReturnRecord] = field(default_factory=list)

    # 前回適用ティック番号（同一ティック内での二重適用防止）
    last_applied_tick: int = -1

    # サイクルカウンタ（記述用、判定に使用しない）
    cycle_count: int = 0

    # 方向連続カウント（減衰適用済み）: 正方向と負方向を別々に追跡
    direction_consecutive_count_positive: float = 0.0
    direction_consecutive_count_negative: float = 0.0

    # 直前の帰還方向ラベル: "positive", "negative", "neutral"
    last_direction_label: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_history": [r.to_dict() for r in self.return_history],
            "last_applied_tick": self.last_applied_tick,
            "cycle_count": self.cycle_count,
            "direction_consecutive_count_positive": self.direction_consecutive_count_positive,
            "direction_consecutive_count_negative": self.direction_consecutive_count_negative,
            "last_direction_label": self.last_direction_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEmotionReturnState":
        return cls(
            return_history=[
                ReturnRecord.from_dict(r) for r in data.get("return_history", [])
            ],
            last_applied_tick=data.get("last_applied_tick", -1),
            cycle_count=data.get("cycle_count", 0),
            direction_consecutive_count_positive=data.get(
                "direction_consecutive_count_positive", 0.0
            ),
            direction_consecutive_count_negative=data.get(
                "direction_consecutive_count_negative", 0.0
            ),
            last_direction_label=data.get("last_direction_label", "neutral"),
        )


# =============================================================================
# Configuration
# =============================================================================

def _mer_defaults() -> dict[str, Any]:
    """Load memory emotion return defaults from coefficient registry."""
    return coefficient_registry.get("memory_emotion_return")


@dataclass
class MemoryEmotionReturnConfig:
    """設定。"""

    # 帰還事実履歴のFIFOウィンドウサイズ（ローカルバッファ管理、外部化対象外）
    history_window_size: int = 50

    # 候補別帰還量上限（各感情軸ごと）(安全弁1)
    per_candidate_max_delta: float = field(default_factory=lambda: _mer_defaults()["per_candidate_max_delta"])

    # 合成後総帰還量上限（各感情軸ごと）(安全弁2)
    # 実行時に max_bias_strength で上書き可能
    total_max_delta: float = field(default_factory=lambda: _mer_defaults()["total_max_delta"])

    # ルーミネーション減衰: 履歴内の同一記憶出現回数の閾値（安全弁3）
    rumination_threshold: int = field(default_factory=lambda: _mer_defaults()["rumination_threshold"])

    # ルーミネーション減衰率: 出現回数に応じた減衰 (回数 * この値で帰還量を削減)
    rumination_decay_factor: float = field(default_factory=lambda: _mer_defaults()["rumination_decay_factor"])

    # 覚醒度による帰還量鈍化の閾値
    low_arousal_threshold: float = field(default_factory=lambda: _mer_defaults()["low_arousal_threshold"])

    # 覚醒度鈍化係数（低覚醒時にかかる係数）
    low_arousal_scale: float = field(default_factory=lambda: _mer_defaults()["low_arousal_scale"])

    # 既存感情値による収束係数（高い感情への帰還は縮小する）
    convergence_scale: float = field(default_factory=lambda: _mer_defaults()["convergence_scale"])

    # 方向連続性追跡: 段階的鮮度減衰の減衰率（0-1, 1に近いほど減衰が弱い）
    direction_freshness_decay: float = field(default_factory=lambda: _mer_defaults()["direction_freshness_decay"])

    # 方向連続性追跡: 変調量の上限（現在の追従速度に対する比率）
    tracking_speed_modulation_ratio_cap: float = field(default_factory=lambda: _mer_defaults()["tracking_speed_modulation_ratio_cap"])

    # 方向連続性追跡: 連続カウントから変調量への変換係数
    tracking_speed_modulation_scale: float = field(default_factory=lambda: _mer_defaults()["tracking_speed_modulation_scale"])


# =============================================================================
# Stage 1: 想起候補の感情痕跡収集
# =============================================================================

def _collect_trace_data(
    candidates: list[Any],
    binding_store: Any,
    system_label: str,
) -> list[dict[str, Any]]:
    """2系統の想起候補リストを入力とし、各候補に対応する感情記憶紐づけ構造から
    感情痕跡を読み取る。

    紐づけが存在しない候補は帰還対象から除外する。
    2系統の候補は等価に扱い、系統間の優先順位を設けない。

    Returns:
        list of dicts with: unit_id, system_label, traces (list of dicts)
        Each trace dict: emotion_label, intensity, valence, freshness
    """
    results: list[dict[str, Any]] = []

    if not candidates or binding_store is None:
        return results

    # Build trace map from binding_store: memory_key -> traces
    trace_map: dict[str, list[dict[str, float]]] = {}
    bindings = getattr(binding_store, "bindings", ()) or ()
    for binding in bindings:
        memory_key = getattr(binding, "memory_key", "")
        if not memory_key:
            continue
        traces = getattr(binding, "traces", ()) or ()
        trace_list: list[dict[str, float]] = []
        for trace in traces:
            label = getattr(trace, "emotion_label", "")
            intensity = getattr(trace, "intensity", 0.0)
            valence = getattr(trace, "valence", 0.0)
            freshness = getattr(trace, "freshness", 0.0)
            if label and isinstance(intensity, (int, float)):
                trace_list.append({
                    "emotion_label": label,
                    "intensity": float(intensity),
                    "valence": float(valence) if isinstance(valence, (int, float)) else 0.0,
                    "freshness": float(freshness) if isinstance(freshness, (int, float)) else 0.0,
                })
        if trace_list:
            trace_map[memory_key] = trace_list

    for candidate in candidates:
        unit_id = getattr(candidate, "unit_id", "")
        source_id = getattr(candidate, "source_id", "")

        # Look up traces by unit_id and source_id
        matched_traces: list[dict[str, float]] = []
        if unit_id and unit_id in trace_map:
            matched_traces = trace_map[unit_id]
        elif source_id and source_id in trace_map:
            matched_traces = trace_map[source_id]

        if matched_traces:
            results.append({
                "unit_id": unit_id or source_id,
                "system_label": system_label,
                "traces": matched_traces,
            })

    return results


# =============================================================================
# Stage 2: 帰還量の導出
# =============================================================================

def _derive_return_amounts(
    trace_entries: list[dict[str, Any]],
    current_emotions: dict[str, float],
    mood_valence: float,
    mood_arousal: float,
    config: MemoryEmotionReturnConfig,
) -> list[dict[str, Any]]:
    """各候補の感情痕跡から、現在の感情ベクトルへの帰還量を導出する。

    帰還量は以下の要素の組み合わせから都度算出される：
    - 感情痕跡の強度（痕跡が強いほど帰還量が大きい）
    - 感情痕跡の鮮度（鮮度が低いほど帰還量が縮小する）
    - 現在の感情ベクトルにおける該当感情の既存値（既に高い感情への帰還は収束的に縮小する）
    - 現在のムードの覚醒度（低覚醒時は帰還量全体が鈍化する）

    帰還量の方向（正負）は痕跡の感情価から導出されるが、固定値ではなく、
    現在のムード方向との関係で都度異なりうる。

    Returns:
        list of dicts with: unit_id, system_label, deltas (dict: label -> float)
    """
    results: list[dict[str, Any]] = []

    for entry in trace_entries:
        unit_id = entry["unit_id"]
        system_label = entry["system_label"]
        traces = entry["traces"]

        deltas: dict[str, float] = {}

        for trace in traces:
            label = trace["emotion_label"]
            intensity = trace["intensity"]
            freshness = trace["freshness"]
            valence = trace["valence"]

            # Base amount from intensity
            base_amount = intensity * 0.1

            # Freshness scaling: lower freshness -> smaller return
            freshness_factor = max(0.0, freshness)
            base_amount *= freshness_factor

            # Direction from trace valence, modulated by current mood
            # Positive valence -> positive delta, negative -> negative delta
            # But mood direction can influence: same-direction is slightly amplified,
            # opposite-direction is slightly dampened
            direction = 1.0
            if valence < 0:
                direction = -1.0
            elif valence == 0.0:
                # Neutral valence: direction depends on mood
                direction = 1.0 if mood_valence >= 0 else -1.0

            # Mood-valence interaction: when mood and trace valence are aligned,
            # return is slightly stronger; when opposed, slightly weaker
            mood_alignment = 1.0
            if valence != 0.0:
                # Same sign: boost, opposite sign: dampen
                if mood_valence * valence > 0:
                    mood_alignment = 1.0 + abs(mood_valence) * 0.2
                elif mood_valence * valence < 0:
                    mood_alignment = 1.0 - abs(mood_valence) * 0.2

            base_amount *= mood_alignment

            # Convergence: already-high emotions get less return (安全弁的)
            current_val = current_emotions.get(label, 0.0)
            if direction > 0 and current_val > 0.3:
                convergence = 1.0 - (current_val - 0.3) * config.convergence_scale
                base_amount *= max(0.1, convergence)

            # Arousal scaling: low arousal dampens return
            if mood_arousal < config.low_arousal_threshold:
                arousal_factor = config.low_arousal_scale + (
                    (1.0 - config.low_arousal_scale) *
                    (mood_arousal / max(0.01, config.low_arousal_threshold))
                )
                base_amount *= arousal_factor

            delta = direction * base_amount

            # Accumulate per-label (multiple traces may affect the same label)
            if label in deltas:
                deltas[label] += delta
            else:
                deltas[label] = delta

        if deltas:
            results.append({
                "unit_id": unit_id,
                "system_label": system_label,
                "deltas": deltas,
            })

    return results


# =============================================================================
# Stage 3: 安全弁の適用
# =============================================================================

def _apply_safety_valves(
    derived_entries: list[dict[str, Any]],
    return_history: list[ReturnRecord],
    config: MemoryEmotionReturnConfig,
    max_bias_strength: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """安全弁を段階2で導出された帰還量に適用する。

    安全弁1: 候補別帰還量上限
    安全弁2: 合成後総帰還量上限
    安全弁3: ルーミネーション減衰

    Returns:
        (modified entries with rumination_applied flag, total_deltas dict)
    """
    # Count occurrences of each unit_id in recent history (for rumination)
    history_counts: dict[str, int] = {}
    for rec in return_history:
        uid = rec.unit_id
        if uid:
            history_counts[uid] = history_counts.get(uid, 0) + 1

    # Effective total max: min of config and max_bias_strength
    effective_total_max = min(config.total_max_delta, max_bias_strength)

    modified_entries: list[dict[str, Any]] = []

    for entry in derived_entries:
        unit_id = entry["unit_id"]
        deltas = dict(entry["deltas"])
        rumination_applied = False

        # Safety valve 3: Rumination decay
        occurrence_count = history_counts.get(unit_id, 0)
        if occurrence_count >= config.rumination_threshold:
            # Decay proportional to occurrence count
            # After rumination_threshold, each additional occurrence further reduces
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
            "unit_id": unit_id,
            "system_label": entry["system_label"],
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
    state: MemoryEmotionReturnState,
    config: MemoryEmotionReturnConfig,
) -> tuple[dict[str, float], list[ReturnRecord]]:
    """安全弁を通過した帰還量を感情ベクトルの各軸に加算する。

    Safety valve 5: Emotion value range clamp (0.0 - 1.0)

    同時に帰還の事実を記録する。

    Returns:
        (clamped_deltas, new_records)
    """
    # Clamp deltas so that applied emotions remain in [0.0, 1.0]
    clamped_deltas: dict[str, float] = {}
    for label, delta in total_deltas.items():
        current_val = current_emotions.get(label, 0.0)
        new_val = current_val + delta
        # Clamp to valid range
        clamped_new = _clamp(new_val, 0.0, 1.0)
        # Actual delta after clamping
        actual_delta = clamped_new - current_val
        if abs(actual_delta) > 1e-6:
            clamped_deltas[label] = actual_delta

    # Create records for each contributing candidate
    new_records: list[ReturnRecord] = []
    now = time.time()
    for entry in modified_entries:
        deltas = entry["deltas"]
        # Only record if there are non-zero deltas
        has_nonzero = any(abs(v) > 1e-6 for v in deltas.values())
        if has_nonzero:
            record = ReturnRecord(
                unit_id=entry["unit_id"],
                recall_system_label=entry["system_label"],
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

class MemoryEmotionReturnProcessor:
    """記憶想起から感情への帰還経路プロセッサ。

    4段パイプライン:
    1. 想起候補の感情痕跡収集
    2. 帰還量の導出
    3. 安全弁の適用
    4. 帰還の適用と事実記録

    出力は感情ベクトルの変化量のみ。
    帰還事実記録はFIFO蓄積のみ。
    判断・行動・評価を直接引き起こさない。
    enrichmentへの直接露出を行わない。
    想起モジュールへの逆流経路を持たない。
    忘却処理への経路を持たない。
    """

    def __init__(self, config: Optional[MemoryEmotionReturnConfig] = None):
        self._config = config or MemoryEmotionReturnConfig()
        self._state = MemoryEmotionReturnState()

    @property
    def state(self) -> MemoryEmotionReturnState:
        return self._state

    @state.setter
    def state(self, value: MemoryEmotionReturnState) -> None:
        self._state = value

    def process(
        self,
        multi_path_candidates: Optional[list[Any]] = None,
        spontaneous_candidates: Optional[list[Any]] = None,
        binding_store: Any = None,
        current_emotions: Optional[dict[str, float]] = None,
        mood_valence: float = 0.0,
        mood_arousal: float = 0.3,
        max_bias_strength: float = 0.15,
        tick_number: int = 0,
    ) -> ReturnResult:
        """4段パイプラインの帰還処理を実行する。

        Args:
            multi_path_candidates: 外部入力トリガー型想起の直近候補リスト (READ-ONLY)
            spontaneous_candidates: 内部状態駆動型想起の直近候補リスト (READ-ONLY)
            binding_store: 感情記憶紐づけ構造 (READ-ONLY)
            current_emotions: 現在の感情ベクトル (READ-ONLY, label -> value)
            mood_valence: 現在のムード感情価
            mood_arousal: 現在のムード覚醒度
            max_bias_strength: 価値方向性の最大バイアス強度
            tick_number: 現在のティック番号

        Returns:
            ReturnResult with emotion_deltas (label -> delta)
        """
        # Safety valve 4: Same-tick prevention
        if tick_number == self._state.last_applied_tick and tick_number >= 0:
            return ReturnResult()

        self._state.cycle_count += 1

        mpr_candidates = multi_path_candidates or []
        sr_candidates = spontaneous_candidates or []
        emotions = current_emotions or {}

        # Empty candidates -> no return
        if not mpr_candidates and not sr_candidates:
            self._state.last_applied_tick = tick_number
            return ReturnResult()

        # No binding store -> no traces to look up
        if binding_store is None:
            self._state.last_applied_tick = tick_number
            return ReturnResult(
                total_candidates_processed=len(mpr_candidates) + len(sr_candidates),
            )

        # === Stage 1: Collect trace data ===
        mpr_traces = _collect_trace_data(mpr_candidates, binding_store, "multi_path")
        sr_traces = _collect_trace_data(sr_candidates, binding_store, "spontaneous")

        # Both systems are equal; no priority between them
        all_traces = mpr_traces + sr_traces

        if not all_traces:
            self._state.last_applied_tick = tick_number
            return ReturnResult(
                total_candidates_processed=len(mpr_candidates) + len(sr_candidates),
            )

        # === Stage 2: Derive return amounts ===
        derived = _derive_return_amounts(
            all_traces, emotions, mood_valence, mood_arousal, self._config,
        )

        if not derived:
            self._state.last_applied_tick = tick_number
            return ReturnResult(
                total_candidates_processed=len(mpr_candidates) + len(sr_candidates),
                candidates_with_traces=len(all_traces),
            )

        # === Stage 3: Apply safety valves ===
        modified_entries, total_deltas = _apply_safety_valves(
            derived, self._state.return_history, self._config, max_bias_strength,
        )

        if not total_deltas:
            self._state.last_applied_tick = tick_number
            return ReturnResult(
                total_candidates_processed=len(mpr_candidates) + len(sr_candidates),
                candidates_with_traces=len(all_traces),
            )

        # === Stage 4: Apply and record ===
        clamped_deltas, new_records = _apply_and_record(
            total_deltas, modified_entries, emotions,
            mood_valence, mood_arousal, tick_number,
            self._state, self._config,
        )

        # Update state: add new records to history
        self._state.return_history.extend(new_records)

        # FIFO trimming
        window = self._config.history_window_size
        if len(self._state.return_history) > window:
            self._state.return_history = self._state.return_history[-window:]

        # Update direction tracking after records are added
        if new_records:
            self._update_direction_tracking(clamped_deltas)

        # Record last applied tick
        self._state.last_applied_tick = tick_number

        logger.debug(
            "Memory emotion return: cycle=%d, tick=%d, candidates=%d, "
            "traces=%d, deltas=%s, records=%d",
            self._state.cycle_count, tick_number,
            len(mpr_candidates) + len(sr_candidates),
            len(all_traces),
            {k: round(v, 4) for k, v in clamped_deltas.items()},
            len(new_records),
        )

        return ReturnResult(
            emotion_deltas=clamped_deltas,
            records_created=len(new_records),
            total_candidates_processed=len(mpr_candidates) + len(sr_candidates),
            candidates_with_traces=len(all_traces),
        )

    def _update_direction_tracking(self, deltas: dict[str, float]) -> None:
        """帰還方向の連続性追跡を更新する。

        帰還量の合計の符号から帰還方向（正/負/中立）を判定し、
        減衰付き同方向連続カウントを更新する。
        逆方向の帰還でカウントをリセットする。

        enrichmentに露出しない。想起モジュールへの逆流なし。
        """
        # Determine direction from sum of all deltas
        total_delta = sum(deltas.values())
        if total_delta > 1e-6:
            current_direction = "positive"
        elif total_delta < -1e-6:
            current_direction = "negative"
        else:
            current_direction = "neutral"

        prev_direction = self._state.last_direction_label

        if current_direction == "neutral":
            # Neutral does not contribute to consecutive counts
            self._state.last_direction_label = current_direction
            return

        if current_direction == "positive":
            if prev_direction == "positive":
                # Same direction: apply freshness decay to existing count, then add 1
                self._state.direction_consecutive_count_positive = (
                    self._state.direction_consecutive_count_positive
                    * self._config.direction_freshness_decay
                    + 1.0
                )
            else:
                # Direction changed: reset negative, start new positive count
                self._state.direction_consecutive_count_negative = 0.0
                self._state.direction_consecutive_count_positive = 1.0
        elif current_direction == "negative":
            if prev_direction == "negative":
                # Same direction: apply freshness decay to existing count, then add 1
                self._state.direction_consecutive_count_negative = (
                    self._state.direction_consecutive_count_negative
                    * self._config.direction_freshness_decay
                    + 1.0
                )
            else:
                # Direction changed: reset positive, start new negative count
                self._state.direction_consecutive_count_positive = 0.0
                self._state.direction_consecutive_count_negative = 1.0

        self._state.last_direction_label = current_direction

    def get_tracking_speed_modulation(
        self,
        current_tracking_speed_valence: float = 0.10,
        current_tracking_speed_arousal: float = 0.10,
    ) -> tuple[float, float]:
        """帰還方向の連続性から追従速度への変調量を導出する。

        変調量は現在の追従速度に対する比率で上限制限される。
        増加方向のみ変調し、減少方向は行わない（固定化助長防止）。
        enrichmentに直接露出しない。

        Args:
            current_tracking_speed_valence: 現在のvalence追従速度
            current_tracking_speed_arousal: 現在のarousal追従速度

        Returns:
            (valence_modulation, arousal_modulation) — 各々 >= 0.0
        """
        # Use the maximum of positive and negative consecutive counts
        max_count = max(
            self._state.direction_consecutive_count_positive,
            self._state.direction_consecutive_count_negative,
        )

        if max_count <= 0.0:
            return 0.0, 0.0

        # Derive raw modulation from consecutive count
        raw_modulation = max_count * self._config.tracking_speed_modulation_scale

        # Cap at ratio of current tracking speed (safety valve 1)
        cap = self._config.tracking_speed_modulation_ratio_cap

        v_mod = min(raw_modulation, current_tracking_speed_valence * cap)
        a_mod = min(raw_modulation, current_tracking_speed_arousal * cap)

        # Only increase direction (safety valve 6: no decrease)
        v_mod = max(0.0, v_mod)
        a_mod = max(0.0, a_mod)

        return v_mod, a_mod

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

def create_memory_emotion_return(
    config: Optional[MemoryEmotionReturnConfig] = None,
) -> MemoryEmotionReturnProcessor:
    """MemoryEmotionReturnProcessor のファクトリ関数。"""
    return MemoryEmotionReturnProcessor(config=config)
