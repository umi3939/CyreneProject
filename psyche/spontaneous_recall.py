"""
psyche/spontaneous_recall.py - 記憶の自発的想起（非参照型想起）

外部入力が存在しない期間においても、内部状態の変動のみを契機として
記憶が浮上する経路を提供する。

設計原則 (design_spontaneous_recall.md 準拠):
- 記憶の内容を変更・加工・評価しない
- 想起結果に基づいて判断・行動・方針を確定しない
- 特定の記憶を恒常的に優先しない
- 想起された記憶の意味・正誤・有用性を判定しない
- 忘却処理の進行に直接影響を与えない（参照頻度を通知しない）
- 既存の外部入力トリガー型想起を置き換えない・上書きしない
- 外部入力の有無を判定して条件分岐を行わない（常に内部状態のみを参照）
- 感情パイプラインへ逆流しない（想起結果が感情状態を直接変更する経路を持たない）
- 既存の自発起動構造の判定に介入しない
- 3経路の候補は経路ラベル付きで等価に並列
- 出力はenrichmentの参照情報としてのみ提供

3経路（外部入力トリガー型想起とは完全に独立）:
1. 感情変動連想経路: 感情の変動（差分）を起点とし、記憶の感情痕跡との近接度で候補列挙
2. 動機連想経路: 動機断片の記述と記憶のトピック属性の重複度で候補列挙
3. 揺らぎ連想経路: 連続性負荷が一定水準を超えているとき、時間的に離散した記憶を候補列挙

安全弁:
1. 経路間等価性（経路ごとの候補上限を同一に設定）
2. ルーミネーション防止（直近想起履歴によるスライディングウィンドウ抑制）
3. 顕著性バイアス抑制（感情痕跡が弱い記憶の一定割合混入）
4. 忘却処理との分離（参照頻度非通知、INVISIBLE記憶除外）
5. 感情パイプラインへの逆流遮断（出力は参照情報のみ）
6. 判断系への非接続（enrichmentの参照情報として等価列挙のみ）
7. 外部入力トリガー型想起との経路分離（経路ラベルで出力元を区別）

Usage::

    recall = create_spontaneous_recall()
    result = recall.process(
        unified_units=units,
        binding_store=binding_store,
        forgetting_state=forgetting_state,
        emotion_snapshot=emotion_snapshot,
        prev_emotion_snapshot=prev_emotion_snapshot,
        motive_store=motive_store,
        strain_state=strain_state,
        direction_vectors=direction_vectors,
        temporal_snapshot=temporal_snapshot,
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class SpontaneousRecallPathLabel(Enum):
    """自発的想起経路ラベル。3経路を等価に識別する。
    外部入力トリガー型想起の経路ラベルとは異なる値を持つ。"""
    EMOTION_DELTA = "emotion_delta"      # 感情変動連想
    MOTIVE_ASSOC = "motive_assoc"        # 動機連想
    FLUCTUATION_ASSOC = "fluctuation_assoc"  # 揺らぎ連想


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SpontaneousRecallCandidate:
    """自発的想起候補。経路ラベル付きの単一候補。

    全候補は等価であり、重み・スコア・優先度を持たない。
    """
    candidate_id: str = field(default_factory=_gen_id)
    unit_id: str = ""
    source_id: str = ""
    summary: str = ""
    path_label: str = SpontaneousRecallPathLabel.EMOTION_DELTA.value
    recall_timestamp: float = field(default_factory=time.time)
    input_snapshot: str = ""  # 想起時の内部状態断面の要約

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "unit_id": self.unit_id,
            "source_id": self.source_id,
            "summary": self.summary,
            "path_label": self.path_label,
            "recall_timestamp": self.recall_timestamp,
            "input_snapshot": self.input_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpontaneousRecallCandidate":
        return cls(
            candidate_id=data.get("candidate_id", _gen_id()),
            unit_id=data.get("unit_id", ""),
            source_id=data.get("source_id", ""),
            summary=data.get("summary", ""),
            path_label=data.get("path_label", SpontaneousRecallPathLabel.EMOTION_DELTA.value),
            recall_timestamp=data.get("recall_timestamp", time.time()),
            input_snapshot=data.get("input_snapshot", ""),
        )


@dataclass
class SpontaneousRecallPathStatistics:
    """経路別統計の断面記述。

    記述のみであり、経路の重み付けには使用しない。
    前サイクルの値を保持しないため、ある経路の過去の実績が将来の経路選択に
    影響を与える構造はない。
    """
    emotion_delta_count: int = 0
    motive_assoc_count: int = 0
    fluctuation_assoc_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_delta_count": self.emotion_delta_count,
            "motive_assoc_count": self.motive_assoc_count,
            "fluctuation_assoc_count": self.fluctuation_assoc_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpontaneousRecallPathStatistics":
        return cls(
            emotion_delta_count=data.get("emotion_delta_count", 0),
            motive_assoc_count=data.get("motive_assoc_count", 0),
            fluctuation_assoc_count=data.get("fluctuation_assoc_count", 0),
        )


# =============================================================================
# Internal State Snapshots (段階1の入力)
# =============================================================================

@dataclass
class InternalEmotionSnapshot:
    """感情状態断面。READ-ONLY参照。"""
    emotions: dict[str, float] = field(default_factory=dict)  # label -> intensity
    mood_valence: float = 0.0
    dominant_emotion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotions": dict(self.emotions),
            "mood_valence": self.mood_valence,
            "dominant_emotion": self.dominant_emotion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalEmotionSnapshot":
        return cls(
            emotions=dict(data.get("emotions", {})),
            mood_valence=data.get("mood_valence", 0.0),
            dominant_emotion=data.get("dominant_emotion", ""),
        )


@dataclass
class InternalStateCrossSections:
    """段階1で抽出される内部状態断面群。

    各断面は数値的な記述であり、解釈・評価を含まない。
    """
    # 感情変動断面: 現在の感情状態の変動量（前回からの差分の絶対値的記述）
    emotion_delta: float = 0.0
    emotion_delta_labels: dict[str, float] = field(default_factory=dict)  # label -> delta

    # 動機圧力断面: 動機断片群の現在の圧力的な量
    motive_pressure: float = 0.0
    motive_descriptions: list[str] = field(default_factory=list)

    # 方向変動断面: 方向ベクトルの変動量
    direction_delta: float = 0.0

    # 連続性揺らぎ断面: 連続性負荷の現在水準
    continuity_strain_level: float = 0.0

    # 時間推移断面: 経過時間に基づく段階的値
    temporal_stage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_delta": self.emotion_delta,
            "emotion_delta_labels": dict(self.emotion_delta_labels),
            "motive_pressure": self.motive_pressure,
            "motive_descriptions": list(self.motive_descriptions),
            "direction_delta": self.direction_delta,
            "continuity_strain_level": self.continuity_strain_level,
            "temporal_stage": self.temporal_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalStateCrossSections":
        return cls(
            emotion_delta=data.get("emotion_delta", 0.0),
            emotion_delta_labels=dict(data.get("emotion_delta_labels", {})),
            motive_pressure=data.get("motive_pressure", 0.0),
            motive_descriptions=list(data.get("motive_descriptions", [])),
            direction_delta=data.get("direction_delta", 0.0),
            continuity_strain_level=data.get("continuity_strain_level", 0.0),
            temporal_stage=data.get("temporal_stage", 0.0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class SpontaneousRecallState:
    """自発的想起の永続化可能な状態。"""
    # 前回の内部状態断面 (変動量計算用、毎サイクル上書き)
    prev_cross_sections: InternalStateCrossSections = field(
        default_factory=InternalStateCrossSections
    )

    # 現サイクルの想起候補リスト (毎サイクル全件入れ替え)
    current_candidates: list[SpontaneousRecallCandidate] = field(default_factory=list)

    # ルーミネーション防止用の想起履歴 (スライディングウィンドウ)
    recent_recall_history: list[str] = field(default_factory=list)

    # 経路別統計 (毎サイクル更新、過去の統計を保持しない)
    path_stats: SpontaneousRecallPathStatistics = field(
        default_factory=SpontaneousRecallPathStatistics
    )

    # サイクルカウンタ
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prev_cross_sections": self.prev_cross_sections.to_dict(),
            "current_candidates": [c.to_dict() for c in self.current_candidates],
            "recent_recall_history": list(self.recent_recall_history),
            "path_stats": self.path_stats.to_dict(),
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpontaneousRecallState":
        return cls(
            prev_cross_sections=InternalStateCrossSections.from_dict(
                data.get("prev_cross_sections", {})
            ),
            current_candidates=[
                SpontaneousRecallCandidate.from_dict(c)
                for c in data.get("current_candidates", [])
            ],
            recent_recall_history=list(data.get("recent_recall_history", [])),
            path_stats=SpontaneousRecallPathStatistics.from_dict(
                data.get("path_stats", {})
            ),
            cycle_count=data.get("cycle_count", 0),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SpontaneousRecallConfig:
    """設定。"""
    # 各経路の候補上限（全経路で同一値: 安全弁1 経路間等価性）
    per_path_limit: int = 5

    # ルーミネーション防止のスライディングウィンドウサイズ（サイクル数分の履歴保持）
    rumination_window_size: int = 30

    # ルーミネーション抑制閾値（この回数以上想起された記憶を抑制対象とする）
    rumination_suppression_threshold: int = 3

    # 顕著性バイアス抑制: 感情痕跡が弱い記憶の強制混入割合 (0.0-1.0)
    weak_trace_ratio: float = 0.2

    # 感情連想の感情痕跡強度閾値（これ以下を「弱い」と見なす）
    weak_trace_threshold: float = 0.3

    # 揺らぎ連想の連続性負荷閾値（これ以上で経路が活性化）
    strain_threshold: float = 0.3

    # 揺らぎ連想の時間的距離下限（秒）: これより近い記憶は揺らぎ経路では候補にしない
    # （時間近接経路では拾われない距離の記憶を候補とするため）
    fluctuation_min_distance: float = 3600.0  # 1時間

    # enrichment に含める候補数
    enrichment_candidate_count: int = 9  # 3経路 x 3件

    # テキスト断片の最大長
    summary_snippet_length: int = 80


# =============================================================================
# Stage 1: 内部状態断面の抽出
# =============================================================================

def extract_cross_sections(
    emotion_snapshot: Optional[InternalEmotionSnapshot] = None,
    prev_emotion_snapshot: Optional[InternalEmotionSnapshot] = None,
    motive_store: Any = None,
    strain_state: Any = None,
    direction_vectors: Any = None,
    temporal_snapshot: Any = None,
) -> InternalStateCrossSections:
    """内部状態から断面群を抽出する。

    外部入力（知覚内容）は一切参照しない。
    各断面は数値的な記述であり、解釈・評価を含まない。
    """
    sections = InternalStateCrossSections()

    # 感情変動断面: 前回からの差分の絶対値的記述
    emo = emotion_snapshot or InternalEmotionSnapshot()
    prev_emo = prev_emotion_snapshot or InternalEmotionSnapshot()

    delta_labels: dict[str, float] = {}
    total_delta = 0.0

    # 現在の感情ラベルの変動
    all_labels = set(list(emo.emotions.keys()) + list(prev_emo.emotions.keys()))
    for label in all_labels:
        curr_val = emo.emotions.get(label, 0.0)
        prev_val = prev_emo.emotions.get(label, 0.0)
        delta = abs(curr_val - prev_val)
        if delta > 0.01:
            delta_labels[label] = delta
            total_delta += delta

    # ムード差分
    mood_delta = abs(emo.mood_valence - prev_emo.mood_valence)
    total_delta += mood_delta

    sections.emotion_delta = _clamp(total_delta)
    sections.emotion_delta_labels = delta_labels

    # 動機圧力断面: 動機断片群の圧力的な量
    if motive_store is not None:
        entries = getattr(motive_store, "entries", []) or []
        descriptions: list[str] = []
        pressure_sum = 0.0
        count = 0
        for entry in entries:
            strength = getattr(entry, "strength", 0.0)
            if isinstance(strength, (int, float)) and strength > 0:
                pressure_sum += strength
                count += 1
            desc = getattr(entry, "description", "")
            if desc:
                descriptions.append(desc[:40])
        sections.motive_pressure = _clamp(pressure_sum / max(1, count)) if count > 0 else 0.0
        sections.motive_descriptions = descriptions[:10]

    # 方向変動断面: 方向ベクトルの変動量
    if direction_vectors is not None:
        vectors = None
        if hasattr(direction_vectors, "vectors"):
            vectors = direction_vectors.vectors
        elif hasattr(direction_vectors, "__iter__"):
            vectors = direction_vectors
        if vectors:
            magnitudes = []
            for v in vectors:
                mag = getattr(v, "magnitude", 0.0)
                if isinstance(mag, (int, float)):
                    magnitudes.append(abs(mag))
            if magnitudes:
                sections.direction_delta = _clamp(max(magnitudes))

    # 連続性揺らぎ断面: 連続性負荷の現在水準
    if strain_state is not None:
        level = getattr(strain_state, "level", None)
        if level is not None:
            level_val = getattr(level, "value", str(level))
            strain_map = {
                "none": 0.0,
                "low": 0.2,
                "moderate": 0.5,
                "high": 0.8,
                "severe": 1.0,
            }
            sections.continuity_strain_level = strain_map.get(level_val, 0.0)

    # 時間推移断面: 経過時間に基づく段階的値
    if temporal_snapshot is not None:
        tick_count = getattr(temporal_snapshot, "tick_count", 0)
        if isinstance(tick_count, (int, float)) and tick_count > 0:
            # 段階的な値: ティック数に応じた緩やかなスケーリング
            sections.temporal_stage = _clamp(min(1.0, tick_count / 100.0))

    return sections


# =============================================================================
# Stage 2: 内部状態に基づく想起候補列挙
# =============================================================================

def _recall_emotion_delta(
    units: list[Any],
    cross_sections: InternalStateCrossSections,
    binding_store: Any,
    config: SpontaneousRecallConfig,
    now: float,
) -> list[SpontaneousRecallCandidate]:
    """感情変動連想経路: 感情の変動（差分）を起点とし、
    記憶の感情痕跡との近接度で候補を列挙する。

    外部入力トリガー型想起の感情連想経路とは異なり、
    「感情の変動」（差分）を起点とする。
    静的な感情値ではなく、変動の発生自体を契機とする。
    """
    cfg = config

    # 上限0以下なら候補なし
    if cfg.per_path_limit <= 0:
        return []

    delta = cross_sections.emotion_delta
    delta_labels = cross_sections.emotion_delta_labels

    # 感情変動がほぼゼロなら候補なし
    if delta < 0.05 and not delta_labels:
        return []

    # BindingStoreからの痕跡マップ構築
    trace_map: dict[str, list[tuple[str, float, float]]] = {}
    if binding_store is not None:
        bindings = getattr(binding_store, "bindings", ()) or ()
        for binding in bindings:
            memory_key = getattr(binding, "memory_key", "")
            if not memory_key:
                continue
            traces = getattr(binding, "traces", ()) or ()
            trace_list: list[tuple[str, float, float]] = []
            for trace in traces:
                label = getattr(trace, "emotion_label", "")
                intensity = getattr(trace, "intensity", 0.0)
                valence = getattr(trace, "valence", 0.0)
                if label:
                    trace_list.append((label, intensity, valence))
            if trace_list:
                trace_map[memory_key] = trace_list

    scored: list[tuple[float, Any]] = []
    weak_scored: list[tuple[float, Any]] = []

    for unit in units:
        unit_id = getattr(unit, "unit_id", "")
        emotional_valence = getattr(unit, "emotional_valence", 0.0)
        emotional_label = getattr(unit, "emotional_label", "")

        proximity = 0.0

        # 変動したラベルとの近接
        if emotional_label and emotional_label in delta_labels:
            proximity += delta_labels[emotional_label] * 0.6

        # BindingStoreの痕跡との変動近接
        binding_traces = trace_map.get(unit_id, [])
        for trace_label, trace_intensity, _trace_valence in binding_traces:
            if trace_label in delta_labels:
                proximity += delta_labels[trace_label] * trace_intensity * 0.3

        # 全体的な感情変動量との相関
        if abs(emotional_valence) > 0.01 and delta > 0.1:
            proximity += delta * abs(emotional_valence) * 0.1

        if proximity <= 0:
            continue

        # 弱い痕跡 / 強い痕跡の分類
        max_trace_intensity = 0.0
        for _, trace_intensity, _ in binding_traces:
            if trace_intensity > max_trace_intensity:
                max_trace_intensity = trace_intensity

        if (max_trace_intensity <= cfg.weak_trace_threshold
                and abs(emotional_valence) <= cfg.weak_trace_threshold):
            weak_scored.append((proximity, unit))
        else:
            scored.append((proximity, unit))

    # 安全弁3: 顕著性バイアス抑制 - 弱い痕跡の一定割合を強制混入
    limit = cfg.per_path_limit
    weak_slots = max(1, int(limit * cfg.weak_trace_ratio))
    strong_slots = limit - weak_slots

    scored.sort(key=lambda x: x[0], reverse=True)
    weak_scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[Any] = []
    for _, unit in scored[:strong_slots]:
        selected.append(unit)
    for _, unit in weak_scored[:weak_slots]:
        selected.append(unit)

    # 枠が埋まらなかった場合は補充
    if len(selected) < limit:
        remaining_weak = weak_scored[weak_slots:]
        remaining_strong = scored[strong_slots:]
        for _, unit in remaining_weak + remaining_strong:
            if len(selected) >= limit:
                break
            if unit not in selected:
                selected.append(unit)

    # 変動ラベルの上位を入力スナップショットに
    top_deltas = sorted(delta_labels.items(), key=lambda x: x[1], reverse=True)[:3]
    input_desc = "delta=" + ",".join(f"{l}:{v:.2f}" for l, v in top_deltas) if top_deltas else f"delta={delta:.2f}"

    candidates: list[SpontaneousRecallCandidate] = []
    for unit in selected:
        summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
        candidates.append(SpontaneousRecallCandidate(
            unit_id=getattr(unit, "unit_id", ""),
            source_id=getattr(unit, "source_id", ""),
            summary=summary,
            path_label=SpontaneousRecallPathLabel.EMOTION_DELTA.value,
            recall_timestamp=now,
            input_snapshot=input_desc,
        ))

    return candidates


def _recall_motive_assoc(
    units: list[Any],
    cross_sections: InternalStateCrossSections,
    config: SpontaneousRecallConfig,
    now: float,
) -> list[SpontaneousRecallCandidate]:
    """動機連想経路: 動機断片の記述と記憶のトピック属性の重複度で候補を列挙する。"""
    cfg = config

    # 上限0以下なら候補なし
    if cfg.per_path_limit <= 0:
        return []

    if cross_sections.motive_pressure < 0.01 and not cross_sections.motive_descriptions:
        return []

    # 動機記述からキーワードを抽出
    motive_keywords: set[str] = set()
    for desc in cross_sections.motive_descriptions:
        for word in desc.lower().split():
            if len(word) >= 2:
                motive_keywords.add(word)

    if not motive_keywords:
        return []

    scored: list[tuple[float, Any]] = []

    for unit in units:
        unit_topics = getattr(unit, "topics", []) or []
        unit_topic_set = {t.lower() for t in unit_topics}
        unit_summary = (getattr(unit, "summary", "") or "").lower()

        relevance = 0.0

        # トピックとの重複
        if unit_topic_set and motive_keywords:
            overlap = len(motive_keywords & unit_topic_set)
            if overlap > 0:
                relevance += min(1.0, overlap / max(1, len(motive_keywords))) * 0.7

        # サマリーとの部分一致
        if unit_summary and motive_keywords:
            match_count = sum(1 for kw in motive_keywords if kw in unit_summary)
            if match_count > 0:
                relevance += min(0.3, match_count * 0.05)

        if relevance > 0:
            scored.append((relevance, unit))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:cfg.per_path_limit]

    input_desc = f"motive_kw={','.join(list(motive_keywords)[:5])}, pressure={cross_sections.motive_pressure:.2f}"

    candidates: list[SpontaneousRecallCandidate] = []
    for _, unit in selected:
        summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
        candidates.append(SpontaneousRecallCandidate(
            unit_id=getattr(unit, "unit_id", ""),
            source_id=getattr(unit, "source_id", ""),
            summary=summary,
            path_label=SpontaneousRecallPathLabel.MOTIVE_ASSOC.value,
            recall_timestamp=now,
            input_snapshot=input_desc,
        ))

    return candidates


def _recall_fluctuation_assoc(
    units: list[Any],
    cross_sections: InternalStateCrossSections,
    config: SpontaneousRecallConfig,
    now: float,
) -> list[SpontaneousRecallCandidate]:
    """揺らぎ連想経路: 連続性負荷が一定水準を超えているとき、
    時間的に離散した（時間近接経路では拾われない距離の）記憶を候補として列挙する。
    連続性の揺らぎが大きい状態では、時間的に遠い記憶が浮上しやすくなる。
    """
    cfg = config

    # 上限0以下なら候補なし
    if cfg.per_path_limit <= 0:
        return []

    # 連続性負荷が閾値未満なら候補なし
    if cross_sections.continuity_strain_level < cfg.strain_threshold:
        return []

    scored: list[tuple[float, Any]] = []

    for unit in units:
        unit_ts = getattr(unit, "timestamp", 0.0)
        if not isinstance(unit_ts, (int, float)) or unit_ts <= 0:
            continue

        distance = abs(now - unit_ts)
        # 時間近接経路では拾われない距離の記憶のみを候補にする
        if distance < cfg.fluctuation_min_distance:
            continue

        # 距離が遠いほど高い（揺らぎが大きいとき遠い記憶が浮上しやすい）
        # 連続性負荷水準に比例してスケーリング
        distance_score = _clamp(
            min(1.0, distance / (cfg.fluctuation_min_distance * 10))
            * cross_sections.continuity_strain_level
        )

        scored.append((distance_score, unit))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:cfg.per_path_limit]

    input_desc = f"strain={cross_sections.continuity_strain_level:.2f}"

    candidates: list[SpontaneousRecallCandidate] = []
    for _, unit in selected:
        summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
        candidates.append(SpontaneousRecallCandidate(
            unit_id=getattr(unit, "unit_id", ""),
            source_id=getattr(unit, "source_id", ""),
            summary=summary,
            path_label=SpontaneousRecallPathLabel.FLUCTUATION_ASSOC.value,
            recall_timestamp=now,
            input_snapshot=input_desc,
        ))

    return candidates


# =============================================================================
# Stage 3: 安全弁の適用
# =============================================================================

def _filter_invisible(
    units: list[Any], forgetting_state: Any,
) -> list[Any]:
    """INVISIBLE記憶を想起候補から除外する。（安全弁4）

    忘却/固定化の段階情報を参照するが、参照頻度には通知しない。
    想起と忘却は単方向の参照関係のみを持つ
    （忘却→想起: 不可視記憶の除外指示。想起→忘却: 経路なし）。
    """
    if forgetting_state is None:
        return list(units)

    invisible_ids: set[str] = set()
    series_index = getattr(forgetting_state, "series_index", []) or []
    for rec in series_index:
        stage = getattr(rec, "forgetting_stage", "active")
        source_id = getattr(rec, "source_id", "")
        if stage == "invisible" and source_id:
            invisible_ids.add(source_id)

    if not invisible_ids:
        return list(units)

    visible: list[Any] = []
    for unit in units:
        source_id = getattr(unit, "source_id", "")
        unit_id = getattr(unit, "unit_id", "")
        if source_id in invisible_ids or unit_id in invisible_ids:
            continue
        visible.append(unit)

    return visible


def _apply_rumination_suppression(
    candidates: list[SpontaneousRecallCandidate],
    recent_recall_history: list[str],
    window_size: int,
    per_path_limit: int,
    suppression_threshold: int,
) -> list[SpontaneousRecallCandidate]:
    """ルーミネーション防止: 直近想起履歴内の記憶の優先度を下げる。（安全弁2）

    抑制は「完全除外」ではなく「候補としての優先度を下げる」方式。
    他に候補が存在しない場合は再選出を許容する。
    これにより、感情A→記憶X想起→記憶Xが再び感情Aと近接→記憶X再想起、
    という循環が構造的に抑制される。
    """
    if not recent_recall_history:
        return candidates

    # 直近ウィンドウ内での想起回数を集計
    max_history = window_size * per_path_limit * 3
    window = recent_recall_history[-max_history:]
    recall_counts: dict[str, int] = {}
    for uid in window:
        recall_counts[uid] = recall_counts.get(uid, 0) + 1

    non_suppressed: list[SpontaneousRecallCandidate] = []
    suppressed: list[SpontaneousRecallCandidate] = []

    for c in candidates:
        if recall_counts.get(c.unit_id, 0) >= suppression_threshold:
            suppressed.append(c)
        else:
            non_suppressed.append(c)

    # 非抑制候補を優先し、抑制候補を末尾に追加（完全除外はしない）
    return non_suppressed + suppressed


# =============================================================================
# Processor
# =============================================================================

class SpontaneousRecallProcessor:
    """記憶の自発的想起プロセッサ。

    4段階の処理構成:
    1. 内部状態断面の抽出
    2. 想起候補の列挙（3経路）
    3. 安全弁の適用
    4. 出力の整形

    出力は参照情報としてのみ流れ、判断・行動・評価を直接引き起こさない。
    忘却処理の参照頻度に通知する経路を持たない。
    感情パイプラインへ逆流する経路を持たない。
    外部入力トリガー型想起とは完全に独立している。
    """

    def __init__(self, config: Optional[SpontaneousRecallConfig] = None):
        self._config = config or SpontaneousRecallConfig()
        self._state = SpontaneousRecallState()

    @property
    def state(self) -> SpontaneousRecallState:
        return self._state

    @state.setter
    def state(self, value: SpontaneousRecallState) -> None:
        self._state = value

    # ─── Main: 4段階の想起処理 ─────────────────────────────────

    def process(
        self,
        unified_units: Optional[list[Any]] = None,
        binding_store: Any = None,
        forgetting_state: Any = None,
        emotion_snapshot: Optional[InternalEmotionSnapshot] = None,
        prev_emotion_snapshot: Optional[InternalEmotionSnapshot] = None,
        motive_store: Any = None,
        strain_state: Any = None,
        direction_vectors: Any = None,
        temporal_snapshot: Any = None,
    ) -> list[SpontaneousRecallCandidate]:
        """4段階の想起処理を実行し、安全弁を適用した候補リストを返す。

        入力:
        - unified_units: 記憶系統統合の共通記述単位リスト
        - binding_store: 感情記憶紐付けのBindingStore
        - forgetting_state: 忘却/固定化のForgettingFixationState
        - emotion_snapshot: 現在の感情断面
        - prev_emotion_snapshot: 前回の感情断面（差分計算用）
        - motive_store: 内的動機のMotiveStore
        - strain_state: 連続性負荷のStrainState
        - direction_vectors: 方向ベクトル
        - temporal_snapshot: 時間認知断面

        Returns:
            3経路の想起候補リスト（経路ラベル付き、等価並列）
        """
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # 使用する前回の感情断面
        effective_prev = prev_emotion_snapshot
        if effective_prev is None:
            # prev_emotion_snapshotが渡されなかった場合、
            # 保存された前回断面を使用する
            effective_prev = InternalEmotionSnapshot.from_dict(
                {
                    "emotions": self._state.prev_cross_sections.emotion_delta_labels,
                    "mood_valence": 0.0,
                    "dominant_emotion": "",
                }
            )
            # 実際には前回の感情スナップショット自体は保存していないので、
            # prev_cross_sections に保存された前回の断面値を通じて差分が計算済み

        # === 段階1: 内部状態断面の抽出 ===
        cross_sections = extract_cross_sections(
            emotion_snapshot=emotion_snapshot,
            prev_emotion_snapshot=effective_prev,
            motive_store=motive_store,
            strain_state=strain_state,
            direction_vectors=direction_vectors,
            temporal_snapshot=temporal_snapshot,
        )

        units = unified_units or []

        # 安全弁4: 忘却処理との分離 - INVISIBLE記憶を除外
        visible_units = _filter_invisible(units, forgetting_state)

        if not visible_units:
            self._state.current_candidates = []
            self._state.path_stats = SpontaneousRecallPathStatistics()
            self._state.prev_cross_sections = cross_sections
            return []

        # === 段階2: 想起候補の列挙（3経路、等価） ===

        # 経路1: 感情変動連想
        emotion_candidates = _recall_emotion_delta(
            visible_units, cross_sections, binding_store, cfg, now,
        )

        # 経路2: 動機連想
        motive_candidates = _recall_motive_assoc(
            visible_units, cross_sections, cfg, now,
        )

        # 経路3: 揺らぎ連想
        fluctuation_candidates = _recall_fluctuation_assoc(
            visible_units, cross_sections, cfg, now,
        )

        # === 段階3: 安全弁の適用 ===

        # 安全弁1: 経路間等価性は各経路内でper_path_limit適用済み

        # 安全弁2: ルーミネーション防止
        all_candidates = emotion_candidates + motive_candidates + fluctuation_candidates
        all_candidates = _apply_rumination_suppression(
            all_candidates,
            self._state.recent_recall_history,
            cfg.rumination_window_size,
            cfg.per_path_limit,
            cfg.rumination_suppression_threshold,
        )

        # === 段階4: 出力の整形 ===

        # 状態更新
        self._state.current_candidates = all_candidates
        self._state.path_stats = SpontaneousRecallPathStatistics(
            emotion_delta_count=len([
                c for c in all_candidates
                if c.path_label == SpontaneousRecallPathLabel.EMOTION_DELTA.value
            ]),
            motive_assoc_count=len([
                c for c in all_candidates
                if c.path_label == SpontaneousRecallPathLabel.MOTIVE_ASSOC.value
            ]),
            fluctuation_assoc_count=len([
                c for c in all_candidates
                if c.path_label == SpontaneousRecallPathLabel.FLUCTUATION_ASSOC.value
            ]),
        )

        # ルーミネーション防止履歴の更新
        for c in all_candidates:
            self._state.recent_recall_history.append(c.unit_id)
        # スライディングウィンドウのトリミング
        max_history = cfg.rumination_window_size * cfg.per_path_limit * 3
        if len(self._state.recent_recall_history) > max_history:
            self._state.recent_recall_history = self._state.recent_recall_history[-max_history:]

        # 前回断面の保存（次サイクルの変動量計算用）
        self._state.prev_cross_sections = cross_sections

        logger.debug(
            "Spontaneous recall: cycle=%d, emotion_delta=%d, motive_assoc=%d, "
            "fluctuation_assoc=%d, total=%d",
            self._state.cycle_count,
            self._state.path_stats.emotion_delta_count,
            self._state.path_stats.motive_assoc_count,
            self._state.path_stats.fluctuation_assoc_count,
            len(all_candidates),
        )

        return all_candidates

    # ─── 参照情報の提供 ──────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        経路ラベル付きの候補一覧として、他のenrichment項目と同列に記述される。
        特定の経路を強調・選別しない。
        出力は参照情報のみ。判断・行動・評価を直接引き起こさない。
        感情パイプラインへの逆流経路を持たない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state
        cfg = self._config

        candidates = st.current_candidates[:cfg.enrichment_candidate_count]

        entries: list[dict[str, Any]] = []
        for c in candidates:
            entries.append({
                "path": c.path_label,
                "summary": c.summary,
                "unit_id": c.unit_id,
            })

        summary_text = get_spontaneous_recall_summary(st)

        return {
            "candidate_count": len(st.current_candidates),
            "path_stats": st.path_stats.to_dict(),
            "entries": entries,
            "summary_text": summary_text,
        }

    def get_recall_candidates(self) -> list[SpontaneousRecallCandidate]:
        """READ-ONLY候補リストを返す。

        想起候補は参照情報としてのみ流れ、判断・行動を確定しない。
        """
        return list(self._state.current_candidates)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "cycle_count": st.cycle_count,
            "candidate_count": len(st.current_candidates),
            "path_stats": st.path_stats.to_dict(),
            "history_length": len(st.recent_recall_history),
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_spontaneous_recall_summary(state: SpontaneousRecallState) -> str:
    """自発的想起状態の要約（enrichment用）。

    全経路を等価に列挙する。特定の経路を強調しない。
    評価判定・行動指示を含まない。
    """
    if not state.current_candidates and state.cycle_count == 0:
        return "自発想起: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    stats = state.path_stats
    if stats.emotion_delta_count > 0:
        parts.append(f"感情変動連想={stats.emotion_delta_count}")
    if stats.motive_assoc_count > 0:
        parts.append(f"動機連想={stats.motive_assoc_count}")
    if stats.fluctuation_assoc_count > 0:
        parts.append(f"揺らぎ連想={stats.fluctuation_assoc_count}")

    total = len(state.current_candidates)
    if total > 0:
        parts.append(f"候補合計={total}")
    else:
        parts.append("候補=0")

    return " ".join(parts) if parts else "自発想起: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_spontaneous_recall(
    config: Optional[SpontaneousRecallConfig] = None,
) -> SpontaneousRecallProcessor:
    """SpontaneousRecallProcessor のファクトリ関数。"""
    return SpontaneousRecallProcessor(config=config)
