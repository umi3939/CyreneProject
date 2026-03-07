"""
psyche/other_model_real_feed.py - 他者モデルへのリアルフィード統合

実対話由来の観測断片（発話反応・応答間隔・話題遷移・感情トーン等）を
抽出・正規化して既存の other_model_input_supply パイプラインを強化する。

設計原則 (design_other_model_real_feed.md 準拠):
- 他者の意図や性質を断定しない
- 単回反応を恒常特性へ昇格させない
- 推定結果のみで応答方針を確定しない
- 外部評価軸を導入して他者を分類しない
- 競合する観測は排除せず並立保持して推定側に揺らぎ情報として渡す
- 供給候補が単一解釈へ収束した場合は競合観測を補充し複線入力へ戻す
- 観測更新が停滞した場合は鮮度低下を反映して過去観測の支配を緩和する
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

class ObservationFragmentType(Enum):
    """観測断片の種類（8値）。"""
    SPEECH_REACTION = "speech_reaction"
    RESPONSE_INTERVAL = "response_interval"
    TOPIC_TRANSITION = "topic_transition"
    EMOTIONAL_TONE = "emotional_tone"
    CONTINUED_ENGAGEMENT = "continued_engagement"
    REJECTION_ACCEPTANCE = "rejection_acceptance"
    CONTEXT_ALIGNMENT = "context_alignment"
    RECENT_HISTORY = "recent_history"


class FragmentFreshness(Enum):
    """観測断片の鮮度。"""
    FRESH = "fresh"
    RECENT = "recent"
    AGING = "aging"
    STALE = "stale"
    FADED = "faded"


class AlignmentStatus(Enum):
    """観測単位の整合状態。"""
    ALIGNED = "aligned"
    PARTIAL = "partial"
    UNALIGNED = "unaligned"
    UNKNOWN = "unknown"


class ConflictStatus(Enum):
    """観測単位の競合状態。"""
    NONE = "none"
    PARALLEL = "parallel"
    CONVERGENCE_RISK = "convergence_risk"


# =============================================================================
# Dataclasses
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ObservationFragment:
    """観測断片。8種の抽出関数から生成される。"""
    fragment_id: str = ""
    type: ObservationFragmentType = ObservationFragmentType.SPEECH_REACTION
    source_description: str = ""
    value: float = 0.5
    text_hint: str = ""
    freshness: FragmentFreshness = FragmentFreshness.FRESH
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.fragment_id:
            self.fragment_id = _gen_id()
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "type": self.type.value,
            "source_description": self.source_description,
            "value": self.value,
            "text_hint": self.text_hint,
            "freshness": self.freshness.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationFragment:
        ftype = ObservationFragmentType.SPEECH_REACTION
        try:
            ftype = ObservationFragmentType(data.get("type", "speech_reaction"))
        except ValueError:
            pass
        fresh = FragmentFreshness.FRESH
        try:
            fresh = FragmentFreshness(data.get("freshness", "fresh"))
        except ValueError:
            pass
        return cls(
            fragment_id=data.get("fragment_id", _gen_id()),
            type=ftype,
            source_description=data.get("source_description", ""),
            value=data.get("value", 0.5),
            text_hint=data.get("text_hint", ""),
            freshness=fresh,
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class ObservationUnit:
    """正規化された観測単位。複数断片から統合されたもの。"""
    unit_id: str = ""
    source_fragment_ids: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    description: str = ""
    value: float = 0.5
    freshness: FragmentFreshness = FragmentFreshness.FRESH
    alignment: AlignmentStatus = AlignmentStatus.UNKNOWN
    conflict_status: ConflictStatus = ConflictStatus.NONE
    competing_unit_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.unit_id:
            self.unit_id = _gen_id()
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_fragment_ids": self.source_fragment_ids,
            "source_types": self.source_types,
            "description": self.description,
            "value": self.value,
            "freshness": self.freshness.value,
            "alignment": self.alignment.value,
            "conflict_status": self.conflict_status.value,
            "competing_unit_ids": self.competing_unit_ids,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationUnit:
        fresh = FragmentFreshness.FRESH
        try:
            fresh = FragmentFreshness(data.get("freshness", "fresh"))
        except ValueError:
            pass
        align = AlignmentStatus.UNKNOWN
        try:
            align = AlignmentStatus(data.get("alignment", "unknown"))
        except ValueError:
            pass
        conflict = ConflictStatus.NONE
        try:
            conflict = ConflictStatus(data.get("conflict_status", "none"))
        except ValueError:
            pass
        return cls(
            unit_id=data.get("unit_id", _gen_id()),
            source_fragment_ids=data.get("source_fragment_ids", []),
            source_types=data.get("source_types", []),
            description=data.get("description", ""),
            value=data.get("value", 0.5),
            freshness=fresh,
            alignment=align,
            conflict_status=conflict,
            competing_unit_ids=data.get("competing_unit_ids", []),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class ConflictRecord:
    """対立する観測単位のペア記録。"""
    conflict_id: str = ""
    unit_id_a: str = ""
    unit_id_b: str = ""
    conflict_aspect: str = ""
    severity: float = 0.0

    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "unit_id_a": self.unit_id_a,
            "unit_id_b": self.unit_id_b,
            "conflict_aspect": self.conflict_aspect,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConflictRecord:
        return cls(
            conflict_id=data.get("conflict_id", _gen_id()),
            unit_id_a=data.get("unit_id_a", ""),
            unit_id_b=data.get("unit_id_b", ""),
            conflict_aspect=data.get("conflict_aspect", ""),
            severity=data.get("severity", 0.0),
        )


@dataclass
class FeedHistoryEntry:
    """投入履歴の1エントリ。"""
    unit_ids: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    cycle_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_ids": self.unit_ids,
            "source_types": self.source_types,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedHistoryEntry:
        return cls(
            unit_ids=data.get("unit_ids", []),
            source_types=data.get("source_types", []),
            timestamp=data.get("timestamp", 0.0),
            cycle_id=data.get("cycle_id", 0),
        )


@dataclass
class HoldbackEntry:
    """未投入保留の1エントリ。"""
    unit_id: str = ""
    source_type: str = ""
    value: float = 0.5
    reason: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_type": self.source_type,
            "value": self.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoldbackEntry:
        return cls(
            unit_id=data.get("unit_id", ""),
            source_type=data.get("source_type", ""),
            value=data.get("value", 0.5),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class RealFeedConfig:
    """リアルフィード処理の設定パラメータ。"""
    max_fragments_per_type: int = 5
    max_observation_units: int = 20
    max_output_units: int = 10
    single_type_dominance_cap: float = 0.4
    freshness_decay_rate: float = 0.05
    stale_threshold: float = 0.15
    recent_series_suppression_count: int = 3
    convergence_inject_threshold: int = 1
    stagnation_cycle_threshold: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_fragments_per_type": self.max_fragments_per_type,
            "max_observation_units": self.max_observation_units,
            "max_output_units": self.max_output_units,
            "single_type_dominance_cap": self.single_type_dominance_cap,
            "freshness_decay_rate": self.freshness_decay_rate,
            "stale_threshold": self.stale_threshold,
            "recent_series_suppression_count": self.recent_series_suppression_count,
            "convergence_inject_threshold": self.convergence_inject_threshold,
            "stagnation_cycle_threshold": self.stagnation_cycle_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealFeedConfig:
        return cls(
            max_fragments_per_type=data.get("max_fragments_per_type", 5),
            max_observation_units=data.get("max_observation_units", 20),
            max_output_units=data.get("max_output_units", 10),
            single_type_dominance_cap=data.get("single_type_dominance_cap", 0.4),
            freshness_decay_rate=data.get("freshness_decay_rate", 0.05),
            stale_threshold=data.get("stale_threshold", 0.15),
            recent_series_suppression_count=data.get("recent_series_suppression_count", 3),
            convergence_inject_threshold=data.get("convergence_inject_threshold", 1),
            stagnation_cycle_threshold=data.get("stagnation_cycle_threshold", 5),
        )


@dataclass
class RealFeedState:
    """リアルフィード処理の全体状態。"""
    config: RealFeedConfig = field(default_factory=RealFeedConfig)
    fragments: list[ObservationFragment] = field(default_factory=list)
    units: list[ObservationUnit] = field(default_factory=list)
    conflict_table: list[ConflictRecord] = field(default_factory=list)
    feed_history: list[FeedHistoryEntry] = field(default_factory=list)
    holdback: list[HoldbackEntry] = field(default_factory=list)
    decay_history: list[float] = field(default_factory=list)
    cycle_count: int = 0
    total_feeds: int = 0
    stagnation_counter: int = 0
    convergence_warnings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fragments": [f.to_dict() for f in self.fragments],
            "units": [u.to_dict() for u in self.units],
            "conflict_table": [c.to_dict() for c in self.conflict_table],
            "feed_history": [h.to_dict() for h in self.feed_history],
            "holdback": [h.to_dict() for h in self.holdback],
            "decay_history": self.decay_history[:50],
            "cycle_count": self.cycle_count,
            "total_feeds": self.total_feeds,
            "stagnation_counter": self.stagnation_counter,
            "convergence_warnings": self.convergence_warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealFeedState:
        cfg = RealFeedConfig.from_dict(data.get("config", {}))
        return cls(
            config=cfg,
            fragments=[ObservationFragment.from_dict(f) for f in data.get("fragments", [])],
            units=[ObservationUnit.from_dict(u) for u in data.get("units", [])],
            conflict_table=[ConflictRecord.from_dict(c) for c in data.get("conflict_table", [])],
            feed_history=[FeedHistoryEntry.from_dict(h) for h in data.get("feed_history", [])],
            holdback=[HoldbackEntry.from_dict(h) for h in data.get("holdback", [])],
            decay_history=data.get("decay_history", []),
            cycle_count=data.get("cycle_count", 0),
            total_feeds=data.get("total_feeds", 0),
            stagnation_counter=data.get("stagnation_counter", 0),
            convergence_warnings=data.get("convergence_warnings", 0),
        )


@dataclass
class FeedResult:
    """process() の出力。"""
    units: list[ObservationUnit] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    source_distribution: dict[str, int] = field(default_factory=dict)
    convergence_warning: bool = False
    stagnation_warning: bool = False
    holdback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": [u.to_dict() for u in self.units],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "source_distribution": self.source_distribution,
            "convergence_warning": self.convergence_warning,
            "stagnation_warning": self.stagnation_warning,
            "holdback_count": self.holdback_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedResult:
        return cls(
            units=[ObservationUnit.from_dict(u) for u in data.get("units", [])],
            conflicts=[ConflictRecord.from_dict(c) for c in data.get("conflicts", [])],
            source_distribution=data.get("source_distribution", {}),
            convergence_warning=data.get("convergence_warning", False),
            stagnation_warning=data.get("stagnation_warning", False),
            holdback_count=data.get("holdback_count", 0),
        )


# =============================================================================
# Fragment Extraction Functions (8 pure, duck-typed)
# =============================================================================

def extract_speech_reaction(
    percept: Any = None,
    stm: Any = None,
) -> Optional[ObservationFragment]:
    """発話の質・感情内容を抽出する。

    percept: text, emotion_valence, intent
    stm: entries[-1].source_text, entries[-1].emotion_label
    """
    text = ""
    valence = 0.0
    intent = "unknown"
    hint = ""

    if percept is not None:
        text = getattr(percept, "text", "")
        valence = getattr(percept, "emotion_valence", 0.0)
        intent = getattr(percept, "intent", "unknown")

    if stm is not None:
        entries = getattr(stm, "entries", [])
        if entries:
            last = entries[-1]
            emotion_label = getattr(last, "emotion_label", "neutral")
            hint = f"intent={intent}, emotion={emotion_label}"
            src_text = getattr(last, "source_text", "")
            if src_text:
                hint += f", text_len={len(src_text)}"

    if not text and not hint:
        return None

    # 値: 感情の絶対値をベースに、テキスト長で微調整
    base_val = min(1.0, abs(valence) * 0.8 + 0.2)
    if text:
        text_factor = min(1.0, len(text) / 200.0) * 0.2
        base_val = _clamp(base_val + text_factor)

    return ObservationFragment(
        type=ObservationFragmentType.SPEECH_REACTION,
        source_description=f"speech: {intent}",
        value=base_val,
        text_hint=hint[:200],
    )


def extract_response_interval(
    stm: Any = None,
    tick_count: int = 0,
) -> Optional[ObservationFragment]:
    """入力間隔パターンを抽出する。

    stm: entries のタイムスタンプ差分
    """
    if stm is None:
        return None

    entries = getattr(stm, "entries", [])
    if len(entries) < 2:
        return None

    # 直近数エントリのタイムスタンプ差分を計算
    recent = entries[-5:] if len(entries) >= 5 else entries
    intervals = []
    for i in range(1, len(recent)):
        ts_prev = getattr(recent[i - 1], "timestamp", 0.0)
        ts_curr = getattr(recent[i], "timestamp", 0.0)
        if ts_curr > ts_prev:
            intervals.append(ts_curr - ts_prev)

    if not intervals:
        return None

    avg_interval = sum(intervals) / len(intervals)

    # 値: 短い間隔=高い応答性 (0-10秒→1.0, 60秒以上→0.1)
    if avg_interval < 10.0:
        val = 0.9
    elif avg_interval < 30.0:
        val = 0.7
    elif avg_interval < 60.0:
        val = 0.4
    else:
        val = 0.1

    hint = f"avg_interval={avg_interval:.1f}s, samples={len(intervals)}"

    return ObservationFragment(
        type=ObservationFragmentType.RESPONSE_INTERVAL,
        source_description="response_interval",
        value=val,
        text_hint=hint,
    )


def extract_topic_transition(
    percept: Any = None,
    stm: Any = None,
) -> Optional[ObservationFragment]:
    """話題変化の度合いを抽出する。

    percept: topics
    stm: entries[-2].source_text vs percept.text (トピック重複)
    """
    if percept is None:
        return None

    current_topics = set(getattr(percept, "topics", []))
    if not current_topics:
        return None

    prev_topics: set[str] = set()
    if stm is not None:
        entries = getattr(stm, "entries", [])
        if entries:
            # 直近エントリからトピックを推定
            for entry in entries[-3:]:
                src = getattr(entry, "source_text", "")
                if src:
                    # 簡易トピック抽出: 既存のperceptトピックとの語彙重複
                    for topic in current_topics:
                        if topic.lower() in src.lower():
                            prev_topics.add(topic)

    if not prev_topics and not current_topics:
        return None

    # 話題遷移度: トピック重複率の逆数
    if prev_topics:
        overlap = len(current_topics & prev_topics)
        total = max(len(current_topics), 1)
        transition_val = 1.0 - (overlap / total)
    else:
        transition_val = 0.7  # 前回トピック不明は中程度の遷移

    return ObservationFragment(
        type=ObservationFragmentType.TOPIC_TRANSITION,
        source_description="topic_transition",
        value=_clamp(transition_val),
        text_hint=f"topics={len(current_topics)}, overlap={len(current_topics & prev_topics) if prev_topics else 0}",
    )


def extract_emotional_tone(
    percept: Any = None,
    psyche: Any = None,
) -> Optional[ObservationFragment]:
    """感情的色合いを抽出する。

    percept: emotion_valence
    psyche: mood.valence, mood.arousal
    """
    valence = 0.0
    arousal = 0.5
    has_data = False

    if percept is not None:
        pv = getattr(percept, "emotion_valence", None)
        if pv is not None:
            valence = pv
            has_data = True

    if psyche is not None:
        mood = getattr(psyche, "mood", None)
        if mood is not None:
            mood_val = getattr(mood, "valence", 0.0)
            arousal = getattr(mood, "arousal", 0.5)
            # 外部入力と内部状態の平均
            if has_data:
                valence = (valence + mood_val) / 2.0
            else:
                valence = mood_val
            has_data = True

    if not has_data:
        return None

    # 値: 感情強度 (符号を保持した絶対値)
    tone_val = _clamp((abs(valence) + arousal) / 2.0)

    return ObservationFragment(
        type=ObservationFragmentType.EMOTIONAL_TONE,
        source_description="emotional_tone",
        value=tone_val,
        text_hint=f"valence={valence:.2f}, arousal={arousal:.2f}",
    )


def extract_continued_engagement(
    stm: Any = None,
    psyche: Any = None,
) -> Optional[ObservationFragment]:
    """継続的関与の程度を抽出する。

    stm: context_continuity_score, entries count
    psyche: drives.social
    """
    continuity = 0.5
    social_drive = 0.5
    entry_count = 0
    has_data = False

    if stm is not None:
        continuity = getattr(stm, "context_continuity_score", 0.5)
        entries = getattr(stm, "entries", [])
        entry_count = len(entries)
        has_data = True

    if psyche is not None:
        drives = getattr(psyche, "drives", None)
        if drives is not None:
            social_drive = getattr(drives, "social", 0.5)
            has_data = True

    if not has_data:
        return None

    # 値: 継続性と社会性の加重平均、エントリ数でブースト
    engagement = continuity * 0.5 + social_drive * 0.3
    entry_boost = min(0.2, entry_count / 50.0)
    engagement = _clamp(engagement + entry_boost)

    return ObservationFragment(
        type=ObservationFragmentType.CONTINUED_ENGAGEMENT,
        source_description="continued_engagement",
        value=engagement,
        text_hint=f"continuity={continuity:.2f}, entries={entry_count}, social={social_drive:.2f}",
    )


def extract_rejection_acceptance(
    stm: Any = None,
    percept: Any = None,
) -> Optional[ObservationFragment]:
    """承認・拒否信号を抽出する。

    stm: entries[-1].intent
    percept: intent, emotion_valence
    """
    intent = "unknown"
    valence = 0.0
    has_data = False

    if percept is not None:
        intent = getattr(percept, "intent", "unknown")
        valence = getattr(percept, "emotion_valence", 0.0)
        if intent != "unknown":
            has_data = True

    if stm is not None:
        entries = getattr(stm, "entries", [])
        if entries:
            last_intent = getattr(entries[-1], "intent", "unknown")
            if last_intent != "unknown":
                intent = last_intent
                has_data = True
            last_valence = getattr(entries[-1], "valence", 0.0)
            if last_valence != 0.0:
                valence = last_valence
                has_data = True

    if not has_data:
        return None

    # 値: ポジティブ(acceptance) = 高い値, ネガティブ(rejection) = 低い値
    # 中立に寄せる: 0.5を基準にvalenceで調整
    acceptance_val = _clamp(0.5 + valence * 0.4)

    # intent ベースのヒント補正
    positive_intents = {"agree", "praise", "thank", "accept", "approve", "greeting"}
    negative_intents = {"disagree", "reject", "complain", "criticize", "deny"}

    if intent.lower() in positive_intents:
        acceptance_val = _clamp(acceptance_val + 0.15)
    elif intent.lower() in negative_intents:
        acceptance_val = _clamp(acceptance_val - 0.15)

    return ObservationFragment(
        type=ObservationFragmentType.REJECTION_ACCEPTANCE,
        source_description=f"rejection_acceptance: {intent}",
        value=acceptance_val,
        text_hint=f"intent={intent}, valence={valence:.2f}",
    )


def extract_context_alignment(
    percept: Any = None,
    stm: Any = None,
    memories: Any = None,
) -> Optional[ObservationFragment]:
    """文脈適合度を抽出する。

    percept: topics
    stm: entries
    memories: recalled keywords (list[dict] with 'keywords')
    """
    current_topics = set()
    if percept is not None:
        current_topics = set(getattr(percept, "topics", []))

    recalled_keywords: set[str] = set()
    if memories is not None:
        if isinstance(memories, list):
            for mem in memories:
                if isinstance(mem, dict):
                    kws = mem.get("keywords", [])
                    if isinstance(kws, list):
                        recalled_keywords.update(kws)

    if not current_topics and not recalled_keywords:
        return None

    # 文脈整合度: 現在のトピックと記憶キーワードの重複度
    if current_topics and recalled_keywords:
        overlap = 0
        for topic in current_topics:
            for kw in recalled_keywords:
                if topic.lower() == kw.lower() or topic.lower() in kw.lower() or kw.lower() in topic.lower():
                    overlap += 1
                    break
        total = max(len(current_topics), 1)
        alignment_val = _clamp(overlap / total)
    elif current_topics:
        alignment_val = 0.3  # トピックあるが記憶参照なし
    else:
        alignment_val = 0.3  # 記憶あるがトピックなし

    return ObservationFragment(
        type=ObservationFragmentType.CONTEXT_ALIGNMENT,
        source_description="context_alignment",
        value=alignment_val,
        text_hint=f"topics={len(current_topics)}, keywords={len(recalled_keywords)}",
    )


def extract_recent_history(
    stm: Any = None,
) -> Optional[ObservationFragment]:
    """直近やりとりの要約を抽出する。

    stm: recent entries[-5:]
    """
    if stm is None:
        return None

    entries = getattr(stm, "entries", [])
    if not entries:
        return None

    recent = entries[-5:] if len(entries) >= 5 else entries

    # 直近エントリの統計
    total_valence = 0.0
    intent_set: set[str] = set()
    for entry in recent:
        total_valence += getattr(entry, "valence", 0.0)
        intent_set.add(getattr(entry, "intent", "unknown"))

    avg_valence = total_valence / max(len(recent), 1)

    # 値: 直近エントリの感情的活性度
    history_val = _clamp(0.5 + avg_valence * 0.3 + len(intent_set) * 0.05)

    return ObservationFragment(
        type=ObservationFragmentType.RECENT_HISTORY,
        source_description="recent_history",
        value=history_val,
        text_hint=f"entries={len(recent)}, avg_valence={avg_valence:.2f}, intents={len(intent_set)}",
    )


# =============================================================================
# Pipeline Functions
# =============================================================================

def normalize_fragments(
    fragments: list[ObservationFragment],
    config: RealFeedConfig,
) -> list[ObservationUnit]:
    """観測断片を観測単位に正規化する。

    同一typeの断片をグループ化し、各グループを1つのObservationUnitに変換する。
    max_fragments_per_type を超える分は鮮度順で切り捨て。
    """
    # typeごとにグループ化
    groups: dict[str, list[ObservationFragment]] = {}
    for frag in fragments:
        key = frag.type.value
        if key not in groups:
            groups[key] = []
        groups[key].append(frag)

    units: list[ObservationUnit] = []
    for type_key, frags in groups.items():
        # 鮮度順ソート (FRESH > RECENT > ... > FADED)
        freshness_order = {
            FragmentFreshness.FRESH: 0,
            FragmentFreshness.RECENT: 1,
            FragmentFreshness.AGING: 2,
            FragmentFreshness.STALE: 3,
            FragmentFreshness.FADED: 4,
        }
        frags.sort(key=lambda f: freshness_order.get(f.freshness, 5))

        # 切り捨て
        frags = frags[:config.max_fragments_per_type]

        # 統合: 値は加重平均、鮮度は最良を採用
        total_val = sum(f.value for f in frags)
        avg_val = total_val / max(len(frags), 1)
        best_freshness = frags[0].freshness if frags else FragmentFreshness.FRESH
        latest_ts = max(f.timestamp for f in frags) if frags else time.time()

        descriptions = [f.source_description for f in frags if f.source_description]
        desc = "; ".join(descriptions[:3])

        unit = ObservationUnit(
            source_fragment_ids=[f.fragment_id for f in frags],
            source_types=[type_key],
            description=desc,
            value=_clamp(avg_val),
            freshness=best_freshness,
            timestamp=latest_ts,
        )
        units.append(unit)

    return units


def align_units(units: list[ObservationUnit]) -> list[ObservationUnit]:
    """観測単位を鮮度・値順でソートし、整合状態を判定する。"""
    freshness_order = {
        FragmentFreshness.FRESH: 0,
        FragmentFreshness.RECENT: 1,
        FragmentFreshness.AGING: 2,
        FragmentFreshness.STALE: 3,
        FragmentFreshness.FADED: 4,
    }
    units.sort(key=lambda u: (freshness_order.get(u.freshness, 5), -u.value))

    if len(units) < 2:
        for u in units:
            u.alignment = AlignmentStatus.ALIGNED
        return units

    # 整合判定: 全体の値の分散で判定
    vals = [u.value for u in units]
    avg = sum(vals) / len(vals)
    variance = sum((v - avg) ** 2 for v in vals) / len(vals)

    for u in units:
        diff = abs(u.value - avg)
        if variance < 0.02 and diff < 0.15:
            u.alignment = AlignmentStatus.ALIGNED
        elif diff < 0.3:
            u.alignment = AlignmentStatus.PARTIAL
        else:
            u.alignment = AlignmentStatus.UNALIGNED

    return units


def detect_feed_duplicates(
    units: list[ObservationUnit],
) -> list[ObservationUnit]:
    """類似観測のグループ化。同一typeで近い値の観測をマージする。"""
    if len(units) <= 1:
        return units

    merged: list[ObservationUnit] = []
    used: set[str] = set()

    for i, u in enumerate(units):
        if u.unit_id in used:
            continue
        # 同一 source_type で値が近い別ユニットを探す
        merge_candidates = []
        for j, other in enumerate(units):
            if i == j or other.unit_id in used:
                continue
            if (u.source_types == other.source_types
                    and abs(u.value - other.value) < 0.1):
                merge_candidates.append(other)

        if merge_candidates:
            # マージ: フラグメントIDを統合、値は平均
            all_frag_ids = list(u.source_fragment_ids)
            total_val = u.value
            for mc in merge_candidates:
                all_frag_ids.extend(mc.source_fragment_ids)
                total_val += mc.value
                used.add(mc.unit_id)

            u.source_fragment_ids = all_frag_ids
            u.value = total_val / (1 + len(merge_candidates))

        merged.append(u)
        used.add(u.unit_id)

    return merged


def detect_feed_conflicts(
    units: list[ObservationUnit],
) -> tuple[list[ObservationUnit], list[ConflictRecord]]:
    """対立観測の検出。並立保持する。

    同一typeで値が大きく異なるペアを検出し、ConflictRecordを生成。
    ユニットは削除せず、conflict_status を更新する。
    """
    conflicts: list[ConflictRecord] = []

    for i, a in enumerate(units):
        for j, b in enumerate(units):
            if i >= j:
                continue
            # 同一typeで値差が大きい = 対立
            if a.source_types == b.source_types and abs(a.value - b.value) > 0.4:
                conflict = ConflictRecord(
                    unit_id_a=a.unit_id,
                    unit_id_b=b.unit_id,
                    conflict_aspect=a.source_types[0] if a.source_types else "unknown",
                    severity=abs(a.value - b.value),
                )
                conflicts.append(conflict)
                a.conflict_status = ConflictStatus.PARALLEL
                b.conflict_status = ConflictStatus.PARALLEL
                if a.unit_id not in b.competing_unit_ids:
                    b.competing_unit_ids.append(a.unit_id)
                if b.unit_id not in a.competing_unit_ids:
                    a.competing_unit_ids.append(b.unit_id)

    return units, conflicts


def apply_freshness(
    units: list[ObservationUnit],
    config: RealFeedConfig,
    current_time: float,
) -> tuple[list[ObservationUnit], list[float]]:
    """時間減衰を適用する。"""
    decay_values: list[float] = []
    freshness_map = {
        FragmentFreshness.FRESH: 1.0,
        FragmentFreshness.RECENT: 0.8,
        FragmentFreshness.AGING: 0.6,
        FragmentFreshness.STALE: 0.3,
        FragmentFreshness.FADED: 0.1,
    }

    surviving: list[ObservationUnit] = []
    for u in units:
        age = current_time - u.timestamp if u.timestamp > 0 else 0.0
        decay = config.freshness_decay_rate * (age / 60.0)  # 分単位
        decay_values.append(decay)

        # 鮮度更新
        current_freshness_val = freshness_map.get(u.freshness, 1.0)
        new_freshness_val = max(0.0, current_freshness_val - decay)

        if new_freshness_val >= 0.8:
            u.freshness = FragmentFreshness.FRESH
        elif new_freshness_val >= 0.6:
            u.freshness = FragmentFreshness.RECENT
        elif new_freshness_val >= 0.4:
            u.freshness = FragmentFreshness.AGING
        elif new_freshness_val >= 0.15:
            u.freshness = FragmentFreshness.STALE
        else:
            u.freshness = FragmentFreshness.FADED

        # FADEDかつstale_threshold以下 → 除外しない（完全消去のみで固定化しない）
        surviving.append(u)

    return surviving, decay_values


def suppress_recent_series(
    units: list[ObservationUnit],
    feed_history: list[FeedHistoryEntry],
    config: RealFeedConfig,
) -> tuple[list[ObservationUnit], list[HoldbackEntry]]:
    """直近投入系列の抑制（自己強化ループ防止）。

    直近N回のfeed_historyで連続投入されたsource_typeのユニットを抑制。
    """
    holdback: list[HoldbackEntry] = []

    if not feed_history or not units:
        return units, holdback

    # 直近N回の投入で出現した source_type を集計
    recent_feeds = feed_history[-config.recent_series_suppression_count:]
    recent_type_counts: dict[str, int] = {}
    for entry in recent_feeds:
        for st in entry.source_types:
            recent_type_counts[st] = recent_type_counts.get(st, 0) + 1

    # 連続投入されたtype (全回登場) を特定
    threshold = len(recent_feeds)
    dominant_types = {
        t for t, count in recent_type_counts.items()
        if count >= threshold and threshold >= config.recent_series_suppression_count
    }

    if not dominant_types:
        return units, holdback

    surviving: list[ObservationUnit] = []
    for u in units:
        u_types = set(u.source_types)
        if u_types & dominant_types and u.freshness != FragmentFreshness.FRESH:
            # FRESH でないものを抑制対象
            holdback.append(HoldbackEntry(
                unit_id=u.unit_id,
                source_type=u.source_types[0] if u.source_types else "",
                value=u.value,
                reason="recent_series_suppression",
                timestamp=u.timestamp,
            ))
        else:
            surviving.append(u)

    return surviving, holdback


def ensure_type_diversity(
    units: list[ObservationUnit],
    holdback: list[HoldbackEntry],
    config: RealFeedConfig,
) -> tuple[list[ObservationUnit], list[HoldbackEntry]]:
    """単一種別支配防止。holdback ↔ 再浮上。"""
    if not units:
        return units, holdback

    # 種別ごとのカウント
    type_counts: dict[str, int] = {}
    for u in units:
        for st in u.source_types:
            type_counts[st] = type_counts.get(st, 0) + 1

    total = max(len(units), 1)
    new_holdback: list[HoldbackEntry] = list(holdback)
    surviving: list[ObservationUnit] = []

    for u in units:
        dominant = False
        for st in u.source_types:
            if type_counts.get(st, 0) / total > config.single_type_dominance_cap:
                dominant = True
                break
        if dominant and len(surviving) > 0:
            # 支配的なtypeのユニットを保留（ただし最低1つは残す）
            type_key = u.source_types[0] if u.source_types else ""
            remaining_of_type = sum(
                1 for s in surviving
                if type_key in s.source_types
            )
            if remaining_of_type >= 1:
                new_holdback.append(HoldbackEntry(
                    unit_id=u.unit_id,
                    source_type=type_key,
                    value=u.value,
                    reason="type_dominance_cap",
                    timestamp=u.timestamp,
                ))
                type_counts[type_key] = type_counts.get(type_key, 1) - 1
                continue
        surviving.append(u)

    # holdback から不足typeを再浮上
    present_types = {st for u in surviving for st in u.source_types}
    all_types = {ft.value for ft in ObservationFragmentType}
    missing_types = all_types - present_types

    still_held: list[HoldbackEntry] = []
    for hb in new_holdback:
        if hb.source_type in missing_types and len(surviving) < config.max_output_units:
            # 再浮上
            surviving.append(ObservationUnit(
                unit_id=hb.unit_id or _gen_id(),
                source_types=[hb.source_type],
                description=f"resurfaced from holdback: {hb.reason}",
                value=hb.value,
                freshness=FragmentFreshness.AGING,
                timestamp=hb.timestamp,
            ))
            missing_types.discard(hb.source_type)
        else:
            still_held.append(hb)

    return surviving, still_held


def check_convergence(
    units: list[ObservationUnit],
    holdback: list[HoldbackEntry],
    config: RealFeedConfig,
) -> tuple[list[ObservationUnit], bool]:
    """安全弁: 単一解釈収束時に競合補充。

    全ユニットの値が狭い範囲に収まっている場合、
    holdback から値の異なるユニットを補充して複線入力に戻す。
    """
    if len(units) <= config.convergence_inject_threshold:
        return units, False

    vals = [u.value for u in units]
    val_range = max(vals) - min(vals)

    if val_range >= 0.2:
        return units, False

    # 収束検出: holdback から値の離れたユニットを補充
    avg_val = sum(vals) / len(vals)
    injected = False

    for hb in holdback:
        if abs(hb.value - avg_val) >= 0.2 and len(units) < config.max_output_units:
            units.append(ObservationUnit(
                unit_id=hb.unit_id or _gen_id(),
                source_types=[hb.source_type],
                description=f"convergence_inject: {hb.reason}",
                value=hb.value,
                freshness=FragmentFreshness.AGING,
                conflict_status=ConflictStatus.CONVERGENCE_RISK,
                timestamp=hb.timestamp,
            ))
            injected = True
            break  # 1つだけ補充

    return units, injected


def check_stagnation(
    state: RealFeedState,
    units: list[ObservationUnit],
) -> tuple[list[ObservationUnit], bool]:
    """安全弁: 停滞時に鮮度低下反映。

    stagnation_counter が閾値を超えた場合、全ユニットの鮮度を1段階下げ、
    新規観測の流入余地を回復する。
    """
    if state.stagnation_counter < state.config.stagnation_cycle_threshold:
        return units, False

    freshness_downgrade = {
        FragmentFreshness.FRESH: FragmentFreshness.RECENT,
        FragmentFreshness.RECENT: FragmentFreshness.AGING,
        FragmentFreshness.AGING: FragmentFreshness.STALE,
        FragmentFreshness.STALE: FragmentFreshness.FADED,
        FragmentFreshness.FADED: FragmentFreshness.FADED,
    }

    for u in units:
        u.freshness = freshness_downgrade.get(u.freshness, u.freshness)

    return units, True


# =============================================================================
# RealFeedProcessor
# =============================================================================

class RealFeedProcessor:
    """リアルフィード処理の主クラス。"""

    def __init__(self, config: Optional[RealFeedConfig] = None):
        self._state = RealFeedState(
            config=config or RealFeedConfig(),
        )

    @property
    def state(self) -> RealFeedState:
        return self._state

    @state.setter
    def state(self, value: RealFeedState) -> None:
        self._state = value

    def process(
        self,
        percept: Any = None,
        stm: Any = None,
        psyche: Any = None,
        dynamics: Any = None,
        recalled_memories: Any = None,
        integration_result: Any = None,
        tick_count: int = 0,
    ) -> FeedResult:
        """8断片抽出 → 正規化 → 整列 → 重複 → 競合 → 鮮度 → 抑制 → 多様性 → 安全弁。

        Args:
            percept: Percept オブジェクト
            stm: ShortTermMemory オブジェクト
            psyche: PsycheState オブジェクト
            dynamics: DynamicsState オブジェクト
            recalled_memories: recall_with_mood の結果 (list[dict])
            integration_result: MemorySystemIntegration の結果
            tick_count: 現在のティックカウント

        Returns:
            FeedResult: 処理結果
        """
        cfg = self._state.config
        now = time.time()

        # ── Step 1: 8断片抽出 ──
        raw_fragments: list[ObservationFragment] = []

        frag = extract_speech_reaction(percept, stm)
        if frag:
            raw_fragments.append(frag)

        frag = extract_response_interval(stm, tick_count)
        if frag:
            raw_fragments.append(frag)

        frag = extract_topic_transition(percept, stm)
        if frag:
            raw_fragments.append(frag)

        frag = extract_emotional_tone(percept, psyche)
        if frag:
            raw_fragments.append(frag)

        frag = extract_continued_engagement(stm, psyche)
        if frag:
            raw_fragments.append(frag)

        frag = extract_rejection_acceptance(stm, percept)
        if frag:
            raw_fragments.append(frag)

        frag = extract_context_alignment(percept, stm, recalled_memories)
        if frag:
            raw_fragments.append(frag)

        frag = extract_recent_history(stm)
        if frag:
            raw_fragments.append(frag)

        # 既存断片に追加
        self._state.fragments.extend(raw_fragments)
        # typeごとのmax超過分を除去
        self._trim_fragments()

        # ── Step 2: normalize_fragments → ObservationUnit化 ──
        units = normalize_fragments(self._state.fragments, cfg)

        # ── Step 3: align_units ──
        units = align_units(units)

        # ── Step 4: detect_feed_duplicates ──
        units = detect_feed_duplicates(units)

        # ── Step 5: detect_feed_conflicts ──
        units, new_conflicts = detect_feed_conflicts(units)
        self._state.conflict_table.extend(new_conflicts)
        # 古い競合記録の制限
        self._state.conflict_table = self._state.conflict_table[-50:]

        # ── Step 6: apply_freshness ──
        units, decay_vals = apply_freshness(units, cfg, now)
        self._state.decay_history.extend(decay_vals)
        self._state.decay_history = self._state.decay_history[-100:]

        # ── Step 7: suppress_recent_series ──
        units, new_holdback = suppress_recent_series(
            units, self._state.feed_history, cfg,
        )

        # ── Step 8: ensure_type_diversity ──
        existing_holdback = list(self._state.holdback) + new_holdback
        units, remaining_holdback = ensure_type_diversity(
            units, existing_holdback, cfg,
        )
        self._state.holdback = remaining_holdback[-30:]

        # ── Step 9: check_convergence ──
        units, convergence_warning = check_convergence(
            units, self._state.holdback, cfg,
        )

        # ── Step 10: check_stagnation ──
        # 停滞判定: 新規断片が少ない場合カウント増加
        if len(raw_fragments) <= 1:
            self._state.stagnation_counter += 1
        else:
            self._state.stagnation_counter = 0

        units, stagnation_warning = check_stagnation(self._state, units)
        if stagnation_warning:
            self._state.stagnation_counter = 0  # リセット

        if convergence_warning:
            self._state.convergence_warnings += 1

        # ── 出力制限 ──
        units = units[:cfg.max_output_units]

        # ── 状態更新 ──
        self._state.units = units
        self._state.cycle_count += 1
        self._state.total_feeds += len(units)

        # 投入履歴追加
        all_source_types = []
        for u in units:
            all_source_types.extend(u.source_types)
        self._state.feed_history.append(FeedHistoryEntry(
            unit_ids=[u.unit_id for u in units],
            source_types=list(set(all_source_types)),
            timestamp=now,
            cycle_id=self._state.cycle_count,
        ))
        self._state.feed_history = self._state.feed_history[-20:]

        # source_distribution 計算
        dist: dict[str, int] = {}
        for u in units:
            for st in u.source_types:
                dist[st] = dist.get(st, 0) + 1

        return FeedResult(
            units=units,
            conflicts=new_conflicts,
            source_distribution=dist,
            convergence_warning=convergence_warning,
            stagnation_warning=stagnation_warning,
            holdback_count=len(self._state.holdback),
        )

    def inject_external_fragments(self, fragments: list[ObservationFragment]) -> None:
        """外部モジュール（action_result等）からの観測断片を注入する。

        行動-結果観測で得られた他者反応の時系列的隣接記録を、
        他者モデルの観測断片の一種として追加する。因果帰属は行わない。
        """
        if not fragments:
            return
        self._state.fragments.extend(fragments)
        self._trim_fragments()

    def _trim_fragments(self) -> None:
        """typeごとの断片数を制限する。"""
        cfg = self._state.config
        groups: dict[str, list[ObservationFragment]] = {}
        for f in self._state.fragments:
            key = f.type.value
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

        trimmed: list[ObservationFragment] = []
        for key, frags in groups.items():
            # 新しい順で保持
            frags.sort(key=lambda f: f.timestamp, reverse=True)
            trimmed.extend(frags[:cfg.max_fragments_per_type])

        self._state.fragments = trimmed


# =============================================================================
# Output Integration
# =============================================================================

def enhance_context_with_feed(
    ctx: Any,
    feed: FeedResult,
) -> Any:
    """既存の ContextSnapshot を差分で調整する。

    注意: この関数は ctx の属性を直接変更する（in-place mutation）。
    呼び出し元は変更される前提で使用すること。

    - CONTINUED_ENGAGEMENT → responsiveness 上方修正
    - EMOTIONAL_TONE → weight 上方修正
    - TOPIC_TRANSITION → density 調整
    - RESPONSE_INTERVAL → pace 調整
    - 既存値を 0.0-1.0 にclamp

    Args:
        ctx: ContextSnapshot オブジェクト (duck typing, in-place変更される)
        feed: FeedResult

    Returns:
        調整後の ContextSnapshot (入力と同じオブジェクト)
    """
    if not feed or not feed.units:
        return ctx

    # ユニットをtype別に集約
    type_values: dict[str, float] = {}
    for u in feed.units:
        for st in u.source_types:
            if st not in type_values:
                type_values[st] = u.value
            else:
                type_values[st] = (type_values[st] + u.value) / 2.0

    # 調整係数 (既存値に対する差分)
    adjustment_weight = 0.3  # 既存値への影響度

    # CONTINUED_ENGAGEMENT → responsiveness
    if ObservationFragmentType.CONTINUED_ENGAGEMENT.value in type_values:
        feed_val = type_values[ObservationFragmentType.CONTINUED_ENGAGEMENT.value]
        current = getattr(ctx, "responsiveness", 0.5)
        delta = (feed_val - current) * adjustment_weight
        new_val = _clamp(current + delta)
        if hasattr(ctx, "responsiveness"):
            ctx.responsiveness = new_val

    # EMOTIONAL_TONE → weight
    if ObservationFragmentType.EMOTIONAL_TONE.value in type_values:
        feed_val = type_values[ObservationFragmentType.EMOTIONAL_TONE.value]
        current = getattr(ctx, "weight", 0.5)
        delta = (feed_val - current) * adjustment_weight
        new_val = _clamp(current + delta)
        if hasattr(ctx, "weight"):
            ctx.weight = new_val

    # TOPIC_TRANSITION → density
    if ObservationFragmentType.TOPIC_TRANSITION.value in type_values:
        feed_val = type_values[ObservationFragmentType.TOPIC_TRANSITION.value]
        current = getattr(ctx, "density", 0.5)
        delta = (feed_val - current) * adjustment_weight
        new_val = _clamp(current + delta)
        if hasattr(ctx, "density"):
            ctx.density = new_val

    # RESPONSE_INTERVAL → pace
    if ObservationFragmentType.RESPONSE_INTERVAL.value in type_values:
        feed_val = type_values[ObservationFragmentType.RESPONSE_INTERVAL.value]
        current = getattr(ctx, "pace", 0.5)
        delta = (feed_val - current) * adjustment_weight
        new_val = _clamp(current + delta)
        if hasattr(ctx, "pace"):
            ctx.pace = new_val

    return ctx


# =============================================================================
# Summary
# =============================================================================

def get_real_feed_summary(processor: RealFeedProcessor) -> str:
    """リアルフィード状態のサマリ文字列を返す。"""
    s = processor.state
    if s.cycle_count == 0:
        return "real_feed: inactive"

    unit_types = set()
    for u in s.units:
        unit_types.update(u.source_types)

    parts = [
        f"cycle={s.cycle_count}",
        f"units={len(s.units)}",
        f"types={len(unit_types)}",
        f"conflicts={len(s.conflict_table)}",
        f"holdback={len(s.holdback)}",
    ]
    if s.convergence_warnings > 0:
        parts.append(f"conv_warn={s.convergence_warnings}")
    if s.stagnation_counter > 0:
        parts.append(f"stagnation={s.stagnation_counter}")

    return "real_feed: " + ", ".join(parts)


# =============================================================================
# Convenience
# =============================================================================

def create_real_feed_processor(
    config: Optional[RealFeedConfig] = None,
) -> RealFeedProcessor:
    """RealFeedProcessor のファクトリ関数。"""
    return RealFeedProcessor(config=config)


# =============================================================================
# Internal Helpers
# =============================================================================

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """値を [lo, hi] にクランプする。"""
    return max(lo, min(hi, v))
