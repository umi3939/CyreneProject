"""
Episodic Memory (エピソード記憶 - 自伝的記憶)

出来事単位の経験を保持し、体験の文脈・感情・自己観測の関係を
失わない形で残すためのモジュール。

既存の短期記憶（痕跡）・長期統計とは異なり、
個別の出来事にその時の感情・自己観測を随伴情報として付与し、
中長期的に保持する構造。

CRITICAL DESIGN PRINCIPLES:
- 行動や判断の正否評価をしない
- 価値・信念・目標の形成や更新を直接行わない
- 出来事の意味づけ・人格化・規範化を行わない
- 設計された人格の方向性を押し付けない
- 記憶は参照のみで、意思決定に直接影響する経路を持たない
- 責任評価や価値更新に接続しない
- 出来事の解釈は固定しない（再要約・再関連付けを許容）
- 同一の出来事に対する見え方は時間と共に変化して良い
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class EpisodeType(Enum):
    """
    Types of episode entries.

    出来事の分類。評価的ではなく、記述的なカテゴリ。
    """
    INTERACTION = "interaction"          # 対話・やりとり
    OBSERVATION = "observation"          # 観察・知覚
    EMOTIONAL_EVENT = "emotional_event"  # 感情的出来事
    STATE_CHANGE = "state_change"        # 内部状態の変化
    CONTEXT_SHIFT = "context_shift"      # 外部文脈の変化
    COMPOSITE = "composite"              # 圧縮された複合エピソード
    UNDETERMINED = "undetermined"        # 未確定


class ImportanceLevel(Enum):
    """
    Abstract importance level for episodes.

    重要度は減衰速度を変調するためのもの。
    善悪・正誤の評価ではない。
    """
    TRIVIAL = "trivial"          # 些末
    MINOR = "minor"              # 軽微
    MODERATE = "moderate"        # 中程度
    NOTABLE = "notable"          # 注目
    SIGNIFICANT = "significant"  # 顕著


class DecayState(Enum):
    """
    Abstract decay state of an episode.

    鮮明度に基づく抽象的な減衰段階。
    """
    FRESH = "fresh"              # 鮮明（直近）
    CLEAR = "clear"              # 明瞭
    FADING = "fading"            # 薄れつつある
    DIM = "dim"                  # おぼろげ
    COMPRESSIBLE = "compressible"  # 圧縮可能


class EpisodeLinkType(Enum):
    """Types of weak references between episodes."""
    TEMPORAL_PROXIMITY = "temporal_proximity"    # 時間的近接
    TOPIC_OVERLAP = "topic_overlap"              # トピック重複
    EMOTIONAL_SIMILARITY = "emotional_similarity"  # 感情的類似
    CAUSAL_SEQUENCE = "causal_sequence"          # 因果的連続
    THEMATIC = "thematic"                        # テーマ的関連


class SearchMode(Enum):
    """Search modes for episode retrieval."""
    BY_TOPIC = "by_topic"
    BY_TIME = "by_time"
    BY_EMOTION = "by_emotion"
    BY_IMPORTANCE = "by_importance"
    COMBINED = "combined"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class EmotionalCompanion:
    """
    感情随伴情報 - その時の感情の概略。

    出来事の中心にはせず、随伴として添付する。
    """
    primary_emotion: str              # 主要感情ラベル
    intensity_level: float            # 0.0〜1.0
    valence: float                    # -1.0〜1.0
    harmony: float                    # 0.0〜1.0 (調和度)
    emotion_description: str          # 人間可読な感情記述
    coexisting_emotions: tuple[str, ...]  # 共存する感情


@dataclass(frozen=True)
class SelfObservationCompanion:
    """
    自己観測随伴情報 - 内省や差分観測の概略。

    出来事に随伴する自己観測データ。
    """
    has_difference: bool              # 自己差分があったか
    difference_magnitude: str         # 差分の大きさ
    difference_nature: str            # 差分の性質
    tendency_description: str         # 傾向の概略
    has_strong_tendency: bool         # 強い傾向があるか
    coherence_level: str              # 自己同一性の整合性
    narrative_trend: str              # 物語の傾向


@dataclass(frozen=True)
class EpisodeLink:
    """
    エピソード間の弱い参照。

    エピソード同士の関連を記録するが、
    因果的必然性を主張しない。
    """
    from_episode_id: str
    to_episode_id: str
    link_type: EpisodeLinkType
    strength: float               # 0.0〜1.0
    description: str


@dataclass(frozen=True)
class EpisodeEntry:
    """
    出来事エントリ - エピソード記憶の最小単位。

    一件の出来事を保持する不変構造。
    評価・善悪判定を含まない。
    """
    episode_id: str
    episode_type: EpisodeType
    summary: str                      # 出来事の要約
    topics: tuple[str, ...]           # 関連トピック
    source_texts: tuple[str, ...]     # 元となったテキスト断片
    timestamp: float
    duration_estimate: float          # 推定所要時間（秒）
    emotional_companion: Optional[EmotionalCompanion]
    self_observation_companion: Optional[SelfObservationCompanion]
    context_summary: str              # 場面・状況の要約
    importance: ImportanceLevel
    vividness: float                  # 0.0〜1.0
    reference_count: int              # 参照された回数
    reinterpretation_count: int       # 再解釈された回数
    is_compressed: bool               # 圧縮されたエピソードか
    compressed_episode_ids: tuple[str, ...]  # 圧縮元のID群

    def get_decay_state(self) -> DecayState:
        """Get abstract decay state from vividness."""
        return determine_decay_state(self.vividness)

    def with_vividness(self, new_vividness: float) -> EpisodeEntry:
        """Create a copy with updated vividness."""
        return EpisodeEntry(
            episode_id=self.episode_id,
            episode_type=self.episode_type,
            summary=self.summary,
            topics=self.topics,
            source_texts=self.source_texts,
            timestamp=self.timestamp,
            duration_estimate=self.duration_estimate,
            emotional_companion=self.emotional_companion,
            self_observation_companion=self.self_observation_companion,
            context_summary=self.context_summary,
            importance=self.importance,
            vividness=max(0.0, min(1.0, new_vividness)),
            reference_count=self.reference_count,
            reinterpretation_count=self.reinterpretation_count,
            is_compressed=self.is_compressed,
            compressed_episode_ids=self.compressed_episode_ids,
        )

    def with_reference(self) -> EpisodeEntry:
        """Create a copy with incremented reference count."""
        return EpisodeEntry(
            episode_id=self.episode_id,
            episode_type=self.episode_type,
            summary=self.summary,
            topics=self.topics,
            source_texts=self.source_texts,
            timestamp=self.timestamp,
            duration_estimate=self.duration_estimate,
            emotional_companion=self.emotional_companion,
            self_observation_companion=self.self_observation_companion,
            context_summary=self.context_summary,
            importance=self.importance,
            vividness=self.vividness,
            reference_count=self.reference_count + 1,
            reinterpretation_count=self.reinterpretation_count,
            is_compressed=self.is_compressed,
            compressed_episode_ids=self.compressed_episode_ids,
        )

    def reinterpret(
        self,
        new_summary: str,
        new_type: Optional[EpisodeType] = None,
    ) -> EpisodeEntry:
        """Create a reinterpreted copy (解釈は固定しない)."""
        return EpisodeEntry(
            episode_id=self.episode_id,
            episode_type=new_type if new_type is not None else self.episode_type,
            summary=new_summary,
            topics=self.topics,
            source_texts=self.source_texts,
            timestamp=self.timestamp,
            duration_estimate=self.duration_estimate,
            emotional_companion=self.emotional_companion,
            self_observation_companion=self.self_observation_companion,
            context_summary=self.context_summary,
            importance=self.importance,
            vividness=self.vividness,
            reference_count=self.reference_count,
            reinterpretation_count=self.reinterpretation_count + 1,
            is_compressed=self.is_compressed,
            compressed_episode_ids=self.compressed_episode_ids,
        )


@dataclass(frozen=True)
class EpisodeStore:
    """
    不変スナップショット - エピソード記憶の全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    episodes: tuple[EpisodeEntry, ...]
    links: tuple[EpisodeLink, ...]
    total_episodes_recorded: int
    total_compressions: int
    average_vividness: float
    active_episode_count: int
    compressed_episode_count: int
    timestamp: float
    description: str

    @classmethod
    def empty(cls, timestamp: Optional[float] = None) -> EpisodeStore:
        """Create an empty store."""
        ts = timestamp or time.time()
        return cls(
            episodes=(),
            links=(),
            total_episodes_recorded=0,
            total_compressions=0,
            average_vividness=0.0,
            active_episode_count=0,
            compressed_episode_count=0,
            timestamp=ts,
            description="No episodes recorded yet.",
        )

    def has_episodes(self) -> bool:
        return len(self.episodes) > 0

    def get_active_episodes(self) -> list[EpisodeEntry]:
        """Get episodes that haven't fully decayed."""
        return [e for e in self.episodes if e.vividness > 0.0]

    def get_fresh_episodes(self) -> list[EpisodeEntry]:
        """Get fresh/clear episodes."""
        return [
            e for e in self.episodes
            if e.get_decay_state() in (DecayState.FRESH, DecayState.CLEAR)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episodes": [
                _episode_to_dict(e) for e in self.episodes
            ],
            "links": [
                {
                    "from_episode_id": link.from_episode_id,
                    "to_episode_id": link.to_episode_id,
                    "link_type": link.link_type.value,
                    "strength": link.strength,
                    "description": link.description,
                }
                for link in self.links
            ],
            "total_episodes_recorded": self.total_episodes_recorded,
            "total_compressions": self.total_compressions,
            "average_vividness": self.average_vividness,
            "active_episode_count": self.active_episode_count,
            "compressed_episode_count": self.compressed_episode_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeStore:
        """Create from dictionary."""
        episodes = tuple(
            _episode_from_dict(e) for e in data.get("episodes", [])
        )
        links = tuple(
            EpisodeLink(
                from_episode_id=link["from_episode_id"],
                to_episode_id=link["to_episode_id"],
                link_type=EpisodeLinkType(link["link_type"]),
                strength=link.get("strength", 0.5),
                description=link.get("description", ""),
            )
            for link in data.get("links", [])
        )
        return cls(
            episodes=episodes,
            links=links,
            total_episodes_recorded=data.get("total_episodes_recorded", 0),
            total_compressions=data.get("total_compressions", 0),
            average_vividness=data.get("average_vividness", 0.0),
            active_episode_count=data.get("active_episode_count", 0),
            compressed_episode_count=data.get("compressed_episode_count", 0),
            timestamp=data.get("timestamp", 0.0),
            description=data.get("description", ""),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class EpisodicMemoryConfig:
    """
    Configuration for the episodic memory system.

    These parameters control episode lifecycle.
    They do NOT affect decisions.
    """
    # Capacity
    max_episodes: int = 200

    # Decay
    base_decay_rate: float = 0.02
    importance_decay_modifier: float = 0.5   # 重要なほど減衰が遅い
    reference_decay_modifier: float = 0.3    # 参照されるほど減衰が遅い

    # Vividness recovery
    reference_vividness_boost: float = 0.15

    # Compression
    compression_vividness_threshold: float = 0.15
    min_episodes_for_compression: int = 3
    compression_result_vividness: float = 0.4

    # Search
    default_max_results: int = 10
    topic_overlap_threshold: float = 0.2
    emotional_similarity_threshold: float = 0.3
    temporal_proximity_window: float = 300.0  # 5 minutes

    # Link generation
    link_strength_threshold: float = 0.2
    max_links_per_episode: int = 5


# =============================================================================
# ID Generation
# =============================================================================

def _generate_episode_id() -> str:
    """Generate a unique episode ID."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# Helper Functions (Pure)
# =============================================================================

def determine_decay_state(vividness: float) -> DecayState:
    """Determine DecayState from vividness value."""
    if vividness >= 0.8:
        return DecayState.FRESH
    elif vividness >= 0.6:
        return DecayState.CLEAR
    elif vividness >= 0.4:
        return DecayState.FADING
    elif vividness >= 0.2:
        return DecayState.DIM
    else:
        return DecayState.COMPRESSIBLE


def classify_episode_type(
    stm_entries: Optional[Any] = None,
    emotional_state: Optional[Any] = None,
    difference_summary: Optional[Any] = None,
    external_context: Optional[Any] = None,
) -> EpisodeType:
    """
    Classify the episode type from input signals.

    STM内容・感情・文脈からタイプを分類する。
    評価的ではなく記述的な分類。
    """
    has_stm = stm_entries is not None
    has_emotion = emotional_state is not None
    has_diff = difference_summary is not None
    has_context = external_context is not None

    if has_stm:
        entries = getattr(stm_entries, "entries", [])
        if entries:
            last = entries[-1] if isinstance(entries, list) else None
            if last is not None:
                source = getattr(last, "source_text", "")
                intent = getattr(last, "intent", "")
                if intent in ("question", "response", "greeting"):
                    return EpisodeType.INTERACTION
                valence = getattr(last, "valence", 0.0)
                if abs(valence) > 0.6:
                    return EpisodeType.EMOTIONAL_EVENT

    if has_emotion:
        intensity = getattr(emotional_state, "intensity", None)
        if intensity is not None:
            intensity_val = getattr(intensity, "value", str(intensity))
            if intensity_val in ("intense", "overwhelming"):
                return EpisodeType.EMOTIONAL_EVENT

    if has_diff:
        has_difference = getattr(difference_summary, "has_difference", False)
        if has_difference:
            magnitude = getattr(difference_summary, "magnitude", None)
            if magnitude is not None:
                mag_val = getattr(magnitude, "value", str(magnitude))
                if mag_val in ("noticeable", "significant", "substantial"):
                    return EpisodeType.STATE_CHANGE

    if has_context:
        if isinstance(external_context, str) and external_context.strip():
            return EpisodeType.CONTEXT_SHIFT
        weight = getattr(external_context, "weight", 0.5)
        if weight > 0.7:
            return EpisodeType.CONTEXT_SHIFT

    if has_stm:
        return EpisodeType.OBSERVATION

    return EpisodeType.UNDETERMINED


def compute_importance(
    emotional_state: Optional[Any] = None,
    difference_summary: Optional[Any] = None,
    stm_entries: Optional[Any] = None,
) -> ImportanceLevel:
    """
    Compute importance level from emotional intensity and self-difference.

    善悪・正誤の評価ではなく、減衰速度の変調のみに使用する。
    """
    score = 0.0

    # Emotional intensity contribution
    if emotional_state is not None:
        intensity = getattr(emotional_state, "intensity", None)
        if intensity is not None:
            intensity_val = getattr(intensity, "value", str(intensity))
            intensity_map = {
                "calm": 0.1,
                "mild": 0.2,
                "moderate": 0.4,
                "intense": 0.7,
                "overwhelming": 1.0,
            }
            score += intensity_map.get(intensity_val, 0.2)

    # Self-difference contribution
    if difference_summary is not None:
        has_diff = getattr(difference_summary, "has_difference", False)
        if has_diff:
            magnitude = getattr(difference_summary, "magnitude", None)
            if magnitude is not None:
                mag_val = getattr(magnitude, "value", str(magnitude))
                mag_map = {
                    "none": 0.0,
                    "minimal": 0.1,
                    "noticeable": 0.3,
                    "significant": 0.6,
                    "substantial": 0.9,
                }
                score += mag_map.get(mag_val, 0.1)

    # STM valence contribution
    if stm_entries is not None:
        entries = getattr(stm_entries, "entries", [])
        if entries:
            last = entries[-1] if isinstance(entries, list) else None
            if last is not None:
                valence = abs(getattr(last, "valence", 0.0))
                score += valence * 0.3

    # Map score to level
    if score >= 1.2:
        return ImportanceLevel.SIGNIFICANT
    elif score >= 0.8:
        return ImportanceLevel.NOTABLE
    elif score >= 0.5:
        return ImportanceLevel.MODERATE
    elif score >= 0.2:
        return ImportanceLevel.MINOR
    else:
        return ImportanceLevel.TRIVIAL


def compute_topic_overlap(
    topics_a: tuple[str, ...],
    topics_b: tuple[str, ...],
) -> float:
    """Compute Jaccard similarity between two topic sets."""
    if not topics_a and not topics_b:
        return 0.0
    set_a = set(t.lower() for t in topics_a)
    set_b = set(t.lower() for t in topics_b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def compute_emotional_similarity(
    companion_a: Optional[EmotionalCompanion],
    companion_b: Optional[EmotionalCompanion],
) -> float:
    """Compute emotional similarity from valence distance."""
    if companion_a is None or companion_b is None:
        return 0.0
    valence_dist = abs(companion_a.valence - companion_b.valence)
    intensity_dist = abs(companion_a.intensity_level - companion_b.intensity_level)
    # Combined distance normalized to 0-1, then inverted for similarity
    combined = (valence_dist / 2.0 + intensity_dist) / 1.5
    return max(0.0, 1.0 - combined)


def compute_temporal_proximity(
    timestamp_a: float,
    timestamp_b: float,
    window: float = 300.0,
) -> float:
    """Compute temporal proximity score (1.0 = same time, decays over window)."""
    if window <= 0:
        return 0.0
    diff = abs(timestamp_a - timestamp_b)
    if diff >= window:
        return 0.0
    return 1.0 - (diff / window)


def generate_episode_summary(
    episode_type: EpisodeType,
    stm_entries: Optional[Any] = None,
    emotional_state: Optional[Any] = None,
    context_summary: str = "",
) -> str:
    """Generate a human-readable summary for an episode."""
    parts = []

    type_labels = {
        EpisodeType.INTERACTION: "Interaction occurred",
        EpisodeType.OBSERVATION: "Observation recorded",
        EpisodeType.EMOTIONAL_EVENT: "Emotional event experienced",
        EpisodeType.STATE_CHANGE: "Internal state change detected",
        EpisodeType.CONTEXT_SHIFT: "Context shift detected",
        EpisodeType.COMPOSITE: "Composite episode",
        EpisodeType.UNDETERMINED: "Episode recorded",
    }
    parts.append(type_labels.get(episode_type, "Episode recorded"))

    if stm_entries is not None:
        entries = getattr(stm_entries, "entries", [])
        if entries:
            last = entries[-1] if isinstance(entries, list) else None
            if last is not None:
                source = getattr(last, "source_text", "")
                if source:
                    parts.append(f"involving: {source[:80]}")

    if emotional_state is not None:
        desc = getattr(emotional_state, "description", "")
        if desc:
            parts.append(f"emotional context: {desc[:60]}")

    if context_summary:
        parts.append(f"in context: {context_summary[:60]}")

    return "; ".join(parts)


def _importance_to_decay_modifier(importance: ImportanceLevel) -> float:
    """Convert importance to a decay rate modifier (lower = slower decay)."""
    modifiers = {
        ImportanceLevel.TRIVIAL: 1.0,
        ImportanceLevel.MINOR: 0.8,
        ImportanceLevel.MODERATE: 0.6,
        ImportanceLevel.NOTABLE: 0.4,
        ImportanceLevel.SIGNIFICANT: 0.2,
    }
    return modifiers.get(importance, 1.0)


def _reference_to_decay_modifier(reference_count: int) -> float:
    """Convert reference count to a decay rate modifier."""
    if reference_count <= 0:
        return 1.0
    # Diminishing returns: each reference reduces decay less
    return max(0.3, 1.0 - (reference_count * 0.15))


# =============================================================================
# Serialization Helpers
# =============================================================================

def _episode_to_dict(entry: EpisodeEntry) -> dict[str, Any]:
    """Convert an EpisodeEntry to a dictionary."""
    result: dict[str, Any] = {
        "episode_id": entry.episode_id,
        "episode_type": entry.episode_type.value,
        "summary": entry.summary,
        "topics": list(entry.topics),
        "source_texts": list(entry.source_texts),
        "timestamp": entry.timestamp,
        "duration_estimate": entry.duration_estimate,
        "context_summary": entry.context_summary,
        "importance": entry.importance.value,
        "vividness": entry.vividness,
        "reference_count": entry.reference_count,
        "reinterpretation_count": entry.reinterpretation_count,
        "is_compressed": entry.is_compressed,
        "compressed_episode_ids": list(entry.compressed_episode_ids),
    }

    if entry.emotional_companion is not None:
        ec = entry.emotional_companion
        result["emotional_companion"] = {
            "primary_emotion": ec.primary_emotion,
            "intensity_level": ec.intensity_level,
            "valence": ec.valence,
            "harmony": ec.harmony,
            "emotion_description": ec.emotion_description,
            "coexisting_emotions": list(ec.coexisting_emotions),
        }
    else:
        result["emotional_companion"] = None

    if entry.self_observation_companion is not None:
        so = entry.self_observation_companion
        result["self_observation_companion"] = {
            "has_difference": so.has_difference,
            "difference_magnitude": so.difference_magnitude,
            "difference_nature": so.difference_nature,
            "tendency_description": so.tendency_description,
            "has_strong_tendency": so.has_strong_tendency,
            "coherence_level": so.coherence_level,
            "narrative_trend": so.narrative_trend,
        }
    else:
        result["self_observation_companion"] = None

    return result


def _episode_from_dict(data: dict[str, Any]) -> EpisodeEntry:
    """Create an EpisodeEntry from a dictionary."""
    ec_data = data.get("emotional_companion")
    emotional_companion = None
    if ec_data is not None:
        emotional_companion = EmotionalCompanion(
            primary_emotion=ec_data.get("primary_emotion", "neutral"),
            intensity_level=ec_data.get("intensity_level", 0.0),
            valence=ec_data.get("valence", 0.0),
            harmony=ec_data.get("harmony", 0.5),
            emotion_description=ec_data.get("emotion_description", ""),
            coexisting_emotions=tuple(ec_data.get("coexisting_emotions", ())),
        )

    so_data = data.get("self_observation_companion")
    self_observation_companion = None
    if so_data is not None:
        self_observation_companion = SelfObservationCompanion(
            has_difference=so_data.get("has_difference", False),
            difference_magnitude=so_data.get("difference_magnitude", "none"),
            difference_nature=so_data.get("difference_nature", "stable"),
            tendency_description=so_data.get("tendency_description", ""),
            has_strong_tendency=so_data.get("has_strong_tendency", False),
            coherence_level=so_data.get("coherence_level", "undefined"),
            narrative_trend=so_data.get("narrative_trend", "undefined"),
        )

    return EpisodeEntry(
        episode_id=data["episode_id"],
        episode_type=EpisodeType(data.get("episode_type", "undetermined")),
        summary=data.get("summary", ""),
        topics=tuple(data.get("topics", ())),
        source_texts=tuple(data.get("source_texts", ())),
        timestamp=data.get("timestamp", 0.0),
        duration_estimate=data.get("duration_estimate", 0.0),
        emotional_companion=emotional_companion,
        self_observation_companion=self_observation_companion,
        context_summary=data.get("context_summary", ""),
        importance=ImportanceLevel(data.get("importance", "trivial")),
        vividness=data.get("vividness", 0.0),
        reference_count=data.get("reference_count", 0),
        reinterpretation_count=data.get("reinterpretation_count", 0),
        is_compressed=data.get("is_compressed", False),
        compressed_episode_ids=tuple(data.get("compressed_episode_ids", ())),
    )


# =============================================================================
# Companion Builders
# =============================================================================

def _build_emotional_companion(
    emotional_state: Optional[Any] = None,
) -> Optional[EmotionalCompanion]:
    """Build EmotionalCompanion from emotional state input."""
    if emotional_state is None:
        return None

    primary = getattr(emotional_state, "description", "")
    if not primary:
        primary = "neutral"

    intensity = 0.5
    intensity_attr = getattr(emotional_state, "intensity", None)
    if intensity_attr is not None:
        intensity_val = getattr(intensity_attr, "value", str(intensity_attr))
        intensity_map = {
            "calm": 0.1, "mild": 0.3, "moderate": 0.5,
            "intense": 0.8, "overwhelming": 1.0,
        }
        intensity = intensity_map.get(intensity_val, 0.5)

    valence = 0.0
    spread = getattr(emotional_state, "spread", None)
    harmony_attr = getattr(emotional_state, "harmony", None)

    harmony = 0.5
    if harmony_attr is not None:
        harmony_val = getattr(harmony_attr, "value", str(harmony_attr))
        harmony_map = {
            "harmonious": 0.9, "mixed": 0.5, "conflicted": 0.2,
        }
        harmony = harmony_map.get(harmony_val, 0.5)

    coexisting: tuple[str, ...] = ()

    return EmotionalCompanion(
        primary_emotion=primary,
        intensity_level=intensity,
        valence=valence,
        harmony=harmony,
        emotion_description=getattr(emotional_state, "description", ""),
        coexisting_emotions=coexisting,
    )


def _build_self_observation_companion(
    difference_summary: Optional[Any] = None,
    tendency_awareness: Optional[Any] = None,
    coherence_state: Optional[Any] = None,
    narrative_state: Optional[Any] = None,
) -> Optional[SelfObservationCompanion]:
    """Build SelfObservationCompanion from various self-observation inputs."""
    if all(x is None for x in [
        difference_summary, tendency_awareness, coherence_state, narrative_state
    ]):
        return None

    has_difference = False
    diff_magnitude = "none"
    diff_nature = "stable"
    if difference_summary is not None:
        has_difference = getattr(difference_summary, "has_difference", False)
        mag = getattr(difference_summary, "magnitude", None)
        diff_magnitude = getattr(mag, "value", str(mag)) if mag else "none"
        nat = getattr(difference_summary, "nature", None)
        diff_nature = getattr(nat, "value", str(nat)) if nat else "stable"

    tendency_desc = ""
    has_strong = False
    if tendency_awareness is not None:
        has_awareness = getattr(tendency_awareness, "has_awareness", False)
        if has_awareness:
            items = getattr(tendency_awareness, "items", [])
            if items:
                descs = [getattr(i, "description", "") for i in items[:3]]
                tendency_desc = "; ".join(d for d in descs if d)
            from .tendency_awareness import AwarenessType
            for item in items:
                at = getattr(item, "awareness_type", None)
                if at == AwarenessType.STRONG_HABIT:
                    has_strong = True
                    break

    coherence_level = "undefined"
    if coherence_state is not None:
        cl = getattr(coherence_state, "coherence_level", None)
        if cl is not None:
            coherence_level = getattr(cl, "value", str(cl))

    narrative_trend = "undefined"
    if narrative_state is not None:
        nt = getattr(narrative_state, "trend", None)
        if nt is not None:
            narrative_trend = getattr(nt, "value", str(nt))

    return SelfObservationCompanion(
        has_difference=has_difference,
        difference_magnitude=diff_magnitude,
        difference_nature=diff_nature,
        tendency_description=tendency_desc,
        has_strong_tendency=has_strong,
        coherence_level=coherence_level,
        narrative_trend=narrative_trend,
    )


def _extract_topics(stm_entries: Optional[Any]) -> tuple[str, ...]:
    """Extract topics from STM entries."""
    if stm_entries is None:
        return ()
    entries = getattr(stm_entries, "entries", [])
    topics: list[str] = []
    for entry in entries:
        entry_topics = getattr(entry, "topics", [])
        for t in entry_topics:
            if t and t not in topics:
                topics.append(t)
    return tuple(topics[:20])


def _extract_source_texts(stm_entries: Optional[Any]) -> tuple[str, ...]:
    """Extract source texts from STM entries."""
    if stm_entries is None:
        return ()
    entries = getattr(stm_entries, "entries", [])
    texts: list[str] = []
    for entry in entries:
        source = getattr(entry, "source_text", "")
        if source and source not in texts:
            texts.append(source)
    return tuple(texts[:10])


# =============================================================================
# Episodic Memory System
# =============================================================================

class EpisodicMemorySystem:
    """
    Episodic Memory System (エピソード記憶)

    出来事単位の経験を保持し、体験の文脈・感情・自己観測を
    随伴情報として付与して中長期的に保持する。

    CRITICAL CONSTRAINTS:
    - 記憶は参照のみで、意思決定に直接影響する経路を持たない
    - 責任評価や価値更新に接続しない
    - 出来事の善悪・成功失敗の判定をしない
    - 目的や価値の生成を行わず、ただ「経験の記録」に留める
    - 出来事の解釈は固定しない（再要約・再関連付けを許容）
    - 重要度や参照頻度により自然減衰を許容
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        self._config = config or EpisodicMemoryConfig()
        self._episodes: list[EpisodeEntry] = []
        self._links: list[EpisodeLink] = []
        self._total_recorded: int = 0
        self._total_compressions: int = 0
        self._last_store: Optional[EpisodeStore] = None

    def record_episode(
        self,
        stm_entries: Optional[Any] = None,
        emotional_state: Optional[Any] = None,
        difference_summary: Optional[Any] = None,
        tendency_awareness: Optional[Any] = None,
        coherence_state: Optional[Any] = None,
        narrative_state: Optional[Any] = None,
        external_context: Optional[Any] = None,
    ) -> EpisodeStore:
        """
        Record a new episode from current state observations.

        入力は全て読み取り専用で参照する。
        """
        current_time = time.time()

        # Classify episode type
        ep_type = classify_episode_type(
            stm_entries=stm_entries,
            emotional_state=emotional_state,
            difference_summary=difference_summary,
            external_context=external_context,
        )

        # Compute importance
        importance = compute_importance(
            emotional_state=emotional_state,
            difference_summary=difference_summary,
            stm_entries=stm_entries,
        )

        # Build companions
        emotional_companion = _build_emotional_companion(emotional_state)
        self_obs_companion = _build_self_observation_companion(
            difference_summary=difference_summary,
            tendency_awareness=tendency_awareness,
            coherence_state=coherence_state,
            narrative_state=narrative_state,
        )

        # Extract topics and source texts
        topics = _extract_topics(stm_entries)
        source_texts = _extract_source_texts(stm_entries)

        # Build context summary
        context_summary = ""
        if external_context is not None:
            if isinstance(external_context, str):
                context_summary = external_context[:200]
            else:
                weight = getattr(external_context, "weight", 0.5)
                pace = getattr(external_context, "pace", 0.5)
                context_summary = f"weight={weight:.2f}, pace={pace:.2f}"

        # Generate summary
        summary = generate_episode_summary(
            ep_type, stm_entries, emotional_state, context_summary,
        )

        # Create episode entry
        episode = EpisodeEntry(
            episode_id=_generate_episode_id(),
            episode_type=ep_type,
            summary=summary,
            topics=topics,
            source_texts=source_texts,
            timestamp=current_time,
            duration_estimate=0.0,
            emotional_companion=emotional_companion,
            self_observation_companion=self_obs_companion,
            context_summary=context_summary,
            importance=importance,
            vividness=1.0,
            reference_count=0,
            reinterpretation_count=0,
            is_compressed=False,
            compressed_episode_ids=(),
        )

        # Add episode
        self._episodes.append(episode)
        self._total_recorded += 1

        # Generate links to existing episodes
        self._generate_links(episode)

        # Build and return store
        return self._build_store(current_time)

    def decay_episodes(self) -> EpisodeStore:
        """
        Apply natural decay to all episodes.

        重要度・参照頻度による減衰変調。
        """
        current_time = time.time()
        new_episodes: list[EpisodeEntry] = []

        for ep in self._episodes:
            # Compute effective decay rate
            importance_mod = _importance_to_decay_modifier(ep.importance)
            reference_mod = _reference_to_decay_modifier(ep.reference_count)
            effective_rate = (
                self._config.base_decay_rate
                * importance_mod
                * self._config.importance_decay_modifier
                + self._config.base_decay_rate
                * reference_mod
                * self._config.reference_decay_modifier
            )
            effective_rate = min(effective_rate, self._config.base_decay_rate * 1.5)

            new_vividness = ep.vividness - effective_rate
            if new_vividness > 0.0:
                new_episodes.append(ep.with_vividness(new_vividness))
            # Episodes with vividness <= 0 are removed

        # Clean up links for removed episodes
        remaining_ids = {e.episode_id for e in new_episodes}
        self._links = [
            link for link in self._links
            if (
                link.from_episode_id in remaining_ids
                and link.to_episode_id in remaining_ids
            )
        ]
        self._episodes = new_episodes

        return self._build_store(current_time)

    def compress_old_episodes(self) -> EpisodeStore:
        """
        Compress old dim episodes into COMPOSITE episodes.

        薄れたエピソードをCOMPOSITEに圧縮する。
        """
        current_time = time.time()

        # Find compressible episodes
        compressible = [
            ep for ep in self._episodes
            if (
                ep.vividness <= self._config.compression_vividness_threshold
                and not ep.is_compressed
                and ep.vividness > 0.0
            )
        ]

        if len(compressible) < self._config.min_episodes_for_compression:
            return self._build_store(current_time)

        # Group by episode type for compression
        by_type: dict[str, list[EpisodeEntry]] = {}
        for ep in compressible:
            key = ep.episode_type.value
            by_type.setdefault(key, []).append(ep)

        for ep_type_val, group in by_type.items():
            if len(group) < self._config.min_episodes_for_compression:
                continue

            # Compress this group
            compressed_ids = tuple(ep.episode_id for ep in group)
            all_topics: list[str] = []
            for ep in group:
                for t in ep.topics:
                    if t not in all_topics:
                        all_topics.append(t)

            type_label = EpisodeType(ep_type_val).value
            summary = (
                f"Compressed {len(group)} {type_label} episodes"
                f" spanning {len(all_topics)} topics"
            )

            composite = EpisodeEntry(
                episode_id=_generate_episode_id(),
                episode_type=EpisodeType.COMPOSITE,
                summary=summary,
                topics=tuple(all_topics[:20]),
                source_texts=(),
                timestamp=max(ep.timestamp for ep in group),
                duration_estimate=sum(ep.duration_estimate for ep in group),
                emotional_companion=group[-1].emotional_companion,
                self_observation_companion=group[-1].self_observation_companion,
                context_summary=f"Composite of {len(group)} episodes",
                importance=max(
                    (ep.importance for ep in group),
                    key=lambda x: list(ImportanceLevel).index(x),
                ),
                vividness=self._config.compression_result_vividness,
                reference_count=0,
                reinterpretation_count=0,
                is_compressed=True,
                compressed_episode_ids=compressed_ids,
            )

            # Remove compressed episodes and their links
            remove_ids = set(compressed_ids)
            self._episodes = [
                ep for ep in self._episodes
                if ep.episode_id not in remove_ids
            ]
            self._links = [
                link for link in self._links
                if (
                    link.from_episode_id not in remove_ids
                    and link.to_episode_id not in remove_ids
                )
            ]

            self._episodes.append(composite)
            self._total_compressions += 1

        return self._build_store(current_time)

    def search_episodes(
        self,
        topics: Optional[list[str]] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        emotion_label: Optional[str] = None,
        valence_range: Optional[tuple[float, float]] = None,
        min_importance: Optional[ImportanceLevel] = None,
        mode: SearchMode = SearchMode.COMBINED,
        max_results: Optional[int] = None,
    ) -> list[EpisodeEntry]:
        """
        Search episodes by various criteria.

        検索は参照のみ。判断に影響しない。
        """
        max_res = max_results or self._config.default_max_results
        candidates = list(self._episodes)

        if mode == SearchMode.BY_TOPIC and topics:
            candidates = self._filter_by_topics(candidates, topics)
        elif mode == SearchMode.BY_TIME:
            candidates = self._filter_by_time(candidates, time_start, time_end)
        elif mode == SearchMode.BY_EMOTION:
            candidates = self._filter_by_emotion(
                candidates, emotion_label, valence_range,
            )
        elif mode == SearchMode.BY_IMPORTANCE and min_importance is not None:
            candidates = self._filter_by_importance(candidates, min_importance)
        elif mode == SearchMode.COMBINED:
            if topics:
                candidates = self._filter_by_topics(candidates, topics)
            if time_start is not None or time_end is not None:
                candidates = self._filter_by_time(
                    candidates, time_start, time_end,
                )
            if emotion_label is not None or valence_range is not None:
                candidates = self._filter_by_emotion(
                    candidates, emotion_label, valence_range,
                )
            if min_importance is not None:
                candidates = self._filter_by_importance(
                    candidates, min_importance,
                )

        # Sort by vividness (most vivid first)
        candidates.sort(key=lambda e: e.vividness, reverse=True)
        return candidates[:max_res]

    def reference_episode(self, episode_id: str) -> None:
        """
        Mark an episode as referenced (boosts vividness).

        参照によりvividness回復。
        """
        for i, ep in enumerate(self._episodes):
            if ep.episode_id == episode_id:
                referenced = ep.with_reference()
                boosted = referenced.with_vividness(
                    referenced.vividness + self._config.reference_vividness_boost,
                )
                self._episodes[i] = boosted
                return

    def reinterpret_episode(
        self,
        episode_id: str,
        new_summary: str,
        new_type: Optional[EpisodeType] = None,
    ) -> None:
        """
        Reinterpret an episode (解釈は固定しない、可逆性).

        後からの再要約や関連付け更新を許容する。
        """
        for i, ep in enumerate(self._episodes):
            if ep.episode_id == episode_id:
                self._episodes[i] = ep.reinterpret(new_summary, new_type)
                return

    def get_store(self) -> EpisodeStore:
        """Get current store snapshot."""
        return self._build_store(time.time())

    def get_last_store(self) -> Optional[EpisodeStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _generate_links(self, new_episode: EpisodeEntry) -> None:
        """Generate links between the new episode and existing episodes."""
        link_count = 0
        for existing in reversed(self._episodes[:-1]):
            if link_count >= self._config.max_links_per_episode:
                break

            # Temporal proximity
            temporal = compute_temporal_proximity(
                new_episode.timestamp,
                existing.timestamp,
                self._config.temporal_proximity_window,
            )
            if temporal >= self._config.link_strength_threshold:
                self._links.append(EpisodeLink(
                    from_episode_id=existing.episode_id,
                    to_episode_id=new_episode.episode_id,
                    link_type=EpisodeLinkType.TEMPORAL_PROXIMITY,
                    strength=temporal,
                    description="Temporally proximate episodes",
                ))
                link_count += 1
                if link_count >= self._config.max_links_per_episode:
                    break

            # Topic overlap
            topic_sim = compute_topic_overlap(
                new_episode.topics, existing.topics,
            )
            if topic_sim >= self._config.topic_overlap_threshold:
                self._links.append(EpisodeLink(
                    from_episode_id=existing.episode_id,
                    to_episode_id=new_episode.episode_id,
                    link_type=EpisodeLinkType.TOPIC_OVERLAP,
                    strength=topic_sim,
                    description=f"Shared topics (overlap={topic_sim:.2f})",
                ))
                link_count += 1
                if link_count >= self._config.max_links_per_episode:
                    break

            # Emotional similarity
            emotional_sim = compute_emotional_similarity(
                new_episode.emotional_companion,
                existing.emotional_companion,
            )
            if emotional_sim >= self._config.emotional_similarity_threshold:
                self._links.append(EpisodeLink(
                    from_episode_id=existing.episode_id,
                    to_episode_id=new_episode.episode_id,
                    link_type=EpisodeLinkType.EMOTIONAL_SIMILARITY,
                    strength=emotional_sim,
                    description=f"Emotional similarity ({emotional_sim:.2f})",
                ))
                link_count += 1

    def _filter_by_topics(
        self,
        candidates: list[EpisodeEntry],
        topics: list[str],
    ) -> list[EpisodeEntry]:
        """Filter episodes by topic overlap."""
        topic_tuple = tuple(topics)
        return [
            ep for ep in candidates
            if compute_topic_overlap(ep.topics, topic_tuple)
            >= self._config.topic_overlap_threshold
        ]

    def _filter_by_time(
        self,
        candidates: list[EpisodeEntry],
        time_start: Optional[float],
        time_end: Optional[float],
    ) -> list[EpisodeEntry]:
        """Filter episodes by time range."""
        result = candidates
        if time_start is not None:
            result = [ep for ep in result if ep.timestamp >= time_start]
        if time_end is not None:
            result = [ep for ep in result if ep.timestamp <= time_end]
        return result

    def _filter_by_emotion(
        self,
        candidates: list[EpisodeEntry],
        emotion_label: Optional[str],
        valence_range: Optional[tuple[float, float]],
    ) -> list[EpisodeEntry]:
        """Filter episodes by emotional criteria."""
        result = []
        for ep in candidates:
            if ep.emotional_companion is None:
                continue
            if emotion_label is not None:
                if emotion_label.lower() not in ep.emotional_companion.primary_emotion.lower():
                    continue
            if valence_range is not None:
                v = ep.emotional_companion.valence
                if v < valence_range[0] or v > valence_range[1]:
                    continue
            result.append(ep)
        return result

    def _filter_by_importance(
        self,
        candidates: list[EpisodeEntry],
        min_importance: ImportanceLevel,
    ) -> list[EpisodeEntry]:
        """Filter episodes by minimum importance level."""
        levels = list(ImportanceLevel)
        min_idx = levels.index(min_importance)
        return [
            ep for ep in candidates
            if levels.index(ep.importance) >= min_idx
        ]

    def _build_store(self, current_time: float) -> EpisodeStore:
        """Build an EpisodeStore snapshot."""
        active = [ep for ep in self._episodes if ep.vividness > 0.0]
        compressed = [ep for ep in self._episodes if ep.is_compressed]
        avg_vividness = (
            sum(ep.vividness for ep in self._episodes) / len(self._episodes)
            if self._episodes else 0.0
        )

        description = _generate_store_description(
            len(self._episodes),
            len(active),
            len(compressed),
            avg_vividness,
            self._total_compressions,
        )

        store = EpisodeStore(
            episodes=tuple(self._episodes),
            links=tuple(self._links),
            total_episodes_recorded=self._total_recorded,
            total_compressions=self._total_compressions,
            average_vividness=round(avg_vividness, 4),
            active_episode_count=len(active),
            compressed_episode_count=len(compressed),
            timestamp=current_time,
            description=description,
        )
        self._last_store = store
        return store


def _generate_store_description(
    total: int,
    active: int,
    compressed: int,
    avg_vividness: float,
    total_compressions: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No episodes recorded yet."

    parts = [f"{active} active episodes out of {total} total"]

    if compressed > 0:
        parts.append(f"{compressed} composite episodes")

    if total_compressions > 0:
        parts.append(f"{total_compressions} compressions performed")

    parts.append(f"average vividness: {avg_vividness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def record_from_chain(
    system: EpisodicMemorySystem,
    emotional_state: Optional[Any] = None,
    short_term_memory: Optional[Any] = None,
    tendency_awareness: Optional[Any] = None,
    difference_summary: Optional[Any] = None,
    coherence_state: Optional[Any] = None,
    narrative_state: Optional[Any] = None,
    external_context: Optional[Any] = None,
) -> EpisodeStore:
    """
    Record an episode from the self-observation chain.

    自己観測チェーンからの統合ヘルパー。
    各入力は読み取り専用で参照される。
    """
    return system.record_episode(
        stm_entries=short_term_memory,
        emotional_state=emotional_state,
        difference_summary=difference_summary,
        tendency_awareness=tendency_awareness,
        coherence_state=coherence_state,
        narrative_state=narrative_state,
        external_context=external_context,
    )


def generate_episodic_memory_tags(
    store: EpisodeStore,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from EpisodeStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags = []

    if not store.has_episodes():
        tags.append({
            "category": "EPISODIC_MEMORY",
            "label": "no_episodes",
            "description": "No episodic memories exist yet",
            "weight": 0.05 * scale,
        })
        return tags

    # Episode count tag
    tags.append({
        "category": "EPISODIC_MEMORY_COUNT",
        "label": f"episodes_{store.active_episode_count}",
        "description": (
            f"Episodic memory holds {store.active_episode_count} active episodes"
        ),
        "weight": 0.06 * scale,
    })

    # Vividness tag
    if store.average_vividness >= 0.7:
        vividness_label = "vivid_memories"
    elif store.average_vividness >= 0.4:
        vividness_label = "moderate_memories"
    else:
        vividness_label = "fading_memories"
    tags.append({
        "category": "EPISODIC_MEMORY_VIVIDNESS",
        "label": vividness_label,
        "description": (
            f"Average episode vividness: {store.average_vividness:.2f}"
        ),
        "weight": 0.07 * scale,
    })

    # Compression tag
    if store.compressed_episode_count > 0:
        tags.append({
            "category": "EPISODIC_MEMORY_COMPRESSION",
            "label": f"compressed_{store.compressed_episode_count}",
            "description": (
                f"{store.compressed_episode_count} composite episodes from compression"
            ),
            "weight": 0.04 * scale,
        })

    # Most recent episode
    fresh = store.get_fresh_episodes()
    if fresh:
        most_recent = fresh[-1]
        tags.append({
            "category": "EPISODIC_MEMORY_RECENT",
            "label": f"recent_{most_recent.episode_type.value}",
            "description": most_recent.summary[:100],
            "weight": 0.08 * scale,
        })

    # Integrated description
    tags.append({
        "category": "EPISODIC_MEMORY_INTEGRATED",
        "label": "episodic_awareness",
        "description": store.description,
        "weight": 0.1 * scale,
    })

    return tags


def get_episodic_memory_summary(store: EpisodeStore) -> str:
    """Get human-readable summary. For introspection/logging only."""
    lines = [
        "=== Episodic Memory State ===",
        f"Total episodes: {len(store.episodes)}",
        f"Active episodes: {store.active_episode_count}",
        f"Composite episodes: {store.compressed_episode_count}",
        f"Total recorded: {store.total_episodes_recorded}",
        f"Total compressions: {store.total_compressions}",
        f"Average vividness: {store.average_vividness:.2f}",
        "",
    ]

    fresh = store.get_fresh_episodes()
    if fresh:
        lines.append("Recent episodes:")
        for ep in fresh[-5:]:
            lines.append(
                f"  [{ep.episode_type.value}] {ep.summary[:80]}"
                f" (vividness: {ep.get_decay_state().value})"
            )
        lines.append("")

    lines.append(f"Integrated: {store.description}")
    return "\n".join(lines)


def get_episodic_memory_for_introspection(
    store: EpisodeStore,
) -> dict[str, Any]:
    """
    Get structured episodic memory data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    type_counts: dict[str, int] = {}
    for ep in store.episodes:
        type_counts[ep.episode_type.value] = (
            type_counts.get(ep.episode_type.value, 0) + 1
        )

    importance_counts: dict[str, int] = {}
    for ep in store.episodes:
        importance_counts[ep.importance.value] = (
            importance_counts.get(ep.importance.value, 0) + 1
        )

    return {
        "has_episodes": store.has_episodes(),
        "total_episodes": len(store.episodes),
        "active_episode_count": store.active_episode_count,
        "compressed_episode_count": store.compressed_episode_count,
        "total_recorded": store.total_episodes_recorded,
        "total_compressions": store.total_compressions,
        "average_vividness": store.average_vividness,
        "episode_type_distribution": type_counts,
        "importance_distribution": importance_counts,
        "link_count": len(store.links),
        "description": store.description,
        "timestamp": store.timestamp,
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: EpisodeStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    エピソード記憶は経験の記録としてのみ扱う。
    """
    public_attrs = [a for a in dir(store) if not a.startswith("_")]

    for attr in public_attrs:
        if callable(getattr(store, attr)):
            continue
        value = getattr(store, attr)

        if attr in ("timestamp",):
            continue
        if isinstance(value, str):
            continue
        if isinstance(value, Enum):
            continue
        if isinstance(value, tuple):
            continue
        if isinstance(value, (int, float)) and attr not in ("timestamp",):
            # Numeric fields that are purely descriptive statistics
            if attr in (
                "total_episodes_recorded", "total_compressions",
                "average_vividness", "active_episode_count",
                "compressed_episode_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: EpisodicMemorySystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    記憶から目標を生成しない。
    """
    forbidden = [
        "generate_goal", "create_goal", "set_goal",
        "force", "fix", "repair", "restore", "correct",
        "prescribe", "enforce",
    ]
    methods = [
        m for m in dir(system)
        if not m.startswith("_") and callable(getattr(system, m))
    ]
    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden:
            if pattern in method_lower:
                return False
    return True


def verify_read_only_principle(system: EpisodicMemorySystem) -> bool:
    """
    Verify the system does not write to external systems.

    読み取り専用原則を破らない。
    """
    forbidden = [
        "update_emotion", "update_memory", "update_tendency",
        "update_value", "update_decision", "update_responsibility",
        "set_emotion", "set_memory", "set_tendency",
        "modify_bias", "apply_to_decision",
    ]
    methods = [
        m for m in dir(system)
        if not m.startswith("_") and callable(getattr(system, m))
    ]
    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden:
            if pattern in method_lower:
                return False
    return True


def verify_no_value_modification(system: EpisodicMemorySystem) -> bool:
    """
    Verify the system does not modify values or beliefs.

    価値・信念の形成や更新を直接行わない。
    """
    forbidden = [
        "update_value", "set_value", "modify_value",
        "update_belief", "set_belief", "modify_belief",
        "define_identity", "set_identity", "fix_identity",
        "evaluate_morality", "judge_action",
    ]
    methods = [
        m for m in dir(system)
        if not m.startswith("_") and callable(getattr(system, m))
    ]
    for method in methods:
        method_lower = method.lower()
        for pattern in forbidden:
            if pattern in method_lower:
                return False
    return True


# =============================================================================
# Convenience / Persistence
# =============================================================================

def create_config(
    max_episodes: int = 200,
    base_decay_rate: float = 0.02,
    compression_vividness_threshold: float = 0.15,
    default_max_results: int = 10,
) -> EpisodicMemoryConfig:
    """Create a custom configuration."""
    return EpisodicMemoryConfig(
        max_episodes=max_episodes,
        base_decay_rate=base_decay_rate,
        compression_vividness_threshold=compression_vividness_threshold,
        default_max_results=default_max_results,
    )


def create_empty_store() -> EpisodeStore:
    """Create an empty episode store."""
    return EpisodeStore.empty()


def create_system(
    config: Optional[EpisodicMemoryConfig] = None,
) -> EpisodicMemorySystem:
    """Create a new EpisodicMemorySystem."""
    return EpisodicMemorySystem(config=config)


def save_episodic_memory(
    store: EpisodeStore,
    filepath: str,
) -> None:
    """Save episodic memory to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_episodic_memory(filepath: str) -> EpisodeStore:
    """Load episodic memory from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EpisodeStore.from_dict(data)
