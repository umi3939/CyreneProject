"""
Intrinsic Motivation (自発的内的動機)

感情や傾向から湧き上がる内的な推進力を弱く形成するモジュール。

現状の課題:
目的系は候補生成と選択の仕組みはあるが、「なぜそれをしたいか」の動機源がない。
自発的内的動機は、感情や傾向から湧き上がる内的な推進力を弱く形成するために必要。
動機は価値や信念を固定しない。行動の決定ではなく、内側で生じる「向き」の痕跡として
自己形成の前段条件を整える。

CRITICAL DESIGN PRINCIPLES:
- 入力は全て Optional[Any] + duck typing（循環import回避）
- 価値・信念・目標の直接設計しない
- 行動の最適化しない
- 正誤・善悪・成功失敗の判断しない
- 規範的方向付けしない
- 自我・人格・信念・評価を直接形成しない
- 接続先: 内省記録層への参照素材、目的候補層への弱い付随情報
- 非接続: 判断選択層・価値更新層・責任評価層・外部出力直接生成層
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class MotiveSourceType(Enum):
    """
    動機の入力源がどこ由来か。

    評価的ではなく、記述的な分類。
    """
    EMOTION = "emotion"              # 感情状態由来
    TENDENCY = "tendency"            # 反復傾向由来
    GOAL_VECTOR = "goal_vector"      # 方向ベクトル由来
    GOAL_CANDIDATE = "goal_candidate"  # 目的候補由来
    MIXED = "mixed"                  # 複数ソース混合


class MotiveAffinity(Enum):
    """
    動機衝動の根拠・性質。

    感情的高揚/習慣的/方向的/志向的/複合/未確定。
    """
    EMOTIONAL_SURGE = "emotional_surge"  # 感情的高揚
    HABITUAL = "habitual"                # 習慣的
    DIRECTIONAL = "directional"          # 方向的
    ASPIRATIONAL = "aspirational"        # 志向的
    COMPOSITE = "composite"              # 複合
    UNDEFINED = "undefined"              # 未確定


class MotiveStrength(Enum):
    """
    動機の強度。

    閾値: 0.7/0.4/0.2/0.05。
    """
    STRONG = "strong"          # 強い動機
    MODERATE = "moderate"      # 中程度
    WEAK = "weak"              # 弱い動機
    FAINT = "faint"            # かすかな動機
    UNDEFINED = "undefined"    # 未確定


class MotiveFreshness(Enum):
    """
    動機の新鮮度。

    閾値: 0.8/0.6/0.4/0.15。
    """
    FRESH = "fresh"        # 直近
    RECENT = "recent"      # まだ活発
    AGING = "aging"        # 薄れつつある
    STALE = "stale"        # ほぼ消えかけ
    FADED = "faded"        # 消失寸前


# =============================================================================
# Level Determination Functions (Pure)
# =============================================================================

def determine_freshness_level(freshness: float) -> MotiveFreshness:
    """Determine MotiveFreshness from a numeric freshness value (0.0-1.0)."""
    if freshness >= 0.8:
        return MotiveFreshness.FRESH
    elif freshness >= 0.6:
        return MotiveFreshness.RECENT
    elif freshness >= 0.4:
        return MotiveFreshness.AGING
    elif freshness >= 0.15:
        return MotiveFreshness.STALE
    else:
        return MotiveFreshness.FADED


def determine_strength_level(strength: float) -> MotiveStrength:
    """Determine MotiveStrength from a numeric strength value (0.0-1.0)."""
    if strength >= 0.7:
        return MotiveStrength.STRONG
    elif strength >= 0.4:
        return MotiveStrength.MODERATE
    elif strength >= 0.2:
        return MotiveStrength.WEAK
    elif strength >= 0.05:
        return MotiveStrength.FAINT
    else:
        return MotiveStrength.UNDEFINED


# =============================================================================
# ID / Key Generation
# =============================================================================

def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex[:12]


def generate_motive_key(text: str) -> str:
    """Generate a motive key from text via MD5 hash."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# =============================================================================
# Emotion Label Mapping (from stm_emotion_coupling.py)
# =============================================================================

_EMOTION_LABEL_MAP: dict[str, str] = {
    "happy": "joy",
    "sad": "sorrow",
    "angry": "anger",
    "surprised": "surprise",
    "scared": "fear",
    "loving": "love",
    "teasing": "fun",
    "joy": "joy",
    "sorrow": "sorrow",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "love": "love",
    "fun": "fun",
}


# =============================================================================
# Core Dataclasses (all frozen)
# =============================================================================

@dataclass(frozen=True)
class MotiveImpulse:
    """
    動機衝動 — 動機に付帯する1単位の衝動。

    NOT evaluative. NOT prescriptive.
    """
    impulse_id: str
    label: str                       # emotion_joy, tendency_approach, etc.
    intensity: float                 # 0.0〜1.0
    valence: float                   # -1.0〜1.0
    freshness: float                 # 0.0〜1.0
    reference_count: int
    affinity: MotiveAffinity
    timestamp: str
    source_description: str

    def get_freshness_level(self) -> MotiveFreshness:
        """Get abstract freshness level."""
        return determine_freshness_level(self.freshness)

    def with_freshness(self, new_freshness: float) -> MotiveImpulse:
        """Create a copy with updated freshness."""
        return MotiveImpulse(
            impulse_id=self.impulse_id,
            label=self.label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=max(0.0, min(1.0, new_freshness)),
            reference_count=self.reference_count,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def with_intensity(self, new_intensity: float) -> MotiveImpulse:
        """Create a copy with updated intensity."""
        return MotiveImpulse(
            impulse_id=self.impulse_id,
            label=self.label,
            intensity=max(0.0, min(1.0, new_intensity)),
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def with_reference(self) -> MotiveImpulse:
        """Create a copy with incremented reference count."""
        return MotiveImpulse(
            impulse_id=self.impulse_id,
            label=self.label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count + 1,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def reattach(self, new_affinity: MotiveAffinity) -> MotiveImpulse:
        """Create a copy with updated affinity."""
        return MotiveImpulse(
            impulse_id=self.impulse_id,
            label=self.label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count,
            affinity=new_affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )


@dataclass(frozen=True)
class MotiveLink:
    """
    根拠リンク。

    NOT evaluative. NOT prescriptive.
    """
    link_id: str
    motive_id: str
    source_type: MotiveSourceType
    source_description: str
    contribution: float  # 0.0〜1.0


@dataclass(frozen=True)
class MotiveEntry:
    """
    動機エントリ — 複数衝動を並立保持するコア構造。

    NOT evaluative. NOT prescriptive.
    """
    motive_id: str
    motive_key: str                          # hashされた動機識別子
    motive_summary: str                      # 動機内容の要約
    impulses: tuple[MotiveImpulse, ...]      # 複数衝動を並立
    motive_links: tuple[str, ...]            # MotiveLink ID群
    freshness: float                         # 0.0〜1.0
    reference_count: int
    creation_timestamp: str
    last_reference_timestamp: str
    revision_count: int
    undetermined_aspects: tuple[str, ...]

    def with_freshness(self, new_freshness: float) -> MotiveEntry:
        """Create a copy with updated freshness."""
        return MotiveEntry(
            motive_id=self.motive_id,
            motive_key=self.motive_key,
            motive_summary=self.motive_summary,
            impulses=self.impulses,
            motive_links=self.motive_links,
            freshness=max(0.0, min(1.0, new_freshness)),
            reference_count=self.reference_count,
            creation_timestamp=self.creation_timestamp,
            last_reference_timestamp=self.last_reference_timestamp,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_reference(self) -> MotiveEntry:
        """Create a copy with incremented reference count."""
        return MotiveEntry(
            motive_id=self.motive_id,
            motive_key=self.motive_key,
            motive_summary=self.motive_summary,
            impulses=self.impulses,
            motive_links=self.motive_links,
            freshness=self.freshness,
            reference_count=self.reference_count + 1,
            creation_timestamp=self.creation_timestamp,
            last_reference_timestamp=str(time.time()),
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_impulses(self, new_impulses: tuple[MotiveImpulse, ...]) -> MotiveEntry:
        """Create a copy with updated impulses."""
        return MotiveEntry(
            motive_id=self.motive_id,
            motive_key=self.motive_key,
            motive_summary=self.motive_summary,
            impulses=new_impulses,
            motive_links=self.motive_links,
            freshness=self.freshness,
            reference_count=self.reference_count,
            creation_timestamp=self.creation_timestamp,
            last_reference_timestamp=self.last_reference_timestamp,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def revise_summary(self, new_summary: str) -> MotiveEntry:
        """Create a copy with revised summary."""
        return MotiveEntry(
            motive_id=self.motive_id,
            motive_key=self.motive_key,
            motive_summary=new_summary,
            impulses=self.impulses,
            motive_links=self.motive_links,
            freshness=self.freshness,
            reference_count=self.reference_count,
            creation_timestamp=self.creation_timestamp,
            last_reference_timestamp=self.last_reference_timestamp,
            revision_count=self.revision_count + 1,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_added_impulse(self, impulse: MotiveImpulse) -> MotiveEntry:
        """Create a copy with an additional impulse."""
        return MotiveEntry(
            motive_id=self.motive_id,
            motive_key=self.motive_key,
            motive_summary=self.motive_summary,
            impulses=self.impulses + (impulse,),
            motive_links=self.motive_links,
            freshness=self.freshness,
            reference_count=self.reference_count,
            creation_timestamp=self.creation_timestamp,
            last_reference_timestamp=self.last_reference_timestamp,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )


@dataclass(frozen=True)
class MotiveStore:
    """
    不変スナップショット — 自発的内的動機の全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    entries: tuple[MotiveEntry, ...]
    motive_links: tuple[MotiveLink, ...]
    total_entries_created: int
    total_impulses_created: int
    total_revisions: int
    total_expirations: int
    average_freshness: float
    average_impulse_count: float
    active_entry_count: int
    timestamp: str
    description: str

    def has_entries(self) -> bool:
        return len(self.entries) > 0

    def get_active_entries(self, stale_threshold: float = 0.15) -> tuple[MotiveEntry, ...]:
        """Get entries with freshness above the stale threshold."""
        return tuple(
            e for e in self.entries
            if e.freshness > stale_threshold
        )

    def get_entries_for_key(self, motive_key: str) -> tuple[MotiveEntry, ...]:
        """Get entries for a specific motive key."""
        return tuple(
            e for e in self.entries
            if e.motive_key == motive_key
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entries": [
                _entry_to_dict(e) for e in self.entries
            ],
            "motive_links": [
                {
                    "link_id": ml.link_id,
                    "motive_id": ml.motive_id,
                    "source_type": ml.source_type.value,
                    "source_description": ml.source_description,
                    "contribution": ml.contribution,
                }
                for ml in self.motive_links
            ],
            "total_entries_created": self.total_entries_created,
            "total_impulses_created": self.total_impulses_created,
            "total_revisions": self.total_revisions,
            "total_expirations": self.total_expirations,
            "average_freshness": self.average_freshness,
            "average_impulse_count": self.average_impulse_count,
            "active_entry_count": self.active_entry_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotiveStore:
        """Create from dictionary."""
        entries = tuple(
            _entry_from_dict(e) for e in data.get("entries", [])
        )
        motive_links = tuple(
            MotiveLink(
                link_id=ml["link_id"],
                motive_id=ml.get("motive_id", ""),
                source_type=MotiveSourceType(ml.get("source_type", "mixed")),
                source_description=ml.get("source_description", ""),
                contribution=ml.get("contribution", 0.0),
            )
            for ml in data.get("motive_links", [])
        )
        return cls(
            entries=entries,
            motive_links=motive_links,
            total_entries_created=data.get("total_entries_created", 0),
            total_impulses_created=data.get("total_impulses_created", 0),
            total_revisions=data.get("total_revisions", 0),
            total_expirations=data.get("total_expirations", 0),
            average_freshness=data.get("average_freshness", 0.0),
            average_impulse_count=data.get("average_impulse_count", 0.0),
            active_entry_count=data.get("active_entry_count", 0),
            timestamp=data.get("timestamp", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class IntrinsicMotivationConfig:
    """
    Configuration for the intrinsic motivation system.

    These parameters control motive lifecycle.
    They do NOT affect decisions.
    """
    max_entries: int = 150                     # 動機エントリ上限
    max_impulses_per_entry: int = 7            # 7感情次元に対応
    base_decay_rate: float = 0.025             # 中期なので適度な減衰
    impulse_decay_rate: float = 0.02
    freshness_boost_on_reference: float = 0.10
    impulse_boost_on_reference: float = 0.06
    stale_threshold: float = 0.15
    min_freshness_for_retention: float = 0.05
    max_motive_links: int = 10
    min_intensity_for_motive: float = 0.1


# =============================================================================
# Serialization Helpers
# =============================================================================

def _impulse_to_dict(impulse: MotiveImpulse) -> dict[str, Any]:
    """Convert a MotiveImpulse to a dictionary."""
    return {
        "impulse_id": impulse.impulse_id,
        "label": impulse.label,
        "intensity": impulse.intensity,
        "valence": impulse.valence,
        "freshness": impulse.freshness,
        "reference_count": impulse.reference_count,
        "affinity": impulse.affinity.value,
        "timestamp": impulse.timestamp,
        "source_description": impulse.source_description,
    }


def _impulse_from_dict(data: dict[str, Any]) -> MotiveImpulse:
    """Create a MotiveImpulse from a dictionary."""
    return MotiveImpulse(
        impulse_id=data["impulse_id"],
        label=data.get("label", ""),
        intensity=data.get("intensity", 0.0),
        valence=data.get("valence", 0.0),
        freshness=data.get("freshness", 0.0),
        reference_count=data.get("reference_count", 0),
        affinity=MotiveAffinity(data.get("affinity", "undefined")),
        timestamp=data.get("timestamp", ""),
        source_description=data.get("source_description", ""),
    )


def _entry_to_dict(entry: MotiveEntry) -> dict[str, Any]:
    """Convert a MotiveEntry to a dictionary."""
    return {
        "motive_id": entry.motive_id,
        "motive_key": entry.motive_key,
        "motive_summary": entry.motive_summary,
        "impulses": [_impulse_to_dict(i) for i in entry.impulses],
        "motive_links": list(entry.motive_links),
        "freshness": entry.freshness,
        "reference_count": entry.reference_count,
        "creation_timestamp": entry.creation_timestamp,
        "last_reference_timestamp": entry.last_reference_timestamp,
        "revision_count": entry.revision_count,
        "undetermined_aspects": list(entry.undetermined_aspects),
    }


def _entry_from_dict(data: dict[str, Any]) -> MotiveEntry:
    """Create a MotiveEntry from a dictionary."""
    return MotiveEntry(
        motive_id=data["motive_id"],
        motive_key=data.get("motive_key", ""),
        motive_summary=data.get("motive_summary", ""),
        impulses=tuple(
            _impulse_from_dict(i) for i in data.get("impulses", [])
        ),
        motive_links=tuple(data.get("motive_links", ())),
        freshness=data.get("freshness", 0.0),
        reference_count=data.get("reference_count", 0),
        creation_timestamp=data.get("creation_timestamp", ""),
        last_reference_timestamp=data.get("last_reference_timestamp", ""),
        revision_count=data.get("revision_count", 0),
        undetermined_aspects=tuple(data.get("undetermined_aspects", ())),
    )


# =============================================================================
# Extraction Functions (Pure, Duck Typing)
# =============================================================================

def extract_from_emotion_state(
    emotion: Optional[Any],
    mood: Optional[Any],
) -> list[tuple[str, str, float, float, str, MotiveSourceType]]:
    """
    Extract motive materials from current emotion state.

    Returns list of (motive_key, label, intensity, valence, source_description, source_type).
    EmotionVector duck typing: joy, anger, sorrow, fear, surprise, love, fun.
    Mood duck typing: valence, arousal.
    各感情 >= 0.15 → 抽出候補, motive_key = "__emotion_motive__"
    label = f"emotion_{field_name}"
    """
    if emotion is None:
        return []

    results: list[tuple[str, str, float, float, str, MotiveSourceType]] = []
    motive_key = "__emotion_motive__"

    # Get mood valence
    mood_valence = 0.0
    if mood is not None:
        if hasattr(mood, "valence"):
            mood_valence = getattr(mood, "valence", 0.0)
        elif isinstance(mood, dict):
            mood_valence = mood.get("valence", 0.0)

    emotion_fields = ["joy", "anger", "sorrow", "fear", "surprise", "love", "fun"]

    # Valence mapping for each emotion
    emotion_valence_map = {
        "joy": 0.5, "anger": -0.5, "sorrow": -0.3,
        "fear": -0.4, "surprise": 0.1, "love": 0.6, "fun": 0.4,
    }

    # Handle EmotionVector object
    if hasattr(emotion, "joy"):
        for field_name in emotion_fields:
            value = getattr(emotion, field_name, 0.0)
            if isinstance(value, (int, float)) and value >= 0.15:
                base_valence = emotion_valence_map.get(field_name, 0.0)
                val = base_valence * 0.7 + mood_valence * 0.3 if isinstance(mood_valence, (int, float)) else base_valence
                label = f"emotion_{field_name}"
                results.append((
                    motive_key, label, min(1.0, value), val,
                    f"Emotion motive: {field_name}={value:.2f}",
                    MotiveSourceType.EMOTION,
                ))

    # Handle dict input
    elif isinstance(emotion, dict):
        for field_name in emotion_fields:
            value = emotion.get(field_name, 0.0)
            if isinstance(value, (int, float)) and value >= 0.15:
                base_valence = emotion_valence_map.get(field_name, 0.0)
                val = base_valence * 0.7 + mood_valence * 0.3 if isinstance(mood_valence, (int, float)) else base_valence
                label = f"emotion_{field_name}"
                results.append((
                    motive_key, label, min(1.0, value), val,
                    f"Emotion motive: {field_name}={value:.2f}",
                    MotiveSourceType.EMOTION,
                ))

    return results


def extract_from_tendencies(
    tendencies_state: Optional[Any],
) -> list[tuple[str, str, float, float, str, MotiveSourceType]]:
    """
    Extract motive materials from repeated tendencies.

    Returns list of (motive_key, label, intensity, valence, source_description, source_type).
    Duck typing: .tendencies[], .pattern.category.value, .strength
    strength >= 0.02 → intensity = min(1.0, strength * 5.0)
    motive_key = generate_motive_key(category_value), label = f"tendency_{category_value}"
    """
    if tendencies_state is None:
        return []

    results: list[tuple[str, str, float, float, str, MotiveSourceType]] = []

    # Handle object with tendencies attribute
    if hasattr(tendencies_state, "tendencies"):
        tendencies = getattr(tendencies_state, "tendencies", [])
        if not tendencies:
            return []

        for tendency in tendencies[:20]:
            strength = getattr(tendency, "strength", 0.0)
            if not isinstance(strength, (int, float)) or strength < 0.02:
                continue

            pattern = getattr(tendency, "pattern", None)
            if pattern is None:
                continue

            category = getattr(pattern, "category", None)
            if category is None:
                continue

            category_value = getattr(category, "value", str(category))
            intensity = min(1.0, strength * 5.0)
            motive_key = generate_motive_key(category_value)
            label = f"tendency_{category_value}"

            results.append((
                motive_key, label, intensity, 0.0,
                f"Tendency motive: {category_value} (strength={strength:.3f})",
                MotiveSourceType.TENDENCY,
            ))

    # Handle dict input
    elif isinstance(tendencies_state, dict):
        tendencies = tendencies_state.get("tendencies", [])
        if not isinstance(tendencies, list):
            return []

        for tendency in tendencies[:20]:
            if not isinstance(tendency, dict):
                continue

            strength = tendency.get("strength", 0.0)
            if not isinstance(strength, (int, float)) or strength < 0.02:
                continue

            pattern = tendency.get("pattern", {})
            if not isinstance(pattern, dict):
                continue

            category = pattern.get("category", "")
            if not category:
                continue

            # category may be a string or dict with "value"
            category_value = category if isinstance(category, str) else str(category)
            intensity = min(1.0, strength * 5.0)
            motive_key = generate_motive_key(category_value)
            label = f"tendency_{category_value}"

            results.append((
                motive_key, label, intensity, 0.0,
                f"Tendency motive: {category_value} (strength={strength:.3f})",
                MotiveSourceType.TENDENCY,
            ))

    return results


def extract_from_goal_vectors(
    vector_state: Optional[Any],
) -> list[tuple[str, str, float, float, str, MotiveSourceType]]:
    """
    Extract motive materials from proto-goal direction vectors.

    Returns list of (motive_key, label, intensity, valence, source_description, source_type).
    Duck typing: .vectors[], .vector_id, .direction(dict), .magnitude
    magnitude >= 0.1 → dominant direction key
    motive_key = generate_motive_key(vector_id), label = f"vector_{dominant_key}"
    """
    if vector_state is None:
        return []

    results: list[tuple[str, str, float, float, str, MotiveSourceType]] = []

    # Handle object with vectors attribute
    if hasattr(vector_state, "vectors"):
        vectors = getattr(vector_state, "vectors", [])
        if not vectors:
            return []

        for vector in vectors[:20]:
            magnitude = getattr(vector, "magnitude", 0.0)
            if not isinstance(magnitude, (int, float)) or magnitude < 0.1:
                continue

            vector_id = getattr(vector, "vector_id", "")
            if not vector_id:
                continue

            direction = getattr(vector, "direction", {})
            if not isinstance(direction, dict) or not direction:
                continue

            # Find dominant direction key
            dominant_key = max(direction, key=lambda k: abs(direction.get(k, 0.0)))
            motive_key = generate_motive_key(vector_id)
            label = f"vector_{dominant_key}"

            results.append((
                motive_key, label, min(1.0, magnitude), 0.0,
                f"Vector motive: {dominant_key} (magnitude={magnitude:.2f})",
                MotiveSourceType.GOAL_VECTOR,
            ))

    # Handle dict input
    elif isinstance(vector_state, dict):
        vectors = vector_state.get("vectors", [])
        if not isinstance(vectors, list):
            return []

        for vector in vectors[:20]:
            if not isinstance(vector, dict):
                continue

            magnitude = vector.get("magnitude", 0.0)
            if not isinstance(magnitude, (int, float)) or magnitude < 0.1:
                continue

            vector_id = vector.get("vector_id", "")
            if not vector_id:
                continue

            direction = vector.get("direction", {})
            if not isinstance(direction, dict) or not direction:
                continue

            dominant_key = max(direction, key=lambda k: abs(direction.get(k, 0.0)))
            motive_key = generate_motive_key(vector_id)
            label = f"vector_{dominant_key}"

            results.append((
                motive_key, label, min(1.0, magnitude), 0.0,
                f"Vector motive: {dominant_key} (magnitude={magnitude:.2f})",
                MotiveSourceType.GOAL_VECTOR,
            ))

    return results


def extract_from_goal_candidates(
    candidate_state: Optional[Any],
) -> list[tuple[str, str, float, float, str, MotiveSourceType]]:
    """
    Extract motive materials from goal candidates.

    Returns list of (motive_key, label, intensity, valence, source_description, source_type).
    Duck typing: .candidates[], .candidate_id, .category, .intensity
    intensity >= 0.1
    motive_key = generate_motive_key(candidate_id), label = f"candidate_{category_value}"
    """
    if candidate_state is None:
        return []

    results: list[tuple[str, str, float, float, str, MotiveSourceType]] = []

    # Handle object with candidates attribute
    if hasattr(candidate_state, "candidates"):
        candidates = getattr(candidate_state, "candidates", [])
        if not candidates:
            return []

        for candidate in candidates[:20]:
            intensity = getattr(candidate, "intensity", 0.0)
            if not isinstance(intensity, (int, float)) or intensity < 0.1:
                continue

            candidate_id = getattr(candidate, "candidate_id", "")
            if not candidate_id:
                continue

            category = getattr(candidate, "category", None)
            if category is None:
                continue

            category_value = getattr(category, "value", str(category))
            motive_key = generate_motive_key(candidate_id)
            label = f"candidate_{category_value}"

            results.append((
                motive_key, label, min(1.0, intensity), 0.0,
                f"Candidate motive: {category_value} (intensity={intensity:.2f})",
                MotiveSourceType.GOAL_CANDIDATE,
            ))

    # Handle dict input
    elif isinstance(candidate_state, dict):
        candidates = candidate_state.get("candidates", [])
        if not isinstance(candidates, list):
            return []

        for candidate in candidates[:20]:
            if not isinstance(candidate, dict):
                continue

            intensity = candidate.get("intensity", 0.0)
            if not isinstance(intensity, (int, float)) or intensity < 0.1:
                continue

            candidate_id = candidate.get("candidate_id", "")
            if not candidate_id:
                continue

            category = candidate.get("category", "")
            if not category:
                continue

            category_value = category if isinstance(category, str) else str(category)
            motive_key = generate_motive_key(candidate_id)
            label = f"candidate_{category_value}"

            results.append((
                motive_key, label, min(1.0, intensity), 0.0,
                f"Candidate motive: {category_value} (intensity={intensity:.2f})",
                MotiveSourceType.GOAL_CANDIDATE,
            ))

    return results


# =============================================================================
# Computation Functions (Pure)
# =============================================================================

def compute_motive_strength(impulses: tuple[MotiveImpulse, ...]) -> float:
    """
    Compute integrated motive strength from impulses.

    Uses weighted aggregation: weight = 1.0 / (1.0 + i * 0.2).
    """
    if not impulses:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for i, impulse in enumerate(impulses):
        weight = 1.0 / (1.0 + i * 0.2)
        weighted_sum += impulse.intensity * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return min(1.0, weighted_sum / total_weight)


def detect_motive_coexistence(
    entries: list[MotiveEntry],
) -> list[tuple[str, str]]:
    """
    Detect coexisting impulse pairs within entries sharing the same motive_key.

    Returns list of (label_a, label_b) pairs.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for entry in entries:
        labels = [imp.label for imp in entry.impulses]
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i >= j:
                    continue
                pair_key = tuple(sorted((a, b)))
                if pair_key not in seen:
                    pairs.append(pair_key)
                    seen.add(pair_key)

    return pairs


def compute_motive_overlay(
    entry: MotiveEntry,
) -> dict[str, float]:
    """
    Compute motive overlay for a motive entry.

    核心機能: 目的候補参照時に動機が「同伴」する。
    effective_intensity = impulse.intensity * impulse.freshness.
    同ラベルは max で統合。
    """
    overlay: dict[str, float] = {}

    for impulse in entry.impulses:
        effective_intensity = impulse.intensity * impulse.freshness
        label = impulse.label
        if label in overlay:
            overlay[label] = max(overlay[label], effective_intensity)
        else:
            overlay[label] = effective_intensity

    return overlay


# =============================================================================
# Intrinsic Motivation System
# =============================================================================

class IntrinsicMotivationSystem:
    """
    Intrinsic Motivation System (自発的内的動機)

    感情や傾向から湧き上がる内的な推進力を弱く形成する。

    CRITICAL CONSTRAINTS:
    - 価値・信念・目標の直接設計しない
    - 行動の最適化しない
    - 正誤・善悪・成功失敗の判断しない
    - 規範的方向付けしない
    - 自我・人格・信念・評価を直接形成しない
    - 判断・目的・価値・責任に一切接続しない
    """

    def __init__(self, config: Optional[IntrinsicMotivationConfig] = None):
        self._config = config or IntrinsicMotivationConfig()
        self._entries: list[MotiveEntry] = []
        self._motive_links: list[MotiveLink] = []
        self._total_entries_created: int = 0
        self._total_impulses_created: int = 0
        self._total_revisions: int = 0
        self._total_expirations: int = 0
        self._last_store: Optional[MotiveStore] = None

    def sense_motives(
        self,
        emotion: Optional[Any] = None,
        mood: Optional[Any] = None,
        tendencies: Optional[Any] = None,
        vectors: Optional[Any] = None,
        candidates: Optional[Any] = None,
    ) -> MotiveStore:
        """
        Sense motives from multiple input sources.

        extract(4系統) → motive_keyでグループ化 → 既存entryにマージ or 新規作成
        → MotiveLink生成 → decay → capacity制限 → snapshot。
        """
        current_time = str(time.time())

        # Extract from all sources
        emotion_extracts = extract_from_emotion_state(emotion, mood)
        tendency_extracts = extract_from_tendencies(tendencies)
        vector_extracts = extract_from_goal_vectors(vectors)
        candidate_extracts = extract_from_goal_candidates(candidates)

        # Combine all extracts (already tagged with source type)
        all_extracts: list[tuple[str, str, float, float, str, MotiveSourceType]] = []
        all_extracts.extend(emotion_extracts)
        all_extracts.extend(tendency_extracts)
        all_extracts.extend(vector_extracts)
        all_extracts.extend(candidate_extracts)

        # Group by motive_key
        groups: dict[str, list[tuple[str, float, float, str, MotiveSourceType]]] = {}
        for mk, label, intensity, val, desc, src in all_extracts:
            if mk not in groups:
                groups[mk] = []
            groups[mk].append((label, intensity, val, desc, src))

        # Process each group
        for motive_key, items in groups.items():
            existing = self._find_entry_for_key(motive_key)

            if existing is not None:
                # Merge impulses into existing entry
                idx = self._entries.index(existing)
                merged = self._merge_impulses(existing, items, current_time)
                self._entries[idx] = merged
            else:
                # Create new entry
                motive_id = _generate_id()
                impulses: list[MotiveImpulse] = []

                for label, intensity, val, desc, src in items:
                    if intensity < self._config.min_intensity_for_motive:
                        continue

                    # Determine affinity based on source type
                    affinity = _source_to_affinity(src)

                    impulse = MotiveImpulse(
                        impulse_id=_generate_id(),
                        label=label,
                        intensity=min(1.0, intensity),
                        valence=val,
                        freshness=1.0,
                        reference_count=0,
                        affinity=affinity,
                        timestamp=current_time,
                        source_description=desc[:80],
                    )
                    impulses.append(impulse)
                    self._total_impulses_created += 1

                if not impulses:
                    continue

                # Limit impulses per entry
                impulses = impulses[:self._config.max_impulses_per_entry]

                # Determine summary from first description
                summary = items[0][3][:120] if items else ""

                # Generate motive links
                new_links = self._generate_motive_links(motive_id, items)
                link_ids = tuple(link.link_id for link in new_links)
                self._motive_links.extend(new_links)

                entry = MotiveEntry(
                    motive_id=motive_id,
                    motive_key=motive_key,
                    motive_summary=summary,
                    impulses=tuple(impulses),
                    motive_links=link_ids,
                    freshness=1.0,
                    reference_count=0,
                    creation_timestamp=current_time,
                    last_reference_timestamp=current_time,
                    revision_count=0,
                    undetermined_aspects=("motive_approximate", "impulse_provisional"),
                )
                self._entries.append(entry)
                self._total_entries_created += 1

        # Apply decay
        self._apply_decay()

        # Enforce capacity
        self._enforce_capacity()

        return self._build_store(current_time)

    def decay_motives(self) -> MotiveStore:
        """
        Apply natural decay to all motive entries.

        entry: ref_modifier = max(0.5, 1.0 - ref_count * 0.05)
        impulse: impulse_ref_mod = max(0.5, 1.0 - impulse.ref_count * 0.08)
        全impulse消滅 AND freshness < min → 除去。
        """
        current_time = str(time.time())
        self._apply_decay()
        return self._build_store(current_time)

    def reference_motive(self, motive_key: str) -> None:
        """
        Mark a motive as referenced.

        ref_count+1, freshness+boost, 各impulse freshness+impulse_boost。
        """
        for i, entry in enumerate(self._entries):
            if entry.motive_key == motive_key:
                # Boost entry freshness
                referenced = entry.with_reference()
                boosted = referenced.with_freshness(
                    referenced.freshness + self._config.freshness_boost_on_reference,
                )

                # Boost each impulse freshness
                new_impulses: list[MotiveImpulse] = []
                for impulse in boosted.impulses:
                    boosted_impulse = impulse.with_reference().with_freshness(
                        impulse.freshness + self._config.impulse_boost_on_reference,
                    )
                    new_impulses.append(boosted_impulse)

                self._entries[i] = boosted.with_impulses(tuple(new_impulses))
                return

    def get_motive_overlay(self, motive_key: str) -> dict[str, float]:
        """
        Get motive overlay for a specific motive key.

        核心機能: 目的候補参照時に動機が「同伴」する。
        reference_motive も呼び出し（再参照で強化）。
        """
        # Reference the motive (strengthens on access)
        self.reference_motive(motive_key)

        for entry in self._entries:
            if entry.motive_key == motive_key:
                return compute_motive_overlay(entry)

        return {}

    def revise_motive(self, motive_key: str, new_summary: str) -> None:
        """Revise a motive's summary."""
        for i, entry in enumerate(self._entries):
            if entry.motive_key == motive_key:
                self._entries[i] = entry.revise_summary(new_summary)
                self._total_revisions += 1
                return

    def get_active_motives(self, max_count: int = 20) -> list[MotiveEntry]:
        """Get active motive entries sorted by freshness."""
        active = [
            e for e in self._entries
            if e.freshness > self._config.stale_threshold
        ]
        active.sort(key=lambda e: e.freshness, reverse=True)
        return active[:max_count]

    def get_store(self) -> MotiveStore:
        """Get current store snapshot."""
        return self._build_store(str(time.time()))

    def get_last_store(self) -> Optional[MotiveStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _apply_decay(self) -> None:
        """Apply decay to all entries, removing expired ones."""
        new_entries: list[MotiveEntry] = []

        for entry in self._entries:
            # Reference count modulates decay
            ref_modifier = max(0.5, 1.0 - entry.reference_count * 0.05)
            freshness_decay = self._config.base_decay_rate * ref_modifier
            new_freshness = entry.freshness - freshness_decay

            # Decay impulses
            new_impulses: list[MotiveImpulse] = []
            for impulse in entry.impulses:
                impulse_ref_mod = max(0.5, 1.0 - impulse.reference_count * 0.08)
                impulse_decay = self._config.impulse_decay_rate * impulse_ref_mod
                new_impulse_freshness = impulse.freshness - impulse_decay

                if new_impulse_freshness > 0.0:
                    new_impulses.append(impulse.with_freshness(new_impulse_freshness))

            # Check if expired: all impulses gone AND freshness below min
            if not new_impulses and new_freshness < self._config.min_freshness_for_retention:
                self._total_expirations += 1
                # Clean up motive links
                motive_link_ids = set(entry.motive_links)
                self._motive_links = [
                    ml for ml in self._motive_links
                    if ml.link_id not in motive_link_ids
                ]
                continue

            updated = entry.with_freshness(new_freshness).with_impulses(tuple(new_impulses))
            new_entries.append(updated)

        self._entries = new_entries

    def _enforce_capacity(self) -> None:
        """Remove weakest entries if over capacity."""
        while len(self._entries) > self._config.max_entries:
            weakest_idx = min(
                range(len(self._entries)),
                key=lambda i: (self._entries[i].freshness, len(self._entries[i].impulses)),
            )
            removed = self._entries.pop(weakest_idx)
            motive_link_ids = set(removed.motive_links)
            self._motive_links = [
                ml for ml in self._motive_links
                if ml.link_id not in motive_link_ids
            ]
            self._total_expirations += 1

    def _generate_motive_links(
        self,
        motive_id: str,
        items: list[tuple[str, float, float, str, MotiveSourceType]],
    ) -> list[MotiveLink]:
        """Generate motive links for a new entry."""
        links: list[MotiveLink] = []
        max_links = self._config.max_motive_links

        for label, intensity, val, desc, src in items[:max_links]:
            idx = len(links)
            contribution = max(0.1, 1.0 - idx * 0.15)

            link = MotiveLink(
                link_id=_generate_id(),
                motive_id=motive_id,
                source_type=src,
                source_description=desc[:80],
                contribution=contribution,
            )
            links.append(link)

        return links

    def _find_entry_for_key(self, motive_key: str) -> Optional[MotiveEntry]:
        """Find an existing entry for a motive key."""
        for entry in self._entries:
            if entry.motive_key == motive_key:
                return entry
        return None

    def _merge_impulses(
        self,
        existing: MotiveEntry,
        items: list[tuple[str, float, float, str, MotiveSourceType]],
        current_time: str,
    ) -> MotiveEntry:
        """Merge new impulses into an existing entry."""
        existing_labels = {imp.label for imp in existing.impulses}
        new_impulses = list(existing.impulses)

        for label, intensity, val, desc, src in items:
            if intensity < self._config.min_intensity_for_motive:
                continue

            if label in existing_labels:
                # Update existing impulse with max intensity
                for j, imp in enumerate(new_impulses):
                    if imp.label == label:
                        if intensity > imp.intensity:
                            new_impulses[j] = imp.with_intensity(intensity).with_freshness(1.0)
                        else:
                            new_impulses[j] = imp.with_freshness(1.0)
                        break
            else:
                if len(new_impulses) < self._config.max_impulses_per_entry:
                    affinity = _source_to_affinity(src)
                    impulse = MotiveImpulse(
                        impulse_id=_generate_id(),
                        label=label,
                        intensity=min(1.0, intensity),
                        valence=val,
                        freshness=1.0,
                        reference_count=0,
                        affinity=affinity,
                        timestamp=current_time,
                        source_description=desc[:80],
                    )
                    new_impulses.append(impulse)
                    existing_labels.add(label)
                    self._total_impulses_created += 1

        # Refresh entry freshness
        return existing.with_impulses(tuple(new_impulses)).with_freshness(1.0)

    def _build_store(self, current_time: str) -> MotiveStore:
        """Build a MotiveStore snapshot."""
        active = [
            e for e in self._entries
            if e.freshness > self._config.stale_threshold
        ]
        avg_freshness = (
            sum(e.freshness for e in self._entries) / len(self._entries)
            if self._entries else 0.0
        )
        avg_impulse_count = (
            sum(len(e.impulses) for e in self._entries) / len(self._entries)
            if self._entries else 0.0
        )

        description = _generate_store_description(
            len(self._entries),
            len(active),
            avg_freshness,
            avg_impulse_count,
            self._total_expirations,
        )

        store = MotiveStore(
            entries=tuple(self._entries),
            motive_links=tuple(self._motive_links),
            total_entries_created=self._total_entries_created,
            total_impulses_created=self._total_impulses_created,
            total_revisions=self._total_revisions,
            total_expirations=self._total_expirations,
            average_freshness=round(avg_freshness, 4),
            average_impulse_count=round(avg_impulse_count, 4),
            active_entry_count=len(active),
            timestamp=current_time,
            description=description,
        )
        self._last_store = store
        return store


def _source_to_affinity(src: MotiveSourceType) -> MotiveAffinity:
    """Map source type to default affinity."""
    if src == MotiveSourceType.EMOTION:
        return MotiveAffinity.EMOTIONAL_SURGE
    elif src == MotiveSourceType.TENDENCY:
        return MotiveAffinity.HABITUAL
    elif src == MotiveSourceType.GOAL_VECTOR:
        return MotiveAffinity.DIRECTIONAL
    elif src == MotiveSourceType.GOAL_CANDIDATE:
        return MotiveAffinity.ASPIRATIONAL
    else:
        return MotiveAffinity.COMPOSITE


def _generate_store_description(
    total: int,
    active: int,
    avg_freshness: float,
    avg_impulse_count: float,
    total_expirations: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No intrinsic motives formed yet."

    parts = [f"{active} active motives out of {total} total"]

    if avg_impulse_count >= 3.0:
        parts.append("rich motive impulses")
    elif avg_impulse_count >= 1.5:
        parts.append("moderate motive impulses")
    else:
        parts.append("sparse motive impulses")

    if total_expirations > 0:
        parts.append(f"{total_expirations} expired")

    parts.append(f"avg freshness: {avg_freshness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def sense_from_chain(
    system: IntrinsicMotivationSystem,
    emotion: Optional[Any] = None,
    mood: Optional[Any] = None,
    tendencies: Optional[Any] = None,
    vectors: Optional[Any] = None,
    candidates: Optional[Any] = None,
) -> MotiveStore:
    """
    Sense motives from the observation chain.

    統合ヘルパー。各入力は読み取り専用で参照される。
    """
    return system.sense_motives(
        emotion=emotion,
        mood=mood,
        tendencies=tendencies,
        vectors=vectors,
        candidates=candidates,
    )


def generate_motive_tags(
    store: Optional[Any],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from MotiveStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags: list[dict[str, Any]] = []

    if store is None:
        tags.append({
            "category": "INTRINSIC_MOTIVE_COUNT",
            "label": "no_motives",
            "description": "No intrinsic motives formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    has_entries = getattr(store, "has_entries", None)
    if callable(has_entries) and not has_entries():
        tags.append({
            "category": "INTRINSIC_MOTIVE_COUNT",
            "label": "no_motives",
            "description": "No intrinsic motives formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    # Motive count
    active_count = getattr(store, "active_entry_count", 0)
    tags.append({
        "category": "INTRINSIC_MOTIVE_COUNT",
        "label": f"motives_{active_count}",
        "description": f"Intrinsic motivation holds {active_count} active motives",
        "weight": 0.06 * scale,
    })

    # Average freshness
    avg_freshness = getattr(store, "average_freshness", 0.0)
    freshness_label = determine_freshness_level(avg_freshness).value
    tags.append({
        "category": "INTRINSIC_MOTIVE_FRESHNESS",
        "label": f"freshness_{freshness_label}",
        "description": f"Average motive freshness: {avg_freshness:.2f}",
        "weight": 0.05 * scale,
    })

    # Impulse richness
    avg_impulse_count = getattr(store, "average_impulse_count", 0.0)
    if avg_impulse_count >= 3.0:
        richness_label = "rich"
    elif avg_impulse_count >= 1.5:
        richness_label = "moderate"
    else:
        richness_label = "sparse"
    tags.append({
        "category": "INTRINSIC_MOTIVE_RICHNESS",
        "label": f"richness_{richness_label}",
        "description": f"Average impulse count per motive: {avg_impulse_count:.1f}",
        "weight": 0.07 * scale,
    })

    # Dominant motive across entries
    entries = getattr(store, "entries", ())
    if entries:
        label_counts: dict[str, int] = {}
        for entry in entries:
            for impulse in entry.impulses:
                label = impulse.label
                label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts:
            dominant = max(label_counts, key=label_counts.get)
            tags.append({
                "category": "INTRINSIC_MOTIVE_DOMINANT",
                "label": f"dominant_{dominant}",
                "description": f"Most frequent motive impulse: {dominant} ({label_counts[dominant]} occurrences)",
                "weight": 0.08 * scale,
            })

    # Integrated description
    description = getattr(store, "description", "")
    if description:
        tags.append({
            "category": "INTRINSIC_MOTIVE_INTEGRATED",
            "label": "motive_awareness",
            "description": description,
            "weight": 0.08 * scale,
        })

    return tags


def get_motive_summary(store: Optional[Any]) -> str:
    """Get human-readable summary. For introspection/logging only."""
    if store is None:
        return "=== Intrinsic Motivation State ===\nNo motives formed yet."

    has_entries = getattr(store, "has_entries", None)
    if callable(has_entries) and not has_entries():
        return "=== Intrinsic Motivation State ===\nNo motives formed yet."

    entries = getattr(store, "entries", ())
    active_count = getattr(store, "active_entry_count", 0)
    total_created = getattr(store, "total_entries_created", 0)
    total_impulses = getattr(store, "total_impulses_created", 0)
    total_revisions = getattr(store, "total_revisions", 0)
    total_expirations = getattr(store, "total_expirations", 0)
    avg_freshness = getattr(store, "average_freshness", 0.0)
    avg_impulse_count = getattr(store, "average_impulse_count", 0.0)
    description = getattr(store, "description", "")

    lines = [
        "=== Intrinsic Motivation State ===",
        f"Total motives: {len(entries)}",
        f"Active motives: {active_count}",
        f"Total created: {total_created}",
        f"Total impulses created: {total_impulses}",
        f"Total revisions: {total_revisions}",
        f"Total expirations: {total_expirations}",
        f"Average freshness: {avg_freshness:.2f}",
        f"Average impulse count: {avg_impulse_count:.1f}",
        "",
    ]

    # Show top entries
    sorted_entries = sorted(entries, key=lambda e: e.freshness, reverse=True)
    if sorted_entries:
        lines.append("Top motives:")
        for entry in sorted_entries[:5]:
            impulse_labels = [imp.label for imp in entry.impulses]
            lines.append(
                f"  [{entry.motive_key}] {entry.motive_summary[:60]}"
                f" impulses: {', '.join(impulse_labels)}"
                f" (freshness: {determine_freshness_level(entry.freshness).value})"
            )
        lines.append("")

    lines.append(f"Integrated: {description}")
    return "\n".join(lines)


def get_motive_for_introspection(
    store: Optional[Any],
) -> dict[str, Any]:
    """
    Get structured motive data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    if store is None:
        return {
            "has_motives": False,
            "total_motives": 0,
            "active_count": 0,
            "average_freshness": 0.0,
            "average_impulse_count": 0.0,
            "impulse_distribution": {},
            "dominant_impulse": "",
            "strongest_motive_summary": "",
            "timestamp": "",
        }

    entries = getattr(store, "entries", ())

    impulse_dist: dict[str, int] = {}
    for entry in entries:
        for impulse in entry.impulses:
            key = impulse.label
            impulse_dist[key] = impulse_dist.get(key, 0) + 1

    dominant_impulse = ""
    if impulse_dist:
        dominant_impulse = max(impulse_dist, key=impulse_dist.get)

    strongest_summary = ""
    if entries:
        strongest = max(entries, key=lambda e: e.freshness)
        strongest_summary = strongest.motive_summary[:120]

    return {
        "has_motives": len(entries) > 0,
        "total_motives": len(entries),
        "active_count": getattr(store, "active_entry_count", 0),
        "average_freshness": getattr(store, "average_freshness", 0.0),
        "average_impulse_count": getattr(store, "average_impulse_count", 0.0),
        "impulse_distribution": impulse_dist,
        "dominant_impulse": dominant_impulse,
        "strongest_motive_summary": strongest_summary,
        "timestamp": getattr(store, "timestamp", ""),
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: MotiveStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    動機は参照素材に留め、直接の意思決定経路を持たない。
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
            if attr in (
                "total_entries_created", "total_impulses_created",
                "total_revisions", "total_expirations",
                "average_freshness", "average_impulse_count",
                "active_entry_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: IntrinsicMotivationSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    動機から目標を生成しない。
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


def verify_read_only_principle(system: IntrinsicMotivationSystem) -> bool:
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


def verify_no_value_modification(system: IntrinsicMotivationSystem) -> bool:
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


def verify_no_motivation_prescription(system: IntrinsicMotivationSystem) -> bool:
    """
    Verify the system has no motivation-prescribing methods.

    動機を処方・強制・矯正・最適化・規範化・評価しない。
    固有の検証関数。
    """
    forbidden = [
        "prescribe_motive", "force_motive", "correct_motive",
        "optimize_motive", "normalize_motive", "rate_motive",
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

def create_config(**kwargs: Any) -> IntrinsicMotivationConfig:
    """Create a custom configuration."""
    return IntrinsicMotivationConfig(**kwargs)


def create_empty_store() -> MotiveStore:
    """Create an empty motive store."""
    return MotiveStore(
        entries=(),
        motive_links=(),
        total_entries_created=0,
        total_impulses_created=0,
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.0,
        average_impulse_count=0.0,
        active_entry_count=0,
        timestamp=str(time.time()),
        description="No intrinsic motives formed yet.",
    )


def create_system(
    config: Optional[IntrinsicMotivationConfig] = None,
) -> IntrinsicMotivationSystem:
    """Create a new IntrinsicMotivationSystem."""
    return IntrinsicMotivationSystem(config=config)


def save_motive_state(
    store: MotiveStore,
    filepath: str,
) -> None:
    """Save motive state to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_motive_state(filepath: str) -> MotiveStore:
    """Load motive state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MotiveStore.from_dict(data)
