"""
Emotional Memory Binding (感情記憶の紐づけ)

特定の記憶に感情が「染み付く」中長期の結びつきを実装するモジュール。

現状の課題:
stm_emotion_couplingは短期の感情連動のみ。出来事や記憶が感情の痕跡を
伴って中長期で保持される構造が不足している。感情痕跡は経験の質感を
保存するだけの層であり、評価・人格設計は行わない。

CRITICAL DESIGN PRINCIPLES:
- 入力は全て Optional[Any] + duck typing（循環import回避）
- 記憶の価値判断、正誤判定、目標更新は行わない
- 感情を理由に行動を最適化しない
- 感情の「正しい意味づけ」を行わない
- 自我・人格・信念・評価を直接形成しない
- 接続先: 記憶参照層への付随情報、内省記録層への観測素材
- 非接続: 判断選択層・目的生成・価値更新・責任評価・外部出力直接生成
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class BindingSourceType(Enum):
    """
    感情痕跡の入力源がどこ由来か。

    評価的ではなく、記述的な分類。
    """
    SHORT_TERM_MEMORY = "short_term_memory"  # 短期記憶由来
    EMOTION_STATE = "emotion_state"          # 感情状態由来
    LONG_TERM_RECALL = "long_term_recall"    # 長期記憶再参照由来
    EPISODIC = "episodic"                    # エピソード記憶由来
    MIXED = "mixed"                          # 複数ソース混合


class TraceAffinity(Enum):
    """
    紐づけの根拠・性質。

    同時発生/再活性化/蓄積/複合/未確定。
    """
    CONCURRENT = "concurrent"      # 同時発生
    REACTIVATED = "reactivated"    # 再活性化
    ACCUMULATED = "accumulated"    # 蓄積
    COMPOSITE = "composite"        # 複合
    UNDEFINED = "undefined"        # 未確定


class TraceStrength(Enum):
    """
    感情痕跡の強度。

    閾値: 0.7/0.4/0.2/0.05。
    """
    STRONG = "strong"          # 強い痕跡
    MODERATE = "moderate"      # 中程度
    WEAK = "weak"              # 弱い痕跡
    FAINT = "faint"            # かすかな痕跡
    UNDEFINED = "undefined"    # 未確定


class TraceFreshness(Enum):
    """
    感情痕跡の新鮮度。

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

def determine_freshness_level(freshness: float) -> TraceFreshness:
    """Determine TraceFreshness from a numeric freshness value (0.0-1.0)."""
    if freshness >= 0.8:
        return TraceFreshness.FRESH
    elif freshness >= 0.6:
        return TraceFreshness.RECENT
    elif freshness >= 0.4:
        return TraceFreshness.AGING
    elif freshness >= 0.15:
        return TraceFreshness.STALE
    else:
        return TraceFreshness.FADED


def determine_strength_level(strength: float) -> TraceStrength:
    """Determine TraceStrength from a numeric strength value (0.0-1.0)."""
    if strength >= 0.7:
        return TraceStrength.STRONG
    elif strength >= 0.4:
        return TraceStrength.MODERATE
    elif strength >= 0.2:
        return TraceStrength.WEAK
    elif strength >= 0.05:
        return TraceStrength.FAINT
    else:
        return TraceStrength.UNDEFINED


# =============================================================================
# ID / Key Generation
# =============================================================================

def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex[:12]


def generate_memory_key(text: str) -> str:
    """Generate a memory key from text via MD5 hash."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


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
class EmotionalTrace:
    """
    感情痕跡 — 記憶に付着する感情の1単位。

    NOT evaluative. NOT prescriptive.
    """
    trace_id: str
    emotion_label: str           # joy, anger, sorrow, fear, surprise, love, fun
    intensity: float             # 0.0〜1.0
    valence: float               # -1.0〜1.0
    freshness: float             # 0.0〜1.0
    reference_count: int
    affinity: TraceAffinity
    timestamp: str
    source_description: str

    def get_freshness_level(self) -> TraceFreshness:
        """Get abstract freshness level."""
        return determine_freshness_level(self.freshness)

    def with_freshness(self, new_freshness: float) -> EmotionalTrace:
        """Create a copy with updated freshness."""
        return EmotionalTrace(
            trace_id=self.trace_id,
            emotion_label=self.emotion_label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=max(0.0, min(1.0, new_freshness)),
            reference_count=self.reference_count,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def with_intensity(self, new_intensity: float) -> EmotionalTrace:
        """Create a copy with updated intensity."""
        return EmotionalTrace(
            trace_id=self.trace_id,
            emotion_label=self.emotion_label,
            intensity=max(0.0, min(1.0, new_intensity)),
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def with_reference(self) -> EmotionalTrace:
        """Create a copy with incremented reference count."""
        return EmotionalTrace(
            trace_id=self.trace_id,
            emotion_label=self.emotion_label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count + 1,
            affinity=self.affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )

    def reattach(self, new_affinity: TraceAffinity) -> EmotionalTrace:
        """Create a copy with updated affinity."""
        return EmotionalTrace(
            trace_id=self.trace_id,
            emotion_label=self.emotion_label,
            intensity=self.intensity,
            valence=self.valence,
            freshness=self.freshness,
            reference_count=self.reference_count,
            affinity=new_affinity,
            timestamp=self.timestamp,
            source_description=self.source_description,
        )


@dataclass(frozen=True)
class BindingLink:
    """
    紐づけの根拠リンク。

    NOT evaluative. NOT prescriptive.
    """
    link_id: str
    binding_id: str
    source_type: BindingSourceType
    source_description: str
    contribution: float  # 0.0〜1.0


@dataclass(frozen=True)
class MemoryBinding:
    """
    記憶と複数感情痕跡の紐づけ（コア構造）。

    NOT evaluative. NOT prescriptive.
    """
    binding_id: str
    memory_key: str                         # hashされた記憶識別子
    memory_summary: str                     # 記憶内容の要約
    traces: tuple[EmotionalTrace, ...]      # 複数感情痕跡を並立
    binding_links: tuple[str, ...]          # BindingLink ID群
    freshness: float                        # 0.0〜1.0
    reference_count: int
    creation_timestamp: str
    last_reference_timestamp: str
    revision_count: int
    undetermined_aspects: tuple[str, ...]

    def with_freshness(self, new_freshness: float) -> MemoryBinding:
        """Create a copy with updated freshness."""
        return replace(self, freshness=max(0.0, min(1.0, new_freshness)))

    def with_reference(self) -> MemoryBinding:
        """Create a copy with incremented reference count."""
        return replace(
            self,
            reference_count=self.reference_count + 1,
            last_reference_timestamp=str(time.time()),
        )

    def with_traces(self, new_traces: tuple[EmotionalTrace, ...]) -> MemoryBinding:
        """Create a copy with updated traces."""
        return replace(self, traces=new_traces)

    def revise_summary(self, new_summary: str) -> MemoryBinding:
        """Create a copy with revised summary."""
        return replace(
            self,
            memory_summary=new_summary,
            revision_count=self.revision_count + 1,
        )

    def with_added_trace(self, trace: EmotionalTrace) -> MemoryBinding:
        """Create a copy with an additional trace."""
        return replace(self, traces=self.traces + (trace,))


@dataclass(frozen=True)
class BindingStore:
    """
    不変スナップショット — 感情記憶紐づけの全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    bindings: tuple[MemoryBinding, ...]
    binding_links: tuple[BindingLink, ...]
    total_bindings_created: int
    total_traces_created: int
    total_revisions: int
    total_expirations: int
    average_freshness: float
    average_trace_count: float
    active_binding_count: int
    timestamp: str
    description: str

    def has_bindings(self) -> bool:
        return len(self.bindings) > 0

    def get_active_bindings(self, stale_threshold: float = 0.15) -> tuple[MemoryBinding, ...]:
        """Get bindings with freshness above the stale threshold."""
        return tuple(
            b for b in self.bindings
            if b.freshness > stale_threshold
        )

    def get_bindings_for_memory(self, memory_key: str) -> tuple[MemoryBinding, ...]:
        """Get bindings for a specific memory key."""
        return tuple(
            b for b in self.bindings
            if b.memory_key == memory_key
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bindings": [
                _binding_to_dict(b) for b in self.bindings
            ],
            "binding_links": [
                {
                    "link_id": bl.link_id,
                    "binding_id": bl.binding_id,
                    "source_type": bl.source_type.value,
                    "source_description": bl.source_description,
                    "contribution": bl.contribution,
                }
                for bl in self.binding_links
            ],
            "total_bindings_created": self.total_bindings_created,
            "total_traces_created": self.total_traces_created,
            "total_revisions": self.total_revisions,
            "total_expirations": self.total_expirations,
            "average_freshness": self.average_freshness,
            "average_trace_count": self.average_trace_count,
            "active_binding_count": self.active_binding_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BindingStore:
        """Create from dictionary."""
        bindings = tuple(
            _binding_from_dict(b) for b in data.get("bindings", [])
        )
        binding_links = tuple(
            BindingLink(
                link_id=bl["link_id"],
                binding_id=bl.get("binding_id", ""),
                source_type=BindingSourceType(bl.get("source_type", "mixed")),
                source_description=bl.get("source_description", ""),
                contribution=bl.get("contribution", 0.0),
            )
            for bl in data.get("binding_links", [])
        )
        return cls(
            bindings=bindings,
            binding_links=binding_links,
            total_bindings_created=data.get("total_bindings_created", 0),
            total_traces_created=data.get("total_traces_created", 0),
            total_revisions=data.get("total_revisions", 0),
            total_expirations=data.get("total_expirations", 0),
            average_freshness=data.get("average_freshness", 0.0),
            average_trace_count=data.get("average_trace_count", 0.0),
            active_binding_count=data.get("active_binding_count", 0),
            timestamp=data.get("timestamp", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class EmotionalMemoryBindingConfig:
    """
    Configuration for the emotional memory binding system.

    These parameters control binding lifecycle.
    They do NOT affect decisions.
    """
    max_bindings: int = 200                    # 記憶は長期保持
    max_traces_per_binding: int = 7            # 7感情次元に対応
    base_decay_rate: float = 0.02              # 中期〜長期なのでゆっくり
    trace_decay_rate: float = 0.015
    freshness_boost_on_reference: float = 0.08
    trace_boost_on_reference: float = 0.05
    stale_threshold: float = 0.15
    min_freshness_for_retention: float = 0.05
    max_binding_links: int = 10
    min_intensity_for_binding: float = 0.1


# =============================================================================
# Serialization Helpers
# =============================================================================

def _trace_to_dict(trace: EmotionalTrace) -> dict[str, Any]:
    """Convert an EmotionalTrace to a dictionary."""
    return {
        "trace_id": trace.trace_id,
        "emotion_label": trace.emotion_label,
        "intensity": trace.intensity,
        "valence": trace.valence,
        "freshness": trace.freshness,
        "reference_count": trace.reference_count,
        "affinity": trace.affinity.value,
        "timestamp": trace.timestamp,
        "source_description": trace.source_description,
    }


def _trace_from_dict(data: dict[str, Any]) -> EmotionalTrace:
    """Create an EmotionalTrace from a dictionary."""
    return EmotionalTrace(
        trace_id=data["trace_id"],
        emotion_label=data.get("emotion_label", ""),
        intensity=data.get("intensity", 0.0),
        valence=data.get("valence", 0.0),
        freshness=data.get("freshness", 0.0),
        reference_count=data.get("reference_count", 0),
        affinity=TraceAffinity(data.get("affinity", "undefined")),
        timestamp=data.get("timestamp", ""),
        source_description=data.get("source_description", ""),
    )


def _binding_to_dict(binding: MemoryBinding) -> dict[str, Any]:
    """Convert a MemoryBinding to a dictionary."""
    return {
        "binding_id": binding.binding_id,
        "memory_key": binding.memory_key,
        "memory_summary": binding.memory_summary,
        "traces": [_trace_to_dict(t) for t in binding.traces],
        "binding_links": list(binding.binding_links),
        "freshness": binding.freshness,
        "reference_count": binding.reference_count,
        "creation_timestamp": binding.creation_timestamp,
        "last_reference_timestamp": binding.last_reference_timestamp,
        "revision_count": binding.revision_count,
        "undetermined_aspects": list(binding.undetermined_aspects),
    }


def _binding_from_dict(data: dict[str, Any]) -> MemoryBinding:
    """Create a MemoryBinding from a dictionary."""
    return MemoryBinding(
        binding_id=data["binding_id"],
        memory_key=data.get("memory_key", ""),
        memory_summary=data.get("memory_summary", ""),
        traces=tuple(
            _trace_from_dict(t) for t in data.get("traces", [])
        ),
        binding_links=tuple(data.get("binding_links", ())),
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

def extract_from_stm(
    stm: Optional[Any],
) -> list[tuple[str, str, float, float, str]]:
    """
    Extract emotional binding materials from short-term memory.

    Returns list of (memory_key, emotion_label, intensity, valence, source_description).
    Reads STM via duck typing: entries[], source_text, emotion_label, raw_intensity, valence.
    """
    if stm is None:
        return []

    results: list[tuple[str, str, float, float, str]] = []

    # Handle object with entries attribute
    if hasattr(stm, "entries"):
        entries = getattr(stm, "entries", [])
        if not entries:
            return []

        for entry in entries[:10]:
            source_text = getattr(entry, "source_text", "")
            emotion_label = getattr(entry, "emotion_label", "")
            raw_intensity = getattr(entry, "raw_intensity", 0.0)
            valence = getattr(entry, "valence", 0.0)

            if not source_text or not emotion_label:
                continue

            # Map emotion label
            mapped = _EMOTION_LABEL_MAP.get(emotion_label, "")
            if not mapped or mapped == "":
                continue

            # Skip low intensity or neutral
            if not isinstance(raw_intensity, (int, float)):
                continue
            if raw_intensity < 0.1:
                continue
            if emotion_label == "neutral":
                continue

            memory_key = generate_memory_key(source_text)
            intensity = min(1.0, raw_intensity)
            val = valence if isinstance(valence, (int, float)) else 0.0

            results.append((memory_key, mapped, intensity, val, source_text[:80]))

    # Handle dict input
    elif isinstance(stm, dict):
        entries = stm.get("entries", [])
        if not entries or not isinstance(entries, list):
            return []

        for entry in entries[:10]:
            if not isinstance(entry, dict):
                continue
            source_text = entry.get("source_text", "")
            emotion_label = entry.get("emotion_label", "")
            raw_intensity = entry.get("raw_intensity", 0.0)
            valence = entry.get("valence", 0.0)

            if not source_text or not emotion_label:
                continue

            mapped = _EMOTION_LABEL_MAP.get(emotion_label, "")
            if not mapped or mapped == "":
                continue

            if not isinstance(raw_intensity, (int, float)):
                continue
            if raw_intensity < 0.1:
                continue
            if emotion_label == "neutral":
                continue

            memory_key = generate_memory_key(source_text)
            intensity = min(1.0, raw_intensity)
            val = valence if isinstance(valence, (int, float)) else 0.0

            results.append((memory_key, mapped, intensity, val, source_text[:80]))

    return results


def extract_from_emotion_state(
    emotion: Optional[Any],
    mood: Optional[Any],
) -> list[tuple[str, str, float, float, str]]:
    """
    Extract emotional binding materials from current emotion state.

    Returns list of (memory_key, emotion_label, intensity, valence, source_description).
    EmotionVector duck typing: joy, anger, sorrow, fear, surprise, love, fun.
    Mood duck typing: valence, arousal.
    """
    if emotion is None:
        return []

    results: list[tuple[str, str, float, float, str]] = []
    memory_key = "__current_emotion_state__"

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
                # Blend base valence with mood valence
                val = base_valence * 0.7 + mood_valence * 0.3 if isinstance(mood_valence, (int, float)) else base_valence
                results.append((memory_key, field_name, min(1.0, value), val, f"Current emotion: {field_name}={value:.2f}"))

    # Handle dict input
    elif isinstance(emotion, dict):
        for field_name in emotion_fields:
            value = emotion.get(field_name, 0.0)
            if isinstance(value, (int, float)) and value >= 0.15:
                base_valence = emotion_valence_map.get(field_name, 0.0)
                val = base_valence * 0.7 + mood_valence * 0.3 if isinstance(mood_valence, (int, float)) else base_valence
                results.append((memory_key, field_name, min(1.0, value), val, f"Current emotion: {field_name}={value:.2f}"))

    return results


def extract_from_recalled_memories(
    memories: Optional[Any],
) -> list[tuple[str, str, float, float, str]]:
    """
    Extract emotional binding materials from recalled memories.

    Returns list of (memory_key, emotion_label, intensity, valence, source_description).
    recall_with_mood result: list[dict] with "summary", "keywords".
    """
    if memories is None:
        return []

    results: list[tuple[str, str, float, float, str]] = []

    if not isinstance(memories, list):
        return []

    for mem in memories[:10]:
        if not isinstance(mem, dict):
            continue

        summary = mem.get("summary", "")
        if not summary:
            continue

        memory_key = generate_memory_key(summary)
        # Default weak binding for recalled memories
        intensity = 0.3
        valence = 0.0

        # Try to detect emotion from keywords
        keywords = mem.get("keywords", [])
        emotion_label = "neutral"  # default: no positive/negative bias
        if isinstance(keywords, list):
            for kw in keywords:
                if isinstance(kw, str):
                    mapped = _EMOTION_LABEL_MAP.get(kw.lower(), "")
                    if mapped:
                        emotion_label = mapped
                        break

        results.append((memory_key, emotion_label, intensity, valence, summary[:80]))

    return results


def extract_from_episodes(
    episodes: Optional[Any],
) -> list[tuple[str, str, float, float, str]]:
    """
    Extract emotional binding materials from episodic memory.

    Returns list of (memory_key, emotion_label, intensity, valence, source_description).
    EpisodeStore/EpisodeEntry duck typing: episodes[], episode_id, summary,
    emotional_companion, vividness.
    EmotionalCompanion: primary_emotion, intensity_level, valence, coexisting_emotions.
    """
    if episodes is None:
        return []

    results: list[tuple[str, str, float, float, str]] = []

    # Handle object with episodes attribute (EpisodeStore)
    if hasattr(episodes, "episodes"):
        episode_list = getattr(episodes, "episodes", [])
        if not episode_list:
            return []

        for entry in episode_list[:10]:
            vividness = getattr(entry, "vividness", 0.0)
            if not isinstance(vividness, (int, float)) or vividness < 0.2:
                continue

            summary = getattr(entry, "summary", "")
            episode_id = getattr(entry, "episode_id", "")
            if not summary and not episode_id:
                continue

            memory_key = generate_memory_key(summary or episode_id)

            companion = getattr(entry, "emotional_companion", None)
            if companion is None:
                continue

            # Primary emotion
            primary_emotion = getattr(companion, "primary_emotion", "")
            intensity_level = getattr(companion, "intensity_level", 0.0)
            valence = getattr(companion, "valence", 0.0)

            mapped = _EMOTION_LABEL_MAP.get(primary_emotion, primary_emotion)
            if mapped and isinstance(intensity_level, (int, float)):
                results.append((memory_key, mapped, min(1.0, intensity_level), valence if isinstance(valence, (int, float)) else 0.0, summary[:80]))

            # Coexisting emotions (weaker)
            coexisting = getattr(companion, "coexisting_emotions", ())
            if isinstance(coexisting, (list, tuple)):
                for coex in coexisting[:3]:
                    coex_mapped = _EMOTION_LABEL_MAP.get(str(coex), str(coex))
                    if coex_mapped and isinstance(intensity_level, (int, float)):
                        coex_intensity = intensity_level * 0.5
                        results.append((memory_key, coex_mapped, min(1.0, coex_intensity), valence if isinstance(valence, (int, float)) else 0.0, f"Coexisting: {coex}"))

    # Handle dict input
    elif isinstance(episodes, dict):
        episode_list = episodes.get("episodes", [])
        if not isinstance(episode_list, list):
            return []

        for entry in episode_list[:10]:
            if not isinstance(entry, dict):
                continue

            vividness = entry.get("vividness", 0.0)
            if not isinstance(vividness, (int, float)) or vividness < 0.2:
                continue

            summary = entry.get("summary", "")
            episode_id = entry.get("episode_id", "")
            if not summary and not episode_id:
                continue

            memory_key = generate_memory_key(summary or episode_id)

            companion = entry.get("emotional_companion", None)
            if companion is None or not isinstance(companion, dict):
                continue

            primary_emotion = companion.get("primary_emotion", "")
            intensity_level = companion.get("intensity_level", 0.0)
            valence = companion.get("valence", 0.0)

            mapped = _EMOTION_LABEL_MAP.get(primary_emotion, primary_emotion)
            if mapped and isinstance(intensity_level, (int, float)):
                results.append((memory_key, mapped, min(1.0, intensity_level), valence if isinstance(valence, (int, float)) else 0.0, summary[:80]))

            coexisting = companion.get("coexisting_emotions", [])
            if isinstance(coexisting, (list, tuple)):
                for coex in coexisting[:3]:
                    coex_mapped = _EMOTION_LABEL_MAP.get(str(coex), str(coex))
                    if coex_mapped and isinstance(intensity_level, (int, float)):
                        coex_intensity = intensity_level * 0.5
                        results.append((memory_key, coex_mapped, min(1.0, coex_intensity), valence if isinstance(valence, (int, float)) else 0.0, f"Coexisting: {coex}"))

    return results


# =============================================================================
# Computation Functions (Pure)
# =============================================================================

def compute_binding_strength(traces: tuple[EmotionalTrace, ...]) -> float:
    """
    Compute integrated binding strength from traces.

    Uses weighted aggregation: weight = 1.0 / (1.0 + i * 0.2).
    """
    if not traces:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for i, trace in enumerate(traces):
        weight = 1.0 / (1.0 + i * 0.2)
        weighted_sum += trace.intensity * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return min(1.0, weighted_sum / total_weight)


def detect_trace_coexistence(
    bindings: list[MemoryBinding],
) -> list[tuple[str, str]]:
    """
    Detect coexisting trace pairs within bindings sharing the same memory_key.

    Returns list of (emotion_label_a, emotion_label_b) pairs.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for binding in bindings:
        labels = [t.emotion_label for t in binding.traces]
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i >= j:
                    continue
                pair_key = tuple(sorted((a, b)))
                if pair_key not in seen:
                    pairs.append(pair_key)
                    seen.add(pair_key)

    return pairs


def compute_emotional_accompaniment(
    binding: MemoryBinding,
) -> dict[str, float]:
    """
    Compute emotional accompaniment for a memory binding.

    核心機能: 記憶再参照時に感情痕跡が「同伴」する。
    effective_intensity = trace.intensity * trace.freshness.
    同ラベルは max で統合。
    """
    accompaniment: dict[str, float] = {}

    for trace in binding.traces:
        effective_intensity = trace.intensity * trace.freshness
        label = trace.emotion_label
        if label in accompaniment:
            accompaniment[label] = max(accompaniment[label], effective_intensity)
        else:
            accompaniment[label] = effective_intensity

    return accompaniment


# =============================================================================
# Emotional Memory Binding System
# =============================================================================

class EmotionalMemoryBindingSystem:
    """
    Emotional Memory Binding System (感情記憶の紐づけ)

    特定の記憶に感情が「染み付く」中長期の結びつきを管理する。

    CRITICAL CONSTRAINTS:
    - 記憶の価値判断、正誤判定、目標更新は行わない
    - 感情を理由に行動を最適化しない
    - 感情の「正しい意味づけ」を行わない
    - 自我・人格・信念・評価を直接形成しない
    - 判断・目的・価値・責任に一切接続しない
    """

    def __init__(self, config: Optional[EmotionalMemoryBindingConfig] = None):
        self._config = config or EmotionalMemoryBindingConfig()
        self._bindings: list[MemoryBinding] = []
        self._binding_links: list[BindingLink] = []
        self._total_bindings_created: int = 0
        self._total_traces_created: int = 0
        self._total_revisions: int = 0
        self._total_expirations: int = 0
        self._last_store: Optional[BindingStore] = None

    def bind_emotions(
        self,
        stm: Optional[Any] = None,
        emotion_state: Optional[Any] = None,
        mood: Optional[Any] = None,
        recalled_memories: Optional[Any] = None,
        episodes: Optional[Any] = None,
    ) -> BindingStore:
        """
        Bind emotions to memories from multiple input sources.

        extract(4系統) → memory_keyでグループ化 → 既存bindingにマージ or 新規作成
        → BindingLink生成 → decay → capacity制限 → snapshot。
        """
        current_time = str(time.time())

        # Extract from all sources
        stm_extracts = extract_from_stm(stm)
        emotion_extracts = extract_from_emotion_state(emotion_state, mood)
        recall_extracts = extract_from_recalled_memories(recalled_memories)
        episode_extracts = extract_from_episodes(episodes)

        # Tag each extract with its source type
        all_extracts: list[tuple[str, str, float, float, str, BindingSourceType]] = []
        for mk, el, intensity, val, desc in stm_extracts:
            all_extracts.append((mk, el, intensity, val, desc, BindingSourceType.SHORT_TERM_MEMORY))
        for mk, el, intensity, val, desc in emotion_extracts:
            all_extracts.append((mk, el, intensity, val, desc, BindingSourceType.EMOTION_STATE))
        for mk, el, intensity, val, desc in recall_extracts:
            all_extracts.append((mk, el, intensity, val, desc, BindingSourceType.LONG_TERM_RECALL))
        for mk, el, intensity, val, desc in episode_extracts:
            all_extracts.append((mk, el, intensity, val, desc, BindingSourceType.EPISODIC))

        # Group by memory_key
        groups: dict[str, list[tuple[str, float, float, str, BindingSourceType]]] = {}
        for mk, el, intensity, val, desc, src in all_extracts:
            if mk not in groups:
                groups[mk] = []
            groups[mk].append((el, intensity, val, desc, src))

        # Process each group
        for memory_key, items in groups.items():
            existing_idx = self._find_binding_index_for_memory(memory_key)

            if existing_idx is not None:
                # Merge traces into existing binding
                existing = self._bindings[existing_idx]
                merged = self._merge_traces(existing, items, current_time)
                self._bindings[existing_idx] = merged
            else:
                # Create new binding
                binding_id = _generate_id()
                traces: list[EmotionalTrace] = []

                for el, intensity, val, desc, src in items:
                    if intensity < self._config.min_intensity_for_binding:
                        continue
                    trace = EmotionalTrace(
                        trace_id=_generate_id(),
                        emotion_label=el,
                        intensity=min(1.0, intensity),
                        valence=val,
                        freshness=1.0,
                        reference_count=0,
                        affinity=TraceAffinity.CONCURRENT,
                        timestamp=current_time,
                        source_description=desc[:80],
                    )
                    traces.append(trace)
                    self._total_traces_created += 1

                if not traces:
                    continue

                # Limit traces per binding
                traces = traces[:self._config.max_traces_per_binding]

                # Determine summary from first description
                summary = items[0][3][:120] if items else ""

                # Generate binding links
                new_links = self._generate_binding_links(binding_id, items)
                link_ids = tuple(link.link_id for link in new_links)
                self._binding_links.extend(new_links)

                binding = MemoryBinding(
                    binding_id=binding_id,
                    memory_key=memory_key,
                    memory_summary=summary,
                    traces=tuple(traces),
                    binding_links=link_ids,
                    freshness=1.0,
                    reference_count=0,
                    creation_timestamp=current_time,
                    last_reference_timestamp=current_time,
                    revision_count=0,
                    undetermined_aspects=("emotion_trace_approximate", "binding_provisional"),
                )
                self._bindings.append(binding)
                self._total_bindings_created += 1

        # Apply decay
        self._apply_decay()

        # Enforce capacity
        self._enforce_capacity()

        return self._build_store(current_time)

    def decay_bindings(self) -> BindingStore:
        """
        Apply natural decay to all bindings.

        binding: ref_modifier = max(0.5, 1.0 - ref_count * 0.05)
        trace: trace_ref_mod = max(0.5, 1.0 - trace.ref_count * 0.08)
        全trace消滅 AND freshness < min → 除去。
        """
        current_time = str(time.time())
        self._apply_decay()
        return self._build_store(current_time)

    def reference_binding(self, memory_key: str) -> None:
        """
        Mark a binding as referenced.

        ref_count+1, freshness+boost, 各trace freshness+trace_boost。
        """
        for i, binding in enumerate(self._bindings):
            if binding.memory_key == memory_key:
                # Boost binding freshness
                referenced = binding.with_reference()
                boosted = referenced.with_freshness(
                    referenced.freshness + self._config.freshness_boost_on_reference,
                )

                # Boost each trace freshness
                new_traces: list[EmotionalTrace] = []
                for trace in boosted.traces:
                    boosted_trace = trace.with_reference().with_freshness(
                        trace.freshness + self._config.trace_boost_on_reference,
                    )
                    new_traces.append(boosted_trace)

                self._bindings[i] = boosted.with_traces(tuple(new_traces))
                return

    def get_emotional_accompaniment(self, memory_key: str) -> dict[str, float]:
        """
        Get emotional accompaniment for a memory.

        核心機能: 記憶再参照時に感情痕跡が「同伴」する。
        reference_binding も呼び出し（再参照で強化）。
        """
        # Reference the binding (strengthens on access)
        self.reference_binding(memory_key)

        for binding in self._bindings:
            if binding.memory_key == memory_key:
                return compute_emotional_accompaniment(binding)

        return {}

    def revise_binding(self, memory_key: str, new_summary: str) -> None:
        """Revise a binding's summary."""
        for i, binding in enumerate(self._bindings):
            if binding.memory_key == memory_key:
                self._bindings[i] = binding.revise_summary(new_summary)
                self._total_revisions += 1
                return

    def get_active_bindings(self, max_count: int = 20) -> list[MemoryBinding]:
        """Get active bindings sorted by freshness."""
        active = [
            b for b in self._bindings
            if b.freshness > self._config.stale_threshold
        ]
        active.sort(key=lambda b: b.freshness, reverse=True)
        return active[:max_count]

    def get_store(self) -> BindingStore:
        """Get current store snapshot."""
        return self._build_store(str(time.time()))

    def get_last_store(self) -> Optional[BindingStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _apply_decay(self) -> None:
        """Apply decay to all bindings, removing expired ones."""
        new_bindings: list[MemoryBinding] = []

        for binding in self._bindings:
            # Reference count modulates decay
            ref_modifier = max(0.5, 1.0 - binding.reference_count * 0.05)
            freshness_decay = self._config.base_decay_rate * ref_modifier
            new_freshness = binding.freshness - freshness_decay

            # Decay traces
            new_traces: list[EmotionalTrace] = []
            for trace in binding.traces:
                trace_ref_mod = max(0.5, 1.0 - trace.reference_count * 0.08)
                trace_decay = self._config.trace_decay_rate * trace_ref_mod
                new_trace_freshness = trace.freshness - trace_decay

                if new_trace_freshness > 0.0:
                    new_traces.append(trace.with_freshness(new_trace_freshness))

            # Check if expired: all traces gone AND freshness below min
            if not new_traces and new_freshness < self._config.min_freshness_for_retention:
                self._total_expirations += 1
                # Clean up binding links
                binding_link_ids = set(binding.binding_links)
                self._binding_links = [
                    bl for bl in self._binding_links
                    if bl.link_id not in binding_link_ids
                ]
                continue

            updated = binding.with_freshness(new_freshness).with_traces(tuple(new_traces))
            new_bindings.append(updated)

        self._bindings = new_bindings

    def _enforce_capacity(self) -> None:
        """Remove weakest bindings if over capacity."""
        excess = len(self._bindings) - self._config.max_bindings
        if excess <= 0:
            return
        indexed = sorted(
            range(len(self._bindings)),
            key=lambda i: (self._bindings[i].freshness, len(self._bindings[i].traces)),
        )
        remove_indices = set(indexed[:excess])
        removed_link_ids: set[str] = set()
        for idx in remove_indices:
            removed_link_ids.update(self._bindings[idx].binding_links)
        self._bindings = [
            b for i, b in enumerate(self._bindings)
            if i not in remove_indices
        ]
        if removed_link_ids:
            self._binding_links = [
                bl for bl in self._binding_links
                if bl.link_id not in removed_link_ids
            ]
        self._total_expirations += excess

    def _generate_binding_links(
        self,
        binding_id: str,
        items: list[tuple[str, float, float, str, BindingSourceType]],
    ) -> list[BindingLink]:
        """Generate binding links for a new binding."""
        links: list[BindingLink] = []
        max_links = self._config.max_binding_links

        for el, intensity, val, desc, src in items[:max_links]:
            idx = len(links)
            contribution = max(0.1, 1.0 - idx * 0.15)

            link = BindingLink(
                link_id=_generate_id(),
                binding_id=binding_id,
                source_type=src,
                source_description=desc[:80],
                contribution=contribution,
            )
            links.append(link)

        return links

    def _find_binding_for_memory(self, memory_key: str) -> Optional[MemoryBinding]:
        """Find an existing binding for a memory key."""
        for binding in self._bindings:
            if binding.memory_key == memory_key:
                return binding
        return None

    def _find_binding_index_for_memory(self, memory_key: str) -> Optional[int]:
        """Find the index of an existing binding for a memory key."""
        for i, binding in enumerate(self._bindings):
            if binding.memory_key == memory_key:
                return i
        return None

    def _merge_traces(
        self,
        existing: MemoryBinding,
        items: list[tuple[str, float, float, str, BindingSourceType]],
        current_time: str,
    ) -> MemoryBinding:
        """Merge new traces into an existing binding."""
        existing_labels = {t.emotion_label for t in existing.traces}
        new_traces = list(existing.traces)

        for el, intensity, val, desc, src in items:
            if intensity < self._config.min_intensity_for_binding:
                continue

            if el in existing_labels:
                # Update existing trace with max intensity
                for j, t in enumerate(new_traces):
                    if t.emotion_label == el:
                        if intensity > t.intensity:
                            new_traces[j] = t.with_intensity(intensity).with_freshness(1.0)
                        else:
                            new_traces[j] = t.with_freshness(1.0)
                        break
            else:
                if len(new_traces) < self._config.max_traces_per_binding:
                    trace = EmotionalTrace(
                        trace_id=_generate_id(),
                        emotion_label=el,
                        intensity=min(1.0, intensity),
                        valence=val,
                        freshness=1.0,
                        reference_count=0,
                        affinity=TraceAffinity.ACCUMULATED,
                        timestamp=current_time,
                        source_description=desc[:80],
                    )
                    new_traces.append(trace)
                    existing_labels.add(el)
                    self._total_traces_created += 1

        # Refresh binding freshness
        return existing.with_traces(tuple(new_traces)).with_freshness(1.0)

    def _build_store(self, current_time: str) -> BindingStore:
        """Build a BindingStore snapshot."""
        active = [
            b for b in self._bindings
            if b.freshness > self._config.stale_threshold
        ]
        avg_freshness = (
            sum(b.freshness for b in self._bindings) / len(self._bindings)
            if self._bindings else 0.0
        )
        avg_trace_count = (
            sum(len(b.traces) for b in self._bindings) / len(self._bindings)
            if self._bindings else 0.0
        )

        description = _generate_store_description(
            len(self._bindings),
            len(active),
            avg_freshness,
            avg_trace_count,
            self._total_expirations,
        )

        store = BindingStore(
            bindings=tuple(self._bindings),
            binding_links=tuple(self._binding_links),
            total_bindings_created=self._total_bindings_created,
            total_traces_created=self._total_traces_created,
            total_revisions=self._total_revisions,
            total_expirations=self._total_expirations,
            average_freshness=round(avg_freshness, 4),
            average_trace_count=round(avg_trace_count, 4),
            active_binding_count=len(active),
            timestamp=current_time,
            description=description,
        )
        self._last_store = store
        return store


def _generate_store_description(
    total: int,
    active: int,
    avg_freshness: float,
    avg_trace_count: float,
    total_expirations: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No emotional memory bindings formed yet."

    parts = [f"{active} active bindings out of {total} total"]

    if avg_trace_count >= 3.0:
        parts.append("rich emotional traces")
    elif avg_trace_count >= 1.5:
        parts.append("moderate emotional traces")
    else:
        parts.append("sparse emotional traces")

    if total_expirations > 0:
        parts.append(f"{total_expirations} expired")

    parts.append(f"avg freshness: {avg_freshness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def bind_from_chain(
    system: EmotionalMemoryBindingSystem,
    stm: Optional[Any] = None,
    emotion: Optional[Any] = None,
    mood: Optional[Any] = None,
    memories: Optional[Any] = None,
    episodes: Optional[Any] = None,
) -> BindingStore:
    """
    Bind emotions from the observation chain.

    統合ヘルパー。各入力は読み取り専用で参照される。
    """
    return system.bind_emotions(
        stm=stm,
        emotion_state=emotion,
        mood=mood,
        recalled_memories=memories,
        episodes=episodes,
    )


def generate_binding_tags(
    store: Optional[Any],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from BindingStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags: list[dict[str, Any]] = []

    if store is None:
        tags.append({
            "category": "EMOTIONAL_BINDING_COUNT",
            "label": "no_bindings",
            "description": "No emotional memory bindings formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    has_bindings = getattr(store, "has_bindings", None)
    if callable(has_bindings) and not has_bindings():
        tags.append({
            "category": "EMOTIONAL_BINDING_COUNT",
            "label": "no_bindings",
            "description": "No emotional memory bindings formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    # Binding count
    active_count = getattr(store, "active_binding_count", 0)
    tags.append({
        "category": "EMOTIONAL_BINDING_COUNT",
        "label": f"bindings_{active_count}",
        "description": f"Emotional memory binding holds {active_count} active bindings",
        "weight": 0.06 * scale,
    })

    # Average freshness
    avg_freshness = getattr(store, "average_freshness", 0.0)
    freshness_label = determine_freshness_level(avg_freshness).value
    tags.append({
        "category": "EMOTIONAL_BINDING_FRESHNESS",
        "label": f"freshness_{freshness_label}",
        "description": f"Average binding freshness: {avg_freshness:.2f}",
        "weight": 0.05 * scale,
    })

    # Trace richness
    avg_trace_count = getattr(store, "average_trace_count", 0.0)
    if avg_trace_count >= 3.0:
        richness_label = "rich"
    elif avg_trace_count >= 1.5:
        richness_label = "moderate"
    else:
        richness_label = "sparse"
    tags.append({
        "category": "EMOTIONAL_BINDING_RICHNESS",
        "label": f"richness_{richness_label}",
        "description": f"Average trace count per binding: {avg_trace_count:.1f}",
        "weight": 0.07 * scale,
    })

    # Dominant emotion across bindings
    bindings = getattr(store, "bindings", ())
    if bindings:
        emotion_counts: dict[str, int] = {}
        for binding in bindings:
            for trace in binding.traces:
                label = trace.emotion_label
                emotion_counts[label] = emotion_counts.get(label, 0) + 1
        if emotion_counts:
            dominant = max(emotion_counts, key=emotion_counts.get)
            tags.append({
                "category": "EMOTIONAL_BINDING_DOMINANT",
                "label": f"dominant_{dominant}",
                "description": f"Most frequent emotional trace: {dominant} ({emotion_counts[dominant]} occurrences)",
                "weight": 0.08 * scale,
            })

    # Integrated description
    description = getattr(store, "description", "")
    if description:
        tags.append({
            "category": "EMOTIONAL_BINDING_INTEGRATED",
            "label": "binding_awareness",
            "description": description,
            "weight": 0.08 * scale,
        })

    return tags


def get_binding_summary(store: Optional[Any]) -> str:
    """Get human-readable summary. For introspection/logging only."""
    if store is None:
        return "=== Emotional Memory Binding State ===\nNo bindings formed yet."

    has_bindings = getattr(store, "has_bindings", None)
    if callable(has_bindings) and not has_bindings():
        return "=== Emotional Memory Binding State ===\nNo bindings formed yet."

    bindings = getattr(store, "bindings", ())
    active_count = getattr(store, "active_binding_count", 0)
    total_created = getattr(store, "total_bindings_created", 0)
    total_traces = getattr(store, "total_traces_created", 0)
    total_revisions = getattr(store, "total_revisions", 0)
    total_expirations = getattr(store, "total_expirations", 0)
    avg_freshness = getattr(store, "average_freshness", 0.0)
    avg_trace_count = getattr(store, "average_trace_count", 0.0)
    description = getattr(store, "description", "")

    lines = [
        "=== Emotional Memory Binding State ===",
        f"Total bindings: {len(bindings)}",
        f"Active bindings: {active_count}",
        f"Total created: {total_created}",
        f"Total traces created: {total_traces}",
        f"Total revisions: {total_revisions}",
        f"Total expirations: {total_expirations}",
        f"Average freshness: {avg_freshness:.2f}",
        f"Average trace count: {avg_trace_count:.1f}",
        "",
    ]

    # Show top bindings
    sorted_bindings = sorted(bindings, key=lambda b: b.freshness, reverse=True)
    if sorted_bindings:
        lines.append("Top bindings:")
        for binding in sorted_bindings[:5]:
            trace_labels = [t.emotion_label for t in binding.traces]
            lines.append(
                f"  [{binding.memory_key}] {binding.memory_summary[:60]}"
                f" traces: {', '.join(trace_labels)}"
                f" (freshness: {determine_freshness_level(binding.freshness).value})"
            )
        lines.append("")

    lines.append(f"Integrated: {description}")
    return "\n".join(lines)


def get_binding_for_introspection(
    store: Optional[Any],
) -> dict[str, Any]:
    """
    Get structured binding data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    if store is None:
        return {
            "has_bindings": False,
            "total_bindings": 0,
            "active_count": 0,
            "average_freshness": 0.0,
            "average_trace_count": 0.0,
            "emotion_distribution": {},
            "dominant_emotion": "",
            "strongest_binding_summary": "",
            "timestamp": "",
        }

    bindings = getattr(store, "bindings", ())

    emotion_dist: dict[str, int] = {}
    for binding in bindings:
        for trace in binding.traces:
            key = trace.emotion_label
            emotion_dist[key] = emotion_dist.get(key, 0) + 1

    dominant_emotion = ""
    if emotion_dist:
        dominant_emotion = max(emotion_dist, key=emotion_dist.get)

    strongest_summary = ""
    if bindings:
        strongest = max(bindings, key=lambda b: b.freshness)
        strongest_summary = strongest.memory_summary[:120]

    return {
        "has_bindings": len(bindings) > 0,
        "total_bindings": len(bindings),
        "active_count": getattr(store, "active_binding_count", 0),
        "average_freshness": getattr(store, "average_freshness", 0.0),
        "average_trace_count": getattr(store, "average_trace_count", 0.0),
        "emotion_distribution": emotion_dist,
        "dominant_emotion": dominant_emotion,
        "strongest_binding_summary": strongest_summary,
        "timestamp": getattr(store, "timestamp", ""),
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: BindingStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    感情痕跡は参照素材に留め、直接の意思決定経路を持たない。
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
                "total_bindings_created", "total_traces_created",
                "total_revisions", "total_expirations",
                "average_freshness", "average_trace_count",
                "active_binding_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: EmotionalMemoryBindingSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    感情痕跡から目標を生成しない。
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


def verify_read_only_principle(system: EmotionalMemoryBindingSystem) -> bool:
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


def verify_no_value_modification(system: EmotionalMemoryBindingSystem) -> bool:
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


def verify_no_emotion_evaluation(system: EmotionalMemoryBindingSystem) -> bool:
    """
    Verify the system has no emotion-evaluating methods.

    感情の正誤・善悪を評価しない。
    """
    forbidden = [
        "evaluate_emotion", "judge_emotion", "correct_emotion",
        "normalize_emotion", "optimize_emotion", "rate_emotion",
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

def create_config(**kwargs: Any) -> EmotionalMemoryBindingConfig:
    """Create a custom configuration."""
    return EmotionalMemoryBindingConfig(**kwargs)


def create_empty_store() -> BindingStore:
    """Create an empty binding store."""
    return BindingStore(
        bindings=(),
        binding_links=(),
        total_bindings_created=0,
        total_traces_created=0,
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.0,
        average_trace_count=0.0,
        active_binding_count=0,
        timestamp=str(time.time()),
        description="No emotional memory bindings formed yet.",
    )


def create_system(
    config: Optional[EmotionalMemoryBindingConfig] = None,
) -> EmotionalMemoryBindingSystem:
    """Create a new EmotionalMemoryBindingSystem."""
    return EmotionalMemoryBindingSystem(config=config)


def save_binding_state(
    store: BindingStore,
    filepath: str,
) -> None:
    """Save binding state to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_binding_state(filepath: str) -> BindingStore:
    """Load binding state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BindingStore.from_dict(data)
