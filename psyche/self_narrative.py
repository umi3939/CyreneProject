"""
Self-Narrative Formation (自己物語形成 - 非規範・観測型)

内部状態の断片を時間的連続として記述し、
自己の変化を「見える化」する基盤を持つためのモジュール。

自我形成における前段条件として、
記憶・感情・傾向・自己参照の断片を一つの時系列文脈に束ねる役割を担う。

CRITICAL DESIGN PRINCIPLES:
- 人格定義・価値付与・信念固定・目標決定を行わない
- 行動方針の正当化装置にもならない
- 入力は全て読み取り専用
- 接続先は内省記録層と自己記述提示層に限定
- 判断選択層、目的層、責任計算層、価値更新層には接続しない
- 物語内容を正誤評価しない
- 単一の「本当の自己」ラベルを固定しない
- 後続観測で叙述を再編集可能にし、単一解釈を固定しない
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class FragmentType(Enum):
    """
    Abstract unit types for narrative fragments.

    事実断片を圧縮する5つの抽象単位。
    NOT evaluative - just descriptive.
    """
    EVENT = "event"                # 出来事
    REACTION = "reaction"          # 反応
    CONTINUATION = "continuation"  # 継続
    CHANGE = "change"              # 変化
    UNDETERMINED = "undetermined"  # 未確定


class LinkType(Enum):
    """Types of connections between narrative fragments."""
    TEMPORAL = "temporal"              # 時間的順序
    THEMATIC = "thematic"              # テーマ的関連
    CONTRAST = "contrast"              # 対比・変化
    CONTINUATION_OF = "continuation_of"  # 同一スレッドの継続


class VividnessLevel(Enum):
    """
    Abstract level of fragment vividness.
    直近ほど鮮明で過去ほど要約化される。
    """
    VIVID = "vivid"                # 鮮明
    CLEAR = "clear"                # 明瞭
    FADING = "fading"              # 薄れつつある
    DIM = "dim"                    # おぼろげ
    DISSIPATING = "dissipating"    # 消散しつつある


class NarrativeCoherence(Enum):
    """How coherent the overall narrative currently feels."""
    COHERENT = "coherent"
    LOOSELY_CONNECTED = "loosely_connected"
    FRAGMENTED = "fragmented"
    UNDEFINED = "undefined"


class NarrativeTrend(Enum):
    """Direction the narrative state is moving."""
    STABLE = "stable"
    ACCUMULATING = "accumulating"    # 断片が蓄積している
    CONDENSING = "condensing"        # 要約化が進んでいる
    DISSOLVING = "dissolving"        # 断片が消散している
    UNDEFINED = "undefined"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class NarrativeFragment:
    """
    A single narrative fragment - an atomic unit of self-description.

    叙述断片の最小単位。観測結果を物語的要素に圧縮したもの。
    NOT evaluative, NOT prescriptive.
    """
    fragment_id: str
    fragment_type: FragmentType
    description: str
    timestamp: float
    vividness: float               # 0.0〜1.0, 時間とともに減衰
    reference_count: int           # 参照された回数（参照で鮮明度回復）
    undetermined_tags: tuple[str, ...]  # 未確定の側面
    source_type: str               # 生成元の入力種別
    rewrite_count: int             # 再編集された回数
    is_summary: bool               # 過去断片の要約かどうか

    def get_vividness_level(self) -> VividnessLevel:
        """Get abstract vividness level."""
        if self.vividness >= 0.8:
            return VividnessLevel.VIVID
        elif self.vividness >= 0.6:
            return VividnessLevel.CLEAR
        elif self.vividness >= 0.4:
            return VividnessLevel.FADING
        elif self.vividness >= 0.2:
            return VividnessLevel.DIM
        else:
            return VividnessLevel.DISSIPATING

    def is_undetermined(self) -> bool:
        """Check if this fragment has undetermined aspects."""
        return (
            self.fragment_type == FragmentType.UNDETERMINED
            or len(self.undetermined_tags) > 0
        )

    def with_vividness(self, new_vividness: float) -> NarrativeFragment:
        """Create a copy with updated vividness."""
        return NarrativeFragment(
            fragment_id=self.fragment_id,
            fragment_type=self.fragment_type,
            description=self.description,
            timestamp=self.timestamp,
            vividness=max(0.0, min(1.0, new_vividness)),
            reference_count=self.reference_count,
            undetermined_tags=self.undetermined_tags,
            source_type=self.source_type,
            rewrite_count=self.rewrite_count,
            is_summary=self.is_summary,
        )

    def with_reference(self) -> NarrativeFragment:
        """Create a copy with incremented reference count."""
        return NarrativeFragment(
            fragment_id=self.fragment_id,
            fragment_type=self.fragment_type,
            description=self.description,
            timestamp=self.timestamp,
            vividness=self.vividness,
            reference_count=self.reference_count + 1,
            undetermined_tags=self.undetermined_tags,
            source_type=self.source_type,
            rewrite_count=self.rewrite_count,
            is_summary=self.is_summary,
        )

    def rewrite(
        self,
        new_type: FragmentType,
        new_description: str,
        new_tags: tuple[str, ...] = (),
    ) -> NarrativeFragment:
        """Create a rewritten copy (re-editing by subsequent observation)."""
        return NarrativeFragment(
            fragment_id=self.fragment_id,
            fragment_type=new_type,
            description=new_description,
            timestamp=self.timestamp,
            vividness=self.vividness,
            reference_count=self.reference_count,
            undetermined_tags=new_tags,
            source_type=self.source_type,
            rewrite_count=self.rewrite_count + 1,
            is_summary=self.is_summary,
        )


@dataclass(frozen=True)
class FragmentLink:
    """A directional link between two narrative fragments."""
    from_id: str
    to_id: str
    link_type: LinkType
    description: str


@dataclass(frozen=True)
class CoherenceInfo:
    """
    Metadata about the coherence of the narrative.
    整合性メタ情報。NOT evaluative.
    """
    level: NarrativeCoherence
    fragment_count: int
    active_fragment_count: int
    summary_fragment_count: int
    average_vividness: float
    total_rewrites: int
    link_count: int
    connectivity: float  # リンクを持つ断片の割合


@dataclass(frozen=True)
class NarrativeState:
    """
    Complete snapshot of the self-narrative state.

    内省と自己記述のみに使用。
    判断・目的・責任・価値には接続しない。
    """
    fragments: tuple[NarrativeFragment, ...]
    links: tuple[FragmentLink, ...]
    coherence: CoherenceInfo
    dissipation_candidates: tuple[str, ...]  # 消散候補の断片ID
    trend: NarrativeTrend
    timestamp: float
    generation_count: int
    description: str

    @classmethod
    def empty(
        cls,
        timestamp: Optional[float] = None,
        generation_count: int = 0,
    ) -> NarrativeState:
        """Create an empty narrative state."""
        ts = timestamp or time.time()
        return cls(
            fragments=(),
            links=(),
            coherence=CoherenceInfo(
                level=NarrativeCoherence.UNDEFINED,
                fragment_count=0,
                active_fragment_count=0,
                summary_fragment_count=0,
                average_vividness=0.0,
                total_rewrites=0,
                link_count=0,
                connectivity=0.0,
            ),
            dissipation_candidates=(),
            trend=NarrativeTrend.UNDEFINED,
            timestamp=ts,
            generation_count=generation_count,
            description="No narrative fragments yet.",
        )

    def has_fragments(self) -> bool:
        return len(self.fragments) > 0

    def get_active_fragments(self) -> list[NarrativeFragment]:
        """Get fragments that haven't dissipated."""
        return [f for f in self.fragments if f.vividness > 0.0]

    def get_vivid_fragments(self) -> list[NarrativeFragment]:
        """Get vivid/clear fragments."""
        return [
            f for f in self.fragments
            if f.get_vividness_level()
            in (VividnessLevel.VIVID, VividnessLevel.CLEAR)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fragments": [
                {
                    "fragment_id": f.fragment_id,
                    "fragment_type": f.fragment_type.value,
                    "description": f.description,
                    "timestamp": f.timestamp,
                    "vividness": f.vividness,
                    "reference_count": f.reference_count,
                    "undetermined_tags": list(f.undetermined_tags),
                    "source_type": f.source_type,
                    "rewrite_count": f.rewrite_count,
                    "is_summary": f.is_summary,
                }
                for f in self.fragments
            ],
            "links": [
                {
                    "from_id": link.from_id,
                    "to_id": link.to_id,
                    "link_type": link.link_type.value,
                    "description": link.description,
                }
                for link in self.links
            ],
            "coherence": {
                "level": self.coherence.level.value,
                "fragment_count": self.coherence.fragment_count,
                "active_fragment_count": self.coherence.active_fragment_count,
                "summary_fragment_count": self.coherence.summary_fragment_count,
                "average_vividness": self.coherence.average_vividness,
                "total_rewrites": self.coherence.total_rewrites,
                "link_count": self.coherence.link_count,
                "connectivity": self.coherence.connectivity,
            },
            "dissipation_candidates": list(self.dissipation_candidates),
            "trend": self.trend.value,
            "timestamp": self.timestamp,
            "generation_count": self.generation_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeState:
        """Create from dictionary."""
        fragments = tuple(
            NarrativeFragment(
                fragment_id=f["fragment_id"],
                fragment_type=FragmentType(f["fragment_type"]),
                description=f["description"],
                timestamp=f["timestamp"],
                vividness=f["vividness"],
                reference_count=f.get("reference_count", 0),
                undetermined_tags=tuple(f.get("undetermined_tags", ())),
                source_type=f["source_type"],
                rewrite_count=f.get("rewrite_count", 0),
                is_summary=f.get("is_summary", False),
            )
            for f in data.get("fragments", [])
        )
        links = tuple(
            FragmentLink(
                from_id=link["from_id"],
                to_id=link["to_id"],
                link_type=LinkType(link["link_type"]),
                description=link["description"],
            )
            for link in data.get("links", [])
        )
        coh = data.get("coherence", {})
        coherence = CoherenceInfo(
            level=NarrativeCoherence(coh.get("level", "undefined")),
            fragment_count=coh.get("fragment_count", 0),
            active_fragment_count=coh.get("active_fragment_count", 0),
            summary_fragment_count=coh.get("summary_fragment_count", 0),
            average_vividness=coh.get("average_vividness", 0.0),
            total_rewrites=coh.get("total_rewrites", 0),
            link_count=coh.get("link_count", 0),
            connectivity=coh.get("connectivity", 0.0),
        )
        return cls(
            fragments=fragments,
            links=links,
            coherence=coherence,
            dissipation_candidates=tuple(
                data.get("dissipation_candidates", ())
            ),
            trend=NarrativeTrend(data.get("trend", "undefined")),
            timestamp=data["timestamp"],
            generation_count=data.get("generation_count", 0),
            description=data.get("description", ""),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SelfNarrativeConfig:
    """
    Configuration for the self-narrative system.

    These parameters control fragment lifecycle.
    They do NOT affect decisions.
    """
    # Fragment lifecycle
    max_fragments: int = 100
    vividness_decay_rate: float = 0.05
    reference_vividness_boost: float = 0.1

    # Thresholds
    dissipation_threshold: float = 0.1
    summarization_threshold: float = 0.3

    # Summarization
    max_fragments_per_summary: int = 5
    summary_vividness: float = 0.4

    # Rewriting
    rewrite_lookback: int = 10

    # Trend detection
    trend_lookback: int = 5


# =============================================================================
# Fragment ID Generation
# =============================================================================

def _generate_fragment_id() -> str:
    """Generate a unique fragment ID."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# Classification Functions
# =============================================================================

def classify_emotion_observation(
    emotion_summary: Optional[Any],
) -> list[tuple[FragmentType, str, str]]:
    """
    Classify emotion observations into narrative fragment candidates.

    Returns list of (fragment_type, description, source_type) tuples.
    """
    if emotion_summary is None:
        return []

    results = []

    # EmotionalStateView from self_model
    if hasattr(emotion_summary, "intensity") and hasattr(
        emotion_summary, "harmony"
    ):
        from .self_model import EmotionalIntensity, EmotionalHarmony

        intensity = emotion_summary.intensity
        harmony = emotion_summary.harmony
        desc = getattr(emotion_summary, "description", "")

        if intensity in (
            EmotionalIntensity.INTENSE,
            EmotionalIntensity.OVERWHELMING,
        ):
            results.append((
                FragmentType.EVENT,
                f"Intense emotional state: {desc}" if desc else
                "Intense emotional state detected",
                "emotion",
            ))
        elif intensity == EmotionalIntensity.MODERATE:
            results.append((
                FragmentType.CONTINUATION,
                f"Moderate emotional state: {desc}" if desc else
                "Moderate emotional state continues",
                "emotion",
            ))

        if harmony == EmotionalHarmony.CONFLICTED:
            results.append((
                FragmentType.REACTION,
                "Emotional conflict experienced",
                "emotion",
            ))

    return results


def classify_memory_observation(
    memory_summary: Optional[Any],
) -> list[tuple[FragmentType, str, str]]:
    """Classify memory observations into narrative fragment candidates."""
    if memory_summary is None:
        return []

    results = []

    if hasattr(memory_summary, "entries") and memory_summary.entries:
        recent = memory_summary.entries[-1]

        if hasattr(recent, "valence"):
            if abs(recent.valence) > 0.5:
                direction = "positive" if recent.valence > 0 else "negative"
                results.append((
                    FragmentType.EVENT,
                    f"Significant stimulus received ({direction})",
                    "memory",
                ))
            elif abs(recent.valence) > 0.2:
                results.append((
                    FragmentType.CONTINUATION,
                    "Mild stimuli observed",
                    "memory",
                ))

    return results


def classify_tendency_observation(
    tendency_awareness: Optional[Any],
) -> list[tuple[FragmentType, str, str]]:
    """Classify tendency observations into narrative fragment candidates."""
    if tendency_awareness is None:
        return []

    if not getattr(tendency_awareness, "has_awareness", False):
        return []

    results = []

    from .tendency_awareness import AwarenessType

    for item in getattr(tendency_awareness, "items", []):
        awareness_type = getattr(item, "awareness_type", None)
        desc = getattr(item, "description", "")

        if awareness_type == AwarenessType.STRONG_HABIT:
            results.append((
                FragmentType.CONTINUATION,
                f"Established tendency persists: {desc}" if desc else
                "Established tendency persists",
                "tendency",
            ))
        elif awareness_type == AwarenessType.HABIT_FORMING:
            results.append((
                FragmentType.CHANGE,
                f"New tendency forming: {desc}" if desc else
                "A new tendency is forming",
                "tendency",
            ))
        elif awareness_type == AwarenessType.FADING_HABIT:
            results.append((
                FragmentType.CHANGE,
                f"Tendency fading: {desc}" if desc else
                "An existing tendency is fading",
                "tendency",
            ))
        elif awareness_type == AwarenessType.SLIGHT_BIAS:
            results.append((
                FragmentType.UNDETERMINED,
                f"Slight bias noted: {desc}" if desc else
                "Slight behavioral bias (significance unclear)",
                "tendency",
            ))

    return results


def classify_difference_observation(
    difference_summary: Optional[Any],
) -> list[tuple[FragmentType, str, str]]:
    """Classify self-difference observations into narrative fragment candidates."""
    if difference_summary is None:
        return []

    has_diff = getattr(difference_summary, "has_difference", False)

    if not has_diff:
        return [(
            FragmentType.CONTINUATION,
            "Self-state remains stable",
            "difference",
        )]

    from .temporal_self_difference import ChangeNature

    nature = getattr(difference_summary, "nature", None)
    magnitude = getattr(difference_summary, "magnitude", None)
    mag_str = magnitude.value if magnitude else "unknown"

    if nature == ChangeNature.STABLE:
        return [(
            FragmentType.CONTINUATION,
            "Self-state is stable despite minor variations",
            "difference",
        )]
    elif nature == ChangeNature.FLUCTUATING:
        return [(
            FragmentType.UNDETERMINED,
            f"Self-state is fluctuating ({mag_str} magnitude)",
            "difference",
        )]
    elif nature == ChangeNature.SHIFTING:
        return [(
            FragmentType.CHANGE,
            f"Self-state is shifting ({mag_str} magnitude)",
            "difference",
        )]
    elif nature == ChangeNature.TRANSFORMED:
        return [(
            FragmentType.CHANGE,
            f"Self-state has transformed ({mag_str} magnitude)",
            "difference",
        )]
    elif nature == ChangeNature.RETURNING:
        return [(
            FragmentType.CHANGE,
            f"Self-state returning to previous pattern ({mag_str} magnitude)",
            "difference",
        )]

    return [(
        FragmentType.UNDETERMINED,
        "Self-difference state unclear",
        "difference",
    )]


def classify_context_observation(
    context_description: Optional[Any],
) -> list[tuple[FragmentType, str, str]]:
    """Classify context observations into narrative fragment candidates."""
    if context_description is None:
        return []

    results = []

    # String context
    if isinstance(context_description, str):
        if context_description.strip():
            results.append((
                FragmentType.EVENT,
                f"Context noted: {context_description}",
                "context",
            ))
        return results

    # ExternalContext from context_sensitivity
    if hasattr(context_description, "weight") and hasattr(
        context_description, "pace"
    ):
        weight = getattr(context_description, "weight", 0.5)
        pace = getattr(context_description, "pace", 0.5)

        if weight > 0.7:
            results.append((
                FragmentType.EVENT,
                "Heavy external context detected",
                "context",
            ))
        elif weight < 0.3:
            results.append((
                FragmentType.CONTINUATION,
                "Light external context continues",
                "context",
            ))

        if pace > 0.7:
            results.append((
                FragmentType.EVENT,
                "Fast-paced external context",
                "context",
            ))

    return results


# =============================================================================
# Rewrite Detection
# =============================================================================

def check_for_rewrites(
    recent_fragments: list[NarrativeFragment],
    new_classifications: list[tuple[FragmentType, str, str]],
    config: SelfNarrativeConfig,
) -> list[NarrativeFragment]:
    """
    Check if recent fragments should be rewritten based on new observations.

    後続観測で叙述を再編集可能にし、単一解釈を固定しない。
    Returns list of rewritten fragments.
    """
    rewrites = []
    lookback = recent_fragments[-config.rewrite_lookback:]

    for fragment in lookback:
        if fragment.is_summary:
            continue

        for new_type, new_desc, new_source in new_classifications:
            if fragment.source_type != new_source:
                continue

            # CHANGE → CONTINUATION: the change didn't persist
            if (
                fragment.fragment_type == FragmentType.CHANGE
                and new_type == FragmentType.CONTINUATION
            ):
                rewrites.append(fragment.rewrite(
                    FragmentType.EVENT,
                    f"Temporary shift (previously: {fragment.description})",
                    ("reinterpreted",),
                ))
                break

            # UNDETERMINED → specific type: clarification
            if (
                fragment.fragment_type == FragmentType.UNDETERMINED
                and new_type not in (
                    FragmentType.UNDETERMINED,
                    FragmentType.CONTINUATION,
                )
            ):
                rewrites.append(fragment.rewrite(
                    new_type,
                    f"Clarified: {new_desc}",
                    (),
                ))
                break

    return rewrites


# =============================================================================
# Summarization
# =============================================================================

def summarize_fragments(
    fragments: list[NarrativeFragment],
    config: SelfNarrativeConfig,
) -> Optional[NarrativeFragment]:
    """
    Summarize multiple dim fragments into a single summary fragment.

    過去ほど要約化される持続構造を実現する。
    """
    if len(fragments) < 2:
        return None

    types_seen = set(f.fragment_type for f in fragments)

    type_counts: dict[str, int] = {}
    for f in fragments:
        key = f.fragment_type.value
        type_counts[key] = type_counts.get(key, 0) + 1

    type_desc = ", ".join(f"{c} {t}" for t, c in type_counts.items())

    # Determine the dominant type for the summary
    dominant_type = max(type_counts, key=type_counts.get)

    return NarrativeFragment(
        fragment_id=_generate_fragment_id(),
        fragment_type=FragmentType(dominant_type),
        description=f"Summary of {len(fragments)} past observations ({type_desc})",
        timestamp=max(f.timestamp for f in fragments),
        vividness=config.summary_vividness,
        reference_count=0,
        undetermined_tags=("summarized",),
        source_type="summary",
        rewrite_count=0,
        is_summary=True,
    )


# =============================================================================
# Coherence Computation
# =============================================================================

def compute_coherence(
    fragments: list[NarrativeFragment],
    links: list[FragmentLink],
    config: SelfNarrativeConfig,
) -> CoherenceInfo:
    """Compute coherence metadata for the current narrative."""
    if not fragments:
        return CoherenceInfo(
            level=NarrativeCoherence.UNDEFINED,
            fragment_count=0,
            active_fragment_count=0,
            summary_fragment_count=0,
            average_vividness=0.0,
            total_rewrites=0,
            link_count=0,
            connectivity=0.0,
        )

    fragment_count = len(fragments)
    active = [f for f in fragments if f.vividness > config.dissipation_threshold]
    active_count = len(active)
    summary_count = sum(1 for f in fragments if f.is_summary)
    avg_vividness = sum(f.vividness for f in fragments) / fragment_count
    total_rewrites = sum(f.rewrite_count for f in fragments)
    link_count = len(links)

    # Compute connectivity: fraction of fragments that have at least one link
    all_ids = {f.fragment_id for f in fragments}
    linked_ids = set()
    for link in links:
        if link.from_id in all_ids:
            linked_ids.add(link.from_id)
        if link.to_id in all_ids:
            linked_ids.add(link.to_id)
    connectivity = len(linked_ids) / fragment_count if fragment_count > 0 else 0.0

    # Determine coherence level
    if connectivity >= 0.7 and avg_vividness >= 0.4:
        level = NarrativeCoherence.COHERENT
    elif connectivity >= 0.4:
        level = NarrativeCoherence.LOOSELY_CONNECTED
    elif fragment_count > 0:
        level = NarrativeCoherence.FRAGMENTED
    else:
        level = NarrativeCoherence.UNDEFINED

    return CoherenceInfo(
        level=level,
        fragment_count=fragment_count,
        active_fragment_count=active_count,
        summary_fragment_count=summary_count,
        average_vividness=round(avg_vividness, 4),
        total_rewrites=total_rewrites,
        link_count=link_count,
        connectivity=round(connectivity, 4),
    )


# =============================================================================
# Trend Determination
# =============================================================================

def determine_narrative_trend(
    coherence_history: list[NarrativeCoherence],
    last_state: Optional[NarrativeState],
    current_fragment_count: int,
    has_summaries: bool,
    config: SelfNarrativeConfig,
) -> NarrativeTrend:
    """Determine the current narrative trend."""
    if len(coherence_history) < 2:
        return NarrativeTrend.UNDEFINED

    if current_fragment_count == 0:
        return NarrativeTrend.DISSOLVING

    if last_state is not None:
        prev_count = last_state.coherence.fragment_count

        if current_fragment_count > prev_count:
            return NarrativeTrend.ACCUMULATING
        elif current_fragment_count < prev_count and has_summaries:
            return NarrativeTrend.CONDENSING
        elif current_fragment_count < prev_count:
            return NarrativeTrend.DISSOLVING

    return NarrativeTrend.STABLE


# =============================================================================
# Description Generation
# =============================================================================

def generate_narrative_description(
    coherence: CoherenceInfo,
    trend: NarrativeTrend,
) -> str:
    """
    Generate human-readable narrative description.

    For introspection/self-description only.
    NOT evaluative. NOT prescriptive.
    """
    if coherence.fragment_count == 0:
        return "No narrative fragments yet."

    parts = []

    if coherence.level == NarrativeCoherence.COHERENT:
        parts.append("The self-narrative forms a connected sequence")
    elif coherence.level == NarrativeCoherence.LOOSELY_CONNECTED:
        parts.append("The self-narrative has some connections but also gaps")
    elif coherence.level == NarrativeCoherence.FRAGMENTED:
        parts.append("The self-narrative consists of disconnected fragments")
    else:
        parts.append("The self-narrative state is unclear")

    parts.append(
        f"with {coherence.active_fragment_count} active fragments"
        f" out of {coherence.fragment_count} total"
    )

    if coherence.summary_fragment_count > 0:
        parts.append(f"({coherence.summary_fragment_count} summarized)")

    if coherence.total_rewrites > 0:
        parts.append(f"and {coherence.total_rewrites} reinterpretations")

    if trend == NarrativeTrend.ACCUMULATING:
        parts.append("— narrative is growing")
    elif trend == NarrativeTrend.CONDENSING:
        parts.append("— narrative is condensing")
    elif trend == NarrativeTrend.DISSOLVING:
        parts.append("— narrative is fading")

    return ", ".join(parts) + "."


# =============================================================================
# Self-Narrative System
# =============================================================================

class SelfNarrativeSystem:
    """
    Self-Narrative Formation System (自己物語形成)

    内部状態の断片を時間的連続として記述し、
    自己の変化を見える化する基盤。

    CRITICAL CONSTRAINTS:
    - 入力は全て読み取り専用
    - 接続先は内省記録層と自己記述提示層に限定
    - 判断選択層、目的層、責任計算層、価値更新層には接続しない
    - 物語内容を正誤評価しない
    - 単一の「本当の自己」ラベルを固定しない
    - 後続観測で叙述を再編集可能
    - 参照されない断片は自然に減衰
    """

    def __init__(self, config: Optional[SelfNarrativeConfig] = None):
        self._config = config or SelfNarrativeConfig()
        self._fragments: list[NarrativeFragment] = []
        self._links: list[FragmentLink] = []
        self._generation_count: int = 0
        self._last_state: Optional[NarrativeState] = None
        self._coherence_history: list[NarrativeCoherence] = []

    def observe_and_generate(
        self,
        emotion_summary: Optional[Any] = None,
        memory_summary: Optional[Any] = None,
        tendency_awareness: Optional[Any] = None,
        difference_summary: Optional[Any] = None,
        context_description: Optional[Any] = None,
    ) -> NarrativeState:
        """
        Observe current state and generate/update the narrative.

        入力として、既存の感情要約、記憶要約、傾向観測、
        自己差分観測、文脈記述を読み取り専用で参照する。

        変換は、事実断片を抽象単位へ圧縮し、
        時系列の叙述断片へ再構成する。
        """
        current_time = time.time()
        self._generation_count += 1

        has_inputs = not all(x is None for x in [
            emotion_summary, memory_summary,
            tendency_awareness, difference_summary,
            context_description,
        ])

        # Steps 1-3: Only when inputs are provided
        if has_inputs:
            # Step 1: Classify observations into fragment candidates
            candidates = []
            candidates.extend(classify_emotion_observation(emotion_summary))
            candidates.extend(classify_memory_observation(memory_summary))
            candidates.extend(classify_tendency_observation(tendency_awareness))
            candidates.extend(classify_difference_observation(difference_summary))
            candidates.extend(classify_context_observation(context_description))

            # Step 2: Check for rewrites of recent fragments
            if self._fragments and candidates:
                rewrites = check_for_rewrites(
                    self._fragments, candidates, self._config,
                )
                for rewritten in rewrites:
                    self._replace_fragment(rewritten)

            # Step 3: Create new fragments and link them
            for ftype, desc, source in candidates:
                fragment = NarrativeFragment(
                    fragment_id=_generate_fragment_id(),
                    fragment_type=ftype,
                    description=desc,
                    timestamp=current_time,
                    vividness=1.0,
                    reference_count=0,
                    undetermined_tags=(
                        ("fresh",) if ftype == FragmentType.UNDETERMINED else ()
                    ),
                    source_type=source,
                    rewrite_count=0,
                    is_summary=False,
                )
                self._add_fragment_with_links(fragment)

        # Step 4: Decay vividness of all existing fragments
        self._decay_all()

        # Step 5: Summarize old dim fragments
        self._summarize_old()

        # Step 6: Remove dissipated fragments (vividness <= 0)
        self._remove_dissipated()

        # Step 7: Enforce max fragment limit
        self._enforce_limit()

        # Build and return state
        state = self._build_state(current_time)
        self._last_state = state
        return state

    def get_last_state(self) -> Optional[NarrativeState]:
        """Get the last generated state (reference only)."""
        return self._last_state

    def get_generation_count(self) -> int:
        """Get how many states have been generated."""
        return self._generation_count

    def reference_fragment(self, fragment_id: str) -> None:
        """
        Mark a fragment as referenced (boosts vividness).

        Called when introspection accesses a fragment.
        参照されない断片は自然に減衰 → 参照で回復。
        """
        for i, f in enumerate(self._fragments):
            if f.fragment_id == fragment_id:
                boosted = f.with_reference()
                boosted = boosted.with_vividness(
                    boosted.vividness + self._config.reference_vividness_boost,
                )
                self._fragments[i] = boosted
                return

    # ----- Internal methods -----

    def _replace_fragment(self, rewritten: NarrativeFragment) -> None:
        """Replace an existing fragment with a rewritten version."""
        for i, f in enumerate(self._fragments):
            if f.fragment_id == rewritten.fragment_id:
                self._fragments[i] = rewritten
                return

    def _add_fragment_with_links(self, fragment: NarrativeFragment) -> None:
        """Add a fragment and create appropriate links."""
        self._fragments.append(fragment)

        if len(self._fragments) < 2:
            return

        # Temporal link to the immediately preceding fragment
        prev = self._fragments[-2]
        self._links.append(FragmentLink(
            from_id=prev.fragment_id,
            to_id=fragment.fragment_id,
            link_type=LinkType.TEMPORAL,
            description="Sequential observation",
        ))

        # Thematic link (same source, still vivid)
        for existing in reversed(self._fragments[:-2]):
            if (
                existing.source_type == fragment.source_type
                and existing.vividness > self._config.summarization_threshold
            ):
                self._links.append(FragmentLink(
                    from_id=existing.fragment_id,
                    to_id=fragment.fragment_id,
                    link_type=LinkType.THEMATIC,
                    description=f"Related by source: {fragment.source_type}",
                ))
                break

        # Contrast link (opposite types from same source)
        contrast_pairs = {
            (FragmentType.CHANGE, FragmentType.CONTINUATION),
            (FragmentType.CONTINUATION, FragmentType.CHANGE),
            (FragmentType.EVENT, FragmentType.CONTINUATION),
        }
        for existing in reversed(self._fragments[:-1]):
            if existing.source_type == fragment.source_type:
                if (existing.fragment_type, fragment.fragment_type) in contrast_pairs:
                    self._links.append(FragmentLink(
                        from_id=existing.fragment_id,
                        to_id=fragment.fragment_id,
                        link_type=LinkType.CONTRAST,
                        description=(
                            f"Contrast: {existing.fragment_type.value}"
                            f" → {fragment.fragment_type.value}"
                        ),
                    ))
                break

        # Continuation link
        if fragment.fragment_type == FragmentType.CONTINUATION:
            for existing in reversed(self._fragments[:-1]):
                if (
                    existing.source_type == fragment.source_type
                    and existing.fragment_type == FragmentType.CONTINUATION
                ):
                    self._links.append(FragmentLink(
                        from_id=existing.fragment_id,
                        to_id=fragment.fragment_id,
                        link_type=LinkType.CONTINUATION_OF,
                        description="Continuation of same observation",
                    ))
                    break

    def _decay_all(self) -> None:
        """Decay vividness of all fragments."""
        self._fragments = [
            f.with_vividness(f.vividness - self._config.vividness_decay_rate)
            for f in self._fragments
        ]

    def _summarize_old(self) -> None:
        """Summarize old fragments that have faded below threshold."""
        eligible = [
            f for f in self._fragments
            if (
                f.vividness <= self._config.summarization_threshold
                and not f.is_summary
                and f.vividness > self._config.dissipation_threshold
            )
        ]

        if len(eligible) < 2:
            return

        # Group by source type
        by_source: dict[str, list[NarrativeFragment]] = {}
        for f in eligible:
            by_source.setdefault(f.source_type, []).append(f)

        for source, group in by_source.items():
            if len(group) >= 2:
                batch = group[:self._config.max_fragments_per_summary]
                summary = summarize_fragments(batch, self._config)
                if summary:
                    batch_ids = {f.fragment_id for f in batch}
                    self._fragments = [
                        f for f in self._fragments
                        if f.fragment_id not in batch_ids
                    ]
                    self._links = [
                        link for link in self._links
                        if (
                            link.from_id not in batch_ids
                            and link.to_id not in batch_ids
                        )
                    ]
                    self._fragments.append(summary)

    def _remove_dissipated(self) -> None:
        """Remove fragments that have fully dissipated."""
        to_remove = {
            f.fragment_id for f in self._fragments
            if f.vividness <= 0.0
        }
        if not to_remove:
            return

        self._fragments = [
            f for f in self._fragments
            if f.fragment_id not in to_remove
        ]
        self._links = [
            link for link in self._links
            if (
                link.from_id not in to_remove
                and link.to_id not in to_remove
            )
        ]

    def _enforce_limit(self) -> None:
        """Enforce maximum fragment count."""
        if len(self._fragments) <= self._config.max_fragments:
            return

        self._fragments.sort(key=lambda f: f.vividness, reverse=True)
        removed = {
            f.fragment_id
            for f in self._fragments[self._config.max_fragments:]
        }
        self._fragments = self._fragments[:self._config.max_fragments]
        self._links = [
            link for link in self._links
            if link.from_id not in removed and link.to_id not in removed
        ]

    def _build_state(self, current_time: float) -> NarrativeState:
        """Build a NarrativeState snapshot."""
        coherence = compute_coherence(
            self._fragments, self._links, self._config,
        )

        self._coherence_history.append(coherence.level)
        max_history = self._config.trend_lookback * 2
        if len(self._coherence_history) > max_history:
            self._coherence_history = self._coherence_history[-max_history:]

        has_summaries = any(f.is_summary for f in self._fragments)
        trend = determine_narrative_trend(
            self._coherence_history,
            self._last_state,
            len(self._fragments),
            has_summaries,
            self._config,
        )

        description = generate_narrative_description(coherence, trend)

        dissipation_ids = tuple(
            f.fragment_id for f in self._fragments
            if (
                f.vividness <= self._config.dissipation_threshold
                and f.vividness > 0
            )
        )

        return NarrativeState(
            fragments=tuple(self._fragments),
            links=tuple(self._links),
            coherence=coherence,
            dissipation_candidates=dissipation_ids,
            trend=trend,
            timestamp=current_time,
            generation_count=self._generation_count,
            description=description,
        )


# =============================================================================
# Integration with Introspection/Self-Description Layers
# =============================================================================

def generate_narrative_tags(
    state: NarrativeState,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from NarrativeState for SelfReferenceSystem integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags = []

    if not state.has_fragments():
        tags.append({
            "category": "SELF_NARRATIVE",
            "label": "no_narrative",
            "description": "No self-narrative fragments exist yet",
            "weight": 0.05 * scale,
        })
        return tags

    # Coherence tag
    tags.append({
        "category": "SELF_NARRATIVE_COHERENCE",
        "label": f"narrative_{state.coherence.level.value}",
        "description": (
            f"Self-narrative coherence: {state.coherence.level.value}"
        ),
        "weight": 0.08 * scale,
    })

    # Trend tag
    if state.trend not in (NarrativeTrend.STABLE, NarrativeTrend.UNDEFINED):
        tags.append({
            "category": "SELF_NARRATIVE_TREND",
            "label": f"narrative_{state.trend.value}",
            "description": f"Narrative trend: {state.trend.value}",
            "weight": 0.05 * scale,
        })

    # Dominant fragment type
    type_counts: dict[str, int] = {}
    for f in state.get_active_fragments():
        type_counts[f.fragment_type.value] = (
            type_counts.get(f.fragment_type.value, 0) + 1
        )
    if type_counts:
        dominant = max(type_counts, key=type_counts.get)
        tags.append({
            "category": "SELF_NARRATIVE_DOMINANT",
            "label": f"narrative_mostly_{dominant}",
            "description": f"Narrative dominated by {dominant} fragments",
            "weight": 0.05 * scale,
        })

    # Most recent vivid fragment
    vivid = state.get_vivid_fragments()
    if vivid:
        most_recent = vivid[-1]
        tags.append({
            "category": "SELF_NARRATIVE_RECENT",
            "label": f"recent_{most_recent.fragment_type.value}",
            "description": most_recent.description,
            "weight": 0.08 * scale,
        })

    # Integrated description
    tags.append({
        "category": "SELF_NARRATIVE_INTEGRATED",
        "label": "narrative_awareness",
        "description": state.description,
        "weight": 0.1 * scale,
    })

    return tags


def get_narrative_summary(state: NarrativeState) -> str:
    """Get human-readable summary. For introspection/logging only."""
    lines = [
        "=== Self-Narrative State ===",
        f"Coherence: {state.coherence.level.value}",
        f"Trend: {state.trend.value}",
        f"Fragments: {state.coherence.active_fragment_count} active"
        f" / {state.coherence.fragment_count} total",
        f"Summary fragments: {state.coherence.summary_fragment_count}",
        f"Links: {state.coherence.link_count}",
        f"Connectivity: {state.coherence.connectivity:.1%}",
        f"Average vividness: {state.coherence.average_vividness:.2f}",
        f"Total rewrites: {state.coherence.total_rewrites}",
        "",
    ]

    if state.has_fragments():
        lines.append("Recent fragments:")
        vivid = state.get_vivid_fragments()
        for f in vivid[-5:]:
            lines.append(
                f"  [{f.fragment_type.value}] {f.description}"
                f" (vividness: {f.get_vividness_level().value})"
            )
        lines.append("")

    if state.dissipation_candidates:
        lines.append(
            f"Dissipation candidates: {len(state.dissipation_candidates)}"
        )

    lines.append(f"Integrated: {state.description}")
    return "\n".join(lines)


def get_narrative_for_introspection(
    state: NarrativeState,
) -> dict[str, Any]:
    """
    Get structured narrative data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    type_counts: dict[str, int] = {}
    for f in state.fragments:
        type_counts[f.fragment_type.value] = (
            type_counts.get(f.fragment_type.value, 0) + 1
        )

    return {
        "has_narrative": state.has_fragments(),
        "coherence_level": state.coherence.level.value,
        "trend": state.trend.value,
        "fragment_count": state.coherence.fragment_count,
        "active_fragment_count": state.coherence.active_fragment_count,
        "summary_count": state.coherence.summary_fragment_count,
        "average_vividness": state.coherence.average_vividness,
        "connectivity": state.coherence.connectivity,
        "total_rewrites": state.coherence.total_rewrites,
        "fragment_type_distribution": type_counts,
        "dissipation_candidate_count": len(state.dissipation_candidates),
        "description": state.description,
        "generation_count": state.generation_count,
        "timestamp": state.timestamp,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config(
    max_fragments: int = 100,
    vividness_decay_rate: float = 0.05,
    dissipation_threshold: float = 0.1,
    summarization_threshold: float = 0.3,
    trend_lookback: int = 5,
) -> SelfNarrativeConfig:
    """Create a custom configuration."""
    return SelfNarrativeConfig(
        max_fragments=max_fragments,
        vividness_decay_rate=vividness_decay_rate,
        dissipation_threshold=dissipation_threshold,
        summarization_threshold=summarization_threshold,
        trend_lookback=trend_lookback,
    )


def create_empty_state() -> NarrativeState:
    """Create an empty narrative state."""
    return NarrativeState.empty()


# =============================================================================
# Self-Observation Chain Integration
# =============================================================================

def observe_from_chain(
    system: SelfNarrativeSystem,
    emotional_state: Optional[Any] = None,
    short_term_memory: Optional[Any] = None,
    tendency_awareness: Optional[Any] = None,
    difference_summary: Optional[Any] = None,
    external_context: Optional[Any] = None,
) -> NarrativeState:
    """
    Integrate self-observation chain outputs into the narrative system.

    自己観測チェーンの出力を自己物語に接続するヘルパー。
    各入力は読み取り専用で参照される。

    Args:
        system: SelfNarrativeSystem instance
        emotional_state: EmotionalStateView from self_model
        short_term_memory: ShortTermMemory (entries with valence)
        tendency_awareness: TendencyAwareness from tendency_awareness
        difference_summary: SelfDifferenceSummary from temporal_self_difference
        external_context: ExternalContext from context_sensitivity, or str
    """
    return system.observe_and_generate(
        emotion_summary=emotional_state,
        memory_summary=short_term_memory,
        tendency_awareness=tendency_awareness,
        difference_summary=difference_summary,
        context_description=external_context,
    )


# =============================================================================
# Verification (Test Support)
# =============================================================================

def verify_no_decision_impact(state: NarrativeState) -> bool:
    """
    Verify that the narrative state has no decision-impacting values.

    自己物語は説明表現のみを生成する観測成果物として扱う。
    """
    public_attrs = [a for a in dir(state) if not a.startswith("_")]

    for attr in public_attrs:
        if callable(getattr(state, attr)):
            continue
        value = getattr(state, attr)

        if attr in ("timestamp", "generation_count"):
            continue
        if isinstance(value, str):
            continue
        if isinstance(value, Enum):
            continue
        if isinstance(value, (tuple, CoherenceInfo)):
            continue
        if isinstance(value, (int, float)) and attr not in (
            "timestamp", "generation_count",
        ):
            return False

    return True


def verify_no_identity_definition(state: NarrativeState) -> bool:
    """
    Verify the narrative does not define identity.

    単一の「本当の自己」ラベルを固定しない。
    """
    forbidden = [
        "true self", "real self", "correct identity",
        "should be", "must be", "need to return",
        "proper self", "restore to",
    ]

    desc = state.description.lower()
    for phrase in forbidden:
        if phrase in desc:
            return False

    for f in state.fragments:
        fdesc = f.description.lower()
        for phrase in forbidden:
            if phrase in fdesc:
                return False

    return True


def verify_no_goal_generation(system: SelfNarrativeSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    物語から目標を生成しない。
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


def verify_read_only_principle(system: SelfNarrativeSystem) -> bool:
    """
    Verify the system does not write to external systems.

    自己参照の読み取り専用原則を破らない。
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
