"""
Introspection Consumption Layer (内省の消費層)

内省観測の結果を「読み取り可能な断片」として再編成し、
自己に関する叙述の素材として循環させるためのモジュール。

現状の課題:
introspection_trace, self_narrative, identity_coherence,
tendency_awareness, episodic_memory が大量の観測データを生成しているが、
消費先（読み取られる構造）がなく沈殿している。

CRITICAL DESIGN PRINCIPLES:
- 入力は全て Optional[Any] + duck typing（循環import回避）
- 判断・目的・価値・責任に一切接続しない
- 内省結果を正解化・評価化しない
- 人格の方向性を固定しない
- 断片の束ね方は固定しない（再要約・再リンクを許容）
- 以前の読み取りと異なる解釈が生まれることを許す
- 出力は観測素材の提供に限定
- 直接的な意思決定や責任計算への接続を持たない
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

class FragmentSourceType(Enum):
    """
    Source type of an introspection fragment.

    どの観測系から断片が抽出されたかを示す。
    評価的ではなく、記述的な分類。
    """
    INTROSPECTION_LOG = "introspection_log"
    SELF_NARRATIVE = "self_narrative"
    IDENTITY_COHERENCE = "identity_coherence"
    TENDENCY_AWARENESS = "tendency_awareness"
    EPISODIC_MEMORY = "episodic_memory"
    MIXED = "mixed"


class BundleCoherence(Enum):
    """
    How coherently a bundle of fragments hangs together.

    束の一貫性。NOT evaluative.
    """
    TIGHT = "tight"            # Strongly related fragments
    LOOSE = "loose"            # Weakly related fragments
    SCATTERED = "scattered"    # Barely related fragments
    UNDEFINED = "undefined"    # Cannot determine


class FragmentFreshness(Enum):
    """
    Abstract freshness level of a fragment.

    断片の鮮度段階。
    """
    FRESH = "fresh"        # Just created or recently referenced
    RECENT = "recent"      # Still active
    AGING = "aging"        # Starting to fade
    STALE = "stale"        # Barely readable
    FADED = "faded"        # About to be removed


# =============================================================================
# ID Generation
# =============================================================================

def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# Freshness Determination
# =============================================================================

def determine_freshness(value: float) -> FragmentFreshness:
    """Determine FragmentFreshness from a numeric freshness value (0.0-1.0)."""
    if value >= 0.8:
        return FragmentFreshness.FRESH
    elif value >= 0.6:
        return FragmentFreshness.RECENT
    elif value >= 0.4:
        return FragmentFreshness.AGING
    elif value >= 0.15:
        return FragmentFreshness.STALE
    else:
        return FragmentFreshness.FADED


# =============================================================================
# Core Dataclasses (all frozen)
# =============================================================================

@dataclass(frozen=True)
class IntrospectionFragment:
    """
    内省観測から抽出された読み取り可能な断片。

    内省の痕跡が「参照されうる状態」になるための最小単位。
    NOT evaluative. NOT prescriptive.
    """
    fragment_id: str
    source_type: FragmentSourceType
    content: str                          # 要約テキスト
    timestamp: float
    freshness: float                      # 0.0〜1.0
    reference_count: int                  # 参照された回数
    source_ids: tuple[str, ...]           # 元の観測ID群
    undetermined_aspects: tuple[str, ...]  # 未確定の側面

    def get_freshness_level(self) -> FragmentFreshness:
        """Get abstract freshness level."""
        return determine_freshness(self.freshness)

    def with_freshness(self, new_freshness: float) -> IntrospectionFragment:
        """Create a copy with updated freshness."""
        return IntrospectionFragment(
            fragment_id=self.fragment_id,
            source_type=self.source_type,
            content=self.content,
            timestamp=self.timestamp,
            freshness=max(0.0, min(1.0, new_freshness)),
            reference_count=self.reference_count,
            source_ids=self.source_ids,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_reference(self) -> IntrospectionFragment:
        """Create a copy with incremented reference count."""
        return IntrospectionFragment(
            fragment_id=self.fragment_id,
            source_type=self.source_type,
            content=self.content,
            timestamp=self.timestamp,
            freshness=self.freshness,
            reference_count=self.reference_count + 1,
            source_ids=self.source_ids,
            undetermined_aspects=self.undetermined_aspects,
        )

    def recompose(self, new_content: str, new_aspects: tuple[str, ...] = ()) -> IntrospectionFragment:
        """Create a recomposed copy (再編成 - 解釈は固定しない)."""
        return IntrospectionFragment(
            fragment_id=self.fragment_id,
            source_type=self.source_type,
            content=new_content,
            timestamp=self.timestamp,
            freshness=self.freshness,
            reference_count=self.reference_count,
            source_ids=self.source_ids,
            undetermined_aspects=new_aspects,
        )


@dataclass(frozen=True)
class FragmentBundle:
    """
    関連する断片の弱い束。

    断片同士の関連を弱く束ねる。
    束ね方は固定しない（再要約・再リンクを許容）。
    """
    bundle_id: str
    fragment_ids: tuple[str, ...]
    theme_description: str
    coherence: BundleCoherence
    timestamp: float
    strength: float  # 0.0〜1.0


@dataclass(frozen=True)
class ConsumptionRecord:
    """
    断片が「読まれた」記録。

    消費は読み取り行為の記録であり、評価ではない。
    """
    record_id: str
    fragment_ids: tuple[str, ...]
    timestamp: float
    context_description: str


@dataclass(frozen=True)
class ConsumptionStore:
    """
    不変スナップショット - 消費層の全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    fragments: tuple[IntrospectionFragment, ...]
    bundles: tuple[FragmentBundle, ...]
    consumption_history: tuple[ConsumptionRecord, ...]
    total_fragments_created: int
    total_consumptions: int
    average_freshness: float
    active_fragment_count: int
    bundle_count: int
    timestamp: float
    description: str

    @classmethod
    def empty(cls, timestamp: Optional[float] = None) -> ConsumptionStore:
        """Create an empty store."""
        ts = timestamp or time.time()
        return cls(
            fragments=(),
            bundles=(),
            consumption_history=(),
            total_fragments_created=0,
            total_consumptions=0,
            average_freshness=0.0,
            active_fragment_count=0,
            bundle_count=0,
            timestamp=ts,
            description="No introspection fragments yet.",
        )

    def has_fragments(self) -> bool:
        return len(self.fragments) > 0

    def get_active_fragments(self) -> list[IntrospectionFragment]:
        """Get fragments that haven't fully faded."""
        return [f for f in self.fragments if f.freshness > 0.0]

    def get_fresh_fragments(self) -> list[IntrospectionFragment]:
        """Get fresh/recent fragments."""
        return [
            f for f in self.fragments
            if f.get_freshness_level() in (FragmentFreshness.FRESH, FragmentFreshness.RECENT)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fragments": [
                _fragment_to_dict(f) for f in self.fragments
            ],
            "bundles": [
                {
                    "bundle_id": b.bundle_id,
                    "fragment_ids": list(b.fragment_ids),
                    "theme_description": b.theme_description,
                    "coherence": b.coherence.value,
                    "timestamp": b.timestamp,
                    "strength": b.strength,
                }
                for b in self.bundles
            ],
            "consumption_history": [
                {
                    "record_id": r.record_id,
                    "fragment_ids": list(r.fragment_ids),
                    "timestamp": r.timestamp,
                    "context_description": r.context_description,
                }
                for r in self.consumption_history
            ],
            "total_fragments_created": self.total_fragments_created,
            "total_consumptions": self.total_consumptions,
            "average_freshness": self.average_freshness,
            "active_fragment_count": self.active_fragment_count,
            "bundle_count": self.bundle_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsumptionStore:
        """Create from dictionary."""
        fragments = tuple(
            _fragment_from_dict(f) for f in data.get("fragments", [])
        )
        bundles = tuple(
            FragmentBundle(
                bundle_id=b["bundle_id"],
                fragment_ids=tuple(b.get("fragment_ids", ())),
                theme_description=b.get("theme_description", ""),
                coherence=BundleCoherence(b.get("coherence", "undefined")),
                timestamp=b.get("timestamp", 0.0),
                strength=b.get("strength", 0.0),
            )
            for b in data.get("bundles", [])
        )
        history = tuple(
            ConsumptionRecord(
                record_id=r["record_id"],
                fragment_ids=tuple(r.get("fragment_ids", ())),
                timestamp=r.get("timestamp", 0.0),
                context_description=r.get("context_description", ""),
            )
            for r in data.get("consumption_history", [])
        )
        return cls(
            fragments=fragments,
            bundles=bundles,
            consumption_history=history,
            total_fragments_created=data.get("total_fragments_created", 0),
            total_consumptions=data.get("total_consumptions", 0),
            average_freshness=data.get("average_freshness", 0.0),
            active_fragment_count=data.get("active_fragment_count", 0),
            bundle_count=data.get("bundle_count", len(bundles)),
            timestamp=data.get("timestamp", 0.0),
            description=data.get("description", ""),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ConsumptionLayerConfig:
    """
    Configuration for the introspection consumption layer.

    These parameters control fragment lifecycle.
    They do NOT affect decisions.
    """
    max_fragments: int = 150
    base_decay_rate: float = 0.03
    bundle_strength_threshold: float = 0.3
    max_bundles: int = 30
    freshness_boost_on_reference: float = 0.12
    stale_threshold: float = 0.15
    max_consumption_history: int = 50


# =============================================================================
# Serialization Helpers
# =============================================================================

def _fragment_to_dict(fragment: IntrospectionFragment) -> dict[str, Any]:
    """Convert an IntrospectionFragment to a dictionary."""
    return {
        "fragment_id": fragment.fragment_id,
        "source_type": fragment.source_type.value,
        "content": fragment.content,
        "timestamp": fragment.timestamp,
        "freshness": fragment.freshness,
        "reference_count": fragment.reference_count,
        "source_ids": list(fragment.source_ids),
        "undetermined_aspects": list(fragment.undetermined_aspects),
    }


def _fragment_from_dict(data: dict[str, Any]) -> IntrospectionFragment:
    """Create an IntrospectionFragment from a dictionary."""
    return IntrospectionFragment(
        fragment_id=data["fragment_id"],
        source_type=FragmentSourceType(data.get("source_type", "mixed")),
        content=data.get("content", ""),
        timestamp=data.get("timestamp", 0.0),
        freshness=data.get("freshness", 0.0),
        reference_count=data.get("reference_count", 0),
        source_ids=tuple(data.get("source_ids", ())),
        undetermined_aspects=tuple(data.get("undetermined_aspects", ())),
    )


# =============================================================================
# Fragment Extraction Functions (pure)
# =============================================================================

def extract_from_introspection(
    summary: Optional[Any],
) -> list[tuple[str, str, list[str]]]:
    """
    Extract fragments from introspection trace summary.

    Returns list of (content, source_type_hint, source_ids).
    Reads TraceLog/summary dict via duck typing.
    """
    if summary is None:
        return []

    results: list[tuple[str, str, list[str]]] = []

    # Handle TraceLog object
    if hasattr(summary, "trace_id") and hasattr(summary, "contributing_factors"):
        trace_id = getattr(summary, "trace_id", "")
        ds = getattr(summary, "decision_snapshot", None)

        policy_label = ""
        outcome_type = ""
        if ds is not None:
            policy_label = getattr(ds, "policy_label", "")
            ot = getattr(ds, "outcome_type", None)
            outcome_type = getattr(ot, "value", str(ot)) if ot is not None else ""

        factors = getattr(summary, "contributing_factors", [])
        top_factors = sorted(
            factors,
            key=lambda f: getattr(f, "contribution_strength", 0.0),
            reverse=True,
        )[:3]

        factor_names = [getattr(f, "name", "") for f in top_factors if getattr(f, "name", "")]

        if policy_label or outcome_type:
            content = f"Introspection trace: {policy_label or 'unknown'} ({outcome_type})"
            if factor_names:
                content += f"; possible factors: {', '.join(factor_names)}"
            results.append((content, "introspection_log", [trace_id]))

    # Handle dict (from get_trace_summary or similar)
    elif isinstance(summary, dict):
        policy = summary.get("policy_label", "")
        outcome = summary.get("outcome_type", "")
        trace_id = summary.get("trace_id", "")
        if policy or outcome:
            content = f"Trace observation: {policy} ({outcome})"
            results.append((content, "introspection_log", [trace_id] if trace_id else []))

    return results


def extract_from_narrative(
    state: Optional[Any],
) -> list[tuple[str, str, list[str]]]:
    """
    Extract fragments from self-narrative state.

    Reads NarrativeState via duck typing.
    References: get_narrative_for_introspection() output.
    """
    if state is None:
        return []

    results: list[tuple[str, str, list[str]]] = []

    # Handle NarrativeState object
    fragments = getattr(state, "fragments", ())
    coherence = getattr(state, "coherence", None)
    trend = getattr(state, "trend", None)

    if fragments:
        # Extract from vivid fragments
        vivid = [
            f for f in fragments
            if getattr(f, "vividness", 0.0) >= 0.6
        ]
        for frag in vivid[:3]:
            desc = getattr(frag, "description", "")
            fid = getattr(frag, "fragment_id", "")
            if desc:
                results.append((
                    f"Narrative fragment: {desc}",
                    "self_narrative",
                    [fid] if fid else [],
                ))

    if coherence is not None:
        level = getattr(coherence, "level", None)
        level_val = getattr(level, "value", str(level)) if level is not None else ""
        avg_vividness = getattr(coherence, "average_vividness", 0.0)
        if level_val and level_val != "undefined":
            results.append((
                f"Narrative coherence: {level_val} (avg vividness: {avg_vividness:.2f})",
                "self_narrative",
                [],
            ))

    if trend is not None:
        trend_val = getattr(trend, "value", str(trend))
        if trend_val and trend_val not in ("undefined", "stable"):
            results.append((
                f"Narrative trend: {trend_val}",
                "self_narrative",
                [],
            ))

    # Handle dict input
    if isinstance(state, dict):
        if state.get("has_narrative", False):
            coherence_level = state.get("coherence_level", "")
            trend_val = state.get("trend", "")
            frag_count = state.get("fragment_count", 0)
            if coherence_level:
                results.append((
                    f"Narrative state: {coherence_level}, {frag_count} fragments, trend={trend_val}",
                    "self_narrative",
                    [],
                ))

    return results


def extract_from_coherence(
    state: Optional[Any],
) -> list[tuple[str, str, list[str]]]:
    """
    Extract fragments from identity coherence state.

    Reads IdentityCoherenceState via duck typing.
    References: get_coherence_for_introspection() output.
    """
    if state is None:
        return []

    results: list[tuple[str, str, list[str]]] = []

    # Handle IdentityCoherenceState object
    level = getattr(state, "level", None)
    shift_overlap = getattr(state, "shift_overlap", None)
    trend = getattr(state, "trend", None)
    description = getattr(state, "description", "")

    if level is not None and not isinstance(state, dict):
        level_val = getattr(level, "value", str(level))
        if level_val and level_val != "undefined":
            content = f"Identity coherence: {level_val}"
            if shift_overlap is not None:
                active_count = getattr(shift_overlap, "active_count", 0)
                intensity = getattr(shift_overlap, "intensity", None)
                intensity_val = getattr(intensity, "value", str(intensity)) if intensity is not None else ""
                if active_count > 0:
                    content += f"; {active_count} active shifts ({intensity_val})"
            results.append((content, "identity_coherence", []))

        if trend is not None:
            trend_val = getattr(trend, "value", str(trend))
            if trend_val and trend_val not in ("undefined", "stable"):
                results.append((
                    f"Coherence trend: {trend_val}",
                    "identity_coherence",
                    [],
                ))

    # Handle dict input
    if isinstance(state, dict):
        level_val = state.get("level", "")
        is_coherent = state.get("is_coherent", None)
        active_shifts = state.get("active_shift_count", 0)
        desc = state.get("description", "")
        if level_val:
            content = f"Coherence observation: {level_val}"
            if active_shifts > 0:
                content += f"; {active_shifts} shifts"
            results.append((content, "identity_coherence", []))

    return results


def extract_from_tendency(
    awareness: Optional[Any],
) -> list[tuple[str, str, list[str]]]:
    """
    Extract fragments from tendency awareness.

    Reads TendencyAwareness via duck typing.
    References: get_awareness_for_introspection() output.
    """
    if awareness is None:
        return []

    results: list[tuple[str, str, list[str]]] = []

    # Handle TendencyAwareness object
    if hasattr(awareness, "items") and hasattr(awareness, "has_awareness"):
        if not getattr(awareness, "has_awareness", False):
            return []

        items = getattr(awareness, "items", [])
        overall_strength = getattr(awareness, "overall_strength", None)
        overall_val = getattr(overall_strength, "value", str(overall_strength)) if overall_strength else ""

        for item in items[:3]:
            desc = getattr(item, "description", "")
            if desc:
                results.append((
                    f"Tendency awareness: {desc}",
                    "tendency_awareness",
                    [],
                ))

        if overall_val and overall_val != "none":
            results.append((
                f"Overall tendency strength: {overall_val}",
                "tendency_awareness",
                [],
            ))

    # Handle dict input
    elif isinstance(awareness, dict):
        if awareness.get("has_tendency_awareness", False):
            descriptions = awareness.get("descriptions", [])
            overall = awareness.get("overall_strength", "")
            for desc in descriptions[:3]:
                results.append((
                    f"Tendency: {desc}",
                    "tendency_awareness",
                    [],
                ))
            if overall:
                results.append((
                    f"Overall tendency: {overall}",
                    "tendency_awareness",
                    [],
                ))

    return results


def extract_from_episodic(
    store: Optional[Any],
) -> list[tuple[str, str, list[str]]]:
    """
    Extract fragments from episodic memory store.

    Reads EpisodeStore via duck typing.
    Extracts from get_fresh_episodes().
    """
    if store is None:
        return []

    results: list[tuple[str, str, list[str]]] = []

    # Handle EpisodeStore object
    if hasattr(store, "get_fresh_episodes"):
        fresh = store.get_fresh_episodes()
        for ep in fresh[:3]:
            summary_text = getattr(ep, "summary", "")
            ep_id = getattr(ep, "episode_id", "")
            ep_type = getattr(ep, "episode_type", None)
            ep_type_val = getattr(ep_type, "value", str(ep_type)) if ep_type is not None else ""
            if summary_text:
                content = f"Episode ({ep_type_val}): {summary_text[:100]}"
                results.append((content, "episodic_memory", [ep_id] if ep_id else []))

        # Overall stats
        avg_vividness = getattr(store, "average_vividness", 0.0)
        active_count = getattr(store, "active_episode_count", 0)
        if active_count > 0:
            results.append((
                f"Episodic memory: {active_count} active episodes, avg vividness {avg_vividness:.2f}",
                "episodic_memory",
                [],
            ))

    # Handle dict input
    elif isinstance(store, dict):
        if store.get("has_episodes", False):
            total = store.get("total_episodes", 0)
            active = store.get("active_episode_count", 0)
            avg_v = store.get("average_vividness", 0.0)
            results.append((
                f"Episodic memory state: {active}/{total} active, vividness {avg_v:.2f}",
                "episodic_memory",
                [],
            ))

    return results


# =============================================================================
# Bundle Generation Functions (pure)
# =============================================================================

def compute_fragment_relevance(
    a: IntrospectionFragment,
    b: IntrospectionFragment,
) -> float:
    """
    Compute relevance between two fragments.

    Based on content similarity, source type, and temporal proximity.
    """
    score = 0.0

    # Source type similarity
    if a.source_type == b.source_type:
        score += 0.3

    # Temporal proximity (within 5 minutes = high relevance)
    time_diff = abs(a.timestamp - b.timestamp)
    if time_diff < 300.0:
        score += 0.3 * (1.0 - time_diff / 300.0)

    # Content word overlap (simple Jaccard on words)
    words_a = set(a.content.lower().split())
    words_b = set(b.content.lower().split())
    if words_a and words_b:
        union = words_a | words_b
        intersection = words_a & words_b
        if union:
            score += 0.4 * (len(intersection) / len(union))

    return min(1.0, score)


def determine_bundle_coherence(
    fragment_ids: tuple[str, ...],
    fragments: dict[str, IntrospectionFragment],
) -> BundleCoherence:
    """Determine coherence level of a bundle."""
    if len(fragment_ids) < 2:
        return BundleCoherence.UNDEFINED

    relevances: list[float] = []
    fids = list(fragment_ids)
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            fa = fragments.get(fids[i])
            fb = fragments.get(fids[j])
            if fa and fb:
                relevances.append(compute_fragment_relevance(fa, fb))

    if not relevances:
        return BundleCoherence.UNDEFINED

    avg = sum(relevances) / len(relevances)
    if avg >= 0.6:
        return BundleCoherence.TIGHT
    elif avg >= 0.3:
        return BundleCoherence.LOOSE
    else:
        return BundleCoherence.SCATTERED


def generate_bundles(
    fragments: list[IntrospectionFragment],
    config: ConsumptionLayerConfig,
) -> list[FragmentBundle]:
    """
    Generate bundles from fragments based on relevance.

    束ね方は固定しない。再要約・再リンクを許容する。
    """
    if len(fragments) < 2:
        return []

    current_time = time.time()
    bundles: list[FragmentBundle] = []
    used: set[str] = set()
    frag_map = {f.fragment_id: f for f in fragments}

    # Group by source type first
    by_source: dict[str, list[IntrospectionFragment]] = {}
    for f in fragments:
        by_source.setdefault(f.source_type.value, []).append(f)

    for source_type, group in by_source.items():
        if len(group) < 2:
            continue

        # Find pairs with high relevance
        cluster_ids: list[str] = []
        for i in range(len(group)):
            if group[i].fragment_id in used:
                continue
            for j in range(i + 1, len(group)):
                if group[j].fragment_id in used:
                    continue
                rel = compute_fragment_relevance(group[i], group[j])
                if rel >= config.bundle_strength_threshold:
                    if group[i].fragment_id not in cluster_ids:
                        cluster_ids.append(group[i].fragment_id)
                    if group[j].fragment_id not in cluster_ids:
                        cluster_ids.append(group[j].fragment_id)

        if len(cluster_ids) >= 2:
            fids = tuple(cluster_ids)
            coherence = determine_bundle_coherence(fids, frag_map)

            # Generate theme description
            contents = [frag_map[fid].content[:40] for fid in fids if fid in frag_map]
            theme = f"Bundle of {len(fids)} {source_type} fragments"

            avg_strength = 0.0
            count = 0
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    fa = frag_map.get(cluster_ids[i])
                    fb = frag_map.get(cluster_ids[j])
                    if fa and fb:
                        avg_strength += compute_fragment_relevance(fa, fb)
                        count += 1
            if count > 0:
                avg_strength /= count

            bundles.append(FragmentBundle(
                bundle_id=_generate_id(),
                fragment_ids=fids,
                theme_description=theme,
                coherence=coherence,
                timestamp=current_time,
                strength=round(avg_strength, 4),
            ))
            used.update(cluster_ids)

            if len(bundles) >= config.max_bundles:
                break

    # Cross-source bundles (fragments from different sources but related)
    if len(bundles) < config.max_bundles:
        remaining = [f for f in fragments if f.fragment_id not in used]
        if len(remaining) >= 2:
            cross_ids: list[str] = []
            for i in range(min(len(remaining), 20)):
                for j in range(i + 1, min(len(remaining), 20)):
                    rel = compute_fragment_relevance(remaining[i], remaining[j])
                    if rel >= config.bundle_strength_threshold:
                        if remaining[i].fragment_id not in cross_ids:
                            cross_ids.append(remaining[i].fragment_id)
                        if remaining[j].fragment_id not in cross_ids:
                            cross_ids.append(remaining[j].fragment_id)

            if len(cross_ids) >= 2:
                fids = tuple(cross_ids[:10])
                coherence = determine_bundle_coherence(fids, frag_map)
                bundles.append(FragmentBundle(
                    bundle_id=_generate_id(),
                    fragment_ids=fids,
                    theme_description=f"Cross-source bundle of {len(fids)} fragments",
                    coherence=coherence,
                    timestamp=current_time,
                    strength=round(config.bundle_strength_threshold, 4),
                ))

    return bundles


# =============================================================================
# System Class: IntrospectionConsumptionSystem
# =============================================================================

class IntrospectionConsumptionSystem:
    """
    Introspection Consumption Layer (内省の消費層)

    内省観測の結果を「読み取り可能な断片」として再編成し、
    自己に関する叙述の素材として循環させる。

    CRITICAL CONSTRAINTS:
    - 判断・目的・価値・責任に一切接続しない
    - 内省結果を正解化・評価化しない
    - 人格の方向性を固定しない
    - 断片の束ね方は固定しない
    - 出力は観測素材の提供に限定
    """

    def __init__(self, config: Optional[ConsumptionLayerConfig] = None):
        self._config = config or ConsumptionLayerConfig()
        self._fragments: list[IntrospectionFragment] = []
        self._bundles: list[FragmentBundle] = []
        self._consumption_history: list[ConsumptionRecord] = []
        self._total_created: int = 0
        self._total_consumptions: int = 0
        self._last_store: Optional[ConsumptionStore] = None

    def consume_observations(
        self,
        introspection_summary: Optional[Any] = None,
        narrative_state: Optional[Any] = None,
        coherence_state: Optional[Any] = None,
        tendency_awareness: Optional[Any] = None,
        episodic_store: Optional[Any] = None,
    ) -> ConsumptionStore:
        """
        Consume observations from various introspection sources.

        各入力をduck typingで読み取り、断片に変換。
        既存断片との関連でバンドルを生成/更新。
        自然減衰を適用。
        """
        current_time = time.time()

        # Step 1: Extract fragments from each source
        raw_extractions: list[tuple[str, str, list[str]]] = []
        raw_extractions.extend(extract_from_introspection(introspection_summary))
        raw_extractions.extend(extract_from_narrative(narrative_state))
        raw_extractions.extend(extract_from_coherence(coherence_state))
        raw_extractions.extend(extract_from_tendency(tendency_awareness))
        raw_extractions.extend(extract_from_episodic(episodic_store))

        # Step 2: Create IntrospectionFragment objects
        source_type_map = {
            "introspection_log": FragmentSourceType.INTROSPECTION_LOG,
            "self_narrative": FragmentSourceType.SELF_NARRATIVE,
            "identity_coherence": FragmentSourceType.IDENTITY_COHERENCE,
            "tendency_awareness": FragmentSourceType.TENDENCY_AWARENESS,
            "episodic_memory": FragmentSourceType.EPISODIC_MEMORY,
        }

        for content, source_hint, source_ids in raw_extractions:
            source_type = source_type_map.get(source_hint, FragmentSourceType.MIXED)
            fragment = IntrospectionFragment(
                fragment_id=_generate_id(),
                source_type=source_type,
                content=content,
                timestamp=current_time,
                freshness=1.0,
                reference_count=0,
                source_ids=tuple(source_ids),
                undetermined_aspects=(),
            )
            self._fragments.append(fragment)
            self._total_created += 1

        # Step 3: Apply decay to existing fragments
        self._decay_fragments()

        # Step 4: Remove faded fragments
        self._remove_faded()

        # Step 5: Enforce fragment limit
        self._enforce_limit()

        # Step 6: Generate/update bundles
        self._bundles = generate_bundles(self._fragments, self._config)

        # Build and return store
        return self._build_store(current_time)

    def decay_fragments(self) -> ConsumptionStore:
        """
        Apply natural decay to all fragments.

        参照頻度による減衰変調。
        """
        current_time = time.time()
        self._decay_fragments()
        self._remove_faded()
        return self._build_store(current_time)

    def rebundle(self) -> ConsumptionStore:
        """
        Rebundle fragments (束ね直し - 固定しない).

        断片の再編成。以前の束ね方と異なる結果が出ることを許す。
        """
        current_time = time.time()
        self._bundles = generate_bundles(self._fragments, self._config)
        return self._build_store(current_time)

    def reference_fragment(self, fragment_id: str) -> None:
        """
        Reference a fragment (参照で鮮度回復).
        """
        for i, f in enumerate(self._fragments):
            if f.fragment_id == fragment_id:
                referenced = f.with_reference()
                boosted = referenced.with_freshness(
                    referenced.freshness + self._config.freshness_boost_on_reference,
                )
                self._fragments[i] = boosted
                return

    def mark_as_consumed(
        self,
        fragment_ids: list[str],
        context: str = "",
    ) -> None:
        """
        Mark fragments as consumed (読まれた記録).
        """
        current_time = time.time()
        record = ConsumptionRecord(
            record_id=_generate_id(),
            fragment_ids=tuple(fragment_ids),
            timestamp=current_time,
            context_description=context,
        )
        self._consumption_history.append(record)
        self._total_consumptions += 1

        # Reference each consumed fragment
        for fid in fragment_ids:
            self.reference_fragment(fid)

        # Trim consumption history
        if len(self._consumption_history) > self._config.max_consumption_history:
            self._consumption_history = self._consumption_history[
                -self._config.max_consumption_history:
            ]

    def get_readable_fragments(
        self,
        max_count: int = 10,
    ) -> list[IntrospectionFragment]:
        """
        Get readable fragments sorted by freshness (descending).
        """
        active = [f for f in self._fragments if f.freshness > self._config.stale_threshold]
        active.sort(key=lambda f: f.freshness, reverse=True)
        return active[:max_count]

    def get_store(self) -> ConsumptionStore:
        """Get current store snapshot."""
        return self._build_store(time.time())

    def get_last_store(self) -> Optional[ConsumptionStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _decay_fragments(self) -> None:
        """Apply natural decay to all fragments."""
        new_fragments: list[IntrospectionFragment] = []
        for f in self._fragments:
            # Reference count modulates decay (more refs = slower decay)
            ref_modifier = max(0.3, 1.0 - (f.reference_count * 0.1))
            effective_rate = self._config.base_decay_rate * ref_modifier
            new_freshness = f.freshness - effective_rate
            new_fragments.append(f.with_freshness(new_freshness))
        self._fragments = new_fragments

    def _remove_faded(self) -> None:
        """Remove fragments that have fully faded."""
        self._fragments = [f for f in self._fragments if f.freshness > 0.0]

        # Clean up bundles referencing removed fragments
        remaining_ids = {f.fragment_id for f in self._fragments}
        self._bundles = [
            b for b in self._bundles
            if any(fid in remaining_ids for fid in b.fragment_ids)
        ]

    def _enforce_limit(self) -> None:
        """Enforce maximum fragment count."""
        if len(self._fragments) <= self._config.max_fragments:
            return

        self._fragments.sort(key=lambda f: f.freshness, reverse=True)
        self._fragments = self._fragments[:self._config.max_fragments]

    def _build_store(self, current_time: float) -> ConsumptionStore:
        """Build a ConsumptionStore snapshot."""
        active = [f for f in self._fragments if f.freshness > 0.0]
        avg_freshness = (
            sum(f.freshness for f in self._fragments) / len(self._fragments)
            if self._fragments else 0.0
        )

        description = _generate_store_description(
            len(self._fragments),
            len(active),
            len(self._bundles),
            avg_freshness,
            self._total_created,
            self._total_consumptions,
        )

        store = ConsumptionStore(
            fragments=tuple(self._fragments),
            bundles=tuple(self._bundles),
            consumption_history=tuple(self._consumption_history),
            total_fragments_created=self._total_created,
            total_consumptions=self._total_consumptions,
            average_freshness=round(avg_freshness, 4),
            active_fragment_count=len(active),
            bundle_count=len(self._bundles),
            timestamp=current_time,
            description=description,
        )
        self._last_store = store
        return store


def _generate_store_description(
    total: int,
    active: int,
    bundle_count: int,
    avg_freshness: float,
    total_created: int,
    total_consumptions: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No introspection fragments yet."

    parts = [f"{active} active fragments out of {total} total"]

    if bundle_count > 0:
        parts.append(f"{bundle_count} bundles")

    if total_consumptions > 0:
        parts.append(f"{total_consumptions} consumption events")

    parts.append(f"average freshness: {avg_freshness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def consume_from_chain(
    system: IntrospectionConsumptionSystem,
    introspection_summary: Optional[Any] = None,
    narrative_state: Optional[Any] = None,
    coherence_state: Optional[Any] = None,
    tendency_awareness: Optional[Any] = None,
    episodic_store: Optional[Any] = None,
) -> ConsumptionStore:
    """
    Consume from the self-observation chain.

    自己観測チェーンからの統合ヘルパー。
    各入力は読み取り専用で参照される。
    """
    return system.consume_observations(
        introspection_summary=introspection_summary,
        narrative_state=narrative_state,
        coherence_state=coherence_state,
        tendency_awareness=tendency_awareness,
        episodic_store=episodic_store,
    )


def generate_consumption_tags(
    store: ConsumptionStore,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from ConsumptionStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags: list[dict[str, Any]] = []

    if not store.has_fragments():
        tags.append({
            "category": "INTROSPECTION_CONSUMPTION",
            "label": "no_fragments",
            "description": "No introspection fragments available",
            "weight": 0.05 * scale,
        })
        return tags

    # Fragment count tag
    tags.append({
        "category": "INTROSPECTION_CONSUMPTION_COUNT",
        "label": f"fragments_{store.active_fragment_count}",
        "description": (
            f"Consumption layer holds {store.active_fragment_count} active fragments"
        ),
        "weight": 0.06 * scale,
    })

    # Freshness tag
    if store.average_freshness >= 0.7:
        freshness_label = "fresh_observations"
    elif store.average_freshness >= 0.4:
        freshness_label = "moderate_observations"
    else:
        freshness_label = "aging_observations"
    tags.append({
        "category": "INTROSPECTION_CONSUMPTION_FRESHNESS",
        "label": freshness_label,
        "description": (
            f"Average fragment freshness: {store.average_freshness:.2f}"
        ),
        "weight": 0.07 * scale,
    })

    # Bundle tag
    if store.bundle_count > 0:
        tags.append({
            "category": "INTROSPECTION_CONSUMPTION_BUNDLES",
            "label": f"bundles_{store.bundle_count}",
            "description": (
                f"{store.bundle_count} fragment bundles formed"
            ),
            "weight": 0.04 * scale,
        })

    # Most recent fresh fragment
    fresh = store.get_fresh_fragments()
    if fresh:
        most_recent = fresh[-1]
        tags.append({
            "category": "INTROSPECTION_CONSUMPTION_RECENT",
            "label": f"recent_{most_recent.source_type.value}",
            "description": most_recent.content[:100],
            "weight": 0.08 * scale,
        })

    # Consumption activity
    if store.total_consumptions > 0:
        tags.append({
            "category": "INTROSPECTION_CONSUMPTION_ACTIVITY",
            "label": f"consumed_{store.total_consumptions}",
            "description": f"{store.total_consumptions} readings performed",
            "weight": 0.05 * scale,
        })

    # Integrated description
    tags.append({
        "category": "INTROSPECTION_CONSUMPTION_INTEGRATED",
        "label": "consumption_awareness",
        "description": store.description,
        "weight": 0.1 * scale,
    })

    return tags


def get_consumption_summary(store: ConsumptionStore) -> str:
    """Get human-readable summary. For introspection/logging only."""
    lines = [
        "=== Introspection Consumption Layer ===",
        f"Total fragments: {len(store.fragments)}",
        f"Active fragments: {store.active_fragment_count}",
        f"Bundles: {store.bundle_count}",
        f"Total created: {store.total_fragments_created}",
        f"Total consumptions: {store.total_consumptions}",
        f"Average freshness: {store.average_freshness:.2f}",
        "",
    ]

    fresh = store.get_fresh_fragments()
    if fresh:
        lines.append("Recent fragments:")
        for f in fresh[-5:]:
            lines.append(
                f"  [{f.source_type.value}] {f.content[:80]}"
                f" (freshness: {f.get_freshness_level().value})"
            )
        lines.append("")

    if store.bundles:
        lines.append("Bundles:")
        for b in store.bundles[:5]:
            lines.append(
                f"  [{b.coherence.value}] {b.theme_description}"
                f" ({len(b.fragment_ids)} fragments, strength: {b.strength:.2f})"
            )
        lines.append("")

    lines.append(f"Integrated: {store.description}")
    return "\n".join(lines)


def get_consumption_for_introspection(
    store: ConsumptionStore,
) -> dict[str, Any]:
    """
    Get structured consumption data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    source_counts: dict[str, int] = {}
    for f in store.fragments:
        source_counts[f.source_type.value] = (
            source_counts.get(f.source_type.value, 0) + 1
        )

    freshness_distribution: dict[str, int] = {}
    for f in store.fragments:
        level = f.get_freshness_level().value
        freshness_distribution[level] = (
            freshness_distribution.get(level, 0) + 1
        )

    coherence_distribution: dict[str, int] = {}
    for b in store.bundles:
        coherence_distribution[b.coherence.value] = (
            coherence_distribution.get(b.coherence.value, 0) + 1
        )

    return {
        "has_fragments": store.has_fragments(),
        "total_fragments": len(store.fragments),
        "active_fragment_count": store.active_fragment_count,
        "bundle_count": store.bundle_count,
        "total_created": store.total_fragments_created,
        "total_consumptions": store.total_consumptions,
        "average_freshness": store.average_freshness,
        "source_distribution": source_counts,
        "freshness_distribution": freshness_distribution,
        "bundle_coherence_distribution": coherence_distribution,
        "consumption_history_count": len(store.consumption_history),
        "description": store.description,
        "timestamp": store.timestamp,
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: ConsumptionStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    消費層は観測素材の提供のみ。
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
                "total_fragments_created", "total_consumptions",
                "average_freshness", "active_fragment_count",
                "bundle_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: IntrospectionConsumptionSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    消費層から目標を生成しない。
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


def verify_read_only_principle(system: IntrospectionConsumptionSystem) -> bool:
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


def verify_no_value_modification(system: IntrospectionConsumptionSystem) -> bool:
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
    max_fragments: int = 150,
    base_decay_rate: float = 0.03,
    bundle_strength_threshold: float = 0.3,
    max_bundles: int = 30,
    freshness_boost_on_reference: float = 0.12,
    stale_threshold: float = 0.15,
    max_consumption_history: int = 50,
) -> ConsumptionLayerConfig:
    """Create a custom configuration."""
    return ConsumptionLayerConfig(
        max_fragments=max_fragments,
        base_decay_rate=base_decay_rate,
        bundle_strength_threshold=bundle_strength_threshold,
        max_bundles=max_bundles,
        freshness_boost_on_reference=freshness_boost_on_reference,
        stale_threshold=stale_threshold,
        max_consumption_history=max_consumption_history,
    )


def create_empty_store() -> ConsumptionStore:
    """Create an empty consumption store."""
    return ConsumptionStore.empty()


def create_system(
    config: Optional[ConsumptionLayerConfig] = None,
) -> IntrospectionConsumptionSystem:
    """Create a new IntrospectionConsumptionSystem."""
    return IntrospectionConsumptionSystem(config=config)


def save_consumption_state(
    store: ConsumptionStore,
    filepath: str,
) -> None:
    """Save consumption state to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_consumption_state(filepath: str) -> ConsumptionStore:
    """Load consumption state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ConsumptionStore.from_dict(data)
