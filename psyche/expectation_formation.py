"""
Expectation Formation (予期・期待の形成)

過去の反復・差分・物語から「次に起きうる展開の仮の見通し」を
弱く生成するためのモジュール。

現状の課題:
時間的連続性は過去方向のみ（temporal_self_difference, self_narrative）で、
未来方向の投射が構造として存在しない。
予期・期待を形成することで連続性を未来方向にも延長する。

CRITICAL DESIGN PRINCIPLES:
- 入力は全て Optional[Any] + duck typing（循環import回避）
- 判断・目的・価値・責任に一切接続しない
- 予期を正解化・評価化しない（正誤・成功失敗を判定しない）
- 人格の方向性を固定しない
- 予期は仮説として保持し固定しない（修正・撤回・分割が可能）
- 予期同士の競合を許容する
- 規範的な方向付けを行わない
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional
import json
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class ExpectationSourceType(Enum):
    """
    予期の根拠がどの入力源由来か。

    評価的ではなく、記述的な分類。
    """
    REPETITION = "repetition"            # 反復傾向由来
    SELF_DIFFERENCE = "self_difference"  # 自己差分由来
    NARRATIVE = "narrative"              # 自己物語由来
    MIXED = "mixed"                      # 複数ソース混合


class ExpectationBasis(Enum):
    """
    予期の推論根拠の種類。

    反復パターンの継続、変化方向の延長、文脈の持続。
    """
    PATTERN_CONTINUATION = "pattern_continuation"  # 反復パターンの継続
    CHANGE_DIRECTION = "change_direction"          # 変化方向の延長
    CONTEXT_PERSISTENCE = "context_persistence"    # 文脈の持続
    COMBINED = "combined"                          # 複合的な根拠
    UNDEFINED = "undefined"                        # 未確定


class ExpectationStrength(Enum):
    """
    予期の確からしさの抽象レベル。

    評価ではなく、予期候補の安定度の記述。
    """
    STRONG = "strong"          # 強い予期
    MODERATE = "moderate"      # 中程度
    WEAK = "weak"              # 弱い予期
    FAINT = "faint"            # かすかな予期
    UNDEFINED = "undefined"    # 未確定


class ExpectationFreshness(Enum):
    """
    予期の新鮮度レベル。

    他モジュールと同じパターン。
    """
    FRESH = "fresh"        # 直近
    RECENT = "recent"      # まだ活発
    AGING = "aging"        # 薄れつつある
    STALE = "stale"        # ほぼ消えかけ
    FADED = "faded"        # 消失寸前


# =============================================================================
# ID Generation
# =============================================================================

def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# Freshness / Strength Level Determination (Pure)
# =============================================================================

def determine_freshness_level(freshness: float) -> ExpectationFreshness:
    """Determine ExpectationFreshness from a numeric freshness value (0.0-1.0)."""
    if freshness >= 0.8:
        return ExpectationFreshness.FRESH
    elif freshness >= 0.6:
        return ExpectationFreshness.RECENT
    elif freshness >= 0.4:
        return ExpectationFreshness.AGING
    elif freshness >= 0.15:
        return ExpectationFreshness.STALE
    else:
        return ExpectationFreshness.FADED


def determine_strength_level(strength: float) -> ExpectationStrength:
    """Determine ExpectationStrength from a numeric strength value (0.0-1.0)."""
    if strength >= 0.7:
        return ExpectationStrength.STRONG
    elif strength >= 0.4:
        return ExpectationStrength.MODERATE
    elif strength >= 0.2:
        return ExpectationStrength.WEAK
    elif strength >= 0.05:
        return ExpectationStrength.FAINT
    else:
        return ExpectationStrength.UNDEFINED


# =============================================================================
# Core Dataclasses (all frozen)
# =============================================================================

@dataclass(frozen=True)
class EvidenceLink:
    """
    予期と根拠の弱い接続。

    予期候補がどの入力源からどの程度の寄与を受けたかを記録する。
    NOT evaluative. NOT prescriptive.
    """
    link_id: str
    expectation_id: str
    source_type: ExpectationSourceType
    source_description: str
    contribution: float  # 0.0〜1.0


@dataclass(frozen=True)
class ExpectationCandidate:
    """
    弱い予期仮説（コア構造）。

    「次に起きうる展開の仮の見通し」を保持する不変構造。
    仮説として保持し固定しない。修正・撤回が可能。
    NOT evaluative. NOT prescriptive.
    """
    expectation_id: str
    source_type: ExpectationSourceType
    basis: ExpectationBasis
    description: str
    timestamp: str
    freshness: float                       # 0.0〜1.0
    strength: float                        # 0.0〜1.0
    reference_count: int
    evidence_ids: tuple[str, ...]          # 根拠リンクID群
    competing_ids: tuple[str, ...]         # 競合する予期のID群
    revision_count: int                    # 修正回数
    undetermined_aspects: tuple[str, ...]  # 未確定の側面

    def get_freshness_level(self) -> ExpectationFreshness:
        """Get abstract freshness level."""
        return determine_freshness_level(self.freshness)

    def get_strength_level(self) -> ExpectationStrength:
        """Get abstract strength level."""
        return determine_strength_level(self.strength)

    def with_freshness(self, new_freshness: float) -> ExpectationCandidate:
        """Create a copy with updated freshness."""
        return replace(self, freshness=max(0.0, min(1.0, new_freshness)))

    def with_strength(self, new_strength: float) -> ExpectationCandidate:
        """Create a copy with updated strength."""
        return replace(self, strength=max(0.0, min(1.0, new_strength)))

    def with_reference(self) -> ExpectationCandidate:
        """Create a copy with incremented reference count."""
        return replace(self, reference_count=self.reference_count + 1)

    def revise(self, new_description: str) -> ExpectationCandidate:
        """Create a revised copy (予期は固定しない、修正可能)."""
        return replace(
            self,
            description=new_description,
            revision_count=self.revision_count + 1,
        )

    def with_competing(self, competing_id: str) -> ExpectationCandidate:
        """Create a copy with an additional competing expectation ID."""
        if competing_id in self.competing_ids:
            return self
        return replace(
            self,
            competing_ids=self.competing_ids + (competing_id,),
        )


@dataclass(frozen=True)
class ExpectationStore:
    """
    不変スナップショット - 予期形成の全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    expectations: tuple[ExpectationCandidate, ...]
    evidence_links: tuple[EvidenceLink, ...]
    total_expectations_created: int
    total_revisions: int
    total_expirations: int
    average_freshness: float
    average_strength: float
    active_expectation_count: int
    competing_pair_count: int
    timestamp: str
    description: str

    def has_expectations(self) -> bool:
        return len(self.expectations) > 0

    def get_active_expectations(self, stale_threshold: float = 0.15) -> tuple[ExpectationCandidate, ...]:
        """Get expectations with freshness above the stale threshold."""
        return tuple(
            e for e in self.expectations
            if e.freshness > stale_threshold
        )

    def get_strong_expectations(self) -> tuple[ExpectationCandidate, ...]:
        """Get expectations with strength above 0.5."""
        return tuple(
            e for e in self.expectations
            if e.strength > 0.5
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expectations": [
                _candidate_to_dict(e) for e in self.expectations
            ],
            "evidence_links": [
                {
                    "link_id": el.link_id,
                    "expectation_id": el.expectation_id,
                    "source_type": el.source_type.value,
                    "source_description": el.source_description,
                    "contribution": el.contribution,
                }
                for el in self.evidence_links
            ],
            "total_expectations_created": self.total_expectations_created,
            "total_revisions": self.total_revisions,
            "total_expirations": self.total_expirations,
            "average_freshness": self.average_freshness,
            "average_strength": self.average_strength,
            "active_expectation_count": self.active_expectation_count,
            "competing_pair_count": self.competing_pair_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectationStore:
        """Create from dictionary."""
        expectations = tuple(
            _candidate_from_dict(e) for e in data.get("expectations", [])
        )
        evidence_links = tuple(
            EvidenceLink(
                link_id=el["link_id"],
                expectation_id=el.get("expectation_id", ""),
                source_type=ExpectationSourceType(el.get("source_type", "mixed")),
                source_description=el.get("source_description", ""),
                contribution=el.get("contribution", 0.0),
            )
            for el in data.get("evidence_links", [])
        )
        return cls(
            expectations=expectations,
            evidence_links=evidence_links,
            total_expectations_created=data.get("total_expectations_created", 0),
            total_revisions=data.get("total_revisions", 0),
            total_expirations=data.get("total_expirations", 0),
            average_freshness=data.get("average_freshness", 0.0),
            average_strength=data.get("average_strength", 0.0),
            active_expectation_count=data.get("active_expectation_count", 0),
            competing_pair_count=data.get("competing_pair_count", 0),
            timestamp=data.get("timestamp", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class ExpectationFormationConfig:
    """
    Configuration for the expectation formation system.

    These parameters control expectation lifecycle.
    They do NOT affect decisions.
    """
    max_expectations: int = 80
    base_decay_rate: float = 0.04
    strength_decay_rate: float = 0.02
    freshness_boost_on_reference: float = 0.10
    stale_threshold: float = 0.15
    min_strength_for_retention: float = 0.05
    competition_overlap_threshold: float = 0.4
    max_evidence_per_expectation: int = 10
    max_competing_per_expectation: int = 5


# =============================================================================
# Serialization Helpers
# =============================================================================

def _candidate_to_dict(candidate: ExpectationCandidate) -> dict[str, Any]:
    """Convert an ExpectationCandidate to a dictionary."""
    return {
        "expectation_id": candidate.expectation_id,
        "source_type": candidate.source_type.value,
        "basis": candidate.basis.value,
        "description": candidate.description,
        "timestamp": candidate.timestamp,
        "freshness": candidate.freshness,
        "strength": candidate.strength,
        "reference_count": candidate.reference_count,
        "evidence_ids": list(candidate.evidence_ids),
        "competing_ids": list(candidate.competing_ids),
        "revision_count": candidate.revision_count,
        "undetermined_aspects": list(candidate.undetermined_aspects),
    }


def _candidate_from_dict(data: dict[str, Any]) -> ExpectationCandidate:
    """Create an ExpectationCandidate from a dictionary."""
    return ExpectationCandidate(
        expectation_id=data["expectation_id"],
        source_type=ExpectationSourceType(data.get("source_type", "mixed")),
        basis=ExpectationBasis(data.get("basis", "undefined")),
        description=data.get("description", ""),
        timestamp=data.get("timestamp", ""),
        freshness=data.get("freshness", 0.0),
        strength=data.get("strength", 0.0),
        reference_count=data.get("reference_count", 0),
        evidence_ids=tuple(data.get("evidence_ids", ())),
        competing_ids=tuple(data.get("competing_ids", ())),
        revision_count=data.get("revision_count", 0),
        undetermined_aspects=tuple(data.get("undetermined_aspects", ())),
    )


# =============================================================================
# Extraction Functions (Pure, Duck Typing)
# =============================================================================

def extract_from_tendency(
    bias: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract expectation materials from tendency bias.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    Reads TendencyBias via duck typing.
    """
    if bias is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Handle TendencyBias object (has_bias, biases, strongest_category, strongest_bias)
    if hasattr(bias, "has_bias") and hasattr(bias, "biases"):
        has_bias = getattr(bias, "has_bias", False)
        if not has_bias:
            return []

        biases = getattr(bias, "biases", {})
        strongest_category = getattr(bias, "strongest_category", "")
        strongest_bias = getattr(bias, "strongest_bias", None)

        if strongest_bias is not None:
            # Tendency object: strength, confidence, pattern.category, total_reinforcements
            strength = getattr(strongest_bias, "strength", 0.0)
            confidence = getattr(strongest_bias, "confidence", 0.0)
            pattern = getattr(strongest_bias, "pattern", None)
            category = ""
            if pattern is not None:
                category = getattr(pattern, "category", "")
            total_r = getattr(strongest_bias, "total_reinforcements", 0)

            if strength >= 0.1:
                desc = f"Tendency toward '{category or strongest_category}' is strengthening"
                if total_r > 3:
                    desc += f" (reinforced {total_r} times)"
                desc += " — this behavioral pattern may continue"
                strength_hint = min(1.0, strength * 0.7 + confidence * 0.3)
                evidence = [f"Tendency '{category or strongest_category}': strength={strength:.2f}"]
                results.append((desc, "pattern_continuation", strength_hint, evidence))

        # Additional weaker tendencies from biases dict
        for cat_name, tendency in (biases.items() if isinstance(biases, dict) else []):
            t_strength = getattr(tendency, "strength", 0.0)
            if t_strength >= 0.2 and cat_name != strongest_category:
                desc = f"Secondary tendency toward '{cat_name}' may also persist"
                evidence = [f"Secondary tendency '{cat_name}': strength={t_strength:.2f}"]
                results.append((desc, "pattern_continuation", t_strength * 0.5, evidence))

    # Handle dict input
    elif isinstance(bias, dict):
        tendency_count = bias.get("tendency_count", 0)
        strongest_category = bias.get("strongest_category", "")
        strongest_bias_val = bias.get("strongest_bias", 0.0)
        active_tendencies = bias.get("active_tendencies", [])

        if tendency_count > 0 and strongest_category:
            strength_hint = min(1.0, strongest_bias_val * 0.7) if isinstance(strongest_bias_val, (int, float)) else 0.3
            desc = f"Tendency toward '{strongest_category}' may continue"
            evidence = [f"Active tendencies: {tendency_count}, strongest: {strongest_category}"]
            results.append((desc, "pattern_continuation", strength_hint, evidence))

    return results


def extract_from_difference(
    summary: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract expectation materials from self-difference summary.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    Reads SelfDifferenceSummary via duck typing.
    """
    if summary is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Handle SelfDifferenceSummary object
    if hasattr(summary, "has_difference") and hasattr(summary, "magnitude"):
        has_difference = getattr(summary, "has_difference", False)
        if not has_difference:
            return []

        magnitude = getattr(summary, "magnitude", None)
        nature = getattr(summary, "nature", None)
        temporal_span = getattr(summary, "temporal_span", None)
        integrated_description = getattr(summary, "integrated_description", "")

        mag_val = getattr(magnitude, "value", str(magnitude)) if magnitude is not None else "none"
        nature_val = getattr(nature, "value", str(nature)) if nature is not None else "stable"
        span_val = getattr(temporal_span, "value", str(temporal_span)) if temporal_span is not None else ""

        # Magnitude → strength mapping
        mag_strength = {
            "none": 0.0, "minimal": 0.15,
            "noticeable": 0.35, "significant": 0.55,
            "substantial": 0.75,
        }
        strength_hint = mag_strength.get(mag_val, 0.2)

        # Nature → basis/description mapping
        nature_map = {
            "shifting": ("change_direction", "The current shift in self-state may continue in this direction"),
            "returning": ("change_direction", "The state appears to be returning toward a previous pattern"),
            "stable": ("context_persistence", "The current stable state is likely to persist"),
            "fluctuating": ("change_direction", "Fluctuations in self-state may continue"),
        }

        basis_hint, base_desc = nature_map.get(
            nature_val,
            ("change_direction", "Self-state change direction may continue"),
        )

        if integrated_description:
            desc = f"{base_desc} — {integrated_description[:100]}"
        else:
            desc = base_desc

        evidence = [f"Self-difference: magnitude={mag_val}, nature={nature_val}"]
        if span_val:
            evidence.append(f"Temporal span: {span_val}")

        # Component differences for additional evidence
        component_diffs = getattr(summary, "component_differences", [])
        for cd in (component_diffs[:3] if isinstance(component_diffs, (list, tuple)) else []):
            comp_name = getattr(cd, "component_name", "")
            change_type = getattr(cd, "change_type", None)
            ct_val = getattr(change_type, "value", str(change_type)) if change_type is not None else ""
            if comp_name:
                evidence.append(f"Component '{comp_name}': {ct_val}")

        results.append((desc, basis_hint, strength_hint, evidence))

    # Handle dict input
    elif isinstance(summary, dict):
        has_difference = summary.get("has_difference", False)
        if not has_difference:
            return []

        magnitude = summary.get("magnitude", "none")
        nature = summary.get("nature", "stable")
        changed_components = summary.get("changed_components", [])

        mag_strength = {"none": 0.0, "minimal": 0.15, "noticeable": 0.35, "significant": 0.55, "substantial": 0.75}
        strength_hint = mag_strength.get(magnitude, 0.2)

        nature_map = {
            "shifting": ("change_direction", "Ongoing shift may continue"),
            "returning": ("change_direction", "Return to previous state may continue"),
            "stable": ("context_persistence", "Current stable state likely persists"),
            "fluctuating": ("change_direction", "Fluctuations may continue"),
        }
        basis_hint, desc = nature_map.get(nature, ("change_direction", "Change direction may persist"))
        evidence = [f"Difference: magnitude={magnitude}, nature={nature}"]
        if changed_components:
            evidence.append(f"Changed: {', '.join(str(c) for c in changed_components[:3])}")
        results.append((desc, basis_hint, strength_hint, evidence))

    return results


def extract_from_narrative(
    state: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract expectation materials from narrative state.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    Reads NarrativeState via duck typing.
    """
    if state is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Handle NarrativeState object
    if hasattr(state, "fragments") and hasattr(state, "coherence"):
        fragments = getattr(state, "fragments", ())
        coherence = getattr(state, "coherence", None)
        trend = getattr(state, "trend", None)
        description = getattr(state, "description", "")

        trend_val = ""
        if trend is not None:
            trend_val = getattr(trend, "value", str(trend))

        # Trend → basis/description mapping
        trend_map = {
            "accumulating": ("context_persistence", "Narrative accumulation suggests continued experience gathering"),
            "condensing": ("context_persistence", "Narrative condensation suggests convergence of themes"),
            "stable": ("context_persistence", "Stable narrative suggests context continuity"),
            "dissolving": ("context_persistence", "Dissolving narrative may lead to new directions"),
        }

        basis_hint, base_desc = trend_map.get(
            trend_val,
            ("context_persistence", "Narrative context may continue"),
        )

        # Coherence → strength influence
        coherence_val = ""
        if coherence is not None:
            level = getattr(coherence, "level", None)
            coherence_val = getattr(level, "value", str(level)) if level is not None else ""

        coherence_strength = {
            "coherent": 0.6, "loosely_coherent": 0.4,
            "fragmented": 0.2, "undefined": 0.15,
        }
        strength_hint = coherence_strength.get(coherence_val, 0.3)

        # Adjust for trend
        if trend_val == "accumulating":
            strength_hint = min(1.0, strength_hint + 0.1)
        elif trend_val == "dissolving":
            strength_hint = max(0.05, strength_hint - 0.15)

        if description:
            desc = f"{base_desc} — {description[:100]}"
        else:
            desc = base_desc

        evidence = []
        if trend_val:
            evidence.append(f"Narrative trend: {trend_val}")
        if coherence_val:
            evidence.append(f"Coherence: {coherence_val}")
        fragment_count = len(fragments) if isinstance(fragments, (list, tuple)) else 0
        if fragment_count > 0:
            evidence.append(f"Fragments: {fragment_count}")

        if evidence:
            results.append((desc, basis_hint, strength_hint, evidence))

    # Handle dict input
    elif isinstance(state, dict):
        trend = state.get("trend", "")
        coherence = state.get("coherence", "")
        fragment_count = state.get("fragment_count", 0)
        avg_vividness = state.get("average_vividness", 0.0)

        if not trend and not coherence:
            return []

        trend_map = {
            "accumulating": ("context_persistence", "Narrative accumulation may continue"),
            "condensing": ("context_persistence", "Narrative condensation may continue"),
            "stable": ("context_persistence", "Stable narrative context persists"),
            "dissolving": ("context_persistence", "Dissolving narrative may shift"),
        }
        basis_hint, desc = trend_map.get(trend, ("context_persistence", "Narrative context may persist"))

        coherence_strength = {"coherent": 0.6, "loosely_coherent": 0.4, "fragmented": 0.2}
        strength_hint = coherence_strength.get(coherence, 0.3)

        evidence = []
        if trend:
            evidence.append(f"Trend: {trend}")
        if coherence:
            evidence.append(f"Coherence: {coherence}")
        if fragment_count > 0:
            evidence.append(f"Fragments: {fragment_count}")
        results.append((desc, basis_hint, strength_hint, evidence))

    return results


# =============================================================================
# Computation Functions (Pure)
# =============================================================================

def compute_evidence_strength(evidence_links: list[EvidenceLink]) -> float:
    """
    Compute integrated evidence strength from multiple evidence links.

    Uses weighted aggregation (not simple average).
    """
    if not evidence_links:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for i, link in enumerate(evidence_links):
        # Earlier evidence gets slightly more weight (diminishing returns)
        weight = 1.0 / (1.0 + i * 0.2)
        weighted_sum += link.contribution * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return min(1.0, weighted_sum / total_weight)


def detect_competitions(
    expectations: list[ExpectationCandidate],
) -> list[tuple[str, str]]:
    """
    Detect competing expectation pairs.

    Two expectations compete if they share source type words
    but have different bases (suggesting different directions).
    Uses description word overlap (Jaccard) + different basis for detection.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(expectations):
        for j, b in enumerate(expectations):
            if i >= j:
                continue

            pair_key = (a.expectation_id, b.expectation_id)
            if pair_key in seen:
                continue

            # Same source type but different basis suggests competition
            if a.source_type == b.source_type and a.basis != b.basis:
                # Check word overlap in descriptions
                words_a = set(a.description.lower().split())
                words_b = set(b.description.lower().split())
                union = words_a | words_b
                if union:
                    jaccard = len(words_a & words_b) / len(union)
                    if jaccard >= 0.2:
                        pairs.append(pair_key)
                        seen.add(pair_key)
                        continue

            # Different basis with similar descriptions also suggests competition
            if a.basis != b.basis:
                words_a = set(a.description.lower().split())
                words_b = set(b.description.lower().split())
                union = words_a | words_b
                if union:
                    jaccard = len(words_a & words_b) / len(union)
                    if jaccard >= 0.4:
                        pairs.append(pair_key)
                        seen.add(pair_key)

    return pairs


def determine_expectation_basis(
    source_types: list[ExpectationSourceType],
) -> ExpectationBasis:
    """Determine ExpectationBasis from a list of source types."""
    if not source_types:
        return ExpectationBasis.UNDEFINED

    unique = set(source_types)
    if len(unique) > 1:
        return ExpectationBasis.COMBINED

    source = source_types[0]
    mapping = {
        ExpectationSourceType.REPETITION: ExpectationBasis.PATTERN_CONTINUATION,
        ExpectationSourceType.SELF_DIFFERENCE: ExpectationBasis.CHANGE_DIRECTION,
        ExpectationSourceType.NARRATIVE: ExpectationBasis.CONTEXT_PERSISTENCE,
        ExpectationSourceType.MIXED: ExpectationBasis.COMBINED,
    }
    return mapping.get(source, ExpectationBasis.UNDEFINED)


def generate_expectation_description(
    basis: ExpectationBasis,
    source_descriptions: list[str],
) -> str:
    """Generate an integrated expectation description."""
    basis_labels = {
        ExpectationBasis.PATTERN_CONTINUATION: "Based on recurring patterns",
        ExpectationBasis.CHANGE_DIRECTION: "Based on observed change direction",
        ExpectationBasis.CONTEXT_PERSISTENCE: "Based on narrative context persistence",
        ExpectationBasis.COMBINED: "Based on multiple sources",
        ExpectationBasis.UNDEFINED: "Weak expectation",
    }
    prefix = basis_labels.get(basis, "Expectation")

    if source_descriptions:
        combined = "; ".join(d[:80] for d in source_descriptions[:3])
        return f"{prefix}: {combined}"
    return prefix


# =============================================================================
# Expectation Formation System
# =============================================================================

class ExpectationFormationSystem:
    """
    Expectation Formation System (予期・期待の形成)

    過去の反復・差分・物語から「次に起きうる展開の仮の見通し」を
    弱く生成する。

    CRITICAL CONSTRAINTS:
    - 判断・目的・価値・責任に一切接続しない
    - 予期を正解化・評価化しない
    - 人格の方向性を固定しない
    - 予期は仮説として保持し固定しない
    - 予期同士の競合を許容する
    - 規範的な方向付けを行わない
    """

    def __init__(self, config: Optional[ExpectationFormationConfig] = None):
        self._config = config or ExpectationFormationConfig()
        self._expectations: list[ExpectationCandidate] = []
        self._evidence_links: list[EvidenceLink] = []
        self._total_created: int = 0
        self._total_revisions: int = 0
        self._total_expirations: int = 0
        self._last_store: Optional[ExpectationStore] = None
        # Cache for detect_competitions result to avoid O(n^2) recomputation
        self._cached_competition_pairs: Optional[list[tuple[str, str]]] = None

    def form_expectations(
        self,
        tendency_bias: Optional[Any] = None,
        difference_summary: Optional[Any] = None,
        narrative_state: Optional[Any] = None,
    ) -> ExpectationStore:
        """
        Form expectations from three input sources.

        3つの入力源からextract → ExpectationCandidate生成 →
        競合検出 → evidence_links生成 → 減衰適用 → スナップショット返却。
        """
        current_time = str(time.time())

        # Extract from each source
        tendency_extracts = extract_from_tendency(tendency_bias)
        difference_extracts = extract_from_difference(difference_summary)
        narrative_extracts = extract_from_narrative(narrative_state)

        # Create new candidates from extracts
        all_extracts: list[tuple[str, str, float, list[str], ExpectationSourceType]] = []
        for desc, basis_hint, strength, evidence in tendency_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ExpectationSourceType.REPETITION))
        for desc, basis_hint, strength, evidence in difference_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ExpectationSourceType.SELF_DIFFERENCE))
        for desc, basis_hint, strength, evidence in narrative_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ExpectationSourceType.NARRATIVE))

        for desc, basis_hint, strength, evidence_descs, source_type in all_extracts:
            # Map basis_hint string to enum
            basis_map = {
                "pattern_continuation": ExpectationBasis.PATTERN_CONTINUATION,
                "change_direction": ExpectationBasis.CHANGE_DIRECTION,
                "context_persistence": ExpectationBasis.CONTEXT_PERSISTENCE,
                "combined": ExpectationBasis.COMBINED,
            }
            basis = basis_map.get(basis_hint, ExpectationBasis.UNDEFINED)

            exp_id = _generate_id()

            # Generate evidence links
            new_links = self._generate_evidence_links(exp_id, source_type, evidence_descs)
            link_ids = tuple(link.link_id for link in new_links)
            self._evidence_links.extend(new_links)

            candidate = ExpectationCandidate(
                expectation_id=exp_id,
                source_type=source_type,
                basis=basis,
                description=desc,
                timestamp=current_time,
                freshness=1.0,
                strength=max(0.0, min(1.0, strength)),
                reference_count=0,
                evidence_ids=link_ids,
                competing_ids=(),
                revision_count=0,
                undetermined_aspects=("outcome_uncertain", "timing_unknown"),
            )
            self._expectations.append(candidate)
            self._total_created += 1

        # Detect competitions among all expectations
        competition_pairs = detect_competitions(self._expectations)
        self._cached_competition_pairs = competition_pairs
        for id_a, id_b in competition_pairs:
            for i, exp in enumerate(self._expectations):
                if exp.expectation_id == id_a:
                    self._expectations[i] = exp.with_competing(id_b)
                elif exp.expectation_id == id_b:
                    self._expectations[i] = exp.with_competing(id_a)

        # Apply natural decay to existing expectations
        self._apply_decay()

        # Enforce max expectations limit
        self._enforce_capacity()

        return self._build_store(current_time)

    def decay_expectations(self) -> ExpectationStore:
        """
        Apply natural decay to all expectations.

        freshness: base_decay_rate で減衰（reference_countで変調）
        strength: strength_decay_rate で減衰
        stale_threshold以下 AND min_strength以下 → 除去
        """
        current_time = str(time.time())
        self._apply_decay()
        return self._build_store(current_time)

    def reference_expectation(self, expectation_id: str) -> None:
        """
        Mark an expectation as referenced (boosts freshness).

        reference_count +1, freshness += freshness_boost_on_reference
        """
        for i, exp in enumerate(self._expectations):
            if exp.expectation_id == expectation_id:
                referenced = exp.with_reference()
                boosted = referenced.with_freshness(
                    referenced.freshness + self._config.freshness_boost_on_reference,
                )
                self._expectations[i] = boosted
                return

    def revise_expectation(self, expectation_id: str, new_description: str) -> None:
        """
        Revise an expectation's description (予期は固定しない).

        後からの修正を許容する。
        """
        for i, exp in enumerate(self._expectations):
            if exp.expectation_id == expectation_id:
                self._expectations[i] = exp.revise(new_description)
                self._total_revisions += 1
                return

    def get_active_expectations(self, max_count: int = 10) -> list[ExpectationCandidate]:
        """Get active expectations sorted by strength."""
        active = [
            e for e in self._expectations
            if e.freshness > self._config.stale_threshold
        ]
        active.sort(key=lambda e: e.strength, reverse=True)
        return active[:max_count]

    def get_store(self) -> ExpectationStore:
        """Get current store snapshot."""
        return self._build_store(str(time.time()))

    def get_last_store(self) -> Optional[ExpectationStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _apply_decay(self) -> None:
        """Apply decay to all expectations, removing expired ones."""
        new_expectations: list[ExpectationCandidate] = []

        for exp in self._expectations:
            # Reference count modulates decay: more references → slower decay
            ref_modifier = max(0.5, 1.0 - exp.reference_count * 0.1)
            freshness_decay = self._config.base_decay_rate * ref_modifier
            new_freshness = exp.freshness - freshness_decay

            new_strength = exp.strength - self._config.strength_decay_rate

            # Check if expired
            if (new_freshness <= self._config.stale_threshold
                    and new_strength <= self._config.min_strength_for_retention):
                self._total_expirations += 1
                # Clean up evidence links for this expectation
                self._evidence_links = [
                    el for el in self._evidence_links
                    if el.expectation_id != exp.expectation_id
                ]
                continue

            updated = exp.with_freshness(new_freshness).with_strength(new_strength)
            new_expectations.append(updated)

        self._expectations = new_expectations

    def _enforce_capacity(self) -> None:
        """Remove weakest expectations if over capacity."""
        while len(self._expectations) > self._config.max_expectations:
            # Find weakest (lowest strength, then lowest freshness)
            weakest_idx = min(
                range(len(self._expectations)),
                key=lambda i: (self._expectations[i].strength, self._expectations[i].freshness),
            )
            removed = self._expectations.pop(weakest_idx)
            self._evidence_links = [
                el for el in self._evidence_links
                if el.expectation_id != removed.expectation_id
            ]
            self._total_expirations += 1

    def _generate_evidence_links(
        self,
        expectation_id: str,
        source_type: ExpectationSourceType,
        source_descriptions: list[str],
    ) -> list[EvidenceLink]:
        """Generate evidence links for a new expectation."""
        links: list[EvidenceLink] = []
        max_links = self._config.max_evidence_per_expectation

        for desc in source_descriptions[:max_links]:
            # Contribution decreases for later evidence items
            idx = len(links)
            contribution = max(0.1, 1.0 - idx * 0.15)

            link = EvidenceLink(
                link_id=_generate_id(),
                expectation_id=expectation_id,
                source_type=source_type,
                source_description=desc,
                contribution=contribution,
            )
            links.append(link)

        return links

    def _build_store(self, current_time: str) -> ExpectationStore:
        """Build an ExpectationStore snapshot."""
        active = [
            e for e in self._expectations
            if e.freshness > self._config.stale_threshold
        ]
        avg_freshness = (
            sum(e.freshness for e in self._expectations) / len(self._expectations)
            if self._expectations else 0.0
        )
        avg_strength = (
            sum(e.strength for e in self._expectations) / len(self._expectations)
            if self._expectations else 0.0
        )

        # Count competing pairs (use cache if available)
        if self._cached_competition_pairs is not None:
            competition_pairs = self._cached_competition_pairs
            self._cached_competition_pairs = None
        else:
            competition_pairs = detect_competitions(self._expectations)

        description = _generate_store_description(
            len(self._expectations),
            len(active),
            avg_freshness,
            avg_strength,
            len(competition_pairs),
            self._total_expirations,
        )

        store = ExpectationStore(
            expectations=tuple(self._expectations),
            evidence_links=tuple(self._evidence_links),
            total_expectations_created=self._total_created,
            total_revisions=self._total_revisions,
            total_expirations=self._total_expirations,
            average_freshness=round(avg_freshness, 4),
            average_strength=round(avg_strength, 4),
            active_expectation_count=len(active),
            competing_pair_count=len(competition_pairs),
            timestamp=current_time,
            description=description,
        )
        self._last_store = store
        return store


def _generate_store_description(
    total: int,
    active: int,
    avg_freshness: float,
    avg_strength: float,
    competing_pairs: int,
    total_expirations: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No expectations formed yet."

    parts = [f"{active} active expectations out of {total} total"]

    if avg_strength >= 0.5:
        parts.append("generally strong expectations")
    elif avg_strength >= 0.2:
        parts.append("moderate strength expectations")
    else:
        parts.append("mostly weak expectations")

    if competing_pairs > 0:
        parts.append(f"{competing_pairs} competing pairs")

    if total_expirations > 0:
        parts.append(f"{total_expirations} expired")

    parts.append(f"avg freshness: {avg_freshness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def form_from_chain(
    system: ExpectationFormationSystem,
    tendency_bias: Optional[Any] = None,
    difference_summary: Optional[Any] = None,
    narrative_state: Optional[Any] = None,
) -> ExpectationStore:
    """
    Form expectations from the self-observation chain.

    自己観測チェーンからの統合ヘルパー。
    各入力は読み取り専用で参照される。
    """
    return system.form_expectations(
        tendency_bias=tendency_bias,
        difference_summary=difference_summary,
        narrative_state=narrative_state,
    )


def generate_expectation_tags(
    store: Optional[Any],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from ExpectationStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags: list[dict[str, Any]] = []

    if store is None:
        tags.append({
            "category": "EXPECTATION_COUNT",
            "label": "no_expectations",
            "description": "No expectations formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    has_expectations = getattr(store, "has_expectations", None)
    if callable(has_expectations) and not has_expectations():
        tags.append({
            "category": "EXPECTATION_COUNT",
            "label": "no_expectations",
            "description": "No expectations formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    # Expectation count
    active_count = getattr(store, "active_expectation_count", 0)
    tags.append({
        "category": "EXPECTATION_COUNT",
        "label": f"expectations_{active_count}",
        "description": f"Expectation formation holds {active_count} active expectations",
        "weight": 0.06 * scale,
    })

    # Average strength
    avg_strength = getattr(store, "average_strength", 0.0)
    expectations = getattr(store, "expectations", ())
    max_strength = max((e.strength for e in expectations), default=0.0) if expectations else 0.0
    strength_label = determine_strength_level(avg_strength).value
    tags.append({
        "category": "EXPECTATION_STRENGTH",
        "label": f"strength_{strength_label}",
        "description": f"Average expectation strength: {avg_strength:.2f}, max: {max_strength:.2f}",
        "weight": 0.07 * scale,
    })

    # Average freshness
    avg_freshness = getattr(store, "average_freshness", 0.0)
    freshness_label = determine_freshness_level(avg_freshness).value
    tags.append({
        "category": "EXPECTATION_FRESHNESS",
        "label": f"freshness_{freshness_label}",
        "description": f"Average expectation freshness: {avg_freshness:.2f}",
        "weight": 0.05 * scale,
    })

    # Competition
    competing_pair_count = getattr(store, "competing_pair_count", 0)
    if competing_pair_count > 0:
        tags.append({
            "category": "EXPECTATION_COMPETITION",
            "label": f"competing_{competing_pair_count}",
            "description": f"{competing_pair_count} pairs of competing expectations",
            "weight": 0.06 * scale,
        })

    # Integrated description
    description = getattr(store, "description", "")
    if description:
        tags.append({
            "category": "EXPECTATION_INTEGRATED",
            "label": "expectation_awareness",
            "description": description,
            "weight": 0.08 * scale,
        })

    return tags


def get_expectation_summary(store: Optional[Any]) -> str:
    """Get human-readable summary. For introspection/logging only."""
    if store is None:
        return "=== Expectation Formation State ===\nNo expectations formed yet."

    has_expectations = getattr(store, "has_expectations", None)
    if callable(has_expectations) and not has_expectations():
        return "=== Expectation Formation State ===\nNo expectations formed yet."

    expectations = getattr(store, "expectations", ())
    active_count = getattr(store, "active_expectation_count", 0)
    total_created = getattr(store, "total_expectations_created", 0)
    total_revisions = getattr(store, "total_revisions", 0)
    total_expirations = getattr(store, "total_expirations", 0)
    avg_freshness = getattr(store, "average_freshness", 0.0)
    avg_strength = getattr(store, "average_strength", 0.0)
    competing_pairs = getattr(store, "competing_pair_count", 0)
    description = getattr(store, "description", "")

    lines = [
        "=== Expectation Formation State ===",
        f"Total expectations: {len(expectations)}",
        f"Active expectations: {active_count}",
        f"Total created: {total_created}",
        f"Total revisions: {total_revisions}",
        f"Total expirations: {total_expirations}",
        f"Average freshness: {avg_freshness:.2f}",
        f"Average strength: {avg_strength:.2f}",
        f"Competing pairs: {competing_pairs}",
        "",
    ]

    # Show top expectations
    sorted_exp = sorted(expectations, key=lambda e: e.strength, reverse=True)
    if sorted_exp:
        lines.append("Top expectations:")
        for exp in sorted_exp[:5]:
            lines.append(
                f"  [{exp.source_type.value}:{exp.basis.value}] "
                f"{exp.description[:80]}"
                f" (strength: {exp.get_strength_level().value}, "
                f"freshness: {exp.get_freshness_level().value})"
            )
        lines.append("")

    lines.append(f"Integrated: {description}")
    return "\n".join(lines)


def get_expectation_for_introspection(
    store: Optional[Any],
) -> dict[str, Any]:
    """
    Get structured expectation data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    if store is None:
        return {
            "has_expectations": False,
            "total_expectations": 0,
            "active_count": 0,
            "average_strength": 0.0,
            "average_freshness": 0.0,
            "source_distribution": {},
            "basis_distribution": {},
            "competing_pair_count": 0,
            "strongest_expectation_description": "",
            "timestamp": "",
        }

    expectations = getattr(store, "expectations", ())

    source_dist: dict[str, int] = {}
    for exp in expectations:
        key = exp.source_type.value
        source_dist[key] = source_dist.get(key, 0) + 1

    basis_dist: dict[str, int] = {}
    for exp in expectations:
        key = exp.basis.value
        basis_dist[key] = basis_dist.get(key, 0) + 1

    strongest_desc = ""
    if expectations:
        strongest = max(expectations, key=lambda e: e.strength)
        strongest_desc = strongest.description[:120]

    return {
        "has_expectations": len(expectations) > 0,
        "total_expectations": len(expectations),
        "active_count": getattr(store, "active_expectation_count", 0),
        "average_strength": getattr(store, "average_strength", 0.0),
        "average_freshness": getattr(store, "average_freshness", 0.0),
        "source_distribution": source_dist,
        "basis_distribution": basis_dist,
        "competing_pair_count": getattr(store, "competing_pair_count", 0),
        "strongest_expectation_description": strongest_desc,
        "timestamp": getattr(store, "timestamp", ""),
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: ExpectationStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    予期は参照素材に留め、直接の意思決定経路を持たない。
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
                "total_expectations_created", "total_revisions",
                "total_expirations", "average_freshness", "average_strength",
                "active_expectation_count", "competing_pair_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: ExpectationFormationSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    予期から目標を生成しない。
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


def verify_read_only_principle(system: ExpectationFormationSystem) -> bool:
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


def verify_no_value_modification(system: ExpectationFormationSystem) -> bool:
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

def create_config(**kwargs: Any) -> ExpectationFormationConfig:
    """Create a custom configuration."""
    return ExpectationFormationConfig(**kwargs)


def create_empty_store() -> ExpectationStore:
    """Create an empty expectation store."""
    return ExpectationStore(
        expectations=(),
        evidence_links=(),
        total_expectations_created=0,
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.0,
        average_strength=0.0,
        active_expectation_count=0,
        competing_pair_count=0,
        timestamp=str(time.time()),
        description="No expectations formed yet.",
    )


def create_system(
    config: Optional[ExpectationFormationConfig] = None,
) -> ExpectationFormationSystem:
    """Create a new ExpectationFormationSystem."""
    return ExpectationFormationSystem(config=config)


def save_expectation_state(
    store: ExpectationStore,
    filepath: str,
) -> None:
    """Save expectation state to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_expectation_state(filepath: str) -> ExpectationStore:
    """Load expectation state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExpectationStore.from_dict(data)
