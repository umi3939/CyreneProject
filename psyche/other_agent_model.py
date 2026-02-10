"""
Other Agent Model (他者モデル)

「相手がどう感じているか」の推測を仮説として弱く保持するモジュール。

現状の課題:
自己側の観測・反応に偏っており、他者の状態推測の独立層が存在しない。
自己と他者の境界を構造として導入し、自我形成の前段条件を整える。

CRITICAL DESIGN PRINCIPLES:
- 入力は全て Optional[Any] + duck typing（循環import回避）
- 他者の意図・価値・信念を断定しない
- 正誤や善悪の評価を付与しない
- 目的や行動の最適化に結び付けない
- 自己像や人格の方向性を固定しない
- 候補は仮説として保持し固定しない、競合を許容する
- 判断・目的・価値・責任に一切接続しない
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import json
import time
import uuid


# =============================================================================
# Abstract Enums
# =============================================================================

class ObservationSourceType(Enum):
    """
    他者観測の入力源がどこ由来か。

    評価的ではなく、記述的な分類。
    """
    EXTERNAL_CONTEXT = "external_context"  # 外部文脈由来
    REACTION_LOG = "reaction_log"          # 反応ログ由来
    SELF_CONTRAST = "self_contrast"        # 自己対比由来
    MIXED = "mixed"                        # 複数ソース混合


class InferenceBasis(Enum):
    """
    推論根拠の種類。

    行動的、文脈的、対比的。
    """
    BEHAVIORAL = "behavioral"      # 行動的根拠
    CONTEXTUAL = "contextual"      # 文脈的根拠
    CONTRAST = "contrast"          # 対比的根拠
    COMBINED = "combined"          # 複合的な根拠
    UNDEFINED = "undefined"        # 未確定


class HypothesisStrength(Enum):
    """
    仮説の確からしさの抽象レベル。

    評価ではなく、仮説候補の安定度の記述。
    """
    STRONG = "strong"          # 強い仮説
    MODERATE = "moderate"      # 中程度
    WEAK = "weak"              # 弱い仮説
    FAINT = "faint"            # かすかな仮説
    UNDEFINED = "undefined"    # 未確定


class HypothesisFreshness(Enum):
    """
    仮説の新鮮度レベル。

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

def determine_freshness_level(freshness: float) -> HypothesisFreshness:
    """Determine HypothesisFreshness from a numeric freshness value (0.0-1.0)."""
    if freshness >= 0.8:
        return HypothesisFreshness.FRESH
    elif freshness >= 0.6:
        return HypothesisFreshness.RECENT
    elif freshness >= 0.4:
        return HypothesisFreshness.AGING
    elif freshness >= 0.15:
        return HypothesisFreshness.STALE
    else:
        return HypothesisFreshness.FADED


def determine_strength_level(strength: float) -> HypothesisStrength:
    """Determine HypothesisStrength from a numeric strength value (0.0-1.0)."""
    if strength >= 0.7:
        return HypothesisStrength.STRONG
    elif strength >= 0.4:
        return HypothesisStrength.MODERATE
    elif strength >= 0.2:
        return HypothesisStrength.WEAK
    elif strength >= 0.05:
        return HypothesisStrength.FAINT
    else:
        return HypothesisStrength.UNDEFINED


# =============================================================================
# Core Dataclasses (all frozen)
# =============================================================================

@dataclass(frozen=True)
class ObservationLink:
    """
    観測と仮説の弱い接続。

    仮説がどの入力源からどの程度の寄与を受けたかを記録する。
    NOT evaluative. NOT prescriptive.
    """
    link_id: str
    hypothesis_id: str
    source_type: ObservationSourceType
    source_description: str
    contribution: float  # 0.0〜1.0


@dataclass(frozen=True)
class OtherStateHypothesis:
    """
    他者の状態仮説（コア構造）。

    「相手がどう感じているか」の推測を保持する不変構造。
    仮説として保持し固定しない。修正・撤回が可能。
    NOT evaluative. NOT prescriptive.
    """
    hypothesis_id: str
    source_type: ObservationSourceType
    basis: InferenceBasis
    description: str
    timestamp: str
    freshness: float                       # 0.0〜1.0
    strength: float                        # 0.0〜1.0
    reference_count: int
    evidence_ids: tuple[str, ...]          # 根拠リンクID群
    competing_ids: tuple[str, ...]         # 競合する仮説のID群
    revision_count: int                    # 修正回数
    undetermined_aspects: tuple[str, ...]  # 未確定の側面

    def get_freshness_level(self) -> HypothesisFreshness:
        """Get abstract freshness level."""
        return determine_freshness_level(self.freshness)

    def get_strength_level(self) -> HypothesisStrength:
        """Get abstract strength level."""
        return determine_strength_level(self.strength)

    def with_freshness(self, new_freshness: float) -> OtherStateHypothesis:
        """Create a copy with updated freshness."""
        return OtherStateHypothesis(
            hypothesis_id=self.hypothesis_id,
            source_type=self.source_type,
            basis=self.basis,
            description=self.description,
            timestamp=self.timestamp,
            freshness=max(0.0, min(1.0, new_freshness)),
            strength=self.strength,
            reference_count=self.reference_count,
            evidence_ids=self.evidence_ids,
            competing_ids=self.competing_ids,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_strength(self, new_strength: float) -> OtherStateHypothesis:
        """Create a copy with updated strength."""
        return OtherStateHypothesis(
            hypothesis_id=self.hypothesis_id,
            source_type=self.source_type,
            basis=self.basis,
            description=self.description,
            timestamp=self.timestamp,
            freshness=self.freshness,
            strength=max(0.0, min(1.0, new_strength)),
            reference_count=self.reference_count,
            evidence_ids=self.evidence_ids,
            competing_ids=self.competing_ids,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_reference(self) -> OtherStateHypothesis:
        """Create a copy with incremented reference count."""
        return OtherStateHypothesis(
            hypothesis_id=self.hypothesis_id,
            source_type=self.source_type,
            basis=self.basis,
            description=self.description,
            timestamp=self.timestamp,
            freshness=self.freshness,
            strength=self.strength,
            reference_count=self.reference_count + 1,
            evidence_ids=self.evidence_ids,
            competing_ids=self.competing_ids,
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )

    def revise(self, new_description: str) -> OtherStateHypothesis:
        """Create a revised copy (仮説は固定しない、修正可能)."""
        return OtherStateHypothesis(
            hypothesis_id=self.hypothesis_id,
            source_type=self.source_type,
            basis=self.basis,
            description=new_description,
            timestamp=self.timestamp,
            freshness=self.freshness,
            strength=self.strength,
            reference_count=self.reference_count,
            evidence_ids=self.evidence_ids,
            competing_ids=self.competing_ids,
            revision_count=self.revision_count + 1,
            undetermined_aspects=self.undetermined_aspects,
        )

    def with_competing(self, competing_id: str) -> OtherStateHypothesis:
        """Create a copy with an additional competing hypothesis ID."""
        if competing_id in self.competing_ids:
            return self
        return OtherStateHypothesis(
            hypothesis_id=self.hypothesis_id,
            source_type=self.source_type,
            basis=self.basis,
            description=self.description,
            timestamp=self.timestamp,
            freshness=self.freshness,
            strength=self.strength,
            reference_count=self.reference_count,
            evidence_ids=self.evidence_ids,
            competing_ids=self.competing_ids + (competing_id,),
            revision_count=self.revision_count,
            undetermined_aspects=self.undetermined_aspects,
        )


@dataclass(frozen=True)
class SelfOtherBoundary:
    """
    自己/他者の境界指標（弱い差分情報）。

    自己と他者の区別を弱く構造化する。
    NOT evaluative. NOT prescriptive.
    """
    boundary_id: str
    self_description: str
    other_description: str
    divergence: float                    # 0.0〜1.0 自己と他者の乖離度
    boundary_aspects: tuple[str, ...]    # どの側面が異なるか
    timestamp: str


@dataclass(frozen=True)
class OtherModelStore:
    """
    不変スナップショット - 他者モデルの全体状態。

    内省記録層に対して読み取り専用で提供する。
    """
    hypotheses: tuple[OtherStateHypothesis, ...]
    observation_links: tuple[ObservationLink, ...]
    boundaries: tuple[SelfOtherBoundary, ...]
    total_hypotheses_created: int
    total_revisions: int
    total_expirations: int
    average_freshness: float
    average_strength: float
    active_hypothesis_count: int
    competing_pair_count: int
    boundary_count: int
    timestamp: str
    description: str

    def has_hypotheses(self) -> bool:
        return len(self.hypotheses) > 0

    def get_active_hypotheses(self, stale_threshold: float = 0.15) -> tuple[OtherStateHypothesis, ...]:
        """Get hypotheses with freshness above the stale threshold."""
        return tuple(
            h for h in self.hypotheses
            if h.freshness > stale_threshold
        )

    def get_strong_hypotheses(self) -> tuple[OtherStateHypothesis, ...]:
        """Get hypotheses with strength above 0.5."""
        return tuple(
            h for h in self.hypotheses
            if h.strength > 0.5
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypotheses": [
                _hypothesis_to_dict(h) for h in self.hypotheses
            ],
            "observation_links": [
                {
                    "link_id": ol.link_id,
                    "hypothesis_id": ol.hypothesis_id,
                    "source_type": ol.source_type.value,
                    "source_description": ol.source_description,
                    "contribution": ol.contribution,
                }
                for ol in self.observation_links
            ],
            "boundaries": [
                {
                    "boundary_id": b.boundary_id,
                    "self_description": b.self_description,
                    "other_description": b.other_description,
                    "divergence": b.divergence,
                    "boundary_aspects": list(b.boundary_aspects),
                    "timestamp": b.timestamp,
                }
                for b in self.boundaries
            ],
            "total_hypotheses_created": self.total_hypotheses_created,
            "total_revisions": self.total_revisions,
            "total_expirations": self.total_expirations,
            "average_freshness": self.average_freshness,
            "average_strength": self.average_strength,
            "active_hypothesis_count": self.active_hypothesis_count,
            "competing_pair_count": self.competing_pair_count,
            "boundary_count": self.boundary_count,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OtherModelStore:
        """Create from dictionary."""
        hypotheses = tuple(
            _hypothesis_from_dict(h) for h in data.get("hypotheses", [])
        )
        observation_links = tuple(
            ObservationLink(
                link_id=ol["link_id"],
                hypothesis_id=ol.get("hypothesis_id", ""),
                source_type=ObservationSourceType(ol.get("source_type", "mixed")),
                source_description=ol.get("source_description", ""),
                contribution=ol.get("contribution", 0.0),
            )
            for ol in data.get("observation_links", [])
        )
        boundaries = tuple(
            SelfOtherBoundary(
                boundary_id=b["boundary_id"],
                self_description=b.get("self_description", ""),
                other_description=b.get("other_description", ""),
                divergence=b.get("divergence", 0.0),
                boundary_aspects=tuple(b.get("boundary_aspects", ())),
                timestamp=b.get("timestamp", ""),
            )
            for b in data.get("boundaries", [])
        )
        return cls(
            hypotheses=hypotheses,
            observation_links=observation_links,
            boundaries=boundaries,
            total_hypotheses_created=data.get("total_hypotheses_created", 0),
            total_revisions=data.get("total_revisions", 0),
            total_expirations=data.get("total_expirations", 0),
            average_freshness=data.get("average_freshness", 0.0),
            average_strength=data.get("average_strength", 0.0),
            active_hypothesis_count=data.get("active_hypothesis_count", 0),
            competing_pair_count=data.get("competing_pair_count", 0),
            boundary_count=data.get("boundary_count", 0),
            timestamp=data.get("timestamp", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class OtherAgentModelConfig:
    """
    Configuration for the other agent model system.

    These parameters control hypothesis lifecycle.
    They do NOT affect decisions.
    """
    max_hypotheses: int = 60
    base_decay_rate: float = 0.05
    strength_decay_rate: float = 0.03
    freshness_boost_on_reference: float = 0.10
    stale_threshold: float = 0.15
    min_strength_for_retention: float = 0.05
    max_evidence_per_hypothesis: int = 8
    max_boundaries: int = 10


# =============================================================================
# Serialization Helpers
# =============================================================================

def _hypothesis_to_dict(hypothesis: OtherStateHypothesis) -> dict[str, Any]:
    """Convert an OtherStateHypothesis to a dictionary."""
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "source_type": hypothesis.source_type.value,
        "basis": hypothesis.basis.value,
        "description": hypothesis.description,
        "timestamp": hypothesis.timestamp,
        "freshness": hypothesis.freshness,
        "strength": hypothesis.strength,
        "reference_count": hypothesis.reference_count,
        "evidence_ids": list(hypothesis.evidence_ids),
        "competing_ids": list(hypothesis.competing_ids),
        "revision_count": hypothesis.revision_count,
        "undetermined_aspects": list(hypothesis.undetermined_aspects),
    }


def _hypothesis_from_dict(data: dict[str, Any]) -> OtherStateHypothesis:
    """Create an OtherStateHypothesis from a dictionary."""
    return OtherStateHypothesis(
        hypothesis_id=data["hypothesis_id"],
        source_type=ObservationSourceType(data.get("source_type", "mixed")),
        basis=InferenceBasis(data.get("basis", "undefined")),
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

def extract_from_external_context(
    context: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract hypothesis materials from external context.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    Reads ExternalContext via duck typing: pace, weight, density, continuity, responsiveness.
    """
    if context is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Handle ExternalContext object (duck typing)
    if hasattr(context, "responsiveness") and hasattr(context, "weight"):
        responsiveness = getattr(context, "responsiveness", 0.5)
        weight = getattr(context, "weight", 0.5)
        pace = getattr(context, "pace", 0.5)
        density = getattr(context, "density", 0.5)
        continuity = getattr(context, "continuity", 0.5)

        # High responsiveness → engaged
        if responsiveness >= 0.7:
            desc = "Other party appears engaged and responsive"
            strength_hint = min(1.0, responsiveness * 0.6)
            evidence = [f"Responsiveness: {responsiveness:.2f}"]
            results.append((desc, "behavioral", strength_hint, evidence))

        # Low responsiveness → disengaged
        elif responsiveness <= 0.3:
            desc = "Other party appears disengaged or distant"
            strength_hint = min(1.0, (1.0 - responsiveness) * 0.5)
            evidence = [f"Responsiveness: {responsiveness:.2f}"]
            results.append((desc, "behavioral", strength_hint, evidence))

        # High weight → heavy atmosphere
        if weight >= 0.7:
            desc = "Interaction atmosphere feels heavy or tense"
            strength_hint = min(1.0, weight * 0.5)
            evidence = [f"Weight: {weight:.2f}"]
            results.append((desc, "contextual", strength_hint, evidence))

        # High pace → energetic interaction
        if pace >= 0.7:
            desc = "Interaction pace suggests energetic exchange"
            strength_hint = min(1.0, pace * 0.4)
            evidence = [f"Pace: {pace:.2f}"]
            results.append((desc, "contextual", strength_hint, evidence))

        # Check for neutral state (no strong signals)
        if not results:
            if 0.3 < responsiveness < 0.7 and 0.3 < weight < 0.7:
                desc = "Other party state appears neutral or ambiguous"
                strength_hint = 0.15
                evidence = [f"Responsiveness: {responsiveness:.2f}, Weight: {weight:.2f}"]
                results.append((desc, "contextual", strength_hint, evidence))

    # Handle dict input
    elif isinstance(context, dict):
        responsiveness = context.get("responsiveness", 0.5)
        weight = context.get("weight", 0.5)
        pace = context.get("pace", 0.5)

        if isinstance(responsiveness, (int, float)):
            if responsiveness >= 0.7:
                desc = "Other party appears engaged and responsive"
                strength_hint = min(1.0, responsiveness * 0.6)
                evidence = [f"Responsiveness: {responsiveness:.2f}"]
                results.append((desc, "behavioral", strength_hint, evidence))
            elif responsiveness <= 0.3:
                desc = "Other party appears disengaged or distant"
                strength_hint = min(1.0, (1.0 - responsiveness) * 0.5)
                evidence = [f"Responsiveness: {responsiveness:.2f}"]
                results.append((desc, "behavioral", strength_hint, evidence))

        if isinstance(weight, (int, float)) and weight >= 0.7:
            desc = "Interaction atmosphere feels heavy or tense"
            strength_hint = min(1.0, weight * 0.5)
            evidence = [f"Weight: {weight:.2f}"]
            results.append((desc, "contextual", strength_hint, evidence))

    return results


def extract_from_reaction_log(
    log: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract hypothesis materials from reaction log / STM.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    Reads STM/reaction log via duck typing: entries list, source_text, intent, emotion_label, valence.
    """
    if log is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Handle object with entries attribute (STM-like)
    if hasattr(log, "entries"):
        entries = getattr(log, "entries", [])
        if not entries:
            return []

        for entry in entries[:5]:  # Limit to recent entries
            intent = getattr(entry, "intent", "")
            emotion_label = getattr(entry, "emotion_label", "")
            valence = getattr(entry, "valence", 0.0)
            source_text = getattr(entry, "source_text", "")

            if intent == "question":
                desc = "Other expressed questioning intent"
                strength_hint = 0.4
                evidence = [f"Intent: question"]
                if source_text:
                    evidence.append(f"Source: {source_text[:60]}")
                results.append((desc, "behavioral", strength_hint, evidence))

            if isinstance(valence, (int, float)):
                if valence > 0.3:
                    desc = "Other party tone appears positive"
                    strength_hint = min(1.0, valence * 0.5)
                    evidence = [f"Valence: {valence:.2f}"]
                    if emotion_label:
                        evidence.append(f"Emotion: {emotion_label}")
                    results.append((desc, "behavioral", strength_hint, evidence))
                elif valence < -0.3:
                    desc = "Other party tone appears negative"
                    strength_hint = min(1.0, abs(valence) * 0.5)
                    evidence = [f"Valence: {valence:.2f}"]
                    if emotion_label:
                        evidence.append(f"Emotion: {emotion_label}")
                    results.append((desc, "behavioral", strength_hint, evidence))

    # Handle dict input
    elif isinstance(log, dict):
        entries = log.get("entries", [])
        if not entries:
            return []

        for entry in (entries[:5] if isinstance(entries, list) else []):
            if not isinstance(entry, dict):
                continue
            intent = entry.get("intent", "")
            valence = entry.get("valence", 0.0)
            emotion_label = entry.get("emotion_label", "")

            if intent == "question":
                desc = "Other expressed questioning intent"
                strength_hint = 0.4
                evidence = [f"Intent: question"]
                results.append((desc, "behavioral", strength_hint, evidence))

            if isinstance(valence, (int, float)):
                if valence > 0.3:
                    desc = "Other party tone appears positive"
                    strength_hint = min(1.0, valence * 0.5)
                    evidence = [f"Valence: {valence:.2f}"]
                    results.append((desc, "behavioral", strength_hint, evidence))
                elif valence < -0.3:
                    desc = "Other party tone appears negative"
                    strength_hint = min(1.0, abs(valence) * 0.5)
                    evidence = [f"Valence: {valence:.2f}"]
                    results.append((desc, "behavioral", strength_hint, evidence))

    return results


def extract_from_self_contrast(
    self_state: Optional[Any],
    other_signals: Optional[Any],
) -> list[tuple[str, str, float, list[str]]]:
    """
    Extract hypothesis materials from self-state / other-signals contrast.

    Returns list of (description, basis_hint, strength_hint, evidence_source_descriptions).
    self_state duck typing: description, intensity.
    other_signals duck typing: responsiveness, weight.
    """
    if self_state is None or other_signals is None:
        return []

    results: list[tuple[str, str, float, list[str]]] = []

    # Get self-state values
    self_intensity = 0.5
    self_description = ""

    if hasattr(self_state, "intensity"):
        self_intensity = getattr(self_state, "intensity", 0.5)
        self_description = getattr(self_state, "description", "")
    elif isinstance(self_state, dict):
        self_intensity = self_state.get("intensity", 0.5)
        self_description = self_state.get("description", "")

    # Get other signal values
    other_responsiveness = 0.5
    other_weight = 0.5

    if hasattr(other_signals, "responsiveness"):
        other_responsiveness = getattr(other_signals, "responsiveness", 0.5)
        other_weight = getattr(other_signals, "weight", 0.5)
    elif isinstance(other_signals, dict):
        other_responsiveness = other_signals.get("responsiveness", 0.5)
        other_weight = other_signals.get("weight", 0.5)

    # Compute divergence between self and other
    if isinstance(self_intensity, (int, float)) and isinstance(other_responsiveness, (int, float)):
        divergence = abs(self_intensity - other_responsiveness)

        if divergence >= 0.4:
            desc = (
                f"Contrast detected between self-state (intensity={self_intensity:.2f}) "
                f"and other signals (responsiveness={other_responsiveness:.2f})"
            )
            strength_hint = min(1.0, divergence * 0.7)
            evidence = [
                f"Self intensity: {self_intensity:.2f}",
                f"Other responsiveness: {other_responsiveness:.2f}",
                f"Divergence: {divergence:.2f}",
            ]
            results.append((desc, "contrast", strength_hint, evidence))

    # Weight divergence
    if isinstance(self_intensity, (int, float)) and isinstance(other_weight, (int, float)):
        weight_div = abs(self_intensity - other_weight)
        if weight_div >= 0.5:
            desc = (
                f"Self-other weight divergence: "
                f"self intensity={self_intensity:.2f}, other weight={other_weight:.2f}"
            )
            strength_hint = min(1.0, weight_div * 0.6)
            evidence = [
                f"Self intensity: {self_intensity:.2f}",
                f"Other weight: {other_weight:.2f}",
            ]
            results.append((desc, "contrast", strength_hint, evidence))

    return results


# =============================================================================
# Computation Functions (Pure)
# =============================================================================

def compute_observation_strength(links: list[ObservationLink]) -> float:
    """
    Compute integrated observation strength from multiple observation links.

    Uses weighted aggregation (not simple average).
    """
    if not links:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for i, link in enumerate(links):
        weight = 1.0 / (1.0 + i * 0.2)
        weighted_sum += link.contribution * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return min(1.0, weighted_sum / total_weight)


def detect_hypothesis_competitions(
    hypotheses: list[OtherStateHypothesis],
) -> list[tuple[str, str]]:
    """
    Detect competing hypothesis pairs.

    Two hypotheses compete if they share source type
    but have different bases (suggesting different directions).
    Uses description word overlap (Jaccard) + different basis for detection.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(hypotheses):
        for j, b in enumerate(hypotheses):
            if i >= j:
                continue

            pair_key = (a.hypothesis_id, b.hypothesis_id)
            if pair_key in seen:
                continue

            # Same source type but different basis suggests competition
            if a.source_type == b.source_type and a.basis != b.basis:
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


def determine_inference_basis(
    source_types: list[ObservationSourceType],
) -> InferenceBasis:
    """Determine InferenceBasis from a list of source types."""
    if not source_types:
        return InferenceBasis.UNDEFINED

    unique = set(source_types)
    if len(unique) > 1:
        return InferenceBasis.COMBINED

    source = source_types[0]
    mapping = {
        ObservationSourceType.EXTERNAL_CONTEXT: InferenceBasis.CONTEXTUAL,
        ObservationSourceType.REACTION_LOG: InferenceBasis.BEHAVIORAL,
        ObservationSourceType.SELF_CONTRAST: InferenceBasis.CONTRAST,
        ObservationSourceType.MIXED: InferenceBasis.COMBINED,
    }
    return mapping.get(source, InferenceBasis.UNDEFINED)


def generate_hypothesis_description(
    basis: InferenceBasis,
    source_descriptions: list[str],
) -> str:
    """Generate an integrated hypothesis description."""
    basis_labels = {
        InferenceBasis.BEHAVIORAL: "Based on observed behavior",
        InferenceBasis.CONTEXTUAL: "Based on contextual signals",
        InferenceBasis.CONTRAST: "Based on self-other contrast",
        InferenceBasis.COMBINED: "Based on multiple sources",
        InferenceBasis.UNDEFINED: "Weak hypothesis",
    }
    prefix = basis_labels.get(basis, "Hypothesis")

    if source_descriptions:
        combined = "; ".join(d[:80] for d in source_descriptions[:3])
        return f"{prefix}: {combined}"
    return prefix


def compute_self_other_boundary(
    self_description: str,
    other_hypotheses: list[OtherStateHypothesis],
) -> SelfOtherBoundary:
    """
    Compute a self-other boundary from self description and other hypotheses.

    Generates a weak boundary indicator showing where self and other diverge.
    """
    if not other_hypotheses:
        return SelfOtherBoundary(
            boundary_id=_generate_id(),
            self_description=self_description or "No self-state description available",
            other_description="No other-state hypotheses available",
            divergence=0.0,
            boundary_aspects=(),
            timestamp=str(time.time()),
        )

    # Aggregate other descriptions
    other_descs = [h.description[:80] for h in other_hypotheses[:5]]
    other_combined = "; ".join(other_descs)

    # Compute divergence from word overlap (lower overlap → higher divergence)
    self_words = set(self_description.lower().split()) if self_description else set()
    other_words = set()
    for h in other_hypotheses:
        other_words.update(h.description.lower().split())

    union = self_words | other_words
    if union:
        overlap = len(self_words & other_words) / len(union)
        divergence = 1.0 - overlap
    else:
        divergence = 0.5

    # Identify boundary aspects
    aspects: list[str] = []
    basis_types = set(h.basis.value for h in other_hypotheses)
    for bt in basis_types:
        aspects.append(f"inference_{bt}")
    if len(other_hypotheses) > 1:
        aspects.append("multiple_hypotheses")

    return SelfOtherBoundary(
        boundary_id=_generate_id(),
        self_description=self_description or "No self-state description available",
        other_description=other_combined[:200],
        divergence=round(min(1.0, max(0.0, divergence)), 4),
        boundary_aspects=tuple(aspects[:5]),
        timestamp=str(time.time()),
    )


# =============================================================================
# Other Agent Model System
# =============================================================================

class OtherAgentModelSystem:
    """
    Other Agent Model System (他者モデル)

    「相手がどう感じているか」の推測を仮説として弱く保持する。

    CRITICAL CONSTRAINTS:
    - 他者の意図・価値・信念を断定しない
    - 正誤や善悪の評価を付与しない
    - 目的や行動の最適化に結び付けない
    - 自己像や人格の方向性を固定しない
    - 候補は仮説として保持し固定しない
    - 競合する候補を許容する
    - 判断・目的・価値・責任に一切接続しない
    """

    def __init__(self, config: Optional[OtherAgentModelConfig] = None):
        self._config = config or OtherAgentModelConfig()
        self._hypotheses: list[OtherStateHypothesis] = []
        self._observation_links: list[ObservationLink] = []
        self._boundaries: list[SelfOtherBoundary] = []
        self._total_created: int = 0
        self._total_revisions: int = 0
        self._total_expirations: int = 0
        self._last_store: Optional[OtherModelStore] = None

    def observe_other(
        self,
        external_context: Optional[Any] = None,
        reaction_log: Optional[Any] = None,
        self_state: Optional[Any] = None,
    ) -> OtherModelStore:
        """
        Observe other agent state from three input sources.

        extract → Hypothesis生成 → 競合検出 → boundary計算 → 減衰 → snapshot。
        """
        current_time = str(time.time())

        # Extract from each source
        context_extracts = extract_from_external_context(external_context)
        reaction_extracts = extract_from_reaction_log(reaction_log)
        contrast_extracts = extract_from_self_contrast(self_state, external_context)

        # Create new hypotheses from extracts
        all_extracts: list[tuple[str, str, float, list[str], ObservationSourceType]] = []
        for desc, basis_hint, strength, evidence in context_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ObservationSourceType.EXTERNAL_CONTEXT))
        for desc, basis_hint, strength, evidence in reaction_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ObservationSourceType.REACTION_LOG))
        for desc, basis_hint, strength, evidence in contrast_extracts:
            all_extracts.append((desc, basis_hint, strength, evidence, ObservationSourceType.SELF_CONTRAST))

        for desc, basis_hint, strength, evidence_descs, source_type in all_extracts:
            basis_map = {
                "behavioral": InferenceBasis.BEHAVIORAL,
                "contextual": InferenceBasis.CONTEXTUAL,
                "contrast": InferenceBasis.CONTRAST,
                "combined": InferenceBasis.COMBINED,
            }
            basis = basis_map.get(basis_hint, InferenceBasis.UNDEFINED)

            hyp_id = _generate_id()

            # Generate observation links
            new_links = self._generate_observation_links(hyp_id, source_type, evidence_descs)
            link_ids = tuple(link.link_id for link in new_links)
            self._observation_links.extend(new_links)

            hypothesis = OtherStateHypothesis(
                hypothesis_id=hyp_id,
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
                undetermined_aspects=("intent_uncertain", "state_approximate"),
            )
            self._hypotheses.append(hypothesis)
            self._total_created += 1

        # Detect competitions among all hypotheses
        competition_pairs = detect_hypothesis_competitions(self._hypotheses)
        for id_a, id_b in competition_pairs:
            for i, hyp in enumerate(self._hypotheses):
                if hyp.hypothesis_id == id_a:
                    self._hypotheses[i] = hyp.with_competing(id_b)
                elif hyp.hypothesis_id == id_b:
                    self._hypotheses[i] = hyp.with_competing(id_a)

        # Compute self-other boundary
        self_desc = ""
        if self_state is not None:
            if hasattr(self_state, "description"):
                self_desc = getattr(self_state, "description", "")
            elif isinstance(self_state, dict):
                self_desc = self_state.get("description", "")

        if self._hypotheses:
            boundary = compute_self_other_boundary(self_desc, self._hypotheses)
            self._boundaries.append(boundary)
            # Enforce max boundaries
            while len(self._boundaries) > self._config.max_boundaries:
                self._boundaries.pop(0)

        # Apply natural decay to existing hypotheses
        self._apply_decay()

        # Enforce max hypotheses limit
        self._enforce_capacity()

        return self._build_store(current_time)

    def decay_hypotheses(self) -> OtherModelStore:
        """
        Apply natural decay to all hypotheses.

        freshness: base_decay_rate で減衰（reference_countで変調）
        strength: strength_decay_rate で減衰
        stale_threshold以下 AND min_strength以下 → 除去
        """
        current_time = str(time.time())
        self._apply_decay()
        return self._build_store(current_time)

    def reference_hypothesis(self, hypothesis_id: str) -> None:
        """
        Mark a hypothesis as referenced (boosts freshness).

        reference_count +1, freshness += freshness_boost_on_reference
        """
        for i, hyp in enumerate(self._hypotheses):
            if hyp.hypothesis_id == hypothesis_id:
                referenced = hyp.with_reference()
                boosted = referenced.with_freshness(
                    referenced.freshness + self._config.freshness_boost_on_reference,
                )
                self._hypotheses[i] = boosted
                return

    def revise_hypothesis(self, hypothesis_id: str, new_description: str) -> None:
        """
        Revise a hypothesis's description (仮説は固定しない).

        後からの修正を許容する。
        """
        for i, hyp in enumerate(self._hypotheses):
            if hyp.hypothesis_id == hypothesis_id:
                self._hypotheses[i] = hyp.revise(new_description)
                self._total_revisions += 1
                return

    def get_active_hypotheses(self, max_count: int = 10) -> list[OtherStateHypothesis]:
        """Get active hypotheses sorted by strength."""
        active = [
            h for h in self._hypotheses
            if h.freshness > self._config.stale_threshold
        ]
        active.sort(key=lambda h: h.strength, reverse=True)
        return active[:max_count]

    def get_store(self) -> OtherModelStore:
        """Get current store snapshot."""
        return self._build_store(str(time.time()))

    def get_last_store(self) -> Optional[OtherModelStore]:
        """Get the last stored snapshot."""
        return self._last_store

    # ----- Internal Methods -----

    def _apply_decay(self) -> None:
        """Apply decay to all hypotheses, removing expired ones."""
        new_hypotheses: list[OtherStateHypothesis] = []

        for hyp in self._hypotheses:
            ref_modifier = max(0.5, 1.0 - hyp.reference_count * 0.1)
            freshness_decay = self._config.base_decay_rate * ref_modifier
            new_freshness = hyp.freshness - freshness_decay

            new_strength = hyp.strength - self._config.strength_decay_rate

            if (new_freshness <= self._config.stale_threshold
                    and new_strength <= self._config.min_strength_for_retention):
                self._total_expirations += 1
                self._observation_links = [
                    ol for ol in self._observation_links
                    if ol.hypothesis_id != hyp.hypothesis_id
                ]
                continue

            updated = hyp.with_freshness(new_freshness).with_strength(new_strength)
            new_hypotheses.append(updated)

        self._hypotheses = new_hypotheses

    def _enforce_capacity(self) -> None:
        """Remove weakest hypotheses if over capacity."""
        while len(self._hypotheses) > self._config.max_hypotheses:
            weakest_idx = min(
                range(len(self._hypotheses)),
                key=lambda i: (self._hypotheses[i].strength, self._hypotheses[i].freshness),
            )
            removed = self._hypotheses.pop(weakest_idx)
            self._observation_links = [
                ol for ol in self._observation_links
                if ol.hypothesis_id != removed.hypothesis_id
            ]
            self._total_expirations += 1

    def _generate_observation_links(
        self,
        hypothesis_id: str,
        source_type: ObservationSourceType,
        source_descriptions: list[str],
    ) -> list[ObservationLink]:
        """Generate observation links for a new hypothesis."""
        links: list[ObservationLink] = []
        max_links = self._config.max_evidence_per_hypothesis

        for desc in source_descriptions[:max_links]:
            idx = len(links)
            contribution = max(0.1, 1.0 - idx * 0.15)

            link = ObservationLink(
                link_id=_generate_id(),
                hypothesis_id=hypothesis_id,
                source_type=source_type,
                source_description=desc,
                contribution=contribution,
            )
            links.append(link)

        return links

    def _build_store(self, current_time: str) -> OtherModelStore:
        """Build an OtherModelStore snapshot."""
        active = [
            h for h in self._hypotheses
            if h.freshness > self._config.stale_threshold
        ]
        avg_freshness = (
            sum(h.freshness for h in self._hypotheses) / len(self._hypotheses)
            if self._hypotheses else 0.0
        )
        avg_strength = (
            sum(h.strength for h in self._hypotheses) / len(self._hypotheses)
            if self._hypotheses else 0.0
        )

        competition_pairs = detect_hypothesis_competitions(self._hypotheses)

        description = _generate_store_description(
            len(self._hypotheses),
            len(active),
            avg_freshness,
            avg_strength,
            len(competition_pairs),
            self._total_expirations,
            len(self._boundaries),
        )

        store = OtherModelStore(
            hypotheses=tuple(self._hypotheses),
            observation_links=tuple(self._observation_links),
            boundaries=tuple(self._boundaries),
            total_hypotheses_created=self._total_created,
            total_revisions=self._total_revisions,
            total_expirations=self._total_expirations,
            average_freshness=round(avg_freshness, 4),
            average_strength=round(avg_strength, 4),
            active_hypothesis_count=len(active),
            competing_pair_count=len(competition_pairs),
            boundary_count=len(self._boundaries),
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
    boundary_count: int,
) -> str:
    """Generate a human-readable store description."""
    if total == 0:
        return "No other-state hypotheses formed yet."

    parts = [f"{active} active hypotheses out of {total} total"]

    if avg_strength >= 0.5:
        parts.append("generally strong hypotheses")
    elif avg_strength >= 0.2:
        parts.append("moderate strength hypotheses")
    else:
        parts.append("mostly weak hypotheses")

    if competing_pairs > 0:
        parts.append(f"{competing_pairs} competing pairs")

    if total_expirations > 0:
        parts.append(f"{total_expirations} expired")

    if boundary_count > 0:
        parts.append(f"{boundary_count} boundaries")

    parts.append(f"avg freshness: {avg_freshness:.2f}")

    return "; ".join(parts) + "."


# =============================================================================
# Integration Functions
# =============================================================================

def observe_from_chain(
    system: OtherAgentModelSystem,
    external_context: Optional[Any] = None,
    reaction_log: Optional[Any] = None,
    self_state: Optional[Any] = None,
) -> OtherModelStore:
    """
    Observe other agent from the observation chain.

    観測チェーンからの統合ヘルパー。
    各入力は読み取り専用で参照される。
    """
    return system.observe_other(
        external_context=external_context,
        reaction_log=reaction_log,
        self_state=self_state,
    )


def generate_other_model_tags(
    store: Optional[Any],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Generate tags from OtherModelStore for introspection integration.

    These tags are for introspection/awareness ONLY.
    They MUST NOT influence decisions.
    """
    tags: list[dict[str, Any]] = []

    if store is None:
        tags.append({
            "category": "OTHER_MODEL_COUNT",
            "label": "no_hypotheses",
            "description": "No other-state hypotheses formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    has_hypotheses = getattr(store, "has_hypotheses", None)
    if callable(has_hypotheses) and not has_hypotheses():
        tags.append({
            "category": "OTHER_MODEL_COUNT",
            "label": "no_hypotheses",
            "description": "No other-state hypotheses formed yet",
            "weight": 0.03 * scale,
        })
        return tags

    # Hypothesis count
    active_count = getattr(store, "active_hypothesis_count", 0)
    tags.append({
        "category": "OTHER_MODEL_COUNT",
        "label": f"hypotheses_{active_count}",
        "description": f"Other agent model holds {active_count} active hypotheses",
        "weight": 0.06 * scale,
    })

    # Average strength
    avg_strength = getattr(store, "average_strength", 0.0)
    hypotheses = getattr(store, "hypotheses", ())
    max_strength = max((h.strength for h in hypotheses), default=0.0) if hypotheses else 0.0
    strength_label = determine_strength_level(avg_strength).value
    tags.append({
        "category": "OTHER_MODEL_STRENGTH",
        "label": f"strength_{strength_label}",
        "description": f"Average hypothesis strength: {avg_strength:.2f}, max: {max_strength:.2f}",
        "weight": 0.07 * scale,
    })

    # Average freshness
    avg_freshness = getattr(store, "average_freshness", 0.0)
    freshness_label = determine_freshness_level(avg_freshness).value
    tags.append({
        "category": "OTHER_MODEL_FRESHNESS",
        "label": f"freshness_{freshness_label}",
        "description": f"Average hypothesis freshness: {avg_freshness:.2f}",
        "weight": 0.05 * scale,
    })

    # Competition
    competing_pair_count = getattr(store, "competing_pair_count", 0)
    if competing_pair_count > 0:
        tags.append({
            "category": "OTHER_MODEL_COMPETITION",
            "label": f"competing_{competing_pair_count}",
            "description": f"{competing_pair_count} pairs of competing hypotheses",
            "weight": 0.06 * scale,
        })

    # Boundary
    boundary_count = getattr(store, "boundary_count", 0)
    if boundary_count > 0:
        tags.append({
            "category": "OTHER_MODEL_BOUNDARY",
            "label": f"boundaries_{boundary_count}",
            "description": f"{boundary_count} self-other boundaries detected",
            "weight": 0.05 * scale,
        })

    # Integrated description
    description = getattr(store, "description", "")
    if description:
        tags.append({
            "category": "OTHER_MODEL_INTEGRATED",
            "label": "other_model_awareness",
            "description": description,
            "weight": 0.08 * scale,
        })

    return tags


def get_other_model_summary(store: Optional[Any]) -> str:
    """Get human-readable summary. For introspection/logging only."""
    if store is None:
        return "=== Other Agent Model State ===\nNo hypotheses formed yet."

    has_hypotheses = getattr(store, "has_hypotheses", None)
    if callable(has_hypotheses) and not has_hypotheses():
        return "=== Other Agent Model State ===\nNo hypotheses formed yet."

    hypotheses = getattr(store, "hypotheses", ())
    active_count = getattr(store, "active_hypothesis_count", 0)
    total_created = getattr(store, "total_hypotheses_created", 0)
    total_revisions = getattr(store, "total_revisions", 0)
    total_expirations = getattr(store, "total_expirations", 0)
    avg_freshness = getattr(store, "average_freshness", 0.0)
    avg_strength = getattr(store, "average_strength", 0.0)
    competing_pairs = getattr(store, "competing_pair_count", 0)
    boundary_count = getattr(store, "boundary_count", 0)
    description = getattr(store, "description", "")

    lines = [
        "=== Other Agent Model State ===",
        f"Total hypotheses: {len(hypotheses)}",
        f"Active hypotheses: {active_count}",
        f"Total created: {total_created}",
        f"Total revisions: {total_revisions}",
        f"Total expirations: {total_expirations}",
        f"Average freshness: {avg_freshness:.2f}",
        f"Average strength: {avg_strength:.2f}",
        f"Competing pairs: {competing_pairs}",
        f"Boundaries: {boundary_count}",
        "",
    ]

    sorted_hyp = sorted(hypotheses, key=lambda h: h.strength, reverse=True)
    if sorted_hyp:
        lines.append("Top hypotheses:")
        for hyp in sorted_hyp[:5]:
            lines.append(
                f"  [{hyp.source_type.value}:{hyp.basis.value}] "
                f"{hyp.description[:80]}"
                f" (strength: {hyp.get_strength_level().value}, "
                f"freshness: {hyp.get_freshness_level().value})"
            )
        lines.append("")

    lines.append(f"Integrated: {description}")
    return "\n".join(lines)


def get_other_model_for_introspection(
    store: Optional[Any],
) -> dict[str, Any]:
    """
    Get structured other model data for IntrospectionTrace integration.

    MUST NOT be used as input to decision-making systems.
    """
    if store is None:
        return {
            "has_hypotheses": False,
            "total_hypotheses": 0,
            "active_count": 0,
            "average_strength": 0.0,
            "average_freshness": 0.0,
            "source_distribution": {},
            "basis_distribution": {},
            "competing_pair_count": 0,
            "boundary_count": 0,
            "strongest_hypothesis_description": "",
            "timestamp": "",
        }

    hypotheses = getattr(store, "hypotheses", ())

    source_dist: dict[str, int] = {}
    for hyp in hypotheses:
        key = hyp.source_type.value
        source_dist[key] = source_dist.get(key, 0) + 1

    basis_dist: dict[str, int] = {}
    for hyp in hypotheses:
        key = hyp.basis.value
        basis_dist[key] = basis_dist.get(key, 0) + 1

    strongest_desc = ""
    if hypotheses:
        strongest = max(hypotheses, key=lambda h: h.strength)
        strongest_desc = strongest.description[:120]

    return {
        "has_hypotheses": len(hypotheses) > 0,
        "total_hypotheses": len(hypotheses),
        "active_count": getattr(store, "active_hypothesis_count", 0),
        "average_strength": getattr(store, "average_strength", 0.0),
        "average_freshness": getattr(store, "average_freshness", 0.0),
        "source_distribution": source_dist,
        "basis_distribution": basis_dist,
        "competing_pair_count": getattr(store, "competing_pair_count", 0),
        "boundary_count": getattr(store, "boundary_count", 0),
        "strongest_hypothesis_description": strongest_desc,
        "timestamp": getattr(store, "timestamp", ""),
    }


# =============================================================================
# Verification Functions (Test Support)
# =============================================================================

def verify_no_decision_impact(store: OtherModelStore) -> bool:
    """
    Verify that the store has no decision-impacting values.

    他者モデルは参照素材に留め、直接の意思決定経路を持たない。
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
                "total_hypotheses_created", "total_revisions",
                "total_expirations", "average_freshness", "average_strength",
                "active_hypothesis_count", "competing_pair_count",
                "boundary_count",
            ):
                continue
            return False

    return True


def verify_no_goal_generation(system: OtherAgentModelSystem) -> bool:
    """
    Verify the system has no goal-generating methods.

    他者モデルから目標を生成しない。
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


def verify_read_only_principle(system: OtherAgentModelSystem) -> bool:
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


def verify_no_value_modification(system: OtherAgentModelSystem) -> bool:
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


def verify_no_intent_assertion(system: OtherAgentModelSystem) -> bool:
    """
    Verify the system has no intent-asserting methods.

    他者の意図を断定するメソッドを持たない。
    """
    forbidden = [
        "assert_intent", "determine_intent", "confirm_intent",
        "judge_intent", "classify_intent",
        "assert_belief", "determine_belief",
        "assert_value", "determine_value",
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

def create_config(**kwargs: Any) -> OtherAgentModelConfig:
    """Create a custom configuration."""
    return OtherAgentModelConfig(**kwargs)


def create_empty_store() -> OtherModelStore:
    """Create an empty other model store."""
    return OtherModelStore(
        hypotheses=(),
        observation_links=(),
        boundaries=(),
        total_hypotheses_created=0,
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.0,
        average_strength=0.0,
        active_hypothesis_count=0,
        competing_pair_count=0,
        boundary_count=0,
        timestamp=str(time.time()),
        description="No other-state hypotheses formed yet.",
    )


def create_system(
    config: Optional[OtherAgentModelConfig] = None,
) -> OtherAgentModelSystem:
    """Create a new OtherAgentModelSystem."""
    return OtherAgentModelSystem(config=config)


def save_other_model_state(
    store: OtherModelStore,
    filepath: str,
) -> None:
    """Save other model state to a JSON file."""
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_other_model_state(filepath: str) -> OtherModelStore:
    """Load other model state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return OtherModelStore.from_dict(data)
