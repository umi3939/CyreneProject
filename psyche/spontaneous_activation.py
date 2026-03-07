"""
psyche/spontaneous_activation.py - 自発性の追加（外部入力非依存の起動経路）

内部状態から起動可能な経路を追加し、外部入力がない局面でも内部状態の
変化を処理可能にする。

設計原則 (design_spontaneous_activation.md 準拠):
- 目的は外部入力非依存の起動経路追加であり、出力頻度増加そのものではない
- 外部入力経路を置き換えない
- 内部動機を単一路線へ固定しない
- 起動成立だけで行動内容を確定しない
- 継続駆動を無制限化しない
- 出力は起動候補情報としてのみ流し、判断・評価・行動決定を直接起動しない
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

class ActivationSourceType(Enum):
    """起動候補の断面種別（8値）。"""
    INTRINSIC_MOTIVATION = "intrinsic_motivation"
    DIRECTION_VECTOR = "direction_vector"
    UNFINISHED_INTENT = "unfinished_intent"
    MEMORY_ECHO = "memory_echo"
    EMOTIONAL_TRANSITION = "emotional_transition"
    RESPONSIBILITY = "responsibility"
    RECENT_ACTION = "recent_action"
    EXTERNAL_INPUT_ABSENCE = "external_input_absence"


class ActivationFreshness(Enum):
    """起動候補の鮮度。"""
    FRESH = "fresh"
    RECENT = "recent"
    AGING = "aging"
    STALE = "stale"
    FADED = "faded"


class CandidateStatus(Enum):
    """起動候補の状態。"""
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    STANDBY = "standby"
    EXPIRED = "expired"


class SuppressionMode(Enum):
    """起動抑制の状態（不可逆禁止ではなく可逆状態）。"""
    NONE = "none"
    ACTIVE = "active"
    DECAYING = "decaying"
    RELEASED = "released"


class ConflictResolution(Enum):
    """競合解決結果。"""
    PARALLEL = "parallel"
    ADOPTED = "adopted"
    UNADOPTED = "unadopted"
    RECYCLED = "recycled"


# =============================================================================
# Helpers
# =============================================================================

_FRESHNESS_ORDER = [
    ActivationFreshness.FRESH,
    ActivationFreshness.RECENT,
    ActivationFreshness.AGING,
    ActivationFreshness.STALE,
    ActivationFreshness.FADED,
]


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ActivationFragment:
    """1つの断面から抽出された起動圧力断片。"""
    fragment_id: str = ""
    source_type: ActivationSourceType = ActivationSourceType.INTRINSIC_MOTIVATION
    value: float = 0.0
    description: str = ""
    freshness: ActivationFreshness = ActivationFreshness.FRESH
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.fragment_id:
            self.fragment_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "source_type": self.source_type.value,
            "value": self.value,
            "description": self.description,
            "freshness": self.freshness.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationFragment":
        return cls(
            fragment_id=d.get("fragment_id", ""),
            source_type=ActivationSourceType(
                d.get("source_type", "intrinsic_motivation")
            ),
            value=d.get("value", 0.0),
            description=d.get("description", ""),
            freshness=ActivationFreshness(d.get("freshness", "fresh")),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class ActivationCandidate:
    """複数断面交差から形成された起動候補。"""
    candidate_id: str = ""
    source_fragment_ids: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    intersection_count: int = 0
    activation_strength: float = 0.0
    description: str = ""
    status: CandidateStatus = CandidateStatus.ACTIVE
    freshness: ActivationFreshness = ActivationFreshness.FRESH
    timestamp: float = 0.0
    cycle_id: int = 0

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_fragment_ids": list(self.source_fragment_ids),
            "source_types": list(self.source_types),
            "intersection_count": self.intersection_count,
            "activation_strength": self.activation_strength,
            "description": self.description,
            "status": self.status.value,
            "freshness": self.freshness.value,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationCandidate":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            source_fragment_ids=d.get("source_fragment_ids", []),
            source_types=d.get("source_types", []),
            intersection_count=d.get("intersection_count", 0),
            activation_strength=d.get("activation_strength", 0.0),
            description=d.get("description", ""),
            status=CandidateStatus(d.get("status", "active")),
            freshness=ActivationFreshness(d.get("freshness", "fresh")),
            timestamp=d.get("timestamp", 0.0),
            cycle_id=d.get("cycle_id", 0),
        )


@dataclass
class ActivationRationale:
    """起動候補の根拠情報。"""
    candidate_id: str = ""
    contributing_sources: list[str] = field(default_factory=list)
    strength_breakdown: dict[str, float] = field(default_factory=dict)
    continuous_diff_reference: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "contributing_sources": list(self.contributing_sources),
            "strength_breakdown": dict(self.strength_breakdown),
            "continuous_diff_reference": self.continuous_diff_reference,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationRationale":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            contributing_sources=d.get("contributing_sources", []),
            strength_breakdown=d.get("strength_breakdown", {}),
            continuous_diff_reference=d.get("continuous_diff_reference", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class SuppressionEntry:
    """起動抑制状態（可逆）。"""
    candidate_id: str = ""
    reason: str = ""
    mode: SuppressionMode = SuppressionMode.NONE
    timestamp: float = 0.0
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "mode": self.mode.value,
            "timestamp": self.timestamp,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SuppressionEntry":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            reason=d.get("reason", ""),
            mode=SuppressionMode(d.get("mode", "none")),
            timestamp=d.get("timestamp", 0.0),
            reversible=d.get("reversible", True),
        )


@dataclass
class StandbyEntry:
    """再起動待機状態。連続起動偏り緩和に使用。希薄化対象。"""
    candidate_id: str = ""
    original_strength: float = 0.0
    standby_reason: str = ""
    timestamp: float = 0.0
    freshness: ActivationFreshness = ActivationFreshness.FRESH

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "original_strength": self.original_strength,
            "standby_reason": self.standby_reason,
            "timestamp": self.timestamp,
            "freshness": self.freshness.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StandbyEntry":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            original_strength=d.get("original_strength", 0.0),
            standby_reason=d.get("standby_reason", ""),
            timestamp=d.get("timestamp", 0.0),
            freshness=ActivationFreshness(d.get("freshness", "fresh")),
        )


@dataclass
class ConflictHistoryEntry:
    """起動候補間の競合履歴。"""
    candidate_id_a: str = ""
    candidate_id_b: str = ""
    resolution: str = "parallel"
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id_a": self.candidate_id_a,
            "candidate_id_b": self.candidate_id_b,
            "resolution": self.resolution,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConflictHistoryEntry":
        return cls(
            candidate_id_a=d.get("candidate_id_a", ""),
            candidate_id_b=d.get("candidate_id_b", ""),
            resolution=d.get("resolution", "parallel"),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class UnadoptedHistoryEntry:
    """未採択候補の履歴。再浮上経路として保持。"""
    candidate_id: str = ""
    source_types: list[str] = field(default_factory=list)
    original_strength: float = 0.0
    reason: str = ""
    timestamp: float = 0.0
    freshness: ActivationFreshness = ActivationFreshness.FRESH

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_types": list(self.source_types),
            "original_strength": self.original_strength,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "freshness": self.freshness.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnadoptedHistoryEntry":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            source_types=d.get("source_types", []),
            original_strength=d.get("original_strength", 0.0),
            reason=d.get("reason", ""),
            timestamp=d.get("timestamp", 0.0),
            freshness=ActivationFreshness(d.get("freshness", "fresh")),
        )


@dataclass
class ContinuousActivationEntry:
    """連続起動履歴。時間的性質として生成・変化・減衰。"""
    candidate_id: str = ""
    source_types: list[str] = field(default_factory=list)
    cycle_id: int = 0
    timestamp: float = 0.0
    freshness: ActivationFreshness = ActivationFreshness.FRESH

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_types": list(self.source_types),
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "freshness": self.freshness.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContinuousActivationEntry":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            source_types=d.get("source_types", []),
            cycle_id=d.get("cycle_id", 0),
            timestamp=d.get("timestamp", 0.0),
            freshness=ActivationFreshness(d.get("freshness", "fresh")),
        )


@dataclass
class SpontaneousDecayEntry:
    """希薄化履歴。"""
    candidate_id: str = ""
    original_freshness: ActivationFreshness = ActivationFreshness.FRESH
    decayed_freshness: ActivationFreshness = ActivationFreshness.FRESH
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "original_freshness": self.original_freshness.value,
            "decayed_freshness": self.decayed_freshness.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpontaneousDecayEntry":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            original_freshness=ActivationFreshness(
                d.get("original_freshness", "fresh")
            ),
            decayed_freshness=ActivationFreshness(
                d.get("decayed_freshness", "fresh")
            ),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class SpontaneousConfig:
    """自発起動の設定。"""
    min_intersection_count: int = 2
    fragment_threshold: float = 0.3
    max_candidates: int = 10
    max_output_candidates: int = 5
    max_history: int = 50
    max_suppression_history: int = 30
    max_unadopted_history: int = 30
    consecutive_adoption_suppression_count: int = 3
    activation_cooldown_cycles: int = 3
    overdense_threshold: int = 5
    single_series_dominance_threshold: float = 0.7
    freshness_decay_rate: float = 0.05
    external_input_recency_threshold: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_intersection_count": self.min_intersection_count,
            "fragment_threshold": self.fragment_threshold,
            "max_candidates": self.max_candidates,
            "max_output_candidates": self.max_output_candidates,
            "max_history": self.max_history,
            "max_suppression_history": self.max_suppression_history,
            "max_unadopted_history": self.max_unadopted_history,
            "consecutive_adoption_suppression_count": self.consecutive_adoption_suppression_count,
            "activation_cooldown_cycles": self.activation_cooldown_cycles,
            "overdense_threshold": self.overdense_threshold,
            "single_series_dominance_threshold": self.single_series_dominance_threshold,
            "freshness_decay_rate": self.freshness_decay_rate,
            "external_input_recency_threshold": self.external_input_recency_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpontaneousConfig":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class SpontaneousState:
    """自発起動の全状態。"""
    config: SpontaneousConfig = field(default_factory=SpontaneousConfig)
    candidates: list[ActivationCandidate] = field(default_factory=list)
    rationales: list[ActivationRationale] = field(default_factory=list)
    suppression_entries: list[SuppressionEntry] = field(default_factory=list)
    standby_entries: list[StandbyEntry] = field(default_factory=list)
    conflict_history: list[ConflictHistoryEntry] = field(default_factory=list)
    unadopted_history: list[UnadoptedHistoryEntry] = field(
        default_factory=list
    )
    continuous_activation_history: list[ContinuousActivationEntry] = field(
        default_factory=list
    )
    decay_history: list[SpontaneousDecayEntry] = field(default_factory=list)
    cycle_count: int = 0
    total_activations: int = 0
    consecutive_activation_count: int = 0
    last_activation_cycle: int = 0
    last_external_input_time: float = 0.0
    recent_adopted_series: list[list[str]] = field(default_factory=list)
    cooldown_remaining: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "rationales": [r.to_dict() for r in self.rationales],
            "suppression_entries": [
                s.to_dict() for s in self.suppression_entries
            ],
            "standby_entries": [s.to_dict() for s in self.standby_entries],
            "conflict_history": [
                c.to_dict() for c in self.conflict_history
            ],
            "unadopted_history": [
                u.to_dict() for u in self.unadopted_history
            ],
            "continuous_activation_history": [
                c.to_dict() for c in self.continuous_activation_history
            ],
            "decay_history": [d.to_dict() for d in self.decay_history],
            "cycle_count": self.cycle_count,
            "total_activations": self.total_activations,
            "consecutive_activation_count": self.consecutive_activation_count,
            "last_activation_cycle": self.last_activation_cycle,
            "last_external_input_time": self.last_external_input_time,
            "recent_adopted_series": [
                list(s) for s in self.recent_adopted_series
            ],
            "cooldown_remaining": self.cooldown_remaining,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpontaneousState":
        cfg = SpontaneousConfig.from_dict(d.get("config", {}))
        return cls(
            config=cfg,
            candidates=[
                ActivationCandidate.from_dict(c)
                for c in d.get("candidates", [])
            ],
            rationales=[
                ActivationRationale.from_dict(r)
                for r in d.get("rationales", [])
            ],
            suppression_entries=[
                SuppressionEntry.from_dict(s)
                for s in d.get("suppression_entries", [])
            ],
            standby_entries=[
                StandbyEntry.from_dict(s)
                for s in d.get("standby_entries", [])
            ],
            conflict_history=[
                ConflictHistoryEntry.from_dict(c)
                for c in d.get("conflict_history", [])
            ],
            unadopted_history=[
                UnadoptedHistoryEntry.from_dict(u)
                for u in d.get("unadopted_history", [])
            ],
            continuous_activation_history=[
                ContinuousActivationEntry.from_dict(c)
                for c in d.get("continuous_activation_history", [])
            ],
            decay_history=[
                SpontaneousDecayEntry.from_dict(de)
                for de in d.get("decay_history", [])
            ],
            cycle_count=d.get("cycle_count", 0),
            total_activations=d.get("total_activations", 0),
            consecutive_activation_count=d.get(
                "consecutive_activation_count", 0
            ),
            last_activation_cycle=d.get("last_activation_cycle", 0),
            last_external_input_time=d.get("last_external_input_time", 0.0),
            recent_adopted_series=d.get("recent_adopted_series", []),
            cooldown_remaining=d.get("cooldown_remaining", 0),
        )


@dataclass
class ActivationResult:
    """process()の出力。起動候補情報のみ。行動決定を含まない。"""
    candidates: list[ActivationCandidate] = field(default_factory=list)
    rationales: list[ActivationRationale] = field(default_factory=list)
    conflicts: list[ConflictHistoryEntry] = field(default_factory=list)
    should_activate: bool = False
    suppression_active: bool = False
    cooldown_active: bool = False
    overdense_warning: bool = False
    single_series_warning: bool = False
    unadopted_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "rationales": [r.to_dict() for r in self.rationales],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "should_activate": self.should_activate,
            "suppression_active": self.suppression_active,
            "cooldown_active": self.cooldown_active,
            "overdense_warning": self.overdense_warning,
            "single_series_warning": self.single_series_warning,
            "unadopted_count": self.unadopted_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationResult":
        return cls(
            candidates=[
                ActivationCandidate.from_dict(c)
                for c in d.get("candidates", [])
            ],
            rationales=[
                ActivationRationale.from_dict(r)
                for r in d.get("rationales", [])
            ],
            conflicts=[
                ConflictHistoryEntry.from_dict(c)
                for c in d.get("conflicts", [])
            ],
            should_activate=d.get("should_activate", False),
            suppression_active=d.get("suppression_active", False),
            cooldown_active=d.get("cooldown_active", False),
            overdense_warning=d.get("overdense_warning", False),
            single_series_warning=d.get("single_series_warning", False),
            unadopted_count=d.get("unadopted_count", 0),
        )


# =============================================================================
# Stage 1: 起動候補抽出 (8 cross-section extraction, pure, duck-typed)
# =============================================================================

def extract_intrinsic_motivation(psyche: Any) -> ActivationFragment:
    """内的動機断面: intrinsic_motivation の活性度。"""
    value = 0.0
    desc_parts: list[str] = []

    drives = getattr(psyche, "drives", None)
    if drives is not None:
        social = getattr(drives, "social", 0.5)
        curiosity = getattr(drives, "curiosity", 0.5)
        expression = getattr(drives, "expression", 0.5)
        drive_avg = (social + curiosity + expression) / 3.0
        value = _clamp(drive_avg)
        desc_parts.append(
            f"drives(s={social:.2f},c={curiosity:.2f},e={expression:.2f})"
        )

    return ActivationFragment(
        source_type=ActivationSourceType.INTRINSIC_MOTIVATION,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_drives",
    )


def extract_direction_vector(psyche: Any) -> ActivationFragment:
    """方向断面: proto_goal_vector の大きさ。"""
    value = 0.0
    desc_parts: list[str] = []

    # proto_goal_vector state is typically accessed via orchestrator cache
    # Duck-type: look for attributes that represent goal vector magnitude
    vectors = getattr(psyche, "vectors", None)
    if vectors and hasattr(vectors, "__iter__"):
        magnitudes = []
        for v in vectors:
            mag = getattr(v, "magnitude", 0.0)
            if mag > 0:
                magnitudes.append(mag)
        if magnitudes:
            value = _clamp(max(magnitudes))
            desc_parts.append(f"max_magnitude={value:.2f}")

    return ActivationFragment(
        source_type=ActivationSourceType.DIRECTION_VECTOR,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_vectors",
    )


def extract_unfinished_intent(psyche: Any) -> ActivationFragment:
    """未完了意図断面: transient_goal / scoped_goal の残存。"""
    value = 0.0
    desc_parts: list[str] = []

    # Check for active transient goal
    active_goal = getattr(psyche, "active_goal", None)
    if active_goal is not None:
        strength = getattr(active_goal, "selection_strength", 0.0)
        if strength > 0:
            value = max(value, _clamp(strength))
            desc_parts.append(f"transient_strength={strength:.2f}")

    # Check for scoped goal
    scoped = getattr(psyche, "scoped_goal", None)
    if scoped is not None:
        sc_strength = getattr(scoped, "strength", 0.0)
        sc_status = getattr(scoped, "status", None)
        if sc_status is not None:
            status_val = getattr(sc_status, "value", str(sc_status))
            if status_val == "active" and sc_strength > 0:
                value = max(value, _clamp(sc_strength))
                desc_parts.append(f"scoped_strength={sc_strength:.2f}")

    return ActivationFragment(
        source_type=ActivationSourceType.UNFINISHED_INTENT,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_intent",
    )


def extract_memory_echo(stm: Any, memories: Any) -> ActivationFragment:
    """記憶残響断面: STMの残留感情・未処理記憶。"""
    value = 0.0
    desc_parts: list[str] = []

    # STM residue
    if stm is not None:
        entries = getattr(stm, "entries", [])
        if entries and hasattr(entries, "__len__"):
            residues = []
            for entry in entries:
                rw = getattr(entry, "residue_weight", 0.0)
                if rw > 0:
                    residues.append(rw)
            if residues:
                avg_residue = sum(residues) / len(residues)
                value = _clamp(avg_residue)
                desc_parts.append(f"stm_residue={avg_residue:.2f}")

        continuity = getattr(stm, "context_continuity_score", 0.0)
        if continuity > 0:
            value = max(value, _clamp(continuity * 0.5))

    # Recalled memories emotional intensity
    if memories and hasattr(memories, "__iter__"):
        mem_count = 0
        for m in memories:
            if isinstance(m, dict):
                imp = m.get("importance", 0)
                if imp and imp >= 3:
                    mem_count += 1
        if mem_count > 0:
            mem_signal = _clamp(mem_count * 0.2)
            value = max(value, mem_signal)
            desc_parts.append(f"high_importance_memories={mem_count}")

    return ActivationFragment(
        source_type=ActivationSourceType.MEMORY_ECHO,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_echo",
    )


def extract_emotional_transition(
    psyche: Any, dynamics: Any
) -> ActivationFragment:
    """感情推移断面: 感情の変化量。"""
    value = 0.0
    desc_parts: list[str] = []

    emotions = getattr(psyche, "emotions", None)
    if emotions is not None:
        emo_dict = {}
        if hasattr(emotions, "as_dict"):
            emo_dict = emotions.as_dict()
        elif hasattr(emotions, "__dict__"):
            emo_dict = {
                k: v for k, v in emotions.__dict__.items()
                if isinstance(v, (int, float))
            }
        if emo_dict:
            intensities = [abs(v) for v in emo_dict.values()]
            max_intensity = max(intensities) if intensities else 0.0
            value = _clamp(max_intensity)
            desc_parts.append(f"max_emotion={max_intensity:.2f}")

    mood = getattr(psyche, "mood", None)
    if mood is not None:
        arousal = getattr(mood, "arousal", 0.0)
        if arousal > 0.5:
            value = max(value, _clamp(arousal * 0.8))
            desc_parts.append(f"arousal={arousal:.2f}")

    return ActivationFragment(
        source_type=ActivationSourceType.EMOTIONAL_TRANSITION,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_transition",
    )


def extract_responsibility(psyche: Any) -> ActivationFragment:
    """責任断面: 未解消の責任の蓄積。"""
    value = 0.0
    desc_parts: list[str] = []

    # responsibility_dispersion state
    resp_units = getattr(psyche, "responsibility_units", None)
    if resp_units and hasattr(resp_units, "__iter__"):
        weights = []
        for u in resp_units:
            w = getattr(u, "weight", 0.0)
            if w > 0:
                weights.append(w)
        if weights:
            total_weight = sum(weights)
            value = _clamp(total_weight * 0.3)
            desc_parts.append(
                f"resp_units={len(weights)},total_w={total_weight:.2f}"
            )
    elif isinstance(resp_units, dict):
        count = len(resp_units)
        if count > 0:
            value = _clamp(count * 0.1)
            desc_parts.append(f"resp_entries={count}")

    return ActivationFragment(
        source_type=ActivationSourceType.RESPONSIBILITY,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_responsibility",
    )


def extract_recent_action(recent_actions: Any) -> ActivationFragment:
    """直近行動履歴断面: 直近行動の残響。"""
    value = 0.0
    desc_parts: list[str] = []

    if recent_actions and hasattr(recent_actions, "__len__"):
        count = len(recent_actions)
        if count > 0:
            value = _clamp(count * 0.15)
            desc_parts.append(f"recent_count={count}")
            # Check for patterns (consecutive silences suggest need to speak)
            silences = 0
            for action in recent_actions:
                if isinstance(action, str):
                    if "silence" in action.lower() or "pass" in action.lower():
                        silences += 1
                elif isinstance(action, dict):
                    if action.get("silence") or action.get("type") == "silence":
                        silences += 1
            if silences > 0:
                silence_ratio = silences / count
                value = max(value, _clamp(silence_ratio * 0.5))
                desc_parts.append(f"silence_ratio={silence_ratio:.2f}")

    return ActivationFragment(
        source_type=ActivationSourceType.RECENT_ACTION,
        value=value,
        description=" ".join(desc_parts) if desc_parts else "no_actions",
    )


def extract_external_input_absence(
    has_external_input: bool,
    last_external_input_time: float,
    config: Optional[SpontaneousConfig] = None,
) -> ActivationFragment:
    """外部入力有無断面: 外部入力不在時間に応じた起動圧。"""
    cfg = config or SpontaneousConfig()
    now = time.time()

    if has_external_input:
        return ActivationFragment(
            source_type=ActivationSourceType.EXTERNAL_INPUT_ABSENCE,
            value=0.0,
            description="external_input_present",
        )

    if last_external_input_time <= 0:
        absence_time = cfg.external_input_recency_threshold
    else:
        absence_time = now - last_external_input_time

    if absence_time < 5.0:
        value = 0.1
    elif absence_time < cfg.external_input_recency_threshold:
        value = _clamp(absence_time / cfg.external_input_recency_threshold)
    else:
        value = 1.0

    return ActivationFragment(
        source_type=ActivationSourceType.EXTERNAL_INPUT_ABSENCE,
        value=value,
        description=f"absence={absence_time:.1f}s",
    )


def extract_all_fragments(
    psyche: Any,
    dynamics: Any,
    stm: Any,
    memories: Any,
    recent_actions: Any,
    has_external_input: bool,
    last_external_input_time: float,
    config: Optional[SpontaneousConfig] = None,
) -> list[ActivationFragment]:
    """8断面すべてから起動圧力断片を抽出する。"""
    return [
        extract_intrinsic_motivation(psyche),
        extract_direction_vector(psyche),
        extract_unfinished_intent(psyche),
        extract_memory_echo(stm, memories),
        extract_emotional_transition(psyche, dynamics),
        extract_responsibility(psyche),
        extract_recent_action(recent_actions),
        extract_external_input_absence(
            has_external_input, last_external_input_time, config
        ),
    ]


# =============================================================================
# Stage 2: 起動条件整列 (Condition Alignment)
# =============================================================================

def form_candidates(
    fragments: list[ActivationFragment],
    config: Optional[SpontaneousConfig] = None,
    cycle_id: int = 0,
) -> list[ActivationCandidate]:
    """複数断面交差で起動候補を形成する。

    単一断面起点ではなく複数断面交差で成立させ、
    単独要素の支配を抑える。
    """
    cfg = config or SpontaneousConfig()
    threshold = cfg.fragment_threshold
    min_count = cfg.min_intersection_count

    # 閾値を超える断片を収集
    qualifying = [f for f in fragments if f.value >= threshold]

    if len(qualifying) < min_count:
        return []

    # 全交差断面からメイン候補を1つ形成
    source_ids = [f.fragment_id for f in qualifying]
    source_types = [f.source_type.value for f in qualifying]
    strength = sum(f.value for f in qualifying) / len(qualifying)

    desc_parts = [
        f"{f.source_type.value}={f.value:.2f}" for f in qualifying
    ]

    candidates: list[ActivationCandidate] = []

    main_candidate = ActivationCandidate(
        source_fragment_ids=source_ids,
        source_types=source_types,
        intersection_count=len(qualifying),
        activation_strength=_clamp(strength),
        description="cross-section: " + ", ".join(desc_parts),
        status=CandidateStatus.ACTIVE,
        cycle_id=cycle_id,
    )
    candidates.append(main_candidate)

    # サブセットからの追加候補（3断面以上の場合）
    if len(qualifying) >= 3:
        # 上位2断面のみのサブ候補
        sorted_frags = sorted(qualifying, key=lambda f: f.value, reverse=True)
        top_frags = sorted_frags[:2]
        sub_strength = sum(f.value for f in top_frags) / len(top_frags)
        sub_types = [f.source_type.value for f in top_frags]

        sub_candidate = ActivationCandidate(
            source_fragment_ids=[f.fragment_id for f in top_frags],
            source_types=sub_types,
            intersection_count=len(top_frags),
            activation_strength=_clamp(sub_strength),
            description="sub-cross: " + ", ".join(
                f"{f.source_type.value}={f.value:.2f}" for f in top_frags
            ),
            status=CandidateStatus.ACTIVE,
            cycle_id=cycle_id,
        )
        candidates.append(sub_candidate)

    # 上限制御
    return candidates[: cfg.max_candidates]


def align_conditions(
    candidates: list[ActivationCandidate],
    continuous_history: list[ContinuousActivationEntry],
    config: Optional[SpontaneousConfig] = None,
) -> tuple[list[ActivationCandidate], list[ActivationRationale]]:
    """内部状態の連続差分を参照し、単回変動のみで起動が恒常化しないようにする。"""
    cfg = config or SpontaneousConfig()
    aligned: list[ActivationCandidate] = []
    rationales: list[ActivationRationale] = []

    # 直近の連続起動パターンを分析
    recent_series_types: list[set[str]] = []
    for entry in continuous_history[-cfg.consecutive_adoption_suppression_count:]:
        recent_series_types.append(set(entry.source_types))

    for candidate in candidates:
        current_types = set(candidate.source_types)

        # 連続差分: 直近の起動と同一パターンかどうか
        overlap_count = 0
        for prev_types in recent_series_types:
            if current_types == prev_types:
                overlap_count += 1

        # 差分参照値: 0.0（完全同一パターン反復）〜 1.0（新規パターン）
        if recent_series_types:
            diff_ref = 1.0 - (overlap_count / len(recent_series_types))
        else:
            diff_ref = 1.0

        # 連続同一パターンなら強度を減衰
        adjusted_strength = candidate.activation_strength * (0.5 + 0.5 * diff_ref)

        aligned_candidate = ActivationCandidate(
            candidate_id=candidate.candidate_id,
            source_fragment_ids=candidate.source_fragment_ids,
            source_types=candidate.source_types,
            intersection_count=candidate.intersection_count,
            activation_strength=_clamp(adjusted_strength),
            description=candidate.description,
            status=candidate.status,
            freshness=candidate.freshness,
            timestamp=candidate.timestamp,
            cycle_id=candidate.cycle_id,
        )
        aligned.append(aligned_candidate)

        rationales.append(ActivationRationale(
            candidate_id=candidate.candidate_id,
            contributing_sources=list(candidate.source_types),
            # strength_breakdown is structurally present for diagnostic use but
            # initialized to zero; per-source strength values are not propagated
            # in the alignment stage (the candidate's aggregate strength is used instead).
            strength_breakdown={
                st: 0.0 for st in candidate.source_types
            },
            continuous_diff_reference=diff_ref,
            timestamp=time.time(),
        ))

    return aligned, rationales


# =============================================================================
# Stage 3: 競合整理 (Conflict Resolution)
# =============================================================================

def resolve_conflicts(
    candidates: list[ActivationCandidate],
    config: Optional[SpontaneousConfig] = None,
) -> tuple[list[ActivationCandidate], list[ActivationCandidate], list[ConflictHistoryEntry]]:
    """複数起動候補を並立保持し、未採択候補を消去せず次回候補化へ戻す。

    Returns:
        (adopted, unadopted, conflict_history)
    """
    cfg = config or SpontaneousConfig()

    if not candidates:
        return [], [], []

    # 強度順にソート（ただし固定優先列は持たない）
    sorted_cands = sorted(
        candidates, key=lambda c: c.activation_strength, reverse=True
    )

    adopted: list[ActivationCandidate] = []
    unadopted: list[ActivationCandidate] = []
    conflicts: list[ConflictHistoryEntry] = []

    # 上位を採用候補として保持（並立）
    for i, cand in enumerate(sorted_cands):
        if i < cfg.max_output_candidates:
            adopted.append(cand)
        else:
            # 未採択だが消去しない
            unadopted_cand = ActivationCandidate(
                candidate_id=cand.candidate_id,
                source_fragment_ids=cand.source_fragment_ids,
                source_types=cand.source_types,
                intersection_count=cand.intersection_count,
                activation_strength=cand.activation_strength,
                description=cand.description,
                status=CandidateStatus.STANDBY,
                freshness=cand.freshness,
                timestamp=cand.timestamp,
                cycle_id=cand.cycle_id,
            )
            unadopted.append(unadopted_cand)

    # 競合記録（採用候補同士の並立）
    for i in range(len(adopted)):
        for j in range(i + 1, len(adopted)):
            conflicts.append(ConflictHistoryEntry(
                candidate_id_a=adopted[i].candidate_id,
                candidate_id_b=adopted[j].candidate_id,
                resolution="parallel",
                timestamp=time.time(),
            ))

    return adopted, unadopted, conflicts


# =============================================================================
# Stage 4: 起動可否判定用情報化 (Activation Feasibility)
# =============================================================================

def check_activation_feasibility(
    candidates: list[ActivationCandidate],
    cooldown_remaining: int,
    consecutive_count: int,
    has_external_input: bool,
    config: Optional[SpontaneousConfig] = None,
) -> tuple[bool, bool, bool]:
    """起動可否を判定するための情報を生成する。

    Returns:
        (should_activate, cooldown_active, overdense_warning)
    """
    cfg = config or SpontaneousConfig()

    if not candidates:
        return False, cooldown_remaining > 0, False

    # クールダウン中は起動しない
    if cooldown_remaining > 0:
        return False, True, False

    # 外部入力存在時は外部入力駆動と競合比較
    # （内部起動のみが常時優先される状態を避ける）
    if has_external_input:
        # 外部入力があれば内部起動は控えめに
        best_strength = max(c.activation_strength for c in candidates)
        if best_strength < 0.7:
            return False, False, False

    # 過密化チェック
    overdense = consecutive_count >= cfg.overdense_threshold

    # 最良候補の強度確認
    best = max(candidates, key=lambda c: c.activation_strength)
    should = best.activation_strength >= cfg.fragment_threshold

    return should, False, overdense


# =============================================================================
# Safety Valves
# =============================================================================

def suppress_consecutive_series(
    candidates: list[ActivationCandidate],
    recent_adopted_series: list[list[str]],
    config: Optional[SpontaneousConfig] = None,
) -> tuple[list[ActivationCandidate], list[ActivationCandidate], list[SuppressionEntry]]:
    """直近採択系列の連続再採択を抑制し、未採択系列の再浮上経路を維持する。

    Returns:
        (passed, suppressed, suppression_entries)
    """
    cfg = config or SpontaneousConfig()
    n = cfg.consecutive_adoption_suppression_count
    recent = recent_adopted_series[-n:]

    if len(recent) < n:
        return list(candidates), [], []

    # 直近N回が全て同一系列かチェック
    all_same = all(set(s) == set(recent[0]) for s in recent) if recent else False

    if not all_same:
        return list(candidates), [], []

    dominant_series = set(recent[0])
    passed: list[ActivationCandidate] = []
    suppressed: list[ActivationCandidate] = []
    entries: list[SuppressionEntry] = []

    suppressed_one = False
    for cand in candidates:
        if set(cand.source_types) == dominant_series and not suppressed_one:
            suppressed.append(cand)
            entries.append(SuppressionEntry(
                candidate_id=cand.candidate_id,
                reason=f"consecutive_series_suppression",
                mode=SuppressionMode.ACTIVE,
                reversible=True,
            ))
            suppressed_one = True
        else:
            passed.append(cand)

    return passed, suppressed, entries


def apply_overdense_cooldown(
    consecutive_count: int,
    config: Optional[SpontaneousConfig] = None,
) -> int:
    """内部起動が過密化した場合、抑制状態を有効化する。

    Returns:
        cooldown_cycles to set
    """
    cfg = config or SpontaneousConfig()
    if consecutive_count >= cfg.overdense_threshold:
        return cfg.activation_cooldown_cycles
    return 0


def restore_candidate_diversity(
    candidates: list[ActivationCandidate],
    unadopted_history: list[UnadoptedHistoryEntry],
    config: Optional[SpontaneousConfig] = None,
) -> tuple[list[ActivationCandidate], list[UnadoptedHistoryEntry], bool]:
    """起動候補が単線化した場合は代替系列を補充し複線候補へ復帰する。

    Returns:
        (result_candidates, updated_unadopted, single_series_warning)
    """
    cfg = config or SpontaneousConfig()

    if not candidates:
        return candidates, unadopted_history, False

    # 候補の系列多様性を確認
    all_types: list[set[str]] = [set(c.source_types) for c in candidates]
    unique_series = len(set(frozenset(t) for t in all_types))

    single_series = unique_series <= 1 and len(candidates) > 0

    if single_series and unadopted_history:
        # 異なる系列のunadoptedを探す
        dominant = all_types[0] if all_types else set()
        for i, entry in enumerate(unadopted_history):
            entry_types = set(entry.source_types)
            if entry_types != dominant and entry.freshness not in (
                ActivationFreshness.STALE, ActivationFreshness.FADED
            ):
                # 再浮上
                recycled = ActivationCandidate(
                    source_types=list(entry.source_types),
                    intersection_count=len(entry.source_types),
                    activation_strength=entry.original_strength * 0.8,
                    description=f"recycled from unadopted",
                    status=CandidateStatus.ACTIVE,
                )
                result = list(candidates) + [recycled]
                new_unadopted = (
                    unadopted_history[:i] + unadopted_history[i + 1:]
                )
                return result, new_unadopted, True

    return candidates, unadopted_history, single_series


def apply_freshness_decay(
    entries: list[StandbyEntry],
) -> tuple[list[StandbyEntry], list[SpontaneousDecayEntry]]:
    """待機エントリの鮮度を時間経過に応じて減衰させる。"""
    now = time.time()
    result: list[StandbyEntry] = []
    decay_entries: list[SpontaneousDecayEntry] = []

    for entry in entries:
        age = now - entry.timestamp
        if age < 5.0:
            f = ActivationFreshness.FRESH
        elif age < 15.0:
            f = ActivationFreshness.RECENT
        elif age < 30.0:
            f = ActivationFreshness.AGING
        elif age < 60.0:
            f = ActivationFreshness.STALE
        else:
            f = ActivationFreshness.FADED

        orig_idx = _FRESHNESS_ORDER.index(entry.freshness)
        new_idx = _FRESHNESS_ORDER.index(f)
        final = _FRESHNESS_ORDER[max(orig_idx, new_idx)]

        if final != entry.freshness:
            decay_entries.append(SpontaneousDecayEntry(
                candidate_id=entry.candidate_id,
                original_freshness=entry.freshness,
                decayed_freshness=final,
                timestamp=now,
            ))

        result.append(StandbyEntry(
            candidate_id=entry.candidate_id,
            original_strength=entry.original_strength,
            standby_reason=entry.standby_reason,
            timestamp=entry.timestamp,
            freshness=final,
        ))

    return result, decay_entries


def decay_unadopted_history(
    history: list[UnadoptedHistoryEntry],
) -> list[UnadoptedHistoryEntry]:
    """未採択履歴の鮮度を減衰させる。"""
    now = time.time()
    result: list[UnadoptedHistoryEntry] = []
    for entry in history:
        age = now - entry.timestamp
        if age < 10.0:
            f = ActivationFreshness.FRESH
        elif age < 30.0:
            f = ActivationFreshness.RECENT
        elif age < 60.0:
            f = ActivationFreshness.AGING
        elif age < 120.0:
            f = ActivationFreshness.STALE
        else:
            f = ActivationFreshness.FADED
        orig_idx = _FRESHNESS_ORDER.index(entry.freshness)
        new_idx = _FRESHNESS_ORDER.index(f)
        final = _FRESHNESS_ORDER[max(orig_idx, new_idx)]
        result.append(UnadoptedHistoryEntry(
            candidate_id=entry.candidate_id,
            source_types=entry.source_types,
            original_strength=entry.original_strength,
            reason=entry.reason,
            timestamp=entry.timestamp,
            freshness=final,
        ))
    return result


# =============================================================================
# Processor
# =============================================================================

class SpontaneousActivationProcessor:
    """自発起動経路の処理器。

    責務は起動候補形成と起動情報受け渡しに限定され、
    採択、実行、責任付与へ書き戻しを行わない。
    """

    def __init__(self, config: Optional[SpontaneousConfig] = None):
        self._state = SpontaneousState(
            config=config or SpontaneousConfig()
        )

    @property
    def state(self) -> SpontaneousState:
        return self._state

    def notify_external_input(self) -> None:
        """外部入力があったことを通知する。"""
        self._state.last_external_input_time = time.time()
        self._state.consecutive_activation_count = 0

    def process(
        self,
        psyche: Any = None,
        dynamics: Any = None,
        stm: Any = None,
        memories: Any = None,
        recent_actions: Any = None,
        has_external_input: bool = False,
        tick_count: int = 0,
    ) -> ActivationResult:
        """起動候補形成パイプラインを実行する。

        1. 起動候補抽出（8断面交差）
        2. 起動条件整列（連続差分参照）
        3. 競合整理（並立保持）
        4. 起動可否判定用情報化
        5. 受け渡し準備
        + 安全弁
        """
        st = self._state
        cfg = st.config
        cycle_id = st.cycle_count + 1
        st.cycle_count = cycle_id

        # 外部入力時刻の更新
        if has_external_input:
            st.last_external_input_time = time.time()

        # クールダウン減算
        if st.cooldown_remaining > 0:
            st.cooldown_remaining -= 1

        # === Stage 1: 起動候補抽出 ===
        fragments = extract_all_fragments(
            psyche=psyche,
            dynamics=dynamics,
            stm=stm,
            memories=memories,
            recent_actions=recent_actions,
            has_external_input=has_external_input,
            last_external_input_time=st.last_external_input_time,
            config=cfg,
        )

        # === 候補形成 ===
        candidates = form_candidates(fragments, cfg, cycle_id)

        # === Stage 2: 起動条件整列 ===
        candidates, rationales = align_conditions(
            candidates, st.continuous_activation_history, cfg,
        )

        # === Stage 3: 競合整理 ===
        adopted, unadopted, conflicts = resolve_conflicts(candidates, cfg)

        # 未採択を履歴へ
        for ua in unadopted:
            st.unadopted_history.append(UnadoptedHistoryEntry(
                candidate_id=ua.candidate_id,
                source_types=list(ua.source_types),
                original_strength=ua.activation_strength,
                reason="conflict_resolution",
                timestamp=time.time(),
            ))
        if len(st.unadopted_history) > cfg.max_unadopted_history:
            st.unadopted_history = st.unadopted_history[
                -cfg.max_unadopted_history:
            ]

        st.conflict_history.extend(conflicts)
        if len(st.conflict_history) > cfg.max_history:
            st.conflict_history = st.conflict_history[-cfg.max_history:]

        # === 自己強化ループ防止 ===
        adopted, series_suppressed, supp_entries = suppress_consecutive_series(
            adopted, st.recent_adopted_series, cfg,
        )
        st.suppression_entries.extend(supp_entries)
        if len(st.suppression_entries) > cfg.max_suppression_history:
            st.suppression_entries = st.suppression_entries[
                -cfg.max_suppression_history:
            ]

        # 抑制された候補はstandbyへ
        for sc in series_suppressed:
            st.standby_entries.append(StandbyEntry(
                candidate_id=sc.candidate_id,
                original_strength=sc.activation_strength,
                standby_reason="consecutive_series",
                timestamp=time.time(),
            ))

        # === 候補多様性復元 ===
        st.unadopted_history = decay_unadopted_history(st.unadopted_history)
        adopted, st.unadopted_history, single_series_warning = (
            restore_candidate_diversity(
                adopted, st.unadopted_history, cfg,
            )
        )

        # === standby鮮度減衰 ===
        st.standby_entries, decay_entries = apply_freshness_decay(
            st.standby_entries
        )
        st.decay_history.extend(decay_entries)
        if len(st.decay_history) > cfg.max_history:
            st.decay_history = st.decay_history[-cfg.max_history:]

        # FADED standbyは除去
        st.standby_entries = [
            s for s in st.standby_entries
            if s.freshness != ActivationFreshness.FADED
        ]

        # === Stage 4: 起動可否判定用情報化 ===
        should_activate, cooldown_active, overdense_warning = (
            check_activation_feasibility(
                adopted, st.cooldown_remaining,
                st.consecutive_activation_count,
                has_external_input, cfg,
            )
        )

        # 過密化クールダウン
        if overdense_warning:
            cd = apply_overdense_cooldown(
                st.consecutive_activation_count, cfg,
            )
            if cd > 0:
                st.cooldown_remaining = cd
                should_activate = False
                cooldown_active = True

        # === 状態更新 ===
        if should_activate and adopted:
            st.total_activations += 1
            st.consecutive_activation_count += 1
            st.last_activation_cycle = cycle_id

            # 連続起動履歴
            for a in adopted[:1]:
                st.continuous_activation_history.append(
                    ContinuousActivationEntry(
                        candidate_id=a.candidate_id,
                        source_types=list(a.source_types),
                        cycle_id=cycle_id,
                        timestamp=time.time(),
                    )
                )
            if len(st.continuous_activation_history) > cfg.max_history:
                st.continuous_activation_history = (
                    st.continuous_activation_history[-cfg.max_history:]
                )

            # 採択系列の記録
            if adopted:
                st.recent_adopted_series.append(
                    list(adopted[0].source_types)
                )
            max_series = cfg.consecutive_adoption_suppression_count * 3
            st.recent_adopted_series = st.recent_adopted_series[-max_series:]
        else:
            if not has_external_input:
                # 外部入力もなく起動もしなかった場合、連続カウントをリセット
                pass
            else:
                st.consecutive_activation_count = 0

        # 候補の保存
        st.candidates = adopted
        st.rationales = rationales

        # === Stage 5: 受け渡し準備 ===
        return ActivationResult(
            candidates=adopted,
            rationales=rationales,
            conflicts=conflicts,
            should_activate=should_activate,
            suppression_active=any(
                e.mode == SuppressionMode.ACTIVE
                for e in st.suppression_entries[-3:]
            ),
            cooldown_active=cooldown_active,
            overdense_warning=overdense_warning,
            single_series_warning=single_series_warning,
            unadopted_count=len(st.unadopted_history),
        )


# =============================================================================
# Summary
# =============================================================================

def get_spontaneous_summary(state: SpontaneousState) -> str:
    """自発起動状態の要約（enrichment用）。"""
    parts: list[str] = []
    parts.append(f"起動={state.total_activations}")
    parts.append(f"cycle={state.cycle_count}")

    if state.cooldown_remaining > 0:
        parts.append(f"cooldown={state.cooldown_remaining}")

    if state.consecutive_activation_count > 0:
        parts.append(f"連続={state.consecutive_activation_count}")

    if state.candidates:
        best = max(
            state.candidates, key=lambda c: c.activation_strength
        )
        parts.append(f"最強={best.activation_strength:.2f}")
        types_str = "/".join(best.source_types[:3])
        parts.append(f"断面={types_str}")

    if state.standby_entries:
        parts.append(f"待機={len(state.standby_entries)}")

    if state.unadopted_history:
        parts.append(f"未採択={len(state.unadopted_history)}")

    return " ".join(parts) if parts else "自発起動: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_spontaneous_processor(
    config: Optional[SpontaneousConfig] = None,
) -> SpontaneousActivationProcessor:
    """SpontaneousActivationProcessorのファクトリ関数。"""
    return SpontaneousActivationProcessor(config=config)
