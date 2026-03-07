"""
psyche/text_dialogue_input.py - 入力経路拡充（テキスト対話経路）

テキスト対話を独立した入力経路として追加し、既存経路（画面知覚等）と
同列で統合前段へ接続する。

設計原則 (design_text_dialogue_input_expansion.md 準拠):
- 本機能の目的は入力手段の増設であり、応答内容や方針の直接変更ではない
- 既存入力経路を無効化しない
- テキスト入力の存在だけで内部状態を確定させない
- 入力経路の追加を評価機構や行動決定機構へ拡張しない
- 単一入力経路への恒常固定を生まない
- 出力は統合前段の入力情報としてのみ渡し、判断・評価・行動決定を直接起動しない
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class InputRouteType(Enum):
    """入力経路の種別。"""
    TEXT = "text"
    SCREEN = "screen"
    API = "api"
    UNKNOWN = "unknown"


class InputFreshness(Enum):
    """入力単位の鮮度。"""
    FRESH = "fresh"
    RECENT = "recent"
    AGING = "aging"
    STALE = "stale"
    FADED = "faded"


class NormalizationStatus(Enum):
    """正規化状態。"""
    RAW = "raw"
    NORMALIZED = "normalized"
    FRAGMENT = "fragment"
    EMPTY = "empty"


class ContextLinkStatus(Enum):
    """文脈連結状態。"""
    LINKED = "linked"
    PARTIAL = "partial"
    UNLINKED = "unlinked"
    BROKEN = "broken"


class DuplicateStatus(Enum):
    """重複判定状態。"""
    UNIQUE = "unique"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SUPPRESSED = "suppressed"


class RouteConflictStatus(Enum):
    """同時入力競合状態。"""
    NONE = "none"
    PARALLEL = "parallel"
    SINGLE_LINE_RISK = "single_line_risk"


# =============================================================================
# Helpers
# =============================================================================

_FRESHNESS_ORDER = [
    InputFreshness.FRESH,
    InputFreshness.RECENT,
    InputFreshness.AGING,
    InputFreshness.STALE,
    InputFreshness.FADED,
]


def _compute_time_based_freshness(age: float) -> InputFreshness:
    """経過時間から鮮度段階を算出する共通ヘルパー。"""
    if age < 5.0:
        return InputFreshness.FRESH
    elif age < 15.0:
        return InputFreshness.RECENT
    elif age < 30.0:
        return InputFreshness.AGING
    elif age < 60.0:
        return InputFreshness.STALE
    else:
        return InputFreshness.FADED


def _apply_freshness_floor(
    original: InputFreshness,
    time_based: InputFreshness,
) -> InputFreshness:
    """鮮度は悪化方向のみ適用する（元の鮮度より良くならない）。"""
    orig_idx = _FRESHNESS_ORDER.index(original)
    new_idx = _FRESHNESS_ORDER.index(time_based)
    return _FRESHNESS_ORDER[max(orig_idx, new_idx)]

_ZEN_TO_HAN = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "０１２３４５６７８９",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789",
)


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _compute_text_overlap(text_a: str, text_b: str) -> float:
    """2つのテキスト間のbigram重複率（Jaccard）。"""
    if not text_a or not text_b:
        return 0.0

    def _bigrams(t: str) -> set[str]:
        t = t.strip()
        if len(t) < 2:
            return {t} if t else set()
        return {t[i : i + 2] for i in range(len(t) - 1)}

    bg_a = _bigrams(text_a)
    bg_b = _bigrams(text_b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = bg_a & bg_b
    union = bg_a | bg_b
    return len(intersection) / len(union) if union else 0.0


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class InputUnit:
    """正規化済み入力単位。"""
    unit_id: str = ""
    route_type: InputRouteType = InputRouteType.UNKNOWN
    raw_text: str = ""
    normalized_text: str = ""
    normalization_status: NormalizationStatus = NormalizationStatus.RAW
    freshness: InputFreshness = InputFreshness.FRESH
    timestamp: float = 0.0
    sender_id: str = ""
    conversation_id: str = ""
    cycle_id: int = 0
    text_length_category: str = "medium"
    is_output_of_current_cycle: bool = False

    def __post_init__(self):
        if not self.unit_id:
            self.unit_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "route_type": self.route_type.value,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalization_status": self.normalization_status.value,
            "freshness": self.freshness.value,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "conversation_id": self.conversation_id,
            "cycle_id": self.cycle_id,
            "text_length_category": self.text_length_category,
            "is_output_of_current_cycle": self.is_output_of_current_cycle,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputUnit":
        return cls(
            unit_id=d.get("unit_id", ""),
            route_type=InputRouteType(d.get("route_type", "unknown")),
            raw_text=d.get("raw_text", ""),
            normalized_text=d.get("normalized_text", ""),
            normalization_status=NormalizationStatus(
                d.get("normalization_status", "raw")
            ),
            freshness=InputFreshness(d.get("freshness", "fresh")),
            timestamp=d.get("timestamp", 0.0),
            sender_id=d.get("sender_id", ""),
            conversation_id=d.get("conversation_id", ""),
            cycle_id=d.get("cycle_id", 0),
            text_length_category=d.get("text_length_category", "medium"),
            is_output_of_current_cycle=d.get("is_output_of_current_cycle", False),
        )


@dataclass
class ContextLink:
    """文脈連結情報。継続入力の接続に使用。連結失敗時でも入力単位は破棄しない。"""
    link_id: str = ""
    unit_id: str = ""
    previous_unit_id: str = ""
    link_status: ContextLinkStatus = ContextLinkStatus.UNLINKED
    continuation_flag: bool = False
    context_overlap: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.link_id:
            self.link_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "unit_id": self.unit_id,
            "previous_unit_id": self.previous_unit_id,
            "link_status": self.link_status.value,
            "continuation_flag": self.continuation_flag,
            "context_overlap": self.context_overlap,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextLink":
        return cls(
            link_id=d.get("link_id", ""),
            unit_id=d.get("unit_id", ""),
            previous_unit_id=d.get("previous_unit_id", ""),
            link_status=ContextLinkStatus(d.get("link_status", "unlinked")),
            continuation_flag=d.get("continuation_flag", False),
            context_overlap=d.get("context_overlap", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class DuplicateRecord:
    """重複判定情報。抑制は可逆であり条件変化で再投入可能。"""
    record_id: str = ""
    unit_id_a: str = ""
    unit_id_b: str = ""
    similarity: float = 0.0
    status: DuplicateStatus = DuplicateStatus.UNIQUE
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "unit_id_a": self.unit_id_a,
            "unit_id_b": self.unit_id_b,
            "similarity": self.similarity,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DuplicateRecord":
        return cls(
            record_id=d.get("record_id", ""),
            unit_id_a=d.get("unit_id_a", ""),
            unit_id_b=d.get("unit_id_b", ""),
            similarity=d.get("similarity", 0.0),
            status=DuplicateStatus(d.get("status", "unique")),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class RouteConflict:
    """同時入力競合情報。排除対象ではなく、下流の比較可能性を維持する保持対象。"""
    conflict_id: str = ""
    unit_id_a: str = ""
    unit_id_b: str = ""
    route_type_a: InputRouteType = InputRouteType.UNKNOWN
    route_type_b: InputRouteType = InputRouteType.UNKNOWN
    conflict_status: RouteConflictStatus = RouteConflictStatus.NONE
    description: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = _gen_id()
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "unit_id_a": self.unit_id_a,
            "unit_id_b": self.unit_id_b,
            "route_type_a": self.route_type_a.value,
            "route_type_b": self.route_type_b.value,
            "conflict_status": self.conflict_status.value,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RouteConflict":
        return cls(
            conflict_id=d.get("conflict_id", ""),
            unit_id_a=d.get("unit_id_a", ""),
            unit_id_b=d.get("unit_id_b", ""),
            route_type_a=InputRouteType(d.get("route_type_a", "unknown")),
            route_type_b=InputRouteType(d.get("route_type_b", "unknown")),
            conflict_status=RouteConflictStatus(
                d.get("conflict_status", "none")
            ),
            description=d.get("description", ""),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class ReceiveHistoryEntry:
    """受信履歴。時間的性質として生成・変化・減衰を行う。"""
    unit_id: str = ""
    route_type: InputRouteType = InputRouteType.UNKNOWN
    timestamp: float = 0.0
    cycle_id: int = 0
    freshness: InputFreshness = InputFreshness.FRESH

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "route_type": self.route_type.value,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "freshness": self.freshness.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReceiveHistoryEntry":
        return cls(
            unit_id=d.get("unit_id", ""),
            route_type=InputRouteType(d.get("route_type", "unknown")),
            timestamp=d.get("timestamp", 0.0),
            cycle_id=d.get("cycle_id", 0),
            freshness=InputFreshness(d.get("freshness", "fresh")),
        )


@dataclass
class SuppressionHistoryEntry:
    """再投入抑制履歴。可逆であり条件変化で再投入可能。"""
    unit_id: str = ""
    route_type: InputRouteType = InputRouteType.UNKNOWN
    reason: str = ""
    timestamp: float = 0.0
    reversible: bool = True
    freshness: InputFreshness = InputFreshness.FRESH

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "route_type": self.route_type.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "reversible": self.reversible,
            "freshness": self.freshness.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SuppressionHistoryEntry":
        return cls(
            unit_id=d.get("unit_id", ""),
            route_type=InputRouteType(d.get("route_type", "unknown")),
            reason=d.get("reason", ""),
            timestamp=d.get("timestamp", 0.0),
            reversible=d.get("reversible", True),
            freshness=InputFreshness(d.get("freshness", "fresh")),
        )


@dataclass
class DecayHistoryEntry:
    """希薄化履歴。"""
    unit_id: str = ""
    original_freshness: InputFreshness = InputFreshness.FRESH
    decayed_freshness: InputFreshness = InputFreshness.FRESH
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "original_freshness": self.original_freshness.value,
            "decayed_freshness": self.decayed_freshness.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecayHistoryEntry":
        return cls(
            unit_id=d.get("unit_id", ""),
            original_freshness=InputFreshness(
                d.get("original_freshness", "fresh")
            ),
            decayed_freshness=InputFreshness(
                d.get("decayed_freshness", "fresh")
            ),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class TextDialogueConfig:
    """テキスト対話入力の設定。"""
    max_units_per_cycle: int = 10
    max_pending: int = 20
    max_history: int = 50
    max_suppression_history: int = 30
    max_decay_history: int = 30
    empty_streak_threshold: int = 5
    single_route_dominance_threshold: float = 0.8
    similarity_threshold: float = 0.85
    freshness_decay_rate: float = 0.05
    recent_adoption_suppression_count: int = 3
    short_text_threshold: int = 10
    long_text_threshold: int = 100
    format_diversity_threshold: float = 0.7
    stale_cycle_threshold: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_units_per_cycle": self.max_units_per_cycle,
            "max_pending": self.max_pending,
            "max_history": self.max_history,
            "max_suppression_history": self.max_suppression_history,
            "max_decay_history": self.max_decay_history,
            "empty_streak_threshold": self.empty_streak_threshold,
            "single_route_dominance_threshold": self.single_route_dominance_threshold,
            "similarity_threshold": self.similarity_threshold,
            "freshness_decay_rate": self.freshness_decay_rate,
            "recent_adoption_suppression_count": self.recent_adoption_suppression_count,
            "short_text_threshold": self.short_text_threshold,
            "long_text_threshold": self.long_text_threshold,
            "format_diversity_threshold": self.format_diversity_threshold,
            "stale_cycle_threshold": self.stale_cycle_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextDialogueConfig":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class TextDialogueState:
    """テキスト対話入力の全状態。"""
    config: TextDialogueConfig = field(default_factory=TextDialogueConfig)
    route_registry: dict[str, bool] = field(
        default_factory=lambda: {"text": True, "screen": True, "api": True}
    )
    pending_units: list[InputUnit] = field(default_factory=list)
    active_units: list[InputUnit] = field(default_factory=list)
    context_links: list[ContextLink] = field(default_factory=list)
    duplicate_records: list[DuplicateRecord] = field(default_factory=list)
    route_conflicts: list[RouteConflict] = field(default_factory=list)
    receive_history: list[ReceiveHistoryEntry] = field(default_factory=list)
    suppression_history: list[SuppressionHistoryEntry] = field(
        default_factory=list
    )
    decay_history: list[DecayHistoryEntry] = field(default_factory=list)
    cycle_count: int = 0
    total_received: int = 0
    empty_streak_counter: int = 0
    route_usage_counts: dict[str, int] = field(
        default_factory=lambda: {"text": 0, "screen": 0, "api": 0}
    )
    recent_adopted_routes: list[str] = field(default_factory=list)
    recent_adopted_formats: list[str] = field(default_factory=list)
    holdback_units: list[InputUnit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "route_registry": dict(self.route_registry),
            "pending_units": [u.to_dict() for u in self.pending_units],
            "active_units": [u.to_dict() for u in self.active_units],
            "context_links": [cl.to_dict() for cl in self.context_links],
            "duplicate_records": [
                dr.to_dict() for dr in self.duplicate_records
            ],
            "route_conflicts": [
                rc.to_dict() for rc in self.route_conflicts
            ],
            "receive_history": [
                rh.to_dict() for rh in self.receive_history
            ],
            "suppression_history": [
                sh.to_dict() for sh in self.suppression_history
            ],
            "decay_history": [dh.to_dict() for dh in self.decay_history],
            "cycle_count": self.cycle_count,
            "total_received": self.total_received,
            "empty_streak_counter": self.empty_streak_counter,
            "route_usage_counts": dict(self.route_usage_counts),
            "recent_adopted_routes": list(self.recent_adopted_routes),
            "recent_adopted_formats": list(self.recent_adopted_formats),
            "holdback_units": [u.to_dict() for u in self.holdback_units],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextDialogueState":
        cfg = TextDialogueConfig.from_dict(d.get("config", {}))
        return cls(
            config=cfg,
            route_registry=d.get(
                "route_registry",
                {"text": True, "screen": True, "api": True},
            ),
            pending_units=[
                InputUnit.from_dict(u) for u in d.get("pending_units", [])
            ],
            active_units=[
                InputUnit.from_dict(u) for u in d.get("active_units", [])
            ],
            context_links=[
                ContextLink.from_dict(cl)
                for cl in d.get("context_links", [])
            ],
            duplicate_records=[
                DuplicateRecord.from_dict(dr)
                for dr in d.get("duplicate_records", [])
            ],
            route_conflicts=[
                RouteConflict.from_dict(rc)
                for rc in d.get("route_conflicts", [])
            ],
            receive_history=[
                ReceiveHistoryEntry.from_dict(rh)
                for rh in d.get("receive_history", [])
            ],
            suppression_history=[
                SuppressionHistoryEntry.from_dict(sh)
                for sh in d.get("suppression_history", [])
            ],
            decay_history=[
                DecayHistoryEntry.from_dict(dh)
                for dh in d.get("decay_history", [])
            ],
            cycle_count=d.get("cycle_count", 0),
            total_received=d.get("total_received", 0),
            empty_streak_counter=d.get("empty_streak_counter", 0),
            route_usage_counts=d.get(
                "route_usage_counts",
                {"text": 0, "screen": 0, "api": 0},
            ),
            recent_adopted_routes=d.get("recent_adopted_routes", []),
            recent_adopted_formats=d.get("recent_adopted_formats", []),
            holdback_units=[
                InputUnit.from_dict(u)
                for u in d.get("holdback_units", [])
            ],
        )


@dataclass
class HandoffResult:
    """受け渡し結果。判断・評価・行動決定を含まない。"""
    units: list[InputUnit] = field(default_factory=list)
    context_links: list[ContextLink] = field(default_factory=list)
    conflicts: list[RouteConflict] = field(default_factory=list)
    route_distribution: dict[str, float] = field(default_factory=dict)
    empty_warning: bool = False
    single_route_warning: bool = False
    diversity_warning: bool = False
    holdback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": [u.to_dict() for u in self.units],
            "context_links": [cl.to_dict() for cl in self.context_links],
            "conflicts": [rc.to_dict() for rc in self.conflicts],
            "route_distribution": dict(self.route_distribution),
            "empty_warning": self.empty_warning,
            "single_route_warning": self.single_route_warning,
            "diversity_warning": self.diversity_warning,
            "holdback_count": self.holdback_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HandoffResult":
        return cls(
            units=[
                InputUnit.from_dict(u) for u in d.get("units", [])
            ],
            context_links=[
                ContextLink.from_dict(cl)
                for cl in d.get("context_links", [])
            ],
            conflicts=[
                RouteConflict.from_dict(rc)
                for rc in d.get("conflicts", [])
            ],
            route_distribution=d.get("route_distribution", {}),
            empty_warning=d.get("empty_warning", False),
            single_route_warning=d.get("single_route_warning", False),
            diversity_warning=d.get("diversity_warning", False),
            holdback_count=d.get("holdback_count", 0),
        )


# =============================================================================
# Stage 1: 受信 (Receive)
# =============================================================================

def receive_input(
    text: str,
    route_type: InputRouteType,
    sender_id: str = "",
    conversation_id: str = "",
    cycle_id: int = 0,
    config: Optional[TextDialogueConfig] = None,
) -> InputUnit:
    """生の入力テキストからInputUnitを生成する。"""
    cfg = config or TextDialogueConfig()
    text_len = len(text.strip()) if text else 0

    if text_len == 0:
        category = "short"
    elif text_len <= cfg.short_text_threshold:
        category = "short"
    elif text_len >= cfg.long_text_threshold:
        category = "long"
    else:
        category = "medium"

    if not text or not text.strip():
        norm_status = NormalizationStatus.EMPTY
    else:
        norm_status = NormalizationStatus.RAW

    return InputUnit(
        route_type=route_type,
        raw_text=text or "",
        normalized_text="",
        normalization_status=norm_status,
        freshness=InputFreshness.FRESH,
        sender_id=sender_id,
        conversation_id=conversation_id,
        cycle_id=cycle_id,
        text_length_category=category,
    )


# =============================================================================
# Stage 2: 正規化 (Normalize)
# =============================================================================

def normalize_unit(unit: InputUnit) -> InputUnit:
    """表記ゆれ・空入力・断片入力を共通入力単位へ整理する。"""
    if unit.normalization_status == NormalizationStatus.EMPTY:
        return InputUnit(
            unit_id=unit.unit_id,
            route_type=unit.route_type,
            raw_text=unit.raw_text,
            normalized_text="",
            normalization_status=NormalizationStatus.EMPTY,
            freshness=unit.freshness,
            timestamp=unit.timestamp,
            sender_id=unit.sender_id,
            conversation_id=unit.conversation_id,
            cycle_id=unit.cycle_id,
            text_length_category=unit.text_length_category,
        )

    raw = unit.raw_text

    # 1. 前後空白除去、連続空白の正規化
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)

    # 2. 表記ゆれ（全角英数字→半角、全角記号の一部→半角）
    text = text.translate(_ZEN_TO_HAN)
    text = text.replace("\uff1f", "?").replace("\uff01", "!").replace("\u3000", " ")

    # 3. 断片入力の判定（極端に短い、句読点のみ等）
    stripped = text.replace(" ", "")
    if len(stripped) <= 2 and all(
        c in "\u3002\u3001\uff01\uff1f!?.,;:\u2026" for c in stripped
    ):
        norm_status = NormalizationStatus.FRAGMENT
    elif not text:
        norm_status = NormalizationStatus.EMPTY
    else:
        norm_status = NormalizationStatus.NORMALIZED

    return InputUnit(
        unit_id=unit.unit_id,
        route_type=unit.route_type,
        raw_text=unit.raw_text,
        normalized_text=text,
        normalization_status=norm_status,
        freshness=unit.freshness,
        timestamp=unit.timestamp,
        sender_id=unit.sender_id,
        conversation_id=unit.conversation_id,
        cycle_id=unit.cycle_id,
        text_length_category=unit.text_length_category,
    )


# =============================================================================
# Stage 3: 文脈付与 (Context Attach)
# =============================================================================

def attach_context(
    unit: InputUnit,
    recent_units: list[InputUnit],
    config: Optional[TextDialogueConfig] = None,
) -> ContextLink:
    """単発入力と継続入力を区別可能な情報を付加する。

    直前対話との接続可能性を保持する。
    連結失敗時でも入力単位自体は破棄せず保持する。
    """
    if not recent_units:
        return ContextLink(
            unit_id=unit.unit_id,
            previous_unit_id="",
            link_status=ContextLinkStatus.UNLINKED,
            continuation_flag=False,
            context_overlap=0.0,
        )

    prev = recent_units[-1]

    same_conversation = bool(
        unit.conversation_id and unit.conversation_id == prev.conversation_id
    )
    same_sender = bool(
        unit.sender_id and unit.sender_id == prev.sender_id
    )
    overlap = _compute_text_overlap(
        unit.normalized_text or unit.raw_text,
        prev.normalized_text or prev.raw_text,
    )
    time_gap = abs(unit.timestamp - prev.timestamp)
    time_close = time_gap < 10.0

    if same_conversation and (same_sender or time_close):
        link_status = (
            ContextLinkStatus.LINKED if overlap > 0.3
            else ContextLinkStatus.PARTIAL
        )
        continuation = True
    elif same_sender and time_close:
        link_status = ContextLinkStatus.PARTIAL
        continuation = True
    elif time_close and overlap > 0.2:
        link_status = ContextLinkStatus.PARTIAL
        continuation = False
    else:
        link_status = ContextLinkStatus.UNLINKED
        continuation = False

    return ContextLink(
        unit_id=unit.unit_id,
        previous_unit_id=prev.unit_id,
        link_status=link_status,
        continuation_flag=continuation,
        context_overlap=overlap,
    )


# =============================================================================
# Stage 4: 既存入力表現への整合 (Align to Existing Format)
# =============================================================================

def align_to_percept_format(
    unit: InputUnit,
    context_link: Optional[ContextLink] = None,
) -> dict[str, Any]:
    """経路差を吸収する抽象化のみを行い、意味解釈や判断は行わない。

    Percept互換のdict形式に変換する。emotion/intent/sentiment等は設定しない。
    """
    text = unit.normalized_text or unit.raw_text or ""
    result: dict[str, Any] = {
        "text": text,
        "meaning": text,
        "emotion": "neutral",
        "intent": "unknown",
        "topics": [],
        "sentiment": 0.0,
        "emotion_valence": 0.0,
        "_route_type": unit.route_type.value,
        "_route_unit_id": unit.unit_id,
        "_sender_id": unit.sender_id,
        "_conversation_id": unit.conversation_id,
        "_text_length_category": unit.text_length_category,
        "_normalization_status": unit.normalization_status.value,
        "_freshness": unit.freshness.value,
    }
    if context_link:
        result["_context_link_status"] = context_link.link_status.value
        result["_continuation_flag"] = context_link.continuation_flag
        result["_context_overlap"] = context_link.context_overlap
    return result


# =============================================================================
# Stage 5: 重複調整 (Dedup)
# =============================================================================

def detect_duplicates(
    units: list[InputUnit],
    history: list[ReceiveHistoryEntry],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[list[InputUnit], list[InputUnit], list[DuplicateRecord]]:
    """同一内容の多重流入を抑制し、異なる内容の同時流入は並立保持する。

    Returns:
        (accepted_units, suppressed_units, duplicate_records)
    """
    cfg = config or TextDialogueConfig()
    accepted: list[InputUnit] = []
    suppressed: list[InputUnit] = []
    records: list[DuplicateRecord] = []
    seen_texts: dict[str, str] = {}

    for unit in units:
        text = unit.normalized_text or unit.raw_text or ""

        if unit.normalization_status == NormalizationStatus.EMPTY:
            accepted.append(unit)
            continue

        # 完全重複チェック
        if text in seen_texts:
            records.append(DuplicateRecord(
                unit_id_a=seen_texts[text],
                unit_id_b=unit.unit_id,
                similarity=1.0,
                status=DuplicateStatus.DUPLICATE,
            ))
            suppressed.append(unit)
            continue

        # 類似重複チェック（bigramベース）
        is_near_dup = False
        for seen_text, seen_id in seen_texts.items():
            sim = _compute_text_overlap(text, seen_text)
            if sim >= cfg.similarity_threshold:
                records.append(DuplicateRecord(
                    unit_id_a=seen_id,
                    unit_id_b=unit.unit_id,
                    similarity=sim,
                    status=DuplicateStatus.NEAR_DUPLICATE,
                ))
                suppressed.append(unit)
                is_near_dup = True
                break

        if not is_near_dup:
            accepted.append(unit)
            seen_texts[text] = unit.unit_id

    return accepted, suppressed, records


# =============================================================================
# Stage 6: 受け渡し準備 (Handoff Prep)
# =============================================================================

def prepare_handoff(
    units: list[InputUnit],
    context_links: list[ContextLink],
    conflicts: list[RouteConflict],
    route_usage_counts: dict[str, int],
    holdback_count: int = 0,
    empty_warning: bool = False,
    single_route_warning: bool = False,
    diversity_warning: bool = False,
) -> HandoffResult:
    """受け渡し結果を構築する。判断・評価・行動決定を直接起動しない。"""
    total = sum(route_usage_counts.values())
    distribution = (
        {r: c / total for r, c in route_usage_counts.items()} if total > 0
        else {}
    )
    return HandoffResult(
        units=list(units),
        context_links=list(context_links),
        conflicts=list(conflicts),
        route_distribution=distribution,
        empty_warning=empty_warning,
        single_route_warning=single_route_warning,
        diversity_warning=diversity_warning,
        holdback_count=holdback_count,
    )


# =============================================================================
# Safety Valves
# =============================================================================

def apply_freshness_decay(
    units: list[InputUnit],
    history: list[ReceiveHistoryEntry],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[list[InputUnit], list[DecayHistoryEntry]]:
    """時間減衰を適用。単回判定が恒久化しないようにする。"""
    decayed_units: list[InputUnit] = []
    decay_entries: list[DecayHistoryEntry] = []
    now = time.time()

    for unit in units:
        age = now - unit.timestamp
        original = unit.freshness
        time_based = _compute_time_based_freshness(age)
        final_fresh = _apply_freshness_floor(original, time_based)

        if final_fresh != original:
            decay_entries.append(DecayHistoryEntry(
                unit_id=unit.unit_id,
                original_freshness=original,
                decayed_freshness=final_fresh,
                timestamp=now,
            ))

        decayed_units.append(InputUnit(
            unit_id=unit.unit_id,
            route_type=unit.route_type,
            raw_text=unit.raw_text,
            normalized_text=unit.normalized_text,
            normalization_status=unit.normalization_status,
            freshness=final_fresh,
            timestamp=unit.timestamp,
            sender_id=unit.sender_id,
            conversation_id=unit.conversation_id,
            cycle_id=unit.cycle_id,
            text_length_category=unit.text_length_category,
            is_output_of_current_cycle=unit.is_output_of_current_cycle,
        ))

    return decayed_units, decay_entries


def decay_receive_history(
    history: list[ReceiveHistoryEntry],
) -> list[ReceiveHistoryEntry]:
    """受信履歴の鮮度を時間経過に応じて減衰させる。"""
    now = time.time()
    result: list[ReceiveHistoryEntry] = []
    for entry in history:
        age = now - entry.timestamp
        time_based = _compute_time_based_freshness(age)
        final = _apply_freshness_floor(entry.freshness, time_based)
        result.append(ReceiveHistoryEntry(
            unit_id=entry.unit_id,
            route_type=entry.route_type,
            timestamp=entry.timestamp,
            cycle_id=entry.cycle_id,
            freshness=final,
        ))
    return result


def decay_suppression_history(
    history: list[SuppressionHistoryEntry],
) -> list[SuppressionHistoryEntry]:
    """抑制履歴の鮮度を時間経過に応じて減衰させる。"""
    now = time.time()
    result: list[SuppressionHistoryEntry] = []
    for entry in history:
        age = now - entry.timestamp
        time_based = _compute_time_based_freshness(age)
        final = _apply_freshness_floor(entry.freshness, time_based)
        result.append(SuppressionHistoryEntry(
            unit_id=entry.unit_id,
            route_type=entry.route_type,
            reason=entry.reason,
            timestamp=entry.timestamp,
            reversible=entry.reversible,
            freshness=final,
        ))
    return result


def suppress_recent_adoption(
    units: list[InputUnit],
    recent_adopted_routes: list[str],
    recent_adopted_formats: list[str],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[list[InputUnit], list[InputUnit], list[SuppressionHistoryEntry]]:
    """直近採用入力形式の連続優遇を抑制し、未採用形式の再浮上経路を維持する。

    Returns:
        (passed_units, suppressed_units, suppression_entries)
    """
    cfg = config or TextDialogueConfig()
    n = cfg.recent_adoption_suppression_count
    passed: list[InputUnit] = []
    suppressed: list[InputUnit] = []
    entries: list[SuppressionHistoryEntry] = []

    recent_r = recent_adopted_routes[-n:]
    recent_f = recent_adopted_formats[-n:]

    # 全て同一経路なら、その経路を1つだけ抑制
    if len(recent_r) >= n and len(set(recent_r)) == 1:
        dominant_route = recent_r[0]
        suppressed_one = False
        for unit in units:
            if unit.route_type.value == dominant_route and not suppressed_one:
                suppressed.append(unit)
                entries.append(SuppressionHistoryEntry(
                    unit_id=unit.unit_id,
                    route_type=unit.route_type,
                    reason=f"recent_adoption_suppression: route={dominant_route}",
                    reversible=True,
                ))
                suppressed_one = True
            else:
                passed.append(unit)
    else:
        passed = list(units)

    # 全て同一形式なら、その形式を1つだけ抑制
    if len(recent_f) >= n and len(set(recent_f)) == 1 and passed:
        dominant_format = recent_f[0]
        new_passed: list[InputUnit] = []
        format_suppressed = False
        for unit in passed:
            if (
                unit.text_length_category == dominant_format
                and not format_suppressed
            ):
                suppressed.append(unit)
                entries.append(SuppressionHistoryEntry(
                    unit_id=unit.unit_id,
                    route_type=unit.route_type,
                    reason=f"format_adoption_suppression: format={dominant_format}",
                    reversible=True,
                ))
                format_suppressed = True
            else:
                new_passed.append(unit)
        if new_passed:
            passed = new_passed

    return passed, suppressed, entries


def check_empty_streak(
    empty_streak_counter: int,
    config: Optional[TextDialogueConfig] = None,
) -> bool:
    """空入力連続の安全弁。経路停止を起こさない。"""
    cfg = config or TextDialogueConfig()
    return empty_streak_counter >= cfg.empty_streak_threshold


def detect_single_route_dominance(
    route_usage_counts: dict[str, int],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[bool, str]:
    """単一経路のみが継続的に有効化される状態を検知する。

    Returns:
        (is_dominant, dominant_route_name)
    """
    cfg = config or TextDialogueConfig()
    total = sum(route_usage_counts.values())
    if total < 3:
        return False, ""
    for route, count in route_usage_counts.items():
        if count / total >= cfg.single_route_dominance_threshold:
            return True, route
    return False, ""


def ensure_format_diversity(
    units: list[InputUnit],
    recent_adopted_formats: list[str],
    holdback_units: list[InputUnit],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[list[InputUnit], list[InputUnit], bool]:
    """短文/長文の片側支配を防止し、入力単位の多様保持を維持する。

    Returns:
        (result_units, updated_holdback, diversity_warning)
    """
    cfg = config or TextDialogueConfig()
    if not units:
        return units, holdback_units, False

    format_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for unit in units:
        cat = unit.text_length_category
        if cat in format_counts:
            format_counts[cat] += 1

    total = sum(format_counts.values())
    if total == 0:
        return units, holdback_units, False

    max_ratio = max(format_counts.values()) / total
    diversity_warning = max_ratio >= cfg.format_diversity_threshold

    if diversity_warning and holdback_units:
        dominant_format = max(format_counts, key=lambda k: format_counts[k])
        inject_candidates = [
            u
            for u in holdback_units
            if u.text_length_category != dominant_format
            and u.freshness not in (InputFreshness.STALE, InputFreshness.FADED)
        ]
        if inject_candidates:
            injected = inject_candidates[0]
            result_units = list(units) + [injected]
            new_holdback = [
                u for u in holdback_units if u.unit_id != injected.unit_id
            ]
            return result_units, new_holdback, diversity_warning

    return units, holdback_units, diversity_warning


def restore_multi_route(
    units: list[InputUnit],
    holdback_units: list[InputUnit],
    route_usage_counts: dict[str, int],
    config: Optional[TextDialogueConfig] = None,
) -> tuple[list[InputUnit], list[InputUnit], bool]:
    """競合入力が単線化した場合は保留中の代替入力を再注入し複線状態を復元する。

    Returns:
        (result_units, updated_holdback, single_route_warning)
    """
    cfg = config or TextDialogueConfig()
    is_dominant, dominant_route = detect_single_route_dominance(
        route_usage_counts, cfg
    )
    if not is_dominant or not holdback_units:
        return units, holdback_units, is_dominant

    alt_candidates = [
        u
        for u in holdback_units
        if u.route_type.value != dominant_route
        and u.freshness not in (InputFreshness.STALE, InputFreshness.FADED)
    ]
    if alt_candidates:
        injected = alt_candidates[0]
        result_units = list(units) + [injected]
        new_holdback = [
            u for u in holdback_units if u.unit_id != injected.unit_id
        ]
        return result_units, new_holdback, True

    return units, holdback_units, is_dominant


def filter_circular_reference(
    units: list[InputUnit],
    current_cycle_id: int,
) -> list[InputUnit]:
    """出力済み入力単位を同サイクル内で再受信しない（循環参照防止）。"""
    return [
        u
        for u in units
        if not u.is_output_of_current_cycle or u.cycle_id != current_cycle_id
    ]


# =============================================================================
# Processor
# =============================================================================

class TextDialogueProcessor:
    """テキスト対話入力経路の処理器。

    責務は入力経路の形成と入力情報の受け渡しに限定され、
    判断層、行動層、責任層へ書き戻しを行わない。
    """

    def __init__(self, config: Optional[TextDialogueConfig] = None):
        self._state = TextDialogueState(
            config=config or TextDialogueConfig()
        )

    @property
    def state(self) -> TextDialogueState:
        return self._state

    def process(
        self,
        text: str,
        route_type: InputRouteType = InputRouteType.TEXT,
        sender_id: str = "",
        conversation_id: str = "",
        existing_percept: Any = None,
        recent_context: Any = None,
        tick_count: int = 0,
    ) -> HandoffResult:
        """6段階処理パイプラインを実行する。

        1. 受信  2. 正規化  3. 文脈付与
        4. 既存入力表現への整合  5. 重複調整  6. 受け渡し準備
        + 安全弁（空入力、単一経路支配、形式多様性、循環参照防止等）
        """
        st = self._state
        cfg = st.config
        cycle_id = st.cycle_count + 1
        st.cycle_count = cycle_id

        # === Stage 1: 受信 ===
        unit = receive_input(
            text=text,
            route_type=route_type,
            sender_id=sender_id,
            conversation_id=conversation_id,
            cycle_id=cycle_id,
            config=cfg,
        )

        if unit.normalization_status == NormalizationStatus.EMPTY:
            st.empty_streak_counter += 1
        else:
            st.empty_streak_counter = 0
        st.total_received += 1

        # === Stage 2: 正規化 ===
        unit = normalize_unit(unit)

        st.receive_history.append(ReceiveHistoryEntry(
            unit_id=unit.unit_id,
            route_type=unit.route_type,
            timestamp=unit.timestamp,
            cycle_id=cycle_id,
            freshness=unit.freshness,
        ))
        if len(st.receive_history) > cfg.max_history:
            st.receive_history = st.receive_history[-cfg.max_history:]

        # 履歴の鮮度減衰
        st.receive_history = decay_receive_history(st.receive_history)
        st.suppression_history = decay_suppression_history(
            st.suppression_history
        )

        # === Stage 3: 文脈付与 ===
        context_link = attach_context(
            unit=unit,
            recent_units=st.active_units[-10:],
            config=cfg,
        )
        st.context_links.append(context_link)
        if len(st.context_links) > cfg.max_history:
            st.context_links = st.context_links[-cfg.max_history:]

        # === 既存経路からの同時入力を収集 ===
        concurrent_units: list[InputUnit] = [unit]
        if existing_percept is not None:
            screen_text = getattr(existing_percept, "text", "")
            if screen_text:
                screen_unit = receive_input(
                    text=screen_text,
                    route_type=InputRouteType.SCREEN,
                    cycle_id=cycle_id,
                    config=cfg,
                )
                screen_unit = normalize_unit(screen_unit)
                concurrent_units.append(screen_unit)

        # === 循環参照防止 ===
        concurrent_units = filter_circular_reference(
            concurrent_units, cycle_id
        )

        # === 保留中ユニットの鮮度減衰 ===
        if st.pending_units:
            decayed_pending, decay_entries = apply_freshness_decay(
                st.pending_units, st.receive_history, cfg
            )
            st.pending_units = decayed_pending
            st.decay_history.extend(decay_entries)
            if len(st.decay_history) > cfg.max_decay_history:
                st.decay_history = st.decay_history[-cfg.max_decay_history:]

        # === Stage 5: 重複調整 ===
        accepted, suppressed, dup_records = detect_duplicates(
            concurrent_units, st.receive_history, cfg
        )
        st.duplicate_records.extend(dup_records)
        if len(st.duplicate_records) > cfg.max_history:
            st.duplicate_records = st.duplicate_records[-cfg.max_history:]

        for sup_unit in suppressed:
            st.holdback_units.append(sup_unit)
            st.suppression_history.append(SuppressionHistoryEntry(
                unit_id=sup_unit.unit_id,
                route_type=sup_unit.route_type,
                reason="duplicate_suppression",
                reversible=True,
            ))
        if len(st.suppression_history) > cfg.max_suppression_history:
            st.suppression_history = st.suppression_history[
                -cfg.max_suppression_history:
            ]

        # === 自己強化ループ防止 ===
        accepted, loop_suppressed, loop_entries = suppress_recent_adoption(
            accepted, st.recent_adopted_routes,
            st.recent_adopted_formats, cfg,
        )
        for sup_unit in loop_suppressed:
            st.holdback_units.append(sup_unit)
        st.suppression_history.extend(loop_entries)

        # === 同時入力競合の検出・保持 ===
        conflicts: list[RouteConflict] = []
        route_types_in = {u.route_type for u in accepted}

        if len(route_types_in) > 1:
            by_route: dict[str, list[InputUnit]] = {}
            for u in accepted:
                by_route.setdefault(u.route_type.value, []).append(u)
            route_list = list(by_route.keys())
            for i in range(len(route_list)):
                for j in range(i + 1, len(route_list)):
                    rt_a, rt_b = route_list[i], route_list[j]
                    for ua in by_route[rt_a]:
                        for ub in by_route[rt_b]:
                            conflicts.append(RouteConflict(
                                unit_id_a=ua.unit_id,
                                unit_id_b=ub.unit_id,
                                route_type_a=ua.route_type,
                                route_type_b=ub.route_type,
                                conflict_status=RouteConflictStatus.PARALLEL,
                                description=f"concurrent: {rt_a} vs {rt_b}",
                            ))

        st.route_conflicts.extend(conflicts)
        if len(st.route_conflicts) > cfg.max_history:
            st.route_conflicts = st.route_conflicts[-cfg.max_history:]

        # === 経路使用カウント更新 ===
        for u in accepted:
            rt = u.route_type.value
            st.route_usage_counts[rt] = st.route_usage_counts.get(rt, 0) + 1

        # === 単一経路支配の検知と復元 ===
        accepted, st.holdback_units, single_route_warning = restore_multi_route(
            accepted, st.holdback_units, st.route_usage_counts, cfg,
        )

        # === 形式多様性の確保 ===
        accepted, st.holdback_units, diversity_warning = ensure_format_diversity(
            accepted, st.recent_adopted_formats, st.holdback_units, cfg,
        )

        # === 空入力安全弁 ===
        empty_warning = check_empty_streak(st.empty_streak_counter, cfg)

        # === 採用履歴の更新 ===
        for u in accepted:
            st.recent_adopted_routes.append(u.route_type.value)
            st.recent_adopted_formats.append(u.text_length_category)
        max_adoption = cfg.recent_adoption_suppression_count * 3
        st.recent_adopted_routes = st.recent_adopted_routes[-max_adoption:]
        st.recent_adopted_formats = st.recent_adopted_formats[-max_adoption:]

        # holdback上限
        if len(st.holdback_units) > cfg.max_pending:
            st.holdback_units = st.holdback_units[-cfg.max_pending:]

        # active_unitsの更新
        st.active_units.extend(accepted)
        if len(st.active_units) > cfg.max_history:
            st.active_units = st.active_units[-cfg.max_history:]

        # === Stage 6: 受け渡し準備 ===
        unit_ids = {u.unit_id for u in accepted}
        relevant_links = [
            cl for cl in st.context_links if cl.unit_id in unit_ids
        ]

        return prepare_handoff(
            units=accepted,
            context_links=relevant_links,
            conflicts=conflicts,
            route_usage_counts=st.route_usage_counts,
            holdback_count=len(st.holdback_units),
            empty_warning=empty_warning,
            single_route_warning=single_route_warning,
            diversity_warning=diversity_warning,
        )


# =============================================================================
# Integration: merge_with_percept
# =============================================================================

def merge_with_percept(
    existing_percept: Any,
    handoff: HandoffResult,
) -> dict[str, Any]:
    """テキスト対話入力をPercept形式と統合する。

    経路差を吸収する抽象化のみ。意味解釈や判断は行わない。
    追加経路は既存経路と同列で接続し、優先固定経路を設けない。
    """
    text = getattr(existing_percept, "text", "") if existing_percept else ""
    meaning = getattr(existing_percept, "meaning", "") if existing_percept else ""
    emotion = (
        getattr(existing_percept, "emotion", "neutral")
        if existing_percept else "neutral"
    )
    intent = (
        getattr(existing_percept, "intent", "unknown")
        if existing_percept else "unknown"
    )
    topics = (
        list(getattr(existing_percept, "topics", []))
        if existing_percept else []
    )
    sentiment = (
        getattr(existing_percept, "sentiment", 0.0)
        if existing_percept else 0.0
    )
    emotion_valence = (
        getattr(existing_percept, "emotion_valence", 0.0)
        if existing_percept else 0.0
    )

    text_units = [
        u for u in handoff.units if u.route_type == InputRouteType.TEXT
    ]

    if not text_units:
        return {
            "text": text,
            "meaning": meaning,
            "emotion": emotion,
            "intent": intent,
            "topics": topics,
            "sentiment": sentiment,
            "emotion_valence": emotion_valence,
            "_route_info": {
                "routes": ["screen"] if text else [],
                "text_input_present": False,
            },
        }

    text_contents = [
        u.normalized_text or u.raw_text
        for u in text_units
        if (u.normalized_text or u.raw_text)
    ]
    merged_text = text_contents[0] if text_contents else text

    routes_present: list[str] = []
    if text:
        routes_present.append("screen")
    for u in text_units:
        if u.route_type.value not in routes_present:
            routes_present.append(u.route_type.value)

    return {
        "text": merged_text,
        "meaning": merged_text,
        "emotion": emotion,
        "intent": intent,
        "topics": topics,
        "sentiment": sentiment,
        "emotion_valence": emotion_valence,
        "_route_info": {
            "routes": routes_present,
            "text_input_present": True,
            "text_unit_count": len(text_units),
            "conflicts": len(handoff.conflicts),
            "empty_warning": handoff.empty_warning,
            "single_route_warning": handoff.single_route_warning,
            "diversity_warning": handoff.diversity_warning,
            "holdback_count": handoff.holdback_count,
        },
    }


# =============================================================================
# Summary
# =============================================================================

def get_text_dialogue_summary(state: TextDialogueState) -> str:
    """テキスト対話入力状態の要約（enrichment用）。"""
    parts: list[str] = []
    parts.append(f"受信={state.total_received}")
    parts.append(f"cycle={state.cycle_count}")

    total_usage = sum(state.route_usage_counts.values())
    if total_usage > 0:
        route_strs = []
        for route, count in sorted(state.route_usage_counts.items()):
            if count > 0:
                pct = count / total_usage * 100
                route_strs.append(f"{route}:{pct:.0f}%")
        if route_strs:
            parts.append("経路=" + "/".join(route_strs))

    if state.empty_streak_counter > 0:
        parts.append(f"空入力連続={state.empty_streak_counter}")

    if state.holdback_units:
        parts.append(f"保留={len(state.holdback_units)}")

    if state.active_units:
        latest = state.active_units[-1]
        parts.append(f"最新経路={latest.route_type.value}")

    return " ".join(parts) if parts else "入力経路: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_text_dialogue_processor(
    config: Optional[TextDialogueConfig] = None,
) -> TextDialogueProcessor:
    """TextDialogueProcessorのファクトリ関数。"""
    return TextDialogueProcessor(config=config)
