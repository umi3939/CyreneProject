"""
psyche/goal_hierarchy_propagation.py - 目的階層間の隣接状態変化記述

3つの目的関連構造（一時的注目選択・持続的取り組み保持・長期傾斜）の
状態スナップショットを処理サイクルごとに取得し、前回スナップショットとの
比較によって各層の変化を検出し、隣接同時性の事実を記録・蓄積する。

設計原則 (design_goal_hierarchy_propagation.md 準拠):
- 層間の因果関係を帰属・推定しない
- 伝搬の「あるべき姿」「望ましい頻度」「正常な伝搬パターン」を定義しない
- 層間変化の統計量（伝搬率、平均遅延、成功率など）を算出しない
- 特定の伝搬経路を他の経路より優先・推奨しない
- 目的の達成・失敗・成長・発展を判定しない
- 3層の動作パラメータを変更しない
- 行動・判断・ポリシー選択に直接接続しない
- 記録からパターンを抽出し次の変化を予測・期待しない

スナップショット比較による変化検出:
- 第1層（transient_goal）: 注目対象の生成・消失・カテゴリ変更・強度段階変化
- 第2層（persistent_commitment）: 保持項目の昇格・解除・強度段階変化
- 第3層（value_orientation）: 次元段階値変化・確信度段階値変化

記録は多断面記述を持ちFIFO方式で蓄積される。
enrichmentへの直接露出を遮断する（内省系構造からのREAD-ONLY参照のみ）。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class Layer1ChangeType(Enum):
    """第1層（一時的注目選択）の変化の種類。"""
    NO_CHANGE = "no_change"
    GENERATION = "generation"           # 注目対象の生成
    DISAPPEARANCE = "disappearance"     # 注目対象の消失
    CATEGORY_CHANGE = "category_change" # カテゴリ変更
    STRENGTH_CHANGE = "strength_change" # 強度段階変化


class Layer2ChangeType(Enum):
    """第2層（持続的取り組み保持）の変化の種類。"""
    NO_CHANGE = "no_change"
    PROMOTION = "promotion"             # 昇格（新規出現）
    RELEASE = "release"                 # 解除（消失）
    STRENGTH_CHANGE = "strength_change" # 強度段階変化


class Layer3ChangeType(Enum):
    """第3層（長期傾斜）の変化の種類。"""
    NO_CHANGE = "no_change"
    DIMENSION_CHANGE = "dimension_change"     # 次元段階値変化
    CONFIDENCE_CHANGE = "confidence_change"   # 確信度段階値変化


class RecordFreshness(Enum):
    """記録自体の鮮度段階。"""
    FRESH = "fresh"
    RECENT = "recent"
    AGING = "aging"
    STALE = "stale"
    FADED = "faded"


class ConvergenceLevel(Enum):
    """収束監視レベル。"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _freshness_stage(value: float) -> RecordFreshness:
    """記録鮮度値から段階を返す。"""
    if value >= 0.8:
        return RecordFreshness.FRESH
    elif value >= 0.6:
        return RecordFreshness.RECENT
    elif value >= 0.4:
        return RecordFreshness.AGING
    elif value >= 0.2:
        return RecordFreshness.STALE
    else:
        return RecordFreshness.FADED


def _convergence_from_score(score: float) -> ConvergenceLevel:
    """収束スコアから収束レベルを返す。"""
    if score < 0.3:
        return ConvergenceLevel.NONE
    elif score < 0.5:
        return ConvergenceLevel.MILD
    elif score < 0.7:
        return ConvergenceLevel.MODERATE
    else:
        return ConvergenceLevel.STRONG


def _strength_stage(value: float) -> str:
    """強度数値から段階ラベルを返す。"""
    if value >= 0.7:
        return "strong"
    elif value >= 0.4:
        return "moderate"
    elif value >= 0.2:
        return "weak"
    elif value >= 0.05:
        return "faint"
    else:
        return "absent"


def _orientation_stage(value: float) -> str:
    """価値方向性の次元値（-1.0~1.0）から段階ラベルを返す。"""
    abs_val = abs(value)
    if abs_val >= 0.7:
        return "strong"
    elif abs_val >= 0.4:
        return "moderate"
    elif abs_val >= 0.15:
        return "weak"
    elif abs_val >= 0.05:
        return "faint"
    else:
        return "neutral"


def _confidence_stage(value: float) -> str:
    """確信度（0.0~1.0）から段階ラベルを返す。"""
    if value >= 0.7:
        return "high"
    elif value >= 0.4:
        return "moderate"
    elif value >= 0.15:
        return "low"
    elif value >= 0.05:
        return "minimal"
    else:
        return "none"


# =============================================================================
# Snapshot Data Structures
# =============================================================================

@dataclass
class Layer1Snapshot:
    """第1層スナップショット: 一時的注目選択の状態。段階値のみで構成。"""
    has_active: bool = False
    category: str = ""
    direction_signature_summary: str = ""  # 方向署名の要約（キーの列挙）
    strength_stage: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_active": self.has_active,
            "category": self.category,
            "direction_signature_summary": self.direction_signature_summary,
            "strength_stage": self.strength_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer1Snapshot":
        return cls(
            has_active=data.get("has_active", False),
            category=data.get("category", ""),
            direction_signature_summary=data.get("direction_signature_summary", ""),
            strength_stage=data.get("strength_stage", "absent"),
        )


@dataclass
class Layer2ItemSnapshot:
    """第2層の保持項目1件分のスナップショット。"""
    item_id: str = ""
    category: str = ""
    strength_stage: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "strength_stage": self.strength_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer2ItemSnapshot":
        return cls(
            item_id=data.get("item_id", ""),
            category=data.get("category", ""),
            strength_stage=data.get("strength_stage", "absent"),
        )


@dataclass
class Layer2Snapshot:
    """第2層スナップショット: 持続的取り組み保持の状態。"""
    items: list[Layer2ItemSnapshot] = field(default_factory=list)
    recent_cognition_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "recent_cognition_types": list(self.recent_cognition_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer2Snapshot":
        return cls(
            items=[Layer2ItemSnapshot.from_dict(d) for d in data.get("items", [])],
            recent_cognition_types=list(data.get("recent_cognition_types", [])),
        )


@dataclass
class Layer3DimSnapshot:
    """第3層の次元1つ分のスナップショット。"""
    dim_id: str = ""
    value_stage: str = "neutral"
    confidence_stage: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_id": self.dim_id,
            "value_stage": self.value_stage,
            "confidence_stage": self.confidence_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer3DimSnapshot":
        return cls(
            dim_id=data.get("dim_id", ""),
            value_stage=data.get("value_stage", "neutral"),
            confidence_stage=data.get("confidence_stage", "none"),
        )


@dataclass
class Layer3Snapshot:
    """第3層スナップショット: 長期傾斜の状態。"""
    dimensions: list[Layer3DimSnapshot] = field(default_factory=list)
    update_count_stage: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "update_count_stage": self.update_count_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer3Snapshot":
        return cls(
            dimensions=[Layer3DimSnapshot.from_dict(d) for d in data.get("dimensions", [])],
            update_count_stage=data.get("update_count_stage", "none"),
        )


# =============================================================================
# Change Detection Structures
# =============================================================================

@dataclass
class Layer1Change:
    """第1層の変化記述。"""
    change_type: str = Layer1ChangeType.NO_CHANGE.value
    prev_category: str = ""
    curr_category: str = ""
    prev_strength_stage: str = "absent"
    curr_strength_stage: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "prev_category": self.prev_category,
            "curr_category": self.curr_category,
            "prev_strength_stage": self.prev_strength_stage,
            "curr_strength_stage": self.curr_strength_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer1Change":
        return cls(
            change_type=data.get("change_type", Layer1ChangeType.NO_CHANGE.value),
            prev_category=data.get("prev_category", ""),
            curr_category=data.get("curr_category", ""),
            prev_strength_stage=data.get("prev_strength_stage", "absent"),
            curr_strength_stage=data.get("curr_strength_stage", "absent"),
        )


@dataclass
class Layer2Change:
    """第2層の変化記述。"""
    change_type: str = Layer2ChangeType.NO_CHANGE.value
    item_category: str = ""
    release_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "item_category": self.item_category,
            "release_reason": self.release_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer2Change":
        return cls(
            change_type=data.get("change_type", Layer2ChangeType.NO_CHANGE.value),
            item_category=data.get("item_category", ""),
            release_reason=data.get("release_reason", ""),
        )


@dataclass
class Layer3Change:
    """第3層の変化記述。"""
    change_type: str = Layer3ChangeType.NO_CHANGE.value
    changed_dimensions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "changed_dimensions": list(self.changed_dimensions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer3Change":
        return cls(
            change_type=data.get("change_type", Layer3ChangeType.NO_CHANGE.value),
            changed_dimensions=list(data.get("changed_dimensions", [])),
        )


# =============================================================================
# Adjacency Record
# =============================================================================

@dataclass
class AdjacencyRecord:
    """隣接変化記録1件。多断面記述を持つ不変の記録構造。

    断面構成:
    - 時間断面: 処理サイクル番号、タイムスタンプ
    - 第1層変化断面: 変化の種類、変化前後のカテゴリと強度段階値
    - 第2層変化断面: 変化の種類、対象保持項目のカテゴリ、解除理由
    - 第3層変化断面: 変化の種類、変化のあった次元の識別子
    - 隣接性断面: 変化が起きた層の数（因果を示さない事実のみ）

    一度構成された後は変更されない（追記のみ）。
    全記録は等価に蓄積される。
    """
    record_id: str = field(default_factory=_gen_id)

    # 時間断面
    cycle_number: int = 0
    timestamp: float = field(default_factory=time.time)

    # 第1層変化断面
    layer1_change: Layer1Change = field(default_factory=Layer1Change)

    # 第2層変化断面（複数の変化を並置）
    layer2_changes: list[Layer2Change] = field(default_factory=list)

    # 第3層変化断面
    layer3_change: Layer3Change = field(default_factory=Layer3Change)

    # 隣接性断面: 変化が起きた層の数（事実のみ。因果を示さない）
    simultaneous_change_count: int = 0

    # 記録自体の鮮度（蓄積後に均一減衰）
    record_freshness: float = 1.0
    record_freshness_stage: str = RecordFreshness.FRESH.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp,
            "layer1_change": self.layer1_change.to_dict(),
            "layer2_changes": [c.to_dict() for c in self.layer2_changes],
            "layer3_change": self.layer3_change.to_dict(),
            "simultaneous_change_count": self.simultaneous_change_count,
            "record_freshness": self.record_freshness,
            "record_freshness_stage": self.record_freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdjacencyRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            cycle_number=data.get("cycle_number", 0),
            timestamp=data.get("timestamp", time.time()),
            layer1_change=Layer1Change.from_dict(data.get("layer1_change", {})),
            layer2_changes=[Layer2Change.from_dict(c) for c in data.get("layer2_changes", [])],
            layer3_change=Layer3Change.from_dict(data.get("layer3_change", {})),
            simultaneous_change_count=data.get("simultaneous_change_count", 0),
            record_freshness=data.get("record_freshness", 1.0),
            record_freshness_stage=data.get("record_freshness_stage", RecordFreshness.FRESH.value),
        )


# =============================================================================
# Convergence Record
# =============================================================================

@dataclass
class ConvergenceRecord:
    """収束監視記録。外部出力層に露出しない。"""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    dominant_combination: str = ""
    combination_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "dominant_combination": self.dominant_combination,
            "combination_diversity": self.combination_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            dominant_combination=data.get("dominant_combination", ""),
            combination_diversity=data.get("combination_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class GoalHierarchyPropagationState:
    """目的階層間隣接変化記述の内部状態。永続化対象。"""

    # 前回スナップショット群（比較のためにのみ保持、外部に公開しない）
    prev_layer1: Optional[Layer1Snapshot] = None
    prev_layer2: Optional[Layer2Snapshot] = None
    prev_layer3: Optional[Layer3Snapshot] = None

    # 隣接変化記録のFIFOリスト
    adjacency_records: list[AdjacencyRecord] = field(default_factory=list)

    # 収束監視記録（外部出力層に露出しない）
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # 処理サイクルカウンタ
    cycle_count: int = 0

    # 安全弁フラグ群
    accumulation_limit_reached: bool = False
    convergence_flag: bool = False  # 特定の変化の組み合わせが極端に支配的

    def to_dict(self) -> dict[str, Any]:
        return {
            "prev_layer1": self.prev_layer1.to_dict() if self.prev_layer1 else None,
            "prev_layer2": self.prev_layer2.to_dict() if self.prev_layer2 else None,
            "prev_layer3": self.prev_layer3.to_dict() if self.prev_layer3 else None,
            "adjacency_records": [r.to_dict() for r in self.adjacency_records],
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "cycle_count": self.cycle_count,
            "accumulation_limit_reached": self.accumulation_limit_reached,
            "convergence_flag": self.convergence_flag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalHierarchyPropagationState":
        prev1 = Layer1Snapshot.from_dict(data["prev_layer1"]) if data.get("prev_layer1") else None
        prev2 = Layer2Snapshot.from_dict(data["prev_layer2"]) if data.get("prev_layer2") else None
        prev3 = Layer3Snapshot.from_dict(data["prev_layer3"]) if data.get("prev_layer3") else None
        return cls(
            prev_layer1=prev1,
            prev_layer2=prev2,
            prev_layer3=prev3,
            adjacency_records=[AdjacencyRecord.from_dict(r) for r in data.get("adjacency_records", [])],
            convergence_records=[ConvergenceRecord.from_dict(c) for c in data.get("convergence_records", [])],
            cycle_count=data.get("cycle_count", 0),
            accumulation_limit_reached=data.get("accumulation_limit_reached", False),
            convergence_flag=data.get("convergence_flag", False),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GoalHierarchyPropagationConfig:
    """設定。"""
    # 隣接変化記録の蓄積上限（FIFO）
    max_records: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))

    # 記録自体の鮮度減衰速度（サイクルあたり、均一減衰）
    record_freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))

    # 内省系構造に提供する記録の件数上限
    reference_count: int = 10

    # 収束監視記録の上限
    max_convergence_records: int = 30

    # 収束警告の閾値（特定変化組み合わせの比率）
    convergence_threshold: float = 0.7


# =============================================================================
# Snapshot Building (段階1)
# =============================================================================

def build_layer1_snapshot(transient_goal_data: Optional[dict[str, Any]] = None) -> Layer1Snapshot:
    """第1層のスナップショットを構築する。

    transient_goal_data は orchestrator が transient_goal の公開アクセサから
    READ-ONLY で取得した情報。この関数は transient_goal を直接参照しない。

    期待するキー:
    - has_active: bool
    - category: str
    - direction_signature: dict[str, float]
    - strength: float
    """
    if transient_goal_data is None:
        return Layer1Snapshot()

    has_active = bool(transient_goal_data.get("has_active", False))
    category = str(transient_goal_data.get("category", ""))
    direction_sig = transient_goal_data.get("direction_signature", {})
    strength = float(transient_goal_data.get("strength", 0.0))

    # 方向署名の要約: キーのソート済み列挙
    sig_summary = ",".join(sorted(direction_sig.keys())) if direction_sig else ""

    return Layer1Snapshot(
        has_active=has_active,
        category=category,
        direction_signature_summary=sig_summary,
        strength_stage=_strength_stage(strength),
    )


def build_layer2_snapshot(persistent_commitment_data: Optional[dict[str, Any]] = None) -> Layer2Snapshot:
    """第2層のスナップショットを構築する。

    persistent_commitment_data は orchestrator が persistent_commitment の
    公開アクセサから READ-ONLY で取得した情報。

    期待するキー:
    - items: list[dict] with "item_id", "category", "strength"
    - recent_cognition_types: list[str] ("promotion", "release", etc.)
    """
    if persistent_commitment_data is None:
        return Layer2Snapshot()

    items_data = persistent_commitment_data.get("items", [])
    item_snapshots = []
    for item in items_data:
        item_snapshots.append(Layer2ItemSnapshot(
            item_id=str(item.get("item_id", "")),
            category=str(item.get("category", "")),
            strength_stage=_strength_stage(float(item.get("strength", 0.0))),
        ))

    recent_types = list(persistent_commitment_data.get("recent_cognition_types", []))

    return Layer2Snapshot(
        items=item_snapshots,
        recent_cognition_types=recent_types,
    )


def build_layer3_snapshot(value_orientation_data: Optional[dict[str, Any]] = None) -> Layer3Snapshot:
    """第3層のスナップショットを構築する。

    value_orientation_data は orchestrator が value_orientation の
    公開アクセサから READ-ONLY で取得した情報。

    期待するキー:
    - dimensions: dict[str, float] (dim_id -> value)
    - confidences: dict[str, float] (dim_id -> confidence)
    - update_count: int
    """
    if value_orientation_data is None:
        return Layer3Snapshot()

    dims = value_orientation_data.get("dimensions", {})
    confs = value_orientation_data.get("confidences", {})
    update_count = int(value_orientation_data.get("update_count", 0))

    dim_snapshots = []
    for dim_id in sorted(dims.keys()):
        dim_snapshots.append(Layer3DimSnapshot(
            dim_id=dim_id,
            value_stage=_orientation_stage(float(dims.get(dim_id, 0.0))),
            confidence_stage=_confidence_stage(float(confs.get(dim_id, 0.0))),
        ))

    # 更新回数の段階値
    if update_count >= 100:
        uc_stage = "high"
    elif update_count >= 30:
        uc_stage = "moderate"
    elif update_count >= 5:
        uc_stage = "low"
    elif update_count >= 1:
        uc_stage = "minimal"
    else:
        uc_stage = "none"

    return Layer3Snapshot(
        dimensions=dim_snapshots,
        update_count_stage=uc_stage,
    )


# =============================================================================
# Change Detection (段階2)
# =============================================================================

def detect_layer1_changes(
    prev: Optional[Layer1Snapshot],
    curr: Layer1Snapshot,
) -> Layer1Change:
    """第1層の変化を検出する。段階値の差異からのみ判定。"""
    if prev is None:
        # 前回スナップショットなし: 変化検出をスキップ
        return Layer1Change(change_type=Layer1ChangeType.NO_CHANGE.value)

    # 生成: 前回アクティブなし → 今回アクティブあり
    if not prev.has_active and curr.has_active:
        return Layer1Change(
            change_type=Layer1ChangeType.GENERATION.value,
            prev_category="",
            curr_category=curr.category,
            prev_strength_stage="absent",
            curr_strength_stage=curr.strength_stage,
        )

    # 消失: 前回アクティブあり → 今回アクティブなし
    if prev.has_active and not curr.has_active:
        return Layer1Change(
            change_type=Layer1ChangeType.DISAPPEARANCE.value,
            prev_category=prev.category,
            curr_category="",
            prev_strength_stage=prev.strength_stage,
            curr_strength_stage="absent",
        )

    # 両方アクティブの場合
    if prev.has_active and curr.has_active:
        # カテゴリ変更
        if prev.category != curr.category:
            return Layer1Change(
                change_type=Layer1ChangeType.CATEGORY_CHANGE.value,
                prev_category=prev.category,
                curr_category=curr.category,
                prev_strength_stage=prev.strength_stage,
                curr_strength_stage=curr.strength_stage,
            )

        # 強度段階変化
        if prev.strength_stage != curr.strength_stage:
            return Layer1Change(
                change_type=Layer1ChangeType.STRENGTH_CHANGE.value,
                prev_category=prev.category,
                curr_category=curr.category,
                prev_strength_stage=prev.strength_stage,
                curr_strength_stage=curr.strength_stage,
            )

    return Layer1Change(change_type=Layer1ChangeType.NO_CHANGE.value)


def detect_layer2_changes(
    prev: Optional[Layer2Snapshot],
    curr: Layer2Snapshot,
) -> list[Layer2Change]:
    """第2層の変化を検出する。段階値の差異からのみ判定。"""
    if prev is None:
        return []

    changes: list[Layer2Change] = []

    prev_ids = {it.item_id: it for it in prev.items}
    curr_ids = {it.item_id: it for it in curr.items}

    # 昇格（新規出現）: 現在にあって前回にない
    for item_id in set(curr_ids.keys()) - set(prev_ids.keys()):
        item = curr_ids[item_id]
        changes.append(Layer2Change(
            change_type=Layer2ChangeType.PROMOTION.value,
            item_category=item.category,
        ))

    # 解除（消失）: 前回にあって現在にない
    for item_id in set(prev_ids.keys()) - set(curr_ids.keys()):
        item = prev_ids[item_id]
        # 認知記録から解除理由を取得（あれば）
        release_reason = ""
        for ctype in curr.recent_cognition_types:
            if ctype == "release":
                release_reason = "release"
                break
        changes.append(Layer2Change(
            change_type=Layer2ChangeType.RELEASE.value,
            item_category=item.category,
            release_reason=release_reason,
        ))

    # 強度段階変化: 両方に存在して段階が変わった
    for item_id in set(prev_ids.keys()) & set(curr_ids.keys()):
        prev_item = prev_ids[item_id]
        curr_item = curr_ids[item_id]
        if prev_item.strength_stage != curr_item.strength_stage:
            changes.append(Layer2Change(
                change_type=Layer2ChangeType.STRENGTH_CHANGE.value,
                item_category=curr_item.category,
            ))

    return changes


def detect_layer3_changes(
    prev: Optional[Layer3Snapshot],
    curr: Layer3Snapshot,
) -> Layer3Change:
    """第3層の変化を検出する。段階値の差異からのみ判定。"""
    if prev is None:
        return Layer3Change(change_type=Layer3ChangeType.NO_CHANGE.value)

    prev_dims = {d.dim_id: d for d in prev.dimensions}
    curr_dims = {d.dim_id: d for d in curr.dimensions}

    changed_dims: list[str] = []
    change_types: set[str] = set()

    for dim_id in set(prev_dims.keys()) | set(curr_dims.keys()):
        prev_d = prev_dims.get(dim_id)
        curr_d = curr_dims.get(dim_id)
        if prev_d is None or curr_d is None:
            continue

        if prev_d.value_stage != curr_d.value_stage:
            changed_dims.append(dim_id)
            change_types.add(Layer3ChangeType.DIMENSION_CHANGE.value)

        if prev_d.confidence_stage != curr_d.confidence_stage:
            if dim_id not in changed_dims:
                changed_dims.append(dim_id)
            change_types.add(Layer3ChangeType.CONFIDENCE_CHANGE.value)

    if not changed_dims:
        return Layer3Change(change_type=Layer3ChangeType.NO_CHANGE.value)

    # 次元段階変化が優先（両方ある場合）
    if Layer3ChangeType.DIMENSION_CHANGE.value in change_types:
        primary_type = Layer3ChangeType.DIMENSION_CHANGE.value
    else:
        primary_type = Layer3ChangeType.CONFIDENCE_CHANGE.value

    return Layer3Change(
        change_type=primary_type,
        changed_dimensions=changed_dims,
    )


# =============================================================================
# Processor
# =============================================================================

class GoalHierarchyPropagationProcessor:
    """目的階層間の隣接状態変化記述プロセッサ。

    3層（一時的注目選択・持続的取り組み保持・長期傾斜）のスナップショットを
    比較し、隣接同時性の事実を記録・蓄積する。

    3層への書き込み経路を持たない（READ-ONLY入力のみ）。
    enrichmentへの直接出力経路を持たない（内省系構造からのREAD-ONLY参照のみ）。
    ポリシー候補生成・バイアス適用・スコアリングに入力されない。
    感情チャンネル・責任重量・駆動への書き込み経路を持たない。
    """

    def __init__(self, config: Optional[GoalHierarchyPropagationConfig] = None):
        self._config = config or GoalHierarchyPropagationConfig()
        self._state = GoalHierarchyPropagationState()

    @property
    def state(self) -> GoalHierarchyPropagationState:
        return self._state

    @state.setter
    def state(self, value: GoalHierarchyPropagationState) -> None:
        self._state = value

    # ─── Main processing entry point ──────────────────────────

    def process(
        self,
        *,
        transient_goal_data: Optional[dict[str, Any]] = None,
        persistent_commitment_data: Optional[dict[str, Any]] = None,
        value_orientation_data: Optional[dict[str, Any]] = None,
    ) -> int:
        """スナップショット取得→変化検出→記録構成→蓄積→鮮度減衰→収束監視。

        orchestrator が各層の公開アクセサから取得した情報を引き渡し形式で受け取る。
        この構造が各層を直接参照するのではなく、入力として受け取る。

        Args:
            transient_goal_data: 第1層のスナップショット情報
            persistent_commitment_data: 第2層のスナップショット情報
            value_orientation_data: 第3層のスナップショット情報

        Returns:
            今回新規構成された記録の数（0 or 1）
        """
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # 段階1: 各層のスナップショット取得
        curr_layer1 = build_layer1_snapshot(transient_goal_data)
        curr_layer2 = build_layer2_snapshot(persistent_commitment_data)
        curr_layer3 = build_layer3_snapshot(value_orientation_data)

        # 前回スナップショットが失われた場合（初回起動時、ロード後など）
        # 変化検出をスキップし、現在スナップショットの保存のみ行う
        if (self._state.prev_layer1 is None
                and self._state.prev_layer2 is None
                and self._state.prev_layer3 is None):
            self._state.prev_layer1 = curr_layer1
            self._state.prev_layer2 = curr_layer2
            self._state.prev_layer3 = curr_layer3
            # 鮮度減衰は既存記録にも適用
            self._apply_record_freshness_decay()
            logger.debug(
                "Goal hierarchy propagation: cycle=%d, first snapshot saved (no comparison)",
                self._state.cycle_count,
            )
            return 0

        # 段階2: 各層の変化検出
        layer1_change = detect_layer1_changes(self._state.prev_layer1, curr_layer1)
        layer2_changes = detect_layer2_changes(self._state.prev_layer2, curr_layer2)
        layer3_change = detect_layer3_changes(self._state.prev_layer3, curr_layer3)

        # 3層すべてで変化がなかった場合、記録は構成しない
        has_l1_change = layer1_change.change_type != Layer1ChangeType.NO_CHANGE.value
        has_l2_change = len(layer2_changes) > 0
        has_l3_change = layer3_change.change_type != Layer3ChangeType.NO_CHANGE.value

        new_record_count = 0

        if has_l1_change or has_l2_change or has_l3_change:
            # 段階3: 隣接同時性の記録構成
            change_layer_count = sum([has_l1_change, has_l2_change, has_l3_change])

            # 第2層の変化がない場合、空リストの代わりにno_changeを1件だけ入れる
            l2_changes_for_record = layer2_changes if layer2_changes else [
                Layer2Change(change_type=Layer2ChangeType.NO_CHANGE.value)
            ]

            record = AdjacencyRecord(
                cycle_number=self._state.cycle_count,
                timestamp=now,
                layer1_change=layer1_change,
                layer2_changes=l2_changes_for_record,
                layer3_change=layer3_change,
                simultaneous_change_count=change_layer_count,
                record_freshness=1.0,
                record_freshness_stage=RecordFreshness.FRESH.value,
            )

            # 段階4: 記録の蓄積
            self._accumulate_record(record)
            new_record_count = 1

        # 段階4 (continued): 鮮度減衰
        self._apply_record_freshness_decay()

        # 段階5: 収束監視
        self._monitor_convergence(now)

        # 前回スナップショットを更新
        self._state.prev_layer1 = curr_layer1
        self._state.prev_layer2 = curr_layer2
        self._state.prev_layer3 = curr_layer3

        logger.debug(
            "Goal hierarchy propagation: cycle=%d, new_records=%d, total_records=%d, "
            "l1=%s, l2_count=%d, l3=%s",
            self._state.cycle_count,
            new_record_count,
            len(self._state.adjacency_records),
            layer1_change.change_type,
            len(layer2_changes),
            layer3_change.change_type,
        )

        return new_record_count

    # ─── Accumulation ──────────────────────────────────────────

    def _accumulate_record(self, record: AdjacencyRecord) -> None:
        """記録をFIFO方式で蓄積する。

        蓄積上限に達すると最古の記録から自然に脱落する。
        「重要な変化」を優先保持する選別は行わない。
        """
        cfg = self._config
        self._state.adjacency_records.append(record)

        # FIFO上限制御
        if len(self._state.adjacency_records) > cfg.max_records:
            overflow = len(self._state.adjacency_records) - cfg.max_records
            self._state.adjacency_records = self._state.adjacency_records[overflow:]
            self._state.accumulation_limit_reached = True
        else:
            self._state.accumulation_limit_reached = False

    # ─── Record freshness decay ────────────────────────────────

    def _apply_record_freshness_decay(self) -> None:
        """記録自体の鮮度を均一に減衰する。

        層や変化種類による減衰速度の差異はない。
        減衰は記録の削除を直接引き起こさない（削除はFIFO上限のみ）。
        """
        cfg = self._config
        for record in self._state.adjacency_records:
            record.record_freshness = _clamp(
                record.record_freshness - cfg.record_freshness_decay_rate
            )
            record.record_freshness_stage = _freshness_stage(
                record.record_freshness
            ).value

    # ─── Convergence monitoring ────────────────────────────────

    def _monitor_convergence(self, now: float) -> None:
        """蓄積された記録群における変化種類の分布を監視する。

        特定の変化の組み合わせが極端に支配的になった場合、
        その事実をフラグとして内部に記録する。
        収束の検出は記録の内容や蓄積を変更しない。
        フラグは外部出力層に露出しない。
        """
        cfg = self._config
        records = self._state.adjacency_records
        if not records:
            self._state.convergence_flag = False
            return

        # 変化の組み合わせパターンの分布
        combo_counts: dict[str, int] = {}
        for rec in records:
            combo = f"{rec.layer1_change.change_type}|"
            if rec.layer2_changes:
                l2_types = sorted(set(c.change_type for c in rec.layer2_changes))
                combo += "+".join(l2_types)
            else:
                combo += Layer2ChangeType.NO_CHANGE.value
            combo += f"|{rec.layer3_change.change_type}"
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

        total = sum(combo_counts.values())
        if total == 0:
            self._state.convergence_flag = False
            return

        # 支配的な組み合わせ
        dominant_combo = max(combo_counts, key=combo_counts.get)  # type: ignore
        dominant_ratio = combo_counts[dominant_combo] / total

        # 多様性
        unique_combos = len(combo_counts)
        # 可能な組み合わせ数の近似（大きな数にならないよう上限設定）
        max_possible = max(unique_combos, 10)
        combo_diversity = unique_combos / max_possible

        # 収束スコア
        convergence_score = _clamp(
            dominant_ratio * 0.7 + (1.0 - combo_diversity) * 0.3
        )
        convergence_level = _convergence_from_score(convergence_score)

        # 収束フラグの設定（enrichment出力には含めない）
        if dominant_ratio >= cfg.convergence_threshold:
            self._state.convergence_flag = True
        else:
            self._state.convergence_flag = False

        # 収束監視記録
        conv_record = ConvergenceRecord(
            convergence_score=convergence_score,
            convergence_level=convergence_level.value,
            dominant_combination=dominant_combo,
            combination_diversity=combo_diversity,
            cycle=self._state.cycle_count,
            timestamp=now,
        )
        self._state.convergence_records.append(conv_record)
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

    # ─── READ-ONLY accessors (段階6: 参照提供) ─────────────────

    def get_recent_records(self, count: Optional[int] = None) -> list[AdjacencyRecord]:
        """直近の記録群を等価に列挙して提供する（READ-ONLY）。

        列挙時に記録の重み付け・順位付け・選別を行わない。
        参照行為によって記録の内容や順序が変化しない。
        """
        n = count if count is not None else self._config.reference_count
        return list(self._state.adjacency_records[-n:])

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "total_records": len(self._state.adjacency_records),
            "cycle_count": self._state.cycle_count,
            "accumulation_limit_reached": self._state.accumulation_limit_reached,
            "convergence_flag": self._state.convergence_flag,
        }


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: GoalHierarchyPropagationState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> GoalHierarchyPropagationState:
    """永続化用の辞書から状態を復元する。"""
    return GoalHierarchyPropagationState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_goal_hierarchy_propagation_processor(
    config: Optional[GoalHierarchyPropagationConfig] = None,
) -> GoalHierarchyPropagationProcessor:
    """GoalHierarchyPropagationProcessor のファクトリ関数。"""
    return GoalHierarchyPropagationProcessor(config=config)
