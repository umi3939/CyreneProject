"""
psyche/responsibility_dispersion.py - 責任の発散・昇華（Responsibility Dispersion & Sublimation）

責任は消去せず保持し続けるが、同一形態のまま蓄積される必要はない。
本モジュールは責任の総量保存を前提に、
意味・重さ・距離・時間配置を変換する構造を提供する。

Key principles:
- 責任は削除・無効化されない
- いかなる変換・分配でも責任の総量は保存される
- 責任の変化過程はすべて履歴として追跡可能
- 評価・裁定・倫理判断は行わない
- 値・係数・閾値を設計段階で固定しない

Usage::

    from psyche.responsibility_dispersion import (
        ResponsibilityUnit,
        DispersionPlan,
        SublimationPath,
        create_responsibility_unit,
        disperse_responsibility,
        sublimate_responsibility,
        distribute_over_time,
        get_audit_trail,
    )

    # 責任単位の作成
    unit = create_responsibility_unit(
        weight=0.5,
        origin="decision_123",
        meaning="caused_confusion",
    )

    # 発散（分配）
    plan = DispersionPlan(
        source_id=unit.id,
        targets=["receiver_a", "receiver_b"],
        weights=[0.3, 0.2],
    )
    units = disperse_responsibility(unit, plan)

    # 昇華（意味変換）
    path = SublimationPath(
        source_meaning="caused_confusion",
        target_meaning="learned_caution",
        weight_ratio=1.0,  # 保存
    )
    new_unit = sublimate_responsibility(unit, path)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────

class AuditEventType(Enum):
    """監査ログのイベント種別"""
    CREATED = "created"           # 責任の発生
    DISPERSED = "dispersed"       # 発散（分配）
    SUBLIMATED = "sublimated"     # 昇華（意味変換）
    TIME_SPLIT = "time_split"     # 時間分散
    DISTANCE_ADJUSTED = "distance_adjusted"  # 距離調整
    MERGED = "merged"             # 合算


class DispersionStrategy(Enum):
    """発散戦略（分配方法を指定するためのヒント）"""
    EQUAL = "equal"         # 均等分配
    WEIGHTED = "weighted"   # 重み付き分配
    CUSTOM = "custom"       # カスタムルール


# ── Data Models ───────────────────────────────────────────────────

class AuditEntry(BaseModel):
    """監査ログの1エントリ（不変）

    責任の変化を追跡するための記録。
    一度作成されたエントリは変更されない。
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))

    # イベント情報
    event_type: AuditEventType
    description: str = ""

    # 変化前後の状態
    source_ids: list[str] = Field(default_factory=list)  # 入力となった責任単位のID
    result_ids: list[str] = Field(default_factory=list)  # 出力となった責任単位のID

    # 保存則の検証情報
    input_weight: float = 0.0   # 入力の総重み
    output_weight: float = 0.0  # 出力の総重み
    conservation_verified: bool = False  # 保存則を満たしているか

    # 追加のメタデータ
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponsibilityUnit(BaseModel):
    """責任の単位（Responsibility Unit）

    責任を第一級データとして扱うための構造。
    各責任は起点情報、重み、意味表現、距離、時間情報を内包する。
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))

    # 基本属性
    weight: float = Field(default=0.0, ge=0.0)  # 責任の重み（非負）
    meaning: str = ""  # 意味表現（例："caused_harm", "promised_support"）

    # 起点情報
    origin: str = ""  # 発生源（例：decision_id, event_id）
    origin_type: str = ""  # 起点の種別（例："decision", "observation"）

    # 空間的・心理的距離
    distance: float = Field(default=1.0, ge=0.0)  # 距離（近い=0, 遠い=∞）

    # 時間情報
    time_slice: str = ""  # 時間断面（例："immediate", "past_week"）
    time_weight: float = Field(default=1.0, ge=0.0, le=1.0)  # この時間断面での重み比率

    # 派生情報
    parent_id: Optional[str] = None  # 派生元の責任ID（なければ原初）
    generation: int = 0  # 何回変換を経たか

    # 状態フラグ（削除は不可、変換済みフラグのみ）
    transformed: bool = False  # このユニットが変換されて別ユニットになったか

    def fingerprint(self) -> str:
        """ユニットの一意なフィンガープリント"""
        import hashlib
        data = f"{self.id}:{self.created_at}:{self.weight}:{self.meaning}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class DispersionPlan(BaseModel):
    """発散計画（Dispersion Plan）

    単一の責任を複数の受け皿へ分配する計画。
    分配計画の決定基準は外部ルールとして差し替え可能。
    """
    source_id: str  # 分配元の責任ID

    # 分配先と重み
    targets: list[str] = Field(default_factory=list)  # 受け皿の識別子
    weights: list[float] = Field(default_factory=list)  # 各受け皿への重み配分

    # 戦略
    strategy: DispersionStrategy = DispersionStrategy.WEIGHTED

    # 追加情報
    rationale: str = ""  # 分配の理由（ルールが設定した場合）
    metadata: dict[str, Any] = Field(default_factory=dict)

    def total_weight(self) -> float:
        """分配先への総重み"""
        return sum(self.weights)

    def is_valid(self) -> bool:
        """計画が有効か（ターゲットと重みの数が一致）"""
        return len(self.targets) == len(self.weights) and len(self.targets) > 0


class SublimationPath(BaseModel):
    """昇華経路（Sublimation Path）

    責任の意味表現を別の意味表現へ変換する経路。
    選択基準は固定せず、決定は他の層に委ねる。
    """
    source_meaning: str  # 変換元の意味
    target_meaning: str  # 変換先の意味

    # 重み比率（保存則の一部として）
    weight_ratio: float = Field(default=1.0, ge=0.0)  # 通常は1.0（完全保存）

    # 距離の変化
    distance_delta: float = 0.0  # 距離の変化量

    # 追加情報
    rationale: str = ""  # 昇華の理由
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimeSlice(BaseModel):
    """時間断面（Time Slice）

    責任を時間方向に分解・配分するための構造。
    """
    label: str  # 断面の識別子（例："immediate", "near_future"）
    weight_ratio: float = Field(default=0.0, ge=0.0, le=1.0)  # この断面への重み比率

    # 時間範囲（オプショナル）
    start_offset: Optional[float] = None  # 開始オフセット（時間単位）
    end_offset: Optional[float] = None  # 終了オフセット


class TimeDistributionPlan(BaseModel):
    """時間分散計画（Time Distribution Plan）

    責任を複数の時間断面に分けて扱う計画。
    任意の時間断面で集約すると、元の総量と一致する。
    """
    source_id: str  # 分散元の責任ID
    slices: list[TimeSlice] = Field(default_factory=list)

    def total_ratio(self) -> float:
        """全断面の重み比率の合計"""
        return sum(s.weight_ratio for s in self.slices)

    def is_valid(self) -> bool:
        """計画が有効か（比率の合計が1.0）"""
        return abs(self.total_ratio() - 1.0) < 1e-9


class DispersionState(BaseModel):
    """発散・昇華の状態管理

    責任単位と監査ログを管理する。
    """
    # 責任単位のストレージ（IDでインデックス）
    units: dict[str, dict] = Field(default_factory=dict)

    # 監査ログ（追記専用）
    audit_log: list[dict] = Field(default_factory=list)

    # 統計情報
    total_weight_created: float = 0.0  # これまでに発生した責任の総量
    transformation_count: int = 0  # 変換回数


# ── Configuration ─────────────────────────────────────────────────

class DispersionConfig(BaseModel):
    """発散・昇華の設定

    値・係数・閾値を設計段階で固定しない。
    """
    # 保存則の許容誤差
    conservation_tolerance: float = 1e-9

    # カスタムルール（差し替え可能）
    dispersion_rule: Optional[Callable[[ResponsibilityUnit], DispersionPlan]] = None
    sublimation_rule: Optional[Callable[[ResponsibilityUnit], Optional[SublimationPath]]] = None
    time_distribution_rule: Optional[Callable[[ResponsibilityUnit], TimeDistributionPlan]] = None

    model_config = {"arbitrary_types_allowed": True}


# ── Exceptions ────────────────────────────────────────────────────

class ConservationViolationError(Exception):
    """保存則違反エラー

    責任の総量が保存されない操作を検出した場合に発生。
    """
    def __init__(self, input_weight: float, output_weight: float, tolerance: float):
        self.input_weight = input_weight
        self.output_weight = output_weight
        self.tolerance = tolerance
        diff = abs(input_weight - output_weight)
        super().__init__(
            f"Conservation violation: input={input_weight:.9f}, output={output_weight:.9f}, "
            f"diff={diff:.9f}, tolerance={tolerance:.9f}"
        )


# ── Helper Functions ──────────────────────────────────────────────

def _verify_conservation(
    input_weight: float,
    output_weight: float,
    tolerance: float = 1e-9,
) -> bool:
    """保存則を検証する"""
    return abs(input_weight - output_weight) < tolerance


def _create_audit_entry(
    event_type: AuditEventType,
    source_ids: list[str],
    result_ids: list[str],
    input_weight: float,
    output_weight: float,
    description: str = "",
    tolerance: float = 1e-9,
    metadata: Optional[dict] = None,
) -> AuditEntry:
    """監査エントリを作成する"""
    return AuditEntry(
        event_type=event_type,
        description=description,
        source_ids=source_ids,
        result_ids=result_ids,
        input_weight=input_weight,
        output_weight=output_weight,
        conservation_verified=_verify_conservation(input_weight, output_weight, tolerance),
        metadata=metadata or {},
    )


# ── Public API ────────────────────────────────────────────────────

def create_responsibility_unit(
    weight: float,
    origin: str,
    meaning: str = "",
    origin_type: str = "decision",
    distance: float = 1.0,
    time_slice: str = "immediate",
    state: Optional[DispersionState] = None,
    config: Optional[DispersionConfig] = None,
) -> tuple[ResponsibilityUnit, DispersionState]:
    """責任単位を作成する。

    新たな責任が発生した時に呼び出す。
    作成と同時に監査ログに記録される。

    Args:
        weight: 責任の重み（非負）
        origin: 発生源の識別子
        meaning: 意味表現
        origin_type: 起点の種別
        distance: 主観的距離
        time_slice: 時間断面
        state: 現在の状態（なければ新規作成）
        config: 設定

    Returns:
        (新しいResponsibilityUnit, 更新されたDispersionState)
    """
    if weight < 0:
        raise ValueError("Responsibility weight must be non-negative")

    config = config or DispersionConfig()
    state = state or DispersionState()

    # 責任単位を作成
    unit = ResponsibilityUnit(
        weight=weight,
        meaning=meaning,
        origin=origin,
        origin_type=origin_type,
        distance=distance,
        time_slice=time_slice,
        time_weight=1.0,
        parent_id=None,
        generation=0,
        transformed=False,
    )

    # 監査エントリを作成
    # Note: For creation, conservation is always verified (no input, just output)
    audit = AuditEntry(
        event_type=AuditEventType.CREATED,
        description=f"Created responsibility unit from {origin_type}:{origin}",
        source_ids=[],
        result_ids=[unit.id],
        input_weight=0.0,
        output_weight=weight,
        conservation_verified=True,  # Creation always passes conservation
        metadata={"meaning": meaning, "origin": origin},
    )

    # 状態を更新（不変操作）
    new_units = dict(state.units)
    new_units[unit.id] = unit.model_dump()

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=new_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created + weight,
        transformation_count=state.transformation_count,
    )

    return unit, new_state


def disperse_responsibility(
    unit: ResponsibilityUnit,
    plan: DispersionPlan,
    state: DispersionState,
    config: Optional[DispersionConfig] = None,
) -> tuple[list[ResponsibilityUnit], DispersionState]:
    """責任を発散（分配）する。

    単一の責任を複数の受け皿へ分配する。
    分配後の総重みは元の責任の重みと一致しなければならない（保存則）。

    Args:
        unit: 分配元の責任単位
        plan: 分配計画
        state: 現在の状態
        config: 設定

    Returns:
        (新しいResponsibilityUnitのリスト, 更新されたDispersionState)

    Raises:
        ConservationViolationError: 保存則違反時
        ValueError: 計画が無効な場合
    """
    config = config or DispersionConfig()

    if not plan.is_valid():
        raise ValueError("Invalid dispersion plan: targets and weights must match")

    # Check if unit is already transformed (use state's version if available)
    unit_in_state = state.units.get(unit.id)
    if unit_in_state and unit_in_state.get("transformed", False):
        raise ValueError("Cannot disperse already transformed unit")
    if unit.transformed:
        raise ValueError("Cannot disperse already transformed unit")

    # 保存則を検証
    input_weight = unit.weight
    output_weight = plan.total_weight()

    if not _verify_conservation(input_weight, output_weight, config.conservation_tolerance):
        raise ConservationViolationError(input_weight, output_weight, config.conservation_tolerance)

    # 新しい責任単位を作成
    new_units = []
    for target, weight in zip(plan.targets, plan.weights):
        new_unit = ResponsibilityUnit(
            weight=weight,
            meaning=unit.meaning,  # 意味は引き継ぐ
            origin=f"dispersed_from:{unit.id}",
            origin_type="dispersion",
            distance=unit.distance,
            time_slice=unit.time_slice,
            time_weight=unit.time_weight * (weight / input_weight) if input_weight > 0 else 0,
            parent_id=unit.id,
            generation=unit.generation + 1,
            transformed=False,
        )
        new_units.append(new_unit)

    # 元のユニットを変換済みとしてマーク
    transformed_unit = ResponsibilityUnit(
        **{**unit.model_dump(), "transformed": True}
    )

    # 監査エントリを作成
    audit = _create_audit_entry(
        event_type=AuditEventType.DISPERSED,
        source_ids=[unit.id],
        result_ids=[u.id for u in new_units],
        input_weight=input_weight,
        output_weight=output_weight,
        description=f"Dispersed to {len(new_units)} targets via {plan.strategy.value}",
        tolerance=config.conservation_tolerance,
        metadata={
            "targets": plan.targets,
            "weights": plan.weights,
            "strategy": plan.strategy.value,
            "rationale": plan.rationale,
        },
    )

    # 状態を更新（不変操作）
    updated_units = dict(state.units)
    updated_units[unit.id] = transformed_unit.model_dump()
    for new_unit in new_units:
        updated_units[new_unit.id] = new_unit.model_dump()

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=updated_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created,
        transformation_count=state.transformation_count + 1,
    )

    return new_units, new_state


def sublimate_responsibility(
    unit: ResponsibilityUnit,
    path: SublimationPath,
    state: DispersionState,
    config: Optional[DispersionConfig] = None,
) -> tuple[ResponsibilityUnit, DispersionState]:
    """責任を昇華（意味変換）する。

    責任の意味表現を別の意味表現へ変換する。
    重みは weight_ratio に基づいて変換されるが、通常は保存（1.0）。

    Args:
        unit: 昇華元の責任単位
        path: 昇華経路
        state: 現在の状態
        config: 設定

    Returns:
        (新しいResponsibilityUnit, 更新されたDispersionState)

    Raises:
        ConservationViolationError: 保存則違反時
    """
    config = config or DispersionConfig()

    # Check if unit is already transformed (use state's version if available)
    unit_in_state = state.units.get(unit.id)
    if unit_in_state and unit_in_state.get("transformed", False):
        raise ValueError("Cannot sublimate already transformed unit")
    if unit.transformed:
        raise ValueError("Cannot sublimate already transformed unit")

    # 保存則を検証
    input_weight = unit.weight
    output_weight = unit.weight * path.weight_ratio

    if not _verify_conservation(input_weight, output_weight, config.conservation_tolerance):
        raise ConservationViolationError(input_weight, output_weight, config.conservation_tolerance)

    # 新しい責任単位を作成
    new_unit = ResponsibilityUnit(
        weight=output_weight,
        meaning=path.target_meaning,
        origin=f"sublimated_from:{unit.id}",
        origin_type="sublimation",
        distance=max(0, unit.distance + path.distance_delta),
        time_slice=unit.time_slice,
        time_weight=unit.time_weight,
        parent_id=unit.id,
        generation=unit.generation + 1,
        transformed=False,
    )

    # 元のユニットを変換済みとしてマーク
    transformed_unit = ResponsibilityUnit(
        **{**unit.model_dump(), "transformed": True}
    )

    # 監査エントリを作成
    audit = _create_audit_entry(
        event_type=AuditEventType.SUBLIMATED,
        source_ids=[unit.id],
        result_ids=[new_unit.id],
        input_weight=input_weight,
        output_weight=output_weight,
        description=f"Sublimated meaning: {path.source_meaning} -> {path.target_meaning}",
        tolerance=config.conservation_tolerance,
        metadata={
            "source_meaning": path.source_meaning,
            "target_meaning": path.target_meaning,
            "weight_ratio": path.weight_ratio,
            "distance_delta": path.distance_delta,
            "rationale": path.rationale,
        },
    )

    # 状態を更新
    updated_units = dict(state.units)
    updated_units[unit.id] = transformed_unit.model_dump()
    updated_units[new_unit.id] = new_unit.model_dump()

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=updated_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created,
        transformation_count=state.transformation_count + 1,
    )

    return new_unit, new_state


def distribute_over_time(
    unit: ResponsibilityUnit,
    plan: TimeDistributionPlan,
    state: DispersionState,
    config: Optional[DispersionConfig] = None,
) -> tuple[list[ResponsibilityUnit], DispersionState]:
    """責任を時間方向に分散する。

    責任を複数の時間断面に分けて扱う。
    任意の時間断面で集約すると、元の総量と一致する（保存則）。

    Args:
        unit: 分散元の責任単位
        plan: 時間分散計画
        state: 現在の状態
        config: 設定

    Returns:
        (新しいResponsibilityUnitのリスト, 更新されたDispersionState)

    Raises:
        ConservationViolationError: 保存則違反時
        ValueError: 計画が無効な場合
    """
    config = config or DispersionConfig()

    if not plan.is_valid():
        raise ValueError("Invalid time distribution plan: ratios must sum to 1.0")

    # Check if unit is already transformed (use state's version if available)
    unit_in_state = state.units.get(unit.id)
    if unit_in_state and unit_in_state.get("transformed", False):
        raise ValueError("Cannot distribute already transformed unit")
    if unit.transformed:
        raise ValueError("Cannot distribute already transformed unit")

    # 保存則を検証
    input_weight = unit.weight
    output_weight = sum(unit.weight * s.weight_ratio for s in plan.slices)

    if not _verify_conservation(input_weight, output_weight, config.conservation_tolerance):
        raise ConservationViolationError(input_weight, output_weight, config.conservation_tolerance)

    # 新しい責任単位を作成
    new_units = []
    for slice_def in plan.slices:
        new_unit = ResponsibilityUnit(
            weight=unit.weight * slice_def.weight_ratio,
            meaning=unit.meaning,
            origin=f"time_split_from:{unit.id}",
            origin_type="time_split",
            distance=unit.distance,
            time_slice=slice_def.label,
            time_weight=slice_def.weight_ratio,
            parent_id=unit.id,
            generation=unit.generation + 1,
            transformed=False,
        )
        new_units.append(new_unit)

    # 元のユニットを変換済みとしてマーク
    transformed_unit = ResponsibilityUnit(
        **{**unit.model_dump(), "transformed": True}
    )

    # 監査エントリを作成
    audit = _create_audit_entry(
        event_type=AuditEventType.TIME_SPLIT,
        source_ids=[unit.id],
        result_ids=[u.id for u in new_units],
        input_weight=input_weight,
        output_weight=output_weight,
        description=f"Distributed to {len(new_units)} time slices",
        tolerance=config.conservation_tolerance,
        metadata={
            "slices": [{"label": s.label, "ratio": s.weight_ratio} for s in plan.slices],
        },
    )

    # 状態を更新
    updated_units = dict(state.units)
    updated_units[unit.id] = transformed_unit.model_dump()
    for new_unit in new_units:
        updated_units[new_unit.id] = new_unit.model_dump()

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=updated_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created,
        transformation_count=state.transformation_count + 1,
    )

    return new_units, new_state


def adjust_distance(
    unit: ResponsibilityUnit,
    new_distance: float,
    state: DispersionState,
    rationale: str = "",
    config: Optional[DispersionConfig] = None,
) -> tuple[ResponsibilityUnit, DispersionState]:
    """責任との主観的距離を調整する。

    重みは保存されるが、距離感が変化する。

    Args:
        unit: 対象の責任単位
        new_distance: 新しい距離
        state: 現在の状態
        rationale: 調整の理由
        config: 設定

    Returns:
        (新しいResponsibilityUnit, 更新されたDispersionState)
    """
    config = config or DispersionConfig()

    # Check if unit is already transformed (use state's version if available)
    unit_in_state = state.units.get(unit.id)
    if unit_in_state and unit_in_state.get("transformed", False):
        raise ValueError("Cannot adjust already transformed unit")
    if unit.transformed:
        raise ValueError("Cannot adjust already transformed unit")

    if new_distance < 0:
        raise ValueError("Distance must be non-negative")

    # 新しい責任単位を作成（重みは保存）
    new_unit = ResponsibilityUnit(
        weight=unit.weight,  # 保存
        meaning=unit.meaning,
        origin=f"distance_adjusted_from:{unit.id}",
        origin_type="distance_adjustment",
        distance=new_distance,
        time_slice=unit.time_slice,
        time_weight=unit.time_weight,
        parent_id=unit.id,
        generation=unit.generation + 1,
        transformed=False,
    )

    # 元のユニットを変換済みとしてマーク
    transformed_unit = ResponsibilityUnit(
        **{**unit.model_dump(), "transformed": True}
    )

    # 監査エントリを作成
    audit = _create_audit_entry(
        event_type=AuditEventType.DISTANCE_ADJUSTED,
        source_ids=[unit.id],
        result_ids=[new_unit.id],
        input_weight=unit.weight,
        output_weight=new_unit.weight,
        description=f"Distance adjusted: {unit.distance:.3f} -> {new_distance:.3f}",
        tolerance=config.conservation_tolerance,
        metadata={
            "old_distance": unit.distance,
            "new_distance": new_distance,
            "rationale": rationale,
        },
    )

    # 状態を更新
    updated_units = dict(state.units)
    updated_units[unit.id] = transformed_unit.model_dump()
    updated_units[new_unit.id] = new_unit.model_dump()

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=updated_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created,
        transformation_count=state.transformation_count + 1,
    )

    return new_unit, new_state


def merge_responsibilities(
    units: list[ResponsibilityUnit],
    merged_meaning: str,
    state: DispersionState,
    config: Optional[DispersionConfig] = None,
) -> tuple[ResponsibilityUnit, DispersionState]:
    """複数の責任を合算する。

    複数の責任単位を1つに統合する。
    合算後の重みは入力の総重みと一致する（保存則）。

    Args:
        units: 合算する責任単位のリスト
        merged_meaning: 合算後の意味表現
        state: 現在の状態
        config: 設定

    Returns:
        (新しいResponsibilityUnit, 更新されたDispersionState)

    Raises:
        ValueError: 空のリストまたは変換済みユニットを含む場合
    """
    config = config or DispersionConfig()

    if not units:
        raise ValueError("Cannot merge empty list of units")

    for u in units:
        # Check if unit is already transformed (use state's version if available)
        unit_in_state = state.units.get(u.id)
        if unit_in_state and unit_in_state.get("transformed", False):
            raise ValueError(f"Cannot merge already transformed unit: {u.id}")
        if u.transformed:
            raise ValueError(f"Cannot merge already transformed unit: {u.id}")

    # 総重みを計算
    input_weight = sum(u.weight for u in units)

    # 距離は重み付き平均
    if input_weight > 0:
        avg_distance = sum(u.weight * u.distance for u in units) / input_weight
    else:
        avg_distance = 1.0

    # 最大の世代を引き継ぐ
    max_generation = max(u.generation for u in units)

    # 新しい責任単位を作成
    new_unit = ResponsibilityUnit(
        weight=input_weight,
        meaning=merged_meaning,
        origin=f"merged_from:{','.join(u.id for u in units)}",
        origin_type="merge",
        distance=avg_distance,
        time_slice=units[0].time_slice,  # 最初のユニットから引き継ぐ
        time_weight=1.0,
        parent_id=None,  # 複数の親があるため特定しない
        generation=max_generation + 1,
        transformed=False,
    )

    # 元のユニットを変換済みとしてマーク
    updated_units = dict(state.units)
    for u in units:
        transformed = ResponsibilityUnit(**{**u.model_dump(), "transformed": True})
        updated_units[u.id] = transformed.model_dump()
    updated_units[new_unit.id] = new_unit.model_dump()

    # 監査エントリを作成
    audit = _create_audit_entry(
        event_type=AuditEventType.MERGED,
        source_ids=[u.id for u in units],
        result_ids=[new_unit.id],
        input_weight=input_weight,
        output_weight=new_unit.weight,
        description=f"Merged {len(units)} units into one",
        tolerance=config.conservation_tolerance,
        metadata={
            "merged_meaning": merged_meaning,
            "source_meanings": [u.meaning for u in units],
        },
    )

    new_audit_log = list(state.audit_log)
    new_audit_log.append(audit.model_dump())

    new_state = DispersionState(
        units=updated_units,
        audit_log=new_audit_log,
        total_weight_created=state.total_weight_created,
        transformation_count=state.transformation_count + 1,
    )

    return new_unit, new_state


# ── Query Functions ───────────────────────────────────────────────

def get_audit_trail(
    state: DispersionState,
    unit_id: Optional[str] = None,
) -> list[AuditEntry]:
    """監査ログを取得する。

    Args:
        state: 状態
        unit_id: 特定のユニットに関連するログのみ取得（オプション）

    Returns:
        AuditEntryのリスト
    """
    entries = [AuditEntry(**entry) for entry in state.audit_log]

    if unit_id is not None:
        entries = [
            e for e in entries
            if unit_id in e.source_ids or unit_id in e.result_ids
        ]

    return entries


def get_unit_by_id(
    state: DispersionState,
    unit_id: str,
) -> Optional[ResponsibilityUnit]:
    """IDで責任単位を取得する。

    Args:
        state: 状態
        unit_id: 責任単位のID

    Returns:
        ResponsibilityUnit（見つからなければNone）
    """
    unit_data = state.units.get(unit_id)
    if unit_data is None:
        return None
    return ResponsibilityUnit(**unit_data)


def get_active_units(state: DispersionState) -> list[ResponsibilityUnit]:
    """未変換（アクティブ）の責任単位を取得する。

    Args:
        state: 状態

    Returns:
        変換されていないResponsibilityUnitのリスト
    """
    return [
        ResponsibilityUnit(**data)
        for data in state.units.values()
        if not data.get("transformed", False)
    ]


def get_total_active_weight(state: DispersionState) -> float:
    """アクティブな責任の総重みを計算する。

    Args:
        state: 状態

    Returns:
        総重み
    """
    return sum(u.weight for u in get_active_units(state))


def get_lineage(
    state: DispersionState,
    unit_id: str,
) -> list[ResponsibilityUnit]:
    """責任単位の系譜（親→祖父母→...）を取得する。

    Args:
        state: 状態
        unit_id: 起点の責任単位ID

    Returns:
        系譜のリスト（起点が先頭、原初が末尾）
    """
    lineage = []
    current_id = unit_id

    while current_id is not None:
        unit = get_unit_by_id(state, current_id)
        if unit is None:
            break
        lineage.append(unit)
        current_id = unit.parent_id

    return lineage


def verify_state_conservation(state: DispersionState) -> dict:
    """状態全体の保存則を検証する。

    Args:
        state: 状態

    Returns:
        検証結果の辞書
    """
    # アクティブな重みの合計
    active_weight = get_total_active_weight(state)

    # 作成された総重み（変換で増減はないはず）
    created_weight = state.total_weight_created

    # 監査ログで保存則違反がないか
    violations = [
        entry for entry in state.audit_log
        if not entry.get("conservation_verified", True)
    ]

    return {
        "active_weight": active_weight,
        "created_weight": created_weight,
        "transformation_count": state.transformation_count,
        "audit_violations": len(violations),
        "is_consistent": len(violations) == 0,
    }


def create_dispersion_state() -> DispersionState:
    """初期状態を作成する。"""
    return DispersionState()


# ── Serialization ─────────────────────────────────────────────────

def to_dict(state: DispersionState) -> dict:
    """永続化用に辞書に変換する。"""
    return state.model_dump()


def from_dict(data: dict) -> DispersionState:
    """辞書から復元する。"""
    try:
        return DispersionState(**data)
    except Exception:
        return create_dispersion_state()


def get_dispersion_summary(state: DispersionState) -> str:
    """人間が読める要約を生成する。"""
    active = get_active_units(state)
    verification = verify_state_conservation(state)

    lines = [
        f"Dispersion State Summary:",
        f"  Active units: {len(active)}",
        f"  Active weight: {verification['active_weight']:.4f}",
        f"  Total created: {verification['created_weight']:.4f}",
        f"  Transformations: {verification['transformation_count']}",
        f"  Conservation: {'OK' if verification['is_consistent'] else 'VIOLATION'}",
    ]

    if active:
        lines.append("  Active meanings:")
        meaning_counts: dict[str, int] = {}
        for u in active:
            meaning_counts[u.meaning] = meaning_counts.get(u.meaning, 0) + 1
        for meaning, count in sorted(meaning_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    - {meaning}: {count}")

    return "\n".join(lines)
