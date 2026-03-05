"""
psyche/expectation_lifecycle_description.py - 予期の成立・消失の事後記述

予期形成が管理する予期の「生成・変化・消失」の事実を事後的に記録し、
蓄積し、参照可能にする。予期形成そのものの動作には介入しない。

設計原則 (design_expectation_lifecycle_description.md 準拠):
- 予期の正誤判定を行わない。予期の終了は事実として記録される
- 因果帰属を行わない。消失の「理由」を推測・帰属しない
- パターン抽出を行わない。寿命や終了形態の統計量を算出しない
- 予期形成パラメータへの書き込みを行わない（READ-ONLY）
- 予期の品質評価を行わない
- 行動・判断への直接接続を行わない

スナップショット比較による状態遷移検出:
- 生成: 前回スナップショットに存在せず現在に存在する予期
- 消失: 前回スナップショットに存在し現在に存在しない予期
- 修正: 両方に存在するが修正回数が増加した予期
- 強度変化: 前回と現在で強度段階が変わった予期
- 鮮度変化: 前回と現在で鮮度段階が変わった予期

遷移記録は多断面記述を持ちFIFO方式で蓄積される。
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

class TransitionType(Enum):
    """遷移の種類。"""
    GENERATION = "generation"       # 生成
    DISAPPEARANCE = "disappearance" # 消失
    REVISION = "revision"           # 修正
    STRENGTH_CHANGE = "strength_change"   # 強度変化
    FRESHNESS_CHANGE = "freshness_change" # 鮮度変化


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


def _strength_stage_label(strength: float) -> str:
    """強度数値から段階ラベルを返す（ExpectationFormation準拠）。"""
    if strength >= 0.7:
        return "strong"
    elif strength >= 0.4:
        return "moderate"
    elif strength >= 0.2:
        return "weak"
    elif strength >= 0.05:
        return "faint"
    else:
        return "undefined"


def _freshness_level_label(freshness: float) -> str:
    """鮮度数値から段階ラベルを返す（ExpectationFormation準拠）。"""
    if freshness >= 0.8:
        return "fresh"
    elif freshness >= 0.6:
        return "recent"
    elif freshness >= 0.4:
        return "aging"
    elif freshness >= 0.15:
        return "stale"
    else:
        return "faded"


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


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TransitionRecord:
    """遷移記録1件。多断面記述を持つ不変の記録構造。

    4断面:
    - 予期断面: 予期の識別子、内容記述、生成源種別、基盤種別
    - 遷移断面: 遷移の種類、遷移時点での強度段階・鮮度段階
    - 時間断面: 処理サイクル番号、タイムスタンプ
    - 競合断面: 遷移時点での競合相手識別子群

    一度構成された記録は変更されない（追記のみ）。
    全記録は等価に蓄積される。
    """
    record_id: str = field(default_factory=_gen_id)

    # 予期断面
    expectation_id: str = ""
    expectation_description: str = ""  # 文字数上限付き
    source_type: str = ""   # ExpectationSourceType.value
    basis_type: str = ""    # ExpectationBasis.value

    # 遷移断面
    transition_type: str = TransitionType.GENERATION.value
    strength_stage: str = ""  # 遷移時点での強度段階ラベル
    freshness_stage: str = "" # 遷移時点での鮮度段階ラベル

    # 時間断面
    cycle_number: int = 0
    timestamp: float = field(default_factory=time.time)

    # 競合断面
    competing_ids: list[str] = field(default_factory=list)

    # 記録自体の鮮度（蓄積後に均一減衰する）
    record_freshness: float = 1.0
    record_freshness_stage: str = RecordFreshness.FRESH.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "expectation_id": self.expectation_id,
            "expectation_description": self.expectation_description,
            "source_type": self.source_type,
            "basis_type": self.basis_type,
            "transition_type": self.transition_type,
            "strength_stage": self.strength_stage,
            "freshness_stage": self.freshness_stage,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp,
            "competing_ids": list(self.competing_ids),
            "record_freshness": self.record_freshness,
            "record_freshness_stage": self.record_freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            expectation_id=data.get("expectation_id", ""),
            expectation_description=data.get("expectation_description", ""),
            source_type=data.get("source_type", ""),
            basis_type=data.get("basis_type", ""),
            transition_type=data.get("transition_type", TransitionType.GENERATION.value),
            strength_stage=data.get("strength_stage", ""),
            freshness_stage=data.get("freshness_stage", ""),
            cycle_number=data.get("cycle_number", 0),
            timestamp=data.get("timestamp", time.time()),
            competing_ids=list(data.get("competing_ids", [])),
            record_freshness=data.get("record_freshness", 1.0),
            record_freshness_stage=data.get("record_freshness_stage", RecordFreshness.FRESH.value),
        )


@dataclass
class LifecycleView:
    """1つの予期のライフサイクル全体像。

    同一の予期識別子に対する遷移記録群から構成される。
    解釈・評価・要約を含まない。遷移記録の単純な集約。
    """
    expectation_id: str = ""
    generation_record: Optional[TransitionRecord] = None
    intermediate_records: list[TransitionRecord] = field(default_factory=list)
    disappearance_record: Optional[TransitionRecord] = None
    is_completed: bool = False  # 消失済みか

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "generation_record": self.generation_record.to_dict() if self.generation_record else None,
            "intermediate_records": [r.to_dict() for r in self.intermediate_records],
            "disappearance_record": self.disappearance_record.to_dict() if self.disappearance_record else None,
            "is_completed": self.is_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifecycleView":
        gen_rec = None
        if data.get("generation_record"):
            gen_rec = TransitionRecord.from_dict(data["generation_record"])
        dis_rec = None
        if data.get("disappearance_record"):
            dis_rec = TransitionRecord.from_dict(data["disappearance_record"])
        return cls(
            expectation_id=data.get("expectation_id", ""),
            generation_record=gen_rec,
            intermediate_records=[
                TransitionRecord.from_dict(r) for r in data.get("intermediate_records", [])
            ],
            disappearance_record=dis_rec,
            is_completed=data.get("is_completed", False),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。"""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    dominant_type: str = ""
    type_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "dominant_type": self.dominant_type,
            "type_diversity": self.type_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            dominant_type=data.get("dominant_type", ""),
            type_diversity=data.get("type_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class SnapshotEntry:
    """前回スナップショットの予期1件分。比較のためにのみ保持。"""
    expectation_id: str = ""
    source_type: str = ""
    basis_type: str = ""
    description: str = ""
    strength: float = 0.0
    freshness: float = 0.0
    revision_count: int = 0
    competing_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "source_type": self.source_type,
            "basis_type": self.basis_type,
            "description": self.description,
            "strength": self.strength,
            "freshness": self.freshness,
            "revision_count": self.revision_count,
            "competing_ids": list(self.competing_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotEntry":
        return cls(
            expectation_id=data.get("expectation_id", ""),
            source_type=data.get("source_type", ""),
            basis_type=data.get("basis_type", ""),
            description=data.get("description", ""),
            strength=data.get("strength", 0.0),
            freshness=data.get("freshness", 0.0),
            revision_count=data.get("revision_count", 0),
            competing_ids=list(data.get("competing_ids", [])),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class ExpectationLifecycleState:
    """予期ライフサイクル記述の内部状態。永続化対象。"""
    # 遷移記録リスト（FIFO、時系列順）
    transition_records: list[TransitionRecord] = field(default_factory=list)

    # 前回スナップショット（比較のためにのみ保持、外部に公開しない）
    previous_snapshot: dict[str, SnapshotEntry] = field(default_factory=dict)

    # 処理サイクルカウンタ
    cycle_count: int = 0

    # 収束監視記録
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # 安全弁フラグ群
    accumulation_limit_reached: bool = False
    convergence_flag: bool = False  # 特定遷移種類が極端に支配的

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_records": [r.to_dict() for r in self.transition_records],
            "previous_snapshot": {
                k: v.to_dict() for k, v in self.previous_snapshot.items()
            },
            "cycle_count": self.cycle_count,
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "accumulation_limit_reached": self.accumulation_limit_reached,
            "convergence_flag": self.convergence_flag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectationLifecycleState":
        return cls(
            transition_records=[
                TransitionRecord.from_dict(r) for r in data.get("transition_records", [])
            ],
            previous_snapshot={
                k: SnapshotEntry.from_dict(v)
                for k, v in data.get("previous_snapshot", {}).items()
            },
            cycle_count=data.get("cycle_count", 0),
            convergence_records=[
                ConvergenceRecord.from_dict(c) for c in data.get("convergence_records", [])
            ],
            accumulation_limit_reached=data.get("accumulation_limit_reached", False),
            convergence_flag=data.get("convergence_flag", False),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExpectationLifecycleConfig:
    """設定。"""
    # 遷移記録の蓄積上限（FIFO）
    max_records: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))

    # 記録自体の鮮度減衰速度（サイクルあたり、均一減衰）
    record_freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))

    # enrichmentに供給する遷移記録の件数上限
    enrichment_count: int = 10

    # 収束監視記録の上限
    max_convergence_records: int = 30

    # 収束警告の閾値（特定遷移種類の比率）
    convergence_threshold: float = 0.7

    # 予期内容記述の文字数上限
    description_max_length: int = 120

    # 強度段階の変化検出閾値（同一段階内の微小変動を無視）
    strength_change_threshold: float = 0.0  # 段階ラベルが変わったら検出

    # 鮮度段階の変化検出閾値
    freshness_change_threshold: float = 0.0  # 段階ラベルが変わったら検出


# =============================================================================
# Processor
# =============================================================================

class ExpectationLifecycleDescriptionProcessor:
    """予期のライフサイクル記述プロセッサ。

    予期形成が保持する予期集合のスナップショットを比較し、
    状態遷移を検出・記録・蓄積する。

    予期形成への書き込み経路を持たない（READ-ONLY）。
    遷移記録はポリシー候補生成・バイアス適用・スコアリングに入力されない。
    遷移記録から目標・動機・行動指示を生成する経路を持たない。
    感情チャンネル・責任重量・価値方向性への書き込み経路を持たない。
    """

    def __init__(self, config: Optional[ExpectationLifecycleConfig] = None):
        self._config = config or ExpectationLifecycleConfig()
        self._state = ExpectationLifecycleState()

    @property
    def state(self) -> ExpectationLifecycleState:
        return self._state

    @state.setter
    def state(self, value: ExpectationLifecycleState) -> None:
        self._state = value

    # ─── Main processing entry point ──────────────────────────

    def process(self, expectation_store: Optional[Any] = None) -> int:
        """スナップショット比較→遷移検出→記録蓄積→鮮度減衰→収束監視。

        予期形成の予期集合を読み取り、前回スナップショットと比較して
        状態遷移を検出する。検出された遷移は記録として蓄積される。

        Args:
            expectation_store: 予期形成が返すExpectationStore（READ-ONLY参照）

        Returns:
            今回新規検出された遷移記録の数
        """
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # 現在のスナップショットを構築
        current_snapshot = self._build_current_snapshot(expectation_store)

        # スナップショット比較による状態遷移検出
        new_records = self._detect_transitions(current_snapshot, now)

        # 遷移記録の蓄積
        self._accumulate_records(new_records)

        # 記録自体の鮮度減衰（均一減衰）
        self._apply_record_freshness_decay()

        # 収束監視
        self._monitor_convergence(now)

        # 前回スナップショットを更新
        self._state.previous_snapshot = current_snapshot

        logger.debug(
            "Expectation lifecycle: cycle=%d, new_transitions=%d, total_records=%d",
            self._state.cycle_count,
            len(new_records),
            len(self._state.transition_records),
        )

        return len(new_records)

    # ─── Snapshot building ─────────────────────────────────────

    def _build_current_snapshot(
        self, expectation_store: Optional[Any],
    ) -> dict[str, SnapshotEntry]:
        """予期形成のExpectationStoreから現在スナップショットを構築する。

        READ-ONLYで参照し、予期形成の内部状態を直接操作しない。
        """
        snapshot: dict[str, SnapshotEntry] = {}
        if expectation_store is None:
            return snapshot

        expectations = getattr(expectation_store, "expectations", ())
        if not expectations:
            return snapshot

        for exp in expectations:
            exp_id = getattr(exp, "expectation_id", "")
            if not exp_id:
                continue

            source_type_raw = getattr(exp, "source_type", None)
            source_val = source_type_raw.value if hasattr(source_type_raw, "value") else str(source_type_raw or "")
            basis_raw = getattr(exp, "basis", None)
            basis_val = basis_raw.value if hasattr(basis_raw, "value") else str(basis_raw or "")
            description = getattr(exp, "description", "")
            strength = getattr(exp, "strength", 0.0)
            if not isinstance(strength, (int, float)):
                strength = 0.0
            freshness = getattr(exp, "freshness", 0.0)
            if not isinstance(freshness, (int, float)):
                freshness = 0.0
            revision_count = getattr(exp, "revision_count", 0)
            competing_ids_raw = getattr(exp, "competing_ids", ())
            competing_ids = list(competing_ids_raw) if competing_ids_raw else []

            snapshot[exp_id] = SnapshotEntry(
                expectation_id=exp_id,
                source_type=source_val,
                basis_type=basis_val,
                description=description,
                strength=float(strength),
                freshness=float(freshness),
                revision_count=int(revision_count),
                competing_ids=competing_ids,
            )

        return snapshot

    # ─── Transition detection ──────────────────────────────────

    def _detect_transitions(
        self,
        current_snapshot: dict[str, SnapshotEntry],
        now: float,
    ) -> list[TransitionRecord]:
        """前回スナップショットと現在スナップショットを比較し遷移を検出する。"""
        cfg = self._config
        cycle = self._state.cycle_count
        prev = self._state.previous_snapshot
        new_records: list[TransitionRecord] = []

        current_ids = set(current_snapshot.keys())
        prev_ids = set(prev.keys())

        # 生成: 前回に存在せず現在に存在
        for exp_id in current_ids - prev_ids:
            entry = current_snapshot[exp_id]
            record = self._create_record(
                entry=entry,
                transition_type=TransitionType.GENERATION,
                cycle=cycle,
                now=now,
            )
            new_records.append(record)

        # 消失: 前回に存在し現在に存在しない
        for exp_id in prev_ids - current_ids:
            entry = prev[exp_id]
            record = self._create_record(
                entry=entry,
                transition_type=TransitionType.DISAPPEARANCE,
                cycle=cycle,
                now=now,
            )
            new_records.append(record)

        # 両方に存在する予期の変化を検出
        for exp_id in current_ids & prev_ids:
            current_entry = current_snapshot[exp_id]
            prev_entry = prev[exp_id]

            # 修正: 修正回数が増加
            if current_entry.revision_count > prev_entry.revision_count:
                record = self._create_record(
                    entry=current_entry,
                    transition_type=TransitionType.REVISION,
                    cycle=cycle,
                    now=now,
                )
                new_records.append(record)

            # 強度変化: 強度段階が変わった
            current_strength_label = _strength_stage_label(current_entry.strength)
            prev_strength_label = _strength_stage_label(prev_entry.strength)
            if current_strength_label != prev_strength_label:
                record = self._create_record(
                    entry=current_entry,
                    transition_type=TransitionType.STRENGTH_CHANGE,
                    cycle=cycle,
                    now=now,
                )
                new_records.append(record)

            # 鮮度変化: 鮮度段階が変わった
            current_freshness_label = _freshness_level_label(current_entry.freshness)
            prev_freshness_label = _freshness_level_label(prev_entry.freshness)
            if current_freshness_label != prev_freshness_label:
                record = self._create_record(
                    entry=current_entry,
                    transition_type=TransitionType.FRESHNESS_CHANGE,
                    cycle=cycle,
                    now=now,
                )
                new_records.append(record)

        return new_records

    def _create_record(
        self,
        entry: SnapshotEntry,
        transition_type: TransitionType,
        cycle: int,
        now: float,
    ) -> TransitionRecord:
        """遷移記録を構成する。"""
        cfg = self._config
        # 内容記述の文字数上限
        desc = entry.description
        if len(desc) > cfg.description_max_length:
            desc = desc[:cfg.description_max_length]

        return TransitionRecord(
            expectation_id=entry.expectation_id,
            expectation_description=desc,
            source_type=entry.source_type,
            basis_type=entry.basis_type,
            transition_type=transition_type.value,
            strength_stage=_strength_stage_label(entry.strength),
            freshness_stage=_freshness_level_label(entry.freshness),
            cycle_number=cycle,
            timestamp=now,
            competing_ids=list(entry.competing_ids),
            record_freshness=1.0,
            record_freshness_stage=RecordFreshness.FRESH.value,
        )

    # ─── Accumulation ──────────────────────────────────────────

    def _accumulate_records(self, new_records: list[TransitionRecord]) -> None:
        """遷移記録をFIFO方式で蓄積する。

        蓄積上限に達すると最も古い記録から自然に脱落する。
        「重要な遷移」を優先保持する選別は行わない。
        """
        cfg = self._config
        self._state.transition_records.extend(new_records)

        # FIFO上限制御
        if len(self._state.transition_records) > cfg.max_records:
            overflow = len(self._state.transition_records) - cfg.max_records
            self._state.transition_records = self._state.transition_records[overflow:]
            self._state.accumulation_limit_reached = True
        else:
            self._state.accumulation_limit_reached = False

    # ─── Record freshness decay ────────────────────────────────

    def _apply_record_freshness_decay(self) -> None:
        """記録自体の鮮度を均一に減衰する。

        遷移種類や予期の属性に依存しない均一な速度で減衰する。
        減衰は記録の削除を直接引き起こさない（削除はFIFO上限のみ）。
        """
        cfg = self._config
        for record in self._state.transition_records:
            record.record_freshness = _clamp(
                record.record_freshness - cfg.record_freshness_decay_rate
            )
            record.record_freshness_stage = _freshness_stage(
                record.record_freshness
            ).value

    # ─── Convergence monitoring ────────────────────────────────

    def _monitor_convergence(self, now: float) -> None:
        """蓄積された遷移記録の遷移種類分布を監視する。

        特定の遷移種類が極端に支配的になった場合、その事実を
        フラグとして記録する。ただし、収束の検出は記録の内容や
        蓄積を変更しない。フラグはenrichment出力に含まれず、
        内部の監視記録としてのみ保持される。
        """
        cfg = self._config
        records = self._state.transition_records
        if not records:
            self._state.convergence_flag = False
            return

        # 遷移種類の分布
        type_counts: dict[str, int] = {}
        for rec in records:
            t = rec.transition_type
            type_counts[t] = type_counts.get(t, 0) + 1

        total = sum(type_counts.values())
        if total == 0:
            self._state.convergence_flag = False
            return

        # 支配的な遷移種類
        dominant_type = max(type_counts, key=type_counts.get)  # type: ignore
        dominant_ratio = type_counts[dominant_type] / total

        # 多様性
        unique_types = len(type_counts)
        type_diversity = unique_types / max(len(TransitionType), 1)

        # 収束スコア
        convergence_score = _clamp(
            dominant_ratio * 0.7 + (1.0 - type_diversity) * 0.3
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
            dominant_type=dominant_type,
            type_diversity=type_diversity,
            cycle=self._state.cycle_count,
            timestamp=now,
        )
        self._state.convergence_records.append(conv_record)
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

    # ─── READ-ONLY accessors ───────────────────────────────────

    def get_recent_records(self, count: Optional[int] = None) -> list[TransitionRecord]:
        """直近の遷移記録を等価列挙で返す（READ-ONLY）。

        件数上限付きで提供。遷移種類の分布や寿命の統計を含めない。
        参照行為によって記録の内容や順序が変化しない。
        参照頻度が蓄積の優先度に影響しない。
        """
        n = count if count is not None else self._config.enrichment_count
        return list(self._state.transition_records[-n:])

    def get_lifecycle_view(self, expectation_id: str) -> Optional[LifecycleView]:
        """指定された予期のライフサイクル全体像を返す（READ-ONLY）。

        同一の予期識別子に対する遷移記録群から構成される。
        解釈・評価・要約を含まない。遷移記録の単純な集約。
        """
        records = [
            r for r in self._state.transition_records
            if r.expectation_id == expectation_id
        ]
        if not records:
            return None

        view = LifecycleView(expectation_id=expectation_id)
        for rec in records:
            if rec.transition_type == TransitionType.GENERATION.value:
                view.generation_record = rec
            elif rec.transition_type == TransitionType.DISAPPEARANCE.value:
                view.disappearance_record = rec
                view.is_completed = True
            else:
                view.intermediate_records.append(rec)

        return view

    def get_lifecycle_counts(self) -> dict[str, int]:
        """現在進行中のライフサイクル数と消失済みライフサイクル数を返す。

        数量的記述のみ。
        """
        # 生成記録のある予期ID集合
        generated_ids: set[str] = set()
        disappeared_ids: set[str] = set()
        for rec in self._state.transition_records:
            if rec.transition_type == TransitionType.GENERATION.value:
                generated_ids.add(rec.expectation_id)
            elif rec.transition_type == TransitionType.DISAPPEARANCE.value:
                disappeared_ids.add(rec.expectation_id)

        # 進行中 = 生成済み - 消失済み
        active_ids = generated_ids - disappeared_ids
        return {
            "active_lifecycles": len(active_ids),
            "completed_lifecycles": len(disappeared_ids),
        }

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        counts = self.get_lifecycle_counts()
        return {
            "total_records": len(self._state.transition_records),
            "cycle_count": self._state.cycle_count,
            "active_lifecycles": counts["active_lifecycles"],
            "completed_lifecycles": counts["completed_lifecycles"],
            "accumulation_limit_reached": self._state.accumulation_limit_reached,
            "convergence_flag": self._state.convergence_flag,
        }

    # ─── Enrichment ────────────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        直近の遷移記録を件数上限付きで等価列挙する。
        遷移種類の分布や寿命の統計を含めない。
        「予期Xが生成された」「予期Yが消失した」「予期Zの強度が変化した」の
        事実記述のみ。

        enrichment出力に収束フラグは含めない（設計書準拠）。
        """
        cfg = self._config
        recent = self.get_recent_records(cfg.enrichment_count)
        counts = self.get_lifecycle_counts()

        entries: list[dict[str, Any]] = []
        for rec in recent:
            # 事実記述のみ
            desc_preview = rec.expectation_description[:80] if rec.expectation_description else ""
            if len(rec.expectation_description) > 80:
                desc_preview += "..."

            entries.append({
                "transition_type": rec.transition_type,
                "expectation_id": rec.expectation_id,
                "expectation_description": desc_preview,
                "source_type": rec.source_type,
                "strength_stage": rec.strength_stage,
                "freshness_stage": rec.freshness_stage,
                "record_freshness_stage": rec.record_freshness_stage,
                "cycle": rec.cycle_number,
            })

        summary_text = get_lifecycle_summary(self._state)

        return {
            "cycle_count": self._state.cycle_count,
            "total_records": len(self._state.transition_records),
            "active_lifecycles": counts["active_lifecycles"],
            "completed_lifecycles": counts["completed_lifecycles"],
            "entries": entries,
            "summary_text": summary_text,
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_lifecycle_summary(state: ExpectationLifecycleState) -> str:
    """予期ライフサイクル記述状態の要約（enrichment用）。

    事実記述のみ。評価判定・行動指示を含まない。
    パターン抽出を行わない。統計量を算出しない。
    """
    if state.cycle_count == 0 and not state.transition_records:
        return "予期ライフサイクル記述: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    total = len(state.transition_records)
    parts.append(f"遷移記録={total}")

    # ライフサイクル数（生成/消失から算出）
    generated_ids: set[str] = set()
    disappeared_ids: set[str] = set()
    for rec in state.transition_records:
        if rec.transition_type == TransitionType.GENERATION.value:
            generated_ids.add(rec.expectation_id)
        elif rec.transition_type == TransitionType.DISAPPEARANCE.value:
            disappeared_ids.add(rec.expectation_id)

    active = len(generated_ids - disappeared_ids)
    completed = len(disappeared_ids)
    if active > 0:
        parts.append(f"進行中={active}")
    if completed > 0:
        parts.append(f"消失済={completed}")

    # 直近の遷移種類（事実記述のみ、最新3件のラベル）
    recent = state.transition_records[-3:]
    if recent:
        recent_labels = [r.transition_type for r in recent]
        parts.append(f"直近=[{','.join(recent_labels)}]")

    return " ".join(parts) if parts else "予期ライフサイクル記述: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_expectation_lifecycle_processor(
    config: Optional[ExpectationLifecycleConfig] = None,
) -> ExpectationLifecycleDescriptionProcessor:
    """ExpectationLifecycleDescriptionProcessor のファクトリ関数。"""
    return ExpectationLifecycleDescriptionProcessor(config=config)
