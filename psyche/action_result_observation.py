"""
psyche/action_result_observation.py - 行動-結果の観測と蓄積

行動（ポリシー選択）後に生じた外部反応や状態変化を、その行動と対にして
保持する経路を提供する。行動と結果を対として構成し、多断面で記述し、
文脈とともに蓄積する。

設計原則 (design_action_result_observation.md 準拠):
- 目的は行動後に生じた変化を観測可能な対として蓄積すること
- 「良い結果を増やす」「悪い結果を減らす」方向への最適化ではない
- 結果は正誤・成否の二値で評価しない。複数の断面で記述する
- 行動と結果の関係は「因果」として断定しない。時系列的隣接の記録にとどめる
- 結果の記録から特定の行動傾向を直接強化または抑制しない
- 結果評価を判断確定へ直接接続しない
- 行動-結果対から「最適な行動パターン」を導出しない
- 蓄積された対を一方向的な改善の根拠にしない
- 出力は参照情報としてのみ流し、判断・評価・行動決定を直接起動しない

6段パイプライン:
1. 対構成 (pair composition)
2. 多断面評価記述 (multi-section description)
3. 文脈帰属付与 (context attribution)
4. 整列蓄積 (alignment and accumulation)
5. 減衰忘却 (decay and forgetting)
6. 受渡準備 (handoff preparation)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ObservationSection(Enum):
    """入力参照の断面種別（8値）。"""
    RECENT_ACTION = "recent_action"              # 直近行動断面
    EXTERNAL_REACTION = "external_reaction"      # 外部反応断面
    INTERNAL_STATE_CHANGE = "internal_state_change"  # 内部状態変化断面
    EMOTION_TRANSITION = "emotion_transition"    # 感情推移断面
    CONTEXT = "context"                          # 文脈断面
    TIME_ELAPSED = "time_elapsed"                # 時間経過断面
    OTHER_OBSERVATION = "other_observation"       # 他者観測断面
    MEMORY_REFERENCE = "memory_reference"        # 記憶参照断面


class FreshnessStage(Enum):
    """鮮度段階（段階的希薄化、memory_forgetting_fixation パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class PairStatus(Enum):
    """行動-結果対の状態。"""
    BUFFERED = "buffered"       # 構成バッファ内（結果未結合）
    PENDING = "pending"         # 結果未観測保留
    COMPOSED = "composed"       # 対構成完了
    ACTIVE = "active"           # 活性状態
    DECAYING = "decaying"       # 減衰中
    INVISIBLE = "invisible"     # 不可視化


class ConvergenceLevel(Enum):
    """収束監視レベル。"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _stage_from_freshness(freshness: float) -> FreshnessStage:
    """鮮度値から段階を返す（memory_forgetting_fixation パターン準拠）。"""
    if freshness >= 0.8:
        return FreshnessStage.ACTIVE
    elif freshness >= 0.6:
        return FreshnessStage.WEAKENING
    elif freshness >= 0.4:
        return FreshnessStage.FADING
    elif freshness >= 0.2:
        return FreshnessStage.NEAR_INVISIBLE
    else:
        return FreshnessStage.INVISIBLE


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
class SectionDescription:
    """1つの断面における結果記述。断面間に優先順位なし。"""
    section: str = ""  # ObservationSection.value
    description: str = ""
    value: float = 0.0
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionDescription":
        return cls(
            section=data.get("section", ""),
            description=data.get("description", ""),
            value=data.get("value", 0.0),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ActionDescription:
    """行動記述（選択されたポリシー情報、選択時の文脈要約）。"""
    policy_label: str = ""
    policy_axis: str = ""
    selection_context: str = ""
    tick_at_action: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_label": self.policy_label,
            "policy_axis": self.policy_axis,
            "selection_context": self.selection_context,
            "tick_at_action": self.tick_at_action,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionDescription":
        return cls(
            policy_label=data.get("policy_label", ""),
            policy_axis=data.get("policy_axis", ""),
            selection_context=data.get("selection_context", ""),
            tick_at_action=data.get("tick_at_action", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ResultDescription:
    """結果記述（多断面の独立記述群）。"""
    sections: list[SectionDescription] = field(default_factory=list)
    tick_at_result: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "tick_at_result": self.tick_at_result,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultDescription":
        return cls(
            sections=[
                SectionDescription.from_dict(s)
                for s in data.get("sections", [])
            ],
            tick_at_result=data.get("tick_at_result", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ContextAttribution:
    """文脈帰属情報（対ごとの状況記録）。因果断定しない。"""
    context_summary: str = ""
    dialogue_state: str = ""
    environment_tags: list[str] = field(default_factory=list)
    tick_at_context: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_summary": self.context_summary,
            "dialogue_state": self.dialogue_state,
            "environment_tags": list(self.environment_tags),
            "tick_at_context": self.tick_at_context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextAttribution":
        return cls(
            context_summary=data.get("context_summary", ""),
            dialogue_state=data.get("dialogue_state", ""),
            environment_tags=list(data.get("environment_tags", [])),
            tick_at_context=data.get("tick_at_context", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ActionResultPair:
    """行動-結果対。時系列的隣接の記録。因果断定しない。"""
    pair_id: str = field(default_factory=_gen_id)
    action: ActionDescription = field(default_factory=ActionDescription)
    result: ResultDescription = field(default_factory=ResultDescription)
    context: ContextAttribution = field(default_factory=ContextAttribution)
    status: str = PairStatus.BUFFERED.value
    freshness: float = 1.0  # 1.0=active, 0.0=invisible
    freshness_stage: str = FreshnessStage.ACTIVE.value
    reference_count: int = 0
    reactivation_count: int = 0
    creation_tick: int = 0
    creation_time: float = field(default_factory=time.time)
    last_reference_time: float = 0.0
    pattern_key: str = ""  # 行動パターン分類キー

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "context": self.context.to_dict(),
            "status": self.status,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
            "reference_count": self.reference_count,
            "reactivation_count": self.reactivation_count,
            "creation_tick": self.creation_tick,
            "creation_time": self.creation_time,
            "last_reference_time": self.last_reference_time,
            "pattern_key": self.pattern_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResultPair":
        return cls(
            pair_id=data.get("pair_id", _gen_id()),
            action=ActionDescription.from_dict(data.get("action", {})),
            result=ResultDescription.from_dict(data.get("result", {})),
            context=ContextAttribution.from_dict(data.get("context", {})),
            status=data.get("status", PairStatus.BUFFERED.value),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
            reference_count=data.get("reference_count", 0),
            reactivation_count=data.get("reactivation_count", 0),
            creation_tick=data.get("creation_tick", 0),
            creation_time=data.get("creation_time", time.time()),
            last_reference_time=data.get("last_reference_time", 0.0),
            pattern_key=data.get("pattern_key", ""),
        )


@dataclass
class SectionWeightRecord:
    """断面重み変動履歴エントリ。"""
    section: str = ""
    weight: float = 0.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "weight": self.weight,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionWeightRecord":
        return cls(
            section=data.get("section", ""),
            weight=data.get("weight", 0.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。"""
    record_id: str = field(default_factory=_gen_id)
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    dominant_pattern: str = ""
    pattern_diversity: float = 1.0
    section_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "dominant_pattern": self.dominant_pattern,
            "pattern_diversity": self.pattern_diversity,
            "section_diversity": self.section_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            dominant_pattern=data.get("dominant_pattern", ""),
            pattern_diversity=data.get("pattern_diversity", 1.0),
            section_diversity=data.get("section_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Inputs (8 cross-sections)
# =============================================================================

@dataclass
class ActionResultInputs:
    """8断面の入力データ。"""
    # 1. 直近行動断面
    selected_policy_label: str = ""
    selected_policy_axis: str = ""
    selection_context_summary: str = ""
    action_tick: int = 0

    # 2. 外部反応断面
    external_response_change: float = 0.0
    external_response_description: str = ""

    # 3. 内部状態変化断面
    internal_state_delta: float = 0.0
    motivation_delta: float = 0.0
    direction_delta: float = 0.0

    # 4. 感情推移断面
    emotion_before: dict[str, float] = field(default_factory=dict)
    emotion_after: dict[str, float] = field(default_factory=dict)

    # 5. 文脈断面
    context_summary: str = ""
    dialogue_state: str = ""
    environment_tags: list[str] = field(default_factory=list)

    # 6. 時間経過断面
    ticks_since_action: int = 0
    elapsed_seconds: float = 0.0

    # 7. 他者観測断面
    other_reaction_change: float = 0.0
    other_reaction_description: str = ""

    # 8. 記憶参照断面
    referenced_memory_ids: list[str] = field(default_factory=list)
    referenced_memory_count: int = 0

    # 9. テキスト断面（自己行動知覚から供給される出力テキスト情報）
    # 既存の8断面と同列であり、優先順位を持たない
    output_text: str = ""

    # メタデータ
    current_tick: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class ActionResultObservationState:
    """行動-結果観測システムの内部状態。"""
    # 行動-結果対の集合
    pairs: list[ActionResultPair] = field(default_factory=list)

    # 構成バッファ（行動記録後、結果観測待ちの未完成対）
    composition_buffer: list[ActionResultPair] = field(default_factory=list)

    # 時系列索引（pair_id の時間順序）
    time_index: list[str] = field(default_factory=list)

    # 断面記述履歴
    section_description_history: list[dict[str, Any]] = field(default_factory=list)

    # 断面重み変動履歴
    section_weight_history: list[SectionWeightRecord] = field(default_factory=list)

    # 減衰履歴
    decay_history: list[dict[str, Any]] = field(default_factory=list)

    # 復帰候補履歴
    recovery_candidates: list[str] = field(default_factory=list)

    # 収束監視状態
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # シグナル供給強度（自己強化ループ防止用の自動減衰対象）
    signal_supply_strength: float = 1.0

    # カウンタ
    cycle_count: int = 0
    total_pairs_composed: int = 0
    total_pairs_decayed: int = 0
    total_pairs_recovered: int = 0
    total_pairs_pending: int = 0

    # 安全弁フラグ
    pattern_convergence_warning: bool = False
    section_bias_warning: bool = False
    signal_attenuation_active: bool = False
    buffer_overflow_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "composition_buffer": [b.to_dict() for b in self.composition_buffer],
            "time_index": list(self.time_index),
            "section_description_history": list(self.section_description_history),
            "section_weight_history": [w.to_dict() for w in self.section_weight_history],
            "decay_history": list(self.decay_history),
            "recovery_candidates": list(self.recovery_candidates),
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "signal_supply_strength": self.signal_supply_strength,
            "cycle_count": self.cycle_count,
            "total_pairs_composed": self.total_pairs_composed,
            "total_pairs_decayed": self.total_pairs_decayed,
            "total_pairs_recovered": self.total_pairs_recovered,
            "total_pairs_pending": self.total_pairs_pending,
            "pattern_convergence_warning": self.pattern_convergence_warning,
            "section_bias_warning": self.section_bias_warning,
            "signal_attenuation_active": self.signal_attenuation_active,
            "buffer_overflow_warning": self.buffer_overflow_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResultObservationState":
        return cls(
            pairs=[
                ActionResultPair.from_dict(p)
                for p in data.get("pairs", [])
            ],
            composition_buffer=[
                ActionResultPair.from_dict(b)
                for b in data.get("composition_buffer", [])
            ],
            time_index=list(data.get("time_index", [])),
            section_description_history=list(
                data.get("section_description_history", [])
            ),
            section_weight_history=[
                SectionWeightRecord.from_dict(w)
                for w in data.get("section_weight_history", [])
            ],
            decay_history=list(data.get("decay_history", [])),
            recovery_candidates=list(data.get("recovery_candidates", [])),
            convergence_records=[
                ConvergenceRecord.from_dict(c)
                for c in data.get("convergence_records", [])
            ],
            signal_supply_strength=data.get("signal_supply_strength", 1.0),
            cycle_count=data.get("cycle_count", 0),
            total_pairs_composed=data.get("total_pairs_composed", 0),
            total_pairs_decayed=data.get("total_pairs_decayed", 0),
            total_pairs_recovered=data.get("total_pairs_recovered", 0),
            total_pairs_pending=data.get("total_pairs_pending", 0),
            pattern_convergence_warning=data.get(
                "pattern_convergence_warning", False
            ),
            section_bias_warning=data.get("section_bias_warning", False),
            signal_attenuation_active=data.get(
                "signal_attenuation_active", False
            ),
            buffer_overflow_warning=data.get("buffer_overflow_warning", False),
        )


# =============================================================================
# Result
# =============================================================================

@dataclass
class ActionResultObservationResult:
    """処理結果（参照情報形式のみ）。判断・評価・行動決定を含まない。"""
    # 新規構成された対
    newly_composed_pairs: list[ActionResultPair] = field(default_factory=list)
    # 全活性対の概要
    active_pair_count: int = 0
    decaying_pair_count: int = 0
    invisible_pair_count: int = 0
    buffered_pair_count: int = 0
    pending_pair_count: int = 0

    # 断面分布
    section_distribution: dict[str, float] = field(default_factory=dict)

    # パターン分布
    pattern_distribution: dict[str, int] = field(default_factory=dict)

    # 鮮度分布
    freshness_distribution: dict[str, int] = field(default_factory=dict)

    # 収束監視情報
    convergence_level: str = ConvergenceLevel.NONE.value
    convergence_score: float = 0.0

    # 安全弁情報
    pattern_convergence_warning: bool = False
    section_bias_warning: bool = False
    signal_attenuation_active: bool = False
    buffer_overflow_warning: bool = False
    diversity_restored: bool = False
    decay_slowed: bool = False

    # シグナル供給強度
    signal_supply_strength: float = 1.0

    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "newly_composed_pairs": [
                p.to_dict() for p in self.newly_composed_pairs
            ],
            "active_pair_count": self.active_pair_count,
            "decaying_pair_count": self.decaying_pair_count,
            "invisible_pair_count": self.invisible_pair_count,
            "buffered_pair_count": self.buffered_pair_count,
            "pending_pair_count": self.pending_pair_count,
            "section_distribution": dict(self.section_distribution),
            "pattern_distribution": dict(self.pattern_distribution),
            "freshness_distribution": dict(self.freshness_distribution),
            "convergence_level": self.convergence_level,
            "convergence_score": self.convergence_score,
            "pattern_convergence_warning": self.pattern_convergence_warning,
            "section_bias_warning": self.section_bias_warning,
            "signal_attenuation_active": self.signal_attenuation_active,
            "buffer_overflow_warning": self.buffer_overflow_warning,
            "diversity_restored": self.diversity_restored,
            "decay_slowed": self.decay_slowed,
            "signal_supply_strength": self.signal_supply_strength,
            "cycle_count": self.cycle_count,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ActionResultConfig:
    """設定。"""
    # 最大蓄積対数
    max_pairs: int = 200

    # 構成バッファ最大数
    max_buffer: int = 30

    # 最大復帰候補数
    max_recovery_candidates: int = 50

    # 最大断面記述履歴数
    max_section_history: int = 100

    # 最大断面重み履歴数
    max_weight_history: int = 100

    # 最大減衰履歴数
    max_decay_history: int = 100

    # 最大収束監視記録数
    max_convergence_records: int = 50

    # 構成バッファの最低待機ティック数（即時構成禁止）
    min_buffer_ticks: int = 3

    # 構成バッファの滞留上限ティック数（結果未観測→保留移行）
    max_buffer_ticks: int = 30

    # 鮮度減衰速度（サイクルあたり）
    freshness_decay_rate: float = 0.02

    # 参照による鮮度回復量
    reference_recovery: float = 0.12

    # 再活性上限回数
    max_reactivation_count: int = 5

    # 収束警告の閾値（パターン集中率）
    convergence_threshold: float = 0.5

    # 断面偏り閾値（単一断面支配率）
    section_bias_threshold: float = 0.6

    # シグナル供給減衰率（偏り検出時）
    signal_attenuation_rate: float = 0.15

    # シグナル供給最低強度
    signal_min_strength: float = 0.1

    # バッファ過密閾値
    buffer_overflow_threshold: int = 20

    # パターン多様性復元時の鮮度回復量
    diversity_recovery_amount: float = 0.1


# =============================================================================
# Processor (6-stage pipeline)
# =============================================================================

class ActionResultObservationProcessor:
    """
    行動-結果の観測と蓄積プロセッサ。

    6段パイプライン:
    1. 対構成 (pair composition)
    2. 多断面評価記述 (multi-section description)
    3. 文脈帰属付与 (context attribution)
    4. 整列蓄積 (alignment and accumulation)
    5. 減衰忘却 (decay and forgetting)
    6. 受渡準備 (handoff preparation)

    出力は参照情報形式のみ。判断・評価・行動決定を直接起動しない。
    """

    def __init__(self, config: Optional[ActionResultConfig] = None):
        self._config = config or ActionResultConfig()
        self._state = ActionResultObservationState()

    @property
    def state(self) -> ActionResultObservationState:
        return self._state

    @state.setter
    def state(self, value: ActionResultObservationState) -> None:
        self._state = value

    def record_action(self, inputs: ActionResultInputs) -> None:
        """行動を構成バッファに記録する。

        結果との結合は min_buffer_ticks 経過後に行われる。
        同一周期内での即時構成を禁止する。
        """
        if not inputs.selected_policy_label:
            return

        action_desc = ActionDescription(
            policy_label=inputs.selected_policy_label,
            policy_axis=inputs.selected_policy_axis,
            selection_context=inputs.selection_context_summary,
            tick_at_action=inputs.current_tick,
            timestamp=time.time(),
        )

        pair = ActionResultPair(
            action=action_desc,
            status=PairStatus.BUFFERED.value,
            creation_tick=inputs.current_tick,
            creation_time=time.time(),
            pattern_key=inputs.selected_policy_axis or inputs.selected_policy_label,
        )

        self._state.composition_buffer.append(pair)

        # バッファ過密制御
        if len(self._state.composition_buffer) > self._config.max_buffer:
            # 最古のバッファを保留状態に移行
            overflow = self._state.composition_buffer[
                : len(self._state.composition_buffer) - self._config.max_buffer
            ]
            for old_pair in overflow:
                old_pair.status = PairStatus.PENDING.value
                self._state.total_pairs_pending += 1
            self._state.composition_buffer = self._state.composition_buffer[
                -self._config.max_buffer :
            ]

    def process(self, inputs: ActionResultInputs) -> ActionResultObservationResult:
        """
        6段パイプラインを実行する。

        出力は参照情報としてのみ流し、
        判断・評価・行動決定を直接起動しない。
        """
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # Stage 1: 対構成 — バッファ内の行動に結果を結合
        newly_composed = self._compose_pairs(inputs, now)

        # Stage 2: 多断面評価記述
        for pair in newly_composed:
            self._describe_multi_section(pair, inputs, now)

        # Stage 3: 文脈帰属付与
        for pair in newly_composed:
            self._attribute_context(pair, inputs, now)

        # Stage 4: 整列蓄積
        self._accumulate_pairs(newly_composed, now)

        # Stage 5: 減衰忘却（memory_forgetting_fixation パターン準拠）
        decay_result = self._apply_decay_and_forgetting(now)

        # Stage 6: 受渡準備（安全弁チェック含む）
        result = self._prepare_handoff(
            newly_composed, decay_result, now
        )

        return result

    def record_reference(self, pair_id: str) -> None:
        """対が参照されたことを記録し、鮮度を回復する。

        再活性上限を設け、特定の対が永続的に高鮮度を
        維持する構造にしない。
        """
        cfg = self._config
        for pair in self._state.pairs:
            if pair.pair_id == pair_id:
                pair.reference_count += 1
                pair.last_reference_time = time.time()

                # 再活性上限チェック
                if pair.reactivation_count < cfg.max_reactivation_count:
                    old_freshness = pair.freshness
                    pair.freshness = _clamp(
                        pair.freshness + cfg.reference_recovery
                    )
                    pair.freshness_stage = _stage_from_freshness(
                        pair.freshness
                    ).value
                    pair.reactivation_count += 1

                    # 減衰中なら復帰
                    if pair.status == PairStatus.DECAYING.value:
                        pair.status = PairStatus.ACTIVE.value
                        self._state.total_pairs_recovered += 1
                        if pair.pair_id not in self._state.recovery_candidates:
                            self._state.recovery_candidates.append(pair.pair_id)
                break

    def get_active_pairs(self) -> list[ActionResultPair]:
        """活性状態の対を返す（参照情報形式）。"""
        return [
            p for p in self._state.pairs
            if p.status in (
                PairStatus.COMPOSED.value,
                PairStatus.ACTIVE.value,
            )
        ]

    def get_pairs_by_pattern(self, pattern_key: str) -> list[ActionResultPair]:
        """パターンキーで対を検索する（参照情報形式）。"""
        return [
            p for p in self._state.pairs
            if p.pattern_key == pattern_key
            and p.status not in (
                PairStatus.INVISIBLE.value,
                PairStatus.BUFFERED.value,
            )
        ]

    def tick(self, inputs: ActionResultInputs) -> ActionResultObservationResult:
        """orchestrator から呼ばれる単一エントリポイント。

        行動記録（record_action）と6段パイプライン実行（process）を
        統合的に実行する。orchestrator は tick() のみを呼べばよい。

        - inputs に selected_policy_label がある場合、構成バッファに行動を記録
        - その後6段パイプラインを実行して結果を返す
        - 出力は参照情報形式のみ。判断・評価・行動決定を直接起動しない
        """
        # 行動記録（ポリシー選択があれば構成バッファに追加）
        if inputs.selected_policy_label:
            self.record_action(inputs)

        # 6段パイプライン実行
        return self.process(inputs)

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        蓄積された行動-結果対の概要（直近の傾向、断面分布、鮮度分布）を、
        外部表現生成時の参照情報として構造化形式で提供する。
        提供情報は記述形式であり、行動指示や評価判定を含まない。
        """
        st = self._state

        active_pairs = [
            p for p in st.pairs
            if p.status in (PairStatus.COMPOSED.value, PairStatus.ACTIVE.value)
        ]
        decaying_pairs = [
            p for p in st.pairs
            if p.status == PairStatus.DECAYING.value
        ]

        # パターン分布
        pattern_counts: dict[str, int] = {}
        for p in active_pairs:
            key = p.pattern_key or "unknown"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        # 鮮度分布
        freshness_dist: dict[str, int] = {}
        for p in st.pairs:
            stage = p.freshness_stage or FreshnessStage.ACTIVE.value
            freshness_dist[stage] = freshness_dist.get(stage, 0) + 1

        # 断面分布（活性対の断面記述をカウント）
        section_dist: dict[str, int] = {}
        for p in active_pairs:
            if p.result:
                for desc in p.result.sections:
                    sec = desc.section
                    section_dist[sec] = section_dist.get(sec, 0) + 1

        # テキスト要約
        summary_text = get_action_result_summary(st)

        return {
            "cycle_count": st.cycle_count,
            "active_count": len(active_pairs),
            "decaying_count": len(decaying_pairs),
            "buffered_count": len(st.composition_buffer),
            "total_composed": st.total_pairs_composed,
            "total_recovered": st.total_pairs_recovered,
            "pattern_distribution": pattern_counts,
            "freshness_distribution": freshness_dist,
            "section_distribution": section_dist,
            "pattern_convergence_warning": st.pattern_convergence_warning,
            "section_bias_warning": st.section_bias_warning,
            "signal_attenuation_active": st.signal_attenuation_active,
            "signal_supply_strength": st.signal_supply_strength,
            "summary_text": summary_text,
        }

    def get_freshness_compatible_info(self) -> list[dict[str, Any]]:
        """記憶忘却モジュールと互換性のある鮮度情報を返す。

        対の鮮度状態を既存の記憶鮮度構造と互換性のある形式で公開する。
        """
        result = []
        for pair in self._state.pairs:
            result.append({
                "id": pair.pair_id,
                "source": "action_result",
                "freshness": pair.freshness,
                "freshness_stage": pair.freshness_stage,
                "status": pair.status,
                "reference_count": pair.reference_count,
                "creation_time": pair.creation_time,
                "last_reference_time": pair.last_reference_time,
            })
        return result

    # ─── Stage 1: 対構成 ────────────────────────────────────────

    def _compose_pairs(
        self, inputs: ActionResultInputs, now: float,
    ) -> list[ActionResultPair]:
        """構成バッファ内の行動に結果記述を結合し、対を構成する。

        行動と結果の間には最低限の経過ティック（バッファ）を要求し、
        同一周期内での即時構成を禁止する。
        """
        cfg = self._config
        newly_composed: list[ActionResultPair] = []
        remaining_buffer: list[ActionResultPair] = []

        for buffered in self._state.composition_buffer:
            ticks_elapsed = inputs.current_tick - buffered.action.tick_at_action

            if ticks_elapsed >= cfg.min_buffer_ticks:
                # 十分なティックが経過 → 結果と結合
                buffered.status = PairStatus.COMPOSED.value
                newly_composed.append(buffered)
            elif ticks_elapsed >= cfg.max_buffer_ticks:
                # 長期滞留 → 結果未観測として保留
                buffered.status = PairStatus.PENDING.value
                self._state.total_pairs_pending += 1
            else:
                # まだバッファ待機中
                remaining_buffer.append(buffered)

        self._state.composition_buffer = remaining_buffer

        # バッファ過密化検出
        if len(self._state.composition_buffer) >= cfg.buffer_overflow_threshold:
            self._state.buffer_overflow_warning = True
            # 滞留記録を保留状態に移行
            overflow_count = len(self._state.composition_buffer) - cfg.buffer_overflow_threshold
            if overflow_count > 0:
                for old in self._state.composition_buffer[:overflow_count]:
                    old.status = PairStatus.PENDING.value
                    self._state.total_pairs_pending += 1
                self._state.composition_buffer = self._state.composition_buffer[
                    overflow_count:
                ]
        else:
            self._state.buffer_overflow_warning = False

        return newly_composed

    # ─── Stage 2: 多断面評価記述 ────────────────────────────────

    def _describe_multi_section(
        self,
        pair: ActionResultPair,
        inputs: ActionResultInputs,
        now: float,
    ) -> None:
        """結果を単一スコアや正誤値に集約せず、複数の独立した断面で記述する。

        断面間に優先順位を設けない。各断面は独立した記述として保持され、
        断面間の重みは動的に変動し、特定断面が恒常的に支配的になる構造を持たない。
        断面間の不一致は解消せず並立記録する。
        """
        sections: list[SectionDescription] = []

        # 外部反応断面
        if abs(inputs.external_response_change) > 0.01 or inputs.external_response_description:
            sections.append(SectionDescription(
                section=ObservationSection.EXTERNAL_REACTION.value,
                description=inputs.external_response_description or "external_change",
                value=inputs.external_response_change,
                timestamp=now,
            ))

        # 内部状態変化断面
        internal_delta = (
            abs(inputs.internal_state_delta)
            + abs(inputs.motivation_delta)
            + abs(inputs.direction_delta)
        ) / 3.0
        if internal_delta > 0.01:
            sections.append(SectionDescription(
                section=ObservationSection.INTERNAL_STATE_CHANGE.value,
                description=f"state={inputs.internal_state_delta:.2f},mot={inputs.motivation_delta:.2f},dir={inputs.direction_delta:.2f}",
                value=internal_delta,
                timestamp=now,
            ))

        # 感情推移断面
        emotion_diff = self._compute_emotion_diff(
            inputs.emotion_before, inputs.emotion_after
        )
        if emotion_diff > 0.01:
            sections.append(SectionDescription(
                section=ObservationSection.EMOTION_TRANSITION.value,
                description=f"emotion_diff={emotion_diff:.2f}",
                value=emotion_diff,
                timestamp=now,
            ))

        # 他者観測断面
        if abs(inputs.other_reaction_change) > 0.01 or inputs.other_reaction_description:
            sections.append(SectionDescription(
                section=ObservationSection.OTHER_OBSERVATION.value,
                description=inputs.other_reaction_description or "other_change",
                value=inputs.other_reaction_change,
                timestamp=now,
            ))

        # 時間経過断面
        sections.append(SectionDescription(
            section=ObservationSection.TIME_ELAPSED.value,
            description=f"ticks={inputs.ticks_since_action},elapsed={inputs.elapsed_seconds:.1f}s",
            value=_clamp(inputs.ticks_since_action / 10.0),
            timestamp=now,
        ))

        # 記憶参照断面
        if inputs.referenced_memory_count > 0:
            sections.append(SectionDescription(
                section=ObservationSection.MEMORY_REFERENCE.value,
                description=f"memories={inputs.referenced_memory_count}",
                value=_clamp(inputs.referenced_memory_count / 5.0),
                timestamp=now,
            ))

        pair.result = ResultDescription(
            sections=sections,
            tick_at_result=inputs.current_tick,
            timestamp=now,
        )

        # 断面記述履歴の更新
        for s in sections:
            self._state.section_description_history.append(s.to_dict())
        if len(self._state.section_description_history) > self._config.max_section_history:
            self._state.section_description_history = (
                self._state.section_description_history[
                    -self._config.max_section_history :
                ]
            )

    def _compute_emotion_diff(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> float:
        """感情状態の前後差分を計算する。"""
        all_keys = set(list(before.keys()) + list(after.keys()))
        if not all_keys:
            return 0.0
        total_diff = 0.0
        for key in all_keys:
            b = before.get(key, 0.0)
            a = after.get(key, 0.0)
            total_diff += abs(a - b)
        return total_diff / len(all_keys) if all_keys else 0.0

    # ─── Stage 3: 文脈帰属付与 ──────────────────────────────────

    def _attribute_context(
        self,
        pair: ActionResultPair,
        inputs: ActionResultInputs,
        now: float,
    ) -> None:
        """行動-結果対に文脈情報を付与する。

        同一の行動であっても文脈が異なれば異なる対として蓄積される。
        因果断定は行わず、「この文脈下でこの行動の後にこの結果が隣接した」
        という記述にとどめる。
        """
        pair.context = ContextAttribution(
            context_summary=inputs.context_summary,
            dialogue_state=inputs.dialogue_state,
            environment_tags=list(inputs.environment_tags),
            tick_at_context=inputs.current_tick,
            timestamp=now,
        )

    # ─── Stage 4: 整列蓄積 ──────────────────────────────────────

    def _accumulate_pairs(
        self,
        newly_composed: list[ActionResultPair],
        now: float,
    ) -> None:
        """構成された行動-結果対を時系列順に保持する。

        蓄積は追記形式で行い、特定パターンの対のみを優先保持する構造にしない。
        """
        cfg = self._config

        for pair in newly_composed:
            pair.status = PairStatus.ACTIVE.value
            pair.freshness = 1.0
            pair.freshness_stage = FreshnessStage.ACTIVE.value
            self._state.pairs.append(pair)
            self._state.time_index.append(pair.pair_id)
            self._state.total_pairs_composed += 1

        # 蓄積上限制御
        # 不可視化済みの対を優先的に除去（完全消去はしない—復帰候補に残す）
        if len(self._state.pairs) > cfg.max_pairs:
            remove_count = len(self._state.pairs) - cfg.max_pairs
            invisible = [
                p for p in self._state.pairs
                if p.status == PairStatus.INVISIBLE.value
            ]
            if invisible:
                remove_targets = invisible[:remove_count]
            else:
                # 不可視がない場合は最古の対を除去
                remove_targets = self._state.pairs[:remove_count]

            remove_ids = {p.pair_id for p in remove_targets}
            # 復帰候補に登録してから除去
            for pid in remove_ids:
                if pid not in self._state.recovery_candidates:
                    self._state.recovery_candidates.append(pid)
            self._state.pairs = [
                p for p in self._state.pairs
                if p.pair_id not in remove_ids
            ]
            self._state.time_index = [
                pid for pid in self._state.time_index
                if pid not in remove_ids
            ]

        # 復帰候補上限
        if len(self._state.recovery_candidates) > cfg.max_recovery_candidates:
            self._state.recovery_candidates = (
                self._state.recovery_candidates[-cfg.max_recovery_candidates :]
            )

    # ─── Stage 5: 減衰忘却 ──────────────────────────────────────

    def _apply_decay_and_forgetting(
        self, now: float,
    ) -> dict[str, int]:
        """蓄積された行動-結果対に時間経過に伴う鮮度の減衰を適用する。

        忘却の仕組みは既存の記憶忘却構造(memory_forgetting_fixation)と
        同一のパターンを適用し、二重の忘却メカニズムを新規に作らない。

        段階的な希薄化を経て不可視化へ向かうが、復帰経路は閉じない。
        """
        cfg = self._config
        newly_decayed = 0
        newly_invisible = 0

        for pair in self._state.pairs:
            if pair.status in (
                PairStatus.BUFFERED.value,
                PairStatus.PENDING.value,
                PairStatus.INVISIBLE.value,
            ):
                continue

            # 鮮度減衰
            pair.freshness = _clamp(pair.freshness - cfg.freshness_decay_rate)
            new_stage = _stage_from_freshness(pair.freshness)
            old_stage = pair.freshness_stage

            if new_stage.value != old_stage:
                pair.freshness_stage = new_stage.value

                # 減衰履歴に記録
                self._state.decay_history.append({
                    "pair_id": pair.pair_id,
                    "old_stage": old_stage,
                    "new_stage": new_stage.value,
                    "freshness": pair.freshness,
                    "timestamp": now,
                })

                if new_stage == FreshnessStage.INVISIBLE:
                    pair.status = PairStatus.INVISIBLE.value
                    newly_invisible += 1
                    self._state.total_pairs_decayed += 1
                    # 復帰候補に登録（不可逆忘却を防ぐ）
                    if pair.pair_id not in self._state.recovery_candidates:
                        self._state.recovery_candidates.append(pair.pair_id)
                elif new_stage in (
                    FreshnessStage.WEAKENING,
                    FreshnessStage.FADING,
                    FreshnessStage.NEAR_INVISIBLE,
                ):
                    if pair.status != PairStatus.DECAYING.value:
                        pair.status = PairStatus.DECAYING.value
                        newly_decayed += 1

        # 減衰履歴のトリミング
        if len(self._state.decay_history) > cfg.max_decay_history:
            self._state.decay_history = (
                self._state.decay_history[-cfg.max_decay_history :]
            )

        return {
            "newly_decayed": newly_decayed,
            "newly_invisible": newly_invisible,
        }

    # ─── Stage 6: 受渡準備 ──────────────────────────────────────

    def _prepare_handoff(
        self,
        newly_composed: list[ActionResultPair],
        decay_result: dict[str, int],
        now: float,
    ) -> ActionResultObservationResult:
        """安全弁チェックを行い結果を返す。

        蓄積された行動-結果対の情報を、他モジュールが参照可能な形式に整える。
        出力は情報としてのみ流し、判断、評価、行動決定を直接起動しない。
        """
        cfg = self._config

        # ── 統計収集 ──
        active_count = sum(
            1 for p in self._state.pairs
            if p.status in (PairStatus.COMPOSED.value, PairStatus.ACTIVE.value)
        )
        decaying_count = sum(
            1 for p in self._state.pairs
            if p.status == PairStatus.DECAYING.value
        )
        invisible_count = sum(
            1 for p in self._state.pairs
            if p.status == PairStatus.INVISIBLE.value
        )
        buffered_count = len(self._state.composition_buffer)
        pending_count = sum(
            1 for p in self._state.pairs
            if p.status == PairStatus.PENDING.value
        )

        # ── 断面分布 ──
        section_distribution = self._compute_section_distribution()

        # ── パターン分布 ──
        pattern_distribution = self._compute_pattern_distribution()

        # ── 鮮度分布 ──
        freshness_distribution = self._compute_freshness_distribution()

        # ── 収束監視 ──
        convergence = self._monitor_convergence(
            pattern_distribution, section_distribution, now
        )

        # ── 安全弁1: パターン収束警告 ──
        diversity_restored = False
        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            self._state.pattern_convergence_warning = True
            # 希薄化中の異パターン対を再浮上させ、複線状態に戻す
            diversity_restored = self._restore_pattern_diversity(now)
        else:
            self._state.pattern_convergence_warning = False

        # ── 安全弁2: 断面偏り警告 ──
        section_biased = self._check_section_bias(section_distribution)
        self._state.section_bias_warning = section_biased

        # ── 安全弁3: シグナル供給強度の自動減衰 ──
        self._adjust_signal_strength(convergence, now)

        # ── 安全弁4: バッファ過密 ──
        # (Stage 1 で処理済み)

        # ── 断面重み変動履歴の更新 ──
        self._update_section_weights(section_distribution, now)

        # ── 自己強化ループ防止 ──
        decay_slowed = self._prevent_self_reinforcement(
            pattern_distribution, now
        )

        return ActionResultObservationResult(
            newly_composed_pairs=newly_composed,
            active_pair_count=active_count,
            decaying_pair_count=decaying_count,
            invisible_pair_count=invisible_count,
            buffered_pair_count=buffered_count,
            pending_pair_count=pending_count,
            section_distribution=section_distribution,
            pattern_distribution=pattern_distribution,
            freshness_distribution=freshness_distribution,
            convergence_level=convergence.convergence_level,
            convergence_score=convergence.convergence_score,
            pattern_convergence_warning=self._state.pattern_convergence_warning,
            section_bias_warning=self._state.section_bias_warning,
            signal_attenuation_active=self._state.signal_attenuation_active,
            buffer_overflow_warning=self._state.buffer_overflow_warning,
            diversity_restored=diversity_restored,
            decay_slowed=decay_slowed,
            signal_supply_strength=self._state.signal_supply_strength,
            cycle_count=self._state.cycle_count,
        )

    # ─── 統計ヘルパー ───────────────────────────────────────────

    def _compute_section_distribution(self) -> dict[str, float]:
        """活性対の断面分布を計算する。"""
        section_counts: dict[str, int] = {}
        total = 0
        for pair in self._state.pairs:
            if pair.status in (
                PairStatus.INVISIBLE.value,
                PairStatus.BUFFERED.value,
                PairStatus.PENDING.value,
            ):
                continue
            for s in pair.result.sections:
                section_counts[s.section] = section_counts.get(s.section, 0) + 1
                total += 1
        if total == 0:
            return {}
        return {k: v / total for k, v in section_counts.items()}

    def _compute_pattern_distribution(self) -> dict[str, int]:
        """活性対のパターン分布を計算する。"""
        pattern_counts: dict[str, int] = {}
        for pair in self._state.pairs:
            if pair.status in (
                PairStatus.INVISIBLE.value,
                PairStatus.BUFFERED.value,
                PairStatus.PENDING.value,
            ):
                continue
            key = pair.pattern_key or "unknown"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        return pattern_counts

    def _compute_freshness_distribution(self) -> dict[str, int]:
        """鮮度段階の分布を計算する。"""
        dist: dict[str, int] = {}
        for pair in self._state.pairs:
            stage = pair.freshness_stage
            dist[stage] = dist.get(stage, 0) + 1
        return dist

    # ─── 収束監視 ────────────────────────────────────────────────

    def _monitor_convergence(
        self,
        pattern_dist: dict[str, int],
        section_dist: dict[str, float],
        now: float,
    ) -> ConvergenceRecord:
        """蓄積傾向の偏りを記録として残す。

        既存の対の評価を変更するのではなく、
        蓄積傾向の偏りを記録として残す。
        """
        cfg = self._config

        # パターン多様性
        total_patterns = sum(pattern_dist.values()) if pattern_dist else 0
        unique_patterns = len(pattern_dist)
        pattern_diversity = (
            unique_patterns / max(total_patterns, 1)
            if total_patterns > 0
            else 1.0
        )

        # 支配的パターン
        dominant_pattern = ""
        dominant_ratio = 0.0
        if total_patterns > 0:
            dominant_pattern = max(
                pattern_dist, key=pattern_dist.get  # type: ignore
            )
            dominant_ratio = pattern_dist[dominant_pattern] / total_patterns

        # 断面多様性
        section_diversity = 1.0
        if section_dist:
            max_section_ratio = max(section_dist.values())
            section_diversity = 1.0 - max_section_ratio

        # 収束スコア
        convergence_score = _clamp(
            (1.0 - pattern_diversity) * 0.5
            + dominant_ratio * 0.3
            + (1.0 - section_diversity) * 0.2
        )
        convergence_level = _convergence_from_score(convergence_score)

        record = ConvergenceRecord(
            convergence_score=convergence_score,
            convergence_level=convergence_level.value,
            dominant_pattern=dominant_pattern,
            pattern_diversity=pattern_diversity,
            section_diversity=section_diversity,
            cycle=self._state.cycle_count,
            timestamp=now,
        )

        self._state.convergence_records.append(record)
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records :]
            )

        return record

    # ─── 安全弁 ──────────────────────────────────────────────────

    def _restore_pattern_diversity(self, now: float) -> bool:
        """パターン単一収束時、希薄化中の異パターン対を復帰候補として再浮上。

        複線状態に戻してから受け渡す。
        """
        cfg = self._config
        restored = False

        # 支配的パターンの特定
        pattern_dist = self._compute_pattern_distribution()
        if not pattern_dist:
            return False

        dominant = max(pattern_dist, key=pattern_dist.get)  # type: ignore

        # 異パターンの減衰中対を再浮上
        for pair in self._state.pairs:
            if pair.pattern_key != dominant and pair.status == PairStatus.DECAYING.value:
                pair.freshness = _clamp(
                    pair.freshness + cfg.diversity_recovery_amount
                )
                pair.freshness_stage = _stage_from_freshness(
                    pair.freshness
                ).value
                if pair.freshness >= 0.4:
                    pair.status = PairStatus.ACTIVE.value
                    self._state.total_pairs_recovered += 1
                restored = True

        return restored

    def _check_section_bias(
        self, section_dist: dict[str, float],
    ) -> bool:
        """断面分布が特定断面に偏っている場合を検出する。"""
        if not section_dist:
            return False
        max_ratio = max(section_dist.values())
        return max_ratio > self._config.section_bias_threshold

    def _adjust_signal_strength(
        self,
        convergence: ConvergenceRecord,
        now: float,
    ) -> None:
        """シグナル供給先への影響が累積し偏りが検出された場合、
        供給強度を段階的に減衰させる。

        既存シグナル（感情・責任）による変動を下回る水準まで
        自動的に縮小する。
        """
        cfg = self._config

        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            # 偏り検出 → シグナル強度減衰
            self._state.signal_supply_strength = _clamp(
                self._state.signal_supply_strength - cfg.signal_attenuation_rate,
                cfg.signal_min_strength,
                1.0,
            )
            self._state.signal_attenuation_active = True
        else:
            # 偏りなし → 徐々に回復（ただし急激な回復はしない）
            if self._state.signal_supply_strength < 1.0:
                self._state.signal_supply_strength = _clamp(
                    self._state.signal_supply_strength
                    + cfg.signal_attenuation_rate * 0.3,
                    cfg.signal_min_strength,
                    1.0,
                )
            if self._state.signal_supply_strength >= 0.9:
                self._state.signal_attenuation_active = False

    def _update_section_weights(
        self,
        section_dist: dict[str, float],
        now: float,
    ) -> None:
        """断面重み変動履歴を更新する。"""
        cfg = self._config
        for section, weight in section_dist.items():
            self._state.section_weight_history.append(SectionWeightRecord(
                section=section,
                weight=weight,
                cycle=self._state.cycle_count,
                timestamp=now,
            ))
        if len(self._state.section_weight_history) > cfg.max_weight_history:
            self._state.section_weight_history = (
                self._state.section_weight_history[-cfg.max_weight_history :]
            )

    def _prevent_self_reinforcement(
        self,
        pattern_dist: dict[str, int],
        now: float,
    ) -> bool:
        """特定パターンの反復蓄積→再選択促進→さらに蓄積の
        正のフィードバックループを抑制する。

        - 直近の蓄積傾向が特定パターンに偏った場合、
          シグナル供給の強度を自動的に減衰させる
        - 蓄積された対の多様性が低下した場合、
          希薄化していた異なるパターンの対を再浮上させる
        - 長期間構成されなかったパターンの対を完全消去しない
        - 不在パターンの情報を「未観測」として保持し、
          不在そのものを偏りの根拠にしない
        """
        cfg = self._config
        slowed = False

        if not pattern_dist:
            return False

        total = sum(pattern_dist.values())
        if total == 0:
            return False

        # パターン集中度チェック
        max_count = max(pattern_dist.values())
        concentration = max_count / total

        if concentration > cfg.convergence_threshold:
            # シグナル供給強度の追加減衰
            self._state.signal_supply_strength = _clamp(
                self._state.signal_supply_strength
                - cfg.signal_attenuation_rate * 0.5,
                cfg.signal_min_strength,
                1.0,
            )

            # 不在パターンの対を完全消去しない
            # → 減衰中の対の減衰を遅延
            for pair in self._state.pairs:
                if (
                    pair.status == PairStatus.DECAYING.value
                    and pair.freshness_stage
                    == FreshnessStage.NEAR_INVISIBLE.value
                ):
                    pair.freshness = _clamp(pair.freshness + 0.05)
                    pair.freshness_stage = _stage_from_freshness(
                        pair.freshness
                    ).value
                    slowed = True

        return slowed


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_action_result_summary(
    state: ActionResultObservationState,
) -> str:
    """行動-結果観測状態の要約（enrichment用）。

    蓄積された行動-結果対の概要（直近の傾向、断面分布、鮮度分布）を、
    外部表現生成時の参照情報としてenrichmentセクションに含める。
    提供情報は記述形式であり、行動指示や評価判定を含まない。
    """
    if state.cycle_count == 0 and not state.pairs and not state.composition_buffer:
        return "行動-結果観測: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    active = sum(
        1 for p in state.pairs
        if p.status in (PairStatus.COMPOSED.value, PairStatus.ACTIVE.value)
    )
    decaying = sum(
        1 for p in state.pairs
        if p.status == PairStatus.DECAYING.value
    )
    buffered = len(state.composition_buffer)

    if active:
        parts.append(f"活性対={active}")
    if decaying:
        parts.append(f"減衰中={decaying}")
    if buffered:
        parts.append(f"バッファ={buffered}")
    if state.total_pairs_composed:
        parts.append(f"構成累計={state.total_pairs_composed}")
    if state.total_pairs_recovered:
        parts.append(f"復帰={state.total_pairs_recovered}")

    # 直近パターン傾向
    pattern_counts: dict[str, int] = {}
    for p in state.pairs:
        if p.status in (PairStatus.COMPOSED.value, PairStatus.ACTIVE.value):
            key = p.pattern_key or "unknown"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
    if pattern_counts:
        top = max(pattern_counts, key=pattern_counts.get)  # type: ignore
        parts.append(f"主パターン={top}({pattern_counts[top]})")

    if state.pattern_convergence_warning:
        parts.append("収束偏向")
    if state.section_bias_warning:
        parts.append("断面偏り")
    if state.signal_attenuation_active:
        parts.append(f"シグナル減衰={state.signal_supply_strength:.2f}")

    return " ".join(parts) if parts else "行動-結果観測: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_action_result_processor(
    config: Optional[ActionResultConfig] = None,
) -> ActionResultObservationProcessor:
    """ActionResultObservationProcessor のファクトリ関数。"""
    return ActionResultObservationProcessor(config=config)
