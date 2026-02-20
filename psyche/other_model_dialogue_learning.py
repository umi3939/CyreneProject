"""
psyche/other_model_dialogue_learning.py - 他者観測の長期蓄積と仮説補助

対話経験の蓄積が推測層の入力を多様化するための構造。
他者の性質・意図・信念・価値の断定ではない。

設計原則 (design_other_model_dialogue_learning.md 準拠):
- 他者像を固定しない。蓄積は仮説生成の材料であり、他者の恒常的特性記述ではない
- 他者の反応を予測しない。文脈に応じた仮説候補間の重み分布の変動を記録するのみ
- 反復パターンのみを強調しない。反復と非反復を等重量で蓄積する
- 既存仮説の強度を直接加算しない。新たな仮説を再生成し旧仮説と競合並立させる
- 「自分がAをしたから相手がBをした」という因果帰属を行わない。時系列的隣接の記録
- 他者について断定しない: 意図、価値、信念、性格、行動傾向の恒常性
- 出力は参照情報としてのみ流し、判断・評価・行動決定を直接起動しない

CRITICAL CONSTRAINTS (other_agent_model から継承):
- 他者の意図・価値・信念を断定しない
- 正誤や善悪の評価を付与しない
- 目的や行動の最適化に結び付けない
- 自己像や人格の方向性を固定しない
- 候補は仮説として保持し固定しない
- 競合する候補を許容する
- 判断・目的・価値・責任に一切接続しない

8段パイプライン:
1. 蓄積候補抽出 (candidate extraction)
2. 正規化・文脈付与 (normalization and context)
3. 相手別整列 (user-separated alignment)
4. 反復・非反復識別 (repetition identification)
5. 仮説再生成材料構成 (hypothesis material composition)
6. 競合並立整理 (competing material arrangement)
7. 減衰・忘却 (decay and forgetting)
8. 受渡準備 (handoff preparation)
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

class InputSection(Enum):
    """入力参照の断面種別（8値）。"""
    SHORT_TERM_FRAGMENTS = "short_term_fragments"      # 短期観測断片集合
    ACTION_RESULT_OTHER = "action_result_other"        # 行動-結果対の他者観測断面
    DIALOGUE_CONTEXT = "dialogue_context"              # 対話文脈断面
    USER_IDENTITY = "user_identity"                    # 相手識別断面
    EMOTION_TONE = "emotion_tone"                      # 感情トーン断面
    RESPONSE_INTERVAL = "response_interval"            # 反応間隔断面
    TOPIC_TRANSITION = "topic_transition"              # 話題遷移断面
    ACCUMULATION_FRESHNESS = "accumulation_freshness"  # 蓄積鮮度断面


class FreshnessStage(Enum):
    """鮮度段階（memory_forgetting_fixation パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class EntryStatus(Enum):
    """蓄積記述の状態。"""
    ACTIVE = "active"
    DECAYING = "decaying"
    INVISIBLE = "invisible"


class PatternType(Enum):
    """パターン種別（反復/非反復を等重量で管理）。"""
    REPETITION = "repetition"      # 反復パターン
    NON_REPETITION = "non_repetition"  # 非反復（想定外の反応）


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
class AccumulationEntry:
    """蓄積記述単位。
    観測内容、文脈情報、相手識別、蓄積時点、出所種別を保持する。
    上書きされず追記される。
    """
    entry_id: str = ""
    user_id: str = ""                  # 相手識別
    source_type: str = ""              # 出所種別 (fragment / action_result)
    observation_type: str = ""         # 観測種別
    description: str = ""              # 観測内容
    value: float = 0.0
    confidence: float = 1.0
    # 文脈情報
    context_summary: str = ""
    dialogue_state: str = ""
    topic: str = ""
    # 感情トーン
    emotion_tone: str = ""
    emotion_value: float = 0.0
    # 時間
    creation_tick: int = 0
    creation_time: float = field(default_factory=time.time)
    # 鮮度
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value
    status: str = EntryStatus.ACTIVE.value
    # 参照管理
    reference_count: int = 0
    reactivation_count: int = 0
    last_reference_time: float = 0.0
    # パターン情報
    pattern_key: str = ""              # 観測内容のキー化（反復検出用）

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "observation_type": self.observation_type,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "context_summary": self.context_summary,
            "dialogue_state": self.dialogue_state,
            "topic": self.topic,
            "emotion_tone": self.emotion_tone,
            "emotion_value": self.emotion_value,
            "creation_tick": self.creation_tick,
            "creation_time": self.creation_time,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
            "status": self.status,
            "reference_count": self.reference_count,
            "reactivation_count": self.reactivation_count,
            "last_reference_time": self.last_reference_time,
            "pattern_key": self.pattern_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccumulationEntry":
        return cls(
            entry_id=data.get("entry_id", ""),
            user_id=data.get("user_id", ""),
            source_type=data.get("source_type", ""),
            observation_type=data.get("observation_type", ""),
            description=data.get("description", ""),
            value=data.get("value", 0.0),
            confidence=data.get("confidence", 1.0),
            context_summary=data.get("context_summary", ""),
            dialogue_state=data.get("dialogue_state", ""),
            topic=data.get("topic", ""),
            emotion_tone=data.get("emotion_tone", ""),
            emotion_value=data.get("emotion_value", 0.0),
            creation_tick=data.get("creation_tick", 0),
            creation_time=data.get("creation_time", time.time()),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
            status=data.get("status", EntryStatus.ACTIVE.value),
            reference_count=data.get("reference_count", 0),
            reactivation_count=data.get("reactivation_count", 0),
            last_reference_time=data.get("last_reference_time", 0.0),
            pattern_key=data.get("pattern_key", ""),
        )


@dataclass
class PatternRecord:
    """反復/非反復パターン記録。反復と非反復を等重量で保持する。"""
    pattern_id: str = ""
    user_id: str = ""
    pattern_type: str = PatternType.REPETITION.value
    pattern_key: str = ""
    observation_type: str = ""
    occurrence_count: int = 1
    last_seen_tick: int = 0
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "user_id": self.user_id,
            "pattern_type": self.pattern_type,
            "pattern_key": self.pattern_key,
            "observation_type": self.observation_type,
            "occurrence_count": self.occurrence_count,
            "last_seen_tick": self.last_seen_tick,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
            "creation_time": self.creation_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternRecord":
        return cls(
            pattern_id=data.get("pattern_id", ""),
            user_id=data.get("user_id", ""),
            pattern_type=data.get("pattern_type", PatternType.REPETITION.value),
            pattern_key=data.get("pattern_key", ""),
            observation_type=data.get("observation_type", ""),
            occurrence_count=data.get("occurrence_count", 1),
            last_seen_tick=data.get("last_seen_tick", 0),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
            creation_time=data.get("creation_time", time.time()),
        )


@dataclass
class HypothesisMaterial:
    """仮説再生成用材料。既存仮説の強度を直接加算しない。"""
    material_id: str = ""
    user_id: str = ""
    source_entry_ids: list[str] = field(default_factory=list)
    observation_type: str = ""
    description: str = ""
    context_summary: str = ""
    pattern_type: str = ""     # repetition / non_repetition
    supporting_count: int = 0
    freshness: float = 1.0
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.material_id:
            self.material_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "user_id": self.user_id,
            "source_entry_ids": list(self.source_entry_ids),
            "observation_type": self.observation_type,
            "description": self.description,
            "context_summary": self.context_summary,
            "pattern_type": self.pattern_type,
            "supporting_count": self.supporting_count,
            "freshness": self.freshness,
            "creation_time": self.creation_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisMaterial":
        return cls(
            material_id=data.get("material_id", ""),
            user_id=data.get("user_id", ""),
            source_entry_ids=list(data.get("source_entry_ids", [])),
            observation_type=data.get("observation_type", ""),
            description=data.get("description", ""),
            context_summary=data.get("context_summary", ""),
            pattern_type=data.get("pattern_type", ""),
            supporting_count=data.get("supporting_count", 0),
            freshness=data.get("freshness", 1.0),
            creation_time=data.get("creation_time", time.time()),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。"""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    dominant_direction: str = ""
    direction_diversity: float = 1.0
    user_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "dominant_direction": self.dominant_direction,
            "direction_diversity": self.direction_diversity,
            "user_diversity": self.user_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            dominant_direction=data.get("dominant_direction", ""),
            direction_diversity=data.get("direction_diversity", 1.0),
            user_diversity=data.get("user_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class DialogueLearningInputs:
    """8断面の入力データ。"""
    # 1. 短期観測断片集合
    short_term_fragments: list[dict[str, Any]] = field(default_factory=list)

    # 2. 行動-結果対の他者観測断面（因果帰属なし、時系列的隣接記録のみ）
    action_result_other_observations: list[dict[str, Any]] = field(default_factory=list)

    # 3. 対話文脈断面
    context_summary: str = ""
    dialogue_state: str = ""
    topic: str = ""

    # 4. 相手識別断面
    user_id: str = ""

    # 5. 感情トーン断面
    emotion_tone: str = ""
    emotion_value: float = 0.0

    # 6. 反応間隔断面
    response_interval_seconds: float = 0.0

    # 7. 話題遷移断面
    topic_changed: bool = False
    previous_topic: str = ""

    # 8. 蓄積鮮度断面
    existing_entry_count: int = 0
    average_freshness: float = 0.0

    # メタデータ
    current_tick: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class DialogueLearningState:
    """内部状態。内部保持12項目を包含する。"""
    # 1. 蓄積記述の集合（相手別に分離、追記可能な可変構造）
    entries: list[AccumulationEntry] = field(default_factory=list)

    # 2. 相手別索引（user_id → entry_id リスト）
    user_index: dict[str, list[str]] = field(default_factory=dict)

    # 3. 反復パターン履歴
    repetition_patterns: list[PatternRecord] = field(default_factory=list)

    # 4. 非反復記録（反復履歴と等重量で保持）
    non_repetition_records: list[PatternRecord] = field(default_factory=list)

    # 5. 仮説材料集合
    hypothesis_materials: list[HypothesisMaterial] = field(default_factory=list)

    # 6. 競合材料履歴
    competing_materials: list[dict[str, Any]] = field(default_factory=list)

    # 7. 鮮度状態（entries に内包）

    # 8. 減衰履歴
    decay_history: list[dict[str, Any]] = field(default_factory=list)

    # 9. 復帰候補履歴
    recovery_candidates: list[str] = field(default_factory=list)

    # 10. 供給履歴（再投入抑制用）
    supply_history: list[dict[str, Any]] = field(default_factory=list)

    # 11. 収束監視状態
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # カウンタ
    cycle_count: int = 0
    total_entries_added: int = 0
    total_entries_decayed: int = 0
    total_entries_recovered: int = 0
    total_materials_generated: int = 0

    # 安全弁フラグ
    single_image_warning: bool = False        # 他者像単一化防止
    confirmation_bias_warning: bool = False    # 確認バイアス防止
    self_fulfilling_warning: bool = False      # 自己成就的予言防止
    user_imbalance_warning: bool = False       # 相手別蓄積量偏り

    # 供給強度
    supply_strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "user_index": {k: list(v) for k, v in self.user_index.items()},
            "repetition_patterns": [p.to_dict() for p in self.repetition_patterns],
            "non_repetition_records": [p.to_dict() for p in self.non_repetition_records],
            "hypothesis_materials": [m.to_dict() for m in self.hypothesis_materials],
            "competing_materials": list(self.competing_materials),
            "decay_history": list(self.decay_history),
            "recovery_candidates": list(self.recovery_candidates),
            "supply_history": list(self.supply_history),
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "cycle_count": self.cycle_count,
            "total_entries_added": self.total_entries_added,
            "total_entries_decayed": self.total_entries_decayed,
            "total_entries_recovered": self.total_entries_recovered,
            "total_materials_generated": self.total_materials_generated,
            "single_image_warning": self.single_image_warning,
            "confirmation_bias_warning": self.confirmation_bias_warning,
            "self_fulfilling_warning": self.self_fulfilling_warning,
            "user_imbalance_warning": self.user_imbalance_warning,
            "supply_strength": self.supply_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogueLearningState":
        return cls(
            entries=[
                AccumulationEntry.from_dict(e) for e in data.get("entries", [])
            ],
            user_index={
                k: list(v) for k, v in data.get("user_index", {}).items()
            },
            repetition_patterns=[
                PatternRecord.from_dict(p) for p in data.get("repetition_patterns", [])
            ],
            non_repetition_records=[
                PatternRecord.from_dict(p) for p in data.get("non_repetition_records", [])
            ],
            hypothesis_materials=[
                HypothesisMaterial.from_dict(m) for m in data.get("hypothesis_materials", [])
            ],
            competing_materials=list(data.get("competing_materials", [])),
            decay_history=list(data.get("decay_history", [])),
            recovery_candidates=list(data.get("recovery_candidates", [])),
            supply_history=list(data.get("supply_history", [])),
            convergence_records=[
                ConvergenceRecord.from_dict(c) for c in data.get("convergence_records", [])
            ],
            cycle_count=data.get("cycle_count", 0),
            total_entries_added=data.get("total_entries_added", 0),
            total_entries_decayed=data.get("total_entries_decayed", 0),
            total_entries_recovered=data.get("total_entries_recovered", 0),
            total_materials_generated=data.get("total_materials_generated", 0),
            single_image_warning=data.get("single_image_warning", False),
            confirmation_bias_warning=data.get("confirmation_bias_warning", False),
            self_fulfilling_warning=data.get("self_fulfilling_warning", False),
            user_imbalance_warning=data.get("user_imbalance_warning", False),
            supply_strength=data.get("supply_strength", 1.0),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。

        永続化されたデータもセッション境界で鮮度が減衰し、有限の寿命を持つ。
        セッションをまたぐごとに蓄積記述の鮮度が一律に低下し、
        十分に希薄化した記述は不可視化へ移行する。
        """
        remove_ids: set[str] = set()
        for entry in self.entries:
            entry.freshness = _clamp(entry.freshness - decay_factor)
            entry.freshness_stage = _stage_from_freshness(entry.freshness).value
            if entry.freshness < 0.1:
                entry.status = EntryStatus.INVISIBLE.value
                remove_ids.add(entry.entry_id)

        for pattern in self.repetition_patterns:
            pattern.freshness = _clamp(pattern.freshness - decay_factor)
            pattern.freshness_stage = _stage_from_freshness(pattern.freshness).value

        for pattern in self.non_repetition_records:
            pattern.freshness = _clamp(pattern.freshness - decay_factor)
            pattern.freshness_stage = _stage_from_freshness(pattern.freshness).value

        for material in self.hypothesis_materials:
            material.freshness = _clamp(material.freshness - decay_factor)

        # 十分に希薄化した記述は復元対象から除外
        if remove_ids:
            self.entries = [
                e for e in self.entries if e.entry_id not in remove_ids
            ]
            for uid in self.user_index:
                self.user_index[uid] = [
                    eid for eid in self.user_index[uid] if eid not in remove_ids
                ]
            self.hypothesis_materials = [
                m for m in self.hypothesis_materials if m.freshness >= 0.1
            ]


# =============================================================================
# Result
# =============================================================================

@dataclass
class DialogueLearningResult:
    """処理結果（参照情報形式のみ）。"""
    # 新規蓄積された記述数
    newly_added_count: int = 0
    # 活性蓄積数
    active_entry_count: int = 0
    decaying_entry_count: int = 0
    invisible_entry_count: int = 0
    # 相手別蓄積数
    user_entry_counts: dict[str, int] = field(default_factory=dict)
    # パターン分布
    repetition_count: int = 0
    non_repetition_count: int = 0
    # 仮説材料数
    material_count: int = 0
    competing_material_count: int = 0
    # 鮮度分布
    freshness_distribution: dict[str, int] = field(default_factory=dict)
    # 収束監視
    convergence_level: str = ConvergenceLevel.NONE.value
    convergence_score: float = 0.0
    # 安全弁
    single_image_warning: bool = False
    confirmation_bias_warning: bool = False
    self_fulfilling_warning: bool = False
    user_imbalance_warning: bool = False
    diversity_restored: bool = False
    # 供給強度
    supply_strength: float = 1.0
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DialogueLearningConfig:
    """設定。"""
    max_entries: int = 300
    max_entries_per_user: int = 100
    max_patterns: int = 100
    max_materials: int = 50
    max_competing_materials: int = 30
    max_decay_history: int = 100
    max_recovery_candidates: int = 50
    max_supply_history: int = 50
    max_convergence_records: int = 50

    # 鮮度減衰速度
    freshness_decay_rate: float = 0.02
    # 相手不在時の追加減衰速度
    absent_user_decay_rate: float = 0.01
    # 参照による鮮度回復量
    reference_recovery: float = 0.12
    # 再活性上限回数
    max_reactivation_count: int = 5

    # 反復検出の閾値（同一パターンキーの最低出現数）
    repetition_threshold: int = 2
    # 収束警告閾値
    convergence_threshold: float = 0.5
    # 供給強度減衰率
    supply_attenuation_rate: float = 0.15
    # 供給最低強度
    supply_min_strength: float = 0.1
    # 相手別蓄積偏り閾値（最大相手/平均の比）
    user_imbalance_threshold: float = 3.0
    # 多様性復元時の鮮度回復量
    diversity_recovery_amount: float = 0.1
    # 蓄積候補の最低信頼度
    min_candidate_confidence: float = 0.1


# =============================================================================
# Processor (8-stage pipeline)
# =============================================================================

class DialogueLearningProcessor:
    """他者観測の長期蓄積と仮説補助プロセッサ。

    8段パイプライン:
    1. 蓄積候補抽出
    2. 正規化・文脈付与
    3. 相手別整列
    4. 反復・非反復識別
    5. 仮説再生成材料構成
    6. 競合並立整理
    7. 減衰・忘却
    8. 受渡準備

    出力は参照情報形式のみ。推測層の仮説強度を直接操作しない。
    """

    def __init__(self, config: Optional[DialogueLearningConfig] = None):
        self._config = config or DialogueLearningConfig()
        self._state = DialogueLearningState()

    @property
    def state(self) -> DialogueLearningState:
        return self._state

    @state.setter
    def state(self, value: DialogueLearningState) -> None:
        self._state = value

    def tick(self, inputs: DialogueLearningInputs) -> DialogueLearningResult:
        """orchestrator から呼ばれる単一エントリポイント。

        8段パイプラインを実行し結果を返す。
        出力は参照情報形式のみ。
        """
        return self.process(inputs)

    def process(self, inputs: DialogueLearningInputs) -> DialogueLearningResult:
        """8段パイプラインを実行する。"""
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # Stage 1: 蓄積候補抽出
        candidates = self._extract_candidates(inputs, now)

        # Stage 2: 正規化・文脈付与
        normalized = self._normalize_and_contextualize(candidates, inputs, now)

        # Stage 3: 相手別整列
        self._align_by_user(normalized, inputs)

        # Stage 4: 反復・非反復識別
        self._identify_patterns(normalized, inputs, now)

        # Stage 5: 仮説再生成材料構成
        self._compose_hypothesis_materials(inputs, now)

        # Stage 6: 競合並立整理
        self._arrange_competing_materials(inputs, now)

        # Stage 7: 減衰・忘却
        decay_result = self._apply_decay_and_forgetting(inputs, now)

        # Stage 8: 受渡準備
        return self._prepare_handoff(normalized, decay_result, now)

    def record_reference(self, entry_id: str) -> None:
        """蓄積記述が参照されたことを記録し、鮮度を回復する。

        再活性上限を設け、特定の蓄積が永続的に高鮮度を維持しない。
        """
        cfg = self._config
        for entry in self._state.entries:
            if entry.entry_id == entry_id:
                entry.reference_count += 1
                entry.last_reference_time = time.time()
                if entry.reactivation_count < cfg.max_reactivation_count:
                    entry.freshness = _clamp(entry.freshness + cfg.reference_recovery)
                    entry.freshness_stage = _stage_from_freshness(entry.freshness).value
                    entry.reactivation_count += 1
                    if entry.status == EntryStatus.DECAYING.value:
                        entry.status = EntryStatus.ACTIVE.value
                        self._state.total_entries_recovered += 1
                break

    def get_active_entries(self, user_id: str = "") -> list[AccumulationEntry]:
        """活性状態の蓄積記述を返す（参照情報形式）。"""
        result = [
            e for e in self._state.entries
            if e.status == EntryStatus.ACTIVE.value
        ]
        if user_id:
            result = [e for e in result if e.user_id == user_id]
        return result

    def get_hypothesis_materials(self, user_id: str = "") -> list[HypothesisMaterial]:
        """仮説再生成用材料を返す（参照情報形式）。
        既存仮説の強度を直接加算しない。
        """
        result = [
            m for m in self._state.hypothesis_materials
            if m.freshness >= 0.2
        ]
        if user_id:
            result = [m for m in result if m.user_id == user_id]
        return result

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        蓄積された対話経験の概要を外部表現生成時の参照情報として提供する。
        提供情報は記述形式であり、他者についての断定的特性記述や行動予測を含まない。
        """
        st = self._state
        active = [e for e in st.entries if e.status == EntryStatus.ACTIVE.value]
        decaying = [e for e in st.entries if e.status == EntryStatus.DECAYING.value]

        user_counts: dict[str, int] = {}
        for e in active:
            user_counts[e.user_id] = user_counts.get(e.user_id, 0) + 1

        freshness_dist: dict[str, int] = {}
        for e in st.entries:
            stage = e.freshness_stage
            freshness_dist[stage] = freshness_dist.get(stage, 0) + 1

        rep_active = sum(
            1 for p in st.repetition_patterns
            if p.freshness >= 0.2
        )
        non_rep_active = sum(
            1 for p in st.non_repetition_records
            if p.freshness >= 0.2
        )

        summary_text = get_dialogue_learning_summary(st)

        return {
            "cycle_count": st.cycle_count,
            "active_count": len(active),
            "decaying_count": len(decaying),
            "user_counts": user_counts,
            "repetition_active": rep_active,
            "non_repetition_active": non_rep_active,
            "material_count": len(st.hypothesis_materials),
            "freshness_distribution": freshness_dist,
            "single_image_warning": st.single_image_warning,
            "confirmation_bias_warning": st.confirmation_bias_warning,
            "self_fulfilling_warning": st.self_fulfilling_warning,
            "user_imbalance_warning": st.user_imbalance_warning,
            "supply_strength": st.supply_strength,
            "summary_text": summary_text,
        }

    # ─── Stage 1: 蓄積候補抽出 ─────────────────────────────────

    def _extract_candidates(
        self, inputs: DialogueLearningInputs, now: float,
    ) -> list[dict[str, Any]]:
        """短期観測断片および行動-結果対から蓄積候補を選出する。

        蓄積対象の選出基準は特定種類の観測に偏らない構造とし、
        すべての観測種別が蓄積候補になりうる状態を維持する。
        """
        cfg = self._config
        candidates: list[dict[str, Any]] = []

        # 短期観測断片集合からの抽出
        for frag in inputs.short_term_fragments:
            confidence = frag.get("confidence", 0.5)
            if confidence < cfg.min_candidate_confidence:
                continue
            candidates.append({
                "source_type": "fragment",
                "observation_type": frag.get("type", "unknown"),
                "description": frag.get("description", ""),
                "value": frag.get("value", 0.0),
                "confidence": confidence,
                "text_hint": frag.get("text_hint", ""),
            })

        # 行動-結果対の他者観測断面から（因果帰属なし、隣接記録のみ）
        for obs in inputs.action_result_other_observations:
            confidence = obs.get("confidence", 0.5)
            if confidence < cfg.min_candidate_confidence:
                continue
            candidates.append({
                "source_type": "action_result",
                "observation_type": obs.get("observation_type", "other_observation"),
                "description": obs.get("description", ""),
                "value": obs.get("value", 0.0),
                "confidence": confidence,
                "text_hint": obs.get("text_hint", ""),
            })

        return candidates

    # ─── Stage 2: 正規化・文脈付与 ──────────────────────────────

    def _normalize_and_contextualize(
        self,
        candidates: list[dict[str, Any]],
        inputs: DialogueLearningInputs,
        now: float,
    ) -> list[AccumulationEntry]:
        """蓄積候補を共通の蓄積記述単位へ変換し、文脈情報を付与する。

        同一の相手反応であっても文脈が異なれば異なる蓄積記述として保持する。
        """
        entries: list[AccumulationEntry] = []
        for cand in candidates:
            obs_type = cand.get("observation_type", "unknown")
            desc = cand.get("description", "")
            # パターンキーの構成
            pattern_key = f"{obs_type}:{desc}" if desc else obs_type

            entry = AccumulationEntry(
                user_id=inputs.user_id,
                source_type=cand.get("source_type", ""),
                observation_type=obs_type,
                description=desc,
                value=cand.get("value", 0.0),
                confidence=cand.get("confidence", 1.0),
                context_summary=inputs.context_summary,
                dialogue_state=inputs.dialogue_state,
                topic=inputs.topic,
                emotion_tone=inputs.emotion_tone,
                emotion_value=inputs.emotion_value,
                creation_tick=inputs.current_tick,
                creation_time=now,
                pattern_key=pattern_key,
            )
            entries.append(entry)
        return entries

    # ─── Stage 3: 相手別整列 ──────────────────────────────────

    def _align_by_user(
        self,
        entries: list[AccumulationEntry],
        inputs: DialogueLearningInputs,
    ) -> None:
        """蓄積記述を相手識別情報に基づいて分離管理する。

        相手別の管理は本機能内に限定し、既存の推測層の構造を変更しない。
        """
        cfg = self._config
        uid = inputs.user_id
        if not uid:
            return

        for entry in entries:
            self._state.entries.append(entry)
            self._state.total_entries_added += 1
            if uid not in self._state.user_index:
                self._state.user_index[uid] = []
            self._state.user_index[uid].append(entry.entry_id)

        # 相手別上限制御
        if uid in self._state.user_index:
            user_entries = self._state.user_index[uid]
            if len(user_entries) > cfg.max_entries_per_user:
                remove_count = len(user_entries) - cfg.max_entries_per_user
                remove_ids = set(user_entries[:remove_count])
                self._state.user_index[uid] = user_entries[remove_count:]
                # 復帰候補に登録してから除去
                for rid in remove_ids:
                    if rid not in self._state.recovery_candidates:
                        self._state.recovery_candidates.append(rid)
                self._state.entries = [
                    e for e in self._state.entries
                    if e.entry_id not in remove_ids
                ]

        # 全体上限制御
        if len(self._state.entries) > cfg.max_entries:
            remove_count = len(self._state.entries) - cfg.max_entries
            invisible = [
                e for e in self._state.entries
                if e.status == EntryStatus.INVISIBLE.value
            ]
            if invisible:
                remove_targets = invisible[:remove_count]
            else:
                remove_targets = self._state.entries[:remove_count]
            remove_ids = {e.entry_id for e in remove_targets}
            for rid in remove_ids:
                if rid not in self._state.recovery_candidates:
                    self._state.recovery_candidates.append(rid)
            self._state.entries = [
                e for e in self._state.entries if e.entry_id not in remove_ids
            ]
            for uid_key in self._state.user_index:
                self._state.user_index[uid_key] = [
                    eid for eid in self._state.user_index[uid_key]
                    if eid not in remove_ids
                ]

        # 復帰候補上限
        if len(self._state.recovery_candidates) > cfg.max_recovery_candidates:
            self._state.recovery_candidates = (
                self._state.recovery_candidates[-cfg.max_recovery_candidates:]
            )

    # ─── Stage 4: 反復・非反復識別 ──────────────────────────────

    def _identify_patterns(
        self,
        new_entries: list[AccumulationEntry],
        inputs: DialogueLearningInputs,
        now: float,
    ) -> None:
        """反復パターンを検出し、非反復も等しく蓄積する。

        反復パターンのみを優遇する重み付けを行わない。
        反復と非反復を等重量で保持することにより、確認バイアスの構造化を防止する。
        """
        cfg = self._config
        uid = inputs.user_id

        for entry in new_entries:
            pkey = entry.pattern_key
            if not pkey:
                continue

            # 既存パターンの検索（同一相手・同一パターンキー）
            existing = None
            for p in self._state.repetition_patterns:
                if p.user_id == uid and p.pattern_key == pkey:
                    existing = p
                    break

            if existing is not None:
                # 反復検出
                existing.occurrence_count += 1
                existing.last_seen_tick = inputs.current_tick
                existing.freshness = _clamp(existing.freshness + 0.05)
                existing.freshness_stage = _stage_from_freshness(existing.freshness).value
            else:
                # 全既存パターンのキー集合
                known_keys = {
                    p.pattern_key for p in self._state.repetition_patterns
                    if p.user_id == uid
                }
                # 同一ユーザーの蓄積記述から同キーの出現数を数える
                user_entries = [
                    e for e in self._state.entries
                    if e.user_id == uid and e.pattern_key == pkey
                ]
                if len(user_entries) >= cfg.repetition_threshold:
                    # 反復として記録
                    self._state.repetition_patterns.append(PatternRecord(
                        user_id=uid,
                        pattern_type=PatternType.REPETITION.value,
                        pattern_key=pkey,
                        observation_type=entry.observation_type,
                        occurrence_count=len(user_entries),
                        last_seen_tick=inputs.current_tick,
                        creation_time=now,
                    ))
                else:
                    # 非反復として記録（想定外の反応、等重量で保持）
                    self._state.non_repetition_records.append(PatternRecord(
                        user_id=uid,
                        pattern_type=PatternType.NON_REPETITION.value,
                        pattern_key=pkey,
                        observation_type=entry.observation_type,
                        occurrence_count=1,
                        last_seen_tick=inputs.current_tick,
                        creation_time=now,
                    ))

        # パターン上限制御
        if len(self._state.repetition_patterns) > cfg.max_patterns:
            self._state.repetition_patterns = (
                self._state.repetition_patterns[-cfg.max_patterns:]
            )
        if len(self._state.non_repetition_records) > cfg.max_patterns:
            self._state.non_repetition_records = (
                self._state.non_repetition_records[-cfg.max_patterns:]
            )

    # ─── Stage 5: 仮説再生成材料構成 ────────────────────────────

    def _compose_hypothesis_materials(
        self,
        inputs: DialogueLearningInputs,
        now: float,
    ) -> None:
        """蓄積記述群から推測層の仮説生成に供給する材料を構成する。

        この材料は既存仮説の強度を直接加算するためのものではなく、
        新たな仮説の生成根拠として提供する。
        """
        cfg = self._config
        uid = inputs.user_id
        if not uid:
            return

        # 活性な蓄積記述を観測種別でグループ化
        active_entries = [
            e for e in self._state.entries
            if e.user_id == uid and e.status == EntryStatus.ACTIVE.value
        ]
        if not active_entries:
            return

        type_groups: dict[str, list[AccumulationEntry]] = {}
        for e in active_entries:
            key = e.observation_type
            if key not in type_groups:
                type_groups[key] = []
            type_groups[key].append(e)

        # 各グループから材料を生成
        for obs_type, group in type_groups.items():
            if len(group) < 1:
                continue

            # 反復/非反復を識別
            pattern_key = group[0].pattern_key if group else ""
            is_repetition = any(
                p.pattern_key == pattern_key
                for p in self._state.repetition_patterns
                if p.user_id == uid
            )
            ptype = (
                PatternType.REPETITION.value if is_repetition
                else PatternType.NON_REPETITION.value
            )

            # 既存材料との重複チェック（供給履歴による再投入抑制）
            recent_supply = [
                s for s in self._state.supply_history
                if s.get("observation_type") == obs_type
                and s.get("user_id") == uid
            ]
            # 最近供給済みならスキップ（同一周期での再投入防止）
            if recent_supply and recent_supply[-1].get("cycle", 0) == self._state.cycle_count:
                continue

            # 材料構成
            material = HypothesisMaterial(
                user_id=uid,
                source_entry_ids=[e.entry_id for e in group[:10]],
                observation_type=obs_type,
                description=group[0].description,
                context_summary=inputs.context_summary,
                pattern_type=ptype,
                supporting_count=len(group),
                creation_time=now,
            )
            self._state.hypothesis_materials.append(material)
            self._state.total_materials_generated += 1

            # 供給履歴に記録
            self._state.supply_history.append({
                "material_id": material.material_id,
                "observation_type": obs_type,
                "user_id": uid,
                "cycle": self._state.cycle_count,
                "timestamp": now,
            })

        # 材料上限制御
        if len(self._state.hypothesis_materials) > cfg.max_materials:
            self._state.hypothesis_materials = (
                self._state.hypothesis_materials[-cfg.max_materials:]
            )
        if len(self._state.supply_history) > cfg.max_supply_history:
            self._state.supply_history = (
                self._state.supply_history[-cfg.max_supply_history:]
            )

    # ─── Stage 6: 競合並立整理 ─────────────────────────────────

    def _arrange_competing_materials(
        self,
        inputs: DialogueLearningInputs,
        now: float,
    ) -> None:
        """同一相手について異なる蓄積記述から導出可能な複数の仮説材料を並立保持する。

        矛盾する材料を排除せず、推測層に揺らぎ情報として渡す。
        材料が単一系列に収束した場合は安全弁が作動する。
        """
        uid = inputs.user_id
        if not uid:
            return

        user_materials = [
            m for m in self._state.hypothesis_materials
            if m.user_id == uid and m.freshness >= 0.2
        ]

        # 観測種別ごとに整理
        type_groups: dict[str, list[HypothesisMaterial]] = {}
        for m in user_materials:
            if m.observation_type not in type_groups:
                type_groups[m.observation_type] = []
            type_groups[m.observation_type].append(m)

        # 競合材料の記録（矛盾する材料を並立保持）
        for obs_type, mats in type_groups.items():
            if len(mats) > 1:
                self._state.competing_materials.append({
                    "user_id": uid,
                    "observation_type": obs_type,
                    "material_count": len(mats),
                    "material_ids": [m.material_id for m in mats],
                    "cycle": self._state.cycle_count,
                    "timestamp": now,
                })

        # 競合材料履歴上限制御
        cfg = self._config
        if len(self._state.competing_materials) > cfg.max_competing_materials:
            self._state.competing_materials = (
                self._state.competing_materials[-cfg.max_competing_materials:]
            )

    # ─── Stage 7: 減衰・忘却 ───────────────────────────────────

    def _apply_decay_and_forgetting(
        self,
        inputs: DialogueLearningInputs,
        now: float,
    ) -> dict[str, int]:
        """蓄積記述の鮮度を時間経過に伴い段階的に減衰させる。

        既存の記憶忘却構造と同一のパターンを適用する。
        段階的な希薄化を経て不可視化へ向かうが、復帰経路は閉じない。
        """
        cfg = self._config
        newly_decayed = 0
        newly_invisible = 0

        # 相手不在検出（当該相手との対話が長期間不在の場合の加速減衰）
        absent_users: set[str] = set()
        active_user = inputs.user_id
        for uid in self._state.user_index:
            if uid != active_user:
                absent_users.add(uid)

        for entry in self._state.entries:
            if entry.status == EntryStatus.INVISIBLE.value:
                continue

            # 基本減衰
            decay = cfg.freshness_decay_rate
            # 相手不在時の追加減衰
            if entry.user_id in absent_users:
                decay += cfg.absent_user_decay_rate

            entry.freshness = _clamp(entry.freshness - decay)
            new_stage = _stage_from_freshness(entry.freshness)
            old_stage = entry.freshness_stage

            if new_stage.value != old_stage:
                entry.freshness_stage = new_stage.value

                self._state.decay_history.append({
                    "entry_id": entry.entry_id,
                    "old_stage": old_stage,
                    "new_stage": new_stage.value,
                    "freshness": entry.freshness,
                    "timestamp": now,
                })

                if new_stage == FreshnessStage.INVISIBLE:
                    entry.status = EntryStatus.INVISIBLE.value
                    newly_invisible += 1
                    self._state.total_entries_decayed += 1
                    if entry.entry_id not in self._state.recovery_candidates:
                        self._state.recovery_candidates.append(entry.entry_id)
                elif new_stage in (
                    FreshnessStage.WEAKENING,
                    FreshnessStage.FADING,
                    FreshnessStage.NEAR_INVISIBLE,
                ):
                    if entry.status != EntryStatus.DECAYING.value:
                        entry.status = EntryStatus.DECAYING.value
                        newly_decayed += 1

        # パターン記録の鮮度減衰（反復・非反復ともに時間減衰対象）
        for pattern in self._state.repetition_patterns:
            pattern.freshness = _clamp(pattern.freshness - cfg.freshness_decay_rate)
            pattern.freshness_stage = _stage_from_freshness(pattern.freshness).value

        for pattern in self._state.non_repetition_records:
            pattern.freshness = _clamp(pattern.freshness - cfg.freshness_decay_rate)
            pattern.freshness_stage = _stage_from_freshness(pattern.freshness).value

        # 仮説材料の鮮度減衰
        for material in self._state.hypothesis_materials:
            material.freshness = _clamp(material.freshness - cfg.freshness_decay_rate)

        # 減衰履歴のトリミング
        if len(self._state.decay_history) > cfg.max_decay_history:
            self._state.decay_history = (
                self._state.decay_history[-cfg.max_decay_history:]
            )

        return {
            "newly_decayed": newly_decayed,
            "newly_invisible": newly_invisible,
        }

    # ─── Stage 8: 受渡準備 ────────────────────────────────────

    def _prepare_handoff(
        self,
        new_entries: list[AccumulationEntry],
        decay_result: dict[str, int],
        now: float,
    ) -> DialogueLearningResult:
        """安全弁チェックを行い結果を返す。

        出力は参照情報としてのみ流し、判断・評価・行動決定を直接起動しない。
        """
        cfg = self._config

        # ── 統計収集 ──
        active_count = sum(
            1 for e in self._state.entries
            if e.status == EntryStatus.ACTIVE.value
        )
        decaying_count = sum(
            1 for e in self._state.entries
            if e.status == EntryStatus.DECAYING.value
        )
        invisible_count = sum(
            1 for e in self._state.entries
            if e.status == EntryStatus.INVISIBLE.value
        )

        user_entry_counts: dict[str, int] = {}
        for e in self._state.entries:
            if e.status != EntryStatus.INVISIBLE.value:
                user_entry_counts[e.user_id] = user_entry_counts.get(e.user_id, 0) + 1

        rep_count = sum(
            1 for p in self._state.repetition_patterns if p.freshness >= 0.2
        )
        non_rep_count = sum(
            1 for p in self._state.non_repetition_records if p.freshness >= 0.2
        )

        freshness_dist = self._compute_freshness_distribution()

        # ── 収束監視 ──
        convergence = self._monitor_convergence(user_entry_counts, now)

        # ── 安全弁1: 他者像単一化防止 ──
        diversity_restored = False
        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            self._state.single_image_warning = True
            diversity_restored = self._restore_diversity(now)
        else:
            self._state.single_image_warning = False

        # ── 安全弁2: 確認バイアス防止 ──
        self._state.confirmation_bias_warning = self._check_confirmation_bias()

        # ── 安全弁3: 自己成就的予言防止 ──
        self._state.self_fulfilling_warning = self._check_self_fulfilling()

        # ── 安全弁4: 相手別蓄積量偏り ──
        self._check_user_imbalance(user_entry_counts)

        # ── 供給強度調整 ──
        self._adjust_supply_strength(convergence, now)

        material_count = sum(
            1 for m in self._state.hypothesis_materials if m.freshness >= 0.2
        )
        competing_count = len(self._state.competing_materials)

        return DialogueLearningResult(
            newly_added_count=len(new_entries),
            active_entry_count=active_count,
            decaying_entry_count=decaying_count,
            invisible_entry_count=invisible_count,
            user_entry_counts=user_entry_counts,
            repetition_count=rep_count,
            non_repetition_count=non_rep_count,
            material_count=material_count,
            competing_material_count=competing_count,
            freshness_distribution=freshness_dist,
            convergence_level=convergence.convergence_level,
            convergence_score=convergence.convergence_score,
            single_image_warning=self._state.single_image_warning,
            confirmation_bias_warning=self._state.confirmation_bias_warning,
            self_fulfilling_warning=self._state.self_fulfilling_warning,
            user_imbalance_warning=self._state.user_imbalance_warning,
            diversity_restored=diversity_restored,
            supply_strength=self._state.supply_strength,
            cycle_count=self._state.cycle_count,
        )

    # ─── 統計ヘルパー ─────────────────────────────────────────

    def _compute_freshness_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for entry in self._state.entries:
            stage = entry.freshness_stage
            dist[stage] = dist.get(stage, 0) + 1
        return dist

    # ─── 収束監視 ──────────────────────────────────────────────

    def _monitor_convergence(
        self,
        user_counts: dict[str, int],
        now: float,
    ) -> ConvergenceRecord:
        """蓄積傾向の偏りを記録として残す。"""
        cfg = self._config

        # 方向多様性（観測種別の分布）
        type_counts: dict[str, int] = {}
        for entry in self._state.entries:
            if entry.status == EntryStatus.INVISIBLE.value:
                continue
            type_counts[entry.observation_type] = (
                type_counts.get(entry.observation_type, 0) + 1
            )
        total_types = sum(type_counts.values()) if type_counts else 0
        unique_types = len(type_counts)
        direction_diversity = (
            unique_types / max(total_types, 1) if total_types > 0 else 1.0
        )

        dominant_direction = ""
        dominant_ratio = 0.0
        if total_types > 0:
            dominant_direction = max(type_counts, key=type_counts.get)  # type: ignore
            dominant_ratio = type_counts[dominant_direction] / total_types

        # ユーザー多様性
        total_users = sum(user_counts.values()) if user_counts else 0
        unique_users = len(user_counts)
        user_diversity = (
            unique_users / max(total_users, 1) if total_users > 0 else 1.0
        )

        convergence_score = _clamp(
            (1.0 - direction_diversity) * 0.5
            + dominant_ratio * 0.3
            + (1.0 - user_diversity) * 0.2
        )
        convergence_level = _convergence_from_score(convergence_score)

        record = ConvergenceRecord(
            convergence_score=convergence_score,
            convergence_level=convergence_level.value,
            dominant_direction=dominant_direction,
            direction_diversity=direction_diversity,
            user_diversity=user_diversity,
            cycle=self._state.cycle_count,
            timestamp=now,
        )

        self._state.convergence_records.append(record)
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

        return record

    # ─── 安全弁 ───────────────────────────────────────────────

    def _restore_diversity(self, now: float) -> bool:
        """他者像単一化防止。

        蓄積記述が特定相手について単一方向の他者像に収束した場合、
        希薄化中の異なる方向の記述を復帰候補として再浮上させ、
        複線状態に戻してから材料供給を行う。
        """
        cfg = self._config
        restored = False

        # 支配的な観測種別の特定
        type_counts: dict[str, int] = {}
        for e in self._state.entries:
            if e.status in (EntryStatus.ACTIVE.value, EntryStatus.DECAYING.value):
                type_counts[e.observation_type] = (
                    type_counts.get(e.observation_type, 0) + 1
                )

        if not type_counts:
            return False

        dominant_type = max(type_counts, key=type_counts.get)  # type: ignore

        # 異なる種別の減衰中記述を再浮上
        for entry in self._state.entries:
            if (
                entry.observation_type != dominant_type
                and entry.status == EntryStatus.DECAYING.value
            ):
                entry.freshness = _clamp(
                    entry.freshness + cfg.diversity_recovery_amount
                )
                entry.freshness_stage = _stage_from_freshness(entry.freshness).value
                if entry.freshness >= 0.4:
                    entry.status = EntryStatus.ACTIVE.value
                    self._state.total_entries_recovered += 1
                restored = True

        return restored

    def _check_confirmation_bias(self) -> bool:
        """確認バイアスの構造化を検出する。

        反復パターンのみが蓄積される偏りを検出。
        非反復記録が反復パターンに比して著しく少ない場合に警告。
        """
        rep_count = sum(
            1 for p in self._state.repetition_patterns if p.freshness >= 0.2
        )
        non_rep_count = sum(
            1 for p in self._state.non_repetition_records if p.freshness >= 0.2
        )

        if rep_count == 0 and non_rep_count == 0:
            return False

        total = rep_count + non_rep_count
        if total == 0:
            return False

        # 反復の比率が高すぎる場合に警告
        rep_ratio = rep_count / total
        if rep_ratio > 0.8 and rep_count >= 3:
            # 非反復記録の参照優先度を一時的に引き上げ
            for p in self._state.non_repetition_records:
                if p.freshness < 0.5:
                    p.freshness = _clamp(p.freshness + 0.05)
                    p.freshness_stage = _stage_from_freshness(p.freshness).value
            return True

        return False

    def _check_self_fulfilling(self) -> bool:
        """自己成就的予言を検出する。

        仮説材料が単一の仮説根拠に収束した場合に警告。
        競合材料履歴から代替根拠を補充し、材料の多様性を復元する。
        """
        materials = [
            m for m in self._state.hypothesis_materials if m.freshness >= 0.2
        ]
        if len(materials) <= 1:
            return False

        # 材料の種別分布
        type_counts: dict[str, int] = {}
        for m in materials:
            type_counts[m.observation_type] = (
                type_counts.get(m.observation_type, 0) + 1
            )

        total = sum(type_counts.values())
        if total == 0:
            return False

        max_ratio = max(type_counts.values()) / total
        if max_ratio > 0.8 and total >= 3:
            return True

        return False

    def _check_user_imbalance(self, user_counts: dict[str, int]) -> None:
        """相手別蓄積量の偏りを検出し、偏りがある場合に鮮度減衰を加速する。"""
        cfg = self._config

        if not user_counts or len(user_counts) < 2:
            self._state.user_imbalance_warning = False
            return

        avg = sum(user_counts.values()) / len(user_counts)
        max_count = max(user_counts.values())

        if avg > 0 and max_count / avg > cfg.user_imbalance_threshold:
            self._state.user_imbalance_warning = True
            # 最大蓄積量の相手の鮮度減衰を加速
            imbalanced_user = max(user_counts, key=user_counts.get)  # type: ignore
            for entry in self._state.entries:
                if entry.user_id == imbalanced_user and entry.status == EntryStatus.ACTIVE.value:
                    entry.freshness = _clamp(
                        entry.freshness - cfg.absent_user_decay_rate
                    )
                    entry.freshness_stage = _stage_from_freshness(entry.freshness).value
        else:
            self._state.user_imbalance_warning = False

    def _adjust_supply_strength(
        self, convergence: ConvergenceRecord, now: float,
    ) -> None:
        """供給強度を調整する。偏り検出時に減衰、それ以外は徐々に回復。"""
        cfg = self._config

        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            self._state.supply_strength = _clamp(
                self._state.supply_strength - cfg.supply_attenuation_rate,
                cfg.supply_min_strength,
                1.0,
            )
        else:
            if self._state.supply_strength < 1.0:
                self._state.supply_strength = _clamp(
                    self._state.supply_strength
                    + cfg.supply_attenuation_rate * 0.3,
                    cfg.supply_min_strength,
                    1.0,
                )


# =============================================================================
# Summary
# =============================================================================

def get_dialogue_learning_summary(state: DialogueLearningState) -> str:
    """enrichment用の要約テキスト。

    他者についての断定的特性記述や行動予測を含まない。
    """
    if state.cycle_count == 0 and not state.entries:
        return "他者蓄積: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    active = sum(
        1 for e in state.entries if e.status == EntryStatus.ACTIVE.value
    )
    decaying = sum(
        1 for e in state.entries if e.status == EntryStatus.DECAYING.value
    )

    if active:
        parts.append(f"活性={active}")
    if decaying:
        parts.append(f"減衰中={decaying}")

    user_count = len([
        uid for uid, eids in state.user_index.items() if eids
    ])
    if user_count:
        parts.append(f"相手={user_count}")

    rep = sum(1 for p in state.repetition_patterns if p.freshness >= 0.2)
    non_rep = sum(1 for p in state.non_repetition_records if p.freshness >= 0.2)
    if rep or non_rep:
        parts.append(f"反復={rep},非反復={non_rep}")

    mat_count = sum(1 for m in state.hypothesis_materials if m.freshness >= 0.2)
    if mat_count:
        parts.append(f"材料={mat_count}")

    if state.single_image_warning:
        parts.append("他者像収束")
    if state.confirmation_bias_warning:
        parts.append("確認偏向")
    if state.self_fulfilling_warning:
        parts.append("自己成就")
    if state.user_imbalance_warning:
        parts.append("相手偏り")

    return " ".join(parts) if parts else "他者蓄積: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_dialogue_learning_processor(
    config: Optional[DialogueLearningConfig] = None,
) -> DialogueLearningProcessor:
    """DialogueLearningProcessor のファクトリ関数。"""
    return DialogueLearningProcessor(config=config)
