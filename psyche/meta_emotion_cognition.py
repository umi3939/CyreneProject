"""
psyche/meta_emotion_cognition.py - メタ感情認知と変動候補生成

感情状態の時間的推移を観測し、推移パターンの特徴を数値特徴量として記述し、
そのパターンからの変動可能性を参照情報として等価に列挙する構造。

本モジュールは感情処理パイプライン（毎ティック実行される感情反応・ダイナミクス・
振幅・独立減衰・短期記憶連動の各処理）のパラメータを一切変更しない。
感情減衰率、ムードの直接値、短期記憶-感情連動の設定値への書き込みは構造的に禁止される。

設計原則 (design_meta_emotion_cognition.md 準拠):
- 感情の「矯正」「最適化」「正常化」を行わない
- 特定の感情状態を「望ましい」「望ましくない」と判定しない
- DECAY_RATE、EmotionDecayConfig、DynamicsConfig、ムードvalence/arousal、
  stm_emotion_couplingのconfigを変更しない
- 数値特徴量にカテゴリラベルを付与しない
- トリガーベースの候補生成をしない（常時列挙・等価候補）
- 変動候補に優劣や推奨順位を設けない
- 「異常な感情パターン」を定義しない
- コード内で「調整」「戦略」「介入」「修正」の語を使わない

7段パイプライン:
1. 感情状態取得 (state acquisition)
2. 推移パターン特徴抽出 (transition pattern feature extraction)
3. 持続パターン検出 (sustained pattern detection)
4. 変動候補列挙 (variation candidate enumeration)
5. 候補整列・競合保持 (candidate alignment and competing retention)
6. 蓄積 (accumulation)
7. 受渡準備 (handoff preparation)
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

class InputSection(Enum):
    """入力参照の断面種別（8値）。"""
    EMOTION_STATE = "emotion_state"              # 感情状態断面
    DYNAMICS_PHASE = "dynamics_phase"            # ダイナミクス相断面
    STM_COUPLING_RESULT = "stm_coupling_result"  # STM-感情連動結果断面
    SELF_MODEL_EMOTION = "self_model_emotion"    # 自己モデル感情記述断面
    AMPLITUDE_STATE = "amplitude_state"          # 振幅状態断面
    DIALOGUE_CONTEXT = "dialogue_context"        # 対話文脈断面
    MEMORY_REFERENCE = "memory_reference"        # 記憶参照断面
    ACCUMULATION_FRESHNESS = "accumulation_freshness"  # 蓄積鮮度断面


class FreshnessStage(Enum):
    """鮮度段階（memory_forgetting_fixation パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class RecordStatus(Enum):
    """蓄積記録の状態。"""
    ACTIVE = "active"
    DECAYING = "decaying"
    INVISIBLE = "invisible"


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
    """鮮度値から段階を返す。"""
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
class TransitionFeature:
    """推移パターン特徴量。数値特徴量のみ。カテゴリラベルを付与しない。

    持続時間、変化速度、振動周期、支配感情の安定度、感情間遷移の頻度
    といった連続的な数値として保持する。
    """
    feature_id: str = ""
    # 時間的特徴量（連続的数値のみ、カテゴリ名称なし）
    duration_ticks: int = 0            # 観測時点までの持続ティック数
    change_speed: float = 0.0          # 感情値の変化速度（差分の絶対値平均）
    oscillation_period: float = 0.0    # 振動周期（符号変化の間隔）
    dominant_stability: float = 0.0    # 支配感情の安定度（0-1, 変動が少ないほど高い）
    transition_frequency: float = 0.0  # 感情間遷移の頻度（支配感情の切替回数/ティック数）
    # ダイナミクス相関連の数値特徴量
    dynamics_phase_value: float = 0.0  # 現在のダイナミクス相の数値表現
    peak_intensity: float = 0.0        # ピーク強度
    accumulated_intensity: float = 0.0 # 蓄積強度
    # 振幅関連
    amplitude_value: float = 1.0       # 振幅係数
    amplitude_boost: float = 0.0       # 蓄積ブースト
    # ムード観測値（READ-ONLY, 変更しない）
    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    # STM連動の観測値（READ-ONLY, 変更しない）
    coupling_continuity: float = 0.0
    coupling_active_entries: int = 0
    # 自己モデル感情記述の観測値
    self_model_intensity: float = 0.0
    self_model_spread: float = 0.0
    self_model_conflict: bool = False
    # 境界値到達の事実記述
    # 感情値が 0.0 または 1.0 に到達した感情次元名の一覧
    # 境界値到達の良し悪しを判定しない。事実記録のみ。
    boundary_dimensions: list[str] = field(default_factory=list)  # 境界値に達している次元名
    boundary_count: int = 0  # 境界値に達している次元の数（冗長だが参照利便性のため）
    # メタデータ
    creation_tick: int = 0
    creation_time: float = field(default_factory=time.time)
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value

    def __post_init__(self):
        if not self.feature_id:
            self.feature_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "duration_ticks": self.duration_ticks,
            "change_speed": self.change_speed,
            "oscillation_period": self.oscillation_period,
            "dominant_stability": self.dominant_stability,
            "transition_frequency": self.transition_frequency,
            "dynamics_phase_value": self.dynamics_phase_value,
            "peak_intensity": self.peak_intensity,
            "accumulated_intensity": self.accumulated_intensity,
            "amplitude_value": self.amplitude_value,
            "amplitude_boost": self.amplitude_boost,
            "mood_valence": self.mood_valence,
            "mood_arousal": self.mood_arousal,
            "coupling_continuity": self.coupling_continuity,
            "coupling_active_entries": self.coupling_active_entries,
            "self_model_intensity": self.self_model_intensity,
            "self_model_spread": self.self_model_spread,
            "self_model_conflict": self.self_model_conflict,
            "boundary_dimensions": list(self.boundary_dimensions),
            "boundary_count": self.boundary_count,
            "creation_tick": self.creation_tick,
            "creation_time": self.creation_time,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionFeature":
        return cls(
            feature_id=data.get("feature_id", ""),
            duration_ticks=data.get("duration_ticks", 0),
            change_speed=data.get("change_speed", 0.0),
            oscillation_period=data.get("oscillation_period", 0.0),
            dominant_stability=data.get("dominant_stability", 0.0),
            transition_frequency=data.get("transition_frequency", 0.0),
            dynamics_phase_value=data.get("dynamics_phase_value", 0.0),
            peak_intensity=data.get("peak_intensity", 0.0),
            accumulated_intensity=data.get("accumulated_intensity", 0.0),
            amplitude_value=data.get("amplitude_value", 1.0),
            amplitude_boost=data.get("amplitude_boost", 0.0),
            mood_valence=data.get("mood_valence", 0.0),
            mood_arousal=data.get("mood_arousal", 0.0),
            coupling_continuity=data.get("coupling_continuity", 0.0),
            coupling_active_entries=data.get("coupling_active_entries", 0),
            self_model_intensity=data.get("self_model_intensity", 0.0),
            self_model_spread=data.get("self_model_spread", 0.0),
            self_model_conflict=data.get("self_model_conflict", False),
            boundary_dimensions=list(data.get("boundary_dimensions", [])),
            boundary_count=data.get("boundary_count", 0),
            creation_tick=data.get("creation_tick", 0),
            creation_time=data.get("creation_time", time.time()),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
        )


@dataclass
class SustainedPattern:
    """持続パターン記述。時系列的に反復または持続する特徴の記録。

    「このような推移特徴が継続している」という記述であり、
    「この推移が異常である」「この推移を変更すべきである」という評価を含まない。
    """
    pattern_id: str = ""
    # 数値特徴量（カテゴリラベルなし）
    sustained_change_speed: float = 0.0
    sustained_dominant_stability: float = 0.0
    sustained_transition_frequency: float = 0.0
    sustained_amplitude: float = 1.0
    sustained_mood_valence: float = 0.0
    sustained_mood_arousal: float = 0.0
    # 持続期間（ティック数）
    sustained_ticks: int = 0
    # 一致した特徴量の数
    matching_feature_count: int = 0
    # メタデータ
    creation_tick: int = 0
    creation_time: float = field(default_factory=time.time)
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value
    status: str = RecordStatus.ACTIVE.value

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "sustained_change_speed": self.sustained_change_speed,
            "sustained_dominant_stability": self.sustained_dominant_stability,
            "sustained_transition_frequency": self.sustained_transition_frequency,
            "sustained_amplitude": self.sustained_amplitude,
            "sustained_mood_valence": self.sustained_mood_valence,
            "sustained_mood_arousal": self.sustained_mood_arousal,
            "sustained_ticks": self.sustained_ticks,
            "matching_feature_count": self.matching_feature_count,
            "creation_tick": self.creation_tick,
            "creation_time": self.creation_time,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SustainedPattern":
        return cls(
            pattern_id=data.get("pattern_id", ""),
            sustained_change_speed=data.get("sustained_change_speed", 0.0),
            sustained_dominant_stability=data.get("sustained_dominant_stability", 0.0),
            sustained_transition_frequency=data.get("sustained_transition_frequency", 0.0),
            sustained_amplitude=data.get("sustained_amplitude", 1.0),
            sustained_mood_valence=data.get("sustained_mood_valence", 0.0),
            sustained_mood_arousal=data.get("sustained_mood_arousal", 0.0),
            sustained_ticks=data.get("sustained_ticks", 0),
            matching_feature_count=data.get("matching_feature_count", 0),
            creation_tick=data.get("creation_tick", 0),
            creation_time=data.get("creation_time", time.time()),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
            status=data.get("status", RecordStatus.ACTIVE.value),
        )


@dataclass
class VariationCandidate:
    """変動候補。現在のパターンからの変動可能性の等価な記述。

    変動候補に優劣や推奨順位を設けない。
    トリガーベースではなく、常時列挙される。
    """
    candidate_id: str = ""
    # 変動方向の数値記述（カテゴリラベルなし、優劣なし）
    delta_change_speed: float = 0.0
    delta_dominant_stability: float = 0.0
    delta_transition_frequency: float = 0.0
    delta_amplitude: float = 0.0
    delta_mood_valence: float = 0.0
    delta_mood_arousal: float = 0.0
    # 変動の根拠となった特徴量のID
    source_feature_id: str = ""
    source_pattern_id: str = ""
    # メタデータ
    creation_tick: int = 0
    creation_time: float = field(default_factory=time.time)
    freshness: float = 1.0

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "delta_change_speed": self.delta_change_speed,
            "delta_dominant_stability": self.delta_dominant_stability,
            "delta_transition_frequency": self.delta_transition_frequency,
            "delta_amplitude": self.delta_amplitude,
            "delta_mood_valence": self.delta_mood_valence,
            "delta_mood_arousal": self.delta_mood_arousal,
            "source_feature_id": self.source_feature_id,
            "source_pattern_id": self.source_pattern_id,
            "creation_tick": self.creation_tick,
            "creation_time": self.creation_time,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariationCandidate":
        return cls(
            candidate_id=data.get("candidate_id", ""),
            delta_change_speed=data.get("delta_change_speed", 0.0),
            delta_dominant_stability=data.get("delta_dominant_stability", 0.0),
            delta_transition_frequency=data.get("delta_transition_frequency", 0.0),
            delta_amplitude=data.get("delta_amplitude", 0.0),
            delta_mood_valence=data.get("delta_mood_valence", 0.0),
            delta_mood_arousal=data.get("delta_mood_arousal", 0.0),
            source_feature_id=data.get("source_feature_id", ""),
            source_pattern_id=data.get("source_pattern_id", ""),
            creation_tick=data.get("creation_tick", 0),
            creation_time=data.get("creation_time", time.time()),
            freshness=data.get("freshness", 1.0),
        )


@dataclass
class CognitionRecord:
    """認知履歴の1記録。時点・推移特徴・変動候補の事実記録。"""
    record_id: str = ""
    tick: int = 0
    feature_id: str = ""
    pattern_ids: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    candidate_count: int = 0
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value
    status: str = RecordStatus.ACTIVE.value
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tick": self.tick,
            "feature_id": self.feature_id,
            "pattern_ids": list(self.pattern_ids),
            "candidate_ids": list(self.candidate_ids),
            "candidate_count": self.candidate_count,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
            "status": self.status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitionRecord":
        return cls(
            record_id=data.get("record_id", ""),
            tick=data.get("tick", 0),
            feature_id=data.get("feature_id", ""),
            pattern_ids=list(data.get("pattern_ids", [])),
            candidate_ids=list(data.get("candidate_ids", [])),
            candidate_count=data.get("candidate_count", 0),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
            status=data.get("status", RecordStatus.ACTIVE.value),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。"""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    candidate_diversity: float = 1.0
    feature_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "candidate_diversity": self.candidate_diversity,
            "feature_diversity": self.feature_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            candidate_diversity=data.get("candidate_diversity", 1.0),
            feature_diversity=data.get("feature_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class MetaEmotionInputs:
    """8断面の入力データ。すべてREAD-ONLY参照。"""
    # 1. 感情状態断面（感情ベクトル、ムード — READ-ONLY）
    emotion_values: dict[str, float] = field(default_factory=dict)
    mood_valence: float = 0.0
    mood_arousal: float = 0.3

    # 2. ダイナミクス相断面（READ-ONLY）
    dynamics_phase: str = "normal"  # normal / peak / rebound
    dynamics_peak_intensity: float = 0.0
    dynamics_accumulated_intensity: float = 0.0

    # 3. STM-感情連動結果断面（READ-ONLY）
    coupling_continuity: float = 0.0
    coupling_active_entries: int = 0

    # 4. 自己モデル感情記述断面（READ-ONLY）
    self_model_spread: float = 0.0     # 拡散度
    self_model_intensity: float = 0.0  # 強度
    self_model_conflict: bool = False   # 共存対

    # 5. 振幅状態断面（READ-ONLY）
    amplitude_value: float = 1.0
    amplitude_boost: float = 0.0

    # 6. 対話文脈断面
    context_summary: str = ""
    dialogue_state: str = ""

    # 7. 記憶参照断面
    referenced_memory_count: int = 0

    # 8. 蓄積鮮度断面
    existing_record_count: int = 0
    average_freshness: float = 0.0

    # メタデータ
    current_tick: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class MetaEmotionState:
    """内部状態。内部保持10項目を包含する。"""
    # 1. 推移パターン特徴量の時系列
    feature_history: list[TransitionFeature] = field(default_factory=list)

    # 2. 持続パターン記述の集合
    sustained_patterns: list[SustainedPattern] = field(default_factory=list)

    # 3. 変動候補の集合（現在の候補）
    current_candidates: list[VariationCandidate] = field(default_factory=list)

    # 4. 認知履歴
    cognition_history: list[CognitionRecord] = field(default_factory=list)

    # 5. 候補生成履歴
    candidate_history: list[dict[str, Any]] = field(default_factory=list)

    # 6. 鮮度状態（feature_history, sustained_patterns, cognition_history に内包）

    # 7. 減衰履歴
    decay_history: list[dict[str, Any]] = field(default_factory=list)

    # 8. 復帰候補履歴
    recovery_candidates: list[str] = field(default_factory=list)

    # 9. 収束監視状態
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # 10. 供給履歴
    supply_history: list[dict[str, Any]] = field(default_factory=list)

    # 感情値の直近履歴（推移計算用、保持のみ）
    _emotion_snapshots: list[dict[str, float]] = field(default_factory=list)
    _dominant_history: list[str] = field(default_factory=list)

    # カウンタ
    cycle_count: int = 0
    total_features_extracted: int = 0
    total_patterns_detected: int = 0
    total_candidates_generated: int = 0
    total_records_decayed: int = 0
    total_records_recovered: int = 0

    # 安全弁フラグ
    candidate_convergence_warning: bool = False   # 変動候補単一方向収束
    feature_bias_warning: bool = False            # 特徴量偏り
    supply_concentration_warning: bool = False    # 供給集中
    accumulation_bias_warning: bool = False       # 蓄積偏り

    # 供給強度
    supply_strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_history": [f.to_dict() for f in self.feature_history],
            "sustained_patterns": [p.to_dict() for p in self.sustained_patterns],
            "current_candidates": [c.to_dict() for c in self.current_candidates],
            "cognition_history": [r.to_dict() for r in self.cognition_history],
            "candidate_history": list(self.candidate_history),
            "decay_history": list(self.decay_history),
            "recovery_candidates": list(self.recovery_candidates),
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "supply_history": list(self.supply_history),
            "emotion_snapshots": list(self._emotion_snapshots),
            "dominant_history": list(self._dominant_history),
            "cycle_count": self.cycle_count,
            "total_features_extracted": self.total_features_extracted,
            "total_patterns_detected": self.total_patterns_detected,
            "total_candidates_generated": self.total_candidates_generated,
            "total_records_decayed": self.total_records_decayed,
            "total_records_recovered": self.total_records_recovered,
            "candidate_convergence_warning": self.candidate_convergence_warning,
            "feature_bias_warning": self.feature_bias_warning,
            "supply_concentration_warning": self.supply_concentration_warning,
            "accumulation_bias_warning": self.accumulation_bias_warning,
            "supply_strength": self.supply_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaEmotionState":
        return cls(
            feature_history=[
                TransitionFeature.from_dict(f) for f in data.get("feature_history", [])
            ],
            sustained_patterns=[
                SustainedPattern.from_dict(p) for p in data.get("sustained_patterns", [])
            ],
            current_candidates=[
                VariationCandidate.from_dict(c) for c in data.get("current_candidates", [])
            ],
            cognition_history=[
                CognitionRecord.from_dict(r) for r in data.get("cognition_history", [])
            ],
            candidate_history=list(data.get("candidate_history", [])),
            decay_history=list(data.get("decay_history", [])),
            recovery_candidates=list(data.get("recovery_candidates", [])),
            convergence_records=[
                ConvergenceRecord.from_dict(c) for c in data.get("convergence_records", [])
            ],
            supply_history=list(data.get("supply_history", [])),
            _emotion_snapshots=list(data.get("emotion_snapshots", [])),
            _dominant_history=list(data.get("dominant_history", [])),
            cycle_count=data.get("cycle_count", 0),
            total_features_extracted=data.get("total_features_extracted", 0),
            total_patterns_detected=data.get("total_patterns_detected", 0),
            total_candidates_generated=data.get("total_candidates_generated", 0),
            total_records_decayed=data.get("total_records_decayed", 0),
            total_records_recovered=data.get("total_records_recovered", 0),
            candidate_convergence_warning=data.get("candidate_convergence_warning", False),
            feature_bias_warning=data.get("feature_bias_warning", False),
            supply_concentration_warning=data.get("supply_concentration_warning", False),
            accumulation_bias_warning=data.get("accumulation_bias_warning", False),
            supply_strength=data.get("supply_strength", 1.0),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。"""
        remove_ids: set[str] = set()

        for feat in self.feature_history:
            feat.freshness = _clamp(feat.freshness - decay_factor)
            feat.freshness_stage = _stage_from_freshness(feat.freshness).value

        for pat in self.sustained_patterns:
            pat.freshness = _clamp(pat.freshness - decay_factor)
            pat.freshness_stage = _stage_from_freshness(pat.freshness).value
            if pat.freshness < 0.1:
                pat.status = RecordStatus.INVISIBLE.value
                remove_ids.add(pat.pattern_id)

        for rec in self.cognition_history:
            rec.freshness = _clamp(rec.freshness - decay_factor)
            rec.freshness_stage = _stage_from_freshness(rec.freshness).value
            if rec.freshness < 0.1:
                rec.status = RecordStatus.INVISIBLE.value
                remove_ids.add(rec.record_id)

        for cand in self.current_candidates:
            cand.freshness = _clamp(cand.freshness - decay_factor)

        if remove_ids:
            self.sustained_patterns = [
                p for p in self.sustained_patterns if p.pattern_id not in remove_ids
            ]
            self.cognition_history = [
                r for r in self.cognition_history if r.record_id not in remove_ids
            ]
            self.current_candidates = [
                c for c in self.current_candidates if c.freshness >= 0.1
            ]


# =============================================================================
# Result
# =============================================================================

@dataclass
class MetaEmotionResult:
    """処理結果（参照情報形式のみ）。"""
    # 抽出された推移パターン特徴量
    current_feature: Optional[TransitionFeature] = None
    # 検出された持続パターン数
    active_pattern_count: int = 0
    decaying_pattern_count: int = 0
    # 列挙された変動候補数
    candidate_count: int = 0
    # 蓄積概要
    cognition_record_count: int = 0
    # 鮮度分布
    freshness_distribution: dict[str, int] = field(default_factory=dict)
    # 収束監視
    convergence_level: str = ConvergenceLevel.NONE.value
    convergence_score: float = 0.0
    # 安全弁
    candidate_convergence_warning: bool = False
    feature_bias_warning: bool = False
    supply_concentration_warning: bool = False
    accumulation_bias_warning: bool = False
    diversity_restored: bool = False
    # 供給強度
    supply_strength: float = 1.0
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MetaEmotionConfig:
    """設定。"""
    # 特徴量履歴の最大保持数
    max_feature_history: int = 50
    # 持続パターンの最大保持数
    max_sustained_patterns: int = 30
    # 変動候補の最大保持数
    max_candidates: int = 30
    # 認知履歴の最大保持数
    max_cognition_history: int = 100
    # 候補生成履歴の最大保持数
    max_candidate_history: int = 50
    # 減衰履歴の最大保持数
    max_decay_history: int = 50
    # 復帰候補の最大保持数
    max_recovery_candidates: int = 30
    # 収束記録の最大保持数
    max_convergence_records: int = 30
    # 供給履歴の最大保持数
    max_supply_history: int = 30
    # 感情スナップショットの最大保持数（推移計算用）
    max_emotion_snapshots: int = 20

    # 鮮度減衰速度
    freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))
    # 参照による鮮度回復量
    reference_recovery: float = 0.1
    # 持続パターン検出の特徴量一致閾値
    sustained_similarity_threshold: float = 0.3
    # 収束警告閾値
    convergence_threshold: float = 0.5
    # 供給強度減衰率
    supply_attenuation_rate: float = 0.15
    # 供給最低強度
    supply_min_strength: float = 0.1
    # 多様性復元時の鮮度回復量
    diversity_recovery_amount: float = 0.1


# =============================================================================
# Processor (7-stage pipeline)
# =============================================================================

class MetaEmotionProcessor:
    """メタ感情認知と変動候補生成プロセッサ。

    7段パイプライン:
    1. 感情状態取得
    2. 推移パターン特徴抽出
    3. 持続パターン検出
    4. 変動候補列挙
    5. 候補整列・競合保持
    6. 蓄積
    7. 受渡準備

    感情処理パイプラインのパラメータを一切変更しない。
    出力は参照情報形式のみ。
    """

    def __init__(self, config: Optional[MetaEmotionConfig] = None):
        self._config = config or MetaEmotionConfig()
        self._state = MetaEmotionState()

    @property
    def state(self) -> MetaEmotionState:
        return self._state

    @state.setter
    def state(self, value: MetaEmotionState) -> None:
        self._state = value

    def tick(self, inputs: MetaEmotionInputs) -> MetaEmotionResult:
        """orchestrator から呼ばれる単一エントリポイント。"""
        return self.process(inputs)

    def process(self, inputs: MetaEmotionInputs) -> MetaEmotionResult:
        """7段パイプラインを実行する。"""
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # Stage 1: 感情状態取得
        snapshot = self._acquire_state(inputs)

        # Stage 2: 推移パターン特徴抽出
        feature = self._extract_transition_features(inputs, snapshot, now)

        # Stage 3: 持続パターン検出
        self._detect_sustained_patterns(feature, inputs, now)

        # Stage 4: 変動候補列挙（常時実行、トリガーベースではない）
        candidates = self._enumerate_variation_candidates(feature, inputs, now)

        # Stage 5: 候補整列・競合保持
        self._align_and_retain_candidates(candidates, now)

        # Stage 6: 蓄積
        self._accumulate(feature, candidates, inputs, now)

        # Stage 7: 受渡準備
        return self._prepare_handoff(feature, now)

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        認知結果の概要を外部表現生成時の参照情報として提供する。
        評価判定（「この感情を変えるべき」等）を含まない。
        """
        st = self._state

        active_patterns = [
            p for p in st.sustained_patterns
            if p.status == RecordStatus.ACTIVE.value
        ]
        decaying_patterns = [
            p for p in st.sustained_patterns
            if p.status == RecordStatus.DECAYING.value
        ]

        active_candidates = [
            c for c in st.current_candidates if c.freshness >= 0.2
        ]

        freshness_dist: dict[str, int] = {}
        for rec in st.cognition_history:
            stage = rec.freshness_stage
            freshness_dist[stage] = freshness_dist.get(stage, 0) + 1

        latest_feature = st.feature_history[-1] if st.feature_history else None

        summary_text = get_meta_emotion_summary(st)

        return {
            "cycle_count": st.cycle_count,
            "active_pattern_count": len(active_patterns),
            "decaying_pattern_count": len(decaying_patterns),
            "candidate_count": len(active_candidates),
            "cognition_record_count": len(st.cognition_history),
            "freshness_distribution": freshness_dist,
            "latest_feature": latest_feature.to_dict() if latest_feature else {},
            "candidate_convergence_warning": st.candidate_convergence_warning,
            "feature_bias_warning": st.feature_bias_warning,
            "supply_concentration_warning": st.supply_concentration_warning,
            "accumulation_bias_warning": st.accumulation_bias_warning,
            "supply_strength": st.supply_strength,
            "summary_text": summary_text,
        }

    # ─── Stage 1: 感情状態取得 ────────────────────────────────────

    def _acquire_state(self, inputs: MetaEmotionInputs) -> dict[str, float]:
        """感情処理パイプラインの出力状態をREAD-ONLYで収集する。

        パイプラインの内部状態を変更しない。
        """
        snapshot = dict(inputs.emotion_values) if inputs.emotion_values else {}
        cfg = self._config

        # 感情スナップショットの蓄積（推移計算用）
        self._state._emotion_snapshots.append(snapshot)
        if len(self._state._emotion_snapshots) > cfg.max_emotion_snapshots:
            self._state._emotion_snapshots = (
                self._state._emotion_snapshots[-cfg.max_emotion_snapshots:]
            )

        # 支配感情の履歴
        dominant = ""
        if snapshot:
            dominant = max(snapshot, key=snapshot.get)  # type: ignore
        self._state._dominant_history.append(dominant)
        if len(self._state._dominant_history) > cfg.max_emotion_snapshots:
            self._state._dominant_history = (
                self._state._dominant_history[-cfg.max_emotion_snapshots:]
            )

        return snapshot

    # ─── Stage 2: 推移パターン特徴抽出 ───────────────────────────

    def _extract_transition_features(
        self,
        inputs: MetaEmotionInputs,
        snapshot: dict[str, float],
        now: float,
    ) -> TransitionFeature:
        """直近の複数ティック分の感情状態の推移から、時間的な特徴を
        数値特徴量として抽出する。

        特徴量はカテゴリ名称ではなく、連続的な数値として保持する。
        特徴量にカテゴリラベルを付与しない。
        """
        cfg = self._config
        snapshots = self._state._emotion_snapshots
        dominant_hist = self._state._dominant_history

        # 変化速度: 直近のスナップショット間の差分の絶対値平均
        change_speed = 0.0
        if len(snapshots) >= 2:
            diffs: list[float] = []
            for i in range(1, len(snapshots)):
                prev = snapshots[i - 1]
                curr = snapshots[i]
                all_keys = set(prev.keys()) | set(curr.keys())
                if all_keys:
                    diff = sum(
                        abs(curr.get(k, 0.0) - prev.get(k, 0.0))
                        for k in all_keys
                    ) / len(all_keys)
                    diffs.append(diff)
            if diffs:
                change_speed = sum(diffs) / len(diffs)

        # 振動周期: 支配感情の符号変化（切替）の間隔
        oscillation_period = 0.0
        if len(dominant_hist) >= 3:
            changes = 0
            for i in range(1, len(dominant_hist)):
                if dominant_hist[i] != dominant_hist[i - 1] and dominant_hist[i]:
                    changes += 1
            if changes > 0:
                oscillation_period = len(dominant_hist) / changes

        # 支配感情の安定度
        dominant_stability = 0.0
        if dominant_hist:
            unique_dominants = set(d for d in dominant_hist if d)
            if unique_dominants:
                most_common = max(
                    unique_dominants,
                    key=lambda d: dominant_hist.count(d),
                )
                dominant_stability = dominant_hist.count(most_common) / len(dominant_hist)

        # 感情間遷移の頻度
        transition_frequency = 0.0
        if len(dominant_hist) >= 2:
            transitions = sum(
                1 for i in range(1, len(dominant_hist))
                if dominant_hist[i] != dominant_hist[i - 1]
                and dominant_hist[i] and dominant_hist[i - 1]
            )
            transition_frequency = transitions / (len(dominant_hist) - 1)

        # ダイナミクス相の数値表現
        dynamics_phase_map = {"normal": 0.0, "peak": 1.0, "rebound": 0.5}
        dynamics_phase_value = dynamics_phase_map.get(inputs.dynamics_phase, 0.0)

        # 境界値到達の事実記述:
        # 感情値が 0.0 以下または 1.0 以上に到達した感情次元名を列挙する。
        # クリッピング境界（0.0, 1.0）そのものを閾値とする。
        # 境界値到達自体を「問題」として扱わない。事実の記録のみ。
        boundary_dims: list[str] = []
        for dim_name, dim_val in inputs.emotion_values.items():
            if dim_val <= 0.0 or dim_val >= 1.0:
                boundary_dims.append(dim_name)

        feature = TransitionFeature(
            duration_ticks=len(snapshots),
            change_speed=change_speed,
            oscillation_period=oscillation_period,
            dominant_stability=dominant_stability,
            transition_frequency=transition_frequency,
            dynamics_phase_value=dynamics_phase_value,
            peak_intensity=inputs.dynamics_peak_intensity,
            accumulated_intensity=inputs.dynamics_accumulated_intensity,
            amplitude_value=inputs.amplitude_value,
            amplitude_boost=inputs.amplitude_boost,
            mood_valence=inputs.mood_valence,
            mood_arousal=inputs.mood_arousal,
            coupling_continuity=inputs.coupling_continuity,
            coupling_active_entries=inputs.coupling_active_entries,
            self_model_intensity=inputs.self_model_intensity,
            self_model_spread=inputs.self_model_spread,
            self_model_conflict=inputs.self_model_conflict,
            boundary_dimensions=boundary_dims,
            boundary_count=len(boundary_dims),
            creation_tick=inputs.current_tick,
            creation_time=now,
        )

        self._state.feature_history.append(feature)
        self._state.total_features_extracted += 1

        # 特徴量履歴の上限制御
        if len(self._state.feature_history) > cfg.max_feature_history:
            self._state.feature_history = (
                self._state.feature_history[-cfg.max_feature_history:]
            )

        return feature

    # ─── Stage 3: 持続パターン検出 ───────────────────────────────

    def _detect_sustained_patterns(
        self,
        feature: TransitionFeature,
        inputs: MetaEmotionInputs,
        now: float,
    ) -> None:
        """抽出された数値特徴量群から、時系列的に反復または持続する特徴を検出する。

        検出されたパターンは「このような推移特徴が継続している」という記述であり、
        「この推移が異常である」「この推移を変更すべきである」という評価を含まない。
        """
        cfg = self._config
        history = self._state.feature_history

        if len(history) < 3:
            return

        # 直近の特徴量群との類似度を計算
        recent = history[-3:]
        avg_change = sum(f.change_speed for f in recent) / len(recent)
        avg_stability = sum(f.dominant_stability for f in recent) / len(recent)
        avg_transition = sum(f.transition_frequency for f in recent) / len(recent)
        avg_amplitude = sum(f.amplitude_value for f in recent) / len(recent)
        avg_valence = sum(f.mood_valence for f in recent) / len(recent)
        avg_arousal = sum(f.mood_arousal for f in recent) / len(recent)

        # 変動度（直近特徴量の分散）
        var_change = sum(
            (f.change_speed - avg_change) ** 2 for f in recent
        ) / len(recent)
        var_stability = sum(
            (f.dominant_stability - avg_stability) ** 2 for f in recent
        ) / len(recent)

        # 変動度が低い = 持続パターンの可能性
        combined_variance = (var_change + var_stability) / 2.0

        if combined_variance < cfg.sustained_similarity_threshold:
            # 既存の持続パターンとの照合
            matched = False
            for pat in self._state.sustained_patterns:
                if pat.status == RecordStatus.INVISIBLE.value:
                    continue
                sim = self._pattern_similarity(pat, avg_change, avg_stability, avg_transition)
                if sim > (1.0 - cfg.sustained_similarity_threshold):
                    # 既存パターンの持続を確認
                    pat.sustained_ticks += 1
                    pat.freshness = _clamp(pat.freshness + 0.02)
                    pat.freshness_stage = _stage_from_freshness(pat.freshness).value
                    matched = True
                    break

            if not matched:
                # 新規持続パターンの記録
                new_pattern = SustainedPattern(
                    sustained_change_speed=avg_change,
                    sustained_dominant_stability=avg_stability,
                    sustained_transition_frequency=avg_transition,
                    sustained_amplitude=avg_amplitude,
                    sustained_mood_valence=avg_valence,
                    sustained_mood_arousal=avg_arousal,
                    sustained_ticks=len(recent),
                    matching_feature_count=len(recent),
                    creation_tick=inputs.current_tick,
                    creation_time=now,
                )
                self._state.sustained_patterns.append(new_pattern)
                self._state.total_patterns_detected += 1

        # パターンの消失検出: 持続パターンと大きく異なる場合は減衰開始
        for pat in self._state.sustained_patterns:
            if pat.status == RecordStatus.ACTIVE.value:
                sim = self._pattern_similarity(
                    pat, feature.change_speed,
                    feature.dominant_stability,
                    feature.transition_frequency,
                )
                if sim < 0.3:
                    pat.status = RecordStatus.DECAYING.value

        # 持続パターン上限制御
        if len(self._state.sustained_patterns) > cfg.max_sustained_patterns:
            self._state.sustained_patterns = (
                self._state.sustained_patterns[-cfg.max_sustained_patterns:]
            )

    def _pattern_similarity(
        self,
        pat: SustainedPattern,
        change_speed: float,
        stability: float,
        transition_freq: float,
    ) -> float:
        """持続パターンと現在の特徴量の類似度を計算する。"""
        diff_change = abs(pat.sustained_change_speed - change_speed)
        diff_stability = abs(pat.sustained_dominant_stability - stability)
        diff_transition = abs(pat.sustained_transition_frequency - transition_freq)
        # 差分の平均を類似度に変換（差分が小さいほど類似度が高い）
        avg_diff = (diff_change + diff_stability + diff_transition) / 3.0
        return _clamp(1.0 - avg_diff)

    # ─── Stage 4: 変動候補列挙 ───────────────────────────────────

    def _enumerate_variation_candidates(
        self,
        feature: TransitionFeature,
        inputs: MetaEmotionInputs,
        now: float,
    ) -> list[VariationCandidate]:
        """現在の推移パターンから、変動しうる可能性を情報として複数列挙する。

        常に実行される。トリガーベースではない。
        変動候補に優劣や推奨順位を設けない。
        等価な可能性として列挙される。
        """
        candidates: list[VariationCandidate] = []

        # 候補1: 現在のパターンが続く場合（現状維持方向）
        candidates.append(VariationCandidate(
            delta_change_speed=0.0,
            delta_dominant_stability=0.0,
            delta_transition_frequency=0.0,
            delta_amplitude=0.0,
            delta_mood_valence=0.0,
            delta_mood_arousal=0.0,
            source_feature_id=feature.feature_id,
            creation_tick=inputs.current_tick,
            creation_time=now,
        ))

        # 候補2: 変化速度が増加する方向
        candidates.append(VariationCandidate(
            delta_change_speed=0.1,
            delta_dominant_stability=-0.05,
            delta_transition_frequency=0.05,
            delta_amplitude=0.0,
            delta_mood_valence=0.0,
            delta_mood_arousal=0.05,
            source_feature_id=feature.feature_id,
            creation_tick=inputs.current_tick,
            creation_time=now,
        ))

        # 候補3: 変化速度が減少する方向
        candidates.append(VariationCandidate(
            delta_change_speed=-0.1,
            delta_dominant_stability=0.05,
            delta_transition_frequency=-0.05,
            delta_amplitude=0.0,
            delta_mood_valence=0.0,
            delta_mood_arousal=-0.05,
            source_feature_id=feature.feature_id,
            creation_tick=inputs.current_tick,
            creation_time=now,
        ))

        # 候補4: 支配感情の安定度が変化する方向
        candidates.append(VariationCandidate(
            delta_change_speed=0.0,
            delta_dominant_stability=-0.1,
            delta_transition_frequency=0.1,
            delta_amplitude=0.0,
            delta_mood_valence=0.0,
            delta_mood_arousal=0.0,
            source_feature_id=feature.feature_id,
            creation_tick=inputs.current_tick,
            creation_time=now,
        ))

        # 候補5: ムード方向の変動
        candidates.append(VariationCandidate(
            delta_change_speed=0.0,
            delta_dominant_stability=0.0,
            delta_transition_frequency=0.0,
            delta_amplitude=0.0,
            delta_mood_valence=0.1 if feature.mood_valence <= 0.0 else -0.1,
            delta_mood_arousal=0.05 if feature.mood_arousal <= 0.5 else -0.05,
            source_feature_id=feature.feature_id,
            creation_tick=inputs.current_tick,
            creation_time=now,
        ))

        # 持続パターンからの候補: パターンが変化する場合
        for pat in self._state.sustained_patterns:
            if pat.status != RecordStatus.ACTIVE.value:
                continue
            # パターンの特徴量とは逆方向の変動可能性
            candidates.append(VariationCandidate(
                delta_change_speed=-pat.sustained_change_speed * 0.2,
                delta_dominant_stability=-pat.sustained_dominant_stability * 0.2 + 0.1,
                delta_transition_frequency=-pat.sustained_transition_frequency * 0.2,
                delta_amplitude=0.0,
                delta_mood_valence=0.0,
                delta_mood_arousal=0.0,
                source_feature_id=feature.feature_id,
                source_pattern_id=pat.pattern_id,
                creation_tick=inputs.current_tick,
                creation_time=now,
            ))

        self._state.total_candidates_generated += len(candidates)

        return candidates

    # ─── Stage 5: 候補整列・競合保持 ─────────────────────────────

    def _align_and_retain_candidates(
        self,
        candidates: list[VariationCandidate],
        now: float,
    ) -> None:
        """列挙された変動候補を並立保持する。

        矛盾する候補を排除せず、揺らぎ情報として保持する。
        """
        cfg = self._config

        # 現在の候補を更新
        self._state.current_candidates = candidates

        # 候補生成履歴に記録
        self._state.candidate_history.append({
            "candidate_count": len(candidates),
            "cycle": self._state.cycle_count,
            "timestamp": now,
        })

        # 候補上限制御
        if len(self._state.current_candidates) > cfg.max_candidates:
            self._state.current_candidates = (
                self._state.current_candidates[-cfg.max_candidates:]
            )
        if len(self._state.candidate_history) > cfg.max_candidate_history:
            self._state.candidate_history = (
                self._state.candidate_history[-cfg.max_candidate_history:]
            )

    # ─── Stage 6: 蓄積 ─────────────────────────────────────────

    def _accumulate(
        self,
        feature: TransitionFeature,
        candidates: list[VariationCandidate],
        inputs: MetaEmotionInputs,
        now: float,
    ) -> None:
        """認知結果を時系列に蓄積し、減衰・忘却を適用する。

        蓄積は時系列的隣接記録であり、事実の記録にとどめる。
        """
        cfg = self._config

        # 認知記録を追加
        record = CognitionRecord(
            tick=inputs.current_tick,
            feature_id=feature.feature_id,
            pattern_ids=[
                p.pattern_id for p in self._state.sustained_patterns
                if p.status == RecordStatus.ACTIVE.value
            ],
            candidate_ids=[c.candidate_id for c in candidates],
            candidate_count=len(candidates),
            timestamp=now,
        )
        self._state.cognition_history.append(record)

        # ── 減衰・忘却（既存の記憶忘却構造と同一パターン） ──
        newly_decayed = 0
        newly_invisible = 0

        for rec in self._state.cognition_history:
            if rec.status == RecordStatus.INVISIBLE.value:
                continue
            rec.freshness = _clamp(rec.freshness - cfg.freshness_decay_rate)
            new_stage = _stage_from_freshness(rec.freshness)
            old_stage = rec.freshness_stage

            if new_stage.value != old_stage:
                rec.freshness_stage = new_stage.value
                self._state.decay_history.append({
                    "record_id": rec.record_id,
                    "old_stage": old_stage,
                    "new_stage": new_stage.value,
                    "freshness": rec.freshness,
                    "timestamp": now,
                })

                if new_stage == FreshnessStage.INVISIBLE:
                    rec.status = RecordStatus.INVISIBLE.value
                    newly_invisible += 1
                    self._state.total_records_decayed += 1
                    if rec.record_id not in self._state.recovery_candidates:
                        self._state.recovery_candidates.append(rec.record_id)
                elif new_stage in (
                    FreshnessStage.WEAKENING,
                    FreshnessStage.FADING,
                    FreshnessStage.NEAR_INVISIBLE,
                ):
                    if rec.status != RecordStatus.DECAYING.value:
                        rec.status = RecordStatus.DECAYING.value
                        newly_decayed += 1

        # 特徴量履歴の鮮度減衰
        for feat in self._state.feature_history:
            feat.freshness = _clamp(feat.freshness - cfg.freshness_decay_rate)
            feat.freshness_stage = _stage_from_freshness(feat.freshness).value

        # 持続パターンの鮮度減衰
        for pat in self._state.sustained_patterns:
            if pat.status == RecordStatus.INVISIBLE.value:
                continue
            pat.freshness = _clamp(pat.freshness - cfg.freshness_decay_rate)
            pat.freshness_stage = _stage_from_freshness(pat.freshness).value
            if pat.freshness < 0.1:
                pat.status = RecordStatus.INVISIBLE.value

        # 候補の鮮度減衰
        for cand in self._state.current_candidates:
            cand.freshness = _clamp(cand.freshness - cfg.freshness_decay_rate)

        # 上限制御
        if len(self._state.cognition_history) > cfg.max_cognition_history:
            overflow = self._state.cognition_history[:-cfg.max_cognition_history]
            for r in overflow:
                if r.record_id not in self._state.recovery_candidates:
                    self._state.recovery_candidates.append(r.record_id)
            self._state.cognition_history = (
                self._state.cognition_history[-cfg.max_cognition_history:]
            )
        if len(self._state.decay_history) > cfg.max_decay_history:
            self._state.decay_history = (
                self._state.decay_history[-cfg.max_decay_history:]
            )
        if len(self._state.recovery_candidates) > cfg.max_recovery_candidates:
            self._state.recovery_candidates = (
                self._state.recovery_candidates[-cfg.max_recovery_candidates:]
            )

    # ─── Stage 7: 受渡準備 ──────────────────────────────────────

    def _prepare_handoff(
        self,
        feature: TransitionFeature,
        now: float,
    ) -> MetaEmotionResult:
        """安全弁チェックを行い結果を返す。

        出力は参照情報形式のみ。判断・評価・行動決定・感情パラメータ変更を
        直接起動しない。
        """
        cfg = self._config

        # 統計収集
        active_patterns = sum(
            1 for p in self._state.sustained_patterns
            if p.status == RecordStatus.ACTIVE.value
        )
        decaying_patterns = sum(
            1 for p in self._state.sustained_patterns
            if p.status == RecordStatus.DECAYING.value
        )
        candidate_count = len(self._state.current_candidates)
        cognition_count = sum(
            1 for r in self._state.cognition_history
            if r.status != RecordStatus.INVISIBLE.value
        )

        freshness_dist = self._compute_freshness_distribution()

        # 収束監視
        convergence = self._monitor_convergence(now)

        # ── 安全弁1: 変動候補単一方向収束 ──
        diversity_restored = False
        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            self._state.candidate_convergence_warning = True
            diversity_restored = self._restore_candidate_diversity(now)
        else:
            self._state.candidate_convergence_warning = False

        # ── 安全弁2: 特徴量偏り ──
        self._state.feature_bias_warning = self._check_feature_bias()

        # ── 安全弁3: 供給集中 ──
        self._state.supply_concentration_warning = self._check_supply_concentration()

        # ── 安全弁4: 蓄積偏り ──
        self._state.accumulation_bias_warning = self._check_accumulation_bias(now)

        # 供給強度
        self._update_supply_strength(convergence, now)

        return MetaEmotionResult(
            current_feature=feature,
            active_pattern_count=active_patterns,
            decaying_pattern_count=decaying_patterns,
            candidate_count=candidate_count,
            cognition_record_count=cognition_count,
            freshness_distribution=freshness_dist,
            convergence_level=convergence.convergence_level,
            convergence_score=convergence.convergence_score,
            candidate_convergence_warning=self._state.candidate_convergence_warning,
            feature_bias_warning=self._state.feature_bias_warning,
            supply_concentration_warning=self._state.supply_concentration_warning,
            accumulation_bias_warning=self._state.accumulation_bias_warning,
            diversity_restored=diversity_restored,
            supply_strength=self._state.supply_strength,
            cycle_count=self._state.cycle_count,
        )

    # ─── 統計ヘルパー ────────────────────────────────────────────

    def _compute_freshness_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for rec in self._state.cognition_history:
            stage = rec.freshness_stage
            dist[stage] = dist.get(stage, 0) + 1
        return dist

    # ─── 収束監視 ────────────────────────────────────────────────

    def _monitor_convergence(self, now: float) -> ConvergenceRecord:
        """変動候補の方向性の偏りを記録として残す。"""
        cfg = self._config
        candidates = self._state.current_candidates

        # 候補の多様性（delta値の分散）
        candidate_diversity = 1.0
        if len(candidates) >= 2:
            speed_vals = [c.delta_change_speed for c in candidates]
            speed_mean = sum(speed_vals) / len(speed_vals)
            speed_var = sum((v - speed_mean) ** 2 for v in speed_vals) / len(speed_vals)

            stab_vals = [c.delta_dominant_stability for c in candidates]
            stab_mean = sum(stab_vals) / len(stab_vals)
            stab_var = sum((v - stab_mean) ** 2 for v in stab_vals) / len(stab_vals)

            candidate_diversity = _clamp((speed_var + stab_var) * 10.0)

        # 特徴量の多様性
        feature_diversity = 1.0
        recent_features = self._state.feature_history[-5:] if self._state.feature_history else []
        if len(recent_features) >= 2:
            cs_vals = [f.change_speed for f in recent_features]
            cs_mean = sum(cs_vals) / len(cs_vals)
            cs_var = sum((v - cs_mean) ** 2 for v in cs_vals) / len(cs_vals)
            feature_diversity = _clamp(cs_var * 20.0)

        convergence_score = _clamp(
            (1.0 - candidate_diversity) * 0.6
            + (1.0 - feature_diversity) * 0.4
        )
        convergence_level = _convergence_from_score(convergence_score)

        record = ConvergenceRecord(
            convergence_score=convergence_score,
            convergence_level=convergence_level.value,
            candidate_diversity=candidate_diversity,
            feature_diversity=feature_diversity,
            cycle=self._state.cycle_count,
            timestamp=now,
        )

        self._state.convergence_records.append(record)
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

        return record

    # ─── 安全弁 ──────────────────────────────────────────────────

    def _restore_candidate_diversity(self, now: float) -> bool:
        """安全弁1: 変動候補が単一方向に収束した場合、
        希薄化中の異なる方向の候補を復帰候補として再浮上させる。
        """
        cfg = self._config
        restored = False

        # 減衰中の持続パターンから異なる方向の候補を補充
        for pat in self._state.sustained_patterns:
            if pat.status == RecordStatus.DECAYING.value:
                pat.freshness = _clamp(pat.freshness + cfg.diversity_recovery_amount)
                pat.freshness_stage = _stage_from_freshness(pat.freshness).value
                if pat.freshness >= 0.4:
                    pat.status = RecordStatus.ACTIVE.value
                    self._state.total_records_recovered += 1
                restored = True

        return restored

    def _check_feature_bias(self) -> bool:
        """安全弁2: 推移パターンの検出が特定の特徴量に偏った場合を検出。"""
        recent = self._state.feature_history[-5:] if self._state.feature_history else []
        if len(recent) < 3:
            return False

        # 変化速度がすべて同じ方向（差が極小）なら偏りあり
        speeds = [f.change_speed for f in recent]
        if max(speeds) - min(speeds) < 0.01 and len(speeds) >= 3:
            # 他の特徴量の参照を補充
            for f in recent:
                if f.freshness < 0.8:
                    f.freshness = _clamp(f.freshness + 0.05)
                    f.freshness_stage = _stage_from_freshness(f.freshness).value
            return True

        return False

    def _check_supply_concentration(self) -> bool:
        """安全弁3: 供給が特定方向の変動候補に集中した場合を検出。"""
        history = self._state.supply_history[-5:] if self._state.supply_history else []
        if len(history) < 3:
            return False

        # 供給された候補のdelta方向の分散
        delta_speeds = [h.get("delta_change_speed", 0.0) for h in history]
        if not delta_speeds:
            return False
        mean_speed = sum(delta_speeds) / len(delta_speeds)
        variance = sum((d - mean_speed) ** 2 for d in delta_speeds) / len(delta_speeds)

        if variance < 0.001:
            return True

        return False

    def _check_accumulation_bias(self, now: float) -> bool:
        """安全弁4: 蓄積記録が特定パターンの認知に偏った場合、
        異なるパターンの過去記録を再浮上させる。
        """
        cfg = self._config
        active_records = [
            r for r in self._state.cognition_history
            if r.status == RecordStatus.ACTIVE.value
        ]
        if len(active_records) < 3:
            return False

        # パターンIDの分布
        pattern_counts: dict[str, int] = {}
        for rec in active_records:
            for pid in rec.pattern_ids:
                pattern_counts[pid] = pattern_counts.get(pid, 0) + 1

        if not pattern_counts:
            return False

        total = sum(pattern_counts.values())
        max_count = max(pattern_counts.values())

        if total > 0 and max_count / total > 0.8 and total >= 3:
            # 異なるパターンの過去記録を再浮上
            for rec in self._state.cognition_history:
                if rec.status == RecordStatus.DECAYING.value:
                    has_different = any(
                        pid not in pattern_counts or pattern_counts[pid] < max_count
                        for pid in rec.pattern_ids
                    )
                    if has_different:
                        rec.freshness = _clamp(rec.freshness + cfg.diversity_recovery_amount)
                        rec.freshness_stage = _stage_from_freshness(rec.freshness).value
                        if rec.freshness >= 0.4:
                            rec.status = RecordStatus.ACTIVE.value
                            self._state.total_records_recovered += 1
            return True

        return False

    def _update_supply_strength(
        self, convergence: ConvergenceRecord, now: float,
    ) -> None:
        """供給強度を更新する。"""
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

def get_meta_emotion_summary(state: MetaEmotionState) -> str:
    """enrichment用の要約テキスト。

    評価判定（「この感情を変えるべき」等）を含まない。
    """
    if state.cycle_count == 0 and not state.feature_history:
        return "メタ感情認知: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    active = sum(
        1 for p in state.sustained_patterns
        if p.status == RecordStatus.ACTIVE.value
    )
    decaying = sum(
        1 for p in state.sustained_patterns
        if p.status == RecordStatus.DECAYING.value
    )

    if active:
        parts.append(f"持続={active}")
    if decaying:
        parts.append(f"減衰中={decaying}")

    cand_count = len(state.current_candidates)
    if cand_count:
        parts.append(f"候補={cand_count}")

    if state.feature_history:
        latest = state.feature_history[-1]
        parts.append(f"速度={latest.change_speed:.3f}")
        parts.append(f"安定={latest.dominant_stability:.2f}")

    if state.candidate_convergence_warning:
        parts.append("候補収束")
    if state.feature_bias_warning:
        parts.append("特徴偏り")
    if state.supply_concentration_warning:
        parts.append("供給集中")
    if state.accumulation_bias_warning:
        parts.append("蓄積偏り")

    return " ".join(parts) if parts else "メタ感情認知: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_meta_emotion_processor(
    config: Optional[MetaEmotionConfig] = None,
) -> MetaEmotionProcessor:
    """MetaEmotionProcessor のファクトリ関数。"""
    return MetaEmotionProcessor(config=config)
