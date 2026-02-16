"""
psyche/value_orientation_validation.py - 価値方向性の実運用検証

実運用中の価値方向性の変化を継続観測し、内部変化と出力傾向の接続状態を
検証情報として形成する。

設計原則 (design_value_orientation_operation_validation.md 準拠):
- 価値方向性そのものを変更しない
- 検証結果をそのまま判断確定へ接続しない
- 評価軸を単一化して出力傾向を矯正しない
- 観測結果による直接介入を行わない
- 検証出力は報告情報形式に限定し、制御命令形式で渡さない
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

class ObservationSourceType(Enum):
    """観測対象の断面種別（8値）。"""
    VALUE_ORIENTATION = "value_orientation"
    ACTION_CANDIDATES = "action_candidates"
    SELECTION_HISTORY = "selection_history"
    CONTEXT = "context"
    EMOTION_TRANSITION = "emotion_transition"
    MEMORY_REFERENCE = "memory_reference"
    RESPONSIBILITY = "responsibility"
    TIME_ELAPSED = "time_elapsed"


class ObservationFreshness(Enum):
    """観測の鮮度。"""
    FRESH = "fresh"
    RECENT = "recent"
    AGING = "aging"
    STALE = "stale"
    FADED = "faded"


class DifferentialType(Enum):
    """差分種別。"""
    INCONSISTENCY = "inconsistency"
    CONVERGENCE = "convergence"
    RE_DIVERGENCE = "re_divergence"


class ValidationStatus(Enum):
    """検証記述の状態。"""
    ACTIVE = "active"
    PENDING = "pending"
    DILUTED = "diluted"
    EXPIRED = "expired"


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _freshness_from_age(age_seconds: float) -> ObservationFreshness:
    """経過時間から鮮度を返す。"""
    if age_seconds < 30.0:
        return ObservationFreshness.FRESH
    elif age_seconds < 120.0:
        return ObservationFreshness.RECENT
    elif age_seconds < 300.0:
        return ObservationFreshness.AGING
    elif age_seconds < 600.0:
        return ObservationFreshness.STALE
    else:
        return ObservationFreshness.FADED


def _freshness_weight(freshness: ObservationFreshness) -> float:
    """鮮度に応じた重み（0.0〜1.0）。"""
    return {
        ObservationFreshness.FRESH: 1.0,
        ObservationFreshness.RECENT: 0.8,
        ObservationFreshness.AGING: 0.5,
        ObservationFreshness.STALE: 0.2,
        ObservationFreshness.FADED: 0.05,
    }.get(freshness, 0.0)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ObservationRecord:
    """観測記録の個別単位。"""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_type: str = ""
    timestamp: float = field(default_factory=time.time)
    dimensions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    freshness: str = ObservationFreshness.FRESH.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "dimensions": dict(self.dimensions),
            "metadata": dict(self.metadata),
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationRecord":
        return cls(
            record_id=data.get("record_id", uuid.uuid4().hex[:12]),
            source_type=data.get("source_type", ""),
            timestamp=data.get("timestamp", time.time()),
            dimensions=dict(data.get("dimensions", {})),
            metadata=dict(data.get("metadata", {})),
            freshness=data.get("freshness", ObservationFreshness.FRESH.value),
        )


@dataclass
class ValidationDescriptionUnit:
    """検証記述単位（元観測への可逆参照を保持）。"""
    unit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_record_ids: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    normalized_values: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: str = ValidationStatus.ACTIVE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_record_ids": list(self.source_record_ids),
            "source_types": list(self.source_types),
            "normalized_values": dict(self.normalized_values),
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationDescriptionUnit":
        return cls(
            unit_id=data.get("unit_id", uuid.uuid4().hex[:12]),
            source_record_ids=list(data.get("source_record_ids", [])),
            source_types=list(data.get("source_types", [])),
            normalized_values=dict(data.get("normalized_values", {})),
            timestamp=data.get("timestamp", time.time()),
            status=data.get("status", ValidationStatus.ACTIVE.value),
        )


@dataclass
class DifferentialEntry:
    """差分履歴の個別エントリ。"""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    diff_type: str = DifferentialType.INCONSISTENCY.value
    source_unit_ids: list[str] = field(default_factory=list)
    dimension: str = ""
    value_before: float = 0.0
    value_after: float = 0.0
    delta: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "diff_type": self.diff_type,
            "source_unit_ids": list(self.source_unit_ids),
            "dimension": self.dimension,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "delta": self.delta,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DifferentialEntry":
        return cls(
            entry_id=data.get("entry_id", uuid.uuid4().hex[:12]),
            diff_type=data.get("diff_type", DifferentialType.INCONSISTENCY.value),
            source_unit_ids=list(data.get("source_unit_ids", [])),
            dimension=data.get("dimension", ""),
            value_before=data.get("value_before", 0.0),
            value_after=data.get("value_after", 0.0),
            delta=data.get("delta", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class TimeSeriesEntry:
    """時系列索引エントリ。"""
    tick: int = 0
    timestamp: float = field(default_factory=time.time)
    unit_id: str = ""
    observation_type: str = "single"  # "single" or "continuous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "unit_id": self.unit_id,
            "observation_type": self.observation_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSeriesEntry":
        return cls(
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            unit_id=data.get("unit_id", ""),
            observation_type=data.get("observation_type", "single"),
        )


# =============================================================================
# Observation Inputs (8 cross-sections)
# =============================================================================

@dataclass
class ValidationInputs:
    """8断面の入力データ。"""
    # 1. 価値方向性断面
    orientation_dimensions: dict[str, float] = field(default_factory=dict)
    orientation_confidences: dict[str, float] = field(default_factory=dict)
    orientation_update_count: int = 0

    # 2. 行動候補断面
    candidate_count: int = 0
    top_candidate_label: str = ""
    top_candidate_score: float = 0.0
    candidate_diversity: float = 0.0

    # 3. 選択履歴断面
    recent_selections: list[str] = field(default_factory=list)
    selection_consistency: float = 0.0

    # 4. 文脈断面
    context_pace: float = 0.0
    context_density: float = 0.0
    context_continuity: float = 0.0

    # 5. 感情推移断面
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    emotions: dict[str, float] = field(default_factory=dict)

    # 6. 記憶参照断面
    recalled_count: int = 0
    has_bindings: bool = False
    episode_count: int = 0

    # 7. 責任断面
    caution_bias: float = 0.0
    empathy_bias: float = 0.0
    responsibility_weight: float = 0.0

    # 8. 時間経過断面
    tick_count: int = 0
    elapsed_since_last: float = 0.0


# =============================================================================
# Validation State
# =============================================================================

@dataclass
class ValidationState:
    """検証システムの内部状態。"""
    # 観測記録集合（追記可能な可変構造）
    observation_records: list[ObservationRecord] = field(default_factory=list)

    # 検証記述単位（可逆参照保持）
    description_units: list[ValidationDescriptionUnit] = field(default_factory=list)

    # 時系列索引
    time_series_index: list[TimeSeriesEntry] = field(default_factory=list)

    # 差分履歴（収束・不一致を並立保持）
    differential_history: list[DifferentialEntry] = field(default_factory=list)

    # 再分岐履歴（差分履歴と並立保持）
    re_divergence_history: list[DifferentialEntry] = field(default_factory=list)

    # 観測鮮度状態
    freshness_map: dict[str, str] = field(default_factory=dict)

    # 保留観測履歴（観測欠落時の補完経路）
    pending_observations: list[ObservationRecord] = field(default_factory=list)

    # 希薄化履歴
    dilution_history: list[dict[str, Any]] = field(default_factory=list)

    # 処理カウンタ
    cycle_count: int = 0
    total_observations: int = 0
    last_observation_time: float = 0.0

    # 安全弁状態
    convergence_warning: bool = False
    gap_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_records": [r.to_dict() for r in self.observation_records],
            "description_units": [u.to_dict() for u in self.description_units],
            "time_series_index": [e.to_dict() for e in self.time_series_index],
            "differential_history": [d.to_dict() for d in self.differential_history],
            "re_divergence_history": [d.to_dict() for d in self.re_divergence_history],
            "freshness_map": dict(self.freshness_map),
            "pending_observations": [r.to_dict() for r in self.pending_observations],
            "dilution_history": list(self.dilution_history),
            "cycle_count": self.cycle_count,
            "total_observations": self.total_observations,
            "last_observation_time": self.last_observation_time,
            "convergence_warning": self.convergence_warning,
            "gap_warning": self.gap_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationState":
        return cls(
            observation_records=[
                ObservationRecord.from_dict(r)
                for r in data.get("observation_records", [])
            ],
            description_units=[
                ValidationDescriptionUnit.from_dict(u)
                for u in data.get("description_units", [])
            ],
            time_series_index=[
                TimeSeriesEntry.from_dict(e)
                for e in data.get("time_series_index", [])
            ],
            differential_history=[
                DifferentialEntry.from_dict(d)
                for d in data.get("differential_history", [])
            ],
            re_divergence_history=[
                DifferentialEntry.from_dict(d)
                for d in data.get("re_divergence_history", [])
            ],
            freshness_map=dict(data.get("freshness_map", {})),
            pending_observations=[
                ObservationRecord.from_dict(r)
                for r in data.get("pending_observations", [])
            ],
            dilution_history=list(data.get("dilution_history", [])),
            cycle_count=data.get("cycle_count", 0),
            total_observations=data.get("total_observations", 0),
            last_observation_time=data.get("last_observation_time", 0.0),
            convergence_warning=data.get("convergence_warning", False),
            gap_warning=data.get("gap_warning", False),
        )


# =============================================================================
# Validation Result (output)
# =============================================================================

@dataclass
class ValidationResult:
    """検証出力（報告情報形式のみ）。"""
    # 観測された差分
    differentials: list[DifferentialEntry] = field(default_factory=list)

    # 検証記述単位の要約
    active_units: int = 0
    pending_units: int = 0

    # 断面カバレッジ
    covered_sources: list[str] = field(default_factory=list)
    uncovered_sources: list[str] = field(default_factory=list)

    # 安全弁トリガ状態
    convergence_warning: bool = False
    gap_warning: bool = False
    alternative_supplemented: bool = False
    pending_reactivated: bool = False

    # 集約情報
    short_term_trend: dict[str, float] = field(default_factory=dict)
    long_term_trend: dict[str, float] = field(default_factory=dict)

    # サイクル
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "differentials": [d.to_dict() for d in self.differentials],
            "active_units": self.active_units,
            "pending_units": self.pending_units,
            "covered_sources": list(self.covered_sources),
            "uncovered_sources": list(self.uncovered_sources),
            "convergence_warning": self.convergence_warning,
            "gap_warning": self.gap_warning,
            "alternative_supplemented": self.alternative_supplemented,
            "pending_reactivated": self.pending_reactivated,
            "short_term_trend": dict(self.short_term_trend),
            "long_term_trend": dict(self.long_term_trend),
            "cycle_count": self.cycle_count,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """検証システムの設定。"""
    # 最大保持観測記録数
    max_observation_records: int = 200

    # 最大保持検証記述単位数
    max_description_units: int = 100

    # 最大差分履歴数
    max_differential_history: int = 150

    # 最大再分岐履歴数
    max_re_divergence_history: int = 100

    # 最大時系列索引数
    max_time_series_entries: int = 200

    # 最大保留観測数
    max_pending_observations: int = 50

    # 最大希薄化履歴数
    max_dilution_history: int = 50

    # 鮮度更新間隔（秒）
    freshness_update_interval: float = 60.0

    # 観測欠落とみなす断面連続不在回数
    gap_threshold: int = 5

    # 収束警告の閾値（差分の単一方向割合）
    convergence_threshold: float = 0.8

    # 短期トレンド計算の直近観測数
    short_term_window: int = 10

    # 長期トレンド計算の直近観測数
    long_term_window: int = 50

    # 希薄化係数（鮮度が STALE 以下の記録に適用）
    dilution_factor: float = 0.9


# =============================================================================
# Processor (6-stage pipeline)
# =============================================================================

class ValueOrientationValidator:
    """
    価値方向性の実運用検証プロセッサ。

    6段パイプライン:
    1. 観測対象抽出
    2. 観測単位正規化
    3. 時系列整列
    4. 差分記述化
    5. 検証出力化
    6. 受け渡し準備
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self._config = config or ValidationConfig()
        self._state = ValidationState()
        # 断面別の連続欠落カウンタ
        self._gap_counters: dict[str, int] = {
            s.value: 0 for s in ObservationSourceType
        }

    @property
    def state(self) -> ValidationState:
        return self._state

    @state.setter
    def state(self, value: ValidationState) -> None:
        self._state = value

    def process(self, inputs: ValidationInputs) -> ValidationResult:
        """
        6段パイプラインを実行し、検証結果を返す。

        検証結果は報告情報形式のみであり、
        判断・評価・行動決定を直接起動しない。
        """
        self._state.cycle_count += 1
        now = time.time()

        # Stage 1: 観測対象抽出
        records = self._extract_observations(inputs, now)

        # Stage 2: 観測単位正規化
        units = self._normalize_observations(records, now)

        # Stage 3: 時系列整列
        self._align_time_series(units, inputs.tick_count, now)

        # Stage 4: 差分記述化
        new_diffs = self._compute_differentials(units, now)

        # Stage 5: 検証出力化
        result = self._build_validation_output(new_diffs, now)

        # Stage 6: 受け渡し準備（安全弁チェック + クリーンアップ）
        self._prepare_handoff(result, now)

        self._state.last_observation_time = now
        return result

    # ─── Stage 1: 観測対象抽出 ───────────────────────────────────

    def _extract_observations(
        self,
        inputs: ValidationInputs,
        now: float,
    ) -> list[ObservationRecord]:
        """8断面から観測記録を抽出する。"""
        records: list[ObservationRecord] = []
        covered: set[str] = set()

        # 1. 価値方向性断面
        if inputs.orientation_dimensions:
            rec = ObservationRecord(
                source_type=ObservationSourceType.VALUE_ORIENTATION.value,
                timestamp=now,
                dimensions=dict(inputs.orientation_dimensions),
                metadata={
                    "confidences": dict(inputs.orientation_confidences),
                    "update_count": inputs.orientation_update_count,
                },
            )
            records.append(rec)
            covered.add(ObservationSourceType.VALUE_ORIENTATION.value)

        # 2. 行動候補断面
        if inputs.candidate_count > 0:
            rec = ObservationRecord(
                source_type=ObservationSourceType.ACTION_CANDIDATES.value,
                timestamp=now,
                dimensions={
                    "count": float(inputs.candidate_count),
                    "top_score": inputs.top_candidate_score,
                    "diversity": inputs.candidate_diversity,
                },
                metadata={"top_label": inputs.top_candidate_label},
            )
            records.append(rec)
            covered.add(ObservationSourceType.ACTION_CANDIDATES.value)

        # 3. 選択履歴断面
        if inputs.recent_selections:
            rec = ObservationRecord(
                source_type=ObservationSourceType.SELECTION_HISTORY.value,
                timestamp=now,
                dimensions={
                    "consistency": inputs.selection_consistency,
                    "count": float(len(inputs.recent_selections)),
                },
                metadata={"recent": inputs.recent_selections[:10]},
            )
            records.append(rec)
            covered.add(ObservationSourceType.SELECTION_HISTORY.value)

        # 4. 文脈断面
        ctx_active = (
            abs(inputs.context_pace) > 0.01
            or abs(inputs.context_density) > 0.01
            or abs(inputs.context_continuity) > 0.01
        )
        if ctx_active:
            rec = ObservationRecord(
                source_type=ObservationSourceType.CONTEXT.value,
                timestamp=now,
                dimensions={
                    "pace": inputs.context_pace,
                    "density": inputs.context_density,
                    "continuity": inputs.context_continuity,
                },
            )
            records.append(rec)
            covered.add(ObservationSourceType.CONTEXT.value)

        # 5. 感情推移断面
        if inputs.emotions:
            rec = ObservationRecord(
                source_type=ObservationSourceType.EMOTION_TRANSITION.value,
                timestamp=now,
                dimensions={
                    "valence": inputs.emotion_valence,
                    "arousal": inputs.emotion_arousal,
                    **{f"emo_{k}": v for k, v in inputs.emotions.items()},
                },
            )
            records.append(rec)
            covered.add(ObservationSourceType.EMOTION_TRANSITION.value)

        # 6. 記憶参照断面
        if inputs.recalled_count > 0 or inputs.has_bindings or inputs.episode_count > 0:
            rec = ObservationRecord(
                source_type=ObservationSourceType.MEMORY_REFERENCE.value,
                timestamp=now,
                dimensions={
                    "recalled": float(inputs.recalled_count),
                    "bindings": 1.0 if inputs.has_bindings else 0.0,
                    "episodes": float(inputs.episode_count),
                },
            )
            records.append(rec)
            covered.add(ObservationSourceType.MEMORY_REFERENCE.value)

        # 7. 責任断面
        resp_active = (
            abs(inputs.caution_bias) > 0.01
            or abs(inputs.empathy_bias) > 0.01
            or abs(inputs.responsibility_weight) > 0.01
        )
        if resp_active:
            rec = ObservationRecord(
                source_type=ObservationSourceType.RESPONSIBILITY.value,
                timestamp=now,
                dimensions={
                    "caution": inputs.caution_bias,
                    "empathy": inputs.empathy_bias,
                    "weight": inputs.responsibility_weight,
                },
            )
            records.append(rec)
            covered.add(ObservationSourceType.RESPONSIBILITY.value)

        # 8. 時間経過断面（常に記録）
        rec = ObservationRecord(
            source_type=ObservationSourceType.TIME_ELAPSED.value,
            timestamp=now,
            dimensions={
                "tick": float(inputs.tick_count),
                "elapsed": inputs.elapsed_since_last,
            },
        )
        records.append(rec)
        covered.add(ObservationSourceType.TIME_ELAPSED.value)

        # 欠落カウンタ更新
        all_sources = {s.value for s in ObservationSourceType}
        for src in all_sources:
            if src in covered:
                self._gap_counters[src] = 0
            else:
                self._gap_counters[src] = self._gap_counters.get(src, 0) + 1

        # 観測記録を状態に追加
        self._state.observation_records.extend(records)
        self._state.total_observations += len(records)

        # 上限トリミング
        cfg = self._config
        if len(self._state.observation_records) > cfg.max_observation_records:
            overflow = len(self._state.observation_records) - cfg.max_observation_records
            removed = self._state.observation_records[:overflow]
            self._state.observation_records = self._state.observation_records[overflow:]
            # 溢れた記録を保留へ移動（欠測を単純消去しない）
            for r in removed:
                r.freshness = ObservationFreshness.FADED.value
            self._state.pending_observations.extend(removed)
            if len(self._state.pending_observations) > cfg.max_pending_observations:
                self._state.pending_observations = self._state.pending_observations[
                    -cfg.max_pending_observations:
                ]

        return records

    # ─── Stage 2: 観測単位正規化 ─────────────────────────────────

    def _normalize_observations(
        self,
        records: list[ObservationRecord],
        now: float,
    ) -> list[ValidationDescriptionUnit]:
        """異なる断面から得られる情報を共通検証記述へ統一する。"""
        units: list[ValidationDescriptionUnit] = []

        for record in records:
            # 各断面のdimensionsを共通形式 [0.0〜1.0] に正規化
            normalized: dict[str, float] = {}

            src = record.source_type
            dims = record.dimensions

            if src == ObservationSourceType.VALUE_ORIENTATION.value:
                # dim values are [-1.0, 1.0] → normalize to [0.0, 1.0]
                for k, v in dims.items():
                    normalized[f"vo_{k}"] = _clamp((v + 1.0) / 2.0)

            elif src == ObservationSourceType.ACTION_CANDIDATES.value:
                normalized["ac_count"] = _clamp(dims.get("count", 0) / 20.0)
                normalized["ac_top_score"] = _clamp(dims.get("top_score", 0))
                normalized["ac_diversity"] = _clamp(dims.get("diversity", 0))

            elif src == ObservationSourceType.SELECTION_HISTORY.value:
                normalized["sh_consistency"] = _clamp(dims.get("consistency", 0))
                normalized["sh_count"] = _clamp(dims.get("count", 0) / 20.0)

            elif src == ObservationSourceType.CONTEXT.value:
                normalized["ctx_pace"] = _clamp(dims.get("pace", 0))
                normalized["ctx_density"] = _clamp(dims.get("density", 0))
                normalized["ctx_continuity"] = _clamp(dims.get("continuity", 0))

            elif src == ObservationSourceType.EMOTION_TRANSITION.value:
                normalized["et_valence"] = _clamp(
                    (dims.get("valence", 0) + 1.0) / 2.0
                )
                normalized["et_arousal"] = _clamp(dims.get("arousal", 0))
                for k, v in dims.items():
                    if k.startswith("emo_"):
                        normalized[f"et_{k}"] = _clamp(v)

            elif src == ObservationSourceType.MEMORY_REFERENCE.value:
                normalized["mr_recalled"] = _clamp(dims.get("recalled", 0) / 10.0)
                normalized["mr_bindings"] = _clamp(dims.get("bindings", 0))
                normalized["mr_episodes"] = _clamp(dims.get("episodes", 0) / 10.0)

            elif src == ObservationSourceType.RESPONSIBILITY.value:
                normalized["rsp_caution"] = _clamp(dims.get("caution", 0))
                normalized["rsp_empathy"] = _clamp(dims.get("empathy", 0))
                normalized["rsp_weight"] = _clamp(dims.get("weight", 0))

            elif src == ObservationSourceType.TIME_ELAPSED.value:
                normalized["te_tick"] = _clamp(dims.get("tick", 0) / 1000.0)
                normalized["te_elapsed"] = _clamp(dims.get("elapsed", 0) / 60.0)

            unit = ValidationDescriptionUnit(
                source_record_ids=[record.record_id],
                source_types=[record.source_type],
                normalized_values=normalized,
                timestamp=now,
            )
            units.append(unit)

        # 状態に追加
        self._state.description_units.extend(units)
        cfg = self._config
        if len(self._state.description_units) > cfg.max_description_units:
            overflow = len(self._state.description_units) - cfg.max_description_units
            # 溢れた単位を希薄化状態にして希薄化履歴へ
            for u in self._state.description_units[:overflow]:
                self._state.dilution_history.append({
                    "unit_id": u.unit_id,
                    "source_types": u.source_types,
                    "timestamp": u.timestamp,
                    "diluted_at": now,
                })
            self._state.description_units = self._state.description_units[overflow:]
            if len(self._state.dilution_history) > cfg.max_dilution_history:
                self._state.dilution_history = self._state.dilution_history[
                    -cfg.max_dilution_history:
                ]

        return units

    # ─── Stage 3: 時系列整列 ─────────────────────────────────────

    def _align_time_series(
        self,
        units: list[ValidationDescriptionUnit],
        tick_count: int,
        now: float,
    ) -> None:
        """単回観測と継続観測を分離し、時系列索引に追加する。"""
        for unit in units:
            # 同一断面の過去エントリがあれば continuous、なければ single
            src_types = set(unit.source_types)
            past_entries = [
                e for e in self._state.time_series_index
                if e.unit_id != unit.unit_id
            ]
            has_prior = any(
                e for e in past_entries
                if any(
                    s in src_types
                    for s in self._get_unit_source_types(e.unit_id)
                )
            )

            entry = TimeSeriesEntry(
                tick=tick_count,
                timestamp=now,
                unit_id=unit.unit_id,
                observation_type="continuous" if has_prior else "single",
            )
            self._state.time_series_index.append(entry)

        # 鮮度更新
        for rec in self._state.observation_records:
            age = now - rec.timestamp
            rec.freshness = _freshness_from_age(age).value
            self._state.freshness_map[rec.record_id] = rec.freshness

        # トリミング
        cfg = self._config
        if len(self._state.time_series_index) > cfg.max_time_series_entries:
            self._state.time_series_index = self._state.time_series_index[
                -cfg.max_time_series_entries:
            ]

    def _get_unit_source_types(self, unit_id: str) -> list[str]:
        """unit_idから断面種別リストを取得する。"""
        for u in self._state.description_units:
            if u.unit_id == unit_id:
                return u.source_types
        return []

    # ─── Stage 4: 差分記述化 ─────────────────────────────────────

    def _compute_differentials(
        self,
        new_units: list[ValidationDescriptionUnit],
        now: float,
    ) -> list[DifferentialEntry]:
        """断面間の不一致・収束・再分岐を並立記録する。"""
        new_diffs: list[DifferentialEntry] = []
        cfg = self._config

        for unit in new_units:
            # 同一断面の直近ユニットと比較
            prior = self._find_prior_unit(unit)
            if prior is None:
                continue

            # 正規化値の各次元で差分を計算
            all_dims = set(unit.normalized_values.keys()) | set(prior.normalized_values.keys())
            for dim in all_dims:
                val_now = unit.normalized_values.get(dim, 0.0)
                val_prev = prior.normalized_values.get(dim, 0.0)
                delta = val_now - val_prev

                if abs(delta) < 0.01:
                    continue

                # 差分種別を判定
                diff_type = self._classify_differential(dim, delta, now)

                entry = DifferentialEntry(
                    diff_type=diff_type.value,
                    source_unit_ids=[prior.unit_id, unit.unit_id],
                    dimension=dim,
                    value_before=val_prev,
                    value_after=val_now,
                    delta=delta,
                    timestamp=now,
                )
                new_diffs.append(entry)

                # 再分岐の場合は再分岐履歴にも追加
                if diff_type == DifferentialType.RE_DIVERGENCE:
                    self._state.re_divergence_history.append(entry)
                    if len(self._state.re_divergence_history) > cfg.max_re_divergence_history:
                        self._state.re_divergence_history = (
                            self._state.re_divergence_history[-cfg.max_re_divergence_history:]
                        )

        # 差分履歴に追加
        self._state.differential_history.extend(new_diffs)
        if len(self._state.differential_history) > cfg.max_differential_history:
            self._state.differential_history = self._state.differential_history[
                -cfg.max_differential_history:
            ]

        return new_diffs

    def _find_prior_unit(
        self, current: ValidationDescriptionUnit
    ) -> Optional[ValidationDescriptionUnit]:
        """同一断面の直前の検証記述単位を探す。"""
        src_types = set(current.source_types)
        # 逆順で探索（新しい方から）
        for unit in reversed(self._state.description_units):
            if unit.unit_id == current.unit_id:
                continue
            if set(unit.source_types) & src_types:
                return unit
        return None

    def _classify_differential(
        self, dimension: str, delta: float, now: float,
    ) -> DifferentialType:
        """差分の種別を判定する。"""
        # 直近の差分履歴で同じ次元の傾向を確認
        recent_diffs = [
            d for d in self._state.differential_history[-20:]
            if d.dimension == dimension
        ]

        if not recent_diffs:
            return DifferentialType.INCONSISTENCY

        # 直近の差分方向を確認
        last_delta = recent_diffs[-1].delta
        if last_delta * delta > 0:
            # 同方向 → 収束
            return DifferentialType.CONVERGENCE
        else:
            # 逆方向 → 再分岐
            return DifferentialType.RE_DIVERGENCE

    # ─── Stage 5: 検証出力化 ─────────────────────────────────────

    def _build_validation_output(
        self,
        new_diffs: list[DifferentialEntry],
        now: float,
    ) -> ValidationResult:
        """観測情報としてのみ出力する。判断・評価・行動決定を起動しない。"""
        cfg = self._config

        # 断面カバレッジ
        all_sources = {s.value for s in ObservationSourceType}
        covered = set()
        for rec in self._state.observation_records[-50:]:
            age = now - rec.timestamp
            if age < 300.0:  # 5分以内
                covered.add(rec.source_type)
        uncovered = all_sources - covered

        # アクティブ / 保留ユニット数
        active_count = sum(
            1 for u in self._state.description_units
            if u.status == ValidationStatus.ACTIVE.value
        )
        pending_count = sum(
            1 for u in self._state.description_units
            if u.status == ValidationStatus.PENDING.value
        )

        # 短期トレンド（直近N観測の平均差分）
        short_term = self._compute_trend(cfg.short_term_window, now)

        # 長期トレンド（直近M観測の平均差分）
        long_term = self._compute_trend(cfg.long_term_window, now)

        return ValidationResult(
            differentials=new_diffs,
            active_units=active_count,
            pending_units=pending_count,
            covered_sources=sorted(covered),
            uncovered_sources=sorted(uncovered),
            convergence_warning=self._state.convergence_warning,
            gap_warning=self._state.gap_warning,
            short_term_trend=short_term,
            long_term_trend=long_term,
            cycle_count=self._state.cycle_count,
        )

    def _compute_trend(self, window: int, now: float) -> dict[str, float]:
        """直近N件の差分から次元ごとの平均変化を計算する。"""
        recent = self._state.differential_history[-window:]
        if not recent:
            return {}

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for d in recent:
            dim = d.dimension
            # 鮮度重み付き
            age = now - d.timestamp
            weight = _freshness_weight(_freshness_from_age(age))
            totals[dim] = totals.get(dim, 0.0) + d.delta * weight
            counts[dim] = counts.get(dim, 0) + 1

        return {
            dim: round(totals[dim] / counts[dim], 4)
            for dim in totals
            if counts[dim] > 0
        }

    # ─── Stage 6: 受け渡し準備 ───────────────────────────────────

    def _prepare_handoff(
        self,
        result: ValidationResult,
        now: float,
    ) -> None:
        """安全弁チェック + 循環参照防止 + 自己強化ループ防止。"""
        cfg = self._config

        # ── 安全弁1: 収束警告 ──
        # 差分の単一方向割合が閾値を超えたら代替系列を補充
        result.alternative_supplemented = False
        if len(self._state.differential_history) >= 10:
            recent_diffs = self._state.differential_history[-20:]
            positive_count = sum(1 for d in recent_diffs if d.delta > 0)
            negative_count = sum(1 for d in recent_diffs if d.delta < 0)
            total = positive_count + negative_count
            if total > 0:
                dominant_ratio = max(positive_count, negative_count) / total
                if dominant_ratio >= cfg.convergence_threshold:
                    self._state.convergence_warning = True
                    # 代替系列の補充: 保留観測から未参照のものを再投入
                    supplemented = self._supplement_alternative_series(now)
                    result.alternative_supplemented = supplemented
                    result.convergence_warning = True
                else:
                    self._state.convergence_warning = False
                    result.convergence_warning = False

        # ── 安全弁2: 観測欠落 ──
        # 連続欠落が閾値を超えた断面があれば保留観測を再評価
        result.pending_reactivated = False
        gap_detected = any(
            count >= cfg.gap_threshold
            for count in self._gap_counters.values()
        )
        if gap_detected:
            self._state.gap_warning = True
            reactivated = self._reactivate_pending(now)
            result.pending_reactivated = reactivated
            result.gap_warning = True
        else:
            self._state.gap_warning = False
            result.gap_warning = False

        # ── 固定化防止: 断面横断の混在参照維持 ──
        # 特定断面だけが検証結果を支配しないよう確認
        self._ensure_cross_section_diversity(now)

        # ── 希薄化処理 ──
        self._apply_dilution(now)

    def _supplement_alternative_series(self, now: float) -> bool:
        """保留観測から代替系列を補充し、複線記述へ復帰する。"""
        if not self._state.pending_observations:
            return False

        supplemented = False
        # 直近の差分で使われていない断面の保留観測を再投入
        recent_sources = set()
        for d in self._state.differential_history[-10:]:
            for uid in d.source_unit_ids:
                for u in self._state.description_units:
                    if u.unit_id == uid:
                        recent_sources.update(u.source_types)

        for pending in list(self._state.pending_observations):
            if pending.source_type not in recent_sources:
                # 再投入
                pending.freshness = ObservationFreshness.AGING.value
                pending.timestamp = now
                self._state.observation_records.append(pending)
                self._state.pending_observations.remove(pending)
                supplemented = True
                if len(self._state.observation_records) >= self._config.max_observation_records:
                    break

        return supplemented

    def _reactivate_pending(self, now: float) -> bool:
        """保留観測を再評価し、検証経路の停止状態を回避する。"""
        if not self._state.pending_observations:
            return False

        reactivated = False
        gap_sources = {
            src for src, count in self._gap_counters.items()
            if count >= self._config.gap_threshold
        }

        for pending in list(self._state.pending_observations):
            if pending.source_type in gap_sources:
                pending.freshness = ObservationFreshness.STALE.value
                pending.timestamp = now
                self._state.observation_records.append(pending)
                self._state.pending_observations.remove(pending)
                reactivated = True
                # 欠落カウンタをリセット
                self._gap_counters[pending.source_type] = 0

        return reactivated

    def _ensure_cross_section_diversity(self, now: float) -> None:
        """特定断面だけが検証結果を支配しないよう確認する。"""
        # 直近の差分での断面出現頻度を計算
        if len(self._state.differential_history) < 5:
            return

        recent = self._state.differential_history[-20:]
        source_counts: dict[str, int] = {}
        for d in recent:
            for uid in d.source_unit_ids:
                for u in self._state.description_units:
                    if u.unit_id == uid:
                        for st in u.source_types:
                            source_counts[st] = source_counts.get(st, 0) + 1

        if not source_counts:
            return

        total = sum(source_counts.values())
        for src, count in source_counts.items():
            if count / total > 0.6:
                # 特定断面が支配的 → 未参照の保留観測を再提示
                for pending in list(self._state.pending_observations[:3]):
                    if pending.source_type != src:
                        pending.freshness = ObservationFreshness.AGING.value
                        self._state.observation_records.append(pending)
                        self._state.pending_observations.remove(pending)
                break

    def _apply_dilution(self, now: float) -> None:
        """鮮度がSTALE以下の観測記録に希薄化を適用する。"""
        cfg = self._config
        for rec in self._state.observation_records:
            if rec.freshness in (
                ObservationFreshness.STALE.value,
                ObservationFreshness.FADED.value,
            ):
                # dimensions の値を希薄化
                for k in rec.dimensions:
                    rec.dimensions[k] *= cfg.dilution_factor

        # 対応する検証記述単位も希薄化
        stale_ids = {
            rec.record_id for rec in self._state.observation_records
            if rec.freshness in (
                ObservationFreshness.STALE.value,
                ObservationFreshness.FADED.value,
            )
        }
        for unit in self._state.description_units:
            if any(rid in stale_ids for rid in unit.source_record_ids):
                if unit.status == ValidationStatus.ACTIVE.value:
                    unit.status = ValidationStatus.DILUTED.value


# =============================================================================
# Summary (enrichment用)
# =============================================================================

def get_validation_summary(state: ValidationState) -> str:
    """検証状態の要約（enrichment用）。"""
    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")
    parts.append(f"obs={state.total_observations}")

    active = sum(
        1 for u in state.description_units
        if u.status == ValidationStatus.ACTIVE.value
    )
    if active:
        parts.append(f"active={active}")

    if state.differential_history:
        recent_diffs = state.differential_history[-10:]
        convergence = sum(
            1 for d in recent_diffs
            if d.diff_type == DifferentialType.CONVERGENCE.value
        )
        inconsistency = sum(
            1 for d in recent_diffs
            if d.diff_type == DifferentialType.INCONSISTENCY.value
        )
        re_div = sum(
            1 for d in recent_diffs
            if d.diff_type == DifferentialType.RE_DIVERGENCE.value
        )
        if convergence:
            parts.append(f"収束={convergence}")
        if inconsistency:
            parts.append(f"不一致={inconsistency}")
        if re_div:
            parts.append(f"再分岐={re_div}")

    if state.convergence_warning:
        parts.append("⚠収束偏向")
    if state.gap_warning:
        parts.append("⚠欠落")

    if state.pending_observations:
        parts.append(f"保留={len(state.pending_observations)}")

    return " ".join(parts) if parts else "検証: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_validation_processor(
    config: Optional[ValidationConfig] = None,
) -> ValueOrientationValidator:
    """ValueOrientationValidatorのファクトリ関数。"""
    return ValueOrientationValidator(config=config)
