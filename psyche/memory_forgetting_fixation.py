"""
psyche/memory_forgetting_fixation.py - 記憶の忘却と固定化

記憶系列の流動性を維持し、参照の単線化を避けるため、
忘却と固定化検知を同時に扱う構造を提供する。

設計原則 (design_memory_forgetting_and_fixation.md 準拠):
- 記憶量の削減が目的ではなく、記憶系列の流動性維持が目的
- 忘却は不可逆な一括消去ではなく段階的希薄化→不可視化
- 固定化は特定記憶の恒久優先ではなく保持傾向の観測
- 固定化検知結果を直接判断確定へ接続しない
- 記憶内容の価値判定を行わない
- 出力は段階忘却候補情報と固定化兆候情報としてのみ流す
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
    """入力参照の断面種別（8値）。"""
    REFERENCE_FREQUENCY = "reference_frequency"
    REUSE_INTERVAL = "reuse_interval"
    TIME_SERIES = "time_series"
    COMPETING_SERIES = "competing_series"
    EMOTION_LINK = "emotion_link"
    CONTEXT_LINK = "context_link"
    PROTECTION_STATE = "protection_state"
    FIXATION_SIGNS = "fixation_signs"


class ForgettingStage(Enum):
    """段階忘却の状態（可逆段階前提）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class FixationLevel(Enum):
    """固定化兆候のレベル。"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"


class SeriesStatus(Enum):
    """記憶系列の状態。"""
    ACTIVE = "active"
    PROTECTED = "protected"
    FORGETTING = "forgetting"
    FIXATING = "fixating"
    RECOVERED = "recovered"


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _stage_from_dilution(dilution: float) -> ForgettingStage:
    """希薄化度から忘却段階を返す。"""
    if dilution < 0.2:
        return ForgettingStage.ACTIVE
    elif dilution < 0.4:
        return ForgettingStage.WEAKENING
    elif dilution < 0.6:
        return ForgettingStage.FADING
    elif dilution < 0.8:
        return ForgettingStage.NEAR_INVISIBLE
    else:
        return ForgettingStage.INVISIBLE


def _fixation_from_score(score: float) -> FixationLevel:
    """固定化スコアから兆候レベルを返す。"""
    if score < 0.3:
        return FixationLevel.NONE
    elif score < 0.5:
        return FixationLevel.MILD
    elif score < 0.7:
        return FixationLevel.MODERATE
    else:
        return FixationLevel.STRONG


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MemorySeriesRecord:
    """記憶系列の追跡レコード。"""
    series_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""  # "episodic" / "binding" / "long_term"
    source_id: str = ""
    reference_count: int = 0
    last_reference_time: float = 0.0
    creation_time: float = field(default_factory=time.time)
    dilution: float = 0.0  # 0.0=active, 1.0=invisible
    forgetting_stage: str = ForgettingStage.ACTIVE.value
    fixation_score: float = 0.0
    fixation_level: str = FixationLevel.NONE.value
    is_protected: bool = False
    emotion_strength: float = 0.0
    reuse_count: int = 0
    status: str = SeriesStatus.ACTIVE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "source": self.source,
            "source_id": self.source_id,
            "reference_count": self.reference_count,
            "last_reference_time": self.last_reference_time,
            "creation_time": self.creation_time,
            "dilution": self.dilution,
            "forgetting_stage": self.forgetting_stage,
            "fixation_score": self.fixation_score,
            "fixation_level": self.fixation_level,
            "is_protected": self.is_protected,
            "emotion_strength": self.emotion_strength,
            "reuse_count": self.reuse_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorySeriesRecord":
        return cls(
            series_id=data.get("series_id", uuid.uuid4().hex[:12]),
            source=data.get("source", ""),
            source_id=data.get("source_id", ""),
            reference_count=data.get("reference_count", 0),
            last_reference_time=data.get("last_reference_time", 0.0),
            creation_time=data.get("creation_time", time.time()),
            dilution=data.get("dilution", 0.0),
            forgetting_stage=data.get("forgetting_stage", ForgettingStage.ACTIVE.value),
            fixation_score=data.get("fixation_score", 0.0),
            fixation_level=data.get("fixation_level", FixationLevel.NONE.value),
            is_protected=data.get("is_protected", False),
            emotion_strength=data.get("emotion_strength", 0.0),
            reuse_count=data.get("reuse_count", 0),
            status=data.get("status", SeriesStatus.ACTIVE.value),
        )


@dataclass
class ForgettingCandidate:
    """忘却候補（段階移行候補）。"""
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    series_id: str = ""
    current_stage: str = ForgettingStage.ACTIVE.value
    proposed_stage: str = ForgettingStage.WEAKENING.value
    dilution: float = 0.0
    time_since_reference: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "series_id": self.series_id,
            "current_stage": self.current_stage,
            "proposed_stage": self.proposed_stage,
            "dilution": self.dilution,
            "time_since_reference": self.time_since_reference,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgettingCandidate":
        return cls(
            candidate_id=data.get("candidate_id", uuid.uuid4().hex[:12]),
            series_id=data.get("series_id", ""),
            current_stage=data.get("current_stage", ForgettingStage.ACTIVE.value),
            proposed_stage=data.get("proposed_stage", ForgettingStage.WEAKENING.value),
            dilution=data.get("dilution", 0.0),
            time_since_reference=data.get("time_since_reference", 0.0),
            reason=data.get("reason", ""),
        )


@dataclass
class FixationSign:
    """固定化兆候（継続観測の重なり）。"""
    sign_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    series_id: str = ""
    score: float = 0.0
    level: str = FixationLevel.NONE.value
    observation_count: int = 0
    indicators: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign_id": self.sign_id,
            "series_id": self.series_id,
            "score": self.score,
            "level": self.level,
            "observation_count": self.observation_count,
            "indicators": list(self.indicators),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixationSign":
        return cls(
            sign_id=data.get("sign_id", uuid.uuid4().hex[:12]),
            series_id=data.get("series_id", ""),
            score=data.get("score", 0.0),
            level=data.get("level", FixationLevel.NONE.value),
            observation_count=data.get("observation_count", 0),
            indicators=list(data.get("indicators", [])),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Inputs (8 cross-sections)
# =============================================================================

@dataclass
class ForgettingFixationInputs:
    """8断面の入力データ。"""
    # 1. 記憶参照頻度断面
    episode_entries: list[dict[str, Any]] = field(default_factory=list)
    binding_entries: list[dict[str, Any]] = field(default_factory=list)
    long_term_entries: list[dict[str, Any]] = field(default_factory=list)
    action_result_entries: list[dict[str, Any]] = field(default_factory=list)

    # 2. 再利用間隔断面
    reuse_history: dict[str, int] = field(default_factory=dict)

    # 3. 時系列断面
    tick_count: int = 0
    elapsed_since_last: float = 0.0

    # 4. 競合系列断面
    active_series_count: int = 0
    dominant_series_id: str = ""

    # 5. 感情連結断面
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    binding_count: int = 0
    average_binding_freshness: float = 0.0

    # 6. 文脈連結断面
    context_continuity: float = 0.0
    context_density: float = 0.0

    # 7. 保護状態断面
    protected_ids: list[str] = field(default_factory=list)

    # 8. 固定化兆候断面（直近の観測）
    repeated_reference_ids: list[str] = field(default_factory=list)
    invisible_alternative_count: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class ForgettingFixationState:
    """忘却・固定化システムの内部状態。"""
    # 記憶系列索引
    series_index: list[MemorySeriesRecord] = field(default_factory=list)

    # 参照履歴
    reference_history: list[dict[str, Any]] = field(default_factory=list)

    # 再利用履歴
    reuse_history: dict[str, int] = field(default_factory=dict)

    # 希薄化状態（series_id → dilution value）
    dilution_map: dict[str, float] = field(default_factory=dict)

    # 固定化兆候履歴（継続観測の重なり）
    fixation_sign_history: list[FixationSign] = field(default_factory=list)

    # 代替系列履歴
    alternative_series: list[str] = field(default_factory=list)

    # 段階忘却状態
    forgetting_candidates: list[ForgettingCandidate] = field(default_factory=list)

    # 復帰候補履歴
    recovery_candidates: list[str] = field(default_factory=list)

    # カウンタ
    cycle_count: int = 0
    total_forgotten: int = 0
    total_recovered: int = 0
    total_fixation_signs: int = 0

    # 安全弁
    convergence_warning: bool = False
    overdense_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_index": [s.to_dict() for s in self.series_index],
            "reference_history": list(self.reference_history),
            "reuse_history": dict(self.reuse_history),
            "dilution_map": dict(self.dilution_map),
            "fixation_sign_history": [f.to_dict() for f in self.fixation_sign_history],
            "alternative_series": list(self.alternative_series),
            "forgetting_candidates": [c.to_dict() for c in self.forgetting_candidates],
            "recovery_candidates": list(self.recovery_candidates),
            "cycle_count": self.cycle_count,
            "total_forgotten": self.total_forgotten,
            "total_recovered": self.total_recovered,
            "total_fixation_signs": self.total_fixation_signs,
            "convergence_warning": self.convergence_warning,
            "overdense_warning": self.overdense_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgettingFixationState":
        return cls(
            series_index=[
                MemorySeriesRecord.from_dict(s)
                for s in data.get("series_index", [])
            ],
            reference_history=list(data.get("reference_history", [])),
            reuse_history=dict(data.get("reuse_history", {})),
            dilution_map=dict(data.get("dilution_map", {})),
            fixation_sign_history=[
                FixationSign.from_dict(f)
                for f in data.get("fixation_sign_history", [])
            ],
            alternative_series=list(data.get("alternative_series", [])),
            forgetting_candidates=[
                ForgettingCandidate.from_dict(c)
                for c in data.get("forgetting_candidates", [])
            ],
            recovery_candidates=list(data.get("recovery_candidates", [])),
            cycle_count=data.get("cycle_count", 0),
            total_forgotten=data.get("total_forgotten", 0),
            total_recovered=data.get("total_recovered", 0),
            total_fixation_signs=data.get("total_fixation_signs", 0),
            convergence_warning=data.get("convergence_warning", False),
            overdense_warning=data.get("overdense_warning", False),
        )


# =============================================================================
# Result
# =============================================================================

@dataclass
class ForgettingFixationResult:
    """処理結果（報告情報形式のみ）。"""
    # 忘却候補
    forgetting_candidates: list[ForgettingCandidate] = field(default_factory=list)
    newly_forgotten: int = 0
    newly_recovered: int = 0

    # 固定化兆候
    fixation_signs: list[FixationSign] = field(default_factory=list)
    newly_fixating: int = 0

    # 系列統計
    active_series: int = 0
    forgetting_series: int = 0
    invisible_series: int = 0
    protected_series: int = 0

    # 安全弁
    convergence_warning: bool = False
    overdense_warning: bool = False
    alternatives_supplemented: bool = False
    forgetting_slowed: bool = False

    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "forgetting_candidates": [c.to_dict() for c in self.forgetting_candidates],
            "newly_forgotten": self.newly_forgotten,
            "newly_recovered": self.newly_recovered,
            "fixation_signs": [f.to_dict() for f in self.fixation_signs],
            "newly_fixating": self.newly_fixating,
            "active_series": self.active_series,
            "forgetting_series": self.forgetting_series,
            "invisible_series": self.invisible_series,
            "protected_series": self.protected_series,
            "convergence_warning": self.convergence_warning,
            "overdense_warning": self.overdense_warning,
            "alternatives_supplemented": self.alternatives_supplemented,
            "forgetting_slowed": self.forgetting_slowed,
            "cycle_count": self.cycle_count,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ForgettingFixationConfig:
    """設定。"""
    # 最大記憶系列数
    max_series: int = 300

    # 最大参照履歴数
    max_reference_history: int = 200

    # 最大固定化兆候履歴数
    max_fixation_history: int = 100

    # 最大忘却候補数
    max_forgetting_candidates: int = 100

    # 最大復帰候補数
    max_recovery_candidates: int = 50

    # 最大代替系列数
    max_alternative_series: int = 50

    # 希薄化進行速度（サイクルあたり）
    dilution_rate: float = 0.02

    # 参照による希薄化回復量
    reference_recovery: float = 0.15

    # 感情結合による保護係数（0.0〜1.0）
    emotion_protection_factor: float = 0.3

    # 固定化兆候の閾値（反復参照回数）
    fixation_reference_threshold: int = 5

    # 固定化兆候確定に必要な交差断面数
    fixation_cross_section_threshold: int = 2

    # 忘却進行の過密化閾値（同時忘却候補数）
    overdense_threshold: int = 10

    # 収束警告の閾値（単一系列への忘却集中率）
    convergence_threshold: float = 0.6


# =============================================================================
# Processor (6-stage pipeline)
# =============================================================================

class MemoryForgettingFixationProcessor:
    """
    記憶の忘却と固定化プロセッサ。

    6段パイプライン:
    1. 忘却候補抽出
    2. 固定化兆候抽出
    3. 候補整列
    4. 競合保持
    5. 段階忘却情報化
    6. 受け渡し準備
    """

    def __init__(self, config: Optional[ForgettingFixationConfig] = None):
        self._config = config or ForgettingFixationConfig()
        self._state = ForgettingFixationState()

    @property
    def state(self) -> ForgettingFixationState:
        return self._state

    @state.setter
    def state(self, value: ForgettingFixationState) -> None:
        self._state = value

    def process(self, inputs: ForgettingFixationInputs) -> ForgettingFixationResult:
        """
        6段パイプラインを実行する。

        出力は段階忘却候補情報と固定化兆候情報としてのみ流し、
        判断・評価・行動決定を直接起動しない。
        """
        self._state.cycle_count += 1
        now = time.time()

        # 系列索引の更新（入力から新規系列を登録、参照を記録）
        self._update_series_index(inputs, now)

        # Stage 1: 忘却候補抽出
        forget_candidates = self._extract_forgetting_candidates(inputs, now)

        # Stage 2: 固定化兆候抽出
        fix_signs = self._extract_fixation_signs(inputs, now)

        # Stage 3: 候補整列
        self._align_candidates(forget_candidates, fix_signs)

        # Stage 4: 競合保持
        self._retain_competition(now)

        # Stage 5: 段階忘却情報化
        newly_forgotten, newly_recovered = self._apply_staged_forgetting(
            forget_candidates, now
        )

        # Stage 6: 受け渡し準備（安全弁チェック）
        result = self._prepare_handoff(
            forget_candidates, fix_signs,
            newly_forgotten, newly_recovered, now,
        )

        return result

    # ─── Series Index Management ─────────────────────────────────

    def _update_series_index(
        self, inputs: ForgettingFixationInputs, now: float,
    ) -> None:
        """入力から記憶系列索引を更新する。"""
        existing_ids = {s.source_id for s in self._state.series_index}

        # エピソード記憶から系列登録
        for entry in inputs.episode_entries:
            sid = entry.get("id", entry.get("episode_id", ""))
            if not sid:
                continue
            if sid not in existing_ids:
                rec = MemorySeriesRecord(
                    source="episodic",
                    source_id=sid,
                    creation_time=now,
                    emotion_strength=abs(entry.get("emotional_valence", 0.0)),
                )
                self._state.series_index.append(rec)
                existing_ids.add(sid)
            else:
                # 参照記録
                self._record_reference(sid, now)

        # 感情結合から系列登録
        for entry in inputs.binding_entries:
            sid = entry.get("id", entry.get("binding_id", ""))
            if not sid:
                continue
            if sid not in existing_ids:
                rec = MemorySeriesRecord(
                    source="binding",
                    source_id=sid,
                    creation_time=now,
                    emotion_strength=entry.get("freshness", 0.0),
                )
                self._state.series_index.append(rec)
                existing_ids.add(sid)
            else:
                self._record_reference(sid, now)

        # 長期記憶から系列登録
        for entry in inputs.long_term_entries:
            sid = entry.get("id", entry.get("memory_id", ""))
            if not sid:
                continue
            if sid not in existing_ids:
                rec = MemorySeriesRecord(
                    source="long_term",
                    source_id=sid,
                    creation_time=now,
                    emotion_strength=abs(entry.get("emotional_valence", 0.0)),
                )
                self._state.series_index.append(rec)
                existing_ids.add(sid)
            else:
                self._record_reference(sid, now)

        # 保護状態の更新
        protected_set = set(inputs.protected_ids)
        for rec in self._state.series_index:
            rec.is_protected = rec.source_id in protected_set

        # トリミング
        if len(self._state.series_index) > self._config.max_series:
            # 不可視化済みの系列を優先的に削除
            invisible = [
                s for s in self._state.series_index
                if s.forgetting_stage == ForgettingStage.INVISIBLE.value
            ]
            if invisible:
                remove_ids = {s.series_id for s in invisible[:len(self._state.series_index) - self._config.max_series]}
                self._state.series_index = [
                    s for s in self._state.series_index
                    if s.series_id not in remove_ids
                ]

    def _record_reference(self, source_id: str, now: float) -> None:
        """参照を記録し、希薄化を回復する。"""
        for rec in self._state.series_index:
            if rec.source_id == source_id:
                rec.reference_count += 1
                rec.last_reference_time = now
                # 希薄化回復
                rec.dilution = _clamp(
                    rec.dilution - self._config.reference_recovery, 0.0, 1.0
                )
                rec.forgetting_stage = _stage_from_dilution(rec.dilution).value
                # 忘却中なら復帰
                if rec.status == SeriesStatus.FORGETTING.value:
                    rec.status = SeriesStatus.RECOVERED.value
                    self._state.total_recovered += 1
                    if rec.source_id not in self._state.recovery_candidates:
                        self._state.recovery_candidates.append(rec.source_id)
                # 再利用履歴
                rec.reuse_count += 1
                self._state.reuse_history[source_id] = (
                    self._state.reuse_history.get(source_id, 0) + 1
                )
                break

        # 参照履歴に追加
        self._state.reference_history.append({
            "source_id": source_id,
            "timestamp": now,
        })
        if len(self._state.reference_history) > self._config.max_reference_history:
            self._state.reference_history = self._state.reference_history[
                -self._config.max_reference_history:
            ]

    # ─── Stage 1: 忘却候補抽出 ───────────────────────────────────

    def _extract_forgetting_candidates(
        self,
        inputs: ForgettingFixationInputs,
        now: float,
    ) -> list[ForgettingCandidate]:
        """参照希薄化した系列を段階移行候補として抽出する。"""
        candidates: list[ForgettingCandidate] = []
        cfg = self._config

        for rec in self._state.series_index:
            # 保護状態は忘却候補化の対象外
            if rec.is_protected:
                continue

            # 既に不可視の系列はスキップ
            if rec.forgetting_stage == ForgettingStage.INVISIBLE.value:
                continue

            # 希薄化進行
            time_since_ref = now - rec.last_reference_time if rec.last_reference_time > 0 else now - rec.creation_time

            # 感情結合による保護
            emotion_protection = rec.emotion_strength * cfg.emotion_protection_factor
            effective_rate = cfg.dilution_rate * (1.0 - emotion_protection)

            # 希薄化更新
            new_dilution = _clamp(rec.dilution + effective_rate, 0.0, 1.0)
            rec.dilution = new_dilution
            self._state.dilution_map[rec.source_id] = new_dilution

            # 段階判定
            new_stage = _stage_from_dilution(new_dilution)
            old_stage = rec.forgetting_stage

            if new_stage.value != old_stage:
                candidate = ForgettingCandidate(
                    series_id=rec.series_id,
                    current_stage=old_stage,
                    proposed_stage=new_stage.value,
                    dilution=new_dilution,
                    time_since_reference=time_since_ref,
                    reason=f"dilution={new_dilution:.3f}, time_since_ref={time_since_ref:.0f}s",
                )
                candidates.append(candidate)

        return candidates

    # ─── Stage 2: 固定化兆候抽出 ─────────────────────────────────

    def _extract_fixation_signs(
        self,
        inputs: ForgettingFixationInputs,
        now: float,
    ) -> list[FixationSign]:
        """反復優位・代替不可視化・再利用偏在を複数断面で観測する。"""
        signs: list[FixationSign] = []
        cfg = self._config

        for rec in self._state.series_index:
            indicators: list[str] = []
            score = 0.0

            # 指標1: 反復参照（参照頻度断面）
            if rec.reference_count >= cfg.fixation_reference_threshold:
                indicators.append("repeated_reference")
                score += 0.3

            # 指標2: 再利用偏在（再利用間隔断面）
            total_reuse = sum(self._state.reuse_history.values()) if self._state.reuse_history else 0
            if total_reuse > 0:
                own_reuse = self._state.reuse_history.get(rec.source_id, 0)
                if own_reuse / total_reuse > 0.3:
                    indicators.append("reuse_concentration")
                    score += 0.25

            # 指標3: 感情結合が強い（感情連結断面）
            if rec.emotion_strength > 0.5:
                indicators.append("strong_emotion_link")
                score += 0.2

            # 指標4: 代替系列の不可視化（競合系列断面）
            if inputs.invisible_alternative_count > 2:
                indicators.append("alternatives_invisible")
                score += 0.15

            # 指標5: 直近の反復参照リストに含まれる（固定化兆候断面）
            if rec.source_id in inputs.repeated_reference_ids:
                indicators.append("recent_repeated")
                score += 0.1

            # 交差断面の閾値チェック
            if len(indicators) >= cfg.fixation_cross_section_threshold:
                score = _clamp(score)
                level = _fixation_from_score(score)

                # 既存の兆候を更新するか新規作成
                existing = self._find_fixation_sign(rec.series_id)
                if existing:
                    existing.score = max(existing.score, score)
                    existing.level = level.value
                    existing.observation_count += 1
                    existing.indicators = list(set(existing.indicators + indicators))
                    existing.timestamp = now
                    signs.append(existing)
                else:
                    sign = FixationSign(
                        series_id=rec.series_id,
                        score=score,
                        level=level.value,
                        observation_count=1,
                        indicators=indicators,
                        timestamp=now,
                    )
                    self._state.fixation_sign_history.append(sign)
                    self._state.total_fixation_signs += 1
                    signs.append(sign)

                # 系列の固定化情報を更新
                rec.fixation_score = score
                rec.fixation_level = level.value
                if rec.status == SeriesStatus.ACTIVE.value:
                    rec.status = SeriesStatus.FIXATING.value

        # トリミング
        if len(self._state.fixation_sign_history) > cfg.max_fixation_history:
            self._state.fixation_sign_history = self._state.fixation_sign_history[
                -cfg.max_fixation_history:
            ]

        return signs

    def _find_fixation_sign(self, series_id: str) -> Optional[FixationSign]:
        """既存の固定化兆候を探す。"""
        for sign in self._state.fixation_sign_history:
            if sign.series_id == series_id:
                return sign
        return None

    # ─── Stage 3: 候補整列 ───────────────────────────────────────

    def _align_candidates(
        self,
        forget_candidates: list[ForgettingCandidate],
        fix_signs: list[FixationSign],
    ) -> None:
        """忘却候補と固定化候補を独立保持する。"""
        # 固定化兆候のある系列は忘却候補から除外
        fixating_ids = {s.series_id for s in fix_signs if s.level != FixationLevel.NONE.value}
        i = 0
        while i < len(forget_candidates):
            if forget_candidates[i].series_id in fixating_ids:
                forget_candidates.pop(i)
            else:
                i += 1

        # 忘却候補の更新
        self._state.forgetting_candidates = forget_candidates[:]
        if len(self._state.forgetting_candidates) > self._config.max_forgetting_candidates:
            self._state.forgetting_candidates = self._state.forgetting_candidates[
                -self._config.max_forgetting_candidates:
            ]

    # ─── Stage 4: 競合保持 ───────────────────────────────────────

    def _retain_competition(self, now: float) -> None:
        """主系列と代替系列を並立保持し、代替系列の再浮上経路を残す。"""
        # アクティブな系列から代替系列を特定
        active = [
            s for s in self._state.series_index
            if s.status in (SeriesStatus.ACTIVE.value, SeriesStatus.RECOVERED.value)
            and s.forgetting_stage in (
                ForgettingStage.ACTIVE.value,
                ForgettingStage.WEAKENING.value,
            )
        ]

        # 参照が少ない系列を代替系列として記録
        if len(active) > 1:
            avg_ref = sum(s.reference_count for s in active) / len(active)
            alternatives = [
                s.source_id for s in active
                if s.reference_count < avg_ref * 0.5
            ]
            self._state.alternative_series = alternatives[
                :self._config.max_alternative_series
            ]

    # ─── Stage 5: 段階忘却情報化 ─────────────────────────────────

    def _apply_staged_forgetting(
        self,
        candidates: list[ForgettingCandidate],
        now: float,
    ) -> tuple[int, int]:
        """段階移行を適用し、忘却/復帰数を返す。"""
        newly_forgotten = 0
        newly_recovered = 0

        for candidate in candidates:
            rec = self._find_series_by_id(candidate.series_id)
            if rec is None:
                continue

            # 段階移行
            rec.forgetting_stage = candidate.proposed_stage
            rec.dilution = candidate.dilution

            if candidate.proposed_stage == ForgettingStage.INVISIBLE.value:
                rec.status = SeriesStatus.FORGETTING.value
                newly_forgotten += 1
                self._state.total_forgotten += 1
                # 復帰候補に登録（不可逆忘却の恒常化を防ぐ）
                if rec.source_id not in self._state.recovery_candidates:
                    self._state.recovery_candidates.append(rec.source_id)
            elif candidate.proposed_stage in (
                ForgettingStage.WEAKENING.value,
                ForgettingStage.FADING.value,
                ForgettingStage.NEAR_INVISIBLE.value,
            ):
                if rec.status != SeriesStatus.FORGETTING.value:
                    rec.status = SeriesStatus.FORGETTING.value

        # 復帰候補のトリミング
        if len(self._state.recovery_candidates) > self._config.max_recovery_candidates:
            self._state.recovery_candidates = self._state.recovery_candidates[
                -self._config.max_recovery_candidates:
            ]

        return newly_forgotten, newly_recovered

    def _find_series_by_id(self, series_id: str) -> Optional[MemorySeriesRecord]:
        """series_idから系列レコードを探す。"""
        for rec in self._state.series_index:
            if rec.series_id == series_id:
                return rec
        return None

    # ─── Stage 6: 受け渡し準備 ───────────────────────────────────

    def _prepare_handoff(
        self,
        forget_candidates: list[ForgettingCandidate],
        fix_signs: list[FixationSign],
        newly_forgotten: int,
        newly_recovered: int,
        now: float,
    ) -> ForgettingFixationResult:
        """安全弁チェックを行い結果を返す。"""
        cfg = self._config

        # 系列統計
        active_count = sum(
            1 for s in self._state.series_index
            if s.status in (SeriesStatus.ACTIVE.value, SeriesStatus.RECOVERED.value)
        )
        forgetting_count = sum(
            1 for s in self._state.series_index
            if s.status == SeriesStatus.FORGETTING.value
        )
        invisible_count = sum(
            1 for s in self._state.series_index
            if s.forgetting_stage == ForgettingStage.INVISIBLE.value
        )
        protected_count = sum(
            1 for s in self._state.series_index
            if s.is_protected
        )

        # ── 安全弁1: 忘却収束警告 ──
        alternatives_supplemented = False
        if len(forget_candidates) > 0:
            series_counts: dict[str, int] = {}
            for c in forget_candidates:
                series_counts[c.series_id] = series_counts.get(c.series_id, 0) + 1
            total = len(forget_candidates)
            max_ratio = max(series_counts.values()) / total if total > 0 else 0
            if max_ratio >= cfg.convergence_threshold:
                self._state.convergence_warning = True
                # 復帰候補と代替系列を補充
                alternatives_supplemented = self._supplement_alternatives(now)
            else:
                self._state.convergence_warning = False

        # ── 安全弁2: 忘却過密化 ──
        forgetting_slowed = False
        if len(forget_candidates) >= cfg.overdense_threshold:
            self._state.overdense_warning = True
            # 復帰経路を先に評価
            forgetting_slowed = self._slow_forgetting(now)
        else:
            self._state.overdense_warning = False

        # ── 自己強化ループ防止 ──
        # 主系列の継続再参照のみが続く場合、代替系列を再提示
        self._prevent_self_reinforcement(now)

        return ForgettingFixationResult(
            forgetting_candidates=forget_candidates,
            newly_forgotten=newly_forgotten,
            newly_recovered=newly_recovered,
            fixation_signs=fix_signs,
            newly_fixating=len(fix_signs),
            active_series=active_count,
            forgetting_series=forgetting_count,
            invisible_series=invisible_count,
            protected_series=protected_count,
            convergence_warning=self._state.convergence_warning,
            overdense_warning=self._state.overdense_warning,
            alternatives_supplemented=alternatives_supplemented,
            forgetting_slowed=forgetting_slowed,
            cycle_count=self._state.cycle_count,
        )

    def _supplement_alternatives(self, now: float) -> bool:
        """復帰候補と代替系列を補充し、複線状態に戻す。"""
        supplemented = False
        for source_id in self._state.recovery_candidates[:5]:
            for rec in self._state.series_index:
                if rec.source_id == source_id and rec.forgetting_stage != ForgettingStage.ACTIVE.value:
                    # 希薄化を部分回復
                    rec.dilution = _clamp(rec.dilution - 0.1, 0.0, 1.0)
                    rec.forgetting_stage = _stage_from_dilution(rec.dilution).value
                    supplemented = True
        return supplemented

    def _slow_forgetting(self, now: float) -> bool:
        """忘却進行を緩和し、復帰経路を先に評価する。"""
        slowed = False
        for rec in self._state.series_index:
            if rec.status == SeriesStatus.FORGETTING.value:
                if rec.forgetting_stage == ForgettingStage.NEAR_INVISIBLE.value:
                    # 不可視化を遅延
                    rec.dilution = _clamp(rec.dilution - 0.05, 0.0, 1.0)
                    rec.forgetting_stage = _stage_from_dilution(rec.dilution).value
                    slowed = True
        return slowed

    def _prevent_self_reinforcement(self, now: float) -> None:
        """主系列の継続再参照のみが続く場合、代替系列の再提示を優先。"""
        if len(self._state.reference_history) < 5:
            return

        recent_refs = self._state.reference_history[-10:]
        ref_ids = [r.get("source_id", "") for r in recent_refs]

        # 単一IDが支配的かチェック
        if not ref_ids:
            return
        from collections import Counter
        counts = Counter(ref_ids)
        most_common_id, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(ref_ids) > 0.6:
            # 代替系列の再提示
            for alt_id in self._state.alternative_series[:3]:
                for rec in self._state.series_index:
                    if rec.source_id == alt_id:
                        rec.dilution = _clamp(rec.dilution - 0.05, 0.0, 1.0)
                        rec.forgetting_stage = _stage_from_dilution(rec.dilution).value
                        break


# =============================================================================
# Summary (enrichment用)
# =============================================================================

def get_forgetting_fixation_summary(state: ForgettingFixationState) -> str:
    """忘却・固定化状態の要約（enrichment用）。"""
    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    active = sum(
        1 for s in state.series_index
        if s.status in (SeriesStatus.ACTIVE.value, SeriesStatus.RECOVERED.value)
    )
    forgetting = sum(
        1 for s in state.series_index
        if s.status == SeriesStatus.FORGETTING.value
    )
    fixating = sum(
        1 for s in state.series_index
        if s.status == SeriesStatus.FIXATING.value
    )

    if active:
        parts.append(f"活性={active}")
    if forgetting:
        parts.append(f"忘却中={forgetting}")
    if fixating:
        parts.append(f"固定化={fixating}")
    if state.total_forgotten:
        parts.append(f"忘却累計={state.total_forgotten}")
    if state.total_recovered:
        parts.append(f"復帰={state.total_recovered}")
    if state.convergence_warning:
        parts.append("⚠収束偏向")
    if state.overdense_warning:
        parts.append("⚠過密")
    if state.recovery_candidates:
        parts.append(f"復帰候補={len(state.recovery_candidates)}")

    return " ".join(parts) if parts else "記憶流動: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_forgetting_fixation_processor(
    config: Optional[ForgettingFixationConfig] = None,
) -> MemoryForgettingFixationProcessor:
    """MemoryForgettingFixationProcessorのファクトリ関数。"""
    return MemoryForgettingFixationProcessor(config=config)
