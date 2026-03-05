"""
psyche/other_boundary_accumulation.py - 他者境界の多相蓄積

他者モデルが生成する自他境界乖離度を相手識別情報と対にして
時間軸上に蓄積し、その推移をREAD-ONLY参照情報として提供する。

設計原則 (design_other_boundary_accumulation.md 準拠):
- 境界の乖離度を制御・調整・最適化しない
- 特定の相手との「適切な距離」「望ましい境界」を定義しない
- 相手との関係性の評価・格付け・ランキングを行わない
- 蓄積された推移からパターンを抽出しない
- 相手別の統計量（平均乖離度、分散等）を算出しない
- 蓄積情報を判断・行動選択・ポリシー選択に接続しない
- 相手との「距離感」「親密度」「信頼度」等の評価的概念を導入しない
- 他者モデルの仮説強度を直接操作しない
- 境界情報に基づいて自己の振る舞いを変更する経路を持たない

3段パイプライン:
1. 境界情報の受領と相手別分離蓄積 (receive and accumulate per user)
2. 鮮度管理と自然消失 (freshness management and natural disappearance)
3. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. FIFO自然消失: 相手別上限+全体上限到達時に最古の記録から押し出し
2. 鮮度段階減衰: 全蓄積記録の鮮度が毎処理で段階的に減少
3. セッション境界減衰: セッション境界で一律の鮮度減衰を適用
4. 相手別固定パターン形成の検出: 乖離度段階値が長期間同一に留まる状態を検出
5. 相手別蓄積量偏りの検出: 偏りのある相手の鮮度減衰を加速
6. 乖離度収束の検出: 段階値分布が単一段階に収束した状態を検出
7. enrichment直接露出の制限: 蓄積記録数と乖離度段階値の分布のみ
8. パターン抽出の構造的禁止: 相関分析・傾向分析・頻度集計を行わない

CRITICAL CONSTRAINTS (other_agent_model から継承):
- 他者の意図・価値・信念を断定しない
- 正誤や善悪の評価を付与しない
- 目的や行動の最適化に結び付けない
- 自己像や人格の方向性を固定しない
- 判断・目的・価値・責任に一切接続しない
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

class FreshnessStage(Enum):
    """鮮度段階（memory_forgetting_fixation / other_model_dialogue_learning パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class DivergenceLevel(Enum):
    """乖離度の段階値列挙型。
    全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。
    区間幅は均等（各0.2幅）。
    """
    LEVEL_0 = "level_0"        # 0.0 - 0.2 未満
    LEVEL_1 = "level_1"        # 0.2 - 0.4 未満
    LEVEL_2 = "level_2"        # 0.4 - 0.6 未満
    LEVEL_3 = "level_3"        # 0.6 - 0.8 未満
    LEVEL_4 = "level_4"        # 0.8 - 1.0


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


def determine_divergence_level(divergence: float) -> DivergenceLevel:
    """乖離度の数値から段階値を決定する。
    段階値の区間幅は均等。評価的含意を持たない。
    安全弁8: パターン抽出禁止。この関数は「乖離度」のみを入力とし、
    推移方向や傾向は入力に含めない。
    """
    if divergence < 0.2:
        return DivergenceLevel.LEVEL_0
    elif divergence < 0.4:
        return DivergenceLevel.LEVEL_1
    elif divergence < 0.6:
        return DivergenceLevel.LEVEL_2
    elif divergence < 0.8:
        return DivergenceLevel.LEVEL_3
    else:
        return DivergenceLevel.LEVEL_4


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
class BoundaryRecord:
    """蓄積記録単位。

    境界乖離度のスナップショットを相手識別子と対にして保持する。
    一度記録されたら内容は変更されない（追記のみ）。鮮度値のみが時間経過に伴い減少する。
    安全弁1: 全記録等価。記録間の重み・スコア・優先度を付与しない。
    安全弁8: パターン抽出禁止。記録間の比較・差分・相関の計算を行わない。
    """
    record_id: str = field(default_factory=_gen_id)
    user_id: str = ""               # 相手識別子
    divergence: float = 0.0         # 乖離度（境界構造体から転記）
    divergence_level: str = DivergenceLevel.LEVEL_0.value  # 乖離度段階値
    boundary_aspects: tuple[str, ...] = ()  # 境界側面の列挙（どの側面が異なるか）
    self_description: str = ""      # 自己記述の要約（固定長切り詰め）
    other_description: str = ""     # 他者記述の要約（固定長切り詰め）
    tick: int = 0                   # 蓄積時点のティック番号
    timestamp: float = field(default_factory=time.time)  # 蓄積時点のタイムスタンプ
    freshness: float = 1.0         # 鮮度値
    freshness_stage: str = FreshnessStage.ACTIVE.value  # 鮮度段階

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "divergence": self.divergence,
            "divergence_level": self.divergence_level,
            "boundary_aspects": list(self.boundary_aspects),
            "self_description": self.self_description,
            "other_description": self.other_description,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            user_id=data.get("user_id", ""),
            divergence=data.get("divergence", 0.0),
            divergence_level=data.get("divergence_level", DivergenceLevel.LEVEL_0.value),
            boundary_aspects=tuple(data.get("boundary_aspects", ())),
            self_description=data.get("self_description", ""),
            other_description=data.get("other_description", ""),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。

    特定の相手について蓄積記録の乖離度段階値が極端に偏った状態の検出記録。
    参照情報としてのみ保持する。
    """
    user_id: str = ""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    dominant_level: str = ""        # 支配的な乖離度段階値
    level_diversity: float = 1.0    # 段階値の多様性
    record_count: int = 0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "dominant_level": self.dominant_level,
            "level_diversity": self.level_diversity,
            "record_count": self.record_count,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            user_id=data.get("user_id", ""),
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            dominant_level=data.get("dominant_level", ""),
            level_diversity=data.get("level_diversity", 1.0),
            record_count=data.get("record_count", 0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class OtherBoundaryAccumulationState:
    """他者境界蓄積の内部状態。永続化対象。"""

    # 1. 蓄積記録の集合（相手別・時系列順に保持する可変構造）
    records: list[BoundaryRecord] = field(default_factory=list)

    # 2. 相手別索引（user_id -> record_id リスト）
    #    other_model_dialogue_learning / situational_self_presentation と同一のデータ構造パターン
    user_index: dict[str, list[str]] = field(default_factory=dict)

    # 3. 収束監視記録
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # 4. カウンタ群
    cycle_count: int = 0
    total_records_added: int = 0
    total_records_pushed_out: int = 0
    total_records_invisible: int = 0

    # 5. 安全弁フラグ群
    fixed_pattern_warning: bool = False       # 相手別固定パターン形成警告
    accumulation_bias_warning: bool = False   # 相手別蓄積量偏り警告
    divergence_convergence_warning: bool = False  # 乖離度収束警告

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "user_index": {k: list(v) for k, v in self.user_index.items()},
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "cycle_count": self.cycle_count,
            "total_records_added": self.total_records_added,
            "total_records_pushed_out": self.total_records_pushed_out,
            "total_records_invisible": self.total_records_invisible,
            "fixed_pattern_warning": self.fixed_pattern_warning,
            "accumulation_bias_warning": self.accumulation_bias_warning,
            "divergence_convergence_warning": self.divergence_convergence_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OtherBoundaryAccumulationState":
        return cls(
            records=[
                BoundaryRecord.from_dict(r) for r in data.get("records", [])
            ],
            user_index={
                k: list(v) for k, v in data.get("user_index", {}).items()
            },
            convergence_records=[
                ConvergenceRecord.from_dict(c)
                for c in data.get("convergence_records", [])
            ],
            cycle_count=data.get("cycle_count", 0),
            total_records_added=data.get("total_records_added", 0),
            total_records_pushed_out=data.get("total_records_pushed_out", 0),
            total_records_invisible=data.get("total_records_invisible", 0),
            fixed_pattern_warning=data.get("fixed_pattern_warning", False),
            accumulation_bias_warning=data.get("accumulation_bias_warning", False),
            divergence_convergence_warning=data.get("divergence_convergence_warning", False),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。

        安全弁3: セッション境界で一律の鮮度減衰を適用し、
        セッションをまたいで特定の蓄積が高鮮度を維持し続けることを防ぐ。
        十分に希薄化した記録は不可視化・除去される。
        """
        remove_ids: set[str] = set()
        for rec in self.records:
            rec.freshness = _clamp(rec.freshness - decay_factor)
            rec.freshness_stage = _stage_from_freshness(rec.freshness).value
            if rec.freshness < 0.1:
                remove_ids.add(rec.record_id)

        if remove_ids:
            self.records = [
                r for r in self.records if r.record_id not in remove_ids
            ]
            for uid in self.user_index:
                self.user_index[uid] = [
                    rid for rid in self.user_index[uid] if rid not in remove_ids
                ]
            self.total_records_invisible += len(remove_ids)


# =============================================================================
# Result
# =============================================================================

@dataclass
class BoundaryAccumulationResult:
    """処理結果（参照情報形式のみ）。

    出力は「情報」としてのみ流れ、判断・評価・行動を直接引き起こさない。
    """
    newly_added: bool = False
    active_record_count: int = 0
    total_record_count: int = 0
    user_record_counts: dict[str, int] = field(default_factory=dict)
    divergence_level_distribution: dict[str, int] = field(default_factory=dict)
    # 安全弁フラグ
    fixed_pattern_warning: bool = False
    accumulation_bias_warning: bool = False
    divergence_convergence_warning: bool = False
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OtherBoundaryAccumulationConfig:
    """設定。"""
    # 相手別の蓄積上限（上限到達時は最古から押し出す: 安全弁1）
    max_records_per_user: int = 50
    # 全体の蓄積上限
    max_records_total: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))
    # 収束監視記録の上限
    max_convergence_records: int = 50
    # 記述要約の最大長
    description_max_length: int = 80

    # 鮮度減衰速度（安全弁2）
    freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))
    # 相手不在時の追加減衰速度
    absent_user_decay_rate: float = 0.01

    # 固定パターン検出の最低記録数
    fixed_pattern_min_records: int = 5
    # 固定パターン検出: 支配率閾値（同一段階値の比率がこれ以上で警告）
    fixed_pattern_dominance_threshold: float = 0.8

    # 蓄積量偏り検出の閾値（最大相手/平均の比）
    accumulation_bias_threshold: float = 3.0

    # 乖離度収束検出の閾値
    divergence_convergence_threshold: float = 0.8


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class OtherBoundaryAccumulationProcessor:
    """他者境界の多相蓄積プロセッサ。

    3段パイプライン:
    1. 境界情報の受領と相手別分離蓄積
    2. 鮮度管理と自然消失
    3. 参照情報としての受渡準備

    入力元: 他者モデルが観測時に生成する境界構造体(SelfOtherBoundary)と
    対話入力経路等から供給される相手識別子のみ。他の入力経路を開設しない。

    出力はenrichmentセクションとREAD-ONLYアクセサのみを通じて提供され、
    いずれの判断系にも直接入力経路を持たない。

    ポリシー選択経路への非接続: 蓄積された境界推移はポリシー候補の生成・選択・バイアス計算に入力されない。
    責任システムへの非接続: 境界推移の蓄積情報は責任の重量・分散・評価に影響しない。
    行動決定経路への非接続: 蓄積情報は行動選択や出力生成に直接作用しない。
    他者モデル仮説強度への非操作: 蓄積情報は他者モデルの仮説の生成・強度変更・競合解消に直接作用しない。
    感情パイプラインへの非接続: 蓄積情報は感情状態の変更に影響しない。
    """

    def __init__(self, config: Optional[OtherBoundaryAccumulationConfig] = None):
        self._config = config or OtherBoundaryAccumulationConfig()
        self._state = OtherBoundaryAccumulationState()

    @property
    def state(self) -> OtherBoundaryAccumulationState:
        return self._state

    @state.setter
    def state(self, value: OtherBoundaryAccumulationState) -> None:
        self._state = value

    def tick(
        self,
        boundary: Optional[Any] = None,
        user_id: str = "",
        current_tick: int = 0,
    ) -> BoundaryAccumulationResult:
        """orchestrator から呼ばれる単一エントリポイント。

        3段パイプラインを実行し結果を返す。
        出力は参照情報形式のみ。

        Args:
            boundary: 他者モデルが生成した境界構造体（SelfOtherBoundary）。
                      duck typing で divergence, boundary_aspects,
                      self_description, other_description を参照する。
            user_id: 対話入力経路から供給される相手識別子。
            current_tick: 現在のティック番号。
        """
        self._state.cycle_count += 1
        now = time.time()

        # Stage 1: 境界情報の受領と相手別分離蓄積
        newly_added = self._receive_and_accumulate(
            boundary, user_id, current_tick, now,
        )

        # Stage 2: 鮮度管理と自然消失
        self._apply_freshness_management(user_id, now)

        # Stage 3: 参照情報としての受渡準備
        return self._prepare_handoff(newly_added, now)

    # ─── Stage 1: 境界情報の受領と相手別分離蓄積 ──────────────────

    def _receive_and_accumulate(
        self,
        boundary: Optional[Any],
        user_id: str,
        current_tick: int,
        now: float,
    ) -> bool:
        """他者モデルが生成した境界情報を受領し、相手別に分離蓄積する。

        蓄積は相手別に分離して管理する。
        分離管理の構造は他者観測の長期蓄積構造の相手別整列の設計思想に倣う。
        すべての蓄積記録は等価であり、記録間の重み・スコア・優先度を付与しない。
        """
        if boundary is None or not user_id:
            return False

        cfg = self._config

        # Duck typing で境界構造体のフィールドを読み取る
        divergence = self._read_divergence(boundary)
        boundary_aspects = self._read_boundary_aspects(boundary)
        self_desc = self._read_self_description(boundary)
        other_desc = self._read_other_description(boundary)

        # 乖離度の段階値を算出
        div_level = determine_divergence_level(divergence)

        record = BoundaryRecord(
            user_id=user_id,
            divergence=_clamp(divergence),
            divergence_level=div_level.value,
            boundary_aspects=boundary_aspects,
            self_description=self_desc[:cfg.description_max_length],
            other_description=other_desc[:cfg.description_max_length],
            tick=current_tick,
            timestamp=now,
        )

        self._state.records.append(record)
        self._state.total_records_added += 1

        # 相手別索引に追加
        if user_id not in self._state.user_index:
            self._state.user_index[user_id] = []
        self._state.user_index[user_id].append(record.record_id)

        # 安全弁1: 相手別上限制御（FIFO最古押し出し）
        self._apply_per_user_pushout(user_id)

        # 安全弁1: 全体上限制御（FIFO最古押し出し）
        self._apply_total_pushout()

        logger.debug(
            "Boundary accumulated: user=%s, tick=%d, divergence=%.4f, "
            "level=%s, aspects=%d",
            user_id, current_tick, divergence,
            div_level.value, len(boundary_aspects),
        )

        return True

    def _read_divergence(self, boundary: Any) -> float:
        """Duck typing で乖離度を読み取る。"""
        if hasattr(boundary, "divergence"):
            val = getattr(boundary, "divergence", 0.0)
            if isinstance(val, (int, float)):
                return float(val)
        if isinstance(boundary, dict):
            val = boundary.get("divergence", 0.0)
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0

    def _read_boundary_aspects(self, boundary: Any) -> tuple[str, ...]:
        """Duck typing で境界側面を読み取る。"""
        if hasattr(boundary, "boundary_aspects"):
            val = getattr(boundary, "boundary_aspects", ())
            if isinstance(val, (tuple, list)):
                return tuple(str(a) for a in val)
        if isinstance(boundary, dict):
            val = boundary.get("boundary_aspects", ())
            if isinstance(val, (tuple, list)):
                return tuple(str(a) for a in val)
        return ()

    def _read_self_description(self, boundary: Any) -> str:
        """Duck typing で自己記述を読み取る。"""
        if hasattr(boundary, "self_description"):
            val = getattr(boundary, "self_description", "")
            if isinstance(val, str):
                return val
        if isinstance(boundary, dict):
            val = boundary.get("self_description", "")
            if isinstance(val, str):
                return val
        return ""

    def _read_other_description(self, boundary: Any) -> str:
        """Duck typing で他者記述を読み取る。"""
        if hasattr(boundary, "other_description"):
            val = getattr(boundary, "other_description", "")
            if isinstance(val, str):
                return val
        if isinstance(boundary, dict):
            val = boundary.get("other_description", "")
            if isinstance(val, str):
                return val
        return ""

    def _apply_per_user_pushout(self, user_id: str) -> None:
        """安全弁1: 相手別の上限数に達した場合、最古の記録から押し出す。
        特定の記録を保護・永続化する仕組みを持たない。
        """
        cfg = self._config
        if user_id not in self._state.user_index:
            return

        user_record_ids = self._state.user_index[user_id]
        if len(user_record_ids) > cfg.max_records_per_user:
            remove_count = len(user_record_ids) - cfg.max_records_per_user
            remove_ids = set(user_record_ids[:remove_count])
            self._state.user_index[user_id] = user_record_ids[remove_count:]
            self._state.records = [
                r for r in self._state.records if r.record_id not in remove_ids
            ]
            self._state.total_records_pushed_out += len(remove_ids)

    def _apply_total_pushout(self) -> None:
        """安全弁1: 全体の上限数に達した場合、最古の記録から押し出す。"""
        cfg = self._config
        if len(self._state.records) > cfg.max_records_total:
            remove_count = len(self._state.records) - cfg.max_records_total
            remove_records = self._state.records[:remove_count]
            remove_ids = {r.record_id for r in remove_records}
            self._state.records = self._state.records[remove_count:]
            self._state.total_records_pushed_out += remove_count

            # 相手別索引からも除去
            for uid in self._state.user_index:
                self._state.user_index[uid] = [
                    rid for rid in self._state.user_index[uid]
                    if rid not in remove_ids
                ]

    # ─── Stage 2: 鮮度管理と自然消失 ─────────────────────────────

    def _apply_freshness_management(
        self, active_user_id: str, now: float,
    ) -> None:
        """安全弁2: 全蓄積記録の鮮度を毎処理サイクルで段階的に減少させる。

        当該相手との対話が不在の場合は追加減衰が発生する。
        十分に希薄化した記録は不可視化へ移行し、やがて除去される。
        """
        cfg = self._config

        # 相手不在の判定（現在対話中でない全ユーザー）
        absent_users: set[str] = set()
        for uid in self._state.user_index:
            if uid != active_user_id:
                absent_users.add(uid)

        newly_invisible = 0
        for rec in self._state.records:
            if rec.freshness_stage == FreshnessStage.INVISIBLE.value:
                continue

            # 基本減衰
            decay = cfg.freshness_decay_rate
            # 相手不在時の追加減衰
            if rec.user_id in absent_users:
                decay += cfg.absent_user_decay_rate

            rec.freshness = _clamp(rec.freshness - decay)
            new_stage = _stage_from_freshness(rec.freshness)
            rec.freshness_stage = new_stage.value

            if new_stage == FreshnessStage.INVISIBLE:
                newly_invisible += 1

        # 不可視化した記録を除去
        if newly_invisible > 0:
            remove_ids: set[str] = set()
            for rec in self._state.records:
                if rec.freshness_stage == FreshnessStage.INVISIBLE.value:
                    remove_ids.add(rec.record_id)

            if remove_ids:
                self._state.records = [
                    r for r in self._state.records if r.record_id not in remove_ids
                ]
                for uid in self._state.user_index:
                    self._state.user_index[uid] = [
                        rid for rid in self._state.user_index[uid]
                        if rid not in remove_ids
                    ]
                self._state.total_records_invisible += len(remove_ids)

    # ─── Stage 3: 参照情報としての受渡準備 ────────────────────────

    def _prepare_handoff(
        self, newly_added: bool, now: float,
    ) -> BoundaryAccumulationResult:
        """安全弁チェックを行い、参照情報形式の結果を返す。

        出力は「情報」としてのみ流れ、判断・評価・行動を直接引き起こさない。
        """
        # 可視状態の記録のみを統計対象
        visible_records = [
            r for r in self._state.records
            if r.freshness_stage != FreshnessStage.INVISIBLE.value
        ]

        active_count = sum(
            1 for r in visible_records
            if r.freshness_stage == FreshnessStage.ACTIVE.value
        )

        # 相手別蓄積数
        user_counts: dict[str, int] = {}
        for rec in visible_records:
            user_counts[rec.user_id] = user_counts.get(rec.user_id, 0) + 1

        # 安全弁7: enrichmentに含めるのは蓄積記録数と乖離度段階値の分布のみ
        div_level_dist = self._compute_divergence_level_distribution(visible_records)

        # 安全弁4: 相手別固定パターン形成の検出
        self._check_fixed_pattern(visible_records, now)

        # 安全弁5: 相手別蓄積量偏りの検出
        self._check_accumulation_bias(user_counts)

        # 安全弁6: 乖離度収束の検出
        self._check_divergence_convergence(visible_records, now)

        return BoundaryAccumulationResult(
            newly_added=newly_added,
            active_record_count=active_count,
            total_record_count=len(visible_records),
            user_record_counts=user_counts,
            divergence_level_distribution=div_level_dist,
            fixed_pattern_warning=self._state.fixed_pattern_warning,
            accumulation_bias_warning=self._state.accumulation_bias_warning,
            divergence_convergence_warning=self._state.divergence_convergence_warning,
            cycle_count=self._state.cycle_count,
        )

    def _compute_divergence_level_distribution(
        self, records: list[BoundaryRecord],
    ) -> dict[str, int]:
        """乖離度段階値の分布を算出する。

        安全弁7: 個別記録の乖離度数値を直接露出せず、段階値の分布のみ提供する。
        安全弁8: 推移方向の記述をenrichmentに含めない。
        """
        dist: dict[str, int] = {}
        for rec in records:
            level = rec.divergence_level
            dist[level] = dist.get(level, 0) + 1
        return dist

    def _check_fixed_pattern(
        self, visible_records: list[BoundaryRecord], now: float,
    ) -> None:
        """安全弁4: 相手別固定パターン形成の検出。

        特定の相手について蓄積記録の乖離度段階値が長期間にわたり
        同一段階に留まる状態を検出し、参照情報として記録する。
        検出結果に基づく自動的な介入・修正は行わない。
        """
        cfg = self._config
        has_fixed = False

        for uid, record_ids in self._state.user_index.items():
            user_records = [
                r for r in visible_records
                if r.user_id == uid
            ]
            if len(user_records) < cfg.fixed_pattern_min_records:
                continue

            # 段階値の分布を算出
            level_counts: dict[str, int] = {}
            for rec in user_records:
                level_counts[rec.divergence_level] = (
                    level_counts.get(rec.divergence_level, 0) + 1
                )

            total = sum(level_counts.values())
            if total == 0:
                continue

            max_count = max(level_counts.values())
            dominance_ratio = max_count / total

            if dominance_ratio >= cfg.fixed_pattern_dominance_threshold:
                has_fixed = True
                dominant_level = max(level_counts, key=level_counts.get)  # type: ignore

                self._state.convergence_records.append(ConvergenceRecord(
                    user_id=uid,
                    convergence_score=dominance_ratio,
                    convergence_level=_convergence_from_score(dominance_ratio).value,
                    dominant_level=dominant_level,
                    level_diversity=len(level_counts) / max(total, 1),
                    record_count=total,
                    cycle=self._state.cycle_count,
                    timestamp=now,
                ))

        self._state.fixed_pattern_warning = has_fixed

        # 収束記録の上限制御
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

    def _check_accumulation_bias(self, user_counts: dict[str, int]) -> None:
        """安全弁5: 相手別蓄積量偏りの検出。

        特定の相手への蓄積が著しく偏った場合に検出する。
        偏りのある相手の蓄積記録の鮮度減衰を加速させることで、
        特定相手への蓄積の過度な集中を緩和する。
        """
        cfg = self._config

        if not user_counts or len(user_counts) < 2:
            self._state.accumulation_bias_warning = False
            return

        avg = sum(user_counts.values()) / len(user_counts)
        max_count = max(user_counts.values())

        if avg > 0 and max_count / avg > cfg.accumulation_bias_threshold:
            self._state.accumulation_bias_warning = True
            # 最大蓄積量の相手の鮮度減衰を加速
            biased_user = max(user_counts, key=user_counts.get)  # type: ignore
            for rec in self._state.records:
                if (
                    rec.user_id == biased_user
                    and rec.freshness_stage != FreshnessStage.INVISIBLE.value
                ):
                    rec.freshness = _clamp(
                        rec.freshness - cfg.absent_user_decay_rate
                    )
                    rec.freshness_stage = _stage_from_freshness(rec.freshness).value
        else:
            self._state.accumulation_bias_warning = False

    def _check_divergence_convergence(
        self, visible_records: list[BoundaryRecord], now: float,
    ) -> None:
        """安全弁6: 乖離度収束の検出。

        蓄積全体または特定相手の蓄積において、乖離度の段階値分布が
        単一段階に収束した状態を検出し、参照情報として記録する。
        自動的な修正は行わない。
        """
        cfg = self._config

        if not visible_records:
            self._state.divergence_convergence_warning = False
            return

        # 全体の段階値分布を確認
        level_counts: dict[str, int] = {}
        for rec in visible_records:
            level_counts[rec.divergence_level] = (
                level_counts.get(rec.divergence_level, 0) + 1
            )

        total = sum(level_counts.values())
        if total == 0:
            self._state.divergence_convergence_warning = False
            return

        max_count = max(level_counts.values())
        convergence_ratio = max_count / total

        if (
            convergence_ratio >= cfg.divergence_convergence_threshold
            and total >= cfg.fixed_pattern_min_records
        ):
            self._state.divergence_convergence_warning = True
            dominant_level = max(level_counts, key=level_counts.get)  # type: ignore

            self._state.convergence_records.append(ConvergenceRecord(
                user_id="__global__",
                convergence_score=convergence_ratio,
                convergence_level=_convergence_from_score(convergence_ratio).value,
                dominant_level=dominant_level,
                level_diversity=len(level_counts) / max(total, 1),
                record_count=total,
                cycle=self._state.cycle_count,
                timestamp=now,
            ))

            # 収束記録の上限制御
            if len(self._state.convergence_records) > cfg.max_convergence_records:
                self._state.convergence_records = (
                    self._state.convergence_records[-cfg.max_convergence_records:]
                )
        else:
            self._state.divergence_convergence_warning = False

    # ─── READ-ONLY アクセサ ──────────────────────────────────────

    def get_user_records(self, user_id: str) -> list[BoundaryRecord]:
        """特定の相手に対する蓄積記録をREAD-ONLYで返す。

        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。
        参照行為によって記録が変化することはない。
        安全弁8: パターン抽出禁止。蓄積記録の等価列挙のみを提供する。
        """
        return [
            r for r in self._state.records
            if r.user_id == user_id
            and r.freshness_stage != FreshnessStage.INVISIBLE.value
        ]

    def get_all_user_ids(self) -> list[str]:
        """蓄積記録が存在する全相手の識別子を返す（READ-ONLY参照）。

        相手間の比較・ランキング・差異抽出を行わない。
        """
        return [
            uid for uid, rids in self._state.user_index.items()
            if rids
        ]

    def get_record_count(self, user_id: str = "") -> int:
        """蓄積記録数を返す（READ-ONLY参照）。

        Args:
            user_id: 指定された場合はその相手のみの蓄積記録数。
                     空の場合は全体の蓄積記録数。
        """
        if user_id:
            return sum(
                1 for r in self._state.records
                if r.user_id == user_id
                and r.freshness_stage != FreshnessStage.INVISIBLE.value
            )
        return sum(
            1 for r in self._state.records
            if r.freshness_stage != FreshnessStage.INVISIBLE.value
        )

    def get_enrichment_data(self, user_id: str = "") -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        安全弁7: enrichmentに含めるのは蓄積記録数と乖離度段階値の分布のみ。
        個別記録の内容・相手間比較・推移方向の記述をenrichmentに含めない。

        出力はポリシー選択パイプラインの入力に直接接続しない。

        Args:
            user_id: 現在の対話相手の識別子。指定がある場合はその相手の情報も返す。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state

        visible_records = [
            r for r in st.records
            if r.freshness_stage != FreshnessStage.INVISIBLE.value
        ]

        # 全体の段階値分布
        div_level_dist = self._compute_divergence_level_distribution(visible_records)

        # 相手数
        active_users = len([
            uid for uid, rids in st.user_index.items() if rids
        ])

        result: dict[str, Any] = {
            "cycle_count": st.cycle_count,
            "total_record_count": len(visible_records),
            "user_count": active_users,
            "divergence_level_distribution": div_level_dist,
            "fixed_pattern_warning": st.fixed_pattern_warning,
            "accumulation_bias_warning": st.accumulation_bias_warning,
            "divergence_convergence_warning": st.divergence_convergence_warning,
        }

        # 現在の対話相手に関する蓄積概要
        if user_id:
            user_records = [
                r for r in visible_records if r.user_id == user_id
            ]
            user_div_dist = self._compute_divergence_level_distribution(user_records)
            result["current_user"] = {
                "record_count": len(user_records),
                "divergence_level_distribution": user_div_dist,
            }

        summary_text = get_boundary_accumulation_summary(st, user_id)
        result["summary_text"] = summary_text

        return result

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す（READ-ONLY参照）。"""
        st = self._state
        visible = sum(
            1 for r in st.records
            if r.freshness_stage != FreshnessStage.INVISIBLE.value
        )
        return {
            "cycle_count": st.cycle_count,
            "total_records_added": st.total_records_added,
            "current_record_count": visible,
            "total_pushed_out": st.total_records_pushed_out,
            "total_invisible": st.total_records_invisible,
            "user_count": len([
                uid for uid, rids in st.user_index.items() if rids
            ]),
            "fixed_pattern_warning": st.fixed_pattern_warning,
            "accumulation_bias_warning": st.accumulation_bias_warning,
            "divergence_convergence_warning": st.divergence_convergence_warning,
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_boundary_accumulation_summary(
    state: OtherBoundaryAccumulationState,
    user_id: str = "",
) -> str:
    """他者境界蓄積状態の要約（enrichment用）。

    安全弁7: 蓄積記録数と乖離度段階値の分布のみ。
    個別記録の内容・推移方向を含めない。
    評価判定・行動指示を含まない。
    """
    if state.cycle_count == 0 and not state.records:
        return "境界蓄積: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    visible_records = [
        r for r in state.records
        if r.freshness_stage != FreshnessStage.INVISIBLE.value
    ]
    if visible_records:
        parts.append(f"蓄積={len(visible_records)}")

    active_users = len([
        uid for uid, rids in state.user_index.items() if rids
    ])
    if active_users > 0:
        parts.append(f"相手={active_users}")

    # 現在の対話相手の蓄積記録数
    if user_id:
        user_count = sum(1 for r in visible_records if r.user_id == user_id)
        if user_count > 0:
            parts.append(f"当相手={user_count}")

    if state.total_records_pushed_out > 0:
        parts.append(f"押出累計={state.total_records_pushed_out}")

    if state.fixed_pattern_warning:
        parts.append("固定パターン")
    if state.accumulation_bias_warning:
        parts.append("蓄積偏り")
    if state.divergence_convergence_warning:
        parts.append("乖離度収束")

    return " ".join(parts) if parts else "境界蓄積: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_other_boundary_accumulation_processor(
    config: Optional[OtherBoundaryAccumulationConfig] = None,
) -> OtherBoundaryAccumulationProcessor:
    """OtherBoundaryAccumulationProcessor のファクトリ関数。"""
    return OtherBoundaryAccumulationProcessor(config=config)
