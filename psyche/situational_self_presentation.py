"""
psyche/situational_self_presentation.py - 状況依存的自己呈示の認知

自己行動知覚が保持する出力記録を相手識別情報と対にして蓄積し、
相手別の自己出力記録の構成を事後的に記述する構造を提供する。

設計原則 (design_situational_self_presentation.md 準拠):
- 相手別の出力パターンを抽出・適用しない。マッピングの形成・保持・参照を行わない
- 出力テキストの品質評価を行わない
- 相手に応じた出力の最適化・調整を行わない
- 相手別の呈示戦略を構成・保持しない
- 出力テキストの意味解析を行わない。テキスト非解釈原則を継承
- 感情パイプラインのパラメータを変更しない
- 他者モデルの仮説強度を直接操作しない
- すべての記録は等価であり、重み・スコア・優先度を付与しない
- ポリシー選択パイプライン・バイアス計算・安定化弁への直接入力経路を持たない

3段パイプライン:
1. 相手別記録の受領と蓄積 (receive and accumulate per user)
2. 構成記述の生成 (composition description generation)
3. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. 記録の等価性: 重み・重要度・代表性を付与しない
2. パターン抽出の禁止: 傾向・パターン・類型を抽出しない
3. FIFO自然消失: 上限到達時の最古押し出し + 鮮度減衰による不可視化
4. 構成記述の非累積性: 過去の構成記述が現在に影響しない
5. 相手別マッピング形成の禁止: 恒常的対応を形成しない
6. enrichment直接露出の制限: 蓄積記録数とポリシーラベル種類数の段階値のみ
7. ポリシー選択経路の遮断: 判断系への直接入力禁止
8. 収束監視: 種類数が極端に少ない状態の検出（参照情報としてのみ記録）
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

class RecordFreshness(Enum):
    """鮮度段階（memory_forgetting_fixation / other_model_dialogue_learning パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class TypeCountLevel(Enum):
    """ポリシーラベル種類数の段階値（behavioral_diversity_description パターン準拠）。
    全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。
    区間幅は均等（各5種類幅）。
    """
    LEVEL_0 = "level_0"            # 0種類
    LEVEL_1_5 = "level_1_5"        # 1-5種類
    LEVEL_6_10 = "level_6_10"      # 6-10種類
    LEVEL_11_15 = "level_11_15"    # 11-15種類
    LEVEL_16_PLUS = "level_16_plus"  # 16種類以上


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


def _freshness_from_value(freshness: float) -> RecordFreshness:
    """鮮度値から段階を返す。"""
    if freshness >= 0.8:
        return RecordFreshness.ACTIVE
    elif freshness >= 0.6:
        return RecordFreshness.WEAKENING
    elif freshness >= 0.4:
        return RecordFreshness.FADING
    elif freshness >= 0.2:
        return RecordFreshness.NEAR_INVISIBLE
    else:
        return RecordFreshness.INVISIBLE


def determine_type_count_level(type_count: int) -> TypeCountLevel:
    """種類数から段階値を決定する。
    behavioral_diversity_description と同形式。
    段階値の区間幅は均等。評価的含意を持たない。
    安全弁2: パターン抽出禁止。この関数は「種類数」のみを入力とし、
    各種類の出現回数は入力に含めない。
    """
    if type_count == 0:
        return TypeCountLevel.LEVEL_0
    elif type_count <= 5:
        return TypeCountLevel.LEVEL_1_5
    elif type_count <= 10:
        return TypeCountLevel.LEVEL_6_10
    elif type_count <= 15:
        return TypeCountLevel.LEVEL_11_15
    else:
        return TypeCountLevel.LEVEL_16_PLUS


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
class PresentationRecord:
    """相手別蓄積記録単位。

    一度記録されたら変更されない（追記のみ）。
    安全弁1: 全記録等価。記録間の重み・スコア・優先度を付与しない。
    安全弁2: パターン抽出禁止。記録間の比較・差分・相関の計算を行わない。
    テキスト非解釈原則を継承。テキストは生のテキスト文字列として保持する。
    """
    record_id: str = field(default_factory=_gen_id)
    user_id: str = ""               # 相手識別子
    response_text: str = ""          # 出力テキスト（プレビュー長に切り詰め）
    policy_label: str = ""           # そのターンで選択されていたポリシーラベル
    tick: int = 0                    # 通知時点のティック番号
    timestamp: float = field(default_factory=time.time)
    # 鮮度
    freshness: float = 1.0
    freshness_stage: str = RecordFreshness.ACTIVE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "response_text": self.response_text,
            "policy_label": self.policy_label,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PresentationRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            user_id=data.get("user_id", ""),
            response_text=data.get("response_text", ""),
            policy_label=data.get("policy_label", ""),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", RecordFreshness.ACTIVE.value),
        )


@dataclass
class CompositionDescription:
    """構成記述。相手別のポリシーラベル種類数の段階値、蓄積記録数、直近ティック番号。

    安全弁4: 構成記述の非累積性。各サイクルで独立に再計算される。
    過去の構成記述が現在の構成記述に影響する累積構造を持たない。
    """
    user_id: str = ""
    policy_label_type_count_level: str = TypeCountLevel.LEVEL_0.value
    record_count: int = 0
    latest_tick: int = 0
    tick: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "policy_label_type_count_level": self.policy_label_type_count_level,
            "record_count": self.record_count,
            "latest_tick": self.latest_tick,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositionDescription":
        return cls(
            user_id=data.get("user_id", ""),
            policy_label_type_count_level=data.get(
                "policy_label_type_count_level", TypeCountLevel.LEVEL_0.value
            ),
            record_count=data.get("record_count", 0),
            latest_tick=data.get("latest_tick", 0),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。安全弁8。"""
    user_id: str = ""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    policy_label_type_count: int = 0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "policy_label_type_count": self.policy_label_type_count,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            user_id=data.get("user_id", ""),
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            policy_label_type_count=data.get("policy_label_type_count", 0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class SituationalSelfPresentationState:
    """状況依存的自己呈示の内部状態。永続化対象。"""

    # 相手別蓄積記録のリスト群（時系列順に蓄積）
    records: list[PresentationRecord] = field(default_factory=list)

    # 相手別索引（user_id -> record_id リスト）
    # other_model_dialogue_learning の相手別索引と同一のデータ構造パターン
    user_index: dict[str, list[str]] = field(default_factory=dict)

    # 相手別構成記述の履歴（FIFO蓄積で有限）
    composition_history: list[CompositionDescription] = field(default_factory=list)

    # 直近処理結果（最新の処理サイクルで生成された構成記述のキャッシュ）
    latest_compositions: dict[str, CompositionDescription] = field(default_factory=dict)

    # 収束監視記録
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # カウンタ
    cycle_count: int = 0
    total_records_added: int = 0
    total_records_pushed_out: int = 0
    total_records_invisible: int = 0

    # 安全弁フラグ
    convergence_warning: bool = False  # 種類数が極端に少ない状態が持続

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "user_index": {k: list(v) for k, v in self.user_index.items()},
            "composition_history": [c.to_dict() for c in self.composition_history],
            "latest_compositions": {
                k: v.to_dict() for k, v in self.latest_compositions.items()
            },
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "cycle_count": self.cycle_count,
            "total_records_added": self.total_records_added,
            "total_records_pushed_out": self.total_records_pushed_out,
            "total_records_invisible": self.total_records_invisible,
            "convergence_warning": self.convergence_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SituationalSelfPresentationState":
        return cls(
            records=[
                PresentationRecord.from_dict(r)
                for r in data.get("records", [])
            ],
            user_index={
                k: list(v) for k, v in data.get("user_index", {}).items()
            },
            composition_history=[
                CompositionDescription.from_dict(c)
                for c in data.get("composition_history", [])
            ],
            latest_compositions={
                k: CompositionDescription.from_dict(v)
                for k, v in data.get("latest_compositions", {}).items()
            },
            convergence_records=[
                ConvergenceRecord.from_dict(c)
                for c in data.get("convergence_records", [])
            ],
            cycle_count=data.get("cycle_count", 0),
            total_records_added=data.get("total_records_added", 0),
            total_records_pushed_out=data.get("total_records_pushed_out", 0),
            total_records_invisible=data.get("total_records_invisible", 0),
            convergence_warning=data.get("convergence_warning", False),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。

        永続化されたデータもセッション境界で鮮度が減衰し、有限の寿命を持つ。
        セッションをまたぐごとに蓄積記録の鮮度が一律に低下し、
        十分に希薄化した記録は不可視化へ移行する。
        """
        remove_ids: set[str] = set()
        for rec in self.records:
            rec.freshness = _clamp(rec.freshness - decay_factor)
            rec.freshness_stage = _freshness_from_value(rec.freshness).value
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
# Configuration
# =============================================================================

@dataclass
class SituationalSelfPresentationConfig:
    """設定。"""
    # 相手別の蓄積上限（上限到達時は最古から押し出す）
    max_records_per_user: int = 50
    # 全体の蓄積上限
    max_records_total: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))
    # 出力テキストのプレビュー長
    text_preview_length: int = 80
    # 構成記述の履歴上限
    max_composition_history: int = 100
    # 収束監視記録の上限
    max_convergence_records: int = 50
    # 鮮度減衰速度
    freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))
    # 相手不在時の追加減衰速度
    absent_user_decay_rate: float = 0.01
    # enrichmentに含める構成記述の件数（現在の対話相手のみ）
    enrichment_count: int = 1
    # 参照情報として提供する直近履歴の件数
    reference_history_count: int = 20


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class SituationalSelfPresentationProcessor:
    """状況依存的自己呈示の認知プロセッサ。

    3段パイプライン:
    1. 相手別記録の受領と蓄積 -- 自己行動記録と相手識別情報を対にして蓄積
    2. 構成記述の生成 -- 相手別のポリシーラベル種類数（段階値）等を記述
    3. 参照情報としての受渡準備 -- enrichmentセクション + READ-ONLYアクセサ

    すべての処理は受動的な記録・記述行為であり、能動的な判断・評価・制御を含まない。
    出力テキストの品質評価、分類・類型化を行わない。
    パターン抽出を行わない。マッピング形成を行わない。
    ポリシー選択パイプライン・バイアス計算・安定化弁への直接入力経路を持たない。
    """

    def __init__(self, config: Optional[SituationalSelfPresentationConfig] = None):
        self._config = config or SituationalSelfPresentationConfig()
        self._state = SituationalSelfPresentationState()

    @property
    def state(self) -> SituationalSelfPresentationState:
        return self._state

    @state.setter
    def state(self, value: SituationalSelfPresentationState) -> None:
        self._state = value

    # ─── Stage 1: 相手別記録の受領と蓄積 ──────────────────────────

    def receive_and_accumulate(
        self,
        user_id: str,
        response_text: str,
        policy_label: str = "",
        tick: int = 0,
    ) -> None:
        """自己行動記録と相手識別情報を対にして蓄積する。

        対話相手が存在しないターン（自発起動など相手なし）では蓄積しない。
        空テキストは記録しない。

        蓄積は相手別に分離して管理する。
        分離管理の構造は他者モデル対話学習の相手別整列の設計思想に倣う。

        すべての記録は等価である。記録間の重み・スコア・優先度を付与しない。
        テキストは生のテキスト文字列として保持し、意味・傾向・パターンを抽出しない。

        Args:
            user_id: 相手識別子。空の場合は蓄積しない。
            response_text: 出力テキスト（自己行動知覚から参照した生テキスト）
            policy_label: そのターンで選択されていたポリシーラベル
            tick: 通知時点のティック番号
        """
        # 対話相手が存在しない場合は蓄積しない
        if not user_id:
            return
        # 空テキストは記録しない
        if not response_text:
            return

        cfg = self._config
        now = time.time()

        # テキストはプレビュー長に切り詰め
        text_preview = response_text[:cfg.text_preview_length]

        record = PresentationRecord(
            user_id=user_id,
            response_text=text_preview,
            policy_label=policy_label,
            tick=tick,
            timestamp=now,
        )

        self._state.records.append(record)
        self._state.total_records_added += 1

        # 相手別索引に追加
        if user_id not in self._state.user_index:
            self._state.user_index[user_id] = []
        self._state.user_index[user_id].append(record.record_id)

        # 相手別上限制御（最古から押し出し）
        self._apply_per_user_pushout(user_id)

        # 全体上限制御（最古から押し出し）
        self._apply_total_pushout()

        logger.debug(
            "Situational self-presentation recorded: user=%s, tick=%d, "
            "policy=%s, text_len=%d",
            user_id, tick, policy_label, len(response_text),
        )

    def _apply_per_user_pushout(self, user_id: str) -> None:
        """相手別の上限数に達した場合、最古の記録から押し出す。"""
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
        """全体の上限数に達した場合、最古の記録から押し出す。"""
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

    # ─── Stage 2: 構成記述の生成 ───────────────────────────────────

    def generate_compositions(self, current_tick: int = 0) -> None:
        """相手別に蓄積された記録群から、構成的特徴を非評価的に記述する。

        記述する内容:
        - 相手別のポリシーラベル種類数（段階値列挙型、behavioral_diversity_description同形式）
        - 相手別の蓄積記録数
        - 相手別の直近記録のティック番号（鮮度の間接的指標）

        記述は「種類数」と「件数」のみで構成する。
        個別のポリシーラベル名を名指しで特徴づける記述を生成しない。
        特定のポリシーラベルの出現頻度を算出しない。
        出現した種類の集合のみを扱う。

        相手間の構成記述を比較・ランキング・差異抽出する処理は含まない。
        各相手の記述は独立して生成され、並置されるのみである。

        安全弁4: 構成記述はその時点の蓄積記録のみから生成される。
        過去の構成記述が現在の構成記述に影響する累積構造を持たない。
        """
        cfg = self._config
        self._state.cycle_count += 1
        now = time.time()

        new_compositions: dict[str, CompositionDescription] = {}

        for uid, record_ids in self._state.user_index.items():
            if not record_ids:
                continue

            # 可視状態（非不可視）の記録のみを対象
            user_records = [
                r for r in self._state.records
                if r.user_id == uid
                and r.freshness_stage != RecordFreshness.INVISIBLE.value
            ]

            if not user_records:
                continue

            # ポリシーラベル種類数を算出
            # 安全弁2: 出現回数を算出しない。set に追加するのみ
            label_types: set[str] = set()
            for rec in user_records:
                if rec.policy_label:
                    label_types.add(rec.policy_label)
            type_count = len(label_types)
            type_level = determine_type_count_level(type_count)

            # 直近記録のティック番号
            latest_tick = max(rec.tick for rec in user_records)

            comp = CompositionDescription(
                user_id=uid,
                policy_label_type_count_level=type_level.value,
                record_count=len(user_records),
                latest_tick=latest_tick,
                tick=current_tick,
                timestamp=now,
            )
            new_compositions[uid] = comp

            # 構成記述の履歴に追加（FIFO）
            self._state.composition_history.append(comp)

        # 直近処理結果を更新
        self._state.latest_compositions = new_compositions

        # 構成記述の履歴上限制御
        if len(self._state.composition_history) > cfg.max_composition_history:
            self._state.composition_history = (
                self._state.composition_history[-cfg.max_composition_history:]
            )

        # 鮮度減衰を適用
        self._apply_freshness_decay(current_tick, now)

        # 安全弁8: 収束監視
        self._monitor_convergence(new_compositions, now)

        logger.debug(
            "Situational self-presentation compositions: cycle=%d, "
            "users=%d, total_records=%d",
            self._state.cycle_count,
            len(new_compositions),
            len(self._state.records),
        )

    def _apply_freshness_decay(self, current_tick: int, now: float) -> None:
        """蓄積記録の鮮度を時間経過に伴い段階的に減衰させる。

        基本減衰と相手不在時の追加減衰を行う。
        十分に希薄化した記録は不可視化へ移行する。
        """
        cfg = self._config

        # 現在のサイクルで更新された相手を特定
        active_users = set(self._state.latest_compositions.keys())

        newly_invisible = 0
        for rec in self._state.records:
            if rec.freshness_stage == RecordFreshness.INVISIBLE.value:
                continue

            # 基本減衰
            decay = cfg.freshness_decay_rate
            # 相手不在時の追加減衰
            if rec.user_id not in active_users:
                decay += cfg.absent_user_decay_rate

            rec.freshness = _clamp(rec.freshness - decay)
            new_stage = _freshness_from_value(rec.freshness)
            rec.freshness_stage = new_stage.value

            if new_stage == RecordFreshness.INVISIBLE:
                newly_invisible += 1

        # 不可視化した記録を除去
        if newly_invisible > 0:
            remove_ids: set[str] = set()
            for rec in self._state.records:
                if rec.freshness_stage == RecordFreshness.INVISIBLE.value:
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

    def _monitor_convergence(
        self,
        compositions: dict[str, CompositionDescription],
        now: float,
    ) -> None:
        """安全弁8: 収束監視。

        特定の相手について蓄積記録のポリシーラベル種類数が極端に少ない状態
        （段階値が最低段階に固定される状態）が持続した場合に検出する。
        検出結果は参照情報としてのみ記録し、自動的な介入・修正を行わない。
        """
        cfg = self._config
        has_convergence = False

        for uid, comp in compositions.items():
            # 記録数が少ない場合は収束判定を行わない
            if comp.record_count < 3:
                continue

            # 種類数段階値が最低段階（LEVEL_0 or LEVEL_1_5で且つrecord_count >= 5）
            type_level = comp.policy_label_type_count_level
            is_low = type_level in (
                TypeCountLevel.LEVEL_0.value,
                TypeCountLevel.LEVEL_1_5.value,
            )

            if is_low and comp.record_count >= 5:
                # 過去の収束記録を確認
                recent_convergence = [
                    c for c in self._state.convergence_records
                    if c.user_id == uid
                ]
                consecutive_low = sum(
                    1 for c in recent_convergence[-3:]
                    if c.convergence_level in (
                        ConvergenceLevel.MODERATE.value,
                        ConvergenceLevel.STRONG.value,
                    )
                )

                # 種類数から収束スコアを計算
                # 記録が多いのに種類数が少ない→収束度が高い
                if type_level == TypeCountLevel.LEVEL_0.value:
                    convergence_score = 0.9
                else:
                    # LEVEL_1_5 の場合、記録数に対する種類数の比率
                    convergence_score = _clamp(
                        1.0 - (5.0 / max(comp.record_count, 1))
                    )

                convergence_level = _convergence_from_score(convergence_score)

                self._state.convergence_records.append(ConvergenceRecord(
                    user_id=uid,
                    convergence_score=convergence_score,
                    convergence_level=convergence_level.value,
                    policy_label_type_count=comp.record_count,
                    cycle=self._state.cycle_count,
                    timestamp=now,
                ))

                if convergence_level in (
                    ConvergenceLevel.MODERATE,
                    ConvergenceLevel.STRONG,
                ):
                    has_convergence = True

        self._state.convergence_warning = has_convergence

        # 収束記録の上限制御
        if len(self._state.convergence_records) > cfg.max_convergence_records:
            self._state.convergence_records = (
                self._state.convergence_records[-cfg.max_convergence_records:]
            )

    # ─── Stage 3: 参照情報としての受渡準備 ──────────────────────────

    def get_enrichment_data(self, user_id: str = "") -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        安全弁6: enrichmentに含めるのは蓄積記録数とポリシーラベル種類数の段階値のみ。
        出力テキストの内容・個別のポリシーラベル名・相手間の比較をenrichmentに含めない。

        安全弁7: ポリシー選択経路の遮断。
        本機能の出力をポリシー選択パイプラインの入力に直接接続しない。

        Args:
            user_id: 現在の対話相手の識別子。指定がある場合はその相手の情報のみを返す。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state

        result: dict[str, Any] = {
            "cycle_count": st.cycle_count,
            "total_records": st.total_records_added,
            "current_record_count": len(st.records),
            "user_count": len([
                uid for uid, rids in st.user_index.items() if rids
            ]),
            "convergence_warning": st.convergence_warning,
        }

        # 現在の対話相手に関する蓄積概要
        if user_id and user_id in st.latest_compositions:
            comp = st.latest_compositions[user_id]
            result["current_user_composition"] = {
                "record_count": comp.record_count,
                "policy_label_type_count_level": comp.policy_label_type_count_level,
                "latest_tick": comp.latest_tick,
            }

        summary_text = get_presentation_summary(st, user_id)
        result["summary_text"] = summary_text

        return result

    def get_user_records(self, user_id: str) -> list[PresentationRecord]:
        """特定の相手に対する蓄積記録をREAD-ONLYで返す。

        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。
        参照行為によって記録が変化することはない。

        Args:
            user_id: 相手識別子

        Returns:
            相手別蓄積記録のリスト（READ-ONLY参照）
        """
        return [
            r for r in self._state.records
            if r.user_id == user_id
            and r.freshness_stage != RecordFreshness.INVISIBLE.value
        ]

    def get_reference_history(self, user_id: str = "") -> list[PresentationRecord]:
        """内省系モジュールがREAD-ONLY参照として利用する時系列的な蓄積履歴。

        直近数件の蓄積記録のリストを返す。
        参照行為によって記録が変化することはない。

        Args:
            user_id: 相手識別子（空の場合は全相手）

        Returns:
            直近の蓄積記録のリスト（READ-ONLY参照）
        """
        cfg = self._config
        records = self._state.records
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        # 可視状態のみ
        visible = [
            r for r in records
            if r.freshness_stage != RecordFreshness.INVISIBLE.value
        ]
        return list(visible[-cfg.reference_history_count:])

    def get_composition_for_user(self, user_id: str) -> Optional[CompositionDescription]:
        """特定の相手の最新構成記述を返す（READ-ONLY参照）。

        Args:
            user_id: 相手識別子

        Returns:
            最新の構成記述。存在しなければNone。
        """
        return self._state.latest_compositions.get(user_id)

    def get_all_compositions(self) -> dict[str, CompositionDescription]:
        """全相手の最新構成記述を返す（READ-ONLY参照）。

        相手間の比較・ランキング・差異抽出は行わない。
        各相手の記述は独立して返される。

        Returns:
            相手識別子 -> 構成記述のマップ
        """
        return dict(self._state.latest_compositions)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "cycle_count": st.cycle_count,
            "total_records_added": st.total_records_added,
            "current_record_count": len(st.records),
            "total_pushed_out": st.total_records_pushed_out,
            "total_invisible": st.total_records_invisible,
            "user_count": len([
                uid for uid, rids in st.user_index.items() if rids
            ]),
            "convergence_warning": st.convergence_warning,
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_presentation_summary(
    state: SituationalSelfPresentationState,
    user_id: str = "",
) -> str:
    """状況依存的自己呈示状態の要約（enrichment用）。

    安全弁6: 蓄積記録数とポリシーラベル種類数の段階値のみ。
    出力テキストの内容・個別のポリシーラベル名を含めない。
    評価判定・行動指示を含まない。
    """
    if state.cycle_count == 0 and not state.records:
        return "自己呈示認知: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    record_count = len(state.records)
    if record_count > 0:
        parts.append(f"蓄積={record_count}")

    active_users = len([
        uid for uid, rids in state.user_index.items() if rids
    ])
    if active_users > 0:
        parts.append(f"相手={active_users}")

    # 現在の対話相手の構成情報
    if user_id and user_id in state.latest_compositions:
        comp = state.latest_compositions[user_id]
        parts.append(f"当相手記録={comp.record_count}")
        parts.append(f"種類段階={comp.policy_label_type_count_level}")

    if state.total_records_pushed_out > 0:
        parts.append(f"押出累計={state.total_records_pushed_out}")

    if state.convergence_warning:
        parts.append("種類数収束")

    return " ".join(parts) if parts else "自己呈示認知: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_situational_self_presentation_processor(
    config: Optional[SituationalSelfPresentationConfig] = None,
) -> SituationalSelfPresentationProcessor:
    """SituationalSelfPresentationProcessor のファクトリ関数。"""
    return SituationalSelfPresentationProcessor(config=config)
