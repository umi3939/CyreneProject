"""
psyche/persistent_commitment.py - 持続的取り組み保持構造

一時的注目選択（transient_goal）からの昇格によってのみ生成される保持項目を管理し、
複数の取り組みの並行保持・強度依存減衰・資源競合・認知記録・弱いバイアス出力を行う。

設計原則 (design_persistent_commitment.md 準拠):
- 唯一の生成経路: transient_goal からの昇格のみ
- 並行保持: 複数の保持項目を同時保持（上限あり）
- 強度依存減衰: 飽和構造により「強いものほど永遠に残る」を防止
- 慣性の時間減衰: 慣性値は強度とは独立して自然減衰（条件1）
- 解除条件: 時間減衰/内部状態変動/競合出現（達成認知は解除トリガーではない、条件2）
- バイアス上限: value_orientation の +-5% 未満（条件3）
- 最大保持期間: 絶対上限（条件5）
- 認知記録: FIFO、READ-ONLY、評価判定なし、パターン抽出禁止

安全弁（6種、条件6）:
1. 単一保持項目集中度の監視
2. 慣性累積上限
3. 同一方向連続保持抑制
4. 最大保持期間の絶対上限
5. バイアス総量上限
6. 全保持項目の強制一括減衰（安全弁の安全弁）

自己強化ループ遮断（4つ）:
1. 再昇格時の初期強度上限 + 既存項目への微小補強上限
2. 慣性値の強度独立時間減衰
3. 同一方向バイアス合計上限
4. 認知記録→判断系の経路遮断
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PersistentCommitmentConfig:
    """持続的取り組み保持構造の設定。"""

    # ─── 保持スロット ───
    max_slots: int = 5                      # 構造的最大上限
    dynamic_slot_base: int = 3              # 動的スロット数の基底値

    # ─── 昇格条件 ───
    min_promotion_strength: float = 0.25    # 昇格に必要な最低残存強度
    min_promotion_ticks: int = 5            # 昇格に必要な最低維持ティック数

    # ─── 強度減衰 ───
    base_decay_rate: float = 0.015          # 基本減衰率
    low_strength_decay_boost: float = 1.5   # 低強度時の減衰加速係数
    high_strength_decay_slow: float = 0.5   # 高強度時の減衰緩和係数
    saturation_threshold: float = 0.85      # 飽和閾値（これ以上で減衰再加速）
    saturation_boost: float = 1.3           # 飽和時の減衰加速係数
    min_strength_threshold: float = 0.05    # この強度以下で解除

    # ─── 慣性減衰 ───
    inertia_decay_rate: float = 0.02        # 慣性の自然減衰率（強度とは独立）

    # ─── 最大保持期間 ───
    max_retention_ticks: int = 200          # 最大保持期間（絶対上限）
    accelerated_decay_multiplier: float = 4.0  # 加速減衰フェーズの倍率

    # ─── 微小補強 ───
    reinforce_amount: float = 0.03          # 同一方向の再注目時の補強量
    reinforce_cap_ratio: float = 0.95       # 初期強度に対する補強上限比率

    # ─── 資源競合 ───
    bandwidth_fluctuation_scale: float = 0.1  # 帯域分配の揺らぎスケール
    bandwidth_deficit_decay_boost: float = 1.5  # 帯域不足時の減衰加速

    # ─── 内部状態距離 ───
    distance_release_threshold: float = 0.7  # 内部状態距離がこれ以上で解除

    # ─── バイアス ───
    max_single_bias: float = 0.10           # 単一保持項目のバイアス上限
    max_total_bias: float = 0.12            # バイアス総量の絶対上限（< vo +-5% = 0.15）
    bias_strength_multiplier: float = 0.12  # 強度→バイアスの変換係数
    superposition_cap: float = 0.14         # transient_goal との重畳上限

    # ─── 認知記録 ───
    max_cognition_records: int = 50         # 認知記録の最大保持数（FIFO）
    enrichment_recent_records: int = 3      # enrichment に含める直近記録数

    # ─── 安全弁 ───
    concentration_threshold: float = 0.7    # 安全弁1: 単一集中度閾値
    concentration_decay_boost: float = 1.5  # 安全弁1: 集中時の減衰加速
    max_total_inertia: float = 3.0          # 安全弁2: 慣性累積上限
    same_direction_threshold: int = 3       # 安全弁3: 同一方向連続閾値
    same_direction_promotion_penalty: float = 0.3  # 安全弁3: 昇格閾値引き上げ量
    consecutive_safety_trigger: int = 3     # 安全弁6: 連続安全弁発動回数閾値
    emergency_decay_ratio: float = 0.15     # 安全弁6: 緊急一括減衰比率

    # ─── 同一方向カウント減衰 ───
    same_direction_decay_interval: int = 10  # 安全弁3補助: 何ティックごとにカウントを1減衰させるか

    # ─── 統計ウィンドウ ───
    recent_activity_window: int = 50         # enrichment公開用: 直近N件の昇格/解除のみ参照


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CommitmentItem:
    """保持項目。"""
    item_id: str = field(default_factory=_gen_id)
    source_goal_id: str = ""
    category: str = ""                      # 昇格元の注目対象のカテゴリ
    direction_signature: dict[str, float] = field(default_factory=dict)
    strength: float = 0.0
    initial_strength: float = 0.0
    inertia: float = 0.0
    promotion_tick: int = 0
    remaining_ticks: int = 200
    bandwidth_share: float = 0.0
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_goal_id": self.source_goal_id,
            "category": self.category,
            "direction_signature": dict(self.direction_signature),
            "strength": self.strength,
            "initial_strength": self.initial_strength,
            "inertia": self.inertia,
            "promotion_tick": self.promotion_tick,
            "remaining_ticks": self.remaining_ticks,
            "bandwidth_share": self.bandwidth_share,
            "released": self.released,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommitmentItem":
        return cls(
            item_id=data.get("item_id", _gen_id()),
            source_goal_id=data.get("source_goal_id", ""),
            category=data.get("category", ""),
            direction_signature=dict(data.get("direction_signature", {})),
            strength=data.get("strength", 0.0),
            initial_strength=data.get("initial_strength", 0.0),
            inertia=data.get("inertia", 0.0),
            promotion_tick=data.get("promotion_tick", 0),
            remaining_ticks=data.get("remaining_ticks", 200),
            bandwidth_share=data.get("bandwidth_share", 0.0),
            released=data.get("released", False),
        )


@dataclass
class CognitionRecord:
    """認知記録。事実の蓄積のみ、評価判定なし。"""
    record_id: str = field(default_factory=_gen_id)
    item_id: str = ""
    record_type: str = ""          # "promotion" / "strength_change" / "release"
    tick: int = 0
    release_reason: str = ""       # "time_decay" / "state_divergence" / "competition" / ""
    residual_strength: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "item_id": self.item_id,
            "record_type": self.record_type,
            "tick": self.tick,
            "release_reason": self.release_reason,
            "residual_strength": self.residual_strength,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitionRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            item_id=data.get("item_id", ""),
            record_type=data.get("record_type", ""),
            tick=data.get("tick", 0),
            release_reason=data.get("release_reason", ""),
            residual_strength=data.get("residual_strength", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class PersistentCommitmentState:
    """持続的取り組み保持構造の内部状態。"""

    items: list[CommitmentItem] = field(default_factory=list)
    cognition_records: list[CognitionRecord] = field(default_factory=list)

    # 安全弁状態（毎ティック再計算、永続化はするが蓄積はしない）
    concentration_ratio: float = 0.0
    total_inertia: float = 0.0
    same_direction_counts: dict[str, int] = field(default_factory=dict)
    bias_total: float = 0.0
    consecutive_safety_triggers: int = 0
    same_direction_last_decay_tick: int = 0  # 同一方向カウントの最終減衰ティック

    # 統計
    total_promotions: int = 0
    total_releases: int = 0
    release_reasons: dict[str, int] = field(default_factory=dict)
    recent_activity_log: list[dict[str, Any]] = field(default_factory=list)  # 直近の昇格/解除ログ

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "cognition_records": [r.to_dict() for r in self.cognition_records],
            "concentration_ratio": self.concentration_ratio,
            "total_inertia": self.total_inertia,
            "same_direction_counts": dict(self.same_direction_counts),
            "bias_total": self.bias_total,
            "consecutive_safety_triggers": self.consecutive_safety_triggers,
            "same_direction_last_decay_tick": self.same_direction_last_decay_tick,
            "total_promotions": self.total_promotions,
            "total_releases": self.total_releases,
            "release_reasons": dict(self.release_reasons),
            "recent_activity_log": list(self.recent_activity_log),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersistentCommitmentState":
        items = [CommitmentItem.from_dict(d) for d in data.get("items", [])]
        records = [CognitionRecord.from_dict(d) for d in data.get("cognition_records", [])]
        return cls(
            items=items,
            cognition_records=records,
            concentration_ratio=data.get("concentration_ratio", 0.0),
            total_inertia=data.get("total_inertia", 0.0),
            same_direction_counts=dict(data.get("same_direction_counts", {})),
            bias_total=data.get("bias_total", 0.0),
            consecutive_safety_triggers=data.get("consecutive_safety_triggers", 0),
            same_direction_last_decay_tick=data.get("same_direction_last_decay_tick", 0),
            total_promotions=data.get("total_promotions", 0),
            total_releases=data.get("total_releases", 0),
            release_reasons=dict(data.get("release_reasons", {})),
            recent_activity_log=list(data.get("recent_activity_log", [])),
        )


# =============================================================================
# Cross-section inputs (8 断面)
# =============================================================================

@dataclass
class CommitmentCrossSectionInputs:
    """8断面からの入力。保持項目の減衰・解除判定に使用する。"""

    # 断面1: 支配的感情と覚醒度の変化量
    dominant_emotion: str = ""
    arousal_delta: float = 0.0

    # 断面2: 文脈連続性の断裂度
    context_disruption: float = 0.0

    # 断面3: 内的推進力の変動幅
    drive_variability: float = 0.0

    # 断面4: 現在の注目と保持項目の方向的距離
    transient_direction_distance: float = 0.0

    # 断面5: 長期傾斜との整合度の変化量
    orientation_alignment_delta: float = 0.0

    # 断面6: 競合する候補の出現状態
    competing_candidate_intensity: float = 0.0

    # 断面7: 責任容量の圧迫度
    responsibility_pressure: float = 0.0

    # 断面8: 非決定性由来の変動量
    scoring_fluctuation_amount: float = 0.0


# =============================================================================
# Processor
# =============================================================================

class PersistentCommitmentProcessor:
    """持続的取り組み保持構造のメインプロセッサ。"""

    def __init__(self, config: Optional[PersistentCommitmentConfig] = None):
        self._config = config or PersistentCommitmentConfig()
        self._state = PersistentCommitmentState()

    @property
    def state(self) -> PersistentCommitmentState:
        return self._state

    @state.setter
    def state(self, value: PersistentCommitmentState) -> None:
        self._state = value

    @property
    def config(self) -> PersistentCommitmentConfig:
        return self._config

    # ─── A. 昇格受け入れ処理 ──────────────────────────────────────

    def try_promote(
        self,
        goal_id: str,
        category: str,
        direction_signature: dict[str, float],
        remaining_strength: float,
        maintained_ticks: int,
        current_tick: int,
    ) -> Optional[CommitmentItem]:
        """transient_goal から昇格条件を判定し、保持項目として格納する。

        唯一の生成経路: transient_goal からのみ昇格可能。

        Args:
            goal_id: 昇格元の注目対象の識別子
            category: 昇格元のカテゴリ
            direction_signature: 方向署名
            remaining_strength: 残存強度
            maintained_ticks: 維持ティック数
            current_tick: 現在のティック

        Returns:
            生成された保持項目。昇格条件を満たさない場合はNone。
        """
        cfg = self._config
        st = self._state

        # 昇格条件チェック
        if remaining_strength < cfg.min_promotion_strength:
            return None
        if maintained_ticks < cfg.min_promotion_ticks:
            return None

        # 安全弁3: 同一方向連続保持抑制
        dir_key = category
        same_count = st.same_direction_counts.get(dir_key, 0)
        effective_min_strength = cfg.min_promotion_strength
        if same_count >= cfg.same_direction_threshold:
            effective_min_strength += cfg.same_direction_promotion_penalty
            if remaining_strength < effective_min_strength:
                logger.debug(
                    "Promotion blocked by same-direction suppression: "
                    "category=%s, count=%d, required=%.3f",
                    category, same_count, effective_min_strength,
                )
                return None

        # 同一方向の既存保持項目がある場合は微小補強
        for item in st.items:
            if item.released:
                continue
            if item.category == category and _direction_similarity(
                item.direction_signature, direction_signature
            ) > 0.7:
                # ループ遮断1: 微小補強（初期強度を超えない）
                reinforce = min(
                    cfg.reinforce_amount,
                    item.initial_strength * cfg.reinforce_cap_ratio - item.strength,
                )
                if reinforce > 0:
                    item.strength = min(
                        item.initial_strength * cfg.reinforce_cap_ratio,
                        item.strength + reinforce,
                    )
                    self._add_cognition_record(
                        item.item_id, "strength_change", current_tick,
                        residual_strength=item.strength,
                    )
                return None  # 新規保持項目は追加しない

        # スロット管理
        active_items = [it for it in st.items if not it.released]
        dynamic_slots = min(cfg.max_slots, cfg.dynamic_slot_base)
        if len(active_items) >= dynamic_slots:
            # 最弱の項目を解除（資源競合経由）
            weakest = min(active_items, key=lambda it: it.strength)
            if remaining_strength > weakest.strength:
                self._release_item(weakest, "competition", current_tick)
            else:
                return None  # 新規昇格の強度が最弱以下なら昇格拒否

        # 保持項目を生成
        initial_strength = _clamp(remaining_strength)
        initial_inertia = _clamp(maintained_ticks / 50.0, 0.1, 1.0)

        item = CommitmentItem(
            source_goal_id=goal_id,
            category=category,
            direction_signature=dict(direction_signature),
            strength=initial_strength,
            initial_strength=initial_strength,
            inertia=initial_inertia,
            promotion_tick=current_tick,
            remaining_ticks=cfg.max_retention_ticks,
            bandwidth_share=0.0,
            released=False,
        )

        st.items.append(item)
        st.total_promotions += 1
        self._log_recent_activity("promotion", current_tick)

        # 同一方向カウント更新
        st.same_direction_counts[dir_key] = same_count + 1

        # 安全弁2: 慣性累積上限チェック
        self._enforce_inertia_cap()

        # 認知記録
        self._add_cognition_record(
            item.item_id, "promotion", current_tick,
            residual_strength=initial_strength,
        )

        logger.debug(
            "Commitment promoted: id=%s, category=%s, strength=%.3f, inertia=%.3f",
            item.item_id, category, initial_strength, initial_inertia,
        )

        return item

    # ─── B-E. ティック処理（減衰・解除・資源競合） ─────────────────

    def tick(
        self,
        inputs: CommitmentCrossSectionInputs,
        current_tick: int,
    ) -> None:
        """毎ティックの保持項目管理処理。

        C. 減衰処理、D. 解除判定、E. 資源競合を実行する。

        Args:
            inputs: 8断面からの入力
            current_tick: 現在のティック
        """
        cfg = self._config
        st = self._state
        active_items = [it for it in st.items if not it.released]

        if not active_items:
            # 保持項目なしの状態は正常
            st.concentration_ratio = 0.0
            st.total_inertia = 0.0
            st.bias_total = 0.0
            return

        # ── E. 資源競合処理（帯域分配） ──
        self._allocate_bandwidth(active_items, inputs)

        # ── C. 減衰処理 ──
        for item in active_items:
            self._apply_decay(item, inputs, current_tick)

        # ── D. 解除判定 ──
        for item in list(active_items):
            if item.released:
                continue
            release_reason = self._check_release(item, inputs)
            if release_reason:
                self._release_item(item, release_reason, current_tick)

        # ── 安全弁の更新 ──
        self._update_safety_valves(current_tick)

    # ─── C. 減衰処理 ──────────────────────────────────────────────

    def _apply_decay(
        self,
        item: CommitmentItem,
        inputs: CommitmentCrossSectionInputs,
        current_tick: int,
    ) -> None:
        """保持項目の強度と慣性を減衰させる。"""
        cfg = self._config

        # 残りティック減少
        item.remaining_ticks -= 1

        # ── 強度の減衰 ──
        decay_rate = cfg.base_decay_rate

        # 強度依存の減衰率変動（飽和構造）
        if item.strength < 0.3:
            decay_rate *= cfg.low_strength_decay_boost
        elif item.strength > cfg.saturation_threshold:
            # 飽和: 高すぎると減衰加速
            decay_rate *= cfg.saturation_boost
        else:
            # 中程度: 減衰緩和
            decay_rate *= cfg.high_strength_decay_slow

        # 安全弁4: 最大保持期間超過で加速減衰
        if item.remaining_ticks <= 0:
            decay_rate *= cfg.accelerated_decay_multiplier

        # 帯域不足による減衰加速
        if item.bandwidth_share < 0.1:
            decay_rate *= cfg.bandwidth_deficit_decay_boost

        # 安全弁1: 集中度監視による減衰加速
        if self._state.concentration_ratio > cfg.concentration_threshold:
            active_items = [it for it in self._state.items if not it.released]
            if active_items:
                strongest = max(active_items, key=lambda it: it.strength)
                if item.item_id == strongest.item_id:
                    decay_rate *= cfg.concentration_decay_boost

        # 慣性による減衰抵抗（慣性が高いほど減衰が緩やか）
        inertia_factor = max(0.3, 1.0 - item.inertia * 0.5)
        decay_rate *= inertia_factor

        # 強度減衰を適用
        item.strength = max(0.0, item.strength - decay_rate)

        # ── 慣性の減衰（強度とは独立） ──
        # ループ遮断2: 慣性は外部から停止できない
        item.inertia = max(0.0, item.inertia - cfg.inertia_decay_rate)

    # ─── D. 解除判定 ──────────────────────────────────────────────

    def _check_release(
        self,
        item: CommitmentItem,
        inputs: CommitmentCrossSectionInputs,
    ) -> str:
        """解除条件をチェックし、該当する理由を返す。空文字なら解除なし。

        解除条件は3つ（達成認知は含めない、条件2）:
        - 時間減衰による自然解除
        - 内部状態変動による解除
        - 競合出現による解除（帯域不足からの衰退）
        """
        cfg = self._config

        # 時間減衰: 強度が下限を下回った
        if item.strength < cfg.min_strength_threshold:
            return "time_decay"

        # 内部状態変動: 8断面の距離が閾値超過
        distance = self._compute_state_distance(item, inputs)
        if distance > cfg.distance_release_threshold:
            return "state_divergence"

        # 競合出現: 帯域を完全に失った
        if item.bandwidth_share <= 0.001:
            return "competition"

        return ""

    def _compute_state_distance(
        self,
        item: CommitmentItem,
        inputs: CommitmentCrossSectionInputs,
    ) -> float:
        """保持項目と現在の内部状態との構造的距離を計算する。"""
        # 8断面の距離成分を等しい重みで合成
        components = [
            abs(inputs.arousal_delta) * 0.5,
            inputs.context_disruption,
            inputs.drive_variability,
            inputs.transient_direction_distance,
            abs(inputs.orientation_alignment_delta),
            inputs.competing_candidate_intensity,
            inputs.responsibility_pressure * 0.5,
            inputs.scoring_fluctuation_amount,
        ]

        if not components:
            return 0.0

        # 平均と最大の中間
        avg = sum(components) / len(components)
        mx = max(components)
        return _clamp((avg + mx) / 2.0)

    # ─── E. 資源競合処理 ──────────────────────────────────────────

    def _allocate_bandwidth(
        self,
        active_items: list[CommitmentItem],
        inputs: CommitmentCrossSectionInputs,
    ) -> None:
        """保持項目間で注意帯域を分配する。揺らぎ付き動的分配。"""
        if not active_items:
            return

        cfg = self._config
        import hashlib
        import struct

        total_strength = sum(it.strength for it in active_items)
        if total_strength <= 0:
            for it in active_items:
                it.bandwidth_share = 0.0
            return

        for item in active_items:
            # 基本分配比率（強度比）
            base_share = item.strength / total_strength

            # 揺らぎの導入（scoring_fluctuation と同様のアプローチ）
            raw = f"{item.item_id}|{inputs.scoring_fluctuation_amount:.6f}|{time.time():.4f}"
            digest = hashlib.sha256(raw.encode("utf-8")).digest()
            int_val = struct.unpack(">Q", digest[:8])[0]
            noise = ((int_val / (2**64 - 1)) * 2.0 - 1.0) * cfg.bandwidth_fluctuation_scale

            item.bandwidth_share = _clamp(base_share + noise, 0.0, 1.0)

        # 正規化（合計を1.0に）
        total_share = sum(it.bandwidth_share for it in active_items)
        if total_share > 0:
            for it in active_items:
                it.bandwidth_share /= total_share

    # ─── 解除処理 ─────────────────────────────────────────────────

    def _release_item(
        self,
        item: CommitmentItem,
        reason: str,
        current_tick: int,
    ) -> None:
        """保持項目を解除する。"""
        item.released = True
        self._state.total_releases += 1
        self._log_recent_activity("release", current_tick, reason=reason)

        # 解除理由の統計
        counts = self._state.release_reasons
        counts[reason] = counts.get(reason, 0) + 1

        # 認知記録
        self._add_cognition_record(
            item.item_id, "release", current_tick,
            release_reason=reason,
            residual_strength=item.strength,
        )

        logger.debug(
            "Commitment released: id=%s, reason=%s, residual=%.3f",
            item.item_id, reason, item.strength,
        )

    # ─── 直近活動ログ ───────────────────────────────────────────

    def _log_recent_activity(
        self,
        activity_type: str,
        current_tick: int,
        reason: str = "",
    ) -> None:
        """直近活動ログに記録する。enrichment公開用のスライディングウィンドウ。"""
        cfg = self._config
        st = self._state
        st.recent_activity_log.append({
            "type": activity_type,  # "promotion" or "release"
            "tick": current_tick,
            "reason": reason,
        })
        # ウィンドウサイズ制限（FIFO）
        if len(st.recent_activity_log) > cfg.recent_activity_window:
            excess = len(st.recent_activity_log) - cfg.recent_activity_window
            st.recent_activity_log = st.recent_activity_log[excess:]

    # ─── F. 認知記録処理 ──────────────────────────────────────────

    def _add_cognition_record(
        self,
        item_id: str,
        record_type: str,
        tick: int,
        release_reason: str = "",
        residual_strength: float = 0.0,
    ) -> None:
        """認知記録を追加する。FIFO、上限付き。"""
        cfg = self._config
        st = self._state

        record = CognitionRecord(
            item_id=item_id,
            record_type=record_type,
            tick=tick,
            release_reason=release_reason,
            residual_strength=residual_strength,
        )
        st.cognition_records.append(record)

        # FIFO: 上限超過で最古を押し出し
        if len(st.cognition_records) > cfg.max_cognition_records:
            excess = len(st.cognition_records) - cfg.max_cognition_records
            st.cognition_records = st.cognition_records[excess:]

    # ─── G. バイアス出力 ──────────────────────────────────────────

    def compute_bias(
        self,
        candidate: dict[str, Any],
    ) -> float:
        """単一の候補に対するバイアスを計算する。

        保持項目の方向情報に基づき、整合する候補にバイアスを加算する。
        認知記録はバイアス計算に使用しない（経路遮断4）。

        Args:
            candidate: ポリシー候補

        Returns:
            バイアス値（上限制限済み）
        """
        cfg = self._config
        active_items = [it for it in self._state.items if not it.released]

        if not active_items:
            return 0.0

        total_bias = 0.0

        # 方向別バイアス集計（ループ遮断3: 同一方向のバイアス合計に上限）
        direction_biases: dict[str, float] = {}

        for item in active_items:
            alignment = _compute_candidate_alignment(candidate, item)
            raw_bias = item.strength * cfg.bias_strength_multiplier * alignment

            # 単一保持項目のバイアス上限
            raw_bias = _clamp(raw_bias, -cfg.max_single_bias, cfg.max_single_bias)

            # 方向別集計
            dir_key = item.category or "unknown"
            current = direction_biases.get(dir_key, 0.0)
            direction_biases[dir_key] = _clamp(
                current + raw_bias,
                -cfg.max_single_bias,
                cfg.max_single_bias,
            )

        total_bias = sum(direction_biases.values())

        # 安全弁5: バイアス総量上限
        total_bias = _clamp(total_bias, -cfg.max_total_bias, cfg.max_total_bias)

        return total_bias

    def apply_bias_to_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """候補リスト全体にバイアスを適用する。

        Args:
            candidates: ポリシー候補リスト

        Returns:
            バイアスが適用された候補リスト（新しいリスト）
        """
        if not candidates:
            return []

        active_items = [it for it in self._state.items if not it.released]
        if not active_items:
            return candidates

        cfg = self._config
        result: list[dict[str, Any]] = []

        for candidate in candidates:
            new_cand = candidate.copy()
            bias = self.compute_bias(candidate)

            original_score = new_cand.get("_score", 0.0)
            new_cand["_score"] = round(original_score + bias, 6)
            new_cand["_persistent_commitment_bias"] = round(bias, 6)

            result.append(new_cand)

        # 再ソート（スコア降順）
        result.sort(key=lambda c: c.get("_score", 0), reverse=True)

        # 安全弁状態更新
        self._state.bias_total = sum(
            abs(c.get("_persistent_commitment_bias", 0.0)) for c in result
        ) / max(len(result), 1)

        return result

    # ─── 安全弁 ───────────────────────────────────────────────────

    def _update_safety_valves(self, current_tick: int) -> None:
        """安全弁状態を毎ティック更新する。"""
        cfg = self._config
        st = self._state
        active_items = [it for it in st.items if not it.released]

        if not active_items:
            st.concentration_ratio = 0.0
            st.total_inertia = 0.0
            st.consecutive_safety_triggers = 0
            return

        # 安全弁1: 単一保持項目集中度
        total_strength = sum(it.strength for it in active_items)
        if total_strength > 0:
            strongest = max(active_items, key=lambda it: it.strength)
            st.concentration_ratio = strongest.strength / total_strength
        else:
            st.concentration_ratio = 0.0

        # 安全弁2: 慣性累積
        st.total_inertia = sum(it.inertia for it in active_items)

        # 安全弁3補助: 同一方向カウントの時間減衰
        if (current_tick - st.same_direction_last_decay_tick) >= cfg.same_direction_decay_interval:
            keys_to_remove: list[str] = []
            for dir_key in st.same_direction_counts:
                st.same_direction_counts[dir_key] = max(
                    0, st.same_direction_counts[dir_key] - 1
                )
                if st.same_direction_counts[dir_key] <= 0:
                    keys_to_remove.append(dir_key)
            for key in keys_to_remove:
                del st.same_direction_counts[key]
            st.same_direction_last_decay_tick = current_tick

        # 安全弁6: 連続安全弁発動の検出
        any_triggered = False
        if st.concentration_ratio > cfg.concentration_threshold:
            any_triggered = True
        if st.total_inertia > cfg.max_total_inertia:
            any_triggered = True

        if any_triggered:
            st.consecutive_safety_triggers += 1
        else:
            st.consecutive_safety_triggers = max(0, st.consecutive_safety_triggers - 1)

        # 安全弁6: 緊急一括減衰
        if st.consecutive_safety_triggers >= cfg.consecutive_safety_trigger:
            for item in active_items:
                item.strength *= (1.0 - cfg.emergency_decay_ratio)
            st.consecutive_safety_triggers = 0
            logger.debug("Emergency mass decay triggered at tick %d", current_tick)

    def _enforce_inertia_cap(self) -> None:
        """安全弁2: 慣性累積上限を適用する。"""
        cfg = self._config
        st = self._state
        active_items = [it for it in st.items if not it.released]

        total_inertia = sum(it.inertia for it in active_items)
        if total_inertia > cfg.max_total_inertia:
            scale = cfg.max_total_inertia / total_inertia
            for item in active_items:
                item.inertia *= scale

    # ─── enrichment ───────────────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment用のデータを返す。

        保持項目を段階値で等価に列挙する。特定の項目を推奨・強調しない。
        認知記録の直近数件を等価に列挙する。
        """
        cfg = self._config
        st = self._state
        active_items = [it for it in st.items if not it.released]

        item_entries: list[dict[str, Any]] = []
        for item in active_items:
            # 段階値記述
            if item.strength > 0.7:
                strength_level = "高"
            elif item.strength > 0.4:
                strength_level = "中"
            elif item.strength > 0.15:
                strength_level = "低"
            else:
                strength_level = "微"

            item_entries.append({
                "category": item.category,
                "strength_level": strength_level,
                "strength": round(item.strength, 3),
                "remaining_ratio": round(
                    item.remaining_ticks / cfg.max_retention_ticks, 2
                ) if cfg.max_retention_ticks > 0 else 0.0,
            })

        # 認知記録: 直近N件を等価に列挙
        recent_records: list[dict[str, Any]] = []
        recent = st.cognition_records[-cfg.enrichment_recent_records:] if st.cognition_records else []
        for rec in recent:
            recent_records.append({
                "type": rec.record_type,
                "release_reason": rec.release_reason,
                "tick": rec.tick,
            })

        summary_text = get_commitment_summary(st)

        return {
            "active_count": len(active_items),
            "items": item_entries,
            "recent_records": recent_records,
            "summary_text": summary_text,
        }

    # ─── 認知記録の読み取り専用アクセス ───────────────────────────

    def get_cognition_records(self) -> list[CognitionRecord]:
        """内省系からのREAD-ONLY参照。判断系への接続は遮断されている。"""
        return list(self._state.cognition_records)

    # ─── ロード時の検証 ───────────────────────────────────────────

    def validate_on_load(self) -> None:
        """ロード時に最大保持期間超過の保持項目を検証する。"""
        for item in self._state.items:
            if not item.released and item.remaining_ticks <= 0:
                # 加速減衰フェーズに移行
                logger.debug(
                    "Commitment item %s: max retention exceeded on load, "
                    "entering accelerated decay",
                    item.item_id,
                )
                # remaining_ticks を 0 にしておくことで、次の tick で加速減衰が適用される

    # ─── サマリ ───────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        active = [it for it in st.items if not it.released]

        # 直近ウィンドウからの統計
        recent_promotions = sum(
            1 for e in st.recent_activity_log if e.get("type") == "promotion"
        )
        recent_releases = sum(
            1 for e in st.recent_activity_log if e.get("type") == "release"
        )
        recent_release_reasons: dict[str, int] = {}
        for entry in st.recent_activity_log:
            if entry.get("type") == "release" and entry.get("reason"):
                r = entry["reason"]
                recent_release_reasons[r] = recent_release_reasons.get(r, 0) + 1

        return {
            "active_count": len(active),
            "recent_promotions": recent_promotions,
            "recent_releases": recent_releases,
            "recent_release_reasons": recent_release_reasons,
            "concentration_ratio": round(st.concentration_ratio, 3),
            "total_inertia": round(st.total_inertia, 3),
        }


# =============================================================================
# Module-level helpers
# =============================================================================

def _direction_similarity(
    sig_a: dict[str, float],
    sig_b: dict[str, float],
) -> float:
    """2つの方向署名の類似度を計算する。"""
    if not sig_a or not sig_b:
        return 0.0

    keys = set(sig_a.keys()) | set(sig_b.keys())
    if not keys:
        return 0.0

    dot = sum(sig_a.get(k, 0.0) * sig_b.get(k, 0.0) for k in keys)
    mag_a = sum(v ** 2 for v in sig_a.values()) ** 0.5
    mag_b = sum(v ** 2 for v in sig_b.values()) ** 0.5

    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0

    return _clamp(dot / (mag_a * mag_b), -1.0, 1.0)


def _compute_candidate_alignment(
    candidate: dict[str, Any],
    item: CommitmentItem,
) -> float:
    """候補と保持項目の方向的整合度を計算する。-1.0〜1.0。

    方向署名（direction_signature）の余弦類似度のみで判定する。
    状態非依存のハードコード対応表は使用しない。
    """
    alignment = 0.0

    # 方向署名との比較（唯一のアライメント判定経路）
    candidate_direction = candidate.get("direction", {})
    if candidate_direction and item.direction_signature:
        alignment = _direction_similarity(item.direction_signature, candidate_direction)

    return _clamp(alignment, -1.0, 1.0)


# =============================================================================
# Summary
# =============================================================================

def get_commitment_summary(state: PersistentCommitmentState) -> str:
    """持続的取り組み保持状態の要約（enrichment用）。

    等価に列挙し、特定の保持項目を強調しない。
    """
    active = [it for it in state.items if not it.released]

    if not active:
        return "持続保持: 待機中"

    parts: list[str] = []
    parts.append(f"保持中={len(active)}")

    # 各保持項目を等価に列挙
    for item in active:
        if item.strength > 0.7:
            level = "高"
        elif item.strength > 0.4:
            level = "中"
        elif item.strength > 0.15:
            level = "低"
        else:
            level = "微"
        parts.append(f"{item.category}({level})")

    # 解除理由の分布（直近ウィンドウから算出）
    recent_release_reasons: dict[str, int] = {}
    for entry in state.recent_activity_log:
        if entry.get("type") == "release" and entry.get("reason"):
            r = entry["reason"]
            recent_release_reasons[r] = recent_release_reasons.get(r, 0) + 1
    if recent_release_reasons:
        reason_parts = []
        for reason, count in recent_release_reasons.items():
            reason_parts.append(f"{reason}={count}")
        parts.append("近況解除=" + ",".join(reason_parts))

    return " ".join(parts)


# =============================================================================
# Factory
# =============================================================================

def create_persistent_commitment_processor(
    config: Optional[PersistentCommitmentConfig] = None,
) -> PersistentCommitmentProcessor:
    """PersistentCommitmentProcessor のファクトリ関数。"""
    return PersistentCommitmentProcessor(config=config)
