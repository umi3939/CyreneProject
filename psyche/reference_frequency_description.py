"""
psyche/reference_frequency_description.py - 参照頻度の構造的記述

複数の記憶系構造にわたって散在する参照回数を横断的に読み取り、
その分布の構造的特徴を非評価的に記述する集約層。

設計原則 (design_reference_frequency_description.md 準拠):
- 12箇所の参照回数を読み取り専用で収集する
- 参照回数の多寡から記録間の優劣を確定しない
- 安定化を推奨しない。固定化を矯正しない
- 経験に意味や価値を付与しない
- 忘却処理に関与しない（経路遮断）
- 外部出力層への直接露出を行わない（経路遮断）
- 閾値による区分を行わない
- 想起経路選択への影響を遮断

断面構造:
  - 生成時刻
  - 構造別の参照回数一覧（構造種別→各記録の参照回数リスト）
  - 断面全体の集中度（少数集中か多数分散かの度合い）
  - 構造別の参照偏在度（どの構造種別に参照が偏っているかの度合い）

断面履歴:
  - 過去に生成された参照断面の時系列リスト（FIFO、有限上限）

変動記述:
  - 直近の断面と過去の断面群を比較した集中度・偏在度の変動方向・変動幅

安全弁（5種）:
  1. 全記録等価維持保証 — 参照回数の分布記述が記録間の重み付け・順位付け・選別に
     使用されることを構造的に防止する
  2. 評価的変換の禁止 — 集中度・偏在度は数値的な分布特徴であり、「良い分布」「悪い分布」
     を示す評価軸を持たない
  3. 累積的傾向の抑制 — 変動記述は断面間の差分から導出されるのみであり、
     「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない
  4. 断面履歴の有限性 — 断面履歴には保持上限があり、古い断面は消失する
  5. 出力経路の限定と不拡張 — 出力先は内省系構造への参照情報に限定される。
     enrichment出力経路を持たない。運用中に出力先を動的に追加する仕組みを持たない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReferenceFrequencyConfig:
    """参照頻度記述モジュールの設定。"""

    # 断面履歴の最大保持件数（安全弁4: 有限性）
    max_snapshot_history: int = 30

    # 変動記述で比較する過去断面の件数
    variation_comparison_count: int = 5


# =============================================================================
# 構造識別子（12箇所）
# =============================================================================

# 設計書で列挙された12箇所の参照回数源を構造種別として定義する。
# 「高頻度」「低頻度」などの段階的区分は行わない（設計書の禁止事項）。

STRUCTURE_EPISODIC = "episodic_memory"
STRUCTURE_BINDING = "emotional_memory_binding"
STRUCTURE_BINDING_TRACE = "emotional_memory_binding_trace"
STRUCTURE_INTROSPECTION = "introspection_consumption"
STRUCTURE_EXPECTATION = "expectation_formation"
STRUCTURE_MOTIVE_ENTRY = "intrinsic_motivation_entry"
STRUCTURE_MOTIVE_IMPULSE = "intrinsic_motivation_impulse"
STRUCTURE_NARRATIVE = "self_narrative"
STRUCTURE_OTHER_MODEL = "other_agent_model"
STRUCTURE_SELF_REFERENCE = "self_reference"
STRUCTURE_ACTION_RESULT = "action_result_observation"
STRUCTURE_DIALOGUE_LEARNING = "other_model_dialogue_learning"
STRUCTURE_FORGETTING = "memory_forgetting_fixation"

ALL_STRUCTURE_KEYS = [
    STRUCTURE_EPISODIC,
    STRUCTURE_BINDING,
    STRUCTURE_BINDING_TRACE,
    STRUCTURE_INTROSPECTION,
    STRUCTURE_EXPECTATION,
    STRUCTURE_MOTIVE_ENTRY,
    STRUCTURE_MOTIVE_IMPULSE,
    STRUCTURE_NARRATIVE,
    STRUCTURE_OTHER_MODEL,
    STRUCTURE_SELF_REFERENCE,
    STRUCTURE_ACTION_RESULT,
    STRUCTURE_DIALOGUE_LEARNING,
    STRUCTURE_FORGETTING,
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ReferenceSnapshot:
    """参照断面。ある時点で収集された全構造の参照回数を格納した断面。

    断面は生成後に変更されない（不変）。
    安全弁1: 全記録等価。参照回数の多寡は記載するが、記録間の
    重み付け・順位付け・選別の信号は含まない。
    安全弁2: 評価的変換の禁止。集中度・偏在度は数値的な分布特徴であり、
    「良い分布」「悪い分布」を示す評価軸を持たない。
    """

    # 生成時刻
    timestamp: float = 0.0

    # 構造別の参照回数一覧
    # key=構造種別識別子, value=各記録の参照回数のリスト
    structure_counts: dict[str, list[int]] = field(default_factory=dict)

    # 断面全体の集中度（0.0=完全分散, 1.0=完全集中）
    # 安全弁2: この値は評価的意味を持たない
    concentration: float = 0.0

    # 構造別の参照偏在度（0.0=完全均等, 1.0=単一構造に全集中）
    # 安全弁2: この値は評価的意味を持たない
    structural_bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "structure_counts": {k: list(v) for k, v in self.structure_counts.items()},
            "concentration": self.concentration,
            "structural_bias": self.structural_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceSnapshot":
        return cls(
            timestamp=data.get("timestamp", 0.0),
            structure_counts={
                k: list(v) for k, v in data.get("structure_counts", {}).items()
            },
            concentration=data.get("concentration", 0.0),
            structural_bias=data.get("structural_bias", 0.0),
        )


@dataclass
class VariationDescription:
    """変動記述。直近の断面と過去の断面群を比較した変動の記録。

    安全弁3: 累積的傾向の抑制。変動記述は断面間の差分から導出されるのみであり、
    「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない。
    新しい断面が生成されるたびに、断面履歴から再導出される。
    """

    # 集中度の変動方向（正=集中方向, 負=分散方向, 0=変化なし）
    concentration_direction: float = 0.0

    # 集中度の変動幅（絶対値）
    concentration_magnitude: float = 0.0

    # 偏在度の変動方向（正=偏在方向, 負=均等化方向, 0=変化なし）
    bias_direction: float = 0.0

    # 偏在度の変動幅（絶対値）
    bias_magnitude: float = 0.0

    # 比較に使用した過去断面の件数
    comparison_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concentration_direction": self.concentration_direction,
            "concentration_magnitude": self.concentration_magnitude,
            "bias_direction": self.bias_direction,
            "bias_magnitude": self.bias_magnitude,
            "comparison_count": self.comparison_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariationDescription":
        return cls(
            concentration_direction=data.get("concentration_direction", 0.0),
            concentration_magnitude=data.get("concentration_magnitude", 0.0),
            bias_direction=data.get("bias_direction", 0.0),
            bias_magnitude=data.get("bias_magnitude", 0.0),
            comparison_count=data.get("comparison_count", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class ReferenceFrequencyState:
    """参照頻度記述モジュールの内部状態。

    安全弁4: 断面履歴の有限性。保持上限あり、古い断面は消失する。
    """

    # 断面履歴（時系列順、先入先出）
    snapshot_history: list[ReferenceSnapshot] = field(default_factory=list)

    # 直近の変動記述（断面が2件以上ある場合のみ有効）
    latest_variation: Optional[VariationDescription] = None

    # 累積カウンタ（診断情報のみ、処理分岐に使用しない）
    total_snapshots_generated: int = 0
    total_snapshots_expired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_history": [s.to_dict() for s in self.snapshot_history],
            "latest_variation": self.latest_variation.to_dict() if self.latest_variation else None,
            "total_snapshots_generated": self.total_snapshots_generated,
            "total_snapshots_expired": self.total_snapshots_expired,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceFrequencyState":
        history = [
            ReferenceSnapshot.from_dict(s)
            for s in data.get("snapshot_history", [])
        ]
        var_data = data.get("latest_variation")
        variation = VariationDescription.from_dict(var_data) if var_data else None
        return cls(
            snapshot_history=history,
            latest_variation=variation,
            total_snapshots_generated=data.get("total_snapshots_generated", 0),
            total_snapshots_expired=data.get("total_snapshots_expired", 0),
        )


# =============================================================================
# 収集: 各構造から参照回数を読み取り専用で収集する
# =============================================================================

def collect_reference_counts(
    *,
    episodic_store: Any = None,
    binding_store: Any = None,
    consumption_store: Any = None,
    expectation_store: Any = None,
    motive_store: Any = None,
    narrative_state: Any = None,
    other_model_store: Any = None,
    self_reference_state: Any = None,
    action_result_state: Any = None,
    dialogue_learning_state: Any = None,
    forgetting_state: Any = None,
) -> dict[str, list[int]]:
    """各構造から参照回数の現在値を読み取り専用で収集する。

    本関数は参照回数を増減させる権限を持たない。
    各構造のオブジェクトは Optional[Any] + duck typing で受け取る（循環import回避）。
    安全弁1: 全記録等価。収集した参照回数にフィルタリング・選別・順位付けを行わない。

    Returns:
        構造種別識別子 → 各記録のreference_countリスト の辞書
    """
    result: dict[str, list[int]] = {}

    # 1. エピソード記憶構造の各記録
    if episodic_store is not None:
        episodes = getattr(episodic_store, "episodes", ())
        result[STRUCTURE_EPISODIC] = [
            getattr(ep, "reference_count", 0) for ep in episodes
        ]
    else:
        result[STRUCTURE_EPISODIC] = []

    # 2. 感情記憶結合構造の各結合記録
    if binding_store is not None:
        bindings = getattr(binding_store, "bindings", ())
        result[STRUCTURE_BINDING] = [
            getattr(b, "reference_count", 0) for b in bindings
        ]
        # 3. 感情記憶結合構造の各痕跡記録
        traces_counts: list[int] = []
        for b in bindings:
            for tr in getattr(b, "traces", ()):
                traces_counts.append(getattr(tr, "reference_count", 0))
        result[STRUCTURE_BINDING_TRACE] = traces_counts
    else:
        result[STRUCTURE_BINDING] = []
        result[STRUCTURE_BINDING_TRACE] = []

    # 4. 内省消費構造の各断片記録
    if consumption_store is not None:
        fragments = getattr(consumption_store, "fragments", ())
        result[STRUCTURE_INTROSPECTION] = [
            getattr(f, "reference_count", 0) for f in fragments
        ]
    else:
        result[STRUCTURE_INTROSPECTION] = []

    # 5. 期待形成構造の各候補記録
    if expectation_store is not None:
        expectations = getattr(expectation_store, "expectations", ())
        result[STRUCTURE_EXPECTATION] = [
            getattr(e, "reference_count", 0) for e in expectations
        ]
    else:
        result[STRUCTURE_EXPECTATION] = []

    # 6. 内発動機構造の各動機記録
    if motive_store is not None:
        entries = getattr(motive_store, "entries", ())
        result[STRUCTURE_MOTIVE_ENTRY] = [
            getattr(e, "reference_count", 0) for e in entries
        ]
        # 7. 内発動機構造の各衝動記録
        impulse_counts: list[int] = []
        for e in entries:
            for imp in getattr(e, "impulses", ()):
                impulse_counts.append(getattr(imp, "reference_count", 0))
        result[STRUCTURE_MOTIVE_IMPULSE] = impulse_counts
    else:
        result[STRUCTURE_MOTIVE_ENTRY] = []
        result[STRUCTURE_MOTIVE_IMPULSE] = []

    # 8. 自己物語構造の各断片記録
    if narrative_state is not None:
        frags = getattr(narrative_state, "fragments", ())
        result[STRUCTURE_NARRATIVE] = [
            getattr(f, "reference_count", 0) for f in frags
        ]
    else:
        result[STRUCTURE_NARRATIVE] = []

    # 9. 他者モデル構造の各仮説記録
    if other_model_store is not None:
        hypotheses = getattr(other_model_store, "hypotheses", ())
        result[STRUCTURE_OTHER_MODEL] = [
            getattr(h, "reference_count", 0) for h in hypotheses
        ]
    else:
        result[STRUCTURE_OTHER_MODEL] = []

    # 10. 自己参照構造の参照回数（単一の整数値）
    if self_reference_state is not None:
        ref_count = getattr(self_reference_state, "reference_count", 0)
        result[STRUCTURE_SELF_REFERENCE] = [ref_count]
    else:
        result[STRUCTURE_SELF_REFERENCE] = []

    # 11. 行動結果観測構造の各行動結果対記録
    if action_result_state is not None:
        pairs = getattr(action_result_state, "pairs", [])
        result[STRUCTURE_ACTION_RESULT] = [
            getattr(p, "reference_count", 0) for p in pairs
        ]
    else:
        result[STRUCTURE_ACTION_RESULT] = []

    # 12. 他者モデル対話学習構造の各学習記録
    if dialogue_learning_state is not None:
        dl_entries = getattr(dialogue_learning_state, "entries", [])
        result[STRUCTURE_DIALOGUE_LEARNING] = [
            getattr(e, "reference_count", 0) for e in dl_entries
        ]
    else:
        result[STRUCTURE_DIALOGUE_LEARNING] = []

    # 13. 記憶忘却固定化構造の各忘却記録
    if forgetting_state is not None:
        series_index = getattr(forgetting_state, "series_index", [])
        result[STRUCTURE_FORGETTING] = [
            getattr(s, "reference_count", 0) for s in series_index
        ]
    else:
        result[STRUCTURE_FORGETTING] = []

    return result


# =============================================================================
# 構造特徴の記述
# =============================================================================

def compute_concentration(all_counts: list[int]) -> float:
    """参照が少数の記録に集中しているか多数に分散しているかの度合いを計算する。

    安全弁2: この値は評価的意味を持たない。
    「良い集中度」「悪い集中度」を示す軸ではない。

    Returns:
        0.0（完全分散または参照なし）〜 1.0（完全集中）
    """
    if not all_counts:
        return 0.0

    total = sum(all_counts)
    if total == 0:
        return 0.0

    n = len(all_counts)
    if n <= 1:
        return 0.0

    # ジニ係数的な集中度指標
    # 全て同一なら 0.0, 一つに全て集中なら 1.0 に近づく
    sorted_counts = sorted(all_counts)
    cumulative_sum = 0.0
    weighted_sum = 0.0
    for i, c in enumerate(sorted_counts):
        cumulative_sum += c
        weighted_sum += (i + 1) * c

    # ジニ係数 = (2 * weighted_sum) / (n * total) - (n + 1) / n
    if total == 0:
        return 0.0
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

    return _clamp(gini)


def compute_structural_bias(structure_counts: dict[str, list[int]]) -> float:
    """どの構造種別に参照が偏在しているかの度合いを計算する。

    安全弁2: この値は評価的意味を持たない。

    Returns:
        0.0（完全均等または参照なし）〜 1.0（単一構造に全集中）
    """
    # 各構造種別の参照合計を算出
    sums: list[int] = []
    for key in ALL_STRUCTURE_KEYS:
        counts = structure_counts.get(key, [])
        sums.append(sum(counts))

    total = sum(sums)
    if total == 0:
        return 0.0

    active_structures = [s for s in sums if s > 0]
    if len(active_structures) <= 1:
        # 参照が1構造にしかない場合、偏在度は最大
        return 1.0 if total > 0 else 0.0

    n = len(sums)
    if n <= 1:
        return 0.0

    # 構造間のジニ係数
    sorted_sums = sorted(sums)
    weighted_sum = 0.0
    for i, s in enumerate(sorted_sums):
        weighted_sum += (i + 1) * s

    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

    return _clamp(gini)


# =============================================================================
# 断面の構成
# =============================================================================

def compose_snapshot(
    structure_counts: dict[str, list[int]],
    timestamp: Optional[float] = None,
) -> ReferenceSnapshot:
    """収集された参照回数群から断面を構成する。

    安全弁1: 全記録等価。断面構成時にフィルタリング・選別を行わない。
    安全弁2: 評価的変換の禁止。

    Args:
        structure_counts: 構造種別→参照回数リスト
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        構成された参照断面（不変）
    """
    ts = timestamp if timestamp is not None else time.time()

    # 全構造の参照回数をフラットに集めて集中度を計算
    all_counts: list[int] = []
    for key in ALL_STRUCTURE_KEYS:
        all_counts.extend(structure_counts.get(key, []))

    concentration = compute_concentration(all_counts)
    structural_bias = compute_structural_bias(structure_counts)

    return ReferenceSnapshot(
        timestamp=ts,
        structure_counts={k: list(v) for k, v in structure_counts.items()},
        concentration=concentration,
        structural_bias=structural_bias,
    )


# =============================================================================
# 変動記述の導出
# =============================================================================

def derive_variation(
    snapshot_history: list[ReferenceSnapshot],
    config: ReferenceFrequencyConfig,
) -> Optional[VariationDescription]:
    """断面履歴から変動記述を導出する。

    安全弁3: 累積的傾向の抑制。変動記述は断面間の差分から導出されるのみ。
    「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない。
    新しい断面が生成されるたびに、断面履歴から再導出される。

    Returns:
        変動記述。断面が2件未満の場合は None。
    """
    if len(snapshot_history) < 2:
        return None

    latest = snapshot_history[-1]

    # 比較対象: 直近を除く末尾N件
    comparison_count = min(
        config.variation_comparison_count,
        len(snapshot_history) - 1,
    )
    past_snapshots = snapshot_history[-(comparison_count + 1):-1]

    if not past_snapshots:
        return None

    # 過去の集中度・偏在度の平均
    past_concentration_avg = sum(s.concentration for s in past_snapshots) / len(past_snapshots)
    past_bias_avg = sum(s.structural_bias for s in past_snapshots) / len(past_snapshots)

    # 変動方向と幅
    concentration_diff = latest.concentration - past_concentration_avg
    bias_diff = latest.structural_bias - past_bias_avg

    return VariationDescription(
        concentration_direction=concentration_diff,
        concentration_magnitude=abs(concentration_diff),
        bias_direction=bias_diff,
        bias_magnitude=abs(bias_diff),
        comparison_count=len(past_snapshots),
    )


# =============================================================================
# メイン処理: 断面生成と履歴管理
# =============================================================================

def process_reference_frequency(
    state: ReferenceFrequencyState,
    *,
    episodic_store: Any = None,
    binding_store: Any = None,
    consumption_store: Any = None,
    expectation_store: Any = None,
    motive_store: Any = None,
    narrative_state: Any = None,
    other_model_store: Any = None,
    self_reference_state: Any = None,
    action_result_state: Any = None,
    dialogue_learning_state: Any = None,
    forgetting_state: Any = None,
    config: Optional[ReferenceFrequencyConfig] = None,
    timestamp: Optional[float] = None,
) -> ReferenceFrequencyState:
    """参照頻度記述の1サイクル処理を実行する。

    オーケストレーション処理の一周期ごとに呼び出される。

    安全弁1: 全記録等価維持保証
    安全弁2: 評価的変換の禁止
    安全弁3: 累積的傾向の抑制
    安全弁4: 断面履歴の有限性
    安全弁5: 出力経路の限定と不拡張（本関数はenrichment出力を持たない）

    入力構造の参照回数は読み取り専用。書き込み能力を付与しない。
    本関数は忘却パイプラインへの出力経路を持たない。
    本関数は外部出力層への出力経路を持たない。
    本関数は想起経路の選択に影響を与えない。

    Args:
        state: 現在の状態
        (各構造のストア/ステート): 読み取り専用で参照
        config: 設定
        timestamp: 断面の生成時刻

    Returns:
        更新されたReferenceFrequencyState
    """
    cfg = config or ReferenceFrequencyConfig()

    # ── 収集 ──
    structure_counts = collect_reference_counts(
        episodic_store=episodic_store,
        binding_store=binding_store,
        consumption_store=consumption_store,
        expectation_store=expectation_store,
        motive_store=motive_store,
        narrative_state=narrative_state,
        other_model_store=other_model_store,
        self_reference_state=self_reference_state,
        action_result_state=action_result_state,
        dialogue_learning_state=dialogue_learning_state,
        forgetting_state=forgetting_state,
    )

    # ── 断面構成 ──
    snapshot = compose_snapshot(structure_counts, timestamp=timestamp)

    # ── 断面の時系列保持 ──
    new_history = list(state.snapshot_history)
    new_history.append(snapshot)

    # 安全弁4: 断面履歴の有限性。保持上限到達時はFIFO
    expired = 0
    if len(new_history) > cfg.max_snapshot_history:
        expired = len(new_history) - cfg.max_snapshot_history
        new_history = new_history[expired:]

    # ── 変動記述の再導出 ──
    # 安全弁3: 新しい断面が生成されるたびに再導出。累積蓄積しない。
    variation = derive_variation(new_history, cfg)

    new_state = ReferenceFrequencyState(
        snapshot_history=new_history,
        latest_variation=variation,
        total_snapshots_generated=state.total_snapshots_generated + 1,
        total_snapshots_expired=state.total_snapshots_expired + expired,
    )

    logger.debug(
        "Reference frequency snapshot: concentration=%.4f, bias=%.4f, "
        "history=%d, expired=%d",
        snapshot.concentration,
        snapshot.structural_bias,
        len(new_history),
        expired,
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁5: 出力先は内省系構造への参照情報に限定。
# enrichmentへの出力経路を持たない。
# 外部出力層への出力経路を持たない。
# 忘却パイプラインへの出力経路を持たない。
# 想起経路の選択への出力経路を持たない。

def get_latest_snapshot(state: ReferenceFrequencyState) -> Optional[ReferenceSnapshot]:
    """最新の断面を返す（内省系参照経路、READ-ONLY）。

    参照行為によって状態が変化することはない。

    Returns:
        直近の参照断面。履歴がなければ None。
    """
    if not state.snapshot_history:
        return None
    return state.snapshot_history[-1]


def get_snapshot_history(state: ReferenceFrequencyState) -> list[ReferenceSnapshot]:
    """断面履歴全体を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    参照行為によって状態が変化することはない。

    Returns:
        断面履歴のコピー。
    """
    return list(state.snapshot_history)


def get_latest_variation(state: ReferenceFrequencyState) -> Optional[VariationDescription]:
    """最新の変動記述を返す（内省系参照経路、READ-ONLY）。

    安全弁3: 累積的傾向の抑制。変動記述は「改善」「悪化」に相当する方向性を持たない。
    参照行為によって状態が変化することはない。

    Returns:
        直近の変動記述。断面が2件未満の場合は None。
    """
    return state.latest_variation


def get_reference_summary(state: ReferenceFrequencyState) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の構造を強調・選別しない。
    安全弁5: enrichment出力経路を持たない。

    Returns:
        サマリ辞書。
    """
    latest = get_latest_snapshot(state)
    variation = get_latest_variation(state)

    summary: dict[str, Any] = {
        "history_count": len(state.snapshot_history),
        "total_generated": state.total_snapshots_generated,
        "total_expired": state.total_snapshots_expired,
    }

    if latest is not None:
        summary["latest_concentration"] = latest.concentration
        summary["latest_structural_bias"] = latest.structural_bias
        summary["latest_timestamp"] = latest.timestamp

        # 構造別の参照合計（全記録等価、特定構造を強調しない）
        structure_totals: dict[str, int] = {}
        for key in ALL_STRUCTURE_KEYS:
            counts = latest.structure_counts.get(key, [])
            structure_totals[key] = sum(counts)
        summary["structure_totals"] = structure_totals

        # 構造別の記録数（全記録等価）
        structure_record_counts: dict[str, int] = {}
        for key in ALL_STRUCTURE_KEYS:
            counts = latest.structure_counts.get(key, [])
            structure_record_counts[key] = len(counts)
        summary["structure_record_counts"] = structure_record_counts

    if variation is not None:
        summary["variation_concentration_direction"] = variation.concentration_direction
        summary["variation_concentration_magnitude"] = variation.concentration_magnitude
        summary["variation_bias_direction"] = variation.bias_direction
        summary["variation_bias_magnitude"] = variation.bias_magnitude
        summary["variation_comparison_count"] = variation.comparison_count

    return summary


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: ReferenceFrequencyState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> ReferenceFrequencyState:
    """永続化用の辞書から状態を復元する。"""
    return ReferenceFrequencyState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_reference_frequency_state() -> ReferenceFrequencyState:
    """初期状態のファクトリ関数。"""
    return ReferenceFrequencyState()


def create_reference_frequency_config(
    max_snapshot_history: int = 30,
    variation_comparison_count: int = 5,
) -> ReferenceFrequencyConfig:
    """設定のファクトリ関数。"""
    return ReferenceFrequencyConfig(
        max_snapshot_history=max_snapshot_history,
        variation_comparison_count=variation_comparison_count,
    )
