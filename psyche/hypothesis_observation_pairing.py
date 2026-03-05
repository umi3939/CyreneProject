"""
psyche/hypothesis_observation_pairing.py - 他者モデル仮説の事後検証経路

仮説記述と観測記述を時系列的隣接に基づいて対構成し、蓄積する構造を提供する。

設計原則 (design_hypothesis_observation_pairing.md 準拠):
- 仮説の正誤を判定しない。仮説が「当たった」「外れた」という評価を付与しない
- 仮説と観測の「整合性」を算出しない。整合度・一致度・合致率を生成しない
- 仮説の修正を行わない。仮説記述の書き換え・撤回・強化は本機能の責務ではない
- 仮説の正確さに基づいて他者モデルの信頼度を変動させない
- 「仮説を修正すべき」「仮説が不正確である」という規範的情報を生成しない
- 観測の選択的蓄積を行わない。仮説と「整合する」観測も「整合しない」観測も等価に対構成
- 他者の意図・性質・信念を断定しない
- 正誤や善悪の評価を付与しない
- 蓄積された隣接対からパターン・傾向・規則性を抽出しない
- 蓄積情報を判断・行動選択・ポリシー選択に接続しない

6段パイプライン:
1. 仮説スナップショット取得 (hypothesis snapshot acquisition)
2. 観測記述取得 (observation description acquisition)
3. 隣接対の構成 (adjacent pair composition)
4. 相手別分離蓄積 (user-separated accumulation)
5. 鮮度管理と自然消失 (freshness management and natural disappearance)
6. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. 全記録の等価性: 重み・重要度・スコア・頻度情報を付与しない
2. 確認バイアスの構造的排除: 対構成基準に内容的整合性を用いない。時間的隣接のみ
3. FIFOによる自然消失: 上限到達時に最古の記録から機械的に押し出す。選択的保持を行わない
4. ルーミネーション防止: 同一対のenrichment連続列挙を制限する
5. パターン抽出の構造的排除: 統計量・頻度分布・傾向・規則性・成功率を算出しない
6. 単方向参照保証: 他者状態推測層・観測供給層・長期蓄積層への逆流経路を持たない
7. 判断系への経路遮断: enrichment等価列挙とREAD-ONLY参照のみ
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    """一意な識別子を生成する。"""
    return uuid.uuid4().hex[:12]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HypothesisSnapshot:
    """仮説記述のスナップショット。

    他者状態推測層から読み取った仮説のある時点の凍結。
    仮説自体を変更しない。読み取りのみ。
    """
    hypothesis_id: str = ""
    description: str = ""
    freshness_value: float = 0.0
    strength_value: float = 0.0
    snapshot_cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "freshness_value": self.freshness_value,
            "strength_value": self.strength_value,
            "snapshot_cycle": self.snapshot_cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HypothesisSnapshot:
        return cls(
            hypothesis_id=data.get("hypothesis_id", ""),
            description=data.get("description", ""),
            freshness_value=data.get("freshness_value", 0.0),
            strength_value=data.get("strength_value", 0.0),
            snapshot_cycle=data.get("snapshot_cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ObservationDescription:
    """観測記述。観測供給層から読み取った観測断片の記述。

    観測断片の内容を改変しない。読み取りのみ。
    """
    fragment_type: str = ""
    description: str = ""
    arrival_cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_type": self.fragment_type,
            "description": self.description,
            "arrival_cycle": self.arrival_cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationDescription:
        return cls(
            fragment_type=data.get("fragment_type", ""),
            description=data.get("description", ""),
            arrival_cycle=data.get("arrival_cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class AdjacentPair:
    """仮説-観測の隣接対。

    仮説記述（先行記述）と観測記述（後続記述）を時間的隣接に基づいて
    対構成した結果。一度構成された隣接対は変更されない（追記のみ）。
    全項目は等価に保持される。特定項目に重み・重要度・優先度を付与しない。
    """
    pair_id: str = field(default_factory=_gen_id)
    # 仮説記述
    hypothesis_id: str = ""
    hypothesis_description: str = ""
    hypothesis_freshness: float = 0.0
    hypothesis_strength: float = 0.0
    hypothesis_snapshot_cycle: int = 0
    # 観測記述
    observation_type: str = ""
    observation_description: str = ""
    observation_arrival_cycle: int = 0
    # 相手識別子
    user_id: str = ""
    # 対構成時のタイムスタンプ
    timestamp: float = field(default_factory=time.time)
    # 鮮度値（処理ごとに段階的に減少）
    freshness: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_description": self.hypothesis_description,
            "hypothesis_freshness": self.hypothesis_freshness,
            "hypothesis_strength": self.hypothesis_strength,
            "hypothesis_snapshot_cycle": self.hypothesis_snapshot_cycle,
            "observation_type": self.observation_type,
            "observation_description": self.observation_description,
            "observation_arrival_cycle": self.observation_arrival_cycle,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdjacentPair:
        return cls(
            pair_id=data.get("pair_id", _gen_id()),
            hypothesis_id=data.get("hypothesis_id", ""),
            hypothesis_description=data.get("hypothesis_description", ""),
            hypothesis_freshness=data.get("hypothesis_freshness", 0.0),
            hypothesis_strength=data.get("hypothesis_strength", 0.0),
            hypothesis_snapshot_cycle=data.get("hypothesis_snapshot_cycle", 0),
            observation_type=data.get("observation_type", ""),
            observation_description=data.get("observation_description", ""),
            observation_arrival_cycle=data.get("observation_arrival_cycle", 0),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", time.time()),
            freshness=data.get("freshness", 1.0),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HypothesisObservationPairingConfig:
    """設定。"""
    # 仮説スナップショットバッファの上限
    max_snapshot_buffer: int = 20

    # スナップショットの保持サイクル数（これを超えると機械的に押し出される）
    snapshot_retention_cycles: int = 10

    # 隣接対の構成に使用するサイクル近接範囲
    # 仮説スナップショットのサイクルと観測到着サイクルの差がこの範囲内で対構成
    cycle_proximity_range: int = 5

    # 相手別蓄積上限
    max_pairs_per_user: int = 50

    # 全体の蓄積上限（FIFO押し出し）
    max_total_pairs: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))

    # 鮮度減衰量（処理ごとに全記録に均一適用）
    freshness_decay_rate: float = field(default_factory=lambda: coefficient_registry.get("description_common", "freshness_decay_rate_002"))

    # 消失水準（これ以下は不可視）
    freshness_invisible_threshold: float = 0.05

    # enrichmentに列挙する直近対の件数上限
    enrichment_count: int = 5

    # ルーミネーション防止: 連続列挙回数の上限
    rumination_consecutive_limit: int = 3

    # ルーミネーション防止: 除外後の復帰までのサイクル数
    rumination_cooldown_cycles: int = 2

    # READ-ONLY参照として提供する直近対の件数
    reference_history_count: int = 20


# =============================================================================
# State
# =============================================================================

@dataclass
class HypothesisObservationPairingState:
    """内部状態。永続化対象。"""
    # 仮説スナップショットバッファ
    snapshot_buffer: list[HypothesisSnapshot] = field(default_factory=list)

    # 相手別蓄積（user_id -> 隣接対リスト）
    user_pairs: dict[str, list[AdjacentPair]] = field(default_factory=dict)

    # 全隣接対リスト（時系列順にFIFO蓄積、相手別蓄積との二重管理）
    all_pairs: list[AdjacentPair] = field(default_factory=list)

    # ルーミネーション防止用の参照記録
    # enrichmentに列挙された隣接対のpair_idと連続列挙回数
    enrichment_consecutive: dict[str, int] = field(default_factory=dict)

    # 処理統計（等価な事実記述のみ）
    total_pairs_created: int = 0
    total_pairs_pushed_out: int = 0
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_buffer": [s.to_dict() for s in self.snapshot_buffer],
            "user_pairs": {
                uid: [p.to_dict() for p in pairs]
                for uid, pairs in self.user_pairs.items()
            },
            "all_pairs": [p.to_dict() for p in self.all_pairs],
            "enrichment_consecutive": dict(self.enrichment_consecutive),
            "total_pairs_created": self.total_pairs_created,
            "total_pairs_pushed_out": self.total_pairs_pushed_out,
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HypothesisObservationPairingState:
        user_pairs_raw = data.get("user_pairs", {})
        user_pairs: dict[str, list[AdjacentPair]] = {}
        for uid, pair_list in user_pairs_raw.items():
            user_pairs[uid] = [AdjacentPair.from_dict(p) for p in pair_list]

        return cls(
            snapshot_buffer=[
                HypothesisSnapshot.from_dict(s)
                for s in data.get("snapshot_buffer", [])
            ],
            user_pairs=user_pairs,
            all_pairs=[
                AdjacentPair.from_dict(p)
                for p in data.get("all_pairs", [])
            ],
            enrichment_consecutive=dict(data.get("enrichment_consecutive", {})),
            total_pairs_created=data.get("total_pairs_created", 0),
            total_pairs_pushed_out=data.get("total_pairs_pushed_out", 0),
            cycle_count=data.get("cycle_count", 0),
        )


# =============================================================================
# Stage 1: 仮説スナップショット取得
# =============================================================================

def acquire_hypothesis_snapshots(
    hypothesis_source: Any,
    current_cycle: int,
) -> list[HypothesisSnapshot]:
    """他者状態推測層から現在の仮説群を読み取り、スナップショットを生成する。

    READ-ONLY参照。仮説の内容を改変しない。
    仮説記述から識別子・記述文・鮮度段階値・強度段階値のみを抽出する。

    Args:
        hypothesis_source: 他者状態推測層（OtherAgentModelSystem等）。READ-ONLY参照。
        current_cycle: 現在の処理サイクル番号

    Returns:
        仮説スナップショットのリスト
    """
    snapshots: list[HypothesisSnapshot] = []
    now = time.time()

    # OtherAgentModelSystem.get_active_hypotheses() or get_store().hypotheses
    hypotheses = []
    if hasattr(hypothesis_source, "get_active_hypotheses"):
        hypotheses = hypothesis_source.get_active_hypotheses()
    elif hasattr(hypothesis_source, "get_store"):
        store = hypothesis_source.get_store()
        if hasattr(store, "hypotheses"):
            hypotheses = store.hypotheses
    elif hasattr(hypothesis_source, "hypotheses"):
        hypotheses = hypothesis_source.hypotheses

    # 直接リストが渡された場合
    if isinstance(hypothesis_source, list):
        hypotheses = hypothesis_source

    for hyp in hypotheses:
        hyp_id = getattr(hyp, "hypothesis_id", "")
        description = getattr(hyp, "description", "")
        freshness = getattr(hyp, "freshness", 0.0)
        strength = getattr(hyp, "strength", 0.0)

        if not description:
            continue

        snapshots.append(HypothesisSnapshot(
            hypothesis_id=hyp_id,
            description=description,
            freshness_value=float(freshness),
            strength_value=float(strength),
            snapshot_cycle=current_cycle,
            timestamp=now,
        ))

    return snapshots


def update_snapshot_buffer(
    buffer: list[HypothesisSnapshot],
    new_snapshots: list[HypothesisSnapshot],
    config: HypothesisObservationPairingConfig,
    current_cycle: int,
) -> list[HypothesisSnapshot]:
    """仮説スナップショットバッファを更新する。

    新規スナップショットを追加し、保持サイクルを超えたものを機械的に押し出す。
    バッファ上限到達時は最古のスナップショットから押し出す。

    Args:
        buffer: 現在のバッファ
        new_snapshots: 新たに取得したスナップショット
        config: 設定
        current_cycle: 現在の処理サイクル番号

    Returns:
        更新後のバッファ
    """
    updated = list(buffer)

    # 新規スナップショットを追加
    updated.extend(new_snapshots)

    # 保持サイクルを超えたスナップショットを機械的に押し出す
    updated = [
        s for s in updated
        if (current_cycle - s.snapshot_cycle) <= config.snapshot_retention_cycles
    ]

    # バッファ上限管理
    if len(updated) > config.max_snapshot_buffer:
        updated = updated[-config.max_snapshot_buffer:]

    return updated


# =============================================================================
# Stage 2: 観測記述取得
# =============================================================================

def acquire_observation_descriptions(
    observation_source: Any,
    current_cycle: int,
) -> list[ObservationDescription]:
    """観測供給層から直近の観測断片群を読み取る。

    READ-ONLY参照。観測断片の内容を改変しない。
    各観測断片の種別・記述を抽出する。

    Args:
        observation_source: 観測供給層（RealFeedProcessor等）。READ-ONLY参照。
        current_cycle: 現在の処理サイクル番号

    Returns:
        観測記述のリスト
    """
    descriptions: list[ObservationDescription] = []
    now = time.time()

    fragments = []

    # RealFeedProcessor パターン
    if hasattr(observation_source, "get_latest_units"):
        fragments = observation_source.get_latest_units()
    elif hasattr(observation_source, "state"):
        state = observation_source.state
        if hasattr(state, "observation_units"):
            fragments = state.observation_units
        elif hasattr(state, "fragments"):
            fragments = state.fragments
    # 直接リストが渡された場合
    elif isinstance(observation_source, list):
        fragments = observation_source

    for frag in fragments:
        frag_type = ""
        frag_desc = ""

        # ObservationFragment / ObservationUnit パターン
        ft = getattr(frag, "type", None)
        if ft is not None:
            frag_type = ft.value if hasattr(ft, "value") else str(ft)

        # description / text_hint / source_description
        frag_desc = getattr(frag, "description", "")
        if not frag_desc:
            frag_desc = getattr(frag, "text_hint", "")
        if not frag_desc:
            frag_desc = getattr(frag, "source_description", "")

        if not frag_desc and not frag_type:
            continue

        if not frag_desc:
            frag_desc = f"[{frag_type}]"

        descriptions.append(ObservationDescription(
            fragment_type=frag_type,
            description=frag_desc,
            arrival_cycle=current_cycle,
            timestamp=now,
        ))

    return descriptions


# =============================================================================
# Stage 3: 隣接対の構成
# =============================================================================

def compose_adjacent_pairs(
    snapshot_buffer: list[HypothesisSnapshot],
    observations: list[ObservationDescription],
    config: HypothesisObservationPairingConfig,
    user_id: str = "",
) -> list[AdjacentPair]:
    """仮説スナップショットと観測記述を時間的隣接に基づいて対構成する。

    対構成の基準は「仮説が保持されていた時間帯と、観測断片が到着した時間帯が
    近接していること」のみ。内容的な関連性・整合性・一致度に基づく対構成は行わない。

    一つの仮説記述に対して複数の観測記述が隣接する場合、それぞれ独立した隣接対として構成する。
    一つの観測記述に対して複数の仮説記述が隣接する場合も同様。

    仮説と「整合する」観測も「整合しない」観測も、同じ基準で対構成する。
    選択的蓄積を行わない。

    Args:
        snapshot_buffer: 仮説スナップショットバッファ
        observations: 観測記述リスト
        config: 設定
        user_id: 相手識別子

    Returns:
        新規構成された隣接対のリスト
    """
    new_pairs: list[AdjacentPair] = []
    now = time.time()

    for snapshot in snapshot_buffer:
        for obs in observations:
            # 時間的隣接の判定: サイクル差が近接範囲内
            cycle_diff = obs.arrival_cycle - snapshot.snapshot_cycle
            if cycle_diff < 0:
                # 仮説スナップショットが観測より後の場合は構成しない
                # （仮説が先行記述、観測が後続記述）
                continue
            if cycle_diff > config.cycle_proximity_range:
                continue

            pair = AdjacentPair(
                hypothesis_id=snapshot.hypothesis_id,
                hypothesis_description=snapshot.description,
                hypothesis_freshness=snapshot.freshness_value,
                hypothesis_strength=snapshot.strength_value,
                hypothesis_snapshot_cycle=snapshot.snapshot_cycle,
                observation_type=obs.fragment_type,
                observation_description=obs.description,
                observation_arrival_cycle=obs.arrival_cycle,
                user_id=user_id,
                timestamp=now,
                freshness=1.0,
            )
            new_pairs.append(pair)

    return new_pairs


# =============================================================================
# Stage 4: 相手別分離蓄積
# =============================================================================

def accumulate_pairs_by_user(
    state: HypothesisObservationPairingState,
    new_pairs: list[AdjacentPair],
    config: HypothesisObservationPairingConfig,
) -> None:
    """隣接対を相手別に分離して蓄積する。

    相手別の分離は識別子の一致のみに基づく。
    相手の属性・重要度・頻度に基づく優先は設けない。
    蓄積は追記形式のみで、既存の対の遡及的変更は行わない。
    上限到達時は最古の対から順にFIFO押し出し。

    Args:
        state: 内部状態（in-place更新）
        new_pairs: 新規構成された隣接対
        config: 設定
    """
    for pair in new_pairs:
        uid = pair.user_id or "__unknown__"

        # 相手別蓄積
        if uid not in state.user_pairs:
            state.user_pairs[uid] = []
        state.user_pairs[uid].append(pair)

        # 相手別上限チェック（FIFO押し出し）
        if len(state.user_pairs[uid]) > config.max_pairs_per_user:
            overflow = len(state.user_pairs[uid]) - config.max_pairs_per_user
            removed = state.user_pairs[uid][:overflow]
            state.user_pairs[uid] = state.user_pairs[uid][overflow:]
            # 全体リストからも除去
            removed_ids = {p.pair_id for p in removed}
            state.all_pairs = [
                p for p in state.all_pairs if p.pair_id not in removed_ids
            ]
            state.total_pairs_pushed_out += overflow

        # 全体蓄積
        state.all_pairs.append(pair)
        state.total_pairs_created += 1

    # 全体上限チェック（FIFO押し出し）
    if len(state.all_pairs) > config.max_total_pairs:
        overflow = len(state.all_pairs) - config.max_total_pairs
        removed = state.all_pairs[:overflow]
        state.all_pairs = state.all_pairs[overflow:]
        state.total_pairs_pushed_out += overflow

        # 相手別蓄積からも除去
        removed_ids = {p.pair_id for p in removed}
        for uid in list(state.user_pairs.keys()):
            state.user_pairs[uid] = [
                p for p in state.user_pairs[uid]
                if p.pair_id not in removed_ids
            ]
            # 空になった相手のエントリを削除
            if not state.user_pairs[uid]:
                del state.user_pairs[uid]


# =============================================================================
# Stage 5: 鮮度管理と自然消失
# =============================================================================

def apply_freshness_decay(
    state: HypothesisObservationPairingState,
    config: HypothesisObservationPairingConfig,
) -> None:
    """全蓄積記録の鮮度を段階的に減少させる。

    減衰速度は全記録で均一。記録の内容に基づく選択的保持を行わない。
    鮮度が消失水準に到達した記録は不可視状態に移行する（全体リストから除去）。
    消失は機械的であり、記録の内容に基づく判断を含まない。

    Args:
        state: 内部状態（in-place更新）
        config: 設定
    """
    # 全隣接対の鮮度を均一に減衰
    for pair in state.all_pairs:
        pair.freshness = _clamp(pair.freshness - config.freshness_decay_rate)

    # 不可視化: 鮮度が消失水準以下の記録を除去
    invisible_ids: set[str] = set()
    visible_pairs: list[AdjacentPair] = []
    for pair in state.all_pairs:
        if pair.freshness <= config.freshness_invisible_threshold:
            invisible_ids.add(pair.pair_id)
            state.total_pairs_pushed_out += 1
        else:
            visible_pairs.append(pair)
    state.all_pairs = visible_pairs

    # 相手別蓄積からも不可視化した記録を除去
    if invisible_ids:
        for uid in list(state.user_pairs.keys()):
            state.user_pairs[uid] = [
                p for p in state.user_pairs[uid]
                if p.pair_id not in invisible_ids
            ]
            if not state.user_pairs[uid]:
                del state.user_pairs[uid]


# =============================================================================
# Stage 6: 参照情報としての受渡準備
# =============================================================================

def prepare_enrichment_pairs(
    state: HypothesisObservationPairingState,
    config: HypothesisObservationPairingConfig,
) -> list[AdjacentPair]:
    """enrichmentへの等価列挙用に直近の隣接対を選出する。

    列挙に際して順序以外の優先度・重要度・選択基準を設けない。
    ルーミネーション防止: 同一対が連続してenrichmentに列挙され続けることを防止する。

    Args:
        state: 内部状態
        config: 設定

    Returns:
        enrichmentに列挙する隣接対のリスト（記録順）
    """
    if not state.all_pairs:
        return []

    # 直近の対から候補を取得（余裕を持って）
    candidates = list(state.all_pairs[-config.enrichment_count * 2:])

    # ルーミネーション防止: 連続列挙回数が上限を超えた対を一時除外
    result: list[AdjacentPair] = []
    for pair in reversed(candidates):  # 直近から
        if len(result) >= config.enrichment_count:
            break

        consecutive = state.enrichment_consecutive.get(pair.pair_id, 0)
        if consecutive >= config.rumination_consecutive_limit:
            # 上限到達: 一時的に除外
            continue
        result.append(pair)

    # 記録順に戻す
    result.reverse()

    # 列挙された対の連続カウントを更新
    listed_ids = {p.pair_id for p in result}
    new_consecutive: dict[str, int] = {}
    for pair_id, count in state.enrichment_consecutive.items():
        if pair_id in listed_ids:
            new_consecutive[pair_id] = count + 1
        else:
            # 列挙から外れた場合はクールダウン減算
            new_count = count - 1
            if new_count > 0:
                new_consecutive[pair_id] = new_count
    # 新規列挙の対を追加
    for pair_id in listed_ids:
        if pair_id not in new_consecutive:
            new_consecutive[pair_id] = 1

    state.enrichment_consecutive = new_consecutive

    return result


def get_reference_history(
    state: HypothesisObservationPairingState,
    config: HypothesisObservationPairingConfig,
    user_id: str = "",
) -> list[AdjacentPair]:
    """蓄積された隣接対の直近分をREAD-ONLY参照として返す。

    全記録を等価に返す。フィルタリング・選別・集約機能を持たない。
    参照行為によって対の内容や順序が変化することはない。

    Args:
        state: 内部状態
        config: 設定
        user_id: 相手識別子（指定時はその相手の隣接対のみ）

    Returns:
        蓄積された隣接対のリスト（READ-ONLY参照）
    """
    if user_id and user_id in state.user_pairs:
        pairs = state.user_pairs[user_id]
        return list(pairs[-config.reference_history_count:])
    return list(state.all_pairs[-config.reference_history_count:])


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_pairing_summary_text(state: HypothesisObservationPairingState) -> str:
    """仮説-観測隣接対蓄積状態の要約（enrichment用）。

    評価判定・行動指示を含まない。等価列挙に限定する。
    パターン抽出を行わない。
    """
    if state.cycle_count == 0 and not state.all_pairs:
        return "仮説-観測隣接対: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    pair_count = len(state.all_pairs)
    parts.append(f"蓄積対={pair_count}")

    user_count = len(state.user_pairs)
    if user_count > 0:
        parts.append(f"相手数={user_count}")

    buffer_count = len(state.snapshot_buffer)
    if buffer_count > 0:
        parts.append(f"バッファ={buffer_count}")

    if state.total_pairs_pushed_out > 0:
        parts.append(f"消失累計={state.total_pairs_pushed_out}")

    return " ".join(parts) if parts else "仮説-観測隣接対: 待機中"


# =============================================================================
# Processor (6-stage pipeline)
# =============================================================================

class HypothesisObservationPairingProcessor:
    """仮説-観測隣接対のプロセッサ。

    6段パイプライン:
    1. 仮説スナップショット取得
    2. 観測記述取得
    3. 隣接対の構成
    4. 相手別分離蓄積
    5. 鮮度管理と自然消失
    6. 参照情報としての受渡準備

    安全弁:
    1. 全記録の等価性
    2. 確認バイアスの構造的排除
    3. FIFOによる自然消失
    4. ルーミネーション防止
    5. パターン抽出の構造的排除
    6. 単方向参照保証
    7. 判断系への経路遮断
    """

    def __init__(self, config: Optional[HypothesisObservationPairingConfig] = None):
        self._config = config or HypothesisObservationPairingConfig()
        self._state = HypothesisObservationPairingState()

    @property
    def state(self) -> HypothesisObservationPairingState:
        return self._state

    @state.setter
    def state(self, value: HypothesisObservationPairingState) -> None:
        self._state = value

    @property
    def config(self) -> HypothesisObservationPairingConfig:
        return self._config

    # ─── Main processing entry point ──────────────────────────

    def process(
        self,
        hypothesis_source: Any = None,
        observation_source: Any = None,
        user_id_source: Any = None,
        current_cycle: Optional[int] = None,
    ) -> int:
        """6段パイプラインの一括実行。

        入力:
        - hypothesis_source: 他者状態推測層（READ-ONLY参照）
        - observation_source: 観測供給層（READ-ONLY参照）
        - user_id_source: 長期蓄積層の相手識別情報（READ-ONLY参照）
        - current_cycle: 現在の処理サイクル番号

        本機能が入力元を変更することはない（単方向参照保証）。

        Returns:
            今回新規構成された隣接対の数
        """
        self._state.cycle_count += 1
        cycle = current_cycle if current_cycle is not None else self._state.cycle_count

        # Stage 1: 仮説スナップショット取得
        if hypothesis_source is not None:
            new_snapshots = acquire_hypothesis_snapshots(hypothesis_source, cycle)
            self._state.snapshot_buffer = update_snapshot_buffer(
                self._state.snapshot_buffer,
                new_snapshots,
                self._config,
                cycle,
            )

        # Stage 2: 観測記述取得
        observations: list[ObservationDescription] = []
        if observation_source is not None:
            observations = acquire_observation_descriptions(observation_source, cycle)

        # 相手識別情報の取得
        user_id = _extract_user_id(user_id_source)

        # Stage 3: 隣接対の構成
        new_pairs = compose_adjacent_pairs(
            self._state.snapshot_buffer,
            observations,
            self._config,
            user_id=user_id,
        )

        # Stage 4: 相手別分離蓄積
        accumulate_pairs_by_user(self._state, new_pairs, self._config)

        # Stage 5: 鮮度管理と自然消失
        apply_freshness_decay(self._state, self._config)

        logger.debug(
            "Hypothesis-observation pairing: cycle=%d, new_pairs=%d, total=%d, buffer=%d",
            self._state.cycle_count,
            len(new_pairs),
            len(self._state.all_pairs),
            len(self._state.snapshot_buffer),
        )

        return len(new_pairs)

    # ─── Stage 6: 参照情報の提供 ──────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        直近の隣接対を記録順に等価列挙する。
        列挙に際して順序以外の優先度・重要度・選択基準を設けない。
        出力は参照情報のみ。判断・行動・評価を直接引き起こさない。

        Returns:
            enrichment用の構造化データ
        """
        enrichment_pairs = prepare_enrichment_pairs(self._state, self._config)

        entries: list[dict[str, Any]] = []
        for pair in enrichment_pairs:
            hyp_preview = pair.hypothesis_description[:80] if pair.hypothesis_description else ""
            if len(pair.hypothesis_description) > 80:
                hyp_preview += "..."
            obs_preview = pair.observation_description[:80] if pair.observation_description else ""
            if len(pair.observation_description) > 80:
                obs_preview += "..."

            entries.append({
                "hypothesis": hyp_preview,
                "hypothesis_freshness": pair.hypothesis_freshness,
                "hypothesis_strength": pair.hypothesis_strength,
                "observation_type": pair.observation_type,
                "observation": obs_preview,
                "user_id": pair.user_id,
                "snapshot_cycle": pair.hypothesis_snapshot_cycle,
                "observation_cycle": pair.observation_arrival_cycle,
            })

        summary_text = get_pairing_summary_text(self._state)

        return {
            "pair_count": len(self._state.all_pairs),
            "user_count": len(self._state.user_pairs),
            "entries": entries,
            "summary_text": summary_text,
        }

    def get_latest_pairs(self, count: Optional[int] = None) -> list[AdjacentPair]:
        """直近の隣接対をREAD-ONLYで返す。

        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。

        Args:
            count: 取得件数。Noneの場合はenrichment_count件を返す。

        Returns:
            直近の隣接対のリスト（READ-ONLY参照）
        """
        n = count if count is not None else self._config.enrichment_count
        return list(self._state.all_pairs[-n:])

    def get_pair_history(self, user_id: str = "") -> list[AdjacentPair]:
        """蓄積された隣接対の履歴をREAD-ONLYで返す。

        内省系構造へのREAD-ONLY参照。
        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。

        Args:
            user_id: 相手識別子（指定時はその相手の隣接対のみ）

        Returns:
            蓄積された隣接対のリスト（READ-ONLY参照）
        """
        return get_reference_history(self._state, self._config, user_id=user_id)

    def get_user_ids(self) -> list[str]:
        """蓄積されている相手識別子のリストを返す。

        相手間で共有・統合・比較する経路を持たない。

        Returns:
            相手識別子のリスト
        """
        return list(self._state.user_pairs.keys())

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "pair_count": len(self._state.all_pairs),
            "user_count": len(self._state.user_pairs),
            "buffer_count": len(self._state.snapshot_buffer),
            "total_pairs_created": self._state.total_pairs_created,
            "total_pairs_pushed_out": self._state.total_pairs_pushed_out,
            "cycle_count": self._state.cycle_count,
        }


# =============================================================================
# Helper: user_id extraction
# =============================================================================

def _extract_user_id(source: Any) -> str:
    """長期蓄積層の相手識別情報を読み取る。

    READ-ONLY参照。新たな相手識別体系を導入しない。
    長期蓄積層の相手識別と同一体系を使用する。

    Args:
        source: 長期蓄積層の出力やuser_idを含む任意オブジェクト

    Returns:
        相手識別子（文字列）
    """
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    # user_id 属性を持つオブジェクト
    uid = getattr(source, "user_id", None)
    if uid and isinstance(uid, str):
        return uid
    # dict の場合
    if isinstance(source, dict):
        return str(source.get("user_id", ""))
    return ""


# =============================================================================
# Save / Load (永続化)
# =============================================================================

def save_pairing_state(state: HypothesisObservationPairingState) -> dict[str, Any]:
    """状態をシリアライズ可能なdictに変換する。"""
    return state.to_dict()


def load_pairing_state(data: dict[str, Any]) -> HypothesisObservationPairingState:
    """dictから状態を復元する。"""
    return HypothesisObservationPairingState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_hypothesis_observation_pairing_processor(
    config: Optional[HypothesisObservationPairingConfig] = None,
) -> HypothesisObservationPairingProcessor:
    """HypothesisObservationPairingProcessor のファクトリ関数。"""
    return HypothesisObservationPairingProcessor(config=config)


def get_hypothesis_observation_pairing_summary(
    processor: HypothesisObservationPairingProcessor,
) -> str:
    """プロセッサの要約文字列を返す。"""
    return get_pairing_summary_text(processor.state)
