"""
psyche/interaction_accumulation.py - 相互作用の蓄積記述

自己行動知覚が保持する自己表出記録と他者モデルリアルフィードが保持する他者反応記録を、
時系列的隣接関係として対構成し、蓄積する構造を提供する。

設計原則 (design_interaction_accumulation.md 準拠):
- 因果帰属を行わない。「自分がXを言ったから相手がYと反応した」という因果断定を生成しない。
  記録するのは「Xの後にYが隣接した」という時系列的事実のみ
- パターン抽出を行わない。蓄積された隣接対から規則性・傾向・パターンを導出しない
- 相互作用の良否判定を行わない。「良い相互作用」「悪い相互作用」といった評価を付与しない
- 応答方針への直接的影響を行わない。ポリシー選択・バイアス計算・安定化弁に直接入力しない
- 他者の意図・性質の推測を行わない。他者の内面推測は他者モデルの責務
- 反応の予測を行わない。予測構造を持たない
- 感情パイプラインのパラメータを変更しない

4段パイプライン:
1. 隣接対の構成 (adjacent pair composition)
2. 対の記述 (pair description)
3. 蓄積と消失 (accumulation and expiry)
4. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. 全記録の等価性: 重み・重要度・頻度情報を付与しない
2. FIFOによる自然消失: 古い対は機械的に押し出す。選択的保持を行わない
3. ルーミネーション防止: 同一対のenrichment連続列挙を防止
4. パターン抽出の構造的排除: 統計量・頻度分布・傾向・規則性を算出しない
5. 判断系への経路遮断: enrichment等価列挙とREAD-ONLY参照のみ
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


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AdjacentPair:
    """隣接対。自己表出と他者反応の時系列的隣接記録。

    一度構成された隣接対は変更されない（追記のみ）。
    全記録は等価であり、重み・スコア・優先度・重要度を付与しない。
    因果帰属を行わない。記録するのは時系列的事実のみ。
    """
    pair_id: str = field(default_factory=_gen_id)
    # 自己表出の内容
    self_text: str = ""
    self_policy_label: str = ""
    self_tick: int = 0
    # 他者反応の観測断片
    other_reaction: str = ""
    other_tick: int = 0
    # 対構成時のタイムスタンプ
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "self_text": self.self_text,
            "self_policy_label": self.self_policy_label,
            "self_tick": self.self_tick,
            "other_reaction": self.other_reaction,
            "other_tick": self.other_tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdjacentPair":
        return cls(
            pair_id=data.get("pair_id", _gen_id()),
            self_text=data.get("self_text", ""),
            self_policy_label=data.get("self_policy_label", ""),
            self_tick=data.get("self_tick", 0),
            other_reaction=data.get("other_reaction", ""),
            other_tick=data.get("other_tick", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class BufferEntry:
    """構成バッファのエントリ。

    自己表出記録を受領したが、対応する他者反応観測がまだ到着していない状態の一時保持。
    他者反応が到着した時点で対構成が行われ、バッファから除去される。
    一定ティック経過後も他者反応が到着しない場合は保留状態に移行する。
    """
    entry_id: str = field(default_factory=_gen_id)
    self_text: str = ""
    self_policy_label: str = ""
    self_tick: int = 0
    self_timestamp: float = field(default_factory=time.time)
    is_pending: bool = False  # 保留状態（タイムアウト済み）

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "self_text": self.self_text,
            "self_policy_label": self.self_policy_label,
            "self_tick": self.self_tick,
            "self_timestamp": self.self_timestamp,
            "is_pending": self.is_pending,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BufferEntry":
        return cls(
            entry_id=data.get("entry_id", _gen_id()),
            self_text=data.get("self_text", ""),
            self_policy_label=data.get("self_policy_label", ""),
            self_tick=data.get("self_tick", 0),
            self_timestamp=data.get("self_timestamp", time.time()),
            is_pending=data.get("is_pending", False),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class InteractionAccumulationState:
    """相互作用蓄積の内部状態。永続化対象。"""
    # 隣接対のリスト（時系列順にFIFO蓄積）
    pairs: list[AdjacentPair] = field(default_factory=list)

    # 構成バッファ（自己表出の一時保持）
    buffer: list[BufferEntry] = field(default_factory=list)

    # ルーミネーション防止用の参照済み記録
    # enrichmentに列挙された対のpair_idと連続列挙回数
    enrichment_consecutive: dict[str, int] = field(default_factory=dict)

    # カウンタ
    total_pairs_created: int = 0
    total_pairs_pushed_out: int = 0
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "buffer": [b.to_dict() for b in self.buffer],
            "enrichment_consecutive": dict(self.enrichment_consecutive),
            "total_pairs_created": self.total_pairs_created,
            "total_pairs_pushed_out": self.total_pairs_pushed_out,
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionAccumulationState":
        return cls(
            pairs=[
                AdjacentPair.from_dict(p)
                for p in data.get("pairs", [])
            ],
            buffer=[
                BufferEntry.from_dict(b)
                for b in data.get("buffer", [])
            ],
            enrichment_consecutive=dict(data.get("enrichment_consecutive", {})),
            total_pairs_created=data.get("total_pairs_created", 0),
            total_pairs_pushed_out=data.get("total_pairs_pushed_out", 0),
            cycle_count=data.get("cycle_count", 0),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InteractionAccumulationConfig:
    """設定。"""
    # 隣接対リストの上限（FIFO押し出し）
    max_pairs: int = 100

    # 構成バッファの上限
    max_buffer: int = 20

    # 対構成のティック近接範囲（自己表出と他者反応のティック差がこの範囲内で対構成）
    tick_proximity_range: int = 5

    # 同一ティック内での即時構成禁止: 最低経過ティック数
    min_tick_gap: int = 1

    # バッファエントリの保留タイムアウト（ティック数）: これを超えると保留状態に移行
    buffer_timeout_ticks: int = 10

    # enrichmentに列挙する直近対の件数上限
    enrichment_count: int = 5

    # ルーミネーション防止: 連続列挙回数の上限
    rumination_consecutive_limit: int = 3

    # ルーミネーション防止: 除外後の復帰までのサイクル数
    rumination_cooldown_cycles: int = 2

    # READ-ONLY参照として提供する直近対の件数
    reference_history_count: int = 20


# =============================================================================
# Stage 1: 隣接対の構成
# =============================================================================

def compose_pairs(
    self_records: list[Any],
    other_units: list[Any],
    buffer: list[BufferEntry],
    config: InteractionAccumulationConfig,
    current_tick: int,
) -> tuple[list[AdjacentPair], list[BufferEntry]]:
    """自己表出記録と他者反応観測を時系列的近接性に基づいて対構成する。

    対構成の基準は時間的隣接のみ。内容的関連性の判定は行わない。
    一つの自己表出に対して複数の他者反応が隣接する場合、それぞれを独立した対として構成する。
    一つの他者反応に対して複数の自己表出が隣接する場合も同様。
    対の構成に優先順位を設けない。

    Args:
        self_records: 自己行動知覚の直近記録リスト（READ-ONLY参照）
        other_units: 他者モデルリアルフィードの観測ユニットリスト（READ-ONLY参照）
        buffer: 構成バッファ（既存のバッファエントリ）
        config: 設定
        current_tick: 現在のティック番号

    Returns:
        (新規構成された隣接対リスト, 更新後のバッファ)
    """
    now = time.time()
    new_pairs: list[AdjacentPair] = []
    updated_buffer = list(buffer)

    # 1. 自己表出記録からバッファに新規エントリを追加
    # バッファに既に存在するティック番号は追加しない
    existing_ticks = {b.self_tick for b in updated_buffer}
    for rec in self_records:
        rec_tick = getattr(rec, "tick", 0)
        rec_text = getattr(rec, "response_text", "")
        rec_policy = getattr(rec, "policy_label", "")
        rec_ts = getattr(rec, "timestamp", now)

        if rec_tick in existing_ticks:
            continue
        if not rec_text:
            continue

        updated_buffer.append(BufferEntry(
            self_text=rec_text,
            self_policy_label=rec_policy,
            self_tick=rec_tick,
            self_timestamp=rec_ts,
        ))
        existing_ticks.add(rec_tick)

    # 2. 他者反応観測ユニットからの対構成
    # 各バッファエントリと各他者反応の組み合わせを時間的隣接で判定
    matched_buffer_ids: set[str] = set()

    for unit in other_units:
        unit_tick = getattr(unit, "tick", 0)
        if not isinstance(unit_tick, int):
            # tickがない場合はcycle_countや他の時間情報を使う
            unit_tick = getattr(unit, "cycle_count", 0)

        # 他者反応の記述を取得
        reaction_desc = _extract_reaction_description(unit)
        if not reaction_desc:
            continue

        for buf_entry in updated_buffer:
            if buf_entry.is_pending:
                continue

            tick_diff = unit_tick - buf_entry.self_tick
            # 時間的隣接の判定: tick_diffが min_tick_gap 以上 tick_proximity_range 以下
            if tick_diff < config.min_tick_gap:
                continue
            if tick_diff > config.tick_proximity_range:
                continue

            # 対を構成
            pair = AdjacentPair(
                self_text=buf_entry.self_text,
                self_policy_label=buf_entry.self_policy_label,
                self_tick=buf_entry.self_tick,
                other_reaction=reaction_desc,
                other_tick=unit_tick,
                timestamp=now,
            )
            new_pairs.append(pair)
            matched_buffer_ids.add(buf_entry.entry_id)

    # 3. 対構成に成功したバッファエントリを除去
    updated_buffer = [
        b for b in updated_buffer
        if b.entry_id not in matched_buffer_ids
    ]

    # 4. タイムアウト処理: 一定ティック経過後も反応が到着しないエントリを保留状態に
    for buf_entry in updated_buffer:
        if not buf_entry.is_pending:
            elapsed_ticks = current_tick - buf_entry.self_tick
            if elapsed_ticks > config.buffer_timeout_ticks:
                buf_entry.is_pending = True

    # 5. バッファの上限管理
    if len(updated_buffer) > config.max_buffer:
        updated_buffer = updated_buffer[-config.max_buffer:]

    return new_pairs, updated_buffer


def _extract_reaction_description(unit: Any) -> str:
    """他者モデルリアルフィードの観測ユニットから反応記述を抽出する。

    READ-ONLYで参照し、内容を改変しない。
    """
    # ObservationUnit の構造に合わせて反応記述を取得
    # type_label と text_hint を組み合わせる
    type_label = ""
    text_hint = ""

    # fragments 属性がある場合
    fragments = getattr(unit, "fragments", None)
    if fragments:
        parts: list[str] = []
        for frag in fragments:
            frag_type = getattr(frag, "type", None)
            frag_hint = getattr(frag, "text_hint", "")
            frag_desc = getattr(frag, "source_description", "")
            if frag_type is not None:
                type_val = frag_type.value if hasattr(frag_type, "value") else str(frag_type)
                parts.append(f"{type_val}:{frag_hint or frag_desc}")
            elif frag_hint or frag_desc:
                parts.append(frag_hint or frag_desc)
        if parts:
            return "; ".join(parts)

    # 単純な text_hint や source_description 属性
    text_hint = getattr(unit, "text_hint", "")
    if text_hint:
        return text_hint

    source_desc = getattr(unit, "source_description", "")
    if source_desc:
        return source_desc

    # summary 属性
    summary = getattr(unit, "summary", "")
    if summary:
        return summary

    # type 情報のみの場合
    unit_type = getattr(unit, "type", None)
    if unit_type is not None:
        type_val = unit_type.value if hasattr(unit_type, "value") else str(unit_type)
        return f"[{type_val}]"

    return ""


# =============================================================================
# Stage 2: 対の記述 (pair description)
# =============================================================================
# 対の記述はAdjacentPairデータ構造自体が担う。
# 設計書の通り、記述は事実の列挙であり、解釈・要約・抽象化を含まない。
# 各要素は等価であり、特定要素に重みや重要度を付与しない。
# compose_pairs() 内でAdjacentPairを構成する際に全要素を設定済み。


# =============================================================================
# Stage 3: 蓄積と消失
# =============================================================================

def accumulate_pairs(
    state: InteractionAccumulationState,
    new_pairs: list[AdjacentPair],
    config: InteractionAccumulationConfig,
) -> None:
    """隣接対を時系列順に蓄積する。

    蓄積は追記形式のみで、既存の対の遡及的変更は行わない。
    上限到達時は最古の対から順にFIFO押し出し。
    押し出しは機械的であり、対の内容に基づく選択的保持は行わない。
    すべての対は等価に蓄積される。

    Args:
        state: 内部状態（in-place更新）
        new_pairs: 新規構成された隣接対
        config: 設定
    """
    for pair in new_pairs:
        state.pairs.append(pair)
        state.total_pairs_created += 1

    # FIFO押し出し（唯一の消失経路）
    if len(state.pairs) > config.max_pairs:
        overflow = len(state.pairs) - config.max_pairs
        state.pairs = state.pairs[overflow:]
        state.total_pairs_pushed_out += overflow


# =============================================================================
# Stage 4: 参照情報としての受渡準備
# =============================================================================

def prepare_enrichment_pairs(
    state: InteractionAccumulationState,
    config: InteractionAccumulationConfig,
) -> list[AdjacentPair]:
    """enrichmentへの等価列挙用に直近の隣接対を選出する。

    列挙に際して順序以外の優先度・重要度・選択基準を設けない。
    ルーミネーション防止: 同一の対が連続してenrichmentに列挙され続けることを防止する。

    Args:
        state: 内部状態
        config: 設定

    Returns:
        enrichmentに列挙する隣接対のリスト（記録順）
    """
    if not state.pairs:
        return []

    # 直近の対から候補を取得
    candidates = list(state.pairs[-config.enrichment_count * 2:])  # 余裕を持って取得

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


# =============================================================================
# Processor (4-stage pipeline)
# =============================================================================

class InteractionAccumulationProcessor:
    """相互作用の蓄積記述プロセッサ。

    4段パイプライン:
    1. 隣接対の構成: 自己表出と他者反応を時系列的近接性で対構成
    2. 対の記述: 事実の列挙（解釈・要約・抽象化を含まない）
    3. 蓄積と消失: FIFO蓄積（追記のみ、遡及的変更なし）
    4. 参照情報としての受渡準備: enrichment等価列挙 + READ-ONLY参照

    因果帰属を行わない。パターン抽出を行わない。良否判定を行わない。
    応答方針への直接的影響を行わない。判断系への経路を持たない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[InteractionAccumulationConfig] = None):
        self._config = config or InteractionAccumulationConfig()
        self._state = InteractionAccumulationState()

    @property
    def state(self) -> InteractionAccumulationState:
        return self._state

    @state.setter
    def state(self, value: InteractionAccumulationState) -> None:
        self._state = value

    # ─── Main processing entry point ──────────────────────────

    def process(
        self,
        self_records: Optional[list[Any]] = None,
        other_units: Optional[list[Any]] = None,
        current_tick: int = 0,
    ) -> int:
        """4段パイプラインの一括実行。

        入力:
        - self_records: 自己行動知覚の直近記録リスト（READ-ONLY参照）
        - other_units: 他者モデルリアルフィードの観測ユニットリスト（READ-ONLY参照）
        - current_tick: 現在のティック番号

        Returns:
            今回新規構成された隣接対の数
        """
        self._state.cycle_count += 1
        records = self_records or []
        units = other_units or []

        # Stage 1: 隣接対の構成
        new_pairs, updated_buffer = compose_pairs(
            self_records=records,
            other_units=units,
            buffer=self._state.buffer,
            config=self._config,
            current_tick=current_tick,
        )

        # Stage 2: 対の記述（AdjacentPair構造内で完了済み）

        # Stage 3: 蓄積と消失
        self._state.buffer = updated_buffer
        accumulate_pairs(self._state, new_pairs, self._config)

        logger.debug(
            "Interaction accumulation: cycle=%d, new_pairs=%d, total=%d, buffer=%d",
            self._state.cycle_count,
            len(new_pairs),
            len(self._state.pairs),
            len(self._state.buffer),
        )

        return len(new_pairs)

    # ─── Stage 4: 参照情報の提供 ──────────────────────────────

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
            # テキストは長い場合があるため切り詰め
            self_preview = pair.self_text[:80] if pair.self_text else ""
            if len(pair.self_text) > 80:
                self_preview += "..."
            other_preview = pair.other_reaction[:80] if pair.other_reaction else ""
            if len(pair.other_reaction) > 80:
                other_preview += "..."

            entries.append({
                "self_text": self_preview,
                "self_policy": pair.self_policy_label,
                "other_reaction": other_preview,
                "self_tick": pair.self_tick,
                "other_tick": pair.other_tick,
            })

        summary_text = get_interaction_summary(self._state)

        return {
            "pair_count": len(self._state.pairs),
            "entries": entries,
            "summary_text": summary_text,
        }

    def get_latest_pairs(self, count: Optional[int] = None) -> list[AdjacentPair]:
        """直近の隣接対をREAD-ONLYで返す。

        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。
        参照行為によって対の内容や順序が変化することはない。

        Args:
            count: 取得件数。Noneの場合はenrichment_count件を返す。

        Returns:
            直近の隣接対のリスト（READ-ONLY参照）
        """
        n = count if count is not None else self._config.enrichment_count
        return list(self._state.pairs[-n:])

    def get_pair_history(self) -> list[AdjacentPair]:
        """蓄積された隣接対の全リストをREAD-ONLYで返す。

        内省系構造へのREAD-ONLY参照。
        全記録を等価に返す。フィルタリング・選別・集約機能を持たない。

        Returns:
            蓄積された隣接対の全リスト（READ-ONLY参照、reference_history_count件まで）
        """
        cfg = self._config
        return list(self._state.pairs[-cfg.reference_history_count:])

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "pair_count": len(self._state.pairs),
            "buffer_count": len(self._state.buffer),
            "total_pairs_created": self._state.total_pairs_created,
            "total_pairs_pushed_out": self._state.total_pairs_pushed_out,
            "cycle_count": self._state.cycle_count,
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_interaction_summary(state: InteractionAccumulationState) -> str:
    """相互作用蓄積状態の要約（enrichment用）。

    評価判定・行動指示を含まない。等価列挙に限定する。
    パターン抽出を行わない。
    """
    if state.cycle_count == 0 and not state.pairs:
        return "相互作用蓄積: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    pair_count = len(state.pairs)
    if pair_count > 0:
        parts.append(f"蓄積対={pair_count}")
    else:
        parts.append("蓄積対=0")

    buffer_count = len(state.buffer)
    if buffer_count > 0:
        parts.append(f"バッファ={buffer_count}")

    if state.total_pairs_pushed_out > 0:
        parts.append(f"押出累計={state.total_pairs_pushed_out}")

    return " ".join(parts) if parts else "相互作用蓄積: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_interaction_accumulation_processor(
    config: Optional[InteractionAccumulationConfig] = None,
) -> InteractionAccumulationProcessor:
    """InteractionAccumulationProcessor のファクトリ関数。"""
    return InteractionAccumulationProcessor(config=config)
