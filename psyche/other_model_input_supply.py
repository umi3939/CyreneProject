"""
psyche/other_model_input_supply.py - 他者モデル入力供給構造

other_agent_model モジュールへの入力供給を統一管理する。
external_context と reaction_log が常に None で渡されていた問題を解消し、
観測情報を正規化・欠損補完・時間順整列して受け渡す。

設計原則 (other_agent_model_input_supply_design.md 準拠):
- 入力供給は「観測情報の受け渡し」を成立させるためだけに設ける
- 他者状態の断定・出力方針の固定・評価軸の導入・判断層への直接介入を絶対にしない
- 供給口は一箇所に統一し、入力生成側を複数化する
- 供給単位には時刻・由来・欠損タグを必須化する
- 循環参照防止: 推測結果を同一周期の入力生成へ戻さない
- 観測欠損時は前回値固定ではなく中立包みへ切替
- 減衰と競合保持を常時有効にする
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SupplyEntry:
    """供給単位のメタ情報。由来・時刻・欠損タグを保持。"""
    source_type: str = "periodic"      # "periodic" | "event"
    timestamp: float = 0.0
    missing_reason: str = ""           # "" | "unobserved" | "stale"


@dataclass
class ContextSnapshot:
    """外部文脈スナップショット。ExternalContext 互換属性を持つ。

    other_agent_model の extract_from_external_context が duck typing で
    responsiveness, weight, pace, density, continuity を読む。
    """
    pace: float = 0.5
    weight: float = 0.5
    density: float = 0.5
    continuity: float = 0.5
    responsiveness: float = 0.5
    timestamp: float = 0.0
    missing_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pace": self.pace,
            "weight": self.weight,
            "density": self.density,
            "continuity": self.continuity,
            "responsiveness": self.responsiveness,
            "timestamp": self.timestamp,
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSnapshot:
        return cls(
            pace=data.get("pace", 0.5),
            weight=data.get("weight", 0.5),
            density=data.get("density", 0.5),
            continuity=data.get("continuity", 0.5),
            responsiveness=data.get("responsiveness", 0.5),
            timestamp=data.get("timestamp", 0.0),
            missing_reason=data.get("missing_reason", ""),
        )


@dataclass
class ReactionBufferEntry:
    """反応履歴バッファの1エントリ。STM StimulusEntry と対応する属性を保持。"""
    source_text: str = ""
    intent: str = "unknown"
    emotion_label: str = "neutral"
    valence: float = 0.0
    timestamp: float = 0.0
    supplied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "intent": self.intent,
            "emotion_label": self.emotion_label,
            "valence": self.valence,
            "timestamp": self.timestamp,
            "supplied": self.supplied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReactionBufferEntry:
        return cls(
            source_text=data.get("source_text", ""),
            intent=data.get("intent", "unknown"),
            emotion_label=data.get("emotion_label", "neutral"),
            valence=data.get("valence", 0.0),
            timestamp=data.get("timestamp", 0.0),
            supplied=data.get("supplied", False),
        )


@dataclass
class InputSupplyState:
    """入力供給の全体状態。

    context_snapshot: 直近の外部文脈スナップショット
    reaction_buffer: 反応履歴バッファ
    supply_cursor: 供給済み位置の進行指標 (循環参照防止)
    last_supply_time: 最後に供給した時刻
    decay_rate: バッファ要素の減衰率
    """
    context_snapshot: ContextSnapshot = field(default_factory=ContextSnapshot)
    reaction_buffer: list[ReactionBufferEntry] = field(default_factory=list)
    supply_cursor: int = 0
    last_supply_time: float = 0.0
    decay_rate: float = 0.05
    max_buffer_size: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_snapshot": self.context_snapshot.to_dict(),
            "reaction_buffer": [e.to_dict() for e in self.reaction_buffer],
            "supply_cursor": self.supply_cursor,
            "last_supply_time": self.last_supply_time,
            "decay_rate": self.decay_rate,
            "max_buffer_size": self.max_buffer_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputSupplyState:
        ctx_data = data.get("context_snapshot", {})
        buf_data = data.get("reaction_buffer", [])
        return cls(
            context_snapshot=ContextSnapshot.from_dict(ctx_data) if ctx_data else ContextSnapshot(),
            reaction_buffer=[ReactionBufferEntry.from_dict(e) for e in buf_data],
            supply_cursor=data.get("supply_cursor", 0),
            last_supply_time=data.get("last_supply_time", 0.0),
            decay_rate=data.get("decay_rate", 0.05),
            max_buffer_size=data.get("max_buffer_size", 20),
        )


# =============================================================================
# STM互換の反応ログラッパー
# =============================================================================

class _ReactionLogProxy:
    """ShortTermMemory 互換の反応ログプロキシ。

    other_agent_model の extract_from_reaction_log が duck typing で
    entries リストの source_text, intent, emotion_label, valence を読む。
    """

    def __init__(self, entries: list[ReactionBufferEntry]):
        self.entries = entries


# =============================================================================
# Core Functions
# =============================================================================

def create_input_supply() -> InputSupplyState:
    """初期状態の InputSupplyState を生成する。"""
    return InputSupplyState(
        context_snapshot=ContextSnapshot(
            missing_reason="unobserved",
            timestamp=time.time(),
        ),
    )


def update_from_percept(
    state: InputSupplyState,
    percept: Optional[Any] = None,
    stm: Optional[Any] = None,
    dynamics: Optional[Any] = None,
    psyche: Optional[Any] = None,
) -> InputSupplyState:
    """周期更新: percept・STM・dynamics・psyche状態からスナップショットとバッファを更新する。

    Args:
        state: 現在の InputSupplyState
        percept: Percept オブジェクト (topics, emotion_valence 等)
        stm: ShortTermMemory オブジェクト (entries, max_entries, context_continuity_score)
        dynamics: DynamicsState オブジェクト
        psyche: PsycheState オブジェクト (mood, emotions)

    Returns:
        更新された InputSupplyState
    """
    now = time.time()

    # ── Context snapshot 計算 ──
    pace = 0.5
    weight = 0.5
    density = 0.5
    continuity_val = 0.5
    responsiveness = 0.5
    missing_reason = ""

    has_any_input = False

    # pace: STMエントリ数 / max_entries (多い=速い)
    if stm is not None and hasattr(stm, "entries") and hasattr(stm, "max_entries"):
        entries = getattr(stm, "entries", [])
        max_entries = getattr(stm, "max_entries", 10)
        if max_entries > 0:
            pace = min(1.0, len(entries) / max_entries)
        has_any_input = True

        # continuity: stm.context_continuity_score
        continuity_val = getattr(stm, "context_continuity_score", 0.5)

        # responsiveness: 直近エントリの有無・鮮度から算出
        if entries:
            latest = entries[-1]
            latest_ts = getattr(latest, "timestamp", 0.0)
            age = now - latest_ts
            # 10秒以内なら高反応性、60秒以上で低反応性
            if age < 10.0:
                responsiveness = 0.8
            elif age < 30.0:
                responsiveness = 0.6
            elif age < 60.0:
                responsiveness = 0.4
            else:
                responsiveness = 0.2
            has_any_input = True
        else:
            responsiveness = 0.3

    # weight: abs(mood.valence) + arousal の平均 (重い雰囲気)
    if psyche is not None and hasattr(psyche, "mood"):
        mood = getattr(psyche, "mood", None)
        if mood is not None:
            mood_valence = abs(getattr(mood, "valence", 0.0))
            arousal = getattr(mood, "arousal", 0.5)
            weight = min(1.0, (mood_valence + arousal) / 2.0)
            has_any_input = True

    # density: len(percept.topics) / 5.0 (トピック密度)
    if percept is not None and hasattr(percept, "topics"):
        topics = getattr(percept, "topics", [])
        density = min(1.0, len(topics) / 5.0)
        has_any_input = True

    if not has_any_input:
        missing_reason = "unobserved"

    context = ContextSnapshot(
        pace=_clamp(pace),
        weight=_clamp(weight),
        density=_clamp(density),
        continuity=_clamp(continuity_val),
        responsiveness=_clamp(responsiveness),
        timestamp=now,
        missing_reason=missing_reason,
    )

    # ── Reaction buffer 更新: STMエントリから追記 ──
    new_buffer = list(state.reaction_buffer)

    if stm is not None and hasattr(stm, "entries"):
        entries = getattr(stm, "entries", [])
        # 既存バッファにないエントリのみ追加 (timestampで判定)
        existing_timestamps = {e.timestamp for e in new_buffer}
        for entry in entries:
            entry_ts = getattr(entry, "timestamp", 0.0)
            if entry_ts not in existing_timestamps:
                new_entry = ReactionBufferEntry(
                    source_text=getattr(entry, "source_text", "")[:200],
                    intent=getattr(entry, "intent", "unknown"),
                    emotion_label=getattr(entry, "emotion_label", "neutral"),
                    valence=getattr(entry, "valence", 0.0),
                    timestamp=entry_ts,
                    supplied=False,
                )
                new_buffer.append(new_entry)
                existing_timestamps.add(entry_ts)

    # バッファサイズ制限 (古い方から削除)
    while len(new_buffer) > state.max_buffer_size:
        new_buffer.pop(0)

    return InputSupplyState(
        context_snapshot=context,
        reaction_buffer=new_buffer,
        supply_cursor=state.supply_cursor,
        last_supply_time=state.last_supply_time,
        decay_rate=state.decay_rate,
        max_buffer_size=state.max_buffer_size,
    )


def decay_buffer(state: InputSupplyState, current_time: float) -> InputSupplyState:
    """古い要素の自然減衰。一定時間経過した要素に missing_reason="stale" を設定。

    Args:
        state: 現在の InputSupplyState
        current_time: 現在時刻

    Returns:
        減衰適用後の InputSupplyState
    """
    stale_threshold = 120.0  # 2分以上前のエントリはstale

    new_buffer: list[ReactionBufferEntry] = []
    for entry in state.reaction_buffer:
        age = current_time - entry.timestamp
        if age > stale_threshold * 3:
            # 6分以上経過 → 完全に除去
            continue
        new_buffer.append(entry)

    # context snapshot も時間経過で stale 化
    ctx = state.context_snapshot
    ctx_age = current_time - ctx.timestamp if ctx.timestamp > 0 else 0.0
    if ctx_age > stale_threshold and ctx.missing_reason == "":
        ctx = ContextSnapshot(
            pace=ctx.pace,
            weight=ctx.weight,
            density=ctx.density,
            continuity=ctx.continuity,
            responsiveness=ctx.responsiveness,
            timestamp=ctx.timestamp,
            missing_reason="stale",
        )

    return InputSupplyState(
        context_snapshot=ctx,
        reaction_buffer=new_buffer,
        supply_cursor=state.supply_cursor,
        last_supply_time=current_time,
        decay_rate=state.decay_rate,
        max_buffer_size=state.max_buffer_size,
    )


def supply_context(state: InputSupplyState) -> ContextSnapshot:
    """context_snapshot を供給する。未観測時は中立値 + missing_reason="unobserved" を返す。

    Args:
        state: 現在の InputSupplyState

    Returns:
        ContextSnapshot (ExternalContext duck typing 互換)
    """
    ctx = state.context_snapshot
    if ctx.missing_reason == "unobserved" or ctx.timestamp == 0.0:
        return ContextSnapshot(
            pace=0.5,
            weight=0.5,
            density=0.5,
            continuity=0.5,
            responsiveness=0.5,
            timestamp=ctx.timestamp,
            missing_reason="unobserved",
        )
    return ctx


def supply_reaction_log(state: InputSupplyState) -> Optional[_ReactionLogProxy]:
    """反応バッファからSTM互換形式のオブジェクトを生成する。

    供給済み要素は supplied=True にするが削除しない。
    supply_cursor 以降の未供給要素を中心に供給する。

    Args:
        state: 現在の InputSupplyState

    Returns:
        ShortTermMemory 互換オブジェクト (entries属性を持つ)。
        バッファが空の場合は None。
    """
    if not state.reaction_buffer:
        return None

    # 全エントリを供給 (supplied済みも含む、再供給可能)
    # ただし stale でないものを優先
    entries = state.reaction_buffer

    # 供給済みマーキング & cursor 更新
    for entry in entries:
        entry.supplied = True
    state.supply_cursor = len(entries)
    state.last_supply_time = time.time()

    return _ReactionLogProxy(entries)


def get_input_supply_summary(state: InputSupplyState) -> str:
    """入力供給状態のサマリ文字列を返す。

    Args:
        state: 現在の InputSupplyState

    Returns:
        人間可読なサマリ文字列
    """
    ctx = state.context_snapshot
    buf_total = len(state.reaction_buffer)
    buf_supplied = sum(1 for e in state.reaction_buffer if e.supplied)
    buf_unsupplied = buf_total - buf_supplied

    parts = [
        f"ctx(pace={ctx.pace:.2f},weight={ctx.weight:.2f},"
        f"density={ctx.density:.2f},resp={ctx.responsiveness:.2f})",
    ]
    if ctx.missing_reason:
        parts.append(f"ctx_status={ctx.missing_reason}")

    parts.append(f"buf={buf_total}(new={buf_unsupplied},supplied={buf_supplied})")

    return "; ".join(parts)


# =============================================================================
# Internal Helpers
# =============================================================================

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """値を [lo, hi] にクランプする。"""
    return max(lo, min(hi, v))
