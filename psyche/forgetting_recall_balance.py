"""
psyche/forgetting_recall_balance.py - 忘却と想起の均衡記述

忘却側と想起側の状態をそれぞれ読み取り専用で参照し、
両者のスナップショットを等価に並置記述する。

設計原則 (design_forgetting_recall_balance.md 準拠):
- 忘却速度を調整・加速・減速しない
- 想起頻度を調整・促進・抑制しない
- 忘却パラメータ（希薄化速度、保護係数等）に書き込まない
- 想起閾値（候補上限、ルーミネーション窓等）に書き込まない
- 忘却と想起の「均衡すべき状態」を定義しない
- 忘却が「速すぎる」「遅すぎる」と評価しない
- 想起が「多すぎる」「少なすぎる」と評価しない
- 忘却と想起の統合指標を算出しない
- 記憶の選別・優先順位付け・取捨選択を行わない
- パターン抽出を行わない

3段パイプライン:
  段階1: 断面抽出（忘却側1構造・想起側2構造からスナップショット読み取り）
  段階2: 並置記述の生成（3断面を等価に構成）
  段階3: FIFO蓄積（有限長履歴への先入先出蓄積）

安全弁（5種）:
  1. 全記録等価維持保証 — 並置記述の全件が等価。忘却断面と想起断面の間にも序列なし
  2. 評価的変換の禁止 — 断面に含まれる値は全て件数のみの記述
  3. パターン抽出の禁止 — 蓄積された履歴から傾向・統計化を行わない
  4. 履歴の有限性 — FIFO自然消失による無限蓄積防止
  5. 出力経路の限定と不拡張 — enrichment参照情報および内省系参照情報のみ
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ForgettingRecallBalanceConfig:
    """忘却と想起の均衡記述モジュールの設定。"""

    # 並置記述の履歴の最大保持件数（安全弁4: FIFO自然消失）
    max_history: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_30"))

    # enrichment出力に含める直近件数
    enrichment_recent_count: int = 5


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ForgettingSectionSnapshot:
    """忘却断面。忘却側1構造から読み取った件数的記述。

    安全弁2: 値は全て件数のみ。評価的判定を含まない。
    """

    # 忘却段階別の系列件数
    active_count: int = 0
    weakening_count: int = 0
    fading_count: int = 0
    near_invisible_count: int = 0
    invisible_count: int = 0

    # 固定化兆候件数
    fixation_sign_count: int = 0

    # 保護中の系列件数
    protected_count: int = 0

    # 直近サイクルでの新規忘却件数
    newly_forgotten_count: int = 0

    # 直近サイクルでの復帰件数
    newly_recovered_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_count": self.active_count,
            "weakening_count": self.weakening_count,
            "fading_count": self.fading_count,
            "near_invisible_count": self.near_invisible_count,
            "invisible_count": self.invisible_count,
            "fixation_sign_count": self.fixation_sign_count,
            "protected_count": self.protected_count,
            "newly_forgotten_count": self.newly_forgotten_count,
            "newly_recovered_count": self.newly_recovered_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgettingSectionSnapshot":
        return cls(
            active_count=data.get("active_count", 0),
            weakening_count=data.get("weakening_count", 0),
            fading_count=data.get("fading_count", 0),
            near_invisible_count=data.get("near_invisible_count", 0),
            invisible_count=data.get("invisible_count", 0),
            fixation_sign_count=data.get("fixation_sign_count", 0),
            protected_count=data.get("protected_count", 0),
            newly_forgotten_count=data.get("newly_forgotten_count", 0),
            newly_recovered_count=data.get("newly_recovered_count", 0),
        )


@dataclass
class ExternalRecallSectionSnapshot:
    """外部トリガー型想起断面。多経路想起構造から読み取った件数的記述。

    安全弁2: 値は全て件数のみ。評価的判定を含まない。
    """

    # 経路別の想起候補件数
    emotional_count: int = 0
    contextual_count: int = 0
    temporal_count: int = 0

    # 候補総件数
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotional_count": self.emotional_count,
            "contextual_count": self.contextual_count,
            "temporal_count": self.temporal_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalRecallSectionSnapshot":
        return cls(
            emotional_count=data.get("emotional_count", 0),
            contextual_count=data.get("contextual_count", 0),
            temporal_count=data.get("temporal_count", 0),
            total_count=data.get("total_count", 0),
        )


@dataclass
class SpontaneousRecallSectionSnapshot:
    """自発的想起断面。自発的想起構造から読み取った件数的記述。

    安全弁2: 値は全て件数のみ。評価的判定を含まない。
    """

    # 経路別の想起候補件数
    emotion_delta_count: int = 0
    motive_assoc_count: int = 0
    fluctuation_assoc_count: int = 0

    # 候補総件数
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_delta_count": self.emotion_delta_count,
            "motive_assoc_count": self.motive_assoc_count,
            "fluctuation_assoc_count": self.fluctuation_assoc_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpontaneousRecallSectionSnapshot":
        return cls(
            emotion_delta_count=data.get("emotion_delta_count", 0),
            motive_assoc_count=data.get("motive_assoc_count", 0),
            fluctuation_assoc_count=data.get("fluctuation_assoc_count", 0),
            total_count=data.get("total_count", 0),
        )


@dataclass
class JuxtapositionEntry:
    """並置記述。3断面を等価に構成した単一エントリ。

    安全弁1: 3断面は構造的に等価。どの断面が他より重要であるという序列を持たない。
    安全弁2: 値は全て件数のみ。評価的判定を含まない。
    生成後に変更されない（不変）。
    """

    # 生成時刻
    timestamp: float = 0.0

    # 忘却断面
    forgetting: ForgettingSectionSnapshot = field(
        default_factory=ForgettingSectionSnapshot
    )

    # 外部トリガー型想起断面
    external_recall: ExternalRecallSectionSnapshot = field(
        default_factory=ExternalRecallSectionSnapshot
    )

    # 自発的想起断面
    spontaneous_recall: SpontaneousRecallSectionSnapshot = field(
        default_factory=SpontaneousRecallSectionSnapshot
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "forgetting": self.forgetting.to_dict(),
            "external_recall": self.external_recall.to_dict(),
            "spontaneous_recall": self.spontaneous_recall.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JuxtapositionEntry":
        return cls(
            timestamp=data.get("timestamp", 0.0),
            forgetting=ForgettingSectionSnapshot.from_dict(
                data.get("forgetting", {})
            ),
            external_recall=ExternalRecallSectionSnapshot.from_dict(
                data.get("external_recall", {})
            ),
            spontaneous_recall=SpontaneousRecallSectionSnapshot.from_dict(
                data.get("spontaneous_recall", {})
            ),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class ForgettingRecallBalanceState:
    """忘却と想起の均衡記述モジュールの内部状態。

    安全弁1: 全記録等価。
    安全弁3: パターン抽出禁止。過去の並置記述から傾向・統計量を導出しない。
    """

    # 並置記述の履歴（時系列順、先入先出）
    history: list[JuxtapositionEntry] = field(default_factory=list)

    # サイクルカウンタ（処理実行回数の記録のみ）
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": [e.to_dict() for e in self.history],
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgettingRecallBalanceState":
        return cls(
            history=[
                JuxtapositionEntry.from_dict(e)
                for e in data.get("history", [])
            ],
            cycle_count=data.get("cycle_count", 0),
        )


# =============================================================================
# 段階1: 断面抽出
# =============================================================================

def extract_forgetting_section(
    forgetting_state: Any,
    forgetting_result: Any,
) -> ForgettingSectionSnapshot:
    """忘却側1構造から忘却断面を読み取り専用で抽出する。

    忘却パラメータへの書き込み遮断（遮断1）:
    忘却側のいかなるパラメータ・内部変数にも書き込まない。

    Args:
        forgetting_state: 忘却/固定化のForgettingFixationState（READ-ONLY）
        forgetting_result: 忘却/固定化のForgettingFixationResult（READ-ONLY）

    Returns:
        忘却断面（件数のみの記述）
    """
    snap = ForgettingSectionSnapshot()

    # series_index から忘却段階別件数を読み取り
    if forgetting_state is not None:
        series_index = getattr(forgetting_state, "series_index", []) or []
        for rec in series_index:
            stage = getattr(rec, "forgetting_stage", "active")
            is_protected = getattr(rec, "is_protected", False)

            if is_protected:
                snap.protected_count += 1

            if stage == "active":
                snap.active_count += 1
            elif stage == "weakening":
                snap.weakening_count += 1
            elif stage == "fading":
                snap.fading_count += 1
            elif stage == "near_invisible":
                snap.near_invisible_count += 1
            elif stage == "invisible":
                snap.invisible_count += 1

    # 処理結果から固定化兆候件数・新規忘却件数・復帰件数を読み取り
    if forgetting_result is not None:
        snap.fixation_sign_count = getattr(
            forgetting_result, "newly_fixating", 0
        )
        snap.newly_forgotten_count = getattr(
            forgetting_result, "newly_forgotten", 0
        )
        snap.newly_recovered_count = getattr(
            forgetting_result, "newly_recovered", 0
        )

    return snap


def extract_external_recall_section(
    multi_path_recall_state: Any,
) -> ExternalRecallSectionSnapshot:
    """外部入力トリガー型の多経路想起構造から想起断面を読み取り専用で抽出する。

    想起閾値への書き込み遮断（遮断2）:
    想起側のいかなるパラメータ・内部変数にも書き込まない。

    Args:
        multi_path_recall_state: MultiPathRecallState（READ-ONLY）

    Returns:
        外部トリガー型想起断面（件数のみの記述）
    """
    snap = ExternalRecallSectionSnapshot()

    if multi_path_recall_state is None:
        return snap

    # path_stats から経路別件数を読み取り
    path_stats = getattr(multi_path_recall_state, "path_stats", None)
    if path_stats is not None:
        snap.emotional_count = getattr(path_stats, "emotional_count", 0)
        snap.contextual_count = getattr(path_stats, "contextual_count", 0)
        snap.temporal_count = getattr(path_stats, "temporal_count", 0)

    # current_candidates から候補総件数を読み取り
    candidates = getattr(multi_path_recall_state, "current_candidates", []) or []
    snap.total_count = len(candidates)

    return snap


def extract_spontaneous_recall_section(
    spontaneous_recall_state: Any,
) -> SpontaneousRecallSectionSnapshot:
    """内部状態トリガー型の自発的想起構造から想起断面を読み取り専用で抽出する。

    想起閾値への書き込み遮断（遮断2）:
    想起側のいかなるパラメータ・内部変数にも書き込まない。

    Args:
        spontaneous_recall_state: SpontaneousRecallState（READ-ONLY）

    Returns:
        自発的想起断面（件数のみの記述）
    """
    snap = SpontaneousRecallSectionSnapshot()

    if spontaneous_recall_state is None:
        return snap

    # path_stats から経路別件数を読み取り
    path_stats = getattr(spontaneous_recall_state, "path_stats", None)
    if path_stats is not None:
        snap.emotion_delta_count = getattr(
            path_stats, "emotion_delta_count", 0
        )
        snap.motive_assoc_count = getattr(
            path_stats, "motive_assoc_count", 0
        )
        snap.fluctuation_assoc_count = getattr(
            path_stats, "fluctuation_assoc_count", 0
        )

    # current_candidates から候補総件数を読み取り
    candidates = getattr(spontaneous_recall_state, "current_candidates", []) or []
    snap.total_count = len(candidates)

    return snap


# =============================================================================
# 段階2: 並置記述の生成
# =============================================================================

def compose_juxtaposition(
    forgetting_section: ForgettingSectionSnapshot,
    external_recall_section: ExternalRecallSectionSnapshot,
    spontaneous_recall_section: SpontaneousRecallSectionSnapshot,
    timestamp: Optional[float] = None,
) -> JuxtapositionEntry:
    """3断面を等価に並置した単一の並置記述を生成する。

    安全弁1: 3断面は構造的に等価。序列を持たない。
    安全弁2: 値は全て件数のみの記述。評価的判定を含まない。

    Args:
        forgetting_section: 忘却断面
        external_recall_section: 外部トリガー型想起断面
        spontaneous_recall_section: 自発的想起断面
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        生成された並置記述（不変）
    """
    ts = timestamp if timestamp is not None else time.time()

    return JuxtapositionEntry(
        timestamp=ts,
        forgetting=forgetting_section,
        external_recall=external_recall_section,
        spontaneous_recall=spontaneous_recall_section,
    )


# =============================================================================
# 段階3: FIFO蓄積
# =============================================================================

def accumulate_entry(
    state: ForgettingRecallBalanceState,
    entry: JuxtapositionEntry,
    config: ForgettingRecallBalanceConfig,
) -> ForgettingRecallBalanceState:
    """並置記述をFIFO履歴に蓄積する。

    安全弁4: 保持上限を超えた最古の記述はFIFOで自然消失する。

    Args:
        state: 現在の状態
        entry: 新規並置記述
        config: 設定

    Returns:
        更新されたForgettingRecallBalanceState
    """
    new_history = list(state.history)
    new_history.append(entry)

    # FIFO: 保持上限を超えた最古の記述を自然消失させる
    if len(new_history) > config.max_history:
        new_history = new_history[-config.max_history:]

    return ForgettingRecallBalanceState(
        history=new_history,
        cycle_count=state.cycle_count + 1,
    )


# =============================================================================
# メイン処理: 1サイクル処理（3段パイプライン統合）
# =============================================================================

def process_forgetting_recall_balance(
    state: ForgettingRecallBalanceState,
    *,
    forgetting_state: Any = None,
    forgetting_result: Any = None,
    multi_path_recall_state: Any = None,
    spontaneous_recall_state: Any = None,
    config: Optional[ForgettingRecallBalanceConfig] = None,
    timestamp: Optional[float] = None,
) -> ForgettingRecallBalanceState:
    """忘却と想起の均衡記述の1サイクル処理を実行する。

    3段パイプライン:
    1. 断面抽出: 忘却側1構造・想起側2構造から件数的記述を読み取り
    2. 並置記述の生成: 3断面を等価に構成
    3. FIFO蓄積: 有限長履歴への先入先出蓄積

    安全弁1: 全記録等価維持保証
    安全弁2: 評価的変換の禁止
    安全弁3: パターン抽出の禁止
    安全弁4: 履歴の有限性
    安全弁5: 出力経路の限定と不拡張

    忘却パラメータへの書き込み遮断（遮断1）
    想起閾値への書き込み遮断（遮断2）
    判断系への非接続（遮断3）

    Args:
        state: 現在の状態
        forgetting_state: 忘却/固定化の内部状態（READ-ONLY）
        forgetting_result: 忘却/固定化の処理結果（READ-ONLY）
        multi_path_recall_state: 多経路想起の内部状態（READ-ONLY）
        spontaneous_recall_state: 自発的想起の内部状態（READ-ONLY）
        config: 設定
        timestamp: 並置記述の生成時刻

    Returns:
        更新されたForgettingRecallBalanceState
    """
    cfg = config or ForgettingRecallBalanceConfig()
    ts = timestamp if timestamp is not None else time.time()

    # 段階1: 断面抽出
    forgetting_section = extract_forgetting_section(
        forgetting_state, forgetting_result,
    )
    external_recall_section = extract_external_recall_section(
        multi_path_recall_state,
    )
    spontaneous_recall_section = extract_spontaneous_recall_section(
        spontaneous_recall_state,
    )

    # 段階2: 並置記述の生成
    entry = compose_juxtaposition(
        forgetting_section,
        external_recall_section,
        spontaneous_recall_section,
        timestamp=ts,
    )

    # 段階3: FIFO蓄積
    new_state = accumulate_entry(state, entry, cfg)

    logger.debug(
        "Forgetting-recall balance: cycle=%d, "
        "forgetting(active=%d, weakening=%d, fading=%d, near_inv=%d, inv=%d), "
        "ext_recall=%d, spont_recall=%d",
        new_state.cycle_count,
        forgetting_section.active_count,
        forgetting_section.weakening_count,
        forgetting_section.fading_count,
        forgetting_section.near_invisible_count,
        forgetting_section.invisible_count,
        external_recall_section.total_count,
        spontaneous_recall_section.total_count,
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁5: 出力先はenrichmentの参照情報および内省系構造への参照情報に限定。
# 忘却処理・想起処理・判断系・感情系・動機系への出力経路を持たない。

def get_recent_entries(
    state: ForgettingRecallBalanceState,
    count: int = 5,
) -> list[JuxtapositionEntry]:
    """直近N件の並置記述を等価に列挙する（内省系参照経路、READ-ONLY）。

    安全弁1: 全件が等価。件の間に序列・重要度の差は存在しない。
    時系列上の位置情報のみが件を区別する。
    参照行為によって状態が変化することはない。

    Args:
        state: 現在の状態
        count: 返却する最大件数

    Returns:
        直近N件の並置記述のコピー
    """
    return list(state.history[-count:])


def get_history(
    state: ForgettingRecallBalanceState,
) -> list[JuxtapositionEntry]:
    """全履歴を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    参照行為によって状態が変化することはない。

    Returns:
        履歴のコピー
    """
    return list(state.history)


def get_balance_summary(
    state: ForgettingRecallBalanceState,
) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の断面を強調・選別しない。
    安全弁2: 評価的変換の禁止。
    安全弁3: パターン抽出の禁止。

    Returns:
        サマリ辞書
    """
    summary: dict[str, Any] = {
        "cycle_count": state.cycle_count,
        "history_count": len(state.history),
    }

    if state.history:
        latest = state.history[-1]
        summary["latest_forgetting"] = latest.forgetting.to_dict()
        summary["latest_external_recall"] = latest.external_recall.to_dict()
        summary["latest_spontaneous_recall"] = latest.spontaneous_recall.to_dict()
        summary["latest_timestamp"] = latest.timestamp

    return summary


# =============================================================================
# Enrichment出力
# =============================================================================
# enrichment出力は段階別件数と経路別件数の数量的記述のみに限定される。
# 「均衡している」「不均衡である」等の評価的判定を含まない。

def get_enrichment_text(state: ForgettingRecallBalanceState) -> str:
    """enrichment出力テキストを生成する。

    段階別件数と経路別件数の数量的記述のみ。
    「均衡している」「不均衡である」「忘却が優勢である」
    「想起が不足している」等の評価的判定を含まない。

    安全弁1: 全記録等価。忘却断面と想起断面を等価に並置する。
    安全弁2: 評価的変換の禁止。
    安全弁5: 出力経路の限定。

    Returns:
        段階別件数と経路別件数の数量的記述テキスト。
        データがなければ「待機中」を含むテキスト。
    """
    if not state.history:
        return "忘却想起均衡: 待機中"

    latest = state.history[-1]
    f = latest.forgetting
    er = latest.external_recall
    sr = latest.spontaneous_recall

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    # 忘却段階別件数（等価に並置）
    stage_parts: list[str] = []
    if f.active_count > 0:
        stage_parts.append(f"活性={f.active_count}")
    if f.weakening_count > 0:
        stage_parts.append(f"弱体化={f.weakening_count}")
    if f.fading_count > 0:
        stage_parts.append(f"退色={f.fading_count}")
    if f.near_invisible_count > 0:
        stage_parts.append(f"準不可視={f.near_invisible_count}")
    if f.invisible_count > 0:
        stage_parts.append(f"不可視={f.invisible_count}")
    if stage_parts:
        parts.append("忘却(" + ",".join(stage_parts) + ")")

    if f.fixation_sign_count > 0:
        parts.append(f"固定化兆候={f.fixation_sign_count}")
    if f.protected_count > 0:
        parts.append(f"保護={f.protected_count}")
    if f.newly_forgotten_count > 0:
        parts.append(f"新規忘却={f.newly_forgotten_count}")
    if f.newly_recovered_count > 0:
        parts.append(f"復帰={f.newly_recovered_count}")

    # 外部トリガー型想起（経路別件数、等価に並置）
    ext_parts: list[str] = []
    if er.emotional_count > 0:
        ext_parts.append(f"感情={er.emotional_count}")
    if er.contextual_count > 0:
        ext_parts.append(f"文脈={er.contextual_count}")
    if er.temporal_count > 0:
        ext_parts.append(f"時間={er.temporal_count}")
    if ext_parts:
        parts.append("外部想起(" + ",".join(ext_parts) + ")")

    # 自発的想起（経路別件数、等価に並置）
    sp_parts: list[str] = []
    if sr.emotion_delta_count > 0:
        sp_parts.append(f"感情変動={sr.emotion_delta_count}")
    if sr.motive_assoc_count > 0:
        sp_parts.append(f"動機={sr.motive_assoc_count}")
    if sr.fluctuation_assoc_count > 0:
        sp_parts.append(f"揺らぎ={sr.fluctuation_assoc_count}")
    if sp_parts:
        parts.append("自発想起(" + ",".join(sp_parts) + ")")

    return " ".join(parts) if parts else "忘却想起均衡: 待機中"


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: ForgettingRecallBalanceState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> ForgettingRecallBalanceState:
    """永続化用の辞書から状態を復元する。"""
    return ForgettingRecallBalanceState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_forgetting_recall_balance_state() -> ForgettingRecallBalanceState:
    """初期状態のファクトリ関数。"""
    return ForgettingRecallBalanceState()


def create_forgetting_recall_balance_config(
    max_history: int | None = None,
    enrichment_recent_count: int = 5,
) -> ForgettingRecallBalanceConfig:
    """設定のファクトリ関数。"""
    kwargs: dict[str, Any] = {"enrichment_recent_count": enrichment_recent_count}
    if max_history is not None:
        kwargs["max_history"] = max_history
    return ForgettingRecallBalanceConfig(**kwargs)
