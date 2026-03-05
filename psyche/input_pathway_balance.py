"""
psyche/input_pathway_balance.py - 入力経路間の均衡記述

3入力経路（テキスト対話、画面知覚、自発起動）の使用実績を横断的に認知し、
その分布状態を非評価的に記述する集約層。

設計原則 (design_input_pathway_balance.md 準拠):
- 3経路間の「均衡すべき比率」「望ましい分布」「理想的な使い分け」を定義しない
- 経路の使用頻度に基づいて経路選択を変更しない
- 偏りを「是正すべき問題」として記述しない
- 使用頻度が低い経路の優先度を引き上げない
- 使用頻度が高い経路を抑制しない
- 経路の使用実績に「良い使い方」「悪い使い方」の評価を付与しない
- 入力内容の解釈・分類・テキスト比較を行わない

3段パイプライン:
  第1段: 経路使用事実の収集（経路種別と時刻のみ）
  第2段: 窓内カウントによる断面構成（段階値変換）
  第3段: 断面のFIFO蓄積

安全弁（7種）:
  1. 全記録等価維持保証 — 使用事実・断面に重み・スコア・優先度を付与しない。
     3経路の段階値は等価に並置される。
  2. パターン抽出禁止 — 蓄積された使用事実および断面から傾向・周期性・統計量・相関を算出しない
  3. 経路選択遮断 — 本機能の出力を経路選択ロジックに接続しない
  4. 判断層遮断 — 本機能の出力を判断バイアス計算・方針候補生成・方針選択・安定化バルブに接続しない
  5. 頻度の単方向累積禁止 — 全期間累積カウンタを保持しない。
     窓から外れた使用事実はカウントに含まれない
  6. 既存経路安全弁の維持保証 — テキスト対話入力構造・自発起動構造の安全弁を維持し、
     迂回する構造を作らない
  7. 出力経路不拡張 — 初期実装で定義した出力経路以外を動的に追加しない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InputPathwayBalanceConfig:
    """入力経路間均衡記述モジュールの設定。"""

    # 使用事実リストの保持上限（安全弁5: 窓外の事実はカウント対象外）
    max_usage_facts: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_200"))

    # 断面履歴の最大保持件数
    max_snapshot_history: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_30"))

    # スライディングウィンドウの幅（使用事実件数）
    sliding_window_size: int = field(default_factory=lambda: coefficient_registry.get("description_common", "window_size_50"))

    # 変動記述で比較する過去断面の件数
    variation_comparison_count: int = 5


# =============================================================================
# 経路種別
# =============================================================================

PATHWAY_TEXT = "text"
PATHWAY_SCREEN = "screen"
PATHWAY_SPONTANEOUS = "spontaneous"

ALL_PATHWAYS = [PATHWAY_TEXT, PATHWAY_SCREEN, PATHWAY_SPONTANEOUS]


# =============================================================================
# 段階値の定義（列挙型）
# =============================================================================
# 段階値は「多い・中程度・少ない・なし」のような列挙型であり、
# 具体的な件数を外部に露出しない。
# 全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。

class UsageLevel(Enum):
    """経路別の窓内使用件数の段階値。
    全段階値は等価。特定段階値を望ましいとしない。
    """
    NONE = "none"           # 0件
    FEW = "few"             # 1-5件
    MODERATE = "moderate"   # 6-15件
    MANY = "many"           # 16件以上


class BiasLevel(Enum):
    """経路間の偏在度の段階値。
    全段階値は等価。特定段階値を望ましいとしない。
    偏在度は「是正すべき問題」ではない。
    """
    EVEN = "even"           # 3経路にほぼ均等に分散
    SLIGHT = "slight"       # やや偏在
    MODERATE = "moderate"   # 中程度の偏在
    CONCENTRATED = "concentrated"  # 少数経路に集中


# =============================================================================
# 段階値の決定関数
# =============================================================================

def determine_usage_level(count: int) -> UsageLevel:
    """窓内使用件数から段階値を決定する。

    段階値は等価。評価的含意を持たない。

    Args:
        count: 窓内使用件数（0以上の整数）

    Returns:
        段階値（全て等価）
    """
    if count == 0:
        return UsageLevel.NONE
    elif count <= 5:
        return UsageLevel.FEW
    elif count <= 15:
        return UsageLevel.MODERATE
    else:
        return UsageLevel.MANY


def determine_bias_level(bias_value: float) -> BiasLevel:
    """偏在度の数値から段階値を決定する。

    段階値は等価。偏在を「是正すべき問題」としない。

    Args:
        bias_value: 偏在度（0.0〜1.0）

    Returns:
        段階値（全て等価）
    """
    if bias_value < 0.2:
        return BiasLevel.EVEN
    elif bias_value < 0.45:
        return BiasLevel.SLIGHT
    elif bias_value < 0.7:
        return BiasLevel.MODERATE
    else:
        return BiasLevel.CONCENTRATED


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_bias_value(counts: dict[str, int]) -> float:
    """3経路のカウントから偏在度を計算する。

    安全弁1: この値は評価的意味を持たない。「良い分布」「悪い分布」を示さない。
    安全弁2: パターン抽出禁止。この値から「均衡すべき比率」を導出しない。

    Returns:
        0.0（完全均等または使用なし）〜 1.0（単一経路に全集中）
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    active_pathways = [c for c in counts.values() if c > 0]
    if len(active_pathways) <= 1:
        return 1.0 if total > 0 else 0.0

    n = len(ALL_PATHWAYS)
    # ジニ係数的な集中度指標
    sorted_counts = sorted(counts.get(p, 0) for p in ALL_PATHWAYS)
    weighted_sum = 0.0
    for i, c in enumerate(sorted_counts):
        weighted_sum += (i + 1) * c

    if total == 0:
        return 0.0
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

    return _clamp(gini)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class UsageFact:
    """経路使用事実。経路種別と時刻のみ。入力内容を含まない。

    安全弁1: 重み・スコア・優先度を付与しない。
    安全弁5: 窓から外れた使用事実はカウントに含まれない。
    """

    # 経路種別
    pathway: str = ""

    # 使用時刻
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageFact":
        return cls(
            pathway=data.get("pathway", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class PathwaySnapshot:
    """経路使用の断面。構成後に変更されない（不変）。

    安全弁1: 全記録等価。3経路の段階値は等価に並置される。
    安全弁2: パターン抽出禁止。断面から傾向を算出しない。
    """

    # 生成時刻
    timestamp: float = 0.0

    # 経路別の窓内使用件数の段階値（3経路分、等価に並置）
    text_usage_level: str = UsageLevel.NONE.value
    screen_usage_level: str = UsageLevel.NONE.value
    spontaneous_usage_level: str = UsageLevel.NONE.value

    # 経路間の偏在度の段階値
    bias_level: str = BiasLevel.EVEN.value

    # 偏在度の数値（変動記述導出用、enrichmentには段階値のみ露出）
    bias_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "text_usage_level": self.text_usage_level,
            "screen_usage_level": self.screen_usage_level,
            "spontaneous_usage_level": self.spontaneous_usage_level,
            "bias_level": self.bias_level,
            "bias_value": self.bias_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathwaySnapshot":
        return cls(
            timestamp=data.get("timestamp", 0.0),
            text_usage_level=data.get("text_usage_level", UsageLevel.NONE.value),
            screen_usage_level=data.get("screen_usage_level", UsageLevel.NONE.value),
            spontaneous_usage_level=data.get("spontaneous_usage_level", UsageLevel.NONE.value),
            bias_level=data.get("bias_level", BiasLevel.EVEN.value),
            bias_value=data.get("bias_value", 0.0),
        )


@dataclass
class PathwayVariation:
    """変動記述。直近の断面と過去の断面群を比較した偏在度の変動。

    安全弁2: パターン抽出禁止。「長期的にこの方向に向かっている」という
    傾向の累積的蓄積を行わない。断面生成のたびに再導出される。
    """

    # 偏在度の変動方向（正=偏在方向, 負=均等化方向, 0=変化なし）
    bias_direction: float = 0.0

    # 偏在度の変動幅（絶対値）
    bias_magnitude: float = 0.0

    # 比較に使用した過去断面の件数
    comparison_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias_direction": self.bias_direction,
            "bias_magnitude": self.bias_magnitude,
            "comparison_count": self.comparison_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathwayVariation":
        return cls(
            bias_direction=data.get("bias_direction", 0.0),
            bias_magnitude=data.get("bias_magnitude", 0.0),
            comparison_count=data.get("comparison_count", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class InputPathwayBalanceState:
    """入力経路間均衡記述モジュールの内部状態。

    安全弁1: 全記録等価。
    安全弁5: 全期間累積カウンタを保持しない。
    """

    # 使用事実リスト（時系列順、先入先出）
    usage_facts: list[UsageFact] = field(default_factory=list)

    # 断面履歴（時系列順、先入先出）
    snapshot_history: list[PathwaySnapshot] = field(default_factory=list)

    # 直近の変動記述（断面が2件以上ある場合のみ有効）
    latest_variation: Optional[PathwayVariation] = None

    # 累積カウンタ（診断情報のみ、処理分岐に使用しない）
    total_snapshots_generated: int = 0
    total_snapshots_expired: int = 0
    total_facts_expired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage_facts": [f.to_dict() for f in self.usage_facts],
            "snapshot_history": [s.to_dict() for s in self.snapshot_history],
            "latest_variation": self.latest_variation.to_dict() if self.latest_variation else None,
            "total_snapshots_generated": self.total_snapshots_generated,
            "total_snapshots_expired": self.total_snapshots_expired,
            "total_facts_expired": self.total_facts_expired,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputPathwayBalanceState":
        facts = [
            UsageFact.from_dict(f)
            for f in data.get("usage_facts", [])
        ]
        history = [
            PathwaySnapshot.from_dict(s)
            for s in data.get("snapshot_history", [])
        ]
        var_data = data.get("latest_variation")
        variation = PathwayVariation.from_dict(var_data) if var_data else None
        return cls(
            usage_facts=facts,
            snapshot_history=history,
            latest_variation=variation,
            total_snapshots_generated=data.get("total_snapshots_generated", 0),
            total_snapshots_expired=data.get("total_snapshots_expired", 0),
            total_facts_expired=data.get("total_facts_expired", 0),
        )


# =============================================================================
# 第1段: 経路使用事実の収集
# =============================================================================
# 1サイクルにつき、3経路のうちどの経路が使用されたかの事実を収集する。
# 収集される情報は「経路種別」と「時刻」のみ。
# 入力内容・処理結果・入力量は収集しない。

def collect_usage_fact(
    *,
    pathway: str,
    timestamp: Optional[float] = None,
) -> UsageFact:
    """経路使用事実を生成する。

    安全弁1: 全記録等価。使用事実に重み・スコアを付与しない。
    安全弁6: 既存経路安全弁の維持保証。本関数は経路の状態を変更しない。

    Args:
        pathway: 経路種別（PATHWAY_TEXT, PATHWAY_SCREEN, PATHWAY_SPONTANEOUS のいずれか）
        timestamp: 使用時刻（None時は現在時刻）

    Returns:
        使用事実
    """
    ts = timestamp if timestamp is not None else time.time()
    return UsageFact(
        pathway=pathway,
        timestamp=ts,
    )


def read_text_dialogue_usage(
    *,
    text_dialogue_state: Any = None,
) -> list[dict[str, Any]]:
    """テキスト対話経路の使用事実を読み取る（READ-ONLY）。

    経路種別と時刻情報のみを読み取る。テキスト内容を読み取らない。
    安全弁6: テキスト対話入力構造の安全弁を維持する。

    Args:
        text_dialogue_state: TextDialogueState（duck typing）

    Returns:
        経路種別と時刻の対のリスト
    """
    if text_dialogue_state is None:
        return []

    # duck typing: active_units を取得
    active_units = getattr(text_dialogue_state, "active_units", None)
    if active_units is None:
        state_obj = getattr(text_dialogue_state, "state", None)
        if state_obj is not None:
            active_units = getattr(state_obj, "active_units", [])
        else:
            active_units = []

    results: list[dict[str, Any]] = []
    for unit in active_units:
        route_type = getattr(unit, "route_type", None)
        if route_type is not None:
            route_val = route_type.value if hasattr(route_type, "value") else str(route_type)
            if route_val == "text":
                ts = getattr(unit, "timestamp", 0.0)
                results.append({"pathway": PATHWAY_TEXT, "timestamp": ts})

    return results


def read_spontaneous_usage(
    *,
    spontaneous_state: Any = None,
) -> list[dict[str, Any]]:
    """自発起動経路の使用事実を読み取る（READ-ONLY）。

    起動の発生事実と時刻情報のみを読み取る。起動理由や候補内容を読み取らない。
    安全弁6: 自発起動構造の安全弁を維持する。

    Args:
        spontaneous_state: SpontaneousState（duck typing）

    Returns:
        経路種別と時刻の対のリスト
    """
    if spontaneous_state is None:
        return []

    # duck typing: activation_history を取得
    history = getattr(spontaneous_state, "activation_history", None)
    if history is None:
        state_obj = getattr(spontaneous_state, "state", None)
        if state_obj is not None:
            history = getattr(state_obj, "activation_history", [])
        else:
            history = []

    results: list[dict[str, Any]] = []
    for entry in history:
        ts = getattr(entry, "timestamp", 0.0)
        # タイムスタンプが辞書の場合
        if isinstance(entry, dict):
            ts = entry.get("timestamp", 0.0)
        results.append({"pathway": PATHWAY_SPONTANEOUS, "timestamp": ts})

    return results


# =============================================================================
# 第2段: 窓内カウントによる断面構成
# =============================================================================
# 収集された使用事実を有限長のスライディングウィンドウ内でカウントし、
# 経路別の使用件数を段階値に変換する。

def compose_snapshot(
    usage_facts: list[UsageFact],
    config: InputPathwayBalanceConfig,
    timestamp: Optional[float] = None,
) -> PathwaySnapshot:
    """使用事実リストの窓内から断面を構成する。

    安全弁1: 全記録等価。断面構成時に特定経路を強調・抑制しない。
    安全弁2: パターン抽出禁止。
    安全弁5: 窓から外れた使用事実はカウントに含まれない。

    Args:
        usage_facts: 使用事実リスト
        config: 設定
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        構成された断面（不変）
    """
    ts = timestamp if timestamp is not None else time.time()

    # スライディングウィンドウ内の事実のみをカウント
    window = usage_facts[-config.sliding_window_size:]

    counts: dict[str, int] = {p: 0 for p in ALL_PATHWAYS}
    for fact in window:
        if fact.pathway in counts:
            counts[fact.pathway] += 1

    # 段階値変換
    text_level = determine_usage_level(counts[PATHWAY_TEXT])
    screen_level = determine_usage_level(counts[PATHWAY_SCREEN])
    spontaneous_level = determine_usage_level(counts[PATHWAY_SPONTANEOUS])

    # 偏在度計算
    bias_val = compute_bias_value(counts)
    bias_lv = determine_bias_level(bias_val)

    return PathwaySnapshot(
        timestamp=ts,
        text_usage_level=text_level.value,
        screen_usage_level=screen_level.value,
        spontaneous_usage_level=spontaneous_level.value,
        bias_level=bias_lv.value,
        bias_value=bias_val,
    )


# =============================================================================
# 変動記述の導出
# =============================================================================

def derive_variation(
    snapshot_history: list[PathwaySnapshot],
    config: InputPathwayBalanceConfig,
) -> Optional[PathwayVariation]:
    """断面履歴から変動記述を導出する。

    安全弁2: パターン抽出禁止。「長期的にこの方向に向かっている」という
    傾向の累積的蓄積を行わない。断面生成のたびに再導出される。

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

    # 過去の偏在度の平均
    past_bias_avg = sum(s.bias_value for s in past_snapshots) / len(past_snapshots)

    # 変動方向と幅
    bias_diff = latest.bias_value - past_bias_avg

    return PathwayVariation(
        bias_direction=bias_diff,
        bias_magnitude=abs(bias_diff),
        comparison_count=len(past_snapshots),
    )


# =============================================================================
# メイン処理: 1サイクル処理
# =============================================================================

def process_input_pathway_balance(
    state: InputPathwayBalanceState,
    *,
    current_pathway: str = "",
    text_dialogue_state: Any = None,
    spontaneous_state: Any = None,
    has_screen_input: bool = False,
    config: Optional[InputPathwayBalanceConfig] = None,
    timestamp: Optional[float] = None,
) -> InputPathwayBalanceState:
    """入力経路間均衡記述の1サイクル処理を実行する。

    オーケストレータの毎ティック処理帯で呼び出される。

    安全弁1: 全記録等価維持保証
    安全弁2: パターン抽出禁止
    安全弁3: 経路選択遮断（本関数は経路選択に対する出力経路を持たない）
    安全弁4: 判断層遮断（本関数は判断バイアス計算等に接続されない）
    安全弁5: 頻度の単方向累積禁止（窓外の事実はカウントに含まれない）
    安全弁6: 既存経路安全弁の維持保証
    安全弁7: 出力経路不拡張

    Args:
        state: 現在の状態
        current_pathway: 現在のサイクルで使用された経路種別
        text_dialogue_state: テキスト対話入力構造（READ-ONLY参照）
        spontaneous_state: 自発起動構造（READ-ONLY参照）
        has_screen_input: 画面知覚経路が使用されたかの事実
        config: 設定
        timestamp: 断面の生成時刻

    Returns:
        更新されたInputPathwayBalanceState
    """
    cfg = config or InputPathwayBalanceConfig()
    ts = timestamp if timestamp is not None else time.time()

    # ── 第1段: 経路使用事実の収集 ──
    new_facts = list(state.usage_facts)

    # 現在のサイクルで使用された経路を事実として記録
    if current_pathway and current_pathway in ALL_PATHWAYS:
        new_facts.append(collect_usage_fact(pathway=current_pathway, timestamp=ts))
    elif has_screen_input:
        new_facts.append(collect_usage_fact(pathway=PATHWAY_SCREEN, timestamp=ts))

    # 使用事実リストの上限管理（安全弁5: 窓外の事実はカウント対象外）
    facts_expired = 0
    if len(new_facts) > cfg.max_usage_facts:
        facts_expired = len(new_facts) - cfg.max_usage_facts
        new_facts = new_facts[facts_expired:]

    # ── 第2段: 窓内カウントによる断面構成 ──
    snapshot = compose_snapshot(new_facts, cfg, timestamp=ts)

    # ── 第3段: 断面のFIFO蓄積 ──
    new_history = list(state.snapshot_history)
    new_history.append(snapshot)

    snapshots_expired = 0
    if len(new_history) > cfg.max_snapshot_history:
        snapshots_expired = len(new_history) - cfg.max_snapshot_history
        new_history = new_history[snapshots_expired:]

    # ── 変動記述の再導出 ──
    # 安全弁2: 新しい断面が生成されるたびに再導出。累積蓄積しない。
    variation = derive_variation(new_history, cfg)

    new_state = InputPathwayBalanceState(
        usage_facts=new_facts,
        snapshot_history=new_history,
        latest_variation=variation,
        total_snapshots_generated=state.total_snapshots_generated + 1,
        total_snapshots_expired=state.total_snapshots_expired + snapshots_expired,
        total_facts_expired=state.total_facts_expired + facts_expired,
    )

    logger.debug(
        "Input pathway balance: text=%s, screen=%s, spontaneous=%s, "
        "bias=%s (%.4f), history=%d",
        snapshot.text_usage_level,
        snapshot.screen_usage_level,
        snapshot.spontaneous_usage_level,
        snapshot.bias_level,
        snapshot.bias_value,
        len(new_history),
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁7: 出力先は内省系構造への参照情報およびenrichment等価列挙に限定。
# 経路選択への出力経路を持たない（安全弁3）。
# 判断バイアス計算等への出力経路を持たない（安全弁4）。

def get_latest_snapshot(state: InputPathwayBalanceState) -> Optional[PathwaySnapshot]:
    """最新の断面を返す（内省系参照経路、READ-ONLY）。

    参照行為によって状態が変化することはない。

    Returns:
        直近の断面。履歴がなければ None。
    """
    if not state.snapshot_history:
        return None
    return state.snapshot_history[-1]


def get_snapshot_history(state: InputPathwayBalanceState) -> list[PathwaySnapshot]:
    """断面履歴全体を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    参照行為によって状態が変化することはない。

    Returns:
        断面履歴のコピー。
    """
    return list(state.snapshot_history)


def get_latest_variation(state: InputPathwayBalanceState) -> Optional[PathwayVariation]:
    """最新の変動記述を返す（内省系参照経路、READ-ONLY）。

    安全弁2: パターン抽出禁止。変動記述は「改善」「悪化」に相当する方向性を持たない。
    参照行為によって状態が変化することはない。

    Returns:
        直近の変動記述。断面が2件未満の場合は None。
    """
    return state.latest_variation


def get_pathway_balance_summary(state: InputPathwayBalanceState) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の経路を強調・選別しない。
    安全弁3: 経路選択遮断。
    安全弁4: 判断層遮断。

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
        summary["text_usage_level"] = latest.text_usage_level
        summary["screen_usage_level"] = latest.screen_usage_level
        summary["spontaneous_usage_level"] = latest.spontaneous_usage_level
        summary["bias_level"] = latest.bias_level
        summary["latest_timestamp"] = latest.timestamp

    if variation is not None:
        summary["variation_bias_direction"] = variation.bias_direction
        summary["variation_bias_magnitude"] = variation.bias_magnitude
        summary["variation_comparison_count"] = variation.comparison_count

    return summary


# =============================================================================
# Enrichment出力
# =============================================================================
# enrichment出力は段階値の等価列挙に限定される。
# 具体的な件数・比率・順位を含まない。
# 「偏っている」「不均衡である」等の評価的語彙を含まない。
# 経路名の優劣を示唆する語彙を使用しない。

def get_enrichment_text(state: InputPathwayBalanceState) -> str:
    """enrichment出力テキストを生成する。

    段階値の等価列挙に限定。具体的な件数・比率・順位を含まない。
    「偏っている」「不均衡である」等の評価的語彙を含まない。
    経路名の優劣を示唆する語彙を使用しない。

    安全弁1: 全記録等価。3経路の段階値は等価に並置される。
    安全弁3: 経路選択遮断。
    安全弁4: 判断層遮断。

    Returns:
        段階値の等価列挙テキスト。データがなければ「待機中」を含むテキスト。
    """
    latest = get_latest_snapshot(state)
    if latest is None:
        return "入力経路均衡: 待機中"

    parts: list[str] = [
        f"テキスト={latest.text_usage_level}",
        f"画面={latest.screen_usage_level}",
        f"自発={latest.spontaneous_usage_level}",
        f"分布={latest.bias_level}",
    ]

    variation = get_latest_variation(state)
    if variation is not None and variation.comparison_count > 0:
        if variation.bias_magnitude > 0.01:
            direction_str = "偏在方向" if variation.bias_direction > 0 else "均等化方向"
            parts.append(f"変動={direction_str}")

    return " ".join(parts)


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: InputPathwayBalanceState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> InputPathwayBalanceState:
    """永続化用の辞書から状態を復元する。"""
    return InputPathwayBalanceState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_input_pathway_balance_state() -> InputPathwayBalanceState:
    """初期状態のファクトリ関数。"""
    return InputPathwayBalanceState()


def create_input_pathway_balance_config(
    max_usage_facts: int = 200,
    max_snapshot_history: int = 30,
    sliding_window_size: int = 50,
    variation_comparison_count: int = 5,
) -> InputPathwayBalanceConfig:
    """設定のファクトリ関数。"""
    return InputPathwayBalanceConfig(
        max_usage_facts=max_usage_facts,
        max_snapshot_history=max_snapshot_history,
        sliding_window_size=sliding_window_size,
        variation_comparison_count=variation_comparison_count,
    )
