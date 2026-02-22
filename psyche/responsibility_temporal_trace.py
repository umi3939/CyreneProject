"""
psyche/responsibility_temporal_trace.py - 責任の時間的推移記述

責任管理構造と責任分散構造からREAD-ONLYで値を読み取り、
スナップショットとして時系列蓄積し、変動度合いを段階値で記述する構造。

設計原則 (design_responsibility_temporal_trace.md 準拠):
- 責任値に対する善悪・多寡・適否の評価を行わない
- 責任値にスコア・レーティング・ランク等の評価的属性を付加しない
- 責任の推移から傾向・パターン・規則性を抽出しない
- 責任の推移に基づいて判断・行動・方針を変化させる経路を持たない
- 責任管理構造への書き込み（責任値の変更・分散操作の発動）を行わない
- 責任状態の「望ましい方向」を定義・示唆しない
- 推移データに基づく統計量（平均・分散・中央値等）を算出しない
- 推移データに対する移動平均・トレンド線・回帰を適用しない

3段パイプライン:
1. スナップショットの取得と蓄積 (snapshot acquisition and accumulation)
2. 段階値による推移記述 (staged-value transition description)
3. 参照情報としての受け渡し準備 (handoff preparation as reference information)

安全弁:
1. パターン抽出禁止: 蓄積データから傾向・周期・規則性を抽出しない
2. 統計量算出禁止: 平均・分散・中央値・回帰等を適用しない（値の範囲のみ使用）
3. 書き込み経路遮断: 責任管理構造・責任分散構造への書き込み経路を構造的に遮断
4. 判断層非接続: 段階値を判断バイアス計算・方針選択・安定弁に接続しない
5. enrichment等価列挙: 全断面を定義順に等価列挙。強調・選別・省略しない
6. 方向性記述の排除: 「増加傾向」「減少傾向」等の方向判定を行わない
7. FIFO自然消失の保証: 選択的削除・条件付き削除は行わない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class VariationLevel(Enum):
    """変動度合いの段階値。

    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    「増加」「減少」等の方向性を含まない。変動の大小のみ。
    """
    LARGE = "large"
    SOMEWHAT_LARGE = "somewhat_large"
    MODERATE = "moderate"
    SOMEWHAT_SMALL = "somewhat_small"
    SMALL = "small"


# =============================================================================
# Section Names (断面名の定義)
# =============================================================================

# 断面名は定義順に固定。列挙順序のランダム化・最適化は行わない。
SECTION_TOTAL_WEIGHT_VARIATION = "total_weight_variation"
SECTION_PENDING_DECISIONS_RETENTION = "pending_decisions_retention"
SECTION_HARM_VARIATION = "harm_variation"
SECTION_CONFIDENCE_VARIATION = "confidence_variation"
SECTION_DISPERSION_ACTIVITY_DENSITY = "dispersion_activity_density"

# 定義順序（固定、変更禁止）
SECTION_ORDER = [
    SECTION_TOTAL_WEIGHT_VARIATION,
    SECTION_PENDING_DECISIONS_RETENTION,
    SECTION_HARM_VARIATION,
    SECTION_CONFIDENCE_VARIATION,
    SECTION_DISPERSION_ACTIVITY_DENSITY,
]

# 断面の日本語ラベル（enrichment用、等価に列挙するためのラベル）
SECTION_LABELS = {
    SECTION_TOTAL_WEIGHT_VARIATION: "責任総重量の変動度合い",
    SECTION_PENDING_DECISIONS_RETENTION: "未評価判断の滞留度合い",
    SECTION_HARM_VARIATION: "損傷値の変動度合い",
    SECTION_CONFIDENCE_VARIATION: "自信値の変動度合い",
    SECTION_DISPERSION_ACTIVITY_DENSITY: "分散活動の密度",
}

# 段階値の日本語ラベル
VARIATION_LABELS = {
    VariationLevel.LARGE: "大",
    VariationLevel.SOMEWHAT_LARGE: "やや大",
    VariationLevel.MODERATE: "普通",
    VariationLevel.SOMEWHAT_SMALL: "やや小",
    VariationLevel.SMALL: "小",
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ResponsibilitySnapshot:
    """責任状態のスナップショット。

    毎ティック呼び出し時に責任管理構造と責任分散構造から読み取った値を
    一つのスナップショットとして記録する。
    一度記録されたら変更されない（追記のみ）。
    重み・スコア・優先度などの評価的属性を持たない（全記録等価）。
    """
    tick: int = 0
    timestamp: float = field(default_factory=time.time)

    # 責任管理構造からの読み取り値（READ-ONLY）
    total_weight: float = 0.0
    pending_decisions: int = 0
    accumulated_harm: float = 0.0
    accumulated_confidence: float = 0.0

    # 責任分散構造からの読み取り値（READ-ONLY）
    dispersion_active_weight: float = 0.0
    dispersion_active_count: int = 0
    dispersion_transformation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "total_weight": self.total_weight,
            "pending_decisions": self.pending_decisions,
            "accumulated_harm": self.accumulated_harm,
            "accumulated_confidence": self.accumulated_confidence,
            "dispersion_active_weight": self.dispersion_active_weight,
            "dispersion_active_count": self.dispersion_active_count,
            "dispersion_transformation_count": self.dispersion_transformation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponsibilitySnapshot":
        return cls(
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            total_weight=data.get("total_weight", 0.0),
            pending_decisions=data.get("pending_decisions", 0),
            accumulated_harm=data.get("accumulated_harm", 0.0),
            accumulated_confidence=data.get("accumulated_confidence", 0.0),
            dispersion_active_weight=data.get("dispersion_active_weight", 0.0),
            dispersion_active_count=data.get("dispersion_active_count", 0),
            dispersion_transformation_count=data.get("dispersion_transformation_count", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class ResponsibilityTemporalTraceState:
    """責任の時間的推移記述の内部状態。"""

    # スナップショットの時系列リスト: FIFO方式のスライディングウィンドウ
    snapshots: list[ResponsibilitySnapshot] = field(default_factory=list)

    # 断面別の段階値スナップショット: 最新の出力（断面名→段階値）
    section_snapshot: dict[str, str] = field(default_factory=dict)

    # 直前の断面別段階値スナップショット: 1回前の処理実行時
    previous_section_snapshot: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "section_snapshot": dict(self.section_snapshot),
            "previous_section_snapshot": dict(self.previous_section_snapshot),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponsibilityTemporalTraceState":
        snapshots = [
            ResponsibilitySnapshot.from_dict(s)
            for s in data.get("snapshots", [])
        ]
        return cls(
            snapshots=snapshots,
            section_snapshot=dict(data.get("section_snapshot", {})),
            previous_section_snapshot=dict(data.get("previous_section_snapshot", {})),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResponsibilityTemporalTraceConfig:
    """設定。"""

    # スナップショットのスライディングウィンドウ上限
    max_snapshots: int = 100

    # 段階値記述で使用するウィンドウ内レコードの最小件数
    # この件数未満の場合はMODERATEとする
    min_records_for_description: int = 3


# =============================================================================
# Variation Description Helpers
# =============================================================================

def _classify_variation(
    values: list[float],
    theoretical_range: float,
    min_count: int = 3,
) -> VariationLevel:
    """ウィンドウ内の値の範囲（最大値と最小値の差）を
    全体の取りうる範囲との比で段階化する。

    方向性（増加・減少）の判定は行わない。
    変動の大小のみを記述する。
    統計量（平均・分散・中央値）は算出しない。

    Args:
        values: ウィンドウ内の値のリスト
        theoretical_range: 理論上の取りうる範囲（最大値-最小値の上限）
        min_count: 段階化に必要な最小件数

    Returns:
        変動度合いの段階値
    """
    if len(values) < min_count:
        return VariationLevel.MODERATE

    if not values:
        return VariationLevel.MODERATE

    if theoretical_range <= 0:
        return VariationLevel.MODERATE

    actual_range = max(values) - min(values)
    ratio = actual_range / theoretical_range

    if ratio >= 0.6:
        return VariationLevel.LARGE
    elif ratio >= 0.3:
        return VariationLevel.SOMEWHAT_LARGE
    elif ratio >= 0.1:
        return VariationLevel.MODERATE
    elif ratio >= 0.03:
        return VariationLevel.SOMEWHAT_SMALL
    else:
        return VariationLevel.SMALL


def _classify_integer_variation(
    values: list[int],
    theoretical_range: float,
    min_count: int = 3,
) -> VariationLevel:
    """整数値リストの変動度合いを段階化する。

    _classify_variation の整数版。float に変換して委譲。
    """
    return _classify_variation(
        [float(v) for v in values],
        theoretical_range,
        min_count,
    )


def _classify_count_density(
    count_values: list[int],
    min_count: int = 3,
) -> VariationLevel:
    """分散側の変換回数がウィンドウ内でどの程度の頻度で増加したかの段階値。

    変換回数の変動（最大-最小の差）をウィンドウサイズとの比で段階化する。
    方向性（増加・減少）の判定は行わない。
    """
    if len(count_values) < min_count:
        return VariationLevel.MODERATE

    if not count_values:
        return VariationLevel.MODERATE

    actual_range = max(count_values) - min(count_values)

    # ウィンドウ内での変換回数の範囲を段階化
    # 範囲が0なら変動なし（SMALL）、大きいほどLARGE方向
    window_size = len(count_values)
    if window_size <= 0:
        return VariationLevel.MODERATE

    # 変換回数の増分をウィンドウサイズで正規化
    ratio = actual_range / max(window_size, 1)

    if ratio >= 2.0:
        return VariationLevel.LARGE
    elif ratio >= 1.0:
        return VariationLevel.SOMEWHAT_LARGE
    elif ratio >= 0.3:
        return VariationLevel.MODERATE
    elif ratio >= 0.1:
        return VariationLevel.SOMEWHAT_SMALL
    else:
        return VariationLevel.SMALL


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class ResponsibilityTemporalTraceProcessor:
    """責任の時間的推移記述プロセッサ。

    3段パイプライン:
    1. スナップショットの取得と蓄積 — 責任管理構造と責任分散構造から読み取りFIFO蓄積
    2. 段階値による推移記述 — 5断面の変動度合いを段階値で記述
    3. 参照情報としての受け渡し準備 — enrichment + READ-ONLYアクセサ

    すべての処理は記述的な変動度合いの記述であり、
    能動的な判断・評価・制御を含まない。
    出力は参照情報としてのみ流れる。

    責任管理構造・責任分散構造への書き込み経路を一切持たない。
    """

    def __init__(self, config: Optional[ResponsibilityTemporalTraceConfig] = None):
        self._config = config or ResponsibilityTemporalTraceConfig()
        self._state = ResponsibilityTemporalTraceState()

    @property
    def state(self) -> ResponsibilityTemporalTraceState:
        return self._state

    @state.setter
    def state(self, value: ResponsibilityTemporalTraceState) -> None:
        self._state = value

    # ─── Stage 1: スナップショットの取得と蓄積 ─────────────────────

    def record_snapshot(
        self,
        tick: int,
        total_weight: float,
        pending_decisions: int,
        accumulated_harm: float,
        accumulated_confidence: float,
        dispersion_active_weight: float,
        dispersion_active_count: int,
        dispersion_transformation_count: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """毎ティック呼び出し: 責任関連の値を読み取りスナップショットとして蓄積する。

        入力は全てREAD-ONLYで読み取った値であり、
        本メソッドは入力元の構造に対する書き込みを一切行わない。

        FIFO方式のスライディングウィンドウで蓄積し、
        上限到達時に最古の記録が押し出される。
        これが唯一の消失経路であり、選択的な削除・フィルタリングは行わない。

        Args:
            tick: 現在のティック番号
            total_weight: 責任総重量
            pending_decisions: 未評価判断数
            accumulated_harm: 蓄積損傷値
            accumulated_confidence: 蓄積自信値
            dispersion_active_weight: 分散側のアクティブ総重量
            dispersion_active_count: 分散側のアクティブ単位数
            dispersion_transformation_count: 分散側の変換回数
            timestamp: タイムスタンプ（指定なしの場合は現在時刻）
        """
        now = timestamp if timestamp is not None else time.time()

        snapshot = ResponsibilitySnapshot(
            tick=tick,
            timestamp=now,
            total_weight=total_weight,
            pending_decisions=pending_decisions,
            accumulated_harm=accumulated_harm,
            accumulated_confidence=accumulated_confidence,
            dispersion_active_weight=dispersion_active_weight,
            dispersion_active_count=dispersion_active_count,
            dispersion_transformation_count=dispersion_transformation_count,
        )

        self._state.snapshots.append(snapshot)

        # 上限による押し出し（唯一の消失経路）
        self._apply_pushout()

        logger.debug(
            "Responsibility snapshot recorded: tick=%d, records=%d",
            tick, len(self._state.snapshots),
        )

    def _apply_pushout(self) -> None:
        """スナップショットの上限押し出し。

        FIFO方式で最古の記録を押し出す。
        選択的削除・条件付き削除は行わない（安全弁7）。
        全スナップショットは等価に扱われ、特定の記録を保護・優遇しない。
        """
        cfg = self._config
        if len(self._state.snapshots) > cfg.max_snapshots:
            pushout_count = len(self._state.snapshots) - cfg.max_snapshots
            self._state.snapshots = self._state.snapshots[pushout_count:]

    # ─── Stage 2: 段階値による推移記述 ────────────────────────────

    def describe_variation(self) -> dict[str, str]:
        """5断面の変動度合いを段階値で記述する。

        すべての断面は等価である。断面間に優先順位・重み付けは存在しない。
        記述は「変動が大きいか小さいか」の段階表現に限定し、
        「増加している」「減少している」等の方向性の判定は行わない。

        変動度合いの算出は、ウィンドウ内の値の範囲（最大値と最小値の差）を
        全体の取りうる範囲との比で段階化する方式とする。

        Returns:
            5断面の段階値を保持する辞書（断面名→VariationLevel.value）
        """
        cfg = self._config
        snapshots = self._state.snapshots
        min_count = cfg.min_records_for_description

        # 直前スナップショットを保持（現在のスナップショットを直前に移動）
        if self._state.section_snapshot:
            self._state.previous_section_snapshot = dict(self._state.section_snapshot)

        new_snapshot: dict[str, str] = {}

        # 責任総重量の変動度合い
        total_weight_level = self._describe_total_weight_variation(snapshots, min_count)
        new_snapshot[SECTION_TOTAL_WEIGHT_VARIATION] = total_weight_level.value

        # 未評価判断の滞留度合い
        pending_level = self._describe_pending_decisions_retention(snapshots, min_count)
        new_snapshot[SECTION_PENDING_DECISIONS_RETENTION] = pending_level.value

        # 損傷値の変動度合い
        harm_level = self._describe_harm_variation(snapshots, min_count)
        new_snapshot[SECTION_HARM_VARIATION] = harm_level.value

        # 自信値の変動度合い
        confidence_level = self._describe_confidence_variation(snapshots, min_count)
        new_snapshot[SECTION_CONFIDENCE_VARIATION] = confidence_level.value

        # 分散活動の密度
        dispersion_level = self._describe_dispersion_activity_density(snapshots, min_count)
        new_snapshot[SECTION_DISPERSION_ACTIVITY_DENSITY] = dispersion_level.value

        self._state.section_snapshot = new_snapshot

        logger.debug(
            "Responsibility variation described: %s",
            {k: v for k, v in new_snapshot.items()},
        )

        return dict(new_snapshot)

    def _describe_total_weight_variation(
        self, snapshots: list[ResponsibilitySnapshot], min_count: int,
    ) -> VariationLevel:
        """責任総重量の変動度合い。

        ウィンドウ内での総重量の変動が大きいか小さいかの段階値。
        総重量の理論範囲は 0.0 - 1.0。
        """
        values = [s.total_weight for s in snapshots]
        return _classify_variation(values, theoretical_range=1.0, min_count=min_count)

    def _describe_pending_decisions_retention(
        self, snapshots: list[ResponsibilitySnapshot], min_count: int,
    ) -> VariationLevel:
        """未評価判断の滞留度合い。

        未評価判断数がウィンドウ内でどの程度滞留しているかの段階値。
        理論範囲は直近の最大値をベースに。最大20件（MAX_RECENT_DECISIONS）を理論上限とする。
        """
        values = [s.pending_decisions for s in snapshots]
        return _classify_integer_variation(
            values, theoretical_range=20.0, min_count=min_count,
        )

    def _describe_harm_variation(
        self, snapshots: list[ResponsibilitySnapshot], min_count: int,
    ) -> VariationLevel:
        """損傷値の変動度合い。

        蓄積損傷値のウィンドウ内変動の段階値。
        蓄積損傷値の理論範囲は 0.0 - 1.0。
        """
        values = [s.accumulated_harm for s in snapshots]
        return _classify_variation(values, theoretical_range=1.0, min_count=min_count)

    def _describe_confidence_variation(
        self, snapshots: list[ResponsibilitySnapshot], min_count: int,
    ) -> VariationLevel:
        """自信値の変動度合い。

        蓄積自信値のウィンドウ内変動の段階値。
        蓄積自信値の理論範囲は 0.0 - 1.0。
        """
        values = [s.accumulated_confidence for s in snapshots]
        return _classify_variation(values, theoretical_range=1.0, min_count=min_count)

    def _describe_dispersion_activity_density(
        self, snapshots: list[ResponsibilitySnapshot], min_count: int,
    ) -> VariationLevel:
        """分散活動の密度。

        分散側の変換回数がウィンドウ内でどの程度の頻度で増加したかの段階値。
        """
        values = [s.dispersion_transformation_count for s in snapshots]
        return _classify_count_density(values, min_count=min_count)

    # ─── Stage 3: 参照情報としての受け渡し準備 ──────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        各断面の段階値を等価に列挙する（安全弁5）。
        特定の断面を強調・選別しない。
        列挙順序は断面の定義順に固定。
        「注目すべき変動」「異常な推移」等の強調表現を使わない。
        方向性（「増加傾向」「減少傾向」）を記述しない（安全弁6）。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state

        summary_text = get_trace_summary(st)

        # 直近の蓄積概要（件数・最古/最新のタイムスタンプ）
        snapshot_count = len(st.snapshots)
        oldest_timestamp: Optional[float] = None
        newest_timestamp: Optional[float] = None
        if snapshot_count > 0:
            oldest_timestamp = st.snapshots[0].timestamp
            newest_timestamp = st.snapshots[-1].timestamp

        return {
            "snapshot_count": snapshot_count,
            "oldest_timestamp": oldest_timestamp,
            "newest_timestamp": newest_timestamp,
            "section_snapshot": dict(st.section_snapshot),
            "summary_text": summary_text,
        }

    def get_section_snapshot(self) -> dict[str, str]:
        """現在の断面別段階値スナップショットをREAD-ONLYで返す。

        他モジュールがREAD-ONLYで参照可能な構造化データ。
        フィルタリング・選別・集約機能をアクセサに持たせない。
        全断面を等価に返す。

        Returns:
            断面名→VariationLevel.value の辞書（コピー）
        """
        return dict(self._state.section_snapshot)

    def get_previous_section_snapshot(self) -> dict[str, str]:
        """直前の段階値スナップショットをREAD-ONLYで返す。

        現在と直前を並置するためだけに保持する。

        Returns:
            断面名→VariationLevel.value の辞書（コピー）
        """
        return dict(self._state.previous_section_snapshot)

    def get_snapshots(self) -> list[dict[str, Any]]:
        """蓄積されたスナップショットをREAD-ONLYで返す。

        フィルタリング・選別・集約機能をアクセサに持たせない。
        蓄積データの全件をそのまま返す。

        Returns:
            スナップショットの辞書リスト（コピー）
        """
        return [s.to_dict() for s in self._state.snapshots]

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "snapshot_count": len(st.snapshots),
            "has_section_snapshot": bool(st.section_snapshot),
            "has_previous_section_snapshot": bool(st.previous_section_snapshot),
            "section_snapshot": dict(st.section_snapshot),
        }

    # ─── Save / Load ──────────────────────────────────────────────

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = ResponsibilityTemporalTraceState.from_dict(data)
        logger.debug(
            "Responsibility temporal trace state loaded: snapshots=%d",
            len(self._state.snapshots),
        )


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_trace_summary(state: ResponsibilityTemporalTraceState) -> str:
    """責任の時間的推移記述の要約（enrichment用）。

    全断面を等価に列挙する（安全弁5）。
    特定の断面を強調・選別しない。
    列挙順序は断面の定義順に固定。
    「注目すべき変動」「異常な推移」等の強調表現を使わない。
    「増加傾向」「減少傾向」等の方向性を記述しない（安全弁6）。
    評価判定・行動指示を含まない。
    """
    if not state.section_snapshot:
        return "責任推移: 待機中"

    parts: list[str] = []
    for section_name in SECTION_ORDER:
        value = state.section_snapshot.get(section_name, "")
        if value:
            label = SECTION_LABELS.get(section_name, section_name)
            variation_enum = VariationLevel(value)
            variation_label = VARIATION_LABELS.get(variation_enum, value)
            parts.append(f"{label}={variation_label}")

    if not parts:
        return "責任推移: 待機中"

    return " ".join(parts)


# =============================================================================
# Factory
# =============================================================================

def create_responsibility_temporal_trace(
    config: Optional[ResponsibilityTemporalTraceConfig] = None,
) -> ResponsibilityTemporalTraceProcessor:
    """ResponsibilityTemporalTraceProcessor のファクトリ関数。"""
    return ResponsibilityTemporalTraceProcessor(config=config)
