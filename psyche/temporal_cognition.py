"""
psyche/temporal_cognition.py - 時間認知構造（時間的特徴記述）

ティック番号と経過秒を入力として受け取り、加えて各モジュールが保持する蓄積統計を
参照して、時間の経過に関する複数の特徴量を並立的に記述する構造を提供する。

設計原則 (design_temporal_cognition.md 準拠):
- 既存のティック数ベース処理を変更しない
- 他モジュールの処理パラメータを直接変更しない
- 「体感時間」を単一の数値として出力しない
- 時間と感情の固定的紐付けを導入しない
- 時間的特徴量に基づく判断・評価を行わない
- 感情パイプライン（Phase 1-2）のパラメータを変更しない
- 自己差分認知の責務を侵食しない
- すべての断面の特徴量は等価であり、特定の断面に重みや重要度を付与しない
- 特徴量のパターン抽出・傾向化・統計処理を行わない
- 6断面の特徴量を1つの「体感時間」変数に統合しない
- enrichment内での強調禁止

3段パイプライン:
1. 経過情報の蓄積 (elapsed record accumulation)
2. 多断面での時間的特徴量の記述 (multi-section temporal feature description)
3. 参照情報としての受渡準備 (handoff preparation as reference information)
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

class DensityLevel(Enum):
    """時間的特徴量の段階値。

    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    """
    DENSE = "dense"
    SOMEWHAT_DENSE = "somewhat_dense"
    NORMAL = "normal"
    SOMEWHAT_SPARSE = "somewhat_sparse"
    SPARSE = "sparse"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ElapsedRecord:
    """経過記録。ティック番号・経過秒・タイムスタンプの組。

    一度記録されたら変更されない（追記のみ）。
    重み・スコア・優先度などの評価的属性を持たない（全記録等価）。
    """
    tick: int = 0
    delta_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "delta_time": self.delta_time,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElapsedRecord":
        return cls(
            tick=data.get("tick", 0),
            delta_time=data.get("delta_time", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class TemporalCognitionState:
    """時間認知の内部状態。"""

    # 経過記録のスライディングウィンドウ: 時系列順に蓄積
    elapsed_records: list[ElapsedRecord] = field(default_factory=list)

    # 断面別の時間的特徴量スナップショット: 6断面それぞれの最新の特徴量
    snapshot: dict[str, str] = field(default_factory=dict)

    # 直前の特徴量スナップショット: 1回前の処理実行時のスナップショット
    previous_snapshot: dict[str, str] = field(default_factory=dict)

    # 外部入力到着記録: タイムスタンプの時系列リスト
    external_input_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_records": [r.to_dict() for r in self.elapsed_records],
            "snapshot": dict(self.snapshot),
            "previous_snapshot": dict(self.previous_snapshot),
            "external_input_timestamps": list(self.external_input_timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalCognitionState":
        elapsed_records = [
            ElapsedRecord.from_dict(r)
            for r in data.get("elapsed_records", [])
        ]
        return cls(
            elapsed_records=elapsed_records,
            snapshot=dict(data.get("snapshot", {})),
            previous_snapshot=dict(data.get("previous_snapshot", {})),
            external_input_timestamps=list(data.get("external_input_timestamps", [])),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TemporalCognitionConfig:
    """設定。"""
    # 経過記録のスライディングウィンドウ上限
    max_elapsed_records: int = 100

    # 外部入力到着記録の上限
    max_external_input_records: int = 100

    # enrichment に含める断面情報のフォーマット設定（列挙順序は断面の定義順に固定）
    # これは設定可能な値ではなく、断面定義順が固定であることの表明

    # 特徴量記述で使用するウィンドウ内レコードの最小件数
    # この件数未満の場合はNORMALとする
    min_records_for_description: int = 3


# =============================================================================
# Section Names (断面名の定義)
# =============================================================================

# 断面名は定義順に固定。列挙順序のランダム化・最適化は行わない。
SECTION_ACTIVITY_DENSITY = "activity_density"
SECTION_MEMORY_INTERVAL = "memory_interval"
SECTION_EMOTION_FREQUENCY = "emotion_frequency"
SECTION_NARRATIVE_INTERVAL = "narrative_interval"
SECTION_EXTERNAL_INPUT_INTERVAL = "external_input_interval"
SECTION_OVERALL_ELAPSED = "overall_elapsed"

# 定義順序（固定、変更禁止）
SECTION_ORDER = [
    SECTION_ACTIVITY_DENSITY,
    SECTION_MEMORY_INTERVAL,
    SECTION_EMOTION_FREQUENCY,
    SECTION_NARRATIVE_INTERVAL,
    SECTION_EXTERNAL_INPUT_INTERVAL,
    SECTION_OVERALL_ELAPSED,
]

# 断面の日本語ラベル（enrichment用、等価に列挙するためのラベル）
SECTION_LABELS = {
    SECTION_ACTIVITY_DENSITY: "活動密度",
    SECTION_MEMORY_INTERVAL: "記憶蓄積間隔",
    SECTION_EMOTION_FREQUENCY: "感情変動頻度",
    SECTION_NARRATIVE_INTERVAL: "物語断片間隔",
    SECTION_EXTERNAL_INPUT_INTERVAL: "外部入力間隔",
    SECTION_OVERALL_ELAPSED: "総合経過",
}

# 段階値の日本語ラベル
DENSITY_LABELS = {
    DensityLevel.DENSE: "密",
    DensityLevel.SOMEWHAT_DENSE: "やや密",
    DensityLevel.NORMAL: "普通",
    DensityLevel.SOMEWHAT_SPARSE: "やや疎",
    DensityLevel.SPARSE: "疎",
}


# =============================================================================
# Feature Description Helpers
# =============================================================================

def _classify_interval_density(intervals: list[float], min_count: int = 3) -> DensityLevel:
    """間隔リストから密度レベルを記述する。

    各断面で共通に使用する段階値記述ヘルパー。
    閾値は相対的（平均との比較）であり、固定的な数値判定ではない。
    判断・評価は含まない。段階的記述のみ。
    """
    if len(intervals) < min_count:
        return DensityLevel.NORMAL

    if not intervals:
        return DensityLevel.NORMAL

    avg = sum(intervals) / len(intervals)

    if avg <= 0:
        return DensityLevel.NORMAL

    # 直近の間隔を参照（最新の3件または全件の少ない方）
    recent_count = min(3, len(intervals))
    recent = intervals[-recent_count:]
    recent_avg = sum(recent) / len(recent)

    ratio = recent_avg / avg

    if ratio < 0.5:
        return DensityLevel.DENSE
    elif ratio < 0.8:
        return DensityLevel.SOMEWHAT_DENSE
    elif ratio <= 1.2:
        return DensityLevel.NORMAL
    elif ratio <= 2.0:
        return DensityLevel.SOMEWHAT_SPARSE
    else:
        return DensityLevel.SPARSE


def _classify_frequency(count: int, window_size: int, min_count: int = 3) -> DensityLevel:
    """ウィンドウ内での頻度から密度レベルを記述する。

    頻度が高い場合はDENSE方向、低い場合はSPARSE方向。
    """
    if window_size < min_count:
        return DensityLevel.NORMAL

    if window_size <= 0:
        return DensityLevel.NORMAL

    ratio = count / window_size

    if ratio >= 0.8:
        return DensityLevel.DENSE
    elif ratio >= 0.5:
        return DensityLevel.SOMEWHAT_DENSE
    elif ratio >= 0.2:
        return DensityLevel.NORMAL
    elif ratio >= 0.1:
        return DensityLevel.SOMEWHAT_SPARSE
    else:
        return DensityLevel.SPARSE


def _classify_tempo(avg_interval: float, min_count: int = 3, record_count: int = 0) -> DensityLevel:
    """平均ティック間隔から総合テンポを記述する。

    間隔が短い場合はDENSE方向、長い場合はSPARSE方向。
    """
    if record_count < min_count:
        return DensityLevel.NORMAL

    if avg_interval <= 0:
        return DensityLevel.NORMAL

    # 平均間隔に基づく段階記述
    # 1秒以下は密、5秒以下はやや密、30秒以下は普通、120秒以下はやや疎、それ以上は疎
    if avg_interval <= 1.0:
        return DensityLevel.DENSE
    elif avg_interval <= 5.0:
        return DensityLevel.SOMEWHAT_DENSE
    elif avg_interval <= 30.0:
        return DensityLevel.NORMAL
    elif avg_interval <= 120.0:
        return DensityLevel.SOMEWHAT_SPARSE
    else:
        return DensityLevel.SPARSE


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class TemporalCognitionProcessor:
    """時間認知プロセッサ。

    3段パイプライン:
    1. 経過情報の蓄積 — ティック番号・経過秒・タイムスタンプを蓄積
    2. 多断面での時間的特徴量の記述 — 6断面の段階値を記述
    3. 参照情報としての受渡準備 — enrichment + READ-ONLYアクセサ

    すべての処理は記述的な特徴量の算出・整理であり、能動的な判断・評価・制御を含まない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[TemporalCognitionConfig] = None):
        self._config = config or TemporalCognitionConfig()
        self._state = TemporalCognitionState()

    @property
    def state(self) -> TemporalCognitionState:
        return self._state

    @state.setter
    def state(self, value: TemporalCognitionState) -> None:
        self._state = value

    # ─── Stage 1: 経過情報の蓄積 ───────────────────────────────────

    def accumulate_elapsed(
        self,
        tick: int,
        delta_time: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """毎ティック呼び出し: 経過情報を蓄積する。

        スライディングウィンドウ方式で保持し、上限到達時は最古から押し出す。

        Args:
            tick: 現在のティック番号
            delta_time: 前回からの経過秒
            timestamp: 蓄積時点のタイムスタンプ（指定なしの場合は現在時刻）
        """
        now = timestamp if timestamp is not None else time.time()

        record = ElapsedRecord(
            tick=tick,
            delta_time=delta_time,
            timestamp=now,
        )

        self._state.elapsed_records.append(record)

        # 上限による押し出し（唯一の消失経路）
        self._apply_elapsed_pushout()

        logger.debug(
            "Elapsed record accumulated: tick=%d, delta_time=%.3f, records=%d",
            tick, delta_time, len(self._state.elapsed_records),
        )

    def _apply_elapsed_pushout(self) -> None:
        """経過記録の上限押し出し。"""
        cfg = self._config
        if len(self._state.elapsed_records) > cfg.max_elapsed_records:
            pushout_count = len(self._state.elapsed_records) - cfg.max_elapsed_records
            self._state.elapsed_records = self._state.elapsed_records[pushout_count:]

    # ─── 外部入力到着記録 ──────────────────────────────────────

    def notify_external_input(self, timestamp: Optional[float] = None) -> None:
        """外部入力（知覚入力）の到着を記録する。

        外部入力間隔断面の記述に使用する。
        上限到達時は最古から押し出す。

        Args:
            timestamp: 到着時点のタイムスタンプ（指定なしの場合は現在時刻）
        """
        now = timestamp if timestamp is not None else time.time()
        self._state.external_input_timestamps.append(now)

        # 上限による押し出し
        cfg = self._config
        if len(self._state.external_input_timestamps) > cfg.max_external_input_records:
            pushout_count = len(self._state.external_input_timestamps) - cfg.max_external_input_records
            self._state.external_input_timestamps = self._state.external_input_timestamps[pushout_count:]

        logger.debug(
            "External input notified: timestamp=%.3f, records=%d",
            now, len(self._state.external_input_timestamps),
        )

    # ─── Stage 2: 多断面での時間的特徴量の記述 ─────────────────────

    def describe_features(
        self,
        episodic_timestamps: Optional[list[float]] = None,
        emotion_change_count: int = 0,
        narrative_timestamps: Optional[list[float]] = None,
    ) -> dict[str, str]:
        """6断面の時間的特徴量を記述する。

        各断面は独立であり、断面間の優先順位・重み付け・統合処理は存在しない。
        すべての断面は等価である。

        Args:
            episodic_timestamps: エピソード記憶のタイムスタンプリスト（READ-ONLY参照）
            emotion_change_count: 直近の感情変動回数（READ-ONLY参照）
            narrative_timestamps: 自己物語断片のタイムスタンプリスト（READ-ONLY参照）

        Returns:
            6断面の特徴量を保持する辞書（断面名→DensityLevel.value）
        """
        cfg = self._config
        records = self._state.elapsed_records
        min_count = cfg.min_records_for_description

        # 直前スナップショットを保持（現在のスナップショットを直前に移動）
        if self._state.snapshot:
            self._state.previous_snapshot = dict(self._state.snapshot)

        new_snapshot: dict[str, str] = {}

        # 活動密度断面: ティック発生間隔の分布特性
        activity_level = self._describe_activity_density(records, min_count)
        new_snapshot[SECTION_ACTIVITY_DENSITY] = activity_level.value

        # 記憶蓄積間隔断面
        memory_level = self._describe_memory_interval(
            episodic_timestamps or [], min_count,
        )
        new_snapshot[SECTION_MEMORY_INTERVAL] = memory_level.value

        # 感情変動頻度断面
        emotion_level = self._describe_emotion_frequency(
            emotion_change_count, len(records), min_count,
        )
        new_snapshot[SECTION_EMOTION_FREQUENCY] = emotion_level.value

        # 物語断片間隔断面
        narrative_level = self._describe_narrative_interval(
            narrative_timestamps or [], min_count,
        )
        new_snapshot[SECTION_NARRATIVE_INTERVAL] = narrative_level.value

        # 外部入力間隔断面
        external_level = self._describe_external_input_interval(min_count)
        new_snapshot[SECTION_EXTERNAL_INPUT_INTERVAL] = external_level.value

        # 総合経過断面: 平均ティック間隔
        overall_level = self._describe_overall_elapsed(records, min_count)
        new_snapshot[SECTION_OVERALL_ELAPSED] = overall_level.value

        self._state.snapshot = new_snapshot

        logger.debug(
            "Temporal features described: %s",
            {k: v for k, v in new_snapshot.items()},
        )

        return dict(new_snapshot)

    def _describe_activity_density(
        self, records: list[ElapsedRecord], min_count: int,
    ) -> DensityLevel:
        """活動密度断面: 直近のティック発生間隔の分布特性。"""
        if len(records) < 2:
            return DensityLevel.NORMAL

        intervals = [r.delta_time for r in records]
        return _classify_interval_density(intervals, min_count)

    def _describe_memory_interval(
        self, timestamps: list[float], min_count: int,
    ) -> DensityLevel:
        """記憶蓄積間隔断面: 記憶蓄積がどのような時間間隔で行われたか。"""
        if len(timestamps) < 2:
            return DensityLevel.NORMAL

        sorted_ts = sorted(timestamps)
        intervals = [
            sorted_ts[i] - sorted_ts[i - 1]
            for i in range(1, len(sorted_ts))
        ]
        return _classify_interval_density(intervals, min_count)

    def _describe_emotion_frequency(
        self, change_count: int, window_size: int, min_count: int,
    ) -> DensityLevel:
        """感情変動頻度断面: 感情変動の頻度。"""
        return _classify_frequency(change_count, window_size, min_count)

    def _describe_narrative_interval(
        self, timestamps: list[float], min_count: int,
    ) -> DensityLevel:
        """物語断片間隔断面: 断片間の時間的間隔。"""
        if len(timestamps) < 2:
            return DensityLevel.NORMAL

        sorted_ts = sorted(timestamps)
        intervals = [
            sorted_ts[i] - sorted_ts[i - 1]
            for i in range(1, len(sorted_ts))
        ]
        return _classify_interval_density(intervals, min_count)

    def _describe_external_input_interval(self, min_count: int) -> DensityLevel:
        """外部入力間隔断面: 外部入力の到着間隔。"""
        timestamps = self._state.external_input_timestamps
        if len(timestamps) < 2:
            return DensityLevel.NORMAL

        sorted_ts = sorted(timestamps)
        intervals = [
            sorted_ts[i] - sorted_ts[i - 1]
            for i in range(1, len(sorted_ts))
        ]
        return _classify_interval_density(intervals, min_count)

    def _describe_overall_elapsed(
        self, records: list[ElapsedRecord], min_count: int,
    ) -> DensityLevel:
        """総合経過断面: 累積経過秒と累積ティック数の比（平均ティック間隔）。"""
        if len(records) < min_count:
            return DensityLevel.NORMAL

        total_elapsed = sum(r.delta_time for r in records)
        total_ticks = len(records)

        if total_ticks <= 0:
            return DensityLevel.NORMAL

        avg_interval = total_elapsed / total_ticks
        return _classify_tempo(avg_interval, min_count, total_ticks)

    # ─── Stage 3: 参照情報としての受渡準備 ──────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        各断面の特徴量を等価に列挙する。
        特定の断面を強調・選別しない。
        列挙順序は断面の定義順に固定。
        「注目すべき密度」「異常な間隔」等の強調表現を使わない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state

        summary_text = get_temporal_summary(st)

        return {
            "elapsed_record_count": len(st.elapsed_records),
            "external_input_count": len(st.external_input_timestamps),
            "snapshot": dict(st.snapshot),
            "summary_text": summary_text,
        }

    def get_snapshot(self) -> dict[str, str]:
        """現在の断面別特徴量スナップショットをREAD-ONLYで返す。

        他モジュールがREAD-ONLYで参照可能な構造化データ。
        フィルタリング・選別・集約機能をアクセサに持たせない。
        全断面を等価に返す。

        Returns:
            断面名→DensityLevel.value の辞書（コピー）
        """
        return dict(self._state.snapshot)

    def get_previous_snapshot(self) -> dict[str, str]:
        """直前の特徴量スナップショットをREAD-ONLYで返す。

        現在のスナップショットと比較して「時間的特徴量がどの断面で変化したか」を
        記述可能にするための構造。ただし、この比較結果に基づいて処理を分岐させることはない。

        Returns:
            断面名→DensityLevel.value の辞書（コピー）
        """
        return dict(self._state.previous_snapshot)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "elapsed_record_count": len(st.elapsed_records),
            "external_input_count": len(st.external_input_timestamps),
            "has_snapshot": bool(st.snapshot),
            "has_previous_snapshot": bool(st.previous_snapshot),
            "snapshot": dict(st.snapshot),
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_temporal_summary(state: TemporalCognitionState) -> str:
    """時間認知状態の要約（enrichment用）。

    全断面を等価に列挙する。
    特定の断面を強調・選別しない。
    列挙順序は断面の定義順に固定。
    「注目すべき密度」「異常な間隔」等の強調表現を使わない。
    評価判定・行動指示を含まない。
    """
    if not state.snapshot:
        return "時間認知: 待機中"

    parts: list[str] = []
    for section_name in SECTION_ORDER:
        value = state.snapshot.get(section_name, "")
        if value:
            label = SECTION_LABELS.get(section_name, section_name)
            density_enum = DensityLevel(value)
            density_label = DENSITY_LABELS.get(density_enum, value)
            parts.append(f"{label}={density_label}")

    if not parts:
        return "時間認知: 待機中"

    return " ".join(parts)


# =============================================================================
# Factory
# =============================================================================

def create_temporal_cognition(
    config: Optional[TemporalCognitionConfig] = None,
) -> TemporalCognitionProcessor:
    """TemporalCognitionProcessor のファクトリ関数。"""
    return TemporalCognitionProcessor(config=config)
