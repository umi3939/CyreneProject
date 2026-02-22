"""
psyche/emotional_backdrop_cognition.py - 感情基調の持続認知

感情状態の時系列をより広い時間窓で観測し、窓内の構成を等価に記述する構造。
数十ターン単位で持続する低強度の感情的構成を構造的に保持・記述する。

本モジュールは感情処理パイプライン（毎ティック実行される感情反応・ダイナミクス・
振幅・独立減衰・短期記憶連動の各処理）のパラメータを一切変更しない。
感情減衰率、ムードの直接値、短期記憶-感情連動の設定値への書き込みは構造的に禁止される。

設計原則 (design_emotional_backdrop_cognition.md 準拠):
- 感情基調を「制御」「設定」「誘導」「矯正」「最適化」しない
- 特定の感情構成を「望ましい」「望ましくない」「異常」「健全」と判定しない
- 感情処理パイプラインのパラメータを変更しない
- 移動平均・加重平均・統合指標・集約スコアを算出しない
- パターンを抽出・命名・分類しない
- 「基調がある」「基調がない」というカテゴリ判定を行わない
- 変動の方向を指定・推奨しない
- 出力は数値列挙に限定し、解釈的テキストを付与しない

4段パイプライン:
1. 感情状態収集 (emotion state collection)
2. 窓内構成記述 (window composition description)
3. 蓄積 (accumulation)
4. 受渡準備 (handoff preparation)

安全弁:
1. 窓内変動性の監視（極端に低い変動性の事実記述、変動促進なし）
2. 蓄積偏り検出（異なる構成の過去記録の再浮上）
3. enrichment出力量制限（窓の全内容をenrichmentに溢れ出させない）
4. 収束監視（蓄積記録全体の多様性監視、低下時に鮮度減衰中の異なる記録の復帰を促進）
5. 自己像固定化遮断（出力を数値列挙に限定、解釈的テキストを付与しない）

経路遮断:
1. 本機能 → 感情処理パイプラインのパラメータ
2. 本機能 → ポリシー候補拡張モジュールへの直接断面供給
3. 本機能 → ムード直接値の書き込み
4. 本機能 → 記憶忘却・固定化パラメータの変更
5. 本機能 → ダイナミクス設定値の変更
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class InputSection(Enum):
    """入力参照の断面種別（8値）。"""
    EMOTION_STATE = "emotion_state"          # 感情状態断面
    MOOD = "mood"                            # ムード断面
    DYNAMICS_PHASE = "dynamics_phase"        # ダイナミクス相断面
    AMPLITUDE = "amplitude"                  # 振幅断面
    META_EMOTION = "meta_emotion"            # メタ感情認知断面
    ACCUMULATION_FRESHNESS = "accumulation_freshness"  # 蓄積鮮度断面
    DIALOGUE_ELAPSED = "dialogue_elapsed"    # 対話経過断面
    TEMPORAL_COGNITION = "temporal_cognition" # 時間認知断面


class FreshnessStage(Enum):
    """鮮度段階（memory_forgetting_fixation パターン準拠）。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


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


def _stage_from_freshness(freshness: float) -> FreshnessStage:
    """鮮度値から段階を返す。"""
    if freshness >= 0.8:
        return FreshnessStage.ACTIVE
    elif freshness >= 0.6:
        return FreshnessStage.WEAKENING
    elif freshness >= 0.4:
        return FreshnessStage.FADING
    elif freshness >= 0.2:
        return FreshnessStage.NEAR_INVISIBLE
    else:
        return FreshnessStage.INVISIBLE


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
class WindowEntry:
    """スライディングウィンドウの1エントリ。

    感情処理パイプラインの出力状態を1時点分として保持する。
    感情ベクトル（各感情次元の値の辞書）、ムード（valence、arousal）、
    ダイナミクス相、振幅係数、収集時のティック番号、収集時のタイムスタンプを含む。
    """
    entry_id: str = ""
    emotion_values: dict[str, float] = field(default_factory=dict)
    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    dynamics_phase: str = "normal"
    amplitude_value: float = 1.0
    tick: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "emotion_values": dict(self.emotion_values),
            "mood_valence": self.mood_valence,
            "mood_arousal": self.mood_arousal,
            "dynamics_phase": self.dynamics_phase,
            "amplitude_value": self.amplitude_value,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowEntry":
        return cls(
            entry_id=data.get("entry_id", ""),
            emotion_values=dict(data.get("emotion_values", {})),
            mood_valence=data.get("mood_valence", 0.0),
            mood_arousal=data.get("mood_arousal", 0.0),
            dynamics_phase=data.get("dynamics_phase", "normal"),
            amplitude_value=data.get("amplitude_value", 1.0),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class CompositionRecord:
    """窓内構成記述の認知記録。

    各処理サイクルで生成された窓内構成記述を1記録として蓄積する。
    蓄積は時系列的隣接記録であり、事実の記録にとどめる。
    鮮度が時間経過で段階的に減衰する。
    """
    record_id: str = ""
    tick: int = 0
    timestamp: float = field(default_factory=time.time)
    window_size: int = 0
    tick_range: int = 0  # 最古と最新のエントリのティック差
    time_range: float = 0.0  # 最古と最新のエントリの時間差
    # 各感情次元の直近値列挙（感情名→値のリスト）
    emotion_series: dict[str, list[float]] = field(default_factory=dict)
    # ムード列挙
    valence_series: list[float] = field(default_factory=list)
    arousal_series: list[float] = field(default_factory=list)
    # ダイナミクス相列挙
    phase_series: list[str] = field(default_factory=list)
    # 窓内変動性情報（安全弁1用）
    low_variability_noted: bool = False
    # 鮮度
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "tick_range": self.tick_range,
            "time_range": self.time_range,
            "emotion_series": {k: list(v) for k, v in self.emotion_series.items()},
            "valence_series": list(self.valence_series),
            "arousal_series": list(self.arousal_series),
            "phase_series": list(self.phase_series),
            "low_variability_noted": self.low_variability_noted,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositionRecord":
        return cls(
            record_id=data.get("record_id", ""),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            window_size=data.get("window_size", 0),
            tick_range=data.get("tick_range", 0),
            time_range=data.get("time_range", 0.0),
            emotion_series={k: list(v) for k, v in data.get("emotion_series", {}).items()},
            valence_series=list(data.get("valence_series", [])),
            arousal_series=list(data.get("arousal_series", [])),
            phase_series=list(data.get("phase_series", [])),
            low_variability_noted=data.get("low_variability_noted", False),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
        )


@dataclass
class ConvergenceRecord:
    """収束監視記録。"""
    convergence_score: float = 0.0
    convergence_level: str = ConvergenceLevel.NONE.value
    composition_diversity: float = 1.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": self.convergence_score,
            "convergence_level": self.convergence_level,
            "composition_diversity": self.composition_diversity,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceRecord":
        return cls(
            convergence_score=data.get("convergence_score", 0.0),
            convergence_level=data.get("convergence_level", ConvergenceLevel.NONE.value),
            composition_diversity=data.get("composition_diversity", 1.0),
            cycle=data.get("cycle", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class BackdropInputs:
    """8断面の入力データ。すべてREAD-ONLY参照。"""
    # 1. 感情状態断面（感情ベクトル — READ-ONLY）
    emotion_values: dict[str, float] = field(default_factory=dict)

    # 2. ムード断面（READ-ONLY）
    mood_valence: float = 0.0
    mood_arousal: float = 0.3

    # 3. ダイナミクス相断面（READ-ONLY）
    dynamics_phase: str = "normal"  # normal / peak / rebound

    # 4. 振幅断面（READ-ONLY）
    amplitude_value: float = 1.0

    # 5. メタ感情認知断面（推移パターン特徴量のみ、持続パターン検出結果・変動候補は参照しない）
    meta_emotion_change_speed: float = 0.0
    meta_emotion_dominant_stability: float = 0.0

    # 6. 蓄積鮮度断面（自己参照）
    existing_record_count: int = 0
    average_freshness: float = 0.0

    # 7. 対話経過断面
    dialogue_elapsed_ticks: int = 0

    # 8. 時間認知断面（利用可能な場合のみ）
    temporal_elapsed_description: str = ""

    # メタデータ
    current_tick: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class BackdropState:
    """内部状態。"""
    # スライディングウィンドウ: 過去の処理サイクルで収集した感情状態エントリ
    sliding_window: list[WindowEntry] = field(default_factory=list)

    # 窓内構成記述の蓄積
    composition_records: list[CompositionRecord] = field(default_factory=list)

    # 減衰履歴
    decay_history: list[dict[str, Any]] = field(default_factory=list)

    # 収束監視状態
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)

    # カウンタ
    cycle_count: int = 0
    total_entries_collected: int = 0
    total_records_created: int = 0
    total_records_decayed: int = 0
    total_records_recovered: int = 0

    # 安全弁フラグ
    low_variability_warning: bool = False
    accumulation_bias_warning: bool = False
    convergence_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sliding_window": [e.to_dict() for e in self.sliding_window],
            "composition_records": [r.to_dict() for r in self.composition_records],
            "decay_history": list(self.decay_history),
            "convergence_records": [c.to_dict() for c in self.convergence_records],
            "cycle_count": self.cycle_count,
            "total_entries_collected": self.total_entries_collected,
            "total_records_created": self.total_records_created,
            "total_records_decayed": self.total_records_decayed,
            "total_records_recovered": self.total_records_recovered,
            "low_variability_warning": self.low_variability_warning,
            "accumulation_bias_warning": self.accumulation_bias_warning,
            "convergence_warning": self.convergence_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackdropState":
        return cls(
            sliding_window=[
                WindowEntry.from_dict(e) for e in data.get("sliding_window", [])
            ],
            composition_records=[
                CompositionRecord.from_dict(r) for r in data.get("composition_records", [])
            ],
            decay_history=list(data.get("decay_history", [])),
            convergence_records=[
                ConvergenceRecord.from_dict(c) for c in data.get("convergence_records", [])
            ],
            cycle_count=data.get("cycle_count", 0),
            total_entries_collected=data.get("total_entries_collected", 0),
            total_records_created=data.get("total_records_created", 0),
            total_records_decayed=data.get("total_records_decayed", 0),
            total_records_recovered=data.get("total_records_recovered", 0),
            low_variability_warning=data.get("low_variability_warning", False),
            accumulation_bias_warning=data.get("accumulation_bias_warning", False),
            convergence_warning=data.get("convergence_warning", False),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。"""
        remove_ids: set[str] = set()

        for rec in self.composition_records:
            rec.freshness = _clamp(rec.freshness - decay_factor)
            rec.freshness_stage = _stage_from_freshness(rec.freshness).value
            if rec.freshness < 0.1:
                remove_ids.add(rec.record_id)

        if remove_ids:
            self.composition_records = [
                r for r in self.composition_records if r.record_id not in remove_ids
            ]


# =============================================================================
# Result
# =============================================================================

@dataclass
class BackdropResult:
    """処理結果（参照情報形式のみ）。"""
    window_size: int = 0
    record_count: int = 0
    tick_range: int = 0
    time_range: float = 0.0
    low_variability_warning: bool = False
    accumulation_bias_warning: bool = False
    convergence_warning: bool = False
    convergence_level: str = ConvergenceLevel.NONE.value
    convergence_score: float = 0.0
    diversity_restored: bool = False
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BackdropConfig:
    """設定。"""
    # スライディングウィンドウの上限件数
    max_window_size: int = 30

    # 蓄積記録の上限件数
    max_composition_records: int = 50

    # 減衰履歴の上限件数
    max_decay_history: int = 30

    # 収束記録の上限件数
    max_convergence_records: int = 20

    # 鮮度減衰速度（処理サイクル毎）
    freshness_decay_rate: float = 0.02

    # 同種認知による鮮度回復量
    reference_recovery: float = 0.1

    # 収束警告閾値
    convergence_threshold: float = 0.5

    # 多様性復元時の鮮度回復量
    diversity_recovery_amount: float = 0.1

    # 変動性が極端に低いとみなす閾値
    low_variability_threshold: float = 0.005

    # enrichmentに含める直近構成記述の件数上限（安全弁3）
    max_enrichment_records: int = 5

    # enrichment出力のサイズ上限（文字数、安全弁3）
    max_enrichment_length: int = 1500


# =============================================================================
# Processor (4-stage pipeline)
# =============================================================================

class EmotionalBackdropProcessor:
    """感情基調の持続認知プロセッサ。

    4段パイプライン:
    1. 感情状態収集: 感情処理パイプラインの出力をREAD-ONLYで収集しウィンドウに追加
    2. 窓内構成記述: ウィンドウ内の感情状態を等価に並置した断面記述として構成
    3. 蓄積: 構成記述を認知記録として時系列に蓄積し、鮮度減衰を適用
    4. 受渡準備: enrichment参照情報を整え、安全弁チェックを実行

    感情処理パイプラインのパラメータを一切変更しない。
    出力は参照情報形式のみ。
    """

    def __init__(self, config: Optional[BackdropConfig] = None):
        self._config = config or BackdropConfig()
        self._state = BackdropState()

    @property
    def state(self) -> BackdropState:
        return self._state

    @state.setter
    def state(self, value: BackdropState) -> None:
        self._state = value

    def tick(self, inputs: BackdropInputs) -> BackdropResult:
        """orchestrator から呼ばれる単一エントリポイント。"""
        return self.process(inputs)

    def process(self, inputs: BackdropInputs) -> BackdropResult:
        """4段パイプラインを実行する。"""
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config

        # Stage 1: 感情状態収集
        self._collect_emotion_state(inputs, now)

        # Stage 2: 窓内構成記述
        composition = self._describe_window_composition(inputs, now)

        # Stage 3: 蓄積
        self._accumulate(composition, now)

        # Stage 4: 受渡準備
        return self._prepare_handoff(now)

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        窓内構成記述の概要（窓のサイズ、時間的範囲、各感情次元の直近の値列挙）を提供する。
        出力は等価列挙形式であり、評価判定を含まない。
        enrichmentに含める情報量には上限を設ける（安全弁3）。
        出力を数値列挙に限定し、解釈的テキストを付与しない（安全弁5）。
        """
        st = self._state
        cfg = self._config

        if not st.sliding_window:
            return {
                "summary_text": "感情基調認知: 待機中",
                "window_size": 0,
                "record_count": 0,
                "entries": [],
            }

        # 窓の概要情報
        window_size = len(st.sliding_window)
        tick_range = 0
        time_range = 0.0
        if window_size >= 2:
            tick_range = st.sliding_window[-1].tick - st.sliding_window[0].tick
            time_range = st.sliding_window[-1].timestamp - st.sliding_window[0].timestamp

        # 直近の蓄積記録（鮮度がinvisible以上のもの）
        visible_records = [
            r for r in st.composition_records
            if r.freshness >= 0.2
        ]
        recent = visible_records[-cfg.max_enrichment_records:]

        entries: list[dict[str, Any]] = []
        for rec in recent:
            # 各感情次元の直近値（上限5次元まで）
            emotion_summary: dict[str, list[float]] = {}
            count = 0
            for dim, values in rec.emotion_series.items():
                if count >= 5:
                    break
                # 直近5値まで
                emotion_summary[dim] = values[-5:]
                count += 1

            entries.append({
                "tick": rec.tick,
                "window_size": rec.window_size,
                "tick_range": rec.tick_range,
                "valence": rec.valence_series[-3:] if rec.valence_series else [],
                "arousal": rec.arousal_series[-3:] if rec.arousal_series else [],
                "emotions": emotion_summary,
                "low_variability": rec.low_variability_noted,
                "freshness_stage": rec.freshness_stage,
            })

        summary_text = get_backdrop_summary(st)

        return {
            "summary_text": summary_text,
            "window_size": window_size,
            "tick_range": tick_range,
            "time_range": time_range,
            "record_count": len(st.composition_records),
            "entries": entries,
            "low_variability_warning": st.low_variability_warning,
            "accumulation_bias_warning": st.accumulation_bias_warning,
            "convergence_warning": st.convergence_warning,
        }

    # ─── Stage 1: 感情状態収集 ────────────────────────────────────

    def _collect_emotion_state(self, inputs: BackdropInputs, now: float) -> None:
        """感情処理パイプラインの出力状態をREAD-ONLYで収集し、
        スライディングウィンドウに1件として追加する。

        窓のサイズは構成可能な上限を持ち、上限を超えた場合は最古の1件がFIFOで押し出される。
        押し出しが唯一のデータ消失経路であり、選択的消去は行わない。
        """
        cfg = self._config

        entry = WindowEntry(
            emotion_values=dict(inputs.emotion_values) if inputs.emotion_values else {},
            mood_valence=inputs.mood_valence,
            mood_arousal=inputs.mood_arousal,
            dynamics_phase=inputs.dynamics_phase,
            amplitude_value=inputs.amplitude_value,
            tick=inputs.current_tick,
            timestamp=now,
        )

        self._state.sliding_window.append(entry)
        self._state.total_entries_collected += 1

        # FIFO押し出し（唯一のデータ消失経路）
        if len(self._state.sliding_window) > cfg.max_window_size:
            overflow = len(self._state.sliding_window) - cfg.max_window_size
            self._state.sliding_window = self._state.sliding_window[overflow:]

    # ─── Stage 2: 窓内構成記述 ───────────────────────────────────

    def _describe_window_composition(
        self, inputs: BackdropInputs, now: float,
    ) -> CompositionRecord:
        """スライディングウィンドウに蓄積された過去の感情状態群を、
        等価に並置した断面記述として構成する。

        移動平均・統合指標を算出しない。値を時系列順にそのまま列挙する。
        すべて等価に並置する。特定の次元や特定の時間帯に重みを付与しない。
        """
        window = self._state.sliding_window
        window_size = len(window)

        # 時間的範囲
        tick_range = 0
        time_range = 0.0
        if window_size >= 2:
            tick_range = window[-1].tick - window[0].tick
            time_range = window[-1].timestamp - window[0].timestamp

        # 各感情次元の値を時系列順に列挙
        emotion_series: dict[str, list[float]] = {}
        for entry in window:
            for dim, val in entry.emotion_values.items():
                if dim not in emotion_series:
                    emotion_series[dim] = []
                emotion_series[dim].append(val)

        # ムードの値を時系列順に列挙
        valence_series = [e.mood_valence for e in window]
        arousal_series = [e.mood_arousal for e in window]

        # ダイナミクス相の推移を時系列順に列挙
        phase_series = [e.dynamics_phase for e in window]

        # 安全弁1: 窓内変動性の監視
        low_variability = self._check_low_variability(emotion_series, valence_series)

        record = CompositionRecord(
            tick=inputs.current_tick,
            timestamp=now,
            window_size=window_size,
            tick_range=tick_range,
            time_range=time_range,
            emotion_series=emotion_series,
            valence_series=valence_series,
            arousal_series=arousal_series,
            phase_series=phase_series,
            low_variability_noted=low_variability,
        )

        return record

    def _check_low_variability(
        self,
        emotion_series: dict[str, list[float]],
        valence_series: list[float],
    ) -> bool:
        """安全弁1: 窓内の感情値の変動性が極端に低いかを検出する。

        変動性が低いこと自体を「問題」として扱わず、変動を促す処理を行わない。
        事実の記述にとどめる。
        """
        cfg = self._config

        if len(valence_series) < 3:
            return False

        # valence の変動性チェック
        val_mean = sum(valence_series) / len(valence_series)
        val_var = sum((v - val_mean) ** 2 for v in valence_series) / len(valence_series)

        # 各感情次元の変動性チェック
        all_low = True
        dim_count = 0
        for dim, values in emotion_series.items():
            if len(values) < 3:
                continue
            dim_count += 1
            dim_mean = sum(values) / len(values)
            dim_var = sum((v - dim_mean) ** 2 for v in values) / len(values)
            if dim_var >= cfg.low_variability_threshold:
                all_low = False
                break

        if dim_count == 0:
            return val_var < cfg.low_variability_threshold

        return all_low and val_var < cfg.low_variability_threshold

    # ─── Stage 3: 蓄積 ─────────────────────────────────────────

    def _accumulate(self, composition: CompositionRecord, now: float) -> None:
        """窓内構成記述を認知記録として時系列に蓄積し、鮮度減衰を適用する。

        蓄積は時系列的隣接記録であり、事実の記録にとどめる。
        新規の忘却メカニズムを作らない（既存パターンと同一の段階的鮮度減衰）。
        """
        cfg = self._config

        # 蓄積記録に追加
        self._state.composition_records.append(composition)
        self._state.total_records_created += 1

        # 鮮度減衰（既存の記憶忘却構造と同一パターン）
        for rec in self._state.composition_records:
            old_freshness = rec.freshness
            rec.freshness = _clamp(rec.freshness - cfg.freshness_decay_rate)
            new_stage = _stage_from_freshness(rec.freshness)
            old_stage = rec.freshness_stage

            if new_stage.value != old_stage:
                rec.freshness_stage = new_stage.value
                self._state.decay_history.append({
                    "record_id": rec.record_id,
                    "old_stage": old_stage,
                    "new_stage": new_stage.value,
                    "freshness": rec.freshness,
                    "timestamp": now,
                })

                if new_stage == FreshnessStage.INVISIBLE:
                    self._state.total_records_decayed += 1

        # 蓄積記録の上限制御
        if len(self._state.composition_records) > cfg.max_composition_records:
            self._state.composition_records = (
                self._state.composition_records[-cfg.max_composition_records:]
            )

        # 減衰履歴の上限制御
        if len(self._state.decay_history) > cfg.max_decay_history:
            self._state.decay_history = (
                self._state.decay_history[-cfg.max_decay_history:]
            )

    # ─── Stage 4: 受渡準備 ──────────────────────────────────────

    def _prepare_handoff(self, now: float) -> BackdropResult:
        """安全弁チェックを行い結果を返す。

        出力は参照情報形式のみ。判断・評価・行動決定・感情パラメータ変更を
        直接起動しない。
        """
        cfg = self._config
        st = self._state

        # 窓の概要
        window_size = len(st.sliding_window)
        tick_range = 0
        time_range = 0.0
        if window_size >= 2:
            tick_range = st.sliding_window[-1].tick - st.sliding_window[0].tick
            time_range = st.sliding_window[-1].timestamp - st.sliding_window[0].timestamp

        visible_records = [
            r for r in st.composition_records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
        ]
        record_count = len(visible_records)

        # 安全弁1: 窓内変動性の監視（最新レコードから）
        if st.composition_records:
            st.low_variability_warning = st.composition_records[-1].low_variability_noted
        else:
            st.low_variability_warning = False

        # 安全弁2: 蓄積偏り検出
        diversity_restored = False
        st.accumulation_bias_warning = self._check_accumulation_bias(now)
        if st.accumulation_bias_warning:
            diversity_restored = self._restore_accumulation_diversity(now)

        # 安全弁4: 収束監視
        convergence = self._monitor_convergence(now)
        if convergence.convergence_level in (
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        ):
            st.convergence_warning = True
            if not diversity_restored:
                diversity_restored = self._restore_accumulation_diversity(now)
        else:
            st.convergence_warning = False

        return BackdropResult(
            window_size=window_size,
            record_count=record_count,
            tick_range=tick_range,
            time_range=time_range,
            low_variability_warning=st.low_variability_warning,
            accumulation_bias_warning=st.accumulation_bias_warning,
            convergence_warning=st.convergence_warning,
            convergence_level=convergence.convergence_level,
            convergence_score=convergence.convergence_score,
            diversity_restored=diversity_restored,
            cycle_count=st.cycle_count,
        )

    # ─── 安全弁2: 蓄積偏り検出 ──────────────────────────────────

    def _check_accumulation_bias(self, now: float) -> bool:
        """蓄積記録が特定の構成記述に偏った場合を検出する。

        構成の多様性を、各記録の感情次元の値分散で近似的に判定する。
        """
        active_records = [
            r for r in self._state.composition_records
            if _stage_from_freshness(r.freshness) in (
                FreshnessStage.ACTIVE, FreshnessStage.WEAKENING,
            )
        ]
        if len(active_records) < 3:
            return False

        # 各記録の valence_series 末尾値（直近の valence）を比較
        last_valences = []
        for rec in active_records[-5:]:
            if rec.valence_series:
                last_valences.append(rec.valence_series[-1])

        if len(last_valences) < 3:
            return False

        mean_val = sum(last_valences) / len(last_valences)
        variance = sum((v - mean_val) ** 2 for v in last_valences) / len(last_valences)

        # 変動性が極端に低い場合は偏り
        return variance < 0.001

    def _restore_accumulation_diversity(self, now: float) -> bool:
        """安全弁2/4: 蓄積の多様性を回復するため、
        鮮度減衰中の異なる構成の記録の鮮度を部分的に回復する。

        復帰経路は閉じない。
        """
        cfg = self._config
        restored = False

        for rec in self._state.composition_records:
            stage = _stage_from_freshness(rec.freshness)
            if stage in (FreshnessStage.FADING, FreshnessStage.NEAR_INVISIBLE):
                rec.freshness = _clamp(rec.freshness + cfg.diversity_recovery_amount)
                rec.freshness_stage = _stage_from_freshness(rec.freshness).value
                self._state.total_records_recovered += 1
                restored = True

        return restored

    # ─── 安全弁4: 収束監視 ──────────────────────────────────────

    def _monitor_convergence(self, now: float) -> ConvergenceRecord:
        """蓄積記録全体の構成の多様性を監視する。"""
        cfg = self._config
        st = self._state

        # 直近の構成記述から多様性を計算
        recent = [
            r for r in st.composition_records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
        ][-5:]

        composition_diversity = 1.0
        if len(recent) >= 2:
            # valence末尾値の分散
            last_vals = [r.valence_series[-1] for r in recent if r.valence_series]
            if len(last_vals) >= 2:
                mean_v = sum(last_vals) / len(last_vals)
                var_v = sum((v - mean_v) ** 2 for v in last_vals) / len(last_vals)
                composition_diversity = _clamp(var_v * 20.0)

        convergence_score = _clamp(1.0 - composition_diversity)
        convergence_level = _convergence_from_score(convergence_score)

        record = ConvergenceRecord(
            convergence_score=convergence_score,
            convergence_level=convergence_level.value,
            composition_diversity=composition_diversity,
            cycle=st.cycle_count,
            timestamp=now,
        )

        st.convergence_records.append(record)
        if len(st.convergence_records) > cfg.max_convergence_records:
            st.convergence_records = st.convergence_records[-cfg.max_convergence_records:]

        return record

    # ─── READ-ONLYアクセサ ──────────────────────────────────────

    def get_window_entries(self) -> list[dict[str, Any]]:
        """スライディングウィンドウの内容をREAD-ONLYで返す。

        フィルタリング・選別・集約機能をアクセサに持たせない。
        全エントリを等価に返す。
        """
        return [e.to_dict() for e in self._state.sliding_window]

    def get_composition_records(self) -> list[dict[str, Any]]:
        """蓄積記録をREAD-ONLYで返す。

        全記録を等価に返す。
        """
        return [r.to_dict() for r in self._state.composition_records]

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "window_size": len(self._state.sliding_window),
            "record_count": len(self._state.composition_records),
            "cycle_count": self._state.cycle_count,
            "low_variability_warning": self._state.low_variability_warning,
            "accumulation_bias_warning": self._state.accumulation_bias_warning,
            "convergence_warning": self._state.convergence_warning,
        }

    # ─── Save / Load ──────────────────────────────────────────────

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = BackdropState.from_dict(data)
        logger.debug(
            "Emotional backdrop cognition state loaded: window=%d, records=%d",
            len(self._state.sliding_window),
            len(self._state.composition_records),
        )


# =============================================================================
# Summary
# =============================================================================

def get_backdrop_summary(state: BackdropState) -> str:
    """enrichment用の要約テキスト。

    評価判定を含まない。数値列挙に限定する（安全弁5）。
    """
    if state.cycle_count == 0 and not state.sliding_window:
        return "感情基調認知: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    window_size = len(state.sliding_window)
    parts.append(f"窓={window_size}")

    if window_size >= 2:
        tick_range = state.sliding_window[-1].tick - state.sliding_window[0].tick
        parts.append(f"範囲={tick_range}t")

    visible_count = sum(
        1 for r in state.composition_records
        if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
    )
    if visible_count:
        parts.append(f"記録={visible_count}")

    if state.low_variability_warning:
        parts.append("低変動")
    if state.accumulation_bias_warning:
        parts.append("蓄積偏り")
    if state.convergence_warning:
        parts.append("収束")

    return " ".join(parts) if parts else "感情基調認知: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_emotional_backdrop_processor(
    config: Optional[BackdropConfig] = None,
) -> EmotionalBackdropProcessor:
    """EmotionalBackdropProcessor のファクトリ関数。"""
    return EmotionalBackdropProcessor(config=config)
