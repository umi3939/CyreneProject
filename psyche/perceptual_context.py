"""
psyche/perceptual_context.py - 知覚入力の内部文脈化（知覚推移の断面記述）

知覚解析が生成する構造化された知覚情報（感情ラベル・意図ラベル・話題リスト・感情価）
の直近系列を保持し、その推移を5つの断面で段階的に記述する構造を提供する。

設計原則 (design_perceptual_context.md / design_perceptual_structuring_degree.md 準拠):
- 知覚情報の自由テキスト（原文・意味解釈文）の系列比較を行わない
- 話題要素間の意味的類似度判定を導入しない（文字列完全一致のみ）
- 知覚推移の「型」「フェーズ」「パターン」を事前定義しない
- 記憶の多経路想起の文脈連想経路の入力構造を変更しない
- 外部文脈感度の連続性値を本機能の出力から自動計算しない
- 知覚推移に基づく判断・評価を行わない
- 感情パイプラインのパラメータを変更しない
- 段階値のパターン抽出・傾向化・統計処理を行わない
- すべての断面の特徴量は等価であり、特定の断面に重みや重要度を付与しない
- enrichment内での強調禁止
- 構造化度の高低を「理解の深さ」として解釈しない
- 構造化度に基づいて知覚処理の動作やパラメータを変更しない
- 構造化度が低い入力を「問題」「異常」「未知」として分類しない

3段パイプライン:
1. 知覚サマリの蓄積 (perceptual summary accumulation)
2. 5断面での知覚推移特徴量の記述 (5-section perceptual transition feature description)
3. 参照情報としての受渡準備 (handoff preparation as reference information)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Enums - 段階値（列挙型）
# =============================================================================

class ChangeFrequency(Enum):
    """変化頻度の段階値。

    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    """
    FREQUENT = "frequent"
    SOMEWHAT_FREQUENT = "somewhat_frequent"
    MODERATE = "moderate"
    SOMEWHAT_RARE = "somewhat_rare"
    RARE = "rare"


class OverlapDegree(Enum):
    """話題重複度の段階値。

    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    """
    HIGH = "high"
    SOMEWHAT_HIGH = "somewhat_high"
    MODERATE = "moderate"
    SOMEWHAT_LOW = "somewhat_low"
    LOW = "low"


class TransitionDirection(Enum):
    """推移方向の段階値。

    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    """
    RISING = "rising"
    SOMEWHAT_RISING = "somewhat_rising"
    FLAT = "flat"
    SOMEWHAT_FALLING = "somewhat_falling"
    FALLING = "falling"


class StructuringDegree(Enum):
    """知覚構造化度の段階値。

    辞書照合による構造化がどの程度発生したかの事実記述。
    数値への還元は行わず、段階的記述として表現する。
    各段階に重み・スコア・優先度は付与しない（全段階等価）。
    「多」が「少」より望ましいという含意を持たない。
    """
    MANY = "many"
    SOMEWHAT_MANY = "somewhat_many"
    MODERATE = "moderate"
    SOMEWHAT_FEW = "somewhat_few"
    FEW = "few"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PerceptualSummary:
    """知覚サマリ。感情ラベル・意図ラベル・話題リスト・感情価・ティック番号の組。

    一度記録されたら変更されない（追記のみ）。
    重み・スコア・優先度などの評価的属性を持たない（全記録等価）。
    自由テキスト（原文・意味解釈文）は保持しない。
    """
    emotion: str = "neutral"
    intent: str = "unknown"
    topics: list[str] = field(default_factory=list)
    emotion_valence: float = 0.0
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "intent": self.intent,
            "topics": list(self.topics),
            "emotion_valence": self.emotion_valence,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerceptualSummary":
        return cls(
            emotion=data.get("emotion", "neutral"),
            intent=data.get("intent", "unknown"),
            topics=list(data.get("topics", [])),
            emotion_valence=float(data.get("emotion_valence", 0.0)),
            tick=data.get("tick", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class PerceptualContextState:
    """知覚文脈化の内部状態。"""

    # 知覚サマリのスライディングウィンドウ: 時系列順に蓄積
    summaries: list[PerceptualSummary] = field(default_factory=list)

    # 断面別の推移特徴量スナップショット: 5断面それぞれの最新の特徴量
    snapshot: dict[str, str] = field(default_factory=dict)

    # 直前の特徴量スナップショット: 1回前の処理実行時のスナップショット
    previous_snapshot: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summaries": [s.to_dict() for s in self.summaries],
            "snapshot": dict(self.snapshot),
            "previous_snapshot": dict(self.previous_snapshot),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerceptualContextState":
        summaries = [
            PerceptualSummary.from_dict(s)
            for s in data.get("summaries", [])
        ]
        return cls(
            summaries=summaries,
            snapshot=dict(data.get("snapshot", {})),
            previous_snapshot=dict(data.get("previous_snapshot", {})),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PerceptualContextConfig:
    """設定。"""
    # 知覚サマリのスライディングウィンドウ上限
    max_summaries: int = field(default_factory=lambda: coefficient_registry.get("description_common", "fifo_limit_50"))

    # 特徴量記述で使用するウィンドウ内レコードの最小件数
    # この件数未満の場合は各断面のデフォルト値とする
    min_records_for_description: int = 3


# =============================================================================
# Section Names (断面名の定義)
# =============================================================================

# 断面名は定義順に固定。列挙順序のランダム化・最適化は行わない。
SECTION_EMOTION_CHANGE_FREQ = "emotion_change_frequency"
SECTION_INTENT_CHANGE_FREQ = "intent_change_frequency"
SECTION_TOPIC_OVERLAP = "topic_overlap"
SECTION_VALENCE_DIRECTION = "valence_direction"
SECTION_STRUCTURING_DEGREE = "structuring_degree"

# 定義順序（固定、変更禁止）
SECTION_ORDER = [
    SECTION_EMOTION_CHANGE_FREQ,
    SECTION_INTENT_CHANGE_FREQ,
    SECTION_TOPIC_OVERLAP,
    SECTION_VALENCE_DIRECTION,
    SECTION_STRUCTURING_DEGREE,
]

# 断面の日本語ラベル（enrichment用、等価に列挙するためのラベル）
SECTION_LABELS = {
    SECTION_EMOTION_CHANGE_FREQ: "感情ラベル変化頻度",
    SECTION_INTENT_CHANGE_FREQ: "意図ラベル変化頻度",
    SECTION_TOPIC_OVERLAP: "話題重複度",
    SECTION_VALENCE_DIRECTION: "感情価推移方向",
    SECTION_STRUCTURING_DEGREE: "知覚構造化度",
}

# 段階値の日本語ラベル
CHANGE_FREQ_LABELS = {
    ChangeFrequency.FREQUENT: "頻繁",
    ChangeFrequency.SOMEWHAT_FREQUENT: "やや頻繁",
    ChangeFrequency.MODERATE: "普通",
    ChangeFrequency.SOMEWHAT_RARE: "やや少ない",
    ChangeFrequency.RARE: "少ない",
}

OVERLAP_LABELS = {
    OverlapDegree.HIGH: "高",
    OverlapDegree.SOMEWHAT_HIGH: "やや高",
    OverlapDegree.MODERATE: "普通",
    OverlapDegree.SOMEWHAT_LOW: "やや低",
    OverlapDegree.LOW: "低",
}

DIRECTION_LABELS = {
    TransitionDirection.RISING: "上昇",
    TransitionDirection.SOMEWHAT_RISING: "やや上昇",
    TransitionDirection.FLAT: "横ばい",
    TransitionDirection.SOMEWHAT_FALLING: "やや下降",
    TransitionDirection.FALLING: "下降",
}

STRUCTURING_DEGREE_LABELS = {
    StructuringDegree.MANY: "多",
    StructuringDegree.SOMEWHAT_MANY: "やや多",
    StructuringDegree.MODERATE: "中程度",
    StructuringDegree.SOMEWHAT_FEW: "やや少",
    StructuringDegree.FEW: "少",
}


# =============================================================================
# Feature Description Helpers
# =============================================================================

def _classify_change_frequency(
    change_count: int,
    window_size: int,
    min_count: int = 3,
) -> ChangeFrequency:
    """隣接間の変化回数から変化頻度の段階値を記述する。

    判断・評価は含まない。段階的記述のみ。
    """
    if window_size < min_count:
        return ChangeFrequency.MODERATE

    # 比較可能な隣接ペア数
    pair_count = window_size - 1
    if pair_count <= 0:
        return ChangeFrequency.MODERATE

    ratio = change_count / pair_count

    if ratio >= 0.8:
        return ChangeFrequency.FREQUENT
    elif ratio >= 0.5:
        return ChangeFrequency.SOMEWHAT_FREQUENT
    elif ratio >= 0.2:
        return ChangeFrequency.MODERATE
    elif ratio >= 0.1:
        return ChangeFrequency.SOMEWHAT_RARE
    else:
        return ChangeFrequency.RARE


def _classify_overlap_degree(
    overlap_count: int,
    max_possible: int,
) -> OverlapDegree:
    """話題の重複要素数から重複度の段階値を記述する。

    文字列完全一致のみ。意味的類似度は使わない。
    判断・評価は含まない。段階的記述のみ。
    """
    if max_possible <= 0:
        return OverlapDegree.MODERATE

    ratio = overlap_count / max_possible

    if ratio >= 0.8:
        return OverlapDegree.HIGH
    elif ratio >= 0.5:
        return OverlapDegree.SOMEWHAT_HIGH
    elif ratio >= 0.2:
        return OverlapDegree.MODERATE
    elif ratio >= 0.1:
        return OverlapDegree.SOMEWHAT_LOW
    else:
        return OverlapDegree.LOW


def _classify_valence_direction(
    valence_series: list[float],
    min_count: int = 3,
) -> TransitionDirection:
    """感情価の系列から推移方向の段階値を記述する。

    最新と最古の単純差分から段階値に変換する。
    統計的回帰分析は行わない。
    判断・評価は含まない。段階的記述のみ。
    """
    if len(valence_series) < min_count:
        return TransitionDirection.FLAT

    oldest = valence_series[0]
    newest = valence_series[-1]
    diff = newest - oldest

    if diff >= 0.5:
        return TransitionDirection.RISING
    elif diff >= 0.15:
        return TransitionDirection.SOMEWHAT_RISING
    elif diff >= -0.15:
        return TransitionDirection.FLAT
    elif diff >= -0.5:
        return TransitionDirection.SOMEWHAT_FALLING
    else:
        return TransitionDirection.FALLING


def _classify_structuring_degree(
    average_ratio: float,
    window_size: int,
    min_count: int = 3,
) -> StructuringDegree:
    """構造化度の平均比率から段階値を記述する。

    3つの比率（感情ラベル非デフォルト率、意図ラベル非デフォルト率、
    話題要素非ゼロ率）の単純平均から段階値に変換する。
    判断・評価は含まない。段階的記述のみ。
    全段階等価。「多」が「少」より望ましいという含意を持たない。
    """
    if window_size < min_count:
        return StructuringDegree.MODERATE

    if average_ratio >= 0.8:
        return StructuringDegree.MANY
    elif average_ratio >= 0.5:
        return StructuringDegree.SOMEWHAT_MANY
    elif average_ratio >= 0.2:
        return StructuringDegree.MODERATE
    elif average_ratio >= 0.1:
        return StructuringDegree.SOMEWHAT_FEW
    else:
        return StructuringDegree.FEW


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class PerceptualContextProcessor:
    """知覚文脈化プロセッサ。

    3段パイプライン:
    1. 知覚サマリの蓄積 -- 感情ラベル・意図ラベル・話題リスト・感情価・ティック番号を蓄積
    2. 5断面での知覚推移特徴量の記述 -- 5断面の段階値を記述
    3. 参照情報としての受渡準備 -- enrichment + READ-ONLYアクセサ

    すべての処理は記述的な特徴量の算出・整理であり、能動的な判断・評価・制御を含まない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[PerceptualContextConfig] = None):
        self._config = config or PerceptualContextConfig()
        self._state = PerceptualContextState()

    @property
    def state(self) -> PerceptualContextState:
        return self._state

    @state.setter
    def state(self, value: PerceptualContextState) -> None:
        self._state = value

    # --- Stage 1: 知覚サマリの蓄積 -------------------------------------------

    def accumulate_summary(
        self,
        emotion: str,
        intent: str,
        topics: list[str],
        emotion_valence: float,
        tick: int,
    ) -> None:
        """毎ティック呼び出し: 知覚サマリを蓄積する。

        スライディングウィンドウ方式で保持し、上限到達時は最古から押し出す。
        自由テキスト（原文・意味解釈文）は保持しない。

        Args:
            emotion: 感情ラベル（ラベル型）
            intent: 意図ラベル（ラベル型）
            topics: 話題リスト（文字列リスト型）
            emotion_valence: 感情価（数値型）
            tick: 現在のティック番号
        """
        summary = PerceptualSummary(
            emotion=emotion,
            intent=intent,
            topics=list(topics) if topics is not None else [],
            emotion_valence=emotion_valence,
            tick=tick,
        )

        self._state.summaries.append(summary)

        # 上限による押し出し（唯一の消失経路）
        self._apply_summary_pushout()

        logger.debug(
            "Perceptual summary accumulated: tick=%d, emotion=%s, intent=%s, "
            "topics=%s, valence=%.3f, summaries=%d",
            tick, emotion, intent, topics, emotion_valence,
            len(self._state.summaries),
        )

    def _apply_summary_pushout(self) -> None:
        """知覚サマリの上限押し出し。"""
        cfg = self._config
        if len(self._state.summaries) > cfg.max_summaries:
            pushout_count = len(self._state.summaries) - cfg.max_summaries
            self._state.summaries = self._state.summaries[pushout_count:]

    # --- Stage 2: 5断面での知覚推移特徴量の記述 --------------------------------

    def describe_features(self) -> dict[str, str]:
        """5断面の知覚推移特徴量を記述する。

        各断面は独立であり、断面間の優先順位・重み付け・統合処理は存在しない。
        すべての断面は等価である。

        知覚入力がないティックでも呼び出し可能。その場合はウィンドウ内の
        既存データに基づいてスナップショットの再記述のみ行う。

        Returns:
            5断面の特徴量を保持する辞書（断面名 -> 段階値の文字列）
        """
        cfg = self._config
        summaries = self._state.summaries
        min_count = cfg.min_records_for_description

        # 直前スナップショットを保持（現在のスナップショットを直前に移動）
        if self._state.snapshot:
            self._state.previous_snapshot = dict(self._state.snapshot)

        new_snapshot: dict[str, str] = {}

        # 感情ラベル変化頻度断面
        emotion_freq = self._describe_emotion_change_frequency(summaries, min_count)
        new_snapshot[SECTION_EMOTION_CHANGE_FREQ] = emotion_freq.value

        # 意図ラベル変化頻度断面
        intent_freq = self._describe_intent_change_frequency(summaries, min_count)
        new_snapshot[SECTION_INTENT_CHANGE_FREQ] = intent_freq.value

        # 話題重複度断面
        topic_overlap = self._describe_topic_overlap(summaries)
        new_snapshot[SECTION_TOPIC_OVERLAP] = topic_overlap.value

        # 感情価推移方向断面
        valence_dir = self._describe_valence_direction(summaries, min_count)
        new_snapshot[SECTION_VALENCE_DIRECTION] = valence_dir.value

        # 知覚構造化度断面
        structuring = self._describe_structuring_degree(summaries, min_count)
        new_snapshot[SECTION_STRUCTURING_DEGREE] = structuring.value

        self._state.snapshot = new_snapshot

        logger.debug(
            "Perceptual features described: %s",
            {k: v for k, v in new_snapshot.items()},
        )

        return dict(new_snapshot)

    def _describe_emotion_change_frequency(
        self,
        summaries: list[PerceptualSummary],
        min_count: int,
    ) -> ChangeFrequency:
        """感情ラベル変化頻度断面:
        隣接する知覚サマリ間の感情ラベルが何回変化したかを数え、
        その変化頻度を段階値で記述する。
        比較は感情ラベルの文字列完全一致のみ。
        """
        if len(summaries) < 2:
            return ChangeFrequency.MODERATE

        change_count = 0
        for i in range(1, len(summaries)):
            if summaries[i].emotion != summaries[i - 1].emotion:
                change_count += 1

        return _classify_change_frequency(change_count, len(summaries), min_count)

    def _describe_intent_change_frequency(
        self,
        summaries: list[PerceptualSummary],
        min_count: int,
    ) -> ChangeFrequency:
        """意図ラベル変化頻度断面:
        隣接する知覚サマリ間の意図ラベルが何回変化したかを数え、
        その変化頻度を段階値で記述する。
        比較は意図ラベルの文字列完全一致のみ。
        """
        if len(summaries) < 2:
            return ChangeFrequency.MODERATE

        change_count = 0
        for i in range(1, len(summaries)):
            if summaries[i].intent != summaries[i - 1].intent:
                change_count += 1

        return _classify_change_frequency(change_count, len(summaries), min_count)

    def _describe_topic_overlap(
        self,
        summaries: list[PerceptualSummary],
    ) -> OverlapDegree:
        """話題重複度断面:
        最新の知覚サマリの話題リストと、直前の知覚サマリの話題リストとの
        文字列完全一致要素数を数え、その重複度を段階値で記述する。
        意味的類似度判定は一切行わない。
        """
        if len(summaries) < 2:
            return OverlapDegree.MODERATE

        latest_topics = summaries[-1].topics
        previous_topics = summaries[-2].topics

        # 文字列完全一致のみ
        latest_set = set(latest_topics)
        previous_set = set(previous_topics)
        overlap_count = len(latest_set & previous_set)

        # 分母は両リストの要素数のうち大きい方
        max_possible = max(len(latest_set), len(previous_set))

        return _classify_overlap_degree(overlap_count, max_possible)

    def _describe_valence_direction(
        self,
        summaries: list[PerceptualSummary],
        min_count: int,
    ) -> TransitionDirection:
        """感情価推移方向断面:
        スライディングウィンドウ内の感情価の系列を参照し、
        推移の方向性を段階値で記述する。
        最新と最古の単純差分から段階値に変換する。
        統計的回帰分析は行わない。
        """
        valence_series = [s.emotion_valence for s in summaries]
        return _classify_valence_direction(valence_series, min_count)

    def _describe_structuring_degree(
        self,
        summaries: list[PerceptualSummary],
        min_count: int,
    ) -> StructuringDegree:
        """知覚構造化度断面:
        ウィンドウ内の知覚サマリ群において、辞書照合による構造化が
        どの程度発生したかを段階値で記述する。

        3つの比率を算出し、その単純平均から段階値に変換する:
        - 感情ラベルがデフォルト（"neutral"）以外に導出されたサマリの割合
        - 意図ラベルがデフォルト（"unknown"）以外に導出されたサマリの割合
        - 話題要素数がゼロでないサマリの割合

        3つの比率それぞれに個別の重みは付与しない。
        構造化度の高低を「理解の深さ」として解釈しない。
        判断・評価は含まない。段階的記述のみ。
        """
        window_size = len(summaries)
        if window_size == 0:
            return StructuringDegree.MODERATE

        emotion_non_default = sum(
            1 for s in summaries if s.emotion != "neutral"
        )
        intent_non_default = sum(
            1 for s in summaries if s.intent != "unknown"
        )
        topics_non_empty = sum(
            1 for s in summaries if len(s.topics) > 0
        )

        emotion_ratio = emotion_non_default / window_size
        intent_ratio = intent_non_default / window_size
        topics_ratio = topics_non_empty / window_size

        average_ratio = (emotion_ratio + intent_ratio + topics_ratio) / 3.0

        return _classify_structuring_degree(average_ratio, window_size, min_count)

    # --- Stage 3: 参照情報としての受渡準備 -------------------------------------

    def get_enrichment_text(self) -> str:
        """prompt enrichment 用のテキストを返す。

        5断面の特徴量を等価に列挙する。
        特定の断面を強調・選別しない。
        列挙順序は断面の定義順に固定。
        「注目すべき変化」「顕著な推移」等の強調表現を使わない。
        評価判定・行動指示を含まない。

        Returns:
            enrichment用のテキスト
        """
        return get_perceptual_context_summary(self._state)

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        各断面の特徴量を等価に列挙する。
        特定の断面を強調・選別しない。
        列挙順序は断面の定義順に固定。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state

        return {
            "summary_count": len(st.summaries),
            "snapshot": dict(st.snapshot),
            "summary_text": get_perceptual_context_summary(st),
        }

    def get_snapshot(self) -> dict[str, str]:
        """現在の断面別特徴量スナップショットをREAD-ONLYで返す。

        他モジュールがREAD-ONLYで参照可能な構造化データ。
        フィルタリング・選別・集約機能をアクセサに持たせない。
        全断面を等価に返す。

        Returns:
            断面名 -> 段階値の文字列 の辞書（コピー）
        """
        return dict(self._state.snapshot)

    def get_previous_snapshot(self) -> dict[str, str]:
        """直前の特徴量スナップショットをREAD-ONLYで返す。

        現在のスナップショットと比較して「推移特徴量がどの断面で変化したか」を
        記述可能にするための構造。ただし、この比較結果に基づいて処理を分岐させることはない。

        Returns:
            断面名 -> 段階値の文字列 の辞書（コピー）
        """
        return dict(self._state.previous_snapshot)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "summary_count": len(st.summaries),
            "has_snapshot": bool(st.snapshot),
            "has_previous_snapshot": bool(st.previous_snapshot),
            "snapshot": dict(st.snapshot),
        }

    # --- Save / Load --------------------------------------------------------

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = PerceptualContextState.from_dict(data)
        logger.debug(
            "Perceptual context state loaded: summaries=%d",
            len(self._state.summaries),
        )


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_perceptual_context_summary(state: PerceptualContextState) -> str:
    """知覚文脈状態の要約（enrichment用）。

    全断面を等価に列挙する。
    特定の断面を強調・選別しない。
    列挙順序は断面の定義順に固定。
    「注目すべき変化」「顕著な推移」等の強調表現を使わない。
    評価判定・行動指示を含まない。
    構造化度に「理解」「認識」「把握」「不明」「困難」等の評価的語彙を使わない。
    """
    if not state.snapshot:
        return "知覚推移: 待機中"

    parts: list[str] = []
    for section_name in SECTION_ORDER:
        value = state.snapshot.get(section_name, "")
        if not value:
            continue

        label = SECTION_LABELS.get(section_name, section_name)

        # 断面ごとに適切なラベル辞書を使用
        if section_name in (SECTION_EMOTION_CHANGE_FREQ, SECTION_INTENT_CHANGE_FREQ):
            freq_enum = ChangeFrequency(value)
            value_label = CHANGE_FREQ_LABELS.get(freq_enum, value)
        elif section_name == SECTION_TOPIC_OVERLAP:
            overlap_enum = OverlapDegree(value)
            value_label = OVERLAP_LABELS.get(overlap_enum, value)
        elif section_name == SECTION_VALENCE_DIRECTION:
            dir_enum = TransitionDirection(value)
            value_label = DIRECTION_LABELS.get(dir_enum, value)
        elif section_name == SECTION_STRUCTURING_DEGREE:
            struct_enum = StructuringDegree(value)
            value_label = STRUCTURING_DEGREE_LABELS.get(struct_enum, value)
        else:
            value_label = value

        parts.append(f"{label}={value_label}")

    if not parts:
        return "知覚推移: 待機中"

    return " ".join(parts)


# =============================================================================
# Factory
# =============================================================================

def create_perceptual_context(
    config: Optional[PerceptualContextConfig] = None,
) -> PerceptualContextProcessor:
    """PerceptualContextProcessor のファクトリ関数。

    デフォルト設定でインスタンスを生成する。
    """
    return create_perceptual_context_processor(config=config)


def create_perceptual_context_processor(
    config: Optional[PerceptualContextConfig] = None,
) -> PerceptualContextProcessor:
    """PerceptualContextProcessor のファクトリ関数。"""
    return PerceptualContextProcessor(config=config)
