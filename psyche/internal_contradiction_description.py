"""
psyche/internal_contradiction_description.py - 内部状態の矛盾並置の構造的記述

複数の内省系モジュールの出力を読み取り専用で参照し、数値的に反対方向を同時に
示している断面対を検出し、解消せず、評価せず、そのまま対として記述する構造。

設計原則 (design_internal_contradiction_description.md 準拠):
- 矛盾を解消しない。矛盾する出力のどちらかを修正・抑制・調整する処理を含まない
- 矛盾に優先度を付けない。「どちらが正しいか」「どちらが重要か」を示さない
- 矛盾を評価しない。矛盾の存在を「正常」「異常」「望ましい」「望ましくない」と判定しない
- 矛盾のパターンを抽出しない。蓄積された矛盾対の間に傾向・周期性・相関を算出しない
- 意味的な矛盾判定をしない。数値的方向の乖離のみで判定する
- 判断・行動・責任の各処理系統に接続しない。出力は参照情報としてのみ流れる
- 感情パイプラインのパラメータを変更しない
- いかなるモジュールの内部状態にも書き込まない（READ-ONLY参照）

5段パイプライン:
1. 断面対の構成 (cross-section pair construction)
2. 乖離の検出 (divergence detection)
3. 矛盾対の記述 (contradiction pair description)
4. 蓄積 (accumulation)
5. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. 全記録等価: 重み・スコア・優先度・重要度を付与しない
2. パターン抽出禁止: 傾向・周期性・相関・頻度統計を算出しない
3. 意味的矛盾判定禁止: 数値的な方向性比較のみ
4. 矛盾解消経路の構造的不在: 解消のための行動提案・推奨・指示を含まない
5. 収束監視: 同一断面対の連続検出回数の上限
6. enrichment直接露出の制限: 出力件数上限・鮮度制限
7. 評価的語彙の排除

経路遮断:
1. 出力 → 入力源モジュール（フィードバック禁止）
2. 出力 → ポリシー選択パイプライン
3. 出力 → 感情パイプライン
4. 出力 → 責任記録・評価処理
5. 出力 → 記憶忘却・固定化パイプライン
6. 出力 → 予期形成
7. 出力 → 記憶系（エピソード記憶・感情記憶結合・多経路想起・自発的想起）
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
# Constants
# =============================================================================

# 評価的語彙の禁止リスト（安全弁7: 混入が検出された場合は除去）
_FORBIDDEN_WORDS = frozenset([
    "異常", "正常", "問題", "改善", "乱れ", "安定",
    "良好", "悪化", "望ましい", "望ましくない",
])

# 断面対の識別名
PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION = "self_model_emotion_vs_meta_emotion"
PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF = "self_image_stability_vs_temporal_diff"
PAIR_IDENTITY_COHERENCE_VS_STABILIZATION = "identity_coherence_vs_stabilization"
PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN = "self_image_continuity_vs_continuity_strain"
PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE = "self_model_emotion_vs_self_image_tone"
PAIR_CROSS_SECTION_INTERNAL = "cross_section_internal"

# 断面対定義リスト（固定、動的選択なし）
PAIR_DEFINITIONS = [
    PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION,
    PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF,
    PAIR_IDENTITY_COHERENCE_VS_STABILIZATION,
    PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN,
    PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE,
    PAIR_CROSS_SECTION_INTERNAL,
]

# 断面対の日本語ラベル（enrichment用）
PAIR_LABELS = {
    PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION: "自己モデル感情-メタ感情変動",
    PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF: "自己像安定感-時間的差分規模",
    PAIR_IDENTITY_COHERENCE_VS_STABILIZATION: "同一性揺らぎ-安定化信号源",
    PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN: "自己像連続性-連続性負荷",
    PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE: "自己モデル感情-自己像感情トーン",
    PAIR_CROSS_SECTION_INTERNAL: "内省横断断面間",
}


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sanitize_text(text: str) -> str:
    """評価的語彙を除去する（安全弁7）。"""
    result = text
    for word in _FORBIDDEN_WORDS:
        if word in result:
            result = result.replace(word, "[...]")
    return result


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ContradictionRecord:
    """矛盾対の記録。

    各記録は以下を含む:
    - 断面対の識別名（2つの断面名の組）
    - 各断面が示す方向の記述（短い文字列）
    - 記録時点のティック番号
    - 記録時点のタイムスタンプ
    - 鮮度（生成時に最大値、時間経過に伴い減衰する）

    全記録は等価であり、特定の矛盾対に重み・スコア・優先度を付与しない。
    """
    record_id: str = ""
    pair_name: str = ""  # 断面対の識別名
    section_a: str = ""  # 断面Aの名称
    section_b: str = ""  # 断面Bの名称
    direction_a: str = ""  # 断面Aが示す方向の記述
    direction_b: str = ""  # 断面Bが示す方向の記述
    tick: int = 0
    timestamp: float = field(default_factory=time.time)
    freshness: float = 1.0

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "pair_name": self.pair_name,
            "section_a": self.section_a,
            "section_b": self.section_b,
            "direction_a": self.direction_a,
            "direction_b": self.direction_b,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContradictionRecord":
        return cls(
            record_id=data.get("record_id", ""),
            pair_name=data.get("pair_name", ""),
            section_a=data.get("section_a", ""),
            section_b=data.get("section_b", ""),
            direction_a=data.get("direction_a", ""),
            direction_b=data.get("direction_b", ""),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            freshness=data.get("freshness", 1.0),
        )


@dataclass
class ContradictionInputs:
    """入力データ。全てREAD-ONLY参照。

    8つの入力源から断面対の構成に必要な数値を受け取る。
    """
    # 自己モデルの統合ビュー（感情側面の出力）
    self_model_emotion_intensity: float = 0.0  # 感情強度 0-1
    self_model_emotion_spread: float = 0.0  # 感情拡散度 0-1
    self_model_emotion_conflict: bool = False  # 感情共存対の有無

    # メタ感情認知の変動候補列挙（感情推移パターンの特徴量記述）
    meta_emotion_change_speed: float = 0.0  # 変化速度
    meta_emotion_dominant_stability: float = 0.0  # 支配感情安定度

    # 自己像統合の暫定的自己像（安定感・変化感・連続性感の出力）
    self_image_stability: float = 0.0  # 安定感 0-1 (grounded=1, turbulent=0)
    self_image_continuity: float = 0.0  # 連続性感 0-1 (continuous=1, disconnected=0)
    self_image_emotional_tone: float = 0.0  # 感情トーン 0-1 (calm=1, intense=0)

    # 同一性揺らぎ認知の揺らぎ状態（シフト源の活性状況）
    identity_coherence_active_shifts: int = 0  # 活性シフト源数
    identity_coherence_level: float = 0.0  # 揺らぎ度合い 0-1 (stable=0, disconnected=1)

    # 時間的自己差分の差分記述（差分の規模・性質）
    temporal_diff_magnitude: float = 0.0  # 差分規模 0-1

    # 連続性負荷の負荷状態（負荷の水準・持続性）
    continuity_strain_level: float = 0.0  # 負荷水準 0-1 (none=0, alienated=1)

    # 内省断面横断記述のスナップショット（6断面の同時期並置）
    cross_section_values: dict[str, float] = field(default_factory=dict)

    # 安定化の構造的記述の断面記録（信号源活性数・差分度合い）
    stabilization_signal_count: int = 0  # 信号源活性数
    stabilization_diff_degree: float = 0.0  # 差分度合い 0-1

    # ティック番号
    current_tick: int = 0


# =============================================================================
# State
# =============================================================================

@dataclass
class ContradictionState:
    """内部状態。

    - 矛盾対記録のスライディングウィンドウ
    - 直前処理結果の参照用保持
    - 収束監視用の連続検出カウンタ
    """
    # 矛盾対記録のスライディングウィンドウ: 時系列順にFIFO
    contradiction_window: list[ContradictionRecord] = field(default_factory=list)

    # 直前処理結果: 外部からの読み取り専用参照にのみ使用
    # 次回処理時の比較対象としては使用しない
    previous_contradictions: list[ContradictionRecord] = field(default_factory=list)

    # 収束監視: 同一断面対の連続検出回数（安全弁5）
    consecutive_counts: dict[str, int] = field(default_factory=dict)

    # 抑制中の断面対（収束上限に達して一時抑制）
    suppressed_pairs: dict[str, int] = field(default_factory=dict)  # pair_name -> 残りティック数

    # カウンタ
    cycle_count: int = 0
    total_contradictions_detected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_window": [r.to_dict() for r in self.contradiction_window],
            "previous_contradictions": [r.to_dict() for r in self.previous_contradictions],
            "consecutive_counts": dict(self.consecutive_counts),
            "suppressed_pairs": dict(self.suppressed_pairs),
            "cycle_count": self.cycle_count,
            "total_contradictions_detected": self.total_contradictions_detected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContradictionState":
        return cls(
            contradiction_window=[
                ContradictionRecord.from_dict(r)
                for r in data.get("contradiction_window", [])
            ],
            previous_contradictions=[
                ContradictionRecord.from_dict(r)
                for r in data.get("previous_contradictions", [])
            ],
            consecutive_counts=dict(data.get("consecutive_counts", {})),
            suppressed_pairs=dict(data.get("suppressed_pairs", {})),
            cycle_count=data.get("cycle_count", 0),
            total_contradictions_detected=data.get("total_contradictions_detected", 0),
        )


# =============================================================================
# Result
# =============================================================================

@dataclass
class ContradictionResult:
    """処理結果（参照情報形式のみ）。"""
    detected_count: int = 0
    window_size: int = 0
    suppressed_pair_count: int = 0
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ContradictionConfig:
    """設定。"""
    # スライディングウィンドウの上限
    max_window_size: int = field(default_factory=lambda: coefficient_registry.get("description_common", "window_size_50"))

    # 鮮度減衰速度（ティック毎）
    freshness_decay_rate: float = 0.03

    # 鮮度の最低段階閾値（これ以下は参照情報から除外されうる）
    freshness_min_visible: float = 0.2

    # 乖離検出の閾値: 方向性の差分がこの値以上なら乖離と判定
    divergence_threshold: float = 0.4

    # 収束監視: 同一断面対の連続検出上限（安全弁5）
    consecutive_limit: int = 5

    # 収束抑制の持続ティック数
    suppression_duration: int = 3

    # enrichmentに出力する矛盾対の件数上限（安全弁6）
    max_enrichment_count: int = 5

    # enrichment出力のサイズ上限（文字数）
    max_enrichment_length: int = 1500

    # 横断的スナップショット内の断面間乖離閾値
    cross_section_divergence_threshold: float = 0.5


# =============================================================================
# Divergence Detection (Stage 2)
# =============================================================================

def _detect_self_model_vs_meta_emotion(inputs: ContradictionInputs, config: ContradictionConfig) -> Optional[ContradictionRecord]:
    """自己モデルの感情側面 と メタ感情認知の変動候補 の乖離検出。

    自己モデルが高い感情強度を示しているのにメタ感情が高い安定度を示している場合、
    または自己モデルが低い感情強度なのにメタ感情が高い変化速度を示している場合を
    数値的な方向乖離として検出する。
    """
    # 方向性: 自己モデル感情強度(高=活性) vs メタ感情安定度(高=安定)
    # 乖離 = 一方が活性を示し他方が安定を示す
    intensity = inputs.self_model_emotion_intensity
    stability = inputs.meta_emotion_dominant_stability

    divergence = abs(intensity - stability)
    if divergence < config.divergence_threshold:
        return None

    direction_a = _sanitize_text(f"感情強度={intensity:.2f}")
    direction_b = _sanitize_text(f"支配感情安定度={stability:.2f}")

    return ContradictionRecord(
        pair_name=PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION,
        section_a="自己モデル感情側面",
        section_b="メタ感情変動候補",
        direction_a=direction_a,
        direction_b=direction_b,
        tick=inputs.current_tick,
        timestamp=time.time(),
    )


def _detect_self_image_stability_vs_temporal_diff(inputs: ContradictionInputs, config: ContradictionConfig) -> Optional[ContradictionRecord]:
    """自己像統合の安定感 と 時間的自己差分の差分規模 の乖離検出。

    自己像が安定を示しているのに差分規模が大きい場合、
    または自己像が不安定なのに差分規模が小さい場合を検出する。
    """
    stability = inputs.self_image_stability
    diff_magnitude = inputs.temporal_diff_magnitude

    # 安定感(高=安定) vs 差分規模(高=変化大): 同時に高い/低いは整合的、逆方向は乖離
    divergence = abs(stability - (1.0 - diff_magnitude))
    if divergence < config.divergence_threshold:
        return None

    direction_a = _sanitize_text(f"自己像安定感={stability:.2f}")
    direction_b = _sanitize_text(f"差分規模={diff_magnitude:.2f}")

    return ContradictionRecord(
        pair_name=PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF,
        section_a="自己像安定感",
        section_b="時間的自己差分規模",
        direction_a=direction_a,
        direction_b=direction_b,
        tick=inputs.current_tick,
        timestamp=time.time(),
    )


def _detect_identity_coherence_vs_stabilization(inputs: ContradictionInputs, config: ContradictionConfig) -> Optional[ContradictionRecord]:
    """同一性揺らぎの揺らぎ度合い と 安定化記述の信号源活性数 の乖離検出。

    同一性が揺らいでいるのに信号源が少ない（安定化圧力が低い）場合、
    または同一性が安定しているのに信号源が多い（安定化圧力が高い）場合を検出する。
    """
    coherence_level = inputs.identity_coherence_level  # 0=stable, 1=disconnected
    # 信号源活性数を正規化（0-6の範囲を0-1に）
    signal_norm = _clamp(inputs.stabilization_signal_count / 6.0)

    divergence = abs(coherence_level - signal_norm)
    if divergence < config.divergence_threshold:
        return None

    direction_a = _sanitize_text(f"揺らぎ度合い={coherence_level:.2f}")
    direction_b = _sanitize_text(f"信号源活性={inputs.stabilization_signal_count}")

    return ContradictionRecord(
        pair_name=PAIR_IDENTITY_COHERENCE_VS_STABILIZATION,
        section_a="同一性揺らぎ度合い",
        section_b="安定化信号源活性数",
        direction_a=direction_a,
        direction_b=direction_b,
        tick=inputs.current_tick,
        timestamp=time.time(),
    )


def _detect_self_image_continuity_vs_strain(inputs: ContradictionInputs, config: ContradictionConfig) -> Optional[ContradictionRecord]:
    """自己像統合の連続性感 と 連続性負荷の負荷水準 の乖離検出。

    自己像が連続性を感じているのに連続性負荷が高い場合、
    または自己像が断絶感を示しているのに連続性負荷が低い場合を検出する。
    """
    continuity = inputs.self_image_continuity  # 1=continuous, 0=disconnected
    strain = inputs.continuity_strain_level  # 0=none, 1=alienated

    # 連続性感(高=連続的) vs 負荷水準(高=負荷大): 逆方向は乖離
    divergence = abs(continuity - (1.0 - strain))
    if divergence < config.divergence_threshold:
        return None

    direction_a = _sanitize_text(f"連続性感={continuity:.2f}")
    direction_b = _sanitize_text(f"負荷水準={strain:.2f}")

    return ContradictionRecord(
        pair_name=PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN,
        section_a="自己像連続性感",
        section_b="連続性負荷水準",
        direction_a=direction_a,
        direction_b=direction_b,
        tick=inputs.current_tick,
        timestamp=time.time(),
    )


def _detect_self_model_emotion_vs_self_image_tone(inputs: ContradictionInputs, config: ContradictionConfig) -> Optional[ContradictionRecord]:
    """自己モデルの感情側面 と 自己像統合の感情トーン の乖離検出。

    自己モデルが感情の活性を示しているのに自己像の感情トーンが穏やかな場合、
    またはその逆の場合を検出する。
    """
    intensity = inputs.self_model_emotion_intensity  # 0=calm, 1=intense
    tone = inputs.self_image_emotional_tone  # 1=calm, 0=intense

    # 両方とも高い→乖離（片方は活性、片方は穏やか）
    divergence = abs(intensity - (1.0 - tone))
    if divergence < config.divergence_threshold:
        return None

    direction_a = _sanitize_text(f"感情強度={intensity:.2f}")
    direction_b = _sanitize_text(f"感情トーン={tone:.2f}")

    return ContradictionRecord(
        pair_name=PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE,
        section_a="自己モデル感情強度",
        section_b="自己像感情トーン",
        direction_a=direction_a,
        direction_b=direction_b,
        tick=inputs.current_tick,
        timestamp=time.time(),
    )


def _detect_cross_section_internal(inputs: ContradictionInputs, config: ContradictionConfig) -> list[ContradictionRecord]:
    """内省断面横断記述のスナップショット内の断面間乖離検出。

    横断的スナップショット内の断面間で数値的な方向乖離を検出する。
    断面の統合は行わない。2つの断面の方向の対比のみ。
    """
    records: list[ContradictionRecord] = []
    values = inputs.cross_section_values
    if len(values) < 2:
        return records

    keys = sorted(values.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_a = keys[i]
            key_b = keys[j]
            val_a = values[key_a]
            val_b = values[key_b]
            divergence = abs(val_a - val_b)
            if divergence >= config.cross_section_divergence_threshold:
                direction_a = _sanitize_text(f"{key_a}={val_a:.2f}")
                direction_b = _sanitize_text(f"{key_b}={val_b:.2f}")
                records.append(ContradictionRecord(
                    pair_name=PAIR_CROSS_SECTION_INTERNAL,
                    section_a=key_a,
                    section_b=key_b,
                    direction_a=direction_a,
                    direction_b=direction_b,
                    tick=inputs.current_tick,
                    timestamp=time.time(),
                ))

    return records


# 断面対定義と検出関数のマッピング
_PAIR_DETECTORS = {
    PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION: _detect_self_model_vs_meta_emotion,
    PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF: _detect_self_image_stability_vs_temporal_diff,
    PAIR_IDENTITY_COHERENCE_VS_STABILIZATION: _detect_identity_coherence_vs_stabilization,
    PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN: _detect_self_image_continuity_vs_strain,
    PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE: _detect_self_model_emotion_vs_self_image_tone,
}


# =============================================================================
# Processor (5-stage pipeline)
# =============================================================================

class InternalContradictionProcessor:
    """内部状態の矛盾並置の構造的記述プロセッサ。

    5段パイプライン:
    1. 断面対の構成: 入力源から比較可能な断面の対を構成
    2. 乖離の検出: 数値的な方向性の乖離を検出
    3. 矛盾対の記述: 乖離が検出された断面対を記述
    4. 蓄積: スライディングウィンドウに蓄積（FIFO）
    5. 参照情報としての受渡準備: enrichmentテキスト生成

    矛盾を解消しない。評価しない。パターン抽出をしない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[ContradictionConfig] = None):
        self._config = config or ContradictionConfig()
        self._state = ContradictionState()

    @property
    def state(self) -> ContradictionState:
        return self._state

    @state.setter
    def state(self, value: ContradictionState) -> None:
        self._state = value

    # ─── Stage 1-3: 断面対の構成 → 乖離の検出 → 矛盾対の記述 ───

    def _detect_contradictions(self, inputs: ContradictionInputs) -> list[ContradictionRecord]:
        """事前に定義された断面組み合わせに基づき、乖離を検出し矛盾対を記述する。

        動的に対を選択する処理は含まない。
        意味的な矛盾判定は含まない。数値的な方向性の乖離のみ。
        """
        cfg = self._config
        detected: list[ContradictionRecord] = []

        # 固定の断面対定義に基づく検出
        for pair_name, detector_fn in _PAIR_DETECTORS.items():
            # 安全弁5: 収束抑制中の断面対はスキップ
            if pair_name in self._state.suppressed_pairs:
                continue
            result = detector_fn(inputs, cfg)
            if result is not None:
                detected.append(result)

        # 内省横断断面内の乖離（PAIR_CROSS_SECTION_INTERNAL）
        if PAIR_CROSS_SECTION_INTERNAL not in self._state.suppressed_pairs:
            cross_records = _detect_cross_section_internal(inputs, cfg)
            detected.extend(cross_records)

        return detected

    # ─── Stage 4: 蓄積 ──────────────────────────────────────────

    def _accumulate(self, detected: list[ContradictionRecord]) -> None:
        """矛盾対をスライディングウィンドウに蓄積する。

        蓄積は時系列順のFIFOであり、最古の記録の押し出しが唯一のデータ消失経路。
        特定の記録を選択的に消去する処理は存在しない。
        同一の断面対が連続して記録される場合でも、各記録は独立した記録として蓄積。
        """
        cfg = self._config

        # 直前処理結果を更新
        self._state.previous_contradictions = list(detected)

        # 新規記録をウィンドウに追加
        for record in detected:
            self._state.contradiction_window.append(record)
            self._state.total_contradictions_detected += 1

        # 上限による押し出し（FIFO: 唯一のデータ消失経路）
        if len(self._state.contradiction_window) > cfg.max_window_size:
            overflow = len(self._state.contradiction_window) - cfg.max_window_size
            self._state.contradiction_window = self._state.contradiction_window[overflow:]

        # 鮮度減衰: 全記録の鮮度を一律に減衰
        for record in self._state.contradiction_window:
            record.freshness = _clamp(record.freshness - cfg.freshness_decay_rate)

    # ─── 安全弁5: 収束監視 ─────────────────────────────────────

    def _monitor_convergence(self, detected: list[ContradictionRecord]) -> None:
        """同一断面対の連続検出回数を監視し、上限に達した場合に一時抑制する。

        これは矛盾の解消ではなく、蓄積の均等性維持のための構造的制約。
        """
        cfg = self._config
        detected_names = set(r.pair_name for r in detected)

        # 今回新たに抑制された断面対（同一ティックでの減算を防ぐ）
        newly_suppressed: set[str] = set()

        for pair_name in PAIR_DEFINITIONS:
            if pair_name in detected_names:
                current = self._state.consecutive_counts.get(pair_name, 0) + 1
                self._state.consecutive_counts[pair_name] = current
                if current >= cfg.consecutive_limit:
                    # 上限到達: 一時的に抑制
                    self._state.suppressed_pairs[pair_name] = cfg.suppression_duration
                    self._state.consecutive_counts[pair_name] = 0
                    newly_suppressed.add(pair_name)
            else:
                # 検出されなかった場合、連続カウントをリセット
                self._state.consecutive_counts[pair_name] = 0

        # 抑制カウンタの減算（今回新たに抑制されたものは減算しない）
        expired = []
        for pair_name in list(self._state.suppressed_pairs.keys()):
            if pair_name in newly_suppressed:
                continue
            self._state.suppressed_pairs[pair_name] -= 1
            if self._state.suppressed_pairs[pair_name] <= 0:
                expired.append(pair_name)
        for pair_name in expired:
            del self._state.suppressed_pairs[pair_name]

    # ─── Stage 5: 参照情報としての受渡準備 ────────────────────

    def get_enrichment_text(self) -> str:
        """enrichmentへの参照テキストを生成する。

        安全弁6: 件数上限を設け、全蓄積を無制限に外部に露出しない。
        出力される矛盾対は直近のもののうち鮮度が最低段階に達していないものに限定。
        特定の矛盾対を強調・選別・要約する処理を含まない。
        等価列挙に限定する。
        """
        cfg = self._config
        window = self._state.contradiction_window

        if not window:
            return "矛盾並置: 待機中"

        # 鮮度が最低段階以上のもののみ
        visible = [r for r in window if r.freshness >= cfg.freshness_min_visible]
        if not visible:
            return "矛盾並置: 待機中"

        # 直近の限られた件数分（安全弁6）
        target = visible[-cfg.max_enrichment_count:]

        parts: list[str] = []
        for record in target:
            label = PAIR_LABELS.get(record.pair_name, record.pair_name)
            line = _sanitize_text(
                f"[t{record.tick}] {label}: {record.section_a}({record.direction_a}) / {record.section_b}({record.direction_b})"
            )
            parts.append(line)

        text = "\n".join(parts)

        # サイズ上限
        if len(text) > cfg.max_enrichment_length:
            text = text[:cfg.max_enrichment_length]

        return text

    def get_contradiction_window(self) -> list[dict[str, Any]]:
        """蓄積リストをREAD-ONLYで返す。

        フィルタリング・選別・集約機能をアクセサに持たせない。
        全記録を等価に返す。
        """
        return [r.to_dict() for r in self._state.contradiction_window]

    def get_previous_contradictions(self) -> list[dict[str, Any]]:
        """直前処理結果をREAD-ONLYで返す。"""
        return [r.to_dict() for r in self._state.previous_contradictions]

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "window_size": len(self._state.contradiction_window),
            "previous_count": len(self._state.previous_contradictions),
            "suppressed_pairs": len(self._state.suppressed_pairs),
            "cycle_count": self._state.cycle_count,
            "total_contradictions_detected": self._state.total_contradictions_detected,
        }

    # ─── Main processing entry point ──────────────────────────

    def process(self, inputs: ContradictionInputs) -> ContradictionResult:
        """5段パイプラインの一括実行。

        入力源の最新出力から矛盾対の検出と記述が新たに行われる。
        入力源モジュールの内部状態には書き込まない（READ-ONLY）。

        Args:
            inputs: 8つの入力源からの読み取り値

        Returns:
            ContradictionResult: 処理結果（参照情報形式のみ）
        """
        self._state.cycle_count += 1

        # Stage 1-3: 断面対の構成 → 乖離の検出 → 矛盾対の記述
        detected = self._detect_contradictions(inputs)

        # Stage 4: 蓄積
        self._accumulate(detected)

        # 安全弁5: 収束監視
        self._monitor_convergence(detected)

        logger.debug(
            "Internal contradiction processed: tick=%d, detected=%d, window=%d",
            inputs.current_tick,
            len(detected),
            len(self._state.contradiction_window),
        )

        # Stage 5: 受渡準備（結果構成）
        return ContradictionResult(
            detected_count=len(detected),
            window_size=len(self._state.contradiction_window),
            suppressed_pair_count=len(self._state.suppressed_pairs),
            cycle_count=self._state.cycle_count,
        )

    # ─── Save / Load ──────────────────────────────────────────

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = ContradictionState.from_dict(data)
        logger.debug(
            "Internal contradiction state loaded: window=%d",
            len(self._state.contradiction_window),
        )


# =============================================================================
# Summary
# =============================================================================

def get_contradiction_summary(state: ContradictionState) -> str:
    """enrichment用の要約テキスト。

    評価判定を含まない。等価列挙に限定する。
    """
    if state.cycle_count == 0 and not state.contradiction_window:
        return "矛盾並置: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    window_size = len(state.contradiction_window)
    if window_size:
        parts.append(f"蓄積={window_size}")

    prev_count = len(state.previous_contradictions)
    if prev_count:
        parts.append(f"直近検出={prev_count}")

    suppressed = len(state.suppressed_pairs)
    if suppressed:
        parts.append(f"抑制中={suppressed}")

    return " ".join(parts) if parts else "矛盾並置: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_contradiction_processor(
    config: Optional[ContradictionConfig] = None,
) -> InternalContradictionProcessor:
    """InternalContradictionProcessor のファクトリ関数。"""
    return InternalContradictionProcessor(config=config)
