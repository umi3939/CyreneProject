"""
psyche/attention_distribution_description.py - 注意配分の構造的記述

1ティックの処理サイクルにおいて、どの入力源・内部信号がどれだけの
処理量を持っていたかを事後的に記述する層。

設計原則 (design_attention_distribution_description.md 準拠):
- 処理帯域の配分を制御・調整・最適化しない
- 「どこに帯域を集中すべきか」という規範を生成しない
- 特定の入力源・内部信号の帯域を増減させるフィードバック経路を持たない
- 帯域の集中・分散に対して評価を付与しない
- 帯域の分布から傾向・周期性・統計量・相関を抽出しない
- 参照頻度記述の責務（累積的な参照回数の横断読み取り）を代替・重複しない
- 入力経路間均衡記述の責務（経路使用事実の記録）を代替・重複しない

断面構造:
  - 生成時刻
  - 各入力源・内部信号の段階値（列挙型、等価に並置）
    - 外部入力源: 知覚入力段階値、テキスト入力段階値、自発起動段階値
    - 内部信号: 感情活性段階値、記憶想起段階値、動機活性段階値、
                目標活性段階値、責任保持段階値
  - 断面全体の集中度（少数集中か多数分散かの度合い）

断面履歴:
  - 過去に生成された断面の時系列リスト（FIFO、有限上限）

変動記述:
  - 直近の断面と過去の断面群を比較した集中度の変動方向・変動幅
  - 比較に使用した過去断面の件数

安全弁（7種）:
  1. 全記録等価維持保証 — 各入力源・内部信号の段階値は等価に並置される。
     特定の入力源・内部信号を重要視・軽視する重み付け・順位付け・選別を行わない
  2. 評価的変換の禁止 — 集中度は数値的な分布特徴であり、
     「良い分布」「悪い分布」を示す評価軸を持たない。
     段階値も全て等価であり、いずれも望ましいとしない
  3. 累積的傾向の抑制 — 変動記述は断面間の差分から導出されるのみであり、
     「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない。
     断面生成のたびに断面履歴から再導出される
  4. 断面履歴の有限性 — 断面履歴には保持上限があり、古い断面は消失する。
     全期間の帯域分布を恒久的に蓄積しない
  5. 帯域制御経路の遮断 — 本機能の出力を処理帯域の配分制御に使用する経路を持たない。
     方針候補生成・方針選択・判断バイアス計算・入力経路選択・安定化バルブへの
     出力経路を構造的に排除する
  6. パターン抽出禁止 — 蓄積された断面から傾向・周期性・統計量・相関を算出しない。
     断面は等価に並置されるのみ
  7. 出力経路不拡張 — 初期実装で定義した出力経路以外を運用中に動的に追加する
     仕組みを持たない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AttentionDistributionConfig:
    """注意配分記述モジュールの設定。"""

    # 断面履歴の最大保持件数（安全弁4: 有限性）
    max_snapshot_history: int = 30

    # 変動記述で比較する過去断面の件数
    variation_comparison_count: int = 5


# =============================================================================
# 入力源・内部信号の識別子
# =============================================================================
# 設計書で列挙された8種の入力源・内部信号。
# 外部入力源: 知覚入力、テキスト入力、自発起動
# 内部信号: 感情活性、記憶想起、動機活性、目標活性、責任保持

SOURCE_PERCEPTION = "perception"
SOURCE_TEXT_INPUT = "text_input"
SOURCE_SPONTANEOUS = "spontaneous"
SIGNAL_EMOTION = "emotion"
SIGNAL_MEMORY = "memory"
SIGNAL_MOTIVATION = "motivation"
SIGNAL_GOAL = "goal"
SIGNAL_RESPONSIBILITY = "responsibility"

ALL_SOURCE_KEYS = [
    SOURCE_PERCEPTION,
    SOURCE_TEXT_INPUT,
    SOURCE_SPONTANEOUS,
    SIGNAL_EMOTION,
    SIGNAL_MEMORY,
    SIGNAL_MOTIVATION,
    SIGNAL_GOAL,
    SIGNAL_RESPONSIBILITY,
]


# =============================================================================
# 段階値の定義（列挙型）
# =============================================================================
# 段階値は列挙型であり、具体的な件数を外部に露出しない。
# 全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。
# 安全弁1: 全記録等価。安全弁2: 評価的変換の禁止。

class QuantityLevel(Enum):
    """入力源・内部信号の処理量の段階値。
    全段階値は等価。特定段階値を望ましいとしない。
    """
    ABSENT = "absent"         # 0（存在しない）
    MINIMAL = "minimal"       # 1
    FEW = "few"               # 2-3
    MODERATE = "moderate"     # 4-7
    MANY = "many"             # 8以上


class ConcentrationLevel(Enum):
    """断面全体の集中度の段階値。
    全段階値は等価。特定段階値を望ましいとしない。
    集中度は「是正すべき問題」ではない。
    安全弁2: 評価的変換の禁止。
    """
    DISPERSED = "dispersed"       # 多数に分散
    SLIGHT = "slight"             # やや集中
    MODERATE = "moderate"         # 中程度の集中
    CONCENTRATED = "concentrated" # 少数に集中


# =============================================================================
# 段階値の決定関数
# =============================================================================

def determine_quantity_level(count: int) -> QuantityLevel:
    """量的指標から段階値を決定する。

    段階値は等価。評価的含意を持たない。
    安全弁1: 全記録等価。安全弁2: 評価的変換の禁止。

    Args:
        count: 量的指標（0以上の整数）

    Returns:
        段階値（全て等価）
    """
    if count <= 0:
        return QuantityLevel.ABSENT
    elif count == 1:
        return QuantityLevel.MINIMAL
    elif count <= 3:
        return QuantityLevel.FEW
    elif count <= 7:
        return QuantityLevel.MODERATE
    else:
        return QuantityLevel.MANY


def determine_concentration_level(concentration: float) -> ConcentrationLevel:
    """集中度の数値から段階値を決定する。

    段階値は等価。集中を「是正すべき問題」としない。
    安全弁2: 評価的変換の禁止。

    Args:
        concentration: 集中度（0.0〜1.0）

    Returns:
        段階値（全て等価）
    """
    if concentration < 0.2:
        return ConcentrationLevel.DISPERSED
    elif concentration < 0.45:
        return ConcentrationLevel.SLIGHT
    elif concentration < 0.7:
        return ConcentrationLevel.MODERATE
    else:
        return ConcentrationLevel.CONCENTRATED


# =============================================================================
# 集中度の計算
# =============================================================================

def compute_concentration(source_quantities: dict[str, int]) -> float:
    """各入力源・内部信号の量的指標から集中度を計算する。

    安全弁2: この値は評価的意味を持たない。
    「良い集中度」「悪い集中度」を示す軸ではない。

    Returns:
        0.0（完全分散または処理なし）〜 1.0（完全集中）
    """
    values = [source_quantities.get(k, 0) for k in ALL_SOURCE_KEYS]
    total = sum(values)
    if total == 0:
        return 0.0

    n = len(values)
    if n <= 1:
        return 0.0

    active_sources = [v for v in values if v > 0]
    if len(active_sources) <= 1:
        return 1.0 if total > 0 else 0.0

    # ジニ係数的な集中度指標
    sorted_values = sorted(values)
    weighted_sum = 0.0
    for i, v in enumerate(sorted_values):
        weighted_sum += (i + 1) * v

    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

    return _clamp(gini)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AttentionSnapshot:
    """注意配分断面。1ティックの処理量分布を格納した断面。

    断面は生成後に変更されない（不変）。
    安全弁1: 全記録等価。各入力源・内部信号の段階値は等価に並置される。
    安全弁2: 評価的変換の禁止。集中度は数値的な分布特徴であり、
    「良い分布」「悪い分布」を示す評価軸を持たない。
    """

    # 生成時刻
    timestamp: float = 0.0

    # 各入力源・内部信号の段階値（等価に並置）
    # 外部入力源
    perception_level: str = QuantityLevel.ABSENT.value
    text_input_level: str = QuantityLevel.ABSENT.value
    spontaneous_level: str = QuantityLevel.ABSENT.value
    # 内部信号
    emotion_level: str = QuantityLevel.ABSENT.value
    memory_level: str = QuantityLevel.ABSENT.value
    motivation_level: str = QuantityLevel.ABSENT.value
    goal_level: str = QuantityLevel.ABSENT.value
    responsibility_level: str = QuantityLevel.ABSENT.value

    # 断面全体の集中度（0.0=完全分散, 1.0=完全集中）
    # 安全弁2: この値は評価的意味を持たない
    concentration: float = 0.0

    # 集中度の段階値
    concentration_level: str = ConcentrationLevel.DISPERSED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "perception_level": self.perception_level,
            "text_input_level": self.text_input_level,
            "spontaneous_level": self.spontaneous_level,
            "emotion_level": self.emotion_level,
            "memory_level": self.memory_level,
            "motivation_level": self.motivation_level,
            "goal_level": self.goal_level,
            "responsibility_level": self.responsibility_level,
            "concentration": self.concentration,
            "concentration_level": self.concentration_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionSnapshot":
        return cls(
            timestamp=data.get("timestamp", 0.0),
            perception_level=data.get("perception_level", QuantityLevel.ABSENT.value),
            text_input_level=data.get("text_input_level", QuantityLevel.ABSENT.value),
            spontaneous_level=data.get("spontaneous_level", QuantityLevel.ABSENT.value),
            emotion_level=data.get("emotion_level", QuantityLevel.ABSENT.value),
            memory_level=data.get("memory_level", QuantityLevel.ABSENT.value),
            motivation_level=data.get("motivation_level", QuantityLevel.ABSENT.value),
            goal_level=data.get("goal_level", QuantityLevel.ABSENT.value),
            responsibility_level=data.get("responsibility_level", QuantityLevel.ABSENT.value),
            concentration=data.get("concentration", 0.0),
            concentration_level=data.get("concentration_level", ConcentrationLevel.DISPERSED.value),
        )


@dataclass
class AttentionVariation:
    """変動記述。直近の断面と過去の断面群を比較した集中度の変動。

    安全弁3: 累積的傾向の抑制。変動記述は断面間の差分から導出されるのみであり、
    「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない。
    新しい断面が生成されるたびに、断面履歴から再導出される。
    """

    # 集中度の変動方向（正=集中方向, 負=分散方向, 0=変化なし）
    concentration_direction: float = 0.0

    # 集中度の変動幅（絶対値）
    concentration_magnitude: float = 0.0

    # 比較に使用した過去断面の件数
    comparison_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concentration_direction": self.concentration_direction,
            "concentration_magnitude": self.concentration_magnitude,
            "comparison_count": self.comparison_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionVariation":
        return cls(
            concentration_direction=data.get("concentration_direction", 0.0),
            concentration_magnitude=data.get("concentration_magnitude", 0.0),
            comparison_count=data.get("comparison_count", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class AttentionDistributionState:
    """注意配分記述モジュールの内部状態。

    安全弁4: 断面履歴の有限性。保持上限あり、古い断面は消失する。
    """

    # 断面履歴（時系列順、先入先出）
    snapshot_history: list[AttentionSnapshot] = field(default_factory=list)

    # 直近の変動記述（断面が2件以上ある場合のみ有効）
    latest_variation: Optional[AttentionVariation] = None

    # 診断カウンタ（処理分岐に使用しない）
    total_snapshots_generated: int = 0
    total_snapshots_expired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_history": [s.to_dict() for s in self.snapshot_history],
            "latest_variation": self.latest_variation.to_dict() if self.latest_variation else None,
            "total_snapshots_generated": self.total_snapshots_generated,
            "total_snapshots_expired": self.total_snapshots_expired,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionDistributionState":
        history = [
            AttentionSnapshot.from_dict(s)
            for s in data.get("snapshot_history", [])
        ]
        var_data = data.get("latest_variation")
        variation = AttentionVariation.from_dict(var_data) if var_data else None
        return cls(
            snapshot_history=history,
            latest_variation=variation,
            total_snapshots_generated=data.get("total_snapshots_generated", 0),
            total_snapshots_expired=data.get("total_snapshots_expired", 0),
        )


# =============================================================================
# 収集: 各構造から量的指標を読み取り専用で収集する
# =============================================================================
# 設計書: 「各入力源・内部信号の量的指標は、該当する構造の公開アクセサを
# 経由して読み取り専用で取得する。内部状態への直接アクセスは行わない。」

def collect_source_quantities(
    *,
    perception_state: Any = None,
    text_dialogue_state: Any = None,
    spontaneous_state: Any = None,
    emotion_state: Any = None,
    memory_state: Any = None,
    motivation_state: Any = None,
    transient_goal_state: Any = None,
    scoped_goal_state: Any = None,
    responsibility_state: Any = None,
    has_perception_input: bool = False,
    has_text_input: bool = False,
    has_spontaneous_activation: bool = False,
    perception_element_count: int = 0,
) -> dict[str, int]:
    """各構造から量的指標の現在値を読み取り専用で収集する。

    本関数は各構造の内部状態を変更する権限を持たない。
    安全弁1: 全記録等価。収集した量的指標にフィルタリング・選別・順位付けを行わない。

    Args:
        perception_state: 知覚構造の状態（duck typing）
        text_dialogue_state: テキスト対話入力構造の状態（duck typing）
        spontaneous_state: 自発起動構造の状態（duck typing）
        emotion_state: 感情構造の状態（duck typing）
        memory_state: 記憶構造の状態（duck typing）
        motivation_state: 動機構造の状態（duck typing）
        transient_goal_state: 一時的目標構造の状態（duck typing）
        scoped_goal_state: スコープ目標構造の状態（duck typing）
        responsibility_state: 責任構造の状態（duck typing）
        has_perception_input: 今回のティックで知覚入力が存在したか
        has_text_input: 今回のティックでテキスト入力が存在したか
        has_spontaneous_activation: 今回のティックで自発起動が発生したか
        perception_element_count: 知覚結果の要素数（直接渡し）

    Returns:
        入力源・内部信号識別子 → 量的指標 の辞書
    """
    result: dict[str, int] = {}

    # ── 外部入力源 ──

    # 知覚構造: 知覚入力が存在したか、知覚結果の要素数
    perception_count = 0
    if has_perception_input:
        perception_count = max(1, perception_element_count)
    elif perception_state is not None:
        # duck typing: 知覚結果の要素数を読み取る
        elements = getattr(perception_state, "elements", None)
        if elements is not None:
            perception_count = len(elements) if hasattr(elements, "__len__") else 0
        elif has_perception_input:
            perception_count = 1
    result[SOURCE_PERCEPTION] = perception_count

    # テキスト対話入力構造: テキスト入力が存在したか
    text_count = 0
    if has_text_input:
        text_count = 1
    elif text_dialogue_state is not None:
        # duck typing: active_units の数を読み取る
        active_units = getattr(text_dialogue_state, "active_units", None)
        if active_units is not None and hasattr(active_units, "__len__"):
            text_count = len(active_units)
    result[SOURCE_TEXT_INPUT] = text_count

    # 自発起動構造: 自発起動が発生したか
    spontaneous_count = 0
    if has_spontaneous_activation:
        spontaneous_count = 1
    elif spontaneous_state is not None:
        # duck typing: 最新の起動結果があるか
        last_result = getattr(spontaneous_state, "last_activation_result", None)
        if last_result is not None:
            activated = getattr(last_result, "activated", False)
            if activated:
                spontaneous_count = 1
    result[SOURCE_SPONTANEOUS] = spontaneous_count

    # ── 内部信号 ──

    # 感情構造: 閾値以上の感情の数
    emotion_count = 0
    if emotion_state is not None:
        # duck typing: EmotionVectorの各感情値を読み取り
        # 一般的にはemotion_vectorからラベル付き値を取得
        emotions = getattr(emotion_state, "emotions", None)
        if emotions is not None and isinstance(emotions, dict):
            # 閾値以上の感情を数える
            emotion_count = sum(1 for v in emotions.values() if isinstance(v, (int, float)) and v > 0.1)
        elif hasattr(emotion_state, "__dict__"):
            # EmotionVector: joy, sadness, anger, etc. のフィールドを持つ
            for attr_name in ("joy", "sadness", "anger", "fear", "disgust", "surprise", "trust", "anticipation"):
                val = getattr(emotion_state, attr_name, None)
                if isinstance(val, (int, float)) and val > 0.1:
                    emotion_count += 1
    result[SIGNAL_EMOTION] = emotion_count

    # 記憶構造: 想起された記憶の件数
    memory_count = 0
    if memory_state is not None:
        # duck typing: recalled_memories or recent_recalls
        recalled = getattr(memory_state, "recalled_memories", None)
        if recalled is not None and hasattr(recalled, "__len__"):
            memory_count = len(recalled)
        else:
            recent = getattr(memory_state, "recent_recalls", None)
            if recent is not None and hasattr(recent, "__len__"):
                memory_count = len(recent)
    result[SIGNAL_MEMORY] = memory_count

    # 動機構造: 活性状態にある動機の件数
    motivation_count = 0
    if motivation_state is not None:
        # duck typing: entries を走査し活性状態のものを数える
        entries = getattr(motivation_state, "entries", None)
        if entries is not None and hasattr(entries, "__len__"):
            for e in entries:
                strength = getattr(e, "strength", None)
                if strength is not None:
                    # MotiveStrength enumの場合
                    str_val = strength.value if hasattr(strength, "value") else str(strength)
                    if str_val not in ("dormant", "none", ""):
                        motivation_count += 1
                else:
                    # 存在するだけでカウント
                    motivation_count += 1
    result[SIGNAL_MOTIVATION] = motivation_count

    # 目標構造: 活性状態にある一時的目標・スコープ目標の有無
    goal_count = 0
    if transient_goal_state is not None:
        # duck typing: active_goals
        active_goals = getattr(transient_goal_state, "active_goals", None)
        if active_goals is not None and hasattr(active_goals, "__len__"):
            goal_count += len(active_goals)
        else:
            # 単一active_goal
            active = getattr(transient_goal_state, "active_goal", None)
            if active is not None:
                goal_count += 1
    if scoped_goal_state is not None:
        scoped = getattr(scoped_goal_state, "current_goal", None)
        if scoped is not None:
            goal_count += 1
    result[SIGNAL_GOAL] = goal_count

    # 責任構造: 保持されている責任ユニットの件数
    responsibility_count = 0
    if responsibility_state is not None:
        # duck typing: units, decisions, entries
        units = getattr(responsibility_state, "units", None)
        if units is not None and hasattr(units, "__len__"):
            responsibility_count = len(units)
        else:
            decisions = getattr(responsibility_state, "decisions", None)
            if decisions is not None and hasattr(decisions, "__len__"):
                responsibility_count = len(decisions)
    result[SIGNAL_RESPONSIBILITY] = responsibility_count

    return result


# =============================================================================
# 断面の構成
# =============================================================================

def compose_snapshot(
    source_quantities: dict[str, int],
    timestamp: Optional[float] = None,
) -> AttentionSnapshot:
    """収集された量的指標から断面を構成する。

    安全弁1: 全記録等価。断面構成時にフィルタリング・選別を行わない。
    安全弁2: 評価的変換の禁止。

    Args:
        source_quantities: 入力源・内部信号識別子→量的指標
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        構成された断面（不変）
    """
    ts = timestamp if timestamp is not None else time.time()

    # 各入力源・内部信号の段階値変換
    perception_lv = determine_quantity_level(source_quantities.get(SOURCE_PERCEPTION, 0))
    text_input_lv = determine_quantity_level(source_quantities.get(SOURCE_TEXT_INPUT, 0))
    spontaneous_lv = determine_quantity_level(source_quantities.get(SOURCE_SPONTANEOUS, 0))
    emotion_lv = determine_quantity_level(source_quantities.get(SIGNAL_EMOTION, 0))
    memory_lv = determine_quantity_level(source_quantities.get(SIGNAL_MEMORY, 0))
    motivation_lv = determine_quantity_level(source_quantities.get(SIGNAL_MOTIVATION, 0))
    goal_lv = determine_quantity_level(source_quantities.get(SIGNAL_GOAL, 0))
    responsibility_lv = determine_quantity_level(source_quantities.get(SIGNAL_RESPONSIBILITY, 0))

    # 集中度計算
    concentration = compute_concentration(source_quantities)
    concentration_lv = determine_concentration_level(concentration)

    return AttentionSnapshot(
        timestamp=ts,
        perception_level=perception_lv.value,
        text_input_level=text_input_lv.value,
        spontaneous_level=spontaneous_lv.value,
        emotion_level=emotion_lv.value,
        memory_level=memory_lv.value,
        motivation_level=motivation_lv.value,
        goal_level=goal_lv.value,
        responsibility_level=responsibility_lv.value,
        concentration=concentration,
        concentration_level=concentration_lv.value,
    )


# =============================================================================
# 変動記述の導出
# =============================================================================

def derive_variation(
    snapshot_history: list[AttentionSnapshot],
    config: AttentionDistributionConfig,
) -> Optional[AttentionVariation]:
    """断面履歴から変動記述を導出する。

    安全弁3: 累積的傾向の抑制。変動記述は断面間の差分から導出されるのみ。
    「長期的にこの方向に向かっている」という傾向の累積的蓄積を行わない。
    新しい断面が生成されるたびに、断面履歴から再導出される。

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

    # 過去の集中度の平均
    past_concentration_avg = sum(s.concentration for s in past_snapshots) / len(past_snapshots)

    # 変動方向と幅
    concentration_diff = latest.concentration - past_concentration_avg

    return AttentionVariation(
        concentration_direction=concentration_diff,
        concentration_magnitude=abs(concentration_diff),
        comparison_count=len(past_snapshots),
    )


# =============================================================================
# メイン処理: 断面生成と履歴管理
# =============================================================================

def process_attention_distribution(
    state: AttentionDistributionState,
    *,
    perception_state: Any = None,
    text_dialogue_state: Any = None,
    spontaneous_state: Any = None,
    emotion_state: Any = None,
    memory_state: Any = None,
    motivation_state: Any = None,
    transient_goal_state: Any = None,
    scoped_goal_state: Any = None,
    responsibility_state: Any = None,
    has_perception_input: bool = False,
    has_text_input: bool = False,
    has_spontaneous_activation: bool = False,
    perception_element_count: int = 0,
    config: Optional[AttentionDistributionConfig] = None,
    timestamp: Optional[float] = None,
) -> AttentionDistributionState:
    """注意配分記述の1サイクル処理を実行する。

    オーケストレーション処理の一周期の完了後に呼び出される。
    設計書: 「本機能の処理は、そのティックの全処理フェーズが完了した後に実行されること。」

    安全弁1: 全記録等価維持保証
    安全弁2: 評価的変換の禁止
    安全弁3: 累積的傾向の抑制
    安全弁4: 断面履歴の有限性
    安全弁5: 帯域制御経路の遮断（本関数は帯域制御に対する出力経路を持たない）
    安全弁6: パターン抽出禁止
    安全弁7: 出力経路不拡張

    入力構造の量的指標は読み取り専用。書き込み能力を付与しない。
    本関数は方針候補生成・方針選択・判断バイアス計算への出力経路を持たない。
    本関数は入力経路選択への出力経路を持たない。
    本関数は安定化バルブへの出力経路を持たない。
    本関数は忘却処理への出力経路を持たない。
    本関数は想起経路選択への出力経路を持たない。

    Args:
        state: 現在の状態
        (各構造のステート): 読み取り専用で参照
        has_perception_input: 今回のティックで知覚入力が存在したか
        has_text_input: 今回のティックでテキスト入力が存在したか
        has_spontaneous_activation: 今回のティックで自発起動が発生したか
        perception_element_count: 知覚結果の要素数
        config: 設定
        timestamp: 断面の生成時刻

    Returns:
        更新されたAttentionDistributionState
    """
    cfg = config or AttentionDistributionConfig()

    # ── 収集 ──
    source_quantities = collect_source_quantities(
        perception_state=perception_state,
        text_dialogue_state=text_dialogue_state,
        spontaneous_state=spontaneous_state,
        emotion_state=emotion_state,
        memory_state=memory_state,
        motivation_state=motivation_state,
        transient_goal_state=transient_goal_state,
        scoped_goal_state=scoped_goal_state,
        responsibility_state=responsibility_state,
        has_perception_input=has_perception_input,
        has_text_input=has_text_input,
        has_spontaneous_activation=has_spontaneous_activation,
        perception_element_count=perception_element_count,
    )

    # ── 断面構成 ──
    snapshot = compose_snapshot(source_quantities, timestamp=timestamp)

    # ── 断面の時系列保持 ──
    new_history = list(state.snapshot_history)
    new_history.append(snapshot)

    # 安全弁4: 断面履歴の有限性。保持上限到達時はFIFO
    expired = 0
    if len(new_history) > cfg.max_snapshot_history:
        expired = len(new_history) - cfg.max_snapshot_history
        new_history = new_history[expired:]

    # ── 変動記述の再導出 ──
    # 安全弁3: 新しい断面が生成されるたびに再導出。累積蓄積しない。
    variation = derive_variation(new_history, cfg)

    new_state = AttentionDistributionState(
        snapshot_history=new_history,
        latest_variation=variation,
        total_snapshots_generated=state.total_snapshots_generated + 1,
        total_snapshots_expired=state.total_snapshots_expired + expired,
    )

    logger.debug(
        "Attention distribution snapshot: concentration=%.4f (%s), "
        "history=%d, expired=%d",
        snapshot.concentration,
        snapshot.concentration_level,
        len(new_history),
        expired,
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁5: 帯域制御経路の遮断。
# 安全弁7: 出力先は内省系構造への参照情報およびenrichment等価列挙に限定。
# 方針候補生成・方針選択・判断バイアス計算への出力経路を持たない。
# 入力経路選択への出力経路を持たない。
# 安定化バルブへの出力経路を持たない。
# 忘却処理への出力経路を持たない。
# 想起経路選択への出力経路を持たない。

def get_latest_snapshot(state: AttentionDistributionState) -> Optional[AttentionSnapshot]:
    """最新の断面を返す（内省系参照経路、READ-ONLY）。

    参照行為によって状態が変化することはない。

    Returns:
        直近の断面。履歴がなければ None。
    """
    if not state.snapshot_history:
        return None
    return state.snapshot_history[-1]


def get_snapshot_history(state: AttentionDistributionState) -> list[AttentionSnapshot]:
    """断面履歴全体を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    参照行為によって状態が変化することはない。

    Returns:
        断面履歴のコピー。
    """
    return list(state.snapshot_history)


def get_latest_variation(state: AttentionDistributionState) -> Optional[AttentionVariation]:
    """最新の変動記述を返す（内省系参照経路、READ-ONLY）。

    安全弁3: 累積的傾向の抑制。変動記述は「改善」「悪化」に相当する方向性を持たない。
    参照行為によって状態が変化することはない。

    Returns:
        直近の変動記述。断面が2件未満の場合は None。
    """
    return state.latest_variation


def get_attention_distribution_summary(state: AttentionDistributionState) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の入力源・内部信号を強調・選別しない。
    安全弁5: 帯域制御経路の遮断。

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
        summary["perception_level"] = latest.perception_level
        summary["text_input_level"] = latest.text_input_level
        summary["spontaneous_level"] = latest.spontaneous_level
        summary["emotion_level"] = latest.emotion_level
        summary["memory_level"] = latest.memory_level
        summary["motivation_level"] = latest.motivation_level
        summary["goal_level"] = latest.goal_level
        summary["responsibility_level"] = latest.responsibility_level
        summary["concentration"] = latest.concentration
        summary["concentration_level"] = latest.concentration_level
        summary["latest_timestamp"] = latest.timestamp

    if variation is not None:
        summary["variation_concentration_direction"] = variation.concentration_direction
        summary["variation_concentration_magnitude"] = variation.concentration_magnitude
        summary["variation_comparison_count"] = variation.comparison_count

    return summary


# =============================================================================
# Enrichment出力
# =============================================================================
# 設計書: 「本機能の断面がenrichment経路を通じて外部出力層に提供される場合、
# 段階値の等価列挙に限定し、「集中している」「分散している」等の
# 評価的語彙を使用しない。」
# enrichment出力は段階値の等価列挙に限定される。
# 具体的な件数・比率・順位を含まない。
# 評価的語彙を含まない。

def get_enrichment_text(state: AttentionDistributionState) -> str:
    """enrichment出力テキストを生成する。

    段階値の等価列挙に限定。具体的な件数・比率・順位を含まない。
    「集中している」「分散している」等の評価的語彙を含まない。

    安全弁1: 全記録等価。各入力源・内部信号の段階値は等価に並置される。
    安全弁2: 評価的変換の禁止。
    安全弁5: 帯域制御経路の遮断。

    Returns:
        段階値の等価列挙テキスト。データがなければ「待機中」を含むテキスト。
    """
    latest = get_latest_snapshot(state)
    if latest is None:
        return "注意配分: 待機中"

    parts: list[str] = [
        f"知覚={latest.perception_level}",
        f"テキスト={latest.text_input_level}",
        f"自発={latest.spontaneous_level}",
        f"感情={latest.emotion_level}",
        f"記憶={latest.memory_level}",
        f"動機={latest.motivation_level}",
        f"目標={latest.goal_level}",
        f"責任={latest.responsibility_level}",
        f"集中度={latest.concentration_level}",
    ]

    variation = get_latest_variation(state)
    if variation is not None and variation.comparison_count > 0:
        if variation.concentration_magnitude > 0.01:
            direction_str = "集中方向" if variation.concentration_direction > 0 else "分散方向"
            parts.append(f"変動={direction_str}")

    return " ".join(parts)


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: AttentionDistributionState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> AttentionDistributionState:
    """永続化用の辞書から状態を復元する。"""
    return AttentionDistributionState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_attention_distribution_state() -> AttentionDistributionState:
    """初期状態のファクトリ関数。"""
    return AttentionDistributionState()


def create_attention_distribution_config(
    max_snapshot_history: int = 30,
    variation_comparison_count: int = 5,
) -> AttentionDistributionConfig:
    """設定のファクトリ関数。"""
    return AttentionDistributionConfig(
        max_snapshot_history=max_snapshot_history,
        variation_comparison_count=variation_comparison_count,
    )
