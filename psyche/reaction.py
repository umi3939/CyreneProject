"""
psyche/reaction.py - Emotion & Drive Reaction Model

Updates PsycheState based on external stimuli (Percept) and time passage.
Handles emotion updates, decay, drive changes, mood update, and **responsibility influence**.

責任（Responsibility）の影響:
- anxiety_baseline: 不安・恐怖系感情のベースラインを上昇
- fear_amplification: 喪失への恐れを増幅

感情振れ幅拡張（Emotion Amplitude Expansion）:
- amplitude_modifier: 感情の変化量（delta）を増減させる
- 方向（符号）は変えず、変化量のみをスケール
- 一時的であり、時間経過で基準値に戻る

ドライブ動態の状態依存化（Drive Dynamics State-Dependent）:
- ドライブの変化量を「固定定数」から「現在の内部状態に依存する導出値」に変更
- 8断面（感情-ドライブ連動/ドライブ間相互作用/目的階層存在/時間経過/覚醒-ドライブ
         /行動結果偏り/内部矛盾有無/結果多様性帰還）
- 安全弁6種: 断面別上限/合成後上限/有効範囲クランプ/相互作用帯域制限/入力不在中立化/非蓄積

ムードの自律化（Mood Autonomy）:
- ムード更新を固定alpha=0.1の指数移動平均から、状態依存的な多入力源更新に変更
- 3段パイプライン: 多入力源からの目標値導出→追従速度導出→更新
- 入力源: 感情ベクトル/ドライブ/恐怖指数/目的階層存在/責任影響/時間認知
- 各入力源に帯域上限あり（感情>ドライブ>目的>恐怖）
- valence/arousal追従速度の独立導出
- 安全弁6種: 入力源別帯域制限/追従速度帯域制限/変動量上限/入力不在中立化/非蓄積/独立性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .fear import fear_drive_boost, fear_emotion_boost
from .responsibility import ResponsibilityInfluence
from .state import DriveVector, EmotionVector, Mood, Percept, PsycheState
from .emotion_amplitude import apply_amplitude_to_delta
from . import coefficient_registry

# Mapping from percept emotion labels to emotion vector fields
_EMOTION_MAP: dict[str, str] = {
    "happy": "joy",
    "sad": "sorrow",
    "angry": "anger",
    "surprised": "surprise",
    "scared": "fear",
    "loving": "love",
    "teasing": "fun",
    "neutral": "",
}

# How strongly a percept's valence affects each emotion (from coefficient registry)
_emo_coeffs = coefficient_registry.get("emotion_processing")
_VALENCE_POSITIVE = _emo_coeffs["valence_positive"]
_VALENCE_NEGATIVE = _emo_coeffs["valence_negative"]

DECAY_RATE = _emo_coeffs["decay_rate"]  # Per-second exponential decay factor


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Drive Dynamics: State-Dependent Configuration
# =============================================================================

# 連動定義テーブル: 各断面×各ドライブ軸の帯域（変動量の絶対値上限）
# 帯域内で実際にどの値を取るかは入力依存で都度異なる
# 価値方向性の影響上限 (max_bias_strength=0.15) と同水準以下

# 断面別の帯域上限 (per-section per-axis absolute max contribution)
# Values loaded from coefficient registry (identical to previous hardcoded values)
_drive_coeffs = coefficient_registry.get("drive_dynamics")
_SECTION_BAND: dict[str, dict[str, float]] = _drive_coeffs["section_band"]

# 合成後の1ティックあたり総変動量の上限 (既存固定加減算の最大値 ~0.15 と同水準)
_TOTAL_CHANGE_LIMIT: float = _drive_coeffs["total_change_limit"]

# 安全弁1: 断面別寄与量の上限 = 上の _SECTION_BAND
# 安全弁2: 合成後総変動量の上限 = _TOTAL_CHANGE_LIMIT
# 安全弁3: ドライブ値の有効範囲クランプ = 0.0-1.0
# 安全弁4: 相互作用帯域制限 = drive_interaction の帯域 < 直接入力帯域
# 安全弁5: 入力不在時の中立化 = 各断面で参照不能時は寄与ゼロ
# 安全弁6: 導出結果の非蓄積 = 純粋関数、状態なし


@dataclass
class DriveContextInputs:
    """ドライブ変動係数導出のための入力コンテキスト。

    各フィールドはOptionalであり、利用不能な場合Noneとする。
    Noneの場合、その断面の寄与は中立値（ゼロ）として扱われる（安全弁5）。
    """
    # 感情ベクトル (7次元)
    emotions: Optional[dict[str, float]] = None
    # ムード (valence, arousal)
    mood_valence: Optional[float] = None
    mood_arousal: Optional[float] = None
    # 他のドライブ値 (同一ベクトル内)
    drives: Optional[dict[str, float]] = None
    # 目的階層の存在情報
    has_transient_goal: bool = False
    persistent_commitment_count: int = 0
    has_scoped_goal: bool = False
    # 時間認知: 入力間隔の段階値 (利用可能な場合)
    time_density_label: Optional[str] = None  # "dense", "normal", "sparse" etc.
    delta_time: float = 1.0
    # 知覚入力
    percept_intent: Optional[str] = None
    percept_emotion: Optional[str] = None
    percept_valence: Optional[float] = None
    # 恐怖指数
    fear_level: float = 0.0
    # 行動多様性記述: 結果断面キー種類数の段階値 (READ-ONLY参照)
    behavioral_diversity_stage_value: Optional[str] = None
    # 内部矛盾並置記述: 直前サイクルの矛盾対検出件数 (READ-ONLY参照)
    contradiction_count: Optional[int] = None
    # 行動多様性記述: 3つの段階値 (結果多様性帰還経路用, READ-ONLY参照)
    result_diversity_section_key_level: Optional[str] = None
    result_diversity_selection_label_level: Optional[str] = None
    result_diversity_candidate_variance_level: Optional[str] = None
    # Phase 26-EXP 帯域拡大: 合成後総変動量の上限に対する一時的乗数
    # None の場合は既存固定値(_TOTAL_CHANGE_LIMIT)を使用
    drive_total_limit_multiplier: Optional[float] = None


def _compute_emotion_drive_coupling(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面1: 感情-ドライブ連動。

    感情ベクトルの各値から各ドライブ軸への影響量を導出する。
    特定の感情が特定のドライブを「必ず」増減させるマッピングは持たない。
    影響の方向と大きさは、ムードの正負・覚醒度・他のドライブ値によって都度異なる。
    """
    band = _SECTION_BAND["emotion_drive_coupling"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    if ctx.emotions is None:
        return result

    emo = ctx.emotions
    valence = ctx.mood_valence if ctx.mood_valence is not None else 0.0
    arousal = ctx.mood_arousal if ctx.mood_arousal is not None else 0.3

    # 正の感情群と負の感情群の比率を算出
    positive_sum = emo.get("joy", 0.0) + emo.get("love", 0.0) + emo.get("fun", 0.0)
    negative_sum = emo.get("sorrow", 0.0) + emo.get("anger", 0.0) + emo.get("fear", 0.0)
    surprise_val = emo.get("surprise", 0.0)

    # social: 正の感情は社会性回復を助け、負の感情は社会性を消耗させる
    # ただしムードの正負で方向が変動する
    social_raw = (positive_sum - negative_sum * 0.5) * 0.1
    # ムードが負の場合、社会的ドライブへの回復が鈍る
    if valence < 0:
        social_raw *= (1.0 + valence * 0.5)  # valence=-1 -> 0.5倍
    result["social"] = max(-band["social"], min(band["social"], social_raw))

    # curiosity: 驚きと好奇心の連動。悲しみは好奇心を抑制
    # A-1: joyからも微弱な正の寄与を追加（回復帯域拡大）
    curiosity_raw = (surprise_val * 0.15 + emo.get("fun", 0.0) * 0.08
                     + emo.get("joy", 0.0) * 0.04
                     - emo.get("sorrow", 0.0) * 0.06)
    # 覚醒度が高いと好奇心への影響が増幅
    curiosity_raw *= (0.7 + arousal * 0.6)
    result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], curiosity_raw))

    # expression: 感情の最大値が高いほど表出ドライブに影響
    # ただし恐怖は表出を抑制する方向に働く
    max_emo = max(emo.values()) if emo else 0.0
    expression_raw = max_emo * 0.08 - emo.get("fear", 0.0) * 0.05
    # ムードが正のとき表出が促進、負のとき抑制
    expression_raw *= (0.8 + valence * 0.4)
    result["expression"] = max(-band["expression"], min(band["expression"], expression_raw))

    return result


def _compute_drive_interaction(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面2: ドライブ間相互作用。

    あるドライブ軸の現在値が、他のドライブ軸の変動速度に影響する。
    固定的な相互作用行列を持たない。他の状態断面にも依存する。
    帯域は直接入力由来の断面より狭い（安全弁4）。
    """
    band = _SECTION_BAND["drive_interaction"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    if ctx.drives is None:
        return result

    drv = ctx.drives
    social = drv.get("social", 0.5)
    curiosity = drv.get("curiosity", 0.5)
    expression = drv.get("expression", 0.5)
    valence = ctx.mood_valence if ctx.mood_valence is not None else 0.0

    # 高い好奇心は社会性をわずかに引き上げる（探索→交流の誘因）
    # ただしムードが負のときこの連動は弱まる
    mood_factor = max(0.3, 1.0 + valence * 0.3)
    social_from_curiosity = (curiosity - 0.5) * 0.04 * mood_factor
    result["social"] = max(-band["social"], min(band["social"], social_from_curiosity))

    # 高い表出ドライブは好奇心をわずかに消耗させる（表出に帯域を使う）
    curiosity_from_expression = -(expression - 0.5) * 0.03
    result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], curiosity_from_expression))

    # 高い社会性ドライブは表出を促進（話したい→表現したい）
    expression_from_social = (social - 0.5) * 0.04 * mood_factor
    result["expression"] = max(-band["expression"], min(band["expression"], expression_from_social))

    return result


def _compute_goal_hierarchy(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面3: 目的階層存在。

    持続的取り組みや一時的目的が存在する場合、
    関連するドライブ軸の回復速度が変化する。
    目的が存在しない場合と存在する場合で変動パターンが構造的に異なる。
    """
    band = _SECTION_BAND["goal_hierarchy"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    # 目的の存在数
    goal_presence = 0
    if ctx.has_transient_goal:
        goal_presence += 1
    goal_presence += ctx.persistent_commitment_count
    if ctx.has_scoped_goal:
        goal_presence += 1

    if goal_presence == 0:
        # 目的不在: ドライブの自然回復がわずかに鈍化（方向性の不在）
        result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], -0.01))
        result["expression"] = max(-band["expression"], min(band["expression"], -0.01))
        return result

    # 目的が存在する場合: 好奇心と表出の回復が促進
    # 存在数に応じて（上限あり）
    goal_factor = min(goal_presence, 3) / 3.0  # 0.33 ~ 1.0
    curiosity_boost = goal_factor * 0.04
    expression_boost = goal_factor * 0.03

    # 持続的取り組みがある場合、社会性も微増（取り組み対象との交流動機）
    if ctx.persistent_commitment_count > 0:
        result["social"] = max(-band["social"], min(band["social"], 0.02))

    result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], curiosity_boost))
    result["expression"] = max(-band["expression"], min(band["expression"], expression_boost))

    return result


def _compute_time_passage(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面4: 時間経過。

    入力間隔の長短によってドライブの自然回復・減衰のパターンが非線形に変化する。
    長時間の無入力では変動パターンが単調な線形回復とは異なる形状をとりうる。
    知覚入力の意図も参照する（既存の反応処理と同様）。
    """
    band = _SECTION_BAND["time_passage"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    dt = ctx.delta_time

    # 時間密度の影響: sparse（長間隔）ではドライブの回復パターンが異なる
    density = ctx.time_density_label
    if density in ("sparse", "somewhat_sparse"):
        # 長間隔: 社会性ドライブが非線形に上昇（孤独感の加速）
        # sqrt(dt) を使って線形より鈍い上昇にする
        social_raw = 0.03 * (dt ** 0.6)
        # 好奇心はゆっくり回復
        curiosity_raw = 0.015 * (dt ** 0.5)
        # 表出は減衰方向
        expression_raw = -0.01 * dt
    elif density in ("dense", "somewhat_dense"):
        # 短間隔: 社会性は急速に満足（対話の密度が高い）
        social_raw = 0.01 * dt - 0.12  # 対話があれば減少
        curiosity_raw = 0.005 * dt
        expression_raw = 0.01 * dt
    else:
        # 通常間隔 or 情報なし: 既存の固定値に近い動作
        social_raw = 0.02 * dt - 0.12  # 時間経過で上昇、対話で減少
        curiosity_raw = 0.01 * dt
        expression_raw = 0.0

    # 知覚入力の意図による修正
    if ctx.percept_intent in ("sharing", "question"):
        curiosity_raw -= 0.04  # A-2: 好奇心充足量を緩和（-0.08→-0.04）
    elif ctx.percept_intent == "expression":
        curiosity_raw += 0.01  # A-3: 他者の表現入力が好奇心を微小に刺激

    # 感情の最大値による表出ドライブへの寄与
    if ctx.emotions:
        max_emo = max(ctx.emotions.values())
        expression_raw += max_emo * 0.04

    # 表出ドライブの時間減衰
    if ctx.drives:
        current_expr = ctx.drives.get("expression", 0.5)
        expression_decay = current_expr * (1.0 - 0.98 ** dt)
        expression_raw -= expression_decay

    result["social"] = max(-band["social"], min(band["social"], social_raw))
    result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], curiosity_raw))
    result["expression"] = max(-band["expression"], min(band["expression"], expression_raw))

    return result


def _compute_arousal_drive(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面5: 覚醒-ドライブ。

    ムードの覚醒度がドライブ変動全体のスケールに影響する。
    高覚醒時と低覚醒時で変動の振幅が異なる。
    """
    band = _SECTION_BAND["arousal_drive"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    arousal = ctx.mood_arousal if ctx.mood_arousal is not None else 0.3
    fear = ctx.fear_level

    # 覚醒度からスケール係数を導出
    # 高覚醒 (>0.6): 全ドライブの変動が活性化
    # 低覚醒 (<0.3): 変動が鈍化
    # 中間: 中立的影響
    if arousal > 0.6:
        scale = (arousal - 0.6) * 0.5  # 0 ~ 0.2
    elif arousal < 0.3:
        scale = -(0.3 - arousal) * 0.3  # -0.09 ~ 0
    else:
        scale = 0.0

    # 恐怖が高い場合、覚醒の影響を抑制
    if fear > 0.3:
        scale *= max(0.3, 1.0 - fear * 0.5)

    # 覚醒スケールを各軸に適用
    # 社会性: 高覚醒で交流意欲増
    result["social"] = max(-band["social"], min(band["social"], scale * 0.8))
    # 好奇心: 高覚醒で探索意欲増
    result["curiosity"] = max(-band["curiosity"], min(band["curiosity"], scale * 1.0))
    # 表出: 高覚醒で表現意欲増
    result["expression"] = max(-band["expression"], min(band["expression"], scale * 0.9))

    return result


def _compute_behavioral_diversity(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面6: 行動結果の偏り度合い。

    行動多様性記述構造の結果断面キー種類数の段階値から、
    ドライブの変動帯域への微弱な寄与を導出する。
    種類数が少ない段階ほど微弱な正の寄与、多い段階ほど中立（ゼロ寄与）に近づく。
    寄与は全ドライブ軸に等しい帯域で配分され、特定軸への選択的影響を持たない。
    入力不在時はゼロ寄与（安全弁5）。
    純粋関数。状態なし（安全弁6）。
    """
    band = _SECTION_BAND["behavioral_diversity"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    if ctx.behavioral_diversity_stage_value is None:
        return result

    # 段階値から数値スケールへの変換
    # 種類数が少ない段階ほど大きな正の寄与、多い段階ほど中立に近づく
    stage = ctx.behavioral_diversity_stage_value
    stage_scale_map = {
        "level_0": 1.0,         # 0種類: 最大寄与
        "level_1_5": 0.6,       # 1-5種類
        "level_6_10": 0.3,      # 6-10種類
        "level_11_15": 0.1,     # 11-15種類
        "level_16_plus": 0.0,   # 16種類以上: 中立
    }
    base_scale = stage_scale_map.get(stage, 0.0)

    if base_scale == 0.0:
        return result

    # ムードの正負で寄与の方向が変動する（固定マッピング防止）
    valence = ctx.mood_valence if ctx.mood_valence is not None else 0.0
    # valence > 0: 正の寄与が強まる
    # valence < 0: 寄与が抑制される
    direction = 0.7 + valence * 0.3  # 0.4 ~ 1.0 range

    raw = base_scale * 0.025 * direction

    for axis in result:
        result[axis] = max(-band[axis], min(band[axis], raw))

    return result


def _compute_internal_contradiction(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面7: 内部矛盾の検出有無。

    内部矛盾並置記述構造の直前サイクルの矛盾対検出件数から、
    ドライブの変動帯域への微弱な寄与を導出する。
    件数がゼロの場合は中立（ゼロ寄与）。
    件数が存在する場合、寄与の方向は覚醒度やムードの正負によって都度異なる
    （「矛盾がある→特定の方向」という固定マッピングを持たない）。
    矛盾の「質」ではなく「検出件数」のみに依存する。
    入力不在時はゼロ寄与（安全弁5）。
    純粋関数。状態なし（安全弁6）。
    """
    band = _SECTION_BAND["internal_contradiction"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    if ctx.contradiction_count is None or ctx.contradiction_count == 0:
        return result

    # 件数のスケール変換（上限あり）
    count = min(ctx.contradiction_count, 6)
    magnitude = count / 6.0  # 0.17 ~ 1.0

    # 寄与の方向は覚醒度とムードの正負で都度異なる
    arousal = ctx.mood_arousal if ctx.mood_arousal is not None else 0.3
    valence = ctx.mood_valence if ctx.mood_valence is not None else 0.0

    # 覚醒度が高く valence が正: 正方向の微弱寄与
    # 覚醒度が低く valence が負: 負方向の微弱寄与
    # 中間: 小さい寄与
    direction = (arousal - 0.5) * 0.4 + valence * 0.3  # -0.5 ~ +0.5 range

    raw = magnitude * 0.03 * direction

    for axis in result:
        result[axis] = max(-band[axis], min(band[axis], raw))

    return result


def _compute_result_diversity_return(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """断面8: 行動結果の蓄積多様性からドライブ帯域への微弱反映。

    行動多様性記述構造の3つの段階値（結果断面キー種類数/選択ラベル種類数/
    候補群サイズ分散度）をREAD-ONLYで読み取り、ドライブ帯域への微弱な寄与を導出する。
    3つの段階値を等価に平均化し、全ドライブ軸に等量配分する。
    入力不在時はゼロ寄与（安全弁5）。
    純粋関数。状態なし（安全弁6）。
    """
    band = _SECTION_BAND["result_diversity_return"]
    result = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}

    section_key = ctx.result_diversity_section_key_level
    selection_label = ctx.result_diversity_selection_label_level
    candidate_variance = ctx.result_diversity_candidate_variance_level

    # 3つ全てNoneの場合はゼロ寄与（安全弁: 入力不在時の中立化）
    if section_key is None and selection_label is None and candidate_variance is None:
        return result

    # TypeCountLevel段階値→数値スケール変換（5段階、等間隔正規化）
    # 最低段階=0.0、最高段階=1.0
    _type_count_scale: dict[str, float] = {
        "level_0": 0.0,
        "level_1_5": 0.25,
        "level_6_10": 0.5,
        "level_11_15": 0.75,
        "level_16_plus": 1.0,
    }

    # DispersionLevel段階値→数値スケール変換（5段階、等間隔正規化）
    _dispersion_scale: dict[str, float] = {
        "empty": 0.0,
        "uniform": 0.25,
        "low": 0.5,
        "moderate": 0.75,
        "high": 1.0,
    }

    # 各段階値をスケール値に変換（None→0.0として扱い、有効入力のみ平均に含める）
    scale_values: list[float] = []

    if section_key is not None:
        scale_values.append(_type_count_scale.get(section_key, 0.0))

    if selection_label is not None:
        scale_values.append(_type_count_scale.get(selection_label, 0.0))

    if candidate_variance is not None:
        scale_values.append(_dispersion_scale.get(candidate_variance, 0.0))

    if not scale_values:
        return result

    # 等価平均化: 特定の断面が他を支配することを防ぐ
    avg_scale = sum(scale_values) / len(scale_values)

    # 帯域変動量の導出: 平均スケール値を各軸に等量配分
    raw = avg_scale * 0.03  # 帯域上限0.03の範囲内

    for axis in result:
        result[axis] = max(-band[axis], min(band[axis], raw))

    return result


def compute_state_dependent_drive_changes(
    ctx: DriveContextInputs,
) -> dict[str, float]:
    """8断面の変動係数を合成し、各ドライブ軸の最終変動量を算出する。

    純粋関数。蓄積なし（安全弁6）。

    合成は加算的であり、各断面の寄与が等価に扱われる。
    合成結果は上限でクランプされる（安全弁2）。

    Returns:
        dict with keys "social", "curiosity", "expression" -> float delta values
    """
    sections = [
        _compute_emotion_drive_coupling(ctx),
        _compute_drive_interaction(ctx),
        _compute_goal_hierarchy(ctx),
        _compute_time_passage(ctx),
        _compute_arousal_drive(ctx),
        _compute_behavioral_diversity(ctx),
        _compute_internal_contradiction(ctx),
        _compute_result_diversity_return(ctx),
    ]

    # 加算合成
    total = {"social": 0.0, "curiosity": 0.0, "expression": 0.0}
    for section in sections:
        for axis in total:
            total[axis] += section.get(axis, 0.0)

    # 安全弁2: 合成後総変動量の上限
    # Phase 26-EXP 帯域拡大: 乗数が指定されている場合は一時的に上限を拡大
    effective_limit = _TOTAL_CHANGE_LIMIT
    if ctx.drive_total_limit_multiplier is not None:
        effective_limit = _TOTAL_CHANGE_LIMIT * ctx.drive_total_limit_multiplier
    for axis in total:
        total[axis] = max(-effective_limit, min(effective_limit, total[axis]))

    return total


# =============================================================================
# Mood Autonomy: State-Dependent Mood Update
# =============================================================================

# 入力源別帯域上限 (valence/arousalへの寄与の絶対値上限)
# 価値方向性の max_bias_strength=0.15 と同水準以下
# Values loaded from coefficient registry
_mood_coeffs = coefficient_registry.get("mood_autonomy")
_MOOD_BAND: dict[str, dict[str, float]] = _mood_coeffs["mood_band"]

# 追従速度の帯域制限 (安全弁2)
_TRACKING_SPEED_MIN: float = _mood_coeffs["tracking_speed_min"]   # 完全停滞を防ぐ下限
_TRACKING_SPEED_MAX: float = _mood_coeffs["tracking_speed_max"]   # 即時追従を防ぐ上限

# 1ティックあたりの変動量上限 (安全弁3)
_MOOD_DELTA_LIMIT: float = _mood_coeffs["mood_delta_limit"]


@dataclass
class MoodContextInputs:
    """ムード更新のための入力コンテキスト。

    各フィールドはOptionalであり、省略された場合はその入力源からの
    寄与がゼロとして扱われる（安全弁4: 入力不在時の中立化）。
    """
    # 感情ベクトル (7次元) — 主要入力源
    emotions: Optional[dict[str, float]] = None
    # ドライブベクトル (3次元)
    drives: Optional[dict[str, float]] = None
    # 現在のムード値
    current_valence: float = 0.0
    current_arousal: float = 0.3
    # 恐怖指数
    fear_level: float = 0.0
    # 目的階層の存在情報
    has_transient_goal: bool = False
    persistent_commitment_count: int = 0
    has_scoped_goal: bool = False
    # 責任影響 (利用可能な場合)
    responsibility_anxiety: float = 0.0
    # 時間認知の段階値 (利用可能な場合)
    time_density_label: Optional[str] = None
    # 経過時間
    delta_time: float = 1.0
    # 感情帰還方向連続性由来の追従速度変調量 (valence, arousal)
    emotion_return_tracking_speed_modulation_valence: Optional[float] = None
    emotion_return_tracking_speed_modulation_arousal: Optional[float] = None


def _derive_mood_targets(ctx: MoodContextInputs) -> tuple[float, float]:
    """段階1: 多入力源からvalence/arousalの目標値を導出する。

    各入力源からの寄与は帯域制限された上で加算される。
    純粋関数（安全弁5: 非蓄積）。

    Returns:
        (target_valence, target_arousal)
    """
    target_valence = 0.0
    target_arousal = 0.0

    # --- 感情由来 ---
    if ctx.emotions is not None:
        emo = ctx.emotions
        positive_sum = emo.get("joy", 0.0) + emo.get("love", 0.0) + emo.get("fun", 0.0)
        negative_sum = emo.get("sorrow", 0.0) + emo.get("anger", 0.0) + emo.get("fear", 0.0)
        # valence: 正-負の差を正規化 (概ね -1..1)
        emo_valence = (positive_sum - negative_sum) / 3.0
        # arousal: 感情の最大値
        emo_arousal = max(emo.values()) if emo else 0.0

        band_v = _MOOD_BAND["emotion"]["valence"]
        band_a = _MOOD_BAND["emotion"]["arousal"]
        target_valence += max(-band_v, min(band_v, emo_valence))
        target_arousal += max(-band_a, min(band_a, emo_arousal))

    # --- ドライブ由来 ---
    if ctx.drives is not None:
        drv = ctx.drives
        social = drv.get("social", 0.5)
        curiosity = drv.get("curiosity", 0.5)
        expression = drv.get("expression", 0.5)

        # ドライブの充足度 (中央値0.5からの距離)
        # 高い社会性+好奇心は微かにvalenceを上げる傾向
        # 表出ドライブが高い場合はarousalに微弱な寄与
        drv_valence = ((social - 0.5) * 0.06 + (curiosity - 0.5) * 0.04)
        drv_arousal = (expression - 0.5) * 0.05 + (curiosity - 0.5) * 0.03

        band_v = _MOOD_BAND["drive"]["valence"]
        band_a = _MOOD_BAND["drive"]["arousal"]
        target_valence += max(-band_v, min(band_v, drv_valence))
        target_arousal += max(-band_a, min(band_a, drv_arousal))

    # --- 目的階層由来 ---
    goal_presence = 0
    if ctx.has_transient_goal:
        goal_presence += 1
    goal_presence += ctx.persistent_commitment_count
    if ctx.has_scoped_goal:
        goal_presence += 1

    if goal_presence > 0:
        # 目的の存在は微弱にvalenceを上げ、arousalをわずかに上げる
        goal_factor = min(goal_presence, 3) / 3.0
        goal_valence = goal_factor * 0.025
        goal_arousal = goal_factor * 0.015

        band_v = _MOOD_BAND["goal"]["valence"]
        band_a = _MOOD_BAND["goal"]["arousal"]
        target_valence += max(-band_v, min(band_v, goal_valence))
        target_arousal += max(-band_a, min(band_a, goal_arousal))

    # --- 恐怖由来 (arousalのみ) ---
    if ctx.fear_level > 0.0:
        # 恐怖指数はarousalを上げる方向にのみ寄与
        fear_arousal = ctx.fear_level * 0.08

        band_a = _MOOD_BAND["fear"]["arousal"]
        target_arousal += max(0.0, min(band_a, fear_arousal))

    return target_valence, target_arousal


def _derive_tracking_speeds(ctx: MoodContextInputs) -> tuple[float, float]:
    """段階2: valence/arousalの追従速度を独立に導出する。

    追従速度は状態依存で変化し、帯域制限される。
    純粋関数（安全弁5: 非蓄積）。

    Returns:
        (valence_speed, arousal_speed) — 各々 [_TRACKING_SPEED_MIN, _TRACKING_SPEED_MAX] 内
    """
    arousal = ctx.current_arousal
    fear = ctx.fear_level

    # --- valence追従速度 ---
    # 基本値: 0.10 (既存のalpha=0.1と同水準)
    v_speed = 0.10

    # 覚醒度が高いほどvalence追従が速い傾向
    if arousal > 0.5:
        v_speed += (arousal - 0.5) * 0.15  # 0.5→1.0で +0.075
    elif arousal < 0.3:
        v_speed -= (0.3 - arousal) * 0.10  # 0.3→0.0で -0.03

    # ドライブ状態による修正
    if ctx.drives is not None:
        # 表出ドライブが高い場合、valence追従がやや速い
        expression = ctx.drives.get("expression", 0.5)
        v_speed += (expression - 0.5) * 0.04

    # 時間経過の影響
    if ctx.time_density_label in ("sparse", "somewhat_sparse"):
        # 長間隔ではvalence追従が遅くなる傾向
        v_speed *= 0.8
    elif ctx.time_density_label in ("dense", "somewhat_dense"):
        # 短間隔ではvalence追従がやや速い
        v_speed *= 1.1

    # --- arousal追従速度 ---
    # 基本値: 0.10
    a_speed = 0.10

    # 恐怖指数が高いほどarousal追従が速い傾向
    if fear > 0.2:
        a_speed += (fear - 0.2) * 0.12  # 0.2→1.0で +0.096

    # 感情の変動幅による修正
    if ctx.emotions is not None:
        emo_vals = list(ctx.emotions.values())
        if emo_vals:
            emo_range = max(emo_vals) - min(emo_vals)
            # 感情変動が大きいとarousal追従が速い
            a_speed += emo_range * 0.06

    # 時間経過の影響
    if ctx.time_density_label in ("sparse", "somewhat_sparse"):
        a_speed *= 0.85
    elif ctx.time_density_label in ("dense", "somewhat_dense"):
        a_speed *= 1.1

    # 感情帰還方向連続性由来の追従速度変調（増加方向のみ）
    if ctx.emotion_return_tracking_speed_modulation_valence is not None:
        v_speed += max(0.0, ctx.emotion_return_tracking_speed_modulation_valence)
    if ctx.emotion_return_tracking_speed_modulation_arousal is not None:
        a_speed += max(0.0, ctx.emotion_return_tracking_speed_modulation_arousal)

    # 帯域制限 (安全弁2)
    v_speed = max(_TRACKING_SPEED_MIN, min(_TRACKING_SPEED_MAX, v_speed))
    a_speed = max(_TRACKING_SPEED_MIN, min(_TRACKING_SPEED_MAX, a_speed))

    return v_speed, a_speed


def compute_autonomous_mood(ctx: MoodContextInputs) -> tuple[float, float]:
    """3段パイプラインでムード更新を行う。

    段階1: 多入力源からの目標値導出
    段階2: 状態依存的な追従速度導出
    段階3: valence/arousalの独立更新

    純粋関数。独自状態の蓄積なし（安全弁5）。
    valenceとarousalは独立に更新される（安全弁6）。

    Returns:
        (new_valence, new_arousal)
    """
    # 段階1: 目標値導出
    target_valence, target_arousal = _derive_mood_targets(ctx)

    # 段階2: 追従速度導出
    v_speed, a_speed = _derive_tracking_speeds(ctx)

    # 段階3: 独立更新
    # valence更新
    v_delta = v_speed * (target_valence - ctx.current_valence)
    # 安全弁3: 変動量上限
    v_delta = max(-_MOOD_DELTA_LIMIT, min(_MOOD_DELTA_LIMIT, v_delta))
    new_valence = ctx.current_valence + v_delta

    # arousal更新
    a_delta = a_speed * (target_arousal - ctx.current_arousal)
    # 安全弁3: 変動量上限
    a_delta = max(-_MOOD_DELTA_LIMIT, min(_MOOD_DELTA_LIMIT, a_delta))
    new_arousal = ctx.current_arousal + a_delta

    return new_valence, new_arousal


def react(
    percept: Percept,
    state: PsycheState,
    delta_time: float = 1.0,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    amplitude_modifier: float = 1.0,
    drive_context: Optional[DriveContextInputs] = None,
    mood_context: Optional[MoodContextInputs] = None,
) -> PsycheState:
    """
    Produce a new PsycheState by reacting to a Percept over delta_time seconds.

    1. Apply emotion stimulus from the Percept (scaled by amplitude_modifier).
    2. Decay all emotions toward zero over time.
    3. Apply responsibility-driven anxiety baseline.
    4. Update drives (social, curiosity, expression) via state-dependent dynamics.
    5. Update mood via autonomous multi-input-source pipeline.

    Args:
        percept: Interpreted stimulus.
        state: Current psychological state (not mutated).
        delta_time: Seconds elapsed since last update.
        responsibility_influence: Optional responsibility influence on emotions.
        amplitude_modifier: Scales emotion change amounts (1.0 = no change).
                          Does NOT change direction, only magnitude.
        drive_context: Optional context for state-dependent drive dynamics.
                      If None, a default context is constructed from available state.
        mood_context: Optional context for autonomous mood update.
                     If None, a default context is constructed from available state.

    Returns:
        New PsycheState reflecting the reaction.
    """
    emo = state.emotions.as_dict()

    # --- 1. Emotion stimulus from Percept ---
    # Direct emotion mapping (with amplitude scaling)
    target_field = _EMOTION_MAP.get(percept.emotion, "")
    if target_field and target_field in emo:
        base_delta = _emo_coeffs["stimulus_base_delta"]
        scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
        emo[target_field] = _clamp(emo[target_field] + scaled_delta)

    # Valence-based secondary effects (with amplitude scaling)
    v = percept.emotion_valence
    if v > 0:
        for field, weight in _VALENCE_POSITIVE.items():
            base_delta = v * weight
            scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
            emo[field] = _clamp(emo[field] + scaled_delta)
    elif v < 0:
        for field, weight in _VALENCE_NEGATIVE.items():
            base_delta = abs(v) * weight
            scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
            emo[field] = _clamp(emo[field] + scaled_delta)

    # --- 2. Time-based emotion decay ---
    decay = DECAY_RATE ** delta_time
    for k in emo:
        emo[k] = _clamp(emo[k] * decay)

    new_emotions = EmotionVector(**emo)

    # --- 2.5 Fear-driven emotion boost ---
    if state.fear_index is not None:
        new_emotions = fear_emotion_boost(state.fear_index, new_emotions)

    # --- 2.6 Responsibility-driven anxiety baseline ---
    # 責任の重みが、不安・恐怖系感情のベースラインを上昇させる
    if responsibility_influence is not None:
        new_emotions = _apply_responsibility_emotion_influence(
            new_emotions, responsibility_influence
        )

    # --- 3. Drive updates (state-dependent dynamics) ---
    drv = state.drives.model_dump()

    # Construct drive context if not provided (安全弁5: 入力不在時は中立化)
    if drive_context is None:
        drive_context = DriveContextInputs(
            emotions=emo,
            mood_valence=state.mood.valence,
            mood_arousal=state.mood.arousal,
            drives=drv.copy(),
            delta_time=delta_time,
            percept_intent=percept.intent,
            percept_emotion=percept.emotion,
            percept_valence=percept.emotion_valence,
            fear_level=state.fear_level,
        )
    else:
        # Ensure emotions and drives reflect current tick's values
        drive_context.emotions = emo
        drive_context.drives = drv.copy()
        drive_context.delta_time = delta_time
        drive_context.percept_intent = percept.intent
        drive_context.percept_emotion = percept.emotion
        drive_context.percept_valence = percept.emotion_valence
        drive_context.mood_valence = state.mood.valence
        drive_context.mood_arousal = state.mood.arousal
        drive_context.fear_level = state.fear_level

    # Compute state-dependent drive changes (pure function, no accumulation)
    drive_deltas = compute_state_dependent_drive_changes(drive_context)

    # Apply changes with clamping (安全弁3: ドライブ値の有効範囲クランプ)
    for axis in ("social", "curiosity", "expression"):
        drv[axis] = _clamp(drv[axis] + drive_deltas[axis])

    new_drives = DriveVector(**drv)

    # --- 3.5 Fear-driven drive boost ---
    if state.fear_index is not None:
        new_drives = fear_drive_boost(state.fear_index, new_drives)

    # --- 4. Autonomous mood update (multi-input-source pipeline) ---
    # Construct mood context if not provided (安全弁4: 入力不在時は中立化)
    if mood_context is None:
        mood_context = MoodContextInputs(
            emotions=new_emotions.as_dict(),
            drives=new_drives.as_dict(),
            current_valence=state.mood.valence,
            current_arousal=state.mood.arousal,
            fear_level=state.fear_level,
            responsibility_anxiety=(
                responsibility_influence.anxiety_baseline
                if responsibility_influence is not None else 0.0
            ),
            delta_time=delta_time,
        )
    else:
        # Ensure emotions/drives reflect current tick's updated values
        mood_context.emotions = new_emotions.as_dict()
        mood_context.drives = new_drives.as_dict()
        mood_context.current_valence = state.mood.valence
        mood_context.current_arousal = state.mood.arousal
        mood_context.fear_level = state.fear_level
        mood_context.delta_time = delta_time
        if responsibility_influence is not None:
            mood_context.responsibility_anxiety = responsibility_influence.anxiety_baseline

    new_valence, new_arousal = compute_autonomous_mood(mood_context)

    # --- 4.5 Responsibility affects mood ---
    # 責任の重みがムードのvalenceを少し下げる（重荷として）
    if responsibility_influence is not None:
        mood_penalty = responsibility_influence.anxiety_baseline * 0.3
        new_valence = new_valence - mood_penalty

    new_mood = Mood(
        valence=_clamp(new_valence, -1.0, 1.0),
        arousal=_clamp(new_arousal, 0.0, 1.0),
    )

    return PsycheState(
        emotions=new_emotions,
        drives=new_drives,
        mood=new_mood,
        identity=state.identity,
        attachment=state.attachment,
        continuity=state.continuity,
        projection=state.projection,
        fear_index=state.fear_index,
    )


def _apply_responsibility_emotion_influence(
    emotions: EmotionVector,
    influence: ResponsibilityInfluence,
) -> EmotionVector:
    """責任の影響を感情に適用する。

    - anxiety_baseline: 恐怖・悲しみのベースラインを上昇
    - fear_amplification: 恐怖を増幅（喪失への恐れ）

    影響は穏やかで、上限が設けられている。
    """
    emo = emotions.as_dict()

    # 不安ベースライン: 恐怖と悲しみの最低値を引き上げ
    anxiety = influence.anxiety_baseline
    if anxiety > 0:
        emo["fear"] = _clamp(max(emo["fear"], anxiety * 0.5))
        emo["sorrow"] = _clamp(max(emo["sorrow"], anxiety * 0.3))

    # 喪失恐怖の増幅
    fear_amp = influence.fear_amplification
    if fear_amp > 0 and emo["fear"] > 0.1:
        emo["fear"] = _clamp(emo["fear"] + fear_amp * 0.2)

    return EmotionVector(**emo)
