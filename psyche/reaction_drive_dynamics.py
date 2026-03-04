"""
psyche/reaction_drive_dynamics.py - Drive Dynamics: State-Dependent Configuration

SD-3の8断面によるドライブ変動量の状態依存的導出。

8断面（感情-ドライブ連動/ドライブ間相互作用/目的階層存在/時間経過/覚醒-ドライブ
       /行動結果偏り/内部矛盾有無/結果多様性帰還）
安全弁6種: 断面別上限/合成後上限/有効範囲クランプ/相互作用帯域制限/入力不在中立化/非蓄積

分割元: reaction.py（等価変換・ロジック変更なし）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import coefficient_registry

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
