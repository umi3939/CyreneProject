"""
psyche/reaction_mood_update.py - Mood Autonomy: State-Dependent Mood Update

SD-4のムード自律化に関わる全要素。

3段パイプライン: 多入力源からの目標値導出→追従速度導出→更新
入力源: 感情ベクトル/ドライブ/恐怖指数/目的階層存在/責任影響/時間認知
各入力源に帯域上限あり（感情>ドライブ>目的>恐怖）
valence/arousal追従速度の独立導出
安全弁6種: 入力源別帯域制限/追従速度帯域制限/変動量上限/入力不在中立化/非蓄積/独立性

分割元: reaction.py（等価変換・ロジック変更なし）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import coefficient_registry

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
