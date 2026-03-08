"""
psyche/decay_rate_modulation.py - 感情蓄積パターンによる感情減衰速度の変調

セッション起動時に、感情基調の持続認知モジュールの蓄積データから
感情変動の振幅的特性を読み取り、感情減衰速度を微弱に変調する。

設計原則 (design_decay_rate_modulation.md 準拠):
- セッション起動時に一度だけ算出し、セッション中は不変
- ティック処理中の感情減衰速度変更は構造的に禁止
- 帯域上限: 基準値に対する上下5%以内
- 基準値への回帰圧力により単方向累積を防止
- 感情基調の持続認知モジュールの蓄積データはREAD-ONLY参照
- 変調結果をenrichmentに直接露出しない
- 感情の「あるべき状態」や「適切な減衰速度」を規範として導入しない

安全弁:
  1. 帯域上限: 基準値 +-5% を超える変調を行わない
  2. 基準値回帰圧力: 偏差に比例した逆方向の力
  3. 起動時限定: セッション中の変更を構造的に禁止
  4. 蓄積データのREAD-ONLY参照: 書き換えない
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 帯域上限: 基準値に対する上下の比率 (+-5%)
_MODULATION_BAND_RATIO: float = 0.05

# 回帰圧力係数: 前回の偏差をこの係数分だけ基準値方向に引き戻す
# 0.0 = 回帰なし, 1.0 = 完全回帰
_REGRESSION_PRESSURE_FACTOR: float = 0.3

# 振幅特性を変調量に変換する係数
# 振幅スコアが 0.0-1.0 の範囲に正規化された後、
# この係数を掛けて帯域比率内の変調量を得る
_AMPLITUDE_TO_MODULATION_SCALE: float = 0.5

# 中立点: この振幅スコア以下は減衰を遅く、以上は速く変調する
_AMPLITUDE_NEUTRAL_POINT: float = 0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_emotion_amplitude_score(
    composition_records: list[dict[str, Any]],
) -> Optional[float]:
    """蓄積された構成記述から感情変動の振幅的特性スコアを算出する。

    各構成記述の感情次元の値列から分散を計算し、
    全次元・全記録の平均分散を正規化してスコア化する。

    Args:
        composition_records: 感情基調の持続認知モジュールの蓄積記録
                            (CompositionRecord.to_dict() 形式のリスト)

    Returns:
        振幅スコア (0.0-1.0)。蓄積データが不十分な場合は None。
    """
    if not composition_records:
        return None

    # 鮮度が active/weakening の記録のみを対象とする
    visible_records = [
        r for r in composition_records
        if r.get("freshness", 0.0) >= 0.6  # ACTIVE or WEAKENING
    ]

    if len(visible_records) < 2:
        return None

    # 各記録の感情次元値列から分散を算出
    all_variances: list[float] = []

    for rec in visible_records:
        emotion_series = rec.get("emotion_series", {})
        if emotion_series:
            for dim_name, values in emotion_series.items():
                if not isinstance(values, list) or len(values) < 3:
                    continue
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                all_variances.append(variance)

        # valence系列の分散も含める
        valence_series = rec.get("valence_series", [])
        if isinstance(valence_series, list) and len(valence_series) >= 3:
            v_mean = sum(valence_series) / len(valence_series)
            v_var = sum((v - v_mean) ** 2 for v in valence_series) / len(valence_series)
            all_variances.append(v_var)

    if not all_variances:
        return None

    # 平均分散を算出
    avg_variance = sum(all_variances) / len(all_variances)

    # 正規化: 分散を 0.0-1.0 のスコアに変換
    # 分散 0.0 → スコア 0.0 (変動なし)
    # 分散 0.05 → スコア ~1.0 (十分な変動)
    # シグモイド様の変換 (飽和特性)
    normalized = _clamp(avg_variance / 0.05, 0.0, 1.0)

    return normalized


def compute_decay_rate_modulation(
    base_decay_rate: float,
    amplitude_score: Optional[float],
    previous_modulated_rate: Optional[float],
) -> float:
    """感情変動振幅スコアから減衰速度の変調後の値を算出する。

    設計書の仕様:
    - 振幅が大きい → 減衰がわずかに速い方向 (decay_rate が低い方向)
    - 振幅が小さい → 減衰がわずかに遅い方向 (decay_rate が高い方向)
    - 帯域は基準値に対する上下5%
    - 基準値への回帰圧力を含む

    注: decay_rate は指数関数的減衰の底 (0 < rate < 1)。
    rate が小さいほど減衰が速い。rate が大きいほど感情が持続する。

    Args:
        base_decay_rate: 係数レジストリの基準値 (e.g., 0.95)
        amplitude_score: 振幅スコア (0.0-1.0)。None の場合は変調なし。
        previous_modulated_rate: 前回セッション終了時の変調後の値。
                                None の場合は回帰圧力なし。

    Returns:
        変調後の減衰速度
    """
    if amplitude_score is None:
        # 蓄積データ不足: 変調なし
        return base_decay_rate

    # 帯域の絶対幅
    band_width = base_decay_rate * _MODULATION_BAND_RATIO

    # 振幅スコアから変調方向と量を決定
    # amplitude_score が中立点より大きい → 減衰を速く → decay_rate を下げる
    # amplitude_score が中立点より小さい → 減衰を遅く → decay_rate を上げる
    deviation_from_neutral = amplitude_score - _AMPLITUDE_NEUTRAL_POINT
    # -0.5 to +0.5 の範囲

    # 変調量: 偏差 * スケール * 帯域幅
    raw_modulation = -deviation_from_neutral * _AMPLITUDE_TO_MODULATION_SCALE * 2.0 * band_width
    # 偏差が+0.5 → modulation = -band_width * SCALE (減衰速く)
    # 偏差が-0.5 → modulation = +band_width * SCALE (減衰遅く)

    # 基準値への回帰圧力
    regression_adjustment = 0.0
    if previous_modulated_rate is not None:
        previous_deviation = previous_modulated_rate - base_decay_rate
        # 前回の偏差を回帰圧力係数分だけ引き戻す
        regression_adjustment = -previous_deviation * _REGRESSION_PRESSURE_FACTOR

    # 最終的な変調量
    total_modulation = raw_modulation + regression_adjustment

    # 安全弁1: 帯域上限の適用
    total_modulation = _clamp(total_modulation, -band_width, band_width)

    modulated_rate = base_decay_rate + total_modulation

    # 二重チェック: 変調後の値が帯域内であることを保証
    min_rate = base_decay_rate - band_width
    max_rate = base_decay_rate + band_width
    modulated_rate = _clamp(modulated_rate, min_rate, max_rate)

    return modulated_rate


def apply_decay_rate_modulation(
    backdrop_state_dict: dict[str, Any],
    base_decay_rate: float,
    previous_modulated_rate: Optional[float],
) -> tuple[float, Optional[float]]:
    """セッション起動時の感情減衰速度変調を実行する。

    この関数はセッション起動時に一度だけ呼び出される。
    ティック中の呼び出しは構造的に禁止される（呼び出し元が起動時のみ）。

    安全弁1: 帯域上限 (+-5%)
    安全弁2: 基準値回帰圧力
    安全弁3: 起動時限定
    安全弁4: 蓄積データのREAD-ONLY参照

    Args:
        backdrop_state_dict: 感情基調の持続認知モジュールの状態辞書
                            (BackdropState.to_dict() の結果)
        base_decay_rate: 係数レジストリの基準値
        previous_modulated_rate: 前回セッション終了時の変調後の値

    Returns:
        (modulated_rate, amplitude_score) のタプル。
        modulated_rate: 変調後の減衰速度
        amplitude_score: 算出された振幅スコア (デバッグ用)
    """
    # 蓄積データから構成記述を読み取る (READ-ONLY)
    composition_records = backdrop_state_dict.get("composition_records", [])

    # 振幅スコアの算出
    amplitude_score = compute_emotion_amplitude_score(composition_records)

    # 変調量の算出
    modulated_rate = compute_decay_rate_modulation(
        base_decay_rate=base_decay_rate,
        amplitude_score=amplitude_score,
        previous_modulated_rate=previous_modulated_rate,
    )

    logger.info(
        "Decay rate modulation: base=%.6f, amplitude_score=%s, "
        "previous=%.6f, modulated=%.6f, delta=%.6f",
        base_decay_rate,
        f"{amplitude_score:.4f}" if amplitude_score is not None else "None",
        previous_modulated_rate if previous_modulated_rate is not None else base_decay_rate,
        modulated_rate,
        modulated_rate - base_decay_rate,
    )

    return modulated_rate, amplitude_score
