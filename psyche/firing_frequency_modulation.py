"""
psyche/firing_frequency_modulation.py - 帰還経路発火頻度による係数帯域限界の変調

design_firing_frequency_modulation.md に基づく実装:
  オーケストレータ内部で記録される帰還経路の累計発火回数を永続化し、
  その累計値に基づいて係数レジストリの帯域上限を起動時に段階的に変調する。

  「よく発火する帰還経路はわずかに広い帯域を持つ」ことで、帰還経路自体が
  経験の蓄積によって構造的に変化する。

設計原則:
- ティック処理中に帯域上限を変更しない（起動時限定）
- 帰還経路の発火/非発火の判定に介入しない
- 帯域上限の変調を通じて特定の判断・行動・方向性を誘導しない
- 変調量が絶対上限（基準値の1.5倍）を超えない
- 変調結果をenrichmentに直接露出しない
- 外部ツール（tools/）のデータを参照しない（オーケストレータ内部のカウントのみ使用）

安全弁:
  1. 対数スケーリング: 累計発火回数に対する帯域拡大量が逓減する飽和的変換
  2. 絶対帯域上限: 基準値の1.5倍を超える拡大の構造的禁止
  3. 起動時限定: セッション中の帯域上限変更を構造的に禁止する（呼び出し元が起動時のみ）
  4. 外部ツール非依存: tools/のデータを参照しない
  5. カウンタの加算のみ: 累計カウンタはリセットや減算を構造的に禁止
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# 絶対帯域上限の倍率: 基準値の1.5倍を超えない
_ABSOLUTE_CAP_MULTIPLIER: float = 1.5

# 対数スケーリングの分母: log(1 + count / scale_factor) で飽和速度を制御
# scale_factor が大きいほど変調の立ち上がりが緩やかになる
_LOG_SCALE_FACTOR: float = 50.0

# 最大変調比率: 対数変換後の値の理論上限（log(1+x)は無限大に発散するため、
# 実質的には _ABSOLUTE_CAP_MULTIPLIER - 1.0 = 0.5 で打ち切る）
_MAX_MODULATION_RATIO: float = _ABSOLUTE_CAP_MULTIPLIER - 1.0


# =============================================================================
# Pathway-to-namespace mapping
# =============================================================================

# 帰還経路識別子 -> (namespace, bandwidth_keys) の対応関係
# 各帰還経路がどの係数レジストリ名前空間のどの帯域上限に対応するか

# Pathway A: memory_emotion_return -> per_candidate_max_delta, total_max_delta
_PATHWAY_A = "memory_emotion_return"
# Pathway C: other_hypothesis_emotion_return -> per_candidate_max_delta, total_max_delta
_PATHWAY_C = "other_hypothesis_emotion_return"
# Pathway D: result_diversity_drive_return -> drive_dynamics.section_band.result_diversity_return
_PATHWAY_D = "result_diversity_drive_return"
# Pathway E: emotion_return_tracking_speed -> memory_emotion_return.tracking_speed_modulation_ratio_cap
_PATHWAY_E = "emotion_return_tracking_speed"


# =============================================================================
# Core computation
# =============================================================================


def compute_log_modulation_ratio(cumulative_count: int) -> float:
    """累計発火回数から対数スケーリングによる変調比率を算出する。

    対数変換により:
    - 初期の発火は帯域拡大に大きく寄与する
    - 発火が累積するにつれて追加の寄与は逓減する

    Args:
        cumulative_count: 累計発火回数（非負整数）

    Returns:
        変調比率 (0.0 ~ _MAX_MODULATION_RATIO)。
        基準値に対する拡大比率。例えば0.1なら基準値の10%拡大。
    """
    if cumulative_count <= 0:
        return 0.0

    # 対数スケーリング: log(1 + count / scale_factor)
    # count=0 -> 0.0
    # count=50 -> log(2) = 0.693
    # count=500 -> log(11) = 2.398
    raw = math.log(1.0 + cumulative_count / _LOG_SCALE_FACTOR)

    # 正規化: raw を 0.0 ~ _MAX_MODULATION_RATIO にマッピング
    # log(1 + x) は無限大に発散するため、上限でクリップ
    ratio = min(raw * (_MAX_MODULATION_RATIO / math.log(1.0 + 1000.0 / _LOG_SCALE_FACTOR)), _MAX_MODULATION_RATIO)

    return ratio


def compute_modulated_value(base_value: float, cumulative_count: int) -> float:
    """基準値と累計発火回数から変調後の帯域上限を算出する。

    変調後の値 = base_value * (1.0 + modulation_ratio)

    安全弁:
    - 対数スケーリング（逓減）
    - 絶対上限（base_value * _ABSOLUTE_CAP_MULTIPLIER）

    Args:
        base_value: 係数レジストリの基準帯域上限値
        cumulative_count: 累計発火回数

    Returns:
        変調後の帯域上限値
    """
    ratio = compute_log_modulation_ratio(cumulative_count)
    modulated = base_value * (1.0 + ratio)

    # 安全弁2: 絶対帯域上限
    absolute_cap = base_value * _ABSOLUTE_CAP_MULTIPLIER
    if modulated > absolute_cap:
        modulated = absolute_cap

    return modulated


# =============================================================================
# Application to processor configs
# =============================================================================


def apply_firing_frequency_modulation(
    pathway_fire_counts: dict[str, int],
    memory_emotion_return_config: Any,
    other_hypothesis_emotion_return_config: Any,
    drive_section_band: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """帰還経路発火頻度に基づく帯域変調を適用する。

    セッション起動時に一度だけ呼び出される。
    ティック中の呼び出しは構造的に禁止される（呼び出し元が起動時のみ）。

    各帰還経路の累計発火回数を読み取り、対応する帯域上限に変調を適用する:
    - Pathway A (memory_emotion_return):
        per_candidate_max_delta, total_max_delta を変調
    - Pathway C (other_hypothesis_emotion_return):
        per_candidate_max_delta, total_max_delta を変調
    - Pathway D (result_diversity_drive_return):
        drive_dynamics.section_band.result_diversity_return の各軸を変調
    - Pathway E (emotion_return_tracking_speed):
        memory_emotion_return.tracking_speed_modulation_ratio_cap を変調

    安全弁:
    1. 対数スケーリング: 変調量の逓減
    2. 絶対上限: 基準値の1.5倍
    3. 起動時限定: この関数自体が起動時にのみ呼ばれる
    4. 外部ツール非依存: pathway_fire_counts はオーケストレータ内部データ
    5. カウンタの加算のみ: 本関数はカウンタを変更しない（読み取りのみ）

    Args:
        pathway_fire_counts: 帰還経路別の累計発火回数
            {pathway_id: count}
        memory_emotion_return_config: MemoryEmotionReturnConfig インスタンス
        other_hypothesis_emotion_return_config: OtherHypothesisEmotionReturnConfig インスタンス
        drive_section_band: _SECTION_BAND 辞書（result_diversity_return を含む）

    Returns:
        適用結果の辞書（情報提供用）:
        {
            "pathway_a": {"count": N, "ratio": R, "modulated_fields": {...}},
            "pathway_c": {...},
            "pathway_d": {...},
            "pathway_e": {...},
        }
    """
    if not isinstance(pathway_fire_counts, dict):
        logger.debug("Firing frequency modulation: invalid fire counts, skipping.")
        return {}

    result: dict[str, Any] = {}

    # ── Pathway A: memory_emotion_return ──
    count_a = _safe_get_count(pathway_fire_counts, _PATHWAY_A)
    if count_a > 0 and memory_emotion_return_config is not None:
        try:
            ratio_a = compute_log_modulation_ratio(count_a)
            modulated_fields_a: dict[str, float] = {}

            # per_candidate_max_delta
            base_pcmd = memory_emotion_return_config.per_candidate_max_delta
            new_pcmd = compute_modulated_value(base_pcmd, count_a)
            memory_emotion_return_config.per_candidate_max_delta = new_pcmd
            modulated_fields_a["per_candidate_max_delta"] = new_pcmd

            # total_max_delta
            base_tmd = memory_emotion_return_config.total_max_delta
            new_tmd = compute_modulated_value(base_tmd, count_a)
            memory_emotion_return_config.total_max_delta = new_tmd
            modulated_fields_a["total_max_delta"] = new_tmd

            result["pathway_a"] = {
                "count": count_a,
                "ratio": ratio_a,
                "modulated_fields": modulated_fields_a,
            }

            logger.info(
                "Firing frequency modulation: pathway_a count=%d ratio=%.4f "
                "per_candidate_max_delta=%.4f->%.4f total_max_delta=%.4f->%.4f",
                count_a, ratio_a,
                base_pcmd, new_pcmd,
                base_tmd, new_tmd,
            )
        except Exception as e:
            logger.debug("Firing frequency modulation: pathway_a failed: %s", e)

    # ── Pathway C: other_hypothesis_emotion_return ──
    count_c = _safe_get_count(pathway_fire_counts, _PATHWAY_C)
    if count_c > 0 and other_hypothesis_emotion_return_config is not None:
        try:
            ratio_c = compute_log_modulation_ratio(count_c)
            modulated_fields_c: dict[str, float] = {}

            # per_candidate_max_delta
            base_pcmd_c = other_hypothesis_emotion_return_config.per_candidate_max_delta
            new_pcmd_c = compute_modulated_value(base_pcmd_c, count_c)
            other_hypothesis_emotion_return_config.per_candidate_max_delta = new_pcmd_c
            modulated_fields_c["per_candidate_max_delta"] = new_pcmd_c

            # total_max_delta
            base_tmd_c = other_hypothesis_emotion_return_config.total_max_delta
            new_tmd_c = compute_modulated_value(base_tmd_c, count_c)
            other_hypothesis_emotion_return_config.total_max_delta = new_tmd_c
            modulated_fields_c["total_max_delta"] = new_tmd_c

            result["pathway_c"] = {
                "count": count_c,
                "ratio": ratio_c,
                "modulated_fields": modulated_fields_c,
            }

            logger.info(
                "Firing frequency modulation: pathway_c count=%d ratio=%.4f "
                "per_candidate_max_delta=%.4f->%.4f total_max_delta=%.4f->%.4f",
                count_c, ratio_c,
                base_pcmd_c, new_pcmd_c,
                base_tmd_c, new_tmd_c,
            )
        except Exception as e:
            logger.debug("Firing frequency modulation: pathway_c failed: %s", e)

    # ── Pathway D: result_diversity_drive_return ──
    count_d = _safe_get_count(pathway_fire_counts, _PATHWAY_D)
    if count_d > 0 and drive_section_band is not None:
        try:
            ratio_d = compute_log_modulation_ratio(count_d)
            modulated_fields_d: dict[str, float] = {}

            rdr_band = drive_section_band.get("result_diversity_return")
            if isinstance(rdr_band, dict):
                for axis, base_val in rdr_band.items():
                    if isinstance(base_val, (int, float)):
                        new_val = compute_modulated_value(float(base_val), count_d)
                        rdr_band[axis] = new_val
                        modulated_fields_d[axis] = new_val

            result["pathway_d"] = {
                "count": count_d,
                "ratio": ratio_d,
                "modulated_fields": modulated_fields_d,
            }

            logger.info(
                "Firing frequency modulation: pathway_d count=%d ratio=%.4f "
                "modulated=%s",
                count_d, ratio_d, modulated_fields_d,
            )
        except Exception as e:
            logger.debug("Firing frequency modulation: pathway_d failed: %s", e)

    # ── Pathway E: emotion_return_tracking_speed ──
    count_e = _safe_get_count(pathway_fire_counts, _PATHWAY_E)
    if count_e > 0 and memory_emotion_return_config is not None:
        try:
            ratio_e = compute_log_modulation_ratio(count_e)
            modulated_fields_e: dict[str, float] = {}

            # tracking_speed_modulation_ratio_cap
            base_cap = memory_emotion_return_config.tracking_speed_modulation_ratio_cap
            new_cap = compute_modulated_value(base_cap, count_e)
            memory_emotion_return_config.tracking_speed_modulation_ratio_cap = new_cap
            modulated_fields_e["tracking_speed_modulation_ratio_cap"] = new_cap

            result["pathway_e"] = {
                "count": count_e,
                "ratio": ratio_e,
                "modulated_fields": modulated_fields_e,
            }

            logger.info(
                "Firing frequency modulation: pathway_e count=%d ratio=%.4f "
                "tracking_speed_modulation_ratio_cap=%.4f->%.4f",
                count_e, ratio_e, base_cap, new_cap,
            )
        except Exception as e:
            logger.debug("Firing frequency modulation: pathway_e failed: %s", e)

    total_pathways_modulated = len(result)
    if total_pathways_modulated > 0:
        logger.info(
            "Firing frequency modulation applied: %d pathways modulated.",
            total_pathways_modulated,
        )
    else:
        logger.debug("Firing frequency modulation: no pathways had cumulative counts.")

    return result


# =============================================================================
# Helpers
# =============================================================================


def _safe_get_count(counts: dict[str, int], pathway_id: str) -> int:
    """カウント辞書から安全に累計発火回数を取得する。"""
    val = counts.get(pathway_id, 0)
    if isinstance(val, (int, float)):
        return max(0, int(val))
    return 0


def get_modulation_summary(
    pathway_fire_counts: dict[str, int],
) -> dict[str, dict[str, float]]:
    """各帰還経路の現在の変調比率を読み取り専用で返す（情報提供用）。

    Args:
        pathway_fire_counts: 帰還経路別の累計発火回数

    Returns:
        {pathway_id: {"count": N, "ratio": R}} の辞書
    """
    if not isinstance(pathway_fire_counts, dict):
        return {}

    result: dict[str, dict[str, float]] = {}
    for pid in [_PATHWAY_A, _PATHWAY_C, _PATHWAY_D, _PATHWAY_E]:
        count = _safe_get_count(pathway_fire_counts, pid)
        ratio = compute_log_modulation_ratio(count)
        result[pid] = {"count": float(count), "ratio": ratio}

    return result
