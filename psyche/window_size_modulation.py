"""
psyche/window_size_modulation.py - 記憶想起頻度によるウィンドウサイズの経験依存化

セッション起動時に忘却と想起の均衡記述モジュールの蓄積データから想起頻度を
読み取り、記述層共通ウィンドウサイズを帯域内で変調する。

設計原則 (design_window_size_modulation.md 準拠):
- ティック処理中にウィンドウサイズを変更しない（起動時限定）
- ウィンドウ内のデータの重み付けや優先順位を変更しない
- 変調量が帯域上限(基準値の±20%)を超えない
- 特定の記憶内容や想起結果を優遇・抑制しない
- 変調結果をenrichmentに直接露出しない
- 忘却と想起の均衡記述モジュールの蓄積データ自体を変更しない

安全弁:
  1. 帯域上限 — 基準値に対する上下±20%で変調量を制限
  2. 基準値回帰圧力 — 前回の変調結果から基準値方向への引き戻し
  3. 整数丸め — ウィンドウサイズは整数値、微小変動を吸収
  4. 起動時限定 — セッション中のウィンドウサイズ変更を構造的に禁止
  5. 蓄積データのREAD-ONLY参照 — 忘却想起均衡記述の蓄積データを読み取るのみ
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import coefficient_registry
from .forgetting_recall_balance import ForgettingRecallBalanceState

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# 帯域上限: 基準値に対する上下の比率 (±20%)
_MODULATION_BAND_RATIO: float = 0.20

# 基準値回帰圧力の強度 (0.0=回帰なし, 1.0=完全回帰)
# 0.3: 前回変調値と新規算出値のブレンドで30%分を基準値方向に引き戻す
_REGRESSION_PRESSURE: float = 0.3

# 想起頻度の段階値定義
# 段階値は想起件数の合計に基づく区間分割
# (上限値, 変調方向係数) の順序付きリスト
# 変調方向係数: -1.0=最大縮小, 0.0=変調なし, 1.0=最大拡大
_RECALL_FREQUENCY_STAGES: list[tuple[int, float]] = [
    (0, -0.8),      # 想起なし: 縮小方向
    (3, -0.4),      # 非常に不活発: 縮小方向
    (10, 0.0),      # 普通: 変調なし
    (20, 0.4),      # 活発: 拡大方向
]
# 上限を超えた場合: 非常に活発 → 拡大方向
_RECALL_FREQUENCY_VERY_ACTIVE_FACTOR: float = 0.8

# 対象ウィンドウサイズキー
_WINDOW_SIZE_KEYS: tuple[str, ...] = (
    "window_size_25",
    "window_size_30",
    "window_size_50",
)


# =============================================================================
# Recall frequency extraction (READ-ONLY)
# =============================================================================

def extract_recall_frequency(
    frb_state: ForgettingRecallBalanceState,
) -> int:
    """忘却想起均衡記述の蓄積データから想起頻度の合計件数を読み取る。

    蓄積データ(history)内の全エントリから、外部トリガー型想起と
    自発的想起の候補総件数を読み取り、合計する。

    安全弁5: READ-ONLY参照のみ。蓄積データへの書き込みなし。

    Args:
        frb_state: 忘却想起均衡記述の内部状態 (READ-ONLY)

    Returns:
        想起頻度の合計件数（外部想起 + 自発想起の候補総件数の合算）
    """
    if not frb_state.history:
        return 0

    total_recall = 0
    for entry in frb_state.history:
        total_recall += entry.external_recall.total_count
        total_recall += entry.spontaneous_recall.total_count

    return total_recall


def classify_recall_frequency(total_recall: int) -> float:
    """想起頻度の合計件数を変調方向係数に変換する。

    段階値区間分割による機械的変換。

    Args:
        total_recall: 想起頻度の合計件数

    Returns:
        変調方向係数 (-1.0 ~ 1.0)
    """
    for upper_bound, factor in _RECALL_FREQUENCY_STAGES:
        if total_recall <= upper_bound:
            return factor

    return _RECALL_FREQUENCY_VERY_ACTIVE_FACTOR


# =============================================================================
# Modulation computation
# =============================================================================

def compute_modulation(
    base_value: int,
    direction_factor: float,
    previous_modulated: Optional[int] = None,
) -> int:
    """ウィンドウサイズの変調後の値を算出する。

    安全弁1: 帯域上限 — 基準値の±20%を超えない。
    安全弁2: 基準値回帰圧力 — 前回変調値がある場合、基準値方向に引き戻す。
    安全弁3: 整数丸め — 変調後の値は整数。

    Args:
        base_value: 係数レジストリの基準値（coefficients.json由来）
        direction_factor: 変調方向係数 (-1.0 ~ 1.0)
        previous_modulated: 前回セッションの変調後の値（Noneなら初回）

    Returns:
        変調後のウィンドウサイズ（整数）
    """
    # 帯域の算出
    max_delta = base_value * _MODULATION_BAND_RATIO

    # 変調量の算出（方向係数 * 最大変調量）
    raw_modulation = direction_factor * max_delta

    # 基準値回帰圧力の適用
    if previous_modulated is not None:
        # 前回変調値と基準値の差分
        prev_deviation = previous_modulated - base_value
        # 回帰圧力: 前回の偏差を基準値方向に引き戻す
        regression = -prev_deviation * _REGRESSION_PRESSURE
        # 新規変調量と回帰圧力のブレンド
        raw_modulation = raw_modulation + regression

    # 変調後の値
    modulated_float = base_value + raw_modulation

    # 安全弁1: 帯域上限の適用
    min_value = base_value - max_delta
    max_value = base_value + max_delta
    modulated_float = max(min_value, min(max_value, modulated_float))

    # 安全弁3: 整数丸め（最小値1を保証）
    result = max(1, round(modulated_float))

    return result


# =============================================================================
# Main entry point (session startup only)
# =============================================================================

def apply_window_size_modulation(
    frb_state: ForgettingRecallBalanceState,
    previous_modulated_values: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """セッション起動時にウィンドウサイズの変調を適用する。

    この関数はセッション起動時に一度だけ呼び出される。
    ティック中の呼び出しは構造的に禁止される（呼び出し元が起動時のみ）。

    処理フロー:
    1. 忘却想起均衡記述の蓄積データから想起頻度を読み取る
    2. 想起頻度を変調方向係数に変換する
    3. 各ウィンドウサイズについて変調後の値を算出する
    4. 係数レジストリの値を変調後の値で更新する

    安全弁1: 帯域上限（基準値の±20%）
    安全弁2: 基準値回帰圧力
    安全弁3: 整数丸め
    安全弁4: 起動時限定
    安全弁5: 蓄積データのREAD-ONLY参照

    Args:
        frb_state: 忘却想起均衡記述の内部状態 (READ-ONLY)
        previous_modulated_values: 前回セッションの変調後の値 (key -> int)

    Returns:
        変調後のウィンドウサイズの辞書 {key: modulated_value}
    """
    prev = previous_modulated_values or {}

    # Step 1: 想起頻度の読み取り
    total_recall = extract_recall_frequency(frb_state)

    # Step 2: 変調方向係数の算出
    direction_factor = classify_recall_frequency(total_recall)

    # Step 3 & 4: 各ウィンドウサイズの変調と適用
    modulated_values: dict[str, int] = {}

    for key in _WINDOW_SIZE_KEYS:
        # 基準値の取得（係数レジストリのデフォルト値）
        try:
            base_value = coefficient_registry.get_defaults()["description_common"][key]
        except (KeyError, TypeError):
            logger.debug(
                "Window size modulation: base value not found for key '%s'", key
            )
            continue

        if not isinstance(base_value, (int, float)):
            continue

        base_int = int(base_value)

        # 前回変調値の取得
        prev_modulated = prev.get(key)

        # 変調後の値を算出
        modulated = compute_modulation(
            base_value=base_int,
            direction_factor=direction_factor,
            previous_modulated=prev_modulated,
        )

        modulated_values[key] = modulated

        # 係数レジストリの値を更新
        _apply_to_registry(key, modulated)

        if modulated != base_int:
            logger.info(
                "Window size modulation: %s = %d (base=%d, factor=%.2f, "
                "prev=%s, recall=%d)",
                key, modulated, base_int, direction_factor,
                prev_modulated, total_recall,
            )
        else:
            logger.debug(
                "Window size modulation: %s = %d (no change, "
                "factor=%.2f, recall=%d)",
                key, modulated, direction_factor, total_recall,
            )

    return modulated_values


def _apply_to_registry(key: str, value: int) -> None:
    """係数レジストリのdescription_common名前空間の値を更新する。

    起動時のみの呼び出しを前提とする。
    係数レジストリの内部構造に直接アクセスし、指定キーの値を更新する。

    Args:
        key: description_common 内のキー名
        value: 設定する整数値
    """
    # coefficient_registry._registry を直接参照して更新
    # ティック中のアクセスではなく起動時のみであることが呼び出し元で保証される
    registry = coefficient_registry._registry
    if registry is None:
        # レジストリ未初期化の場合は自動初期化
        coefficient_registry.load()
        registry = coefficient_registry._registry

    if registry is not None and "description_common" in registry:
        registry["description_common"][key] = value


# =============================================================================
# Save / Load for previous modulated values
# =============================================================================

def save_modulated_values(modulated: dict[str, int]) -> dict[str, Any]:
    """変調後の値を永続化用辞書に変換する。"""
    return dict(modulated)


def load_modulated_values(data: Any) -> dict[str, int]:
    """永続化用辞書から変調後の値を復元する。"""
    if not isinstance(data, dict):
        return {}
    result: dict[str, int] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, (int, float)):
            result[k] = int(v)
    return result
