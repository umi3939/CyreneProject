"""
psyche/scoring_fluctuation.py - スコアリングの構造的揺らぎ

同一の内部状態・同一の知覚入力でも、ポリシー選択が決定論的に固定されない
ことを保証する揺らぎ層。

設計原則 (design_scoring_fluctuation.md 準拠):
- 揺らぎは外部ノイズではなく、内部状態の変動から導出される
- 振幅は value_orientation の max_bias_strength (±5%) より厳密に小さい
- 永続化対象の状態を持たない（呼び出しごとに完結）
- 揺らぎの結果は蓄積されない（パターン化・固定化が構造的に不可能）
- 入力源への逆流なし（読み取り専用）

5段パイプライン:
  段階1: 各入力源から変動度（スカラー）を抽出
  段階2: 変動度を合成（最大値と平均値の中間）
  段階3: 振幅を上限・下限で制限
  段階4: ポリシー別の揺らぎ値を生成
  段階5: 既存スコアに加算

安全弁:
  1. 振幅の絶対上限（value_orientation の max_bias_strength 未満）
  2. 状態蓄積の禁止（永続化対象なし）
  3. 入力源への逆流遮断（読み取り専用）
  4. 長期価値軸更新経路への非介入（揺らぎ値は伝播しない）
  5. 下限による消失防止（揺らぎがゼロにならない）
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScoringFluctuationConfig:
    """揺らぎモジュールの設定。

    安全弁1: amplitude_cap は value_orientation の max_bias_strength (デフォルト0.15)
    より厳密に小さい値でなければならない。
    """

    # 振幅の絶対上限（安全弁1）
    # value_orientation の max_bias_strength (0.15) より厳密に小さい
    amplitude_cap: float = 0.12

    # 振幅の下限（安全弁5: 揺らぎが完全に消失しない）
    amplitude_floor: float = 0.005

    # value_orientation の max_bias_strength（超過チェック用）
    vo_max_bias_strength: float = 0.15

    # 各入力源の重み（変動度合成時に使用）
    weight_emotion: float = 0.3
    weight_stm: float = 0.25
    weight_drives: float = 0.25
    weight_elapsed: float = 0.2

    # 経過時間の最大考慮秒数（これ以上は頭打ち）
    max_elapsed_seconds: float = 300.0

    # 経過時間の影響係数
    elapsed_scale: float = 0.5

    def __post_init__(self):
        """安全弁1: amplitude_cap が vo_max_bias_strength 以上なら拒否。"""
        if self.amplitude_cap >= self.vo_max_bias_strength:
            self.amplitude_cap = self.vo_max_bias_strength * 0.8
        # 下限が上限以上にならないように
        if self.amplitude_floor >= self.amplitude_cap:
            self.amplitude_floor = self.amplitude_cap * 0.04


# =============================================================================
# Helper: clamp
# =============================================================================

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# Stage 1: 変動量の抽出
# =============================================================================

def extract_emotion_variability(emotions: dict[str, float]) -> float:
    """感情ベクトルの多次元的な偏りから変動度を抽出する。

    均衡状態（全次元が近い値）に近いほど変動度は小さく、
    偏り（一部の次元だけ高い等）が大きいほど変動度は大きい。

    Args:
        emotions: 感情次元名→値 (0.0-1.0) の辞書

    Returns:
        0.0-1.0 の変動度スカラー
    """
    if not emotions:
        return 0.0

    values = list(emotions.values())
    if not values:
        return 0.0

    n = len(values)
    if n <= 1:
        return abs(values[0]) if values else 0.0

    mean = sum(values) / n
    # 分散 = 偏りの指標
    variance = sum((v - mean) ** 2 for v in values) / n
    # 最大値と最小値の差 = 次元間の差分
    spread = max(values) - min(values)

    # 分散と広がりを組み合わせて変動度を算出
    # variance の最大値は 0.25 (0と1のみの場合), spread の最大値は 1.0
    variability = (variance * 2.0 + spread * 0.5) / 2.0

    return _clamp(variability)


def extract_stm_variability(
    stm_entry_count: int,
    stm_time_span: float,
    stm_residue_intensity: float,
    stm_continuity: float,
) -> float:
    """短期記憶の蓄積状態の「形状」から変動度を抽出する。

    記憶の内容は参照しない。構造的特徴のみ。

    Args:
        stm_entry_count: STMに保持されている要素数
        stm_time_span: 最新要素と最古要素の時間差（秒）
        stm_residue_intensity: 残留影響の強度合計
        stm_continuity: 文脈継続スコア (0.0-1.0)

    Returns:
        0.0-1.0 の変動度スカラー
    """
    # STMが空の場合、形状から変動度を抽出しようがない
    if stm_entry_count <= 0:
        return 0.0

    # 要素数: 多いほど活発 (0-10 → 0.0-1.0)
    count_factor = _clamp(stm_entry_count / 10.0)

    # 時間幅: 広いほど多様な経験 (0-120秒 → 0.0-1.0)
    time_factor = _clamp(stm_time_span / 120.0)

    # 残留影響: 高いほど余韻が強い (0.0-5.0 → 0.0-1.0)
    residue_factor = _clamp(stm_residue_intensity / 5.0)

    # 文脈継続: 低いほど変化が多い（反転して使う）
    discontinuity_factor = _clamp(1.0 - stm_continuity)

    # 組み合わせ
    variability = (
        count_factor * 0.25
        + time_factor * 0.25
        + residue_factor * 0.25
        + discontinuity_factor * 0.25
    )

    return _clamp(variability)


def extract_drive_variability(drives: dict[str, float]) -> float:
    """駆動状態の次元間の不均衡から変動度を抽出する。

    駆動の絶対値ではなく、駆動間の相対的な不均衡の度合いを使用。

    Args:
        drives: 駆動次元名→値 (0.0-1.0) の辞書

    Returns:
        0.0-1.0 の変動度スカラー
    """
    if not drives:
        return 0.0

    values = list(drives.values())
    if not values:
        return 0.0

    n = len(values)
    if n <= 1:
        return 0.0

    mean = sum(values) / n
    # 分散: 駆動間の不均衡
    variance = sum((v - mean) ** 2 for v in values) / n
    # variance の理論最大は ~0.222 (3次元の場合、2つが1.0で1つが0.0)
    # 正規化して 0.0-1.0 にする
    variability = _clamp(variance * 4.5)

    return variability


def extract_elapsed_variability(
    elapsed_seconds: float,
    config: ScoringFluctuationConfig,
) -> float:
    """前回のポリシー選択からの経過時間から変動度を抽出する。

    経過時間が長いほど内部状態の微小変化が累積しているため、
    揺らぎの振幅が微増する。ただし上限が存在する。

    Args:
        elapsed_seconds: 前回のポリシー選択からの経過秒数
        config: 設定

    Returns:
        0.0-1.0 の変動度スカラー
    """
    if elapsed_seconds <= 0:
        return 0.0

    capped = min(elapsed_seconds, config.max_elapsed_seconds)
    # 対数的に増加（急激に上がりすぎない）
    import math
    ratio = capped / config.max_elapsed_seconds
    variability = ratio * config.elapsed_scale

    return _clamp(variability)


# =============================================================================
# Stage 2: 変動度の合成
# =============================================================================

def compose_variability(
    emotion_var: float,
    stm_var: float,
    drive_var: float,
    elapsed_var: float,
    config: ScoringFluctuationConfig,
) -> float:
    """複数の入力源からの変動度を合成する。

    合成は加重平均ではなく、最大値と平均値の中間的な統合とする。
    これにより、単一の入力源が揺らぎを支配することを防ぐ。

    Args:
        emotion_var: 感情由来の変動度
        stm_var: STM由来の変動度
        drive_var: 駆動由来の変動度
        elapsed_var: 経過時間由来の変動度
        config: 設定

    Returns:
        0.0-1.0 の合成変動度
    """
    values = [emotion_var, stm_var, drive_var, elapsed_var]
    weights = [
        config.weight_emotion,
        config.weight_stm,
        config.weight_drives,
        config.weight_elapsed,
    ]

    if not values:
        return 0.0

    # 加重平均
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_weight

    # 最大値
    max_val = max(values)

    # 最大値と平均値の中間（単一入力源支配を防ぐ）
    composed = (weighted_avg + max_val) / 2.0

    return _clamp(composed)


# =============================================================================
# Stage 3: 振幅の制限
# =============================================================================

def limit_amplitude(
    composed_variability: float,
    config: ScoringFluctuationConfig,
) -> float:
    """合成された変動度を振幅の上限と下限で制限する。

    安全弁1: 上限は value_orientation の max_bias_strength より厳密に小さい。
    安全弁5: 下限は微小値で、揺らぎが完全に消失しないことを保証。

    Args:
        composed_variability: 合成変動度 (0.0-1.0)
        config: 設定

    Returns:
        制限された振幅値
    """
    # 変動度をスケーリングして振幅にする
    amplitude = composed_variability * config.amplitude_cap

    # 安全弁1: 上限制限
    amplitude = min(amplitude, config.amplitude_cap)

    # 安全弁5: 下限制限（完全消失防止）
    amplitude = max(amplitude, config.amplitude_floor)

    return amplitude


# =============================================================================
# Stage 4: ポリシー別の揺らぎ値生成
# =============================================================================

def _derive_hash_float(
    policy_label: str,
    drive_target: str,
    emotion_var: float,
    stm_var: float,
    drive_var: float,
    elapsed_var: float,
    timestamp: float,
) -> float:
    """内部状態の変動成分とポリシー特性から決定論的だが非固定の
    ハッシュベース浮動小数点を導出する。

    同一ポリシーでも内部状態の変動成分が異なれば異なる値が生成される。
    同一の入力でもタイムスタンプが異なれば異なる値が生成される。

    Returns:
        -1.0 から 1.0 の範囲の浮動小数点数
    """
    # ポリシーの特性と内部状態の変動成分を組み合わせてハッシュの入力を作る
    # 浮動小数点を適度な精度で丸めて、微小な差異で値が変わるようにする
    raw = (
        f"{policy_label}|{drive_target}|"
        f"{emotion_var:.6f}|{stm_var:.6f}|{drive_var:.6f}|{elapsed_var:.6f}|"
        f"{timestamp:.4f}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    # 最初の8バイトから64ビット整数を取得
    int_val = struct.unpack(">Q", digest[:8])[0]
    # 0.0 - 1.0 に正規化
    normalized = int_val / (2**64 - 1)
    # -1.0 - 1.0 に変換
    return (normalized * 2.0) - 1.0


def generate_per_policy_fluctuations(
    candidates: list[dict[str, Any]],
    amplitude: float,
    emotion_var: float,
    stm_var: float,
    drive_var: float,
    elapsed_var: float,
) -> list[dict[str, float]]:
    """各ポリシー候補に対して個別の揺らぎ値を生成する。

    揺らぎ値の生成には、各ポリシーの特性（drive_target, expected_drive_change）
    と内部状態の変動成分の相互作用を用いる。

    Args:
        candidates: ポリシー候補リスト
        amplitude: 制限された振幅
        emotion_var: 感情変動度
        stm_var: STM変動度
        drive_var: 駆動変動度
        elapsed_var: 経過時間変動度

    Returns:
        各候補に対応する {"policy_label": str, "fluctuation": float} のリスト
    """
    timestamp = time.time()
    results: list[dict[str, float]] = []

    for candidate in candidates:
        policy_label = candidate.get("policy_label", "")
        drive_target = candidate.get("drive_target", "")

        # expected_drive_change から変化量の特徴を抽出
        expected_change = candidate.get("expected_drive_change", {})
        change_sum = sum(abs(v) for v in expected_change.values()) if expected_change else 0.0

        # ハッシュベースで方向と大きさを決定
        hash_val = _derive_hash_float(
            policy_label=policy_label,
            drive_target=drive_target,
            emotion_var=emotion_var,
            stm_var=stm_var,
            drive_var=drive_var,
            elapsed_var=elapsed_var,
            timestamp=timestamp,
        )

        # 変化量の特徴を揺らぎに少し影響させる
        # change_sum が大きい候補ほど揺らぎの影響を受けやすい
        change_factor = _clamp(0.5 + change_sum * 0.5, 0.3, 1.0)

        fluctuation = hash_val * amplitude * change_factor

        results.append({
            "policy_label": policy_label,
            "fluctuation": fluctuation,
        })

    return results


# =============================================================================
# Stage 5: スコアへの加算
# =============================================================================

def apply_fluctuations_to_candidates(
    candidates: list[dict[str, Any]],
    fluctuations: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """生成された揺らぎ値を各候補の既存スコアに加算する。

    加算は単純な足し算であり、既存スコアの構造を変更しない。
    揺らぎの値は観測用に候補情報へ付記される。

    Args:
        candidates: ポリシー候補リスト
        fluctuations: 各候補への揺らぎ値

    Returns:
        揺らぎが加算された候補リスト（新しいリスト）
    """
    # fluctuation の辞書を作成
    fluct_map: dict[str, float] = {}
    for f in fluctuations:
        label = f.get("policy_label", "")
        if label:
            fluct_map[label] = f.get("fluctuation", 0.0)

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        new_candidate = candidate.copy()
        label = candidate.get("policy_label", "")
        fluct_val = fluct_map.get(label, 0.0)

        original_score = candidate.get("_score", 0.0)
        new_candidate["_score"] = round(original_score + fluct_val, 6)
        new_candidate["_pre_fluctuation_score"] = original_score
        new_candidate["_fluctuation"] = round(fluct_val, 6)
        new_candidate["_fluctuation_applied"] = True

        result.append(new_candidate)

    # 再ソート（スコア降順）
    result.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return result


# =============================================================================
# STM情報の抽出ヘルパー
# =============================================================================

def extract_stm_info(stm: Any) -> dict[str, float]:
    """ShortTermMemory からスコアリング揺らぎに必要な情報を抽出する。

    STM の内容は参照しない。構造的特徴のみ。

    Args:
        stm: ShortTermMemory インスタンス（None可）

    Returns:
        {"entry_count", "time_span", "residue_intensity", "continuity"} の辞書
    """
    if stm is None:
        return {
            "entry_count": 0,
            "time_span": 0.0,
            "residue_intensity": 0.0,
            "continuity": 0.0,
        }

    entries = getattr(stm, "entries", [])
    entry_count = len(entries)

    # 時間幅
    time_span = 0.0
    if entry_count >= 2:
        timestamps = [getattr(e, "timestamp", 0.0) for e in entries]
        timestamps = [t for t in timestamps if t > 0]
        if len(timestamps) >= 2:
            time_span = max(timestamps) - min(timestamps)

    # 残留影響の強度合計
    residue_intensity = 0.0
    unprocessed = getattr(stm, "get_unprocessed_residue", None)
    if callable(unprocessed):
        for entry in unprocessed():
            residue_intensity += getattr(entry, "residue_weight", 0.0) * getattr(entry, "raw_intensity", 0.0)

    # 文脈継続スコア
    continuity = getattr(stm, "context_continuity_score", 0.0)

    return {
        "entry_count": entry_count,
        "time_span": time_span,
        "residue_intensity": residue_intensity,
        "continuity": continuity,
    }


# =============================================================================
# メインパイプライン関数
# =============================================================================

def apply_scoring_fluctuation(
    candidates: list[dict[str, Any]],
    emotions: dict[str, float],
    drives: dict[str, float],
    stm: Any = None,
    elapsed_seconds: float = 0.0,
    config: Optional[ScoringFluctuationConfig] = None,
) -> list[dict[str, Any]]:
    """5段パイプラインを実行し、揺らぎを適用した候補リストを返す。

    安全弁2: 本関数は内部状態を保持せず、呼び出しごとに完結する。
    安全弁3: 入力（emotions, drives, stm）は読み取り専用で変更しない。
    安全弁4: 揺らぎ値は返却される候補のメタデータにのみ付記され、
             長期価値軸やその他の構造には伝播しない。

    Args:
        candidates: ポリシー候補リスト（バイアス適用済み）
        emotions: 感情次元名→値 の辞書（EmotionVector.as_dict() の結果）
        drives: 駆動次元名→値 の辞書（DriveVector.as_dict() の結果）
        stm: ShortTermMemory インスタンス（None可）
        elapsed_seconds: 前回のポリシー選択からの経過秒数
        config: 設定（None時はデフォルト）

    Returns:
        揺らぎが加算されたポリシー候補リスト（新しいリスト）。
        元の candidates は変更しない。
    """
    if not candidates:
        return []

    cfg = config or ScoringFluctuationConfig()

    # ── 段階1: 変動量の抽出 ──
    emotion_var = extract_emotion_variability(emotions)

    stm_info = extract_stm_info(stm)
    stm_var = extract_stm_variability(
        stm_entry_count=int(stm_info["entry_count"]),
        stm_time_span=stm_info["time_span"],
        stm_residue_intensity=stm_info["residue_intensity"],
        stm_continuity=stm_info["continuity"],
    )

    drive_var = extract_drive_variability(drives)

    elapsed_var = extract_elapsed_variability(elapsed_seconds, cfg)

    # ── 段階2: 変動度の合成 ──
    composed = compose_variability(
        emotion_var=emotion_var,
        stm_var=stm_var,
        drive_var=drive_var,
        elapsed_var=elapsed_var,
        config=cfg,
    )

    # ── 段階3: 振幅の制限 ──
    amplitude = limit_amplitude(composed, cfg)

    # ── 段階4: ポリシー別の揺らぎ値生成 ──
    fluctuations = generate_per_policy_fluctuations(
        candidates=candidates,
        amplitude=amplitude,
        emotion_var=emotion_var,
        stm_var=stm_var,
        drive_var=drive_var,
        elapsed_var=elapsed_var,
    )

    # ── 段階5: スコアへの加算 ──
    result = apply_fluctuations_to_candidates(candidates, fluctuations)

    logger.debug(
        "Scoring fluctuation applied: composed_var=%.4f, amplitude=%.4f, "
        "candidates=%d",
        composed, amplitude, len(result),
    )

    return result


# =============================================================================
# ユーティリティ
# =============================================================================

def get_fluctuation_summary(candidates: list[dict[str, Any]]) -> str:
    """揺らぎ適用結果のサマリーを返す。"""
    if not candidates:
        return "Fluctuation: no candidates"

    applied = [c for c in candidates if c.get("_fluctuation_applied")]
    if not applied:
        return "Fluctuation: not applied"

    flucts = [c.get("_fluctuation", 0.0) for c in applied]
    max_f = max(flucts) if flucts else 0.0
    min_f = min(flucts) if flucts else 0.0
    avg_f = sum(flucts) / len(flucts) if flucts else 0.0

    return (
        f"Fluctuation: applied to {len(applied)} candidates, "
        f"range=[{min_f:.4f}, {max_f:.4f}], avg={avg_f:.4f}"
    )


def create_fluctuation_config(
    amplitude_cap: float = 0.12,
    amplitude_floor: float = 0.005,
    vo_max_bias_strength: float = 0.15,
) -> ScoringFluctuationConfig:
    """設定を作成する。

    Args:
        amplitude_cap: 振幅上限（vo_max_bias_strength より小さくなければならない）
        amplitude_floor: 振幅下限
        vo_max_bias_strength: value_orientation の max_bias_strength

    Returns:
        ScoringFluctuationConfig
    """
    return ScoringFluctuationConfig(
        amplitude_cap=amplitude_cap,
        amplitude_floor=amplitude_floor,
        vo_max_bias_strength=vo_max_bias_strength,
    )
