"""
psyche/stabilization_description.py - 安定化の構造的記述

複数の信号源の活性状況とtemporal_self_differenceの差分度合いを
横断的に読み取り、2つの断面として非評価的に記述する集約層。

設計原則 (design_stabilization_description.md 準拠):
- 6箇所の信号源から「出力が存在するか否か」の二値のみを読み取る
- temporal_self_differenceの差分サマリーの規模・性質を読み取り専用で参照する
- 安定化を推奨しない。変動を矯正しない
- 記録間の優劣を確定しない（全記録等価）
- 断面間の相関・因果・傾向を算出しない（パターン抽出禁止）
- 閾値による段階的カテゴリ化を行わない
- enrichmentに直接露出しない（経路遮断）
- 忘却パイプラインに入力を供給しない（経路遮断）
- 想起経路選択への影響を遮断
- 判断・行動・責任システムに接続しない

断面構造:
  - 断面1: 信号源の多寡（非ゼロ出力を持つ信号源の個数）
  - 断面2: 差分度合い参照値（temporal_self_differenceの規模と性質の対）
  - 記録時点のティック番号

蓄積リスト:
  - 断面記録を時系列で保持するFIFOリスト（有限上限、最古押し出しが唯一の消失経路）

安全弁（5種）:
  1. 全記録等価 — 蓄積リスト内の全記録に重み・スコア・優先度を付与しない
  2. パターン抽出禁止 — 蓄積された記録から傾向、周期性、統計量を算出しない
  3. enrichment直接露出遮断 — 本機能の出力をget_prompt_enrichment()に含めない
  4. 忘却経路遮断 — 本機能の出力を記憶の忘却・固定化処理の入力に使用しない
  5. 出力経路不拡張 — 初期実装で定義した出力経路以外の経路を追加しない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StabilizationDescriptionConfig:
    """安定化記述モジュールの設定。"""

    # 蓄積リストの最大保持件数（安全弁: 有限性、最古押し出しが唯一の消失経路）
    max_history: int = 30


# =============================================================================
# 信号源識別子（6箇所）
# =============================================================================

# 設計書で列挙された6箇所の信号源。
# 記憶系は reference_frequency_description が12箇所横断するため含めない。

SIGNAL_EMOTION = "dominant_emotion_intensity"
SIGNAL_STM_ENTRIES = "stm_entry_count"
SIGNAL_TRANSIENT_GOAL = "transient_goal_active"
SIGNAL_PERSISTENT_COMMITMENT = "persistent_commitment_unreleased"
SIGNAL_SPONTANEOUS_CANDIDATE = "spontaneous_candidate_recent"
SIGNAL_EXTERNAL_INPUT = "external_input_recent"

ALL_SIGNAL_KEYS = [
    SIGNAL_EMOTION,
    SIGNAL_STM_ENTRIES,
    SIGNAL_TRANSIENT_GOAL,
    SIGNAL_PERSISTENT_COMMITMENT,
    SIGNAL_SPONTANEOUS_CANDIDATE,
    SIGNAL_EXTERNAL_INPUT,
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StabilizationRecord:
    """断面記録。ある時点で読み取った2断面の非評価的記録。

    断面は生成後に変更されない（不変）。
    安全弁1: 全記録等価。記録間の重み・スコア・優先度を付与しない。
    安全弁2: パターン抽出禁止。記録間の比較・差分・相関の計算を行わない。
    """

    # 断面1: 信号源活性数（非ゼロ出力を持つ信号源の個数、整数値）
    active_signal_count: int = 0

    # 各信号源の活性状態（信号源キー → True/False）
    # 安全弁1: 全記録等価。重み付け・順序付けを行わない
    signal_states: dict[str, bool] = field(default_factory=dict)

    # 断面2: 差分度合い参照値（temporal_self_differenceから読み取った値）
    # 規模情報: 差分が検出されたかどうか、差分の量的表現（文字列値）
    diff_magnitude: str = "undefined"
    # 性質情報: 安定/シフト/変容のいずれか（文字列値）
    diff_nature: str = "undefined"

    # 記録時点のティック番号
    tick: int = 0

    # 生成時刻
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_signal_count": self.active_signal_count,
            "signal_states": dict(self.signal_states),
            "diff_magnitude": self.diff_magnitude,
            "diff_nature": self.diff_nature,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StabilizationRecord":
        return cls(
            active_signal_count=data.get("active_signal_count", 0),
            signal_states=dict(data.get("signal_states", {})),
            diff_magnitude=data.get("diff_magnitude", "undefined"),
            diff_nature=data.get("diff_nature", "undefined"),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", 0.0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class StabilizationDescriptionState:
    """安定化記述モジュールの内部状態。

    安全弁1: 全記録等価。蓄積リスト内の全記録に重み・スコア・優先度を付与しない。
    """

    # 蓄積リスト（時系列順、先入先出）
    history: list[StabilizationRecord] = field(default_factory=list)

    # 直前の断面記録のキャッシュ（参照受渡用）
    latest_record: Optional[StabilizationRecord] = None

    # 累積カウンタ（診断情報のみ、処理分岐に使用しない）
    total_records_generated: int = 0
    total_records_expired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": [r.to_dict() for r in self.history],
            "latest_record": self.latest_record.to_dict() if self.latest_record else None,
            "total_records_generated": self.total_records_generated,
            "total_records_expired": self.total_records_expired,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StabilizationDescriptionState":
        history = [
            StabilizationRecord.from_dict(r)
            for r in data.get("history", [])
        ]
        lr_data = data.get("latest_record")
        latest = StabilizationRecord.from_dict(lr_data) if lr_data else None
        return cls(
            history=history,
            latest_record=latest,
            total_records_generated=data.get("total_records_generated", 0),
            total_records_expired=data.get("total_records_expired", 0),
        )


# =============================================================================
# 第1段: 読み取り — 信号源から値をREAD-ONLYで読み取る
# =============================================================================

def read_signal_sources(
    *,
    emotion_intensity: float = 0.0,
    stm_entry_count: int = 0,
    transient_goal_active: bool = False,
    persistent_commitment_unreleased_count: int = 0,
    spontaneous_candidate_exists: bool = False,
    has_external_input: bool = False,
) -> dict[str, bool]:
    """各信号源から「出力が存在するか否か」の二値のみを読み取る。

    読み取りは全てREAD-ONLYであり、読み取り元の状態を変更しない。
    安全弁1: 全記録等価。読み取った値の重み付け・比較・順序付けは行わない。

    Args:
        emotion_intensity: 感情システムの支配的感情の強度
        stm_entry_count: 短期記憶の保持エントリ数
        transient_goal_active: 一時的目的がアクティブかどうか
        persistent_commitment_unreleased_count: 持続的取り組み保持の未解放項目数
        spontaneous_candidate_exists: 自発起動の直近候補有無
        has_external_input: 直近ティックで外部入力があったかどうか

    Returns:
        信号源キー → 活性状態（True/False）の辞書
    """
    return {
        SIGNAL_EMOTION: emotion_intensity > 0.0,
        SIGNAL_STM_ENTRIES: stm_entry_count > 0,
        SIGNAL_TRANSIENT_GOAL: transient_goal_active,
        SIGNAL_PERSISTENT_COMMITMENT: persistent_commitment_unreleased_count > 0,
        SIGNAL_SPONTANEOUS_CANDIDATE: spontaneous_candidate_exists,
        SIGNAL_EXTERNAL_INPUT: has_external_input,
    }


def read_diff_reference(
    *,
    diff_summary: Any = None,
) -> tuple[str, str]:
    """temporal_self_differenceの差分サマリーを読み取り専用で参照する。

    新たに変動幅を計算することは行わない。
    既にtemporal_self_differenceが計算済みの値を「読む」のみ。

    Args:
        diff_summary: SelfDifferenceSummary（duck typing）

    Returns:
        (magnitude_str, nature_str) の対。差分情報がなければ ("undefined", "undefined")。
    """
    if diff_summary is None:
        return ("undefined", "undefined")

    # magnitude: DifferenceMagnitude の .value を取得
    magnitude_raw = getattr(diff_summary, "magnitude", None)
    if magnitude_raw is not None:
        magnitude_str = getattr(magnitude_raw, "value", str(magnitude_raw))
    else:
        magnitude_str = "undefined"

    # nature: ChangeNature の .value を取得
    nature_raw = getattr(diff_summary, "nature", None)
    if nature_raw is not None:
        nature_str = getattr(nature_raw, "value", str(nature_raw))
    else:
        nature_str = "undefined"

    return (str(magnitude_str), str(nature_str))


# =============================================================================
# 第2段: 断面構成 — 2断面をひとまとまりの記録として構成する
# =============================================================================

def compose_record(
    signal_states: dict[str, bool],
    diff_magnitude: str,
    diff_nature: str,
    tick: int = 0,
    timestamp: Optional[float] = None,
) -> StabilizationRecord:
    """2つの断面をひとまとまりの記録として構成する。

    各断面の値はそのまま記録する（変換・正規化・丸め・段階化を行わない）。
    安全弁1: 全記録等価。
    安全弁2: パターン抽出禁止。

    Args:
        signal_states: 信号源の活性状態辞書
        diff_magnitude: 差分の規模情報（文字列）
        diff_nature: 差分の性質情報（文字列）
        tick: ティック番号
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        構成された断面記録（不変）
    """
    ts = timestamp if timestamp is not None else time.time()
    active_count = sum(1 for v in signal_states.values() if v)

    return StabilizationRecord(
        active_signal_count=active_count,
        signal_states=dict(signal_states),
        diff_magnitude=diff_magnitude,
        diff_nature=diff_nature,
        tick=tick,
        timestamp=ts,
    )


# =============================================================================
# 第3段: 蓄積と参照受渡 — FIFO蓄積 + READ-ONLYアクセサ
# =============================================================================

def accumulate_record(
    state: StabilizationDescriptionState,
    record: StabilizationRecord,
    config: Optional[StabilizationDescriptionConfig] = None,
) -> StabilizationDescriptionState:
    """構成された記録をFIFO蓄積に追加する。

    蓄積上限を超えた場合、最古の記録が押し出されて消失する（唯一の消失経路）。
    安全弁1: 全記録等価。特定の記録を強調、保護、優先的に保持する機構を持たない。

    Args:
        state: 現在の状態
        record: 新しい断面記録
        config: 設定

    Returns:
        更新されたStabilizationDescriptionState
    """
    cfg = config or StabilizationDescriptionConfig()

    new_history = list(state.history)
    new_history.append(record)

    # FIFO: 上限超過時は最古押し出し（唯一の消失経路）
    expired = 0
    if len(new_history) > cfg.max_history:
        expired = len(new_history) - cfg.max_history
        new_history = new_history[expired:]

    return StabilizationDescriptionState(
        history=new_history,
        latest_record=record,
        total_records_generated=state.total_records_generated + 1,
        total_records_expired=state.total_records_expired + expired,
    )


# =============================================================================
# メイン処理: 1サイクル処理
# =============================================================================

def process_stabilization_description(
    state: StabilizationDescriptionState,
    *,
    emotion_intensity: float = 0.0,
    stm_entry_count: int = 0,
    transient_goal_active: bool = False,
    persistent_commitment_unreleased_count: int = 0,
    spontaneous_candidate_exists: bool = False,
    has_external_input: bool = False,
    diff_summary: Any = None,
    tick: int = 0,
    config: Optional[StabilizationDescriptionConfig] = None,
    timestamp: Optional[float] = None,
) -> StabilizationDescriptionState:
    """安定化記述の1サイクル処理を実行する。

    安全弁1: 全記録等価
    安全弁2: パターン抽出禁止
    安全弁3: enrichment直接露出遮断（本関数はenrichment出力を持たない）
    安全弁4: 忘却経路遮断（本関数は忘却パイプラインへの出力経路を持たない）
    安全弁5: 出力経路不拡張（出力先は内省系READ-ONLYアクセサのみ）

    入力構造の値は読み取り専用。書き込み能力を付与しない。

    Args:
        state: 現在の状態
        emotion_intensity: 感情システムの支配的感情の強度
        stm_entry_count: 短期記憶の保持エントリ数
        transient_goal_active: 一時的目的がアクティブかどうか
        persistent_commitment_unreleased_count: 持続的取り組み保持の未解放項目数
        spontaneous_candidate_exists: 自発起動の直近候補有無
        has_external_input: 直近ティックで外部入力があったかどうか
        diff_summary: temporal_self_differenceのSelfDifferenceSummary
        tick: ティック番号
        config: 設定
        timestamp: 断面の生成時刻

    Returns:
        更新されたStabilizationDescriptionState
    """
    cfg = config or StabilizationDescriptionConfig()

    # ── 第1段: 読み取り ──
    signal_states = read_signal_sources(
        emotion_intensity=emotion_intensity,
        stm_entry_count=stm_entry_count,
        transient_goal_active=transient_goal_active,
        persistent_commitment_unreleased_count=persistent_commitment_unreleased_count,
        spontaneous_candidate_exists=spontaneous_candidate_exists,
        has_external_input=has_external_input,
    )

    diff_magnitude, diff_nature = read_diff_reference(
        diff_summary=diff_summary,
    )

    # ── 第2段: 断面構成 ──
    record = compose_record(
        signal_states=signal_states,
        diff_magnitude=diff_magnitude,
        diff_nature=diff_nature,
        tick=tick,
        timestamp=timestamp,
    )

    # ── 第3段: 蓄積と参照受渡 ──
    new_state = accumulate_record(state, record, cfg)

    logger.debug(
        "Stabilization description: active_signals=%d, magnitude=%s, nature=%s, "
        "history=%d, expired=%d",
        record.active_signal_count,
        record.diff_magnitude,
        record.diff_nature,
        len(new_state.history),
        new_state.total_records_expired - state.total_records_expired,
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁5: 出力先は内省系構造への参照情報に限定。
# enrichmentへの出力経路を持たない。
# 忘却パイプラインへの出力経路を持たない。
# 想起経路の選択への出力経路を持たない。
# ポリシー選択への出力経路を持たない。
# 感情パイプラインへの出力経路を持たない。

def get_latest_record(state: StabilizationDescriptionState) -> Optional[StabilizationRecord]:
    """最新の断面記録を返す（内省系参照経路、READ-ONLY）。

    参照行為によって状態が変化することはない。

    Returns:
        直近の断面記録。履歴がなければ None。
    """
    return state.latest_record


def get_record_history(state: StabilizationDescriptionState) -> list[StabilizationRecord]:
    """蓄積リスト全体を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    安全弁2: パターン抽出禁止。蓄積された記録から傾向等を算出しない。
    参照行為によって状態が変化することはない。

    Returns:
        蓄積リストのコピー。
    """
    return list(state.history)


def get_stabilization_summary(state: StabilizationDescriptionState) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の記録を強調・選別しない。
    安全弁3: enrichment出力経路を持たない。
    安全弁5: 出力経路不拡張。

    Returns:
        サマリ辞書。
    """
    latest = state.latest_record

    summary: dict[str, Any] = {
        "history_count": len(state.history),
        "total_generated": state.total_records_generated,
        "total_expired": state.total_records_expired,
    }

    if latest is not None:
        summary["latest_active_signal_count"] = latest.active_signal_count
        summary["latest_diff_magnitude"] = latest.diff_magnitude
        summary["latest_diff_nature"] = latest.diff_nature
        summary["latest_tick"] = latest.tick
        summary["latest_timestamp"] = latest.timestamp
        summary["latest_signal_states"] = dict(latest.signal_states)

    return summary


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: StabilizationDescriptionState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> StabilizationDescriptionState:
    """永続化用の辞書から状態を復元する。"""
    return StabilizationDescriptionState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_stabilization_description_state() -> StabilizationDescriptionState:
    """初期状態のファクトリ関数。"""
    return StabilizationDescriptionState()


def create_stabilization_description_config(
    max_history: int = 30,
) -> StabilizationDescriptionConfig:
    """設定のファクトリ関数。"""
    return StabilizationDescriptionConfig(
        max_history=max_history,
    )
