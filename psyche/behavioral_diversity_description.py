"""
psyche/behavioral_diversity_description.py - 行動多様性の構造的記述

行動結果観測構造と選択帰属構造が既に保持している記録群を横断的に読み取り、
その構造的多様性の度合いを非評価的・非方向的に記述する集約層。

設計原則 (design_behavioral_diversity_description.md 準拠):
- 行動結果観測構造および選択帰属構造からのREAD-ONLY横断読み取り
- 読み取り結果からの断面構成（多様度の構造的特徴の記述）
- 断面履歴のFIFO蓄積
- 内省系モジュールへのREAD-ONLYアクセサ
- 記述するのは「多様性」そのものであり「頻度」ではない
- 「断面キーAがN回出現した」ではなく「種類数」のみを扱う
- 個別のキーやラベルを名指しで特徴づける記述を生成しない

3断面:
  - 断面A: 結果断面キーの種類数記述（段階値）
  - 断面B: 選択ラベルの種類数記述（段階値）
  - 断面C: 候補群サイズの分散度記述（段階値）

安全弁（8種）:
  1. 全記録等価 — 蓄積リスト内の全断面記録に重み・スコア・優先度を付与しない
  2. パターン抽出禁止 — 蓄積された記録から傾向、周期性、統計量を算出しない
  3. enrichment直接露出遮断 — 本機能の出力をget_prompt_enrichment()に含めない
     get_enrichment_data()メソッドを持たない
  4. 忘却経路遮断 — 本機能の出力を記憶の忘却・固定化処理の入力に使用しない
  5. 想起経路遮断 — 本機能の出力を記憶の想起候補の選択に影響させない
  6. 頻度情報の構造的不在 — 個別の断面キーやポリシーラベルの出現回数を
     算出・保持する処理を持たない。種類数のみ
  7. 出力経路不拡張 — 初期実装で定義した出力経路以外の経路を追加しない
  8. 既存モジュール安全弁の維持保証 — 行動結果観測構造の4安全弁、
     選択帰属構造の5遮断をいずれも維持し、これらの安全弁が保護する
     経路遮断を迂回する構造を作らない
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BehavioralDiversityConfig:
    """行動多様性記述モジュールの設定。"""

    # 蓄積リストの最大保持件数（安全弁: 有限性、最古押し出しが唯一の消失経路）
    max_history: int = 30


# =============================================================================
# 段階値の定義（列挙型）
# =============================================================================
# 設計書: 段階値は種類数の範囲区間を定義する列挙型とする。
# 各段階値には名称のみが付与され、「良い段階」「悪い段階」のような
# 評価的含意を持たない。
# 段階値の区間幅は均等とし、特定の区間に注目させる偏りを設けない。
# 段階値の数は固定とし、動的に追加・削除しない。

class TypeCountLevel(Enum):
    """種類数の段階値（断面A, 断面B共通）。
    全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。
    区間幅は均等（各5種類幅）。
    """
    LEVEL_0 = "level_0"      # 0種類
    LEVEL_1_5 = "level_1_5"  # 1-5種類
    LEVEL_6_10 = "level_6_10"  # 6-10種類
    LEVEL_11_15 = "level_11_15"  # 11-15種類
    LEVEL_16_PLUS = "level_16_plus"  # 16種類以上


class DispersionLevel(Enum):
    """候補群サイズの分散度段階値（断面C）。
    全記録で候補群サイズが同一であるか、複数の異なるサイズが混在しているかのみを記述する。
    サイズの大小に評価を付与しない。
    全段階値は等価であり、いずれの段階値も他の段階値より望ましいとしない。
    """
    EMPTY = "empty"          # 記録なし
    UNIFORM = "uniform"      # 全記録で候補群サイズが同一
    LOW = "low"              # 2種類のサイズが混在
    MODERATE = "moderate"    # 3-4種類のサイズが混在
    HIGH = "high"            # 5種類以上のサイズが混在


# =============================================================================
# 段階値の決定関数
# =============================================================================

def determine_type_count_level(type_count: int) -> TypeCountLevel:
    """種類数から段階値を決定する。

    段階値の区間幅は均等。評価的含意を持たない。

    安全弁6: 頻度情報の構造的不在。
    この関数は「種類数」のみを入力とし、各種類の出現回数は入力に含めない。

    Args:
        type_count: 種類の数（0以上の整数）

    Returns:
        段階値（全て等価）
    """
    if type_count == 0:
        return TypeCountLevel.LEVEL_0
    elif type_count <= 5:
        return TypeCountLevel.LEVEL_1_5
    elif type_count <= 10:
        return TypeCountLevel.LEVEL_6_10
    elif type_count <= 15:
        return TypeCountLevel.LEVEL_11_15
    else:
        return TypeCountLevel.LEVEL_16_PLUS


def determine_dispersion_level(distinct_size_count: int) -> DispersionLevel:
    """候補群サイズの異なる種類数から分散度段階値を決定する。

    段階値は等価。サイズの大小に評価を付与しない。

    Args:
        distinct_size_count: 候補群サイズの異なる種類の数（0以上の整数）

    Returns:
        分散度段階値（全て等価）
    """
    if distinct_size_count == 0:
        return DispersionLevel.EMPTY
    elif distinct_size_count == 1:
        return DispersionLevel.UNIFORM
    elif distinct_size_count == 2:
        return DispersionLevel.LOW
    elif distinct_size_count <= 4:
        return DispersionLevel.MODERATE
    else:
        return DispersionLevel.HIGH


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DiversityRecord:
    """断面記録。ある時点で読み取った3断面の非評価的記録。

    断面は生成後に変更されない（不変）。
    安全弁1: 全記録等価。記録間の重み・スコア・優先度を付与しない。
    安全弁2: パターン抽出禁止。記録間の比較・差分・相関の計算を行わない。
    安全弁6: 頻度情報の構造的不在。個別キーの出現回数を含まない。
    """

    # 断面A: 結果断面キー種類数の段階値（列挙型の値）
    section_key_type_count_level: str = TypeCountLevel.LEVEL_0.value

    # 断面B: 選択ラベル種類数の段階値（列挙型の値）
    policy_label_type_count_level: str = TypeCountLevel.LEVEL_0.value

    # 断面C: 候補群サイズ分散度の段階値（列挙型の値）
    candidate_size_dispersion_level: str = DispersionLevel.EMPTY.value

    # 記録時点のティック番号
    tick: int = 0

    # 生成時刻
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_key_type_count_level": self.section_key_type_count_level,
            "policy_label_type_count_level": self.policy_label_type_count_level,
            "candidate_size_dispersion_level": self.candidate_size_dispersion_level,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiversityRecord":
        return cls(
            section_key_type_count_level=data.get(
                "section_key_type_count_level", TypeCountLevel.LEVEL_0.value
            ),
            policy_label_type_count_level=data.get(
                "policy_label_type_count_level", TypeCountLevel.LEVEL_0.value
            ),
            candidate_size_dispersion_level=data.get(
                "candidate_size_dispersion_level", DispersionLevel.EMPTY.value
            ),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", 0.0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class BehavioralDiversityState:
    """行動多様性記述モジュールの内部状態。

    安全弁1: 全記録等価。蓄積リスト内の全記録に重み・スコア・優先度を付与しない。
    安全弁2: パターン抽出禁止。蓄積された記録から傾向等を算出しない。
    """

    # 蓄積リスト（時系列順、先入先出）
    history: list[DiversityRecord] = field(default_factory=list)

    # 直前の断面記録のキャッシュ（参照受渡用）
    latest_record: Optional[DiversityRecord] = None

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
    def from_dict(cls, data: dict[str, Any]) -> "BehavioralDiversityState":
        history = [
            DiversityRecord.from_dict(r)
            for r in data.get("history", [])
        ]
        lr_data = data.get("latest_record")
        latest = DiversityRecord.from_dict(lr_data) if lr_data else None
        return cls(
            history=history,
            latest_record=latest,
            total_records_generated=data.get("total_records_generated", 0),
            total_records_expired=data.get("total_records_expired", 0),
        )


# =============================================================================
# 第1段: 読み取り — 2構造からREAD-ONLYで種別情報を読み取る
# =============================================================================
# 設計書: 本機能は新たな計測ロジックを追加しない。
# 既存の2構造が内部に保持している記録群を読み取り専用で参照する。

def read_section_key_types(
    *,
    action_result_state: Any = None,
) -> int:
    """行動結果観測構造から可視状態の記録群に含まれる断面キーの種別数を読み取る。

    READ-ONLY。読み取り元の状態を変更しない。

    設計書:
    - 蓄積リスト内の可視状態にある記録に限定する
    - 希薄化中・不可視化済みの記録は読み取り対象外
    - 読み取るのは「どの断面キーが存在するか」の種別情報のみ
    - 各断面キーの出現回数、鮮度状態、断面の内容値は読み取らない

    安全弁6: 頻度情報の構造的不在。出現回数を算出しない。
    安全弁8: 既存モジュール安全弁の維持保証。読み取り専用。

    Args:
        action_result_state: ActionResultObservationProcessor（duck typing）

    Returns:
        断面キーの種類数（整数）。状態がなければ0。
    """
    if action_result_state is None:
        return 0

    # duck typing: pairs 属性を取得
    pairs = getattr(action_result_state, "pairs", None)
    if pairs is None:
        # Processor の場合 state.pairs
        state_obj = getattr(action_result_state, "state", None)
        if state_obj is not None:
            pairs = getattr(state_obj, "pairs", [])
        else:
            pairs = []

    # 可視状態のフィルタリング: invisible, buffered を除外
    # 設計書: 希薄化中・不可視化済みの記録は読み取り対象外
    visible_statuses = {"composed", "active", "decaying", "pending"}

    # 種別の収集: set を使って種類のみを追跡
    # 安全弁6: 出現回数を算出しない。set に追加するのみ
    section_key_types: set[str] = set()

    for pair in pairs:
        pair_status = getattr(pair, "status", "")
        if pair_status not in visible_statuses:
            continue

        result = getattr(pair, "result", None)
        if result is None:
            continue

        sections = getattr(result, "sections", [])
        for sec in sections:
            sec_key = getattr(sec, "section", "")
            if sec_key:
                section_key_types.add(sec_key)

    return len(section_key_types)


def read_policy_label_types(
    *,
    selection_attribution_state: Any = None,
) -> int:
    """選択帰属構造からポリシーラベルの種別数を読み取る。

    READ-ONLY。読み取り元の状態を変更しない。

    設計書:
    - 読み取るのは「どのポリシーラベルが蓄積内に存在するか」の種別情報のみ
    - 各ラベルの出現回数は読み取らない

    安全弁6: 頻度情報の構造的不在。出現回数を算出しない。
    安全弁8: 既存モジュール安全弁の維持保証。読み取り専用。

    Args:
        selection_attribution_state: SelectionAttributionRecorder（duck typing）

    Returns:
        ポリシーラベルの種類数（整数）。状態がなければ0。
    """
    if selection_attribution_state is None:
        return 0

    # duck typing: records を取得
    records = getattr(selection_attribution_state, "records", None)
    if records is None:
        # Recorder の場合 state.records
        state_obj = getattr(selection_attribution_state, "state", None)
        if state_obj is not None:
            records = getattr(state_obj, "records", [])
        else:
            records = []

    # 安全弁6: 出現回数を算出しない。set に追加するのみ
    label_types: set[str] = set()

    for record in records:
        label = getattr(record, "selected_policy_label", "")
        if label:
            label_types.add(label)

    return len(label_types)


def read_candidate_size_types(
    *,
    selection_attribution_state: Any = None,
) -> int:
    """選択帰属構造から候補群サイズの異なる種類数を読み取る。

    READ-ONLY。読み取り元の状態を変更しない。

    設計書:
    - 各記録の候補群構成から候補群のサイズ（候補がいくつ存在したかの整数）を読み取る
    - 全記録で候補群サイズが同一であるか、複数の異なるサイズが混在しているかのみを記述

    安全弁6: 頻度情報の構造的不在。出現回数を算出しない。

    Args:
        selection_attribution_state: SelectionAttributionRecorder（duck typing）

    Returns:
        候補群サイズの異なる種類の数（整数）。状態がなければ0。
    """
    if selection_attribution_state is None:
        return 0

    # duck typing: records を取得
    records = getattr(selection_attribution_state, "records", None)
    if records is None:
        state_obj = getattr(selection_attribution_state, "state", None)
        if state_obj is not None:
            records = getattr(state_obj, "records", [])
        else:
            records = []

    # 安全弁6: 出現回数を算出しない。set に追加するのみ
    size_types: set[int] = set()

    for record in records:
        count = getattr(record, "candidate_count", 0)
        size_types.add(count)

    return len(size_types)


# =============================================================================
# 第2段: 断面構成 — 3断面をひとまとまりの記録として構成する
# =============================================================================

def compose_record(
    section_key_type_count: int,
    policy_label_type_count: int,
    candidate_size_type_count: int,
    tick: int = 0,
    timestamp: Optional[float] = None,
) -> DiversityRecord:
    """3つの断面をひとまとまりの記録として構成する。

    各断面の値はそのまま記録する。断面間の重み付け、比較、統合、
    優先順位付けは行わない。3つの断面は並置されるのみであり、
    断面間の関係性を記述・推論しない。

    安全弁1: 全記録等価。
    安全弁2: パターン抽出禁止。
    安全弁6: 頻度情報の構造的不在。種類数のみ。

    Args:
        section_key_type_count: 結果断面キーの種類数
        policy_label_type_count: ポリシーラベルの種類数
        candidate_size_type_count: 候補群サイズの異なる種類数
        tick: ティック番号
        timestamp: 生成時刻（None時は現在時刻）

    Returns:
        構成された断面記録（不変）
    """
    ts = timestamp if timestamp is not None else time.time()

    return DiversityRecord(
        section_key_type_count_level=determine_type_count_level(
            section_key_type_count
        ).value,
        policy_label_type_count_level=determine_type_count_level(
            policy_label_type_count
        ).value,
        candidate_size_dispersion_level=determine_dispersion_level(
            candidate_size_type_count
        ).value,
        tick=tick,
        timestamp=ts,
    )


# =============================================================================
# 第3段: 蓄積と参照受渡 — FIFO蓄積 + READ-ONLYアクセサ
# =============================================================================

def accumulate_record(
    state: BehavioralDiversityState,
    record: DiversityRecord,
    config: Optional[BehavioralDiversityConfig] = None,
) -> BehavioralDiversityState:
    """構成された記録をFIFO蓄積に追加する。

    蓄積上限を超えた場合、最古の記録が押し出されて消失する（唯一の消失経路）。
    安全弁1: 全記録等価。特定の記録を強調、保護、優先的に保持する機構を持たない。

    Args:
        state: 現在の状態
        record: 新しい断面記録
        config: 設定

    Returns:
        更新されたBehavioralDiversityState
    """
    cfg = config or BehavioralDiversityConfig()

    new_history = list(state.history)
    new_history.append(record)

    # FIFO: 上限超過時は最古押し出し（唯一の消失経路）
    expired = 0
    if len(new_history) > cfg.max_history:
        expired = len(new_history) - cfg.max_history
        new_history = new_history[expired:]

    return BehavioralDiversityState(
        history=new_history,
        latest_record=record,
        total_records_generated=state.total_records_generated + 1,
        total_records_expired=state.total_records_expired + expired,
    )


# =============================================================================
# メイン処理: 1サイクル処理
# =============================================================================

def process_behavioral_diversity(
    state: BehavioralDiversityState,
    *,
    action_result_state: Any = None,
    selection_attribution_state: Any = None,
    tick: int = 0,
    config: Optional[BehavioralDiversityConfig] = None,
    timestamp: Optional[float] = None,
) -> BehavioralDiversityState:
    """行動多様性記述の1サイクル処理を実行する。

    安全弁1: 全記録等価
    安全弁2: パターン抽出禁止
    安全弁3: enrichment直接露出遮断（本関数はenrichment出力を持たない）
    安全弁4: 忘却経路遮断（本関数は忘却パイプラインへの出力経路を持たない）
    安全弁5: 想起経路遮断（本関数は想起経路への出力経路を持たない）
    安全弁6: 頻度情報の構造的不在（種類数のみ）
    安全弁7: 出力経路不拡張（出力先は内省系READ-ONLYアクセサのみ）
    安全弁8: 既存モジュール安全弁の維持保証

    入力構造の値は読み取り専用。書き込み能力を付与しない。
    段階値の計算は、その時点の読み取り値のみに依存する。
    過去の計算結果が現在の計算に影響を与える累積構造を持たない。

    Args:
        state: 現在の状態
        action_result_state: 行動結果観測構造（READ-ONLY参照）
        selection_attribution_state: 選択帰属構造（READ-ONLY参照）
        tick: ティック番号
        config: 設定
        timestamp: 断面の生成時刻

    Returns:
        更新されたBehavioralDiversityState
    """
    cfg = config or BehavioralDiversityConfig()

    # ── 第1段: 読み取り ──
    section_key_type_count = read_section_key_types(
        action_result_state=action_result_state,
    )

    policy_label_type_count = read_policy_label_types(
        selection_attribution_state=selection_attribution_state,
    )

    candidate_size_type_count = read_candidate_size_types(
        selection_attribution_state=selection_attribution_state,
    )

    # ── 第2段: 断面構成 ──
    record = compose_record(
        section_key_type_count=section_key_type_count,
        policy_label_type_count=policy_label_type_count,
        candidate_size_type_count=candidate_size_type_count,
        tick=tick,
        timestamp=timestamp,
    )

    # ── 第3段: 蓄積と参照受渡 ──
    new_state = accumulate_record(state, record, cfg)

    logger.debug(
        "Behavioral diversity: section_keys=%s, labels=%s, dispersion=%s, "
        "history=%d, expired=%d",
        record.section_key_type_count_level,
        record.policy_label_type_count_level,
        record.candidate_size_dispersion_level,
        len(new_state.history),
        new_state.total_records_expired - state.total_records_expired,
    )

    return new_state


# =============================================================================
# 内省系参照経路（READ-ONLY出力）
# =============================================================================
# 安全弁7: 出力先は内省系構造への参照情報に限定。
# enrichmentへの出力経路を持たない（安全弁3）。
# 忘却パイプラインへの出力経路を持たない（安全弁4）。
# 想起経路の選択への出力経路を持たない（安全弁5）。
# ポリシー選択への出力経路を持たない。
# 感情パイプラインへの出力経路を持たない。
# 責任計算への出力経路を持たない。
# 反復傾向構造への出力経路を持たない。

def get_latest_record(state: BehavioralDiversityState) -> Optional[DiversityRecord]:
    """最新の断面記録を返す（内省系参照経路、READ-ONLY）。

    参照行為によって状態が変化することはない。

    Returns:
        直近の断面記録。履歴がなければ None。
    """
    return state.latest_record


def get_record_history(state: BehavioralDiversityState) -> list[DiversityRecord]:
    """蓄積リスト全体を返す（内省系参照経路、READ-ONLY）。

    安全弁1: 全記録等価。フィルタリング・選別機能をアクセサに持たせない。
    安全弁2: パターン抽出禁止。蓄積された記録から傾向等を算出しない。
    参照行為によって状態が変化することはない。

    Returns:
        蓄積リストのコピー。
    """
    return list(state.history)


def get_diversity_summary(state: BehavioralDiversityState) -> dict[str, Any]:
    """内省系モジュール向けの参照情報サマリを返す（READ-ONLY）。

    安全弁1: 全記録等価。特定の記録を強調・選別しない。
    安全弁3: enrichment出力経路を持たない。
    安全弁6: 頻度情報の構造的不在。種類数の段階値のみ。
    安全弁7: 出力経路不拡張。

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
        summary["latest_section_key_type_count_level"] = latest.section_key_type_count_level
        summary["latest_policy_label_type_count_level"] = latest.policy_label_type_count_level
        summary["latest_candidate_size_dispersion_level"] = latest.candidate_size_dispersion_level
        summary["latest_tick"] = latest.tick
        summary["latest_timestamp"] = latest.timestamp

    return summary


# =============================================================================
# Save / Load
# =============================================================================

def save_state(state: BehavioralDiversityState) -> dict[str, Any]:
    """状態を永続化用の辞書に変換する。"""
    return state.to_dict()


def load_state(data: dict[str, Any]) -> BehavioralDiversityState:
    """永続化用の辞書から状態を復元する。"""
    return BehavioralDiversityState.from_dict(data)


# =============================================================================
# Factory
# =============================================================================

def create_behavioral_diversity_state() -> BehavioralDiversityState:
    """初期状態のファクトリ関数。"""
    return BehavioralDiversityState()


def create_behavioral_diversity_config(
    max_history: int = 30,
) -> BehavioralDiversityConfig:
    """設定のファクトリ関数。"""
    return BehavioralDiversityConfig(
        max_history=max_history,
    )
