"""
psyche/selection_attribution.py - 選択帰属

方針選択処理が実行された事実を、その時点の候補群の構成とともに記録し、
内部の他の構造が参照可能な情報として保持する。

設計原則 (design_selection_attribution.md 準拠):
- 記録された選択事実に基づいて、次回以降の方針選択を変更・誘導・加速・抑制しない。
  記録は方針選択処理への入力経路を持たない
- 選択の「正しさ」「適切さ」「一貫性」「妥当性」を評価しない
- 特定の選択パターンを検出・抽出・分類・類型化しない
- 選択の頻度・偏り・傾向を算出しない
- 記録に重み・スコア・優先度・重要度を付与しない。全記録は等価である
- 「よく選ばれるポリシー」「避けられるポリシー」といった統計的知見を導出しない
- 感情処理パイプラインのパラメータを変更しない
- 反復傾向の構造と機能的に重複する処理を行わない
- スコア情報は受領しない

経路遮断（5つ）:
1. 候補生成への経路遮断
2. バイアス計算への経路遮断
3. 安定化弁への経路遮断
4. 感情処理への経路遮断
5. 責任計算への経路遮断

安全弁（5つ）:
1. 全記録等価の保証
2. パターン抽出の禁止
3. 方針選択処理への経路遮断の不変性
4. enrichment内での等価列挙の維持
5. 蓄積上限による自然な入れ替わりの保証
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SelectionRecord:
    """選択記録。方針選択が実行された事実を一つの記録として保持する。

    一度構成された記録は不変であり、後から変更されない。
    重み・スコア・優先度・重要度を付与しない（全記録等価）。
    パターン抽出・傾向化は行わない。
    """
    record_id: str = field(default_factory=_gen_id)

    # 選択されたポリシーラベル（文字列）
    selected_policy_label: str = ""

    # 候補群のポリシーラベル一覧（文字列のリスト、候補数の上限あり、超過時は先頭から切り詰め）
    candidate_labels: list[str] = field(default_factory=list)

    # 候補群のサイズ（候補がいくつ存在したかの整数）
    candidate_count: int = 0

    # ティック番号（整数）
    tick: int = 0

    # タイムスタンプ（浮動小数点数）
    timestamp: float = field(default_factory=time.time)

    # 選択時に適用されていたバイアス源の名前一覧（名前のみ、スコア・重み・方向性は記録しない）
    bias_source_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "selected_policy_label": self.selected_policy_label,
            "candidate_labels": list(self.candidate_labels),
            "candidate_count": self.candidate_count,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "bias_source_labels": list(self.bias_source_labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectionRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            selected_policy_label=data.get("selected_policy_label", ""),
            candidate_labels=list(data.get("candidate_labels", [])),
            candidate_count=data.get("candidate_count", 0),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            bias_source_labels=list(data.get("bias_source_labels", [])),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class SelectionAttributionState:
    """選択帰属の内部状態。"""

    # 選択記録のリスト: 時系列順に蓄積される記録群
    records: list[SelectionRecord] = field(default_factory=list)

    # 直近記録の参照: 最新の一件を即座に参照可能な形で別途保持
    latest_record: Optional[SelectionRecord] = None

    # 累積カウンタ: 受領した記録の総数（診断情報のみ、処理分岐に使用しない）
    total_records_received: int = 0

    # 累積カウンタ: 押し出された記録の総数（診断情報のみ、処理分岐に使用しない）
    total_records_pushed_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "latest_record": self.latest_record.to_dict() if self.latest_record else None,
            "total_records_received": self.total_records_received,
            "total_records_pushed_out": self.total_records_pushed_out,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectionAttributionState":
        records = [
            SelectionRecord.from_dict(r)
            for r in data.get("records", [])
        ]
        latest_data = data.get("latest_record")
        latest = SelectionRecord.from_dict(latest_data) if latest_data else None
        return cls(
            records=records,
            latest_record=latest,
            total_records_received=data.get("total_records_received", 0),
            total_records_pushed_out=data.get("total_records_pushed_out", 0),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SelectionAttributionConfig:
    """設定。"""
    # 選択記録の最大保持数（上限到達時は最古から押し出す）
    max_records: int = 50

    # 候補群ラベル一覧の上限（超過時は先頭から切り詰め）
    max_candidate_labels: int = 20

    # enrichment に含める直近記録数
    enrichment_recent_count: int = 5

    # 内省系参照経路で提供する直近履歴の件数
    reference_history_count: int = 20


# =============================================================================
# Recorder
# =============================================================================

class SelectionAttributionRecorder:
    """選択帰属レコーダー。

    方針選択処理の「下流」に位置し、選択結果を受動的に受領する。
    方針選択処理の内部状態や、候補のスコア、バイアスの内訳といった
    選択過程の詳細は参照しない。受領するのは「何が候補にあり、何が選ばれたか」
    という事実のみである。

    出力経路:
    1. enrichment経路: 直近の選択事実を等価に列挙した参照情報を提供する。
       この情報は代弁処理が参照する文脈情報の一部として流れるのみであり、
       方針選択処理には到達しない。
    2. 内省系参照経路: 蓄積された記録のリストを読み取り専用で提供する。
       参照行為によって記録が変化することはない。

    いずれの経路も、方針選択処理（候補生成・バイアス計算・安定化・最終選択）
    への入力経路を構造的に持たない。
    """

    def __init__(self, config: Optional[SelectionAttributionConfig] = None):
        self._config = config or SelectionAttributionConfig()
        self._state = SelectionAttributionState()

    @property
    def state(self) -> SelectionAttributionState:
        return self._state

    @state.setter
    def state(self, value: SelectionAttributionState) -> None:
        self._state = value

    # ─── 記録の構成と蓄積 ──────────────────────────────────────

    def record_selection(
        self,
        selected_policy_label: str,
        candidate_labels: list[str],
        tick: int = 0,
        bias_source_labels: list[str] = None,
    ) -> SelectionRecord:
        """方針選択が実行された事実を記録する。

        選択結果と候補群の構成を一つの記録として構成する。
        構成された記録は不変であり、後から変更されない。

        スコア情報は受領しない。スコアの記録は、特定の選択に
        「合理的な根拠があった」という解釈を暗黙的に導入し、
        スコアの高い選択をスコアの低い選択より「良い選択」として
        区別する経路を開くため。

        バイアス源ラベルはスコア・重み・方向性を含まず、
        名前のみを事実として記録する。

        Args:
            selected_policy_label: 選び取られた候補のポリシーラベル
            candidate_labels: 選択時点で存在していた候補のポリシーラベル一覧
            tick: 選択が行われたティック番号
            bias_source_labels: 選択時に適用されていたバイアス源の名前一覧（オプション）

        Returns:
            生成された選択記録
        """
        now = time.time()
        cfg = self._config

        # 候補群ラベル一覧の上限適用（超過時は先頭から切り詰め）
        actual_count = len(candidate_labels)
        trimmed_labels = candidate_labels
        if len(trimmed_labels) > cfg.max_candidate_labels:
            trimmed_labels = trimmed_labels[:cfg.max_candidate_labels]

        record = SelectionRecord(
            selected_policy_label=selected_policy_label,
            candidate_labels=list(trimmed_labels),
            candidate_count=actual_count,
            tick=tick,
            timestamp=now,
            bias_source_labels=list(bias_source_labels) if bias_source_labels else [],
        )

        # 蓄積
        self._state.records.append(record)
        self._state.latest_record = record
        self._state.total_records_received += 1

        # 上限による押し出し（唯一の消失経路）
        self._apply_pushout()

        logger.debug(
            "Selection recorded: tick=%d, selected=%s, candidates=%d",
            tick, selected_policy_label, actual_count,
        )

        return record

    def _apply_pushout(self) -> None:
        """上限数に達した場合、最古の記録から押し出す。

        特定の記録を永続的に保持する機構は存在しない。
        すべての記録はいずれ押し出される。
        上限到達時の最古押し出しが唯一の消失経路。
        """
        cfg = self._config
        if len(self._state.records) > cfg.max_records:
            pushout_count = len(self._state.records) - cfg.max_records
            self._state.records = self._state.records[pushout_count:]
            self._state.total_records_pushed_out += pushout_count

    # ─── 直近記録の即時参照 ────────────────────────────────────

    def get_latest_record(self) -> Optional[SelectionRecord]:
        """最新の一件を即座に参照可能な形で返す。

        Returns:
            最新の選択記録。記録がなければNone。
        """
        return self._state.latest_record

    # ─── enrichment用の等価列挙 ─────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        直近N件の記録をそのまま等価に列挙する。
        列挙に際して特定の記録を強調・選別・要約しない。
        「注目すべき選択」「重要な選択」「特徴的な選択」等の強調語を含めない。
        頻度情報を付加しない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state
        cfg = self._config

        recent = st.records[-cfg.enrichment_recent_count:] if st.records else []

        recent_entries: list[dict[str, Any]] = []
        for rec in recent:
            recent_entries.append({
                "selected_policy_label": rec.selected_policy_label,
                "candidate_labels": rec.candidate_labels,
                "candidate_count": rec.candidate_count,
                "tick": rec.tick,
            })

        summary_text = get_selection_attribution_summary(st)

        return {
            "record_count": len(st.records),
            "recent_entries": recent_entries,
            "summary_text": summary_text,
        }

    # ─── 内省系参照経路 ────────────────────────────────────────

    def get_all_records(self) -> list[SelectionRecord]:
        """蓄積リスト全体をREAD-ONLYで返す。

        内省系モジュールがREAD-ONLYで参照可能な蓄積リスト。
        フィルタリング・選別機能をアクセサに持たせない。
        蓄積リスト全体を返す。
        参照行為によって記録が変化することはない。

        Returns:
            蓄積された選択記録のリスト（コピー）
        """
        return list(self._state.records)

    def get_reference_history(self) -> list[SelectionRecord]:
        """内省系参照経路として直近の記録リストを返す。

        読み取り専用で提供する。参照行為によって記録が変化することはない。

        Returns:
            直近の選択記録のリスト（READ-ONLY参照）
        """
        cfg = self._config
        return list(self._state.records[-cfg.reference_history_count:])

    # ─── サマリ ────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "total_records_received": st.total_records_received,
            "current_record_count": len(st.records),
            "total_pushed_out": st.total_records_pushed_out,
            "has_latest": st.latest_record is not None,
            "latest_tick": st.latest_record.tick if st.latest_record else 0,
            "latest_selected": st.latest_record.selected_policy_label if st.latest_record else "",
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_selection_attribution_summary(state: SelectionAttributionState) -> str:
    """選択帰属状態の要約（enrichment用）。

    全ての記録を等価に扱い、特定の記録を強調しない。
    頻度・偏り・傾向を算出しない。
    評価判定・行動指示を含まない。
    """
    if state.total_records_received == 0:
        return "選択帰属: 待機中"

    parts: list[str] = []
    parts.append(f"記録数={state.total_records_received}")

    if state.latest_record is not None:
        rec = state.latest_record
        if rec.selected_policy_label:
            parts.append(f"直近選択={rec.selected_policy_label}")
        parts.append(f"候補数={rec.candidate_count}")
        parts.append(f"tick={rec.tick}")

    active_count = len(state.records)
    if active_count:
        parts.append(f"蓄積={active_count}")

    if state.total_records_pushed_out > 0:
        parts.append(f"押出累計={state.total_records_pushed_out}")

    return " ".join(parts) if parts else "選択帰属: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_selection_attribution_recorder(
    config: Optional[SelectionAttributionConfig] = None,
) -> SelectionAttributionRecorder:
    """SelectionAttributionRecorder のファクトリ関数。"""
    return SelectionAttributionRecorder(config=config)
