"""
psyche/self_action_perception.py - 自己行動知覚

Geminiの応答テキストをpsyche内部に通知し、「自分が外部に出した表出テキスト」を
内部参照可能にする構造を提供する。

設計原則 (design_self_action_perception.md 準拠):
- 応答テキストの品質評価を行わない。「良い発言」「悪い発言」「適切な発言」といった
  評価判定は本機能の範囲外
- 応答テキストに基づいて判断・行動を直接変更しない
- Geminiの出力を事後検閲・修正しない
- ポリシー遵守度の判定を行わない
- 出力パターンを分類・類型化しない
- 感情パイプライン（Phase 1-2）のパラメータを変更しない
- すべての記録は等価であり、重み・スコア・優先度を付与しない
- テキストは生のテキスト文字列として保持し、意味・傾向・パターンを抽出しない
- 判断系（ポリシー選択・バイアス計算・安定化弁）への直接入力経路を禁止

3段パイプライン:
1. 受領と保持 (receive and retain)
2. 既存構造への情報補完 (complement to existing structures)
3. 参照情報としての受渡準備 (handoff preparation)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RecordStatus(Enum):
    """自己行動記録の状態。"""
    ACTIVE = "active"
    PUSHED_OUT = "pushed_out"


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SelfActionRecord:
    """自己行動記録。応答テキスト・ポリシーラベル・ティック番号・タイムスタンプの対。

    一度記録されたら変更されない（追記のみ）。
    重み・スコア・優先度などの評価的属性を持たない（全記録等価）。
    テキストの解釈・分類・類型化は行わない（テキスト非解釈）。
    """
    record_id: str = field(default_factory=_gen_id)
    response_text: str = ""
    policy_label: str = ""
    tick: int = 0
    timestamp: float = field(default_factory=time.time)
    status: str = RecordStatus.ACTIVE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "response_text": self.response_text,
            "policy_label": self.policy_label,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfActionRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            response_text=data.get("response_text", ""),
            policy_label=data.get("policy_label", ""),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
            status=data.get("status", RecordStatus.ACTIVE.value),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class SelfActionPerceptionState:
    """自己行動知覚の内部状態。"""
    # 自己行動記録のリスト（時系列順に蓄積）
    records: list[SelfActionRecord] = field(default_factory=list)

    # 直近自己行動記録（最新の1件を直接参照可能な形で保持）
    latest_record: Optional[SelfActionRecord] = None

    # カウンタ
    total_records_received: int = 0
    total_records_pushed_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "latest_record": self.latest_record.to_dict() if self.latest_record else None,
            "total_records_received": self.total_records_received,
            "total_records_pushed_out": self.total_records_pushed_out,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfActionPerceptionState":
        records = [
            SelfActionRecord.from_dict(r)
            for r in data.get("records", [])
        ]
        latest_data = data.get("latest_record")
        latest = SelfActionRecord.from_dict(latest_data) if latest_data else None
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
class SelfActionPerceptionConfig:
    """設定。"""
    # 自己行動記録の最大保持数（上限到達時は最古から押し出す）
    max_records: int = 50

    # enrichment に含める直近記録数
    enrichment_recent_count: int = 3

    # 参照情報として提供する直近履歴の件数
    reference_history_count: int = 10


# =============================================================================
# Recorder (3-stage pipeline)
# =============================================================================

class SelfActionPerceptionRecorder:
    """自己行動知覚レコーダー。

    3段パイプライン:
    1. 受領と保持 — 応答テキストとポリシーラベルを対にして保持
    2. 既存構造への情報補完 — action_result_observationの入力にテキスト情報を追加
    3. 参照情報としての受渡準備 — enrichmentセクション追加 + READ-ONLY参照

    すべての処理は受動的な記録行為であり、能動的な判断・評価・制御を含まない。
    応答テキストの品質評価、分類・類型化、ポリシー遵守度の判定を行わない。
    """

    def __init__(self, config: Optional[SelfActionPerceptionConfig] = None):
        self._config = config or SelfActionPerceptionConfig()
        self._state = SelfActionPerceptionState()

    @property
    def state(self) -> SelfActionPerceptionState:
        return self._state

    @state.setter
    def state(self, value: SelfActionPerceptionState) -> None:
        self._state = value

    # ─── Stage 1: 受領と保持 ───────────────────────────────────────

    def receive_response(
        self,
        response_text: str,
        policy_label: str = "",
        tick: int = 0,
    ) -> SelfActionRecord:
        """思考エンジンからの通知を受領し、自己行動記録を保持する。

        応答テキストとポリシーラベルを対にした記録を内部に保持する。
        記録は時系列順に蓄積され、上限数に達した場合は最古のものから押し出される。

        沈黙選択時（出力テキストなし）は通知されないことを前提とする。
        空テキストが渡された場合は記録しない。

        Args:
            response_text: Geminiの応答テキスト（生のテキスト文字列）
            policy_label: そのターンで選択されていたポリシーラベル
            tick: 通知時点のティック番号

        Returns:
            生成された自己行動記録
        """
        # 空テキストは記録しない（沈黙時は通知されない前提）
        if not response_text:
            logger.debug("Empty response text, skipping self-action record")
            return SelfActionRecord()

        now = time.time()

        record = SelfActionRecord(
            response_text=response_text,
            policy_label=policy_label,
            tick=tick,
            timestamp=now,
        )

        self._state.records.append(record)
        self._state.latest_record = record
        self._state.total_records_received += 1

        # 上限による押し出し（唯一の消失経路）
        self._apply_pushout()

        logger.debug(
            "Self-action recorded: tick=%d, policy=%s, text_len=%d",
            tick, policy_label, len(response_text),
        )

        return record

    def _apply_pushout(self) -> None:
        """上限数に達した場合、最古の記録から押し出す。

        特定の記録を永続的に保持する機能はない。
        上限到達時の最古押し出しが唯一の消失経路。
        """
        cfg = self._config
        if len(self._state.records) > cfg.max_records:
            pushout_count = len(self._state.records) - cfg.max_records
            for old in self._state.records[:pushout_count]:
                old.status = RecordStatus.PUSHED_OUT.value
            self._state.records = self._state.records[pushout_count:]
            self._state.total_records_pushed_out += pushout_count

    # ─── Stage 2: 既存構造への情報補完 ────────────────────────────

    def get_text_for_action_result(self) -> str:
        """直近の自己行動記録のテキストを返す。

        行動-結果観測の「行動」情報に「実際に出力されたテキスト」を補完するために使用される。
        行動-結果観測の既存データ構造に「テキスト断面」を追加する形。
        新しい断面は既存の8断面と同列であり、優先順位を持たない。

        Returns:
            直近の出力テキスト。記録がなければ空文字列。
        """
        if self._state.latest_record is not None:
            return self._state.latest_record.response_text
        return ""

    def get_policy_for_action_result(self) -> str:
        """直近の自己行動記録のポリシーラベルを返す。

        Returns:
            直近のポリシーラベル。記録がなければ空文字列。
        """
        if self._state.latest_record is not None:
            return self._state.latest_record.policy_label
        return ""

    # ─── Stage 3: 参照情報としての受渡準備 ────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        直近の自己行動の要約的記述を提供する。
        直近の応答テキストとポリシーラベルの対を簡潔に記述する。
        提供情報は記述形式であり、評価判定・行動指示を含まない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state
        cfg = self._config

        # 直近の記録を取得
        recent = st.records[-cfg.enrichment_recent_count:] if st.records else []

        recent_entries: list[dict[str, Any]] = []
        for rec in recent:
            # テキストは長い場合があるため要約的に切り詰め
            text_preview = rec.response_text[:100] if rec.response_text else ""
            if len(rec.response_text) > 100:
                text_preview += "..."
            recent_entries.append({
                "policy_label": rec.policy_label,
                "text_preview": text_preview,
                "tick": rec.tick,
            })

        summary_text = get_self_action_summary(st)

        return {
            "total_records": st.total_records_received,
            "current_record_count": len(st.records),
            "recent_entries": recent_entries,
            "summary_text": summary_text,
        }

    def get_reference_history(self) -> list[SelfActionRecord]:
        """内省系モジュールがREAD-ONLY参照として利用する時系列的な自己行動履歴。

        直近数件の自己行動記録のリストを返す。
        参照行為によって記録が変化することはない。

        Returns:
            直近の自己行動記録のリスト（READ-ONLY参照）
        """
        cfg = self._config
        return list(self._state.records[-cfg.reference_history_count:])

    def get_latest_record(self) -> Optional[SelfActionRecord]:
        """最新の1件を直接参照可能な形で返す。

        直前のターンで出力したテキストを即座に参照するための構造。

        Returns:
            最新の自己行動記録。記録がなければNone。
        """
        return self._state.latest_record

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "total_records_received": st.total_records_received,
            "current_record_count": len(st.records),
            "total_pushed_out": st.total_records_pushed_out,
            "has_latest": st.latest_record is not None,
            "latest_tick": st.latest_record.tick if st.latest_record else 0,
            "latest_policy": st.latest_record.policy_label if st.latest_record else "",
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_self_action_summary(state: SelfActionPerceptionState) -> str:
    """自己行動知覚状態の要約（enrichment用）。

    直近の自己行動を簡潔に記述する。
    評価判定・行動指示を含まない。
    """
    if state.total_records_received == 0:
        return "自己行動知覚: 待機中"

    parts: list[str] = []
    parts.append(f"記録数={state.total_records_received}")

    if state.latest_record is not None:
        rec = state.latest_record
        # ポリシーラベルの情報
        if rec.policy_label:
            parts.append(f"直近ポリシー={rec.policy_label}")
        # テキスト長の情報（テキスト内容は解釈しない）
        if rec.response_text:
            parts.append(f"直近テキスト長={len(rec.response_text)}")
        parts.append(f"tick={rec.tick}")

    active_count = sum(
        1 for r in state.records
        if r.status == RecordStatus.ACTIVE.value
    )
    if active_count:
        parts.append(f"活性={active_count}")

    if state.total_records_pushed_out > 0:
        parts.append(f"押出累計={state.total_records_pushed_out}")

    return " ".join(parts) if parts else "自己行動知覚: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_self_action_perception_recorder(
    config: Optional[SelfActionPerceptionConfig] = None,
) -> SelfActionPerceptionRecorder:
    """SelfActionPerceptionRecorder のファクトリ関数。"""
    return SelfActionPerceptionRecorder(config=config)
