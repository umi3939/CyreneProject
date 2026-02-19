"""
psyche/intent_action_gap.py - 意図-行動間の乖離認知

ポリシー選択（内部処理系統が選び取った方針）と、実際の外部出力（代弁者が生成したテキスト）の
間の差異を多断面で記述・蓄積する構造を提供する。

設計原則 (design_intent_action_gap.md 準拠):
- 乖離の「大きさ」を数値統合スコア化しない。乖離度・一致度・遵守率といった単一の
  数値指標を算出しない
- 「一致が望ましい」という規範を導入しない。差異は存在の記述であり、良否の判定ではない
- ポリシー選択への直接フィードバックを行わない。「前回乖離したポリシーを回避する」
  「乖離の少ないポリシーを選好する」といった経路を構造的に禁止
- 出力テキストの品質評価・分類を行わない
- 閾値判定を行わない
- 感情パイプラインのパラメータを変更しない
- 自己矯正ループを形成しない
- すべての記録は等価であり、重み・スコア・重要度を付与しない
- パターン抽出・傾向化を禁止
- enrichment内での強調禁止

3段パイプライン:
1. 対の構成 (pair construction)
2. 多断面での差異記述 (multi-facet difference description)
3. 蓄積と参照情報の受渡準備 (accumulation and handoff preparation)
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
class GapRecord:
    """意図-行動間の乖離記録。多断面での差異記述を保持する。

    一度記録されたら変更されない（追記のみ）。
    重み・スコア・優先度・重要度などの評価的属性を持たない（全記録等価）。
    差異の有無・程度を本機能が判定することはない。
    """
    record_id: str = field(default_factory=_gen_id)

    # ラベル断面: ポリシーラベル（方針名）の文字列記録
    policy_label: str = ""

    # テキスト断面: 出力テキストの先頭部分（長さ上限あり）の記録
    text_snippet: str = ""

    # 時間断面: 対が構成された時点のティック番号
    tick: int = 0

    # 文脈断面: ポリシー選択時に参照された根拠情報の要約（利用可能な場合）
    context_info: str = ""

    # タイムスタンプ
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "policy_label": self.policy_label,
            "text_snippet": self.text_snippet,
            "tick": self.tick,
            "context_info": self.context_info,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GapRecord":
        return cls(
            record_id=data.get("record_id", _gen_id()),
            policy_label=data.get("policy_label", ""),
            text_snippet=data.get("text_snippet", ""),
            tick=data.get("tick", 0),
            context_info=data.get("context_info", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class IntentActionGapState:
    """意図-行動間の乖離認知の内部状態。"""

    # 乖離記録リスト: 時系列順に蓄積される多断面記録群
    records: list[GapRecord] = field(default_factory=list)

    # 直近乖離記録: 最新の1件を直接参照可能な形で保持
    latest_record: Optional[GapRecord] = None

    # スキップ計数: 対構成に失敗した回数の累積（診断情報のみ、処理分岐に影響しない）
    skip_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "latest_record": self.latest_record.to_dict() if self.latest_record else None,
            "skip_count": self.skip_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentActionGapState":
        records = [
            GapRecord.from_dict(r)
            for r in data.get("records", [])
        ]
        latest_data = data.get("latest_record")
        latest = GapRecord.from_dict(latest_data) if latest_data else None
        return cls(
            records=records,
            latest_record=latest,
            skip_count=data.get("skip_count", 0),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IntentActionGapConfig:
    """設定。"""
    # 乖離記録の最大保持数（上限到達時は最古から押し出す）
    max_records: int = 50

    # テキスト断面の先頭部分の長さ上限
    text_snippet_max_length: int = 150

    # enrichment に含める直近記録数
    enrichment_recent_count: int = 3


# =============================================================================
# Recorder (3-stage pipeline)
# =============================================================================

class IntentActionGapRecorder:
    """意図-行動間の乖離認知レコーダー。

    3段パイプライン:
    1. 対の構成 — 自己行動知覚の記録からポリシーラベルと出力テキストの対を取り出す
    2. 多断面での差異記述 — ラベル断面・テキスト断面・時間断面・文脈断面で記述
    3. 蓄積と参照情報の受渡準備 — 時系列蓄積 + enrichment + READ-ONLYアクセサ

    すべての処理は記述的観測行為であり、判断・評価・修正を含まない。
    乖離の有無・程度を判定しない。
    ポリシー選択パイプラインへの逆方向経路を持たない。
    """

    def __init__(self, config: Optional[IntentActionGapConfig] = None):
        self._config = config or IntentActionGapConfig()
        self._state = IntentActionGapState()

    @property
    def state(self) -> IntentActionGapState:
        return self._state

    @state.setter
    def state(self, value: IntentActionGapState) -> None:
        self._state = value

    # ─── Stage 1: 対の構成 ───────────────────────────────────────

    def process_action_record(
        self,
        response_text: str,
        policy_label: str,
        tick: int = 0,
        context_info: str = "",
    ) -> Optional[GapRecord]:
        """自己行動知覚の記録からポリシーラベルと出力テキストの対を構成し、
        多断面で記述・蓄積する。

        対構成に失敗した場合（ポリシーラベルまたは出力テキストが欠損している場合）は
        スキップする。スキップの事実は記録されるが、スキップの理由を分析する処理は行わない。

        Args:
            response_text: 自己行動知覚が保持する応答テキスト（生テキスト）
            policy_label: 同一ターンで選択されたポリシーラベル
            tick: 対が構成された時点のティック番号
            context_info: ポリシー選択時に参照された根拠情報（利用可能な場合）

        Returns:
            生成されたGapRecord。スキップした場合はNone。
        """
        # Stage 1: 対の構成
        # ポリシーラベルまたは出力テキストが欠損している場合はスキップ
        if not response_text or not policy_label:
            self._state.skip_count += 1
            logger.debug(
                "Pair construction skipped: response_text=%s, policy_label=%s (skip_count=%d)",
                bool(response_text), bool(policy_label), self._state.skip_count,
            )
            return None

        # Stage 2: 多断面での差異記述
        record = self._describe_multi_facet(
            response_text=response_text,
            policy_label=policy_label,
            tick=tick,
            context_info=context_info,
        )

        # Stage 3: 蓄積と参照情報の受渡準備
        self._accumulate(record)

        return record

    # ─── Stage 2: 多断面での差異記述 ────────────────────────────

    def _describe_multi_facet(
        self,
        response_text: str,
        policy_label: str,
        tick: int,
        context_info: str,
    ) -> GapRecord:
        """構成された対について、多断面で差異を記述する。

        各断面は独立した記述であり、断面間の優先順位・重み付け・統合処理は存在しない。
        差異の有無・程度を本機能が判定することはない。
        """
        cfg = self._config
        now = time.time()

        # テキスト断面: 出力テキストの先頭部分（長さ上限あり）の記録
        text_snippet = response_text[:cfg.text_snippet_max_length]

        record = GapRecord(
            policy_label=policy_label,
            text_snippet=text_snippet,
            tick=tick,
            context_info=context_info,
            timestamp=now,
        )

        return record

    # ─── Stage 3: 蓄積と参照情報の受渡準備 ────────────────────────

    def _accumulate(self, record: GapRecord) -> None:
        """多断面記録を時系列順に蓄積する。

        蓄積リストは上限を持ち、上限到達時は最古の記録から押し出される。
        特定の記録を選択的に消去する処理は存在しない。
        """
        self._state.records.append(record)
        self._state.latest_record = record

        # 上限による押し出し（唯一の消失経路）
        self._apply_pushout()

        logger.debug(
            "Gap record accumulated: tick=%d, policy=%s, records=%d",
            record.tick, record.policy_label, len(self._state.records),
        )

    def _apply_pushout(self) -> None:
        """上限数に達した場合、最古の記録から押し出す。

        上限到達時の最古押し出しが唯一の消失経路。
        特定の記録を永続的に保持して固定的自己像の形成基盤を作らない。
        """
        cfg = self._config
        if len(self._state.records) > cfg.max_records:
            pushout_count = len(self._state.records) - cfg.max_records
            self._state.records = self._state.records[pushout_count:]

    # ─── 参照情報の提供 ──────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        直近の数件について、ポリシーラベルとテキスト断片の対を等価に列挙する。
        列挙に際して特定の記録を強調・選別しない。
        「重要な乖離」「注目すべき差異」等の強調表現を使わない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state
        cfg = self._config

        recent = st.records[-cfg.enrichment_recent_count:] if st.records else []

        recent_entries: list[dict[str, Any]] = []
        for rec in recent:
            recent_entries.append({
                "policy_label": rec.policy_label,
                "text_snippet": rec.text_snippet,
                "tick": rec.tick,
            })

        summary_text = get_gap_summary(st)

        return {
            "record_count": len(st.records),
            "skip_count": st.skip_count,
            "recent_entries": recent_entries,
            "summary_text": summary_text,
        }

    def get_all_records(self) -> list[GapRecord]:
        """蓄積リスト全体をREAD-ONLYで返す。

        内省系モジュールがREAD-ONLYで参照可能な蓄積リスト。
        フィルタリング・選別機能をアクセサに持たせない。
        蓄積リスト全体を返す。
        参照行為によって記録が変化することはない。

        Returns:
            蓄積された乖離記録のリスト（コピー）
        """
        return list(self._state.records)

    def get_latest_record(self) -> Optional[GapRecord]:
        """最新の1件を直接参照可能な形で返す。

        直前のターンの意図-行動対を即座に参照するための構造。

        Returns:
            最新の乖離記録。記録がなければNone。
        """
        return self._state.latest_record

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "record_count": len(st.records),
            "skip_count": st.skip_count,
            "has_latest": st.latest_record is not None,
            "latest_tick": st.latest_record.tick if st.latest_record else 0,
            "latest_policy": st.latest_record.policy_label if st.latest_record else "",
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_gap_summary(state: IntentActionGapState) -> str:
    """意図-行動乖離状態の要約（enrichment用）。

    全ての記録を等価に列挙し、特定の記録を強調しない。
    「重要な乖離」「注目すべき差異」等の強調表現を使わない。
    評価判定・行動指示を含まない。
    """
    if not state.records and state.skip_count == 0:
        return "意図-行動対: 待機中"

    parts: list[str] = []
    parts.append(f"記録数={len(state.records)}")

    if state.latest_record is not None:
        rec = state.latest_record
        if rec.policy_label:
            parts.append(f"直近ポリシー={rec.policy_label}")
        if rec.text_snippet:
            parts.append(f"直近テキスト長={len(rec.text_snippet)}")
        parts.append(f"tick={rec.tick}")

    if state.skip_count > 0:
        parts.append(f"スキップ={state.skip_count}")

    return " ".join(parts) if parts else "意図-行動対: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_intent_action_gap_recorder(
    config: Optional[IntentActionGapConfig] = None,
) -> IntentActionGapRecorder:
    """IntentActionGapRecorder のファクトリ関数。"""
    return IntentActionGapRecorder(config=config)
