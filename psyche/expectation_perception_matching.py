"""
psyche/expectation_perception_matching.py - 予期照合記述（予期-知覚マッチング）

予期形成の出力（ExpectationStore）と知覚の出力（Percept）を構造的フィールド比較し、
対応の程度を段階値で記述・蓄積する。

設計原則 (design_expectation_perception_matching.md 準拠):
- 予期の正誤判定を行わない。段階値記述のみ
- パターン抽出を行わない。統計集計メソッドを持たない
- 感情への直接接続を行わない
- 予期形成への書き込みを行わない（READ-ONLY）
- 行動・判断・ポリシー選択への直接接続を行わない
- 知覚の再解釈・修正を行わない
- 「驚き」「新奇性」「予測誤差」の概念化をしない

安全弁:
1. 記録上限によるFIFO脱落（MAX_RECORDS=50）
2. 段階値列挙の固定（CorrespondenceLevel: 5値）
3. 照合記録の不変性（frozen dataclass）
4. 入力欠落時の無処理
5. enrichment内での等価列挙
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MAX_RECORDS = 50
MAX_TEXT_LENGTH = 200
MAX_TOPICS_COUNT = 10
SENTIMENT_MIN = -1.0
SENTIMENT_MAX = 1.0


# =============================================================================
# Enums
# =============================================================================

class CorrespondenceLevel(Enum):
    """対応の程度を記述する5段階値。

    二値（一致/不一致）でも連続値でもなく、段階記述としてのみ存在する。
    価値判断語（CORRECT/MATCH/SUCCESS/FAILURE等）を含まない。
    """
    HIGH_CORRESPONDENCE = "high_correspondence"
    MODERATE_CORRESPONDENCE = "moderate_correspondence"
    LOW_CORRESPONDENCE = "low_correspondence"
    NO_CORRESPONDENCE = "no_correspondence"
    INDETERMINATE = "indeterminate"


# =============================================================================
# Data Structures
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class MatchingRecord:
    """照合記録1件。不変の多断面記述。

    予期スナップショット + 知覚スナップショット + 対応段階値 + 文脈情報。
    一度構成された記録は変更されない（追記のみ）。
    """
    # 照合記録ID
    record_id: str

    # 予期側情報
    expectation_id: str
    expectation_description: str  # 文字数上限付き（MAX_TEXT_LENGTH）
    expectation_source_type: str  # ExpectationSourceType.value
    expectation_basis: str  # ExpectationBasis.value
    expectation_strength: str  # ExpectationStrength段階ラベル
    expectation_freshness: str  # ExpectationFreshness段階ラベル

    # 知覚側情報（文字数/要素数上限付き）
    percept_meaning: str  # MAX_TEXT_LENGTH
    percept_intent: str
    percept_topics: tuple[str, ...]  # MAX_TOPICS_COUNT
    percept_sentiment: float

    # 対応断面群（3断面）
    content_correspondence: CorrespondenceLevel  # 内容断面
    topic_correspondence: CorrespondenceLevel  # 話題断面
    intent_correspondence: CorrespondenceLevel  # 意図断面

    # 文脈情報
    tick: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "expectation_id": self.expectation_id,
            "expectation_description": self.expectation_description,
            "expectation_source_type": self.expectation_source_type,
            "expectation_basis": self.expectation_basis,
            "expectation_strength": self.expectation_strength,
            "expectation_freshness": self.expectation_freshness,
            "percept_meaning": self.percept_meaning,
            "percept_intent": self.percept_intent,
            "percept_topics": list(self.percept_topics),
            "percept_sentiment": self.percept_sentiment,
            "content_correspondence": self.content_correspondence.value,
            "topic_correspondence": self.topic_correspondence.value,
            "intent_correspondence": self.intent_correspondence.value,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchingRecord:
        return cls(
            record_id=data.get("record_id", _gen_id()),
            expectation_id=data.get("expectation_id", ""),
            expectation_description=data.get("expectation_description", ""),
            expectation_source_type=data.get("expectation_source_type", ""),
            expectation_basis=data.get("expectation_basis", ""),
            expectation_strength=data.get("expectation_strength", ""),
            expectation_freshness=data.get("expectation_freshness", ""),
            percept_meaning=data.get("percept_meaning", ""),
            percept_intent=data.get("percept_intent", ""),
            percept_topics=tuple(data.get("percept_topics", [])),
            percept_sentiment=float(data.get("percept_sentiment", 0.0)),
            content_correspondence=CorrespondenceLevel(
                data.get("content_correspondence", "indeterminate")
            ),
            topic_correspondence=CorrespondenceLevel(
                data.get("topic_correspondence", "indeterminate")
            ),
            intent_correspondence=CorrespondenceLevel(
                data.get("intent_correspondence", "indeterminate")
            ),
            tick=int(data.get("tick", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


# =============================================================================
# State (for persistence)
# =============================================================================

@dataclass
class MatchingState:
    """照合記録リストの永続化状態。"""
    records: list[MatchingRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchingState:
        records = [
            MatchingRecord.from_dict(r)
            for r in data.get("records", [])
        ]
        return cls(records=records)


# =============================================================================
# Correspondence Determination (Pure Functions)
# =============================================================================

def _tokenize(text: str) -> set[str]:
    """テキストを単語集合に分割する。"""
    return set(text.lower().split())


def _compute_content_correspondence(
    description: str, meaning: str,
) -> CorrespondenceLevel:
    """予期内容記述と知覚意味記述のテキスト間構造的類似を段階値で記述する。"""
    if not description.strip() or not meaning.strip():
        return CorrespondenceLevel.INDETERMINATE

    desc_tokens = _tokenize(description)
    meaning_tokens = _tokenize(meaning)

    if not desc_tokens or not meaning_tokens:
        return CorrespondenceLevel.INDETERMINATE

    # 双方向の包含率
    overlap = desc_tokens & meaning_tokens
    if not overlap:
        return CorrespondenceLevel.NO_CORRESPONDENCE

    ratio_from_desc = len(overlap) / len(desc_tokens)
    ratio_from_meaning = len(overlap) / len(meaning_tokens)
    combined = (ratio_from_desc + ratio_from_meaning) / 2.0

    if combined >= 0.6:
        return CorrespondenceLevel.HIGH_CORRESPONDENCE
    elif combined >= 0.3:
        return CorrespondenceLevel.MODERATE_CORRESPONDENCE
    else:
        return CorrespondenceLevel.LOW_CORRESPONDENCE


def _compute_topic_correspondence(
    description: str, topics: tuple[str, ...],
) -> CorrespondenceLevel:
    """予期内容記述に含まれる話題要素と知覚話題の重複を段階値で記述する。"""
    if not description.strip() or not topics:
        return CorrespondenceLevel.INDETERMINATE

    desc_lower = description.lower()
    hit_count = sum(1 for t in topics if t.lower() in desc_lower)

    if not topics:
        return CorrespondenceLevel.INDETERMINATE

    ratio = hit_count / len(topics)

    if ratio >= 0.6:
        return CorrespondenceLevel.HIGH_CORRESPONDENCE
    elif ratio >= 0.3:
        return CorrespondenceLevel.MODERATE_CORRESPONDENCE
    elif ratio > 0.0:
        return CorrespondenceLevel.LOW_CORRESPONDENCE
    else:
        return CorrespondenceLevel.NO_CORRESPONDENCE


def _compute_intent_correspondence(
    description: str, intent: str,
) -> CorrespondenceLevel:
    """予期が想定した展開と知覚された意図の対応を段階値で記述する。"""
    if not description.strip() or not intent.strip():
        return CorrespondenceLevel.INDETERMINATE

    desc_lower = description.lower()
    intent_lower = intent.lower()

    # 意図ラベルが予期記述に直接含まれるか
    if intent_lower in desc_lower:
        return CorrespondenceLevel.HIGH_CORRESPONDENCE

    # 意図ラベルの単語が予期記述の単語と重複するか
    intent_tokens = _tokenize(intent)
    desc_tokens = _tokenize(description)
    overlap = intent_tokens & desc_tokens

    if overlap:
        ratio = len(overlap) / len(intent_tokens) if intent_tokens else 0.0
        if ratio >= 0.5:
            return CorrespondenceLevel.MODERATE_CORRESPONDENCE
        else:
            return CorrespondenceLevel.LOW_CORRESPONDENCE

    return CorrespondenceLevel.NO_CORRESPONDENCE


# =============================================================================
# Matcher
# =============================================================================

class ExpectationPerceptionMatcher:
    """予期照合記述の処理・蓄積・アクセサ。

    予期形成の出力と知覚の出力を構造的フィールド比較し、
    対応の程度を段階値で記述・蓄積する。

    構造的分離:
    - 予期形成への書き込み経路なし（READ-ONLY参照のみ）
    - 感情システムへの入出力経路なし
    - ポリシー選択への出力経路なし
    - 照合記録リストは外部からの直接変更経路なし
    """

    def __init__(self) -> None:
        self._state = MatchingState()

    @property
    def state(self) -> MatchingState:
        """永続化用の状態。"""
        return self._state

    @state.setter
    def state(self, value: MatchingState) -> None:
        self._state = value

    def process(
        self,
        expectation_store: Any,
        percept: Any,
        tick: int = 0,
    ) -> None:
        """照合処理を実行し、記録を蓄積する。

        予期候補と知覚が同一tick内に共存する場合に照合記録を生成する。
        入力欠落時は無処理（安全弁4）。
        """
        # ── 入力検証 ──
        if expectation_store is None or percept is None:
            return

        # 型チェック: expectation_store
        expectations = getattr(expectation_store, "expectations", None)
        if expectations is None:
            logger.debug("Expectation store has no expectations attribute, skipping")
            return

        if not isinstance(expectations, (list, tuple)):
            logger.debug("Expectations is not iterable, skipping")
            return

        # 型チェック: percept
        percept_text = getattr(percept, "text", None)
        percept_meaning = getattr(percept, "meaning", None)
        percept_intent = getattr(percept, "intent", None)
        percept_topics = getattr(percept, "topics", None)
        percept_sentiment = getattr(percept, "sentiment", None)

        if not isinstance(percept_text, str) or not isinstance(percept_meaning, str):
            logger.debug("Percept text/meaning is not str, skipping")
            return

        # 入力欠落チェック（安全弁4）
        if not expectations:
            return

        # 知覚が空の場合は無処理
        meaning_stripped = percept_meaning.strip() if percept_meaning else ""
        text_stripped = percept_text.strip() if percept_text else ""
        topics_list = percept_topics if isinstance(percept_topics, (list, tuple)) else []

        if not meaning_stripped and not text_stripped and not topics_list:
            return

        # ── 知覚フィールドの切り詰め ──
        safe_meaning = meaning_stripped[:MAX_TEXT_LENGTH]
        safe_intent = str(percept_intent or "")[:MAX_TEXT_LENGTH]
        safe_topics = tuple(str(t) for t in topics_list[:MAX_TOPICS_COUNT])
        safe_sentiment = float(percept_sentiment) if isinstance(percept_sentiment, (int, float)) else 0.0
        safe_sentiment = max(SENTIMENT_MIN, min(SENTIMENT_MAX, safe_sentiment))

        # ── 各予期候補との照合 ──
        now = time.time()
        for exp in expectations:
            try:
                self._process_one(exp, safe_meaning, safe_intent, safe_topics, safe_sentiment, tick, now)
            except Exception as e:
                logger.debug("Matching record creation failed for one expectation: %s", e)

    def _process_one(
        self,
        exp: Any,
        meaning: str,
        intent: str,
        topics: tuple[str, ...],
        sentiment: float,
        tick: int,
        now: float,
    ) -> None:
        """1つの予期候補に対する照合記録を生成する。"""
        # 予期フィールドの読み取り（READ-ONLY）
        exp_id = str(getattr(exp, "expectation_id", "") or "")
        exp_desc_raw = str(getattr(exp, "description", "") or "")
        exp_desc = exp_desc_raw[:MAX_TEXT_LENGTH]

        exp_source_type_raw = getattr(exp, "source_type", None)
        exp_source_type = exp_source_type_raw.value if hasattr(exp_source_type_raw, "value") else str(exp_source_type_raw or "")

        exp_basis_raw = getattr(exp, "basis", None)
        exp_basis = exp_basis_raw.value if hasattr(exp_basis_raw, "value") else str(exp_basis_raw or "")

        # 強度・鮮度の段階ラベル
        exp_strength_raw = getattr(exp, "strength", None)
        if isinstance(exp_strength_raw, (int, float)):
            strength_level = getattr(exp, "get_strength_level", None)
            if callable(strength_level):
                exp_strength = strength_level().value
            else:
                exp_strength = str(exp_strength_raw)
        elif hasattr(exp_strength_raw, "value"):
            exp_strength = exp_strength_raw.value
        else:
            exp_strength = str(exp_strength_raw or "")

        exp_freshness_raw = getattr(exp, "freshness", None)
        if isinstance(exp_freshness_raw, (int, float)):
            freshness_level = getattr(exp, "get_freshness_level", None)
            if callable(freshness_level):
                exp_freshness = freshness_level().value
            else:
                exp_freshness = str(exp_freshness_raw)
        elif hasattr(exp_freshness_raw, "value"):
            exp_freshness = exp_freshness_raw.value
        else:
            exp_freshness = str(exp_freshness_raw or "")

        # ── 第1段: フィールド対応の記述 ──
        content_level = _compute_content_correspondence(exp_desc, meaning)
        topic_level = _compute_topic_correspondence(exp_desc, topics)
        intent_level = _compute_intent_correspondence(exp_desc, intent)

        # ── 第2段+第3段: 多断面記述の構成と蓄積 ──
        record = MatchingRecord(
            record_id=_gen_id(),
            expectation_id=exp_id,
            expectation_description=exp_desc,
            expectation_source_type=exp_source_type,
            expectation_basis=exp_basis,
            expectation_strength=exp_strength,
            expectation_freshness=exp_freshness,
            percept_meaning=meaning,
            percept_intent=intent,
            percept_topics=topics,
            percept_sentiment=sentiment,
            content_correspondence=content_level,
            topic_correspondence=topic_level,
            intent_correspondence=intent_level,
            tick=tick,
            timestamp=now,
        )

        self._state.records.append(record)

        # FIFO上限（安全弁1）
        if len(self._state.records) > MAX_RECORDS:
            self._state.records = self._state.records[-MAX_RECORDS:]

    # ── アクセサ（読み取り専用） ──

    def record_count(self) -> int:
        """照合記録の件数を返す。"""
        return len(self._state.records)

    def get_recent_records(self, count: int) -> tuple[MatchingRecord, ...]:
        """直近の照合記録群を返す。集計・統計処理禁止 (no aggregation/statistics)。

        返り値は記録の生データそのまま。解釈・強調・要約は含まない。
        """
        if count <= 0:
            return ()
        return tuple(self._state.records[-count:])

    def get_enrichment_data(self) -> dict[str, Any]:
        """enrichment出力用のデータを返す。

        記録件数と直近記録の断面段階値の等価列挙のみ。
        特定の記録を強調・優先する表現を含まない（安全弁5）。
        """
        total = self.record_count()
        if total == 0:
            return {
                "record_count": 0,
                "summary_text": "",
                "recent_records": [],
            }

        recent = self.get_recent_records(5)
        record_summaries = []
        for r in recent:
            record_summaries.append({
                "content": r.content_correspondence.value,
                "topic": r.topic_correspondence.value,
                "intent": r.intent_correspondence.value,
                "tick": r.tick,
            })

        # 等価列挙テキスト（安全弁5: 強調なし）
        parts = [f"予期照合記録: {total}件"]
        for rs in record_summaries:
            parts.append(
                f"  内容={rs['content']}, "
                f"話題={rs['topic']}, "
                f"意図={rs['intent']}"
            )
        summary_text = "\n".join(parts)

        return {
            "record_count": total,
            "summary_text": summary_text,
            "recent_records": record_summaries,
        }
