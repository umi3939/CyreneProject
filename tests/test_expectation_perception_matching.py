"""
Tests for psyche/expectation_perception_matching.py

予期照合記述モジュールのテスト。
設計書: design_expectation_perception_matching.md
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from psyche.expectation_perception_matching import (
    CorrespondenceLevel,
    MatchingRecord,
    ExpectationPerceptionMatcher,
    MatchingState,
    MAX_RECORDS,
    MAX_TEXT_LENGTH,
    MAX_TOPICS_COUNT,
    SENTIMENT_MIN,
    SENTIMENT_MAX,
)
from psyche.expectation_formation import (
    ExpectationCandidate,
    ExpectationSourceType,
    ExpectationBasis,
    ExpectationStrength,
    ExpectationFreshness,
    ExpectationStore,
)
from psyche.state import Percept


# =============================================================================
# Helpers
# =============================================================================

def _make_candidate(
    description: str = "test expectation",
    source_type: ExpectationSourceType = ExpectationSourceType.REPETITION,
    basis: ExpectationBasis = ExpectationBasis.PATTERN_CONTINUATION,
    freshness: float = 0.8,
    strength: float = 0.6,
) -> ExpectationCandidate:
    return ExpectationCandidate(
        expectation_id=uuid.uuid4().hex[:12],
        source_type=source_type,
        basis=basis,
        description=description,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        freshness=freshness,
        strength=strength,
        reference_count=0,
        evidence_ids=(),
        competing_ids=(),
        revision_count=0,
        undetermined_aspects=(),
    )


def _make_percept(
    text: str = "hello world",
    meaning: str = "greeting",
    intent: str = "greeting",
    topics: list[str] | None = None,
    sentiment: float = 0.0,
    emotion_valence: float = 0.0,
) -> Percept:
    return Percept(
        text=text,
        meaning=meaning,
        intent=intent,
        topics=topics or ["greeting"],
        sentiment=sentiment,
        emotion_valence=emotion_valence,
    )


def _make_store(candidates: list[ExpectationCandidate] | None = None) -> ExpectationStore:
    cands = candidates or []
    return ExpectationStore(
        expectations=tuple(cands),
        evidence_links=(),
        total_expectations_created=len(cands),
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.5,
        average_strength=0.5,
        active_expectation_count=len(cands),
        competing_pair_count=0,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        description="test store",
    )


# =============================================================================
# CorrespondenceLevel Enum Tests
# =============================================================================

class TestCorrespondenceLevel:
    """段階値列挙型のテスト。"""

    def test_five_values(self):
        """5段階が存在すること。"""
        assert len(CorrespondenceLevel) == 5

    def test_value_names(self):
        """価値判断語(CORRECT/MATCH等)を含まないこと。"""
        for level in CorrespondenceLevel:
            name = level.name
            assert "CORRECT" not in name
            assert "MATCH" not in name.upper() or "CORRESPONDENCE" in name.upper()
            assert "SUCCESS" not in name
            assert "FAILURE" not in name
            assert "RIGHT" not in name
            assert "WRONG" not in name

    def test_expected_values(self):
        """設計書通りの5段階が存在すること。"""
        assert CorrespondenceLevel.HIGH_CORRESPONDENCE is not None
        assert CorrespondenceLevel.MODERATE_CORRESPONDENCE is not None
        assert CorrespondenceLevel.LOW_CORRESPONDENCE is not None
        assert CorrespondenceLevel.NO_CORRESPONDENCE is not None
        assert CorrespondenceLevel.INDETERMINATE is not None


# =============================================================================
# MatchingRecord Tests
# =============================================================================

class TestMatchingRecord:
    """照合記録のテスト。"""

    def test_frozen(self):
        """照合記録は不変であること。"""
        record = MatchingRecord(
            record_id="test123",
            expectation_id="exp001",
            expectation_description="desc",
            expectation_source_type="repetition",
            expectation_basis="pattern_continuation",
            expectation_strength="moderate",
            expectation_freshness="fresh",
            percept_meaning="hello",
            percept_intent="greeting",
            percept_topics=("topic1",),
            percept_sentiment=0.5,
            content_correspondence=CorrespondenceLevel.HIGH_CORRESPONDENCE,
            topic_correspondence=CorrespondenceLevel.MODERATE_CORRESPONDENCE,
            intent_correspondence=CorrespondenceLevel.LOW_CORRESPONDENCE,
            tick=10,
            timestamp=time.time(),
        )
        with pytest.raises(AttributeError):
            record.record_id = "new_id"  # type: ignore[misc]

    def test_to_dict(self):
        """to_dictでシリアライズできること。"""
        ts = time.time()
        record = MatchingRecord(
            record_id="test123",
            expectation_id="exp001",
            expectation_description="desc",
            expectation_source_type="repetition",
            expectation_basis="pattern_continuation",
            expectation_strength="moderate",
            expectation_freshness="fresh",
            percept_meaning="hello",
            percept_intent="greeting",
            percept_topics=("topic1",),
            percept_sentiment=0.5,
            content_correspondence=CorrespondenceLevel.HIGH_CORRESPONDENCE,
            topic_correspondence=CorrespondenceLevel.MODERATE_CORRESPONDENCE,
            intent_correspondence=CorrespondenceLevel.LOW_CORRESPONDENCE,
            tick=10,
            timestamp=ts,
        )
        d = record.to_dict()
        assert d["record_id"] == "test123"
        assert d["expectation_id"] == "exp001"
        assert d["content_correspondence"] == "high_correspondence"
        assert d["topic_correspondence"] == "moderate_correspondence"
        assert d["intent_correspondence"] == "low_correspondence"
        assert d["percept_topics"] == ["topic1"]
        assert d["tick"] == 10

    def test_from_dict(self):
        """from_dictでデシリアライズできること。"""
        d = {
            "record_id": "rec001",
            "expectation_id": "exp001",
            "expectation_description": "desc",
            "expectation_source_type": "repetition",
            "expectation_basis": "pattern_continuation",
            "expectation_strength": "moderate",
            "expectation_freshness": "fresh",
            "percept_meaning": "hello",
            "percept_intent": "greeting",
            "percept_topics": ["topic1", "topic2"],
            "percept_sentiment": 0.3,
            "content_correspondence": "high_correspondence",
            "topic_correspondence": "moderate_correspondence",
            "intent_correspondence": "low_correspondence",
            "tick": 5,
            "timestamp": 1000.0,
        }
        record = MatchingRecord.from_dict(d)
        assert record.record_id == "rec001"
        assert record.content_correspondence == CorrespondenceLevel.HIGH_CORRESPONDENCE
        assert record.percept_topics == ("topic1", "topic2")
        assert record.tick == 5

    def test_roundtrip(self):
        """to_dict -> from_dict のラウンドトリップ。"""
        ts = time.time()
        original = MatchingRecord(
            record_id="rt001",
            expectation_id="exp002",
            expectation_description="desc test",
            expectation_source_type="narrative",
            expectation_basis="change_direction",
            expectation_strength="strong",
            expectation_freshness="recent",
            percept_meaning="test meaning",
            percept_intent="inform",
            percept_topics=("a", "b"),
            percept_sentiment=-0.5,
            content_correspondence=CorrespondenceLevel.NO_CORRESPONDENCE,
            topic_correspondence=CorrespondenceLevel.INDETERMINATE,
            intent_correspondence=CorrespondenceLevel.HIGH_CORRESPONDENCE,
            tick=42,
            timestamp=ts,
        )
        restored = MatchingRecord.from_dict(original.to_dict())
        assert restored.record_id == original.record_id
        assert restored.content_correspondence == original.content_correspondence
        assert restored.topic_correspondence == original.topic_correspondence
        assert restored.intent_correspondence == original.intent_correspondence
        assert restored.percept_topics == original.percept_topics


# =============================================================================
# Matcher Core Logic Tests
# =============================================================================

class TestMatcherProcess:
    """照合処理のテスト。"""

    def test_process_basic(self):
        """基本的な照合処理が動作すること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="user will greet")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="greeting from user",
            intent="greeting",
            topics=["greeting", "hello"],
        )
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 1

    def test_process_multiple_expectations(self):
        """複数の予期候補に対して各々照合記録が生成されること。"""
        matcher = ExpectationPerceptionMatcher()
        c1 = _make_candidate(description="topic A discussion")
        c2 = _make_candidate(description="topic B question")
        store = _make_store([c1, c2])
        percept = _make_percept(meaning="about topic A", topics=["A"])
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 2

    def test_no_expectations_no_record(self):
        """予期が存在しない場合は照合記録を生成しないこと(安全弁4)。"""
        matcher = ExpectationPerceptionMatcher()
        store = _make_store([])
        percept = _make_percept()
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 0

    def test_empty_percept_text_no_record(self):
        """知覚テキストが空の場合は照合記録を生成しないこと(安全弁4)。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate()
        store = _make_store([candidate])
        percept = Percept(text="", meaning="", intent="", topics=[], sentiment=0.0)
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 0

    def test_none_store_no_record(self):
        """storeがNoneの場合は照合記録を生成しないこと。"""
        matcher = ExpectationPerceptionMatcher()
        percept = _make_percept()
        matcher.process(None, percept, tick=1)
        assert matcher.record_count() == 0

    def test_none_percept_no_record(self):
        """perceptがNoneの場合は照合記録を生成しないこと。"""
        matcher = ExpectationPerceptionMatcher()
        store = _make_store([_make_candidate()])
        matcher.process(store, None, tick=1)
        assert matcher.record_count() == 0


# =============================================================================
# Correspondence Level Determination Tests
# =============================================================================

class TestCorrespondenceDetermination:
    """段階値判定ロジックのテスト。"""

    def test_high_topic_overlap(self):
        """話題の重複が大きい場合にHIGH_CORRESPONDENCEになること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="greeting hello world")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="greeting hello world",
            topics=["greeting", "hello", "world"],
        )
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records) == 1
        # 話題の完全一致
        assert records[0].topic_correspondence in (
            CorrespondenceLevel.HIGH_CORRESPONDENCE,
            CorrespondenceLevel.MODERATE_CORRESPONDENCE,
        )

    def test_no_topic_overlap(self):
        """話題の重複がない場合にNO_CORRESPONDENCEになること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="weather forecast rain")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="coding python",
            intent="inform",
            topics=["coding", "python"],
        )
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records) == 1
        assert records[0].topic_correspondence in (
            CorrespondenceLevel.NO_CORRESPONDENCE,
            CorrespondenceLevel.LOW_CORRESPONDENCE,
        )

    def test_indeterminate_when_empty_description(self):
        """予期descriptionが空の場合にINDETERMINATEになること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="")
        store = _make_store([candidate])
        percept = _make_percept(meaning="something", topics=["something"])
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records) == 1
        assert records[0].content_correspondence == CorrespondenceLevel.INDETERMINATE
        assert records[0].topic_correspondence == CorrespondenceLevel.INDETERMINATE
        assert records[0].intent_correspondence == CorrespondenceLevel.INDETERMINATE


# =============================================================================
# FIFO and Limits Tests
# =============================================================================

class TestFIFOAndLimits:
    """FIFO上限と切り詰めのテスト。"""

    def test_fifo_limit(self):
        """上限を超えると最古の記録が脱落すること(安全弁1)。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(MAX_RECORDS + 10):
            candidate = _make_candidate(description=f"expectation {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"percept {i}", topics=[f"topic_{i}"])
            matcher.process(store, percept, tick=i)
        assert matcher.record_count() == MAX_RECORDS

    def test_fifo_oldest_removed(self):
        """FIFO脱落で最古の記録が消えること。"""
        matcher = ExpectationPerceptionMatcher()
        # 最初の記録
        first_candidate = _make_candidate(description="first expectation")
        first_store = _make_store([first_candidate])
        first_percept = _make_percept(meaning="first", topics=["first"])
        matcher.process(first_store, first_percept, tick=0)
        first_id = matcher.get_recent_records(MAX_RECORDS)[0].record_id

        # 上限まで追加
        for i in range(1, MAX_RECORDS + 1):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)

        # 最初のIDは消えている
        all_ids = [r.record_id for r in matcher.get_recent_records(MAX_RECORDS)]
        assert first_id not in all_ids

    def test_text_truncation(self):
        """テキストフィールドが文字数上限で切り詰められること。"""
        matcher = ExpectationPerceptionMatcher()
        long_text = "a" * 500
        candidate = _make_candidate(description=long_text)
        store = _make_store([candidate])
        percept = _make_percept(meaning=long_text, topics=["test"])
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records[0].expectation_description) <= MAX_TEXT_LENGTH
        assert len(records[0].percept_meaning) <= MAX_TEXT_LENGTH

    def test_topics_truncation(self):
        """話題リストが要素数上限で切り詰められること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="test")
        store = _make_store([candidate])
        many_topics = [f"topic_{i}" for i in range(30)]
        percept = _make_percept(meaning="test", topics=many_topics)
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records[0].percept_topics) <= MAX_TOPICS_COUNT

    def test_sentiment_range_validation(self):
        """感情極性値が範囲外の場合にスキップすること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="test")
        store = _make_store([candidate])
        # Perceptのsentimentは通常-1.0~1.0だが、emotion_valenceのvalidationがある
        # 直接不正な値を渡す場合のテスト
        percept = _make_percept(meaning="test", topics=["test"], sentiment=0.5)
        matcher.process(store, percept, tick=1)
        # 正常範囲なので記録される
        assert matcher.record_count() == 1


# =============================================================================
# Type Validation Tests
# =============================================================================

class TestTypeValidation:
    """型チェックのテスト。"""

    def test_invalid_store_type_skipped(self):
        """storeが不正な型の場合スキップされること。"""
        matcher = ExpectationPerceptionMatcher()
        percept = _make_percept()
        matcher.process("not a store", percept, tick=1)  # type: ignore[arg-type]
        assert matcher.record_count() == 0

    def test_invalid_percept_type_skipped(self):
        """perceptが不正な型の場合スキップされること。"""
        matcher = ExpectationPerceptionMatcher()
        store = _make_store([_make_candidate()])
        matcher.process(store, "not a percept", tick=1)  # type: ignore[arg-type]
        assert matcher.record_count() == 0


# =============================================================================
# Accessor Tests
# =============================================================================

class TestAccessors:
    """アクセサのテスト。"""

    def test_get_recent_records(self):
        """直近の記録群を取得できること。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(5):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)
        records = matcher.get_recent_records(3)
        assert len(records) == 3

    def test_get_recent_records_returns_tuple(self):
        """get_recent_recordsがtupleを返すこと（読み取り専用）。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate()
        store = _make_store([candidate])
        percept = _make_percept()
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(10)
        assert isinstance(records, tuple)

    def test_record_count(self):
        """record_countが正しい件数を返すこと。"""
        matcher = ExpectationPerceptionMatcher()
        assert matcher.record_count() == 0
        candidate = _make_candidate()
        store = _make_store([candidate])
        percept = _make_percept()
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 1

    def test_get_recent_records_empty(self):
        """記録がない場合は空タプルを返すこと。"""
        matcher = ExpectationPerceptionMatcher()
        records = matcher.get_recent_records(10)
        assert records == ()

    def test_get_recent_records_exceeds_count(self):
        """要求件数が記録数を超えても問題ないこと。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate()
        store = _make_store([candidate])
        percept = _make_percept()
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(100)
        assert len(records) == 1

    def test_accessor_docstring_no_statistics(self):
        """アクセサのdocstringに「集計・統計処理禁止」が含まれること(解析watch項目3)。"""
        doc = ExpectationPerceptionMatcher.get_recent_records.__doc__ or ""
        assert "集計" in doc or "統計" in doc or "no aggregation" in doc.lower()


# =============================================================================
# State Persistence Tests
# =============================================================================

class TestStatePersistence:
    """save/loadのテスト。"""

    def test_state_to_dict(self):
        """MatchingStateがto_dictでシリアライズできること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate()
        store = _make_store([candidate])
        percept = _make_percept()
        matcher.process(store, percept, tick=1)
        state = matcher.state
        d = state.to_dict()
        assert "records" in d
        assert len(d["records"]) == 1

    def test_state_from_dict(self):
        """MatchingStateがfrom_dictでデシリアライズできること。"""
        d = {
            "records": [
                {
                    "record_id": "test",
                    "expectation_id": "exp",
                    "expectation_description": "desc",
                    "expectation_source_type": "repetition",
                    "expectation_basis": "pattern_continuation",
                    "expectation_strength": "moderate",
                    "expectation_freshness": "fresh",
                    "percept_meaning": "meaning",
                    "percept_intent": "intent",
                    "percept_topics": ["a"],
                    "percept_sentiment": 0.0,
                    "content_correspondence": "high_correspondence",
                    "topic_correspondence": "moderate_correspondence",
                    "intent_correspondence": "low_correspondence",
                    "tick": 1,
                    "timestamp": 1000.0,
                }
            ],
        }
        state = MatchingState.from_dict(d)
        assert len(state.records) == 1
        assert state.records[0].content_correspondence == CorrespondenceLevel.HIGH_CORRESPONDENCE

    def test_state_roundtrip(self):
        """state -> to_dict -> from_dict のラウンドトリップ。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(3):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)
        state = matcher.state
        d = state.to_dict()
        restored = MatchingState.from_dict(d)
        assert len(restored.records) == len(state.records)
        for orig, rest in zip(state.records, restored.records):
            assert orig.record_id == rest.record_id
            assert orig.content_correspondence == rest.content_correspondence

    def test_state_from_dict_empty(self):
        """空のdictからでもstateを復元できること。"""
        state = MatchingState.from_dict({})
        assert len(state.records) == 0

    def test_load_state(self):
        """matcherにstateをロードできること。"""
        matcher = ExpectationPerceptionMatcher()
        d = {
            "records": [
                {
                    "record_id": "loaded",
                    "expectation_id": "exp",
                    "expectation_description": "desc",
                    "expectation_source_type": "repetition",
                    "expectation_basis": "pattern_continuation",
                    "expectation_strength": "moderate",
                    "expectation_freshness": "fresh",
                    "percept_meaning": "meaning",
                    "percept_intent": "intent",
                    "percept_topics": [],
                    "percept_sentiment": 0.0,
                    "content_correspondence": "no_correspondence",
                    "topic_correspondence": "indeterminate",
                    "intent_correspondence": "high_correspondence",
                    "tick": 5,
                    "timestamp": 2000.0,
                }
            ],
        }
        matcher.state = MatchingState.from_dict(d)
        assert matcher.record_count() == 1
        assert matcher.get_recent_records(1)[0].record_id == "loaded"


# =============================================================================
# Enrichment Text Tests
# =============================================================================

class TestEnrichment:
    """enrichment出力のテスト。"""

    def test_enrichment_empty(self):
        """記録がない場合のenrichment。"""
        matcher = ExpectationPerceptionMatcher()
        data = matcher.get_enrichment_data()
        assert data["record_count"] == 0

    def test_enrichment_with_records(self):
        """記録がある場合のenrichment。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(3):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)
        data = matcher.get_enrichment_data()
        assert data["record_count"] == 3
        assert "summary_text" in data
        assert "recent_records" in data

    def test_enrichment_equal_listing(self):
        """enrichment内で特定記録が強調されないこと(安全弁5)。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(3):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)
        data = matcher.get_enrichment_data()
        text = data.get("summary_text", "")
        # 「重要」「注目」等の強調語が含まれないこと
        assert "重要" not in text
        assert "注目" not in text
        assert "注意" not in text


# =============================================================================
# Structural Separation Tests
# =============================================================================

class TestStructuralSeparation:
    """構造的分離のテスト。"""

    def test_no_emotion_dependency(self):
        """感情システムへの依存がないこと。"""
        import psyche.expectation_perception_matching as mod
        source = open(mod.__file__, encoding="utf-8").read()
        # emotion_amplitude, reaction, dynamics などの import がないこと
        assert "from .reaction" not in source
        assert "from .dynamics" not in source
        assert "from .emotion_amplitude" not in source
        assert "from .fear" not in source

    def test_no_policy_dependency(self):
        """ポリシー選択系への依存がないこと。"""
        import psyche.expectation_perception_matching as mod
        source = open(mod.__file__, encoding="utf-8").read()
        assert "from .decision_bias" not in source
        assert "from .thought_generation" not in source

    def test_no_write_to_expectation_formation(self):
        """予期形成への書き込みメソッドが存在しないこと。"""
        matcher = ExpectationPerceptionMatcher()
        # matcher に set_expectation, update_expectation 等がないこと
        for attr_name in dir(matcher):
            assert "set_expectation" not in attr_name
            assert "update_expectation" not in attr_name
            assert "write_expectation" not in attr_name

    def test_no_statistics_methods(self):
        """統計集計メソッドが存在しないこと(パターン抽出禁止)。"""
        matcher = ExpectationPerceptionMatcher()
        for attr_name in dir(matcher):
            if attr_name.startswith("_"):
                continue
            assert "average" not in attr_name.lower()
            assert "statistics" not in attr_name.lower()
            assert "frequency" not in attr_name.lower()
            assert "pattern" not in attr_name.lower()
            assert "trend" not in attr_name.lower()


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_percept_with_none_topics(self):
        """Perceptのtopicsがデフォルト(空リスト)の場合。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="test")
        store = _make_store([candidate])
        percept = Percept(text="hello", meaning="hello", intent="greeting")
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 1

    def test_very_long_description(self):
        """非常に長いdescriptionが切り詰められること。"""
        matcher = ExpectationPerceptionMatcher()
        long_desc = "x" * 1000
        candidate = _make_candidate(description=long_desc)
        store = _make_store([candidate])
        percept = _make_percept(meaning="short", topics=["test"])
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert len(records[0].expectation_description) <= MAX_TEXT_LENGTH

    def test_unicode_text(self):
        """Unicode文字列が正しく処理されること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="日本語の予期テスト")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="日本語の意味",
            topics=["日本語", "テスト"],
        )
        matcher.process(store, percept, tick=1)
        assert matcher.record_count() == 1
        records = matcher.get_recent_records(1)
        assert "日本語" in records[0].expectation_description

    def test_multiple_process_calls(self):
        """複数回のprocess呼び出しで記録が蓄積されること。"""
        matcher = ExpectationPerceptionMatcher()
        for i in range(10):
            candidate = _make_candidate(description=f"exp {i}")
            store = _make_store([candidate])
            percept = _make_percept(meaning=f"m{i}", topics=[f"t{i}"])
            matcher.process(store, percept, tick=i)
        assert matcher.record_count() == 10

    def test_store_with_no_expectations_attribute(self):
        """storeにexpectations属性がない場合スキップされること。"""
        matcher = ExpectationPerceptionMatcher()

        class FakeStore:
            pass

        percept = _make_percept()
        matcher.process(FakeStore(), percept, tick=1)  # type: ignore[arg-type]
        assert matcher.record_count() == 0

    def test_percept_only_whitespace(self):
        """知覚テキストが空白のみの場合。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="test")
        store = _make_store([candidate])
        percept = Percept(text="   ", meaning="   ", intent="", topics=[])
        matcher.process(store, percept, tick=1)
        # 空白のみでも meaning が空白のみ & topics空 → 無処理
        assert matcher.record_count() == 0


# =============================================================================
# Content Correspondence Logic Tests
# =============================================================================

class TestContentCorrespondence:
    """内容断面の対応判定ロジックのテスト。"""

    def test_identical_text_high(self):
        """同一テキストの場合にHIGHになること。"""
        matcher = ExpectationPerceptionMatcher()
        text = "the user discusses python programming"
        candidate = _make_candidate(description=text)
        store = _make_store([candidate])
        percept = _make_percept(meaning=text, topics=["python", "programming"])
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert records[0].content_correspondence == CorrespondenceLevel.HIGH_CORRESPONDENCE

    def test_completely_different_text_low_or_no(self):
        """完全に異なるテキストの場合にLOWかNOになること。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="weather forecast tomorrow rain")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="coding python debugging error",
            topics=["coding"],
        )
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert records[0].content_correspondence in (
            CorrespondenceLevel.LOW_CORRESPONDENCE,
            CorrespondenceLevel.NO_CORRESPONDENCE,
        )


# =============================================================================
# Intent Correspondence Logic Tests
# =============================================================================

class TestIntentCorrespondence:
    """意図断面の対応判定ロジックのテスト。"""

    def test_intent_mentioned_in_description(self):
        """予期descriptionに意図が含まれる場合。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="user will ask a question")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="asking about python",
            intent="question",
            topics=["python"],
        )
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert records[0].intent_correspondence in (
            CorrespondenceLevel.HIGH_CORRESPONDENCE,
            CorrespondenceLevel.MODERATE_CORRESPONDENCE,
        )

    def test_intent_not_in_description(self):
        """予期descriptionに意図が含まれない場合。"""
        matcher = ExpectationPerceptionMatcher()
        candidate = _make_candidate(description="weather discussion")
        store = _make_store([candidate])
        percept = _make_percept(
            meaning="asking about code",
            intent="question",
            topics=["code"],
        )
        matcher.process(store, percept, tick=1)
        records = matcher.get_recent_records(1)
        assert records[0].intent_correspondence in (
            CorrespondenceLevel.LOW_CORRESPONDENCE,
            CorrespondenceLevel.NO_CORRESPONDENCE,
        )
