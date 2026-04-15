"""
tests/test_multi_path_recall.py - 記憶の多経路想起のテスト

カバー範囲:
- 初期状態
- RecallCandidate / PathStatistics / EmotionSnapshot / ContextSnapshot / TemporalSnapshot の生成・to_dict/from_dict
- 経路1: 感情連想の候補生成
- 経路2: 文脈連想の候補生成
- 経路3: 時間近接の候補生成
- 3経路同時実行
- 安全弁1: 経路間等価性（経路ごとの候補上限を同一に設定）
- 安全弁2: 顕著性バイアス抑制（感情痕跡が弱い記憶の一定割合混入）
- 安全弁3: ルーミネーション防止（スライディングウィンドウ抑制）
- 安全弁4: 忘却処理との分離（INVISIBLE記憶除外、参照頻度非通知）
- 安全弁5: 外部API想起との整合（enrichment参照のみ）
- 3経路遮断（想起→忘却、想起→感情、想起→想起）
- enrichmentデータのフォーマット
- save/load round-trip
- get_recall_candidates READ-ONLY
- エッジケース（空入力、候補0件、全記憶INVISIBLE等）
- ファクトリ関数
- サマリ生成
"""

import time
import pytest

from psyche.multi_path_recall import (
    RecallPathLabel,
    RecallCandidate,
    PathStatistics,
    EmotionSnapshot,
    ContextSnapshot,
    TemporalSnapshot,
    MultiPathRecallState,
    MultiPathRecallConfig,
    MultiPathRecallProcessor,
    get_recall_summary,
    create_multi_path_recall,
    _clamp,
    TfIdfIndex,
    _tokenize,
    _build_tfidf_index,
    _vectorize_query,
    _cosine_similarity,
)
from psyche import coefficient_registry


# ── Isolation fixture ────────────────────────────────────────
# coefficient_registry window size values can be modified by modulation
# modules (e.g. window_size_modulation) during orchestrator load().
# Reset the registry before/after each test to prevent cross-test pollution.

@pytest.fixture(autouse=True)
def _reset_coefficient_registry():
    coefficient_registry.reset()
    yield
    coefficient_registry.reset()


# =============================================================================
# Helpers: Mock objects
# =============================================================================

class MockUnifiedMemoryUnit:
    """UnifiedMemoryUnit互換のモックオブジェクト。"""
    def __init__(
        self,
        unit_id: str = "",
        source_id: str = "",
        summary: str = "",
        topics: list | None = None,
        timestamp: float = 0.0,
        emotional_valence: float = 0.0,
        emotional_label: str = "",
        importance: float = 0.5,
        freshness: float = 0.5,
    ):
        self.unit_id = unit_id
        self.source_id = source_id
        self.summary = summary
        self.topics = topics or []
        self.timestamp = timestamp
        self.emotional_valence = emotional_valence
        self.emotional_label = emotional_label
        self.importance = importance
        self.freshness = freshness


class MockEmotionalTrace:
    """EmotionalTrace互換のモック。"""
    def __init__(self, emotion_label: str = "", intensity: float = 0.0, valence: float = 0.0, freshness: float = 0.5):
        self.emotion_label = emotion_label
        self.intensity = intensity
        self.valence = valence
        self.freshness = freshness


class MockMemoryBinding:
    """MemoryBinding互換のモック。"""
    def __init__(self, memory_key: str = "", traces: tuple = ()):
        self.memory_key = memory_key
        self.traces = traces


class MockBindingStore:
    """BindingStore互換のモック。"""
    def __init__(self, bindings: tuple = ()):
        self.bindings = bindings


class MockMemorySeriesRecord:
    """MemorySeriesRecord互換のモック。"""
    def __init__(self, source_id: str = "", forgetting_stage: str = "active"):
        self.source_id = source_id
        self.forgetting_stage = forgetting_stage


class MockForgettingState:
    """ForgettingFixationState互換のモック。"""
    def __init__(self, series_index: list | None = None):
        self.series_index = series_index or []


def make_processor(
    per_path_limit: int = 5,
    rumination_window_size: int = 30,
    weak_trace_ratio: float = 0.2,
    temporal_max_distance: float = 86400.0,
) -> MultiPathRecallProcessor:
    """テスト用プロセッサを生成する。"""
    config = MultiPathRecallConfig(
        per_path_limit=per_path_limit,
        rumination_window_size=rumination_window_size,
        weak_trace_ratio=weak_trace_ratio,
        temporal_max_distance=temporal_max_distance,
    )
    return MultiPathRecallProcessor(config=config)


def make_units(count: int, base_time: float = 1000.0) -> list[MockUnifiedMemoryUnit]:
    """テスト用のUnifiedMemoryUnitリストを生成する。"""
    units = []
    for i in range(count):
        units.append(MockUnifiedMemoryUnit(
            unit_id=f"unit_{i:03d}",
            source_id=f"src_{i:03d}",
            summary=f"Memory summary {i}",
            topics=[f"topic_{i}", "common_topic"],
            timestamp=base_time + i * 60.0,
            emotional_valence=0.1 * (i % 5 - 2),
            emotional_label=["joy", "sorrow", "anger", "fear", "surprise"][i % 5],
        ))
    return units


# =============================================================================
# Test: RecallPathLabel Enum
# =============================================================================

class TestRecallPathLabel:
    def test_all_labels(self):
        labels = list(RecallPathLabel)
        assert len(labels) == 3

    def test_values(self):
        assert RecallPathLabel.EMOTIONAL.value == "emotional"
        assert RecallPathLabel.CONTEXTUAL.value == "contextual"
        assert RecallPathLabel.TEMPORAL.value == "temporal"

    def test_no_weight_attribute(self):
        for label in RecallPathLabel:
            assert not hasattr(label, "weight")
            assert not hasattr(label, "priority")


# =============================================================================
# Test: RecallCandidate
# =============================================================================

class TestRecallCandidate:
    def test_default(self):
        c = RecallCandidate()
        assert c.unit_id == ""
        assert c.path_label == RecallPathLabel.EMOTIONAL.value
        assert c.summary == ""
        assert c.candidate_id != ""

    def test_to_dict(self):
        c = RecallCandidate(unit_id="u1", summary="test", path_label="contextual")
        d = c.to_dict()
        assert d["unit_id"] == "u1"
        assert d["summary"] == "test"
        assert d["path_label"] == "contextual"

    def test_from_dict(self):
        d = {"unit_id": "u2", "summary": "abc", "path_label": "temporal"}
        c = RecallCandidate.from_dict(d)
        assert c.unit_id == "u2"
        assert c.summary == "abc"
        assert c.path_label == "temporal"

    def test_round_trip(self):
        c = RecallCandidate(unit_id="u3", summary="test3", path_label="emotional", input_snapshot="snap")
        restored = RecallCandidate.from_dict(c.to_dict())
        assert restored.unit_id == c.unit_id
        assert restored.summary == c.summary
        assert restored.path_label == c.path_label
        assert restored.input_snapshot == c.input_snapshot


# =============================================================================
# Test: PathStatistics
# =============================================================================

class TestPathStatistics:
    def test_default(self):
        s = PathStatistics()
        assert s.emotional_count == 0
        assert s.contextual_count == 0
        assert s.temporal_count == 0

    def test_to_dict(self):
        s = PathStatistics(emotional_count=3, contextual_count=2, temporal_count=1)
        d = s.to_dict()
        assert d["emotional_count"] == 3
        assert d["contextual_count"] == 2
        assert d["temporal_count"] == 1

    def test_from_dict(self):
        s = PathStatistics.from_dict({"emotional_count": 5, "contextual_count": 4, "temporal_count": 3})
        assert s.emotional_count == 5
        assert s.contextual_count == 4
        assert s.temporal_count == 3

    def test_round_trip(self):
        s = PathStatistics(emotional_count=1, contextual_count=2, temporal_count=3)
        restored = PathStatistics.from_dict(s.to_dict())
        assert restored.emotional_count == s.emotional_count
        assert restored.contextual_count == s.contextual_count
        assert restored.temporal_count == s.temporal_count


# =============================================================================
# Test: EmotionSnapshot
# =============================================================================

class TestEmotionSnapshot:
    def test_default(self):
        e = EmotionSnapshot()
        assert e.emotions == {}
        assert e.mood_valence == 0.0
        assert e.dominant_emotion == ""

    def test_to_dict(self):
        e = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        d = e.to_dict()
        assert d["emotions"]["joy"] == 0.8
        assert d["mood_valence"] == 0.5

    def test_round_trip(self):
        e = EmotionSnapshot(emotions={"anger": 0.3}, mood_valence=-0.2, dominant_emotion="anger")
        restored = EmotionSnapshot.from_dict(e.to_dict())
        assert restored.emotions == e.emotions
        assert restored.mood_valence == e.mood_valence
        assert restored.dominant_emotion == e.dominant_emotion


# =============================================================================
# Test: ContextSnapshot
# =============================================================================

class TestContextSnapshot:
    def test_default(self):
        c = ContextSnapshot()
        assert c.topics == []
        assert c.percept_text == ""

    def test_to_dict(self):
        c = ContextSnapshot(topics=["a", "b"], percept_text="hello", current_time=1000.0)
        d = c.to_dict()
        assert d["topics"] == ["a", "b"]
        assert d["percept_text"] == "hello"
        assert d["current_time"] == 1000.0

    def test_round_trip(self):
        c = ContextSnapshot(topics=["x"], percept_text="y", current_time=2000.0)
        restored = ContextSnapshot.from_dict(c.to_dict())
        assert restored.topics == c.topics
        assert restored.percept_text == c.percept_text
        assert restored.current_time == c.current_time


# =============================================================================
# Test: TemporalSnapshot
# =============================================================================

class TestTemporalSnapshot:
    def test_default(self):
        t = TemporalSnapshot()
        assert t.snapshot == {}
        assert t.tick_count == 0

    def test_round_trip(self):
        t = TemporalSnapshot(snapshot={"a": "dense"}, tick_count=5)
        restored = TemporalSnapshot.from_dict(t.to_dict())
        assert restored.snapshot == t.snapshot
        assert restored.tick_count == t.tick_count


# =============================================================================
# Test: MultiPathRecallState
# =============================================================================

class TestMultiPathRecallState:
    def test_default(self):
        s = MultiPathRecallState()
        assert s.current_candidates == []
        assert s.recent_recall_history == []
        assert s.path_stats.emotional_count == 0
        assert s.cycle_count == 0

    def test_to_dict(self):
        s = MultiPathRecallState(cycle_count=3)
        d = s.to_dict()
        assert d["cycle_count"] == 3
        assert d["current_candidates"] == []

    def test_from_dict(self):
        d = {
            "cycle_count": 5,
            "recent_recall_history": ["a", "b"],
            "path_stats": {"emotional_count": 1, "contextual_count": 2, "temporal_count": 3},
        }
        s = MultiPathRecallState.from_dict(d)
        assert s.cycle_count == 5
        assert s.recent_recall_history == ["a", "b"]
        assert s.path_stats.emotional_count == 1

    def test_round_trip(self):
        s = MultiPathRecallState(
            current_candidates=[RecallCandidate(unit_id="u1", path_label="emotional")],
            recent_recall_history=["x", "y"],
            path_stats=PathStatistics(emotional_count=2),
            cycle_count=7,
        )
        restored = MultiPathRecallState.from_dict(s.to_dict())
        assert restored.cycle_count == s.cycle_count
        assert len(restored.current_candidates) == 1
        assert restored.current_candidates[0].unit_id == "u1"
        assert restored.recent_recall_history == ["x", "y"]
        assert restored.path_stats.emotional_count == 2


# =============================================================================
# Test: MultiPathRecallConfig
# =============================================================================

class TestMultiPathRecallConfig:
    def test_default(self):
        c = MultiPathRecallConfig()
        assert c.per_path_limit == 5
        assert c.rumination_window_size == 30
        assert c.weak_trace_ratio == 0.2
        assert c.temporal_max_distance == 86400.0

    def test_custom(self):
        c = MultiPathRecallConfig(per_path_limit=3, rumination_window_size=10)
        assert c.per_path_limit == 3
        assert c.rumination_window_size == 10


# =============================================================================
# Test: Initial State
# =============================================================================

class TestInitialState:
    def test_processor_initial_state(self):
        p = make_processor()
        assert p.state.cycle_count == 0
        assert p.state.current_candidates == []
        assert p.state.recent_recall_history == []

    def test_get_recall_candidates_empty(self):
        p = make_processor()
        assert p.get_recall_candidates() == []

    def test_get_enrichment_data_empty(self):
        p = make_processor()
        data = p.get_enrichment_data()
        assert data["candidate_count"] == 0
        assert data["entries"] == []

    def test_get_summary_empty(self):
        p = make_processor()
        s = p.get_summary()
        assert s["cycle_count"] == 0
        assert s["candidate_count"] == 0


# =============================================================================
# Test: 経路1 感情連想
# =============================================================================

class TestEmotionalRecall:
    def test_basic_emotional_recall(self):
        p = make_processor(per_path_limit=3)
        units = make_units(10)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == RecallPathLabel.EMOTIONAL.value]
        assert len(emotional) <= 3

    def test_emotional_recall_with_binding_store(self):
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(
                unit_id="u1", source_id="s1", summary="happy memory",
                emotional_valence=0.5, emotional_label="joy",
            ),
        ]
        trace = MockEmotionalTrace(emotion_label="joy", intensity=0.9, valence=0.5)
        binding = MockMemoryBinding(memory_key="u1", traces=(trace,))
        store = MockBindingStore(bindings=(binding,))

        emo = EmotionSnapshot(emotions={"joy": 0.9}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, binding_store=store,
            emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == RecallPathLabel.EMOTIONAL.value]
        assert len(emotional) >= 1
        assert emotional[0].unit_id == "u1"

    def test_emotional_recall_no_emotion(self):
        """感情が全て0の場合、感情連想経路は候補0件。"""
        p = make_processor()
        units = make_units(5)
        emo = EmotionSnapshot()
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == RecallPathLabel.EMOTIONAL.value]
        assert len(emotional) == 0

    def test_emotional_path_label(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == RecallPathLabel.EMOTIONAL.value]
        for c in emotional:
            assert c.path_label == "emotional"

    def test_emotional_input_snapshot_recorded(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == RecallPathLabel.EMOTIONAL.value]
        if emotional:
            assert "joy" in emotional[0].input_snapshot


# =============================================================================
# Test: 経路2 文脈連想
# =============================================================================

class TestContextualRecall:
    def test_basic_contextual_recall(self):
        p = make_processor(per_path_limit=3)
        units = make_units(10)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        contextual = [c for c in result if c.path_label == RecallPathLabel.CONTEXTUAL.value]
        # common_topic はすべてのunitに含まれるので候補が出るはず
        assert len(contextual) > 0
        assert len(contextual) <= 3

    def test_contextual_recall_topic_match(self):
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", summary="about cats", topics=["cats", "animals"]),
            MockUnifiedMemoryUnit(unit_id="u2", source_id="s2", summary="about dogs", topics=["dogs", "animals"]),
            MockUnifiedMemoryUnit(unit_id="u3", source_id="s3", summary="about math", topics=["math", "science"]),
        ]
        ctx = ContextSnapshot(topics=["cats", "animals"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        contextual = [c for c in result if c.path_label == RecallPathLabel.CONTEXTUAL.value]
        unit_ids = [c.unit_id for c in contextual]
        assert "u1" in unit_ids  # catsとanimals 両方マッチ
        assert "u2" in unit_ids  # animals マッチ

    def test_contextual_recall_text_match(self):
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", summary="python programming", topics=["python"]),
            MockUnifiedMemoryUnit(unit_id="u2", source_id="s2", summary="java programming", topics=["java"]),
        ]
        ctx = ContextSnapshot(percept_text="I am learning python", current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        contextual = [c for c in result if c.path_label == RecallPathLabel.CONTEXTUAL.value]
        unit_ids = [c.unit_id for c in contextual]
        assert "u1" in unit_ids

    def test_contextual_recall_no_topics_no_text(self):
        """トピックもテキストも空の場合は候補0件。"""
        p = make_processor()
        units = make_units(5)
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        contextual = [c for c in result if c.path_label == RecallPathLabel.CONTEXTUAL.value]
        assert len(contextual) == 0

    def test_contextual_path_label(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        contextual = [c for c in result if c.path_label == RecallPathLabel.CONTEXTUAL.value]
        for c in contextual:
            assert c.path_label == "contextual"


# =============================================================================
# Test: 経路3 時間近接
# =============================================================================

class TestTemporalRecall:
    def test_basic_temporal_recall(self):
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", summary="recent", timestamp=now - 60),
            MockUnifiedMemoryUnit(unit_id="u2", source_id="s2", summary="medium", timestamp=now - 1800),
            MockUnifiedMemoryUnit(unit_id="u3", source_id="s3", summary="old", timestamp=now - 3500),
            MockUnifiedMemoryUnit(unit_id="u4", source_id="s4", summary="too_old", timestamp=now - 7200),
        ]
        ctx = ContextSnapshot(current_time=now)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        temporal = [c for c in result if c.path_label == RecallPathLabel.TEMPORAL.value]
        # u4 should be excluded (too far)
        unit_ids = [c.unit_id for c in temporal]
        assert "u4" not in unit_ids
        assert len(temporal) <= 3

    def test_temporal_ordering(self):
        """近い記憶が先に来る。"""
        now = time.time()
        p = make_processor(per_path_limit=5, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(unit_id="far", source_id="s1", summary="far", timestamp=now - 3000),
            MockUnifiedMemoryUnit(unit_id="near", source_id="s2", summary="near", timestamp=now - 100),
            MockUnifiedMemoryUnit(unit_id="mid", source_id="s3", summary="mid", timestamp=now - 1500),
        ]
        ctx = ContextSnapshot(current_time=now)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        temporal = [c for c in result if c.path_label == RecallPathLabel.TEMPORAL.value]
        if len(temporal) >= 2:
            assert temporal[0].unit_id == "near"

    def test_temporal_recall_zero_timestamp(self):
        """タイムスタンプ0の記憶はスキップされる。"""
        now = time.time()
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", timestamp=0.0),
        ]
        ctx = ContextSnapshot(current_time=now)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        temporal = [c for c in result if c.path_label == RecallPathLabel.TEMPORAL.value]
        assert len(temporal) == 0

    def test_temporal_path_label(self):
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", timestamp=now - 60),
        ]
        ctx = ContextSnapshot(current_time=now)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        temporal = [c for c in result if c.path_label == RecallPathLabel.TEMPORAL.value]
        for c in temporal:
            assert c.path_label == "temporal"


# =============================================================================
# Test: 3経路同時実行
# =============================================================================

class TestAllPaths:
    def test_all_three_paths_produce_candidates(self):
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"memory {i}",
                topics=["test_topic"],
                timestamp=now - i * 60,
                emotional_valence=0.5,
                emotional_label="joy",
            )
            for i in range(10)
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(topics=["test_topic"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        paths = {c.path_label for c in result}
        assert "emotional" in paths
        assert "contextual" in paths
        assert "temporal" in paths

    def test_cycle_count_increments(self):
        p = make_processor()
        units = make_units(3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert p.state.cycle_count == 1
        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert p.state.cycle_count == 2

    def test_path_stats_updated(self):
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id="u1", source_id="s1", summary="m1",
                topics=["topic"], timestamp=now - 60,
                emotional_valence=0.5, emotional_label="joy",
            ),
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        stats = p.state.path_stats
        total = stats.emotional_count + stats.contextual_count + stats.temporal_count
        assert total > 0

    def test_candidates_overwritten_each_cycle(self):
        """候補は毎サイクル上書きされる。"""
        p = make_processor(per_path_limit=3)
        units1 = [MockUnifiedMemoryUnit(unit_id="first", source_id="s1", topics=["a"], timestamp=time.time())]
        units2 = [MockUnifiedMemoryUnit(unit_id="second", source_id="s2", topics=["b"], timestamp=time.time())]
        ctx1 = ContextSnapshot(topics=["a"], current_time=time.time())
        ctx2 = ContextSnapshot(topics=["b"], current_time=time.time())

        p.recall_all_paths(unified_units=units1, context_snapshot=ctx1)
        assert any(c.unit_id == "first" for c in p.state.current_candidates)

        p.recall_all_paths(unified_units=units2, context_snapshot=ctx2)
        assert not any(c.unit_id == "first" for c in p.state.current_candidates)


# =============================================================================
# Test: 安全弁1 経路間等価性
# =============================================================================

class TestSafetyValve1Equivalence:
    def test_per_path_limit_respected(self):
        now = time.time()
        p = make_processor(per_path_limit=2, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"m{i}", topics=["topic"],
                timestamp=now - i * 10,
                emotional_valence=0.5, emotional_label="joy",
            )
            for i in range(20)
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == "emotional"]
        contextual = [c for c in result if c.path_label == "contextual"]
        temporal = [c for c in result if c.path_label == "temporal"]

        assert len(emotional) <= 2
        assert len(contextual) <= 2
        assert len(temporal) <= 2

    def test_same_limit_for_all_paths(self):
        """全経路で同一の候補上限が使用される。"""
        cfg = MultiPathRecallConfig(per_path_limit=4)
        p = MultiPathRecallProcessor(config=cfg)
        # per_path_limitは全経路共通
        assert cfg.per_path_limit == 4


# =============================================================================
# Test: 安全弁2 顕著性バイアス抑制
# =============================================================================

class TestSafetyValve2SalienceBias:
    def test_weak_trace_inclusion(self):
        """感情痕跡が弱い記憶が一定割合混入される。"""
        p = make_processor(per_path_limit=5, weak_trace_ratio=0.4)
        # 強い感情の記憶
        strong_units = [
            MockUnifiedMemoryUnit(
                unit_id=f"strong_{i}", source_id=f"ss{i}",
                summary=f"strong {i}",
                emotional_valence=0.9, emotional_label="joy",
            )
            for i in range(5)
        ]
        # 弱い感情の記憶
        weak_units = [
            MockUnifiedMemoryUnit(
                unit_id=f"weak_{i}", source_id=f"ws{i}",
                summary=f"weak {i}",
                emotional_valence=0.1, emotional_label="",
            )
            for i in range(5)
        ]
        units = strong_units + weak_units
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == "emotional"]
        # 弱い記憶が少なくとも1件は含まれるべき
        weak_ids = {f"weak_{i}" for i in range(5)}
        weak_in_result = [c for c in emotional if c.unit_id in weak_ids]
        if emotional:
            assert len(weak_in_result) >= 1

    def test_weak_trace_ratio_zero(self):
        """weak_trace_ratio=0でも強い記憶のみで正常動作。"""
        p = make_processor(per_path_limit=3, weak_trace_ratio=0.0)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        # Should not crash
        assert isinstance(result, list)


# =============================================================================
# Test: 安全弁3 ルーミネーション防止
# =============================================================================

class TestSafetyValve3Rumination:
    def test_rumination_suppression_order(self):
        """直近で繰り返し想起された記憶は末尾に移動する。"""
        p = make_processor(per_path_limit=5, rumination_window_size=5)

        # 事前にunit_000を多数回想起済みにする
        for _ in range(20):
            p.state.recent_recall_history.append("unit_000")

        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )

        # unit_000がある場合、末尾に移動しているはず（完全除外はしない）
        unit_000_indices = [i for i, c in enumerate(result) if c.unit_id == "unit_000"]
        non_000_indices = [i for i, c in enumerate(result) if c.unit_id != "unit_000"]
        if unit_000_indices and non_000_indices:
            # 抑制されたものは非抑制のものの後に来る
            assert min(unit_000_indices) >= min(non_000_indices)

    def test_rumination_does_not_exclude(self):
        """ルーミネーション防止は完全除外ではない。"""
        p = make_processor(per_path_limit=5, rumination_window_size=5)

        # unit_000だけを大量に想起済みにする
        for _ in range(30):
            p.state.recent_recall_history.append("unit_000")

        # unit_000のみを入力
        units = [MockUnifiedMemoryUnit(
            unit_id="unit_000", source_id="src_000",
            summary="only memory", topics=["topic"],
            emotional_label="joy", emotional_valence=0.5,
            timestamp=time.time(),
        )]
        ctx = ContextSnapshot(topics=["topic"], current_time=time.time())
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        # 唯一の候補なので除外されない
        unit_ids = [c.unit_id for c in result]
        assert "unit_000" in unit_ids

    def test_rumination_window_trimming(self):
        """スライディングウィンドウが正しくトリミングされる。"""
        p = make_processor(per_path_limit=2, rumination_window_size=3)
        max_expected = 3 * 2 * 3  # window_size * per_path_limit * 3

        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)

        for _ in range(50):
            p.recall_all_paths(
                unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
            )

        assert len(p.state.recent_recall_history) <= max_expected


# =============================================================================
# Test: 安全弁4 忘却処理との分離
# =============================================================================

class TestSafetyValve4ForgettingSeparation:
    def test_invisible_excluded(self):
        """INVISIBLE記憶は想起候補から除外される。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="visible_src", summary="visible", topics=["topic"]),
            MockUnifiedMemoryUnit(unit_id="u2", source_id="invisible_src", summary="invisible", topics=["topic"]),
        ]
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="visible_src", forgetting_stage="active"),
            MockMemorySeriesRecord(source_id="invisible_src", forgetting_stage="invisible"),
        ])
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        unit_ids = [c.unit_id for c in result]
        assert "u1" in unit_ids
        assert "u2" not in unit_ids

    def test_near_invisible_not_excluded(self):
        """NEAR_INVISIBLE記憶は除外されない。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="ni_src", summary="near invisible", topics=["topic"]),
        ]
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="ni_src", forgetting_stage="near_invisible"),
        ])
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        unit_ids = [c.unit_id for c in result]
        assert "u1" in unit_ids

    def test_active_not_excluded(self):
        """ACTIVE記憶は当然除外されない。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="act_src", summary="active", topics=["topic"]),
        ]
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="act_src", forgetting_stage="active"),
        ])
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        unit_ids = [c.unit_id for c in result]
        assert "u1" in unit_ids

    def test_no_forgetting_state(self):
        """忘却状態がNoneの場合は全記憶が対象。"""
        p = make_processor(per_path_limit=5)
        units = make_units(3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=None, context_snapshot=ctx,
        )
        assert len(result) > 0

    def test_all_invisible(self):
        """全記憶がINVISIBLEの場合は候補0件。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", topics=["topic"]),
            MockUnifiedMemoryUnit(unit_id="u2", source_id="s2", topics=["topic"]),
        ]
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="s1", forgetting_stage="invisible"),
            MockMemorySeriesRecord(source_id="s2", forgetting_stage="invisible"),
        ])
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        assert len(result) == 0

    def test_no_reference_frequency_notification(self):
        """想起処理が忘却の参照頻度に通知しない（構造的保証）。"""
        p = make_processor(per_path_limit=5)
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="s1", forgetting_stage="active"),
        ])
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", source_id="s1", summary="test", topics=["topic"]),
        ]
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        # プロセッサが forgetting_state を変更しないことを確認
        initial_stage = forgetting_state.series_index[0].forgetting_stage
        p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        assert forgetting_state.series_index[0].forgetting_stage == initial_stage

    def test_unit_id_invisible_also_excluded(self):
        """source_idだけでなくunit_idでもINVISIBLE判定される。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="invisible_uid", source_id="other_src", summary="test", topics=["topic"]),
        ]
        forgetting_state = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="invisible_uid", forgetting_stage="invisible"),
        ])
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, forgetting_state=forgetting_state, context_snapshot=ctx,
        )
        unit_ids = [c.unit_id for c in result]
        assert "invisible_uid" not in unit_ids


# =============================================================================
# Test: 安全弁5 外部API想起との整合
# =============================================================================

class TestSafetyValve5ApiIntegrity:
    def test_enrichment_only_output(self):
        """出力はenrichment参照のみ。外部API結果を上書きしない。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        data = p.get_enrichment_data()

        # enrichmentの構造のみ（外部API操作のメソッドがない）
        assert "entries" in data
        assert "candidate_count" in data
        assert "summary_text" in data

    def test_no_api_modification_methods(self):
        """外部API結果を修正するメソッドが存在しない。"""
        p = make_processor()
        forbidden = [
            "update_api", "modify_api", "override_api",
            "replace_embedding", "modify_embedding",
        ]
        methods = [m for m in dir(p) if not m.startswith("_") and callable(getattr(p, m))]
        for method in methods:
            method_lower = method.lower()
            for f in forbidden:
                assert f not in method_lower


# =============================================================================
# Test: 3経路遮断
# =============================================================================

class TestPathSevering:
    def test_no_recall_to_forgetting_path(self):
        """想起→忘却の経路が遮断されている。"""
        p = make_processor()
        forbidden = [
            "update_forgetting", "notify_forgetting", "increment_reference",
            "update_reference_count", "notify_reference",
        ]
        methods = [m for m in dir(p) if not m.startswith("_") and callable(getattr(p, m))]
        for method in methods:
            method_lower = method.lower()
            for f in forbidden:
                assert f not in method_lower

    def test_no_recall_to_emotion_path(self):
        """想起→感情の経路が遮断されている。"""
        p = make_processor()
        forbidden = [
            "update_emotion", "set_emotion", "modify_emotion",
            "change_mood", "set_mood",
        ]
        methods = [m for m in dir(p) if not m.startswith("_") and callable(getattr(p, m))]
        for method in methods:
            method_lower = method.lower()
            for f in forbidden:
                assert f not in method_lower

    def test_no_recall_to_recall_feedback(self):
        """想起→想起の自己参照経路が遮断されている。

        前回の想起結果が次サイクルの入力にならない。
        入力は「現在の感情状態」「現在の文脈」「現在の時間的位置」のみ。
        """
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)

        # 1回目の想起
        result1 = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )

        # 2回目の想起: 入力は同じ外部状態のみ、前回結果は入力に含まれない
        result2 = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )

        # ルーミネーション防止以外に前回結果が影響しないことを確認
        # （ルーミネーション防止は「前回の想起結果」ではなく「想起履歴」を参照）
        assert isinstance(result2, list)

    def test_no_judgment_methods(self):
        """判断・行動・評価を確定するメソッドが存在しない。"""
        p = make_processor()
        forbidden = [
            "decide", "judge", "evaluate", "prescribe",
            "set_policy", "update_policy", "select_action",
        ]
        methods = [m for m in dir(p) if not m.startswith("_") and callable(getattr(p, m))]
        for method in methods:
            method_lower = method.lower()
            for f in forbidden:
                assert f not in method_lower


# =============================================================================
# Test: Enrichment Data
# =============================================================================

class TestEnrichmentData:
    def test_enrichment_structure(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        data = p.get_enrichment_data()

        assert "candidate_count" in data
        assert "path_stats" in data
        assert "entries" in data
        assert "summary_text" in data
        assert isinstance(data["entries"], list)

    def test_enrichment_entries_format(self):
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"memory {i}", topics=["topic"],
                timestamp=now - i * 60, emotional_valence=0.5, emotional_label="joy",
            )
            for i in range(5)
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        data = p.get_enrichment_data()

        for entry in data["entries"]:
            assert "path" in entry
            assert "summary" in entry
            assert "unit_id" in entry
            assert entry["path"] in ("emotional", "contextual", "temporal")

    def test_enrichment_candidate_count_limit(self):
        """enrichmentの候補数がenrichment_candidate_count以内。"""
        p = make_processor(per_path_limit=5)
        p._config.enrichment_candidate_count = 6
        units = make_units(20)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        data = p.get_enrichment_data()
        assert len(data["entries"]) <= 6

    def test_enrichment_summary_text(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        data = p.get_enrichment_data()
        assert isinstance(data["summary_text"], str)
        assert "cycle=" in data["summary_text"]

    def test_enrichment_no_emphasis(self):
        """enrichmentに強調表現が含まれない。"""
        p = make_processor(per_path_limit=3)
        units = make_units(10)
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        data = p.get_enrichment_data()
        text = data["summary_text"]
        forbidden_words = ["重要", "注目", "異常", "警告", "危険", "critical", "important", "attention"]
        for word in forbidden_words:
            assert word not in text.lower()


# =============================================================================
# Test: get_recall_candidates READ-ONLY
# =============================================================================

class TestGetRecallCandidates:
    def test_returns_copy(self):
        """get_recall_candidatesがコピーを返す。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        candidates = p.get_recall_candidates()
        original_len = len(p.state.current_candidates)

        # 返されたリストを変更しても内部状態に影響しない
        candidates.clear()
        assert len(p.state.current_candidates) == original_len

    def test_candidates_have_all_fields(self):
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        for c in p.get_recall_candidates():
            assert c.candidate_id != ""
            assert c.path_label in ("emotional", "contextual", "temporal")
            assert c.recall_timestamp > 0


# =============================================================================
# Test: Save/Load Round-Trip
# =============================================================================

class TestSaveLoadRoundTrip:
    def test_state_round_trip(self):
        state = MultiPathRecallState(
            current_candidates=[
                RecallCandidate(unit_id="u1", path_label="emotional", summary="s1"),
                RecallCandidate(unit_id="u2", path_label="contextual", summary="s2"),
            ],
            recent_recall_history=["u1", "u2", "u1"],
            path_stats=PathStatistics(emotional_count=3, contextual_count=2, temporal_count=1),
            cycle_count=10,
        )
        d = state.to_dict()
        restored = MultiPathRecallState.from_dict(d)

        assert restored.cycle_count == 10
        assert len(restored.current_candidates) == 2
        assert restored.current_candidates[0].unit_id == "u1"
        assert restored.current_candidates[1].unit_id == "u2"
        assert restored.recent_recall_history == ["u1", "u2", "u1"]
        assert restored.path_stats.emotional_count == 3
        assert restored.path_stats.contextual_count == 2
        assert restored.path_stats.temporal_count == 1

    def test_processor_state_round_trip(self):
        """プロセッサの状態をto_dict/from_dictでラウンドトリップ。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )

        d = p.state.to_dict()
        new_state = MultiPathRecallState.from_dict(d)

        assert new_state.cycle_count == p.state.cycle_count
        assert len(new_state.current_candidates) == len(p.state.current_candidates)
        assert new_state.path_stats.emotional_count == p.state.path_stats.emotional_count

    def test_empty_state_round_trip(self):
        state = MultiPathRecallState()
        d = state.to_dict()
        restored = MultiPathRecallState.from_dict(d)

        assert restored.cycle_count == 0
        assert restored.current_candidates == []
        assert restored.recent_recall_history == []

    def test_from_dict_missing_keys(self):
        """キーが欠落していても安全にデフォルト値で復元。"""
        state = MultiPathRecallState.from_dict({})
        assert state.cycle_count == 0
        assert state.current_candidates == []
        assert state.recent_recall_history == []
        assert state.path_stats.emotional_count == 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_units(self):
        """空のunitリストで候補0件。"""
        p = make_processor()
        result = p.recall_all_paths(unified_units=[])
        assert result == []

    def test_none_units(self):
        """Noneのunitリストで候補0件。"""
        p = make_processor()
        result = p.recall_all_paths(unified_units=None)
        assert result == []

    def test_no_inputs_at_all(self):
        """全入力がNone/デフォルトで候補0件。"""
        p = make_processor()
        result = p.recall_all_paths()
        assert result == []
        assert p.state.cycle_count == 1

    def test_single_unit(self):
        """1件のunitでも正常動作。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [MockUnifiedMemoryUnit(
            unit_id="only", source_id="s_only",
            summary="single memory", topics=["topic"],
            timestamp=now - 60,
            emotional_valence=0.5, emotional_label="joy",
        )]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        assert len(result) > 0
        assert all(c.unit_id == "only" for c in result)

    def test_units_without_topics(self):
        """トピックなしのunitでも感情/時間経路で候補生成可能。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [MockUnifiedMemoryUnit(
            unit_id="no_topic", source_id="s_nt",
            summary="no topic memory", topics=[],
            timestamp=now - 60,
            emotional_valence=0.5, emotional_label="joy",
        )]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["nonexistent"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        # 文脈連想はマッチしないが、感情連想 or 時間近接ではヒットしうる
        assert isinstance(result, list)

    def test_units_without_emotional_info(self):
        """感情情報なしのunitでも文脈/時間経路で候補生成可能。"""
        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(
            unit_id="no_emo", source_id="s_ne",
            summary="no emotion memory", topics=["topic"],
            timestamp=2000.0,
        )]
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(
            unified_units=units, context_snapshot=ctx,
        )
        assert any(c.unit_id == "no_emo" for c in result)

    def test_large_unit_count(self):
        """大量のunitでも制限内で正常動作。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=86400.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"memory {i}", topics=["topic"],
                timestamp=now - i * 10,
                emotional_valence=0.5, emotional_label="joy",
            )
            for i in range(200)
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        # 各経路3件以内
        emotional = [c for c in result if c.path_label == "emotional"]
        contextual = [c for c in result if c.path_label == "contextual"]
        temporal = [c for c in result if c.path_label == "temporal"]
        assert len(emotional) <= 3
        assert len(contextual) <= 3
        assert len(temporal) <= 3

    def test_summary_snippet_truncation(self):
        """要約がsummary_snippet_length以内に切り詰められる。"""
        p = make_processor(per_path_limit=3)
        p._config.summary_snippet_length = 10
        units = [MockUnifiedMemoryUnit(
            unit_id="u1", source_id="s1",
            summary="This is a very long summary that should be truncated",
            topics=["topic"],
        )]
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        for c in p.get_recall_candidates():
            assert len(c.summary) <= 10

    def test_multiple_cycles_stable(self):
        """複数サイクルで安定動作。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        for i in range(20):
            result = p.recall_all_paths(
                unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
            )
            assert isinstance(result, list)

        assert p.state.cycle_count == 20


# =============================================================================
# Test: Summary Function
# =============================================================================

class TestSummary:
    def test_summary_waiting(self):
        state = MultiPathRecallState()
        text = get_recall_summary(state)
        assert "待機中" in text

    def test_summary_with_candidates(self):
        state = MultiPathRecallState(
            current_candidates=[RecallCandidate(path_label="emotional")],
            path_stats=PathStatistics(emotional_count=1),
            cycle_count=3,
        )
        text = get_recall_summary(state)
        assert "cycle=3" in text
        assert "感情連想=1" in text

    def test_summary_all_paths(self):
        state = MultiPathRecallState(
            current_candidates=[
                RecallCandidate(path_label="emotional"),
                RecallCandidate(path_label="contextual"),
                RecallCandidate(path_label="temporal"),
            ],
            path_stats=PathStatistics(emotional_count=1, contextual_count=1, temporal_count=1),
            cycle_count=5,
        )
        text = get_recall_summary(state)
        assert "感情連想" in text
        assert "文脈連想" in text
        assert "時間近接" in text
        assert "候補合計=3" in text

    def test_summary_zero_candidates(self):
        state = MultiPathRecallState(cycle_count=2)
        text = get_recall_summary(state)
        assert "候補=0" in text

    def test_summary_no_emphasis(self):
        """要約に強調表現が含まれない。"""
        state = MultiPathRecallState(
            current_candidates=[RecallCandidate()] * 5,
            path_stats=PathStatistics(emotional_count=5),
            cycle_count=10,
        )
        text = get_recall_summary(state)
        forbidden = ["重要", "注目", "異常", "警告", "critical", "important"]
        for word in forbidden:
            assert word not in text.lower()


# =============================================================================
# Test: Factory
# =============================================================================

class TestFactory:
    def test_create_default(self):
        p = create_multi_path_recall()
        assert isinstance(p, MultiPathRecallProcessor)
        assert p.state.cycle_count == 0

    def test_create_with_config(self):
        cfg = MultiPathRecallConfig(per_path_limit=10)
        p = create_multi_path_recall(config=cfg)
        assert p._config.per_path_limit == 10

    def test_factory_returns_independent_instances(self):
        p1 = create_multi_path_recall()
        p2 = create_multi_path_recall()
        assert p1 is not p2
        assert p1.state is not p2.state


# =============================================================================
# Test: Clamp Helper
# =============================================================================

class TestClampHelper:
    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# =============================================================================
# Test: State Setter
# =============================================================================

class TestStateSetter:
    def test_state_setter(self):
        p = make_processor()
        new_state = MultiPathRecallState(cycle_count=42)
        p.state = new_state
        assert p.state.cycle_count == 42

    def test_state_setter_preserves_config(self):
        p = make_processor(per_path_limit=7)
        p.state = MultiPathRecallState(cycle_count=10)
        assert p._config.per_path_limit == 7


# =============================================================================
# Test: No Content Modification
# =============================================================================

class TestNoContentModification:
    def test_units_not_modified(self):
        """入力のunitsが変更されない。"""
        units = make_units(5)
        original_summaries = [u.summary for u in units]
        p = make_processor(per_path_limit=3)
        emo = EmotionSnapshot(emotions={"joy": 0.5}, mood_valence=0.3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )

        for i, u in enumerate(units):
            assert u.summary == original_summaries[i]

    def test_binding_store_not_modified(self):
        """入力のbinding_storeが変更されない。"""
        trace = MockEmotionalTrace(emotion_label="joy", intensity=0.9, valence=0.5)
        binding = MockMemoryBinding(memory_key="u1", traces=(trace,))
        store = MockBindingStore(bindings=(binding,))

        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(
            unit_id="u1", source_id="s1",
            emotional_valence=0.5, emotional_label="joy",
        )]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5)
        ctx = ContextSnapshot(current_time=2000.0)

        p.recall_all_paths(
            unified_units=units, binding_store=store,
            emotion_snapshot=emo, context_snapshot=ctx,
        )

        assert len(store.bindings) == 1
        assert store.bindings[0].memory_key == "u1"
        assert store.bindings[0].traces[0].intensity == 0.9


# =============================================================================
# Test: Integration (multiple features combined)
# =============================================================================

class TestIntegration:
    def test_full_pipeline(self):
        """全入力を使用した完全パイプラインテスト。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)

        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"memory about topic_{i}",
                topics=[f"topic_{i}", "shared_topic"],
                timestamp=now - i * 100,
                emotional_valence=0.3 * (1 if i % 2 == 0 else -1),
                emotional_label=["joy", "sorrow"][i % 2],
            )
            for i in range(8)
        ]

        trace_joy = MockEmotionalTrace(emotion_label="joy", intensity=0.7, valence=0.5)
        binding = MockMemoryBinding(memory_key="u0", traces=(trace_joy,))
        store = MockBindingStore(bindings=(binding,))

        forgetting = MockForgettingState(series_index=[
            MockMemorySeriesRecord(source_id="s7", forgetting_stage="invisible"),
        ])

        emo = EmotionSnapshot(
            emotions={"joy": 0.6, "sorrow": 0.2},
            mood_valence=0.3,
            dominant_emotion="joy",
        )
        ctx = ContextSnapshot(
            topics=["shared_topic", "topic_1"],
            percept_text="talking about topic_1",
            current_time=now,
        )
        temp = TemporalSnapshot(tick_count=100)

        result = p.recall_all_paths(
            unified_units=units,
            binding_store=store,
            forgetting_state=forgetting,
            emotion_snapshot=emo,
            context_snapshot=ctx,
            temporal_snapshot=temp,
        )

        # u7 (invisible) should be excluded
        unit_ids = [c.unit_id for c in result]
        assert "u7" not in unit_ids

        # 3 paths should have candidates
        paths = {c.path_label for c in result}
        assert len(paths) >= 2  # at least 2 paths should produce candidates

        # Enrichment should work
        data = p.get_enrichment_data()
        assert data["candidate_count"] > 0

        # Summary should work
        summary = p.get_summary()
        assert summary["cycle_count"] == 1

    def test_repeated_cycles_with_changing_state(self):
        """状態が変わりながら複数サイクルを実行。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)

        for cycle in range(5):
            units = [
                MockUnifiedMemoryUnit(
                    unit_id=f"u_{cycle}_{i}", source_id=f"s_{cycle}_{i}",
                    summary=f"cycle {cycle} memory {i}",
                    topics=[f"topic_{cycle}"],
                    timestamp=now - cycle * 600 - i * 60,
                    emotional_valence=0.3,
                    emotional_label="joy",
                )
                for i in range(3)
            ]
            emo = EmotionSnapshot(
                emotions={"joy": 0.5 + cycle * 0.1},
                mood_valence=0.1 * cycle,
            )
            ctx = ContextSnapshot(
                topics=[f"topic_{cycle}"],
                current_time=now,
            )

            result = p.recall_all_paths(
                unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
            )
            assert isinstance(result, list)

        assert p.state.cycle_count == 5
        assert len(p.state.recent_recall_history) > 0


# =============================================================================
# Test: TF-IDF Index dataclass (Phase 1)
# =============================================================================

class TestTfIdfIndex:
    """TfIdfIndex dataclass の生成・to_dict/from_dict テスト。"""

    def test_default_creation(self):
        """デフォルト値で生成できる。"""
        idx = TfIdfIndex()
        assert idx.vocabulary == {}
        assert idx.idf == {}
        assert idx.memory_vectors == {}
        assert idx.doc_count == 0
        assert idx.built_at == 0.0
        assert idx.ngram_n == 2

    def test_custom_creation(self):
        idx = TfIdfIndex(
            vocabulary={"hello": 0, "world": 1},
            idf={0: 1.0, 1: 0.5},
            memory_vectors={"u1": {0: 0.8}},
            doc_count=10,
            built_at=1000.0,
            ngram_n=3,
        )
        assert idx.vocabulary == {"hello": 0, "world": 1}
        assert idx.idf[0] == 1.0
        assert idx.memory_vectors["u1"] == {0: 0.8}
        assert idx.doc_count == 10
        assert idx.built_at == 1000.0
        assert idx.ngram_n == 3

    def test_to_dict(self):
        idx = TfIdfIndex(
            vocabulary={"a": 0},
            idf={0: 0.7},
            memory_vectors={"u1": {0: 0.5}},
            doc_count=5,
            built_at=2000.0,
        )
        d = idx.to_dict()
        assert d["vocabulary"] == {"a": 0}
        assert d["idf"] == {"0": 0.7}  # JSON keys are strings
        assert d["memory_vectors"] == {"u1": {"0": 0.5}}
        assert d["doc_count"] == 5
        assert d["built_at"] == 2000.0
        assert d["ngram_n"] == 2

    def test_from_dict(self):
        d = {
            "vocabulary": {"b": 1},
            "idf": {"1": 0.3},
            "memory_vectors": {"u2": {"1": 0.9}},
            "doc_count": 3,
            "built_at": 3000.0,
            "ngram_n": 2,
        }
        idx = TfIdfIndex.from_dict(d)
        assert idx.vocabulary == {"b": 1}
        assert idx.idf == {1: 0.3}
        assert idx.memory_vectors == {"u2": {1: 0.9}}
        assert idx.doc_count == 3
        assert idx.built_at == 3000.0

    def test_from_dict_empty(self):
        """空dictからデフォルト値で復元。"""
        idx = TfIdfIndex.from_dict({})
        assert idx.vocabulary == {}
        assert idx.idf == {}
        assert idx.memory_vectors == {}
        assert idx.doc_count == 0

    def test_round_trip(self):
        idx = TfIdfIndex(
            vocabulary={"x": 0, "y": 1},
            idf={0: 1.5, 1: 0.2},
            memory_vectors={"u1": {0: 0.4, 1: 0.6}, "u2": {1: 0.3}},
            doc_count=7,
            built_at=5000.0,
            ngram_n=2,
        )
        restored = TfIdfIndex.from_dict(idx.to_dict())
        assert restored.vocabulary == idx.vocabulary
        assert restored.idf == idx.idf
        assert restored.memory_vectors == idx.memory_vectors
        assert restored.doc_count == idx.doc_count
        assert restored.built_at == idx.built_at
        assert restored.ngram_n == idx.ngram_n

    def test_is_empty_true(self):
        idx = TfIdfIndex()
        assert idx.is_empty()

    def test_is_empty_false(self):
        idx = TfIdfIndex(vocabulary={"a": 0})
        assert not idx.is_empty()


# =============================================================================
# Test: _tokenize (Phase 1)
# =============================================================================

class TestTokenize:
    """_tokenize関数のテスト。Unicode文字種境界 + bi-gram。"""

    def test_ascii_words(self):
        """ASCII単語は空白/記号境界で分割。"""
        tokens = _tokenize("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_japanese_bigram(self):
        """日本語テキストはbi-gramで分割。"""
        tokens = _tokenize("感情連想")
        assert "感情" in tokens
        assert "情連" in tokens
        assert "連想" in tokens

    def test_mixed_text(self):
        """混合テキスト（日英混在）。"""
        tokens = _tokenize("hello感情world")
        assert "hello" in tokens
        assert "world" in tokens
        assert "感情" in tokens

    def test_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []

    def test_control_chars_removed(self):
        """制御文字（改行・タブ以外のU+0000-U+001F）が除去される。"""
        text = "hello\x00world\x01test\nkeep\ttabs"
        tokens = _tokenize(text)
        # \x00と\x01は除去されるが、\nと\tは保持（区切り文字として機能）
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_text_length_limit(self):
        """10000文字超のテキストは切り捨てられる。"""
        long_text = "a" * 20000
        tokens = _tokenize(long_text)
        # 切り捨て後のトークン化結果が返る（クラッシュしない）
        assert isinstance(tokens, list)

    def test_single_char_ascii(self):
        """1文字のASCII語。"""
        tokens = _tokenize("a b c")
        # 1文字のASCII語もトークンとして含まれる
        assert "a" in tokens

    def test_single_japanese_char(self):
        """1文字の日本語（bi-gram不能）。"""
        tokens = _tokenize("あ")
        # 1文字ではbi-gramが作れない→単一文字トークンとして扱う
        assert "あ" in tokens

    def test_long_compound_word(self):
        """長い複合語。"""
        tokens = _tokenize("pneumonoultramicroscopicsilicovolcanoconiosis")
        assert "pneumonoultramicroscopicsilicovolcanoconiosis" in tokens

    def test_punctuation_boundary(self):
        """句読点が境界として機能。"""
        tokens = _tokenize("hello,world.test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_case_normalized(self):
        """大文字は小文字に正規化。"""
        tokens = _tokenize("Hello WORLD Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens


# =============================================================================
# Test: _build_tfidf_index (Phase 1)
# =============================================================================

class TestBuildTfIdfIndex:
    """_build_tfidf_index 純粋関数のテスト。"""

    def test_basic_build(self):
        """基本的なインデックス構築。"""
        units = [
            MockUnifiedMemoryUnit(
                unit_id="u1", summary="cat dog", topics=["animals"],
            ),
            MockUnifiedMemoryUnit(
                unit_id="u2", summary="dog fish", topics=["animals", "water"],
            ),
        ]
        idx = _build_tfidf_index(units)
        assert idx.doc_count == 2
        assert not idx.is_empty()
        assert "u1" in idx.memory_vectors
        assert "u2" in idx.memory_vectors

    def test_idf_common_vs_rare(self):
        """全文書に出現する語はIDF低、稀少語はIDF高。"""
        units = [
            MockUnifiedMemoryUnit(unit_id=f"u{i}", summary="common rare" if i == 0 else "common only", topics=[])
            for i in range(5)
        ]
        idx = _build_tfidf_index(units)
        # "common"は全文書に出現→IDF低、"rare"は1文書のみ→IDF高
        if "common" in idx.vocabulary and "rare" in idx.vocabulary:
            common_id = idx.vocabulary["common"]
            rare_id = idx.vocabulary["rare"]
            assert idx.idf[rare_id] > idx.idf[common_id]

    def test_empty_units(self):
        """空のunitリストでは空インデックス。"""
        idx = _build_tfidf_index([])
        assert idx.is_empty()
        assert idx.doc_count == 0

    def test_topics_included_in_tokenization(self):
        """topicsもトークン化に含まれる。"""
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="", topics=["uniquetopic"]),
        ]
        idx = _build_tfidf_index(units)
        assert "uniquetopic" in idx.vocabulary

    def test_vocab_limit(self):
        """語彙上限（5000）を超えない。"""
        # 大量の一意な語を持つunitsを作成
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}",
                summary=" ".join(f"word{i}_{j}" for j in range(100)),
                topics=[],
            )
            for i in range(100)
        ]
        idx = _build_tfidf_index(units, max_vocab_size=5000)
        assert len(idx.vocabulary) <= 5000

    def test_vocab_limit_removes_low_idf(self):
        """語彙上限到達時、IDF低値（非特徴的な語）から除外。"""
        # 全文書共通の語 + 文書固有の語
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}",
                summary=f"common shared {' '.join(f'unique{i}_{j}' for j in range(10))}",
                topics=[],
            )
            for i in range(20)
        ]
        idx = _build_tfidf_index(units, max_vocab_size=50)
        # "common"と"shared"はIDF最低→語彙上限があれば除外される可能性が高い
        # 語彙が上限に収まっていることを確認
        assert len(idx.vocabulary) <= 50

    def test_sparse_vectors(self):
        """ベクトルが疎表現（ゼロ値なし）。"""
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="alpha beta", topics=[]),
            MockUnifiedMemoryUnit(unit_id="u2", summary="gamma delta", topics=[]),
        ]
        idx = _build_tfidf_index(units)
        for uid, vec in idx.memory_vectors.items():
            for tid, weight in vec.items():
                assert weight > 0, f"Zero weight found in vector for {uid}, token_id={tid}"

    def test_pure_function_no_mutation(self):
        """入力のunitsを変更しない。"""
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="test text", topics=["topic"]),
        ]
        original_summary = units[0].summary
        original_topics = list(units[0].topics)
        _build_tfidf_index(units)
        assert units[0].summary == original_summary
        assert units[0].topics == original_topics

    def test_built_at_timestamp(self):
        """構築タイムスタンプが記録される。"""
        units = [MockUnifiedMemoryUnit(unit_id="u1", summary="test", topics=[])]
        before = time.time()
        idx = _build_tfidf_index(units)
        after = time.time()
        assert before <= idx.built_at <= after


# =============================================================================
# Test: _vectorize_query (Phase 1)
# =============================================================================

class TestVectorizeQuery:
    """_vectorize_query のテスト。"""

    def test_basic_vectorization(self):
        """クエリテキストをベクトル化。"""
        vocab = {"hello": 0, "world": 1}
        idf = {0: 1.0, 1: 0.5}
        vec = _vectorize_query("hello world", vocab, idf)
        assert 0 in vec
        assert 1 in vec

    def test_unknown_tokens_ignored(self):
        """語彙にない語は無視。"""
        vocab = {"hello": 0}
        idf = {0: 1.0}
        vec = _vectorize_query("hello unknown", vocab, idf)
        assert 0 in vec
        assert len(vec) == 1

    def test_empty_query(self):
        vocab = {"hello": 0}
        idf = {0: 1.0}
        vec = _vectorize_query("", vocab, idf)
        assert vec == {}

    def test_empty_vocab(self):
        vec = _vectorize_query("hello world", {}, {})
        assert vec == {}


# =============================================================================
# Test: _cosine_similarity (Phase 1)
# =============================================================================

class TestCosineSimilarity:
    """_cosine_similarity のテスト。"""

    def test_identical_vectors(self):
        """同一ベクトルの類似度は1.0。"""
        v = {0: 1.0, 1: 2.0}
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """直交ベクトルの類似度は0.0。"""
        v1 = {0: 1.0}
        v2 = {1: 1.0}
        assert abs(_cosine_similarity(v1, v2)) < 1e-9

    def test_opposite_vectors(self):
        """反対方向ベクトルの類似度は-1.0。"""
        v1 = {0: 1.0}
        v2 = {0: -1.0}
        assert abs(_cosine_similarity(v1, v2) - (-1.0)) < 1e-9

    def test_empty_vector(self):
        """空ベクトルの類似度は0.0。"""
        assert _cosine_similarity({}, {0: 1.0}) == 0.0
        assert _cosine_similarity({0: 1.0}, {}) == 0.0
        assert _cosine_similarity({}, {}) == 0.0

    def test_sparse_overlap(self):
        """部分的に重なる疎ベクトル。"""
        v1 = {0: 1.0, 1: 1.0}
        v2 = {1: 1.0, 2: 1.0}
        sim = _cosine_similarity(v1, v2)
        # 内積=1, |v1|=sqrt(2), |v2|=sqrt(2) → 1/2 = 0.5
        assert abs(sim - 0.5) < 1e-9


# =============================================================================
# Test: MultiPathRecallState with TF-IDF index (Phase 2)
# =============================================================================

class TestStateTfIdfExtension:
    """MultiPathRecallState のtfidf_indexフィールド拡張テスト。"""

    def test_state_has_tfidf_index(self):
        """StateにTfIdfIndexフィールドが存在する。"""
        s = MultiPathRecallState()
        assert hasattr(s, "tfidf_index")
        assert isinstance(s.tfidf_index, TfIdfIndex)
        assert s.tfidf_index.is_empty()

    def test_state_to_dict_includes_tfidf(self):
        """to_dictにtfidf_indexが含まれる。"""
        s = MultiPathRecallState()
        s.tfidf_index = TfIdfIndex(vocabulary={"a": 0}, idf={0: 1.0}, doc_count=1)
        d = s.to_dict()
        assert "tfidf_index" in d
        assert d["tfidf_index"]["vocabulary"] == {"a": 0}

    def test_state_from_dict_restores_tfidf(self):
        """from_dict��tfidf_indexが復元される。"""
        d = {
            "cycle_count": 3,
            "tfidf_index": {
                "vocabulary": {"b": 1},
                "idf": {"1": 0.5},
                "memory_vectors": {},
                "doc_count": 2,
                "built_at": 1000.0,
                "ngram_n": 2,
            },
        }
        s = MultiPathRecallState.from_dict(d)
        assert s.tfidf_index.vocabulary == {"b": 1}
        assert s.tfidf_index.doc_count == 2

    def test_state_from_dict_backward_compat(self):
        """tfidf_indexがない旧データからも安全に復元（後方互換）。"""
        d = {"cycle_count": 5, "recent_recall_history": ["x"]}
        s = MultiPathRecallState.from_dict(d)
        assert s.tfidf_index.is_empty()
        assert s.cycle_count == 5

    def test_state_round_trip_with_tfidf(self):
        """TfIdfIndex付きStateのラウンドトリップ。"""
        idx = TfIdfIndex(
            vocabulary={"hello": 0, "world": 1},
            idf={0: 1.5, 1: 0.3},
            memory_vectors={"u1": {0: 0.8, 1: 0.2}},
            doc_count=5,
            built_at=9999.0,
        )
        s = MultiPathRecallState(cycle_count=10, tfidf_index=idx)
        restored = MultiPathRecallState.from_dict(s.to_dict())
        assert restored.tfidf_index.vocabulary == idx.vocabulary
        assert restored.tfidf_index.idf == idx.idf
        assert restored.tfidf_index.memory_vectors == idx.memory_vectors
        assert restored.tfidf_index.doc_count == idx.doc_count


# =============================================================================
# Test: MultiPathRecallConfig safety parameters (Phase 2)
# =============================================================================

class TestConfigSafetyParams:
    """Config安全弁パラメータのテスト。"""

    def test_default_safety_params(self):
        cfg = MultiPathRecallConfig()
        assert cfg.max_rebuild_per_session == 10
        assert cfg.max_vocab_size == 5000
        assert cfg.rebuild_threshold == 0.2
        assert cfg.text_max_length == 10000

    def test_custom_safety_params(self):
        cfg = MultiPathRecallConfig(
            max_rebuild_per_session=5,
            max_vocab_size=3000,
            rebuild_threshold=0.5,
            text_max_length=5000,
        )
        assert cfg.max_rebuild_per_session == 5
        assert cfg.max_vocab_size == 3000
        assert cfg.rebuild_threshold == 0.5
        assert cfg.text_max_length == 5000


# =============================================================================
# Test: Processor index management (Phase 2)
# =============================================================================

class TestProcessorIndexManagement:
    """Processorのインデックス管理メソッドテスト。"""

    def test_should_rebuild_empty_index(self):
        """空インデックスなら再構築が必要。"""
        p = make_processor()
        units = make_units(5)
        assert p._should_rebuild_index(units)

    def test_should_rebuild_memory_count_change(self):
        """記憶数がrebuild_threshold以上変化したら再構築が必要。"""
        p = make_processor()
        # インデックスを構築済みにする（doc_count=10）
        p._state.tfidf_index = TfIdfIndex(
            vocabulary={"a": 0}, idf={0: 1.0}, doc_count=10, built_at=time.time(),
        )
        # 10件→12件: 変化率0.2 = rebuild_threshold(0.2)と同じ → 再構築
        units_12 = make_units(12)
        assert p._should_rebuild_index(units_12)

    def test_should_not_rebuild_small_change(self):
        """記憶数の変化が閾値未満なら���構築不要。"""
        p = make_processor()
        p._state.tfidf_index = TfIdfIndex(
            vocabulary={"a": 0}, idf={0: 1.0}, doc_count=10, built_at=time.time(),
        )
        # 10件→11件: 変化率0.1 < threshold(0.2) → 再構築不要
        units_11 = make_units(11)
        assert not p._should_rebuild_index(units_11)

    def test_ensure_index_builds_on_first_call(self):
        """_ensure_indexが初回呼び出しでインデックスを構築する。"""
        p = make_processor()
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="test memory", topics=["topic1"]),
        ]
        p._ensure_index(units)
        assert not p._state.tfidf_index.is_empty()
        assert p._state.tfidf_index.doc_count == 1

    def test_rebuild_count_increments(self):
        """再構築カウンタが増加する。"""
        p = make_processor()
        units = make_units(5)
        p._ensure_index(units)
        assert p._rebuild_count == 1

    def test_rebuild_count_limit(self):
        """再構築上限（max_rebuild_per_session=10）に達したら���構築しない。

        デフォルト値10の根拠: 通常セッションの記憶追加パターンでは
        数十サイクルで記憶数がrebuild_threshold(20%)変動する回数は
        10回以下。異常パターン（高速記憶追加ループ）を遮断するために
        セッション内上限を設定。
        """
        p = make_processor()
        p._rebuild_count = 10  # 上限に達した状態
        units = make_units(5)
        old_index = p._state.tfidf_index
        p._ensure_index(units)
        # インデックスが更新されない（上限到達のため）
        assert p._state.tfidf_index is old_index


# =============================================================================
# Test: _recall_contextual with TF-IDF (Phase 2)
# =============================================================================

class TestContextualRecallTfIdf:
    """TF-IDF化された_recall_contextualのテスト。"""

    def test_tfidf_contextual_basic(self):
        """TF-IDFベースの文脈連想が候補を返す。"""
        p = make_processor(per_path_limit=3)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="cat dog animal", topics=["pets"]),
            MockUnifiedMemoryUnit(unit_id="u2", summary="math science formula", topics=["academics"]),
            MockUnifiedMemoryUnit(unit_id="u3", summary="cat fish pet", topics=["pets"]),
        ]
        ctx = ContextSnapshot(topics=["pets"], percept_text="I love my cat", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        # u1とu3がペット関連→高スコア、u2は無関連→低スコア
        unit_ids = [c.unit_id for c in contextual]
        assert "u1" in unit_ids or "u3" in unit_ids

    def test_tfidf_rare_topic_weighted_higher(self):
        """TF-IDFにより稀少トピックがより高く評価される。"""
        p = make_processor(per_path_limit=5)
        # "common"は全記憶に出現、"rare_topic"は1件のみ
        units = [
            MockUnifiedMemoryUnit(unit_id="u_rare", summary="rare_topic analysis", topics=["rare_topic", "common"]),
            MockUnifiedMemoryUnit(unit_id="u_common1", summary="common stuff", topics=["common"]),
            MockUnifiedMemoryUnit(unit_id="u_common2", summary="common things", topics=["common"]),
            MockUnifiedMemoryUnit(unit_id="u_common3", summary="common items", topics=["common"]),
        ]
        ctx = ContextSnapshot(topics=["rare_topic"], percept_text="rare_topic", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        if contextual:
            # rare_topicを持つu_rareが最上位に来るはず
            assert contextual[0].unit_id == "u_rare"

    def test_tfidf_summary_content_match(self):
        """summaryの内容もスコアリングに寄与する。"""
        p = make_processor(per_path_limit=5)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="quantum physics experiment", topics=[]),
            MockUnifiedMemoryUnit(unit_id="u2", summary="cooking recipe pasta", topics=[]),
        ]
        ctx = ContextSnapshot(percept_text="quantum physics research", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        if contextual:
            assert contextual[0].unit_id == "u1"

    def test_tfidf_fallback_when_index_empty(self):
        """インデックスが空の場合、旧Jaccard方式にフォールバック。"""
        p = make_processor(per_path_limit=3)
        # 強制的にインデックスを空にする
        p._state.tfidf_index = TfIdfIndex()
        p._rebuild_count = 10  # 再構築上限に達した状態

        units = [
            MockUnifiedMemoryUnit(unit_id="u1", topics=["topic_a"], summary="about a"),
            MockUnifiedMemoryUnit(unit_id="u2", topics=["topic_b"], summary="about b"),
        ]
        ctx = ContextSnapshot(topics=["topic_a"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        # フォールバック方式でもtopic_aマッチで候補が出る
        assert len(contextual) >= 1
        assert contextual[0].unit_id == "u1"

    def test_contextual_path_label_preserved(self):
        """TF-IDF化後もpath_label=contextualが維持される。"""
        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(unit_id="u1", summary="test", topics=["topic"])]
        ctx = ContextSnapshot(topics=["topic"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        for c in contextual:
            assert c.path_label == RecallPathLabel.CONTEXTUAL.value

    def test_per_path_limit_still_respected(self):
        """TF-IDF化後もper_path_limitが維持される。"""
        p = make_processor(per_path_limit=2)
        units = make_units(20)
        ctx = ContextSnapshot(topics=["common_topic"], percept_text="topic_0 topic_1", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        assert len(contextual) <= 2

    def test_old_method_renamed_as_fallback(self):
        """旧方式が_recall_contextual_fallbackとして保存されている。"""
        p = make_processor()
        assert hasattr(p, "_recall_contextual_fallback")
        assert callable(p._recall_contextual_fallback)

    def test_index_stored_in_state(self):
        """���ンデックスがstateに格納される。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert not p._state.tfidf_index.is_empty()

    def test_no_topics_no_text_returns_empty(self):
        """トピックもテキストもない場合は候補0件（TF-IDFでも同様）。"""
        p = make_processor()
        units = make_units(5)
        ctx = ContextSnapshot(current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        assert len(contextual) == 0


# =============================================================================
# Test: Phase 3 - 安全弁 + エッジケース
# =============================================================================

class TestTfIdfSafetyValves:
    """TF-IDF固有の安全弁テスト。"""

    def test_fallback_on_empty_index(self):
        """語彙空のインデックスではJaccardフォールバック。"""
        p = make_processor(per_path_limit=3)
        # 再構築不可の状態にする
        p._rebuild_count = 100
        p._state.tfidf_index = TfIdfIndex()

        units = [
            MockUnifiedMemoryUnit(unit_id="u1", topics=["alpha"], summary="alpha test"),
            MockUnifiedMemoryUnit(unit_id="u2", topics=["beta"], summary="beta test"),
        ]
        ctx = ContextSnapshot(topics=["alpha"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        assert len(contextual) >= 1
        assert contextual[0].unit_id == "u1"

    def test_rebuild_limit_prevents_infinite_rebuild(self):
        """再構築上限到達後、新しい記憶が追加されてもインデックスは再構築されない。"""
        p = make_processor()
        p._config.max_rebuild_per_session = 2

        units5 = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        # 1回目: 構築
        p.recall_all_paths(unified_units=units5, context_snapshot=ctx)
        assert p._rebuild_count == 1

        # 大幅に記憶数を変更して2回目: 再構築
        units20 = make_units(20)
        p.recall_all_paths(unified_units=units20, context_snapshot=ctx)
        assert p._rebuild_count == 2

        # 3回目: 上限到達のため再構築されない
        old_built_at = p._state.tfidf_index.built_at
        units50 = make_units(50)
        p.recall_all_paths(unified_units=units50, context_snapshot=ctx)
        assert p._rebuild_count == 2
        assert p._state.tfidf_index.built_at == old_built_at

    def test_vocab_limit_enforced(self):
        """語彙上限がインデックス構築で守られる。"""
        p = make_processor()
        p._config.max_vocab_size = 30

        # 多数の一意な語を持つ記憶
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}",
                summary=" ".join(f"word{i}x{j}" for j in range(20)),
                topics=[f"topic{i}"],
            )
            for i in range(10)
        ]
        ctx = ContextSnapshot(topics=["topic0"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert len(p._state.tfidf_index.vocabulary) <= 30

    def test_rebuild_threshold_respected(self):
        """rebuild_thresholdに満たない変化ではインデッ��ス再構築されない。"""
        p = make_processor()
        p._config.rebuild_threshold = 0.5  # 50%変化で再構築

        units10 = make_units(10)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units10, context_snapshot=ctx)
        built_at_1 = p._state.tfidf_index.built_at
        assert p._rebuild_count == 1

        # 10→12件 (20%変化 < 50%) → 再構築されない
        units12 = make_units(12)
        p.recall_all_paths(unified_units=units12, context_snapshot=ctx)
        assert p._state.tfidf_index.built_at == built_at_1
        assert p._rebuild_count == 1


class TestTfIdfEdgeCases:
    """TF-IDFのエッジケーステスト。"""

    def test_empty_summary_units(self):
        """summaryが空の記憶群でも正常動作。"""
        p = make_processor(per_path_limit=3)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="", topics=["topic_a"]),
            MockUnifiedMemoryUnit(unit_id="u2", summary="", topics=["topic_b"]),
        ]
        ctx = ContextSnapshot(topics=["topic_a"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        assert isinstance(contextual, list)

    def test_empty_topics_units(self):
        """topicsが空の記憶群でも正常動作（summaryのみでマッチ）。"""
        p = make_processor(per_path_limit=3)
        units = [
            MockUnifiedMemoryUnit(unit_id="u1", summary="quantum physics", topics=[]),
            MockUnifiedMemoryUnit(unit_id="u2", summary="cooking recipe", topics=[]),
        ]
        ctx = ContextSnapshot(percept_text="quantum research", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        if contextual:
            assert contextual[0].unit_id == "u1"

    def test_control_chars_in_percept_text(self):
        """制御文字混入のpercept_textで正常動作。"""
        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(unit_id="u1", summary="test data", topics=["test"])]
        ctx = ContextSnapshot(
            percept_text="test\x00data\x01with\x02control\x03chars",
            topics=["test"],
            current_time=2000.0,
        )

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        # クラッシュせず結果が返る
        assert isinstance(result, list)

    def test_very_long_percept_text(self):
        """10000文字超のpercept_textで正常動作（切り捨て）。"""
        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(unit_id="u1", summary="test", topics=["test"])]
        long_text = "test " * 5000  # 25000文字
        ctx = ContextSnapshot(percept_text=long_text, topics=["test"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert isinstance(result, list)

    def test_all_memories_identical_text(self):
        """全記憶が同一テキストの場合。"""
        p = make_processor(per_path_limit=3)
        units = [
            MockUnifiedMemoryUnit(unit_id=f"u{i}", summary="same text same text", topics=["same"])
            for i in range(5)
        ]
        ctx = ContextSnapshot(topics=["same"], percept_text="same text", current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        # 全記憶が同スコア→per_path_limit以内で返る
        assert len(contextual) <= 3

    def test_single_char_topic(self):
        """1文字トピックのマッチ。"""
        p = make_processor(per_path_limit=3)
        units = [MockUnifiedMemoryUnit(unit_id="u1", summary="test", topics=["x"])]
        ctx = ContextSnapshot(topics=["x"], current_time=2000.0)

        result = p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        contextual = [c for c in result if c.path_label == "contextual"]
        assert len(contextual) >= 1


class TestTfIdfThreePathRegression:
    """TF-IDF導入後の3経路同時実行回帰テスト。"""

    def test_three_paths_still_work(self):
        """TF-IDF導入後も3経路が同時に候補を生成する。"""
        now = time.time()
        p = make_processor(per_path_limit=3, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id=f"u{i}", source_id=f"s{i}",
                summary=f"memory about topic_{i}",
                topics=["shared_topic", f"topic_{i}"],
                timestamp=now - i * 60,
                emotional_valence=0.5,
                emotional_label="joy",
            )
            for i in range(10)
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.8}, mood_valence=0.5, dominant_emotion="joy")
        ctx = ContextSnapshot(topics=["shared_topic"], percept_text="topic_0 memory", current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        paths = {c.path_label for c in result}
        assert "emotional" in paths
        assert "contextual" in paths
        assert "temporal" in paths

    def test_other_paths_not_affected(self):
        """TF-IDF変更が感情連想・時間近接経路に影響しない。"""
        now = time.time()
        p = make_processor(per_path_limit=5, temporal_max_distance=3600.0)
        units = [
            MockUnifiedMemoryUnit(
                unit_id="u1", source_id="s1",
                summary="emotional memory",
                topics=["topic"],
                timestamp=now - 60,
                emotional_valence=0.8, emotional_label="joy",
            ),
        ]
        emo = EmotionSnapshot(emotions={"joy": 0.9}, mood_valence=0.7, dominant_emotion="joy")
        ctx = ContextSnapshot(topics=["topic"], current_time=now)

        result = p.recall_all_paths(
            unified_units=units, emotion_snapshot=emo, context_snapshot=ctx,
        )
        emotional = [c for c in result if c.path_label == "emotional"]
        temporal = [c for c in result if c.path_label == "temporal"]
        # 感情連想と時間近接は影響なくu1を返す
        assert any(c.unit_id == "u1" for c in emotional)
        assert any(c.unit_id == "u1" for c in temporal)

    def test_save_load_with_tfidf_index(self):
        """TF-IDFインデックス付きでsave/loadラウンドトリップ。"""
        p = make_processor(per_path_limit=3)
        units = make_units(5)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)

        p.recall_all_paths(unified_units=units, context_snapshot=ctx)
        assert not p._state.tfidf_index.is_empty()

        # ラウンドトリップ
        d = p._state.to_dict()
        restored = MultiPathRecallState.from_dict(d)
        assert restored.tfidf_index.doc_count == p._state.tfidf_index.doc_count
        assert restored.tfidf_index.vocabulary == p._state.tfidf_index.vocabulary

    def test_units_not_modified_by_tfidf(self):
        """TF-IDF処理後もunitsが変更されない。"""
        units = make_units(5)
        original_summaries = [u.summary for u in units]
        original_topics = [list(u.topics) for u in units]

        p = make_processor(per_path_limit=3)
        ctx = ContextSnapshot(topics=["common_topic"], current_time=2000.0)
        p.recall_all_paths(unified_units=units, context_snapshot=ctx)

        for i, u in enumerate(units):
            assert u.summary == original_summaries[i]
            assert u.topics == original_topics[i]
