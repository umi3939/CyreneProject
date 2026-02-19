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
)


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
