"""
tests/test_perceptual_context.py - 知覚入力の内部文脈化テスト

design_perceptual_context.md に基づくテスト:
- 初期状態テスト
- 知覚サマリ蓄積テスト
- 各断面の特徴量記述テスト（4断面全て）
- スライディングウィンドウのFIFO動作テスト
- enrichment形式テスト
- save/loadテスト
- 安全弁テスト（7つ）
- 経路遮断テスト
- 知覚不在ティックの動作テスト
- エッジケーステスト
- topics文字列完全一致テスト
"""

import pytest

from psyche.perceptual_context import (
    ChangeFrequency,
    OverlapDegree,
    TransitionDirection,
    PerceptualSummary,
    PerceptualContextState,
    PerceptualContextConfig,
    PerceptualContextProcessor,
    SECTION_EMOTION_CHANGE_FREQ,
    SECTION_INTENT_CHANGE_FREQ,
    SECTION_TOPIC_OVERLAP,
    SECTION_VALENCE_DIRECTION,
    SECTION_ORDER,
    SECTION_LABELS,
    CHANGE_FREQ_LABELS,
    OVERLAP_LABELS,
    DIRECTION_LABELS,
    get_perceptual_context_summary,
    create_perceptual_context,
    create_perceptual_context_processor,
    _classify_change_frequency,
    _classify_overlap_degree,
    _classify_valence_direction,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def processor():
    """デフォルト設定のプロセッサを返す。"""
    return PerceptualContextProcessor()


@pytest.fixture
def small_window_processor():
    """小さいウィンドウのプロセッサ（FIFO テスト用）。"""
    config = PerceptualContextConfig(max_summaries=5)
    return PerceptualContextProcessor(config=config)


@pytest.fixture
def processor_with_data():
    """データが蓄積されたプロセッサを返す。"""
    proc = PerceptualContextProcessor()
    proc.accumulate_summary("happy", "greeting", ["weather", "sports"], 0.5, tick=1)
    proc.accumulate_summary("happy", "greeting", ["weather", "music"], 0.6, tick=2)
    proc.accumulate_summary("sad", "question", ["politics", "news"], -0.3, tick=3)
    proc.accumulate_summary("neutral", "sharing", ["politics", "tech"], 0.0, tick=4)
    proc.accumulate_summary("excited", "sharing", ["tech", "games"], 0.8, tick=5)
    return proc


# =============================================================================
# Initial State Tests
# =============================================================================

class TestInitialState:
    """初期状態テスト。"""

    def test_empty_summaries(self, processor):
        assert len(processor.state.summaries) == 0

    def test_empty_snapshot(self, processor):
        assert processor.state.snapshot == {}

    def test_empty_previous_snapshot(self, processor):
        assert processor.state.previous_snapshot == {}

    def test_get_snapshot_returns_empty(self, processor):
        assert processor.get_snapshot() == {}

    def test_get_previous_snapshot_returns_empty(self, processor):
        assert processor.get_previous_snapshot() == {}

    def test_get_summary_initial(self, processor):
        summary = processor.get_summary()
        assert summary["summary_count"] == 0
        assert summary["has_snapshot"] is False
        assert summary["has_previous_snapshot"] is False

    def test_enrichment_text_initial(self, processor):
        text = processor.get_enrichment_text()
        assert "待機中" in text

    def test_enrichment_data_initial(self, processor):
        data = processor.get_enrichment_data()
        assert data["summary_count"] == 0
        assert data["snapshot"] == {}

    def test_default_config(self, processor):
        assert processor._config.max_summaries == 50
        assert processor._config.min_records_for_description == 3


# =============================================================================
# Accumulation Tests
# =============================================================================

class TestAccumulation:
    """知覚サマリ蓄積テスト。"""

    def test_single_accumulation(self, processor):
        processor.accumulate_summary("happy", "greeting", ["weather"], 0.5, tick=1)
        assert len(processor.state.summaries) == 1
        s = processor.state.summaries[0]
        assert s.emotion == "happy"
        assert s.intent == "greeting"
        assert s.topics == ["weather"]
        assert s.emotion_valence == 0.5
        assert s.tick == 1

    def test_multiple_accumulation(self, processor):
        for i in range(5):
            processor.accumulate_summary("neutral", "sharing", ["topic"], 0.0, tick=i)
        assert len(processor.state.summaries) == 5

    def test_accumulation_preserves_order(self, processor):
        processor.accumulate_summary("happy", "greeting", [], 0.5, tick=1)
        processor.accumulate_summary("sad", "question", [], -0.3, tick=2)
        processor.accumulate_summary("neutral", "sharing", [], 0.0, tick=3)
        assert processor.state.summaries[0].tick == 1
        assert processor.state.summaries[1].tick == 2
        assert processor.state.summaries[2].tick == 3

    def test_topics_are_copied(self, processor):
        """蓄積時にtopicsリストがコピーされることを確認。"""
        original_topics = ["a", "b"]
        processor.accumulate_summary("neutral", "sharing", original_topics, 0.0, tick=1)
        original_topics.append("c")
        assert processor.state.summaries[0].topics == ["a", "b"]

    def test_no_text_stored(self, processor):
        """自由テキストが保持されないことを確認。"""
        processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=1)
        s = processor.state.summaries[0]
        assert not hasattr(s, "text")
        assert not hasattr(s, "meaning")


# =============================================================================
# Sliding Window FIFO Tests
# =============================================================================

class TestSlidingWindowFIFO:
    """スライディングウィンドウのFIFO動作テスト。"""

    def test_pushout_at_limit(self, small_window_processor):
        proc = small_window_processor
        for i in range(7):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        assert len(proc.state.summaries) == 5

    def test_pushout_removes_oldest(self, small_window_processor):
        proc = small_window_processor
        for i in range(7):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        # 最古のtick=0, tick=1 が押し出されている
        assert proc.state.summaries[0].tick == 2
        assert proc.state.summaries[-1].tick == 6

    def test_pushout_preserves_order(self, small_window_processor):
        proc = small_window_processor
        for i in range(10):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        ticks = [s.tick for s in proc.state.summaries]
        assert ticks == [5, 6, 7, 8, 9]

    def test_exactly_at_limit(self, small_window_processor):
        proc = small_window_processor
        for i in range(5):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        assert len(proc.state.summaries) == 5
        assert proc.state.summaries[0].tick == 0

    def test_one_over_limit(self, small_window_processor):
        proc = small_window_processor
        for i in range(6):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        assert len(proc.state.summaries) == 5
        assert proc.state.summaries[0].tick == 1


# =============================================================================
# Emotion Change Frequency Section Tests
# =============================================================================

class TestEmotionChangeFrequency:
    """感情ラベル変化頻度断面テスト。"""

    def test_no_change(self, processor):
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", [], 0.5, tick=i)
        result = processor.describe_features()
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.RARE.value

    def test_every_change(self, processor):
        emotions = ["happy", "sad", "neutral", "excited", "angry"]
        for i, e in enumerate(emotions):
            processor.accumulate_summary(e, "greeting", [], 0.0, tick=i)
        result = processor.describe_features()
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.FREQUENT.value

    def test_moderate_change(self, processor):
        # 5 entries, 1 change out of 4 pairs = 0.25 ratio -> MODERATE
        processor.accumulate_summary("happy", "greeting", [], 0.5, tick=0)
        processor.accumulate_summary("happy", "greeting", [], 0.5, tick=1)
        processor.accumulate_summary("sad", "greeting", [], -0.3, tick=2)
        processor.accumulate_summary("sad", "greeting", [], -0.3, tick=3)
        processor.accumulate_summary("sad", "greeting", [], -0.3, tick=4)
        result = processor.describe_features()
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.MODERATE.value

    def test_insufficient_data(self, processor):
        processor.accumulate_summary("happy", "greeting", [], 0.5, tick=0)
        result = processor.describe_features()
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.MODERATE.value

    def test_string_exact_match_only(self, processor):
        """文字列完全一致のみの確認。大文字小文字が異なれば変化とカウント。"""
        for i in range(5):
            emotion = "Happy" if i % 2 == 0 else "happy"
            processor.accumulate_summary(emotion, "greeting", [], 0.0, tick=i)
        result = processor.describe_features()
        # 4 changes out of 4 pairs = 1.0 ratio -> FREQUENT
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.FREQUENT.value


# =============================================================================
# Intent Change Frequency Section Tests
# =============================================================================

class TestIntentChangeFrequency:
    """意図ラベル変化頻度断面テスト。"""

    def test_no_change(self, processor):
        for i in range(5):
            processor.accumulate_summary("neutral", "greeting", [], 0.0, tick=i)
        result = processor.describe_features()
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.RARE.value

    def test_every_change(self, processor):
        intents = ["greeting", "question", "sharing", "farewell", "request"]
        for i, intent in enumerate(intents):
            processor.accumulate_summary("neutral", intent, [], 0.0, tick=i)
        result = processor.describe_features()
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.FREQUENT.value

    def test_insufficient_data(self, processor):
        processor.accumulate_summary("neutral", "greeting", [], 0.0, tick=0)
        result = processor.describe_features()
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.MODERATE.value

    def test_string_exact_match_only(self, processor):
        """文字列完全一致のみの確認。"""
        for i in range(5):
            intent = "Question" if i % 2 == 0 else "question"
            processor.accumulate_summary("neutral", intent, [], 0.0, tick=i)
        result = processor.describe_features()
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.FREQUENT.value


# =============================================================================
# Topic Overlap Section Tests
# =============================================================================

class TestTopicOverlap:
    """話題重複度断面テスト。"""

    def test_complete_overlap(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "c"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "c"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.HIGH.value

    def test_no_overlap(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "c"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["d", "e", "f"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_partial_overlap(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "c", "d"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "e", "f"], 0.0, tick=1)
        result = processor.describe_features()
        # 2 overlap out of 4 max = 0.5 -> SOMEWHAT_HIGH
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.SOMEWHAT_HIGH.value

    def test_insufficient_data(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["a"], 0.0, tick=0)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.MODERATE.value

    def test_empty_topics(self, processor):
        processor.accumulate_summary("neutral", "sharing", [], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", [], 0.0, tick=1)
        result = processor.describe_features()
        # 0 max -> MODERATE (default)
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.MODERATE.value

    def test_one_empty_one_with_topics(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["a", "b"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", [], 0.0, tick=1)
        result = processor.describe_features()
        # 0 overlap, max=2 -> 0.0 -> LOW
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_exact_string_match_only(self, processor):
        """文字列完全一致のみ。大文字小文字が異なれば不一致。"""
        processor.accumulate_summary("neutral", "sharing", ["Weather", "Sports"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["weather", "sports"], 0.0, tick=1)
        result = processor.describe_features()
        # 0 overlap out of 2 = 0.0 -> LOW
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_only_latest_and_previous_compared(self, processor):
        """最新と直前の2件だけが比較対象であることを確認。"""
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "c"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["d", "e", "f"], 0.0, tick=1)
        processor.accumulate_summary("neutral", "sharing", ["d", "e", "f"], 0.0, tick=2)
        result = processor.describe_features()
        # tick=1 vs tick=2: complete overlap
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.HIGH.value

    def test_substring_not_matched(self, processor):
        """部分文字列は一致しない。"""
        processor.accumulate_summary("neutral", "sharing", ["weather"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["weath"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_duplicate_topics_in_list(self, processor):
        """リスト内に重複要素がある場合。setに変換されるので重複は除去。"""
        processor.accumulate_summary("neutral", "sharing", ["a", "a", "b"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["a", "b", "b"], 0.0, tick=1)
        result = processor.describe_features()
        # set(["a","a","b"]) = {"a","b"}, set(["a","b","b"]) = {"a","b"}
        # overlap=2, max=2, ratio=1.0 -> HIGH
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.HIGH.value


# =============================================================================
# Valence Direction Section Tests
# =============================================================================

class TestValenceDirection:
    """感情価推移方向断面テスト。"""

    def test_rising(self, processor):
        valences = [-0.5, -0.3, -0.1, 0.1, 0.3]
        for i, v in enumerate(valences):
            processor.accumulate_summary("neutral", "sharing", [], v, tick=i)
        result = processor.describe_features()
        # diff = 0.3 - (-0.5) = 0.8 >= 0.5 -> RISING
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.RISING.value

    def test_falling(self, processor):
        valences = [0.5, 0.3, 0.1, -0.1, -0.3]
        for i, v in enumerate(valences):
            processor.accumulate_summary("neutral", "sharing", [], v, tick=i)
        result = processor.describe_features()
        # diff = -0.3 - 0.5 = -0.8 <= -0.5 -> FALLING
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.FALLING.value

    def test_flat(self, processor):
        valences = [0.0, 0.05, -0.05, 0.02, 0.01]
        for i, v in enumerate(valences):
            processor.accumulate_summary("neutral", "sharing", [], v, tick=i)
        result = processor.describe_features()
        # diff = 0.01 - 0.0 = 0.01 -> FLAT
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.FLAT.value

    def test_somewhat_rising(self, processor):
        valences = [0.0, 0.05, 0.1, 0.15, 0.2]
        for i, v in enumerate(valences):
            processor.accumulate_summary("neutral", "sharing", [], v, tick=i)
        result = processor.describe_features()
        # diff = 0.2 - 0.0 = 0.2 >= 0.15 -> SOMEWHAT_RISING
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.SOMEWHAT_RISING.value

    def test_somewhat_falling(self, processor):
        valences = [0.2, 0.15, 0.1, 0.05, 0.0]
        for i, v in enumerate(valences):
            processor.accumulate_summary("neutral", "sharing", [], v, tick=i)
        result = processor.describe_features()
        # diff = 0.0 - 0.2 = -0.2 -> SOMEWHAT_FALLING
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.SOMEWHAT_FALLING.value

    def test_insufficient_data(self, processor):
        processor.accumulate_summary("neutral", "sharing", [], 0.5, tick=0)
        result = processor.describe_features()
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.FLAT.value


# =============================================================================
# Enrichment Format Tests
# =============================================================================

class TestEnrichmentFormat:
    """enrichment形式テスト。"""

    def test_enrichment_text_with_data(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        text = proc.get_enrichment_text()
        assert "待機中" not in text
        # 4断面すべてが含まれること
        for label in SECTION_LABELS.values():
            assert label in text

    def test_enrichment_text_order(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        text = proc.get_enrichment_text()
        # 定義順に列挙されていること
        labels_in_order = [SECTION_LABELS[s] for s in SECTION_ORDER]
        positions = [text.index(label) for label in labels_in_order]
        assert positions == sorted(positions)

    def test_enrichment_data_structure(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        data = proc.get_enrichment_data()
        assert "summary_count" in data
        assert "snapshot" in data
        assert "summary_text" in data
        assert data["summary_count"] == 5
        assert len(data["snapshot"]) == 4

    def test_enrichment_text_no_emphasis(self, processor_with_data):
        """enrichment内での強調表現が含まれないことを確認。"""
        proc = processor_with_data
        proc.describe_features()
        text = proc.get_enrichment_text()
        forbidden_phrases = [
            "注目すべき", "顕著な", "重要な", "異常な",
            "特に", "注意", "好調", "不調",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in text

    def test_enrichment_waiting_when_empty(self, processor):
        text = processor.get_enrichment_text()
        assert "待機中" in text

    def test_enrichment_all_sections_equal(self, processor_with_data):
        """全断面が等価に列挙されていることを確認。"""
        proc = processor_with_data
        proc.describe_features()
        text = proc.get_enrichment_text()
        # 全ての断面ラベルが含まれる
        for label in SECTION_LABELS.values():
            assert label in text


# =============================================================================
# Snapshot Tests
# =============================================================================

class TestSnapshot:
    """スナップショットテスト。"""

    def test_snapshot_after_describe(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        snapshot = proc.get_snapshot()
        assert len(snapshot) == 4
        assert SECTION_EMOTION_CHANGE_FREQ in snapshot
        assert SECTION_INTENT_CHANGE_FREQ in snapshot
        assert SECTION_TOPIC_OVERLAP in snapshot
        assert SECTION_VALENCE_DIRECTION in snapshot

    def test_snapshot_is_copy(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        snapshot1 = proc.get_snapshot()
        snapshot1["extra_key"] = "should_not_affect"
        snapshot2 = proc.get_snapshot()
        assert "extra_key" not in snapshot2

    def test_previous_snapshot_updated(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        first_snapshot = proc.get_snapshot()

        # もう1件追加して再記述
        proc.accumulate_summary("angry", "request", ["new_topic"], -0.8, tick=6)
        proc.describe_features()

        prev_snapshot = proc.get_previous_snapshot()
        assert prev_snapshot == first_snapshot

    def test_previous_snapshot_is_copy(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        proc.describe_features()  # 2回目で previous_snapshot が設定される
        prev = proc.get_previous_snapshot()
        prev["extra_key"] = "should_not_affect"
        prev2 = proc.get_previous_snapshot()
        assert "extra_key" not in prev2


# =============================================================================
# Save/Load Tests
# =============================================================================

class TestSaveLoad:
    """save/loadテスト。"""

    def test_state_to_dict(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        state_dict = proc.state.to_dict()
        assert "summaries" in state_dict
        assert "snapshot" in state_dict
        assert "previous_snapshot" in state_dict
        assert len(state_dict["summaries"]) == 5

    def test_state_from_dict(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()
        state_dict = proc.state.to_dict()

        restored = PerceptualContextState.from_dict(state_dict)
        assert len(restored.summaries) == 5
        assert restored.snapshot == proc.state.snapshot
        assert restored.previous_snapshot == proc.state.previous_snapshot

    def test_summary_to_dict(self):
        s = PerceptualSummary(
            emotion="happy", intent="greeting",
            topics=["a", "b"], emotion_valence=0.5, tick=10,
        )
        d = s.to_dict()
        assert d["emotion"] == "happy"
        assert d["intent"] == "greeting"
        assert d["topics"] == ["a", "b"]
        assert d["emotion_valence"] == 0.5
        assert d["tick"] == 10

    def test_summary_from_dict(self):
        d = {
            "emotion": "sad", "intent": "question",
            "topics": ["x", "y"], "emotion_valence": -0.3, "tick": 5,
        }
        s = PerceptualSummary.from_dict(d)
        assert s.emotion == "sad"
        assert s.intent == "question"
        assert s.topics == ["x", "y"]
        assert s.emotion_valence == -0.3
        assert s.tick == 5

    def test_round_trip(self, processor_with_data):
        proc = processor_with_data
        proc.describe_features()

        # Save
        state_dict = proc.state.to_dict()

        # Load into new processor
        new_proc = PerceptualContextProcessor()
        new_proc.state = PerceptualContextState.from_dict(state_dict)

        # Verify
        assert len(new_proc.state.summaries) == len(proc.state.summaries)
        assert new_proc.get_snapshot() == proc.get_snapshot()

        # describe_features should work on restored state
        result = new_proc.describe_features()
        assert len(result) == 4

    def test_save_load_empty_state(self):
        state = PerceptualContextState()
        d = state.to_dict()
        restored = PerceptualContextState.from_dict(d)
        assert len(restored.summaries) == 0
        assert restored.snapshot == {}
        assert restored.previous_snapshot == {}

    def test_from_dict_with_missing_keys(self):
        restored = PerceptualContextState.from_dict({})
        assert len(restored.summaries) == 0
        assert restored.snapshot == {}
        assert restored.previous_snapshot == {}

    def test_summary_from_dict_with_missing_keys(self):
        s = PerceptualSummary.from_dict({})
        assert s.emotion == "neutral"
        assert s.intent == "unknown"
        assert s.topics == []
        assert s.emotion_valence == 0.0
        assert s.tick == 0

    def test_processor_save_method(self, processor_with_data):
        """PerceptualContextProcessor.save() が state.to_dict() と同じ結果を返す。"""
        proc = processor_with_data
        proc.describe_features()
        saved = proc.save()
        assert saved == proc.state.to_dict()
        assert "summaries" in saved
        assert "snapshot" in saved
        assert "previous_snapshot" in saved

    def test_processor_load_method(self, processor_with_data):
        """PerceptualContextProcessor.load() でデータが正しく復元される。"""
        proc = processor_with_data
        proc.describe_features()
        saved = proc.save()

        new_proc = PerceptualContextProcessor()
        new_proc.load(saved)

        assert len(new_proc.state.summaries) == len(proc.state.summaries)
        assert new_proc.get_snapshot() == proc.get_snapshot()
        assert new_proc.get_previous_snapshot() == proc.get_previous_snapshot()

    def test_save_load_enrichment_text_match(self, processor_with_data):
        """save/load後のenrichmentテキストが一致する。"""
        proc = processor_with_data
        proc.describe_features()
        original_text = proc.get_enrichment_text()

        saved = proc.save()
        new_proc = PerceptualContextProcessor()
        new_proc.load(saved)

        restored_text = new_proc.get_enrichment_text()
        assert restored_text == original_text

    def test_save_load_continue_processing(self, processor_with_data):
        """save/load後に処理を継続できる。"""
        proc = processor_with_data
        proc.describe_features()
        saved = proc.save()

        new_proc = PerceptualContextProcessor()
        new_proc.load(saved)

        # 復元後に新しいデータを蓄積して処理できる
        new_proc.accumulate_summary("angry", "request", ["new_topic"], -0.7, tick=6)
        new_proc.accumulate_summary("neutral", "sharing", ["another"], 0.1, tick=7)
        result = new_proc.describe_features()
        assert len(result) == 4

        # previous_snapshot が更新されている
        prev = new_proc.get_previous_snapshot()
        assert len(prev) == 4

        # enrichment も動作する
        text = new_proc.get_enrichment_text()
        assert "待機中" not in text

        # 蓄積データが増えている
        assert len(new_proc.state.summaries) == 7


# =============================================================================
# Topics None Guard Tests
# =============================================================================

class TestTopicsNoneGuard:
    """topics=None 時の accumulate_summary の挙動テスト。"""

    def test_topics_none_does_not_raise(self, processor):
        """topics=None でも TypeError にならない。"""
        processor.accumulate_summary("happy", "greeting", None, 0.5, tick=1)
        assert len(processor.state.summaries) == 1
        assert processor.state.summaries[0].topics == []

    def test_topics_none_describe_features(self, processor):
        """topics=None で蓄積した後も describe_features が正常動作する。"""
        processor.accumulate_summary("happy", "greeting", None, 0.5, tick=1)
        processor.accumulate_summary("sad", "question", None, -0.3, tick=2)
        processor.accumulate_summary("neutral", "sharing", ["a"], 0.0, tick=3)
        result = processor.describe_features()
        assert len(result) == 4

    def test_topics_none_and_normal_mixed(self, processor):
        """topics=None と通常の topics を混在させても正常動作する。"""
        processor.accumulate_summary("happy", "greeting", None, 0.5, tick=1)
        processor.accumulate_summary("happy", "greeting", ["weather"], 0.5, tick=2)
        result = processor.describe_features()
        # None -> [] なので、[] vs ["weather"] の比較
        # overlap=0, max=1, ratio=0.0 -> LOW
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value


# =============================================================================
# Safety Valve Tests (7 valves)
# =============================================================================

class TestSafetyValves:
    """安全弁テスト（7つ）。"""

    def test_valve1_no_text_comparison(self, processor):
        """安全弁1: テキスト比較禁止 -- PerceptualSummaryにtext/meaningフィールドがない。"""
        s = PerceptualSummary()
        fields = s.__dataclass_fields__
        assert "text" not in fields
        assert "meaning" not in fields

    def test_valve2_topics_exact_match_only(self, processor):
        """安全弁2: topics意味判定禁止 -- 文字列完全一致のみ。"""
        processor.accumulate_summary("neutral", "sharing", ["weather forecast"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["weather"], 0.0, tick=1)
        result = processor.describe_features()
        # "weather forecast" != "weather" -> 不一致
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_valve3_section_equivalence(self, processor_with_data):
        """安全弁3: 断面等価性 -- 全断面の特徴量に重みや重要度がない。"""
        proc = processor_with_data
        result = proc.describe_features()
        # 全4断面が含まれている
        assert len(result) == 4
        # 各断面が独立した値を持つ（辞書の値として）
        for section_name in SECTION_ORDER:
            assert section_name in result

    def test_valve4_no_cross_section_integration(self, processor_with_data):
        """安全弁4: 断面間統合禁止 -- 4断面を統合した単一指標がない。"""
        proc = processor_with_data
        result = proc.describe_features()
        # 4つの独立した断面のみ。統合的な「総合知覚推移指標」は存在しない
        assert len(result) == 4
        snapshot = proc.get_snapshot()
        # 統合キーが存在しないことを確認
        for key in snapshot:
            assert key in SECTION_ORDER

    def test_valve5_no_pattern_extraction(self, processor):
        """安全弁5: パターン抽出禁止 -- スナップショットの時系列パターン抽出がない。"""
        for i in range(10):
            processor.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
            processor.describe_features()
        # describe_featuresを複数回呼んでも、スナップショットの推移パターンは抽出されない
        summary = processor.get_summary()
        # パターン関連のキーが存在しないこと
        assert "pattern" not in summary
        assert "trend" not in summary
        assert "history" not in summary

    def test_valve6_no_type_definition(self, processor_with_data):
        """安全弁6: 推移の「型」定義禁止 -- 段階値の並列記述のみ。"""
        proc = processor_with_data
        result = proc.describe_features()
        # 各断面の値はEnum値の文字列であり、型分類ではない
        valid_freq_values = {f.value for f in ChangeFrequency}
        valid_overlap_values = {o.value for o in OverlapDegree}
        valid_dir_values = {d.value for d in TransitionDirection}

        assert result[SECTION_EMOTION_CHANGE_FREQ] in valid_freq_values
        assert result[SECTION_INTENT_CHANGE_FREQ] in valid_freq_values
        assert result[SECTION_TOPIC_OVERLAP] in valid_overlap_values
        assert result[SECTION_VALENCE_DIRECTION] in valid_dir_values

    def test_valve7_enrichment_no_emphasis(self, processor_with_data):
        """安全弁7: enrichment内での強調禁止。"""
        proc = processor_with_data
        proc.describe_features()
        text = proc.get_enrichment_text()
        # 強調表現が含まれない
        forbidden = [
            "注目", "顕著", "重要", "異常", "特に",
            "注意が必要", "好調", "不調", "警告",
        ]
        for word in forbidden:
            assert word not in text


# =============================================================================
# Pathway Block Tests (4 pathways)
# =============================================================================

class TestPathwayBlocks:
    """経路遮断テスト（4つ）。"""

    def test_block1_no_emotion_pipeline_output(self, processor_with_data):
        """経路遮断1: 推移特徴量→感情パイプライン。
        プロセッサに感情パイプラインへの書き込みメソッドがない。
        """
        proc = processor_with_data
        proc.describe_features()
        # 出力はsnapshot/enrichmentのみ。感情に影響するメソッドがないことを確認
        public_methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        emotion_modifying_methods = [
            m for m in public_methods
            if any(keyword in m.lower() for keyword in [
                "set_emotion", "modify_emotion", "update_emotion",
                "change_emotion", "adjust_emotion",
            ])
        ]
        assert emotion_modifying_methods == []

    def test_block2_no_perception_feedback(self, processor_with_data):
        """経路遮断2: 推移特徴量→知覚解析。
        プロセッサに知覚解析への書き込みメソッドがない。
        """
        proc = processor_with_data
        proc.describe_features()
        public_methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        perception_modifying_methods = [
            m for m in public_methods
            if any(keyword in m.lower() for keyword in [
                "set_percept", "modify_percept", "update_percept",
                "bias_percept", "adjust_percept",
            ])
        ]
        assert perception_modifying_methods == []

    def test_block3_no_recall_internal_output(self, processor_with_data):
        """経路遮断3: 推移特徴量→multi_path_recall内部処理。
        プロセッサに想起経路への書き込みメソッドがない。
        """
        proc = processor_with_data
        proc.describe_features()
        public_methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        recall_modifying_methods = [
            m for m in public_methods
            if any(keyword in m.lower() for keyword in [
                "set_recall", "modify_recall", "update_recall",
                "score_recall", "filter_recall",
            ])
        ]
        assert recall_modifying_methods == []

    def test_block4_no_context_sensitivity_output(self, processor_with_data):
        """経路遮断4: 推移特徴量→context_sensitivity連続性パラメータ。
        プロセッサにcontext_sensitivityへの書き込みメソッドがない。
        """
        proc = processor_with_data
        proc.describe_features()
        public_methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        context_modifying_methods = [
            m for m in public_methods
            if any(keyword in m.lower() for keyword in [
                "set_context", "modify_context", "update_context",
                "set_sensitivity", "modify_sensitivity",
            ])
        ]
        assert context_modifying_methods == []

    def test_output_is_readonly(self, processor_with_data):
        """出力がREAD-ONLYであることを確認（辞書のコピーが返される）。"""
        proc = processor_with_data
        proc.describe_features()
        snapshot = proc.get_snapshot()
        snapshot["hacked"] = "true"
        assert "hacked" not in proc.get_snapshot()


# =============================================================================
# Absent Perception Tick Tests
# =============================================================================

class TestAbsentPerceptionTick:
    """知覚不在ティックの動作テスト。"""

    def test_describe_without_new_accumulation(self, processor_with_data):
        """蓄積なしでdescribe_featuresを呼んでもエラーにならない。"""
        proc = processor_with_data
        result1 = proc.describe_features()
        # 新たな蓄積なしで再度呼び出し
        result2 = proc.describe_features()
        # ウィンドウ内の既存データに基づく再記述
        assert result2 == result1

    def test_describe_on_empty_state(self, processor):
        """空の状態でdescribe_featuresを呼んでもエラーにならない。"""
        result = processor.describe_features()
        assert len(result) == 4
        # 全てデフォルト値
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.MODERATE.value
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.MODERATE.value
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.MODERATE.value
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.FLAT.value

    def test_previous_snapshot_preserved_on_absent_tick(self, processor_with_data):
        """知覚不在ティックでもprevious_snapshotが正しく設定される。"""
        proc = processor_with_data
        proc.describe_features()
        first = proc.get_snapshot()
        proc.describe_features()  # 蓄積なし
        prev = proc.get_previous_snapshot()
        assert prev == first


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_single_entry(self, processor):
        processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=0)
        result = processor.describe_features()
        # 全てデフォルト値
        assert len(result) == 4

    def test_two_entries(self, processor):
        processor.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=0)
        processor.accumulate_summary("sad", "question", ["b"], -0.5, tick=1)
        result = processor.describe_features()
        assert len(result) == 4
        # 2件ではmin_records_for_description=3未満なので頻度はMODERATE
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.MODERATE.value

    def test_exactly_min_records(self, processor):
        processor.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=0)
        processor.accumulate_summary("sad", "question", ["b"], -0.3, tick=1)
        processor.accumulate_summary("neutral", "sharing", ["c"], 0.0, tick=2)
        result = processor.describe_features()
        # 3件でmin_records_for_description=3なので計算が行われる
        assert len(result) == 4

    def test_very_large_window(self):
        config = PerceptualContextConfig(max_summaries=1000)
        proc = PerceptualContextProcessor(config=config)
        for i in range(500):
            emotion = "happy" if i % 3 == 0 else "sad"
            proc.accumulate_summary(emotion, "sharing", [], 0.0, tick=i)
        assert len(proc.state.summaries) == 500
        result = proc.describe_features()
        assert len(result) == 4

    def test_extreme_valence_values(self, processor):
        processor.accumulate_summary("neutral", "sharing", [], -1.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", [], -1.0, tick=1)
        processor.accumulate_summary("neutral", "sharing", [], 1.0, tick=2)
        result = processor.describe_features()
        # diff = 1.0 - (-1.0) = 2.0 >= 0.5 -> RISING
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.RISING.value

    def test_identical_entries(self, processor):
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=i)
        result = processor.describe_features()
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.RARE.value
        assert result[SECTION_INTENT_CHANGE_FREQ] == ChangeFrequency.RARE.value
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.HIGH.value
        assert result[SECTION_VALENCE_DIRECTION] == TransitionDirection.FLAT.value

    def test_empty_emotion_label(self, processor):
        for i in range(5):
            processor.accumulate_summary("", "sharing", [], 0.0, tick=i)
        result = processor.describe_features()
        # 全て同じ空文字列なので変化なし
        assert result[SECTION_EMOTION_CHANGE_FREQ] == ChangeFrequency.RARE.value

    def test_unicode_topics(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["天気", "スポーツ"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["天気", "音楽"], 0.0, tick=1)
        result = processor.describe_features()
        # "天気" is shared, "スポーツ" vs "音楽" differ
        # overlap=1, max=2, ratio=0.5 -> SOMEWHAT_HIGH
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.SOMEWHAT_HIGH.value


# =============================================================================
# Topics Exact String Match Tests
# =============================================================================

class TestTopicsExactStringMatch:
    """topics文字列完全一致テスト。"""

    def test_case_sensitivity(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["Python"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["python"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_whitespace_difference(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["hello world"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["hello  world"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_trailing_space_difference(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["topic "], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["topic"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value

    def test_exact_match_succeeds(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["exact_match"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["exact_match"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.HIGH.value

    def test_similar_but_not_equal(self, processor):
        processor.accumulate_summary("neutral", "sharing", ["machine learning"], 0.0, tick=0)
        processor.accumulate_summary("neutral", "sharing", ["Machine Learning"], 0.0, tick=1)
        result = processor.describe_features()
        assert result[SECTION_TOPIC_OVERLAP] == OverlapDegree.LOW.value


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """ヘルパー関数テスト。"""

    def test_classify_change_frequency_thresholds(self):
        # ratio >= 0.8 -> FREQUENT
        assert _classify_change_frequency(8, 11, 3) == ChangeFrequency.FREQUENT
        # ratio >= 0.5 -> SOMEWHAT_FREQUENT
        assert _classify_change_frequency(5, 11, 3) == ChangeFrequency.SOMEWHAT_FREQUENT
        # ratio >= 0.2 -> MODERATE
        assert _classify_change_frequency(2, 11, 3) == ChangeFrequency.MODERATE
        # ratio >= 0.1 -> SOMEWHAT_RARE
        assert _classify_change_frequency(1, 11, 3) == ChangeFrequency.SOMEWHAT_RARE
        # ratio < 0.1 -> RARE
        assert _classify_change_frequency(0, 11, 3) == ChangeFrequency.RARE

    def test_classify_change_frequency_below_min(self):
        assert _classify_change_frequency(5, 2, 3) == ChangeFrequency.MODERATE

    def test_classify_overlap_degree_thresholds(self):
        # ratio >= 0.8 -> HIGH
        assert _classify_overlap_degree(4, 5) == OverlapDegree.HIGH
        # ratio >= 0.5 -> SOMEWHAT_HIGH
        assert _classify_overlap_degree(3, 5) == OverlapDegree.SOMEWHAT_HIGH
        # ratio >= 0.2 -> MODERATE
        assert _classify_overlap_degree(1, 5) == OverlapDegree.MODERATE
        # ratio >= 0.1 -> SOMEWHAT_LOW
        assert _classify_overlap_degree(1, 10) == OverlapDegree.SOMEWHAT_LOW
        # ratio < 0.1 -> LOW
        assert _classify_overlap_degree(0, 10) == OverlapDegree.LOW

    def test_classify_overlap_degree_zero_max(self):
        assert _classify_overlap_degree(0, 0) == OverlapDegree.MODERATE

    def test_classify_valence_direction_thresholds(self):
        # diff >= 0.5 -> RISING
        assert _classify_valence_direction([0.0, 0.0, 0.5], 3) == TransitionDirection.RISING
        # diff >= 0.15 -> SOMEWHAT_RISING
        assert _classify_valence_direction([0.0, 0.0, 0.2], 3) == TransitionDirection.SOMEWHAT_RISING
        # diff in [-0.15, 0.15) -> FLAT
        assert _classify_valence_direction([0.0, 0.0, 0.1], 3) == TransitionDirection.FLAT
        # diff >= -0.5 -> SOMEWHAT_FALLING
        assert _classify_valence_direction([0.0, 0.0, -0.2], 3) == TransitionDirection.SOMEWHAT_FALLING
        # diff < -0.5 -> FALLING
        assert _classify_valence_direction([0.0, 0.0, -0.6], 3) == TransitionDirection.FALLING

    def test_classify_valence_direction_below_min(self):
        assert _classify_valence_direction([0.5, 1.0], 3) == TransitionDirection.FLAT


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_perceptual_context(self):
        proc = create_perceptual_context()
        assert isinstance(proc, PerceptualContextProcessor)
        assert len(proc.state.summaries) == 0

    def test_create_perceptual_context_with_config(self):
        config = PerceptualContextConfig(max_summaries=10)
        proc = create_perceptual_context(config=config)
        assert proc._config.max_summaries == 10

    def test_create_perceptual_context_processor(self):
        proc = create_perceptual_context_processor()
        assert isinstance(proc, PerceptualContextProcessor)

    def test_create_perceptual_context_processor_with_config(self):
        config = PerceptualContextConfig(max_summaries=20, min_records_for_description=5)
        proc = create_perceptual_context_processor(config=config)
        assert proc._config.max_summaries == 20
        assert proc._config.min_records_for_description == 5


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """列挙型テスト。"""

    def test_change_frequency_values(self):
        assert ChangeFrequency.FREQUENT.value == "frequent"
        assert ChangeFrequency.SOMEWHAT_FREQUENT.value == "somewhat_frequent"
        assert ChangeFrequency.MODERATE.value == "moderate"
        assert ChangeFrequency.SOMEWHAT_RARE.value == "somewhat_rare"
        assert ChangeFrequency.RARE.value == "rare"

    def test_overlap_degree_values(self):
        assert OverlapDegree.HIGH.value == "high"
        assert OverlapDegree.SOMEWHAT_HIGH.value == "somewhat_high"
        assert OverlapDegree.MODERATE.value == "moderate"
        assert OverlapDegree.SOMEWHAT_LOW.value == "somewhat_low"
        assert OverlapDegree.LOW.value == "low"

    def test_transition_direction_values(self):
        assert TransitionDirection.RISING.value == "rising"
        assert TransitionDirection.SOMEWHAT_RISING.value == "somewhat_rising"
        assert TransitionDirection.FLAT.value == "flat"
        assert TransitionDirection.SOMEWHAT_FALLING.value == "somewhat_falling"
        assert TransitionDirection.FALLING.value == "falling"

    def test_all_change_frequency_have_labels(self):
        for freq in ChangeFrequency:
            assert freq in CHANGE_FREQ_LABELS

    def test_all_overlap_degree_have_labels(self):
        for degree in OverlapDegree:
            assert degree in OVERLAP_LABELS

    def test_all_transition_direction_have_labels(self):
        for direction in TransitionDirection:
            assert direction in DIRECTION_LABELS


# =============================================================================
# Summary Function Tests
# =============================================================================

class TestSummaryFunction:
    """get_perceptual_context_summary テスト。"""

    def test_empty_state(self):
        state = PerceptualContextState()
        result = get_perceptual_context_summary(state)
        assert "待機中" in result

    def test_with_snapshot(self):
        state = PerceptualContextState(
            snapshot={
                SECTION_EMOTION_CHANGE_FREQ: ChangeFrequency.FREQUENT.value,
                SECTION_INTENT_CHANGE_FREQ: ChangeFrequency.MODERATE.value,
                SECTION_TOPIC_OVERLAP: OverlapDegree.HIGH.value,
                SECTION_VALENCE_DIRECTION: TransitionDirection.RISING.value,
            }
        )
        result = get_perceptual_context_summary(state)
        assert "感情ラベル変化頻度=頻繁" in result
        assert "意図ラベル変化頻度=普通" in result
        assert "話題重複度=高" in result
        assert "感情価推移方向=上昇" in result

    def test_partial_snapshot(self):
        state = PerceptualContextState(
            snapshot={
                SECTION_EMOTION_CHANGE_FREQ: ChangeFrequency.RARE.value,
            }
        )
        result = get_perceptual_context_summary(state)
        assert "感情ラベル変化頻度=少ない" in result

    def test_empty_snapshot_dict(self):
        state = PerceptualContextState(snapshot={})
        result = get_perceptual_context_summary(state)
        assert "待機中" in result


# =============================================================================
# State Setter Tests
# =============================================================================

class TestStateSetter:
    """state setter テスト。"""

    def test_set_state(self, processor):
        new_state = PerceptualContextState(
            summaries=[
                PerceptualSummary(emotion="happy", intent="greeting",
                                  topics=["a"], emotion_valence=0.5, tick=1)
            ]
        )
        processor.state = new_state
        assert len(processor.state.summaries) == 1
        assert processor.state.summaries[0].emotion == "happy"


# =============================================================================
# Integration-like Tests
# =============================================================================

class TestIntegration:
    """統合的なテスト。"""

    def test_full_pipeline(self):
        """3段パイプライン全体の動作確認。"""
        proc = create_perceptual_context()

        # Stage 1: 蓄積
        proc.accumulate_summary("happy", "greeting", ["weather"], 0.5, tick=1)
        proc.accumulate_summary("happy", "question", ["weather", "sports"], 0.6, tick=2)
        proc.accumulate_summary("sad", "question", ["politics"], -0.3, tick=3)
        proc.accumulate_summary("sad", "sharing", ["politics", "tech"], -0.1, tick=4)
        proc.accumulate_summary("neutral", "sharing", ["tech"], 0.0, tick=5)

        # Stage 2: 特徴量記述
        result = proc.describe_features()
        assert len(result) == 4

        # Stage 3: 参照情報
        text = proc.get_enrichment_text()
        assert "待機中" not in text
        snapshot = proc.get_snapshot()
        assert len(snapshot) == 4

        data = proc.get_enrichment_data()
        assert data["summary_count"] == 5

    def test_multiple_describe_cycles(self):
        """複数回のdescribeサイクルでprevious_snapshotが正しく更新される。"""
        proc = create_perceptual_context()

        proc.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=1)
        proc.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=2)
        proc.accumulate_summary("happy", "greeting", ["a"], 0.5, tick=3)
        proc.describe_features()
        snap1 = proc.get_snapshot()

        proc.accumulate_summary("sad", "question", ["b"], -0.5, tick=4)
        proc.accumulate_summary("angry", "request", ["c"], -0.8, tick=5)
        proc.accumulate_summary("excited", "sharing", ["d"], 0.9, tick=6)
        proc.describe_features()
        snap2 = proc.get_snapshot()
        prev2 = proc.get_previous_snapshot()
        assert prev2 == snap1
        # snap2 should be different from snap1 since we changed the data
        # (at least some sections should differ)
        assert snap2 != snap1

    def test_window_pushout_during_pipeline(self):
        """パイプライン中にウィンドウ押し出しが発生しても正常動作。"""
        config = PerceptualContextConfig(max_summaries=3)
        proc = PerceptualContextProcessor(config=config)

        for i in range(5):
            proc.accumulate_summary("neutral", "sharing", [], 0.0, tick=i)
        assert len(proc.state.summaries) == 3

        result = proc.describe_features()
        assert len(result) == 4
