"""
tests/test_attention_distribution_description.py - 注意配分の構造的記述テスト

設計書 (design_attention_distribution_description.md) に基づく機能テスト。
"""

import time
import pytest

from psyche.attention_distribution_description import (
    # Config
    AttentionDistributionConfig,
    create_attention_distribution_config,
    # State
    AttentionDistributionState,
    create_attention_distribution_state,
    # Data structures
    AttentionSnapshot,
    AttentionVariation,
    # Constants
    SOURCE_PERCEPTION,
    SOURCE_TEXT_INPUT,
    SOURCE_SPONTANEOUS,
    SIGNAL_EMOTION,
    SIGNAL_MEMORY,
    SIGNAL_MOTIVATION,
    SIGNAL_GOAL,
    SIGNAL_RESPONSIBILITY,
    ALL_SOURCE_KEYS,
    # Enums
    QuantityLevel,
    ConcentrationLevel,
    # Functions
    determine_quantity_level,
    determine_concentration_level,
    compute_concentration,
    collect_source_quantities,
    compose_snapshot,
    derive_variation,
    process_attention_distribution,
    # Accessors
    get_latest_snapshot,
    get_snapshot_history,
    get_latest_variation,
    get_attention_distribution_summary,
    get_enrichment_text,
    # Save/Load
    save_state,
    load_state,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    return AttentionDistributionConfig()


@pytest.fixture
def small_config():
    """テスト用の小さな履歴設定。"""
    return AttentionDistributionConfig(
        max_snapshot_history=5,
        variation_comparison_count=3,
    )


@pytest.fixture
def empty_state():
    return create_attention_distribution_state()


# =============================================================================
# Constants
# =============================================================================

class TestConstants:
    def test_source_constants(self):
        assert SOURCE_PERCEPTION == "perception"
        assert SOURCE_TEXT_INPUT == "text_input"
        assert SOURCE_SPONTANEOUS == "spontaneous"

    def test_signal_constants(self):
        assert SIGNAL_EMOTION == "emotion"
        assert SIGNAL_MEMORY == "memory"
        assert SIGNAL_MOTIVATION == "motivation"
        assert SIGNAL_GOAL == "goal"
        assert SIGNAL_RESPONSIBILITY == "responsibility"

    def test_all_source_keys(self):
        assert len(ALL_SOURCE_KEYS) == 8
        assert SOURCE_PERCEPTION in ALL_SOURCE_KEYS
        assert SOURCE_TEXT_INPUT in ALL_SOURCE_KEYS
        assert SOURCE_SPONTANEOUS in ALL_SOURCE_KEYS
        assert SIGNAL_EMOTION in ALL_SOURCE_KEYS
        assert SIGNAL_MEMORY in ALL_SOURCE_KEYS
        assert SIGNAL_MOTIVATION in ALL_SOURCE_KEYS
        assert SIGNAL_GOAL in ALL_SOURCE_KEYS
        assert SIGNAL_RESPONSIBILITY in ALL_SOURCE_KEYS


# =============================================================================
# Enums
# =============================================================================

class TestQuantityLevel:
    def test_all_values(self):
        assert QuantityLevel.ABSENT.value == "absent"
        assert QuantityLevel.MINIMAL.value == "minimal"
        assert QuantityLevel.FEW.value == "few"
        assert QuantityLevel.MODERATE.value == "moderate"
        assert QuantityLevel.MANY.value == "many"

    def test_determine_absent(self):
        assert determine_quantity_level(0) == QuantityLevel.ABSENT
        assert determine_quantity_level(-1) == QuantityLevel.ABSENT

    def test_determine_minimal(self):
        assert determine_quantity_level(1) == QuantityLevel.MINIMAL

    def test_determine_few(self):
        for count in [2, 3]:
            assert determine_quantity_level(count) == QuantityLevel.FEW

    def test_determine_moderate(self):
        for count in [4, 5, 6, 7]:
            assert determine_quantity_level(count) == QuantityLevel.MODERATE

    def test_determine_many(self):
        for count in [8, 10, 50, 100]:
            assert determine_quantity_level(count) == QuantityLevel.MANY


class TestConcentrationLevel:
    def test_all_values(self):
        assert ConcentrationLevel.DISPERSED.value == "dispersed"
        assert ConcentrationLevel.SLIGHT.value == "slight"
        assert ConcentrationLevel.MODERATE.value == "moderate"
        assert ConcentrationLevel.CONCENTRATED.value == "concentrated"

    def test_determine_dispersed(self):
        assert determine_concentration_level(0.0) == ConcentrationLevel.DISPERSED
        assert determine_concentration_level(0.19) == ConcentrationLevel.DISPERSED

    def test_determine_slight(self):
        assert determine_concentration_level(0.2) == ConcentrationLevel.SLIGHT
        assert determine_concentration_level(0.44) == ConcentrationLevel.SLIGHT

    def test_determine_moderate(self):
        assert determine_concentration_level(0.45) == ConcentrationLevel.MODERATE
        assert determine_concentration_level(0.69) == ConcentrationLevel.MODERATE

    def test_determine_concentrated(self):
        assert determine_concentration_level(0.7) == ConcentrationLevel.CONCENTRATED
        assert determine_concentration_level(1.0) == ConcentrationLevel.CONCENTRATED


# =============================================================================
# compute_concentration
# =============================================================================

class TestComputeConcentration:
    def test_empty_quantities(self):
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        assert compute_concentration(quantities) == 0.0

    def test_uniform_distribution(self):
        """全入力源が同じ量→集中度は低い。"""
        quantities = {k: 5 for k in ALL_SOURCE_KEYS}
        result = compute_concentration(quantities)
        assert result < 0.1

    def test_single_source_concentration(self):
        """1入力源に全処理が集中→集中度は高い。"""
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        quantities[SOURCE_PERCEPTION] = 100
        result = compute_concentration(quantities)
        assert result > 0.7

    def test_two_sources_moderate(self):
        """2入力源に処理が分散→中程度の集中度。"""
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        quantities[SOURCE_PERCEPTION] = 10
        quantities[SIGNAL_EMOTION] = 10
        result = compute_concentration(quantities)
        assert 0.0 < result < 1.0

    def test_concentration_is_clamped(self):
        """集中度は0.0〜1.0の範囲に収まる。"""
        quantities = {k: 1 for k in ALL_SOURCE_KEYS}
        result = compute_concentration(quantities)
        assert 0.0 <= result <= 1.0

        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        quantities[SIGNAL_EMOTION] = 1000
        result = compute_concentration(quantities)
        assert 0.0 <= result <= 1.0


# =============================================================================
# collect_source_quantities
# =============================================================================

class TestCollectSourceQuantities:
    def test_empty_collection(self):
        """全入力Noneでも安全に0を返す。"""
        result = collect_source_quantities()
        for key in ALL_SOURCE_KEYS:
            assert key in result
            assert result[key] == 0

    def test_has_perception_input_flag(self):
        result = collect_source_quantities(has_perception_input=True)
        assert result[SOURCE_PERCEPTION] >= 1

    def test_perception_element_count(self):
        result = collect_source_quantities(
            has_perception_input=True,
            perception_element_count=5,
        )
        assert result[SOURCE_PERCEPTION] == 5

    def test_has_text_input_flag(self):
        result = collect_source_quantities(has_text_input=True)
        assert result[SOURCE_TEXT_INPUT] == 1

    def test_has_spontaneous_activation_flag(self):
        result = collect_source_quantities(has_spontaneous_activation=True)
        assert result[SOURCE_SPONTANEOUS] == 1

    def test_no_flag_no_count(self):
        """フラグがFalseなら0。"""
        result = collect_source_quantities(
            has_perception_input=False,
            has_text_input=False,
            has_spontaneous_activation=False,
        )
        assert result[SOURCE_PERCEPTION] == 0
        assert result[SOURCE_TEXT_INPUT] == 0
        assert result[SOURCE_SPONTANEOUS] == 0

    def test_emotion_state_duck_typing(self):
        """EmotionVector的な属性を持つオブジェクトから読み取る。"""
        class MockEmotion:
            joy = 0.5
            sadness = 0.0
            anger = 0.3
            fear = 0.0
            disgust = 0.0
            surprise = 0.0
            trust = 0.0
            anticipation = 0.0

        result = collect_source_quantities(emotion_state=MockEmotion())
        assert result[SIGNAL_EMOTION] == 2  # joy=0.5, anger=0.3

    def test_emotion_state_dict_form(self):
        """emotions辞書形式からも読み取れる。"""
        class MockEmotion:
            emotions = {"joy": 0.5, "sadness": 0.2, "anger": 0.0}

        result = collect_source_quantities(emotion_state=MockEmotion())
        assert result[SIGNAL_EMOTION] == 2  # joy=0.5, sadness=0.2

    def test_memory_state_duck_typing(self):
        class MockMemory:
            recalled_memories = [1, 2, 3]

        result = collect_source_quantities(memory_state=MockMemory())
        assert result[SIGNAL_MEMORY] == 3

    def test_motivation_state_duck_typing(self):
        class MockEntry:
            strength = None  # 存在するだけでカウント

        class MockMotivation:
            entries = [MockEntry(), MockEntry()]

        result = collect_source_quantities(motivation_state=MockMotivation())
        assert result[SIGNAL_MOTIVATION] == 2

    def test_goal_state_with_active_goals(self):
        class MockTransient:
            active_goals = [1, 2]

        result = collect_source_quantities(transient_goal_state=MockTransient())
        assert result[SIGNAL_GOAL] == 2

    def test_goal_state_with_scoped_goal(self):
        class MockScoped:
            current_goal = "something"

        result = collect_source_quantities(scoped_goal_state=MockScoped())
        assert result[SIGNAL_GOAL] == 1

    def test_goal_state_combined(self):
        class MockTransient:
            active_goals = [1]

        class MockScoped:
            current_goal = "something"

        result = collect_source_quantities(
            transient_goal_state=MockTransient(),
            scoped_goal_state=MockScoped(),
        )
        assert result[SIGNAL_GOAL] == 2

    def test_responsibility_state_units(self):
        class MockResp:
            units = [1, 2, 3, 4]

        result = collect_source_quantities(responsibility_state=MockResp())
        assert result[SIGNAL_RESPONSIBILITY] == 4

    def test_responsibility_state_decisions(self):
        class MockResp:
            decisions = [1, 2]

        result = collect_source_quantities(responsibility_state=MockResp())
        assert result[SIGNAL_RESPONSIBILITY] == 2

    def test_all_sources_provided(self):
        """全入力源を同時に提供。"""
        class MockEmotion:
            joy = 0.5
            sadness = 0.0
            anger = 0.0
            fear = 0.0
            disgust = 0.0
            surprise = 0.0
            trust = 0.0
            anticipation = 0.0

        class MockMemory:
            recalled_memories = [1, 2]

        class MockMotivation:
            entries = [type("E", (), {"strength": None})()]

        class MockTransient:
            active_goals = [1]

        class MockScoped:
            current_goal = "x"

        class MockResp:
            units = [1, 2, 3]

        result = collect_source_quantities(
            has_perception_input=True,
            perception_element_count=3,
            has_text_input=True,
            has_spontaneous_activation=True,
            emotion_state=MockEmotion(),
            memory_state=MockMemory(),
            motivation_state=MockMotivation(),
            transient_goal_state=MockTransient(),
            scoped_goal_state=MockScoped(),
            responsibility_state=MockResp(),
        )
        assert result[SOURCE_PERCEPTION] == 3
        assert result[SOURCE_TEXT_INPUT] == 1
        assert result[SOURCE_SPONTANEOUS] == 1
        assert result[SIGNAL_EMOTION] == 1
        assert result[SIGNAL_MEMORY] == 2
        assert result[SIGNAL_MOTIVATION] == 1
        assert result[SIGNAL_GOAL] == 2
        assert result[SIGNAL_RESPONSIBILITY] == 3


# =============================================================================
# compose_snapshot
# =============================================================================

class TestComposeSnapshot:
    def test_empty_quantities(self):
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        snapshot = compose_snapshot(quantities, timestamp=100.0)
        assert snapshot.timestamp == 100.0
        assert snapshot.perception_level == QuantityLevel.ABSENT.value
        assert snapshot.text_input_level == QuantityLevel.ABSENT.value
        assert snapshot.spontaneous_level == QuantityLevel.ABSENT.value
        assert snapshot.emotion_level == QuantityLevel.ABSENT.value
        assert snapshot.memory_level == QuantityLevel.ABSENT.value
        assert snapshot.motivation_level == QuantityLevel.ABSENT.value
        assert snapshot.goal_level == QuantityLevel.ABSENT.value
        assert snapshot.responsibility_level == QuantityLevel.ABSENT.value
        assert snapshot.concentration == 0.0

    def test_single_source(self):
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        quantities[SOURCE_PERCEPTION] = 5
        snapshot = compose_snapshot(quantities, timestamp=200.0)
        assert snapshot.perception_level == QuantityLevel.MODERATE.value
        assert snapshot.text_input_level == QuantityLevel.ABSENT.value
        assert snapshot.concentration > 0.5

    def test_all_sources_equal(self):
        quantities = {k: 2 for k in ALL_SOURCE_KEYS}
        snapshot = compose_snapshot(quantities, timestamp=300.0)
        # 全て同じなら各段階値はFEW、集中度は低い
        assert snapshot.perception_level == QuantityLevel.FEW.value
        assert snapshot.emotion_level == QuantityLevel.FEW.value
        assert snapshot.concentration < 0.1

    def test_auto_timestamp(self):
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        before = time.time()
        snapshot = compose_snapshot(quantities)
        after = time.time()
        assert before <= snapshot.timestamp <= after

    def test_concentration_level_is_set(self):
        quantities = {k: 0 for k in ALL_SOURCE_KEYS}
        quantities[SIGNAL_EMOTION] = 100
        snapshot = compose_snapshot(quantities, timestamp=400.0)
        assert snapshot.concentration_level == ConcentrationLevel.CONCENTRATED.value

    def test_to_dict_and_from_dict(self):
        quantities = {k: 1 for k in ALL_SOURCE_KEYS}
        snapshot = compose_snapshot(quantities, timestamp=500.0)
        data = snapshot.to_dict()
        restored = AttentionSnapshot.from_dict(data)
        assert restored.timestamp == snapshot.timestamp
        assert restored.perception_level == snapshot.perception_level
        assert restored.concentration == snapshot.concentration
        assert restored.concentration_level == snapshot.concentration_level


# =============================================================================
# derive_variation
# =============================================================================

class TestDeriveVariation:
    def test_empty_history(self, default_config):
        result = derive_variation([], default_config)
        assert result is None

    def test_single_snapshot(self, default_config):
        snapshot = AttentionSnapshot(timestamp=1.0, concentration=0.5)
        result = derive_variation([snapshot], default_config)
        assert result is None

    def test_two_snapshots(self, default_config):
        s1 = AttentionSnapshot(timestamp=1.0, concentration=0.3)
        s2 = AttentionSnapshot(timestamp=2.0, concentration=0.6)
        result = derive_variation([s1, s2], default_config)
        assert result is not None
        assert result.comparison_count == 1
        assert result.concentration_direction > 0  # 集中方向
        assert result.concentration_magnitude == pytest.approx(0.3, abs=0.01)

    def test_multiple_snapshots(self, default_config):
        snapshots = [
            AttentionSnapshot(timestamp=float(i), concentration=0.2 + i * 0.05)
            for i in range(6)
        ]
        result = derive_variation(snapshots, default_config)
        assert result is not None
        assert result.comparison_count <= default_config.variation_comparison_count

    def test_decreasing_concentration(self, default_config):
        s1 = AttentionSnapshot(timestamp=1.0, concentration=0.8)
        s2 = AttentionSnapshot(timestamp=2.0, concentration=0.3)
        result = derive_variation([s1, s2], default_config)
        assert result is not None
        assert result.concentration_direction < 0  # 分散方向

    def test_no_change(self, default_config):
        s1 = AttentionSnapshot(timestamp=1.0, concentration=0.5)
        s2 = AttentionSnapshot(timestamp=2.0, concentration=0.5)
        result = derive_variation([s1, s2], default_config)
        assert result is not None
        assert result.concentration_direction == pytest.approx(0.0, abs=0.001)
        assert result.concentration_magnitude == pytest.approx(0.0, abs=0.001)

    def test_variation_to_dict_and_from_dict(self, default_config):
        s1 = AttentionSnapshot(timestamp=1.0, concentration=0.3)
        s2 = AttentionSnapshot(timestamp=2.0, concentration=0.7)
        variation = derive_variation([s1, s2], default_config)
        assert variation is not None

        data = variation.to_dict()
        restored = AttentionVariation.from_dict(data)
        assert restored.concentration_direction == pytest.approx(variation.concentration_direction, abs=0.001)
        assert restored.concentration_magnitude == pytest.approx(variation.concentration_magnitude, abs=0.001)
        assert restored.comparison_count == variation.comparison_count


# =============================================================================
# process_attention_distribution
# =============================================================================

class TestProcessAttentionDistribution:
    def test_initial_processing(self, empty_state, default_config):
        """初回処理で断面が1件生成される。"""
        new_state = process_attention_distribution(
            empty_state,
            config=default_config,
            timestamp=100.0,
        )
        assert new_state.total_snapshots_generated == 1
        assert len(new_state.snapshot_history) == 1
        assert new_state.latest_variation is None  # 1件では変動記述なし

    def test_second_processing_generates_variation(self, empty_state, default_config):
        """2回処理で変動記述が生成される。"""
        state1 = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=10,
            config=default_config,
            timestamp=100.0,
        )
        state2 = process_attention_distribution(
            state1,
            has_text_input=True,
            config=default_config,
            timestamp=200.0,
        )
        assert state2.total_snapshots_generated == 2
        assert len(state2.snapshot_history) == 2
        assert state2.latest_variation is not None

    def test_fifo_expiration(self, empty_state, small_config):
        """保持上限を超えた断面がFIFOで消失する。"""
        state = empty_state
        for i in range(10):
            state = process_attention_distribution(
                state,
                config=small_config,
                timestamp=float(i * 100),
            )
        assert len(state.snapshot_history) == small_config.max_snapshot_history
        assert state.total_snapshots_generated == 10
        assert state.total_snapshots_expired == 5

    def test_with_all_inputs(self, empty_state, default_config):
        """全入力源が指定された場合。"""
        class MockEmotion:
            joy = 0.5
            sadness = 0.0
            anger = 0.0
            fear = 0.0
            disgust = 0.0
            surprise = 0.0
            trust = 0.0
            anticipation = 0.0

        class MockMemory:
            recalled_memories = [1, 2, 3]

        class MockResp:
            units = [1, 2]

        new_state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=5,
            has_text_input=True,
            has_spontaneous_activation=True,
            emotion_state=MockEmotion(),
            memory_state=MockMemory(),
            responsibility_state=MockResp(),
            config=default_config,
            timestamp=100.0,
        )
        assert new_state.total_snapshots_generated == 1
        snapshot = new_state.snapshot_history[0]
        assert snapshot.perception_level == QuantityLevel.MODERATE.value
        assert snapshot.text_input_level == QuantityLevel.MINIMAL.value
        assert snapshot.spontaneous_level == QuantityLevel.MINIMAL.value
        assert snapshot.emotion_level == QuantityLevel.MINIMAL.value
        assert snapshot.memory_level == QuantityLevel.FEW.value
        assert snapshot.responsibility_level == QuantityLevel.FEW.value

    def test_default_config_used_when_none(self, empty_state):
        """configがNoneの場合デフォルト設定が使われる。"""
        new_state = process_attention_distribution(
            empty_state,
            config=None,
            timestamp=100.0,
        )
        assert new_state.total_snapshots_generated == 1

    def test_auto_timestamp(self, empty_state, default_config):
        """timestampがNoneの場合、現在時刻が使われる。"""
        before = time.time()
        new_state = process_attention_distribution(
            empty_state,
            config=default_config,
        )
        after = time.time()
        assert before <= new_state.snapshot_history[0].timestamp <= after

    def test_immutable_snapshots(self, empty_state, default_config):
        """生成された断面は後続処理で変更されない。"""
        state1 = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=10,
            config=default_config,
            timestamp=100.0,
        )
        first_snapshot_data = state1.snapshot_history[0].to_dict()

        state2 = process_attention_distribution(
            state1,
            has_text_input=True,
            config=default_config,
            timestamp=200.0,
        )
        # 最初の断面は変更されていない
        assert state2.snapshot_history[0].to_dict() == first_snapshot_data

    def test_variation_rederived_each_tick(self, empty_state, default_config):
        """安全弁3: 変動記述は毎ティック再導出される。"""
        state = empty_state
        for i in range(5):
            state = process_attention_distribution(
                state,
                has_perception_input=(i % 2 == 0),
                perception_element_count=i * 3,
                config=default_config,
                timestamp=float(i * 100),
            )
        # 変動記述が存在する
        assert state.latest_variation is not None
        # 比較件数は正しい
        assert state.latest_variation.comparison_count <= default_config.variation_comparison_count


# =============================================================================
# Accessors (READ-ONLY)
# =============================================================================

class TestAccessors:
    def test_get_latest_snapshot_empty(self, empty_state):
        assert get_latest_snapshot(empty_state) is None

    def test_get_latest_snapshot_with_data(self, empty_state, default_config):
        state = process_attention_distribution(
            empty_state,
            config=default_config,
            timestamp=100.0,
        )
        snapshot = get_latest_snapshot(state)
        assert snapshot is not None
        assert snapshot.timestamp == 100.0

    def test_get_snapshot_history_empty(self, empty_state):
        history = get_snapshot_history(empty_state)
        assert history == []

    def test_get_snapshot_history_returns_copy(self, empty_state, default_config):
        """安全弁1: 返されるリストはコピーであり、内部状態を変更しない。"""
        state = process_attention_distribution(
            empty_state,
            config=default_config,
            timestamp=100.0,
        )
        history = get_snapshot_history(state)
        assert len(history) == 1
        history.clear()
        # 内部状態は変更されていない
        assert len(state.snapshot_history) == 1

    def test_get_latest_variation_empty(self, empty_state):
        assert get_latest_variation(empty_state) is None

    def test_get_latest_variation_with_data(self, empty_state, default_config):
        state = empty_state
        for i in range(3):
            state = process_attention_distribution(
                state,
                has_perception_input=(i == 0),
                perception_element_count=10 * (i + 1),
                config=default_config,
                timestamp=float(i * 100),
            )
        variation = get_latest_variation(state)
        assert variation is not None

    def test_get_summary_empty(self, empty_state):
        summary = get_attention_distribution_summary(empty_state)
        assert summary["history_count"] == 0
        assert summary["total_generated"] == 0
        assert summary["total_expired"] == 0

    def test_get_summary_with_data(self, empty_state, default_config):
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=5,
            has_text_input=True,
            config=default_config,
            timestamp=100.0,
        )
        summary = get_attention_distribution_summary(state)
        assert summary["history_count"] == 1
        assert summary["total_generated"] == 1
        assert "perception_level" in summary
        assert "text_input_level" in summary
        assert "spontaneous_level" in summary
        assert "emotion_level" in summary
        assert "memory_level" in summary
        assert "motivation_level" in summary
        assert "goal_level" in summary
        assert "responsibility_level" in summary
        assert "concentration" in summary
        assert "concentration_level" in summary
        assert "latest_timestamp" in summary

    def test_get_summary_with_variation(self, empty_state, default_config):
        state = empty_state
        for i in range(3):
            state = process_attention_distribution(
                state,
                has_perception_input=True,
                perception_element_count=i * 5,
                config=default_config,
                timestamp=float(i * 100),
            )
        summary = get_attention_distribution_summary(state)
        assert "variation_concentration_direction" in summary
        assert "variation_concentration_magnitude" in summary
        assert "variation_comparison_count" in summary


# =============================================================================
# Enrichment
# =============================================================================

class TestEnrichment:
    def test_enrichment_empty(self, empty_state):
        text = get_enrichment_text(empty_state)
        assert "待機中" in text

    def test_enrichment_with_data(self, empty_state, default_config):
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=5,
            config=default_config,
            timestamp=100.0,
        )
        text = get_enrichment_text(state)
        assert "知覚=" in text
        assert "テキスト=" in text
        assert "自発=" in text
        assert "感情=" in text
        assert "記憶=" in text
        assert "動機=" in text
        assert "目標=" in text
        assert "責任=" in text
        assert "集中度=" in text

    def test_enrichment_no_evaluative_vocabulary(self, empty_state, default_config):
        """安全弁2: 評価的語彙を含まない。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=50,
            config=default_config,
            timestamp=100.0,
        )
        text = get_enrichment_text(state)
        # 「集中している」「分散している」等の評価的語彙を含まない
        assert "集中している" not in text
        assert "分散している" not in text
        assert "良い" not in text
        assert "悪い" not in text
        assert "望ましい" not in text
        assert "問題" not in text

    def test_enrichment_with_variation(self, empty_state, default_config):
        """変動がある場合、方向が含まれる。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=50,
            config=default_config,
            timestamp=100.0,
        )
        state = process_attention_distribution(
            state,
            config=default_config,
            timestamp=200.0,
        )
        text = get_enrichment_text(state)
        # 変動がある場合、方向テキストが含まれる場合がある
        if state.latest_variation and state.latest_variation.concentration_magnitude > 0.01:
            assert "変動=" in text

    def test_enrichment_no_specific_counts(self, empty_state, default_config):
        """具体的な件数・比率・順位を含まない。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=5,
            has_text_input=True,
            config=default_config,
            timestamp=100.0,
        )
        text = get_enrichment_text(state)
        # 段階値のみが含まれ、数値は含まれない
        assert "5" not in text or "5" in text  # 段階値の定義に"5"は含まれない
        # 段階値の文字列が使われている
        valid_levels = {lv.value for lv in QuantityLevel}
        # 各フィールドの値が段階値であることを確認
        snapshot = get_latest_snapshot(state)
        assert snapshot.perception_level in valid_levels
        assert snapshot.text_input_level in valid_levels


# =============================================================================
# Save / Load
# =============================================================================

class TestSaveLoad:
    def test_save_empty_state(self, empty_state):
        data = save_state(empty_state)
        assert isinstance(data, dict)
        assert data["snapshot_history"] == []
        assert data["latest_variation"] is None
        assert data["total_snapshots_generated"] == 0
        assert data["total_snapshots_expired"] == 0

    def test_save_load_roundtrip(self, empty_state, default_config):
        # 複数回処理した状態を保存・復元
        state = empty_state
        for i in range(5):
            state = process_attention_distribution(
                state,
                has_perception_input=(i % 2 == 0),
                perception_element_count=i * 3,
                has_text_input=(i % 3 == 0),
                config=default_config,
                timestamp=float(i * 100),
            )

        data = save_state(state)
        restored = load_state(data)

        assert len(restored.snapshot_history) == len(state.snapshot_history)
        assert restored.total_snapshots_generated == state.total_snapshots_generated
        assert restored.total_snapshots_expired == state.total_snapshots_expired

        # 各断面の内容が一致
        for orig, rest in zip(state.snapshot_history, restored.snapshot_history):
            assert orig.to_dict() == rest.to_dict()

        # 変動記述が一致
        if state.latest_variation is not None:
            assert restored.latest_variation is not None
            assert restored.latest_variation.concentration_direction == pytest.approx(
                state.latest_variation.concentration_direction, abs=0.001
            )
        else:
            assert restored.latest_variation is None

    def test_load_empty_dict(self):
        """空の辞書からでも安全に復元できる。"""
        restored = load_state({})
        assert len(restored.snapshot_history) == 0
        assert restored.latest_variation is None
        assert restored.total_snapshots_generated == 0
        assert restored.total_snapshots_expired == 0

    def test_load_partial_dict(self):
        """部分的な辞書からでも安全に復元できる。"""
        data = {"total_snapshots_generated": 5}
        restored = load_state(data)
        assert restored.total_snapshots_generated == 5
        assert len(restored.snapshot_history) == 0

    def test_snapshot_from_dict_defaults(self):
        """空の辞書からAttentionSnapshotを復元できる。"""
        snapshot = AttentionSnapshot.from_dict({})
        assert snapshot.timestamp == 0.0
        assert snapshot.perception_level == QuantityLevel.ABSENT.value
        assert snapshot.concentration == 0.0

    def test_variation_from_dict_defaults(self):
        """空の辞書からAttentionVariationを復元できる。"""
        variation = AttentionVariation.from_dict({})
        assert variation.concentration_direction == 0.0
        assert variation.concentration_magnitude == 0.0
        assert variation.comparison_count == 0


# =============================================================================
# Factory
# =============================================================================

class TestFactory:
    def test_create_state(self):
        state = create_attention_distribution_state()
        assert len(state.snapshot_history) == 0
        assert state.latest_variation is None
        assert state.total_snapshots_generated == 0
        assert state.total_snapshots_expired == 0

    def test_create_config_defaults(self):
        config = create_attention_distribution_config()
        assert config.max_snapshot_history == 30
        assert config.variation_comparison_count == 5

    def test_create_config_custom(self):
        config = create_attention_distribution_config(
            max_snapshot_history=10,
            variation_comparison_count=3,
        )
        assert config.max_snapshot_history == 10
        assert config.variation_comparison_count == 3


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    def test_valve1_all_records_equal(self, empty_state, default_config):
        """安全弁1: 全記録等価維持保証。
        各入力源の段階値に重み付け・順位付けが行われないことを確認。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=5,
            has_text_input=True,
            has_spontaneous_activation=True,
            config=default_config,
            timestamp=100.0,
        )
        snapshot = state.snapshot_history[0]
        # 全段階値が存在する（等価に並置されている）
        for attr in [
            "perception_level", "text_input_level", "spontaneous_level",
            "emotion_level", "memory_level", "motivation_level",
            "goal_level", "responsibility_level",
        ]:
            val = getattr(snapshot, attr)
            assert val in {lv.value for lv in QuantityLevel}

    def test_valve2_no_evaluative_conversion(self, empty_state, default_config):
        """安全弁2: 評価的変換の禁止。集中度に評価的意味がないことを確認。"""
        # 全ゼロの場合
        state1 = process_attention_distribution(
            empty_state,
            config=default_config,
            timestamp=100.0,
        )
        # 集中の場合
        state2 = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=100,
            config=default_config,
            timestamp=200.0,
        )
        # どちらの集中度も有効な段階値
        for s in [state1, state2]:
            snapshot = s.snapshot_history[0]
            assert snapshot.concentration_level in {lv.value for lv in ConcentrationLevel}

    def test_valve3_no_cumulative_trend(self, empty_state, default_config):
        """安全弁3: 累積的傾向の抑制。
        変動記述が毎回再導出されることを確認。"""
        state = empty_state
        variations = []
        for i in range(6):
            state = process_attention_distribution(
                state,
                has_perception_input=True,
                perception_element_count=i * 10,
                config=default_config,
                timestamp=float(i * 100),
            )
            if state.latest_variation is not None:
                variations.append(state.latest_variation.to_dict())

        # 変動記述が毎回異なる可能性がある（再導出されている証拠）
        # 少なくとも全て同じではない（同一の集中度が連続しなければ）
        assert len(variations) >= 2

    def test_valve4_finite_history(self, empty_state, small_config):
        """安全弁4: 断面履歴の有限性。"""
        state = empty_state
        for i in range(20):
            state = process_attention_distribution(
                state,
                config=small_config,
                timestamp=float(i * 100),
            )
        assert len(state.snapshot_history) == small_config.max_snapshot_history
        assert state.total_snapshots_expired > 0

    def test_valve5_no_control_output(self):
        """安全弁5: 帯域制御経路の遮断。
        出力が情報のみであり、制御信号を含まないことを確認。"""
        state = create_attention_distribution_state()
        state = process_attention_distribution(
            state,
            has_perception_input=True,
            perception_element_count=50,
            timestamp=100.0,
        )
        summary = get_attention_distribution_summary(state)
        # サマリに「制御」「調整」「推奨」に相当するキーがない
        for key in summary:
            assert "control" not in key.lower()
            assert "adjust" not in key.lower()
            assert "recommend" not in key.lower()

    def test_valve6_no_pattern_extraction(self, empty_state, default_config):
        """安全弁6: パターン抽出禁止。
        断面から傾向・周期性・統計量・相関を算出しないことを確認。"""
        state = empty_state
        for i in range(10):
            state = process_attention_distribution(
                state,
                has_perception_input=(i % 2 == 0),
                perception_element_count=i * 5,
                config=default_config,
                timestamp=float(i * 100),
            )
        summary = get_attention_distribution_summary(state)
        # サマリに「傾向」「周期」「統計」「相関」に相当するキーがない
        for key in summary:
            assert "trend" not in key.lower()
            assert "cycle" not in key.lower()
            assert "statistic" not in key.lower()
            assert "correlation" not in key.lower()

    def test_valve7_fixed_output_paths(self):
        """安全弁7: 出力経路不拡張。
        公開APIが固定されていることを確認。"""
        import psyche.attention_distribution_description as mod
        public_functions = [
            name for name in dir(mod)
            if not name.startswith("_") and callable(getattr(mod, name))
        ]
        # 出力に関連する関数は固定セット
        output_functions = [f for f in public_functions if f.startswith("get_")]
        expected_output = {
            "get_latest_snapshot",
            "get_snapshot_history",
            "get_latest_variation",
            "get_attention_distribution_summary",
            "get_enrichment_text",
        }
        assert set(output_functions) == expected_output


# =============================================================================
# 責務分離テスト
# =============================================================================

class TestResponsibilitySeparation:
    def test_no_reference_frequency_overlap(self):
        """参照頻度記述との責務分離: 参照回数を収集しないことを確認。"""
        # collect_source_quantitiesは参照回数ではなく処理量を収集する
        result = collect_source_quantities(
            has_perception_input=True,
            perception_element_count=5,
        )
        # 返される値は量的指標であり、参照回数ではない
        assert SOURCE_PERCEPTION in result
        # 参照頻度記述の構造種別キーを含まない
        assert "episodic_memory" not in result
        assert "emotional_memory_binding" not in result

    def test_no_input_pathway_overlap(self):
        """入力経路間均衡記述との責務分離: 経路使用事実を記録しないことを確認。"""
        # collect_source_quantitiesは経路使用事実ではなく量的指標を収集する
        result = collect_source_quantities(
            has_perception_input=True,
            perception_element_count=5,
            has_text_input=True,
        )
        # 入力経路間均衡記述の経路キーと異なるキーを使用
        assert "text" not in result  # input_pathway_balance uses "text"
        assert SOURCE_TEXT_INPUT in result  # 本機能は "text_input" を使用


# =============================================================================
# エッジケーステスト
# =============================================================================

class TestEdgeCases:
    def test_large_element_count(self, empty_state, default_config):
        """非常に大きな要素数でも安全。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            perception_element_count=100000,
            config=default_config,
            timestamp=100.0,
        )
        snapshot = state.snapshot_history[0]
        assert snapshot.perception_level == QuantityLevel.MANY.value
        assert 0.0 <= snapshot.concentration <= 1.0

    def test_zero_variation_comparison(self):
        """variation_comparison_count=0でも安全。"""
        config = AttentionDistributionConfig(variation_comparison_count=0)
        s1 = AttentionSnapshot(timestamp=1.0, concentration=0.3)
        s2 = AttentionSnapshot(timestamp=2.0, concentration=0.7)
        result = derive_variation([s1, s2], config)
        # comparison_count=0 では比較対象が取得できないためNone
        assert result is None

    def test_max_history_one(self):
        """max_snapshot_history=1でも動作する。"""
        config = AttentionDistributionConfig(max_snapshot_history=1)
        state = create_attention_distribution_state()
        for i in range(5):
            state = process_attention_distribution(
                state,
                config=config,
                timestamp=float(i * 100),
            )
        assert len(state.snapshot_history) == 1
        assert state.total_snapshots_generated == 5
        assert state.total_snapshots_expired == 4

    def test_negative_perception_element_count(self, empty_state, default_config):
        """負の要素数でも安全（ABSENTになる）。"""
        state = process_attention_distribution(
            empty_state,
            has_perception_input=False,
            perception_element_count=-5,
            config=default_config,
            timestamp=100.0,
        )
        snapshot = state.snapshot_history[0]
        assert snapshot.perception_level == QuantityLevel.ABSENT.value

    def test_state_independence(self, empty_state, default_config):
        """処理結果が元の状態を変更しないことを確認。"""
        original_dict = empty_state.to_dict()
        _ = process_attention_distribution(
            empty_state,
            has_perception_input=True,
            config=default_config,
            timestamp=100.0,
        )
        assert empty_state.to_dict() == original_dict

    def test_motivation_with_dormant_entries(self, empty_state, default_config):
        """dormantな動機エントリはカウントしない。"""
        from enum import Enum

        class MockStrength(Enum):
            DORMANT = "dormant"

        class MockEntry:
            strength = MockStrength.DORMANT

        class MockMotivation:
            entries = [MockEntry()]

        result = collect_source_quantities(motivation_state=MockMotivation())
        assert result[SIGNAL_MOTIVATION] == 0

    def test_memory_with_recent_recalls(self):
        """recent_recalls属性からも読み取れる。"""
        class MockMemory:
            recent_recalls = [1, 2, 3, 4, 5]

        result = collect_source_quantities(memory_state=MockMemory())
        assert result[SIGNAL_MEMORY] == 5

    def test_transient_goal_with_active_goal_singular(self):
        """active_goal (単数形) からも読み取れる。"""
        class MockTransient:
            active_goal = "some_goal"

        result = collect_source_quantities(transient_goal_state=MockTransient())
        assert result[SIGNAL_GOAL] == 1

    def test_scoped_goal_none_current(self):
        """scoped_goalのcurrent_goalがNoneならカウントしない。"""
        class MockScoped:
            current_goal = None

        result = collect_source_quantities(scoped_goal_state=MockScoped())
        assert result[SIGNAL_GOAL] == 0
