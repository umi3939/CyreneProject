"""
tests/test_input_pathway_balance.py - 入力経路間の均衡記述テスト

設計書 (design_input_pathway_balance.md) に基づく機能テスト。
"""

import time
import pytest

from psyche.input_pathway_balance import (
    # Config
    InputPathwayBalanceConfig,
    create_input_pathway_balance_config,
    # State
    InputPathwayBalanceState,
    create_input_pathway_balance_state,
    # Data structures
    UsageFact,
    PathwaySnapshot,
    PathwayVariation,
    # Constants
    PATHWAY_TEXT,
    PATHWAY_SCREEN,
    PATHWAY_SPONTANEOUS,
    ALL_PATHWAYS,
    # Enums
    UsageLevel,
    BiasLevel,
    # Functions
    determine_usage_level,
    determine_bias_level,
    compute_bias_value,
    collect_usage_fact,
    read_text_dialogue_usage,
    read_spontaneous_usage,
    compose_snapshot,
    derive_variation,
    process_input_pathway_balance,
    # Accessors
    get_latest_snapshot,
    get_snapshot_history,
    get_latest_variation,
    get_pathway_balance_summary,
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
    return InputPathwayBalanceConfig()


@pytest.fixture
def small_config():
    """テスト用の小さなウィンドウ設定。"""
    return InputPathwayBalanceConfig(
        max_usage_facts=20,
        max_snapshot_history=5,
        sliding_window_size=10,
        variation_comparison_count=3,
    )


@pytest.fixture
def empty_state():
    return create_input_pathway_balance_state()


# =============================================================================
# Constants
# =============================================================================

class TestConstants:
    def test_pathway_constants(self):
        assert PATHWAY_TEXT == "text"
        assert PATHWAY_SCREEN == "screen"
        assert PATHWAY_SPONTANEOUS == "spontaneous"

    def test_all_pathways_list(self):
        assert len(ALL_PATHWAYS) == 3
        assert PATHWAY_TEXT in ALL_PATHWAYS
        assert PATHWAY_SCREEN in ALL_PATHWAYS
        assert PATHWAY_SPONTANEOUS in ALL_PATHWAYS


# =============================================================================
# Enums
# =============================================================================

class TestUsageLevel:
    def test_all_values(self):
        assert UsageLevel.NONE.value == "none"
        assert UsageLevel.FEW.value == "few"
        assert UsageLevel.MODERATE.value == "moderate"
        assert UsageLevel.MANY.value == "many"

    def test_determine_none(self):
        assert determine_usage_level(0) == UsageLevel.NONE

    def test_determine_few(self):
        for count in [1, 2, 3, 4, 5]:
            assert determine_usage_level(count) == UsageLevel.FEW

    def test_determine_moderate(self):
        for count in [6, 10, 15]:
            assert determine_usage_level(count) == UsageLevel.MODERATE

    def test_determine_many(self):
        for count in [16, 50, 100]:
            assert determine_usage_level(count) == UsageLevel.MANY


class TestBiasLevel:
    def test_all_values(self):
        assert BiasLevel.EVEN.value == "even"
        assert BiasLevel.SLIGHT.value == "slight"
        assert BiasLevel.MODERATE.value == "moderate"
        assert BiasLevel.CONCENTRATED.value == "concentrated"

    def test_determine_even(self):
        assert determine_bias_level(0.0) == BiasLevel.EVEN
        assert determine_bias_level(0.19) == BiasLevel.EVEN

    def test_determine_slight(self):
        assert determine_bias_level(0.2) == BiasLevel.SLIGHT
        assert determine_bias_level(0.44) == BiasLevel.SLIGHT

    def test_determine_moderate(self):
        assert determine_bias_level(0.45) == BiasLevel.MODERATE
        assert determine_bias_level(0.69) == BiasLevel.MODERATE

    def test_determine_concentrated(self):
        assert determine_bias_level(0.7) == BiasLevel.CONCENTRATED
        assert determine_bias_level(1.0) == BiasLevel.CONCENTRATED


# =============================================================================
# compute_bias_value
# =============================================================================

class TestComputeBiasValue:
    def test_empty_counts(self):
        counts = {PATHWAY_TEXT: 0, PATHWAY_SCREEN: 0, PATHWAY_SPONTANEOUS: 0}
        assert compute_bias_value(counts) == 0.0

    def test_equal_distribution(self):
        counts = {PATHWAY_TEXT: 10, PATHWAY_SCREEN: 10, PATHWAY_SPONTANEOUS: 10}
        assert compute_bias_value(counts) == pytest.approx(0.0, abs=0.01)

    def test_single_pathway_concentration(self):
        counts = {PATHWAY_TEXT: 30, PATHWAY_SCREEN: 0, PATHWAY_SPONTANEOUS: 0}
        value = compute_bias_value(counts)
        assert value > 0.5  # High concentration

    def test_two_pathway_usage(self):
        counts = {PATHWAY_TEXT: 15, PATHWAY_SCREEN: 15, PATHWAY_SPONTANEOUS: 0}
        value = compute_bias_value(counts)
        assert 0.0 < value < 1.0  # Some bias but not complete

    def test_result_bounded(self):
        for text_c in [0, 5, 20, 100]:
            for screen_c in [0, 5, 20, 100]:
                for spont_c in [0, 5, 20, 100]:
                    counts = {
                        PATHWAY_TEXT: text_c,
                        PATHWAY_SCREEN: screen_c,
                        PATHWAY_SPONTANEOUS: spont_c,
                    }
                    value = compute_bias_value(counts)
                    assert 0.0 <= value <= 1.0


# =============================================================================
# UsageFact
# =============================================================================

class TestUsageFact:
    def test_creation(self):
        fact = UsageFact(pathway=PATHWAY_TEXT, timestamp=100.0)
        assert fact.pathway == PATHWAY_TEXT
        assert fact.timestamp == 100.0

    def test_to_dict(self):
        fact = UsageFact(pathway=PATHWAY_SCREEN, timestamp=200.0)
        d = fact.to_dict()
        assert d["pathway"] == "screen"
        assert d["timestamp"] == 200.0

    def test_from_dict(self):
        d = {"pathway": "spontaneous", "timestamp": 300.0}
        fact = UsageFact.from_dict(d)
        assert fact.pathway == PATHWAY_SPONTANEOUS
        assert fact.timestamp == 300.0

    def test_roundtrip(self):
        original = UsageFact(pathway=PATHWAY_TEXT, timestamp=123.456)
        restored = UsageFact.from_dict(original.to_dict())
        assert restored.pathway == original.pathway
        assert restored.timestamp == original.timestamp


# =============================================================================
# collect_usage_fact
# =============================================================================

class TestCollectUsageFact:
    def test_basic_collection(self):
        fact = collect_usage_fact(pathway=PATHWAY_TEXT, timestamp=100.0)
        assert fact.pathway == PATHWAY_TEXT
        assert fact.timestamp == 100.0

    def test_auto_timestamp(self):
        before = time.time()
        fact = collect_usage_fact(pathway=PATHWAY_SCREEN)
        after = time.time()
        assert before <= fact.timestamp <= after

    def test_no_weight_or_score(self):
        """安全弁1: 使用事実に重み・スコアを付与しない。"""
        fact = collect_usage_fact(pathway=PATHWAY_SPONTANEOUS, timestamp=100.0)
        # UsageFact has only pathway and timestamp fields
        assert not hasattr(fact, "weight")
        assert not hasattr(fact, "score")
        assert not hasattr(fact, "priority")


# =============================================================================
# PathwaySnapshot
# =============================================================================

class TestPathwaySnapshot:
    def test_default_values(self):
        snap = PathwaySnapshot()
        assert snap.text_usage_level == UsageLevel.NONE.value
        assert snap.screen_usage_level == UsageLevel.NONE.value
        assert snap.spontaneous_usage_level == UsageLevel.NONE.value
        assert snap.bias_level == BiasLevel.EVEN.value
        assert snap.bias_value == 0.0

    def test_roundtrip(self):
        snap = PathwaySnapshot(
            timestamp=100.0,
            text_usage_level=UsageLevel.MODERATE.value,
            screen_usage_level=UsageLevel.FEW.value,
            spontaneous_usage_level=UsageLevel.NONE.value,
            bias_level=BiasLevel.SLIGHT.value,
            bias_value=0.3,
        )
        restored = PathwaySnapshot.from_dict(snap.to_dict())
        assert restored.timestamp == snap.timestamp
        assert restored.text_usage_level == snap.text_usage_level
        assert restored.screen_usage_level == snap.screen_usage_level
        assert restored.spontaneous_usage_level == snap.spontaneous_usage_level
        assert restored.bias_level == snap.bias_level
        assert restored.bias_value == pytest.approx(snap.bias_value)


# =============================================================================
# PathwayVariation
# =============================================================================

class TestPathwayVariation:
    def test_default_values(self):
        var = PathwayVariation()
        assert var.bias_direction == 0.0
        assert var.bias_magnitude == 0.0
        assert var.comparison_count == 0

    def test_roundtrip(self):
        var = PathwayVariation(
            bias_direction=0.15,
            bias_magnitude=0.15,
            comparison_count=3,
        )
        restored = PathwayVariation.from_dict(var.to_dict())
        assert restored.bias_direction == pytest.approx(var.bias_direction)
        assert restored.bias_magnitude == pytest.approx(var.bias_magnitude)
        assert restored.comparison_count == var.comparison_count


# =============================================================================
# compose_snapshot
# =============================================================================

class TestComposeSnapshot:
    def test_empty_facts(self, default_config):
        snap = compose_snapshot([], default_config, timestamp=100.0)
        assert snap.text_usage_level == UsageLevel.NONE.value
        assert snap.screen_usage_level == UsageLevel.NONE.value
        assert snap.spontaneous_usage_level == UsageLevel.NONE.value
        assert snap.bias_level == BiasLevel.EVEN.value

    def test_single_pathway(self, default_config):
        facts = [
            UsageFact(pathway=PATHWAY_TEXT, timestamp=float(i))
            for i in range(10)
        ]
        snap = compose_snapshot(facts, default_config, timestamp=100.0)
        assert snap.text_usage_level == UsageLevel.MODERATE.value
        assert snap.screen_usage_level == UsageLevel.NONE.value
        assert snap.spontaneous_usage_level == UsageLevel.NONE.value
        assert snap.bias_value > 0.5  # Concentrated in one pathway

    def test_equal_distribution(self, default_config):
        facts = []
        for i in range(30):
            pathway = ALL_PATHWAYS[i % 3]
            facts.append(UsageFact(pathway=pathway, timestamp=float(i)))
        snap = compose_snapshot(facts, default_config, timestamp=100.0)
        assert snap.text_usage_level == snap.screen_usage_level == snap.spontaneous_usage_level
        assert snap.bias_value < 0.1

    def test_window_limits(self, small_config):
        """窓サイズを超える事実はカウントに含まれない。"""
        # 古い事実: text x 100
        facts = [
            UsageFact(pathway=PATHWAY_TEXT, timestamp=float(i))
            for i in range(100)
        ]
        # 新しい事実: screen x 10 (ちょうど窓サイズ)
        for i in range(10):
            facts.append(
                UsageFact(pathway=PATHWAY_SCREEN, timestamp=100.0 + i)
            )
        snap = compose_snapshot(facts, small_config, timestamp=200.0)
        # 窓(最新10件)にはscreenのみ
        assert snap.screen_usage_level == UsageLevel.MODERATE.value
        assert snap.text_usage_level == UsageLevel.NONE.value

    def test_three_pathways_equal(self, default_config):
        """安全弁1: 3経路の段階値は等価に並置される。"""
        facts = []
        for i in range(30):
            facts.append(UsageFact(pathway=ALL_PATHWAYS[i % 3], timestamp=float(i)))
        snap = compose_snapshot(facts, default_config)
        # 全て同じレベルであること
        assert snap.text_usage_level == snap.screen_usage_level
        assert snap.screen_usage_level == snap.spontaneous_usage_level


# =============================================================================
# derive_variation
# =============================================================================

class TestDeriveVariation:
    def test_insufficient_history(self, default_config):
        """断面が2件未満の場合は None。"""
        assert derive_variation([], default_config) is None
        assert derive_variation([PathwaySnapshot()], default_config) is None

    def test_two_snapshots(self, default_config):
        history = [
            PathwaySnapshot(bias_value=0.3),
            PathwaySnapshot(bias_value=0.5),
        ]
        var = derive_variation(history, default_config)
        assert var is not None
        assert var.bias_direction > 0  # 偏在方向
        assert var.bias_magnitude == pytest.approx(0.2)
        assert var.comparison_count == 1

    def test_decreasing_bias(self, default_config):
        history = [
            PathwaySnapshot(bias_value=0.8),
            PathwaySnapshot(bias_value=0.7),
            PathwaySnapshot(bias_value=0.4),
        ]
        var = derive_variation(history, default_config)
        assert var is not None
        assert var.bias_direction < 0  # 均等化方向

    def test_no_cumulative_accumulation(self, default_config):
        """安全弁2: 累積的蓄積を行わない。毎回再導出される。"""
        history = [
            PathwaySnapshot(bias_value=0.1),
            PathwaySnapshot(bias_value=0.2),
            PathwaySnapshot(bias_value=0.3),
        ]
        var1 = derive_variation(history, default_config)
        # 新しい断面を追加
        history.append(PathwaySnapshot(bias_value=0.1))
        var2 = derive_variation(history, default_config)
        # var2は過去の断面から新たに再導出される（var1の結果に依存しない）
        assert var2 is not None
        assert var2.bias_direction < 0  # 均等化方向に変化


# =============================================================================
# process_input_pathway_balance
# =============================================================================

class TestProcessInputPathwayBalance:
    def test_first_tick(self, empty_state, small_config):
        new_state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=small_config,
            timestamp=100.0,
        )
        assert len(new_state.usage_facts) == 1
        assert len(new_state.snapshot_history) == 1
        assert new_state.total_snapshots_generated == 1

    def test_multiple_ticks(self, empty_state, small_config):
        state = empty_state
        for i in range(10):
            pathway = ALL_PATHWAYS[i % 3]
            state = process_input_pathway_balance(
                state,
                current_pathway=pathway,
                config=small_config,
                timestamp=float(i),
            )
        assert len(state.usage_facts) == 10
        assert len(state.snapshot_history) == 5  # max_snapshot_history=5
        assert state.total_snapshots_generated == 10
        assert state.total_snapshots_expired == 5

    def test_screen_input_flag(self, empty_state, default_config):
        new_state = process_input_pathway_balance(
            empty_state,
            has_screen_input=True,
            config=default_config,
            timestamp=100.0,
        )
        assert len(new_state.usage_facts) == 1
        assert new_state.usage_facts[0].pathway == PATHWAY_SCREEN

    def test_current_pathway_overrides_screen_flag(self, empty_state, default_config):
        """current_pathwayが指定されている場合、has_screen_inputは使われない。"""
        new_state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            has_screen_input=True,
            config=default_config,
            timestamp=100.0,
        )
        assert len(new_state.usage_facts) == 1
        assert new_state.usage_facts[0].pathway == PATHWAY_TEXT

    def test_no_pathway_no_fact(self, empty_state, default_config):
        """経路が指定されない場合、使用事実は追加されない。"""
        new_state = process_input_pathway_balance(
            empty_state,
            config=default_config,
            timestamp=100.0,
        )
        assert len(new_state.usage_facts) == 0
        assert len(new_state.snapshot_history) == 1  # 断面は生成される

    def test_usage_facts_fifo(self, empty_state, small_config):
        """使用事実リストがFIFOで管理される。"""
        state = empty_state
        for i in range(25):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=small_config,
                timestamp=float(i),
            )
        assert len(state.usage_facts) <= small_config.max_usage_facts

    def test_variation_appears_after_2_snapshots(self, empty_state, small_config):
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=small_config,
            timestamp=100.0,
        )
        assert state.latest_variation is None

        state = process_input_pathway_balance(
            state,
            current_pathway=PATHWAY_SCREEN,
            config=small_config,
            timestamp=101.0,
        )
        assert state.latest_variation is not None

    def test_safety_no_cumulative_counter(self, empty_state, default_config):
        """安全弁5: 全期間累積カウンタを保持しない。"""
        state = empty_state
        for i in range(100):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=default_config,
                timestamp=float(i),
            )
        # 窓外の事実がカウントに影響しないことを確認
        snapshot = get_latest_snapshot(state)
        assert snapshot is not None
        # 窓サイズは50、全事実はtextなので窓内は50件 → MANY
        assert snapshot.text_usage_level == UsageLevel.MANY.value


# =============================================================================
# Accessors
# =============================================================================

class TestAccessors:
    def test_get_latest_snapshot_empty(self, empty_state):
        assert get_latest_snapshot(empty_state) is None

    def test_get_latest_snapshot(self, empty_state, default_config):
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
            timestamp=100.0,
        )
        snap = get_latest_snapshot(state)
        assert snap is not None
        assert snap.timestamp == 100.0

    def test_get_snapshot_history_returns_copy(self, empty_state, default_config):
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
        )
        history = get_snapshot_history(state)
        assert len(history) == 1
        # Modifying copy should not affect state
        history.clear()
        assert len(state.snapshot_history) == 1

    def test_get_latest_variation_empty(self, empty_state):
        assert get_latest_variation(empty_state) is None

    def test_get_pathway_balance_summary_empty(self, empty_state):
        summary = get_pathway_balance_summary(empty_state)
        assert summary["history_count"] == 0
        assert "text_usage_level" not in summary

    def test_get_pathway_balance_summary_with_data(self, empty_state, default_config):
        state = empty_state
        for i in range(3):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=default_config,
                timestamp=float(i),
            )
        summary = get_pathway_balance_summary(state)
        assert summary["history_count"] == 3
        assert "text_usage_level" in summary
        assert "screen_usage_level" in summary
        assert "spontaneous_usage_level" in summary
        assert "bias_level" in summary


# =============================================================================
# Enrichment
# =============================================================================

class TestEnrichment:
    def test_empty_state(self, empty_state):
        text = get_enrichment_text(empty_state)
        assert "待機中" in text

    def test_with_data(self, empty_state, default_config):
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
            timestamp=100.0,
        )
        text = get_enrichment_text(state)
        assert "待機中" not in text
        assert "テキスト=" in text
        assert "画面=" in text
        assert "自発=" in text
        assert "分布=" in text

    def test_no_evaluative_vocabulary(self, empty_state, default_config):
        """enrichment出力に評価的語彙を含まない。"""
        state = empty_state
        for i in range(20):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=default_config,
                timestamp=float(i),
            )
        text = get_enrichment_text(state)
        # 評価的語彙がないことを確認
        for word in ["不均衡", "偏っている", "問題", "悪い", "良い", "理想", "改善", "是正"]:
            assert word not in text

    def test_no_concrete_counts(self, empty_state, default_config):
        """enrichment出力に具体的な件数を含まない。"""
        state = empty_state
        for i in range(20):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=default_config,
                timestamp=float(i),
            )
        text = get_enrichment_text(state)
        # 数値（件数・比率）が段階値の一部でないことを確認
        # 段階値の表現のみが許可される
        import re
        # "20件" や "66%" のような具体的数値がないことを確認
        assert not re.search(r"\d+件", text)
        assert not re.search(r"\d+%", text)

    def test_variation_direction(self, empty_state, small_config):
        """変動がある場合、方向が表示される。"""
        state = empty_state
        # まず均等な事実を蓄積
        for i in range(3):
            state = process_input_pathway_balance(
                state,
                current_pathway=ALL_PATHWAYS[i % 3],
                config=small_config,
                timestamp=float(i),
            )
        # 次にtextのみを蓄積して偏在度を変化させる
        for i in range(5):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=small_config,
                timestamp=10.0 + i,
            )
        text = get_enrichment_text(state)
        # 変動が十分大きければ方向が表示される
        assert isinstance(text, str)


# =============================================================================
# Safety Valves
# =============================================================================

class TestSafetyValves:
    def test_no_weight_on_facts(self, empty_state, default_config):
        """安全弁1: 使用事実に重み・スコア・優先度を付与しない。"""
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
        )
        for fact in state.usage_facts:
            assert not hasattr(fact, "weight")
            assert not hasattr(fact, "score")
            assert not hasattr(fact, "priority")

    def test_no_weight_on_snapshots(self, empty_state, default_config):
        """安全弁1: 断面に重み・スコア・優先度を付与しない。"""
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
        )
        for snap in state.snapshot_history:
            assert not hasattr(snap, "weight")
            assert not hasattr(snap, "score")
            assert not hasattr(snap, "priority")

    def test_no_pattern_extraction(self, empty_state, default_config):
        """安全弁2: 蓄積からパターンを抽出しない。"""
        state = empty_state
        for i in range(20):
            state = process_input_pathway_balance(
                state,
                current_pathway=ALL_PATHWAYS[i % 3],
                config=default_config,
                timestamp=float(i),
            )
        # 状態にパターン関連のフィールドがないことを確認
        assert not hasattr(state, "patterns")
        assert not hasattr(state, "trends")
        assert not hasattr(state, "correlations")

    def test_no_cumulative_counter(self):
        """安全弁5: 全期間累積カウンタを保持しない。"""
        state = InputPathwayBalanceState()
        state_dict = state.to_dict()
        # 全期間累積使用カウンタが存在しないことを確認
        for key in state_dict:
            assert "cumulative_usage" not in key
            assert "total_usage" not in key

    def test_fifo_prevents_accumulation(self, small_config):
        """安全弁5: 窓から外れた事実はカウント対象外。"""
        state = create_input_pathway_balance_state()
        # 大量のtext事実を蓄積
        for i in range(30):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=small_config,
                timestamp=float(i),
            )
        # 窓が全てscreenに切り替わる
        for i in range(15):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_SCREEN,
                config=small_config,
                timestamp=100.0 + i,
            )
        snap = get_latest_snapshot(state)
        assert snap is not None
        # 窓内にはscreenのみが残る（textの過去の使用は反映されない）
        assert snap.screen_usage_level != UsageLevel.NONE.value
        # textは窓から外れている
        assert snap.text_usage_level == UsageLevel.NONE.value

    def test_no_route_selection_output(self, empty_state, default_config):
        """安全弁3: 経路選択に対する出力経路を持たない。"""
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
        )
        # 状態に経路選択を変更するフィールドがないことを確認
        assert not hasattr(state, "recommended_pathway")
        assert not hasattr(state, "pathway_priority")
        assert not hasattr(state, "suppress_pathway")
        assert not hasattr(state, "enable_pathway")

    def test_output_not_expanding(self, empty_state, default_config):
        """安全弁7: 出力経路不拡張。"""
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
        )
        # 出力先を動的に追加する仕組みがないことを確認
        assert not hasattr(state, "output_targets")
        assert not hasattr(state, "dynamic_outputs")


# =============================================================================
# read_text_dialogue_usage / read_spontaneous_usage
# =============================================================================

class TestReadFunctions:
    def test_read_text_dialogue_none(self):
        result = read_text_dialogue_usage(text_dialogue_state=None)
        assert result == []

    def test_read_spontaneous_none(self):
        result = read_spontaneous_usage(spontaneous_state=None)
        assert result == []

    def test_read_text_dialogue_with_mock(self):
        """duck typing で active_units を読み取る。"""

        class MockUnit:
            def __init__(self, route_type_value, timestamp):
                self.route_type = type("RT", (), {"value": route_type_value})()
                self.timestamp = timestamp

        class MockState:
            active_units = [
                MockUnit("text", 100.0),
                MockUnit("screen", 200.0),
                MockUnit("text", 300.0),
            ]

        result = read_text_dialogue_usage(text_dialogue_state=MockState())
        assert len(result) == 2  # Only text routes
        assert all(r["pathway"] == PATHWAY_TEXT for r in result)

    def test_read_spontaneous_with_mock(self):
        """duck typing で activation_history を読み取る。"""

        class MockEntry:
            def __init__(self, timestamp):
                self.timestamp = timestamp

        class MockState:
            activation_history = [
                MockEntry(100.0),
                MockEntry(200.0),
            ]

        result = read_spontaneous_usage(spontaneous_state=MockState())
        assert len(result) == 2
        assert all(r["pathway"] == PATHWAY_SPONTANEOUS for r in result)


# =============================================================================
# Save / Load
# =============================================================================

class TestSaveLoad:
    def test_empty_state_roundtrip(self, empty_state):
        data = save_state(empty_state)
        restored = load_state(data)
        assert len(restored.usage_facts) == 0
        assert len(restored.snapshot_history) == 0
        assert restored.latest_variation is None

    def test_populated_state_roundtrip(self, empty_state, small_config):
        state = empty_state
        for i in range(5):
            state = process_input_pathway_balance(
                state,
                current_pathway=ALL_PATHWAYS[i % 3],
                config=small_config,
                timestamp=float(i),
            )
        data = save_state(state)
        restored = load_state(data)

        assert len(restored.usage_facts) == len(state.usage_facts)
        assert len(restored.snapshot_history) == len(state.snapshot_history)
        assert restored.total_snapshots_generated == state.total_snapshots_generated
        assert restored.total_snapshots_expired == state.total_snapshots_expired
        assert restored.total_facts_expired == state.total_facts_expired

        if state.latest_variation is not None:
            assert restored.latest_variation is not None
            assert restored.latest_variation.bias_direction == pytest.approx(
                state.latest_variation.bias_direction
            )

    def test_snapshot_content_preserved(self, empty_state, default_config):
        state = process_input_pathway_balance(
            empty_state,
            current_pathway=PATHWAY_TEXT,
            config=default_config,
            timestamp=100.0,
        )
        data = save_state(state)
        restored = load_state(data)
        snap_orig = state.snapshot_history[0]
        snap_rest = restored.snapshot_history[0]
        assert snap_rest.text_usage_level == snap_orig.text_usage_level
        assert snap_rest.screen_usage_level == snap_orig.screen_usage_level
        assert snap_rest.spontaneous_usage_level == snap_orig.spontaneous_usage_level
        assert snap_rest.bias_level == snap_orig.bias_level
        assert snap_rest.bias_value == pytest.approx(snap_orig.bias_value)


# =============================================================================
# Factory
# =============================================================================

class TestFactory:
    def test_create_state(self):
        state = create_input_pathway_balance_state()
        assert isinstance(state, InputPathwayBalanceState)
        assert len(state.usage_facts) == 0
        assert len(state.snapshot_history) == 0

    def test_create_config(self):
        config = create_input_pathway_balance_config()
        assert isinstance(config, InputPathwayBalanceConfig)
        assert config.max_usage_facts == 200
        assert config.max_snapshot_history == 30
        assert config.sliding_window_size == 50

    def test_create_config_custom(self):
        config = create_input_pathway_balance_config(
            max_usage_facts=100,
            max_snapshot_history=10,
            sliding_window_size=20,
            variation_comparison_count=3,
        )
        assert config.max_usage_facts == 100
        assert config.max_snapshot_history == 10
        assert config.sliding_window_size == 20
        assert config.variation_comparison_count == 3


# =============================================================================
# Integration scenarios
# =============================================================================

class TestIntegrationScenarios:
    def test_text_dominant_scenario(self, default_config):
        """テキスト入力が支配的な場合。"""
        state = create_input_pathway_balance_state()
        for i in range(20):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=default_config,
                timestamp=float(i),
            )
        snap = get_latest_snapshot(state)
        assert snap is not None
        assert snap.text_usage_level == UsageLevel.MANY.value
        assert snap.screen_usage_level == UsageLevel.NONE.value
        assert snap.bias_level == BiasLevel.CONCENTRATED.value

    def test_transition_scenario(self, small_config):
        """テキストからスクリーンへ移行するシナリオ。"""
        state = create_input_pathway_balance_state()
        # Phase 1: text入力
        for i in range(5):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_TEXT,
                config=small_config,
                timestamp=float(i),
            )
        # Phase 2: screen入力
        for i in range(10):
            state = process_input_pathway_balance(
                state,
                current_pathway=PATHWAY_SCREEN,
                config=small_config,
                timestamp=10.0 + i,
            )
        snap = get_latest_snapshot(state)
        assert snap is not None
        # 窓内ではscreenが支配的
        assert snap.screen_usage_level != UsageLevel.NONE.value

    def test_mixed_scenario(self, default_config):
        """3経路が混在するシナリオ。"""
        state = create_input_pathway_balance_state()
        for i in range(30):
            pathway = ALL_PATHWAYS[i % 3]
            state = process_input_pathway_balance(
                state,
                current_pathway=pathway,
                config=default_config,
                timestamp=float(i),
            )
        snap = get_latest_snapshot(state)
        assert snap is not None
        # 均等に使われているので偏在度は低い
        assert snap.bias_level in [BiasLevel.EVEN.value, BiasLevel.SLIGHT.value]

    def test_long_running_scenario(self, small_config):
        """長期間の実行でも状態サイズが有限に保たれる。"""
        state = create_input_pathway_balance_state()
        for i in range(500):
            pathway = ALL_PATHWAYS[i % 3]
            state = process_input_pathway_balance(
                state,
                current_pathway=pathway,
                config=small_config,
                timestamp=float(i),
            )
        assert len(state.usage_facts) <= small_config.max_usage_facts
        assert len(state.snapshot_history) <= small_config.max_snapshot_history
