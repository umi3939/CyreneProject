"""
tests/test_behavioral_diversity_description.py - 行動多様性の構造的記述テスト

design_behavioral_diversity_description.md に基づく包括的テスト。

テスト要件:
- 第1段読み取りテスト（2構造からのREAD-ONLY読み取り確認）
- 第2段断面構成テスト（3断面の段階値決定確認）
- 第3段蓄積テスト（FIFO上限超過時の押し出し確認）
- READ-ONLYアクセサテスト
- save/loadラウンドトリップテスト
- enrichment非露出テスト（get_enrichment_data が存在しないことの確認）
- 安全弁テスト（8種）
- 頻度情報の構造的不在テスト
- パターン抽出禁止テスト
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from psyche.behavioral_diversity_description import (
    # Config
    BehavioralDiversityConfig,
    # Enums
    TypeCountLevel,
    DispersionLevel,
    # Level determination
    determine_type_count_level,
    determine_dispersion_level,
    # Data structures
    DiversityRecord,
    BehavioralDiversityState,
    # Stage 1: Read
    read_section_key_types,
    read_policy_label_types,
    read_candidate_size_types,
    # Stage 2: Compose
    compose_record,
    # Stage 3: Accumulate
    accumulate_record,
    # Main process
    process_behavioral_diversity,
    # Accessors
    get_latest_record,
    get_record_history,
    get_diversity_summary,
    # Save/Load
    save_state,
    load_state,
    # Factory
    create_behavioral_diversity_state,
    create_behavioral_diversity_config,
)


# =============================================================================
# Mock structures for testing
# =============================================================================

@dataclass
class MockSectionDescription:
    """行動結果観測構造のSectionDescriptionのモック。"""
    section: str = ""
    description: str = ""
    value: float = 0.0


@dataclass
class MockResultDescription:
    """行動結果観測構造のResultDescriptionのモック。"""
    sections: list[MockSectionDescription] = field(default_factory=list)


@dataclass
class MockActionResultPair:
    """行動結果観測構造のActionResultPairのモック。"""
    pair_id: str = ""
    status: str = "active"
    result: Optional[MockResultDescription] = None
    freshness: float = 1.0


@dataclass
class MockActionResultObservationState:
    """行動結果観測構造のStateのモック。"""
    pairs: list[MockActionResultPair] = field(default_factory=list)


@dataclass
class MockActionResultProcessor:
    """行動結果観測構造のProcessorのモック。"""
    state: MockActionResultObservationState = field(
        default_factory=MockActionResultObservationState
    )


@dataclass
class MockSelectionRecord:
    """選択帰属構造のSelectionRecordのモック。"""
    record_id: str = ""
    selected_policy_label: str = ""
    candidate_labels: list[str] = field(default_factory=list)
    candidate_count: int = 0
    tick: int = 0


@dataclass
class MockSelectionAttributionState:
    """選択帰属構造のStateのモック。"""
    records: list[MockSelectionRecord] = field(default_factory=list)


@dataclass
class MockSelectionAttributionRecorder:
    """選択帰属構造のRecorderのモック。"""
    state: MockSelectionAttributionState = field(
        default_factory=MockSelectionAttributionState
    )


# =============================================================================
# Helper functions
# =============================================================================

def _make_ar_processor_with_sections(section_keys: list[str], status: str = "active") -> MockActionResultProcessor:
    """指定された断面キーを持つモックProcessorを作成する。"""
    pairs = []
    for key in section_keys:
        pair = MockActionResultPair(
            pair_id=f"pair_{key}",
            status=status,
            result=MockResultDescription(
                sections=[MockSectionDescription(section=key)]
            ),
        )
        pairs.append(pair)
    return MockActionResultProcessor(
        state=MockActionResultObservationState(pairs=pairs)
    )


def _make_sa_recorder_with_labels(
    labels: list[str],
    candidate_counts: Optional[list[int]] = None,
) -> MockSelectionAttributionRecorder:
    """指定されたラベルと候補数を持つモックRecorderを作成する。"""
    records = []
    for i, label in enumerate(labels):
        count = candidate_counts[i] if candidate_counts else 3
        records.append(MockSelectionRecord(
            record_id=f"rec_{i}",
            selected_policy_label=label,
            candidate_count=count,
            tick=i,
        ))
    return MockSelectionAttributionRecorder(
        state=MockSelectionAttributionState(records=records)
    )


# =============================================================================
# 段階値決定テスト
# =============================================================================

class TestTypeCountLevel:
    """種類数の段階値決定テスト。"""

    def test_zero_types(self):
        assert determine_type_count_level(0) == TypeCountLevel.LEVEL_0

    def test_1_to_5_types(self):
        for n in [1, 2, 3, 4, 5]:
            assert determine_type_count_level(n) == TypeCountLevel.LEVEL_1_5

    def test_6_to_10_types(self):
        for n in [6, 7, 8, 9, 10]:
            assert determine_type_count_level(n) == TypeCountLevel.LEVEL_6_10

    def test_11_to_15_types(self):
        for n in [11, 12, 13, 14, 15]:
            assert determine_type_count_level(n) == TypeCountLevel.LEVEL_11_15

    def test_16_plus_types(self):
        for n in [16, 20, 50, 100]:
            assert determine_type_count_level(n) == TypeCountLevel.LEVEL_16_PLUS

    def test_levels_are_equal_valued(self):
        """全段階値は等価であり、数値的な大小を持たない。"""
        # Enum値はすべて文字列であり、数値的順序を持たないことを確認
        for level in TypeCountLevel:
            assert isinstance(level.value, str)


class TestDispersionLevel:
    """分散度の段階値決定テスト。"""

    def test_zero_distinct_sizes(self):
        assert determine_dispersion_level(0) == DispersionLevel.EMPTY

    def test_one_distinct_size(self):
        assert determine_dispersion_level(1) == DispersionLevel.UNIFORM

    def test_two_distinct_sizes(self):
        assert determine_dispersion_level(2) == DispersionLevel.LOW

    def test_three_to_four_distinct_sizes(self):
        assert determine_dispersion_level(3) == DispersionLevel.MODERATE
        assert determine_dispersion_level(4) == DispersionLevel.MODERATE

    def test_five_plus_distinct_sizes(self):
        for n in [5, 6, 10, 20]:
            assert determine_dispersion_level(n) == DispersionLevel.HIGH

    def test_levels_are_equal_valued(self):
        """全段階値は等価であり、評価的含意を持たない。"""
        for level in DispersionLevel:
            assert isinstance(level.value, str)


# =============================================================================
# 第1段: 読み取りテスト
# =============================================================================

class TestReadSectionKeyTypes:
    """行動結果観測構造からの断面キー種別数読み取りテスト。"""

    def test_none_input(self):
        """入力がNoneの場合は0を返す。"""
        assert read_section_key_types(action_result_state=None) == 0

    def test_empty_pairs(self):
        """対が空の場合は0を返す。"""
        proc = MockActionResultProcessor()
        assert read_section_key_types(action_result_state=proc) == 0

    def test_single_section_key(self):
        """断面キーが1種類の場合。"""
        proc = _make_ar_processor_with_sections(["external_reaction"])
        assert read_section_key_types(action_result_state=proc) == 1

    def test_multiple_distinct_section_keys(self):
        """断面キーが複数種類の場合。"""
        proc = _make_ar_processor_with_sections([
            "external_reaction", "emotion_transition", "time_elapsed",
        ])
        assert read_section_key_types(action_result_state=proc) == 3

    def test_duplicate_section_keys_counted_as_one(self):
        """同じ断面キーが複数回出現しても種類数は1。
        安全弁6: 頻度情報の構造的不在。出現回数ではなく種類数のみ。
        """
        proc = _make_ar_processor_with_sections([
            "external_reaction", "external_reaction", "external_reaction",
        ])
        assert read_section_key_types(action_result_state=proc) == 1

    def test_invisible_pairs_excluded(self):
        """不可視化済みの対は読み取り対象外。"""
        proc = _make_ar_processor_with_sections(
            ["external_reaction"],
            status="invisible",
        )
        assert read_section_key_types(action_result_state=proc) == 0

    def test_buffered_pairs_excluded(self):
        """バッファ内の対は読み取り対象外。"""
        proc = _make_ar_processor_with_sections(
            ["external_reaction"],
            status="buffered",
        )
        assert read_section_key_types(action_result_state=proc) == 0

    def test_decaying_pairs_included(self):
        """減衰中の対は読み取り対象に含める（可視状態）。"""
        proc = _make_ar_processor_with_sections(
            ["external_reaction"],
            status="decaying",
        )
        assert read_section_key_types(action_result_state=proc) == 1

    def test_composed_pairs_included(self):
        """構成完了の対は読み取り対象に含める。"""
        proc = _make_ar_processor_with_sections(
            ["external_reaction"],
            status="composed",
        )
        assert read_section_key_types(action_result_state=proc) == 1

    def test_direct_state_object(self):
        """Processorではなく直接State オブジェクトを渡した場合。"""
        state = MockActionResultObservationState(
            pairs=[
                MockActionResultPair(
                    status="active",
                    result=MockResultDescription(
                        sections=[MockSectionDescription(section="test_key")]
                    ),
                )
            ]
        )
        assert read_section_key_types(action_result_state=state) == 1

    def test_multiple_sections_per_pair(self):
        """1つの対に複数の断面がある場合。"""
        proc = MockActionResultProcessor(
            state=MockActionResultObservationState(
                pairs=[
                    MockActionResultPair(
                        status="active",
                        result=MockResultDescription(
                            sections=[
                                MockSectionDescription(section="key_a"),
                                MockSectionDescription(section="key_b"),
                                MockSectionDescription(section="key_a"),  # 重複
                            ]
                        ),
                    )
                ]
            )
        )
        assert read_section_key_types(action_result_state=proc) == 2


class TestReadPolicyLabelTypes:
    """選択帰属構造からのポリシーラベル種別数読み取りテスト。"""

    def test_none_input(self):
        assert read_policy_label_types(selection_attribution_state=None) == 0

    def test_empty_records(self):
        rec = MockSelectionAttributionRecorder()
        assert read_policy_label_types(selection_attribution_state=rec) == 0

    def test_single_label(self):
        rec = _make_sa_recorder_with_labels(["policy_a"])
        assert read_policy_label_types(selection_attribution_state=rec) == 1

    def test_multiple_distinct_labels(self):
        rec = _make_sa_recorder_with_labels(["policy_a", "policy_b", "policy_c"])
        assert read_policy_label_types(selection_attribution_state=rec) == 3

    def test_duplicate_labels_counted_as_one(self):
        """安全弁6: 出現回数ではなく種類数のみ。"""
        rec = _make_sa_recorder_with_labels(["policy_a", "policy_a", "policy_a"])
        assert read_policy_label_types(selection_attribution_state=rec) == 1

    def test_empty_label_excluded(self):
        """空のラベルは除外される。"""
        rec = _make_sa_recorder_with_labels(["", "policy_a"])
        assert read_policy_label_types(selection_attribution_state=rec) == 1

    def test_direct_state_object(self):
        """直接Stateを渡した場合。"""
        state = MockSelectionAttributionState(
            records=[
                MockSelectionRecord(selected_policy_label="label_x"),
                MockSelectionRecord(selected_policy_label="label_y"),
            ]
        )
        assert read_policy_label_types(selection_attribution_state=state) == 2


class TestReadCandidateSizeTypes:
    """候補群サイズの異なる種類数読み取りテスト。"""

    def test_none_input(self):
        assert read_candidate_size_types(selection_attribution_state=None) == 0

    def test_empty_records(self):
        rec = MockSelectionAttributionRecorder()
        assert read_candidate_size_types(selection_attribution_state=rec) == 0

    def test_all_same_size(self):
        rec = _make_sa_recorder_with_labels(
            ["a", "b", "c"],
            candidate_counts=[5, 5, 5],
        )
        assert read_candidate_size_types(selection_attribution_state=rec) == 1

    def test_two_different_sizes(self):
        rec = _make_sa_recorder_with_labels(
            ["a", "b", "c"],
            candidate_counts=[5, 3, 5],
        )
        assert read_candidate_size_types(selection_attribution_state=rec) == 2

    def test_all_different_sizes(self):
        rec = _make_sa_recorder_with_labels(
            ["a", "b", "c", "d", "e"],
            candidate_counts=[1, 2, 3, 4, 5],
        )
        assert read_candidate_size_types(selection_attribution_state=rec) == 5


# =============================================================================
# 第2段: 断面構成テスト
# =============================================================================

class TestComposeRecord:
    """断面構成テスト。"""

    def test_all_zero(self):
        """全て0の場合のデフォルト段階値。"""
        record = compose_record(0, 0, 0, tick=1, timestamp=100.0)
        assert record.section_key_type_count_level == TypeCountLevel.LEVEL_0.value
        assert record.policy_label_type_count_level == TypeCountLevel.LEVEL_0.value
        assert record.candidate_size_dispersion_level == DispersionLevel.EMPTY.value
        assert record.tick == 1
        assert record.timestamp == 100.0

    def test_mixed_levels(self):
        """異なる段階値の組み合わせ。"""
        record = compose_record(3, 8, 4, tick=5, timestamp=200.0)
        assert record.section_key_type_count_level == TypeCountLevel.LEVEL_1_5.value
        assert record.policy_label_type_count_level == TypeCountLevel.LEVEL_6_10.value
        assert record.candidate_size_dispersion_level == DispersionLevel.MODERATE.value

    def test_high_values(self):
        """高い値の場合。"""
        record = compose_record(20, 20, 10, tick=10, timestamp=300.0)
        assert record.section_key_type_count_level == TypeCountLevel.LEVEL_16_PLUS.value
        assert record.policy_label_type_count_level == TypeCountLevel.LEVEL_16_PLUS.value
        assert record.candidate_size_dispersion_level == DispersionLevel.HIGH.value

    def test_timestamp_auto_generated(self):
        """timestampがNoneの場合は自動生成される。"""
        before = time.time()
        record = compose_record(0, 0, 0)
        after = time.time()
        assert before <= record.timestamp <= after

    def test_sections_are_independent(self):
        """3断面は独立しており、断面間の関係性を持たない。"""
        # 断面間の重み付け・比較・統合は行わない
        record = compose_record(5, 10, 2)
        # 各断面が独立して決定されていることを確認
        assert record.section_key_type_count_level == TypeCountLevel.LEVEL_1_5.value
        assert record.policy_label_type_count_level == TypeCountLevel.LEVEL_6_10.value
        assert record.candidate_size_dispersion_level == DispersionLevel.LOW.value


# =============================================================================
# 第3段: 蓄積テスト
# =============================================================================

class TestAccumulateRecord:
    """FIFO蓄積テスト。"""

    def test_basic_accumulation(self):
        """基本的な蓄積。"""
        state = create_behavioral_diversity_state()
        record = compose_record(3, 5, 2, tick=1, timestamp=100.0)
        new_state = accumulate_record(state, record)

        assert len(new_state.history) == 1
        assert new_state.latest_record is record
        assert new_state.total_records_generated == 1
        assert new_state.total_records_expired == 0

    def test_fifo_pushout(self):
        """上限超過時のFIFO押し出し。"""
        config = BehavioralDiversityConfig(max_history=3)
        state = create_behavioral_diversity_state()

        for i in range(5):
            record = compose_record(i, i, i, tick=i, timestamp=float(i))
            state = accumulate_record(state, record, config)

        assert len(state.history) == 3
        assert state.total_records_generated == 5
        assert state.total_records_expired == 2
        # 最古の2件が押し出されている
        assert state.history[0].tick == 2

    def test_latest_record_updated(self):
        """latest_recordが最新のものに更新される。"""
        state = create_behavioral_diversity_state()
        record1 = compose_record(1, 1, 1, tick=1)
        record2 = compose_record(2, 2, 2, tick=2)

        state = accumulate_record(state, record1)
        assert state.latest_record is record1

        state = accumulate_record(state, record2)
        assert state.latest_record is record2

    def test_no_protected_records(self):
        """安全弁1: 特定の記録を保護・固定・優先的に保持する機構がない。"""
        config = BehavioralDiversityConfig(max_history=2)
        state = create_behavioral_diversity_state()

        # 全ての記録は等価に押し出される
        for i in range(10):
            record = compose_record(i % 5, i % 3, i % 4, tick=i)
            state = accumulate_record(state, record, config)

        assert len(state.history) == 2
        # 最新の2件のみが残る
        assert state.history[0].tick == 8
        assert state.history[1].tick == 9


# =============================================================================
# メイン処理テスト
# =============================================================================

class TestProcessBehavioralDiversity:
    """process_behavioral_diversityのテスト。"""

    def test_with_no_input(self):
        """入力なしでも処理可能。"""
        state = create_behavioral_diversity_state()
        new_state = process_behavioral_diversity(state, tick=1)

        assert len(new_state.history) == 1
        assert new_state.latest_record is not None
        assert new_state.latest_record.section_key_type_count_level == TypeCountLevel.LEVEL_0.value
        assert new_state.latest_record.policy_label_type_count_level == TypeCountLevel.LEVEL_0.value
        assert new_state.latest_record.candidate_size_dispersion_level == DispersionLevel.EMPTY.value

    def test_with_action_result(self):
        """行動結果観測構造からの読み取り。"""
        state = create_behavioral_diversity_state()
        ar_proc = _make_ar_processor_with_sections([
            "external_reaction", "emotion_transition", "time_elapsed",
        ])

        new_state = process_behavioral_diversity(
            state,
            action_result_state=ar_proc,
            tick=1,
        )

        assert new_state.latest_record.section_key_type_count_level == TypeCountLevel.LEVEL_1_5.value

    def test_with_selection_attribution(self):
        """選択帰属構造からの読み取り。"""
        state = create_behavioral_diversity_state()
        sa_rec = _make_sa_recorder_with_labels(
            ["policy_a", "policy_b"],
            candidate_counts=[3, 5],
        )

        new_state = process_behavioral_diversity(
            state,
            selection_attribution_state=sa_rec,
            tick=1,
        )

        assert new_state.latest_record.policy_label_type_count_level == TypeCountLevel.LEVEL_1_5.value
        assert new_state.latest_record.candidate_size_dispersion_level == DispersionLevel.LOW.value

    def test_with_both_inputs(self):
        """両方の入力構造からの読み取り。"""
        state = create_behavioral_diversity_state()
        ar_proc = _make_ar_processor_with_sections([
            "a", "b", "c", "d", "e", "f", "g",
        ])
        sa_rec = _make_sa_recorder_with_labels(
            ["x", "y", "z"],
            candidate_counts=[1, 1, 1],
        )

        new_state = process_behavioral_diversity(
            state,
            action_result_state=ar_proc,
            selection_attribution_state=sa_rec,
            tick=1,
        )

        assert new_state.latest_record.section_key_type_count_level == TypeCountLevel.LEVEL_6_10.value
        assert new_state.latest_record.policy_label_type_count_level == TypeCountLevel.LEVEL_1_5.value
        assert new_state.latest_record.candidate_size_dispersion_level == DispersionLevel.UNIFORM.value

    def test_no_cumulative_dependency(self):
        """段階値の計算は、その時点の読み取り値のみに依存する。
        過去の計算結果が現在の計算に影響を与える累積構造を持たない。
        """
        state = create_behavioral_diversity_state()

        # 最初の処理: 3種類
        ar1 = _make_ar_processor_with_sections(["a", "b", "c"])
        state = process_behavioral_diversity(state, action_result_state=ar1, tick=1)
        assert state.latest_record.section_key_type_count_level == TypeCountLevel.LEVEL_1_5.value

        # 2回目の処理: 0種類（過去の結果に依存しない）
        state = process_behavioral_diversity(state, tick=2)
        assert state.latest_record.section_key_type_count_level == TypeCountLevel.LEVEL_0.value

    def test_multiple_cycles(self):
        """複数サイクルの蓄積。"""
        state = create_behavioral_diversity_state()
        config = BehavioralDiversityConfig(max_history=5)

        for i in range(10):
            state = process_behavioral_diversity(
                state,
                tick=i,
                config=config,
            )

        assert len(state.history) == 5
        assert state.total_records_generated == 10
        assert state.total_records_expired == 5


# =============================================================================
# READ-ONLYアクセサテスト
# =============================================================================

class TestAccessors:
    """内省系参照経路テスト。"""

    def test_get_latest_record_empty(self):
        state = create_behavioral_diversity_state()
        assert get_latest_record(state) is None

    def test_get_latest_record_with_data(self):
        state = create_behavioral_diversity_state()
        state = process_behavioral_diversity(state, tick=1)
        latest = get_latest_record(state)
        assert latest is not None
        assert latest.tick == 1

    def test_get_record_history_empty(self):
        state = create_behavioral_diversity_state()
        history = get_record_history(state)
        assert history == []

    def test_get_record_history_returns_copy(self):
        """蓄積リストのコピーを返す（READ-ONLY）。"""
        state = create_behavioral_diversity_state()
        state = process_behavioral_diversity(state, tick=1)
        history = get_record_history(state)
        assert len(history) == 1

        # コピーを変更しても元の状態に影響しない
        history.clear()
        assert len(state.history) == 1

    def test_get_diversity_summary_empty(self):
        state = create_behavioral_diversity_state()
        summary = get_diversity_summary(state)
        assert summary["history_count"] == 0
        assert summary["total_generated"] == 0
        assert "latest_tick" not in summary

    def test_get_diversity_summary_with_data(self):
        state = create_behavioral_diversity_state()
        state = process_behavioral_diversity(state, tick=5, timestamp=1234.0)
        summary = get_diversity_summary(state)
        assert summary["history_count"] == 1
        assert summary["total_generated"] == 1
        assert summary["latest_tick"] == 5
        assert "latest_section_key_type_count_level" in summary
        assert "latest_policy_label_type_count_level" in summary
        assert "latest_candidate_size_dispersion_level" in summary


# =============================================================================
# Save/Load ラウンドトリップテスト
# =============================================================================

class TestSaveLoad:
    """永続化テスト。"""

    def test_empty_state_roundtrip(self):
        """空の状態のラウンドトリップ。"""
        state = create_behavioral_diversity_state()
        data = save_state(state)
        restored = load_state(data)
        assert len(restored.history) == 0
        assert restored.latest_record is None
        assert restored.total_records_generated == 0

    def test_populated_state_roundtrip(self):
        """データを含む状態のラウンドトリップ。"""
        state = create_behavioral_diversity_state()
        for i in range(5):
            state = process_behavioral_diversity(state, tick=i, timestamp=float(i * 100))

        data = save_state(state)
        restored = load_state(data)

        assert len(restored.history) == len(state.history)
        assert restored.total_records_generated == state.total_records_generated
        assert restored.total_records_expired == state.total_records_expired
        assert restored.latest_record is not None
        assert restored.latest_record.tick == state.latest_record.tick
        assert restored.latest_record.section_key_type_count_level == state.latest_record.section_key_type_count_level

    def test_record_to_dict_from_dict(self):
        """DiversityRecordのto_dict/from_dict。"""
        record = compose_record(7, 12, 3, tick=42, timestamp=999.0)
        data = record.to_dict()
        restored = DiversityRecord.from_dict(data)

        assert restored.section_key_type_count_level == record.section_key_type_count_level
        assert restored.policy_label_type_count_level == record.policy_label_type_count_level
        assert restored.candidate_size_dispersion_level == record.candidate_size_dispersion_level
        assert restored.tick == record.tick
        assert restored.timestamp == record.timestamp

    def test_state_to_dict_from_dict(self):
        """BehavioralDiversityStateのto_dict/from_dict。"""
        state = create_behavioral_diversity_state()
        state = process_behavioral_diversity(state, tick=1)
        state = process_behavioral_diversity(state, tick=2)

        data = state.to_dict()
        restored = BehavioralDiversityState.from_dict(data)

        assert len(restored.history) == 2
        assert restored.total_records_generated == 2


# =============================================================================
# enrichment 非露出テスト
# =============================================================================

class TestEnrichmentBlocking:
    """安全弁3: enrichment直接露出遮断テスト。"""

    def test_no_get_enrichment_data_method(self):
        """get_enrichment_data メソッドが存在しないこと。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_enrichment_data")

    def test_no_enrichment_in_summary(self):
        """サマリにenrichmentへの直接露出情報が含まれないこと。"""
        state = create_behavioral_diversity_state()
        state = process_behavioral_diversity(state, tick=1)
        summary = get_diversity_summary(state)
        # サマリは内省系向けであり、enrichmentキーを含まない
        assert "enrichment" not in str(summary).lower() or "enrichment" not in summary


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    """8種の安全弁テスト。"""

    def test_sv1_all_records_equal(self):
        """安全弁1: 全記録等価。
        蓄積リスト内の全断面記録に重み・スコア・優先度を付与しない。
        """
        state = create_behavioral_diversity_state()
        for i in range(5):
            state = process_behavioral_diversity(state, tick=i)

        for record in state.history:
            # 重み・スコア・優先度のフィールドが存在しないこと
            assert not hasattr(record, "weight")
            assert not hasattr(record, "score")
            assert not hasattr(record, "priority")
            assert not hasattr(record, "importance")

    def test_sv2_no_pattern_extraction(self):
        """安全弁2: パターン抽出禁止。
        蓄積された記録から傾向、周期性、統計量を算出しない。
        """
        import psyche.behavioral_diversity_description as mod
        # モジュールにパターン抽出関数が存在しないこと
        assert not hasattr(mod, "detect_pattern")
        assert not hasattr(mod, "compute_trend")
        assert not hasattr(mod, "compute_statistics")
        assert not hasattr(mod, "detect_periodicity")
        assert not hasattr(mod, "derive_variation")
        assert not hasattr(mod, "compute_correlation")

    def test_sv3_no_enrichment_exposure(self):
        """安全弁3: enrichment直接露出遮断。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_enrichment_data")

    def test_sv4_no_forgetting_pathway(self):
        """安全弁4: 忘却経路遮断。
        本機能の出力を記憶の忘却・固定化処理の入力に使用しない。
        """
        import psyche.behavioral_diversity_description as mod
        # 忘却関連の関数や出力経路がないこと
        assert not hasattr(mod, "get_forgetting_input")
        assert not hasattr(mod, "supply_to_forgetting")

    def test_sv5_no_recall_pathway(self):
        """安全弁5: 想起経路遮断。
        本機能の出力を記憶の想起候補の選択に影響させない。
        """
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_recall_influence")
        assert not hasattr(mod, "supply_to_recall")

    def test_sv6_no_frequency_info(self):
        """安全弁6: 頻度情報の構造的不在。
        個別の断面キーやポリシーラベルの出現回数を算出・保持しない。
        """
        state = create_behavioral_diversity_state()
        ar_proc = _make_ar_processor_with_sections([
            "a", "a", "b", "b", "b", "c",
        ])
        sa_rec = _make_sa_recorder_with_labels([
            "x", "x", "y", "z", "z", "z",
        ])

        state = process_behavioral_diversity(
            state,
            action_result_state=ar_proc,
            selection_attribution_state=sa_rec,
            tick=1,
        )

        record = state.latest_record
        assert record is not None

        # 記録に出現回数の情報が含まれていないこと
        assert not hasattr(record, "frequency")
        assert not hasattr(record, "counts")
        assert not hasattr(record, "occurrences")
        record_dict = record.to_dict()
        for key in record_dict:
            assert "count" not in key.lower() or key == "section_key_type_count_level" or key == "policy_label_type_count_level"
            assert "frequency" not in key.lower()

    def test_sv7_no_output_path_extension(self):
        """安全弁7: 出力経路不拡張。
        初期実装で定義した出力経路以外の経路がないこと。
        """
        import psyche.behavioral_diversity_description as mod
        # 出力経路は get_latest_record, get_record_history, get_diversity_summary のみ
        output_functions = [
            name for name in dir(mod)
            if name.startswith("get_") and callable(getattr(mod, name))
        ]
        expected_outputs = {"get_latest_record", "get_record_history", "get_diversity_summary"}
        actual_outputs = set(output_functions)
        # 想定外の出力経路がないこと
        unexpected = actual_outputs - expected_outputs
        assert not unexpected, f"Unexpected output paths: {unexpected}"

    def test_sv8_no_existing_safety_valve_weakening(self):
        """安全弁8: 既存モジュール安全弁の維持保証。
        本機能はREAD-ONLYであり、既存構造の状態を変更しない。
        """
        # 行動結果観測構造のモックを作成
        pairs = [
            MockActionResultPair(
                pair_id="p1",
                status="active",
                freshness=0.8,
                result=MockResultDescription(
                    sections=[MockSectionDescription(section="test")]
                ),
            )
        ]
        ar_state = MockActionResultObservationState(pairs=pairs)
        ar_proc = MockActionResultProcessor(state=ar_state)

        # 選択帰属構造のモックを作成
        sa_records = [MockSelectionRecord(
            selected_policy_label="policy_test",
            candidate_count=5,
        )]
        sa_state = MockSelectionAttributionState(records=sa_records)
        sa_rec = MockSelectionAttributionRecorder(state=sa_state)

        # 処理前の状態を記録
        original_pair_status = ar_proc.state.pairs[0].status
        original_pair_freshness = ar_proc.state.pairs[0].freshness
        original_label = sa_rec.state.records[0].selected_policy_label
        original_count = sa_rec.state.records[0].candidate_count

        # 処理実行
        state = create_behavioral_diversity_state()
        process_behavioral_diversity(
            state,
            action_result_state=ar_proc,
            selection_attribution_state=sa_rec,
            tick=1,
        )

        # 既存構造の状態が変更されていないこと
        assert ar_proc.state.pairs[0].status == original_pair_status
        assert ar_proc.state.pairs[0].freshness == original_pair_freshness
        assert sa_rec.state.records[0].selected_policy_label == original_label
        assert sa_rec.state.records[0].candidate_count == original_count


# =============================================================================
# 評価的含意の不在テスト
# =============================================================================

class TestNoEvaluativeImplication:
    """多様性に方向性を与えない。全段階値が等価。"""

    def test_level_values_are_non_numeric(self):
        """段階値の値が数値的な大小を持たない文字列であること。"""
        for level in TypeCountLevel:
            assert isinstance(level.value, str)
            # 数値文字列でないこと
            assert not level.value.isdigit()

        for level in DispersionLevel:
            assert isinstance(level.value, str)
            assert not level.value.isdigit()

    def test_no_evaluative_words_in_enum_values(self):
        """段階値の名称に評価的含意を含めない。"""
        evaluative_words = [
            "good", "bad", "better", "worse", "optimal", "ideal",
            "poor", "excellent", "desired", "undesired",
        ]
        for level in TypeCountLevel:
            for word in evaluative_words:
                assert word not in level.value.lower()
                assert word not in level.name.lower()

        for level in DispersionLevel:
            for word in evaluative_words:
                assert word not in level.value.lower()
                assert word not in level.name.lower()

    def test_no_threshold_evaluation(self):
        """安全弁: 閾値による評価的カテゴリ化を行わない。
        段階値は列挙型として等価に並置されるのみ。
        「高多様性」「低多様性」のような二値的・評価的なラベル変換を行わない。
        """
        import psyche.behavioral_diversity_description as mod
        # 評価的カテゴリ化関数がないこと
        assert not hasattr(mod, "is_high_diversity")
        assert not hasattr(mod, "is_low_diversity")
        assert not hasattr(mod, "evaluate_diversity")
        assert not hasattr(mod, "assess_diversity")


# =============================================================================
# 出力経路遮断テスト
# =============================================================================

class TestOutputPathBlocking:
    """判断・行動・責任システムとの構造的分離テスト。"""

    def test_no_policy_selection_output(self):
        """ポリシー選択への出力経路を持たない。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_policy_bias")
        assert not hasattr(mod, "compute_bias")
        assert not hasattr(mod, "apply_to_candidates")

    def test_no_emotion_pipeline_output(self):
        """感情パイプラインへの出力経路を持たない。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_emotion_influence")
        assert not hasattr(mod, "apply_to_emotion")

    def test_no_responsibility_output(self):
        """責任計算への出力経路を持たない。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_responsibility_input")
        assert not hasattr(mod, "supply_to_responsibility")

    def test_no_tendency_output(self):
        """反復傾向構造への出力経路を持たない。"""
        import psyche.behavioral_diversity_description as mod
        assert not hasattr(mod, "get_tendency_input")
        assert not hasattr(mod, "supply_to_tendency")


# =============================================================================
# Factory テスト
# =============================================================================

class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_state(self):
        state = create_behavioral_diversity_state()
        assert isinstance(state, BehavioralDiversityState)
        assert len(state.history) == 0
        assert state.latest_record is None
        assert state.total_records_generated == 0

    def test_create_config_defaults(self):
        config = create_behavioral_diversity_config()
        assert config.max_history == 30

    def test_create_config_custom(self):
        config = create_behavioral_diversity_config(max_history=50)
        assert config.max_history == 50
