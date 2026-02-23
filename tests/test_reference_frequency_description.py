"""
tests/test_reference_frequency_description.py - 参照頻度の構造的記述モジュールのテスト

テスト対象: psyche/reference_frequency_description.py

安全弁の検証を含む:
  1. 全記録等価維持保証
  2. 評価的変換の禁止
  3. 累積的傾向の抑制
  4. 断面履歴の有限性
  5. 出力経路の限定と不拡張
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from psyche.reference_frequency_description import (
    ALL_STRUCTURE_KEYS,
    STRUCTURE_ACTION_RESULT,
    STRUCTURE_BINDING,
    STRUCTURE_BINDING_TRACE,
    STRUCTURE_DIALOGUE_LEARNING,
    STRUCTURE_EPISODIC,
    STRUCTURE_EXPECTATION,
    STRUCTURE_FORGETTING,
    STRUCTURE_INTROSPECTION,
    STRUCTURE_MOTIVE_ENTRY,
    STRUCTURE_MOTIVE_IMPULSE,
    STRUCTURE_MULTI_PATH_RECALL,
    STRUCTURE_NARRATIVE,
    STRUCTURE_OTHER_MODEL,
    STRUCTURE_SELF_REFERENCE,
    STRUCTURE_SPONTANEOUS_RECALL,
    ReferenceFrequencyConfig,
    ReferenceFrequencyState,
    ReferenceSnapshot,
    VariationDescription,
    collect_reference_counts,
    compose_snapshot,
    compute_concentration,
    compute_structural_bias,
    create_reference_frequency_config,
    create_reference_frequency_state,
    derive_variation,
    get_latest_snapshot,
    get_latest_variation,
    get_reference_summary,
    get_snapshot_history,
    load_state,
    process_reference_frequency,
    save_state,
)


# =============================================================================
# Mock Data Structures (duck typing)
# =============================================================================

@dataclass(frozen=True)
class MockEpisodeEntry:
    reference_count: int = 0


@dataclass(frozen=True)
class MockEpisodeStore:
    episodes: tuple = ()


@dataclass(frozen=True)
class MockEmotionalTrace:
    reference_count: int = 0


@dataclass(frozen=True)
class MockMemoryBinding:
    reference_count: int = 0
    traces: tuple = ()


@dataclass(frozen=True)
class MockBindingStore:
    bindings: tuple = ()


@dataclass(frozen=True)
class MockIntrospectionFragment:
    reference_count: int = 0


@dataclass(frozen=True)
class MockConsumptionStore:
    fragments: tuple = ()


@dataclass(frozen=True)
class MockExpectationCandidate:
    reference_count: int = 0


@dataclass(frozen=True)
class MockExpectationStore:
    expectations: tuple = ()


@dataclass(frozen=True)
class MockMotiveImpulse:
    reference_count: int = 0


@dataclass(frozen=True)
class MockMotiveEntry:
    reference_count: int = 0
    impulses: tuple = ()


@dataclass(frozen=True)
class MockMotiveStore:
    entries: tuple = ()


@dataclass(frozen=True)
class MockNarrativeFragment:
    reference_count: int = 0


@dataclass(frozen=True)
class MockNarrativeState:
    fragments: tuple = ()


@dataclass(frozen=True)
class MockHypothesis:
    reference_count: int = 0


@dataclass(frozen=True)
class MockOtherModelStore:
    hypotheses: tuple = ()


@dataclass
class MockSelfReferenceState:
    reference_count: int = 0


@dataclass
class MockActionResultPair:
    reference_count: int = 0


@dataclass
class MockActionResultState:
    pairs: list = field(default_factory=list)


@dataclass
class MockDialogueLearningEntry:
    reference_count: int = 0


@dataclass
class MockDialogueLearningState:
    entries: list = field(default_factory=list)


@dataclass
class MockMemorySeriesRecord:
    reference_count: int = 0


@dataclass
class MockForgettingState:
    series_index: list = field(default_factory=list)


@dataclass(frozen=True)
class MockPathStatistics:
    """multi_path_recall の PathStatistics 相当のモック。"""
    emotional_count: int = 0
    contextual_count: int = 0
    temporal_count: int = 0


@dataclass
class MockMultiPathRecallState:
    """multi_path_recall の state 相当のモック。"""
    path_stats: Optional[MockPathStatistics] = None


@dataclass(frozen=True)
class MockSpontaneousRecallPathStatistics:
    """spontaneous_recall の SpontaneousRecallPathStatistics 相当のモック。"""
    emotion_delta_count: int = 0
    motive_assoc_count: int = 0
    fluctuation_assoc_count: int = 0


@dataclass
class MockSpontaneousRecallState:
    """spontaneous_recall の state 相当のモック。"""
    path_stats: Optional[MockSpontaneousRecallPathStatistics] = None


# =============================================================================
# Helper: 全構造のモックを作成
# =============================================================================

def _make_full_mocks(
    ep_counts: list[int] | None = None,
    bind_counts: list[int] | None = None,
    trace_counts: list[list[int]] | None = None,
    intro_counts: list[int] | None = None,
    expect_counts: list[int] | None = None,
    motive_counts: list[int] | None = None,
    impulse_counts: list[list[int]] | None = None,
    narr_counts: list[int] | None = None,
    other_counts: list[int] | None = None,
    self_ref_count: int = 0,
    action_counts: list[int] | None = None,
    dialogue_counts: list[int] | None = None,
    forgetting_counts: list[int] | None = None,
    multi_path_recall_counts: tuple[int, int, int] | None = None,
    spontaneous_recall_counts: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """テスト用のモックオブジェクト群を作成する。"""
    # Episodic
    ep = MockEpisodeStore(
        episodes=tuple(MockEpisodeEntry(c) for c in (ep_counts or []))
    )

    # Binding with traces
    if bind_counts is not None:
        bindings = []
        for i, bc in enumerate(bind_counts):
            tc = (trace_counts or [[]])[i] if trace_counts and i < len(trace_counts) else []
            bindings.append(MockMemoryBinding(
                reference_count=bc,
                traces=tuple(MockEmotionalTrace(t) for t in tc),
            ))
        bind = MockBindingStore(bindings=tuple(bindings))
    else:
        bind = MockBindingStore()

    # Consumption
    intro = MockConsumptionStore(
        fragments=tuple(MockIntrospectionFragment(c) for c in (intro_counts or []))
    )

    # Expectation
    expect = MockExpectationStore(
        expectations=tuple(MockExpectationCandidate(c) for c in (expect_counts or []))
    )

    # Motive
    if motive_counts is not None:
        entries = []
        for i, mc in enumerate(motive_counts):
            ic = (impulse_counts or [[]])[i] if impulse_counts and i < len(impulse_counts) else []
            entries.append(MockMotiveEntry(
                reference_count=mc,
                impulses=tuple(MockMotiveImpulse(imp) for imp in ic),
            ))
        motive = MockMotiveStore(entries=tuple(entries))
    else:
        motive = MockMotiveStore()

    # Narrative
    narr = MockNarrativeState(
        fragments=tuple(MockNarrativeFragment(c) for c in (narr_counts or []))
    )

    # Other Model
    other = MockOtherModelStore(
        hypotheses=tuple(MockHypothesis(c) for c in (other_counts or []))
    )

    # Self Reference
    self_ref = MockSelfReferenceState(reference_count=self_ref_count)

    # Action Result
    action = MockActionResultState(
        pairs=[MockActionResultPair(c) for c in (action_counts or [])]
    )

    # Dialogue Learning
    dialogue = MockDialogueLearningState(
        entries=[MockDialogueLearningEntry(c) for c in (dialogue_counts or [])]
    )

    # Forgetting
    forgetting = MockForgettingState(
        series_index=[MockMemorySeriesRecord(c) for c in (forgetting_counts or [])]
    )

    # Multi Path Recall
    if multi_path_recall_counts is not None:
        mpr = MockMultiPathRecallState(
            path_stats=MockPathStatistics(
                emotional_count=multi_path_recall_counts[0],
                contextual_count=multi_path_recall_counts[1],
                temporal_count=multi_path_recall_counts[2],
            )
        )
    else:
        mpr = None

    # Spontaneous Recall
    if spontaneous_recall_counts is not None:
        sr = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(
                emotion_delta_count=spontaneous_recall_counts[0],
                motive_assoc_count=spontaneous_recall_counts[1],
                fluctuation_assoc_count=spontaneous_recall_counts[2],
            )
        )
    else:
        sr = None

    return {
        "episodic_store": ep,
        "binding_store": bind,
        "consumption_store": intro,
        "expectation_store": expect,
        "motive_store": motive,
        "narrative_state": narr,
        "other_model_store": other,
        "self_reference_state": self_ref,
        "action_result_state": action,
        "dialogue_learning_state": dialogue,
        "forgetting_state": forgetting,
        "multi_path_recall_state": mpr,
        "spontaneous_recall_state": sr,
    }


# =============================================================================
# Tests: Config
# =============================================================================

class TestConfig:
    def test_default_config(self):
        cfg = ReferenceFrequencyConfig()
        assert cfg.max_snapshot_history == 30
        assert cfg.variation_comparison_count == 5

    def test_custom_config(self):
        cfg = create_reference_frequency_config(
            max_snapshot_history=10,
            variation_comparison_count=3,
        )
        assert cfg.max_snapshot_history == 10
        assert cfg.variation_comparison_count == 3


# =============================================================================
# Tests: Data Structures
# =============================================================================

class TestReferenceSnapshot:
    def test_default(self):
        snap = ReferenceSnapshot()
        assert snap.timestamp == 0.0
        assert snap.structure_counts == {}
        assert snap.concentration == 0.0
        assert snap.structural_bias == 0.0

    def test_to_dict_from_dict(self):
        snap = ReferenceSnapshot(
            timestamp=100.0,
            structure_counts={STRUCTURE_EPISODIC: [1, 2, 3]},
            concentration=0.5,
            structural_bias=0.3,
        )
        d = snap.to_dict()
        restored = ReferenceSnapshot.from_dict(d)
        assert restored.timestamp == 100.0
        assert restored.structure_counts[STRUCTURE_EPISODIC] == [1, 2, 3]
        assert restored.concentration == 0.5
        assert restored.structural_bias == 0.3

    def test_from_dict_empty(self):
        restored = ReferenceSnapshot.from_dict({})
        assert restored.timestamp == 0.0
        assert restored.structure_counts == {}


class TestVariationDescription:
    def test_default(self):
        var = VariationDescription()
        assert var.concentration_direction == 0.0
        assert var.concentration_magnitude == 0.0
        assert var.bias_direction == 0.0
        assert var.bias_magnitude == 0.0
        assert var.comparison_count == 0

    def test_to_dict_from_dict(self):
        var = VariationDescription(
            concentration_direction=0.1,
            concentration_magnitude=0.1,
            bias_direction=-0.05,
            bias_magnitude=0.05,
            comparison_count=3,
        )
        d = var.to_dict()
        restored = VariationDescription.from_dict(d)
        assert restored.concentration_direction == 0.1
        assert restored.concentration_magnitude == 0.1
        assert restored.bias_direction == -0.05
        assert restored.bias_magnitude == 0.05
        assert restored.comparison_count == 3


class TestReferenceFrequencyState:
    def test_default(self):
        state = ReferenceFrequencyState()
        assert state.snapshot_history == []
        assert state.latest_variation is None
        assert state.total_snapshots_generated == 0
        assert state.total_snapshots_expired == 0

    def test_to_dict_from_dict(self):
        snap = ReferenceSnapshot(timestamp=1.0, concentration=0.5, structural_bias=0.3)
        var = VariationDescription(concentration_direction=0.1, comparison_count=2)
        state = ReferenceFrequencyState(
            snapshot_history=[snap],
            latest_variation=var,
            total_snapshots_generated=5,
            total_snapshots_expired=2,
        )
        d = state.to_dict()
        restored = ReferenceFrequencyState.from_dict(d)
        assert len(restored.snapshot_history) == 1
        assert restored.snapshot_history[0].concentration == 0.5
        assert restored.latest_variation is not None
        assert restored.latest_variation.concentration_direction == 0.1
        assert restored.total_snapshots_generated == 5
        assert restored.total_snapshots_expired == 2

    def test_to_dict_from_dict_no_variation(self):
        state = ReferenceFrequencyState()
        d = state.to_dict()
        restored = ReferenceFrequencyState.from_dict(d)
        assert restored.latest_variation is None


# =============================================================================
# Tests: collect_reference_counts
# =============================================================================

class TestCollectReferenceCounts:
    def test_all_none(self):
        """全てNoneの場合、全構造が空リストになる。"""
        result = collect_reference_counts()
        for key in ALL_STRUCTURE_KEYS:
            assert key in result
            assert result[key] == []

    def test_episodic_collection(self):
        store = MockEpisodeStore(
            episodes=(MockEpisodeEntry(3), MockEpisodeEntry(7), MockEpisodeEntry(0))
        )
        result = collect_reference_counts(episodic_store=store)
        assert result[STRUCTURE_EPISODIC] == [3, 7, 0]

    def test_binding_and_trace_collection(self):
        binding1 = MockMemoryBinding(
            reference_count=5,
            traces=(MockEmotionalTrace(2), MockEmotionalTrace(4)),
        )
        binding2 = MockMemoryBinding(
            reference_count=1,
            traces=(MockEmotionalTrace(0),),
        )
        store = MockBindingStore(bindings=(binding1, binding2))
        result = collect_reference_counts(binding_store=store)
        assert result[STRUCTURE_BINDING] == [5, 1]
        assert result[STRUCTURE_BINDING_TRACE] == [2, 4, 0]

    def test_introspection_collection(self):
        store = MockConsumptionStore(
            fragments=(MockIntrospectionFragment(10), MockIntrospectionFragment(1))
        )
        result = collect_reference_counts(consumption_store=store)
        assert result[STRUCTURE_INTROSPECTION] == [10, 1]

    def test_expectation_collection(self):
        store = MockExpectationStore(
            expectations=(MockExpectationCandidate(2), MockExpectationCandidate(8))
        )
        result = collect_reference_counts(expectation_store=store)
        assert result[STRUCTURE_EXPECTATION] == [2, 8]

    def test_motive_entry_and_impulse_collection(self):
        entry = MockMotiveEntry(
            reference_count=3,
            impulses=(MockMotiveImpulse(1), MockMotiveImpulse(6)),
        )
        store = MockMotiveStore(entries=(entry,))
        result = collect_reference_counts(motive_store=store)
        assert result[STRUCTURE_MOTIVE_ENTRY] == [3]
        assert result[STRUCTURE_MOTIVE_IMPULSE] == [1, 6]

    def test_narrative_collection(self):
        state = MockNarrativeState(
            fragments=(MockNarrativeFragment(0), MockNarrativeFragment(5))
        )
        result = collect_reference_counts(narrative_state=state)
        assert result[STRUCTURE_NARRATIVE] == [0, 5]

    def test_other_model_collection(self):
        store = MockOtherModelStore(
            hypotheses=(MockHypothesis(4), MockHypothesis(2), MockHypothesis(7))
        )
        result = collect_reference_counts(other_model_store=store)
        assert result[STRUCTURE_OTHER_MODEL] == [4, 2, 7]

    def test_self_reference_collection(self):
        """自己参照は単一の整数値がリスト[int]として収集される。"""
        state = MockSelfReferenceState(reference_count=12)
        result = collect_reference_counts(self_reference_state=state)
        assert result[STRUCTURE_SELF_REFERENCE] == [12]

    def test_action_result_collection(self):
        state = MockActionResultState(
            pairs=[MockActionResultPair(3), MockActionResultPair(0)]
        )
        result = collect_reference_counts(action_result_state=state)
        assert result[STRUCTURE_ACTION_RESULT] == [3, 0]

    def test_dialogue_learning_collection(self):
        state = MockDialogueLearningState(
            entries=[MockDialogueLearningEntry(5), MockDialogueLearningEntry(2)]
        )
        result = collect_reference_counts(dialogue_learning_state=state)
        assert result[STRUCTURE_DIALOGUE_LEARNING] == [5, 2]

    def test_forgetting_collection(self):
        state = MockForgettingState(
            series_index=[MockMemorySeriesRecord(1), MockMemorySeriesRecord(9)]
        )
        result = collect_reference_counts(forgetting_state=state)
        assert result[STRUCTURE_FORGETTING] == [1, 9]

    def test_full_collection(self):
        """全構造から収集した場合、全15キーが揃う。"""
        mocks = _make_full_mocks(
            ep_counts=[1, 2],
            bind_counts=[3],
            trace_counts=[[4, 5]],
            intro_counts=[6],
            expect_counts=[7],
            motive_counts=[8],
            impulse_counts=[[9, 10]],
            narr_counts=[11],
            other_counts=[12],
            self_ref_count=13,
            action_counts=[14],
            dialogue_counts=[15],
            forgetting_counts=[16],
            multi_path_recall_counts=(17, 18, 19),
            spontaneous_recall_counts=(20, 21, 22),
        )
        result = collect_reference_counts(**mocks)
        assert result[STRUCTURE_EPISODIC] == [1, 2]
        assert result[STRUCTURE_BINDING] == [3]
        assert result[STRUCTURE_BINDING_TRACE] == [4, 5]
        assert result[STRUCTURE_INTROSPECTION] == [6]
        assert result[STRUCTURE_EXPECTATION] == [7]
        assert result[STRUCTURE_MOTIVE_ENTRY] == [8]
        assert result[STRUCTURE_MOTIVE_IMPULSE] == [9, 10]
        assert result[STRUCTURE_NARRATIVE] == [11]
        assert result[STRUCTURE_OTHER_MODEL] == [12]
        assert result[STRUCTURE_SELF_REFERENCE] == [13]
        assert result[STRUCTURE_ACTION_RESULT] == [14]
        assert result[STRUCTURE_DIALOGUE_LEARNING] == [15]
        assert result[STRUCTURE_FORGETTING] == [16]
        assert result[STRUCTURE_MULTI_PATH_RECALL] == [17, 18, 19]
        assert result[STRUCTURE_SPONTANEOUS_RECALL] == [20, 21, 22]

    def test_multi_path_recall_collection(self):
        """多経路想起の各経路別カウントが収集される。"""
        state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(
                emotional_count=3,
                contextual_count=7,
                temporal_count=2,
            )
        )
        result = collect_reference_counts(multi_path_recall_state=state)
        assert result[STRUCTURE_MULTI_PATH_RECALL] == [3, 7, 2]

    def test_multi_path_recall_none_returns_empty(self):
        """multi_path_recall_state が None の場合、空リストを返す。"""
        result = collect_reference_counts()
        assert result[STRUCTURE_MULTI_PATH_RECALL] == []

    def test_multi_path_recall_no_path_stats(self):
        """path_stats が None の MultiPathRecallState の場合、空リストを返す。"""
        state = MockMultiPathRecallState(path_stats=None)
        result = collect_reference_counts(multi_path_recall_state=state)
        assert result[STRUCTURE_MULTI_PATH_RECALL] == []

    def test_spontaneous_recall_collection(self):
        """自発的想起の各経路別カウントが収集される。"""
        state = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(
                emotion_delta_count=5,
                motive_assoc_count=1,
                fluctuation_assoc_count=4,
            )
        )
        result = collect_reference_counts(spontaneous_recall_state=state)
        assert result[STRUCTURE_SPONTANEOUS_RECALL] == [5, 1, 4]

    def test_spontaneous_recall_none_returns_empty(self):
        """spontaneous_recall_state が None の場合、空リストを返す。"""
        result = collect_reference_counts()
        assert result[STRUCTURE_SPONTANEOUS_RECALL] == []

    def test_spontaneous_recall_no_path_stats(self):
        """path_stats が None の SpontaneousRecallState の場合、空リストを返す。"""
        state = MockSpontaneousRecallState(path_stats=None)
        result = collect_reference_counts(spontaneous_recall_state=state)
        assert result[STRUCTURE_SPONTANEOUS_RECALL] == []

    def test_both_recall_none_returns_empty_lists(self):
        """両方の想起状態が None の場合、両方の構造キーで空リストを返す。"""
        result = collect_reference_counts()
        assert result[STRUCTURE_MULTI_PATH_RECALL] == []
        assert result[STRUCTURE_SPONTANEOUS_RECALL] == []

    def test_read_only_no_side_effects(self):
        """収集後、元のモックオブジェクトの参照回数が変化しないことを確認。"""
        entry = MockEpisodeEntry(reference_count=5)
        store = MockEpisodeStore(episodes=(entry,))
        collect_reference_counts(episodic_store=store)
        assert entry.reference_count == 5


# =============================================================================
# Tests: compute_concentration
# =============================================================================

class TestComputeConcentration:
    def test_empty(self):
        assert compute_concentration([]) == 0.0

    def test_all_zero(self):
        assert compute_concentration([0, 0, 0]) == 0.0

    def test_single_element(self):
        assert compute_concentration([5]) == 0.0

    def test_uniform_distribution(self):
        """均等分布の場合、集中度は0に近い。"""
        conc = compute_concentration([5, 5, 5, 5, 5])
        assert conc == pytest.approx(0.0, abs=0.01)

    def test_concentrated_distribution(self):
        """一つに集中した場合、集中度は高い。"""
        conc = compute_concentration([0, 0, 0, 0, 100])
        assert conc > 0.5

    def test_moderate_concentration(self):
        """偏りのある分布では中程度の集中度。"""
        conc = compute_concentration([1, 1, 1, 1, 10])
        assert 0.0 < conc < 1.0

    def test_result_in_range(self):
        """結果は常に0.0-1.0の範囲内。"""
        for counts in [[1, 100], [0, 0, 0, 1], [10, 10], [1, 2, 3, 4, 5]]:
            conc = compute_concentration(counts)
            assert 0.0 <= conc <= 1.0

    def test_two_elements_equal(self):
        conc = compute_concentration([10, 10])
        assert conc == pytest.approx(0.0, abs=0.01)

    def test_two_elements_extreme(self):
        conc = compute_concentration([0, 100])
        assert conc > 0.3


# =============================================================================
# Tests: compute_structural_bias
# =============================================================================

class TestComputeStructuralBias:
    def test_empty_counts(self):
        """全構造が空の場合、偏在度は0。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        assert compute_structural_bias(counts) == 0.0

    def test_all_zero(self):
        """全構造の参照回数が0の場合、偏在度は0。"""
        counts = {k: [0, 0] for k in ALL_STRUCTURE_KEYS}
        assert compute_structural_bias(counts) == 0.0

    def test_single_structure_has_references(self):
        """一つの構造にのみ参照がある場合、偏在度は最大。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [10, 5, 3]
        assert compute_structural_bias(counts) == 1.0

    def test_uniform_distribution(self):
        """全構造に均等に分散した場合、偏在度は低い。"""
        counts = {k: [10] for k in ALL_STRUCTURE_KEYS}
        bias = compute_structural_bias(counts)
        assert bias < 0.2

    def test_biased_distribution(self):
        """一部構造に偏った場合、偏在度は中程度以上。"""
        counts = {k: [1] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [100, 50]
        bias = compute_structural_bias(counts)
        assert bias > 0.3

    def test_result_in_range(self):
        """結果は常に0.0-1.0の範囲内。"""
        for _ in range(5):
            counts = {k: [i] for i, k in enumerate(ALL_STRUCTURE_KEYS)}
            bias = compute_structural_bias(counts)
            assert 0.0 <= bias <= 1.0


# =============================================================================
# Tests: compose_snapshot
# =============================================================================

class TestComposeSnapshot:
    def test_empty_counts(self):
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        snap = compose_snapshot(counts, timestamp=100.0)
        assert snap.timestamp == 100.0
        assert snap.concentration == 0.0
        assert snap.structural_bias == 0.0

    def test_with_data(self):
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [0, 0, 0, 0, 50]
        snap = compose_snapshot(counts, timestamp=200.0)
        assert snap.timestamp == 200.0
        assert snap.concentration > 0.0
        assert snap.structural_bias > 0.0
        assert snap.structure_counts[STRUCTURE_EPISODIC] == [0, 0, 0, 0, 50]

    def test_auto_timestamp(self):
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        before = time.time()
        snap = compose_snapshot(counts)
        after = time.time()
        assert before <= snap.timestamp <= after

    def test_snapshot_preserves_all_keys(self):
        counts = {k: [i] for i, k in enumerate(ALL_STRUCTURE_KEYS)}
        snap = compose_snapshot(counts)
        for key in ALL_STRUCTURE_KEYS:
            assert key in snap.structure_counts


# =============================================================================
# Tests: derive_variation
# =============================================================================

class TestDeriveVariation:
    def test_insufficient_history(self):
        """断面が1件以下の場合、Noneを返す。"""
        cfg = ReferenceFrequencyConfig()
        assert derive_variation([], cfg) is None
        assert derive_variation([ReferenceSnapshot()], cfg) is None

    def test_two_snapshots(self):
        """2件の断面があれば変動記述が生成される。"""
        cfg = ReferenceFrequencyConfig()
        s1 = ReferenceSnapshot(concentration=0.3, structural_bias=0.2)
        s2 = ReferenceSnapshot(concentration=0.5, structural_bias=0.4)
        var = derive_variation([s1, s2], cfg)
        assert var is not None
        assert var.concentration_direction == pytest.approx(0.2, abs=0.001)
        assert var.concentration_magnitude == pytest.approx(0.2, abs=0.001)
        assert var.bias_direction == pytest.approx(0.2, abs=0.001)
        assert var.bias_magnitude == pytest.approx(0.2, abs=0.001)
        assert var.comparison_count == 1

    def test_multiple_snapshots_averages(self):
        """複数の過去断面の平均との差分。"""
        cfg = ReferenceFrequencyConfig(variation_comparison_count=3)
        s1 = ReferenceSnapshot(concentration=0.2, structural_bias=0.1)
        s2 = ReferenceSnapshot(concentration=0.4, structural_bias=0.3)
        s3 = ReferenceSnapshot(concentration=0.6, structural_bias=0.5)
        s4 = ReferenceSnapshot(concentration=0.5, structural_bias=0.4)

        var = derive_variation([s1, s2, s3, s4], cfg)
        assert var is not None
        # 比較対象: s1, s2, s3 → 平均 conc=(0.2+0.4+0.6)/3=0.4, bias=(0.1+0.3+0.5)/3=0.3
        assert var.concentration_direction == pytest.approx(0.1, abs=0.001)
        assert var.bias_direction == pytest.approx(0.1, abs=0.001)
        assert var.comparison_count == 3

    def test_variation_no_accumulation(self):
        """安全弁3: 変動記述は断面間の差分から再導出される。累積しない。"""
        cfg = ReferenceFrequencyConfig(variation_comparison_count=2)
        # 最初に上昇傾向
        history1 = [
            ReferenceSnapshot(concentration=0.1),
            ReferenceSnapshot(concentration=0.3),
            ReferenceSnapshot(concentration=0.5),
        ]
        var1 = derive_variation(history1, cfg)
        assert var1 is not None
        assert var1.concentration_direction > 0

        # 次に下降傾向（全く別の値）
        history2 = [
            ReferenceSnapshot(concentration=0.8),
            ReferenceSnapshot(concentration=0.6),
            ReferenceSnapshot(concentration=0.2),
        ]
        var2 = derive_variation(history2, cfg)
        assert var2 is not None
        assert var2.concentration_direction < 0

        # 累積的な傾向蓄積がないことを確認
        # 各呼び出しは独立に導出される


# =============================================================================
# Tests: process_reference_frequency
# =============================================================================

class TestProcessReferenceFrequency:
    def test_first_snapshot(self):
        """初回処理で断面が1件生成される。"""
        state = create_reference_frequency_state()
        mocks = _make_full_mocks(ep_counts=[1, 2, 3])
        new_state = process_reference_frequency(state, **mocks, timestamp=100.0)
        assert len(new_state.snapshot_history) == 1
        assert new_state.snapshot_history[0].timestamp == 100.0
        assert new_state.total_snapshots_generated == 1
        assert new_state.total_snapshots_expired == 0
        assert new_state.latest_variation is None  # 1件なので変動記述なし

    def test_second_snapshot_with_variation(self):
        """2回目の処理で変動記述が生成される。"""
        state = create_reference_frequency_state()
        mocks1 = _make_full_mocks(ep_counts=[1, 1, 1])
        state = process_reference_frequency(state, **mocks1, timestamp=100.0)

        mocks2 = _make_full_mocks(ep_counts=[0, 0, 100])
        state = process_reference_frequency(state, **mocks2, timestamp=200.0)

        assert len(state.snapshot_history) == 2
        assert state.total_snapshots_generated == 2
        assert state.latest_variation is not None

    def test_fifo_expiration(self):
        """安全弁4: 断面履歴の有限性。上限到達時にFIFO。"""
        cfg = create_reference_frequency_config(max_snapshot_history=3)
        state = create_reference_frequency_state()

        for i in range(5):
            mocks = _make_full_mocks(ep_counts=[i])
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i),
            )

        assert len(state.snapshot_history) == 3
        assert state.total_snapshots_generated == 5
        assert state.total_snapshots_expired == 2
        # 最古の断面は消失している（timestamp=0.0, 1.0は消失）
        assert state.snapshot_history[0].timestamp == 2.0
        assert state.snapshot_history[1].timestamp == 3.0
        assert state.snapshot_history[2].timestamp == 4.0

    def test_no_enrichment_output(self):
        """安全弁5: enrichment出力経路を持たない。
        process_reference_frequencyの返り値にenrichment関連のキーがないことを確認。"""
        state = create_reference_frequency_state()
        mocks = _make_full_mocks(ep_counts=[1])
        new_state = process_reference_frequency(state, **mocks)
        # ReferenceFrequencyStateにget_enrichment_dataメソッドが存在しないことを確認
        assert not hasattr(new_state, "get_enrichment_data")

    def test_all_none_inputs(self):
        """全入力がNoneでもエラーにならない。"""
        state = create_reference_frequency_state()
        new_state = process_reference_frequency(state, timestamp=1.0)
        assert len(new_state.snapshot_history) == 1
        assert new_state.snapshot_history[0].concentration == 0.0
        assert new_state.snapshot_history[0].structural_bias == 0.0

    def test_state_immutability(self):
        """処理後に元のstateが変更されないこと。"""
        state = create_reference_frequency_state()
        mocks = _make_full_mocks(ep_counts=[1, 2])
        new_state = process_reference_frequency(state, **mocks)
        assert len(state.snapshot_history) == 0  # 元のstateは変更されない
        assert len(new_state.snapshot_history) == 1

    def test_recall_states_in_snapshot(self):
        """process_reference_frequency に渡された想起状態がスナップショットに含まれる。"""
        state = create_reference_frequency_state()
        mpr_state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(
                emotional_count=4,
                contextual_count=6,
                temporal_count=2,
            )
        )
        sr_state = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(
                emotion_delta_count=3,
                motive_assoc_count=1,
                fluctuation_assoc_count=5,
            )
        )
        new_state = process_reference_frequency(
            state,
            multi_path_recall_state=mpr_state,
            spontaneous_recall_state=sr_state,
            timestamp=1.0,
        )
        snap = new_state.snapshot_history[0]
        assert snap.structure_counts[STRUCTURE_MULTI_PATH_RECALL] == [4, 6, 2]
        assert snap.structure_counts[STRUCTURE_SPONTANEOUS_RECALL] == [3, 1, 5]

    def test_variation_recalculated_each_time(self):
        """安全弁3: 変動記述は毎回再導出される。"""
        cfg = create_reference_frequency_config(variation_comparison_count=2)
        state = create_reference_frequency_state()

        # 集中度が上昇する3ステップ
        for i, counts in enumerate([[1, 1, 1], [0, 0, 10], [0, 0, 100]]):
            mocks = _make_full_mocks(ep_counts=counts)
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i),
            )

        var = state.latest_variation
        assert var is not None
        # 変動記述は直近と過去の差分から計算される（累積ではない）
        assert var.comparison_count == 2


# =============================================================================
# Tests: READ-ONLY output functions
# =============================================================================

class TestReadOnlyOutputs:
    def test_get_latest_snapshot_empty(self):
        state = create_reference_frequency_state()
        assert get_latest_snapshot(state) is None

    def test_get_latest_snapshot(self):
        snap = ReferenceSnapshot(timestamp=42.0, concentration=0.5)
        state = ReferenceFrequencyState(snapshot_history=[snap])
        result = get_latest_snapshot(state)
        assert result is not None
        assert result.timestamp == 42.0

    def test_get_snapshot_history_empty(self):
        state = create_reference_frequency_state()
        assert get_snapshot_history(state) == []

    def test_get_snapshot_history_returns_copy(self):
        """返されるリストはコピーであり、元のstateに影響しない。"""
        snap = ReferenceSnapshot(timestamp=1.0)
        state = ReferenceFrequencyState(snapshot_history=[snap])
        history = get_snapshot_history(state)
        history.append(ReferenceSnapshot(timestamp=2.0))
        assert len(state.snapshot_history) == 1  # 元は変更されない

    def test_get_latest_variation_empty(self):
        state = create_reference_frequency_state()
        assert get_latest_variation(state) is None

    def test_get_latest_variation(self):
        var = VariationDescription(concentration_direction=0.1)
        state = ReferenceFrequencyState(latest_variation=var)
        result = get_latest_variation(state)
        assert result is not None
        assert result.concentration_direction == 0.1


# =============================================================================
# Tests: get_reference_summary
# =============================================================================

class TestGetReferenceSummary:
    def test_empty_state(self):
        state = create_reference_frequency_state()
        summary = get_reference_summary(state)
        assert summary["history_count"] == 0
        assert summary["total_generated"] == 0
        assert "latest_concentration" not in summary

    def test_with_snapshot(self):
        counts = {k: [5] for k in ALL_STRUCTURE_KEYS}
        snap = compose_snapshot(counts, timestamp=1.0)
        state = ReferenceFrequencyState(
            snapshot_history=[snap],
            total_snapshots_generated=1,
        )
        summary = get_reference_summary(state)
        assert summary["history_count"] == 1
        assert "latest_concentration" in summary
        assert "structure_totals" in summary
        assert "structure_record_counts" in summary

        # 全構造キーが含まれること
        for key in ALL_STRUCTURE_KEYS:
            assert key in summary["structure_totals"]
            assert key in summary["structure_record_counts"]

    def test_with_variation(self):
        snap1 = ReferenceSnapshot(concentration=0.3, structural_bias=0.2)
        snap2 = ReferenceSnapshot(concentration=0.5, structural_bias=0.4)
        var = VariationDescription(
            concentration_direction=0.2,
            concentration_magnitude=0.2,
            bias_direction=0.2,
            bias_magnitude=0.2,
            comparison_count=1,
        )
        state = ReferenceFrequencyState(
            snapshot_history=[snap1, snap2],
            latest_variation=var,
            total_snapshots_generated=2,
        )
        summary = get_reference_summary(state)
        assert summary["variation_concentration_direction"] == 0.2
        assert summary["variation_comparison_count"] == 1

    def test_summary_no_evaluation(self):
        """安全弁2: サマリに評価的語彙が含まれないことの構造的確認。
        サマリは数値のみであり、「良い」「悪い」「改善」「悪化」等の
        評価的変換を含まない。"""
        counts = {k: [10] for k in ALL_STRUCTURE_KEYS}
        snap = compose_snapshot(counts)
        state = ReferenceFrequencyState(snapshot_history=[snap])
        summary = get_reference_summary(state)
        # サマリのキーに評価的なものがないことを確認
        for key in summary:
            assert "good" not in key.lower()
            assert "bad" not in key.lower()
            assert "important" not in key.lower()
            assert "unimportant" not in key.lower()


# =============================================================================
# Tests: Save / Load
# =============================================================================

class TestSaveLoad:
    def test_roundtrip_empty(self):
        state = create_reference_frequency_state()
        data = save_state(state)
        restored = load_state(data)
        assert restored.snapshot_history == []
        assert restored.latest_variation is None
        assert restored.total_snapshots_generated == 0

    def test_roundtrip_with_data(self):
        state = create_reference_frequency_state()
        mocks = _make_full_mocks(
            ep_counts=[1, 2, 3],
            bind_counts=[4],
            trace_counts=[[5, 6]],
        )
        state = process_reference_frequency(state, **mocks, timestamp=100.0)
        state = process_reference_frequency(state, **mocks, timestamp=200.0)

        data = save_state(state)
        restored = load_state(data)

        assert len(restored.snapshot_history) == 2
        assert restored.total_snapshots_generated == 2
        assert restored.latest_variation is not None

    def test_roundtrip_preserves_structure_counts(self):
        counts = {STRUCTURE_EPISODIC: [1, 2, 3], STRUCTURE_BINDING: [4, 5]}
        snap = compose_snapshot(counts, timestamp=1.0)
        state = ReferenceFrequencyState(snapshot_history=[snap])

        data = save_state(state)
        restored = load_state(data)

        restored_snap = restored.snapshot_history[0]
        assert restored_snap.structure_counts.get(STRUCTURE_EPISODIC) == [1, 2, 3]
        assert restored_snap.structure_counts.get(STRUCTURE_BINDING) == [4, 5]


# =============================================================================
# Tests: Factory functions
# =============================================================================

class TestFactoryFunctions:
    def test_create_state(self):
        state = create_reference_frequency_state()
        assert isinstance(state, ReferenceFrequencyState)
        assert state.snapshot_history == []
        assert state.latest_variation is None

    def test_create_config(self):
        cfg = create_reference_frequency_config(
            max_snapshot_history=50,
            variation_comparison_count=10,
        )
        assert isinstance(cfg, ReferenceFrequencyConfig)
        assert cfg.max_snapshot_history == 50
        assert cfg.variation_comparison_count == 10


# =============================================================================
# Tests: Safety Valves (安全弁)
# =============================================================================

class TestSafetyValves:
    def test_valve1_all_records_equivalent(self):
        """安全弁1: 全記録等価維持保証。
        参照回数の多寡に基づいてフィルタリング・選別が行われないことを確認。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [0, 0, 0, 100]  # 大きな偏り
        snap = compose_snapshot(counts)

        # 断面にはフィルタされた記録がない（全4件が保存されている）
        assert len(snap.structure_counts[STRUCTURE_EPISODIC]) == 4
        assert snap.structure_counts[STRUCTURE_EPISODIC] == [0, 0, 0, 100]

    def test_valve2_no_evaluative_labels(self):
        """安全弁2: 評価的変換の禁止。
        ReferenceSnapshotに「良い」「悪い」「重要」「不要」等のラベルがないことを確認。"""
        snap = ReferenceSnapshot(concentration=0.9, structural_bias=0.8)
        d = snap.to_dict()
        # 辞書のキーに評価的ラベルがないことを確認
        assert "important" not in str(d).lower()
        assert "unimportant" not in str(d).lower()
        assert "good" not in str(d).lower()
        assert "bad" not in str(d).lower()
        assert "threshold" not in str(d).lower()

    def test_valve3_no_cumulative_trend(self):
        """安全弁3: 累積的傾向の抑制。
        VariationDescriptionに「トレンド」「傾向」を累積する構造がないことを確認。"""
        var = VariationDescription()
        d = var.to_dict()
        # 累積的な傾向を示すフィールドがないことを確認
        assert "trend" not in d
        assert "cumulative" not in str(d).lower()
        assert "accumulated" not in str(d).lower()

    def test_valve4_finite_history(self):
        """安全弁4: 断面履歴の有限性。上限を超えた断面が消失すること。"""
        cfg = create_reference_frequency_config(max_snapshot_history=2)
        state = create_reference_frequency_state()

        for i in range(10):
            mocks = _make_full_mocks(ep_counts=[i])
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i),
            )

        assert len(state.snapshot_history) == 2
        assert state.total_snapshots_expired == 8

    def test_valve5_no_enrichment_method(self):
        """安全弁5: 出力経路の限定。
        enrichment出力メソッドが存在しないことを確認。"""
        state = create_reference_frequency_state()
        assert not hasattr(state, "get_enrichment_data")
        assert not hasattr(state, "enrichment")

        # モジュールレベルでもenrichment関数が存在しない
        import psyche.reference_frequency_description as mod
        # get_enrichment_data が存在しないことを確認
        assert not hasattr(mod, "get_enrichment_data")

    def test_valve5_no_dynamic_output_expansion(self):
        """安全弁5: 出力経路の不拡張。
        ReferenceFrequencyStateに動的出力先追加の仕組みがないことを確認。"""
        state = create_reference_frequency_state()
        # 出力先リストやコールバック登録機構が存在しないことを確認
        assert not hasattr(state, "output_targets")
        assert not hasattr(state, "register_output")
        assert not hasattr(state, "add_output_path")


# =============================================================================
# Tests: Pathway isolation (経路遮断)
# =============================================================================

class TestPathwayIsolation:
    def test_no_forgetting_pipeline_output(self):
        """忘却パイプラインへの出力経路が存在しないことを確認。"""
        import psyche.reference_frequency_description as mod
        # 忘却関連の出力関数が存在しないこと
        assert not hasattr(mod, "notify_forgetting")
        assert not hasattr(mod, "send_to_forgetting")
        assert not hasattr(mod, "forgetting_output")

    def test_no_recall_path_output(self):
        """想起経路選択への出力経路が存在しないことを確認。"""
        import psyche.reference_frequency_description as mod
        assert not hasattr(mod, "influence_recall")
        assert not hasattr(mod, "recall_weight")
        assert not hasattr(mod, "recall_priority")

    def test_no_external_output_path(self):
        """外部出力層への出力経路が存在しないことを確認。"""
        import psyche.reference_frequency_description as mod
        assert not hasattr(mod, "get_enrichment_data")
        assert not hasattr(mod, "to_external")
        assert not hasattr(mod, "external_output")

    def test_process_does_not_modify_inputs(self):
        """入力構造の参照回数を変更しないことを確認。"""
        ep_entry = MockEpisodeEntry(reference_count=5)
        store = MockEpisodeStore(episodes=(ep_entry,))
        action_pair = MockActionResultPair(reference_count=3)
        action_state = MockActionResultState(pairs=[action_pair])

        state = create_reference_frequency_state()
        process_reference_frequency(
            state,
            episodic_store=store,
            action_result_state=action_state,
        )

        # 入力は変更されていない
        assert ep_entry.reference_count == 5
        assert action_pair.reference_count == 3


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_large_reference_counts(self):
        """大きな参照回数値での動作。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [1000000, 0, 0]
        snap = compose_snapshot(counts)
        assert 0.0 <= snap.concentration <= 1.0
        assert 0.0 <= snap.structural_bias <= 1.0

    def test_many_records(self):
        """多数の記録がある場合の動作。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = list(range(1000))
        snap = compose_snapshot(counts)
        assert 0.0 <= snap.concentration <= 1.0

    def test_all_same_reference_count(self):
        """全記録が同じ参照回数の場合。"""
        counts = {k: [5, 5, 5] for k in ALL_STRUCTURE_KEYS}
        snap = compose_snapshot(counts)
        assert snap.concentration == pytest.approx(0.0, abs=0.01)
        # 各構造の合計が同じなので偏在度も低い
        assert snap.structural_bias < 0.2

    def test_single_record_in_single_structure(self):
        """単一構造に1件のみの場合。"""
        counts = {k: [] for k in ALL_STRUCTURE_KEYS}
        counts[STRUCTURE_EPISODIC] = [1]
        snap = compose_snapshot(counts)
        assert snap.concentration == 0.0  # 1件なので集中度は0
        assert snap.structural_bias == 1.0  # 単一構造のみ

    def test_process_multiple_cycles(self):
        """複数サイクルの処理が正常に動作すること。"""
        state = create_reference_frequency_state()
        for i in range(20):
            mocks = _make_full_mocks(ep_counts=[i, i + 1])
            state = process_reference_frequency(state, **mocks, timestamp=float(i))

        assert len(state.snapshot_history) == 20
        assert state.total_snapshots_generated == 20
        assert state.latest_variation is not None

    def test_config_max_history_one(self):
        """max_snapshot_history=1の場合でもクラッシュしない。"""
        cfg = create_reference_frequency_config(max_snapshot_history=1)
        state = create_reference_frequency_state()

        for i in range(5):
            mocks = _make_full_mocks(ep_counts=[i])
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i),
            )

        assert len(state.snapshot_history) == 1
        assert state.total_snapshots_expired == 4
        # 1件しかないので変動記述はNone
        assert state.latest_variation is None

    def test_variation_comparison_count_larger_than_history(self):
        """variation_comparison_countが履歴件数より大きい場合。"""
        cfg = create_reference_frequency_config(variation_comparison_count=100)
        state = create_reference_frequency_state()

        for i in range(3):
            mocks = _make_full_mocks(ep_counts=[i])
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i),
            )

        # クラッシュせず、利用可能な過去断面で変動記述が生成される
        assert state.latest_variation is not None
        assert state.latest_variation.comparison_count == 2  # 3-1=2件の過去断面

    def test_zero_reference_counts_everywhere(self):
        """全構造の全記録が参照回数0の場合。"""
        mocks = _make_full_mocks(
            ep_counts=[0, 0, 0],
            bind_counts=[0],
            trace_counts=[[0]],
            intro_counts=[0],
            expect_counts=[0],
            motive_counts=[0],
            impulse_counts=[[0]],
            narr_counts=[0],
            other_counts=[0],
            self_ref_count=0,
            action_counts=[0],
            dialogue_counts=[0],
            forgetting_counts=[0],
        )
        state = create_reference_frequency_state()
        new_state = process_reference_frequency(state, **mocks)
        snap = new_state.snapshot_history[0]
        assert snap.concentration == 0.0
        assert snap.structural_bias == 0.0


# =============================================================================
# Tests: Structure keys
# =============================================================================

class TestStructureKeys:
    def test_all_keys_present(self):
        """15の構造キーが全て定義されていること。"""
        assert len(ALL_STRUCTURE_KEYS) == 15

    def test_keys_are_unique(self):
        """全ての構造キーがユニークであること。"""
        assert len(set(ALL_STRUCTURE_KEYS)) == len(ALL_STRUCTURE_KEYS)

    def test_collect_returns_all_keys(self):
        """collect_reference_countsが全キーを返すこと。"""
        result = collect_reference_counts()
        for key in ALL_STRUCTURE_KEYS:
            assert key in result


# =============================================================================
# Tests: Integration-like scenario
# =============================================================================

class TestIntegrationScenario:
    def test_full_lifecycle(self):
        """完全なライフサイクル: 生成→蓄積→FIFO→変動記述→保存→復元。"""
        cfg = create_reference_frequency_config(
            max_snapshot_history=5,
            variation_comparison_count=3,
        )
        state = create_reference_frequency_state()

        # 7サイクル実行（5件上限なので2件消失する）
        for i in range(7):
            mocks = _make_full_mocks(
                ep_counts=[i, i * 2],
                bind_counts=[i],
                trace_counts=[[i]],
                self_ref_count=i,
            )
            state = process_reference_frequency(
                state, **mocks, config=cfg, timestamp=float(i * 10),
            )

        # 状態の確認
        assert len(state.snapshot_history) == 5
        assert state.total_snapshots_generated == 7
        assert state.total_snapshots_expired == 2
        assert state.latest_variation is not None

        # 最古の断面はtimestamp=20.0（0.0と10.0は消失）
        assert state.snapshot_history[0].timestamp == 20.0

        # 保存と復元
        data = save_state(state)
        restored = load_state(data)
        assert len(restored.snapshot_history) == 5
        assert restored.total_snapshots_generated == 7
        assert restored.latest_variation is not None

        # 復元後も処理を継続できる
        mocks = _make_full_mocks(ep_counts=[100])
        restored = process_reference_frequency(
            restored, **mocks, config=cfg, timestamp=100.0,
        )
        assert len(restored.snapshot_history) == 5
        assert restored.total_snapshots_generated == 8
        assert restored.total_snapshots_expired == 3

    def test_concentration_tracks_changes(self):
        """集中度が参照分布の変化に追従すること。"""
        state = create_reference_frequency_state()

        # Step 1: 均等分布（self_reference=5で全構造に5を配置）
        mocks1 = _make_full_mocks(
            ep_counts=[5, 5, 5, 5, 5],
            bind_counts=[5],
            trace_counts=[[5]],
            intro_counts=[5],
            expect_counts=[5],
            motive_counts=[5],
            impulse_counts=[[5]],
            narr_counts=[5],
            other_counts=[5],
            self_ref_count=5,
            action_counts=[5],
            dialogue_counts=[5],
            forgetting_counts=[5],
        )
        state = process_reference_frequency(state, **mocks1, timestamp=1.0)
        snap1 = state.snapshot_history[-1]
        assert snap1.concentration < 0.1

        # Step 2: 集中分布
        mocks2 = _make_full_mocks(ep_counts=[0, 0, 0, 0, 100])
        state = process_reference_frequency(state, **mocks2, timestamp=2.0)
        snap2 = state.snapshot_history[-1]
        assert snap2.concentration > snap1.concentration

        # 変動記述が集中方向を示す
        var = state.latest_variation
        assert var is not None
        assert var.concentration_direction > 0

    def test_structural_bias_tracks_changes(self):
        """偏在度が構造間分布の変化に追従すること。"""
        state = create_reference_frequency_state()

        # Step 1: 複数構造に分散
        mocks1 = _make_full_mocks(
            ep_counts=[5],
            bind_counts=[5],
            intro_counts=[5],
            expect_counts=[5],
            narr_counts=[5],
        )
        state = process_reference_frequency(state, **mocks1, timestamp=1.0)
        snap1 = state.snapshot_history[-1]

        # Step 2: 単一構造に集中
        mocks2 = _make_full_mocks(ep_counts=[100])
        state = process_reference_frequency(state, **mocks2, timestamp=2.0)
        snap2 = state.snapshot_history[-1]

        assert snap2.structural_bias > snap1.structural_bias
