"""
tests/test_forgetting_recall_balance.py - 忘却と想起の均衡記述テスト

設計書 (design_forgetting_recall_balance.md) に基づく機能テスト。
"""

import time
import pytest

from psyche.forgetting_recall_balance import (
    # Config
    ForgettingRecallBalanceConfig,
    create_forgetting_recall_balance_config,
    # State
    ForgettingRecallBalanceState,
    create_forgetting_recall_balance_state,
    # Data structures
    ForgettingSectionSnapshot,
    ExternalRecallSectionSnapshot,
    SpontaneousRecallSectionSnapshot,
    JuxtapositionEntry,
    # Stage 1: extraction
    extract_forgetting_section,
    extract_external_recall_section,
    extract_spontaneous_recall_section,
    # Stage 2: composition
    compose_juxtaposition,
    # Stage 3: accumulation
    accumulate_entry,
    # Main processing
    process_forgetting_recall_balance,
    # READ-ONLY accessors
    get_recent_entries,
    get_history,
    get_balance_summary,
    get_enrichment_text,
    # Save/Load
    save_state,
    load_state,
)


# =============================================================================
# Mock objects for READ-ONLY input
# =============================================================================

class MockSeriesRecord:
    """忘却/固定化の系列レコードのモック。"""
    def __init__(self, forgetting_stage="active", is_protected=False):
        self.forgetting_stage = forgetting_stage
        self.is_protected = is_protected


class MockForgettingFixationState:
    """忘却/固定化の内部状態のモック。"""
    def __init__(self, series_index=None):
        self.series_index = series_index or []


class MockForgettingFixationResult:
    """忘却/固定化の処理結果のモック。"""
    def __init__(self, newly_forgotten=0, newly_recovered=0, newly_fixating=0):
        self.newly_forgotten = newly_forgotten
        self.newly_recovered = newly_recovered
        self.newly_fixating = newly_fixating


class MockPathStatistics:
    """多経路想起の経路別統計のモック。"""
    def __init__(self, emotional_count=0, contextual_count=0, temporal_count=0):
        self.emotional_count = emotional_count
        self.contextual_count = contextual_count
        self.temporal_count = temporal_count


class MockRecallCandidate:
    """想起候補のモック。"""
    pass


class MockMultiPathRecallState:
    """多経路想起の内部状態のモック。"""
    def __init__(self, path_stats=None, current_candidates=None):
        self.path_stats = path_stats or MockPathStatistics()
        self.current_candidates = current_candidates or []


class MockSpontaneousRecallPathStatistics:
    """自発的想起の経路別統計のモック。"""
    def __init__(self, emotion_delta_count=0, motive_assoc_count=0, fluctuation_assoc_count=0):
        self.emotion_delta_count = emotion_delta_count
        self.motive_assoc_count = motive_assoc_count
        self.fluctuation_assoc_count = fluctuation_assoc_count


class MockSpontaneousRecallCandidate:
    """自発的想起候補のモック。"""
    pass


class MockSpontaneousRecallState:
    """自発的想起の内部状態のモック。"""
    def __init__(self, path_stats=None, current_candidates=None):
        self.path_stats = path_stats or MockSpontaneousRecallPathStatistics()
        self.current_candidates = current_candidates or []


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    return ForgettingRecallBalanceConfig()


@pytest.fixture
def small_config():
    """テスト用の小さな設定。"""
    return ForgettingRecallBalanceConfig(
        max_history=5,
        enrichment_recent_count=3,
    )


@pytest.fixture
def empty_state():
    return create_forgetting_recall_balance_state()


# =============================================================================
# Config
# =============================================================================

class TestConfig:
    def test_default_values(self):
        cfg = ForgettingRecallBalanceConfig()
        assert cfg.max_history == 30
        assert cfg.enrichment_recent_count == 5

    def test_custom_values(self):
        cfg = create_forgetting_recall_balance_config(
            max_history=50,
            enrichment_recent_count=10,
        )
        assert cfg.max_history == 50
        assert cfg.enrichment_recent_count == 10


# =============================================================================
# Data Structures
# =============================================================================

class TestForgettingSectionSnapshot:
    def test_default(self):
        snap = ForgettingSectionSnapshot()
        assert snap.active_count == 0
        assert snap.weakening_count == 0
        assert snap.fading_count == 0
        assert snap.near_invisible_count == 0
        assert snap.invisible_count == 0
        assert snap.fixation_sign_count == 0
        assert snap.protected_count == 0
        assert snap.newly_forgotten_count == 0
        assert snap.newly_recovered_count == 0

    def test_to_dict(self):
        snap = ForgettingSectionSnapshot(active_count=5, invisible_count=2)
        d = snap.to_dict()
        assert d["active_count"] == 5
        assert d["invisible_count"] == 2

    def test_from_dict(self):
        d = {"active_count": 3, "weakening_count": 1, "protected_count": 2}
        snap = ForgettingSectionSnapshot.from_dict(d)
        assert snap.active_count == 3
        assert snap.weakening_count == 1
        assert snap.protected_count == 2

    def test_roundtrip(self):
        snap = ForgettingSectionSnapshot(
            active_count=10, weakening_count=3, fading_count=2,
            near_invisible_count=1, invisible_count=0,
            fixation_sign_count=1, protected_count=4,
            newly_forgotten_count=2, newly_recovered_count=1,
        )
        d = snap.to_dict()
        snap2 = ForgettingSectionSnapshot.from_dict(d)
        assert snap2.active_count == 10
        assert snap2.newly_recovered_count == 1


class TestExternalRecallSectionSnapshot:
    def test_default(self):
        snap = ExternalRecallSectionSnapshot()
        assert snap.emotional_count == 0
        assert snap.contextual_count == 0
        assert snap.temporal_count == 0
        assert snap.total_count == 0

    def test_to_dict(self):
        snap = ExternalRecallSectionSnapshot(emotional_count=3, total_count=5)
        d = snap.to_dict()
        assert d["emotional_count"] == 3
        assert d["total_count"] == 5

    def test_from_dict(self):
        d = {"emotional_count": 2, "contextual_count": 1, "temporal_count": 3, "total_count": 6}
        snap = ExternalRecallSectionSnapshot.from_dict(d)
        assert snap.emotional_count == 2
        assert snap.total_count == 6

    def test_roundtrip(self):
        snap = ExternalRecallSectionSnapshot(
            emotional_count=5, contextual_count=3, temporal_count=2, total_count=10,
        )
        d = snap.to_dict()
        snap2 = ExternalRecallSectionSnapshot.from_dict(d)
        assert snap2.emotional_count == 5
        assert snap2.total_count == 10


class TestSpontaneousRecallSectionSnapshot:
    def test_default(self):
        snap = SpontaneousRecallSectionSnapshot()
        assert snap.emotion_delta_count == 0
        assert snap.motive_assoc_count == 0
        assert snap.fluctuation_assoc_count == 0
        assert snap.total_count == 0

    def test_to_dict(self):
        snap = SpontaneousRecallSectionSnapshot(emotion_delta_count=2, total_count=4)
        d = snap.to_dict()
        assert d["emotion_delta_count"] == 2
        assert d["total_count"] == 4

    def test_from_dict(self):
        d = {"emotion_delta_count": 1, "motive_assoc_count": 2, "fluctuation_assoc_count": 3, "total_count": 6}
        snap = SpontaneousRecallSectionSnapshot.from_dict(d)
        assert snap.emotion_delta_count == 1
        assert snap.total_count == 6

    def test_roundtrip(self):
        snap = SpontaneousRecallSectionSnapshot(
            emotion_delta_count=4, motive_assoc_count=3,
            fluctuation_assoc_count=2, total_count=9,
        )
        d = snap.to_dict()
        snap2 = SpontaneousRecallSectionSnapshot.from_dict(d)
        assert snap2.emotion_delta_count == 4
        assert snap2.total_count == 9


class TestJuxtapositionEntry:
    def test_default(self):
        entry = JuxtapositionEntry()
        assert entry.timestamp == 0.0
        assert entry.forgetting.active_count == 0
        assert entry.external_recall.total_count == 0
        assert entry.spontaneous_recall.total_count == 0

    def test_to_dict(self):
        entry = JuxtapositionEntry(
            timestamp=1000.0,
            forgetting=ForgettingSectionSnapshot(active_count=5),
            external_recall=ExternalRecallSectionSnapshot(total_count=3),
            spontaneous_recall=SpontaneousRecallSectionSnapshot(total_count=2),
        )
        d = entry.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["forgetting"]["active_count"] == 5
        assert d["external_recall"]["total_count"] == 3
        assert d["spontaneous_recall"]["total_count"] == 2

    def test_from_dict(self):
        d = {
            "timestamp": 2000.0,
            "forgetting": {"active_count": 7},
            "external_recall": {"emotional_count": 4, "total_count": 6},
            "spontaneous_recall": {"emotion_delta_count": 3, "total_count": 5},
        }
        entry = JuxtapositionEntry.from_dict(d)
        assert entry.timestamp == 2000.0
        assert entry.forgetting.active_count == 7
        assert entry.external_recall.emotional_count == 4
        assert entry.spontaneous_recall.emotion_delta_count == 3

    def test_roundtrip(self):
        entry = JuxtapositionEntry(
            timestamp=3000.0,
            forgetting=ForgettingSectionSnapshot(
                active_count=10, weakening_count=3, invisible_count=1,
                fixation_sign_count=2, protected_count=4,
            ),
            external_recall=ExternalRecallSectionSnapshot(
                emotional_count=5, contextual_count=3, temporal_count=2, total_count=10,
            ),
            spontaneous_recall=SpontaneousRecallSectionSnapshot(
                emotion_delta_count=4, motive_assoc_count=2,
                fluctuation_assoc_count=1, total_count=7,
            ),
        )
        d = entry.to_dict()
        entry2 = JuxtapositionEntry.from_dict(d)
        assert entry2.timestamp == 3000.0
        assert entry2.forgetting.fixation_sign_count == 2
        assert entry2.external_recall.contextual_count == 3
        assert entry2.spontaneous_recall.fluctuation_assoc_count == 1


# =============================================================================
# State
# =============================================================================

class TestState:
    def test_default(self):
        state = create_forgetting_recall_balance_state()
        assert state.cycle_count == 0
        assert len(state.history) == 0

    def test_to_dict(self):
        state = ForgettingRecallBalanceState(cycle_count=5)
        d = state.to_dict()
        assert d["cycle_count"] == 5
        assert d["history"] == []

    def test_from_dict(self):
        d = {
            "cycle_count": 3,
            "history": [
                {"timestamp": 1000.0, "forgetting": {}, "external_recall": {}, "spontaneous_recall": {}},
            ],
        }
        state = ForgettingRecallBalanceState.from_dict(d)
        assert state.cycle_count == 3
        assert len(state.history) == 1

    def test_roundtrip(self):
        state = ForgettingRecallBalanceState(
            history=[
                JuxtapositionEntry(timestamp=1000.0),
                JuxtapositionEntry(timestamp=2000.0),
            ],
            cycle_count=10,
        )
        d = state.to_dict()
        state2 = ForgettingRecallBalanceState.from_dict(d)
        assert state2.cycle_count == 10
        assert len(state2.history) == 2


# =============================================================================
# Stage 1: 断面抽出
# =============================================================================

class TestExtractForgettingSection:
    def test_none_inputs(self):
        snap = extract_forgetting_section(None, None)
        assert snap.active_count == 0
        assert snap.invisible_count == 0

    def test_with_state(self):
        state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active"),
            MockSeriesRecord("active"),
            MockSeriesRecord("weakening"),
            MockSeriesRecord("fading"),
            MockSeriesRecord("near_invisible"),
            MockSeriesRecord("invisible"),
            MockSeriesRecord("active", is_protected=True),
        ])
        snap = extract_forgetting_section(state, None)
        assert snap.active_count == 3  # 2 active + 1 active protected
        assert snap.weakening_count == 1
        assert snap.fading_count == 1
        assert snap.near_invisible_count == 1
        assert snap.invisible_count == 1
        assert snap.protected_count == 1

    def test_with_result(self):
        result = MockForgettingFixationResult(
            newly_forgotten=2, newly_recovered=1, newly_fixating=3,
        )
        snap = extract_forgetting_section(None, result)
        assert snap.newly_forgotten_count == 2
        assert snap.newly_recovered_count == 1
        assert snap.fixation_sign_count == 3

    def test_combined_state_and_result(self):
        state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active"),
            MockSeriesRecord("invisible"),
        ])
        result = MockForgettingFixationResult(
            newly_forgotten=1, newly_recovered=0, newly_fixating=1,
        )
        snap = extract_forgetting_section(state, result)
        assert snap.active_count == 1
        assert snap.invisible_count == 1
        assert snap.newly_forgotten_count == 1
        assert snap.fixation_sign_count == 1


class TestExtractExternalRecallSection:
    def test_none_input(self):
        snap = extract_external_recall_section(None)
        assert snap.total_count == 0
        assert snap.emotional_count == 0

    def test_with_state(self):
        state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(
                emotional_count=5, contextual_count=3, temporal_count=2,
            ),
            current_candidates=[MockRecallCandidate() for _ in range(10)],
        )
        snap = extract_external_recall_section(state)
        assert snap.emotional_count == 5
        assert snap.contextual_count == 3
        assert snap.temporal_count == 2
        assert snap.total_count == 10

    def test_empty_state(self):
        state = MockMultiPathRecallState()
        snap = extract_external_recall_section(state)
        assert snap.total_count == 0
        assert snap.emotional_count == 0

    def test_no_path_stats(self):
        """path_statsがNoneの場合。"""
        state = MockMultiPathRecallState(
            path_stats=None,
            current_candidates=[MockRecallCandidate()],
        )
        snap = extract_external_recall_section(state)
        assert snap.emotional_count == 0
        assert snap.total_count == 1


class TestExtractSpontaneousRecallSection:
    def test_none_input(self):
        snap = extract_spontaneous_recall_section(None)
        assert snap.total_count == 0
        assert snap.emotion_delta_count == 0

    def test_with_state(self):
        state = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(
                emotion_delta_count=4, motive_assoc_count=3, fluctuation_assoc_count=2,
            ),
            current_candidates=[MockSpontaneousRecallCandidate() for _ in range(9)],
        )
        snap = extract_spontaneous_recall_section(state)
        assert snap.emotion_delta_count == 4
        assert snap.motive_assoc_count == 3
        assert snap.fluctuation_assoc_count == 2
        assert snap.total_count == 9

    def test_empty_state(self):
        state = MockSpontaneousRecallState()
        snap = extract_spontaneous_recall_section(state)
        assert snap.total_count == 0

    def test_no_path_stats(self):
        """path_statsがNoneの場合。"""
        state = MockSpontaneousRecallState(
            path_stats=None,
            current_candidates=[MockSpontaneousRecallCandidate()],
        )
        snap = extract_spontaneous_recall_section(state)
        assert snap.emotion_delta_count == 0
        assert snap.total_count == 1


# =============================================================================
# Stage 2: 並置記述の生成
# =============================================================================

class TestComposeJuxtaposition:
    def test_basic(self):
        f = ForgettingSectionSnapshot(active_count=5)
        er = ExternalRecallSectionSnapshot(total_count=3)
        sr = SpontaneousRecallSectionSnapshot(total_count=2)
        entry = compose_juxtaposition(f, er, sr, timestamp=1000.0)
        assert entry.timestamp == 1000.0
        assert entry.forgetting.active_count == 5
        assert entry.external_recall.total_count == 3
        assert entry.spontaneous_recall.total_count == 2

    def test_auto_timestamp(self):
        f = ForgettingSectionSnapshot()
        er = ExternalRecallSectionSnapshot()
        sr = SpontaneousRecallSectionSnapshot()
        before = time.time()
        entry = compose_juxtaposition(f, er, sr)
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_three_sections_are_equal(self):
        """安全弁1: 3断面は構造的に等価。序列を持たない。"""
        f = ForgettingSectionSnapshot(active_count=10)
        er = ExternalRecallSectionSnapshot(total_count=5)
        sr = SpontaneousRecallSectionSnapshot(total_count=3)
        entry = compose_juxtaposition(f, er, sr)
        # 全断面が独立して存在することを確認
        assert entry.forgetting is not None
        assert entry.external_recall is not None
        assert entry.spontaneous_recall is not None


# =============================================================================
# Stage 3: FIFO蓄積
# =============================================================================

class TestAccumulateEntry:
    def test_basic(self):
        state = create_forgetting_recall_balance_state()
        entry = JuxtapositionEntry(timestamp=1000.0)
        cfg = ForgettingRecallBalanceConfig()
        new_state = accumulate_entry(state, entry, cfg)
        assert new_state.cycle_count == 1
        assert len(new_state.history) == 1

    def test_fifo_trimming(self):
        """安全弁4: 保持上限を超えた最古の記述はFIFOで自然消失する。"""
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(timestamp=float(i)) for i in range(5)],
            cycle_count=5,
        )
        cfg = ForgettingRecallBalanceConfig(max_history=5)
        entry = JuxtapositionEntry(timestamp=100.0)
        new_state = accumulate_entry(state, entry, cfg)
        assert len(new_state.history) == 5
        assert new_state.history[0].timestamp == 1.0  # oldest (0.0) was trimmed
        assert new_state.history[-1].timestamp == 100.0

    def test_cycle_count_increment(self):
        state = create_forgetting_recall_balance_state()
        cfg = ForgettingRecallBalanceConfig()
        entry = JuxtapositionEntry()
        state2 = accumulate_entry(state, entry, cfg)
        assert state2.cycle_count == 1
        state3 = accumulate_entry(state2, entry, cfg)
        assert state3.cycle_count == 2

    def test_immutability_of_previous_state(self):
        """元の状態は変更されない。"""
        state = create_forgetting_recall_balance_state()
        cfg = ForgettingRecallBalanceConfig()
        entry = JuxtapositionEntry()
        new_state = accumulate_entry(state, entry, cfg)
        assert len(state.history) == 0
        assert len(new_state.history) == 1


# =============================================================================
# Main Processing
# =============================================================================

class TestProcessForgettingRecallBalance:
    def test_empty_inputs(self, empty_state):
        new_state = process_forgetting_recall_balance(
            empty_state, timestamp=1000.0,
        )
        assert new_state.cycle_count == 1
        assert len(new_state.history) == 1
        entry = new_state.history[0]
        assert entry.forgetting.active_count == 0
        assert entry.external_recall.total_count == 0
        assert entry.spontaneous_recall.total_count == 0

    def test_with_all_inputs(self, empty_state):
        fg_state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active"),
            MockSeriesRecord("active"),
            MockSeriesRecord("weakening"),
            MockSeriesRecord("invisible"),
        ])
        fg_result = MockForgettingFixationResult(
            newly_forgotten=1, newly_recovered=0, newly_fixating=2,
        )
        mr_state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(
                emotional_count=3, contextual_count=2, temporal_count=1,
            ),
            current_candidates=[MockRecallCandidate() for _ in range(6)],
        )
        sr_state = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(
                emotion_delta_count=2, motive_assoc_count=1, fluctuation_assoc_count=0,
            ),
            current_candidates=[MockSpontaneousRecallCandidate() for _ in range(3)],
        )

        new_state = process_forgetting_recall_balance(
            empty_state,
            forgetting_state=fg_state,
            forgetting_result=fg_result,
            multi_path_recall_state=mr_state,
            spontaneous_recall_state=sr_state,
            timestamp=2000.0,
        )

        assert new_state.cycle_count == 1
        assert len(new_state.history) == 1
        entry = new_state.history[0]

        # 忘却断面
        assert entry.forgetting.active_count == 2
        assert entry.forgetting.weakening_count == 1
        assert entry.forgetting.invisible_count == 1
        assert entry.forgetting.newly_forgotten_count == 1
        assert entry.forgetting.fixation_sign_count == 2

        # 外部トリガー型想起断面
        assert entry.external_recall.emotional_count == 3
        assert entry.external_recall.contextual_count == 2
        assert entry.external_recall.temporal_count == 1
        assert entry.external_recall.total_count == 6

        # 自発的想起断面
        assert entry.spontaneous_recall.emotion_delta_count == 2
        assert entry.spontaneous_recall.motive_assoc_count == 1
        assert entry.spontaneous_recall.total_count == 3

    def test_multiple_cycles(self, empty_state):
        state = empty_state
        for i in range(5):
            state = process_forgetting_recall_balance(
                state, timestamp=float(i * 1000),
            )
        assert state.cycle_count == 5
        assert len(state.history) == 5

    def test_fifo_on_processing(self, empty_state):
        """安全弁4: FIFO蓄積。"""
        cfg = ForgettingRecallBalanceConfig(max_history=3)
        state = empty_state
        for i in range(5):
            state = process_forgetting_recall_balance(
                state, config=cfg, timestamp=float(i * 1000),
            )
        assert state.cycle_count == 5
        assert len(state.history) == 3
        # 最古は2000.0 (0, 1000 が消失)
        assert state.history[0].timestamp == 2000.0

    def test_no_write_to_forgetting(self, empty_state):
        """遮断1: 忘却パラメータへの書き込み遮断。"""
        fg_state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active"),
        ])
        original_stage = fg_state.series_index[0].forgetting_stage
        process_forgetting_recall_balance(
            empty_state, forgetting_state=fg_state,
        )
        # 忘却状態が変更されていないことを確認
        assert fg_state.series_index[0].forgetting_stage == original_stage

    def test_no_write_to_recall(self, empty_state):
        """遮断2: 想起閾値への書き込み遮断。"""
        mr_state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(emotional_count=5),
            current_candidates=[MockRecallCandidate()],
        )
        original_count = mr_state.path_stats.emotional_count
        original_candidates = len(mr_state.current_candidates)
        process_forgetting_recall_balance(
            empty_state, multi_path_recall_state=mr_state,
        )
        # 想起状態が変更されていないことを確認
        assert mr_state.path_stats.emotional_count == original_count
        assert len(mr_state.current_candidates) == original_candidates

    def test_no_pattern_extraction(self, empty_state):
        """安全弁3: パターン抽出の禁止。状態にパターン蓄積フィールドがないことを確認。"""
        state = empty_state
        for i in range(10):
            state = process_forgetting_recall_balance(state)
        # stateにはhistoryとcycle_count以外のフィールドがない
        d = state.to_dict()
        assert set(d.keys()) == {"history", "cycle_count"}


# =============================================================================
# READ-ONLY Accessors
# =============================================================================

class TestGetRecentEntries:
    def test_empty(self, empty_state):
        entries = get_recent_entries(empty_state)
        assert entries == []

    def test_returns_copies(self, empty_state):
        """参照行為によって状態が変化しない。"""
        state = process_forgetting_recall_balance(empty_state, timestamp=1000.0)
        entries = get_recent_entries(state)
        assert len(entries) == 1
        entries.clear()
        assert len(get_recent_entries(state)) == 1

    def test_limited_count(self):
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(timestamp=float(i)) for i in range(10)],
            cycle_count=10,
        )
        entries = get_recent_entries(state, count=3)
        assert len(entries) == 3
        assert entries[0].timestamp == 7.0
        assert entries[-1].timestamp == 9.0

    def test_all_entries_equal(self):
        """安全弁1: 全件が等価。"""
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(timestamp=float(i)) for i in range(5)],
            cycle_count=5,
        )
        entries = get_recent_entries(state, count=5)
        # 全エントリが同じ型であること
        for entry in entries:
            assert isinstance(entry, JuxtapositionEntry)


class TestGetHistory:
    def test_empty(self, empty_state):
        history = get_history(empty_state)
        assert history == []

    def test_returns_copies(self):
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(timestamp=1000.0)],
            cycle_count=1,
        )
        history = get_history(state)
        assert len(history) == 1
        history.clear()
        assert len(get_history(state)) == 1


class TestGetBalanceSummary:
    def test_empty(self, empty_state):
        summary = get_balance_summary(empty_state)
        assert summary["cycle_count"] == 0
        assert summary["history_count"] == 0
        assert "latest_forgetting" not in summary

    def test_with_data(self):
        state = ForgettingRecallBalanceState(
            history=[
                JuxtapositionEntry(
                    timestamp=1000.0,
                    forgetting=ForgettingSectionSnapshot(active_count=5),
                    external_recall=ExternalRecallSectionSnapshot(total_count=3),
                    spontaneous_recall=SpontaneousRecallSectionSnapshot(total_count=2),
                ),
            ],
            cycle_count=1,
        )
        summary = get_balance_summary(state)
        assert summary["cycle_count"] == 1
        assert summary["history_count"] == 1
        assert summary["latest_forgetting"]["active_count"] == 5
        assert summary["latest_external_recall"]["total_count"] == 3
        assert summary["latest_spontaneous_recall"]["total_count"] == 2
        assert summary["latest_timestamp"] == 1000.0


# =============================================================================
# Enrichment
# =============================================================================

class TestGetEnrichmentText:
    def test_empty(self, empty_state):
        text = get_enrichment_text(empty_state)
        assert "待機中" in text

    def test_with_data(self):
        state = ForgettingRecallBalanceState(
            history=[
                JuxtapositionEntry(
                    timestamp=1000.0,
                    forgetting=ForgettingSectionSnapshot(
                        active_count=10, weakening_count=3, invisible_count=1,
                        fixation_sign_count=2, protected_count=4,
                        newly_forgotten_count=1, newly_recovered_count=0,
                    ),
                    external_recall=ExternalRecallSectionSnapshot(
                        emotional_count=5, contextual_count=3, temporal_count=2,
                        total_count=10,
                    ),
                    spontaneous_recall=SpontaneousRecallSectionSnapshot(
                        emotion_delta_count=4, motive_assoc_count=2,
                        fluctuation_assoc_count=1, total_count=7,
                    ),
                ),
            ],
            cycle_count=1,
        )
        text = get_enrichment_text(state)
        assert "cycle=1" in text
        assert "活性=10" in text
        assert "弱体化=3" in text
        assert "不可視=1" in text
        assert "固定化兆候=2" in text
        assert "保護=4" in text
        assert "新規忘却=1" in text
        assert "外部想起" in text
        assert "感情=5" in text
        assert "文脈=3" in text
        assert "時間=2" in text
        assert "自発想起" in text
        assert "感情変動=4" in text
        assert "動機=2" in text
        assert "揺らぎ=1" in text

    def test_no_evaluative_language(self):
        """安全弁2: 評価的変換の禁止。"""
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(
                forgetting=ForgettingSectionSnapshot(active_count=1, invisible_count=100),
                external_recall=ExternalRecallSectionSnapshot(total_count=0),
            )],
            cycle_count=1,
        )
        text = get_enrichment_text(state)
        # 評価的語彙が含まれないことを確認
        forbidden_words = [
            "均衡", "不均衡", "優勢", "不足", "速すぎ", "遅すぎ",
            "多すぎ", "少なすぎ", "問題", "改善", "悪化",
        ]
        for word in forbidden_words:
            assert word not in text, f"Evaluative word found: {word}"

    def test_only_numeric_counts(self):
        """安全弁2: 値は全て件数のみの記述。"""
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(
                forgetting=ForgettingSectionSnapshot(active_count=5, weakening_count=2),
                external_recall=ExternalRecallSectionSnapshot(
                    emotional_count=3, total_count=3,
                ),
                spontaneous_recall=SpontaneousRecallSectionSnapshot(
                    emotion_delta_count=1, total_count=1,
                ),
            )],
            cycle_count=1,
        )
        text = get_enrichment_text(state)
        # 件数的記述のみが含まれる（比率・統計量を含まない）
        assert "%" not in text
        assert "率" not in text
        assert "比" not in text

    def test_zero_counts_omitted(self):
        """件数がゼロの項目は出力されない。"""
        state = ForgettingRecallBalanceState(
            history=[JuxtapositionEntry(
                forgetting=ForgettingSectionSnapshot(active_count=5),
                external_recall=ExternalRecallSectionSnapshot(),
                spontaneous_recall=SpontaneousRecallSectionSnapshot(),
            )],
            cycle_count=1,
        )
        text = get_enrichment_text(state)
        assert "活性=5" in text
        assert "外部想起" not in text
        assert "自発想起" not in text


# =============================================================================
# Save / Load
# =============================================================================

class TestSaveLoad:
    def test_save_empty(self, empty_state):
        d = save_state(empty_state)
        assert d["cycle_count"] == 0
        assert d["history"] == []

    def test_load_empty(self):
        state = load_state({"cycle_count": 0, "history": []})
        assert state.cycle_count == 0
        assert len(state.history) == 0

    def test_roundtrip(self):
        state = ForgettingRecallBalanceState(
            history=[
                JuxtapositionEntry(
                    timestamp=1000.0,
                    forgetting=ForgettingSectionSnapshot(
                        active_count=10, weakening_count=3,
                    ),
                    external_recall=ExternalRecallSectionSnapshot(
                        emotional_count=5, total_count=8,
                    ),
                    spontaneous_recall=SpontaneousRecallSectionSnapshot(
                        emotion_delta_count=2, total_count=4,
                    ),
                ),
                JuxtapositionEntry(
                    timestamp=2000.0,
                    forgetting=ForgettingSectionSnapshot(
                        active_count=8, invisible_count=2,
                    ),
                ),
            ],
            cycle_count=2,
        )
        d = save_state(state)
        state2 = load_state(d)
        assert state2.cycle_count == 2
        assert len(state2.history) == 2
        assert state2.history[0].forgetting.active_count == 10
        assert state2.history[1].forgetting.invisible_count == 2

    def test_load_missing_keys(self):
        """不完全なデータからの復元。"""
        state = load_state({})
        assert state.cycle_count == 0
        assert len(state.history) == 0

    def test_load_partial_entry(self):
        """部分的なエントリデータからの復元。"""
        state = load_state({
            "cycle_count": 1,
            "history": [{"timestamp": 500.0}],
        })
        assert state.cycle_count == 1
        assert len(state.history) == 1
        assert state.history[0].timestamp == 500.0
        assert state.history[0].forgetting.active_count == 0


# =============================================================================
# Safety Valves (安全弁)
# =============================================================================

class TestSafetyValves:
    def test_sv1_all_records_equal(self):
        """安全弁1: 全記録等価維持保証。"""
        state = create_forgetting_recall_balance_state()
        for i in range(10):
            state = process_forgetting_recall_balance(state, timestamp=float(i))
        # 全エントリが同じ構造を持つ
        for entry in state.history:
            assert hasattr(entry, "forgetting")
            assert hasattr(entry, "external_recall")
            assert hasattr(entry, "spontaneous_recall")
            assert hasattr(entry, "timestamp")

    def test_sv2_no_evaluative_conversion(self):
        """安全弁2: 評価的変換の禁止。"""
        # 断面はすべて件数のみ
        snap = ForgettingSectionSnapshot(active_count=1, invisible_count=100)
        d = snap.to_dict()
        for key in d:
            assert isinstance(d[key], int), f"Non-integer field: {key}={d[key]}"

    def test_sv3_no_pattern_extraction(self):
        """安全弁3: パターン抽出の禁止。"""
        state = create_forgetting_recall_balance_state()
        for i in range(20):
            state = process_forgetting_recall_balance(state)
        # stateにパターン累積フィールドがないことを確認
        d = state.to_dict()
        assert "pattern" not in str(d).lower()
        assert "trend" not in str(d).lower()
        assert "statistics" not in str(d).lower()

    def test_sv4_fifo_bounded(self):
        """安全弁4: 履歴の有限性。"""
        cfg = ForgettingRecallBalanceConfig(max_history=5)
        state = create_forgetting_recall_balance_state()
        for i in range(100):
            state = process_forgetting_recall_balance(state, config=cfg)
        assert len(state.history) <= 5

    def test_sv5_output_limited(self):
        """安全弁5: 出力経路の限定と不拡張。"""
        # 出力関数は get_enrichment_text, get_recent_entries, get_history,
        # get_balance_summary のみ
        # これら以外の出力関数がモジュールに存在しないことを確認
        from psyche import forgetting_recall_balance as mod
        import inspect
        public_funcs = [
            name for name in dir(mod)
            if not name.startswith("_")
            and callable(getattr(mod, name))
            and (
                getattr(getattr(mod, name), "__module__", None)
                == mod.__name__
            )
        ]
        # 既知の安全な出力関数群
        allowed_funcs = {
            "extract_forgetting_section",
            "extract_external_recall_section",
            "extract_spontaneous_recall_section",
            "compose_juxtaposition",
            "accumulate_entry",
            "process_forgetting_recall_balance",
            "get_recent_entries",
            "get_history",
            "get_balance_summary",
            "get_enrichment_text",
            "save_state",
            "load_state",
            "create_forgetting_recall_balance_state",
            "create_forgetting_recall_balance_config",
        }
        # データクラス・設定クラスも許容
        allowed_classes = {
            "ForgettingRecallBalanceConfig",
            "ForgettingSectionSnapshot",
            "ExternalRecallSectionSnapshot",
            "SpontaneousRecallSectionSnapshot",
            "JuxtapositionEntry",
            "ForgettingRecallBalanceState",
        }
        for name in public_funcs:
            obj = getattr(mod, name)
            if isinstance(obj, type):
                assert name in allowed_classes, f"Unexpected class: {name}"
            else:
                assert name in allowed_funcs, f"Unexpected function: {name}"


# =============================================================================
# 遮断の検証
# =============================================================================

class TestIsolation:
    def test_forgetting_parameter_isolation(self):
        """遮断1: 忘却パラメータへの非書き込みの保証。"""
        fg_state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active"),
            MockSeriesRecord("weakening"),
        ])
        original_stages = [
            rec.forgetting_stage for rec in fg_state.series_index
        ]
        state = create_forgetting_recall_balance_state()
        process_forgetting_recall_balance(
            state, forgetting_state=fg_state,
        )
        # 忘却状態が変更されていないことを確認
        for i, rec in enumerate(fg_state.series_index):
            assert rec.forgetting_stage == original_stages[i]

    def test_recall_threshold_isolation(self):
        """遮断2: 想起閾値への非書き込みの保証。"""
        mr_state = MockMultiPathRecallState(
            path_stats=MockPathStatistics(emotional_count=5),
            current_candidates=[MockRecallCandidate() for _ in range(3)],
        )
        sr_state = MockSpontaneousRecallState(
            path_stats=MockSpontaneousRecallPathStatistics(emotion_delta_count=2),
            current_candidates=[MockSpontaneousRecallCandidate() for _ in range(2)],
        )
        state = create_forgetting_recall_balance_state()
        process_forgetting_recall_balance(
            state,
            multi_path_recall_state=mr_state,
            spontaneous_recall_state=sr_state,
        )
        # 想起状態が変更されていないことを確認
        assert mr_state.path_stats.emotional_count == 5
        assert len(mr_state.current_candidates) == 3
        assert sr_state.path_stats.emotion_delta_count == 2
        assert len(sr_state.current_candidates) == 2

    def test_no_cycle_reference(self):
        """循環遮断: 出力が入力元に逆流しない。"""
        state = create_forgetting_recall_balance_state()
        # processの戻り値はForgettingRecallBalanceState のみ
        new_state = process_forgetting_recall_balance(state)
        assert isinstance(new_state, ForgettingRecallBalanceState)
        # stateに忘却・想起の制御パラメータが存在しない
        d = new_state.to_dict()
        assert "dilution" not in str(d)
        assert "threshold" not in str(d)
        assert "rumination" not in str(d)

    def test_previous_entry_does_not_affect_current(self):
        """可逆性: 前回の記述が次回の記述の内容を制約しない。"""
        state = create_forgetting_recall_balance_state()
        # 1回目: 忘却側にデータあり
        fg_state = MockForgettingFixationState(series_index=[
            MockSeriesRecord("active") for _ in range(10)
        ])
        state = process_forgetting_recall_balance(
            state, forgetting_state=fg_state, timestamp=1000.0,
        )
        # 2回目: 忘却側にデータなし
        state = process_forgetting_recall_balance(
            state, timestamp=2000.0,
        )
        # 2回目のエントリは1回目の影響を受けない
        assert state.history[-1].forgetting.active_count == 0
        assert state.history[-2].forgetting.active_count == 10
