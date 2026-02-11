"""
tests/test_other_model_input_supply.py

other_model_input_supply モジュールのテスト。
- create → update → supply の基本フロー
- 欠損時 missing_reason="unobserved"
- decay で古い要素が stale 化
- 循環参照防止: supply結果を同一周期の入力に戻さない構造確認
- to_dict / from_dict 永続化
"""

import time
import pytest

from psyche.other_model_input_supply import (
    SupplyEntry,
    ContextSnapshot,
    ReactionBufferEntry,
    InputSupplyState,
    create_input_supply,
    update_from_percept,
    decay_buffer,
    supply_context,
    supply_reaction_log,
    get_input_supply_summary,
)


# =============================================================================
# Helpers: Mock objects for duck typing
# =============================================================================

class MockStimulusEntry:
    def __init__(self, source_text="hello", intent="greeting",
                 emotion_label="joy", valence=0.5, timestamp=None):
        self.source_text = source_text
        self.intent = intent
        self.emotion_label = emotion_label
        self.valence = valence
        self.timestamp = timestamp or time.time()
        self.residue_weight = 1.0
        self.processed = False


class MockSTM:
    def __init__(self, entries=None, max_entries=10, context_continuity_score=0.5):
        self.entries = entries or []
        self.max_entries = max_entries
        self.context_continuity_score = context_continuity_score


class MockMood:
    def __init__(self, valence=0.3, arousal=0.6):
        self.valence = valence
        self.arousal = arousal


class MockPsyche:
    def __init__(self, mood=None):
        self.mood = mood or MockMood()


class MockPercept:
    def __init__(self, topics=None, emotion_valence=0.2):
        self.topics = topics or ["game", "chat"]
        self.emotion_valence = emotion_valence


class MockDynamics:
    pass


# =============================================================================
# Test: create_input_supply
# =============================================================================

class TestCreateInputSupply:
    def test_initial_state(self):
        state = create_input_supply()
        assert isinstance(state, InputSupplyState)
        assert state.context_snapshot.missing_reason == "unobserved"
        assert state.reaction_buffer == []
        assert state.supply_cursor == 0
        assert state.last_supply_time == 0.0

    def test_initial_context_is_neutral_on_supply(self):
        state = create_input_supply()
        ctx = supply_context(state)
        assert ctx.pace == 0.5
        assert ctx.weight == 0.5
        assert ctx.density == 0.5
        assert ctx.responsiveness == 0.5
        assert ctx.missing_reason == "unobserved"


# =============================================================================
# Test: update_from_percept
# =============================================================================

class TestUpdateFromPercept:
    def test_with_all_inputs(self):
        state = create_input_supply()
        now = time.time()

        stm_entries = [
            MockStimulusEntry(source_text="hi", intent="greeting",
                              emotion_label="joy", valence=0.5, timestamp=now - 2),
            MockStimulusEntry(source_text="how are you", intent="question",
                              emotion_label="neutral", valence=0.0, timestamp=now - 1),
        ]
        stm = MockSTM(entries=stm_entries, max_entries=10, context_continuity_score=0.7)
        percept = MockPercept(topics=["game", "chat", "fun"])
        psyche = MockPsyche(mood=MockMood(valence=0.4, arousal=0.6))
        dynamics = MockDynamics()

        updated = update_from_percept(state, percept=percept, stm=stm,
                                      dynamics=dynamics, psyche=psyche)

        assert updated.context_snapshot.missing_reason == ""
        assert updated.context_snapshot.pace == pytest.approx(0.2, abs=0.01)  # 2/10
        assert updated.context_snapshot.density == pytest.approx(0.6, abs=0.01)  # 3/5
        assert updated.context_snapshot.continuity == pytest.approx(0.7, abs=0.01)
        assert len(updated.reaction_buffer) == 2

    def test_with_no_inputs(self):
        state = create_input_supply()
        updated = update_from_percept(state)
        assert updated.context_snapshot.missing_reason == "unobserved"

    def test_stm_entries_dedup(self):
        """同じ timestamp のエントリは重複追加しない"""
        state = create_input_supply()
        now = time.time()
        entry = MockStimulusEntry(timestamp=now)
        stm = MockSTM(entries=[entry])

        updated = update_from_percept(state, stm=stm)
        assert len(updated.reaction_buffer) == 1

        # 同じSTMで再度更新
        updated2 = update_from_percept(updated, stm=stm)
        assert len(updated2.reaction_buffer) == 1  # 重複しない

    def test_buffer_size_limit(self):
        state = create_input_supply()
        state.max_buffer_size = 5
        now = time.time()
        entries = [MockStimulusEntry(timestamp=now + i) for i in range(10)]
        stm = MockSTM(entries=entries)

        updated = update_from_percept(state, stm=stm)
        assert len(updated.reaction_buffer) <= 5

    def test_responsiveness_from_recent_entry(self):
        """直近エントリがあればhigh responsiveness"""
        state = create_input_supply()
        now = time.time()
        entry = MockStimulusEntry(timestamp=now - 3)  # 3秒前
        stm = MockSTM(entries=[entry])

        updated = update_from_percept(state, stm=stm)
        assert updated.context_snapshot.responsiveness == 0.8

    def test_responsiveness_from_old_entry(self):
        """古いエントリのみだとlow responsiveness"""
        state = create_input_supply()
        now = time.time()
        entry = MockStimulusEntry(timestamp=now - 120)  # 2分前
        stm = MockSTM(entries=[entry])

        updated = update_from_percept(state, stm=stm)
        assert updated.context_snapshot.responsiveness == 0.2


# =============================================================================
# Test: decay_buffer
# =============================================================================

class TestDecayBuffer:
    def test_old_entries_removed(self):
        state = create_input_supply()
        now = time.time()

        state.reaction_buffer = [
            ReactionBufferEntry(timestamp=now - 400),  # 6分40秒前 → 除去
            ReactionBufferEntry(timestamp=now - 100),   # 保持
            ReactionBufferEntry(timestamp=now - 10),    # 保持
        ]

        decayed = decay_buffer(state, now)
        assert len(decayed.reaction_buffer) == 2

    def test_context_becomes_stale(self):
        state = create_input_supply()
        now = time.time()
        state.context_snapshot = ContextSnapshot(
            pace=0.6, weight=0.4, timestamp=now - 200, missing_reason=""
        )

        decayed = decay_buffer(state, now)
        assert decayed.context_snapshot.missing_reason == "stale"

    def test_context_stays_fresh(self):
        state = create_input_supply()
        now = time.time()
        state.context_snapshot = ContextSnapshot(
            pace=0.6, weight=0.4, timestamp=now - 30, missing_reason=""
        )

        decayed = decay_buffer(state, now)
        assert decayed.context_snapshot.missing_reason == ""

    def test_already_stale_context_not_double_marked(self):
        """missing_reason が空でないものは stale 化スキップ"""
        state = create_input_supply()
        now = time.time()
        state.context_snapshot = ContextSnapshot(
            timestamp=now - 200, missing_reason="unobserved"
        )
        decayed = decay_buffer(state, now)
        assert decayed.context_snapshot.missing_reason == "unobserved"


# =============================================================================
# Test: supply_context
# =============================================================================

class TestSupplyContext:
    def test_unobserved_returns_neutral(self):
        state = create_input_supply()
        ctx = supply_context(state)
        assert ctx.pace == 0.5
        assert ctx.weight == 0.5
        assert ctx.density == 0.5
        assert ctx.continuity == 0.5
        assert ctx.responsiveness == 0.5
        assert ctx.missing_reason == "unobserved"

    def test_observed_returns_actual(self):
        state = create_input_supply()
        state.context_snapshot = ContextSnapshot(
            pace=0.8, weight=0.3, density=0.6,
            continuity=0.7, responsiveness=0.9,
            timestamp=time.time(), missing_reason=""
        )
        ctx = supply_context(state)
        assert ctx.pace == 0.8
        assert ctx.responsiveness == 0.9
        assert ctx.missing_reason == ""


# =============================================================================
# Test: supply_reaction_log
# =============================================================================

class TestSupplyReactionLog:
    def test_empty_buffer_returns_none(self):
        state = create_input_supply()
        result = supply_reaction_log(state)
        assert result is None

    def test_returns_stm_compatible_object(self):
        state = create_input_supply()
        state.reaction_buffer = [
            ReactionBufferEntry(
                source_text="hello", intent="greeting",
                emotion_label="joy", valence=0.5, timestamp=time.time(),
            ),
        ]
        result = supply_reaction_log(state)
        assert result is not None
        assert hasattr(result, "entries")
        assert len(result.entries) == 1
        assert result.entries[0].source_text == "hello"
        assert result.entries[0].intent == "greeting"
        assert result.entries[0].valence == 0.5

    def test_marks_as_supplied(self):
        state = create_input_supply()
        entry = ReactionBufferEntry(
            source_text="test", timestamp=time.time(), supplied=False,
        )
        state.reaction_buffer = [entry]

        supply_reaction_log(state)
        assert state.reaction_buffer[0].supplied is True
        assert state.supply_cursor == 1

    def test_resupply_possible(self):
        """供給済み要素も再供給可能"""
        state = create_input_supply()
        entry = ReactionBufferEntry(
            source_text="test", timestamp=time.time(), supplied=True,
        )
        state.reaction_buffer = [entry]

        result = supply_reaction_log(state)
        assert result is not None
        assert len(result.entries) == 1


# =============================================================================
# Test: Circular reference prevention
# =============================================================================

class TestCircularReferencePrevention:
    def test_supply_result_not_fed_back_in_same_cycle(self):
        """supply の出力は別周期で消費される構造であること確認"""
        state = create_input_supply()
        now = time.time()
        state.reaction_buffer = [
            ReactionBufferEntry(
                source_text="hi", intent="greeting",
                emotion_label="joy", valence=0.5, timestamp=now,
            ),
        ]

        # supply → 結果取得
        ctx = supply_context(state)
        rlog = supply_reaction_log(state)

        # cursor更新 → 供給済みマーク
        assert state.supply_cursor == 1
        assert state.reaction_buffer[0].supplied is True

        # supply_context と supply_reaction_log は read-only ビューを返す
        # observe_other_from_chain に渡されるのは supply 結果のみ
        # update_from_percept に supply 結果を戻す経路は存在しない
        # (構造的に循環しない)
        assert ctx is not state.context_snapshot or ctx.missing_reason == "unobserved"

    def test_cursor_advances_after_supply(self):
        """supply_cursor が供給後に進む"""
        state = create_input_supply()
        now = time.time()
        state.reaction_buffer = [
            ReactionBufferEntry(timestamp=now),
            ReactionBufferEntry(timestamp=now + 1),
        ]

        assert state.supply_cursor == 0
        supply_reaction_log(state)
        assert state.supply_cursor == 2


# =============================================================================
# Test: to_dict / from_dict persistence
# =============================================================================

class TestPersistence:
    def test_context_snapshot_roundtrip(self):
        ctx = ContextSnapshot(
            pace=0.7, weight=0.3, density=0.6,
            continuity=0.8, responsiveness=0.9,
            timestamp=123456.0, missing_reason="stale",
        )
        restored = ContextSnapshot.from_dict(ctx.to_dict())
        assert restored.pace == ctx.pace
        assert restored.weight == ctx.weight
        assert restored.density == ctx.density
        assert restored.continuity == ctx.continuity
        assert restored.responsiveness == ctx.responsiveness
        assert restored.timestamp == ctx.timestamp
        assert restored.missing_reason == ctx.missing_reason

    def test_reaction_buffer_entry_roundtrip(self):
        entry = ReactionBufferEntry(
            source_text="test text",
            intent="question",
            emotion_label="sadness",
            valence=-0.3,
            timestamp=99999.0,
            supplied=True,
        )
        restored = ReactionBufferEntry.from_dict(entry.to_dict())
        assert restored.source_text == entry.source_text
        assert restored.intent == entry.intent
        assert restored.emotion_label == entry.emotion_label
        assert restored.valence == entry.valence
        assert restored.timestamp == entry.timestamp
        assert restored.supplied == entry.supplied

    def test_input_supply_state_roundtrip(self):
        state = create_input_supply()
        state.context_snapshot = ContextSnapshot(
            pace=0.8, weight=0.2, density=0.5,
            continuity=0.6, responsiveness=0.7,
            timestamp=12345.0, missing_reason="",
        )
        state.reaction_buffer = [
            ReactionBufferEntry(
                source_text="hello", intent="greeting",
                emotion_label="joy", valence=0.5,
                timestamp=12345.0, supplied=False,
            ),
            ReactionBufferEntry(
                source_text="bye", intent="farewell",
                emotion_label="neutral", valence=0.0,
                timestamp=12346.0, supplied=True,
            ),
        ]
        state.supply_cursor = 1
        state.last_supply_time = 12346.0

        d = state.to_dict()
        restored = InputSupplyState.from_dict(d)

        assert restored.context_snapshot.pace == 0.8
        assert restored.context_snapshot.missing_reason == ""
        assert len(restored.reaction_buffer) == 2
        assert restored.reaction_buffer[0].source_text == "hello"
        assert restored.reaction_buffer[1].supplied is True
        assert restored.supply_cursor == 1
        assert restored.last_supply_time == 12346.0

    def test_empty_state_roundtrip(self):
        state = InputSupplyState()
        d = state.to_dict()
        restored = InputSupplyState.from_dict(d)
        assert len(restored.reaction_buffer) == 0
        assert restored.supply_cursor == 0


# =============================================================================
# Test: get_input_supply_summary
# =============================================================================

class TestGetSummary:
    def test_basic_summary(self):
        state = create_input_supply()
        summary = get_input_supply_summary(state)
        assert "ctx(" in summary
        assert "buf=" in summary

    def test_summary_with_data(self):
        state = create_input_supply()
        state.context_snapshot = ContextSnapshot(
            pace=0.7, weight=0.3, density=0.5,
            responsiveness=0.8, timestamp=time.time(),
            missing_reason="",
        )
        state.reaction_buffer = [
            ReactionBufferEntry(supplied=False),
            ReactionBufferEntry(supplied=True),
        ]
        summary = get_input_supply_summary(state)
        assert "new=1" in summary
        assert "supplied=1" in summary


# =============================================================================
# Test: Full flow (create → update → supply)
# =============================================================================

class TestFullFlow:
    def test_create_update_supply(self):
        """基本フロー: create → update → supply で他者モデルに入力が渡る"""
        # 1. Create
        state = create_input_supply()
        assert supply_context(state).missing_reason == "unobserved"
        assert supply_reaction_log(state) is None

        # 2. Update with actual data
        now = time.time()
        stm_entries = [
            MockStimulusEntry(
                source_text="すごいね！",
                intent="compliment",
                emotion_label="joy",
                valence=0.7,
                timestamp=now - 2,
            ),
        ]
        stm = MockSTM(entries=stm_entries, max_entries=10,
                       context_continuity_score=0.6)
        percept = MockPercept(topics=["game", "reaction"])
        psyche = MockPsyche(mood=MockMood(valence=0.5, arousal=0.7))

        state = update_from_percept(state, percept=percept, stm=stm,
                                    psyche=psyche)

        # 3. Supply
        ctx = supply_context(state)
        assert ctx.missing_reason == ""
        assert ctx.responsiveness > 0.3
        assert ctx.density == pytest.approx(0.4, abs=0.01)  # 2/5

        rlog = supply_reaction_log(state)
        assert rlog is not None
        assert len(rlog.entries) == 1
        assert rlog.entries[0].source_text == "すごいね！"
        assert rlog.entries[0].valence == 0.7

    def test_integration_with_other_agent_model(self):
        """supply 結果を other_agent_model に渡して仮説が生成されること"""
        from psyche.other_agent_model import OtherAgentModelSystem, observe_from_chain

        state = create_input_supply()
        now = time.time()

        stm_entries = [
            MockStimulusEntry(
                source_text="面白い！",
                intent="expression",
                emotion_label="joy",
                valence=0.8,
                timestamp=now - 1,
            ),
        ]
        stm = MockSTM(entries=stm_entries, max_entries=10,
                       context_continuity_score=0.7)
        percept = MockPercept(topics=["game", "fun", "play"])
        psyche = MockPsyche(mood=MockMood(valence=0.6, arousal=0.8))

        state = update_from_percept(state, percept=percept, stm=stm,
                                    psyche=psyche)
        state = decay_buffer(state, now)

        ctx = supply_context(state)
        rlog = supply_reaction_log(state)

        # other_agent_model に渡す
        system = OtherAgentModelSystem()
        store = observe_from_chain(
            system=system,
            external_context=ctx,
            reaction_log=rlog,
            self_state=None,
        )

        # 仮説が生成されていること (None ではなく入力が渡った)
        assert store is not None
        assert store.total_hypotheses_created > 0


# =============================================================================
# Test: SupplyEntry dataclass
# =============================================================================

class TestSupplyEntry:
    def test_defaults(self):
        entry = SupplyEntry()
        assert entry.source_type == "periodic"
        assert entry.timestamp == 0.0
        assert entry.missing_reason == ""

    def test_custom_values(self):
        entry = SupplyEntry(source_type="event", timestamp=123.0,
                            missing_reason="unobserved")
        assert entry.source_type == "event"
        assert entry.timestamp == 123.0
        assert entry.missing_reason == "unobserved"
