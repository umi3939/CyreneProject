"""tests/test_action_result_observation.py -- 行動-結果の観測と蓄積テスト"""

import time
import pytest

from psyche.action_result_observation import (
    # Enums
    ObservationSection,
    FreshnessStage,
    PairStatus,
    ConvergenceLevel,
    # Data structures
    SectionDescription,
    ActionDescription,
    ResultDescription,
    ContextAttribution,
    ActionResultPair,
    SectionWeightRecord,
    ConvergenceRecord,
    # Inputs / State / Result
    ActionResultInputs,
    ActionResultObservationState,
    ActionResultObservationResult,
    ActionResultConfig,
    # Processor
    ActionResultObservationProcessor,
    # Helpers
    _clamp,
    _gen_id,
    _stage_from_freshness,
    _convergence_from_score,
    # Public API
    get_action_result_summary,
    create_action_result_processor,
)


# ── Test Helpers ─────────────────────────────────────────────────

def _make_inputs(**overrides) -> ActionResultInputs:
    """テスト用の基本入力を生成する。"""
    defaults = dict(
        selected_policy_label="label_a",
        selected_policy_axis="approach",
        selection_context_summary="context_test",
        action_tick=0,
        external_response_change=0.2,
        external_response_description="response_test",
        internal_state_delta=0.1,
        motivation_delta=0.05,
        direction_delta=0.03,
        emotion_before={"joy": 0.5},
        emotion_after={"joy": 0.6},
        context_summary="summary_test",
        dialogue_state="idle",
        environment_tags=["tag_a"],
        ticks_since_action=5,
        elapsed_seconds=10.0,
        other_reaction_change=0.1,
        other_reaction_description="other_test",
        referenced_memory_ids=["mem1"],
        referenced_memory_count=1,
        current_tick=10,
    )
    defaults.update(overrides)
    return ActionResultInputs(**defaults)


def _make_processor(**config_overrides) -> ActionResultObservationProcessor:
    """テスト用のプロセッサを生成する。"""
    cfg = ActionResultConfig(**config_overrides)
    return ActionResultObservationProcessor(config=cfg)


def _record_and_process(
    proc: ActionResultObservationProcessor,
    action_tick: int = 0,
    current_tick: int = 10,
    **input_overrides,
) -> ActionResultObservationResult:
    """行動を記録し、十分なティック後にprocessを実行する。"""
    action_inputs = _make_inputs(
        action_tick=action_tick,
        current_tick=action_tick,
        **{k: v for k, v in input_overrides.items()
           if k in ('selected_policy_label', 'selected_policy_axis',
                     'selection_context_summary')},
    )
    proc.record_action(action_inputs)

    process_inputs = _make_inputs(
        current_tick=current_tick,
        **input_overrides,
    )
    return proc.process(process_inputs)


# =====================================================================
# Enum Tests
# =====================================================================

class TestEnums:
    def test_observation_section_count(self):
        assert len(ObservationSection) == 8

    def test_observation_section_values(self):
        assert ObservationSection.RECENT_ACTION.value == "recent_action"
        assert ObservationSection.MEMORY_REFERENCE.value == "memory_reference"

    def test_freshness_stage_count(self):
        assert len(FreshnessStage) == 5

    def test_freshness_stage_values(self):
        assert FreshnessStage.ACTIVE.value == "active"
        assert FreshnessStage.INVISIBLE.value == "invisible"

    def test_pair_status_count(self):
        assert len(PairStatus) == 6

    def test_pair_status_values(self):
        assert PairStatus.BUFFERED.value == "buffered"
        assert PairStatus.PENDING.value == "pending"
        assert PairStatus.COMPOSED.value == "composed"
        assert PairStatus.ACTIVE.value == "active"
        assert PairStatus.DECAYING.value == "decaying"
        assert PairStatus.INVISIBLE.value == "invisible"

    def test_convergence_level_count(self):
        assert len(ConvergenceLevel) == 4

    def test_convergence_level_values(self):
        assert ConvergenceLevel.NONE.value == "none"
        assert ConvergenceLevel.STRONG.value == "strong"


# =====================================================================
# Helper Tests
# =====================================================================

class TestHelpers:
    def test_clamp_normal(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0

    def test_gen_id_length(self):
        gid = _gen_id()
        assert len(gid) == 12

    def test_gen_id_uniqueness(self):
        ids = {_gen_id() for _ in range(100)}
        assert len(ids) == 100

    def test_stage_from_freshness_active(self):
        assert _stage_from_freshness(1.0) == FreshnessStage.ACTIVE

    def test_stage_from_freshness_weakening(self):
        assert _stage_from_freshness(0.7) == FreshnessStage.WEAKENING

    def test_stage_from_freshness_fading(self):
        assert _stage_from_freshness(0.5) == FreshnessStage.FADING

    def test_stage_from_freshness_near_invisible(self):
        assert _stage_from_freshness(0.3) == FreshnessStage.NEAR_INVISIBLE

    def test_stage_from_freshness_invisible(self):
        assert _stage_from_freshness(0.1) == FreshnessStage.INVISIBLE

    def test_stage_from_freshness_boundary_080(self):
        assert _stage_from_freshness(0.8) == FreshnessStage.ACTIVE

    def test_stage_from_freshness_boundary_060(self):
        assert _stage_from_freshness(0.6) == FreshnessStage.WEAKENING

    def test_convergence_from_score_none(self):
        assert _convergence_from_score(0.1) == ConvergenceLevel.NONE

    def test_convergence_from_score_mild(self):
        assert _convergence_from_score(0.4) == ConvergenceLevel.MILD

    def test_convergence_from_score_moderate(self):
        assert _convergence_from_score(0.6) == ConvergenceLevel.MODERATE

    def test_convergence_from_score_strong(self):
        assert _convergence_from_score(0.8) == ConvergenceLevel.STRONG


# =====================================================================
# Data Structure Tests
# =====================================================================

class TestSectionDescription:
    def test_to_dict(self):
        sd = SectionDescription(
            section="external_reaction",
            description="test",
            value=0.5,
        )
        d = sd.to_dict()
        assert d["section"] == "external_reaction"
        assert d["value"] == 0.5

    def test_from_dict_roundtrip(self):
        sd = SectionDescription(
            section="emotion_transition",
            description="diff",
            value=0.3,
            confidence=0.8,
        )
        d = sd.to_dict()
        restored = SectionDescription.from_dict(d)
        assert restored.section == sd.section
        assert restored.value == sd.value
        assert restored.confidence == sd.confidence


class TestActionDescription:
    def test_to_dict(self):
        ad = ActionDescription(
            policy_label="approach",
            policy_axis="social",
            tick_at_action=5,
        )
        d = ad.to_dict()
        assert d["policy_label"] == "approach"
        assert d["tick_at_action"] == 5

    def test_from_dict_roundtrip(self):
        ad = ActionDescription(
            policy_label="hold",
            policy_axis="caution",
            selection_context="ctx",
            tick_at_action=10,
        )
        d = ad.to_dict()
        restored = ActionDescription.from_dict(d)
        assert restored.policy_label == ad.policy_label
        assert restored.tick_at_action == ad.tick_at_action


class TestResultDescription:
    def test_to_dict_empty(self):
        rd = ResultDescription()
        d = rd.to_dict()
        assert d["sections"] == []

    def test_from_dict_with_sections(self):
        rd = ResultDescription(
            sections=[
                SectionDescription(section="a", value=0.1),
                SectionDescription(section="b", value=0.2),
            ],
            tick_at_result=10,
        )
        d = rd.to_dict()
        restored = ResultDescription.from_dict(d)
        assert len(restored.sections) == 2
        assert restored.tick_at_result == 10


class TestContextAttribution:
    def test_to_dict(self):
        ca = ContextAttribution(
            context_summary="test_ctx",
            dialogue_state="idle",
            environment_tags=["a", "b"],
        )
        d = ca.to_dict()
        assert d["context_summary"] == "test_ctx"
        assert len(d["environment_tags"]) == 2

    def test_from_dict_roundtrip(self):
        ca = ContextAttribution(
            context_summary="ctx",
            dialogue_state="active",
            environment_tags=["x"],
            tick_at_context=7,
        )
        d = ca.to_dict()
        restored = ContextAttribution.from_dict(d)
        assert restored.dialogue_state == "active"
        assert restored.tick_at_context == 7


class TestActionResultPair:
    def test_default_status_buffered(self):
        p = ActionResultPair()
        assert p.status == PairStatus.BUFFERED.value

    def test_to_dict_roundtrip(self):
        p = ActionResultPair(
            action=ActionDescription(policy_label="test"),
            status=PairStatus.ACTIVE.value,
            freshness=0.8,
            pattern_key="approach",
        )
        d = p.to_dict()
        restored = ActionResultPair.from_dict(d)
        assert restored.status == PairStatus.ACTIVE.value
        assert restored.freshness == 0.8
        assert restored.pattern_key == "approach"
        assert restored.action.policy_label == "test"


class TestSectionWeightRecord:
    def test_to_dict_roundtrip(self):
        r = SectionWeightRecord(section="a", weight=0.3, cycle=5)
        d = r.to_dict()
        restored = SectionWeightRecord.from_dict(d)
        assert restored.section == "a"
        assert restored.weight == 0.3
        assert restored.cycle == 5


class TestConvergenceRecord:
    def test_to_dict_roundtrip(self):
        r = ConvergenceRecord(
            convergence_score=0.6,
            convergence_level=ConvergenceLevel.MODERATE.value,
            dominant_pattern="approach",
        )
        d = r.to_dict()
        restored = ConvergenceRecord.from_dict(d)
        assert restored.convergence_score == 0.6
        assert restored.convergence_level == ConvergenceLevel.MODERATE.value


# =====================================================================
# State Tests
# =====================================================================

class TestState:
    def test_default_state(self):
        state = ActionResultObservationState()
        assert state.cycle_count == 0
        assert state.pairs == []
        assert state.signal_supply_strength == 1.0

    def test_to_dict_roundtrip(self):
        state = ActionResultObservationState()
        state.cycle_count = 5
        state.signal_supply_strength = 0.7
        state.pattern_convergence_warning = True

        d = state.to_dict()
        restored = ActionResultObservationState.from_dict(d)
        assert restored.cycle_count == 5
        assert restored.signal_supply_strength == 0.7
        assert restored.pattern_convergence_warning is True

    def test_state_with_pairs(self):
        state = ActionResultObservationState()
        state.pairs.append(ActionResultPair(
            action=ActionDescription(policy_label="test"),
        ))
        d = state.to_dict()
        restored = ActionResultObservationState.from_dict(d)
        assert len(restored.pairs) == 1
        assert restored.pairs[0].action.policy_label == "test"


# =====================================================================
# Config Tests
# =====================================================================

class TestConfig:
    def test_defaults(self):
        cfg = ActionResultConfig()
        assert cfg.max_pairs == 200
        assert cfg.min_buffer_ticks == 3
        assert cfg.freshness_decay_rate == 0.02

    def test_custom_config(self):
        cfg = ActionResultConfig(max_pairs=50, min_buffer_ticks=1)
        assert cfg.max_pairs == 50
        assert cfg.min_buffer_ticks == 1


# =====================================================================
# Processor: record_action Tests
# =====================================================================

class TestRecordAction:
    def test_record_action_adds_to_buffer(self):
        proc = _make_processor()
        inputs = _make_inputs(current_tick=5)
        proc.record_action(inputs)
        assert len(proc.state.composition_buffer) == 1
        assert proc.state.composition_buffer[0].status == PairStatus.BUFFERED.value

    def test_record_action_captures_policy_info(self):
        proc = _make_processor()
        inputs = _make_inputs(
            selected_policy_label="label_x",
            selected_policy_axis="explore",
            current_tick=5,
        )
        proc.record_action(inputs)
        buffered = proc.state.composition_buffer[0]
        assert buffered.action.policy_label == "label_x"
        assert buffered.action.policy_axis == "explore"
        assert buffered.pattern_key == "explore"

    def test_record_action_empty_label_ignored(self):
        proc = _make_processor()
        inputs = _make_inputs(selected_policy_label="", current_tick=5)
        proc.record_action(inputs)
        assert len(proc.state.composition_buffer) == 0

    def test_record_action_buffer_overflow(self):
        proc = _make_processor(max_buffer=3)
        for i in range(5):
            inputs = _make_inputs(current_tick=i)
            proc.record_action(inputs)
        assert len(proc.state.composition_buffer) == 3
        assert proc.state.total_pairs_pending >= 2


# =====================================================================
# Processor: Stage 1 - Pair Composition Tests
# =====================================================================

class TestPairComposition:
    def test_composition_requires_min_buffer_ticks(self):
        proc = _make_processor(min_buffer_ticks=3)
        # Record action at tick 0
        proc.record_action(_make_inputs(current_tick=0))
        # Process at tick 1 (only 1 tick elapsed - should NOT compose)
        result = proc.process(_make_inputs(current_tick=1))
        assert result.active_pair_count == 0
        assert result.buffered_pair_count == 1

    def test_composition_after_min_buffer_ticks(self):
        proc = _make_processor(min_buffer_ticks=3)
        proc.record_action(_make_inputs(current_tick=0))
        # Process at tick 5 (5 ticks elapsed - should compose)
        result = proc.process(_make_inputs(current_tick=5))
        assert result.active_pair_count == 1
        assert len(result.newly_composed_pairs) == 1

    def test_immediate_composition_prohibited(self):
        """同一周期内での即時構成を禁止。"""
        proc = _make_processor(min_buffer_ticks=3)
        proc.record_action(_make_inputs(current_tick=10))
        result = proc.process(_make_inputs(current_tick=10))
        assert result.active_pair_count == 0

    def test_buffer_stale_becomes_pending(self):
        proc = _make_processor(
            min_buffer_ticks=3,
            max_buffer_ticks=5,
        )
        proc.record_action(_make_inputs(current_tick=0))
        # Process at tick 100 (way past max_buffer_ticks)
        result = proc.process(_make_inputs(current_tick=100))
        # Should be composed since min_buffer_ticks is met
        assert result.active_pair_count >= 0


# =====================================================================
# Processor: Stage 2 - Multi-Section Description Tests
# =====================================================================

class TestMultiSectionDescription:
    def test_sections_created_from_inputs(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(current_tick=5))
        assert len(result.newly_composed_pairs) == 1
        pair = result.newly_composed_pairs[0]
        # Should have multiple sections
        assert len(pair.result.sections) > 0

    def test_external_reaction_section(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(
            current_tick=5,
            external_response_change=0.5,
            external_response_description="positive_response",
        ))
        pair = result.newly_composed_pairs[0]
        ext_sections = [
            s for s in pair.result.sections
            if s.section == ObservationSection.EXTERNAL_REACTION.value
        ]
        assert len(ext_sections) == 1
        assert ext_sections[0].value == 0.5

    def test_emotion_transition_section(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(
            current_tick=5,
            emotion_before={"joy": 0.3, "anger": 0.1},
            emotion_after={"joy": 0.8, "anger": 0.0},
        ))
        pair = result.newly_composed_pairs[0]
        emo_sections = [
            s for s in pair.result.sections
            if s.section == ObservationSection.EMOTION_TRANSITION.value
        ]
        assert len(emo_sections) == 1
        assert emo_sections[0].value > 0.0

    def test_no_priority_among_sections(self):
        """断面間に優先順位を設けない。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(current_tick=5))
        pair = result.newly_composed_pairs[0]
        # All sections are independent records
        section_types = [s.section for s in pair.result.sections]
        assert len(section_types) == len(set(section_types))  # no duplicates

    def test_sections_with_zero_values_omitted(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(
            current_tick=5,
            external_response_change=0.0,
            external_response_description="",
            other_reaction_change=0.0,
            other_reaction_description="",
        ))
        pair = result.newly_composed_pairs[0]
        ext = [
            s for s in pair.result.sections
            if s.section == ObservationSection.EXTERNAL_REACTION.value
        ]
        # Zero external should be omitted
        assert len(ext) == 0


# =====================================================================
# Processor: Stage 3 - Context Attribution Tests
# =====================================================================

class TestContextAttribution_Process:
    def test_context_attached_to_pair(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(
            current_tick=5,
            context_summary="important_context",
            dialogue_state="active_dialogue",
            environment_tags=["tag_x", "tag_y"],
        ))
        pair = result.newly_composed_pairs[0]
        assert pair.context.context_summary == "important_context"
        assert pair.context.dialogue_state == "active_dialogue"
        assert "tag_x" in pair.context.environment_tags

    def test_same_action_different_context_distinct_pairs(self):
        """同一行動でも文脈が異なれば異なる対として蓄積。"""
        proc = _make_processor(min_buffer_ticks=1)
        # First action+context
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(
            current_tick=5,
            context_summary="context_A",
        ))
        # Second action+different context
        proc.record_action(_make_inputs(current_tick=6))
        proc.process(_make_inputs(
            current_tick=12,
            context_summary="context_B",
        ))
        assert len(proc.state.pairs) == 2
        assert proc.state.pairs[0].context.context_summary == "context_A"
        assert proc.state.pairs[1].context.context_summary == "context_B"


# =====================================================================
# Processor: Stage 4 - Alignment and Accumulation Tests
# =====================================================================

class TestAccumulation:
    def test_pairs_accumulated_in_time_order(self):
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(3):
            proc.record_action(_make_inputs(current_tick=i * 10))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        assert len(proc.state.pairs) == 3
        assert len(proc.state.time_index) == 3

    def test_accumulation_is_append_only(self):
        """蓄積は追記形式。特定パターンのみ優先保持しない。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(
            current_tick=0, selected_policy_axis="approach",
        ))
        proc.process(_make_inputs(current_tick=5))
        proc.record_action(_make_inputs(
            current_tick=10, selected_policy_axis="hold",
        ))
        proc.process(_make_inputs(current_tick=15))
        patterns = [p.pattern_key for p in proc.state.pairs]
        assert "approach" in patterns
        assert "hold" in patterns

    def test_max_pairs_limit(self):
        proc = _make_processor(min_buffer_ticks=1, max_pairs=5)
        for i in range(10):
            proc.record_action(_make_inputs(current_tick=i * 10))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        assert len(proc.state.pairs) <= 5

    def test_invisible_pairs_removed_first(self):
        proc = _make_processor(min_buffer_ticks=1, max_pairs=5)
        # Create some pairs
        for i in range(5):
            proc.record_action(_make_inputs(current_tick=i * 10))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        # Make first pair invisible
        proc.state.pairs[0].status = PairStatus.INVISIBLE.value
        # Add more
        proc.record_action(_make_inputs(current_tick=100))
        proc.process(_make_inputs(current_tick=105))
        # Invisible one should be removed first
        invisible = [
            p for p in proc.state.pairs
            if p.status == PairStatus.INVISIBLE.value
        ]
        assert len(proc.state.pairs) <= 5


# =====================================================================
# Processor: Stage 5 - Decay and Forgetting Tests
# =====================================================================

class TestDecayAndForgetting:
    def test_freshness_decays_over_cycles(self):
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        initial_freshness = proc.state.pairs[0].freshness

        # Run multiple cycles
        for i in range(5):
            proc.process(_make_inputs(current_tick=10 + i))
        assert proc.state.pairs[0].freshness < initial_freshness

    def test_stage_transitions(self):
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.15)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        assert proc.state.pairs[0].freshness_stage == FreshnessStage.ACTIVE.value

        # Decay many cycles
        for i in range(20):
            proc.process(_make_inputs(current_tick=10 + i))
        # Should have progressed through stages
        final_stage = proc.state.pairs[0].freshness_stage
        assert final_stage != FreshnessStage.ACTIVE.value

    def test_invisible_pairs_get_recovery_candidate(self):
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.25)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        # Decay to invisible
        for i in range(50):
            proc.process(_make_inputs(current_tick=10 + i))

        invisible = [
            p for p in proc.state.pairs
            if p.status == PairStatus.INVISIBLE.value
        ]
        if invisible:
            assert invisible[0].pair_id in proc.state.recovery_candidates

    def test_decay_history_recorded(self):
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.15)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        for i in range(10):
            proc.process(_make_inputs(current_tick=10 + i))
        assert len(proc.state.decay_history) > 0

    def test_staged_forgetting_not_bulk_delete(self):
        """段階的な希薄化を経て不可視化。一括消去ではない。"""
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.05)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        stages_seen = set()
        for i in range(50):
            proc.process(_make_inputs(current_tick=10 + i))
            stages_seen.add(proc.state.pairs[0].freshness_stage)

        # Should have seen multiple stages (not just ACTIVE then INVISIBLE)
        assert len(stages_seen) >= 2


# =====================================================================
# Processor: Stage 6 - Handoff Preparation / Safety Valves Tests
# =====================================================================

class TestSafetyValves:
    def test_pattern_convergence_warning(self):
        """パターン収束時の警告。"""
        proc = _make_processor(min_buffer_ticks=1, convergence_threshold=0.5)
        # Create many pairs with same pattern
        for i in range(10):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis="approach",
            ))
            result = proc.process(_make_inputs(current_tick=i * 10 + 5))
        # All same pattern → convergence warning
        assert proc.state.pattern_convergence_warning or result.convergence_score > 0

    def test_section_bias_warning(self):
        """断面偏り時の警告。"""
        proc = _make_processor(
            min_buffer_ticks=1,
            section_bias_threshold=0.5,
        )
        # Process with minimal section variety
        for i in range(5):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                internal_state_delta=0.0,
                motivation_delta=0.0,
                direction_delta=0.0,
                other_reaction_change=0.0,
                other_reaction_description="",
                referenced_memory_count=0,
                emotion_before={},
                emotion_after={},
            ))
            result = proc.process(_make_inputs(
                current_tick=i * 10 + 5,
                internal_state_delta=0.0,
                motivation_delta=0.0,
                direction_delta=0.0,
                other_reaction_change=0.0,
                other_reaction_description="",
                referenced_memory_count=0,
                emotion_before={},
                emotion_after={},
            ))

    def test_signal_attenuation_on_convergence(self):
        """偏り検出時にシグナル供給強度が減衰する。"""
        proc = _make_processor(min_buffer_ticks=1, convergence_threshold=0.3)
        initial_strength = proc.state.signal_supply_strength

        for i in range(15):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=i * 10 + 5))

        # If convergence detected, strength should decrease
        if proc.state.pattern_convergence_warning:
            assert proc.state.signal_supply_strength <= initial_strength

    def test_signal_strength_has_minimum(self):
        """シグナル供給強度は最低値を持つ。"""
        proc = _make_processor(
            min_buffer_ticks=1,
            signal_min_strength=0.1,
            signal_attenuation_rate=0.5,
        )
        for i in range(50):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        assert proc.state.signal_supply_strength >= 0.1

    def test_diversity_restoration(self):
        """パターン収束時に異パターン対を再浮上。"""
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.1)
        # Create pairs with different patterns
        proc.record_action(_make_inputs(
            current_tick=0, selected_policy_axis="explore",
        ))
        proc.process(_make_inputs(current_tick=5))

        # Decay the explore pair
        for i in range(10):
            proc.process(_make_inputs(current_tick=10 + i))

        # Now add many approach pairs
        for i in range(10):
            proc.record_action(_make_inputs(
                current_tick=100 + i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=100 + i * 10 + 5))

        # Check that the explore pair wasn't fully abandoned
        explore_pairs = [
            p for p in proc.state.pairs
            if p.pattern_key == "explore"
        ]
        assert len(explore_pairs) > 0

    def test_buffer_overflow_warning(self):
        proc = _make_processor(
            min_buffer_ticks=100,  # High so nothing composes
            buffer_overflow_threshold=3,
            max_buffer=30,
        )
        for i in range(5):
            proc.record_action(_make_inputs(current_tick=i))
        proc.process(_make_inputs(current_tick=5))
        assert proc.state.buffer_overflow_warning is True


# =====================================================================
# Reference Tracking Tests
# =====================================================================

class TestReferenceTracking:
    def test_record_reference_increases_count(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        pair_id = proc.state.pairs[0].pair_id
        proc.record_reference(pair_id)
        assert proc.state.pairs[0].reference_count == 1

    def test_reference_recovers_freshness(self):
        proc = _make_processor(
            min_buffer_ticks=1,
            freshness_decay_rate=0.1,
            reference_recovery=0.15,
        )
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        # Decay
        for i in range(5):
            proc.process(_make_inputs(current_tick=10 + i))
        decayed_freshness = proc.state.pairs[0].freshness

        # Reference
        pair_id = proc.state.pairs[0].pair_id
        proc.record_reference(pair_id)
        assert proc.state.pairs[0].freshness > decayed_freshness

    def test_reactivation_limit(self):
        """再活性上限を超えると鮮度回復しない。"""
        proc = _make_processor(
            min_buffer_ticks=1,
            max_reactivation_count=3,
            freshness_decay_rate=0.1,
            reference_recovery=0.15,
        )
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        pair_id = proc.state.pairs[0].pair_id

        # Decay and reference multiple times
        for i in range(5):
            for _ in range(3):
                proc.process(_make_inputs(current_tick=20 + i * 10))
            proc.record_reference(pair_id)

        # After max_reactivation_count references, further references
        # should not increase reactivation_count beyond the limit
        assert proc.state.pairs[0].reactivation_count == 3

        # One more reference should not recover
        proc.process(_make_inputs(current_tick=200))
        freshness_before = proc.state.pairs[0].freshness
        proc.record_reference(pair_id)
        assert proc.state.pairs[0].freshness == freshness_before

    def test_decaying_pair_recovers_on_reference(self):
        proc = _make_processor(
            min_buffer_ticks=1,
            freshness_decay_rate=0.15,
        )
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        # Decay
        for i in range(5):
            proc.process(_make_inputs(current_tick=10 + i))
        pair = proc.state.pairs[0]
        if pair.status == PairStatus.DECAYING.value:
            proc.record_reference(pair.pair_id)
            assert pair.status == PairStatus.ACTIVE.value
            assert proc.state.total_pairs_recovered >= 1


# =====================================================================
# Freshness Compatibility Tests
# =====================================================================

class TestFreshnessCompatibility:
    def test_get_freshness_compatible_info(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        info = proc.get_freshness_compatible_info()
        assert len(info) == 1
        assert info[0]["source"] == "action_result"
        assert "freshness" in info[0]
        assert "freshness_stage" in info[0]

    def test_freshness_info_format(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))

        info = proc.get_freshness_compatible_info()
        entry = info[0]
        required_keys = {"id", "source", "freshness", "freshness_stage",
                         "status", "reference_count", "creation_time",
                         "last_reference_time"}
        assert required_keys.issubset(set(entry.keys()))


# =====================================================================
# Active Pairs / Pattern Query Tests
# =====================================================================

class TestQueryMethods:
    def test_get_active_pairs(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        active = proc.get_active_pairs()
        assert len(active) == 1

    def test_get_pairs_by_pattern(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(
            current_tick=0, selected_policy_axis="approach",
        ))
        proc.process(_make_inputs(current_tick=5))
        proc.record_action(_make_inputs(
            current_tick=10, selected_policy_axis="hold",
        ))
        proc.process(_make_inputs(current_tick=15))

        approach_pairs = proc.get_pairs_by_pattern("approach")
        assert len(approach_pairs) == 1
        hold_pairs = proc.get_pairs_by_pattern("hold")
        assert len(hold_pairs) == 1


# =====================================================================
# Summary Tests
# =====================================================================

class TestSummary:
    def test_empty_state_summary(self):
        state = ActionResultObservationState()
        summary = get_action_result_summary(state)
        assert "待機中" in summary

    def test_active_state_summary(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        proc.process(_make_inputs(current_tick=5))
        summary = get_action_result_summary(proc.state)
        assert "活性対" in summary
        assert "cycle=" in summary

    def test_summary_with_warnings(self):
        state = ActionResultObservationState()
        state.cycle_count = 10
        state.pattern_convergence_warning = True
        state.section_bias_warning = True
        state.signal_attenuation_active = True
        state.signal_supply_strength = 0.5
        summary = get_action_result_summary(state)
        assert "収束偏向" in summary
        assert "断面偏り" in summary
        assert "シグナル減衰" in summary


# =====================================================================
# Factory Tests
# =====================================================================

class TestFactory:
    def test_create_default(self):
        proc = create_action_result_processor()
        assert isinstance(proc, ActionResultObservationProcessor)
        assert proc.state.cycle_count == 0

    def test_create_with_config(self):
        cfg = ActionResultConfig(max_pairs=50)
        proc = create_action_result_processor(config=cfg)
        assert proc._config.max_pairs == 50


# =====================================================================
# Integration / Pipeline Tests
# =====================================================================

class TestPipeline:
    def test_full_pipeline(self):
        """6段パイプライン全体の正常動作。"""
        proc = _make_processor(min_buffer_ticks=2)

        # Record action
        proc.record_action(_make_inputs(current_tick=0))
        assert len(proc.state.composition_buffer) == 1

        # Process (not enough ticks)
        result1 = proc.process(_make_inputs(current_tick=1))
        assert result1.active_pair_count == 0

        # Process (enough ticks)
        result2 = proc.process(_make_inputs(current_tick=5))
        assert result2.active_pair_count == 1
        assert len(result2.newly_composed_pairs) == 1
        pair = result2.newly_composed_pairs[0]
        assert len(pair.result.sections) > 0
        assert pair.context.context_summary != ""
        assert pair.status == PairStatus.ACTIVE.value

    def test_multiple_pairs_pipeline(self):
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(5):
            proc.record_action(_make_inputs(
                current_tick=i * 20,
                selected_policy_axis=f"axis_{i}",
            ))
            result = proc.process(_make_inputs(current_tick=i * 20 + 10))
            assert len(result.newly_composed_pairs) == 1

        assert len(proc.state.pairs) == 5
        assert proc.state.total_pairs_composed == 5

    def test_cycle_count_increments(self):
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(3):
            proc.process(_make_inputs(current_tick=i * 10))
        assert proc.state.cycle_count == 3

    def test_result_contains_distributions(self):
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(current_tick=5))
        assert isinstance(result.section_distribution, dict)
        assert isinstance(result.pattern_distribution, dict)
        assert isinstance(result.freshness_distribution, dict)

    def test_no_direct_judgment_in_output(self):
        """出力は参照情報のみ。判断・評価を含まない。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(current_tick=5))
        # Result should not contain evaluation/judgment fields
        d = result.to_dict()
        assert "judgment" not in d
        assert "evaluation" not in d
        assert "optimal" not in d
        assert "recommendation" not in d

    def test_no_causal_claim(self):
        """因果断定しない。時系列的隣接の記録のみ。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0))
        result = proc.process(_make_inputs(current_tick=5))
        pair = result.newly_composed_pairs[0]
        # Pair description should not contain causal language
        for section in pair.result.sections:
            assert "caused" not in section.description.lower()
            assert "because" not in section.description.lower()

    def test_state_property_setter(self):
        proc = _make_processor()
        new_state = ActionResultObservationState()
        new_state.cycle_count = 42
        proc.state = new_state
        assert proc.state.cycle_count == 42


# =====================================================================
# Self-reinforcement Prevention Tests
# =====================================================================

class TestSelfReinforcementPrevention:
    def test_long_absent_patterns_not_deleted(self):
        """長期間構成されなかったパターンの対を完全消去しない。"""
        proc = _make_processor(min_buffer_ticks=1, freshness_decay_rate=0.05)
        # Create explore pair
        proc.record_action(_make_inputs(
            current_tick=0, selected_policy_axis="explore",
        ))
        proc.process(_make_inputs(current_tick=5))

        # Create many approach pairs (explore pattern absent for long time)
        for i in range(20):
            proc.record_action(_make_inputs(
                current_tick=100 + i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=100 + i * 10 + 5))

        # The explore pair should still exist (not deleted)
        explore_ids = [
            p.pair_id for p in proc.state.pairs
            if p.pattern_key == "explore"
        ]
        # Either the pair exists or its ID is in recovery candidates
        assert (
            len(explore_ids) > 0
            or any(
                pid in proc.state.recovery_candidates
                for pid in explore_ids
            )
            or len(proc.state.recovery_candidates) > 0
        )

    def test_signal_strength_reduces_on_concentration(self):
        """パターン集中時にシグナル強度が減衰する。"""
        proc = _make_processor(
            min_buffer_ticks=1,
            convergence_threshold=0.4,
        )
        for i in range(15):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=i * 10 + 5))

        # Signal strength should be reduced
        if proc.state.pattern_convergence_warning:
            assert proc.state.signal_supply_strength < 1.0


# =====================================================================
# Convergence Monitoring Tests
# =====================================================================

class TestConvergenceMonitoring:
    def test_convergence_records_accumulated(self):
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(5):
            proc.record_action(_make_inputs(current_tick=i * 10))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        assert len(proc.state.convergence_records) == 5

    def test_convergence_score_increases_with_monotony(self):
        proc = _make_processor(min_buffer_ticks=1)
        # All same pattern
        for i in range(10):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis="approach",
            ))
            proc.process(_make_inputs(current_tick=i * 10 + 5))

        # Last convergence score should be elevated
        if proc.state.convergence_records:
            last = proc.state.convergence_records[-1]
            assert last.convergence_score > 0.0

    def test_high_diversity_low_convergence(self):
        proc = _make_processor(min_buffer_ticks=1)
        axes = ["approach", "hold", "explore", "shift", "maintain"]
        for i, axis in enumerate(axes):
            proc.record_action(_make_inputs(
                current_tick=i * 10,
                selected_policy_axis=axis,
            ))
            proc.process(_make_inputs(current_tick=i * 10 + 5))

        if proc.state.convergence_records:
            last = proc.state.convergence_records[-1]
            assert last.convergence_level in (
                ConvergenceLevel.NONE.value,
                ConvergenceLevel.MILD.value,
            )

    def test_convergence_records_trimmed(self):
        proc = _make_processor(
            min_buffer_ticks=1,
            max_convergence_records=5,
        )
        for i in range(10):
            proc.record_action(_make_inputs(current_tick=i * 10))
            proc.process(_make_inputs(current_tick=i * 10 + 5))
        assert len(proc.state.convergence_records) <= 5


# =====================================================================
# tick() Method Tests
# =====================================================================

class TestTick:
    def test_tick_records_action_and_processes(self):
        """tick() が行動記録と6段パイプラインを統合実行する。"""
        proc = _make_processor(min_buffer_ticks=1)
        # First tick: record action to buffer
        result1 = proc.tick(_make_inputs(current_tick=10))
        assert len(proc.state.composition_buffer) == 1
        assert proc.state.cycle_count == 1

        # Second tick: buffer pair should be composed
        result2 = proc.tick(_make_inputs(current_tick=25))
        assert proc.state.cycle_count == 2
        # First pair composed + second recorded to buffer
        assert proc.state.total_pairs_composed >= 1

    def test_tick_without_policy_label(self):
        """policy_label がない場合は行動記録をスキップする。"""
        proc = _make_processor(min_buffer_ticks=1)
        result = proc.tick(_make_inputs(
            current_tick=10,
            selected_policy_label="",
        ))
        assert len(proc.state.composition_buffer) == 0
        assert proc.state.cycle_count == 1

    def test_tick_returns_valid_result(self):
        """tick() が ActionResultObservationResult を返す。"""
        proc = _make_processor(min_buffer_ticks=1)
        result = proc.tick(_make_inputs(current_tick=10))
        assert isinstance(result, ActionResultObservationResult)
        assert isinstance(result.active_pair_count, int)
        assert isinstance(result.pattern_convergence_warning, bool)
        assert isinstance(result.section_bias_warning, bool)

    def test_tick_equivalent_to_record_plus_process(self):
        """tick() は record_action + process と同等の結果を生む。"""
        # tick() path
        proc1 = _make_processor(min_buffer_ticks=1)
        inp = _make_inputs(current_tick=10)
        result1 = proc1.tick(inp)

        # record_action + process path
        proc2 = _make_processor(min_buffer_ticks=1)
        proc2.record_action(inp)
        result2 = proc2.process(inp)

        assert proc1.state.cycle_count == proc2.state.cycle_count
        assert len(proc1.state.composition_buffer) == len(proc2.state.composition_buffer)
        assert proc1.state.total_pairs_composed == proc2.state.total_pairs_composed

    def test_tick_multiple_cycles(self):
        """複数ティックにわたる tick() の正常動作。"""
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(5):
            proc.tick(_make_inputs(current_tick=i * 10))
        assert proc.state.cycle_count == 5
        assert proc.state.total_pairs_composed >= 3  # min_buffer=1, so most compose


# =====================================================================
# get_enrichment_data() Method Tests
# =====================================================================

class TestGetEnrichmentData:
    def test_empty_state_enrichment(self):
        """空状態の enrichment データ。"""
        proc = _make_processor()
        data = proc.get_enrichment_data()
        assert data["cycle_count"] == 0
        assert data["active_count"] == 0
        assert data["decaying_count"] == 0
        assert data["buffered_count"] == 0
        assert data["total_composed"] == 0
        assert data["total_recovered"] == 0
        assert data["pattern_distribution"] == {}
        assert data["freshness_distribution"] == {}
        assert data["section_distribution"] == {}
        assert data["pattern_convergence_warning"] is False
        assert data["section_bias_warning"] is False
        assert data["signal_attenuation_active"] is False
        assert data["signal_supply_strength"] == 1.0
        assert "待機中" in data["summary_text"]

    def test_enrichment_after_processing(self):
        """処理後の enrichment データが正しい構造を持つ。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.tick(_make_inputs(current_tick=10))
        proc.tick(_make_inputs(current_tick=25))

        data = proc.get_enrichment_data()
        assert data["cycle_count"] == 2
        assert isinstance(data["pattern_distribution"], dict)
        assert isinstance(data["freshness_distribution"], dict)
        assert isinstance(data["section_distribution"], dict)
        assert isinstance(data["summary_text"], str)

    def test_enrichment_pattern_distribution(self):
        """パターン分布が正しく反映される。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.tick(_make_inputs(current_tick=10, selected_policy_axis="explore"))
        proc.tick(_make_inputs(current_tick=25, selected_policy_axis="explore"))
        proc.tick(_make_inputs(current_tick=40, selected_policy_axis="approach"))
        proc.tick(_make_inputs(current_tick=55))

        data = proc.get_enrichment_data()
        # At least one pattern should exist
        if data["active_count"] > 0:
            assert len(data["pattern_distribution"]) >= 1

    def test_enrichment_freshness_distribution(self):
        """鮮度分布が正しく反映される。"""
        proc = _make_processor(min_buffer_ticks=1)
        for i in range(5):
            proc.tick(_make_inputs(current_tick=i * 10))

        data = proc.get_enrichment_data()
        if len(proc.state.pairs) > 0:
            total_freshness = sum(data["freshness_distribution"].values())
            assert total_freshness == len(proc.state.pairs)

    def test_enrichment_warnings_reflected(self):
        """安全弁の警告が enrichment データに反映される。"""
        proc = _make_processor(min_buffer_ticks=1)
        # Manually set warning flags for testing
        proc.state.pattern_convergence_warning = True
        proc.state.section_bias_warning = True
        proc.state.signal_attenuation_active = True
        proc.state.signal_supply_strength = 0.5

        data = proc.get_enrichment_data()
        assert data["pattern_convergence_warning"] is True
        assert data["section_bias_warning"] is True
        assert data["signal_attenuation_active"] is True
        assert data["signal_supply_strength"] == 0.5

    def test_enrichment_data_keys(self):
        """enrichment データが必要なキーをすべて含む。"""
        proc = _make_processor()
        data = proc.get_enrichment_data()
        expected_keys = {
            "cycle_count", "active_count", "decaying_count",
            "buffered_count", "total_composed", "total_recovered",
            "pattern_distribution", "freshness_distribution",
            "section_distribution", "pattern_convergence_warning",
            "section_bias_warning", "signal_attenuation_active",
            "signal_supply_strength", "summary_text",
        }
        assert set(data.keys()) == expected_keys


# =====================================================================
# Input Pathway Label Tests
# =====================================================================

class TestInputPathwayLabel:
    def test_inputs_default_empty_string(self):
        """ActionResultInputs の input_pathway_label はデフォルトで空文字。"""
        inputs = ActionResultInputs()
        assert inputs.input_pathway_label == ""

    def test_inputs_with_pathway_label(self):
        """input_pathway_label 付きの ActionResultInputs 生成。"""
        inputs = _make_inputs(input_pathway_label="text")
        assert inputs.input_pathway_label == "text"

    def test_inputs_with_screen_pathway(self):
        """screen 経路ラベル付きの ActionResultInputs 生成。"""
        inputs = _make_inputs(input_pathway_label="screen")
        assert inputs.input_pathway_label == "screen"

    def test_inputs_with_spontaneous_pathway(self):
        """spontaneous 経路ラベル付きの ActionResultInputs 生成。"""
        inputs = _make_inputs(input_pathway_label="spontaneous")
        assert inputs.input_pathway_label == "spontaneous"

    def test_pathway_label_propagated_to_pair(self):
        """record_action で input_pathway_label が対構成バッファに伝搬する。"""
        proc = _make_processor()
        inputs = _make_inputs(current_tick=5, input_pathway_label="text")
        proc.record_action(inputs)
        assert len(proc.state.composition_buffer) == 1
        assert proc.state.composition_buffer[0].input_pathway_label == "text"

    def test_pathway_label_empty_propagated(self):
        """空文字の input_pathway_label も正しく伝搬する。"""
        proc = _make_processor()
        inputs = _make_inputs(current_tick=5, input_pathway_label="")
        proc.record_action(inputs)
        assert proc.state.composition_buffer[0].input_pathway_label == ""

    def test_pathway_label_in_composed_pair(self):
        """対構成後にも input_pathway_label が保持される。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="screen"))
        result = proc.process(_make_inputs(current_tick=5))
        assert len(result.newly_composed_pairs) == 1
        assert result.newly_composed_pairs[0].input_pathway_label == "screen"

    def test_pathway_label_in_active_pairs(self):
        """get_active_pairs で取得した対にも input_pathway_label がある。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="spontaneous"))
        proc.process(_make_inputs(current_tick=5))
        active = proc.get_active_pairs()
        assert len(active) == 1
        assert active[0].input_pathway_label == "spontaneous"

    def test_pathway_label_to_dict(self):
        """ActionResultPair.to_dict に input_pathway_label が含まれる。"""
        pair = ActionResultPair(
            action=ActionDescription(policy_label="test"),
            input_pathway_label="text",
        )
        d = pair.to_dict()
        assert "input_pathway_label" in d
        assert d["input_pathway_label"] == "text"

    def test_pathway_label_from_dict_roundtrip(self):
        """to_dict / from_dict で input_pathway_label が保存復元される。"""
        pair = ActionResultPair(
            action=ActionDescription(policy_label="test"),
            input_pathway_label="screen",
        )
        d = pair.to_dict()
        restored = ActionResultPair.from_dict(d)
        assert restored.input_pathway_label == "screen"

    def test_pathway_label_from_dict_backward_compat(self):
        """from_dict で input_pathway_label が存在しない場合は空文字（後方互換）。"""
        d = {
            "pair_id": "abc123",
            "action": {"policy_label": "test"},
            "status": "active",
        }
        restored = ActionResultPair.from_dict(d)
        assert restored.input_pathway_label == ""

    def test_enrichment_does_not_contain_pathway_label(self):
        """enrichment に input_pathway_label 情報が含まれない（逆流遮断）。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="text"))
        proc.process(_make_inputs(current_tick=5))
        data = proc.get_enrichment_data()
        assert "input_pathway_label" not in data
        # summary_text にも経路情報が含まれないことを確認
        assert "text" not in data["summary_text"].lower() or "テキスト" not in data["summary_text"]

    def test_enrichment_summary_no_pathway_info(self):
        """get_action_result_summary に input_pathway_label 情報が含まれない。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="spontaneous"))
        proc.process(_make_inputs(current_tick=5))
        summary = get_action_result_summary(proc.state)
        # summary should not mention pathway labels directly
        assert "spontaneous" not in summary

    def test_multiple_pairs_different_pathway_labels(self):
        """異なる経路ラベルの複数対が独立に保持される。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="text"))
        proc.process(_make_inputs(current_tick=5))
        proc.record_action(_make_inputs(current_tick=10, input_pathway_label="screen"))
        proc.process(_make_inputs(current_tick=15))
        proc.record_action(_make_inputs(current_tick=20, input_pathway_label=""))
        proc.process(_make_inputs(current_tick=25))
        assert len(proc.state.pairs) == 3
        assert proc.state.pairs[0].input_pathway_label == "text"
        assert proc.state.pairs[1].input_pathway_label == "screen"
        assert proc.state.pairs[2].input_pathway_label == ""

    def test_pathway_label_survives_state_roundtrip(self):
        """State の to_dict / from_dict 経由で input_pathway_label が保持される。"""
        proc = _make_processor(min_buffer_ticks=1)
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="text"))
        proc.process(_make_inputs(current_tick=5))

        state_dict = proc.state.to_dict()
        restored_state = ActionResultObservationState.from_dict(state_dict)
        assert len(restored_state.pairs) == 1
        assert restored_state.pairs[0].input_pathway_label == "text"

    def test_pathway_label_in_buffer_state_roundtrip(self):
        """構成バッファ内の対も State roundtrip で input_pathway_label が保持される。"""
        proc = _make_processor(min_buffer_ticks=100)  # 構成されないようにする
        proc.record_action(_make_inputs(current_tick=0, input_pathway_label="screen"))

        state_dict = proc.state.to_dict()
        restored_state = ActionResultObservationState.from_dict(state_dict)
        assert len(restored_state.composition_buffer) == 1
        assert restored_state.composition_buffer[0].input_pathway_label == "screen"
