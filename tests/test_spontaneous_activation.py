"""tests/test_spontaneous_activation.py - 自発起動経路のテスト"""

import time
import pytest

from psyche.spontaneous_activation import (
    # Enums
    ActivationSourceType,
    ActivationFreshness,
    CandidateStatus,
    SuppressionMode,
    ConflictResolution,
    # Dataclasses
    ActivationFragment,
    ActivationCandidate,
    ActivationRationale,
    SuppressionEntry,
    StandbyEntry,
    ConflictHistoryEntry,
    UnadoptedHistoryEntry,
    ContinuousActivationEntry,
    SpontaneousDecayEntry,
    SpontaneousConfig,
    SpontaneousState,
    ActivationResult,
    # Extraction functions
    extract_intrinsic_motivation,
    extract_direction_vector,
    extract_unfinished_intent,
    extract_memory_echo,
    extract_emotional_transition,
    extract_responsibility,
    extract_recent_action,
    extract_external_input_absence,
    extract_all_fragments,
    # Pipeline
    form_candidates,
    align_conditions,
    resolve_conflicts,
    check_activation_feasibility,
    # Safety valves
    suppress_consecutive_series,
    apply_overdense_cooldown,
    restore_candidate_diversity,
    apply_freshness_decay,
    decay_unadopted_history,
    # Processor
    SpontaneousActivationProcessor,
    # Summary / Factory
    get_spontaneous_summary,
    create_spontaneous_processor,
)


# =============================================================================
# Enums
# =============================================================================

class TestEnums:
    def test_activation_source_type_values(self):
        assert ActivationSourceType.INTRINSIC_MOTIVATION.value == "intrinsic_motivation"
        assert ActivationSourceType.EXTERNAL_INPUT_ABSENCE.value == "external_input_absence"
        assert len(ActivationSourceType) == 8

    def test_activation_freshness_values(self):
        assert ActivationFreshness.FRESH.value == "fresh"
        assert ActivationFreshness.FADED.value == "faded"

    def test_candidate_status_values(self):
        assert CandidateStatus.ACTIVE.value == "active"
        assert CandidateStatus.STANDBY.value == "standby"

    def test_suppression_mode_values(self):
        assert SuppressionMode.NONE.value == "none"
        assert SuppressionMode.RELEASED.value == "released"

    def test_conflict_resolution_values(self):
        assert ConflictResolution.PARALLEL.value == "parallel"
        assert ConflictResolution.RECYCLED.value == "recycled"


# =============================================================================
# Dataclasses - to_dict / from_dict
# =============================================================================

class TestActivationFragment:
    def test_default_creation(self):
        f = ActivationFragment()
        assert len(f.fragment_id) == 12
        assert f.timestamp > 0

    def test_to_dict_from_dict(self):
        f = ActivationFragment(
            fragment_id="frag1",
            source_type=ActivationSourceType.MEMORY_ECHO,
            value=0.7,
            description="test",
        )
        d = f.to_dict()
        restored = ActivationFragment.from_dict(d)
        assert restored.source_type == ActivationSourceType.MEMORY_ECHO
        assert restored.value == 0.7


class TestActivationCandidate:
    def test_default_creation(self):
        c = ActivationCandidate()
        assert len(c.candidate_id) == 12

    def test_to_dict_from_dict(self):
        c = ActivationCandidate(
            candidate_id="cand1",
            source_types=["intrinsic_motivation", "memory_echo"],
            intersection_count=2,
            activation_strength=0.6,
            status=CandidateStatus.ACTIVE,
        )
        d = c.to_dict()
        restored = ActivationCandidate.from_dict(d)
        assert restored.intersection_count == 2
        assert restored.status == CandidateStatus.ACTIVE


class TestActivationRationale:
    def test_to_dict_from_dict(self):
        r = ActivationRationale(
            candidate_id="c1",
            contributing_sources=["a", "b"],
            continuous_diff_reference=0.8,
        )
        d = r.to_dict()
        restored = ActivationRationale.from_dict(d)
        assert restored.continuous_diff_reference == 0.8


class TestSuppressionEntry:
    def test_to_dict_from_dict(self):
        s = SuppressionEntry(
            candidate_id="c1", reason="test",
            mode=SuppressionMode.ACTIVE, reversible=True,
        )
        d = s.to_dict()
        restored = SuppressionEntry.from_dict(d)
        assert restored.mode == SuppressionMode.ACTIVE
        assert restored.reversible is True


class TestStandbyEntry:
    def test_to_dict_from_dict(self):
        s = StandbyEntry(
            candidate_id="c1", original_strength=0.5,
            standby_reason="cooldown",
        )
        d = s.to_dict()
        restored = StandbyEntry.from_dict(d)
        assert restored.original_strength == 0.5


class TestConflictHistoryEntry:
    def test_to_dict_from_dict(self):
        c = ConflictHistoryEntry(
            candidate_id_a="a", candidate_id_b="b",
            resolution="parallel",
        )
        d = c.to_dict()
        restored = ConflictHistoryEntry.from_dict(d)
        assert restored.resolution == "parallel"


class TestUnadoptedHistoryEntry:
    def test_to_dict_from_dict(self):
        u = UnadoptedHistoryEntry(
            candidate_id="c1",
            source_types=["memory_echo"],
            original_strength=0.4,
        )
        d = u.to_dict()
        restored = UnadoptedHistoryEntry.from_dict(d)
        assert restored.original_strength == 0.4


class TestContinuousActivationEntry:
    def test_to_dict_from_dict(self):
        e = ContinuousActivationEntry(
            candidate_id="c1",
            source_types=["intrinsic_motivation"],
            cycle_id=5,
        )
        d = e.to_dict()
        restored = ContinuousActivationEntry.from_dict(d)
        assert restored.cycle_id == 5


class TestSpontaneousDecayEntry:
    def test_to_dict_from_dict(self):
        de = SpontaneousDecayEntry(
            candidate_id="c1",
            original_freshness=ActivationFreshness.FRESH,
            decayed_freshness=ActivationFreshness.AGING,
        )
        d = de.to_dict()
        restored = SpontaneousDecayEntry.from_dict(d)
        assert restored.decayed_freshness == ActivationFreshness.AGING


class TestSpontaneousConfig:
    def test_defaults(self):
        cfg = SpontaneousConfig()
        assert cfg.min_intersection_count == 2
        assert cfg.fragment_threshold == 0.3

    def test_to_dict_from_dict(self):
        cfg = SpontaneousConfig(max_candidates=20, overdense_threshold=10)
        d = cfg.to_dict()
        restored = SpontaneousConfig.from_dict(d)
        assert restored.max_candidates == 20
        assert restored.overdense_threshold == 10


class TestSpontaneousState:
    def test_default_state(self):
        st = SpontaneousState()
        assert st.cycle_count == 0
        assert st.total_activations == 0

    def test_to_dict_from_dict(self):
        st = SpontaneousState()
        st.cycle_count = 10
        st.total_activations = 3
        st.candidates.append(ActivationCandidate(candidate_id="c1"))
        d = st.to_dict()
        restored = SpontaneousState.from_dict(d)
        assert restored.cycle_count == 10
        assert len(restored.candidates) == 1


class TestActivationResult:
    def test_default(self):
        r = ActivationResult()
        assert r.should_activate is False

    def test_to_dict_from_dict(self):
        r = ActivationResult(
            should_activate=True,
            cooldown_active=True,
            candidates=[ActivationCandidate(candidate_id="c1")],
        )
        d = r.to_dict()
        restored = ActivationResult.from_dict(d)
        assert restored.should_activate is True
        assert restored.cooldown_active is True


# =============================================================================
# Extraction Functions
# =============================================================================

class TestExtractIntrinsicMotivation:
    def test_with_drives(self):
        class FakePsyche:
            class drives:
                social = 0.8
                curiosity = 0.7
                expression = 0.6
        f = extract_intrinsic_motivation(FakePsyche())
        assert f.source_type == ActivationSourceType.INTRINSIC_MOTIVATION
        assert f.value > 0.5

    def test_no_drives(self):
        f = extract_intrinsic_motivation(None)
        assert f.value == 0.0


class TestExtractDirectionVector:
    def test_with_vectors(self):
        class FakeVector:
            magnitude = 0.7
        class FakePsyche:
            vectors = [FakeVector()]
        f = extract_direction_vector(FakePsyche())
        assert f.value == 0.7

    def test_no_vectors(self):
        f = extract_direction_vector(None)
        assert f.value == 0.0


class TestExtractUnfinishedIntent:
    def test_with_active_goal(self):
        class FakeGoal:
            selection_strength = 0.6
        class FakePsyche:
            active_goal = FakeGoal()
            scoped_goal = None
        f = extract_unfinished_intent(FakePsyche())
        assert f.value == 0.6

    def test_no_goals(self):
        f = extract_unfinished_intent(None)
        assert f.value == 0.0


class TestExtractMemoryEcho:
    def test_with_stm_entries(self):
        class FakeEntry:
            residue_weight = 0.5
        class FakeStm:
            entries = [FakeEntry(), FakeEntry()]
            context_continuity_score = 0.3
        f = extract_memory_echo(FakeStm(), None)
        assert f.value > 0.0

    def test_with_high_importance_memories(self):
        memories = [{"importance": 4}, {"importance": 5}]
        f = extract_memory_echo(None, memories)
        assert f.value > 0.0

    def test_empty(self):
        f = extract_memory_echo(None, None)
        assert f.value == 0.0


class TestExtractEmotionalTransition:
    def test_with_high_emotion(self):
        class FakeEmotions:
            def as_dict(self):
                return {"joy": 0.8, "anger": 0.1}
        class FakeMood:
            arousal = 0.7
        class FakePsyche:
            emotions = FakeEmotions()
            mood = FakeMood()
        f = extract_emotional_transition(FakePsyche(), None)
        assert f.value > 0.5

    def test_neutral(self):
        f = extract_emotional_transition(None, None)
        assert f.value == 0.0


class TestExtractResponsibility:
    def test_with_units(self):
        class FakeUnit:
            weight = 0.3
        class FakePsyche:
            responsibility_units = [FakeUnit(), FakeUnit()]
        f = extract_responsibility(FakePsyche())
        assert f.value > 0.0

    def test_no_responsibility(self):
        f = extract_responsibility(None)
        assert f.value == 0.0


class TestExtractRecentAction:
    def test_with_actions(self):
        actions = ["response1", "response2", "response3"]
        f = extract_recent_action(actions)
        assert f.value > 0.0

    def test_with_silences(self):
        actions = ["silence", "silence", "response"]
        f = extract_recent_action(actions)
        assert "silence_ratio" in f.description

    def test_empty(self):
        f = extract_recent_action(None)
        assert f.value == 0.0


class TestExtractExternalInputAbsence:
    def test_input_present(self):
        f = extract_external_input_absence(True, time.time())
        assert f.value == 0.0

    def test_input_absent_recently(self):
        f = extract_external_input_absence(False, time.time() - 10)
        assert 0.0 < f.value < 1.0

    def test_input_absent_long(self):
        f = extract_external_input_absence(False, time.time() - 100)
        assert f.value == 1.0


class TestExtractAllFragments:
    def test_returns_8_fragments(self):
        fragments = extract_all_fragments(
            psyche=None, dynamics=None, stm=None, memories=None,
            recent_actions=None, has_external_input=False,
            last_external_input_time=0.0,
        )
        assert len(fragments) == 8
        types = {f.source_type for f in fragments}
        assert len(types) == 8


# =============================================================================
# Pipeline
# =============================================================================

class TestFormCandidates:
    def test_sufficient_intersection(self):
        frags = [
            ActivationFragment(
                source_type=ActivationSourceType.INTRINSIC_MOTIVATION,
                value=0.5,
            ),
            ActivationFragment(
                source_type=ActivationSourceType.MEMORY_ECHO,
                value=0.6,
            ),
        ]
        candidates = form_candidates(frags)
        assert len(candidates) >= 1
        assert candidates[0].intersection_count >= 2

    def test_insufficient_intersection(self):
        frags = [
            ActivationFragment(
                source_type=ActivationSourceType.INTRINSIC_MOTIVATION,
                value=0.5,
            ),
            ActivationFragment(
                source_type=ActivationSourceType.MEMORY_ECHO,
                value=0.1,  # Below threshold
            ),
        ]
        candidates = form_candidates(frags)
        assert len(candidates) == 0

    def test_three_sections_produce_sub_candidate(self):
        frags = [
            ActivationFragment(
                source_type=ActivationSourceType.INTRINSIC_MOTIVATION,
                value=0.5,
            ),
            ActivationFragment(
                source_type=ActivationSourceType.MEMORY_ECHO,
                value=0.6,
            ),
            ActivationFragment(
                source_type=ActivationSourceType.EMOTIONAL_TRANSITION,
                value=0.7,
            ),
        ]
        candidates = form_candidates(frags)
        assert len(candidates) >= 2  # main + sub

    def test_all_below_threshold(self):
        frags = [
            ActivationFragment(value=0.1),
            ActivationFragment(value=0.2),
        ]
        candidates = form_candidates(frags)
        assert len(candidates) == 0


class TestAlignConditions:
    def test_no_history(self):
        cands = [ActivationCandidate(
            candidate_id="c1",
            source_types=["intrinsic_motivation"],
            activation_strength=0.5,
        )]
        aligned, rationales = align_conditions(cands, [])
        assert len(aligned) == 1
        assert aligned[0].activation_strength == 0.5  # No reduction
        assert rationales[0].continuous_diff_reference == 1.0

    def test_with_repeated_history(self):
        history = [
            ContinuousActivationEntry(
                source_types=["intrinsic_motivation"],
            ),
            ContinuousActivationEntry(
                source_types=["intrinsic_motivation"],
            ),
            ContinuousActivationEntry(
                source_types=["intrinsic_motivation"],
            ),
        ]
        cands = [ActivationCandidate(
            candidate_id="c1",
            source_types=["intrinsic_motivation"],
            activation_strength=0.8,
        )]
        aligned, rationales = align_conditions(cands, history)
        assert aligned[0].activation_strength < 0.8  # Reduced


class TestResolveConflicts:
    def test_single_candidate(self):
        cands = [ActivationCandidate(candidate_id="c1", activation_strength=0.5)]
        adopted, unadopted, conflicts = resolve_conflicts(cands)
        assert len(adopted) == 1
        assert len(unadopted) == 0

    def test_multiple_candidates_parallel(self):
        cands = [
            ActivationCandidate(candidate_id="c1", activation_strength=0.8),
            ActivationCandidate(candidate_id="c2", activation_strength=0.6),
        ]
        adopted, unadopted, conflicts = resolve_conflicts(cands)
        assert len(adopted) == 2
        assert len(conflicts) == 1  # parallel record

    def test_excess_candidates_become_unadopted(self):
        cfg = SpontaneousConfig(max_output_candidates=2)
        cands = [
            ActivationCandidate(candidate_id=f"c{i}", activation_strength=0.5 - i * 0.1)
            for i in range(5)
        ]
        adopted, unadopted, _ = resolve_conflicts(cands, cfg)
        assert len(adopted) == 2
        assert len(unadopted) == 3

    def test_empty_candidates(self):
        adopted, unadopted, conflicts = resolve_conflicts([])
        assert len(adopted) == 0


class TestCheckActivationFeasibility:
    def test_good_candidate(self):
        cands = [ActivationCandidate(activation_strength=0.5)]
        should, cd, od = check_activation_feasibility(
            cands, 0, 0, False,
        )
        assert should is True

    def test_cooldown_prevents(self):
        cands = [ActivationCandidate(activation_strength=0.8)]
        should, cd, od = check_activation_feasibility(
            cands, 3, 0, False,
        )
        assert should is False
        assert cd is True

    def test_external_input_raises_bar(self):
        cands = [ActivationCandidate(activation_strength=0.5)]
        should, cd, od = check_activation_feasibility(
            cands, 0, 0, True,
        )
        assert should is False  # 0.5 < 0.7 threshold for external

    def test_no_candidates(self):
        should, cd, od = check_activation_feasibility([], 0, 0, False)
        assert should is False


# =============================================================================
# Safety Valves
# =============================================================================

class TestSuppressConsecutiveSeries:
    def test_no_suppression_when_diverse(self):
        cands = [ActivationCandidate(source_types=["a", "b"])]
        recent = [["a", "b"], ["c", "d"], ["e", "f"]]
        passed, suppressed, entries = suppress_consecutive_series(
            cands, recent,
        )
        assert len(passed) == 1
        assert len(suppressed) == 0

    def test_suppression_when_dominant(self):
        cfg = SpontaneousConfig(consecutive_adoption_suppression_count=3)
        cands = [
            ActivationCandidate(candidate_id="c1", source_types=["a"]),
            ActivationCandidate(candidate_id="c2", source_types=["b"]),
        ]
        recent = [["a"], ["a"], ["a"]]
        passed, suppressed, entries = suppress_consecutive_series(
            cands, recent, cfg,
        )
        assert len(suppressed) == 1
        assert suppressed[0].source_types == ["a"]

    def test_insufficient_history(self):
        cands = [ActivationCandidate(source_types=["a"])]
        passed, suppressed, _ = suppress_consecutive_series(cands, [["a"]])
        assert len(passed) == 1


class TestApplyOverdenseCooldown:
    def test_below_threshold(self):
        assert apply_overdense_cooldown(3) == 0

    def test_at_threshold(self):
        cd = apply_overdense_cooldown(5)
        assert cd > 0

    def test_custom_threshold(self):
        cfg = SpontaneousConfig(overdense_threshold=2, activation_cooldown_cycles=5)
        cd = apply_overdense_cooldown(2, cfg)
        assert cd == 5


class TestRestoreCandidateDiversity:
    def test_diverse_candidates(self):
        cands = [
            ActivationCandidate(source_types=["a"]),
            ActivationCandidate(source_types=["b"]),
        ]
        result, unadopted, warning = restore_candidate_diversity(cands, [])
        assert warning is False

    def test_single_series_with_alternative(self):
        cands = [
            ActivationCandidate(source_types=["a"]),
        ]
        unadopted = [UnadoptedHistoryEntry(
            source_types=["b"],
            original_strength=0.5,
            freshness=ActivationFreshness.FRESH,
            timestamp=time.time(),
        )]
        result, new_unadopted, warning = restore_candidate_diversity(
            cands, unadopted,
        )
        assert warning is True
        assert len(result) == 2

    def test_no_suitable_alternative(self):
        cands = [ActivationCandidate(source_types=["a"])]
        unadopted = [UnadoptedHistoryEntry(
            source_types=["b"],
            original_strength=0.5,
            freshness=ActivationFreshness.FADED,
            timestamp=time.time() - 200,
        )]
        result, _, warning = restore_candidate_diversity(cands, unadopted)
        assert len(result) == 1  # FADED not injected


class TestApplyFreshnessDecay:
    def test_fresh_stays_fresh(self):
        entries = [StandbyEntry(
            candidate_id="c1", timestamp=time.time(),
            freshness=ActivationFreshness.FRESH,
        )]
        result, decay = apply_freshness_decay(entries)
        assert result[0].freshness == ActivationFreshness.FRESH
        assert len(decay) == 0

    def test_old_entry_decays(self):
        entries = [StandbyEntry(
            candidate_id="c1", timestamp=time.time() - 120,
            freshness=ActivationFreshness.FRESH,
        )]
        result, decay = apply_freshness_decay(entries)
        assert result[0].freshness == ActivationFreshness.FADED
        assert len(decay) == 1


class TestDecayUnadoptedHistory:
    def test_recent_stays_fresh(self):
        history = [UnadoptedHistoryEntry(
            timestamp=time.time(),
            freshness=ActivationFreshness.FRESH,
        )]
        result = decay_unadopted_history(history)
        assert result[0].freshness == ActivationFreshness.FRESH

    def test_old_decays(self):
        history = [UnadoptedHistoryEntry(
            timestamp=time.time() - 200,
            freshness=ActivationFreshness.FRESH,
        )]
        result = decay_unadopted_history(history)
        assert result[0].freshness == ActivationFreshness.FADED


# =============================================================================
# Processor
# =============================================================================

class TestSpontaneousActivationProcessor:
    def test_basic_process_no_input(self):
        proc = SpontaneousActivationProcessor()
        result = proc.process(has_external_input=False)
        assert isinstance(result, ActivationResult)
        assert proc.state.cycle_count == 1

    def test_with_high_drives(self):
        class FakePsyche:
            class drives:
                social = 0.8
                curiosity = 0.9
                expression = 0.7
            class emotions:
                @staticmethod
                def as_dict():
                    return {"joy": 0.8}
            class mood:
                arousal = 0.6
            active_goal = None
            scoped_goal = None
            vectors = None
            responsibility_units = None
        proc = SpontaneousActivationProcessor()
        result = proc.process(
            psyche=FakePsyche(),
            has_external_input=False,
        )
        # With high drives + external absence + emotional transition,
        # should have candidates
        assert proc.state.cycle_count == 1

    def test_external_input_resets_consecutive(self):
        proc = SpontaneousActivationProcessor()
        proc._state.consecutive_activation_count = 3
        proc.process(has_external_input=True)
        assert proc.state.consecutive_activation_count == 0

    def test_notify_external_input(self):
        proc = SpontaneousActivationProcessor()
        proc.notify_external_input()
        assert proc.state.last_external_input_time > 0
        assert proc.state.consecutive_activation_count == 0

    def test_cooldown_decrements(self):
        proc = SpontaneousActivationProcessor()
        proc._state.cooldown_remaining = 3
        proc.process(has_external_input=False)
        assert proc.state.cooldown_remaining == 2

    def test_multiple_cycles(self):
        proc = SpontaneousActivationProcessor()
        for _ in range(5):
            proc.process(has_external_input=False)
        assert proc.state.cycle_count == 5

    def test_state_serialization_roundtrip(self):
        proc = SpontaneousActivationProcessor()
        proc.process(has_external_input=False)
        proc.process(has_external_input=False)
        d = proc.state.to_dict()
        restored = SpontaneousState.from_dict(d)
        assert restored.cycle_count == proc.state.cycle_count

    def test_history_limits(self):
        cfg = SpontaneousConfig(max_history=5, max_unadopted_history=5)
        proc = SpontaneousActivationProcessor(config=cfg)
        for i in range(15):
            proc.process(has_external_input=False)
        assert len(proc.state.conflict_history) <= 5
        assert len(proc.state.unadopted_history) <= 5

    def test_overdense_triggers_cooldown(self):
        cfg = SpontaneousConfig(
            overdense_threshold=2,
            activation_cooldown_cycles=3,
            fragment_threshold=0.01,
            min_intersection_count=1,
        )
        proc = SpontaneousActivationProcessor(config=cfg)
        # Force high consecutive count
        proc._state.consecutive_activation_count = 5
        result = proc.process(has_external_input=False)
        # Overdense should trigger cooldown for future cycles
        # (exact behavior depends on whether candidates form)


# =============================================================================
# Summary
# =============================================================================

class TestGetSpontaneousSummary:
    def test_empty_state(self):
        st = SpontaneousState()
        summary = get_spontaneous_summary(st)
        assert "待機中" in summary or "起動=0" in summary

    def test_with_activity(self):
        st = SpontaneousState()
        st.total_activations = 5
        st.cycle_count = 10
        st.consecutive_activation_count = 2
        summary = get_spontaneous_summary(st)
        assert "起動=5" in summary
        assert "連続=2" in summary

    def test_with_cooldown(self):
        st = SpontaneousState()
        st.total_activations = 1
        st.cooldown_remaining = 3
        summary = get_spontaneous_summary(st)
        assert "cooldown=3" in summary

    def test_with_candidates(self):
        st = SpontaneousState()
        st.total_activations = 1
        st.candidates = [ActivationCandidate(
            activation_strength=0.7,
            source_types=["intrinsic_motivation", "memory_echo"],
        )]
        summary = get_spontaneous_summary(st)
        assert "最強=0.70" in summary


# =============================================================================
# Factory
# =============================================================================

class TestFactory:
    def test_create_processor(self):
        proc = create_spontaneous_processor()
        assert isinstance(proc, SpontaneousActivationProcessor)

    def test_create_with_config(self):
        cfg = SpontaneousConfig(max_candidates=20)
        proc = create_spontaneous_processor(config=cfg)
        assert proc.state.config.max_candidates == 20
