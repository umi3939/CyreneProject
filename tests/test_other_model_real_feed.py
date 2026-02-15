"""tests/test_other_model_real_feed.py - 他者モデルリアルフィード統合テスト"""

import time

import pytest

from psyche.other_model_real_feed import (
    # Enums
    ObservationFragmentType,
    FragmentFreshness,
    AlignmentStatus,
    ConflictStatus,
    # Dataclasses
    ObservationFragment,
    ObservationUnit,
    ConflictRecord,
    FeedHistoryEntry,
    HoldbackEntry,
    RealFeedConfig,
    RealFeedState,
    FeedResult,
    # Extraction functions
    extract_speech_reaction,
    extract_response_interval,
    extract_topic_transition,
    extract_emotional_tone,
    extract_continued_engagement,
    extract_rejection_acceptance,
    extract_context_alignment,
    extract_recent_history,
    # Pipeline functions
    normalize_fragments,
    align_units,
    detect_feed_duplicates,
    detect_feed_conflicts,
    apply_freshness,
    suppress_recent_series,
    ensure_type_diversity,
    check_convergence,
    check_stagnation,
    # Processor
    RealFeedProcessor,
    create_real_feed_processor,
    # Output
    enhance_context_with_feed,
    # Summary
    get_real_feed_summary,
)


# =============================================================================
# Duck-typing helpers
# =============================================================================

class MockPercept:
    def __init__(self, text="", topics=None, emotion_valence=0.0, intent="unknown"):
        self.text = text
        self.topics = topics or []
        self.emotion_valence = emotion_valence
        self.intent = intent


class MockSTMEntry:
    def __init__(self, source_text="", intent="unknown", emotion_label="neutral",
                 valence=0.0, timestamp=0.0):
        self.source_text = source_text
        self.intent = intent
        self.emotion_label = emotion_label
        self.valence = valence
        self.timestamp = timestamp


class MockSTM:
    def __init__(self, entries=None, max_entries=10, context_continuity_score=0.5):
        self.entries = entries or []
        self.max_entries = max_entries
        self.context_continuity_score = context_continuity_score


class MockMood:
    def __init__(self, valence=0.0, arousal=0.5):
        self.valence = valence
        self.arousal = arousal


class MockDrives:
    def __init__(self, social=0.5, curiosity=0.5, expression=0.5):
        self.social = social
        self.curiosity = curiosity
        self.expression = expression


class MockPsyche:
    def __init__(self, mood=None, drives=None):
        self.mood = mood or MockMood()
        self.drives = drives or MockDrives()


class MockContextSnapshot:
    def __init__(self, pace=0.5, weight=0.5, density=0.5,
                 continuity=0.5, responsiveness=0.5):
        self.pace = pace
        self.weight = weight
        self.density = density
        self.continuity = continuity
        self.responsiveness = responsiveness


# =============================================================================
# Enum Tests
# =============================================================================

class TestObservationFragmentType:
    def test_values(self):
        assert len(ObservationFragmentType) == 8

    def test_all_values(self):
        expected = {
            "speech_reaction", "response_interval", "topic_transition",
            "emotional_tone", "continued_engagement", "rejection_acceptance",
            "context_alignment", "recent_history",
        }
        actual = {ft.value for ft in ObservationFragmentType}
        assert actual == expected


class TestFragmentFreshness:
    def test_values(self):
        assert len(FragmentFreshness) == 5

    def test_all_values(self):
        expected = {"fresh", "recent", "aging", "stale", "faded"}
        actual = {f.value for f in FragmentFreshness}
        assert actual == expected


class TestAlignmentStatus:
    def test_values(self):
        assert len(AlignmentStatus) == 4

    def test_all_values(self):
        expected = {"aligned", "partial", "unaligned", "unknown"}
        actual = {a.value for a in AlignmentStatus}
        assert actual == expected


class TestConflictStatus:
    def test_values(self):
        assert len(ConflictStatus) == 3

    def test_all_values(self):
        expected = {"none", "parallel", "convergence_risk"}
        actual = {c.value for c in ConflictStatus}
        assert actual == expected


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestObservationFragment:
    def test_defaults(self):
        f = ObservationFragment()
        assert f.fragment_id != ""
        assert f.type == ObservationFragmentType.SPEECH_REACTION
        assert f.value == 0.5
        assert f.freshness == FragmentFreshness.FRESH

    def test_to_dict_from_dict(self):
        f = ObservationFragment(
            fragment_id="test123",
            type=ObservationFragmentType.EMOTIONAL_TONE,
            source_description="test",
            value=0.7,
            text_hint="hint",
            freshness=FragmentFreshness.RECENT,
            timestamp=1000.0,
        )
        d = f.to_dict()
        f2 = ObservationFragment.from_dict(d)
        assert f2.fragment_id == "test123"
        assert f2.type == ObservationFragmentType.EMOTIONAL_TONE
        assert f2.value == pytest.approx(0.7)
        assert f2.freshness == FragmentFreshness.RECENT

    def test_from_dict_invalid_type(self):
        f = ObservationFragment.from_dict({"type": "invalid_type"})
        assert f.type == ObservationFragmentType.SPEECH_REACTION

    def test_from_dict_invalid_freshness(self):
        f = ObservationFragment.from_dict({"freshness": "invalid"})
        assert f.freshness == FragmentFreshness.FRESH


class TestObservationUnit:
    def test_defaults(self):
        u = ObservationUnit()
        assert u.unit_id != ""
        assert u.alignment == AlignmentStatus.UNKNOWN
        assert u.conflict_status == ConflictStatus.NONE

    def test_to_dict_from_dict(self):
        u = ObservationUnit(
            unit_id="u001",
            source_fragment_ids=["f1", "f2"],
            source_types=["speech_reaction"],
            value=0.8,
            alignment=AlignmentStatus.ALIGNED,
            conflict_status=ConflictStatus.PARALLEL,
            competing_unit_ids=["u002"],
            timestamp=2000.0,
        )
        d = u.to_dict()
        u2 = ObservationUnit.from_dict(d)
        assert u2.unit_id == "u001"
        assert u2.source_fragment_ids == ["f1", "f2"]
        assert u2.alignment == AlignmentStatus.ALIGNED
        assert u2.conflict_status == ConflictStatus.PARALLEL
        assert u2.competing_unit_ids == ["u002"]

    def test_from_dict_invalid_enums(self):
        u = ObservationUnit.from_dict({
            "alignment": "invalid",
            "conflict_status": "invalid",
            "freshness": "invalid",
        })
        assert u.alignment == AlignmentStatus.UNKNOWN
        assert u.conflict_status == ConflictStatus.NONE
        assert u.freshness == FragmentFreshness.FRESH


class TestConflictRecord:
    def test_defaults(self):
        c = ConflictRecord()
        assert c.conflict_id != ""
        assert c.severity == 0.0

    def test_to_dict_from_dict(self):
        c = ConflictRecord(
            conflict_id="c001",
            unit_id_a="u1",
            unit_id_b="u2",
            conflict_aspect="emotional_tone",
            severity=0.6,
        )
        d = c.to_dict()
        c2 = ConflictRecord.from_dict(d)
        assert c2.conflict_id == "c001"
        assert c2.severity == pytest.approx(0.6)


class TestFeedHistoryEntry:
    def test_defaults(self):
        h = FeedHistoryEntry()
        assert h.unit_ids == []
        assert h.cycle_id == 0

    def test_to_dict_from_dict(self):
        h = FeedHistoryEntry(
            unit_ids=["u1"], source_types=["speech_reaction"],
            timestamp=1000.0, cycle_id=5,
        )
        d = h.to_dict()
        h2 = FeedHistoryEntry.from_dict(d)
        assert h2.cycle_id == 5


class TestHoldbackEntry:
    def test_defaults(self):
        h = HoldbackEntry()
        assert h.reason == ""

    def test_to_dict_from_dict(self):
        h = HoldbackEntry(
            unit_id="u1", source_type="emotional_tone",
            value=0.3, reason="suppression", timestamp=500.0,
        )
        d = h.to_dict()
        h2 = HoldbackEntry.from_dict(d)
        assert h2.reason == "suppression"


class TestRealFeedConfig:
    def test_defaults(self):
        c = RealFeedConfig()
        assert c.max_fragments_per_type == 5
        assert c.max_observation_units == 20
        assert c.max_output_units == 10
        assert c.single_type_dominance_cap == pytest.approx(0.4)
        assert c.freshness_decay_rate == pytest.approx(0.05)
        assert c.stale_threshold == pytest.approx(0.15)
        assert c.recent_series_suppression_count == 3
        assert c.convergence_inject_threshold == 1
        assert c.stagnation_cycle_threshold == 5

    def test_to_dict_from_dict(self):
        c = RealFeedConfig(max_output_units=15, stagnation_cycle_threshold=10)
        d = c.to_dict()
        c2 = RealFeedConfig.from_dict(d)
        assert c2.max_output_units == 15
        assert c2.stagnation_cycle_threshold == 10


class TestRealFeedState:
    def test_defaults(self):
        s = RealFeedState()
        assert s.cycle_count == 0
        assert s.total_feeds == 0
        assert s.stagnation_counter == 0
        assert s.convergence_warnings == 0

    def test_to_dict_from_dict(self):
        s = RealFeedState(cycle_count=10, total_feeds=50)
        s.fragments.append(ObservationFragment(fragment_id="f1"))
        s.units.append(ObservationUnit(unit_id="u1"))
        d = s.to_dict()
        s2 = RealFeedState.from_dict(d)
        assert s2.cycle_count == 10
        assert len(s2.fragments) == 1
        assert len(s2.units) == 1


class TestFeedResult:
    def test_defaults(self):
        r = FeedResult()
        assert r.units == []
        assert r.convergence_warning is False

    def test_to_dict_from_dict(self):
        r = FeedResult(
            units=[ObservationUnit(unit_id="u1")],
            source_distribution={"speech_reaction": 1},
            convergence_warning=True,
            holdback_count=3,
        )
        d = r.to_dict()
        r2 = FeedResult.from_dict(d)
        assert len(r2.units) == 1
        assert r2.convergence_warning is True
        assert r2.holdback_count == 3


# =============================================================================
# Extraction Function Tests
# =============================================================================

class TestExtractSpeechReaction:
    def test_with_percept_and_stm(self):
        percept = MockPercept(text="Hello world", emotion_valence=0.5, intent="greeting")
        entry = MockSTMEntry(source_text="Hi there", emotion_label="joy", intent="greeting")
        stm = MockSTM(entries=[entry])
        frag = extract_speech_reaction(percept, stm)
        assert frag is not None
        assert frag.type == ObservationFragmentType.SPEECH_REACTION
        assert 0.0 <= frag.value <= 1.0

    def test_none_inputs(self):
        frag = extract_speech_reaction(None, None)
        assert frag is None

    def test_empty_percept(self):
        percept = MockPercept(text="", emotion_valence=0.0)
        frag = extract_speech_reaction(percept, None)
        assert frag is None

    def test_high_valence(self):
        percept = MockPercept(text="Amazing!", emotion_valence=0.9, intent="praise")
        frag = extract_speech_reaction(percept, None)
        assert frag is not None
        assert frag.value > 0.5


class TestExtractResponseInterval:
    def test_with_multiple_entries(self):
        now = time.time()
        entries = [
            MockSTMEntry(timestamp=now - 30),
            MockSTMEntry(timestamp=now - 20),
            MockSTMEntry(timestamp=now - 5),
        ]
        stm = MockSTM(entries=entries)
        frag = extract_response_interval(stm, 10)
        assert frag is not None
        assert frag.type == ObservationFragmentType.RESPONSE_INTERVAL

    def test_none_stm(self):
        assert extract_response_interval(None, 0) is None

    def test_single_entry(self):
        stm = MockSTM(entries=[MockSTMEntry(timestamp=time.time())])
        assert extract_response_interval(stm, 0) is None

    def test_fast_responses(self):
        now = time.time()
        entries = [
            MockSTMEntry(timestamp=now - 5),
            MockSTMEntry(timestamp=now - 3),
            MockSTMEntry(timestamp=now),
        ]
        stm = MockSTM(entries=entries)
        frag = extract_response_interval(stm, 3)
        assert frag is not None
        assert frag.value >= 0.7  # Fast responses = high value


class TestExtractTopicTransition:
    def test_with_topics(self):
        percept = MockPercept(topics=["gaming", "music"])
        stm = MockSTM(entries=[MockSTMEntry(source_text="I love gaming")])
        frag = extract_topic_transition(percept, stm)
        assert frag is not None
        assert frag.type == ObservationFragmentType.TOPIC_TRANSITION

    def test_none_percept(self):
        assert extract_topic_transition(None, None) is None

    def test_no_topics(self):
        percept = MockPercept(topics=[])
        assert extract_topic_transition(percept, None) is None

    def test_no_overlap(self):
        percept = MockPercept(topics=["cooking", "travel"])
        stm = MockSTM(entries=[MockSTMEntry(source_text="gaming discussion")])
        frag = extract_topic_transition(percept, stm)
        assert frag is not None
        # No overlap = high transition
        assert frag.value >= 0.5


class TestExtractEmotionalTone:
    def test_with_percept_and_psyche(self):
        percept = MockPercept(emotion_valence=0.6)
        psyche = MockPsyche(mood=MockMood(valence=0.4, arousal=0.7))
        frag = extract_emotional_tone(percept, psyche)
        assert frag is not None
        assert frag.type == ObservationFragmentType.EMOTIONAL_TONE
        assert 0.0 <= frag.value <= 1.0

    def test_none_inputs(self):
        assert extract_emotional_tone(None, None) is None

    def test_psyche_only(self):
        psyche = MockPsyche(mood=MockMood(valence=-0.3, arousal=0.8))
        frag = extract_emotional_tone(None, psyche)
        assert frag is not None

    def test_percept_only(self):
        percept = MockPercept(emotion_valence=0.5)
        frag = extract_emotional_tone(percept, None)
        assert frag is not None


class TestExtractContinuedEngagement:
    def test_with_stm_and_psyche(self):
        stm = MockSTM(
            entries=[MockSTMEntry()] * 5,
            context_continuity_score=0.8,
        )
        psyche = MockPsyche(drives=MockDrives(social=0.7))
        frag = extract_continued_engagement(stm, psyche)
        assert frag is not None
        assert frag.type == ObservationFragmentType.CONTINUED_ENGAGEMENT

    def test_none_inputs(self):
        assert extract_continued_engagement(None, None) is None

    def test_high_engagement(self):
        stm = MockSTM(
            entries=[MockSTMEntry()] * 20,
            context_continuity_score=0.9,
        )
        psyche = MockPsyche(drives=MockDrives(social=0.9))
        frag = extract_continued_engagement(stm, psyche)
        assert frag is not None
        assert frag.value > 0.5


class TestExtractRejectionAcceptance:
    def test_positive_intent(self):
        percept = MockPercept(intent="agree", emotion_valence=0.5)
        frag = extract_rejection_acceptance(None, percept)
        assert frag is not None
        assert frag.type == ObservationFragmentType.REJECTION_ACCEPTANCE
        assert frag.value > 0.5  # Positive

    def test_negative_intent(self):
        percept = MockPercept(intent="reject", emotion_valence=-0.5)
        frag = extract_rejection_acceptance(None, percept)
        assert frag is not None
        assert frag.value < 0.5  # Negative

    def test_none_inputs(self):
        assert extract_rejection_acceptance(None, None) is None

    def test_stm_based(self):
        entry = MockSTMEntry(intent="praise", valence=0.6)
        stm = MockSTM(entries=[entry])
        frag = extract_rejection_acceptance(stm, None)
        assert frag is not None


class TestExtractContextAlignment:
    def test_with_overlap(self):
        percept = MockPercept(topics=["gaming", "music"])
        memories = [{"keywords": ["gaming", "stream"]}]
        frag = extract_context_alignment(percept, None, memories)
        assert frag is not None
        assert frag.type == ObservationFragmentType.CONTEXT_ALIGNMENT
        assert frag.value > 0.0

    def test_no_overlap(self):
        percept = MockPercept(topics=["cooking"])
        memories = [{"keywords": ["gaming"]}]
        frag = extract_context_alignment(percept, None, memories)
        assert frag is not None
        assert frag.value == pytest.approx(0.0)

    def test_none_inputs(self):
        assert extract_context_alignment(None, None, None) is None

    def test_topics_only(self):
        percept = MockPercept(topics=["gaming"])
        frag = extract_context_alignment(percept, None, None)
        assert frag is not None
        assert frag.value == pytest.approx(0.3)


class TestExtractRecentHistory:
    def test_with_entries(self):
        entries = [
            MockSTMEntry(valence=0.3, intent="chat"),
            MockSTMEntry(valence=-0.1, intent="question"),
            MockSTMEntry(valence=0.5, intent="praise"),
        ]
        stm = MockSTM(entries=entries)
        frag = extract_recent_history(stm)
        assert frag is not None
        assert frag.type == ObservationFragmentType.RECENT_HISTORY

    def test_none_stm(self):
        assert extract_recent_history(None) is None

    def test_empty_entries(self):
        stm = MockSTM(entries=[])
        assert extract_recent_history(stm) is None


# =============================================================================
# Pipeline Function Tests
# =============================================================================

class TestNormalizeFragments:
    def test_basic(self):
        frags = [
            ObservationFragment(type=ObservationFragmentType.SPEECH_REACTION, value=0.6),
            ObservationFragment(type=ObservationFragmentType.EMOTIONAL_TONE, value=0.8),
        ]
        units = normalize_fragments(frags, RealFeedConfig())
        assert len(units) == 2

    def test_same_type_merged(self):
        frags = [
            ObservationFragment(type=ObservationFragmentType.SPEECH_REACTION, value=0.6),
            ObservationFragment(type=ObservationFragmentType.SPEECH_REACTION, value=0.8),
        ]
        units = normalize_fragments(frags, RealFeedConfig())
        assert len(units) == 1  # Same type merged into one unit
        assert units[0].value == pytest.approx(0.7)

    def test_max_fragments_per_type(self):
        frags = [
            ObservationFragment(type=ObservationFragmentType.SPEECH_REACTION, value=0.1 * i)
            for i in range(10)
        ]
        cfg = RealFeedConfig(max_fragments_per_type=3)
        units = normalize_fragments(frags, cfg)
        assert len(units) == 1
        # Should only use 3 fragments

    def test_empty(self):
        units = normalize_fragments([], RealFeedConfig())
        assert units == []


class TestAlignUnits:
    def test_single_unit(self):
        units = [ObservationUnit(value=0.5)]
        result = align_units(units)
        assert result[0].alignment == AlignmentStatus.ALIGNED

    def test_aligned_units(self):
        units = [
            ObservationUnit(value=0.5),
            ObservationUnit(value=0.52),
        ]
        result = align_units(units)
        for u in result:
            assert u.alignment in (AlignmentStatus.ALIGNED, AlignmentStatus.PARTIAL)

    def test_unaligned_units(self):
        units = [
            ObservationUnit(value=0.1),
            ObservationUnit(value=0.9),
        ]
        result = align_units(units)
        has_unaligned = any(u.alignment == AlignmentStatus.UNALIGNED for u in result)
        assert has_unaligned


class TestDetectFeedDuplicates:
    def test_no_duplicates(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"], value=0.5),
            ObservationUnit(source_types=["emotional_tone"], value=0.5),
        ]
        result = detect_feed_duplicates(units)
        assert len(result) == 2

    def test_merge_similar(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"], value=0.5),
            ObservationUnit(source_types=["speech_reaction"], value=0.55),
        ]
        result = detect_feed_duplicates(units)
        assert len(result) == 1

    def test_keep_different(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"], value=0.1),
            ObservationUnit(source_types=["speech_reaction"], value=0.9),
        ]
        result = detect_feed_duplicates(units)
        assert len(result) == 2  # Values too different to merge

    def test_empty(self):
        assert detect_feed_duplicates([]) == []


class TestDetectFeedConflicts:
    def test_no_conflicts(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"], value=0.5),
            ObservationUnit(source_types=["emotional_tone"], value=0.5),
        ]
        result_units, conflicts = detect_feed_conflicts(units)
        assert len(conflicts) == 0

    def test_detects_conflict(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"], value=0.1),
            ObservationUnit(source_types=["speech_reaction"], value=0.9),
        ]
        result_units, conflicts = detect_feed_conflicts(units)
        assert len(conflicts) == 1
        assert conflicts[0].severity > 0.4
        assert result_units[0].conflict_status == ConflictStatus.PARALLEL
        assert result_units[1].conflict_status == ConflictStatus.PARALLEL

    def test_competing_unit_ids(self):
        units = [
            ObservationUnit(unit_id="a", source_types=["x"], value=0.0),
            ObservationUnit(unit_id="b", source_types=["x"], value=0.9),
        ]
        detect_feed_conflicts(units)
        assert "b" in units[0].competing_unit_ids
        assert "a" in units[1].competing_unit_ids


class TestApplyFreshness:
    def test_fresh_stays_fresh(self):
        units = [ObservationUnit(
            freshness=FragmentFreshness.FRESH,
            timestamp=time.time(),
        )]
        result, decays = apply_freshness(units, RealFeedConfig(), time.time())
        assert result[0].freshness == FragmentFreshness.FRESH

    def test_old_unit_decays(self):
        old_time = time.time() - 600  # 10 minutes ago
        units = [ObservationUnit(
            freshness=FragmentFreshness.FRESH,
            timestamp=old_time,
        )]
        result, decays = apply_freshness(units, RealFeedConfig(), time.time())
        assert result[0].freshness != FragmentFreshness.FRESH

    def test_returns_decay_values(self):
        units = [ObservationUnit(timestamp=time.time())]
        result, decays = apply_freshness(units, RealFeedConfig(), time.time())
        assert len(decays) == 1


class TestSuppressRecentSeries:
    def test_no_history(self):
        units = [ObservationUnit(source_types=["speech_reaction"])]
        result, holdback = suppress_recent_series(units, [], RealFeedConfig())
        assert len(result) == 1
        assert len(holdback) == 0

    def test_suppresses_dominant_type(self):
        history = [
            FeedHistoryEntry(source_types=["speech_reaction"], cycle_id=i)
            for i in range(3)
        ]
        units = [
            ObservationUnit(
                source_types=["speech_reaction"],
                freshness=FragmentFreshness.RECENT,
            ),
            ObservationUnit(
                source_types=["emotional_tone"],
                freshness=FragmentFreshness.FRESH,
            ),
        ]
        cfg = RealFeedConfig(recent_series_suppression_count=3)
        result, holdback = suppress_recent_series(units, history, cfg)
        # speech_reaction should be suppressed (RECENT, not FRESH)
        assert len(holdback) == 1
        assert holdback[0].source_type == "speech_reaction"

    def test_fresh_not_suppressed(self):
        history = [
            FeedHistoryEntry(source_types=["speech_reaction"], cycle_id=i)
            for i in range(3)
        ]
        units = [ObservationUnit(
            source_types=["speech_reaction"],
            freshness=FragmentFreshness.FRESH,
        )]
        cfg = RealFeedConfig(recent_series_suppression_count=3)
        result, holdback = suppress_recent_series(units, history, cfg)
        assert len(result) == 1  # FRESH is kept


class TestEnsureTypeDiversity:
    def test_no_dominance(self):
        units = [
            ObservationUnit(source_types=["speech_reaction"]),
            ObservationUnit(source_types=["emotional_tone"]),
            ObservationUnit(source_types=["topic_transition"]),
        ]
        result, holdback = ensure_type_diversity(units, [], RealFeedConfig())
        assert len(result) == 3

    def test_resurfaces_from_holdback(self):
        units = [ObservationUnit(source_types=["speech_reaction"])]
        holdback = [HoldbackEntry(
            source_type="emotional_tone",
            value=0.5,
            timestamp=time.time(),
        )]
        result, remaining = ensure_type_diversity(
            units, holdback, RealFeedConfig(),
        )
        # emotional_tone should be resurfaced
        types = {st for u in result for st in u.source_types}
        assert "emotional_tone" in types

    def test_empty_units(self):
        result, holdback = ensure_type_diversity([], [], RealFeedConfig())
        assert result == []


class TestCheckConvergence:
    def test_no_convergence(self):
        units = [
            ObservationUnit(value=0.2),
            ObservationUnit(value=0.8),
        ]
        result, warning = check_convergence(units, [], RealFeedConfig())
        assert warning is False

    def test_detects_convergence(self):
        units = [
            ObservationUnit(value=0.5),
            ObservationUnit(value=0.52),
        ]
        holdback = [HoldbackEntry(value=0.1, source_type="test")]
        result, warning = check_convergence(units, holdback, RealFeedConfig())
        assert warning is True
        assert len(result) == 3  # Injected one from holdback

    def test_too_few_units(self):
        units = [ObservationUnit(value=0.5)]
        result, warning = check_convergence(units, [], RealFeedConfig())
        assert warning is False


class TestCheckStagnation:
    def test_no_stagnation(self):
        state = RealFeedState(stagnation_counter=2)
        units = [ObservationUnit()]
        result, warning = check_stagnation(state, units)
        assert warning is False

    def test_detects_stagnation(self):
        state = RealFeedState(stagnation_counter=5)
        units = [ObservationUnit(freshness=FragmentFreshness.FRESH)]
        result, warning = check_stagnation(state, units)
        assert warning is True
        assert result[0].freshness == FragmentFreshness.RECENT  # Downgraded


# =============================================================================
# Processor Tests
# =============================================================================

class TestRealFeedProcessor:
    def test_create(self):
        proc = create_real_feed_processor()
        assert proc.state.cycle_count == 0

    def test_create_with_config(self):
        cfg = RealFeedConfig(max_output_units=5)
        proc = create_real_feed_processor(cfg)
        assert proc.state.config.max_output_units == 5

    def test_process_empty(self):
        proc = RealFeedProcessor()
        result = proc.process()
        assert isinstance(result, FeedResult)
        assert proc.state.cycle_count == 1

    def test_process_with_data(self):
        proc = RealFeedProcessor()
        percept = MockPercept(
            text="Hello!", topics=["gaming"], emotion_valence=0.5,
            intent="greeting",
        )
        entries = [
            MockSTMEntry(source_text="Hi", intent="greeting",
                         emotion_label="joy", valence=0.3,
                         timestamp=time.time() - 10),
            MockSTMEntry(source_text="How are you?", intent="question",
                         emotion_label="neutral", valence=0.0,
                         timestamp=time.time()),
        ]
        stm = MockSTM(entries=entries, context_continuity_score=0.7)
        psyche = MockPsyche(
            mood=MockMood(valence=0.3, arousal=0.6),
            drives=MockDrives(social=0.7),
        )
        memories = [{"keywords": ["gaming", "chat"]}]

        result = proc.process(
            percept=percept, stm=stm, psyche=psyche,
            recalled_memories=memories, tick_count=10,
        )

        assert len(result.units) > 0
        assert proc.state.cycle_count == 1
        assert proc.state.total_feeds > 0

    def test_multiple_cycles(self):
        proc = RealFeedProcessor()
        percept = MockPercept(text="test", emotion_valence=0.3, intent="chat")
        stm = MockSTM(entries=[
            MockSTMEntry(timestamp=time.time() - 5),
            MockSTMEntry(timestamp=time.time()),
        ])
        psyche = MockPsyche()

        for _ in range(5):
            result = proc.process(percept=percept, stm=stm, psyche=psyche)

        assert proc.state.cycle_count == 5
        assert len(proc.state.feed_history) <= 20

    def test_state_persistence(self):
        proc = RealFeedProcessor()
        percept = MockPercept(text="test", emotion_valence=0.5, intent="chat")
        stm = MockSTM(entries=[
            MockSTMEntry(timestamp=time.time() - 5),
            MockSTMEntry(timestamp=time.time()),
        ])
        proc.process(percept=percept, stm=stm)

        # Serialize and restore
        d = proc.state.to_dict()
        proc2 = RealFeedProcessor()
        proc2.state = RealFeedState.from_dict(d)
        assert proc2.state.cycle_count == 1

    def test_source_distribution(self):
        proc = RealFeedProcessor()
        percept = MockPercept(
            text="Hello", topics=["test"], emotion_valence=0.5,
            intent="greeting",
        )
        stm = MockSTM(
            entries=[
                MockSTMEntry(source_text="hi", valence=0.3, intent="greeting",
                             timestamp=time.time() - 5),
                MockSTMEntry(source_text="hey", valence=0.2, intent="chat",
                             timestamp=time.time()),
            ],
            context_continuity_score=0.6,
        )
        psyche = MockPsyche(mood=MockMood(valence=0.3, arousal=0.5))

        result = proc.process(percept=percept, stm=stm, psyche=psyche)
        assert isinstance(result.source_distribution, dict)


# =============================================================================
# Output Integration Tests
# =============================================================================

class TestEnhanceContextWithFeed:
    def test_no_feed(self):
        ctx = MockContextSnapshot(pace=0.5)
        result = enhance_context_with_feed(ctx, FeedResult())
        assert result.pace == pytest.approx(0.5)

    def test_adjusts_responsiveness(self):
        ctx = MockContextSnapshot(responsiveness=0.3)
        feed = FeedResult(units=[
            ObservationUnit(
                source_types=["continued_engagement"],
                value=0.9,
            ),
        ])
        result = enhance_context_with_feed(ctx, feed)
        assert result.responsiveness > 0.3  # Boosted

    def test_adjusts_weight(self):
        ctx = MockContextSnapshot(weight=0.3)
        feed = FeedResult(units=[
            ObservationUnit(
                source_types=["emotional_tone"],
                value=0.9,
            ),
        ])
        result = enhance_context_with_feed(ctx, feed)
        assert result.weight > 0.3

    def test_adjusts_density(self):
        ctx = MockContextSnapshot(density=0.3)
        feed = FeedResult(units=[
            ObservationUnit(
                source_types=["topic_transition"],
                value=0.9,
            ),
        ])
        result = enhance_context_with_feed(ctx, feed)
        assert result.density > 0.3

    def test_adjusts_pace(self):
        ctx = MockContextSnapshot(pace=0.3)
        feed = FeedResult(units=[
            ObservationUnit(
                source_types=["response_interval"],
                value=0.9,
            ),
        ])
        result = enhance_context_with_feed(ctx, feed)
        assert result.pace > 0.3

    def test_values_clamped(self):
        ctx = MockContextSnapshot(pace=0.95)
        feed = FeedResult(units=[
            ObservationUnit(
                source_types=["response_interval"],
                value=1.0,
            ),
        ])
        result = enhance_context_with_feed(ctx, feed)
        assert result.pace <= 1.0

    def test_does_not_overwrite(self):
        ctx = MockContextSnapshot(pace=0.5, weight=0.5, density=0.5, responsiveness=0.5)
        feed = FeedResult(units=[
            ObservationUnit(source_types=["speech_reaction"], value=0.8),
        ])
        result = enhance_context_with_feed(ctx, feed)
        # speech_reaction doesn't map to any of the 4 context fields
        assert result.pace == pytest.approx(0.5)
        assert result.weight == pytest.approx(0.5)


# =============================================================================
# Summary Tests
# =============================================================================

class TestGetRealFeedSummary:
    def test_inactive(self):
        proc = RealFeedProcessor()
        s = get_real_feed_summary(proc)
        assert "inactive" in s

    def test_active(self):
        proc = RealFeedProcessor()
        proc.state.cycle_count = 3
        proc.state.units = [ObservationUnit(source_types=["speech_reaction"])]
        s = get_real_feed_summary(proc)
        assert "cycle=3" in s
        assert "units=1" in s

    def test_with_warnings(self):
        proc = RealFeedProcessor()
        proc.state.cycle_count = 1
        proc.state.convergence_warnings = 2
        proc.state.stagnation_counter = 3
        s = get_real_feed_summary(proc)
        assert "conv_warn=2" in s
        assert "stagnation=3" in s
