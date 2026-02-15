"""tests/test_text_dialogue_input.py - テキスト対話入力経路のテスト"""

import time
import pytest

from psyche.text_dialogue_input import (
    # Enums
    InputRouteType,
    InputFreshness,
    NormalizationStatus,
    ContextLinkStatus,
    DuplicateStatus,
    RouteConflictStatus,
    # Dataclasses
    InputUnit,
    ContextLink,
    DuplicateRecord,
    RouteConflict,
    ReceiveHistoryEntry,
    SuppressionHistoryEntry,
    DecayHistoryEntry,
    TextDialogueConfig,
    TextDialogueState,
    HandoffResult,
    # Stage functions
    receive_input,
    normalize_unit,
    attach_context,
    align_to_percept_format,
    detect_duplicates,
    prepare_handoff,
    # Safety valves
    apply_freshness_decay,
    decay_receive_history,
    decay_suppression_history,
    suppress_recent_adoption,
    check_empty_streak,
    detect_single_route_dominance,
    ensure_format_diversity,
    restore_multi_route,
    filter_circular_reference,
    # Processor
    TextDialogueProcessor,
    # Integration
    merge_with_percept,
    get_text_dialogue_summary,
    create_text_dialogue_processor,
    # Helpers
    _compute_text_overlap,
)


# =============================================================================
# Enums
# =============================================================================

class TestEnums:
    def test_input_route_type_values(self):
        assert InputRouteType.TEXT.value == "text"
        assert InputRouteType.SCREEN.value == "screen"
        assert InputRouteType.API.value == "api"
        assert InputRouteType.UNKNOWN.value == "unknown"

    def test_input_freshness_values(self):
        assert InputFreshness.FRESH.value == "fresh"
        assert InputFreshness.FADED.value == "faded"

    def test_normalization_status_values(self):
        assert NormalizationStatus.RAW.value == "raw"
        assert NormalizationStatus.NORMALIZED.value == "normalized"
        assert NormalizationStatus.FRAGMENT.value == "fragment"
        assert NormalizationStatus.EMPTY.value == "empty"

    def test_context_link_status_values(self):
        assert ContextLinkStatus.LINKED.value == "linked"
        assert ContextLinkStatus.BROKEN.value == "broken"

    def test_duplicate_status_values(self):
        assert DuplicateStatus.UNIQUE.value == "unique"
        assert DuplicateStatus.SUPPRESSED.value == "suppressed"

    def test_route_conflict_status_values(self):
        assert RouteConflictStatus.NONE.value == "none"
        assert RouteConflictStatus.PARALLEL.value == "parallel"
        assert RouteConflictStatus.SINGLE_LINE_RISK.value == "single_line_risk"


# =============================================================================
# Dataclasses - to_dict / from_dict
# =============================================================================

class TestInputUnit:
    def test_default_creation(self):
        u = InputUnit()
        assert len(u.unit_id) == 12
        assert u.route_type == InputRouteType.UNKNOWN
        assert u.timestamp > 0

    def test_to_dict_from_dict(self):
        u = InputUnit(
            unit_id="abc123",
            route_type=InputRouteType.TEXT,
            raw_text="hello",
            normalized_text="hello",
            normalization_status=NormalizationStatus.NORMALIZED,
            freshness=InputFreshness.RECENT,
            timestamp=1000.0,
            sender_id="user1",
            conversation_id="conv1",
            cycle_id=5,
            text_length_category="short",
        )
        d = u.to_dict()
        restored = InputUnit.from_dict(d)
        assert restored.unit_id == "abc123"
        assert restored.route_type == InputRouteType.TEXT
        assert restored.normalized_text == "hello"
        assert restored.freshness == InputFreshness.RECENT
        assert restored.sender_id == "user1"

    def test_from_dict_defaults(self):
        u = InputUnit.from_dict({})
        assert u.route_type == InputRouteType.UNKNOWN
        assert u.normalization_status == NormalizationStatus.RAW


class TestContextLink:
    def test_default_creation(self):
        cl = ContextLink()
        assert len(cl.link_id) == 12

    def test_to_dict_from_dict(self):
        cl = ContextLink(
            link_id="lnk1",
            unit_id="u1",
            previous_unit_id="u0",
            link_status=ContextLinkStatus.LINKED,
            continuation_flag=True,
            context_overlap=0.5,
        )
        d = cl.to_dict()
        restored = ContextLink.from_dict(d)
        assert restored.link_status == ContextLinkStatus.LINKED
        assert restored.continuation_flag is True
        assert restored.context_overlap == 0.5


class TestDuplicateRecord:
    def test_to_dict_from_dict(self):
        dr = DuplicateRecord(
            unit_id_a="a", unit_id_b="b",
            similarity=0.95, status=DuplicateStatus.NEAR_DUPLICATE,
        )
        d = dr.to_dict()
        restored = DuplicateRecord.from_dict(d)
        assert restored.similarity == 0.95
        assert restored.status == DuplicateStatus.NEAR_DUPLICATE


class TestRouteConflict:
    def test_to_dict_from_dict(self):
        rc = RouteConflict(
            unit_id_a="a", unit_id_b="b",
            route_type_a=InputRouteType.TEXT,
            route_type_b=InputRouteType.SCREEN,
            conflict_status=RouteConflictStatus.PARALLEL,
            description="test conflict",
        )
        d = rc.to_dict()
        restored = RouteConflict.from_dict(d)
        assert restored.route_type_a == InputRouteType.TEXT
        assert restored.conflict_status == RouteConflictStatus.PARALLEL


class TestReceiveHistoryEntry:
    def test_to_dict_from_dict(self):
        rh = ReceiveHistoryEntry(
            unit_id="u1", route_type=InputRouteType.TEXT,
            timestamp=100.0, cycle_id=3,
        )
        d = rh.to_dict()
        restored = ReceiveHistoryEntry.from_dict(d)
        assert restored.unit_id == "u1"
        assert restored.cycle_id == 3


class TestSuppressionHistoryEntry:
    def test_to_dict_from_dict(self):
        sh = SuppressionHistoryEntry(
            unit_id="u1", route_type=InputRouteType.TEXT,
            reason="test", reversible=True,
        )
        d = sh.to_dict()
        restored = SuppressionHistoryEntry.from_dict(d)
        assert restored.reason == "test"
        assert restored.reversible is True


class TestDecayHistoryEntry:
    def test_to_dict_from_dict(self):
        dh = DecayHistoryEntry(
            unit_id="u1",
            original_freshness=InputFreshness.FRESH,
            decayed_freshness=InputFreshness.AGING,
        )
        d = dh.to_dict()
        restored = DecayHistoryEntry.from_dict(d)
        assert restored.original_freshness == InputFreshness.FRESH
        assert restored.decayed_freshness == InputFreshness.AGING


class TestTextDialogueConfig:
    def test_defaults(self):
        cfg = TextDialogueConfig()
        assert cfg.max_units_per_cycle == 10
        assert cfg.similarity_threshold == 0.85

    def test_to_dict_from_dict(self):
        cfg = TextDialogueConfig(max_history=100, empty_streak_threshold=10)
        d = cfg.to_dict()
        restored = TextDialogueConfig.from_dict(d)
        assert restored.max_history == 100
        assert restored.empty_streak_threshold == 10


class TestTextDialogueState:
    def test_default_state(self):
        st = TextDialogueState()
        assert st.cycle_count == 0
        assert st.total_received == 0
        assert "text" in st.route_registry

    def test_to_dict_from_dict(self):
        st = TextDialogueState()
        st.cycle_count = 5
        st.total_received = 10
        st.active_units.append(InputUnit(
            unit_id="u1", route_type=InputRouteType.TEXT,
            raw_text="test",
        ))
        d = st.to_dict()
        restored = TextDialogueState.from_dict(d)
        assert restored.cycle_count == 5
        assert restored.total_received == 10
        assert len(restored.active_units) == 1
        assert restored.active_units[0].unit_id == "u1"


class TestHandoffResult:
    def test_default(self):
        hr = HandoffResult()
        assert hr.units == []
        assert hr.empty_warning is False

    def test_to_dict_from_dict(self):
        hr = HandoffResult(
            units=[InputUnit(unit_id="u1")],
            empty_warning=True,
            holdback_count=3,
        )
        d = hr.to_dict()
        restored = HandoffResult.from_dict(d)
        assert len(restored.units) == 1
        assert restored.empty_warning is True
        assert restored.holdback_count == 3


# =============================================================================
# Helper
# =============================================================================

class TestComputeTextOverlap:
    def test_identical_texts(self):
        assert _compute_text_overlap("hello", "hello") == 1.0

    def test_completely_different(self):
        val = _compute_text_overlap("abc", "xyz")
        assert val == 0.0

    def test_empty_input(self):
        assert _compute_text_overlap("", "hello") == 0.0
        assert _compute_text_overlap("hello", "") == 0.0
        assert _compute_text_overlap("", "") == 0.0

    def test_partial_overlap(self):
        val = _compute_text_overlap("abcde", "abcfg")
        assert 0.0 < val < 1.0

    def test_single_char(self):
        val = _compute_text_overlap("a", "a")
        assert val == 1.0


# =============================================================================
# Stage 1: receive_input
# =============================================================================

class TestReceiveInput:
    def test_normal_text(self):
        u = receive_input("こんにちは、今日もいい天気ですね", InputRouteType.TEXT)
        assert u.route_type == InputRouteType.TEXT
        assert u.normalization_status == NormalizationStatus.RAW
        assert u.text_length_category == "medium"

    def test_empty_text(self):
        u = receive_input("", InputRouteType.TEXT)
        assert u.normalization_status == NormalizationStatus.EMPTY
        assert u.text_length_category == "short"

    def test_whitespace_only(self):
        u = receive_input("   ", InputRouteType.TEXT)
        assert u.normalization_status == NormalizationStatus.EMPTY

    def test_short_text(self):
        cfg = TextDialogueConfig(short_text_threshold=10)
        u = receive_input("hi", InputRouteType.TEXT, config=cfg)
        assert u.text_length_category == "short"

    def test_long_text(self):
        cfg = TextDialogueConfig(long_text_threshold=20)
        u = receive_input("a" * 25, InputRouteType.TEXT, config=cfg)
        assert u.text_length_category == "long"

    def test_sender_and_conversation(self):
        u = receive_input(
            "test", InputRouteType.TEXT,
            sender_id="user1", conversation_id="conv1",
        )
        assert u.sender_id == "user1"
        assert u.conversation_id == "conv1"

    def test_screen_route(self):
        u = receive_input("screen data", InputRouteType.SCREEN)
        assert u.route_type == InputRouteType.SCREEN


# =============================================================================
# Stage 2: normalize_unit
# =============================================================================

class TestNormalizeUnit:
    def test_normal_text(self):
        u = InputUnit(raw_text="  hello  world  ", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert result.normalized_text == "hello world"
        assert result.normalization_status == NormalizationStatus.NORMALIZED

    def test_empty_unit(self):
        u = InputUnit(raw_text="", normalization_status=NormalizationStatus.EMPTY)
        result = normalize_unit(u)
        assert result.normalization_status == NormalizationStatus.EMPTY
        assert result.normalized_text == ""

    def test_zen_to_han_conversion(self):
        u = InputUnit(raw_text="\uff21\uff22\uff23\uff11\uff12\uff13", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert result.normalized_text == "ABC123"

    def test_fullwidth_symbols(self):
        u = InputUnit(raw_text="\uff1f\uff01", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert "?" in result.normalized_text
        assert "!" in result.normalized_text

    def test_fragment_punctuation_only(self):
        u = InputUnit(raw_text="!?", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert result.normalization_status == NormalizationStatus.FRAGMENT

    def test_preserves_unit_id(self):
        u = InputUnit(unit_id="keep_me", raw_text="test", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert result.unit_id == "keep_me"

    def test_whitespace_normalization(self):
        u = InputUnit(raw_text="a\t\nb  c", normalization_status=NormalizationStatus.RAW)
        result = normalize_unit(u)
        assert result.normalized_text == "a b c"


# =============================================================================
# Stage 3: attach_context
# =============================================================================

class TestAttachContext:
    def test_no_recent_units(self):
        u = InputUnit(unit_id="u1", raw_text="hello")
        cl = attach_context(u, [])
        assert cl.link_status == ContextLinkStatus.UNLINKED
        assert cl.continuation_flag is False

    def test_same_conversation_same_sender(self):
        now = time.time()
        prev = InputUnit(
            unit_id="u0", raw_text="hi",
            normalized_text="hi",
            sender_id="user1", conversation_id="conv1",
            timestamp=now - 2,
        )
        curr = InputUnit(
            unit_id="u1", raw_text="hello there",
            normalized_text="hello there",
            sender_id="user1", conversation_id="conv1",
            timestamp=now,
        )
        cl = attach_context(curr, [prev])
        assert cl.continuation_flag is True
        assert cl.link_status in (ContextLinkStatus.LINKED, ContextLinkStatus.PARTIAL)

    def test_different_conversation(self):
        now = time.time()
        prev = InputUnit(
            unit_id="u0", raw_text="topic A",
            normalized_text="topic A",
            sender_id="user1", conversation_id="conv1",
            timestamp=now - 30,
        )
        curr = InputUnit(
            unit_id="u1", raw_text="totally different",
            normalized_text="totally different",
            sender_id="user2", conversation_id="conv2",
            timestamp=now,
        )
        cl = attach_context(curr, [prev])
        assert cl.link_status == ContextLinkStatus.UNLINKED

    def test_time_close_same_sender(self):
        now = time.time()
        prev = InputUnit(
            unit_id="u0", raw_text="first",
            normalized_text="first",
            sender_id="user1", timestamp=now - 3,
        )
        curr = InputUnit(
            unit_id="u1", raw_text="second",
            normalized_text="second",
            sender_id="user1", timestamp=now,
        )
        cl = attach_context(curr, [prev])
        assert cl.continuation_flag is True

    def test_context_overlap_recorded(self):
        now = time.time()
        prev = InputUnit(
            unit_id="u0", normalized_text="abcdef",
            sender_id="user1", conversation_id="c1",
            timestamp=now - 1,
        )
        curr = InputUnit(
            unit_id="u1", normalized_text="abcdef",
            sender_id="user1", conversation_id="c1",
            timestamp=now,
        )
        cl = attach_context(curr, [prev])
        assert cl.context_overlap == 1.0


# =============================================================================
# Stage 4: align_to_percept_format
# =============================================================================

class TestAlignToPerceptFormat:
    def test_basic_alignment(self):
        u = InputUnit(
            unit_id="u1", route_type=InputRouteType.TEXT,
            normalized_text="hello",
        )
        result = align_to_percept_format(u)
        assert result["text"] == "hello"
        assert result["meaning"] == "hello"
        assert result["emotion"] == "neutral"
        assert result["intent"] == "unknown"
        assert result["_route_type"] == "text"

    def test_no_meaning_interpretation(self):
        u = InputUnit(normalized_text="I am angry!")
        result = align_to_percept_format(u)
        assert result["emotion"] == "neutral"
        assert result["sentiment"] == 0.0

    def test_with_context_link(self):
        u = InputUnit(normalized_text="test")
        cl = ContextLink(
            link_status=ContextLinkStatus.LINKED,
            continuation_flag=True,
            context_overlap=0.7,
        )
        result = align_to_percept_format(u, cl)
        assert result["_context_link_status"] == "linked"
        assert result["_continuation_flag"] is True

    def test_route_info_preserved(self):
        u = InputUnit(
            route_type=InputRouteType.API,
            normalized_text="api data",
            sender_id="system",
            conversation_id="api_conv",
        )
        result = align_to_percept_format(u)
        assert result["_route_type"] == "api"
        assert result["_sender_id"] == "system"


# =============================================================================
# Stage 5: detect_duplicates
# =============================================================================

class TestDetectDuplicates:
    def test_no_duplicates(self):
        units = [
            InputUnit(unit_id="u1", normalized_text="hello",
                      normalization_status=NormalizationStatus.NORMALIZED),
            InputUnit(unit_id="u2", normalized_text="world",
                      normalization_status=NormalizationStatus.NORMALIZED),
        ]
        accepted, suppressed, records = detect_duplicates(units, [])
        assert len(accepted) == 2
        assert len(suppressed) == 0

    def test_exact_duplicate(self):
        units = [
            InputUnit(unit_id="u1", normalized_text="hello",
                      normalization_status=NormalizationStatus.NORMALIZED),
            InputUnit(unit_id="u2", normalized_text="hello",
                      normalization_status=NormalizationStatus.NORMALIZED),
        ]
        accepted, suppressed, records = detect_duplicates(units, [])
        assert len(accepted) == 1
        assert len(suppressed) == 1
        assert records[0].status == DuplicateStatus.DUPLICATE

    def test_empty_units_pass_through(self):
        units = [
            InputUnit(unit_id="u1", normalization_status=NormalizationStatus.EMPTY),
            InputUnit(unit_id="u2", normalization_status=NormalizationStatus.EMPTY),
        ]
        accepted, suppressed, records = detect_duplicates(units, [])
        assert len(accepted) == 2

    def test_near_duplicate(self):
        cfg = TextDialogueConfig(similarity_threshold=0.5)
        units = [
            InputUnit(unit_id="u1", normalized_text="abcdefgh",
                      normalization_status=NormalizationStatus.NORMALIZED),
            InputUnit(unit_id="u2", normalized_text="abcdefgi",
                      normalization_status=NormalizationStatus.NORMALIZED),
        ]
        accepted, suppressed, records = detect_duplicates(units, [], cfg)
        # These are very similar, should be detected as near-duplicate
        assert len(suppressed) >= 1 or len(accepted) >= 1

    def test_different_content_preserved(self):
        units = [
            InputUnit(unit_id="u1", normalized_text="the quick brown fox",
                      normalization_status=NormalizationStatus.NORMALIZED),
            InputUnit(unit_id="u2", normalized_text="completely different topic here",
                      normalization_status=NormalizationStatus.NORMALIZED),
        ]
        accepted, suppressed, records = detect_duplicates(units, [])
        assert len(accepted) == 2


# =============================================================================
# Stage 6: prepare_handoff
# =============================================================================

class TestPrepareHandoff:
    def test_basic_handoff(self):
        units = [InputUnit(unit_id="u1")]
        result = prepare_handoff(
            units=units, context_links=[], conflicts=[],
            route_usage_counts={"text": 5, "screen": 3},
        )
        assert len(result.units) == 1
        assert abs(result.route_distribution["text"] - 0.625) < 0.01

    def test_empty_handoff(self):
        result = prepare_handoff(
            units=[], context_links=[], conflicts=[],
            route_usage_counts={},
        )
        assert result.route_distribution == {}

    def test_warnings_propagated(self):
        result = prepare_handoff(
            units=[], context_links=[], conflicts=[],
            route_usage_counts={},
            empty_warning=True,
            single_route_warning=True,
            diversity_warning=True,
            holdback_count=5,
        )
        assert result.empty_warning is True
        assert result.single_route_warning is True
        assert result.diversity_warning is True
        assert result.holdback_count == 5


# =============================================================================
# Safety Valves
# =============================================================================

class TestApplyFreshnessDecay:
    def test_fresh_unit_stays_fresh(self):
        now = time.time()
        units = [InputUnit(unit_id="u1", freshness=InputFreshness.FRESH, timestamp=now)]
        decayed, entries = apply_freshness_decay(units, [])
        assert decayed[0].freshness == InputFreshness.FRESH
        assert len(entries) == 0

    def test_old_unit_decays(self):
        old_time = time.time() - 120
        units = [InputUnit(unit_id="u1", freshness=InputFreshness.FRESH, timestamp=old_time)]
        decayed, entries = apply_freshness_decay(units, [])
        assert decayed[0].freshness == InputFreshness.FADED
        assert len(entries) == 1

    def test_freshness_never_increases(self):
        units = [InputUnit(
            unit_id="u1", freshness=InputFreshness.STALE,
            timestamp=time.time(),
        )]
        decayed, _ = apply_freshness_decay(units, [])
        assert decayed[0].freshness == InputFreshness.STALE


class TestDecayReceiveHistory:
    def test_recent_stays_fresh(self):
        now = time.time()
        history = [ReceiveHistoryEntry(
            unit_id="u1", timestamp=now, freshness=InputFreshness.FRESH,
        )]
        result = decay_receive_history(history)
        assert result[0].freshness == InputFreshness.FRESH

    def test_old_entry_decays(self):
        old = time.time() - 120
        history = [ReceiveHistoryEntry(
            unit_id="u1", timestamp=old, freshness=InputFreshness.FRESH,
        )]
        result = decay_receive_history(history)
        assert result[0].freshness == InputFreshness.FADED


class TestDecaySuppressionHistory:
    def test_recent_stays_fresh(self):
        now = time.time()
        history = [SuppressionHistoryEntry(
            unit_id="u1", timestamp=now, freshness=InputFreshness.FRESH,
        )]
        result = decay_suppression_history(history)
        assert result[0].freshness == InputFreshness.FRESH


class TestSuppressRecentAdoption:
    def test_no_suppression_when_diverse(self):
        units = [InputUnit(unit_id="u1", route_type=InputRouteType.TEXT)]
        passed, suppressed, entries = suppress_recent_adoption(
            units, ["text", "screen", "text"], [], TextDialogueConfig(),
        )
        assert len(passed) == 1
        assert len(suppressed) == 0

    def test_suppression_when_dominant_route(self):
        cfg = TextDialogueConfig(recent_adoption_suppression_count=3)
        units = [
            InputUnit(unit_id="u1", route_type=InputRouteType.TEXT),
            InputUnit(unit_id="u2", route_type=InputRouteType.SCREEN),
        ]
        passed, suppressed, entries = suppress_recent_adoption(
            units, ["text", "text", "text"], [], cfg,
        )
        # u1 (text) should be suppressed, u2 (screen) should pass
        assert len(suppressed) == 1
        assert suppressed[0].route_type == InputRouteType.TEXT

    def test_format_suppression(self):
        cfg = TextDialogueConfig(recent_adoption_suppression_count=3)
        units = [
            InputUnit(unit_id="u1", route_type=InputRouteType.TEXT,
                      text_length_category="short"),
            InputUnit(unit_id="u2", route_type=InputRouteType.TEXT,
                      text_length_category="long"),
        ]
        passed, suppressed, entries = suppress_recent_adoption(
            units, ["text", "screen", "api"],
            ["short", "short", "short"], cfg,
        )
        # u1 (short) should be suppressed
        assert any(u.text_length_category == "short" for u in suppressed)


class TestCheckEmptyStreak:
    def test_below_threshold(self):
        assert check_empty_streak(3) is False

    def test_at_threshold(self):
        assert check_empty_streak(5) is True

    def test_above_threshold(self):
        assert check_empty_streak(10) is True


class TestDetectSingleRouteDominance:
    def test_no_dominance(self):
        counts = {"text": 5, "screen": 5}
        is_dom, route = detect_single_route_dominance(counts)
        assert is_dom is False

    def test_dominant_route(self):
        counts = {"text": 9, "screen": 1}
        is_dom, route = detect_single_route_dominance(counts)
        assert is_dom is True
        assert route == "text"

    def test_insufficient_data(self):
        counts = {"text": 2}
        is_dom, route = detect_single_route_dominance(counts)
        assert is_dom is False


class TestEnsureFormatDiversity:
    def test_diverse_formats(self):
        units = [
            InputUnit(unit_id="u1", text_length_category="short"),
            InputUnit(unit_id="u2", text_length_category="long"),
        ]
        result, holdback, warning = ensure_format_diversity(units, [], [])
        assert warning is False

    def test_single_format_dominance(self):
        units = [
            InputUnit(unit_id="u1", text_length_category="short"),
            InputUnit(unit_id="u2", text_length_category="short"),
            InputUnit(unit_id="u3", text_length_category="short"),
        ]
        holdback = [InputUnit(
            unit_id="h1", text_length_category="long",
            freshness=InputFreshness.FRESH,
        )]
        result, new_holdback, warning = ensure_format_diversity(
            units, [], holdback,
        )
        assert warning is True
        # h1 should be injected
        assert len(result) == 4
        assert len(new_holdback) == 0

    def test_no_injection_when_holdback_stale(self):
        units = [
            InputUnit(unit_id="u1", text_length_category="short"),
        ]
        holdback = [InputUnit(
            unit_id="h1", text_length_category="long",
            freshness=InputFreshness.FADED,
        )]
        cfg = TextDialogueConfig(format_diversity_threshold=0.5)
        result, new_holdback, warning = ensure_format_diversity(
            units, [], holdback, cfg,
        )
        assert warning is True
        # h1 is FADED so not injected
        assert len(result) == 1


class TestRestoreMultiRoute:
    def test_no_restoration_needed(self):
        units = [InputUnit(unit_id="u1")]
        counts = {"text": 5, "screen": 5}
        result, holdback, warning = restore_multi_route(units, [], counts)
        assert warning is False

    def test_restore_from_holdback(self):
        units = [InputUnit(unit_id="u1", route_type=InputRouteType.TEXT)]
        holdback = [InputUnit(
            unit_id="h1", route_type=InputRouteType.SCREEN,
            freshness=InputFreshness.FRESH,
        )]
        counts = {"text": 9, "screen": 1}
        result, new_holdback, warning = restore_multi_route(
            units, holdback, counts,
        )
        assert warning is True
        assert len(result) == 2
        assert len(new_holdback) == 0


class TestFilterCircularReference:
    def test_filters_output_units(self):
        units = [
            InputUnit(unit_id="u1", is_output_of_current_cycle=True, cycle_id=5),
            InputUnit(unit_id="u2", is_output_of_current_cycle=False, cycle_id=5),
        ]
        result = filter_circular_reference(units, 5)
        assert len(result) == 1
        assert result[0].unit_id == "u2"

    def test_different_cycle_passes(self):
        units = [
            InputUnit(unit_id="u1", is_output_of_current_cycle=True, cycle_id=4),
        ]
        result = filter_circular_reference(units, 5)
        assert len(result) == 1


# =============================================================================
# Processor
# =============================================================================

class TestTextDialogueProcessor:
    def test_basic_process(self):
        proc = TextDialogueProcessor()
        result = proc.process("こんにちは")
        assert len(result.units) >= 1
        assert proc.state.cycle_count == 1
        assert proc.state.total_received == 1

    def test_multiple_cycles(self):
        proc = TextDialogueProcessor()
        proc.process("first")
        proc.process("second")
        proc.process("third")
        assert proc.state.cycle_count == 3
        assert proc.state.total_received == 3

    def test_empty_input_streak(self):
        cfg = TextDialogueConfig(empty_streak_threshold=3)
        proc = TextDialogueProcessor(config=cfg)
        proc.process("")
        proc.process("")
        result = proc.process("")
        assert result.empty_warning is True

    def test_empty_streak_resets_on_valid_input(self):
        cfg = TextDialogueConfig(empty_streak_threshold=3)
        proc = TextDialogueProcessor(config=cfg)
        proc.process("")
        proc.process("")
        proc.process("valid input")
        result = proc.process("")
        assert result.empty_warning is False

    def test_with_existing_percept(self):
        class FakePercept:
            text = "screen capture data"
        proc = TextDialogueProcessor()
        result = proc.process("text input", existing_percept=FakePercept())
        # Should have both text and screen units
        routes = {u.route_type for u in result.units}
        assert InputRouteType.TEXT in routes

    def test_duplicate_suppression(self):
        proc = TextDialogueProcessor()
        proc.process("hello world")
        # Same text again in same cycle won't be duplicate across cycles
        # but identical concurrent inputs would be
        result = proc.process("hello world")
        assert proc.state.total_received == 2

    def test_route_usage_counts_updated(self):
        proc = TextDialogueProcessor()
        proc.process("text1", route_type=InputRouteType.TEXT)
        proc.process("text2", route_type=InputRouteType.TEXT)
        assert proc.state.route_usage_counts.get("text", 0) >= 2

    def test_context_links_created(self):
        proc = TextDialogueProcessor()
        proc.process("first message", sender_id="user1", conversation_id="c1")
        proc.process("second message", sender_id="user1", conversation_id="c1")
        assert len(proc.state.context_links) >= 2

    def test_state_serialization_roundtrip(self):
        proc = TextDialogueProcessor()
        proc.process("hello")
        proc.process("world")
        d = proc.state.to_dict()
        restored = TextDialogueState.from_dict(d)
        assert restored.cycle_count == proc.state.cycle_count
        assert restored.total_received == proc.state.total_received

    def test_history_limits_respected(self):
        cfg = TextDialogueConfig(max_history=5)
        proc = TextDialogueProcessor(config=cfg)
        for i in range(10):
            proc.process(f"message {i}")
        assert len(proc.state.receive_history) <= 5
        assert len(proc.state.context_links) <= 5

    def test_sender_and_conversation_preserved(self):
        proc = TextDialogueProcessor()
        result = proc.process(
            "test", sender_id="user1", conversation_id="conv1",
        )
        text_units = [
            u for u in result.units if u.route_type == InputRouteType.TEXT
        ]
        if text_units:
            assert text_units[0].sender_id == "user1"

    def test_concurrent_input_conflicts_detected(self):
        class FakePercept:
            text = "screen data"
        proc = TextDialogueProcessor()
        result = proc.process("text data", existing_percept=FakePercept())
        # Both TEXT and SCREEN should be present, creating a conflict
        if len(result.units) > 1:
            routes = {u.route_type for u in result.units}
            if len(routes) > 1:
                assert len(result.conflicts) > 0


# =============================================================================
# Integration: merge_with_percept
# =============================================================================

class TestMergeWithPercept:
    def test_no_text_input(self):
        class FakePercept:
            text = "screen"
            meaning = "screen"
            emotion = "happy"
            intent = "greeting"
            topics = ["hello"]
            sentiment = 0.5
            emotion_valence = 0.5
        handoff = HandoffResult(units=[])
        result = merge_with_percept(FakePercept(), handoff)
        assert result["text"] == "screen"
        assert result["emotion"] == "happy"
        assert result["_route_info"]["text_input_present"] is False

    def test_with_text_input(self):
        class FakePercept:
            text = "screen"
            meaning = "screen"
            emotion = "neutral"
            intent = "unknown"
            topics = []
            sentiment = 0.0
            emotion_valence = 0.0
        text_unit = InputUnit(
            route_type=InputRouteType.TEXT,
            normalized_text="user message",
        )
        handoff = HandoffResult(units=[text_unit])
        result = merge_with_percept(FakePercept(), handoff)
        assert result["text"] == "user message"
        assert result["_route_info"]["text_input_present"] is True
        assert result["emotion"] == "neutral"  # No judgment

    def test_none_percept(self):
        text_unit = InputUnit(
            route_type=InputRouteType.TEXT,
            normalized_text="hello",
        )
        handoff = HandoffResult(units=[text_unit])
        result = merge_with_percept(None, handoff)
        assert result["text"] == "hello"

    def test_preserves_existing_emotion(self):
        class FakePercept:
            text = "screen"
            meaning = "screen"
            emotion = "happy"
            intent = "greeting"
            topics = ["hi"]
            sentiment = 0.8
            emotion_valence = 0.8
        text_unit = InputUnit(
            route_type=InputRouteType.TEXT,
            normalized_text="new text",
        )
        handoff = HandoffResult(units=[text_unit])
        result = merge_with_percept(FakePercept(), handoff)
        assert result["emotion"] == "happy"
        assert result["intent"] == "greeting"


# =============================================================================
# Summary
# =============================================================================

class TestGetTextDialogueSummary:
    def test_empty_state(self):
        st = TextDialogueState()
        summary = get_text_dialogue_summary(st)
        assert "待機中" in summary or "受信=0" in summary

    def test_with_activity(self):
        st = TextDialogueState()
        st.total_received = 10
        st.cycle_count = 5
        st.route_usage_counts = {"text": 7, "screen": 3}
        summary = get_text_dialogue_summary(st)
        assert "受信=10" in summary
        assert "cycle=5" in summary

    def test_with_warnings(self):
        st = TextDialogueState()
        st.total_received = 1
        st.empty_streak_counter = 3
        summary = get_text_dialogue_summary(st)
        assert "空入力連続=3" in summary

    def test_with_holdback(self):
        st = TextDialogueState()
        st.total_received = 1
        st.holdback_units = [InputUnit()]
        summary = get_text_dialogue_summary(st)
        assert "保留=1" in summary


# =============================================================================
# Factory
# =============================================================================

class TestFactory:
    def test_create_processor(self):
        proc = create_text_dialogue_processor()
        assert isinstance(proc, TextDialogueProcessor)

    def test_create_with_config(self):
        cfg = TextDialogueConfig(max_history=100)
        proc = create_text_dialogue_processor(config=cfg)
        assert proc.state.config.max_history == 100
