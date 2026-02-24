"""
tests/test_brain_context.py - Tests for dialogue context management.

Verifies:
1. ContextEntry creation (immutability, attributes)
2. DialogueContextManager.add_entry (FIFO, timestamps)
3. DialogueContextManager.reset_session (prev_session_last, clear)
4. DialogueContextManager.render_text (format, gap annotations, prev session)
5. DialogueContextManager.get_entries / get_window_entries (read-only)
6. Time gap annotations (3-level staged thresholds)
7. FIFO eviction when exceeding max_entries
8. Window size limiting
9. expression._format_history (backward compat: str list and ContextEntry)
10. Safety: no policy labels, pathway, or partner_id in rendered text
"""

import time
import pytest

from brain import ContextEntry, DialogueContextManager, _TIME_GAP_THRESHOLDS
from psyche.expression import _format_history, _build_render_prompt
from psyche.state import PsycheState


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def ctx():
    """Default context manager with small limits for testing."""
    return DialogueContextManager(max_entries=10, window_size=5)


@pytest.fixture
def base_ts():
    """A fixed base timestamp for deterministic tests."""
    return 1000.0


# ── 1. ContextEntry ────────────────────────────────────────────

class TestContextEntry:

    def test_creation(self):
        e = ContextEntry(
            speaker_label="キュレネ",
            text="こんにちは",
            pathway="text",
            partner_id="user1",
            timestamp=100.0,
        )
        assert e.speaker_label == "キュレネ"
        assert e.text == "こんにちは"
        assert e.pathway == "text"
        assert e.partner_id == "user1"
        assert e.timestamp == 100.0

    def test_immutability(self):
        e = ContextEntry("キュレネ", "hi", "text", "u1", 1.0)
        with pytest.raises(AttributeError):
            e.text = "changed"

    def test_equality(self):
        e1 = ContextEntry("A", "hello", "text", "u1", 1.0)
        e2 = ContextEntry("A", "hello", "text", "u1", 1.0)
        assert e1 == e2

    def test_different_entries_not_equal(self):
        e1 = ContextEntry("A", "hello", "text", "u1", 1.0)
        e2 = ContextEntry("B", "hello", "text", "u1", 1.0)
        assert e1 != e2

    def test_has_all_five_attributes(self):
        e = ContextEntry("s", "t", "p", "pid", 0.0)
        assert hasattr(e, "speaker_label")
        assert hasattr(e, "text")
        assert hasattr(e, "pathway")
        assert hasattr(e, "partner_id")
        assert hasattr(e, "timestamp")


# ── 2. add_entry ───────────────────────────────────────────────

class TestAddEntry:

    def test_basic_add(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "hello", "text", "u1", base_ts)
        assert len(ctx) == 1
        entries = ctx.get_entries()
        assert entries[0].speaker_label == "キュレネ"
        assert entries[0].text == "hello"

    def test_auto_timestamp(self, ctx):
        before = time.monotonic()
        ctx.add_entry("キュレネ", "test", "text", "u1")
        after = time.monotonic()
        e = ctx.get_entries()[0]
        assert before <= e.timestamp <= after

    def test_multiple_adds(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "hi", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "hello", "text", "u1", base_ts + 1)
        assert len(ctx) == 2

    def test_fifo_eviction(self, base_ts):
        ctx = DialogueContextManager(max_entries=3, window_size=3)
        for i in range(5):
            ctx.add_entry("s", f"msg{i}", "text", "u1", base_ts + i)
        assert len(ctx) == 3
        entries = ctx.get_entries()
        assert entries[0].text == "msg2"
        assert entries[1].text == "msg3"
        assert entries[2].text == "msg4"

    def test_pathway_stored(self, ctx, base_ts):
        ctx.add_entry("画面情報", "screen", "vision", "viewer", base_ts)
        assert ctx.get_entries()[0].pathway == "vision"

    def test_partner_id_stored(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "hi", "text", "alice", base_ts)
        assert ctx.get_entries()[0].partner_id == "alice"


# ── 3. reset_session ──────────────────────────────────────────

class TestResetSession:

    def test_clears_entries(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "a", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "b", "text", "u1", base_ts + 1)
        ctx.reset_session()
        assert len(ctx) == 0

    def test_saves_last_entry(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "first", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "last", "text", "u1", base_ts + 1)
        ctx.reset_session()
        prev = ctx.prev_session_last
        assert prev is not None
        assert prev.text == "last"
        assert prev.speaker_label == "キュレネ"

    def test_empty_reset_no_crash(self, ctx):
        ctx.reset_session()
        assert ctx.prev_session_last is None

    def test_double_reset_updates_prev(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "session1_last", "text", "u1", base_ts)
        ctx.reset_session()
        assert ctx.prev_session_last.text == "session1_last"
        ctx.add_entry("キュレネ", "session2_last", "text", "u1", base_ts + 100)
        ctx.reset_session()
        assert ctx.prev_session_last.text == "session2_last"

    def test_entries_after_reset(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "before", "text", "u1", base_ts)
        ctx.reset_session()
        ctx.add_entry("キュレネ", "after", "text", "u1", base_ts + 100)
        assert len(ctx) == 1
        assert ctx.get_entries()[0].text == "after"


# ── 4. render_text ─────────────────────────────────────────────

class TestRenderText:

    def test_empty_entries(self, ctx):
        assert ctx.render_text() == ""

    def test_single_entry(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "hello", "text", "u1", base_ts)
        text = ctx.render_text()
        assert "[キュレネ] hello" in text

    def test_multiple_entries_format(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "こんにちは", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "やっほー", "text", "u1", base_ts + 1)
        text = ctx.render_text()
        assert "[ユーザー] こんにちは" in text
        assert "[キュレネ] やっほー" in text

    def test_no_pathway_in_output(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "test", "vision", "viewer", base_ts)
        text = ctx.render_text()
        assert "vision" not in text
        assert "viewer" not in text.replace("[", "").replace("]", "").lower()

    def test_no_partner_id_in_output(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "hi", "text", "alice123", base_ts)
        text = ctx.render_text()
        assert "alice123" not in text

    def test_prev_session_entry_in_render(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "prev_last", "text", "u1", base_ts)
        ctx.reset_session()
        ctx.add_entry("ユーザー", "new_hi", "text", "u1", base_ts + 500)
        text = ctx.render_text()
        assert "前回最後の発話" in text
        assert "prev_last" in text
        assert "new_hi" in text

    def test_window_size_limits_output(self, base_ts):
        ctx = DialogueContextManager(max_entries=20, window_size=3)
        for i in range(10):
            ctx.add_entry("s", f"msg{i}", "text", "u1", base_ts + i)
        text = ctx.render_text()
        assert "msg7" in text
        assert "msg8" in text
        assert "msg9" in text
        assert "msg6" not in text


# ── 5. get_entries / get_window_entries ────────────────────────

class TestDataAccess:

    def test_get_entries_returns_copy(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "a", "text", "u1", base_ts)
        entries = ctx.get_entries()
        entries.clear()
        assert len(ctx) == 1

    def test_get_window_entries_limited(self, base_ts):
        ctx = DialogueContextManager(max_entries=20, window_size=3)
        for i in range(10):
            ctx.add_entry("s", f"m{i}", "text", "u1", base_ts + i)
        window = ctx.get_window_entries()
        assert len(window) == 3
        assert window[0].text == "m7"
        assert window[2].text == "m9"

    def test_get_window_entries_when_fewer_than_window(self, ctx, base_ts):
        ctx.add_entry("s", "only", "text", "u1", base_ts)
        window = ctx.get_window_entries()
        assert len(window) == 1


# ── 6. Time gap annotations ───────────────────────────────────

class TestTimeGapAnnotations:

    def test_no_gap_annotation_below_threshold(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "a", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "b", "text", "u1", base_ts + 5)
        text = ctx.render_text()
        assert "間があった" not in text
        assert "時間が経った" not in text

    def test_short_gap_annotation(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "a", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "b", "text", "u1", base_ts + 20)
        text = ctx.render_text()
        assert "（少し間があった）" in text

    def test_medium_gap_annotation(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "a", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "b", "text", "u1", base_ts + 90)
        text = ctx.render_text()
        assert "（しばらく間があった）" in text

    def test_long_gap_annotation(self, ctx, base_ts):
        ctx.add_entry("ユーザー", "a", "text", "u1", base_ts)
        ctx.add_entry("キュレネ", "b", "text", "u1", base_ts + 400)
        text = ctx.render_text()
        assert "（かなり時間が経った）" in text

    def test_gap_at_exact_threshold_15(self, ctx, base_ts):
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 15.0)
        text = ctx.render_text()
        assert "（少し間があった）" in text

    def test_gap_at_exact_threshold_60(self, ctx, base_ts):
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 60.0)
        text = ctx.render_text()
        assert "（しばらく間があった）" in text

    def test_gap_at_exact_threshold_300(self, ctx, base_ts):
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 300.0)
        text = ctx.render_text()
        assert "（かなり時間が経った）" in text

    def test_gap_just_below_15(self, ctx, base_ts):
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 14.9)
        text = ctx.render_text()
        assert "間があった" not in text

    def test_gap_annotation_between_prev_session_and_current(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "bye", "text", "u1", base_ts)
        ctx.reset_session()
        ctx.add_entry("ユーザー", "hi again", "text", "u1", base_ts + 600)
        text = ctx.render_text()
        assert "（かなり時間が経った）" in text

    def test_multiple_gap_annotations(self, ctx, base_ts):
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 20)
        ctx.add_entry("c", "z", "text", "u1", base_ts + 120)
        text = ctx.render_text()
        assert "（少し間があった）" in text
        assert "（しばらく間があった）" in text

    def test_custom_gap_thresholds(self, base_ts):
        custom = [(10.0, "(custom gap)")]
        ctx = DialogueContextManager(
            max_entries=10, window_size=5,
            time_gap_thresholds=custom,
        )
        ctx.add_entry("a", "x", "text", "u1", base_ts)
        ctx.add_entry("b", "y", "text", "u1", base_ts + 12)
        text = ctx.render_text()
        assert "(custom gap)" in text


# ── 7. FIFO boundary behavior ─────────────────────────────────

class TestFIFOBoundary:

    def test_max_entries_one(self, base_ts):
        ctx = DialogueContextManager(max_entries=1, window_size=1)
        ctx.add_entry("a", "first", "text", "u1", base_ts)
        ctx.add_entry("b", "second", "text", "u1", base_ts + 1)
        assert len(ctx) == 1
        assert ctx.get_entries()[0].text == "second"

    def test_max_entries_exact_fill(self, base_ts):
        ctx = DialogueContextManager(max_entries=3, window_size=3)
        for i in range(3):
            ctx.add_entry("s", f"m{i}", "text", "u1", base_ts + i)
        assert len(ctx) == 3

    def test_negative_max_entries_clamped_to_one(self):
        ctx = DialogueContextManager(max_entries=-5, window_size=1)
        ctx.add_entry("s", "test", "text", "u1", 1.0)
        assert len(ctx) == 1


# ── 8. Window size behavior ───────────────────────────────────

class TestWindowSize:

    def test_window_larger_than_entries(self, base_ts):
        ctx = DialogueContextManager(max_entries=10, window_size=20)
        ctx.add_entry("s", "only", "text", "u1", base_ts)
        window = ctx.get_window_entries()
        assert len(window) == 1

    def test_window_one(self, base_ts):
        ctx = DialogueContextManager(max_entries=10, window_size=1)
        for i in range(5):
            ctx.add_entry("s", f"m{i}", "text", "u1", base_ts + i)
        window = ctx.get_window_entries()
        assert len(window) == 1
        assert window[0].text == "m4"


# ── 9. expression._format_history ─────────────────────────────

class TestFormatHistory:

    def test_empty_list(self):
        assert _format_history([]) == ""

    def test_none_input(self):
        assert _format_history(None) == ""

    def test_legacy_string_list(self):
        history = ["[ユーザー] hi", "[キュレネ] hello"]
        result = _format_history(history)
        assert "[ユーザー] hi" in result
        assert "[キュレネ] hello" in result

    def test_legacy_string_list_truncated_to_five(self):
        history = [f"line{i}" for i in range(10)]
        result = _format_history(history)
        assert "line5" in result
        assert "line9" in result
        assert "line4" not in result

    def test_structured_entries(self):
        entries = [
            ContextEntry("ユーザー", "hi", "text", "u1", 100.0),
            ContextEntry("キュレネ", "hello", "text", "u1", 101.0),
        ]
        result = _format_history(entries)
        assert "[ユーザー] hi" in result
        assert "[キュレネ] hello" in result

    def test_structured_entries_no_pathway(self):
        entries = [
            ContextEntry("キュレネ", "test", "vision", "viewer", 100.0),
        ]
        result = _format_history(entries)
        assert "vision" not in result
        assert "viewer" not in result

    def test_structured_entries_no_partner_id(self):
        entries = [
            ContextEntry("ユーザー", "hi", "text", "alice", 100.0),
        ]
        result = _format_history(entries)
        assert "alice" not in result

    def test_structured_entries_gap_annotation(self):
        entries = [
            ContextEntry("ユーザー", "a", "text", "u1", 100.0),
            ContextEntry("キュレネ", "b", "text", "u1", 200.0),
        ]
        result = _format_history(entries)
        assert "（しばらく間があった）" in result

    def test_structured_entries_truncated_to_five(self):
        entries = [
            ContextEntry("s", f"m{i}", "text", "u1", 100.0 + i)
            for i in range(10)
        ]
        result = _format_history(entries)
        assert "m5" in result
        assert "m9" in result
        assert "m4" not in result

    def test_structured_entries_long_gap(self):
        entries = [
            ContextEntry("a", "x", "text", "u1", 100.0),
            ContextEntry("b", "y", "text", "u1", 500.0),
        ]
        result = _format_history(entries)
        assert "（かなり時間が経った）" in result


# ── 10. Safety: no policy labels leak ──────────────────────────

class TestSafetyPolicyLabelsExcluded:

    def test_render_text_no_policy_label(self, ctx, base_ts):
        """Policy labels must never appear in rendered text."""
        ctx.add_entry("キュレネ", "response text", "text", "u1", base_ts)
        text = ctx.render_text()
        # policy_label is never part of ContextEntry, so cannot leak
        assert "policy" not in text.lower()

    def test_context_entry_has_no_policy_field(self):
        e = ContextEntry("s", "t", "p", "pid", 0.0)
        assert not hasattr(e, "policy_label")
        assert not hasattr(e, "score")
        assert not hasattr(e, "enrichment")

    def test_format_history_structured_no_internal_metadata(self):
        entries = [
            ContextEntry("キュレネ", "hello", "internal", "internal", 100.0),
        ]
        result = _format_history(entries)
        assert "internal" not in result.lower()


# ── 11. _build_render_prompt with structured entries ───────────

class TestBuildRenderPromptWithStructured:

    def test_structured_entries_in_prompt(self):
        state = PsycheState()
        policy = {"policy_label": "共感する", "rationale": "test"}
        persona = {"name": "キュレネ", "tone": "sweet"}
        entries = [
            ContextEntry("ユーザー", "こんにちは", "text", "u1", 100.0),
            ContextEntry("キュレネ", "やっほー", "text", "u1", 101.0),
        ]
        prompt = _build_render_prompt(
            state, policy, [], persona,
            recent_history=entries,
        )
        assert "こんにちは" in prompt
        assert "やっほー" in prompt
        # Internal metadata must not appear
        assert "u1" not in prompt.split("直近の会話")[1].split("関連記憶")[0]

    def test_legacy_string_list_in_prompt(self):
        state = PsycheState()
        policy = {"policy_label": "共感する", "rationale": "test"}
        persona = {"name": "キュレネ", "tone": "sweet"}
        history = ["[ユーザー] hi", "[キュレネ] hello"]
        prompt = _build_render_prompt(
            state, policy, [], persona,
            recent_history=history,
        )
        assert "[ユーザー] hi" in prompt
        assert "[キュレネ] hello" in prompt

    def test_none_history_shows_placeholder(self):
        state = PsycheState()
        policy = {"policy_label": "共感する", "rationale": "test"}
        persona = {"name": "キュレネ", "tone": "sweet"}
        prompt = _build_render_prompt(
            state, policy, [], persona,
            recent_history=None,
        )
        assert "(なし)" in prompt


# ── 12. Default thresholds consistency ─────────────────────────

class TestDefaultThresholds:

    def test_thresholds_sorted_descending(self):
        thresholds = [t for t, _ in _TIME_GAP_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_three_levels(self):
        assert len(_TIME_GAP_THRESHOLDS) == 3

    def test_all_annotations_are_japanese(self):
        for _, ann in _TIME_GAP_THRESHOLDS:
            assert "（" in ann
            assert "）" in ann


# ── 13. Edge cases ─────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_text_entry(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "", "text", "u1", base_ts)
        text = ctx.render_text()
        assert "[キュレネ] " in text

    def test_very_long_text_entry(self, ctx, base_ts):
        long_text = "あ" * 5000
        ctx.add_entry("キュレネ", long_text, "text", "u1", base_ts)
        entries = ctx.get_entries()
        assert len(entries[0].text) == 5000

    def test_special_characters_in_text(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "♪♡★\n改行あり", "text", "u1", base_ts)
        text = ctx.render_text()
        assert "♪♡★" in text

    def test_render_text_with_only_prev_session(self, ctx, base_ts):
        ctx.add_entry("キュレネ", "old", "text", "u1", base_ts)
        ctx.reset_session()
        # No current entries, only prev session
        text = ctx.render_text()
        assert text == ""  # No current window entries to render

    def test_len_method(self, ctx, base_ts):
        assert len(ctx) == 0
        ctx.add_entry("s", "t", "text", "u1", base_ts)
        assert len(ctx) == 1

    def test_all_pathways(self, ctx, base_ts):
        for pathway in ["vision", "text", "internal"]:
            ctx.add_entry("s", f"test_{pathway}", pathway, "u1", base_ts)
        entries = ctx.get_entries()
        assert entries[0].pathway == "vision"
        assert entries[1].pathway == "text"
        assert entries[2].pathway == "internal"
