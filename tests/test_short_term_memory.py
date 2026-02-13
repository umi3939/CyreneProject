"""
tests/test_short_term_memory.py - Tests for Short-Term Emotional Memory System

Verifies:
1.  StimulusEntry defaults
2.  ShortTermMemory defaults (empty entries, max_entries=10)
3.  add_stimulus creates new entry, returns new instance
4.  add_stimulus trims to max_entries
5.  get_unprocessed_residue filters processed entries
6.  get_weighted_residue_summary aggregation
7.  mark_processed all (None)
8.  mark_processed specific indices
9.  compute_context_overlap Jaccard: full, partial, none, empty
10. update_context continuous (union) vs non-continuous (replace)
11. reset_on_discontinuity clears everything
12. apply_decay with default function (exponential)
13. apply_decay removes entries below min_weight
14. apply_decay with custom function
15. to_dict / from_dict roundtrip
16. compute_residue_influence basic
17. compute_residue_influence with scale_factor override
18. Immutability: all operations return new instances
"""

import pytest
import time

from psyche.short_term_memory import (
    StimulusEntry,
    ShortTermMemory,
    ResidueInfluence,
    compute_residue_influence,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_entry(
    emotion_label: str = "neutral",
    raw_intensity: float = 0.5,
    valence: float = 0.0,
    residue_weight: float = 1.0,
    processed: bool = False,
    topics: list[str] | None = None,
    timestamp: float | None = None,
) -> StimulusEntry:
    """Convenience factory for StimulusEntry with explicit control."""
    return StimulusEntry(
        source_text="test",
        topics=topics if topics is not None else ["topic_a"],
        emotion_label=emotion_label,
        intent="inform",
        raw_intensity=raw_intensity,
        valence=valence,
        residue_weight=residue_weight,
        processed=processed,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


def _make_memory(
    entries: list[StimulusEntry] | None = None,
    max_entries: int = 10,
    context_topics: list[str] | None = None,
    continuity_score: float = 0.0,
    scale_factors: dict[str, float] | None = None,
    last_update_time: float | None = None,
) -> ShortTermMemory:
    """Convenience factory for ShortTermMemory."""
    sf = scale_factors if scale_factors is not None else {
        "residue_influence": 1.0,
        "accumulation_rate": 1.0,
        "continuity_threshold": 0.0,
        "decay_base": 1.0,
    }
    return ShortTermMemory(
        entries=entries if entries is not None else [],
        max_entries=max_entries,
        last_update_time=last_update_time if last_update_time is not None else time.time(),
        current_context_topics=context_topics if context_topics is not None else [],
        context_continuity_score=continuity_score,
        scale_factors=sf,
    )


# ── 1. StimulusEntry defaults ─────────────────────────────────


class TestStimulusEntryDefaults:
    """Verify default field values on StimulusEntry."""

    def test_default_source_text(self):
        entry = StimulusEntry()
        assert entry.source_text == ""

    def test_default_topics(self):
        entry = StimulusEntry()
        assert entry.topics == []

    def test_default_emotion_label(self):
        entry = StimulusEntry()
        assert entry.emotion_label == "neutral"

    def test_default_intent(self):
        entry = StimulusEntry()
        assert entry.intent == "unknown"

    def test_default_raw_intensity(self):
        entry = StimulusEntry()
        assert entry.raw_intensity == 0.0

    def test_default_valence(self):
        entry = StimulusEntry()
        assert entry.valence == 0.0

    def test_default_residue_weight(self):
        entry = StimulusEntry()
        assert entry.residue_weight == 1.0

    def test_default_processed(self):
        entry = StimulusEntry()
        assert entry.processed is False

    def test_timestamp_is_set(self):
        before = time.time()
        entry = StimulusEntry()
        after = time.time()
        assert before <= entry.timestamp <= after


# ── 2. ShortTermMemory defaults ───────────────────────────────


class TestShortTermMemoryDefaults:
    """Verify default field values on ShortTermMemory."""

    def test_empty_entries(self):
        mem = ShortTermMemory()
        assert mem.entries == []

    def test_max_entries_default(self):
        mem = ShortTermMemory()
        assert mem.max_entries == 10

    def test_context_topics_default(self):
        mem = ShortTermMemory()
        assert mem.current_context_topics == []

    def test_continuity_score_default(self):
        mem = ShortTermMemory()
        assert mem.context_continuity_score == 0.0

    def test_scale_factors_have_required_keys(self):
        mem = ShortTermMemory()
        assert "residue_influence" in mem.scale_factors
        assert "accumulation_rate" in mem.scale_factors
        assert "continuity_threshold" in mem.scale_factors
        assert "decay_base" in mem.scale_factors


# ── 3. add_stimulus creates entry & returns new instance ──────


class TestAddStimulus:
    """Verify add_stimulus creates entries correctly."""

    def test_returns_new_instance(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        assert new_mem is not mem

    def test_adds_one_entry(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        assert len(new_mem.entries) == 1

    def test_entry_values_correct(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        entry = new_mem.entries[0]
        assert entry.source_text == "hello"
        assert entry.topics == ["greet"]
        assert entry.emotion_label == "happy"
        assert entry.intent == "greet"
        assert entry.raw_intensity == 0.8
        assert entry.valence == 0.5

    def test_new_entry_unprocessed(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        assert new_mem.entries[0].processed is False

    def test_new_entry_full_residue_weight(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        assert new_mem.entries[0].residue_weight == 1.0

    def test_original_unchanged(self):
        mem = _make_memory()
        mem.add_stimulus("hello", ["greet"], "happy", "greet", 0.8, 0.5)
        assert len(mem.entries) == 0

    def test_preserves_existing_entries(self):
        existing = _make_entry(emotion_label="sad", raw_intensity=0.3)
        mem = _make_memory(entries=[existing])
        new_mem = mem.add_stimulus("new", ["topic"], "happy", "greet", 0.7, 0.4)
        assert len(new_mem.entries) == 2
        assert new_mem.entries[0].emotion_label == "sad"
        assert new_mem.entries[1].emotion_label == "happy"

    def test_preserves_scale_factors(self):
        mem = _make_memory(scale_factors={"residue_influence": 2.0, "decay_base": 0.9,
                                          "accumulation_rate": 1.0, "continuity_threshold": 0.0})
        new_mem = mem.add_stimulus("t", ["x"], "joy", "share", 0.5, 0.5)
        assert new_mem.scale_factors["residue_influence"] == 2.0
        assert new_mem.scale_factors["decay_base"] == 0.9

    def test_preserves_context_topics(self):
        mem = _make_memory(context_topics=["alpha", "beta"])
        new_mem = mem.add_stimulus("t", ["x"], "joy", "share", 0.5, 0.5)
        assert new_mem.current_context_topics == ["alpha", "beta"]


# ── 4. add_stimulus trims to max_entries ──────────────────────


class TestAddStimulusTrimming:
    """Verify oldest entries are trimmed when max_entries exceeded."""

    def test_trims_oldest_when_full(self):
        entries = [_make_entry(emotion_label=f"e{i}") for i in range(10)]
        mem = _make_memory(entries=entries, max_entries=10)
        new_mem = mem.add_stimulus("new", ["x"], "e_new", "act", 0.5, 0.5)
        assert len(new_mem.entries) == 10
        # oldest entry (e0) should be gone
        labels = [e.emotion_label for e in new_mem.entries]
        assert "e0" not in labels
        assert "e_new" in labels

    def test_trims_with_small_max(self):
        mem = _make_memory(max_entries=2)
        mem = mem.add_stimulus("a", ["x"], "e1", "act", 0.5, 0.0)
        mem = mem.add_stimulus("b", ["x"], "e2", "act", 0.5, 0.0)
        mem = mem.add_stimulus("c", ["x"], "e3", "act", 0.5, 0.0)
        assert len(mem.entries) == 2
        labels = [e.emotion_label for e in mem.entries]
        assert labels == ["e2", "e3"]


# ── 5. get_unprocessed_residue ────────────────────────────────


class TestGetUnprocessedResidue:
    """Verify filtering of processed entries."""

    def test_all_unprocessed(self):
        entries = [_make_entry(processed=False) for _ in range(3)]
        mem = _make_memory(entries=entries)
        result = mem.get_unprocessed_residue()
        assert len(result) == 3

    def test_all_processed(self):
        entries = [_make_entry(processed=True) for _ in range(3)]
        mem = _make_memory(entries=entries)
        result = mem.get_unprocessed_residue()
        assert len(result) == 0

    def test_mixed(self):
        entries = [
            _make_entry(processed=False, emotion_label="a"),
            _make_entry(processed=True, emotion_label="b"),
            _make_entry(processed=False, emotion_label="c"),
        ]
        mem = _make_memory(entries=entries)
        result = mem.get_unprocessed_residue()
        assert len(result) == 2
        labels = [e.emotion_label for e in result]
        assert "a" in labels
        assert "c" in labels

    def test_empty_memory(self):
        mem = _make_memory()
        result = mem.get_unprocessed_residue()
        assert result == []


# ── 6. get_weighted_residue_summary ───────────────────────────


class TestGetWeightedResidueSummary:
    """Verify weighted aggregation by emotion label."""

    def test_single_entry(self):
        entry = _make_entry(emotion_label="joy", raw_intensity=0.8, residue_weight=1.0)
        mem = _make_memory(entries=[entry])
        summary = mem.get_weighted_residue_summary()
        assert summary == {"joy": pytest.approx(0.8)}

    def test_multiple_same_label(self):
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.5, residue_weight=1.0),
            _make_entry(emotion_label="joy", raw_intensity=0.3, residue_weight=0.5),
        ]
        mem = _make_memory(entries=entries)
        summary = mem.get_weighted_residue_summary()
        # 1.0*0.5 + 0.5*0.3 = 0.65
        assert summary == {"joy": pytest.approx(0.65)}

    def test_different_labels(self):
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.6, residue_weight=1.0),
            _make_entry(emotion_label="sad", raw_intensity=0.4, residue_weight=0.8),
        ]
        mem = _make_memory(entries=entries)
        summary = mem.get_weighted_residue_summary()
        assert summary["joy"] == pytest.approx(0.6)
        assert summary["sad"] == pytest.approx(0.32)

    def test_skips_processed(self):
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.5, residue_weight=1.0, processed=False),
            _make_entry(emotion_label="joy", raw_intensity=0.3, residue_weight=1.0, processed=True),
        ]
        mem = _make_memory(entries=entries)
        summary = mem.get_weighted_residue_summary()
        assert summary == {"joy": pytest.approx(0.5)}

    def test_empty_returns_empty(self):
        mem = _make_memory()
        summary = mem.get_weighted_residue_summary()
        assert summary == {}


# ── 7. mark_processed all (None) ─────────────────────────────


class TestMarkProcessedAll:
    """Verify marking all entries as processed."""

    def test_marks_all(self):
        entries = [
            _make_entry(processed=False),
            _make_entry(processed=False),
            _make_entry(processed=False),
        ]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed()
        assert all(e.processed for e in new_mem.entries)

    def test_returns_new_instance(self):
        entries = [_make_entry(processed=False)]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed()
        assert new_mem is not mem

    def test_original_unchanged(self):
        entries = [_make_entry(processed=False)]
        mem = _make_memory(entries=entries)
        mem.mark_processed()
        assert mem.entries[0].processed is False

    def test_already_processed_stays_processed(self):
        entries = [_make_entry(processed=True)]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed()
        assert new_mem.entries[0].processed is True


# ── 8. mark_processed specific indices ───────────────────────


class TestMarkProcessedSpecific:
    """Verify marking specific entries as processed."""

    def test_mark_single_index(self):
        entries = [
            _make_entry(processed=False, emotion_label="a"),
            _make_entry(processed=False, emotion_label="b"),
            _make_entry(processed=False, emotion_label="c"),
        ]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed(entry_indices=[1])
        assert new_mem.entries[0].processed is False
        assert new_mem.entries[1].processed is True
        assert new_mem.entries[2].processed is False

    def test_mark_multiple_indices(self):
        entries = [
            _make_entry(processed=False),
            _make_entry(processed=False),
            _make_entry(processed=False),
        ]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed(entry_indices=[0, 2])
        assert new_mem.entries[0].processed is True
        assert new_mem.entries[1].processed is False
        assert new_mem.entries[2].processed is True

    def test_empty_index_list_marks_none(self):
        entries = [_make_entry(processed=False)]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed(entry_indices=[])
        assert new_mem.entries[0].processed is False

    def test_preserves_entry_data(self):
        entry = _make_entry(
            emotion_label="joy",
            raw_intensity=0.7,
            valence=0.3,
            residue_weight=0.9,
        )
        mem = _make_memory(entries=[entry])
        new_mem = mem.mark_processed(entry_indices=[0])
        e = new_mem.entries[0]
        assert e.emotion_label == "joy"
        assert e.raw_intensity == 0.7
        assert e.valence == 0.3
        assert e.residue_weight == 0.9
        assert e.processed is True


# ── 9. compute_context_overlap (Jaccard) ─────────────────────


class TestComputeContextOverlap:
    """Verify Jaccard similarity computation."""

    def test_full_overlap(self):
        mem = _make_memory(context_topics=["a", "b", "c"])
        score = mem.compute_context_overlap(["a", "b", "c"])
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self):
        mem = _make_memory(context_topics=["a", "b", "c"])
        # intersection={a,b}, union={a,b,c,d} => 2/4 = 0.5
        score = mem.compute_context_overlap(["a", "b", "d"])
        assert score == pytest.approx(0.5)

    def test_no_overlap(self):
        mem = _make_memory(context_topics=["a", "b"])
        score = mem.compute_context_overlap(["c", "d"])
        assert score == pytest.approx(0.0)

    def test_empty_current_context(self):
        mem = _make_memory(context_topics=[])
        score = mem.compute_context_overlap(["a", "b"])
        assert score == pytest.approx(0.0)

    def test_empty_new_topics(self):
        mem = _make_memory(context_topics=["a", "b"])
        score = mem.compute_context_overlap([])
        assert score == pytest.approx(0.0)

    def test_both_empty(self):
        mem = _make_memory(context_topics=[])
        score = mem.compute_context_overlap([])
        assert score == pytest.approx(0.0)

    def test_single_element_match(self):
        mem = _make_memory(context_topics=["x"])
        score = mem.compute_context_overlap(["x"])
        assert score == pytest.approx(1.0)

    def test_superset(self):
        mem = _make_memory(context_topics=["a", "b"])
        # intersection={a,b}, union={a,b,c} => 2/3
        score = mem.compute_context_overlap(["a", "b", "c"])
        assert score == pytest.approx(2.0 / 3.0)

    def test_subset(self):
        mem = _make_memory(context_topics=["a", "b", "c"])
        # intersection={a}, union={a,b,c} => 1/3
        score = mem.compute_context_overlap(["a"])
        assert score == pytest.approx(1.0 / 3.0)


# ── 10. update_context continuous vs non-continuous ───────────


class TestUpdateContext:
    """Verify context accumulation (union) vs replacement."""

    def test_continuous_merges_topics(self):
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.update_context(["b", "c"], is_continuous=True)
        topics = set(new_mem.current_context_topics)
        assert topics == {"a", "b", "c"}

    def test_non_continuous_replaces_topics(self):
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.update_context(["c", "d"], is_continuous=False)
        assert set(new_mem.current_context_topics) == {"c", "d"}

    def test_returns_new_instance(self):
        mem = _make_memory(context_topics=["a"])
        new_mem = mem.update_context(["b"], is_continuous=True)
        assert new_mem is not mem

    def test_original_context_unchanged(self):
        mem = _make_memory(context_topics=["a", "b"])
        mem.update_context(["c"], is_continuous=True)
        assert set(mem.current_context_topics) == {"a", "b"}

    def test_continuity_score_updated(self):
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.update_context(["a", "b"], is_continuous=True)
        # overlap is computed against original context: full overlap = 1.0
        assert new_mem.context_continuity_score == pytest.approx(1.0)

    def test_non_continuous_with_no_overlap_score_zero(self):
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.update_context(["c", "d"], is_continuous=False)
        # overlap computed against original context: no overlap = 0.0
        assert new_mem.context_continuity_score == pytest.approx(0.0)

    def test_continuous_from_empty(self):
        mem = _make_memory(context_topics=[])
        new_mem = mem.update_context(["x", "y"], is_continuous=True)
        assert set(new_mem.current_context_topics) == {"x", "y"}

    def test_non_continuous_replaces_with_copy(self):
        """Ensure the new topics list is a copy, not a reference."""
        original_topics = ["c", "d"]
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.update_context(original_topics, is_continuous=False)
        original_topics.append("e")
        assert "e" not in new_mem.current_context_topics


# ── 11. reset_on_discontinuity ───────────────────────────────


class TestResetOnDiscontinuity:
    """Verify full reset clears entries and context."""

    def test_clears_entries(self):
        entries = [_make_entry(), _make_entry()]
        mem = _make_memory(entries=entries)
        new_mem = mem.reset_on_discontinuity()
        assert new_mem.entries == []

    def test_clears_context_topics(self):
        mem = _make_memory(context_topics=["a", "b"])
        new_mem = mem.reset_on_discontinuity()
        assert new_mem.current_context_topics == []

    def test_resets_continuity_score(self):
        mem = _make_memory(continuity_score=0.8)
        new_mem = mem.reset_on_discontinuity()
        assert new_mem.context_continuity_score == 0.0

    def test_preserves_max_entries(self):
        mem = _make_memory(max_entries=5)
        new_mem = mem.reset_on_discontinuity()
        assert new_mem.max_entries == 5

    def test_preserves_scale_factors(self):
        sf = {"residue_influence": 2.0, "decay_base": 0.9,
              "accumulation_rate": 1.0, "continuity_threshold": 0.0}
        mem = _make_memory(scale_factors=sf)
        new_mem = mem.reset_on_discontinuity()
        assert new_mem.scale_factors["residue_influence"] == 2.0

    def test_returns_new_instance(self):
        mem = _make_memory()
        new_mem = mem.reset_on_discontinuity()
        assert new_mem is not mem


# ── 12. apply_decay with default function ────────────────────


class TestApplyDecayDefault:
    """Verify default exponential decay behavior."""

    def test_returns_new_instance(self):
        now = time.time()
        entries = [_make_entry(timestamp=now)]
        mem = _make_memory(entries=entries, last_update_time=now)
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert new_mem is not mem

    def test_decay_base_1_no_change(self):
        """decay_base=1.0 means weight * 1.0^dt = weight (no decay)."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=0.8)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 1.0, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        new_mem = mem.apply_decay(current_time=now + 10.0)
        assert new_mem.entries[0].residue_weight == pytest.approx(0.8)

    def test_decay_reduces_weight(self):
        """With decay_base < 1.0, weight should decrease over time."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 0.5, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        # After 1 second: weight = 1.0 * 0.5^1 = 0.5
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert new_mem.entries[0].residue_weight == pytest.approx(0.5)

    def test_decay_exponential_math(self):
        """Verify exact exponential: weight * decay_base^elapsed."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 0.9, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        dt = 3.0
        new_mem = mem.apply_decay(current_time=now + dt)
        expected = 1.0 * (0.9 ** dt)
        assert new_mem.entries[0].residue_weight == pytest.approx(expected)

    def test_updates_last_update_time(self):
        now = time.time()
        entry = _make_entry(timestamp=now)
        mem = _make_memory(entries=[entry], last_update_time=now)
        future = now + 5.0
        new_mem = mem.apply_decay(current_time=future)
        assert new_mem.last_update_time == pytest.approx(future)

    def test_preserves_context(self):
        now = time.time()
        entry = _make_entry(timestamp=now)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            context_topics=["a", "b"],
            continuity_score=0.7,
        )
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert new_mem.current_context_topics == ["a", "b"]
        assert new_mem.context_continuity_score == 0.7

    def test_preserves_processed_flag(self):
        now = time.time()
        entry = _make_entry(timestamp=now, processed=True)
        mem = _make_memory(entries=[entry], last_update_time=now)
        new_mem = mem.apply_decay(current_time=now + 0.5)
        assert new_mem.entries[0].processed is True


# ── 13. apply_decay removes entries below min_weight ─────────


class TestApplyDecayRemoval:
    """Verify entries with weight below min_residue_weight are removed."""

    def test_removes_below_threshold(self):
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=0.01)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 0.1, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        # After 2 seconds: 0.01 * 0.1^2 = 0.0001 < 0.001 threshold
        new_mem = mem.apply_decay(current_time=now + 2.0)
        assert len(new_mem.entries) == 0

    def test_keeps_above_threshold(self):
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 0.9, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        # After 1 second: 1.0 * 0.9^1 = 0.9 >> 0.001
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert len(new_mem.entries) == 1

    def test_partial_removal(self):
        """One entry survives, one is removed."""
        now = time.time()
        strong = _make_entry(timestamp=now, residue_weight=1.0, emotion_label="strong")
        weak = _make_entry(timestamp=now, residue_weight=0.002, emotion_label="weak")
        mem = _make_memory(
            entries=[strong, weak],
            last_update_time=now,
            scale_factors={"decay_base": 0.1, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        # After 1s: strong = 1.0*0.1 = 0.1 (kept), weak = 0.002*0.1 = 0.0002 (removed)
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert len(new_mem.entries) == 1
        assert new_mem.entries[0].emotion_label == "strong"

    def test_custom_min_weight(self):
        """Use a high min_weight threshold to aggressively cull."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"decay_base": 0.9, "min_residue_weight": 0.95,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        # After 1s: 1.0 * 0.9^1 = 0.9 < 0.95 threshold => removed
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert len(new_mem.entries) == 0


# ── 14. apply_decay with custom function ─────────────────────


class TestApplyDecayCustom:
    """Verify custom decay function is used instead of default."""

    def test_custom_halving(self):
        """Custom function that halves weight regardless of time."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"min_residue_weight": 0.001, "decay_base": 1.0,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )

        def halve(weight: float, dt: float) -> float:
            return weight * 0.5

        new_mem = mem.apply_decay(decay_function=halve, current_time=now + 1.0)
        assert new_mem.entries[0].residue_weight == pytest.approx(0.5)

    def test_custom_zero_function_removes_all(self):
        """Custom function returning 0 should remove everything."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"min_residue_weight": 0.001, "decay_base": 1.0,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )

        def zero_decay(weight: float, dt: float) -> float:
            return 0.0

        new_mem = mem.apply_decay(decay_function=zero_decay, current_time=now + 1.0)
        assert len(new_mem.entries) == 0

    def test_custom_identity_preserves_all(self):
        """Custom function returning original weight should preserve entries."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=0.8)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"min_residue_weight": 0.001, "decay_base": 1.0,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )

        def identity(weight: float, dt: float) -> float:
            return weight

        new_mem = mem.apply_decay(decay_function=identity, current_time=now + 100.0)
        assert len(new_mem.entries) == 1
        assert new_mem.entries[0].residue_weight == pytest.approx(0.8)

    def test_custom_time_dependent(self):
        """Custom function that uses elapsed time."""
        now = time.time()
        entry = _make_entry(timestamp=now, residue_weight=1.0)
        mem = _make_memory(
            entries=[entry],
            last_update_time=now,
            scale_factors={"min_residue_weight": 0.001, "decay_base": 1.0,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )

        def linear_decay(weight: float, dt: float) -> float:
            return max(0.0, weight - 0.1 * dt)

        # After 5s: 1.0 - 0.1*5 = 0.5
        new_mem = mem.apply_decay(decay_function=linear_decay, current_time=now + 5.0)
        assert new_mem.entries[0].residue_weight == pytest.approx(0.5)


# ── 15. to_dict / from_dict roundtrip ────────────────────────


class TestSerialization:
    """Verify to_dict and from_dict produce faithful roundtrips."""

    def test_empty_memory_roundtrip(self):
        mem = _make_memory()
        d = mem.to_dict()
        restored = ShortTermMemory.from_dict(d)
        assert restored.entries == []
        assert restored.max_entries == mem.max_entries
        assert restored.current_context_topics == []
        assert restored.context_continuity_score == 0.0

    def test_with_entries_roundtrip(self):
        now = time.time()
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.8, valence=0.5,
                        residue_weight=0.9, processed=False, timestamp=now,
                        topics=["music", "dance"]),
            _make_entry(emotion_label="sad", raw_intensity=0.3, valence=-0.4,
                        residue_weight=0.5, processed=True, timestamp=now - 10,
                        topics=["loss"]),
        ]
        mem = _make_memory(
            entries=entries,
            max_entries=5,
            context_topics=["music", "loss"],
            continuity_score=0.6,
            scale_factors={"residue_influence": 1.5, "decay_base": 0.95,
                           "accumulation_rate": 1.0, "continuity_threshold": 0.3},
            last_update_time=now,
        )

        d = mem.to_dict()
        restored = ShortTermMemory.from_dict(d)

        assert len(restored.entries) == 2
        assert restored.max_entries == 5
        assert set(restored.current_context_topics) == {"music", "loss"}
        assert restored.context_continuity_score == pytest.approx(0.6)
        assert restored.scale_factors["residue_influence"] == 1.5
        assert restored.scale_factors["decay_base"] == 0.95

    def test_entry_fields_preserved(self):
        now = time.time()
        entry = _make_entry(
            emotion_label="anger",
            raw_intensity=0.9,
            valence=-0.8,
            residue_weight=0.7,
            processed=True,
            topics=["conflict"],
            timestamp=now,
        )
        mem = _make_memory(entries=[entry], last_update_time=now)
        d = mem.to_dict()
        restored = ShortTermMemory.from_dict(d)
        e = restored.entries[0]

        assert e.source_text == "test"
        assert e.topics == ["conflict"]
        assert e.emotion_label == "anger"
        assert e.intent == "inform"
        assert e.raw_intensity == pytest.approx(0.9)
        assert e.valence == pytest.approx(-0.8)
        assert e.timestamp == pytest.approx(now)
        assert e.residue_weight == pytest.approx(0.7)
        assert e.processed is True

    def test_to_dict_returns_dict(self):
        mem = _make_memory()
        d = mem.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_expected_keys(self):
        mem = _make_memory()
        d = mem.to_dict()
        expected_keys = {
            "entries", "max_entries", "last_update_time",
            "current_context_topics", "context_continuity_score", "scale_factors",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_with_missing_keys_uses_defaults(self):
        """from_dict should handle missing keys gracefully."""
        restored = ShortTermMemory.from_dict({})
        assert restored.entries == []
        assert restored.max_entries == 10
        assert restored.current_context_topics == []
        assert restored.context_continuity_score == 0.0


# ── 16. compute_residue_influence basic ──────────────────────


class TestComputeResidueInfluence:
    """Verify basic residue influence computation."""

    def test_returns_residue_influence_type(self):
        mem = _make_memory()
        result = compute_residue_influence(mem)
        assert isinstance(result, ResidueInfluence)

    def test_empty_memory(self):
        mem = _make_memory()
        result = compute_residue_influence(mem)
        assert result.emotion_influences == {}
        assert result.total_intensity == 0.0

    def test_single_unprocessed_entry(self):
        entry = _make_entry(emotion_label="joy", raw_intensity=0.6, residue_weight=1.0)
        mem = _make_memory(entries=[entry])
        result = compute_residue_influence(mem)
        assert result.emotion_influences == {"joy": pytest.approx(0.6)}
        assert result.total_intensity == pytest.approx(0.6)

    def test_multiple_entries(self):
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.5, residue_weight=1.0),
            _make_entry(emotion_label="sad", raw_intensity=0.3, residue_weight=0.8),
        ]
        mem = _make_memory(entries=entries)
        result = compute_residue_influence(mem)
        assert result.emotion_influences["joy"] == pytest.approx(0.5)
        assert result.emotion_influences["sad"] == pytest.approx(0.24)
        assert result.total_intensity == pytest.approx(0.74)

    def test_uses_default_scale_from_memory(self):
        mem = _make_memory(scale_factors={
            "residue_influence": 2.5, "decay_base": 1.0,
            "accumulation_rate": 1.0, "continuity_threshold": 0.0,
        })
        result = compute_residue_influence(mem)
        assert result.scale == 2.5

    def test_includes_continuity_score(self):
        mem = _make_memory(continuity_score=0.8)
        result = compute_residue_influence(mem)
        assert result.continuity == 0.8

    def test_skips_processed_entries(self):
        entries = [
            _make_entry(emotion_label="joy", raw_intensity=0.5, processed=False),
            _make_entry(emotion_label="sad", raw_intensity=0.3, processed=True),
        ]
        mem = _make_memory(entries=entries)
        result = compute_residue_influence(mem)
        assert "joy" in result.emotion_influences
        assert "sad" not in result.emotion_influences


# ── 17. compute_residue_influence with scale_factor override ──


class TestComputeResidueInfluenceScaleOverride:
    """Verify scale_factor parameter overrides memory's default."""

    def test_override_scale(self):
        mem = _make_memory(scale_factors={
            "residue_influence": 1.0, "decay_base": 1.0,
            "accumulation_rate": 1.0, "continuity_threshold": 0.0,
        })
        result = compute_residue_influence(mem, scale_factor=3.0)
        assert result.scale == 3.0

    def test_override_does_not_change_memory(self):
        mem = _make_memory(scale_factors={
            "residue_influence": 1.0, "decay_base": 1.0,
            "accumulation_rate": 1.0, "continuity_threshold": 0.0,
        })
        compute_residue_influence(mem, scale_factor=5.0)
        assert mem.scale_factors["residue_influence"] == 1.0

    def test_none_uses_memory_default(self):
        mem = _make_memory(scale_factors={
            "residue_influence": 2.0, "decay_base": 1.0,
            "accumulation_rate": 1.0, "continuity_threshold": 0.0,
        })
        result = compute_residue_influence(mem, scale_factor=None)
        assert result.scale == 2.0


# ── 18. Immutability ─────────────────────────────────────────


class TestImmutability:
    """Verify all mutating operations return new instances."""

    def test_add_stimulus_immutable(self):
        mem = _make_memory()
        new_mem = mem.add_stimulus("t", ["x"], "joy", "share", 0.5, 0.5)
        assert mem is not new_mem
        assert len(mem.entries) == 0
        assert len(new_mem.entries) == 1

    def test_mark_processed_immutable(self):
        entries = [_make_entry(processed=False)]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed()
        assert mem is not new_mem
        assert mem.entries[0].processed is False
        assert new_mem.entries[0].processed is True

    def test_mark_processed_specific_immutable(self):
        entries = [_make_entry(processed=False), _make_entry(processed=False)]
        mem = _make_memory(entries=entries)
        new_mem = mem.mark_processed(entry_indices=[0])
        assert mem is not new_mem
        assert mem.entries[0].processed is False

    def test_update_context_immutable(self):
        mem = _make_memory(context_topics=["a"])
        new_mem = mem.update_context(["b"], is_continuous=True)
        assert mem is not new_mem
        assert set(mem.current_context_topics) == {"a"}
        assert "b" in new_mem.current_context_topics

    def test_reset_on_discontinuity_immutable(self):
        entries = [_make_entry()]
        mem = _make_memory(entries=entries, context_topics=["a"])
        new_mem = mem.reset_on_discontinuity()
        assert mem is not new_mem
        assert len(mem.entries) == 1
        assert len(new_mem.entries) == 0

    def test_apply_decay_immutable(self):
        now = time.time()
        entries = [_make_entry(timestamp=now, residue_weight=1.0)]
        mem = _make_memory(
            entries=entries,
            last_update_time=now,
            scale_factors={"decay_base": 0.5, "min_residue_weight": 0.001,
                           "residue_influence": 1.0, "accumulation_rate": 1.0,
                           "continuity_threshold": 0.0},
        )
        new_mem = mem.apply_decay(current_time=now + 1.0)
        assert mem is not new_mem
        assert mem.entries[0].residue_weight == 1.0
        assert new_mem.entries[0].residue_weight == pytest.approx(0.5)

    def test_scale_factors_not_shared_reference(self):
        """Verify scale_factors dict is copied, not shared."""
        mem = _make_memory()
        new_mem = mem.add_stimulus("t", ["x"], "joy", "share", 0.5, 0.5)
        new_mem.scale_factors["custom_key"] = 99.0
        assert "custom_key" not in mem.scale_factors

    def test_entries_list_not_shared_reference(self):
        """Verify entries list is copied, not shared."""
        entry = _make_entry()
        mem = _make_memory(entries=[entry])
        new_mem = mem.add_stimulus("t", ["x"], "joy", "share", 0.5, 0.5)
        assert len(mem.entries) == 1
        assert len(new_mem.entries) == 2
