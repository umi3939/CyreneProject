"""
Tests for psyche/memory_link.py

Mood-Congruent Memory Recall のテスト。
_estimate_memory_valence と recall_with_mood の全ケースを網羅。
"""

import pytest
import pytest_asyncio

from psyche.memory_link import _estimate_memory_valence, recall_with_mood
from psyche.state import Percept, PsycheState, Mood
from psyche.pillars import AttachmentState


# =============================================================================
# Mock Memory Object
# =============================================================================

class MockMemory:
    """Mock memory object with async recall(query, top_k) method."""

    def __init__(self, candidates: list[dict] | None = None):
        self._candidates = candidates or []
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    async def recall(self, query: str, top_k: int = 6) -> list[dict]:
        self.last_query = query
        self.last_top_k = top_k
        return self._candidates[:top_k]


# =============================================================================
# Helpers
# =============================================================================

def _make_mem(summary: str = "", keywords: list[str] | None = None) -> dict:
    """Create a minimal memory dict."""
    d: dict = {"summary": summary}
    if keywords is not None:
        d["keywords"] = keywords
    return d


def _make_state(valence: float = 0.0, attachment: AttachmentState | None = None) -> PsycheState:
    """Create a PsycheState with a given mood valence and optional attachment."""
    return PsycheState(
        mood=Mood(valence=valence),
        attachment=attachment,
    )


def _make_percept(text: str = "hello") -> Percept:
    return Percept(text=text)


# =============================================================================
# Test _estimate_memory_valence
# =============================================================================

class TestEstimateMemoryValence:
    """Tests for _estimate_memory_valence."""

    def test_positive_keywords_return_positive_value(self):
        """Positive keywords in summary produce a positive valence."""
        mem = _make_mem(summary="I feel happy and joy")
        result = _estimate_memory_valence(mem)
        assert result > 0.0
        assert result <= 1.0

    def test_negative_keywords_return_negative_value(self):
        """Negative keywords in summary produce a negative valence."""
        mem = _make_mem(summary="I feel sad and angry")
        result = _estimate_memory_valence(mem)
        assert result < 0.0
        assert result >= -1.0

    def test_no_keywords_return_zero(self):
        """No matching keywords produce 0.0 valence."""
        mem = _make_mem(summary="the weather is cloudy today")
        result = _estimate_memory_valence(mem)
        assert result == 0.0

    def test_mixed_keywords_partial_positive(self):
        """Mixed positive and negative keywords return a proportional value."""
        # 2 positive (happy, joy) and 1 negative (sad) => (2 - 1) / 3 ~ 0.333
        mem = _make_mem(summary="happy joy sad")
        result = _estimate_memory_valence(mem)
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_mixed_keywords_partial_negative(self):
        """More negative than positive keywords return negative value."""
        # 1 positive (happy) and 2 negative (sad, angry) => (1 - 2) / 3 ~ -0.333
        mem = _make_mem(summary="happy sad angry")
        result = _estimate_memory_valence(mem)
        assert result == pytest.approx(-1 / 3, abs=0.01)

    def test_mixed_keywords_equal_returns_zero(self):
        """Equal positive and negative keywords return 0.0."""
        # 1 positive (happy) and 1 negative (sad) => (1 - 1) / 2 = 0
        mem = _make_mem(summary="happy sad")
        result = _estimate_memory_valence(mem)
        assert result == 0.0

    def test_uses_summary_field(self):
        """Keywords in summary field are detected."""
        mem = _make_mem(summary="great day")
        result = _estimate_memory_valence(mem)
        assert result > 0.0

    def test_uses_keywords_field(self):
        """Keywords in the keywords list field are detected."""
        mem = _make_mem(summary="nothing special", keywords=["love", "fun"])
        result = _estimate_memory_valence(mem)
        assert result > 0.0

    def test_uses_both_summary_and_keywords(self):
        """Both summary and keywords contribute to valence detection."""
        # summary has 1 positive (happy), keywords has 1 negative (fear)
        mem = _make_mem(summary="happy times", keywords=["fear"])
        result = _estimate_memory_valence(mem)
        # (1 - 1) / 2 = 0.0
        assert result == 0.0

    def test_keywords_field_combined_with_summary(self):
        """Keywords in both fields accumulate hits."""
        # summary: "good" (1 positive), keywords: ["great", "thank"] (2 positive) => 3 positive, 0 negative
        mem = _make_mem(summary="good stuff", keywords=["great", "thank"])
        result = _estimate_memory_valence(mem)
        assert result == 1.0

    def test_japanese_positive_keywords(self):
        """Japanese positive keywords are detected."""
        mem = _make_mem(summary="今日は嬉しい日でした。感謝します")
        result = _estimate_memory_valence(mem)
        assert result > 0.0

    def test_japanese_negative_keywords(self):
        """Japanese negative keywords are detected."""
        mem = _make_mem(summary="悲しい出来事があり不安を感じた")
        result = _estimate_memory_valence(mem)
        assert result < 0.0

    def test_empty_summary_no_keywords(self):
        """Empty summary and no keywords return 0.0."""
        mem = _make_mem(summary="")
        result = _estimate_memory_valence(mem)
        assert result == 0.0

    def test_missing_summary_and_keywords(self):
        """Missing fields don't raise; return 0.0."""
        result = _estimate_memory_valence({})
        assert result == 0.0

    def test_case_insensitive_matching(self):
        """Keyword matching is case-insensitive."""
        mem = _make_mem(summary="HAPPY and GREAT day")
        result = _estimate_memory_valence(mem)
        assert result > 0.0

    def test_all_positive_keywords_return_one(self):
        """When only positive keywords are present, returns 1.0."""
        mem = _make_mem(summary="happy joy love fun good great thank")
        result = _estimate_memory_valence(mem)
        assert result == 1.0

    def test_all_negative_keywords_return_minus_one(self):
        """When only negative keywords are present, returns -1.0."""
        mem = _make_mem(summary="sad angry fear bad hate worry pain")
        result = _estimate_memory_valence(mem)
        assert result == -1.0


# =============================================================================
# Test recall_with_mood
# =============================================================================

class TestRecallWithMoodEmptyCandidates:
    """Tests for recall_with_mood when memory returns no candidates."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_list(self):
        """When memory.recall returns empty list, recall_with_mood returns []."""
        memory = MockMemory(candidates=[])
        state = _make_state(valence=0.5)
        percept = _make_percept("test query")
        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_candidates_still_calls_recall(self):
        """Even with no results, memory.recall is called."""
        memory = MockMemory(candidates=[])
        state = _make_state(valence=0.0)
        percept = _make_percept("some query")
        await recall_with_mood(percept, state, memory, top_k=3)
        assert memory.last_query == "some query"


class TestRecallWithMoodReRanking:
    """Tests for mood-congruent re-ranking behavior."""

    @pytest.mark.asyncio
    async def test_positive_mood_prefers_positive_memories(self):
        """With positive mood, positive memories should be ranked higher."""
        positive_mem = _make_mem(summary="happy joy love", keywords=["good"])
        negative_mem = _make_mem(summary="sad angry fear", keywords=["bad"])
        neutral_mem = _make_mem(summary="the weather is cloudy")

        # Put negative first, then neutral, then positive in original order
        candidates = [negative_mem, neutral_mem, positive_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.8)  # strong positive mood
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert len(result) == 3
        # Positive memory should be ranked first despite being last in original order
        # positive_mem has valence ~1.0, congruence_bonus = 0.8 * 1.0 * 2.0 = 1.6
        # negative_mem has valence ~-1.0, congruence_bonus = 0.8 * -1.0 * 2.0 = -1.6
        assert result[0] is positive_mem or result[0]["summary"] == positive_mem["summary"]

    @pytest.mark.asyncio
    async def test_negative_mood_prefers_negative_memories(self):
        """With negative mood, negative memories should be ranked higher."""
        positive_mem = _make_mem(summary="happy joy love")
        negative_mem = _make_mem(summary="sad angry fear")

        # Put positive first in original order
        candidates = [positive_mem, negative_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=-0.8)  # strong negative mood
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        assert len(result) == 2
        # Negative memory should be ranked first
        # negative_mem has valence ~-1.0, congruence_bonus = -0.8 * -1.0 * 2.0 = 1.6
        # positive_mem has valence ~1.0, congruence_bonus = -0.8 * 1.0 * 2.0 = -1.6
        assert result[0] is negative_mem or result[0]["summary"] == negative_mem["summary"]

    @pytest.mark.asyncio
    async def test_neutral_mood_preserves_original_order(self):
        """With neutral mood (valence=0), congruence bonus is 0 for all, so position-based rank holds."""
        mem_a = _make_mem(summary="happy event")
        mem_b = _make_mem(summary="sad event")
        mem_c = _make_mem(summary="neutral event")

        candidates = [mem_a, mem_b, mem_c]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)  # neutral mood
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert len(result) == 3
        # With valence=0, congruence_bonus = 0 for all.
        # base_score = fetch_k - idx, so original order preserved
        assert result[0] is mem_a
        assert result[1] is mem_b
        assert result[2] is mem_c


class TestRecallWithMoodTopK:
    """Tests for top_k limit and fetch_k calculation."""

    @pytest.mark.asyncio
    async def test_respects_top_k_limit(self):
        """Returns at most top_k results."""
        candidates = [_make_mem(summary=f"memory {i}") for i in range(10)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_returns_fewer_than_top_k_if_not_enough_candidates(self):
        """When fewer candidates exist than top_k, returns all of them."""
        candidates = [_make_mem(summary="only one")]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=5)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetches_2x_top_k_candidates(self):
        """recall_with_mood requests 2*top_k (min 6) candidates from memory."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(20)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        await recall_with_mood(percept, state, memory, top_k=5)
        # fetch_k = max(5*2, 6) = 10
        assert memory.last_top_k == 10

    @pytest.mark.asyncio
    async def test_fetch_k_minimum_is_6(self):
        """When top_k=2, fetch_k should be max(4, 6) = 6."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(10)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        await recall_with_mood(percept, state, memory, top_k=2)
        assert memory.last_top_k == 6

    @pytest.mark.asyncio
    async def test_fetch_k_for_top_k_1(self):
        """When top_k=1, fetch_k should be max(2, 6) = 6."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(10)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        await recall_with_mood(percept, state, memory, top_k=1)
        assert memory.last_top_k == 6

    @pytest.mark.asyncio
    async def test_fetch_k_for_top_k_4(self):
        """When top_k=4, fetch_k should be max(8, 6) = 8."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(10)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        await recall_with_mood(percept, state, memory, top_k=4)
        assert memory.last_top_k == 8

    @pytest.mark.asyncio
    async def test_default_top_k_is_3(self):
        """Default top_k is 3, so fetch_k = max(6, 6) = 6."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(10)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory)
        assert len(result) == 3
        assert memory.last_top_k == 6


class TestRecallWithMoodAttachmentBonus:
    """Tests for the attachment partner bonus."""

    @pytest.mark.asyncio
    async def test_attachment_partner_bonus_applied(self):
        """Memories mentioning a top partner get a +2.0 bonus."""
        # Partner "Alice" has bond
        attachment = AttachmentState(bonds={"Alice": 0.9, "Bob": 0.3})

        partner_mem = _make_mem(summary="talked with alice today")
        non_partner_mem = _make_mem(summary="went to the store")

        # Put non-partner first so it has higher base_score from position
        candidates = [non_partner_mem, partner_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # non_partner_mem: base_score = 6 - 0 = 6, bonus = 0 => final = 6
        # partner_mem: base_score = 6 - 1 = 5, bonus = +2.0 => final = 7
        # Partner mem should rank first
        assert result[0] is partner_mem

    @pytest.mark.asyncio
    async def test_attachment_partner_bonus_in_keywords(self):
        """Partner name found in keywords field also triggers bonus."""
        attachment = AttachmentState(bonds={"Bob": 0.8})

        partner_mem = _make_mem(summary="event today", keywords=["Bob", "meeting"])
        other_mem = _make_mem(summary="different event", keywords=["work"])

        candidates = [other_mem, partner_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # other_mem: base=6, bonus=0 => 6
        # partner_mem: base=5, bonus=+2.0 => 7
        assert result[0] is partner_mem

    @pytest.mark.asyncio
    async def test_attachment_bonus_only_once_per_memory(self):
        """Even if multiple partners match, bonus is applied only once (+2.0 total)."""
        attachment = AttachmentState(bonds={"Alice": 0.9, "Bob": 0.8})

        # Memory mentions both Alice and Bob
        multi_partner_mem = _make_mem(summary="alice and bob talked")
        other_mem = _make_mem(summary="unrelated event")

        candidates = [other_mem, multi_partner_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # other: base=6, bonus=0 => 6
        # multi_partner: base=5, bonus=+2.0 (breaks after first match) => 7
        assert result[0] is multi_partner_mem

    @pytest.mark.asyncio
    async def test_no_attachment_state_no_bonus(self):
        """When state.attachment is None, no partner bonus is applied."""
        mem_a = _make_mem(summary="talked with alice today")
        mem_b = _make_mem(summary="went to the store")

        # mem_a is first, has higher base_score
        candidates = [mem_a, mem_b]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=None)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # No attachment => no partner bonus. Position order preserved.
        assert result[0] is mem_a
        assert result[1] is mem_b

    @pytest.mark.asyncio
    async def test_attachment_empty_bonds_no_bonus(self):
        """When attachment exists but has no bonds, no partner bonus is applied."""
        attachment = AttachmentState(bonds={})

        mem_a = _make_mem(summary="alice event")
        mem_b = _make_mem(summary="bob event")

        candidates = [mem_a, mem_b]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # No bonds => no partners => position order
        assert result[0] is mem_a
        assert result[1] is mem_b

    @pytest.mark.asyncio
    async def test_partner_name_case_insensitive(self):
        """Partner name matching is case-insensitive."""
        attachment = AttachmentState(bonds={"ALICE": 0.9})

        partner_mem = _make_mem(summary="alice was here")
        other_mem = _make_mem(summary="some other event")

        candidates = [other_mem, partner_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=2)
        # "ALICE".lower() = "alice", and "alice" is in "alice was here"
        assert result[0] is partner_mem


class TestRecallWithMoodCombinedScoring:
    """Tests for the combined scoring of mood congruence + attachment bonus."""

    @pytest.mark.asyncio
    async def test_mood_and_attachment_combined(self):
        """Mood congruence and attachment bonus work together."""
        attachment = AttachmentState(bonds={"Alice": 0.9})

        # Positive memory mentioning partner, with positive mood
        best_mem = _make_mem(summary="happy time with alice", keywords=["joy"])
        # Positive memory without partner
        good_mem = _make_mem(summary="happy joy love")
        # Negative memory
        bad_mem = _make_mem(summary="sad angry fear")

        candidates = [bad_mem, good_mem, best_mem]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.8, attachment=attachment)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        # best_mem: base=6-2=4, mood=0.8*1.0*2.0=1.6, partner=+2.0 => 7.6
        # good_mem: base=6-1=5, mood=0.8*1.0*2.0=1.6 => 6.6
        # bad_mem: base=6-0=6, mood=0.8*-1.0*2.0=-1.6 => 4.4
        assert result[0] is best_mem
        assert result[1] is good_mem
        assert result[2] is bad_mem

    @pytest.mark.asyncio
    async def test_congruence_bonus_scales_with_valence_magnitude(self):
        """Stronger mood valence leads to larger congruence bonus."""
        pos_mem = _make_mem(summary="happy joy")
        neg_mem = _make_mem(summary="sad angry")
        candidates = [neg_mem, pos_mem]

        # Weak positive mood
        memory_weak = MockMemory(candidates=list(candidates))
        state_weak = _make_state(valence=0.2)
        percept = _make_percept("test")
        result_weak = await recall_with_mood(percept, state_weak, memory_weak, top_k=2)

        # Strong positive mood
        memory_strong = MockMemory(candidates=list(candidates))
        state_strong = _make_state(valence=0.9)
        result_strong = await recall_with_mood(percept, state_strong, memory_strong, top_k=2)

        # With strong positive mood, positive memory should more likely be first
        # With weak mood, position advantage may prevail
        # But with strong mood (0.9), congruence_bonus for pos = 0.9*1.0*2.0 = 1.8
        # neg_mem: base=6, bonus=0.9*-1.0*2.0=-1.8 => 4.2
        # pos_mem: base=5, bonus=0.9*1.0*2.0=1.8 => 6.8
        assert result_strong[0] is pos_mem

    @pytest.mark.asyncio
    async def test_percept_text_used_as_query(self):
        """percept.text is passed as the query to memory.recall."""
        memory = MockMemory(candidates=[])
        state = _make_state(valence=0.0)
        percept = _make_percept("specific query text")

        await recall_with_mood(percept, state, memory, top_k=3)
        assert memory.last_query == "specific query text"

    @pytest.mark.asyncio
    async def test_results_sorted_descending_by_final_score(self):
        """Results are sorted by final_score in descending order."""
        # Create memories with clear valence differences
        strong_pos = _make_mem(summary="happy joy love fun good great thank")
        mild_pos = _make_mem(summary="happy")
        neutral = _make_mem(summary="the weather today")
        mild_neg = _make_mem(summary="sad")
        strong_neg = _make_mem(summary="sad angry fear bad hate worry pain")

        # Place them in reverse order of expected ranking
        candidates = [strong_neg, mild_neg, neutral, mild_pos, strong_pos]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=1.0)  # maximum positive mood
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=5)

        # With valence=1.0, strong positive mem gets biggest congruence boost
        # Verify the first result is a positive memory
        first_valence = _estimate_memory_valence(result[0])
        last_valence = _estimate_memory_valence(result[-1])
        assert first_valence >= last_valence


class TestRecallWithMoodEdgeCases:
    """Edge case tests for recall_with_mood."""

    @pytest.mark.asyncio
    async def test_single_candidate(self):
        """Single candidate is returned regardless of mood."""
        mem = _make_mem(summary="only memory")
        memory = MockMemory(candidates=[mem])
        state = _make_state(valence=0.5)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert len(result) == 1
        assert result[0] is mem

    @pytest.mark.asyncio
    async def test_all_neutral_memories_preserve_position_order(self):
        """When all memories are neutral, position-based scoring dominates."""
        mems = [_make_mem(summary=f"neutral memory {i}") for i in range(6)]
        memory = MockMemory(candidates=mems)
        state = _make_state(valence=0.5)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        # All congruence bonuses are 0 (neutral memories), so base_score order holds
        assert result[0] is mems[0]
        assert result[1] is mems[1]
        assert result[2] is mems[2]

    @pytest.mark.asyncio
    async def test_memory_without_summary_or_keywords(self):
        """Memories missing summary/keywords fields don't cause errors."""
        mem = {}  # no summary, no keywords
        memory = MockMemory(candidates=[mem])
        state = _make_state(valence=0.5)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=3)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_top_k_larger_than_candidates(self):
        """When top_k exceeds available candidates, returns all available."""
        candidates = [_make_mem(summary=f"m{i}") for i in range(2)]
        memory = MockMemory(candidates=candidates)
        state = _make_state(valence=0.0)
        percept = _make_percept("test")

        result = await recall_with_mood(percept, state, memory, top_k=10)
        assert len(result) == 2
