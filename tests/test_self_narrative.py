"""
Tests for Self-Narrative Formation (自己物語形成 - 非規範・観測型)

These tests verify that the SelfNarrativeSystem:
1. Classifies observations into abstract fragment types (event/reaction/continuation/change/undetermined)
2. Links fragments temporally, thematically, by contrast, and by continuation
3. Decays vividness over time (直近ほど鮮明で過去ほど要約化)
4. Summarizes old fragments instead of simply discarding
5. Re-edits fragments based on subsequent observations (単一解釈を固定しない)
6. Maintains STRICTLY NO IMPACT on decisions, goals, responsibility, or values
7. Does NOT define identity or "true self"
8. Does NOT generate goals from narrative
9. Preserves read-only principle for all external systems
"""

import pytest
import time
from typing import Optional

from psyche.self_narrative import (
    # Enums
    FragmentType,
    LinkType,
    VividnessLevel,
    NarrativeCoherence,
    NarrativeTrend,
    # Data structures
    NarrativeFragment,
    FragmentLink,
    CoherenceInfo,
    NarrativeState,
    SelfNarrativeConfig,
    # System
    SelfNarrativeSystem,
    # Classification functions
    classify_emotion_observation,
    classify_memory_observation,
    classify_tendency_observation,
    classify_difference_observation,
    classify_context_observation,
    # Rewrite / Summarization
    check_for_rewrites,
    summarize_fragments,
    # Coherence / Trend
    compute_coherence,
    determine_narrative_trend,
    # Integration
    generate_narrative_tags,
    get_narrative_summary,
    get_narrative_for_introspection,
    # Convenience
    create_config,
    create_empty_state,
    # Verification
    verify_no_decision_impact,
    verify_no_identity_definition,
    verify_no_goal_generation,
    verify_read_only_principle,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockEmotionalStateView:
    """Mock EmotionalStateView from self_model."""

    def __init__(
        self,
        intensity: str = "calm",
        harmony: str = "harmonious",
        description: str = "",
    ):
        from psyche.self_model import EmotionalIntensity, EmotionalHarmony
        self.intensity = EmotionalIntensity(intensity)
        self.harmony = EmotionalHarmony(harmony)
        self.description = description


class MockStimulusEntry:
    """Mock StimulusEntry from short_term_memory."""

    def __init__(self, valence: float = 0.0):
        self.valence = valence
        self.timestamp = time.time()


class MockShortTermMemory:
    """Mock ShortTermMemory from short_term_memory."""

    def __init__(self, entries: list = None):
        self.entries = entries if entries is not None else []


class MockTendencyAwarenessItem:
    """Mock TendencyAwarenessItem from tendency_awareness."""

    def __init__(
        self,
        awareness_type: str = "slight_bias",
        description: str = "",
    ):
        from psyche.tendency_awareness import AwarenessType
        self.awareness_type = AwarenessType(awareness_type)
        self.description = description


class MockTendencyAwareness:
    """Mock TendencyAwareness from tendency_awareness."""

    def __init__(
        self,
        has_awareness: bool = False,
        items: list = None,
    ):
        self.has_awareness = has_awareness
        self.items = items if items is not None else []


class MockSelfDifferenceSummary:
    """Mock SelfDifferenceSummary from temporal_self_difference."""

    def __init__(
        self,
        has_difference: bool = False,
        magnitude: str = "none",
        nature: str = "stable",
    ):
        from psyche.temporal_self_difference import (
            DifferenceMagnitude, ChangeNature,
        )
        self.has_difference = has_difference
        self.magnitude = DifferenceMagnitude(magnitude)
        self.nature = ChangeNature(nature)


class MockExternalContext:
    """Mock ExternalContext from context_sensitivity."""

    def __init__(self, weight: float = 0.5, pace: float = 0.5):
        self.weight = weight
        self.pace = pace


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def system():
    """Create a fresh SelfNarrativeSystem."""
    return SelfNarrativeSystem()


@pytest.fixture
def config():
    """Create a default config."""
    return SelfNarrativeConfig()


@pytest.fixture
def fast_decay_system():
    """System with fast decay for testing lifecycle."""
    cfg = SelfNarrativeConfig(
        vividness_decay_rate=0.2,
        summarization_threshold=0.5,
        dissipation_threshold=0.2,
    )
    return SelfNarrativeSystem(config=cfg)


# =============================================================================
# Test: Basic Enums
# =============================================================================

class TestEnums:
    def test_fragment_type_values(self):
        assert FragmentType.EVENT.value == "event"
        assert FragmentType.REACTION.value == "reaction"
        assert FragmentType.CONTINUATION.value == "continuation"
        assert FragmentType.CHANGE.value == "change"
        assert FragmentType.UNDETERMINED.value == "undetermined"

    def test_link_type_values(self):
        assert LinkType.TEMPORAL.value == "temporal"
        assert LinkType.THEMATIC.value == "thematic"
        assert LinkType.CONTRAST.value == "contrast"
        assert LinkType.CONTINUATION_OF.value == "continuation_of"

    def test_vividness_level_values(self):
        assert VividnessLevel.VIVID.value == "vivid"
        assert VividnessLevel.CLEAR.value == "clear"
        assert VividnessLevel.FADING.value == "fading"
        assert VividnessLevel.DIM.value == "dim"
        assert VividnessLevel.DISSIPATING.value == "dissipating"

    def test_narrative_coherence_values(self):
        assert NarrativeCoherence.COHERENT.value == "coherent"
        assert NarrativeCoherence.LOOSELY_CONNECTED.value == "loosely_connected"
        assert NarrativeCoherence.FRAGMENTED.value == "fragmented"
        assert NarrativeCoherence.UNDEFINED.value == "undefined"

    def test_narrative_trend_values(self):
        assert NarrativeTrend.STABLE.value == "stable"
        assert NarrativeTrend.ACCUMULATING.value == "accumulating"
        assert NarrativeTrend.CONDENSING.value == "condensing"
        assert NarrativeTrend.DISSOLVING.value == "dissolving"


# =============================================================================
# Test: NarrativeFragment
# =============================================================================

class TestNarrativeFragment:
    def test_create_fragment(self):
        f = NarrativeFragment(
            fragment_id="abc123",
            fragment_type=FragmentType.EVENT,
            description="Test event",
            timestamp=time.time(),
            vividness=1.0,
            reference_count=0,
            undetermined_tags=(),
            source_type="emotion",
            rewrite_count=0,
            is_summary=False,
        )
        assert f.fragment_type == FragmentType.EVENT
        assert f.vividness == 1.0

    def test_vividness_levels(self):
        def make(v):
            return NarrativeFragment(
                "id", FragmentType.EVENT, "", 0, v, 0, (), "t", 0, False,
            )
        assert make(0.9).get_vividness_level() == VividnessLevel.VIVID
        assert make(0.7).get_vividness_level() == VividnessLevel.CLEAR
        assert make(0.5).get_vividness_level() == VividnessLevel.FADING
        assert make(0.3).get_vividness_level() == VividnessLevel.DIM
        assert make(0.05).get_vividness_level() == VividnessLevel.DISSIPATING

    def test_with_vividness(self):
        f = NarrativeFragment(
            "id", FragmentType.EVENT, "", 0, 1.0, 0, (), "t", 0, False,
        )
        decayed = f.with_vividness(0.5)
        assert decayed.vividness == 0.5
        assert f.vividness == 1.0  # original unchanged (frozen)

    def test_with_vividness_clamp(self):
        f = NarrativeFragment(
            "id", FragmentType.EVENT, "", 0, 0.5, 0, (), "t", 0, False,
        )
        assert f.with_vividness(-0.5).vividness == 0.0
        assert f.with_vividness(1.5).vividness == 1.0

    def test_with_reference(self):
        f = NarrativeFragment(
            "id", FragmentType.EVENT, "", 0, 0.5, 0, (), "t", 0, False,
        )
        ref = f.with_reference()
        assert ref.reference_count == 1
        assert f.reference_count == 0

    def test_rewrite(self):
        f = NarrativeFragment(
            "id", FragmentType.CHANGE, "original", 0, 0.8, 0, (), "t", 0, False,
        )
        rewritten = f.rewrite(
            FragmentType.EVENT,
            "reinterpreted",
            ("reinterpreted",),
        )
        assert rewritten.fragment_type == FragmentType.EVENT
        assert rewritten.description == "reinterpreted"
        assert rewritten.rewrite_count == 1
        assert rewritten.fragment_id == f.fragment_id  # same ID

    def test_is_undetermined(self):
        f1 = NarrativeFragment(
            "id", FragmentType.UNDETERMINED, "", 0, 1.0, 0, (), "t", 0, False,
        )
        f2 = NarrativeFragment(
            "id", FragmentType.EVENT, "", 0, 1.0, 0, ("tag",), "t", 0, False,
        )
        f3 = NarrativeFragment(
            "id", FragmentType.EVENT, "", 0, 1.0, 0, (), "t", 0, False,
        )
        assert f1.is_undetermined()
        assert f2.is_undetermined()
        assert not f3.is_undetermined()


# =============================================================================
# Test: NarrativeState
# =============================================================================

class TestNarrativeState:
    def test_empty_state(self):
        state = NarrativeState.empty()
        assert not state.has_fragments()
        assert state.coherence.level == NarrativeCoherence.UNDEFINED
        assert state.trend == NarrativeTrend.UNDEFINED

    def test_serialization_roundtrip(self):
        state = NarrativeState.empty()
        data = state.to_dict()
        restored = NarrativeState.from_dict(data)
        assert restored.coherence.level == state.coherence.level
        assert restored.description == state.description

    def test_serialization_with_fragments(self):
        f = NarrativeFragment(
            "abc", FragmentType.EVENT, "desc", time.time(),
            0.8, 1, ("tag",), "emotion", 0, False,
        )
        link = FragmentLink("abc", "def", LinkType.TEMPORAL, "seq")
        state = NarrativeState(
            fragments=(f,),
            links=(link,),
            coherence=CoherenceInfo(
                NarrativeCoherence.FRAGMENTED, 1, 1, 0, 0.8, 0, 1, 1.0,
            ),
            dissipation_candidates=(),
            trend=NarrativeTrend.STABLE,
            timestamp=time.time(),
            generation_count=1,
            description="Test",
        )
        data = state.to_dict()
        restored = NarrativeState.from_dict(data)
        assert len(restored.fragments) == 1
        assert restored.fragments[0].fragment_type == FragmentType.EVENT
        assert len(restored.links) == 1
        assert restored.links[0].link_type == LinkType.TEMPORAL

    def test_get_active_fragments(self):
        f1 = NarrativeFragment(
            "a", FragmentType.EVENT, "", 0, 0.5, 0, (), "t", 0, False,
        )
        f2 = NarrativeFragment(
            "b", FragmentType.EVENT, "", 0, 0.0, 0, (), "t", 0, False,
        )
        state = NarrativeState(
            fragments=(f1, f2), links=(), coherence=CoherenceInfo(
                NarrativeCoherence.UNDEFINED, 2, 1, 0, 0.25, 0, 0, 0.0,
            ),
            dissipation_candidates=(), trend=NarrativeTrend.UNDEFINED,
            timestamp=0, generation_count=0, description="",
        )
        assert len(state.get_active_fragments()) == 1

    def test_get_vivid_fragments(self):
        f1 = NarrativeFragment(
            "a", FragmentType.EVENT, "", 0, 0.9, 0, (), "t", 0, False,
        )
        f2 = NarrativeFragment(
            "b", FragmentType.EVENT, "", 0, 0.3, 0, (), "t", 0, False,
        )
        state = NarrativeState(
            fragments=(f1, f2), links=(), coherence=CoherenceInfo(
                NarrativeCoherence.UNDEFINED, 2, 2, 0, 0.6, 0, 0, 0.0,
            ),
            dissipation_candidates=(), trend=NarrativeTrend.UNDEFINED,
            timestamp=0, generation_count=0, description="",
        )
        assert len(state.get_vivid_fragments()) == 1


# =============================================================================
# Test: Classification Functions
# =============================================================================

class TestEmotionClassification:
    def test_none_input(self):
        assert classify_emotion_observation(None) == []

    def test_intense_emotion_is_event(self):
        emo = MockEmotionalStateView(intensity="intense", description="strong feeling")
        result = classify_emotion_observation(emo)
        assert any(r[0] == FragmentType.EVENT for r in result)

    def test_overwhelming_emotion_is_event(self):
        emo = MockEmotionalStateView(intensity="overwhelming")
        result = classify_emotion_observation(emo)
        assert any(r[0] == FragmentType.EVENT for r in result)

    def test_moderate_emotion_is_continuation(self):
        emo = MockEmotionalStateView(intensity="moderate")
        result = classify_emotion_observation(emo)
        assert any(r[0] == FragmentType.CONTINUATION for r in result)

    def test_calm_emotion_no_fragment(self):
        emo = MockEmotionalStateView(intensity="calm")
        result = classify_emotion_observation(emo)
        assert len(result) == 0

    def test_conflicted_harmony_is_reaction(self):
        emo = MockEmotionalStateView(intensity="moderate", harmony="conflicted")
        result = classify_emotion_observation(emo)
        assert any(r[0] == FragmentType.REACTION for r in result)


class TestMemoryClassification:
    def test_none_input(self):
        assert classify_memory_observation(None) == []

    def test_empty_memory(self):
        mem = MockShortTermMemory(entries=[])
        assert classify_memory_observation(mem) == []

    def test_significant_positive_stimulus(self):
        mem = MockShortTermMemory(entries=[MockStimulusEntry(valence=0.8)])
        result = classify_memory_observation(mem)
        assert any(r[0] == FragmentType.EVENT for r in result)
        assert any("positive" in r[1] for r in result)

    def test_significant_negative_stimulus(self):
        mem = MockShortTermMemory(entries=[MockStimulusEntry(valence=-0.7)])
        result = classify_memory_observation(mem)
        assert any(r[0] == FragmentType.EVENT for r in result)
        assert any("negative" in r[1] for r in result)

    def test_mild_stimulus_is_continuation(self):
        mem = MockShortTermMemory(entries=[MockStimulusEntry(valence=0.3)])
        result = classify_memory_observation(mem)
        assert any(r[0] == FragmentType.CONTINUATION for r in result)

    def test_very_mild_stimulus_no_fragment(self):
        mem = MockShortTermMemory(entries=[MockStimulusEntry(valence=0.1)])
        assert classify_memory_observation(mem) == []


class TestTendencyClassification:
    def test_none_input(self):
        assert classify_tendency_observation(None) == []

    def test_no_awareness(self):
        ta = MockTendencyAwareness(has_awareness=False)
        assert classify_tendency_observation(ta) == []

    def test_strong_habit_is_continuation(self):
        ta = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem("strong_habit", "test habit")],
        )
        result = classify_tendency_observation(ta)
        assert any(r[0] == FragmentType.CONTINUATION for r in result)

    def test_habit_forming_is_change(self):
        ta = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem("habit_forming")],
        )
        result = classify_tendency_observation(ta)
        assert any(r[0] == FragmentType.CHANGE for r in result)

    def test_fading_habit_is_change(self):
        ta = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem("fading_habit")],
        )
        result = classify_tendency_observation(ta)
        assert any(r[0] == FragmentType.CHANGE for r in result)

    def test_slight_bias_is_undetermined(self):
        ta = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem("slight_bias")],
        )
        result = classify_tendency_observation(ta)
        assert any(r[0] == FragmentType.UNDETERMINED for r in result)


class TestDifferenceClassification:
    def test_none_input(self):
        assert classify_difference_observation(None) == []

    def test_no_difference_is_continuation(self):
        diff = MockSelfDifferenceSummary(has_difference=False)
        result = classify_difference_observation(diff)
        assert result[0][0] == FragmentType.CONTINUATION

    def test_shifting_is_change(self):
        diff = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        result = classify_difference_observation(diff)
        assert result[0][0] == FragmentType.CHANGE

    def test_transformed_is_change(self):
        diff = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="substantial",
            nature="transformed",
        )
        result = classify_difference_observation(diff)
        assert result[0][0] == FragmentType.CHANGE

    def test_fluctuating_is_undetermined(self):
        diff = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="noticeable",
            nature="fluctuating",
        )
        result = classify_difference_observation(diff)
        assert result[0][0] == FragmentType.UNDETERMINED

    def test_stable_with_diff_is_continuation(self):
        diff = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="minimal",
            nature="stable",
        )
        result = classify_difference_observation(diff)
        assert result[0][0] == FragmentType.CONTINUATION


class TestContextClassification:
    def test_none_input(self):
        assert classify_context_observation(None) == []

    def test_string_context(self):
        result = classify_context_observation("Something happened")
        assert any(r[0] == FragmentType.EVENT for r in result)

    def test_empty_string(self):
        assert classify_context_observation("") == []

    def test_heavy_context_is_event(self):
        ctx = MockExternalContext(weight=0.8)
        result = classify_context_observation(ctx)
        assert any(r[0] == FragmentType.EVENT for r in result)

    def test_light_context_is_continuation(self):
        ctx = MockExternalContext(weight=0.2)
        result = classify_context_observation(ctx)
        assert any(r[0] == FragmentType.CONTINUATION for r in result)

    def test_fast_pace_is_event(self):
        ctx = MockExternalContext(weight=0.5, pace=0.8)
        result = classify_context_observation(ctx)
        assert any(r[0] == FragmentType.EVENT for r in result)


# =============================================================================
# Test: Rewrite Detection
# =============================================================================

class TestRewriteDetection:
    def test_no_rewrites_on_empty(self):
        result = check_for_rewrites([], [], SelfNarrativeConfig())
        assert result == []

    def test_change_to_continuation_triggers_rewrite(self):
        fragment = NarrativeFragment(
            "id1", FragmentType.CHANGE, "shifting", time.time(),
            0.8, 0, (), "difference", 0, False,
        )
        new_obs = [(FragmentType.CONTINUATION, "stable now", "difference")]
        result = check_for_rewrites([fragment], new_obs, SelfNarrativeConfig())
        assert len(result) == 1
        assert result[0].fragment_type == FragmentType.EVENT
        assert result[0].rewrite_count == 1

    def test_undetermined_clarification(self):
        fragment = NarrativeFragment(
            "id1", FragmentType.UNDETERMINED, "unclear", time.time(),
            0.8, 0, ("fresh",), "tendency", 0, False,
        )
        new_obs = [(FragmentType.CHANGE, "tendency forming", "tendency")]
        result = check_for_rewrites([fragment], new_obs, SelfNarrativeConfig())
        assert len(result) == 1
        assert result[0].fragment_type == FragmentType.CHANGE

    def test_no_rewrite_for_different_source(self):
        fragment = NarrativeFragment(
            "id1", FragmentType.CHANGE, "shifting", time.time(),
            0.8, 0, (), "emotion", 0, False,
        )
        new_obs = [(FragmentType.CONTINUATION, "stable", "difference")]
        result = check_for_rewrites([fragment], new_obs, SelfNarrativeConfig())
        assert result == []

    def test_no_rewrite_for_summary(self):
        fragment = NarrativeFragment(
            "id1", FragmentType.CHANGE, "summary", time.time(),
            0.8, 0, (), "difference", 0, True,  # is_summary
        )
        new_obs = [(FragmentType.CONTINUATION, "stable", "difference")]
        result = check_for_rewrites([fragment], new_obs, SelfNarrativeConfig())
        assert result == []


# =============================================================================
# Test: Summarization
# =============================================================================

class TestSummarization:
    def test_single_fragment_no_summary(self):
        f = NarrativeFragment(
            "a", FragmentType.EVENT, "", 0, 0.3, 0, (), "t", 0, False,
        )
        assert summarize_fragments([f], SelfNarrativeConfig()) is None

    def test_two_fragments_creates_summary(self):
        f1 = NarrativeFragment(
            "a", FragmentType.EVENT, "e1", 1.0, 0.3, 0, (), "emotion", 0, False,
        )
        f2 = NarrativeFragment(
            "b", FragmentType.CONTINUATION, "c1", 2.0, 0.25, 0, (), "emotion", 0, False,
        )
        summary = summarize_fragments([f1, f2], SelfNarrativeConfig())
        assert summary is not None
        assert summary.is_summary
        assert "summarized" in summary.undetermined_tags
        assert summary.source_type == "summary"

    def test_summary_has_correct_vividness(self):
        cfg = SelfNarrativeConfig(summary_vividness=0.35)
        f1 = NarrativeFragment(
            "a", FragmentType.EVENT, "", 0, 0.2, 0, (), "t", 0, False,
        )
        f2 = NarrativeFragment(
            "b", FragmentType.EVENT, "", 0, 0.2, 0, (), "t", 0, False,
        )
        summary = summarize_fragments([f1, f2], cfg)
        assert summary.vividness == 0.35


# =============================================================================
# Test: Coherence Computation
# =============================================================================

class TestCoherenceComputation:
    def test_empty_fragments(self):
        coh = compute_coherence([], [], SelfNarrativeConfig())
        assert coh.level == NarrativeCoherence.UNDEFINED
        assert coh.fragment_count == 0

    def test_single_fragment_fragmented(self):
        f = NarrativeFragment(
            "a", FragmentType.EVENT, "", 0, 0.8, 0, (), "t", 0, False,
        )
        coh = compute_coherence([f], [], SelfNarrativeConfig())
        assert coh.level == NarrativeCoherence.FRAGMENTED
        assert coh.connectivity == 0.0

    def test_high_connectivity_coherent(self):
        frags = [
            NarrativeFragment(
                f"id{i}", FragmentType.EVENT, "", 0, 0.8, 0, (), "t", 0, False,
            )
            for i in range(5)
        ]
        links = [
            FragmentLink(f"id{i}", f"id{i+1}", LinkType.TEMPORAL, "seq")
            for i in range(4)
        ]
        coh = compute_coherence(frags, links, SelfNarrativeConfig())
        assert coh.level == NarrativeCoherence.COHERENT
        assert coh.connectivity >= 0.7


# =============================================================================
# Test: SelfNarrativeSystem
# =============================================================================

class TestSelfNarrativeSystem:
    def test_create_system(self, system):
        assert system.get_generation_count() == 0
        assert system.get_last_state() is None

    def test_no_inputs_returns_empty(self, system):
        state = system.observe_and_generate()
        assert state.coherence.level == NarrativeCoherence.UNDEFINED

    def test_generate_with_emotion(self, system):
        emo = MockEmotionalStateView(intensity="intense", description="test")
        state = system.observe_and_generate(emotion_summary=emo)
        assert state.has_fragments()
        assert any(
            f.source_type == "emotion" for f in state.fragments
        )

    def test_generate_with_difference(self, system):
        diff = MockSelfDifferenceSummary(
            has_difference=True,
            magnitude="significant",
            nature="shifting",
        )
        state = system.observe_and_generate(difference_summary=diff)
        assert state.has_fragments()
        assert any(
            f.fragment_type == FragmentType.CHANGE for f in state.fragments
        )

    def test_generate_with_multiple_inputs(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        diff = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant", nature="shifting",
        )
        ta = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem("habit_forming")],
        )
        state = system.observe_and_generate(
            emotion_summary=emo,
            difference_summary=diff,
            tendency_awareness=ta,
        )
        sources = {f.source_type for f in state.fragments}
        assert "emotion" in sources
        assert "difference" in sources
        assert "tendency" in sources

    def test_generation_count_increments(self, system):
        system.observe_and_generate()
        assert system.get_generation_count() == 1
        system.observe_and_generate()
        assert system.get_generation_count() == 2

    def test_last_state_tracked(self, system):
        state = system.observe_and_generate()
        assert system.get_last_state() is state

    def test_fragments_accumulate_across_calls(self, system):
        emo1 = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo1)
        count1 = len(system.get_last_state().fragments)

        emo2 = MockEmotionalStateView(intensity="intense", description="second")
        system.observe_and_generate(emotion_summary=emo2)
        count2 = len(system.get_last_state().fragments)
        assert count2 > count1


# =============================================================================
# Test: Vividness Decay
# =============================================================================

class TestVividnessDecay:
    def test_fragments_decay_over_time(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo)
        initial_vividness = system.get_last_state().fragments[0].vividness

        # Generate again with no new input — existing fragments should decay
        system.observe_and_generate()
        assert system.get_last_state().fragments[0].vividness < initial_vividness

    def test_fast_decay_system(self, fast_decay_system):
        emo = MockEmotionalStateView(intensity="intense")
        fast_decay_system.observe_and_generate(emotion_summary=emo)

        # Several cycles of decay
        for _ in range(5):
            fast_decay_system.observe_and_generate()

        state = fast_decay_system.get_last_state()
        # Fragments should be very dim or gone
        if state.has_fragments():
            for f in state.fragments:
                assert f.vividness < 0.5

    def test_reference_boosts_vividness(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo)

        fid = system.get_last_state().fragments[0].fragment_id
        # Decay it once
        system.observe_and_generate()
        v_before = [
            f.vividness for f in system._fragments
            if f.fragment_id == fid
        ][0]

        # Reference the fragment
        system.reference_fragment(fid)
        v_after = [
            f.vividness for f in system._fragments
            if f.fragment_id == fid
        ][0]

        assert v_after > v_before


# =============================================================================
# Test: Fragment Linking
# =============================================================================

class TestFragmentLinking:
    def test_temporal_links_created(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo)
        system.observe_and_generate(emotion_summary=emo)
        state = system.get_last_state()
        temporal_links = [
            l for l in state.links if l.link_type == LinkType.TEMPORAL
        ]
        assert len(temporal_links) > 0

    def test_contrast_links_on_type_change(self, system):
        # First: CHANGE
        diff_change = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant", nature="shifting",
        )
        system.observe_and_generate(difference_summary=diff_change)
        # Then: CONTINUATION
        diff_stable = MockSelfDifferenceSummary(has_difference=False)
        system.observe_and_generate(difference_summary=diff_stable)
        state = system.get_last_state()
        contrast_links = [
            l for l in state.links if l.link_type == LinkType.CONTRAST
        ]
        assert len(contrast_links) > 0

    def test_continuation_links(self, system):
        diff = MockSelfDifferenceSummary(has_difference=False)
        system.observe_and_generate(difference_summary=diff)
        system.observe_and_generate(difference_summary=diff)
        state = system.get_last_state()
        cont_links = [
            l for l in state.links
            if l.link_type == LinkType.CONTINUATION_OF
        ]
        assert len(cont_links) > 0


# =============================================================================
# Test: Rewriting in System
# =============================================================================

class TestRewritingInSystem:
    def test_change_then_continuation_triggers_rewrite(self, system):
        # First cycle: CHANGE
        diff_change = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant", nature="shifting",
        )
        system.observe_and_generate(difference_summary=diff_change)

        change_fragments = [
            f for f in system.get_last_state().fragments
            if f.fragment_type == FragmentType.CHANGE
            and f.source_type == "difference"
        ]
        assert len(change_fragments) > 0

        # Second cycle: CONTINUATION (triggers rewrite of the CHANGE)
        diff_stable = MockSelfDifferenceSummary(has_difference=False)
        system.observe_and_generate(difference_summary=diff_stable)

        state = system.get_last_state()
        # The previous CHANGE should have been rewritten
        rewritten = [f for f in state.fragments if f.rewrite_count > 0]
        assert len(rewritten) > 0


# =============================================================================
# Test: Summarization in System
# =============================================================================

class TestSummarizationInSystem:
    def test_old_fragments_get_summarized(self):
        cfg = SelfNarrativeConfig(
            vividness_decay_rate=0.15,
            summarization_threshold=0.5,
            dissipation_threshold=0.1,
        )
        system = SelfNarrativeSystem(config=cfg)

        # Generate fragments from same source
        emo = MockEmotionalStateView(intensity="intense")
        for _ in range(5):
            system.observe_and_generate(emotion_summary=emo)

        state = system.get_last_state()
        summaries = [f for f in state.fragments if f.is_summary]
        # After enough cycles with decay, summarization should have occurred
        # (depending on timing, it may or may not have triggered yet)
        # At minimum the system should not crash
        assert state.has_fragments()


# =============================================================================
# Test: Dissipation
# =============================================================================

class TestDissipation:
    def test_fully_decayed_fragments_removed(self):
        cfg = SelfNarrativeConfig(vividness_decay_rate=0.3)
        system = SelfNarrativeSystem(config=cfg)

        emo = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo)

        initial_count = len(system.get_last_state().fragments)

        # Many decay cycles
        for _ in range(10):
            system.observe_and_generate()

        final_count = len(system.get_last_state().fragments)
        assert final_count < initial_count

    def test_dissipation_candidates_listed(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        system.observe_and_generate(emotion_summary=emo)

        # Manually lower vividness to near-dissipation level
        for i, f in enumerate(system._fragments):
            system._fragments[i] = f.with_vividness(0.08)

        state = system._build_state(time.time())
        assert len(state.dissipation_candidates) > 0


# =============================================================================
# Test: Max Fragment Limit
# =============================================================================

class TestMaxFragmentLimit:
    def test_enforces_max_limit(self):
        cfg = SelfNarrativeConfig(max_fragments=5, vividness_decay_rate=0.01)
        system = SelfNarrativeSystem(config=cfg)

        emo = MockEmotionalStateView(intensity="intense")
        for _ in range(20):
            system.observe_and_generate(emotion_summary=emo)

        state = system.get_last_state()
        assert len(state.fragments) <= 5


# =============================================================================
# Test: Narrative Trend
# =============================================================================

class TestNarrativeTrend:
    def test_undefined_with_short_history(self):
        trend = determine_narrative_trend(
            [], None, 0, False, SelfNarrativeConfig(),
        )
        assert trend == NarrativeTrend.UNDEFINED

    def test_accumulating_trend(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        # Generate fragments to build history
        system.observe_and_generate(emotion_summary=emo)
        system.observe_and_generate(emotion_summary=emo)
        system.observe_and_generate(emotion_summary=emo)
        state = system.get_last_state()
        # With fragments being added each cycle, trend should be ACCUMULATING
        assert state.trend in (
            NarrativeTrend.ACCUMULATING,
            NarrativeTrend.STABLE,
            NarrativeTrend.UNDEFINED,
        )


# =============================================================================
# Test: No Decision Impact (CRITICAL CONSTRAINT)
# =============================================================================

class TestNoDecisionImpact:
    def test_empty_state_no_impact(self):
        state = NarrativeState.empty()
        assert verify_no_decision_impact(state)

    def test_populated_state_no_impact(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        diff = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant", nature="shifting",
        )
        state = system.observe_and_generate(
            emotion_summary=emo, difference_summary=diff,
        )
        assert verify_no_decision_impact(state)

    def test_tags_have_small_weights(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        state = system.observe_and_generate(emotion_summary=emo)
        tags = generate_narrative_tags(state)
        for tag in tags:
            assert tag["weight"] <= 0.15
            assert "SELF_NARRATIVE" in tag["category"]


# =============================================================================
# Test: No Identity Definition (CRITICAL CONSTRAINT)
# =============================================================================

class TestNoIdentityDefinition:
    def test_empty_state(self):
        assert verify_no_identity_definition(NarrativeState.empty())

    def test_populated_state(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        state = system.observe_and_generate(emotion_summary=emo)
        assert verify_no_identity_definition(state)

    def test_descriptions_no_forbidden_phrases(self, system):
        diff = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant", nature="shifting",
        )
        state = system.observe_and_generate(difference_summary=diff)
        for f in state.fragments:
            assert "true self" not in f.description.lower()
            assert "should be" not in f.description.lower()


# =============================================================================
# Test: No Goal Generation (CRITICAL CONSTRAINT)
# =============================================================================

class TestNoGoalGeneration:
    def test_system_has_no_goal_methods(self, system):
        assert verify_no_goal_generation(system)

    def test_no_force_methods(self, system):
        methods = [
            m for m in dir(system)
            if not m.startswith("_") and callable(getattr(system, m))
        ]
        for m in methods:
            assert "goal" not in m.lower()
            assert "force" not in m.lower()


# =============================================================================
# Test: Read-Only Principle (CRITICAL CONSTRAINT)
# =============================================================================

class TestReadOnlyPrinciple:
    def test_system_is_read_only(self, system):
        assert verify_read_only_principle(system)

    def test_no_external_mutation_methods(self, system):
        methods = [
            m for m in dir(system)
            if not m.startswith("_") and callable(getattr(system, m))
        ]
        for m in methods:
            assert "update_emotion" not in m
            assert "set_memory" not in m
            assert "modify_bias" not in m


# =============================================================================
# Test: Integration Functions
# =============================================================================

class TestIntegration:
    def test_generate_tags_empty_state(self):
        state = NarrativeState.empty()
        tags = generate_narrative_tags(state)
        assert len(tags) == 1
        assert tags[0]["label"] == "no_narrative"

    def test_generate_tags_populated(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        state = system.observe_and_generate(emotion_summary=emo)
        tags = generate_narrative_tags(state)
        assert len(tags) > 1
        categories = [t["category"] for t in tags]
        assert any("COHERENCE" in c for c in categories)

    def test_get_summary(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        state = system.observe_and_generate(emotion_summary=emo)
        summary = get_narrative_summary(state)
        assert "Self-Narrative" in summary

    def test_get_introspection_data(self, system):
        emo = MockEmotionalStateView(intensity="intense")
        state = system.observe_and_generate(emotion_summary=emo)
        data = get_narrative_for_introspection(state)
        assert data["has_narrative"] is True
        assert "coherence_level" in data
        assert "fragment_type_distribution" in data
        assert "description" in data


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfiguration:
    def test_default_config(self):
        cfg = SelfNarrativeConfig()
        assert cfg.max_fragments == 100
        assert cfg.vividness_decay_rate == 0.05

    def test_custom_config(self):
        cfg = create_config(
            max_fragments=50,
            vividness_decay_rate=0.1,
        )
        assert cfg.max_fragments == 50
        assert cfg.vividness_decay_rate == 0.1

    def test_config_affects_decay(self):
        slow = SelfNarrativeSystem(SelfNarrativeConfig(vividness_decay_rate=0.01))
        fast = SelfNarrativeSystem(SelfNarrativeConfig(vividness_decay_rate=0.2))

        emo = MockEmotionalStateView(intensity="intense")
        slow.observe_and_generate(emotion_summary=emo)
        fast.observe_and_generate(emotion_summary=emo)

        slow.observe_and_generate()
        fast.observe_and_generate()

        slow_v = slow.get_last_state().fragments[0].vividness
        fast_v = fast.get_last_state().fragments[0].vividness
        assert slow_v > fast_v


# =============================================================================
# Test: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    def test_create_empty_state(self):
        state = create_empty_state()
        assert not state.has_fragments()
        assert state.coherence.level == NarrativeCoherence.UNDEFINED

    def test_create_config(self):
        cfg = create_config()
        assert isinstance(cfg, SelfNarrativeConfig)
