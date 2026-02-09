"""
Tests for Episodic Memory (エピソード記憶 - 自伝的記憶)

These tests verify that the EpisodicMemorySystem:
1. Records episodes from STM, emotional state, and self-observation inputs
2. Classifies episodes into types (interaction/observation/emotional_event/etc.)
3. Computes importance from emotional intensity and self-difference
4. Builds emotional and self-observation companions
5. Links episodes by temporal proximity, topic overlap, and emotional similarity
6. Decays vividness naturally (importance/reference modulated)
7. Compresses old episodes into COMPOSITE entries
8. Searches episodes by topic, time, emotion, importance, and combined
9. Supports episode referencing (vividness recovery) and reinterpretation
10. Maintains STRICTLY NO IMPACT on decisions, goals, responsibility, or values
11. Does NOT generate goals from memory
12. Preserves read-only principle for all external systems
13. Does NOT modify values or beliefs
"""

import pytest
import time
import json
import os
import tempfile
from typing import Optional

from psyche.episodic_memory import (
    # Enums
    EpisodeType,
    ImportanceLevel,
    DecayState,
    EpisodeLinkType,
    SearchMode,
    # Data structures
    EmotionalCompanion,
    SelfObservationCompanion,
    EpisodeLink,
    EpisodeEntry,
    EpisodeStore,
    EpisodicMemoryConfig,
    # System
    EpisodicMemorySystem,
    # Helper functions
    determine_decay_state,
    classify_episode_type,
    compute_importance,
    compute_topic_overlap,
    compute_emotional_similarity,
    compute_temporal_proximity,
    generate_episode_summary,
    # Integration functions
    record_from_chain,
    generate_episodic_memory_tags,
    get_episodic_memory_summary,
    get_episodic_memory_for_introspection,
    # Verification
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    # Convenience / Persistence
    create_config,
    create_empty_store,
    create_system,
    save_episodic_memory,
    load_episodic_memory,
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
        self.spread = None


class MockStimulusEntry:
    """Mock StimulusEntry from short_term_memory."""

    def __init__(
        self,
        valence: float = 0.0,
        source_text: str = "",
        topics: Optional[list] = None,
        intent: str = "unknown",
    ):
        self.valence = valence
        self.source_text = source_text
        self.topics = topics if topics is not None else []
        self.intent = intent
        self.timestamp = time.time()


class MockShortTermMemory:
    """Mock ShortTermMemory from short_term_memory."""

    def __init__(self, entries: Optional[list] = None):
        self.entries = entries if entries is not None else []


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
        items: Optional[list] = None,
    ):
        self.has_awareness = has_awareness
        self.items = items if items is not None else []


class MockCoherenceState:
    """Mock coherence state."""

    def __init__(self, coherence_level: str = "undefined"):
        from psyche.identity_coherence import CoherenceLevel
        self.coherence_level = CoherenceLevel(coherence_level)


class MockNarrativeState:
    """Mock narrative state."""

    def __init__(self, trend: str = "undefined"):
        from psyche.self_narrative import NarrativeTrend
        self.trend = NarrativeTrend(trend)


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
    """Create a fresh EpisodicMemorySystem."""
    return EpisodicMemorySystem()


@pytest.fixture
def config():
    """Create a default config."""
    return EpisodicMemoryConfig()


@pytest.fixture
def fast_decay_system():
    """System with fast decay for testing lifecycle."""
    cfg = EpisodicMemoryConfig(
        base_decay_rate=0.2,
        compression_vividness_threshold=0.4,
        min_episodes_for_compression=2,
    )
    return EpisodicMemorySystem(config=cfg)


@pytest.fixture
def populated_system():
    """System with several episodes pre-recorded."""
    sys = EpisodicMemorySystem()
    for i in range(5):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(
                valence=0.3 * (i + 1),
                source_text=f"event_{i}",
                topics=[f"topic_{i}", "common"],
            )
        ])
        sys.record_episode(stm_entries=stm)
    return sys


# =============================================================================
# Test: Basic Enums
# =============================================================================

class TestEnums:
    def test_episode_type_values(self):
        assert EpisodeType.INTERACTION.value == "interaction"
        assert EpisodeType.OBSERVATION.value == "observation"
        assert EpisodeType.EMOTIONAL_EVENT.value == "emotional_event"
        assert EpisodeType.STATE_CHANGE.value == "state_change"
        assert EpisodeType.CONTEXT_SHIFT.value == "context_shift"
        assert EpisodeType.COMPOSITE.value == "composite"
        assert EpisodeType.UNDETERMINED.value == "undetermined"

    def test_importance_level_values(self):
        assert ImportanceLevel.TRIVIAL.value == "trivial"
        assert ImportanceLevel.MINOR.value == "minor"
        assert ImportanceLevel.MODERATE.value == "moderate"
        assert ImportanceLevel.NOTABLE.value == "notable"
        assert ImportanceLevel.SIGNIFICANT.value == "significant"

    def test_decay_state_values(self):
        assert DecayState.FRESH.value == "fresh"
        assert DecayState.CLEAR.value == "clear"
        assert DecayState.FADING.value == "fading"
        assert DecayState.DIM.value == "dim"
        assert DecayState.COMPRESSIBLE.value == "compressible"

    def test_episode_link_type_values(self):
        assert EpisodeLinkType.TEMPORAL_PROXIMITY.value == "temporal_proximity"
        assert EpisodeLinkType.TOPIC_OVERLAP.value == "topic_overlap"
        assert EpisodeLinkType.EMOTIONAL_SIMILARITY.value == "emotional_similarity"
        assert EpisodeLinkType.CAUSAL_SEQUENCE.value == "causal_sequence"
        assert EpisodeLinkType.THEMATIC.value == "thematic"

    def test_search_mode_values(self):
        assert SearchMode.BY_TOPIC.value == "by_topic"
        assert SearchMode.BY_TIME.value == "by_time"
        assert SearchMode.BY_EMOTION.value == "by_emotion"
        assert SearchMode.BY_IMPORTANCE.value == "by_importance"
        assert SearchMode.COMBINED.value == "combined"


# =============================================================================
# Test: EmotionalCompanion
# =============================================================================

class TestEmotionalCompanion:
    def test_create(self):
        ec = EmotionalCompanion(
            primary_emotion="joy",
            intensity_level=0.8,
            valence=0.7,
            harmony=0.9,
            emotion_description="Feeling joyful",
            coexisting_emotions=("excitement",),
        )
        assert ec.primary_emotion == "joy"
        assert ec.intensity_level == 0.8
        assert ec.valence == 0.7
        assert ec.harmony == 0.9
        assert ec.coexisting_emotions == ("excitement",)

    def test_frozen(self):
        ec = EmotionalCompanion("joy", 0.8, 0.7, 0.9, "", ())
        with pytest.raises(AttributeError):
            ec.primary_emotion = "sadness"


# =============================================================================
# Test: SelfObservationCompanion
# =============================================================================

class TestSelfObservationCompanion:
    def test_create(self):
        so = SelfObservationCompanion(
            has_difference=True,
            difference_magnitude="moderate",
            difference_nature="shifting",
            tendency_description="habit forming",
            has_strong_tendency=False,
            coherence_level="stable",
            narrative_trend="accumulating",
        )
        assert so.has_difference is True
        assert so.difference_magnitude == "moderate"

    def test_frozen(self):
        so = SelfObservationCompanion(False, "none", "stable", "", False, "", "")
        with pytest.raises(AttributeError):
            so.has_difference = True


# =============================================================================
# Test: EpisodeEntry
# =============================================================================

class TestEpisodeEntry:
    def _make_entry(self, vividness: float = 1.0, **kwargs) -> EpisodeEntry:
        defaults = {
            "episode_id": "test_ep",
            "episode_type": EpisodeType.OBSERVATION,
            "summary": "Test episode",
            "topics": ("topic_a",),
            "source_texts": ("text_a",),
            "timestamp": time.time(),
            "duration_estimate": 0.0,
            "emotional_companion": None,
            "self_observation_companion": None,
            "context_summary": "",
            "importance": ImportanceLevel.MINOR,
            "vividness": vividness,
            "reference_count": 0,
            "reinterpretation_count": 0,
            "is_compressed": False,
            "compressed_episode_ids": (),
        }
        defaults.update(kwargs)
        return EpisodeEntry(**defaults)

    def test_create(self):
        ep = self._make_entry()
        assert ep.episode_type == EpisodeType.OBSERVATION
        assert ep.vividness == 1.0

    def test_decay_state_fresh(self):
        assert self._make_entry(vividness=0.9).get_decay_state() == DecayState.FRESH

    def test_decay_state_clear(self):
        assert self._make_entry(vividness=0.7).get_decay_state() == DecayState.CLEAR

    def test_decay_state_fading(self):
        assert self._make_entry(vividness=0.5).get_decay_state() == DecayState.FADING

    def test_decay_state_dim(self):
        assert self._make_entry(vividness=0.3).get_decay_state() == DecayState.DIM

    def test_decay_state_compressible(self):
        assert self._make_entry(vividness=0.1).get_decay_state() == DecayState.COMPRESSIBLE

    def test_with_vividness(self):
        ep = self._make_entry(vividness=1.0)
        decayed = ep.with_vividness(0.5)
        assert decayed.vividness == 0.5
        assert ep.vividness == 1.0  # original unchanged

    def test_with_vividness_clamp_low(self):
        ep = self._make_entry(vividness=0.5)
        assert ep.with_vividness(-0.5).vividness == 0.0

    def test_with_vividness_clamp_high(self):
        ep = self._make_entry(vividness=0.5)
        assert ep.with_vividness(1.5).vividness == 1.0

    def test_with_reference(self):
        ep = self._make_entry()
        ref = ep.with_reference()
        assert ref.reference_count == 1
        assert ep.reference_count == 0

    def test_reinterpret(self):
        ep = self._make_entry()
        reinterpreted = ep.reinterpret("New summary", EpisodeType.INTERACTION)
        assert reinterpreted.summary == "New summary"
        assert reinterpreted.episode_type == EpisodeType.INTERACTION
        assert reinterpreted.reinterpretation_count == 1
        assert reinterpreted.episode_id == ep.episode_id

    def test_reinterpret_keep_type(self):
        ep = self._make_entry()
        reinterpreted = ep.reinterpret("New summary")
        assert reinterpreted.episode_type == ep.episode_type


# =============================================================================
# Test: EpisodeStore
# =============================================================================

class TestEpisodeStore:
    def test_empty_store(self):
        store = EpisodeStore.empty()
        assert not store.has_episodes()
        assert store.active_episode_count == 0
        assert store.total_episodes_recorded == 0

    def test_get_active_episodes(self):
        ep1 = EpisodeEntry(
            "ep1", EpisodeType.OBSERVATION, "s1", (), (), 0, 0,
            None, None, "", ImportanceLevel.MINOR, 0.5, 0, 0, False, (),
        )
        ep2 = EpisodeEntry(
            "ep2", EpisodeType.OBSERVATION, "s2", (), (), 0, 0,
            None, None, "", ImportanceLevel.MINOR, 0.0, 0, 0, False, (),
        )
        store = EpisodeStore(
            episodes=(ep1, ep2), links=(), total_episodes_recorded=2,
            total_compressions=0, average_vividness=0.25,
            active_episode_count=1, compressed_episode_count=0,
            timestamp=time.time(), description="test",
        )
        active = store.get_active_episodes()
        assert len(active) == 1
        assert active[0].episode_id == "ep1"

    def test_get_fresh_episodes(self):
        ep1 = EpisodeEntry(
            "ep1", EpisodeType.OBSERVATION, "s1", (), (), 0, 0,
            None, None, "", ImportanceLevel.MINOR, 0.9, 0, 0, False, (),
        )
        ep2 = EpisodeEntry(
            "ep2", EpisodeType.OBSERVATION, "s2", (), (), 0, 0,
            None, None, "", ImportanceLevel.MINOR, 0.3, 0, 0, False, (),
        )
        store = EpisodeStore(
            episodes=(ep1, ep2), links=(), total_episodes_recorded=2,
            total_compressions=0, average_vividness=0.6,
            active_episode_count=2, compressed_episode_count=0,
            timestamp=time.time(), description="test",
        )
        fresh = store.get_fresh_episodes()
        assert len(fresh) == 1
        assert fresh[0].episode_id == "ep1"

    def test_serialization_roundtrip(self):
        ec = EmotionalCompanion("joy", 0.8, 0.5, 0.9, "joyful", ("excitement",))
        so = SelfObservationCompanion(True, "moderate", "shifting", "desc", False, "stable", "accumulating")
        ep = EpisodeEntry(
            "ep1", EpisodeType.INTERACTION, "test summary", ("topic",), ("text",),
            time.time(), 5.0, ec, so, "context", ImportanceLevel.NOTABLE,
            0.8, 2, 1, False, (),
        )
        link = EpisodeLink("ep0", "ep1", EpisodeLinkType.TEMPORAL_PROXIMITY, 0.9, "linked")
        store = EpisodeStore(
            episodes=(ep,), links=(link,), total_episodes_recorded=1,
            total_compressions=0, average_vividness=0.8,
            active_episode_count=1, compressed_episode_count=0,
            timestamp=time.time(), description="test store",
        )
        data = store.to_dict()
        restored = EpisodeStore.from_dict(data)
        assert len(restored.episodes) == 1
        assert restored.episodes[0].episode_id == "ep1"
        assert restored.episodes[0].emotional_companion.primary_emotion == "joy"
        assert restored.episodes[0].self_observation_companion.has_difference is True
        assert len(restored.links) == 1
        assert restored.links[0].link_type == EpisodeLinkType.TEMPORAL_PROXIMITY


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestHelperFunctions:
    def test_determine_decay_state(self):
        assert determine_decay_state(0.9) == DecayState.FRESH
        assert determine_decay_state(0.7) == DecayState.CLEAR
        assert determine_decay_state(0.5) == DecayState.FADING
        assert determine_decay_state(0.3) == DecayState.DIM
        assert determine_decay_state(0.1) == DecayState.COMPRESSIBLE

    def test_classify_episode_type_interaction(self):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(intent="question"),
        ])
        result = classify_episode_type(stm_entries=stm)
        assert result == EpisodeType.INTERACTION

    def test_classify_episode_type_emotional(self):
        emotion = MockEmotionalStateView(intensity="intense")
        result = classify_episode_type(emotional_state=emotion)
        assert result == EpisodeType.EMOTIONAL_EVENT

    def test_classify_episode_type_state_change(self):
        diff = MockSelfDifferenceSummary(
            has_difference=True, magnitude="significant",
        )
        result = classify_episode_type(difference_summary=diff)
        assert result == EpisodeType.STATE_CHANGE

    def test_classify_episode_type_context_shift_string(self):
        result = classify_episode_type(external_context="new game started")
        assert result == EpisodeType.CONTEXT_SHIFT

    def test_classify_episode_type_context_shift_heavy(self):
        ctx = MockExternalContext(weight=0.8)
        result = classify_episode_type(external_context=ctx)
        assert result == EpisodeType.CONTEXT_SHIFT

    def test_classify_episode_type_observation(self):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(valence=0.1),
        ])
        result = classify_episode_type(stm_entries=stm)
        assert result == EpisodeType.OBSERVATION

    def test_classify_episode_type_undetermined(self):
        result = classify_episode_type()
        assert result == EpisodeType.UNDETERMINED

    def test_classify_high_valence_emotional_event(self):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(valence=0.8),
        ])
        result = classify_episode_type(stm_entries=stm)
        assert result == EpisodeType.EMOTIONAL_EVENT

    def test_compute_importance_trivial(self):
        result = compute_importance()
        assert result == ImportanceLevel.TRIVIAL

    def test_compute_importance_with_intense_emotion(self):
        emotion = MockEmotionalStateView(intensity="intense")
        result = compute_importance(emotional_state=emotion)
        assert result in (ImportanceLevel.MODERATE, ImportanceLevel.NOTABLE)

    def test_compute_importance_with_large_difference(self):
        diff = MockSelfDifferenceSummary(has_difference=True, magnitude="significant")
        result = compute_importance(difference_summary=diff)
        assert result in (ImportanceLevel.MODERATE, ImportanceLevel.NOTABLE)

    def test_compute_importance_combined(self):
        emotion = MockEmotionalStateView(intensity="overwhelming")
        diff = MockSelfDifferenceSummary(has_difference=True, magnitude="substantial")
        result = compute_importance(
            emotional_state=emotion, difference_summary=diff,
        )
        assert result == ImportanceLevel.SIGNIFICANT

    def test_compute_topic_overlap_identical(self):
        result = compute_topic_overlap(("a", "b"), ("a", "b"))
        assert result == 1.0

    def test_compute_topic_overlap_none(self):
        result = compute_topic_overlap(("a",), ("b",))
        assert result == 0.0

    def test_compute_topic_overlap_partial(self):
        result = compute_topic_overlap(("a", "b"), ("b", "c"))
        assert 0.3 <= result <= 0.4  # 1/3

    def test_compute_topic_overlap_empty(self):
        result = compute_topic_overlap((), ())
        assert result == 0.0

    def test_compute_topic_overlap_case_insensitive(self):
        result = compute_topic_overlap(("Hello",), ("hello",))
        assert result == 1.0

    def test_compute_emotional_similarity_identical(self):
        ec = EmotionalCompanion("joy", 0.8, 0.5, 0.9, "", ())
        result = compute_emotional_similarity(ec, ec)
        assert result == 1.0

    def test_compute_emotional_similarity_opposite(self):
        ec1 = EmotionalCompanion("joy", 1.0, 1.0, 0.9, "", ())
        ec2 = EmotionalCompanion("sad", 0.0, -1.0, 0.1, "", ())
        result = compute_emotional_similarity(ec1, ec2)
        assert result < 0.3

    def test_compute_emotional_similarity_none(self):
        ec = EmotionalCompanion("joy", 0.8, 0.5, 0.9, "", ())
        assert compute_emotional_similarity(ec, None) == 0.0
        assert compute_emotional_similarity(None, ec) == 0.0

    def test_compute_temporal_proximity_same_time(self):
        t = time.time()
        result = compute_temporal_proximity(t, t)
        assert result == 1.0

    def test_compute_temporal_proximity_outside_window(self):
        t = time.time()
        result = compute_temporal_proximity(t, t - 600, window=300)
        assert result == 0.0

    def test_compute_temporal_proximity_half_window(self):
        t = time.time()
        result = compute_temporal_proximity(t, t - 150, window=300)
        assert 0.49 <= result <= 0.51

    def test_generate_episode_summary_basic(self):
        result = generate_episode_summary(EpisodeType.OBSERVATION)
        assert "Observation recorded" in result

    def test_generate_episode_summary_with_stm(self):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(source_text="player attacked enemy"),
        ])
        result = generate_episode_summary(
            EpisodeType.INTERACTION, stm_entries=stm,
        )
        assert "player attacked enemy" in result


# =============================================================================
# Test: EpisodicMemorySystem - Recording
# =============================================================================

class TestRecording:
    def test_record_empty(self, system):
        store = system.record_episode()
        assert store.has_episodes()
        assert len(store.episodes) == 1

    def test_record_with_stm(self, system):
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(
                valence=0.5,
                source_text="hello world",
                topics=["greeting"],
            ),
        ])
        store = system.record_episode(stm_entries=stm)
        assert len(store.episodes) == 1
        ep = store.episodes[0]
        assert "greeting" in ep.topics

    def test_record_with_emotion(self, system):
        emotion = MockEmotionalStateView(
            intensity="intense", description="strong joy",
        )
        store = system.record_episode(emotional_state=emotion)
        ep = store.episodes[0]
        assert ep.emotional_companion is not None
        assert ep.emotional_companion.intensity_level == 0.8

    def test_record_with_self_observation(self, system):
        diff = MockSelfDifferenceSummary(
            has_difference=True, magnitude="noticeable", nature="shifting",
        )
        tendency = MockTendencyAwareness(
            has_awareness=True,
            items=[MockTendencyAwarenessItem(
                awareness_type="habit_forming",
                description="new pattern",
            )],
        )
        store = system.record_episode(
            difference_summary=diff,
            tendency_awareness=tendency,
        )
        ep = store.episodes[0]
        assert ep.self_observation_companion is not None
        assert ep.self_observation_companion.has_difference is True

    def test_record_with_coherence_and_narrative(self, system):
        coh = MockCoherenceState(coherence_level="stable")
        nar = MockNarrativeState(trend="accumulating")
        store = system.record_episode(
            coherence_state=coh,
            narrative_state=nar,
        )
        ep = store.episodes[0]
        assert ep.self_observation_companion is not None
        assert ep.self_observation_companion.coherence_level == "stable"
        assert ep.self_observation_companion.narrative_trend == "accumulating"

    def test_record_multiple(self, system):
        for _ in range(5):
            system.record_episode()
        store = system.get_store()
        assert len(store.episodes) == 5
        assert store.total_episodes_recorded == 5

    def test_total_recorded_increments(self, system):
        system.record_episode()
        system.record_episode()
        store = system.get_store()
        assert store.total_episodes_recorded == 2

    def test_record_with_context_string(self, system):
        store = system.record_episode(external_context="new game")
        ep = store.episodes[0]
        assert ep.context_summary == "new game"

    def test_record_with_context_object(self, system):
        ctx = MockExternalContext(weight=0.8, pace=0.3)
        store = system.record_episode(external_context=ctx)
        ep = store.episodes[0]
        assert "weight=0.80" in ep.context_summary


# =============================================================================
# Test: Link Generation
# =============================================================================

class TestLinkGeneration:
    def test_temporal_links_created(self, system):
        system.record_episode()
        system.record_episode()
        store = system.get_store()
        temporal_links = [
            l for l in store.links
            if l.link_type == EpisodeLinkType.TEMPORAL_PROXIMITY
        ]
        assert len(temporal_links) > 0

    def test_topic_overlap_links(self, system):
        stm1 = MockShortTermMemory(entries=[
            MockStimulusEntry(topics=["game", "action"]),
        ])
        stm2 = MockShortTermMemory(entries=[
            MockStimulusEntry(topics=["game", "adventure"]),
        ])
        system.record_episode(stm_entries=stm1)
        system.record_episode(stm_entries=stm2)
        store = system.get_store()
        topic_links = [
            l for l in store.links
            if l.link_type == EpisodeLinkType.TOPIC_OVERLAP
        ]
        assert len(topic_links) > 0

    def test_emotional_similarity_links(self, system):
        emotion1 = MockEmotionalStateView(
            intensity="intense", description="excited",
        )
        emotion2 = MockEmotionalStateView(
            intensity="intense", description="thrilled",
        )
        system.record_episode(emotional_state=emotion1)
        system.record_episode(emotional_state=emotion2)
        store = system.get_store()
        emotion_links = [
            l for l in store.links
            if l.link_type == EpisodeLinkType.EMOTIONAL_SIMILARITY
        ]
        assert len(emotion_links) > 0

    def test_max_links_per_episode(self):
        cfg = EpisodicMemoryConfig(max_links_per_episode=2)
        sys = EpisodicMemorySystem(config=cfg)
        for _ in range(10):
            sys.record_episode()
        store = sys.get_store()
        last_ep_id = store.episodes[-1].episode_id
        links_to_last = [
            l for l in store.links
            if l.to_episode_id == last_ep_id
        ]
        assert len(links_to_last) <= 2


# =============================================================================
# Test: Decay
# =============================================================================

class TestDecay:
    def test_decay_reduces_vividness(self, system):
        system.record_episode()
        store_before = system.get_store()
        assert store_before.episodes[0].vividness == 1.0
        store_after = system.decay_episodes()
        assert store_after.episodes[0].vividness < 1.0

    def test_decay_removes_zero_vividness(self, fast_decay_system):
        fast_decay_system.record_episode()
        # Decay many times to reach 0
        for _ in range(20):
            fast_decay_system.decay_episodes()
        store = fast_decay_system.get_store()
        assert len(store.episodes) == 0

    def test_important_episodes_decay_slower(self):
        cfg = EpisodicMemoryConfig(base_decay_rate=0.1)
        sys = EpisodicMemorySystem(config=cfg)
        # Record trivial and significant episodes
        sys.record_episode()  # likely trivial
        emotion = MockEmotionalStateView(intensity="overwhelming")
        diff = MockSelfDifferenceSummary(has_difference=True, magnitude="substantial")
        sys.record_episode(emotional_state=emotion, difference_summary=diff)

        # Decay once
        store = sys.decay_episodes()
        trivial = [
            e for e in store.episodes
            if e.importance == ImportanceLevel.TRIVIAL
        ]
        significant = [
            e for e in store.episodes
            if e.importance in (ImportanceLevel.NOTABLE, ImportanceLevel.SIGNIFICANT)
        ]
        if trivial and significant:
            # Important episodes should have higher vividness
            assert significant[0].vividness >= trivial[0].vividness

    def test_referenced_episodes_decay_slower(self, system):
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        system.reference_episode(ep_id)
        system.reference_episode(ep_id)
        system.record_episode()

        store = system.decay_episodes()
        referenced_ep = next(
            (e for e in store.episodes if e.episode_id == ep_id), None,
        )
        other = [e for e in store.episodes if e.episode_id != ep_id]
        if referenced_ep and other:
            assert referenced_ep.vividness >= other[0].vividness

    def test_decay_cleans_up_links(self, fast_decay_system):
        fast_decay_system.record_episode()
        fast_decay_system.record_episode()
        for _ in range(20):
            fast_decay_system.decay_episodes()
        store = fast_decay_system.get_store()
        assert len(store.links) == 0


# =============================================================================
# Test: Compression
# =============================================================================

class TestCompression:
    def test_compress_creates_composite(self):
        cfg = EpisodicMemoryConfig(
            compression_vividness_threshold=0.9,
            min_episodes_for_compression=2,
        )
        sys = EpisodicMemorySystem(config=cfg)
        # Record and reduce vividness
        for _ in range(3):
            sys.record_episode()
        # Manually set low vividness via decay
        for ep in sys._episodes:
            idx = sys._episodes.index(ep)
            sys._episodes[idx] = ep.with_vividness(0.5)

        store = sys.compress_old_episodes()
        composites = [e for e in store.episodes if e.is_compressed]
        assert len(composites) >= 1

    def test_compress_preserves_topics(self):
        cfg = EpisodicMemoryConfig(
            compression_vividness_threshold=0.9,
            min_episodes_for_compression=2,
        )
        sys = EpisodicMemorySystem(config=cfg)
        stm1 = MockShortTermMemory(entries=[
            MockStimulusEntry(topics=["alpha"]),
        ])
        stm2 = MockShortTermMemory(entries=[
            MockStimulusEntry(topics=["beta"]),
        ])
        sys.record_episode(stm_entries=stm1)
        sys.record_episode(stm_entries=stm2)
        for i, ep in enumerate(sys._episodes):
            sys._episodes[i] = ep.with_vividness(0.5)
        store = sys.compress_old_episodes()
        composites = [e for e in store.episodes if e.is_compressed]
        if composites:
            all_topics = composites[0].topics
            assert "alpha" in all_topics or "beta" in all_topics

    def test_no_compress_when_insufficient(self, system):
        system.record_episode()
        store = system.compress_old_episodes()
        composites = [e for e in store.episodes if e.is_compressed]
        assert len(composites) == 0

    def test_compress_increments_total(self):
        cfg = EpisodicMemoryConfig(
            compression_vividness_threshold=0.9,
            min_episodes_for_compression=2,
        )
        sys = EpisodicMemorySystem(config=cfg)
        for _ in range(4):
            sys.record_episode()
        for i, ep in enumerate(sys._episodes):
            sys._episodes[i] = ep.with_vividness(0.5)
        store = sys.compress_old_episodes()
        assert store.total_compressions >= 1


# =============================================================================
# Test: Search
# =============================================================================

class TestSearch:
    def test_search_by_topic(self, populated_system):
        results = populated_system.search_episodes(
            topics=["topic_0", "common"],
            mode=SearchMode.BY_TOPIC,
        )
        assert len(results) > 0
        for r in results:
            assert compute_topic_overlap(
                r.topics, ("topic_0", "common"),
            ) >= 0.2

    def test_search_by_time(self, populated_system):
        now = time.time()
        results = populated_system.search_episodes(
            time_start=now - 10,
            time_end=now + 10,
            mode=SearchMode.BY_TIME,
        )
        assert len(results) > 0

    def test_search_by_emotion(self, system):
        emotion = MockEmotionalStateView(
            intensity="intense", description="joy intense",
        )
        system.record_episode(emotional_state=emotion)
        results = system.search_episodes(
            emotion_label="joy",
            mode=SearchMode.BY_EMOTION,
        )
        assert len(results) > 0

    def test_search_by_importance(self, system):
        emotion = MockEmotionalStateView(intensity="overwhelming")
        diff = MockSelfDifferenceSummary(has_difference=True, magnitude="substantial")
        system.record_episode(emotional_state=emotion, difference_summary=diff)
        system.record_episode()  # trivial
        results = system.search_episodes(
            min_importance=ImportanceLevel.NOTABLE,
            mode=SearchMode.BY_IMPORTANCE,
        )
        for r in results:
            idx = list(ImportanceLevel).index(r.importance)
            min_idx = list(ImportanceLevel).index(ImportanceLevel.NOTABLE)
            assert idx >= min_idx

    def test_search_combined(self, populated_system):
        now = time.time()
        results = populated_system.search_episodes(
            topics=["common"],
            time_start=now - 10,
            mode=SearchMode.COMBINED,
        )
        assert len(results) > 0

    def test_search_max_results(self, populated_system):
        results = populated_system.search_episodes(max_results=2)
        assert len(results) <= 2

    def test_search_returns_sorted_by_vividness(self, populated_system):
        results = populated_system.search_episodes()
        for i in range(len(results) - 1):
            assert results[i].vividness >= results[i + 1].vividness

    def test_search_empty_returns_empty(self, system):
        results = system.search_episodes(
            topics=["nonexistent"],
            mode=SearchMode.BY_TOPIC,
        )
        assert len(results) == 0


# =============================================================================
# Test: Reference & Reinterpretation
# =============================================================================

class TestReferenceAndReinterpretation:
    def test_reference_boosts_vividness(self, system):
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        system.decay_episodes()
        vividness_before = next(
            e.vividness for e in system._episodes if e.episode_id == ep_id
        )
        system.reference_episode(ep_id)
        vividness_after = next(
            e.vividness for e in system._episodes if e.episode_id == ep_id
        )
        assert vividness_after > vividness_before

    def test_reference_increments_count(self, system):
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        system.reference_episode(ep_id)
        ep = next(e for e in system._episodes if e.episode_id == ep_id)
        assert ep.reference_count == 1

    def test_reference_nonexistent_is_noop(self, system):
        system.record_episode()
        system.reference_episode("nonexistent_id")
        # Should not raise

    def test_reinterpret_changes_summary(self, system):
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        system.reinterpret_episode(ep_id, "Reinterpreted summary")
        ep = next(e for e in system._episodes if e.episode_id == ep_id)
        assert ep.summary == "Reinterpreted summary"
        assert ep.reinterpretation_count == 1

    def test_reinterpret_changes_type(self, system):
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        system.reinterpret_episode(
            ep_id, "Now an interaction", EpisodeType.INTERACTION,
        )
        ep = next(e for e in system._episodes if e.episode_id == ep_id)
        assert ep.episode_type == EpisodeType.INTERACTION

    def test_reinterpret_nonexistent_is_noop(self, system):
        system.record_episode()
        system.reinterpret_episode("nonexistent_id", "new summary")
        # Should not raise


# =============================================================================
# Test: Integration Functions
# =============================================================================

class TestIntegrationFunctions:
    def test_record_from_chain(self, system):
        emotion = MockEmotionalStateView(
            intensity="moderate", description="calm",
        )
        stm = MockShortTermMemory(entries=[
            MockStimulusEntry(valence=0.3, topics=["test"]),
        ])
        store = record_from_chain(
            system,
            emotional_state=emotion,
            short_term_memory=stm,
        )
        assert store.has_episodes()

    def test_generate_tags_empty(self):
        store = create_empty_store()
        tags = generate_episodic_memory_tags(store)
        assert len(tags) == 1
        assert tags[0]["label"] == "no_episodes"

    def test_generate_tags_populated(self, populated_system):
        store = populated_system.get_store()
        tags = generate_episodic_memory_tags(store)
        assert len(tags) >= 3
        categories = [t["category"] for t in tags]
        assert "EPISODIC_MEMORY_COUNT" in categories
        assert "EPISODIC_MEMORY_VIVIDNESS" in categories
        assert "EPISODIC_MEMORY_INTEGRATED" in categories

    def test_generate_tags_with_scale(self, populated_system):
        store = populated_system.get_store()
        tags_1 = generate_episodic_memory_tags(store, scale=1.0)
        tags_2 = generate_episodic_memory_tags(store, scale=2.0)
        for t1, t2 in zip(tags_1, tags_2):
            assert abs(t2["weight"] - t1["weight"] * 2.0) < 0.01

    def test_get_summary(self, populated_system):
        store = populated_system.get_store()
        summary = get_episodic_memory_summary(store)
        assert "Episodic Memory State" in summary
        assert "Active episodes" in summary

    def test_get_for_introspection(self, populated_system):
        store = populated_system.get_store()
        data = get_episodic_memory_for_introspection(store)
        assert data["has_episodes"] is True
        assert data["total_episodes"] == 5
        assert "episode_type_distribution" in data
        assert "importance_distribution" in data

    def test_get_for_introspection_empty(self):
        store = create_empty_store()
        data = get_episodic_memory_for_introspection(store)
        assert data["has_episodes"] is False


# =============================================================================
# Test: Verification Functions
# =============================================================================

class TestVerification:
    def test_verify_no_decision_impact(self, populated_system):
        store = populated_system.get_store()
        assert verify_no_decision_impact(store) is True

    def test_verify_no_decision_impact_empty(self):
        store = create_empty_store()
        assert verify_no_decision_impact(store) is True

    def test_verify_no_goal_generation(self, system):
        assert verify_no_goal_generation(system) is True

    def test_verify_read_only_principle(self, system):
        assert verify_read_only_principle(system) is True

    def test_verify_no_value_modification(self, system):
        assert verify_no_value_modification(system) is True


# =============================================================================
# Test: Convenience & Persistence
# =============================================================================

class TestConvenience:
    def test_create_config(self):
        cfg = create_config(max_episodes=100, base_decay_rate=0.05)
        assert cfg.max_episodes == 100
        assert cfg.base_decay_rate == 0.05

    def test_create_empty_store(self):
        store = create_empty_store()
        assert not store.has_episodes()

    def test_create_system(self):
        sys = create_system()
        assert sys is not None
        store = sys.get_store()
        assert not store.has_episodes()

    def test_create_system_with_config(self):
        cfg = create_config(max_episodes=50)
        sys = create_system(config=cfg)
        assert sys._config.max_episodes == 50

    def test_save_and_load(self, populated_system):
        store = populated_system.get_store()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            filepath = f.name
        try:
            save_episodic_memory(store, filepath)
            loaded = load_episodic_memory(filepath)
            assert len(loaded.episodes) == len(store.episodes)
            assert loaded.total_episodes_recorded == store.total_episodes_recorded
        finally:
            os.unlink(filepath)


# =============================================================================
# Test: get_store / get_last_store
# =============================================================================

class TestStoreAccess:
    def test_get_store_returns_current(self, system):
        system.record_episode()
        store = system.get_store()
        assert store.has_episodes()

    def test_get_last_store_initially_none(self):
        sys = EpisodicMemorySystem()
        assert sys.get_last_store() is None

    def test_get_last_store_after_record(self, system):
        system.record_episode()
        last = system.get_last_store()
        assert last is not None
        assert last.has_episodes()


# =============================================================================
# Test: Config Defaults
# =============================================================================

class TestConfigDefaults:
    def test_default_config_values(self):
        cfg = EpisodicMemoryConfig()
        assert cfg.max_episodes == 200
        assert cfg.base_decay_rate == 0.02
        assert cfg.importance_decay_modifier == 0.5
        assert cfg.reference_decay_modifier == 0.3
        assert cfg.reference_vividness_boost == 0.15
        assert cfg.compression_vividness_threshold == 0.15
        assert cfg.min_episodes_for_compression == 3
        assert cfg.compression_result_vividness == 0.4
        assert cfg.default_max_results == 10
        assert cfg.max_links_per_episode == 5


# =============================================================================
# Test: Design Constraint Verification
# =============================================================================

class TestDesignConstraints:
    """Verify the system adheres to design document constraints."""

    def test_no_decision_method(self, system):
        """System must not have decision-making methods."""
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["decide", "choose", "select_action", "make_decision"]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), (
                    f"Found decision method: {method}"
                )

    def test_no_evaluation_method(self, system):
        """System must not evaluate success/failure or right/wrong."""
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = [
            "evaluate_success", "evaluate_failure",
            "judge_right", "judge_wrong",
            "assess_morality",
        ]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), (
                    f"Found evaluation method: {method}"
                )

    def test_episodes_have_no_score(self, populated_system):
        """Episodes must not have correctness scores."""
        store = populated_system.get_store()
        for ep in store.episodes:
            assert not hasattr(ep, "correctness_score")
            assert not hasattr(ep, "success_score")
            assert not hasattr(ep, "moral_score")

    def test_store_description_is_descriptive(self, populated_system):
        """Store description must be descriptive, not prescriptive."""
        store = populated_system.get_store()
        forbidden = [
            "should", "must", "need to", "correct",
            "wrong", "right", "fix",
        ]
        desc_lower = store.description.lower()
        for word in forbidden:
            assert word not in desc_lower, (
                f"Found prescriptive word '{word}' in description"
            )

    def test_episodes_preserve_interpretation_mutability(self, system):
        """Reinterpretation must be supported (解釈は固定しない)."""
        system.record_episode()
        ep_id = system.get_store().episodes[0].episode_id
        original_summary = system._episodes[0].summary
        system.reinterpret_episode(ep_id, "New interpretation")
        assert system._episodes[0].summary != original_summary
        assert system._episodes[0].reinterpretation_count == 1
