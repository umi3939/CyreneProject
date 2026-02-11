"""
Tests for psyche/emotional_memory_binding.py

感情記憶の紐づけ（Emotional Memory Binding）のテスト。
~114件のテストケース。
"""

import json
import os
import tempfile
import unittest

from psyche.emotional_memory_binding import (
    # Enums
    BindingSourceType,
    TraceAffinity,
    TraceStrength,
    TraceFreshness,
    # Level determination
    determine_freshness_level,
    determine_strength_level,
    # Key generation
    generate_memory_key,
    # Dataclasses
    EmotionalTrace,
    BindingLink,
    MemoryBinding,
    BindingStore,
    EmotionalMemoryBindingConfig,
    # Extraction functions
    extract_from_stm,
    extract_from_emotion_state,
    extract_from_recalled_memories,
    extract_from_episodes,
    # Computation functions
    compute_binding_strength,
    detect_trace_coexistence,
    compute_emotional_accompaniment,
    # System
    EmotionalMemoryBindingSystem,
    # Integration
    bind_from_chain,
    generate_binding_tags,
    get_binding_summary,
    get_binding_for_introspection,
    # Verification
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    verify_no_emotion_evaluation,
    # Convenience
    create_config,
    create_empty_store,
    create_system,
    save_binding_state,
    load_binding_state,
)


# =============================================================================
# Test Helpers
# =============================================================================

def _make_trace(
    emotion_label: str = "joy",
    intensity: float = 0.5,
    valence: float = 0.3,
    freshness: float = 0.8,
    reference_count: int = 0,
    affinity: TraceAffinity = TraceAffinity.CONCURRENT,
) -> EmotionalTrace:
    return EmotionalTrace(
        trace_id="trace_test_01",
        emotion_label=emotion_label,
        intensity=intensity,
        valence=valence,
        freshness=freshness,
        reference_count=reference_count,
        affinity=affinity,
        timestamp="12345",
        source_description="test source",
    )


def _make_binding(
    memory_key: str = "mem_key_01",
    traces: tuple = None,
    freshness: float = 0.8,
    reference_count: int = 0,
) -> MemoryBinding:
    if traces is None:
        traces = (_make_trace(),)
    return MemoryBinding(
        binding_id="bind_test_01",
        memory_key=memory_key,
        memory_summary="test memory summary",
        traces=traces,
        binding_links=(),
        freshness=freshness,
        reference_count=reference_count,
        creation_timestamp="12345",
        last_reference_timestamp="12345",
        revision_count=0,
        undetermined_aspects=("test_aspect",),
    )


class _MockSTMEntry:
    def __init__(self, source_text, emotion_label, raw_intensity, valence=0.0):
        self.source_text = source_text
        self.emotion_label = emotion_label
        self.raw_intensity = raw_intensity
        self.valence = valence


class _MockSTM:
    def __init__(self, entries=None):
        self.entries = entries or []


class _MockEmotionVector:
    def __init__(self, **kwargs):
        self.joy = kwargs.get("joy", 0.0)
        self.anger = kwargs.get("anger", 0.0)
        self.sorrow = kwargs.get("sorrow", 0.0)
        self.fear = kwargs.get("fear", 0.0)
        self.surprise = kwargs.get("surprise", 0.0)
        self.love = kwargs.get("love", 0.0)
        self.fun = kwargs.get("fun", 0.0)


class _MockMood:
    def __init__(self, valence=0.0, arousal=0.5):
        self.valence = valence
        self.arousal = arousal


class _MockEmotionalCompanion:
    def __init__(self, primary_emotion="joy", intensity_level=0.5, valence=0.3, coexisting_emotions=()):
        self.primary_emotion = primary_emotion
        self.intensity_level = intensity_level
        self.valence = valence
        self.coexisting_emotions = coexisting_emotions


class _MockEpisodeEntry:
    def __init__(self, episode_id="ep1", summary="test episode", vividness=0.7,
                 emotional_companion=None):
        self.episode_id = episode_id
        self.summary = summary
        self.vividness = vividness
        self.emotional_companion = emotional_companion


class _MockEpisodeStore:
    def __init__(self, episodes=None):
        self.episodes = episodes or []


# =============================================================================
# TestEnums
# =============================================================================

class TestEnums(unittest.TestCase):
    """Test all Enum values."""

    def test_binding_source_type_values(self):
        self.assertEqual(BindingSourceType.SHORT_TERM_MEMORY.value, "short_term_memory")
        self.assertEqual(BindingSourceType.EMOTION_STATE.value, "emotion_state")
        self.assertEqual(BindingSourceType.LONG_TERM_RECALL.value, "long_term_recall")
        self.assertEqual(BindingSourceType.EPISODIC.value, "episodic")
        self.assertEqual(BindingSourceType.MIXED.value, "mixed")

    def test_trace_affinity_values(self):
        self.assertEqual(TraceAffinity.CONCURRENT.value, "concurrent")
        self.assertEqual(TraceAffinity.REACTIVATED.value, "reactivated")
        self.assertEqual(TraceAffinity.ACCUMULATED.value, "accumulated")
        self.assertEqual(TraceAffinity.COMPOSITE.value, "composite")
        self.assertEqual(TraceAffinity.UNDEFINED.value, "undefined")

    def test_trace_strength_values(self):
        self.assertEqual(TraceStrength.STRONG.value, "strong")
        self.assertEqual(TraceStrength.MODERATE.value, "moderate")
        self.assertEqual(TraceStrength.WEAK.value, "weak")
        self.assertEqual(TraceStrength.FAINT.value, "faint")
        self.assertEqual(TraceStrength.UNDEFINED.value, "undefined")

    def test_trace_freshness_values(self):
        self.assertEqual(TraceFreshness.FRESH.value, "fresh")
        self.assertEqual(TraceFreshness.RECENT.value, "recent")
        self.assertEqual(TraceFreshness.AGING.value, "aging")
        self.assertEqual(TraceFreshness.STALE.value, "stale")
        self.assertEqual(TraceFreshness.FADED.value, "faded")

    def test_all_enums_have_expected_count(self):
        self.assertEqual(len(BindingSourceType), 5)
        self.assertEqual(len(TraceAffinity), 5)
        self.assertEqual(len(TraceStrength), 5)
        self.assertEqual(len(TraceFreshness), 5)


# =============================================================================
# TestDetermineFreshnessLevel
# =============================================================================

class TestDetermineFreshnessLevel(unittest.TestCase):
    def test_fresh(self):
        self.assertEqual(determine_freshness_level(0.9), TraceFreshness.FRESH)

    def test_recent(self):
        self.assertEqual(determine_freshness_level(0.65), TraceFreshness.RECENT)

    def test_aging(self):
        self.assertEqual(determine_freshness_level(0.45), TraceFreshness.AGING)

    def test_stale(self):
        self.assertEqual(determine_freshness_level(0.2), TraceFreshness.STALE)

    def test_faded(self):
        self.assertEqual(determine_freshness_level(0.05), TraceFreshness.FADED)


# =============================================================================
# TestDetermineStrengthLevel
# =============================================================================

class TestDetermineStrengthLevel(unittest.TestCase):
    def test_strong(self):
        self.assertEqual(determine_strength_level(0.8), TraceStrength.STRONG)

    def test_moderate(self):
        self.assertEqual(determine_strength_level(0.5), TraceStrength.MODERATE)

    def test_weak(self):
        self.assertEqual(determine_strength_level(0.25), TraceStrength.WEAK)

    def test_faint(self):
        self.assertEqual(determine_strength_level(0.08), TraceStrength.FAINT)

    def test_undefined(self):
        self.assertEqual(determine_strength_level(0.02), TraceStrength.UNDEFINED)


# =============================================================================
# TestEmotionalTrace
# =============================================================================

class TestEmotionalTrace(unittest.TestCase):
    def test_creation(self):
        t = _make_trace()
        self.assertEqual(t.emotion_label, "joy")
        self.assertEqual(t.intensity, 0.5)

    def test_get_freshness_level(self):
        t = _make_trace(freshness=0.9)
        self.assertEqual(t.get_freshness_level(), TraceFreshness.FRESH)

    def test_with_freshness(self):
        t = _make_trace(freshness=0.5)
        t2 = t.with_freshness(0.9)
        self.assertAlmostEqual(t2.freshness, 0.9)
        self.assertAlmostEqual(t.freshness, 0.5)  # immutable

    def test_with_freshness_clamp(self):
        t = _make_trace()
        t2 = t.with_freshness(1.5)
        self.assertAlmostEqual(t2.freshness, 1.0)
        t3 = t.with_freshness(-0.5)
        self.assertAlmostEqual(t3.freshness, 0.0)

    def test_with_intensity(self):
        t = _make_trace(intensity=0.3)
        t2 = t.with_intensity(0.7)
        self.assertAlmostEqual(t2.intensity, 0.7)

    def test_with_reference(self):
        t = _make_trace(reference_count=2)
        t2 = t.with_reference()
        self.assertEqual(t2.reference_count, 3)
        self.assertEqual(t.reference_count, 2)  # immutable

    def test_reattach(self):
        t = _make_trace(affinity=TraceAffinity.CONCURRENT)
        t2 = t.reattach(TraceAffinity.REACTIVATED)
        self.assertEqual(t2.affinity, TraceAffinity.REACTIVATED)
        self.assertEqual(t.affinity, TraceAffinity.CONCURRENT)

    def test_frozen(self):
        t = _make_trace()
        with self.assertRaises(AttributeError):
            t.intensity = 0.9


# =============================================================================
# TestBindingLink
# =============================================================================

class TestBindingLink(unittest.TestCase):
    def test_creation(self):
        bl = BindingLink(
            link_id="link01",
            binding_id="bind01",
            source_type=BindingSourceType.SHORT_TERM_MEMORY,
            source_description="test",
            contribution=0.8,
        )
        self.assertEqual(bl.link_id, "link01")
        self.assertEqual(bl.contribution, 0.8)

    def test_frozen(self):
        bl = BindingLink(
            link_id="link01",
            binding_id="bind01",
            source_type=BindingSourceType.SHORT_TERM_MEMORY,
            source_description="test",
            contribution=0.8,
        )
        with self.assertRaises(AttributeError):
            bl.contribution = 0.5


# =============================================================================
# TestMemoryBinding
# =============================================================================

class TestMemoryBinding(unittest.TestCase):
    def test_creation(self):
        b = _make_binding()
        self.assertEqual(b.memory_key, "mem_key_01")
        self.assertEqual(len(b.traces), 1)

    def test_with_freshness(self):
        b = _make_binding(freshness=0.5)
        b2 = b.with_freshness(0.9)
        self.assertAlmostEqual(b2.freshness, 0.9)
        self.assertAlmostEqual(b.freshness, 0.5)

    def test_with_reference(self):
        b = _make_binding(reference_count=1)
        b2 = b.with_reference()
        self.assertEqual(b2.reference_count, 2)

    def test_with_traces(self):
        b = _make_binding()
        new_trace = _make_trace(emotion_label="anger")
        b2 = b.with_traces((new_trace,))
        self.assertEqual(len(b2.traces), 1)
        self.assertEqual(b2.traces[0].emotion_label, "anger")

    def test_revise_summary(self):
        b = _make_binding()
        b2 = b.revise_summary("new summary")
        self.assertEqual(b2.memory_summary, "new summary")
        self.assertEqual(b2.revision_count, 1)

    def test_with_added_trace(self):
        b = _make_binding()
        new_trace = _make_trace(emotion_label="sorrow")
        b2 = b.with_added_trace(new_trace)
        self.assertEqual(len(b2.traces), 2)

    def test_frozen(self):
        b = _make_binding()
        with self.assertRaises(AttributeError):
            b.freshness = 0.1

    def test_freshness_clamp(self):
        b = _make_binding()
        b2 = b.with_freshness(1.5)
        self.assertAlmostEqual(b2.freshness, 1.0)


# =============================================================================
# TestBindingStore
# =============================================================================

class TestBindingStore(unittest.TestCase):
    def _make_store(self, bindings=None):
        return BindingStore(
            bindings=bindings or (),
            binding_links=(),
            total_bindings_created=0,
            total_traces_created=0,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.0,
            average_trace_count=0.0,
            active_binding_count=0,
            timestamp="12345",
            description="test",
        )

    def test_empty_store(self):
        s = self._make_store()
        self.assertFalse(s.has_bindings())

    def test_has_bindings(self):
        b = _make_binding()
        s = self._make_store(bindings=(b,))
        self.assertTrue(s.has_bindings())

    def test_get_active_bindings(self):
        b1 = _make_binding(freshness=0.8)
        b2 = _make_binding(freshness=0.1)
        s = self._make_store(bindings=(b1, b2))
        active = s.get_active_bindings()
        self.assertEqual(len(active), 1)

    def test_get_bindings_for_memory(self):
        b1 = _make_binding(memory_key="key1")
        b2 = _make_binding(memory_key="key2")
        s = self._make_store(bindings=(b1, b2))
        result = s.get_bindings_for_memory("key1")
        self.assertEqual(len(result), 1)

    def test_to_dict(self):
        b = _make_binding()
        s = self._make_store(bindings=(b,))
        d = s.to_dict()
        self.assertIn("bindings", d)
        self.assertEqual(len(d["bindings"]), 1)

    def test_from_dict(self):
        b = _make_binding()
        s = self._make_store(bindings=(b,))
        d = s.to_dict()
        s2 = BindingStore.from_dict(d)
        self.assertEqual(len(s2.bindings), 1)

    def test_roundtrip_serialization(self):
        b = _make_binding()
        s = self._make_store(bindings=(b,))
        d = s.to_dict()
        s2 = BindingStore.from_dict(d)
        self.assertEqual(s2.bindings[0].memory_key, b.memory_key)
        self.assertEqual(len(s2.bindings[0].traces), len(b.traces))


# =============================================================================
# TestEmotionalMemoryBindingConfig
# =============================================================================

class TestEmotionalMemoryBindingConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = EmotionalMemoryBindingConfig()
        self.assertEqual(cfg.max_bindings, 200)
        self.assertEqual(cfg.max_traces_per_binding, 7)
        self.assertAlmostEqual(cfg.base_decay_rate, 0.02)

    def test_custom_values(self):
        cfg = EmotionalMemoryBindingConfig(max_bindings=100, base_decay_rate=0.05)
        self.assertEqual(cfg.max_bindings, 100)
        self.assertAlmostEqual(cfg.base_decay_rate, 0.05)


# =============================================================================
# TestExtractFromSTM
# =============================================================================

class TestExtractFromSTM(unittest.TestCase):
    def test_none_input(self):
        result = extract_from_stm(None)
        self.assertEqual(result, [])

    def test_object_input(self):
        stm = _MockSTM(entries=[
            _MockSTMEntry("hello world", "happy", 0.7, 0.5),
            _MockSTMEntry("goodbye", "sad", 0.6, -0.3),
        ])
        result = extract_from_stm(stm)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], "joy")  # happy → joy
        self.assertEqual(result[1][1], "sorrow")  # sad → sorrow

    def test_dict_input(self):
        stm = {
            "entries": [
                {"source_text": "test text", "emotion_label": "angry", "raw_intensity": 0.8, "valence": -0.5},
            ]
        }
        result = extract_from_stm(stm)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "anger")

    def test_neutral_skip(self):
        stm = _MockSTM(entries=[
            _MockSTMEntry("test", "neutral", 0.5),
        ])
        result = extract_from_stm(stm)
        self.assertEqual(len(result), 0)

    def test_low_intensity_skip(self):
        stm = _MockSTM(entries=[
            _MockSTMEntry("test", "happy", 0.05),
        ])
        result = extract_from_stm(stm)
        self.assertEqual(len(result), 0)


# =============================================================================
# TestExtractFromEmotionState
# =============================================================================

class TestExtractFromEmotionState(unittest.TestCase):
    def test_none_input(self):
        result = extract_from_emotion_state(None, None)
        self.assertEqual(result, [])

    def test_object_input(self):
        emotion = _MockEmotionVector(joy=0.8, anger=0.3)
        mood = _MockMood(valence=0.4)
        result = extract_from_emotion_state(emotion, mood)
        # joy >= 0.15 and anger >= 0.15
        labels = [r[1] for r in result]
        self.assertIn("joy", labels)
        self.assertIn("anger", labels)

    def test_dict_input(self):
        emotion = {"joy": 0.6, "fear": 0.2}
        mood = {"valence": -0.1}
        result = extract_from_emotion_state(emotion, mood)
        labels = [r[1] for r in result]
        self.assertIn("joy", labels)
        self.assertIn("fear", labels)

    def test_low_emotions_skip(self):
        emotion = _MockEmotionVector(joy=0.05, anger=0.1)
        result = extract_from_emotion_state(emotion, None)
        # joy < 0.15, anger < 0.15 → nothing
        self.assertEqual(len(result), 0)


# =============================================================================
# TestExtractFromRecalledMemories
# =============================================================================

class TestExtractFromRecalledMemories(unittest.TestCase):
    def test_none_input(self):
        result = extract_from_recalled_memories(None)
        self.assertEqual(result, [])

    def test_list_input(self):
        memories = [
            {"summary": "A happy event", "keywords": ["joy", "celebration"]},
            {"summary": "A sad moment"},
        ]
        result = extract_from_recalled_memories(memories)
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        result = extract_from_recalled_memories([])
        self.assertEqual(result, [])

    def test_positive_keyword_mapping(self):
        memories = [
            {"summary": "An angry encounter", "keywords": ["anger"]},
        ]
        result = extract_from_recalled_memories(memories)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "anger")


# =============================================================================
# TestExtractFromEpisodes
# =============================================================================

class TestExtractFromEpisodes(unittest.TestCase):
    def test_none_input(self):
        result = extract_from_episodes(None)
        self.assertEqual(result, [])

    def test_object_input(self):
        companion = _MockEmotionalCompanion(
            primary_emotion="joy", intensity_level=0.7, valence=0.5
        )
        entry = _MockEpisodeEntry(summary="test episode", vividness=0.8,
                                   emotional_companion=companion)
        store = _MockEpisodeStore(episodes=[entry])
        result = extract_from_episodes(store)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0][1], "joy")

    def test_dict_input(self):
        episodes = {
            "episodes": [{
                "episode_id": "ep1",
                "summary": "test dict episode",
                "vividness": 0.7,
                "emotional_companion": {
                    "primary_emotion": "fear",
                    "intensity_level": 0.6,
                    "valence": -0.4,
                    "coexisting_emotions": [],
                },
            }]
        }
        result = extract_from_episodes(episodes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "fear")

    def test_low_vividness_skip(self):
        companion = _MockEmotionalCompanion()
        entry = _MockEpisodeEntry(vividness=0.1, emotional_companion=companion)
        store = _MockEpisodeStore(episodes=[entry])
        result = extract_from_episodes(store)
        self.assertEqual(len(result), 0)

    def test_coexisting_emotions(self):
        companion = _MockEmotionalCompanion(
            primary_emotion="joy", intensity_level=0.8, valence=0.5,
            coexisting_emotions=("surprise", "love")
        )
        entry = _MockEpisodeEntry(summary="episode with coex", vividness=0.8,
                                   emotional_companion=companion)
        store = _MockEpisodeStore(episodes=[entry])
        result = extract_from_episodes(store)
        # 1 primary + 2 coexisting = 3
        self.assertEqual(len(result), 3)
        labels = [r[1] for r in result]
        self.assertIn("joy", labels)
        self.assertIn("surprise", labels)
        self.assertIn("love", labels)


# =============================================================================
# TestComputeBindingStrength
# =============================================================================

class TestComputeBindingStrength(unittest.TestCase):
    def test_empty_traces(self):
        result = compute_binding_strength(())
        self.assertAlmostEqual(result, 0.0)

    def test_single_trace(self):
        t = _make_trace(intensity=0.5)
        result = compute_binding_strength((t,))
        self.assertAlmostEqual(result, 0.5)

    def test_multiple_traces(self):
        t1 = _make_trace(intensity=0.8)
        t2 = _make_trace(intensity=0.4)
        result = compute_binding_strength((t1, t2))
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


# =============================================================================
# TestComputeEmotionalAccompaniment
# =============================================================================

class TestComputeEmotionalAccompaniment(unittest.TestCase):
    def test_empty_binding(self):
        b = _make_binding(traces=())
        result = compute_emotional_accompaniment(b)
        self.assertEqual(result, {})

    def test_single_trace(self):
        t = _make_trace(emotion_label="joy", intensity=0.8, freshness=0.6)
        b = _make_binding(traces=(t,))
        result = compute_emotional_accompaniment(b)
        self.assertIn("joy", result)
        self.assertAlmostEqual(result["joy"], 0.8 * 0.6)

    def test_multiple_traces_same_label_max(self):
        t1 = _make_trace(emotion_label="joy", intensity=0.5, freshness=0.8)
        t2 = EmotionalTrace(
            trace_id="trace_02", emotion_label="joy", intensity=0.9,
            freshness=0.6, valence=0.3, reference_count=0,
            affinity=TraceAffinity.CONCURRENT, timestamp="12345",
            source_description="test",
        )
        b = _make_binding(traces=(t1, t2))
        result = compute_emotional_accompaniment(b)
        # max(0.5*0.8, 0.9*0.6) = max(0.4, 0.54) = 0.54
        self.assertAlmostEqual(result["joy"], 0.54)


# =============================================================================
# TestGenerateMemoryKey
# =============================================================================

class TestGenerateMemoryKey(unittest.TestCase):
    def test_consistency(self):
        key1 = generate_memory_key("hello world")
        key2 = generate_memory_key("hello world")
        self.assertEqual(key1, key2)

    def test_different_texts(self):
        key1 = generate_memory_key("hello")
        key2 = generate_memory_key("world")
        self.assertNotEqual(key1, key2)


# =============================================================================
# TestEmotionalMemoryBindingSystem
# =============================================================================

class TestEmotionalMemoryBindingSystem(unittest.TestCase):
    def test_creation(self):
        system = EmotionalMemoryBindingSystem()
        store = system.get_store()
        self.assertFalse(store.has_bindings())

    def test_bind_from_stm(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("hello world", "happy", 0.7, 0.5),
        ])
        store = system.bind_emotions(stm=stm)
        self.assertTrue(store.has_bindings())

    def test_bind_from_emotion_state(self):
        system = EmotionalMemoryBindingSystem()
        emotion = _MockEmotionVector(joy=0.8)
        store = system.bind_emotions(emotion_state=emotion)
        self.assertTrue(store.has_bindings())

    def test_decay_bindings(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("test decay", "happy", 0.5),
        ])
        system.bind_emotions(stm=stm)
        store = system.decay_bindings()
        # After one decay, should still exist
        self.assertTrue(store.has_bindings())

    def test_reference_binding(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("reference test", "happy", 0.7),
        ])
        system.bind_emotions(stm=stm)
        memory_key = generate_memory_key("reference test")
        system.reference_binding(memory_key)
        # After reference, check store
        store = system.get_store()
        binding = store.get_bindings_for_memory(memory_key)
        self.assertEqual(len(binding), 1)
        self.assertGreater(binding[0].reference_count, 0)

    def test_get_emotional_accompaniment(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("accompaniment test", "happy", 0.8),
        ])
        system.bind_emotions(stm=stm)
        memory_key = generate_memory_key("accompaniment test")
        accompaniment = system.get_emotional_accompaniment(memory_key)
        self.assertIn("joy", accompaniment)

    def test_accompaniment_nonexistent_key(self):
        system = EmotionalMemoryBindingSystem()
        result = system.get_emotional_accompaniment("nonexistent")
        self.assertEqual(result, {})

    def test_revise_binding(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("revise test", "happy", 0.6),
        ])
        system.bind_emotions(stm=stm)
        memory_key = generate_memory_key("revise test")
        system.revise_binding(memory_key, "revised summary")
        store = system.get_store()
        bindings = store.get_bindings_for_memory(memory_key)
        self.assertEqual(bindings[0].memory_summary, "revised summary")

    def test_get_active_bindings(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("active test 1", "happy", 0.7),
            _MockSTMEntry("active test 2", "angry", 0.8),
        ])
        system.bind_emotions(stm=stm)
        active = system.get_active_bindings()
        self.assertGreater(len(active), 0)

    def test_merge_traces_existing_binding(self):
        system = EmotionalMemoryBindingSystem()
        # First binding with "happy"
        stm1 = _MockSTM(entries=[
            _MockSTMEntry("merge test", "happy", 0.5),
        ])
        system.bind_emotions(stm=stm1)

        # Second call with same text but different emotion
        stm2 = _MockSTM(entries=[
            _MockSTMEntry("merge test", "angry", 0.6),
        ])
        system.bind_emotions(stm=stm2)

        memory_key = generate_memory_key("merge test")
        store = system.get_store()
        bindings = store.get_bindings_for_memory(memory_key)
        self.assertEqual(len(bindings), 1)
        # Should have both traces
        labels = [t.emotion_label for t in bindings[0].traces]
        self.assertIn("joy", labels)
        self.assertIn("anger", labels)

    def test_capacity_enforcement(self):
        config = EmotionalMemoryBindingConfig(max_bindings=3)
        system = EmotionalMemoryBindingSystem(config=config)
        for i in range(5):
            stm = _MockSTM(entries=[
                _MockSTMEntry(f"capacity test {i}", "happy", 0.7),
            ])
            system.bind_emotions(stm=stm)
        store = system.get_store()
        self.assertLessEqual(len(store.bindings), 3)

    def test_get_last_store(self):
        system = EmotionalMemoryBindingSystem()
        self.assertIsNone(system.get_last_store())
        stm = _MockSTM(entries=[
            _MockSTMEntry("last store test", "happy", 0.5),
        ])
        system.bind_emotions(stm=stm)
        last = system.get_last_store()
        self.assertIsNotNone(last)

    def test_multiple_sources(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("multi source", "happy", 0.6),
        ])
        emotion = _MockEmotionVector(anger=0.5)
        store = system.bind_emotions(stm=stm, emotion_state=emotion)
        self.assertTrue(store.has_bindings())

    def test_all_none_input(self):
        system = EmotionalMemoryBindingSystem()
        store = system.bind_emotions()
        self.assertFalse(store.has_bindings())

    def test_custom_config(self):
        config = EmotionalMemoryBindingConfig(
            base_decay_rate=0.1,
            trace_decay_rate=0.05,
        )
        system = EmotionalMemoryBindingSystem(config=config)
        stm = _MockSTM(entries=[
            _MockSTMEntry("config test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        self.assertTrue(store.has_bindings())


# =============================================================================
# TestBindFromChain
# =============================================================================

class TestBindFromChain(unittest.TestCase):
    def test_basic_chain(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("chain test", "happy", 0.7),
        ])
        store = bind_from_chain(system, stm=stm)
        self.assertTrue(store.has_bindings())

    def test_chain_with_all_sources(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("full chain", "surprised", 0.5),
        ])
        emotion = _MockEmotionVector(love=0.6)
        mood = _MockMood(valence=0.3)
        memories = [{"summary": "old memory", "keywords": ["joy"]}]
        companion = _MockEmotionalCompanion(primary_emotion="fear", intensity_level=0.4)
        ep_entry = _MockEpisodeEntry(summary="ep", vividness=0.6, emotional_companion=companion)
        episodes = _MockEpisodeStore(episodes=[ep_entry])

        store = bind_from_chain(system, stm=stm, emotion=emotion,
                                mood=mood, memories=memories, episodes=episodes)
        self.assertTrue(store.has_bindings())


# =============================================================================
# TestGenerateBindingTags
# =============================================================================

class TestGenerateBindingTags(unittest.TestCase):
    def test_none_store(self):
        tags = generate_binding_tags(None)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["category"], "EMOTIONAL_BINDING_COUNT")

    def test_with_bindings(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("tag test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        tags = generate_binding_tags(store)
        self.assertGreater(len(tags), 1)
        categories = [t["category"] for t in tags]
        self.assertIn("EMOTIONAL_BINDING_COUNT", categories)

    def test_scale_factor(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("scale test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        tags_1 = generate_binding_tags(store, scale=1.0)
        tags_2 = generate_binding_tags(store, scale=2.0)
        # Weights should differ
        w1 = tags_1[0]["weight"]
        w2 = tags_2[0]["weight"]
        self.assertAlmostEqual(w2, w1 * 2.0)

    def test_categories_present(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("cat test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        tags = generate_binding_tags(store)
        categories = {t["category"] for t in tags}
        self.assertIn("EMOTIONAL_BINDING_COUNT", categories)
        self.assertIn("EMOTIONAL_BINDING_FRESHNESS", categories)
        self.assertIn("EMOTIONAL_BINDING_RICHNESS", categories)


# =============================================================================
# TestGetBindingSummary
# =============================================================================

class TestGetBindingSummary(unittest.TestCase):
    def test_none_store(self):
        summary = get_binding_summary(None)
        self.assertIn("No bindings", summary)

    def test_with_bindings(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("summary test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        summary = get_binding_summary(store)
        self.assertIn("Emotional Memory Binding State", summary)
        self.assertIn("Active bindings", summary)


# =============================================================================
# TestGetBindingForIntrospection
# =============================================================================

class TestGetBindingForIntrospection(unittest.TestCase):
    def test_none_store(self):
        result = get_binding_for_introspection(None)
        self.assertFalse(result["has_bindings"])
        self.assertEqual(result["total_bindings"], 0)

    def test_with_bindings(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("intro test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        result = get_binding_for_introspection(store)
        self.assertTrue(result["has_bindings"])
        self.assertGreater(result["total_bindings"], 0)
        self.assertIn("emotion_distribution", result)
        self.assertIn("dominant_emotion", result)


# =============================================================================
# TestVerification
# =============================================================================

class TestVerification(unittest.TestCase):
    def test_no_decision_impact(self):
        store = create_empty_store()
        self.assertTrue(verify_no_decision_impact(store))

    def test_no_goal_generation(self):
        system = EmotionalMemoryBindingSystem()
        self.assertTrue(verify_no_goal_generation(system))

    def test_read_only_principle(self):
        system = EmotionalMemoryBindingSystem()
        self.assertTrue(verify_read_only_principle(system))

    def test_no_value_modification(self):
        system = EmotionalMemoryBindingSystem()
        self.assertTrue(verify_no_value_modification(system))

    def test_no_emotion_evaluation(self):
        system = EmotionalMemoryBindingSystem()
        self.assertTrue(verify_no_emotion_evaluation(system))

    def test_store_with_bindings_no_decision_impact(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("verify test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)
        self.assertTrue(verify_no_decision_impact(store))


# =============================================================================
# TestConvenience
# =============================================================================

class TestConvenience(unittest.TestCase):
    def test_create_config(self):
        cfg = create_config(max_bindings=50)
        self.assertEqual(cfg.max_bindings, 50)

    def test_create_empty_store(self):
        store = create_empty_store()
        self.assertFalse(store.has_bindings())

    def test_create_system(self):
        system = create_system()
        self.assertIsNotNone(system)
        store = system.get_store()
        self.assertFalse(store.has_bindings())

    def test_create_system_with_config(self):
        cfg = create_config(max_bindings=10)
        system = create_system(config=cfg)
        self.assertIsNotNone(system)


# =============================================================================
# TestPersistence
# =============================================================================

class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("persist test", "happy", 0.7),
        ])
        store = system.bind_emotions(stm=stm)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_binding_state(store, filepath)
            loaded = load_binding_state(filepath)
            self.assertEqual(len(loaded.bindings), len(store.bindings))
            self.assertTrue(loaded.has_bindings())
        finally:
            os.unlink(filepath)

    def test_roundtrip_preserves_data(self):
        system = EmotionalMemoryBindingSystem()
        stm = _MockSTM(entries=[
            _MockSTMEntry("roundtrip test", "angry", 0.6, -0.3),
        ])
        store = system.bind_emotions(stm=stm)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_binding_state(store, filepath)
            loaded = load_binding_state(filepath)

            original_binding = store.bindings[0]
            loaded_binding = loaded.bindings[0]
            self.assertEqual(original_binding.memory_key, loaded_binding.memory_key)
            self.assertEqual(len(original_binding.traces), len(loaded_binding.traces))
            self.assertEqual(
                original_binding.traces[0].emotion_label,
                loaded_binding.traces[0].emotion_label,
            )
        finally:
            os.unlink(filepath)


# =============================================================================
# TestDesignPrinciples
# =============================================================================

class TestDesignPrinciples(unittest.TestCase):
    """Design constraint meta-tests."""

    def test_no_decision_method(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("decide", method.lower())
            self.assertNotIn("choose", method.lower())
            self.assertNotIn("select_action", method.lower())

    def test_no_value_update_method(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("update_value", method.lower())
            self.assertNotIn("set_belief", method.lower())

    def test_no_goal_method(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("generate_goal", method.lower())
            self.assertNotIn("create_goal", method.lower())

    def test_no_responsibility_method(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("update_responsibility", method.lower())
            self.assertNotIn("evaluate_outcome", method.lower())

    def test_no_identity_definition(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("define_identity", method.lower())
            self.assertNotIn("set_identity", method.lower())

    def test_no_emotion_evaluation(self):
        system = EmotionalMemoryBindingSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            self.assertNotIn("evaluate_emotion", method.lower())
            self.assertNotIn("judge_emotion", method.lower())
            self.assertNotIn("correct_emotion", method.lower())

    def test_traces_are_frozen(self):
        t = _make_trace()
        with self.assertRaises(AttributeError):
            t.emotion_label = "anger"

    def test_bindings_are_frozen(self):
        b = _make_binding()
        with self.assertRaises(AttributeError):
            b.memory_summary = "changed"


if __name__ == "__main__":
    unittest.main()
