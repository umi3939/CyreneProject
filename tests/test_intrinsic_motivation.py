"""
Tests for psyche/intrinsic_motivation.py

自発的内的動機（Intrinsic Motivation）のテスト。
~114件のテストケース。
"""

import json
import os
import tempfile
import unittest

from psyche.intrinsic_motivation import (
    # Enums
    MotiveSourceType,
    MotiveAffinity,
    MotiveStrength,
    MotiveFreshness,
    # Level determination
    determine_freshness_level,
    determine_strength_level,
    # Key generation
    generate_motive_key,
    # Dataclasses
    MotiveImpulse,
    MotiveLink,
    MotiveEntry,
    MotiveStore,
    IntrinsicMotivationConfig,
    # Extraction functions
    extract_from_emotion_state,
    extract_from_tendencies,
    extract_from_goal_vectors,
    extract_from_goal_candidates,
    # Computation functions
    compute_motive_strength,
    detect_motive_coexistence,
    compute_motive_overlay,
    # System
    IntrinsicMotivationSystem,
    # Integration
    sense_from_chain,
    generate_motive_tags,
    get_motive_summary,
    get_motive_for_introspection,
    # Verification
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    verify_no_motivation_prescription,
    # Convenience
    create_config,
    create_empty_store,
    create_system,
    save_motive_state,
    load_motive_state,
)


# =============================================================================
# Test Helpers
# =============================================================================

def _make_impulse(
    label: str = "emotion_joy",
    intensity: float = 0.5,
    valence: float = 0.3,
    freshness: float = 0.8,
    reference_count: int = 0,
    affinity: MotiveAffinity = MotiveAffinity.EMOTIONAL_SURGE,
) -> MotiveImpulse:
    return MotiveImpulse(
        impulse_id="impulse_test_01",
        label=label,
        intensity=intensity,
        valence=valence,
        freshness=freshness,
        reference_count=reference_count,
        affinity=affinity,
        timestamp="12345",
        source_description="test source",
    )


def _make_entry(
    motive_key: str = "motive_key_01",
    impulses: tuple = None,
    freshness: float = 0.8,
    reference_count: int = 0,
) -> MotiveEntry:
    if impulses is None:
        impulses = (_make_impulse(),)
    return MotiveEntry(
        motive_id="entry_test_01",
        motive_key=motive_key,
        motive_summary="test motive summary",
        impulses=impulses,
        motive_links=(),
        freshness=freshness,
        reference_count=reference_count,
        creation_timestamp="12345",
        last_reference_timestamp="12345",
        revision_count=0,
        undetermined_aspects=("test_aspect",),
    )


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


class _MockTendencyPattern:
    def __init__(self, category_value="approach"):
        self.category = type("Cat", (), {"value": category_value})()


class _MockTendency:
    def __init__(self, strength=0.1, category_value="approach"):
        self.strength = strength
        self.pattern = _MockTendencyPattern(category_value)


class _MockTendencyState:
    def __init__(self, tendencies=None):
        self.tendencies = tendencies or []


class _MockVector:
    def __init__(self, vector_id="v1", direction=None, magnitude=0.5):
        self.vector_id = vector_id
        self.direction = direction or {"approach": 0.8, "explore": 0.2}
        self.magnitude = magnitude


class _MockVectorState:
    def __init__(self, vectors=None):
        self.vectors = vectors or []


class _MockCandidate:
    def __init__(self, candidate_id="c1", category_value="approach", intensity=0.5):
        self.candidate_id = candidate_id
        self.category = type("Cat", (), {"value": category_value})()
        self.intensity = intensity


class _MockCandidateState:
    def __init__(self, candidates=None):
        self.candidates = candidates or []


# =============================================================================
# TestEnums
# =============================================================================

class TestEnums(unittest.TestCase):
    """Test all Enum values."""

    def test_motive_source_type_values(self):
        self.assertEqual(MotiveSourceType.EMOTION.value, "emotion")
        self.assertEqual(MotiveSourceType.TENDENCY.value, "tendency")
        self.assertEqual(MotiveSourceType.GOAL_VECTOR.value, "goal_vector")
        self.assertEqual(MotiveSourceType.GOAL_CANDIDATE.value, "goal_candidate")
        self.assertEqual(MotiveSourceType.MIXED.value, "mixed")

    def test_motive_affinity_values(self):
        self.assertEqual(MotiveAffinity.EMOTIONAL_SURGE.value, "emotional_surge")
        self.assertEqual(MotiveAffinity.HABITUAL.value, "habitual")
        self.assertEqual(MotiveAffinity.DIRECTIONAL.value, "directional")
        self.assertEqual(MotiveAffinity.ASPIRATIONAL.value, "aspirational")
        self.assertEqual(MotiveAffinity.COMPOSITE.value, "composite")
        self.assertEqual(MotiveAffinity.UNDEFINED.value, "undefined")

    def test_motive_strength_values(self):
        self.assertEqual(MotiveStrength.STRONG.value, "strong")
        self.assertEqual(MotiveStrength.MODERATE.value, "moderate")
        self.assertEqual(MotiveStrength.WEAK.value, "weak")
        self.assertEqual(MotiveStrength.FAINT.value, "faint")
        self.assertEqual(MotiveStrength.UNDEFINED.value, "undefined")

    def test_motive_freshness_values(self):
        self.assertEqual(MotiveFreshness.FRESH.value, "fresh")
        self.assertEqual(MotiveFreshness.RECENT.value, "recent")
        self.assertEqual(MotiveFreshness.AGING.value, "aging")
        self.assertEqual(MotiveFreshness.STALE.value, "stale")
        self.assertEqual(MotiveFreshness.FADED.value, "faded")

    def test_all_enums_have_expected_count(self):
        self.assertEqual(len(MotiveSourceType), 5)
        self.assertEqual(len(MotiveAffinity), 6)
        self.assertEqual(len(MotiveStrength), 5)
        self.assertEqual(len(MotiveFreshness), 5)


# =============================================================================
# TestDetermineFreshnessLevel
# =============================================================================

class TestDetermineFreshnessLevel(unittest.TestCase):
    def test_fresh(self):
        self.assertEqual(determine_freshness_level(0.9), MotiveFreshness.FRESH)

    def test_recent(self):
        self.assertEqual(determine_freshness_level(0.65), MotiveFreshness.RECENT)

    def test_aging(self):
        self.assertEqual(determine_freshness_level(0.45), MotiveFreshness.AGING)

    def test_stale(self):
        self.assertEqual(determine_freshness_level(0.2), MotiveFreshness.STALE)

    def test_faded(self):
        self.assertEqual(determine_freshness_level(0.05), MotiveFreshness.FADED)


# =============================================================================
# TestDetermineStrengthLevel
# =============================================================================

class TestDetermineStrengthLevel(unittest.TestCase):
    def test_strong(self):
        self.assertEqual(determine_strength_level(0.8), MotiveStrength.STRONG)

    def test_moderate(self):
        self.assertEqual(determine_strength_level(0.5), MotiveStrength.MODERATE)

    def test_weak(self):
        self.assertEqual(determine_strength_level(0.25), MotiveStrength.WEAK)

    def test_faint(self):
        self.assertEqual(determine_strength_level(0.08), MotiveStrength.FAINT)

    def test_undefined(self):
        self.assertEqual(determine_strength_level(0.02), MotiveStrength.UNDEFINED)


# =============================================================================
# TestMotiveImpulse
# =============================================================================

class TestMotiveImpulse(unittest.TestCase):
    def test_creation(self):
        imp = _make_impulse()
        self.assertEqual(imp.label, "emotion_joy")
        self.assertEqual(imp.intensity, 0.5)

    def test_get_freshness_level(self):
        imp = _make_impulse(freshness=0.9)
        self.assertEqual(imp.get_freshness_level(), MotiveFreshness.FRESH)

    def test_with_freshness(self):
        imp = _make_impulse(freshness=0.5)
        imp2 = imp.with_freshness(0.9)
        self.assertAlmostEqual(imp2.freshness, 0.9)
        self.assertAlmostEqual(imp.freshness, 0.5)  # immutable

    def test_with_freshness_clamp(self):
        imp = _make_impulse()
        imp2 = imp.with_freshness(1.5)
        self.assertAlmostEqual(imp2.freshness, 1.0)
        imp3 = imp.with_freshness(-0.5)
        self.assertAlmostEqual(imp3.freshness, 0.0)

    def test_with_intensity(self):
        imp = _make_impulse(intensity=0.3)
        imp2 = imp.with_intensity(0.7)
        self.assertAlmostEqual(imp2.intensity, 0.7)

    def test_with_reference(self):
        imp = _make_impulse(reference_count=2)
        imp2 = imp.with_reference()
        self.assertEqual(imp2.reference_count, 3)
        self.assertEqual(imp.reference_count, 2)  # immutable

    def test_reattach(self):
        imp = _make_impulse(affinity=MotiveAffinity.EMOTIONAL_SURGE)
        imp2 = imp.reattach(MotiveAffinity.HABITUAL)
        self.assertEqual(imp2.affinity, MotiveAffinity.HABITUAL)
        self.assertEqual(imp.affinity, MotiveAffinity.EMOTIONAL_SURGE)

    def test_frozen(self):
        imp = _make_impulse()
        with self.assertRaises(AttributeError):
            imp.intensity = 0.9


# =============================================================================
# TestMotiveLink
# =============================================================================

class TestMotiveLink(unittest.TestCase):
    def test_creation(self):
        link = MotiveLink(
            link_id="link01",
            motive_id="m01",
            source_type=MotiveSourceType.EMOTION,
            source_description="test desc",
            contribution=0.8,
        )
        self.assertEqual(link.link_id, "link01")
        self.assertEqual(link.contribution, 0.8)

    def test_frozen(self):
        link = MotiveLink(
            link_id="link01",
            motive_id="m01",
            source_type=MotiveSourceType.EMOTION,
            source_description="test desc",
            contribution=0.8,
        )
        with self.assertRaises(AttributeError):
            link.contribution = 0.5


# =============================================================================
# TestMotiveEntry
# =============================================================================

class TestMotiveEntry(unittest.TestCase):
    def test_creation(self):
        entry = _make_entry()
        self.assertEqual(entry.motive_key, "motive_key_01")
        self.assertEqual(len(entry.impulses), 1)

    def test_with_freshness(self):
        entry = _make_entry(freshness=0.5)
        entry2 = entry.with_freshness(0.9)
        self.assertAlmostEqual(entry2.freshness, 0.9)
        self.assertAlmostEqual(entry.freshness, 0.5)

    def test_with_reference(self):
        entry = _make_entry(reference_count=1)
        entry2 = entry.with_reference()
        self.assertEqual(entry2.reference_count, 2)

    def test_with_impulses(self):
        entry = _make_entry()
        imp2 = _make_impulse(label="emotion_anger")
        entry2 = entry.with_impulses((entry.impulses[0], imp2))
        self.assertEqual(len(entry2.impulses), 2)

    def test_revise_summary(self):
        entry = _make_entry()
        entry2 = entry.revise_summary("new summary")
        self.assertEqual(entry2.motive_summary, "new summary")
        self.assertEqual(entry2.revision_count, 1)

    def test_with_added_impulse(self):
        entry = _make_entry()
        imp2 = _make_impulse(label="emotion_anger")
        entry2 = entry.with_added_impulse(imp2)
        self.assertEqual(len(entry2.impulses), 2)

    def test_frozen(self):
        entry = _make_entry()
        with self.assertRaises(AttributeError):
            entry.freshness = 0.9

    def test_freshness_clamp(self):
        entry = _make_entry()
        entry2 = entry.with_freshness(1.5)
        self.assertAlmostEqual(entry2.freshness, 1.0)
        entry3 = entry.with_freshness(-0.5)
        self.assertAlmostEqual(entry3.freshness, 0.0)


# =============================================================================
# TestMotiveStore
# =============================================================================

class TestMotiveStore(unittest.TestCase):
    def test_empty_store(self):
        store = create_empty_store()
        self.assertFalse(store.has_entries())
        self.assertEqual(len(store.entries), 0)

    def test_has_entries(self):
        store = MotiveStore(
            entries=(_make_entry(),),
            motive_links=(),
            total_entries_created=1,
            total_impulses_created=1,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.8,
            average_impulse_count=1.0,
            active_entry_count=1,
            timestamp="12345",
            description="test",
        )
        self.assertTrue(store.has_entries())

    def test_get_active_entries(self):
        e1 = _make_entry(freshness=0.8)
        e2 = _make_entry(freshness=0.1, motive_key="key2")
        store = MotiveStore(
            entries=(e1, e2),
            motive_links=(),
            total_entries_created=2,
            total_impulses_created=2,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.45,
            average_impulse_count=1.0,
            active_entry_count=1,
            timestamp="12345",
            description="test",
        )
        active = store.get_active_entries()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].motive_key, "motive_key_01")

    def test_get_entries_for_key(self):
        e1 = _make_entry(motive_key="key_a")
        e2 = _make_entry(motive_key="key_b")
        store = MotiveStore(
            entries=(e1, e2),
            motive_links=(),
            total_entries_created=2,
            total_impulses_created=2,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.8,
            average_impulse_count=1.0,
            active_entry_count=2,
            timestamp="12345",
            description="test",
        )
        found = store.get_entries_for_key("key_a")
        self.assertEqual(len(found), 1)

    def test_to_dict(self):
        store = create_empty_store()
        d = store.to_dict()
        self.assertIn("entries", d)
        self.assertIn("motive_links", d)
        self.assertIn("total_entries_created", d)

    def test_from_dict(self):
        store = create_empty_store()
        d = store.to_dict()
        restored = MotiveStore.from_dict(d)
        self.assertEqual(len(restored.entries), 0)
        self.assertEqual(restored.total_entries_created, 0)

    def test_round_trip_serialization(self):
        imp = _make_impulse()
        entry = _make_entry(impulses=(imp,))
        link = MotiveLink(
            link_id="l1", motive_id="m1",
            source_type=MotiveSourceType.EMOTION,
            source_description="desc", contribution=0.8,
        )
        store = MotiveStore(
            entries=(entry,),
            motive_links=(link,),
            total_entries_created=1,
            total_impulses_created=1,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.8,
            average_impulse_count=1.0,
            active_entry_count=1,
            timestamp="12345",
            description="test",
        )
        d = store.to_dict()
        restored = MotiveStore.from_dict(d)
        self.assertEqual(len(restored.entries), 1)
        self.assertEqual(len(restored.motive_links), 1)
        self.assertEqual(restored.entries[0].motive_key, "motive_key_01")
        self.assertEqual(restored.motive_links[0].link_id, "l1")


# =============================================================================
# TestIntrinsicMotivationConfig
# =============================================================================

class TestIntrinsicMotivationConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = IntrinsicMotivationConfig()
        self.assertEqual(cfg.max_entries, 150)
        self.assertEqual(cfg.max_impulses_per_entry, 7)
        self.assertAlmostEqual(cfg.base_decay_rate, 0.025)

    def test_custom_config(self):
        cfg = create_config(max_entries=50, base_decay_rate=0.05)
        self.assertEqual(cfg.max_entries, 50)
        self.assertAlmostEqual(cfg.base_decay_rate, 0.05)


# =============================================================================
# TestExtractFromEmotionState
# =============================================================================

class TestExtractFromEmotionState(unittest.TestCase):
    def test_none_returns_empty(self):
        result = extract_from_emotion_state(None, None)
        self.assertEqual(result, [])

    def test_object_extraction(self):
        emotion = _MockEmotionVector(joy=0.7, anger=0.3)
        mood = _MockMood(valence=0.2)
        result = extract_from_emotion_state(emotion, mood)
        self.assertTrue(len(result) >= 2)
        labels = [r[1] for r in result]
        self.assertIn("emotion_joy", labels)
        self.assertIn("emotion_anger", labels)

    def test_dict_extraction(self):
        emotion = {"joy": 0.5, "anger": 0.2}
        mood = {"valence": 0.1}
        result = extract_from_emotion_state(emotion, mood)
        self.assertTrue(len(result) >= 2)

    def test_low_emotions_filtered(self):
        emotion = _MockEmotionVector(joy=0.05, anger=0.01)
        result = extract_from_emotion_state(emotion, None)
        self.assertEqual(len(result), 0)


# =============================================================================
# TestExtractFromTendencies
# =============================================================================

class TestExtractFromTendencies(unittest.TestCase):
    def test_none_returns_empty(self):
        result = extract_from_tendencies(None)
        self.assertEqual(result, [])

    def test_object_extraction(self):
        state = _MockTendencyState([
            _MockTendency(strength=0.1, category_value="approach"),
            _MockTendency(strength=0.05, category_value="exploration"),
        ])
        result = extract_from_tendencies(state)
        self.assertTrue(len(result) >= 2)
        labels = [r[1] for r in result]
        self.assertIn("tendency_approach", labels)
        self.assertIn("tendency_exploration", labels)

    def test_dict_extraction(self):
        state = {
            "tendencies": [
                {"strength": 0.1, "pattern": {"category": "approach"}},
                {"strength": 0.05, "pattern": {"category": "avoidance"}},
            ]
        }
        result = extract_from_tendencies(state)
        self.assertTrue(len(result) >= 2)

    def test_low_strength_filtered(self):
        state = _MockTendencyState([
            _MockTendency(strength=0.005),
        ])
        result = extract_from_tendencies(state)
        self.assertEqual(len(result), 0)


# =============================================================================
# TestExtractFromGoalVectors
# =============================================================================

class TestExtractFromGoalVectors(unittest.TestCase):
    def test_none_returns_empty(self):
        result = extract_from_goal_vectors(None)
        self.assertEqual(result, [])

    def test_object_extraction(self):
        state = _MockVectorState([
            _MockVector(vector_id="v1", direction={"approach": 0.8, "explore": 0.2}, magnitude=0.6),
        ])
        result = extract_from_goal_vectors(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "vector_approach")

    def test_dict_extraction(self):
        state = {
            "vectors": [
                {"vector_id": "v2", "direction": {"explore": 0.9}, "magnitude": 0.5},
            ]
        }
        result = extract_from_goal_vectors(state)
        self.assertEqual(len(result), 1)

    def test_low_magnitude_filtered(self):
        state = _MockVectorState([
            _MockVector(magnitude=0.05),
        ])
        result = extract_from_goal_vectors(state)
        self.assertEqual(len(result), 0)


# =============================================================================
# TestExtractFromGoalCandidates
# =============================================================================

class TestExtractFromGoalCandidates(unittest.TestCase):
    def test_none_returns_empty(self):
        result = extract_from_goal_candidates(None)
        self.assertEqual(result, [])

    def test_object_extraction(self):
        state = _MockCandidateState([
            _MockCandidate(candidate_id="c1", category_value="exploration", intensity=0.6),
        ])
        result = extract_from_goal_candidates(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "candidate_exploration")

    def test_dict_extraction(self):
        state = {
            "candidates": [
                {"candidate_id": "c2", "category": "connection", "intensity": 0.4},
            ]
        }
        result = extract_from_goal_candidates(state)
        self.assertEqual(len(result), 1)

    def test_low_intensity_filtered(self):
        state = _MockCandidateState([
            _MockCandidate(intensity=0.05),
        ])
        result = extract_from_goal_candidates(state)
        self.assertEqual(len(result), 0)


# =============================================================================
# TestComputeMotiveStrength
# =============================================================================

class TestComputeMotiveStrength(unittest.TestCase):
    def test_empty_impulses(self):
        result = compute_motive_strength(())
        self.assertAlmostEqual(result, 0.0)

    def test_single_impulse(self):
        imp = _make_impulse(intensity=0.6)
        result = compute_motive_strength((imp,))
        self.assertAlmostEqual(result, 0.6)

    def test_multiple_impulses(self):
        imp1 = _make_impulse(intensity=0.8)
        imp2 = _make_impulse(intensity=0.4)
        result = compute_motive_strength((imp1, imp2))
        # weight[0] = 1.0, weight[1] = 1.0/1.2 = 0.833
        # weighted_sum = 0.8*1.0 + 0.4*0.833 = 0.8 + 0.333 = 1.133
        # total_weight = 1.0 + 0.833 = 1.833
        # result = 1.133 / 1.833 ≈ 0.618
        self.assertTrue(0.5 < result < 0.8)


# =============================================================================
# TestComputeMotiveOverlay
# =============================================================================

class TestComputeMotiveOverlay(unittest.TestCase):
    def test_empty_entry(self):
        entry = _make_entry(impulses=())
        result = compute_motive_overlay(entry)
        self.assertEqual(result, {})

    def test_single_impulse_overlay(self):
        imp = _make_impulse(intensity=0.6, freshness=0.8)
        entry = _make_entry(impulses=(imp,))
        result = compute_motive_overlay(entry)
        self.assertIn("emotion_joy", result)
        self.assertAlmostEqual(result["emotion_joy"], 0.48)  # 0.6 * 0.8

    def test_multiple_impulses_max_merge(self):
        imp1 = _make_impulse(label="emotion_joy", intensity=0.6, freshness=0.8)
        imp2 = MotiveImpulse(
            impulse_id="imp2", label="emotion_joy", intensity=0.9, freshness=0.5,
            valence=0.3, reference_count=0, affinity=MotiveAffinity.EMOTIONAL_SURGE,
            timestamp="12345", source_description="test",
        )
        entry = _make_entry(impulses=(imp1, imp2))
        result = compute_motive_overlay(entry)
        # max(0.6*0.8, 0.9*0.5) = max(0.48, 0.45) = 0.48
        self.assertAlmostEqual(result["emotion_joy"], 0.48)


# =============================================================================
# TestGenerateMotiveKey
# =============================================================================

class TestGenerateMotiveKey(unittest.TestCase):
    def test_consistency(self):
        key1 = generate_motive_key("test_text")
        key2 = generate_motive_key("test_text")
        self.assertEqual(key1, key2)

    def test_different_texts(self):
        key1 = generate_motive_key("text_a")
        key2 = generate_motive_key("text_b")
        self.assertNotEqual(key1, key2)


# =============================================================================
# TestIntrinsicMotivationSystem
# =============================================================================

class TestIntrinsicMotivationSystem(unittest.TestCase):
    def test_creation(self):
        system = create_system()
        self.assertIsInstance(system, IntrinsicMotivationSystem)

    def test_sense_with_none_inputs(self):
        system = create_system()
        store = system.sense_motives()
        self.assertIsInstance(store, MotiveStore)
        self.assertFalse(store.has_entries())

    def test_sense_with_emotion(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, anger=0.3)
        mood = _MockMood(valence=0.2)
        store = system.sense_motives(emotion=emotion, mood=mood)
        self.assertTrue(store.has_entries())

    def test_sense_with_tendencies(self):
        system = create_system()
        state = _MockTendencyState([
            _MockTendency(strength=0.1, category_value="approach"),
        ])
        store = system.sense_motives(tendencies=state)
        self.assertTrue(store.has_entries())

    def test_sense_with_vectors(self):
        system = create_system()
        state = _MockVectorState([
            _MockVector(vector_id="v1", magnitude=0.5),
        ])
        store = system.sense_motives(vectors=state)
        self.assertTrue(store.has_entries())

    def test_sense_with_candidates(self):
        system = create_system()
        state = _MockCandidateState([
            _MockCandidate(candidate_id="c1", intensity=0.5),
        ])
        store = system.sense_motives(candidates=state)
        self.assertTrue(store.has_entries())

    def test_decay_motives(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        system.sense_motives(emotion=emotion)
        store = system.decay_motives()
        # After decay, freshness should decrease
        if store.has_entries():
            for entry in store.entries:
                self.assertLess(entry.freshness, 1.0)

    def test_reference_motive(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        # Get the motive key
        if store.has_entries():
            key = store.entries[0].motive_key
            system.reference_motive(key)
            store2 = system.get_store()
            entry = store2.entries[0]
            self.assertGreater(entry.reference_count, 0)

    def test_get_motive_overlay(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        if store.has_entries():
            key = store.entries[0].motive_key
            overlay = system.get_motive_overlay(key)
            self.assertIsInstance(overlay, dict)

    def test_revise_motive(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        if store.has_entries():
            key = store.entries[0].motive_key
            system.revise_motive(key, "revised summary")
            store2 = system.get_store()
            self.assertEqual(store2.entries[0].motive_summary, "revised summary")

    def test_get_active_motives(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, love=0.5)
        system.sense_motives(emotion=emotion)
        active = system.get_active_motives()
        self.assertIsInstance(active, list)

    def test_get_store(self):
        system = create_system()
        store = system.get_store()
        self.assertIsInstance(store, MotiveStore)

    def test_get_last_store(self):
        system = create_system()
        self.assertIsNone(system.get_last_store())
        system.sense_motives()
        self.assertIsNotNone(system.get_last_store())

    def test_merge_existing_entry(self):
        system = create_system()
        emotion1 = _MockEmotionVector(joy=0.7)
        system.sense_motives(emotion=emotion1)
        # Sense again with higher joy → should merge
        emotion2 = _MockEmotionVector(joy=0.9, anger=0.4)
        store = system.sense_motives(emotion=emotion2)
        # Should still be one entry for __emotion_motive__ key
        emotion_entries = [e for e in store.entries if e.motive_key == "__emotion_motive__"]
        self.assertEqual(len(emotion_entries), 1)
        # But may have more impulses
        self.assertGreaterEqual(len(emotion_entries[0].impulses), 1)

    def test_capacity_enforcement(self):
        config = create_config(max_entries=3)
        system = create_system(config)
        # Add many different entries
        for i in range(10):
            state = _MockCandidateState([
                _MockCandidate(candidate_id=f"cand_{i}", intensity=0.5),
            ])
            system.sense_motives(candidates=state)
        store = system.get_store()
        self.assertLessEqual(len(store.entries), 3)

    def test_custom_config(self):
        config = create_config(max_entries=50, base_decay_rate=0.1)
        system = create_system(config)
        self.assertEqual(system._config.max_entries, 50)


# =============================================================================
# TestSenseFromChain
# =============================================================================

class TestSenseFromChain(unittest.TestCase):
    def test_chain_integration(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        mood = _MockMood(valence=0.3)
        store = sense_from_chain(system, emotion=emotion, mood=mood)
        self.assertIsInstance(store, MotiveStore)
        self.assertTrue(store.has_entries())

    def test_chain_with_all_sources(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.5)
        mood = _MockMood()
        tendencies = _MockTendencyState([_MockTendency(strength=0.1)])
        vectors = _MockVectorState([_MockVector(magnitude=0.5)])
        candidates = _MockCandidateState([_MockCandidate(intensity=0.4)])
        store = sense_from_chain(system, emotion, mood, tendencies, vectors, candidates)
        self.assertTrue(store.has_entries())
        self.assertGreater(store.active_entry_count, 0)


# =============================================================================
# TestGenerateMotiveTags
# =============================================================================

class TestGenerateMotiveTags(unittest.TestCase):
    def test_none_store(self):
        tags = generate_motive_tags(None)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["category"], "INTRINSIC_MOTIVE_COUNT")

    def test_store_with_entries(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, anger=0.3)
        store = system.sense_motives(emotion=emotion)
        tags = generate_motive_tags(store)
        self.assertTrue(len(tags) >= 3)
        categories = [t["category"] for t in tags]
        self.assertIn("INTRINSIC_MOTIVE_COUNT", categories)
        self.assertIn("INTRINSIC_MOTIVE_FRESHNESS", categories)
        self.assertIn("INTRINSIC_MOTIVE_RICHNESS", categories)

    def test_scale_affects_weight(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        tags_1 = generate_motive_tags(store, scale=1.0)
        tags_2 = generate_motive_tags(store, scale=2.0)
        # Weights should differ
        w1 = sum(t["weight"] for t in tags_1)
        w2 = sum(t["weight"] for t in tags_2)
        self.assertGreater(w2, w1)

    def test_empty_store(self):
        store = create_empty_store()
        tags = generate_motive_tags(store)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["label"], "no_motives")


# =============================================================================
# TestGetMotiveSummary
# =============================================================================

class TestGetMotiveSummary(unittest.TestCase):
    def test_none_store(self):
        summary = get_motive_summary(None)
        self.assertIn("No motives formed yet", summary)

    def test_store_with_entries(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, anger=0.3)
        store = system.sense_motives(emotion=emotion)
        summary = get_motive_summary(store)
        self.assertIn("Intrinsic Motivation State", summary)
        self.assertIn("Active motives", summary)


# =============================================================================
# TestGetMotiveForIntrospection
# =============================================================================

class TestGetMotiveForIntrospection(unittest.TestCase):
    def test_none_store(self):
        result = get_motive_for_introspection(None)
        self.assertFalse(result["has_motives"])
        self.assertEqual(result["total_motives"], 0)

    def test_store_with_entries(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        result = get_motive_for_introspection(store)
        self.assertTrue(result["has_motives"])
        self.assertGreater(result["total_motives"], 0)
        self.assertIn("impulse_distribution", result)


# =============================================================================
# TestVerification
# =============================================================================

class TestVerification(unittest.TestCase):
    def test_no_decision_impact(self):
        store = create_empty_store()
        self.assertTrue(verify_no_decision_impact(store))

    def test_no_goal_generation(self):
        system = create_system()
        self.assertTrue(verify_no_goal_generation(system))

    def test_read_only_principle(self):
        system = create_system()
        self.assertTrue(verify_read_only_principle(system))

    def test_no_value_modification(self):
        system = create_system()
        self.assertTrue(verify_no_value_modification(system))

    def test_no_motivation_prescription(self):
        system = create_system()
        self.assertTrue(verify_no_motivation_prescription(system))

    def test_store_with_entries_no_decision_impact(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        self.assertTrue(verify_no_decision_impact(store))


# =============================================================================
# TestConvenience
# =============================================================================

class TestConvenience(unittest.TestCase):
    def test_create_config(self):
        cfg = create_config(max_entries=100)
        self.assertEqual(cfg.max_entries, 100)

    def test_create_empty_store(self):
        store = create_empty_store()
        self.assertFalse(store.has_entries())

    def test_create_system(self):
        system = create_system()
        self.assertIsInstance(system, IntrinsicMotivationSystem)

    def test_create_system_with_config(self):
        cfg = create_config(max_entries=50)
        system = create_system(cfg)
        self.assertEqual(system._config.max_entries, 50)


# =============================================================================
# TestPersistence
# =============================================================================

class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, anger=0.3)
        store = system.sense_motives(emotion=emotion)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_motive_state(store, filepath)
            loaded = load_motive_state(filepath)
            self.assertEqual(len(loaded.entries), len(store.entries))
            self.assertEqual(loaded.total_entries_created, store.total_entries_created)
        finally:
            os.unlink(filepath)

    def test_round_trip_preserves_data(self):
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, love=0.5, anger=0.2)
        mood = _MockMood(valence=0.4)
        store = system.sense_motives(emotion=emotion, mood=mood)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_motive_state(store, filepath)
            loaded = load_motive_state(filepath)

            # Verify structure
            self.assertEqual(len(loaded.entries), len(store.entries))
            self.assertEqual(loaded.active_entry_count, store.active_entry_count)
            self.assertAlmostEqual(loaded.average_freshness, store.average_freshness, places=3)

            # Verify entries
            for orig, rest in zip(store.entries, loaded.entries):
                self.assertEqual(orig.motive_key, rest.motive_key)
                self.assertEqual(len(orig.impulses), len(rest.impulses))
        finally:
            os.unlink(filepath)


# =============================================================================
# TestDesignPrinciples
# =============================================================================

class TestDesignPrinciples(unittest.TestCase):
    """Meta-tests verifying design constraints from the design doc."""

    def test_no_decision_method(self):
        """System has no method that directly decides or selects actions."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["decide", "select_action", "choose", "execute"]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests decision-making capability")

    def test_no_value_setting(self):
        """System has no method that directly sets values or beliefs."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["set_value", "set_belief", "define_identity"]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests value-setting capability")

    def test_no_action_optimization(self):
        """System has no method that optimizes behavior."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["optimize", "maximize", "minimize"]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests optimization capability")

    def test_no_judgment(self):
        """System has no method for moral/correctness judgment."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["judge", "evaluate_morality", "score_correctness"]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests judgment capability")

    def test_no_norm_direction(self):
        """System has no method for normative direction."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["prescribe", "enforce", "normalize_motive"]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests normative direction")

    def test_motives_are_provisional(self):
        """Motive entries carry undetermined_aspects indicating provisional nature."""
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store = system.sense_motives(emotion=emotion)
        if store.has_entries():
            entry = store.entries[0]
            self.assertTrue(len(entry.undetermined_aspects) > 0)

    def test_motives_coexist(self):
        """Multiple motives can coexist without conflict resolution."""
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7, anger=0.5, fear=0.3)
        tendencies = _MockTendencyState([_MockTendency(strength=0.1)])
        store = system.sense_motives(emotion=emotion, tendencies=tendencies)
        # Multiple entries should exist
        self.assertGreater(len(store.entries), 0)

    def test_motives_decay_naturally(self):
        """Motives decay over time without external intervention."""
        system = create_system()
        emotion = _MockEmotionVector(joy=0.7)
        store1 = system.sense_motives(emotion=emotion)
        initial_freshness = store1.entries[0].freshness if store1.has_entries() else 0

        # Decay multiple times
        for _ in range(5):
            store2 = system.decay_motives()

        if store2.has_entries():
            self.assertLess(store2.entries[0].freshness, initial_freshness)

    def test_no_external_write(self):
        """System does not write to external state."""
        system = create_system()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = [
            "update_emotion", "update_memory", "set_emotion",
            "modify_bias", "apply_to_decision",
        ]
        for method in methods:
            for pattern in forbidden:
                self.assertNotIn(pattern, method.lower(),
                    f"Method '{method}' suggests external write capability")


if __name__ == "__main__":
    unittest.main()
