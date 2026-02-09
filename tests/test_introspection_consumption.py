"""
Tests for Introspection Consumption Layer (内省の消費層)

Comprehensive tests covering:
- Enums and constants
- IntrospectionFragment dataclass
- FragmentBundle dataclass
- ConsumptionRecord dataclass
- ConsumptionStore dataclass (including serialization)
- ConsumptionLayerConfig
- Fragment extraction functions (pure)
- Bundle generation functions (pure)
- IntrospectionConsumptionSystem
- Integration functions
- Verification functions
- Convenience / persistence functions
"""

import json
import os
import tempfile
import time

import pytest

from psyche.introspection_consumption import (
    # Enums
    FragmentSourceType,
    BundleCoherence,
    FragmentFreshness,
    # Dataclasses
    IntrospectionFragment,
    FragmentBundle,
    ConsumptionRecord,
    ConsumptionStore,
    ConsumptionLayerConfig,
    # System
    IntrospectionConsumptionSystem,
    # Pure functions
    determine_freshness,
    extract_from_introspection,
    extract_from_narrative,
    extract_from_coherence,
    extract_from_tendency,
    extract_from_episodic,
    compute_fragment_relevance,
    determine_bundle_coherence,
    generate_bundles,
    # Integration
    consume_from_chain,
    generate_consumption_tags,
    get_consumption_summary,
    get_consumption_for_introspection,
    # Verification
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    # Convenience
    create_config,
    create_empty_store,
    create_system,
    save_consumption_state,
    load_consumption_state,
)


# =============================================================================
# Helper: create test fragments
# =============================================================================

def _make_fragment(
    content: str = "test fragment",
    source_type: FragmentSourceType = FragmentSourceType.INTROSPECTION_LOG,
    freshness: float = 1.0,
    reference_count: int = 0,
    source_ids: tuple = (),
    undetermined: tuple = (),
    fragment_id: str = "",
    timestamp: float = 0.0,
) -> IntrospectionFragment:
    return IntrospectionFragment(
        fragment_id=fragment_id or f"frag_{id(content)}",
        source_type=source_type,
        content=content,
        timestamp=timestamp or time.time(),
        freshness=freshness,
        reference_count=reference_count,
        source_ids=source_ids,
        undetermined_aspects=undetermined,
    )


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    def test_fragment_source_type_values(self):
        assert FragmentSourceType.INTROSPECTION_LOG.value == "introspection_log"
        assert FragmentSourceType.SELF_NARRATIVE.value == "self_narrative"
        assert FragmentSourceType.IDENTITY_COHERENCE.value == "identity_coherence"
        assert FragmentSourceType.TENDENCY_AWARENESS.value == "tendency_awareness"
        assert FragmentSourceType.EPISODIC_MEMORY.value == "episodic_memory"
        assert FragmentSourceType.MIXED.value == "mixed"

    def test_bundle_coherence_values(self):
        assert BundleCoherence.TIGHT.value == "tight"
        assert BundleCoherence.LOOSE.value == "loose"
        assert BundleCoherence.SCATTERED.value == "scattered"
        assert BundleCoherence.UNDEFINED.value == "undefined"

    def test_fragment_freshness_values(self):
        assert FragmentFreshness.FRESH.value == "fresh"
        assert FragmentFreshness.RECENT.value == "recent"
        assert FragmentFreshness.AGING.value == "aging"
        assert FragmentFreshness.STALE.value == "stale"
        assert FragmentFreshness.FADED.value == "faded"

    def test_fragment_source_type_count(self):
        assert len(FragmentSourceType) == 6

    def test_bundle_coherence_count(self):
        assert len(BundleCoherence) == 4

    def test_fragment_freshness_count(self):
        assert len(FragmentFreshness) == 5


# =============================================================================
# Freshness Determination Tests
# =============================================================================

class TestDetermineFreshness:
    def test_fresh(self):
        assert determine_freshness(0.9) == FragmentFreshness.FRESH
        assert determine_freshness(0.8) == FragmentFreshness.FRESH
        assert determine_freshness(1.0) == FragmentFreshness.FRESH

    def test_recent(self):
        assert determine_freshness(0.7) == FragmentFreshness.RECENT
        assert determine_freshness(0.6) == FragmentFreshness.RECENT

    def test_aging(self):
        assert determine_freshness(0.5) == FragmentFreshness.AGING
        assert determine_freshness(0.4) == FragmentFreshness.AGING

    def test_stale(self):
        assert determine_freshness(0.3) == FragmentFreshness.STALE
        assert determine_freshness(0.15) == FragmentFreshness.STALE

    def test_faded(self):
        assert determine_freshness(0.1) == FragmentFreshness.FADED
        assert determine_freshness(0.0) == FragmentFreshness.FADED


# =============================================================================
# IntrospectionFragment Tests
# =============================================================================

class TestIntrospectionFragment:
    def test_creation(self):
        f = _make_fragment()
        assert f.content == "test fragment"
        assert f.source_type == FragmentSourceType.INTROSPECTION_LOG
        assert f.freshness == 1.0
        assert f.reference_count == 0

    def test_frozen(self):
        f = _make_fragment()
        with pytest.raises(AttributeError):
            f.freshness = 0.5

    def test_get_freshness_level(self):
        assert _make_fragment(freshness=0.9).get_freshness_level() == FragmentFreshness.FRESH
        assert _make_fragment(freshness=0.5).get_freshness_level() == FragmentFreshness.AGING
        assert _make_fragment(freshness=0.1).get_freshness_level() == FragmentFreshness.FADED

    def test_with_freshness(self):
        f = _make_fragment(freshness=0.8)
        f2 = f.with_freshness(0.5)
        assert f2.freshness == 0.5
        assert f.freshness == 0.8  # original unchanged

    def test_with_freshness_clamped(self):
        f = _make_fragment()
        assert f.with_freshness(1.5).freshness == 1.0
        assert f.with_freshness(-0.5).freshness == 0.0

    def test_with_reference(self):
        f = _make_fragment(reference_count=2)
        f2 = f.with_reference()
        assert f2.reference_count == 3
        assert f.reference_count == 2  # original unchanged

    def test_recompose(self):
        f = _make_fragment(content="old content")
        f2 = f.recompose("new content", ("recomposed",))
        assert f2.content == "new content"
        assert f2.undetermined_aspects == ("recomposed",)
        assert f.content == "old content"  # original unchanged

    def test_recompose_preserves_id(self):
        f = _make_fragment(fragment_id="keep_this")
        f2 = f.recompose("new")
        assert f2.fragment_id == "keep_this"


# =============================================================================
# FragmentBundle Tests
# =============================================================================

class TestFragmentBundle:
    def test_creation(self):
        b = FragmentBundle(
            bundle_id="b1",
            fragment_ids=("f1", "f2"),
            theme_description="test bundle",
            coherence=BundleCoherence.TIGHT,
            timestamp=time.time(),
            strength=0.7,
        )
        assert b.bundle_id == "b1"
        assert len(b.fragment_ids) == 2
        assert b.coherence == BundleCoherence.TIGHT
        assert b.strength == 0.7

    def test_frozen(self):
        b = FragmentBundle("b1", ("f1",), "desc", BundleCoherence.LOOSE, 0.0, 0.5)
        with pytest.raises(AttributeError):
            b.strength = 0.9


# =============================================================================
# ConsumptionRecord Tests
# =============================================================================

class TestConsumptionRecord:
    def test_creation(self):
        r = ConsumptionRecord(
            record_id="r1",
            fragment_ids=("f1", "f2"),
            timestamp=time.time(),
            context_description="read for narrative",
        )
        assert r.record_id == "r1"
        assert len(r.fragment_ids) == 2
        assert r.context_description == "read for narrative"


# =============================================================================
# ConsumptionStore Tests
# =============================================================================

class TestConsumptionStore:
    def test_empty(self):
        store = ConsumptionStore.empty()
        assert not store.has_fragments()
        assert store.active_fragment_count == 0
        assert store.bundle_count == 0
        assert store.total_fragments_created == 0
        assert store.average_freshness == 0.0

    def test_has_fragments(self):
        store = ConsumptionStore(
            fragments=(_make_fragment(),),
            bundles=(),
            consumption_history=(),
            total_fragments_created=1,
            total_consumptions=0,
            average_freshness=1.0,
            active_fragment_count=1,
            bundle_count=0,
            timestamp=time.time(),
            description="test",
        )
        assert store.has_fragments()

    def test_get_active_fragments(self):
        f1 = _make_fragment(freshness=0.5, fragment_id="a1")
        f2 = _make_fragment(freshness=0.0, fragment_id="a2")
        store = ConsumptionStore(
            fragments=(f1, f2),
            bundles=(), consumption_history=(),
            total_fragments_created=2, total_consumptions=0,
            average_freshness=0.25, active_fragment_count=1,
            bundle_count=0, timestamp=time.time(), description="",
        )
        active = store.get_active_fragments()
        assert len(active) == 1
        assert active[0].fragment_id == "a1"

    def test_get_fresh_fragments(self):
        f1 = _make_fragment(freshness=0.9, fragment_id="fresh1")
        f2 = _make_fragment(freshness=0.3, fragment_id="stale1")
        store = ConsumptionStore(
            fragments=(f1, f2),
            bundles=(), consumption_history=(),
            total_fragments_created=2, total_consumptions=0,
            average_freshness=0.6, active_fragment_count=2,
            bundle_count=0, timestamp=time.time(), description="",
        )
        fresh = store.get_fresh_fragments()
        assert len(fresh) == 1
        assert fresh[0].fragment_id == "fresh1"

    def test_serialization_roundtrip(self):
        f1 = _make_fragment(freshness=0.7, fragment_id="s1", content="hello")
        b1 = FragmentBundle("b1", ("s1",), "theme", BundleCoherence.LOOSE, 100.0, 0.5)
        r1 = ConsumptionRecord("r1", ("s1",), 100.0, "ctx")
        store = ConsumptionStore(
            fragments=(f1,), bundles=(b1,), consumption_history=(r1,),
            total_fragments_created=1, total_consumptions=1,
            average_freshness=0.7, active_fragment_count=1,
            bundle_count=1, timestamp=100.0, description="test store",
        )
        d = store.to_dict()
        restored = ConsumptionStore.from_dict(d)
        assert len(restored.fragments) == 1
        assert restored.fragments[0].content == "hello"
        assert restored.fragments[0].freshness == 0.7
        assert len(restored.bundles) == 1
        assert restored.bundles[0].coherence == BundleCoherence.LOOSE
        assert len(restored.consumption_history) == 1
        assert restored.total_fragments_created == 1

    def test_serialization_json_roundtrip(self):
        store = ConsumptionStore.empty()
        data = json.dumps(store.to_dict())
        restored = ConsumptionStore.from_dict(json.loads(data))
        assert not restored.has_fragments()


# =============================================================================
# ConsumptionLayerConfig Tests
# =============================================================================

class TestConsumptionLayerConfig:
    def test_defaults(self):
        config = ConsumptionLayerConfig()
        assert config.max_fragments == 150
        assert config.base_decay_rate == 0.03
        assert config.bundle_strength_threshold == 0.3
        assert config.max_bundles == 30
        assert config.freshness_boost_on_reference == 0.12
        assert config.stale_threshold == 0.15
        assert config.max_consumption_history == 50

    def test_custom(self):
        config = create_config(max_fragments=50, base_decay_rate=0.1)
        assert config.max_fragments == 50
        assert config.base_decay_rate == 0.1


# =============================================================================
# Extract Functions Tests
# =============================================================================

class TestExtractFromIntrospection:
    def test_none_input(self):
        assert extract_from_introspection(None) == []

    def test_trace_log_object(self):
        class MockDecisionSnapshot:
            policy_label = "speak"
            outcome_type = type("OT", (), {"value": "speech"})()

        class MockFactor:
            name = "joy"
            contribution_strength = 0.8

        class MockTrace:
            trace_id = "trace123"
            contributing_factors = [MockFactor()]
            decision_snapshot = MockDecisionSnapshot()

        results = extract_from_introspection(MockTrace())
        assert len(results) >= 1
        assert "speak" in results[0][0]
        assert "joy" in results[0][0]

    def test_dict_input(self):
        d = {"policy_label": "silence", "outcome_type": "silence", "trace_id": "t1"}
        results = extract_from_introspection(d)
        assert len(results) >= 1
        assert "silence" in results[0][0]


class TestExtractFromNarrative:
    def test_none_input(self):
        assert extract_from_narrative(None) == []

    def test_narrative_state_object(self):
        class MockFragment:
            fragment_id = "nf1"
            description = "Emotional state changed"
            vividness = 0.8

        class MockCoherence:
            level = type("L", (), {"value": "coherent"})()
            average_vividness = 0.7

        class MockNarrative:
            fragments = [MockFragment()]
            coherence = MockCoherence()
            trend = type("T", (), {"value": "accumulating"})()

        results = extract_from_narrative(MockNarrative())
        assert len(results) >= 1

    def test_dict_input(self):
        d = {"has_narrative": True, "coherence_level": "coherent",
             "trend": "stable", "fragment_count": 5}
        results = extract_from_narrative(d)
        assert len(results) >= 1


class TestExtractFromCoherence:
    def test_none_input(self):
        assert extract_from_coherence(None) == []

    def test_coherence_state_object(self):
        class MockOverlap:
            active_count = 2
            intensity = type("I", (), {"value": "moderate"})()

        class MockCoherence:
            level = type("L", (), {"value": "slightly_shifting"})()
            shift_overlap = MockOverlap()
            trend = type("T", (), {"value": "diverging"})()
            description = "Some shift detected"

        results = extract_from_coherence(MockCoherence())
        assert len(results) >= 1
        assert "slightly_shifting" in results[0][0]

    def test_dict_input(self):
        d = {"level": "stable", "active_shift_count": 0, "description": "stable"}
        results = extract_from_coherence(d)
        assert len(results) >= 1


class TestExtractFromTendency:
    def test_none_input(self):
        assert extract_from_tendency(None) == []

    def test_no_awareness(self):
        class MockAwareness:
            has_awareness = False
            items = []

        assert extract_from_tendency(MockAwareness()) == []

    def test_with_awareness(self):
        class MockItem:
            description = "I tend to approach situations"

        class MockAwareness:
            has_awareness = True
            items = [MockItem()]
            overall_strength = type("S", (), {"value": "moderate"})()

        results = extract_from_tendency(MockAwareness())
        assert len(results) >= 1
        assert "approach" in results[0][0]

    def test_dict_input(self):
        d = {"has_tendency_awareness": True,
             "descriptions": ["habit forming"], "overall_strength": "strong"}
        results = extract_from_tendency(d)
        assert len(results) >= 1


class TestExtractFromEpisodic:
    def test_none_input(self):
        assert extract_from_episodic(None) == []

    def test_episode_store_object(self):
        class MockEpisode:
            episode_id = "ep1"
            summary = "Interaction occurred"
            episode_type = type("T", (), {"value": "interaction"})()

        class MockStore:
            average_vividness = 0.8
            active_episode_count = 3
            def get_fresh_episodes(self):
                return [MockEpisode()]

        results = extract_from_episodic(MockStore())
        assert len(results) >= 1

    def test_dict_input(self):
        d = {"has_episodes": True, "total_episodes": 5,
             "active_episode_count": 3, "average_vividness": 0.7}
        results = extract_from_episodic(d)
        assert len(results) >= 1


# =============================================================================
# Bundle Functions Tests
# =============================================================================

class TestComputeFragmentRelevance:
    def test_same_source_same_time(self):
        ts = time.time()
        a = _make_fragment(content="hello world test", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts, fragment_id="r1")
        b = _make_fragment(content="hello world test", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts, fragment_id="r2")
        rel = compute_fragment_relevance(a, b)
        assert rel >= 0.7  # Same source + same time + same content

    def test_different_everything(self):
        a = _make_fragment(content="alpha beta gamma", source_type=FragmentSourceType.INTROSPECTION_LOG, timestamp=1000.0, fragment_id="r3")
        b = _make_fragment(content="delta epsilon zeta", source_type=FragmentSourceType.EPISODIC_MEMORY, timestamp=2000.0, fragment_id="r4")
        rel = compute_fragment_relevance(a, b)
        assert rel < 0.3

    def test_temporal_proximity(self):
        ts = time.time()
        a = _make_fragment(content="x", timestamp=ts, fragment_id="r5")
        b = _make_fragment(content="y", timestamp=ts + 10, fragment_id="r6")
        rel = compute_fragment_relevance(a, b)
        # Same source but different content, close in time
        assert rel > 0.0

    def test_result_bounded(self):
        a = _make_fragment(content="same same", fragment_id="r7")
        b = _make_fragment(content="same same", fragment_id="r8")
        assert compute_fragment_relevance(a, b) <= 1.0


class TestDetermineBundleCoherence:
    def test_empty(self):
        assert determine_bundle_coherence((), {}) == BundleCoherence.UNDEFINED

    def test_single(self):
        assert determine_bundle_coherence(("f1",), {}) == BundleCoherence.UNDEFINED

    def test_tight(self):
        ts = time.time()
        fa = _make_fragment(content="hello world test data", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts, fragment_id="fa")
        fb = _make_fragment(content="hello world test data", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts, fragment_id="fb")
        frag_map = {"fa": fa, "fb": fb}
        coh = determine_bundle_coherence(("fa", "fb"), frag_map)
        assert coh == BundleCoherence.TIGHT


class TestGenerateBundles:
    def test_empty(self):
        config = ConsumptionLayerConfig()
        assert generate_bundles([], config) == []

    def test_single_fragment(self):
        config = ConsumptionLayerConfig()
        assert generate_bundles([_make_fragment()], config) == []

    def test_related_fragments_bundle(self):
        config = ConsumptionLayerConfig(bundle_strength_threshold=0.2)
        ts = time.time()
        frags = [
            _make_fragment(content="narrative coherence stable state", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts, fragment_id="gb1"),
            _make_fragment(content="narrative coherence stable state", source_type=FragmentSourceType.SELF_NARRATIVE, timestamp=ts + 1, fragment_id="gb2"),
        ]
        bundles = generate_bundles(frags, config)
        assert len(bundles) >= 1


# =============================================================================
# IntrospectionConsumptionSystem Tests
# =============================================================================

class TestIntrospectionConsumptionSystem:
    def test_creation(self):
        system = IntrospectionConsumptionSystem()
        store = system.get_store()
        assert not store.has_fragments()

    def test_consume_observations_no_input(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations()
        assert not store.has_fragments()

    def test_consume_with_introspection(self):
        system = IntrospectionConsumptionSystem()
        trace = {"policy_label": "speak", "outcome_type": "speech", "trace_id": "t1"}
        store = system.consume_observations(introspection_summary=trace)
        assert store.has_fragments()
        assert store.total_fragments_created > 0

    def test_consume_with_narrative(self):
        system = IntrospectionConsumptionSystem()
        narrative = {"has_narrative": True, "coherence_level": "coherent",
                     "trend": "stable", "fragment_count": 3}
        store = system.consume_observations(narrative_state=narrative)
        assert store.has_fragments()

    def test_consume_with_coherence(self):
        system = IntrospectionConsumptionSystem()
        coherence = {"level": "stable", "active_shift_count": 0, "description": "stable"}
        store = system.consume_observations(coherence_state=coherence)
        assert store.has_fragments()

    def test_consume_with_tendency(self):
        system = IntrospectionConsumptionSystem()
        tendency = {"has_tendency_awareness": True,
                    "descriptions": ["habit forming"], "overall_strength": "moderate"}
        store = system.consume_observations(tendency_awareness=tendency)
        assert store.has_fragments()

    def test_consume_with_episodic(self):
        system = IntrospectionConsumptionSystem()
        episodic = {"has_episodes": True, "total_episodes": 5,
                    "active_episode_count": 3, "average_vividness": 0.8}
        store = system.consume_observations(episodic_store=episodic)
        assert store.has_fragments()

    def test_consume_all_sources(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "speak", "outcome_type": "speech"},
            narrative_state={"has_narrative": True, "coherence_level": "coherent",
                             "trend": "stable", "fragment_count": 3},
            coherence_state={"level": "stable", "active_shift_count": 0},
            tendency_awareness={"has_tendency_awareness": True,
                                "descriptions": ["test"], "overall_strength": "moderate"},
            episodic_store={"has_episodes": True, "total_episodes": 2,
                            "active_episode_count": 2, "average_vividness": 0.5},
        )
        assert store.active_fragment_count >= 5

    def test_decay_fragments(self):
        system = IntrospectionConsumptionSystem()
        system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        store1 = system.get_store()
        initial_freshness = store1.average_freshness

        store2 = system.decay_fragments()
        assert store2.average_freshness < initial_freshness

    def test_reference_fragment_boosts_freshness(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        fid = store.fragments[0].fragment_id
        initial_freshness = store.fragments[0].freshness

        # Decay first
        system.decay_fragments()

        # Reference to boost
        system.reference_fragment(fid)
        store2 = system.get_store()
        referenced = [f for f in store2.fragments if f.fragment_id == fid]
        assert len(referenced) == 1
        assert referenced[0].reference_count == 1

    def test_mark_as_consumed(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        fid = store.fragments[0].fragment_id
        system.mark_as_consumed([fid], "for testing")
        store2 = system.get_store()
        assert store2.total_consumptions == 1
        assert len(store2.consumption_history) == 1

    def test_get_readable_fragments(self):
        system = IntrospectionConsumptionSystem()
        system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
            narrative_state={"has_narrative": True, "coherence_level": "coherent",
                             "trend": "stable", "fragment_count": 3},
        )
        readable = system.get_readable_fragments(max_count=5)
        assert len(readable) > 0
        # Should be sorted by freshness descending
        if len(readable) >= 2:
            assert readable[0].freshness >= readable[1].freshness

    def test_get_last_store(self):
        system = IntrospectionConsumptionSystem()
        assert system.get_last_store() is None
        system.consume_observations()
        assert system.get_last_store() is not None

    def test_rebundle(self):
        system = IntrospectionConsumptionSystem()
        system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        store = system.rebundle()
        assert isinstance(store, ConsumptionStore)

    def test_fragment_limit_enforced(self):
        config = ConsumptionLayerConfig(max_fragments=5)
        system = IntrospectionConsumptionSystem(config=config)
        for i in range(10):
            system.consume_observations(
                introspection_summary={"policy_label": f"test_{i}", "outcome_type": "speech", "trace_id": f"t{i}"},
            )
        store = system.get_store()
        assert len(store.fragments) <= 5

    def test_consumption_history_limit(self):
        config = ConsumptionLayerConfig(max_consumption_history=3)
        system = IntrospectionConsumptionSystem(config=config)
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        fid = store.fragments[0].fragment_id
        for i in range(5):
            system.mark_as_consumed([fid], f"ctx_{i}")
        store2 = system.get_store()
        assert len(store2.consumption_history) <= 3

    def test_repeated_decay_removes_fragments(self):
        config = ConsumptionLayerConfig(base_decay_rate=0.5)
        system = IntrospectionConsumptionSystem(config=config)
        system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        # Decay several times until fragments are gone
        for _ in range(5):
            system.decay_fragments()
        store = system.get_store()
        assert store.active_fragment_count == 0

    def test_reference_slows_decay(self):
        config = ConsumptionLayerConfig(base_decay_rate=0.1)
        system = IntrospectionConsumptionSystem(config=config)
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        fid = store.fragments[0].fragment_id

        # Reference it many times
        for _ in range(5):
            system.reference_fragment(fid)

        # Decay
        store2 = system.decay_fragments()
        referenced = [f for f in store2.fragments if f.fragment_id == fid]
        assert len(referenced) == 1
        # Freshness should still be relatively high due to references
        assert referenced[0].freshness > 0.5


# =============================================================================
# Integration Functions Tests
# =============================================================================

class TestConsumeFromChain:
    def test_basic(self):
        system = IntrospectionConsumptionSystem()
        store = consume_from_chain(
            system,
            introspection_summary={"policy_label": "speak", "outcome_type": "speech"},
        )
        assert store.has_fragments()


class TestGenerateConsumptionTags:
    def test_empty_store(self):
        store = ConsumptionStore.empty()
        tags = generate_consumption_tags(store)
        assert len(tags) == 1
        assert tags[0]["label"] == "no_fragments"

    def test_with_fragments(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        tags = generate_consumption_tags(store)
        assert len(tags) >= 3  # count, freshness, recent, integrated

    def test_scale_parameter(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        tags1 = generate_consumption_tags(store, scale=1.0)
        tags2 = generate_consumption_tags(store, scale=2.0)
        assert tags2[0]["weight"] > tags1[0]["weight"]

    def test_with_consumption_activity(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        fid = store.fragments[0].fragment_id
        system.mark_as_consumed([fid], "test")
        store2 = system.get_store()
        tags = generate_consumption_tags(store2)
        activity_tags = [t for t in tags if t["category"] == "INTROSPECTION_CONSUMPTION_ACTIVITY"]
        assert len(activity_tags) == 1


class TestGetConsumptionSummary:
    def test_empty(self):
        store = ConsumptionStore.empty()
        summary = get_consumption_summary(store)
        assert "Introspection Consumption Layer" in summary

    def test_with_data(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        summary = get_consumption_summary(store)
        assert "Active fragments" in summary
        assert "Average freshness" in summary


class TestGetConsumptionForIntrospection:
    def test_empty(self):
        store = ConsumptionStore.empty()
        result = get_consumption_for_introspection(store)
        assert result["has_fragments"] is False
        assert result["total_fragments"] == 0

    def test_with_data(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        result = get_consumption_for_introspection(store)
        assert result["has_fragments"] is True
        assert result["total_fragments"] > 0
        assert "source_distribution" in result
        assert "freshness_distribution" in result
        assert "description" in result
        assert "timestamp" in result


# =============================================================================
# Verification Tests
# =============================================================================

class TestVerification:
    def test_no_decision_impact(self):
        store = ConsumptionStore.empty()
        assert verify_no_decision_impact(store) is True

    def test_no_decision_impact_with_data(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        assert verify_no_decision_impact(store) is True

    def test_no_goal_generation(self):
        system = IntrospectionConsumptionSystem()
        assert verify_no_goal_generation(system) is True

    def test_read_only_principle(self):
        system = IntrospectionConsumptionSystem()
        assert verify_read_only_principle(system) is True

    def test_no_value_modification(self):
        system = IntrospectionConsumptionSystem()
        assert verify_no_value_modification(system) is True


# =============================================================================
# Convenience / Persistence Tests
# =============================================================================

class TestConvenience:
    def test_create_config(self):
        config = create_config(max_fragments=50)
        assert config.max_fragments == 50

    def test_create_empty_store(self):
        store = create_empty_store()
        assert not store.has_fragments()

    def test_create_system(self):
        system = create_system()
        assert system.get_last_store() is None

    def test_create_system_with_config(self):
        config = create_config(max_fragments=10)
        system = create_system(config)
        store = system.get_store()
        assert isinstance(store, ConsumptionStore)


class TestPersistence:
    def test_save_and_load(self):
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            filepath = f.name

        try:
            save_consumption_state(store, filepath)
            loaded = load_consumption_state(filepath)
            assert loaded.has_fragments()
            assert loaded.total_fragments_created == store.total_fragments_created
            assert len(loaded.fragments) == len(store.fragments)
            assert loaded.fragments[0].content == store.fragments[0].content
        finally:
            os.unlink(filepath)

    def test_save_empty_store(self):
        store = ConsumptionStore.empty()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            filepath = f.name

        try:
            save_consumption_state(store, filepath)
            loaded = load_consumption_state(filepath)
            assert not loaded.has_fragments()
        finally:
            os.unlink(filepath)


# =============================================================================
# Design Principle Tests
# =============================================================================

class TestDesignPrinciples:
    """Tests verifying adherence to design principles."""

    def test_no_judgment_in_descriptions(self):
        """Verify descriptions don't contain evaluative language."""
        system = IntrospectionConsumptionSystem()
        store = system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
            narrative_state={"has_narrative": True, "coherence_level": "fragmented",
                             "trend": "dissolving", "fragment_count": 1},
        )
        forbidden = [
            "good", "bad", "correct", "incorrect", "wrong", "right",
            "should", "must", "need to fix", "proper",
        ]
        desc = store.description.lower()
        for word in forbidden:
            assert word not in desc, f"Description contains '{word}'"

    def test_no_goal_methods_exist(self):
        """Verify no goal-generating methods exist."""
        system = IntrospectionConsumptionSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        goal_methods = [m for m in methods if "goal" in m.lower()]
        assert len(goal_methods) == 0

    def test_no_decision_methods_exist(self):
        """Verify no decision-making methods exist."""
        system = IntrospectionConsumptionSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        decision_methods = [m for m in methods if "decide" in m.lower() or "choose" in m.lower()]
        assert len(decision_methods) == 0

    def test_fragments_are_immutable(self):
        """Verify fragments are frozen dataclasses."""
        f = _make_fragment()
        with pytest.raises(AttributeError):
            f.content = "changed"
        with pytest.raises(AttributeError):
            f.freshness = 0.0

    def test_store_is_immutable(self):
        """Verify ConsumptionStore is frozen."""
        store = ConsumptionStore.empty()
        with pytest.raises(AttributeError):
            store.description = "changed"

    def test_bundle_is_immutable(self):
        """Verify FragmentBundle is frozen."""
        b = FragmentBundle("b1", ("f1",), "desc", BundleCoherence.LOOSE, 0.0, 0.5)
        with pytest.raises(AttributeError):
            b.strength = 0.9

    def test_record_is_immutable(self):
        """Verify ConsumptionRecord is frozen."""
        r = ConsumptionRecord("r1", ("f1",), 0.0, "ctx")
        with pytest.raises(AttributeError):
            r.context_description = "changed"

    def test_recompose_allows_different_interpretation(self):
        """Verify recompose creates new interpretation without fixing."""
        f = _make_fragment(content="original interpretation")
        f2 = f.recompose("different interpretation")
        assert f2.content != f.content
        assert f2.fragment_id == f.fragment_id

    def test_bundle_not_fixed(self):
        """Verify rebundling produces potentially different results."""
        system = IntrospectionConsumptionSystem()
        system.consume_observations(
            introspection_summary={"policy_label": "test", "outcome_type": "speech"},
        )
        store1 = system.rebundle()
        store2 = system.rebundle()
        # Both calls should succeed (bundles are regenerated, not fixed)
        assert isinstance(store1, ConsumptionStore)
        assert isinstance(store2, ConsumptionStore)

    def test_duck_typing_no_import_required(self):
        """Verify inputs work via duck typing without importing source types."""
        system = IntrospectionConsumptionSystem()
        # Using plain dicts (duck typing) should work fine
        store = system.consume_observations(
            introspection_summary={"policy_label": "x"},
            narrative_state={"has_narrative": True, "coherence_level": "c",
                             "trend": "s", "fragment_count": 1},
            coherence_state={"level": "s", "active_shift_count": 0},
            tendency_awareness={"has_tendency_awareness": False},
            episodic_store={"has_episodes": False},
        )
        assert isinstance(store, ConsumptionStore)
