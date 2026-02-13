"""Tests for psyche/memory_system_integration.py."""

import time
import pytest

from psyche.memory_system_integration import (
    MemorySource,
    TemporalPhase,
    UnifiedMemoryUnit,
    DuplicateEntry,
    ConflictEntry,
    ReferenceHistoryEntry,
    IntegrationContext,
    IntegrationConfig,
    IntegrationState,
    IntegrationResult,
    MemorySystemIntegrator,
    create_integrator,
    create_config,
    get_integration_summary,
    get_integration_summary_text,
    normalize_episodic,
    normalize_long_term,
    normalize_bindings,
    detect_duplicates,
    detect_conflicts,
    check_conflict_health,
    _make_unit_id,
    _determine_temporal_phase,
    _compute_relevance,
    _estimate_valence_from_keywords,
    _compute_topic_overlap,
    _compute_summary_similarity,
    _rank_units,
    _ensure_source_diversity,
)


# ── Enums ────────────────────────────────────────────────────


class TestMemorySource:
    def test_values(self):
        assert MemorySource.EPISODIC.value == "episodic"
        assert MemorySource.LONG_TERM.value == "long_term"
        assert MemorySource.BINDING.value == "binding"

    def test_all_members(self):
        assert len(MemorySource) == 3


class TestTemporalPhase:
    def test_values(self):
        assert TemporalPhase.IMMEDIATE.value == "immediate"
        assert TemporalPhase.RECENT.value == "recent"
        assert TemporalPhase.MEDIUM.value == "medium"
        assert TemporalPhase.DISTANT.value == "distant"

    def test_all_members(self):
        assert len(TemporalPhase) == 4


# ── UnifiedMemoryUnit ────────────────────────────────────────


class TestUnifiedMemoryUnit:
    def test_defaults(self):
        u = UnifiedMemoryUnit()
        assert u.unit_id == ""
        assert u.source == MemorySource.EPISODIC
        assert u.certainty == 0.5
        assert u.relevance == 0.0

    def test_to_dict_from_dict(self):
        u = UnifiedMemoryUnit(
            unit_id="abc",
            source=MemorySource.LONG_TERM,
            summary="test summary",
            topics=["a", "b"],
            temporal_phase=TemporalPhase.RECENT,
            timestamp=1000.0,
            certainty=0.8,
            relevance=0.6,
            emotional_valence=-0.3,
        )
        d = u.to_dict()
        u2 = UnifiedMemoryUnit.from_dict(d)
        assert u2.unit_id == "abc"
        assert u2.source == MemorySource.LONG_TERM
        assert u2.temporal_phase == TemporalPhase.RECENT
        assert u2.certainty == 0.8
        assert u2.emotional_valence == pytest.approx(-0.3)

    def test_from_dict_invalid_source(self):
        u = UnifiedMemoryUnit.from_dict({"source": "invalid"})
        assert u.source == MemorySource.EPISODIC

    def test_from_dict_invalid_phase(self):
        u = UnifiedMemoryUnit.from_dict({"temporal_phase": "invalid"})
        assert u.temporal_phase == TemporalPhase.DISTANT

    def test_to_memory_dict(self):
        u = UnifiedMemoryUnit(
            unit_id="x",
            summary="hello",
            topics=["a"],
            importance=0.8,
            source=MemorySource.EPISODIC,
            certainty=0.9,
        )
        md = u.to_memory_dict()
        assert md["summary"] == "hello"
        assert md["keywords"] == ["a"]
        assert md["importance"] == 4  # int(0.8 * 5)
        assert md["_source"] == "episodic"
        assert md["_integrated"] is True


# ── DuplicateEntry ───────────────────────────────────────────


class TestDuplicateEntry:
    def test_to_dict_from_dict(self):
        d = DuplicateEntry(
            group_id="g1",
            unit_ids=["a", "b"],
            sources=["episodic", "long_term"],
            similarity=0.7,
            topic_overlap=0.5,
        )
        data = d.to_dict()
        d2 = DuplicateEntry.from_dict(data)
        assert d2.group_id == "g1"
        assert d2.similarity == 0.7


# ── ConflictEntry ────────────────────────────────────────────


class TestConflictEntry:
    def test_to_dict_from_dict(self):
        c = ConflictEntry(
            conflict_id="c1",
            unit_id_a="a",
            unit_id_b="b",
            conflict_type="valence_mismatch",
            severity=0.8,
            visible=True,
        )
        data = c.to_dict()
        c2 = ConflictEntry.from_dict(data)
        assert c2.conflict_type == "valence_mismatch"
        assert c2.visible is True

    def test_defaults(self):
        c = ConflictEntry()
        assert c.turn_hidden == 0
        assert c.visible is True


# ── ReferenceHistoryEntry ────────────────────────────────────


class TestReferenceHistoryEntry:
    def test_to_dict_from_dict(self):
        r = ReferenceHistoryEntry(
            unit_id="u1", turn=5, relevance_at_ref=0.8, decay_factor=0.9
        )
        data = r.to_dict()
        r2 = ReferenceHistoryEntry.from_dict(data)
        assert r2.unit_id == "u1"
        assert r2.decay_factor == 0.9


# ── IntegrationContext ───────────────────────────────────────


class TestIntegrationContext:
    def test_to_dict_from_dict(self):
        ctx = IntegrationContext(
            emotions={"joy": 0.7},
            mood_valence=0.5,
            percept_topics=["game"],
            percept_text="楽しいゲーム",
            percept_intent="sharing",
            tick_count=10,
        )
        data = ctx.to_dict()
        ctx2 = IntegrationContext.from_dict(data)
        assert ctx2.emotions == {"joy": 0.7}
        assert ctx2.percept_intent == "sharing"
        assert ctx2.tick_count == 10


# ── IntegrationConfig ────────────────────────────────────────


class TestIntegrationConfig:
    def test_defaults(self):
        cfg = IntegrationConfig()
        assert cfg.max_unified_units == 30
        assert cfg.max_output_candidates == 10
        assert cfg.single_source_cap == 0.6

    def test_to_dict_from_dict(self):
        cfg = IntegrationConfig(max_unified_units=50)
        data = cfg.to_dict()
        cfg2 = IntegrationConfig.from_dict(data)
        assert cfg2.max_unified_units == 50

    def test_factory(self):
        cfg = create_config(max_output_candidates=20)
        assert cfg.max_output_candidates == 20


# ── IntegrationState ─────────────────────────────────────────


class TestIntegrationState:
    def test_to_dict_from_dict(self):
        state = IntegrationState(
            turn_count=5,
            total_integrations=100,
            convergence_warnings=2,
        )
        state.duplicate_table.append(
            DuplicateEntry(group_id="g1", similarity=0.5)
        )
        state.conflict_table.append(
            ConflictEntry(conflict_id="c1", severity=0.6)
        )
        data = state.to_dict()
        s2 = IntegrationState.from_dict(data)
        assert s2.turn_count == 5
        assert s2.total_integrations == 100
        assert len(s2.duplicate_table) == 1
        assert len(s2.conflict_table) == 1


# ── IntegrationResult ────────────────────────────────────────


class TestIntegrationResult:
    def test_to_dict_from_dict(self):
        r = IntegrationResult(
            candidates=[UnifiedMemoryUnit(unit_id="u1", summary="test")],
            convergence_warning=True,
            source_distribution={"episodic": 1},
        )
        data = r.to_dict()
        r2 = IntegrationResult.from_dict(data)
        assert len(r2.candidates) == 1
        assert r2.convergence_warning is True

    def test_to_memory_list(self):
        r = IntegrationResult(
            candidates=[
                UnifiedMemoryUnit(unit_id="u1", summary="s1"),
                UnifiedMemoryUnit(unit_id="u2", summary="s2"),
            ]
        )
        mem_list = r.to_memory_list()
        assert len(mem_list) == 2
        assert mem_list[0]["summary"] == "s1"
        assert mem_list[1]["_integrated"] is True


# ── Helper Functions ─────────────────────────────────────────


class TestMakeUnitId:
    def test_deterministic(self):
        id1 = _make_unit_id("episodic", "e1")
        id2 = _make_unit_id("episodic", "e1")
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _make_unit_id("episodic", "e1")
        id2 = _make_unit_id("long_term", "e1")
        assert id1 != id2

    def test_length(self):
        uid = _make_unit_id("test", "abc")
        assert len(uid) == 12


class TestDetermineTemporalPhase:
    def test_immediate(self):
        cfg = IntegrationConfig()
        now = time.time()
        p = _determine_temporal_phase(now - 60, now, cfg)
        assert p == TemporalPhase.IMMEDIATE

    def test_recent(self):
        cfg = IntegrationConfig()
        now = time.time()
        p = _determine_temporal_phase(now - 1800, now, cfg)
        assert p == TemporalPhase.RECENT

    def test_medium(self):
        cfg = IntegrationConfig()
        now = time.time()
        p = _determine_temporal_phase(now - 43200, now, cfg)
        assert p == TemporalPhase.MEDIUM

    def test_distant(self):
        cfg = IntegrationConfig()
        now = time.time()
        p = _determine_temporal_phase(now - 200000, now, cfg)
        assert p == TemporalPhase.DISTANT

    def test_negative_elapsed(self):
        cfg = IntegrationConfig()
        now = time.time()
        p = _determine_temporal_phase(now + 1000, now, cfg)
        assert p == TemporalPhase.IMMEDIATE


class TestComputeRelevance:
    def test_topic_overlap(self):
        ctx = IntegrationContext(
            percept_topics=["game", "fun"],
            mood_valence=0.0,
        )
        r = _compute_relevance(["game", "other"], 0.0, ctx)
        assert r > 0.0

    def test_text_match(self):
        ctx = IntegrationContext(
            percept_text="楽しいゲーム",
            mood_valence=0.0,
        )
        r = _compute_relevance(["ゲーム"], 0.0, ctx)
        assert r > 0.0

    def test_mood_congruence(self):
        ctx = IntegrationContext(mood_valence=0.8)
        r = _compute_relevance([], 0.7, ctx)
        assert r > 0.0

    def test_no_relevance(self):
        ctx = IntegrationContext()
        r = _compute_relevance([], 0.0, ctx)
        assert r == 0.0

    def test_capped_at_one(self):
        ctx = IntegrationContext(
            percept_topics=["a", "b", "c"],
            percept_text="a b c d e",
            mood_valence=1.0,
        )
        r = _compute_relevance(["a", "b", "c", "d", "e"], 1.0, ctx)
        assert r <= 1.0


class TestEstimateValence:
    def test_positive(self):
        v = _estimate_valence_from_keywords(["happy", "fun"])
        assert v > 0.0

    def test_negative(self):
        v = _estimate_valence_from_keywords(["sad", "angry"])
        assert v < 0.0

    def test_neutral(self):
        v = _estimate_valence_from_keywords(["weather", "tree"])
        assert v == 0.0

    def test_clamped(self):
        v = _estimate_valence_from_keywords(["happy"] * 20)
        assert v <= 1.0


class TestTopicOverlap:
    def test_full_overlap(self):
        o = _compute_topic_overlap(["a", "b"], ["a", "b"])
        assert o == pytest.approx(1.0)

    def test_no_overlap(self):
        o = _compute_topic_overlap(["a"], ["b"])
        assert o == pytest.approx(0.0)

    def test_partial_overlap(self):
        o = _compute_topic_overlap(["a", "b"], ["b", "c"])
        assert 0.0 < o < 1.0

    def test_empty(self):
        assert _compute_topic_overlap([], ["a"]) == 0.0
        assert _compute_topic_overlap(["a"], []) == 0.0

    def test_case_insensitive(self):
        o = _compute_topic_overlap(["Game"], ["game"])
        assert o == pytest.approx(1.0)


class TestSummarySimilarity:
    def test_identical(self):
        s = _compute_summary_similarity("hello world", "hello world")
        assert s == pytest.approx(1.0)

    def test_no_match(self):
        s = _compute_summary_similarity("hello", "goodbye")
        assert s == pytest.approx(0.0)

    def test_empty(self):
        assert _compute_summary_similarity("", "hello") == 0.0
        assert _compute_summary_similarity("hello", "") == 0.0


# ── Normalization ────────────────────────────────────────────


class _MockEpisodeEntry:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockEpisodeStore:
    def __init__(self, entries=None):
        self.entries = entries or []


class _MockEmotionalCompanion:
    def __init__(self, valence=0.0, primary_emotion=""):
        self.valence = valence
        self.primary_emotion = primary_emotion


class TestNormalizeEpisodic:
    def test_basic_normalization(self):
        entry = _MockEpisodeEntry(
            episode_id="ep1",
            summary="Test episode",
            topics=["game"],
            timestamp=time.time() - 60,
            vividness=0.8,
            reference_count=2,
            importance=None,
            emotional_companion=_MockEmotionalCompanion(0.5, "joy"),
        )
        store = _MockEpisodeStore([entry])
        ctx = IntegrationContext(current_time=time.time())
        cfg = IntegrationConfig()
        units = normalize_episodic(store, ctx, cfg)
        assert len(units) == 1
        assert units[0].source == MemorySource.EPISODIC
        assert units[0].summary == "Test episode"
        assert units[0].emotional_valence == 0.5

    def test_empty_store(self):
        store = _MockEpisodeStore([])
        ctx = IntegrationContext()
        cfg = IntegrationConfig()
        units = normalize_episodic(store, ctx, cfg)
        assert units == []

    def test_skip_empty_summary(self):
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="", topics=[], timestamp=0,
            vividness=0.5, reference_count=0, importance=None,
            emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        units = normalize_episodic(store, IntegrationContext(), IntegrationConfig())
        assert units == []

    def test_importance_enum_mapping(self):
        class MockImportance:
            value = "notable"
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="Test", topics=[], timestamp=0,
            vividness=0.5, reference_count=0,
            importance=MockImportance(),
            emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        units = normalize_episodic(store, IntegrationContext(), IntegrationConfig())
        assert units[0].importance == pytest.approx(0.7)

    def test_none_episodes(self):
        units = normalize_episodic(None, IntegrationContext(), IntegrationConfig())
        assert units == []


class TestNormalizeLongTerm:
    def test_basic(self):
        memories = [
            {"id": "m1", "summary": "Long term memory", "keywords": ["test"],
             "importance": 4, "date": str(time.time() - 3600)},
        ]
        ctx = IntegrationContext(current_time=time.time())
        cfg = IntegrationConfig()
        units = normalize_long_term(memories, ctx, cfg)
        assert len(units) == 1
        assert units[0].source == MemorySource.LONG_TERM
        assert units[0].importance == pytest.approx(0.8)

    def test_empty(self):
        assert normalize_long_term(None, IntegrationContext(), IntegrationConfig()) == []
        assert normalize_long_term([], IntegrationContext(), IntegrationConfig()) == []

    def test_skip_empty_summary(self):
        memories = [{"id": "m1", "summary": ""}]
        units = normalize_long_term(memories, IntegrationContext(), IntegrationConfig())
        assert units == []

    def test_iso_date_parsing(self):
        memories = [
            {"id": "m1", "summary": "Test", "keywords": [],
             "importance": 3, "date": "2024-01-15T10:00:00"},
        ]
        units = normalize_long_term(memories, IntegrationContext(), IntegrationConfig())
        assert len(units) == 1
        assert units[0].timestamp > 0


class _MockBindingTrace:
    def __init__(self, intensity=0.5, freshness=0.8, valence=0.3, emotion_label="joy"):
        self.intensity = intensity
        self.freshness = freshness
        self.valence = valence
        self.emotion_label = emotion_label


class _MockBinding:
    def __init__(self, binding_id="b1", memory_summary="bound memory",
                 freshness=0.7, reference_count=1, traces=None,
                 creation_timestamp=""):
        self.binding_id = binding_id
        self.memory_summary = memory_summary
        self.freshness = freshness
        self.reference_count = reference_count
        self.traces = traces or []
        self.creation_timestamp = creation_timestamp


class _MockBindingStore:
    def __init__(self, bindings=None):
        self.bindings = bindings or []


class TestNormalizeBindings:
    def test_basic(self):
        trace = _MockBindingTrace(intensity=0.8, freshness=0.9, valence=0.5, emotion_label="joy")
        binding = _MockBinding(traces=[trace])
        store = _MockBindingStore([binding])
        units = normalize_bindings(store, IntegrationContext(), IntegrationConfig())
        assert len(units) == 1
        assert units[0].source == MemorySource.BINDING
        assert units[0].emotional_valence == 0.5

    def test_empty(self):
        assert normalize_bindings(None, IntegrationContext(), IntegrationConfig()) == []

    def test_skip_empty_summary(self):
        binding = _MockBinding(memory_summary="")
        store = _MockBindingStore([binding])
        units = normalize_bindings(store, IntegrationContext(), IntegrationConfig())
        assert units == []


# ── Duplicate Detection ──────────────────────────────────────


class TestDetectDuplicates:
    def test_detects_cross_source_duplicates(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", source=MemorySource.EPISODIC,
                topics=["game", "fun"], summary="playing a game",
            ),
            UnifiedMemoryUnit(
                unit_id="u2", source=MemorySource.LONG_TERM,
                topics=["game", "fun"], summary="game session",
            ),
        ]
        cfg = IntegrationConfig(topic_similarity_threshold=0.3)
        dups = detect_duplicates(units, cfg)
        assert len(dups) >= 1

    def test_ignores_same_source(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", source=MemorySource.EPISODIC,
                topics=["game"], summary="game1",
            ),
            UnifiedMemoryUnit(
                unit_id="u2", source=MemorySource.EPISODIC,
                topics=["game"], summary="game2",
            ),
        ]
        cfg = IntegrationConfig(topic_similarity_threshold=0.3)
        dups = detect_duplicates(units, cfg)
        assert len(dups) == 0

    def test_no_overlap(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", source=MemorySource.EPISODIC,
                topics=["game"], summary="game",
            ),
            UnifiedMemoryUnit(
                unit_id="u2", source=MemorySource.LONG_TERM,
                topics=["cooking"], summary="cooking",
            ),
        ]
        cfg = IntegrationConfig(topic_similarity_threshold=0.5)
        dups = detect_duplicates(units, cfg)
        assert len(dups) == 0

    def test_respects_max_limit(self):
        units = []
        for i in range(10):
            src = MemorySource.EPISODIC if i % 2 == 0 else MemorySource.LONG_TERM
            units.append(UnifiedMemoryUnit(
                unit_id=f"u{i}", source=src,
                topics=["shared_topic"], summary="same thing",
            ))
        cfg = IntegrationConfig(max_duplicates=3, topic_similarity_threshold=0.1)
        dups = detect_duplicates(units, cfg)
        assert len(dups) <= 3


# ── Conflict Detection ───────────────────────────────────────


class TestDetectConflicts:
    def test_valence_mismatch(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", topics=["game"],
                emotional_valence=0.8,
            ),
            UnifiedMemoryUnit(
                unit_id="u2", topics=["game"],
                emotional_valence=-0.5,
            ),
        ]
        cfg = IntegrationConfig(conflict_valence_threshold=0.5)
        conflicts = detect_conflicts(units, cfg)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "valence_mismatch"

    def test_importance_gap(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", source=MemorySource.EPISODIC,
                topics=["game", "fun"], importance=0.9,
            ),
            UnifiedMemoryUnit(
                unit_id="u2", source=MemorySource.LONG_TERM,
                topics=["game", "fun"], importance=0.2,
            ),
        ]
        cfg = IntegrationConfig(conflict_importance_threshold=0.4)
        conflicts = detect_conflicts(units, cfg)
        assert any(c.conflict_type == "importance_gap" for c in conflicts)

    def test_no_conflict(self):
        units = [
            UnifiedMemoryUnit(
                unit_id="u1", topics=["game"],
                emotional_valence=0.5, importance=0.5,
            ),
            UnifiedMemoryUnit(
                unit_id="u2", topics=["cooking"],
                emotional_valence=0.4, importance=0.6,
            ),
        ]
        cfg = IntegrationConfig()
        conflicts = detect_conflicts(units, cfg)
        assert len(conflicts) == 0


# ── Ranking ──────────────────────────────────────────────────


class TestRanking:
    def test_higher_relevance_first(self):
        units = [
            UnifiedMemoryUnit(unit_id="low", relevance=0.1),
            UnifiedMemoryUnit(unit_id="high", relevance=0.9),
        ]
        ranked = _rank_units(units, {}, [], IntegrationConfig())
        assert ranked[0].unit_id == "high"

    def test_recency_suppression(self):
        units = [
            UnifiedMemoryUnit(unit_id="u1", relevance=0.5),
            UnifiedMemoryUnit(unit_id="u2", relevance=0.5),
        ]
        history = [
            ReferenceHistoryEntry(unit_id="u1", turn=i) for i in range(5)
        ]
        cfg = IntegrationConfig(recency_suppression_count=3)
        ranked = _rank_units(units, {}, history, cfg)
        # u1 should be suppressed, u2 should be first
        assert ranked[0].unit_id == "u2"


# ── Source Diversity ─────────────────────────────────────────


class TestSourceDiversity:
    def test_limits_single_source(self):
        units = [
            UnifiedMemoryUnit(unit_id=f"e{i}", source=MemorySource.EPISODIC)
            for i in range(8)
        ] + [
            UnifiedMemoryUnit(unit_id="lt1", source=MemorySource.LONG_TERM),
            UnifiedMemoryUnit(unit_id="lt2", source=MemorySource.LONG_TERM),
        ]
        cfg = IntegrationConfig(single_source_cap=0.6, max_output_candidates=10)
        selected, warning = _ensure_source_diversity(units, cfg)
        # Episodic shouldn't dominate
        ep_count = sum(1 for u in selected if u.source == MemorySource.EPISODIC)
        assert ep_count / len(selected) <= 0.7  # some tolerance

    def test_empty_input(self):
        selected, warning = _ensure_source_diversity([], IntegrationConfig())
        assert selected == []
        assert warning is False

    def test_convergence_warning(self):
        units = [
            UnifiedMemoryUnit(unit_id=f"e{i}", source=MemorySource.EPISODIC)
            for i in range(10)
        ]
        cfg = IntegrationConfig(single_source_cap=0.5, max_output_candidates=5)
        selected, warning = _ensure_source_diversity(units, cfg)
        assert warning is True


# ── Conflict Health Check ────────────────────────────────────


class TestConflictHealthCheck:
    def test_restores_hidden_conflict(self):
        state = IntegrationState()
        state.config.conflict_hidden_restore_turns = 3
        conflict = ConflictEntry(
            conflict_id="c1", visible=False, turn_hidden=2,
        )
        state.conflict_table.append(conflict)
        check_conflict_health(state)
        assert conflict.visible is True
        assert conflict.turn_hidden == 0

    def test_does_not_restore_too_early(self):
        state = IntegrationState()
        state.config.conflict_hidden_restore_turns = 5
        conflict = ConflictEntry(
            conflict_id="c1", visible=False, turn_hidden=1,
        )
        state.conflict_table.append(conflict)
        check_conflict_health(state)
        assert conflict.visible is False
        assert conflict.turn_hidden == 2

    def test_skips_visible_conflicts(self):
        state = IntegrationState()
        conflict = ConflictEntry(conflict_id="c1", visible=True, turn_hidden=0)
        state.conflict_table.append(conflict)
        check_conflict_health(state)
        assert conflict.turn_hidden == 0


# ── MemorySystemIntegrator ───────────────────────────────────


class TestMemorySystemIntegrator:
    def test_create_integrator(self):
        integrator = create_integrator()
        assert isinstance(integrator, MemorySystemIntegrator)
        assert integrator.state.turn_count == 0

    def test_integrate_empty(self):
        integrator = create_integrator()
        result = integrator.integrate()
        assert isinstance(result, IntegrationResult)
        assert len(result.candidates) == 0
        assert integrator.state.turn_count == 1

    def test_integrate_with_long_term_memories(self):
        integrator = create_integrator()
        memories = [
            {"id": "m1", "summary": "Test memory 1", "keywords": ["test"],
             "importance": 4, "date": str(time.time() - 600)},
            {"id": "m2", "summary": "Test memory 2", "keywords": ["other"],
             "importance": 3, "date": str(time.time() - 1200)},
        ]
        ctx = IntegrationContext(
            percept_topics=["test"],
            current_time=time.time(),
        )
        result = integrator.integrate(
            long_term_memories=memories,
            context=ctx,
        )
        assert len(result.candidates) >= 1
        assert "long_term" in result.source_distribution

    def test_integrate_with_episodic(self):
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="Episode 1", topics=["game"],
            timestamp=time.time() - 120, vividness=0.8, reference_count=1,
            importance=None, emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        integrator = create_integrator()
        result = integrator.integrate(
            episodes=store,
            context=IntegrationContext(current_time=time.time()),
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].source == MemorySource.EPISODIC

    def test_integrate_mixed_sources(self):
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="Episode game", topics=["game"],
            timestamp=time.time() - 120, vividness=0.8, reference_count=0,
            importance=None,
            emotional_companion=_MockEmotionalCompanion(0.5, "joy"),
        )
        store = _MockEpisodeStore([entry])
        memories = [
            {"id": "m1", "summary": "Long term game", "keywords": ["game"],
             "importance": 4, "date": str(time.time() - 600)},
        ]
        integrator = create_integrator()
        ctx = IntegrationContext(
            percept_topics=["game"],
            current_time=time.time(),
        )
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=ctx,
        )
        assert len(result.candidates) == 2
        sources = {c.source.value for c in result.candidates}
        assert "episodic" in sources
        assert "long_term" in sources

    def test_duplicate_detection_in_integrate(self):
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="playing a game together",
            topics=["game", "fun"], timestamp=time.time() - 120,
            vividness=0.8, reference_count=0, importance=None,
            emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        memories = [
            {"id": "m1", "summary": "we played a game together",
             "keywords": ["game", "fun"], "importance": 4,
             "date": str(time.time() - 600)},
        ]
        integrator = create_integrator()
        ctx = IntegrationContext(current_time=time.time())
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=ctx,
        )
        assert len(result.duplicate_groups) >= 1

    def test_turn_count_increments(self):
        integrator = create_integrator()
        integrator.integrate()
        integrator.integrate()
        integrator.integrate()
        assert integrator.state.turn_count == 3

    def test_reference_history_updated(self):
        memories = [
            {"id": "m1", "summary": "Test", "keywords": ["test"],
             "importance": 3, "date": str(time.time())},
        ]
        integrator = create_integrator()
        integrator.integrate(
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        assert len(integrator.state.reference_history) > 0

    def test_reuse_history_updated(self):
        memories = [
            {"id": "m1", "summary": "Test", "keywords": ["test"],
             "importance": 3, "date": str(time.time())},
        ]
        integrator = create_integrator()
        integrator.integrate(
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        assert len(integrator.state.reuse_history) > 0

    def test_state_persistence(self):
        integrator = create_integrator()
        integrator.integrate()
        state_dict = integrator.state.to_dict()
        new_state = IntegrationState.from_dict(state_dict)
        assert new_state.turn_count == 1


# ── Summary Functions ────────────────────────────────────────


class TestSummaryFunctions:
    def test_get_integration_summary(self):
        integrator = create_integrator()
        summary = get_integration_summary(integrator)
        assert "turn_count" in summary
        assert "total_integrations" in summary

    def test_get_integration_summary_text_empty(self):
        integrator = create_integrator()
        text = get_integration_summary_text(integrator)
        assert text == ""

    def test_get_integration_summary_text_nonempty(self):
        integrator = create_integrator()
        integrator._state.total_integrations = 5
        integrator._state.convergence_warnings = 1
        text = get_integration_summary_text(integrator)
        assert "統合記憶=5件" in text
        assert "収束警告=1回" in text


# ── Design Constraints ───────────────────────────────────────


class TestDesignConstraints:
    """設計書の制約事項を検証するテスト。"""

    def test_output_is_candidates_only(self):
        """統合出力は候補集合と付随情報に限定し、確定命令形式で渡さない。"""
        integrator = create_integrator()
        memories = [
            {"id": "m1", "summary": "Test", "keywords": ["test"],
             "importance": 3, "date": str(time.time())},
        ]
        result = integrator.integrate(
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        assert isinstance(result, IntegrationResult)
        assert isinstance(result.candidates, list)
        # No action/command fields
        assert not hasattr(result, 'action')
        assert not hasattr(result, 'command')

    def test_read_only_access(self):
        """各記憶系統へのアクセスは読み取り中心。統合処理から元記憶への直接改変経路を持たない。"""
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="Original", topics=["test"],
            timestamp=time.time(), vividness=0.5, reference_count=0,
            importance=None, emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        integrator = create_integrator()
        integrator.integrate(
            episodes=store,
            context=IntegrationContext(current_time=time.time()),
        )
        # Original data should be unchanged
        assert entry.summary == "Original"
        assert entry.episode_id == "ep1"

    def test_no_memory_content_modification(self):
        """記憶内容そのものの改変は行わない。"""
        memories = [
            {"id": "m1", "summary": "Original summary", "keywords": ["test"],
             "importance": 3, "date": str(time.time())},
        ]
        integrator = create_integrator()
        integrator.integrate(
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        # Original dict should be unchanged
        assert memories[0]["summary"] == "Original summary"

    def test_conflict_preserved_not_resolved(self):
        """矛盾を解消せず併存させ、参照時に単線化しない。"""
        integrator = create_integrator()
        # Create conflicting memories
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="Game was fun", topics=["game"],
            timestamp=time.time() - 60, vividness=0.9, reference_count=0,
            importance=None,
            emotional_companion=_MockEmotionalCompanion(0.9, "joy"),
        )
        store = _MockEpisodeStore([entry])
        memories = [
            {"id": "m1", "summary": "Game was terrible", "keywords": ["game"],
             "importance": 4, "date": str(time.time() - 120)},
        ]
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        # Both candidates should be present (not merged into one)
        assert len(result.candidates) == 2

    def test_duplicate_parallel_preservation(self):
        """重複調整は統合消去ではなく、同一事象の複数視点を並立保持。"""
        entry = _MockEpisodeEntry(
            episode_id="ep1", summary="played game together",
            topics=["game", "fun"], timestamp=time.time() - 60,
            vividness=0.8, reference_count=0, importance=None,
            emotional_companion=None,
        )
        store = _MockEpisodeStore([entry])
        memories = [
            {"id": "m1", "summary": "we played game together",
             "keywords": ["game", "fun"], "importance": 3,
             "date": str(time.time() - 120)},
        ]
        integrator = create_integrator()
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        # Both should be present, not merged
        assert len(result.candidates) == 2
        # Duplicates detected but both preserved
        if result.duplicate_groups:
            for group in result.duplicate_groups:
                assert len(group.unit_ids) >= 2

    def test_no_circular_reference(self):
        """統合出力を同ターン内で再入力しない。"""
        integrator = create_integrator()
        memories = [
            {"id": "m1", "summary": "Test", "keywords": ["test"],
             "importance": 3, "date": str(time.time())},
        ]
        result1 = integrator.integrate(
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        # Re-integrating with different data shouldn't include previous output
        result2 = integrator.integrate(
            long_term_memories=[
                {"id": "m2", "summary": "Other", "keywords": ["other"],
                 "importance": 2, "date": str(time.time())},
            ],
            context=IntegrationContext(current_time=time.time()),
        )
        # result2 candidates should only be from m2, not from result1
        for c in result2.candidates:
            assert c.source_id != "m1"

    def test_source_diversity_maintained(self):
        """出所横断の混在提示を維持する。"""
        integrator = create_integrator()
        entries = []
        for i in range(8):
            entries.append(_MockEpisodeEntry(
                episode_id=f"ep{i}", summary=f"Episode {i}",
                topics=[f"topic{i}"], timestamp=time.time() - i * 60,
                vividness=0.8, reference_count=0, importance=None,
                emotional_companion=None,
            ))
        store = _MockEpisodeStore(entries)
        memories = [
            {"id": f"m{i}", "summary": f"Memory {i}",
             "keywords": [f"topic{i}"], "importance": 3,
             "date": str(time.time() - i * 60)}
            for i in range(3)
        ]
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        if len(result.candidates) >= 3:
            sources = {c.source.value for c in result.candidates}
            assert len(sources) >= 2

    def test_no_single_source_priority(self):
        """単一系統の記憶を恒常的に優先しない。"""
        integrator = create_integrator()
        cfg = IntegrationConfig(single_source_cap=0.6, max_output_candidates=10)
        integrator._state.config = cfg
        # Even with many episodic entries, long_term should get a chance
        entries = [
            _MockEpisodeEntry(
                episode_id=f"ep{i}", summary=f"Ep {i}",
                topics=["common"], timestamp=time.time(),
                vividness=0.9, reference_count=0, importance=None,
                emotional_companion=None,
            )
            for i in range(15)
        ]
        store = _MockEpisodeStore(entries)
        memories = [
            {"id": "m1", "summary": "LT memory", "keywords": ["common"],
             "importance": 4, "date": str(time.time())},
        ]
        result = integrator.integrate(
            episodes=store,
            long_term_memories=memories,
            context=IntegrationContext(current_time=time.time()),
        )
        ep_count = sum(1 for c in result.candidates if c.source == MemorySource.EPISODIC)
        total = len(result.candidates)
        if total >= 3:
            assert ep_count / total <= 0.8
