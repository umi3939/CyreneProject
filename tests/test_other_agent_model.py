"""
Tests for psyche/other_agent_model.py

他者モデル（Other Agent Model）のテスト。
~100件のテストで全機能をカバーする。
"""

import json
import os
import tempfile
import pytest
from types import SimpleNamespace

from psyche.other_agent_model import (
    # Enums
    ObservationSourceType,
    InferenceBasis,
    HypothesisStrength,
    HypothesisFreshness,
    # Level determination
    determine_freshness_level,
    determine_strength_level,
    # Dataclasses
    ObservationLink,
    OtherStateHypothesis,
    SelfOtherBoundary,
    OtherModelStore,
    OtherAgentModelConfig,
    # Extraction functions
    extract_from_external_context,
    extract_from_reaction_log,
    extract_from_self_contrast,
    # Computation functions
    compute_observation_strength,
    detect_hypothesis_competitions,
    determine_inference_basis,
    generate_hypothesis_description,
    compute_self_other_boundary,
    # System
    OtherAgentModelSystem,
    # Integration functions
    observe_from_chain,
    generate_other_model_tags,
    get_other_model_summary,
    get_other_model_for_introspection,
    # Verification functions
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    verify_no_intent_assertion,
    # Convenience / Persistence
    create_config,
    create_empty_store,
    create_system,
    save_other_model_state,
    load_other_model_state,
)


# =============================================================================
# TestEnums
# =============================================================================

class TestEnums:
    """全Enum値の検証。"""

    def test_observation_source_type_values(self):
        assert ObservationSourceType.EXTERNAL_CONTEXT.value == "external_context"
        assert ObservationSourceType.REACTION_LOG.value == "reaction_log"
        assert ObservationSourceType.SELF_CONTRAST.value == "self_contrast"
        assert ObservationSourceType.MIXED.value == "mixed"

    def test_inference_basis_values(self):
        assert InferenceBasis.BEHAVIORAL.value == "behavioral"
        assert InferenceBasis.CONTEXTUAL.value == "contextual"
        assert InferenceBasis.CONTRAST.value == "contrast"
        assert InferenceBasis.COMBINED.value == "combined"
        assert InferenceBasis.UNDEFINED.value == "undefined"

    def test_hypothesis_strength_values(self):
        assert HypothesisStrength.STRONG.value == "strong"
        assert HypothesisStrength.MODERATE.value == "moderate"
        assert HypothesisStrength.WEAK.value == "weak"
        assert HypothesisStrength.FAINT.value == "faint"
        assert HypothesisStrength.UNDEFINED.value == "undefined"

    def test_hypothesis_freshness_values(self):
        assert HypothesisFreshness.FRESH.value == "fresh"
        assert HypothesisFreshness.RECENT.value == "recent"
        assert HypothesisFreshness.AGING.value == "aging"
        assert HypothesisFreshness.STALE.value == "stale"
        assert HypothesisFreshness.FADED.value == "faded"

    def test_enum_member_counts(self):
        assert len(ObservationSourceType) == 4
        assert len(InferenceBasis) == 5
        assert len(HypothesisStrength) == 5
        assert len(HypothesisFreshness) == 5


# =============================================================================
# TestDetermineFreshnessLevel
# =============================================================================

class TestDetermineFreshnessLevel:
    """鮮度レベル判定。"""

    def test_fresh(self):
        assert determine_freshness_level(0.9) == HypothesisFreshness.FRESH
        assert determine_freshness_level(0.8) == HypothesisFreshness.FRESH

    def test_recent(self):
        assert determine_freshness_level(0.7) == HypothesisFreshness.RECENT
        assert determine_freshness_level(0.6) == HypothesisFreshness.RECENT

    def test_aging(self):
        assert determine_freshness_level(0.5) == HypothesisFreshness.AGING
        assert determine_freshness_level(0.4) == HypothesisFreshness.AGING

    def test_stale(self):
        assert determine_freshness_level(0.3) == HypothesisFreshness.STALE
        assert determine_freshness_level(0.15) == HypothesisFreshness.STALE

    def test_faded(self):
        assert determine_freshness_level(0.1) == HypothesisFreshness.FADED
        assert determine_freshness_level(0.0) == HypothesisFreshness.FADED


# =============================================================================
# TestDetermineStrengthLevel
# =============================================================================

class TestDetermineStrengthLevel:
    """強度レベル判定。"""

    def test_strong(self):
        assert determine_strength_level(0.9) == HypothesisStrength.STRONG
        assert determine_strength_level(0.7) == HypothesisStrength.STRONG

    def test_moderate(self):
        assert determine_strength_level(0.5) == HypothesisStrength.MODERATE
        assert determine_strength_level(0.4) == HypothesisStrength.MODERATE

    def test_weak(self):
        assert determine_strength_level(0.3) == HypothesisStrength.WEAK
        assert determine_strength_level(0.2) == HypothesisStrength.WEAK

    def test_faint(self):
        assert determine_strength_level(0.1) == HypothesisStrength.FAINT
        assert determine_strength_level(0.05) == HypothesisStrength.FAINT

    def test_undefined(self):
        assert determine_strength_level(0.04) == HypothesisStrength.UNDEFINED
        assert determine_strength_level(0.0) == HypothesisStrength.UNDEFINED


# =============================================================================
# TestObservationLink
# =============================================================================

class TestObservationLink:
    """観測リンク生成。"""

    def test_creation(self):
        link = ObservationLink(
            link_id="link1",
            hypothesis_id="hyp1",
            source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            source_description="Test observation",
            contribution=0.8,
        )
        assert link.link_id == "link1"
        assert link.hypothesis_id == "hyp1"
        assert link.source_type == ObservationSourceType.EXTERNAL_CONTEXT
        assert link.contribution == 0.8

    def test_frozen(self):
        link = ObservationLink(
            link_id="link1",
            hypothesis_id="hyp1",
            source_type=ObservationSourceType.REACTION_LOG,
            source_description="Test",
            contribution=0.5,
        )
        with pytest.raises(AttributeError):
            link.contribution = 0.9


# =============================================================================
# TestOtherStateHypothesis
# =============================================================================

class TestOtherStateHypothesis:
    """仮説の生成・変異メソッド。"""

    def _make_hypothesis(self, **kwargs):
        defaults = dict(
            hypothesis_id="hyp1",
            source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL,
            description="Test hypothesis",
            timestamp="12345",
            freshness=0.8,
            strength=0.6,
            reference_count=0,
            evidence_ids=("ev1",),
            competing_ids=(),
            revision_count=0,
            undetermined_aspects=("intent_uncertain",),
        )
        defaults.update(kwargs)
        return OtherStateHypothesis(**defaults)

    def test_creation(self):
        h = self._make_hypothesis()
        assert h.hypothesis_id == "hyp1"
        assert h.freshness == 0.8
        assert h.strength == 0.6

    def test_get_freshness_level(self):
        h = self._make_hypothesis(freshness=0.9)
        assert h.get_freshness_level() == HypothesisFreshness.FRESH

    def test_get_strength_level(self):
        h = self._make_hypothesis(strength=0.8)
        assert h.get_strength_level() == HypothesisStrength.STRONG

    def test_with_freshness(self):
        h = self._make_hypothesis(freshness=0.5)
        updated = h.with_freshness(0.9)
        assert updated.freshness == 0.9
        assert h.freshness == 0.5  # Original unchanged

    def test_with_freshness_clamped(self):
        h = self._make_hypothesis()
        assert h.with_freshness(1.5).freshness == 1.0
        assert h.with_freshness(-0.5).freshness == 0.0

    def test_with_strength(self):
        h = self._make_hypothesis(strength=0.3)
        updated = h.with_strength(0.7)
        assert updated.strength == 0.7

    def test_with_reference(self):
        h = self._make_hypothesis(reference_count=2)
        updated = h.with_reference()
        assert updated.reference_count == 3

    def test_revise(self):
        h = self._make_hypothesis(revision_count=0)
        revised = h.revise("New description")
        assert revised.description == "New description"
        assert revised.revision_count == 1

    def test_with_competing(self):
        h = self._make_hypothesis(competing_ids=())
        updated = h.with_competing("comp1")
        assert "comp1" in updated.competing_ids
        # Adding same ID again returns self
        same = updated.with_competing("comp1")
        assert same is updated


# =============================================================================
# TestSelfOtherBoundary
# =============================================================================

class TestSelfOtherBoundary:
    """境界指標の生成。"""

    def test_creation(self):
        b = SelfOtherBoundary(
            boundary_id="b1",
            self_description="Self feels calm",
            other_description="Other appears agitated",
            divergence=0.7,
            boundary_aspects=("emotional_state",),
            timestamp="12345",
        )
        assert b.boundary_id == "b1"
        assert b.divergence == 0.7

    def test_frozen(self):
        b = SelfOtherBoundary(
            boundary_id="b1",
            self_description="calm",
            other_description="agitated",
            divergence=0.5,
            boundary_aspects=(),
            timestamp="12345",
        )
        with pytest.raises(AttributeError):
            b.divergence = 0.9

    def test_aspects_tuple(self):
        b = SelfOtherBoundary(
            boundary_id="b1",
            self_description="calm",
            other_description="neutral",
            divergence=0.3,
            boundary_aspects=("emotion", "engagement"),
            timestamp="12345",
        )
        assert len(b.boundary_aspects) == 2


# =============================================================================
# TestOtherModelStore
# =============================================================================

class TestOtherModelStore:
    """ストア生成・フィルタ・シリアライゼーション。"""

    def _make_store(self, hypotheses=(), **kwargs):
        defaults = dict(
            hypotheses=hypotheses,
            observation_links=(),
            boundaries=(),
            total_hypotheses_created=0,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.0,
            average_strength=0.0,
            active_hypothesis_count=0,
            competing_pair_count=0,
            boundary_count=0,
            timestamp="12345",
            description="Test store",
        )
        defaults.update(kwargs)
        return OtherModelStore(**defaults)

    def test_empty_store(self):
        store = self._make_store()
        assert not store.has_hypotheses()

    def test_has_hypotheses(self):
        h = OtherStateHypothesis(
            hypothesis_id="h1",
            source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL,
            description="test",
            timestamp="12345",
            freshness=0.8,
            strength=0.6,
            reference_count=0,
            evidence_ids=(),
            competing_ids=(),
            revision_count=0,
            undetermined_aspects=(),
        )
        store = self._make_store(hypotheses=(h,))
        assert store.has_hypotheses()

    def test_get_active_hypotheses(self):
        h_active = OtherStateHypothesis(
            hypothesis_id="h1", source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL, description="active", timestamp="12345",
            freshness=0.8, strength=0.6, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0, undetermined_aspects=(),
        )
        h_stale = OtherStateHypothesis(
            hypothesis_id="h2", source_type=ObservationSourceType.REACTION_LOG,
            basis=InferenceBasis.CONTEXTUAL, description="stale", timestamp="12345",
            freshness=0.1, strength=0.3, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0, undetermined_aspects=(),
        )
        store = self._make_store(hypotheses=(h_active, h_stale))
        active = store.get_active_hypotheses()
        assert len(active) == 1
        assert active[0].hypothesis_id == "h1"

    def test_get_strong_hypotheses(self):
        h_strong = OtherStateHypothesis(
            hypothesis_id="h1", source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL, description="strong", timestamp="12345",
            freshness=0.8, strength=0.7, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0, undetermined_aspects=(),
        )
        h_weak = OtherStateHypothesis(
            hypothesis_id="h2", source_type=ObservationSourceType.REACTION_LOG,
            basis=InferenceBasis.CONTEXTUAL, description="weak", timestamp="12345",
            freshness=0.8, strength=0.2, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0, undetermined_aspects=(),
        )
        store = self._make_store(hypotheses=(h_strong, h_weak))
        strong = store.get_strong_hypotheses()
        assert len(strong) == 1

    def test_to_dict_from_dict_roundtrip(self):
        h = OtherStateHypothesis(
            hypothesis_id="h1", source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL, description="test", timestamp="12345",
            freshness=0.8, strength=0.6, reference_count=1,
            evidence_ids=("ev1",), competing_ids=("comp1",), revision_count=0,
            undetermined_aspects=("intent_uncertain",),
        )
        link = ObservationLink(
            link_id="l1", hypothesis_id="h1",
            source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            source_description="test evidence", contribution=0.8,
        )
        boundary = SelfOtherBoundary(
            boundary_id="b1", self_description="calm", other_description="agitated",
            divergence=0.7, boundary_aspects=("emotion",), timestamp="12345",
        )
        store = OtherModelStore(
            hypotheses=(h,), observation_links=(link,), boundaries=(boundary,),
            total_hypotheses_created=1, total_revisions=0, total_expirations=0,
            average_freshness=0.8, average_strength=0.6,
            active_hypothesis_count=1, competing_pair_count=0,
            boundary_count=1, timestamp="12345", description="Test",
        )
        data = store.to_dict()
        restored = OtherModelStore.from_dict(data)
        assert len(restored.hypotheses) == 1
        assert restored.hypotheses[0].hypothesis_id == "h1"
        assert len(restored.observation_links) == 1
        assert len(restored.boundaries) == 1
        assert restored.boundary_count == 1

    def test_from_dict_empty(self):
        store = OtherModelStore.from_dict({})
        assert not store.has_hypotheses()


# =============================================================================
# TestOtherAgentModelConfig
# =============================================================================

class TestOtherAgentModelConfig:
    """設定のデフォルト値・カスタム値。"""

    def test_defaults(self):
        config = OtherAgentModelConfig()
        assert config.max_hypotheses == 60
        assert config.base_decay_rate == 0.05
        assert config.strength_decay_rate == 0.03
        assert config.freshness_boost_on_reference == 0.10
        assert config.stale_threshold == 0.15
        assert config.min_strength_for_retention == 0.05
        assert config.max_evidence_per_hypothesis == 8
        assert config.max_boundaries == 10

    def test_custom_values(self):
        config = OtherAgentModelConfig(max_hypotheses=30, base_decay_rate=0.1)
        assert config.max_hypotheses == 30
        assert config.base_decay_rate == 0.1


# =============================================================================
# TestExtractFromExternalContext
# =============================================================================

class TestExtractFromExternalContext:
    """外部文脈からの抽出。"""

    def test_none_input(self):
        assert extract_from_external_context(None) == []

    def test_object_high_responsiveness(self):
        ctx = SimpleNamespace(
            responsiveness=0.8, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        results = extract_from_external_context(ctx)
        assert len(results) >= 1
        assert "engaged" in results[0][0].lower()

    def test_dict_input(self):
        ctx = {"responsiveness": 0.9, "weight": 0.3, "pace": 0.5}
        results = extract_from_external_context(ctx)
        assert len(results) >= 1
        assert "engaged" in results[0][0].lower()

    def test_neutral_context(self):
        ctx = SimpleNamespace(
            responsiveness=0.5, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        results = extract_from_external_context(ctx)
        # Neutral state should produce at least one result
        assert len(results) >= 1
        assert "neutral" in results[0][0].lower() or "ambiguous" in results[0][0].lower()


# =============================================================================
# TestExtractFromReactionLog
# =============================================================================

class TestExtractFromReactionLog:
    """反応ログからの抽出。"""

    def test_none_input(self):
        assert extract_from_reaction_log(None) == []

    def test_object_with_entries(self):
        entry = SimpleNamespace(
            intent="question", emotion_label="curious", valence=0.5, source_text="Why?",
        )
        log = SimpleNamespace(entries=[entry])
        results = extract_from_reaction_log(log)
        assert len(results) >= 1
        assert any("question" in r[0].lower() for r in results)

    def test_dict_input(self):
        log = {
            "entries": [
                {"intent": "question", "valence": 0.6, "emotion_label": "curious"},
            ],
        }
        results = extract_from_reaction_log(log)
        assert len(results) >= 1

    def test_empty_entries(self):
        log = SimpleNamespace(entries=[])
        assert extract_from_reaction_log(log) == []


# =============================================================================
# TestExtractFromSelfContrast
# =============================================================================

class TestExtractFromSelfContrast:
    """自己対比からの抽出。"""

    def test_none_input(self):
        assert extract_from_self_contrast(None, None) == []
        assert extract_from_self_contrast(None, SimpleNamespace()) == []

    def test_contrast_detected(self):
        self_state = SimpleNamespace(intensity=0.9, description="highly emotional")
        other_signals = SimpleNamespace(responsiveness=0.2, weight=0.3)
        results = extract_from_self_contrast(self_state, other_signals)
        assert len(results) >= 1
        assert any("contrast" in r[0].lower() for r in results)

    def test_no_contrast(self):
        self_state = SimpleNamespace(intensity=0.5, description="neutral")
        other_signals = SimpleNamespace(responsiveness=0.5, weight=0.5)
        results = extract_from_self_contrast(self_state, other_signals)
        assert len(results) == 0

    def test_dict_input(self):
        self_state = {"intensity": 0.9, "description": "intense"}
        other_signals = {"responsiveness": 0.1, "weight": 0.2}
        results = extract_from_self_contrast(self_state, other_signals)
        assert len(results) >= 1


# =============================================================================
# TestComputeObservationStrength
# =============================================================================

class TestComputeObservationStrength:
    """観測強度の計算。"""

    def test_empty(self):
        assert compute_observation_strength([]) == 0.0

    def test_single_link(self):
        link = ObservationLink(
            link_id="l1", hypothesis_id="h1",
            source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            source_description="test", contribution=0.8,
        )
        result = compute_observation_strength([link])
        assert 0.0 < result <= 1.0

    def test_multiple_links(self):
        links = [
            ObservationLink(
                link_id=f"l{i}", hypothesis_id="h1",
                source_type=ObservationSourceType.EXTERNAL_CONTEXT,
                source_description=f"test{i}", contribution=0.7,
            )
            for i in range(3)
        ]
        result = compute_observation_strength(links)
        assert 0.0 < result <= 1.0


# =============================================================================
# TestDetectCompetitions
# =============================================================================

class TestDetectCompetitions:
    """競合検出。"""

    def _make_hyp(self, hyp_id, source_type, basis, description):
        return OtherStateHypothesis(
            hypothesis_id=hyp_id, source_type=source_type, basis=basis,
            description=description, timestamp="12345",
            freshness=0.8, strength=0.6, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0,
            undetermined_aspects=(),
        )

    def test_no_competition(self):
        h1 = self._make_hyp("h1", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.BEHAVIORAL, "completely different topic alpha")
        h2 = self._make_hyp("h2", ObservationSourceType.REACTION_LOG,
                            InferenceBasis.BEHAVIORAL, "another unrelated subject beta")
        pairs = detect_hypothesis_competitions([h1, h2])
        assert len(pairs) == 0

    def test_competition_same_source_different_basis(self):
        h1 = self._make_hyp("h1", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.BEHAVIORAL, "Other party appears engaged and responsive")
        h2 = self._make_hyp("h2", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.CONTEXTUAL, "Other party appears disengaged and distant")
        pairs = detect_hypothesis_competitions([h1, h2])
        assert len(pairs) >= 1

    def test_multiple_competitions(self):
        h1 = self._make_hyp("h1", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.BEHAVIORAL, "Other party appears engaged")
        h2 = self._make_hyp("h2", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.CONTEXTUAL, "Other party appears disengaged")
        h3 = self._make_hyp("h3", ObservationSourceType.EXTERNAL_CONTEXT,
                            InferenceBasis.CONTRAST, "Other party appears neutral")
        pairs = detect_hypothesis_competitions([h1, h2, h3])
        assert len(pairs) >= 1


# =============================================================================
# TestComputeSelfOtherBoundary
# =============================================================================

class TestComputeSelfOtherBoundary:
    """自己/他者境界の計算。"""

    def _make_hyp(self, hyp_id, description, basis=InferenceBasis.BEHAVIORAL):
        return OtherStateHypothesis(
            hypothesis_id=hyp_id, source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=basis, description=description, timestamp="12345",
            freshness=0.8, strength=0.6, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0,
            undetermined_aspects=(),
        )

    def test_empty_hypotheses(self):
        boundary = compute_self_other_boundary("self is calm", [])
        assert boundary.divergence == 0.0

    def test_with_hypotheses(self):
        h = self._make_hyp("h1", "other appears agitated")
        boundary = compute_self_other_boundary("self is calm", [h])
        assert boundary.divergence > 0.0
        assert boundary.self_description == "self is calm"

    def test_high_divergence(self):
        h = self._make_hyp("h1", "completely different entirely unrelated words")
        boundary = compute_self_other_boundary("nothing similar at all", [h])
        assert boundary.divergence > 0.5


# =============================================================================
# TestOtherAgentModelSystem
# =============================================================================

class TestOtherAgentModelSystem:
    """システムクラスの主要操作。"""

    def test_init_default_config(self):
        system = OtherAgentModelSystem()
        store = system.get_store()
        assert not store.has_hypotheses()

    def test_init_custom_config(self):
        config = OtherAgentModelConfig(max_hypotheses=10)
        system = OtherAgentModelSystem(config=config)
        assert system._config.max_hypotheses == 10

    def test_observe_with_external_context(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        assert store.has_hypotheses()
        assert store.total_hypotheses_created > 0

    def test_observe_with_reaction_log(self):
        system = OtherAgentModelSystem()
        entry = SimpleNamespace(
            intent="question", emotion_label="curious", valence=0.6, source_text="Why?",
        )
        log = SimpleNamespace(entries=[entry])
        store = system.observe_other(reaction_log=log)
        assert store.has_hypotheses()

    def test_observe_with_self_contrast(self):
        system = OtherAgentModelSystem()
        self_state = SimpleNamespace(intensity=0.9, description="highly emotional")
        ctx = SimpleNamespace(
            responsiveness=0.2, weight=0.3, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx, self_state=self_state)
        assert store.has_hypotheses()

    def test_observe_all_sources(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.8, pace=0.5, density=0.5, continuity=0.5,
        )
        entry = SimpleNamespace(
            intent="question", emotion_label="", valence=0.5, source_text="What?",
        )
        log = SimpleNamespace(entries=[entry])
        self_state = SimpleNamespace(intensity=0.1, description="calm")
        store = system.observe_other(
            external_context=ctx, reaction_log=log, self_state=self_state,
        )
        assert store.has_hypotheses()
        assert store.total_hypotheses_created >= 2

    def test_decay_hypotheses(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store1 = system.observe_other(external_context=ctx)
        initial_freshness = store1.average_freshness

        store2 = system.decay_hypotheses()
        assert store2.average_freshness < initial_freshness

    def test_reference_hypothesis(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        hyp_id = store.hypotheses[0].hypothesis_id
        original_ref = store.hypotheses[0].reference_count

        system.reference_hypothesis(hyp_id)
        new_store = system.get_store()
        updated = [h for h in new_store.hypotheses if h.hypothesis_id == hyp_id]
        assert len(updated) == 1
        assert updated[0].reference_count == original_ref + 1

    def test_revise_hypothesis(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        hyp_id = store.hypotheses[0].hypothesis_id

        system.revise_hypothesis(hyp_id, "Revised description")
        new_store = system.get_store()
        updated = [h for h in new_store.hypotheses if h.hypothesis_id == hyp_id]
        assert len(updated) == 1
        assert updated[0].description == "Revised description"
        assert updated[0].revision_count == 1

    def test_get_active_hypotheses(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.8, pace=0.5, density=0.5, continuity=0.5,
        )
        system.observe_other(external_context=ctx)
        active = system.get_active_hypotheses(max_count=5)
        assert len(active) >= 1
        # Should be sorted by strength
        if len(active) > 1:
            assert active[0].strength >= active[1].strength

    def test_get_last_store(self):
        system = OtherAgentModelSystem()
        assert system.get_last_store() is None

        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        system.observe_other(external_context=ctx)
        last = system.get_last_store()
        assert last is not None
        assert last.has_hypotheses()

    def test_boundary_creation(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        self_state = SimpleNamespace(intensity=0.3, description="calm state")
        store = system.observe_other(external_context=ctx, self_state=self_state)
        assert store.boundary_count >= 1

    def test_capacity_enforcement(self):
        config = OtherAgentModelConfig(max_hypotheses=3)
        system = OtherAgentModelSystem(config=config)
        # Add many hypotheses
        for _ in range(5):
            ctx = SimpleNamespace(
                responsiveness=0.9, weight=0.8, pace=0.7, density=0.5, continuity=0.5,
            )
            system.observe_other(external_context=ctx)
        store = system.get_store()
        assert len(store.hypotheses) <= 3

    def test_observe_none_inputs(self):
        system = OtherAgentModelSystem()
        store = system.observe_other()
        assert not store.has_hypotheses()


# =============================================================================
# TestObserveFromChain
# =============================================================================

class TestObserveFromChain:
    """チェーン統合。"""

    def test_chain_integration(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = observe_from_chain(system, external_context=ctx)
        assert store.has_hypotheses()

    def test_chain_with_all_inputs(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.8, weight=0.7, pace=0.5, density=0.5, continuity=0.5,
        )
        entry = SimpleNamespace(
            intent="question", emotion_label="", valence=0.4, source_text="",
        )
        log = SimpleNamespace(entries=[entry])
        self_state = SimpleNamespace(intensity=0.2, description="calm")
        store = observe_from_chain(system, ctx, log, self_state)
        assert store.has_hypotheses()


# =============================================================================
# TestGenerateOtherModelTags
# =============================================================================

class TestGenerateOtherModelTags:
    """タグ生成。"""

    def test_none_store(self):
        tags = generate_other_model_tags(None)
        assert len(tags) == 1
        assert tags[0]["category"] == "OTHER_MODEL_COUNT"

    def test_empty_store(self):
        store = create_empty_store()
        tags = generate_other_model_tags(store)
        assert len(tags) == 1
        assert tags[0]["label"] == "no_hypotheses"

    def test_with_hypotheses(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.8, pace=0.5, density=0.5, continuity=0.5,
        )
        self_state = SimpleNamespace(intensity=0.1, description="calm")
        store = system.observe_other(external_context=ctx, self_state=self_state)
        tags = generate_other_model_tags(store)
        assert len(tags) >= 3
        categories = [t["category"] for t in tags]
        assert "OTHER_MODEL_COUNT" in categories
        assert "OTHER_MODEL_STRENGTH" in categories
        assert "OTHER_MODEL_FRESHNESS" in categories

    def test_scale_parameter(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        tags_1 = generate_other_model_tags(store, scale=1.0)
        tags_2 = generate_other_model_tags(store, scale=2.0)
        # Scale 2.0 should produce higher weights
        for t1, t2 in zip(tags_1, tags_2):
            if t1["category"] == t2["category"]:
                assert t2["weight"] > t1["weight"]


# =============================================================================
# TestGetOtherModelSummary
# =============================================================================

class TestGetOtherModelSummary:
    """サマリー生成。"""

    def test_none_store(self):
        summary = get_other_model_summary(None)
        assert "No hypotheses" in summary

    def test_with_hypotheses(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        summary = get_other_model_summary(store)
        assert "Other Agent Model State" in summary
        assert "Total hypotheses" in summary


# =============================================================================
# TestGetOtherModelForIntrospection
# =============================================================================

class TestGetOtherModelForIntrospection:
    """内省用データ取得。"""

    def test_none_store(self):
        data = get_other_model_for_introspection(None)
        assert data["has_hypotheses"] is False
        assert data["total_hypotheses"] == 0
        assert data["boundary_count"] == 0

    def test_with_hypotheses(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        data = get_other_model_for_introspection(store)
        assert data["has_hypotheses"] is True
        assert data["total_hypotheses"] > 0
        assert "source_distribution" in data
        assert "basis_distribution" in data


# =============================================================================
# TestVerification
# =============================================================================

class TestVerification:
    """全検証関数。"""

    def test_no_decision_impact(self):
        store = create_empty_store()
        assert verify_no_decision_impact(store) is True

    def test_no_goal_generation(self):
        system = OtherAgentModelSystem()
        assert verify_no_goal_generation(system) is True

    def test_read_only_principle(self):
        system = OtherAgentModelSystem()
        assert verify_read_only_principle(system) is True

    def test_no_value_modification(self):
        system = OtherAgentModelSystem()
        assert verify_no_value_modification(system) is True

    def test_no_intent_assertion(self):
        system = OtherAgentModelSystem()
        assert verify_no_intent_assertion(system) is True

    def test_store_verification_with_data(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        assert verify_no_decision_impact(store) is True


# =============================================================================
# TestConvenience
# =============================================================================

class TestConvenience:
    """コンビニエンス関数。"""

    def test_create_config(self):
        config = create_config(max_hypotheses=20)
        assert config.max_hypotheses == 20

    def test_create_empty_store(self):
        store = create_empty_store()
        assert not store.has_hypotheses()
        assert store.boundary_count == 0

    def test_create_system(self):
        system = create_system()
        assert isinstance(system, OtherAgentModelSystem)

    def test_create_system_with_config(self):
        config = create_config(max_hypotheses=15)
        system = create_system(config=config)
        assert system._config.max_hypotheses == 15


# =============================================================================
# TestPersistence
# =============================================================================

class TestPersistence:
    """保存・読込のラウンドトリップ。"""

    def test_save_load_roundtrip(self):
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        self_state = SimpleNamespace(intensity=0.3, description="calm")
        store = system.observe_other(external_context=ctx, self_state=self_state)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            filepath = f.name

        try:
            save_other_model_state(store, filepath)
            loaded = load_other_model_state(filepath)
            assert loaded.has_hypotheses()
            assert loaded.total_hypotheses_created == store.total_hypotheses_created
            assert len(loaded.hypotheses) == len(store.hypotheses)
            assert len(loaded.boundaries) == len(store.boundaries)
        finally:
            os.unlink(filepath)

    def test_save_load_empty(self):
        store = create_empty_store()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            filepath = f.name

        try:
            save_other_model_state(store, filepath)
            loaded = load_other_model_state(filepath)
            assert not loaded.has_hypotheses()
        finally:
            os.unlink(filepath)


# =============================================================================
# TestDesignPrinciples
# =============================================================================

class TestDesignPrinciples:
    """設計制約メタテスト。"""

    def test_no_decision_methods(self):
        """他者モデルシステムに判断メソッドが存在しない。"""
        system = OtherAgentModelSystem()
        forbidden = ["decide", "choose", "select_action", "optimize"]
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            for f in forbidden:
                assert f not in method.lower(), f"Forbidden pattern '{f}' found in method '{method}'"

    def test_no_evaluation_methods(self):
        """他者モデルシステムに評価メソッドが存在しない。"""
        system = OtherAgentModelSystem()
        forbidden = ["evaluate_morality", "judge_correctness", "rate_behavior"]
        methods = [m for m in dir(system) if not m.startswith("_")]
        for method in methods:
            for f in forbidden:
                assert f not in method.lower(), f"Forbidden pattern '{f}' found in method '{method}'"

    def test_hypotheses_are_immutable(self):
        """仮説はfrozen dataclass。"""
        h = OtherStateHypothesis(
            hypothesis_id="h1", source_type=ObservationSourceType.EXTERNAL_CONTEXT,
            basis=InferenceBasis.BEHAVIORAL, description="test", timestamp="12345",
            freshness=0.8, strength=0.6, reference_count=0,
            evidence_ids=(), competing_ids=(), revision_count=0,
            undetermined_aspects=(),
        )
        with pytest.raises(AttributeError):
            h.description = "modified"

    def test_store_is_immutable(self):
        """ストアはfrozen dataclass。"""
        store = create_empty_store()
        with pytest.raises(AttributeError):
            store.description = "modified"

    def test_boundary_is_immutable(self):
        """境界はfrozen dataclass。"""
        b = SelfOtherBoundary(
            boundary_id="b1", self_description="self", other_description="other",
            divergence=0.5, boundary_aspects=(), timestamp="12345",
        )
        with pytest.raises(AttributeError):
            b.divergence = 0.9

    def test_competing_hypotheses_allowed(self):
        """競合する仮説が許容される。"""
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.8, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        # Multiple hypotheses from different bases should be able to coexist
        assert len(store.hypotheses) >= 1

    def test_revision_is_possible(self):
        """仮説の修正が可能。"""
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        hyp_id = store.hypotheses[0].hypothesis_id
        system.revise_hypothesis(hyp_id, "Revised hypothesis")
        new_store = system.get_store()
        revised = [h for h in new_store.hypotheses if h.hypothesis_id == hyp_id]
        assert revised[0].description == "Revised hypothesis"

    def test_undetermined_aspects_present(self):
        """仮説に未確定の側面が含まれる。"""
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        store = system.observe_other(external_context=ctx)
        for h in store.hypotheses:
            assert len(h.undetermined_aspects) > 0

    def test_no_intent_assertion_in_descriptions(self):
        """仮説の記述に断定的な意図の主張がない。"""
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.8, pace=0.5, density=0.5, continuity=0.5,
        )
        entry = SimpleNamespace(
            intent="question", emotion_label="", valence=0.6, source_text="What?",
        )
        log = SimpleNamespace(entries=[entry])
        store = system.observe_other(external_context=ctx, reaction_log=log)
        for h in store.hypotheses:
            desc_lower = h.description.lower()
            # Descriptions should use hedging language
            assert "definitely" not in desc_lower
            assert "certainly" not in desc_lower
            assert "must be" not in desc_lower

    def test_self_other_separation(self):
        """自己と他者の分離が構造的に保たれている。"""
        system = OtherAgentModelSystem()
        ctx = SimpleNamespace(
            responsiveness=0.9, weight=0.5, pace=0.5, density=0.5, continuity=0.5,
        )
        self_state = SimpleNamespace(intensity=0.3, description="calm self-state")
        store = system.observe_other(external_context=ctx, self_state=self_state)
        # Boundaries should be created separating self and other
        assert store.boundary_count >= 1
        boundaries = store.boundaries
        for b in boundaries:
            assert b.self_description != "" or b.other_description != ""


# =============================================================================
# TestDetermineInferenceBasis
# =============================================================================

class TestDetermineInferenceBasis:
    """推論根拠の決定。"""

    def test_empty(self):
        assert determine_inference_basis([]) == InferenceBasis.UNDEFINED

    def test_single_external(self):
        assert determine_inference_basis([ObservationSourceType.EXTERNAL_CONTEXT]) == InferenceBasis.CONTEXTUAL

    def test_single_reaction(self):
        assert determine_inference_basis([ObservationSourceType.REACTION_LOG]) == InferenceBasis.BEHAVIORAL

    def test_single_contrast(self):
        assert determine_inference_basis([ObservationSourceType.SELF_CONTRAST]) == InferenceBasis.CONTRAST

    def test_mixed(self):
        result = determine_inference_basis([
            ObservationSourceType.EXTERNAL_CONTEXT,
            ObservationSourceType.REACTION_LOG,
        ])
        assert result == InferenceBasis.COMBINED


# =============================================================================
# TestGenerateHypothesisDescription
# =============================================================================

class TestGenerateHypothesisDescription:
    """仮説記述の生成。"""

    def test_behavioral(self):
        desc = generate_hypothesis_description(InferenceBasis.BEHAVIORAL, ["test observation"])
        assert "observed behavior" in desc.lower()

    def test_empty_sources(self):
        desc = generate_hypothesis_description(InferenceBasis.CONTEXTUAL, [])
        assert "contextual" in desc.lower()

    def test_multiple_sources(self):
        desc = generate_hypothesis_description(
            InferenceBasis.COMBINED, ["source1", "source2", "source3"],
        )
        assert "source1" in desc
        assert "source2" in desc
