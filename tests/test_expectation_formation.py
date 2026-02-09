"""
Tests for psyche/expectation_formation.py

予期・期待の形成（Expectation Formation）のテスト。
~100件のテストケースで全コンポーネントを検証する。
"""

import json
import os
import tempfile
import time
import pytest

from psyche.expectation_formation import (
    # Enums
    ExpectationSourceType,
    ExpectationBasis,
    ExpectationStrength,
    ExpectationFreshness,
    # Structures
    EvidenceLink,
    ExpectationCandidate,
    ExpectationStore,
    ExpectationFormationConfig,
    # Pure helpers
    determine_freshness_level,
    determine_strength_level,
    # Extraction functions
    extract_from_tendency,
    extract_from_difference,
    extract_from_narrative,
    # Computation functions
    compute_evidence_strength,
    detect_competitions,
    determine_expectation_basis,
    generate_expectation_description,
    # System
    ExpectationFormationSystem,
    # Integration
    form_from_chain,
    generate_expectation_tags,
    get_expectation_summary,
    get_expectation_for_introspection,
    # Verification
    verify_no_decision_impact,
    verify_no_goal_generation,
    verify_read_only_principle,
    verify_no_value_modification,
    # Convenience
    create_config,
    create_empty_store,
    create_system,
    save_expectation_state,
    load_expectation_state,
)


# =============================================================================
# Helper: Mock objects for duck typing
# =============================================================================

class MockTendency:
    def __init__(self, strength=0.5, confidence=0.5, category="conversation", total_reinforcements=5):
        self.strength = strength
        self.confidence = confidence
        self.total_reinforcements = total_reinforcements
        self.pattern = type("Pattern", (), {"category": category})()


class MockTendencyBias:
    def __init__(self, has_bias=True, strongest_category="conversation", strongest_bias=None, biases=None):
        self.has_bias = has_bias
        self.strongest_category = strongest_category
        self.strongest_bias = strongest_bias or MockTendency()
        self.biases = biases or {}


class MockMagnitude:
    def __init__(self, value="noticeable"):
        self.value = value


class MockNature:
    def __init__(self, value="shifting"):
        self.value = value


class MockTemporalSpan:
    def __init__(self, value="short"):
        self.value = value


class MockComponentDiff:
    def __init__(self, component_name="emotional", change_type_value="increased"):
        self.component_name = component_name
        self.change_type = type("CT", (), {"value": change_type_value})()


class MockDifferenceSummary:
    def __init__(self, has_difference=True, magnitude="noticeable", nature="shifting",
                 temporal_span="short", integrated_description="Emotional shift observed"):
        self.has_difference = has_difference
        self.magnitude = MockMagnitude(magnitude)
        self.nature = MockNature(nature)
        self.temporal_span = MockTemporalSpan(temporal_span)
        self.integrated_description = integrated_description
        self.component_differences = [MockComponentDiff()]


class MockCoherence:
    def __init__(self, level="coherent", average_vividness=0.7):
        self.level = type("Level", (), {"value": level})()
        self.average_vividness = average_vividness


class MockNarrativeState:
    def __init__(self, fragments=None, coherence_level="coherent",
                 trend="accumulating", description="Test narrative"):
        self.fragments = fragments or [type("F", (), {"vividness": 0.8, "fragment_id": "f1", "description": "test"})()]
        self.coherence = MockCoherence(coherence_level)
        self.trend = type("Trend", (), {"value": trend})()
        self.description = description


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    def test_expectation_source_type_values(self):
        assert ExpectationSourceType.REPETITION.value == "repetition"
        assert ExpectationSourceType.SELF_DIFFERENCE.value == "self_difference"
        assert ExpectationSourceType.NARRATIVE.value == "narrative"
        assert ExpectationSourceType.MIXED.value == "mixed"

    def test_expectation_basis_values(self):
        assert ExpectationBasis.PATTERN_CONTINUATION.value == "pattern_continuation"
        assert ExpectationBasis.CHANGE_DIRECTION.value == "change_direction"
        assert ExpectationBasis.CONTEXT_PERSISTENCE.value == "context_persistence"
        assert ExpectationBasis.COMBINED.value == "combined"
        assert ExpectationBasis.UNDEFINED.value == "undefined"

    def test_expectation_strength_values(self):
        assert ExpectationStrength.STRONG.value == "strong"
        assert ExpectationStrength.MODERATE.value == "moderate"
        assert ExpectationStrength.WEAK.value == "weak"
        assert ExpectationStrength.FAINT.value == "faint"
        assert ExpectationStrength.UNDEFINED.value == "undefined"

    def test_expectation_freshness_values(self):
        assert ExpectationFreshness.FRESH.value == "fresh"
        assert ExpectationFreshness.RECENT.value == "recent"
        assert ExpectationFreshness.AGING.value == "aging"
        assert ExpectationFreshness.STALE.value == "stale"
        assert ExpectationFreshness.FADED.value == "faded"

    def test_all_enums_have_expected_count(self):
        assert len(ExpectationSourceType) == 4
        assert len(ExpectationBasis) == 5
        assert len(ExpectationStrength) == 5
        assert len(ExpectationFreshness) == 5


# =============================================================================
# Test Freshness Level Determination
# =============================================================================

class TestDetermineFreshnessLevel:
    def test_fresh(self):
        assert determine_freshness_level(0.9) == ExpectationFreshness.FRESH
        assert determine_freshness_level(0.8) == ExpectationFreshness.FRESH

    def test_recent(self):
        assert determine_freshness_level(0.7) == ExpectationFreshness.RECENT
        assert determine_freshness_level(0.6) == ExpectationFreshness.RECENT

    def test_aging(self):
        assert determine_freshness_level(0.5) == ExpectationFreshness.AGING
        assert determine_freshness_level(0.4) == ExpectationFreshness.AGING

    def test_stale(self):
        assert determine_freshness_level(0.3) == ExpectationFreshness.STALE
        assert determine_freshness_level(0.15) == ExpectationFreshness.STALE

    def test_faded(self):
        assert determine_freshness_level(0.1) == ExpectationFreshness.FADED
        assert determine_freshness_level(0.0) == ExpectationFreshness.FADED


# =============================================================================
# Test Strength Level Determination
# =============================================================================

class TestDetermineStrengthLevel:
    def test_strong(self):
        assert determine_strength_level(0.9) == ExpectationStrength.STRONG
        assert determine_strength_level(0.7) == ExpectationStrength.STRONG

    def test_moderate(self):
        assert determine_strength_level(0.5) == ExpectationStrength.MODERATE
        assert determine_strength_level(0.4) == ExpectationStrength.MODERATE

    def test_weak(self):
        assert determine_strength_level(0.3) == ExpectationStrength.WEAK
        assert determine_strength_level(0.2) == ExpectationStrength.WEAK

    def test_faint(self):
        assert determine_strength_level(0.1) == ExpectationStrength.FAINT
        assert determine_strength_level(0.05) == ExpectationStrength.FAINT

    def test_undefined(self):
        assert determine_strength_level(0.03) == ExpectationStrength.UNDEFINED
        assert determine_strength_level(0.0) == ExpectationStrength.UNDEFINED


# =============================================================================
# Test EvidenceLink
# =============================================================================

class TestEvidenceLink:
    def test_creation(self):
        link = EvidenceLink(
            link_id="l1",
            expectation_id="e1",
            source_type=ExpectationSourceType.REPETITION,
            source_description="Test evidence",
            contribution=0.8,
        )
        assert link.link_id == "l1"
        assert link.expectation_id == "e1"
        assert link.source_type == ExpectationSourceType.REPETITION
        assert link.contribution == 0.8

    def test_frozen(self):
        link = EvidenceLink(
            link_id="l1", expectation_id="e1",
            source_type=ExpectationSourceType.REPETITION,
            source_description="Test", contribution=0.5,
        )
        with pytest.raises(AttributeError):
            link.contribution = 0.9


# =============================================================================
# Test ExpectationCandidate
# =============================================================================

class TestExpectationCandidate:
    def _make_candidate(self, **kwargs):
        defaults = {
            "expectation_id": "e1",
            "source_type": ExpectationSourceType.REPETITION,
            "basis": ExpectationBasis.PATTERN_CONTINUATION,
            "description": "Test expectation",
            "timestamp": "123.456",
            "freshness": 0.8,
            "strength": 0.6,
            "reference_count": 0,
            "evidence_ids": ("l1",),
            "competing_ids": (),
            "revision_count": 0,
            "undetermined_aspects": ("outcome_uncertain",),
        }
        defaults.update(kwargs)
        return ExpectationCandidate(**defaults)

    def test_creation(self):
        c = self._make_candidate()
        assert c.expectation_id == "e1"
        assert c.freshness == 0.8
        assert c.strength == 0.6

    def test_get_freshness_level(self):
        c = self._make_candidate(freshness=0.9)
        assert c.get_freshness_level() == ExpectationFreshness.FRESH

    def test_get_strength_level(self):
        c = self._make_candidate(strength=0.8)
        assert c.get_strength_level() == ExpectationStrength.STRONG

    def test_with_freshness(self):
        c = self._make_candidate(freshness=0.8)
        updated = c.with_freshness(0.5)
        assert updated.freshness == 0.5
        assert c.freshness == 0.8  # original unchanged

    def test_with_freshness_clamped(self):
        c = self._make_candidate(freshness=0.8)
        assert c.with_freshness(1.5).freshness == 1.0
        assert c.with_freshness(-0.5).freshness == 0.0

    def test_with_reference(self):
        c = self._make_candidate(reference_count=2)
        updated = c.with_reference()
        assert updated.reference_count == 3
        assert c.reference_count == 2

    def test_revise(self):
        c = self._make_candidate(description="old", revision_count=0)
        revised = c.revise("new description")
        assert revised.description == "new description"
        assert revised.revision_count == 1
        assert c.description == "old"

    def test_with_competing(self):
        c = self._make_candidate(competing_ids=())
        updated = c.with_competing("e2")
        assert "e2" in updated.competing_ids
        # Duplicate should be idempotent
        same = updated.with_competing("e2")
        assert same.competing_ids == updated.competing_ids


# =============================================================================
# Test ExpectationStore
# =============================================================================

class TestExpectationStore:
    def _make_store(self, num_expectations=2):
        candidates = []
        for i in range(num_expectations):
            candidates.append(ExpectationCandidate(
                expectation_id=f"e{i}",
                source_type=ExpectationSourceType.REPETITION,
                basis=ExpectationBasis.PATTERN_CONTINUATION,
                description=f"Test expectation {i}",
                timestamp="123.456",
                freshness=0.8 - i * 0.3,
                strength=0.7 - i * 0.2,
                reference_count=0,
                evidence_ids=(),
                competing_ids=(),
                revision_count=0,
                undetermined_aspects=(),
            ))
        return ExpectationStore(
            expectations=tuple(candidates),
            evidence_links=(),
            total_expectations_created=num_expectations,
            total_revisions=0,
            total_expirations=0,
            average_freshness=0.65,
            average_strength=0.6,
            active_expectation_count=num_expectations,
            competing_pair_count=0,
            timestamp="123.456",
            description="Test store",
        )

    def test_has_expectations(self):
        store = self._make_store(2)
        assert store.has_expectations() is True

    def test_has_no_expectations(self):
        store = self._make_store(0)
        assert store.has_expectations() is False

    def test_get_active_expectations(self):
        store = self._make_store(3)
        active = store.get_active_expectations(stale_threshold=0.15)
        assert len(active) >= 1

    def test_get_strong_expectations(self):
        store = self._make_store(2)
        strong = store.get_strong_expectations()
        assert len(strong) >= 1
        for e in strong:
            assert e.strength > 0.5

    def test_serialization_roundtrip(self):
        store = self._make_store(2)
        d = store.to_dict()
        restored = ExpectationStore.from_dict(d)
        assert restored.total_expectations_created == store.total_expectations_created
        assert len(restored.expectations) == len(store.expectations)
        assert restored.description == store.description

    def test_to_dict_structure(self):
        store = self._make_store(1)
        d = store.to_dict()
        assert "expectations" in d
        assert "evidence_links" in d
        assert "total_expectations_created" in d
        assert "average_freshness" in d


# =============================================================================
# Test ExpectationFormationConfig
# =============================================================================

class TestExpectationFormationConfig:
    def test_default_values(self):
        config = ExpectationFormationConfig()
        assert config.max_expectations == 80
        assert config.base_decay_rate == 0.04
        assert config.strength_decay_rate == 0.02
        assert config.freshness_boost_on_reference == 0.10
        assert config.stale_threshold == 0.15
        assert config.min_strength_for_retention == 0.05

    def test_custom_values(self):
        config = ExpectationFormationConfig(
            max_expectations=50,
            base_decay_rate=0.06,
        )
        assert config.max_expectations == 50
        assert config.base_decay_rate == 0.06


# =============================================================================
# Test Extract From Tendency
# =============================================================================

class TestExtractFromTendency:
    def test_none_input(self):
        assert extract_from_tendency(None) == []

    def test_object_input(self):
        bias = MockTendencyBias(has_bias=True, strongest_category="conversation")
        results = extract_from_tendency(bias)
        assert len(results) >= 1
        desc, basis_hint, strength, evidence = results[0]
        assert "conversation" in desc.lower() or "tendency" in desc.lower()
        assert basis_hint == "pattern_continuation"
        assert 0.0 <= strength <= 1.0

    def test_dict_input(self):
        bias = {
            "tendency_count": 3,
            "strongest_category": "exploration",
            "strongest_bias": 0.6,
            "active_tendencies": ["a", "b", "c"],
        }
        results = extract_from_tendency(bias)
        assert len(results) >= 1
        assert "exploration" in results[0][0].lower()

    def test_no_bias(self):
        bias = MockTendencyBias(has_bias=False)
        results = extract_from_tendency(bias)
        assert results == []


# =============================================================================
# Test Extract From Difference
# =============================================================================

class TestExtractFromDifference:
    def test_none_input(self):
        assert extract_from_difference(None) == []

    def test_object_input(self):
        summary = MockDifferenceSummary(has_difference=True, nature="shifting", magnitude="noticeable")
        results = extract_from_difference(summary)
        assert len(results) >= 1
        desc, basis_hint, strength, evidence = results[0]
        assert basis_hint == "change_direction"
        assert 0.0 <= strength <= 1.0

    def test_dict_input(self):
        summary = {
            "has_difference": True,
            "magnitude": "significant",
            "nature": "returning",
            "changed_components": ["emotional", "tendency"],
        }
        results = extract_from_difference(summary)
        assert len(results) >= 1
        assert results[0][1] == "change_direction"

    def test_stable_nature(self):
        summary = MockDifferenceSummary(has_difference=True, nature="stable", magnitude="minimal")
        results = extract_from_difference(summary)
        assert len(results) >= 1
        assert results[0][1] == "context_persistence"


# =============================================================================
# Test Extract From Narrative
# =============================================================================

class TestExtractFromNarrative:
    def test_none_input(self):
        assert extract_from_narrative(None) == []

    def test_object_input(self):
        state = MockNarrativeState(trend="accumulating", coherence_level="coherent")
        results = extract_from_narrative(state)
        assert len(results) >= 1
        desc, basis_hint, strength, evidence = results[0]
        assert basis_hint == "context_persistence"
        assert strength > 0.0

    def test_dict_input(self):
        state = {
            "trend": "condensing",
            "coherence": "loosely_coherent",
            "fragment_count": 5,
            "average_vividness": 0.6,
        }
        results = extract_from_narrative(state)
        assert len(results) >= 1

    def test_dissolving_trend(self):
        state = MockNarrativeState(trend="dissolving", coherence_level="fragmented")
        results = extract_from_narrative(state)
        assert len(results) >= 1
        # Dissolving should have lower strength
        _, _, strength, _ = results[0]
        assert strength < 0.5


# =============================================================================
# Test Compute Evidence Strength
# =============================================================================

class TestComputeEvidenceStrength:
    def test_empty(self):
        assert compute_evidence_strength([]) == 0.0

    def test_single(self):
        link = EvidenceLink(
            link_id="l1", expectation_id="e1",
            source_type=ExpectationSourceType.REPETITION,
            source_description="Test", contribution=0.8,
        )
        result = compute_evidence_strength([link])
        assert 0.0 < result <= 1.0

    def test_multiple(self):
        links = [
            EvidenceLink("l1", "e1", ExpectationSourceType.REPETITION, "A", 0.9),
            EvidenceLink("l2", "e1", ExpectationSourceType.SELF_DIFFERENCE, "B", 0.6),
            EvidenceLink("l3", "e1", ExpectationSourceType.NARRATIVE, "C", 0.3),
        ]
        result = compute_evidence_strength(links)
        assert 0.0 < result <= 1.0


# =============================================================================
# Test Detect Competitions
# =============================================================================

class TestDetectCompetitions:
    def _make_candidate(self, eid, source_type, basis, description):
        return ExpectationCandidate(
            expectation_id=eid,
            source_type=source_type,
            basis=basis,
            description=description,
            timestamp="123",
            freshness=0.8,
            strength=0.6,
            reference_count=0,
            evidence_ids=(),
            competing_ids=(),
            revision_count=0,
            undetermined_aspects=(),
        )

    def test_no_competition(self):
        exps = [
            self._make_candidate("e1", ExpectationSourceType.REPETITION,
                                 ExpectationBasis.PATTERN_CONTINUATION, "alpha beta gamma"),
            self._make_candidate("e2", ExpectationSourceType.NARRATIVE,
                                 ExpectationBasis.CONTEXT_PERSISTENCE, "delta epsilon zeta"),
        ]
        pairs = detect_competitions(exps)
        assert len(pairs) == 0

    def test_competition_same_source_different_basis(self):
        exps = [
            self._make_candidate("e1", ExpectationSourceType.SELF_DIFFERENCE,
                                 ExpectationBasis.CHANGE_DIRECTION,
                                 "emotional shift may continue in this direction"),
            self._make_candidate("e2", ExpectationSourceType.SELF_DIFFERENCE,
                                 ExpectationBasis.CONTEXT_PERSISTENCE,
                                 "emotional state may stabilize in this direction"),
        ]
        pairs = detect_competitions(exps)
        assert len(pairs) >= 1

    def test_multiple_competitions(self):
        exps = [
            self._make_candidate("e1", ExpectationSourceType.REPETITION,
                                 ExpectationBasis.PATTERN_CONTINUATION,
                                 "tendency toward exploration will continue"),
            self._make_candidate("e2", ExpectationSourceType.REPETITION,
                                 ExpectationBasis.CHANGE_DIRECTION,
                                 "tendency toward exploration may change direction"),
            self._make_candidate("e3", ExpectationSourceType.NARRATIVE,
                                 ExpectationBasis.CONTEXT_PERSISTENCE,
                                 "completely different topic here"),
        ]
        pairs = detect_competitions(exps)
        assert len(pairs) >= 1


# =============================================================================
# Test ExpectationFormationSystem
# =============================================================================

class TestExpectationFormationSystem:
    def test_creation(self):
        system = ExpectationFormationSystem()
        store = system.get_store()
        assert store.has_expectations() is False

    def test_form_with_tendency(self):
        system = ExpectationFormationSystem()
        bias = MockTendencyBias(has_bias=True)
        store = system.form_expectations(tendency_bias=bias)
        assert store.has_expectations() is True
        assert store.total_expectations_created >= 1

    def test_form_with_difference(self):
        system = ExpectationFormationSystem()
        diff = MockDifferenceSummary(has_difference=True)
        store = system.form_expectations(difference_summary=diff)
        assert store.has_expectations() is True

    def test_form_with_narrative(self):
        system = ExpectationFormationSystem()
        narrative = MockNarrativeState()
        store = system.form_expectations(narrative_state=narrative)
        assert store.has_expectations() is True

    def test_form_with_all_sources(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(
            tendency_bias=MockTendencyBias(has_bias=True),
            difference_summary=MockDifferenceSummary(has_difference=True),
            narrative_state=MockNarrativeState(),
        )
        assert store.has_expectations() is True
        assert store.total_expectations_created >= 3

    def test_form_with_none_inputs(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations()
        assert store.has_expectations() is False

    def test_decay(self):
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        store_before = system.get_store()

        # Apply multiple decay cycles
        for _ in range(5):
            store_after = system.decay_expectations()

        if store_after.has_expectations():
            assert store_after.average_freshness <= store_before.average_freshness

    def test_reference_expectation(self):
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        store = system.get_store()
        if store.expectations:
            exp_id = store.expectations[0].expectation_id
            old_freshness = store.expectations[0].freshness
            system.reference_expectation(exp_id)
            updated = system.get_store()
            found = [e for e in updated.expectations if e.expectation_id == exp_id]
            assert len(found) == 1
            assert found[0].reference_count >= 1

    def test_revise_expectation(self):
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        store = system.get_store()
        if store.expectations:
            exp_id = store.expectations[0].expectation_id
            system.revise_expectation(exp_id, "Revised description")
            updated = system.get_store()
            found = [e for e in updated.expectations if e.expectation_id == exp_id]
            assert len(found) == 1
            assert found[0].description == "Revised description"
            assert found[0].revision_count >= 1

    def test_get_active_expectations(self):
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        active = system.get_active_expectations(max_count=5)
        assert isinstance(active, list)

    def test_get_last_store(self):
        system = ExpectationFormationSystem()
        assert system.get_last_store() is None
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        assert system.get_last_store() is not None

    def test_capacity_limit(self):
        config = ExpectationFormationConfig(max_expectations=3)
        system = ExpectationFormationSystem(config=config)
        # Add many expectations
        for i in range(10):
            bias = MockTendencyBias(
                has_bias=True,
                strongest_category=f"category_{i}",
                strongest_bias=MockTendency(strength=0.5 + i * 0.01, category=f"cat_{i}"),
            )
            system.form_expectations(tendency_bias=bias)
        store = system.get_store()
        assert len(store.expectations) <= 3

    def test_evidence_links_generated(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        assert len(store.evidence_links) >= 1

    def test_multiple_form_calls_accumulate(self):
        system = ExpectationFormationSystem()
        store1 = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        count1 = store1.total_expectations_created
        store2 = system.form_expectations(difference_summary=MockDifferenceSummary(has_difference=True))
        assert store2.total_expectations_created > count1


# =============================================================================
# Test Form From Chain
# =============================================================================

class TestFormFromChain:
    def test_basic(self):
        system = ExpectationFormationSystem()
        store = form_from_chain(
            system,
            tendency_bias=MockTendencyBias(has_bias=True),
        )
        assert store.has_expectations() is True

    def test_all_sources(self):
        system = ExpectationFormationSystem()
        store = form_from_chain(
            system,
            tendency_bias=MockTendencyBias(has_bias=True),
            difference_summary=MockDifferenceSummary(has_difference=True),
            narrative_state=MockNarrativeState(),
        )
        assert store.total_expectations_created >= 3


# =============================================================================
# Test Generate Expectation Tags
# =============================================================================

class TestGenerateExpectationTags:
    def test_none_store(self):
        tags = generate_expectation_tags(None)
        assert len(tags) >= 1
        assert tags[0]["category"] == "EXPECTATION_COUNT"
        assert tags[0]["label"] == "no_expectations"

    def test_with_expectations(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        tags = generate_expectation_tags(store)
        assert len(tags) >= 3
        categories = [t["category"] for t in tags]
        assert "EXPECTATION_COUNT" in categories
        assert "EXPECTATION_STRENGTH" in categories
        assert "EXPECTATION_FRESHNESS" in categories

    def test_scale(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        tags_normal = generate_expectation_tags(store, scale=1.0)
        tags_scaled = generate_expectation_tags(store, scale=2.0)
        if tags_normal and tags_scaled:
            assert tags_scaled[0]["weight"] > tags_normal[0]["weight"]

    def test_category_types(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(
            tendency_bias=MockTendencyBias(has_bias=True),
            difference_summary=MockDifferenceSummary(has_difference=True),
        )
        tags = generate_expectation_tags(store)
        for tag in tags:
            assert "category" in tag
            assert "label" in tag
            assert "description" in tag
            assert "weight" in tag


# =============================================================================
# Test Get Expectation Summary
# =============================================================================

class TestGetExpectationSummary:
    def test_empty(self):
        summary = get_expectation_summary(None)
        assert "No expectations" in summary

    def test_with_expectations(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        summary = get_expectation_summary(store)
        assert "Expectation Formation State" in summary
        assert "Total expectations" in summary


# =============================================================================
# Test Get Expectation For Introspection
# =============================================================================

class TestGetExpectationForIntrospection:
    def test_empty(self):
        result = get_expectation_for_introspection(None)
        assert result["has_expectations"] is False
        assert result["total_expectations"] == 0

    def test_with_expectations(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        result = get_expectation_for_introspection(store)
        assert result["has_expectations"] is True
        assert result["total_expectations"] >= 1
        assert "source_distribution" in result
        assert "basis_distribution" in result
        assert "strongest_expectation_description" in result


# =============================================================================
# Test Verification Functions
# =============================================================================

class TestVerification:
    def test_no_decision_impact(self):
        store = create_empty_store()
        assert verify_no_decision_impact(store) is True

    def test_no_goal_generation(self):
        system = ExpectationFormationSystem()
        assert verify_no_goal_generation(system) is True

    def test_read_only_principle(self):
        system = ExpectationFormationSystem()
        assert verify_read_only_principle(system) is True

    def test_no_value_modification(self):
        system = ExpectationFormationSystem()
        assert verify_no_value_modification(system) is True

    def test_store_with_expectations_no_decision_impact(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        assert verify_no_decision_impact(store) is True


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenience:
    def test_create_config(self):
        config = create_config(max_expectations=50, base_decay_rate=0.05)
        assert config.max_expectations == 50
        assert config.base_decay_rate == 0.05

    def test_create_empty_store(self):
        store = create_empty_store()
        assert store.has_expectations() is False
        assert store.total_expectations_created == 0

    def test_create_system(self):
        system = create_system()
        assert isinstance(system, ExpectationFormationSystem)

    def test_create_system_with_config(self):
        config = create_config(max_expectations=30)
        system = create_system(config=config)
        store = system.get_store()
        assert store.has_expectations() is False


# =============================================================================
# Test Persistence (Save / Load)
# =============================================================================

class TestPersistence:
    def test_save_and_load(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_expectation_state(store, filepath)
            loaded = load_expectation_state(filepath)
            assert loaded.total_expectations_created == store.total_expectations_created
            assert len(loaded.expectations) == len(store.expectations)
            assert loaded.average_freshness == store.average_freshness
        finally:
            os.unlink(filepath)

    def test_roundtrip_preserves_structure(self):
        system = ExpectationFormationSystem()
        store = system.form_expectations(
            tendency_bias=MockTendencyBias(has_bias=True),
            difference_summary=MockDifferenceSummary(has_difference=True),
            narrative_state=MockNarrativeState(),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_expectation_state(store, filepath)
            loaded = load_expectation_state(filepath)

            assert len(loaded.expectations) == len(store.expectations)
            assert len(loaded.evidence_links) == len(store.evidence_links)

            for orig, rest in zip(store.expectations, loaded.expectations):
                assert orig.expectation_id == rest.expectation_id
                assert orig.source_type == rest.source_type
                assert orig.basis == rest.basis
                assert orig.description == rest.description
        finally:
            os.unlink(filepath)


# =============================================================================
# Test Determine Expectation Basis
# =============================================================================

class TestDetermineExpectationBasis:
    def test_single_repetition(self):
        result = determine_expectation_basis([ExpectationSourceType.REPETITION])
        assert result == ExpectationBasis.PATTERN_CONTINUATION

    def test_single_difference(self):
        result = determine_expectation_basis([ExpectationSourceType.SELF_DIFFERENCE])
        assert result == ExpectationBasis.CHANGE_DIRECTION

    def test_single_narrative(self):
        result = determine_expectation_basis([ExpectationSourceType.NARRATIVE])
        assert result == ExpectationBasis.CONTEXT_PERSISTENCE

    def test_multiple_sources(self):
        result = determine_expectation_basis([
            ExpectationSourceType.REPETITION,
            ExpectationSourceType.NARRATIVE,
        ])
        assert result == ExpectationBasis.COMBINED

    def test_empty(self):
        result = determine_expectation_basis([])
        assert result == ExpectationBasis.UNDEFINED


# =============================================================================
# Test Generate Expectation Description
# =============================================================================

class TestGenerateExpectationDescription:
    def test_with_sources(self):
        desc = generate_expectation_description(
            ExpectationBasis.PATTERN_CONTINUATION,
            ["Tendency toward exploration is strong"],
        )
        assert "Based on recurring patterns" in desc
        assert "exploration" in desc

    def test_no_sources(self):
        desc = generate_expectation_description(ExpectationBasis.UNDEFINED, [])
        assert "Weak expectation" in desc


# =============================================================================
# Test Design Principles (Meta-tests)
# =============================================================================

class TestDesignPrinciples:
    """
    設計制約のメタテスト。
    予期形成が判断・目的・価値・責任に接続しないことを検証する。
    """

    def test_no_decision_methods(self):
        """System has no decision-making methods."""
        system = ExpectationFormationSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden_patterns = ["decide", "choose", "select_action", "determine_action"]
        for method in methods:
            for pattern in forbidden_patterns:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"

    def test_no_evaluation_methods(self):
        """System has no success/failure evaluation methods."""
        system = ExpectationFormationSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["evaluate_success", "judge_outcome", "score_result", "check_correctness"]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"

    def test_no_goal_setting(self):
        """System has no goal/purpose setting methods."""
        system = ExpectationFormationSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["set_goal", "create_goal", "generate_goal", "set_purpose"]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"

    def test_no_value_update(self):
        """System has no value/belief update methods."""
        system = ExpectationFormationSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["update_value", "set_belief", "modify_identity"]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"

    def test_no_external_state_modification(self):
        """System does not modify external state."""
        system = ExpectationFormationSystem()
        methods = [m for m in dir(system) if not m.startswith("_")]
        forbidden = ["update_emotion", "update_memory", "modify_tendency", "apply_to_decision"]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower(), f"Found forbidden pattern '{pattern}' in method '{method}'"

    def test_expectations_are_hypothetical(self):
        """Expectations can be revised and are not fixed."""
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        store = system.get_store()
        if store.expectations:
            exp = store.expectations[0]
            revised = exp.revise("Completely different prediction")
            assert revised.description != exp.description
            assert revised.revision_count == exp.revision_count + 1

    def test_competition_allowed(self):
        """Competing expectations are allowed to coexist."""
        system = ExpectationFormationSystem()
        system.form_expectations(
            tendency_bias=MockTendencyBias(has_bias=True),
            difference_summary=MockDifferenceSummary(has_difference=True),
        )
        store = system.get_store()
        # System should accept multiple expectations from different sources
        assert len(store.expectations) >= 2

    def test_natural_decay(self):
        """Expectations naturally decay over time."""
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        initial = system.get_store()

        for _ in range(3):
            system.decay_expectations()

        after = system.get_store()
        if initial.has_expectations() and after.has_expectations():
            assert after.average_freshness <= initial.average_freshness

    def test_undetermined_aspects(self):
        """New expectations have undetermined aspects."""
        system = ExpectationFormationSystem()
        system.form_expectations(tendency_bias=MockTendencyBias(has_bias=True))
        store = system.get_store()
        if store.expectations:
            assert len(store.expectations[0].undetermined_aspects) > 0

    def test_store_is_frozen(self):
        """Store is immutable."""
        store = create_empty_store()
        with pytest.raises(AttributeError):
            store.description = "modified"
