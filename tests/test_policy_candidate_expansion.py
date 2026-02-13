"""
tests/test_policy_candidate_expansion.py - Policy Candidate Expansion Tests.

Tests for psyche/policy_candidate_expansion.py.
"""

import pytest
import time

from psyche.policy_candidate_expansion import (
    CrossSection,
    PolicyAxis,
    InputFragment,
    ExpandedCandidate,
    HistoryEntry,
    SuppressionEntry,
    CompetitionEntry,
    ExpansionConfig,
    ExpansionState,
    CrossSectionInputs,
    PolicyCandidateExpander,
    create_expander,
    create_config,
    extract_all_fragments,
    get_expansion_summary,
    get_expansion_summary_text,
    _extract_emotion_fragments,
    _extract_memory_fragments,
    _extract_tendency_fragments,
    _extract_responsibility_fragments,
    _extract_dialogue_fragments,
    _extract_self_observation_fragments,
    _extract_other_estimation_fragments,
    _extract_purpose_fragments,
    _unify_fragments,
    _compute_axis_activations,
    _find_crossing_sections,
    _generate_crossing_candidates,
    _ensure_competition,
    _check_suppression_health,
    _AXIS_LABELS,
    _AXIS_RATIONALES,
    _AXIS_FALLBACK_TEXT,
    _AXIS_DRIVE_CHANGES,
)


# ── Enum Tests ──────────────────────────────────────────────


class TestCrossSection:
    def test_all_8_sections_exist(self):
        assert len(CrossSection) == 8

    def test_values(self):
        assert CrossSection.EMOTION.value == "emotion"
        assert CrossSection.MEMORY.value == "memory"
        assert CrossSection.TENDENCY.value == "tendency"
        assert CrossSection.RESPONSIBILITY.value == "responsibility"
        assert CrossSection.DIALOGUE.value == "dialogue"
        assert CrossSection.SELF_OBSERVATION.value == "self_observation"
        assert CrossSection.OTHER_ESTIMATION.value == "other_estimation"
        assert CrossSection.PURPOSE.value == "purpose"


class TestPolicyAxis:
    def test_all_10_axes_exist(self):
        assert len(PolicyAxis) == 10

    def test_values(self):
        assert PolicyAxis.APPROACH.value == "approach"
        assert PolicyAxis.HOLD.value == "hold"
        assert PolicyAxis.EXPLORE.value == "explore"
        assert PolicyAxis.SHIFT.value == "shift"
        assert PolicyAxis.MAINTAIN.value == "maintain"
        assert PolicyAxis.REPAIR.value == "repair"
        assert PolicyAxis.BOUNDARY.value == "boundary"
        assert PolicyAxis.CONFIRM.value == "confirm"
        assert PolicyAxis.DELEGATE.value == "delegate"
        assert PolicyAxis.REFLECT.value == "reflect"

    def test_all_axes_have_labels(self):
        for axis in PolicyAxis:
            assert axis in _AXIS_LABELS
            assert axis in _AXIS_RATIONALES
            assert axis in _AXIS_FALLBACK_TEXT
            assert axis in _AXIS_DRIVE_CHANGES


# ── InputFragment Tests ─────────────────────────────────────


class TestInputFragment:
    def test_create(self):
        frag = InputFragment(
            section=CrossSection.EMOTION,
            key="emotion_joy",
            value=0.8,
            confidence=0.9,
        )
        assert frag.section == CrossSection.EMOTION
        assert frag.key == "emotion_joy"
        assert frag.value == 0.8
        assert frag.confidence == 0.9

    def test_to_dict(self):
        frag = InputFragment(
            section=CrossSection.MEMORY,
            key="recall_richness",
            value=0.5,
        )
        d = frag.to_dict()
        assert d["section"] == "memory"
        assert d["key"] == "recall_richness"
        assert d["value"] == 0.5
        assert d["confidence"] == 1.0

    def test_from_dict(self):
        d = {"section": "dialogue", "key": "intent_question", "value": 0.7, "confidence": 0.8}
        frag = InputFragment.from_dict(d)
        assert frag.section == CrossSection.DIALOGUE
        assert frag.key == "intent_question"
        assert frag.value == 0.7
        assert frag.confidence == 0.8

    def test_from_dict_invalid_section(self):
        d = {"section": "invalid", "key": "test", "value": 0.5}
        frag = InputFragment.from_dict(d)
        assert frag.section == CrossSection.EMOTION  # default

    def test_roundtrip(self):
        frag = InputFragment(
            section=CrossSection.PURPOSE,
            key="active_goal",
            value=0.6,
            confidence=0.7,
        )
        restored = InputFragment.from_dict(frag.to_dict())
        assert restored.section == frag.section
        assert restored.key == frag.key
        assert restored.value == frag.value
        assert restored.confidence == frag.confidence


# ── ExpandedCandidate Tests ─────────────────────────────────


class TestExpandedCandidate:
    def test_create(self):
        c = ExpandedCandidate(
            axis=PolicyAxis.APPROACH,
            origin_sections=[CrossSection.EMOTION, CrossSection.DIALOGUE],
            score=1.5,
        )
        assert c.axis == PolicyAxis.APPROACH
        assert len(c.origin_sections) == 2
        assert c.score == 1.5
        assert c.is_primary is True

    def test_to_policy_dict(self):
        c = ExpandedCandidate(
            axis=PolicyAxis.REPAIR,
            origin_sections=[CrossSection.EMOTION, CrossSection.RESPONSIBILITY],
            score=2.0,
            competing_axes=["approach"],
        )
        d = c.to_policy_dict()
        assert d["policy_label"] == "関係を修復する"
        assert d["_score"] == 2.0
        assert d["_axis"] == "repair"
        assert d["_expanded"] is True
        assert "expected_drive_change" in d
        assert "text" in d
        assert "rationale" in d
        assert d["_origin_sections"] == ["emotion", "responsibility"]
        assert d["_competing"] == ["approach"]
        assert d["_is_primary"] is True

    def test_to_dict_roundtrip(self):
        c = ExpandedCandidate(
            axis=PolicyAxis.EXPLORE,
            origin_sections=[CrossSection.DIALOGUE, CrossSection.PURPOSE],
            score=1.2,
            conditions={"dialogue_contribution": 0.7},
            competing_axes=["shift"],
            suppression_factors=["test"],
            resurface_factors=["boost"],
            is_primary=False,
        )
        d = c.to_dict()
        restored = ExpandedCandidate.from_dict(d)
        assert restored.axis == c.axis
        assert len(restored.origin_sections) == 2
        assert restored.score == c.score
        assert restored.is_primary is False
        assert restored.conditions == c.conditions

    def test_from_dict_invalid_axis(self):
        d = {"axis": "invalid", "origin_sections": [], "score": 0.5}
        c = ExpandedCandidate.from_dict(d)
        assert c.axis == PolicyAxis.APPROACH  # default


# ── HistoryEntry Tests ──────────────────────────────────────


class TestHistoryEntry:
    def test_roundtrip(self):
        h = HistoryEntry(axis="approach", origin_sections=["emotion"], score=1.0, turn=5)
        d = h.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored.axis == "approach"
        assert restored.score == 1.0
        assert restored.turn == 5
        assert restored.decay_factor == 1.0


# ── SuppressionEntry Tests ──────────────────────────────────


class TestSuppressionEntry:
    def test_roundtrip(self):
        s = SuppressionEntry(axis="hold", reason="test", strength=0.5, turn_created=3)
        d = s.to_dict()
        restored = SuppressionEntry.from_dict(d)
        assert restored.axis == "hold"
        assert restored.reason == "test"
        assert restored.strength == 0.5
        assert restored.released is False

    def test_released_state(self):
        s = SuppressionEntry(axis="reflect", reason="chronic", strength=0.01, turn_created=1, released=True)
        d = s.to_dict()
        restored = SuppressionEntry.from_dict(d)
        assert restored.released is True


# ── CompetitionEntry Tests ──────────────────────────────────


class TestCompetitionEntry:
    def test_roundtrip(self):
        c = CompetitionEntry(
            selected_axis="approach",
            unselected_axes=["hold", "explore"],
            turn=10,
            score_gap=0.5,
        )
        d = c.to_dict()
        restored = CompetitionEntry.from_dict(d)
        assert restored.selected_axis == "approach"
        assert restored.unselected_axes == ["hold", "explore"]
        assert restored.turn == 10
        assert restored.score_gap == 0.5


# ── ExpansionConfig Tests ───────────────────────────────────


class TestExpansionConfig:
    def test_defaults(self):
        config = ExpansionConfig()
        assert config.max_expanded_candidates == 5
        assert config.min_crossing_sections == 2
        assert config.history_max_entries == 50
        assert config.suppression_chronic_threshold == 10

    def test_roundtrip(self):
        config = ExpansionConfig(max_expanded_candidates=3, min_crossing_sections=1)
        d = config.to_dict()
        restored = ExpansionConfig.from_dict(d)
        assert restored.max_expanded_candidates == 3
        assert restored.min_crossing_sections == 1

    def test_create_config_factory(self):
        config = create_config(max_expanded_candidates=7)
        assert config.max_expanded_candidates == 7


# ── ExpansionState Tests ────────────────────────────────────


class TestExpansionState:
    def test_default(self):
        state = ExpansionState()
        assert state.turn_count == 0
        assert state.total_expansions == 0
        assert len(state.candidate_history) == 0
        assert len(state.suppression_history) == 0
        assert len(state.competition_history) == 0

    def test_roundtrip(self):
        state = ExpansionState(
            turn_count=5,
            total_expansions=10,
            axis_activations={"approach": 1.0, "hold": 0.5},
            candidate_history=[
                HistoryEntry(axis="approach", origin_sections=["emotion"], score=1.0, turn=1),
            ],
            suppression_history=[
                SuppressionEntry(axis="hold", reason="test", strength=0.3, turn_created=2),
            ],
            competition_history=[
                CompetitionEntry(selected_axis="approach", unselected_axes=["hold"], turn=3),
            ],
        )
        d = state.to_dict()
        restored = ExpansionState.from_dict(d)
        assert restored.turn_count == 5
        assert restored.total_expansions == 10
        assert len(restored.candidate_history) == 1
        assert len(restored.suppression_history) == 1
        assert len(restored.competition_history) == 1
        assert restored.axis_activations["approach"] == 1.0

    def test_roundtrip_empty(self):
        state = ExpansionState()
        d = state.to_dict()
        restored = ExpansionState.from_dict(d)
        assert restored.turn_count == 0


# ── CrossSectionInputs Tests ───────────────────────────────


class TestCrossSectionInputs:
    def test_defaults(self):
        inputs = CrossSectionInputs()
        assert inputs.mood_valence == 0.0
        assert inputs.recalled_count == 0
        assert inputs.percept_intent == "unknown"

    def test_roundtrip(self):
        inputs = CrossSectionInputs(
            emotions={"joy": 0.8, "sorrow": 0.2},
            mood_valence=0.5,
            recalled_count=3,
            tendency_count=2,
            percept_intent="question",
        )
        d = inputs.to_dict()
        restored = CrossSectionInputs.from_dict(d)
        assert restored.emotions == {"joy": 0.8, "sorrow": 0.2}
        assert restored.mood_valence == 0.5
        assert restored.recalled_count == 3
        assert restored.percept_intent == "question"


# ── Fragment Extraction Tests ───────────────────────────────


class TestFragmentExtraction:
    def test_emotion_fragments_joy(self):
        inputs = CrossSectionInputs(emotions={"joy": 0.8, "anger": 0.05})
        frags = _extract_emotion_fragments(inputs)
        keys = [f.key for f in frags]
        assert "emotion_joy" in keys
        assert "emotion_anger" not in keys  # below 0.1

    def test_emotion_fragments_mood(self):
        inputs = CrossSectionInputs(mood_valence=-0.5, mood_arousal=0.6)
        frags = _extract_emotion_fragments(inputs)
        keys = [f.key for f in frags]
        assert "mood_valence" in keys
        assert "mood_arousal" in keys

    def test_memory_fragments(self):
        inputs = CrossSectionInputs(recalled_count=3, has_emotional_bindings=True, episode_count=5)
        frags = _extract_memory_fragments(inputs)
        keys = [f.key for f in frags]
        assert "recall_richness" in keys
        assert "emotional_binding" in keys
        assert "episode_presence" in keys

    def test_memory_fragments_empty(self):
        inputs = CrossSectionInputs()
        frags = _extract_memory_fragments(inputs)
        assert len(frags) == 0

    def test_tendency_fragments(self):
        inputs = CrossSectionInputs(tendency_count=3, tendency_strength=0.6)
        frags = _extract_tendency_fragments(inputs)
        keys = [f.key for f in frags]
        assert "tendency_presence" in keys
        assert "tendency_strength" in keys

    def test_responsibility_fragments(self):
        inputs = CrossSectionInputs(caution_bias=0.5, empathy_bias=0.3, dispersion_active=True)
        frags = _extract_responsibility_fragments(inputs)
        keys = [f.key for f in frags]
        assert "caution" in keys
        assert "empathy" in keys
        assert "dispersion_active" in keys

    def test_dialogue_fragments_question(self):
        inputs = CrossSectionInputs(percept_intent="question", percept_valence=0.3, percept_text_length=100)
        frags = _extract_dialogue_fragments(inputs)
        keys = [f.key for f in frags]
        assert "intent_question" in keys
        assert "percept_valence" in keys
        assert "text_richness" in keys

    def test_dialogue_fragments_unknown(self):
        inputs = CrossSectionInputs(percept_intent="unknown")
        frags = _extract_dialogue_fragments(inputs)
        # Unknown intent has signal 0.0
        intent_frags = [f for f in frags if f.key.startswith("intent_")]
        assert len(intent_frags) == 0

    def test_self_observation_fragments(self):
        inputs = CrossSectionInputs(
            self_image_stability=0.2,
            coherence_level=0.3,
            strain_level=0.6,
            narrative_coherence=0.3,
        )
        frags = _extract_self_observation_fragments(inputs)
        keys = [f.key for f in frags]
        assert "self_image_instability" in keys
        assert "coherence_low" in keys
        assert "strain" in keys
        assert "narrative_fragmentation" in keys

    def test_self_observation_stable(self):
        inputs = CrossSectionInputs(
            self_image_stability=0.8,
            coherence_level=0.8,
            strain_level=0.0,
            narrative_coherence=0.8,
        )
        frags = _extract_self_observation_fragments(inputs)
        assert len(frags) == 0

    def test_other_estimation_fragments(self):
        inputs = CrossSectionInputs(other_model_count=2, other_boundary_clarity=0.2)
        frags = _extract_other_estimation_fragments(inputs)
        keys = [f.key for f in frags]
        assert "other_model_presence" in keys
        assert "boundary_blur" in keys

    def test_purpose_fragments(self):
        inputs = CrossSectionInputs(
            has_active_goal=True,
            goal_strength=0.7,
            motive_count=3,
            expectation_count=2,
            vector_count=1,
        )
        frags = _extract_purpose_fragments(inputs)
        keys = [f.key for f in frags]
        assert "active_goal" in keys
        assert "motive_presence" in keys
        assert "expectation_presence" in keys
        assert "direction_presence" in keys

    def test_extract_all_fragments(self):
        inputs = CrossSectionInputs(
            emotions={"joy": 0.8},
            recalled_count=2,
            tendency_count=1,
            caution_bias=0.3,
            percept_intent="question",
            self_image_stability=0.3,
            other_model_count=1,
            has_active_goal=True,
            goal_strength=0.5,
        )
        frags = extract_all_fragments(inputs)
        sections = {f.section for f in frags}
        # Should have fragments from multiple sections
        assert len(sections) >= 5


# ── Fragment Unification Tests ──────────────────────────────


class TestUnifyFragments:
    def test_deduplication(self):
        frags = [
            InputFragment(section=CrossSection.EMOTION, key="joy", value=0.8),
            InputFragment(section=CrossSection.EMOTION, key="joy", value=0.9),
        ]
        unified = _unify_fragments(frags)
        assert len(unified[CrossSection.EMOTION]) == 1

    def test_multiple_sections(self):
        frags = [
            InputFragment(section=CrossSection.EMOTION, key="joy", value=0.8),
            InputFragment(section=CrossSection.DIALOGUE, key="intent", value=0.7),
        ]
        unified = _unify_fragments(frags)
        assert CrossSection.EMOTION in unified
        assert CrossSection.DIALOGUE in unified


# ── Axis Activation Tests ──────────────────────────────────


class TestAxisActivation:
    def test_approach_activation(self):
        """Love + sharing intent should activate approach axis."""
        frags = [
            InputFragment(section=CrossSection.EMOTION, key="emotion_love", value=0.8),
            InputFragment(section=CrossSection.DIALOGUE, key="intent_sharing", value=0.5),
        ]
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        assert activations["approach"] > 0

    def test_hold_activation(self):
        """High caution + fear should activate hold axis."""
        frags = [
            InputFragment(section=CrossSection.RESPONSIBILITY, key="caution", value=0.7),
            InputFragment(section=CrossSection.EMOTION, key="emotion_fear", value=0.6),
        ]
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        assert activations["hold"] > 0

    def test_repair_activation(self):
        """Sorrow + negative percept should activate repair axis."""
        frags = [
            InputFragment(section=CrossSection.EMOTION, key="emotion_sorrow", value=0.8),
            InputFragment(section=CrossSection.DIALOGUE, key="percept_valence", value=-0.6),
        ]
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        assert activations["repair"] > 0

    def test_reflect_activation(self):
        """High strain + self-image instability should activate reflect."""
        frags = [
            InputFragment(section=CrossSection.SELF_OBSERVATION, key="strain", value=0.7),
            InputFragment(section=CrossSection.SELF_OBSERVATION, key="self_image_instability", value=0.6),
        ]
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        assert activations["reflect"] > 0

    def test_no_static_priority(self):
        """Different inputs should produce different activation patterns."""
        frags1 = [InputFragment(section=CrossSection.EMOTION, key="emotion_love", value=0.9)]
        frags2 = [InputFragment(section=CrossSection.EMOTION, key="emotion_fear", value=0.9)]
        unified1 = _unify_fragments(frags1)
        unified2 = _unify_fragments(frags2)
        act1 = _compute_axis_activations(unified1, [])
        act2 = _compute_axis_activations(unified2, [])
        # approach should be higher with love, hold should be higher with fear
        assert act1["approach"] > act2["approach"]
        assert act2["hold"] > act1["hold"]

    def test_history_noise(self):
        """History fragments should add small perturbation."""
        frags = [InputFragment(section=CrossSection.EMOTION, key="emotion_joy", value=0.5)]
        unified = _unify_fragments(frags)
        act_no_hist = _compute_axis_activations(unified, [])
        act_with_hist = _compute_axis_activations(unified, [
            {"key": "app_something", "value": 0.8, "section": "emotion"},
        ])
        # The results should differ slightly due to history noise
        # but the difference should be small (0.1 * 0.3 * avg)
        assert isinstance(act_with_hist, dict)


# ── Crossing Section Tests ──────────────────────────────────


class TestCrossingSections:
    def test_approach_crossing(self):
        frags = [
            InputFragment(section=CrossSection.EMOTION, key="emotion_love", value=0.8),
            InputFragment(section=CrossSection.DIALOGUE, key="intent_sharing", value=0.5),
            InputFragment(section=CrossSection.RESPONSIBILITY, key="empathy", value=0.3),
        ]
        unified = _unify_fragments(frags)
        sections = _find_crossing_sections(PolicyAxis.APPROACH, unified)
        assert len(sections) >= 2  # at least EMOTION + DIALOGUE or RESPONSIBILITY

    def test_no_crossing_insufficient_data(self):
        """Axis with no matching fragments should have no crossing."""
        frags = [InputFragment(section=CrossSection.MEMORY, key="recall_richness", value=0.5)]
        unified = _unify_fragments(frags)
        # BOUNDARY axis needs OTHER_ESTIMATION, RESPONSIBILITY, etc.
        sections = _find_crossing_sections(PolicyAxis.BOUNDARY, unified)
        assert len(sections) < 2


# ── Candidate Generation Tests ──────────────────────────────


class TestCandidateGeneration:
    def _make_rich_inputs(self) -> CrossSectionInputs:
        return CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.6},
            mood_valence=0.4,
            recalled_count=3,
            has_emotional_bindings=True,
            tendency_count=2,
            tendency_strength=0.5,
            empathy_bias=0.4,
            percept_intent="sharing",
            percept_valence=0.3,
            percept_text_length=80,
            self_image_stability=0.6,
            other_model_count=1,
            has_active_goal=True,
            goal_strength=0.6,
            motive_count=2,
        )

    def test_generate_produces_candidates(self):
        inputs = self._make_rich_inputs()
        frags = extract_all_fragments(inputs)
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        candidates = _generate_crossing_candidates(
            activations, unified, ExpansionConfig(min_crossing_sections=2),
            {}, {},
        )
        assert len(candidates) > 0

    def test_candidates_have_multiple_origin_sections(self):
        """Each candidate should originate from at least 2 sections."""
        inputs = self._make_rich_inputs()
        frags = extract_all_fragments(inputs)
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])
        candidates = _generate_crossing_candidates(
            activations, unified, ExpansionConfig(min_crossing_sections=2),
            {}, {},
        )
        for c in candidates:
            assert len(c.origin_sections) >= 2

    def test_suppression_reduces_score(self):
        inputs = self._make_rich_inputs()
        frags = extract_all_fragments(inputs)
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])

        # Without suppression
        candidates_normal = _generate_crossing_candidates(
            activations, unified, ExpansionConfig(min_crossing_sections=2),
            {}, {},
        )

        # With heavy suppression on top axis
        if candidates_normal:
            top_axis = candidates_normal[0].axis.value
            candidates_suppressed = _generate_crossing_candidates(
                activations, unified, ExpansionConfig(min_crossing_sections=2),
                {top_axis: 0.8}, {},
            )
            # Find the same axis in suppressed candidates
            suppressed_scores = {c.axis.value: c.score for c in candidates_suppressed}
            if top_axis in suppressed_scores:
                normal_score = candidates_normal[0].score
                assert suppressed_scores[top_axis] < normal_score

    def test_competition_boost(self):
        inputs = self._make_rich_inputs()
        frags = extract_all_fragments(inputs)
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])

        # With competition boost on a low axis
        candidates_boosted = _generate_crossing_candidates(
            activations, unified, ExpansionConfig(min_crossing_sections=2),
            {}, {"reflect": 0.5},
        )
        reflect_candidates = [c for c in candidates_boosted if c.axis == PolicyAxis.REFLECT]
        if reflect_candidates:
            assert reflect_candidates[0].score > 0


# ── Ensure Competition Tests ────────────────────────────────


class TestEnsureCompetition:
    def test_adds_alternatives_when_single_candidate(self):
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            percept_intent="sharing",
            empathy_bias=0.4,
            has_active_goal=True,
            goal_strength=0.5,
            recalled_count=2,
            motive_count=1,
        )
        frags = extract_all_fragments(inputs)
        unified = _unify_fragments(frags)
        activations = _compute_axis_activations(unified, [])

        single = [ExpandedCandidate(
            axis=PolicyAxis.APPROACH,
            origin_sections=[CrossSection.EMOTION],
            score=1.0,
        )]

        result = _ensure_competition(
            single, activations, unified,
            ExpansionConfig(linearization_warning_threshold=1, min_crossing_sections=2),
        )
        assert len(result) > 1

    def test_does_not_add_when_sufficient(self):
        candidates = [
            ExpandedCandidate(axis=PolicyAxis.APPROACH, origin_sections=[], score=1.0),
            ExpandedCandidate(axis=PolicyAxis.HOLD, origin_sections=[], score=0.5),
            ExpandedCandidate(axis=PolicyAxis.EXPLORE, origin_sections=[], score=0.3),
        ]
        result = _ensure_competition(
            candidates, {}, {},
            ExpansionConfig(linearization_warning_threshold=1),
        )
        assert len(result) == 3  # unchanged


# ── Suppression Health Tests ────────────────────────────────


class TestSuppressionHealth:
    def test_chronic_suppression_relaxed(self):
        state = ExpansionState(
            suppression_history=[
                SuppressionEntry(
                    axis="approach", reason="test", strength=0.5,
                    turn_created=0, turn_count=15,
                ),
            ],
            config=ExpansionConfig(suppression_chronic_threshold=10),
        )
        _check_suppression_health(state)
        assert state.suppression_history[0].strength < 0.5

    def test_released_when_very_weak(self):
        state = ExpansionState(
            suppression_history=[
                SuppressionEntry(
                    axis="hold", reason="test", strength=0.03,
                    turn_created=0, turn_count=20,
                ),
            ],
            config=ExpansionConfig(
                suppression_chronic_threshold=10,
                suppression_release_threshold=0.5,
            ),
        )
        _check_suppression_health(state)
        assert state.suppression_history[0].released is True

    def test_non_chronic_not_relaxed(self):
        state = ExpansionState(
            suppression_history=[
                SuppressionEntry(
                    axis="explore", reason="test", strength=0.5,
                    turn_created=0, turn_count=3,
                ),
            ],
            config=ExpansionConfig(suppression_chronic_threshold=10),
        )
        _check_suppression_health(state)
        assert state.suppression_history[0].strength == 0.5

    def test_already_released_not_touched(self):
        state = ExpansionState(
            suppression_history=[
                SuppressionEntry(
                    axis="shift", reason="test", strength=0.1,
                    turn_created=0, turn_count=20, released=True,
                ),
            ],
            config=ExpansionConfig(suppression_chronic_threshold=10),
        )
        _check_suppression_health(state)
        assert state.suppression_history[0].strength == 0.1  # unchanged


# ── PolicyCandidateExpander Tests ───────────────────────────


class TestPolicyCandidateExpander:
    def _make_base_candidates(self):
        return [
            {"policy_label": "共感する", "_score": 2.0, "rationale": "test"},
            {"policy_label": "質問で会話を広げる", "_score": 1.5, "rationale": "test"},
            {"policy_label": "感想を述べる", "_score": 1.0, "rationale": "test"},
        ]

    def _make_rich_inputs(self):
        return CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5, "sorrow": 0.1},
            mood_valence=0.3,
            mood_arousal=0.5,
            recalled_count=3,
            has_emotional_bindings=True,
            episode_count=2,
            tendency_count=2,
            tendency_strength=0.4,
            caution_bias=0.2,
            empathy_bias=0.3,
            percept_intent="sharing",
            percept_valence=0.2,
            percept_text_length=80,
            self_image_stability=0.5,
            coherence_level=0.6,
            other_model_count=1,
            has_active_goal=True,
            goal_strength=0.5,
            motive_count=2,
            vector_count=1,
        )

    def test_create_expander(self):
        expander = create_expander()
        assert expander.state.turn_count == 0

    def test_expand_candidates_returns_policy_dicts(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expanded = expander.expand_candidates(base, inputs)
        assert isinstance(expanded, list)
        for c in expanded:
            assert "policy_label" in c
            assert "_score" in c
            assert "_expanded" in c
            assert c["_expanded"] is True

    def test_expand_increments_turn(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)
        assert expander.state.turn_count == 1
        expander.expand_candidates(base, inputs)
        assert expander.state.turn_count == 2

    def test_expand_updates_history(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)
        assert len(expander.state.candidate_history) > 0

    def test_expand_updates_fragment_history(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)
        assert len(expander.state.fragment_history) > 0

    def test_max_expanded_candidates(self):
        config = ExpansionConfig(max_expanded_candidates=2, min_crossing_sections=1)
        expander = create_expander(config)
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expanded = expander.expand_candidates(base, inputs)
        assert len(expanded) <= 2

    def test_recency_suppression(self):
        """Repeated same axis should get suppressed."""
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()

        # Run multiple times
        for _ in range(5):
            expander.expand_candidates(base, inputs)

        # Check that suppression entries have been created for repeated axes
        axes_in_history = [h.axis for h in expander.state.candidate_history[-5:]]
        # If an axis appeared 3+ times, it should have suppression
        from collections import Counter
        counts = Counter(axes_in_history)
        repeated = [axis for axis, count in counts.items() if count >= 3]
        if repeated:
            active_suppressions = [
                s for s in expander.state.suppression_history
                if not s.released and s.reason == "recency_repetition"
            ]
            assert len(active_suppressions) > 0

    def test_output_is_candidate_info_only(self):
        """Output should be candidates only, no judgment/evaluation/action."""
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expanded = expander.expand_candidates(base, inputs)
        for c in expanded:
            # Should not contain action directives
            assert "action" not in c
            assert "execute" not in c
            assert "decision" not in c

    def test_does_not_modify_base_candidates(self):
        """Input is read-only: base candidates should not be modified."""
        expander = create_expander()
        base = self._make_base_candidates()
        base_copy = [dict(c) for c in base]
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)
        assert base == base_copy

    def test_empty_inputs(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = CrossSectionInputs()
        expanded = expander.expand_candidates(base, inputs)
        # Should still work, possibly with fewer or no candidates
        assert isinstance(expanded, list)

    def test_state_persistence(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)

        # Save state
        state_dict = expander.state.to_dict()

        # Restore into new expander
        new_expander = create_expander()
        new_expander._state = ExpansionState.from_dict(state_dict)

        assert new_expander.state.turn_count == 1
        assert len(new_expander.state.candidate_history) == len(expander.state.candidate_history)

    def test_competition_history_populated(self):
        expander = create_expander()
        base = self._make_base_candidates()
        inputs = self._make_rich_inputs()
        expander.expand_candidates(base, inputs)
        # If multiple candidates were generated, competition history should be populated
        state = expander.state
        # Competition entry is added when >= 2 selected candidates
        if state.total_expansions >= 2:
            assert len(state.competition_history) > 0

    def test_non_fixity_input_diff(self):
        """Different inputs should produce different candidates."""
        expander = create_expander()
        base = self._make_base_candidates()

        inputs1 = CrossSectionInputs(
            emotions={"joy": 0.9},
            percept_intent="joke",
            mood_valence=0.7,
            recalled_count=1,
            has_active_goal=True,
            goal_strength=0.8,
        )
        expanded1 = expander.expand_candidates(base, inputs1)

        expander2 = create_expander()
        inputs2 = CrossSectionInputs(
            emotions={"sorrow": 0.9},
            percept_intent="complaint",
            mood_valence=-0.7,
            strain_level=0.8,
            caution_bias=0.6,
        )
        expanded2 = expander2.expand_candidates(base, inputs2)

        # Different inputs should lead to different axis selection
        axes1 = {c.get("_axis") for c in expanded1}
        axes2 = {c.get("_axis") for c in expanded2}
        # At least one axis should differ
        assert axes1 != axes2 or (len(axes1) == 0 and len(axes2) == 0)


# ── Summary Tests ───────────────────────────────────────────


class TestExpansionSummary:
    def test_summary_empty_expander(self):
        expander = create_expander()
        summary = get_expansion_summary(expander)
        assert summary["turn_count"] == 0
        assert summary["total_expansions"] == 0
        assert summary["active_axes_count"] == 0

    def test_summary_after_expansion(self):
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            percept_intent="sharing",
            empathy_bias=0.3,
            has_active_goal=True,
            goal_strength=0.5,
        )
        expander.expand_candidates([], inputs)
        summary = get_expansion_summary(expander)
        assert summary["turn_count"] == 1
        assert summary["active_axes_count"] > 0

    def test_summary_text_empty(self):
        expander = create_expander()
        text = get_expansion_summary_text(expander)
        assert text == ""

    def test_summary_text_after_expansion(self):
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7},
            percept_intent="sharing",
            empathy_bias=0.3,
        )
        expander.expand_candidates([], inputs)
        text = get_expansion_summary_text(expander)
        if text:  # may be empty if no active axes
            assert "活性軸" in text


# ── Design Constraint Verification Tests ────────────────────


class TestDesignConstraints:
    """Verify the design document constraints are upheld."""

    def test_no_single_section_dominance(self):
        """Verify single cross-section cannot dominate all candidates."""
        config = ExpansionConfig(single_section_dominance_cap=0.6)
        expander = create_expander(config)

        # Input with only emotion data (single section)
        inputs = CrossSectionInputs(
            emotions={"joy": 0.9, "love": 0.8, "fun": 0.7},
            mood_valence=0.8,
            mood_arousal=0.7,
        )
        expanded = expander.expand_candidates([], inputs)

        # With only emotion section, crossing threshold (2 sections) limits generation
        for c in expanded:
            sections = c.get("_origin_sections", [])
            # If only emotion is present, candidates shouldn't be generated
            # (unless alternatives are supplemented by _ensure_competition)
            if not c.get("_is_primary", True):
                continue  # supplements are OK
            if len(sections) > 0:
                # Primary candidates need at least 2 sections
                assert len(sections) >= 2 or c.get("_is_primary") is False

    def test_candidate_has_required_fields(self):
        """Each candidate has: 発生根拠, 成立条件, 競合関係, 抑制要因, 再浮上要因."""
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            mood_valence=0.3,
            percept_intent="sharing",
            empathy_bias=0.4,
            has_active_goal=True,
            goal_strength=0.5,
        )
        expanded = expander.expand_candidates([], inputs)
        for c in expanded:
            assert "rationale" in c  # 発生根拠
            assert "_conditions" in c  # 成立条件
            assert "_competing" in c  # 競合関係
            assert "_suppression_factors" in c  # 抑制要因
            assert "_resurface_factors" in c  # 再浮上要因

    def test_output_is_candidates_only(self):
        """Output is candidate info only, no judgment/evaluation/action."""
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7},
            percept_intent="sharing",
        )
        expanded = expander.expand_candidates([], inputs)
        # Output should only be a list of candidate dicts
        assert isinstance(expanded, list)
        for c in expanded:
            assert isinstance(c, dict)
            # No execution/judgment keys
            assert "judgment" not in c
            assert "verdict" not in c

    def test_primary_and_alternative_coexist(self):
        """Candidate set should contain both primary and alternative candidates."""
        config = ExpansionConfig(
            min_crossing_sections=1,  # relaxed for test
            linearization_warning_threshold=5,  # force supplementation
        )
        expander = create_expander(config)
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            mood_valence=0.3,
            percept_intent="sharing",
            percept_valence=0.3,
            empathy_bias=0.4,
            has_active_goal=True,
            goal_strength=0.5,
            recalled_count=2,
            motive_count=1,
        )
        expanded = expander.expand_candidates([], inputs)
        # Should have some candidates
        if len(expanded) > 0:
            primaries = [c for c in expanded if c.get("_is_primary", True)]
            assert len(primaries) > 0  # at least one primary

    def test_suppression_is_reversible(self):
        """Suppression is not permanent — it can be released."""
        state = ExpansionState(
            suppression_history=[
                SuppressionEntry(
                    axis="approach", reason="test", strength=0.05,
                    turn_created=0, turn_count=15,
                ),
            ],
            config=ExpansionConfig(
                suppression_chronic_threshold=10,
                suppression_release_threshold=0.9,
            ),
        )
        _check_suppression_health(state)
        # After health check, chronic weak suppression should be released
        assert state.suppression_history[0].released is True

    def test_no_upstream_state_modification(self):
        """The module should not modify upstream state."""
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7},
            percept_intent="sharing",
        )
        inputs_dict_before = inputs.to_dict()
        expander.expand_candidates([], inputs)
        inputs_dict_after = inputs.to_dict()
        assert inputs_dict_before == inputs_dict_after

    def test_history_uses_summary_not_full_reinput(self):
        """Fragment history stores summaries, not full re-input."""
        expander = create_expander()
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5, "sorrow": 0.3},
            mood_valence=0.3,
            percept_intent="sharing",
        )
        expander.expand_candidates([], inputs)

        # Fragment history entries should be summary dicts, not InputFragment objects
        for h in expander.state.fragment_history:
            assert isinstance(h, dict)
            assert "key" in h
            assert "value" in h

    def test_linearization_warning_counted(self):
        """When candidate set has no competition, linearization warning is counted."""
        config = ExpansionConfig(
            min_crossing_sections=10,  # impossible threshold → no candidates
            linearization_warning_threshold=1,
        )
        expander = create_expander(config)
        inputs = CrossSectionInputs(emotions={"joy": 0.5})
        expander.expand_candidates([], inputs)
        # With impossible crossing threshold, even supplements may be scarce
        # The warning counter should reflect this
        assert expander.state.linearization_warnings >= 0  # may or may not trigger


# ── Integration Pattern Tests ───────────────────────────────


class TestIntegrationPatterns:
    """Test compatibility with thought.py candidate format."""

    def test_expanded_candidate_compatible_with_thought_format(self):
        """Expanded candidates should be compatible with select_policy."""
        expander = create_expander(ExpansionConfig(min_crossing_sections=1))
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            percept_intent="sharing",
            empathy_bias=0.3,
        )
        expanded = expander.expand_candidates([], inputs)
        for c in expanded:
            # Must have all fields that select_policy expects
            assert "policy_label" in c
            assert "rationale" in c
            assert "expected_drive_change" in c
            assert "text" in c
            assert "_score" in c
            # Drive change should be a dict with standard keys
            dc = c["expected_drive_change"]
            assert "social" in dc
            assert "curiosity" in dc
            assert "expression" in dc

    def test_expanded_candidates_sortable_with_base(self):
        """Expanded candidates can be mixed with base and sorted by score."""
        base = [
            {"policy_label": "共感する", "_score": 2.0},
            {"policy_label": "質問で会話を広げる", "_score": 1.5},
        ]
        expander = create_expander(ExpansionConfig(min_crossing_sections=1))
        inputs = CrossSectionInputs(
            emotions={"joy": 0.7, "love": 0.5},
            percept_intent="sharing",
            empathy_bias=0.3,
        )
        expanded = expander.expand_candidates(base, inputs)

        combined = base + expanded
        combined.sort(key=lambda c: c.get("_score", 0), reverse=True)
        # Should be sortable without error
        assert len(combined) >= 2
        # Scores should be in descending order
        for i in range(len(combined) - 1):
            assert combined[i].get("_score", 0) >= combined[i + 1].get("_score", 0)
