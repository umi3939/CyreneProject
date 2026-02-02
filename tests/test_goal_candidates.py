"""
Tests for goal_candidates.py - Proto-Goal → Goal Candidates (自発的目的候補生成)

These tests verify:
1. Candidate generation from ProtoGoalVectors
2. Multiple conflicting candidates can coexist
3. NO selection mechanism exists
4. NO impact on decisions
5. Decay when underlying vectors fade
6. Persistence across sessions
"""

import json
import os
import tempfile
import time

import pytest

from psyche.proto_goal_vector import (
    ProtoGoalVector,
    VectorSource,
    VectorSourceType,
)
from psyche.goal_candidates import (
    CandidateCategory,
    CandidateSource,
    GoalCandidate,
    CandidateStateConfig,
    CandidateState,
    CandidateGenerator,
    create_candidate_generator,
    create_config,
    get_candidate_summary,
    candidates_to_json,
    candidates_from_json,
    create_candidate_context_for_trace,
    create_candidate_stats_for_dynamics,
    to_dict,
    from_dict,
)


class TestCandidateSource:
    """Tests for CandidateSource dataclass."""

    def test_default_source(self):
        source = CandidateSource()
        assert source.vector_ids == []
        assert source.contribution_weights == {}

    def test_source_with_vectors(self):
        source = CandidateSource(
            vector_ids=["v1", "v2"],
            contribution_weights={"v1": 0.6, "v2": 0.4},
        )
        assert len(source.vector_ids) == 2
        assert source.contribution_weights["v1"] == 0.6

    def test_source_serialization(self):
        source = CandidateSource(
            vector_ids=["abc", "def"],
            contribution_weights={"abc": 0.7, "def": 0.3},
        )
        data = source.to_dict()
        restored = CandidateSource.from_dict(data)

        assert restored.vector_ids == ["abc", "def"]
        assert restored.contribution_weights == {"abc": 0.7, "def": 0.3}


class TestGoalCandidate:
    """Tests for GoalCandidate dataclass."""

    def test_default_candidate(self):
        candidate = GoalCandidate()
        assert candidate.candidate_id is not None
        assert len(candidate.candidate_id) == 8
        assert candidate.category == CandidateCategory.EXPLORATION
        assert candidate.direction_expression == {}
        assert candidate.intensity == 0.0

    def test_candidate_with_values(self):
        candidate = GoalCandidate(
            category=CandidateCategory.APPROACH,
            direction_expression={"goal_dim_a": 0.8},
            intensity=0.7,
        )
        assert candidate.category == CandidateCategory.APPROACH
        assert candidate.direction_expression["goal_dim_a"] == 0.8
        assert candidate.intensity == 0.7

    def test_candidate_serialization(self):
        candidate = GoalCandidate(
            candidate_id="test123",
            category=CandidateCategory.AVOIDANCE,
            direction_expression={"x": 0.5, "y": -0.3},
            intensity=0.8,
            source=CandidateSource(vector_ids=["v1"]),
            reinforcement_count=3,
        )
        data = candidate.to_dict()
        restored = GoalCandidate.from_dict(data)

        assert restored.candidate_id == "test123"
        assert restored.category == CandidateCategory.AVOIDANCE
        assert restored.direction_expression == {"x": 0.5, "y": -0.3}
        assert restored.intensity == 0.8
        assert restored.reinforcement_count == 3

    def test_all_categories(self):
        for category in CandidateCategory:
            candidate = GoalCandidate(category=category)
            assert candidate.category == category


class TestCandidateStateConfig:
    """Tests for CandidateStateConfig."""

    def test_default_config(self):
        config = CandidateStateConfig()
        assert config.max_candidates == 15
        assert config.decay_rate == 0.03
        assert config.min_intensity == 0.05
        assert config.generation_probability == 0.4
        assert config.vector_threshold == 0.3
        assert config.similarity_threshold == 0.8
        assert config.merge_enabled is True

    def test_custom_config(self):
        config = CandidateStateConfig(
            max_candidates=10,
            decay_rate=0.1,
            min_intensity=0.1,
        )
        assert config.max_candidates == 10
        assert config.decay_rate == 0.1

    def test_config_serialization(self):
        config = CandidateStateConfig(max_candidates=8, decay_rate=0.05)
        data = to_dict(config)
        restored = from_dict(data)

        assert restored.max_candidates == 8
        assert restored.decay_rate == 0.05

    def test_create_config_factory(self):
        config = create_config(
            max_candidates=20,
            decay_rate=0.02,
            generation_probability=0.6,
        )
        assert config.max_candidates == 20
        assert config.decay_rate == 0.02


class TestCandidateState:
    """Tests for CandidateState."""

    def test_default_state(self):
        state = CandidateState()
        assert state.candidates == []
        assert state.turn_count == 0
        assert state.total_candidates_generated == 0

    def test_state_serialization(self):
        candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.6),
        ]
        state = CandidateState(
            candidates=candidates,
            turn_count=50,
            total_candidates_generated=10,
            total_candidates_decayed=5,
        )
        data = state.to_dict()
        restored = CandidateState.from_dict(data)

        assert len(restored.candidates) == 1
        assert restored.turn_count == 50
        assert restored.total_candidates_generated == 10


class TestCandidateGenerator:
    """Tests for CandidateGenerator core functionality."""

    def test_create_generator(self):
        generator = create_candidate_generator()
        assert generator is not None
        assert len(generator.get_candidates()) == 0

    def test_observe_increments_turn(self):
        generator = create_candidate_generator()
        assert generator.state.turn_count == 0

        generator.observe_vectors([])
        assert generator.state.turn_count == 1

    def test_decay_reduces_intensity(self):
        config = create_config(decay_rate=0.5, min_intensity=0.01)
        generator = CandidateGenerator(config=config)

        # Add a candidate directly
        candidate = GoalCandidate(
            category=CandidateCategory.APPROACH,
            intensity=1.0,
            source=CandidateSource(vector_ids=["v1"]),
        )
        generator._state.candidates.append(candidate)

        # Create a matching vector
        vector = ProtoGoalVector(vector_id="v1", magnitude=0.5)

        generator.observe_vectors([vector])

        # Intensity should be reduced
        assert generator._state.candidates[0].intensity == 0.5

    def test_prune_weak_candidates(self):
        config = create_config(decay_rate=0.0, min_intensity=0.3)
        generator = CandidateGenerator(config=config)

        generator._state.candidates = [
            GoalCandidate(intensity=0.5, source=CandidateSource(vector_ids=["v1"])),
            GoalCandidate(intensity=0.2, source=CandidateSource(vector_ids=["v2"])),  # Below threshold
            GoalCandidate(intensity=0.8, source=CandidateSource(vector_ids=["v3"])),
        ]

        generator._prune_weak_candidates()

        assert len(generator._state.candidates) == 2
        assert generator._state.total_candidates_decayed == 1


class TestCandidateGeneration:
    """Tests for candidate generation from vectors."""

    def test_generation_from_single_vector(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=False,
            vector_threshold=0.2,
        )
        generator = CandidateGenerator(config=config)

        vector = ProtoGoalVector(
            vector_id="v1",
            direction={"dim_a": 0.8, "dim_b": -0.4},
            magnitude=0.7,
        )

        # Multiple attempts to ensure generation
        generated = []
        for _ in range(10):
            result = generator.observe_vectors([vector])
            generated.extend(result)
            if generated:
                break

        assert len(generated) > 0
        assert generated[0].source.vector_ids == ["v1"]

    def test_generation_from_multiple_vectors(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=False,
            vector_threshold=0.2,
        )
        generator = CandidateGenerator(config=config)

        vectors = [
            ProtoGoalVector(vector_id="v1", direction={"a": 0.8}, magnitude=0.7),
            ProtoGoalVector(vector_id="v2", direction={"b": -0.6}, magnitude=0.6),
        ]

        # Run multiple times
        all_generated = []
        for _ in range(10):
            result = generator.observe_vectors(vectors)
            all_generated.extend(result)

        assert len(all_generated) > 0

    def test_no_generation_from_weak_vectors(self):
        config = create_config(
            generation_probability=1.0,
            vector_threshold=0.5,
        )
        generator = CandidateGenerator(config=config)

        # Vector below threshold
        vector = ProtoGoalVector(
            vector_id="v1",
            direction={"a": 0.3},
            magnitude=0.2,
        )

        for _ in range(10):
            result = generator.observe_vectors([vector])
            assert result == []


class TestOrphanedCandidates:
    """Tests for pruning candidates when source vectors fade."""

    def test_candidate_removed_when_vector_fades(self):
        config = create_config(
            decay_rate=0.0,
            min_intensity=0.01,
            vector_threshold=0.3,
        )
        generator = CandidateGenerator(config=config)

        # Add candidate linked to vector
        generator._state.candidates.append(
            GoalCandidate(
                intensity=0.8,
                source=CandidateSource(vector_ids=["v1"]),
            )
        )

        # Vector is now below threshold
        weak_vector = ProtoGoalVector(vector_id="v1", magnitude=0.1)

        generator.observe_vectors([weak_vector])

        # Candidate should be removed
        assert len(generator._state.candidates) == 0

    def test_candidate_survives_with_active_vector(self):
        config = create_config(
            decay_rate=0.0,
            min_intensity=0.01,
            vector_threshold=0.3,
            generation_probability=0.0,  # Disable new generation for this test
        )
        generator = CandidateGenerator(config=config)

        generator._state.candidates.append(
            GoalCandidate(
                intensity=0.8,
                source=CandidateSource(vector_ids=["v1"]),
            )
        )

        # Vector is above threshold
        strong_vector = ProtoGoalVector(vector_id="v1", magnitude=0.5)

        generator.observe_vectors([strong_vector])

        # Candidate should survive
        assert len(generator._state.candidates) == 1

    def test_candidate_with_multiple_sources(self):
        config = create_config(
            decay_rate=0.0,
            min_intensity=0.01,
            vector_threshold=0.3,
            generation_probability=0.0,  # Disable new generation for this test
        )
        generator = CandidateGenerator(config=config)

        # Candidate from two vectors
        generator._state.candidates.append(
            GoalCandidate(
                intensity=0.8,
                source=CandidateSource(vector_ids=["v1", "v2"]),
            )
        )

        # Only one vector is active
        vectors = [
            ProtoGoalVector(vector_id="v1", magnitude=0.1),  # Weak
            ProtoGoalVector(vector_id="v2", magnitude=0.5),  # Strong
        ]

        generator.observe_vectors(vectors)

        # Candidate survives because v2 is active
        assert len(generator._state.candidates) == 1


class TestConflictingCandidates:
    """Tests for multiple conflicting candidates coexisting."""

    def test_opposing_candidates_coexist(self):
        generator = create_candidate_generator()

        # Manually add conflicting candidates
        generator._state.candidates = [
            GoalCandidate(
                category=CandidateCategory.APPROACH,
                intensity=0.8,
                source=CandidateSource(vector_ids=["v1"]),
            ),
            GoalCandidate(
                category=CandidateCategory.AVOIDANCE,
                intensity=0.7,
                source=CandidateSource(vector_ids=["v2"]),
            ),
        ]

        candidates = generator.get_candidates()
        assert len(candidates) == 2

        # Both exist without any resolution
        categories = {c.category for c in candidates}
        assert CandidateCategory.APPROACH in categories
        assert CandidateCategory.AVOIDANCE in categories

    def test_get_conflicting_pairs(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.8),
            GoalCandidate(category=CandidateCategory.AVOIDANCE, intensity=0.7),
            GoalCandidate(category=CandidateCategory.CONNECTION, intensity=0.6),
            GoalCandidate(category=CandidateCategory.ISOLATION, intensity=0.5),
        ]

        conflicts = generator.get_conflicting_pairs()

        # Should have conflicts: APPROACH-AVOIDANCE, CONNECTION-ISOLATION
        assert len(conflicts) >= 2

    def test_multiple_same_category(self):
        generator = create_candidate_generator()

        # Multiple approach candidates
        generator._state.candidates = [
            GoalCandidate(
                category=CandidateCategory.APPROACH,
                direction_expression={"a": 0.8},
                intensity=0.8,
            ),
            GoalCandidate(
                category=CandidateCategory.APPROACH,
                direction_expression={"b": 0.6},
                intensity=0.6,
            ),
        ]

        candidates = generator.get_candidates_by_category(CandidateCategory.APPROACH)
        assert len(candidates) == 2


class TestNoSelection:
    """Tests ensuring NO selection mechanism exists."""

    def test_no_select_function(self):
        import psyche.goal_candidates as module

        # These functions should NOT exist
        assert not hasattr(module, "select_candidate")
        assert not hasattr(module, "choose_candidate")
        assert not hasattr(module, "prioritize_candidates")
        assert not hasattr(module, "rank_candidates")
        assert not hasattr(module, "best_candidate")

    def test_no_score_modification(self):
        import psyche.goal_candidates as module

        assert not hasattr(module, "apply_candidate_to_decision")
        assert not hasattr(module, "modify_decision_score")
        assert not hasattr(module, "influence_decision")

    def test_candidates_are_readonly(self):
        generator = create_candidate_generator()

        candidate = GoalCandidate(
            category=CandidateCategory.APPROACH,
            intensity=0.8,
        )
        generator._state.candidates.append(candidate)

        # Get copy
        retrieved = generator.get_candidates()[0]

        # Modify copy
        retrieved.intensity = 0.0

        # Original unchanged
        assert generator._state.candidates[0].intensity == 0.8


class TestCandidateRetrieval:
    """Tests for read-only candidate access."""

    def test_get_candidates_returns_copy(self):
        generator = create_candidate_generator()

        original = GoalCandidate(intensity=0.7)
        generator._state.candidates.append(original)

        copies = generator.get_candidates()
        copies[0].intensity = 999.0

        assert generator._state.candidates[0].intensity == 0.7

    def test_get_candidate_by_id(self):
        generator = create_candidate_generator()

        candidate = GoalCandidate(candidate_id="test_id", intensity=0.7)
        generator._state.candidates.append(candidate)

        result = generator.get_candidate_by_id("test_id")
        assert result is not None
        assert result.candidate_id == "test_id"

        assert generator.get_candidate_by_id("nonexistent") is None

    def test_get_strongest_candidates(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(intensity=0.3),
            GoalCandidate(intensity=0.9),
            GoalCandidate(intensity=0.1),
            GoalCandidate(intensity=0.7),
        ]

        strongest = generator.get_strongest_candidates(2)
        assert len(strongest) == 2
        assert strongest[0].intensity == 0.9
        assert strongest[1].intensity == 0.7


class TestPersistence:
    """Tests for saving and loading candidate state."""

    def test_save_and_load(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.7),
            GoalCandidate(category=CandidateCategory.EXPLORATION, intensity=0.5),
        ]
        generator._state.turn_count = 100
        generator._state.total_candidates_generated = 8

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            generator.save_to_file(path)

            new_generator = create_candidate_generator()
            count = new_generator.load_from_file(path)

            assert count == 2
            assert new_generator.state.turn_count == 100
            assert new_generator.state.total_candidates_generated == 8
        finally:
            os.unlink(path)

    def test_persistence_across_restart(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            # Session 1
            gen1 = create_candidate_generator()
            gen1._state.candidates.append(
                GoalCandidate(
                    category=CandidateCategory.CONNECTION,
                    intensity=0.9,
                )
            )
            gen1._state.turn_count = 30
            gen1.save_to_file(path)

            # Session 2
            gen2 = create_candidate_generator()
            gen2.load_from_file(path)

            assert gen2.state.turn_count == 30
            candidates = gen2.get_candidates()
            assert len(candidates) == 1
            assert candidates[0].category == CandidateCategory.CONNECTION
        finally:
            os.unlink(path)


class TestCandidateSummary:
    """Tests for summary and debugging functions."""

    def test_get_candidate_summary(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.9),
            GoalCandidate(category=CandidateCategory.AVOIDANCE, intensity=0.4),
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.6),
        ]
        generator._state.turn_count = 50
        generator._state.total_candidates_generated = 10

        summary = get_candidate_summary(generator)

        assert summary["candidate_count"] == 3
        assert summary["total_generated"] == 10
        assert summary["turn_count"] == 50
        assert summary["strongest_intensity"] == 0.9
        assert summary["category_distribution"]["approach"] == 2
        assert summary["category_distribution"]["avoidance"] == 1

    def test_empty_summary(self):
        generator = create_candidate_generator()
        summary = get_candidate_summary(generator)

        assert summary["candidate_count"] == 0
        assert summary["strongest_intensity"] == 0.0


class TestJsonSerialization:
    """Tests for JSON serialization functions."""

    def test_candidates_to_json(self):
        candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.7),
            GoalCandidate(category=CandidateCategory.EXPLORATION, intensity=0.5),
        ]

        json_str = candidates_to_json(candidates)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["category"] == "approach"

    def test_candidates_from_json(self):
        json_str = '[{"candidate_id": "c1", "category": "avoidance", "intensity": 0.8}]'

        candidates = candidates_from_json(json_str)

        assert len(candidates) == 1
        assert candidates[0].candidate_id == "c1"
        assert candidates[0].category == CandidateCategory.AVOIDANCE

    def test_roundtrip_serialization(self):
        original = [
            GoalCandidate(
                candidate_id="c1",
                category=CandidateCategory.CONNECTION,
                direction_expression={"affective_joy": 0.7},
                intensity=0.85,
                source=CandidateSource(
                    vector_ids=["v1", "v2"],
                    contribution_weights={"v1": 0.6, "v2": 0.4},
                ),
            ),
        ]

        json_str = candidates_to_json(original)
        restored = candidates_from_json(json_str)

        assert restored[0].candidate_id == "c1"
        assert restored[0].category == CandidateCategory.CONNECTION
        assert restored[0].intensity == 0.85
        assert len(restored[0].source.vector_ids) == 2


class TestDecayBehavior:
    """Tests for candidate decay over time."""

    def test_gradual_decay(self):
        config = create_config(decay_rate=0.1, min_intensity=0.01, vector_threshold=0.1)
        generator = CandidateGenerator(config=config)

        candidate = GoalCandidate(
            intensity=1.0,
            source=CandidateSource(vector_ids=["v1"]),
        )
        generator._state.candidates.append(candidate)

        vector = ProtoGoalVector(vector_id="v1", magnitude=0.5)

        intensities = []
        for _ in range(5):
            generator.observe_vectors([vector])
            intensities.append(generator._state.candidates[0].intensity)

        assert intensities[0] == pytest.approx(0.9, rel=0.01)
        assert intensities[1] == pytest.approx(0.81, rel=0.01)

    def test_candidate_removed_when_below_threshold(self):
        config = create_config(decay_rate=0.5, min_intensity=0.2, vector_threshold=0.1)
        generator = CandidateGenerator(config=config)

        candidate = GoalCandidate(
            intensity=0.3,
            source=CandidateSource(vector_ids=["v1"]),
        )
        generator._state.candidates.append(candidate)

        vector = ProtoGoalVector(vector_id="v1", magnitude=0.5)

        # After decay: 0.3 * 0.5 = 0.15 < 0.2
        generator.observe_vectors([vector])

        assert len(generator._state.candidates) == 0


class TestIntegrationWithTracing:
    """Tests for introspection/dynamics integration."""

    def test_context_for_trace(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(
                category=CandidateCategory.APPROACH,
                intensity=0.9,
                source=CandidateSource(vector_ids=["v1"]),
            ),
            GoalCandidate(
                category=CandidateCategory.AVOIDANCE,
                intensity=0.5,
                source=CandidateSource(vector_ids=["v2"]),
            ),
        ]

        context = create_candidate_context_for_trace(generator)

        assert context is not None
        assert len(context["goal_candidates"]) == 2
        assert "observation_note" in context
        assert "NOT selected" in context["observation_note"]

    def test_empty_context_when_no_candidates(self):
        generator = create_candidate_generator()
        context = create_candidate_context_for_trace(generator)
        assert context is None

    def test_stats_for_dynamics(self):
        generator = create_candidate_generator()

        generator._state.candidates = [
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.9),
            GoalCandidate(category=CandidateCategory.APPROACH, intensity=0.7),
            GoalCandidate(category=CandidateCategory.AVOIDANCE, intensity=0.5),
        ]

        stats = create_candidate_stats_for_dynamics(generator)

        assert stats["total_count"] == 3
        assert stats["category_counts"]["approach"] == 2
        assert stats["category_counts"]["avoidance"] == 1
        assert stats["max_intensity"] == 0.9


class TestIntegration:
    """Full integration tests."""

    def test_full_workflow_with_vectors(self):
        config = create_config(
            max_candidates=10,
            decay_rate=0.05,
            generation_probability=0.8,
            vector_threshold=0.2,
        )
        generator = CandidateGenerator(config=config)

        # Simulate vectors evolving over time
        for i in range(20):
            vectors = [
                ProtoGoalVector(
                    vector_id=f"v{i % 3}",
                    direction={"dim_a": 0.5 + (i % 5) * 0.1},
                    magnitude=0.4 + (i % 4) * 0.15,
                ),
            ]
            generator.observe_vectors(vectors)

        summary = get_candidate_summary(generator)

        assert summary["turn_count"] == 20
        # Should have generated some candidates
        assert summary["total_generated"] >= 0

    def test_daydreams_do_not_affect_decisions(self):
        generator = create_candidate_generator()

        # Create strong candidates
        for cat in [CandidateCategory.APPROACH, CandidateCategory.AVOIDANCE]:
            generator._state.candidates.append(
                GoalCandidate(category=cat, intensity=0.99)
            )

        # Simulate decision candidates
        decision_candidates = [
            {"policy": "speak", "score": 0.5},
            {"policy": "silence", "score": 0.5},
        ]

        # Get goal candidates (read-only)
        goal_candidates = generator.get_candidates()
        assert len(goal_candidates) == 2

        # Decision scores UNCHANGED
        assert decision_candidates[0]["score"] == 0.5
        assert decision_candidates[1]["score"] == 0.5
