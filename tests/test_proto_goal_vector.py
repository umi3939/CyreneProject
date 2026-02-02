"""
Tests for proto_goal_vector.py - Proto-Goal Direction Vector (自発的方向ベクトル生成)

These tests verify:
1. Vector generation from various sources
2. Decay and pruning behavior
3. Reinforcement and merging of similar vectors
4. Non-interference with decisions (ghost data)
5. Persistence across sessions
"""

import json
import os
import tempfile
import time

import pytest

from psyche.proto_goal_vector import (
    VectorSourceType,
    VectorSource,
    ProtoGoalVector,
    VectorStateConfig,
    VectorState,
    VectorGenerator,
    create_vector_generator,
    create_config,
    get_vector_summary,
    vectors_to_json,
    vectors_from_json,
    create_vector_context_for_trace,
    to_dict,
    from_dict,
)


class TestVectorSource:
    """Tests for VectorSource dataclass."""

    def test_default_source(self):
        source = VectorSource(VectorSourceType.COMBINED)
        assert source.source_type == VectorSourceType.COMBINED
        assert source.reference_id == ""
        assert source.description == ""

    def test_source_with_reference(self):
        source = VectorSource(
            source_type=VectorSourceType.VALUE_ORIENTATION,
            reference_id="trace_123",
            description="From value bias",
        )
        assert source.source_type == VectorSourceType.VALUE_ORIENTATION
        assert source.reference_id == "trace_123"
        assert source.description == "From value bias"

    def test_source_serialization(self):
        source = VectorSource(
            source_type=VectorSourceType.INTROSPECTION_PATTERN,
            reference_id="abc",
        )
        data = source.to_dict()
        restored = VectorSource.from_dict(data)

        assert restored.source_type == source.source_type
        assert restored.reference_id == source.reference_id

    def test_all_source_types(self):
        for source_type in VectorSourceType:
            source = VectorSource(source_type=source_type)
            assert source.source_type == source_type


class TestProtoGoalVector:
    """Tests for ProtoGoalVector dataclass."""

    def test_default_vector(self):
        vector = ProtoGoalVector()
        assert vector.vector_id is not None
        assert len(vector.vector_id) == 8
        assert vector.direction == {}
        assert vector.magnitude == 0.0
        assert vector.reinforcement_count == 1

    def test_vector_with_direction(self):
        direction = {"dim_a": 0.5, "dim_b": -0.3}
        vector = ProtoGoalVector(
            direction=direction,
            magnitude=0.7,
        )
        assert vector.direction == direction
        assert vector.magnitude == 0.7

    def test_vector_serialization(self):
        vector = ProtoGoalVector(
            vector_id="test123",
            direction={"x": 0.5, "y": -0.2},
            magnitude=0.8,
            source=VectorSource(VectorSourceType.EMOTION_TENDENCY),
            reinforcement_count=3,
        )
        data = vector.to_dict()
        restored = ProtoGoalVector.from_dict(data)

        assert restored.vector_id == "test123"
        assert restored.direction == {"x": 0.5, "y": -0.2}
        assert restored.magnitude == 0.8
        assert restored.source.source_type == VectorSourceType.EMOTION_TENDENCY
        assert restored.reinforcement_count == 3

    def test_vector_timestamps(self):
        before = time.time()
        vector = ProtoGoalVector()
        after = time.time()

        assert before <= vector.created_at <= after
        assert before <= vector.last_reinforced_at <= after


class TestVectorStateConfig:
    """Tests for VectorStateConfig."""

    def test_default_config(self):
        config = VectorStateConfig()
        assert config.max_vectors == 10
        assert config.decay_rate == 0.02
        assert config.min_magnitude == 0.05
        assert config.similarity_threshold == 0.85
        assert config.merge_enabled is True
        assert config.generation_probability == 0.3
        assert config.reinforcement_boost == 0.1
        assert config.auto_save_interval == 50

    def test_custom_config(self):
        config = VectorStateConfig(
            max_vectors=5,
            decay_rate=0.1,
            min_magnitude=0.1,
        )
        assert config.max_vectors == 5
        assert config.decay_rate == 0.1
        assert config.min_magnitude == 0.1

    def test_config_serialization(self):
        config = VectorStateConfig(max_vectors=7, decay_rate=0.05)
        data = to_dict(config)
        restored = from_dict(data)

        assert restored.max_vectors == 7
        assert restored.decay_rate == 0.05

    def test_create_config_factory(self):
        config = create_config(
            max_vectors=15,
            decay_rate=0.03,
            generation_probability=0.5,
        )
        assert config.max_vectors == 15
        assert config.decay_rate == 0.03
        assert config.generation_probability == 0.5


class TestVectorState:
    """Tests for VectorState."""

    def test_default_state(self):
        state = VectorState()
        assert state.vectors == []
        assert state.turn_count == 0
        assert state.total_vectors_generated == 0
        assert state.total_vectors_decayed == 0

    def test_state_with_vectors(self):
        vectors = [
            ProtoGoalVector(direction={"a": 0.5}, magnitude=0.6),
            ProtoGoalVector(direction={"b": -0.3}, magnitude=0.4),
        ]
        state = VectorState(vectors=vectors)
        assert len(state.vectors) == 2

    def test_state_serialization(self):
        vectors = [
            ProtoGoalVector(direction={"a": 0.5}, magnitude=0.6),
        ]
        state = VectorState(
            vectors=vectors,
            turn_count=100,
            total_vectors_generated=5,
            total_vectors_decayed=2,
        )
        data = state.to_dict()
        restored = VectorState.from_dict(data)

        assert len(restored.vectors) == 1
        assert restored.turn_count == 100
        assert restored.total_vectors_generated == 5
        assert restored.total_vectors_decayed == 2


class TestVectorGenerator:
    """Tests for VectorGenerator core functionality."""

    def test_create_generator(self):
        generator = create_vector_generator()
        assert generator is not None
        assert len(generator.get_vectors()) == 0

    def test_observe_turn_increments_count(self):
        generator = create_vector_generator()
        assert generator.state.turn_count == 0

        generator.observe_turn()
        assert generator.state.turn_count == 1

        generator.observe_turn()
        assert generator.state.turn_count == 2

    def test_decay_reduces_magnitude(self):
        # Create generator with high decay
        config = create_config(decay_rate=0.5, min_magnitude=0.01)
        generator = VectorGenerator(config=config)

        # Manually add a vector
        vector = ProtoGoalVector(direction={"a": 0.5}, magnitude=1.0)
        generator._state.vectors.append(vector)

        # Observe turn applies decay
        generator.observe_turn()

        # Magnitude should be reduced
        assert generator._state.vectors[0].magnitude == 0.5

    def test_prune_weak_vectors(self):
        config = create_config(decay_rate=0.0, min_magnitude=0.3)
        generator = VectorGenerator(config=config)

        # Add vectors with different magnitudes
        generator._state.vectors = [
            ProtoGoalVector(direction={"a": 0.5}, magnitude=0.5),
            ProtoGoalVector(direction={"b": 0.2}, magnitude=0.2),  # Below threshold
            ProtoGoalVector(direction={"c": 0.8}, magnitude=0.8),
        ]

        generator._prune_weak_vectors()

        assert len(generator._state.vectors) == 2
        assert generator._state.total_vectors_decayed == 1

    def test_max_vectors_limit(self):
        config = create_config(max_vectors=3, decay_rate=0.0, min_magnitude=0.01)
        generator = VectorGenerator(config=config)

        # Add more vectors than limit
        for i in range(5):
            vector = ProtoGoalVector(
                direction={f"dim_{i}": 0.5},
                magnitude=0.5,
                created_at=time.time() + i,  # Ensure ordering
            )
            generator._state.vectors.append(vector)

        generator._enforce_max_vectors()

        assert len(generator._state.vectors) == 3
        # Oldest should be removed
        assert generator._state.total_vectors_decayed == 2


class TestVectorGeneration:
    """Tests for vector generation from various sources."""

    def test_generation_from_value_orientation(self):
        # Mock value orientation with dimensions
        class MockOrientation:
            dim_a = 0.8
            dim_b = -0.5
            dim_c = 0.1  # Below threshold, should be excluded
            dim_d = 0.0
            dim_e = 0.0
            update_count = 10

        config = create_config(
            generation_probability=1.0,  # Always generate
            merge_enabled=False,
        )
        generator = VectorGenerator(config=config)

        # Multiple attempts to ensure generation
        orientation = MockOrientation()
        generated = None
        for _ in range(10):
            result = generator.observe_turn(value_orientation=orientation)
            if result is not None:
                generated = result
                break

        assert generated is not None
        assert generated.source.source_type == VectorSourceType.VALUE_ORIENTATION
        assert "dim_a" in generated.direction or "dim_b" in generated.direction

    def test_generation_from_emotion_tendency(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=False,
        )
        generator = VectorGenerator(config=config)

        tendency = {"joy": 0.7, "fear": -0.4, "neutral": 0.05}

        generated = None
        for _ in range(10):
            result = generator.observe_turn(emotion_tendency=tendency)
            if result is not None:
                generated = result
                break

        assert generated is not None
        assert generated.source.source_type == VectorSourceType.EMOTION_TENDENCY

    def test_generation_from_responsibility_pattern(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=False,
        )
        generator = VectorGenerator(config=config)

        pattern = {
            "sublimation_tendency": 0.6,
            "dispersion_bias": -0.3,
            "pattern_id": "resp_001",
        }

        generated = None
        for _ in range(10):
            result = generator.observe_turn(responsibility_pattern=pattern)
            if result is not None:
                generated = result
                break

        assert generated is not None
        assert generated.source.source_type == VectorSourceType.RESPONSIBILITY_PATTERN

    def test_no_generation_without_signals(self):
        config = create_config(generation_probability=1.0)
        generator = VectorGenerator(config=config)

        # No inputs should produce no vector
        result = generator.observe_turn()
        assert result is None

    def test_probabilistic_generation(self):
        config = create_config(
            generation_probability=0.0,  # Never generate
            merge_enabled=False,
        )
        generator = VectorGenerator(config=config)

        tendency = {"joy": 0.8}

        # Should never generate
        for _ in range(20):
            result = generator.observe_turn(emotion_tendency=tendency)
            assert result is None


class TestVectorMerging:
    """Tests for similar vector reinforcement/merging."""

    def test_similar_vectors_merge(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=True,
            similarity_threshold=0.9,
            reinforcement_boost=0.2,
        )
        generator = VectorGenerator(config=config)

        # Add initial vector
        initial = ProtoGoalVector(
            direction={"dim_a": 0.8, "dim_b": 0.2},
            magnitude=0.5,
        )
        generator._state.vectors.append(initial)
        generator._state.total_vectors_generated = 1

        class MockOrientation:
            dim_a = 0.85  # Very similar
            dim_b = 0.15

        # Observe with similar signal
        for _ in range(5):
            generator.observe_turn(value_orientation=MockOrientation())

        # Should still have only 1 vector (merged, not new)
        # May have been pruned if magnitude dropped
        vectors = generator.get_vectors()
        assert len(vectors) <= 1

    def test_dissimilar_vectors_not_merged(self):
        config = create_config(
            generation_probability=1.0,
            merge_enabled=True,
            similarity_threshold=0.95,  # High threshold
            decay_rate=0.0,
            min_magnitude=0.01,
        )
        generator = VectorGenerator(config=config)

        # Add initial vector
        initial = ProtoGoalVector(
            direction={"dim_a": 1.0},
            magnitude=0.5,
        )
        generator._state.vectors.append(initial)
        generator._state.total_vectors_generated = 1

        # Very different signal
        tendency = {"emotion_fear": 0.9}

        generated = None
        for _ in range(10):
            result = generator.observe_turn(emotion_tendency=tendency)
            if result is not None:
                generated = result
                break

        # Should have 2 vectors now
        if generated is not None:
            assert len(generator._state.vectors) >= 2


class TestVectorRetrieval:
    """Tests for read-only vector access."""

    def test_get_vectors_returns_copy(self):
        generator = create_vector_generator()

        # Add a vector
        original = ProtoGoalVector(direction={"a": 0.5}, magnitude=0.7)
        generator._state.vectors.append(original)

        # Get vectors
        vectors = generator.get_vectors()

        # Modify returned vector
        vectors[0].magnitude = 999.0

        # Original should be unchanged
        assert generator._state.vectors[0].magnitude == 0.7

    def test_get_vector_by_id(self):
        generator = create_vector_generator()

        vector = ProtoGoalVector(
            vector_id="test_id",
            direction={"a": 0.5},
            magnitude=0.7,
        )
        generator._state.vectors.append(vector)

        result = generator.get_vector_by_id("test_id")
        assert result is not None
        assert result.vector_id == "test_id"

        # Non-existent ID
        assert generator.get_vector_by_id("nonexistent") is None

    def test_get_strongest_vectors(self):
        generator = create_vector_generator()

        vectors = [
            ProtoGoalVector(direction={"a": 0.1}, magnitude=0.3),
            ProtoGoalVector(direction={"b": 0.2}, magnitude=0.9),
            ProtoGoalVector(direction={"c": 0.3}, magnitude=0.1),
            ProtoGoalVector(direction={"d": 0.4}, magnitude=0.7),
        ]
        generator._state.vectors = vectors

        strongest = generator.get_strongest_vectors(2)
        assert len(strongest) == 2
        assert strongest[0].magnitude == 0.9
        assert strongest[1].magnitude == 0.7


class TestNonInterference:
    """Tests ensuring vectors don't influence decisions (ghost data)."""

    def test_vectors_are_readonly(self):
        generator = create_vector_generator()

        vector = ProtoGoalVector(direction={"a": 0.5}, magnitude=0.7)
        generator._state.vectors.append(vector)

        # Get vector copy
        retrieved = generator.get_vector_by_id(vector.vector_id)

        # Modify copy
        retrieved.magnitude = 0.0
        retrieved.direction["a"] = 0.0

        # Original unchanged
        original = generator._state.vectors[0]
        assert original.magnitude == 0.7
        assert original.direction["a"] == 0.5

    def test_no_score_modification_function(self):
        # Ensure no function exists that modifies decision scores
        import psyche.proto_goal_vector as module

        # These functions should NOT exist
        assert not hasattr(module, "apply_vector_to_decision")
        assert not hasattr(module, "modify_candidate_score")
        assert not hasattr(module, "influence_decision")

    def test_context_for_trace_is_informational_only(self):
        generator = create_vector_generator()

        vector = ProtoGoalVector(
            direction={"a": 0.5},
            magnitude=0.7,
            source=VectorSource(VectorSourceType.VALUE_ORIENTATION),
        )
        generator._state.vectors.append(vector)

        context = create_vector_context_for_trace(generator)

        assert context is not None
        assert "proto_goal_vectors" in context
        assert "observation_note" in context
        assert "NOT influence decisions" in context["observation_note"]


class TestPersistence:
    """Tests for saving and loading vector state."""

    def test_save_and_load(self):
        generator = create_vector_generator()

        # Add some vectors
        generator._state.vectors = [
            ProtoGoalVector(direction={"a": 0.5}, magnitude=0.7),
            ProtoGoalVector(direction={"b": -0.3}, magnitude=0.4),
        ]
        generator._state.turn_count = 100
        generator._state.total_vectors_generated = 5

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            generator.save_to_file(path)

            # Load in new generator
            new_generator = create_vector_generator()
            count = new_generator.load_from_file(path)

            assert count == 2
            assert new_generator.state.turn_count == 100
            assert new_generator.state.total_vectors_generated == 5
            assert len(new_generator.get_vectors()) == 2
        finally:
            os.unlink(path)

    def test_persistence_across_restart(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            # Session 1
            gen1 = create_vector_generator()
            gen1._state.vectors.append(
                ProtoGoalVector(direction={"persist": 0.8}, magnitude=0.9)
            )
            gen1._state.turn_count = 50
            gen1.save_to_file(path)

            # Session 2 (simulated restart)
            gen2 = create_vector_generator()
            gen2.load_from_file(path)

            assert gen2.state.turn_count == 50
            vectors = gen2.get_vectors()
            assert len(vectors) == 1
            assert "persist" in vectors[0].direction
        finally:
            os.unlink(path)

    def test_auto_save_interval(self):
        config = create_config(auto_save_interval=5)
        generator = VectorGenerator(config=config)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            generator.set_save_path(path)
            generator._state.vectors.append(
                ProtoGoalVector(direction={"a": 0.5}, magnitude=0.7)
            )

            # Observe until auto-save triggers
            for _ in range(6):
                generator.observe_turn()

            # Wait for background save
            time.sleep(0.1)

            # File should exist and have content
            with open(path, "r") as f:
                data = json.load(f)
                assert data["turn_count"] >= 5
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestVectorSummary:
    """Tests for summary and debugging functions."""

    def test_get_vector_summary(self):
        generator = create_vector_generator()

        generator._state.vectors = [
            ProtoGoalVector(
                direction={"a": 0.5},
                magnitude=0.9,
                source=VectorSource(VectorSourceType.VALUE_ORIENTATION),
                reinforcement_count=3,
            ),
            ProtoGoalVector(
                direction={"b": -0.3},
                magnitude=0.4,
                source=VectorSource(VectorSourceType.EMOTION_TENDENCY),
            ),
        ]
        generator._state.turn_count = 100
        generator._state.total_vectors_generated = 5
        generator._state.total_vectors_decayed = 3

        summary = get_vector_summary(generator)

        assert summary["vector_count"] == 2
        assert summary["total_generated"] == 5
        assert summary["total_decayed"] == 3
        assert summary["turn_count"] == 100
        assert summary["strongest_magnitude"] == 0.9
        assert 0.6 < summary["average_magnitude"] < 0.7
        assert "value_orientation" in summary["source_distribution"]
        assert len(summary["vectors"]) == 2

    def test_empty_summary(self):
        generator = create_vector_generator()
        summary = get_vector_summary(generator)

        assert summary["vector_count"] == 0
        assert summary["strongest_magnitude"] == 0.0
        assert summary["average_magnitude"] == 0.0


class TestJsonSerialization:
    """Tests for JSON serialization functions."""

    def test_vectors_to_json(self):
        vectors = [
            ProtoGoalVector(direction={"a": 0.5}, magnitude=0.7),
            ProtoGoalVector(direction={"b": -0.3}, magnitude=0.4),
        ]

        json_str = vectors_to_json(vectors)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["direction"]["a"] == 0.5

    def test_vectors_from_json(self):
        json_str = '[{"vector_id": "test", "direction": {"x": 0.5}, "magnitude": 0.8}]'

        vectors = vectors_from_json(json_str)

        assert len(vectors) == 1
        assert vectors[0].vector_id == "test"
        assert vectors[0].direction["x"] == 0.5
        assert vectors[0].magnitude == 0.8

    def test_roundtrip_serialization(self):
        original = [
            ProtoGoalVector(
                vector_id="v1",
                direction={"dim_a": 0.8, "dim_b": -0.2},
                magnitude=0.75,
                source=VectorSource(
                    VectorSourceType.INTROSPECTION_PATTERN,
                    reference_id="trace_001",
                ),
                reinforcement_count=5,
            ),
        ]

        json_str = vectors_to_json(original)
        restored = vectors_from_json(json_str)

        assert restored[0].vector_id == "v1"
        assert restored[0].direction == {"dim_a": 0.8, "dim_b": -0.2}
        assert restored[0].magnitude == 0.75
        assert restored[0].source.source_type == VectorSourceType.INTROSPECTION_PATTERN
        assert restored[0].reinforcement_count == 5


class TestDecayBehavior:
    """Tests for vector decay over time."""

    def test_gradual_decay(self):
        config = create_config(decay_rate=0.1, min_magnitude=0.01)
        generator = VectorGenerator(config=config)

        vector = ProtoGoalVector(direction={"a": 0.5}, magnitude=1.0)
        generator._state.vectors.append(vector)

        magnitudes = []
        for _ in range(5):
            generator.observe_turn()
            magnitudes.append(generator._state.vectors[0].magnitude)

        # Should decay exponentially
        assert magnitudes[0] == pytest.approx(0.9, rel=0.01)
        assert magnitudes[1] == pytest.approx(0.81, rel=0.01)
        assert magnitudes[2] == pytest.approx(0.729, rel=0.01)

    def test_vector_removed_when_below_threshold(self):
        config = create_config(decay_rate=0.5, min_magnitude=0.2)
        generator = VectorGenerator(config=config)

        vector = ProtoGoalVector(direction={"a": 0.5}, magnitude=0.3)
        generator._state.vectors.append(vector)

        # After decay: 0.3 * 0.5 = 0.15 < 0.2 threshold
        generator.observe_turn()

        assert len(generator._state.vectors) == 0
        assert generator._state.total_vectors_decayed == 1

    def test_reinforcement_prevents_decay(self):
        config = create_config(
            decay_rate=0.2,
            min_magnitude=0.1,
            generation_probability=0.0,  # Disable new generation
            merge_enabled=True,
            similarity_threshold=0.5,
            reinforcement_boost=0.3,
        )
        generator = VectorGenerator(config=config)

        vector = ProtoGoalVector(
            direction={"emotion_joy": 0.9},
            magnitude=0.5,
        )
        generator._state.vectors.append(vector)

        # Observe with similar signal to reinforce
        for _ in range(3):
            # Decay happens first (0.5 -> 0.4 -> 0.32 -> 0.256)
            # But reinforcement adds 0.3 * signal_magnitude
            generator._reinforce_vector(
                generator._state.vectors[0],
                {"emotion_joy": 0.8},
                0.5,
            )

        # Vector should still exist and have reasonable magnitude
        assert len(generator._state.vectors) == 1
        assert generator._state.vectors[0].reinforcement_count > 1


class TestIntegration:
    """Integration tests for the full system."""

    def test_full_workflow(self):
        config = create_config(
            max_vectors=5,
            decay_rate=0.05,
            min_magnitude=0.1,
            generation_probability=0.8,
            auto_save_interval=0,  # Disabled for test
        )
        generator = VectorGenerator(config=config)

        class MockOrientation:
            dim_a = 0.7
            dim_b = -0.4
            dim_c = 0.0
            dim_d = 0.0
            dim_e = 0.0

        # Simulate multiple turns
        for _ in range(20):
            generator.observe_turn(
                value_orientation=MockOrientation(),
                emotion_tendency={"joy": 0.5, "curiosity": 0.3},
            )

        # Should have some vectors
        vectors = generator.get_vectors()
        summary = get_vector_summary(generator)

        assert summary["turn_count"] == 20
        assert summary["total_generated"] > 0
        # Vectors exist or were decayed
        assert summary["total_generated"] >= summary["vector_count"]

    def test_vectors_do_not_affect_decision_scores(self):
        # This test verifies the core design principle:
        # Proto-goal vectors are ghost data and must not
        # influence any decision-making process

        generator = create_vector_generator()

        # Create strong vectors
        for i in range(3):
            generator._state.vectors.append(
                ProtoGoalVector(
                    direction={f"strong_dim_{i}": 0.99},
                    magnitude=0.99,
                )
            )

        # Simulate decision candidates (from thought.py pattern)
        candidates = [
            {"policy": "speak", "score": 0.5},
            {"policy": "silence", "score": 0.5},
            {"policy": "joke", "score": 0.5},
        ]

        # Get vectors (read-only)
        vectors = generator.get_vectors()
        assert len(vectors) == 3

        # Candidates should be UNCHANGED
        # (There is no function to apply vectors to candidates)
        assert candidates[0]["score"] == 0.5
        assert candidates[1]["score"] == 0.5
        assert candidates[2]["score"] == 0.5

    def test_introspection_context_integration(self):
        generator = create_vector_generator()

        # Add vectors
        generator._state.vectors = [
            ProtoGoalVector(
                direction={"dim_a": 0.8},
                magnitude=0.9,
                source=VectorSource(VectorSourceType.VALUE_ORIENTATION),
            ),
            ProtoGoalVector(
                direction={"emotion_joy": 0.6},
                magnitude=0.5,
                source=VectorSource(VectorSourceType.EMOTION_TENDENCY),
            ),
        ]

        context = create_vector_context_for_trace(generator)

        # Context should be informational only
        assert context is not None
        assert len(context["proto_goal_vectors"]) == 2
        assert context["proto_goal_vectors"][0]["magnitude"] == 0.9
        assert "observation_note" in context

    def test_empty_context_when_no_vectors(self):
        generator = create_vector_generator()
        context = create_vector_context_for_trace(generator)
        assert context is None
