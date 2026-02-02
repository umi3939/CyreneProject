"""
Tests for repeated_tendency.py - Repeated Scoped-Goal Tendency (反復傾向の形成)

These tests verify:
1. Tendency formation from repetition (not single use)
2. Natural decay when repetition stops
3. WEAK influence - creates "slope" not "wall"
4. MEDIUM-TERM: distinct from ValueOrientation (long) and TransientGoal (short)
5. No hardcoded rules - this is habit/inertia, NOT personality
"""

import json
import os
import tempfile

import pytest

from psyche.goal_candidates import CandidateCategory
from psyche.scoped_goal import ScopedGoal, ScopeStatus
from psyche.repeated_tendency import (
    TendencyPattern,
    UsageRecord,
    Tendency,
    TendencyBias,
    RepeatedTendencyConfig,
    RepeatedTendencyState,
    RepeatedTendencySystem,
    create_system,
    create_config,
    apply_tendency_bias_to_candidate,
    apply_tendency_bias_to_candidates,
    get_tendency_summary,
    create_tendency_context_for_trace,
    create_tendency_stats_for_dynamics,
    to_dict,
    from_dict,
)


class TestTendencyPattern:
    """Tests for TendencyPattern dataclass."""

    def test_default_pattern(self):
        pattern = TendencyPattern()
        assert pattern.pattern_id is not None
        assert pattern.category == CandidateCategory.EXPLORATION
        assert pattern.direction_signature == {}

    def test_pattern_with_values(self):
        pattern = TendencyPattern(
            category=CandidateCategory.CONNECTION,
            direction_signature={"social": 0.7},
        )
        assert pattern.category == CandidateCategory.CONNECTION
        assert pattern.direction_signature["social"] == 0.7

    def test_pattern_serialization(self):
        pattern = TendencyPattern(
            pattern_id="p1",
            category=CandidateCategory.EXPRESSION,
            direction_signature={"x": 0.5},
        )
        data = pattern.to_dict()
        restored = TendencyPattern.from_dict(data)

        assert restored.pattern_id == "p1"
        assert restored.category == CandidateCategory.EXPRESSION


class TestTendency:
    """Tests for Tendency dataclass."""

    def test_default_tendency(self):
        tendency = Tendency()
        assert tendency.tendency_id is not None
        assert tendency.strength == 0.0
        assert tendency.confidence == 0.0

    def test_tendency_serialization(self):
        tendency = Tendency(
            tendency_id="t1",
            strength=0.1,
            confidence=0.5,
            total_reinforcements=5,
        )
        data = tendency.to_dict()
        restored = Tendency.from_dict(data)

        assert restored.tendency_id == "t1"
        assert restored.strength == 0.1
        assert restored.total_reinforcements == 5


class TestTendencyBias:
    """Tests for TendencyBias dataclass."""

    def test_no_bias(self):
        bias = TendencyBias(has_bias=False)
        assert not bias.has_bias
        assert bias.strongest_bias == 0.0

    def test_with_bias(self):
        bias = TendencyBias(
            has_bias=True,
            biases={"connection": 0.05, "expression": 0.03},
            strongest_category=CandidateCategory.CONNECTION,
            strongest_bias=0.05,
        )
        assert bias.has_bias
        assert bias.strongest_bias == 0.05


class TestRepeatedTendencyConfig:
    """Tests for RepeatedTendencyConfig."""

    def test_default_config(self):
        config = RepeatedTendencyConfig()
        # Check weak by design
        assert config.max_strength == 0.15  # Weak
        assert config.max_bias == 0.06       # Very weak
        assert config.min_repetitions == 3   # Not single use

    def test_config_serialization(self):
        config = RepeatedTendencyConfig(
            min_repetitions=5,
            max_strength=0.2,
        )
        data = to_dict(config)
        restored = from_dict(data)

        assert restored.min_repetitions == 5
        assert restored.max_strength == 0.2


class TestRepeatedTendencySystem:
    """Tests for RepeatedTendencySystem core functionality."""

    def test_create_system(self):
        system = create_system()
        assert system is not None
        assert len(system.get_tendencies()) == 0

    def test_single_use_does_not_form_tendency(self):
        config = create_config(min_repetitions=3)
        system = RepeatedTendencySystem(config=config)

        # Create a scoped goal that was used
        scope = ScopedGoal(
            category=CandidateCategory.CONNECTION,
            status=ScopeStatus.USED,
            action_taken=True,
        )

        # Single use
        result = system.observe_turn(scoped_goal_used=scope)

        # No tendency formed from single use
        assert result is None
        assert len(system.get_tendencies()) == 0

    def test_repetition_forms_tendency(self):
        config = create_config(min_repetitions=3, recency_window=50)
        system = RepeatedTendencySystem(config=config)

        # Repeat same type of goal multiple times
        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                direction_alignment={"social": 0.5},
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Tendency should have formed
        tendencies = system.get_tendencies()
        assert len(tendencies) == 1
        assert tendencies[0].pattern.category == CandidateCategory.CONNECTION

    def test_reinforcement_increases_strength(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.05,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form tendency
        for _ in range(2):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        initial_strength = system.get_tendencies()[0].strength

        # Reinforce
        scope = ScopedGoal(
            category=CandidateCategory.EXPRESSION,
            status=ScopeStatus.USED,
            action_taken=True,
        )
        system.observe_turn(scoped_goal_used=scope)

        # Strength increased
        assert system.get_tendencies()[0].strength > initial_strength


class TestDecayMechanism:
    """Tests for natural decay when repetition stops."""

    def test_decay_when_not_used(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.05,  # Higher to survive decay
            decay_rate=0.01,  # Lower decay rate
            miss_decay_multiplier=1.0,  # No acceleration
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form tendency with more reinforcements
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.APPROACH,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        initial_strength = system.get_tendencies()[0].strength

        # Turns without using that tendency
        for _ in range(3):
            system.observe_turn(scoped_goal_used=None)

        # Strength decayed but still exists
        assert len(system.get_tendencies()) > 0
        assert system.get_tendencies()[0].strength < initial_strength

    def test_tendency_fades_completely(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.02,
            decay_rate=0.05,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form weak tendency
        for _ in range(2):
            scope = ScopedGoal(
                category=CandidateCategory.ISOLATION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        assert len(system.get_tendencies()) == 1

        # Many turns without use
        for _ in range(20):
            system.observe_turn(scoped_goal_used=None)

        # Tendency should have faded away
        assert len(system.get_tendencies()) == 0
        assert system.state.total_tendencies_faded >= 1

    def test_consecutive_misses_accelerate_decay(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.10,  # Higher to survive multiple decays
            decay_rate=0.02,
            miss_decay_multiplier=2.0,  # Second miss decays faster
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form strong tendency
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        initial_strength = system.get_tendencies()[0].strength

        # First miss - decay_rate is 0.02
        system.observe_turn(scoped_goal_used=None)
        strength_after_1 = system.get_tendencies()[0].strength
        first_decay = initial_strength - strength_after_1

        # Second miss - should decay by decay_rate * miss_decay_multiplier = 0.04
        system.observe_turn(scoped_goal_used=None)
        strength_after_2 = system.get_tendencies()[0].strength
        second_decay = strength_after_1 - strength_after_2

        # The second decay should be larger due to miss_decay_multiplier
        assert second_decay > first_decay * 1.5  # Multiplier is 2.0


class TestWeakInfluence:
    """Tests ensuring influence is WEAK - creates slope, not wall."""

    def test_bias_is_capped(self):
        config = create_config(max_bias=0.06)
        system = RepeatedTendencySystem(config=config)

        # Form strong tendency
        for _ in range(10):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        bias = system.get_bias()

        # Even with strong tendency, bias is capped
        assert bias.strongest_bias <= 0.06

    def test_apply_bias_small_effect(self):
        bias = TendencyBias(
            has_bias=True,
            biases={"connection": 0.06},
            strongest_category=CandidateCategory.CONNECTION,
            strongest_bias=0.06,
        )

        candidate = {"policy": "engage", "score": 0.5}

        result = apply_tendency_bias_to_candidate(candidate, bias)

        # Effect should be very small
        score_change = abs(result["score"] - 0.5)
        assert score_change <= 0.06

    def test_all_candidates_remain_viable(self):
        bias = TendencyBias(
            has_bias=True,
            biases={"approach": 0.06},
            strongest_category=CandidateCategory.APPROACH,
            strongest_bias=0.06,
        )

        candidates = [
            {"policy": "engage", "score": 0.5},
            {"policy": "withdraw", "score": 0.5},
            {"policy": "silence", "score": 0.5},
        ]

        results = apply_tendency_bias_to_candidates(candidates, bias)

        # All candidates still viable
        assert len(results) == 3
        for r in results:
            assert r["score"] > 0.3  # Still viable

    def test_no_bias_no_effect(self):
        bias = TendencyBias(has_bias=False)

        candidate = {"policy": "speak", "score": 0.5}
        result = apply_tendency_bias_to_candidate(candidate, bias)

        assert result["score"] == 0.5


class TestMediumTermDistinction:
    """Tests ensuring this is MEDIUM-TERM, distinct from long and short term."""

    def test_faster_formation_than_value_orientation(self):
        config = create_config(min_repetitions=3)
        system = RepeatedTendencySystem(config=config)

        # Only 3 repetitions to form
        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.EXPLORATION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Tendency formed quickly (unlike ValueOrientation which needs many updates)
        assert len(system.get_tendencies()) == 1

    def test_slower_decay_than_scoped_goal(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.05,  # Higher strength
            decay_rate=0.005,  # Very slow decay
            miss_decay_multiplier=1.0,  # No acceleration
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form tendency with more reinforcements
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        initial_strength = system.get_tendencies()[0].strength

        # A few turns without use
        for _ in range(3):
            system.observe_turn(scoped_goal_used=None)

        # Still has significant strength (unlike ScopedGoal which is gone after 1 turn)
        assert len(system.get_tendencies()) > 0
        remaining_strength = system.get_tendencies()[0].strength
        assert remaining_strength > initial_strength * 0.5

    def test_weaker_than_transient_goal(self):
        config = create_config(max_bias=0.06)
        system = RepeatedTendencySystem(config=config)

        # Form strong tendency
        for _ in range(10):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        bias = system.get_bias()

        # max_bias is 0.06, while TransientGoal has max_bias of 0.12
        assert bias.strongest_bias <= 0.06


class TestNoHardcodedRules:
    """Tests ensuring no personality/belief/rule is hardcoded."""

    def test_tendency_does_not_block_choices(self):
        bias = TendencyBias(
            has_bias=True,
            biases={"approach": 0.06},
            strongest_category=CandidateCategory.APPROACH,
            strongest_bias=0.06,
        )

        # Opposite choice is still valid
        opposite_candidate = {"policy": "withdraw", "score": 0.5}
        result = apply_tendency_bias_to_candidate(opposite_candidate, bias)

        # Still has positive score
        assert result["score"] > 0

    def test_different_categories_can_form_tendencies(self):
        config = create_config(
            min_repetitions=2,
            strength_increment=0.05,  # Higher to survive decay
            decay_rate=0.005,  # Very slow decay
            miss_decay_multiplier=1.0,  # No acceleration
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Form tendency for CONNECTION with extra reinforcements
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Form tendency for ISOLATION (opposite!)
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.ISOLATION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Both tendencies exist (no personality constraint)
        tendencies = system.get_tendencies()
        categories = {t.pattern.category for t in tendencies}

        assert CandidateCategory.CONNECTION in categories
        assert CandidateCategory.ISOLATION in categories

    def test_no_success_failure_evaluation(self):
        """Verify there's no success/failure concept."""
        import psyche.repeated_tendency as module

        assert not hasattr(module, "evaluate_tendency")
        assert not hasattr(module, "mark_success")
        assert not hasattr(module, "mark_failure")


class TestPersistence:
    """Tests for optional persistence."""

    def test_save_and_load(self):
        system = create_system()

        # Form tendencies
        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            system.save_to_file(path)

            new_system = create_system()
            count = new_system.load_from_file(path)

            assert count >= 1
            assert len(new_system.get_tendencies()) >= 1
        finally:
            os.unlink(path)


class TestSummaryAndTracing:
    """Tests for summary and tracing functions."""

    def test_get_tendency_summary(self):
        config = create_config(min_repetitions=2, recency_window=50)
        system = RepeatedTendencySystem(config=config)

        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.APPROACH,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        summary = get_tendency_summary(system)

        assert summary["tendency_count"] >= 1
        assert summary["total_formed"] >= 1
        assert "category_strengths" in summary

    def test_context_for_trace(self):
        config = create_config(min_repetitions=2, recency_window=50)
        system = RepeatedTendencySystem(config=config)

        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        context = create_tendency_context_for_trace(system)

        assert context is not None
        assert "observation_note" in context
        assert "habit" in context["observation_note"].lower()

    def test_empty_context_when_no_tendencies(self):
        system = create_system()
        context = create_tendency_context_for_trace(system)
        assert context is None

    def test_stats_for_dynamics(self):
        config = create_config(min_repetitions=2, recency_window=50)
        system = RepeatedTendencySystem(config=config)

        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        stats = create_tendency_stats_for_dynamics(system)

        assert stats["tendency_count"] >= 1
        assert stats["total_strength"] > 0


class TestSimilarityMatching:
    """Tests for pattern similarity matching."""

    def test_same_category_different_direction_forms_separate(self):
        config = create_config(
            min_repetitions=2,
            similarity_threshold=0.9,  # High threshold
            strength_increment=0.05,  # Higher to survive decay
            decay_rate=0.005,  # Very slow decay
            miss_decay_multiplier=1.0,  # No acceleration
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Pattern 1: CONNECTION with direction A (extra reinforcements)
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                direction_alignment={"social": 0.8, "emotional": 0.3},
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Pattern 2: CONNECTION with very different direction
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                direction_alignment={"practical": 0.9, "logical": 0.4},
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Should have 2 separate tendencies
        tendencies = system.get_tendencies()
        assert len(tendencies) >= 2

    def test_similar_patterns_merge(self):
        config = create_config(
            min_repetitions=2,
            similarity_threshold=0.5,  # Low threshold
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Very similar patterns
        for i in range(4):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                direction_alignment={"creative": 0.8 + i * 0.01},  # Slight variation
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Should merge into one tendency
        tendencies = system.get_tendencies()
        assert len(tendencies) == 1
        # Should have been reinforced multiple times
        assert tendencies[0].total_reinforcements >= 2


class TestIntegration:
    """Full integration tests."""

    def test_complete_habit_formation_and_decay(self):
        config = create_config(
            min_repetitions=3,
            strength_increment=0.03,
            decay_rate=0.02,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Phase 1: Form habit through repetition
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.APPROACH,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Habit formed
        assert len(system.get_tendencies()) == 1
        peak_strength = system.get_tendencies()[0].strength

        # Phase 2: Stop the habit
        for _ in range(10):
            system.observe_turn(scoped_goal_used=None)

        # Habit weakened but may still exist
        if system.get_tendencies():
            assert system.get_tendencies()[0].strength < peak_strength

        # Phase 3: Fully fade (more turns)
        for _ in range(50):
            system.observe_turn(scoped_goal_used=None)

        # Habit should be gone
        assert len(system.get_tendencies()) == 0

    def test_bias_creates_slope_not_wall(self):
        config = create_config(min_repetitions=2, max_bias=0.06, recency_window=50)
        system = RepeatedTendencySystem(config=config)

        # Form strong tendency toward APPROACH
        for _ in range(10):
            scope = ScopedGoal(
                category=CandidateCategory.APPROACH,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        bias = system.get_bias()

        # Test with candidates
        candidates = [
            {"policy": "engage", "score": 0.5},    # Aligned
            {"policy": "withdraw", "score": 0.5},  # Opposite
        ]

        results = apply_tendency_bias_to_candidates(candidates, bias)

        # Aligned might be slightly higher
        engage_score = results[0]["score"]
        withdraw_score = results[1]["score"]

        # But the difference is small (slope, not wall)
        score_diff = abs(engage_score - withdraw_score)
        assert score_diff < 0.1

        # Both are still viable choices
        assert withdraw_score > 0.3
