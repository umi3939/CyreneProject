"""
Tests for transient_goal.py - Transient Goal Selection (一時的目的選択)

These tests verify:
1. At most ONE active goal at any time
2. WEAK/SUBTLE bias on decision candidates
3. LIGHT responsibility (low weight, high distance)
4. Release/switch is NOT failure - natural sublimation
5. Integration with introspection trace
6. Persistence across sessions
"""

import json
import os
import tempfile
import time

import pytest

from psyche.goal_candidates import GoalCandidate, CandidateCategory
from psyche.transient_goal import (
    GoalReleaseReason,
    ActiveGoal,
    GoalBias,
    LightResponsibility,
    TransientGoalConfig,
    TransientGoalState,
    TransientGoalManager,
    create_manager,
    create_config,
    apply_goal_bias_to_candidate,
    apply_goal_bias_to_candidates,
    get_transient_goal_summary,
    create_goal_context_for_trace,
    create_goal_stats_for_dynamics,
    get_responsibilities_for_dispersion,
    to_dict,
    from_dict,
)


class TestActiveGoal:
    """Tests for ActiveGoal dataclass."""

    def test_default_active_goal(self):
        goal = ActiveGoal()
        assert goal.goal_id is not None
        assert len(goal.goal_id) == 8
        assert goal.candidate_id == ""
        assert goal.selection_strength == 0.0

    def test_active_goal_with_values(self):
        goal = ActiveGoal(
            candidate_id="c123",
            candidate_category=CandidateCategory.APPROACH,
            selection_strength=0.7,
            direction_alignment={"a": 0.5},
        )
        assert goal.candidate_id == "c123"
        assert goal.candidate_category == CandidateCategory.APPROACH
        assert goal.selection_strength == 0.7

    def test_active_goal_serialization(self):
        goal = ActiveGoal(
            goal_id="g1",
            candidate_id="c1",
            candidate_category=CandidateCategory.CONNECTION,
            selection_strength=0.8,
            initial_strength=0.9,
        )
        data = goal.to_dict()
        restored = ActiveGoal.from_dict(data)

        assert restored.goal_id == "g1"
        assert restored.candidate_id == "c1"
        assert restored.candidate_category == CandidateCategory.CONNECTION
        assert restored.selection_strength == 0.8


class TestGoalBias:
    """Tests for GoalBias dataclass."""

    def test_inactive_bias(self):
        bias = GoalBias(is_active=False)
        assert not bias.is_active
        assert bias.bias_strength == 0.0

    def test_active_bias(self):
        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.EXPRESSION,
            bias_strength=0.1,
        )
        assert bias.is_active
        assert bias.bias_strength == 0.1


class TestLightResponsibility:
    """Tests for LightResponsibility dataclass."""

    def test_default_responsibility(self):
        resp = LightResponsibility()
        # Check it's LIGHT by default
        assert resp.weight == 0.1  # Low
        assert resp.distance == 0.8  # High (far)
        assert resp.time_slice_turns == 10  # Short

    def test_responsibility_serialization(self):
        resp = LightResponsibility(
            source_goal_id="g1",
            weight=0.05,
            distance=0.9,
            release_reason=GoalReleaseReason.NATURAL_DECAY,
        )
        data = resp.to_dict()
        restored = LightResponsibility.from_dict(data)

        assert restored.source_goal_id == "g1"
        assert restored.weight == 0.05
        assert restored.release_reason == GoalReleaseReason.NATURAL_DECAY


class TestTransientGoalConfig:
    """Tests for TransientGoalConfig."""

    def test_default_config(self):
        config = TransientGoalConfig()
        assert config.decay_rate == 0.02
        assert config.bias_multiplier == 0.15
        assert config.max_bias == 0.12  # Keeps influence subtle
        assert config.responsibility_weight == 0.1  # Light
        assert config.responsibility_distance == 0.8  # Far

    def test_config_serialization(self):
        config = TransientGoalConfig(decay_rate=0.05, max_bias=0.1)
        data = to_dict(config)
        restored = from_dict(data)

        assert restored.decay_rate == 0.05
        assert restored.max_bias == 0.1


class TestTransientGoalManager:
    """Tests for TransientGoalManager core functionality."""

    def test_create_manager(self):
        manager = create_manager()
        assert manager is not None
        assert not manager.has_active_goal
        assert manager.active_goal is None

    def test_select_goal(self):
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.APPROACH,
            intensity=0.7,
            direction_expression={"a": 0.5},
        )

        active = manager.select_goal(candidate)

        assert manager.has_active_goal
        assert active.candidate_id == "c1"
        assert active.candidate_category == CandidateCategory.APPROACH
        assert active.selection_strength >= 0.3  # At least min_selection_strength

    def test_only_one_active_goal(self):
        manager = create_manager()

        c1 = GoalCandidate(candidate_id="c1", intensity=0.7)
        c2 = GoalCandidate(candidate_id="c2", intensity=0.8)

        manager.select_goal(c1)
        assert manager.active_goal.candidate_id == "c1"

        # Selecting another replaces, doesn't add
        manager.select_goal(c2)
        assert manager.active_goal.candidate_id == "c2"

        # Still only one active
        assert manager.has_active_goal

    def test_switch_counts(self):
        manager = create_manager()

        c1 = GoalCandidate(candidate_id="c1", intensity=0.7)
        c2 = GoalCandidate(candidate_id="c2", intensity=0.8)

        manager.select_goal(c1)
        assert manager.state.total_switches == 0

        manager.select_goal(c2)
        assert manager.state.total_switches == 1

    def test_release_goal(self):
        manager = create_manager()

        candidate = GoalCandidate(candidate_id="c1", intensity=0.7)
        manager.select_goal(candidate)

        assert manager.has_active_goal

        resp = manager.release_goal(GoalReleaseReason.MANUAL_RELEASE)

        assert not manager.has_active_goal
        assert manager.active_goal is None
        assert resp is not None
        assert resp.release_reason == GoalReleaseReason.MANUAL_RELEASE

    def test_release_creates_light_responsibility(self):
        manager = create_manager()

        candidate = GoalCandidate(candidate_id="c1", intensity=0.7)
        manager.select_goal(candidate)

        # Selection creates responsibility
        assert len(manager.state.pending_responsibilities) == 1

        manager.release_goal()

        # Release creates another responsibility
        assert len(manager.state.pending_responsibilities) == 2

        # Check they're light
        for resp in manager.state.pending_responsibilities:
            assert resp.weight <= 0.1  # Low
            assert resp.distance >= 0.8  # High


class TestDecayBehavior:
    """Tests for goal strength decay."""

    def test_decay_over_turns(self):
        config = create_config(decay_rate=0.1, min_active_strength=0.01)
        manager = TransientGoalManager(config=config)

        candidate = GoalCandidate(candidate_id="c1", intensity=0.8)
        manager.select_goal(candidate, strength=1.0)

        strengths = []
        for _ in range(5):
            manager.observe_turn()
            if manager.active_goal:
                strengths.append(manager.active_goal.selection_strength)

        # Should decay
        assert strengths[0] == pytest.approx(0.9, rel=0.01)
        assert strengths[1] == pytest.approx(0.81, rel=0.01)

    def test_natural_decay_release(self):
        config = create_config(decay_rate=0.5, min_active_strength=0.2)
        manager = TransientGoalManager(config=config)

        candidate = GoalCandidate(candidate_id="c1", intensity=0.5)
        manager.select_goal(candidate, strength=0.3)

        # After decay: 0.3 * 0.5 = 0.15 < 0.2 threshold
        reason = manager.observe_turn()

        assert reason == GoalReleaseReason.NATURAL_DECAY
        assert not manager.has_active_goal
        assert manager.state.total_natural_decays == 1


class TestSubtleBias:
    """Tests ensuring bias is SUBTLE, not forceful."""

    def test_bias_is_weak(self):
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.EXPRESSION,
            intensity=1.0,  # Maximum intensity
        )
        manager.select_goal(candidate, strength=1.0)  # Maximum strength

        bias = manager.get_bias()

        # Even at maximum, bias should be capped
        assert bias.bias_strength <= 0.12  # max_bias default

    def test_apply_bias_small_effect(self):
        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.EXPRESSION,
            bias_strength=0.1,  # 10% max effect
        )

        candidate = {"policy": "speak", "score": 0.5}

        result = apply_goal_bias_to_candidate(candidate, bias)

        # Score change should be small
        score_change = abs(result["score"] - 0.5)
        assert score_change <= 0.1  # At most 10% change

    def test_bias_does_not_filter(self):
        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.APPROACH,
            bias_strength=0.12,
        )

        candidates = [
            {"policy": "speak", "score": 0.5},
            {"policy": "silence", "score": 0.5},
            {"policy": "joke", "score": 0.5},
        ]

        results = apply_goal_bias_to_candidates(candidates, bias)

        # All candidates still present
        assert len(results) == 3

        # All have positive scores
        for r in results:
            assert r["score"] > 0

    def test_inactive_bias_no_effect(self):
        bias = GoalBias(is_active=False)

        candidate = {"policy": "speak", "score": 0.5}
        result = apply_goal_bias_to_candidate(candidate, bias)

        assert result["score"] == 0.5


class TestLightResponsibility:
    """Tests ensuring responsibility is LIGHT, not burdensome."""

    def test_responsibility_is_light(self):
        config = create_config()
        manager = TransientGoalManager(config=config)

        candidate = GoalCandidate(candidate_id="c1", intensity=0.9)
        manager.select_goal(candidate)

        # Check responsibility is light
        assert len(manager.state.pending_responsibilities) == 1
        resp = manager.state.pending_responsibilities[0]

        assert resp.weight == config.responsibility_weight  # 0.1 default
        assert resp.distance == config.responsibility_distance  # 0.8 default

    def test_responsibility_sublimation_over_time(self):
        config = create_config(responsibility_time_slice=5)
        manager = TransientGoalManager(config=config)

        candidate = GoalCandidate(candidate_id="c1", intensity=0.7)
        manager.select_goal(candidate)

        # Get initial weight
        initial_weight = manager.get_pending_responsibility_weight()

        # Process turns
        for _ in range(10):
            manager.observe_turn()

        # Weight should be reduced (sublimated)
        final_weight = manager.get_pending_responsibility_weight()
        assert final_weight < initial_weight

    def test_release_not_treated_as_failure(self):
        manager = create_manager()

        candidate = GoalCandidate(candidate_id="c1", intensity=0.7)
        manager.select_goal(candidate)
        manager.release_goal(GoalReleaseReason.SWITCHED)

        # Check responsibilities for dispersion
        resps = get_responsibilities_for_dispersion(manager)

        for r in resps:
            assert r["is_failure"] is False  # NEVER a failure
            assert r["sublimation_eligible"] is True


class TestIntrospectionIntegration:
    """Tests for introspection trace integration."""

    def test_context_when_active(self):
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.CONNECTION,
            intensity=0.8,
        )
        manager.select_goal(candidate)

        context = create_goal_context_for_trace(manager)

        assert context is not None
        assert "active_goal" in context
        assert context["active_goal"]["candidate_id"] == "c1"
        assert context["active_goal"]["category"] == "connection"
        assert "observation_note" in context
        assert "subtle bias" in context["observation_note"]

    def test_context_when_inactive(self):
        manager = create_manager()

        context = create_goal_context_for_trace(manager)
        assert context is None

    def test_stats_for_dynamics(self):
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.EXPRESSION,
            intensity=0.7,
        )
        manager.select_goal(candidate, strength=0.8)

        stats = create_goal_stats_for_dynamics(manager)

        assert stats["is_active"] is True
        assert stats["active_category"] == "expression"
        assert stats["active_strength"] == pytest.approx(0.8, rel=0.01)


class TestPersistence:
    """Tests for saving and loading state."""

    def test_save_and_load(self):
        manager = create_manager()

        candidate = GoalCandidate(candidate_id="c1", intensity=0.7)
        manager.select_goal(candidate, strength=0.75)
        manager.state.turn_count = 100

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            manager.save_to_file(path)

            new_manager = create_manager()
            has_goal = new_manager.load_from_file(path)

            assert has_goal is True
            assert new_manager.has_active_goal
            assert new_manager.active_goal.candidate_id == "c1"
            assert new_manager.state.turn_count == 100
        finally:
            os.unlink(path)

    def test_persistence_across_restart(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            # Session 1
            m1 = create_manager()
            candidate = GoalCandidate(candidate_id="c1", intensity=0.8)
            m1.select_goal(candidate)
            m1.state.total_goals_selected = 5
            m1.save_to_file(path)

            # Session 2
            m2 = create_manager()
            m2.load_from_file(path)

            assert m2.has_active_goal
            assert m2.state.total_goals_selected == 5
        finally:
            os.unlink(path)


class TestAutoSelection:
    """Tests for automatic goal selection."""

    def test_auto_select_from_strong_candidates(self):
        config = create_config(
            auto_select_threshold=0.5,
            auto_select_probability=1.0,  # Always auto-select for test
        )
        manager = TransientGoalManager(config=config)

        candidates = [
            GoalCandidate(candidate_id="c1", intensity=0.8),
            GoalCandidate(candidate_id="c2", intensity=0.3),  # Below threshold
        ]

        # No goal yet
        assert not manager.has_active_goal

        manager.observe_turn(available_candidates=candidates)

        # Should have auto-selected c1
        assert manager.has_active_goal
        assert manager.active_goal.candidate_id == "c1"

    def test_no_auto_select_when_goal_active(self):
        config = create_config(auto_select_probability=1.0)
        manager = TransientGoalManager(config=config)

        # Manually select a goal
        c1 = GoalCandidate(candidate_id="c1", intensity=0.5)
        manager.select_goal(c1)

        # Offer stronger candidates
        candidates = [
            GoalCandidate(candidate_id="c2", intensity=0.9),
        ]

        manager.observe_turn(available_candidates=candidates)

        # Should still have c1, not auto-switched
        assert manager.active_goal.candidate_id == "c1"


class TestSummary:
    """Tests for summary functions."""

    def test_get_summary_with_active_goal(self):
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.APPROACH,
            intensity=0.8,
        )
        manager.select_goal(candidate)

        summary = get_transient_goal_summary(manager)

        assert summary["has_active_goal"] is True
        assert summary["active_goal"]["candidate_id"] == "c1"
        assert summary["active_goal"]["category"] == "approach"

    def test_get_summary_without_active_goal(self):
        manager = create_manager()

        summary = get_transient_goal_summary(manager)

        assert summary["has_active_goal"] is False
        assert summary["active_goal"] is None


class TestIntegration:
    """Full integration tests."""

    def test_full_lifecycle(self):
        manager = create_manager()

        # 1. Select a goal
        c1 = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.CONNECTION,
            intensity=0.8,
            direction_expression={"social": 0.7},
        )
        manager.select_goal(c1)

        assert manager.has_active_goal
        assert manager.state.total_goals_selected == 1

        # 2. Get bias for decision
        bias = manager.get_bias()
        assert bias.is_active
        assert bias.bias_strength > 0
        assert bias.bias_strength <= 0.12

        # 3. Apply bias to candidates (subtle effect)
        decision_candidates = [
            {"policy": "speak", "score": 0.5},
            {"policy": "silence", "score": 0.5},
        ]
        biased = apply_goal_bias_to_candidates(decision_candidates, bias)

        # All still present, scores adjusted slightly
        assert len(biased) == 2
        for b in biased:
            assert b["score"] > 0

        # 4. Create introspection context
        context = create_goal_context_for_trace(manager)
        assert context is not None
        assert context["active_goal"]["category"] == "connection"

        # 5. Simulate turns with decay
        for _ in range(20):
            manager.observe_turn()

        # 6. Check responsibility sublimation
        weight = manager.get_pending_responsibility_weight()
        assert weight < 0.1  # Reduced from initial

        # 7. Release is NOT failure
        resps = get_responsibilities_for_dispersion(manager)
        for r in resps:
            assert r["is_failure"] is False

    def test_switch_without_failure(self):
        manager = create_manager()

        c1 = GoalCandidate(candidate_id="c1", intensity=0.7)
        c2 = GoalCandidate(candidate_id="c2", intensity=0.8)

        manager.select_goal(c1)
        manager.select_goal(c2)  # Switch

        # c1's responsibility should be marked as SWITCHED, not failure
        resps = get_responsibilities_for_dispersion(manager)

        switched_resps = [
            r for r in resps
            if r.get("release_reason") == "switched"
        ]
        assert len(switched_resps) >= 1

        for r in switched_resps:
            assert r["is_failure"] is False

    def test_bias_composition_with_decision_system(self):
        """Test that goal bias composes with other biases."""
        manager = create_manager()

        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.EXPRESSION,
            intensity=0.9,
        )
        manager.select_goal(candidate)

        bias = manager.get_bias()

        # Simulate decision candidates with various scores
        candidates = [
            {"policy": "elaborate", "score": 0.6},
            {"policy": "silence", "score": 0.7},
            {"policy": "joke", "score": 0.5},
        ]

        # Apply goal bias
        biased = apply_goal_bias_to_candidates(candidates, bias)

        # All candidates still viable
        for b in biased:
            assert 0.0 < b["score"] < 1.0

        # The original ranking may shift slightly but not dramatically
        # (this tests subtle influence)
        original_winner = max(candidates, key=lambda x: x["score"])
        biased_winner = max(biased, key=lambda x: x["score"])

        # Winner might change, but all candidates remain competitive
        min_biased_score = min(b["score"] for b in biased)
        max_biased_score = max(b["score"] for b in biased)

        # Spread shouldn't be huge (subtle effect)
        assert max_biased_score - min_biased_score < 0.5


class TestJapanesePolicyLabelAlignment:
    """Tests verifying that category_policy_affinity uses Japanese labels
    matching thought.py POLICIES, so alignment calculation actually works."""

    def test_approach_category_matches_japanese_labels(self):
        """APPROACH category should match Japanese policy labels like '共感する'."""
        from psyche.transient_goal import _calculate_alignment

        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.APPROACH,
            bias_strength=0.12,
        )
        # "共感する" is in APPROACH affinity list
        candidate = {"policy": "共感する", "score": 0.5}
        alignment = _calculate_alignment(candidate, bias)
        assert alignment > 0, "Japanese policy label should produce positive alignment"

    def test_expression_category_matches_japanese_labels(self):
        """EXPRESSION category should match Japanese policy labels like '感想を述べる'."""
        from psyche.transient_goal import _calculate_alignment

        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.EXPRESSION,
            bias_strength=0.12,
        )
        for label in ["感想を述べる", "冗談を言う", "からかう", "自分の経験を話す"]:
            candidate = {"policy": label, "score": 0.5}
            alignment = _calculate_alignment(candidate, bias)
            assert alignment > 0, f"'{label}' should produce positive alignment for EXPRESSION"

    def test_avoidance_category_matches_japanese_labels(self):
        """AVOIDANCE category should match '黙って聞く' etc."""
        from psyche.transient_goal import _calculate_alignment

        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.AVOIDANCE,
            bias_strength=0.12,
        )
        for label in ["黙って聞く", "見守る", "話題を変える", "確認する"]:
            candidate = {"policy": label, "score": 0.5}
            alignment = _calculate_alignment(candidate, bias)
            assert alignment > 0, f"'{label}' should produce positive alignment for AVOIDANCE"

    def test_exploration_category_matches_japanese_labels(self):
        """EXPLORATION category should match '質問で会話を広げる' etc."""
        from psyche.transient_goal import _calculate_alignment

        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.EXPLORATION,
            bias_strength=0.12,
        )
        for label in ["質問で会話を広げる", "確認する", "提案する"]:
            candidate = {"policy": label, "score": 0.5}
            alignment = _calculate_alignment(candidate, bias)
            assert alignment > 0, f"'{label}' should produce positive alignment for EXPLORATION"

    def test_bias_actually_changes_score_with_japanese_label(self):
        """Verify that apply_goal_bias_to_candidate changes score with Japanese label."""
        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.EXPRESSION,
            bias_strength=0.12,
        )
        candidate = {"policy": "感想を述べる", "score": 0.5}
        result = apply_goal_bias_to_candidate(candidate, bias)
        assert result["score"] != 0.5, "Score should change when Japanese label matches"
        assert result["score"] > 0.5, "Aligned label should increase score"

    def test_non_matching_label_no_alignment(self):
        """A label not in the category's affinity list should get 0 alignment."""
        from psyche.transient_goal import _calculate_alignment

        bias = GoalBias(
            is_active=True,
            goal_id="g1",
            candidate_category=CandidateCategory.AVOIDANCE,
            bias_strength=0.12,
        )
        # "からかう" is EXPRESSION, not AVOIDANCE
        candidate = {"policy": "からかう", "score": 0.5}
        alignment = _calculate_alignment(candidate, bias)
        assert alignment == 0.0, "Non-matching label should produce 0 alignment"
