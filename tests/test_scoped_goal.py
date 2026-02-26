"""
Tests for scoped_goal.py - Scoped Goal Commitment (行動スコープ限定の目的コミット)

These tests verify:
1. EPHEMERAL nature - ScopedGoal is never persisted
2. Lives for ONE turn only, then destroyed
3. NO success/failure judgment
4. Lightweight responsibility generation
5. Flow: TransientGoal → ScopedGoal → Decision → Responsibility
"""

import pytest

from psyche.goal_candidates import CandidateCategory, GoalCandidate
from psyche.transient_goal import (
    ActiveGoal,
    TransientGoalManager,
    create_manager as create_transient_manager,
)
from psyche.scoped_goal import (
    ScopeType,
    ScopeStatus,
    ScopedGoal,
    ScopedBias,
    ScopedResponsibility,
    ScopedGoalConfig,
    ScopedGoalSystem,
    create_system,
    create_config,
    apply_scoped_bias_to_candidate,
    apply_scoped_bias_to_candidates,
    get_scoped_goal_summary,
    create_scope_context_for_trace,
    get_responsibilities_for_integration,
    execute_scoped_decision_flow,
    to_dict,
    from_dict,
)


class TestScopedGoal:
    """Tests for ScopedGoal dataclass."""

    def test_default_scoped_goal(self):
        goal = ScopedGoal()
        assert goal.scope_id is not None
        assert len(goal.scope_id) == 8
        assert goal.status == ScopeStatus.ACTIVE
        assert goal.action_taken is False

    def test_scoped_goal_ephemeral_marker(self):
        goal = ScopedGoal()
        # Should have ephemeral marker
        assert hasattr(goal, "_ephemeral")
        assert goal._ephemeral is True

    def test_scoped_goal_with_values(self):
        goal = ScopedGoal(
            source_goal_id="g1",
            source_candidate_id="c1",
            category=CandidateCategory.CONNECTION,
            strength=0.08,
            scope_type=ScopeType.SINGLE_TURN,
        )
        assert goal.source_goal_id == "g1"
        assert goal.category == CandidateCategory.CONNECTION
        assert goal.strength == 0.08


class TestScopedBias:
    """Tests for ScopedBias dataclass."""

    def test_inactive_bias(self):
        bias = ScopedBias(is_active=False)
        assert not bias.is_active
        assert bias.bias_strength == 0.0

    def test_active_bias(self):
        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.EXPRESSION,
            bias_strength=0.08,
        )
        assert bias.is_active
        assert bias.bias_strength == 0.08

    def test_bias_to_dict_for_logging(self):
        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            bias_strength=0.05,
        )
        data = bias.to_dict()
        assert data["is_active"] is True
        assert data["scope_id"] == "s1"


class TestScopedResponsibility:
    """Tests for ScopedResponsibility dataclass."""

    def test_default_responsibility_is_lightweight(self):
        resp = ScopedResponsibility()
        # Very light weight
        assert resp.weight == 0.05
        # Very far distance
        assert resp.distance == 0.9
        # No success/failure
        assert resp.success is None
        assert resp.failure is None

    def test_responsibility_to_dict(self):
        resp = ScopedResponsibility(
            source_scope_id="s1",
            source_goal_id="g1",
        )
        data = resp.to_dict()

        assert data["source_scope_id"] == "s1"
        assert data["is_evaluated"] is False  # NEVER evaluated
        assert data["is_failure"] is False    # NEVER a failure
        assert data["sublimation_eligible"] is True


class TestScopedGoalConfig:
    """Tests for ScopedGoalConfig."""

    def test_default_config(self):
        config = ScopedGoalConfig()
        assert config.base_strength == 0.08
        assert config.max_strength == 0.10  # Very weak
        assert config.responsibility_weight == 0.05  # Very light
        assert config.responsibility_distance == 0.9  # Very far

    def test_config_serialization(self):
        config = ScopedGoalConfig(base_strength=0.06, max_strength=0.08)
        data = to_dict(config)
        restored = from_dict(data)

        assert restored.base_strength == 0.06
        assert restored.max_strength == 0.08


class TestScopedGoalSystem:
    """Tests for ScopedGoalSystem core functionality."""

    def test_create_system(self):
        system = create_system()
        assert system is not None
        assert not system.has_active_scope
        assert system.current_scope is None

    def test_begin_turn_without_transient_goal(self):
        system = create_system()

        scope = system.begin_turn(active_goal=None)

        assert scope is None
        assert not system.has_active_scope

    def test_begin_turn_with_transient_goal(self):
        config = create_config(commit_probability=1.0)  # Always commit
        system = ScopedGoalSystem(config=config)

        active_goal = ActiveGoal(
            goal_id="g1",
            candidate_id="c1",
            candidate_category=CandidateCategory.APPROACH,
            selection_strength=0.7,
            direction_alignment={"a": 0.5},
        )

        scope = system.begin_turn(active_goal=active_goal)

        assert scope is not None
        assert system.has_active_scope
        assert scope.source_goal_id == "g1"
        assert scope.category == CandidateCategory.APPROACH

    def test_only_one_scope_at_a_time(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal1 = ActiveGoal(goal_id="g1", selection_strength=0.7)
        goal2 = ActiveGoal(goal_id="g2", selection_strength=0.8)

        # First commit succeeds
        scope1 = system.begin_turn(active_goal=goal1)
        assert scope1 is not None

        # Second commit fails (already have active scope)
        scope2 = system.begin_turn(active_goal=goal2)
        assert scope2 is None

        # Still have first scope
        assert system.current_scope.source_goal_id == "g1"


class TestEphemeralNature:
    """Tests ensuring ScopedGoal is EPHEMERAL and never persisted."""

    def test_scope_expires_after_end_turn(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1", selection_strength=0.7)
        system.begin_turn(active_goal=goal)

        assert system.has_active_scope

        # End turn
        system.end_turn()

        # Scope is now expired
        assert system.current_scope.status == ScopeStatus.EXPIRED

    def test_scope_cleared_on_next_turn(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1", selection_strength=0.7)
        system.begin_turn(active_goal=goal)
        system.end_turn()

        # Next turn clears the expired scope
        system.begin_turn(active_goal=None)

        assert system.current_scope is None
        assert not system.has_active_scope

    def test_no_persistence_methods(self):
        """Verify there's no save/load for ScopedGoal."""
        system = create_system()

        # System should NOT have save_to_file or load_from_file
        assert not hasattr(system, "save_to_file")
        assert not hasattr(system, "load_from_file")

    def test_scope_not_carried_over(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        # Turn 1: Create scope
        goal = ActiveGoal(goal_id="g1", selection_strength=0.7)
        system.begin_turn(active_goal=goal)
        assert system.has_active_scope

        # End turn 1
        system.end_turn()

        # Turn 2: No transient goal, scope should be gone
        system.begin_turn(active_goal=None)
        assert not system.has_active_scope


class TestDecisionBias:
    """Tests for decision bias application."""

    def test_bias_is_weak(self):
        config = create_config(
            base_strength=0.10,
            max_strength=0.10,
            commit_probability=1.0,
        )
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1", selection_strength=1.0)
        system.begin_turn(active_goal=goal)

        bias = system.get_bias()

        # Even at maximum, bias should be very weak
        assert bias.bias_strength <= 0.10

    def test_apply_bias_subtle_effect(self):
        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.EXPRESSION,
            bias_strength=0.08,
        )

        candidate = {"policy": "speak", "score": 0.5}

        result = apply_scoped_bias_to_candidate(candidate, bias)

        # Score change should be very small
        score_change = abs(result["score"] - 0.5)
        assert score_change <= 0.08  # At most 8% change

    def test_inactive_bias_no_effect(self):
        bias = ScopedBias(is_active=False)

        candidate = {"policy": "speak", "score": 0.5}
        result = apply_scoped_bias_to_candidate(candidate, bias)

        assert result["score"] == 0.5

    def test_bias_applies_to_all_candidates(self):
        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.APPROACH,
            bias_strength=0.08,
        )

        candidates = [
            {"policy": "speak", "score": 0.5},
            {"policy": "silence", "score": 0.5},
            {"policy": "joke", "score": 0.5},
        ]

        results = apply_scoped_bias_to_candidates(candidates, bias)

        # All candidates present
        assert len(results) == 3
        # All have positive scores
        for r in results:
            assert r["score"] > 0


class TestResponsibilityGeneration:
    """Tests for lightweight responsibility generation."""

    def test_record_action_creates_responsibility(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(
            goal_id="g1",
            candidate_category=CandidateCategory.CONNECTION,
        )
        system.begin_turn(active_goal=goal)

        # Record an action
        resp = system.record_action(decision_info={"policy": "speak"})

        assert resp is not None
        assert resp.source_goal_id == "g1"
        assert resp.category == CandidateCategory.CONNECTION

    def test_responsibility_is_lightweight(self):
        config = create_config(
            commit_probability=1.0,
            responsibility_weight=0.05,
            responsibility_distance=0.9,
        )
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)
        resp = system.record_action()

        # Very light
        assert resp.weight == 0.05
        # Very far
        assert resp.distance == 0.9

    def test_no_success_failure_judgment(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)
        resp = system.record_action()

        # NEVER evaluated for success/failure
        assert resp.success is None
        assert resp.failure is None

    def test_responsibilities_for_integration(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)
        system.record_action()

        resps = get_responsibilities_for_integration(system)

        assert len(resps) == 1
        r = resps[0]
        assert r["is_evaluated"] is False
        assert r["is_failure"] is False
        assert r["is_success"] is False
        assert r["sublimation_eligible"] is True

    def test_no_action_no_responsibility(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)

        # Don't record action
        system.end_turn()

        resps = system.get_pending_responsibilities()
        assert len(resps) == 0


class TestScopeLifecycle:
    """Tests for complete scope lifecycle."""

    def test_full_lifecycle_flow(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        # 1. Begin turn with transient goal
        goal = ActiveGoal(
            goal_id="g1",
            candidate_id="c1",
            candidate_category=CandidateCategory.EXPRESSION,
        )
        scope = system.begin_turn(active_goal=goal)

        assert scope is not None
        assert scope.status == ScopeStatus.ACTIVE

        # 2. Get bias
        bias = system.get_bias()
        assert bias.is_active

        # 3. Record action
        resp = system.record_action()
        assert scope.status == ScopeStatus.USED
        assert scope.action_taken is True

        # 4. End turn
        system.end_turn()
        assert scope.status == ScopeStatus.EXPIRED

        # 5. Next turn clears
        system.begin_turn(active_goal=None)
        assert system.current_scope is None

    def test_action_marks_scope_as_used(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)

        assert system.current_scope.status == ScopeStatus.ACTIVE

        system.record_action()

        assert system.current_scope.status == ScopeStatus.USED


class TestConnectionFlow:
    """Tests for TransientGoal → ScopedGoal → Decision → Responsibility flow."""

    def test_complete_connection_flow(self):
        # Create transient goal manager
        transient_manager = create_transient_manager()

        # Select a transient goal
        candidate = GoalCandidate(
            candidate_id="c1",
            category=CandidateCategory.CONNECTION,
            intensity=0.8,
            direction_expression={"social": 0.7},
        )
        transient_manager.select_goal(candidate)

        # Create scoped goal system
        scoped_config = create_config(commit_probability=1.0)
        scoped_system = ScopedGoalSystem(config=scoped_config)

        # Begin turn with transient manager
        scope = scoped_system.begin_turn(transient_manager=transient_manager)

        assert scope is not None
        assert scope.source_candidate_id == "c1"
        assert scope.category == CandidateCategory.CONNECTION

        # Get bias for decision
        bias = scoped_system.get_bias()
        assert bias.is_active

        # Apply to decision candidates
        candidates = [
            {"policy": "engage", "score": 0.5},
            {"policy": "withdraw", "score": 0.5},
        ]
        biased = apply_scoped_bias_to_candidates(candidates, bias)

        # All candidates still valid
        assert len(biased) == 2

        # Record action (simulating a decision was made)
        resp = scoped_system.record_action()

        # Responsibility created
        assert resp is not None
        assert resp.category == CandidateCategory.CONNECTION

        # End turn
        scoped_system.end_turn()

        # Verify responsibility is lightweight
        resps = get_responsibilities_for_integration(scoped_system)
        assert len(resps) == 1
        assert resps[0]["weight"] <= 0.05
        assert resps[0]["distance"] >= 0.9
        assert resps[0]["is_failure"] is False

    def test_execute_scoped_decision_flow_helper(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(
            goal_id="g1",
            candidate_category=CandidateCategory.EXPRESSION,
        )
        system.begin_turn(active_goal=goal)

        candidates = [
            {"policy": "speak", "score": 0.6},
            {"policy": "silence", "score": 0.4},
        ]

        # Use helper function with simple selection
        def select_highest(cands):
            return max(cands, key=lambda x: x["score"])

        biased, selected, resp = execute_scoped_decision_flow(
            system, candidates, select_fn=select_highest
        )

        assert len(biased) == 2
        assert selected is not None
        assert resp is not None
        assert system.current_scope.action_taken is True


class TestIntrospectionIntegration:
    """Tests for introspection trace integration."""

    def test_context_when_active_scope(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(
            goal_id="g1",
            candidate_category=CandidateCategory.EXPLORATION,
        )
        system.begin_turn(active_goal=goal)

        context = create_scope_context_for_trace(system)

        assert context is not None
        assert "scoped_goal" in context
        assert context["scoped_goal"]["source_goal_id"] == "g1"
        assert "observation_note" in context
        assert "EPHEMERAL" in context["observation_note"]

    def test_context_when_no_scope(self):
        system = create_system()

        context = create_scope_context_for_trace(system)

        assert context is None


class TestSummary:
    """Tests for summary functions."""

    def test_summary_with_active_scope(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(
            goal_id="g1",
            candidate_category=CandidateCategory.APPROACH,
        )
        system.begin_turn(active_goal=goal)

        summary = get_scoped_goal_summary(system)

        assert summary["has_active_scope"] is True
        assert summary["current_scope"]["source_goal_id"] == "g1"
        assert summary["current_scope"]["category"] == "approach"

    def test_summary_without_scope(self):
        system = create_system()

        summary = get_scoped_goal_summary(system)

        assert summary["has_active_scope"] is False
        assert summary["current_scope"] is None


class TestProbabilisticCommit:
    """Tests for probabilistic commitment."""

    def test_zero_probability_never_commits(self):
        config = create_config(commit_probability=0.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1", selection_strength=1.0)

        # Try many times
        for _ in range(20):
            system._clear_expired_scope()
            scope = system._attempt_commit(goal)
            assert scope is None

    def test_full_probability_always_commits(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1", selection_strength=1.0)

        scope = system._attempt_commit(goal)
        assert scope is not None

    def test_strength_has_variance(self):
        config = create_config(
            base_strength=0.08,
            strength_variance=0.02,
            commit_probability=1.0,
        )

        strengths = []
        for _ in range(20):
            system = ScopedGoalSystem(config=config)
            goal = ActiveGoal(goal_id="g1")
            scope = system.begin_turn(active_goal=goal)
            if scope:
                strengths.append(scope.strength)

        # Should have some variance
        if len(strengths) > 1:
            assert min(strengths) != max(strengths)


class TestClearResponsibilities:
    """Tests for clearing responsibilities."""

    def test_clear_returns_and_removes(self):
        config = create_config(commit_probability=1.0)
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)
        system.record_action()

        assert len(system._pending_responsibilities) == 1

        # Clear
        cleared = system.clear_responsibilities()

        assert len(cleared) == 1
        assert len(system._pending_responsibilities) == 0

    def test_total_pending_weight(self):
        config = create_config(
            commit_probability=1.0,
            responsibility_weight=0.05,
        )
        system = ScopedGoalSystem(config=config)

        goal = ActiveGoal(goal_id="g1")
        system.begin_turn(active_goal=goal)
        system.record_action()

        weight = system.get_total_pending_weight()
        assert weight == pytest.approx(0.05, rel=0.01)


class TestJapanesePolicyLabelAlignment:
    """Tests verifying that category_policy_affinity uses Japanese labels
    matching thought.py POLICIES, so alignment calculation actually works."""

    def test_approach_category_matches_japanese_labels(self):
        """APPROACH category should match Japanese policy labels like '共感する'."""
        from psyche.scoped_goal import _calculate_scope_alignment

        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.APPROACH,
            bias_strength=0.08,
        )
        for label in ["共感する", "励ます", "質問で会話を広げる", "提案する"]:
            candidate = {"policy": label, "score": 0.5}
            alignment = _calculate_scope_alignment(candidate, bias)
            assert alignment > 0, f"'{label}' should produce positive alignment for APPROACH"

    def test_expression_category_matches_japanese_labels(self):
        """EXPRESSION category should match Japanese policy labels."""
        from psyche.scoped_goal import _calculate_scope_alignment

        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.EXPRESSION,
            bias_strength=0.08,
        )
        for label in ["感想を述べる", "冗談を言う", "からかう", "自分の経験を話す"]:
            candidate = {"policy": label, "score": 0.5}
            alignment = _calculate_scope_alignment(candidate, bias)
            assert alignment > 0, f"'{label}' should produce positive alignment for EXPRESSION"

    def test_bias_actually_changes_score_with_japanese_label(self):
        """Verify that apply_scoped_bias_to_candidate changes score with Japanese label."""
        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.CONNECTION,
            bias_strength=0.08,
        )
        candidate = {"policy": "共感する", "score": 0.5}
        result = apply_scoped_bias_to_candidate(candidate, bias)
        assert result["score"] != 0.5, "Score should change when Japanese label matches"
        assert result["score"] > 0.5, "Aligned label should increase score"

    def test_non_matching_label_no_alignment(self):
        """A label not in the category's affinity list should get 0 alignment."""
        from psyche.scoped_goal import _calculate_scope_alignment

        bias = ScopedBias(
            is_active=True,
            scope_id="s1",
            category=CandidateCategory.ISOLATION,
            bias_strength=0.08,
        )
        # "反論する" is not in ISOLATION
        candidate = {"policy": "反論する", "score": 0.5}
        alignment = _calculate_scope_alignment(candidate, bias)
        assert alignment == 0.0, "Non-matching label should produce 0 alignment"
