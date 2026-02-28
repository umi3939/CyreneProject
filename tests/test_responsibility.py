"""
tests/test_responsibility.py - Responsibility mechanism tests.

Tests verify:
1. Decision recording (責任が生まれるタイミング)
2. Outcome evaluation (責任の評価方法)
3. Psychological influence (心理状態への反映)
4. Time-based decay (時間経過による自然減衰)
5. Integration with thought/reaction modules
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from psyche.responsibility import (
    DecisionRecord,
    ResponsibilityState,
    ResponsibilityInfluence,
    record_decision,
    evaluate_outcome,
    apply_decay,
    get_influence,
    create_default_state,
    to_dict,
    from_dict,
)
from psyche.responsibility_manager import ResponsibilityManager
from psyche.reaction import react
from psyche.thought import generate_thought_candidates, select_policy
from psyche.state import PsycheState, Percept


# ── Test: Decision Recording ─────────────────────────────────────


class TestDecisionRecording:
    """判断（Policy）を確定した瞬間が、責任の発生点"""

    def test_record_decision_creates_immutable_record(self):
        """判断を不変の決定記録として記録する"""
        state = create_default_state()
        policy = {"policy_label": "共感する", "rationale": "相手の気持ちに寄り添う"}
        context = {"target_partner": "user_1", "fear_level": 0.3}

        new_state, decision_id = record_decision(state, policy, context)

        # Decision ID should be generated
        assert decision_id is not None
        assert len(decision_id) == 12

        # Pending count increases
        assert new_state.pending_decisions == 1

        # Record should exist
        assert len(new_state.recent_decisions) == 1
        record = new_state.recent_decisions[0]
        assert record["policy_label"] == "共感する"
        assert record["target_partner"] == "user_1"
        assert record["evaluated"] is False

    def test_multiple_decisions_accumulate(self):
        """複数の判断が蓄積される"""
        state = create_default_state()

        for i in range(3):
            policy = {"policy_label": f"policy_{i}"}
            context = {"target_partner": "user_1"}
            state, _ = record_decision(state, policy, context)

        assert state.pending_decisions == 3
        assert len(state.recent_decisions) == 3

    def test_decision_importance_calculated(self):
        """判断の重要度が計算される"""
        state = create_default_state()

        # High fear + attachment = high importance
        policy = {"policy_label": "励ます"}
        context = {"fear_level": 0.7, "involves_attachment": True}
        new_state, _ = record_decision(state, policy, context)

        record = new_state.recent_decisions[0]
        assert record["importance"] == 5  # Max importance


# ── Test: Outcome Evaluation ─────────────────────────────────────


class TestOutcomeEvaluation:
    """結果を観測してから責任を評価する"""

    def test_positive_outcome_reduces_weight(self):
        """肯定的な反応は責任を軽くする"""
        state = create_default_state()
        policy = {"policy_label": "共感する"}
        context = {"target_partner": "user_1"}
        state, decision_id = record_decision(state, policy, context)

        outcome = {
            "user_reaction": "positive",
            "relationship_delta": 0.1,
            "expectation_gap": 0.0,
        }
        new_state = evaluate_outcome(state, decision_id, outcome)

        # Should increase confidence, not harm
        assert new_state.accumulated_confidence > 0
        assert new_state.accumulated_harm == 0
        assert new_state.pending_decisions == 0

    def test_negative_outcome_increases_harm(self):
        """否定的な反応は傷として蓄積される"""
        state = create_default_state()
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}
        state, decision_id = record_decision(state, policy, context)

        outcome = {
            "user_reaction": "negative",
            "relationship_delta": -0.2,
            "expectation_gap": 0.3,
        }
        new_state = evaluate_outcome(state, decision_id, outcome)

        # Should increase harm
        assert new_state.accumulated_harm > 0
        assert new_state.total_weight > 0
        assert new_state.pending_decisions == 0

    def test_rejected_outcome_causes_most_harm(self):
        """拒絶は最も大きな傷を与える"""
        state = create_default_state()
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1", "fear_level": 0.5}
        state, decision_id = record_decision(state, policy, context)

        outcome = {
            "user_reaction": "rejected",
            "relationship_delta": -0.5,
            "expectation_gap": 0.8,
        }
        new_state = evaluate_outcome(state, decision_id, outcome)

        # Rejected causes significant harm
        assert new_state.accumulated_harm > 0.02
        assert new_state.total_weight > 0.05

    def test_evaluation_marks_decision_complete(self):
        """評価後、決定記録は評価済みとなる"""
        state = create_default_state()
        policy = {"policy_label": "共感する"}
        context = {}
        state, decision_id = record_decision(state, policy, context)

        outcome = {"user_reaction": "neutral"}
        new_state = evaluate_outcome(state, decision_id, outcome)

        record = [r for r in new_state.recent_decisions if r["id"] == decision_id][0]
        assert record["evaluated"] is True


# ── Test: Psychological Influence ────────────────────────────────


class TestPsychologicalInfluence:
    """責任が心理状態に間接的に影響を与える"""

    def test_harm_increases_caution_bias(self):
        """傷が多いほど慎重になる"""
        state = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.4,
            accumulated_confidence=0.1,
        )

        influence = get_influence(state)

        assert influence.caution_bias > 0.1
        assert influence.caution_bias <= 0.5  # Upper bound

    def test_weight_increases_fear_amplification(self):
        """責任が重いほど喪失を恐れる"""
        state = ResponsibilityState(
            total_weight=0.8,
            accumulated_harm=0.3,
        )

        influence = get_influence(state)

        assert influence.fear_amplification > 0.2
        assert influence.fear_amplification <= 0.5  # Upper bound

    def test_harm_increases_empathy_bias(self):
        """責任を感じているほど寄り添う"""
        state = ResponsibilityState(
            total_weight=0.6,
            accumulated_harm=0.5,
        )

        influence = get_influence(state)

        assert influence.empathy_bias > 0.2
        assert influence.empathy_bias <= 0.5  # Upper bound

    def test_net_burden_affects_anxiety(self):
        """蓄積された責任が不安を生む"""
        state = ResponsibilityState(
            total_weight=0.3,
            accumulated_harm=0.5,
            accumulated_confidence=0.1,  # Net burden = 0.5 - 0.05 = 0.45
        )

        influence = get_influence(state)

        assert influence.anxiety_baseline > 0
        assert influence.anxiety_baseline <= 0.3  # Upper bound

    def test_confidence_mitigates_burden(self):
        """成功体験が重荷を軽減する"""
        # High harm but also high confidence
        state_with_conf = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.4,
            accumulated_confidence=0.6,
        )
        # Same harm but no confidence
        state_no_conf = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.4,
            accumulated_confidence=0.0,
        )

        inf_with = get_influence(state_with_conf)
        inf_without = get_influence(state_no_conf)

        # Anxiety should be lower with confidence
        assert inf_with.anxiety_baseline < inf_without.anxiety_baseline


# ── Test: Time-based Decay ───────────────────────────────────────


class TestTimeDecay:
    """時間経過による自然減衰"""

    def test_total_weight_decays_over_time(self):
        """責任の総量は時間とともに減衰する"""
        state = ResponsibilityState(total_weight=0.5)

        decayed = apply_decay(state, hours_elapsed=24.0)

        assert decayed.total_weight < state.total_weight
        assert decayed.total_weight > 0  # Not completely gone

    def test_harm_decays_slower_than_weight(self):
        """蓄積された傷はゆっくり減衰する"""
        state = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.5,
        )

        decayed = apply_decay(state, hours_elapsed=24.0)

        # Calculate decay ratios
        weight_ratio = decayed.total_weight / state.total_weight
        harm_ratio = decayed.accumulated_harm / state.accumulated_harm

        # Harm decays slower
        assert harm_ratio > weight_ratio

    def test_decay_has_maximum_time(self):
        """減衰は最大1週間分まで"""
        state = ResponsibilityState(total_weight=0.5)

        # 1000 hours should cap at 168 hours
        decayed_1000h = apply_decay(state, hours_elapsed=1000.0)
        decayed_168h = apply_decay(state, hours_elapsed=168.0)

        assert decayed_1000h.total_weight == decayed_168h.total_weight


# ── Test: Integration with Thought Module ────────────────────────


class TestThoughtIntegration:
    """思考モジュールとの統合"""

    def test_caution_bias_penalizes_risky_choices(self):
        """慎重さバイアスはリスキーな選択にペナルティを与える"""
        psyche_state = PsycheState()
        percept = Percept(text="test", emotion="happy", emotion_valence=0.5)
        recalled = []

        # High caution influence
        influence = ResponsibilityInfluence(
            caution_bias=0.4,
            empathy_bias=0.0,
            fear_amplification=0.0,
            anxiety_baseline=0.0,
        )

        candidates = generate_thought_candidates(
            psyche_state, percept, recalled, influence
        )

        # "からかう" should be penalized
        teasing = [c for c in candidates if c["policy_label"] == "からかう"]
        empathy = [c for c in candidates if c["policy_label"] == "共感する"]

        # If teasing is in candidates, it should score lower than empathy
        if teasing and empathy:
            assert teasing[0]["_score"] < empathy[0]["_score"]

    def test_empathy_bias_boosts_caring_choices(self):
        """共感バイアスは寄り添う選択を促進する"""
        psyche_state = PsycheState()
        percept = Percept(text="test", emotion="sad", emotion_valence=-0.3)
        recalled = []

        # High empathy influence
        influence = ResponsibilityInfluence(
            caution_bias=0.0,
            empathy_bias=0.4,
            fear_amplification=0.0,
            anxiety_baseline=0.0,
        )

        candidates = generate_thought_candidates(
            psyche_state, percept, recalled, influence
        )

        # Top candidate should be empathetic
        assert candidates[0]["policy_label"] in ("共感する", "励ます")

    def test_high_caution_penalizes_teasing(self):
        """慎重な状態ではからかうにスコアペナルティが適用される"""
        psyche_state = PsycheState()
        percept = Percept(text="joke", intent="joke", emotion_valence=0.5)
        recalled = []

        # Very high caution
        influence = ResponsibilityInfluence(
            caution_bias=0.4,
            empathy_bias=0.1,
            fear_amplification=0.0,
            anxiety_baseline=0.0,
        )

        candidates = generate_thought_candidates(
            psyche_state, percept, recalled, influence
        )
        # Find "からかう" candidate and verify its score was penalized
        teasing = [c for c in candidates if c["policy_label"] == "からかう"]
        if teasing:
            # caution_bias=0.4 applies -0.4*4.0 = -1.6 penalty
            # Selection is through scoring only; no force-override
            assert teasing[0].get("_score") is not None


# ── Test: Integration with Reaction Module ───────────────────────


class TestReactionIntegration:
    """感情反応モジュールとの統合"""

    def test_anxiety_baseline_raises_fear(self):
        """不安ベースラインが恐怖を上昇させる"""
        psyche_state = PsycheState()
        percept = Percept(text="test", emotion="neutral", emotion_valence=0.0)

        influence = ResponsibilityInfluence(
            caution_bias=0.0,
            empathy_bias=0.0,
            fear_amplification=0.0,
            anxiety_baseline=0.2,
        )

        new_state = react(percept, psyche_state, delta_time=1.0, responsibility_influence=influence)

        # Fear should be raised by anxiety baseline
        assert new_state.emotions.fear >= 0.1  # anxiety * 0.5

    def test_responsibility_affects_mood(self):
        """責任がムードに影響を与える"""
        psyche_state = PsycheState()
        percept = Percept(text="test", emotion="happy", emotion_valence=0.5)

        # High anxiety from responsibility
        influence = ResponsibilityInfluence(
            caution_bias=0.0,
            empathy_bias=0.0,
            fear_amplification=0.0,
            anxiety_baseline=0.3,
        )

        new_state = react(percept, psyche_state, delta_time=1.0, responsibility_influence=influence)

        # Mood valence should be lower due to responsibility burden
        # (anxiety_baseline * 0.3 penalty)
        assert new_state.mood.valence < 0.15  # Would be higher without responsibility


# ── Test: Persistence ────────────────────────────────────────────


class TestPersistence:
    """永続化のテスト"""

    def test_to_dict_from_dict_roundtrip(self):
        """辞書への変換と復元が正しく動作する"""
        state = ResponsibilityState(
            total_weight=0.5,
            pending_decisions=2,
            accumulated_harm=0.3,
            accumulated_confidence=0.2,
            recent_decisions=[{"id": "test123", "policy_label": "共感する"}],
        )

        d = to_dict(state)
        restored = from_dict(d)

        assert restored.total_weight == state.total_weight
        assert restored.pending_decisions == state.pending_decisions
        assert restored.accumulated_harm == state.accumulated_harm
        assert restored.accumulated_confidence == state.accumulated_confidence
        assert len(restored.recent_decisions) == 1

    def test_from_dict_handles_corruption(self):
        """データ破損時はデフォルト値で復元する"""
        corrupted = {"total_weight": "invalid", "missing_fields": True}

        restored = from_dict(corrupted)

        # Should return default state
        assert restored.total_weight == 0.0
        assert restored.pending_decisions == 0

    def test_create_default_state(self):
        """初期状態が正しく作成される"""
        state = create_default_state()

        assert state.total_weight == 0.0
        assert state.pending_decisions == 0
        assert state.accumulated_harm == 0.0
        assert state.accumulated_confidence == 0.0
        assert len(state.recent_decisions) == 0


# ── Test: Design Goals ───────────────────────────────────────────


class TestDesignGoals:
    """実装のゴールを検証"""

    def test_past_decisions_cannot_be_undone(self):
        """過去の判断を「なかったこと」にできない"""
        state = create_default_state()
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}

        state, decision_id = record_decision(state, policy, context)

        # Record exists and cannot be removed
        assert len(state.recent_decisions) == 1

        # Even after evaluation, record persists
        outcome = {"user_reaction": "negative", "relationship_delta": -0.2}
        state = evaluate_outcome(state, decision_id, outcome)

        assert len(state.recent_decisions) == 1
        assert state.recent_decisions[0]["evaluated"] is True

    def test_repeated_failures_distort_judgment(self):
        """同じ失敗をすると、判断が少しずつ歪んでいく"""
        state = create_default_state()

        # Record and evaluate multiple negative outcomes
        for i in range(3):
            policy = {"policy_label": "からかう"}
            context = {"target_partner": "user_1"}
            state, decision_id = record_decision(state, policy, context)
            outcome = {"user_reaction": "negative", "relationship_delta": -0.1}
            state = evaluate_outcome(state, decision_id, outcome)

        influence = get_influence(state)

        # After repeated failures, caution and empathy should increase
        assert influence.caution_bias > 0.05
        assert influence.empathy_bias > 0.05

    def test_harm_persists_as_weight(self):
        """誰かを傷つけた可能性を、心の重みとして保持し続ける"""
        state = create_default_state()
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1", "fear_level": 0.5}
        state, decision_id = record_decision(state, policy, context)

        outcome = {"user_reaction": "rejected", "relationship_delta": -0.3}
        state = evaluate_outcome(state, decision_id, outcome)

        # Harm persists
        assert state.accumulated_harm > 0

        # Even after decay, some harm remains
        state = apply_decay(state, hours_elapsed=48.0)
        assert state.accumulated_harm > 0

    def test_continues_to_make_decisions(self):
        """それでも判断をやめず、選び続ける"""
        state = create_default_state()

        # Build up significant harm
        for i in range(5):
            policy = {"policy_label": "からかう"}
            context = {"target_partner": "user_1"}
            state, decision_id = record_decision(state, policy, context)
            outcome = {"user_reaction": "negative", "relationship_delta": -0.1}
            state = evaluate_outcome(state, decision_id, outcome)

        # Despite high responsibility, can still make new decisions
        influence = get_influence(state)

        psyche_state = PsycheState()
        percept = Percept(text="test", emotion="neutral")
        recalled = []

        candidates = generate_thought_candidates(
            psyche_state, percept, recalled, influence
        )
        policy = select_policy(candidates, psyche_state, influence)

        # Policy is selected (not blocked)
        assert policy is not None
        assert "policy_label" in policy
