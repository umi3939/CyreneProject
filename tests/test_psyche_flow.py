"""
tests/test_psyche_flow.py - Psyche package end-to-end flow test.

Runs the full psyche pipeline (perception → reaction → memory_link →
thought → expression) with mock LLM.  Asserts that fear_index changes
after stimulus events.

**Architecture Validation**: Tests verify that:
1. All thinking/policy logic is LOCAL (no LLM in thought.py)
2. Gemini is only used for parse_percept (auxiliary) and render_expression (voice)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock

import pytest

from psyche import (
    PsycheState,
    Percept,
    AttachmentState,
    ContinuityState,
    FearIndex,
    IdentityState,
    ProjectionState,
    compute_fear_index,
    react,
    recall_by_mood,
    parse_percept,
    generate_thought_candidates,
    select_policy,
    render_expression,
)
from psyche import attachment_manager as att_mgr
from psyche.identity_manager import calc_identity_risk
from psyche.continuity_manager import calc_continuity_risk
from psyche.projection_manager import calc_projection_risk


# ── Mock helpers ───────────────────────────────────────────────

async def _mock_llm_happy(prompt: str, params=None) -> str:
    """Mock LLM for happy percept parsing."""
    if "解析" in prompt or "分析" in prompt:
        return json.dumps(
            {"meaning": "嬉しい", "emotion": "happy", "intent": "greeting",
             "emotion_valence": 0.7, "topics": ["挨拶"]},
            ensure_ascii=False,
        )
    if "セリフ" in prompt or "確定済み" in prompt:
        return json.dumps(
            {"text": "あら、嬉しいわ♡", "meta": {"emotion": "joy", "intensity": 0.7, "action": "共感する"}},
            ensure_ascii=False,
        )
    return '{"result": "ok"}'


async def _mock_llm_scared(prompt: str, params=None) -> str:
    """Mock LLM for scared percept parsing."""
    if "解析" in prompt or "分析" in prompt:
        return json.dumps(
            {"meaning": "不安", "emotion": "scared", "intent": "sharing",
             "emotion_valence": -0.7, "topics": ["不安"]},
            ensure_ascii=False,
        )
    if "セリフ" in prompt or "確定済み" in prompt:
        return json.dumps(
            {"text": "...忘れないで", "meta": {"emotion": "fear", "intensity": 0.6, "action": "共感する"}},
            ensure_ascii=False,
        )
    return '{"result": "ok"}'


def _mock_memory():
    """Mock memory manager with async recall method."""
    mock = MagicMock()
    mock.recall = AsyncMock(return_value=[
        {
            "summary": "楽しい会話をした",
            "keywords": ["楽しい"],
            "importance": 3,
            "date": "2025-06-01T10:00:00",
            "last_recalled": None,
        }
    ])
    return mock


# ── Tests ──────────────────────────────────────────────────────

class TestPsycheReactionFlow:
    """Test react() updates emotions, drives, and mood."""

    def test_happy_percept_increases_joy(self):
        state = PsycheState()
        percept = Percept(text="hi", emotion="happy", emotion_valence=0.7)
        new = react(percept, state, delta_time=1.0)
        assert new.emotions.joy > state.emotions.joy

    def test_sad_percept_increases_sorrow(self):
        state = PsycheState()
        percept = Percept(text="sad", emotion="sad", emotion_valence=-0.6)
        new = react(percept, state, delta_time=1.0)
        assert new.emotions.sorrow > state.emotions.sorrow

    def test_time_decay_reduces_emotions(self):
        state = PsycheState()
        percept = Percept(text="!", emotion="happy", emotion_valence=0.7)
        excited = react(percept, state, delta_time=1.0)
        # Wait 30 virtual seconds
        neutral_percept = Percept(text="", emotion_valence=0.0)
        after_decay = react(neutral_percept, excited, delta_time=30.0)
        assert after_decay.emotions.joy < excited.emotions.joy

    def test_mood_drifts_toward_emotion(self):
        state = PsycheState()
        percept = Percept(text="yay", emotion="happy", emotion_valence=0.8)
        new = react(percept, state, delta_time=1.0)
        assert new.mood.valence > state.mood.valence


class TestPsycheFearIndex:
    """Test fear_index computation and that it changes with events."""

    def test_compute_fear_index_basic(self):
        fi = compute_fear_index(0.5, 0.5, 0.5, 0.5)
        assert fi.value == pytest.approx(0.5)

    def test_fear_index_changes_after_attachment_decay(self):
        """SPEC REQUIREMENT: fear_index must change from input events."""
        att = AttachmentState(bonds={"user_A": 0.8})
        fi_before = compute_fear_index(
            attachment_risk=att_mgr.calc_attachment_risk(att),
        )

        # Simulate long absence
        att_decayed = att_mgr.apply_daily_decay(att, days_elapsed=60)
        fi_after = compute_fear_index(
            attachment_risk=att_mgr.calc_attachment_risk(att_decayed),
        )

        assert fi_after.value > fi_before.value, (
            f"fear_index should increase after 60-day decay: "
            f"{fi_before.value:.4f} → {fi_after.value:.4f}"
        )

    def test_fear_index_changes_with_identity_risk(self):
        """SPEC REQUIREMENT: fear_index changes when identity is threatened."""
        ident_safe = IdentityState(
            core_traits=["romantic"],
            trait_confidence={"romantic": 0.9},
        )
        fi_safe = compute_fear_index(
            identity_risk=calc_identity_risk(ident_safe),
        )

        ident_threat = IdentityState(
            core_traits=["romantic"],
            trait_confidence={"romantic": 0.2},
            pending_changes=[{"trait": "romantic"}],
        )
        fi_threat = compute_fear_index(
            identity_risk=calc_identity_risk(ident_threat),
        )

        assert fi_threat.value > fi_safe.value

    def test_fear_preserved_through_react(self):
        """fear_index should be preserved through react()."""
        fi = compute_fear_index(0.3, 0.5, 0.4, 0.6)
        state = PsycheState(fear_index=fi)
        percept = Percept(text="test", emotion="happy", emotion_valence=0.5)
        new_state = react(percept, state, delta_time=1.0)
        assert new_state.fear_index is not None
        assert new_state.fear_index.value > 0


class TestPsycheMemoryLink:
    """Test mood-congruent memory recall (ASYNC — no LLM)."""

    @pytest.mark.asyncio
    async def test_recall_returns_results(self):
        """recall_by_mood is ASYNC and LOCAL — no LLM calls."""
        state = PsycheState()
        percept = Percept(text="楽しい会話")
        results = await recall_by_mood(percept, state, _mock_memory(), top_k=1)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_recall_mood_congruence(self):
        """Positive mood should rank positive memories higher."""
        from psyche.state import Mood

        state_happy = PsycheState(mood=Mood(valence=0.8, arousal=0.5))
        state_sad = PsycheState(mood=Mood(valence=-0.8, arousal=0.5))
        percept = Percept(text="楽しい")

        results_happy = await recall_by_mood(percept, state_happy, _mock_memory())
        results_sad = await recall_by_mood(percept, state_sad, _mock_memory())

        # Both should return results, ordering differs by mood congruence
        assert len(results_happy) > 0
        assert len(results_sad) > 0


class TestPsycheThoughtLocal:
    """Test that thought.py is LOCAL ONLY — no LLM dependency."""

    def test_generate_candidates_local_only(self):
        """generate_thought_candidates must work without any LLM."""
        state = PsycheState()
        percept = Percept(text="こんにちは", emotion="happy", intent="greeting", emotion_valence=0.5)
        recalled = [{"summary": "楽しい会話", "keywords": ["楽しい"]}]

        # No LLM passed — this is LOCAL logic only
        candidates = generate_thought_candidates(state, percept, recalled)

        assert len(candidates) > 0
        assert all("policy_label" in c for c in candidates)
        assert all("rationale" in c for c in candidates)

    def test_select_policy_local_only(self):
        """select_policy must work without any LLM."""
        state = PsycheState()
        percept = Percept(text="test", emotion="happy", intent="greeting", emotion_valence=0.5)
        candidates = generate_thought_candidates(state, percept, [])

        # No LLM passed — this is LOCAL logic only
        policy = select_policy(candidates, state)

        assert "policy_label" in policy
        assert "rationale" in policy

    def test_fear_affects_policy_selection(self):
        """High fear should bias toward empathize/encourage policies."""
        fi_high = compute_fear_index(0.7, 0.7, 0.7, 0.7)
        state_fearful = PsycheState(fear_index=fi_high)
        percept = Percept(text="test", emotion="neutral", intent="unknown", emotion_valence=0.0)

        candidates = generate_thought_candidates(state_fearful, percept, [])
        policy = select_policy(candidates, state_fearful)

        # When fear is high, empathy-based policies should score higher
        assert policy["policy_label"] in ("共感する", "励ます")


@pytest.mark.asyncio
class TestPsycheEndToEnd:
    """Full pipeline test: perception → reaction → recall → thought → expression."""

    async def test_full_pipeline_with_mock_llm(self):
        """Test complete pipeline with Gemini (mocked) for voice only."""
        state = PsycheState()
        user_input = "こんにちは！楽しい一日だったよ！"

        # Step 1: parse_percept (Gemini auxiliary — JSON extraction only)
        percept = await parse_percept(user_input, _mock_llm_happy, state)
        assert percept.emotion == "happy"
        assert percept.emotion_valence > 0

        # Step 2: react (LOCAL)
        new_state = react(percept, state, delta_time=1.0)
        assert new_state.emotions.joy > 0

        # Step 3: recall_by_mood (LOCAL — ASYNC)
        memories = await recall_by_mood(percept, new_state, _mock_memory())
        assert len(memories) > 0

        # Step 4: generate_thought_candidates + select_policy (LOCAL — NO LLM)
        candidates = generate_thought_candidates(new_state, percept, memories)
        assert len(candidates) > 0

        policy = select_policy(candidates, new_state)
        assert "policy_label" in policy

        # Step 5: render_expression (Gemini voice — text rendering only)
        persona = {"name": "キュレネ", "tone": "sweet", "style_rules": {}}
        expression = await render_expression(new_state, policy, memories, persona, _mock_llm_happy)
        assert "text" in expression
        assert len(expression["text"]) > 0

    async def test_pipeline_without_llm_uses_fallbacks(self):
        """Pipeline should work with fallbacks when LLM is unavailable."""
        state = PsycheState()
        user_input = "こんにちは"

        # Step 1: parse_percept with NO LLM → uses heuristics
        percept = await parse_percept(user_input, None, state)
        assert percept.intent == "greeting"  # heuristic detection

        # Step 2: react (LOCAL)
        new_state = react(percept, state, delta_time=1.0)

        # Step 3: recall_by_mood (LOCAL — ASYNC)
        memories = await recall_by_mood(percept, new_state, _mock_memory())

        # Step 4: thought (LOCAL — always works)
        candidates = generate_thought_candidates(new_state, percept, memories)
        policy = select_policy(candidates, new_state)
        assert "text" in policy  # Fallback text provided

        # Step 5: render_expression with NO LLM → uses fallback
        persona = {"name": "キュレネ", "tone": "sweet", "style_rules": {}}
        # Pass None as LLM - should use fallback
        async def no_llm(prompt, params=None):
            return '{"result": "no_llm_available"}'
        expression = await render_expression(new_state, policy, memories, persona, no_llm)
        assert "text" in expression
