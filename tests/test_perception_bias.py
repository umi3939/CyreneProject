"""
tests/test_perception_bias.py - Tests for perceptual bias (mood-valence-based).

Verifies:
1. _compute_valence_bias returns correct bias from mood.valence
2. Bias is clamped to _BIAS_BANDWIDTH (0.04)
3. State=None produces zero bias (safety valve 4)
4. _heuristic_parse with state produces biased valence
5. _heuristic_parse without state is unbiased (backward compat)
6. Bias does not affect emotion label selection
7. Bias does not affect intent detection
8. Bias is not applied in LLM enrichment stage
9. parse_percept passes state to _heuristic_parse
10. Valence clamping after bias application
11. Bias direction follows mood.valence sign
12. Bias magnitude proportional to mood.valence within bandwidth
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest

from psyche.perception import (
    _BIAS_BANDWIDTH,
    _BIAS_COEFFICIENT,
    _compute_valence_bias,
    _heuristic_parse,
    parse_percept,
    _get_perception_coeffs,
)

# Initialize lazy-loaded coefficients and rebind from the module
# (direct import binds to the initial None values)
_get_perception_coeffs()
import psyche.perception as _perc_mod
_BIAS_BANDWIDTH = _perc_mod._BIAS_BANDWIDTH
_BIAS_COEFFICIENT = _perc_mod._BIAS_COEFFICIENT

from psyche.state import EmotionVector, Mood, Percept, PsycheState


# ── Helpers ───────────────────────────────────────────────────

def _make_state(mood_valence: float = 0.0, mood_arousal: float = 0.3) -> PsycheState:
    """Create a PsycheState with specified mood.valence."""
    return PsycheState(
        emotions=EmotionVector(),
        mood=Mood(valence=mood_valence, arousal=mood_arousal),
    )


@contextmanager
def _hide_llm_wrapper():
    """Temporarily hide src.llm_wrapper so _llm_enrich uses provided fn."""
    saved = {}
    keys_to_hide = [k for k in sys.modules if k == "src.llm_wrapper" or k.startswith("src.llm_wrapper.")]
    for k in keys_to_hide:
        saved[k] = sys.modules.pop(k)
    sys.modules["src.llm_wrapper"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.modules.pop("src.llm_wrapper", None)
        sys.modules.update(saved)


# ── 1. _compute_valence_bias basic behavior ──────────────────

class TestComputeValenceBias:
    """Tests for the pure bias computation function."""

    def test_none_state_returns_zero(self):
        """Safety valve 4: no state => zero bias."""
        assert _compute_valence_bias(None) == 0.0

    def test_zero_valence_returns_zero(self):
        """Neutral mood => zero bias."""
        state = _make_state(mood_valence=0.0)
        assert _compute_valence_bias(state) == 0.0

    def test_positive_valence_produces_positive_bias(self):
        """Positive mood.valence => positive bias."""
        state = _make_state(mood_valence=0.3)
        bias = _compute_valence_bias(state)
        assert bias > 0.0

    def test_negative_valence_produces_negative_bias(self):
        """Negative mood.valence => negative bias."""
        state = _make_state(mood_valence=-0.3)
        bias = _compute_valence_bias(state)
        assert bias < 0.0

    def test_bias_magnitude_proportional(self):
        """Larger |valence| produces larger |bias| (within bandwidth)."""
        state_small = _make_state(mood_valence=0.1)
        state_large = _make_state(mood_valence=0.3)
        bias_small = _compute_valence_bias(state_small)
        bias_large = _compute_valence_bias(state_large)
        assert abs(bias_large) > abs(bias_small)

    def test_bias_clamped_at_positive_bandwidth(self):
        """Safety valve 1: bias cannot exceed +_BIAS_BANDWIDTH."""
        state = _make_state(mood_valence=1.0)
        bias = _compute_valence_bias(state)
        assert bias <= _BIAS_BANDWIDTH
        # valence=1.0 * coefficient=0.1 = 0.1 > 0.04 => clamped
        assert bias == _BIAS_BANDWIDTH

    def test_bias_clamped_at_negative_bandwidth(self):
        """Safety valve 1: bias cannot go below -_BIAS_BANDWIDTH."""
        state = _make_state(mood_valence=-1.0)
        bias = _compute_valence_bias(state)
        assert bias >= -_BIAS_BANDWIDTH
        assert bias == -_BIAS_BANDWIDTH

    def test_bias_within_bandwidth_not_clamped(self):
        """When raw bias is within bandwidth, it is not clamped."""
        # valence=0.2 * 0.1 = 0.02 < 0.04
        state = _make_state(mood_valence=0.2)
        bias = _compute_valence_bias(state)
        expected = 0.2 * _BIAS_COEFFICIENT
        assert abs(bias - expected) < 1e-10

    def test_bias_exact_calculation(self):
        """Verify exact computation: valence * coefficient, then clamp."""
        state = _make_state(mood_valence=0.35)
        bias = _compute_valence_bias(state)
        raw = 0.35 * _BIAS_COEFFICIENT  # 0.035
        expected = max(-_BIAS_BANDWIDTH, min(_BIAS_BANDWIDTH, raw))
        assert abs(bias - expected) < 1e-10

    def test_bandwidth_constant_value(self):
        """Verify _BIAS_BANDWIDTH is 0.04 (base_delta 0.2 / 5)."""
        assert _BIAS_BANDWIDTH == 0.04

    def test_coefficient_constant_value(self):
        """Verify _BIAS_COEFFICIENT is 0.1."""
        assert _BIAS_COEFFICIENT == 0.1


# ── 2. _heuristic_parse with bias ────────────────────────────

class TestHeuristicParseWithBias:
    """Tests for _heuristic_parse with state-dependent bias."""

    def test_no_state_no_bias(self):
        """Without state, valence is unbiased (backward compat)."""
        percept = _heuristic_parse("嬉しいね")
        assert percept.emotion_valence == 0.7
        assert percept.sentiment == 0.7

    def test_none_state_no_bias(self):
        """Explicitly passing None produces same as no state."""
        percept = _heuristic_parse("嬉しいね", state=None)
        assert percept.emotion_valence == 0.7

    def test_positive_mood_increases_valence(self):
        """Positive mood adds positive bias to emotion_valence."""
        state = _make_state(mood_valence=0.3)
        percept = _heuristic_parse("嬉しいね", state=state)
        # 0.7 + 0.03 = 0.73
        expected = 0.7 + 0.3 * _BIAS_COEFFICIENT
        assert abs(percept.emotion_valence - expected) < 1e-10

    def test_negative_mood_decreases_valence(self):
        """Negative mood adds negative bias to emotion_valence."""
        state = _make_state(mood_valence=-0.3)
        percept = _heuristic_parse("嬉しいね", state=state)
        # 0.7 + (-0.03) = 0.67
        expected = 0.7 + (-0.3 * _BIAS_COEFFICIENT)
        assert abs(percept.emotion_valence - expected) < 1e-10

    def test_sentiment_also_biased(self):
        """Sentiment field should also reflect bias."""
        state = _make_state(mood_valence=0.3)
        percept = _heuristic_parse("嬉しいね", state=state)
        assert percept.sentiment == percept.emotion_valence

    def test_negative_keyword_with_positive_mood(self):
        """Bias shifts negative valence toward zero."""
        state = _make_state(mood_valence=0.3)
        percept_biased = _heuristic_parse("悲しい日", state=state)
        percept_neutral = _heuristic_parse("悲しい日")
        # -0.6 + 0.03 = -0.57 vs -0.6
        assert percept_biased.emotion_valence > percept_neutral.emotion_valence

    def test_negative_keyword_with_negative_mood(self):
        """Negative mood makes negative valence more negative."""
        state = _make_state(mood_valence=-0.3)
        percept_biased = _heuristic_parse("悲しい日", state=state)
        percept_neutral = _heuristic_parse("悲しい日")
        assert percept_biased.emotion_valence < percept_neutral.emotion_valence

    def test_emotion_label_not_changed_by_bias(self):
        """Bias affects valence score only, not emotion label."""
        state = _make_state(mood_valence=1.0)  # max positive
        percept = _heuristic_parse("悲しい日", state=state)
        # Emotion label should still be "sad" despite positive bias
        assert percept.emotion == "sad"

    def test_intent_not_affected_by_bias(self):
        """Bias does not affect intent detection."""
        state = _make_state(mood_valence=1.0)
        percept = _heuristic_parse("こんにちは嬉しい", state=state)
        assert percept.intent == "greeting"

    def test_topics_not_affected_by_bias(self):
        """Bias does not affect topic extraction."""
        state = _make_state(mood_valence=1.0)
        percept = _heuristic_parse("嬉しいけど悲しい", state=state)
        assert "嬉しい" in percept.topics
        assert "悲しい" in percept.topics

    def test_neutral_text_with_positive_mood(self):
        """Neutral text (no keywords) gets bias applied to zero valence."""
        state = _make_state(mood_valence=0.3)
        percept = _heuristic_parse("天気がいいですね", state=state)
        expected_bias = 0.3 * _BIAS_COEFFICIENT  # 0.03
        assert abs(percept.emotion_valence - expected_bias) < 1e-10
        assert percept.emotion == "neutral"  # label unchanged

    def test_neutral_text_with_negative_mood(self):
        """Neutral text with negative mood produces slightly negative valence."""
        state = _make_state(mood_valence=-0.3)
        percept = _heuristic_parse("天気がいいですね", state=state)
        expected_bias = -0.3 * _BIAS_COEFFICIENT  # -0.03
        assert abs(percept.emotion_valence - expected_bias) < 1e-10

    def test_valence_clamped_after_bias_positive(self):
        """Valence + bias should still be clamped to 1.0."""
        # 愛してる has valence=0.9, bias with max mood = 0.04
        # 0.9 + 0.04 = 0.94 (within range, no clamping needed)
        state = _make_state(mood_valence=1.0)
        percept = _heuristic_parse("愛してる", state=state)
        assert percept.emotion_valence <= 1.0
        assert abs(percept.emotion_valence - 0.94) < 1e-10

    def test_valence_clamped_after_bias_negative(self):
        """Valence + bias should still be clamped to -1.0."""
        # 恐ろし has valence=-0.7, bias with max negative mood = -0.04
        # -0.7 + (-0.04) = -0.74 (within range)
        state = _make_state(mood_valence=-1.0)
        percept = _heuristic_parse("恐ろしい日", state=state)
        assert percept.emotion_valence >= -1.0
        assert abs(percept.emotion_valence - (-0.74)) < 1e-10

    def test_max_bias_is_small_relative_to_base_delta(self):
        """Verify bias bandwidth is 1/5 of reaction.py base_delta (0.2)."""
        assert _BIAS_BANDWIDTH == 0.2 / 5


# ── 3. parse_percept integration with bias ───────────────────

class TestParsePerceptBiasIntegration:
    """Tests for parse_percept passing state through to heuristic."""

    @pytest.mark.asyncio
    async def test_parse_percept_without_state_no_bias(self):
        """parse_percept without state has no bias."""
        percept = await parse_percept("嬉しいね")
        assert percept.emotion_valence == 0.7

    @pytest.mark.asyncio
    async def test_parse_percept_with_state_has_bias(self):
        """parse_percept with state applies bias in heuristic stage."""
        state = _make_state(mood_valence=0.3)
        percept = await parse_percept("嬉しいね", state=state)
        expected = 0.7 + 0.3 * _BIAS_COEFFICIENT
        assert abs(percept.emotion_valence - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_llm_enrichment_ignores_bias(self):
        """LLM enrichment stage does not apply bias - uses LLM's own valence."""
        state = _make_state(mood_valence=1.0)
        llm_response = json.dumps({
            "meaning": "嬉しいこと",
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 0.5,
            "topics": ["感情"],
        }, ensure_ascii=False)
        mock_fn = AsyncMock(return_value=llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn, state=state)
        # LLM returns 0.5 directly, no bias added
        assert percept.emotion_valence == 0.5

    @pytest.mark.asyncio
    async def test_llm_fallback_preserves_biased_baseline(self):
        """When LLM fails, baseline (biased) is preserved."""
        state = _make_state(mood_valence=0.3)
        mock_fn = AsyncMock(return_value="invalid json!!!")
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn, state=state)
        # Falls back to biased heuristic
        expected = 0.7 + 0.3 * _BIAS_COEFFICIENT
        assert abs(percept.emotion_valence - expected) < 1e-10


# ── 4. Safety valve verification ─────────────────────────────

class TestSafetyValves:
    """Verify all safety valves from the design spec."""

    def test_sv1_bandwidth_absolute_upper_bound(self):
        """Safety valve 1: bias magnitude never exceeds _BIAS_BANDWIDTH."""
        for v in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            state = _make_state(mood_valence=v)
            bias = _compute_valence_bias(state)
            assert abs(bias) <= _BIAS_BANDWIDTH + 1e-10

    def test_sv3_no_accumulation(self):
        """Safety valve 3: bias is recomputed each call, no accumulation."""
        state = _make_state(mood_valence=0.3)
        bias1 = _compute_valence_bias(state)
        bias2 = _compute_valence_bias(state)
        bias3 = _compute_valence_bias(state)
        assert bias1 == bias2 == bias3  # Identical across calls

    def test_sv4_no_state_neutralizes(self):
        """Safety valve 4: no state => zero bias."""
        assert _compute_valence_bias(None) == 0.0

    def test_bias_is_pure_function(self):
        """Bias computation is deterministic: same input => same output."""
        state = _make_state(mood_valence=0.5)
        results = [_compute_valence_bias(state) for _ in range(10)]
        assert len(set(results)) == 1

    def test_bias_reversible_with_opposite_valence(self):
        """Bias direction reverses when valence flips."""
        state_pos = _make_state(mood_valence=0.3)
        state_neg = _make_state(mood_valence=-0.3)
        bias_pos = _compute_valence_bias(state_pos)
        bias_neg = _compute_valence_bias(state_neg)
        assert abs(bias_pos + bias_neg) < 1e-10  # They cancel out
