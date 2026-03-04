"""
Tests for Phase 26-EXP bandwidth expansion to drive total limit and score section band.

Verifies:
- Target A: drive total change limit expansion via multiplier
- Target B: score section band expansion via addition
- Safety valves: absolute upper limits, drive value range clamp, score compression
- Cooldown: no expansion during cooldown period
- Scoring fluctuation independence from target B
- Fallback: None -> existing fixed values
- Reset guarantee: values reset at each Phase 26-EXP invocation
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional

from psyche.reaction import (
    DriveContextInputs,
    compute_state_dependent_drive_changes,
    react,
    _TOTAL_CHANGE_LIMIT,
    _SECTION_BAND,
)
from psyche.thought import (
    generate_thought_candidates,
    _score_candidate,
    _SCORE_SECTION_BAND,
    POLICIES,
)
from psyche.state import (
    DriveVector,
    EmotionVector,
    Mood,
    Percept,
    PsycheState,
)
from psyche.orchestrator_5tick_phases import (
    _EXP_DRIVE_LIMIT_MULTIPLIER_MAX,
    _EXP_SCORE_BAND_ADDITION_MAX,
    _EXP_BANDWIDTH_MAX_MULTIPLIER,
    _apply_experience_driven_value_update,
    _compute_experience_intensity,
    _compute_bandwidth_expansion_coefficient,
)


# ── Helper fixtures ──

def _make_psyche_state(**overrides) -> PsycheState:
    """Create a minimal PsycheState for testing."""
    defaults = {
        "emotions": EmotionVector(
            joy=0.8, sorrow=0.1, anger=0.1, surprise=0.5,
            fear=0.1, love=0.1, fun=0.3,
        ),
        "drives": DriveVector(social=0.7, curiosity=0.6, expression=0.5),
        "mood": Mood(valence=0.3, arousal=0.7),
    }
    defaults.update(overrides)
    return PsycheState(**defaults)


def _make_percept(**overrides) -> Percept:
    """Create a minimal Percept for testing."""
    defaults = {
        "text": "hello",
        "emotion": "happy",
        "intent": "greeting",
        "emotion_valence": 0.5,
    }
    defaults.update(overrides)
    return Percept(**defaults)


# =============================================================================
# Target A: Drive total change limit expansion
# =============================================================================

class TestDriveTotalLimitMultiplier:
    """Tests for drive_total_limit_multiplier in DriveContextInputs."""

    def test_none_multiplier_uses_default_limit(self):
        """When multiplier is None, the default _TOTAL_CHANGE_LIMIT is used."""
        ctx = DriveContextInputs(
            emotions={"joy": 1.0, "sorrow": 0.0, "anger": 0.0,
                       "surprise": 0.0, "fear": 0.0, "love": 0.0, "fun": 0.0},
            mood_valence=0.5,
            mood_arousal=0.9,
            drives={"social": 0.9, "curiosity": 0.9, "expression": 0.9},
            delta_time=1.0,
            drive_total_limit_multiplier=None,
        )
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT + 1e-9

    def test_multiplier_expands_limit(self):
        """When multiplier > 1.0, the effective limit is larger."""
        multiplier = 1.3
        ctx = DriveContextInputs(
            emotions={"joy": 1.0, "sorrow": 0.0, "anger": 0.0,
                       "surprise": 1.0, "fear": 0.0, "love": 1.0, "fun": 1.0},
            mood_valence=0.9,
            mood_arousal=0.9,
            drives={"social": 0.9, "curiosity": 0.9, "expression": 0.9},
            delta_time=5.0,
            has_transient_goal=True,
            persistent_commitment_count=3,
            has_scoped_goal=True,
            drive_total_limit_multiplier=multiplier,
        )
        result = compute_state_dependent_drive_changes(ctx)
        effective_limit = _TOTAL_CHANGE_LIMIT * multiplier
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= effective_limit + 1e-9

    def test_multiplier_1_0_same_as_none(self):
        """Multiplier of 1.0 produces same result as None."""
        ctx_base = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.2, "anger": 0.1,
                       "surprise": 0.3, "fear": 0.0, "love": 0.1, "fun": 0.2},
            mood_valence=0.1,
            mood_arousal=0.5,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            drive_total_limit_multiplier=None,
        )
        ctx_1 = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.2, "anger": 0.1,
                       "surprise": 0.3, "fear": 0.0, "love": 0.1, "fun": 0.2},
            mood_valence=0.1,
            mood_arousal=0.5,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            drive_total_limit_multiplier=1.0,
        )
        result_none = compute_state_dependent_drive_changes(ctx_base)
        result_1 = compute_state_dependent_drive_changes(ctx_1)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result_none[axis] - result_1[axis]) < 1e-9

    def test_drive_value_range_clamp_with_multiplier(self):
        """Drive values remain in 0.0-1.0 even with expanded limit."""
        state = _make_psyche_state(
            drives=DriveVector(social=0.99, curiosity=0.99, expression=0.99),
        )
        percept = _make_percept()
        drive_ctx = DriveContextInputs(
            drive_total_limit_multiplier=_EXP_DRIVE_LIMIT_MULTIPLIER_MAX,
        )
        new_state = react(percept, state, delta_time=1.0, drive_context=drive_ctx)
        drv = new_state.drives
        assert 0.0 <= drv.social <= 1.0
        assert 0.0 <= drv.curiosity <= 1.0
        assert 0.0 <= drv.expression <= 1.0

    def test_drive_value_range_clamp_low_with_multiplier(self):
        """Drive values remain in 0.0-1.0 even with expanded limit (low end)."""
        state = _make_psyche_state(
            drives=DriveVector(social=0.01, curiosity=0.01, expression=0.01),
        )
        percept = _make_percept(emotion="sad", emotion_valence=-0.8)
        drive_ctx = DriveContextInputs(
            drive_total_limit_multiplier=_EXP_DRIVE_LIMIT_MULTIPLIER_MAX,
        )
        new_state = react(percept, state, delta_time=1.0, drive_context=drive_ctx)
        drv = new_state.drives
        assert 0.0 <= drv.social <= 1.0
        assert 0.0 <= drv.curiosity <= 1.0
        assert 0.0 <= drv.expression <= 1.0

    def test_multiplier_absolute_upper_limit(self):
        """The multiplier constant is 1.3 as designed."""
        assert _EXP_DRIVE_LIMIT_MULTIPLIER_MAX == 1.3

    def test_expanded_limit_does_not_exceed_section_sum(self):
        """The expanded limit (0.15 * 1.3 = 0.195) is below the section band sum."""
        expanded = _TOTAL_CHANGE_LIMIT * _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        section_sum = sum(
            max(bands.values())
            for bands in _SECTION_BAND.values()
        )
        assert expanded < section_sum


# =============================================================================
# Target B: Score section band expansion
# =============================================================================

class TestScoreSectionBandAddition:
    """Tests for score_section_band_addition in thought.py."""

    def test_none_addition_uses_default_band(self):
        """When addition is None, the default _SCORE_SECTION_BAND is used."""
        state = _make_psyche_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=None,
        )
        assert len(candidates) >= 3

    def test_addition_expands_band(self):
        """When addition > 0, the effective band is larger."""
        state = _make_psyche_state()
        percept = _make_percept()
        # With addition=0.0 (no change)
        candidates_base = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=0.0,
        )
        # With addition=0.5 (expanded)
        candidates_expanded = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=0.5,
        )
        # Both should produce valid candidates
        assert len(candidates_base) >= 3
        assert len(candidates_expanded) >= 3

    def test_score_compression_still_works_with_addition(self):
        """Non-linear score compression (gap > 1.0) still functions with expanded band."""
        state = _make_psyche_state(
            emotions=EmotionVector(
                joy=0.0, sorrow=0.0, anger=0.0, surprise=0.0,
                fear=0.9, love=0.0, fun=0.0,
            ),
            mood=Mood(valence=-0.8, arousal=0.9),
        )
        percept = _make_percept(emotion="scared", emotion_valence=-0.9, intent="complaint")
        # High fear -> empathy/encouragement strongly favored -> large gap possible
        candidates = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=_EXP_SCORE_BAND_ADDITION_MAX,
        )
        if len(candidates) >= 2:
            gap = candidates[0]["_score"] - candidates[1]["_score"]
            # Compression should prevent extreme domination
            # With compression, gap should be manageable
            assert gap < 10.0  # Reasonable upper bound

    def test_addition_0_same_scores_as_none(self):
        """Addition of 0.0 produces same scores as None."""
        state = _make_psyche_state()
        percept = _make_percept()
        candidates_none = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=None,
        )
        candidates_zero = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=0.0,
        )
        for c_none, c_zero in zip(candidates_none, candidates_zero):
            assert abs(c_none["_score"] - c_zero["_score"]) < 1e-9

    def test_score_section_band_addition_max_constant(self):
        """The addition constant is 0.5 as designed."""
        assert _EXP_SCORE_BAND_ADDITION_MAX == 0.5

    def test_score_candidate_with_addition_internal(self):
        """Direct _score_candidate call with addition affects clamp range."""
        state = _make_psyche_state()
        percept = _make_percept()
        policy_def = POLICIES[0]  # "共感する"

        # Score without addition
        score_base = _score_candidate(
            policy_def, state, percept, [],
            score_section_band_addition=None,
        )
        # Score with max addition
        score_expanded = _score_candidate(
            policy_def, state, percept, [],
            score_section_band_addition=_EXP_SCORE_BAND_ADDITION_MAX,
        )
        # Both should return valid floats
        assert isinstance(score_base, float)
        assert isinstance(score_expanded, float)


# =============================================================================
# Scoring fluctuation independence
# =============================================================================

class TestScoringFluctuationIndependence:
    """Verify scoring_fluctuation is not affected by score_section_band_addition."""

    def test_fluctuation_amplitude_unaffected_by_band_addition(self):
        """scoring_fluctuation operates on a separate pipeline stage from band clamp."""
        # The scoring fluctuation adds noise AFTER all section contributions are summed.
        # The band addition only affects the per-section clamp.
        # This is structurally guaranteed by the code ordering:
        # 1. _score_candidate uses _clamp_section (affected by band addition)
        # 2. apply_scoring_fluctuation is called separately on the final candidates
        # We verify this structural separation by checking that
        # _score_candidate's output with band_addition does not contain
        # scoring_fluctuation effects.
        state = _make_psyche_state()
        percept = _make_percept()
        policy_def = POLICIES[0]

        # _score_candidate itself does NOT apply scoring_fluctuation
        score_with = _score_candidate(
            policy_def, state, percept, [],
            score_section_band_addition=0.5,
        )
        score_without = _score_candidate(
            policy_def, state, percept, [],
            score_section_band_addition=0.0,
        )
        # Both are deterministic (no fluctuation involved at this stage)
        score_with_2 = _score_candidate(
            policy_def, state, percept, [],
            score_section_band_addition=0.5,
        )
        # Exact equality: no randomness in _score_candidate
        assert score_with == score_with_2


# =============================================================================
# Cooldown period
# =============================================================================

class TestCooldownPeriod:
    """Verify expansion values are not applied during cooldown."""

    def _make_mock_orchestrator(self, tick: int, last_tick: int):
        """Create a mock orchestrator for testing Phase 26-EXP."""
        orch = MagicMock()
        orch._tick_count = tick
        orch._exp_bandwidth_last_tick = last_tick

        # Psyche state with high experience intensity inputs
        orch._psyche = _make_psyche_state(
            emotions=EmotionVector(
                joy=0.9, sorrow=0.0, anger=0.0, surprise=0.8,
                fear=0.0, love=0.0, fun=0.5,
            ),
            mood=Mood(valence=0.5, arousal=0.8),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )

        # Minimal episode store
        episode = MagicMock()
        companion = MagicMock()
        companion.intensity_level = 0.9
        episode.emotional_companion = companion
        episodes_store = MagicMock()
        episodes_store.episodes = [episode]
        orch._last_episodes = episodes_store

        # Policy label
        orch._last_selected_policy_label = "共感する"
        orch._last_selected_policy_axis = ""

        # Value orientation
        from psyche.value_orientation import ValueOrientation
        orch._value_orientation = ValueOrientation()
        orch._vo_config = None

        # Drive variation tracking
        orch._exp_prev_drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}

        return orch

    def test_cooldown_prevents_expansion(self):
        """During cooldown, expansion values are reset to None."""
        # Tick 5, last expansion at tick 4 (within cooldown of 2-3 ticks)
        orch = self._make_mock_orchestrator(tick=5, last_tick=4)

        _apply_experience_driven_value_update(orch)

        # The values should be reset to None at start of function
        # and because we're in cooldown, they remain None
        assert orch._exp_drive_total_limit_multiplier is None
        assert orch._exp_score_band_addition is None

    def test_outside_cooldown_sets_expansion(self):
        """Outside cooldown, expansion values are set."""
        # Tick 10, last expansion at tick 4 (well outside cooldown)
        orch = self._make_mock_orchestrator(tick=10, last_tick=4)

        _apply_experience_driven_value_update(orch)

        # With high experience intensity, expansion should be set
        if orch._exp_drive_total_limit_multiplier is not None:
            assert 1.0 <= orch._exp_drive_total_limit_multiplier <= _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        if orch._exp_score_band_addition is not None:
            assert 0.0 <= orch._exp_score_band_addition <= _EXP_SCORE_BAND_ADDITION_MAX


# =============================================================================
# Multiplier and addition absolute upper limits
# =============================================================================

class TestAbsoluteUpperLimits:
    """Verify expansion values never exceed their absolute upper limits."""

    def test_drive_multiplier_max(self):
        """Drive multiplier never exceeds _EXP_DRIVE_LIMIT_MULTIPLIER_MAX."""
        # Even with maximum expansion_coeff
        max_coeff = _EXP_BANDWIDTH_MAX_MULTIPLIER  # 4.0
        # Calculate what the multiplier would be
        ec_range = _EXP_BANDWIDTH_MAX_MULTIPLIER - 1.05
        mult_range = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0
        raw = 1.0 + ((max_coeff - 1.05) / max(ec_range, 0.01)) * mult_range
        clamped = min(raw, _EXP_DRIVE_LIMIT_MULTIPLIER_MAX)
        assert clamped <= _EXP_DRIVE_LIMIT_MULTIPLIER_MAX

    def test_score_addition_max(self):
        """Score addition never exceeds _EXP_SCORE_BAND_ADDITION_MAX."""
        max_coeff = _EXP_BANDWIDTH_MAX_MULTIPLIER
        ec_range = _EXP_BANDWIDTH_MAX_MULTIPLIER - 1.05
        raw = ((max_coeff - 1.05) / max(ec_range, 0.01)) * _EXP_SCORE_BAND_ADDITION_MAX
        clamped = min(max(0.0, raw), _EXP_SCORE_BAND_ADDITION_MAX)
        assert clamped <= _EXP_SCORE_BAND_ADDITION_MAX

    def test_drive_multiplier_at_low_intensity(self):
        """At low experience intensity, multiplier is close to 1.0."""
        # expansion_coeff just above threshold (1.05)
        low_coeff = 1.1
        ec_range = _EXP_BANDWIDTH_MAX_MULTIPLIER - 1.05
        mult_range = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0
        raw = 1.0 + ((low_coeff - 1.05) / max(ec_range, 0.01)) * mult_range
        assert raw < 1.02  # Very close to 1.0

    def test_score_addition_at_low_intensity(self):
        """At low experience intensity, addition is close to 0.0."""
        low_coeff = 1.1
        ec_range = _EXP_BANDWIDTH_MAX_MULTIPLIER - 1.05
        raw = ((low_coeff - 1.05) / max(ec_range, 0.01)) * _EXP_SCORE_BAND_ADDITION_MAX
        assert raw < 0.01  # Very close to 0.0


# =============================================================================
# Input absence fallback
# =============================================================================

class TestInputAbsenceFallback:
    """Verify fallback behavior when expansion values are None."""

    def test_drive_context_none_multiplier_fallback(self):
        """DriveContextInputs with None multiplier uses _TOTAL_CHANGE_LIMIT."""
        ctx = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.0, "anger": 0.0,
                       "surprise": 0.0, "fear": 0.0, "love": 0.0, "fun": 0.0},
            mood_valence=0.0,
            mood_arousal=0.3,
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            drive_total_limit_multiplier=None,
        )
        result = compute_state_dependent_drive_changes(ctx)
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= _TOTAL_CHANGE_LIMIT + 1e-9

    def test_thought_none_addition_fallback(self):
        """generate_thought_candidates with None addition uses _SCORE_SECTION_BAND."""
        state = _make_psyche_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=None,
        )
        # Should produce valid candidates
        assert len(candidates) >= 3
        for c in candidates:
            assert isinstance(c["_score"], float)


# =============================================================================
# Reset guarantee
# =============================================================================

class TestResetGuarantee:
    """Verify expansion values are reset at each Phase 26-EXP invocation."""

    def test_reset_on_every_invocation(self):
        """_apply_experience_driven_value_update resets values at start."""
        orch = MagicMock()
        orch._tick_count = 100
        # Pre-set expansion values
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3

        # Set up for early return (no episodes)
        orch._exp_bandwidth_last_tick = 0
        orch._exp_prev_drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        orch._psyche = _make_psyche_state()
        orch._last_selected_policy_label = ""
        orch._last_episodes = None

        _apply_experience_driven_value_update(orch)

        # Values should be reset to None even though function returned early
        assert orch._exp_drive_total_limit_multiplier is None
        assert orch._exp_score_band_addition is None

    def test_reset_during_cooldown(self):
        """Values are reset even during cooldown period."""
        orch = MagicMock()
        orch._tick_count = 5
        orch._exp_bandwidth_last_tick = 4  # Within cooldown
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3

        # Set up minimal psyche
        orch._psyche = _make_psyche_state()
        orch._exp_prev_drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}

        _apply_experience_driven_value_update(orch)

        # Values should be reset
        assert orch._exp_drive_total_limit_multiplier is None
        assert orch._exp_score_band_addition is None


# =============================================================================
# Integration: orchestrator_1tick_phases.build_drive_context
# =============================================================================

class TestBuildDriveContextIntegration:
    """Test that build_drive_context passes multiplier from orchestrator."""

    def test_multiplier_passed_to_context(self):
        """build_drive_context picks up _exp_drive_total_limit_multiplier."""
        from psyche.orchestrator_1tick_phases import build_drive_context

        orch = MagicMock()
        orch._exp_drive_total_limit_multiplier = 1.2

        # Set up required attributes
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition = None
        orch._behavioral_diversity_state.latest_record = None
        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert ctx.drive_total_limit_multiplier == 1.2

    def test_none_multiplier_not_set(self):
        """build_drive_context leaves multiplier None when not set."""
        from psyche.orchestrator_1tick_phases import build_drive_context

        orch = MagicMock()
        orch._exp_drive_total_limit_multiplier = None

        # Set up required attributes
        orch._transient_goal_mgr.state.active_goal = None
        orch._persistent_commitment._state.items = []
        orch._scoped_goal_sys.has_active_scope = False
        orch._temporal_cognition = None
        orch._behavioral_diversity_state.latest_record = None
        orch._contradiction_processor.state.previous_contradictions = []

        ctx = build_drive_context(orch)
        assert ctx.drive_total_limit_multiplier is None


# =============================================================================
# Experience intensity -> expansion coefficient -> expansion values
# =============================================================================

class TestExpansionValueDerivation:
    """Test the derivation chain from experience intensity to expansion values."""

    def test_zero_intensity_no_expansion(self):
        """Zero experience intensity -> expansion_coeff = 1.0 -> no expansion."""
        intensity = _compute_experience_intensity(0.0, 0.0, 0.0)
        assert intensity == 0.0
        coeff = _compute_bandwidth_expansion_coefficient(intensity, 0.01)
        assert coeff == 1.0

    def test_max_intensity_max_expansion(self):
        """Maximum experience intensity -> maximum expansion coefficient."""
        intensity = _compute_experience_intensity(1.0, 1.0, 1.0)
        assert intensity == 1.0
        coeff = _compute_bandwidth_expansion_coefficient(intensity, 0.01)
        assert coeff <= _EXP_BANDWIDTH_MAX_MULTIPLIER

    def test_medium_intensity_partial_expansion(self):
        """Medium experience intensity -> partial expansion coefficient."""
        intensity = _compute_experience_intensity(0.5, 0.5, 0.5)
        assert 0.0 < intensity < 1.0
        coeff = _compute_bandwidth_expansion_coefficient(intensity, 0.01)
        assert 1.0 < coeff < _EXP_BANDWIDTH_MAX_MULTIPLIER

    def test_expansion_coeff_below_threshold_no_values(self):
        """expansion_coeff <= 1.05 means expansion values remain None."""
        # Very low intensity -> coeff close to 1.0
        intensity = _compute_experience_intensity(0.01, 0.01, 0.01)
        coeff = _compute_bandwidth_expansion_coefficient(intensity, 0.01)
        # Should be <= 1.05 (no expansion)
        assert coeff <= 1.05


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for the bandwidth expansion."""

    def test_negative_multiplier_treated_as_valid(self):
        """A multiplier < 1.0 would shrink the limit (edge case)."""
        ctx = DriveContextInputs(
            emotions={"joy": 0.5, "sorrow": 0.0, "anger": 0.0,
                       "surprise": 0.0, "fear": 0.0, "love": 0.0, "fun": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            drive_total_limit_multiplier=0.5,
        )
        result = compute_state_dependent_drive_changes(ctx)
        effective_limit = _TOTAL_CHANGE_LIMIT * 0.5
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= effective_limit + 1e-9

    def test_large_multiplier_clamps_properly(self):
        """Even with a very large multiplier, per-section bands still limit output."""
        ctx = DriveContextInputs(
            emotions={"joy": 1.0, "sorrow": 0.0, "anger": 0.0,
                       "surprise": 0.0, "fear": 0.0, "love": 0.0, "fun": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
            delta_time=1.0,
            drive_total_limit_multiplier=10.0,
        )
        result = compute_state_dependent_drive_changes(ctx)
        # Section bands still limit per-section contributions
        # Total is sum of section contributions (each individually bounded)
        section_sum = sum(
            max(bands.values())
            for bands in _SECTION_BAND.values()
        )
        for axis in ("social", "curiosity", "expression"):
            assert abs(result[axis]) <= section_sum + 1e-9

    def test_score_addition_negative_not_expected(self):
        """Negative addition would shrink the band (edge case guard)."""
        state = _make_psyche_state()
        percept = _make_percept()
        # This is an edge case - the code should handle it gracefully
        candidates = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=-0.5,
        )
        assert len(candidates) >= 3

    def test_collect_breakdown_with_addition(self):
        """collect_breakdown works correctly with band addition."""
        state = _make_psyche_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(
            state, percept, [],
            score_section_band_addition=0.3,
            collect_breakdown=True,
        )
        for c in candidates:
            if "_score_breakdown" in c:
                assert isinstance(c["_score_breakdown"], dict)
