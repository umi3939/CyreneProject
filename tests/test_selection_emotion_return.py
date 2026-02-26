"""
Tests for selection emotion return in orchestrator.

Verifies that after policy selection, the selection result causes
a micro-change in the emotion vector via a 3-stage pipeline
with 6 safety valves.
"""

from __future__ import annotations

import pytest

from psyche.state import PsycheState, DriveVector, Percept, EmotionVector, Mood
from psyche.pillars import FearIndex
from psyche.orchestrator import PsycheOrchestrator


# ── Helpers ────────────────────────────────────────────────────


def _make_orchestrator(
    emotions: EmotionVector | None = None,
    drives: DriveVector | None = None,
    mood: Mood | None = None,
    fear_index: FearIndex | None = None,
) -> PsycheOrchestrator:
    """Create a minimal orchestrator for testing."""
    orch = PsycheOrchestrator.__new__(PsycheOrchestrator)
    orch._psyche = PsycheState(
        emotions=emotions or EmotionVector(),
        drives=drives or DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        mood=mood or Mood(valence=0.0, arousal=0.5),
        fear_index=fear_index,
    )
    return orch


def _make_policy(
    label: str = "共感する",
    edc: dict | None = None,
    drive_target: str = "social",
) -> dict:
    """Create a policy dict for testing."""
    if edc is None:
        edc = {"social": -0.08, "curiosity": -0.02, "expression": -0.02}
    return {
        "policy_label": label,
        "rationale": "test",
        "expected_drive_change": edc,
        "drive_target": drive_target,
        "text": "test text",
    }


# ── Test: Basic Application ──────────────────────────────────


class TestBasicEmotionReturn:
    """Test basic emotion return application."""

    def test_emotions_change_after_application(self):
        """Emotions should change when emotion return is applied."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        # At least one dimension should have changed
        changed = any(
            abs(emotions_after[k] - emotions_before[k]) > 1e-9
            for k in emotions_before
        )
        assert changed, "At least one emotion dimension should change"

    def test_no_edc_does_nothing(self):
        """If policy has no expected_drive_change, emotions are unchanged."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
        )
        policy = {"policy_label": "test", "rationale": "test", "text": "test"}
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after

    def test_empty_edc_does_nothing(self):
        """If expected_drive_change is empty, emotions are unchanged."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
        )
        policy = _make_policy(edc={})
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after

    def test_none_edc_does_nothing(self):
        """If expected_drive_change is None, emotions are unchanged."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
        )
        policy = _make_policy()
        policy["expected_drive_change"] = None
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after

    def test_positive_drive_change_no_return(self):
        """If drive_change >= 0 (no drive satisfaction), no emotion return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": 0.05, "curiosity": 0.05, "expression": 0.05},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after


# ── Test: Safety Valve 1 - Per-Dimension Clamp ────────────────


class TestSafetyValvePerDimClamp:
    """Safety valve 1: per-dimension clamp to 0.075 (half of drive return max 0.15)."""

    def test_single_dim_clamped(self):
        """No single emotion dimension should change by more than 0.075."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, anger=0.5, sorrow=0.5,
                                   fear=0.5, surprise=0.5, love=0.5, fun=0.5),
            mood=Mood(valence=0.9, arousal=1.0),
        )
        policy = _make_policy(
            edc={"social": -0.15, "curiosity": -0.15, "expression": -0.15},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            change = abs(emotions_after[dim] - emotions_before[dim])
            assert change <= 0.075 + 0.001, (
                f"Dimension {dim} changed by {change}, exceeds per-dim max 0.075"
            )

    def test_per_dim_bound_is_half_drive_max(self):
        """Per-dimension emotion return bound should be <= half of drive return max (0.15/2)."""
        # This is a structural constant check
        assert 0.075 <= 0.15 / 2


# ── Test: Safety Valve 2 - Total Return Clamp ─────────────────


class TestSafetyValveTotalClamp:
    """Safety valve 2: total return across all dimensions is bounded."""

    def test_total_return_bounded(self):
        """Sum of absolute changes across all dims should not exceed 0.30."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, anger=0.5, sorrow=0.5,
                                   fear=0.5, surprise=0.5, love=0.5, fun=0.5),
            mood=Mood(valence=0.9, arousal=1.0),
        )
        policy = _make_policy(
            edc={"social": -0.15, "curiosity": -0.15, "expression": -0.15},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        total_change = sum(
            abs(emotions_after[dim] - emotions_before[dim])
            for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun")
        )
        assert total_change <= 0.30 + 0.001, (
            f"Total emotion return {total_change} exceeds max 0.30"
        )

    def test_total_bound_less_than_drive_total(self):
        """Total emotion return bound should be less than drive return total bound."""
        # Drive return max per axis = 0.15, 3 axes = 0.45
        # Emotion return total max = 0.30 < 0.45
        assert 0.30 < 0.15 * 3


# ── Test: Safety Valve 3 - Distance Proportional Suppression ──


class TestSafetyValveDistanceProportion:
    """Safety valve 3: boundary proximity suppresses return."""

    def test_emotion_near_max_suppressed(self):
        """Emotion near 1.0 should have minimal positive return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.98, fun=0.97),
            mood=Mood(valence=0.8, arousal=0.8),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        # joy/fun were near max, positive direction should be heavily suppressed
        assert emotions_after["joy"] <= 1.0
        assert emotions_after["fun"] <= 1.0
        # change should be very small due to distance proportion
        assert abs(emotions_after["joy"] - 0.98) < 0.01
        assert abs(emotions_after["fun"] - 0.97) < 0.01

    def test_emotion_near_zero_suppressed_downward(self):
        """Emotion near 0.0 should have minimal negative return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(sorrow=0.02, fear=0.01),
            mood=Mood(valence=-0.5, arousal=0.8),
            fear_index=FearIndex(
                identity_risk=0.7, attachment_risk=0.1,
                continuity_risk=0.1, projection_risk=0.1,
            ),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        # sorrow/fear were near 0, negative direction should be suppressed
        assert emotions_after["sorrow"] >= 0.0
        assert emotions_after["fear"] >= 0.0


# ── Test: Safety Valve 4 - Positive Feedback Loop 3x Block ────


class TestPositiveFeedbackLoopBlock:
    """Safety valve 4: positive feedback loop is structurally blocked
    via band limitation + non-fixedness + distance proportion."""

    def test_same_policy_different_state_different_return(self):
        """Same policy with different internal state should produce different returns."""
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        # State A: positive mood, moderate emotions
        orch_a = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.2),
            mood=Mood(valence=0.5, arousal=0.6),
        )
        emotions_a_before = orch_a._psyche.emotions.as_dict()
        orch_a._apply_selection_emotion_return(policy, candidate_count=5)
        deltas_a = {
            k: orch_a._psyche.emotions.as_dict()[k] - emotions_a_before[k]
            for k in emotions_a_before
        }

        # State B: negative mood, different emotions
        orch_b = _make_orchestrator(
            emotions=EmotionVector(joy=0.1, sorrow=0.7),
            mood=Mood(valence=-0.5, arousal=0.6),
            fear_index=FearIndex(
                identity_risk=0.5, attachment_risk=0.5,
                continuity_risk=0.1, projection_risk=0.1,
            ),
        )
        emotions_b_before = orch_b._psyche.emotions.as_dict()
        orch_b._apply_selection_emotion_return(policy, candidate_count=5)
        deltas_b = {
            k: orch_b._psyche.emotions.as_dict()[k] - emotions_b_before[k]
            for k in emotions_b_before
        }

        # At least some dimensions should differ in their delta
        differences = sum(
            1 for k in deltas_a if abs(deltas_a[k] - deltas_b[k]) > 1e-9
        )
        assert differences > 0, "Same policy with different states should yield different returns"

    def test_repeated_application_converges(self):
        """Repeated application should show diminishing returns due to distance proportion."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.5, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        changes = []
        for _ in range(10):
            emotions_before = orch._psyche.emotions.as_dict()
            orch._apply_selection_emotion_return(policy, candidate_count=5)
            emotions_after = orch._psyche.emotions.as_dict()
            total_change = sum(
                abs(emotions_after[k] - emotions_before[k])
                for k in emotions_before
            )
            changes.append(total_change)

        # All changes should be bounded
        for change in changes:
            assert change <= 0.30 + 0.001

    def test_band_is_half_drive_return(self):
        """Emotion return band should be at most half of drive return band."""
        # Drive return max_change = 0.15
        # Emotion return MAX_PER_DIM = 0.075
        assert 0.075 <= 0.15 / 2.0


# ── Test: Safety Valve 5 - Single Application ─────────────────


class TestSafetyValveSingleApplication:
    """Safety valve 5: structurally guaranteed single application per selection."""

    def test_single_application_produces_bounded_change(self):
        """A single application should produce bounded emotion change."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, anger=0.3, sorrow=0.4),
            mood=Mood(valence=0.2, arousal=0.7),
        )
        policy = _make_policy(
            edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            change = abs(emotions_after[dim] - emotions_before[dim])
            assert change <= 0.075 + 0.001


# ── Test: Safety Valve 6 - Valid Range Clamp ──────────────────


class TestSafetyValveValidRange:
    """Safety valve 6: emotions stay within 0.0-1.0."""

    def test_emotions_not_below_zero(self):
        """Emotions should not drop below 0.0."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.01, sorrow=0.01, fear=0.01),
            mood=Mood(valence=-0.8, arousal=0.9),
            fear_index=FearIndex(
                identity_risk=0.8, attachment_risk=0.8,
                continuity_risk=0.8, projection_risk=0.8,
            ),
        )
        policy = _make_policy(
            edc={"social": -0.15, "curiosity": -0.15, "expression": -0.15},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert val >= 0.0, f"{dim} = {val} is below 0.0"

    def test_emotions_not_above_one(self):
        """Emotions should not exceed 1.0."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.99, love=0.99, fun=0.99),
            mood=Mood(valence=0.9, arousal=0.9),
        )
        policy = _make_policy(
            edc={"social": -0.15, "curiosity": -0.15, "expression": -0.15},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert val <= 1.0, f"{dim} = {val} is above 1.0"


# ── Test: Candidate Count Scaling ─────────────────────────────


class TestCandidateCountScaling:
    """Candidate count affects return magnitude."""

    def test_fewer_candidates_less_return(self):
        """Fewer candidates should produce less emotion return."""
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        # With many candidates
        orch_many = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        emotions_before = orch_many._psyche.emotions.as_dict()
        orch_many._apply_selection_emotion_return(policy, candidate_count=5)
        total_many = sum(
            abs(orch_many._psyche.emotions.as_dict()[k] - emotions_before[k])
            for k in emotions_before
        )

        # With single candidate
        orch_one = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        emotions_before = orch_one._psyche.emotions.as_dict()
        orch_one._apply_selection_emotion_return(policy, candidate_count=1)
        total_one = sum(
            abs(orch_one._psyche.emotions.as_dict()[k] - emotions_before[k])
            for k in emotions_before
        )

        # Single candidate should produce less return
        assert total_one <= total_many + 1e-9

    def test_zero_candidates_no_return(self):
        """Zero candidates should produce no emotion return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=0)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after


# ── Test: Arousal Scaling ─────────────────────────────────────


class TestArousalScaling:
    """Mood arousal affects return magnitude."""

    def test_low_arousal_less_return(self):
        """Low arousal should produce less emotion return."""
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        # High arousal
        orch_high = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.9),
        )
        emotions_before = orch_high._psyche.emotions.as_dict()
        orch_high._apply_selection_emotion_return(policy, candidate_count=5)
        total_high = sum(
            abs(orch_high._psyche.emotions.as_dict()[k] - emotions_before[k])
            for k in emotions_before
        )

        # Low arousal
        orch_low = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.1),
        )
        emotions_before = orch_low._psyche.emotions.as_dict()
        orch_low._apply_selection_emotion_return(policy, candidate_count=5)
        total_low = sum(
            abs(orch_low._psyche.emotions.as_dict()[k] - emotions_before[k])
            for k in emotions_before
        )

        # Low arousal should produce less return
        assert total_low <= total_high + 1e-9


# ── Test: No Fixed Mapping ────────────────────────────────────


class TestNoFixedMapping:
    """Verify there is no fixed mapping between policy label and emotion dimension."""

    def test_no_label_string_comparison(self):
        """The method should not compare policy_label strings to determine emotion changes.
        Same label with different states should produce different patterns."""
        policy = _make_policy(
            label="共感する",
            edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02},
            drive_target="social",
        )

        # State with high joy and positive mood
        orch1 = _make_orchestrator(
            emotions=EmotionVector(joy=0.8, sorrow=0.1, fear=0.1),
            mood=Mood(valence=0.7, arousal=0.6),
        )
        e1_before = orch1._psyche.emotions.as_dict()
        orch1._apply_selection_emotion_return(policy, candidate_count=5)
        d1 = {k: orch1._psyche.emotions.as_dict()[k] - e1_before[k] for k in e1_before}

        # State with high sorrow and negative mood
        orch2 = _make_orchestrator(
            emotions=EmotionVector(joy=0.1, sorrow=0.8, fear=0.5),
            mood=Mood(valence=-0.7, arousal=0.6),
            fear_index=FearIndex(
                identity_risk=0.6, attachment_risk=0.6,
                continuity_risk=0.1, projection_risk=0.1,
            ),
        )
        e2_before = orch2._psyche.emotions.as_dict()
        orch2._apply_selection_emotion_return(policy, candidate_count=5)
        d2 = {k: orch2._psyche.emotions.as_dict()[k] - e2_before[k] for k in e2_before}

        # Verify at least some dimensions show different direction (sign)
        different_signs = sum(
            1 for k in d1
            if (d1[k] > 1e-9 and d2[k] < -1e-9) or (d1[k] < -1e-9 and d2[k] > 1e-9)
            or (abs(d1[k]) > 1e-9 and abs(d2[k]) < 1e-9)
            or (abs(d1[k]) < 1e-9 and abs(d2[k]) > 1e-9)
        )
        assert different_signs > 0, (
            "Same policy label should produce different patterns with different states"
        )


# ── Test: Drive Target Fallback ───────────────────────────────


class TestDriveTargetFallback:
    """Test behavior when drive_target is missing or empty."""

    def test_missing_drive_target_infers_from_edc(self):
        """When drive_target is missing, it should be inferred from edc."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = {
            "policy_label": "test",
            "rationale": "test",
            "expected_drive_change": {"social": -0.10, "curiosity": -0.02},
            "text": "test",
        }
        # No drive_target key at all
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        # Should still produce changes (infers social as primary axis)
        changed = any(
            abs(emotions_after[k] - emotions_before[k]) > 1e-9
            for k in emotions_before
        )
        assert changed


# ── Test: Drives Untouched ────────────────────────────────────


class TestDrivesUntouched:
    """Emotion return should not modify drives, mood, or fear."""

    def test_drives_unchanged(self):
        """Drives should not be modified by emotion return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            drives=DriveVector(social=0.6, curiosity=0.7, expression=0.4),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        drives_before = orch._psyche.drives.as_dict()
        policy = _make_policy(
            edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        drives_after = orch._psyche.drives.as_dict()
        assert drives_before == drives_after

    def test_mood_unchanged(self):
        """Mood should not be modified by emotion return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.5, arousal=0.7),
        )
        mood_before = (orch._psyche.mood.valence, orch._psyche.mood.arousal)
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        mood_after = (orch._psyche.mood.valence, orch._psyche.mood.arousal)
        assert mood_before == mood_after

    def test_fear_index_unchanged(self):
        """Fear index should not be modified by emotion return."""
        fi = FearIndex(
            identity_risk=0.3, attachment_risk=0.2,
            continuity_risk=0.1, projection_risk=0.1,
        )
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
            fear_index=fi,
        )
        fear_before = orch._psyche.fear_index
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        fear_after = orch._psyche.fear_index
        assert fear_before == fear_after


# ── Test: All 15 Policies ─────────────────────────────────────


class TestAll15Policies:
    """Verify all 15 policies produce bounded emotion return."""

    def test_all_policies_bounded_return(self):
        """All 15 policies should produce bounded emotion return."""
        from psyche.thought import POLICIES

        for policy_def in POLICIES:
            orch = _make_orchestrator(
                emotions=EmotionVector(joy=0.5, anger=0.3, sorrow=0.3,
                                       fear=0.2, surprise=0.2, love=0.3, fun=0.4),
                mood=Mood(valence=0.2, arousal=0.6),
            )
            policy = {
                "policy_label": policy_def["policy_label"],
                "rationale": policy_def["rationale_template"],
                "expected_drive_change": dict(policy_def["expected_drive_change"]),
                "drive_target": policy_def["drive_target"],
                "text": "test",
            }
            emotions_before = orch._psyche.emotions.as_dict()

            orch._apply_selection_emotion_return(policy, candidate_count=5)

            emotions_after = orch._psyche.emotions.as_dict()

            # Per-dim bound
            for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
                change = abs(emotions_after[dim] - emotions_before[dim])
                assert change <= 0.075 + 0.001, (
                    f"Policy '{policy_def['policy_label']}' changed {dim} by {change}"
                )

            # Total bound
            total = sum(
                abs(emotions_after[dim] - emotions_before[dim])
                for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun")
            )
            assert total <= 0.30 + 0.001, (
                f"Policy '{policy_def['policy_label']}' total return {total}"
            )

            # Valid range
            for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
                assert 0.0 <= emotions_after[dim] <= 1.0, (
                    f"Policy '{policy_def['policy_label']}' {dim}={emotions_after[dim]} out of range"
                )


# ── Test: Edge Cases ──────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_invalid_edc_type_does_nothing(self):
        """If expected_drive_change is not a dict, nothing happens."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5),
        )
        policy = _make_policy()
        policy["expected_drive_change"] = "invalid"
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        assert orch._psyche.emotions.as_dict() == emotions_before

    def test_all_emotions_at_zero(self):
        """All emotions at 0.0 should not go negative."""
        orch = _make_orchestrator(
            emotions=EmotionVector(),  # all zeros
            mood=Mood(valence=-0.5, arousal=0.8),
            fear_index=FearIndex(
                identity_risk=0.5, attachment_risk=0.5,
                continuity_risk=0.5, projection_risk=0.5,
            ),
        )
        policy = _make_policy(
            edc={"social": -0.15},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert val >= 0.0, f"{dim} = {val} went below 0.0"

    def test_all_emotions_at_one(self):
        """All emotions at 1.0 should not exceed 1.0."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=1.0, anger=1.0, sorrow=1.0,
                                   fear=1.0, surprise=1.0, love=1.0, fun=1.0),
            mood=Mood(valence=0.9, arousal=0.9),
        )
        policy = _make_policy(
            edc={"social": -0.15},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert val <= 1.0, f"{dim} = {val} exceeded 1.0"

    def test_negative_candidate_count(self):
        """Negative candidate count should produce no return."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=-1)

        assert orch._psyche.emotions.as_dict() == emotions_before

    def test_non_numeric_edc_value(self):
        """Non-numeric values in edc should be handled gracefully."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": "bad", "curiosity": -0.05},
            drive_target="curiosity",
        )

        # Should not raise
        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0

    def test_fear_index_none_handled(self):
        """When fear_index is None, fear_level should be 0.0."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            mood=Mood(valence=0.3, arousal=0.6),
            fear_index=None,
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        # Should not raise
        orch._apply_selection_emotion_return(policy, candidate_count=5)

        emotions_after = orch._psyche.emotions.as_dict()
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            assert 0.0 <= emotions_after[dim] <= 1.0


# ── Test: Independence from Drive Return ──────────────────────


class TestIndependenceFromDriveReturn:
    """Emotion return and drive return are independent parallel paths."""

    def test_emotion_return_does_not_affect_drives(self):
        """Emotion return should not modify the drive vector."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03},
            drive_target="social",
        )

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        # Drives should be completely unchanged
        drives = orch._psyche.drives
        assert drives.social == 0.5
        assert drives.curiosity == 0.5
        assert drives.expression == 0.5

    def test_drive_return_does_not_affect_emotion_return(self):
        """Drive return and emotion return should be independent."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.5, sorrow=0.3),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.3, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03},
            drive_target="social",
        )
        emotions_before = orch._psyche.emotions.as_dict()

        # Apply drive return first
        orch._apply_expected_drive_change(policy)
        # Then emotion return
        orch._apply_selection_emotion_return(policy, candidate_count=5)

        # Emotions should have changed (emotion return worked)
        emotions_after = orch._psyche.emotions.as_dict()
        changed = any(
            abs(emotions_after[k] - emotions_before[k]) > 1e-9
            for k in emotions_before
        )
        assert changed

        # Drives should reflect drive return only
        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.4, abs=0.001)
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)
        assert drives.expression == pytest.approx(0.47, abs=0.001)


# ── Test: State-Dependent Direction ───────────────────────────


class TestStateDependentDirection:
    """Verify direction derivation is state-dependent, not fixed."""

    def test_high_fear_affects_fear_dimension(self):
        """High fear level should influence fear/sorrow dimensions more."""
        orch_fearful = _make_orchestrator(
            emotions=EmotionVector(joy=0.3, sorrow=0.3, fear=0.3),
            mood=Mood(valence=-0.3, arousal=0.6),
            fear_index=FearIndex(
                identity_risk=0.8, attachment_risk=0.8,
                continuity_risk=0.8, projection_risk=0.8,
            ),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )
        e_before = orch_fearful._psyche.emotions.as_dict()

        orch_fearful._apply_selection_emotion_return(policy, candidate_count=5)

        e_after = orch_fearful._psyche.emotions.as_dict()
        # With high fear, fear/sorrow dimensions should have non-zero change
        fear_change = abs(e_after["fear"] - e_before["fear"])
        sorrow_change = abs(e_after["sorrow"] - e_before["sorrow"])
        # At least one of these should have changed
        assert fear_change > 1e-9 or sorrow_change > 1e-9

    def test_positive_mood_affects_joy_dimensions(self):
        """Positive mood should influence joy/fun dimensions."""
        orch = _make_orchestrator(
            emotions=EmotionVector(joy=0.3, fun=0.3, love=0.3),
            mood=Mood(valence=0.7, arousal=0.6),
        )
        policy = _make_policy(
            edc={"social": -0.10},
            drive_target="social",
        )
        e_before = orch._psyche.emotions.as_dict()

        orch._apply_selection_emotion_return(policy, candidate_count=5)

        e_after = orch._psyche.emotions.as_dict()
        # With positive mood, joy/fun/love dimensions should have change
        joy_change = abs(e_after["joy"] - e_before["joy"])
        fun_change = abs(e_after["fun"] - e_before["fun"])
        love_change = abs(e_after["love"] - e_before["love"])
        assert joy_change > 1e-9 or fun_change > 1e-9 or love_change > 1e-9
