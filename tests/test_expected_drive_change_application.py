"""
Tests for expected_drive_change application in orchestrator.

Verifies that after policy selection, the expected_drive_change
values are applied to drives with all 5 safety valves.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from psyche.state import PsycheState, DriveVector, Percept, EmotionVector, Mood
from psyche.orchestrator import PsycheOrchestrator


# ── Helpers ────────────────────────────────────────────────────


def _make_orchestrator() -> PsycheOrchestrator:
    """Create a minimal orchestrator for testing."""
    orch = PsycheOrchestrator.__new__(PsycheOrchestrator)
    # Initialize minimal required state
    orch._psyche = PsycheState(
        drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
    )
    return orch


def _make_policy(
    label: str = "共感する",
    edc: dict | None = None,
) -> dict:
    """Create a policy dict for testing."""
    if edc is None:
        edc = {"social": -0.08, "curiosity": -0.02, "expression": -0.02}
    return {
        "policy_label": label,
        "rationale": "test",
        "expected_drive_change": edc,
        "text": "test text",
    }


# ── Test: Basic Application ──────────────────────────────────


class TestBasicApplication:
    """Test basic drive return application."""

    def test_drives_decrease_after_application(self):
        """Drives should decrease when expected_drive_change is applied."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.42, abs=0.001)
        assert drives.curiosity == pytest.approx(0.48, abs=0.001)
        assert drives.expression == pytest.approx(0.48, abs=0.001)

    def test_all_15_policies_apply_correctly(self):
        """All 15 policies should apply their declared values."""
        from psyche.thought import POLICIES

        for policy_def in POLICIES:
            orch = _make_orchestrator()
            policy = {
                "policy_label": policy_def["policy_label"],
                "rationale": policy_def["rationale_template"],
                "expected_drive_change": dict(policy_def["expected_drive_change"]),
                "text": "test",
            }

            orch._apply_expected_drive_change(policy)

            drives = orch._psyche.drives
            edc = policy_def["expected_drive_change"]
            for axis in ("social", "curiosity", "expression"):
                expected_delta = edc.get(axis, 0.0)
                if expected_delta > 0.0:
                    # Positive values are blocked by safety valve
                    assert getattr(drives, axis) == 0.5
                elif abs(expected_delta) > 0.15:
                    # Clamped to max_change
                    assert getattr(drives, axis) == pytest.approx(0.5 - 0.15, abs=0.001)
                else:
                    assert getattr(drives, axis) == pytest.approx(0.5 + expected_delta, abs=0.001)

    def test_drives_change_accumulates_across_ticks(self):
        """Drive changes should accumulate over multiple applications."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02})

        # Apply three times (simulating three separate policy selections)
        orch._apply_expected_drive_change(policy)
        orch._apply_expected_drive_change(policy)
        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.5 - 0.08 * 3, abs=0.001)
        assert drives.curiosity == pytest.approx(0.5 - 0.02 * 3, abs=0.001)
        assert drives.expression == pytest.approx(0.5 - 0.02 * 3, abs=0.001)

    def test_no_edc_key_does_nothing(self):
        """If policy has no expected_drive_change, drives are unchanged."""
        orch = _make_orchestrator()
        policy = {"policy_label": "test", "rationale": "test", "text": "test"}

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5
        assert drives.curiosity == 0.5
        assert drives.expression == 0.5

    def test_empty_edc_does_nothing(self):
        """If expected_drive_change is empty dict, drives are unchanged."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5
        assert drives.curiosity == 0.5
        assert drives.expression == 0.5

    def test_none_edc_does_nothing(self):
        """If expected_drive_change is None, drives are unchanged."""
        orch = _make_orchestrator()
        policy = _make_policy()
        policy["expected_drive_change"] = None

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5
        assert drives.curiosity == 0.5
        assert drives.expression == 0.5


# ── Test: Safety Valve 1 - Per-Axis Clamp ────────────────────


class TestSafetyValveAxisClamp:
    """Safety valve 1: per-axis clamp to max_bias_strength."""

    def test_large_negative_is_clamped(self):
        """Delta exceeding max_change (0.15) should be clamped."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.50, "curiosity": 0.0, "expression": 0.0})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        # -0.50 should be clamped to -0.15
        assert drives.social == pytest.approx(0.35, abs=0.001)

    def test_exactly_max_change_is_not_clamped(self):
        """Delta exactly at max_change (0.15) should not be clamped further."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.15, "curiosity": 0.0, "expression": 0.0})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.35, abs=0.001)

    def test_each_axis_clamped_independently(self):
        """Each axis should be clamped independently."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.30, "curiosity": -0.05, "expression": -0.25})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.35, abs=0.001)  # clamped from -0.30
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)  # not clamped
        assert drives.expression == pytest.approx(0.35, abs=0.001)  # clamped from -0.25


# ── Test: Safety Valve 2 - Drive Range Clamp ─────────────────


class TestSafetyValveDriveRange:
    """Safety valve 2: drives stay within 0.0-1.0."""

    def test_drive_does_not_go_below_zero(self):
        """Drive should not drop below 0.0."""
        orch = _make_orchestrator()
        # Set drives near zero
        orch._psyche = PsycheState(
            drives=DriveVector(social=0.05, curiosity=0.05, expression=0.05),
        )
        policy = _make_policy(edc={"social": -0.15, "curiosity": -0.15, "expression": -0.15})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.0
        assert drives.curiosity == 0.0
        assert drives.expression == 0.0

    def test_drive_at_zero_stays_zero(self):
        """Drive already at 0.0 stays at 0.0."""
        orch = _make_orchestrator()
        orch._psyche = PsycheState(
            drives=DriveVector(social=0.0, curiosity=0.0, expression=0.0),
        )
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.10, "expression": -0.10})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.0
        assert drives.curiosity == 0.0
        assert drives.expression == 0.0


# ── Test: Safety Valve 3 - Single Application ────────────────


class TestSafetyValveSingleApplication:
    """Safety valve 3: structural guarantee of single application per selection."""

    def test_method_called_once_per_select_policy_dict(self):
        """_apply_expected_drive_change is called exactly once per select_policy_dict."""
        # This is structurally guaranteed by the code: the call site is inside
        # select_policy_dict, which calls it once. We test that the method
        # applies changes exactly once (not multiplicatively).
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03})

        # Single application
        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.40, abs=0.001)
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)
        assert drives.expression == pytest.approx(0.47, abs=0.001)


# ── Test: Safety Valve 4 - Declaration Immutability ──────────


class TestSafetyValveDeclarationImmutability:
    """Safety valve 4: original declaration values are not modified."""

    def test_original_policy_edc_unchanged(self):
        """The policy dict's expected_drive_change should not be modified."""
        orch = _make_orchestrator()
        edc_original = {"social": -0.08, "curiosity": -0.02, "expression": -0.02}
        policy = _make_policy(edc=edc_original)

        orch._apply_expected_drive_change(policy)

        # Original values should be preserved
        assert policy["expected_drive_change"]["social"] == -0.08
        assert policy["expected_drive_change"]["curiosity"] == -0.02
        assert policy["expected_drive_change"]["expression"] == -0.02

    def test_policy_definition_constants_unchanged(self):
        """POLICIES list constants should not be modified."""
        from psyche.thought import POLICIES

        original_values = [
            dict(p["expected_drive_change"]) for p in POLICIES
        ]

        orch = _make_orchestrator()
        for policy_def in POLICIES:
            policy = {
                "policy_label": policy_def["policy_label"],
                "rationale": policy_def["rationale_template"],
                "expected_drive_change": policy_def["expected_drive_change"],
                "text": "test",
            }
            orch._apply_expected_drive_change(policy)

        # Verify all constants are unchanged
        for i, policy_def in enumerate(POLICIES):
            assert policy_def["expected_drive_change"] == original_values[i]


# ── Test: Safety Valve 5 - Positive Return Block ─────────────


class TestSafetyValvePositiveBlock:
    """Safety valve 5: positive drive changes are not applied."""

    def test_positive_values_blocked(self):
        """Positive expected_drive_change values should not be applied."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": 0.10, "curiosity": -0.05, "expression": 0.05})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5  # positive blocked
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)  # negative applied
        assert drives.expression == 0.5  # positive blocked

    def test_zero_value_does_nothing(self):
        """Zero expected_drive_change should not modify drives."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": 0.0, "curiosity": 0.0, "expression": 0.0})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5
        assert drives.curiosity == 0.5
        assert drives.expression == 0.5

    def test_all_existing_policies_are_nonpositive(self):
        """Verify all 15 policies have non-positive expected_drive_change values."""
        from psyche.thought import POLICIES

        for policy_def in POLICIES:
            edc = policy_def["expected_drive_change"]
            for axis, value in edc.items():
                assert value <= 0.0, (
                    f"Policy '{policy_def['policy_label']}' has positive "
                    f"expected_drive_change for {axis}: {value}"
                )


# ── Test: Self-Suppression Structure ─────────────────────────


class TestSelfSuppression:
    """Test the self-suppressive structure described in the design."""

    def test_repeated_selection_decreases_drive(self):
        """Repeated selection of the same policy should decrease its target drive."""
        orch = _make_orchestrator()
        # Policy targets social drive
        policy = _make_policy(
            label="共感する",
            edc={"social": -0.08, "curiosity": -0.02, "expression": -0.02},
        )

        initial_social = orch._psyche.drives.social
        for _ in range(5):
            orch._apply_expected_drive_change(policy)

        final_social = orch._psyche.drives.social
        # Social drive should have decreased
        assert final_social < initial_social
        # Specifically: 0.5 - 5*0.08 = 0.1
        assert final_social == pytest.approx(0.1, abs=0.001)

    def test_drive_floor_prevents_negative(self):
        """Repeated application should be bounded by floor clamp."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.10, "expression": -0.10})

        # Apply many times
        for _ in range(20):
            orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social >= 0.0
        assert drives.curiosity >= 0.0
        assert drives.expression >= 0.0


# ── Test: Emotions/Fear/Mood Untouched ───────────────────────


class TestNoSideEffects:
    """Verify no side effects on other state components."""

    def test_emotions_unchanged(self):
        """Emotions should not be modified by drive return."""
        orch = _make_orchestrator()
        orch._psyche = PsycheState(
            emotions=EmotionVector(joy=0.7, sorrow=0.3, anger=0.1),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        emotions_before = orch._psyche.emotions.as_dict()
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03})

        orch._apply_expected_drive_change(policy)

        emotions_after = orch._psyche.emotions.as_dict()
        assert emotions_before == emotions_after

    def test_mood_unchanged(self):
        """Mood should not be modified by drive return."""
        orch = _make_orchestrator()
        orch._psyche = PsycheState(
            mood=Mood(valence=0.3, arousal=0.6),
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        )
        mood_before = (orch._psyche.mood.valence, orch._psyche.mood.arousal)
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03})

        orch._apply_expected_drive_change(policy)

        mood_after = (orch._psyche.mood.valence, orch._psyche.mood.arousal)
        assert mood_before == mood_after

    def test_fear_index_unchanged(self):
        """Fear index should not be modified by drive return."""
        orch = _make_orchestrator()
        fear_before = orch._psyche.fear_index
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.05, "expression": -0.03})

        orch._apply_expected_drive_change(policy)

        fear_after = orch._psyche.fear_index
        assert fear_before == fear_after


# ── Test: Edge Cases ─────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_invalid_edc_type_does_nothing(self):
        """If expected_drive_change is not a dict, nothing happens."""
        orch = _make_orchestrator()
        policy = _make_policy()
        policy["expected_drive_change"] = "invalid"

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5

    def test_non_numeric_axis_value_skipped(self):
        """Non-numeric axis values should be skipped."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": "bad", "curiosity": -0.05, "expression": -0.02})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == 0.5  # skipped
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)
        assert drives.expression == pytest.approx(0.48, abs=0.001)

    def test_extra_axes_ignored(self):
        """Extra axes not in (social, curiosity, expression) should be ignored."""
        orch = _make_orchestrator()
        policy = _make_policy(
            edc={
                "social": -0.05,
                "curiosity": -0.05,
                "expression": -0.05,
                "unknown_axis": -0.99,
            }
        )

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.45, abs=0.001)
        assert drives.curiosity == pytest.approx(0.45, abs=0.001)
        assert drives.expression == pytest.approx(0.45, abs=0.001)

    def test_partial_axes(self):
        """Only specified axes should be changed."""
        orch = _make_orchestrator()
        policy = _make_policy(edc={"social": -0.10})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.40, abs=0.001)
        assert drives.curiosity == 0.5  # unchanged
        assert drives.expression == 0.5  # unchanged

    def test_drives_at_high_values(self):
        """Application on high drive values should work correctly."""
        orch = _make_orchestrator()
        orch._psyche = PsycheState(
            drives=DriveVector(social=0.95, curiosity=0.99, expression=1.0),
        )
        policy = _make_policy(edc={"social": -0.10, "curiosity": -0.10, "expression": -0.10})

        orch._apply_expected_drive_change(policy)

        drives = orch._psyche.drives
        assert drives.social == pytest.approx(0.85, abs=0.001)
        assert drives.curiosity == pytest.approx(0.89, abs=0.001)
        assert drives.expression == pytest.approx(0.90, abs=0.001)


# ── Test: Max Change Bound ───────────────────────────────────


class TestMaxChangeBound:
    """Verify max_change is consistent with ValueOrientationConfig.max_bias_strength."""

    def test_max_change_equals_vo_max_bias_strength(self):
        """The max_change value should equal ValueOrientationConfig.max_bias_strength."""
        from psyche.value_orientation import ValueOrientationConfig

        config = ValueOrientationConfig()
        # The orchestrator uses 0.15 which should match the default
        assert config.max_bias_strength == 0.15

    def test_no_single_application_exceeds_bound(self):
        """No single application should change any axis by more than 0.15."""
        from psyche.thought import POLICIES

        for policy_def in POLICIES:
            orch = _make_orchestrator()
            drives_before = orch._psyche.drives.as_dict()

            policy = {
                "policy_label": policy_def["policy_label"],
                "rationale": policy_def["rationale_template"],
                "expected_drive_change": dict(policy_def["expected_drive_change"]),
                "text": "test",
            }
            orch._apply_expected_drive_change(policy)

            drives_after = orch._psyche.drives.as_dict()
            for axis in ("social", "curiosity", "expression"):
                change = abs(drives_after[axis] - drives_before[axis])
                assert change <= 0.15 + 0.001, (
                    f"Policy '{policy_def['policy_label']}' changed {axis} by {change}"
                )
