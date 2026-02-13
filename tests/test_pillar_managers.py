"""Tests for the four pillar manager modules:

- psyche/attachment_manager.py
- psyche/continuity_manager.py
- psyche/identity_manager.py
- psyche/projection_manager.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from psyche.attachment_manager import (
    apply_daily_decay,
    calc_attachment_risk,
    get_top_partners,
    update_bond,
)
from psyche.continuity_manager import (
    audit_memory_health,
    calc_continuity_risk,
    compress_and_cleanup,
    maybe_save,
)
from psyche.identity_manager import (
    apply_identity_change,
    calc_identity_risk,
    calc_identity_risk_from_values,
    propose_identity_change,
)
from psyche.pillars import (
    AttachmentState,
    ContinuityState,
    IdentityState,
    ProjectionState,
)
from psyche.projection_manager import (
    add_goal,
    calc_projection_risk,
    remove_goal,
    reset,
    update_goal_progress,
)


# ========================================================================
# attachment_manager.py
# ========================================================================


# -- update_bond: positive events ----------------------------------------


class TestUpdateBondPositive:
    def test_positive_new_partner(self) -> None:
        state = AttachmentState()
        result = update_bond(state, "alice", "positive", 0.5)
        assert result.bonds["alice"] == pytest.approx(0.05)

    def test_positive_existing_partner(self) -> None:
        state = AttachmentState(bonds={"alice": 0.2})
        result = update_bond(state, "alice", "positive", 1.0)
        # 0.2 + 1.0 * 0.1 = 0.3
        assert result.bonds["alice"] == pytest.approx(0.3)

    def test_positive_default_intensity(self) -> None:
        state = AttachmentState()
        result = update_bond(state, "bob")
        # default event_type="positive", intensity=0.5 => 0.0 + 0.5*0.1 = 0.05
        assert result.bonds["bob"] == pytest.approx(0.05)

    def test_positive_high_intensity(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = update_bond(state, "alice", "positive", 1.0)
        # 0.5 + 1.0 * 0.1 = 0.6
        assert result.bonds["alice"] == pytest.approx(0.6)

    def test_positive_clamped_at_1(self) -> None:
        state = AttachmentState(bonds={"alice": 0.98})
        result = update_bond(state, "alice", "positive", 1.0)
        # 0.98 + 0.1 = 1.08 => clamp to 1.0
        assert result.bonds["alice"] == pytest.approx(1.0)

    def test_positive_does_not_affect_other_bonds(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5, "bob": 0.3})
        result = update_bond(state, "alice", "positive", 1.0)
        assert result.bonds["bob"] == pytest.approx(0.3)


# -- update_bond: negative events ----------------------------------------


class TestUpdateBondNegative:
    def test_negative_event(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = update_bond(state, "alice", "negative", 1.0)
        # 0.5 - 1.0 * 0.15 = 0.35
        assert result.bonds["alice"] == pytest.approx(0.35)

    def test_negative_default_intensity(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = update_bond(state, "alice", "negative")
        # 0.5 - 0.5 * 0.15 = 0.5 - 0.075 = 0.425
        assert result.bonds["alice"] == pytest.approx(0.425)

    def test_negative_clamped_at_0(self) -> None:
        state = AttachmentState(bonds={"alice": 0.05})
        result = update_bond(state, "alice", "negative", 1.0)
        # 0.05 - 0.15 = -0.10 => clamp to 0.0
        assert result.bonds["alice"] == pytest.approx(0.0)

    def test_negative_on_new_partner(self) -> None:
        state = AttachmentState()
        result = update_bond(state, "alice", "negative", 0.5)
        # 0.0 - 0.5 * 0.15 = -0.075 => clamp to 0.0
        assert result.bonds["alice"] == pytest.approx(0.0)


# -- update_bond: immutability -------------------------------------------


class TestUpdateBondImmutability:
    def test_returns_new_state(self) -> None:
        state = AttachmentState()
        result = update_bond(state, "alice")
        assert result is not state

    def test_original_bonds_unchanged(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        _ = update_bond(state, "alice", "positive", 1.0)
        assert state.bonds["alice"] == pytest.approx(0.5)

    def test_risk_updated_on_result(self) -> None:
        state = AttachmentState()
        result = update_bond(state, "alice", "positive", 1.0)
        # bond = 0.1, which is < 0.3 => risk = 0.6
        assert result.risk == pytest.approx(0.6)


# -- apply_daily_decay ---------------------------------------------------


class TestApplyDailyDecay:
    def test_single_day_decay(self) -> None:
        state = AttachmentState(bonds={"alice": 1.0})
        result = apply_daily_decay(state, 1.0)
        assert result.bonds["alice"] == pytest.approx(0.98)

    def test_default_days_elapsed(self) -> None:
        state = AttachmentState(bonds={"alice": 1.0})
        result = apply_daily_decay(state)
        assert result.bonds["alice"] == pytest.approx(0.98)

    def test_multi_day_decay(self) -> None:
        state = AttachmentState(bonds={"alice": 1.0})
        result = apply_daily_decay(state, 10.0)
        expected = 0.98 ** 10
        assert result.bonds["alice"] == pytest.approx(expected)

    def test_zero_day_no_decay(self) -> None:
        state = AttachmentState(bonds={"alice": 0.7})
        result = apply_daily_decay(state, 0.0)
        assert result.bonds["alice"] == pytest.approx(0.7)

    def test_decay_all_bonds(self) -> None:
        state = AttachmentState(bonds={"alice": 0.8, "bob": 0.4})
        result = apply_daily_decay(state, 1.0)
        assert result.bonds["alice"] == pytest.approx(0.8 * 0.98)
        assert result.bonds["bob"] == pytest.approx(0.4 * 0.98)

    def test_decay_returns_new_state(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = apply_daily_decay(state)
        assert result is not state

    def test_decay_original_unchanged(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        _ = apply_daily_decay(state)
        assert state.bonds["alice"] == pytest.approx(0.5)

    def test_decay_risk_updated(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = apply_daily_decay(state, 1.0)
        # 0.5 * 0.98 = 0.49, which is < 0.5 => risk = 0.3
        assert result.risk == pytest.approx(0.3)

    def test_heavy_decay_clamps_near_zero(self) -> None:
        state = AttachmentState(bonds={"alice": 0.01})
        result = apply_daily_decay(state, 1000.0)
        # 0.01 * 0.98^1000 is effectively 0
        assert result.bonds["alice"] == pytest.approx(0.0, abs=1e-9)


# -- get_top_partners ----------------------------------------------------


class TestGetTopPartners:
    def test_empty_bonds(self) -> None:
        state = AttachmentState()
        assert get_top_partners(state) == []

    def test_single_partner(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = get_top_partners(state)
        assert result == [("alice", 0.5)]

    def test_sorted_descending(self) -> None:
        state = AttachmentState(bonds={"alice": 0.3, "bob": 0.8, "carol": 0.5})
        result = get_top_partners(state)
        assert result == [("bob", 0.8), ("carol", 0.5), ("alice", 0.3)]

    def test_default_n_is_3(self) -> None:
        state = AttachmentState(bonds={
            "a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5,
        })
        result = get_top_partners(state)
        assert len(result) == 3
        assert result[0] == ("e", 0.5)

    def test_custom_n(self) -> None:
        state = AttachmentState(bonds={
            "a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4,
        })
        result = get_top_partners(state, n=2)
        assert len(result) == 2
        assert result[0] == ("d", 0.4)
        assert result[1] == ("c", 0.3)

    def test_n_larger_than_bonds(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        result = get_top_partners(state, n=10)
        assert len(result) == 1


# -- calc_attachment_risk ------------------------------------------------


class TestCalcAttachmentRisk:
    def test_no_bonds(self) -> None:
        state = AttachmentState()
        assert calc_attachment_risk(state) == pytest.approx(0.7)

    def test_weak_bond_below_03(self) -> None:
        state = AttachmentState(bonds={"alice": 0.2})
        assert calc_attachment_risk(state) == pytest.approx(0.6)

    def test_bond_exactly_03(self) -> None:
        state = AttachmentState(bonds={"alice": 0.3})
        # max_bond = 0.3, not < 0.3, goes to next check: < 0.5 => 0.3
        assert calc_attachment_risk(state) == pytest.approx(0.3)

    def test_moderate_bond_below_05(self) -> None:
        state = AttachmentState(bonds={"alice": 0.4})
        assert calc_attachment_risk(state) == pytest.approx(0.3)

    def test_bond_exactly_05(self) -> None:
        state = AttachmentState(bonds={"alice": 0.5})
        # max_bond = 0.5, not < 0.5 => max(0.0, 0.2 - 0.5*0.1) = max(0, 0.15) = 0.15
        assert calc_attachment_risk(state) == pytest.approx(0.15)

    def test_strong_bond(self) -> None:
        state = AttachmentState(bonds={"alice": 0.8})
        # max(0.0, 0.2 - 0.8 * 0.1) = max(0, 0.12) = 0.12
        assert calc_attachment_risk(state) == pytest.approx(0.12)

    def test_max_bond(self) -> None:
        state = AttachmentState(bonds={"alice": 1.0})
        # max(0.0, 0.2 - 1.0 * 0.1) = max(0, 0.1) = 0.1
        assert calc_attachment_risk(state) == pytest.approx(0.1)

    def test_very_strong_bond_formula_floor(self) -> None:
        # max(0.0, 0.2 - 2.0 * 0.1) = max(0, 0.0) = 0.0
        # but bonds are clamped [0,1] by pydantic; test with max bond=1.0
        state = AttachmentState(bonds={"alice": 1.0})
        # 0.2 - 0.1 = 0.1 >= 0 => 0.1
        assert calc_attachment_risk(state) == pytest.approx(0.1)

    def test_risk_uses_max_bond(self) -> None:
        state = AttachmentState(bonds={"alice": 0.1, "bob": 0.8})
        # max_bond = 0.8 => max(0, 0.2 - 0.08) = 0.12
        assert calc_attachment_risk(state) == pytest.approx(0.12)


# ========================================================================
# continuity_manager.py
# ========================================================================


# -- maybe_save ----------------------------------------------------------


class TestMaybeSave:
    def _make_mock(self) -> MagicMock:
        return MagicMock()

    def test_importance_3_saves(self) -> None:
        state = ContinuityState()
        assert maybe_save("event", "resp", state, self._make_mock(), importance=3)

    def test_importance_above_3_saves(self) -> None:
        state = ContinuityState()
        assert maybe_save("event", "resp", state, self._make_mock(), importance=5)

    def test_importance_below_3_no_save(self) -> None:
        state = ContinuityState()
        assert not maybe_save("event", "resp", state, self._make_mock(), importance=2)

    def test_importance_1_no_save(self) -> None:
        state = ContinuityState()
        assert not maybe_save("event", "resp", state, self._make_mock(), importance=1)

    def test_attachment_event_overrides_low_importance(self) -> None:
        state = ContinuityState()
        result = maybe_save(
            "event", "resp", state, self._make_mock(),
            importance=1, is_attachment_event=True,
        )
        assert result is True

    def test_attachment_event_with_high_importance(self) -> None:
        state = ContinuityState()
        result = maybe_save(
            "event", "resp", state, self._make_mock(),
            importance=5, is_attachment_event=True,
        )
        assert result is True

    def test_no_attachment_low_importance(self) -> None:
        state = ContinuityState()
        result = maybe_save(
            "event", "resp", state, self._make_mock(),
            importance=2, is_attachment_event=False,
        )
        assert result is False

    def test_default_importance_is_3(self) -> None:
        state = ContinuityState()
        # default importance=3 => should save
        assert maybe_save("event", "resp", state, self._make_mock())

    def test_default_is_attachment_event_false(self) -> None:
        state = ContinuityState()
        # importance=2, default is_attachment_event=False => no save
        assert not maybe_save("event", "resp", state, self._make_mock(), importance=2)


# -- compress_and_cleanup ------------------------------------------------


class TestCompressAndCleanup:
    def test_stub_returns_zero(self) -> None:
        mm = MagicMock()
        assert compress_and_cleanup(mm) == 0

    def test_stub_with_custom_max_age(self) -> None:
        mm = MagicMock()
        assert compress_and_cleanup(mm, max_age_days=30) == 0

    def test_stub_with_default_max_age(self) -> None:
        mm = MagicMock()
        assert compress_and_cleanup(mm, max_age_days=90) == 0


# -- calc_continuity_risk ------------------------------------------------


class TestCalcContinuityRisk:
    def test_zero_memories(self) -> None:
        assert calc_continuity_risk(0, 0) == pytest.approx(0.6)

    def test_few_memories_below_5(self) -> None:
        assert calc_continuity_risk(3, 0) == pytest.approx(0.6)

    def test_exactly_5_memories(self) -> None:
        # 5 is not < 5 => count_risk = 0.3
        assert calc_continuity_risk(5, 0) == pytest.approx(0.3)

    def test_memories_between_5_and_20(self) -> None:
        assert calc_continuity_risk(10, 0) == pytest.approx(0.3)

    def test_exactly_19_memories(self) -> None:
        assert calc_continuity_risk(19, 0) == pytest.approx(0.3)

    def test_exactly_20_memories(self) -> None:
        # 20 is not < 20 => count_risk = 0.1
        assert calc_continuity_risk(20, 0) == pytest.approx(0.1)

    def test_many_memories(self) -> None:
        assert calc_continuity_risk(100, 0) == pytest.approx(0.1)

    def test_compression_adds_risk(self) -> None:
        # count_risk = 0.1, compression_risk = 2 * 0.1 = 0.2 => total = 0.3
        assert calc_continuity_risk(50, 2) == pytest.approx(0.3)

    def test_compression_risk_capped_at_04(self) -> None:
        # 10 * 0.1 = 1.0, but min(1.0, 0.4) = 0.4
        assert calc_continuity_risk(50, 10) == pytest.approx(0.5)  # 0.1 + 0.4

    def test_total_risk_capped_at_1(self) -> None:
        # count_risk = 0.6, compression_risk = min(10*0.1, 0.4) = 0.4
        # total = 1.0, capped at 1.0
        assert calc_continuity_risk(0, 10) == pytest.approx(1.0)

    def test_default_args(self) -> None:
        # defaults: memory_count=0, recent_compressions=0
        assert calc_continuity_risk() == pytest.approx(0.6)

    def test_single_compression(self) -> None:
        # count_risk=0.1, compression_risk=0.1 => 0.2
        assert calc_continuity_risk(50, 1) == pytest.approx(0.2)

    def test_four_compressions(self) -> None:
        # count_risk=0.1, compression_risk=0.4 => 0.5
        assert calc_continuity_risk(50, 4) == pytest.approx(0.5)


# -- audit_memory_health -------------------------------------------------


class TestAuditMemoryHealth:
    def test_healthy_state(self) -> None:
        state = ContinuityState(
            memory_count=50,
            oldest_memory_age_days=30,
            compression_events=0,
            risk=0.1,
        )
        result = audit_memory_health(state)
        assert result["memory_count"] == 50
        assert result["oldest_memory_age_days"] == 30
        assert result["compression_events"] == 0
        assert result["risk"] == pytest.approx(0.1)
        assert result["status"] == "healthy"

    def test_warning_state(self) -> None:
        state = ContinuityState(
            memory_count=10,
            risk=0.4,
        )
        result = audit_memory_health(state)
        assert result["status"] == "warning"

    def test_critical_state(self) -> None:
        state = ContinuityState(
            memory_count=2,
            risk=0.7,
        )
        result = audit_memory_health(state)
        assert result["status"] == "critical"

    def test_boundary_healthy_warning(self) -> None:
        # risk < 0.3 => healthy
        state = ContinuityState(risk=0.29)
        assert audit_memory_health(state)["status"] == "healthy"

    def test_boundary_at_03_is_warning(self) -> None:
        # risk = 0.3, not < 0.3 => warning
        state = ContinuityState(risk=0.3)
        assert audit_memory_health(state)["status"] == "warning"

    def test_boundary_warning_critical(self) -> None:
        # risk < 0.6 => warning
        state = ContinuityState(risk=0.59)
        assert audit_memory_health(state)["status"] == "warning"

    def test_boundary_at_06_is_critical(self) -> None:
        # risk = 0.6, not < 0.6 => critical
        state = ContinuityState(risk=0.6)
        assert audit_memory_health(state)["status"] == "critical"

    def test_default_state_fields(self) -> None:
        state = ContinuityState()
        result = audit_memory_health(state)
        assert result["memory_count"] == 0
        assert result["oldest_memory_age_days"] == 0
        assert result["compression_events"] == 0
        assert result["risk"] == pytest.approx(0.0)
        assert result["status"] == "healthy"


# ========================================================================
# identity_manager.py
# ========================================================================


# -- propose_identity_change ---------------------------------------------


class TestProposeIdentityChange:
    def test_core_trait_requires_confirmation(self) -> None:
        state = IdentityState(core_traits=["curiosity", "empathy"])
        change = {"trait": "curiosity", "new_value": "indifference"}
        result = propose_identity_change(state, change)
        assert result["requires_confirmation"] is True
        assert result["change"] is change
        assert "core trait" in result["reason"]

    def test_non_core_trait_no_confirmation(self) -> None:
        state = IdentityState(core_traits=["curiosity", "empathy"])
        change = {"trait": "humor", "new_value": "sarcasm"}
        result = propose_identity_change(state, change)
        assert result["requires_confirmation"] is False
        assert "no conflict" in result["reason"]

    def test_empty_core_traits_no_conflict(self) -> None:
        state = IdentityState()
        change = {"trait": "curiosity", "new_value": "strong"}
        result = propose_identity_change(state, change)
        assert result["requires_confirmation"] is False

    def test_change_dict_preserved(self) -> None:
        state = IdentityState(core_traits=["empathy"])
        change = {"trait": "empathy", "new_value": "detachment", "extra": "data"}
        result = propose_identity_change(state, change)
        assert result["change"] == change
        assert result["change"]["extra"] == "data"

    def test_empty_trait_key(self) -> None:
        state = IdentityState(core_traits=["curiosity"])
        change = {"new_value": "something"}  # no "trait" key
        result = propose_identity_change(state, change)
        # trait = "" which is not in core_traits
        assert result["requires_confirmation"] is False


# -- apply_identity_change -----------------------------------------------


class TestApplyIdentityChange:
    def test_adds_new_trait(self) -> None:
        state = IdentityState(core_traits=["curiosity"])
        change = {"trait": "empathy", "new_value": "high"}
        result = apply_identity_change(state, change)
        assert "empathy" in result.core_traits
        assert "curiosity" in result.core_traits

    def test_existing_trait_not_duplicated(self) -> None:
        state = IdentityState(core_traits=["curiosity"])
        change = {"trait": "curiosity", "new_value": "stronger"}
        result = apply_identity_change(state, change)
        assert result.core_traits.count("curiosity") == 1

    def test_trait_confidence_set(self) -> None:
        state = IdentityState()
        change = {"trait": "empathy", "new_value": "high"}
        result = apply_identity_change(state, change)
        # New trait gets default confidence of 0.5
        assert result.trait_confidence["empathy"] == pytest.approx(0.5)

    def test_existing_confidence_preserved(self) -> None:
        state = IdentityState(
            core_traits=["curiosity"],
            trait_confidence={"curiosity": 0.8},
        )
        change = {"trait": "curiosity", "new_value": "stronger"}
        result = apply_identity_change(state, change)
        # Existing confidence is preserved (get returns existing 0.8)
        assert result.trait_confidence["curiosity"] == pytest.approx(0.8)

    def test_removes_change_from_pending(self) -> None:
        change = {"trait": "empathy", "new_value": "high"}
        state = IdentityState(pending_changes=[change, {"trait": "other"}])
        result = apply_identity_change(state, change)
        assert change not in result.pending_changes
        assert len(result.pending_changes) == 1

    def test_returns_new_state(self) -> None:
        state = IdentityState()
        change = {"trait": "empathy", "new_value": "high"}
        result = apply_identity_change(state, change)
        assert result is not state

    def test_original_state_unchanged(self) -> None:
        state = IdentityState(core_traits=["curiosity"])
        change = {"trait": "empathy", "new_value": "high"}
        _ = apply_identity_change(state, change)
        assert "empathy" not in state.core_traits

    def test_risk_recalculated(self) -> None:
        state = IdentityState(
            pending_changes=[{"trait": "a"}, {"trait": "b"}, {"trait": "c"}],
        )
        change = {"trait": "a", "new_value": "x"}
        result = apply_identity_change(state, change)
        # 2 pending left => pending_risk = 2*0.15 = 0.3
        assert result.risk > 0.0

    def test_empty_trait_key(self) -> None:
        state = IdentityState()
        change = {"new_value": "something"}  # trait=""
        result = apply_identity_change(state, change)
        # Empty string trait is not added (empty string is falsy)
        assert "" not in result.core_traits


# -- calc_identity_risk --------------------------------------------------


class TestCalcIdentityRisk:
    def test_no_pending_high_confidence(self) -> None:
        state = IdentityState(
            trait_confidence={"curiosity": 0.9, "empathy": 0.8},
        )
        risk = calc_identity_risk(state)
        # pending_risk = 0, avg_conf = 0.85, conf_risk = max(0, 0.5-0.85) = 0
        assert risk == pytest.approx(0.0)

    def test_many_pending_changes(self) -> None:
        state = IdentityState(
            pending_changes=[{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}],
            trait_confidence={"x": 0.9},
        )
        risk = calc_identity_risk(state)
        # pending_risk = min(4*0.15, 0.5) = min(0.6, 0.5) = 0.5
        # avg_conf = 0.9, conf_risk = max(0, 0.5-0.9) = 0
        assert risk == pytest.approx(0.5)

    def test_low_confidence(self) -> None:
        state = IdentityState(
            trait_confidence={"curiosity": 0.1, "empathy": 0.1},
        )
        risk = calc_identity_risk(state)
        # pending_risk=0, avg_conf=0.1, conf_risk = max(0, 0.5-0.1) = 0.4
        assert risk == pytest.approx(0.4)

    def test_no_traits_moderate_risk(self) -> None:
        state = IdentityState()
        risk = calc_identity_risk(state)
        # pending_risk=0, no confidence dict => conf_risk=0.3
        assert risk == pytest.approx(0.3)

    def test_combined_risk_capped(self) -> None:
        state = IdentityState(
            pending_changes=[{} for _ in range(10)],
            trait_confidence={"x": 0.0},
        )
        risk = calc_identity_risk(state)
        # pending_risk = min(10*0.15, 0.5) = 0.5
        # avg_conf = 0.0, conf_risk = max(0, 0.5 - 0.0) = 0.5
        # total = 1.0, min(1.0, 1.0) = 1.0
        assert risk == pytest.approx(1.0)

    def test_delegates_to_from_values(self) -> None:
        state = IdentityState(
            pending_changes=[{"a": 1}],
            trait_confidence={"x": 0.6},
        )
        direct = calc_identity_risk(state)
        via_helper = calc_identity_risk_from_values(
            state.pending_changes, state.trait_confidence,
        )
        assert direct == pytest.approx(via_helper)


# -- calc_identity_risk_from_values --------------------------------------


class TestCalcIdentityRiskFromValues:
    def test_empty_pending_empty_confidence(self) -> None:
        # pending_risk=0, conf_risk=0.3
        assert calc_identity_risk_from_values([], {}) == pytest.approx(0.3)

    def test_one_pending(self) -> None:
        # pending_risk=0.15, conf_risk=0.3
        assert calc_identity_risk_from_values([{}], {}) == pytest.approx(0.45)

    def test_pending_risk_cap(self) -> None:
        # 4 pending => 0.6, capped at 0.5
        assert calc_identity_risk_from_values(
            [{} for _ in range(4)], {"x": 0.5},
        ) == pytest.approx(0.5)  # 0.5 + max(0, 0.5-0.5)=0 => 0.5

    def test_high_confidence_zero_conf_risk(self) -> None:
        # avg_conf = 0.9, conf_risk = max(0, 0.5 - 0.9) = 0
        assert calc_identity_risk_from_values(
            [], {"a": 0.9},
        ) == pytest.approx(0.0)

    def test_medium_confidence(self) -> None:
        # avg_conf = 0.3, conf_risk = max(0, 0.5 - 0.3) = 0.2
        assert calc_identity_risk_from_values(
            [], {"a": 0.3},
        ) == pytest.approx(0.2)

    def test_total_capped_at_1(self) -> None:
        result = calc_identity_risk_from_values(
            [{} for _ in range(10)], {"a": 0.0},
        )
        assert result == pytest.approx(1.0)

    def test_exactly_at_confidence_threshold(self) -> None:
        # avg_conf = 0.5, conf_risk = max(0, 0.5 - 0.5) = 0
        assert calc_identity_risk_from_values(
            [], {"a": 0.5},
        ) == pytest.approx(0.0)

    def test_three_pending(self) -> None:
        # pending_risk = 3 * 0.15 = 0.45
        assert calc_identity_risk_from_values(
            [{}, {}, {}], {"a": 0.9},
        ) == pytest.approx(0.45)


# ========================================================================
# projection_manager.py
# ========================================================================


# -- add_goal ------------------------------------------------------------


class TestAddGoal:
    def test_adds_goal_to_empty(self) -> None:
        state = ProjectionState()
        result = add_goal(state, "Learn Python")
        assert len(result.goals) == 1
        assert result.goals[0]["description"] == "Learn Python"

    def test_goal_has_id(self) -> None:
        state = ProjectionState()
        result = add_goal(state, "Learn Python")
        assert "id" in result.goals[0]
        assert len(result.goals[0]["id"]) == 8

    def test_goal_initial_progress_zero(self) -> None:
        state = ProjectionState()
        result = add_goal(state, "Learn Python")
        assert result.goals[0]["progress"] == pytest.approx(0.0)

    def test_goal_initial_status_active(self) -> None:
        state = ProjectionState()
        result = add_goal(state, "Learn Python")
        assert result.goals[0]["status"] == "active"

    def test_adds_to_existing_goals(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "Old", "progress": 0.5, "status": "active"},
        ])
        result = add_goal(state, "New Goal")
        assert len(result.goals) == 2

    def test_returns_new_state(self) -> None:
        state = ProjectionState()
        result = add_goal(state, "Learn Python")
        assert result is not state

    def test_risk_updated_after_add(self) -> None:
        state = ProjectionState()
        # No goals => risk = 0.7
        assert state.risk == pytest.approx(0.0)  # default
        result = add_goal(state, "Learn Python")
        # One active goal with progress=0 => stalled => 0.6
        assert result.risk == pytest.approx(0.6)

    def test_unique_ids(self) -> None:
        state = ProjectionState()
        result1 = add_goal(state, "Goal 1")
        result2 = add_goal(result1, "Goal 2")
        assert result2.goals[0]["id"] != result2.goals[1]["id"]


# -- update_goal_progress ------------------------------------------------


class TestUpdateGoalProgress:
    def test_increase_progress(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.0, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.3)
        assert result.goals[0]["progress"] == pytest.approx(0.3)

    def test_progress_clamped_at_1(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.8, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.5)
        assert result.goals[0]["progress"] == pytest.approx(1.0)

    def test_progress_clamped_at_0(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.2, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", -0.5)
        assert result.goals[0]["progress"] == pytest.approx(0.0)

    def test_completed_at_1(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.9, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.1)
        assert result.goals[0]["progress"] == pytest.approx(1.0)
        assert result.goals[0]["status"] == "completed"

    def test_partial_progress_stays_active(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.0, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.5)
        assert result.goals[0]["status"] == "active"

    def test_nonexistent_goal_no_change(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = update_goal_progress(state, "nonexistent", 0.3)
        assert result.goals[0]["progress"] == pytest.approx(0.5)

    def test_other_goals_unchanged(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G1", "progress": 0.5, "status": "active"},
            {"id": "def", "description": "G2", "progress": 0.2, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.3)
        assert result.goals[1]["progress"] == pytest.approx(0.2)

    def test_returns_new_state(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = update_goal_progress(state, "abc", 0.1)
        assert result is not state

    def test_risk_updated_after_progress(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.0, "status": "active"},
        ])
        # stalled => risk 0.6
        result_stalled = update_goal_progress(state, "abc", 0.0)
        assert result_stalled.risk == pytest.approx(0.6)
        # progressing => risk 0.1
        result_prog = update_goal_progress(state, "abc", 0.5)
        assert result_prog.risk == pytest.approx(0.1)


# -- remove_goal ---------------------------------------------------------


class TestRemoveGoal:
    def test_remove_existing_goal(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G1", "progress": 0.5, "status": "active"},
            {"id": "def", "description": "G2", "progress": 0.2, "status": "active"},
        ])
        result = remove_goal(state, "abc")
        assert len(result.goals) == 1
        assert result.goals[0]["id"] == "def"

    def test_remove_nonexistent_goal(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = remove_goal(state, "nonexistent")
        assert len(result.goals) == 1

    def test_remove_last_goal(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = remove_goal(state, "abc")
        assert len(result.goals) == 0
        # No goals => risk = 0.7
        assert result.risk == pytest.approx(0.7)

    def test_returns_new_state(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = remove_goal(state, "abc")
        assert result is not state


# -- reset ---------------------------------------------------------------


class TestProjectionReset:
    def test_reset_clears_goals(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = reset(state)
        assert len(result.goals) == 0

    def test_reset_risk_is_no_goals(self) -> None:
        state = ProjectionState(goals=[
            {"id": "abc", "description": "G", "progress": 0.5, "status": "active"},
        ])
        result = reset(state)
        assert result.risk == pytest.approx(0.7)

    def test_reset_returns_new_state(self) -> None:
        state = ProjectionState()
        result = reset(state)
        assert result is not state

    def test_reset_from_empty(self) -> None:
        state = ProjectionState()
        result = reset(state)
        assert len(result.goals) == 0
        assert result.risk == pytest.approx(0.7)


# -- calc_projection_risk ------------------------------------------------


class TestCalcProjectionRisk:
    def test_no_goals(self) -> None:
        state = ProjectionState()
        assert calc_projection_risk(state) == pytest.approx(0.7)

    def test_all_completed(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G", "progress": 1.0, "status": "completed"},
        ])
        assert calc_projection_risk(state) == pytest.approx(0.5)

    def test_all_stalled(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G", "progress": 0.0, "status": "active"},
        ])
        assert calc_projection_risk(state) == pytest.approx(0.6)

    def test_progressing(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G", "progress": 0.5, "status": "active"},
        ])
        assert calc_projection_risk(state) == pytest.approx(0.1)

    def test_mixed_stalled_and_progressing(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G1", "progress": 0.0, "status": "active"},
            {"id": "b", "description": "G2", "progress": 0.5, "status": "active"},
        ])
        # At least one progressing => 0.1
        assert calc_projection_risk(state) == pytest.approx(0.1)

    def test_completed_and_active_progressing(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G1", "progress": 1.0, "status": "completed"},
            {"id": "b", "description": "G2", "progress": 0.3, "status": "active"},
        ])
        assert calc_projection_risk(state) == pytest.approx(0.1)

    def test_completed_and_active_stalled(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G1", "progress": 1.0, "status": "completed"},
            {"id": "b", "description": "G2", "progress": 0.0, "status": "active"},
        ])
        # Active exists, but all active are stalled => 0.6
        assert calc_projection_risk(state) == pytest.approx(0.6)

    def test_no_active_multiple_completed(self) -> None:
        state = ProjectionState(goals=[
            {"id": "a", "description": "G1", "progress": 1.0, "status": "completed"},
            {"id": "b", "description": "G2", "progress": 1.0, "status": "completed"},
        ])
        assert calc_projection_risk(state) == pytest.approx(0.5)
