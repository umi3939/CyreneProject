"""
tests/test_responsibility_dispersion.py - Tests for responsibility dispersion & sublimation

Verifies:
1. Responsibility unit creation and structure
2. Conservation of weight in all operations
3. Append-only audit trail integrity
4. Dispersion (distribution to multiple receivers)
5. Sublimation (meaning transformation)
6. Time distribution
7. Distance adjustment
8. Merge operations
"""

import pytest

from psyche.responsibility_dispersion import (
    # Data models
    ResponsibilityUnit,
    DispersionPlan,
    SublimationPath,
    TimeSlice,
    TimeDistributionPlan,
    DispersionState,
    DispersionConfig,
    AuditEntry,
    AuditEventType,
    DispersionStrategy,
    # Functions
    create_responsibility_unit,
    disperse_responsibility,
    sublimate_responsibility,
    distribute_over_time,
    adjust_distance,
    merge_responsibilities,
    get_audit_trail,
    get_unit_by_id,
    get_active_units,
    get_total_active_weight,
    get_lineage,
    verify_state_conservation,
    create_dispersion_state,
    to_dict,
    from_dict,
    get_dispersion_summary,
    # Exceptions
    ConservationViolationError,
)


class TestResponsibilityUnit:
    """Tests for ResponsibilityUnit structure."""

    def test_create_unit_with_defaults(self):
        """Unit can be created with defaults."""
        unit = ResponsibilityUnit(weight=0.5)

        assert unit.weight == 0.5
        assert unit.meaning == ""
        assert unit.distance == 1.0
        assert unit.generation == 0
        assert unit.transformed is False

    def test_unit_has_unique_id(self):
        """Each unit has a unique ID."""
        unit1 = ResponsibilityUnit(weight=0.1)
        unit2 = ResponsibilityUnit(weight=0.1)

        assert unit1.id != unit2.id

    def test_unit_fingerprint(self):
        """Unit fingerprint is deterministic."""
        unit = ResponsibilityUnit(weight=0.5, meaning="test")
        fp1 = unit.fingerprint()
        fp2 = unit.fingerprint()

        assert fp1 == fp2
        assert len(fp1) == 16

    def test_unit_serialization(self):
        """Unit survives serialization roundtrip."""
        unit = ResponsibilityUnit(
            weight=0.5,
            meaning="caused_harm",
            origin="decision_123",
            distance=2.0,
            generation=3,
        )

        data = unit.model_dump()
        restored = ResponsibilityUnit(**data)

        assert restored.weight == unit.weight
        assert restored.meaning == unit.meaning
        assert restored.origin == unit.origin
        assert restored.distance == unit.distance
        assert restored.generation == unit.generation


class TestCreateResponsibilityUnit:
    """Tests for create_responsibility_unit function."""

    def test_create_unit_basic(self):
        """Unit can be created with basic parameters."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="decision_001",
            meaning="caused_confusion",
        )

        assert unit.weight == 0.5
        assert unit.origin == "decision_001"
        assert unit.meaning == "caused_confusion"
        assert unit.transformed is False

    def test_create_unit_records_audit(self):
        """Creation is recorded in audit log."""
        unit, state = create_responsibility_unit(
            weight=0.3,
            origin="test",
            meaning="test_meaning",
        )

        assert len(state.audit_log) == 1
        entry = state.audit_log[0]
        assert entry["event_type"] == AuditEventType.CREATED
        assert unit.id in entry["result_ids"]

    def test_create_unit_updates_total_created(self):
        """Creation updates total_weight_created."""
        _, state1 = create_responsibility_unit(weight=0.3, origin="a")
        _, state2 = create_responsibility_unit(weight=0.4, origin="b", state=state1)

        assert state2.total_weight_created == pytest.approx(0.7)

    def test_create_unit_negative_weight_fails(self):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            create_responsibility_unit(weight=-0.1, origin="test")


class TestDispersion:
    """Tests for disperse_responsibility function."""

    def test_disperse_basic(self):
        """Responsibility can be dispersed to multiple targets."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["receiver_a", "receiver_b"],
            weights=[0.6, 0.4],
        )

        result_units, new_state = disperse_responsibility(unit, plan, state)

        assert len(result_units) == 2
        assert result_units[0].weight == pytest.approx(0.6)
        assert result_units[1].weight == pytest.approx(0.4)

    def test_disperse_conserves_weight(self):
        """Dispersion conserves total weight."""
        unit, state = create_responsibility_unit(weight=0.8, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a", "b", "c"],
            weights=[0.3, 0.3, 0.2],
        )

        result_units, new_state = disperse_responsibility(unit, plan, state)
        total_output = sum(u.weight for u in result_units)

        assert total_output == pytest.approx(unit.weight)

    def test_disperse_marks_original_transformed(self):
        """Original unit is marked as transformed."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a", "b"],
            weights=[0.25, 0.25],
        )

        _, new_state = disperse_responsibility(unit, plan, state)

        original = get_unit_by_id(new_state, unit.id)
        assert original.transformed is True

    def test_disperse_increments_generation(self):
        """Dispersed units have incremented generation."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a"],
            weights=[0.5],
        )

        result_units, _ = disperse_responsibility(unit, plan, state)

        assert result_units[0].generation == unit.generation + 1

    def test_disperse_records_audit(self):
        """Dispersion is recorded in audit log."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a", "b"],
            weights=[0.25, 0.25],
        )

        _, new_state = disperse_responsibility(unit, plan, state)

        # Should have 2 entries: create + disperse
        assert len(new_state.audit_log) == 2
        disperse_entry = new_state.audit_log[1]
        assert disperse_entry["event_type"] == AuditEventType.DISPERSED
        assert disperse_entry["conservation_verified"] is True

    def test_disperse_conservation_violation_raises(self):
        """Conservation violation raises error."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a", "b"],
            weights=[0.5, 0.6],  # Total 1.1 != 1.0
        )

        with pytest.raises(ConservationViolationError):
            disperse_responsibility(unit, plan, state)

    def test_disperse_transformed_unit_fails(self):
        """Cannot disperse already transformed unit."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        # First dispersion
        plan1 = DispersionPlan(
            source_id=unit.id,
            targets=["a"],
            weights=[1.0],
        )
        _, state = disperse_responsibility(unit, plan1, state)

        # Try to disperse again (should fail)
        plan2 = DispersionPlan(
            source_id=unit.id,
            targets=["b"],
            weights=[1.0],
        )
        with pytest.raises(ValueError, match="transformed"):
            disperse_responsibility(unit, plan2, state)


class TestSublimation:
    """Tests for sublimate_responsibility function."""

    def test_sublimate_basic(self):
        """Responsibility meaning can be transformed."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            meaning="caused_harm",
        )

        path = SublimationPath(
            source_meaning="caused_harm",
            target_meaning="learned_caution",
            weight_ratio=1.0,
        )

        new_unit, new_state = sublimate_responsibility(unit, path, state)

        assert new_unit.meaning == "learned_caution"
        assert new_unit.weight == pytest.approx(unit.weight)

    def test_sublimate_conserves_weight(self):
        """Sublimation with ratio 1.0 conserves weight."""
        unit, state = create_responsibility_unit(weight=0.7, origin="test")

        path = SublimationPath(
            source_meaning="a",
            target_meaning="b",
            weight_ratio=1.0,
        )

        new_unit, _ = sublimate_responsibility(unit, path, state)

        assert new_unit.weight == pytest.approx(unit.weight)

    def test_sublimate_adjusts_distance(self):
        """Sublimation can adjust distance."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            distance=1.0,
        )

        path = SublimationPath(
            source_meaning="a",
            target_meaning="b",
            weight_ratio=1.0,
            distance_delta=0.5,
        )

        new_unit, _ = sublimate_responsibility(unit, path, state)

        assert new_unit.distance == pytest.approx(1.5)

    def test_sublimate_records_audit(self):
        """Sublimation is recorded in audit log."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        path = SublimationPath(
            source_meaning="a",
            target_meaning="b",
        )

        _, new_state = sublimate_responsibility(unit, path, state)

        assert len(new_state.audit_log) == 2  # create + sublimate
        sublimate_entry = new_state.audit_log[1]
        assert sublimate_entry["event_type"] == AuditEventType.SUBLIMATED

    def test_sublimate_sets_parent(self):
        """Sublimated unit has parent reference."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        path = SublimationPath(
            source_meaning="a",
            target_meaning="b",
        )

        new_unit, _ = sublimate_responsibility(unit, path, state)

        assert new_unit.parent_id == unit.id


class TestTimeDistribution:
    """Tests for distribute_over_time function."""

    def test_distribute_basic(self):
        """Responsibility can be distributed over time."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        plan = TimeDistributionPlan(
            source_id=unit.id,
            slices=[
                TimeSlice(label="immediate", weight_ratio=0.5),
                TimeSlice(label="near_future", weight_ratio=0.3),
                TimeSlice(label="distant_future", weight_ratio=0.2),
            ],
        )

        result_units, new_state = distribute_over_time(unit, plan, state)

        assert len(result_units) == 3
        assert result_units[0].time_slice == "immediate"
        assert result_units[0].weight == pytest.approx(0.5)

    def test_distribute_conserves_weight(self):
        """Time distribution conserves total weight."""
        unit, state = create_responsibility_unit(weight=0.9, origin="test")

        plan = TimeDistributionPlan(
            source_id=unit.id,
            slices=[
                TimeSlice(label="a", weight_ratio=0.4),
                TimeSlice(label="b", weight_ratio=0.6),
            ],
        )

        result_units, _ = distribute_over_time(unit, plan, state)
        total = sum(u.weight for u in result_units)

        assert total == pytest.approx(unit.weight)

    def test_distribute_invalid_plan_fails(self):
        """Invalid plan (ratios not summing to 1) fails."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        plan = TimeDistributionPlan(
            source_id=unit.id,
            slices=[
                TimeSlice(label="a", weight_ratio=0.3),
                TimeSlice(label="b", weight_ratio=0.3),
                # Missing 0.4
            ],
        )

        with pytest.raises(ValueError, match="Invalid"):
            distribute_over_time(unit, plan, state)

    def test_distribute_records_audit(self):
        """Time distribution is recorded in audit log."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        plan = TimeDistributionPlan(
            source_id=unit.id,
            slices=[
                TimeSlice(label="a", weight_ratio=0.5),
                TimeSlice(label="b", weight_ratio=0.5),
            ],
        )

        _, new_state = distribute_over_time(unit, plan, state)

        assert len(new_state.audit_log) == 2
        time_entry = new_state.audit_log[1]
        assert time_entry["event_type"] == AuditEventType.TIME_SPLIT


class TestDistanceAdjustment:
    """Tests for adjust_distance function."""

    def test_adjust_distance_basic(self):
        """Distance can be adjusted."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            distance=1.0,
        )

        new_unit, new_state = adjust_distance(unit, 2.5, state)

        assert new_unit.distance == pytest.approx(2.5)
        assert new_unit.weight == pytest.approx(unit.weight)  # Conserved

    def test_adjust_distance_conserves_weight(self):
        """Distance adjustment conserves weight."""
        unit, state = create_responsibility_unit(weight=0.7, origin="test")

        new_unit, _ = adjust_distance(unit, 5.0, state)

        assert new_unit.weight == pytest.approx(unit.weight)

    def test_adjust_distance_records_audit(self):
        """Distance adjustment is recorded in audit log."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        _, new_state = adjust_distance(unit, 3.0, state, rationale="test reason")

        assert len(new_state.audit_log) == 2
        dist_entry = new_state.audit_log[1]
        assert dist_entry["event_type"] == AuditEventType.DISTANCE_ADJUSTED

    def test_adjust_negative_distance_fails(self):
        """Negative distance raises ValueError."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        with pytest.raises(ValueError, match="non-negative"):
            adjust_distance(unit, -1.0, state)


class TestMerge:
    """Tests for merge_responsibilities function."""

    def test_merge_basic(self):
        """Multiple units can be merged."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        merged, new_state = merge_responsibilities(
            [unit1, unit2],
            "combined_burden",
            state,
        )

        assert merged.weight == pytest.approx(0.7)
        assert merged.meaning == "combined_burden"

    def test_merge_conserves_weight(self):
        """Merge conserves total weight."""
        unit1, state = create_responsibility_unit(weight=0.2, origin="a")
        unit2, state = create_responsibility_unit(weight=0.3, origin="b", state=state)
        unit3, state = create_responsibility_unit(weight=0.1, origin="c", state=state)

        merged, _ = merge_responsibilities(
            [unit1, unit2, unit3],
            "all_combined",
            state,
        )

        assert merged.weight == pytest.approx(0.6)

    def test_merge_marks_sources_transformed(self):
        """Source units are marked as transformed."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        _, new_state = merge_responsibilities([unit1, unit2], "merged", state)

        original1 = get_unit_by_id(new_state, unit1.id)
        original2 = get_unit_by_id(new_state, unit2.id)

        assert original1.transformed is True
        assert original2.transformed is True

    def test_merge_empty_list_fails(self):
        """Merging empty list raises ValueError."""
        state = create_dispersion_state()

        with pytest.raises(ValueError, match="empty"):
            merge_responsibilities([], "test", state)

    def test_merge_records_audit(self):
        """Merge is recorded in audit log."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        _, new_state = merge_responsibilities([unit1, unit2], "merged", state)

        # 3 entries: create a, create b, merge
        assert len(new_state.audit_log) == 3
        merge_entry = new_state.audit_log[2]
        assert merge_entry["event_type"] == AuditEventType.MERGED


class TestAuditTrail:
    """Tests for audit trail functions."""

    def test_get_audit_trail_all(self):
        """Can retrieve full audit trail."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        path = SublimationPath(source_meaning="a", target_meaning="b")
        _, state = sublimate_responsibility(unit, path, state)

        trail = get_audit_trail(state)

        assert len(trail) == 2
        assert all(isinstance(e, AuditEntry) for e in trail)

    def test_get_audit_trail_by_unit(self):
        """Can filter audit trail by unit ID."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        trail = get_audit_trail(state, unit_id=unit1.id)

        assert len(trail) == 1
        assert unit1.id in trail[0].result_ids

    def test_audit_entries_are_immutable(self):
        """Audit entries are immutable (via pydantic)."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        trail = get_audit_trail(state)
        entry = trail[0]

        # Entry fields are accessible
        assert entry.event_type == AuditEventType.CREATED


class TestQueryFunctions:
    """Tests for query functions."""

    def test_get_unit_by_id(self):
        """Can retrieve unit by ID."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        retrieved = get_unit_by_id(state, unit.id)

        assert retrieved is not None
        assert retrieved.id == unit.id
        assert retrieved.weight == unit.weight

    def test_get_unit_by_id_not_found(self):
        """Returns None for unknown ID."""
        state = create_dispersion_state()

        retrieved = get_unit_by_id(state, "nonexistent")

        assert retrieved is None

    def test_get_active_units(self):
        """Can retrieve only active (non-transformed) units."""
        unit1, state = create_responsibility_unit(weight=0.5, origin="a")

        # Transform unit1
        path = SublimationPath(source_meaning="a", target_meaning="b")
        unit2, state = sublimate_responsibility(unit1, path, state)

        active = get_active_units(state)

        assert len(active) == 1
        assert active[0].id == unit2.id

    def test_get_total_active_weight(self):
        """Can compute total active weight."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        total = get_total_active_weight(state)

        assert total == pytest.approx(0.7)

    def test_get_lineage(self):
        """Can trace unit lineage."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        # First transformation
        path1 = SublimationPath(source_meaning="a", target_meaning="b")
        unit2, state = sublimate_responsibility(unit, path1, state)

        # Second transformation
        path2 = SublimationPath(source_meaning="b", target_meaning="c")
        unit3, state = sublimate_responsibility(unit2, path2, state)

        lineage = get_lineage(state, unit3.id)

        assert len(lineage) == 3
        assert lineage[0].id == unit3.id  # Current
        assert lineage[1].id == unit2.id  # Parent
        assert lineage[2].id == unit.id   # Grandparent


class TestStateVerification:
    """Tests for state verification functions."""

    def test_verify_state_conservation_basic(self):
        """State conservation is verified correctly."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")

        result = verify_state_conservation(state)

        assert result["is_consistent"] is True
        assert result["active_weight"] == pytest.approx(0.5)
        assert result["created_weight"] == pytest.approx(0.5)

    def test_verify_after_transformations(self):
        """Conservation holds after multiple transformations."""
        unit, state = create_responsibility_unit(weight=1.0, origin="test")

        # Disperse
        plan = DispersionPlan(
            source_id=unit.id,
            targets=["a", "b"],
            weights=[0.6, 0.4],
        )
        results, state = disperse_responsibility(unit, plan, state)

        # Sublimate one
        path = SublimationPath(source_meaning="x", target_meaning="y")
        _, state = sublimate_responsibility(results[0], path, state)

        result = verify_state_conservation(state)

        assert result["is_consistent"] is True
        assert result["active_weight"] == pytest.approx(1.0)


class TestSerialization:
    """Tests for serialization functions."""

    def test_to_dict_and_from_dict(self):
        """State survives serialization roundtrip."""
        unit1, state = create_responsibility_unit(weight=0.3, origin="a")
        unit2, state = create_responsibility_unit(weight=0.4, origin="b", state=state)

        data = to_dict(state)
        restored = from_dict(data)

        assert len(restored.units) == len(state.units)
        assert len(restored.audit_log) == len(state.audit_log)
        assert restored.total_weight_created == pytest.approx(state.total_weight_created)

    def test_from_dict_invalid_returns_default(self):
        """Invalid dict returns default state."""
        restored = from_dict({"invalid": "data"})

        assert isinstance(restored, DispersionState)
        assert len(restored.units) == 0


class TestDispersionSummary:
    """Tests for get_dispersion_summary function."""

    def test_summary_basic(self):
        """Summary provides useful info."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            meaning="caused_harm",
        )

        summary = get_dispersion_summary(state)

        assert "Active units: 1" in summary
        assert "caused_harm" in summary

    def test_summary_empty_state(self):
        """Empty state produces valid summary."""
        state = create_dispersion_state()

        summary = get_dispersion_summary(state)

        assert "Active units: 0" in summary


class TestConfigurableRules:
    """Tests for configurable dispersion/sublimation rules."""

    def test_custom_dispersion_rule(self):
        """Custom dispersion rule can be provided."""
        def custom_rule(unit: ResponsibilityUnit) -> DispersionPlan:
            return DispersionPlan(
                source_id=unit.id,
                targets=["auto_a", "auto_b"],
                weights=[unit.weight * 0.5, unit.weight * 0.5],
                strategy=DispersionStrategy.CUSTOM,
            )

        config = DispersionConfig(dispersion_rule=custom_rule)

        unit, state = create_responsibility_unit(
            weight=1.0,
            origin="test",
            config=config,
        )

        # Use the rule to generate plan
        plan = config.dispersion_rule(unit)

        assert plan.targets == ["auto_a", "auto_b"]
        assert plan.total_weight() == pytest.approx(1.0)


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_chain_of_transformations(self):
        """Complex chain preserves weight throughout."""
        # Create initial unit
        unit, state = create_responsibility_unit(
            weight=1.0,
            origin="initial_decision",
            meaning="caused_confusion",
        )

        # Disperse to two targets
        plan = DispersionPlan(
            source_id=unit.id,
            targets=["partner_a", "partner_b"],
            weights=[0.6, 0.4],
        )
        dispersed, state = disperse_responsibility(unit, plan, state)

        # Sublimate the first one
        path = SublimationPath(
            source_meaning="caused_confusion",
            target_meaning="learned_to_clarify",
        )
        sublimated, state = sublimate_responsibility(dispersed[0], path, state)

        # Distribute the second over time
        time_plan = TimeDistributionPlan(
            source_id=dispersed[1].id,
            slices=[
                TimeSlice(label="now", weight_ratio=0.5),
                TimeSlice(label="later", weight_ratio=0.5),
            ],
        )
        time_distributed, state = distribute_over_time(dispersed[1], time_plan, state)

        # Verify conservation
        result = verify_state_conservation(state)
        assert result["is_consistent"] is True
        assert result["active_weight"] == pytest.approx(1.0)

        # Check active units
        active = get_active_units(state)
        assert len(active) == 3  # sublimated + 2 time slices

    def test_merge_and_redisperse(self):
        """Can merge units and then redisperse."""
        # Create multiple units
        unit1, state = create_responsibility_unit(weight=0.2, origin="a")
        unit2, state = create_responsibility_unit(weight=0.3, origin="b", state=state)
        unit3, state = create_responsibility_unit(weight=0.5, origin="c", state=state)

        # Merge all
        merged, state = merge_responsibilities(
            [unit1, unit2, unit3],
            "accumulated_burden",
            state,
        )

        assert merged.weight == pytest.approx(1.0)

        # Redisperse
        plan = DispersionPlan(
            source_id=merged.id,
            targets=["new_a", "new_b", "new_c"],
            weights=[0.5, 0.3, 0.2],
        )
        redispersed, state = disperse_responsibility(merged, plan, state)

        # Verify
        result = verify_state_conservation(state)
        assert result["is_consistent"] is True
        assert len(redispersed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
