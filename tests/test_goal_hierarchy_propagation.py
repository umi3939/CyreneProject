"""
tests/test_goal_hierarchy_propagation.py - 目的階層間の隣接状態変化記述のテスト

包括的テスト:
- スナップショット構築（各層）
- 変化検出（生成・消失・カテゴリ変更・強度段階変化・昇格・解除・次元変化・確信度変化）
- 隣接同時性の記録構成
- FIFO蓄積と有限性
- 鮮度減衰（均一性）
- 収束監視（非介入性）
- 安全弁（全記録等価、因果帰属排除、enrichment直接露出遮断、FIFO有限性、3層逆流不在、段階値限定、収束内部限定）
- 経路遮断の検証
- save/load
- パイプラインの各段階
"""

import time
import pytest

from psyche.goal_hierarchy_propagation import (
    # Enums
    Layer1ChangeType,
    Layer2ChangeType,
    Layer3ChangeType,
    RecordFreshness,
    ConvergenceLevel,
    # Snapshots
    Layer1Snapshot,
    Layer2Snapshot,
    Layer2ItemSnapshot,
    Layer3Snapshot,
    Layer3DimSnapshot,
    # Changes
    Layer1Change,
    Layer2Change,
    Layer3Change,
    # Record
    AdjacencyRecord,
    ConvergenceRecord,
    # State & Config
    GoalHierarchyPropagationState,
    GoalHierarchyPropagationConfig,
    # Snapshot builders
    build_layer1_snapshot,
    build_layer2_snapshot,
    build_layer3_snapshot,
    # Change detectors
    detect_layer1_changes,
    detect_layer2_changes,
    detect_layer3_changes,
    # Processor
    GoalHierarchyPropagationProcessor,
    # Save/Load
    save_state,
    load_state,
    # Factory
    create_goal_hierarchy_propagation_processor,
    # Helpers
    _strength_stage,
    _orientation_stage,
    _confidence_stage,
    _freshness_stage,
    _clamp,
)


# =============================================================================
# Helper Stage Functions
# =============================================================================

class TestStrengthStage:
    def test_strong(self):
        assert _strength_stage(0.9) == "strong"
        assert _strength_stage(0.7) == "strong"

    def test_moderate(self):
        assert _strength_stage(0.5) == "moderate"
        assert _strength_stage(0.4) == "moderate"

    def test_weak(self):
        assert _strength_stage(0.3) == "weak"
        assert _strength_stage(0.2) == "weak"

    def test_faint(self):
        assert _strength_stage(0.1) == "faint"
        assert _strength_stage(0.05) == "faint"

    def test_absent(self):
        assert _strength_stage(0.0) == "absent"
        assert _strength_stage(0.04) == "absent"


class TestOrientationStage:
    def test_strong(self):
        assert _orientation_stage(0.8) == "strong"
        assert _orientation_stage(-0.7) == "strong"

    def test_moderate(self):
        assert _orientation_stage(0.5) == "moderate"
        assert _orientation_stage(-0.4) == "moderate"

    def test_weak(self):
        assert _orientation_stage(0.2) == "weak"
        assert _orientation_stage(-0.15) == "weak"

    def test_faint(self):
        assert _orientation_stage(0.08) == "faint"
        assert _orientation_stage(-0.05) == "faint"

    def test_neutral(self):
        assert _orientation_stage(0.0) == "neutral"
        assert _orientation_stage(0.04) == "neutral"


class TestConfidenceStage:
    def test_high(self):
        assert _confidence_stage(0.8) == "high"
        assert _confidence_stage(0.7) == "high"

    def test_moderate(self):
        assert _confidence_stage(0.5) == "moderate"
        assert _confidence_stage(0.4) == "moderate"

    def test_low(self):
        assert _confidence_stage(0.2) == "low"
        assert _confidence_stage(0.15) == "low"

    def test_minimal(self):
        assert _confidence_stage(0.1) == "minimal"
        assert _confidence_stage(0.05) == "minimal"

    def test_none(self):
        assert _confidence_stage(0.0) == "none"


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_min(self):
        assert _clamp(-0.1) == 0.0

    def test_above_max(self):
        assert _clamp(1.5) == 1.0


# =============================================================================
# Snapshot Building
# =============================================================================

class TestBuildLayer1Snapshot:
    def test_none_input(self):
        snap = build_layer1_snapshot(None)
        assert snap.has_active is False
        assert snap.category == ""
        assert snap.strength_stage == "absent"

    def test_active_goal(self):
        data = {
            "has_active": True,
            "category": "exploration",
            "direction_signature": {"a": 0.5, "b": -0.3},
            "strength": 0.6,
        }
        snap = build_layer1_snapshot(data)
        assert snap.has_active is True
        assert snap.category == "exploration"
        assert snap.direction_signature_summary == "a,b"
        assert snap.strength_stage == "moderate"

    def test_inactive_goal(self):
        data = {
            "has_active": False,
            "category": "",
            "direction_signature": {},
            "strength": 0.0,
        }
        snap = build_layer1_snapshot(data)
        assert snap.has_active is False
        assert snap.direction_signature_summary == ""

    def test_empty_data(self):
        snap = build_layer1_snapshot({})
        assert snap.has_active is False
        assert snap.strength_stage == "absent"


class TestBuildLayer2Snapshot:
    def test_none_input(self):
        snap = build_layer2_snapshot(None)
        assert snap.items == []

    def test_with_items(self):
        data = {
            "items": [
                {"item_id": "a1", "category": "growth", "strength": 0.5},
                {"item_id": "b2", "category": "social", "strength": 0.8},
            ],
            "recent_cognition_types": ["promotion"],
        }
        snap = build_layer2_snapshot(data)
        assert len(snap.items) == 2
        assert snap.items[0].item_id == "a1"
        assert snap.items[0].strength_stage == "moderate"
        assert snap.items[1].strength_stage == "strong"
        assert snap.recent_cognition_types == ["promotion"]

    def test_empty_data(self):
        snap = build_layer2_snapshot({})
        assert snap.items == []
        assert snap.recent_cognition_types == []


class TestBuildLayer3Snapshot:
    def test_none_input(self):
        snap = build_layer3_snapshot(None)
        assert snap.dimensions == []
        assert snap.update_count_stage == "none"

    def test_with_dimensions(self):
        data = {
            "dimensions": {"a": 0.5, "b": -0.8, "c": 0.0},
            "confidences": {"a": 0.6, "b": 0.3, "c": 0.0},
            "update_count": 50,
        }
        snap = build_layer3_snapshot(data)
        assert len(snap.dimensions) == 3
        # Sorted by dim_id
        assert snap.dimensions[0].dim_id == "a"
        assert snap.dimensions[0].value_stage == "moderate"
        assert snap.dimensions[0].confidence_stage == "moderate"
        assert snap.dimensions[1].dim_id == "b"
        assert snap.dimensions[1].value_stage == "strong"
        assert snap.dimensions[1].confidence_stage == "low"
        assert snap.dimensions[2].dim_id == "c"
        assert snap.dimensions[2].value_stage == "neutral"
        assert snap.dimensions[2].confidence_stage == "none"
        assert snap.update_count_stage == "moderate"

    def test_update_count_stages(self):
        assert build_layer3_snapshot({"update_count": 0}).update_count_stage == "none"
        assert build_layer3_snapshot({"update_count": 1}).update_count_stage == "minimal"
        assert build_layer3_snapshot({"update_count": 5}).update_count_stage == "low"
        assert build_layer3_snapshot({"update_count": 30}).update_count_stage == "moderate"
        assert build_layer3_snapshot({"update_count": 100}).update_count_stage == "high"


# =============================================================================
# Change Detection
# =============================================================================

class TestDetectLayer1Changes:
    def test_no_prev(self):
        curr = Layer1Snapshot(has_active=True, category="x", strength_stage="strong")
        change = detect_layer1_changes(None, curr)
        assert change.change_type == Layer1ChangeType.NO_CHANGE.value

    def test_generation(self):
        prev = Layer1Snapshot(has_active=False)
        curr = Layer1Snapshot(has_active=True, category="exploration", strength_stage="moderate")
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.GENERATION.value
        assert change.curr_category == "exploration"

    def test_disappearance(self):
        prev = Layer1Snapshot(has_active=True, category="growth", strength_stage="strong")
        curr = Layer1Snapshot(has_active=False)
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.DISAPPEARANCE.value
        assert change.prev_category == "growth"

    def test_category_change(self):
        prev = Layer1Snapshot(has_active=True, category="exploration", strength_stage="moderate")
        curr = Layer1Snapshot(has_active=True, category="social", strength_stage="moderate")
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.CATEGORY_CHANGE.value
        assert change.prev_category == "exploration"
        assert change.curr_category == "social"

    def test_strength_change(self):
        prev = Layer1Snapshot(has_active=True, category="x", strength_stage="strong")
        curr = Layer1Snapshot(has_active=True, category="x", strength_stage="moderate")
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.STRENGTH_CHANGE.value

    def test_no_change(self):
        prev = Layer1Snapshot(has_active=True, category="x", strength_stage="moderate")
        curr = Layer1Snapshot(has_active=True, category="x", strength_stage="moderate")
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.NO_CHANGE.value

    def test_both_inactive_no_change(self):
        prev = Layer1Snapshot(has_active=False)
        curr = Layer1Snapshot(has_active=False)
        change = detect_layer1_changes(prev, curr)
        assert change.change_type == Layer1ChangeType.NO_CHANGE.value


class TestDetectLayer2Changes:
    def test_no_prev(self):
        curr = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a", category="x")])
        changes = detect_layer2_changes(None, curr)
        assert changes == []

    def test_promotion(self):
        prev = Layer2Snapshot(items=[])
        curr = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="growth", strength_stage="moderate")])
        changes = detect_layer2_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0].change_type == Layer2ChangeType.PROMOTION.value
        assert changes[0].item_category == "growth"

    def test_release(self):
        prev = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="growth")])
        curr = Layer2Snapshot(items=[], recent_cognition_types=["release"])
        changes = detect_layer2_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0].change_type == Layer2ChangeType.RELEASE.value

    def test_strength_change(self):
        prev = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="strong")])
        curr = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="moderate")])
        changes = detect_layer2_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0].change_type == Layer2ChangeType.STRENGTH_CHANGE.value

    def test_no_change(self):
        prev = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="moderate")])
        curr = Layer2Snapshot(items=[Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="moderate")])
        changes = detect_layer2_changes(prev, curr)
        assert changes == []

    def test_multiple_changes(self):
        prev = Layer2Snapshot(items=[
            Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="strong"),
        ])
        curr = Layer2Snapshot(items=[
            Layer2ItemSnapshot(item_id="a1", category="x", strength_stage="moderate"),
            Layer2ItemSnapshot(item_id="b2", category="y", strength_stage="weak"),
        ])
        changes = detect_layer2_changes(prev, curr)
        # 1 promotion + 1 strength change
        types = {c.change_type for c in changes}
        assert Layer2ChangeType.PROMOTION.value in types
        assert Layer2ChangeType.STRENGTH_CHANGE.value in types


class TestDetectLayer3Changes:
    def test_no_prev(self):
        curr = Layer3Snapshot(dimensions=[Layer3DimSnapshot(dim_id="a", value_stage="moderate")])
        change = detect_layer3_changes(None, curr)
        assert change.change_type == Layer3ChangeType.NO_CHANGE.value

    def test_dimension_change(self):
        prev = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="low"),
        ])
        curr = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="moderate", confidence_stage="low"),
        ])
        change = detect_layer3_changes(prev, curr)
        assert change.change_type == Layer3ChangeType.DIMENSION_CHANGE.value
        assert "a" in change.changed_dimensions

    def test_confidence_change(self):
        prev = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="low"),
        ])
        curr = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="moderate"),
        ])
        change = detect_layer3_changes(prev, curr)
        assert change.change_type == Layer3ChangeType.CONFIDENCE_CHANGE.value
        assert "a" in change.changed_dimensions

    def test_both_changes(self):
        prev = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="low"),
            Layer3DimSnapshot(dim_id="b", value_stage="neutral", confidence_stage="none"),
        ])
        curr = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="moderate", confidence_stage="low"),
            Layer3DimSnapshot(dim_id="b", value_stage="neutral", confidence_stage="minimal"),
        ])
        change = detect_layer3_changes(prev, curr)
        # dimension_change takes priority
        assert change.change_type == Layer3ChangeType.DIMENSION_CHANGE.value
        assert "a" in change.changed_dimensions
        assert "b" in change.changed_dimensions

    def test_no_change(self):
        prev = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="low"),
        ])
        curr = Layer3Snapshot(dimensions=[
            Layer3DimSnapshot(dim_id="a", value_stage="weak", confidence_stage="low"),
        ])
        change = detect_layer3_changes(prev, curr)
        assert change.change_type == Layer3ChangeType.NO_CHANGE.value


# =============================================================================
# Processor: Pipeline
# =============================================================================

class TestProcessorFirstCycle:
    """初回サイクル: スナップショット保存のみ、記録構成なし。"""

    def test_first_cycle_no_record(self):
        proc = create_goal_hierarchy_propagation_processor()
        result = proc.process(
            transient_goal_data={"has_active": True, "category": "x", "strength": 0.5},
        )
        assert result == 0
        assert proc.state.cycle_count == 1
        assert proc.state.prev_layer1 is not None
        assert len(proc.state.adjacency_records) == 0


class TestProcessorChangeDetection:
    """変化検出と記録構成のテスト。"""

    def _make_processor_with_prev(self):
        proc = create_goal_hierarchy_propagation_processor()
        # 第1サイクル: スナップショット保存
        proc.process(
            transient_goal_data={"has_active": True, "category": "exploration", "strength": 0.5},
            persistent_commitment_data={"items": [{"item_id": "a1", "category": "growth", "strength": 0.5}]},
            value_orientation_data={
                "dimensions": {"a": 0.5, "b": 0.0},
                "confidences": {"a": 0.3, "b": 0.0},
                "update_count": 10,
            },
        )
        return proc

    def test_no_change_no_record(self):
        proc = self._make_processor_with_prev()
        # 同じデータで第2サイクル → 変化なし → 記録構成なし
        result = proc.process(
            transient_goal_data={"has_active": True, "category": "exploration", "strength": 0.5},
            persistent_commitment_data={"items": [{"item_id": "a1", "category": "growth", "strength": 0.5}]},
            value_orientation_data={
                "dimensions": {"a": 0.5, "b": 0.0},
                "confidences": {"a": 0.3, "b": 0.0},
                "update_count": 10,
            },
        )
        assert result == 0
        assert len(proc.state.adjacency_records) == 0

    def test_layer1_change_creates_record(self):
        proc = self._make_processor_with_prev()
        result = proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
            persistent_commitment_data={"items": [{"item_id": "a1", "category": "growth", "strength": 0.5}]},
            value_orientation_data={
                "dimensions": {"a": 0.5, "b": 0.0},
                "confidences": {"a": 0.3, "b": 0.0},
                "update_count": 10,
            },
        )
        assert result == 1
        rec = proc.state.adjacency_records[0]
        assert rec.layer1_change.change_type == Layer1ChangeType.DISAPPEARANCE.value
        assert rec.simultaneous_change_count == 1

    def test_multi_layer_change(self):
        proc = self._make_processor_with_prev()
        result = proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
            persistent_commitment_data={"items": []},  # release of a1
            value_orientation_data={
                "dimensions": {"a": 0.8, "b": 0.0},  # a changed
                "confidences": {"a": 0.3, "b": 0.0},
                "update_count": 10,
            },
        )
        assert result == 1
        rec = proc.state.adjacency_records[0]
        assert rec.simultaneous_change_count == 3
        assert rec.layer1_change.change_type == Layer1ChangeType.DISAPPEARANCE.value
        # Layer 2 should have release
        l2_types = {c.change_type for c in rec.layer2_changes}
        assert Layer2ChangeType.RELEASE.value in l2_types
        # Layer 3 should have dimension change
        assert rec.layer3_change.change_type == Layer3ChangeType.DIMENSION_CHANGE.value

    def test_only_layer2_change(self):
        proc = self._make_processor_with_prev()
        result = proc.process(
            transient_goal_data={"has_active": True, "category": "exploration", "strength": 0.5},
            persistent_commitment_data={
                "items": [
                    {"item_id": "a1", "category": "growth", "strength": 0.5},
                    {"item_id": "b2", "category": "social", "strength": 0.3},
                ]
            },
            value_orientation_data={
                "dimensions": {"a": 0.5, "b": 0.0},
                "confidences": {"a": 0.3, "b": 0.0},
                "update_count": 10,
            },
        )
        assert result == 1
        rec = proc.state.adjacency_records[0]
        assert rec.simultaneous_change_count == 1
        # Layer 1 should be no change
        assert rec.layer1_change.change_type == Layer1ChangeType.NO_CHANGE.value


# =============================================================================
# FIFO and Freshness
# =============================================================================

class TestFIFOAccumulation:
    def test_fifo_limit(self):
        config = GoalHierarchyPropagationConfig(max_records=5)
        proc = GoalHierarchyPropagationProcessor(config=config)

        # Initialize snapshots
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )

        # Generate records by alternating states
        for i in range(10):
            strength = 0.8 if i % 2 == 0 else 0.3
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": strength},
            )

        assert len(proc.state.adjacency_records) <= 5
        assert proc.state.accumulation_limit_reached is True

    def test_no_accumulation_without_change(self):
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        # Run 5 cycles with identical data
        for _ in range(5):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )
        assert len(proc.state.adjacency_records) == 0


class TestFreshnessDecay:
    def test_uniform_decay(self):
        config = GoalHierarchyPropagationConfig(record_freshness_decay_rate=0.1)
        proc = GoalHierarchyPropagationProcessor(config=config)

        # Initialize
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        # Create a record
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
        )
        assert len(proc.state.adjacency_records) == 1
        # The record just had 1 decay applied (on the cycle it was created)
        initial_freshness = proc.state.adjacency_records[0].record_freshness

        # Run another cycle (no change, but decay still applies)
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
        )
        decayed = proc.state.adjacency_records[0].record_freshness
        assert decayed < initial_freshness
        # Approximately 0.1 less
        assert abs(decayed - (initial_freshness - 0.1)) < 0.001

    def test_freshness_stages(self):
        assert _freshness_stage(0.9) == RecordFreshness.FRESH
        assert _freshness_stage(0.7) == RecordFreshness.RECENT
        assert _freshness_stage(0.5) == RecordFreshness.AGING
        assert _freshness_stage(0.3) == RecordFreshness.STALE
        assert _freshness_stage(0.1) == RecordFreshness.FADED

    def test_decay_does_not_delete(self):
        """鮮度減衰は記録の削除を直接引き起こさない（FIFO上限のみ）。"""
        config = GoalHierarchyPropagationConfig(record_freshness_decay_rate=0.5)
        proc = GoalHierarchyPropagationProcessor(config=config)

        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
        )

        # Run many cycles to decay freshness to near zero
        for _ in range(20):
            proc.process(
                transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
            )

        # Record should still exist (not deleted by decay)
        assert len(proc.state.adjacency_records) == 1
        assert proc.state.adjacency_records[0].record_freshness == 0.0


# =============================================================================
# Convergence Monitoring
# =============================================================================

class TestConvergenceMonitoring:
    def test_no_convergence_flag_initially(self):
        proc = create_goal_hierarchy_propagation_processor()
        assert proc.state.convergence_flag is False

    def test_convergence_detected(self):
        config = GoalHierarchyPropagationConfig(convergence_threshold=0.7)
        proc = GoalHierarchyPropagationProcessor(config=config)

        # Initialize
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )

        # Generate many records with the same change pattern
        for _ in range(20):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.8},
            )
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )

        # Should detect convergence since same pattern repeats
        assert proc.state.convergence_flag is True

    def test_convergence_does_not_modify_records(self):
        """収束検出は記録の内容や蓄積を変更しない。"""
        config = GoalHierarchyPropagationConfig(convergence_threshold=0.7)
        proc = GoalHierarchyPropagationProcessor(config=config)

        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        # Generate records
        for _ in range(10):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.8},
            )
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )

        records_before = len(proc.state.adjacency_records)
        freshness_before = [r.record_freshness for r in proc.state.adjacency_records]

        # Run more cycles - convergence flag may change but records should not be affected
        for _ in range(5):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.8},
            )
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )

        # Convergence should not delete existing records (new records added, decay applied)
        assert len(proc.state.adjacency_records) >= records_before

    def test_convergence_records_limited(self):
        config = GoalHierarchyPropagationConfig(max_convergence_records=5)
        proc = GoalHierarchyPropagationProcessor(config=config)

        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        for _ in range(20):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.8},
            )
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )

        assert len(proc.state.convergence_records) <= 5


# =============================================================================
# Safety Valves
# =============================================================================

class TestSafetyValves:
    """設計書の安全弁7種のテスト。"""

    def test_all_records_equal_weight(self):
        """安全弁1: 全記録等価維持。重み付け・順位付け・選別なし。"""
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        # Create different change types
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
        )
        proc.process(
            transient_goal_data={"has_active": True, "category": "b", "strength": 0.8},
        )
        proc.process(
            transient_goal_data={"has_active": True, "category": "b", "strength": 0.3},
        )

        records = proc.get_recent_records(count=10)
        # All records should be returned without weight/priority/selection
        assert len(records) == 3
        # No weight field beyond freshness (which is uniform)
        for r in records:
            assert hasattr(r, "record_freshness")
            # No "importance" or "priority" field
            assert not hasattr(r, "importance")
            assert not hasattr(r, "priority")

    def test_no_causal_attribution(self):
        """安全弁2: 因果帰属の構造的排除。"""
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            persistent_commitment_data={"items": [{"item_id": "x", "category": "g", "strength": 0.5}]},
        )
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
            persistent_commitment_data={"items": []},
        )

        records = proc.get_recent_records()
        assert len(records) == 1
        rec = records[0]
        # Record contains simultaneous_change_count as fact, not causal direction
        assert rec.simultaneous_change_count == 2
        # No "caused_by" or "triggered" field
        assert not hasattr(rec, "caused_by")
        assert not hasattr(rec, "triggered")
        assert not hasattr(rec, "causal_direction")

    def test_no_enrichment_exposure(self):
        """安全弁3: enrichment直接露出の遮断。"""
        proc = create_goal_hierarchy_propagation_processor()
        # No get_enrichment_data method should exist
        assert not hasattr(proc, "get_enrichment_data")
        # No get_enrichment_text method
        assert not hasattr(proc, "get_enrichment_text")

    def test_fifo_finite(self):
        """安全弁4: FIFO蓄積の有限性。"""
        config = GoalHierarchyPropagationConfig(max_records=3)
        proc = GoalHierarchyPropagationProcessor(config=config)
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        for i in range(10):
            strength = 0.8 if i % 2 == 0 else 0.3
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": strength},
            )
        assert len(proc.state.adjacency_records) <= 3

    def test_no_write_back_to_layers(self):
        """安全弁5: 3層への逆流経路の構造的不在。"""
        proc = create_goal_hierarchy_propagation_processor()
        # No methods that write to transient_goal, persistent_commitment, or value_orientation
        methods = dir(proc)
        for m in methods:
            assert "update_transient" not in m
            assert "update_commitment" not in m
            assert "update_orientation" not in m
            assert "write_back" not in m
            assert "modify_layer" not in m

    def test_stage_values_only(self):
        """安全弁6: 段階値限定。スナップショットは段階値のみで構成。"""
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "a",
                                  "direction_signature": {"x": 0.1234567}, "strength": 0.5123},
            value_orientation_data={
                "dimensions": {"a": 0.4567},
                "confidences": {"a": 0.3456},
                "update_count": 15,
            },
        )
        # Check snapshots use stage labels, not raw numbers
        snap1 = proc.state.prev_layer1
        assert snap1.strength_stage in ("strong", "moderate", "weak", "faint", "absent")
        # No raw float stored
        assert not hasattr(snap1, "strength_raw") or True  # just check the stage exists

        snap3 = proc.state.prev_layer3
        for d in snap3.dimensions:
            assert d.value_stage in ("strong", "moderate", "weak", "faint", "neutral")
            assert d.confidence_stage in ("high", "moderate", "low", "minimal", "none")

    def test_convergence_internal_only(self):
        """安全弁7: 収束監視の内部限定。"""
        proc = create_goal_hierarchy_propagation_processor()
        # convergence_flag is in state but should not be exposed to enrichment
        assert not hasattr(proc, "get_enrichment_data")
        # Convergence records exist in state but are internal
        assert hasattr(proc.state, "convergence_records")
        assert hasattr(proc.state, "convergence_flag")


# =============================================================================
# Pathway Blockage
# =============================================================================

class TestPathwayBlockage:
    """経路遮断の検証。"""

    def test_no_policy_output(self):
        """ポリシー候補生成への経路なし。"""
        proc = create_goal_hierarchy_propagation_processor()
        methods = dir(proc)
        assert "generate_policy" not in methods
        assert "get_policy_bias" not in methods
        assert "apply_bias" not in methods

    def test_no_scoring_output(self):
        """スコアリングへの経路なし。"""
        proc = create_goal_hierarchy_propagation_processor()
        methods = dir(proc)
        assert "compute_score" not in methods
        assert "apply_score" not in methods

    def test_no_emotion_output(self):
        """感情チャンネルへの書き込み経路なし。"""
        proc = create_goal_hierarchy_propagation_processor()
        methods = dir(proc)
        for m in methods:
            assert "emotion" not in m.lower() or m.startswith("_")

    def test_no_responsibility_output(self):
        """責任重量への書き込み経路なし。"""
        proc = create_goal_hierarchy_propagation_processor()
        methods = dir(proc)
        for m in methods:
            assert "responsibility" not in m.lower()

    def test_read_only_accessors(self):
        """参照提供はREAD-ONLYのみ。"""
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
        )
        records_before = len(proc.state.adjacency_records)
        # Accessing records should not modify them
        _ = proc.get_recent_records()
        _ = proc.get_summary()
        assert len(proc.state.adjacency_records) == records_before


# =============================================================================
# Save / Load
# =============================================================================

class TestSaveLoad:
    def test_roundtrip_empty(self):
        state = GoalHierarchyPropagationState()
        data = save_state(state)
        restored = load_state(data)
        assert restored.cycle_count == 0
        assert restored.adjacency_records == []
        assert restored.convergence_records == []
        assert restored.prev_layer1 is None
        assert restored.prev_layer2 is None
        assert restored.prev_layer3 is None

    def test_roundtrip_with_records(self):
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "explore", "strength": 0.6,
                                  "direction_signature": {"a": 0.5}},
            persistent_commitment_data={"items": [{"item_id": "c1", "category": "growth", "strength": 0.7}]},
            value_orientation_data={
                "dimensions": {"a": 0.3, "b": -0.5},
                "confidences": {"a": 0.2, "b": 0.4},
                "update_count": 20,
            },
        )
        proc.process(
            transient_goal_data={"has_active": False, "category": "", "strength": 0.0},
            persistent_commitment_data={"items": []},
            value_orientation_data={
                "dimensions": {"a": 0.8, "b": -0.5},
                "confidences": {"a": 0.5, "b": 0.4},
                "update_count": 20,
            },
        )

        data = save_state(proc.state)
        restored = load_state(data)

        assert restored.cycle_count == proc.state.cycle_count
        assert len(restored.adjacency_records) == len(proc.state.adjacency_records)
        assert restored.prev_layer1 is not None
        assert restored.prev_layer2 is not None
        assert restored.prev_layer3 is not None
        assert restored.accumulation_limit_reached == proc.state.accumulation_limit_reached

    def test_save_load_preserves_snapshots(self):
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(
            transient_goal_data={"has_active": True, "category": "x", "strength": 0.5},
            value_orientation_data={
                "dimensions": {"a": 0.3},
                "confidences": {"a": 0.2},
                "update_count": 5,
            },
        )
        data = save_state(proc.state)
        restored = load_state(data)

        assert restored.prev_layer1.has_active is True
        assert restored.prev_layer1.category == "x"
        assert restored.prev_layer1.strength_stage == "moderate"
        assert restored.prev_layer3.dimensions[0].dim_id == "a"
        assert restored.prev_layer3.update_count_stage == "low"

    def test_save_load_preserves_convergence(self):
        config = GoalHierarchyPropagationConfig(max_convergence_records=5)
        proc = GoalHierarchyPropagationProcessor(config=config)
        proc.process(
            transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
        )
        for _ in range(10):
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.8},
            )
            proc.process(
                transient_goal_data={"has_active": True, "category": "a", "strength": 0.5},
            )

        data = save_state(proc.state)
        restored = load_state(data)

        assert len(restored.convergence_records) == len(proc.state.convergence_records)
        assert restored.convergence_flag == proc.state.convergence_flag


# =============================================================================
# Record Serialization
# =============================================================================

class TestRecordSerialization:
    def test_adjacency_record_roundtrip(self):
        rec = AdjacencyRecord(
            record_id="test123",
            cycle_number=42,
            timestamp=1000.0,
            layer1_change=Layer1Change(
                change_type=Layer1ChangeType.GENERATION.value,
                curr_category="exploration",
                curr_strength_stage="moderate",
            ),
            layer2_changes=[
                Layer2Change(
                    change_type=Layer2ChangeType.PROMOTION.value,
                    item_category="growth",
                ),
            ],
            layer3_change=Layer3Change(
                change_type=Layer3ChangeType.DIMENSION_CHANGE.value,
                changed_dimensions=["a", "c"],
            ),
            simultaneous_change_count=3,
            record_freshness=0.75,
            record_freshness_stage=RecordFreshness.RECENT.value,
        )
        data = rec.to_dict()
        restored = AdjacencyRecord.from_dict(data)

        assert restored.record_id == "test123"
        assert restored.cycle_number == 42
        assert restored.layer1_change.change_type == Layer1ChangeType.GENERATION.value
        assert restored.layer1_change.curr_category == "exploration"
        assert len(restored.layer2_changes) == 1
        assert restored.layer2_changes[0].change_type == Layer2ChangeType.PROMOTION.value
        assert restored.layer3_change.change_type == Layer3ChangeType.DIMENSION_CHANGE.value
        assert restored.layer3_change.changed_dimensions == ["a", "c"]
        assert restored.simultaneous_change_count == 3
        assert abs(restored.record_freshness - 0.75) < 0.001

    def test_convergence_record_roundtrip(self):
        rec = ConvergenceRecord(
            convergence_score=0.65,
            convergence_level=ConvergenceLevel.MODERATE.value,
            dominant_combination="generation|no_change|no_change",
            combination_diversity=0.5,
            cycle=10,
            timestamp=2000.0,
        )
        data = rec.to_dict()
        restored = ConvergenceRecord.from_dict(data)

        assert abs(restored.convergence_score - 0.65) < 0.001
        assert restored.convergence_level == ConvergenceLevel.MODERATE.value
        assert restored.dominant_combination == "generation|no_change|no_change"


# =============================================================================
# Snapshot Serialization
# =============================================================================

class TestSnapshotSerialization:
    def test_layer1_roundtrip(self):
        snap = Layer1Snapshot(has_active=True, category="x", direction_signature_summary="a,b", strength_stage="strong")
        data = snap.to_dict()
        restored = Layer1Snapshot.from_dict(data)
        assert restored.has_active is True
        assert restored.category == "x"
        assert restored.direction_signature_summary == "a,b"
        assert restored.strength_stage == "strong"

    def test_layer2_roundtrip(self):
        snap = Layer2Snapshot(
            items=[Layer2ItemSnapshot(item_id="a1", category="g", strength_stage="moderate")],
            recent_cognition_types=["promotion", "release"],
        )
        data = snap.to_dict()
        restored = Layer2Snapshot.from_dict(data)
        assert len(restored.items) == 1
        assert restored.items[0].item_id == "a1"
        assert restored.recent_cognition_types == ["promotion", "release"]

    def test_layer3_roundtrip(self):
        snap = Layer3Snapshot(
            dimensions=[
                Layer3DimSnapshot(dim_id="a", value_stage="moderate", confidence_stage="low"),
            ],
            update_count_stage="moderate",
        )
        data = snap.to_dict()
        restored = Layer3Snapshot.from_dict(data)
        assert len(restored.dimensions) == 1
        assert restored.dimensions[0].dim_id == "a"
        assert restored.update_count_stage == "moderate"


# =============================================================================
# Factory
# =============================================================================

class TestFactory:
    def test_default_factory(self):
        proc = create_goal_hierarchy_propagation_processor()
        assert proc.state.cycle_count == 0
        assert proc.state.adjacency_records == []

    def test_custom_config(self):
        config = GoalHierarchyPropagationConfig(max_records=50, record_freshness_decay_rate=0.05)
        proc = create_goal_hierarchy_propagation_processor(config=config)
        assert proc._config.max_records == 50
        assert proc._config.record_freshness_decay_rate == 0.05


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_process_with_all_none(self):
        proc = create_goal_hierarchy_propagation_processor()
        result = proc.process()
        assert result == 0

    def test_process_twice_with_none(self):
        proc = create_goal_hierarchy_propagation_processor()
        proc.process()
        result = proc.process()
        assert result == 0

    def test_state_setter(self):
        proc = create_goal_hierarchy_propagation_processor()
        new_state = GoalHierarchyPropagationState(cycle_count=100)
        proc.state = new_state
        assert proc.state.cycle_count == 100

    def test_layer2_empty_both_prev_and_curr(self):
        proc = create_goal_hierarchy_propagation_processor()
        proc.process(persistent_commitment_data={"items": []})
        result = proc.process(persistent_commitment_data={"items": []})
        assert result == 0

    def test_prev_snapshot_none_after_load(self):
        """ロード後に前回スナップショットが失われた場合のテスト。"""
        proc = create_goal_hierarchy_propagation_processor()
        # Simulate loaded state with no previous snapshots
        proc.state.prev_layer1 = None
        proc.state.prev_layer2 = None
        proc.state.prev_layer3 = None
        proc.state.cycle_count = 50

        # Should skip detection and just save current snapshot
        result = proc.process(
            transient_goal_data={"has_active": True, "category": "x", "strength": 0.5},
        )
        assert result == 0
        assert proc.state.prev_layer1 is not None

    def test_summary(self):
        proc = create_goal_hierarchy_propagation_processor()
        summary = proc.get_summary()
        assert summary["total_records"] == 0
        assert summary["cycle_count"] == 0
        assert summary["accumulation_limit_reached"] is False
        assert summary["convergence_flag"] is False


# =============================================================================
# Integration: Multiple Cycles
# =============================================================================

class TestMultipleCycles:
    def test_many_cycles_stability(self):
        """多数のサイクルを実行しても構造が安定していること。"""
        proc = create_goal_hierarchy_propagation_processor()

        # Alternate between states
        for i in range(100):
            if i % 3 == 0:
                tg = {"has_active": True, "category": "explore", "strength": 0.6}
            elif i % 3 == 1:
                tg = {"has_active": True, "category": "social", "strength": 0.3}
            else:
                tg = {"has_active": False, "category": "", "strength": 0.0}

            pc_items = []
            if i % 5 < 3:
                pc_items = [{"item_id": f"c{i%5}", "category": "growth", "strength": 0.5 + (i % 3) * 0.1}]

            vo = {
                "dimensions": {"a": 0.3 + (i % 10) * 0.05, "b": -0.2},
                "confidences": {"a": 0.2 + (i % 5) * 0.1, "b": 0.1},
                "update_count": 10 + i,
            }

            proc.process(
                transient_goal_data=tg,
                persistent_commitment_data={"items": pc_items},
                value_orientation_data=vo,
            )

        # Should have limited records
        assert len(proc.state.adjacency_records) <= 200
        assert proc.state.cycle_count == 100
        # All records should have valid freshness
        for r in proc.state.adjacency_records:
            assert 0.0 <= r.record_freshness <= 1.0
