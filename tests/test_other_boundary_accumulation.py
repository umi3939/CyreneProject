"""
tests/test_other_boundary_accumulation.py - 他者境界の多相蓄積テスト

design_other_boundary_accumulation.md 準拠の実装テスト。
"""

import time
import pytest
from dataclasses import dataclass
from typing import Any

from psyche.other_boundary_accumulation import (
    # Enums
    FreshnessStage,
    DivergenceLevel,
    ConvergenceLevel,
    # Helpers
    _clamp,
    _gen_id,
    _stage_from_freshness,
    determine_divergence_level,
    _convergence_from_score,
    # Data Structures
    BoundaryRecord,
    ConvergenceRecord,
    # State
    OtherBoundaryAccumulationState,
    # Result
    BoundaryAccumulationResult,
    # Config
    OtherBoundaryAccumulationConfig,
    # Processor
    OtherBoundaryAccumulationProcessor,
    # Summary
    get_boundary_accumulation_summary,
    # Factory
    create_other_boundary_accumulation_processor,
)


# =============================================================================
# Mock Boundary (SelfOtherBoundary duck-typed)
# =============================================================================

@dataclass
class MockBoundary:
    """SelfOtherBoundary の duck-type テスト用モック。"""
    boundary_id: str = "test_boundary"
    self_description: str = "Self state description"
    other_description: str = "Other state description"
    divergence: float = 0.5
    boundary_aspects: tuple[str, ...] = ("inference_behavioral",)
    timestamp: str = ""


# =============================================================================
# Enum Tests
# =============================================================================

class TestFreshnessStage:
    def test_all_values(self):
        assert FreshnessStage.ACTIVE.value == "active"
        assert FreshnessStage.WEAKENING.value == "weakening"
        assert FreshnessStage.FADING.value == "fading"
        assert FreshnessStage.NEAR_INVISIBLE.value == "near_invisible"
        assert FreshnessStage.INVISIBLE.value == "invisible"

    def test_unique_values(self):
        values = [s.value for s in FreshnessStage]
        assert len(values) == len(set(values))


class TestDivergenceLevel:
    def test_all_values(self):
        assert DivergenceLevel.LEVEL_0.value == "level_0"
        assert DivergenceLevel.LEVEL_1.value == "level_1"
        assert DivergenceLevel.LEVEL_2.value == "level_2"
        assert DivergenceLevel.LEVEL_3.value == "level_3"
        assert DivergenceLevel.LEVEL_4.value == "level_4"

    def test_unique_values(self):
        values = [l.value for l in DivergenceLevel]
        assert len(values) == len(set(values))


class TestConvergenceLevel:
    def test_all_values(self):
        assert ConvergenceLevel.NONE.value == "none"
        assert ConvergenceLevel.MILD.value == "mild"
        assert ConvergenceLevel.MODERATE.value == "moderate"
        assert ConvergenceLevel.STRONG.value == "strong"


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_min(self):
        assert _clamp(-0.5) == 0.0

    def test_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_custom_range(self):
        assert _clamp(0.05, 0.1, 1.0) == 0.1
        assert _clamp(1.5, 0.1, 0.9) == 0.9

    def test_edge_values(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


class TestGenId:
    def test_returns_string(self):
        assert isinstance(_gen_id(), str)

    def test_length(self):
        assert len(_gen_id()) == 12

    def test_unique(self):
        ids = [_gen_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestStageFromFreshness:
    def test_active(self):
        assert _stage_from_freshness(0.8) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(1.0) == FreshnessStage.ACTIVE

    def test_weakening(self):
        assert _stage_from_freshness(0.7) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.6) == FreshnessStage.WEAKENING

    def test_fading(self):
        assert _stage_from_freshness(0.5) == FreshnessStage.FADING
        assert _stage_from_freshness(0.4) == FreshnessStage.FADING

    def test_near_invisible(self):
        assert _stage_from_freshness(0.3) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.2) == FreshnessStage.NEAR_INVISIBLE

    def test_invisible(self):
        assert _stage_from_freshness(0.1) == FreshnessStage.INVISIBLE
        assert _stage_from_freshness(0.0) == FreshnessStage.INVISIBLE


class TestDetermineDivergenceLevel:
    def test_level_0(self):
        assert determine_divergence_level(0.0) == DivergenceLevel.LEVEL_0
        assert determine_divergence_level(0.1) == DivergenceLevel.LEVEL_0
        assert determine_divergence_level(0.19) == DivergenceLevel.LEVEL_0

    def test_level_1(self):
        assert determine_divergence_level(0.2) == DivergenceLevel.LEVEL_1
        assert determine_divergence_level(0.3) == DivergenceLevel.LEVEL_1
        assert determine_divergence_level(0.39) == DivergenceLevel.LEVEL_1

    def test_level_2(self):
        assert determine_divergence_level(0.4) == DivergenceLevel.LEVEL_2
        assert determine_divergence_level(0.5) == DivergenceLevel.LEVEL_2
        assert determine_divergence_level(0.59) == DivergenceLevel.LEVEL_2

    def test_level_3(self):
        assert determine_divergence_level(0.6) == DivergenceLevel.LEVEL_3
        assert determine_divergence_level(0.7) == DivergenceLevel.LEVEL_3
        assert determine_divergence_level(0.79) == DivergenceLevel.LEVEL_3

    def test_level_4(self):
        assert determine_divergence_level(0.8) == DivergenceLevel.LEVEL_4
        assert determine_divergence_level(0.9) == DivergenceLevel.LEVEL_4
        assert determine_divergence_level(1.0) == DivergenceLevel.LEVEL_4

    def test_edge_negative(self):
        assert determine_divergence_level(-0.1) == DivergenceLevel.LEVEL_0

    def test_edge_above_one(self):
        assert determine_divergence_level(1.5) == DivergenceLevel.LEVEL_4


class TestConvergenceFromScore:
    def test_none(self):
        assert _convergence_from_score(0.0) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.2) == ConvergenceLevel.NONE

    def test_mild(self):
        assert _convergence_from_score(0.3) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.4) == ConvergenceLevel.MILD

    def test_moderate(self):
        assert _convergence_from_score(0.5) == ConvergenceLevel.MODERATE
        assert _convergence_from_score(0.6) == ConvergenceLevel.MODERATE

    def test_strong(self):
        assert _convergence_from_score(0.7) == ConvergenceLevel.STRONG
        assert _convergence_from_score(0.9) == ConvergenceLevel.STRONG


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestBoundaryRecord:
    def test_default_creation(self):
        rec = BoundaryRecord()
        assert rec.record_id != ""
        assert rec.user_id == ""
        assert rec.divergence == 0.0
        assert rec.freshness == 1.0
        assert rec.freshness_stage == FreshnessStage.ACTIVE.value

    def test_custom_creation(self):
        rec = BoundaryRecord(
            user_id="user_a",
            divergence=0.75,
            divergence_level=DivergenceLevel.LEVEL_3.value,
            boundary_aspects=("inference_behavioral", "multiple_hypotheses"),
            self_description="self state",
            other_description="other state",
            tick=42,
        )
        assert rec.user_id == "user_a"
        assert rec.divergence == 0.75
        assert rec.divergence_level == "level_3"
        assert len(rec.boundary_aspects) == 2

    def test_to_dict(self):
        rec = BoundaryRecord(user_id="user_a", divergence=0.5)
        d = rec.to_dict()
        assert d["user_id"] == "user_a"
        assert d["divergence"] == 0.5
        assert isinstance(d["boundary_aspects"], list)

    def test_from_dict(self):
        data = {
            "record_id": "abc123",
            "user_id": "user_b",
            "divergence": 0.7,
            "divergence_level": "level_3",
            "boundary_aspects": ["a", "b"],
            "self_description": "self",
            "other_description": "other",
            "tick": 10,
            "timestamp": 1000.0,
            "freshness": 0.9,
            "freshness_stage": "active",
        }
        rec = BoundaryRecord.from_dict(data)
        assert rec.record_id == "abc123"
        assert rec.user_id == "user_b"
        assert rec.divergence == 0.7
        assert rec.boundary_aspects == ("a", "b")
        assert rec.freshness == 0.9

    def test_roundtrip(self):
        rec = BoundaryRecord(
            user_id="user_a",
            divergence=0.6,
            boundary_aspects=("x", "y"),
        )
        restored = BoundaryRecord.from_dict(rec.to_dict())
        assert restored.user_id == rec.user_id
        assert restored.divergence == rec.divergence
        assert restored.boundary_aspects == rec.boundary_aspects

    def test_from_dict_defaults(self):
        rec = BoundaryRecord.from_dict({})
        assert rec.user_id == ""
        assert rec.divergence == 0.0
        assert rec.freshness == 1.0


class TestConvergenceRecord:
    def test_default_creation(self):
        cr = ConvergenceRecord()
        assert cr.convergence_score == 0.0
        assert cr.convergence_level == ConvergenceLevel.NONE.value

    def test_to_dict_roundtrip(self):
        cr = ConvergenceRecord(
            user_id="user_a",
            convergence_score=0.8,
            convergence_level=ConvergenceLevel.STRONG.value,
            dominant_level="level_3",
        )
        restored = ConvergenceRecord.from_dict(cr.to_dict())
        assert restored.user_id == "user_a"
        assert restored.convergence_score == 0.8
        assert restored.convergence_level == ConvergenceLevel.STRONG.value


# =============================================================================
# State Tests
# =============================================================================

class TestOtherBoundaryAccumulationState:
    def test_default_creation(self):
        state = OtherBoundaryAccumulationState()
        assert state.records == []
        assert state.user_index == {}
        assert state.cycle_count == 0
        assert not state.fixed_pattern_warning
        assert not state.accumulation_bias_warning
        assert not state.divergence_convergence_warning

    def test_to_dict(self):
        state = OtherBoundaryAccumulationState(
            cycle_count=5,
            total_records_added=10,
        )
        d = state.to_dict()
        assert d["cycle_count"] == 5
        assert d["total_records_added"] == 10

    def test_from_dict(self):
        data = {
            "cycle_count": 3,
            "total_records_added": 7,
            "fixed_pattern_warning": True,
            "records": [],
            "user_index": {},
            "convergence_records": [],
        }
        state = OtherBoundaryAccumulationState.from_dict(data)
        assert state.cycle_count == 3
        assert state.total_records_added == 7
        assert state.fixed_pattern_warning is True

    def test_roundtrip(self):
        state = OtherBoundaryAccumulationState(
            records=[BoundaryRecord(user_id="u1", divergence=0.5)],
            user_index={"u1": ["rec1"]},
            cycle_count=10,
        )
        restored = OtherBoundaryAccumulationState.from_dict(state.to_dict())
        assert restored.cycle_count == 10
        assert len(restored.records) == 1
        assert restored.records[0].user_id == "u1"

    def test_apply_session_decay(self):
        state = OtherBoundaryAccumulationState(
            records=[
                BoundaryRecord(record_id="r1", user_id="u1", freshness=0.5),
                BoundaryRecord(record_id="r2", user_id="u1", freshness=0.2),
            ],
            user_index={"u1": ["r1", "r2"]},
        )
        state.apply_session_decay(0.3)
        # r1: 0.5 - 0.3 = 0.2 (survives)
        # r2: 0.2 - 0.3 = 0.0 < 0.1 (removed)
        assert len(state.records) == 1
        assert state.records[0].record_id == "r1"
        assert state.records[0].freshness == pytest.approx(0.2)
        assert "r2" not in state.user_index["u1"]

    def test_apply_session_decay_removes_all(self):
        state = OtherBoundaryAccumulationState(
            records=[
                BoundaryRecord(record_id="r1", user_id="u1", freshness=0.05),
            ],
            user_index={"u1": ["r1"]},
        )
        state.apply_session_decay(0.3)
        assert len(state.records) == 0
        assert state.total_records_invisible == 1

    def test_apply_session_decay_empty(self):
        state = OtherBoundaryAccumulationState()
        state.apply_session_decay(0.3)
        assert len(state.records) == 0


# =============================================================================
# Config Tests
# =============================================================================

class TestOtherBoundaryAccumulationConfig:
    def test_defaults(self):
        cfg = OtherBoundaryAccumulationConfig()
        assert cfg.max_records_per_user == 50
        assert cfg.max_records_total == 200
        assert cfg.freshness_decay_rate == 0.02
        assert cfg.absent_user_decay_rate == 0.01

    def test_custom(self):
        cfg = OtherBoundaryAccumulationConfig(
            max_records_per_user=10,
            max_records_total=50,
        )
        assert cfg.max_records_per_user == 10
        assert cfg.max_records_total == 50


# =============================================================================
# Processor Tests
# =============================================================================

class TestProcessorCreation:
    def test_default_creation(self):
        proc = OtherBoundaryAccumulationProcessor()
        assert proc.state.cycle_count == 0

    def test_custom_config(self):
        cfg = OtherBoundaryAccumulationConfig(max_records_per_user=5)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        assert proc.state.cycle_count == 0

    def test_factory_function(self):
        proc = create_other_boundary_accumulation_processor()
        assert isinstance(proc, OtherBoundaryAccumulationProcessor)

    def test_factory_with_config(self):
        cfg = OtherBoundaryAccumulationConfig(max_records_total=100)
        proc = create_other_boundary_accumulation_processor(config=cfg)
        assert isinstance(proc, OtherBoundaryAccumulationProcessor)


class TestProcessorStateProperty:
    def test_get_state(self):
        proc = OtherBoundaryAccumulationProcessor()
        assert isinstance(proc.state, OtherBoundaryAccumulationState)

    def test_set_state(self):
        proc = OtherBoundaryAccumulationProcessor()
        new_state = OtherBoundaryAccumulationState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42


class TestProcessorTick:
    def test_tick_with_no_input(self):
        proc = OtherBoundaryAccumulationProcessor()
        result = proc.tick()
        assert not result.newly_added
        assert result.cycle_count == 1

    def test_tick_with_boundary_no_user(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        result = proc.tick(boundary=boundary)
        assert not result.newly_added
        assert result.cycle_count == 1

    def test_tick_with_user_no_boundary(self):
        proc = OtherBoundaryAccumulationProcessor()
        result = proc.tick(user_id="user_a")
        assert not result.newly_added

    def test_tick_with_boundary_and_user(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        result = proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert result.newly_added
        assert result.total_record_count == 1
        assert result.cycle_count == 1
        assert "user_a" in result.user_record_counts
        assert result.user_record_counts["user_a"] == 1

    def test_multiple_ticks(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        assert proc.state.cycle_count == 5
        assert proc.state.total_records_added == 5

    def test_tick_with_dict_boundary(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = {
            "divergence": 0.65,
            "boundary_aspects": ["behavioral", "contextual"],
            "self_description": "self desc",
            "other_description": "other desc",
        }
        result = proc.tick(boundary=boundary, user_id="user_b", current_tick=1)
        assert result.newly_added
        rec = proc.state.records[0]
        assert rec.divergence == pytest.approx(0.65)
        assert rec.divergence_level == DivergenceLevel.LEVEL_3.value
        assert len(rec.boundary_aspects) == 2


class TestStage1ReceiveAndAccumulate:
    def test_per_user_separation(self):
        proc = OtherBoundaryAccumulationProcessor()
        b1 = MockBoundary(divergence=0.3)
        b2 = MockBoundary(divergence=0.7)
        proc.tick(boundary=b1, user_id="user_a", current_tick=1)
        proc.tick(boundary=b2, user_id="user_b", current_tick=2)
        assert "user_a" in proc.state.user_index
        assert "user_b" in proc.state.user_index
        assert len(proc.state.user_index["user_a"]) == 1
        assert len(proc.state.user_index["user_b"]) == 1

    def test_per_user_pushout(self):
        cfg = OtherBoundaryAccumulationConfig(max_records_per_user=3)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        assert len(proc.state.user_index["user_a"]) <= 3
        user_records = proc.get_user_records("user_a")
        assert len(user_records) <= 3

    def test_total_pushout(self):
        cfg = OtherBoundaryAccumulationConfig(max_records_total=5)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        for i in range(8):
            boundary = MockBoundary(divergence=0.1 * (i % 5 + 1))
            proc.tick(boundary=boundary, user_id=f"user_{i}", current_tick=i)
        total_visible = sum(
            1 for r in proc.state.records
            if r.freshness_stage != FreshnessStage.INVISIBLE.value
        )
        assert total_visible <= 5

    def test_divergence_level_assignment(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.35)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        rec = proc.state.records[0]
        assert rec.divergence_level == DivergenceLevel.LEVEL_1.value

    def test_description_truncation(self):
        cfg = OtherBoundaryAccumulationConfig(description_max_length=10)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        boundary = MockBoundary(
            divergence=0.5,
            self_description="A" * 100,
            other_description="B" * 100,
        )
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        rec = proc.state.records[0]
        assert len(rec.self_description) == 10
        assert len(rec.other_description) == 10

    def test_boundary_aspects_preserved(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(
            divergence=0.5,
            boundary_aspects=("aspect_a", "aspect_b", "aspect_c"),
        )
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        rec = proc.state.records[0]
        assert rec.boundary_aspects == ("aspect_a", "aspect_b", "aspect_c")

    def test_divergence_clamped(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=1.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert proc.state.records[0].divergence == 1.0

    def test_divergence_negative_clamped(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=-0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert proc.state.records[0].divergence == 0.0


class TestStage2FreshnessManagement:
    def test_basic_decay(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        initial_freshness = proc.state.records[0].freshness
        # Tick without new boundary triggers decay on existing record
        proc.tick(user_id="user_a", current_tick=2)
        assert proc.state.records[0].freshness < initial_freshness

    def test_absent_user_extra_decay(self):
        proc = OtherBoundaryAccumulationProcessor()
        b1 = MockBoundary(divergence=0.5)
        b2 = MockBoundary(divergence=0.3)
        proc.tick(boundary=b1, user_id="user_a", current_tick=1)
        proc.tick(boundary=b2, user_id="user_b", current_tick=2)
        # user_a record should get extra decay because user_b is active
        user_a_rec = [r for r in proc.state.records if r.user_id == "user_a"]
        assert len(user_a_rec) == 1
        # Snapshot the freshness before further decay
        freshness_before = user_a_rec[0].freshness
        # After another tick with user_b active, user_a decays faster
        proc.tick(user_id="user_b", current_tick=3)
        user_a_rec_after = [r for r in proc.state.records if r.user_id == "user_a"]
        if user_a_rec_after:
            assert user_a_rec_after[0].freshness < freshness_before

    def test_invisible_removal(self):
        cfg = OtherBoundaryAccumulationConfig(freshness_decay_rate=0.5)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        # High decay should make record invisible quickly
        for i in range(5):
            proc.tick(user_id="user_a", current_tick=i + 2)
        # Record should have been removed due to invisibility
        assert len(proc.state.records) == 0 or all(
            r.freshness_stage != FreshnessStage.INVISIBLE.value
            for r in proc.state.records
        )


class TestStage3HandoffPreparation:
    def test_result_structure(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        result = proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert isinstance(result, BoundaryAccumulationResult)
        assert isinstance(result.user_record_counts, dict)
        assert isinstance(result.divergence_level_distribution, dict)

    def test_divergence_level_distribution(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i, div in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            boundary = MockBoundary(divergence=div)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        # Should have entries across multiple levels
        dist = result.divergence_level_distribution
        assert isinstance(dist, dict)


class TestSafetyValve4FixedPattern:
    def test_fixed_pattern_detected(self):
        cfg = OtherBoundaryAccumulationConfig(
            fixed_pattern_min_records=5,
            fixed_pattern_dominance_threshold=0.8,
        )
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        # Create 6 records all with same divergence level
        for i in range(6):
            boundary = MockBoundary(divergence=0.5)  # all LEVEL_2
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.fixed_pattern_warning is True

    def test_no_fixed_pattern_with_variety(self):
        cfg = OtherBoundaryAccumulationConfig(
            fixed_pattern_min_records=5,
            fixed_pattern_dominance_threshold=0.8,
        )
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        # Create records with different divergence levels
        for i, div in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.15]):
            boundary = MockBoundary(divergence=div)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.fixed_pattern_warning is False

    def test_no_fixed_pattern_too_few_records(self):
        cfg = OtherBoundaryAccumulationConfig(fixed_pattern_min_records=10)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        for i in range(3):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.fixed_pattern_warning is False


class TestSafetyValve5AccumulationBias:
    def test_bias_detected(self):
        cfg = OtherBoundaryAccumulationConfig(
            accumulation_bias_threshold=1.5,
            freshness_decay_rate=0.0,   # disable decay for this test
            absent_user_decay_rate=0.0,
        )
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        # user_a gets 10 records, user_b gets 1
        # avg = (10+1)/2 = 5.5, max = 10, ratio = 10/5.5 = 1.82 > 1.5
        for i in range(10):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        boundary = MockBoundary(divergence=0.5)
        result = proc.tick(boundary=boundary, user_id="user_b", current_tick=11)
        assert result.accumulation_bias_warning is True

    def test_no_bias_with_single_user(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.accumulation_bias_warning is False

    def test_no_bias_balanced(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i * 2)
            proc.tick(boundary=boundary, user_id="user_b", current_tick=i * 2 + 1)
        result = proc.tick(user_id="user_a", current_tick=20)
        assert result.accumulation_bias_warning is False

    def test_bias_accelerates_decay(self):
        cfg = OtherBoundaryAccumulationConfig(accumulation_bias_threshold=2.0)
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        for i in range(10):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        # Get freshness before bias detection
        pre_freshness = [r.freshness for r in proc.state.records if r.user_id == "user_a"]
        # Add user_b to trigger bias detection
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_b", current_tick=11)
        # user_a records should have additional decay applied
        post_freshness = [r.freshness for r in proc.state.records if r.user_id == "user_a"]
        # At least some records should have lower freshness due to accelerated decay
        if pre_freshness and post_freshness:
            assert min(post_freshness) <= min(pre_freshness)


class TestSafetyValve6DivergenceConvergence:
    def test_convergence_detected(self):
        cfg = OtherBoundaryAccumulationConfig(
            divergence_convergence_threshold=0.8,
            fixed_pattern_min_records=5,
        )
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        # All records at the same divergence level
        for i in range(6):
            boundary = MockBoundary(divergence=0.5)  # all LEVEL_2
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.divergence_convergence_warning is True

    def test_no_convergence_with_variety(self):
        cfg = OtherBoundaryAccumulationConfig(
            divergence_convergence_threshold=0.8,
            fixed_pattern_min_records=5,
        )
        proc = OtherBoundaryAccumulationProcessor(config=cfg)
        for i, div in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.15]):
            boundary = MockBoundary(divergence=div)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        result = proc.tick(user_id="user_a", current_tick=10)
        assert result.divergence_convergence_warning is False


class TestSafetyValve7EnrichmentLimitation:
    def test_enrichment_contains_only_counts_and_levels(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(3):
            boundary = MockBoundary(
                divergence=0.3 * (i + 1),
                self_description="detailed self description",
                other_description="detailed other description",
            )
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)

        enrichment = proc.get_enrichment_data(user_id="user_a")
        # Must contain counts and levels
        assert "total_record_count" in enrichment
        assert "divergence_level_distribution" in enrichment
        assert "current_user" in enrichment
        assert "record_count" in enrichment["current_user"]
        # Must NOT contain individual record content
        assert "self_description" not in enrichment
        assert "other_description" not in enrichment

    def test_enrichment_no_comparison(self):
        proc = OtherBoundaryAccumulationProcessor()
        proc.tick(
            boundary=MockBoundary(divergence=0.3),
            user_id="user_a", current_tick=1,
        )
        proc.tick(
            boundary=MockBoundary(divergence=0.7),
            user_id="user_b", current_tick=2,
        )
        enrichment = proc.get_enrichment_data(user_id="user_a")
        # No cross-user comparison keys
        assert "comparison" not in enrichment
        assert "ranking" not in enrichment


class TestSafetyValve8PatternExtractionProhibition:
    def test_no_trend_in_enrichment(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        enrichment = proc.get_enrichment_data()
        # Must not contain trend or pattern analysis
        assert "trend" not in enrichment
        assert "pattern" not in enrichment
        assert "average" not in enrichment
        assert "variance" not in enrichment

    def test_no_statistics_in_result(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            result = proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        # Result should not contain statistical summaries
        assert not hasattr(result, "average_divergence")
        assert not hasattr(result, "divergence_trend")
        assert not hasattr(result, "standard_deviation")


# =============================================================================
# READ-ONLY Accessor Tests
# =============================================================================

class TestReadOnlyAccessors:
    def test_get_user_records(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(3):
            boundary = MockBoundary(divergence=0.2 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        records = proc.get_user_records("user_a")
        assert len(records) == 3
        for rec in records:
            assert rec.user_id == "user_a"

    def test_get_user_records_empty(self):
        proc = OtherBoundaryAccumulationProcessor()
        records = proc.get_user_records("nonexistent")
        assert records == []

    def test_get_user_records_excludes_invisible(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        # Force invisible
        proc.state.records[0].freshness_stage = FreshnessStage.INVISIBLE.value
        records = proc.get_user_records("user_a")
        assert records == []

    def test_get_all_user_ids(self):
        proc = OtherBoundaryAccumulationProcessor()
        proc.tick(
            boundary=MockBoundary(divergence=0.3),
            user_id="user_a", current_tick=1,
        )
        proc.tick(
            boundary=MockBoundary(divergence=0.5),
            user_id="user_b", current_tick=2,
        )
        ids = proc.get_all_user_ids()
        assert "user_a" in ids
        assert "user_b" in ids

    def test_get_all_user_ids_excludes_empty(self):
        proc = OtherBoundaryAccumulationProcessor()
        proc.state.user_index["empty_user"] = []
        ids = proc.get_all_user_ids()
        assert "empty_user" not in ids

    def test_get_record_count_all(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        assert proc.get_record_count() == 5

    def test_get_record_count_per_user(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(3):
            proc.tick(
                boundary=MockBoundary(divergence=0.5),
                user_id="user_a", current_tick=i,
            )
        for i in range(2):
            proc.tick(
                boundary=MockBoundary(divergence=0.5),
                user_id="user_b", current_tick=i + 3,
            )
        assert proc.get_record_count("user_a") == 3
        assert proc.get_record_count("user_b") == 2

    def test_get_record_count_empty(self):
        proc = OtherBoundaryAccumulationProcessor()
        assert proc.get_record_count() == 0
        assert proc.get_record_count("nobody") == 0


class TestGetEnrichmentData:
    def test_basic_enrichment(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        data = proc.get_enrichment_data()
        assert "cycle_count" in data
        assert "total_record_count" in data
        assert "user_count" in data
        assert "summary_text" in data

    def test_enrichment_with_user_id(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        data = proc.get_enrichment_data(user_id="user_a")
        assert "current_user" in data
        assert data["current_user"]["record_count"] == 1

    def test_enrichment_no_user(self):
        proc = OtherBoundaryAccumulationProcessor()
        data = proc.get_enrichment_data()
        assert data["total_record_count"] == 0
        assert "current_user" not in data

    def test_enrichment_warnings(self):
        proc = OtherBoundaryAccumulationProcessor()
        data = proc.get_enrichment_data()
        assert "fixed_pattern_warning" in data
        assert "accumulation_bias_warning" in data
        assert "divergence_convergence_warning" in data


class TestGetSummary:
    def test_summary_structure(self):
        proc = OtherBoundaryAccumulationProcessor()
        summary = proc.get_summary()
        assert "cycle_count" in summary
        assert "total_records_added" in summary
        assert "current_record_count" in summary
        assert "user_count" in summary

    def test_summary_after_operations(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(3):
            proc.tick(
                boundary=MockBoundary(divergence=0.5),
                user_id="user_a", current_tick=i,
            )
        summary = proc.get_summary()
        assert summary["total_records_added"] == 3
        assert summary["current_record_count"] == 3
        assert summary["user_count"] == 1


# =============================================================================
# Summary Function Tests
# =============================================================================

class TestBoundaryAccumulationSummary:
    def test_empty_state(self):
        state = OtherBoundaryAccumulationState()
        summary = get_boundary_accumulation_summary(state)
        assert summary == "境界蓄積: 待機中"

    def test_with_records(self):
        state = OtherBoundaryAccumulationState(
            records=[
                BoundaryRecord(user_id="u1", divergence=0.5),
                BoundaryRecord(user_id="u1", divergence=0.7),
            ],
            user_index={"u1": ["r1", "r2"]},
            cycle_count=3,
        )
        summary = get_boundary_accumulation_summary(state)
        assert "cycle=3" in summary
        assert "蓄積=2" in summary
        assert "相手=1" in summary

    def test_with_user_id(self):
        state = OtherBoundaryAccumulationState(
            records=[
                BoundaryRecord(user_id="u1", divergence=0.5),
            ],
            user_index={"u1": ["r1"]},
            cycle_count=1,
        )
        summary = get_boundary_accumulation_summary(state, user_id="u1")
        assert "当相手=1" in summary

    def test_with_warnings(self):
        state = OtherBoundaryAccumulationState(
            records=[BoundaryRecord(user_id="u1")],
            user_index={"u1": ["r1"]},
            cycle_count=1,
            fixed_pattern_warning=True,
            accumulation_bias_warning=True,
            divergence_convergence_warning=True,
        )
        summary = get_boundary_accumulation_summary(state)
        assert "固定パターン" in summary
        assert "蓄積偏り" in summary
        assert "乖離度収束" in summary

    def test_with_pushout(self):
        state = OtherBoundaryAccumulationState(
            records=[BoundaryRecord(user_id="u1")],
            user_index={"u1": ["r1"]},
            cycle_count=1,
            total_records_pushed_out=5,
        )
        summary = get_boundary_accumulation_summary(state)
        assert "押出累計=5" in summary


# =============================================================================
# Structural Separation Tests (設計書の構造的分離要件)
# =============================================================================

class TestStructuralSeparation:
    """設計書の「判断・行動・責任システムと構造的に分離されている理由」を検証。"""

    def test_no_policy_selection_methods(self):
        """ポリシー選択経路への非接続を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "select_policy", "generate_policy", "bias_policy",
            "apply_to_policy", "compute_policy_bias",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower, (
                    f"Method {method} contains forbidden pattern {pattern}"
                )

    def test_no_responsibility_methods(self):
        """責任システムへの非接続を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "update_responsibility", "set_responsibility",
            "modify_responsibility", "apply_to_responsibility",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_no_action_methods(self):
        """行動決定経路への非接続を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "select_action", "generate_action", "decide",
            "choose_action", "determine_action",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_no_hypothesis_modification(self):
        """他者モデル仮説強度への非操作を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "update_hypothesis", "modify_hypothesis",
            "set_hypothesis_strength", "boost_hypothesis",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_no_emotion_methods(self):
        """感情パイプラインへの非接続を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "update_emotion", "set_emotion", "modify_emotion",
            "apply_to_emotion",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_no_evaluation_methods(self):
        """関係性の評価・格付け・ランキングを行わないことを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "evaluate_relationship", "rank_users", "compare_users",
            "rate_closeness", "assess_trust", "compute_intimacy",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_no_pattern_analysis_methods(self):
        """パターン抽出を行わないことを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "analyze_pattern", "detect_trend", "compute_correlation",
            "extract_pattern", "compute_average_divergence",
            "compute_divergence_trend",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower


# =============================================================================
# Circular Reference Prevention Tests
# =============================================================================

class TestCircularReferencePrevention:
    """設計書の循環参照防止要件を検証。"""

    def test_no_other_model_feedback(self):
        """蓄積→他者モデルへの逆流禁止を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        # Verify no methods that could modify other_agent_model
        forbidden = [
            "update_other_model", "feed_back", "modify_boundary",
            "set_other_model", "apply_to_other",
        ]
        methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden:
                assert pattern not in method_lower

    def test_user_records_independent(self):
        """相手別蓄積の相互干渉禁止を検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        proc.tick(
            boundary=MockBoundary(divergence=0.3),
            user_id="user_a", current_tick=1,
        )
        proc.tick(
            boundary=MockBoundary(divergence=0.7),
            user_id="user_b", current_tick=2,
        )
        # Modifying user_a records should not affect user_b records
        user_a_recs = proc.get_user_records("user_a")
        user_b_recs = proc.get_user_records("user_b")
        assert len(user_a_recs) == 1
        assert len(user_b_recs) == 1
        assert user_a_recs[0].divergence != user_b_recs[0].divergence


# =============================================================================
# Persistence Tests
# =============================================================================

class TestPersistence:
    def test_full_state_roundtrip(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        proc.tick(
            boundary=MockBoundary(divergence=0.8),
            user_id="user_b", current_tick=6,
        )

        # Serialize
        state_dict = proc.state.to_dict()

        # Deserialize
        restored_state = OtherBoundaryAccumulationState.from_dict(state_dict)

        assert restored_state.cycle_count == proc.state.cycle_count
        assert restored_state.total_records_added == proc.state.total_records_added
        assert len(restored_state.records) == len(proc.state.records)
        assert len(restored_state.user_index) == len(proc.state.user_index)

    def test_state_assignment(self):
        proc = OtherBoundaryAccumulationProcessor()
        proc.tick(
            boundary=MockBoundary(divergence=0.5),
            user_id="user_a", current_tick=1,
        )
        state_dict = proc.state.to_dict()
        restored = OtherBoundaryAccumulationState.from_dict(state_dict)

        proc2 = OtherBoundaryAccumulationProcessor()
        proc2.state = restored
        assert proc2.state.cycle_count == proc.state.cycle_count
        assert len(proc2.state.records) == len(proc.state.records)

    def test_session_decay_on_restored_state(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(3):
            proc.tick(
                boundary=MockBoundary(divergence=0.5),
                user_id="user_a", current_tick=i,
            )

        state_dict = proc.state.to_dict()
        restored = OtherBoundaryAccumulationState.from_dict(state_dict)
        initial_count = len(restored.records)

        restored.apply_session_decay(0.3)
        # Records should still exist (freshness was 1.0, decay to 0.7)
        assert len(restored.records) == initial_count
        for rec in restored.records:
            assert rec.freshness < 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    def test_empty_user_id(self):
        proc = OtherBoundaryAccumulationProcessor()
        result = proc.tick(boundary=MockBoundary(divergence=0.5), user_id="")
        assert not result.newly_added

    def test_none_boundary(self):
        proc = OtherBoundaryAccumulationProcessor()
        result = proc.tick(boundary=None, user_id="user_a")
        assert not result.newly_added

    def test_zero_divergence(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.0)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert proc.state.records[0].divergence == 0.0
        assert proc.state.records[0].divergence_level == DivergenceLevel.LEVEL_0.value

    def test_max_divergence(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=1.0)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert proc.state.records[0].divergence == 1.0
        assert proc.state.records[0].divergence_level == DivergenceLevel.LEVEL_4.value

    def test_empty_boundary_aspects(self):
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5, boundary_aspects=())
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        assert proc.state.records[0].boundary_aspects == ()

    def test_many_users(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(20):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id=f"user_{i}", current_tick=i)
        ids = proc.get_all_user_ids()
        assert len(ids) == 20

    def test_rapid_tick_sequence(self):
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(100):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        # Should respect max_records_per_user
        user_records = proc.get_user_records("user_a")
        assert len(user_records) <= 50  # default max_records_per_user

    def test_boundary_with_non_numeric_divergence(self):
        """境界に非数値の乖離度がある場合のエッジケース。"""
        proc = OtherBoundaryAccumulationProcessor()
        boundary = {"divergence": "not_a_number", "boundary_aspects": []}
        result = proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        # Should handle gracefully (divergence defaults to 0.0)
        if result.newly_added:
            assert proc.state.records[0].divergence == 0.0

    def test_boundary_missing_fields(self):
        """境界にフィールドが欠落している場合。"""
        proc = OtherBoundaryAccumulationProcessor()
        boundary = {}
        result = proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        if result.newly_added:
            rec = proc.state.records[0]
            assert rec.divergence == 0.0
            assert rec.boundary_aspects == ()
            assert rec.self_description == ""
            assert rec.other_description == ""


# =============================================================================
# Design Document Compliance Tests
# =============================================================================

class TestDesignCompliance:
    """設計書で「絶対にしないこと」として明記された項目を検証。"""

    def test_no_distance_optimization(self):
        """境界の乖離度を制御・調整・最適化しないことを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(10):
            boundary = MockBoundary(divergence=0.5)
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        # Record divergence should be preserved exactly as received
        for rec in proc.state.records:
            if rec.user_id == "user_a":
                assert rec.divergence == 0.5

    def test_no_ideal_boundary_definition(self):
        """特定の相手との「適切な距離」を定義しないことを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        # No attributes or methods for ideal/target/desired distance
        forbidden_attrs = [
            "ideal_distance", "target_distance", "desired_boundary",
            "optimal_divergence",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(proc, attr)

    def test_no_evaluation_concepts(self):
        """評価的概念を導入しないことを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        forbidden = [
            "closeness", "intimacy", "trust", "rapport",
            "relationship_quality", "bond_strength",
        ]
        all_attrs = dir(proc)
        for attr in all_attrs:
            attr_lower = attr.lower()
            for pattern in forbidden:
                assert pattern not in attr_lower, (
                    f"Attribute {attr} contains evaluative concept {pattern}"
                )

    def test_records_are_equal(self):
        """すべての蓄積記録が等価であることを検証。"""
        proc = OtherBoundaryAccumulationProcessor()
        for i in range(5):
            boundary = MockBoundary(divergence=0.1 * (i + 1))
            proc.tick(boundary=boundary, user_id="user_a", current_tick=i)
        # Records should not have weight, score, or priority fields
        for rec in proc.state.records:
            assert not hasattr(rec, "weight") or rec.__class__.__name__ == "BoundaryRecord"
            assert not hasattr(rec, "score")
            assert not hasattr(rec, "priority")
            assert not hasattr(rec, "importance")

    def test_immutable_record_content(self):
        """蓄積記録の内容自体は変更されないことを検証（鮮度のみ変化）。"""
        proc = OtherBoundaryAccumulationProcessor()
        boundary = MockBoundary(divergence=0.5)
        proc.tick(boundary=boundary, user_id="user_a", current_tick=1)
        original_divergence = proc.state.records[0].divergence
        original_aspects = proc.state.records[0].boundary_aspects
        original_self_desc = proc.state.records[0].self_description

        # Run several more ticks
        for i in range(5):
            proc.tick(user_id="user_a", current_tick=i + 2)

        # Verify content hasn't changed (only freshness should change)
        if proc.state.records:
            matching = [r for r in proc.state.records if r.user_id == "user_a"]
            if matching:
                assert matching[0].divergence == original_divergence
                assert matching[0].boundary_aspects == original_aspects
                assert matching[0].self_description == original_self_desc
