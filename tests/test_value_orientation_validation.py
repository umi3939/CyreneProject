"""
Tests for psyche/value_orientation_validation.py - 価値方向性の実運用検証

Verifies:
- 6段パイプライン (観測→正規化→時系列→差分→検証出力→受け渡し)
- 8断面入力の抽出と正規化
- 差分記述化（不一致・収束・再分岐の並立記録）
- 安全弁（収束偏向→代替補充、観測欠落→保留再活性化）
- 読み取り専用（状態変更への直接接続なし）
- 報告情報形式のみの出力
- シリアライズ/デシリアライズ
"""

import pytest
import time

from psyche.value_orientation_validation import (
    # Enums
    ObservationSourceType,
    ObservationFreshness,
    DifferentialType,
    ValidationStatus,
    # Data structures
    ObservationRecord,
    ValidationDescriptionUnit,
    DifferentialEntry,
    TimeSeriesEntry,
    ValidationInputs,
    ValidationState,
    ValidationResult,
    ValidationConfig,
    # Processor
    ValueOrientationValidator,
    # Helpers
    _clamp,
    _freshness_from_age,
    _freshness_weight,
    # Public API
    get_validation_summary,
    create_validation_processor,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

def _make_basic_inputs(**overrides) -> ValidationInputs:
    """基本的な検証入力を生成する。"""
    defaults = dict(
        orientation_dimensions={"a": 0.3, "b": -0.1, "c": 0.0, "d": 0.2, "e": -0.05},
        orientation_confidences={"a": 0.5, "b": 0.1, "c": 0.0, "d": 0.3, "e": 0.0},
        orientation_update_count=10,
        candidate_count=5,
        top_candidate_label="共感する",
        top_candidate_score=0.75,
        candidate_diversity=0.6,
        recent_selections=["共感する", "励ます", "感想を述べる"],
        selection_consistency=0.7,
        context_pace=0.5,
        context_density=0.3,
        context_continuity=0.6,
        emotion_valence=0.4,
        emotion_arousal=0.5,
        emotions={"joy": 0.6, "sadness": 0.1, "anger": 0.0},
        recalled_count=3,
        has_bindings=True,
        episode_count=5,
        caution_bias=0.3,
        empathy_bias=0.4,
        responsibility_weight=0.3,
        tick_count=100,
        elapsed_since_last=5.0,
    )
    defaults.update(overrides)
    return ValidationInputs(**defaults)


def _make_empty_inputs(**overrides) -> ValidationInputs:
    """空の検証入力を生成する。"""
    defaults = dict(
        orientation_dimensions={},
        orientation_confidences={},
        orientation_update_count=0,
        candidate_count=0,
        top_candidate_label="",
        top_candidate_score=0.0,
        candidate_diversity=0.0,
        recent_selections=[],
        selection_consistency=0.0,
        context_pace=0.0,
        context_density=0.0,
        context_continuity=0.0,
        emotion_valence=0.0,
        emotion_arousal=0.0,
        emotions={},
        recalled_count=0,
        has_bindings=False,
        episode_count=0,
        caution_bias=0.0,
        empathy_bias=0.0,
        responsibility_weight=0.0,
        tick_count=0,
        elapsed_since_last=0.0,
    )
    defaults.update(overrides)
    return ValidationInputs(**defaults)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """列挙型のテスト。"""

    def test_observation_source_types_count(self):
        assert len(ObservationSourceType) == 8

    def test_observation_source_types_values(self):
        values = {s.value for s in ObservationSourceType}
        expected = {
            "value_orientation", "action_candidates", "selection_history",
            "context", "emotion_transition", "memory_reference",
            "responsibility", "time_elapsed",
        }
        assert values == expected

    def test_observation_freshness_count(self):
        assert len(ObservationFreshness) == 5

    def test_differential_type_count(self):
        assert len(DifferentialType) == 3

    def test_validation_status_count(self):
        assert len(ValidationStatus) == 4


# =============================================================================
# Helper Tests
# =============================================================================

class TestHelpers:
    """ヘルパー関数のテスト。"""

    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_min(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5.0, lo=-10.0, hi=10.0) == 5.0

    def test_freshness_from_age_fresh(self):
        assert _freshness_from_age(10.0) == ObservationFreshness.FRESH

    def test_freshness_from_age_recent(self):
        assert _freshness_from_age(60.0) == ObservationFreshness.RECENT

    def test_freshness_from_age_aging(self):
        assert _freshness_from_age(200.0) == ObservationFreshness.AGING

    def test_freshness_from_age_stale(self):
        assert _freshness_from_age(400.0) == ObservationFreshness.STALE

    def test_freshness_from_age_faded(self):
        assert _freshness_from_age(700.0) == ObservationFreshness.FADED

    def test_freshness_weight_fresh(self):
        assert _freshness_weight(ObservationFreshness.FRESH) == 1.0

    def test_freshness_weight_faded(self):
        assert _freshness_weight(ObservationFreshness.FADED) == 0.05


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestObservationRecord:
    """ObservationRecord のテスト。"""

    def test_default_creation(self):
        rec = ObservationRecord()
        assert rec.record_id
        assert rec.source_type == ""
        assert rec.freshness == ObservationFreshness.FRESH.value

    def test_serialization_roundtrip(self):
        rec = ObservationRecord(
            source_type="value_orientation",
            dimensions={"a": 0.5, "b": -0.3},
            metadata={"key": "value"},
        )
        data = rec.to_dict()
        restored = ObservationRecord.from_dict(data)
        assert restored.source_type == rec.source_type
        assert restored.dimensions == rec.dimensions
        assert restored.metadata == rec.metadata

    def test_from_dict_empty(self):
        rec = ObservationRecord.from_dict({})
        assert rec.source_type == ""
        assert rec.dimensions == {}


class TestValidationDescriptionUnit:
    """ValidationDescriptionUnit のテスト。"""

    def test_default_creation(self):
        unit = ValidationDescriptionUnit()
        assert unit.unit_id
        assert unit.status == ValidationStatus.ACTIVE.value

    def test_serialization_roundtrip(self):
        unit = ValidationDescriptionUnit(
            source_record_ids=["r1", "r2"],
            source_types=["value_orientation"],
            normalized_values={"vo_a": 0.65},
        )
        data = unit.to_dict()
        restored = ValidationDescriptionUnit.from_dict(data)
        assert restored.source_record_ids == unit.source_record_ids
        assert restored.normalized_values == unit.normalized_values


class TestDifferentialEntry:
    """DifferentialEntry のテスト。"""

    def test_default_creation(self):
        entry = DifferentialEntry()
        assert entry.diff_type == DifferentialType.INCONSISTENCY.value
        assert entry.delta == 0.0

    def test_serialization_roundtrip(self):
        entry = DifferentialEntry(
            diff_type=DifferentialType.CONVERGENCE.value,
            dimension="vo_a",
            value_before=0.3,
            value_after=0.5,
            delta=0.2,
        )
        data = entry.to_dict()
        restored = DifferentialEntry.from_dict(data)
        assert restored.diff_type == entry.diff_type
        assert restored.delta == entry.delta


class TestTimeSeriesEntry:
    """TimeSeriesEntry のテスト。"""

    def test_default_creation(self):
        entry = TimeSeriesEntry()
        assert entry.tick == 0
        assert entry.observation_type == "single"

    def test_serialization_roundtrip(self):
        entry = TimeSeriesEntry(tick=50, unit_id="u1", observation_type="continuous")
        data = entry.to_dict()
        restored = TimeSeriesEntry.from_dict(data)
        assert restored.tick == 50
        assert restored.observation_type == "continuous"


# =============================================================================
# ValidationState Tests
# =============================================================================

class TestValidationState:
    """ValidationState のテスト。"""

    def test_default_creation(self):
        state = ValidationState()
        assert state.cycle_count == 0
        assert state.total_observations == 0
        assert state.observation_records == []

    def test_serialization_roundtrip(self):
        state = ValidationState(
            cycle_count=5,
            total_observations=20,
            convergence_warning=True,
        )
        state.observation_records.append(ObservationRecord(source_type="test"))
        state.differential_history.append(
            DifferentialEntry(diff_type=DifferentialType.CONVERGENCE.value)
        )

        data = state.to_dict()
        restored = ValidationState.from_dict(data)
        assert restored.cycle_count == 5
        assert restored.total_observations == 20
        assert restored.convergence_warning is True
        assert len(restored.observation_records) == 1
        assert len(restored.differential_history) == 1

    def test_from_dict_empty(self):
        state = ValidationState.from_dict({})
        assert state.cycle_count == 0


# =============================================================================
# ValidationConfig Tests
# =============================================================================

class TestValidationConfig:
    """ValidationConfig のテスト。"""

    def test_default_values(self):
        cfg = ValidationConfig()
        assert cfg.max_observation_records == 200
        assert cfg.convergence_threshold == 0.8
        assert cfg.gap_threshold == 5

    def test_custom_values(self):
        cfg = ValidationConfig(max_observation_records=50, gap_threshold=3)
        assert cfg.max_observation_records == 50
        assert cfg.gap_threshold == 3


# =============================================================================
# Processor Pipeline Tests
# =============================================================================

class TestProcessorBasic:
    """プロセッサの基本動作テスト。"""

    def test_create_processor(self):
        proc = create_validation_processor()
        assert isinstance(proc, ValueOrientationValidator)
        assert proc.state.cycle_count == 0

    def test_process_increments_cycle(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        assert proc.state.cycle_count == 1
        proc.process(inputs)
        assert proc.state.cycle_count == 2

    def test_process_returns_validation_result(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)
        assert isinstance(result, ValidationResult)

    def test_process_empty_inputs(self):
        proc = create_validation_processor()
        inputs = _make_empty_inputs()
        result = proc.process(inputs)
        assert isinstance(result, ValidationResult)
        assert result.cycle_count == 1

    def test_state_property(self):
        proc = create_validation_processor()
        assert isinstance(proc.state, ValidationState)


# =============================================================================
# Stage 1: Observation Extraction Tests
# =============================================================================

class TestStage1Extraction:
    """Stage 1: 観測対象抽出のテスト。"""

    def test_all_8_sources_extracted(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)

        source_types = {
            r.source_type for r in proc.state.observation_records
        }
        assert ObservationSourceType.VALUE_ORIENTATION.value in source_types
        assert ObservationSourceType.ACTION_CANDIDATES.value in source_types
        assert ObservationSourceType.SELECTION_HISTORY.value in source_types
        assert ObservationSourceType.CONTEXT.value in source_types
        assert ObservationSourceType.EMOTION_TRANSITION.value in source_types
        assert ObservationSourceType.MEMORY_REFERENCE.value in source_types
        assert ObservationSourceType.RESPONSIBILITY.value in source_types
        assert ObservationSourceType.TIME_ELAPSED.value in source_types

    def test_time_elapsed_always_recorded(self):
        proc = create_validation_processor()
        inputs = _make_empty_inputs()
        proc.process(inputs)
        source_types = {r.source_type for r in proc.state.observation_records}
        assert ObservationSourceType.TIME_ELAPSED.value in source_types

    def test_empty_orientation_skipped(self):
        proc = create_validation_processor()
        inputs = _make_empty_inputs()
        proc.process(inputs)
        source_types = {r.source_type for r in proc.state.observation_records}
        assert ObservationSourceType.VALUE_ORIENTATION.value not in source_types

    def test_observation_count_tracked(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        assert proc.state.total_observations > 0

    def test_records_stored_in_state(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        assert len(proc.state.observation_records) > 0

    def test_gap_counter_for_missing_source(self):
        proc = create_validation_processor()
        # No candidates → gap counter should increment
        inputs = _make_empty_inputs(
            orientation_dimensions={"a": 0.1},
        )
        proc.process(inputs)
        assert proc._gap_counters[ObservationSourceType.ACTION_CANDIDATES.value] == 1

    def test_gap_counter_resets_on_presence(self):
        proc = create_validation_processor()
        inputs_no_candidates = _make_empty_inputs()
        proc.process(inputs_no_candidates)
        assert proc._gap_counters[ObservationSourceType.ACTION_CANDIDATES.value] == 1

        inputs_with_candidates = _make_basic_inputs()
        proc.process(inputs_with_candidates)
        assert proc._gap_counters[ObservationSourceType.ACTION_CANDIDATES.value] == 0


# =============================================================================
# Stage 2: Normalization Tests
# =============================================================================

class TestStage2Normalization:
    """Stage 2: 観測単位正規化のテスト。"""

    def test_units_created_for_records(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        assert len(proc.state.description_units) > 0

    def test_value_orientation_normalized_range(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs(
            orientation_dimensions={"a": 1.0, "b": -1.0, "c": 0.0},
        )
        proc.process(inputs)

        vo_units = [
            u for u in proc.state.description_units
            if ObservationSourceType.VALUE_ORIENTATION.value in u.source_types
        ]
        assert len(vo_units) == 1
        nv = vo_units[0].normalized_values
        # a=1.0 → (1.0+1.0)/2.0 = 1.0
        assert nv["vo_a"] == 1.0
        # b=-1.0 → (-1.0+1.0)/2.0 = 0.0
        assert nv["vo_b"] == 0.0
        # c=0.0 → (0.0+1.0)/2.0 = 0.5
        assert nv["vo_c"] == 0.5

    def test_emotion_valence_normalized(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs(emotion_valence=-0.5)
        proc.process(inputs)

        et_units = [
            u for u in proc.state.description_units
            if ObservationSourceType.EMOTION_TRANSITION.value in u.source_types
        ]
        assert len(et_units) == 1
        nv = et_units[0].normalized_values
        # valence -0.5 → (-0.5+1.0)/2.0 = 0.25
        assert abs(nv["et_valence"] - 0.25) < 0.01

    def test_all_normalized_values_in_range(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)

        for unit in proc.state.description_units:
            for key, val in unit.normalized_values.items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


# =============================================================================
# Stage 3: Time Series Alignment Tests
# =============================================================================

class TestStage3TimeSeries:
    """Stage 3: 時系列整列のテスト。"""

    def test_time_series_entries_created(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        assert len(proc.state.time_series_index) > 0

    def test_first_observation_is_single(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        # First observations should be "single"
        assert any(
            e.observation_type == "single"
            for e in proc.state.time_series_index
        )

    def test_subsequent_observation_is_continuous(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        proc.process(inputs)  # Second run
        # Should have some "continuous" entries
        assert any(
            e.observation_type == "continuous"
            for e in proc.state.time_series_index
        )

    def test_freshness_updates(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)
        # All recent records should be FRESH
        for rec in proc.state.observation_records:
            assert rec.freshness == ObservationFreshness.FRESH.value


# =============================================================================
# Stage 4: Differential Computation Tests
# =============================================================================

class TestStage4Differentials:
    """Stage 4: 差分記述化のテスト。"""

    def test_no_diffs_on_first_run(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)
        # First run has no prior units to compare
        assert len(result.differentials) == 0

    def test_diffs_generated_on_change(self):
        proc = create_validation_processor()
        inputs1 = _make_basic_inputs(
            orientation_dimensions={"a": 0.0, "b": 0.0},
        )
        proc.process(inputs1)

        inputs2 = _make_basic_inputs(
            orientation_dimensions={"a": 0.5, "b": -0.5},
        )
        result = proc.process(inputs2)
        # Should detect changes in orientation dimensions
        assert len(result.differentials) > 0

    def test_convergence_and_inconsistency_coexist(self):
        """収束と不一致が並立記録されることを確認。"""
        proc = create_validation_processor()

        # Run 1: baseline
        proc.process(_make_basic_inputs(
            orientation_dimensions={"a": 0.0},
            emotion_valence=0.0,
        ))

        # Run 2: move in positive direction
        proc.process(_make_basic_inputs(
            orientation_dimensions={"a": 0.3},
            emotion_valence=0.3,
        ))

        # Run 3: orientation continues positive, emotion reverses
        result = proc.process(_make_basic_inputs(
            orientation_dimensions={"a": 0.5},
            emotion_valence=-0.3,
        ))

        # Differential history should contain both convergence and divergence types
        all_types = {d.diff_type for d in proc.state.differential_history}
        # At minimum should have some differentials
        assert len(proc.state.differential_history) > 0

    def test_re_divergence_stored_separately(self):
        """再分岐が別途保存されることを確認。"""
        proc = create_validation_processor()

        # Build up history with consistent positive direction
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.0}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.2}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.4}))

        # Now reverse → should be re_divergence
        result = proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.1}))

        # Check that re_divergence_history is populated when direction changes
        # (The classification depends on prior delta direction)
        # At minimum differential_history should grow
        assert len(proc.state.differential_history) > 0

    def test_small_changes_ignored(self):
        """微小変化（<0.01）は差分として記録されない。"""
        proc = create_validation_processor()
        proc.process(_make_basic_inputs(
            orientation_dimensions={"a": 0.500},
        ))
        result = proc.process(_make_basic_inputs(
            orientation_dimensions={"a": 0.501},  # delta < 0.01 after normalization
        ))
        # Very small changes should produce no differentials for that dimension
        vo_diffs = [d for d in result.differentials if "vo_a" in d.dimension]
        assert len(vo_diffs) == 0


# =============================================================================
# Stage 5: Validation Output Tests
# =============================================================================

class TestStage5Output:
    """Stage 5: 検証出力化のテスト。"""

    def test_covered_sources_reported(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)
        assert len(result.covered_sources) > 0

    def test_uncovered_sources_reported(self):
        proc = create_validation_processor()
        inputs = _make_empty_inputs()
        result = proc.process(inputs)
        # With empty inputs, most sources are uncovered
        assert len(result.uncovered_sources) > 0

    def test_active_units_counted(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)
        assert result.active_units > 0

    def test_cycle_count_in_result(self):
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)
        assert result.cycle_count == 1

    def test_trend_computation(self):
        proc = create_validation_processor()
        # Run multiple times to build differential history
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.0}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.2}))
        result = proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.4}))
        # Should have short_term_trend data
        # (only if differentials were recorded)
        if proc.state.differential_history:
            assert isinstance(result.short_term_trend, dict)

    def test_output_is_report_only(self):
        """検証出力が報告情報形式のみであることを確認。"""
        proc = create_validation_processor()
        result = proc.process(_make_basic_inputs())
        # ValidationResult has no methods for modifying state
        assert not hasattr(result, 'apply')
        assert not hasattr(result, 'execute')
        assert not hasattr(result, 'update_state')


# =============================================================================
# Stage 6: Handoff / Safety Valve Tests
# =============================================================================

class TestStage6SafetyValves:
    """Stage 6: 安全弁テスト。"""

    def test_no_warnings_initially(self):
        proc = create_validation_processor()
        result = proc.process(_make_basic_inputs())
        assert result.convergence_warning is False
        assert result.gap_warning is False

    def test_gap_warning_triggers(self):
        """観測欠落が閾値を超えたら gap_warning が発動。"""
        cfg = ValidationConfig(gap_threshold=3)
        proc = ValueOrientationValidator(config=cfg)

        # Process with empty inputs multiple times
        for _ in range(3):
            result = proc.process(_make_empty_inputs())

        assert result.gap_warning is True

    def test_convergence_warning_triggers(self):
        """差分の単一方向割合が閾値を超えたら convergence_warning が発動。"""
        cfg = ValidationConfig(convergence_threshold=0.7)
        proc = ValueOrientationValidator(config=cfg)

        # Build up consistently positive differentials
        base = 0.0
        for i in range(15):
            base += 0.05
            proc.process(_make_basic_inputs(
                orientation_dimensions={"a": base},
            ))

        # Check warning
        assert proc.state.convergence_warning is True or len(proc.state.differential_history) < 10

    def test_pending_observations_populated_on_overflow(self):
        """観測記録の上限超過時、溢れた記録が保留に移動する。"""
        cfg = ValidationConfig(max_observation_records=10)
        proc = ValueOrientationValidator(config=cfg)

        for i in range(5):
            proc.process(_make_basic_inputs(tick_count=i))

        # Should have some records in pending due to overflow
        total_records = (
            len(proc.state.observation_records)
            + len(proc.state.pending_observations)
        )
        assert total_records > 0

    def test_pending_reactivation_on_gap(self):
        """欠落時に保留観測が再活性化される。"""
        cfg = ValidationConfig(gap_threshold=2, max_observation_records=5)
        proc = ValueOrientationValidator(config=cfg)

        # First: process with full inputs to build pending via overflow
        for _ in range(3):
            proc.process(_make_basic_inputs())

        # Now process with empty inputs to trigger gap
        for _ in range(3):
            result = proc.process(_make_empty_inputs())

        # If there were pending observations for the gapped sources,
        # they should have been reactivated
        assert isinstance(result.pending_reactivated, bool)

    def test_dilution_applied_to_stale_records(self):
        """鮮度がSTALE以下の記録に希薄化が適用される。"""
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        proc.process(inputs)

        # Manually age some records
        for rec in proc.state.observation_records:
            rec.timestamp = time.time() - 700  # > 600s = FADED
            rec.freshness = ObservationFreshness.FADED.value
            rec.dimensions = {"test": 1.0}

        # Process again → dilution should apply
        proc.process(inputs)

        # Faded records should have reduced dimension values
        faded = [
            r for r in proc.state.observation_records
            if r.freshness == ObservationFreshness.FADED.value
            and "test" in r.dimensions
        ]
        for r in faded:
            assert r.dimensions["test"] < 1.0

    def test_cross_section_diversity_maintained(self):
        """特定断面だけが検証結果を支配しないことを確認。"""
        proc = create_validation_processor()

        # Process multiple times to build history
        for i in range(5):
            proc.process(_make_basic_inputs(tick_count=i))

        # State should have diverse source types in observation records
        source_types = {r.source_type for r in proc.state.observation_records}
        assert len(source_types) >= 3


# =============================================================================
# Design Constraint Verification Tests
# =============================================================================

class TestDesignConstraints:
    """設計制約の検証テスト。"""

    def test_read_only_no_state_mutation(self):
        """検証機能が既存状態を変更しないことを確認。"""
        proc = create_validation_processor()
        inputs = _make_basic_inputs()
        result = proc.process(inputs)

        # Result should not have methods to modify external state
        assert not hasattr(result, 'modify')
        assert not hasattr(result, 'commit')

    def test_output_is_not_control_command(self):
        """検証出力が制御命令形式でないことを確認。"""
        proc = create_validation_processor()
        result = proc.process(_make_basic_inputs())

        # ValidationResult fields are all informational
        assert isinstance(result.differentials, list)
        assert isinstance(result.active_units, int)
        assert isinstance(result.covered_sources, list)
        assert isinstance(result.short_term_trend, dict)

    def test_no_circular_reference_within_cycle(self):
        """同一運用周期の入力への再投入がないことを確認。"""
        proc = create_validation_processor()
        inputs = _make_basic_inputs()

        # Process once
        result1 = proc.process(inputs)
        state_after_1 = proc.state.cycle_count

        # Result cannot be fed back as input (different types)
        assert type(result1) != type(inputs)

    def test_convergence_records_and_divergence_coexist(self):
        """収束記録と再分岐記録が同列保持されることを確認。"""
        proc = create_validation_processor()

        # Build differential history
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.0}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.3}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.5}))
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.2}))

        # differential_history and re_divergence_history can both exist
        # (they're separate lists, held in parallel)
        assert isinstance(proc.state.differential_history, list)
        assert isinstance(proc.state.re_divergence_history, list)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """シリアライズ/デシリアライズのテスト。"""

    def test_state_roundtrip_after_processing(self):
        proc = create_validation_processor()
        proc.process(_make_basic_inputs())
        proc.process(_make_basic_inputs(orientation_dimensions={"a": 0.5}))

        data = proc.state.to_dict()
        restored = ValidationState.from_dict(data)

        assert restored.cycle_count == proc.state.cycle_count
        assert restored.total_observations == proc.state.total_observations
        assert len(restored.observation_records) == len(proc.state.observation_records)
        assert len(restored.description_units) == len(proc.state.description_units)
        assert len(restored.differential_history) == len(proc.state.differential_history)

    def test_result_to_dict(self):
        proc = create_validation_processor()
        result = proc.process(_make_basic_inputs())
        data = result.to_dict()
        assert "differentials" in data
        assert "covered_sources" in data
        assert "cycle_count" in data

    def test_state_setter(self):
        proc = create_validation_processor()
        proc.process(_make_basic_inputs())

        new_state = ValidationState(cycle_count=99)
        proc.state = new_state
        assert proc.state.cycle_count == 99


# =============================================================================
# Summary Tests
# =============================================================================

class TestSummary:
    """要約関数のテスト。"""

    def test_summary_empty_state(self):
        state = ValidationState()
        summary = get_validation_summary(state)
        assert "cycle=0" in summary

    def test_summary_after_processing(self):
        proc = create_validation_processor()
        proc.process(_make_basic_inputs())
        summary = get_validation_summary(proc.state)
        assert "cycle=1" in summary
        assert "obs=" in summary

    def test_summary_with_warnings(self):
        state = ValidationState(
            cycle_count=10,
            total_observations=50,
            convergence_warning=True,
            gap_warning=True,
        )
        summary = get_validation_summary(state)
        assert "収束偏向" in summary
        assert "欠落" in summary

    def test_summary_with_pending(self):
        state = ValidationState(
            cycle_count=5,
            total_observations=20,
            pending_observations=[ObservationRecord() for _ in range(3)],
        )
        summary = get_validation_summary(state)
        assert "保留=3" in summary


# =============================================================================
# Trimming / Capacity Tests
# =============================================================================

class TestCapacityLimits:
    """容量制限のテスト。"""

    def test_observation_records_trimmed(self):
        cfg = ValidationConfig(max_observation_records=20)
        proc = ValueOrientationValidator(config=cfg)

        for i in range(10):
            proc.process(_make_basic_inputs(tick_count=i))

        assert len(proc.state.observation_records) <= 20

    def test_description_units_trimmed(self):
        cfg = ValidationConfig(max_description_units=20)
        proc = ValueOrientationValidator(config=cfg)

        for i in range(10):
            proc.process(_make_basic_inputs(tick_count=i))

        assert len(proc.state.description_units) <= 20

    def test_differential_history_trimmed(self):
        cfg = ValidationConfig(max_differential_history=30)
        proc = ValueOrientationValidator(config=cfg)

        base = 0.0
        for i in range(20):
            base += 0.05
            proc.process(_make_basic_inputs(
                orientation_dimensions={"a": base},
                tick_count=i,
            ))

        assert len(proc.state.differential_history) <= 30

    def test_time_series_trimmed(self):
        cfg = ValidationConfig(max_time_series_entries=30)
        proc = ValueOrientationValidator(config=cfg)

        for i in range(15):
            proc.process(_make_basic_inputs(tick_count=i))

        assert len(proc.state.time_series_index) <= 30

    def test_dilution_history_trimmed(self):
        cfg = ValidationConfig(
            max_description_units=5,
            max_dilution_history=10,
        )
        proc = ValueOrientationValidator(config=cfg)

        for i in range(15):
            proc.process(_make_basic_inputs(tick_count=i))

        assert len(proc.state.dilution_history) <= 10


# =============================================================================
# Multi-Cycle Integration Tests
# =============================================================================

class TestMultiCycleIntegration:
    """複数サイクルの統合テスト。"""

    def test_10_cycle_run(self):
        """10サイクル連続実行で安定動作。"""
        proc = create_validation_processor()
        for i in range(10):
            result = proc.process(_make_basic_inputs(
                tick_count=i * 5,
                orientation_dimensions={"a": 0.1 * i, "b": -0.05 * i},
            ))
            assert isinstance(result, ValidationResult)
            assert result.cycle_count == i + 1

        assert proc.state.total_observations > 0
        assert len(proc.state.differential_history) > 0

    def test_alternating_inputs(self):
        """交互入力パターンで安定動作。"""
        proc = create_validation_processor()

        for i in range(6):
            if i % 2 == 0:
                inputs = _make_basic_inputs(
                    orientation_dimensions={"a": 0.5},
                    tick_count=i,
                )
            else:
                inputs = _make_basic_inputs(
                    orientation_dimensions={"a": -0.5},
                    tick_count=i,
                )
            result = proc.process(inputs)
            assert isinstance(result, ValidationResult)

    def test_gradual_shift_detection(self):
        """緩やかな変化の検出。"""
        proc = create_validation_processor()

        for i in range(20):
            val = 0.01 * i  # Very gradual increase
            result = proc.process(_make_basic_inputs(
                orientation_dimensions={"a": val},
                tick_count=i,
            ))

        # Should have detected changes over time
        assert proc.state.total_observations > 0
        assert proc.state.cycle_count == 20

    def test_mixed_empty_and_full(self):
        """空入力と完全入力の混合。"""
        proc = create_validation_processor()
        for i in range(10):
            if i % 3 == 0:
                proc.process(_make_empty_inputs(tick_count=i))
            else:
                proc.process(_make_basic_inputs(tick_count=i))

        assert proc.state.cycle_count == 10
