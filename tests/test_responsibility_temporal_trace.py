"""
tests/test_responsibility_temporal_trace.py - 責任の時間的推移記述のテスト

カバー範囲:
- 初期状態
- スナップショット蓄積（Stage 1）
- スライディングウィンドウFIFO
- 5断面の段階値記述（Stage 2）
- 参照情報受渡準備（Stage 3）: enrichment + READ-ONLYアクセサ
- 直前スナップショットの保持・更新
- save/load round-trip
- 安全弁（パターン抽出禁止、統計量算出禁止、書き込み経路遮断、
          判断層非接続、enrichment等価列挙、方向性記述排除、FIFO自然消失保証）
- エッジケース（空データ、上限超過、最小件数未満等）
- ファクトリ
- 統合テスト
"""

import time
import pytest

from psyche.responsibility_temporal_trace import (
    VariationLevel,
    ResponsibilitySnapshot,
    ResponsibilityTemporalTraceState,
    ResponsibilityTemporalTraceConfig,
    ResponsibilityTemporalTraceProcessor,
    get_trace_summary,
    create_responsibility_temporal_trace,
    SECTION_ORDER,
    SECTION_TOTAL_WEIGHT_VARIATION,
    SECTION_PENDING_DECISIONS_RETENTION,
    SECTION_HARM_VARIATION,
    SECTION_CONFIDENCE_VARIATION,
    SECTION_DISPERSION_ACTIVITY_DENSITY,
    SECTION_LABELS,
    VARIATION_LABELS,
    _classify_variation,
    _classify_integer_variation,
    _classify_count_density,
)


# =============================================================================
# Helpers
# =============================================================================

def make_processor(
    max_snapshots: int = 100,
    min_records_for_description: int = 3,
) -> ResponsibilityTemporalTraceProcessor:
    """テスト用プロセッサを生成する。"""
    config = ResponsibilityTemporalTraceConfig(
        max_snapshots=max_snapshots,
        min_records_for_description=min_records_for_description,
    )
    return ResponsibilityTemporalTraceProcessor(config=config)


def record_n(
    processor: ResponsibilityTemporalTraceProcessor,
    n: int,
    base_tick: int = 1,
    total_weight: float = 0.5,
    pending_decisions: int = 0,
    accumulated_harm: float = 0.0,
    accumulated_confidence: float = 0.0,
    dispersion_active_weight: float = 0.0,
    dispersion_active_count: int = 0,
    dispersion_transformation_count: int = 0,
    base_timestamp: float = 1000.0,
) -> None:
    """n件のスナップショットを蓄積する。"""
    for i in range(n):
        processor.record_snapshot(
            tick=base_tick + i,
            total_weight=total_weight,
            pending_decisions=pending_decisions,
            accumulated_harm=accumulated_harm,
            accumulated_confidence=accumulated_confidence,
            dispersion_active_weight=dispersion_active_weight,
            dispersion_active_count=dispersion_active_count,
            dispersion_transformation_count=dispersion_transformation_count,
            timestamp=base_timestamp + i,
        )


# =============================================================================
# Test: VariationLevel Enum
# =============================================================================

class TestVariationLevel:
    def test_all_levels(self):
        levels = list(VariationLevel)
        assert len(levels) == 5

    def test_values(self):
        assert VariationLevel.LARGE.value == "large"
        assert VariationLevel.SOMEWHAT_LARGE.value == "somewhat_large"
        assert VariationLevel.MODERATE.value == "moderate"
        assert VariationLevel.SOMEWHAT_SMALL.value == "somewhat_small"
        assert VariationLevel.SMALL.value == "small"

    def test_no_weight_attribute(self):
        """各段階に重み・スコア・優先度は付与しない。"""
        for level in VariationLevel:
            assert not hasattr(level, "weight")
            assert not hasattr(level, "score")
            assert not hasattr(level, "priority")

    def test_no_directional_labels(self):
        """段階値は方向性（増加/減少）を含まない。"""
        for level in VariationLevel:
            assert "increase" not in level.value.lower()
            assert "decrease" not in level.value.lower()
            assert "up" not in level.value.lower()
            assert "down" not in level.value.lower()


# =============================================================================
# Test: Section Definitions
# =============================================================================

class TestSectionDefinitions:
    def test_section_order_length(self):
        assert len(SECTION_ORDER) == 5

    def test_all_sections_in_order(self):
        assert SECTION_TOTAL_WEIGHT_VARIATION in SECTION_ORDER
        assert SECTION_PENDING_DECISIONS_RETENTION in SECTION_ORDER
        assert SECTION_HARM_VARIATION in SECTION_ORDER
        assert SECTION_CONFIDENCE_VARIATION in SECTION_ORDER
        assert SECTION_DISPERSION_ACTIVITY_DENSITY in SECTION_ORDER

    def test_all_sections_have_labels(self):
        for section in SECTION_ORDER:
            assert section in SECTION_LABELS

    def test_all_variation_levels_have_labels(self):
        for level in VariationLevel:
            assert level in VARIATION_LABELS

    def test_labels_no_evaluation(self):
        """ラベルに評価的表現を含まない。"""
        for label in SECTION_LABELS.values():
            assert "望ましい" not in label
            assert "異常" not in label
            assert "健全" not in label
            assert "注目" not in label


# =============================================================================
# Test: ResponsibilitySnapshot
# =============================================================================

class TestResponsibilitySnapshot:
    def test_default_values(self):
        snap = ResponsibilitySnapshot()
        assert snap.tick == 0
        assert snap.total_weight == 0.0
        assert snap.pending_decisions == 0
        assert snap.accumulated_harm == 0.0
        assert snap.accumulated_confidence == 0.0
        assert snap.dispersion_active_weight == 0.0
        assert snap.dispersion_active_count == 0
        assert snap.dispersion_transformation_count == 0

    def test_to_dict(self):
        snap = ResponsibilitySnapshot(
            tick=5,
            timestamp=1234.0,
            total_weight=0.3,
            pending_decisions=2,
            accumulated_harm=0.1,
            accumulated_confidence=0.2,
            dispersion_active_weight=0.5,
            dispersion_active_count=3,
            dispersion_transformation_count=7,
        )
        d = snap.to_dict()
        assert d["tick"] == 5
        assert d["timestamp"] == 1234.0
        assert d["total_weight"] == 0.3
        assert d["pending_decisions"] == 2
        assert d["accumulated_harm"] == 0.1
        assert d["accumulated_confidence"] == 0.2
        assert d["dispersion_active_weight"] == 0.5
        assert d["dispersion_active_count"] == 3
        assert d["dispersion_transformation_count"] == 7

    def test_from_dict(self):
        data = {
            "tick": 10,
            "timestamp": 2000.0,
            "total_weight": 0.7,
            "pending_decisions": 5,
            "accumulated_harm": 0.4,
            "accumulated_confidence": 0.6,
            "dispersion_active_weight": 0.8,
            "dispersion_active_count": 4,
            "dispersion_transformation_count": 12,
        }
        snap = ResponsibilitySnapshot.from_dict(data)
        assert snap.tick == 10
        assert snap.timestamp == 2000.0
        assert snap.total_weight == 0.7
        assert snap.pending_decisions == 5
        assert snap.accumulated_harm == 0.4
        assert snap.accumulated_confidence == 0.6
        assert snap.dispersion_active_weight == 0.8
        assert snap.dispersion_active_count == 4
        assert snap.dispersion_transformation_count == 12

    def test_from_dict_missing_keys(self):
        snap = ResponsibilitySnapshot.from_dict({})
        assert snap.tick == 0
        assert snap.total_weight == 0.0
        assert snap.pending_decisions == 0

    def test_roundtrip(self):
        snap = ResponsibilitySnapshot(
            tick=3, timestamp=500.0, total_weight=0.5,
            pending_decisions=1, accumulated_harm=0.2,
            accumulated_confidence=0.3,
            dispersion_active_weight=0.1,
            dispersion_active_count=2,
            dispersion_transformation_count=4,
        )
        restored = ResponsibilitySnapshot.from_dict(snap.to_dict())
        assert restored.tick == snap.tick
        assert restored.total_weight == snap.total_weight
        assert restored.pending_decisions == snap.pending_decisions
        assert restored.accumulated_harm == snap.accumulated_harm
        assert restored.accumulated_confidence == snap.accumulated_confidence
        assert restored.dispersion_active_weight == snap.dispersion_active_weight
        assert restored.dispersion_active_count == snap.dispersion_active_count
        assert restored.dispersion_transformation_count == snap.dispersion_transformation_count

    def test_no_evaluation_attributes(self):
        """スナップショットに評価的属性がないことを確認。"""
        snap = ResponsibilitySnapshot()
        assert not hasattr(snap, "weight")
        assert not hasattr(snap, "score")
        assert not hasattr(snap, "priority")
        assert not hasattr(snap, "importance")


# =============================================================================
# Test: State
# =============================================================================

class TestState:
    def test_initial_state(self):
        state = ResponsibilityTemporalTraceState()
        assert state.snapshots == []
        assert state.section_snapshot == {}
        assert state.previous_section_snapshot == {}

    def test_to_dict(self):
        state = ResponsibilityTemporalTraceState()
        d = state.to_dict()
        assert "snapshots" in d
        assert "section_snapshot" in d
        assert "previous_section_snapshot" in d
        assert d["snapshots"] == []
        assert d["section_snapshot"] == {}
        assert d["previous_section_snapshot"] == {}

    def test_from_dict_empty(self):
        state = ResponsibilityTemporalTraceState.from_dict({})
        assert state.snapshots == []
        assert state.section_snapshot == {}
        assert state.previous_section_snapshot == {}

    def test_roundtrip(self):
        state = ResponsibilityTemporalTraceState()
        state.snapshots.append(ResponsibilitySnapshot(tick=1, timestamp=100.0, total_weight=0.5))
        state.section_snapshot = {"total_weight_variation": "moderate"}
        state.previous_section_snapshot = {"total_weight_variation": "small"}

        restored = ResponsibilityTemporalTraceState.from_dict(state.to_dict())
        assert len(restored.snapshots) == 1
        assert restored.snapshots[0].tick == 1
        assert restored.snapshots[0].total_weight == 0.5
        assert restored.section_snapshot == {"total_weight_variation": "moderate"}
        assert restored.previous_section_snapshot == {"total_weight_variation": "small"}


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfig:
    def test_default_values(self):
        cfg = ResponsibilityTemporalTraceConfig()
        assert cfg.max_snapshots == 100
        assert cfg.min_records_for_description == 3

    def test_custom_values(self):
        cfg = ResponsibilityTemporalTraceConfig(
            max_snapshots=50,
            min_records_for_description=5,
        )
        assert cfg.max_snapshots == 50
        assert cfg.min_records_for_description == 5


# =============================================================================
# Test: Classification Helpers
# =============================================================================

class TestClassifyVariation:
    def test_below_min_count(self):
        result = _classify_variation([0.1, 0.2], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.MODERATE

    def test_empty_list(self):
        result = _classify_variation([], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.MODERATE

    def test_zero_range(self):
        result = _classify_variation([0.5, 0.5, 0.5], theoretical_range=0.0, min_count=3)
        assert result == VariationLevel.MODERATE

    def test_no_variation(self):
        result = _classify_variation([0.5, 0.5, 0.5, 0.5], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.SMALL

    def test_large_variation(self):
        result = _classify_variation([0.0, 0.3, 0.7, 1.0], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.LARGE

    def test_somewhat_large_variation(self):
        result = _classify_variation([0.2, 0.4, 0.5, 0.55], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.SOMEWHAT_LARGE

    def test_moderate_variation(self):
        result = _classify_variation([0.4, 0.45, 0.5, 0.52], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.MODERATE

    def test_somewhat_small_variation(self):
        result = _classify_variation([0.5, 0.51, 0.52, 0.53], theoretical_range=1.0, min_count=3)
        assert result == VariationLevel.SOMEWHAT_SMALL

    def test_negative_range(self):
        result = _classify_variation([0.5, 0.5, 0.5], theoretical_range=-1.0, min_count=3)
        assert result == VariationLevel.MODERATE


class TestClassifyIntegerVariation:
    def test_delegates_to_classify_variation(self):
        result = _classify_integer_variation([0, 5, 10, 15], theoretical_range=20.0, min_count=3)
        float_result = _classify_variation([0.0, 5.0, 10.0, 15.0], theoretical_range=20.0, min_count=3)
        assert result == float_result

    def test_below_min_count(self):
        result = _classify_integer_variation([1, 2], theoretical_range=20.0, min_count=3)
        assert result == VariationLevel.MODERATE


class TestClassifyCountDensity:
    def test_below_min_count(self):
        result = _classify_count_density([1, 2], min_count=3)
        assert result == VariationLevel.MODERATE

    def test_empty_list(self):
        result = _classify_count_density([], min_count=3)
        assert result == VariationLevel.MODERATE

    def test_no_change(self):
        result = _classify_count_density([5, 5, 5, 5], min_count=3)
        assert result == VariationLevel.SMALL

    def test_high_frequency(self):
        result = _classify_count_density([0, 3, 6, 10], min_count=3)
        assert result in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_moderate_frequency(self):
        result = _classify_count_density([0, 1, 2, 3, 4], min_count=3)
        assert result in (VariationLevel.MODERATE, VariationLevel.SOMEWHAT_LARGE)


# =============================================================================
# Test: Processor - Stage 1 (Snapshot Accumulation)
# =============================================================================

class TestStage1:
    def test_initial_empty(self):
        proc = make_processor()
        assert len(proc.state.snapshots) == 0

    def test_record_single_snapshot(self):
        proc = make_processor()
        proc.record_snapshot(
            tick=1,
            total_weight=0.3,
            pending_decisions=2,
            accumulated_harm=0.1,
            accumulated_confidence=0.2,
            dispersion_active_weight=0.5,
            dispersion_active_count=3,
            dispersion_transformation_count=7,
            timestamp=1000.0,
        )
        assert len(proc.state.snapshots) == 1
        snap = proc.state.snapshots[0]
        assert snap.tick == 1
        assert snap.total_weight == 0.3
        assert snap.pending_decisions == 2
        assert snap.accumulated_harm == 0.1
        assert snap.accumulated_confidence == 0.2
        assert snap.dispersion_active_weight == 0.5
        assert snap.dispersion_active_count == 3
        assert snap.dispersion_transformation_count == 7
        assert snap.timestamp == 1000.0

    def test_record_multiple_snapshots(self):
        proc = make_processor()
        record_n(proc, 5)
        assert len(proc.state.snapshots) == 5

    def test_fifo_pushout(self):
        proc = make_processor(max_snapshots=5)
        record_n(proc, 8, base_tick=1, base_timestamp=1000.0)
        assert len(proc.state.snapshots) == 5
        # 最古は tick=4 であるべき（1,2,3 が押し出される）
        assert proc.state.snapshots[0].tick == 4
        assert proc.state.snapshots[-1].tick == 8

    def test_fifo_no_selective_deletion(self):
        """FIFO方式で最古から押し出すのみ。選択的削除は行わない（安全弁7）。"""
        proc = make_processor(max_snapshots=3)
        # 異なる値のスナップショットを蓄積
        for i in range(5):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,
                pending_decisions=i,
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        assert len(proc.state.snapshots) == 3
        # tick=3,4,5 が残る
        assert proc.state.snapshots[0].tick == 3
        assert proc.state.snapshots[1].tick == 4
        assert proc.state.snapshots[2].tick == 5

    def test_default_timestamp(self):
        """タイムスタンプ未指定時は現在時刻が使用される。"""
        proc = make_processor()
        before = time.time()
        proc.record_snapshot(
            tick=1,
            total_weight=0.0,
            pending_decisions=0,
            accumulated_harm=0.0,
            accumulated_confidence=0.0,
            dispersion_active_weight=0.0,
            dispersion_active_count=0,
            dispersion_transformation_count=0,
        )
        after = time.time()
        ts = proc.state.snapshots[0].timestamp
        assert before <= ts <= after

    def test_snapshots_are_equal(self):
        """全スナップショットは等価。重み・スコア・優先度がない。"""
        proc = make_processor()
        record_n(proc, 5)
        for snap in proc.state.snapshots:
            assert not hasattr(snap, "weight") or snap.__class__.__name__ == "ResponsibilitySnapshot"
            assert not hasattr(snap, "score")
            assert not hasattr(snap, "priority")


# =============================================================================
# Test: Processor - Stage 2 (Variation Description)
# =============================================================================

class TestStage2:
    def test_describe_with_no_data(self):
        proc = make_processor()
        result = proc.describe_variation()
        assert len(result) == 5
        # 全断面がMODERATE
        for section in SECTION_ORDER:
            assert result[section] == VariationLevel.MODERATE.value

    def test_describe_with_insufficient_data(self):
        proc = make_processor(min_records_for_description=3)
        record_n(proc, 2)
        result = proc.describe_variation()
        for section in SECTION_ORDER:
            assert result[section] == VariationLevel.MODERATE.value

    def test_describe_with_constant_values(self):
        proc = make_processor()
        record_n(proc, 10, total_weight=0.5)
        result = proc.describe_variation()
        assert result[SECTION_TOTAL_WEIGHT_VARIATION] == VariationLevel.SMALL.value

    def test_describe_with_varying_total_weight(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,  # 0.0 to 0.9
                pending_decisions=0,
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        level = VariationLevel(result[SECTION_TOTAL_WEIGHT_VARIATION])
        assert level in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_describe_with_varying_harm(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.5,
                pending_decisions=0,
                accumulated_harm=i * 0.08,  # 0.0 to 0.72
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        level = VariationLevel(result[SECTION_HARM_VARIATION])
        assert level in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_describe_with_varying_confidence(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.5,
                pending_decisions=0,
                accumulated_harm=0.0,
                accumulated_confidence=i * 0.06,  # 0.0 to 0.54
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        level = VariationLevel(result[SECTION_CONFIDENCE_VARIATION])
        assert level in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_describe_with_varying_pending(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.5,
                pending_decisions=i * 2,  # 0 to 18
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        level = VariationLevel(result[SECTION_PENDING_DECISIONS_RETENTION])
        assert level in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_describe_with_varying_dispersion(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.5,
                pending_decisions=0,
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=i * 3,  # 0 to 27
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        level = VariationLevel(result[SECTION_DISPERSION_ACTIVITY_DENSITY])
        assert level in (VariationLevel.LARGE, VariationLevel.SOMEWHAT_LARGE)

    def test_all_sections_present(self):
        proc = make_processor()
        record_n(proc, 5)
        result = proc.describe_variation()
        for section in SECTION_ORDER:
            assert section in result

    def test_all_sections_equal(self):
        """全断面は等価。断面間に優先順位・重み付けは存在しない。"""
        proc = make_processor()
        record_n(proc, 5)
        result = proc.describe_variation()
        # 各断面が独立にVariationLevelの値を持つ
        for section in SECTION_ORDER:
            assert result[section] in [v.value for v in VariationLevel]

    def test_no_direction_in_output(self):
        """出力に方向性（増加/減少）が含まれないことを確認（安全弁6）。"""
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,
                pending_decisions=i,
                accumulated_harm=i * 0.05,
                accumulated_confidence=i * 0.03,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=i,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        for section, value in result.items():
            assert "increase" not in value.lower()
            assert "decrease" not in value.lower()
            assert "rising" not in value.lower()
            assert "falling" not in value.lower()


# =============================================================================
# Test: Processor - Stage 3 (Handoff Preparation)
# =============================================================================

class TestStage3:
    def test_enrichment_data_empty(self):
        proc = make_processor()
        data = proc.get_enrichment_data()
        assert data["snapshot_count"] == 0
        assert data["oldest_timestamp"] is None
        assert data["newest_timestamp"] is None
        assert data["section_snapshot"] == {}
        assert "summary_text" in data

    def test_enrichment_data_with_snapshots(self):
        proc = make_processor()
        record_n(proc, 5, base_timestamp=1000.0)
        proc.describe_variation()
        data = proc.get_enrichment_data()
        assert data["snapshot_count"] == 5
        assert data["oldest_timestamp"] == 1000.0
        assert data["newest_timestamp"] == 1004.0
        assert len(data["section_snapshot"]) == 5

    def test_enrichment_equal_listing(self):
        """enrichmentで全断面を等価に列挙する（安全弁5）。"""
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        data = proc.get_enrichment_data()
        section_snapshot = data["section_snapshot"]
        # 全断面が含まれる
        for section in SECTION_ORDER:
            assert section in section_snapshot

    def test_enrichment_no_emphasis(self):
        """enrichment出力に強調表現が含まれないことを確認（安全弁5）。"""
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,
                pending_decisions=i,
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        proc.describe_variation()
        data = proc.get_enrichment_data()
        summary = data["summary_text"]
        assert "注目" not in summary
        assert "異常" not in summary
        assert "警告" not in summary
        assert "重要" not in summary

    def test_section_snapshot_readonly(self):
        """get_section_snapshot はコピーを返す。"""
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        snap1 = proc.get_section_snapshot()
        snap1["extra_key"] = "extra_value"
        snap2 = proc.get_section_snapshot()
        assert "extra_key" not in snap2

    def test_previous_section_snapshot(self):
        """直前のスナップショットが保持される。"""
        proc = make_processor()
        record_n(proc, 5)
        first_result = proc.describe_variation()

        # 状態を変えて再記述
        for i in range(5):
            proc.record_snapshot(
                tick=10 + i,
                total_weight=0.9,
                pending_decisions=10,
                accumulated_harm=0.8,
                accumulated_confidence=0.1,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=i * 5,
                timestamp=2000.0 + i,
            )
        second_result = proc.describe_variation()

        prev = proc.get_previous_section_snapshot()
        assert prev == first_result

    def test_previous_snapshot_readonly(self):
        """get_previous_section_snapshot はコピーを返す。"""
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        prev = proc.get_previous_section_snapshot()
        prev["extra"] = "extra"
        assert "extra" not in proc.get_previous_section_snapshot()

    def test_get_snapshots_returns_all(self):
        """get_snapshots は全件をそのまま返す。フィルタリングなし。"""
        proc = make_processor()
        record_n(proc, 7)
        snaps = proc.get_snapshots()
        assert len(snaps) == 7
        for i, snap in enumerate(snaps):
            assert snap["tick"] == i + 1

    def test_get_snapshots_readonly(self):
        """get_snapshots はコピーを返す。"""
        proc = make_processor()
        record_n(proc, 3)
        snaps1 = proc.get_snapshots()
        snaps1.append({"extra": True})
        snaps2 = proc.get_snapshots()
        assert len(snaps2) == 3

    def test_get_summary(self):
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        summary = proc.get_summary()
        assert summary["snapshot_count"] == 5
        assert summary["has_section_snapshot"] is True
        assert summary["has_previous_section_snapshot"] is False  # Only one describe call


# =============================================================================
# Test: Summary (enrichment text)
# =============================================================================

class TestSummary:
    def test_waiting_state(self):
        state = ResponsibilityTemporalTraceState()
        text = get_trace_summary(state)
        assert "待機中" in text

    def test_with_sections(self):
        state = ResponsibilityTemporalTraceState()
        state.section_snapshot = {
            SECTION_TOTAL_WEIGHT_VARIATION: VariationLevel.MODERATE.value,
            SECTION_PENDING_DECISIONS_RETENTION: VariationLevel.SMALL.value,
            SECTION_HARM_VARIATION: VariationLevel.LARGE.value,
            SECTION_CONFIDENCE_VARIATION: VariationLevel.SOMEWHAT_SMALL.value,
            SECTION_DISPERSION_ACTIVITY_DENSITY: VariationLevel.SOMEWHAT_LARGE.value,
        }
        text = get_trace_summary(state)
        assert "待機中" not in text
        # 全ラベルが含まれる
        for section in SECTION_ORDER:
            label = SECTION_LABELS[section]
            assert label in text

    def test_no_direction_in_summary(self):
        """要約に方向性表現が含まれないことを確認（安全弁6）。"""
        state = ResponsibilityTemporalTraceState()
        state.section_snapshot = {
            SECTION_TOTAL_WEIGHT_VARIATION: VariationLevel.LARGE.value,
            SECTION_PENDING_DECISIONS_RETENTION: VariationLevel.LARGE.value,
            SECTION_HARM_VARIATION: VariationLevel.LARGE.value,
            SECTION_CONFIDENCE_VARIATION: VariationLevel.LARGE.value,
            SECTION_DISPERSION_ACTIVITY_DENSITY: VariationLevel.LARGE.value,
        }
        text = get_trace_summary(state)
        assert "増加" not in text
        assert "減少" not in text
        assert "上昇" not in text
        assert "低下" not in text

    def test_no_evaluation_in_summary(self):
        """要約に評価的表現が含まれないことを確認。"""
        state = ResponsibilityTemporalTraceState()
        state.section_snapshot = {
            SECTION_TOTAL_WEIGHT_VARIATION: VariationLevel.LARGE.value,
        }
        text = get_trace_summary(state)
        assert "望ましい" not in text
        assert "異常" not in text
        assert "注目" not in text

    def test_section_order_in_summary(self):
        """要約内の断面列挙が定義順に従うことを確認。"""
        state = ResponsibilityTemporalTraceState()
        state.section_snapshot = {
            section: VariationLevel.MODERATE.value
            for section in SECTION_ORDER
        }
        text = get_trace_summary(state)
        # 各ラベルの出現位置が定義順に並ぶ
        positions = []
        for section in SECTION_ORDER:
            label = SECTION_LABELS[section]
            pos = text.find(label)
            assert pos >= 0
            positions.append(pos)
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1]


# =============================================================================
# Test: Save / Load
# =============================================================================

class TestSaveLoad:
    def test_save_empty(self):
        proc = make_processor()
        data = proc.save()
        assert data["snapshots"] == []
        assert data["section_snapshot"] == {}
        assert data["previous_section_snapshot"] == {}

    def test_save_with_data(self):
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        data = proc.save()
        assert len(data["snapshots"]) == 5
        assert len(data["section_snapshot"]) == 5

    def test_load(self):
        proc = make_processor()
        record_n(proc, 5)
        proc.describe_variation()
        saved = proc.save()

        proc2 = make_processor()
        proc2.load(saved)
        assert len(proc2.state.snapshots) == 5
        assert proc2.state.section_snapshot == proc.state.section_snapshot

    def test_roundtrip(self):
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.05,
                pending_decisions=i,
                accumulated_harm=i * 0.03,
                accumulated_confidence=i * 0.02,
                dispersion_active_weight=i * 0.01,
                dispersion_active_count=i,
                dispersion_transformation_count=i * 2,
                timestamp=1000.0 + i,
            )
        proc.describe_variation()
        first_describe = proc.describe_variation()

        saved = proc.save()
        proc2 = make_processor()
        proc2.load(saved)

        assert len(proc2.state.snapshots) == 10
        assert proc2.state.section_snapshot == proc.state.section_snapshot
        assert proc2.state.previous_section_snapshot == proc.state.previous_section_snapshot

    def test_load_empty_data(self):
        proc = make_processor()
        proc.load({})
        assert len(proc.state.snapshots) == 0
        assert proc.state.section_snapshot == {}


# =============================================================================
# Test: Safety Valves
# =============================================================================

class TestSafetyValves:
    def test_no_pattern_extraction(self):
        """安全弁1: 蓄積データから傾向・周期・規則性を抽出しない。

        プロセッサに trend, pattern, period, cycle 等の属性やメソッドがないことを確認。
        """
        proc = make_processor()
        assert not hasattr(proc, "extract_pattern")
        assert not hasattr(proc, "detect_trend")
        assert not hasattr(proc, "find_cycle")
        assert not hasattr(proc, "detect_periodicity")

    def test_no_statistics(self):
        """安全弁2: 平均・分散・中央値・回帰等を適用しない。

        プロセッサに mean, variance, median, regression 等の属性がないことを確認。
        """
        proc = make_processor()
        assert not hasattr(proc, "compute_mean")
        assert not hasattr(proc, "compute_variance")
        assert not hasattr(proc, "compute_median")
        assert not hasattr(proc, "compute_regression")
        assert not hasattr(proc, "moving_average")

    def test_no_write_path(self):
        """安全弁3: 責任管理構造・責任分散構造への書き込み経路を構造的に遮断。

        プロセッサにset_weight, modify_responsibility 等がないことを確認。
        """
        proc = make_processor()
        assert not hasattr(proc, "set_weight")
        assert not hasattr(proc, "modify_responsibility")
        assert not hasattr(proc, "update_responsibility")
        assert not hasattr(proc, "disperse")
        assert not hasattr(proc, "sublimate")
        assert not hasattr(proc, "adjust_weight")

    def test_no_judgment_connection(self):
        """安全弁4: 段階値を判断バイアス計算・方針選択等に接続しない。

        プロセッサに apply_bias, get_bias, compute_bias 等がないことを確認。
        """
        proc = make_processor()
        assert not hasattr(proc, "apply_bias")
        assert not hasattr(proc, "get_bias")
        assert not hasattr(proc, "compute_bias")
        assert not hasattr(proc, "apply_to_policy")
        assert not hasattr(proc, "get_policy_influence")

    def test_enrichment_equal_listing(self):
        """安全弁5: enrichment出力で全断面を等価に列挙。"""
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.5 + i * 0.05,
                pending_decisions=i,
                accumulated_harm=0.1,
                accumulated_confidence=0.2,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        proc.describe_variation()
        data = proc.get_enrichment_data()
        section_snapshot = data["section_snapshot"]
        # 全5断面が含まれる
        assert len(section_snapshot) == 5
        for section in SECTION_ORDER:
            assert section in section_snapshot

    def test_no_direction_description(self):
        """安全弁6: 「増加傾向」「減少傾向」等の方向判定を行わない。"""
        proc = make_processor()
        # 単調増加するデータ
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,
                pending_decisions=0,
                accumulated_harm=0.0,
                accumulated_confidence=0.0,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=1000.0 + i,
            )
        result = proc.describe_variation()
        # 結果は変動の大きさのみを示し、方向性は示さない
        for value in result.values():
            assert value in [v.value for v in VariationLevel]
            # VariationLevel の値に方向表現がないことはenum定義で保証済み

    def test_fifo_only_deletion(self):
        """安全弁7: スナップショットの消失はFIFO押し出しのみ。"""
        proc = make_processor(max_snapshots=5)
        record_n(proc, 10, base_tick=1)
        # 全件数は max_snapshots 以下
        assert len(proc.state.snapshots) <= 5
        # 残っているのは最新5件
        ticks = [s.tick for s in proc.state.snapshots]
        assert ticks == [6, 7, 8, 9, 10]


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_zero_total_weight_all_records(self):
        proc = make_processor()
        record_n(proc, 5, total_weight=0.0)
        result = proc.describe_variation()
        assert result[SECTION_TOTAL_WEIGHT_VARIATION] == VariationLevel.SMALL.value

    def test_max_total_weight_all_records(self):
        proc = make_processor()
        record_n(proc, 5, total_weight=1.0)
        result = proc.describe_variation()
        assert result[SECTION_TOTAL_WEIGHT_VARIATION] == VariationLevel.SMALL.value

    def test_single_record(self):
        proc = make_processor()
        record_n(proc, 1)
        result = proc.describe_variation()
        for section in SECTION_ORDER:
            assert result[section] == VariationLevel.MODERATE.value

    def test_exact_min_count(self):
        proc = make_processor(min_records_for_description=3)
        record_n(proc, 3, total_weight=0.5)
        result = proc.describe_variation()
        # Should not be MODERATE due to min_count (values are all 0.5 => SMALL)
        assert result[SECTION_TOTAL_WEIGHT_VARIATION] == VariationLevel.SMALL.value

    def test_large_window(self):
        proc = make_processor(max_snapshots=1000)
        record_n(proc, 500)
        assert len(proc.state.snapshots) == 500
        result = proc.describe_variation()
        for section in SECTION_ORDER:
            assert result[section] in [v.value for v in VariationLevel]

    def test_describe_before_any_record(self):
        proc = make_processor()
        result = proc.describe_variation()
        for section in SECTION_ORDER:
            assert result[section] == VariationLevel.MODERATE.value

    def test_multiple_describe_calls(self):
        proc = make_processor()
        record_n(proc, 5)
        r1 = proc.describe_variation()
        r2 = proc.describe_variation()
        # 同じデータなので同じ結果
        assert r1 == r2
        # ただし previous_section_snapshot が更新される
        prev = proc.get_previous_section_snapshot()
        assert prev == r1

    def test_all_zero_values(self):
        proc = make_processor()
        record_n(
            proc, 5,
            total_weight=0.0,
            pending_decisions=0,
            accumulated_harm=0.0,
            accumulated_confidence=0.0,
            dispersion_active_weight=0.0,
            dispersion_active_count=0,
            dispersion_transformation_count=0,
        )
        result = proc.describe_variation()
        # 全て変動なし
        for section in SECTION_ORDER:
            assert result[section] == VariationLevel.SMALL.value


# =============================================================================
# Test: Factory
# =============================================================================

class TestFactory:
    def test_create_default(self):
        proc = create_responsibility_temporal_trace()
        assert isinstance(proc, ResponsibilityTemporalTraceProcessor)
        assert len(proc.state.snapshots) == 0

    def test_create_with_config(self):
        config = ResponsibilityTemporalTraceConfig(max_snapshots=50)
        proc = create_responsibility_temporal_trace(config=config)
        assert proc._config.max_snapshots == 50

    def test_create_none_config(self):
        proc = create_responsibility_temporal_trace(config=None)
        assert proc._config.max_snapshots == 100  # default


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    def test_full_pipeline(self):
        """3段パイプラインの全体通しテスト。"""
        proc = make_processor()

        # Stage 1: 多様な値でスナップショット蓄積
        for i in range(20):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.1 + (i % 5) * 0.15,
                pending_decisions=i % 4,
                accumulated_harm=0.05 + (i % 3) * 0.1,
                accumulated_confidence=0.1 + (i % 6) * 0.05,
                dispersion_active_weight=(i % 7) * 0.1,
                dispersion_active_count=i % 5,
                dispersion_transformation_count=i,
                timestamp=1000.0 + i,
            )

        assert len(proc.state.snapshots) == 20

        # Stage 2: 推移記述
        result = proc.describe_variation()
        assert len(result) == 5
        for section in SECTION_ORDER:
            assert result[section] in [v.value for v in VariationLevel]

        # Stage 3: enrichment data
        data = proc.get_enrichment_data()
        assert data["snapshot_count"] == 20
        assert data["oldest_timestamp"] == 1000.0
        assert data["newest_timestamp"] == 1019.0
        assert len(data["section_snapshot"]) == 5

    def test_pipeline_with_save_load(self):
        """パイプライン実行後にsave/loadしても状態が保持される。"""
        proc = make_processor()
        for i in range(15):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=0.2 + (i % 4) * 0.1,
                pending_decisions=i % 3,
                accumulated_harm=0.1,
                accumulated_confidence=0.05 + (i % 5) * 0.05,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=i * 2,
                timestamp=1000.0 + i,
            )
        proc.describe_variation()
        original_snapshot = proc.get_section_snapshot()

        saved = proc.save()
        proc2 = make_processor()
        proc2.load(saved)

        assert proc2.get_section_snapshot() == original_snapshot
        assert len(proc2.get_snapshots()) == 15

    def test_pipeline_repeated_describe(self):
        """describe_variation を複数回呼び出した場合の previous_section_snapshot の更新。"""
        proc = make_processor()
        record_n(proc, 5, total_weight=0.3)
        r1 = proc.describe_variation()

        # データを追加して状態を変える
        for i in range(5):
            proc.record_snapshot(
                tick=10 + i,
                total_weight=0.8,
                pending_decisions=5,
                accumulated_harm=0.5,
                accumulated_confidence=0.1,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=0,
                timestamp=2000.0 + i,
            )
        r2 = proc.describe_variation()

        # previous は r1 であるべき
        assert proc.get_previous_section_snapshot() == r1

    def test_no_mutation_of_inputs(self):
        """入力値の変更がプロセッサ内部に伝播しないことを確認。"""
        proc = make_processor()
        proc.record_snapshot(
            tick=1,
            total_weight=0.5,
            pending_decisions=3,
            accumulated_harm=0.2,
            accumulated_confidence=0.1,
            dispersion_active_weight=0.3,
            dispersion_active_count=2,
            dispersion_transformation_count=5,
            timestamp=1000.0,
        )
        # 内部データは入力とは独立
        snap = proc.state.snapshots[0]
        assert snap.total_weight == 0.5
        assert snap.pending_decisions == 3

    def test_read_only_principle(self):
        """本モジュールがREAD-ONLYの原則を守っていることの構造的確認。

        公開メソッドに責任管理構造・責任分散構造への書き込みに
        相当するメソッドが存在しないことを確認する。
        """
        proc = make_processor()
        public_methods = [
            m for m in dir(proc)
            if not m.startswith("_") and callable(getattr(proc, m))
        ]
        # 許可されたメソッド
        allowed = {
            "record_snapshot",
            "describe_variation",
            "get_enrichment_data",
            "get_section_snapshot",
            "get_previous_section_snapshot",
            "get_snapshots",
            "get_summary",
            "save",
            "load",
        }
        for method in public_methods:
            assert method in allowed, f"Unexpected public method: {method}"

    def test_enrichment_output_is_informational_only(self):
        """enrichment出力が情報のみで、判断・評価・行動指示を含まないことを確認。"""
        proc = make_processor()
        for i in range(10):
            proc.record_snapshot(
                tick=i + 1,
                total_weight=i * 0.1,
                pending_decisions=i,
                accumulated_harm=i * 0.05,
                accumulated_confidence=i * 0.03,
                dispersion_active_weight=0.0,
                dispersion_active_count=0,
                dispersion_transformation_count=i,
                timestamp=1000.0 + i,
            )
        proc.describe_variation()
        data = proc.get_enrichment_data()

        # enrichment データに判断・評価・行動のキーがない
        assert "recommendation" not in data
        assert "action" not in data
        assert "judgment" not in data
        assert "evaluation" not in data
        assert "suggestion" not in data
        assert "bias" not in data
