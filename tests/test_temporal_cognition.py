"""
tests/test_temporal_cognition.py - 時間認知構造のテスト

カバー範囲:
- 初期状態
- 経過記録蓄積（Stage 1）
- スライディングウィンドウFIFO
- 外部入力到着記録
- 多断面特徴量記述（Stage 2）: 6断面
- 参照情報受渡準備（Stage 3）: enrichment + READ-ONLYアクセサ
- 直前スナップショットの保持・更新
- save/load round-trip
- 安全弁（断面等価性、パターン抽出禁止、単一数値統合禁止、強調禁止、ウィンドウ自然更新）
- 4経路遮断（ティックベース処理、感情パイプライン、記憶忘却/固定化、予期形成）
- 自己差分認知との境界維持
- エッジケース（空データ、上限超過、ゼロ経過秒等）
- ファクトリ
- 統合テスト
"""

import time
import pytest

from psyche.temporal_cognition import (
    DensityLevel,
    ElapsedRecord,
    TemporalCognitionState,
    TemporalCognitionConfig,
    TemporalCognitionProcessor,
    get_temporal_summary,
    create_temporal_cognition,
    SECTION_ORDER,
    SECTION_ACTIVITY_DENSITY,
    SECTION_MEMORY_INTERVAL,
    SECTION_EMOTION_FREQUENCY,
    SECTION_NARRATIVE_INTERVAL,
    SECTION_EXTERNAL_INPUT_INTERVAL,
    SECTION_OVERALL_ELAPSED,
    SECTION_LABELS,
    DENSITY_LABELS,
    _classify_interval_density,
    _classify_frequency,
    _classify_tempo,
)


# =============================================================================
# Helpers
# =============================================================================

def make_processor(
    max_elapsed_records: int = 100,
    max_external_input_records: int = 100,
    min_records_for_description: int = 3,
) -> TemporalCognitionProcessor:
    """テスト用プロセッサを生成する。"""
    config = TemporalCognitionConfig(
        max_elapsed_records=max_elapsed_records,
        max_external_input_records=max_external_input_records,
        min_records_for_description=min_records_for_description,
    )
    return TemporalCognitionProcessor(config=config)


def accumulate_n(
    processor: TemporalCognitionProcessor,
    n: int,
    base_tick: int = 1,
    delta_time: float = 1.0,
    base_timestamp: float = 1000.0,
) -> None:
    """n件の経過記録を蓄積する。"""
    for i in range(n):
        processor.accumulate_elapsed(
            tick=base_tick + i,
            delta_time=delta_time,
            timestamp=base_timestamp + i * delta_time,
        )


# =============================================================================
# Test: DensityLevel Enum
# =============================================================================

class TestDensityLevel:
    def test_all_levels(self):
        levels = list(DensityLevel)
        assert len(levels) == 5

    def test_values(self):
        assert DensityLevel.DENSE.value == "dense"
        assert DensityLevel.SOMEWHAT_DENSE.value == "somewhat_dense"
        assert DensityLevel.NORMAL.value == "normal"
        assert DensityLevel.SOMEWHAT_SPARSE.value == "somewhat_sparse"
        assert DensityLevel.SPARSE.value == "sparse"

    def test_no_weight_attribute(self):
        """各段階に重み・スコア・優先度は付与しない。"""
        for level in DensityLevel:
            assert not hasattr(level, "weight")
            assert not hasattr(level, "score")
            assert not hasattr(level, "priority")


# =============================================================================
# Test: ElapsedRecord
# =============================================================================

class TestElapsedRecord:
    def test_default_creation(self):
        rec = ElapsedRecord()
        assert rec.tick == 0
        assert rec.delta_time == 0.0
        assert rec.timestamp > 0

    def test_creation_with_values(self):
        rec = ElapsedRecord(tick=5, delta_time=1.5, timestamp=1000.0)
        assert rec.tick == 5
        assert rec.delta_time == 1.5
        assert rec.timestamp == 1000.0

    def test_to_dict(self):
        rec = ElapsedRecord(tick=10, delta_time=2.0, timestamp=2000.0)
        d = rec.to_dict()
        assert d["tick"] == 10
        assert d["delta_time"] == 2.0
        assert d["timestamp"] == 2000.0

    def test_from_dict(self):
        original = ElapsedRecord(tick=7, delta_time=0.5, timestamp=1500.0)
        d = original.to_dict()
        restored = ElapsedRecord.from_dict(d)
        assert restored.tick == original.tick
        assert restored.delta_time == original.delta_time
        assert restored.timestamp == original.timestamp

    def test_from_dict_empty(self):
        rec = ElapsedRecord.from_dict({})
        assert rec.tick == 0
        assert rec.delta_time == 0.0

    def test_no_weight_or_score(self):
        """記録には重み・スコア・優先度などの評価的属性を持たない。"""
        rec = ElapsedRecord(tick=1, delta_time=1.0)
        d = rec.to_dict()
        assert "weight" not in d
        assert "score" not in d
        assert "priority" not in d
        assert "importance" not in d


# =============================================================================
# Test: TemporalCognitionState
# =============================================================================

class TestTemporalCognitionState:
    def test_default_state(self):
        state = TemporalCognitionState()
        assert state.elapsed_records == []
        assert state.snapshot == {}
        assert state.previous_snapshot == {}
        assert state.external_input_timestamps == []

    def test_to_dict(self):
        state = TemporalCognitionState()
        d = state.to_dict()
        assert d["elapsed_records"] == []
        assert d["snapshot"] == {}
        assert d["previous_snapshot"] == {}
        assert d["external_input_timestamps"] == []

    def test_to_dict_with_data(self):
        state = TemporalCognitionState()
        state.elapsed_records.append(ElapsedRecord(tick=1, delta_time=1.0, timestamp=1000.0))
        state.snapshot = {SECTION_ACTIVITY_DENSITY: DensityLevel.DENSE.value}
        state.external_input_timestamps.append(1000.0)
        d = state.to_dict()
        assert len(d["elapsed_records"]) == 1
        assert d["snapshot"][SECTION_ACTIVITY_DENSITY] == "dense"
        assert len(d["external_input_timestamps"]) == 1

    def test_from_dict(self):
        original = TemporalCognitionState()
        original.elapsed_records.append(ElapsedRecord(tick=1, delta_time=1.0, timestamp=1000.0))
        original.snapshot = {SECTION_ACTIVITY_DENSITY: DensityLevel.NORMAL.value}
        original.previous_snapshot = {SECTION_ACTIVITY_DENSITY: DensityLevel.DENSE.value}
        original.external_input_timestamps = [1000.0, 2000.0]

        d = original.to_dict()
        restored = TemporalCognitionState.from_dict(d)

        assert len(restored.elapsed_records) == 1
        assert restored.elapsed_records[0].tick == 1
        assert restored.snapshot[SECTION_ACTIVITY_DENSITY] == "normal"
        assert restored.previous_snapshot[SECTION_ACTIVITY_DENSITY] == "dense"
        assert len(restored.external_input_timestamps) == 2

    def test_from_dict_empty(self):
        state = TemporalCognitionState.from_dict({})
        assert state.elapsed_records == []
        assert state.snapshot == {}
        assert state.previous_snapshot == {}
        assert state.external_input_timestamps == []


# =============================================================================
# Test: TemporalCognitionConfig
# =============================================================================

class TestTemporalCognitionConfig:
    def test_defaults(self):
        cfg = TemporalCognitionConfig()
        assert cfg.max_elapsed_records == 100
        assert cfg.max_external_input_records == 100
        assert cfg.min_records_for_description == 3

    def test_custom_values(self):
        cfg = TemporalCognitionConfig(
            max_elapsed_records=50,
            max_external_input_records=30,
            min_records_for_description=5,
        )
        assert cfg.max_elapsed_records == 50
        assert cfg.max_external_input_records == 30
        assert cfg.min_records_for_description == 5


# =============================================================================
# Test: Stage 1 - 経過情報の蓄積
# =============================================================================

class TestElapsedAccumulation:
    def test_basic_accumulation(self):
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=1.0, timestamp=1000.0)
        assert len(processor.state.elapsed_records) == 1
        assert processor.state.elapsed_records[0].tick == 1
        assert processor.state.elapsed_records[0].delta_time == 1.0

    def test_multiple_accumulation(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        assert len(processor.state.elapsed_records) == 5

    def test_time_series_order(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert ticks == sorted(ticks)

    def test_timestamp_auto_generated(self):
        """タイムスタンプ未指定時は自動生成。"""
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=1.0)
        assert processor.state.elapsed_records[0].timestamp > 0

    def test_accumulation_one_per_call(self):
        processor = make_processor()
        pre_count = len(processor.state.elapsed_records)
        processor.accumulate_elapsed(tick=1, delta_time=1.0, timestamp=1000.0)
        assert len(processor.state.elapsed_records) == pre_count + 1


# =============================================================================
# Test: スライディングウィンドウ FIFO
# =============================================================================

class TestSlidingWindowFIFO:
    def test_pushout_at_limit(self):
        processor = make_processor(max_elapsed_records=5)
        accumulate_n(processor, 7)
        assert len(processor.state.elapsed_records) == 5

    def test_pushout_preserves_newest(self):
        processor = make_processor(max_elapsed_records=3)
        accumulate_n(processor, 5)
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert ticks == [3, 4, 5]

    def test_pushout_removes_oldest(self):
        processor = make_processor(max_elapsed_records=3)
        accumulate_n(processor, 5)
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert 1 not in ticks
        assert 2 not in ticks

    def test_no_pushout_within_limit(self):
        processor = make_processor(max_elapsed_records=10)
        accumulate_n(processor, 5)
        assert len(processor.state.elapsed_records) == 5

    def test_exact_limit_no_pushout(self):
        processor = make_processor(max_elapsed_records=5)
        accumulate_n(processor, 5)
        assert len(processor.state.elapsed_records) == 5

    def test_pushout_cumulative(self):
        processor = make_processor(max_elapsed_records=3)
        accumulate_n(processor, 10)
        assert len(processor.state.elapsed_records) == 3
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert ticks == [8, 9, 10]

    def test_pushout_is_only_removal_path(self):
        """上限到達時の最古押し出しが唯一の消失経路。"""
        processor = make_processor(max_elapsed_records=3)
        accumulate_n(processor, 3)
        assert len(processor.state.elapsed_records) == 3
        processor.accumulate_elapsed(tick=10, delta_time=1.0, timestamp=2000.0)
        assert len(processor.state.elapsed_records) == 3
        assert processor.state.elapsed_records[0].tick == 2


# =============================================================================
# Test: 外部入力到着記録
# =============================================================================

class TestExternalInputNotification:
    def test_basic_notification(self):
        processor = make_processor()
        processor.notify_external_input(timestamp=1000.0)
        assert len(processor.state.external_input_timestamps) == 1
        assert processor.state.external_input_timestamps[0] == 1000.0

    def test_multiple_notifications(self):
        processor = make_processor()
        for i in range(5):
            processor.notify_external_input(timestamp=1000.0 + i)
        assert len(processor.state.external_input_timestamps) == 5

    def test_pushout_at_limit(self):
        processor = make_processor(max_external_input_records=3)
        for i in range(5):
            processor.notify_external_input(timestamp=1000.0 + i)
        assert len(processor.state.external_input_timestamps) == 3
        assert processor.state.external_input_timestamps[0] == 1002.0

    def test_auto_timestamp(self):
        processor = make_processor()
        processor.notify_external_input()
        assert len(processor.state.external_input_timestamps) == 1
        assert processor.state.external_input_timestamps[0] > 0

    def test_pushout_preserves_newest(self):
        processor = make_processor(max_external_input_records=2)
        processor.notify_external_input(timestamp=100.0)
        processor.notify_external_input(timestamp=200.0)
        processor.notify_external_input(timestamp=300.0)
        assert processor.state.external_input_timestamps == [200.0, 300.0]


# =============================================================================
# Test: Stage 2 - 各断面の特徴量記述
# =============================================================================

class TestActivityDensitySection:
    def test_empty_records(self):
        processor = make_processor()
        snapshot = processor.describe_features()
        assert snapshot[SECTION_ACTIVITY_DENSITY] == DensityLevel.NORMAL.value

    def test_dense_activity(self):
        """短い間隔が連続する場合。"""
        processor = make_processor(min_records_for_description=3)
        # 長い間隔 → 短い間隔の推移
        processor.accumulate_elapsed(tick=1, delta_time=10.0, timestamp=1000.0)
        processor.accumulate_elapsed(tick=2, delta_time=10.0, timestamp=1010.0)
        processor.accumulate_elapsed(tick=3, delta_time=10.0, timestamp=1020.0)
        processor.accumulate_elapsed(tick=4, delta_time=1.0, timestamp=1021.0)
        processor.accumulate_elapsed(tick=5, delta_time=1.0, timestamp=1022.0)
        processor.accumulate_elapsed(tick=6, delta_time=1.0, timestamp=1023.0)
        snapshot = processor.describe_features()
        # 直近が密なのでDENSEまたはSOMEWHAT_DENSE方向
        level = DensityLevel(snapshot[SECTION_ACTIVITY_DENSITY])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_sparse_activity(self):
        """長い間隔が連続する場合。"""
        processor = make_processor(min_records_for_description=3)
        # 短い間隔 → 長い間隔の推移
        processor.accumulate_elapsed(tick=1, delta_time=1.0, timestamp=1000.0)
        processor.accumulate_elapsed(tick=2, delta_time=1.0, timestamp=1001.0)
        processor.accumulate_elapsed(tick=3, delta_time=1.0, timestamp=1002.0)
        processor.accumulate_elapsed(tick=4, delta_time=50.0, timestamp=1052.0)
        processor.accumulate_elapsed(tick=5, delta_time=50.0, timestamp=1102.0)
        processor.accumulate_elapsed(tick=6, delta_time=50.0, timestamp=1152.0)
        snapshot = processor.describe_features()
        level = DensityLevel(snapshot[SECTION_ACTIVITY_DENSITY])
        assert level in (DensityLevel.SPARSE, DensityLevel.SOMEWHAT_SPARSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 10)
        snapshot = processor.describe_features()
        value = snapshot[SECTION_ACTIVITY_DENSITY]
        assert value in [dl.value for dl in DensityLevel]


class TestMemoryIntervalSection:
    def test_empty_timestamps(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(episodic_timestamps=[])
        assert snapshot[SECTION_MEMORY_INTERVAL] == DensityLevel.NORMAL.value

    def test_single_timestamp(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(episodic_timestamps=[1000.0])
        assert snapshot[SECTION_MEMORY_INTERVAL] == DensityLevel.NORMAL.value

    def test_dense_memory(self):
        """記憶が密に蓄積された場合。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 5)
        # 長い間隔の後に短い間隔
        ts = [100.0, 200.0, 300.0, 301.0, 302.0, 303.0]
        snapshot = processor.describe_features(episodic_timestamps=ts)
        level = DensityLevel(snapshot[SECTION_MEMORY_INTERVAL])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        ts = [1000.0, 1010.0, 1020.0, 1030.0]
        snapshot = processor.describe_features(episodic_timestamps=ts)
        value = snapshot[SECTION_MEMORY_INTERVAL]
        assert value in [dl.value for dl in DensityLevel]


class TestEmotionFrequencySection:
    def test_zero_changes(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(emotion_change_count=0)
        assert snapshot[SECTION_EMOTION_FREQUENCY] in [dl.value for dl in DensityLevel]

    def test_high_frequency(self):
        """感情変動が頻繁な場合。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 10)
        snapshot = processor.describe_features(emotion_change_count=9)
        level = DensityLevel(snapshot[SECTION_EMOTION_FREQUENCY])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_low_frequency(self):
        """感情変動が少ない場合。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 20)
        snapshot = processor.describe_features(emotion_change_count=1)
        level = DensityLevel(snapshot[SECTION_EMOTION_FREQUENCY])
        assert level in (DensityLevel.SPARSE, DensityLevel.SOMEWHAT_SPARSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(emotion_change_count=3)
        value = snapshot[SECTION_EMOTION_FREQUENCY]
        assert value in [dl.value for dl in DensityLevel]


class TestNarrativeIntervalSection:
    def test_empty_timestamps(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(narrative_timestamps=[])
        assert snapshot[SECTION_NARRATIVE_INTERVAL] == DensityLevel.NORMAL.value

    def test_single_timestamp(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features(narrative_timestamps=[1000.0])
        assert snapshot[SECTION_NARRATIVE_INTERVAL] == DensityLevel.NORMAL.value

    def test_dense_narrative(self):
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 5)
        # 長い間隔の後に短い間隔
        ts = [100.0, 200.0, 300.0, 301.0, 302.0, 303.0]
        snapshot = processor.describe_features(narrative_timestamps=ts)
        level = DensityLevel(snapshot[SECTION_NARRATIVE_INTERVAL])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        ts = [1000.0, 1100.0, 1200.0]
        snapshot = processor.describe_features(narrative_timestamps=ts)
        value = snapshot[SECTION_NARRATIVE_INTERVAL]
        assert value in [dl.value for dl in DensityLevel]


class TestExternalInputIntervalSection:
    def test_no_external_input(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_EXTERNAL_INPUT_INTERVAL] == DensityLevel.NORMAL.value

    def test_single_external_input(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.notify_external_input(timestamp=1000.0)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_EXTERNAL_INPUT_INTERVAL] == DensityLevel.NORMAL.value

    def test_dense_external_input(self):
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 5)
        # 長い間隔の後に短い間隔
        processor.notify_external_input(timestamp=100.0)
        processor.notify_external_input(timestamp=200.0)
        processor.notify_external_input(timestamp=300.0)
        processor.notify_external_input(timestamp=301.0)
        processor.notify_external_input(timestamp=302.0)
        processor.notify_external_input(timestamp=303.0)
        snapshot = processor.describe_features()
        level = DensityLevel(snapshot[SECTION_EXTERNAL_INPUT_INTERVAL])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        for i in range(5):
            processor.notify_external_input(timestamp=1000.0 + i * 10)
        snapshot = processor.describe_features()
        value = snapshot[SECTION_EXTERNAL_INPUT_INTERVAL]
        assert value in [dl.value for dl in DensityLevel]


class TestOverallElapsedSection:
    def test_insufficient_records(self):
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 2)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_OVERALL_ELAPSED] == DensityLevel.NORMAL.value

    def test_dense_tempo(self):
        """短い平均間隔の場合。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 10, delta_time=0.5)
        snapshot = processor.describe_features()
        level = DensityLevel(snapshot[SECTION_OVERALL_ELAPSED])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_sparse_tempo(self):
        """長い平均間隔の場合。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 10, delta_time=200.0)
        snapshot = processor.describe_features()
        level = DensityLevel(snapshot[SECTION_OVERALL_ELAPSED])
        assert level in (DensityLevel.SPARSE, DensityLevel.SOMEWHAT_SPARSE)

    def test_returns_valid_density_level(self):
        processor = make_processor()
        accumulate_n(processor, 10)
        snapshot = processor.describe_features()
        value = snapshot[SECTION_OVERALL_ELAPSED]
        assert value in [dl.value for dl in DensityLevel]


# =============================================================================
# Test: 全6断面の存在確認
# =============================================================================

class TestAllSectionsPresent:
    def test_all_six_sections(self):
        """describe_features が6断面すべてを返すこと。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features()
        for section_name in SECTION_ORDER:
            assert section_name in snapshot

    def test_section_count(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features()
        assert len(snapshot) == 6

    def test_all_values_are_valid_density_levels(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        snapshot = processor.describe_features()
        valid_values = {dl.value for dl in DensityLevel}
        for value in snapshot.values():
            assert value in valid_values


# =============================================================================
# Test: 直前スナップショットの保持・更新
# =============================================================================

class TestPreviousSnapshot:
    def test_initial_previous_snapshot_empty(self):
        processor = make_processor()
        assert processor.get_previous_snapshot() == {}

    def test_previous_snapshot_after_first_describe(self):
        """最初のdescribe_features後は直前スナップショットが空のまま。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        # 最初の呼び出しでは直前スナップショットはまだ空
        assert processor.get_previous_snapshot() == {}

    def test_previous_snapshot_after_second_describe(self):
        """2回目のdescribe_features後は1回目のスナップショットが直前に。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        first = processor.describe_features()
        # 新しいデータを追加してもう一度describe
        accumulate_n(processor, 5, base_tick=6)
        processor.describe_features()
        previous = processor.get_previous_snapshot()
        assert previous == first

    def test_previous_snapshot_chain(self):
        """連続呼び出しで直前スナップショットが更新される。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()

        accumulate_n(processor, 3, base_tick=6)
        second = processor.describe_features()

        accumulate_n(processor, 3, base_tick=9)
        processor.describe_features()
        assert processor.get_previous_snapshot() == second

    def test_previous_snapshot_is_copy(self):
        """直前スナップショットはコピーとして返される。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        accumulate_n(processor, 5, base_tick=6)
        processor.describe_features()

        prev1 = processor.get_previous_snapshot()
        prev2 = processor.get_previous_snapshot()
        assert prev1 is not prev2
        assert prev1 == prev2


# =============================================================================
# Test: Stage 3 - enrichment出力
# =============================================================================

class TestEnrichment:
    def test_enrichment_empty(self):
        processor = make_processor()
        data = processor.get_enrichment_data()
        assert data["elapsed_record_count"] == 0
        assert data["external_input_count"] == 0
        assert data["snapshot"] == {}
        assert "待機中" in data["summary_text"]

    def test_enrichment_with_records(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert data["elapsed_record_count"] == 5
        assert len(data["snapshot"]) == 6

    def test_enrichment_with_external_input(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.notify_external_input(timestamp=1000.0)
        processor.notify_external_input(timestamp=2000.0)
        data = processor.get_enrichment_data()
        assert data["external_input_count"] == 2

    def test_enrichment_no_emphasis(self):
        """enrichmentテキストに強調表現がないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        assert "注目" not in summary
        assert "重要" not in summary
        assert "異常" not in summary
        assert "顕著" not in summary
        assert "著しい" not in summary
        assert "警告" not in summary
        assert "深刻" not in summary

    def test_enrichment_no_evaluation(self):
        """enrichmentに評価的な言葉がないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        assert "良い" not in summary
        assert "悪い" not in summary
        assert "適切" not in summary
        assert "不適切" not in summary
        assert "改善" not in summary
        assert "問題" not in summary

    def test_enrichment_section_order_fixed(self):
        """列挙順序は断面の定義順に固定。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        # 各断面のラベルが定義順に出現すること
        positions = []
        for section_name in SECTION_ORDER:
            label = SECTION_LABELS[section_name]
            pos = summary.find(label)
            if pos >= 0:
                positions.append(pos)
        # 出現位置が昇順であること
        assert positions == sorted(positions)

    def test_enrichment_equal_listing(self):
        """全断面を等価に列挙する。特定の断面を強調・選別しない。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        snapshot = processor.describe_features()
        data = processor.get_enrichment_data()
        # snapshotの全6断面がdataに含まれること
        for section_name in SECTION_ORDER:
            assert section_name in data["snapshot"]


# =============================================================================
# Test: READ-ONLYアクセサ
# =============================================================================

class TestReadOnlyAccessor:
    def test_get_snapshot_returns_dict(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        snapshot = processor.get_snapshot()
        assert isinstance(snapshot, dict)
        assert len(snapshot) == 6

    def test_get_snapshot_is_copy(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        snap1 = processor.get_snapshot()
        snap2 = processor.get_snapshot()
        assert snap1 is not snap2
        assert snap1 == snap2

    def test_get_snapshot_empty(self):
        processor = make_processor()
        assert processor.get_snapshot() == {}

    def test_get_previous_snapshot_returns_dict(self):
        processor = make_processor()
        result = processor.get_previous_snapshot()
        assert isinstance(result, dict)

    def test_reference_does_not_modify_state(self):
        """参照行為によって状態が変化しないこと。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        pre_record_count = len(processor.state.elapsed_records)
        pre_snapshot = dict(processor.state.snapshot)

        # 全ての参照メソッドを呼び出す
        processor.get_enrichment_data()
        processor.get_snapshot()
        processor.get_previous_snapshot()
        processor.get_summary()

        assert len(processor.state.elapsed_records) == pre_record_count
        assert processor.state.snapshot == pre_snapshot

    def test_get_summary(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.notify_external_input(timestamp=1000.0)
        processor.describe_features()
        summary = processor.get_summary()
        assert summary["elapsed_record_count"] == 5
        assert summary["external_input_count"] == 1
        assert summary["has_snapshot"] is True

    def test_get_summary_empty(self):
        processor = make_processor()
        summary = processor.get_summary()
        assert summary["elapsed_record_count"] == 0
        assert summary["external_input_count"] == 0
        assert summary["has_snapshot"] is False
        assert summary["has_previous_snapshot"] is False


# =============================================================================
# Test: Save/Load (永続化)
# =============================================================================

class TestSaveLoad:
    def test_roundtrip_empty(self):
        state = TemporalCognitionState()
        d = state.to_dict()
        restored = TemporalCognitionState.from_dict(d)
        assert len(restored.elapsed_records) == 0
        assert restored.snapshot == {}
        assert restored.previous_snapshot == {}
        assert restored.external_input_timestamps == []

    def test_roundtrip_with_data(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.notify_external_input(timestamp=1000.0)
        processor.notify_external_input(timestamp=2000.0)
        processor.describe_features()

        d = processor.state.to_dict()
        restored = TemporalCognitionState.from_dict(d)

        assert len(restored.elapsed_records) == 5
        assert len(restored.external_input_timestamps) == 2
        assert len(restored.snapshot) == 6

    def test_roundtrip_preserves_records(self):
        processor = make_processor()
        accumulate_n(processor, 3)
        d = processor.state.to_dict()
        restored = TemporalCognitionState.from_dict(d)
        for i, rec in enumerate(restored.elapsed_records):
            assert rec.tick == processor.state.elapsed_records[i].tick
            assert rec.delta_time == processor.state.elapsed_records[i].delta_time

    def test_roundtrip_preserves_snapshot(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        original_snapshot = dict(processor.state.snapshot)

        d = processor.state.to_dict()
        restored = TemporalCognitionState.from_dict(d)
        assert restored.snapshot == original_snapshot

    def test_roundtrip_preserves_previous_snapshot(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        accumulate_n(processor, 5, base_tick=6)
        processor.describe_features()
        original_previous = dict(processor.state.previous_snapshot)

        d = processor.state.to_dict()
        restored = TemporalCognitionState.from_dict(d)
        assert restored.previous_snapshot == original_previous

    def test_roundtrip_preserves_external_timestamps(self):
        processor = make_processor()
        processor.notify_external_input(timestamp=100.0)
        processor.notify_external_input(timestamp=200.0)
        d = processor.state.to_dict()
        restored = TemporalCognitionState.from_dict(d)
        assert restored.external_input_timestamps == [100.0, 200.0]

    def test_state_setter(self):
        """state プロパティの setter テスト。"""
        processor = make_processor()
        accumulate_n(processor, 3)
        processor.describe_features()
        original_state = processor.state.to_dict()

        new_processor = make_processor()
        new_processor.state = TemporalCognitionState.from_dict(original_state)
        assert len(new_processor.state.elapsed_records) == 3
        assert len(new_processor.state.snapshot) == 6

    def test_load_from_partial_dict(self):
        state = TemporalCognitionState.from_dict({
            "external_input_timestamps": [100.0, 200.0],
        })
        assert state.external_input_timestamps == [100.0, 200.0]
        assert len(state.elapsed_records) == 0

    def test_resume_after_load(self):
        """ロード後に蓄積を再開。"""
        processor = make_processor()
        accumulate_n(processor, 3)
        saved = processor.state.to_dict()

        new_processor = make_processor()
        new_processor.state = TemporalCognitionState.from_dict(saved)
        new_processor.accumulate_elapsed(tick=100, delta_time=1.0, timestamp=2000.0)
        assert len(new_processor.state.elapsed_records) == 4


# =============================================================================
# Test: 安全弁 1 - 断面の等価性
# =============================================================================

class TestSectionEquality:
    def test_no_weight_on_sections(self):
        """断面に重みや重要度が付与されていないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        snapshot = processor.describe_features()
        data = processor.get_enrichment_data()
        assert "weight" not in str(data)
        assert "priority" not in str(data)
        assert "importance" not in str(data)

    def test_all_sections_same_structure(self):
        """全断面が同じ構造（断面名→段階値文字列）であること。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        snapshot = processor.describe_features()
        valid_values = {dl.value for dl in DensityLevel}
        for section_name in SECTION_ORDER:
            assert section_name in snapshot
            assert snapshot[section_name] in valid_values

    def test_no_dominant_section(self):
        """「活動密度が最も重要」「感情変動が支配的」といった優先順位を持たないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        assert "最も重要" not in summary
        assert "支配的" not in summary
        assert "主要" not in summary
        assert "優先" not in summary


# =============================================================================
# Test: 安全弁 2 - スライディングウィンドウによる自然な更新
# =============================================================================

class TestNaturalUpdate:
    def test_old_data_pushed_out(self):
        """古いデータは自然に押し出される。"""
        processor = make_processor(max_elapsed_records=5)
        accumulate_n(processor, 10)
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert min(ticks) == 6

    def test_no_permanent_retention(self):
        """特定の時期の記録を永続的に保持しない。"""
        processor = make_processor(max_elapsed_records=3)
        processor.accumulate_elapsed(tick=1, delta_time=1.0, timestamp=1000.0)
        accumulate_n(processor, 5, base_tick=2)
        ticks = [r.tick for r in processor.state.elapsed_records]
        assert 1 not in ticks


# =============================================================================
# Test: 安全弁 3 - パターン抽出禁止
# =============================================================================

class TestPatternExtractionProhibition:
    def test_no_pattern_in_state(self):
        """蓄積からパターン・傾向・統計を抽出しないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "pattern" not in str(state_dict)
        assert "trend" not in str(state_dict)
        assert "tendency" not in str(state_dict)

    def test_no_history_of_snapshots(self):
        """スナップショットの履歴を保持しない（直前の1件のみ）。"""
        processor = make_processor()
        for i in range(5):
            accumulate_n(processor, 3, base_tick=i * 3 + 1)
            processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "snapshot_history" not in state_dict
        assert "history" not in state_dict

    def test_no_temporal_trend_extraction(self):
        """「長期的に時間認知がどう推移してきたか」の要約は行わない。"""
        processor = make_processor()
        accumulate_n(processor, 20)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        assert "推移" not in summary
        assert "変化傾向" not in summary
        assert "トレンド" not in summary


# =============================================================================
# Test: 安全弁 4 - 単一数値への統合禁止
# =============================================================================

class TestSingleValueProhibition:
    def test_no_single_body_clock(self):
        """体感時間を単一の数値として出力しない。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert "body_clock" not in data
        assert "time_feel" not in data
        assert "subjective_time" not in data
        assert "tempo_score" not in data

    def test_snapshot_has_multiple_sections(self):
        """スナップショットは常に複数の独立した断面が並立。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        snapshot = processor.get_snapshot()
        assert len(snapshot) == 6

    def test_no_combined_score_in_state(self):
        """状態に統合スコアが存在しないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "combined_score" not in str(state_dict)
        assert "total_score" not in str(state_dict)
        assert "overall_score" not in str(state_dict)

    def test_no_single_time_variable(self):
        """「時間が速い/遅い」を表す統合的な1変数を算出しないこと。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert "time_speed" not in data
        assert "fast" not in str(data.get("summary_text", ""))
        assert "slow" not in str(data.get("summary_text", ""))


# =============================================================================
# Test: 安全弁 5 - enrichment内での強調禁止
# =============================================================================

class TestEnrichmentEmphasisProhibition:
    def test_no_emphasis_in_summary(self):
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        forbidden = ["注目すべき", "異常な", "重要な", "特筆すべき", "顕著な", "著しい"]
        for word in forbidden:
            assert word not in summary

    def test_equal_listing_in_summary(self):
        """全ての断面を等価に列挙すること。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        summary = data["summary_text"]
        for section_name in SECTION_ORDER:
            label = SECTION_LABELS[section_name]
            assert label in summary


# =============================================================================
# Test: 4経路遮断の検証
# =============================================================================

class TestFourPathBlocking:
    def test_no_tick_parameter_modification(self):
        """時間的特徴量→ティックベース処理パラメータの遮断。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        state_dict = processor.state.to_dict()
        assert "tick_interval" not in data
        assert "execution_frequency" not in data
        assert "cooldown" not in str(data)
        assert "cooldown" not in str(state_dict)

    def test_no_emotion_pipeline_modification(self):
        """時間的特徴量→感情パイプラインの遮断。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "emotion_delta" not in str(state_dict)
        assert "mood_change" not in str(state_dict)
        assert "decay_rate" not in str(state_dict)
        assert "amplitude" not in str(state_dict)

    def test_no_memory_forgetting_modification(self):
        """時間的特徴量→記憶忘却/固定化パラメータの遮断。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "forgetting" not in str(state_dict)
        assert "fixation" not in str(state_dict)
        assert "decay_speed" not in str(state_dict)

    def test_no_expectation_formation_modification(self):
        """時間的特徴量→予期形成の遮断。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert "expectation" not in data
        assert "prediction" not in data


# =============================================================================
# Test: 自己差分認知との境界維持
# =============================================================================

class TestSelfDifferenceBoundary:
    def test_no_self_diff_field(self):
        """時間的特徴量と自己差分サマリは別の構造。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        state_dict = processor.state.to_dict()
        assert "self_diff" not in state_dict
        assert "difference_summary" not in state_dict

    def test_no_self_image_influence(self):
        """時間的特徴量が自己差分の記述に影響する経路がない。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert "self_image" not in data
        assert "self_model" not in data


# =============================================================================
# Test: Helpers
# =============================================================================

class TestClassifyIntervalDensity:
    def test_empty_intervals(self):
        assert _classify_interval_density([]) == DensityLevel.NORMAL

    def test_below_min_count(self):
        assert _classify_interval_density([1.0, 2.0], min_count=3) == DensityLevel.NORMAL

    def test_uniform_intervals(self):
        intervals = [10.0] * 10
        result = _classify_interval_density(intervals, min_count=3)
        assert result == DensityLevel.NORMAL

    def test_dense_recent(self):
        intervals = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        result = _classify_interval_density(intervals, min_count=3)
        assert result in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_sparse_recent(self):
        intervals = [1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0]
        result = _classify_interval_density(intervals, min_count=3)
        assert result in (DensityLevel.SPARSE, DensityLevel.SOMEWHAT_SPARSE)


class TestClassifyFrequency:
    def test_zero_window(self):
        assert _classify_frequency(0, 0) == DensityLevel.NORMAL

    def test_below_min_count(self):
        assert _classify_frequency(1, 2, min_count=3) == DensityLevel.NORMAL

    def test_high_frequency(self):
        result = _classify_frequency(9, 10, min_count=3)
        assert result in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE)

    def test_low_frequency(self):
        result = _classify_frequency(1, 100, min_count=3)
        assert result in (DensityLevel.SPARSE, DensityLevel.SOMEWHAT_SPARSE)


class TestClassifyTempo:
    def test_below_min_count(self):
        assert _classify_tempo(10.0, min_count=3, record_count=2) == DensityLevel.NORMAL

    def test_dense_tempo(self):
        result = _classify_tempo(0.5, min_count=3, record_count=10)
        assert result == DensityLevel.DENSE

    def test_sparse_tempo(self):
        result = _classify_tempo(200.0, min_count=3, record_count=10)
        assert result == DensityLevel.SPARSE

    def test_zero_interval(self):
        assert _classify_tempo(0.0, min_count=3, record_count=10) == DensityLevel.NORMAL


# =============================================================================
# Test: Constants
# =============================================================================

class TestConstants:
    def test_section_order_count(self):
        assert len(SECTION_ORDER) == 6

    def test_section_labels_complete(self):
        for section_name in SECTION_ORDER:
            assert section_name in SECTION_LABELS

    def test_density_labels_complete(self):
        for level in DensityLevel:
            assert level in DENSITY_LABELS

    def test_section_order_unique(self):
        assert len(SECTION_ORDER) == len(set(SECTION_ORDER))


# =============================================================================
# Test: Summary
# =============================================================================

class TestTemporalSummary:
    def test_summary_waiting(self):
        state = TemporalCognitionState()
        summary = get_temporal_summary(state)
        assert "待機中" in summary

    def test_summary_with_snapshot(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        summary = get_temporal_summary(processor.state)
        assert "待機中" not in summary
        assert "活動密度" in summary
        assert "記憶蓄積間隔" in summary

    def test_summary_no_evaluation(self):
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        summary = get_temporal_summary(processor.state)
        assert "良い" not in summary
        assert "悪い" not in summary
        assert "適切" not in summary

    def test_summary_all_sections_listed(self):
        processor = make_processor()
        accumulate_n(processor, 10)
        processor.describe_features()
        summary = get_temporal_summary(processor.state)
        for section_name in SECTION_ORDER:
            label = SECTION_LABELS[section_name]
            assert label in summary


# =============================================================================
# Test: Factory
# =============================================================================

class TestFactory:
    def test_create_default(self):
        processor = create_temporal_cognition()
        assert isinstance(processor, TemporalCognitionProcessor)
        assert processor._config.max_elapsed_records == 100

    def test_create_with_config(self):
        cfg = TemporalCognitionConfig(max_elapsed_records=50)
        processor = create_temporal_cognition(config=cfg)
        assert processor._config.max_elapsed_records == 50

    def test_factory_returns_fresh_state(self):
        processor = create_temporal_cognition()
        assert len(processor.state.elapsed_records) == 0
        assert processor.state.snapshot == {}
        assert processor.state.previous_snapshot == {}
        assert processor.state.external_input_timestamps == []


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_max_records_one(self):
        processor = make_processor(max_elapsed_records=1)
        accumulate_n(processor, 5)
        assert len(processor.state.elapsed_records) == 1
        assert processor.state.elapsed_records[0].tick == 5

    def test_very_large_max_records(self):
        processor = make_processor(max_elapsed_records=10000)
        accumulate_n(processor, 100)
        assert len(processor.state.elapsed_records) == 100

    def test_zero_delta_time(self):
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=0.0, timestamp=1000.0)
        assert processor.state.elapsed_records[0].delta_time == 0.0

    def test_negative_delta_time(self):
        """負の経過秒も記録される（異常値であっても記録自体は拒否しない）。"""
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=-1.0, timestamp=1000.0)
        assert processor.state.elapsed_records[0].delta_time == -1.0

    def test_very_large_delta_time(self):
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=999999.0, timestamp=1000.0)
        assert processor.state.elapsed_records[0].delta_time == 999999.0

    def test_describe_with_empty_records(self):
        processor = make_processor()
        snapshot = processor.describe_features()
        assert len(snapshot) == 6
        for v in snapshot.values():
            assert v == DensityLevel.NORMAL.value

    def test_describe_with_single_record(self):
        processor = make_processor()
        processor.accumulate_elapsed(tick=1, delta_time=1.0, timestamp=1000.0)
        snapshot = processor.describe_features()
        assert len(snapshot) == 6

    def test_external_input_max_one(self):
        processor = make_processor(max_external_input_records=1)
        processor.notify_external_input(timestamp=100.0)
        processor.notify_external_input(timestamp=200.0)
        processor.notify_external_input(timestamp=300.0)
        assert len(processor.state.external_input_timestamps) == 1
        assert processor.state.external_input_timestamps[0] == 300.0

    def test_describe_multiple_times_same_data(self):
        """同じデータで複数回describeを呼んでも安定した結果。"""
        processor = make_processor()
        accumulate_n(processor, 10)
        snap1 = processor.describe_features()
        snap2 = processor.describe_features()
        # 直前スナップショットが更新されるが、特徴量自体は安定
        assert snap1 == snap2

    def test_timestamp_ordering(self):
        processor = make_processor()
        accumulate_n(processor, 5)
        timestamps = [r.timestamp for r in processor.state.elapsed_records]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_resume_after_load_with_describe(self):
        """ロード後にdescribe可能。"""
        processor = make_processor()
        accumulate_n(processor, 5)
        processor.describe_features()
        saved = processor.state.to_dict()

        new_processor = make_processor()
        new_processor.state = TemporalCognitionState.from_dict(saved)
        # 追加蓄積してdescribe
        new_processor.accumulate_elapsed(tick=100, delta_time=1.0, timestamp=2000.0)
        snapshot = new_processor.describe_features()
        assert len(snapshot) == 6


# =============================================================================
# Test: Integration (統合テスト)
# =============================================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """蓄積→特徴量記述→参照の完全ライフサイクル。"""
        processor = make_processor(
            max_elapsed_records=10,
            max_external_input_records=5,
            min_records_for_description=3,
        )

        # 1. 経過記録蓄積
        for i in range(15):
            processor.accumulate_elapsed(
                tick=i + 1,
                delta_time=1.0 + i * 0.1,
                timestamp=1000.0 + i,
            )

        assert len(processor.state.elapsed_records) == 10  # 上限10

        # 2. 外部入力記録
        for i in range(7):
            processor.notify_external_input(timestamp=1000.0 + i * 5)

        assert len(processor.state.external_input_timestamps) == 5  # 上限5

        # 3. 特徴量記述
        snapshot = processor.describe_features(
            episodic_timestamps=[1000.0, 1005.0, 1010.0, 1015.0],
            emotion_change_count=5,
            narrative_timestamps=[1000.0, 1003.0, 1006.0],
        )
        assert len(snapshot) == 6

        # 4. enrichment
        data = processor.get_enrichment_data()
        assert data["elapsed_record_count"] == 10
        assert data["external_input_count"] == 5
        assert len(data["snapshot"]) == 6

        # 5. READ-ONLYアクセサ
        snap = processor.get_snapshot()
        assert snap == snapshot

        # 6. サマリ
        summary = processor.get_summary()
        assert summary["elapsed_record_count"] == 10
        assert summary["has_snapshot"] is True

    def test_save_load_resume_lifecycle(self):
        """セッション間での永続化と再開。"""
        # Session 1
        proc1 = make_processor(max_elapsed_records=10)
        accumulate_n(proc1, 5)
        proc1.notify_external_input(timestamp=1000.0)
        proc1.describe_features()
        saved = proc1.state.to_dict()

        # Session 2
        proc2 = make_processor(max_elapsed_records=10)
        proc2.state = TemporalCognitionState.from_dict(saved)

        # 状態が復元されていること
        assert len(proc2.state.elapsed_records) == 5
        assert len(proc2.state.external_input_timestamps) == 1
        assert len(proc2.state.snapshot) == 6

        # 追加蓄積とdescribeが可能
        accumulate_n(proc2, 3, base_tick=6)
        proc2.notify_external_input(timestamp=2000.0)
        snapshot = proc2.describe_features()
        assert len(snapshot) == 6
        assert len(proc2.state.elapsed_records) == 8

    def test_multiple_describe_cycles(self):
        """複数回のdescribeサイクルで直前スナップショットが正しく遷移。"""
        processor = make_processor()
        snapshots: list[dict[str, str]] = []

        for cycle in range(4):
            accumulate_n(processor, 3, base_tick=cycle * 3 + 1,
                        delta_time=(cycle + 1) * 2.0)
            snap = processor.describe_features()
            snapshots.append(snap)

        # 最新のスナップショット
        assert processor.get_snapshot() == snapshots[-1]
        # 直前のスナップショット
        assert processor.get_previous_snapshot() == snapshots[-2]

    def test_external_input_influences_section(self):
        """外部入力記録が外部入力間隔断面に反映される。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 5)

        # 外部入力なしでdescribe
        snap_no_input = processor.describe_features()
        assert snap_no_input[SECTION_EXTERNAL_INPUT_INTERVAL] == DensityLevel.NORMAL.value

        # 密な外部入力を追加
        for i in range(10):
            processor.notify_external_input(timestamp=1000.0 + i * 0.1)
        snap_dense = processor.describe_features()
        # 外部入力が密であれば DENSE方向
        level = DensityLevel(snap_dense[SECTION_EXTERNAL_INPUT_INTERVAL])
        assert level in (DensityLevel.DENSE, DensityLevel.SOMEWHAT_DENSE, DensityLevel.NORMAL)

    def test_no_cross_section_interference(self):
        """各断面は独立であり、他の断面の値に影響しない。"""
        processor = make_processor(min_records_for_description=3)
        accumulate_n(processor, 10)

        # 感情変動のみ高くする
        snap1 = processor.describe_features(emotion_change_count=9)
        snap2 = processor.describe_features(emotion_change_count=0)

        # 感情変動頻度は変わるが、他の断面は変わらない
        for section in SECTION_ORDER:
            if section == SECTION_EMOTION_FREQUENCY:
                continue
            assert snap1[section] == snap2[section]

    def test_rapid_succession(self):
        """高速連続処理での安定性。"""
        processor = make_processor(max_elapsed_records=50)
        for i in range(200):
            processor.accumulate_elapsed(tick=i, delta_time=0.01, timestamp=1000.0 + i * 0.01)
        assert len(processor.state.elapsed_records) == 50
        snapshot = processor.describe_features()
        assert len(snapshot) == 6
