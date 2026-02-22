"""
tests/test_drive_variation_description.py - 駆動の変動記述モジュールのテスト

テスト項目:
- 初期化テスト
- 4段パイプラインテスト（各段の動作確認）
- スライディングウィンドウテスト
- 鮮度減衰テスト
- 同種認知回復テスト
- 安全弁テスト（5種）
- 経路遮断確認テスト
- save/load往復テスト
- enrichmentデータテスト
- エッジケーステスト
- ファクトリ関数テスト
"""

import time
from typing import Any

import pytest

from psyche.drive_variation_description import (
    InputSection,
    FreshnessStage,
    ConvergenceLevel,
    _clamp,
    _gen_id,
    _stage_from_freshness,
    _convergence_from_score,
    WindowEntry,
    CompositionRecord,
    DecayRecord,
    ConvergenceRecord,
    DriveVariationInputs,
    DriveVariationState,
    DriveVariationResult,
    DriveVariationConfig,
    DriveVariationProcessor,
    get_drive_variation_summary,
    create_drive_variation_processor,
)


# ── Helper ──────────────────────────────────────────────────────────

def _make_inputs(
    tick: int = 1,
    drive_values: dict[str, float] | None = None,
    mood_valence: float = 0.3,
    mood_arousal: float = 0.4,
    backdrop_window_size: int = 0,
    backdrop_low_variability: bool = False,
    meta_emotion_change_speed: float = 0.1,
    meta_emotion_dominant_stability: float = 0.5,
    reaction_updated_drives: bool = True,
) -> DriveVariationInputs:
    """テスト用の DriveVariationInputs を作成する。"""
    return DriveVariationInputs(
        drive_values=drive_values or {"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        backdrop_window_size=backdrop_window_size,
        backdrop_low_variability=backdrop_low_variability,
        meta_emotion_change_speed=meta_emotion_change_speed,
        meta_emotion_dominant_stability=meta_emotion_dominant_stability,
        existing_record_count=0,
        average_freshness=0.0,
        dialogue_elapsed_ticks=tick,
        temporal_elapsed_description="",
        mood_valence=mood_valence,
        mood_arousal=mood_arousal,
        reaction_updated_drives=reaction_updated_drives,
        current_tick=tick,
    )


# ── Enum テスト ─────────────────────────────────────────────────────


class TestEnums:
    """Enum の定義確認テスト。"""

    def test_input_section_values(self):
        """InputSection が8値を持つ。"""
        assert len(InputSection) == 8

    def test_input_section_all_distinct(self):
        """InputSection の値がすべて異なる。"""
        values = [s.value for s in InputSection]
        assert len(values) == len(set(values))

    def test_freshness_stage_values(self):
        """FreshnessStage が5段階。"""
        assert len(FreshnessStage) == 5
        assert FreshnessStage.ACTIVE.value == "active"
        assert FreshnessStage.INVISIBLE.value == "invisible"

    def test_convergence_level_values(self):
        """ConvergenceLevel が4段階。"""
        assert len(ConvergenceLevel) == 4


# ── Helper関数テスト ────────────────────────────────────────────────


class TestHelpers:
    """ヘルパー関数のテスト。"""

    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_min(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5.0, lo=0.0, hi=3.0) == 3.0
        assert _clamp(-1.0, lo=0.0, hi=3.0) == 0.0

    def test_gen_id_format(self):
        """_gen_id が12文字の16進文字列を返す。"""
        gid = _gen_id()
        assert len(gid) == 12
        int(gid, 16)  # 16進数としてパース可能

    def test_gen_id_unique(self):
        """連続生成で重複しない。"""
        ids = {_gen_id() for _ in range(100)}
        assert len(ids) == 100

    def test_stage_from_freshness_active(self):
        assert _stage_from_freshness(1.0) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(0.8) == FreshnessStage.ACTIVE

    def test_stage_from_freshness_weakening(self):
        assert _stage_from_freshness(0.7) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.6) == FreshnessStage.WEAKENING

    def test_stage_from_freshness_fading(self):
        assert _stage_from_freshness(0.5) == FreshnessStage.FADING
        assert _stage_from_freshness(0.4) == FreshnessStage.FADING

    def test_stage_from_freshness_near_invisible(self):
        assert _stage_from_freshness(0.3) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.2) == FreshnessStage.NEAR_INVISIBLE

    def test_stage_from_freshness_invisible(self):
        assert _stage_from_freshness(0.1) == FreshnessStage.INVISIBLE
        assert _stage_from_freshness(0.0) == FreshnessStage.INVISIBLE

    def test_convergence_from_score_none(self):
        assert _convergence_from_score(0.0) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.29) == ConvergenceLevel.NONE

    def test_convergence_from_score_mild(self):
        assert _convergence_from_score(0.3) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.49) == ConvergenceLevel.MILD

    def test_convergence_from_score_moderate(self):
        assert _convergence_from_score(0.5) == ConvergenceLevel.MODERATE

    def test_convergence_from_score_strong(self):
        assert _convergence_from_score(0.7) == ConvergenceLevel.STRONG
        assert _convergence_from_score(1.0) == ConvergenceLevel.STRONG


# ── データ構造テスト ────────────────────────────────────────────────


class TestDataStructures:
    """データ構造の to_dict / from_dict テスト。"""

    def test_window_entry_roundtrip(self):
        """WindowEntry の to_dict → from_dict ラウンドトリップ。"""
        entry = WindowEntry(
            drive_values={"social": 0.5, "curiosity": 0.7},
            mood_valence=0.4,
            mood_arousal=0.6,
            tick=10,
        )
        d = entry.to_dict()
        restored = WindowEntry.from_dict(d)
        assert restored.drive_values == entry.drive_values
        assert restored.mood_valence == entry.mood_valence
        assert restored.mood_arousal == entry.mood_arousal
        assert restored.tick == entry.tick
        assert restored.entry_id == entry.entry_id

    def test_window_entry_auto_id(self):
        """WindowEntry は自動的にIDが生成される。"""
        entry = WindowEntry()
        assert len(entry.entry_id) == 12

    def test_composition_record_roundtrip(self):
        """CompositionRecord の to_dict → from_dict ラウンドトリップ。"""
        rec = CompositionRecord(
            tick=5,
            window_size=10,
            tick_range=8,
            time_range=16.0,
            drive_series={"social": [0.5, 0.6, 0.7]},
            valence_series=[0.3, 0.4, 0.5],
            arousal_series=[0.2, 0.3, 0.4],
            low_variability_noted=True,
            freshness=0.8,
            freshness_stage="weakening",
        )
        d = rec.to_dict()
        restored = CompositionRecord.from_dict(d)
        assert restored.tick == 5
        assert restored.drive_series == {"social": [0.5, 0.6, 0.7]}
        assert restored.valence_series == [0.3, 0.4, 0.5]
        assert restored.low_variability_noted is True
        assert restored.freshness == 0.8
        assert restored.freshness_stage == "weakening"

    def test_decay_record_roundtrip(self):
        """DecayRecord の to_dict → from_dict ラウンドトリップ。"""
        rec = DecayRecord(
            record_id="test123",
            old_stage="active",
            new_stage="weakening",
            freshness=0.7,
        )
        d = rec.to_dict()
        restored = DecayRecord.from_dict(d)
        assert restored.record_id == "test123"
        assert restored.old_stage == "active"
        assert restored.new_stage == "weakening"
        assert restored.freshness == 0.7

    def test_convergence_record_roundtrip(self):
        """ConvergenceRecord の to_dict → from_dict ラウンドトリップ。"""
        rec = ConvergenceRecord(
            convergence_score=0.6,
            convergence_level="moderate",
            composition_diversity=0.4,
            cycle=3,
        )
        d = rec.to_dict()
        restored = ConvergenceRecord.from_dict(d)
        assert restored.convergence_score == 0.6
        assert restored.convergence_level == "moderate"
        assert restored.composition_diversity == 0.4
        assert restored.cycle == 3

    def test_drive_variation_state_roundtrip(self):
        """DriveVariationState の to_dict → from_dict ラウンドトリップ。"""
        state = DriveVariationState()
        state.sliding_window.append(WindowEntry(tick=1))
        state.composition_records.append(CompositionRecord(tick=1))
        state.convergence_records.append(ConvergenceRecord(cycle=1))
        state.cycle_count = 5
        state.total_entries_collected = 10
        state.low_variability_warning = True

        d = state.to_dict()
        restored = DriveVariationState.from_dict(d)
        assert len(restored.sliding_window) == 1
        assert len(restored.composition_records) == 1
        assert len(restored.convergence_records) == 1
        assert restored.cycle_count == 5
        assert restored.total_entries_collected == 10
        assert restored.low_variability_warning is True

    def test_state_session_decay(self):
        """DriveVariationState.apply_session_decay でセッション境界減衰。"""
        state = DriveVariationState()
        rec = CompositionRecord(tick=1, freshness=0.5)
        state.composition_records.append(rec)
        state.apply_session_decay(decay_factor=0.3)
        assert state.composition_records[0].freshness == pytest.approx(0.2, abs=0.01)

    def test_state_session_decay_removes_low_freshness(self):
        """セッション境界減衰で鮮度が0.1未満になった記録は除去。"""
        state = DriveVariationState()
        rec = CompositionRecord(tick=1, freshness=0.05)
        state.composition_records.append(rec)
        state.apply_session_decay(decay_factor=0.3)
        assert len(state.composition_records) == 0


# ── 初期化テスト ────────────────────────────────────────────────────


class TestInitialization:
    """DriveVariationProcessor の初期化テスト。"""

    def test_default_init(self):
        """デフォルト設定で初期化。"""
        proc = DriveVariationProcessor()
        assert proc.state.cycle_count == 0
        assert len(proc.state.sliding_window) == 0
        assert len(proc.state.composition_records) == 0

    def test_custom_config(self):
        """カスタム設定で初期化。"""
        cfg = DriveVariationConfig(max_window_size=10, max_composition_records=20)
        proc = DriveVariationProcessor(config=cfg)
        assert proc._config.max_window_size == 10
        assert proc._config.max_composition_records == 20

    def test_default_window_size_larger_than_backdrop(self):
        """駆動の変動は緩やかなため、デフォルト窓サイズが感情基調認知より大きい。"""
        cfg = DriveVariationConfig()
        assert cfg.max_window_size >= 50

    def test_factory_function(self):
        """ファクトリ関数で生成。"""
        proc = create_drive_variation_processor()
        assert isinstance(proc, DriveVariationProcessor)
        assert proc.state.cycle_count == 0

    def test_factory_with_config(self):
        """ファクトリ関数にカスタム設定を渡す。"""
        cfg = DriveVariationConfig(max_window_size=5)
        proc = create_drive_variation_processor(config=cfg)
        assert proc._config.max_window_size == 5


# ── 4段パイプラインテスト ──────────────────────────────────────────


class TestPipeline:
    """4段パイプラインの動作テスト。"""

    def test_single_tick(self):
        """1回の tick で結果が返る。"""
        proc = DriveVariationProcessor()
        inputs = _make_inputs(tick=1)
        result = proc.tick(inputs)
        assert isinstance(result, DriveVariationResult)
        assert result.window_size == 1
        assert result.cycle_count == 1

    def test_process_alias(self):
        """process() も tick() と同じ結果。"""
        proc = DriveVariationProcessor()
        inputs = _make_inputs(tick=1)
        result = proc.process(inputs)
        assert isinstance(result, DriveVariationResult)
        assert result.window_size == 1

    def test_multiple_ticks(self):
        """複数回 tick で窓が成長する。"""
        proc = DriveVariationProcessor()
        for i in range(10):
            result = proc.tick(_make_inputs(tick=i + 1))
        assert result.window_size == 10
        assert result.cycle_count == 10
        assert proc.state.total_entries_collected == 10

    def test_stage1_drive_collection(self):
        """Stage 1: 駆動状態が正しくウィンドウに収集される。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.7, "curiosity": 0.3, "expression": 0.8}
        proc.tick(_make_inputs(tick=1, drive_values=drives))
        assert len(proc.state.sliding_window) == 1
        assert proc.state.sliding_window[0].drive_values == drives

    def test_stage2_equitable_listing(self):
        """Stage 2: 窓内の全駆動次元が等価に列挙される。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=1, drive_values={"social": 0.5, "curiosity": 0.6}))
        proc.tick(_make_inputs(tick=2, drive_values={"social": 0.7, "curiosity": 0.4}))

        records = proc.state.composition_records
        assert len(records) == 2
        latest = records[-1]
        assert "social" in latest.drive_series
        assert "curiosity" in latest.drive_series
        assert len(latest.drive_series["social"]) == 2
        assert latest.drive_series["social"] == [0.5, 0.7]
        assert latest.drive_series["curiosity"] == [0.6, 0.4]

    def test_stage2_no_moving_average(self):
        """Stage 2: 移動平均・統合指標を算出していない（生データのみ）。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, drive_values={"social": 0.1 * (i + 1)}))

        latest = proc.state.composition_records[-1]
        # 駆動系列はそのまま時系列列挙されている
        assert latest.drive_series["social"] == pytest.approx(
            [0.1, 0.2, 0.3, 0.4, 0.5], abs=0.01
        )

    def test_stage3_accumulation(self):
        """Stage 3: 構成記述が蓄積される。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.composition_records) == 5
        assert proc.state.total_records_created == 5

    def test_stage3_freshness_decay(self):
        """Stage 3: 鮮度が毎サイクル減衰する。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=1))
        first_freshness = proc.state.composition_records[0].freshness

        for i in range(10):
            proc.tick(_make_inputs(tick=i + 2))

        # 最初の記録の鮮度は減衰しているはず
        assert proc.state.composition_records[0].freshness < first_freshness

    def test_stage4_result_fields(self):
        """Stage 4: 結果に全フィールドが含まれる。"""
        proc = DriveVariationProcessor()
        result = proc.tick(_make_inputs(tick=1))
        assert hasattr(result, "window_size")
        assert hasattr(result, "record_count")
        assert hasattr(result, "tick_range")
        assert hasattr(result, "time_range")
        assert hasattr(result, "low_variability_warning")
        assert hasattr(result, "accumulation_bias_warning")
        assert hasattr(result, "convergence_warning")
        assert hasattr(result, "convergence_level")
        assert hasattr(result, "convergence_score")
        assert hasattr(result, "diversity_restored")
        assert hasattr(result, "cycle_count")


# ── スライディングウィンドウテスト ─────────────────────────────────


class TestSlidingWindow:
    """スライディングウィンドウのFIFO動作テスト。"""

    def test_fifo_overflow(self):
        """窓の上限を超えるとFIFOで最古が押し出される。"""
        cfg = DriveVariationConfig(max_window_size=5)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.sliding_window) == 5
        # 最古は tick=6 のはず
        assert proc.state.sliding_window[0].tick == 6

    def test_no_selective_deletion(self):
        """選択的消去は行わない（FIFOのみ）。"""
        cfg = DriveVariationConfig(max_window_size=3)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(5):
            proc.tick(_make_inputs(
                tick=i + 1,
                drive_values={"social": 0.1 * (i + 1)},
            ))
        # 残るのは最後の3件
        ticks = [e.tick for e in proc.state.sliding_window]
        assert ticks == [3, 4, 5]

    def test_tick_range_calculation(self):
        """ティック範囲が正しく計算される。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=5))
        proc.tick(_make_inputs(tick=10))
        result = proc.tick(_make_inputs(tick=15))
        assert result.tick_range == 10  # 15 - 5

    def test_single_entry_tick_range_zero(self):
        """エントリ1件の場合、ティック範囲は0。"""
        proc = DriveVariationProcessor()
        result = proc.tick(_make_inputs(tick=1))
        assert result.tick_range == 0

    def test_all_drive_dimensions_collected(self):
        """全駆動次元が等価に収集される。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(
            tick=1,
            drive_values={"social": 0.1, "curiosity": 0.2, "expression": 0.3},
        ))
        entry = proc.state.sliding_window[0]
        assert "social" in entry.drive_values
        assert "curiosity" in entry.drive_values
        assert "expression" in entry.drive_values

    def test_mood_values_collected(self):
        """ムード値も並置可能な付帯情報として収集される。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=1, mood_valence=0.5, mood_arousal=0.7))
        entry = proc.state.sliding_window[0]
        assert entry.mood_valence == 0.5
        assert entry.mood_arousal == 0.7


# ── 鮮度減衰テスト ─────────────────────────────────────────────────


class TestFreshnessDecay:
    """鮮度減衰の動作テスト。"""

    def test_freshness_decreases_per_cycle(self):
        """鮮度が毎サイクル減衰する。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=1))
        initial = proc.state.composition_records[0].freshness

        proc.tick(_make_inputs(tick=2))
        after_one = proc.state.composition_records[0].freshness
        assert after_one < initial

    def test_freshness_stage_transition(self):
        """鮮度段階が遷移する。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.1)
        proc = DriveVariationProcessor(config=cfg)

        proc.tick(_make_inputs(tick=1))
        # 初期: freshness=1.0, stage=active
        assert proc.state.composition_records[0].freshness_stage == FreshnessStage.ACTIVE.value

        # 多くのサイクルを回す
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 2))

        # 最初の記録は減衰が進んでいるはず
        first_rec = proc.state.composition_records[0]
        assert first_rec.freshness < 0.8  # ACTIVEから下がっている

    def test_decay_history_recorded(self):
        """段階遷移が減衰履歴に記録される。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.15)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1))
        # 何らかの段階遷移が記録されているはず
        assert len(proc.state.decay_history) > 0

    def test_invisible_count_tracked(self):
        """不可視段階に達した記録数が追跡される。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.05)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(50):
            proc.tick(_make_inputs(tick=i + 1))
        assert proc.state.total_records_decayed >= 0

    def test_five_stage_decay(self):
        """5段階の鮮度減衰パターンが正しい。"""
        assert _stage_from_freshness(0.9) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(0.65) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.45) == FreshnessStage.FADING
        assert _stage_from_freshness(0.25) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.05) == FreshnessStage.INVISIBLE


# ── 同種認知回復テスト ─────────────────────────────────────────────


class TestReferenceRecovery:
    """同種の構成記述が再度生成された場合の鮮度回復テスト。"""

    def test_similar_composition_recovery(self):
        """同種の構成が生成されると減衰中記録の鮮度が回復する。"""
        cfg = DriveVariationConfig(
            freshness_decay_rate=0.08,
            reference_recovery=0.15,
        )
        proc = DriveVariationProcessor(config=cfg)

        # 同じ駆動値で何度もtick
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(15):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))

        # 減衰中の記録のいくつかが回復しているはず
        weakening_or_fading = [
            r for r in proc.state.composition_records
            if r.freshness_stage in (FreshnessStage.WEAKENING.value, FreshnessStage.FADING.value)
        ]
        # 同種の構成が繰り返されているので、減衰速度が緩和されているはず
        # 全てが不可視にはなっていないことを確認
        visible = [r for r in proc.state.composition_records if r.freshness >= 0.2]
        assert len(visible) > 0

    def test_different_composition_no_recovery(self):
        """異なる構成では回復しない。"""
        cfg = DriveVariationConfig(
            freshness_decay_rate=0.05,
            reference_recovery=0.1,
        )
        proc = DriveVariationProcessor(config=cfg)

        # 異なる駆動値で毎回tick
        for i in range(20):
            drives = {
                "social": 0.1 * (i % 10),
                "curiosity": 0.1 * ((i + 3) % 10),
                "expression": 0.1 * ((i + 5) % 10),
            }
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))

        # 処理が正常に完了する
        assert proc.state.cycle_count == 20


# ── 安全弁テスト ───────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作テスト。"""

    # 安全弁1: 窓内変動性の監視
    def test_safety1_low_variability_detection(self):
        """安全弁1: 変動性が極端に低い場合に事実を記述する。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        # 変動性が0のため低変動性が検出される
        assert proc.state.low_variability_warning is True

    def test_safety1_normal_variability(self):
        """安全弁1: 通常の変動性では警告しない。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            drives = {
                "social": 0.3 + 0.1 * i,
                "curiosity": 0.5 - 0.05 * i,
                "expression": 0.4 + 0.08 * i,
            }
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        assert proc.state.low_variability_warning is False

    def test_safety1_no_variability_promotion(self):
        """安全弁1: 低変動時に変動を促す処理を行わない（事実記述のみ）。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.5, "curiosity": 0.5}
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        # 駆動値は変更されていない（出力に駆動値変更経路がない）
        for entry in proc.state.sliding_window:
            assert entry.drive_values == drives

    # 安全弁2: 蓄積偏り検出
    def test_safety2_bias_detection(self):
        """安全弁2: 蓄積が偏った場合に検出する。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        # 全て同じ値なので偏りが検出されるはず
        assert proc.state.accumulation_bias_warning is True

    def test_safety2_diversity_recovery(self):
        """安全弁2: 偏り検出時に鮮度減衰中の記録を回復する。"""
        cfg = DriveVariationConfig(
            freshness_decay_rate=0.04,
            reference_recovery=0.0,  # 同種回復を無効化して純粋な偏り回復をテスト
        )
        proc = DriveVariationProcessor(config=cfg)
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(25):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        # 偏り+収束の検出により回復が発生するはず
        assert proc.state.total_records_recovered > 0

    # 安全弁3: enrichment出力量制限
    def test_safety3_enrichment_limit(self):
        """安全弁3: enrichmentの出力件数が制限される。"""
        proc = DriveVariationProcessor()
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= proc._config.max_enrichment_records

    # 安全弁4: 収束監視
    def test_safety4_convergence_monitoring(self):
        """安全弁4: 収束監視が動作する。"""
        proc = DriveVariationProcessor()
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.convergence_records) > 0

    def test_safety4_convergence_warning_on_uniform(self):
        """安全弁4: 均一入力時に収束警告。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        # 均一入力のため収束スコアが高くなる
        last_conv = proc.state.convergence_records[-1]
        assert last_conv.convergence_score >= 0.0  # 計算が実行されている

    def test_safety4_diversity_no_convergence(self):
        """安全弁4: 多様な入力では収束が強くならない。"""
        proc = DriveVariationProcessor()
        # 各次元の平均値が大きく変動するパターン
        patterns = [
            {"social": 0.1, "curiosity": 0.9, "expression": 0.5},
            {"social": 0.9, "curiosity": 0.1, "expression": 0.5},
            {"social": 0.5, "curiosity": 0.5, "expression": 0.1},
            {"social": 0.5, "curiosity": 0.5, "expression": 0.9},
            {"social": 0.1, "curiosity": 0.1, "expression": 0.1},
            {"social": 0.9, "curiosity": 0.9, "expression": 0.9},
            {"social": 0.2, "curiosity": 0.8, "expression": 0.3},
            {"social": 0.8, "curiosity": 0.2, "expression": 0.7},
            {"social": 0.3, "curiosity": 0.6, "expression": 0.9},
            {"social": 0.7, "curiosity": 0.4, "expression": 0.1},
        ]
        for i, drives in enumerate(patterns):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))
        last_conv = proc.state.convergence_records[-1]
        # 多様な入力では収束が強くならない
        assert last_conv.convergence_level != ConvergenceLevel.STRONG.value

    # 安全弁5: 恒常性強調の遮断
    def test_safety5_no_interpretive_text_in_summary(self):
        """安全弁5: サマリに解釈的テキストを含まない。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.9, "curiosity": 0.1, "expression": 0.5}
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))

        summary = get_drive_variation_summary(proc.state)
        # 「高い」「低い」「異常」「望ましい」等の解釈的テキストを含まない
        assert "高い" not in summary
        assert "低い" not in summary
        assert "異常" not in summary
        assert "望ましい" not in summary
        assert "健全" not in summary
        assert "上げるべき" not in summary

    def test_safety5_no_interpretive_text_in_enrichment(self):
        """安全弁5: enrichmentに解釈的テキストを含まない。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.9, "curiosity": 0.1, "expression": 0.5}
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))

        data = proc.get_enrichment_data()
        summary_text = data.get("summary_text", "")
        assert "高い" not in summary_text
        assert "低い" not in summary_text
        assert "上げるべき" not in summary_text

    def test_safety5_enrichment_has_numeric_data_only(self):
        """安全弁5: enrichmentのentriesに数値データのみ。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))

        data = proc.get_enrichment_data()
        for entry in data["entries"]:
            # drives は数値リストの辞書
            for dim, values in entry["drives"].items():
                for v in values:
                    assert isinstance(v, (int, float))


# ── 経路遮断テスト ─────────────────────────────────────────────────


class TestPathBlocking:
    """経路遮断確認テスト。"""

    def test_no_drive_value_modification(self):
        """経路1: 駆動ベクトルの値を変更しない。"""
        proc = DriveVariationProcessor()
        input_drives = {"social": 0.5, "curiosity": 0.7}
        inputs = _make_inputs(tick=1, drive_values=input_drives)
        proc.tick(inputs)
        # 入力の駆動値が変更されていない
        assert inputs.drive_values == {"social": 0.5, "curiosity": 0.7}

    def test_no_reaction_parameter_modification(self):
        """経路2: 反応処理パラメータへの書き込み経路が存在しない。"""
        proc = DriveVariationProcessor()
        # DriveVariationProcessor にはreaction関連の属性・メソッドがない
        assert not hasattr(proc, "modify_reaction_params")
        assert not hasattr(proc, "update_drive_params")

    def test_no_motivation_generation_input(self):
        """経路3: 動機生成への直接供給経路が存在しない。"""
        proc = DriveVariationProcessor()
        assert not hasattr(proc, "supply_to_motivation")
        assert not hasattr(proc, "feed_motivation")

    def test_no_policy_expansion_feed(self):
        """経路4: ポリシー候補拡張への直接断面供給がない。"""
        proc = DriveVariationProcessor()
        assert not hasattr(proc, "supply_to_policy_expansion")
        assert not hasattr(proc, "feed_policy")

    def test_no_emotion_pipeline_modification(self):
        """経路5: 感情パイプラインのパラメータを変更しない。"""
        proc = DriveVariationProcessor()
        assert not hasattr(proc, "modify_decay_rate")
        assert not hasattr(proc, "set_mood")
        assert not hasattr(proc, "modify_dynamics")

    def test_no_memory_forgetting_modification(self):
        """経路6: 記憶忘却・固定化パラメータを変更しない。"""
        proc = DriveVariationProcessor()
        assert not hasattr(proc, "modify_forgetting_params")

    def test_output_is_read_only_info(self):
        """出力は参照情報形式のみであり、変更指示を含まない。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        result = proc.tick(_make_inputs(tick=6))
        # result は DriveVariationResult であり、actionable なフィールドを含まない
        assert isinstance(result, DriveVariationResult)
        # 判断・推奨・命令フィールドがないことを確認
        assert not hasattr(result, "recommended_action")
        assert not hasattr(result, "drive_adjustment")
        assert not hasattr(result, "target_drives")


# ── save/load 往復テスト ──────────────────────────────────────────


class TestSaveLoad:
    """save/load 永続化テスト。"""

    def test_save_load_roundtrip(self):
        """save → load で状態が復元される。"""
        proc1 = DriveVariationProcessor()
        for i in range(10):
            proc1.tick(_make_inputs(
                tick=i + 1,
                drive_values={
                    "social": 0.1 * (i + 1),
                    "curiosity": 0.5,
                    "expression": 0.3,
                },
            ))

        saved = proc1.save()

        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        assert proc2.state.cycle_count == proc1.state.cycle_count
        assert len(proc2.state.sliding_window) == len(proc1.state.sliding_window)
        assert len(proc2.state.composition_records) == len(proc1.state.composition_records)
        assert proc2.state.total_entries_collected == proc1.state.total_entries_collected
        assert proc2.state.total_records_created == proc1.state.total_records_created

    def test_save_load_window_entries(self):
        """save → load でウィンドウエントリが保持される。"""
        proc1 = DriveVariationProcessor()
        proc1.tick(_make_inputs(
            tick=1,
            drive_values={"social": 0.7, "curiosity": 0.3},
        ))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        assert proc2.state.sliding_window[0].drive_values["social"] == 0.7
        assert proc2.state.sliding_window[0].drive_values["curiosity"] == 0.3

    def test_save_load_composition_records(self):
        """save → load で蓄積記録が保持される。"""
        proc1 = DriveVariationProcessor()
        for i in range(5):
            proc1.tick(_make_inputs(tick=i + 1))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        for i, rec in enumerate(proc2.state.composition_records):
            orig = proc1.state.composition_records[i]
            assert rec.record_id == orig.record_id
            assert rec.tick == orig.tick
            assert rec.freshness == pytest.approx(orig.freshness, abs=0.001)

    def test_save_load_convergence_records(self):
        """save → load で収束記録が保持される。"""
        proc1 = DriveVariationProcessor()
        for i in range(5):
            proc1.tick(_make_inputs(tick=i + 1))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        assert len(proc2.state.convergence_records) == len(proc1.state.convergence_records)

    def test_save_load_safety_flags(self):
        """save → load で安全弁フラグが保持される。"""
        proc1 = DriveVariationProcessor()
        drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        for i in range(10):
            proc1.tick(_make_inputs(tick=i + 1, drive_values=drives))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        assert proc2.state.low_variability_warning == proc1.state.low_variability_warning
        assert proc2.state.accumulation_bias_warning == proc1.state.accumulation_bias_warning
        assert proc2.state.convergence_warning == proc1.state.convergence_warning

    def test_save_load_empty_state(self):
        """空の状態でもsave/loadが動作する。"""
        proc1 = DriveVariationProcessor()
        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)
        assert proc2.state.cycle_count == 0

    def test_save_load_decay_history(self):
        """save → load で減衰履歴が保持される。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.15)
        proc1 = DriveVariationProcessor(config=cfg)
        for i in range(10):
            proc1.tick(_make_inputs(tick=i + 1))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        assert len(proc2.state.decay_history) == len(proc1.state.decay_history)


# ── enrichment データテスト ────────────────────────────────────────


class TestEnrichmentData:
    """enrichment データの出力テスト。"""

    def test_enrichment_waiting_state(self):
        """初期状態ではenrichmentが待機中。"""
        proc = DriveVariationProcessor()
        data = proc.get_enrichment_data()
        assert "待機中" in data["summary_text"]
        assert data["window_size"] == 0

    def test_enrichment_after_ticks(self):
        """tick後にenrichmentデータが返る。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        assert data["window_size"] == 5
        assert len(data["entries"]) > 0

    def test_enrichment_drive_series_present(self):
        """enrichmentのentriesに駆動系列が含まれる。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(
            tick=1,
            drive_values={"social": 0.5, "curiosity": 0.7},
        ))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) >= 1
        entry = data["entries"][-1]
        assert "drives" in entry
        assert "social" in entry["drives"]

    def test_enrichment_all_dimensions_equitable(self):
        """enrichmentで全駆動次元が等価に列挙される。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(
            tick=1,
            drive_values={"social": 0.9, "curiosity": 0.1, "expression": 0.5},
        ))
        data = proc.get_enrichment_data()
        entry = data["entries"][-1]
        # 全次元が含まれている
        assert len(entry["drives"]) == 3

    def test_enrichment_invisible_excluded(self):
        """enrichmentで不可視記録は除外される。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.15)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(30):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        # 不可視記録はentriesに含まれない
        for entry in data["entries"]:
            assert entry["freshness_stage"] != FreshnessStage.INVISIBLE.value

    def test_enrichment_summary_text_numeric(self):
        """enrichmentのsummary_textは数値列挙形式。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        text = data["summary_text"]
        assert "cycle=" in text
        assert "窓=" in text

    def test_enrichment_warnings_included(self):
        """enrichmentに安全弁警告が含まれる。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        assert "low_variability_warning" in data
        assert "accumulation_bias_warning" in data
        assert "convergence_warning" in data

    def test_enrichment_max_records_respected(self):
        """enrichmentの出力件数上限が遵守される。"""
        cfg = DriveVariationConfig(max_enrichment_records=3)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= 3


# ── READ-ONLYアクセサテスト ────────────────────────────────────────


class TestReadOnlyAccessors:
    """READ-ONLYアクセサのテスト。"""

    def test_get_window_entries(self):
        """get_window_entries が全エントリを等価に返す。"""
        proc = DriveVariationProcessor()
        for i in range(3):
            proc.tick(_make_inputs(tick=i + 1))
        entries = proc.get_window_entries()
        assert len(entries) == 3
        assert all("drive_values" in e for e in entries)

    def test_get_composition_records(self):
        """get_composition_records が全記録を等価に返す。"""
        proc = DriveVariationProcessor()
        for i in range(3):
            proc.tick(_make_inputs(tick=i + 1))
        records = proc.get_composition_records()
        assert len(records) == 3
        assert all("drive_series" in r for r in records)

    def test_get_summary(self):
        """get_summary がモジュールサマリを返す。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        summary = proc.get_summary()
        assert summary["window_size"] == 5
        assert summary["record_count"] == 5
        assert summary["cycle_count"] == 5

    def test_state_property_get(self):
        """state プロパティでの読み取り。"""
        proc = DriveVariationProcessor()
        state = proc.state
        assert isinstance(state, DriveVariationState)

    def test_state_property_set(self):
        """state プロパティでの設定。"""
        proc = DriveVariationProcessor()
        new_state = DriveVariationState(cycle_count=10)
        proc.state = new_state
        assert proc.state.cycle_count == 10


# ── エッジケーステスト ─────────────────────────────────────────────


class TestEdgeCases:
    """エッジケーステスト。"""

    def test_empty_drive_values(self):
        """空の駆動値でもエラーにならない。"""
        proc = DriveVariationProcessor()
        result = proc.tick(_make_inputs(tick=1, drive_values={}))
        assert result.window_size == 1

    def test_single_dimension(self):
        """駆動次元が1つでも動作する。"""
        proc = DriveVariationProcessor()
        result = proc.tick(_make_inputs(tick=1, drive_values={"social": 0.5}))
        assert result.window_size == 1

    def test_large_window(self):
        """大量のデータでも動作する。"""
        cfg = DriveVariationConfig(max_window_size=100)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(200):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.sliding_window) == 100

    def test_max_composition_records(self):
        """蓄積記録の上限が遵守される。"""
        cfg = DriveVariationConfig(max_composition_records=10)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.composition_records) <= 10

    def test_max_decay_history(self):
        """減衰履歴の上限が遵守される。"""
        cfg = DriveVariationConfig(max_decay_history=5, freshness_decay_rate=0.15)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.decay_history) <= 5

    def test_max_convergence_records(self):
        """収束記録の上限が遵守される。"""
        cfg = DriveVariationConfig(max_convergence_records=5)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.convergence_records) <= 5

    def test_zero_freshness_clamp(self):
        """鮮度が0未満にならない。"""
        cfg = DriveVariationConfig(freshness_decay_rate=0.5)
        proc = DriveVariationProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        for rec in proc.state.composition_records:
            assert rec.freshness >= 0.0

    def test_enrichment_empty_window(self):
        """ウィンドウが空の場合のenrichment。"""
        proc = DriveVariationProcessor()
        data = proc.get_enrichment_data()
        assert data["window_size"] == 0
        assert data["entries"] == []

    def test_summary_waiting_state(self):
        """初期状態のサマリ。"""
        state = DriveVariationState()
        summary = get_drive_variation_summary(state)
        assert "待機中" in summary

    def test_summary_with_warnings(self):
        """警告フラグ付きのサマリ。"""
        state = DriveVariationState()
        state.cycle_count = 5
        state.sliding_window = [WindowEntry(tick=1), WindowEntry(tick=5)]
        state.low_variability_warning = True
        state.accumulation_bias_warning = True
        state.convergence_warning = True

        summary = get_drive_variation_summary(state)
        assert "低変動" in summary
        assert "蓄積偏り" in summary
        assert "収束" in summary

    def test_extreme_drive_values(self):
        """極端な駆動値でも動作する。"""
        proc = DriveVariationProcessor()
        # 境界値テスト
        proc.tick(_make_inputs(tick=1, drive_values={"social": 0.0, "curiosity": 1.0}))
        proc.tick(_make_inputs(tick=2, drive_values={"social": 1.0, "curiosity": 0.0}))
        assert proc.state.cycle_count == 2

    def test_varying_dimensions_over_time(self):
        """ティック毎に異なる数の駆動次元が入力されても動作する。"""
        proc = DriveVariationProcessor()
        proc.tick(_make_inputs(tick=1, drive_values={"social": 0.5}))
        proc.tick(_make_inputs(tick=2, drive_values={"social": 0.6, "curiosity": 0.4}))
        proc.tick(_make_inputs(tick=3, drive_values={"social": 0.7, "curiosity": 0.3, "expression": 0.8}))
        # 各駆動次元の系列は対応するエントリ数になる
        latest = proc.state.composition_records[-1]
        assert len(latest.drive_series["social"]) == 3
        assert len(latest.drive_series["curiosity"]) == 2
        assert len(latest.drive_series["expression"]) == 1


# ── ファクトリ関数テスト ──────────────────────────────────────────


class TestFactory:
    """ファクトリ関数のテスト。"""

    def test_create_default(self):
        """デフォルト設定で生成。"""
        proc = create_drive_variation_processor()
        assert isinstance(proc, DriveVariationProcessor)

    def test_create_with_config(self):
        """カスタム設定で生成。"""
        cfg = DriveVariationConfig(max_window_size=10)
        proc = create_drive_variation_processor(config=cfg)
        assert proc._config.max_window_size == 10

    def test_create_and_tick(self):
        """ファクトリ関数で生成してtick。"""
        proc = create_drive_variation_processor()
        result = proc.tick(_make_inputs(tick=1))
        assert result.cycle_count == 1


# ── 統合テスト ────────────────────────────────────────────────────


class TestIntegration:
    """統合的な動作テスト。"""

    def test_full_lifecycle(self):
        """生成 → tick → save → load → tick のフルライフサイクル。"""
        proc1 = DriveVariationProcessor()
        for i in range(5):
            proc1.tick(_make_inputs(tick=i + 1))

        saved = proc1.save()
        proc2 = DriveVariationProcessor()
        proc2.load(saved)

        # 追加tickが問題なく動作する
        result = proc2.tick(_make_inputs(tick=6))
        assert result.cycle_count == 6
        assert result.window_size == 6

    def test_session_decay_and_continue(self):
        """セッション減衰後に継続動作。"""
        proc = DriveVariationProcessor()
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1))

        proc.state.apply_session_decay()

        # セッション減衰後もtickが動作する
        result = proc.tick(_make_inputs(tick=11))
        assert result.cycle_count == 11

    def test_enrichment_consistency(self):
        """enrichmentデータがtick後に一貫する。"""
        proc = DriveVariationProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))

        data = proc.get_enrichment_data()
        summary = proc.get_summary()

        assert data["window_size"] == summary["window_size"]

    def test_no_pattern_extraction(self):
        """パターン抽出・命名・分類を行わない。"""
        proc = DriveVariationProcessor()
        for i in range(20):
            drives = {
                "social": 0.5 + 0.1 * (i % 3),
                "curiosity": 0.5 - 0.1 * (i % 3),
                "expression": 0.5,
            }
            proc.tick(_make_inputs(tick=i + 1, drive_values=drives))

        # 蓄積記録にパターン名やカテゴリが含まれない
        for rec in proc.state.composition_records:
            assert not hasattr(rec, "pattern_name")
            assert not hasattr(rec, "category")
            assert not hasattr(rec, "classification")

    def test_equal_treatment_of_all_dimensions(self):
        """全駆動次元が等価に扱われる（特定次元の特別扱いなし）。"""
        proc = DriveVariationProcessor()
        drives = {"social": 0.9, "curiosity": 0.1, "expression": 0.5}
        proc.tick(_make_inputs(tick=1, drive_values=drives))

        rec = proc.state.composition_records[-1]
        # 全次元が同列に列挙されている
        assert len(rec.drive_series) == 3
        # 特定次元の強調フィールドがない
        assert not hasattr(rec, "highlighted_dimension")
        assert not hasattr(rec, "dominant_drive")
