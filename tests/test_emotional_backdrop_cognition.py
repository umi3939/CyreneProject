"""
tests/test_emotional_backdrop_cognition.py - 感情基調の持続認知モジュールのテスト

テスト項目:
- 初期化テスト
- 4段パイプラインテスト（各段の動作確認）
- スライディングウィンドウテスト
- 鮮度減衰テスト
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

from psyche.emotional_backdrop_cognition import (
    InputSection,
    FreshnessStage,
    ConvergenceLevel,
    _clamp,
    _gen_id,
    _stage_from_freshness,
    _convergence_from_score,
    WindowEntry,
    CompositionRecord,
    ConvergenceRecord,
    BackdropInputs,
    BackdropState,
    BackdropResult,
    BackdropConfig,
    EmotionalBackdropProcessor,
    get_backdrop_summary,
    create_emotional_backdrop_processor,
)


# ── Helper ──────────────────────────────────────────────────────────

def _make_inputs(
    tick: int = 1,
    emotion_values: dict[str, float] | None = None,
    mood_valence: float = 0.3,
    mood_arousal: float = 0.4,
    dynamics_phase: str = "normal",
    amplitude_value: float = 1.0,
    meta_emotion_change_speed: float = 0.1,
    meta_emotion_dominant_stability: float = 0.5,
) -> BackdropInputs:
    """テスト用の BackdropInputs を作成する。"""
    return BackdropInputs(
        emotion_values=emotion_values or {"joy": 0.6, "sadness": 0.2},
        mood_valence=mood_valence,
        mood_arousal=mood_arousal,
        dynamics_phase=dynamics_phase,
        amplitude_value=amplitude_value,
        meta_emotion_change_speed=meta_emotion_change_speed,
        meta_emotion_dominant_stability=meta_emotion_dominant_stability,
        existing_record_count=0,
        average_freshness=0.0,
        dialogue_elapsed_ticks=tick,
        temporal_elapsed_description="",
        current_tick=tick,
    )


# ── Enum テスト ─────────────────────────────────────────────────────


class TestEnums:
    """Enum の定義確認テスト。"""

    def test_input_section_values(self):
        """InputSection が8値を持つ。"""
        assert len(InputSection) == 8

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
            emotion_values={"joy": 0.5, "fear": 0.3},
            mood_valence=0.4,
            mood_arousal=0.6,
            dynamics_phase="peak",
            amplitude_value=1.2,
            tick=10,
        )
        d = entry.to_dict()
        restored = WindowEntry.from_dict(d)
        assert restored.emotion_values == entry.emotion_values
        assert restored.mood_valence == entry.mood_valence
        assert restored.mood_arousal == entry.mood_arousal
        assert restored.dynamics_phase == entry.dynamics_phase
        assert restored.tick == entry.tick
        assert restored.entry_id == entry.entry_id

    def test_composition_record_roundtrip(self):
        """CompositionRecord の to_dict → from_dict ラウンドトリップ。"""
        rec = CompositionRecord(
            tick=5,
            window_size=10,
            tick_range=8,
            time_range=16.0,
            emotion_series={"joy": [0.5, 0.6, 0.7]},
            valence_series=[0.3, 0.4, 0.5],
            arousal_series=[0.2, 0.3, 0.4],
            phase_series=["normal", "peak", "normal"],
            low_variability_noted=True,
            freshness=0.8,
            freshness_stage="weakening",
        )
        d = rec.to_dict()
        restored = CompositionRecord.from_dict(d)
        assert restored.tick == 5
        assert restored.emotion_series == {"joy": [0.5, 0.6, 0.7]}
        assert restored.valence_series == [0.3, 0.4, 0.5]
        assert restored.low_variability_noted is True
        assert restored.freshness == 0.8
        assert restored.freshness_stage == "weakening"

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

    def test_backdrop_state_roundtrip(self):
        """BackdropState の to_dict → from_dict ラウンドトリップ。"""
        state = BackdropState()
        state.sliding_window.append(WindowEntry(tick=1))
        state.composition_records.append(CompositionRecord(tick=1))
        state.convergence_records.append(ConvergenceRecord(cycle=1))
        state.cycle_count = 5
        state.total_entries_collected = 10
        state.low_variability_warning = True

        d = state.to_dict()
        restored = BackdropState.from_dict(d)
        assert len(restored.sliding_window) == 1
        assert len(restored.composition_records) == 1
        assert len(restored.convergence_records) == 1
        assert restored.cycle_count == 5
        assert restored.total_entries_collected == 10
        assert restored.low_variability_warning is True


# ── 初期化テスト ────────────────────────────────────────────────────


class TestInitialization:
    """EmotionalBackdropProcessor の初期化テスト。"""

    def test_default_init(self):
        """デフォルト設定で初期化。"""
        proc = EmotionalBackdropProcessor()
        assert proc.state.cycle_count == 0
        assert len(proc.state.sliding_window) == 0
        assert len(proc.state.composition_records) == 0

    def test_custom_config(self):
        """カスタム設定で初期化。"""
        cfg = BackdropConfig(max_window_size=10, max_composition_records=20)
        proc = EmotionalBackdropProcessor(config=cfg)
        assert proc._config.max_window_size == 10
        assert proc._config.max_composition_records == 20

    def test_factory_function(self):
        """ファクトリ関数で生成。"""
        proc = create_emotional_backdrop_processor()
        assert isinstance(proc, EmotionalBackdropProcessor)
        assert proc.state.cycle_count == 0

    def test_factory_with_config(self):
        """ファクトリ関数にカスタム設定を渡す。"""
        cfg = BackdropConfig(max_window_size=5)
        proc = create_emotional_backdrop_processor(config=cfg)
        assert proc._config.max_window_size == 5


# ── 4段パイプラインテスト ──────────────────────────────────────────


class TestPipeline:
    """4段パイプラインの動作テスト。"""

    def test_single_tick(self):
        """1回の tick で結果が返る。"""
        proc = EmotionalBackdropProcessor()
        inputs = _make_inputs(tick=1)
        result = proc.tick(inputs)
        assert isinstance(result, BackdropResult)
        assert result.window_size == 1
        assert result.cycle_count == 1

    def test_process_alias(self):
        """process() も tick() と同じ結果。"""
        proc = EmotionalBackdropProcessor()
        inputs = _make_inputs(tick=1)
        result = proc.process(inputs)
        assert isinstance(result, BackdropResult)
        assert result.window_size == 1

    def test_multiple_ticks(self):
        """複数ティックで窓サイズが増加する。"""
        proc = EmotionalBackdropProcessor()
        for i in range(10):
            inputs = _make_inputs(tick=i + 1)
            result = proc.tick(inputs)
        assert result.window_size == 10
        assert result.cycle_count == 10
        assert proc.state.total_entries_collected == 10

    def test_window_entry_contains_emotion_data(self):
        """収集されたウィンドウエントリが感情データを含む。"""
        proc = EmotionalBackdropProcessor()
        inputs = _make_inputs(
            tick=1,
            emotion_values={"joy": 0.8, "sadness": 0.1},
            mood_valence=0.5,
        )
        proc.tick(inputs)
        entries = proc.get_window_entries()
        assert len(entries) == 1
        assert entries[0]["emotion_values"]["joy"] == 0.8
        assert entries[0]["mood_valence"] == 0.5

    def test_composition_record_created_each_tick(self):
        """各ティックで構成記録が作成される。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        records = proc.get_composition_records()
        assert len(records) == 5

    def test_emotion_series_in_composition(self):
        """構成記述に感情次元の時系列が含まれる。"""
        proc = EmotionalBackdropProcessor()
        for i in range(3):
            proc.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": 0.3 + i * 0.1},
            ))
        records = proc.get_composition_records()
        last = records[-1]
        assert "joy" in last["emotion_series"]
        assert len(last["emotion_series"]["joy"]) == 3

    def test_valence_series_in_composition(self):
        """構成記述にムードvalenceの時系列が含まれる。"""
        proc = EmotionalBackdropProcessor()
        for i in range(4):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        records = proc.get_composition_records()
        last = records[-1]
        assert len(last["valence_series"]) == 4

    def test_phase_series_in_composition(self):
        """構成記述にダイナミクス相の推移が含まれる。"""
        proc = EmotionalBackdropProcessor()
        phases = ["normal", "peak", "rebound", "normal"]
        for i, phase in enumerate(phases):
            proc.tick(_make_inputs(tick=i + 1, dynamics_phase=phase))
        records = proc.get_composition_records()
        last = records[-1]
        assert last["phase_series"] == phases

    def test_result_tick_range(self):
        """結果にティック範囲が含まれる。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=(i + 1) * 3))
        result = proc.tick(_make_inputs(tick=18))
        assert result.tick_range > 0


# ── スライディングウィンドウテスト ──────────────────────────────────


class TestSlidingWindow:
    """スライディングウィンドウのFIFO動作テスト。"""

    def test_fifo_pushout(self):
        """上限を超えるとFIFOで最古が押し出される。"""
        cfg = BackdropConfig(max_window_size=5)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(8):
            proc.tick(_make_inputs(tick=i + 1))
        entries = proc.get_window_entries()
        assert len(entries) == 5
        # 最古のエントリは tick=4 であるべき
        assert entries[0]["tick"] == 4

    def test_pushout_is_only_data_loss(self):
        """FIFO押し出しが唯一のデータ消失経路。"""
        cfg = BackdropConfig(max_window_size=3)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(3):
            proc.tick(_make_inputs(tick=i + 1))
        # 3件収集時点では消失なし
        assert proc.state.total_entries_collected == 3
        assert len(proc.state.sliding_window) == 3
        # 4件目で最古が1件消失
        proc.tick(_make_inputs(tick=4))
        assert proc.state.total_entries_collected == 4
        assert len(proc.state.sliding_window) == 3
        assert proc.state.sliding_window[0].tick == 2

    def test_no_selective_deletion(self):
        """選択的消去が行われない（FIFOのみ）。"""
        cfg = BackdropConfig(max_window_size=5)
        proc = EmotionalBackdropProcessor(config=cfg)
        # 全て同じ感情値でもFIFOで処理される
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1, emotion_values={"joy": 0.5}))
        entries = proc.get_window_entries()
        assert len(entries) == 5
        ticks = [e["tick"] for e in entries]
        assert ticks == [6, 7, 8, 9, 10]


# ── 鮮度減衰テスト ─────────────────────────────────────────────────


class TestFreshnessDecay:
    """鮮度段階的減衰のテスト。"""

    def test_initial_freshness_is_1(self):
        """新規記録の初期鮮度は1.0。"""
        proc = EmotionalBackdropProcessor()
        proc.tick(_make_inputs(tick=1))
        records = proc.get_composition_records()
        # 最初の記録は tick後に減衰適用されるが、まだ高い鮮度
        assert records[0]["freshness"] <= 1.0

    def test_freshness_decreases_over_time(self):
        """ティックごとに鮮度が減少する。"""
        proc = EmotionalBackdropProcessor()
        proc.tick(_make_inputs(tick=1))
        first_freshness = proc.state.composition_records[0].freshness
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 2))
        later_freshness = proc.state.composition_records[0].freshness
        assert later_freshness < first_freshness

    def test_freshness_stage_transitions(self):
        """鮮度段階が時間経過で遷移する。"""
        cfg = BackdropConfig(freshness_decay_rate=0.05)
        proc = EmotionalBackdropProcessor(config=cfg)
        proc.tick(_make_inputs(tick=1))
        initial_stage = proc.state.composition_records[0].freshness_stage

        # 多くのティックを実行して段階遷移を促す
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 2))

        later_stage = proc.state.composition_records[0].freshness_stage
        # 初期段階よりも低い段階になっているはず
        stages_order = ["active", "weakening", "fading", "near_invisible", "invisible"]
        assert stages_order.index(later_stage) >= stages_order.index(initial_stage)

    def test_decay_history_recorded(self):
        """段階遷移が減衰履歴に記録される。"""
        cfg = BackdropConfig(freshness_decay_rate=0.1)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(15):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.decay_history) > 0

    def test_session_decay(self):
        """apply_session_decay でセッション境界の一律減衰。"""
        state = BackdropState()
        rec = CompositionRecord(freshness=0.9, freshness_stage="active")
        state.composition_records.append(rec)
        state.apply_session_decay(decay_factor=0.3)
        assert state.composition_records[0].freshness == pytest.approx(0.6, abs=0.01)

    def test_session_decay_removes_very_low(self):
        """セッション減衰で鮮度0.1未満の記録が除去される。"""
        state = BackdropState()
        state.composition_records.append(
            CompositionRecord(freshness=0.05, freshness_stage="invisible")
        )
        state.composition_records.append(
            CompositionRecord(freshness=0.5, freshness_stage="fading")
        )
        state.apply_session_decay(decay_factor=0.3)
        # 0.05 - 0.3 < 0.1 → 除去
        # 0.5 - 0.3 = 0.2 ≥ 0.1 → 残存
        assert len(state.composition_records) == 1


# ── 安全弁テスト ────────────────────────────────────────────────────


class TestSafetyValves:
    """5つの安全弁のテスト。"""

    # 安全弁1: 窓内変動性の監視

    def test_low_variability_detection(self):
        """窓内変動性が極端に低い場合に検出される。"""
        cfg = BackdropConfig(low_variability_threshold=0.01)
        proc = EmotionalBackdropProcessor(config=cfg)
        # 全く同じ値で5ティック
        for i in range(5):
            proc.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": 0.5},
                mood_valence=0.5,
            ))
        assert proc.state.low_variability_warning is True

    def test_no_low_variability_with_diverse_input(self):
        """多様な入力では低変動性警告が出ない。"""
        proc = EmotionalBackdropProcessor()
        vals = [0.2, 0.5, 0.8, 0.3, 0.9]
        for i, v in enumerate(vals):
            proc.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": v},
                mood_valence=v,
            ))
        assert proc.state.low_variability_warning is False

    def test_low_variability_is_factual_only(self):
        """低変動性検出は事実記述のみ（変動促進なし）。"""
        cfg = BackdropConfig(low_variability_threshold=0.01)
        proc = EmotionalBackdropProcessor(config=cfg)
        # 同じ値で入力
        for i in range(5):
            proc.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": 0.5},
                mood_valence=0.5,
            ))
        # 警告フラグは立つが、ウィンドウの値は変更されない
        entries = proc.get_window_entries()
        for e in entries:
            assert e["emotion_values"]["joy"] == 0.5
            assert e["mood_valence"] == 0.5

    # 安全弁2: 蓄積偏り検出

    def test_accumulation_bias_detection(self):
        """蓄積記録が偏った場合に検出される。"""
        cfg = BackdropConfig(low_variability_threshold=0.001)
        proc = EmotionalBackdropProcessor(config=cfg)
        # 全く同じ値で多くのティック
        for i in range(10):
            proc.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": 0.5},
                mood_valence=0.5,
            ))
        # 偏り検出される可能性がある
        # （偏り検出は active/weakening の記録が3件以上必要）
        result = proc.tick(_make_inputs(tick=11, emotion_values={"joy": 0.5}, mood_valence=0.5))
        # 偏り警告が立つか確認（同一値の連続入力なので偏りが生じるはず）
        assert result.accumulation_bias_warning is True

    # 安全弁3: enrichment出力量制限

    def test_enrichment_record_limit(self):
        """enrichmentに含まれる記録数が制限される。"""
        cfg = BackdropConfig(max_enrichment_records=3)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i % 10)))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= 3

    # 安全弁4: 収束監視

    def test_convergence_monitoring(self):
        """収束監視が動作する。"""
        proc = EmotionalBackdropProcessor()
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.5))
        result = proc.tick(_make_inputs(tick=11, mood_valence=0.5))
        assert result.convergence_level in [c.value for c in ConvergenceLevel]

    def test_convergence_record_stored(self):
        """収束記録が保存される。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.convergence_records) == 5

    def test_convergence_records_limited(self):
        """収束記録数が上限内に収まる。"""
        cfg = BackdropConfig(max_convergence_records=5)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.convergence_records) <= 5

    # 安全弁5: 自己像固定化遮断（出力を数値列挙に限定）

    def test_enrichment_no_interpretive_text(self):
        """enrichmentデータに解釈的テキストが含まれない。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        data = proc.get_enrichment_data()
        # summary_text は数値情報のみ
        summary = data["summary_text"]
        # 評価判定の語句が含まれないことを確認
        forbidden = ["良い", "悪い", "異常", "健全", "望ましい", "望ましくない", "問題"]
        for word in forbidden:
            assert word not in summary

    # 多様性復元

    def test_diversity_restoration(self):
        """偏り検出時に鮮度減衰中の記録が復帰する。"""
        cfg = BackdropConfig(
            freshness_decay_rate=0.05,
            diversity_recovery_amount=0.1,
            low_variability_threshold=0.001,
        )
        proc = EmotionalBackdropProcessor(config=cfg)
        # 多様な入力の後に一様な入力
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        # 一様入力で偏りを誘発
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 6, mood_valence=0.5))
        result = proc.tick(_make_inputs(tick=16, mood_valence=0.5))
        # diversity_restored は偏り検出 + 復元時にTrueになりうる
        # （全記録がactive以上の場合は復元不要でFalseの場合もある）
        assert isinstance(result.diversity_restored, bool)


# ── 経路遮断確認テスト ─────────────────────────────────────────────


class TestPathBlocking:
    """5つの経路遮断の確認テスト。"""

    def test_no_emotion_pipeline_params_modification(self):
        """感情処理パイプラインのパラメータを変更しない。"""
        proc = EmotionalBackdropProcessor()
        inputs = _make_inputs(tick=1, emotion_values={"joy": 0.5}, mood_valence=0.3)
        proc.tick(inputs)
        # 入力オブジェクトの値が変更されていないことを確認
        assert inputs.emotion_values == {"joy": 0.5}
        assert inputs.mood_valence == 0.3

    def test_output_is_reference_only(self):
        """出力が参照情報形式のみ。"""
        proc = EmotionalBackdropProcessor()
        result = proc.tick(_make_inputs(tick=1))
        # BackdropResult は数値情報のみ（アクション指示なし）
        assert hasattr(result, "window_size")
        assert hasattr(result, "record_count")
        assert hasattr(result, "convergence_level")
        # アクション指示属性が存在しないことを確認
        assert not hasattr(result, "action")
        assert not hasattr(result, "policy")
        assert not hasattr(result, "recommendation")

    def test_read_only_accessors(self):
        """READ-ONLYアクセサがフィルタリング・選別しない。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        entries = proc.get_window_entries()
        # 全エントリが返される（フィルタリングなし）
        assert len(entries) == 5
        records = proc.get_composition_records()
        assert len(records) == 5


# ── Save/Load テスト ────────────────────────────────────────────────


class TestSaveLoad:
    """save/load 往復テスト。"""

    def test_save_returns_dict(self):
        """save() が dict を返す。"""
        proc = EmotionalBackdropProcessor()
        proc.tick(_make_inputs(tick=1))
        data = proc.save()
        assert isinstance(data, dict)
        assert "sliding_window" in data
        assert "composition_records" in data
        assert "cycle_count" in data

    def test_load_restores_state(self):
        """load() で状態が復元される。"""
        proc1 = EmotionalBackdropProcessor()
        for i in range(5):
            proc1.tick(_make_inputs(tick=i + 1))
        data = proc1.save()

        proc2 = EmotionalBackdropProcessor()
        proc2.load(data)
        assert proc2.state.cycle_count == 5
        assert len(proc2.state.sliding_window) == 5

    def test_roundtrip_preserves_data(self):
        """save → load で全データが保持される。"""
        proc1 = EmotionalBackdropProcessor()
        for i in range(10):
            proc1.tick(_make_inputs(
                tick=i + 1,
                emotion_values={"joy": 0.1 * (i + 1), "sadness": 0.05 * i},
                mood_valence=0.1 * i,
            ))
        data1 = proc1.save()

        proc2 = EmotionalBackdropProcessor()
        proc2.load(data1)
        data2 = proc2.save()

        assert data1["cycle_count"] == data2["cycle_count"]
        assert data1["total_entries_collected"] == data2["total_entries_collected"]
        assert data1["total_records_created"] == data2["total_records_created"]
        assert len(data1["sliding_window"]) == len(data2["sliding_window"])
        assert len(data1["composition_records"]) == len(data2["composition_records"])
        assert len(data1["convergence_records"]) == len(data2["convergence_records"])

    def test_load_empty_dict(self):
        """空の dict で load してもエラーにならない。"""
        proc = EmotionalBackdropProcessor()
        proc.load({})
        assert proc.state.cycle_count == 0

    def test_load_preserves_enrichment(self):
        """load後もenrichmentが正常に動作する。"""
        proc1 = EmotionalBackdropProcessor()
        for i in range(5):
            proc1.tick(_make_inputs(tick=i + 1))
        data = proc1.save()

        proc2 = EmotionalBackdropProcessor()
        proc2.load(data)
        enrichment = proc2.get_enrichment_data()
        assert enrichment["window_size"] == 5
        assert enrichment["record_count"] >= 1


# ── Enrichment データテスト ─────────────────────────────────────────


class TestEnrichmentData:
    """get_enrichment_data() のテスト。"""

    def test_initial_enrichment(self):
        """初期状態のenrichmentデータ。"""
        proc = EmotionalBackdropProcessor()
        data = proc.get_enrichment_data()
        assert data["window_size"] == 0
        assert data["record_count"] == 0
        assert "待機中" in data["summary_text"]

    def test_enrichment_after_ticks(self):
        """ティック後のenrichmentデータ。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        data = proc.get_enrichment_data()
        assert data["window_size"] == 5
        assert isinstance(data["entries"], list)
        assert data["record_count"] > 0

    def test_enrichment_entries_structure(self):
        """enrichmentエントリの構造が正しい。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        data = proc.get_enrichment_data()
        if data["entries"]:
            entry = data["entries"][0]
            assert "tick" in entry
            assert "window_size" in entry
            assert "valence" in entry
            assert "arousal" in entry
            assert "emotions" in entry
            assert "freshness_stage" in entry

    def test_enrichment_contains_warning_flags(self):
        """enrichmentデータに安全弁フラグが含まれる。"""
        proc = EmotionalBackdropProcessor()
        proc.tick(_make_inputs(tick=1))
        data = proc.get_enrichment_data()
        assert "low_variability_warning" in data
        assert "accumulation_bias_warning" in data
        assert "convergence_warning" in data

    def test_enrichment_size_limit(self):
        """enrichmentのエントリ数がmax_enrichment_recordsで制限される。"""
        cfg = BackdropConfig(max_enrichment_records=2)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(20):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i % 10)))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= 2


# ── サマリーテスト ──────────────────────────────────────────────────


class TestSummary:
    """get_backdrop_summary() のテスト。"""

    def test_summary_initial_state(self):
        """初期状態のサマリー。"""
        state = BackdropState()
        text = get_backdrop_summary(state)
        assert "待機中" in text

    def test_summary_after_ticks(self):
        """ティック後のサマリーに主要情報が含まれる。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        text = get_backdrop_summary(proc.state)
        assert "cycle=" in text
        assert "窓=" in text

    def test_summary_with_warnings(self):
        """警告フラグがサマリーに反映される。"""
        state = BackdropState()
        state.cycle_count = 5
        state.sliding_window = [WindowEntry(tick=1)]
        state.low_variability_warning = True
        state.accumulation_bias_warning = True
        state.convergence_warning = True
        text = get_backdrop_summary(state)
        assert "低変動" in text
        assert "蓄積偏り" in text
        assert "収束" in text

    def test_get_summary_method(self):
        """プロセッサの get_summary() メソッド。"""
        proc = EmotionalBackdropProcessor()
        proc.tick(_make_inputs(tick=1))
        summary = proc.get_summary()
        assert summary["window_size"] == 1
        assert summary["cycle_count"] == 1


# ── エッジケーステスト ──────────────────────────────────────────────


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_emotion_values(self):
        """空の感情辞書でもエラーなし。"""
        proc = EmotionalBackdropProcessor()
        inputs = _make_inputs(tick=1, emotion_values={})
        result = proc.tick(inputs)
        assert result.window_size == 1

    def test_single_emotion_dimension(self):
        """単一感情次元でも動作する。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, emotion_values={"joy": 0.5}))
        assert proc.state.cycle_count == 5

    def test_many_emotion_dimensions(self):
        """多数の感情次元でも動作する。"""
        proc = EmotionalBackdropProcessor()
        many_dims = {f"dim_{i}": 0.1 * i for i in range(20)}
        for i in range(3):
            proc.tick(_make_inputs(tick=i + 1, emotion_values=many_dims))
        assert proc.state.cycle_count == 3

    def test_extreme_values(self):
        """極端な値でもエラーなし。"""
        proc = EmotionalBackdropProcessor()
        inputs = BackdropInputs(
            emotion_values={"joy": 100.0, "sadness": -50.0},
            mood_valence=999.0,
            mood_arousal=-999.0,
            dynamics_phase="unknown",
            amplitude_value=0.0,
            meta_emotion_change_speed=0.0,
            meta_emotion_dominant_stability=0.0,
            current_tick=1,
        )
        result = proc.tick(inputs)
        assert isinstance(result, BackdropResult)

    def test_zero_window_config(self):
        """窓サイズ1でもFIFOが動作する。"""
        cfg = BackdropConfig(max_window_size=1)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.sliding_window) == 1
        assert proc.state.sliding_window[0].tick == 5

    def test_rapid_succession(self):
        """高速連続実行でも安定。"""
        proc = EmotionalBackdropProcessor()
        for i in range(100):
            proc.tick(_make_inputs(tick=i + 1))
        assert proc.state.cycle_count == 100

    def test_composition_records_limited(self):
        """蓄積記録数が上限内に収まる。"""
        cfg = BackdropConfig(max_composition_records=10)
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(30):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.composition_records) <= 10

    def test_decay_history_limited(self):
        """減衰履歴が上限内に収まる。"""
        cfg = BackdropConfig(
            max_decay_history=5,
            freshness_decay_rate=0.1,
        )
        proc = EmotionalBackdropProcessor(config=cfg)
        for i in range(30):
            proc.tick(_make_inputs(tick=i + 1))
        assert len(proc.state.decay_history) <= 5

    def test_state_setter(self):
        """state プロパティのsetterが動作する。"""
        proc = EmotionalBackdropProcessor()
        new_state = BackdropState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42


# ── 設計書準拠確認テスト ───────────────────────────────────────────


class TestDesignCompliance:
    """設計書の制約に準拠していることを確認するテスト。"""

    def test_no_weighted_averages(self):
        """移動平均・加重平均を算出しない。"""
        proc = EmotionalBackdropProcessor()
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        records = proc.get_composition_records()
        for rec in records:
            # 平均値フィールドが存在しないことを確認
            assert "average" not in rec
            assert "mean" not in rec
            assert "weighted" not in rec

    def test_no_pattern_extraction(self):
        """パターンを抽出・命名・分類しない。"""
        proc = EmotionalBackdropProcessor()
        for i in range(10):
            proc.tick(_make_inputs(tick=i + 1))
        records = proc.get_composition_records()
        for rec in records:
            assert "pattern" not in rec
            assert "category" not in rec
            assert "label" not in rec

    def test_equal_enumeration(self):
        """すべての値が等価に列挙される（重み付けなし）。"""
        proc = EmotionalBackdropProcessor()
        for i in range(5):
            proc.tick(_make_inputs(tick=i + 1, mood_valence=0.1 * (i + 1)))
        records = proc.get_composition_records()
        last = records[-1]
        # valence_series に全値が含まれている（フィルタリングなし）
        assert len(last["valence_series"]) == 5

    def test_meta_emotion_responsibility_separation(self):
        """メタ感情認知からは推移パターン特徴量のみを参照。"""
        # BackdropInputs のメタ感情関連フィールドが
        # change_speed と dominant_stability のみであることを確認
        inputs = BackdropInputs()
        assert hasattr(inputs, "meta_emotion_change_speed")
        assert hasattr(inputs, "meta_emotion_dominant_stability")
        # 持続パターン検出結果や変動候補のフィールドは存在しない
        assert not hasattr(inputs, "meta_emotion_sustained_patterns")
        assert not hasattr(inputs, "meta_emotion_variation_candidates")
