"""
tests/test_emotion_cooccurrence_description.py

感情間の共起記述モジュールのテスト。
"""

from __future__ import annotations

import time
import pytest

from psyche.emotion_cooccurrence_description import (
    FreshnessStage,
    DiversityLevel,
    CooccurrencePair,
    CooccurrenceRecord,
    CooccurrenceState,
    CooccurrenceResult,
    CooccurrenceConfig,
    EmotionCooccurrenceDescriptionProcessor,
    get_cooccurrence_summary,
    create_cooccurrence_processor,
    _clamp,
    _gen_id,
    _stage_from_freshness,
    _make_pair_key,
    _composition_signature,
)


# =============================================================================
# Helpers
# =============================================================================

def _base_emotions(**overrides) -> dict[str, float]:
    """テスト用感情ベクトルを生成する。"""
    emo = {
        "joy": 0.0,
        "anger": 0.0,
        "sorrow": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "love": 0.0,
        "fun": 0.0,
    }
    emo.update(overrides)
    return emo


# =============================================================================
# Enum tests
# =============================================================================

class TestEnums:
    def test_freshness_stage_values(self):
        assert FreshnessStage.ACTIVE.value == "active"
        assert FreshnessStage.WEAKENING.value == "weakening"
        assert FreshnessStage.FADING.value == "fading"
        assert FreshnessStage.NEAR_INVISIBLE.value == "near_invisible"
        assert FreshnessStage.INVISIBLE.value == "invisible"

    def test_diversity_level_values(self):
        assert DiversityLevel.HIGH.value == "high"
        assert DiversityLevel.MEDIUM.value == "medium"
        assert DiversityLevel.LOW.value == "low"


# =============================================================================
# Helper function tests
# =============================================================================

class TestHelpers:
    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_gen_id(self):
        id1 = _gen_id()
        id2 = _gen_id()
        assert isinstance(id1, str)
        assert len(id1) == 12
        assert id1 != id2

    def test_stage_from_freshness(self):
        assert _stage_from_freshness(1.0) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(0.8) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(0.7) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.6) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.5) == FreshnessStage.FADING
        assert _stage_from_freshness(0.4) == FreshnessStage.FADING
        assert _stage_from_freshness(0.3) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.2) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.1) == FreshnessStage.INVISIBLE
        assert _stage_from_freshness(0.0) == FreshnessStage.INVISIBLE

    def test_make_pair_key_sorted(self):
        assert _make_pair_key("anger", "joy") == ("anger", "joy")
        assert _make_pair_key("joy", "anger") == ("anger", "joy")
        assert _make_pair_key("fear", "fear") == ("fear", "fear")

    def test_composition_signature(self):
        pairs = [
            {"emotion_a": "joy", "emotion_b": "anger"},
            {"emotion_a": "fear", "emotion_b": "sorrow"},
        ]
        sig = _composition_signature(pairs)
        assert isinstance(sig, frozenset)
        assert len(sig) == 2
        assert ("anger", "joy") in sig
        assert ("fear", "sorrow") in sig

    def test_composition_signature_empty(self):
        sig = _composition_signature([])
        assert sig == frozenset()


# =============================================================================
# CooccurrencePair tests
# =============================================================================

class TestCooccurrencePair:
    def test_creation(self):
        pair = CooccurrencePair(
            emotion_a="joy", emotion_b="fear",
            value_a=0.5, value_b=0.3,
        )
        assert pair.emotion_a == "joy"
        assert pair.emotion_b == "fear"
        assert pair.value_a == 0.5
        assert pair.value_b == 0.3

    def test_to_dict(self):
        pair = CooccurrencePair(
            emotion_a="joy", emotion_b="anger",
            value_a=0.7, value_b=0.4,
        )
        d = pair.to_dict()
        assert d["emotion_a"] == "joy"
        assert d["emotion_b"] == "anger"
        assert d["value_a"] == 0.7
        assert d["value_b"] == 0.4

    def test_from_dict(self):
        d = {"emotion_a": "love", "emotion_b": "sorrow", "value_a": 0.8, "value_b": 0.2}
        pair = CooccurrencePair.from_dict(d)
        assert pair.emotion_a == "love"
        assert pair.emotion_b == "sorrow"

    def test_from_dict_defaults(self):
        pair = CooccurrencePair.from_dict({})
        assert pair.emotion_a == ""
        assert pair.value_a == 0.0

    def test_roundtrip(self):
        pair = CooccurrencePair(emotion_a="x", emotion_b="y", value_a=0.1, value_b=0.9)
        restored = CooccurrencePair.from_dict(pair.to_dict())
        assert restored.emotion_a == pair.emotion_a
        assert restored.value_b == pair.value_b


# =============================================================================
# CooccurrenceRecord tests
# =============================================================================

class TestCooccurrenceRecord:
    def test_auto_id(self):
        rec = CooccurrenceRecord()
        assert len(rec.record_id) == 12

    def test_explicit_id(self):
        rec = CooccurrenceRecord(record_id="test123")
        assert rec.record_id == "test123"

    def test_no_cooccurrence_flag(self):
        rec = CooccurrenceRecord(no_cooccurrence=True, pairs=[])
        assert rec.no_cooccurrence is True
        assert len(rec.pairs) == 0

    def test_composition_signature(self):
        rec = CooccurrenceRecord(
            pairs=[
                CooccurrencePair(emotion_a="anger", emotion_b="joy", value_a=0.5, value_b=0.3),
                CooccurrencePair(emotion_a="fear", emotion_b="sorrow", value_a=0.4, value_b=0.6),
            ]
        )
        sig = rec.composition_signature
        assert ("anger", "joy") in sig
        assert ("fear", "sorrow") in sig

    def test_to_dict_from_dict_roundtrip(self):
        rec = CooccurrenceRecord(
            tick=5,
            pairs=[CooccurrencePair(emotion_a="a", emotion_b="b", value_a=0.1, value_b=0.2)],
            no_cooccurrence=False,
            freshness=0.7,
            freshness_stage=FreshnessStage.WEAKENING.value,
        )
        d = rec.to_dict()
        restored = CooccurrenceRecord.from_dict(d)
        assert restored.record_id == rec.record_id
        assert restored.tick == 5
        assert len(restored.pairs) == 1
        assert restored.freshness == 0.7
        assert restored.freshness_stage == "weakening"


# =============================================================================
# CooccurrenceState tests
# =============================================================================

class TestCooccurrenceState:
    def test_empty_state(self):
        st = CooccurrenceState()
        assert len(st.records) == 0
        assert st.cycle_count == 0

    def test_to_dict_from_dict(self):
        st = CooccurrenceState(
            records=[
                CooccurrenceRecord(
                    tick=1,
                    pairs=[CooccurrencePair(emotion_a="a", emotion_b="b", value_a=0.5, value_b=0.5)],
                ),
            ],
            cycle_count=10,
            total_records_created=10,
            diversity_level=DiversityLevel.MEDIUM.value,
        )
        d = st.to_dict()
        restored = CooccurrenceState.from_dict(d)
        assert len(restored.records) == 1
        assert restored.cycle_count == 10
        assert restored.diversity_level == "medium"

    def test_apply_session_decay(self):
        st = CooccurrenceState(
            records=[
                CooccurrenceRecord(freshness=1.0),
                CooccurrenceRecord(freshness=0.5),
                CooccurrenceRecord(freshness=0.2),
            ]
        )
        st.apply_session_decay(decay_factor=0.3)
        # 1.0 -> 0.7, 0.5 -> 0.2, 0.2 -> -0.1 (removed)
        assert len(st.records) == 2
        assert st.records[0].freshness == pytest.approx(0.7)
        assert st.records[1].freshness == pytest.approx(0.2)

    def test_apply_session_decay_all_removed(self):
        st = CooccurrenceState(
            records=[
                CooccurrenceRecord(freshness=0.05),
                CooccurrenceRecord(freshness=0.08),
            ]
        )
        st.apply_session_decay(decay_factor=0.3)
        assert len(st.records) == 0


# =============================================================================
# CooccurrenceConfig tests
# =============================================================================

class TestCooccurrenceConfig:
    def test_defaults(self):
        cfg = CooccurrenceConfig()
        assert cfg.max_records == 50
        assert cfg.cooccurrence_threshold == 0.15
        assert cfg.freshness_decay_rate == 0.02
        assert cfg.max_enrichment_records == 5

    def test_custom_config(self):
        cfg = CooccurrenceConfig(max_records=100, cooccurrence_threshold=0.2)
        assert cfg.max_records == 100
        assert cfg.cooccurrence_threshold == 0.2


# =============================================================================
# Processor - Basic operation tests
# =============================================================================

class TestProcessorBasic:
    def test_create_processor(self):
        proc = create_cooccurrence_processor()
        assert proc.state.cycle_count == 0
        assert len(proc.state.records) == 0

    def test_single_tick_no_cooccurrence(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.5)
        result = proc.tick(emotions)
        assert result.no_cooccurrence is True
        assert result.pair_count == 0
        assert result.cycle_count == 1
        assert len(proc.state.records) == 1

    def test_single_tick_with_cooccurrence(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.5, anger=0.3)
        result = proc.tick(emotions)
        assert result.no_cooccurrence is False
        assert result.pair_count == 1
        assert result.cycle_count == 1

    def test_multiple_cooccurrence_pairs(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.5, anger=0.3, fear=0.4)
        result = proc.tick(emotions)
        assert result.pair_count == 3  # joy-anger, joy-fear, anger-fear

    def test_threshold_filtering(self):
        cfg = CooccurrenceConfig(cooccurrence_threshold=0.3)
        proc = create_cooccurrence_processor(config=cfg)
        emotions = _base_emotions(joy=0.5, anger=0.2, fear=0.4)
        result = proc.tick(emotions)
        assert result.pair_count == 1  # joy-fear only (anger below threshold)

    def test_all_below_threshold(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.1, anger=0.05)
        result = proc.tick(emotions)
        assert result.no_cooccurrence is True

    def test_empty_emotions(self):
        proc = create_cooccurrence_processor()
        result = proc.tick({})
        assert result.no_cooccurrence is True

    def test_pair_order_is_lexicographic(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.5, anger=0.3)
        proc.tick(emotions)
        rec = proc.state.records[-1]
        assert rec.pairs[0].emotion_a == "anger"  # anger < joy alphabetically
        assert rec.pairs[0].emotion_b == "joy"


# =============================================================================
# Processor - FIFO tests
# =============================================================================

class TestProcessorFIFO:
    def test_fifo_capacity(self):
        cfg = CooccurrenceConfig(max_records=5)
        proc = create_cooccurrence_processor(config=cfg)
        for i in range(10):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        assert len(proc.state.records) == 5

    def test_fifo_oldest_removed(self):
        cfg = CooccurrenceConfig(max_records=3)
        proc = create_cooccurrence_processor(config=cfg)
        for i in range(5):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        assert proc.state.records[0].tick == 3  # ticks 1, 2 removed
        assert proc.state.records[-1].tick == 5

    def test_total_created_counter(self):
        cfg = CooccurrenceConfig(max_records=3)
        proc = create_cooccurrence_processor(config=cfg)
        for i in range(5):
            proc.tick(_base_emotions(joy=0.5))
        assert proc.state.total_records_created == 5


# =============================================================================
# Processor - Freshness decay tests
# =============================================================================

class TestProcessorFreshnessDecay:
    def test_freshness_decreases_each_tick(self):
        cfg = CooccurrenceConfig(freshness_decay_rate=0.1)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        first_freshness = proc.state.records[0].freshness
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        assert proc.state.records[0].freshness < first_freshness

    def test_freshness_stage_transitions(self):
        cfg = CooccurrenceConfig(freshness_decay_rate=0.15)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Initial: active (1.0)
        assert proc.state.records[0].freshness_stage == "active"

        # After several ticks, freshness decays
        # Each tick applies decay to all records. After initial tick: 1.0 - 0.15 = 0.85 (active).
        # After 1 more tick: 0.85 - 0.15 = 0.70 (weakening).
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Record[0] freshness = 1.0 - 0.15*2 = 0.70 -> weakening
        assert _stage_from_freshness(proc.state.records[0].freshness) in (
            FreshnessStage.WEAKENING, FreshnessStage.FADING,
        )

    def test_invisible_records_counted(self):
        cfg = CooccurrenceConfig(freshness_decay_rate=0.25, max_records=100)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Decay until invisible
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.1))  # no cooccurrence ticks
        assert proc.state.total_records_decayed > 0


# =============================================================================
# Processor - Safety valve tests
# =============================================================================

class TestProcessorSafetyValves:
    def test_diversity_high_with_varied_inputs(self):
        proc = create_cooccurrence_processor()
        emotions_list = [
            _base_emotions(joy=0.5, anger=0.3),
            _base_emotions(fear=0.5, sorrow=0.3),
            _base_emotions(love=0.5, surprise=0.3),
            _base_emotions(joy=0.5, fear=0.3),
            _base_emotions(anger=0.5, sorrow=0.3),
        ]
        for emo in emotions_list:
            proc.tick(emo)
        assert proc.state.diversity_level == DiversityLevel.HIGH.value

    def test_diversity_low_with_same_input(self):
        cfg = CooccurrenceConfig(low_diversity_threshold=2)
        proc = create_cooccurrence_processor(config=cfg)
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        assert proc.state.diversity_level == DiversityLevel.LOW.value

    def test_accumulation_bias_detected(self):
        cfg = CooccurrenceConfig(bias_consecutive_threshold=3)
        proc = create_cooccurrence_processor(config=cfg)
        for _ in range(5):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        assert proc.state.accumulation_bias_warning is True

    def test_no_bias_with_varied_input(self):
        cfg = CooccurrenceConfig(bias_consecutive_threshold=3)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        proc.tick(_base_emotions(fear=0.5, sorrow=0.3))
        proc.tick(_base_emotions(love=0.5, surprise=0.3))
        assert proc.state.accumulation_bias_warning is False

    def test_diversity_restoration(self):
        cfg = CooccurrenceConfig(
            low_diversity_threshold=1,
            freshness_decay_rate=0.15,
            diversity_recovery_amount=0.2,
        )
        proc = create_cooccurrence_processor(config=cfg)
        # First, create diverse records
        proc.tick(_base_emotions(joy=0.5, fear=0.3))
        proc.tick(_base_emotions(anger=0.5, sorrow=0.3))
        # Let them decay
        for _ in range(5):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Check that recovery happened
        assert proc.state.total_records_recovered > 0

    def test_convergence_warning(self):
        cfg = CooccurrenceConfig(low_diversity_threshold=2)
        proc = create_cooccurrence_processor(config=cfg)
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        result = proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # With only one unique pair composition, diversity should be LOW
        assert result.convergence_warning is True

    def test_no_convergence_warning_with_diversity(self):
        proc = create_cooccurrence_processor()
        different_emotions = [
            _base_emotions(joy=0.5, anger=0.3),
            _base_emotions(fear=0.5, sorrow=0.3),
            _base_emotions(love=0.5, surprise=0.3),
            _base_emotions(joy=0.3, fear=0.5, anger=0.4),
            _base_emotions(sorrow=0.5, love=0.3, fun=0.4),
        ]
        for emo in different_emotions:
            proc.tick(emo)
        result = proc.tick(_base_emotions(joy=0.5, fear=0.3))
        assert result.convergence_warning is False


# =============================================================================
# Processor - Enrichment tests
# =============================================================================

class TestProcessorEnrichment:
    def test_enrichment_waiting(self):
        proc = create_cooccurrence_processor()
        data = proc.get_enrichment_data()
        assert "待機中" in data["summary_text"]
        assert data["record_count"] == 0
        assert data["entries"] == []

    def test_enrichment_with_data(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.get_enrichment_data()
        assert data["record_count"] == 1
        assert data["visible_count"] == 1
        assert len(data["entries"]) == 1

    def test_enrichment_entry_structure_with_pairs(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.get_enrichment_data()
        entry = data["entries"][0]
        assert "tick" in entry
        assert "pairs" in entry
        assert "freshness_stage" in entry
        pair = entry["pairs"][0]
        assert "a" in pair
        assert "b" in pair
        assert "va" in pair
        assert "vb" in pair

    def test_enrichment_entry_no_cooccurrence(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5))
        data = proc.get_enrichment_data()
        entry = data["entries"][0]
        assert entry["no_cooccurrence"] is True

    def test_enrichment_limit(self):
        cfg = CooccurrenceConfig(max_enrichment_records=3)
        proc = create_cooccurrence_processor(config=cfg)
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= 3

    def test_enrichment_excludes_invisible(self):
        cfg = CooccurrenceConfig(freshness_decay_rate=0.3, max_records=100)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Decay heavily
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.1))
        data = proc.get_enrichment_data()
        # First record should be invisible or near-invisible
        for entry in data["entries"]:
            assert entry.get("freshness_stage") != "invisible"

    def test_enrichment_no_frequency_info(self):
        """安全弁5: 頻度情報を含まない。"""
        proc = create_cooccurrence_processor()
        for _ in range(5):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.get_enrichment_data()
        # No "count", "frequency", "occurrence_count" keys
        for entry in data["entries"]:
            assert "count" not in entry
            assert "frequency" not in entry
            assert "occurrence_count" not in entry

    def test_enrichment_no_interpretive_text(self):
        """安全弁4: 解釈的テキストの不付与。"""
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.get_enrichment_data()
        summary = data["summary_text"]
        # No interpretive phrases
        forbidden = ["よく一緒に", "珍しい", "特徴的", "異常", "健全", "望ましい"]
        for phrase in forbidden:
            assert phrase not in summary


# =============================================================================
# Processor - Save/Load tests
# =============================================================================

class TestProcessorSaveLoad:
    def test_save_empty(self):
        proc = create_cooccurrence_processor()
        data = proc.save()
        assert data["cycle_count"] == 0
        assert data["records"] == []

    def test_save_with_data(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        data = proc.save()
        assert data["cycle_count"] == 1
        assert len(data["records"]) == 1

    def test_load(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        proc.tick(_base_emotions(fear=0.5, sorrow=0.3))
        saved = proc.save()

        proc2 = create_cooccurrence_processor()
        proc2.load(saved)
        assert proc2.state.cycle_count == 2
        assert len(proc2.state.records) == 2

    def test_roundtrip(self):
        proc = create_cooccurrence_processor()
        for i in range(5):
            proc.tick(_base_emotions(joy=0.5, anger=0.3 + i * 0.05))
        saved = proc.save()
        proc2 = create_cooccurrence_processor()
        proc2.load(saved)
        saved2 = proc2.save()
        assert saved == saved2

    def test_session_decay_on_load(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        proc.tick(_base_emotions(fear=0.5, sorrow=0.3))
        saved = proc.save()
        proc2 = create_cooccurrence_processor()
        proc2.load(saved)
        proc2.state.apply_session_decay()
        # Records should have reduced freshness
        for rec in proc2.state.records:
            assert rec.freshness < 1.0


# =============================================================================
# Processor - READ-ONLY accessor tests
# =============================================================================

class TestProcessorAccessors:
    def test_get_records(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        records = proc.get_records()
        assert len(records) == 1
        assert isinstance(records[0], dict)

    def test_get_visible_records(self):
        cfg = CooccurrenceConfig(freshness_decay_rate=0.3, max_records=100)
        proc = create_cooccurrence_processor(config=cfg)
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.1))
        visible = proc.get_visible_records()
        all_records = proc.get_records()
        assert len(visible) <= len(all_records)

    def test_get_summary(self):
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        summary = proc.get_summary()
        assert summary["record_count"] == 1
        assert summary["cycle_count"] == 1
        assert "diversity_level" in summary
        assert "accumulation_bias_warning" in summary
        assert "convergence_warning" in summary


# =============================================================================
# Summary function tests
# =============================================================================

class TestSummary:
    def test_summary_waiting(self):
        st = CooccurrenceState()
        text = get_cooccurrence_summary(st)
        assert "待機中" in text

    def test_summary_with_data(self):
        st = CooccurrenceState(
            cycle_count=5,
            records=[
                CooccurrenceRecord(
                    pairs=[CooccurrencePair(emotion_a="a", emotion_b="b", value_a=0.5, value_b=0.5)],
                    freshness=0.9,
                ),
            ],
            diversity_level=DiversityLevel.HIGH.value,
        )
        text = get_cooccurrence_summary(st)
        assert "cycle=5" in text
        assert "多様性=high" in text

    def test_summary_no_cooccurrence(self):
        st = CooccurrenceState(
            cycle_count=1,
            records=[
                CooccurrenceRecord(no_cooccurrence=True, freshness=0.9),
            ],
        )
        text = get_cooccurrence_summary(st)
        assert "共起なし" in text

    def test_summary_warnings(self):
        st = CooccurrenceState(
            cycle_count=10,
            records=[CooccurrenceRecord(freshness=0.9)],
            accumulation_bias_warning=True,
            convergence_warning=True,
        )
        text = get_cooccurrence_summary(st)
        assert "蓄積偏り" in text
        assert "収束" in text


# =============================================================================
# Factory function test
# =============================================================================

class TestFactory:
    def test_create_default(self):
        proc = create_cooccurrence_processor()
        assert isinstance(proc, EmotionCooccurrenceDescriptionProcessor)
        assert proc.state.cycle_count == 0

    def test_create_with_config(self):
        cfg = CooccurrenceConfig(max_records=10)
        proc = create_cooccurrence_processor(config=cfg)
        assert proc._config.max_records == 10


# =============================================================================
# Safety: No write-back to emotion pipeline
# =============================================================================

class TestNoWriteBack:
    def test_emotion_values_not_modified(self):
        """感情処理パイプラインのパラメータを変更しないことを確認。"""
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(joy=0.5, anger=0.3)
        original = dict(emotions)
        proc.tick(emotions)
        # Emotions dict should be unchanged
        assert emotions == original

    def test_no_policy_output(self):
        """ポリシー候補拡張・判断バイアスへの直接出力がないことを確認。"""
        proc = create_cooccurrence_processor()
        proc.tick(_base_emotions(joy=0.5, anger=0.3))
        result = proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # Result contains only information, no policy/bias data
        assert not hasattr(result, "policy_candidates")
        assert not hasattr(result, "bias")
        assert not hasattr(result, "decision")

    def test_no_frequency_counting(self):
        """安全弁5: 頻度情報の遮断。共起ペアの出現回数を集計しない。"""
        proc = create_cooccurrence_processor()
        for _ in range(10):
            proc.tick(_base_emotions(joy=0.5, anger=0.3))
        # No frequency/count attributes in state or records
        for rec in proc.state.records:
            assert not hasattr(rec, "frequency")
            assert not hasattr(rec, "occurrence_count")


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    def test_single_emotion_above_threshold(self):
        proc = create_cooccurrence_processor()
        result = proc.tick(_base_emotions(joy=0.5))
        assert result.no_cooccurrence is True
        assert result.pair_count == 0

    def test_all_emotions_above_threshold(self):
        proc = create_cooccurrence_processor()
        emotions = _base_emotions(
            joy=0.5, anger=0.3, sorrow=0.4,
            fear=0.6, surprise=0.2, love=0.7, fun=0.3,
        )
        result = proc.tick(emotions)
        # With 7 emotions above threshold, C(7,2) = 21 pairs
        assert result.pair_count == 21

    def test_exactly_at_threshold(self):
        cfg = CooccurrenceConfig(cooccurrence_threshold=0.3)
        proc = create_cooccurrence_processor(config=cfg)
        emotions = _base_emotions(joy=0.3, anger=0.3)
        result = proc.tick(emotions)
        assert result.pair_count == 1  # exactly at threshold is included

    def test_just_below_threshold(self):
        cfg = CooccurrenceConfig(cooccurrence_threshold=0.3)
        proc = create_cooccurrence_processor(config=cfg)
        emotions = _base_emotions(joy=0.29, anger=0.3)
        result = proc.tick(emotions)
        assert result.no_cooccurrence is True  # joy below threshold

    def test_many_ticks_stability(self):
        """長期実行でエラーが起きないことを確認。"""
        proc = create_cooccurrence_processor()
        for i in range(100):
            val = 0.1 + (i % 10) * 0.05
            emotions = _base_emotions(joy=val, anger=0.3, fear=val * 0.8)
            result = proc.tick(emotions)
            assert result.cycle_count == i + 1
        assert len(proc.state.records) <= 50  # max_records default

    def test_state_setter(self):
        proc = create_cooccurrence_processor()
        new_state = CooccurrenceState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42
