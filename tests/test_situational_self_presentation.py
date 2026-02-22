"""
tests/test_situational_self_presentation.py - 状況依存的自己呈示の認知テスト

全機能テスト、save/load往復テスト、安全弁テスト、エッジケーステスト。
"""

import json
import time

import pytest

from psyche.situational_self_presentation import (
    RecordFreshness,
    TypeCountLevel,
    ConvergenceLevel,
    PresentationRecord,
    CompositionDescription,
    ConvergenceRecord,
    SituationalSelfPresentationState,
    SituationalSelfPresentationConfig,
    SituationalSelfPresentationProcessor,
    determine_type_count_level,
    get_presentation_summary,
    create_situational_self_presentation_processor,
    _clamp,
    _gen_id,
    _freshness_from_value,
    _convergence_from_score,
)


# =============================================================================
# Helper functions
# =============================================================================

def _make_processor(
    max_records_per_user: int = 50,
    max_records_total: int = 200,
    **kwargs,
) -> SituationalSelfPresentationProcessor:
    """テスト用プロセッサを生成する。"""
    config = SituationalSelfPresentationConfig(
        max_records_per_user=max_records_per_user,
        max_records_total=max_records_total,
        **kwargs,
    )
    return SituationalSelfPresentationProcessor(config=config)


# =============================================================================
# Enum & Helper Tests
# =============================================================================

class TestEnums:
    def test_record_freshness_values(self):
        assert RecordFreshness.ACTIVE.value == "active"
        assert RecordFreshness.WEAKENING.value == "weakening"
        assert RecordFreshness.FADING.value == "fading"
        assert RecordFreshness.NEAR_INVISIBLE.value == "near_invisible"
        assert RecordFreshness.INVISIBLE.value == "invisible"

    def test_type_count_level_values(self):
        assert TypeCountLevel.LEVEL_0.value == "level_0"
        assert TypeCountLevel.LEVEL_1_5.value == "level_1_5"
        assert TypeCountLevel.LEVEL_6_10.value == "level_6_10"
        assert TypeCountLevel.LEVEL_11_15.value == "level_11_15"
        assert TypeCountLevel.LEVEL_16_PLUS.value == "level_16_plus"

    def test_convergence_level_values(self):
        assert ConvergenceLevel.NONE.value == "none"
        assert ConvergenceLevel.MILD.value == "mild"
        assert ConvergenceLevel.MODERATE.value == "moderate"
        assert ConvergenceLevel.STRONG.value == "strong"


class TestHelpers:
    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.5, 0.2, 0.8) == 0.5
        assert _clamp(0.1, 0.2, 0.8) == 0.2
        assert _clamp(0.9, 0.2, 0.8) == 0.8

    def test_gen_id(self):
        id1 = _gen_id()
        id2 = _gen_id()
        assert len(id1) == 12
        assert id1 != id2

    def test_freshness_from_value(self):
        assert _freshness_from_value(1.0) == RecordFreshness.ACTIVE
        assert _freshness_from_value(0.8) == RecordFreshness.ACTIVE
        assert _freshness_from_value(0.7) == RecordFreshness.WEAKENING
        assert _freshness_from_value(0.6) == RecordFreshness.WEAKENING
        assert _freshness_from_value(0.5) == RecordFreshness.FADING
        assert _freshness_from_value(0.4) == RecordFreshness.FADING
        assert _freshness_from_value(0.3) == RecordFreshness.NEAR_INVISIBLE
        assert _freshness_from_value(0.2) == RecordFreshness.NEAR_INVISIBLE
        assert _freshness_from_value(0.1) == RecordFreshness.INVISIBLE
        assert _freshness_from_value(0.0) == RecordFreshness.INVISIBLE

    def test_determine_type_count_level(self):
        assert determine_type_count_level(0) == TypeCountLevel.LEVEL_0
        assert determine_type_count_level(1) == TypeCountLevel.LEVEL_1_5
        assert determine_type_count_level(5) == TypeCountLevel.LEVEL_1_5
        assert determine_type_count_level(6) == TypeCountLevel.LEVEL_6_10
        assert determine_type_count_level(10) == TypeCountLevel.LEVEL_6_10
        assert determine_type_count_level(11) == TypeCountLevel.LEVEL_11_15
        assert determine_type_count_level(15) == TypeCountLevel.LEVEL_11_15
        assert determine_type_count_level(16) == TypeCountLevel.LEVEL_16_PLUS
        assert determine_type_count_level(100) == TypeCountLevel.LEVEL_16_PLUS

    def test_convergence_from_score(self):
        assert _convergence_from_score(0.0) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.2) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.3) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.4) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.5) == ConvergenceLevel.MODERATE
        assert _convergence_from_score(0.6) == ConvergenceLevel.MODERATE
        assert _convergence_from_score(0.7) == ConvergenceLevel.STRONG
        assert _convergence_from_score(1.0) == ConvergenceLevel.STRONG


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestPresentationRecord:
    def test_creation(self):
        rec = PresentationRecord(
            user_id="user_a",
            response_text="Hello",
            policy_label="empathic",
            tick=5,
        )
        assert rec.user_id == "user_a"
        assert rec.response_text == "Hello"
        assert rec.policy_label == "empathic"
        assert rec.tick == 5
        assert rec.freshness == 1.0
        assert rec.freshness_stage == RecordFreshness.ACTIVE.value
        assert len(rec.record_id) == 12

    def test_to_dict_from_dict_roundtrip(self):
        rec = PresentationRecord(
            user_id="user_b",
            response_text="Test text",
            policy_label="neutral",
            tick=10,
            freshness=0.7,
            freshness_stage=RecordFreshness.WEAKENING.value,
        )
        d = rec.to_dict()
        restored = PresentationRecord.from_dict(d)
        assert restored.user_id == rec.user_id
        assert restored.response_text == rec.response_text
        assert restored.policy_label == rec.policy_label
        assert restored.tick == rec.tick
        assert restored.freshness == rec.freshness
        assert restored.freshness_stage == rec.freshness_stage

    def test_from_dict_defaults(self):
        rec = PresentationRecord.from_dict({})
        assert rec.user_id == ""
        assert rec.response_text == ""
        assert rec.policy_label == ""
        assert rec.tick == 0
        assert rec.freshness == 1.0


class TestCompositionDescription:
    def test_creation(self):
        comp = CompositionDescription(
            user_id="user_a",
            policy_label_type_count_level=TypeCountLevel.LEVEL_1_5.value,
            record_count=3,
            latest_tick=10,
            tick=15,
        )
        assert comp.user_id == "user_a"
        assert comp.record_count == 3
        assert comp.latest_tick == 10

    def test_to_dict_from_dict_roundtrip(self):
        comp = CompositionDescription(
            user_id="user_c",
            policy_label_type_count_level=TypeCountLevel.LEVEL_6_10.value,
            record_count=8,
            latest_tick=20,
            tick=25,
        )
        d = comp.to_dict()
        restored = CompositionDescription.from_dict(d)
        assert restored.user_id == comp.user_id
        assert restored.policy_label_type_count_level == comp.policy_label_type_count_level
        assert restored.record_count == comp.record_count
        assert restored.latest_tick == comp.latest_tick


class TestConvergenceRecord:
    def test_creation(self):
        cr = ConvergenceRecord(
            user_id="user_a",
            convergence_score=0.5,
            convergence_level=ConvergenceLevel.MODERATE.value,
        )
        assert cr.user_id == "user_a"
        assert cr.convergence_score == 0.5

    def test_to_dict_from_dict_roundtrip(self):
        cr = ConvergenceRecord(
            user_id="user_b",
            convergence_score=0.8,
            convergence_level=ConvergenceLevel.STRONG.value,
            policy_label_type_count=10,
            cycle=5,
        )
        d = cr.to_dict()
        restored = ConvergenceRecord.from_dict(d)
        assert restored.user_id == cr.user_id
        assert restored.convergence_score == cr.convergence_score
        assert restored.convergence_level == cr.convergence_level


# =============================================================================
# State Tests
# =============================================================================

class TestState:
    def test_initial_state(self):
        state = SituationalSelfPresentationState()
        assert state.records == []
        assert state.user_index == {}
        assert state.composition_history == []
        assert state.latest_compositions == {}
        assert state.convergence_records == []
        assert state.cycle_count == 0
        assert state.total_records_added == 0
        assert state.total_records_pushed_out == 0
        assert state.total_records_invisible == 0
        assert state.convergence_warning is False

    def test_to_dict_from_dict_roundtrip(self):
        state = SituationalSelfPresentationState()
        state.records.append(PresentationRecord(
            user_id="user_a",
            response_text="test",
            policy_label="empathic",
            tick=1,
        ))
        state.user_index["user_a"] = [state.records[0].record_id]
        state.cycle_count = 5
        state.total_records_added = 10

        d = state.to_dict()
        restored = SituationalSelfPresentationState.from_dict(d)
        assert len(restored.records) == 1
        assert restored.records[0].user_id == "user_a"
        assert "user_a" in restored.user_index
        assert restored.cycle_count == 5
        assert restored.total_records_added == 10

    def test_apply_session_decay(self):
        state = SituationalSelfPresentationState()
        rec1 = PresentationRecord(
            user_id="user_a", response_text="a", freshness=0.5,
        )
        rec2 = PresentationRecord(
            user_id="user_b", response_text="b", freshness=0.2,
        )
        state.records = [rec1, rec2]
        state.user_index = {
            "user_a": [rec1.record_id],
            "user_b": [rec2.record_id],
        }

        state.apply_session_decay(decay_factor=0.3)

        # rec1: 0.5 - 0.3 = 0.2 (survives)
        # rec2: 0.2 - 0.3 = -0.1 -> 0.0 (removed, < 0.1)
        assert len(state.records) == 1
        assert state.records[0].user_id == "user_a"
        assert state.records[0].freshness == pytest.approx(0.2, abs=0.01)
        assert len(state.user_index["user_b"]) == 0

    def test_apply_session_decay_removes_all_if_very_stale(self):
        state = SituationalSelfPresentationState()
        rec = PresentationRecord(
            user_id="user_a", response_text="a", freshness=0.05,
        )
        state.records = [rec]
        state.user_index = {"user_a": [rec.record_id]}

        state.apply_session_decay(decay_factor=0.1)
        assert len(state.records) == 0


# =============================================================================
# Processor Tests - Stage 1: Receive and Accumulate
# =============================================================================

class TestReceiveAndAccumulate:
    def test_basic_receive(self):
        proc = _make_processor()
        proc.receive_and_accumulate(
            user_id="user_a",
            response_text="Hello world",
            policy_label="empathic",
            tick=1,
        )
        assert len(proc.state.records) == 1
        assert proc.state.records[0].user_id == "user_a"
        assert proc.state.records[0].policy_label == "empathic"
        assert proc.state.total_records_added == 1
        assert "user_a" in proc.state.user_index
        assert len(proc.state.user_index["user_a"]) == 1

    def test_empty_user_id_skips(self):
        proc = _make_processor()
        proc.receive_and_accumulate(
            user_id="",
            response_text="Hello",
            policy_label="empathic",
            tick=1,
        )
        assert len(proc.state.records) == 0
        assert proc.state.total_records_added == 0

    def test_empty_text_skips(self):
        proc = _make_processor()
        proc.receive_and_accumulate(
            user_id="user_a",
            response_text="",
            policy_label="empathic",
            tick=1,
        )
        assert len(proc.state.records) == 0

    def test_text_preview_truncation(self):
        proc = _make_processor(text_preview_length=10)
        proc.receive_and_accumulate(
            user_id="user_a",
            response_text="This is a very long text that should be truncated",
            policy_label="empathic",
            tick=1,
        )
        assert len(proc.state.records[0].response_text) == 10

    def test_multiple_users(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "Hello A", "empathic", 1)
        proc.receive_and_accumulate("user_b", "Hello B", "neutral", 2)
        proc.receive_and_accumulate("user_a", "Hello A again", "empathic", 3)

        assert len(proc.state.records) == 3
        assert len(proc.state.user_index["user_a"]) == 2
        assert len(proc.state.user_index["user_b"]) == 1

    def test_per_user_pushout(self):
        proc = _make_processor(max_records_per_user=3)
        for i in range(5):
            proc.receive_and_accumulate("user_a", f"text_{i}", "policy", i)

        assert len(proc.state.user_index["user_a"]) == 3
        # Oldest 2 records should have been pushed out
        assert proc.state.total_records_pushed_out == 2
        # Only 3 records remain
        user_records = [
            r for r in proc.state.records if r.user_id == "user_a"
        ]
        assert len(user_records) == 3

    def test_total_pushout(self):
        proc = _make_processor(max_records_total=5)
        for i in range(8):
            proc.receive_and_accumulate(f"user_{i}", f"text_{i}", "policy", i)

        assert len(proc.state.records) == 5
        assert proc.state.total_records_pushed_out == 3

    def test_all_records_equal_no_weight(self):
        """安全弁1: 全記録等価。重み・スコア・優先度を付与しない。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.receive_and_accumulate("user_a", "text_2", "neutral", 2)

        for rec in proc.state.records:
            # No weight, score, or priority attributes
            assert not hasattr(rec, "weight")
            assert not hasattr(rec, "score")
            assert not hasattr(rec, "priority")
            assert not hasattr(rec, "importance")

    def test_no_text_interpretation(self):
        """テキスト非解釈原則。テキストは生の文字列として保持。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "I feel angry!", "empathic", 1)

        rec = proc.state.records[0]
        # No semantic analysis attributes
        assert not hasattr(rec, "sentiment")
        assert not hasattr(rec, "category")
        assert not hasattr(rec, "meaning")


# =============================================================================
# Processor Tests - Stage 2: Composition Description Generation
# =============================================================================

class TestGenerateCompositions:
    def test_basic_composition(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.receive_and_accumulate("user_a", "text_2", "neutral", 2)
        proc.receive_and_accumulate("user_a", "text_3", "empathic", 3)

        proc.generate_compositions(current_tick=4)

        assert proc.state.cycle_count == 1
        assert "user_a" in proc.state.latest_compositions
        comp = proc.state.latest_compositions["user_a"]
        assert comp.record_count == 3
        # 2 unique labels: empathic, neutral
        assert comp.policy_label_type_count_level == TypeCountLevel.LEVEL_1_5.value
        assert comp.latest_tick == 3

    def test_multiple_user_compositions(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "text_2", "neutral", 2)

        proc.generate_compositions(current_tick=3)

        assert "user_a" in proc.state.latest_compositions
        assert "user_b" in proc.state.latest_compositions

    def test_composition_history_fifo(self):
        proc = _make_processor(max_composition_history=3)
        proc.receive_and_accumulate("user_a", "t1", "p1", 1)

        for i in range(5):
            proc.generate_compositions(current_tick=i + 2)

        assert len(proc.state.composition_history) <= 3

    def test_composition_non_cumulative(self):
        """安全弁4: 構成記述の非累積性。
        過去の構成記述が現在に影響しない。各サイクルで独立に再計算。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        comp1 = proc.state.latest_compositions["user_a"]
        count1 = comp1.record_count

        # 新しい記録を追加
        proc.receive_and_accumulate("user_a", "t2", "neutral", 3)
        proc.generate_compositions(current_tick=4)

        comp2 = proc.state.latest_compositions["user_a"]
        # 新しい記述は過去の記述に依存しない
        assert comp2.record_count == 2  # 独立に再計算

    def test_no_pattern_extraction(self):
        """安全弁2: パターン抽出禁止。"""
        proc = _make_processor()
        # 同じポリシーラベルを多数蓄積
        for i in range(10):
            proc.receive_and_accumulate("user_a", f"text_{i}", "empathic", i)

        proc.generate_compositions(current_tick=11)
        comp = proc.state.latest_compositions["user_a"]

        # 構成記述にはパターン・傾向の記述がない
        assert not hasattr(comp, "pattern")
        assert not hasattr(comp, "tendency")
        assert not hasattr(comp, "frequency")

    def test_no_frequency_counting(self):
        """個別のポリシーラベルの出現頻度を算出しない。"""
        proc = _make_processor()
        for i in range(5):
            proc.receive_and_accumulate("user_a", f"text_{i}", "empathic", i)
        proc.receive_and_accumulate("user_a", "text_5", "neutral", 5)

        proc.generate_compositions(current_tick=6)
        comp = proc.state.latest_compositions["user_a"]

        # 種類数のみ。出現頻度は含まない
        assert comp.policy_label_type_count_level in [
            t.value for t in TypeCountLevel
        ]
        assert not hasattr(comp, "label_frequencies")
        assert not hasattr(comp, "label_counts")

    def test_no_cross_user_comparison(self):
        """相手間の構成記述を比較・ランキング・差異抽出する処理は含まない。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "t2", "neutral", 2)

        proc.generate_compositions(current_tick=3)

        # 各相手の記述は独立
        comp_a = proc.state.latest_compositions["user_a"]
        comp_b = proc.state.latest_compositions["user_b"]
        assert comp_a.user_id == "user_a"
        assert comp_b.user_id == "user_b"
        # No comparison or ranking attributes
        assert not hasattr(comp_a, "rank")
        assert not hasattr(comp_a, "comparison")

    def test_empty_labels_result_in_level_0(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "", 1)

        proc.generate_compositions(current_tick=2)
        comp = proc.state.latest_compositions["user_a"]
        assert comp.policy_label_type_count_level == TypeCountLevel.LEVEL_0.value


# =============================================================================
# Processor Tests - Stage 3: Reference Handoff
# =============================================================================

class TestReferenceHandoff:
    def test_get_enrichment_data(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_a", "t2", "neutral", 2)
        proc.generate_compositions(current_tick=3)

        data = proc.get_enrichment_data(user_id="user_a")
        assert "cycle_count" in data
        assert "total_records" in data
        assert "current_record_count" in data
        assert "user_count" in data
        assert "summary_text" in data
        assert "current_user_composition" in data

    def test_enrichment_limits_content(self):
        """安全弁6: enrichmentに含めるのは蓄積記録数とポリシーラベル種類数の段階値のみ。
        出力テキストの内容を含めない。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "secret text content", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        data = proc.get_enrichment_data(user_id="user_a")

        # enrichmentデータに出力テキストの内容が含まれない
        data_str = json.dumps(data)
        assert "secret text content" not in data_str

        # 個別のポリシーラベル名が含まれない
        assert "empathic" not in data_str

    def test_enrichment_no_comparison(self):
        """相手間の比較をenrichmentに含めない。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "t2", "neutral", 2)
        proc.generate_compositions(current_tick=3)

        data = proc.get_enrichment_data(user_id="user_a")
        # user_b の情報は含まれない
        data_str = json.dumps(data)
        assert "user_b" not in data_str

    def test_enrichment_without_user_id(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        data = proc.get_enrichment_data()
        assert "current_user_composition" not in data
        assert "summary_text" in data

    def test_get_user_records(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "t2", "neutral", 2)
        proc.receive_and_accumulate("user_a", "t3", "empathic", 3)

        records_a = proc.get_user_records("user_a")
        assert len(records_a) == 2
        assert all(r.user_id == "user_a" for r in records_a)

        records_b = proc.get_user_records("user_b")
        assert len(records_b) == 1

    def test_get_reference_history(self):
        proc = _make_processor(reference_history_count=5)
        for i in range(10):
            proc.receive_and_accumulate("user_a", f"text_{i}", "p", i)

        history = proc.get_reference_history("user_a")
        assert len(history) <= 5

    def test_get_reference_history_all_users(self):
        proc = _make_processor(reference_history_count=5)
        for i in range(3):
            proc.receive_and_accumulate("user_a", f"text_a_{i}", "p", i)
            proc.receive_and_accumulate("user_b", f"text_b_{i}", "p", i + 10)

        history = proc.get_reference_history()
        assert len(history) <= 5
        # Contains records from both users
        user_ids = {r.user_id for r in history}
        assert "user_a" in user_ids or "user_b" in user_ids

    def test_get_composition_for_user(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        comp = proc.get_composition_for_user("user_a")
        assert comp is not None
        assert comp.user_id == "user_a"

        comp_none = proc.get_composition_for_user("nonexistent")
        assert comp_none is None

    def test_get_all_compositions(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "t2", "neutral", 2)
        proc.generate_compositions(current_tick=3)

        all_comps = proc.get_all_compositions()
        assert "user_a" in all_comps
        assert "user_b" in all_comps

    def test_get_summary(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        summary = proc.get_summary()
        assert "cycle_count" in summary
        assert "total_records_added" in summary
        assert "current_record_count" in summary
        assert "total_pushed_out" in summary
        assert "total_invisible" in summary
        assert "user_count" in summary
        assert "convergence_warning" in summary


# =============================================================================
# Freshness Decay Tests
# =============================================================================

class TestFreshnessDecay:
    def test_basic_decay(self):
        proc = _make_processor(freshness_decay_rate=0.1)
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        initial_freshness = proc.state.records[0].freshness

        proc.generate_compositions(current_tick=2)

        # Freshness should decrease
        assert proc.state.records[0].freshness < initial_freshness

    def test_absent_user_extra_decay(self):
        proc = _make_processor(
            freshness_decay_rate=0.05,
            absent_user_decay_rate=0.05,
        )
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "t2", "neutral", 2)

        # Only generate compositions for user_b (user_a is absent)
        proc.state.user_index["user_a"] = []  # simulate empty to make not in compositions
        proc.receive_and_accumulate("user_b", "t3", "neutral", 3)
        # Regenerate compositions -- user_a's records still exist but user_a has no index entries
        proc.generate_compositions(current_tick=4)

        # user_b records: basic decay only (0.05)
        # user_a records: no composition (absent), basic + absent decay (0.1)
        rec_a = [r for r in proc.state.records if r.user_id == "user_a"]
        rec_b = [r for r in proc.state.records if r.user_id == "user_b"]

        if rec_a and rec_b:
            # user_a gets extra decay
            assert rec_a[0].freshness <= rec_b[0].freshness

    def test_invisible_records_removed(self):
        proc = _make_processor(freshness_decay_rate=0.5)
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.state.records[0].freshness = 0.3  # Set low freshness

        proc.generate_compositions(current_tick=2)

        # Record should become invisible and be removed
        # 0.3 - 0.5 = -0.2 -> clamped to 0.0 -> INVISIBLE -> removed
        assert len(proc.state.records) == 0
        assert proc.state.total_records_invisible == 1


# =============================================================================
# Safety Valve Tests
# =============================================================================

class TestSafetyValves:
    def test_all_records_equal(self):
        """安全弁1: 全記録等価。重み・スコア・優先度を付与しない。"""
        proc = _make_processor()
        for i in range(5):
            proc.receive_and_accumulate("user_a", f"t_{i}", f"p_{i}", i)

        for rec in proc.state.records:
            assert rec.freshness == 1.0  # All start with same freshness
            assert not hasattr(rec, "weight")
            assert not hasattr(rec, "score")
            assert not hasattr(rec, "priority")
            assert not hasattr(rec, "importance")
            assert not hasattr(rec, "representative")

    def test_no_pattern_extraction(self):
        """安全弁2: パターン抽出禁止。"""
        proc = _make_processor()
        # Repeat same policy many times
        for i in range(20):
            proc.receive_and_accumulate("user_a", f"t_{i}", "empathic", i)

        proc.generate_compositions(current_tick=21)

        # No pattern/tendency data structures
        state = proc.state
        assert not hasattr(state, "patterns")
        assert not hasattr(state, "tendencies")
        assert not hasattr(state, "statistics")

    def test_fifo_natural_expiry(self):
        """安全弁3: FIFO自然消失。"""
        proc = _make_processor(max_records_per_user=3)
        for i in range(5):
            proc.receive_and_accumulate("user_a", f"t_{i}", "p", i)

        # Only 3 most recent remain
        assert len([r for r in proc.state.records if r.user_id == "user_a"]) == 3
        assert proc.state.total_records_pushed_out == 2

    def test_composition_non_cumulative(self):
        """安全弁4: 構成記述の非累積性。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)
        comp1 = proc.state.latest_compositions["user_a"]

        proc.generate_compositions(current_tick=3)
        comp2 = proc.state.latest_compositions["user_a"]

        # Each composition is independent (same data, but regenerated)
        assert comp2.tick == 3  # Different tick
        assert comp1.tick == 2

    def test_no_mapping_formation(self):
        """安全弁5: 相手別マッピング形成の禁止。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_a", "t2", "empathic", 2)

        # No mapping from user_id to output tendency
        state = proc.state
        assert not hasattr(state, "user_mappings")
        assert not hasattr(state, "user_tendencies")
        assert not hasattr(state, "user_profiles")
        assert not hasattr(state, "output_strategy")

    def test_enrichment_exposure_limits(self):
        """安全弁6: enrichment直接露出の制限。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "sensitive text", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        data = proc.get_enrichment_data(user_id="user_a")
        data_str = json.dumps(data)

        # No output text content in enrichment
        assert "sensitive text" not in data_str
        # No individual policy label names in enrichment
        assert "empathic" not in data_str

        # Only record count and type count level
        if "current_user_composition" in data:
            comp = data["current_user_composition"]
            assert "record_count" in comp
            assert "policy_label_type_count_level" in comp
            assert "response_text" not in comp

    def test_policy_selection_path_blocked(self):
        """安全弁7: ポリシー選択経路の遮断。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        # No attributes for bias calculation, stabilization valve input
        state = proc.state
        assert not hasattr(state, "bias")
        assert not hasattr(state, "policy_suggestion")
        assert not hasattr(state, "valve_input")

        # Enrichment data does not provide policy suggestions
        data = proc.get_enrichment_data(user_id="user_a")
        assert "policy_suggestion" not in data
        assert "bias" not in data

    def test_convergence_monitoring(self):
        """安全弁8: 収束監視。"""
        proc = _make_processor()
        # Many records with same policy label -> low type count
        for i in range(10):
            proc.receive_and_accumulate("user_a", f"text_{i}", "empathic", i)

        proc.generate_compositions(current_tick=11)

        # Should detect convergence (1 type with 10 records)
        assert len(proc.state.convergence_records) > 0

    def test_convergence_no_auto_intervention(self):
        """安全弁8: 検出結果は参照情報としてのみ記録し、自動的な介入・修正を行わない。"""
        proc = _make_processor()
        for i in range(10):
            proc.receive_and_accumulate("user_a", f"text_{i}", "empathic", i)

        proc.generate_compositions(current_tick=11)

        # Records are not modified by convergence detection
        for rec in proc.state.records:
            assert rec.freshness == pytest.approx(1.0 - 0.02, abs=0.01)

    def test_convergence_not_triggered_for_few_records(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.receive_and_accumulate("user_a", "t2", "empathic", 2)

        proc.generate_compositions(current_tick=3)
        # Few records -> no convergence detection
        assert len(proc.state.convergence_records) == 0


# =============================================================================
# Save/Load Roundtrip Tests
# =============================================================================

class TestSaveLoadRoundtrip:
    def test_state_roundtrip(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.receive_and_accumulate("user_b", "text_2", "neutral", 2)
        proc.receive_and_accumulate("user_a", "text_3", "analytical", 3)
        proc.generate_compositions(current_tick=4)

        # Save
        state_dict = proc.state.to_dict()

        # Load into new processor
        proc2 = _make_processor()
        proc2.state = SituationalSelfPresentationState.from_dict(state_dict)

        # Verify
        assert len(proc2.state.records) == len(proc.state.records)
        assert proc2.state.cycle_count == proc.state.cycle_count
        assert proc2.state.total_records_added == proc.state.total_records_added
        assert set(proc2.state.user_index.keys()) == set(proc.state.user_index.keys())
        assert len(proc2.state.composition_history) == len(proc.state.composition_history)

    def test_json_serialization_roundtrip(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        # Serialize to JSON
        state_json = json.dumps(proc.state.to_dict(), ensure_ascii=False)
        # Deserialize
        state_data = json.loads(state_json)
        restored_state = SituationalSelfPresentationState.from_dict(state_data)

        assert len(restored_state.records) == 1
        assert restored_state.records[0].user_id == "user_a"
        assert restored_state.cycle_count == 1

    def test_roundtrip_with_convergence_records(self):
        proc = _make_processor()
        for i in range(10):
            proc.receive_and_accumulate("user_a", f"text_{i}", "empathic", i)
        proc.generate_compositions(current_tick=11)

        state_dict = proc.state.to_dict()
        restored = SituationalSelfPresentationState.from_dict(state_dict)

        assert len(restored.convergence_records) == len(proc.state.convergence_records)

    def test_roundtrip_preserves_freshness(self):
        proc = _make_processor(freshness_decay_rate=0.1)
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        original_freshness = proc.state.records[0].freshness
        state_dict = proc.state.to_dict()
        restored = SituationalSelfPresentationState.from_dict(state_dict)

        assert restored.records[0].freshness == pytest.approx(original_freshness, abs=0.01)

    def test_session_boundary_decay_after_load(self):
        """セッション境界での鮮度減衰テスト。"""
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "text_1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        state_dict = proc.state.to_dict()
        restored = SituationalSelfPresentationState.from_dict(state_dict)
        restored.apply_session_decay(decay_factor=0.3)

        # Freshness should have decreased
        for rec in restored.records:
            assert rec.freshness < 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    def test_empty_state_enrichment(self):
        proc = _make_processor()
        data = proc.get_enrichment_data()
        assert data["cycle_count"] == 0
        assert data["total_records"] == 0
        assert "待機中" in data["summary_text"]

    def test_empty_state_summary(self):
        state = SituationalSelfPresentationState()
        summary = get_presentation_summary(state)
        assert "待機中" in summary

    def test_many_users(self):
        proc = _make_processor(max_records_total=100)
        for i in range(20):
            proc.receive_and_accumulate(f"user_{i}", f"text_{i}", "policy", i)

        proc.generate_compositions(current_tick=21)
        assert len(proc.state.latest_compositions) == 20

    def test_single_record_per_user(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        comp = proc.state.latest_compositions["user_a"]
        assert comp.record_count == 1

    def test_no_policy_labels(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "", 1)
        proc.receive_and_accumulate("user_a", "t2", "", 2)
        proc.generate_compositions(current_tick=3)

        comp = proc.state.latest_compositions["user_a"]
        assert comp.policy_label_type_count_level == TypeCountLevel.LEVEL_0.value

    def test_diverse_policy_labels(self):
        proc = _make_processor()
        labels = [f"policy_{i}" for i in range(20)]
        for i, label in enumerate(labels):
            proc.receive_and_accumulate("user_a", f"text_{i}", label, i)

        proc.generate_compositions(current_tick=21)
        comp = proc.state.latest_compositions["user_a"]
        assert comp.policy_label_type_count_level == TypeCountLevel.LEVEL_16_PLUS.value

    def test_repeated_generate_compositions(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)

        for i in range(10):
            proc.generate_compositions(current_tick=i + 2)

        assert proc.state.cycle_count == 10

    def test_user_index_cleanup_after_total_pushout(self):
        proc = _make_processor(max_records_total=3)
        proc.receive_and_accumulate("user_a", "t1", "p1", 1)
        proc.receive_and_accumulate("user_a", "t2", "p2", 2)
        proc.receive_and_accumulate("user_b", "t3", "p3", 3)
        proc.receive_and_accumulate("user_c", "t4", "p4", 4)

        # Total pushout should have removed the oldest record
        assert len(proc.state.records) == 3

    def test_state_property_setter(self):
        proc = _make_processor()
        new_state = SituationalSelfPresentationState()
        new_state.cycle_count = 42
        proc.state = new_state
        assert proc.state.cycle_count == 42


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory:
    def test_create_with_default_config(self):
        proc = create_situational_self_presentation_processor()
        assert proc.state.cycle_count == 0
        assert len(proc.state.records) == 0

    def test_create_with_custom_config(self):
        config = SituationalSelfPresentationConfig(
            max_records_per_user=10,
            max_records_total=50,
        )
        proc = create_situational_self_presentation_processor(config=config)
        assert proc._config.max_records_per_user == 10
        assert proc._config.max_records_total == 50


# =============================================================================
# Summary Tests
# =============================================================================

class TestSummary:
    def test_summary_waiting(self):
        state = SituationalSelfPresentationState()
        summary = get_presentation_summary(state)
        assert "待機中" in summary

    def test_summary_with_data(self):
        proc = _make_processor()
        proc.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc.generate_compositions(current_tick=2)

        summary = get_presentation_summary(proc.state, user_id="user_a")
        assert "cycle=1" in summary
        assert "蓄積=1" in summary
        assert "相手=1" in summary

    def test_summary_with_convergence_warning(self):
        state = SituationalSelfPresentationState()
        state.cycle_count = 5
        state.convergence_warning = True
        state.records.append(PresentationRecord(
            user_id="user_a", response_text="t", policy_label="p",
        ))
        state.user_index["user_a"] = [state.records[0].record_id]

        summary = get_presentation_summary(state)
        assert "種類数収束" in summary

    def test_summary_with_pushout(self):
        state = SituationalSelfPresentationState()
        state.cycle_count = 5
        state.total_records_pushed_out = 10
        state.records.append(PresentationRecord(
            user_id="user_a", response_text="t", policy_label="p",
        ))
        state.user_index["user_a"] = [state.records[0].record_id]

        summary = get_presentation_summary(state)
        assert "押出累計=10" in summary


# =============================================================================
# Integration-like Tests
# =============================================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """全ライフサイクルテスト: 蓄積→構成記述→参照→減衰→消失。"""
        proc = _make_processor(
            max_records_per_user=5,
            freshness_decay_rate=0.15,
        )

        # 蓄積
        for i in range(3):
            proc.receive_and_accumulate("user_a", f"text_{i}", f"policy_{i}", i)

        # 構成記述生成
        proc.generate_compositions(current_tick=3)
        assert "user_a" in proc.state.latest_compositions

        # 参照
        records = proc.get_user_records("user_a")
        assert len(records) == 3

        enrichment = proc.get_enrichment_data(user_id="user_a")
        assert enrichment["current_record_count"] == 3

        # 繰り返し生成で減衰
        for i in range(20):
            proc.generate_compositions(current_tick=4 + i)

        # 鮮度が低下している
        remaining = proc.get_user_records("user_a")
        if remaining:
            for rec in remaining:
                assert rec.freshness < 1.0

    def test_multi_user_lifecycle(self):
        """複数相手のライフサイクルテスト。"""
        proc = _make_processor()

        # User A
        for i in range(5):
            proc.receive_and_accumulate("user_a", f"text_a_{i}", "empathic", i)

        # User B
        for i in range(3):
            proc.receive_and_accumulate("user_b", f"text_b_{i}", "neutral", 10 + i)

        proc.generate_compositions(current_tick=15)

        comp_a = proc.get_composition_for_user("user_a")
        comp_b = proc.get_composition_for_user("user_b")

        assert comp_a is not None
        assert comp_b is not None
        assert comp_a.record_count == 5
        assert comp_b.record_count == 3

    def test_save_load_continue(self):
        """save/load後の継続操作テスト。"""
        proc1 = _make_processor()
        proc1.receive_and_accumulate("user_a", "t1", "empathic", 1)
        proc1.generate_compositions(current_tick=2)

        # Save
        state_dict = proc1.state.to_dict()

        # Load into new processor
        proc2 = _make_processor()
        proc2.state = SituationalSelfPresentationState.from_dict(state_dict)

        # Continue operation
        proc2.receive_and_accumulate("user_a", "t2", "neutral", 3)
        proc2.generate_compositions(current_tick=4)

        assert proc2.state.total_records_added == 2
        assert proc2.state.cycle_count == 2
        comp = proc2.get_composition_for_user("user_a")
        assert comp is not None
        assert comp.record_count == 2
