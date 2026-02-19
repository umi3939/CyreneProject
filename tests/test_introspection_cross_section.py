"""
tests/test_introspection_cross_section.py - 内省断面間の横断的記述のテスト

カバー範囲:
- 初期状態
- 各断面の要約テスト（Stage 1）: 6断面 × 複数入力型
- スナップショット構成テスト（Stage 2）
- スライディングウィンドウのFIFO動作テスト
- enrichment形式テスト（Stage 3）
- READ-ONLYアクセサテスト
- save/load round-trip
- 安全弁テスト（5つ）:
  1. パターン抽出禁止
  2. 全断面等価性
  3. 統合禁止
  4. ウィンドウサイズ制限
  5. 書き込み経路遮断
- 経路遮断テスト（5つ）
- 不在断面テスト
- エッジケーステスト
- ファクトリテスト
- 統合テスト
"""

import time
import pytest

from psyche.introspection_cross_section import (
    CrossSectionSnapshot,
    IntrospectionCrossSectionState,
    IntrospectionCrossSectionConfig,
    IntrospectionCrossSectionProcessor,
    create_introspection_cross_section,
    SECTION_ORDER,
    SECTION_SELF_MODEL,
    SECTION_TEMPORAL_SELF_DIFFERENCE,
    SECTION_IDENTITY_COHERENCE,
    SECTION_SELF_NARRATIVE,
    SECTION_INTROSPECTION_CONSUMPTION,
    SECTION_META_EMOTION_COGNITION,
    SECTION_LABELS,
    ABSENT_MARKER,
    EXTENSION_CANDIDATES,
    _EVALUATIVE_WORDS,
    _truncate,
    _sanitize_summary,
    _extract_self_model_summary,
    _extract_temporal_self_diff_summary,
    _extract_identity_coherence_summary,
    _extract_self_narrative_summary,
    _extract_introspection_consumption_summary,
    _extract_meta_emotion_summary,
    _SECTION_EXTRACTORS,
)


# =============================================================================
# Helpers
# =============================================================================

def make_processor(
    max_snapshots: int = 10,
    max_summary_length: int = 200,
    max_enrichment_length: int = 2000,
    max_enrichment_snapshots: int = 3,
) -> IntrospectionCrossSectionProcessor:
    """テスト用プロセッサを生成する。"""
    config = IntrospectionCrossSectionConfig(
        max_snapshots=max_snapshots,
        max_summary_length=max_summary_length,
        max_enrichment_length=max_enrichment_length,
        max_enrichment_snapshots=max_enrichment_snapshots,
    )
    return IntrospectionCrossSectionProcessor(config=config)


def make_full_outputs(prefix: str = "data") -> dict:
    """全6断面が揃ったモジュール出力を生成する。"""
    return {
        SECTION_SELF_MODEL: f"{prefix}_self_model_output",
        SECTION_TEMPORAL_SELF_DIFFERENCE: f"{prefix}_temporal_diff_output",
        SECTION_IDENTITY_COHERENCE: f"{prefix}_identity_coherence_output",
        SECTION_SELF_NARRATIVE: f"{prefix}_self_narrative_output",
        SECTION_INTROSPECTION_CONSUMPTION: f"{prefix}_introspection_consumption_output",
        SECTION_META_EMOTION_COGNITION: f"{prefix}_meta_emotion_output",
    }


def process_n(
    processor: IntrospectionCrossSectionProcessor,
    n: int,
    base_tick: int = 1,
    base_timestamp: float = 1000.0,
) -> list[CrossSectionSnapshot]:
    """n件のスナップショットを処理する。"""
    results = []
    for i in range(n):
        outputs = make_full_outputs(prefix=f"tick{base_tick + i}")
        snap = processor.process(
            module_outputs=outputs,
            tick=base_tick + i,
            timestamp=base_timestamp + i * 10.0,
        )
        results.append(snap)
    return results


# =============================================================================
# 初期状態テスト
# =============================================================================

class TestInitialState:
    """初期状態のテスト。"""

    def test_initial_window_empty(self):
        proc = make_processor()
        assert proc.state.snapshot_window == []

    def test_initial_previous_snapshot_none(self):
        proc = make_processor()
        assert proc.state.previous_snapshot is None

    def test_initial_get_snapshot_window_empty(self):
        proc = make_processor()
        assert proc.get_snapshot_window() == []

    def test_initial_get_previous_snapshot_none(self):
        proc = make_processor()
        assert proc.get_previous_snapshot() is None

    def test_initial_get_latest_snapshot_none(self):
        proc = make_processor()
        assert proc.get_latest_snapshot() is None

    def test_initial_enrichment_waiting(self):
        proc = make_processor()
        text = proc.get_enrichment_text()
        assert "待機中" in text

    def test_initial_summary(self):
        proc = make_processor()
        summary = proc.get_summary()
        assert summary["window_size"] == 0
        assert summary["has_previous_snapshot"] is False


# =============================================================================
# 各断面の要約テスト (Stage 1)
# =============================================================================

class TestSectionSummaryExtraction:
    """各断面の要約抽出テスト。"""

    def test_string_input(self):
        result = _extract_self_model_summary("hello world", 200)
        assert result == "hello world"

    def test_dict_with_summary_key(self):
        result = _extract_self_model_summary({"summary": "summary text"}, 200)
        assert result == "summary text"

    def test_dict_with_description_key(self):
        result = _extract_self_model_summary({"description": "desc text"}, 200)
        assert result == "desc text"

    def test_dict_with_text_key(self):
        result = _extract_self_model_summary({"text": "text val"}, 200)
        assert result == "text val"

    def test_dict_fallback_to_listing(self):
        result = _extract_self_model_summary({"a": 1, "b": 2}, 200)
        assert "a: 1" in result
        assert "b: 2" in result

    def test_none_input_returns_absent(self):
        result = _extract_self_model_summary(None, 200)
        assert result == ABSENT_MARKER

    def test_empty_string_returns_absent(self):
        result = _extract_self_model_summary("", 200)
        assert result == ABSENT_MARKER

    def test_whitespace_string_returns_absent(self):
        result = _extract_self_model_summary("   ", 200)
        assert result == ABSENT_MARKER

    def test_integer_input_as_string(self):
        result = _extract_self_model_summary(42, 200)
        assert result == "42"

    def test_truncation(self):
        long_text = "a" * 500
        result = _extract_self_model_summary(long_text, 100)
        assert len(result) <= 100

    def test_evaluative_word_sanitized(self):
        result = _extract_self_model_summary("状態は良好です", 200)
        assert "良好" not in result
        assert "[...]" in result

    def test_temporal_diff_string(self):
        result = _extract_temporal_self_diff_summary("diff data", 200)
        assert result == "diff data"

    def test_temporal_diff_dict_with_diff_summary(self):
        result = _extract_temporal_self_diff_summary({"diff_summary": "diff s"}, 200)
        assert result == "diff s"

    def test_temporal_diff_none(self):
        result = _extract_temporal_self_diff_summary(None, 200)
        assert result == ABSENT_MARKER

    def test_identity_coherence_string(self):
        result = _extract_identity_coherence_summary("coherence data", 200)
        assert result == "coherence data"

    def test_identity_coherence_dict_with_stage(self):
        result = _extract_identity_coherence_summary({"stage": "high"}, 200)
        assert result == "high"

    def test_identity_coherence_dict_with_direction(self):
        result = _extract_identity_coherence_summary({"direction": "convergent"}, 200)
        assert result == "convergent"

    def test_self_narrative_string(self):
        result = _extract_self_narrative_summary("narrative data", 200)
        assert result == "narrative data"

    def test_self_narrative_dict_with_fragment(self):
        result = _extract_self_narrative_summary({"fragment": "frag text"}, 200)
        assert result == "frag text"

    def test_self_narrative_dict_with_narrative(self):
        result = _extract_self_narrative_summary({"narrative": "narr text"}, 200)
        assert result == "narr text"

    def test_introspection_consumption_string(self):
        result = _extract_introspection_consumption_summary("consumed data", 200)
        assert result == "consumed data"

    def test_introspection_consumption_list_of_strings(self):
        result = _extract_introspection_consumption_summary(["a", "b", "c"], 200)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_introspection_consumption_list_of_dicts(self):
        result = _extract_introspection_consumption_summary(
            [{"summary": "s1"}, {"text": "t2"}], 200
        )
        assert "s1" in result
        assert "t2" in result

    def test_introspection_consumption_empty_list(self):
        result = _extract_introspection_consumption_summary([], 200)
        assert result == ABSENT_MARKER

    def test_meta_emotion_string(self):
        result = _extract_meta_emotion_summary("meta data", 200)
        assert result == "meta data"

    def test_meta_emotion_dict_with_pattern_candidates(self):
        result = _extract_meta_emotion_summary(
            {"pattern_candidates": "candidate text"}, 200
        )
        assert result == "candidate text"

    def test_meta_emotion_none(self):
        result = _extract_meta_emotion_summary(None, 200)
        assert result == ABSENT_MARKER

    def test_all_extractors_registered(self):
        """全6断面の抽出関数が登録されている。"""
        for section_name in SECTION_ORDER:
            assert section_name in _SECTION_EXTRACTORS

    def test_each_extractor_independent(self):
        """各断面の要約は独立に生成される（他の断面を参照しない）。"""
        # 抽出関数のシグネチャが(output, max_length)のみであることを確認
        for section_name in SECTION_ORDER:
            extractor = _SECTION_EXTRACTORS[section_name]
            # 各抽出関数はoutputとmax_lengthのみを引数に取る
            result = extractor("test", 200)
            assert isinstance(result, str)


# =============================================================================
# スナップショット構成テスト (Stage 2)
# =============================================================================

class TestSnapshotConstruction:
    """スナップショット構成テスト。"""

    def test_single_snapshot_creation(self):
        proc = make_processor()
        outputs = make_full_outputs()
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        assert snap.tick == 1
        assert snap.timestamp == 1000.0
        assert len(snap.sections) == 6

    def test_all_sections_present_in_snapshot(self):
        proc = make_processor()
        outputs = make_full_outputs()
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert section_name in snap.sections

    def test_snapshot_sections_contain_output_data(self):
        proc = make_processor()
        outputs = make_full_outputs(prefix="mydata")
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert snap.sections[section_name] != ABSENT_MARKER

    def test_snapshot_immutable_after_creation(self):
        """スナップショットは構成後に変更されない。"""
        proc = make_processor()
        outputs1 = make_full_outputs(prefix="first")
        snap1 = proc.process(outputs1, tick=1, timestamp=1000.0)
        snap1_sections = dict(snap1.sections)

        outputs2 = make_full_outputs(prefix="second")
        proc.process(outputs2, tick=2, timestamp=2000.0)

        # 最初のスナップショットの内容は変わらない
        assert snap1.sections == snap1_sections

    def test_snapshot_added_to_window(self):
        proc = make_processor()
        outputs = make_full_outputs()
        proc.process(outputs, tick=1, timestamp=1000.0)
        assert len(proc.state.snapshot_window) == 1

    def test_multiple_snapshots_accumulate(self):
        proc = make_processor()
        process_n(proc, 5)
        assert len(proc.state.snapshot_window) == 5

    def test_previous_snapshot_updated(self):
        proc = make_processor()
        process_n(proc, 2)
        prev = proc.get_previous_snapshot()
        assert prev is not None
        assert prev["tick"] == 1

    def test_previous_snapshot_none_after_first(self):
        """最初のスナップショットでは直前は存在しない（processの中で更新される前）。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        # 1件目のprocess時点ではwindowが空だったのでprevious_snapshotはNone
        assert proc.state.previous_snapshot is None


# =============================================================================
# スライディングウィンドウのFIFO動作テスト
# =============================================================================

class TestSlidingWindowFIFO:
    """スライディングウィンドウのFIFO動作テスト。"""

    def test_window_respects_max_size(self):
        proc = make_processor(max_snapshots=5)
        process_n(proc, 10)
        assert len(proc.state.snapshot_window) == 5

    def test_oldest_pushed_out_first(self):
        proc = make_processor(max_snapshots=3)
        process_n(proc, 5, base_tick=1)
        # tick 1, 2 が押し出され、3, 4, 5 が残る
        ticks = [s.tick for s in proc.state.snapshot_window]
        assert ticks == [3, 4, 5]

    def test_fifo_order_preserved(self):
        proc = make_processor(max_snapshots=4)
        process_n(proc, 8, base_tick=10)
        ticks = [s.tick for s in proc.state.snapshot_window]
        assert ticks == [14, 15, 16, 17]  # 10-13 pushed out

    def test_window_size_one(self):
        proc = make_processor(max_snapshots=1)
        process_n(proc, 3, base_tick=1)
        assert len(proc.state.snapshot_window) == 1
        assert proc.state.snapshot_window[0].tick == 3

    def test_pushout_is_only_data_loss_path(self):
        """押し出しが唯一のデータ消失経路。
        特定のスナップショットを選択的に消去する処理は存在しない。"""
        proc = make_processor(max_snapshots=5)
        process_n(proc, 3)
        # 3件あり上限未満なので全て残る
        assert len(proc.state.snapshot_window) == 3


# =============================================================================
# enrichment形式テスト (Stage 3)
# =============================================================================

class TestEnrichmentFormat:
    """enrichment形式のテスト。"""

    def test_enrichment_contains_all_sections(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        for label in SECTION_LABELS.values():
            assert label in text

    def test_enrichment_contains_tick_label(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=42, timestamp=1000.0)
        text = proc.get_enrichment_text()
        assert "t42" in text

    def test_enrichment_section_order_fixed(self):
        """列挙順序は断面の定義順に固定。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        positions = []
        for section_name in SECTION_ORDER:
            label = SECTION_LABELS[section_name]
            pos = text.find(label)
            assert pos >= 0, f"Label {label} not found in enrichment"
            positions.append(pos)
        # 位置は昇順であること
        assert positions == sorted(positions)

    def test_enrichment_limited_snapshot_count(self):
        proc = make_processor(max_enrichment_snapshots=2)
        process_n(proc, 5, base_tick=1)
        text = proc.get_enrichment_text()
        # tick4とtick5のみ含まれる
        assert "t4" in text
        assert "t5" in text
        assert "t1" not in text

    def test_enrichment_size_limit(self):
        proc = make_processor(max_enrichment_length=50)
        process_n(proc, 5, base_tick=1)
        text = proc.get_enrichment_text()
        assert len(text) <= 50

    def test_enrichment_uses_separator(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        # 断面間は "/" で区切られる
        assert " / " in text

    def test_enrichment_no_evaluative_words(self):
        """enrichment出力に評価的語彙が含まれない。"""
        proc = make_processor()
        outputs = make_full_outputs()
        outputs[SECTION_SELF_MODEL] = "状態は良好で正常です"
        proc.process(outputs, tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        for word in _EVALUATIVE_WORDS:
            assert word not in text


# =============================================================================
# READ-ONLYアクセサテスト
# =============================================================================

class TestReadOnlyAccessors:
    """READ-ONLYアクセサのテスト。"""

    def test_get_snapshot_window_returns_copies(self):
        proc = make_processor()
        process_n(proc, 3)
        window = proc.get_snapshot_window()
        # 変更しても内部状態に影響しない
        window.clear()
        assert len(proc.get_snapshot_window()) == 3

    def test_get_snapshot_window_dict_format(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        window = proc.get_snapshot_window()
        assert len(window) == 1
        snap = window[0]
        assert "sections" in snap
        assert "tick" in snap
        assert "timestamp" in snap

    def test_get_previous_snapshot_returns_copy(self):
        proc = make_processor()
        process_n(proc, 2)
        prev = proc.get_previous_snapshot()
        assert prev is not None
        prev["tick"] = 9999
        actual_prev = proc.get_previous_snapshot()
        assert actual_prev["tick"] != 9999

    def test_get_latest_snapshot(self):
        proc = make_processor()
        process_n(proc, 3, base_tick=10)
        latest = proc.get_latest_snapshot()
        assert latest is not None
        assert latest["tick"] == 12

    def test_get_latest_snapshot_returns_copy(self):
        proc = make_processor()
        process_n(proc, 2)
        latest = proc.get_latest_snapshot()
        latest["tick"] = 9999
        actual_latest = proc.get_latest_snapshot()
        assert actual_latest["tick"] != 9999

    def test_get_summary(self):
        proc = make_processor(max_snapshots=5)
        process_n(proc, 3)
        summary = proc.get_summary()
        assert summary["window_size"] == 3
        assert summary["has_previous_snapshot"] is True
        assert summary["max_snapshots"] == 5


# =============================================================================
# save/load round-trip テスト
# =============================================================================

class TestSaveLoad:
    """save/load round-tripテスト。"""

    def test_save_returns_dict(self):
        proc = make_processor()
        process_n(proc, 3)
        data = proc.save()
        assert isinstance(data, dict)
        assert "snapshot_window" in data
        assert "previous_snapshot" in data

    def test_load_restores_state(self):
        proc1 = make_processor()
        process_n(proc1, 5, base_tick=1)
        data = proc1.save()

        proc2 = make_processor()
        proc2.load(data)
        assert len(proc2.state.snapshot_window) == 5
        ticks = [s.tick for s in proc2.state.snapshot_window]
        assert ticks == [1, 2, 3, 4, 5]

    def test_save_load_round_trip_window(self):
        proc1 = make_processor()
        process_n(proc1, 3, base_tick=10)
        data = proc1.save()

        proc2 = make_processor()
        proc2.load(data)
        w1 = proc1.get_snapshot_window()
        w2 = proc2.get_snapshot_window()
        assert w1 == w2

    def test_save_load_previous_snapshot(self):
        proc1 = make_processor()
        process_n(proc1, 3, base_tick=1)
        data = proc1.save()

        proc2 = make_processor()
        proc2.load(data)
        prev1 = proc1.get_previous_snapshot()
        prev2 = proc2.get_previous_snapshot()
        assert prev1 == prev2

    def test_save_load_empty_state(self):
        proc1 = make_processor()
        data = proc1.save()

        proc2 = make_processor()
        proc2.load(data)
        assert len(proc2.state.snapshot_window) == 0
        assert proc2.state.previous_snapshot is None

    def test_save_load_preserves_enrichment(self):
        proc1 = make_processor()
        process_n(proc1, 2)
        text1 = proc1.get_enrichment_text()
        data = proc1.save()

        proc2 = make_processor()
        proc2.load(data)
        text2 = proc2.get_enrichment_text()
        assert text1 == text2

    def test_snapshot_to_dict_from_dict(self):
        snap = CrossSectionSnapshot(
            sections={"a": "va", "b": "vb"},
            tick=42,
            timestamp=1234.5,
        )
        d = snap.to_dict()
        restored = CrossSectionSnapshot.from_dict(d)
        assert restored.sections == snap.sections
        assert restored.tick == snap.tick
        assert restored.timestamp == snap.timestamp

    def test_state_to_dict_from_dict(self):
        state = IntrospectionCrossSectionState(
            snapshot_window=[
                CrossSectionSnapshot(sections={"x": "y"}, tick=1, timestamp=100.0),
            ],
            previous_snapshot=CrossSectionSnapshot(
                sections={"a": "b"}, tick=0, timestamp=50.0
            ),
        )
        d = state.to_dict()
        restored = IntrospectionCrossSectionState.from_dict(d)
        assert len(restored.snapshot_window) == 1
        assert restored.snapshot_window[0].tick == 1
        assert restored.previous_snapshot is not None
        assert restored.previous_snapshot.tick == 0


# =============================================================================
# 安全弁テスト（5つ）
# =============================================================================

class TestSafetyValve1_PatternExtractionProhibition:
    """安全弁1: パターン抽出の禁止。
    蓄積されたスナップショット群から断面間の相関・因果・傾向を
    算出する処理が存在しないことを確認する。
    """

    def test_no_correlation_method(self):
        proc = make_processor()
        # 相関算出メソッドが存在しないことを確認
        assert not hasattr(proc, "compute_correlation")
        assert not hasattr(proc, "compute_causation")
        assert not hasattr(proc, "compute_trend")

    def test_no_pattern_extraction_method(self):
        proc = make_processor()
        assert not hasattr(proc, "extract_patterns")
        assert not hasattr(proc, "detect_patterns")

    def test_no_statistical_processing(self):
        proc = make_processor()
        assert not hasattr(proc, "compute_statistics")
        assert not hasattr(proc, "aggregate")

    def test_snapshot_window_is_raw_data(self):
        """スナップショットは加工されていない生データ。"""
        proc = make_processor()
        process_n(proc, 5)
        window = proc.get_snapshot_window()
        for snap in window:
            assert "sections" in snap
            # 相関・因果・傾向のキーが存在しないこと
            assert "correlation" not in snap
            assert "trend" not in snap
            assert "pattern" not in snap


class TestSafetyValve2_AllSectionsEquivalent:
    """安全弁2: 全断面の等価性。
    特定の断面に重み・注目度・重要度を付与しないことを確認する。
    """

    def test_no_weight_in_snapshot(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        snap = proc.get_latest_snapshot()
        for section_name in SECTION_ORDER:
            section_val = snap["sections"][section_name]
            assert isinstance(section_val, str)
            # 重みフィールドが付いていないこと
        assert "weight" not in snap
        assert "priority" not in snap
        assert "importance" not in snap

    def test_no_attention_field(self):
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        snap = proc.get_latest_snapshot()
        assert "attention" not in snap
        assert "focus" not in snap

    def test_enrichment_no_emphasis(self):
        """enrichmentテキストで特定の断面を特別に強調する表現を使わない。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        # 強調表現の不在を確認
        assert "**" not in text  # Markdown bold
        assert "!!" not in text
        assert "注目" not in text
        assert "重要" not in text

    def test_all_sections_same_format_in_enrichment(self):
        """全断面が同じフォーマットで列挙される。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        # 全断面ラベルが含まれていること
        for section_name in SECTION_ORDER:
            label = SECTION_LABELS[section_name]
            assert f"{label}=" in text


class TestSafetyValve3_IntegrationProhibition:
    """安全弁3: 統合の禁止。
    6断面の要約を1つの要約文・スコア・状態記述にまとめることを禁止する。
    """

    def test_no_unified_summary_method(self):
        proc = make_processor()
        assert not hasattr(proc, "get_unified_summary")
        assert not hasattr(proc, "get_overall_state")
        assert not hasattr(proc, "get_integrated_state")

    def test_no_single_score(self):
        proc = make_processor()
        assert not hasattr(proc, "get_score")
        assert not hasattr(proc, "compute_score")

    def test_snapshot_preserves_individual_sections(self):
        """スナップショットは6断面が独立に並列した形式。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        snap = proc.get_latest_snapshot()
        assert len(snap["sections"]) == 6
        # 統合キーが存在しないこと
        assert "integrated" not in snap
        assert "unified" not in snap
        assert "overall" not in snap

    def test_enrichment_not_single_sentence(self):
        """enrichmentが「内省全体の状態は...」という単一要約文ではない。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        assert "内省全体" not in text
        assert "内省状態は" not in text


class TestSafetyValve4_WindowSizeLimit:
    """安全弁4: ウィンドウサイズの制限。
    スライディングウィンドウの上限を制限し、長期的な蓄積を構造的に防ぐ。
    """

    def test_window_size_limited(self):
        proc = make_processor(max_snapshots=5)
        process_n(proc, 100)
        assert len(proc.state.snapshot_window) == 5

    def test_default_window_limit(self):
        config = IntrospectionCrossSectionConfig()
        assert config.max_snapshots > 0
        assert config.max_snapshots <= 100  # 合理的な上限

    def test_window_limit_prevents_long_term_accumulation(self):
        """長期的なデータ蓄積が構造的に制限される。"""
        proc = make_processor(max_snapshots=3)
        process_n(proc, 50, base_tick=1)
        # 最新3件のみ残る
        assert len(proc.state.snapshot_window) == 3
        oldest_tick = proc.state.snapshot_window[0].tick
        assert oldest_tick == 48  # 1+47


class TestSafetyValve5_WritePathDisconnection:
    """安全弁5: 判断系・行動系・感情パイプラインへの書き込み経路の遮断。
    本機能の出力からいかなるモジュールの状態を書き換える経路が存在しないことを確認する。
    """

    def test_no_write_methods(self):
        """書き込みメソッドが存在しないこと。"""
        proc = make_processor()
        assert not hasattr(proc, "update_policy")
        assert not hasattr(proc, "update_emotion")
        assert not hasattr(proc, "update_memory")
        assert not hasattr(proc, "update_expectation")
        assert not hasattr(proc, "write_to_module")

    def test_output_is_read_only_data(self):
        """出力はenrichmentテキストとREAD-ONLYの構造化データのみ。"""
        proc = make_processor()
        process_n(proc, 3)
        # enrichmentテキスト: 文字列
        text = proc.get_enrichment_text()
        assert isinstance(text, str)
        # スナップショットウィンドウ: リスト
        window = proc.get_snapshot_window()
        assert isinstance(window, list)
        # 直前スナップショット: 辞書
        prev = proc.get_previous_snapshot()
        assert isinstance(prev, dict)

    def test_process_returns_snapshot_only(self):
        """process()の戻り値はスナップショットのみ。"""
        proc = make_processor()
        result = proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        assert isinstance(result, CrossSectionSnapshot)


# =============================================================================
# 経路遮断テスト（5つ）
# =============================================================================

class TestPathDisconnection:
    """5つの経路遮断を確認するテスト。

    1. 横断的記述 → 各内省系モジュールの入力
    2. 横断的記述 → ポリシー選択パイプライン
    3. 横断的記述 → 感情パイプライン
    4. 横断的記述 → 記憶忘却/固定化パラメータ
    5. 横断的記述 → 予期形成
    """

    def test_no_introspection_module_input_method(self):
        """経路1: 各内省系モジュールの入力への接続がない。"""
        proc = make_processor()
        assert not hasattr(proc, "feed_to_self_model")
        assert not hasattr(proc, "feed_to_temporal_self_difference")
        assert not hasattr(proc, "feed_to_identity_coherence")
        assert not hasattr(proc, "feed_to_self_narrative")
        assert not hasattr(proc, "feed_to_introspection_consumption")
        assert not hasattr(proc, "feed_to_meta_emotion_cognition")

    def test_no_policy_pipeline_method(self):
        """経路2: ポリシー選択パイプラインへの接続がない。"""
        proc = make_processor()
        assert not hasattr(proc, "feed_to_policy")
        assert not hasattr(proc, "update_bias")
        assert not hasattr(proc, "update_stability_valve")

    def test_no_emotion_pipeline_method(self):
        """経路3: 感情パイプラインへの接続がない。"""
        proc = make_processor()
        assert not hasattr(proc, "feed_to_emotion")
        assert not hasattr(proc, "update_emotion_params")
        assert not hasattr(proc, "modify_decay")

    def test_no_memory_forgetting_method(self):
        """経路4: 記憶忘却/固定化パラメータへの接続がない。"""
        proc = make_processor()
        assert not hasattr(proc, "feed_to_forgetting")
        assert not hasattr(proc, "update_fixation_threshold")
        assert not hasattr(proc, "modify_forgetting_speed")

    def test_no_expectation_formation_method(self):
        """経路5: 予期形成への接続がない。"""
        proc = make_processor()
        assert not hasattr(proc, "feed_to_expectation")
        assert not hasattr(proc, "update_expectation_strength")
        assert not hasattr(proc, "modify_expectation_duration")

    def test_process_only_reads_input(self):
        """process()は入力を読み取るのみで、入力辞書を変更しない。"""
        proc = make_processor()
        outputs = make_full_outputs()
        outputs_copy = dict(outputs)
        proc.process(outputs, tick=1, timestamp=1000.0)
        assert outputs == outputs_copy


# =============================================================================
# 不在断面テスト
# =============================================================================

class TestAbsentSections:
    """不在断面のテスト。"""

    def test_all_sections_absent(self):
        proc = make_processor()
        snap = proc.process({}, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert snap.sections[section_name] == ABSENT_MARKER

    def test_partial_sections_absent(self):
        proc = make_processor()
        outputs = {
            SECTION_SELF_MODEL: "present data",
            # 他は不在
        }
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        assert snap.sections[SECTION_SELF_MODEL] != ABSENT_MARKER
        assert snap.sections[SECTION_TEMPORAL_SELF_DIFFERENCE] == ABSENT_MARKER
        assert snap.sections[SECTION_IDENTITY_COHERENCE] == ABSENT_MARKER

    def test_absent_not_treated_as_error(self):
        """不在を異常やエラーとして扱わない。"""
        proc = make_processor()
        # 例外が発生しないこと
        snap = proc.process({}, tick=1, timestamp=1000.0)
        assert snap is not None

    def test_absent_in_enrichment(self):
        proc = make_processor()
        proc.process({}, tick=1, timestamp=1000.0)
        text = proc.get_enrichment_text()
        assert ABSENT_MARKER in text

    def test_none_values_treated_as_absent(self):
        proc = make_processor()
        outputs = {section: None for section in SECTION_ORDER}
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert snap.sections[section_name] == ABSENT_MARKER


# =============================================================================
# エッジケーステスト
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_tick_zero(self):
        proc = make_processor()
        snap = proc.process(make_full_outputs(), tick=0, timestamp=0.0)
        assert snap.tick == 0

    def test_negative_tick(self):
        proc = make_processor()
        snap = proc.process(make_full_outputs(), tick=-1, timestamp=0.0)
        assert snap.tick == -1

    def test_very_large_tick(self):
        proc = make_processor()
        snap = proc.process(make_full_outputs(), tick=999999, timestamp=1000.0)
        assert snap.tick == 999999

    def test_timestamp_defaults_to_current_time(self):
        proc = make_processor()
        before = time.time()
        snap = proc.process(make_full_outputs(), tick=1)
        after = time.time()
        assert before <= snap.timestamp <= after

    def test_empty_string_outputs(self):
        proc = make_processor()
        outputs = {section: "" for section in SECTION_ORDER}
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert snap.sections[section_name] == ABSENT_MARKER

    def test_whitespace_outputs(self):
        proc = make_processor()
        outputs = {section: "   " for section in SECTION_ORDER}
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert snap.sections[section_name] == ABSENT_MARKER

    def test_very_long_output_truncated(self):
        proc = make_processor(max_summary_length=50)
        outputs = {section: "x" * 1000 for section in SECTION_ORDER}
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        for section_name in SECTION_ORDER:
            assert len(snap.sections[section_name]) <= 50

    def test_dict_output_with_empty_values(self):
        proc = make_processor()
        outputs = {
            SECTION_SELF_MODEL: {"summary": "", "description": "", "text": ""},
        }
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        assert snap.sections[SECTION_SELF_MODEL] == ABSENT_MARKER

    def test_unknown_section_in_outputs_ignored(self):
        proc = make_processor()
        outputs = make_full_outputs()
        outputs["unknown_section"] = "some data"
        snap = proc.process(outputs, tick=1, timestamp=1000.0)
        assert "unknown_section" not in snap.sections

    def test_rapid_successive_processing(self):
        proc = make_processor(max_snapshots=5)
        for i in range(100):
            proc.process(make_full_outputs(prefix=f"r{i}"), tick=i, timestamp=float(i))
        assert len(proc.state.snapshot_window) == 5

    def test_window_size_zero_config(self):
        """ウィンドウサイズ0の場合（エッジ）、追加後すぐに押し出される。"""
        proc = make_processor(max_snapshots=0)
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        assert len(proc.state.snapshot_window) == 0

    def test_load_with_extra_keys_ignored(self):
        """ロードデータに不明なキーがあっても無視する。"""
        proc = make_processor()
        data = {
            "snapshot_window": [],
            "previous_snapshot": None,
            "unknown_key": "extra",
        }
        proc.load(data)
        assert len(proc.state.snapshot_window) == 0


# =============================================================================
# ファクトリテスト
# =============================================================================

class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_default(self):
        proc = create_introspection_cross_section()
        assert isinstance(proc, IntrospectionCrossSectionProcessor)

    def test_create_with_custom_config(self):
        config = IntrospectionCrossSectionConfig(max_snapshots=5)
        proc = create_introspection_cross_section(config=config)
        assert proc._config.max_snapshots == 5

    def test_factory_returns_fresh_state(self):
        proc = create_introspection_cross_section()
        assert len(proc.state.snapshot_window) == 0
        assert proc.state.previous_snapshot is None


# =============================================================================
# 定数テスト
# =============================================================================

class TestConstants:
    """定数・定義のテスト。"""

    def test_section_order_has_6_sections(self):
        assert len(SECTION_ORDER) == 6

    def test_section_labels_match_order(self):
        for section_name in SECTION_ORDER:
            assert section_name in SECTION_LABELS

    def test_absent_marker_is_string(self):
        assert isinstance(ABSENT_MARKER, str)
        assert len(ABSENT_MARKER) > 0

    def test_extension_candidates_defined(self):
        """拡張候補モジュールが定義のみ保持されている。"""
        assert len(EXTENSION_CANDIDATES) > 0
        # 拡張候補は初期対象に含まれない
        for candidate in EXTENSION_CANDIDATES:
            assert candidate not in SECTION_ORDER

    def test_evaluative_words_defined(self):
        assert len(_EVALUATIVE_WORDS) > 0


# =============================================================================
# ヘルパー関数テスト
# =============================================================================

class TestHelperFunctions:
    """ヘルパー関数のテスト。"""

    def test_truncate_short_text(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_exact_length(self):
        assert _truncate("hello", 5) == "hello"

    def test_truncate_long_text(self):
        assert _truncate("hello world", 5) == "hello"

    def test_sanitize_no_evaluative(self):
        assert _sanitize_summary("普通のテキスト") == "普通のテキスト"

    def test_sanitize_removes_evaluative(self):
        result = _sanitize_summary("これは良好な状態")
        assert "良好" not in result
        assert "[...]" in result

    def test_sanitize_multiple_evaluative(self):
        result = _sanitize_summary("良好で正常な異常がない")
        assert "良好" not in result
        assert "正常" not in result
        assert "異常" not in result


# =============================================================================
# self_image_integrationとの責務分離テスト
# =============================================================================

class TestResponsibilitySeparation:
    """self_image_integrationとの責務分離。
    本機能は「並置」であり「統合」ではない。
    """

    def test_output_is_juxtaposition_not_integration(self):
        """出力は各断面が独立に並列した形式（並置）。"""
        proc = make_processor()
        proc.process(make_full_outputs(), tick=1, timestamp=1000.0)
        snap = proc.get_latest_snapshot()
        # 全6断面が独立に存在
        assert len(snap["sections"]) == 6
        # 統合されたキーが存在しない
        for key in snap:
            assert key in ("sections", "tick", "timestamp")


# =============================================================================
# temporal_cognitionとの責務分離テスト
# =============================================================================

class TestTemporalCognitionSeparation:
    """temporal_cognitionとの責務分離。
    本機能は「何が」の記述、時間認知は「いつ」の記述。
    """

    def test_no_temporal_feature_computation(self):
        """時間的特徴量の算出は行わない。"""
        proc = make_processor()
        assert not hasattr(proc, "describe_features")
        assert not hasattr(proc, "classify_density")
        assert not hasattr(proc, "classify_tempo")


# =============================================================================
# 統合テスト
# =============================================================================

class TestIntegration:
    """統合テスト。"""

    def test_full_pipeline(self):
        """3段パイプラインの一括実行テスト。"""
        proc = make_processor(max_snapshots=5, max_enrichment_snapshots=2)

        # 5件処理
        process_n(proc, 5, base_tick=1, base_timestamp=1000.0)

        # ウィンドウ確認
        assert len(proc.state.snapshot_window) == 5
        window = proc.get_snapshot_window()
        assert len(window) == 5

        # enrichment確認
        text = proc.get_enrichment_text()
        assert "t4" in text
        assert "t5" in text

        # 直前スナップショット確認
        prev = proc.get_previous_snapshot()
        assert prev is not None
        assert prev["tick"] == 4

        # 最新スナップショット確認
        latest = proc.get_latest_snapshot()
        assert latest["tick"] == 5

    def test_full_pipeline_with_pushout(self):
        """押し出しが発生するパイプラインテスト。"""
        proc = make_processor(max_snapshots=3)

        process_n(proc, 10, base_tick=1)

        assert len(proc.state.snapshot_window) == 3
        ticks = [s.tick for s in proc.state.snapshot_window]
        assert ticks == [8, 9, 10]

    def test_save_load_then_continue(self):
        """save/load後に処理を継続できる。"""
        proc1 = make_processor(max_snapshots=5)
        process_n(proc1, 3, base_tick=1, base_timestamp=1000.0)
        data = proc1.save()

        proc2 = make_processor(max_snapshots=5)
        proc2.load(data)
        process_n(proc2, 2, base_tick=4, base_timestamp=1030.0)

        assert len(proc2.state.snapshot_window) == 5
        ticks = [s.tick for s in proc2.state.snapshot_window]
        assert ticks == [1, 2, 3, 4, 5]

    def test_mixed_absent_and_present_sections(self):
        """一部の断面が不在の場合の統合テスト。"""
        proc = make_processor()
        outputs = {
            SECTION_SELF_MODEL: "model output",
            SECTION_META_EMOTION_COGNITION: {"summary": "meta output"},
        }
        snap = proc.process(outputs, tick=1, timestamp=1000.0)

        # 提供された断面は値が入り、その他は不在
        assert snap.sections[SECTION_SELF_MODEL] != ABSENT_MARKER
        assert snap.sections[SECTION_META_EMOTION_COGNITION] != ABSENT_MARKER
        assert snap.sections[SECTION_TEMPORAL_SELF_DIFFERENCE] == ABSENT_MARKER
        assert snap.sections[SECTION_IDENTITY_COHERENCE] == ABSENT_MARKER

        # enrichmentに全断面が含まれる
        text = proc.get_enrichment_text()
        for label in SECTION_LABELS.values():
            assert label in text

    def test_state_setter(self):
        """state setterのテスト。"""
        proc = make_processor()
        new_state = IntrospectionCrossSectionState(
            snapshot_window=[
                CrossSectionSnapshot(
                    sections={s: "test" for s in SECTION_ORDER},
                    tick=99,
                    timestamp=9999.0,
                ),
            ],
        )
        proc.state = new_state
        assert len(proc.state.snapshot_window) == 1
        assert proc.state.snapshot_window[0].tick == 99
