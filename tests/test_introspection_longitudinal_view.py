"""
tests/test_introspection_longitudinal_view.py - 内省の時間的縦断参照のテスト

テスト対象: psyche/introspection_longitudinal_view.py

テスト範囲:
- 全機能テスト（3段パイプライン、enrichment生成、READ-ONLYアクセサ）
- パターン抽出がないことのテスト（安全弁1）
- 全断面等価性テスト（安全弁2）
- 全時点等価性テスト（安全弁3）
- 独自状態蓄積がないことのテスト（安全弁4）
- エッジケーステスト（空入力、1件のみ、全断面同値など）
"""

import time
import pytest

from psyche.introspection_cross_section import (
    SECTION_ORDER,
    SECTION_LABELS,
    ABSENT_MARKER,
)
from psyche.introspection_longitudinal_view import (
    # Configuration
    LongitudinalViewConfig,
    # Data structures
    TimePointEntry,
    SectionTimeline,
    LongitudinalView,
    # Pipeline functions
    _stage1_get_snapshots,
    _stage2_transform_to_longitudinal,
    _stage3_prepare_handoff,
    # Enrichment
    _generate_enrichment_text,
    # Public API
    get_enrichment_data,
    get_longitudinal_view,
    get_section_timeline,
    # Processor class
    IntrospectionLongitudinalViewProcessor,
    create_introspection_longitudinal_view,
)


# =============================================================================
# Helpers: snapshot generation
# =============================================================================

def _make_snapshot(
    tick: int,
    sections: dict[str, str] | None = None,
    timestamp: float | None = None,
) -> dict:
    """テスト用スナップショット辞書を生成。"""
    if sections is None:
        sections = {
            name: f"value_{name}_t{tick}"
            for name in SECTION_ORDER
        }
    if timestamp is None:
        timestamp = 1000.0 + tick * 10.0
    return {
        "sections": sections,
        "tick": tick,
        "timestamp": timestamp,
    }


def _make_snapshots(count: int) -> list[dict]:
    """複数のスナップショットを生成。"""
    return [_make_snapshot(tick=i + 1) for i in range(count)]


# =============================================================================
# Test: Data Structures
# =============================================================================

class TestTimePointEntry:
    """TimePointEntryのテスト。"""

    def test_creation(self):
        entry = TimePointEntry(value="test", tick=1, timestamp=1000.0)
        assert entry.value == "test"
        assert entry.tick == 1
        assert entry.timestamp == 1000.0

    def test_no_evaluative_attributes(self):
        """評価的属性（重み・スコア・優先度・重要度）を持たないことを確認。"""
        entry = TimePointEntry(value="test", tick=1, timestamp=1000.0)
        assert not hasattr(entry, "weight")
        assert not hasattr(entry, "score")
        assert not hasattr(entry, "priority")
        assert not hasattr(entry, "importance")


class TestSectionTimeline:
    """SectionTimelineのテスト。"""

    def test_creation_empty(self):
        tl = SectionTimeline(section_name="self_model")
        assert tl.section_name == "self_model"
        assert tl.entries == []

    def test_creation_with_entries(self):
        entries = [
            TimePointEntry(value=f"v{i}", tick=i, timestamp=float(i))
            for i in range(3)
        ]
        tl = SectionTimeline(section_name="test", entries=entries)
        assert len(tl.entries) == 3

    def test_to_dict(self):
        entries = [
            TimePointEntry(value="v1", tick=1, timestamp=100.0),
            TimePointEntry(value="v2", tick=2, timestamp=200.0),
        ]
        tl = SectionTimeline(section_name="self_model", entries=entries)
        d = tl.to_dict()
        assert d["section_name"] == "self_model"
        assert len(d["entries"]) == 2
        assert d["entries"][0]["value"] == "v1"
        assert d["entries"][0]["tick"] == 1
        assert d["entries"][0]["timestamp"] == 100.0
        assert d["entries"][1]["value"] == "v2"

    def test_no_evaluative_attributes(self):
        """タイムラインに評価的属性がないことを確認。"""
        tl = SectionTimeline(section_name="test")
        assert not hasattr(tl, "weight")
        assert not hasattr(tl, "trend")
        assert not hasattr(tl, "pattern")
        assert not hasattr(tl, "direction")


class TestLongitudinalView:
    """LongitudinalViewのテスト。"""

    def test_creation_empty(self):
        view = LongitudinalView()
        assert view.timelines == {}

    def test_to_dict(self):
        tl = SectionTimeline(
            section_name="self_model",
            entries=[TimePointEntry(value="v1", tick=1, timestamp=100.0)],
        )
        view = LongitudinalView(timelines={"self_model": tl})
        d = view.to_dict()
        assert "timelines" in d
        assert "self_model" in d["timelines"]


# =============================================================================
# Test: Stage 1 - Snapshot Window Retrieval
# =============================================================================

class TestStage1:
    """第1段: スナップショットウィンドウの取得テスト。"""

    def test_empty_input(self):
        """空入力は空リストを返す（異常やエラーとして扱わない）。"""
        result = _stage1_get_snapshots([])
        assert result == []

    def test_none_like_input(self):
        """空リストを返す。"""
        result = _stage1_get_snapshots([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_passthrough(self):
        """スナップショットリストをそのまま通過させる。"""
        snaps = _make_snapshots(3)
        result = _stage1_get_snapshots(snaps)
        assert len(result) == 3
        assert result is snaps  # same reference, no copy

    def test_single_snapshot(self):
        """単一スナップショット。"""
        snaps = [_make_snapshot(tick=1)]
        result = _stage1_get_snapshots(snaps)
        assert len(result) == 1


# =============================================================================
# Test: Stage 2 - Section-wise Timeline Juxtaposition
# =============================================================================

class TestStage2:
    """第2段: 断面別の時系列並置への変換テスト。"""

    def test_all_sections_present(self):
        """全6断面のタイムラインが生成されることを確認。"""
        snaps = _make_snapshots(3)
        timelines = _stage2_transform_to_longitudinal(snaps)
        assert len(timelines) == len(SECTION_ORDER)
        for name in SECTION_ORDER:
            assert name in timelines

    def test_time_order_preserved(self):
        """時間順が保持されることを確認。"""
        snaps = _make_snapshots(5)
        timelines = _stage2_transform_to_longitudinal(snaps)
        for name in SECTION_ORDER:
            tl = timelines[name]
            ticks = [e.tick for e in tl.entries]
            assert ticks == [1, 2, 3, 4, 5]

    def test_values_correctly_extracted(self):
        """各断面の値が正しく抽出されることを確認。"""
        snaps = _make_snapshots(2)
        timelines = _stage2_transform_to_longitudinal(snaps)
        for name in SECTION_ORDER:
            tl = timelines[name]
            assert tl.entries[0].value == f"value_{name}_t1"
            assert tl.entries[1].value == f"value_{name}_t2"

    def test_absent_section_preserved(self):
        """不在断面がABSENT_MARKERとして保持されることを確認。"""
        # 一部の断面のみ存在するスナップショット
        snap = {
            "sections": {"self_model": "test_value"},
            "tick": 1,
            "timestamp": 1000.0,
        }
        timelines = _stage2_transform_to_longitudinal([snap])
        # self_modelは値がある
        assert timelines["self_model"].entries[0].value == "test_value"
        # その他の断面はABSENT_MARKER
        for name in SECTION_ORDER:
            if name != "self_model":
                assert timelines[name].entries[0].value == ABSENT_MARKER

    def test_empty_sections_dict(self):
        """sectionsが空辞書の場合、全断面がABSENT_MARKER。"""
        snap = {"sections": {}, "tick": 1, "timestamp": 1000.0}
        timelines = _stage2_transform_to_longitudinal([snap])
        for name in SECTION_ORDER:
            assert timelines[name].entries[0].value == ABSENT_MARKER

    def test_timestamp_preserved(self):
        """タイムスタンプが保持されることを確認。"""
        snaps = [
            _make_snapshot(tick=1, timestamp=100.0),
            _make_snapshot(tick=2, timestamp=200.0),
        ]
        timelines = _stage2_transform_to_longitudinal(snaps)
        for name in SECTION_ORDER:
            tl = timelines[name]
            assert tl.entries[0].timestamp == 100.0
            assert tl.entries[1].timestamp == 200.0

    def test_no_diff_calculation(self):
        """時点間の差分・変化量を算出しないことを確認。"""
        snaps = _make_snapshots(3)
        timelines = _stage2_transform_to_longitudinal(snaps)
        for name in SECTION_ORDER:
            tl = timelines[name]
            for entry in tl.entries:
                # エントリにdiffやchange属性がないことを確認
                assert not hasattr(entry, "diff")
                assert not hasattr(entry, "change")
                assert not hasattr(entry, "delta")

    def test_no_filtering(self):
        """値に基づくフィルタリング・選別が行われないことを確認。"""
        # 同一値を含むスナップショット
        sections = {name: "same_value" for name in SECTION_ORDER}
        snaps = [
            {"sections": dict(sections), "tick": i, "timestamp": float(i)}
            for i in range(5)
        ]
        timelines = _stage2_transform_to_longitudinal(snaps)
        # 全時点が保持される（フィルタリングされない）
        for name in SECTION_ORDER:
            assert len(timelines[name].entries) == 5


# =============================================================================
# Test: Stage 3 - Handoff Preparation
# =============================================================================

class TestStage3:
    """第3段: 参照情報としての受渡準備テスト。"""

    def test_returns_longitudinal_view(self):
        """LongitudinalViewを返すことを確認。"""
        timelines = {
            name: SectionTimeline(section_name=name)
            for name in SECTION_ORDER
        }
        view = _stage3_prepare_handoff(timelines)
        assert isinstance(view, LongitudinalView)

    def test_timelines_preserved(self):
        """タイムラインがそのまま保持されることを確認。"""
        timelines = {
            name: SectionTimeline(
                section_name=name,
                entries=[TimePointEntry(value="v", tick=1, timestamp=1.0)],
            )
            for name in SECTION_ORDER
        }
        view = _stage3_prepare_handoff(timelines)
        assert len(view.timelines) == len(SECTION_ORDER)


# =============================================================================
# Test: Enrichment Generation
# =============================================================================

class TestEnrichmentGeneration:
    """enrichment生成テスト。"""

    def test_empty_view(self):
        """空のビューの場合「待機中」を返す。"""
        view = LongitudinalView()
        config = LongitudinalViewConfig()
        text = _generate_enrichment_text(view, config)
        assert "待機中" in text

    def test_no_entries_view(self):
        """エントリなしの場合「待機中」を返す。"""
        timelines = {
            name: SectionTimeline(section_name=name)
            for name in SECTION_ORDER
        }
        view = LongitudinalView(timelines=timelines)
        config = LongitudinalViewConfig()
        text = _generate_enrichment_text(view, config)
        assert "待機中" in text

    def test_basic_output(self):
        """基本的なenrichmentテキスト出力。"""
        snaps = _make_snapshots(3)
        timelines = _stage2_transform_to_longitudinal(snaps)
        view = _stage3_prepare_handoff(timelines)
        config = LongitudinalViewConfig()
        text = _generate_enrichment_text(view, config)

        # 全断面のラベルが含まれることを確認（等価性）
        for name in SECTION_ORDER:
            label = SECTION_LABELS.get(name, name)
            assert label in text

    def test_section_order_in_output(self):
        """断面の定義順に列挙されることを確認。"""
        snaps = _make_snapshots(2)
        timelines = _stage2_transform_to_longitudinal(snaps)
        view = _stage3_prepare_handoff(timelines)
        config = LongitudinalViewConfig()
        text = _generate_enrichment_text(view, config)

        # 各ラベルの位置を取得し、定義順であることを確認
        positions = []
        for name in SECTION_ORDER:
            label = SECTION_LABELS.get(name, name)
            pos = text.find(label)
            assert pos >= 0, f"Label {label} not found in enrichment text"
            positions.append(pos)
        assert positions == sorted(positions), "Section order not preserved"

    def test_max_enrichment_timepoints(self):
        """enrichmentの時点数上限が適用されることを確認。"""
        snaps = _make_snapshots(20)
        timelines = _stage2_transform_to_longitudinal(snaps)
        view = _stage3_prepare_handoff(timelines)
        config = LongitudinalViewConfig(max_enrichment_timepoints=5)
        text = _generate_enrichment_text(view, config)

        # 直近5件分のみが含まれることを確認
        # tick=16,17,18,19,20が含まれ、tick=1は含まれない
        first_section = SECTION_ORDER[0]
        label = SECTION_LABELS[first_section]
        line = [l for l in text.split("\n") if label in l][0]
        assert "t20" in line
        assert "t16" in line
        assert "t1:" not in line

    def test_max_enrichment_length(self):
        """enrichmentのサイズ上限が適用されることを確認。"""
        snaps = _make_snapshots(20)
        timelines = _stage2_transform_to_longitudinal(snaps)
        view = _stage3_prepare_handoff(timelines)
        config = LongitudinalViewConfig(max_enrichment_length=100)
        text = _generate_enrichment_text(view, config)
        assert len(text) <= 100

    def test_no_evaluative_words(self):
        """評価的語彙が使用されないことを確認。"""
        evaluative_words = [
            "良好", "異常", "正常", "理想的", "問題", "改善",
            "乱れ", "安定", "良い", "悪い", "適切", "不適切",
            "増加", "減少", "収束", "発散", "傾向",
        ]
        snaps = _make_snapshots(5)
        timelines = _stage2_transform_to_longitudinal(snaps)
        view = _stage3_prepare_handoff(timelines)
        config = LongitudinalViewConfig()
        text = _generate_enrichment_text(view, config)

        for word in evaluative_words:
            assert word not in text, f"Evaluative word '{word}' found in enrichment text"


# =============================================================================
# Test: Public API - get_enrichment_data
# =============================================================================

class TestGetEnrichmentData:
    """get_enrichment_data のテスト。"""

    def test_empty_input(self):
        result = get_enrichment_data([])
        assert result["summary_text"] == "内省縦断: 待機中"
        assert result["section_count"] == 0
        assert result["timepoint_count"] == 0

    def test_basic(self):
        snaps = _make_snapshots(3)
        result = get_enrichment_data(snaps)
        assert result["section_count"] == len(SECTION_ORDER)
        assert result["timepoint_count"] == 3
        assert "待機中" not in result["summary_text"]

    def test_custom_config(self):
        snaps = _make_snapshots(5)
        config = LongitudinalViewConfig(max_enrichment_timepoints=2)
        result = get_enrichment_data(snaps, config)
        assert result["timepoint_count"] == 5
        # enrichmentテキストには直近2件分のみ
        text = result["summary_text"]
        assert "t5" in text
        assert "t4" in text

    def test_stateless(self):
        """2回呼び出しても状態が蓄積されないことを確認。"""
        snaps1 = _make_snapshots(2)
        snaps2 = _make_snapshots(3)
        result1 = get_enrichment_data(snaps1)
        result2 = get_enrichment_data(snaps2)
        assert result1["timepoint_count"] == 2
        assert result2["timepoint_count"] == 3


# =============================================================================
# Test: Public API - get_longitudinal_view
# =============================================================================

class TestGetLongitudinalView:
    """get_longitudinal_view のテスト。"""

    def test_empty_input(self):
        result = get_longitudinal_view([])
        assert result["timelines"] == {}

    def test_basic(self):
        snaps = _make_snapshots(3)
        result = get_longitudinal_view(snaps)
        assert "timelines" in result
        assert len(result["timelines"]) == len(SECTION_ORDER)

    def test_all_sections_present(self):
        snaps = _make_snapshots(2)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            assert name in result["timelines"]

    def test_entries_time_ordered(self):
        snaps = _make_snapshots(5)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            ticks = [e["tick"] for e in entries]
            assert ticks == sorted(ticks)

    def test_all_timepoints_included(self):
        """全時点が等価に含まれることを確認（安全弁3）。"""
        snaps = _make_snapshots(5)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert len(entries) == 5

    def test_stateless_no_accumulation(self):
        """呼び出し間で状態が蓄積されないことを確認（安全弁4）。"""
        snaps1 = _make_snapshots(2)
        result1 = get_longitudinal_view(snaps1)

        snaps2 = _make_snapshots(3)
        result2 = get_longitudinal_view(snaps2)

        # 1回目は2時点、2回目は3時点（蓄積されない）
        for name in SECTION_ORDER:
            assert len(result1["timelines"][name]["entries"]) == 2
            assert len(result2["timelines"][name]["entries"]) == 3

    def test_no_pattern_extraction(self):
        """パターン抽出がないことを確認（安全弁1）。"""
        snaps = _make_snapshots(10)
        result = get_longitudinal_view(snaps)
        # 結果にpattern, trend, direction, correlation等のキーがないことを確認
        forbidden_keys = [
            "pattern", "trend", "direction", "correlation",
            "tendency", "statistics", "average", "variance",
            "mean", "median", "mode", "regression",
        ]
        for name in SECTION_ORDER:
            tl = result["timelines"][name]
            for key in forbidden_keys:
                assert key not in tl, f"Forbidden key '{key}' found in timeline"
            for entry in tl["entries"]:
                for key in forbidden_keys:
                    assert key not in entry, f"Forbidden key '{key}' found in entry"


# =============================================================================
# Test: Public API - get_section_timeline
# =============================================================================

class TestGetSectionTimeline:
    """get_section_timeline のテスト。"""

    def test_empty_input(self):
        result = get_section_timeline([], "self_model")
        assert result["section_name"] == "self_model"
        assert result["entries"] == []

    def test_basic(self):
        snaps = _make_snapshots(3)
        result = get_section_timeline(snaps, "self_model")
        assert result["section_name"] == "self_model"
        assert len(result["entries"]) == 3

    def test_unknown_section(self):
        """存在しない断面名の場合、空のタイムラインを返す。"""
        snaps = _make_snapshots(3)
        result = get_section_timeline(snaps, "nonexistent_section")
        assert result["section_name"] == "nonexistent_section"
        assert result["entries"] == []

    def test_each_section_equivalent(self):
        """どの断面を指定しても同一の処理が等価に適用される（安全弁2）。"""
        snaps = _make_snapshots(3)
        for name in SECTION_ORDER:
            result = get_section_timeline(snaps, name)
            assert result["section_name"] == name
            assert len(result["entries"]) == 3
            # 各エントリが正しい構造を持つ
            for entry in result["entries"]:
                assert "value" in entry
                assert "tick" in entry
                assert "timestamp" in entry

    def test_values_match_snapshots(self):
        """取得された値がスナップショットの値と一致することを確認。"""
        snaps = _make_snapshots(2)
        for name in SECTION_ORDER:
            result = get_section_timeline(snaps, name)
            for i, entry in enumerate(result["entries"]):
                expected_value = snaps[i]["sections"][name]
                assert entry["value"] == expected_value

    def test_absent_value_preserved(self):
        """不在値がABSENT_MARKERとして保持されることを確認。"""
        snap = {
            "sections": {"self_model": "present"},
            "tick": 1,
            "timestamp": 1000.0,
        }
        result = get_section_timeline([snap], "identity_coherence")
        assert result["entries"][0]["value"] == ABSENT_MARKER


# =============================================================================
# Test: Processor Class
# =============================================================================

class TestProcessor:
    """IntrospectionLongitudinalViewProcessor のテスト。"""

    def test_creation_default(self):
        proc = create_introspection_longitudinal_view()
        assert isinstance(proc, IntrospectionLongitudinalViewProcessor)

    def test_creation_with_config(self):
        config = LongitudinalViewConfig(max_enrichment_timepoints=5)
        proc = create_introspection_longitudinal_view(config)
        summary = proc.get_summary()
        assert summary["max_enrichment_timepoints"] == 5

    def test_process_empty(self):
        proc = create_introspection_longitudinal_view()
        result = proc.process([])
        assert result["timelines"] == {}

    def test_process_basic(self):
        proc = create_introspection_longitudinal_view()
        snaps = _make_snapshots(3)
        result = proc.process(snaps)
        assert "timelines" in result
        assert len(result["timelines"]) == len(SECTION_ORDER)

    def test_get_enrichment_data(self):
        proc = create_introspection_longitudinal_view()
        snaps = _make_snapshots(3)
        result = proc.get_enrichment_data(snaps)
        assert "summary_text" in result
        assert result["section_count"] == len(SECTION_ORDER)

    def test_get_longitudinal_view(self):
        proc = create_introspection_longitudinal_view()
        snaps = _make_snapshots(3)
        result = proc.get_longitudinal_view(snaps)
        assert "timelines" in result

    def test_get_section_timeline(self):
        proc = create_introspection_longitudinal_view()
        snaps = _make_snapshots(3)
        result = proc.get_section_timeline(snaps, "self_model")
        assert result["section_name"] == "self_model"
        assert len(result["entries"]) == 3

    def test_get_summary(self):
        proc = create_introspection_longitudinal_view()
        summary = proc.get_summary()
        assert summary["has_state"] is False
        assert "max_enrichment_timepoints" in summary
        assert "max_enrichment_length" in summary

    def test_no_internal_state_accumulation(self):
        """プロセッサが独自の内部状態を蓄積しないことを確認（安全弁4）。"""
        proc = create_introspection_longitudinal_view()

        # 1回目
        snaps1 = _make_snapshots(2)
        result1 = proc.process(snaps1)

        # 2回目（異なる入力）
        snaps2 = _make_snapshots(5)
        result2 = proc.process(snaps2)

        # 1回目の結果が2回目に影響しない
        for name in SECTION_ORDER:
            assert len(result1["timelines"][name]["entries"]) == 2
            assert len(result2["timelines"][name]["entries"]) == 5

    def test_multiple_calls_independent(self):
        """複数回呼び出しが独立であることを確認。"""
        proc = create_introspection_longitudinal_view()

        for i in range(5):
            snaps = _make_snapshots(i + 1)
            result = proc.process(snaps)
            for name in SECTION_ORDER:
                assert len(result["timelines"][name]["entries"]) == i + 1


# =============================================================================
# Test: Safety Valve 1 - No Pattern Extraction
# =============================================================================

class TestSafetyValve1NoPatternExtraction:
    """安全弁1: パターン抽出の禁止テスト。"""

    def test_no_trend_in_output(self):
        """出力に傾向・方向性・周期性が含まれないことを確認。"""
        snaps = _make_snapshots(10)
        result = get_longitudinal_view(snaps)
        # 辞書内にtrend/direction/pattern/cycleキーがないことを確認
        for name in SECTION_ORDER:
            tl_dict = result["timelines"][name]
            assert "trend" not in tl_dict
            assert "direction" not in tl_dict
            assert "pattern" not in tl_dict
            assert "cycle" not in tl_dict
            assert "correlation" not in tl_dict
            assert "statistics" not in tl_dict

    def test_no_statistical_processing(self):
        """統計処理が行われないことを確認。"""
        snaps = _make_snapshots(10)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            tl_dict = result["timelines"][name]
            assert "average" not in tl_dict
            assert "variance" not in tl_dict
            assert "mean" not in tl_dict
            assert "std" not in tl_dict
            assert "min" not in tl_dict
            assert "max" not in tl_dict

    def test_enrichment_no_pattern_words(self):
        """enrichmentテキストにパターン関連語が含まれないことを確認。"""
        snaps = _make_snapshots(5)
        result = get_enrichment_data(snaps)
        text = result["summary_text"]
        pattern_words = ["傾向", "パターン", "周期", "相関", "回帰", "統計"]
        for word in pattern_words:
            assert word not in text


# =============================================================================
# Test: Safety Valve 2 - All-Section Equivalence
# =============================================================================

class TestSafetyValve2AllSectionEquivalence:
    """安全弁2: 全断面の等価性テスト。"""

    def test_all_sections_same_entry_count(self):
        """全断面が同じエントリ数を持つことを確認。"""
        snaps = _make_snapshots(5)
        result = get_longitudinal_view(snaps)
        counts = [
            len(result["timelines"][name]["entries"])
            for name in SECTION_ORDER
        ]
        assert len(set(counts)) == 1, "All sections should have same entry count"

    def test_all_sections_same_structure(self):
        """全断面のエントリが同一の構造を持つことを確認。"""
        snaps = _make_snapshots(3)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            for entry in entries:
                assert set(entry.keys()) == {"value", "tick", "timestamp"}

    def test_no_section_has_special_weight(self):
        """特定の断面に重み・注目度・重要度が付与されていないことを確認。"""
        snaps = _make_snapshots(5)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            tl_dict = result["timelines"][name]
            assert "weight" not in tl_dict
            assert "importance" not in tl_dict
            assert "priority" not in tl_dict
            assert "attention" not in tl_dict

    def test_enrichment_all_sections_present(self):
        """enrichmentテキストに全断面が等価に含まれることを確認。"""
        snaps = _make_snapshots(3)
        result = get_enrichment_data(snaps)
        text = result["summary_text"]
        for name in SECTION_ORDER:
            label = SECTION_LABELS.get(name, name)
            assert label in text, f"Section '{label}' not found in enrichment"

    def test_section_order_consistent(self):
        """断面の列挙順が定義順に固定されていることを確認。"""
        snaps = _make_snapshots(3)
        result = get_longitudinal_view(snaps)
        timeline_keys = list(result["timelines"].keys())
        for i, name in enumerate(SECTION_ORDER):
            assert timeline_keys[i] == name

    def test_get_section_timeline_equivalence(self):
        """どの断面を指定しても同じ件数のエントリが返ることを確認。"""
        snaps = _make_snapshots(5)
        counts = []
        for name in SECTION_ORDER:
            result = get_section_timeline(snaps, name)
            counts.append(len(result["entries"]))
        assert len(set(counts)) == 1


# =============================================================================
# Test: Safety Valve 3 - All-Timepoint Equivalence
# =============================================================================

class TestSafetyValve3AllTimepointEquivalence:
    """安全弁3: 全時点の等価性テスト。"""

    def test_all_timepoints_included(self):
        """全時点が含まれることを確認（特定時点の除外なし）。"""
        snaps = _make_snapshots(10)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            ticks = [e["tick"] for e in entries]
            assert ticks == list(range(1, 11))

    def test_no_timepoint_filtering(self):
        """「変化が大きかった時点」「特異な値の時点」の選別がないことを確認。"""
        # 同じ値が続いた後に異なる値が来るパターン
        sections_same = {name: "same_value" for name in SECTION_ORDER}
        sections_diff = {name: "different_value" for name in SECTION_ORDER}
        snaps = [
            {"sections": dict(sections_same), "tick": 1, "timestamp": 100.0},
            {"sections": dict(sections_same), "tick": 2, "timestamp": 200.0},
            {"sections": dict(sections_diff), "tick": 3, "timestamp": 300.0},
            {"sections": dict(sections_same), "tick": 4, "timestamp": 400.0},
        ]
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            # 全4時点が含まれる
            assert len(entries) == 4

    def test_no_temporal_weighting(self):
        """「最新の値が最も重要」等の時間的重み付けがないことを確認。"""
        snaps = _make_snapshots(5)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            for entry in entries:
                assert "weight" not in entry
                assert "importance" not in entry
                assert "recency" not in entry

    def test_no_timepoint_highlighting(self):
        """enrichmentで特定の時点が強調されないことを確認。"""
        snaps = _make_snapshots(5)
        result = get_enrichment_data(snaps)
        text = result["summary_text"]
        # 強調記号（**や!!や★など）が含まれないことを確認
        assert "**" not in text
        assert "!!" not in text
        assert "★" not in text
        assert "※" not in text


# =============================================================================
# Test: Safety Valve 4 - No Internal State Accumulation
# =============================================================================

class TestSafetyValve4NoStateAccumulation:
    """安全弁4: 独自の状態蓄積の禁止テスト。"""

    def test_function_api_stateless(self):
        """関数API（get_longitudinal_view等）が状態を持たないことを確認。"""
        # 異なる入力で呼び出し、結果が独立であることを確認
        snaps_a = _make_snapshots(3)
        snaps_b = _make_snapshots(7)

        result_a = get_longitudinal_view(snaps_a)
        result_b = get_longitudinal_view(snaps_b)

        for name in SECTION_ORDER:
            assert len(result_a["timelines"][name]["entries"]) == 3
            assert len(result_b["timelines"][name]["entries"]) == 7

    def test_processor_stateless(self):
        """プロセッサクラスが呼び出し間で状態を蓄積しないことを確認。"""
        proc = create_introspection_longitudinal_view()

        # 3回異なる入力で呼び出し
        for n in [2, 5, 1]:
            snaps = _make_snapshots(n)
            result = proc.process(snaps)
            for name in SECTION_ORDER:
                assert len(result["timelines"][name]["entries"]) == n

    def test_no_save_load_needed(self):
        """プロセッサにsave/loadが不要であることを確認（独自状態なし）。"""
        proc = create_introspection_longitudinal_view()
        assert not hasattr(proc, "save") or not callable(getattr(proc, "save", None))
        assert not hasattr(proc, "load") or not callable(getattr(proc, "load", None))

    def test_summary_shows_no_state(self):
        """サマリーが状態なしを示すことを確認。"""
        proc = create_introspection_longitudinal_view()
        summary = proc.get_summary()
        assert summary["has_state"] is False


# =============================================================================
# Test: Safety Valve 5 - Pathway Severance
# =============================================================================

class TestSafetyValve5PathwaySeverance:
    """安全弁5: 書き込み経路の遮断テスト。"""

    def test_output_is_dict_copy(self):
        """出力が辞書のコピーであり、元データを変更しないことを確認。"""
        snaps = _make_snapshots(3)
        result = get_longitudinal_view(snaps)

        # 出力を変更しても元のスナップショットに影響しない
        result["timelines"]["self_model"]["entries"][0]["value"] = "modified"
        assert snaps[0]["sections"]["self_model"] != "modified"

    def test_no_write_methods(self):
        """プロセッサに書き込みメソッドがないことを確認。"""
        proc = create_introspection_longitudinal_view()
        # 書き込み系のメソッド名が存在しないことを確認
        write_methods = [
            "update", "set", "write", "modify", "mutate",
            "record", "store", "push", "append",
        ]
        for method_name in write_methods:
            assert not hasattr(proc, method_name), (
                f"Processor should not have write method '{method_name}'"
            )


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_empty_snapshots(self):
        """空のスナップショットリスト。"""
        result = get_longitudinal_view([])
        assert result["timelines"] == {}
        enrichment = get_enrichment_data([])
        assert "待機中" in enrichment["summary_text"]

    def test_single_snapshot(self):
        """単一スナップショット。"""
        snaps = _make_snapshots(1)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert len(entries) == 1

    def test_all_sections_same_value(self):
        """全断面が同一値のスナップショット。"""
        sections = {name: "identical_value" for name in SECTION_ORDER}
        snap = {"sections": sections, "tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == "identical_value"

    def test_all_sections_absent(self):
        """全断面が不在のスナップショット。"""
        snap = {"sections": {}, "tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == ABSENT_MARKER

    def test_mixed_absent_and_present(self):
        """不在と存在が混在するスナップショット系列。"""
        snaps = [
            {"sections": {"self_model": "v1"}, "tick": 1, "timestamp": 100.0},
            {"sections": {"identity_coherence": "v2"}, "tick": 2, "timestamp": 200.0},
            {"sections": {name: f"v3_{name}" for name in SECTION_ORDER}, "tick": 3, "timestamp": 300.0},
        ]
        result = get_longitudinal_view(snaps)

        # self_model: v1, absent, v3_self_model
        sm_entries = result["timelines"]["self_model"]["entries"]
        assert sm_entries[0]["value"] == "v1"
        assert sm_entries[1]["value"] == ABSENT_MARKER
        assert sm_entries[2]["value"] == "v3_self_model"

        # identity_coherence: absent, v2, v3_identity_coherence
        ic_entries = result["timelines"]["identity_coherence"]["entries"]
        assert ic_entries[0]["value"] == ABSENT_MARKER
        assert ic_entries[1]["value"] == "v2"
        assert ic_entries[2]["value"] == "v3_identity_coherence"

    def test_large_number_of_snapshots(self):
        """大量のスナップショット。"""
        snaps = _make_snapshots(100)
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            assert len(result["timelines"][name]["entries"]) == 100

    def test_missing_tick_field(self):
        """tickフィールドが欠落したスナップショット。"""
        snap = {
            "sections": {"self_model": "test"},
            "timestamp": 1000.0,
        }
        result = get_longitudinal_view([snap])
        entries = result["timelines"]["self_model"]["entries"]
        assert entries[0]["tick"] == 0  # default

    def test_missing_timestamp_field(self):
        """timestampフィールドが欠落したスナップショット。"""
        snap = {
            "sections": {"self_model": "test"},
            "tick": 1,
        }
        result = get_longitudinal_view([snap])
        entries = result["timelines"]["self_model"]["entries"]
        assert entries[0]["timestamp"] == 0.0  # default

    def test_missing_sections_field(self):
        """sectionsフィールドが欠落したスナップショット。"""
        snap = {"tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == ABSENT_MARKER

    def test_empty_string_values(self):
        """空文字列の断面値。"""
        sections = {name: "" for name in SECTION_ORDER}
        snap = {"sections": sections, "tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == ""

    def test_very_long_values(self):
        """非常に長い断面値。"""
        long_value = "x" * 10000
        sections = {name: long_value for name in SECTION_ORDER}
        snap = {"sections": sections, "tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == long_value

    def test_enrichment_size_limit_with_long_values(self):
        """長い値でもenrichmentのサイズ上限が守られること。"""
        long_value = "x" * 1000
        sections = {name: long_value for name in SECTION_ORDER}
        snaps = [
            {"sections": dict(sections), "tick": i, "timestamp": float(i)}
            for i in range(10)
        ]
        config = LongitudinalViewConfig(max_enrichment_length=500)
        result = get_enrichment_data(snaps, config)
        assert len(result["summary_text"]) <= 500

    def test_identical_ticks(self):
        """同一tickの複数スナップショット（異常ではないが考慮）。"""
        snaps = [
            _make_snapshot(tick=1, timestamp=100.0),
            _make_snapshot(tick=1, timestamp=101.0),
        ]
        result = get_longitudinal_view(snaps)
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert len(entries) == 2  # 両方保持される


# =============================================================================
# Test: Integration with introspection_cross_section
# =============================================================================

class TestIntegrationWithCrossSection:
    """横断的記述モジュールとの統合テスト。"""

    def test_cross_section_section_order_used(self):
        """横断的記述のSECTION_ORDERがそのまま使用されることを確認。"""
        snaps = _make_snapshots(3)
        result = get_longitudinal_view(snaps)
        assert list(result["timelines"].keys()) == SECTION_ORDER

    def test_cross_section_absent_marker_used(self):
        """横断的記述のABSENT_MARKERがそのまま使用されることを確認。"""
        snap = {"sections": {}, "tick": 1, "timestamp": 1000.0}
        result = get_longitudinal_view([snap])
        for name in SECTION_ORDER:
            assert result["timelines"][name]["entries"][0]["value"] == ABSENT_MARKER

    def test_cross_section_section_labels_used_in_enrichment(self):
        """横断的記述のSECTION_LABELSがenrichmentで使用されることを確認。"""
        snaps = _make_snapshots(2)
        result = get_enrichment_data(snaps)
        text = result["summary_text"]
        for name in SECTION_ORDER:
            label = SECTION_LABELS[name]
            assert label in text

    def test_snapshot_format_compatibility(self):
        """横断的記述のto_dict()出力形式と互換性があることを確認。"""
        from psyche.introspection_cross_section import CrossSectionSnapshot
        snap = CrossSectionSnapshot(
            sections={name: f"v_{name}" for name in SECTION_ORDER},
            tick=5,
            timestamp=1234.5,
        )
        snap_dict = snap.to_dict()
        result = get_longitudinal_view([snap_dict])
        for name in SECTION_ORDER:
            entries = result["timelines"][name]["entries"]
            assert entries[0]["value"] == f"v_{name}"
            assert entries[0]["tick"] == 5
            assert entries[0]["timestamp"] == 1234.5
