"""
tests/test_perceptual_structuring_degree.py - 知覚構造化度断面テスト

design_perceptual_structuring_degree.md に基づくテスト:
- 知覚構造化度の段階値記述テスト
- 段階値分類ヘルパーテスト
- 構造化度がenrichmentに等価に含まれるテスト
- 安全弁テスト（5つ）
- デフォルト状態テスト
- ウィンドウ最小件数テスト
- save/load互換性テスト
- エッジケーステスト
"""

import pytest

from psyche.perceptual_context import (
    StructuringDegree,
    PerceptualSummary,
    PerceptualContextState,
    PerceptualContextConfig,
    PerceptualContextProcessor,
    SECTION_STRUCTURING_DEGREE,
    SECTION_ORDER,
    SECTION_LABELS,
    STRUCTURING_DEGREE_LABELS,
    get_perceptual_context_summary,
    _classify_structuring_degree,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def processor():
    """デフォルト設定のプロセッサを返す。"""
    return PerceptualContextProcessor()


@pytest.fixture
def small_window_processor():
    """小さいウィンドウのプロセッサ。"""
    config = PerceptualContextConfig(max_summaries=5)
    return PerceptualContextProcessor(config=config)


# =============================================================================
# StructuringDegree Enum Tests
# =============================================================================

class TestStructuringDegreeEnum:
    """知覚構造化度の列挙型テスト。"""

    def test_enum_has_5_members(self):
        """5段階であること。"""
        assert len(StructuringDegree) == 5

    def test_enum_values(self):
        """各段階値が正しいこと。"""
        assert StructuringDegree.MANY.value == "many"
        assert StructuringDegree.SOMEWHAT_MANY.value == "somewhat_many"
        assert StructuringDegree.MODERATE.value == "moderate"
        assert StructuringDegree.SOMEWHAT_FEW.value == "somewhat_few"
        assert StructuringDegree.FEW.value == "few"

    def test_labels_exist_for_all_members(self):
        """全段階に日本語ラベルが存在すること。"""
        for member in StructuringDegree:
            assert member in STRUCTURING_DEGREE_LABELS

    def test_labels_content(self):
        """日本語ラベルの内容が正しいこと。"""
        assert STRUCTURING_DEGREE_LABELS[StructuringDegree.MANY] == "多"
        assert STRUCTURING_DEGREE_LABELS[StructuringDegree.SOMEWHAT_MANY] == "やや多"
        assert STRUCTURING_DEGREE_LABELS[StructuringDegree.MODERATE] == "中程度"
        assert STRUCTURING_DEGREE_LABELS[StructuringDegree.SOMEWHAT_FEW] == "やや少"
        assert STRUCTURING_DEGREE_LABELS[StructuringDegree.FEW] == "少"


# =============================================================================
# Section Definition Tests
# =============================================================================

class TestSectionDefinition:
    """断面定義テスト。"""

    def test_section_name_constant(self):
        """断面名定数が存在すること。"""
        assert SECTION_STRUCTURING_DEGREE == "structuring_degree"

    def test_section_in_order(self):
        """SECTION_ORDERに含まれていること。"""
        assert SECTION_STRUCTURING_DEGREE in SECTION_ORDER

    def test_section_order_has_5_sections(self):
        """SECTION_ORDERが5断面であること。"""
        assert len(SECTION_ORDER) == 5

    def test_section_is_last_in_order(self):
        """SECTION_ORDERの末尾にあること（既存4断面の後）。"""
        assert SECTION_ORDER[-1] == SECTION_STRUCTURING_DEGREE

    def test_section_label_exists(self):
        """SECTION_LABELSに日本語ラベルが存在すること。"""
        assert SECTION_STRUCTURING_DEGREE in SECTION_LABELS

    def test_section_label_content(self):
        """日本語ラベルが正しいこと。"""
        assert SECTION_LABELS[SECTION_STRUCTURING_DEGREE] == "知覚構造化度"


# =============================================================================
# _classify_structuring_degree Helper Tests
# =============================================================================

class TestClassifyStructuringDegree:
    """分類ヘルパー関数テスト。"""

    def test_below_min_count_returns_moderate(self):
        """ウィンドウサイズが最小件数未満の場合はMODERATE。"""
        assert _classify_structuring_degree(0.9, 2, min_count=3) == StructuringDegree.MODERATE

    def test_ratio_0_0_returns_few(self):
        """平均比率0.0はFEW。"""
        assert _classify_structuring_degree(0.0, 5, min_count=3) == StructuringDegree.FEW

    def test_ratio_0_05_returns_few(self):
        """平均比率0.05はFEW。"""
        assert _classify_structuring_degree(0.05, 5, min_count=3) == StructuringDegree.FEW

    def test_ratio_0_1_returns_somewhat_few(self):
        """平均比率0.1はSOMEWHAT_FEW。"""
        assert _classify_structuring_degree(0.1, 5, min_count=3) == StructuringDegree.SOMEWHAT_FEW

    def test_ratio_0_15_returns_somewhat_few(self):
        """平均比率0.15はSOMEWHAT_FEW。"""
        assert _classify_structuring_degree(0.15, 5, min_count=3) == StructuringDegree.SOMEWHAT_FEW

    def test_ratio_0_2_returns_moderate(self):
        """平均比率0.2はMODERATE。"""
        assert _classify_structuring_degree(0.2, 5, min_count=3) == StructuringDegree.MODERATE

    def test_ratio_0_35_returns_moderate(self):
        """平均比率0.35はMODERATE。"""
        assert _classify_structuring_degree(0.35, 5, min_count=3) == StructuringDegree.MODERATE

    def test_ratio_0_5_returns_somewhat_many(self):
        """平均比率0.5はSOMEWHAT_MANY。"""
        assert _classify_structuring_degree(0.5, 5, min_count=3) == StructuringDegree.SOMEWHAT_MANY

    def test_ratio_0_65_returns_somewhat_many(self):
        """平均比率0.65はSOMEWHAT_MANY。"""
        assert _classify_structuring_degree(0.65, 5, min_count=3) == StructuringDegree.SOMEWHAT_MANY

    def test_ratio_0_8_returns_many(self):
        """平均比率0.8はMANY。"""
        assert _classify_structuring_degree(0.8, 5, min_count=3) == StructuringDegree.MANY

    def test_ratio_1_0_returns_many(self):
        """平均比率1.0はMANY。"""
        assert _classify_structuring_degree(1.0, 5, min_count=3) == StructuringDegree.MANY

    def test_exact_boundary_0_1(self):
        """境界値0.1は正確にSOMEWHAT_FEW。"""
        assert _classify_structuring_degree(0.1, 10, min_count=3) == StructuringDegree.SOMEWHAT_FEW

    def test_exact_boundary_0_2(self):
        """境界値0.2は正確にMODERATE。"""
        assert _classify_structuring_degree(0.2, 10, min_count=3) == StructuringDegree.MODERATE

    def test_exact_boundary_0_5(self):
        """境界値0.5は正確にSOMEWHAT_MANY。"""
        assert _classify_structuring_degree(0.5, 10, min_count=3) == StructuringDegree.SOMEWHAT_MANY

    def test_exact_boundary_0_8(self):
        """境界値0.8は正確にMANY。"""
        assert _classify_structuring_degree(0.8, 10, min_count=3) == StructuringDegree.MANY


# =============================================================================
# Processor Integration Tests - _describe_structuring_degree
# =============================================================================

class TestDescribeStructuringDegree:
    """プロセッサの知覚構造化度断面記述テスト。"""

    def test_empty_summaries_returns_moderate(self, processor):
        """サマリなしの場合はMODERATE。"""
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_below_min_count_returns_moderate(self, processor):
        """最小件数未満の場合はMODERATE（安全弁4）。"""
        processor.accumulate_summary("happy", "greeting", ["topic1"], 0.5, tick=1)
        processor.accumulate_summary("sad", "question", ["topic2"], -0.3, tick=2)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_all_defaults_returns_few(self, processor):
        """全てデフォルト値の場合はFEW。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.FEW.value

    def test_all_structured_returns_many(self, processor):
        """全て構造化済みの場合はMANY。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MANY.value

    def test_mixed_partially_structured(self, processor):
        """一部のみ構造化の場合は中間の段階値。"""
        # 3/5 emotion non-default, 2/5 intent non-default, 3/5 topics non-empty
        processor.accumulate_summary("happy", "greeting", ["topic1"], 0.5, tick=1)
        processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=2)
        processor.accumulate_summary("sad", "question", ["topic2"], -0.3, tick=3)
        processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=4)
        processor.accumulate_summary("angry", "unknown", ["topic3"], -0.5, tick=5)
        snapshot = processor.describe_features()
        # emotion: 3/5=0.6, intent: 2/5=0.4, topics: 3/5=0.6
        # average = (0.6+0.4+0.6)/3 = 0.533... -> SOMEWHAT_MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.SOMEWHAT_MANY.value

    def test_only_emotion_structured(self, processor):
        """感情ラベルのみ構造化の場合。"""
        for i in range(5):
            processor.accumulate_summary("happy", "unknown", [], 0.0, tick=i)
        snapshot = processor.describe_features()
        # emotion: 5/5=1.0, intent: 0/5=0.0, topics: 0/5=0.0
        # average = 1.0/3 = 0.333... -> MODERATE
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_only_intent_structured(self, processor):
        """意図ラベルのみ構造化の場合。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "greeting", [], 0.0, tick=i)
        snapshot = processor.describe_features()
        # emotion: 0/5=0.0, intent: 5/5=1.0, topics: 0/5=0.0
        # average = 1.0/3 = 0.333... -> MODERATE
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_only_topics_present(self, processor):
        """話題のみ存在の場合。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "unknown", ["topic"], 0.0, tick=i)
        snapshot = processor.describe_features()
        # emotion: 0/5=0.0, intent: 0/5=0.0, topics: 5/5=1.0
        # average = 1.0/3 = 0.333... -> MODERATE
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_two_of_three_structured(self, processor):
        """3つの指標のうち2つが全て構造化の場合。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", [], 0.5, tick=i)
        snapshot = processor.describe_features()
        # emotion: 5/5=1.0, intent: 5/5=1.0, topics: 0/5=0.0
        # average = 2.0/3 = 0.666... -> SOMEWHAT_MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.SOMEWHAT_MANY.value

    def test_snapshot_includes_structuring_degree(self, processor):
        """describe_features結果のスナップショットに含まれること。"""
        for i in range(3):
            processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=i)
        processor.describe_features()
        snapshot = processor.get_snapshot()
        assert SECTION_STRUCTURING_DEGREE in snapshot

    def test_previous_snapshot_tracks_change(self, processor):
        """直前スナップショットが構造化度を追跡すること。"""
        # First: all defaults -> FEW
        for i in range(3):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        processor.describe_features()

        # Second: all structured -> MANY (previous should be FEW)
        for i in range(3, 8):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        prev = processor.get_previous_snapshot()
        assert SECTION_STRUCTURING_DEGREE in prev


# =============================================================================
# Enrichment Tests
# =============================================================================

class TestEnrichmentIntegration:
    """enrichment統合テスト。"""

    def test_enrichment_text_includes_structuring_degree(self, processor):
        """enrichmentテキストに知覚構造化度が含まれること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        assert "知覚構造化度" in text

    def test_enrichment_text_uses_japanese_label(self, processor):
        """enrichmentテキストが日本語ラベルを使用すること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        assert "知覚構造化度=多" in text

    def test_enrichment_text_few_label(self, processor):
        """構造化度FEWの日本語ラベル。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        assert "知覚構造化度=少" in text

    def test_enrichment_text_moderate_label(self, processor):
        """構造化度MODERATEの日本語ラベル。"""
        for i in range(5):
            processor.accumulate_summary("happy", "unknown", [], 0.0, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        assert "知覚構造化度=中程度" in text

    def test_enrichment_data_includes_structuring_degree(self, processor):
        """enrichmentデータのsnapshotに含まれること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        data = processor.get_enrichment_data()
        assert SECTION_STRUCTURING_DEGREE in data["snapshot"]

    def test_enrichment_text_equal_listing(self, processor):
        """全5断面が等価に列挙されること（特別な強調なし）。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        # All 5 sections should be present, separated by spaces
        assert text.count("=") == 5


# =============================================================================
# Safety Valve Tests
# =============================================================================

class TestSafetyValves:
    """安全弁テスト。"""

    def test_sv1_only_degree_value_in_enrichment(self, processor):
        """安全弁1: enrichmentに比率の数値や個々のサマリ内訳を含めない。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        # Should not contain raw ratios or percentages
        assert "0.333" not in text
        assert "0.666" not in text
        assert "%" not in text
        # Should only contain the degree label
        labels = list(STRUCTURING_DEGREE_LABELS.values())
        found = sum(1 for label in labels if label in text)
        assert found >= 1  # At least one label is present

    def test_sv2_no_evaluative_vocabulary(self, processor):
        """安全弁2: 評価的語彙を使わない。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        processor.describe_features()
        text = processor.get_enrichment_text()
        forbidden_words = ["理解", "認識", "把握", "不明", "困難", "品質", "精度"]
        for word in forbidden_words:
            assert word not in text, f"Forbidden word '{word}' found in enrichment text"

    def test_sv3_no_processing_branch(self, processor):
        """安全弁3: 構造化度の値に基づいて他の処理を分岐させない。"""
        # Describe features with FEW structuring degree
        for i in range(5):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        snap_few = processor.describe_features()

        # Reset and describe with MANY structuring degree
        processor2 = PerceptualContextProcessor()
        for i in range(5):
            processor2.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        snap_many = processor2.describe_features()

        # The other 4 sections should have the same number of keys
        # (structuring degree does not affect other sections)
        other_sections = [k for k in snap_few if k != SECTION_STRUCTURING_DEGREE]
        other_sections2 = [k for k in snap_many if k != SECTION_STRUCTURING_DEGREE]
        assert set(other_sections) == set(other_sections2)

    def test_sv4_min_count_default(self, processor):
        """安全弁4: 最小件数未満の場合はデフォルトのMODERATE。"""
        processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=1)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_sv5_all_degrees_equal(self):
        """安全弁5: 全段階等価 -- 段階値に重み・優先度を持たない。"""
        # Just verify no numeric ordering or weight is embedded in the enum
        for member in StructuringDegree:
            assert isinstance(member.value, str)
            # No numeric attribute
            assert not hasattr(member, 'weight')
            assert not hasattr(member, 'priority')
            assert not hasattr(member, 'score')


# =============================================================================
# Save/Load Compatibility Tests
# =============================================================================

class TestSaveLoadCompatibility:
    """save/load互換性テスト。"""

    def test_save_load_preserves_structuring_degree(self, processor):
        """save/loadで構造化度が保存・復元されること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()

        saved = processor.save()
        new_processor = PerceptualContextProcessor()
        new_processor.load(saved)

        assert new_processor.get_snapshot()[SECTION_STRUCTURING_DEGREE] == \
            processor.get_snapshot()[SECTION_STRUCTURING_DEGREE]

    def test_load_without_structuring_degree_key(self, processor):
        """古いデータ（構造化度キーなし）をloadしても動作すること。"""
        # Simulate old save data without structuring_degree in snapshot
        old_data = {
            "summaries": [
                {"emotion": "happy", "intent": "greeting", "topics": ["t"],
                 "emotion_valence": 0.5, "tick": 1},
            ],
            "snapshot": {
                "emotion_change_frequency": "moderate",
                "intent_change_frequency": "moderate",
                "topic_overlap": "moderate",
                "valence_direction": "flat",
                # No structuring_degree key
            },
            "previous_snapshot": {},
        }
        processor.load(old_data)
        # describe_features should recalculate and add it
        snapshot = processor.describe_features()
        assert SECTION_STRUCTURING_DEGREE in snapshot

    def test_roundtrip_save_load(self, processor):
        """完全なラウンドトリップテスト。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        processor.describe_features()

        saved = processor.save()
        restored = PerceptualContextProcessor()
        restored.load(saved)
        restored.describe_features()

        assert restored.get_snapshot() == processor.get_snapshot()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_single_summary_below_min_count(self, processor):
        """1件のみの場合はMODERATE。"""
        processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=1)
        snapshot = processor.describe_features()
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MODERATE.value

    def test_exactly_min_count(self, processor):
        """ちょうど最小件数の場合は実計算が行われること。"""
        processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=1)
        processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=2)
        processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=3)
        snapshot = processor.describe_features()
        # 3/3 emotion, 3/3 intent, 3/3 topics -> avg=1.0 -> MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MANY.value

    def test_large_window(self, processor):
        """大量のサマリでも正常に動作すること。"""
        for i in range(100):
            if i % 2 == 0:
                processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=i)
            else:
                processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        snapshot = processor.describe_features()
        assert SECTION_STRUCTURING_DEGREE in snapshot
        # Value should be a valid StructuringDegree
        StructuringDegree(snapshot[SECTION_STRUCTURING_DEGREE])

    def test_topics_empty_list_counted_as_unstructured(self, processor):
        """空の話題リストは構造化されていないとみなされること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", [], 0.5, tick=i)
        snapshot = processor.describe_features()
        # emotion: 5/5=1.0, intent: 5/5=1.0, topics: 0/5=0.0
        # average = 2.0/3 = 0.666... -> SOMEWHAT_MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.SOMEWHAT_MANY.value

    def test_neutral_emotion_is_default(self, processor):
        """"neutral"はデフォルト感情ラベルとしてカウントされること。"""
        for i in range(5):
            processor.accumulate_summary("neutral", "greeting", ["t"], 0.5, tick=i)
        snapshot = processor.describe_features()
        # emotion: 0/5=0.0, intent: 5/5=1.0, topics: 5/5=1.0
        # average = 2.0/3 = 0.666... -> SOMEWHAT_MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.SOMEWHAT_MANY.value

    def test_unknown_intent_is_default(self, processor):
        """'unknown'はデフォルト意図ラベルとしてカウントされること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "unknown", ["t"], 0.5, tick=i)
        snapshot = processor.describe_features()
        # emotion: 5/5=1.0, intent: 0/5=0.0, topics: 5/5=1.0
        # average = 2.0/3 = 0.666... -> SOMEWHAT_MANY
        assert snapshot[SECTION_STRUCTURING_DEGREE] == StructuringDegree.SOMEWHAT_MANY.value

    def test_no_weights_between_three_ratios(self, processor):
        """3つの比率間に重みがないこと（等価扱い）。"""
        # Case 1: only emotion structured (ratio=0.333)
        proc1 = PerceptualContextProcessor()
        for i in range(6):
            proc1.accumulate_summary("happy", "unknown", [], 0.0, tick=i)
        snap1 = proc1.describe_features()

        # Case 2: only intent structured (ratio=0.333)
        proc2 = PerceptualContextProcessor()
        for i in range(6):
            proc2.accumulate_summary("neutral", "greeting", [], 0.0, tick=i)
        snap2 = proc2.describe_features()

        # Case 3: only topics present (ratio=0.333)
        proc3 = PerceptualContextProcessor()
        for i in range(6):
            proc3.accumulate_summary("neutral", "unknown", ["t"], 0.0, tick=i)
        snap3 = proc3.describe_features()

        # All three should produce the same degree (no weighting)
        assert snap1[SECTION_STRUCTURING_DEGREE] == snap2[SECTION_STRUCTURING_DEGREE]
        assert snap2[SECTION_STRUCTURING_DEGREE] == snap3[SECTION_STRUCTURING_DEGREE]

    def test_describe_features_returns_5_sections(self, processor):
        """describe_featuresが5断面を返すこと。"""
        for i in range(3):
            processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=i)
        snapshot = processor.describe_features()
        assert len(snapshot) == 5

    def test_recalculation_on_each_describe(self, processor):
        """describe_features呼び出しごとに再算出されること（前回値を参照しない）。"""
        # First: all defaults
        for i in range(3):
            processor.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        snap1 = processor.describe_features()
        val1 = snap1[SECTION_STRUCTURING_DEGREE]

        # Add structured data and re-describe
        for i in range(3, 53):
            processor.accumulate_summary("happy", "greeting", ["topic"], 0.5, tick=i)
        snap2 = processor.describe_features()
        val2 = snap2[SECTION_STRUCTURING_DEGREE]

        # The degree should change because window content changed
        assert val1 != val2

    def test_windowing_affects_structuring_degree(self, small_window_processor):
        """ウィンドウ押し出しが構造化度に影響すること。"""
        proc = small_window_processor
        # Fill window with structured data
        for i in range(5):
            proc.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=i)
        snap1 = proc.describe_features()
        assert snap1[SECTION_STRUCTURING_DEGREE] == StructuringDegree.MANY.value

        # Push out with unstructured data
        for i in range(5, 10):
            proc.accumulate_summary("neutral", "unknown", [], 0.0, tick=i)
        snap2 = proc.describe_features()
        assert snap2[SECTION_STRUCTURING_DEGREE] == StructuringDegree.FEW.value


# =============================================================================
# get_perceptual_context_summary Tests for new section
# =============================================================================

class TestPerceptualContextSummary:
    """enrichment要約関数テスト。"""

    def test_summary_includes_all_5_sections(self, processor):
        """要約テキストに5断面全てが含まれること。"""
        for i in range(5):
            processor.accumulate_summary("happy", "greeting", ["t"], 0.5, tick=i)
        processor.describe_features()
        text = get_perceptual_context_summary(processor.state)
        assert "知覚構造化度" in text
        assert "感情ラベル変化頻度" in text
        assert "意図ラベル変化頻度" in text
        assert "話題重複度" in text
        assert "感情価推移方向" in text

    def test_summary_with_empty_state_returns_waiting(self):
        """空状態では待機中が返ること。"""
        state = PerceptualContextState()
        text = get_perceptual_context_summary(state)
        assert text == "知覚推移: 待機中"

    def test_summary_structuring_degree_label_many(self):
        """MANYの場合は'多'と表示されること。"""
        state = PerceptualContextState(
            snapshot={SECTION_STRUCTURING_DEGREE: StructuringDegree.MANY.value}
        )
        text = get_perceptual_context_summary(state)
        assert "知覚構造化度=多" in text

    def test_summary_structuring_degree_label_few(self):
        """FEWの場合は'少'と表示されること。"""
        state = PerceptualContextState(
            snapshot={SECTION_STRUCTURING_DEGREE: StructuringDegree.FEW.value}
        )
        text = get_perceptual_context_summary(state)
        assert "知覚構造化度=少" in text

    def test_summary_structuring_degree_label_somewhat_many(self):
        """SOMEWHAT_MANYの場合は'やや多'と表示されること。"""
        state = PerceptualContextState(
            snapshot={SECTION_STRUCTURING_DEGREE: StructuringDegree.SOMEWHAT_MANY.value}
        )
        text = get_perceptual_context_summary(state)
        assert "知覚構造化度=やや多" in text

    def test_summary_structuring_degree_label_somewhat_few(self):
        """SOMEWHAT_FEWの場合は'やや少'と表示されること。"""
        state = PerceptualContextState(
            snapshot={SECTION_STRUCTURING_DEGREE: StructuringDegree.SOMEWHAT_FEW.value}
        )
        text = get_perceptual_context_summary(state)
        assert "知覚構造化度=やや少" in text

    def test_summary_structuring_degree_label_moderate(self):
        """MODERATEの場合は'中程度'と表示されること。"""
        state = PerceptualContextState(
            snapshot={SECTION_STRUCTURING_DEGREE: StructuringDegree.MODERATE.value}
        )
        text = get_perceptual_context_summary(state)
        assert "知覚構造化度=中程度" in text
