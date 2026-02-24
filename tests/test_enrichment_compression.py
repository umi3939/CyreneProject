"""
tests/test_enrichment_compression.py - enrichment圧縮ヘルパーのテスト

テスト対象: psyche/enrichment_compression.py
設計書: design_enrichment_compression.md

テスト項目:
- 第1段: 項目別変動度の算出（二値判定）
- 第2段: 記述粒度の選択（全文記述 / 短縮形記述）
- 第3段: フォーマット圧縮（セクションヘッダ簡潔化、空セクション短縮、フッター簡潔化）
- 統合: build_compressed_enrichment パイプライン
- 安全弁1: キャッシュ不在時のフォールバック
- 安全弁2: 圧縮率の下限監視
- 安全弁3: 短縮形の固定性
- 安全弁4: セクション消失の防止
- 安全弁5: 個別項目の圧縮無効化経路（除外リスト）
- フィードバック経路遮断の検証
- エッジケース
"""

import logging
import pytest

from psyche.enrichment_compression import (
    ALWAYS_FULL_LABELS,
    SECTION_HEADER_MAP,
    ORIGINAL_FOOTER,
    COMPRESSED_FOOTER,
    STABLE_SUFFIX,
    COMPRESSION_RATIO_WARNING_THRESHOLD,
    detect_item_changed,
    apply_item_granularity,
    compress_section,
    compress_footer,
    compute_compression_ratio,
    log_compression_ratio,
    build_compressed_enrichment,
)


# =============================================================================
# 第1段: detect_item_changed テスト
# =============================================================================

class TestDetectItemChanged:
    """第1段: 項目別変動度の算出（二値判定）。"""

    def test_changed_when_no_cache(self):
        """安全弁1: キャッシュ不在時は変動あり扱い。"""
        assert detect_item_changed("テスト項目", "some text", {}) is True

    def test_changed_when_label_not_in_cache(self):
        """前回キャッシュにラベルが存在しない場合は変動あり。"""
        cache = {"別ラベル": "old text"}
        assert detect_item_changed("テスト項目", "some text", cache) is True

    def test_no_change_when_identical(self):
        """テキストが完全に同一の場合は変動なし。"""
        text = "感情連動: coupling=0.5"
        cache = {"感情連動": text}
        assert detect_item_changed("感情連動", text, cache) is False

    def test_changed_when_text_differs(self):
        """テキスト内容が異なる場合は変動あり。"""
        cache = {"テスト": "old text"}
        assert detect_item_changed("テスト", "new text", cache) is True

    def test_changed_when_length_differs(self):
        """テキスト長が異なる場合は変動あり。"""
        cache = {"テスト": "abc"}
        assert detect_item_changed("テスト", "abcd", cache) is True

    def test_changed_same_length_different_content(self):
        """同じ長さでも内容が違えば変動あり。"""
        cache = {"テスト": "abc"}
        assert detect_item_changed("テスト", "xyz", cache) is True

    # -- 安全弁5: 除外リストのテスト --

    def test_always_full_for_emotion(self):
        """安全弁5: 「感情」は除外リストに含まれ常に変動ありと判定。"""
        text = "same text"
        cache = {"感情": text}
        assert detect_item_changed("感情", text, cache) is True

    def test_always_full_for_mood(self):
        """安全弁5: 「ムード」は除外リストに含まれ常に変動ありと判定。"""
        text = "same text"
        cache = {"ムード": text}
        assert detect_item_changed("ムード", text, cache) is True

    def test_always_full_for_drive(self):
        """安全弁5: 「ドライブ」は除外リストに含まれ常に変動ありと判定。"""
        text = "same text"
        cache = {"ドライブ": text}
        assert detect_item_changed("ドライブ", text, cache) is True

    def test_always_full_for_dominant_emotion(self):
        """安全弁5: 「支配的感情」は除外リストに含まれ常に変動ありと判定。"""
        text = "same text"
        cache = {"支配的感情": text}
        assert detect_item_changed("支配的感情", text, cache) is True

    def test_always_full_labels_is_frozenset(self):
        """安全弁5: 除外リストはfrozensetで実行時変更不可。"""
        assert isinstance(ALWAYS_FULL_LABELS, frozenset)

    def test_always_full_labels_contents(self):
        """除外リストに必要な項目が含まれている。"""
        expected = {"感情", "ムード", "ドライブ", "支配的感情"}
        assert ALWAYS_FULL_LABELS == expected


# =============================================================================
# 第2段: apply_item_granularity テスト
# =============================================================================

class TestApplyItemGranularity:
    """第2段: 記述粒度の選択。"""

    def test_full_text_when_changed(self):
        """変動ありの場合は全文記述。"""
        text = "感情連動: coupling=0.5, intensity=high"
        result = apply_item_granularity("感情連動", text, changed=True)
        assert result == text

    def test_shortened_when_not_changed(self):
        """変動なしの場合は短縮形記述。"""
        result = apply_item_granularity("感情連動", "some long text", changed=False)
        assert result == f"感情連動: {STABLE_SUFFIX}"

    def test_stable_suffix_is_fixed(self):
        """安全弁3: 短縮形は固定文字列。"""
        assert STABLE_SUFFIX == "(安定)"

    def test_shortened_format(self):
        """短縮形の形式は「ラベル: (安定)」。"""
        result = apply_item_granularity("テスト", "x", changed=False)
        assert result == "テスト: (安定)"

    def test_full_text_preserves_exact_content(self):
        """全文記述はモジュール出力テキストをそのまま使用。"""
        text = "多経路想起: 3経路\n  [emotion] summary1\n  [context] summary2"
        result = apply_item_granularity("多経路想起", text, changed=True)
        assert result == text


# =============================================================================
# 第3段: compress_section テスト
# =============================================================================

class TestCompressSection:
    """第3段: セクション単位のフォーマット圧縮。"""

    def test_header_simplification(self):
        """セクションヘッダが簡潔化される。"""
        lines = ["感情: joy=0.8"]
        result = compress_section("【心理状態（内面）】", lines)
        assert result.startswith("[内面]")

    def test_all_headers_mapped(self):
        """全5セクションのヘッダが簡潔化マッピングに存在する。"""
        expected_keys = {
            "【心理状態（内面）】",
            "【自己認識】",
            "【動機・目標】",
            "【記憶・内省】",
            "【判断傾向】",
        }
        assert set(SECTION_HEADER_MAP.keys()) == expected_keys

    def test_all_stable_section_compressed(self):
        """全項目が短縮形の場合、セクション全体が1行に圧縮。"""
        lines = [
            f"感情連動: {STABLE_SUFFIX}",
            f"責任: {STABLE_SUFFIX}",
            f"安定弁: {STABLE_SUFFIX}",
        ]
        result = compress_section("【心理状態（内面）】", lines)
        assert result == f"[内面] {STABLE_SUFFIX}"

    def test_mixed_section_keeps_all_lines(self):
        """変動ありの項目が1つでもあればセクションは全行保持。"""
        lines = [
            "感情: joy=0.8, surprise=0.5",
            f"感情連動: {STABLE_SUFFIX}",
        ]
        result = compress_section("【心理状態（内面）】", lines)
        assert "[内面]" in result
        assert "感情: joy=0.8" in result
        assert f"感情連動: {STABLE_SUFFIX}" in result

    def test_empty_section_not_lost(self):
        """安全弁4: 空のセクションでもセクション名は残る。"""
        result = compress_section("【心理状態（内面）】", [])
        assert "[内面]" in result
        assert STABLE_SUFFIX in result

    def test_unknown_header_passes_through(self):
        """マッピングにないヘッダはそのまま使用。"""
        lines = ["test line"]
        result = compress_section("【未知セクション】", lines)
        assert result.startswith("【未知セクション】")

    def test_section_with_single_non_stable_item(self):
        """安定でない項目が1つだけのセクション。"""
        lines = ["感情: joy=0.9"]
        result = compress_section("【心理状態（内面）】", lines)
        assert "[内面]" in result
        assert "感情: joy=0.9" in result


# =============================================================================
# フッター圧縮テスト
# =============================================================================

class TestCompressFooter:
    """フッターの簡潔化。"""

    def test_original_footer_compressed(self):
        """元のフッターが簡潔化される。"""
        result = compress_footer(ORIGINAL_FOOTER)
        assert result == COMPRESSED_FOOTER

    def test_unknown_footer_passes_through(self):
        """不明なフッターはそのまま返す。"""
        custom = "custom footer text"
        result = compress_footer(custom)
        assert result == custom

    def test_compressed_footer_is_shorter(self):
        """圧縮後のフッターは元より短い。"""
        assert len(COMPRESSED_FOOTER) < len(ORIGINAL_FOOTER)


# =============================================================================
# 圧縮率テスト
# =============================================================================

class TestComputeCompressionRatio:
    """圧縮率算出。"""

    def test_identical_text_ratio_1(self):
        """同一テキストなら圧縮率1.0。"""
        text = "some text"
        assert compute_compression_ratio(text, text) == 1.0

    def test_shorter_compressed_text(self):
        """圧縮後が短ければ圧縮率は1未満。"""
        original = "a" * 100
        compressed = "a" * 50
        ratio = compute_compression_ratio(original, compressed)
        assert ratio == pytest.approx(0.5)

    def test_empty_original_returns_1(self):
        """空の元テキストは1.0を返す（ゼロ除算回避）。"""
        assert compute_compression_ratio("", "anything") == 1.0

    def test_both_empty_returns_1(self):
        """両方空でも1.0。"""
        assert compute_compression_ratio("", "") == 1.0

    def test_longer_compressed_above_1(self):
        """圧縮後の方が長ければ1.0を超える。"""
        ratio = compute_compression_ratio("ab", "abcdef")
        assert ratio > 1.0


# =============================================================================
# ログ出力テスト
# =============================================================================

class TestLogCompressionRatio:
    """安全弁2: 圧縮率の下限監視。"""

    def test_warning_on_low_ratio(self, caplog):
        """極端に低い圧縮率で警告ログが出力される。"""
        with caplog.at_level(logging.WARNING, logger="psyche.enrichment_compression"):
            log_compression_ratio(0.1)
        assert "below threshold" in caplog.text

    def test_debug_on_normal_ratio(self, caplog):
        """通常の圧縮率ではデバッグログ。"""
        with caplog.at_level(logging.DEBUG, logger="psyche.enrichment_compression"):
            log_compression_ratio(0.8)
        assert "0.80" in caplog.text

    def test_threshold_value(self):
        """閾値が設定されている。"""
        assert COMPRESSION_RATIO_WARNING_THRESHOLD == 0.3


# =============================================================================
# build_compressed_enrichment 統合テスト
# =============================================================================

class TestBuildCompressedEnrichment:
    """enrichmentテキスト全体の圧縮パイプライン。"""

    def _make_sections(self, items_per_section=None):
        """テスト用のsections_dataを作成。"""
        if items_per_section is None:
            items_per_section = {
                "【心理状態（内面）】": [
                    ("感情", "感情: joy=0.8, surprise=0.3"),
                    ("ムード", "ムード: valence=0.5, arousal=0.6"),
                    ("感情連動", "感情連動: coupling=0.5"),
                ],
                "【自己認識】": [
                    ("自己像", "自己像: stability=0.7"),
                    ("一貫性", "一貫性: level=high"),
                ],
            }
        sections_data = []
        for header, items in items_per_section.items():
            sections_data.append({"header": header, "items": items})
        return sections_data

    def test_first_tick_all_full_text(self):
        """初回（キャッシュ空）は全項目が全文記述。"""
        sections = self._make_sections()
        text, cache, ratio = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 全項目のテキストがそのまま含まれる
        assert "感情: joy=0.8" in text
        assert "ムード: valence=0.5" in text
        assert "感情連動: coupling=0.5" in text
        assert "自己像: stability=0.7" in text
        # キャッシュに全項目が保存される
        assert "感情" in cache
        assert "ムード" in cache
        assert "感情連動" in cache
        assert "自己像" in cache
        assert "一貫性" in cache

    def test_second_tick_unchanged_items_shortened(self):
        """2回目で変化なしの項目が短縮形になる。"""
        sections = self._make_sections()
        # 1回目: キャッシュ構築
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 2回目: 同じデータ
        text2, cache2, ratio2 = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        # 感情・ムード（除外リスト）は全文記述を維持
        assert "感情: joy=0.8" in text2
        assert "ムード: valence=0.5" in text2
        # 感情連動（除外リスト外）は短縮形
        assert f"感情連動: {STABLE_SUFFIX}" in text2
        # 自己認識セクションは全項目安定 → セクション全体が圧縮される
        assert f"[自己] {STABLE_SUFFIX}" in text2

    def test_changed_item_reverts_to_full(self):
        """変化した項目は全文記述に復帰。"""
        sections1 = self._make_sections()
        _, cache1, _ = build_compressed_enrichment(sections1, {}, ORIGINAL_FOOTER)
        # 2回目: 感情連動を変更
        sections2 = self._make_sections({
            "【心理状態（内面）】": [
                ("感情", "感情: joy=0.8, surprise=0.3"),
                ("ムード", "ムード: valence=0.5, arousal=0.6"),
                ("感情連動", "感情連動: coupling=0.9"),  # changed
            ],
            "【自己認識】": [
                ("自己像", "自己像: stability=0.7"),
                ("一貫性", "一貫性: level=high"),
            ],
        })
        text2, _, _ = build_compressed_enrichment(sections2, cache1, ORIGINAL_FOOTER)
        # 感情連動は変化したので全文記述
        assert "感情連動: coupling=0.9" in text2
        assert f"感情連動: {STABLE_SUFFIX}" not in text2

    def test_compression_ratio_decreases_on_second_tick(self):
        """2回目のほうが圧縮率が低い（より多くが短縮形になるため）。"""
        sections = self._make_sections()
        _, cache1, ratio1 = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        _, _, ratio2 = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        # 初回はフッター圧縮のみ、2回目は項目短縮もあるので ratio < ratio1
        assert ratio2 < ratio1

    def test_footer_compressed(self):
        """フッターが簡潔化される。"""
        sections = self._make_sections()
        text, _, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert COMPRESSED_FOOTER in text
        assert ORIGINAL_FOOTER not in text

    def test_empty_sections_skipped(self):
        """項目なしのセクションはスキップされる。"""
        sections = [
            {"header": "【心理状態（内面）】", "items": []},
            {"header": "【自己認識】", "items": [("自己像", "自己像: test")]},
        ]
        text, _, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert "[内面]" not in text
        assert "自己像: test" in text

    def test_all_stable_section_collapses(self):
        """全項目が安定状態のセクションは1行に圧縮。"""
        sections = self._make_sections({
            "【自己認識】": [
                ("自己像", "自己像: stability=0.7"),
                ("一貫性", "一貫性: level=high"),
            ],
        })
        # 1回目でキャッシュ構築
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 2回目: 同じデータ → 全項目安定
        text2, _, _ = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        assert f"[自己] {STABLE_SUFFIX}" in text2

    def test_cache_is_new_dict(self):
        """キャッシュは毎回新しい辞書。"""
        sections = self._make_sections()
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        _, cache2, _ = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        assert cache1 is not cache2

    def test_cache_contains_original_text_not_shortened(self):
        """キャッシュにはモジュール出力の元テキストが保存される（短縮形ではない）。"""
        sections = self._make_sections()
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 2回目
        _, cache2, _ = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        # キャッシュは元テキスト
        assert cache2["感情連動"] == "感情連動: coupling=0.5"
        assert STABLE_SUFFIX not in cache2["感情連動"]

    def test_header_mapping_applied(self):
        """セクションヘッダの簡潔化マッピングが適用される。"""
        sections = self._make_sections({
            "【心理状態（内面）】": [("感情", "感情: test")],
            "【自己認識】": [("自己像", "自己像: test")],
            "【動機・目標】": [("動機", "動機: test")],
            "【記憶・内省】": [("エピソード記憶", "エピソード記憶: test")],
            "【判断傾向】": [("判断バイアス", "判断バイアス: test")],
        })
        text, _, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert "[内面]" in text
        assert "[自己]" in text
        assert "[動機]" in text
        assert "[記憶]" in text
        assert "[判断]" in text
        # 元のヘッダは使われていない
        assert "【心理状態（内面）】" not in text


# =============================================================================
# 安全弁の検証テスト
# =============================================================================

class TestSafetyValves:
    """安全弁の動作確認。"""

    def test_sv1_cache_clear_reverts_to_full(self):
        """安全弁1: キャッシュをクリアすると全項目が全文記述に復帰。"""
        sections = [{"header": "【自己認識】", "items": [
            ("自己像", "自己像: stability=0.7"),
        ]}]
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # キャッシュありで短縮形（セクション全体が安定→圧縮される）
        text2, _, _ = build_compressed_enrichment(sections, cache1, ORIGINAL_FOOTER)
        assert STABLE_SUFFIX in text2
        # キャッシュクリア → 全文復帰
        text3, _, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert "自己像: stability=0.7" in text3

    def test_sv2_low_ratio_logs_warning(self, caplog):
        """安全弁2: 極端に低い圧縮率で警告が出るが処理は継続。"""
        with caplog.at_level(logging.WARNING, logger="psyche.enrichment_compression"):
            log_compression_ratio(0.1)
        assert "below threshold" in caplog.text

    def test_sv3_stable_suffix_is_constant(self):
        """安全弁3: 短縮形は固定文字列。"""
        # 複数回呼んでも同じ
        r1 = apply_item_granularity("test", "text", changed=False)
        r2 = apply_item_granularity("test", "different text", changed=False)
        assert r1 == r2 == f"test: {STABLE_SUFFIX}"

    def test_sv4_all_stable_section_still_visible(self):
        """安全弁4: 全項目が安定でもセクション名は消失しない。"""
        result = compress_section("【心理状態（内面）】", [
            f"test1: {STABLE_SUFFIX}",
            f"test2: {STABLE_SUFFIX}",
        ])
        assert "[内面]" in result

    def test_sv5_always_full_labels_cannot_be_modified(self):
        """安全弁5: frozensetなので追加・削除不可。"""
        with pytest.raises(AttributeError):
            ALWAYS_FULL_LABELS.add("new_label")
        with pytest.raises(AttributeError):
            ALWAYS_FULL_LABELS.discard("感情")


# =============================================================================
# フィードバック経路遮断の検証
# =============================================================================

class TestFeedbackIsolation:
    """圧縮結果がモジュール内部状態にフィードバックしないことを検証。"""

    def test_return_types(self):
        """build_compressed_enrichmentの戻り値は(str, dict, float)のみ。"""
        sections = [{"header": "【自己認識】", "items": [
            ("自己像", "自己像: test"),
        ]}]
        result = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert isinstance(result, tuple)
        assert len(result) == 3
        text, cache, ratio = result
        assert isinstance(text, str)
        assert isinstance(cache, dict)
        assert isinstance(ratio, float)

    def test_no_side_effects_on_prev_cache(self):
        """prev_cacheが変更されないこと。"""
        prev_cache = {"自己像": "old text"}
        prev_cache_copy = dict(prev_cache)
        sections = [{"header": "【自己認識】", "items": [
            ("自己像", "自己像: new text"),
        ]}]
        build_compressed_enrichment(sections, prev_cache, ORIGINAL_FOOTER)
        assert prev_cache == prev_cache_copy

    def test_compression_ratio_not_in_output_text(self):
        """圧縮率がenrichmentテキストに含まれない。"""
        sections = [{"header": "【自己認識】", "items": [
            ("自己像", "自己像: test"),
        ]}]
        text, _, ratio = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert str(ratio) not in text
        assert "圧縮" not in text
        assert "compression" not in text.lower()


# =============================================================================
# エッジケース
# =============================================================================

class TestEdgeCases:
    """境界条件のテスト。"""

    def test_empty_sections_list(self):
        """セクションリストが空。"""
        text, cache, ratio = build_compressed_enrichment([], {}, ORIGINAL_FOOTER)
        assert isinstance(text, str)
        assert cache == {}
        assert COMPRESSED_FOOTER in text

    def test_empty_text_item(self):
        """空文字列のテキスト項目。"""
        sections = [{"header": "【自己認識】", "items": [
            ("空ラベル", ""),
        ]}]
        text, cache, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert "空ラベル" in cache

    def test_very_long_text_item(self):
        """非常に長いテキスト項目。"""
        long_text = "x" * 10000
        sections = [{"header": "【自己認識】", "items": [
            ("長テキスト", long_text),
        ]}]
        text, cache, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert long_text in text
        assert cache["長テキスト"] == long_text

    def test_multiline_text_item(self):
        """改行を含むテキスト項目。"""
        multi = "line1\nline2\nline3"
        sections = [{"header": "【自己認識】", "items": [
            ("複数行", multi),
        ]}]
        text, cache, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert multi in text
        assert cache["複数行"] == multi

    def test_unicode_text(self):
        """Unicode文字を含むテキスト。"""
        unicode_text = "感情: 喜び=0.8, 驚き=0.3"
        sections = [{"header": "【心理状態（内面）】", "items": [
            ("感情", unicode_text),
        ]}]
        text, cache, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        assert unicode_text in text

    def test_duplicate_labels_across_sections(self):
        """異なるセクション間で同じラベルが使用された場合。"""
        sections = [
            {"header": "【自己認識】", "items": [("テスト", "テスト: A")]},
            {"header": "【判断傾向】", "items": [("テスト", "テスト: B")]},
        ]
        text, cache, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 後のセクションの値でキャッシュが上書きされる
        assert cache["テスト"] == "テスト: B"

    def test_custom_footer(self):
        """カスタムフッターはそのまま通過。"""
        custom_footer = "custom instruction"
        sections = [{"header": "【自己認識】", "items": [
            ("テスト", "テスト: value"),
        ]}]
        text, _, _ = build_compressed_enrichment(sections, {}, custom_footer)
        assert custom_footer in text

    def test_three_ticks_cache_evolution(self):
        """3ティック分のキャッシュ進化。"""
        sections_t1 = [{"header": "【自己認識】", "items": [
            ("A", "A: v1"), ("B", "B: v1"),
        ]}]
        sections_t2 = [{"header": "【自己認識】", "items": [
            ("A", "A: v2"), ("B", "B: v1"),  # Aのみ変化
        ]}]
        sections_t3 = [{"header": "【自己認識】", "items": [
            ("A", "A: v2"), ("B", "B: v1"),  # 変化なし
        ]}]

        # tick 1: 全文
        text1, cache1, _ = build_compressed_enrichment(sections_t1, {}, ORIGINAL_FOOTER)
        assert "A: v1" in text1
        assert "B: v1" in text1

        # tick 2: Aは変化（全文）、Bは安定（短縮形）
        text2, cache2, _ = build_compressed_enrichment(sections_t2, cache1, ORIGINAL_FOOTER)
        assert "A: v2" in text2
        assert f"B: {STABLE_SUFFIX}" in text2

        # tick 3: A,B両方安定（短縮形）→セクション全体圧縮
        text3, cache3, _ = build_compressed_enrichment(sections_t3, cache2, ORIGINAL_FOOTER)
        assert f"[自己] {STABLE_SUFFIX}" in text3


# =============================================================================
# save/load永続化対象外の検証
# =============================================================================

class TestNoPersistence:
    """設計書の要件: save/loadの永続化対象外。"""

    def test_no_save_function(self):
        """enrichment_compressionモジュールにsave関数がない。"""
        import psyche.enrichment_compression as mod
        assert not hasattr(mod, "save_state")
        assert not hasattr(mod, "save_compression_state")

    def test_no_load_function(self):
        """enrichment_compressionモジュールにload関数がない。"""
        import psyche.enrichment_compression as mod
        assert not hasattr(mod, "load_state")
        assert not hasattr(mod, "load_compression_state")
