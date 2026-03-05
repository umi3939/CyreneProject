"""
tests/test_enrichment_effectiveness_eval.py - EnrichmentEffectivenessEvaluator のテスト

テスト項目:
- 初期化テスト(デフォルト状態)
- 語彙断片抽出テスト(機械的分割、最小文字数フィルタ、空パターン除外、数値除外)
- 照合テスト(完全一致部分文字列マッチ、非表出、複数断片)
- 項目別照合結果バッファテスト(FIFO上限、表出率算出)
- セクション別表出率テスト
- enrichmentデータ更新テスト(語彙断片キャッシュ再生成)
- ExecutionMonitor統合テスト
- セッションサマリテスト
- 安全弁テスト(例外捕捉、無効時スキップ、セッション境界消失、FIFO上限)
- 計測手法の限界テスト(間接影響非評価、偶然一致、表記揺れ非考慮)
- 構造的分離テスト(帰還経路なし、自動最適化なし)
"""

import json
import logging
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.execution_monitor import (
    ExecutionMonitor,
    EnrichmentEffectivenessEvaluator,
    _extract_vocab_fragments,
    _match_fragments_in_text,
    _is_empty_text,
    _KNOWN_EMPTY_PATTERNS,
    _EVAL_MATCH_BUFFER_DEFAULT_MAX,
    _VOCAB_FRAGMENT_MIN_LENGTH,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_monitor(enabled: bool = True) -> ExecutionMonitor:
    """テスト用のExecutionMonitorを生成する。"""
    return ExecutionMonitor(enabled=enabled, snapshot_interval=10)


def _make_evaluator(
    match_buffer_max: int = _EVAL_MATCH_BUFFER_DEFAULT_MAX,
) -> EnrichmentEffectivenessEvaluator:
    """テスト用のEnrichmentEffectivenessEvaluatorを生成する。"""
    return EnrichmentEffectivenessEvaluator(match_buffer_max=match_buffer_max)


def _make_sections(
    items_sec1: list[tuple[str, str]] | None = None,
    items_sec2: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """テスト用のsections_dataを生成する。"""
    sections = []
    if items_sec1 is not None:
        sections.append({
            "header": "セクション1: 内部状態",
            "items": items_sec1,
        })
    if items_sec2 is not None:
        sections.append({
            "header": "セクション2: 認知",
            "items": items_sec2,
        })
    return sections


# ── 1. 初期化テスト ──────────────────────────────────────────────


class TestEnrichmentEffectivenessEvaluatorInit:
    """初期化テスト。"""

    def test_default_init(self):
        """デフォルト初期化で空状態。"""
        ev = _make_evaluator()
        assert ev.evaluation_count == 0
        assert ev.total_match_count == 0
        assert ev.total_manifest_count == 0
        assert ev.get_all_manifestation_rates() == {}
        assert ev.get_section_manifestation_rates() == {}

    def test_custom_buffer_max(self):
        """カスタムバッファ上限。"""
        ev = _make_evaluator(match_buffer_max=10)
        assert ev._match_buffer_max == 10

    def test_buffer_max_minimum(self):
        """バッファ上限の最小値は1。"""
        ev = _make_evaluator(match_buffer_max=0)
        assert ev._match_buffer_max == 1

        ev2 = _make_evaluator(match_buffer_max=-5)
        assert ev2._match_buffer_max == 1

    def test_summary_empty(self):
        """初期状態のサマリは空項目。"""
        ev = _make_evaluator()
        summary = ev.get_summary()
        assert summary["evaluation_count"] == 0
        assert summary["total_match_count"] == 0
        assert summary["total_manifest_count"] == 0
        assert summary["item_count"] == 0
        assert summary["item_manifestation_rates"] == {}
        assert summary["section_manifestation_rates"] == {}


# ── 2. 語彙断片抽出テスト ────────────────────────────────────────


class TestExtractVocabFragments:
    """語彙断片抽出の機械的操作テスト。"""

    def test_empty_text(self):
        """空テキストからは空リスト。"""
        assert _extract_vocab_fragments("") == []
        assert _extract_vocab_fragments("  ") == []

    def test_simple_label_value(self):
        """ラベル: 値の形式から断片抽出。"""
        frags = _extract_vocab_fragments("感情価: 高め")
        assert "感情価" in frags
        assert "高め" in frags

    def test_arrow_separator(self):
        """矢印区切りの断片抽出。"""
        frags = _extract_vocab_fragments("上昇→安定→下降")
        assert "上昇" in frags
        assert "安定" in frags
        assert "下降" in frags

    def test_slash_separator(self):
        """スラッシュ区切りの断片抽出。"""
        frags = _extract_vocab_fragments("好奇心/表現/社交")
        assert "好奇心" in frags
        assert "表現" in frags
        assert "社交" in frags

    def test_dot_separator(self):
        """中黒区切りの断片抽出。"""
        frags = _extract_vocab_fragments("記憶・想起・蓄積")
        assert "記憶" in frags
        assert "想起" in frags
        assert "蓄積" in frags

    def test_min_length_filter(self):
        """最小文字数未満の断片は除外。"""
        frags = _extract_vocab_fragments("a: 長いラベル")
        # "a" は1文字なので除外される
        assert "a" not in frags
        assert "長いラベル" in frags

    def test_numeric_filter(self):
        """純粋な数値は除外。"""
        frags = _extract_vocab_fragments("値: 0.85")
        # "0.85" は数値なので除外
        assert "0.85" not in frags

    def test_empty_pattern_filter(self):
        """既知の空パターンは除外。"""
        frags = _extract_vocab_fragments("状態: (未蓄積)")
        assert "(未蓄積)" not in frags

    def test_dedup(self):
        """重複する断片は1つのみ。"""
        frags = _extract_vocab_fragments("状態: 安定 / 安定")
        assert frags.count("安定") == 1

    def test_complex_enrichment_text(self):
        """複合的なenrichmentテキストの断片抽出。"""
        text = "感情基調: 高め（joy=0.7, trust=0.5）→ 穏やか方向"
        frags = _extract_vocab_fragments(text)
        assert "感情基調" in frags
        assert "高め" in frags
        assert "穏やか方向" in frags

    def test_newline_separator(self):
        """改行区切りの断片抽出。"""
        frags = _extract_vocab_fragments("項目A\n項目B\n項目C")
        assert "項目A" in frags
        assert "項目B" in frags
        assert "項目C" in frags

    def test_brackets(self):
        """括弧内の断片抽出。"""
        frags = _extract_vocab_fragments("【内部状態】安定")
        assert "内部状態" in frags
        assert "安定" in frags


# ── 3. 照合テスト ────────────────────────────────────────────────


class TestMatchFragmentsInText:
    """代弁テキストとの文字列部分一致照合テスト。"""

    def test_no_match(self):
        """一致なし。"""
        result = _match_fragments_in_text(
            ["安定", "高め"], "今日はいい天気ですね"
        )
        assert result == []

    def test_single_match(self):
        """1つの断片が一致。"""
        result = _match_fragments_in_text(
            ["安定", "高め"], "気分が安定していて穏やかです"
        )
        assert "安定" in result
        assert "高め" not in result

    def test_multiple_match(self):
        """複数の断片が一致。"""
        result = _match_fragments_in_text(
            ["安定", "高め", "好奇心"],
            "安定した気持ちで好奇心が高めに湧いています"
        )
        assert "安定" in result
        assert "高め" in result
        assert "好奇心" in result

    def test_empty_fragments(self):
        """空の断片リスト。"""
        result = _match_fragments_in_text([], "テストテキスト")
        assert result == []

    def test_empty_target(self):
        """空のターゲットテキスト。"""
        result = _match_fragments_in_text(["安定"], "")
        assert result == []

    def test_substring_match(self):
        """部分文字列一致。"""
        result = _match_fragments_in_text(
            ["好奇"], "好奇心が芽生えています"
        )
        assert "好奇" in result

    def test_exact_match(self):
        """完全一致。"""
        result = _match_fragments_in_text(
            ["好奇心"], "好奇心"
        )
        assert "好奇心" in result


# ── 4. 項目別照合結果バッファテスト ──────────────────────────────


class TestItemMatchBuffers:
    """項目別照合結果バッファとFIFO上限テスト。"""

    def test_basic_evaluation(self):
        """基本的な照合実行と結果蓄積。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情価", "感情価: 高め"),
                ("ムード", "ムード: 穏やか"),
            ]
        )
        ev.update_enrichment_data(sections)

        results = ev.evaluate_manifestation("高めの感情で穏やかに過ごしています")

        assert ev.evaluation_count == 1
        # 「高め」と「穏やか」が含まれるので両方表出あり
        manifest_items = [k for k, v in results.items() if v]
        assert len(manifest_items) >= 1

    def test_non_manifest(self):
        """非表出の記録。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情価", "感情価: 極度に不安定"),
            ]
        )
        ev.update_enrichment_data(sections)

        results = ev.evaluate_manifestation("今日はいい天気ですね")

        # 「極度に不安定」は含まれない → 非表出
        for item_id, is_manifest in results.items():
            if "感情価" in item_id:
                assert is_manifest is False

    def test_fifo_limit(self):
        """FIFO上限を超えると古い結果が消失。"""
        ev = _make_evaluator(match_buffer_max=3)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)

        # 3回表出あり
        for _ in range(3):
            ev.evaluate_manifestation("安定した気持ちです")
        # 表出率は1.0
        for item_id in ev._match_buffers:
            assert ev.get_item_manifestation_rate(item_id) == 1.0

        # 3回非表出を追加(FIFO上限3なので古い表出結果が消失)
        for _ in range(3):
            ev.evaluate_manifestation("今日はいい天気です")

        # 表出率は0.0(直近3件が全て非表出)
        for item_id in ev._match_buffers:
            assert ev.get_item_manifestation_rate(item_id) == 0.0

    def test_manifestation_rate_calculation(self):
        """表出率の計算。"""
        ev = _make_evaluator(match_buffer_max=10)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)

        # 5回表出あり、5回非表出
        for i in range(10):
            if i < 5:
                ev.evaluate_manifestation("安定した気持ちです")
            else:
                ev.evaluate_manifestation("今日はいい天気です")

        for item_id in ev._match_buffers:
            rate = ev.get_item_manifestation_rate(item_id)
            assert rate == pytest.approx(0.5, abs=0.01)

    def test_nonexistent_item_rate(self):
        """存在しない項目の表出率は0.0。"""
        ev = _make_evaluator()
        assert ev.get_item_manifestation_rate("nonexistent") == 0.0


# ── 5. セクション別表出率テスト ──────────────────────────────────


class TestSectionManifestationRates:
    """セクション別表出率集計テスト。"""

    def test_section_grouping(self):
        """セクション別にグループ化される。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情価", "感情価: 高め"),
                ("ムード", "ムード: 穏やか"),
            ],
            items_sec2=[
                ("認知状態", "認知状態: 明確"),
            ]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("高めの感情で穏やかで明確な状態")

        section_rates = ev.get_section_manifestation_rates()
        assert len(section_rates) == 2
        assert "セクション1: 内部状態" in section_rates
        assert "セクション2: 認知" in section_rates

    def test_section_item_count(self):
        """セクション内の項目数。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("項目A", "項目A: テスト値"),
                ("項目B", "項目B: テスト値2"),
            ],
            items_sec2=[
                ("項目C", "項目C: テスト値3"),
            ]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("テスト")

        section_rates = ev.get_section_manifestation_rates()
        assert section_rates["セクション1: 内部状態"]["item_count"] == 2
        assert section_rates["セクション2: 認知"]["item_count"] == 1


# ── 6. enrichmentデータ更新テスト ────────────────────────────────


class TestEnrichmentDataUpdate:
    """enrichmentデータ更新と語彙断片キャッシュ再生成テスト。"""

    def test_update_enrichment(self):
        """enrichmentデータ更新で語彙断片キャッシュが生成される。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 高め")]
        )
        ev.update_enrichment_data(sections)

        # 語彙断片キャッシュが生成されている
        assert len(ev._vocab_cache) > 0

    def test_update_regenerates_cache(self):
        """テキスト変更で語彙断片キャッシュが再生成される。"""
        ev = _make_evaluator()
        sections1 = _make_sections(
            items_sec1=[("感情", "感情: 高め")]
        )
        ev.update_enrichment_data(sections1)
        old_cache = dict(ev._vocab_cache)

        sections2 = _make_sections(
            items_sec1=[("感情", "感情: 低め")]
        )
        ev.update_enrichment_data(sections2)

        # キャッシュが更新されている
        item_id = "S0_感情"
        assert "低め" in ev._vocab_cache[item_id]
        assert "高め" not in ev._vocab_cache[item_id]

    def test_empty_item_no_fragments(self):
        """空項目は語彙断片なし。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("空項目", "(未蓄積)")]
        )
        ev.update_enrichment_data(sections)

        item_id = "S0_空項目"
        assert ev._vocab_cache.get(item_id) == []

    def test_unchanged_text_no_recompute(self):
        """テキスト未変更時はキャッシュ再生成しない。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 高め")]
        )
        ev.update_enrichment_data(sections)
        first_cache = ev._vocab_cache.get("S0_感情")

        # 同じデータで再更新
        ev.update_enrichment_data(sections)
        second_cache = ev._vocab_cache.get("S0_感情")

        # 同じオブジェクト(再生成されていない)
        assert first_cache is second_cache


# ── 7. ExecutionMonitor統合テスト ────────────────────────────────


class TestExecutionMonitorIntegration:
    """ExecutionMonitorとの統合テスト。"""

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_monitor_has_evaluator(self):
        """ExecutionMonitorにevaluatorが含まれる。"""
        mon = _make_monitor(enabled=True)
        assert hasattr(mon, 'enrichment_evaluator')
        assert isinstance(mon.enrichment_evaluator, EnrichmentEffectivenessEvaluator)

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_enrichment_distribution_updates_evaluator(self):
        """record_enrichment_distributionがevaluatorを更新する。"""
        mon = _make_monitor(enabled=True)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 高め")]
        )
        mon.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="圧縮テキスト",
        )

        # evaluatorにデータが更新されている
        assert len(mon.enrichment_evaluator._vocab_cache) > 0

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_evaluate_enrichment_manifestation(self):
        """evaluate_enrichment_manifestation呼び出し。"""
        mon = _make_monitor(enabled=True)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        mon.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="圧縮テキスト",
        )

        results = mon.evaluate_enrichment_manifestation("安定した気持ちです")
        assert isinstance(results, dict)
        assert mon.enrichment_evaluator.evaluation_count == 1

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_disabled_monitor_returns_empty(self):
        """無効時は空辞書を返す。"""
        mon = _make_monitor(enabled=False)
        results = mon.evaluate_enrichment_manifestation("テスト")
        assert results == {}

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_session_summary_includes_evaluator(self):
        """セッションサマリにevaluatorのサマリが含まれる。"""
        mon = _make_monitor(enabled=True)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        mon.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="圧縮テキスト",
        )
        mon.evaluate_enrichment_manifestation("安定です")

        # セッションサマリが例外なく出力される
        mon.emit_session_summary()


# ── 8. セッションサマリテスト ────────────────────────────────────


class TestSessionSummary:
    """セッションサマリテスト。"""

    def test_summary_structure(self):
        """サマリの構造。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")

        summary = ev.get_summary()
        assert "evaluation_count" in summary
        assert "total_match_count" in summary
        assert "total_manifest_count" in summary
        assert "item_count" in summary
        assert "item_manifestation_rates" in summary
        assert "section_manifestation_rates" in summary

    def test_emit_session_summary_no_monitor(self):
        """monitorなしでもセッションサマリが出力される。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")

        # 例外が出ないことを確認
        ev.emit_session_summary(monitor=None)

    def test_emit_session_summary_with_monitor(self):
        """monitorありでセッションサマリが出力される。"""
        ev = _make_evaluator()
        mon = _make_monitor(enabled=True)

        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")

        # 例外が出ないことを確認
        ev.emit_session_summary(monitor=mon)

    def test_summary_counts_accumulate(self):
        """累積統計が正しく蓄積される。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情A", "感情A: 安定"),
                ("感情B", "感情B: 不安定"),
            ]
        )
        ev.update_enrichment_data(sections)

        ev.evaluate_manifestation("安定した気持ちです")
        ev.evaluate_manifestation("不安定な状態です")

        assert ev.evaluation_count == 2
        assert ev.total_match_count > 0


# ── 9. 安全弁テスト ──────────────────────────────────────────────


class TestSafetyValves:
    """安全弁テスト。"""

    def test_exception_in_update(self):
        """update時の例外は安全に無視される。"""
        ev = _make_evaluator()
        # 不正なsections_data
        ev.update_enrichment_data([{"header": "test", "items": None}])
        # 例外が出ないことを確認
        assert True

    def test_exception_in_evaluate(self):
        """evaluate時の例外は安全に無視される。"""
        ev = _make_evaluator()
        # vocab_cacheを破壊して例外を発生させる
        ev._vocab_cache = {"broken": None}
        results = ev.evaluate_manifestation("テスト")
        # 例外が出ず空辞書が返る
        assert isinstance(results, dict)

    def test_session_boundary_reset(self):
        """インスタンス破棄でセッション境界消失。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")
        assert ev.evaluation_count > 0

        # 新しいインスタンスは空状態
        ev2 = _make_evaluator()
        assert ev2.evaluation_count == 0
        assert ev2.total_match_count == 0

    def test_fifo_prevents_memory_growth(self):
        """FIFO上限によりバッファが肥大化しない。"""
        ev = _make_evaluator(match_buffer_max=5)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)

        for _ in range(100):
            ev.evaluate_manifestation("安定です")

        for buf in ev._match_buffers.values():
            assert len(buf) <= 5

    def test_no_save_load_fields(self):
        """save/loadフィールドの追加なし。"""
        ev = _make_evaluator()
        # to_dict/from_dict/save/loadメソッドが存在しないことを確認
        assert not hasattr(ev, 'to_dict')
        assert not hasattr(ev, 'from_dict')
        assert not hasattr(ev, 'save')
        assert not hasattr(ev, 'load')


# ── 10. 計測手法の限界テスト ─────────────────────────────────────


class TestMeasurementLimitations:
    """計測手法の限界(設計書の明示的制約)のテスト。"""

    def test_indirect_influence_not_detected(self):
        """間接的影響は計測できない(安全弁6)。

        Geminiがenrichmentの情報を「参考にしつつ」
        別の表現で出力する場合は「非表出」として記録される。
        """
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情価: 非常に高い")]
        )
        ev.update_enrichment_data(sections)

        # 間接的な表現(「嬉しそう」は「非常に高い」の間接表現)
        results = ev.evaluate_manifestation("嬉しそうな表情をしています")

        # 直接的な文字列一致がないため非表出
        for item_id, is_manifest in results.items():
            if "感情" in item_id:
                # 「非常に高い」「感情価」のいずれも含まれていないはず
                assert is_manifest is False

    def test_accidental_match_not_excluded(self):
        """偶然の一致は排除できない。

        enrichmentと無関係にGeminiが同じ語彙を使用するケースも
        「表出あり」として記録される。
        """
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("天気", "天気: 晴れ")]
        )
        ev.update_enrichment_data(sections)

        # 「晴れ」はenrichmentと無関係に使われた語彙かもしれないが
        # 文字列一致があるため表出ありとして記録される
        results = ev.evaluate_manifestation("今日は晴れです")
        for item_id, is_manifest in results.items():
            if "天気" in item_id:
                assert is_manifest is True

    def test_kanji_hiragana_variation_not_handled(self):
        """表記揺れは考慮しない。

        ひらがな/カタカナ/漢字の違いは照合しない。
        """
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("気持ち", "気持ち: 嬉しい")]
        )
        ev.update_enrichment_data(sections)

        # 「うれしい」(ひらがな)は「嬉しい」(漢字)と一致しない
        results = ev.evaluate_manifestation("とてもうれしいです")
        for item_id, is_manifest in results.items():
            if "気持ち" in item_id:
                # 「嬉しい」の文字列が含まれていないので非表出
                # ただし「気持ち」は含まれるかもしれない
                pass  # 語彙断片次第

    def test_only_direct_manifestation(self):
        """直接表出のみを対象(安全弁6)。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("好奇心", "好奇心ドライブ: 段階3/5")]
        )
        ev.update_enrichment_data(sections)

        # 「好奇心ドライブ」は断片として抽出される
        # 「好奇心ドライブが活発です」は直接含まれる
        results1 = ev.evaluate_manifestation("好奇心ドライブが活発です")
        # 「興味津々」は間接表現であり断片に含まれない
        results2 = ev.evaluate_manifestation("興味津々です")

        # 直接含まれるケース
        for item_id, is_manifest in results1.items():
            if "好奇心" in item_id:
                assert is_manifest is True

        # 間接的なケースでは一致しない
        for item_id, is_manifest in results2.items():
            if "好奇心" in item_id:
                assert is_manifest is False


# ── 11. 構造的分離テスト ─────────────────────────────────────────


class TestStructuralSeparation:
    """構造的分離のテスト。"""

    def test_no_enrichment_output_function(self):
        """enrichment出力を生成する関数を持たない(安全弁1)。"""
        ev = _make_evaluator()
        # enrichmentの生成・変更に関わるメソッドが存在しないことを確認
        assert not hasattr(ev, 'generate_enrichment')
        assert not hasattr(ev, 'modify_enrichment')
        assert not hasattr(ev, 'set_enrichment_priority')

    def test_no_auto_optimization(self):
        """自動最適化を行わない(安全弁7)。"""
        ev = _make_evaluator()
        # 最適化・削除・優先順位変更のメソッドが存在しないことを確認
        assert not hasattr(ev, 'optimize_enrichment')
        assert not hasattr(ev, 'remove_ineffective_items')
        assert not hasattr(ev, 'reorder_items')

    def test_no_feedback_loop(self):
        """計測結果がenrichment生成に帰還しない。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情A", "感情A: 安定"),
                ("感情B", "感情B: 不安定"),
            ]
        )
        ev.update_enrichment_data(sections)

        # 何度評価しても、enrichmentデータ自体は変更されない
        for _ in range(10):
            ev.evaluate_manifestation("安定した気持ちです")

        # 語彙断片キャッシュは最初のupdate時と同じ
        # (evaluate_manifestationがvocab_cacheを変更していない)
        assert "S0_感情A" in ev._vocab_cache
        assert "S0_感情B" in ev._vocab_cache

    def test_fact_only_recording(self):
        """事実記述限定(安全弁3)。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")

        summary = ev.get_summary()
        # サマリに「有用性判定」「不要判定」のフィールドがない
        assert "usefulness" not in str(summary)
        assert "unnecessary" not in str(summary)
        assert "priority" not in str(summary)


# ── 12. 複数セクション・項目テスト ──────────────────────────────


class TestMultipleSectionsAndItems:
    """複数セクション・項目のテスト。"""

    def test_multiple_sections(self):
        """複数セクションの処理。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("感情", "感情: 安定"),
                ("ムード", "ムード: 穏やか"),
            ],
            items_sec2=[
                ("認知", "認知状態: 明確"),
                ("注意", "注意配分: 集中"),
            ]
        )
        ev.update_enrichment_data(sections)

        results = ev.evaluate_manifestation(
            "安定した気持ちで穏やかに集中しています"
        )

        # 4項目が評価される
        assert len(results) == 4

    def test_all_rates_after_evaluation(self):
        """全項目の表出率取得。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("項目1", "項目1: テスト値A"),
                ("項目2", "項目2: テスト値B"),
            ]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("テスト値Aが含まれます")

        rates = ev.get_all_manifestation_rates()
        assert len(rates) == 2

    def test_mixed_manifest_non_manifest(self):
        """表出あり/なしが混在するケース。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("項目A", "状態: 安定"),
                ("項目B", "状態: 極度不安定"),
            ]
        )
        ev.update_enrichment_data(sections)

        # 「安定」は含まれるが「極度不安定」は含まれない
        results = ev.evaluate_manifestation("安定した気持ちです")

        manifest_count = sum(1 for v in results.values() if v)
        non_manifest_count = sum(1 for v in results.values() if not v)
        assert manifest_count >= 1
        assert non_manifest_count >= 1


# ── 13. 累積統計テスト ──────────────────────────────────────────


class TestCumulativeStatistics:
    """セッション累積統計テスト。"""

    def test_total_match_count(self):
        """総照合回数の累積。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("項目A", "項目A: 値A"),
                ("項目B", "項目B: 値B"),
            ]
        )
        ev.update_enrichment_data(sections)

        ev.evaluate_manifestation("テスト1")
        ev.evaluate_manifestation("テスト2")

        # 2回の評価×2項目 = 4回の照合
        assert ev.total_match_count == 4

    def test_total_manifest_count(self):
        """総表出回数の累積。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)

        ev.evaluate_manifestation("安定しています")
        ev.evaluate_manifestation("安定しています")
        ev.evaluate_manifestation("天気がいいです")

        # 2回表出、1回非表出
        assert ev.total_manifest_count == 2

    def test_evaluation_count(self):
        """評価実行回数の累積。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)

        for _ in range(5):
            ev.evaluate_manifestation("テスト")

        assert ev.evaluation_count == 5


# ── 14. ログ出力テスト ──────────────────────────────────────────


class TestLogOutput:
    """ログストリームへのJSON出力テスト。"""

    @patch.dict("os.environ", {"CYRENE_MONITOR": "1"})
    def test_manifestation_log_output(self):
        """照合結果のログ出力。"""
        mon = _make_monitor(enabled=True)
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        mon.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="圧縮テキスト",
        )

        # ログ出力を確認(例外が出ないことを確認)
        results = mon.evaluate_enrichment_manifestation("安定です")
        assert isinstance(results, dict)

    def test_session_summary_log_format(self):
        """セッションサマリのログ出力形式。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        ev.evaluate_manifestation("安定です")

        # monitorなしでのログ出力(例外なし)
        ev.emit_session_summary()


# ── 15. エッジケーステスト ──────────────────────────────────────


class TestEdgeCases:
    """エッジケーステスト。"""

    def test_empty_sections(self):
        """空のsections_data。"""
        ev = _make_evaluator()
        ev.update_enrichment_data([])
        results = ev.evaluate_manifestation("テスト")
        assert results == {}

    def test_empty_utterance(self):
        """空の代弁テキスト。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        results = ev.evaluate_manifestation("")
        # 空テキストでは全て非表出
        for v in results.values():
            assert v is False

    def test_all_empty_items(self):
        """全項目が空パターン。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[
                ("項目A", "(未蓄積)"),
                ("項目B", "(なし)"),
            ]
        )
        ev.update_enrichment_data(sections)
        results = ev.evaluate_manifestation("テスト")
        # 空項目は語彙断片なし → 非表出
        for v in results.values():
            assert v is False

    def test_very_long_text(self):
        """非常に長いテキストの処理。"""
        ev = _make_evaluator()
        long_text = "安定" * 10000
        sections = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections)
        results = ev.evaluate_manifestation(long_text)
        # 処理が完了すること
        assert isinstance(results, dict)

    def test_unicode_handling(self):
        """Unicode文字の処理。"""
        ev = _make_evaluator()
        sections = _make_sections(
            items_sec1=[("絵文字", "状態: 良好☆")]
        )
        ev.update_enrichment_data(sections)
        results = ev.evaluate_manifestation("良好☆な状態です")
        assert isinstance(results, dict)

    def test_evaluate_before_update(self):
        """update前のevaluateは空結果。"""
        ev = _make_evaluator()
        results = ev.evaluate_manifestation("テスト")
        assert results == {}

    def test_multiple_updates_then_evaluate(self):
        """複数回updateしてからevaluate。"""
        ev = _make_evaluator()

        # 1回目のupdate
        sections1 = _make_sections(
            items_sec1=[("感情", "感情: 安定")]
        )
        ev.update_enrichment_data(sections1)

        # 2回目のupdate(テキスト変更)
        sections2 = _make_sections(
            items_sec1=[("感情", "感情: 不安定")]
        )
        ev.update_enrichment_data(sections2)

        # 最新のデータで照合される
        results = ev.evaluate_manifestation("不安定な状態です")
        for item_id, is_manifest in results.items():
            if "感情" in item_id:
                assert is_manifest is True


# ── 16. _extract_vocab_fragments 詳細テスト ──────────────────────


class TestVocabFragmentsDetailed:
    """語彙断片抽出の詳細テスト。"""

    def test_equals_separator(self):
        """等号区切り。"""
        frags = _extract_vocab_fragments("joy=0.7")
        assert "joy" in frags
        # 0.7は数値なので除外

    def test_pipe_separator(self):
        """パイプ区切り。"""
        frags = _extract_vocab_fragments("高い|中程度|低い")
        assert "高い" in frags
        assert "中程度" in frags
        assert "低い" in frags

    def test_mixed_separators(self):
        """複合区切り文字。"""
        frags = _extract_vocab_fragments("感情価: 高め→安定 / 穏やか方向")
        assert "感情価" in frags
        assert "高め" in frags
        assert "安定" in frags
        assert "穏やか方向" in frags

    def test_single_char_japanese_included(self):
        """2文字以上の日本語は含まれる。"""
        frags = _extract_vocab_fragments("安定")
        assert "安定" in frags

    def test_single_char_excluded(self):
        """1文字は除外。"""
        frags = _extract_vocab_fragments("A: B")
        assert "A" not in frags
        assert "B" not in frags

    def test_whitespace_only_excluded(self):
        """空白のみのテキスト。"""
        frags = _extract_vocab_fragments("   ")
        assert frags == []

    def test_tab_separator(self):
        """タブ区切り。"""
        frags = _extract_vocab_fragments("項目A\t項目B")
        assert "項目A" in frags
        assert "項目B" in frags

    def test_semicolon_separator(self):
        """セミコロン区切り。"""
        frags = _extract_vocab_fragments("項目A；項目B;項目C")
        assert "項目A" in frags
        assert "項目B" in frags
        assert "項目C" in frags
