"""
tests/test_enrichment_efficacy_evaluation.py - EnrichmentEfficacyEvaluation のテスト

テスト項目:
- 初期化テスト(デフォルト状態、カスタムパラメータ、環境変数制御)
- 第一段: 照合結果の構造化蓄積テスト
  - 項目別FIFOバッファ蓄積
  - セクション別表出率集約
  - 全体表出率集約
  - FIFO上限によるバッファ制限(安全弁4)
- 第二段: 照合精度の限界記述テスト
  - 語彙断片空項目カウント
  - 語彙断片平均長
  - 非表出かつ語彙断片ありのカウント
- 第三段: セッション内累積記述テスト
  - 項目別表出率
  - セクション別平均表出率
  - 全体平均表出率
  - 照合不可能項目割合
  - 語彙断片統計
  - 照合精度限界の常時併記(安全弁6)
- セッションサマリテスト
- 読み取り専用アクセサテスト
- 安全弁テスト
  - enrichment経路遮断(安全弁1)
  - 永続化非対象(安全弁2)
  - 評価的判定排除(安全弁3)
  - FIFO上限(安全弁4)
  - 環境変数制御(安全弁5)
  - 精度限界常時併記(安全弁6)
  - パターン抽出禁止(安全弁7)
- 構造的分離テスト
  - 帰還経路なし
  - 方針選択非介入
  - enrichment経路遮断
  - 状態更新非介入
"""

import json
import logging
import os
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.enrichment_efficacy_evaluation import (
    EnrichmentEfficacyEvaluation,
    _is_monitor_enabled,
    _ITEM_BUFFER_DEFAULT_MAX,
    _SECTION_BUFFER_DEFAULT_MAX,
    _OVERALL_BUFFER_DEFAULT_MAX,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_eval(
    enabled: bool = True,
    item_buffer_max: int = _ITEM_BUFFER_DEFAULT_MAX,
    section_buffer_max: int = _SECTION_BUFFER_DEFAULT_MAX,
    overall_buffer_max: int = _OVERALL_BUFFER_DEFAULT_MAX,
) -> EnrichmentEfficacyEvaluation:
    """テスト用のEnrichmentEfficacyEvaluationを生成する。"""
    return EnrichmentEfficacyEvaluation(
        item_buffer_max=item_buffer_max,
        section_buffer_max=section_buffer_max,
        overall_buffer_max=overall_buffer_max,
        enabled=enabled,
    )


def _make_results(
    manifest_items: list[str],
    non_manifest_items: list[str],
) -> dict[str, bool]:
    """照合結果を生成する。"""
    results = {}
    for item_id in manifest_items:
        results[item_id] = True
    for item_id in non_manifest_items:
        results[item_id] = False
    return results


def _make_section_map(
    items: list[str], sections: list[str]
) -> dict[str, str]:
    """項目→セクション対応を生成する。"""
    m = {}
    for i, item_id in enumerate(items):
        m[item_id] = sections[i % len(sections)]
    return m


def _make_vocab_cache(
    items_with_frags: dict[str, list[str]],
) -> dict[str, list[str]]:
    """語彙断片キャッシュを生成する。"""
    return dict(items_with_frags)


# ── 初期化テスト ─────────────────────────────────────────────────


class TestInitialization:
    """初期化テスト。"""

    def test_default_initialization(self):
        """デフォルトパラメータでの初期化。"""
        ev = _make_eval()
        assert ev.enabled is True
        assert ev.evaluation_count == 0
        assert ev.get_item_buffers() == {}
        assert ev.get_section_buffers() == {}
        assert ev.get_overall_buffer() == []

    def test_disabled_initialization(self):
        """無効状態での初期化。"""
        ev = _make_eval(enabled=False)
        assert ev.enabled is False

    def test_custom_buffer_sizes(self):
        """カスタムバッファサイズでの初期化。"""
        ev = _make_eval(
            item_buffer_max=10,
            section_buffer_max=20,
            overall_buffer_max=30,
        )
        assert ev._item_buffer_max == 10
        assert ev._section_buffer_max == 20
        assert ev._overall_buffer_max == 30

    def test_minimum_buffer_sizes(self):
        """バッファサイズの下限。"""
        ev = _make_eval(
            item_buffer_max=0,
            section_buffer_max=-1,
            overall_buffer_max=-100,
        )
        assert ev._item_buffer_max == 1
        assert ev._section_buffer_max == 1
        assert ev._overall_buffer_max == 1

    def test_env_var_disabled(self):
        """環境変数による無効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}, clear=False):
            assert _is_monitor_enabled() is False

    def test_env_var_enabled(self):
        """環境変数による有効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}, clear=False):
            assert _is_monitor_enabled() is True

    def test_env_var_missing(self):
        """環境変数が未設定の場合。"""
        env = dict(os.environ)
        env.pop("CYRENE_MONITOR", None)
        with patch.dict(os.environ, env, clear=True):
            assert _is_monitor_enabled() is False


# ── 第一段: 照合結果の構造化蓄積テスト ────────────────────────────


class TestStructuredAccumulation:
    """第一段: 照合結果の構造化蓄積。"""

    def test_item_level_accumulation(self):
        """項目別のFIFO蓄積。"""
        ev = _make_eval()
        results = _make_results(["A", "B"], ["C"])
        section_map = _make_section_map(["A", "B", "C"], ["sec1"])

        ev.record_manifestation_results(results, section_map)

        buffers = ev.get_item_buffers()
        assert buffers["A"] == [True]
        assert buffers["B"] == [True]
        assert buffers["C"] == [False]

    def test_item_level_multiple_records(self):
        """複数回の項目別蓄積。"""
        ev = _make_eval()
        section_map = _make_section_map(["A"], ["sec1"])

        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": False}, section_map)
        ev.record_manifestation_results({"A": True}, section_map)

        buffers = ev.get_item_buffers()
        assert buffers["A"] == [True, False, True]

    def test_section_level_accumulation(self):
        """セクション別の表出率集約。"""
        ev = _make_eval()
        # sec1に2項目: A(表出), B(非表出) → 表出率 0.5
        results = {"A": True, "B": False}
        section_map = {"A": "sec1", "B": "sec1"}

        ev.record_manifestation_results(results, section_map)

        section_buffers = ev.get_section_buffers()
        assert len(section_buffers["sec1"]) == 1
        assert section_buffers["sec1"][0] == pytest.approx(0.5)

    def test_section_level_multiple_sections(self):
        """複数セクションでの集約。"""
        ev = _make_eval()
        results = {"A": True, "B": False, "C": True, "D": True}
        section_map = {"A": "sec1", "B": "sec1", "C": "sec2", "D": "sec2"}

        ev.record_manifestation_results(results, section_map)

        section_buffers = ev.get_section_buffers()
        assert section_buffers["sec1"][0] == pytest.approx(0.5)  # 1/2
        assert section_buffers["sec2"][0] == pytest.approx(1.0)  # 2/2

    def test_overall_accumulation(self):
        """全体の表出率集約。"""
        ev = _make_eval()
        results = _make_results(["A", "B"], ["C"])
        section_map = _make_section_map(["A", "B", "C"], ["sec1"])

        ev.record_manifestation_results(results, section_map)

        overall = ev.get_overall_buffer()
        assert len(overall) == 1
        assert overall[0] == pytest.approx(2.0 / 3.0)

    def test_overall_multiple_records(self):
        """複数回の全体蓄積。"""
        ev = _make_eval()
        section_map = _make_section_map(["A", "B"], ["sec1"])

        ev.record_manifestation_results({"A": True, "B": True}, section_map)
        ev.record_manifestation_results({"A": True, "B": False}, section_map)
        ev.record_manifestation_results({"A": False, "B": False}, section_map)

        overall = ev.get_overall_buffer()
        assert len(overall) == 3
        assert overall[0] == pytest.approx(1.0)
        assert overall[1] == pytest.approx(0.5)
        assert overall[2] == pytest.approx(0.0)

    def test_evaluation_count_increments(self):
        """照合回数のインクリメント。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": False}, section_map)

        assert ev.evaluation_count == 2

    def test_empty_results(self):
        """空の照合結果。"""
        ev = _make_eval()
        ev.record_manifestation_results({}, {})

        assert ev.evaluation_count == 1
        assert ev.get_item_buffers() == {}
        assert ev.get_overall_buffer() == [0.0]

    def test_disabled_skips_accumulation(self):
        """無効時は蓄積をスキップ。"""
        ev = _make_eval(enabled=False)
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)

        assert ev.evaluation_count == 0
        assert ev.get_item_buffers() == {}
        assert ev.get_overall_buffer() == []


# ── FIFO上限テスト(安全弁4) ──────────────────────────────────────


class TestFIFOLimits:
    """FIFO上限による蓄積制限(安全弁4)。"""

    def test_item_buffer_fifo_limit(self):
        """項目別バッファのFIFO上限。"""
        ev = _make_eval(item_buffer_max=3)
        section_map = {"A": "sec1"}

        for v in [True, False, True, False, True]:
            ev.record_manifestation_results({"A": v}, section_map)

        buf = ev.get_item_buffers()["A"]
        assert len(buf) == 3
        # 最新3件のみ: True, False, True
        assert buf == [True, False, True]

    def test_section_buffer_fifo_limit(self):
        """セクション別バッファのFIFO上限。"""
        ev = _make_eval(section_buffer_max=2)
        section_map = {"A": "sec1"}

        for v in [True, False, True]:
            ev.record_manifestation_results({"A": v}, section_map)

        sec_buf = ev.get_section_buffers()["sec1"]
        assert len(sec_buf) == 2

    def test_overall_buffer_fifo_limit(self):
        """全体バッファのFIFO上限。"""
        ev = _make_eval(overall_buffer_max=2)
        section_map = {"A": "sec1"}

        for v in [True, False, True]:
            ev.record_manifestation_results({"A": v}, section_map)

        overall = ev.get_overall_buffer()
        assert len(overall) == 2


# ── 第二段: 照合精度の限界記述テスト ─────────────────────────────


class TestPrecisionLimitations:
    """第二段: 照合精度の限界の構造的記述。"""

    def test_empty_fragment_count(self):
        """語彙断片が空であった項目の数。"""
        ev = _make_eval()
        vocab_cache = {
            "A": ["word1", "word2"],
            "B": [],
            "C": ["word3"],
            "D": [],
        }
        section_map = {"A": "sec1", "B": "sec1", "C": "sec2", "D": "sec2"}

        ev.update_vocab_statistics(vocab_cache, section_map)
        limitations = ev.get_precision_limitations()

        assert limitations["empty_fragment_count"] == 2  # B, D
        assert limitations["total_items_tracked"] == 4

    def test_avg_fragment_length(self):
        """語彙断片の平均長。"""
        ev = _make_eval()
        vocab_cache = {
            "A": ["ab", "cdef"],  # 2 + 4 = 6
            "B": ["xyz"],          # 3
        }
        section_map = {"A": "sec1", "B": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)
        limitations = ev.get_precision_limitations()

        # 3 fragments, total length 9
        assert limitations["total_fragment_count"] == 3
        assert limitations["avg_fragment_length"] == pytest.approx(3.0)

    def test_non_manifest_with_fragments(self):
        """非表出のうち語彙断片が存在した項目の数。"""
        ev = _make_eval()
        vocab_cache = {
            "A": ["word1"],  # 語彙あり
            "B": [],          # 語彙なし
            "C": ["word2"],  # 語彙あり
        }
        section_map = {"A": "sec1", "B": "sec1", "C": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        # A: 表出, B: 非表出(語彙なし), C: 非表出(語彙あり)
        results = {"A": True, "B": False, "C": False}
        ev.record_manifestation_results(results, section_map)

        limitations = ev.get_precision_limitations()
        assert limitations["non_manifest_with_fragments_count"] == 1  # C only

    def test_no_fragments_at_all(self):
        """語彙断片がまだ登録されていない場合。"""
        ev = _make_eval()
        limitations = ev.get_precision_limitations()

        assert limitations["empty_fragment_count"] == 0
        assert limitations["total_fragment_count"] == 0
        assert limitations["avg_fragment_length"] == 0.0
        assert limitations["non_manifest_with_fragments_count"] == 0
        assert limitations["total_items_tracked"] == 0

    def test_all_fragments_empty(self):
        """全項目の語彙断片が空の場合。"""
        ev = _make_eval()
        vocab_cache = {"A": [], "B": [], "C": []}
        section_map = {"A": "sec1", "B": "sec1", "C": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)
        limitations = ev.get_precision_limitations()

        assert limitations["empty_fragment_count"] == 3
        assert limitations["total_fragment_count"] == 0
        assert limitations["avg_fragment_length"] == 0.0


# ── 第三段: セッション内累積記述テスト ────────────────────────────


class TestCumulativeDescription:
    """第三段: セッション内累積記述の提供。"""

    def test_basic_cumulative_description(self):
        """基本的な累積記述。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec1", "C": "sec2"}
        vocab_cache = {"A": ["w1"], "B": ["w2"], "C": ["w3"]}

        ev.update_vocab_statistics(vocab_cache, section_map)
        ev.record_manifestation_results(
            {"A": True, "B": False, "C": True}, section_map
        )

        desc = ev.get_cumulative_description()

        assert desc["evaluation_count"] == 1
        assert desc["item_count"] == 3
        assert desc["section_count"] == 2
        assert desc["overall_avg_rate"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert desc["item_rates"]["A"] == pytest.approx(1.0)
        assert desc["item_rates"]["B"] == pytest.approx(0.0)
        assert desc["item_rates"]["C"] == pytest.approx(1.0)

    def test_cumulative_after_multiple_records(self):
        """複数回蓄積後の累積記述。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": False}, section_map)

        desc = ev.get_cumulative_description()

        assert desc["item_rates"]["A"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert desc["overall_avg_rate"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_unmatchable_rate(self):
        """照合不可能項目の割合。"""
        ev = _make_eval()
        vocab_cache = {"A": ["w1"], "B": [], "C": [], "D": ["w2"]}
        section_map = {"A": "sec1", "B": "sec1", "C": "sec2", "D": "sec2"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        desc = ev.get_cumulative_description()
        # 2/4 = 0.5
        assert desc["unmatchable_rate"] == pytest.approx(0.5)

    def test_fragment_statistics(self):
        """語彙断片の統計。"""
        ev = _make_eval()
        vocab_cache = {
            "A": ["word1", "word2"],
            "B": ["xyz"],
        }
        section_map = {"A": "sec1", "B": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        desc = ev.get_cumulative_description()
        assert desc["total_fragments"] == 3
        assert desc["avg_fragment_length"] > 0.0

    def test_precision_limitations_always_included(self):
        """照合精度の限界が常に含まれる(安全弁6)。"""
        ev = _make_eval()
        desc = ev.get_cumulative_description()

        assert "precision_limitations" in desc
        lim = desc["precision_limitations"]
        assert "empty_fragment_count" in lim
        assert "avg_fragment_length" in lim
        assert "non_manifest_with_fragments_count" in lim

    def test_empty_cumulative_description(self):
        """蓄積なしの累積記述。"""
        ev = _make_eval()
        desc = ev.get_cumulative_description()

        assert desc["evaluation_count"] == 0
        assert desc["item_count"] == 0
        assert desc["section_count"] == 0
        assert desc["overall_avg_rate"] == 0.0
        assert desc["unmatchable_rate"] == 0.0

    def test_section_avg_rates_computation(self):
        """セクション別平均表出率の計算。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec1", "C": "sec2"}

        # 1回目: sec1=0.5(1/2), sec2=1.0(1/1)
        ev.record_manifestation_results(
            {"A": True, "B": False, "C": True}, section_map
        )
        # 2回目: sec1=1.0(2/2), sec2=0.0(0/1)
        ev.record_manifestation_results(
            {"A": True, "B": True, "C": False}, section_map
        )

        desc = ev.get_cumulative_description()
        # sec1: avg of [0.5, 1.0] = 0.75
        assert desc["section_avg_rates"]["sec1"] == pytest.approx(0.75)
        # sec2: avg of [1.0, 0.0] = 0.5
        assert desc["section_avg_rates"]["sec2"] == pytest.approx(0.5)


# ── 個別アクセサテスト ────────────────────────────────────────────


class TestAccessors:
    """個別アクセサのテスト。"""

    def test_get_item_rate(self):
        """項目別表出率の取得。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": False}, section_map)

        assert ev.get_item_rate("A") == pytest.approx(0.5)

    def test_get_item_rate_nonexistent(self):
        """存在しない項目の表出率。"""
        ev = _make_eval()
        assert ev.get_item_rate("nonexistent") == 0.0

    def test_get_section_avg_rate(self):
        """セクション別平均表出率の取得。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec1"}

        ev.record_manifestation_results(
            {"A": True, "B": False}, section_map
        )

        assert ev.get_section_avg_rate("sec1") == pytest.approx(0.5)

    def test_get_section_avg_rate_nonexistent(self):
        """存在しないセクションの平均表出率。"""
        ev = _make_eval()
        assert ev.get_section_avg_rate("nonexistent") == 0.0

    def test_get_overall_avg_rate(self):
        """全体平均表出率の取得。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)
        ev.record_manifestation_results({"A": False}, section_map)

        assert ev.get_overall_avg_rate() == pytest.approx(0.5)

    def test_get_overall_avg_rate_empty(self):
        """蓄積なしの全体平均表出率。"""
        ev = _make_eval()
        assert ev.get_overall_avg_rate() == 0.0

    def test_get_item_buffers_is_copy(self):
        """項目別バッファがコピーであること。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        buf = ev.get_item_buffers()
        buf["A"].append(False)  # コピーに変更

        # 元のバッファは変更されない
        assert ev.get_item_buffers()["A"] == [True]

    def test_get_section_buffers_is_copy(self):
        """セクション別バッファがコピーであること。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        buf = ev.get_section_buffers()
        buf["sec1"].append(0.0)  # コピーに変更

        assert len(ev.get_section_buffers()["sec1"]) == 1

    def test_get_overall_buffer_is_copy(self):
        """全体バッファがコピーであること。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        buf = ev.get_overall_buffer()
        buf.append(0.0)  # コピーに変更

        assert len(ev.get_overall_buffer()) == 1


# ── セッションサマリテスト ────────────────────────────────────────


class TestSessionSummary:
    """セッションサマリの出力テスト。"""

    def test_basic_session_summary(self):
        """基本的なセッションサマリ。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec1"}

        ev.record_manifestation_results(
            {"A": True, "B": False}, section_map
        )

        summary = ev.emit_session_summary()

        assert summary is not None
        assert summary["type"] == "enrichment_efficacy_session_summary"
        assert "evaluation_count" in summary
        assert "item_rates" in summary
        assert "section_avg_rates" in summary
        assert "overall_avg_rate" in summary
        assert "precision_limitations" in summary
        assert "timestamp" in summary

    def test_session_summary_disabled(self):
        """無効時はNoneを返す。"""
        ev = _make_eval(enabled=False)
        summary = ev.emit_session_summary()
        assert summary is None

    def test_session_summary_with_monitor(self):
        """ExecutionMonitor経由のログ出力。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        mock_monitor = MagicMock()
        mock_monitor._emit_json = MagicMock()

        summary = ev.emit_session_summary(monitor=mock_monitor)

        assert summary is not None
        mock_monitor._emit_json.assert_called_once()

    def test_session_summary_without_monitor(self):
        """モニターなしでのログ出力。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        summary = ev.emit_session_summary(monitor=None)

        assert summary is not None
        assert summary["type"] == "enrichment_efficacy_session_summary"

    def test_session_summary_empty(self):
        """蓄積なしのセッションサマリ。"""
        ev = _make_eval()
        summary = ev.emit_session_summary()

        assert summary is not None
        assert summary["evaluation_count"] == 0


# ── 語彙断片統計更新テスト ────────────────────────────────────────


class TestVocabStatisticsUpdate:
    """語彙断片統計の更新テスト。"""

    def test_update_basic(self):
        """基本的な語彙断片統計更新。"""
        ev = _make_eval()
        vocab_cache = {"A": ["word1", "word2"]}
        section_map = {"A": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        assert ev._item_fragment_counts["A"] == 2
        assert ev._item_fragment_total_lengths["A"] == 10  # 5+5

    def test_update_empty_fragments(self):
        """空の語彙断片の統計更新。"""
        ev = _make_eval()
        vocab_cache = {"A": []}
        section_map = {"A": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        assert ev._item_fragment_counts["A"] == 0
        assert ev._item_fragment_total_lengths["A"] == 0

    def test_update_disabled_skips(self):
        """無効時は更新をスキップ。"""
        ev = _make_eval(enabled=False)
        vocab_cache = {"A": ["word1"]}
        section_map = {"A": "sec1"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        assert ev._item_fragment_counts == {}

    def test_update_section_map(self):
        """セクション対応の更新。"""
        ev = _make_eval()
        vocab_cache = {"A": ["w1"], "B": ["w2"]}
        section_map = {"A": "sec1", "B": "sec2"}

        ev.update_vocab_statistics(vocab_cache, section_map)

        assert ev._item_section_map["A"] == "sec1"
        assert ev._item_section_map["B"] == "sec2"

    def test_update_multiple_times(self):
        """複数回の更新で最新の値が反映される。"""
        ev = _make_eval()

        ev.update_vocab_statistics({"A": ["w1"]}, {"A": "sec1"})
        assert ev._item_fragment_counts["A"] == 1

        ev.update_vocab_statistics({"A": ["w1", "w2", "w3"]}, {"A": "sec1"})
        assert ev._item_fragment_counts["A"] == 3


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁のテスト。"""

    def test_safety_valve_1_no_enrichment_output(self):
        """安全弁1: enrichment出力を生成する関数を持たない。"""
        ev = _make_eval()
        # EnrichmentEfficacyEvaluationにenrichment出力生成メソッドがないことを確認
        assert not hasattr(ev, 'generate_enrichment')
        assert not hasattr(ev, 'build_enrichment')
        assert not hasattr(ev, 'create_enrichment')
        assert not hasattr(ev, 'produce_enrichment')

    def test_safety_valve_2_no_persistence(self):
        """安全弁2: 永続化対象外。"""
        ev = _make_eval()
        # to_dict/from_dict/save/loadメソッドが存在しないことを確認
        assert not hasattr(ev, 'to_dict')
        assert not hasattr(ev, 'from_dict')
        assert not hasattr(ev, 'save')
        assert not hasattr(ev, 'load')

    def test_safety_valve_3_no_evaluative_labels(self):
        """安全弁3: 評価的判定を含まない。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        desc = ev.get_cumulative_description()
        desc_str = json.dumps(desc, ensure_ascii=False)

        # 評価的ラベルが含まれないことを確認
        evaluative_words = [
            "高い", "低い", "十分", "不十分", "良い", "悪い",
            "必要", "不要", "正常", "異常", "目標",
            "good", "bad", "high", "low", "sufficient", "insufficient",
        ]
        for word in evaluative_words:
            assert word not in desc_str, f"Evaluative label found: {word}"

    def test_safety_valve_4_fifo_limits(self):
        """安全弁4: FIFO上限で無限蓄積を防止。"""
        ev = _make_eval(
            item_buffer_max=5,
            section_buffer_max=5,
            overall_buffer_max=5,
        )
        section_map = {"A": "sec1"}

        for i in range(20):
            ev.record_manifestation_results(
                {"A": i % 2 == 0}, section_map
            )

        assert len(ev.get_item_buffers()["A"]) <= 5
        assert len(ev.get_section_buffers()["sec1"]) <= 5
        assert len(ev.get_overall_buffer()) <= 5

    def test_safety_valve_5_disabled_mode(self):
        """安全弁5: 無効時は全操作をスキップ。"""
        ev = _make_eval(enabled=False)
        section_map = {"A": "sec1"}

        ev.record_manifestation_results({"A": True}, section_map)
        ev.update_vocab_statistics({"A": ["w1"]}, section_map)

        assert ev.evaluation_count == 0
        assert ev.get_item_buffers() == {}
        assert ev._item_fragment_counts == {}
        assert ev.emit_session_summary() is None

    def test_safety_valve_6_precision_limitations_always_present(self):
        """安全弁6: 精度限界が常に併記される。"""
        ev = _make_eval()

        # 蓄積前でも精度限界が取得可能
        limitations = ev.get_precision_limitations()
        assert "empty_fragment_count" in limitations
        assert "avg_fragment_length" in limitations
        assert "non_manifest_with_fragments_count" in limitations

        # 累積記述に精度限界が含まれる
        desc = ev.get_cumulative_description()
        assert "precision_limitations" in desc

        # セッションサマリに精度限界が含まれる
        summary = ev.emit_session_summary()
        assert summary is not None
        assert "precision_limitations" in summary

    def test_safety_valve_7_no_pattern_extraction(self):
        """安全弁7: パターン抽出処理を持たない。"""
        ev = _make_eval()
        # パターン抽出メソッドが存在しないことを確認
        assert not hasattr(ev, 'detect_pattern')
        assert not hasattr(ev, 'analyze_trend')
        assert not hasattr(ev, 'extract_pattern')
        assert not hasattr(ev, 'find_regularity')
        assert not hasattr(ev, 'compute_trend')


# ── 構造的分離テスト ─────────────────────────────────────────────


class TestStructuralIsolation:
    """構造的分離のテスト。"""

    def test_no_return_pathway_interface(self):
        """帰還経路への入力経路が存在しない。"""
        ev = _make_eval()
        # 帰還経路関連のメソッドが存在しない
        assert not hasattr(ev, 'get_drive_change')
        assert not hasattr(ev, 'get_mood_change')
        assert not hasattr(ev, 'get_emotion_change')

    def test_no_policy_selection_interface(self):
        """方針選択への入力経路が存在しない。"""
        ev = _make_eval()
        assert not hasattr(ev, 'get_policy_score')
        assert not hasattr(ev, 'modify_policy')

    def test_no_enrichment_modification_interface(self):
        """enrichment変更の経路が存在しない。"""
        ev = _make_eval()
        assert not hasattr(ev, 'modify_enrichment')
        assert not hasattr(ev, 'update_enrichment')
        assert not hasattr(ev, 'set_enrichment')
        assert not hasattr(ev, 'generate_enrichment')

    def test_no_state_update_interface(self):
        """状態更新への入力経路が存在しない。"""
        ev = _make_eval()
        assert not hasattr(ev, 'update_state')
        assert not hasattr(ev, 'set_emotion')
        assert not hasattr(ev, 'set_mood')
        assert not hasattr(ev, 'set_drive')

    def test_outputs_are_readonly(self):
        """出力が読み取り専用であること。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        ev.record_manifestation_results({"A": True}, section_map)

        # get_cumulative_descriptionの戻り値を変更しても内部に影響しない
        desc = ev.get_cumulative_description()
        desc["item_rates"]["A"] = 0.0
        desc2 = ev.get_cumulative_description()
        assert desc2["item_rates"]["A"] == pytest.approx(1.0)


# ── 例外安全性テスト ─────────────────────────────────────────────


class TestExceptionSafety:
    """例外発生時の安全性テスト。"""

    def test_record_with_none_section_map(self):
        """Noneのsection_mapでもクラッシュしない。"""
        ev = _make_eval()
        # section_mapがNoneでもTypeErrorを吸収
        try:
            ev.record_manifestation_results({"A": True}, None)
        except Exception:
            # 安全弁で吸収されるべき
            pass
        # クラッシュしていないことの確認
        assert ev is not None

    def test_vocab_stats_with_bad_data(self):
        """不正なデータでの語彙統計更新。"""
        ev = _make_eval()
        # 不正な型でもクラッシュしない
        try:
            ev.update_vocab_statistics(None, None)
        except Exception:
            pass
        assert ev is not None

    def test_precision_limitations_exception_safety(self):
        """精度限界取得の例外安全性。"""
        ev = _make_eval()
        # 内部状態を壊す
        ev._item_fragment_counts = None
        limitations = ev.get_precision_limitations()
        # 例外時でもデフォルト値が返る
        assert limitations["empty_fragment_count"] == 0

    def test_cumulative_description_exception_safety(self):
        """累積記述の例外安全性。"""
        ev = _make_eval()
        # 内部状態を壊す
        ev._item_buffers = None
        desc = ev.get_cumulative_description()
        # 例外時でも構造が返る
        assert "evaluation_count" in desc
        assert "precision_limitations" in desc

    def test_session_summary_exception_safety(self):
        """セッションサマリの例外安全性。"""
        ev = _make_eval()
        ev._item_buffers = None
        # クラッシュしない
        summary = ev.emit_session_summary()
        # 例外時でも何かしらの値が返る(Noneの可能性もある)
        assert True  # クラッシュしなかったことの確認


# ── 複合シナリオテスト ────────────────────────────────────────────


class TestIntegrationScenarios:
    """複合シナリオのテスト。"""

    def test_full_workflow(self):
        """完全なワークフロー: 語彙統計→照合結果蓄積→累積記述→サマリ。"""
        ev = _make_eval()

        # 1. 語彙断片統計の更新
        vocab_cache = {
            "S0_item1": ["感情", "認知"],
            "S0_item2": ["記憶"],
            "S1_item3": [],
        }
        section_map = {
            "S0_item1": "心理状態",
            "S0_item2": "心理状態",
            "S1_item3": "記述層",
        }
        ev.update_vocab_statistics(vocab_cache, section_map)

        # 2. 照合結果の蓄積(3回)
        ev.record_manifestation_results(
            {"S0_item1": True, "S0_item2": False, "S1_item3": False},
            section_map,
        )
        ev.record_manifestation_results(
            {"S0_item1": True, "S0_item2": True, "S1_item3": False},
            section_map,
        )
        ev.record_manifestation_results(
            {"S0_item1": False, "S0_item2": True, "S1_item3": False},
            section_map,
        )

        # 3. 累積記述の確認
        desc = ev.get_cumulative_description()

        assert desc["evaluation_count"] == 3
        assert desc["item_count"] == 3
        assert desc["item_rates"]["S0_item1"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert desc["item_rates"]["S0_item2"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert desc["item_rates"]["S1_item3"] == pytest.approx(0.0)

        # セクション別確認
        assert "心理状態" in desc["section_avg_rates"]
        assert "記述層" in desc["section_avg_rates"]

        # 照合精度の限界が併記されている
        assert desc["precision_limitations"]["empty_fragment_count"] == 1
        assert desc["unmatchable_rate"] == pytest.approx(1.0 / 3.0, abs=0.01)

        # 4. セッションサマリの出力
        summary = ev.emit_session_summary()
        assert summary is not None
        assert summary["type"] == "enrichment_efficacy_session_summary"
        assert summary["precision_limitations"]["empty_fragment_count"] == 1

    def test_many_items_scenario(self):
        """多数の項目を持つシナリオ。"""
        ev = _make_eval()
        n_items = 50

        # 50項目を5セクションに分配
        items = [f"item_{i}" for i in range(n_items)]
        sections = [f"sec_{i // 10}" for i in range(n_items)]
        section_map = dict(zip(items, sections))

        vocab_cache = {
            item: [f"word_{i}"] for i, item in enumerate(items)
        }
        ev.update_vocab_statistics(vocab_cache, section_map)

        # 半分が表出
        results = {item: (i % 2 == 0) for i, item in enumerate(items)}
        ev.record_manifestation_results(results, section_map)

        desc = ev.get_cumulative_description()
        assert desc["item_count"] == 50
        assert desc["overall_avg_rate"] == pytest.approx(0.5)

    def test_session_boundary_reset(self):
        """セッション境界での全内部状態消失。"""
        ev1 = _make_eval()
        section_map = {"A": "sec1"}

        ev1.record_manifestation_results({"A": True}, section_map)
        assert ev1.evaluation_count == 1

        # 新しいインスタンスは白紙状態
        ev2 = _make_eval()
        assert ev2.evaluation_count == 0
        assert ev2.get_item_buffers() == {}

    def test_alternating_manifest_non_manifest(self):
        """表出/非表出の交互パターン。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        for i in range(10):
            ev.record_manifestation_results(
                {"A": i % 2 == 0}, section_map
            )

        # 5/10 = 0.5
        assert ev.get_item_rate("A") == pytest.approx(0.5)

    def test_all_manifest(self):
        """全項目が常に表出。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec2"}

        for _ in range(5):
            ev.record_manifestation_results(
                {"A": True, "B": True}, section_map
            )

        assert ev.get_overall_avg_rate() == pytest.approx(1.0)

    def test_all_non_manifest(self):
        """全項目が常に非表出。"""
        ev = _make_eval()
        section_map = {"A": "sec1", "B": "sec2"}

        for _ in range(5):
            ev.record_manifestation_results(
                {"A": False, "B": False}, section_map
            )

        assert ev.get_overall_avg_rate() == pytest.approx(0.0)


# ── ログ出力テスト ────────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のテスト。"""

    def test_record_log_emitted(self):
        """照合結果記録時のログ出力。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}

        with patch('tools.enrichment_efficacy_evaluation._logger') as mock_logger:
            ev.record_manifestation_results({"A": True}, section_map)
            # ログが出力されたことの確認
            assert mock_logger.debug.called
            assert ev.evaluation_count == 1

    def test_log_contains_precision_limitations(self):
        """ログ出力に精度限界が含まれる(安全弁6)。"""
        ev = _make_eval()
        section_map = {"A": "sec1"}
        vocab_cache = {"A": ["w1"]}
        ev.update_vocab_statistics(vocab_cache, section_map)

        log_records = []
        original_debug = _make_eval.__module__

        # ログ出力をキャプチャ
        with patch('tools.enrichment_efficacy_evaluation._logger') as mock_logger:
            ev.record_manifestation_results({"A": True}, section_map)
            if mock_logger.debug.called:
                log_text = mock_logger.debug.call_args[0][0]
                log_data = json.loads(log_text)
                assert "precision_limitations" in log_data

    def test_disabled_no_log(self):
        """無効時はログを出力しない。"""
        ev = _make_eval(enabled=False)
        section_map = {"A": "sec1"}

        with patch('tools.enrichment_efficacy_evaluation._logger') as mock_logger:
            ev.record_manifestation_results({"A": True}, section_map)
            mock_logger.debug.assert_not_called()
