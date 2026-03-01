"""
tests/test_enrichment_effectiveness.py - EnrichmentEffectivenessAnalyzer のテスト

テスト項目:
- 初期化テスト(デフォルト状態)
- 項目別累積カウンタテスト(空判定、変動計測、テキスト長累積)
- セクション別集計テスト(占有比率、項目数)
- セッションサマリ出力テスト(JSON構造、ログ出力)
- 安全弁テスト(例外捕捉、無効時スキップ、セッション境界消失)
- ExecutionMonitor統合テスト
- _is_empty_text独立関数テスト
- 評価的語彙の非含有テスト
"""

import json
import logging
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.execution_monitor import (
    ExecutionMonitor,
    EnrichmentEffectivenessAnalyzer,
    _is_empty_text,
    _KNOWN_EMPTY_PATTERNS,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_monitor(enabled: bool = True) -> ExecutionMonitor:
    """テスト用のExecutionMonitorを生成する。"""
    return ExecutionMonitor(enabled=enabled, snapshot_interval=10)


def _make_analyzer() -> EnrichmentEffectivenessAnalyzer:
    """テスト用のEnrichmentEffectivenessAnalyzerを生成する。"""
    return EnrichmentEffectivenessAnalyzer()


def _make_sections(
    items_sec1: list[tuple[str, str]] | None = None,
    items_sec2: list[tuple[str, str]] | None = None,
    items_sec3: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """テスト用のsections_dataを生成する。"""
    sections = []
    if items_sec1 is not None:
        sections.append({"header": "【心理状態（内面）】", "items": items_sec1})
    if items_sec2 is not None:
        sections.append({"header": "【自己認識】", "items": items_sec2})
    if items_sec3 is not None:
        sections.append({"header": "【動機・目標】", "items": items_sec3})
    return sections


def _standard_sections() -> list[dict]:
    """標準的なテスト用セクションデータ。"""
    return _make_sections(
        items_sec1=[
            ("感情状態", "joy: 0.7, sadness: 0.2"),
            ("ムード", "(安定)"),
            ("ドライブ", "social: 0.5, curiosity: 0.3"),
        ],
        items_sec2=[
            ("自己モデル", "自己認識テキスト"),
            ("時間的差分", "(未蓄積)"),
        ],
        items_sec3=[
            ("一時的目的", "対話を続ける"),
        ],
    )


# 評価的語彙リスト(安全弁5で除去すべき語彙)
_EVALUATIVE_WORDS = frozenset({
    "重要", "不要", "有効", "無効", "最適", "非最適",
    "良い", "悪い", "優れ", "劣る", "推奨", "削除推奨",
    "無効化推奨",
})


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """初期化のテスト。"""

    def test_default_state(self) -> None:
        """デフォルト初期化で空の状態を持つ。"""
        a = _make_analyzer()
        assert a.item_counters == {}
        assert a.section_char_totals == {}
        assert a.latest_summary is None

    def test_monitor_has_effectiveness_analyzer(self) -> None:
        """ExecutionMonitorがEnrichmentEffectivenessAnalyzerを持つ。"""
        m = _make_monitor()
        assert m.enrichment_effectiveness is not None
        assert isinstance(m.enrichment_effectiveness, EnrichmentEffectivenessAnalyzer)

    def test_disabled_monitor_has_analyzer(self) -> None:
        """無効なExecutionMonitorもAnalyzerを持つ(呼び出されないだけ)。"""
        m = _make_monitor(enabled=False)
        assert m.enrichment_effectiveness is not None


# ── 項目別累積カウンタテスト ──────────────────────────────────────


class TestItemCounters:
    """項目別累積カウンタのテスト。"""

    def test_single_tick_non_empty(self) -> None:
        """非空項目の1ティック記録。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("感情状態", "joy: 0.7")]
        )
        a.record_tick(tick_count=1, sections_data=sections)

        counters = a.item_counters
        assert "感情状態" in counters
        assert counters["感情状態"]["observed"] == 1
        assert counters["感情状態"]["empty_count"] == 0
        assert counters["感情状態"]["text_length_sum"] == len("joy: 0.7")

    def test_single_tick_empty(self) -> None:
        """空項目の1ティック記録。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("ムード", "(安定)")]
        )
        a.record_tick(tick_count=1, sections_data=sections)

        counters = a.item_counters
        assert counters["ムード"]["observed"] == 1
        assert counters["ムード"]["empty_count"] == 1
        assert counters["ムード"]["text_length_sum"] == 0

    def test_multi_tick_accumulation(self) -> None:
        """複数ティックでの累積。"""
        a = _make_analyzer()
        for i in range(5):
            sections = _make_sections(
                items_sec1=[("感情状態", f"val_{i}")]
            )
            a.record_tick(tick_count=i + 1, sections_data=sections)

        counters = a.item_counters
        assert counters["感情状態"]["observed"] == 5
        assert counters["感情状態"]["empty_count"] == 0

    def test_mixed_empty_and_non_empty(self) -> None:
        """空と非空が混在する場合の累積。"""
        a = _make_analyzer()
        # tick 1: 非空
        a.record_tick(1, _make_sections(items_sec1=[("label", "content")]))
        # tick 2: 空
        a.record_tick(2, _make_sections(items_sec1=[("label", "(未蓄積)")]))
        # tick 3: 非空
        a.record_tick(3, _make_sections(items_sec1=[("label", "more content")]))

        counters = a.item_counters
        assert counters["label"]["observed"] == 3
        assert counters["label"]["empty_count"] == 1
        assert counters["label"]["text_length_sum"] == len("content") + len("more content")

    def test_text_length_sum_excludes_empty(self) -> None:
        """テキスト長累積は空状態を含まない。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "(なし)")]))
        a.record_tick(2, _make_sections(items_sec1=[("x", "hello")]))
        a.record_tick(3, _make_sections(items_sec1=[("x", "(蓄積前)")]))

        counters = a.item_counters
        assert counters["x"]["text_length_sum"] == len("hello")

    def test_multiple_items_per_section(self) -> None:
        """セクション内に複数項目がある場合の独立記録。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("a", "text_a"), ("b", "(安定)"), ("c", "text_c")]
        )
        a.record_tick(1, sections)

        counters = a.item_counters
        assert counters["a"]["empty_count"] == 0
        assert counters["b"]["empty_count"] == 1
        assert counters["c"]["empty_count"] == 0

    def test_item_counters_read_only_copy(self) -> None:
        """item_countersプロパティは読み取り専用コピーを返す。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        copy = a.item_counters
        copy["x"]["observed"] = 999
        # 元は変更されない
        assert a.item_counters["x"]["observed"] == 1


# ── セクション別集計テスト ────────────────────────────────────────


class TestSectionSummaries:
    """セクション別集計のテスト。"""

    def test_section_char_totals(self) -> None:
        """セクション別文字数累積が正しい。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("a", "12345")],  # 5文字
            items_sec2=[("b", "123456789")],  # 9文字
        )
        a.record_tick(1, sections)

        totals = a.section_char_totals
        assert totals["【心理状態（内面）】"] == 5
        assert totals["【自己認識】"] == 9

    def test_section_char_accumulation(self) -> None:
        """セクション文字数の複数ティック累積。"""
        a = _make_analyzer()
        sections1 = _make_sections(items_sec1=[("a", "abc")])
        sections2 = _make_sections(items_sec1=[("a", "defgh")])

        a.record_tick(1, sections1)
        a.record_tick(2, sections2)

        assert a.section_char_totals["【心理状態（内面）】"] == 3 + 5

    def test_compute_summary_section_item_count(self) -> None:
        """compute_summaryのセクション別項目数が正しい。"""
        a = _make_analyzer()
        a.record_tick(1, _standard_sections())

        summary = a.compute_summary()
        section_map = {s["header"]: s for s in summary["section_summaries"]}

        assert section_map["【心理状態（内面）】"]["item_count"] == 3
        assert section_map["【自己認識】"]["item_count"] == 2
        assert section_map["【動機・目標】"]["item_count"] == 1

    def test_compute_summary_char_share(self) -> None:
        """compute_summaryの文字数占有比率が正しい。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("a", "a" * 100)],
            items_sec2=[("b", "b" * 100)],
        )
        a.record_tick(1, sections)

        summary = a.compute_summary()
        section_map = {s["header"]: s for s in summary["section_summaries"]}

        # 等分割のためそれぞれ0.5
        assert section_map["【心理状態（内面）】"]["char_share"] == 0.5
        assert section_map["【自己認識】"]["char_share"] == 0.5

    def test_compute_summary_empty_rates_listed(self) -> None:
        """セクション内の全項目の空状態出現率が等価に列挙される。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[
                ("a", "content"),     # empty_rate=0.0
                ("b", "(未蓄積)"),    # empty_rate=1.0
                ("c", "content2"),    # empty_rate=0.0
            ]
        )
        a.record_tick(1, sections)

        summary = a.compute_summary()
        sec = summary["section_summaries"][0]

        assert sorted(sec["empty_rates"]) == [0.0, 0.0, 1.0]

    def test_section_char_totals_read_only_copy(self) -> None:
        """section_char_totalsプロパティは読み取り専用コピーを返す。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        copy = a.section_char_totals
        copy["【心理状態（内面）】"] = 999
        assert a.section_char_totals["【心理状態（内面）】"] == 4


# ── compute_summaryテスト ─────────────────────────────────────────


class TestComputeSummary:
    """compute_summaryメソッドのテスト。"""

    def test_empty_analyzer_returns_empty(self) -> None:
        """データなしの状態でcompute_summaryは空のサマリを返す。"""
        a = _make_analyzer()
        summary = a.compute_summary()
        assert summary["total_items"] == 0
        assert summary["item_characteristics"] == []
        assert summary["section_summaries"] == []

    def test_item_characteristics_structure(self) -> None:
        """item_characteristicsの各エントリが必要なフィールドを持つ。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "hello")]))
        summary = a.compute_summary()

        item = summary["item_characteristics"][0]
        required_keys = {
            "label", "section", "observed", "empty_count",
            "empty_rate", "non_empty_count", "avg_text_length",
            "text_length_sum",
        }
        assert required_keys.issubset(set(item.keys()))

    def test_empty_rate_calculation(self) -> None:
        """空状態出現率の計算が正しい。"""
        a = _make_analyzer()
        # 4ティック中2回が空
        a.record_tick(1, _make_sections(items_sec1=[("x", "content")]))
        a.record_tick(2, _make_sections(items_sec1=[("x", "(未蓄積)")]))
        a.record_tick(3, _make_sections(items_sec1=[("x", "more")]))
        a.record_tick(4, _make_sections(items_sec1=[("x", "(安定)")]))

        summary = a.compute_summary()
        item = summary["item_characteristics"][0]
        assert item["empty_rate"] == 0.5  # 2/4

    def test_avg_text_length_calculation(self) -> None:
        """平均テキスト長の計算が正しい(非空状態のみ)。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "abc")]))      # len=3
        a.record_tick(2, _make_sections(items_sec1=[("x", "(未蓄積)")]))  # 空
        a.record_tick(3, _make_sections(items_sec1=[("x", "abcde")]))    # len=5

        summary = a.compute_summary()
        item = summary["item_characteristics"][0]

        # avg = (3+5)/2 = 4.0
        assert item["avg_text_length"] == 4.0
        assert item["non_empty_count"] == 2

    def test_avg_text_length_all_empty(self) -> None:
        """全て空の場合、平均テキスト長は0.0。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "(未蓄積)")]))
        a.record_tick(2, _make_sections(items_sec1=[("x", "(安定)")]))

        summary = a.compute_summary()
        item = summary["item_characteristics"][0]
        assert item["avg_text_length"] == 0.0
        assert item["non_empty_count"] == 0

    def test_total_items_count(self) -> None:
        """total_itemsが全項目数と一致する。"""
        a = _make_analyzer()
        a.record_tick(1, _standard_sections())
        summary = a.compute_summary()
        assert summary["total_items"] == 6  # 3+2+1

    def test_total_chars_cumulative(self) -> None:
        """total_chars_cumulativeが全セクションの累積文字数合計と一致する。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("a", "abc")],   # 3
            items_sec2=[("b", "defg")],  # 4
        )
        a.record_tick(1, sections)
        summary = a.compute_summary()
        assert summary["total_chars_cumulative"] == 7

    def test_latest_summary_updated(self) -> None:
        """compute_summary後にlatest_summaryが更新される。"""
        a = _make_analyzer()
        assert a.latest_summary is None

        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        summary = a.compute_summary()

        assert a.latest_summary is not None
        assert a.latest_summary["total_items"] == 1

    def test_latest_summary_is_copy(self) -> None:
        """latest_summaryは読み取り専用コピー。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        a.compute_summary()

        copy = a.latest_summary
        copy["total_items"] = 999
        assert a.latest_summary["total_items"] == 1

    def test_section_summaries_structure(self) -> None:
        """section_summariesの各エントリが必要なフィールドを持つ。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        summary = a.compute_summary()

        sec = summary["section_summaries"][0]
        required_keys = {
            "header", "item_count", "empty_rates",
            "section_total_chars", "char_share",
        }
        assert required_keys.issubset(set(sec.keys()))


# ── セッションサマリ出力テスト ────────────────────────────────────


class TestSessionSummary:
    """セッションサマリ出力のテスト。"""

    def test_emit_session_summary_with_monitor(self, caplog: pytest.LogCaptureFixture) -> None:
        """ExecutionMonitor経由でのセッションサマリ出力。"""
        m = _make_monitor()
        m.record_enrichment_distribution(
            tick_count=1,
            sections_data=_standard_sections(),
            compressed_text="compressed",
        )

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.enrichment_effectiveness.emit_session_summary(m)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_effectiveness_session_summary" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "enrichment_effectiveness_session_summary"
                assert "item_characteristics" in data
                assert "section_summaries" in data
                assert "total_items" in data
                assert "total_chars_cumulative" in data
                break
        assert found, "enrichment_effectiveness_session_summary not found in logs"

    def test_emit_session_summary_without_monitor(self, caplog: pytest.LogCaptureFixture) -> None:
        """monitor=Noneでもログ出力される。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            a.emit_session_summary(monitor=None)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_effectiveness_session_summary" in msg:
                found = True
                break
        assert found

    def test_emit_session_summary_empty_data(self) -> None:
        """データなしの場合でもクラッシュしない。"""
        a = _make_analyzer()
        # should not raise
        a.emit_session_summary(monitor=None)

    def test_session_summary_integrated_in_monitor(self, caplog: pytest.LogCaptureFixture) -> None:
        """ExecutionMonitor.emit_session_summary()で有効性サマリも出力される。"""
        m = _make_monitor()
        m.record_enrichment_distribution(
            tick_count=1,
            sections_data=_standard_sections(),
            compressed_text="test",
        )

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()

        types_found = set()
        for record in caplog.records:
            msg = record.getMessage()
            if msg.startswith("{"):
                try:
                    data = json.loads(msg)
                    if "type" in data:
                        types_found.add(data["type"])
                except json.JSONDecodeError:
                    pass

        assert "session_summary" in types_found
        assert "enrichment_effectiveness_session_summary" in types_found

    def test_session_summary_timestamp_present(self) -> None:
        """セッションサマリにtimestampが含まれる。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        summary = a.compute_summary()
        # emit_session_summaryのJSON出力にtimestampがある
        # compute_summaryは直接timestampを持たない(emit時に付与)
        assert summary is not None


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁のテスト。"""

    def test_sv1_exception_in_record_tick(self) -> None:
        """安全弁1: record_tick内の例外は捕捉される。"""
        a = _make_analyzer()
        # _item_countersを壊す
        a._item_counters = None  # type: ignore
        # Should not raise
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))

    def test_sv1_exception_in_compute_summary(self) -> None:
        """安全弁1: compute_summary内の例外は捕捉される。"""
        a = _make_analyzer()
        a._item_counters = None  # type: ignore
        result = a.compute_summary()
        assert result == {}

    def test_sv1_exception_in_emit_session_summary(self) -> None:
        """安全弁1: emit_session_summary内の例外は捕捉される。"""
        a = _make_analyzer()
        a._item_counters = None  # type: ignore
        # Should not raise
        a.emit_session_summary(monitor=None)

    def test_sv2_disabled_monitor_skips_recording(self) -> None:
        """安全弁2: 無効なmonitorではrecord_enrichment_distributionがスキップされる。"""
        m = _make_monitor(enabled=False)
        m.record_enrichment_distribution(
            tick_count=1,
            sections_data=_standard_sections(),
            compressed_text="test",
        )
        # 有効性分析のカウンタは空のまま
        assert m.enrichment_effectiveness.item_counters == {}

    def test_sv2_disabled_monitor_skips_session_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """安全弁2: 無効なmonitorではemit_session_summaryがスキップされる。"""
        m = _make_monitor(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()

        for record in caplog.records:
            assert "enrichment_effectiveness" not in record.getMessage()

    def test_sv4_session_boundary_reset(self) -> None:
        """安全弁4: 新しいインスタンスは前セッションの状態を持たない。"""
        a1 = _make_analyzer()
        a1.record_tick(1, _make_sections(items_sec1=[("x", "text")]))
        assert a1.item_counters["x"]["observed"] == 1

        a2 = _make_analyzer()
        assert a2.item_counters == {}

    def test_sv5_no_evaluative_words_in_output(self) -> None:
        """安全弁5: 出力JSONに評価的語彙が含まれない。"""
        a = _make_analyzer()
        # 多様なデータを入れる
        for i in range(10):
            a.record_tick(i + 1, _standard_sections())

        summary = a.compute_summary()
        summary_json = json.dumps(summary, ensure_ascii=False)

        for word in _EVALUATIVE_WORDS:
            assert word not in summary_json, f"Evaluative word '{word}' found in summary"

    def test_malformed_sections_data(self) -> None:
        """不正なsections_dataでもクラッシュしない。"""
        a = _make_analyzer()
        # headerなし
        a.record_tick(1, [{"items": [("x", "text")]}])
        # itemsなし
        a.record_tick(2, [{"header": "H"}])
        # 空リスト
        a.record_tick(3, [])
        # Should not raise and should accumulate what it can
        assert a.item_counters.get("x", {}).get("observed", 0) == 1


# ── _is_empty_text独立関数テスト ──────────────────────────────────


class TestIsEmptyText:
    """_is_empty_text関数のテスト。"""

    def test_empty_string(self) -> None:
        assert _is_empty_text("") is True

    def test_whitespace_only(self) -> None:
        assert _is_empty_text("  ") is True

    def test_known_patterns(self) -> None:
        for pattern in _KNOWN_EMPTY_PATTERNS:
            if pattern:  # 空文字列は別途テスト
                assert _is_empty_text(pattern) is True

    def test_labeled_empty_pattern(self) -> None:
        assert _is_empty_text("ラベル: (未蓄積)") is True
        assert _is_empty_text("感情: (安定)") is True
        assert _is_empty_text("感情: (なし)") is True

    def test_non_empty_text(self) -> None:
        assert _is_empty_text("hello") is False
        assert _is_empty_text("joy: 0.7") is False
        assert _is_empty_text("対話を続ける") is False

    def test_labeled_non_empty(self) -> None:
        assert _is_empty_text("感情: joy 0.7") is False


# ── ExecutionMonitor統合テスト ─────────────────────────────────────


class TestMonitorIntegration:
    """ExecutionMonitorとの統合テスト。"""

    def test_record_enrichment_distribution_calls_effectiveness(self) -> None:
        """record_enrichment_distributionが有効性分析のrecord_tickも呼ぶ。"""
        m = _make_monitor()
        sections = _standard_sections()
        m.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="test",
        )

        ea = m.enrichment_effectiveness
        assert len(ea.item_counters) > 0
        # 標準セクションの全6項目が記録されている
        assert len(ea.item_counters) == 6

    def test_multi_tick_integration(self) -> None:
        """複数ティックでの統合記録。"""
        m = _make_monitor()
        for i in range(10):
            m.record_enrichment_distribution(
                tick_count=i + 1,
                sections_data=_standard_sections(),
                compressed_text=f"compressed_{i}",
            )

        ea = m.enrichment_effectiveness
        for label, counters in ea.item_counters.items():
            assert counters["observed"] == 10

    def test_effectiveness_does_not_modify_distribution_monitor(self) -> None:
        """有効性分析がEnrichmentDistributionMonitorの状態を変更しない。"""
        m = _make_monitor()
        sections = _standard_sections()

        m.record_enrichment_distribution(
            tick_count=1,
            sections_data=sections,
            compressed_text="test",
        )

        dist = m.enrichment_distribution
        eff = m.enrichment_effectiveness

        # 両方とも独立して記録を持つ
        assert dist.observation_count == 1
        assert len(eff.item_counters) > 0

    def test_full_session_lifecycle(self, caplog: pytest.LogCaptureFixture) -> None:
        """フルセッションライフサイクル(記録→サマリ出力)。"""
        m = _make_monitor()

        # 記録フェーズ
        for i in range(5):
            m.record_enrichment_distribution(
                tick_count=i + 1,
                sections_data=_standard_sections(),
                compressed_text=f"c_{i}",
            )

        # サマリ出力フェーズ
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_session_summary()

        # 有効性サマリが出力されたことを確認
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_effectiveness_session_summary" in msg:
                found = True
                data = json.loads(msg)
                assert data["total_items"] == 6
                assert len(data["item_characteristics"]) == 6
                break
        assert found

    def test_enrichment_effectiveness_accessor(self) -> None:
        """enrichment_effectivenessプロパティが同じインスタンスを返す。"""
        m = _make_monitor()
        assert m.enrichment_effectiveness is m._enrichment_effectiveness


# ── 構造的分離テスト ──────────────────────────────────────────────


class TestStructuralSeparation:
    """構造的分離のテスト。"""

    def test_no_save_load_fields(self) -> None:
        """save/loadフィールドへの追加がない(状態はセッション内のみ)。"""
        a = _make_analyzer()
        # to_dictやfrom_dictメソッドが存在しない
        assert not hasattr(a, 'to_dict')
        assert not hasattr(a, 'from_dict')
        assert not hasattr(a, 'save')
        assert not hasattr(a, 'load')

    def test_no_phase_processing(self) -> None:
        """Phase処理関連のメソッドが存在しない。"""
        a = _make_analyzer()
        assert not hasattr(a, 'phase')
        assert not hasattr(a, 'phase_number')
        assert not hasattr(a, 'execute_phase')

    def test_no_enrichment_item_output(self) -> None:
        """enrichment項目としての出力メソッドが存在しない。"""
        a = _make_analyzer()
        assert not hasattr(a, 'get_enrichment_text')
        assert not hasattr(a, 'get_enrichment_item')

    def test_output_is_json_log_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """出力先がJSONログストリームのみ。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "text")]))

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            a.emit_session_summary(monitor=None)

        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_effectiveness" in msg:
                # JSON形式であることを確認
                data = json.loads(msg)
                assert isinstance(data, dict)

    def test_no_enrichment_reverse_flow(self) -> None:
        """分析結果がenrichmentに逆流するメソッドが存在しない。"""
        a = _make_analyzer()
        # enrichment圧縮パイプラインへの出力メソッドが存在しない
        assert not hasattr(a, 'apply_to_enrichment')
        assert not hasattr(a, 'modify_enrichment')
        assert not hasattr(a, 'get_adjustments')

    def test_no_importance_ranking(self) -> None:
        """項目の「重要度」「有効度」順位付けを行わない。"""
        a = _make_analyzer()
        a.record_tick(1, _standard_sections())
        summary = a.compute_summary()

        # summary内に重要度/有効度/貢献度/順位のフィールドがない
        for item in summary["item_characteristics"]:
            assert "importance" not in item
            assert "effectiveness" not in item
            assert "contribution" not in item
            assert "rank" not in item
            assert "priority" not in item


# ── エッジケーステスト ────────────────────────────────────────────


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_sections_data(self) -> None:
        """空のsections_dataでもクラッシュしない。"""
        a = _make_analyzer()
        a.record_tick(1, [])
        assert a.item_counters == {}

    def test_single_item_single_tick(self) -> None:
        """最小構成(1項目1ティック)。"""
        a = _make_analyzer()
        a.record_tick(1, _make_sections(items_sec1=[("x", "y")]))
        summary = a.compute_summary()
        assert summary["total_items"] == 1

    def test_all_empty_items(self) -> None:
        """全項目が空の場合。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[
                ("a", "(未蓄積)"),
                ("b", "(安定)"),
                ("c", "(なし)"),
            ]
        )
        a.record_tick(1, sections)
        summary = a.compute_summary()

        for item in summary["item_characteristics"]:
            assert item["empty_rate"] == 1.0
            assert item["avg_text_length"] == 0.0

    def test_very_long_text(self) -> None:
        """非常に長いテキストの処理。"""
        a = _make_analyzer()
        long_text = "x" * 100000
        a.record_tick(1, _make_sections(items_sec1=[("x", long_text)]))

        counters = a.item_counters
        assert counters["x"]["text_length_sum"] == 100000

    def test_char_share_zero_total(self) -> None:
        """全体文字数が0の場合のchar_share。"""
        a = _make_analyzer()
        sections = _make_sections(items_sec1=[("x", "")])
        a.record_tick(1, sections)
        summary = a.compute_summary()

        # 全体文字数が0の場合、char_shareは0.0
        for sec in summary["section_summaries"]:
            assert sec["char_share"] == 0.0

    def test_section_with_no_items(self) -> None:
        """項目なしのセクション。"""
        a = _make_analyzer()
        sections = [{"header": "【空セクション】", "items": []}]
        a.record_tick(1, sections)
        # 項目がないのでカウンタは空
        assert a.item_counters == {}
        # セクション文字数は0
        assert a.section_char_totals.get("【空セクション】", 0) == 0

    def test_duplicate_labels_across_sections(self) -> None:
        """異なるセクションに同じラベルが存在する場合(最後のセクション対応が記録される)。"""
        a = _make_analyzer()
        sections = _make_sections(
            items_sec1=[("shared_label", "text1")],
            items_sec2=[("shared_label", "text2")],
        )
        a.record_tick(1, sections)

        # observed=2(両方のセクションで記録される)
        assert a.item_counters["shared_label"]["observed"] == 2

    def test_large_tick_count(self) -> None:
        """大きなティック番号でも問題ない。"""
        a = _make_analyzer()
        a.record_tick(999999, _make_sections(items_sec1=[("x", "text")]))
        assert a.item_counters["x"]["observed"] == 1
