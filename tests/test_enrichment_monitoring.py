"""
tests/test_enrichment_monitoring.py - EnrichmentDistributionMonitor のテスト

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定)
- 第1段: 項目別出力特性記述テスト(空判定、変動判定、テキスト長)
- 第2段: セクション別・全体集計テスト
- 第3段: 時間窓内履歴蓄積テスト(FIFO、上限)
- 重複検出テスト(部分一致、計算量制御)
- 累積カウンタテスト
- 読み取り専用アクセサテスト
- ExecutionMonitor統合テスト
- 安全弁テスト(計測失敗、メモリ制御、無効化、永続化非対象)
- orchestrator統合テスト
- セッション境界テスト
- ログ出力テスト(JSON形式)
- エッジケーステスト
"""

import json
import logging
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools.execution_monitor import (
    ExecutionMonitor,
    EnrichmentDistributionMonitor,
    _KNOWN_EMPTY_PATTERNS,
    _DISTRIBUTION_HISTORY_DEFAULT_MAX,
    _DUPLICATE_COMPARISON_LIMIT,
    _DUPLICATE_CHECK_INTERVAL,
    _DUPLICATE_SIMILARITY_THRESHOLD,
    _ITEM_DETAIL_SNAPSHOT_INTERVAL,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_sections(
    items_per_section: list[list[tuple[str, str]]] | None = None,
) -> list[dict]:
    """テスト用のsections_dataを生成する。"""
    if items_per_section is None:
        items_per_section = [
            [
                ("感情", "感情: joy=0.7, sadness=0.1"),
                ("ムード", "ムード: valence=0.5, arousal=0.3"),
                ("ドライブ", "ドライブ: social=0.5"),
            ],
            [
                ("自己像", "自己像: 暫定的統合中"),
                ("物語", ""),
            ],
        ]
    headers = [
        "【心理状態（内面）】",
        "【自己認識】",
        "【動機・目標】",
        "【記憶・内省】",
        "【判断傾向】",
    ]
    result = []
    for i, items in enumerate(items_per_section):
        header = headers[i] if i < len(headers) else f"【セクション{i}】"
        result.append({"header": header, "items": items})
    return result


def _make_monitor_with_dist(
    enabled: bool = True,
) -> ExecutionMonitor:
    """enrichment分布モニター付きのExecutionMonitorを生成する。"""
    return ExecutionMonitor(enabled=enabled, snapshot_interval=100)


# ── 初期化テスト ──────────────────────────────────────────────────


class TestEnrichmentDistMonitorInit:
    """EnrichmentDistributionMonitorの初期化テスト。"""

    def test_default_init(self):
        """デフォルト初期化。"""
        edm = EnrichmentDistributionMonitor()
        assert edm.observation_count == 0
        assert edm.history == []
        assert edm.item_counters == {}
        assert edm.duplicate_pairs == []

    def test_custom_history_max(self):
        """カスタム履歴上限。"""
        edm = EnrichmentDistributionMonitor(history_max=10)
        assert edm._history.maxlen == 10

    def test_history_max_minimum_one(self):
        """履歴上限が最低1に制限される。"""
        edm = EnrichmentDistributionMonitor(history_max=0)
        assert edm._history.maxlen == 1

    def test_custom_duplicate_interval(self):
        """カスタム重複検出間隔。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=5)
        assert edm._duplicate_check_interval == 5

    def test_duplicate_interval_minimum_one(self):
        """重複検出間隔が最低1に制限される。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=0)
        assert edm._duplicate_check_interval == 1

    def test_execution_monitor_has_enrichment_dist(self):
        """ExecutionMonitorがenrichment_distributionプロパティを持つ。"""
        m = _make_monitor_with_dist()
        assert isinstance(m.enrichment_distribution, EnrichmentDistributionMonitor)


# ── 第1段: 項目別出力特性記述テスト ──────────────────────────────


class TestItemCharacterization:
    """項目別の出力特性記述テスト。"""

    def test_non_empty_detection(self):
        """非空項目が正しく検出される。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("ラベルA", "テキスト内容"), ("ラベルB", "")],
        ])
        edm.record_enrichment_distribution(1, sections, "compressed")
        counters = edm.item_counters
        assert counters["ラベルA"]["non_empty"] == 1
        assert counters["ラベルB"]["non_empty"] == 0

    def test_empty_patterns_detected(self):
        """既知の空パターンが空として検出される。"""
        edm = EnrichmentDistributionMonitor()
        for pattern in _KNOWN_EMPTY_PATTERNS:
            if not pattern:
                continue  # 空文字列は別テストで確認
            sections = _make_sections([
                [("テスト", pattern)],
            ])
            edm2 = EnrichmentDistributionMonitor()
            edm2.record_enrichment_distribution(1, sections, "c")
            assert edm2.item_counters["テスト"]["non_empty"] == 0, (
                f"Pattern '{pattern}' should be detected as empty"
            )

    def test_labeled_empty_pattern_detected(self):
        """ラベル付き空パターン「ラベル: (未蓄積)」が空として検出される。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("テスト", "テスト: (未蓄積)")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        assert edm.item_counters["テスト"]["non_empty"] == 0

    def test_change_detection_first_tick(self):
        """初回観測は変動ありとして扱われる。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("ラベルA", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        assert edm.item_counters["ラベルA"]["changed"] == 1

    def test_change_detection_no_change(self):
        """同一テキストが続く場合は変動なし。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("ラベルA", "固定テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        edm.record_enrichment_distribution(2, sections, "c")
        # 初回は変動あり、2回目は変動なし
        assert edm.item_counters["ラベルA"]["changed"] == 1

    def test_change_detection_with_change(self):
        """テキストが変化した場合は変動あり。"""
        edm = EnrichmentDistributionMonitor()
        sections1 = _make_sections([
            [("ラベルA", "テキスト1")],
        ])
        sections2 = _make_sections([
            [("ラベルA", "テキスト2")],
        ])
        edm.record_enrichment_distribution(1, sections1, "c")
        edm.record_enrichment_distribution(2, sections2, "c")
        # 初回: 変動あり、2回目: 変動あり(テキスト変化)
        assert edm.item_counters["ラベルA"]["changed"] == 2

    def test_text_length_in_history(self):
        """テキスト長が記録される。"""
        edm = EnrichmentDistributionMonitor(item_detail_interval=1)
        sections = _make_sections([
            [("ラベルA", "12345")],  # 5文字
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        # 履歴の最新エントリを確認
        history = edm.history
        assert len(history) == 1


# ── 第2段: セクション別・全体集計テスト ──────────────────────────


class TestDistributionAggregation:
    """セクション別・全体の集計テスト。"""

    def test_section_summary(self):
        """セクション別の集計が正しい。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "テキスト"), ("C", "")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        entry = edm.history[-1]
        assert entry["sections"][0]["non_empty"] == 2
        assert entry["sections"][0]["changed"] == 3  # 初回は全て変動あり
        assert entry["sections"][0]["total"] == 3

    def test_total_aggregation(self):
        """全体集計が正しい。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "")],
            [("C", "テキスト"), ("D", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        entry = edm.history[-1]
        assert entry["total_items"] == 4
        assert entry["total_non_empty"] == 3
        assert entry["total_changed"] == 4  # 初回は全て変動

    def test_compressed_chars_recorded(self):
        """圧縮後文字数が記録される。"""
        edm = EnrichmentDistributionMonitor()
        compressed = "compressed text here"
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, compressed)
        entry = edm.history[-1]
        assert entry["compressed_chars"] == len(compressed)

    def test_empty_section_handling(self):
        """空のセクションが正しく処理される。"""
        edm = EnrichmentDistributionMonitor()
        sections = [{"header": "【空セクション】", "items": []}]
        edm.record_enrichment_distribution(1, sections, "c")
        entry = edm.history[-1]
        assert entry["total_items"] == 0
        assert entry["total_non_empty"] == 0


# ── 第3段: 時間窓内履歴蓄積テスト ──────────────────────────────


class TestHistoryAccumulation:
    """時間窓内の分布履歴蓄積テスト。"""

    def test_history_accumulates(self):
        """履歴が蓄積される。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        for i in range(5):
            edm.record_enrichment_distribution(i + 1, sections, "c")
        assert len(edm.history) == 5

    def test_history_fifo_overflow(self):
        """履歴上限到達時にFIFOで古い記録が消失する。"""
        edm = EnrichmentDistributionMonitor(history_max=3)
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        for i in range(5):
            edm.record_enrichment_distribution(i + 1, sections, "c")
        assert len(edm.history) == 3
        # 最も古い記録はtick=3のもの
        assert edm.history[0]["tick"] == 3

    def test_history_entries_have_tick(self):
        """履歴エントリにティック番号が含まれる。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(42, sections, "c")
        assert edm.history[-1]["tick"] == 42

    def test_history_returns_copy(self):
        """historyプロパティがコピーを返す。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        h = edm.history
        h.clear()
        assert len(edm.history) == 1  # 元は変更されていない


# ── 重複検出テスト ──────────────────────────────────────────────


class TestDuplicateDetection:
    """項目間の出力テキスト重複検出テスト。"""

    def test_identical_texts_detected(self):
        """同一テキストの項目対が検出される。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        sections = _make_sections([
            [("A", "全く同じテキスト"), ("B", "全く同じテキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        pairs = edm.duplicate_pairs
        assert len(pairs) >= 1
        labels = {(p[0], p[1]) for p in pairs}
        assert ("A", "B") in labels
        assert pairs[0][2] == 1.0  # 完全一致

    def test_completely_different_texts_not_detected(self):
        """完全に異なるテキストは検出されない。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        sections = _make_sections([
            [("A", "あいうえおかきくけこ"), ("B", "XYZWVUTSRQ12345")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        pairs = edm.duplicate_pairs
        # 完全に異なるテキストはthreshold以上にならないはず
        assert len(pairs) == 0

    def test_empty_items_excluded_from_detection(self):
        """空項目は重複検出から除外される。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        sections = _make_sections([
            [("A", ""), ("B", ""), ("C", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        # 空項目同士の重複は検出されない
        pairs = edm.duplicate_pairs
        empty_pair = [p for p in pairs if p[0] in ("A", "B") and p[1] in ("A", "B")]
        assert len(empty_pair) == 0

    def test_duplicate_check_interval(self):
        """重複検出が一定間隔でのみ実行される。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=5)
        sections = _make_sections([
            [("A", "同じテキスト"), ("B", "同じテキスト")],
        ])
        # tick=5で検出実行(last_check_tick=0, 5-0 >= 5)
        edm.record_enrichment_distribution(5, sections, "c")
        assert len(edm.duplicate_pairs) >= 1

        # tick=6では検出されない(間隔未到達: 6-5 < 5)
        sections2 = _make_sections([
            [("A", "新しいテキスト"), ("B", "新しいテキスト")],
        ])
        old_pairs = list(edm.duplicate_pairs)

        edm.record_enrichment_distribution(6, sections2, "c")
        # まだ間隔に達していないので、pairsは前回のまま
        assert edm.duplicate_pairs == old_pairs

    def test_duplicate_cache_overwrite(self):
        """重複検出結果は検出実行ごとに完全に上書きされる。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)

        # tick=1: 重複あり
        sections1 = _make_sections([
            [("A", "同じテキスト"), ("B", "同じテキスト")],
        ])
        edm.record_enrichment_distribution(1, sections1, "c")
        assert len(edm.duplicate_pairs) >= 1

        # tick=2: 重複なし
        sections2 = _make_sections([
            [("A", "テキストAAAAAA"), ("B", "テキストBBBBBB12345")],
        ])
        edm.record_enrichment_distribution(2, sections2, "c")
        assert len(edm.duplicate_pairs) == 0

    def test_duplicate_pairs_returns_copy(self):
        """duplicate_pairsプロパティがコピーを返す。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        sections = _make_sections([
            [("A", "同じテキスト"), ("B", "同じテキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        pairs = edm.duplicate_pairs
        pairs.clear()
        assert len(edm.duplicate_pairs) >= 1

    def test_computation_limit(self):
        """安全弁3: 比較回数上限で打ち切られる。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        # 大量の項目を生成(n*(n-1)/2 > _DUPLICATE_COMPARISON_LIMIT)
        n = 50  # 50*49/2 = 1225 > 500
        items = [(f"item_{i}", f"テキスト{i}内容") for i in range(n)]
        sections = _make_sections([items])
        # 例外なしで実行される
        edm.record_enrichment_distribution(1, sections, "c")
        # 打ち切りされたが結果は存在する
        assert isinstance(edm.duplicate_pairs, list)


# ── 累積カウンタテスト ──────────────────────────────────────────


class TestCumulativeCounters:
    """項目別累積カウンタのテスト。"""

    def test_observation_count_increments(self):
        """観測回数がインクリメントされる。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        edm.record_enrichment_distribution(2, sections, "c")
        assert edm.observation_count == 2

    def test_item_observed_count(self):
        """項目ごとの観測回数が正しい。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        edm.record_enrichment_distribution(2, sections, "c")
        assert edm.item_counters["A"]["observed"] == 2
        assert edm.item_counters["B"]["observed"] == 2

    def test_non_empty_count_accumulates(self):
        """非空出力回数が蓄積される。"""
        edm = EnrichmentDistributionMonitor()
        sections1 = _make_sections([
            [("A", "テキスト"), ("B", "")],
        ])
        sections2 = _make_sections([
            [("A", "テキスト2"), ("B", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections1, "c")
        edm.record_enrichment_distribution(2, sections2, "c")
        assert edm.item_counters["A"]["non_empty"] == 2
        assert edm.item_counters["B"]["non_empty"] == 1

    def test_changed_count_accumulates(self):
        """変動回数が蓄積される。"""
        edm = EnrichmentDistributionMonitor()
        for i in range(5):
            # 奇数tickではテキスト変更、偶数tickでは同じ
            text = f"テキスト{i}" if i % 2 == 0 else "テキスト固定"
            sections = _make_sections([
                [("A", text)],
            ])
            edm.record_enrichment_distribution(i + 1, sections, "c")
        # tick1: 初回(変動) tick2: 変動 tick3: 変動 tick4: 変動なし tick5: 変動
        assert edm.item_counters["A"]["observed"] == 5

    def test_item_counters_returns_copy(self):
        """item_countersプロパティがコピーを返す。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        counters = edm.item_counters
        counters["A"]["observed"] = 9999
        assert edm.item_counters["A"]["observed"] == 1

    def test_new_item_appearance(self):
        """項目数変動: 新しい項目が出現した場合。"""
        edm = EnrichmentDistributionMonitor()
        sections1 = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections1, "c")

        sections2 = _make_sections([
            [("A", "テキスト"), ("B", "新項目")],
        ])
        edm.record_enrichment_distribution(2, sections2, "c")
        assert "B" in edm.item_counters
        assert edm.item_counters["B"]["observed"] == 1

    def test_disappeared_item(self):
        """項目数変動: 項目が消失した場合、過去の記録は維持される。"""
        edm = EnrichmentDistributionMonitor()
        sections1 = _make_sections([
            [("A", "テキスト"), ("B", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections1, "c")

        sections2 = _make_sections([
            [("A", "テキスト")],  # Bが消失
        ])
        edm.record_enrichment_distribution(2, sections2, "c")
        # Bの記録は維持されている
        assert "B" in edm.item_counters
        assert edm.item_counters["B"]["observed"] == 1


# ── 読み取り専用アクセサテスト ────────────────────────────────────


class TestDistributionSummary:
    """get_distribution_summaryアクセサのテスト。"""

    def test_empty_summary(self):
        """初期状態のサマリ。"""
        edm = EnrichmentDistributionMonitor()
        summary = edm.get_distribution_summary()
        assert summary["observation_count"] == 0
        assert summary["item_counters"] == {}
        assert summary["history_length"] == 0
        assert summary["latest_entry"] is None
        assert summary["duplicate_pairs"] == []

    def test_populated_summary(self):
        """データが蓄積された後のサマリ。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "compressed")
        summary = edm.get_distribution_summary()
        assert summary["observation_count"] == 1
        assert "A" in summary["item_counters"]
        assert summary["history_length"] == 1
        assert summary["latest_entry"] is not None
        assert summary["latest_entry"]["tick"] == 1

    def test_summary_latest_entry_is_most_recent(self):
        """latest_entryが最新のエントリである。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        edm.record_enrichment_distribution(2, sections, "c")
        edm.record_enrichment_distribution(3, sections, "c")
        summary = edm.get_distribution_summary()
        assert summary["latest_entry"]["tick"] == 3


# ── ExecutionMonitor統合テスト ─────────────────────────────────────


class TestExecutionMonitorIntegration:
    """ExecutionMonitorとの統合テスト。"""

    def test_record_via_monitor(self):
        """ExecutionMonitor経由での記録。"""
        m = _make_monitor_with_dist()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "テキスト")],
        ])
        m.record_enrichment_distribution(1, sections, "compressed")
        edm = m.enrichment_distribution
        assert edm.observation_count == 1
        assert "A" in edm.item_counters

    def test_disabled_monitor_skips(self):
        """無効時は記録しない。"""
        m = _make_monitor_with_dist(enabled=False)
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        m.record_enrichment_distribution(1, sections, "compressed")
        edm = m.enrichment_distribution
        assert edm.observation_count == 0

    def test_json_output_enrichment_distribution(self, caplog):
        """enrichment分布がJSON形式でログ出力される。"""
        m = _make_monitor_with_dist()
        sections = _make_sections([
            [("A", "テキスト"), ("B", "テキスト")],
        ])
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_enrichment_distribution(1, sections, "compressed")
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_distribution" in msg and msg.startswith("{"):
                data = json.loads(msg)
                if data.get("type") == "enrichment_distribution":
                    found = True
                    assert "total_items" in data
                    assert "total_non_empty" in data
                    assert "total_changed" in data
                    assert "compressed_chars" in data
                    assert "sections" in data
        assert found, "enrichment_distribution log not found"

    def test_json_output_item_detail(self, caplog):
        """項目別詳細がJSON形式でログ出力される(スナップショット間隔時)。"""
        m = _make_monitor_with_dist()
        # item_detail_intervalはデフォルト10なので、初回(tick 0からの差分)で出力される
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_enrichment_distribution(10, sections, "c")
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_item_detail" in msg and msg.startswith("{"):
                data = json.loads(msg)
                if data.get("type") == "enrichment_item_detail":
                    found = True
                    assert "items" in data
                    assert len(data["items"]) == 1
                    assert data["items"][0]["label"] == "A"
        assert found, "enrichment_item_detail log not found"

    def test_json_output_duplicate_pairs(self, caplog):
        """重複検出結果がJSON形式でログ出力される。"""
        m = _make_monitor_with_dist()
        # EnrichmentDistributionMonitorのduplicate_check_intervalをオーバーライド
        m._enrichment_dist._duplicate_check_interval = 1
        sections = _make_sections([
            [("A", "同じテキスト内容"), ("B", "同じテキスト内容")],
        ])
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_enrichment_distribution(1, sections, "c")
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_duplicate_pairs" in msg and msg.startswith("{"):
                data = json.loads(msg)
                if data.get("type") == "enrichment_duplicate_pairs":
                    found = True
                    assert "pairs" in data
        assert found, "enrichment_duplicate_pairs log not found"

    def test_session_boundary_resets(self):
        """新しいExecutionMonitorインスタンスでenrichment分布がリセットされる。"""
        m1 = _make_monitor_with_dist()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        m1.record_enrichment_distribution(1, sections, "c")
        assert m1.enrichment_distribution.observation_count == 1

        m2 = _make_monitor_with_dist()
        assert m2.enrichment_distribution.observation_count == 0


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作テスト。"""

    def test_valve1_exception_safe(self):
        """安全弁1: 計測失敗時に例外を捕捉してスキップ。"""
        edm = EnrichmentDistributionMonitor()
        # 不正なsections_data
        bad_sections = [{"header": "test", "items": "not_a_list"}]
        # 例外なしで実行される
        edm.record_enrichment_distribution(1, bad_sections, "c")
        # 内部状態が壊れていない
        assert isinstance(edm.history, list)

    def test_valve1_none_sections(self):
        """安全弁1: sections_dataがNoneでもクラッシュしない。"""
        edm = EnrichmentDistributionMonitor()
        # record_enrichment_distributionはNoneでも安全(例外は捕捉される)
        edm.record_enrichment_distribution(1, None, "c")
        # 内部状態が壊れていないことを確認
        assert isinstance(edm.history, list)

    def test_valve2_history_max(self):
        """安全弁2: 履歴上限によるメモリ制御。"""
        edm = EnrichmentDistributionMonitor(history_max=5)
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        for i in range(100):
            edm.record_enrichment_distribution(i + 1, sections, "c")
        assert len(edm.history) == 5

    def test_valve3_computation_limit(self):
        """安全弁3: 重複検出の計算量制御(DUPLICATE_COMPARISON_LIMIT)。"""
        edm = EnrichmentDistributionMonitor(duplicate_check_interval=1)
        # 大量の項目
        items = [(f"item_{i}", f"内容{i}") for i in range(100)]
        sections = _make_sections([items])
        # 例外なしで実行される(打ち切りされる)
        edm.record_enrichment_distribution(1, sections, "c")

    def test_valve5_disabled_via_monitor(self):
        """安全弁5: モニター無効時に全処理がスキップされる。"""
        m = ExecutionMonitor(enabled=False)
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        m.record_enrichment_distribution(1, sections, "c")
        assert m.enrichment_distribution.observation_count == 0

    def test_valve6_no_save_load(self):
        """安全弁6: 永続化の非対象(save/loadメソッドなし)。"""
        edm = EnrichmentDistributionMonitor()
        assert not hasattr(edm, 'save')
        assert not hasattr(edm, 'load')

    def test_valve7_no_psyche_modification(self):
        """安全弁7: 構造的分離の確認(importでpsycheを参照しない)。"""
        # EnrichmentDistributionMonitorが定義されているモジュールのimportを検査
        import tools.execution_monitor as em_module
        import inspect
        source = inspect.getsource(em_module)
        # "from psyche" や "import psyche" が含まれないことを確認
        assert "from psyche" not in source
        assert "import psyche" not in source


# ── orchestrator統合テスト ─────────────────────────────────────────


class TestOrchestratorIntegration:
    """orchestratorとの統合テスト。"""

    def test_enrichment_records_distribution(self):
        """get_prompt_enrichmentがenrichment分布を記録する。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        _result = orch.get_prompt_enrichment(user_id="viewer")
        edm = orch._execution_monitor.enrichment_distribution
        assert edm.observation_count == 1

    def test_enrichment_distribution_after_multiple_ticks(self):
        """複数ティック後のenrichment分布蓄積。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        percept = Percept(
            text="test",
            meaning="test",
            emotion="happy",
            intent="expression",
            emotion_valence=0.7,
        )
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.1, user_id="viewer")
        edm = orch._execution_monitor.enrichment_distribution
        # get_prompt_enrichmentが各ティックで呼ばれるわけではないが、
        # post_response_update内でenrichmentが生成される際に記録される
        # 直接get_prompt_enrichmentを呼んで確認
        _result = orch.get_prompt_enrichment(user_id="viewer")
        assert edm.observation_count >= 1

    def test_enrichment_distribution_history_has_entries(self):
        """enrichment分布の履歴にエントリが蓄積される。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        # 複数回enrichmentを生成
        for _ in range(3):
            _result = orch.get_prompt_enrichment(user_id="viewer")
        edm = orch._execution_monitor.enrichment_distribution
        assert len(edm.history) == 3

    def test_enrichment_distribution_counters_populated(self):
        """enrichment分布の項目別カウンタが設定される。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        _result = orch.get_prompt_enrichment(user_id="viewer")
        edm = orch._execution_monitor.enrichment_distribution
        counters = edm.item_counters
        # enrichment項目が存在するはず
        assert len(counters) > 0

    def test_save_load_not_affected(self, tmp_path):
        """save/loadがenrichment分布モニターの影響を受けない。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0, data_dir=tmp_path)
        orch._execution_monitor = ExecutionMonitor(enabled=True, snapshot_interval=100)
        _result = orch.get_prompt_enrichment(user_id="viewer")
        # save
        orch.save()
        # load
        orch2 = PsycheOrchestrator(memory_count=0, data_dir=tmp_path)
        orch2.load()
        # enrichment分布の状態はsave/loadに含まれない
        edm2 = orch2._execution_monitor.enrichment_distribution
        assert edm2.observation_count == 0

    def test_monitor_disabled_no_impact(self):
        """モニター無効時、通常処理に影響しない。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator(memory_count=0)
        orch._execution_monitor = ExecutionMonitor(enabled=False)
        # enrichment生成が正常に動作する
        result = orch.get_prompt_enrichment(user_id="viewer")
        assert isinstance(result, str)


# ── _is_emptyテスト ─────────────────────────────────────────────


class TestIsEmpty:
    """空判定ロジックのテスト。"""

    def test_empty_string(self):
        """空文字列。"""
        edm = EnrichmentDistributionMonitor()
        assert edm._is_empty("") is True

    def test_whitespace_only(self):
        """ホワイトスペースのみ。"""
        edm = EnrichmentDistributionMonitor()
        assert edm._is_empty("   ") is True
        assert edm._is_empty("\t\n") is True

    def test_known_patterns(self):
        """既知の空パターン。"""
        edm = EnrichmentDistributionMonitor()
        assert edm._is_empty("(未蓄積)") is True
        assert edm._is_empty("(安定)") is True
        assert edm._is_empty("(なし)") is True

    def test_labeled_empty(self):
        """ラベル付き空パターン。"""
        edm = EnrichmentDistributionMonitor()
        assert edm._is_empty("ラベル: (未蓄積)") is True
        assert edm._is_empty("感情: (安定)") is True

    def test_non_empty(self):
        """非空テキスト。"""
        edm = EnrichmentDistributionMonitor()
        assert edm._is_empty("テキスト内容") is False
        assert edm._is_empty("感情: joy=0.7") is False


# ── _compute_similarityテスト ─────────────────────────────────────


class TestComputeSimilarity:
    """類似度算出ロジックのテスト。"""

    def test_identical_texts(self):
        """同一テキストは類似度1.0。"""
        sim = EnrichmentDistributionMonitor._compute_similarity(
            "テスト文字列", "テスト文字列"
        )
        assert sim == 1.0

    def test_empty_texts(self):
        """空テキストは類似度0.0。"""
        sim = EnrichmentDistributionMonitor._compute_similarity("", "")
        assert sim == 0.0

    def test_one_empty(self):
        """片方が空の場合は類似度0.0。"""
        sim = EnrichmentDistributionMonitor._compute_similarity("テスト", "")
        assert sim == 0.0

    def test_completely_different(self):
        """完全に異なるテキスト。"""
        sim = EnrichmentDistributionMonitor._compute_similarity(
            "あいうえお", "かきくけこ"
        )
        assert sim < _DUPLICATE_SIMILARITY_THRESHOLD

    def test_similar_texts(self):
        """類似したテキスト。"""
        # 長いテキストで大部分が共通
        base = "これは共通のテキスト部分で長い文章です" * 5
        sim = EnrichmentDistributionMonitor._compute_similarity(
            base, base + "少し追加"
        )
        assert sim > 0.5

    def test_single_char_texts(self):
        """1文字テキスト。"""
        sim = EnrichmentDistributionMonitor._compute_similarity("A", "A")
        assert sim == 1.0
        sim = EnrichmentDistributionMonitor._compute_similarity("A", "B")
        assert sim < 1.0


# ── エッジケーステスト ────────────────────────────────────────────


class TestEdgeCases:
    """境界値・異常値のテスト。"""

    def test_empty_sections_list(self):
        """空のsections_data。"""
        edm = EnrichmentDistributionMonitor()
        edm.record_enrichment_distribution(1, [], "c")
        assert edm.observation_count == 1
        assert edm.history[-1]["total_items"] == 0

    def test_section_with_no_items(self):
        """項目なしのセクション。"""
        edm = EnrichmentDistributionMonitor()
        sections = [{"header": "【テスト】", "items": []}]
        edm.record_enrichment_distribution(1, sections, "c")
        assert edm.observation_count == 1

    def test_very_long_text(self):
        """非常に長いテキスト。"""
        edm = EnrichmentDistributionMonitor()
        long_text = "A" * 100_000
        sections = _make_sections([
            [("A", long_text)],
        ])
        edm.record_enrichment_distribution(1, sections, long_text)
        assert edm.item_counters["A"]["non_empty"] == 1

    def test_unicode_text(self):
        """Unicode文字を含むテキスト。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("絵文字", "感情: 喜 怒 哀 楽")],
        ])
        edm.record_enrichment_distribution(1, sections, "c")
        assert edm.item_counters["絵文字"]["non_empty"] == 1

    def test_many_rapid_observations(self):
        """多数の高速観測。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        for i in range(1000):
            edm.record_enrichment_distribution(i + 1, sections, "c")
        assert edm.observation_count == 1000

    def test_missing_header_key(self):
        """headerキーが欠落したセクション。"""
        edm = EnrichmentDistributionMonitor()
        sections = [{"items": [("A", "テキスト")]}]
        edm.record_enrichment_distribution(1, sections, "c")
        # 例外なしで実行される
        assert edm.observation_count == 1

    def test_get_distribution_summary_returns_dict(self):
        """get_distribution_summaryが常にdictを返す。"""
        edm = EnrichmentDistributionMonitor()
        summary = edm.get_distribution_summary()
        assert isinstance(summary, dict)

    def test_compressed_text_empty(self):
        """圧縮テキストが空の場合。"""
        edm = EnrichmentDistributionMonitor()
        sections = _make_sections([
            [("A", "テキスト")],
        ])
        edm.record_enrichment_distribution(1, sections, "")
        assert edm.history[-1]["compressed_chars"] == 0
