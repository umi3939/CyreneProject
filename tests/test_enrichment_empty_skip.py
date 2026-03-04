"""
tests/test_enrichment_empty_skip.py - enrichment空項目スキップ判定のテスト

テスト対象: psyche/enrichment_compression.py (EmptySkipTracker)
設計書: design_enrichment_empty_skip.md

テスト項目:
- 基本動作: 空状態判定、連続カウンタ、スキップ判定
- 安全弁1: 再生成間隔の上限
- 安全弁2: 圧縮除外リスト（ALWAYS_FULL_LABELS）の項目はスキップ対象外
- 安全弁3: 初回起動時は全項目生成
- 安全弁4: 空/非空の判定基準の単一性（EMPTY_STATE_MARKERのみ）
- 安全弁5: 永続化非対象
- 安全弁6: 項目等価性の維持
- 即時復帰: 空→有データ遷移時のカウンタリセット
- フィードバック経路遮断
- エッジケース
"""

import pytest

from psyche.enrichment_compression import (
    EmptySkipTracker,
    EMPTY_STATE_MARKER,
    ALWAYS_FULL_LABELS,
    SKIP_REGEN_MAX_INTERVAL,
    SKIP_CONSECUTIVE_EMPTY_THRESHOLD,
    is_empty_state_text,
    normalize_empty_state,
)


# =============================================================================
# 定数の検証
# =============================================================================

class TestConstants:
    """設計書で定義された定数の検証。"""

    def test_skip_regen_max_interval_is_positive(self):
        """再生成間隔の上限は正の整数。"""
        assert SKIP_REGEN_MAX_INTERVAL > 0
        assert isinstance(SKIP_REGEN_MAX_INTERVAL, int)

    def test_skip_consecutive_empty_threshold_is_positive(self):
        """連続空閾値は正の整数。"""
        assert SKIP_CONSECUTIVE_EMPTY_THRESHOLD > 0
        assert isinstance(SKIP_CONSECUTIVE_EMPTY_THRESHOLD, int)

    def test_threshold_less_than_max_interval(self):
        """閾値は再生成間隔の上限より小さい（意味のある範囲）。"""
        assert SKIP_CONSECUTIVE_EMPTY_THRESHOLD < SKIP_REGEN_MAX_INTERVAL


# =============================================================================
# EmptySkipTracker 基本動作テスト
# =============================================================================

class TestEmptySkipTrackerBasic:
    """EmptySkipTrackerの基本動作。"""

    def test_initial_state(self):
        """初期状態では全てのカウンタがゼロ。"""
        tracker = EmptySkipTracker()
        assert tracker.get_consecutive_empty_count("任意ラベル") == 0
        assert tracker.get_last_regen_tick("任意ラベル") == 0
        assert tracker.get_skip_count() == 0

    def test_first_tick_no_skip(self):
        """安全弁3: 初回ティックでは全項目がスキップされない。"""
        tracker = EmptySkipTracker()
        # 初回はmark_first_tick_doneが呼ばれていない
        assert tracker.should_skip("テスト", 0) is False
        assert tracker.should_skip("テスト", 1) is False

    def test_mark_first_tick_done(self):
        """mark_first_tick_done後にスキップ判定が有効化される。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        # まだカウンタがゼロなのでスキップされない
        assert tracker.should_skip("テスト", 1) is False

    def test_empty_state_increments_counter(self):
        """空状態テキストでカウンタが増加する。"""
        tracker = EmptySkipTracker()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        tracker.update_after_generation("テスト", empty_text, 0)
        assert tracker.get_consecutive_empty_count("テスト") == 1
        tracker.update_after_generation("テスト", empty_text, 1)
        assert tracker.get_consecutive_empty_count("テスト") == 2

    def test_non_empty_resets_counter(self):
        """有データテキストでカウンタがリセットされる。"""
        tracker = EmptySkipTracker()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        tracker.update_after_generation("テスト", empty_text, 0)
        tracker.update_after_generation("テスト", empty_text, 1)
        assert tracker.get_consecutive_empty_count("テスト") == 2
        # 有データで即座にリセット
        tracker.update_after_generation("テスト", "テスト: value=0.5", 2)
        assert tracker.get_consecutive_empty_count("テスト") == 0

    def test_skip_after_threshold(self):
        """連続空がSKIP_CONSECUTIVE_EMPTY_THRESHOLD回に達するとスキップ。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("テスト", empty_text, tick)
        assert tracker.get_consecutive_empty_count("テスト") == SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        # 閾値到達後はスキップ
        assert tracker.should_skip("テスト", SKIP_CONSECUTIVE_EMPTY_THRESHOLD) is True

    def test_no_skip_below_threshold(self):
        """連続空が閾値未満ではスキップしない。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1):
            tracker.update_after_generation("テスト", empty_text, tick)
        assert tracker.should_skip("テスト", SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1) is False

    def test_last_regen_tick_updated(self):
        """update_after_generationで最終再生成ティックが更新される。"""
        tracker = EmptySkipTracker()
        tracker.update_after_generation("テスト", "data", 5)
        assert tracker.get_last_regen_tick("テスト") == 5
        tracker.update_after_generation("テスト", "data", 10)
        assert tracker.get_last_regen_tick("テスト") == 10


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    """設計書に定義された安全弁の検証。"""

    def test_sv1_forced_regen_after_max_interval(self):
        """安全弁1: 再生成間隔の上限を超えるとスキップ解除。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        # 閾値分の空ティックを蓄積
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("テスト", empty_text, tick)
        # スキップ有効
        next_tick = SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        assert tracker.should_skip("テスト", next_tick) is True
        # last_regen_tick = SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1
        last_regen = SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1
        # max_interval経過後はスキップ解除
        forced_regen_tick = last_regen + SKIP_REGEN_MAX_INTERVAL
        assert tracker.should_skip("テスト", forced_regen_tick) is False

    def test_sv1_forced_regen_exactly_at_boundary(self):
        """安全弁1: ちょうどmax_intervalティック経過で再生成。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        # tick 0で再生成
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("テスト", empty_text, tick)
        # last_regen_tick = SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1
        last_regen = SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1
        # max_interval後のティック
        boundary_tick = last_regen + SKIP_REGEN_MAX_INTERVAL
        assert tracker.should_skip("テスト", boundary_tick) is False

    def test_sv2_always_full_labels_never_skipped(self):
        """安全弁2: ALWAYS_FULL_LABELS項目はスキップ対象外。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        for label in ALWAYS_FULL_LABELS:
            empty_text = f"{label}: {EMPTY_STATE_MARKER}"
            for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD + 5):
                tracker.update_after_generation(label, empty_text, tick)
            # カウンタは蓄積されるが、should_skipはFalse
            assert tracker.should_skip(label, 100) is False

    def test_sv3_first_tick_all_generated(self):
        """安全弁3: 初回ティックでは全項目がスキップされない。"""
        tracker = EmptySkipTracker()
        # mark_first_tick_done呼ばない → 全てFalse
        labels = ["感情", "ムード", "テスト1", "テスト2", "空ラベル"]
        for label in labels:
            assert tracker.should_skip(label, 0) is False

    def test_sv4_uses_empty_state_marker_only(self):
        """安全弁4: 判定基準はEMPTY_STATE_MARKERの含有のみ。"""
        tracker = EmptySkipTracker()
        # EMPTY_STATE_MARKERを含むテキスト → 空判定
        tracker.update_after_generation("A", f"A: {EMPTY_STATE_MARKER}", 0)
        assert tracker.get_consecutive_empty_count("A") == 1
        # EMPTY_STATE_MARKERを含まないテキスト → 非空判定
        tracker.update_after_generation("B", "B: some data", 0)
        assert tracker.get_consecutive_empty_count("B") == 0
        # 独自のパターンではない
        tracker.update_after_generation("C", "C: (空)", 0)
        assert tracker.get_consecutive_empty_count("C") == 0  # マーカーなし→非空

    def test_sv5_no_save_load_methods(self):
        """安全弁5: EmptySkipTrackerにsave/loadメソッドがない。"""
        tracker = EmptySkipTracker()
        assert not hasattr(tracker, "save")
        assert not hasattr(tracker, "load")
        assert not hasattr(tracker, "to_dict")
        assert not hasattr(tracker, "from_dict")

    def test_sv6_item_equivalence_non_empty_always_regenerated(self):
        """安全弁6: 空でない項目は全て等価に毎ティック再生成される。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        # 有データの項目はカウンタ0 → should_skipは常にFalse
        tracker.update_after_generation("A", "A: data1", 0)
        tracker.update_after_generation("B", "B: data2", 0)
        assert tracker.should_skip("A", 1) is False
        assert tracker.should_skip("B", 1) is False


# =============================================================================
# 即時復帰テスト
# =============================================================================

class TestImmediateRecovery:
    """空→有データ遷移時の即時復帰。"""

    def test_recovery_after_long_empty_streak(self):
        """長期間空だった項目が有データになると即座に復帰。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        # 長期間空
        for tick in range(20):
            tracker.update_after_generation("テスト", empty_text, tick)
        assert tracker.get_consecutive_empty_count("テスト") == 20
        # 有データに遷移
        tracker.update_after_generation("テスト", "テスト: value=0.5", 20)
        assert tracker.get_consecutive_empty_count("テスト") == 0
        # 即座にスキップ対象外
        assert tracker.should_skip("テスト", 21) is False

    def test_recovery_then_re_empty(self):
        """復帰後に再び空になった場合、カウンタが1から再開。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        # 空状態 → スキップ対象に
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("テスト", empty_text, tick)
        assert tracker.should_skip("テスト", SKIP_CONSECUTIVE_EMPTY_THRESHOLD) is True
        # 復帰
        tracker.update_after_generation(
            "テスト", "テスト: data", SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        )
        assert tracker.get_consecutive_empty_count("テスト") == 0
        # 再び空に
        tracker.update_after_generation(
            "テスト", empty_text, SKIP_CONSECUTIVE_EMPTY_THRESHOLD + 1
        )
        assert tracker.get_consecutive_empty_count("テスト") == 1
        # まだ閾値未満なのでスキップされない
        assert tracker.should_skip(
            "テスト", SKIP_CONSECUTIVE_EMPTY_THRESHOLD + 2
        ) is False


# =============================================================================
# 複数項目の独立性テスト
# =============================================================================

class TestMultipleItems:
    """複数項目間のカウンタ独立性。"""

    def test_counters_are_independent(self):
        """各項目のカウンタは独立。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text_a = f"A: {EMPTY_STATE_MARKER}"
        # Aのみ空
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("A", empty_text_a, tick)
            tracker.update_after_generation("B", "B: data", tick)
        # Aはスキップ対象、Bはスキップ対象外
        assert tracker.should_skip("A", SKIP_CONSECUTIVE_EMPTY_THRESHOLD) is True
        assert tracker.should_skip("B", SKIP_CONSECUTIVE_EMPTY_THRESHOLD) is False

    def test_skip_count_reflects_all_items(self):
        """get_skip_countは全項目のスキップ対象数を正しく返す。"""
        tracker = EmptySkipTracker()
        empty_a = f"A: {EMPTY_STATE_MARKER}"
        empty_b = f"B: {EMPTY_STATE_MARKER}"
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("A", empty_a, tick)
            tracker.update_after_generation("B", empty_b, tick)
            tracker.update_after_generation("C", "C: data", tick)
        assert tracker.get_skip_count() == 2  # A, B

    def test_mixed_empty_and_non_empty(self):
        """空と非空が混在する場合の正しい動作。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        labels = ["A", "B", "C", "D", "E"]
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            for i, label in enumerate(labels):
                if i % 2 == 0:  # A, C, E は空
                    tracker.update_after_generation(
                        label, f"{label}: {EMPTY_STATE_MARKER}", tick
                    )
                else:  # B, D は有データ
                    tracker.update_after_generation(
                        label, f"{label}: data", tick
                    )
        # A, C, E はスキップ対象
        for label in ["A", "C", "E"]:
            assert tracker.should_skip(
                label, SKIP_CONSECUTIVE_EMPTY_THRESHOLD
            ) is True
        # B, D はスキップ対象外
        for label in ["B", "D"]:
            assert tracker.should_skip(
                label, SKIP_CONSECUTIVE_EMPTY_THRESHOLD
            ) is False


# =============================================================================
# フィードバック経路遮断テスト
# =============================================================================

class TestFeedbackIsolation:
    """スキップ判定がpsyche内部状態に影響しないことの検証。"""

    def test_tracker_has_no_psyche_references(self):
        """EmptySkipTrackerはpsycheの内部状態への参照を持たない。"""
        tracker = EmptySkipTracker()
        # 内部属性にpsyche関連のものがないことを確認
        attrs = [a for a in dir(tracker) if not a.startswith("__")]
        for attr in attrs:
            assert "psyche" not in attr.lower()
            assert "emotion" not in attr.lower()
            assert "mood" not in attr.lower()
            assert "drive" not in attr.lower()

    def test_skip_result_not_in_enrichment_text(self):
        """スキップ判定の結果がenrichmentテキストに露出しないことを確認。"""
        # EmptySkipTracker自体はテキスト生成しない
        tracker = EmptySkipTracker()
        # should_skipはboolのみ返す
        result = tracker.should_skip("テスト", 0)
        assert isinstance(result, bool)

    def test_update_returns_none(self):
        """update_after_generationは副作用のみ（戻り値なし）。"""
        tracker = EmptySkipTracker()
        result = tracker.update_after_generation("テスト", "data", 0)
        assert result is None


# =============================================================================
# エッジケース
# =============================================================================

class TestEdgeCases:
    """境界条件のテスト。"""

    def test_tick_zero(self):
        """ティック0での動作。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        assert tracker.should_skip("テスト", 0) is False

    def test_very_large_tick(self):
        """非常に大きなティック番号での動作。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"
        tracker.update_after_generation("テスト", empty_text, 1000000)
        assert tracker.get_last_regen_tick("テスト") == 1000000

    def test_empty_label(self):
        """空文字列のラベル。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        tracker.update_after_generation("", f": {EMPTY_STATE_MARKER}", 0)
        assert tracker.get_consecutive_empty_count("") == 1

    def test_unicode_label(self):
        """Unicode文字を含むラベル。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        label = "感情基調"
        tracker.update_after_generation(label, f"{label}: {EMPTY_STATE_MARKER}", 0)
        assert tracker.get_consecutive_empty_count(label) == 1

    def test_marker_substring_in_text(self):
        """EMPTY_STATE_MARKERがテキスト中に部分文字列として含まれる場合。"""
        tracker = EmptySkipTracker()
        # マーカーを含むテキストは空判定
        text_with_marker = f"テスト: 前のデータ {EMPTY_STATE_MARKER} 後のデータ"
        tracker.update_after_generation("テスト", text_with_marker, 0)
        assert tracker.get_consecutive_empty_count("テスト") == 1

    def test_text_without_marker_is_non_empty(self):
        """EMPTY_STATE_MARKERを含まないテキストは非空。"""
        tracker = EmptySkipTracker()
        tracker.update_after_generation("テスト", "テスト: 何かのデータ", 0)
        assert tracker.get_consecutive_empty_count("テスト") == 0

    def test_multiple_mark_first_tick_done_calls(self):
        """mark_first_tick_doneの複数回呼び出しは安全。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        tracker.mark_first_tick_done()
        tracker.mark_first_tick_done()
        # 問題なく動作
        assert tracker.should_skip("テスト", 0) is False

    def test_should_skip_for_unknown_label(self):
        """未登録ラベルのスキップ判定。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        # カウンタ0 → スキップしない
        assert tracker.should_skip("未登録ラベル", 100) is False


# =============================================================================
# 統合シナリオテスト
# =============================================================================

class TestIntegrationScenarios:
    """実運用に近いシナリオの検証。"""

    def test_session_startup_scenario(self):
        """セッション起動時のシナリオ: 多くの項目が空→徐々に蓄積。"""
        tracker = EmptySkipTracker()
        labels = [
            "感情基調", "多経路想起", "内省横断", "矛盾並置",
            "感情共起", "忘却想起均衡", "注意配分",
        ]
        empty_labels_initially = labels[:5]
        data_labels_initially = labels[5:]

        # tick 0: 初回ティック（全項目生成）
        for label in empty_labels_initially:
            tracker.update_after_generation(
                label, f"{label}: {EMPTY_STATE_MARKER}", 0
            )
        for label in data_labels_initially:
            tracker.update_after_generation(
                label, f"{label}: data_value", 0
            )
        tracker.mark_first_tick_done()

        # tick 1~2: 引き続き空（閾値未満）
        for tick in range(1, SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            for label in empty_labels_initially:
                assert tracker.should_skip(label, tick) is False
                tracker.update_after_generation(
                    label, f"{label}: {EMPTY_STATE_MARKER}", tick
                )

        # tick 3+: 閾値到達、スキップ開始
        tick = SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        for label in empty_labels_initially:
            assert tracker.should_skip(label, tick) is True
        # 有データ項目はスキップされない
        for label in data_labels_initially:
            assert tracker.should_skip(label, tick) is False

    def test_gradual_activation_scenario(self):
        """段階的に有データになるシナリオ。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        labels = ["A", "B", "C"]

        # 全項目空を閾値分蓄積
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            for label in labels:
                tracker.update_after_generation(
                    label, f"{label}: {EMPTY_STATE_MARKER}", tick
                )

        # 全てスキップ対象
        tick = SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        for label in labels:
            assert tracker.should_skip(label, tick) is True

        # Aが有データになる（強制再生成時に検出）
        tracker.update_after_generation("A", "A: real_data", tick)
        assert tracker.should_skip("A", tick + 1) is False  # 即座に復帰
        # B, Cはまだスキップ中
        assert tracker.should_skip("B", tick + 1) is True
        assert tracker.should_skip("C", tick + 1) is True

    def test_forced_regen_detects_transition(self):
        """安全弁1: 強制再生成時に空→有データ遷移を検出。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        empty_text = f"テスト: {EMPTY_STATE_MARKER}"

        # 空を蓄積してスキップ対象に
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("テスト", empty_text, tick)

        # スキップ中
        current = SKIP_CONSECUTIVE_EMPTY_THRESHOLD
        assert tracker.should_skip("テスト", current) is True

        # max_interval経過 → 強制再生成
        last_regen = SKIP_CONSECUTIVE_EMPTY_THRESHOLD - 1
        forced_tick = last_regen + SKIP_REGEN_MAX_INTERVAL
        assert tracker.should_skip("テスト", forced_tick) is False
        # 強制再生成で有データを検出
        tracker.update_after_generation("テスト", "テスト: data", forced_tick)
        assert tracker.get_consecutive_empty_count("テスト") == 0
        # 以降は毎ティック再生成
        assert tracker.should_skip("テスト", forced_tick + 1) is False

    def test_alternating_empty_non_empty(self):
        """空と有データが交互の場合、スキップ対象にならない。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        for tick in range(20):
            if tick % 2 == 0:
                tracker.update_after_generation(
                    "テスト", f"テスト: {EMPTY_STATE_MARKER}", tick
                )
            else:
                tracker.update_after_generation(
                    "テスト", "テスト: data", tick
                )
        # 連続空は最大1回 → 閾値未満
        assert tracker.get_consecutive_empty_count("テスト") <= 1
        assert tracker.should_skip("テスト", 20) is False

    def test_always_full_items_with_empty_data(self):
        """ALWAYS_FULL_LABELS項目が空データでもスキップされない。"""
        tracker = EmptySkipTracker()
        tracker.mark_first_tick_done()
        for label in ALWAYS_FULL_LABELS:
            for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD + 5):
                tracker.update_after_generation(
                    label, f"{label}: {EMPTY_STATE_MARKER}", tick
                )
            # 空カウンタは蓄積されるが、should_skipは常にFalse
            assert tracker.get_consecutive_empty_count(label) > 0
            assert tracker.should_skip(label, 100) is False

    def test_skip_count_changes_with_recovery(self):
        """get_skip_countが復帰に応じて変化する。"""
        tracker = EmptySkipTracker()
        empty_a = f"A: {EMPTY_STATE_MARKER}"
        empty_b = f"B: {EMPTY_STATE_MARKER}"
        for tick in range(SKIP_CONSECUTIVE_EMPTY_THRESHOLD):
            tracker.update_after_generation("A", empty_a, tick)
            tracker.update_after_generation("B", empty_b, tick)
        assert tracker.get_skip_count() == 2
        # Aが復帰
        tracker.update_after_generation("A", "A: data", SKIP_CONSECUTIVE_EMPTY_THRESHOLD)
        assert tracker.get_skip_count() == 1
        # Bも復帰
        tracker.update_after_generation("B", "B: data", SKIP_CONSECUTIVE_EMPTY_THRESHOLD + 1)
        assert tracker.get_skip_count() == 0


# =============================================================================
# is_empty_state_textとの整合性テスト
# =============================================================================

class TestConsistencyWithEmptyStateDetection:
    """EmptySkipTrackerとis_empty_state_text/normalize_empty_stateの整合性。"""

    def test_normalized_empty_triggers_counter(self):
        """normalize_empty_state適用後のテキストがカウンタを増加させる。"""
        tracker = EmptySkipTracker()
        # 空テキストをnormalize_empty_stateで変換
        normalized = normalize_empty_state("テスト", "")
        assert EMPTY_STATE_MARKER in normalized
        # trackerに渡す
        tracker.update_after_generation("テスト", normalized, 0)
        assert tracker.get_consecutive_empty_count("テスト") == 1

    def test_normalized_non_empty_does_not_trigger(self):
        """normalize_empty_state適用後の有データテキストはカウンタ増加しない。"""
        tracker = EmptySkipTracker()
        normalized = normalize_empty_state("テスト", "テスト: actual data")
        assert EMPTY_STATE_MARKER not in normalized
        tracker.update_after_generation("テスト", normalized, 0)
        assert tracker.get_consecutive_empty_count("テスト") == 0

    def test_all_known_empty_patterns_result_in_marker(self):
        """既知の空パターンはnormalize後にマーカーを含む。"""
        tracker = EmptySkipTracker()
        empty_patterns = ["", "  ", "(なし)", "(空)", "(蓄積前)"]
        for i, pattern in enumerate(empty_patterns):
            assert is_empty_state_text(pattern) is True
            normalized = normalize_empty_state(f"test{i}", pattern)
            tracker.update_after_generation(f"test{i}", normalized, 0)
            assert tracker.get_consecutive_empty_count(f"test{i}") == 1
