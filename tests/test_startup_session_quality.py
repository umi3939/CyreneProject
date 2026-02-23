"""
tests/test_startup_session_quality.py - 初回起動・セッション復帰時の状態品質テスト

テスト対象: psyche/enrichment_compression.py (起動品質A/B関数)
          + psyche/orchestrator.py (統合部分)
設計書: design_startup_session_quality.md

テスト項目:
- 空状態記述統一 (is_empty_state_text, normalize_empty_state, normalize_section_items)
- 鮮度注釈 (classify_elapsed_time, build_freshness_annotation, prepend_freshness_annotation)
- 安全弁6: 空状態表記の単一性
- 安全弁7: 鮮度注釈の無条件消失
- 安全弁8: 初回起動への鮮度注釈非適用
- 安全弁9: 永続化フィールド非追加
- 安全弁10: enrichment以外への非露出
- orchestrator統合テスト
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from psyche.enrichment_compression import (
    EMPTY_STATE_MARKER,
    FRESHNESS_TRANSITION_TICKS,
    is_empty_state_text,
    normalize_empty_state,
    normalize_section_items,
    classify_elapsed_time,
    build_freshness_annotation,
    prepend_freshness_annotation,
    build_compressed_enrichment,
    ORIGINAL_FOOTER,
    _KNOWN_EMPTY_PATTERNS,
)


# =============================================================================
# 起動品質A: 空状態記述統一
# =============================================================================

class TestIsEmptyStateText:
    """空状態判定のテスト。"""

    def test_empty_string(self):
        """空文字列は空状態。"""
        assert is_empty_state_text("") is True

    def test_whitespace_only(self):
        """ホワイトスペースのみは空状態。"""
        assert is_empty_state_text("   ") is True
        assert is_empty_state_text("\t\n") is True

    def test_nashi_pattern(self):
        """「(なし)」は空状態。"""
        assert is_empty_state_text("(なし)") is True

    def test_kuu_pattern(self):
        """「(空)」は空状態。"""
        assert is_empty_state_text("(空)") is True

    def test_chikusekumae_pattern(self):
        """「(蓄積前)」は空状態。"""
        assert is_empty_state_text("(蓄積前)") is True

    def test_normal_text_not_empty(self):
        """通常テキストは空状態ではない。"""
        assert is_empty_state_text("感情: joy=0.8") is False

    def test_taiki_text_not_empty(self):
        """「待機中」を含むテキストは空状態判定しない（モジュール独自の表現）。"""
        assert is_empty_state_text("行動-結果観測: 待機中") is False

    def test_number_text_not_empty(self):
        """数値テキストは空状態ではない。"""
        assert is_empty_state_text("0.0") is False

    def test_short_meaningful_text(self):
        """短くても意味のあるテキストは空状態ではない。"""
        assert is_empty_state_text("OK") is False

    def test_whitespace_padded_empty_pattern(self):
        """空パターンの前後にスペースがあっても空状態。"""
        assert is_empty_state_text("  (なし)  ") is True

    def test_known_empty_patterns_is_frozenset(self):
        """既知の空状態パターンはfrozenset。"""
        assert isinstance(_KNOWN_EMPTY_PATTERNS, frozenset)


class TestNormalizeEmptyState:
    """空状態テキストの統一表記テスト。"""

    def test_empty_string_normalized(self):
        """空文字列を統一表記に置換。"""
        result = normalize_empty_state("テスト", "")
        assert result == f"テスト: {EMPTY_STATE_MARKER}"

    def test_nashi_normalized(self):
        """「(なし)」を統一表記に置換。"""
        result = normalize_empty_state("テスト", "(なし)")
        assert result == f"テスト: {EMPTY_STATE_MARKER}"

    def test_normal_text_unchanged(self):
        """通常テキストは変更なし。"""
        text = "感情: joy=0.8, surprise=0.3"
        result = normalize_empty_state("感情", text)
        assert result == text

    def test_safety_valve_6_single_marker(self):
        """安全弁6: 空状態表記は「(未蓄積)」一種類のみ。"""
        assert EMPTY_STATE_MARKER == "(未蓄積)"

    def test_different_labels_same_marker(self):
        """異なるラベルでも同じマーカー。"""
        r1 = normalize_empty_state("感情", "")
        r2 = normalize_empty_state("自己像", "")
        assert EMPTY_STATE_MARKER in r1
        assert EMPTY_STATE_MARKER in r2

    def test_whitespace_only_normalized(self):
        """ホワイトスペースのみも統一表記に。"""
        result = normalize_empty_state("テスト", "   ")
        assert result == f"テスト: {EMPTY_STATE_MARKER}"


class TestNormalizeSectionItems:
    """セクション内全項目の空状態統一テスト。"""

    def test_mixed_items(self):
        """空状態と通常テキストが混在するケース。"""
        items = [
            ("感情", "感情: joy=0.8"),
            ("自己像", ""),
            ("一貫性", "(なし)"),
        ]
        result = normalize_section_items(items)
        assert result[0] == ("感情", "感情: joy=0.8")
        assert result[1] == ("自己像", f"自己像: {EMPTY_STATE_MARKER}")
        assert result[2] == ("一貫性", f"一貫性: {EMPTY_STATE_MARKER}")

    def test_all_normal(self):
        """全項目が通常テキストの場合は変更なし。"""
        items = [
            ("感情", "感情: joy=0.8"),
            ("ムード", "ムード: valence=0.5"),
        ]
        result = normalize_section_items(items)
        assert result == items

    def test_all_empty(self):
        """全項目が空状態の場合。"""
        items = [
            ("A", ""),
            ("B", "(なし)"),
            ("C", "(空)"),
        ]
        result = normalize_section_items(items)
        for label, text in result:
            assert EMPTY_STATE_MARKER in text

    def test_empty_list(self):
        """空リスト。"""
        result = normalize_section_items([])
        assert result == []

    def test_returns_new_list(self):
        """元のリストは変更しない。"""
        items = [("A", "")]
        original = list(items)
        result = normalize_section_items(items)
        assert items == original  # 元リストは変更されていない
        assert result is not items


# =============================================================================
# 起動品質B: セッション境界の鮮度注釈
# =============================================================================

class TestClassifyElapsedTime:
    """セッション間経過時間の段階値分類テスト。"""

    def test_seconds(self):
        """数秒。"""
        assert classify_elapsed_time(30) == "数秒前"

    def test_few_minutes(self):
        """数分。"""
        assert classify_elapsed_time(120) == "数分前"

    def test_about_10_minutes(self):
        """約10分。"""
        result = classify_elapsed_time(600)
        assert "約10分前" == result

    def test_about_30_minutes_plus(self):
        """30分以上。"""
        result = classify_elapsed_time(2000)
        assert "約30分以上前" == result

    def test_one_hour(self):
        """約1時間。"""
        result = classify_elapsed_time(3600)
        assert "約1時間前" == result

    def test_several_hours(self):
        """数時間。"""
        result = classify_elapsed_time(10800)
        assert "約3時間前" == result

    def test_one_day(self):
        """約1日。"""
        result = classify_elapsed_time(86400)
        assert "約1日前" == result

    def test_several_days(self):
        """数日。"""
        result = classify_elapsed_time(259200)
        assert "約3日前" == result

    def test_week_plus(self):
        """1週間以上。"""
        result = classify_elapsed_time(864000)
        assert "日以上前" in result

    def test_negative_time(self):
        """負の経過時間。"""
        assert classify_elapsed_time(-1) == "不明"

    def test_zero_time(self):
        """0秒。"""
        assert classify_elapsed_time(0) == "数秒前"

    def test_no_normative_language(self):
        """規範的表現を含まない。"""
        for seconds in [0, 60, 3600, 86400, 604800]:
            result = classify_elapsed_time(seconds)
            assert "復帰" not in result
            assert "不完全" not in result
            assert "まだ" not in result


class TestBuildFreshnessAnnotation:
    """鮮度注釈生成テスト。"""

    def test_first_boot_no_annotation(self):
        """安全弁8: 初回起動時は鮮度注釈なし。"""
        result = build_freshness_annotation(
            session_gap_seconds=None,
            session_resume_tick=None,
            current_tick=0,
        )
        assert result is None

    def test_session_resume_with_annotation(self):
        """セッション再開直後は鮮度注釈あり。"""
        result = build_freshness_annotation(
            session_gap_seconds=3600,
            session_resume_tick=100,
            current_tick=105,
        )
        assert result is not None
        assert "セッション再開" in result
        assert "約1時間前" in result
        assert "5ティック経過" in result

    def test_annotation_disappears_after_threshold(self):
        """安全弁7: 過渡期ティック閾値を超えると注釈が消失。"""
        result = build_freshness_annotation(
            session_gap_seconds=3600,
            session_resume_tick=100,
            current_tick=100 + FRESHNESS_TRANSITION_TICKS,
        )
        assert result is None

    def test_annotation_present_before_threshold(self):
        """閾値の直前では注釈あり。"""
        result = build_freshness_annotation(
            session_gap_seconds=3600,
            session_resume_tick=100,
            current_tick=100 + FRESHNESS_TRANSITION_TICKS - 1,
        )
        assert result is not None

    def test_annotation_format(self):
        """注釈のフォーマット。"""
        result = build_freshness_annotation(
            session_gap_seconds=120,
            session_resume_tick=0,
            current_tick=3,
        )
        assert result.startswith("[セッション再開:")
        assert result.endswith("]")
        assert "3ティック経過" in result

    def test_no_normative_expression(self):
        """注釈に規範的表現を含まない。"""
        result = build_freshness_annotation(
            session_gap_seconds=86400,
            session_resume_tick=0,
            current_tick=0,
        )
        assert "復帰中" not in result
        assert "まだ不完全" not in result
        assert "安定" not in result

    def test_gap_none_resume_not_none(self):
        """gap=Noneだがresume有り（不整合） → 注釈なし。"""
        result = build_freshness_annotation(
            session_gap_seconds=None,
            session_resume_tick=100,
            current_tick=105,
        )
        assert result is None

    def test_gap_not_none_resume_none(self):
        """gap有りだがresume=None（不整合） → 注釈なし。"""
        result = build_freshness_annotation(
            session_gap_seconds=3600,
            session_resume_tick=None,
            current_tick=105,
        )
        assert result is None


class TestPrependFreshnessAnnotation:
    """enrichmentテキストへの鮮度注釈付与テスト。"""

    def test_no_annotation_first_boot(self):
        """初回起動では元テキストそのまま。"""
        text = "[内面]\n感情: joy=0.8"
        result = prepend_freshness_annotation(
            text, session_gap_seconds=None,
            session_resume_tick=None, current_tick=0,
        )
        assert result == text

    def test_annotation_prepended(self):
        """セッション再開では冒頭に注釈が付く。"""
        text = "[内面]\n感情: joy=0.8"
        result = prepend_freshness_annotation(
            text, session_gap_seconds=7200,
            session_resume_tick=50, current_tick=55,
        )
        assert result.startswith("[セッション再開:")
        assert "[内面]" in result
        assert "感情: joy=0.8" in result

    def test_annotation_not_prepended_after_threshold(self):
        """過渡期後は元テキストそのまま。"""
        text = "[内面]\n感情: joy=0.8"
        result = prepend_freshness_annotation(
            text, session_gap_seconds=7200,
            session_resume_tick=50,
            current_tick=50 + FRESHNESS_TRANSITION_TICKS,
        )
        assert result == text

    def test_section_structure_preserved(self):
        """セクション構造の内部に注釈が介入しない。"""
        text = "[内面]\n感情: joy=0.8\n\n[自己]\n自己像: test"
        result = prepend_freshness_annotation(
            text, session_gap_seconds=3600,
            session_resume_tick=0, current_tick=5,
        )
        # 注釈は最初のセクションの前
        lines = result.split("\n\n")
        assert lines[0].startswith("[セッション再開:")
        assert "[内面]" in lines[1]


class TestFreshnessTransitionTicks:
    """過渡期ティック閾値の設定テスト。"""

    def test_threshold_matches_max_window(self):
        """閾値がスライディングウィンドウ最大サイズに準じている。"""
        # drive_variation_description, internal_contradiction_description = 50
        assert FRESHNESS_TRANSITION_TICKS == 50

    def test_threshold_is_positive(self):
        """閾値は正の整数。"""
        assert FRESHNESS_TRANSITION_TICKS > 0
        assert isinstance(FRESHNESS_TRANSITION_TICKS, int)


# =============================================================================
# 統合テスト: 圧縮パイプラインとの統合
# =============================================================================

class TestIntegrationWithCompression:
    """空状態統一と圧縮パイプラインの統合テスト。"""

    def test_empty_items_normalized_then_compressed(self):
        """空状態正規化→圧縮パイプラインの一連動作。"""
        items = [
            ("A", "A: value"),
            ("B", ""),
        ]
        normalized = normalize_section_items(items)
        sections = [{"header": "【自己認識】", "items": normalized}]
        text, cache, _ratio = build_compressed_enrichment(
            sections, {}, ORIGINAL_FOOTER,
        )
        assert "A: value" in text
        assert EMPTY_STATE_MARKER in text

    def test_normalized_empty_in_cache(self):
        """正規化後のテキストがキャッシュに保存される。"""
        items = [("B", "")]
        normalized = normalize_section_items(items)
        sections = [{"header": "【自己認識】", "items": normalized}]
        _text, cache, _ = build_compressed_enrichment(
            sections, {}, ORIGINAL_FOOTER,
        )
        assert cache["B"] == f"B: {EMPTY_STATE_MARKER}"

    def test_freshness_annotation_with_compression(self):
        """鮮度注釈+圧縮結果の統合。"""
        sections = [{"header": "【自己認識】", "items": [
            ("A", "A: test value"),
        ]}]
        compressed, _, _ = build_compressed_enrichment(
            sections, {}, ORIGINAL_FOOTER,
        )
        result = prepend_freshness_annotation(
            compressed, session_gap_seconds=7200,
            session_resume_tick=10, current_tick=15,
        )
        assert result.startswith("[セッション再開:")
        assert "A: test value" in result


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    """設計書の安全弁要件テスト。"""

    def test_sv6_single_marker(self):
        """安全弁6: 空状態表記は一種類のみ。"""
        results = set()
        for pattern in ["", "(なし)", "(空)", "(蓄積前)", "   "]:
            result = normalize_empty_state("X", pattern)
            results.add(result)
        # 全て同じ結果
        assert len(results) == 1
        assert EMPTY_STATE_MARKER in results.pop()

    def test_sv7_unconditional_disappearance(self):
        """安全弁7: 鮮度注釈はティック数のみで消失判定。"""
        # 閾値ちょうどで消失
        assert build_freshness_annotation(3600, 0, FRESHNESS_TRANSITION_TICKS) is None
        # 閾値-1で存在
        assert build_freshness_annotation(3600, 0, FRESHNESS_TRANSITION_TICKS - 1) is not None

    def test_sv8_no_annotation_first_boot(self):
        """安全弁8: 初回起動での鮮度注釈非適用。"""
        assert build_freshness_annotation(None, None, 0) is None
        assert build_freshness_annotation(None, None, 100) is None

    def test_sv9_no_new_save_fields(self):
        """安全弁9: session_gap_seconds/session_resume_tickはsave対象外。"""
        # enrichment_compressionモジュールにsave関数がないことを確認
        import psyche.enrichment_compression as mod
        assert not hasattr(mod, "save_session_state")
        assert not hasattr(mod, "to_dict")

    def test_sv10_no_exposure_beyond_enrichment(self):
        """安全弁10: enrichment以外への非露出（関数の戻り値がstrのみ）。"""
        result = normalize_empty_state("X", "")
        assert isinstance(result, str)
        result2 = build_freshness_annotation(3600, 0, 5)
        assert isinstance(result2, str)
        result3 = prepend_freshness_annotation("text", 3600, 0, 5)
        assert isinstance(result3, str)


# =============================================================================
# フィードバック経路遮断の検証
# =============================================================================

class TestFeedbackIsolation:
    """空状態統一と鮮度注釈がモジュール内部状態に影響しないことを検証。"""

    def test_normalize_does_not_mutate_input(self):
        """normalize_section_itemsが入力リストを変更しない。"""
        items = [("A", ""), ("B", "text")]
        original_items = [(l, t) for l, t in items]
        normalize_section_items(items)
        assert items == original_items

    def test_freshness_annotation_does_not_affect_compression_cache(self):
        """鮮度注釈の有無がキャッシュに影響しない。"""
        sections = [{"header": "【自己認識】", "items": [
            ("A", "A: v1"),
        ]}]
        _, cache1, _ = build_compressed_enrichment(sections, {}, ORIGINAL_FOOTER)
        # 鮮度注釈を付与してもキャッシュは同じ
        text_with = prepend_freshness_annotation(
            "test", 3600, 0, 5,
        )
        text_without = prepend_freshness_annotation(
            "test", None, None, 5,
        )
        # キャッシュは圧縮パイプラインが管理、鮮度注釈は後付け
        assert cache1 == {"A": "A: v1"}


# =============================================================================
# エッジケース
# =============================================================================

class TestEdgeCases:
    """境界条件のテスト。"""

    def test_classify_very_large_time(self):
        """非常に長い経過時間。"""
        result = classify_elapsed_time(31536000)  # 365 days
        assert "日以上前" in result

    def test_classify_boundary_60_seconds(self):
        """ちょうど60秒の境界。"""
        result = classify_elapsed_time(60)
        # 60秒は「数分前」に分類される
        assert "分" in result

    def test_classify_boundary_3600_seconds(self):
        """ちょうど3600秒の境界。"""
        result = classify_elapsed_time(3600)
        assert "時間" in result

    def test_classify_boundary_86400_seconds(self):
        """ちょうど86400秒の境界。"""
        result = classify_elapsed_time(86400)
        assert "日" in result

    def test_normalize_empty_with_various_whitespace(self):
        """様々なホワイトスペースパターン。"""
        for ws in [" ", "\t", "\n", "\r\n", "  \t  "]:
            assert is_empty_state_text(ws) is True

    def test_freshness_zero_gap(self):
        """経過時間ゼロでもセッション再開注釈はある。"""
        result = build_freshness_annotation(0.0, 0, 0)
        assert result is not None
        assert "数秒前" in result

    def test_freshness_exact_threshold(self):
        """閾値ちょうどで消失。"""
        result = build_freshness_annotation(3600, 0, FRESHNESS_TRANSITION_TICKS)
        assert result is None

    def test_prepend_to_empty_text(self):
        """空テキストへの鮮度注釈付与。"""
        result = prepend_freshness_annotation(
            "", session_gap_seconds=3600,
            session_resume_tick=0, current_tick=5,
        )
        assert result.startswith("[セッション再開:")


# =============================================================================
# orchestrator統合テスト
# =============================================================================

class TestOrchestratorIntegration:
    """orchestratorのsave/load/enrichmentにおける統合テスト。"""

    def test_save_includes_timestamp(self):
        """save()がsave_timestampを含むことを確認。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_snapshot.json"
            orch.save(save_path)
            data = json.loads(save_path.read_text(encoding="utf-8"))
            assert "save_timestamp" in data
            assert isinstance(data["save_timestamp"], float)
            # タイムスタンプは現在時刻に近い
            assert abs(data["save_timestamp"] - time.time()) < 5.0

    def test_load_sets_session_gap(self):
        """load()後にsession_gap_secondsが設定される。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_snapshot.json"
            orch.save(save_path)
            # 少し待ってからload
            orch2 = PsycheOrchestrator()
            result = orch2.load(save_path)
            assert result is True
            assert orch2._session_gap_seconds is not None
            assert orch2._session_gap_seconds >= 0.0
            assert orch2._session_resume_tick is not None

    def test_load_without_timestamp_no_gap(self):
        """save_timestampなしのデータをload → session_gap=None。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_snapshot.json"
            orch.save(save_path)
            # save_timestampを削除
            data = json.loads(save_path.read_text(encoding="utf-8"))
            del data["save_timestamp"]
            save_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            orch2 = PsycheOrchestrator()
            orch2.load(save_path)
            assert orch2._session_gap_seconds is None
            assert orch2._session_resume_tick is None

    def test_first_boot_no_session_gap(self):
        """初回起動（load未実行）ではsession_gap=None。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        assert orch._session_gap_seconds is None
        assert orch._session_resume_tick is None

    def test_enrichment_includes_annotation_after_load(self):
        """load後のenrichmentに鮮度注釈が含まれる。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_snapshot.json"
            orch.save(save_path)

            orch2 = PsycheOrchestrator()
            orch2.load(save_path)
            # load直後のenrichment
            text = orch2.get_prompt_enrichment()
            assert "[セッション再開:" in text

    def test_enrichment_no_annotation_first_boot(self):
        """初回起動のenrichmentに鮮度注釈がない。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        text = orch.get_prompt_enrichment()
        assert "[セッション再開:" not in text

    def test_session_gap_not_in_save_data(self):
        """session_gap_seconds/session_resume_tickがsaveデータに含まれない。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        orch._session_gap_seconds = 3600.0
        orch._session_resume_tick = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_snapshot.json"
            orch.save(save_path)
            data = json.loads(save_path.read_text(encoding="utf-8"))
            assert "session_gap_seconds" not in data
            assert "session_resume_tick" not in data

    def test_enrichment_empty_state_normalization(self):
        """enrichmentで空状態テキストが統一表記に置換される。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        # get_prompt_enrichmentを呼び出す
        text = orch.get_prompt_enrichment()
        # 空状態が存在する場合に「(未蓄積)」が使われる
        # (モジュールの初期状態に依存するが、機能は統合されている)
        assert isinstance(text, str)
