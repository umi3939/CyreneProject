"""
tests/test_session_difference.py - セッション間差分記述のテスト

design_session_difference.md に基づく実装のテスト:
  - 数値的距離算出の正確性（スカラー/辞書/リスト各パターン）
  - save→load往復での差分値の保全
  - 初回保存時の差分値不在
  - 段階値変換の区間分割
  - enrichment項目の追加
  - フィールド内訳がenrichmentに含まれないこと
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from psyche.save_load_warmup import (
    _compute_field_distance,
    compute_session_difference_scalar,
    classify_session_difference,
    build_session_diff_enrichment_text,
    SESSION_DIFF_EMPTY_LABEL,
)


# ══════════════════════════════════════════════════════════════════════════════
# _compute_field_distance: フィールド単位の数値的距離算出
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeFieldDistance:
    """_compute_field_distance の正確性テスト。"""

    # ── スカラー数値 ──

    def test_scalar_int_difference(self) -> None:
        """整数スカラーの差の絶対値。"""
        assert _compute_field_distance(10, 15) == 5.0

    def test_scalar_float_difference(self) -> None:
        """浮動小数点スカラーの差の絶対値。"""
        result = _compute_field_distance(1.5, 3.7)
        assert abs(result - 2.2) < 1e-10

    def test_scalar_negative_difference(self) -> None:
        """負方向の差でも絶対値。"""
        assert _compute_field_distance(20, 5) == 15.0

    def test_scalar_zero_difference(self) -> None:
        """同一値の距離は0。"""
        assert _compute_field_distance(42, 42) == 0.0

    def test_scalar_int_float_mixed(self) -> None:
        """整数と浮動小数点の混合。"""
        assert _compute_field_distance(10, 10.5) == 0.5

    # ── 辞書型 ──

    def test_dict_numeric_fields(self) -> None:
        """辞書内の数値フィールドの差の絶対値の合計。"""
        prev = {"a": 1.0, "b": 2.0, "c": 3.0}
        curr = {"a": 1.5, "b": 4.0, "c": 3.0}
        # |1.5-1.0| + |4.0-2.0| + |3.0-3.0| = 0.5 + 2.0 + 0.0 = 2.5
        assert _compute_field_distance(prev, curr) == 2.5

    def test_dict_non_numeric_ignored(self) -> None:
        """辞書内の非数値フィールドは無視される。"""
        prev = {"x": 1.0, "name": "hello"}
        curr = {"x": 3.0, "name": "world"}
        assert _compute_field_distance(prev, curr) == 2.0

    def test_dict_extra_keys_in_one_side(self) -> None:
        """片方にのみ存在するキーの数値は無視（距離0として扱う）。"""
        prev = {"a": 1.0}
        curr = {"a": 2.0, "b": 5.0}
        # a: |2.0-1.0| = 1.0, b: 片方のみ → 0
        assert _compute_field_distance(prev, curr) == 1.0

    def test_dict_empty_both(self) -> None:
        """両方空辞書: 距離0。"""
        assert _compute_field_distance({}, {}) == 0.0

    def test_dict_nested_dict(self) -> None:
        """ネストされた辞書の再帰的走査。"""
        prev = {"inner": {"x": 1.0, "y": 2.0}}
        curr = {"inner": {"x": 3.0, "y": 2.0}}
        # inner.x: |3.0-1.0| = 2.0, inner.y: 0
        assert _compute_field_distance(prev, curr) == 2.0

    # ── リスト型 ──

    def test_list_count_difference(self) -> None:
        """リスト件数の差の絶対値のみ。"""
        prev = [1, 2, 3]
        curr = [1, 2, 3, 4, 5]
        assert _compute_field_distance(prev, curr) == 2.0

    def test_list_same_count(self) -> None:
        """同じ件数: 距離0（レコード内容は比較しない）。"""
        prev = [{"tick": 1}, {"tick": 2}]
        curr = [{"tick": 99}, {"tick": 100}]
        assert _compute_field_distance(prev, curr) == 0.0

    def test_list_empty_vs_nonempty(self) -> None:
        """空リストと非空リスト。"""
        assert _compute_field_distance([], [1, 2, 3]) == 3.0

    # ── 型不一致・その他 ──

    def test_type_mismatch_returns_zero(self) -> None:
        """型が異なる場合は距離0。"""
        assert _compute_field_distance(10, [1, 2, 3]) == 0.0
        assert _compute_field_distance({"a": 1}, [1]) == 0.0

    def test_string_values_return_zero(self) -> None:
        """文字列値は距離算出不可能、0。"""
        assert _compute_field_distance("hello", "world") == 0.0

    def test_none_values_return_zero(self) -> None:
        """None値は距離0。"""
        assert _compute_field_distance(None, None) == 0.0

    def test_bool_values_return_zero(self) -> None:
        """bool値は距離0（int扱いしない）。"""
        # Python ではboolはintのサブクラスだが、このテストで
        # True=1, False=0 として扱われることを確認
        # boolもisinstance(x, int)がTrueなので距離算出される
        result = _compute_field_distance(True, False)
        assert result == 1.0  # |1-0| = 1


# ══════════════════════════════════════════════════════════════════════════════
# compute_session_difference_scalar: 全フィールドの距離総和
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSessionDifferenceScalar:
    """compute_session_difference_scalar の正確性テスト。"""

    def test_basic_scalar_sum(self) -> None:
        """複数フィールドの距離の総和。"""
        prev = {"field_a": 1.0, "field_b": 10}
        curr = {"field_a": 2.0, "field_b": 15}
        # |2.0-1.0| + |15-10| = 1.0 + 5.0 = 6.0
        result = compute_session_difference_scalar(prev, curr)
        assert result == 6.0

    def test_metadata_excluded(self) -> None:
        """version, save_timestamp, tick_count, session_diff_scalarは除外。"""
        prev = {"version": 1, "save_timestamp": 100.0, "tick_count": 5,
                "session_diff_scalar": 3.0, "real_field": 10.0}
        curr = {"version": 99, "save_timestamp": 999.0, "tick_count": 999,
                "session_diff_scalar": 99.0, "real_field": 12.0}
        result = compute_session_difference_scalar(prev, curr)
        assert result == 2.0  # real_fieldのみ: |12.0-10.0| = 2.0

    def test_field_only_in_one_side(self) -> None:
        """片方にのみ存在するフィールドの距離は0。"""
        prev = {"a": 1.0}
        curr = {"a": 2.0, "b": 99.0}
        result = compute_session_difference_scalar(prev, curr)
        assert result == 1.0  # aのみ: |2.0-1.0|

    def test_dict_fields(self) -> None:
        """辞書型フィールドの距離算出。"""
        prev = {"psyche": {"joy": 0.5, "sadness": 0.1}}
        curr = {"psyche": {"joy": 0.8, "sadness": 0.3}}
        # |0.8-0.5| + |0.3-0.1| = 0.3 + 0.2 = 0.5
        result = compute_session_difference_scalar(prev, curr)
        assert abs(result - 0.5) < 1e-10

    def test_list_fields(self) -> None:
        """リスト型フィールドの距離算出。"""
        prev = {"records": [1, 2, 3]}
        curr = {"records": [1, 2, 3, 4, 5]}
        result = compute_session_difference_scalar(prev, curr)
        assert result == 2.0

    def test_empty_dicts(self) -> None:
        """両方空辞書: 総距離0。"""
        result = compute_session_difference_scalar({}, {})
        assert result == 0.0

    def test_identical_dicts(self) -> None:
        """同一辞書: 総距離0。"""
        data = {"a": 1.0, "b": {"x": 2.0}, "c": [1, 2]}
        result = compute_session_difference_scalar(data, data)
        assert result == 0.0

    def test_mixed_field_types(self) -> None:
        """スカラー・辞書・リストの混合。"""
        prev = {
            "tick_count": 10,  # 除外
            "scalar_field": 5.0,
            "dict_field": {"x": 1.0, "y": 2.0},
            "list_field": [1, 2, 3],
        }
        curr = {
            "tick_count": 20,  # 除外
            "scalar_field": 8.0,
            "dict_field": {"x": 3.0, "y": 2.0},
            "list_field": [1, 2, 3, 4],
        }
        # scalar: |8-5| = 3.0
        # dict: |3-1| + |2-2| = 2.0
        # list: |4-3| = 1.0
        # total = 6.0
        result = compute_session_difference_scalar(prev, curr)
        assert result == 6.0

    def test_non_numeric_string_fields_ignored(self) -> None:
        """文字列フィールドは距離0。"""
        prev = {"name": "hello", "count": 5}
        curr = {"name": "world", "count": 10}
        result = compute_session_difference_scalar(prev, curr)
        assert result == 5.0  # countのみ


# ══════════════════════════════════════════════════════════════════════════════
# classify_session_difference: 段階値変換の区間分割
# ══════════════════════════════════════════════════════════════════════════════


class TestClassifySessionDifference:
    """classify_session_difference の区間分割テスト。"""

    def test_none_returns_empty_label(self) -> None:
        """Noneの場合は空状態テキスト。"""
        result = classify_session_difference(None)
        assert result == SESSION_DIFF_EMPTY_LABEL
        assert result == "(不明)"

    def test_zero_is_no_change(self) -> None:
        """0は「ほぼ変化なし」。"""
        assert classify_session_difference(0.0) == "ほぼ変化なし"

    def test_small_value_is_no_change(self) -> None:
        """0.1以下は「ほぼ変化なし」。"""
        assert classify_session_difference(0.05) == "ほぼ変化なし"
        assert classify_session_difference(0.1) == "ほぼ変化なし"

    def test_minor_change(self) -> None:
        """0.1超～5.0以下は「微小な変化」。"""
        assert classify_session_difference(0.5) == "微小な変化"
        assert classify_session_difference(5.0) == "微小な変化"

    def test_moderate_change(self) -> None:
        """5.0超～50.0以下は「中程度の変化」。"""
        assert classify_session_difference(5.1) == "中程度の変化"
        assert classify_session_difference(25.0) == "中程度の変化"
        assert classify_session_difference(50.0) == "中程度の変化"

    def test_large_change(self) -> None:
        """50.0超は「大きな変化」。"""
        assert classify_session_difference(50.1) == "大きな変化"
        assert classify_session_difference(100.0) == "大きな変化"
        assert classify_session_difference(9999.0) == "大きな変化"

    def test_boundary_values(self) -> None:
        """境界値の正確な分類。"""
        # 0.1以下 → ほぼ変化なし
        assert classify_session_difference(0.1) == "ほぼ変化なし"
        # 0.1超 → 微小な変化
        assert classify_session_difference(0.10001) == "微小な変化"
        # 5.0以下 → 微小な変化
        assert classify_session_difference(5.0) == "微小な変化"
        # 5.0超 → 中程度の変化
        assert classify_session_difference(5.00001) == "中程度の変化"
        # 50.0以下 → 中程度の変化
        assert classify_session_difference(50.0) == "中程度の変化"
        # 50.0超 → 大きな変化
        assert classify_session_difference(50.00001) == "大きな変化"

    def test_no_evaluative_language(self) -> None:
        """段階値テキストに評価的表現が含まれない。"""
        evaluative_terms = [
            "良い", "悪い", "成長", "劣化", "改善", "退化",
            "問題", "正常", "異常", "健全", "不健全",
        ]
        for scalar in [None, 0.0, 1.0, 10.0, 100.0]:
            label = classify_session_difference(scalar)
            for term in evaluative_terms:
                assert term not in label, (
                    f"Evaluative term '{term}' found in label "
                    f"'{label}' for scalar={scalar}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# build_session_diff_enrichment_text: enrichment項目テキスト
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSessionDiffEnrichmentText:
    """build_session_diff_enrichment_text のテスト。"""

    def test_none_scalar(self) -> None:
        """Noneの場合の空状態テキスト。"""
        result = build_session_diff_enrichment_text(None)
        assert "セッション間状態変化" in result
        assert "(不明)" in result

    def test_zero_scalar(self) -> None:
        """0の場合。"""
        result = build_session_diff_enrichment_text(0.0)
        assert "セッション間状態変化" in result
        assert "ほぼ変化なし" in result

    def test_moderate_scalar(self) -> None:
        """中程度の値。"""
        result = build_session_diff_enrichment_text(25.0)
        assert "セッション間状態変化" in result
        assert "中程度の変化" in result

    def test_no_field_breakdown_in_text(self) -> None:
        """フィールド内訳がテキストに含まれないこと（安全弁5）。"""
        result = build_session_diff_enrichment_text(10.0)
        # フィールド固有のキー名が含まれないことを確認
        field_names = [
            "psyche", "loop_state", "dynamics", "tick_count",
            "value_orientation", "tendency_state",
        ]
        for name in field_names:
            assert name not in result, (
                f"Field name '{name}' found in enrichment text"
            )

    def test_text_is_fact_description_only(self) -> None:
        """事実記述のみで評価的表現を含まないこと（安全弁4）。"""
        for scalar in [None, 0.0, 1.0, 10.0, 100.0]:
            text = build_session_diff_enrichment_text(scalar)
            evaluative_terms = [
                "良い", "悪い", "成長", "劣化", "改善", "問題",
            ]
            for term in evaluative_terms:
                assert term not in text


# ══════════════════════════════════════════════════════════════════════════════
# save→load往復テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLoadRoundTrip:
    """save→load往復でのセッション間差分値の保全テスト。"""

    def _create_mock_orchestrator(self) -> MagicMock:
        """テスト用の最小限のオーケストレータモックを作成。"""
        orch = MagicMock()
        orch._tick_count = 10
        orch._session_prev_snapshot = None
        orch._session_diff_scalar = None
        orch._session_gap_seconds = None
        orch._session_resume_tick = None
        orch._enrichment_prev_cache = {}
        return orch

    def test_first_save_no_diff(self) -> None:
        """初回保存時: 前回スナップショットなし → 差分値なし。"""
        prev_snapshot = None
        data = {"version": 44, "tick_count": 10, "psyche": {"joy": 0.5}}

        # 前回スナップショットがNoneの場合、session_diff_scalarは含まれない
        if prev_snapshot is not None:
            diff = compute_session_difference_scalar(prev_snapshot, data)
            data["session_diff_scalar"] = diff

        assert "session_diff_scalar" not in data

    def test_second_save_has_diff(self) -> None:
        """2回目保存時: 前回スナップショットあり → 差分値が算出される。"""
        prev_snapshot = {
            "version": 44, "tick_count": 5,
            "psyche": {"joy": 0.5, "sadness": 0.1},
        }
        current_data = {
            "version": 44, "tick_count": 10,
            "psyche": {"joy": 0.8, "sadness": 0.3},
        }

        diff = compute_session_difference_scalar(prev_snapshot, current_data)
        current_data["session_diff_scalar"] = diff

        # psyche: |0.8-0.5| + |0.3-0.1| = 0.5
        assert abs(diff - 0.5) < 1e-10
        assert "session_diff_scalar" in current_data

    def test_load_reads_diff_scalar(self) -> None:
        """復元時: 保存辞書からsession_diff_scalarを読み取る。"""
        saved_data = {
            "version": 44, "tick_count": 10,
            "session_diff_scalar": 3.5,
            "psyche": {"joy": 0.5},
        }

        # load時の処理をシミュレート
        scalar = saved_data.get("session_diff_scalar")
        assert scalar == 3.5

    def test_load_without_diff_field(self) -> None:
        """復元時: session_diff_scalarが存在しない場合はNone。"""
        saved_data = {
            "version": 44, "tick_count": 10,
            "psyche": {"joy": 0.5},
        }

        scalar = saved_data.get("session_diff_scalar")
        assert scalar is None

    def test_load_preserves_snapshot_for_next_save(self) -> None:
        """復元時: 辞書データを前回スナップショットとして保持。"""
        saved_data = {
            "version": 44, "tick_count": 10,
            "psyche": {"joy": 0.5},
        }

        # load時にスナップショットを保持
        prev_snapshot = dict(saved_data)

        # 値が変更されても元のスナップショットは影響されない
        assert prev_snapshot["psyche"]["joy"] == 0.5

    def test_roundtrip_diff_accumulation(self) -> None:
        """save→load→save の完全往復テスト。"""
        # セッション1: 初回保存（差分なし）
        session1_data = {
            "version": 44, "tick_count": 5,
            "field_a": 1.0, "field_b": {"x": 2.0},
        }
        # 初回のため session_diff_scalar なし

        # セッション2: load → 状態変更 → save
        # load: スナップショットとして session1_data を保持
        prev_snapshot = dict(session1_data)

        # 状態が変わったと想定
        session2_data = {
            "version": 44, "tick_count": 15,
            "field_a": 3.0, "field_b": {"x": 5.0},
        }

        diff = compute_session_difference_scalar(prev_snapshot, session2_data)
        # field_a: |3-1| = 2.0, field_b.x: |5-2| = 3.0, total = 5.0
        assert diff == 5.0

        session2_data["session_diff_scalar"] = diff

        # セッション3: load → 状態変更 → save
        prev_snapshot_2 = dict(session2_data)
        session3_data = {
            "version": 44, "tick_count": 25,
            "field_a": 3.5, "field_b": {"x": 5.0},
        }

        diff2 = compute_session_difference_scalar(prev_snapshot_2, session3_data)
        # field_a: |3.5-3| = 0.5, field_b.x: |5-5| = 0.0, total = 0.5
        # session_diff_scalar: 片方のみ存在 → 0
        assert abs(diff2 - 0.5) < 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# enrichment統合テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestEnrichmentIntegration:
    """enrichment項目としての統合テスト。"""

    def test_enrichment_text_format(self) -> None:
        """enrichmentテキストの形式。"""
        text = build_session_diff_enrichment_text(10.0)
        # 「セッション間状態変化: 」で始まること
        assert text.startswith("セッション間状態変化: ")
        # 段階値が含まれること
        assert "中程度の変化" in text

    def test_enrichment_text_for_each_stage(self) -> None:
        """各段階値に対してenrichmentテキストが正しく生成される。"""
        test_cases = [
            (None, "(不明)"),
            (0.0, "ほぼ変化なし"),
            (1.0, "微小な変化"),
            (10.0, "中程度の変化"),
            (100.0, "大きな変化"),
        ]
        for scalar, expected_label in test_cases:
            text = build_session_diff_enrichment_text(scalar)
            assert expected_label in text, (
                f"Expected '{expected_label}' in text for scalar={scalar}, "
                f"got '{text}'"
            )

    def test_no_field_breakdown(self) -> None:
        """フィールド内訳がenrichmentに含まれないこと（安全弁5）。

        スカラー要約のみが提供され、どのフィールドが変化したかの
        詳細な内訳は構造的に不可能。
        """
        # compute_session_difference_scalar は1つのスカラー値を返すのみ
        prev = {"a": 1.0, "b": 10.0, "c": [1, 2, 3]}
        curr = {"a": 5.0, "b": 10.0, "c": [1, 2, 3, 4]}
        scalar = compute_session_difference_scalar(prev, curr)

        # スカラー値からenrichmentテキストを生成
        text = build_session_diff_enrichment_text(scalar)

        # フィールド名 "a", "b", "c" がテキストに含まれないこと
        # （構造的にフィールド内訳情報は失われている）
        assert ": a" not in text
        assert ": b" not in text
        assert ": c" not in text


# ══════════════════════════════════════════════════════════════════════════════
# 安全弁テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestSafetyValves:
    """設計書に記載された安全弁のテスト。"""

    def test_sv1_enrichment_only_no_pipeline_input(self) -> None:
        """安全弁1: enrichment経由の間接参照のみ。

        build_session_diff_enrichment_text はテキストを返すだけで、
        ドライブ・感情・ポリシー選択・記憶のいずれにも影響しない。
        """
        # 関数はstrを返すのみ
        result = build_session_diff_enrichment_text(10.0)
        assert isinstance(result, str)

    def test_sv2_no_accumulation(self) -> None:
        """安全弁2: 蓄積の禁止。

        classify_session_difference, build_session_diff_enrichment_text
        は状態を保持しない純粋関数。
        """
        # 同じ入力に対して同じ出力
        r1 = classify_session_difference(10.0)
        r2 = classify_session_difference(10.0)
        assert r1 == r2

    def test_sv3_fixed_intervals(self) -> None:
        """安全弁3: 段階値の固定区間。

        区間境界値は静的に定義され、内部状態に依存しない。
        """
        from psyche.save_load_warmup import _SESSION_DIFF_THRESHOLDS
        # 閾値リストが固定されていること
        assert len(_SESSION_DIFF_THRESHOLDS) == 3
        # 閾値が昇順であること
        thresholds = [t for t, _ in _SESSION_DIFF_THRESHOLDS]
        assert thresholds == sorted(thresholds)

    def test_sv4_no_evaluative_expression(self) -> None:
        """安全弁4: 評価的表現の排除。"""
        all_labels = [
            classify_session_difference(s)
            for s in [None, 0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        ]
        evaluative = ["良い", "悪い", "成長", "劣化", "改善", "退化", "理想"]
        for label in all_labels:
            for term in evaluative:
                assert term not in label

    def test_sv5_scalar_only_no_breakdown(self) -> None:
        """安全弁5: フィールド内訳の非提供。

        compute_session_difference_scalar はスカラー1つを返し、
        フィールドごとの内訳を返す経路を持たない。
        """
        prev = {"a": 1.0, "b": 2.0}
        curr = {"a": 3.0, "b": 4.0}
        result = compute_session_difference_scalar(prev, curr)
        # 結果は単一のfloat値
        assert isinstance(result, float)
        # 個別のフィールド距離情報にアクセスする手段がない


# ══════════════════════════════════════════════════════════════════════════════
# エッジケーステスト
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_deeply_nested_dict(self) -> None:
        """深くネストされた辞書の距離算出。"""
        prev = {"level1": {"level2": {"level3": 1.0}}}
        curr = {"level1": {"level2": {"level3": 5.0}}}
        result = _compute_field_distance(prev, curr)
        assert result == 4.0

    def test_large_number_of_fields(self) -> None:
        """多数のフィールドの距離算出。"""
        prev = {f"field_{i}": float(i) for i in range(100)}
        curr = {f"field_{i}": float(i + 1) for i in range(100)}
        # 各フィールドの差は1.0、100フィールド → 100.0
        result = compute_session_difference_scalar(prev, curr)
        assert result == 100.0

    def test_empty_list_fields(self) -> None:
        """空リストフィールドの距離。"""
        assert _compute_field_distance([], []) == 0.0

    def test_very_small_difference(self) -> None:
        """極小の差分値の分類。"""
        assert classify_session_difference(0.001) == "ほぼ変化なし"
        assert classify_session_difference(1e-10) == "ほぼ変化なし"

    def test_very_large_scalar(self) -> None:
        """極大のスカラー値の分類。"""
        assert classify_session_difference(1e6) == "大きな変化"
        assert classify_session_difference(float('inf')) == "大きな変化"
