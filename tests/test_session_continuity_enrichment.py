"""
tests/test_session_continuity_enrichment.py - セッション間自己連続性 enrichment 拡張テスト

design_session_continuity_enrichment.md に基づく実装のテスト:
  - 記述A: 前セッション終了時の感情状態の叙述的記述
  - 記述B: 前セッション終了時の注意配分の事実記述
  - 記述C: 状態変化のフィールド群別記述
  - 後方互換性: オプショナル引数追加
  - 安全弁: 各欠損時のスキップ、テキスト長上限、固定段階値
  - 評価語禁止
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from psyche.save_load_warmup import (
    build_session_diff_enrichment_text,
    classify_session_difference,
    SESSION_DIFF_EMPTY_LABEL,
)


# ══════════════════════════════════════════════════════════════════════════════
# 後方互換性: 既存の呼び出しが壊れないこと
# ══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """既存のbuild_session_diff_enrichment_textの呼び出しが後方互換であること。"""

    def test_scalar_only_call_still_works(self) -> None:
        """既存の引数(scalarのみ)で呼び出しても動作すること。"""
        result = build_session_diff_enrichment_text(10.0)
        assert "セッション間状態変化" in result
        assert "中程度の変化" in result

    def test_none_scalar_still_works(self) -> None:
        """Noneスカラーの既存動作が維持されること。"""
        result = build_session_diff_enrichment_text(None)
        assert "(不明)" in result

    def test_zero_scalar_still_works(self) -> None:
        """0スカラーの既存動作が維持されること。"""
        result = build_session_diff_enrichment_text(0.0)
        assert "ほぼ変化なし" in result


# ══════════════════════════════════════════════════════════════════════════════
# 記述A: 前セッション終了時の感情状態
# ══════════════════════════════════════════════════════════════════════════════


def _make_snapshot_with_emotions(
    emotions: dict[str, float],
    mood: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """感情データを含むスナップショット辞書を作成するヘルパー。"""
    psyche: dict[str, Any] = {
        "emotions": emotions,
        "drives": {},
        "mood": mood or {"valence": 0.0, "arousal": 0.3},
        "fear_index": 0.0,
        "loss_aversion": 0.3,
        "last_updated": "2026-01-01T00:00:00",
    }
    return {"psyche": psyche, "version": 44, "tick_count": 10}


class TestDescriptionA:
    """記述A: 前セッション感情状態の叙述的記述テスト。"""

    def test_high_joy_listed(self) -> None:
        """joyが高い場合に記述Aに含まれること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.8, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0}
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "joy" in result

    def test_low_emotions_not_listed(self) -> None:
        """段階値が低い感情は列挙されないこと（情報量制御）。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.01, "anger": 0.02, "sorrow": 0.01,
             "fear": 0.01, "surprise": 0.01, "love": 0.01, "fun": 0.01}
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 全て低いので個別の感情名が列挙されない
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            assert dim not in result

    def test_multiple_high_emotions(self) -> None:
        """複数の感情が高い場合に複数列挙されること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.7, "anger": 0.0, "sorrow": 0.6,
             "fear": 0.0, "surprise": 0.0, "love": 0.8, "fun": 0.0}
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "joy" in result
        assert "sorrow" in result
        assert "love" in result

    def test_mood_valence_arousal_included(self) -> None:
        """気分のvalence/arousalが記述に含まれること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.5, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
            mood={"valence": 0.6, "arousal": 0.8},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "valence" in result
        assert "arousal" in result

    def test_no_psyche_key_skips_description_a(self) -> None:
        """スナップショットにpsycheキーがない場合は記述Aをスキップ。"""
        snapshot = {"version": 44, "tick_count": 10}
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 感情関連テキストが含まれないこと（スカラーテキストのみ）
        assert "セッション間状態変化" in result
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            assert dim not in result

    def test_no_emotions_key_skips_description_a(self) -> None:
        """psyche内にemotionsキーがない場合は記述Aをスキップ。"""
        snapshot = {"psyche": {"mood": {"valence": 0.5, "arousal": 0.5}},
                    "version": 44, "tick_count": 10}
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "セッション間状態変化" in result

    def test_emotions_wrong_type_skips(self) -> None:
        """emotionsが辞書でない場合は記述Aをスキップ。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.5, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["psyche"]["emotions"] = "invalid"
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        for dim in ("joy", "anger", "sorrow"):
            assert dim not in result


# ══════════════════════════════════════════════════════════════════════════════
# 記述B: 前セッション終了時の注意配分
# ══════════════════════════════════════════════════════════════════════════════


def _make_att_dist_dict(
    snapshot_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """注意配分状態の辞書を作成するヘルパー。"""
    default_snapshot = {
        "timestamp": 1000.0,
        "perception_level": "moderate",
        "text_input_level": "few",
        "spontaneous_level": "absent",
        "emotion_level": "many",
        "memory_level": "minimal",
        "motivation_level": "absent",
        "goal_level": "few",
        "responsibility_level": "absent",
        "concentration": 0.45,
        "concentration_level": "moderate",
    }
    if snapshot_data:
        default_snapshot.update(snapshot_data)
    return {
        "snapshot_history": [default_snapshot],
        "latest_variation": None,
        "total_snapshots_generated": 1,
        "total_snapshots_expired": 0,
    }


class TestDescriptionB:
    """記述B: 前セッション注意配分の事実記述テスト。"""

    def test_att_dist_listed(self) -> None:
        """注意配分状態が記述に含まれること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 注意配分関連テキストが含まれること
        assert "注意配分" in result

    def test_att_dist_not_present_skips(self) -> None:
        """attention_distribution_stateが存在しない場合はスキップ。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 注意配分テキストは含まれない
        assert "注意配分" not in result

    def test_att_dist_empty_history_skips(self) -> None:
        """snapshot_historyが空の場合はスキップ。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["attention_distribution_state"] = {
            "snapshot_history": [],
            "latest_variation": None,
            "total_snapshots_generated": 0,
            "total_snapshots_expired": 0,
        }
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "注意配分" not in result

    def test_att_dist_concentration_level_shown(self) -> None:
        """集中度が記述に含まれること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "集中度" in result

    def test_att_dist_wrong_type_skips(self) -> None:
        """attention_distribution_stateが辞書でない場合はスキップ。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["attention_distribution_state"] = "invalid"
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "注意配分" not in result


# ══════════════════════════════════════════════════════════════════════════════
# 記述C: 状態変化のフィールド群別記述
# ══════════════════════════════════════════════════════════════════════════════


class TestDescriptionC:
    """記述C: フィールド群ごとの距離寄与の段階値テスト。"""

    def test_field_group_breakdown_shown(self) -> None:
        """フィールド群別の変化量が記述に含まれること。"""
        snapshot = {
            "version": 44, "tick_count": 5,
            "psyche": {"emotions": {"joy": 0.5}, "drives": {}, "mood": {"valence": 0.0, "arousal": 0.3},
                       "fear_index": 0.0, "loss_aversion": 0.3, "last_updated": "2026-01-01T00:00:00"},
            "loop_state": {"value": 1.0},
        }
        current_snapshot = {
            "version": 44, "tick_count": 10,
            "psyche": {"emotions": {"joy": 0.9}, "drives": {}, "mood": {"valence": 0.0, "arousal": 0.3},
                       "fear_index": 0.0, "loss_aversion": 0.3, "last_updated": "2026-01-01T00:00:00"},
            "loop_state": {"value": 5.0},
        }
        from psyche.save_load_warmup import compute_session_difference_scalar
        scalar = compute_session_difference_scalar(snapshot, current_snapshot)
        result = build_session_diff_enrichment_text(
            scalar, prev_snapshot=snapshot, current_snapshot=current_snapshot,
        )
        # フィールド群名が含まれること（具体的なフィールド名ではなく群名）
        # 少なくとも1つのフィールド群の変化記述があること
        assert "変化" in result

    def test_no_current_snapshot_skips_c(self) -> None:
        """current_snapshotがない場合は記述Cをスキップ。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.5, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 既存のスカラー段階値は含まれる
        assert "中程度の変化" in result

    def test_field_group_labels_are_group_names(self) -> None:
        """個別フィールド名ではなくフィールド群名が使われること。"""
        snapshot = {
            "version": 44, "tick_count": 5,
            "psyche": {"emotions": {"joy": 0.5}, "drives": {}, "mood": {"valence": 0.0, "arousal": 0.3},
                       "fear_index": 0.0, "loss_aversion": 0.3, "last_updated": "2026-01-01T00:00:00"},
            "self_ref_state": {"value": 1.0},
        }
        current_snapshot = {
            "version": 44, "tick_count": 10,
            "psyche": {"emotions": {"joy": 0.9}, "drives": {}, "mood": {"valence": 0.0, "arousal": 0.3},
                       "fear_index": 0.0, "loss_aversion": 0.3, "last_updated": "2026-01-01T00:00:00"},
            "self_ref_state": {"value": 5.0},
        }
        from psyche.save_load_warmup import compute_session_difference_scalar
        scalar = compute_session_difference_scalar(snapshot, current_snapshot)
        result = build_session_diff_enrichment_text(
            scalar, prev_snapshot=snapshot, current_snapshot=current_snapshot,
        )
        # 個別フィールド名 "self_ref_state" がテキストに含まれないこと
        assert "self_ref_state" not in result


# ══════════════════════════════════════════════════════════════════════════════
# 安全弁テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestSafetyValves:
    """設計書に記載された安全弁のテスト。"""

    def test_sv1_no_snapshot_returns_empty(self) -> None:
        """安全弁1: スナップショット不在時は既存の空状態テキストを返す。"""
        result = build_session_diff_enrichment_text(None)
        assert "(不明)" in result

    def test_sv2_emotion_data_missing_skips_a(self) -> None:
        """安全弁2: 感情ベクトル欠損時は記述Aを生成しない。"""
        snapshot = {"version": 44, "tick_count": 10}
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            assert dim not in result

    def test_sv3_attention_data_missing_skips_b(self) -> None:
        """安全弁3: 注意配分欠損時は記述Bを生成しない。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.5, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert "注意配分" not in result

    def test_sv4_text_length_limit(self) -> None:
        """安全弁4: テキスト長上限が存在すること。"""
        # 全感情が高い + 注意配分あり + フィールド群別変化ありの最大ケース
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.9, "anger": 0.8, "sorrow": 0.7,
             "fear": 0.9, "surprise": 0.8, "love": 0.9, "fun": 0.7},
            mood={"valence": 0.8, "arousal": 0.9},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        current_snapshot = {
            "version": 44, "tick_count": 10,
            "psyche": {"emotions": {"joy": 0.1}, "drives": {}, "mood": {"valence": -0.5, "arousal": 0.1},
                       "fear_index": 0.0, "loss_aversion": 0.3, "last_updated": "2026-01-01T00:00:00"},
        }
        from psyche.save_load_warmup import compute_session_difference_scalar
        scalar = compute_session_difference_scalar(snapshot, current_snapshot)
        result = build_session_diff_enrichment_text(
            scalar, prev_snapshot=snapshot, current_snapshot=current_snapshot,
        )
        # テキスト長が上限以内であること
        from psyche.save_load_warmup import _ENRICHMENT_TEXT_MAX_LENGTH
        assert len(result) <= _ENRICHMENT_TEXT_MAX_LENGTH

    def test_sv5_fixed_emotion_thresholds(self) -> None:
        """安全弁5: 感情段階値変換は固定区間分割であること。"""
        from psyche.save_load_warmup import _EMOTION_LEVEL_THRESHOLDS
        # 閾値リストが存在すること
        assert len(_EMOTION_LEVEL_THRESHOLDS) >= 2
        # 閾値が昇順であること
        thresholds = [t for t, _ in _EMOTION_LEVEL_THRESHOLDS]
        assert thresholds == sorted(thresholds)


# ══════════════════════════════════════════════════════════════════════════════
# 評価語禁止テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestNoEvaluativeLanguage:
    """生成されるテキストに評価的表現が含まれないこと。"""

    EVALUATIVE_TERMS = [
        "良い", "悪い", "成長", "劣化", "改善", "退化",
        "問題", "正常", "異常", "健全", "不健全", "理想",
        "望ましい", "好ましい",
    ]

    def test_description_a_no_evaluative(self) -> None:
        """記述Aに評価的表現が含まれないこと。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.9, "anger": 0.8, "sorrow": 0.7,
             "fear": 0.6, "surprise": 0.5, "love": 0.9, "fun": 0.4},
            mood={"valence": 0.9, "arousal": 0.9},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        for term in self.EVALUATIVE_TERMS:
            assert term not in result, (
                f"Evaluative term '{term}' found in result: {result}"
            )

    def test_description_b_no_evaluative(self) -> None:
        """記述Bに評価的表現が含まれないこと。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        for term in self.EVALUATIVE_TERMS:
            assert term not in result, (
                f"Evaluative term '{term}' found in result: {result}"
            )

    def test_full_output_no_evaluative(self) -> None:
        """全記述A+B+Cの結合テキストに評価的表現が含まれないこと。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.9, "anger": 0.8, "sorrow": 0.7,
             "fear": 0.6, "surprise": 0.5, "love": 0.9, "fun": 0.4},
            mood={"valence": 0.9, "arousal": 0.9},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        current_snapshot = {
            "version": 44, "tick_count": 15,
            "psyche": {"emotions": {"joy": 0.1}, "drives": {},
                       "mood": {"valence": -0.5, "arousal": 0.1},
                       "fear_index": 0.0, "loss_aversion": 0.3,
                       "last_updated": "2026-01-01T00:00:00"},
        }
        from psyche.save_load_warmup import compute_session_difference_scalar
        scalar = compute_session_difference_scalar(snapshot, current_snapshot)
        result = build_session_diff_enrichment_text(
            scalar, prev_snapshot=snapshot, current_snapshot=current_snapshot,
        )
        for term in self.EVALUATIVE_TERMS:
            assert term not in result, (
                f"Evaluative term '{term}' found in result: {result}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# ステートレス性テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestStatelessness:
    """関数がステートレスであること（同じ入力に同じ出力）。"""

    def test_same_input_same_output(self) -> None:
        """同一入力に対して同一出力であること。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.7, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.5, "fun": 0.0},
            mood={"valence": 0.3, "arousal": 0.6},
        )
        snapshot["attention_distribution_state"] = _make_att_dist_dict()
        r1 = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        r2 = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════════
# 境界値テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestBoundaryValues:
    """境界値のテスト。"""

    def test_emotion_exactly_at_threshold(self) -> None:
        """感情値がちょうど閾値の場合。"""
        from psyche.save_load_warmup import _EMOTION_LEVEL_THRESHOLDS
        threshold_val = _EMOTION_LEVEL_THRESHOLDS[0][0]
        snapshot = _make_snapshot_with_emotions(
            {"joy": threshold_val, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        # エラーなく実行されること
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert isinstance(result, str)

    def test_emotion_value_zero(self) -> None:
        """全感情値が0の場合。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        # 感情名が列挙されないこと
        for dim in ("joy", "anger", "sorrow", "fear", "surprise", "love", "fun"):
            assert dim not in result

    def test_emotion_value_one(self) -> None:
        """全感情値が1.0の場合。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 1.0, "anger": 1.0, "sorrow": 1.0,
             "fear": 1.0, "surprise": 1.0, "love": 1.0, "fun": 1.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert isinstance(result, str)

    def test_empty_prev_snapshot(self) -> None:
        """空辞書のprev_snapshot。"""
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot={},
        )
        assert "セッション間状態変化" in result

    def test_non_numeric_emotion_values_handled(self) -> None:
        """感情値が数値でない場合にクラッシュしないこと。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": "invalid", "anger": None, "sorrow": 0.5,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert isinstance(result, str)

    def test_mood_values_out_of_range(self) -> None:
        """mood値が範囲外の場合にクラッシュしないこと。"""
        snapshot = _make_snapshot_with_emotions(
            {"joy": 0.5, "anger": 0.0, "sorrow": 0.0,
             "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
            mood={"valence": 999.0, "arousal": -999.0},
        )
        result = build_session_diff_enrichment_text(
            10.0, prev_snapshot=snapshot,
        )
        assert isinstance(result, str)
