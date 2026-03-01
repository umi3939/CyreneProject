"""
tests/test_expression_quality_verification.py - 代弁品質事実記録のテスト

tools/expression_quality_verification.py の全機能をテストする。

テスト対象:
- 環境変数制御（安全弁5）
- 初期状態
- 第1段: 対構成（記録生成）
- 第2段: 蓄積（FIFOバッファ、安全弁4）
- 第3段: 参照提供
- セッションサマリー
- 感情語彙辞書（安全弁6）
- テキスト特徴量抽出
- 設定バリデーション
- 構造的分離の確認
- パターン抽出禁止（安全弁7）
- 不変性の確認
"""

from __future__ import annotations

import os
import time
import pytest
from typing import Any
from unittest.mock import patch

from tools.expression_quality_verification import (
    ExpressionQualityVerification,
    ExpressionQualityConfig,
    ExpressionRecord,
    _is_monitor_enabled,
    _extract_emotion_labels,
    _has_question_mark,
    _count_sentences,
    _EXPRESSION_EMOTION_VOCAB,
)


# ── テスト用ヘルパー ──────────────────────────────────────────────────


def _make_meta(
    emotion: str = "joy",
    intensity: float = 0.5,
    action: str = "共感する",
) -> dict[str, Any]:
    """テスト用のメタ情報辞書を作成する。"""
    return {"emotion": emotion, "intensity": intensity, "action": action}


def _record_one(
    monitor: ExpressionQualityVerification,
    tick: int = 1,
    pathway: str = "text",
    policy_label: str = "共感する",
    policy_rationale: str = "相手の気持ちに寄り添う",
    emotion_label: str = "happy",
    emotion_intensity: float = 0.6,
    mood_valence: float = 0.3,
    mood_arousal: float = 0.4,
    enrichment_chars: int = 500,
    utterance: str = "ふふっ、楽しいわね♪",
    meta: dict[str, Any] | None = None,
    fallback: bool = False,
) -> ExpressionRecord | None:
    """テスト用に1件の記録を生成する。"""
    if meta is None:
        meta = _make_meta()
    return monitor.record_expression(
        tick_number=tick,
        input_pathway=pathway,
        policy_label=policy_label,
        policy_rationale=policy_rationale,
        policy_emotion_label=emotion_label,
        policy_emotion_intensity=emotion_intensity,
        policy_mood_valence=mood_valence,
        policy_mood_arousal=mood_arousal,
        enrichment_char_count=enrichment_chars,
        utterance_text=utterance,
        utterance_meta=meta,
        is_fallback=fallback,
    )


# ── 環境変数制御のテスト ──────────────────────────────────────────────


class TestEnvironmentControl:
    """安全弁5: 環境変数による完全無効化のテスト。"""

    def test_monitor_disabled_by_default(self) -> None:
        """環境変数未設定時はモニタリング無効。"""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_monitor_enabled() is False

    def test_monitor_disabled_when_zero(self) -> None:
        """CYRENE_MONITOR=0 で無効。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            assert _is_monitor_enabled() is False

    def test_monitor_enabled_when_one(self) -> None:
        """CYRENE_MONITOR=1 で有効。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            assert _is_monitor_enabled() is True

    def test_explicit_enable_overrides_env(self) -> None:
        """明示的なenabled=Trueが環境変数に優先する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            monitor = ExpressionQualityVerification(enabled=True)
            assert monitor.enabled is True

    def test_explicit_disable_overrides_env(self) -> None:
        """明示的なenabled=Falseが環境変数に優先する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            monitor = ExpressionQualityVerification(enabled=False)
            assert monitor.enabled is False

    def test_disabled_monitor_record_returns_none(self) -> None:
        """無効時はrecord_expressionがNoneを返す。"""
        monitor = ExpressionQualityVerification(enabled=False)
        result = _record_one(monitor)
        assert result is None

    def test_disabled_monitor_recent_returns_empty(self) -> None:
        """無効時はget_recent_recordsが空リストを返す。"""
        monitor = ExpressionQualityVerification(enabled=False)
        assert monitor.get_recent_records() == []

    def test_disabled_monitor_summary_returns_none(self) -> None:
        """無効時はemit_session_summaryがNoneを返す。"""
        monitor = ExpressionQualityVerification(enabled=False)
        assert monitor.emit_session_summary() is None

    def test_disabled_monitor_no_accumulation(self) -> None:
        """無効時は記録が蓄積されない。"""
        monitor = ExpressionQualityVerification(enabled=False)
        _record_one(monitor)
        assert monitor.record_count == 0
        assert monitor.fallback_count == 0
        assert monitor.buffer_size == 0


# ── 初期状態のテスト ──────────────────────────────────────────────────


class TestInitialState:
    """初期化直後の状態確認。"""

    def test_initial_record_count_zero(self) -> None:
        """初期状態で記録件数がゼロ。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert monitor.record_count == 0

    def test_initial_fallback_count_zero(self) -> None:
        """初期状態でフォールバック件数がゼロ。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert monitor.fallback_count == 0

    def test_initial_buffer_empty(self) -> None:
        """初期状態でバッファが空。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert monitor.buffer_size == 0

    def test_initial_recent_records_empty(self) -> None:
        """初期状態で直近記録が空。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert monitor.get_recent_records() == []

    def test_initial_summary_empty(self) -> None:
        """初期状態のサマリーが全ゼロ。"""
        monitor = ExpressionQualityVerification(enabled=True)
        summary = monitor.get_summary()
        assert summary["record_count"] == 0
        assert summary["fallback_count"] == 0
        assert summary["buffer_size"] == 0


# ── 第1段: 対構成のテスト ────────────────────────────────────────────


class TestRecordConstruction:
    """第1段: 対構成（記録生成）のテスト。"""

    def test_record_returns_expression_record(self) -> None:
        """記録はExpressionRecordインスタンスを返す。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor)
        assert isinstance(record, ExpressionRecord)

    def test_record_tick_number(self) -> None:
        """ティック番号が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, tick=42)
        assert record is not None
        assert record.tick_number == 42

    def test_record_input_pathway(self) -> None:
        """入力経路ラベルが正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for pathway in ("vision", "text", "internal"):
            record = _record_one(monitor, pathway=pathway)
            assert record is not None
            assert record.input_pathway == pathway

    def test_record_invalid_pathway_normalized(self) -> None:
        """無効な入力経路ラベルは"unknown"に正規化される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, pathway="invalid")
        assert record is not None
        assert record.input_pathway == "unknown"

    def test_record_policy_fields(self) -> None:
        """方針側フィールドが正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(
            monitor,
            policy_label="質問する",
            policy_rationale="詳細を確認する",
            emotion_label="curious",
            emotion_intensity=0.7,
            mood_valence=0.2,
            mood_arousal=0.5,
            enrichment_chars=800,
        )
        assert record is not None
        assert record.policy_label == "質問する"
        assert record.policy_rationale == "詳細を確認する"
        assert record.policy_emotion_label == "curious"
        assert record.policy_emotion_intensity == 0.7
        assert record.policy_mood_valence == 0.2
        assert record.policy_mood_arousal == 0.5
        assert record.enrichment_char_count == 800

    def test_record_utterance_text(self) -> None:
        """発話テキスト全文が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "そうなんだ！嬉しいわね♪"
        record = _record_one(monitor, utterance=text)
        assert record is not None
        assert record.utterance_text == text

    def test_record_utterance_char_count(self) -> None:
        """発話テキスト文字数が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "abc"
        record = _record_one(monitor, utterance=text)
        assert record is not None
        assert record.utterance_char_count == 3

    def test_record_utterance_sentence_count(self) -> None:
        """発話テキスト文数が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "嬉しい。楽しい！"
        record = _record_one(monitor, utterance=text)
        assert record is not None
        assert record.utterance_sentence_count == 2

    def test_record_utterance_meta(self) -> None:
        """発話メタ情報が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        meta = _make_meta(emotion="sadness", intensity=0.8, action="慰める")
        record = _record_one(monitor, meta=meta)
        assert record is not None
        assert record.utterance_meta_emotion == "sadness"
        assert record.utterance_meta_intensity == 0.8
        assert record.utterance_meta_action == "慰める"

    def test_record_fallback_flag(self) -> None:
        """フォールバックフラグが正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record_normal = _record_one(monitor, fallback=False)
        record_fb = _record_one(monitor, fallback=True)
        assert record_normal is not None
        assert record_normal.is_fallback is False
        assert record_fb is not None
        assert record_fb.is_fallback is True

    def test_record_emotion_labels(self) -> None:
        """発話テキスト内の感情語彙ラベルが記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "嬉しい！怖いかも..."
        record = _record_one(monitor, utterance=text)
        assert record is not None
        labels = list(record.utterance_emotion_labels)
        assert "happy" in labels
        assert "scared" in labels

    def test_record_question_mark_present(self) -> None:
        """疑問符有無が正しく記録される（あり）。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="どうしたの？")
        assert record is not None
        assert record.has_question_mark is True

    def test_record_question_mark_absent(self) -> None:
        """疑問符有無が正しく記録される（なし）。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="そうね。")
        assert record is not None
        assert record.has_question_mark is False

    def test_record_timestamp_set(self) -> None:
        """タイムスタンプが設定される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        before = time.time()
        record = _record_one(monitor)
        after = time.time()
        assert record is not None
        assert before <= record.timestamp <= after

    def test_record_with_none_meta(self) -> None:
        """メタ情報がNoneでも安全に処理される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = monitor.record_expression(
            tick_number=1,
            input_pathway="text",
            policy_label="共感する",
            policy_rationale="テスト",
            policy_emotion_label="happy",
            policy_emotion_intensity=0.5,
            policy_mood_valence=0.0,
            policy_mood_arousal=0.0,
            enrichment_char_count=0,
            utterance_text="テスト",
            utterance_meta=None,  # type: ignore
            is_fallback=False,
        )
        assert record is not None
        assert record.utterance_meta_emotion == ""
        assert record.utterance_meta_intensity == 0.0
        assert record.utterance_meta_action == ""

    def test_record_with_empty_utterance(self) -> None:
        """空の発話テキストでも安全に処理される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="")
        assert record is not None
        assert record.utterance_text == ""
        assert record.utterance_char_count == 0
        assert record.utterance_sentence_count == 0
        assert record.utterance_emotion_labels == ()
        assert record.has_question_mark is False


# ── 第2段: 蓄積のテスト ──────────────────────────────────────────────


class TestAccumulation:
    """第2段: 蓄積（FIFOバッファ）のテスト。"""

    def test_buffer_grows_with_records(self) -> None:
        """記録のたびにバッファが成長する。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(5):
            _record_one(monitor, tick=i)
        assert monitor.buffer_size == 5

    def test_record_count_increments(self) -> None:
        """記録件数カウンタが正しくインクリメントされる。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(3):
            _record_one(monitor, tick=i)
        assert monitor.record_count == 3

    def test_fallback_count_increments(self) -> None:
        """フォールバック件数カウンタが正しくインクリメントされる。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1, fallback=False)
        _record_one(monitor, tick=2, fallback=True)
        _record_one(monitor, tick=3, fallback=True)
        assert monitor.fallback_count == 2
        assert monitor.record_count == 3

    def test_fifo_upper_limit(self) -> None:
        """安全弁4: FIFOバッファの上限超過で最古の記録が消失する。"""
        config = ExpressionQualityConfig(max_buffer_size=5)
        monitor = ExpressionQualityVerification(config=config, enabled=True)
        for i in range(10):
            _record_one(monitor, tick=i)
        # バッファサイズは上限を超えない
        assert monitor.buffer_size == 5
        # 記録件数カウンタはインクリメントされ続ける
        assert monitor.record_count == 10
        # 最古の記録（tick=0-4）が消失し、tick=5-9が残る
        recent = monitor.get_recent_records(count=5)
        ticks = [r["tick_number"] for r in recent]
        assert 0 not in ticks
        assert 9 in ticks

    def test_fifo_preserves_order(self) -> None:
        """FIFOバッファが記録の時間順を保持する。"""
        config = ExpressionQualityConfig(max_buffer_size=10)
        monitor = ExpressionQualityVerification(config=config, enabled=True)
        for i in range(5):
            _record_one(monitor, tick=i)
        # 新しい順に返される
        recent = monitor.get_recent_records(count=5)
        ticks = [r["tick_number"] for r in recent]
        assert ticks == [4, 3, 2, 1, 0]


# ── 第3段: 参照提供のテスト ──────────────────────────────────────────


class TestRecentRecords:
    """第3段: 参照提供のテスト。"""

    def test_returns_dict_list(self) -> None:
        """辞書のリストを返す。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor)
        recent = monitor.get_recent_records()
        assert isinstance(recent, list)
        assert len(recent) == 1
        assert isinstance(recent[0], dict)

    def test_returns_recent_n(self) -> None:
        """指定した件数だけ返す。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(10):
            _record_one(monitor, tick=i)
        recent = monitor.get_recent_records(count=3)
        assert len(recent) == 3
        # 新しい順
        assert recent[0]["tick_number"] == 9
        assert recent[1]["tick_number"] == 8
        assert recent[2]["tick_number"] == 7

    def test_returns_default_count(self) -> None:
        """デフォルト件数を返す。"""
        config = ExpressionQualityConfig(recent_count=5)
        monitor = ExpressionQualityVerification(config=config, enabled=True)
        for i in range(10):
            _record_one(monitor, tick=i)
        recent = monitor.get_recent_records()
        assert len(recent) == 5

    def test_returns_all_when_fewer_than_count(self) -> None:
        """バッファ内の件数が指定件数より少ない場合は全件返す。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(3):
            _record_one(monitor, tick=i)
        recent = monitor.get_recent_records(count=10)
        assert len(recent) == 3

    def test_record_dict_has_all_fields(self) -> None:
        """記録辞書が全フィールドを含む。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor)
        recent = monitor.get_recent_records(count=1)
        assert len(recent) == 1
        record = recent[0]
        expected_fields = {
            "tick_number", "input_pathway", "timestamp",
            "policy_label", "policy_rationale",
            "policy_emotion_label", "policy_emotion_intensity",
            "policy_mood_valence", "policy_mood_arousal",
            "enrichment_char_count",
            "utterance_text", "utterance_char_count", "utterance_sentence_count",
            "utterance_meta_emotion", "utterance_meta_intensity",
            "utterance_meta_action",
            "is_fallback",
            "utterance_emotion_labels", "has_question_mark",
        }
        assert set(record.keys()) == expected_fields

    def test_no_aggregation_in_records(self) -> None:
        """安全弁7: 記録に集計・統計・傾向情報が含まれない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(5):
            _record_one(monitor, tick=i)
        recent = monitor.get_recent_records(count=5)
        for record in recent:
            # 集計・統計関連のキーが存在しないことを確認
            for forbidden_key in (
                "average", "mean", "total", "trend", "pattern",
                "score", "quality", "match", "deviation",
                "count", "frequency", "ratio",
            ):
                assert forbidden_key not in record, (
                    f"Record should not contain aggregation key '{forbidden_key}'"
                )


# ── セッションサマリーのテスト ────────────────────────────────────────


class TestSessionSummary:
    """セッションサマリーのテスト。"""

    def test_emit_session_summary(self) -> None:
        """セッションサマリーが正しい構造を持つ。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1)
        _record_one(monitor, tick=2, fallback=True)
        summary = monitor.emit_session_summary()
        assert summary is not None
        assert summary["type"] == "expression_quality_session_summary"
        assert summary["record_count"] == 2
        assert summary["fallback_count"] == 1
        assert summary["buffer_size"] == 2
        assert "timestamp" in summary

    def test_get_summary_read_only(self) -> None:
        """get_summaryが読み取り専用情報を返す。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1)
        _record_one(monitor, tick=2, fallback=True)
        summary = monitor.get_summary()
        assert summary["record_count"] == 2
        assert summary["fallback_count"] == 1
        assert summary["buffer_size"] == 2


# ── 感情語彙辞書のテスト ────────────────────────────────────────────


class TestEmotionVocab:
    """安全弁6: 感情語彙辞書の静的固定のテスト。"""

    def test_vocab_is_dict(self) -> None:
        """辞書がdict型である。"""
        assert isinstance(_EXPRESSION_EMOTION_VOCAB, dict)

    def test_vocab_not_empty(self) -> None:
        """辞書が空でない。"""
        assert len(_EXPRESSION_EMOTION_VOCAB) > 0

    def test_vocab_values_are_strings(self) -> None:
        """辞書の値はすべて文字列(ラベル)。"""
        for keyword, label in _EXPRESSION_EMOTION_VOCAB.items():
            assert isinstance(keyword, str), f"Key '{keyword}' is not str"
            assert isinstance(label, str), f"Value for '{keyword}' is not str"

    def test_vocab_covers_major_emotions(self) -> None:
        """辞書が主要な感情ラベルをカバーしている。"""
        labels = set(_EXPRESSION_EMOTION_VOCAB.values())
        expected = {"happy", "sad", "angry", "surprised", "scared", "loving"}
        assert expected.issubset(labels)


# ── テキスト特徴量抽出のテスト ────────────────────────────────────────


class TestTextFeatures:
    """テキスト特徴量抽出関数のテスト。"""

    def test_extract_emotion_labels_basic(self) -> None:
        """基本的な感情語彙の抽出。"""
        labels = _extract_emotion_labels("嬉しい！怖い...")
        assert "happy" in labels
        assert "scared" in labels

    def test_extract_emotion_labels_no_duplicates(self) -> None:
        """同一ラベルの重複は除去される。"""
        labels = _extract_emotion_labels("嬉しい、楽しい、幸せ")
        assert labels.count("happy") == 1

    def test_extract_emotion_labels_empty(self) -> None:
        """感情語彙が含まれない場合は空リスト。"""
        labels = _extract_emotion_labels("今日は天気がいい")
        assert labels == []

    def test_extract_emotion_labels_empty_text(self) -> None:
        """空テキストでは空リスト。"""
        labels = _extract_emotion_labels("")
        assert labels == []

    def test_has_question_mark_fullwidth(self) -> None:
        """全角疑問符の検出。"""
        assert _has_question_mark("どうしたの？") is True

    def test_has_question_mark_halfwidth(self) -> None:
        """半角疑問符の検出。"""
        assert _has_question_mark("Really?") is True

    def test_has_question_mark_absent(self) -> None:
        """疑問符なし。"""
        assert _has_question_mark("そうだね。") is False

    def test_count_sentences_period(self) -> None:
        """句点で分割。"""
        assert _count_sentences("こんにちは。元気？") == 2

    def test_count_sentences_exclamation(self) -> None:
        """感嘆符で分割。"""
        assert _count_sentences("すごい！楽しい！") == 2

    def test_count_sentences_mixed(self) -> None:
        """混合区切り文字。"""
        assert _count_sentences("そう。本当？すごい！") == 3

    def test_count_sentences_no_delimiter(self) -> None:
        """区切り文字なしの場合は1文。"""
        assert _count_sentences("テキスト") == 1

    def test_count_sentences_empty(self) -> None:
        """空テキストは0文。"""
        assert _count_sentences("") == 0

    def test_count_sentences_whitespace_only(self) -> None:
        """空白のみのテキストは0文。"""
        assert _count_sentences("   ") == 0

    def test_count_sentences_halfwidth(self) -> None:
        """半角の区切り文字。"""
        assert _count_sentences("Hello! World?") == 2


# ── 設定バリデーションのテスト ────────────────────────────────────────


class TestConfigValidation:
    """設定パラメータのバリデーションテスト。"""

    def test_default_config(self) -> None:
        """デフォルト設定値。"""
        config = ExpressionQualityConfig()
        assert config.max_buffer_size == 200
        assert config.recent_count == 20

    def test_negative_buffer_size_reset(self) -> None:
        """負のバッファサイズはデフォルトにリセットされる。"""
        config = ExpressionQualityConfig(max_buffer_size=-1)
        assert config.max_buffer_size == 200

    def test_zero_buffer_size_reset(self) -> None:
        """ゼロのバッファサイズはデフォルトにリセットされる。"""
        config = ExpressionQualityConfig(max_buffer_size=0)
        assert config.max_buffer_size == 200

    def test_negative_recent_count_reset(self) -> None:
        """負のrecent_countはデフォルトにリセットされる。"""
        config = ExpressionQualityConfig(recent_count=-1)
        assert config.recent_count == 20

    def test_recent_count_exceeds_buffer(self) -> None:
        """recent_countがバッファサイズを超えた場合はバッファサイズに合わせる。"""
        config = ExpressionQualityConfig(max_buffer_size=10, recent_count=50)
        assert config.recent_count == 10

    def test_custom_config(self) -> None:
        """カスタム設定が正しく適用される。"""
        config = ExpressionQualityConfig(max_buffer_size=500, recent_count=50)
        assert config.max_buffer_size == 500
        assert config.recent_count == 50


# ── ExpressionRecordの不変性テスト ────────────────────────────────────


class TestRecordImmutability:
    """記録構造体の不変性テスト。"""

    def test_record_is_frozen(self) -> None:
        """ExpressionRecordは不変(frozen dataclass)。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor)
        assert record is not None
        with pytest.raises(AttributeError):
            record.tick_number = 999  # type: ignore

    def test_record_to_dict_returns_copy(self) -> None:
        """to_dictは新しい辞書を返す（元のデータは影響を受けない）。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor)
        assert record is not None
        d1 = record.to_dict()
        d2 = record.to_dict()
        assert d1 == d2
        assert d1 is not d2

    def test_emotion_labels_as_tuple(self) -> None:
        """感情語彙ラベルはtupleとして記録される（不変）。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="嬉しい")
        assert record is not None
        assert isinstance(record.utterance_emotion_labels, tuple)

    def test_to_dict_emotion_labels_as_list(self) -> None:
        """to_dictでは感情語彙ラベルがlistに変換される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="嬉しい")
        assert record is not None
        d = record.to_dict()
        assert isinstance(d["utterance_emotion_labels"], list)


# ── 構造的分離の確認テスト ────────────────────────────────────────────


class TestStructuralIsolation:
    """構造的分離の確認テスト。"""

    def test_no_enrichment_method(self) -> None:
        """安全弁1: enrichment出力を生成するメソッドを持たない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        # enrichment関連のメソッドが存在しないことを確認
        assert not hasattr(monitor, "get_enrichment")
        assert not hasattr(monitor, "get_prompt_enrichment")
        assert not hasattr(monitor, "generate_enrichment")

    def test_no_save_load_method(self) -> None:
        """安全弁2: save/load関連のメソッドを持たない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert not hasattr(monitor, "to_dict")
        assert not hasattr(monitor, "from_dict")
        assert not hasattr(monitor, "save")
        assert not hasattr(monitor, "load")

    def test_no_state_update_method(self) -> None:
        """状態更新系のメソッドを持たない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        assert not hasattr(monitor, "update_state")
        assert not hasattr(monitor, "update_emotion")
        assert not hasattr(monitor, "update_mood")

    def test_no_score_or_quality_field(self) -> None:
        """安全弁3: 品質スコア・一致度フィールドが存在しない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, utterance="テスト")
        assert record is not None
        d = record.to_dict()
        for key in d:
            assert "score" not in key.lower()
            assert "quality" not in key.lower()
            assert "match" not in key.lower()
            assert "deviation" not in key.lower()
            assert "accuracy" not in key.lower()


# ── 複数記録の混合テスト ──────────────────────────────────────────────


class TestMixedRecords:
    """複数の異なるタイプの記録を混合したテスト。"""

    def test_mixed_pathways(self) -> None:
        """異なる入力経路の記録を混合。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1, pathway="vision")
        _record_one(monitor, tick=2, pathway="text")
        _record_one(monitor, tick=3, pathway="internal")
        recent = monitor.get_recent_records(count=3)
        pathways = [r["input_pathway"] for r in recent]
        assert "internal" in pathways
        assert "text" in pathways
        assert "vision" in pathways

    def test_mixed_fallback_and_normal(self) -> None:
        """フォールバックと通常記録の混合。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1, fallback=False)
        _record_one(monitor, tick=2, fallback=True)
        _record_one(monitor, tick=3, fallback=False)
        assert monitor.record_count == 3
        assert monitor.fallback_count == 1

    def test_various_utterance_lengths(self) -> None:
        """異なる長さの発話テキスト。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor, tick=1, utterance="あ")
        _record_one(monitor, tick=2, utterance="嬉しいな。楽しいわ。最高！")
        recent = monitor.get_recent_records(count=2)
        # 新しい順
        assert recent[0]["utterance_char_count"] > recent[1]["utterance_char_count"]


# ── エッジケースのテスト ──────────────────────────────────────────────


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_large_enrichment_char_count(self) -> None:
        """大きなenrichment文字数が正しく記録される。"""
        monitor = ExpressionQualityVerification(enabled=True)
        record = _record_one(monitor, enrichment_chars=100000)
        assert record is not None
        assert record.enrichment_char_count == 100000

    def test_unicode_utterance(self) -> None:
        """Unicode文字を含む発話テキスト。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "ふふっ♪ 楽しいわね〜🎵"
        record = _record_one(monitor, utterance=text)
        assert record is not None
        assert record.utterance_text == text

    def test_very_long_utterance(self) -> None:
        """非常に長い発話テキスト。"""
        monitor = ExpressionQualityVerification(enabled=True)
        text = "あ" * 10000
        record = _record_one(monitor, utterance=text)
        assert record is not None
        assert record.utterance_char_count == 10000

    def test_meta_with_unexpected_types(self) -> None:
        """メタ情報に予期しない型の値が含まれる場合。"""
        monitor = ExpressionQualityVerification(enabled=True)
        meta = {"emotion": 123, "intensity": "not_a_number", "action": None}
        record = _record_one(monitor, meta=meta)
        assert record is not None
        # 文字列変換で安全に処理される
        assert record.utterance_meta_emotion == "123"
        # 不正な型はデフォルト値
        assert record.utterance_meta_intensity == 0.0
        assert record.utterance_meta_action == "None"

    def test_rapid_successive_records(self) -> None:
        """連続した高速記録。"""
        monitor = ExpressionQualityVerification(enabled=True)
        for i in range(100):
            result = _record_one(monitor, tick=i)
            assert result is not None
        assert monitor.record_count == 100

    def test_buffer_at_exact_limit(self) -> None:
        """バッファがちょうど上限に達した場合。"""
        config = ExpressionQualityConfig(max_buffer_size=5)
        monitor = ExpressionQualityVerification(config=config, enabled=True)
        for i in range(5):
            _record_one(monitor, tick=i)
        assert monitor.buffer_size == 5
        # もう1件追加すると最古が消える
        _record_one(monitor, tick=5)
        assert monitor.buffer_size == 5
        recent = monitor.get_recent_records(count=5)
        assert recent[-1]["tick_number"] == 1  # tick=0が消失

    def test_get_recent_with_count_zero(self) -> None:
        """count=0の場合でもエラーにならない（最小1件）。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor)
        # count=0はmax(1, 0)=1に正規化される
        recent = monitor.get_recent_records(count=0)
        assert len(recent) == 1

    def test_get_recent_with_negative_count(self) -> None:
        """負のcountの場合でもエラーにならない。"""
        monitor = ExpressionQualityVerification(enabled=True)
        _record_one(monitor)
        recent = monitor.get_recent_records(count=-5)
        assert len(recent) == 1


# ── ログ出力のテスト ──────────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のテスト。"""

    def test_record_emits_log(self, caplog) -> None:
        """記録時にログが出力される。"""
        import logging
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.expression_quality"):
            monitor = ExpressionQualityVerification(enabled=True)
            _record_one(monitor)
        # ログが出力されたことを確認
        assert len(caplog.records) >= 1

    def test_session_summary_emits_log(self, caplog) -> None:
        """セッションサマリー時にログが出力される。"""
        import logging
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.expression_quality"):
            monitor = ExpressionQualityVerification(enabled=True)
            _record_one(monitor)
            monitor.emit_session_summary()
        # 記録 + サマリーの2件
        assert len(caplog.records) >= 2

    def test_log_contains_json(self, caplog) -> None:
        """ログ出力がJSON形式である。"""
        import json
        import logging
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.expression_quality"):
            monitor = ExpressionQualityVerification(enabled=True)
            _record_one(monitor)
        for log_record in caplog.records:
            # JSONとしてパースできることを確認
            parsed = json.loads(log_record.message)
            assert isinstance(parsed, dict)
