"""
tests/test_enrichment_consumption_viz.py - EnrichmentConsumptionMeasurement のテスト

設計書: design_enrichment_consumption_viz.md

テスト項目:
- 初期化テスト(デフォルト設定、カスタム設定、環境変数制御)
- 記録テスト(基本記録、複数記録、占有率算出)
- FIFOバッファテスト(上限、自然消失)
- セッション累積統計テスト(計測回数、占有率最大/最小)
- セクション別文字数テスト
- 空項目スキップ効果記録テスト
- 圧縮前後の文字数記録テスト
- 読み取り専用アクセサテスト
- セッションサマリテスト
- 安全弁テスト(計測失敗の無視、無効時の完全スキップ、FIFO上限、永続化非対象)
- psyche非参照テスト
- enrichment帰還経路遮断テスト
- ログ出力テスト
- 文字数近似の制約テスト
- 統合テスト
"""

import json
import logging
import os
import time
from typing import Any, Optional
from unittest.mock import patch

import pytest

from tools.pipeline_measurement import (
    EnrichmentConsumptionMeasurement,
    EnrichmentConsumptionRecord,
    _DEFAULT_CONSUMPTION_BUFFER_MAX,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_measurement(
    enabled: bool = True,
    buffer_max: int = 200,
) -> EnrichmentConsumptionMeasurement:
    """テスト用のEnrichmentConsumptionMeasurementを生成する。"""
    return EnrichmentConsumptionMeasurement(enabled=enabled, buffer_max=buffer_max)


def _sample_enrichment_text() -> str:
    """テスト用のenrichmentテキスト(圧縮後)。"""
    return (
        "[内面]\n"
        "感情: joy=0.7, sadness=0.1\n"
        "ムード: valence=0.5, arousal=0.3\n"
        "[自己]\n"
        "自己モデル: (安定)\n"
        "[動機]\n"
        "内発動機: curiosity=0.6\n"
        "[記憶]\n"
        "記憶統合: recent_episode_count=3\n"
        "[判断]\n"
        "判断傾向: (安定)"
    )


def _sample_enrichment_text_before_compression() -> str:
    """テスト用のenrichmentテキスト(圧縮前)。"""
    return (
        "【心理状態（内面）】\n"
        "感情: joy=0.7, sadness=0.1, anger=0.0, fear=0.0, surprise=0.0, disgust=0.0, trust=0.3\n"
        "ムード: valence=0.5, arousal=0.3\n"
        "【自己認識】\n"
        "自己モデル: 現在の統合ビューは安定している。変動なし。\n"
        "【動機・目標】\n"
        "内発動機: curiosity=0.6, exploration=0.4\n"
        "【記憶・内省】\n"
        "記憶統合: recent_episode_count=3, binding_count=5\n"
        "【判断傾向】\n"
        "判断傾向: 安定的なパターンが継続している。特記事項なし。"
    )


def _sample_prompt_total() -> str:
    """テスト用のプロンプト全体テキスト。"""
    enrichment = _sample_enrichment_text()
    return (
        "以下の情報に基づいてセリフをJSON形式で出力してください。\n"
        f"\n{enrichment}\n"
        "\nキャラクター名: キュレネ\n"
        "選択された方針: 共感する\n"
        "トーン: romantic, sweet\n"
    )


def _sample_section_texts() -> dict[str, str]:
    """テスト用のセクション別テキスト。"""
    return {
        "内面": "感情: joy=0.7, sadness=0.1\nムード: valence=0.5, arousal=0.3",
        "自己": "自己モデル: (安定)",
        "動機": "内発動機: curiosity=0.6",
        "記憶": "記憶統合: recent_episode_count=3",
        "判断": "判断傾向: (安定)",
    }


def _record_sample(m: EnrichmentConsumptionMeasurement) -> None:
    """サンプルデータで1回記録する。"""
    m.record_consumption(
        enrichment_text=_sample_enrichment_text(),
        enrichment_text_before_compression=_sample_enrichment_text_before_compression(),
        prompt_total_text=_sample_prompt_total(),
        non_empty_item_count=35,
        total_item_count=49,
        section_texts=_sample_section_texts(),
    )


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """EnrichmentConsumptionMeasurementの初期化テスト。"""

    def test_default_init_disabled(self):
        """デフォルトでは計測は無効(CYRENE_MONITOR未設定)。"""
        with patch.dict(os.environ, {}, clear=False):
            if "CYRENE_MONITOR" in os.environ:
                del os.environ["CYRENE_MONITOR"]
            m = EnrichmentConsumptionMeasurement()
            assert m.enabled is False

    def test_enabled_via_constructor(self):
        """コンストラクタで明示的に有効化。"""
        m = _make_measurement(enabled=True)
        assert m.enabled is True

    def test_disabled_via_constructor(self):
        """コンストラクタで明示的に無効化。"""
        m = _make_measurement(enabled=False)
        assert m.enabled is False

    def test_enabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=1で有効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = EnrichmentConsumptionMeasurement()
            assert m.enabled is True

    def test_disabled_via_env_var(self):
        """環境変数CYRENE_MONITOR=0で無効化。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            m = EnrichmentConsumptionMeasurement()
            assert m.enabled is False

    def test_constructor_overrides_env(self):
        """コンストラクタの指定が環境変数より優先される。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            m = EnrichmentConsumptionMeasurement(enabled=False)
            assert m.enabled is False

    def test_initial_state_empty(self):
        """初期状態で全統計が空/ゼロ。"""
        m = _make_measurement()
        assert m.buffer_size == 0
        assert m.total_measurement_count == 0
        assert m.occupancy_ratio_max is None
        assert m.occupancy_ratio_min is None

    def test_default_buffer_max(self):
        """デフォルトのFIFOバッファ上限が適用される。"""
        assert _DEFAULT_CONSUMPTION_BUFFER_MAX == 200

    def test_buffer_max_applied(self):
        """FIFOバッファの上限が適用される。"""
        m = _make_measurement(buffer_max=3)
        for _ in range(5):
            _record_sample(m)
        assert m.buffer_size == 3

    def test_buffer_max_minimum(self):
        """FIFOバッファの上限は最低1。"""
        m = _make_measurement(buffer_max=0)
        _record_sample(m)
        assert m.buffer_size == 1


# ── EnrichmentConsumptionRecord テスト ──────────────────────────────


class TestEnrichmentConsumptionRecord:
    """EnrichmentConsumptionRecordの単体テスト。"""

    def test_init(self):
        """初期状態。"""
        r = EnrichmentConsumptionRecord(
            enrichment_chars=500,
            enrichment_chars_before_compression=800,
            prompt_total_chars=2000,
            occupancy_ratio=0.25,
            non_empty_item_count=35,
            total_item_count=49,
            section_chars={"内面": 100, "自己": 50},
        )
        assert r.enrichment_chars == 500
        assert r.enrichment_chars_before_compression == 800
        assert r.prompt_total_chars == 2000
        assert r.occupancy_ratio == 0.25
        assert r.non_empty_item_count == 35
        assert r.total_item_count == 49
        assert r.section_chars == {"内面": 100, "自己": 50}
        assert r.timestamp > 0

    def test_to_dict(self):
        """辞書表現の構造。"""
        r = EnrichmentConsumptionRecord(
            enrichment_chars=500,
            enrichment_chars_before_compression=800,
            prompt_total_chars=2000,
            occupancy_ratio=0.25,
            non_empty_item_count=35,
            total_item_count=49,
            section_chars={"内面": 100},
        )
        d = r.to_dict()
        assert d["enrichment_chars"] == 500
        assert d["enrichment_chars_before_compression"] == 800
        assert d["prompt_total_chars"] == 2000
        assert d["occupancy_ratio"] == 0.25
        assert d["non_empty_item_count"] == 35
        assert d["total_item_count"] == 49
        assert d["section_chars"] == {"内面": 100}
        assert "timestamp" in d

    def test_to_dict_occupancy_ratio_rounded(self):
        """占有率は小数点以下4桁に丸められる。"""
        r = EnrichmentConsumptionRecord(
            enrichment_chars=500,
            enrichment_chars_before_compression=800,
            prompt_total_chars=2000,
            occupancy_ratio=0.123456789,
            non_empty_item_count=35,
            total_item_count=49,
            section_chars={},
        )
        d = r.to_dict()
        assert d["occupancy_ratio"] == 0.1235

    def test_section_chars_is_copy(self):
        """section_charsは入力辞書のコピーを保持する。"""
        original = {"内面": 100, "自己": 50}
        r = EnrichmentConsumptionRecord(
            enrichment_chars=500,
            enrichment_chars_before_compression=800,
            prompt_total_chars=2000,
            occupancy_ratio=0.25,
            non_empty_item_count=35,
            total_item_count=49,
            section_chars=original,
        )
        original["内面"] = 999
        assert r.section_chars["内面"] == 100


# ── 記録テスト ──────────────────────────────────────────────────────


class TestRecording:
    """record_consumptionの基本テスト。"""

    def test_basic_recording(self):
        """基本的な記録が正しく行われる。"""
        m = _make_measurement()
        _record_sample(m)
        assert m.buffer_size == 1
        assert m.total_measurement_count == 1

    def test_enrichment_chars_recorded(self):
        """enrichment文字数が記録される。"""
        m = _make_measurement()
        enrichment = "テスト" * 50  # 150文字
        m.record_consumption(
            enrichment_text=enrichment,
            enrichment_text_before_compression=enrichment + "拡張" * 50,
            prompt_total_text=enrichment + "プロンプト" * 100,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": "テスト"},
        )
        summary = m.get_summary()
        record = summary["latest_record"]
        assert record["enrichment_chars"] == len(enrichment)

    def test_prompt_total_chars_recorded(self):
        """プロンプト全体文字数が記録される。"""
        m = _make_measurement()
        prompt = "プロンプト全体" * 200
        m.record_consumption(
            enrichment_text="enrichment",
            enrichment_text_before_compression="enrichment_before",
            prompt_total_text=prompt,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["prompt_total_chars"] == len(prompt)

    def test_occupancy_ratio_calculation(self):
        """占有率が正しく算出される。"""
        m = _make_measurement()
        enrichment = "a" * 250
        prompt = "a" * 1000
        m.record_consumption(
            enrichment_text=enrichment,
            enrichment_text_before_compression=enrichment,
            prompt_total_text=prompt,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["occupancy_ratio"] == pytest.approx(0.25, abs=0.001)

    def test_occupancy_ratio_zero_prompt(self):
        """プロンプト全体が空の場合、占有率は0.0。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="enrichment",
            enrichment_text_before_compression="enrichment",
            prompt_total_text="",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["occupancy_ratio"] == 0.0

    def test_non_empty_item_count_recorded(self):
        """非空項目数が記録される。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="enrichment",
            enrichment_text_before_compression="enrichment",
            prompt_total_text="prompt",
            non_empty_item_count=42,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["non_empty_item_count"] == 42
        assert record["total_item_count"] == 49

    def test_multiple_recordings(self):
        """複数の記録が正しく蓄積される。"""
        m = _make_measurement()
        for i in range(5):
            m.record_consumption(
                enrichment_text="a" * (100 + i * 10),
                enrichment_text_before_compression="a" * (200 + i * 10),
                prompt_total_text="a" * 1000,
                non_empty_item_count=30 + i,
                total_item_count=49,
                section_texts={},
            )
        assert m.buffer_size == 5
        assert m.total_measurement_count == 5

    def test_compression_chars_before_and_after(self):
        """圧縮前後の文字数が記録される。"""
        m = _make_measurement()
        before = "a" * 800
        after = "a" * 500
        m.record_consumption(
            enrichment_text=after,
            enrichment_text_before_compression=before,
            prompt_total_text="a" * 2000,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["enrichment_chars"] == 500
        assert record["enrichment_chars_before_compression"] == 800


# ── FIFOバッファテスト ────────────────────────────────────────────


class TestFIFOBuffer:
    """FIFOバッファの上限と自然消失。"""

    def test_fifo_eviction(self):
        """上限を超えると最古の記録が自然消失する。"""
        m = _make_measurement(buffer_max=3)
        for i in range(5):
            m.record_consumption(
                enrichment_text=f"enrichment_{i}",
                enrichment_text_before_compression=f"enrichment_{i}",
                prompt_total_text=f"prompt_{i}",
                non_empty_item_count=i,
                total_item_count=49,
                section_texts={},
            )
        assert m.buffer_size == 3
        # 最古の記録(i=0,1)は消失し、i=2,3,4が残る
        latest = m.get_summary()["latest_record"]
        assert latest["non_empty_item_count"] == 4

    def test_fifo_does_not_affect_cumulative(self):
        """FIFO消失はセッション累積統計に影響しない。"""
        m = _make_measurement(buffer_max=3)
        for i in range(10):
            m.record_consumption(
                enrichment_text="a" * 100,
                enrichment_text_before_compression="a" * 100,
                prompt_total_text="a" * 1000,
                non_empty_item_count=30,
                total_item_count=49,
                section_texts={},
            )
        assert m.buffer_size == 3
        assert m.total_measurement_count == 10


# ── セッション累積統計テスト ────────────────────────────────────────


class TestSessionCumulative:
    """セッション累積統計の更新。"""

    def test_occupancy_ratio_max_tracking(self):
        """占有率の最大値が正しく追跡される。"""
        m = _make_measurement()
        # 占有率 0.1, 0.3, 0.2 → max=0.3
        for ratio_numerator in [100, 300, 200]:
            m.record_consumption(
                enrichment_text="a" * ratio_numerator,
                enrichment_text_before_compression="a" * ratio_numerator,
                prompt_total_text="a" * 1000,
                non_empty_item_count=30,
                total_item_count=49,
                section_texts={},
            )
        assert m.occupancy_ratio_max == pytest.approx(0.3, abs=0.001)

    def test_occupancy_ratio_min_tracking(self):
        """占有率の最小値が正しく追跡される。"""
        m = _make_measurement()
        # 占有率 0.3, 0.1, 0.2 → min=0.1
        for ratio_numerator in [300, 100, 200]:
            m.record_consumption(
                enrichment_text="a" * ratio_numerator,
                enrichment_text_before_compression="a" * ratio_numerator,
                prompt_total_text="a" * 1000,
                non_empty_item_count=30,
                total_item_count=49,
                section_texts={},
            )
        assert m.occupancy_ratio_min == pytest.approx(0.1, abs=0.001)

    def test_single_measurement_max_min_equal(self):
        """1回の記録では最大値と最小値が同じ。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="a" * 200,
            enrichment_text_before_compression="a" * 200,
            prompt_total_text="a" * 1000,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        assert m.occupancy_ratio_max == m.occupancy_ratio_min
        assert m.occupancy_ratio_max == pytest.approx(0.2, abs=0.001)


# ── セクション別文字数テスト ────────────────────────────────────────


class TestSectionChars:
    """セクション別文字数の記録。"""

    def test_section_chars_recorded(self):
        """セクション別文字数が正しく記録される。"""
        m = _make_measurement()
        sections = {
            "内面": "a" * 100,
            "自己": "b" * 50,
            "動機": "c" * 75,
            "記憶": "d" * 120,
            "判断": "e" * 30,
        }
        m.record_consumption(
            enrichment_text="enrichment",
            enrichment_text_before_compression="enrichment",
            prompt_total_text="prompt",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts=sections,
        )
        record = m.get_summary()["latest_record"]
        assert record["section_chars"]["内面"] == 100
        assert record["section_chars"]["自己"] == 50
        assert record["section_chars"]["動機"] == 75
        assert record["section_chars"]["記憶"] == 120
        assert record["section_chars"]["判断"] == 30

    def test_empty_section_texts(self):
        """セクションテキストが空の場合。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="enrichment",
            enrichment_text_before_compression="enrichment",
            prompt_total_text="prompt",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["section_chars"] == {}

    def test_section_chars_per_tick_variation(self):
        """ティックごとにセクション文字数が変動する。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": "a" * 100},
        )
        m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": "a" * 200},
        )
        # Latest should be 200
        latest_sections = m.get_latest_section_chars()
        assert latest_sections is not None
        assert latest_sections["内面"] == 200


# ── 読み取り専用アクセサテスト ────────────────────────────────────


class TestReadOnlyAccessor:
    """読み取り専用アクセサのテスト。"""

    def test_get_latest_occupancy_ratio_empty(self):
        """記録がない場合はNoneを返す。"""
        m = _make_measurement()
        assert m.get_latest_occupancy_ratio() is None

    def test_get_latest_occupancy_ratio_with_data(self):
        """記録がある場合は直近の占有率を返す。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="a" * 200,
            enrichment_text_before_compression="a" * 200,
            prompt_total_text="a" * 1000,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        ratio = m.get_latest_occupancy_ratio()
        assert ratio is not None
        assert ratio == pytest.approx(0.2, abs=0.001)

    def test_get_latest_section_chars_empty(self):
        """記録がない場合はNoneを返す。"""
        m = _make_measurement()
        assert m.get_latest_section_chars() is None

    def test_get_latest_section_chars_with_data(self):
        """記録がある場合は直近のセクション別文字数を返す。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": "a" * 50, "記憶": "b" * 30},
        )
        chars = m.get_latest_section_chars()
        assert chars is not None
        assert chars["内面"] == 50
        assert chars["記憶"] == 30

    def test_get_latest_section_chars_returns_copy(self):
        """セクション別文字数は読み取り専用コピーを返す。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": "a" * 50},
        )
        c1 = m.get_latest_section_chars()
        c2 = m.get_latest_section_chars()
        assert c1 is not c2
        assert c1 == c2

    def test_get_latest_non_empty_item_count_empty(self):
        """記録がない場合はNoneを返す。"""
        m = _make_measurement()
        assert m.get_latest_non_empty_item_count() is None

    def test_get_latest_non_empty_item_count_with_data(self):
        """記録がある場合は直近の非空項目数を返す。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=42,
            total_item_count=49,
            section_texts={},
        )
        count = m.get_latest_non_empty_item_count()
        assert count == 42

    def test_get_summary_empty(self):
        """空状態でget_summaryが正しい構造を返す。"""
        m = _make_measurement()
        s = m.get_summary()
        assert s["total_measurement_count"] == 0
        assert s["occupancy_ratio_max"] is None
        assert s["occupancy_ratio_min"] is None
        assert s["buffer_size"] == 0
        assert s["latest_record"] is None

    def test_get_summary_with_data(self):
        """データがある状態でget_summaryが正しい値を返す。"""
        m = _make_measurement()
        _record_sample(m)
        s = m.get_summary()
        assert s["total_measurement_count"] == 1
        assert s["occupancy_ratio_max"] is not None
        assert s["occupancy_ratio_min"] is not None
        assert s["buffer_size"] == 1
        assert s["latest_record"] is not None

    def test_get_summary_returns_copy(self):
        """get_summaryは読み取り専用コピーを返す。"""
        m = _make_measurement()
        _record_sample(m)
        s1 = m.get_summary()
        s2 = m.get_summary()
        assert s1 is not s2


# ── セッションサマリテスト ────────────────────────────────────────


class TestSessionSummary:
    """セッションサマリの出力。"""

    def test_emit_consumption_summary(self, caplog):
        """セッションサマリがログに出力される。"""
        m = _make_measurement()
        _record_sample(m)

        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_consumption_summary()

        found = False
        for record in caplog.records:
            if "enrichment_consumption_session_summary" in record.getMessage():
                found = True
                data = json.loads(record.getMessage())
                assert data["total_measurement_count"] == 1
                assert data["occupancy_ratio_max"] is not None
                assert data["occupancy_ratio_min"] is not None
                break
        assert found, "enrichment_consumption_session_summary log not found"

    def test_emit_summary_when_disabled(self, caplog):
        """無効時にはサマリは出力されない。"""
        m = _make_measurement(enabled=False)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_consumption_summary()
        for record in caplog.records:
            assert "enrichment_consumption_session_summary" not in record.getMessage()


# ── 安全弁テスト ─────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁の動作確認。"""

    def test_disabled_record_noop(self):
        """無効時にrecord_consumptionは何もしない(安全弁5)。"""
        m = _make_measurement(enabled=False)
        _record_sample(m)
        assert m.buffer_size == 0
        assert m.total_measurement_count == 0

    def test_fifo_buffer_limit(self):
        """FIFOバッファの上限が正しく適用される(安全弁4)。"""
        m = _make_measurement(buffer_max=5)
        for _ in range(20):
            _record_sample(m)
        assert m.buffer_size == 5

    def test_no_enrichment_output_methods(self):
        """enrichment出力を生成する関数を持たない(安全弁1)。"""
        m = _make_measurement()
        assert not hasattr(m, "get_enrichment")
        assert not hasattr(m, "enrichment")
        assert not hasattr(m, "get_prompt_enrichment")

    def test_no_save_load_methods(self):
        """save/loadメソッドが存在しない(安全弁2)。"""
        m = _make_measurement()
        assert not hasattr(m, "save")
        assert not hasattr(m, "load")
        assert not hasattr(m, "to_dict")
        assert not hasattr(m, "from_dict")

    def test_session_boundary_clears_state(self):
        """新しいインスタンスは空の状態で開始する(安全弁2)。"""
        m1 = _make_measurement()
        _record_sample(m1)
        assert m1.buffer_size == 1

        m2 = _make_measurement()
        assert m2.buffer_size == 0
        assert m2.total_measurement_count == 0

    def test_no_value_judgment_in_summary(self):
        """サマリに「望ましい水準」「過大判定」を含まない(安全弁3)。"""
        m = _make_measurement()
        _record_sample(m)
        summary = m.get_summary()
        summary_str = json.dumps(summary, ensure_ascii=False)
        assert "望ましい" not in summary_str
        assert "過大" not in summary_str
        assert "過小" not in summary_str
        assert "適切" not in summary_str
        assert "不適切" not in summary_str

    def test_measurement_failure_ignored(self):
        """record_consumption内部で問題が起きても安全に無視される(安全弁1)。"""
        m = _make_measurement()
        # section_textsがNoneでも例外にならない
        # (通常はdict[str, str]だが、安全弁により例外が捕捉される)
        try:
            m.record_consumption(
                enrichment_text="e",
                enrichment_text_before_compression="e",
                prompt_total_text="p",
                non_empty_item_count=30,
                total_item_count=49,
                section_texts=None,  # type: ignore
            )
        except Exception:
            pytest.fail("record_consumption should not raise on bad input")


# ── psyche非参照テスト ───────────────────────────────────────────


class TestPsycheIsolation:
    """psycheモジュールとの構造的分離の確認。"""

    def test_no_psyche_imports(self):
        """pipeline_measurement.pyがpsycheモジュールをインポートしない。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        assert "from psyche" not in source
        assert "import psyche" not in source

    def test_no_orchestrator_dependency(self):
        """orchestratorへの依存がない(importレベル)。"""
        import tools.pipeline_measurement as pm
        import inspect
        source = inspect.getsource(pm)
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "orchestrator" not in line.lower()


# ── enrichment帰還経路遮断テスト ──────────────────────────────────


class TestNoFeedbackLoop:
    """計測結果がenrichmentに帰還しないことの確認。"""

    def test_no_enrichment_modification_methods(self):
        """enrichmentの圧縮率・対象項目を変更するメソッドが存在しない。"""
        m = _make_measurement()
        assert not hasattr(m, "set_compression_ratio")
        assert not hasattr(m, "adjust_enrichment")
        assert not hasattr(m, "modify_enrichment")

    def test_no_gemini_context_methods(self):
        """Geminiのコンテキスト設定を変更するメソッドが存在しない。"""
        m = _make_measurement()
        assert not hasattr(m, "set_temperature")
        assert not hasattr(m, "set_max_tokens")
        assert not hasattr(m, "adjust_context")

    def test_record_does_not_return_action(self):
        """record_consumptionは戻り値なし(帰還経路なし)。"""
        m = _make_measurement()
        result = m.record_consumption(
            enrichment_text="e",
            enrichment_text_before_compression="e",
            prompt_total_text="p",
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        assert result is None


# ── ログ出力テスト ───────────────────────────────────────────────


class TestLogOutput:
    """ログ出力のフォーマット検証。"""

    def test_consumption_log_format(self, caplog):
        """enrichment_consumptionログのJSON構造。"""
        m = _make_measurement()
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.record_consumption(
                enrichment_text="a" * 200,
                enrichment_text_before_compression="a" * 400,
                prompt_total_text="a" * 1000,
                non_empty_item_count=35,
                total_item_count=49,
                section_texts={"内面": "a" * 100},
            )

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_consumption" in msg and "session_summary" not in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "enrichment_consumption"
                assert "timestamp" in data
                assert data["enrichment_chars"] == 200
                assert data["enrichment_chars_before_compression"] == 400
                assert data["prompt_total_chars"] == 1000
                assert data["occupancy_ratio"] == 0.2
                assert data["non_empty_item_count"] == 35
                assert data["total_item_count"] == 49
                assert data["section_chars"]["内面"] == 100
                break
        assert found, "enrichment_consumption log not found"

    def test_session_summary_log_format(self, caplog):
        """enrichment_consumption_session_summaryログのJSON構造。"""
        m = _make_measurement()
        _record_sample(m)
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor"):
            m.emit_consumption_summary()
        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "enrichment_consumption_session_summary" in msg:
                found = True
                data = json.loads(msg)
                assert data["type"] == "enrichment_consumption_session_summary"
                assert "timestamp" in data
                assert "total_measurement_count" in data
                assert "occupancy_ratio_max" in data
                assert "occupancy_ratio_min" in data
                assert "buffer_size" in data
                break
        assert found, "enrichment_consumption_session_summary log not found"


# ── 文字数近似の制約テスト ──────────────────────────────────────


class TestCharCountApproximation:
    """文字数ベース計測の制約の確認。"""

    def test_japanese_text_char_count(self):
        """日本語テキストの文字数が正しく記録される。"""
        m = _make_measurement()
        japanese_text = "感情: joy=0.7, 悲しみ=0.1, 怒り=0.0"
        m.record_consumption(
            enrichment_text=japanese_text,
            enrichment_text_before_compression=japanese_text,
            prompt_total_text=japanese_text + "追加テキスト" * 10,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={"内面": japanese_text},
        )
        record = m.get_summary()["latest_record"]
        assert record["enrichment_chars"] == len(japanese_text)

    def test_mixed_text_char_count(self):
        """日英混在テキストの文字数が正しく記録される。"""
        m = _make_measurement()
        mixed = "joy=0.7, 悲しみ=0.1, curiosity高"
        m.record_consumption(
            enrichment_text=mixed,
            enrichment_text_before_compression=mixed,
            prompt_total_text=mixed * 5,
            non_empty_item_count=30,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["enrichment_chars"] == len(mixed)

    def test_empty_enrichment_text(self):
        """空のenrichmentテキスト。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="",
            enrichment_text_before_compression="",
            prompt_total_text="prompt" * 100,
            non_empty_item_count=0,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["enrichment_chars"] == 0
        assert record["occupancy_ratio"] == 0.0


# ── 統合テスト ───────────────────────────────────────────────────


class TestIntegration:
    """enrichment消費量計測の統合テスト。"""

    def test_full_session_lifecycle(self):
        """セッションのライフサイクル: 複数計測→サマリ出力。"""
        m = _make_measurement()

        # ティック1: 低占有率
        m.record_consumption(
            enrichment_text="a" * 100,
            enrichment_text_before_compression="a" * 200,
            prompt_total_text="a" * 1000,
            non_empty_item_count=20,
            total_item_count=49,
            section_texts={"内面": "a" * 40, "記憶": "a" * 30},
        )

        # ティック2: 高占有率
        m.record_consumption(
            enrichment_text="b" * 500,
            enrichment_text_before_compression="b" * 800,
            prompt_total_text="b" * 1000,
            non_empty_item_count=45,
            total_item_count=49,
            section_texts={"内面": "b" * 150, "記憶": "b" * 200},
        )

        # ティック3: 中間占有率
        m.record_consumption(
            enrichment_text="c" * 300,
            enrichment_text_before_compression="c" * 500,
            prompt_total_text="c" * 1000,
            non_empty_item_count=35,
            total_item_count=49,
            section_texts={"内面": "c" * 100, "記憶": "c" * 100},
        )

        assert m.total_measurement_count == 3
        assert m.buffer_size == 3
        assert m.occupancy_ratio_max == pytest.approx(0.5, abs=0.001)
        assert m.occupancy_ratio_min == pytest.approx(0.1, abs=0.001)

        # 最新の記録
        latest = m.get_summary()["latest_record"]
        assert latest["enrichment_chars"] == 300
        assert latest["non_empty_item_count"] == 35

        # アクセサ
        assert m.get_latest_occupancy_ratio() == pytest.approx(0.3, abs=0.001)
        assert m.get_latest_non_empty_item_count() == 35

    def test_session_boundary_isolation(self):
        """セッション境界で全内部状態が消失する。"""
        m1 = _make_measurement()
        for _ in range(10):
            _record_sample(m1)
        assert m1.total_measurement_count == 10

        m2 = _make_measurement()
        assert m2.total_measurement_count == 0
        assert m2.buffer_size == 0
        assert m2.occupancy_ratio_max is None
        assert m2.occupancy_ratio_min is None

    def test_high_occupancy_no_action(self):
        """高い占有率でも自動調整は行われない(事実記録のみ)。"""
        m = _make_measurement()
        # 占有率90%
        m.record_consumption(
            enrichment_text="a" * 900,
            enrichment_text_before_compression="a" * 1500,
            prompt_total_text="a" * 1000,
            non_empty_item_count=49,
            total_item_count=49,
            section_texts={},
        )
        # 占有率90%が記録されるだけ。圧縮率の変更等は行われない
        assert m.get_latest_occupancy_ratio() == pytest.approx(0.9, abs=0.001)
        assert m.total_measurement_count == 1

    def test_low_occupancy_no_action(self):
        """低い占有率でも自動調整は行われない(事実記録のみ)。"""
        m = _make_measurement()
        # 占有率1%
        m.record_consumption(
            enrichment_text="a" * 10,
            enrichment_text_before_compression="a" * 20,
            prompt_total_text="a" * 1000,
            non_empty_item_count=5,
            total_item_count=49,
            section_texts={},
        )
        assert m.get_latest_occupancy_ratio() == pytest.approx(0.01, abs=0.001)

    def test_coexists_with_pipeline_measurement(self):
        """PipelineMeasurementと共存可能。"""
        from tools.pipeline_measurement import PipelineMeasurement
        pm = PipelineMeasurement(enabled=True)
        ecm = _make_measurement()

        pm.begin_pipeline("vision")
        pm.record_phase("perception_api", 0.1)
        pm.end_pipeline()

        _record_sample(ecm)

        assert pm.record_count == 1
        assert ecm.buffer_size == 1

    def test_coexists_with_coverage_measurement(self):
        """PerceptionCoverageMeasurementと共存可能。"""
        from tools.pipeline_measurement import PerceptionCoverageMeasurement
        pcm = PerceptionCoverageMeasurement(enabled=True)
        ecm = _make_measurement()

        pcm.record_perception("joy", "greeting", ["hello"], False)
        _record_sample(ecm)

        assert pcm.total_count == 1
        assert ecm.buffer_size == 1

    def test_coexists_with_diversity_measurement(self):
        """PerceptionDiversityMeasurementと共存可能。"""
        from tools.pipeline_measurement import PerceptionDiversityMeasurement
        pdm = PerceptionDiversityMeasurement(enabled=True)
        ecm = _make_measurement()

        pdm.record_perception_diversity("joy", "greeting", 1, 10, 0.5, True)
        _record_sample(ecm)

        assert pdm.total_count == 1
        assert ecm.buffer_size == 1

    def test_realistic_enrichment_data(self):
        """実際のenrichmentデータに近い計測。"""
        m = _make_measurement()

        # 49項目のうち35項目が非空
        enrichment_text = _sample_enrichment_text()
        enrichment_before = _sample_enrichment_text_before_compression()
        prompt_total = _sample_prompt_total()
        section_texts = _sample_section_texts()

        m.record_consumption(
            enrichment_text=enrichment_text,
            enrichment_text_before_compression=enrichment_before,
            prompt_total_text=prompt_total,
            non_empty_item_count=35,
            total_item_count=49,
            section_texts=section_texts,
        )

        record = m.get_summary()["latest_record"]

        # enrichment文字数がプロンプト全体文字数以下であること
        assert record["enrichment_chars"] <= record["prompt_total_chars"]

        # 占有率が0-1の範囲内であること
        assert 0.0 <= record["occupancy_ratio"] <= 1.0

        # 非空項目数が合計項目数以下であること
        assert record["non_empty_item_count"] <= record["total_item_count"]

        # 圧縮後の文字数が圧縮前以下であること
        assert record["enrichment_chars"] <= record["enrichment_chars_before_compression"]

        # 5セクションの文字数が記録されていること
        assert len(record["section_chars"]) == 5


# ── エッジケーステスト ───────────────────────────────────────────


class TestEdgeCases:
    """エッジケースの処理。"""

    def test_very_large_enrichment(self):
        """非常に大きなenrichmentテキストでも記録可能。"""
        m = _make_measurement()
        large = "あ" * 100000
        m.record_consumption(
            enrichment_text=large,
            enrichment_text_before_compression=large,
            prompt_total_text=large + "追加" * 10000,
            non_empty_item_count=49,
            total_item_count=49,
            section_texts={"内面": large},
        )
        assert m.buffer_size == 1
        record = m.get_summary()["latest_record"]
        assert record["enrichment_chars"] == 100000

    def test_enrichment_larger_than_prompt(self):
        """enrichmentがプロンプト全体より大きい場合(理論上あり得ないが安全弁)。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="a" * 2000,
            enrichment_text_before_compression="a" * 2000,
            prompt_total_text="a" * 1000,
            non_empty_item_count=49,
            total_item_count=49,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        # 占有率が1.0を超えるが、エラーにならない
        assert record["occupancy_ratio"] == pytest.approx(2.0, abs=0.001)

    def test_rapid_consecutive_recordings(self):
        """高速で連続的に記録しても問題なし。"""
        m = _make_measurement(buffer_max=50)
        for i in range(100):
            m.record_consumption(
                enrichment_text=f"e_{i}",
                enrichment_text_before_compression=f"e_{i}_before",
                prompt_total_text=f"p_{i}" * 100,
                non_empty_item_count=30,
                total_item_count=49,
                section_texts={},
            )
        assert m.buffer_size == 50
        assert m.total_measurement_count == 100

    def test_zero_total_item_count(self):
        """項目数が0の場合。"""
        m = _make_measurement()
        m.record_consumption(
            enrichment_text="",
            enrichment_text_before_compression="",
            prompt_total_text="prompt",
            non_empty_item_count=0,
            total_item_count=0,
            section_texts={},
        )
        record = m.get_summary()["latest_record"]
        assert record["non_empty_item_count"] == 0
        assert record["total_item_count"] == 0
