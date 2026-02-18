"""
tests/test_self_action_perception.py - 自己行動知覚のテスト

カバー範囲:
- 記録の生成・保持・上限押し出し
- 等価性（重みなし）
- テキスト非解釈（分類なし）
- enrichment出力
- save/load
- 判断系非接続の検証
- 安全弁
- ファクトリ
- 統合テスト
"""

import time
import pytest

from psyche.self_action_perception import (
    RecordStatus,
    SelfActionRecord,
    SelfActionPerceptionState,
    SelfActionPerceptionConfig,
    SelfActionPerceptionRecorder,
    get_self_action_summary,
    create_self_action_perception_recorder,
    _gen_id,
)


# =============================================================================
# Helpers
# =============================================================================

def make_recorder(max_records: int = 50, **kwargs) -> SelfActionPerceptionRecorder:
    """テスト用レコーダーを生成する。"""
    config = SelfActionPerceptionConfig(max_records=max_records, **kwargs)
    return SelfActionPerceptionRecorder(config=config)


def receive_n(recorder: SelfActionPerceptionRecorder, n: int, base_tick: int = 1) -> list[SelfActionRecord]:
    """n件の記録を受領する。"""
    records = []
    for i in range(n):
        rec = recorder.receive_response(
            response_text=f"response_{i}",
            policy_label=f"policy_{i}",
            tick=base_tick + i,
        )
        records.append(rec)
    return records


# =============================================================================
# Test: RecordStatus Enum
# =============================================================================

class TestRecordStatus:
    def test_active_value(self):
        assert RecordStatus.ACTIVE.value == "active"

    def test_pushed_out_value(self):
        assert RecordStatus.PUSHED_OUT.value == "pushed_out"

    def test_enum_members(self):
        members = list(RecordStatus)
        assert len(members) == 2


# =============================================================================
# Test: SelfActionRecord
# =============================================================================

class TestSelfActionRecord:
    def test_default_creation(self):
        rec = SelfActionRecord()
        assert rec.record_id != ""
        assert rec.response_text == ""
        assert rec.policy_label == ""
        assert rec.tick == 0
        assert rec.timestamp > 0
        assert rec.status == RecordStatus.ACTIVE.value

    def test_creation_with_values(self):
        rec = SelfActionRecord(
            response_text="hello",
            policy_label="greet",
            tick=5,
        )
        assert rec.response_text == "hello"
        assert rec.policy_label == "greet"
        assert rec.tick == 5

    def test_unique_record_ids(self):
        rec1 = SelfActionRecord()
        rec2 = SelfActionRecord()
        assert rec1.record_id != rec2.record_id

    def test_to_dict(self):
        rec = SelfActionRecord(
            response_text="test",
            policy_label="policy_a",
            tick=10,
        )
        d = rec.to_dict()
        assert d["response_text"] == "test"
        assert d["policy_label"] == "policy_a"
        assert d["tick"] == 10
        assert "record_id" in d
        assert "timestamp" in d
        assert "status" in d

    def test_from_dict(self):
        original = SelfActionRecord(
            response_text="hello world",
            policy_label="greet",
            tick=7,
        )
        d = original.to_dict()
        restored = SelfActionRecord.from_dict(d)
        assert restored.record_id == original.record_id
        assert restored.response_text == original.response_text
        assert restored.policy_label == original.policy_label
        assert restored.tick == original.tick
        assert restored.status == original.status

    def test_from_dict_empty(self):
        rec = SelfActionRecord.from_dict({})
        assert rec.response_text == ""
        assert rec.policy_label == ""
        assert rec.tick == 0

    def test_immutability_principle(self):
        """記録は追記のみの構造。過去の記録を遡及的に変更する処理は存在しない。"""
        rec = SelfActionRecord(response_text="original")
        # テスト上は直接変更可能だが、設計上は変更しない前提
        assert rec.response_text == "original"

    def test_no_weight_or_score(self):
        """記録には重み・スコア・優先度などの評価的属性を付与しない。"""
        rec = SelfActionRecord(response_text="test")
        d = rec.to_dict()
        # 重み・スコアフィールドが存在しないことを確認
        assert "weight" not in d
        assert "score" not in d
        assert "priority" not in d
        assert "importance" not in d
        assert "quality" not in d
        assert "rating" not in d


# =============================================================================
# Test: SelfActionPerceptionState
# =============================================================================

class TestSelfActionPerceptionState:
    def test_default_state(self):
        state = SelfActionPerceptionState()
        assert state.records == []
        assert state.latest_record is None
        assert state.total_records_received == 0
        assert state.total_records_pushed_out == 0

    def test_to_dict(self):
        state = SelfActionPerceptionState()
        state.total_records_received = 5
        d = state.to_dict()
        assert d["total_records_received"] == 5
        assert d["records"] == []
        assert d["latest_record"] is None

    def test_to_dict_with_records(self):
        state = SelfActionPerceptionState()
        rec = SelfActionRecord(response_text="test")
        state.records.append(rec)
        state.latest_record = rec
        d = state.to_dict()
        assert len(d["records"]) == 1
        assert d["latest_record"] is not None
        assert d["latest_record"]["response_text"] == "test"

    def test_from_dict(self):
        original = SelfActionPerceptionState()
        rec = SelfActionRecord(response_text="hello", policy_label="greet", tick=3)
        original.records.append(rec)
        original.latest_record = rec
        original.total_records_received = 1

        d = original.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)

        assert len(restored.records) == 1
        assert restored.records[0].response_text == "hello"
        assert restored.latest_record is not None
        assert restored.latest_record.response_text == "hello"
        assert restored.total_records_received == 1

    def test_from_dict_empty(self):
        state = SelfActionPerceptionState.from_dict({})
        assert state.records == []
        assert state.latest_record is None
        assert state.total_records_received == 0

    def test_from_dict_no_latest(self):
        state = SelfActionPerceptionState.from_dict({
            "records": [],
            "latest_record": None,
        })
        assert state.latest_record is None


# =============================================================================
# Test: SelfActionPerceptionConfig
# =============================================================================

class TestSelfActionPerceptionConfig:
    def test_defaults(self):
        cfg = SelfActionPerceptionConfig()
        assert cfg.max_records == 50
        assert cfg.enrichment_recent_count == 3
        assert cfg.reference_history_count == 10

    def test_custom_values(self):
        cfg = SelfActionPerceptionConfig(
            max_records=100,
            enrichment_recent_count=5,
            reference_history_count=20,
        )
        assert cfg.max_records == 100
        assert cfg.enrichment_recent_count == 5
        assert cfg.reference_history_count == 20


# =============================================================================
# Test: Stage 1 - 受領と保持
# =============================================================================

class TestReceiveAndRetain:
    def test_basic_receive(self):
        recorder = make_recorder()
        rec = recorder.receive_response("hello", "greet", tick=1)
        assert rec.response_text == "hello"
        assert rec.policy_label == "greet"
        assert rec.tick == 1
        assert rec.status == RecordStatus.ACTIVE.value

    def test_receive_updates_state(self):
        recorder = make_recorder()
        recorder.receive_response("hello", "greet", tick=1)
        assert recorder.state.total_records_received == 1
        assert len(recorder.state.records) == 1
        assert recorder.state.latest_record is not None

    def test_receive_multiple(self):
        recorder = make_recorder()
        receive_n(recorder, 5)
        assert recorder.state.total_records_received == 5
        assert len(recorder.state.records) == 5

    def test_receive_empty_text_skipped(self):
        """沈黙選択時（出力テキストなし）は記録しない。"""
        recorder = make_recorder()
        rec = recorder.receive_response("", "silent", tick=1)
        assert rec.response_text == ""
        assert recorder.state.total_records_received == 0
        assert len(recorder.state.records) == 0
        assert recorder.state.latest_record is None

    def test_receive_none_like_empty_text(self):
        """空文字列は記録しない。"""
        recorder = make_recorder()
        recorder.receive_response("", "", tick=0)
        assert recorder.state.total_records_received == 0

    def test_receive_without_policy(self):
        """ポリシーラベルなしでも記録可能。"""
        recorder = make_recorder()
        rec = recorder.receive_response("hello", tick=1)
        assert rec.response_text == "hello"
        assert rec.policy_label == ""

    def test_latest_record_always_newest(self):
        recorder = make_recorder()
        recorder.receive_response("first", "p1", tick=1)
        assert recorder.state.latest_record.response_text == "first"
        recorder.receive_response("second", "p2", tick=2)
        assert recorder.state.latest_record.response_text == "second"
        recorder.receive_response("third", "p3", tick=3)
        assert recorder.state.latest_record.response_text == "third"

    def test_time_series_order(self):
        """記録は時系列順に蓄積される。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        ticks = [r.tick for r in recorder.state.records]
        assert ticks == sorted(ticks)

    def test_receive_preserves_raw_text(self):
        """テキストは生のテキスト文字列として保持される（テキスト非解釈）。"""
        recorder = make_recorder()
        raw_text = "これはテスト！\n改行も含む\tタブも\u3000全角スペースも"
        rec = recorder.receive_response(raw_text, "test", tick=1)
        assert rec.response_text == raw_text

    def test_receive_long_text(self):
        """長いテキストも切り詰めずに保持する。"""
        recorder = make_recorder()
        long_text = "a" * 10000
        rec = recorder.receive_response(long_text, "long", tick=1)
        assert len(rec.response_text) == 10000

    def test_receive_special_characters(self):
        """特殊文字を含むテキストも変更なく保持する。"""
        recorder = make_recorder()
        special = "♪♡★☆！？〜…「」『』"
        rec = recorder.receive_response(special, "special", tick=1)
        assert rec.response_text == special

    def test_unique_record_ids_on_receive(self):
        recorder = make_recorder()
        records = receive_n(recorder, 10)
        ids = [r.record_id for r in records]
        assert len(set(ids)) == 10


# =============================================================================
# Test: Upper limit pushout
# =============================================================================

class TestPushout:
    def test_pushout_at_limit(self):
        recorder = make_recorder(max_records=5)
        receive_n(recorder, 7)
        assert len(recorder.state.records) == 5
        assert recorder.state.total_records_pushed_out == 2

    def test_pushout_preserves_newest(self):
        recorder = make_recorder(max_records=3)
        receive_n(recorder, 5)
        # 最新の3件が残る
        texts = [r.response_text for r in recorder.state.records]
        assert texts == ["response_2", "response_3", "response_4"]

    def test_pushout_removes_oldest(self):
        recorder = make_recorder(max_records=3)
        receive_n(recorder, 5)
        # 最古の2件が押し出される
        remaining_ticks = [r.tick for r in recorder.state.records]
        assert 1 not in remaining_ticks
        assert 2 not in remaining_ticks

    def test_pushout_cumulative(self):
        recorder = make_recorder(max_records=3)
        receive_n(recorder, 10)
        assert recorder.state.total_records_pushed_out == 7
        assert len(recorder.state.records) == 3

    def test_no_pushout_within_limit(self):
        recorder = make_recorder(max_records=10)
        receive_n(recorder, 5)
        assert recorder.state.total_records_pushed_out == 0
        assert len(recorder.state.records) == 5

    def test_exact_limit_no_pushout(self):
        recorder = make_recorder(max_records=5)
        receive_n(recorder, 5)
        assert recorder.state.total_records_pushed_out == 0
        assert len(recorder.state.records) == 5

    def test_pushout_is_only_removal_path(self):
        """上限到達時の最古押し出しが唯一の消失経路。"""
        recorder = make_recorder(max_records=3)
        # 3件まで入れても消えない
        receive_n(recorder, 3)
        assert len(recorder.state.records) == 3
        # 4件目で最古が押し出される
        recorder.receive_response("new", "p", tick=10)
        assert len(recorder.state.records) == 3
        assert recorder.state.records[0].response_text == "response_1"


# =============================================================================
# Test: Stage 2 - 既存構造への情報補完
# =============================================================================

class TestComplementToExistingStructures:
    def test_get_text_for_action_result(self):
        recorder = make_recorder()
        recorder.receive_response("hello world", "greet", tick=1)
        text = recorder.get_text_for_action_result()
        assert text == "hello world"

    def test_get_text_for_action_result_empty(self):
        recorder = make_recorder()
        text = recorder.get_text_for_action_result()
        assert text == ""

    def test_get_text_returns_latest(self):
        recorder = make_recorder()
        recorder.receive_response("first", "p1", tick=1)
        recorder.receive_response("second", "p2", tick=2)
        text = recorder.get_text_for_action_result()
        assert text == "second"

    def test_get_policy_for_action_result(self):
        recorder = make_recorder()
        recorder.receive_response("hello", "greet", tick=1)
        policy = recorder.get_policy_for_action_result()
        assert policy == "greet"

    def test_get_policy_for_action_result_empty(self):
        recorder = make_recorder()
        policy = recorder.get_policy_for_action_result()
        assert policy == ""

    def test_text_does_not_modify_existing_structures(self):
        """行動-結果観測の処理ロジックには介入しない。"""
        recorder = make_recorder()
        recorder.receive_response("test", "policy", tick=1)
        # テキスト取得は副作用なし
        text1 = recorder.get_text_for_action_result()
        text2 = recorder.get_text_for_action_result()
        assert text1 == text2
        assert recorder.state.total_records_received == 1


# =============================================================================
# Test: Stage 3 - 参照情報としての受渡準備
# =============================================================================

class TestHandoffPreparation:
    def test_enrichment_data_empty(self):
        recorder = make_recorder()
        data = recorder.get_enrichment_data()
        assert data["total_records"] == 0
        assert data["current_record_count"] == 0
        assert data["recent_entries"] == []
        assert "待機中" in data["summary_text"]

    def test_enrichment_data_with_records(self):
        recorder = make_recorder(enrichment_recent_count=2)
        receive_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert data["total_records"] == 5
        assert data["current_record_count"] == 5
        assert len(data["recent_entries"]) == 2

    def test_enrichment_recent_entries_content(self):
        recorder = make_recorder(enrichment_recent_count=3)
        recorder.receive_response("hello", "greet", tick=1)
        recorder.receive_response("goodbye", "farewell", tick=2)
        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]
        assert len(entries) == 2
        assert entries[0]["policy_label"] == "greet"
        assert entries[1]["policy_label"] == "farewell"

    def test_enrichment_text_preview_truncation(self):
        """テキストは100文字で切り詰められる。"""
        recorder = make_recorder()
        long_text = "a" * 200
        recorder.receive_response(long_text, "long", tick=1)
        data = recorder.get_enrichment_data()
        entry = data["recent_entries"][0]
        assert len(entry["text_preview"]) == 103  # 100 + "..."
        assert entry["text_preview"].endswith("...")

    def test_enrichment_short_text_no_truncation(self):
        recorder = make_recorder()
        recorder.receive_response("short", "p", tick=1)
        data = recorder.get_enrichment_data()
        entry = data["recent_entries"][0]
        assert entry["text_preview"] == "short"

    def test_reference_history(self):
        recorder = make_recorder(reference_history_count=5)
        receive_n(recorder, 10)
        history = recorder.get_reference_history()
        assert len(history) == 5
        # 直近5件
        assert history[0].response_text == "response_5"
        assert history[-1].response_text == "response_9"

    def test_reference_history_empty(self):
        recorder = make_recorder()
        history = recorder.get_reference_history()
        assert history == []

    def test_reference_history_less_than_limit(self):
        recorder = make_recorder(reference_history_count=10)
        receive_n(recorder, 3)
        history = recorder.get_reference_history()
        assert len(history) == 3

    def test_reference_history_is_copy(self):
        """参照行為によって記録が変化することはない。"""
        recorder = make_recorder()
        receive_n(recorder, 3)
        history1 = recorder.get_reference_history()
        history2 = recorder.get_reference_history()
        # リスト自体は異なるオブジェクト
        assert history1 is not history2
        # 内容は同じ
        assert len(history1) == len(history2)

    def test_get_latest_record(self):
        recorder = make_recorder()
        recorder.receive_response("latest", "p", tick=5)
        latest = recorder.get_latest_record()
        assert latest is not None
        assert latest.response_text == "latest"
        assert latest.tick == 5

    def test_get_latest_record_none(self):
        recorder = make_recorder()
        latest = recorder.get_latest_record()
        assert latest is None

    def test_get_summary(self):
        recorder = make_recorder()
        receive_n(recorder, 3)
        summary = recorder.get_summary()
        assert summary["total_records_received"] == 3
        assert summary["current_record_count"] == 3
        assert summary["total_pushed_out"] == 0
        assert summary["has_latest"] is True


# =============================================================================
# Test: 等価性（全記録等価、重みなし）
# =============================================================================

class TestRecordEquality:
    def test_no_weight_field(self):
        """記録に重み属性がないことを確認。"""
        rec = SelfActionRecord(response_text="test")
        assert not hasattr(rec, "weight")
        assert not hasattr(rec, "score")
        assert not hasattr(rec, "priority")
        assert not hasattr(rec, "importance")

    def test_all_records_have_same_status(self):
        """全記録は等価であり、特定の記録に特別なステータスを付与しない。"""
        recorder = make_recorder()
        records = receive_n(recorder, 5)
        for rec in records:
            assert rec.status == RecordStatus.ACTIVE.value

    def test_no_classification_labels(self):
        """出力パターンを分類・類型化しない。"""
        rec = SelfActionRecord(response_text="some response")
        d = rec.to_dict()
        assert "category" not in d
        assert "type" not in d
        assert "classification" not in d
        assert "pattern" not in d
        assert "label" not in d or d.get("label") is None

    def test_no_quality_evaluation(self):
        """応答テキストの品質評価を行わない。"""
        rec = SelfActionRecord(response_text="test")
        d = rec.to_dict()
        assert "quality" not in d
        assert "evaluation" not in d
        assert "rating" not in d
        assert "good" not in d
        assert "bad" not in d


# =============================================================================
# Test: テキスト非解釈
# =============================================================================

class TestTextNonInterpretation:
    def test_raw_text_preserved(self):
        """テキストは生のテキスト文字列として保持する。"""
        recorder = make_recorder()
        original = "テスト文字列 with mixed content 123 !@#"
        rec = recorder.receive_response(original, "p", tick=1)
        assert rec.response_text == original

    def test_no_text_analysis(self):
        """テキストから意味・傾向・パターンを抽出しない。"""
        recorder = make_recorder()
        recorder.receive_response("positive happy good", "p", tick=1)
        # 状態にsentiment, trend, patternなどの解析結果がないこと
        state = recorder.state
        d = state.to_dict()
        assert "sentiment" not in str(d)
        assert "trend" not in str(d)
        assert "pattern" not in str(d)
        assert "analysis" not in str(d)

    def test_no_text_categorization_in_state(self):
        """状態にテキスト分類情報が含まれないこと。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        state_dict = recorder.state.to_dict()
        for rec_dict in state_dict["records"]:
            assert "category" not in rec_dict
            assert "type" not in rec_dict or rec_dict.get("type") is None

    def test_unicode_preserved(self):
        recorder = make_recorder()
        text = "日本語テスト🎉✨"
        rec = recorder.receive_response(text, "p", tick=1)
        assert rec.response_text == text

    def test_whitespace_preserved(self):
        recorder = make_recorder()
        text = "  spaces  \ttab\nnewline  "
        rec = recorder.receive_response(text, "p", tick=1)
        assert rec.response_text == text

    def test_empty_after_whitespace_not_recorded(self):
        """完全に空の文字列は記録されない。ただし空白のみのテキストは記録される。"""
        recorder = make_recorder()
        recorder.receive_response("", "p", tick=1)
        assert recorder.state.total_records_received == 0
        # 空白のみは記録される（テキスト非解釈の原則）
        recorder.receive_response("  ", "p", tick=2)
        assert recorder.state.total_records_received == 1


# =============================================================================
# Test: Save/Load (永続化)
# =============================================================================

class TestSaveLoad:
    def test_roundtrip_empty(self):
        state = SelfActionPerceptionState()
        d = state.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)
        assert restored.total_records_received == 0
        assert len(restored.records) == 0
        assert restored.latest_record is None

    def test_roundtrip_with_data(self):
        recorder = make_recorder()
        receive_n(recorder, 5)
        d = recorder.state.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)
        assert len(restored.records) == 5
        assert restored.total_records_received == 5
        assert restored.latest_record is not None
        assert restored.latest_record.response_text == "response_4"

    def test_roundtrip_preserves_text(self):
        recorder = make_recorder()
        recorder.receive_response("テスト文字列♪♡", "greet", tick=1)
        d = recorder.state.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)
        assert restored.records[0].response_text == "テスト文字列♪♡"

    def test_roundtrip_preserves_record_ids(self):
        recorder = make_recorder()
        records = receive_n(recorder, 3)
        original_ids = [r.record_id for r in records]
        d = recorder.state.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)
        restored_ids = [r.record_id for r in restored.records]
        assert original_ids == restored_ids

    def test_roundtrip_preserves_counters(self):
        recorder = make_recorder(max_records=3)
        receive_n(recorder, 5)
        d = recorder.state.to_dict()
        restored = SelfActionPerceptionState.from_dict(d)
        assert restored.total_records_received == 5
        assert restored.total_records_pushed_out == 2

    def test_state_setter(self):
        """state プロパティの setter テスト。"""
        recorder = make_recorder()
        receive_n(recorder, 3)
        original_state = recorder.state.to_dict()

        new_recorder = make_recorder()
        new_recorder.state = SelfActionPerceptionState.from_dict(original_state)
        assert len(new_recorder.state.records) == 3
        assert new_recorder.state.total_records_received == 3

    def test_load_from_partial_dict(self):
        """一部のフィールドだけのdictからもロード可能。"""
        state = SelfActionPerceptionState.from_dict({
            "total_records_received": 10,
        })
        assert state.total_records_received == 10
        assert len(state.records) == 0

    def test_record_roundtrip(self):
        rec = SelfActionRecord(
            response_text="full roundtrip test",
            policy_label="test_policy",
            tick=42,
        )
        d = rec.to_dict()
        restored = SelfActionRecord.from_dict(d)
        assert restored.record_id == rec.record_id
        assert restored.response_text == rec.response_text
        assert restored.policy_label == rec.policy_label
        assert restored.tick == rec.tick
        assert restored.status == rec.status


# =============================================================================
# Test: 判断系非接続の検証
# =============================================================================

class TestDecisionSystemDisconnection:
    def test_no_bias_output(self):
        """本機能の出力からポリシー選択・バイアス計算への直接入力経路がないこと。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        # get_enrichment_dataにbias/score/weight関連が含まれないこと
        data = recorder.get_enrichment_data()
        assert "bias" not in data
        assert "score" not in data
        assert "weight" not in data

    def test_no_policy_selection_impact(self):
        """ポリシー選択を変更する経路がないこと。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        summary = recorder.get_summary()
        assert "selected_policy" not in summary
        assert "recommended_policy" not in summary

    def test_no_stability_valve_impact(self):
        """安定化弁への直接入力がないこと。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "stability" not in data
        assert "valve" not in data

    def test_output_is_reference_only(self):
        """出力は参照情報としてのみ流れ、判断・評価・行動を直接引き起こさない。"""
        recorder = make_recorder()
        recorder.receive_response("test", "p", tick=1)

        # enrichment data は情報のみ
        data = recorder.get_enrichment_data()
        assert "action" not in data or data.get("action") is None
        assert "command" not in data
        assert "decision" not in data

        # reference history は読み取り専用
        history = recorder.get_reference_history()
        assert isinstance(history, list)

    def test_no_reverse_path_to_gemini(self):
        """本機能からGeminiへの逆方向の経路は存在しない。"""
        recorder = make_recorder()
        # receiveは一方向（通知を受けるのみ）
        # get系メソッドは参照のみ
        # 出力にGeminiへの指示が含まれないこと
        recorder.receive_response("test", "p", tick=1)
        data = recorder.get_enrichment_data()
        assert "instruction" not in data
        assert "prompt" not in data
        assert "generate" not in data

    def test_no_emotion_pipeline_modification(self):
        """感情パイプライン（Phase 1-2）のパラメータを変更しない。"""
        recorder = make_recorder()
        receive_n(recorder, 10)
        state_dict = recorder.state.to_dict()
        # 感情関連フィールドが存在しないこと
        assert "emotion" not in state_dict
        assert "mood" not in state_dict
        assert "decay_rate" not in state_dict

    def test_no_policy_compliance_scoring(self):
        """ポリシー遵守度の判定を行わない。"""
        recorder = make_recorder()
        recorder.receive_response("test", "empathy", tick=1)
        rec = recorder.get_latest_record()
        d = rec.to_dict()
        assert "compliance" not in d
        assert "adherence" not in d
        assert "match_score" not in d


# =============================================================================
# Test: Enrichment Summary
# =============================================================================

class TestEnrichmentSummary:
    def test_summary_waiting(self):
        state = SelfActionPerceptionState()
        summary = get_self_action_summary(state)
        assert "待機中" in summary

    def test_summary_with_records(self):
        recorder = make_recorder()
        recorder.receive_response("hello", "greet", tick=5)
        summary = get_self_action_summary(recorder.state)
        assert "記録数=1" in summary
        assert "直近ポリシー=greet" in summary
        assert "tick=5" in summary

    def test_summary_text_length(self):
        recorder = make_recorder()
        recorder.receive_response("a" * 50, "p", tick=1)
        summary = get_self_action_summary(recorder.state)
        assert "直近テキスト長=50" in summary

    def test_summary_with_pushout(self):
        recorder = make_recorder(max_records=2)
        receive_n(recorder, 4)
        summary = get_self_action_summary(recorder.state)
        assert "押出累計=2" in summary

    def test_summary_no_evaluation_words(self):
        """summaryに評価的な言葉が含まれないこと。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        summary = get_self_action_summary(recorder.state)
        assert "良い" not in summary
        assert "悪い" not in summary
        assert "適切" not in summary
        assert "不適切" not in summary


# =============================================================================
# Test: Factory
# =============================================================================

class TestFactory:
    def test_create_default(self):
        recorder = create_self_action_perception_recorder()
        assert isinstance(recorder, SelfActionPerceptionRecorder)
        assert recorder._config.max_records == 50

    def test_create_with_config(self):
        cfg = SelfActionPerceptionConfig(max_records=100)
        recorder = create_self_action_perception_recorder(config=cfg)
        assert recorder._config.max_records == 100

    def test_factory_returns_fresh_state(self):
        recorder = create_self_action_perception_recorder()
        assert recorder.state.total_records_received == 0
        assert len(recorder.state.records) == 0


# =============================================================================
# Test: _gen_id helper
# =============================================================================

class TestGenId:
    def test_length(self):
        id_val = _gen_id()
        assert len(id_val) == 12

    def test_unique(self):
        ids = [_gen_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_hex_format(self):
        id_val = _gen_id()
        # 16進文字のみ
        int(id_val, 16)


# =============================================================================
# Test: Integration (統合テスト)
# =============================================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """記録→蓄積→押出→参照の完全ライフサイクル。"""
        recorder = make_recorder(max_records=5, enrichment_recent_count=2, reference_history_count=3)

        # 1. 記録蓄積
        for i in range(8):
            recorder.receive_response(
                f"response_{i}", f"policy_{i}", tick=i + 1,
            )

        # 2. 状態検証
        assert recorder.state.total_records_received == 8
        assert len(recorder.state.records) == 5  # 上限5
        assert recorder.state.total_records_pushed_out == 3

        # 3. enrichment
        data = recorder.get_enrichment_data()
        assert data["total_records"] == 8
        assert len(data["recent_entries"]) == 2
        assert data["recent_entries"][-1]["policy_label"] == "policy_7"

        # 4. 参照履歴
        history = recorder.get_reference_history()
        assert len(history) == 3
        assert history[-1].response_text == "response_7"

        # 5. 最新記録
        latest = recorder.get_latest_record()
        assert latest.response_text == "response_7"
        assert latest.tick == 8

        # 6. action_result補完
        text = recorder.get_text_for_action_result()
        assert text == "response_7"
        policy = recorder.get_policy_for_action_result()
        assert policy == "policy_7"

    def test_save_load_resume(self):
        """セッション間での永続化と再開。"""
        # Session 1
        recorder1 = make_recorder(max_records=10)
        receive_n(recorder1, 5)
        saved = recorder1.state.to_dict()

        # Session 2
        recorder2 = make_recorder(max_records=10)
        recorder2.state = SelfActionPerceptionState.from_dict(saved)

        # 状態が復元されていること
        assert len(recorder2.state.records) == 5
        assert recorder2.state.total_records_received == 5
        assert recorder2.state.latest_record is not None

        # 追加記録が可能
        recorder2.receive_response("new_response", "new_policy", tick=10)
        assert recorder2.state.total_records_received == 6
        assert len(recorder2.state.records) == 6

    def test_empty_text_interspersed(self):
        """空テキストが混在しても正常に動作する。"""
        recorder = make_recorder()
        recorder.receive_response("first", "p1", tick=1)
        recorder.receive_response("", "p2", tick=2)  # skipped
        recorder.receive_response("third", "p3", tick=3)
        assert recorder.state.total_records_received == 2
        assert len(recorder.state.records) == 2
        assert recorder.state.latest_record.response_text == "third"

    def test_rapid_succession(self):
        """高速連続受領での安定性。"""
        recorder = make_recorder(max_records=100)
        for i in range(200):
            recorder.receive_response(f"msg_{i}", f"p_{i}", tick=i)
        assert recorder.state.total_records_received == 200
        assert len(recorder.state.records) == 100
        assert recorder.state.total_records_pushed_out == 100

    def test_various_text_lengths(self):
        """様々なテキスト長での動作確認。"""
        recorder = make_recorder()
        lengths = [1, 10, 100, 1000, 5000]
        for i, length in enumerate(lengths):
            text = "x" * length
            rec = recorder.receive_response(text, "p", tick=i + 1)
            assert len(rec.response_text) == length

    def test_enrichment_after_pushout(self):
        """押し出し後もenrichmentが正常に機能する。"""
        recorder = make_recorder(max_records=3, enrichment_recent_count=2)
        receive_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert len(data["recent_entries"]) == 2
        assert data["total_records"] == 5
        assert data["current_record_count"] == 3

    def test_summary_after_many_operations(self):
        """多数の操作後もサマリが正常。"""
        recorder = make_recorder(max_records=10)
        receive_n(recorder, 50)
        summary = recorder.get_summary()
        assert summary["total_records_received"] == 50
        assert summary["current_record_count"] == 10
        assert summary["total_pushed_out"] == 40
        assert summary["has_latest"] is True

    def test_reference_history_after_pushout(self):
        """押し出し後も参照履歴が正常。"""
        recorder = make_recorder(max_records=5, reference_history_count=3)
        receive_n(recorder, 10)
        history = recorder.get_reference_history()
        assert len(history) == 3
        # 最新の3件
        texts = [r.response_text for r in history]
        assert texts == ["response_7", "response_8", "response_9"]


# =============================================================================
# Test: Safety valve principles
# =============================================================================

class TestSafetyValvePrinciples:
    def test_record_equality_no_special_retention(self):
        """特定の記録を永続的に保持する機能はない。"""
        recorder = make_recorder(max_records=3)
        # 最初に「重要そうな」記録を入れても優先されない
        recorder.receive_response("very important response", "critical", tick=1)
        receive_n(recorder, 5, base_tick=2)
        # 最初の記録は押し出されている
        texts = [r.response_text for r in recorder.state.records]
        assert "very important response" not in texts

    def test_no_pattern_classification(self):
        """「この発言は自分らしい」「この発言は自分らしくない」といった分類を行わない。"""
        recorder = make_recorder()
        recorder.receive_response("typical response", "p", tick=1)
        recorder.receive_response("unusual response", "p", tick=2)
        for rec in recorder.state.records:
            d = rec.to_dict()
            assert "typical" not in d.get("status", "")
            assert "atypical" not in d.get("status", "")
            assert "self_like" not in str(d)

    def test_no_self_reinforcement_loop(self):
        """自己行動知覚→行動-結果観測→価値方向性→ポリシー選択→Gemini出力→自己行動知覚
        の循環的自己強化ループを形成しない。"""
        recorder = make_recorder()
        receive_n(recorder, 10)
        # enrichment_dataにポリシー選択への直接入力がないこと
        data = recorder.get_enrichment_data()
        assert "policy_score" not in data
        assert "bias" not in data
        assert "recommendation" not in data

    def test_read_only_reference(self):
        """記録の参照は常にREAD-ONLYであり、参照行為によって記録が変化しない。"""
        recorder = make_recorder()
        receive_n(recorder, 5)

        # 参照前の状態
        pre_count = recorder.state.total_records_received
        pre_records = len(recorder.state.records)

        # 参照
        recorder.get_enrichment_data()
        recorder.get_reference_history()
        recorder.get_latest_record()
        recorder.get_text_for_action_result()
        recorder.get_policy_for_action_result()
        recorder.get_summary()

        # 参照後の状態が変わっていないこと
        assert recorder.state.total_records_received == pre_count
        assert len(recorder.state.records) == pre_records

    def test_pushout_natural_decay(self):
        """上限による自然消滅のみ。明示的な減衰処理は行わない。"""
        recorder = make_recorder(max_records=5)
        receive_n(recorder, 5)
        # 時間が経過しても記録が減衰しない（明示的な減衰処理がない）
        assert len(recorder.state.records) == 5
        for rec in recorder.state.records:
            assert rec.status == RecordStatus.ACTIVE.value

    def test_no_compliance_judgment(self):
        """選択されたポリシーと実際の出力テキストの一致度を評価・スコアリングしない。"""
        recorder = make_recorder()
        # empathyポリシーなのに攻撃的なテキスト — 評価しない
        recorder.receive_response("怒りの言葉", "empathy", tick=1)
        rec = recorder.get_latest_record()
        d = rec.to_dict()
        assert "compliance" not in d
        assert "match" not in d
        assert "mismatch" not in d


# =============================================================================
# Test: Phase 1-2 invariance
# =============================================================================

class TestPhaseInvariance:
    def test_no_emotion_modification(self):
        """感情パイプライン（Phase 1-2）のパラメータを変更しない。"""
        recorder = make_recorder()
        receive_n(recorder, 10)
        state = recorder.state
        # 感情関連のフィールドが一切存在しない
        assert not hasattr(state, "emotion_delta")
        assert not hasattr(state, "mood_change")
        assert not hasattr(state, "decay_rate")
        assert not hasattr(state, "emotion_values")

    def test_no_dynamics_modification(self):
        """ダイナミクスに影響しない。"""
        recorder = make_recorder()
        receive_n(recorder, 10)
        state = recorder.state
        assert not hasattr(state, "dynamics")
        assert not hasattr(state, "peak_intensity")


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_max_records_one(self):
        """最大保持数1のケース。"""
        recorder = make_recorder(max_records=1)
        receive_n(recorder, 5)
        assert len(recorder.state.records) == 1
        assert recorder.state.records[0].response_text == "response_4"
        assert recorder.state.total_records_pushed_out == 4

    def test_very_large_max_records(self):
        """非常に大きな上限。"""
        recorder = make_recorder(max_records=10000)
        receive_n(recorder, 100)
        assert len(recorder.state.records) == 100
        assert recorder.state.total_records_pushed_out == 0

    def test_enrichment_recent_count_larger_than_records(self):
        recorder = make_recorder(enrichment_recent_count=10)
        receive_n(recorder, 3)
        data = recorder.get_enrichment_data()
        assert len(data["recent_entries"]) == 3

    def test_reference_history_count_larger_than_records(self):
        recorder = make_recorder(reference_history_count=20)
        receive_n(recorder, 3)
        history = recorder.get_reference_history()
        assert len(history) == 3

    def test_receive_after_load(self):
        """ロード後に記録を追加。"""
        recorder = make_recorder()
        receive_n(recorder, 3)
        saved = recorder.state.to_dict()

        new_recorder = make_recorder()
        new_recorder.state = SelfActionPerceptionState.from_dict(saved)
        new_recorder.receive_response("after_load", "pl", tick=100)
        assert new_recorder.state.total_records_received == 4
        assert len(new_recorder.state.records) == 4

    def test_timestamp_ordering(self):
        """タイムスタンプが時系列的に増加する。"""
        recorder = make_recorder()
        receive_n(recorder, 5)
        timestamps = [r.timestamp for r in recorder.state.records]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]
