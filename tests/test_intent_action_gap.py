"""
tests/test_intent_action_gap.py - 意図-行動間の乖離認知のテスト

カバー範囲:
- 対構成（Stage 1）
- 多断面差異記述（Stage 2）
- 蓄積・FIFO押し出し（Stage 3）
- 記録の等価性（重みなし）
- スキップ計数
- enrichment出力
- READ-ONLYアクセサ
- save/load
- 安全弁（ポリシー選択非接続、パターン抽出禁止、強調禁止）
- 3経路遮断の検証
- ファクトリ
- 統合テスト
"""

import time
import pytest

from psyche.intent_action_gap import (
    GapRecord,
    IntentActionGapState,
    IntentActionGapConfig,
    IntentActionGapRecorder,
    get_gap_summary,
    create_intent_action_gap_recorder,
    _gen_id,
)


# =============================================================================
# Helpers
# =============================================================================

def make_recorder(max_records: int = 50, **kwargs) -> IntentActionGapRecorder:
    """テスト用レコーダーを生成する。"""
    config = IntentActionGapConfig(max_records=max_records, **kwargs)
    return IntentActionGapRecorder(config=config)


def process_n(
    recorder: IntentActionGapRecorder,
    n: int,
    base_tick: int = 1,
    context_info: str = "",
) -> list[GapRecord]:
    """n件の乖離記録を処理する。"""
    records = []
    for i in range(n):
        rec = recorder.process_action_record(
            response_text=f"response_{i}",
            policy_label=f"policy_{i}",
            tick=base_tick + i,
            context_info=context_info,
        )
        if rec is not None:
            records.append(rec)
    return records


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
        int(id_val, 16)


# =============================================================================
# Test: GapRecord
# =============================================================================

class TestGapRecord:
    def test_default_creation(self):
        rec = GapRecord()
        assert rec.record_id != ""
        assert rec.policy_label == ""
        assert rec.text_snippet == ""
        assert rec.tick == 0
        assert rec.context_info == ""
        assert rec.timestamp > 0

    def test_creation_with_values(self):
        rec = GapRecord(
            policy_label="empathy",
            text_snippet="hello world",
            tick=5,
            context_info="mood-based selection",
        )
        assert rec.policy_label == "empathy"
        assert rec.text_snippet == "hello world"
        assert rec.tick == 5
        assert rec.context_info == "mood-based selection"

    def test_unique_record_ids(self):
        rec1 = GapRecord()
        rec2 = GapRecord()
        assert rec1.record_id != rec2.record_id

    def test_to_dict(self):
        rec = GapRecord(
            policy_label="greet",
            text_snippet="hello",
            tick=10,
            context_info="context",
        )
        d = rec.to_dict()
        assert d["policy_label"] == "greet"
        assert d["text_snippet"] == "hello"
        assert d["tick"] == 10
        assert d["context_info"] == "context"
        assert "record_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        original = GapRecord(
            policy_label="empathy",
            text_snippet="test text",
            tick=7,
            context_info="ctx",
        )
        d = original.to_dict()
        restored = GapRecord.from_dict(d)
        assert restored.record_id == original.record_id
        assert restored.policy_label == original.policy_label
        assert restored.text_snippet == original.text_snippet
        assert restored.tick == original.tick
        assert restored.context_info == original.context_info

    def test_from_dict_empty(self):
        rec = GapRecord.from_dict({})
        assert rec.policy_label == ""
        assert rec.text_snippet == ""
        assert rec.tick == 0
        assert rec.context_info == ""

    def test_no_weight_or_score(self):
        """記録には重み・スコア・優先度・重要度などの評価的属性を付与しない。"""
        rec = GapRecord(policy_label="test", text_snippet="text")
        d = rec.to_dict()
        assert "weight" not in d
        assert "score" not in d
        assert "priority" not in d
        assert "importance" not in d
        assert "quality" not in d
        assert "rating" not in d
        assert "gap_size" not in d
        assert "divergence" not in d
        assert "compliance" not in d

    def test_no_evaluation_attribute(self):
        """差異の有無・程度を判定する属性がないこと。"""
        rec = GapRecord(policy_label="p", text_snippet="t")
        assert not hasattr(rec, "gap_magnitude")
        assert not hasattr(rec, "match_degree")
        assert not hasattr(rec, "consistency")
        assert not hasattr(rec, "alignment")

    def test_four_facets_present(self):
        """4断面（ラベル・テキスト・時間・文脈）が存在すること。"""
        rec = GapRecord(
            policy_label="label",
            text_snippet="snippet",
            tick=42,
            context_info="context",
        )
        d = rec.to_dict()
        assert "policy_label" in d
        assert "text_snippet" in d
        assert "tick" in d
        assert "context_info" in d


# =============================================================================
# Test: IntentActionGapState
# =============================================================================

class TestIntentActionGapState:
    def test_default_state(self):
        state = IntentActionGapState()
        assert state.records == []
        assert state.latest_record is None
        assert state.skip_count == 0

    def test_to_dict(self):
        state = IntentActionGapState()
        state.skip_count = 3
        d = state.to_dict()
        assert d["skip_count"] == 3
        assert d["records"] == []
        assert d["latest_record"] is None

    def test_to_dict_with_records(self):
        state = IntentActionGapState()
        rec = GapRecord(policy_label="p", text_snippet="t")
        state.records.append(rec)
        state.latest_record = rec
        d = state.to_dict()
        assert len(d["records"]) == 1
        assert d["latest_record"] is not None
        assert d["latest_record"]["policy_label"] == "p"

    def test_from_dict(self):
        original = IntentActionGapState()
        rec = GapRecord(policy_label="p", text_snippet="t", tick=3)
        original.records.append(rec)
        original.latest_record = rec
        original.skip_count = 2

        d = original.to_dict()
        restored = IntentActionGapState.from_dict(d)

        assert len(restored.records) == 1
        assert restored.records[0].policy_label == "p"
        assert restored.latest_record is not None
        assert restored.latest_record.policy_label == "p"
        assert restored.skip_count == 2

    def test_from_dict_empty(self):
        state = IntentActionGapState.from_dict({})
        assert state.records == []
        assert state.latest_record is None
        assert state.skip_count == 0

    def test_from_dict_no_latest(self):
        state = IntentActionGapState.from_dict({
            "records": [],
            "latest_record": None,
        })
        assert state.latest_record is None


# =============================================================================
# Test: IntentActionGapConfig
# =============================================================================

class TestIntentActionGapConfig:
    def test_defaults(self):
        cfg = IntentActionGapConfig()
        assert cfg.max_records == 50
        assert cfg.text_snippet_max_length == 150
        assert cfg.enrichment_recent_count == 3

    def test_custom_values(self):
        cfg = IntentActionGapConfig(
            max_records=100,
            text_snippet_max_length=200,
            enrichment_recent_count=5,
        )
        assert cfg.max_records == 100
        assert cfg.text_snippet_max_length == 200
        assert cfg.enrichment_recent_count == 5


# =============================================================================
# Test: Stage 1 - 対の構成
# =============================================================================

class TestPairConstruction:
    def test_basic_pair(self):
        """ポリシーラベルと出力テキストの対を構成する。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            response_text="hello world",
            policy_label="greet",
            tick=1,
        )
        assert rec is not None
        assert rec.policy_label == "greet"
        assert rec.tick == 1

    def test_pair_with_context(self):
        """文脈断面付きの対構成。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            response_text="hello",
            policy_label="greet",
            tick=1,
            context_info="mood-based selection",
        )
        assert rec is not None
        assert rec.context_info == "mood-based selection"

    def test_skip_missing_response_text(self):
        """出力テキスト欠損でスキップ。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            response_text="",
            policy_label="greet",
            tick=1,
        )
        assert rec is None
        assert recorder.state.skip_count == 1

    def test_skip_missing_policy_label(self):
        """ポリシーラベル欠損でスキップ。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            response_text="hello",
            policy_label="",
            tick=1,
        )
        assert rec is None
        assert recorder.state.skip_count == 1

    def test_skip_both_missing(self):
        """両方欠損でスキップ。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            response_text="",
            policy_label="",
            tick=1,
        )
        assert rec is None
        assert recorder.state.skip_count == 1

    def test_skip_count_accumulates(self):
        """スキップ計数は累積する。"""
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)
        recorder.process_action_record("t", "", tick=2)
        recorder.process_action_record("", "", tick=3)
        assert recorder.state.skip_count == 3

    def test_skip_does_not_affect_records(self):
        """スキップしても既存記録に影響しない。"""
        recorder = make_recorder()
        recorder.process_action_record("text", "policy", tick=1)
        recorder.process_action_record("", "policy", tick=2)  # skip
        assert len(recorder.state.records) == 1
        assert recorder.state.skip_count == 1

    def test_one_record_per_call(self):
        """1回の呼び出しから1つの対が生成される。"""
        recorder = make_recorder()
        pre_count = len(recorder.state.records)
        recorder.process_action_record("text", "policy", tick=1)
        assert len(recorder.state.records) == pre_count + 1


# =============================================================================
# Test: Stage 2 - 多断面での差異記述
# =============================================================================

class TestMultiFacetDescription:
    def test_label_facet(self):
        """ラベル断面: ポリシーラベルの文字列記録。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("text", "empathy", tick=1)
        assert rec.policy_label == "empathy"

    def test_text_facet_short(self):
        """テキスト断面: 短いテキストはそのまま記録。"""
        recorder = make_recorder(text_snippet_max_length=150)
        rec = recorder.process_action_record("short text", "p", tick=1)
        assert rec.text_snippet == "short text"

    def test_text_facet_truncation(self):
        """テキスト断面: 長いテキストは先頭部分のみ記録。"""
        recorder = make_recorder(text_snippet_max_length=10)
        rec = recorder.process_action_record("a" * 100, "p", tick=1)
        assert len(rec.text_snippet) == 10
        assert rec.text_snippet == "a" * 10

    def test_text_facet_exact_limit(self):
        """テキスト断面: ちょうど上限のテキスト。"""
        recorder = make_recorder(text_snippet_max_length=50)
        rec = recorder.process_action_record("b" * 50, "p", tick=1)
        assert len(rec.text_snippet) == 50

    def test_time_facet(self):
        """時間断面: ティック番号の記録。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("text", "p", tick=42)
        assert rec.tick == 42

    def test_context_facet_present(self):
        """文脈断面: 根拠情報が利用可能な場合。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("text", "p", tick=1, context_info="reason info")
        assert rec.context_info == "reason info"

    def test_context_facet_absent(self):
        """文脈断面: 根拠情報が利用不可能な場合は空。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("text", "p", tick=1)
        assert rec.context_info == ""

    def test_facets_independent(self):
        """各断面は独立した記述。断面間の優先順位・重み付け・統合処理は存在しない。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            "text", "policy", tick=5, context_info="ctx",
        )
        d = rec.to_dict()
        # 統合スコアや優先順位が存在しないこと
        assert "combined_score" not in d
        assert "facet_priority" not in d
        assert "facet_weight" not in d

    def test_no_difference_judgment(self):
        """差異の有無・程度を本機能が判定しないこと。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("angry text", "empathy", tick=1)
        d = rec.to_dict()
        assert "gap" not in d or d.get("gap") is None
        assert "mismatch" not in d
        assert "match" not in d
        assert "divergence" not in d
        assert "alignment" not in d


# =============================================================================
# Test: Stage 3 - 蓄積とFIFO押し出し
# =============================================================================

class TestAccumulationAndPushout:
    def test_basic_accumulation(self):
        """記録が時系列順に蓄積される。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        assert len(recorder.state.records) == 5

    def test_time_series_order(self):
        """記録は時系列順に蓄積される。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        ticks = [r.tick for r in recorder.state.records]
        assert ticks == sorted(ticks)

    def test_latest_record_updated(self):
        """直近乖離記録は最新の1件を保持。"""
        recorder = make_recorder()
        recorder.process_action_record("first", "p1", tick=1)
        assert recorder.state.latest_record.policy_label == "p1"
        recorder.process_action_record("second", "p2", tick=2)
        assert recorder.state.latest_record.policy_label == "p2"

    def test_pushout_at_limit(self):
        """上限到達時は最古から押し出す。"""
        recorder = make_recorder(max_records=5)
        process_n(recorder, 7)
        assert len(recorder.state.records) == 5

    def test_pushout_preserves_newest(self):
        """押し出し後は最新のN件が残る。"""
        recorder = make_recorder(max_records=3)
        process_n(recorder, 5)
        policies = [r.policy_label for r in recorder.state.records]
        assert policies == ["policy_2", "policy_3", "policy_4"]

    def test_pushout_removes_oldest(self):
        """最古の記録が押し出される。"""
        recorder = make_recorder(max_records=3)
        process_n(recorder, 5)
        remaining_ticks = [r.tick for r in recorder.state.records]
        assert 1 not in remaining_ticks
        assert 2 not in remaining_ticks

    def test_no_pushout_within_limit(self):
        """上限内では押し出さない。"""
        recorder = make_recorder(max_records=10)
        process_n(recorder, 5)
        assert len(recorder.state.records) == 5

    def test_exact_limit_no_pushout(self):
        """ちょうど上限の場合は押し出さない。"""
        recorder = make_recorder(max_records=5)
        process_n(recorder, 5)
        assert len(recorder.state.records) == 5

    def test_pushout_cumulative(self):
        """大量の記録での累積的押し出し。"""
        recorder = make_recorder(max_records=3)
        process_n(recorder, 10)
        assert len(recorder.state.records) == 3
        # 最新3件のみ残る
        policies = [r.policy_label for r in recorder.state.records]
        assert policies == ["policy_7", "policy_8", "policy_9"]

    def test_pushout_is_only_removal_path(self):
        """上限到達時の最古押し出しが唯一の消失経路。"""
        recorder = make_recorder(max_records=3)
        process_n(recorder, 3)
        assert len(recorder.state.records) == 3
        # 4件目で最古が押し出される
        recorder.process_action_record("new", "new_p", tick=10)
        assert len(recorder.state.records) == 3
        assert recorder.state.records[0].policy_label == "policy_1"

    def test_immutability_of_records(self):
        """一度記録された乖離記録は変更されない。新規記録の追加のみ。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("text", "policy", tick=1)
        original_id = rec.record_id
        original_label = rec.policy_label

        # 新しい記録を追加しても既存記録は変更されない
        recorder.process_action_record("text2", "policy2", tick=2)
        assert recorder.state.records[0].record_id == original_id
        assert recorder.state.records[0].policy_label == original_label


# =============================================================================
# Test: 記録の等価性
# =============================================================================

class TestRecordEquality:
    def test_no_weight_field(self):
        """記録に重み属性がないこと。"""
        rec = GapRecord(policy_label="p", text_snippet="t")
        assert not hasattr(rec, "weight")
        assert not hasattr(rec, "score")
        assert not hasattr(rec, "priority")
        assert not hasattr(rec, "importance")

    def test_all_records_equal(self):
        """すべての乖離記録は等価である。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        for rec in recorder.state.records:
            d = rec.to_dict()
            assert "weight" not in d
            assert "score" not in d
            assert "priority" not in d

    def test_no_special_retention(self):
        """特定の記録を永続的に保持する機能はない。"""
        recorder = make_recorder(max_records=3)
        # 最初に入れた記録も等しく押し出される
        recorder.process_action_record("important", "critical", tick=1)
        process_n(recorder, 5, base_tick=2)
        policies = [r.policy_label for r in recorder.state.records]
        assert "critical" not in policies

    def test_same_policy_different_text(self):
        """同一ポリシーラベルに対して異なるテキスト断片が繰り返し記録されうる。"""
        recorder = make_recorder()
        recorder.process_action_record("text_a", "empathy", tick=1)
        recorder.process_action_record("text_b", "empathy", tick=2)
        recorder.process_action_record("text_c", "empathy", tick=3)
        assert len(recorder.state.records) == 3
        snippets = [r.text_snippet for r in recorder.state.records]
        assert snippets == ["text_a", "text_b", "text_c"]


# =============================================================================
# Test: スキップ計数
# =============================================================================

class TestSkipCount:
    def test_initial_skip_count(self):
        state = IntentActionGapState()
        assert state.skip_count == 0

    def test_skip_increments(self):
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)
        assert recorder.state.skip_count == 1
        recorder.process_action_record("t", "", tick=2)
        assert recorder.state.skip_count == 2

    def test_skip_does_not_affect_processing(self):
        """スキップ計数は診断情報のみであり、処理の分岐には影響しない。"""
        recorder = make_recorder()
        # 10回スキップさせる
        for i in range(10):
            recorder.process_action_record("", "p", tick=i)
        assert recorder.state.skip_count == 10

        # その後の正常な記録には影響しない
        rec = recorder.process_action_record("text", "policy", tick=100)
        assert rec is not None
        assert len(recorder.state.records) == 1

    def test_skip_persisted(self):
        """スキップ計数は永続化対象。"""
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)
        recorder.process_action_record("", "p", tick=2)
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        assert restored.skip_count == 2

    def test_skip_interspersed(self):
        """スキップと正常記録が混在する場合。"""
        recorder = make_recorder()
        recorder.process_action_record("text1", "p1", tick=1)  # OK
        recorder.process_action_record("", "p2", tick=2)       # skip
        recorder.process_action_record("text3", "p3", tick=3)  # OK
        recorder.process_action_record("text4", "", tick=4)    # skip
        recorder.process_action_record("text5", "p5", tick=5)  # OK
        assert len(recorder.state.records) == 3
        assert recorder.state.skip_count == 2


# =============================================================================
# Test: Enrichment 出力
# =============================================================================

class TestEnrichment:
    def test_enrichment_empty(self):
        recorder = make_recorder()
        data = recorder.get_enrichment_data()
        assert data["record_count"] == 0
        assert data["skip_count"] == 0
        assert data["recent_entries"] == []
        assert "待機中" in data["summary_text"]

    def test_enrichment_with_records(self):
        recorder = make_recorder(enrichment_recent_count=2)
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert data["record_count"] == 5
        assert len(data["recent_entries"]) == 2

    def test_enrichment_recent_entries_content(self):
        recorder = make_recorder(enrichment_recent_count=3)
        recorder.process_action_record("hello", "greet", tick=1)
        recorder.process_action_record("goodbye", "farewell", tick=2)
        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]
        assert len(entries) == 2
        assert entries[0]["policy_label"] == "greet"
        assert entries[1]["policy_label"] == "farewell"

    def test_enrichment_entry_structure(self):
        """各エントリがポリシーラベルとテキスト断片の対を含むこと。"""
        recorder = make_recorder()
        recorder.process_action_record("text", "policy", tick=5)
        data = recorder.get_enrichment_data()
        entry = data["recent_entries"][0]
        assert "policy_label" in entry
        assert "text_snippet" in entry
        assert "tick" in entry

    def test_enrichment_no_emphasis(self):
        """enrichmentテキストに強調表現がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        summary = data["summary_text"]
        # 「重要な乖離」「注目すべき差異」等の強調表現を使わない
        assert "重要" not in summary
        assert "注目" not in summary
        assert "重大" not in summary
        assert "顕著" not in summary
        assert "著しい" not in summary

    def test_enrichment_no_evaluation(self):
        """enrichmentに評価的な言葉がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        summary = data["summary_text"]
        assert "良い" not in summary
        assert "悪い" not in summary
        assert "適切" not in summary
        assert "不適切" not in summary
        assert "改善" not in summary

    def test_enrichment_equal_listing(self):
        """列挙に際して特定の記録を強調・選別しない。"""
        recorder = make_recorder(enrichment_recent_count=3)
        process_n(recorder, 3)
        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]
        # 全エントリの構造が同一であること
        for entry in entries:
            assert set(entry.keys()) == {"policy_label", "text_snippet", "tick"}

    def test_enrichment_fewer_than_recent_count(self):
        """記録数がenrichment_recent_countより少ない場合。"""
        recorder = make_recorder(enrichment_recent_count=10)
        process_n(recorder, 3)
        data = recorder.get_enrichment_data()
        assert len(data["recent_entries"]) == 3

    def test_enrichment_with_skips(self):
        """スキップ計数がenrichmentに含まれること。"""
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)  # skip
        recorder.process_action_record("text", "p", tick=2)
        data = recorder.get_enrichment_data()
        assert data["skip_count"] == 1


# =============================================================================
# Test: READ-ONLY アクセサ
# =============================================================================

class TestReadOnlyAccessor:
    def test_get_all_records_returns_list(self):
        recorder = make_recorder()
        process_n(recorder, 5)
        records = recorder.get_all_records()
        assert isinstance(records, list)
        assert len(records) == 5

    def test_get_all_records_is_copy(self):
        """参照行為によって記録が変化しない。返されるリストはコピー。"""
        recorder = make_recorder()
        process_n(recorder, 3)
        records1 = recorder.get_all_records()
        records2 = recorder.get_all_records()
        assert records1 is not records2
        assert len(records1) == len(records2)

    def test_get_all_records_no_filtering(self):
        """フィルタリング・選別機能をアクセサに持たせない。蓄積リスト全体を返す。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        records = recorder.get_all_records()
        assert len(records) == 10

    def test_get_all_records_empty(self):
        recorder = make_recorder()
        records = recorder.get_all_records()
        assert records == []

    def test_get_latest_record(self):
        recorder = make_recorder()
        recorder.process_action_record("text", "p", tick=5)
        latest = recorder.get_latest_record()
        assert latest is not None
        assert latest.policy_label == "p"
        assert latest.tick == 5

    def test_get_latest_record_none(self):
        recorder = make_recorder()
        latest = recorder.get_latest_record()
        assert latest is None

    def test_get_latest_always_newest(self):
        recorder = make_recorder()
        recorder.process_action_record("first", "p1", tick=1)
        recorder.process_action_record("second", "p2", tick=2)
        recorder.process_action_record("third", "p3", tick=3)
        latest = recorder.get_latest_record()
        assert latest.policy_label == "p3"

    def test_reference_does_not_modify_state(self):
        """参照行為によって記録が変化しないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        pre_count = len(recorder.state.records)
        pre_skip = recorder.state.skip_count

        # 全ての参照メソッドを呼び出す
        recorder.get_enrichment_data()
        recorder.get_all_records()
        recorder.get_latest_record()
        recorder.get_summary()

        assert len(recorder.state.records) == pre_count
        assert recorder.state.skip_count == pre_skip

    def test_get_summary(self):
        recorder = make_recorder()
        process_n(recorder, 3)
        summary = recorder.get_summary()
        assert summary["record_count"] == 3
        assert summary["skip_count"] == 0
        assert summary["has_latest"] is True
        assert summary["latest_tick"] == 3
        assert summary["latest_policy"] == "policy_2"

    def test_get_summary_empty(self):
        recorder = make_recorder()
        summary = recorder.get_summary()
        assert summary["record_count"] == 0
        assert summary["skip_count"] == 0
        assert summary["has_latest"] is False
        assert summary["latest_tick"] == 0
        assert summary["latest_policy"] == ""


# =============================================================================
# Test: Save/Load (永続化)
# =============================================================================

class TestSaveLoad:
    def test_roundtrip_empty(self):
        state = IntentActionGapState()
        d = state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        assert len(restored.records) == 0
        assert restored.latest_record is None
        assert restored.skip_count == 0

    def test_roundtrip_with_data(self):
        recorder = make_recorder()
        process_n(recorder, 5)
        recorder.process_action_record("", "p", tick=100)  # skip
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        assert len(restored.records) == 5
        assert restored.latest_record is not None
        assert restored.latest_record.policy_label == "policy_4"
        assert restored.skip_count == 1

    def test_roundtrip_preserves_facets(self):
        """4断面が永続化で保持されること。"""
        recorder = make_recorder()
        recorder.process_action_record(
            "hello world", "greet", tick=42, context_info="ctx data",
        )
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        rec = restored.records[0]
        assert rec.policy_label == "greet"
        assert rec.text_snippet == "hello world"
        assert rec.tick == 42
        assert rec.context_info == "ctx data"

    def test_roundtrip_preserves_record_ids(self):
        recorder = make_recorder()
        records = process_n(recorder, 3)
        original_ids = [r.record_id for r in records]
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        restored_ids = [r.record_id for r in restored.records]
        assert original_ids == restored_ids

    def test_roundtrip_preserves_skip_count(self):
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)
        recorder.process_action_record("", "p", tick=2)
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        assert restored.skip_count == 2

    def test_state_setter(self):
        """state プロパティの setter テスト。"""
        recorder = make_recorder()
        process_n(recorder, 3)
        original_state = recorder.state.to_dict()

        new_recorder = make_recorder()
        new_recorder.state = IntentActionGapState.from_dict(original_state)
        assert len(new_recorder.state.records) == 3

    def test_load_from_partial_dict(self):
        """一部のフィールドだけのdictからもロード可能。"""
        state = IntentActionGapState.from_dict({
            "skip_count": 10,
        })
        assert state.skip_count == 10
        assert len(state.records) == 0

    def test_record_roundtrip(self):
        rec = GapRecord(
            policy_label="test_policy",
            text_snippet="test text",
            tick=42,
            context_info="test context",
        )
        d = rec.to_dict()
        restored = GapRecord.from_dict(d)
        assert restored.record_id == rec.record_id
        assert restored.policy_label == rec.policy_label
        assert restored.text_snippet == rec.text_snippet
        assert restored.tick == rec.tick
        assert restored.context_info == rec.context_info


# =============================================================================
# Test: 安全弁 - ポリシー選択非接続
# =============================================================================

class TestPolicySelectionDisconnection:
    def test_no_bias_output(self):
        """乖離記録からポリシー選択・バイアス計算への直接入力経路がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "bias" not in data
        assert "score" not in data
        assert "weight" not in data

    def test_no_policy_recommendation(self):
        """ポリシー推薦を含まないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        summary = recorder.get_summary()
        assert "recommended_policy" not in summary
        assert "avoid_policy" not in summary
        assert "preferred_policy" not in summary

    def test_no_policy_avoidance(self):
        """「前回乖離したポリシーを回避する」経路がないこと。"""
        recorder = make_recorder()
        # 同じポリシーで複数回記録しても回避リストが生成されない
        for i in range(5):
            recorder.process_action_record(f"text_{i}", "same_policy", tick=i + 1)
        data = recorder.get_enrichment_data()
        assert "avoidance" not in str(data)
        assert "blacklist" not in str(data)

    def test_no_stability_valve_impact(self):
        """安定化弁への直接入力がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "stability" not in data
        assert "valve" not in data

    def test_no_policy_expansion_input(self):
        """ポリシー候補拡張への入力がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "expansion" not in data
        assert "candidate" not in data

    def test_output_is_reference_only(self):
        """出力は参照情報としてのみ流れること。"""
        recorder = make_recorder()
        recorder.process_action_record("text", "p", tick=1)
        data = recorder.get_enrichment_data()
        assert "action" not in data or data.get("action") is None
        assert "command" not in data
        assert "decision" not in data


# =============================================================================
# Test: 安全弁 - パターン抽出禁止
# =============================================================================

class TestPatternExtractionProhibition:
    def test_no_pattern_in_state(self):
        """蓄積からパターン・傾向・統計を抽出しないこと。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        state_dict = recorder.state.to_dict()
        assert "pattern" not in str(state_dict)
        assert "trend" not in str(state_dict)
        assert "statistic" not in str(state_dict)
        assert "frequency" not in str(state_dict)

    def test_no_aggregation(self):
        """蓄積はそのまま蓄積として存在し、集約・要約・傾向化は行わない。"""
        recorder = make_recorder()
        # 同じポリシーを多数回使っても集約されない
        for i in range(10):
            recorder.process_action_record(f"text_{i}", "repeated_policy", tick=i + 1)
        assert len(recorder.state.records) == 10
        # 各記録は個別に保持される
        for rec in recorder.state.records:
            assert rec.policy_label == "repeated_policy"

    def test_no_tendency_classification(self):
        """「この乖離は典型的」「この乖離は異常」といった分類を行わない。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        for rec in recorder.state.records:
            d = rec.to_dict()
            assert "typical" not in str(d)
            assert "atypical" not in str(d)
            assert "anomaly" not in str(d)
            assert "normal" not in str(d)


# =============================================================================
# Test: 安全弁 - 強調禁止
# =============================================================================

class TestEmphasisProhibition:
    def test_enrichment_no_emphasis_words(self):
        """enrichmentテキストに乖離の存在を特別に強調する表現を使わない。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        data = recorder.get_enrichment_data()
        summary = data["summary_text"]
        assert "重要な乖離" not in summary
        assert "注目すべき差異" not in summary
        assert "大きな乖離" not in summary
        assert "深刻" not in summary
        assert "警告" not in summary

    def test_all_entries_same_structure(self):
        """全ての記録を等価に列挙すること。特別扱いする記録がないこと。"""
        recorder = make_recorder(enrichment_recent_count=5)
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]
        # 全エントリのキーセットが同一であること
        key_sets = [set(entry.keys()) for entry in entries]
        for ks in key_sets:
            assert ks == key_sets[0]


# =============================================================================
# Test: 3経路遮断の検証
# =============================================================================

class TestThreePathBlocking:
    def test_gap_to_policy_selection_blocked(self):
        """乖離記録→ポリシー選択経路の遮断。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        data = recorder.get_enrichment_data()
        # ポリシー選好度・回避度が存在しないこと
        assert "preference" not in str(data)
        assert "avoidance" not in str(data)
        assert "score" not in str(data)

    def test_gap_to_action_result_blocked(self):
        """乖離記録→行動-結果観測経路の遮断。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        # 乖離記録に行動-結果観測への影響を示すフィールドがないこと
        for rec in recorder.state.records:
            d = rec.to_dict()
            assert "action_result" not in d
            assert "outcome" not in d

    def test_gap_to_expectation_blocked(self):
        """乖離記録→予期形成経路の遮断。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "expectation" not in data
        assert "prediction" not in data
        assert "forecast" not in data

    def test_no_self_correction_loop(self):
        """自己矯正ループを形成しないこと。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        data = recorder.get_enrichment_data()
        assert "correction" not in str(data)
        assert "adjustment" not in str(data)
        assert "fix" not in str(data)

    def test_no_emotion_pipeline_modification(self):
        """感情パイプラインのパラメータを変更しない。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        state_dict = recorder.state.to_dict()
        assert "emotion" not in state_dict
        assert "mood" not in state_dict
        assert "decay_rate" not in str(state_dict)

    def test_no_threshold_judgment(self):
        """閾値判定を行わないこと。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        state_dict = recorder.state.to_dict()
        assert "threshold" not in str(state_dict)
        assert "trigger" not in str(state_dict)


# =============================================================================
# Test: Enrichment Summary
# =============================================================================

class TestGapSummary:
    def test_summary_waiting(self):
        state = IntentActionGapState()
        summary = get_gap_summary(state)
        assert "待機中" in summary

    def test_summary_with_records(self):
        recorder = make_recorder()
        recorder.process_action_record("hello", "greet", tick=5)
        summary = get_gap_summary(recorder.state)
        assert "記録数=1" in summary
        assert "直近ポリシー=greet" in summary
        assert "tick=5" in summary

    def test_summary_text_length(self):
        recorder = make_recorder()
        recorder.process_action_record("a" * 50, "p", tick=1)
        summary = get_gap_summary(recorder.state)
        assert "直近テキスト長=50" in summary

    def test_summary_with_skips(self):
        recorder = make_recorder()
        recorder.process_action_record("", "p", tick=1)  # skip
        recorder.process_action_record("text", "p", tick=2)
        summary = get_gap_summary(recorder.state)
        assert "スキップ=1" in summary

    def test_summary_no_evaluation_words(self):
        """summaryに評価的な言葉が含まれないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        summary = get_gap_summary(recorder.state)
        assert "良い" not in summary
        assert "悪い" not in summary
        assert "適切" not in summary
        assert "不適切" not in summary
        assert "乖離が大きい" not in summary
        assert "乖離が小さい" not in summary


# =============================================================================
# Test: Factory
# =============================================================================

class TestFactory:
    def test_create_default(self):
        recorder = create_intent_action_gap_recorder()
        assert isinstance(recorder, IntentActionGapRecorder)
        assert recorder._config.max_records == 50

    def test_create_with_config(self):
        cfg = IntentActionGapConfig(max_records=100)
        recorder = create_intent_action_gap_recorder(config=cfg)
        assert recorder._config.max_records == 100

    def test_factory_returns_fresh_state(self):
        recorder = create_intent_action_gap_recorder()
        assert len(recorder.state.records) == 0
        assert recorder.state.latest_record is None
        assert recorder.state.skip_count == 0


# =============================================================================
# Test: Phase 1-2 invariance
# =============================================================================

class TestPhaseInvariance:
    def test_no_emotion_modification(self):
        """感情パイプライン（Phase 1-2）のパラメータを変更しない。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        state = recorder.state
        assert not hasattr(state, "emotion_delta")
        assert not hasattr(state, "mood_change")
        assert not hasattr(state, "decay_rate")
        assert not hasattr(state, "emotion_values")

    def test_no_dynamics_modification(self):
        """ダイナミクスに影響しない。"""
        recorder = make_recorder()
        process_n(recorder, 10)
        state = recorder.state
        assert not hasattr(state, "dynamics")
        assert not hasattr(state, "peak_intensity")


# =============================================================================
# Test: 予期差分参照経路との境界
# =============================================================================

class TestExpectationGapBoundary:
    def test_separate_from_expectation_diff(self):
        """乖離記録と予期差分記録は別の構造であること。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        state_dict = recorder.state.to_dict()
        # 予期差分関連のフィールドが存在しないこと
        assert "expectation_diff" not in state_dict
        assert "expected_change" not in str(state_dict)

    def test_no_influence_on_expectation(self):
        """乖離記録が予期差分記録の生成・参照・蓄積に影響する経路がないこと。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert "expectation" not in data
        assert "expected" not in str(data)


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_max_records_one(self):
        """最大保持数1のケース。"""
        recorder = make_recorder(max_records=1)
        process_n(recorder, 5)
        assert len(recorder.state.records) == 1
        assert recorder.state.records[0].policy_label == "policy_4"

    def test_very_large_max_records(self):
        """非常に大きな上限。"""
        recorder = make_recorder(max_records=10000)
        process_n(recorder, 100)
        assert len(recorder.state.records) == 100

    def test_text_snippet_max_length_one(self):
        """テキスト断面が1文字の場合。"""
        recorder = make_recorder(text_snippet_max_length=1)
        rec = recorder.process_action_record("hello", "p", tick=1)
        assert rec.text_snippet == "h"

    def test_text_snippet_zero_length(self):
        """テキスト断面が0文字の場合（空になる）。"""
        recorder = make_recorder(text_snippet_max_length=0)
        rec = recorder.process_action_record("hello", "p", tick=1)
        assert rec.text_snippet == ""

    def test_receive_after_load(self):
        """ロード後に記録を追加。"""
        recorder = make_recorder()
        process_n(recorder, 3)
        saved = recorder.state.to_dict()

        new_recorder = make_recorder()
        new_recorder.state = IntentActionGapState.from_dict(saved)
        new_recorder.process_action_record("after_load", "pl", tick=100)
        assert len(new_recorder.state.records) == 4

    def test_timestamp_ordering(self):
        """タイムスタンプが時系列的に増加する。"""
        recorder = make_recorder()
        process_n(recorder, 5)
        timestamps = [r.timestamp for r in recorder.state.records]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_unicode_text(self):
        """ユニコードテキストの処理。"""
        recorder = make_recorder()
        rec = recorder.process_action_record("日本語テスト", "p", tick=1)
        assert rec.text_snippet == "日本語テスト"

    def test_special_characters_in_context(self):
        """文脈情報に特殊文字が含まれる場合。"""
        recorder = make_recorder()
        rec = recorder.process_action_record(
            "text", "p", tick=1, context_info="特殊文字: ♪♡★☆"
        )
        assert rec.context_info == "特殊文字: ♪♡★☆"


# =============================================================================
# Test: Integration (統合テスト)
# =============================================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """記録→蓄積→押出→参照の完全ライフサイクル。"""
        recorder = make_recorder(
            max_records=5,
            enrichment_recent_count=2,
            text_snippet_max_length=50,
        )

        # 1. 記録蓄積
        for i in range(8):
            recorder.process_action_record(
                f"response_{i}", f"policy_{i}", tick=i + 1,
                context_info=f"context_{i}",
            )

        # 2. 状態検証
        assert len(recorder.state.records) == 5  # 上限5
        assert recorder.state.skip_count == 0

        # 3. 最新記録
        latest = recorder.get_latest_record()
        assert latest.policy_label == "policy_7"
        assert latest.tick == 8

        # 4. enrichment
        data = recorder.get_enrichment_data()
        assert data["record_count"] == 5
        assert len(data["recent_entries"]) == 2
        assert data["recent_entries"][-1]["policy_label"] == "policy_7"

        # 5. 全記録参照
        all_records = recorder.get_all_records()
        assert len(all_records) == 5
        assert all_records[-1].policy_label == "policy_7"

        # 6. サマリ
        summary = recorder.get_summary()
        assert summary["record_count"] == 5
        assert summary["has_latest"] is True

    def test_save_load_resume(self):
        """セッション間での永続化と再開。"""
        # Session 1
        recorder1 = make_recorder(max_records=10)
        process_n(recorder1, 5)
        recorder1.process_action_record("", "p", tick=100)  # skip
        saved = recorder1.state.to_dict()

        # Session 2
        recorder2 = make_recorder(max_records=10)
        recorder2.state = IntentActionGapState.from_dict(saved)

        # 状態が復元されていること
        assert len(recorder2.state.records) == 5
        assert recorder2.state.skip_count == 1
        assert recorder2.state.latest_record is not None

        # 追加記録が可能
        recorder2.process_action_record("new_response", "new_policy", tick=200)
        assert len(recorder2.state.records) == 6

    def test_skip_interspersed_lifecycle(self):
        """スキップと正常記録が混在するライフサイクル。"""
        recorder = make_recorder(max_records=5)
        recorder.process_action_record("t1", "p1", tick=1)
        recorder.process_action_record("", "p2", tick=2)    # skip
        recorder.process_action_record("t3", "p3", tick=3)
        recorder.process_action_record("t4", "", tick=4)    # skip
        recorder.process_action_record("t5", "p5", tick=5)
        recorder.process_action_record("t6", "p6", tick=6)
        recorder.process_action_record("t7", "p7", tick=7)
        recorder.process_action_record("t8", "p8", tick=8)  # triggers pushout

        assert len(recorder.state.records) == 5
        assert recorder.state.skip_count == 2
        assert recorder.state.latest_record.policy_label == "p8"

    def test_rapid_succession(self):
        """高速連続処理での安定性。"""
        recorder = make_recorder(max_records=100)
        for i in range(200):
            recorder.process_action_record(f"msg_{i}", f"p_{i}", tick=i)
        assert len(recorder.state.records) == 100

    def test_enrichment_after_pushout(self):
        """押し出し後もenrichmentが正常に機能する。"""
        recorder = make_recorder(max_records=3, enrichment_recent_count=2)
        process_n(recorder, 5)
        data = recorder.get_enrichment_data()
        assert len(data["recent_entries"]) == 2
        assert data["record_count"] == 3

    def test_all_records_after_pushout(self):
        """押し出し後もget_all_recordsが正常に機能する。"""
        recorder = make_recorder(max_records=5)
        process_n(recorder, 10)
        all_records = recorder.get_all_records()
        assert len(all_records) == 5
        # 最新5件のみ
        policies = [r.policy_label for r in all_records]
        assert policies == ["policy_5", "policy_6", "policy_7", "policy_8", "policy_9"]

    def test_context_info_propagation(self):
        """文脈断面が正しく伝播・蓄積されること。"""
        recorder = make_recorder()
        recorder.process_action_record("t1", "p1", tick=1, context_info="ctx1")
        recorder.process_action_record("t2", "p2", tick=2, context_info="")
        recorder.process_action_record("t3", "p3", tick=3, context_info="ctx3")

        records = recorder.get_all_records()
        assert records[0].context_info == "ctx1"
        assert records[1].context_info == ""
        assert records[2].context_info == "ctx3"

    def test_text_snippet_truncation_in_lifecycle(self):
        """テキスト断面の切り詰めがライフサイクルを通じて正しく動作すること。"""
        recorder = make_recorder(text_snippet_max_length=10)
        recorder.process_action_record("a" * 100, "p", tick=1)
        rec = recorder.get_latest_record()
        assert len(rec.text_snippet) == 10

        # save/load後も切り詰め状態が維持されること
        d = recorder.state.to_dict()
        restored = IntentActionGapState.from_dict(d)
        assert len(restored.records[0].text_snippet) == 10
