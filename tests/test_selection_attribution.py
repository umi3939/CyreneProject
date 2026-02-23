"""
tests/test_selection_attribution.py - 選択帰属モジュールのテスト

テスト対象: psyche/selection_attribution.py
設計書: design_selection_attribution.md
"""

import time
import pytest

from psyche.selection_attribution import (
    SelectionRecord,
    SelectionAttributionState,
    SelectionAttributionConfig,
    SelectionAttributionRecorder,
    create_selection_attribution_recorder,
    get_selection_attribution_summary,
)


# =============================================================================
# SelectionRecord tests
# =============================================================================

class TestSelectionRecord:
    """選択記録のデータ構造テスト。"""

    def test_default_creation(self):
        """デフォルト値で記録が生成される。"""
        rec = SelectionRecord()
        assert rec.record_id != ""
        assert rec.selected_policy_label == ""
        assert rec.candidate_labels == []
        assert rec.candidate_count == 0
        assert rec.tick == 0
        assert rec.timestamp > 0

    def test_creation_with_values(self):
        """値を指定して記録が生成される。"""
        rec = SelectionRecord(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B", "policy_C"],
            candidate_count=3,
            tick=10,
        )
        assert rec.selected_policy_label == "policy_A"
        assert rec.candidate_labels == ["policy_A", "policy_B", "policy_C"]
        assert rec.candidate_count == 3
        assert rec.tick == 10

    def test_to_dict(self):
        """to_dictで辞書に変換できる。"""
        rec = SelectionRecord(
            selected_policy_label="policy_X",
            candidate_labels=["policy_X", "policy_Y"],
            candidate_count=2,
            tick=5,
        )
        d = rec.to_dict()
        assert d["selected_policy_label"] == "policy_X"
        assert d["candidate_labels"] == ["policy_X", "policy_Y"]
        assert d["candidate_count"] == 2
        assert d["tick"] == 5
        assert "record_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        """from_dictで辞書から復元できる。"""
        d = {
            "record_id": "abc123",
            "selected_policy_label": "policy_Z",
            "candidate_labels": ["policy_Z", "policy_W"],
            "candidate_count": 2,
            "tick": 7,
            "timestamp": 1234567890.0,
        }
        rec = SelectionRecord.from_dict(d)
        assert rec.record_id == "abc123"
        assert rec.selected_policy_label == "policy_Z"
        assert rec.candidate_labels == ["policy_Z", "policy_W"]
        assert rec.candidate_count == 2
        assert rec.tick == 7
        assert rec.timestamp == 1234567890.0

    def test_from_dict_missing_fields(self):
        """from_dictで一部フィールドが欠落しても復元できる。"""
        d = {}
        rec = SelectionRecord.from_dict(d)
        assert rec.record_id != ""
        assert rec.selected_policy_label == ""
        assert rec.candidate_labels == []
        assert rec.candidate_count == 0
        assert rec.tick == 0

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict -> from_dict のラウンドトリップが一致する。"""
        original = SelectionRecord(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B", "policy_C"],
            candidate_count=3,
            tick=42,
        )
        d = original.to_dict()
        restored = SelectionRecord.from_dict(d)
        assert restored.record_id == original.record_id
        assert restored.selected_policy_label == original.selected_policy_label
        assert restored.candidate_labels == original.candidate_labels
        assert restored.candidate_count == original.candidate_count
        assert restored.tick == original.tick
        assert restored.timestamp == original.timestamp

    def test_no_weight_or_score_attributes(self):
        """記録に重み・スコア・優先度の属性が存在しないことを確認（全記録等価）。"""
        rec = SelectionRecord(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A"],
            candidate_count=1,
            tick=1,
        )
        d = rec.to_dict()
        # 禁止される評価的属性が存在しないこと
        assert "weight" not in d
        assert "score" not in d
        assert "priority" not in d
        assert "importance" not in d
        assert "rank" not in d


# =============================================================================
# SelectionAttributionState tests
# =============================================================================

class TestSelectionAttributionState:
    """選択帰属状態のテスト。"""

    def test_default_state(self):
        """デフォルト状態が正しい。"""
        st = SelectionAttributionState()
        assert st.records == []
        assert st.latest_record is None
        assert st.total_records_received == 0
        assert st.total_records_pushed_out == 0

    def test_to_dict_empty(self):
        """空状態のto_dict。"""
        st = SelectionAttributionState()
        d = st.to_dict()
        assert d["records"] == []
        assert d["latest_record"] is None
        assert d["total_records_received"] == 0
        assert d["total_records_pushed_out"] == 0

    def test_to_dict_with_records(self):
        """記録ありの状態のto_dict。"""
        rec = SelectionRecord(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B"],
            candidate_count=2,
            tick=1,
        )
        st = SelectionAttributionState(
            records=[rec],
            latest_record=rec,
            total_records_received=1,
            total_records_pushed_out=0,
        )
        d = st.to_dict()
        assert len(d["records"]) == 1
        assert d["latest_record"] is not None
        assert d["total_records_received"] == 1

    def test_from_dict_empty(self):
        """空辞書からの復元。"""
        st = SelectionAttributionState.from_dict({})
        assert st.records == []
        assert st.latest_record is None
        assert st.total_records_received == 0
        assert st.total_records_pushed_out == 0

    def test_to_dict_from_dict_roundtrip(self):
        """状態のto_dict -> from_dictラウンドトリップ。"""
        rec = SelectionRecord(
            selected_policy_label="p1",
            candidate_labels=["p1", "p2"],
            candidate_count=2,
            tick=5,
        )
        original = SelectionAttributionState(
            records=[rec],
            latest_record=rec,
            total_records_received=10,
            total_records_pushed_out=3,
        )
        d = original.to_dict()
        restored = SelectionAttributionState.from_dict(d)
        assert len(restored.records) == 1
        assert restored.records[0].selected_policy_label == "p1"
        assert restored.latest_record is not None
        assert restored.latest_record.selected_policy_label == "p1"
        assert restored.total_records_received == 10
        assert restored.total_records_pushed_out == 3


# =============================================================================
# SelectionAttributionConfig tests
# =============================================================================

class TestSelectionAttributionConfig:
    """設定のテスト。"""

    def test_default_config(self):
        """デフォルト設定値の確認。"""
        cfg = SelectionAttributionConfig()
        assert cfg.max_records == 50
        assert cfg.max_candidate_labels == 20
        assert cfg.enrichment_recent_count == 5
        assert cfg.reference_history_count == 20

    def test_custom_config(self):
        """カスタム設定値の確認。"""
        cfg = SelectionAttributionConfig(
            max_records=10,
            max_candidate_labels=5,
            enrichment_recent_count=3,
            reference_history_count=8,
        )
        assert cfg.max_records == 10
        assert cfg.max_candidate_labels == 5
        assert cfg.enrichment_recent_count == 3
        assert cfg.reference_history_count == 8


# =============================================================================
# SelectionAttributionRecorder tests
# =============================================================================

class TestSelectionAttributionRecorder:
    """選択帰属レコーダーのテスト。"""

    def test_creation_default(self):
        """デフォルト設定でレコーダーが作成される。"""
        recorder = SelectionAttributionRecorder()
        assert recorder.state.total_records_received == 0
        assert recorder.state.records == []
        assert recorder.state.latest_record is None

    def test_creation_with_config(self):
        """カスタム設定でレコーダーが作成される。"""
        cfg = SelectionAttributionConfig(max_records=5)
        recorder = SelectionAttributionRecorder(config=cfg)
        assert recorder._config.max_records == 5

    def test_record_selection_basic(self):
        """基本的な記録構成と蓄積。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B", "policy_C"],
            tick=1,
        )
        assert rec.selected_policy_label == "policy_A"
        assert rec.candidate_labels == ["policy_A", "policy_B", "policy_C"]
        assert rec.candidate_count == 3
        assert rec.tick == 1
        assert recorder.state.total_records_received == 1
        assert len(recorder.state.records) == 1
        assert recorder.state.latest_record is rec

    def test_record_selection_multiple(self):
        """複数回の記録蓄積。"""
        recorder = SelectionAttributionRecorder()
        rec1 = recorder.record_selection(
            selected_policy_label="p1",
            candidate_labels=["p1", "p2"],
            tick=1,
        )
        rec2 = recorder.record_selection(
            selected_policy_label="p2",
            candidate_labels=["p1", "p2", "p3"],
            tick=2,
        )
        rec3 = recorder.record_selection(
            selected_policy_label="p3",
            candidate_labels=["p3"],
            tick=3,
        )
        assert recorder.state.total_records_received == 3
        assert len(recorder.state.records) == 3
        assert recorder.state.latest_record is rec3
        assert recorder.state.records[0] is rec1
        assert recorder.state.records[1] is rec2
        assert recorder.state.records[2] is rec3

    def test_record_selection_empty_label(self):
        """空文字列のポリシーラベルでも記録される。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            selected_policy_label="",
            candidate_labels=[],
            tick=0,
        )
        assert rec.selected_policy_label == ""
        assert rec.candidate_labels == []
        assert rec.candidate_count == 0
        assert recorder.state.total_records_received == 1

    def test_record_selection_unique_ids(self):
        """各記録のIDが一意である。"""
        recorder = SelectionAttributionRecorder()
        ids = set()
        for i in range(20):
            rec = recorder.record_selection(
                selected_policy_label=f"p{i}",
                candidate_labels=[f"p{i}"],
                tick=i,
            )
            ids.add(rec.record_id)
        assert len(ids) == 20

    def test_record_selection_timestamps_monotonic(self):
        """タイムスタンプが単調増加する。"""
        recorder = SelectionAttributionRecorder()
        timestamps = []
        for i in range(5):
            rec = recorder.record_selection(
                selected_policy_label=f"p{i}",
                candidate_labels=[f"p{i}"],
                tick=i,
            )
            timestamps.append(rec.timestamp)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    # ─── 上限到達時の最古押し出し ─────────────────────────────

    def test_pushout_at_limit(self):
        """上限到達時に最古の記録から押し出される。"""
        cfg = SelectionAttributionConfig(max_records=3)
        recorder = SelectionAttributionRecorder(config=cfg)

        rec1 = recorder.record_selection("p1", ["p1"], tick=1)
        rec2 = recorder.record_selection("p2", ["p2"], tick=2)
        rec3 = recorder.record_selection("p3", ["p3"], tick=3)

        assert len(recorder.state.records) == 3
        assert recorder.state.total_records_pushed_out == 0

        # 4件目で最古が押し出される
        rec4 = recorder.record_selection("p4", ["p4"], tick=4)
        assert len(recorder.state.records) == 3
        assert recorder.state.total_records_pushed_out == 1
        assert recorder.state.records[0].selected_policy_label == "p2"
        assert recorder.state.records[1].selected_policy_label == "p3"
        assert recorder.state.records[2].selected_policy_label == "p4"

    def test_pushout_multiple(self):
        """上限超過時に複数の最古記録が押し出される。"""
        cfg = SelectionAttributionConfig(max_records=3)
        recorder = SelectionAttributionRecorder(config=cfg)

        for i in range(10):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        assert len(recorder.state.records) == 3
        assert recorder.state.total_records_received == 10
        assert recorder.state.total_records_pushed_out == 7
        # 直近3件が残る
        assert recorder.state.records[0].selected_policy_label == "p7"
        assert recorder.state.records[1].selected_policy_label == "p8"
        assert recorder.state.records[2].selected_policy_label == "p9"

    def test_pushout_is_only_deletion_path(self):
        """押し出しが唯一の消失経路であることの確認。
        特定の記録を選択的に消去するメソッドが存在しない。"""
        recorder = SelectionAttributionRecorder()
        # public methods にdelete/remove系が存在しないことを確認
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "delete" not in method_name.lower()
            assert "remove" not in method_name.lower()
            assert "clear" not in method_name.lower()
            assert "purge" not in method_name.lower()

    def test_no_protection_or_pinning(self):
        """特定の記録を永続的に保持する「保護」「固定」メカニズムが存在しない。"""
        rec = SelectionRecord(
            selected_policy_label="important_policy",
            candidate_labels=["important_policy"],
            candidate_count=1,
            tick=1,
        )
        d = rec.to_dict()
        # 保護・固定関連の属性がないこと
        assert "protected" not in d
        assert "pinned" not in d
        assert "fixed" not in d
        assert "important" not in d

    # ─── 候補群ラベルの上限切り詰め ──────────────────────────

    def test_candidate_labels_trimming(self):
        """候補群ラベルが上限を超えた場合、先頭から切り詰められる。"""
        cfg = SelectionAttributionConfig(max_candidate_labels=3)
        recorder = SelectionAttributionRecorder(config=cfg)

        labels = ["p1", "p2", "p3", "p4", "p5"]
        rec = recorder.record_selection("p1", labels, tick=1)

        assert len(rec.candidate_labels) == 3
        assert rec.candidate_labels == ["p1", "p2", "p3"]
        # candidate_count は元の候補数を保持
        assert rec.candidate_count == 5

    def test_candidate_labels_within_limit(self):
        """候補群ラベルが上限以内の場合、そのまま保持される。"""
        cfg = SelectionAttributionConfig(max_candidate_labels=10)
        recorder = SelectionAttributionRecorder(config=cfg)

        labels = ["p1", "p2", "p3"]
        rec = recorder.record_selection("p1", labels, tick=1)

        assert rec.candidate_labels == ["p1", "p2", "p3"]
        assert rec.candidate_count == 3

    # ─── 全記録等価の確認 ────────────────────────────────────

    def test_all_records_equal_weight(self):
        """全記録が等価であること（重み・スコア・優先度の属性なし）。"""
        recorder = SelectionAttributionRecorder()
        for i in range(5):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        for rec in recorder.state.records:
            d = rec.to_dict()
            assert "weight" not in d
            assert "score" not in d
            assert "priority" not in d
            assert "importance" not in d
            assert "rank" not in d
            assert "frequency" not in d

    # ─── get_latest_record ──────────────────────────────────

    def test_get_latest_record_empty(self):
        """記録なしの場合、get_latest_recordはNoneを返す。"""
        recorder = SelectionAttributionRecorder()
        assert recorder.get_latest_record() is None

    def test_get_latest_record_after_recording(self):
        """記録後、直近の記録が返される。"""
        recorder = SelectionAttributionRecorder()
        rec1 = recorder.record_selection("p1", ["p1"], tick=1)
        assert recorder.get_latest_record() is rec1

        rec2 = recorder.record_selection("p2", ["p2"], tick=2)
        assert recorder.get_latest_record() is rec2

    # ─── enrichment ──────────────────────────────────────────

    def test_get_enrichment_data_empty(self):
        """記録なしの場合のenrichmentデータ。"""
        recorder = SelectionAttributionRecorder()
        data = recorder.get_enrichment_data()
        assert data["record_count"] == 0
        assert data["recent_entries"] == []
        assert "待機中" in data["summary_text"]

    def test_get_enrichment_data_with_records(self):
        """記録ありの場合のenrichmentデータ。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("p1", ["p1", "p2"], tick=1)
        recorder.record_selection("p2", ["p1", "p2", "p3"], tick=2)

        data = recorder.get_enrichment_data()
        assert data["record_count"] == 2
        assert len(data["recent_entries"]) == 2
        assert data["recent_entries"][0]["selected_policy_label"] == "p1"
        assert data["recent_entries"][1]["selected_policy_label"] == "p2"

    def test_get_enrichment_data_respects_recent_count(self):
        """enrichmentは直近N件のみを返す。"""
        cfg = SelectionAttributionConfig(enrichment_recent_count=2)
        recorder = SelectionAttributionRecorder(config=cfg)
        for i in range(5):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        data = recorder.get_enrichment_data()
        assert len(data["recent_entries"]) == 2
        assert data["recent_entries"][0]["selected_policy_label"] == "p3"
        assert data["recent_entries"][1]["selected_policy_label"] == "p4"

    def test_enrichment_entries_contain_required_fields(self):
        """enrichmentエントリが必要なフィールドを含む。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("pA", ["pA", "pB"], tick=10)

        data = recorder.get_enrichment_data()
        entry = data["recent_entries"][0]
        assert "selected_policy_label" in entry
        assert "candidate_labels" in entry
        assert "candidate_count" in entry
        assert "tick" in entry

    def test_enrichment_no_emphasis(self):
        """enrichmentに強調語が含まれないこと。"""
        recorder = SelectionAttributionRecorder()
        for i in range(10):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        data = recorder.get_enrichment_data()
        summary = data["summary_text"]
        # 禁止される強調語が含まれないことを確認
        forbidden_words = ["注目", "重要", "特徴的", "顕著", "notable", "important"]
        for word in forbidden_words:
            assert word not in summary

    def test_enrichment_equal_listing(self):
        """enrichmentが等価列挙であること（順序変更・選別なし）。"""
        cfg = SelectionAttributionConfig(enrichment_recent_count=3)
        recorder = SelectionAttributionRecorder(config=cfg)
        recorder.record_selection("pX", ["pX"], tick=1)
        recorder.record_selection("pY", ["pY"], tick=2)
        recorder.record_selection("pZ", ["pZ"], tick=3)

        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]
        # 時系列順（古い→新しい）で等価に列挙
        assert entries[0]["selected_policy_label"] == "pX"
        assert entries[1]["selected_policy_label"] == "pY"
        assert entries[2]["selected_policy_label"] == "pZ"

    # ─── 内省系参照経路 ──────────────────────────────────────

    def test_get_all_records_empty(self):
        """記録なしの場合、空リストが返される。"""
        recorder = SelectionAttributionRecorder()
        assert recorder.get_all_records() == []

    def test_get_all_records_returns_copy(self):
        """get_all_recordsはリストのコピーを返す。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("p1", ["p1"], tick=1)

        records1 = recorder.get_all_records()
        records2 = recorder.get_all_records()
        assert records1 is not records2
        assert records1 is not recorder.state.records

    def test_get_all_records_no_side_effect(self):
        """get_all_recordsの呼び出しが内部状態に影響しない。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("p1", ["p1"], tick=1)

        state_before = recorder.state.total_records_received
        records_before = len(recorder.state.records)

        _ = recorder.get_all_records()
        _ = recorder.get_all_records()
        _ = recorder.get_all_records()

        assert recorder.state.total_records_received == state_before
        assert len(recorder.state.records) == records_before

    def test_get_reference_history_empty(self):
        """記録なしの場合の参照履歴。"""
        recorder = SelectionAttributionRecorder()
        assert recorder.get_reference_history() == []

    def test_get_reference_history_limit(self):
        """参照履歴は設定された上限件数まで返す。"""
        cfg = SelectionAttributionConfig(reference_history_count=3)
        recorder = SelectionAttributionRecorder(config=cfg)
        for i in range(10):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        history = recorder.get_reference_history()
        assert len(history) == 3
        assert history[0].selected_policy_label == "p7"
        assert history[1].selected_policy_label == "p8"
        assert history[2].selected_policy_label == "p9"

    def test_get_reference_history_returns_copy(self):
        """参照履歴はコピーを返す。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("p1", ["p1"], tick=1)

        h1 = recorder.get_reference_history()
        h2 = recorder.get_reference_history()
        assert h1 is not h2

    # ─── save/load ラウンドトリップ ───────────────────────────

    def test_save_load_roundtrip_empty(self):
        """空状態のsave/loadラウンドトリップ。"""
        recorder = SelectionAttributionRecorder()
        d = recorder.state.to_dict()
        restored_state = SelectionAttributionState.from_dict(d)

        assert len(restored_state.records) == 0
        assert restored_state.latest_record is None
        assert restored_state.total_records_received == 0
        assert restored_state.total_records_pushed_out == 0

    def test_save_load_roundtrip_with_records(self):
        """記録ありのsave/loadラウンドトリップ。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("pA", ["pA", "pB", "pC"], tick=10)
        recorder.record_selection("pB", ["pA", "pB"], tick=11)
        recorder.record_selection("pC", ["pC"], tick=12)

        d = recorder.state.to_dict()
        restored_state = SelectionAttributionState.from_dict(d)

        assert len(restored_state.records) == 3
        assert restored_state.records[0].selected_policy_label == "pA"
        assert restored_state.records[0].candidate_labels == ["pA", "pB", "pC"]
        assert restored_state.records[0].candidate_count == 3
        assert restored_state.records[1].selected_policy_label == "pB"
        assert restored_state.records[2].selected_policy_label == "pC"
        assert restored_state.latest_record is not None
        assert restored_state.latest_record.selected_policy_label == "pC"
        assert restored_state.total_records_received == 3
        assert restored_state.total_records_pushed_out == 0

    def test_save_load_roundtrip_with_pushout(self):
        """押し出し後のsave/loadラウンドトリップ。"""
        cfg = SelectionAttributionConfig(max_records=3)
        recorder = SelectionAttributionRecorder(config=cfg)
        for i in range(5):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        d = recorder.state.to_dict()
        restored_state = SelectionAttributionState.from_dict(d)

        assert len(restored_state.records) == 3
        assert restored_state.total_records_received == 5
        assert restored_state.total_records_pushed_out == 2
        assert restored_state.records[0].selected_policy_label == "p2"

    def test_save_load_state_setter(self):
        """state setterによる状態復元。"""
        recorder1 = SelectionAttributionRecorder()
        recorder1.record_selection("pA", ["pA", "pB"], tick=1)
        recorder1.record_selection("pB", ["pA", "pB"], tick=2)

        d = recorder1.state.to_dict()
        restored_state = SelectionAttributionState.from_dict(d)

        recorder2 = SelectionAttributionRecorder()
        recorder2.state = restored_state

        assert len(recorder2.state.records) == 2
        assert recorder2.state.latest_record.selected_policy_label == "pB"
        assert recorder2.state.total_records_received == 2

    # ─── サマリ ──────────────────────────────────────────────

    def test_get_summary_empty(self):
        """空状態のサマリ。"""
        recorder = SelectionAttributionRecorder()
        summary = recorder.get_summary()
        assert summary["total_records_received"] == 0
        assert summary["current_record_count"] == 0
        assert summary["total_pushed_out"] == 0
        assert summary["has_latest"] is False
        assert summary["latest_tick"] == 0
        assert summary["latest_selected"] == ""

    def test_get_summary_with_records(self):
        """記録ありのサマリ。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("pX", ["pX", "pY"], tick=5)

        summary = recorder.get_summary()
        assert summary["total_records_received"] == 1
        assert summary["current_record_count"] == 1
        assert summary["has_latest"] is True
        assert summary["latest_tick"] == 5
        assert summary["latest_selected"] == "pX"

    # ─── エッジケース ────────────────────────────────────────

    def test_single_candidate(self):
        """候補が1つだけの場合。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection("only_one", ["only_one"], tick=1)
        assert rec.candidate_count == 1
        assert rec.candidate_labels == ["only_one"]

    def test_empty_candidate_list(self):
        """候補リストが空の場合。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection("selected", [], tick=1)
        assert rec.candidate_count == 0
        assert rec.candidate_labels == []

    def test_many_candidates(self):
        """多数の候補がある場合。"""
        recorder = SelectionAttributionRecorder()
        labels = [f"p{i}" for i in range(100)]
        rec = recorder.record_selection("p0", labels, tick=1)
        # デフォルトmax_candidate_labels=20でトリミングされる
        assert rec.candidate_count == 100
        assert len(rec.candidate_labels) == 20

    def test_duplicate_candidate_labels(self):
        """重複する候補ラベルがあっても記録される（フィルタリングしない）。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            "pA", ["pA", "pA", "pB", "pB"], tick=1,
        )
        assert rec.candidate_labels == ["pA", "pA", "pB", "pB"]
        assert rec.candidate_count == 4

    def test_rapid_successive_recordings(self):
        """連続的な高速記録。"""
        recorder = SelectionAttributionRecorder()
        for i in range(100):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)
        assert recorder.state.total_records_received == 100
        assert len(recorder.state.records) == 50  # デフォルト上限

    def test_max_records_one(self):
        """上限が1の場合、常に最新の1件のみ保持。"""
        cfg = SelectionAttributionConfig(max_records=1)
        recorder = SelectionAttributionRecorder(config=cfg)

        recorder.record_selection("p1", ["p1"], tick=1)
        assert len(recorder.state.records) == 1

        recorder.record_selection("p2", ["p2"], tick=2)
        assert len(recorder.state.records) == 1
        assert recorder.state.records[0].selected_policy_label == "p2"
        assert recorder.state.total_records_pushed_out == 1


# =============================================================================
# get_selection_attribution_summary tests
# =============================================================================

class TestGetSelectionAttributionSummary:
    """サマリ関数のテスト。"""

    def test_empty_state(self):
        """空状態で待機中メッセージが返される。"""
        st = SelectionAttributionState()
        text = get_selection_attribution_summary(st)
        assert "待機中" in text

    def test_with_records(self):
        """記録ありの状態でサマリが返される。"""
        rec = SelectionRecord(
            selected_policy_label="pA",
            candidate_labels=["pA", "pB"],
            candidate_count=2,
            tick=5,
        )
        st = SelectionAttributionState(
            records=[rec],
            latest_record=rec,
            total_records_received=1,
        )
        text = get_selection_attribution_summary(st)
        assert "記録数=1" in text
        assert "直近選択=pA" in text
        assert "候補数=2" in text
        assert "tick=5" in text

    def test_no_frequency_or_pattern_info(self):
        """サマリに頻度・パターン情報が含まれないこと。"""
        rec = SelectionRecord(
            selected_policy_label="pA",
            candidate_labels=["pA"],
            candidate_count=1,
            tick=1,
        )
        st = SelectionAttributionState(
            records=[rec],
            latest_record=rec,
            total_records_received=10,
        )
        text = get_selection_attribution_summary(st)
        # 禁止される情報が含まれないことを確認
        assert "頻度" not in text
        assert "パターン" not in text
        assert "傾向" not in text
        assert "偏り" not in text
        assert "選好" not in text

    def test_pushout_info(self):
        """押し出し情報が含まれる。"""
        st = SelectionAttributionState(
            records=[],
            latest_record=SelectionRecord(
                selected_policy_label="pA",
                candidate_labels=["pA"],
                candidate_count=1,
                tick=1,
            ),
            total_records_received=10,
            total_records_pushed_out=5,
        )
        text = get_selection_attribution_summary(st)
        assert "押出累計=5" in text


# =============================================================================
# Factory function tests
# =============================================================================

class TestFactory:
    """ファクトリ関数のテスト。"""

    def test_create_default(self):
        """デフォルトのファクトリ。"""
        recorder = create_selection_attribution_recorder()
        assert isinstance(recorder, SelectionAttributionRecorder)
        assert recorder.state.total_records_received == 0

    def test_create_with_config(self):
        """設定付きのファクトリ。"""
        cfg = SelectionAttributionConfig(max_records=10)
        recorder = create_selection_attribution_recorder(config=cfg)
        assert recorder._config.max_records == 10


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    """設計書で定義された5つの安全弁のテスト。"""

    def test_safety_valve_1_all_records_equal(self):
        """安全弁1: 全記録等価の保証。記録に評価的属性が存在しない。"""
        recorder = SelectionAttributionRecorder()
        for i in range(10):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        for rec in recorder.state.records:
            d = rec.to_dict()
            # 評価的属性が存在しないこと
            # bias_source_labels は名前のみの事実記録であり評価的属性ではない
            for key in d:
                assert key in {
                    "record_id", "selected_policy_label",
                    "candidate_labels", "candidate_count",
                    "tick", "timestamp", "bias_source_labels",
                }

    def test_safety_valve_2_no_pattern_extraction(self):
        """安全弁2: パターン抽出の禁止。
        レコーダーにパターン抽出メソッドが存在しない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "pattern" not in method_name.lower()
            assert "frequency" not in method_name.lower()
            assert "tendency" not in method_name.lower()
            assert "statistics" not in method_name.lower()
            assert "aggregate" not in method_name.lower()

    def test_safety_valve_3_no_feedback_path(self):
        """安全弁3: 方針選択処理への経路遮断の不変性。
        記録・取得系のメソッドのみが存在し、判断系への接続メソッドがない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            # 判断系への接続を示唆するメソッド名がないことを確認
            assert "bias" not in method_name.lower()
            assert "influence" not in method_name.lower()
            assert "modify" not in method_name.lower()
            assert "adjust" not in method_name.lower()
            assert "feedback" not in method_name.lower()

    def test_safety_valve_4_enrichment_equal_listing(self):
        """安全弁4: enrichment内での等価列挙の維持。
        特定の記録を他の記録より目立たせる加工がない。"""
        cfg = SelectionAttributionConfig(enrichment_recent_count=5)
        recorder = SelectionAttributionRecorder(config=cfg)
        for i in range(5):
            recorder.record_selection(f"p{i}", [f"p{i}"], tick=i)

        data = recorder.get_enrichment_data()
        entries = data["recent_entries"]

        # 全エントリが同じフィールド構成
        for entry in entries:
            assert set(entry.keys()) == {
                "selected_policy_label", "candidate_labels",
                "candidate_count", "tick",
            }

        # 順序がtime order（挿入順）であること
        for i in range(len(entries) - 1):
            assert entries[i]["tick"] <= entries[i + 1]["tick"]

    def test_safety_valve_5_no_protection_mechanism(self):
        """安全弁5: 蓄積上限による自然な入れ替わりの保証。
        すべての記録は上限到達時に最古から押し出される。
        「保護」「固定」「重要フラグ」等の仕組みが存在しない。"""
        cfg = SelectionAttributionConfig(max_records=3)
        recorder = SelectionAttributionRecorder(config=cfg)

        # 最初の記録も上限到達時には押し出される
        first_rec = recorder.record_selection("first", ["first"], tick=1)
        recorder.record_selection("second", ["second"], tick=2)
        recorder.record_selection("third", ["third"], tick=3)
        recorder.record_selection("fourth", ["fourth"], tick=4)

        # 最初の記録が確実に押し出されている
        remaining_labels = [r.selected_policy_label for r in recorder.state.records]
        assert "first" not in remaining_labels
        assert len(remaining_labels) == 3


# =============================================================================
# 経路遮断テスト
# =============================================================================

class TestPathBlocking:
    """設計書で定義された5つの経路遮断のテスト。"""

    def test_no_candidate_generation_input(self):
        """経路遮断1: 選択記録が候補生成処理の入力に含まれない。
        レコーダーが候補生成に関連するメソッドを持たない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "generate" not in method_name.lower()
            assert "candidate_gen" not in method_name.lower()

    def test_no_bias_calculation_input(self):
        """経路遮断2: 選択記録がバイアス計算処理の入力に含まれない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "bias" not in method_name.lower()
            assert "compute" not in method_name.lower()

    def test_no_stability_valve_input(self):
        """経路遮断3: 選択記録が安定化弁の入力に含まれない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "stability" not in method_name.lower()
            assert "valve" not in method_name.lower()

    def test_no_emotion_processing_input(self):
        """経路遮断4: 選択記録が感情更新処理の入力に含まれない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "emotion" not in method_name.lower()
            assert "react" not in method_name.lower()

    def test_no_responsibility_calculation_input(self):
        """経路遮断5: 選択記録が責任の記録・評価処理の入力に含まれない。"""
        recorder = SelectionAttributionRecorder()
        public_methods = [m for m in dir(recorder) if not m.startswith('_')]
        for method_name in public_methods:
            assert "responsibility" not in method_name.lower()

    def test_output_is_read_only(self):
        """出力はすべて読み取り専用である。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("pA", ["pA", "pB"], tick=1)

        # get_all_recordsの返り値を変更しても内部状態に影響しない
        records = recorder.get_all_records()
        original_count = len(recorder.state.records)
        records.clear()
        assert len(recorder.state.records) == original_count

        # get_reference_historyの返り値を変更しても内部状態に影響しない
        history = recorder.get_reference_history()
        history.clear()
        assert len(recorder.state.records) == original_count


# =============================================================================
# bias_source_labels テスト
# =============================================================================

class TestBiasSourceLabels:
    """bias_source_labels フィールドのテスト。"""

    def test_record_selection_without_bias_labels(self):
        """bias_source_labelsを渡さない場合、空リストになる（後方互換性）。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B"],
            tick=1,
        )
        assert rec.bias_source_labels == []

    def test_record_selection_with_bias_labels(self):
        """bias_source_labelsを渡した場合、正しく記録される。"""
        recorder = SelectionAttributionRecorder()
        labels = ["decision_bias", "context_sensitivity", "stability_valve"]
        rec = recorder.record_selection(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A", "policy_B"],
            tick=1,
            bias_source_labels=labels,
        )
        assert rec.bias_source_labels == labels

    def test_record_selection_with_empty_bias_labels(self):
        """空リストを明示的に渡した場合。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A"],
            tick=1,
            bias_source_labels=[],
        )
        assert rec.bias_source_labels == []

    def test_record_selection_with_none_bias_labels(self):
        """Noneを明示的に渡した場合、空リストになる。"""
        recorder = SelectionAttributionRecorder()
        rec = recorder.record_selection(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A"],
            tick=1,
            bias_source_labels=None,
        )
        assert rec.bias_source_labels == []

    def test_bias_labels_in_to_dict(self):
        """to_dictにbias_source_labelsが含まれる。"""
        rec = SelectionRecord(
            selected_policy_label="policy_A",
            candidate_labels=["policy_A"],
            candidate_count=1,
            tick=1,
            bias_source_labels=["decision_bias", "value_orientation"],
        )
        d = rec.to_dict()
        assert "bias_source_labels" in d
        assert d["bias_source_labels"] == ["decision_bias", "value_orientation"]

    def test_bias_labels_from_dict(self):
        """from_dictでbias_source_labelsが復元される。"""
        d = {
            "record_id": "test123",
            "selected_policy_label": "policy_A",
            "candidate_labels": ["policy_A"],
            "candidate_count": 1,
            "tick": 1,
            "timestamp": 1234567890.0,
            "bias_source_labels": ["stability_valve", "scoring_fluctuation"],
        }
        rec = SelectionRecord.from_dict(d)
        assert rec.bias_source_labels == ["stability_valve", "scoring_fluctuation"]

    def test_bias_labels_from_dict_missing(self):
        """from_dictでbias_source_labelsが欠落しても空リストで復元される（後方互換性）。"""
        d = {
            "record_id": "test123",
            "selected_policy_label": "policy_A",
            "candidate_labels": ["policy_A"],
            "candidate_count": 1,
            "tick": 1,
            "timestamp": 1234567890.0,
        }
        rec = SelectionRecord.from_dict(d)
        assert rec.bias_source_labels == []

    def test_bias_labels_roundtrip(self):
        """to_dict -> from_dict のラウンドトリップでbias_source_labelsが保持される。"""
        original = SelectionRecord(
            selected_policy_label="policy_X",
            candidate_labels=["policy_X", "policy_Y"],
            candidate_count=2,
            tick=5,
            bias_source_labels=["decision_bias", "context_sensitivity", "scoring_fluctuation"],
        )
        d = original.to_dict()
        restored = SelectionRecord.from_dict(d)
        assert restored.bias_source_labels == original.bias_source_labels

    def test_bias_labels_state_roundtrip(self):
        """SelectionAttributionStateのto_dict -> from_dictでbias_source_labelsが保持される。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection(
            "pA", ["pA", "pB"], tick=1,
            bias_source_labels=["decision_bias", "value_orientation"],
        )
        recorder.record_selection(
            "pB", ["pA", "pB", "pC"], tick=2,
            bias_source_labels=["stability_valve"],
        )

        d = recorder.state.to_dict()
        restored = SelectionAttributionState.from_dict(d)

        assert restored.records[0].bias_source_labels == ["decision_bias", "value_orientation"]
        assert restored.records[1].bias_source_labels == ["stability_valve"]
        assert restored.latest_record.bias_source_labels == ["stability_valve"]

    def test_bias_labels_not_in_enrichment_entries(self):
        """enrichmentエントリにbias_source_labelsが含まれないこと（遮断）。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection(
            "pA", ["pA", "pB"], tick=1,
            bias_source_labels=["decision_bias", "value_orientation"],
        )

        data = recorder.get_enrichment_data()
        for entry in data["recent_entries"]:
            assert "bias_source_labels" not in entry
            assert "bias" not in str(entry).lower()

    def test_bias_labels_not_in_enrichment_summary(self):
        """enrichmentサマリにバイアス関連情報が含まれないこと（遮断）。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection(
            "pA", ["pA", "pB"], tick=1,
            bias_source_labels=["decision_bias", "context_sensitivity", "scoring_fluctuation"],
        )

        data = recorder.get_enrichment_data()
        summary = data["summary_text"]
        assert "bias" not in summary.lower()
        assert "decision_bias" not in summary
        assert "context_sensitivity" not in summary
        assert "scoring_fluctuation" not in summary

    def test_bias_labels_accessible_via_get_all_records(self):
        """内省系参照経路(get_all_records)でbias_source_labelsが参照可能。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection(
            "pA", ["pA"], tick=1,
            bias_source_labels=["decision_bias", "stability_valve"],
        )

        records = recorder.get_all_records()
        assert records[0].bias_source_labels == ["decision_bias", "stability_valve"]

    def test_bias_labels_accessible_via_get_reference_history(self):
        """内省系参照経路(get_reference_history)でbias_source_labelsが参照可能。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection(
            "pA", ["pA"], tick=1,
            bias_source_labels=["value_orientation", "persistent_commitment"],
        )

        history = recorder.get_reference_history()
        assert history[0].bias_source_labels == ["value_orientation", "persistent_commitment"]

    def test_bias_labels_no_scores_or_weights(self):
        """bias_source_labelsには名前のみが記録され、スコア・重み・方向性は含まれない。"""
        recorder = SelectionAttributionRecorder()
        labels = ["decision_bias", "context_sensitivity", "stability_valve",
                  "value_orientation", "persistent_commitment", "scoring_fluctuation"]
        rec = recorder.record_selection(
            "pA", ["pA"], tick=1,
            bias_source_labels=labels,
        )
        # すべて単純な文字列であること
        for label in rec.bias_source_labels:
            assert isinstance(label, str)
        # to_dict内でも名前リストのみ
        d = rec.to_dict()
        for label in d["bias_source_labels"]:
            assert isinstance(label, str)

    def test_bias_labels_preserved_after_pushout(self):
        """押し出し後も残った記録のbias_source_labelsが保持される。"""
        cfg = SelectionAttributionConfig(max_records=2)
        recorder = SelectionAttributionRecorder(config=cfg)
        recorder.record_selection("p1", ["p1"], tick=1, bias_source_labels=["a"])
        recorder.record_selection("p2", ["p2"], tick=2, bias_source_labels=["b", "c"])
        recorder.record_selection("p3", ["p3"], tick=3, bias_source_labels=["d"])

        # p1は押し出された、p2とp3が残る
        assert len(recorder.state.records) == 2
        assert recorder.state.records[0].bias_source_labels == ["b", "c"]
        assert recorder.state.records[1].bias_source_labels == ["d"]

    def test_all_records_equal_with_bias_labels(self):
        """bias_source_labels付きでも全記録等価の原則が維持される。"""
        recorder = SelectionAttributionRecorder()
        recorder.record_selection("p1", ["p1"], tick=1, bias_source_labels=["a", "b"])
        recorder.record_selection("p2", ["p2"], tick=2, bias_source_labels=["c"])
        recorder.record_selection("p3", ["p3"], tick=3, bias_source_labels=[])

        for rec in recorder.state.records:
            d = rec.to_dict()
            # 重み・スコア・優先度の属性が存在しない
            assert "weight" not in d
            assert "score" not in d
            assert "priority" not in d
            assert "importance" not in d
            assert "rank" not in d
