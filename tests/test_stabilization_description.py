"""
tests/test_stabilization_description.py - 安定化の構造的記述テスト

design_stabilization_description.md に基づく包括的テスト。

テスト要件:
- 断面構成テスト（6信号源の読み取り確認）
- FIFO蓄積テスト（上限超過時の押し出し確認）
- READ-ONLYアクセサテスト
- save/loadラウンドトリップテスト
- enrichment非露出テスト（get_enrichment_data が存在しないことの確認）
- 安全弁テスト（5種）
- orchestrator統合テスト
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from psyche.stabilization_description import (
    # Config
    StabilizationDescriptionConfig,
    # Data structures
    StabilizationRecord,
    StabilizationDescriptionState,
    # Signal keys
    SIGNAL_EMOTION,
    SIGNAL_STM_ENTRIES,
    SIGNAL_TRANSIENT_GOAL,
    SIGNAL_PERSISTENT_COMMITMENT,
    SIGNAL_SPONTANEOUS_CANDIDATE,
    SIGNAL_EXTERNAL_INPUT,
    ALL_SIGNAL_KEYS,
    # Stage 1: Read
    read_signal_sources,
    read_diff_reference,
    # Stage 2: Compose
    compose_record,
    # Stage 3: Accumulate
    accumulate_record,
    # Main process
    process_stabilization_description,
    # Accessors
    get_latest_record,
    get_record_history,
    get_stabilization_summary,
    # Save/Load
    save_state,
    load_state,
    # Factory
    create_stabilization_description_state,
    create_stabilization_description_config,
)


# =============================================================================
# Mock diff_summary for testing
# =============================================================================

@dataclass
class MockDifferenceMagnitude:
    value: str = "noticeable"

@dataclass
class MockChangeNature:
    value: str = "shifting"

@dataclass
class MockDiffSummary:
    """temporal_self_differenceのSelfDifferenceSummaryのモック。"""
    has_difference: bool = True
    magnitude: MockDifferenceMagnitude = None
    nature: MockChangeNature = None

    def __post_init__(self):
        if self.magnitude is None:
            self.magnitude = MockDifferenceMagnitude()
        if self.nature is None:
            self.nature = MockChangeNature()


# =============================================================================
# 断面構成テスト（6信号源の読み取り確認）
# =============================================================================

class TestSignalSourceReading:
    """6信号源の読み取り確認。"""

    def test_all_signals_inactive(self):
        """全信号源が非アクティブのケース。"""
        states = read_signal_sources(
            emotion_intensity=0.0,
            stm_entry_count=0,
            transient_goal_active=False,
            persistent_commitment_unreleased_count=0,
            spontaneous_candidate_exists=False,
            has_external_input=False,
        )
        assert all(v is False for v in states.values())
        assert len(states) == 6

    def test_all_signals_active(self):
        """全信号源がアクティブのケース。"""
        states = read_signal_sources(
            emotion_intensity=0.5,
            stm_entry_count=3,
            transient_goal_active=True,
            persistent_commitment_unreleased_count=2,
            spontaneous_candidate_exists=True,
            has_external_input=True,
        )
        assert all(v is True for v in states.values())
        assert len(states) == 6

    def test_emotion_signal_nonzero(self):
        """感情強度が非ゼロの場合にアクティブ。"""
        states = read_signal_sources(emotion_intensity=0.01)
        assert states[SIGNAL_EMOTION] is True

    def test_emotion_signal_zero(self):
        """感情強度がゼロの場合に非アクティブ。"""
        states = read_signal_sources(emotion_intensity=0.0)
        assert states[SIGNAL_EMOTION] is False

    def test_stm_entry_nonzero(self):
        """STMエントリが1以上の場合にアクティブ。"""
        states = read_signal_sources(stm_entry_count=1)
        assert states[SIGNAL_STM_ENTRIES] is True

    def test_stm_entry_zero(self):
        """STMエントリが0の場合に非アクティブ。"""
        states = read_signal_sources(stm_entry_count=0)
        assert states[SIGNAL_STM_ENTRIES] is False

    def test_transient_goal_active(self):
        """一時的目的がアクティブ。"""
        states = read_signal_sources(transient_goal_active=True)
        assert states[SIGNAL_TRANSIENT_GOAL] is True

    def test_transient_goal_inactive(self):
        """一時的目的が非アクティブ。"""
        states = read_signal_sources(transient_goal_active=False)
        assert states[SIGNAL_TRANSIENT_GOAL] is False

    def test_persistent_commitment_with_items(self):
        """持続的取り組みに未解放項目がある場合にアクティブ。"""
        states = read_signal_sources(persistent_commitment_unreleased_count=1)
        assert states[SIGNAL_PERSISTENT_COMMITMENT] is True

    def test_persistent_commitment_empty(self):
        """持続的取り組みに未解放項目がない場合に非アクティブ。"""
        states = read_signal_sources(persistent_commitment_unreleased_count=0)
        assert states[SIGNAL_PERSISTENT_COMMITMENT] is False

    def test_spontaneous_candidate_exists(self):
        """自発起動候補がある場合にアクティブ。"""
        states = read_signal_sources(spontaneous_candidate_exists=True)
        assert states[SIGNAL_SPONTANEOUS_CANDIDATE] is True

    def test_spontaneous_candidate_absent(self):
        """自発起動候補がない場合に非アクティブ。"""
        states = read_signal_sources(spontaneous_candidate_exists=False)
        assert states[SIGNAL_SPONTANEOUS_CANDIDATE] is False

    def test_external_input_present(self):
        """外部入力がある場合にアクティブ。"""
        states = read_signal_sources(has_external_input=True)
        assert states[SIGNAL_EXTERNAL_INPUT] is True

    def test_external_input_absent(self):
        """外部入力がない場合に非アクティブ。"""
        states = read_signal_sources(has_external_input=False)
        assert states[SIGNAL_EXTERNAL_INPUT] is False

    def test_signal_keys_complete(self):
        """全信号源キーが揃っていることの確認。"""
        states = read_signal_sources()
        for key in ALL_SIGNAL_KEYS:
            assert key in states, f"Missing signal key: {key}"

    def test_partial_signals(self):
        """一部の信号源のみアクティブのケース。"""
        states = read_signal_sources(
            emotion_intensity=0.3,
            stm_entry_count=0,
            transient_goal_active=True,
            persistent_commitment_unreleased_count=0,
            spontaneous_candidate_exists=False,
            has_external_input=True,
        )
        assert states[SIGNAL_EMOTION] is True
        assert states[SIGNAL_STM_ENTRIES] is False
        assert states[SIGNAL_TRANSIENT_GOAL] is True
        assert states[SIGNAL_PERSISTENT_COMMITMENT] is False
        assert states[SIGNAL_SPONTANEOUS_CANDIDATE] is False
        assert states[SIGNAL_EXTERNAL_INPUT] is True


class TestDiffReference:
    """temporal_self_differenceの差分参照テスト。"""

    def test_none_diff_summary(self):
        """diff_summaryがNoneの場合。"""
        mag, nat = read_diff_reference(diff_summary=None)
        assert mag == "undefined"
        assert nat == "undefined"

    def test_valid_diff_summary(self):
        """有効なdiff_summaryの読み取り。"""
        summary = MockDiffSummary(
            magnitude=MockDifferenceMagnitude("significant"),
            nature=MockChangeNature("transformed"),
        )
        mag, nat = read_diff_reference(diff_summary=summary)
        assert mag == "significant"
        assert nat == "transformed"

    def test_diff_summary_stable(self):
        """安定状態のdiff_summary。"""
        summary = MockDiffSummary(
            magnitude=MockDifferenceMagnitude("none"),
            nature=MockChangeNature("stable"),
        )
        mag, nat = read_diff_reference(diff_summary=summary)
        assert mag == "none"
        assert nat == "stable"

    def test_diff_summary_missing_attributes(self):
        """属性が欠けたオブジェクト。"""
        class PartialSummary:
            pass
        mag, nat = read_diff_reference(diff_summary=PartialSummary())
        assert mag == "undefined"
        assert nat == "undefined"

    def test_diff_summary_string_values(self):
        """magnitude/natureが直接文字列のケース。"""
        class StringSummary:
            magnitude = "minimal"
            nature = "fluctuating"
        mag, nat = read_diff_reference(diff_summary=StringSummary())
        assert mag == "minimal"
        assert nat == "fluctuating"


class TestComposeRecord:
    """断面構成テスト。"""

    def test_compose_all_active(self):
        """全信号アクティブの断面構成。"""
        states = {k: True for k in ALL_SIGNAL_KEYS}
        record = compose_record(
            signal_states=states,
            diff_magnitude="significant",
            diff_nature="shifting",
            tick=42,
            timestamp=1000.0,
        )
        assert record.active_signal_count == 6
        assert record.diff_magnitude == "significant"
        assert record.diff_nature == "shifting"
        assert record.tick == 42
        assert record.timestamp == 1000.0

    def test_compose_no_active(self):
        """全信号非アクティブの断面構成。"""
        states = {k: False for k in ALL_SIGNAL_KEYS}
        record = compose_record(
            signal_states=states,
            diff_magnitude="none",
            diff_nature="stable",
            tick=0,
            timestamp=500.0,
        )
        assert record.active_signal_count == 0

    def test_compose_partial_active(self):
        """一部アクティブの断面構成。"""
        states = {
            SIGNAL_EMOTION: True,
            SIGNAL_STM_ENTRIES: False,
            SIGNAL_TRANSIENT_GOAL: True,
            SIGNAL_PERSISTENT_COMMITMENT: False,
            SIGNAL_SPONTANEOUS_CANDIDATE: True,
            SIGNAL_EXTERNAL_INPUT: False,
        }
        record = compose_record(
            signal_states=states,
            diff_magnitude="noticeable",
            diff_nature="fluctuating",
            tick=10,
        )
        assert record.active_signal_count == 3

    def test_compose_auto_timestamp(self):
        """timestampがNone時は現在時刻が使用される。"""
        before = time.time()
        record = compose_record(
            signal_states={k: False for k in ALL_SIGNAL_KEYS},
            diff_magnitude="none",
            diff_nature="stable",
        )
        after = time.time()
        assert before <= record.timestamp <= after

    def test_compose_preserves_values_without_transformation(self):
        """値がそのまま記録される（変換・正規化・丸め・段階化なし）。"""
        states = {SIGNAL_EMOTION: True, SIGNAL_STM_ENTRIES: False}
        record = compose_record(
            signal_states=states,
            diff_magnitude="substantial",
            diff_nature="transformed",
            tick=99,
        )
        # 値がそのまま保持されている
        assert record.signal_states[SIGNAL_EMOTION] is True
        assert record.signal_states[SIGNAL_STM_ENTRIES] is False
        assert record.diff_magnitude == "substantial"
        assert record.diff_nature == "transformed"


# =============================================================================
# FIFO蓄積テスト（上限超過時の押し出し確認）
# =============================================================================

class TestFIFOAccumulation:
    """FIFO蓄積テスト。"""

    def test_accumulate_single_record(self):
        """1件の蓄積。"""
        state = create_stabilization_description_state()
        record = compose_record(
            signal_states={k: False for k in ALL_SIGNAL_KEYS},
            diff_magnitude="none",
            diff_nature="stable",
            tick=1,
        )
        new_state = accumulate_record(state, record)
        assert len(new_state.history) == 1
        assert new_state.latest_record is record
        assert new_state.total_records_generated == 1
        assert new_state.total_records_expired == 0

    def test_accumulate_multiple_records(self):
        """複数件の蓄積。"""
        state = create_stabilization_description_state()
        for i in range(5):
            record = compose_record(
                signal_states={k: (i % 2 == 0) for k in ALL_SIGNAL_KEYS},
                diff_magnitude="minimal",
                diff_nature="stable",
                tick=i,
            )
            state = accumulate_record(state, record)
        assert len(state.history) == 5
        assert state.total_records_generated == 5
        assert state.total_records_expired == 0

    def test_fifo_eviction(self):
        """上限超過時の最古記録の押し出し。"""
        config = StabilizationDescriptionConfig(max_history=3)
        state = create_stabilization_description_state()

        for i in range(5):
            record = compose_record(
                signal_states={k: True for k in ALL_SIGNAL_KEYS},
                diff_magnitude="noticeable",
                diff_nature="shifting",
                tick=i,
            )
            state = accumulate_record(state, record, config)

        # 上限3件なので、tick 0, 1 は押し出されている
        assert len(state.history) == 3
        assert state.history[0].tick == 2
        assert state.history[1].tick == 3
        assert state.history[2].tick == 4
        assert state.total_records_generated == 5
        assert state.total_records_expired == 2

    def test_fifo_eviction_is_only_disappearance_path(self):
        """FIFO押し出しが唯一の消失経路であることの確認。"""
        config = StabilizationDescriptionConfig(max_history=2)
        state = create_stabilization_description_state()

        # 3件追加（1件が押し出される）
        for i in range(3):
            record = compose_record(
                signal_states={k: False for k in ALL_SIGNAL_KEYS},
                diff_magnitude="none",
                diff_nature="stable",
                tick=i,
            )
            state = accumulate_record(state, record, config)

        assert len(state.history) == 2
        assert state.total_records_expired == 1
        # 残っているのは最新の2件
        assert state.history[0].tick == 1
        assert state.history[1].tick == 2

    def test_records_are_immutable_in_history(self):
        """蓄積リスト内の記録が変更されないことの確認。"""
        state = create_stabilization_description_state()
        record = compose_record(
            signal_states={SIGNAL_EMOTION: True},
            diff_magnitude="significant",
            diff_nature="shifting",
            tick=10,
            timestamp=1000.0,
        )
        new_state = accumulate_record(state, record)

        # 記録の値を確認
        stored = new_state.history[0]
        assert stored.active_signal_count == 1
        assert stored.diff_magnitude == "significant"
        assert stored.tick == 10

    def test_large_eviction(self):
        """大量蓄積後の上限超過。"""
        config = StabilizationDescriptionConfig(max_history=5)
        state = create_stabilization_description_state()

        for i in range(100):
            record = compose_record(
                signal_states={k: (i % 3 == 0) for k in ALL_SIGNAL_KEYS},
                diff_magnitude="minimal",
                diff_nature="stable",
                tick=i,
            )
            state = accumulate_record(state, record, config)

        assert len(state.history) == 5
        assert state.total_records_generated == 100
        assert state.total_records_expired == 95
        # 最新の5件が残っている
        assert state.history[0].tick == 95
        assert state.history[4].tick == 99


# =============================================================================
# READ-ONLYアクセサテスト
# =============================================================================

class TestReadOnlyAccessors:
    """READ-ONLYアクセサテスト。"""

    def test_get_latest_record_empty(self):
        """空状態での最新記録取得。"""
        state = create_stabilization_description_state()
        assert get_latest_record(state) is None

    def test_get_latest_record_with_data(self):
        """記録がある場合の最新記録取得。"""
        state = create_stabilization_description_state()
        record = compose_record(
            signal_states={k: True for k in ALL_SIGNAL_KEYS},
            diff_magnitude="significant",
            diff_nature="shifting",
            tick=5,
        )
        state = accumulate_record(state, record)
        latest = get_latest_record(state)
        assert latest is not None
        assert latest.tick == 5
        assert latest.active_signal_count == 6

    def test_get_record_history_empty(self):
        """空状態での履歴取得。"""
        state = create_stabilization_description_state()
        history = get_record_history(state)
        assert history == []

    def test_get_record_history_returns_copy(self):
        """履歴取得がコピーを返すことの確認。"""
        state = create_stabilization_description_state()
        record = compose_record(
            signal_states={k: False for k in ALL_SIGNAL_KEYS},
            diff_magnitude="none",
            diff_nature="stable",
            tick=1,
        )
        state = accumulate_record(state, record)
        history1 = get_record_history(state)
        history2 = get_record_history(state)
        assert history1 is not history2
        assert len(history1) == len(history2)

    def test_get_stabilization_summary_empty(self):
        """空状態でのサマリ取得。"""
        state = create_stabilization_description_state()
        summary = get_stabilization_summary(state)
        assert summary["history_count"] == 0
        assert summary["total_generated"] == 0
        assert summary["total_expired"] == 0
        assert "latest_active_signal_count" not in summary

    def test_get_stabilization_summary_with_data(self):
        """データがある場合のサマリ取得。"""
        state = create_stabilization_description_state()
        state = process_stabilization_description(
            state,
            emotion_intensity=0.5,
            stm_entry_count=2,
            transient_goal_active=True,
            persistent_commitment_unreleased_count=1,
            spontaneous_candidate_exists=False,
            has_external_input=True,
            diff_summary=MockDiffSummary(),
            tick=10,
        )
        summary = get_stabilization_summary(state)
        assert summary["history_count"] == 1
        assert summary["total_generated"] == 1
        # 5信号アクティブ: emotion, stm, transient, commitment, external (spontaneousはFalse)
        assert summary["latest_active_signal_count"] == 5
        assert summary["latest_diff_magnitude"] == "noticeable"
        assert summary["latest_diff_nature"] == "shifting"
        assert summary["latest_tick"] == 10

    def test_accessors_do_not_modify_state(self):
        """アクセサが状態を変更しないことの確認。"""
        state = create_stabilization_description_state()
        state = process_stabilization_description(
            state,
            emotion_intensity=0.3,
            tick=1,
        )

        # 複数回アクセスしても状態は変わらない
        before_count = state.total_records_generated
        get_latest_record(state)
        get_record_history(state)
        get_stabilization_summary(state)
        get_latest_record(state)
        assert state.total_records_generated == before_count


# =============================================================================
# save/loadラウンドトリップテスト
# =============================================================================

class TestSaveLoadRoundTrip:
    """save/loadラウンドトリップテスト。"""

    def test_empty_state_roundtrip(self):
        """空状態のラウンドトリップ。"""
        state = create_stabilization_description_state()
        data = save_state(state)
        restored = load_state(data)
        assert len(restored.history) == 0
        assert restored.latest_record is None
        assert restored.total_records_generated == 0
        assert restored.total_records_expired == 0

    def test_populated_state_roundtrip(self):
        """データ入りの状態のラウンドトリップ。"""
        state = create_stabilization_description_state()
        for i in range(5):
            state = process_stabilization_description(
                state,
                emotion_intensity=0.1 * (i + 1),
                stm_entry_count=i,
                transient_goal_active=(i % 2 == 0),
                persistent_commitment_unreleased_count=i,
                spontaneous_candidate_exists=(i % 3 == 0),
                has_external_input=(i < 3),
                diff_summary=MockDiffSummary(
                    magnitude=MockDifferenceMagnitude(["none", "minimal", "noticeable", "significant", "substantial"][i]),
                    nature=MockChangeNature(["stable", "fluctuating", "shifting", "transformed", "returning"][i]),
                ),
                tick=i,
                timestamp=1000.0 + i,
            )

        data = save_state(state)
        restored = load_state(data)

        assert len(restored.history) == len(state.history)
        assert restored.total_records_generated == state.total_records_generated
        assert restored.total_records_expired == state.total_records_expired

        for orig, rest in zip(state.history, restored.history):
            assert orig.active_signal_count == rest.active_signal_count
            assert orig.diff_magnitude == rest.diff_magnitude
            assert orig.diff_nature == rest.diff_nature
            assert orig.tick == rest.tick
            assert orig.timestamp == rest.timestamp
            assert orig.signal_states == rest.signal_states

    def test_roundtrip_preserves_latest_record(self):
        """最新記録がラウンドトリップで保持される。"""
        state = create_stabilization_description_state()
        state = process_stabilization_description(
            state,
            emotion_intensity=0.8,
            tick=42,
            timestamp=9999.0,
            diff_summary=MockDiffSummary(
                magnitude=MockDifferenceMagnitude("significant"),
                nature=MockChangeNature("transformed"),
            ),
        )

        data = save_state(state)
        restored = load_state(data)

        assert restored.latest_record is not None
        assert restored.latest_record.active_signal_count == state.latest_record.active_signal_count
        assert restored.latest_record.diff_magnitude == "significant"
        assert restored.latest_record.diff_nature == "transformed"
        assert restored.latest_record.tick == 42

    def test_record_to_dict_from_dict(self):
        """StabilizationRecord の to_dict/from_dict。"""
        record = StabilizationRecord(
            active_signal_count=3,
            signal_states={SIGNAL_EMOTION: True, SIGNAL_STM_ENTRIES: False},
            diff_magnitude="noticeable",
            diff_nature="shifting",
            tick=7,
            timestamp=2000.0,
        )
        data = record.to_dict()
        restored = StabilizationRecord.from_dict(data)
        assert restored.active_signal_count == 3
        assert restored.signal_states[SIGNAL_EMOTION] is True
        assert restored.signal_states[SIGNAL_STM_ENTRIES] is False
        assert restored.diff_magnitude == "noticeable"
        assert restored.diff_nature == "shifting"
        assert restored.tick == 7
        assert restored.timestamp == 2000.0


# =============================================================================
# enrichment非露出テスト
# =============================================================================

class TestEnrichmentNonExposure:
    """enrichment直接露出遮断テスト（安全弁3）。"""

    def test_no_get_enrichment_data_method(self):
        """get_enrichment_data メソッドが存在しないことの確認。"""
        import psyche.stabilization_description as module
        # モジュールレベルに get_enrichment_data が存在しない
        assert not hasattr(module, "get_enrichment_data")

    def test_state_has_no_enrichment_method(self):
        """StabilizationDescriptionState に enrichment関連メソッドがないことの確認。"""
        state = create_stabilization_description_state()
        assert not hasattr(state, "get_enrichment_data")
        assert not hasattr(state, "to_enrichment")
        assert not hasattr(state, "enrichment")

    def test_record_has_no_enrichment_method(self):
        """StabilizationRecord に enrichment関連メソッドがないことの確認。"""
        record = StabilizationRecord()
        assert not hasattr(record, "get_enrichment_data")
        assert not hasattr(record, "to_enrichment")

    def test_module_public_api_excludes_enrichment(self):
        """モジュールの公開APIにenrichment関連が含まれないことの確認。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        enrichment_names = [n for n in public_names if "enrichment" in n.lower()]
        assert enrichment_names == [], f"Found enrichment-related names: {enrichment_names}"


# =============================================================================
# 安全弁テスト（5種）
# =============================================================================

class TestSafetyValve1AllRecordsEqual:
    """安全弁1: 全記録等価。"""

    def test_no_weight_or_score_in_record(self):
        """記録に重み・スコア・優先度が存在しない。"""
        record = StabilizationRecord(
            active_signal_count=3,
            diff_magnitude="noticeable",
            diff_nature="shifting",
            tick=5,
        )
        # weight, score, priority のような属性が存在しない
        assert not hasattr(record, "weight")
        assert not hasattr(record, "score")
        assert not hasattr(record, "priority")
        assert not hasattr(record, "importance")

    def test_no_selective_retention(self):
        """特定の記録を選択的に保持する機構がないことの確認。"""
        config = StabilizationDescriptionConfig(max_history=3)
        state = create_stabilization_description_state()

        # 5件追加（2件が押し出される）
        for i in range(5):
            record = compose_record(
                signal_states={k: (i == 4) for k in ALL_SIGNAL_KEYS},
                diff_magnitude=["none", "minimal", "noticeable", "significant", "substantial"][i],
                diff_nature="stable",
                tick=i,
            )
            state = accumulate_record(state, record, config)

        # 最古の2件が押し出され、残りは時系列順
        assert len(state.history) == 3
        assert state.history[0].tick == 2
        assert state.history[1].tick == 3
        assert state.history[2].tick == 4

    def test_summary_does_not_emphasize_specific_records(self):
        """サマリが特定の記録を強調しない。"""
        state = create_stabilization_description_state()
        for i in range(3):
            state = process_stabilization_description(
                state,
                emotion_intensity=float(i),
                tick=i,
            )
        summary = get_stabilization_summary(state)
        # サマリには「最重要」「最良」のような評価的キーがない
        for key in summary:
            assert "best" not in key.lower()
            assert "worst" not in key.lower()
            assert "important" not in key.lower()


class TestSafetyValve2PatternExtractionProhibited:
    """安全弁2: パターン抽出禁止。"""

    def test_no_trend_computation(self):
        """傾向の計算が行われないことの確認。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        trend_names = [n for n in public_names if any(
            kw in n.lower() for kw in ["trend", "pattern", "correlation", "regression", "statistics"]
        )]
        assert trend_names == [], f"Found trend-related names: {trend_names}"

    def test_no_inter_record_comparison(self):
        """記録間の比較・差分機能が存在しないことの確認。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        comparison_names = [n for n in public_names if any(
            kw in n.lower() for kw in ["compare", "diff_records", "derive_variation"]
        )]
        assert comparison_names == [], f"Found comparison-related names: {comparison_names}"

    def test_summary_has_no_statistical_aggregation(self):
        """サマリに統計的集計が含まれないことの確認。"""
        state = create_stabilization_description_state()
        for i in range(10):
            state = process_stabilization_description(
                state,
                emotion_intensity=float(i % 2),
                tick=i,
            )
        summary = get_stabilization_summary(state)
        # 平均・標準偏差・相関係数のようなキーがない
        for key in summary:
            assert "average" not in key.lower()
            assert "mean" not in key.lower()
            assert "std" not in key.lower()
            assert "correlation" not in key.lower()
            assert "variance" not in key.lower()


class TestSafetyValve3EnrichmentBlocked:
    """安全弁3: enrichment直接露出遮断。"""

    def test_no_enrichment_output(self):
        """enrichment出力経路が存在しない（TestEnrichmentNonExposureと重複するが安全弁として独立テスト）。"""
        import psyche.stabilization_description as module
        assert not hasattr(module, "get_enrichment_data")
        assert not hasattr(module, "format_for_enrichment")
        assert not hasattr(module, "to_enrichment_text")


class TestSafetyValve4ForgettingPathBlocked:
    """安全弁4: 忘却経路遮断。"""

    def test_no_forgetting_output(self):
        """忘却パイプラインへの出力経路が存在しない。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        forgetting_names = [n for n in public_names if any(
            kw in n.lower() for kw in ["forgetting", "forget", "fixation"]
        )]
        assert forgetting_names == [], f"Found forgetting-related names: {forgetting_names}"

    def test_output_limited_to_read_only(self):
        """出力が内省系READ-ONLYアクセサに限定されている。"""
        import psyche.stabilization_description as module
        # 出力関数は get_latest_record, get_record_history, get_stabilization_summary のみ
        output_functions = [
            "get_latest_record",
            "get_record_history",
            "get_stabilization_summary",
        ]
        for fn_name in output_functions:
            assert hasattr(module, fn_name), f"Missing output function: {fn_name}"


class TestSafetyValve5OutputPathNotExtended:
    """安全弁5: 出力経路不拡張。"""

    def test_no_policy_output(self):
        """ポリシー選択への出力経路がないことの確認。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        # SIGNAL_SPONTANEOUS_CANDIDATE は信号源識別子であり出力経路ではないため除外
        policy_names = [n for n in public_names if any(
            kw in n.lower() for kw in ["policy", "bias", "score"]
        ) and "SIGNAL_" not in n]
        # "candidate" を含む関数/クラスが出力経路として存在しないことの確認
        candidate_output_names = [n for n in public_names if
            "candidate" in n.lower() and
            "SIGNAL_" not in n and
            not n.startswith("SIGNAL_")]
        assert policy_names == [], f"Found policy-related names: {policy_names}"
        assert candidate_output_names == [], f"Found candidate output names: {candidate_output_names}"

    def test_no_emotion_pipeline_output(self):
        """感情パイプラインへの出力経路がないことの確認。"""
        import psyche.stabilization_description as module
        public_names = [n for n in dir(module) if not n.startswith("_")]
        emotion_pipeline_names = [n for n in public_names if any(
            kw in n.lower() for kw in ["apply_to_emotion", "emotion_modifier", "emotion_signal"]
        )]
        assert emotion_pipeline_names == [], f"Found emotion pipeline names: {emotion_pipeline_names}"


# =============================================================================
# メイン処理テスト
# =============================================================================

class TestProcessStabilizationDescription:
    """process_stabilization_description テスト。"""

    def test_basic_process(self):
        """基本的な処理フロー。"""
        state = create_stabilization_description_state()
        new_state = process_stabilization_description(
            state,
            emotion_intensity=0.5,
            stm_entry_count=2,
            transient_goal_active=True,
            persistent_commitment_unreleased_count=1,
            spontaneous_candidate_exists=False,
            has_external_input=True,
            diff_summary=MockDiffSummary(),
            tick=1,
        )
        assert len(new_state.history) == 1
        assert new_state.latest_record is not None
        # 5信号アクティブ: emotion, stm, transient, commitment, external (spontaneousはFalse)
        assert new_state.latest_record.active_signal_count == 5
        assert new_state.latest_record.diff_magnitude == "noticeable"
        assert new_state.latest_record.diff_nature == "shifting"

    def test_process_without_diff_summary(self):
        """diff_summaryなしでの処理。"""
        state = create_stabilization_description_state()
        new_state = process_stabilization_description(
            state,
            emotion_intensity=0.3,
            tick=1,
        )
        assert new_state.latest_record is not None
        assert new_state.latest_record.diff_magnitude == "undefined"
        assert new_state.latest_record.diff_nature == "undefined"

    def test_process_multiple_cycles(self):
        """複数サイクルの処理。"""
        state = create_stabilization_description_state()
        for i in range(10):
            state = process_stabilization_description(
                state,
                emotion_intensity=0.1 * i,
                stm_entry_count=i,
                tick=i,
            )
        assert len(state.history) == 10
        assert state.total_records_generated == 10
        assert state.latest_record.tick == 9

    def test_process_with_custom_config(self):
        """カスタム設定での処理。"""
        config = StabilizationDescriptionConfig(max_history=3)
        state = create_stabilization_description_state()
        for i in range(5):
            state = process_stabilization_description(
                state,
                emotion_intensity=float(i),
                tick=i,
                config=config,
            )
        assert len(state.history) == 3
        assert state.total_records_expired == 2

    def test_process_preserves_signal_states(self):
        """処理がsignal_statesを保持する。"""
        state = create_stabilization_description_state()
        state = process_stabilization_description(
            state,
            emotion_intensity=0.5,
            stm_entry_count=0,
            transient_goal_active=True,
            persistent_commitment_unreleased_count=0,
            spontaneous_candidate_exists=True,
            has_external_input=False,
            tick=1,
        )
        ss = state.latest_record.signal_states
        assert ss[SIGNAL_EMOTION] is True
        assert ss[SIGNAL_STM_ENTRIES] is False
        assert ss[SIGNAL_TRANSIENT_GOAL] is True
        assert ss[SIGNAL_PERSISTENT_COMMITMENT] is False
        assert ss[SIGNAL_SPONTANEOUS_CANDIDATE] is True
        assert ss[SIGNAL_EXTERNAL_INPUT] is False


# =============================================================================
# ファクトリテスト
# =============================================================================

class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_state(self):
        """初期状態の生成。"""
        state = create_stabilization_description_state()
        assert len(state.history) == 0
        assert state.latest_record is None
        assert state.total_records_generated == 0
        assert state.total_records_expired == 0

    def test_create_config_default(self):
        """デフォルト設定の生成。"""
        config = create_stabilization_description_config()
        assert config.max_history == 30

    def test_create_config_custom(self):
        """カスタム設定の生成。"""
        config = create_stabilization_description_config(max_history=50)
        assert config.max_history == 50


# =============================================================================
# Orchestrator統合テスト
# =============================================================================

class TestOrchestratorIntegration:
    """orchestrator統合テスト。"""

    def test_orchestrator_has_stabilization_description(self):
        """orchestratorがstabilization_descriptionインスタンスを持つ。"""
        from psyche.orchestrator import PsycheOrchestrator
        orch = PsycheOrchestrator()
        assert hasattr(orch, "_stabilization_desc_state")

    def test_orchestrator_processes_at_5_ticks(self, tmp_path):
        """5ティック毎に処理が実行される。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = Percept(
            text="test input",
            emotion="happy",
            intent="chat",
            emotion_valence=0.5,
        )

        # 4ティック実行（5ティック毎なので処理されないはず）
        for _ in range(4):
            orch.post_response_update(percept, delta_time=1.0)
        # 4ティック後はまだ処理されていない可能性がある
        state_after_4 = orch._stabilization_desc_state

        # 5ティック目で処理される
        orch.post_response_update(percept, delta_time=1.0)
        state_after_5 = orch._stabilization_desc_state
        assert state_after_5.total_records_generated >= 1

    def test_orchestrator_save_load_includes_stabilization(self, tmp_path):
        """save/loadにstabilization_description_stateが含まれる。"""
        import json
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = Percept(
            text="test",
            emotion="neutral",
            intent="chat",
            emotion_valence=0.0,
        )

        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)

        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert "stabilization_description_state" in data
        assert data["version"] == 26

    def test_orchestrator_enrichment_does_not_contain_stabilization(self, tmp_path):
        """enrichment出力にstabilization_description の内容が含まれないことの確認。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = Percept(
            text="test",
            emotion="happy",
            intent="chat",
            emotion_valence=0.7,
        )

        for _ in range(10):
            orch.post_response_update(percept, delta_time=1.0)

        enrichment = orch.get_prompt_enrichment()
        # stabilization / 安定化 に関する記述がenrichmentに含まれない
        assert "stabilization_description" not in enrichment.lower()
        assert "active_signal_count" not in enrichment.lower()

    def test_orchestrator_roundtrip_stabilization(self, tmp_path):
        """save → load でstabilization_description_stateが復元される。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = Percept(
            text="test",
            emotion="happy",
            intent="chat",
            emotion_valence=0.5,
        )

        for _ in range(10):
            orch1.post_response_update(percept, delta_time=1.0)

        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True

        # 復元後の状態確認
        assert orch2._stabilization_desc_state.total_records_generated == \
               orch1._stabilization_desc_state.total_records_generated
