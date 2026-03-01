"""
tests/test_500tick_stability.py - 500ティック安定性テスト

design_500tick_stability_test.md に基づく3グループの検証:
  グループA: 500ティック連続稼働の構造的制約検証
  グループB: 中間保存・復元を含む500ティック安定性検証
  グループC: 統計記録収集（非判定・事実記述のみ）

psycheの変更なし。テストファイルのみ。
アサーションは構造的制約（値の範囲、上限遵守、API正常動作）に限定する。
統計記録にpass/fail閾値を設けない。
"""

import json
import sys
import time
from pathlib import Path

import pytest

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept


# ── Helpers ───────────────────────────────────────────────────────


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent="expression",
        emotion_valence=valence,
    )


# 既存の200ティック安定性テストと同一の入力パターン循環リスト
EMOTIONS = [
    "happy", "sad", "angry", "neutral", "surprised",
    "loving", "teasing", "scared", "happy", "neutral",
]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3, 0.8, 0.4, -0.5, 0.6, 0.0]


def _run_ticks(
    orch: PsycheOrchestrator,
    count: int,
    *,
    with_policy: bool = False,
    policy_interval: int = 5,
    start_offset: int = 0,
) -> None:
    """指定ティック数だけ多様な感情入力で更新する。

    with_policy=True の場合、policy_interval ティック毎に select_policy_dict を呼ぶ。
    start_offset は入力パターンの開始位置を指定する（循環リストのインデックスオフセット）。
    """
    for i in range(count):
        idx = (i + start_offset) % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i + start_offset}",
        )
        orch.post_response_update(percept, delta_time=1.0)
        if with_policy and (i + 1) % policy_interval == 0:
            orch.select_policy_dict(percept, [])


def _run_ticks_with_stats(
    orch: PsycheOrchestrator,
    count: int,
    *,
    policy_interval: int = 10,
    start_offset: int = 0,
) -> dict:
    """指定ティック数だけ実行し、100ティックごとの統計を収集する。

    Returns:
        5区間(0-100, 100-200, 200-300, 300-400, 400-500)ごとの統計辞書
    """
    stats = {}
    current_segment_labels = []
    current_segment_start = start_offset

    for i in range(count):
        global_tick = i + start_offset
        idx = global_tick % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{global_tick}",
        )
        orch.post_response_update(percept, delta_time=1.0)

        if (i + 1) % policy_interval == 0:
            policy = orch.select_policy_dict(percept, [])
            label = policy.get("policy_label", "unknown")
            current_segment_labels.append(label)

        # 100ティックごとに区間統計を記録
        if (i + 1) % 100 == 0:
            segment_idx = (i + 1) // 100
            segment_key = f"{current_segment_start}-{current_segment_start + 100}"

            enrichment = orch.get_prompt_enrichment()
            enrichment_len = len(enrichment) if enrichment else 0

            # 蓄積構造の件数を収集
            accumulation_counts = _collect_accumulation_counts(orch)

            stats[segment_key] = {
                "policy_labels": list(set(current_segment_labels)),
                "policy_label_count": len(set(current_segment_labels)),
                "enrichment_char_count": enrichment_len,
                "accumulation_counts": accumulation_counts,
            }

            current_segment_labels = []
            current_segment_start += 100

    return stats


def _collect_accumulation_counts(orch: PsycheOrchestrator) -> dict:
    """蓄積構造の現在件数を収集する。"""
    counts = {}

    if hasattr(orch, '_expectation_action_diff_log'):
        counts["expectation_action_diff_log"] = len(orch._expectation_action_diff_log)

    if orch._temporal_cognition is not None:
        state = orch._temporal_cognition.state
        if hasattr(state, 'elapsed_records'):
            counts["temporal_cognition_elapsed_records"] = len(state.elapsed_records)

    if orch._action_result_observer is not None:
        state = orch._action_result_observer.state
        if hasattr(state, 'pairs'):
            counts["action_result_pairs"] = len(state.pairs)

    if orch._emotion_cooccurrence_processor is not None:
        state = orch._emotion_cooccurrence_processor.state
        if hasattr(state, 'records'):
            counts["emotion_cooccurrence_records"] = len(state.records)

    if orch._interaction_accumulation is not None:
        state = orch._interaction_accumulation.state
        if hasattr(state, 'pairs'):
            counts["interaction_accumulation_pairs"] = len(state.pairs)

    if orch._contradiction_processor is not None:
        state = orch._contradiction_processor.state
        if hasattr(state, 'window'):
            counts["contradiction_window"] = len(state.window)

    if orch._drive_variation_processor is not None:
        state = orch._drive_variation_processor.state
        if hasattr(state, 'sliding_window'):
            counts["drive_variation_sliding_window"] = len(state.sliding_window)

    if orch._emotional_backdrop_processor is not None:
        state = orch._emotional_backdrop_processor.state
        if hasattr(state, 'sliding_window'):
            counts["emotional_backdrop_sliding_window"] = len(state.sliding_window)

    if orch._introspection_cross_section is not None:
        state = orch._introspection_cross_section._state
        if hasattr(state, 'snapshot_window'):
            counts["introspection_cross_section_snapshot_window"] = len(state.snapshot_window)

    return counts


# ══════════════════════════════════════════════════════════════════
# グループA: 500ティック連続稼働（構造的制約のみ）
# ══════════════════════════════════════════════════════════════════


class Test500TickContinuousRun:
    """500ティック連続稼働の構造的制約検証。

    アサーションは構造的制約に限定:
    - 例外不発生
    - 全感情値・ドライブ値が 0-1 範囲内
    - 蓄積構造の上限遵守
    - enrichment生成の正常性
    - 方針選択の正常動作
    """

    @pytest.mark.slow
    def test_500_ticks_no_exception(self):
        """500ティック連続実行で例外が発生しないこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        assert orch.tick_count == 500

    @pytest.mark.slow
    def test_500_ticks_emotions_in_range(self):
        """500ティック後も全感情値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range at tick 500: {val}"
            )

    @pytest.mark.slow
    def test_500_ticks_drives_in_range(self):
        """500ティック後も全ドライブ値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, (
                f"Drive {name} out of range at tick 500: {val}"
            )

    @pytest.mark.slow
    def test_500_ticks_fear_in_range(self):
        """500ティック後も fear_level が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        assert 0.0 <= orch.fear_level <= 1.0

    @pytest.mark.slow
    def test_500_ticks_expectation_diff_log_bounded(self):
        """500ティック後に expectation_action_diff_log が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        assert len(orch._expectation_action_diff_log) <= 500, (
            "expectation_action_diff_log should be bounded after 500 ticks"
        )

    @pytest.mark.slow
    def test_500_ticks_temporal_cognition_bounded(self):
        """500ティック後に temporal_cognition の蓄積が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._temporal_cognition is not None:
            state = orch._temporal_cognition.state
            if hasattr(state, 'elapsed_records'):
                assert len(state.elapsed_records) <= 300, (
                    "temporal_cognition elapsed_records should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_action_result_pairs_bounded(self):
        """500ティック後に action_result_observer の pairs が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._action_result_observer is not None:
            state = orch._action_result_observer.state
            if hasattr(state, 'pairs'):
                assert len(state.pairs) <= 200, (
                    "action_result pairs should be bounded at 500 ticks"
                )

    @pytest.mark.slow
    def test_500_ticks_emotion_cooccurrence_bounded(self):
        """500ティック後に emotion_cooccurrence の records が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._emotion_cooccurrence_processor is not None:
            state = orch._emotion_cooccurrence_processor.state
            if hasattr(state, 'records'):
                assert len(state.records) <= 100, (
                    "emotion_cooccurrence records should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_interaction_accumulation_bounded(self):
        """500ティック後に interaction_accumulation の pairs が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._interaction_accumulation is not None:
            state = orch._interaction_accumulation.state
            if hasattr(state, 'pairs'):
                assert len(state.pairs) <= 200, (
                    "interaction_accumulation pairs should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_contradiction_bounded(self):
        """500ティック後に contradiction_processor の窓が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._contradiction_processor is not None:
            state = orch._contradiction_processor.state
            if hasattr(state, 'window'):
                assert len(state.window) <= 100, (
                    "contradiction window should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_drive_variation_bounded(self):
        """500ティック後に drive_variation の sliding_window が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._drive_variation_processor is not None:
            state = orch._drive_variation_processor.state
            if hasattr(state, 'sliding_window'):
                assert len(state.sliding_window) <= 100, (
                    "drive_variation sliding_window should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_emotional_backdrop_bounded(self):
        """500ティック後に emotional_backdrop の sliding_window が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._emotional_backdrop_processor is not None:
            state = orch._emotional_backdrop_processor.state
            if hasattr(state, 'sliding_window'):
                assert len(state.sliding_window) <= 100, (
                    "emotional_backdrop sliding_window should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_introspection_cross_section_bounded(self):
        """500ティック後に introspection_cross_section のスナップショット窓が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        if orch._introspection_cross_section is not None:
            state = orch._introspection_cross_section._state
            if hasattr(state, 'snapshot_window'):
                assert len(state.snapshot_window) <= 50, (
                    "introspection_cross_section snapshot_window should be bounded"
                )

    @pytest.mark.slow
    def test_500_ticks_enrichment_valid(self):
        """500ティック後も enrichment が正常に生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 100
        assert "[内面]" in enrichment

    @pytest.mark.slow
    def test_500_ticks_enrichment_bounded_size(self):
        """500ティック後も enrichment のサイズが構造的上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        enrichment = orch.get_prompt_enrichment()
        assert len(enrichment) <= 80000, (
            f"Enrichment too large at 500 ticks: {len(enrichment)} chars"
        )

    @pytest.mark.slow
    def test_500_ticks_policy_selection_valid(self):
        """500ティック後も select_policy_dict が正常動作。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 500, with_policy=True, policy_interval=10)
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    @pytest.mark.slow
    def test_500_ticks_checkpoint_enrichment(self):
        """500ティック中、100ティック毎に enrichment を生成し毎回正常。"""
        orch = PsycheOrchestrator()
        enrichments = []
        for i in range(500):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
                text=f"テスト入力{i}",
            )
            orch.post_response_update(percept, delta_time=1.0)
            if (i + 1) % 10 == 0:
                orch.select_policy_dict(percept, [])
            if (i + 1) % 100 == 0:
                e = orch.get_prompt_enrichment()
                assert isinstance(e, str) and len(e) > 0, (
                    f"Enrichment at tick {i+1} is empty or not a string"
                )
                enrichments.append(e)
        assert len(enrichments) == 5  # 100, 200, 300, 400, 500


# ══════════════════════════════════════════════════════════════════
# グループB: 中間保存・復元（永続化整合性）
# ══════════════════════════════════════════════════════════════════


class Test500TickSaveLoadResilience:
    """100ティックごとの保存・復元サイクルを5回繰り返して500ティック完走する検証。"""

    @pytest.mark.slow
    def test_500_ticks_with_save_load_cycles(self, tmp_path):
        """100ティックごとに save->新規インスタンス->load->続行 を繰り返し500ティック完走。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )
            expected_tick = (cycle + 1) * 100
            assert orch.tick_count == expected_tick, (
                f"Tick count mismatch at cycle {cycle}: "
                f"expected {expected_tick}, got {orch.tick_count}"
            )
            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                loaded = orch.load()
                assert loaded is True, (
                    f"Load failed at cycle {cycle}"
                )
                assert orch.tick_count == expected_tick, (
                    f"Tick count after load mismatch at cycle {cycle}: "
                    f"expected {expected_tick}, got {orch.tick_count}"
                )

        assert orch.tick_count == 500

    @pytest.mark.slow
    def test_500_ticks_save_load_enrichment_valid(self, tmp_path):
        """500ティック(save/load 5回)後のenrichmentが正常に生成されること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )
            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                orch.load()

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 100
        assert "[内面]" in enrichment

    @pytest.mark.slow
    def test_500_ticks_save_load_policy_valid(self, tmp_path):
        """500ティック(save/load 5回)後のpolicy選択が正常動作すること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )
            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                orch.load()

        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    @pytest.mark.slow
    def test_500_ticks_save_load_emotions_in_range(self, tmp_path):
        """500ティック(save/load 5回)後も全感情値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )
            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                orch.load()

        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range at tick 500 (with save/load): {val}"
            )

    @pytest.mark.slow
    def test_500_ticks_save_load_drives_in_range(self, tmp_path):
        """500ティック(save/load 5回)後も全ドライブ値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )
            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                orch.load()

        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, (
                f"Drive {name} out of range at tick 500 (with save/load): {val}"
            )

    @pytest.mark.slow
    def test_500_ticks_save_load_checkpoint_enrichment(self, tmp_path):
        """save/load 5回の各チェックポイントでenrichmentが正常。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        checkpoint_enrichments = []

        for cycle in range(5):
            _run_ticks(
                orch, 100,
                with_policy=True, policy_interval=10,
                start_offset=cycle * 100,
            )

            e = orch.get_prompt_enrichment()
            assert isinstance(e, str) and len(e) > 0, (
                f"Enrichment at checkpoint {cycle+1} is empty or not a string"
            )
            checkpoint_enrichments.append(len(e))

            orch.save()

            if cycle < 4:
                orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
                orch.load()

        assert len(checkpoint_enrichments) == 5


# ══════════════════════════════════════════════════════════════════
# グループC: 統計記録（非判定・事実記述のみ）
# ══════════════════════════════════════════════════════════════════


class Test500TickStatisticsRecord:
    """500ティック中の統計記録を収集し標準出力する。

    全テストはassertを使わない。
    事実の記録と出力のみを行い、pass/fail判定を含めない。
    テストフレームワーク上は常にpassする。
    """

    @pytest.mark.slow
    def test_record_policy_label_diversity(self, capsys):
        """500ティック中の方針選択ラベルの種類を5区間ごとに記録し標準出力する。"""
        orch = PsycheOrchestrator()
        stats = _run_ticks_with_stats(orch, 500, policy_interval=10)

        print("\n=== 500 tick policy label diversity record ===")
        for segment, data in sorted(stats.items()):
            print(
                f"  segment {segment}: "
                f"label_count={data['policy_label_count']}, "
                f"labels={data['policy_labels']}"
            )
        print("=== end ===")

    @pytest.mark.slow
    def test_record_enrichment_size(self, capsys):
        """500ティック中のenrichment文字数を5区間ごとに記録し標準出力する。"""
        orch = PsycheOrchestrator()
        stats = _run_ticks_with_stats(orch, 500, policy_interval=10)

        print("\n=== 500 tick enrichment size record ===")
        for segment, data in sorted(stats.items()):
            print(
                f"  segment {segment}: "
                f"enrichment_chars={data['enrichment_char_count']}"
            )
        print("=== end ===")

    @pytest.mark.slow
    def test_record_accumulation_counts(self, capsys):
        """500ティック中の蓄積構造の件数推移を5区間ごとに記録し標準出力する。"""
        orch = PsycheOrchestrator()
        stats = _run_ticks_with_stats(orch, 500, policy_interval=10)

        print("\n=== 500 tick accumulation counts record ===")
        for segment, data in sorted(stats.items()):
            print(f"  segment {segment}:")
            for acc_name, acc_count in sorted(data['accumulation_counts'].items()):
                print(f"    {acc_name}: {acc_count}")
        print("=== end ===")
