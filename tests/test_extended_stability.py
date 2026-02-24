"""
tests/test_extended_stability.py - 拡張安定性・境界条件・経路切替テスト

C5-9: テスト基盤の更なる拡充

3領域をカバー:
1. 200ティック長時間安定性テスト — メモリリーク・FIFO溢れ検出
2. 境界条件テスト — 各モジュールの蓄積上限・FIFO・ウィンドウサイズの境界値
3. 経路切替パターンテスト — 画面→テキスト→自発→画面の入力経路切替時の状態整合性

アサーションは構造的制約に限定し、特定の出力値（感情値の収束先等）は検証しない。
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
) -> None:
    """指定ティック数だけ多様な感情入力で更新する。

    with_policy=True の場合、policy_interval ティック毎に select_policy_dict を呼ぶ。
    """
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)
        if with_policy and (i + 1) % policy_interval == 0:
            orch.select_policy_dict(percept, [])


# ══════════════════════════════════════════════════════════════════
# 領域1: 200ティック長時間安定性テスト
# ══════════════════════════════════════════════════════════════════


class TestLongRunStability200:
    """200ティック長時間安定性テスト。

    構造的制約のアサーションに限定:
    - クラッシュしない
    - メモリリークしない（蓄積系が上限を超えない）
    - FIFOが溢れない
    - 全感情値・ドライブ値が 0-1 範囲内
    """

    def test_200_ticks_no_exception(self):
        """200ティック連続実行で例外が発生しないこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True, policy_interval=10)
        assert orch.tick_count == 200

    def test_200_ticks_emotions_in_range(self):
        """200ティック後も全感情値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range at tick 200: {val}"
            )

    def test_200_ticks_drives_in_range(self):
        """200ティック後も全ドライブ値が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, (
                f"Drive {name} out of range at tick 200: {val}"
            )

    def test_200_ticks_fear_in_range(self):
        """200ティック後も fear_level が 0-1 の範囲内。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        assert 0.0 <= orch.fear_level <= 1.0

    def test_200_ticks_expectation_diff_log_bounded(self):
        """200ティック後に expectation_action_diff_log が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        assert len(orch._expectation_action_diff_log) <= 200, (
            "expectation_action_diff_log should be bounded after 200 ticks"
        )

    def test_200_ticks_temporal_cognition_bounded(self):
        """200ティック後に temporal_cognition の蓄積が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._temporal_cognition is not None:
            state = orch._temporal_cognition.state
            if hasattr(state, 'elapsed_records'):
                assert len(state.elapsed_records) <= 300, (
                    "temporal_cognition elapsed_records should be bounded"
                )

    def test_200_ticks_action_result_pairs_bounded(self):
        """200ティック後に action_result_observer の pairs が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._action_result_observer is not None:
            state = orch._action_result_observer.state
            if hasattr(state, 'pairs'):
                assert len(state.pairs) <= 200, (
                    "action_result pairs should be bounded at 200"
                )

    def test_200_ticks_emotion_cooccurrence_bounded(self):
        """200ティック後に emotion_cooccurrence の records が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._emotion_cooccurrence_processor is not None:
            state = orch._emotion_cooccurrence_processor.state
            if hasattr(state, 'records'):
                assert len(state.records) <= 100, (
                    "emotion_cooccurrence records should be bounded"
                )

    def test_200_ticks_interaction_accumulation_bounded(self):
        """200ティック後に interaction_accumulation の pairs が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._interaction_accumulation is not None:
            state = orch._interaction_accumulation.state
            if hasattr(state, 'pairs'):
                assert len(state.pairs) <= 200, (
                    "interaction_accumulation pairs should be bounded"
                )

    def test_200_ticks_contradiction_bounded(self):
        """200ティック後に contradiction_processor の窓が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._contradiction_processor is not None:
            state = orch._contradiction_processor.state
            if hasattr(state, 'window'):
                assert len(state.window) <= 100, (
                    "contradiction window should be bounded"
                )

    def test_200_ticks_drive_variation_bounded(self):
        """200ティック後に drive_variation の sliding_window が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._drive_variation_processor is not None:
            state = orch._drive_variation_processor.state
            if hasattr(state, 'sliding_window'):
                assert len(state.sliding_window) <= 100, (
                    "drive_variation sliding_window should be bounded"
                )

    def test_200_ticks_emotional_backdrop_bounded(self):
        """200ティック後に emotional_backdrop の sliding_window が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._emotional_backdrop_processor is not None:
            state = orch._emotional_backdrop_processor.state
            if hasattr(state, 'sliding_window'):
                assert len(state.sliding_window) <= 100, (
                    "emotional_backdrop sliding_window should be bounded"
                )

    def test_200_ticks_introspection_cross_section_bounded(self):
        """200ティック後に introspection_cross_section のスナップショット窓が上限以下。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        if orch._introspection_cross_section is not None:
            state = orch._introspection_cross_section._state
            if hasattr(state, 'snapshot_window'):
                assert len(state.snapshot_window) <= 50, (
                    "introspection_cross_section snapshot_window should be bounded"
                )

    def test_200_ticks_enrichment_valid(self):
        """200ティック後も enrichment が正常に生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 100
        assert "[内面]" in enrichment

    def test_200_ticks_enrichment_bounded_size(self):
        """200ティック後も enrichment のサイズが妥当な範囲。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        enrichment = orch.get_prompt_enrichment()
        assert len(enrichment) <= 80000, (
            f"Enrichment too large at 200 ticks: {len(enrichment)} chars"
        )

    def test_200_ticks_policy_selection_valid(self):
        """200ティック後も select_policy_dict が正常動作。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 200, with_policy=True)
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_200_ticks_save_load_round_trip(self, tmp_path):
        """200ティック後の save -> load が正常に動作。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch, 200, with_policy=True)
        orch.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 200

        # load 後に操作可能
        _run_ticks(orch2, 10)
        assert orch2.tick_count == 210
        enrichment = orch2.get_prompt_enrichment()
        assert len(enrichment) > 0

    def test_200_ticks_midway_save_load(self, tmp_path):
        """100ティック毎に save/load を挟んでも 200 ティック完走。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch, 100, with_policy=True)
        assert orch.tick_count == 100
        orch.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 100
        _run_ticks(orch2, 100, with_policy=True)
        assert orch2.tick_count == 200

        # 全公開 API が正常動作
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_200_ticks_enrichment_periodic_check(self):
        """200ティック中、50ティック毎に enrichment を生成し毎回正常。"""
        orch = PsycheOrchestrator()
        enrichments = []
        for i in range(200):
            idx = i % len(EMOTIONS)
            percept = _make_percept(
                emotion=EMOTIONS[idx],
                valence=VALENCES[idx],
            )
            orch.post_response_update(percept, delta_time=1.0)
            if (i + 1) % 10 == 0:
                orch.select_policy_dict(percept, [])
            if (i + 1) % 50 == 0:
                e = orch.get_prompt_enrichment()
                assert isinstance(e, str) and len(e) > 0
                enrichments.append(e)
        assert len(enrichments) == 4  # 50, 100, 150, 200

    def test_200_ticks_extreme_emotions(self):
        """200ティック中に極端な感情入力を繰り返しても範囲内に収まる。"""
        orch = PsycheOrchestrator()
        extreme_pairs = [
            ("happy", 1.0), ("sad", -1.0), ("angry", -1.0),
            ("loving", 1.0), ("scared", -1.0),
        ]
        for i in range(200):
            emotion, valence = extreme_pairs[i % len(extreme_pairs)]
            percept = _make_percept(emotion=emotion, valence=valence)
            orch.post_response_update(percept, delta_time=1.0)
            if (i + 1) % 10 == 0:
                orch.select_policy_dict(percept, [])

        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range after 200 extreme ticks: {val}"
            )
        drv = orch.psyche.drives.as_dict()
        for name, val in drv.items():
            assert 0.0 <= val <= 1.0, (
                f"Drive {name} out of range after 200 extreme ticks: {val}"
            )


# ══════════════════════════════════════════════════════════════════
# 領域2: 境界条件テスト
# ══════════════════════════════════════════════════════════════════


class TestBoundaryConditions:
    """各モジュールの蓄積上限・FIFO・ウィンドウサイズの境界値テスト。

    アサーションは構造的制約に限定。
    """

    def test_all_emotions_at_zero(self):
        """全感情値が 0.0 の状態でも正常動作する。"""
        orch = PsycheOrchestrator()
        # neutral (valence=0.0) で 10 ティック
        for _ in range(10):
            percept = _make_percept(emotion="neutral", valence=0.0)
            orch.post_response_update(percept, delta_time=0.0)
        assert orch.tick_count == 10
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch.select_policy_dict(_make_percept(emotion="neutral", valence=0.0), [])
        assert isinstance(policy, dict)

    def test_all_emotions_at_max_valence(self):
        """valence=1.0 の状態でも正常動作する。"""
        orch = PsycheOrchestrator()
        for _ in range(10):
            percept = _make_percept(emotion="happy", valence=1.0)
            orch.post_response_update(percept, delta_time=1.0)
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, f"Emotion {name} out of range: {val}"

    def test_all_emotions_at_min_valence(self):
        """valence=-1.0 の状態でも正常動作する。"""
        orch = PsycheOrchestrator()
        for _ in range(10):
            percept = _make_percept(emotion="sad", valence=-1.0)
            orch.post_response_update(percept, delta_time=1.0)
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, f"Emotion {name} out of range: {val}"

    def test_delta_time_zero(self):
        """delta_time=0.0 でも正常動作する。"""
        orch = PsycheOrchestrator()
        for _ in range(10):
            percept = _make_percept()
            orch.post_response_update(percept, delta_time=0.0)
        assert orch.tick_count == 10
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_delta_time_large(self):
        """delta_time が大きい値でも正常動作する（長い沈黙後の復帰を想定）。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        # 大きな delta_time で 2 ティック目
        orch.post_response_update(percept, delta_time=3600.0)
        assert orch.tick_count == 2
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range with large delta_time: {val}"
            )

    def test_empty_text_percept(self):
        """空テキストの Percept でも正常動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept(text="")
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_long_text_percept(self):
        """非常に長いテキストの Percept でも正常動作する。"""
        orch = PsycheOrchestrator()
        long_text = "テスト" * 10000
        percept = _make_percept(text=long_text)
        orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 1

    def test_cold_start_all_apis(self):
        """全デフォルト値（cold-start）で全公開 API が動作する。"""
        orch = PsycheOrchestrator()
        # ティック 0 で全 API を呼ぶ
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        policy = orch.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)
        result = orch.check_spontaneous_activation()
        # spontaneous は初回では起動しないかもしれないが、例外なし
        # (result は None or SpontaneousResult)

    def test_cold_start_save_load(self, tmp_path):
        """ティック 0 で save/load が正常動作する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch.save()
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 0

    def test_rapid_consecutive_policy_selections(self):
        """同一ティック内で select_policy_dict を連続10回呼んでもエラーなし。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        for _ in range(10):
            policy = orch.select_policy_dict(percept, [])
            assert isinstance(policy, dict)
            assert "policy_label" in policy

    def test_single_emotion_repeated(self):
        """同一感情入力の繰り返しでも正常動作する。"""
        orch = PsycheOrchestrator()
        percept = _make_percept(emotion="happy", valence=0.7)
        for _ in range(30):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 30
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0

    def test_alternating_extreme_emotions(self):
        """極端に交互する感情入力でも正常動作する。"""
        orch = PsycheOrchestrator()
        for i in range(30):
            if i % 2 == 0:
                percept = _make_percept(emotion="happy", valence=1.0)
            else:
                percept = _make_percept(emotion="sad", valence=-1.0)
            orch.post_response_update(percept, delta_time=1.0)
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0

    def test_recalled_memories_empty_list(self):
        """recalled_memories が空リストでも正常動作する。"""
        orch = PsycheOrchestrator()
        orch.set_recalled_memories([])
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)

    def test_recalled_memories_large_list(self):
        """recalled_memories が大量でも正常動作する。"""
        orch = PsycheOrchestrator()
        memories = [
            {"summary": f"記憶{i}", "date": "2026-01-01", "keywords": [f"kw{i}"]}
            for i in range(100)
        ]
        orch.set_recalled_memories(memories)
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, memories)
        assert isinstance(policy, dict)

    def test_fifo_boundary_at_exact_limit(self):
        """蓄積系がちょうど上限に達する付近でも正常動作する。

        50ティック（多くのFIFOの上限に近い）で構造的整合性を確認。
        """
        orch = PsycheOrchestrator()
        _run_ticks(orch, 50, with_policy=True, policy_interval=3)
        # 蓄積系の境界チェック
        if orch._emotion_cooccurrence_processor is not None:
            state = orch._emotion_cooccurrence_processor.state
            if hasattr(state, 'records'):
                # 上限以下であること（上限は通常50）
                assert len(state.records) <= 50 + 5, (
                    "emotion_cooccurrence records near boundary"
                )

    def test_enrichment_after_no_policy_selection(self):
        """select_policy_dict を一度も呼ばずに enrichment を生成。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 20)
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0
        assert "[内面]" in enrichment

    def test_multiple_user_ids(self):
        """異なる user_id での post_response_update が正常動作する。"""
        orch = PsycheOrchestrator()
        user_ids = ["viewer", "user_a", "user_b", "viewer"]
        for i, uid in enumerate(user_ids):
            percept = _make_percept(
                emotion=EMOTIONS[i % len(EMOTIONS)],
                valence=VALENCES[i % len(VALENCES)],
            )
            orch.post_response_update(percept, delta_time=1.0, user_id=uid)
        assert orch.tick_count == 4


# ══════════════════════════════════════════════════════════════════
# 領域3: 経路切替パターンテスト
# ══════════════════════════════════════════════════════════════════


class TestPathwaySwitching:
    """画面→テキスト→自発→画面の入力経路切替時の状態整合性テスト。

    orchestrator の3経路:
    1. 画面経路: post_response_update(percept)
    2. テキスト経路: process_text_input(text) → post_response_update
    3. 自発経路: check_spontaneous_activation()
    """

    def test_screen_to_text_switch(self):
        """画面入力→テキスト入力への切替でエラーなし。"""
        orch = PsycheOrchestrator()
        # 画面入力 5 ティック
        _run_ticks(orch, 5)
        # テキスト入力
        result = orch.process_text_input("テキスト入力テスト", sender_id="user_a")
        # テキスト入力の結果に関わらず、画面入力に戻る
        _run_ticks(orch, 5)
        assert orch.tick_count == 10

    def test_text_to_screen_switch(self):
        """テキスト入力→画面入力への切替でエラーなし。"""
        orch = PsycheOrchestrator()
        # テキスト入力
        orch.process_text_input("テキスト入力テスト1", sender_id="user_a")
        orch.process_text_input("テキスト入力テスト2", sender_id="user_a")
        # 画面入力
        _run_ticks(orch, 10)
        assert orch.tick_count == 10
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_screen_to_spontaneous_switch(self):
        """画面入力→自発起動チェックへの切替でエラーなし。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # 自発起動チェック
        result = orch.check_spontaneous_activation()
        # result は None or SpontaneousResult
        # 画面入力に戻る
        _run_ticks(orch, 5)
        assert orch.tick_count == 10

    def test_spontaneous_to_screen_switch(self):
        """自発起動チェック→画面入力への切替でエラーなし。"""
        orch = PsycheOrchestrator()
        # 自発起動チェック（ティック0で）
        result = orch.check_spontaneous_activation()
        # 画面入力
        _run_ticks(orch, 10)
        assert orch.tick_count == 10

    def test_full_pathway_rotation(self):
        """画面→テキスト→自発→画面の完全ローテーション。"""
        orch = PsycheOrchestrator()

        # Phase 1: 画面入力 5 ティック
        _run_ticks(orch, 5)
        assert orch.tick_count == 5

        # Phase 2: テキスト入力
        orch.process_text_input("経路切替テスト", sender_id="user_a")

        # Phase 3: 画面入力 + ポリシー選択
        percept = _make_percept()
        orch.post_response_update(percept, delta_time=1.0)
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)

        # Phase 4: 自発起動チェック
        result = orch.check_spontaneous_activation()

        # Phase 5: 画面入力に戻る
        _run_ticks(orch, 5)
        assert orch.tick_count == 11

        # 全 API が正常
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_repeated_pathway_rotation(self):
        """経路ローテーションを 5 回繰り返してもエラーなし。"""
        orch = PsycheOrchestrator()
        for cycle in range(5):
            # 画面入力
            percept = _make_percept(
                emotion=EMOTIONS[cycle % len(EMOTIONS)],
                valence=VALENCES[cycle % len(VALENCES)],
                text=f"画面{cycle}",
            )
            orch.post_response_update(percept, delta_time=1.0)

            # テキスト入力
            orch.process_text_input(f"テキスト{cycle}", sender_id=f"user_{cycle}")

            # 自発起動チェック
            orch.check_spontaneous_activation()

            # ポリシー選択
            orch.select_policy_dict(percept, [])

        assert orch.tick_count == 5
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_text_input_does_not_increment_tick(self):
        """process_text_input は tick_count を増加させないこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        tick_before = orch.tick_count
        orch.process_text_input("テスト", sender_id="user_a")
        assert orch.tick_count == tick_before, (
            "process_text_input should not increment tick_count"
        )

    def test_spontaneous_does_not_increment_tick(self):
        """check_spontaneous_activation は tick_count を増加させないこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        tick_before = orch.tick_count
        orch.check_spontaneous_activation()
        assert orch.tick_count == tick_before, (
            "check_spontaneous_activation should not increment tick_count"
        )

    def test_text_input_between_ticks(self):
        """ティック間にテキスト入力を挟んでも状態整合性が保たれる。"""
        orch = PsycheOrchestrator()
        for i in range(20):
            percept = _make_percept(
                emotion=EMOTIONS[i % len(EMOTIONS)],
                valence=VALENCES[i % len(VALENCES)],
            )
            orch.post_response_update(percept, delta_time=1.0)
            # ティック間にテキスト入力
            if i % 3 == 0:
                orch.process_text_input(f"割り込みテキスト{i}", sender_id="user_a")
        assert orch.tick_count == 20
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0

    def test_spontaneous_between_ticks(self):
        """ティック間に自発起動チェックを挟んでも状態整合性が保たれる。"""
        orch = PsycheOrchestrator()
        for i in range(20):
            percept = _make_percept(
                emotion=EMOTIONS[i % len(EMOTIONS)],
                valence=VALENCES[i % len(VALENCES)],
            )
            orch.post_response_update(percept, delta_time=1.0)
            # ティック間に自発起動チェック
            if i % 4 == 0:
                orch.check_spontaneous_activation()
        assert orch.tick_count == 20
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0

    def test_all_three_between_ticks(self):
        """ティック間にテキスト+自発+ポリシー選択を全て挟む。"""
        orch = PsycheOrchestrator()
        for i in range(20):
            percept = _make_percept(
                emotion=EMOTIONS[i % len(EMOTIONS)],
                valence=VALENCES[i % len(VALENCES)],
            )
            orch.post_response_update(percept, delta_time=1.0)
            # 毎ティック全経路チェック
            orch.process_text_input(f"テキスト{i}", sender_id="user_a")
            orch.check_spontaneous_activation()
            if (i + 1) % 5 == 0:
                policy = orch.select_policy_dict(percept, [])
                assert isinstance(policy, dict)
        assert orch.tick_count == 20

    def test_pathway_switch_with_save_load(self, tmp_path):
        """経路切替 + save/load の組み合わせでもエラーなし。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        # 画面入力
        _run_ticks(orch, 5, with_policy=True)
        # テキスト入力
        orch.process_text_input("テスト", sender_id="user_a")
        # 自発起動チェック
        orch.check_spontaneous_activation()
        # save
        orch.save()

        # load
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 5

        # load 後に経路切替
        orch2.process_text_input("テスト2", sender_id="user_a")
        orch2.check_spontaneous_activation()
        _run_ticks(orch2, 5, with_policy=True)
        assert orch2.tick_count == 10

    def test_notify_self_output_between_pathways(self):
        """経路切替間に notify_self_output を呼んでもエラーなし。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 3)
        policy = orch.select_policy_dict(_make_percept(), [])
        label = policy.get("policy_label", "test")
        # notify_self_output
        orch.notify_self_output("テスト応答テキスト", policy_label=label)
        # テキスト入力
        orch.process_text_input("返信テスト", sender_id="user_a")
        # 画面入力
        _run_ticks(orch, 3)
        assert orch.tick_count == 6

    def test_mixed_emotions_across_pathways(self):
        """異なる経路で異なる感情を入力しても状態整合性が保たれる。"""
        orch = PsycheOrchestrator()

        # 画面: happy
        for _ in range(5):
            orch.post_response_update(
                _make_percept(emotion="happy", valence=0.9),
                delta_time=1.0,
            )

        # テキスト: sad context
        orch.process_text_input("悲しいテキスト", sender_id="user_a")

        # 画面: angry
        for _ in range(5):
            orch.post_response_update(
                _make_percept(emotion="angry", valence=-0.8),
                delta_time=1.0,
            )

        assert orch.tick_count == 10
        emo = orch.psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range after mixed pathway input: {val}"
            )

        # enrichment と policy が正常
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)
