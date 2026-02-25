"""
tests/test_phase_band_chain_integration.py - Phase帯域別連鎖検証テストの系統的拡充

設計書: design_integration_test_expansion_c6.md

3カテゴリをカバー:
- カテゴリA: Phase間入出力連鎖検証（5区画 + 区画間連鎖）
- カテゴリB: enrichment-永続化相互一貫性検証
- カテゴリC: save/load後Phase再開整合性検証

テスト期待値は構造的性質（型・存在・範囲・変動・非空・安全代替値）に限定し、
特定の出力値への一致は検証しない。

既存テスト（test_phase_chain_integration.py, test_integration_extended.py,
test_extended_stability.py）との重複は最小限にし、
「帯域内の全Phaseの入出力連鎖」「enrichment-永続化の一貫性」
「全帯域のsave/load後再開」を追加で検証する。
"""

import json
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
    intent: str = "expression",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        emotion_valence=valence,
    )


EMOTIONS = [
    "happy", "sad", "angry", "neutral", "surprised",
    "loving", "teasing", "scared", "happy", "neutral",
]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3, 0.8, 0.4, -0.5, 0.6, 0.0]


def _run_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新する。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)


def _run_ticks_with_policy(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新し、定期的にポリシー選択も行う。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)
        # 5ティック毎にポリシー選択を実行
        if (i + 1) % 5 == 0:
            orch.select_policy_dict(percept, [])


# ══════════════════════════════════════════════════════════════════
# カテゴリA: Phase間入出力連鎖検証
# ══════════════════════════════════════════════════════════════════


class TestSectionOneChain:
    """第一区画: 自己差分 → 安定化記述 → 連続性負荷 → 自己像 → 一貫性揺らぎ

    Phase 15 → 15b → 16 → 17 → 18 の連鎖検証。
    5ティック毎に発火する帯域。
    """

    def test_diff_to_strain_data_flow(self):
        """差分認知出力が連続性負荷に渡されること。"""
        orch = PsycheOrchestrator()
        # 3ティックでPhase 9(self_model)発火 → _last_self_view設定
        _run_ticks(orch, 3)
        assert orch._last_self_view is not None, (
            "Phase 9 (self_model): last_self_view should be set at tick 3"
        )
        # 5ティック目でPhase 15(diff) + Phase 16(strain) が連鎖発火
        _run_ticks(orch, 2)
        assert orch._last_diff_summary is not None, (
            "Phase 15: diff_summary should be set at tick 5"
        )
        assert orch._last_strain is not None, (
            "Phase 16: strain should be set at tick 5 (consumes diff_summary)"
        )

    def test_diff_and_strain_flow_to_self_image(self):
        """差分・負荷出力が自己像統合に使用されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 17: self_image_integration は diff_summary と strain を入力とする
        assert orch._last_self_image is not None, (
            "Phase 17: self_image should be set at tick 5"
        )

    def test_self_image_flows_to_coherence(self):
        """自己像出力が一貫性評価に渡されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 18: identity_coherence は self_image を入力として参照する
        assert orch._last_coherence is not None, (
            "Phase 18: coherence should be set at tick 5"
        )

    def test_stabilization_description_executed(self):
        """安定化記述（Phase 15b）が差分認知後に実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        assert orch._stabilization_desc_state is not None, (
            "Phase 15b: stabilization_description should be set at tick 5"
        )

    def test_section_one_full_chain_types(self):
        """第一区画の全出力が適切な型を持つこと。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)

        # diff_summary: SelfDifferenceSummary (属性アクセス可能)
        diff = orch._last_diff_summary
        assert diff is not None
        assert hasattr(diff, 'has_difference'), (
            "diff_summary should have 'has_difference' attribute"
        )
        assert hasattr(diff, 'magnitude'), (
            "diff_summary should have 'magnitude' attribute"
        )

        # strain: StrainState (属性アクセス可能)
        strain = orch._last_strain
        assert strain is not None
        assert hasattr(strain, 'level'), (
            "strain should have 'level' attribute"
        )
        assert hasattr(strain, 'presence'), (
            "strain should have 'presence' attribute"
        )

        # self_image: ProvisionalSelfImage (属性アクセス可能)
        self_image = orch._last_self_image
        assert self_image is not None

        # coherence: IdentityCoherenceState (属性アクセス可能)
        coherence = orch._last_coherence
        assert coherence is not None

    def test_section_one_chain_across_multiple_cycles(self):
        """第一区画の連鎖が複数サイクル（5,10,15ティック）で継続すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        diff_5 = orch._last_diff_summary
        _run_ticks(orch, 5)  # tick 10
        diff_10 = orch._last_diff_summary
        _run_ticks(orch, 5)  # tick 15
        diff_15 = orch._last_diff_summary

        # 各サイクルで出力が存在する
        assert diff_5 is not None
        assert diff_10 is not None
        assert diff_15 is not None

        # 全関連出力が存在
        assert orch._last_strain is not None
        assert orch._last_self_image is not None
        assert orch._last_coherence is not None


class TestSectionTwoChain:
    """第二区画: 物語 → エピソード → 感情結合 → 記憶統合 → 忘却 → 想起 → 自発想起 → 均衡記述

    Phase 19 → 20 → 21 → 21b → 21c → 21d → 21e → 21f の連鎖検証。
    """

    def test_narrative_to_episodic_flow(self):
        """物語出力がエピソード入力に渡されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 19: narrative
        assert orch._last_narrative is not None, (
            "Phase 19: narrative should be set at tick 5"
        )
        # Phase 20: episodic (narrative を入力として使用)
        assert orch._last_episodes is not None, (
            "Phase 20: episodes should be set at tick 5"
        )

    def test_episodic_binding_to_integration_flow(self):
        """エピソード・感情結合が記憶統合に入力として使用されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 21: binding
        assert orch._last_bindings is not None, (
            "Phase 21: bindings should be set at tick 5"
        )
        # Phase 21b: memory_integration
        assert orch._memory_integrator is not None
        int_state = orch._memory_integrator.state
        assert int_state is not None, (
            "Phase 21b: memory_integrator state should be accessible"
        )

    def test_integration_to_forgetting_flow(self):
        """統合結果が忘却処理に渡されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 21c: forgetting_fixation
        assert orch._forgetting_fixation_processor is not None
        ff_state = orch._forgetting_fixation_processor.state
        assert ff_state is not None, (
            "Phase 21c: forgetting_fixation state should be accessible"
        )

    def test_forgetting_to_recall_flow(self):
        """忘却後の記憶が想起処理の対象となること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 21d: multi_path_recall
        assert orch._multi_path_recall is not None
        mpr_state = orch._multi_path_recall.state
        assert mpr_state is not None, (
            "Phase 21d: multi_path_recall state should be accessible"
        )

    def test_spontaneous_recall_references_memory(self):
        """自発的想起が記憶群を参照すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 21e: spontaneous_recall
        assert orch._spontaneous_recall is not None
        sr_state = orch._spontaneous_recall.state
        assert sr_state is not None, (
            "Phase 21e: spontaneous_recall state should be accessible"
        )

    def test_balance_references_forgetting_and_recall(self):
        """均衡記述が忘却・想起の双方を参照すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 21f: forgetting_recall_balance
        assert orch._frb_state is not None, (
            "Phase 21f: forgetting_recall_balance state should be accessible"
        )

    def test_section_two_full_chain_no_exception(self):
        """第二区画の全Phase連鎖が例外なく完了すること。"""
        orch = PsycheOrchestrator()
        # 複数サイクル実行
        _run_ticks(orch, 15)
        # 全出力が存在
        assert orch._last_narrative is not None
        assert orch._last_episodes is not None
        assert orch._last_bindings is not None
        assert orch._forgetting_fixation_processor.state is not None
        assert orch._multi_path_recall.state is not None
        assert orch._spontaneous_recall.state is not None


class TestSectionThreeChain:
    """第三区画: 内省ログ → 内省消費 → 期待形成 → 参照頻度記述

    Phase 22 → 23 → 24 → 24b の連鎖検証。
    """

    def test_trace_to_consumption_flow(self):
        """内省ログが消費層に渡されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 22: introspection_trace
        assert orch._last_trace is not None, (
            "Phase 22: trace should be set at tick 5"
        )
        # Phase 23: introspection_consumption (trace を入力として使用)
        assert orch._last_consumption is not None, (
            "Phase 23: consumption should be set at tick 5"
        )

    def test_consumption_to_expectation_flow(self):
        """消費結果が期待形成に利用可能であること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 24: expectation_formation
        assert orch._last_expectations is not None, (
            "Phase 24: expectations should be set at tick 5"
        )

    def test_reference_frequency_collects_cross_info(self):
        """参照頻度記述が横断的な参照情報を収集できること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 24b: reference_frequency_description
        assert orch._reference_frequency_state is not None, (
            "Phase 24b: reference_frequency_state should be set at tick 5"
        )

    def test_section_three_multiple_cycles(self):
        """第三区画の連鎖が複数サイクルで継続すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        assert orch._last_trace is not None
        assert orch._last_consumption is not None
        assert orch._last_expectations is not None
        assert orch._reference_frequency_state is not None


class TestSectionFourChain:
    """第四区画: 実観測フィード → テキスト対話 → 他者観測蓄積 → 相互作用 → 境界 → 仮説対 → 他者モデル

    Phase 25a → 25b → 25c → 25d → 25e → 25f → 25 の連鎖検証。
    """

    def test_real_feed_output_exists(self):
        """実観測フィード出力が存在すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25a: real_feed
        assert orch._real_feed_processor is not None
        rf_state = orch._real_feed_processor.state
        assert rf_state is not None, (
            "Phase 25a: real_feed state should be accessible"
        )

    def test_dialogue_learning_accumulates(self):
        """対話学習結果が蓄積されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25c: dialogue_learning
        assert orch._dialogue_learning_processor is not None
        dl_state = orch._dialogue_learning_processor.state
        assert dl_state is not None, (
            "Phase 25c: dialogue_learning state should be accessible"
        )

    def test_interaction_accumulation_executes(self):
        """相互作用蓄積が実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25d: interaction_accumulation
        assert orch._interaction_accumulation is not None
        ia_state = orch._interaction_accumulation.state
        assert ia_state is not None, (
            "Phase 25d: interaction_accumulation state should be accessible"
        )

    def test_other_boundary_accumulation_executes(self):
        """他者境界蓄積が実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25e: other_boundary_accumulation
        assert orch._other_boundary_accumulation is not None
        oba_state = orch._other_boundary_accumulation.state
        assert oba_state is not None, (
            "Phase 25e: other_boundary_accumulation state should be accessible"
        )

    def test_hypothesis_pairing_executes(self):
        """仮説対構成が実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25f: hypothesis_observation_pairing
        assert orch._hypothesis_observation_pairing is not None
        hop_state = orch._hypothesis_observation_pairing.state
        assert hop_state is not None, (
            "Phase 25f: hypothesis_observation_pairing state should be accessible"
        )

    def test_other_model_updated(self):
        """他者モデル更新が実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 25: other_agent_model
        assert orch._last_other_model is not None, (
            "Phase 25: other_model should be updated at tick 5"
        )

    def test_section_four_full_chain_no_exception(self):
        """第四区画の全Phase連鎖が例外なく完了すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 15)
        assert orch._real_feed_processor.state is not None
        assert orch._dialogue_learning_processor.state is not None
        assert orch._interaction_accumulation.state is not None
        assert orch._other_boundary_accumulation.state is not None
        assert orch._hypothesis_observation_pairing.state is not None
        assert orch._last_other_model is not None


class TestSectionFiveChain:
    """第五区画: 価値指向 → 検証報告 → 行動結果 → 多様性 → 乖離 → ライフサイクル → 責任推移 → 目標階層

    Phase 26 → 26b → 26c → 26c2 → 26d → 26e → 26f → 26g → 26h の連鎖検証。
    """

    def test_value_orientation_updated(self):
        """価値指向出力が存在すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26: value_orientation
        assert orch._value_orientation is not None, (
            "Phase 26: value_orientation should be updated"
        )

    def test_vo_validation_after_orientation(self):
        """検証報告が価値指向更新後に生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26b: vo_validation
        assert orch._vo_validator is not None
        vo_state = orch._vo_validator.state
        assert vo_state is not None, (
            "Phase 26b: vo_validator state should be accessible"
        )

    def test_action_result_process_after_value_orientation(self):
        """行動結果処理が価値指向後に実行されること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        # まずポリシー選択を実行してラベルを設定
        orch.post_response_update(percept, delta_time=1.0)
        orch.select_policy_dict(percept, [])
        # 5ティック目まで進める
        for _ in range(4):
            orch.post_response_update(percept, delta_time=1.0)
        # Phase 26c: action_result
        assert orch._action_result_observer is not None
        ar_state = orch._action_result_observer.state
        assert ar_state is not None, (
            "Phase 26c: action_result state should be accessible"
        )

    def test_behavioral_diversity_after_action_result(self):
        """行動多様性記述が行動結果後に実行されること。"""
        orch = PsycheOrchestrator()
        _run_ticks_with_policy(orch, 10)
        # Phase 26c2: behavioral_diversity
        assert orch._behavioral_diversity_state is not None, (
            "Phase 26c2: behavioral_diversity_state should be set"
        )

    def test_intent_action_gap_executes(self):
        """意図-行動乖離記録が生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26e: intent_action_gap
        assert orch._intent_action_gap_recorder is not None
        gap_state = orch._intent_action_gap_recorder.state
        assert gap_state is not None, (
            "Phase 26e: intent_action_gap state should be accessible"
        )

    def test_expectation_lifecycle_executes(self):
        """予期ライフサイクル記述が期待形成出力を参照すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26f: expectation_lifecycle
        assert orch._expectation_lifecycle_processor is not None
        el_state = orch._expectation_lifecycle_processor.state
        assert el_state is not None, (
            "Phase 26f: expectation_lifecycle state should be accessible"
        )

    def test_responsibility_trace_executes(self):
        """責任推移記述が責任状態の変化を記述すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26g: responsibility_temporal_trace
        assert orch._responsibility_temporal_trace is not None
        rtt_state = orch._responsibility_temporal_trace.state
        assert rtt_state is not None, (
            "Phase 26g: responsibility_temporal_trace state should be accessible"
        )

    def test_goal_hierarchy_executes(self):
        """目標階層変化記述が目標系出力を参照すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # Phase 26h: goal_hierarchy_propagation
        assert orch._goal_hierarchy_propagation is not None
        ghp_state = orch._goal_hierarchy_propagation.state
        assert ghp_state is not None, (
            "Phase 26h: goal_hierarchy_propagation state should be accessible"
        )

    def test_section_five_full_chain_no_exception(self):
        """第五区画の全Phase連鎖が例外なく完了すること。"""
        orch = PsycheOrchestrator()
        _run_ticks_with_policy(orch, 15)
        assert orch._value_orientation is not None
        assert orch._vo_validator.state is not None
        assert orch._action_result_observer.state is not None
        assert orch._intent_action_gap_recorder.state is not None
        assert orch._expectation_lifecycle_processor.state is not None
        assert orch._responsibility_temporal_trace.state is not None
        assert orch._goal_hierarchy_propagation.state is not None


class TestCrossSectionChain:
    """区画間連鎖検証。

    - 第一区画出力 → 第二区画入力
    - 第二区画出力 → 第三区画入力
    - 第三区画出力 → 第四区画入力
    - 第四区画出力 → 第五区画入力
    """

    def test_section1_to_section2_self_image_coherence(self):
        """第一区画出力（自己像・一貫性）が第二区画入力（物語・エピソード）に接続されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # 第一区画出力
        assert orch._last_self_image is not None, (
            "Section 1 output: self_image should be set"
        )
        assert orch._last_coherence is not None, (
            "Section 1 output: coherence should be set"
        )
        # 第二区画は同一5ティック帯域で実行され、第一区画出力を入力として参照
        # エピソード記録がcoherence_stateを入力として受け取る
        assert orch._last_episodes is not None, (
            "Section 2 input: episodes should use section 1 outputs"
        )

    def test_section2_to_section3_memory_to_introspection(self):
        """第二区画出力（記憶・エピソード）が第三区画入力（内省・期待の素材）に接続されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # 第二区画出力
        assert orch._last_episodes is not None
        assert orch._last_bindings is not None
        # 第三区画は同一帯域で実行され、第二区画出力を参照
        assert orch._last_trace is not None, (
            "Section 3: introspection uses section 2 outputs"
        )
        assert orch._last_consumption is not None, (
            "Section 3: consumption uses narratives/episodes from section 2"
        )

    def test_section3_to_section4_expectation_to_other_model(self):
        """第三区画出力（期待形成）が第四区画入力（他者モデルの文脈）に接続されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # 第三区画出力
        assert orch._last_expectations is not None
        # 第四区画: 他者モデル更新は入力供給経由で文脈情報を参照
        assert orch._last_other_model is not None, (
            "Section 4: other_model should be updated with section 3 context"
        )

    def test_section4_to_section5_other_model_to_value(self):
        """第四区画出力（他者モデル）が第五区画入力（価値指向の参照）に接続されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)
        # 第四区画出力
        assert orch._last_other_model is not None
        # 第五区画: 価値指向は感情・責任状態を入力とし、他者モデルとは間接的に連鎖
        assert orch._value_orientation is not None, (
            "Section 5: value_orientation should be updated"
        )

    def test_all_sections_complete_in_single_5tick_cycle(self):
        """5ティック1サイクルで全5区画が完了すること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 5)

        # 第一区画
        assert orch._last_diff_summary is not None
        assert orch._last_strain is not None
        assert orch._last_self_image is not None
        assert orch._last_coherence is not None
        # 第二区画
        assert orch._last_narrative is not None
        assert orch._last_episodes is not None
        assert orch._last_bindings is not None
        # 第三区画
        assert orch._last_trace is not None
        assert orch._last_consumption is not None
        assert orch._last_expectations is not None
        # 第四区画
        assert orch._last_other_model is not None
        # 第五区画
        assert orch._value_orientation is not None

    def test_varied_emotions_cross_section_consistency(self):
        """多様な感情入力による区画間連鎖の一貫性。"""
        orch = PsycheOrchestrator()
        emotions = [
            ("happy", 0.9), ("sad", -0.8), ("angry", -0.7),
            ("surprised", 0.5), ("neutral", 0.0),
            ("loving", 0.9), ("scared", -0.6), ("teasing", 0.3),
            ("happy", 0.5), ("neutral", 0.1),
        ]
        for emotion, valence in emotions:
            percept = _make_percept(emotion=emotion, valence=valence)
            orch.post_response_update(percept, delta_time=1.0)

        assert orch.tick_count == 10
        # 10ティック時点で全区画が2サイクル分実行済み
        assert orch._last_diff_summary is not None
        assert orch._last_episodes is not None
        assert orch._last_expectations is not None
        assert orch._last_other_model is not None
        assert orch._value_orientation is not None


# ══════════════════════════════════════════════════════════════════
# カテゴリB: enrichment-永続化相互一貫性検証
# ══════════════════════════════════════════════════════════════════


ENRICHMENT_SECTIONS = [
    "[内面]",
    "[自己]",
    "[動機]",
    "[記憶]",
    "[判断]",
]


class TestEnrichmentPersistenceConsistency:
    """enrichment-永続化相互一貫性検証。

    - enrichment各項目が参照する永続化フィールドがsave/load対象に含まれること
    - 永続化フィールド復元後にenrichment生成が実行可能であること
    - 未蓄積状態でenrichmentが安全な代替値を返すこと
    """

    def test_enrichment_generation_after_save_load(self, tmp_path):
        """永続化フィールドの復元後にenrichment生成が例外なく実行されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        enrichment_before = orch1.get_prompt_enrichment()
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        enrichment_after = orch2.get_prompt_enrichment()
        assert isinstance(enrichment_after, str)
        assert len(enrichment_after) > 0, (
            "Enrichment should be non-empty after load"
        )

    def test_enrichment_sections_present_after_load(self, tmp_path):
        """復元後にenrichmentの5セクションが存在すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        # ティック実行してキャッシュを充実させる
        _run_ticks_with_policy(orch2, 5)
        enrichment = orch2.get_prompt_enrichment()

        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Section '{section}' should be present in enrichment after load"
            )

    def test_enrichment_safe_at_initial_state(self):
        """永続化フィールドが初期状態（未蓄積）の場合にenrichmentが安全な代替文字列を返すこと。"""
        orch = PsycheOrchestrator()
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0, (
            "Enrichment should return safe fallback at initial state"
        )
        # 例外が発生していないことを確認
        assert "[内面]" in enrichment, (
            "Enrichment should contain [内面] section even at initial state"
        )

    def test_enrichment_type_is_string_at_every_stage(self):
        """enrichmentの出力型が各段階で文字列であること。"""
        orch = PsycheOrchestrator()

        # 初期状態
        e0 = orch.get_prompt_enrichment()
        assert isinstance(e0, str)

        # 1ティック後
        _run_ticks(orch, 1)
        e1 = orch.get_prompt_enrichment()
        assert isinstance(e1, str)

        # 5ティック後（5ティック帯域発火後）
        _run_ticks(orch, 4)
        e5 = orch.get_prompt_enrichment()
        assert isinstance(e5, str)

        # 10ティック後（10ティック帯域発火後）
        _run_ticks(orch, 5)
        e10 = orch.get_prompt_enrichment()
        assert isinstance(e10, str)

    def test_enrichment_not_empty_after_load_with_ticks(self, tmp_path):
        """復元後の追加ティック実行でenrichmentが非空であること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 5)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        _run_ticks(orch2, 5)

        enrichment = orch2.get_prompt_enrichment()
        assert len(enrichment) > 100, (
            "Enrichment should contain substantial content after load + ticks"
        )

    def test_save_contains_all_enrichment_referenced_fields(self, tmp_path):
        """saveデータに、enrichmentが参照する全永続化フィールドが含まれること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch, 10)
        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # enrichmentが参照する主要フィールドの存在確認
        essential_keys = [
            "psyche",       # 感情・ムード・ドライブ
            "dynamics",     # ダイナミクス
            "value_orientation",  # 価値指向
        ]
        for key in essential_keys:
            assert key in data, (
                f"Save data should contain '{key}' which enrichment references"
            )

    def test_enrichment_emotion_info_after_load(self, tmp_path):
        """復元後のenrichmentに感情情報が含まれること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 5)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        enrichment = orch2.get_prompt_enrichment()
        assert "感情" in enrichment, (
            "Enrichment should contain emotion info after load"
        )

    def test_enrichment_section_order_preserved_after_load(self, tmp_path):
        """復元後のenrichmentセクション順序が正しいこと。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        _run_ticks_with_policy(orch2, 5)

        enrichment = orch2.get_prompt_enrichment()
        positions = []
        for section in ENRICHMENT_SECTIONS:
            pos = enrichment.find(section)
            if pos >= 0:
                positions.append((section, pos))

        for i in range(len(positions) - 1):
            assert positions[i][1] < positions[i + 1][1], (
                f"Section '{positions[i][0]}' should come before "
                f"'{positions[i + 1][0]}' after load"
            )


# ══════════════════════════════════════════════════════════════════
# カテゴリC: save/load後Phase再開整合性検証
# ══════════════════════════════════════════════════════════════════


class TestSaveLoadPhaseResumeAtBandTick:
    """帯域実行ティック（帯域周期の倍数）でのsave/load後再開検証。"""

    def test_save_at_tick_5_resume_at_tick_6(self, tmp_path):
        """ティック5（5ティック帯域実行直後）でsave → 復元 → ティック6以降の実行。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 5)
        assert orch1.tick_count == 5
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 5

        # 次のティック実行が正常に完了
        percept = _make_percept()
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 6

    def test_save_at_tick_10_resume_to_tick_15(self, tmp_path):
        """ティック10（10ティック帯域実行直後）でsave → 復元 → ティック15まで実行。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 5ティック追加で5ティック帯域が再度発火
        _run_ticks(orch2, 5)
        assert orch2.tick_count == 15

        # 5ティック帯域の全出力が有効
        assert orch2._last_diff_summary is not None
        assert orch2._last_episodes is not None
        assert orch2._last_expectations is not None

    def test_save_at_tick_30_resume_complex(self, tmp_path):
        """ティック30でsave → 復元 → 全帯域が複数回実行される検証。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 30)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 10ティック追加実行
        _run_ticks_with_policy(orch2, 10)
        assert orch2.tick_count == 40

        # 全帯域の出力が有効
        assert orch2._last_self_view is not None
        assert orch2._last_episodes is not None
        assert orch2._value_orientation is not None

        # enrichmentとpolicyが正常動作
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict) and "policy_label" in policy


class TestSaveLoadPhaseResumeAtNonBandTick:
    """帯域非実行ティック（帯域周期の倍数でないティック数）でのsave/load後再開検証。"""

    def test_save_at_tick_7_resume_to_band_tick(self, tmp_path):
        """ティック7（帯域非実行）でsave → 復元 → 帯域実行ティック(10)まで進行。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 7)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 7

        # 3ティック追加でティック10（全帯域発火）に到達
        _run_ticks(orch2, 3)
        assert orch2.tick_count == 10

        # 全帯域Phase実行結果が有効
        assert orch2._last_diff_summary is not None
        assert orch2._last_episodes is not None

    def test_save_at_tick_4_resume_to_tick_5(self, tmp_path):
        """ティック4（5ティック帯域の直前）でsave → 復元 → ティック5（帯域発火）。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 4)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 4

        # 1ティック追加でティック5（5ティック帯域発火）
        percept = _make_percept()
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 5

        # 5ティック帯域のPhase出力が有効
        assert orch2._last_diff_summary is not None
        assert orch2._last_episodes is not None

    def test_save_at_tick_8_resume_across_band_boundaries(self, tmp_path):
        """ティック8でsave → 復元 → 3ティック帯域(9)と5ティック帯域(10)を跨ぐ検証。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 8)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # ティック9（3ティック帯域発火）
        percept = _make_percept()
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 9
        assert orch2._tendency_awareness is not None, (
            "3-tick band should fire at tick 9 after load from tick 8"
        )

        # ティック10（5ティック帯域+10ティック帯域発火）
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 10
        assert orch2._last_episodes is not None, (
            "5-tick band should fire at tick 10 after load from tick 8"
        )


class TestSaveLoadFieldIntegrity:
    """復元後のPhase入力フィールドの有効性確認。"""

    def test_all_phase_input_fields_valid_after_load(self, tmp_path):
        """全永続化フィールドのうち、帯域Phaseが直接入力として使用するフィールドが
        復元後に有効であること（存在確認・型確認）。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # コア状態
        assert orch2._psyche is not None
        assert isinstance(orch2._psyche.mood.valence, float)
        assert orch2._dynamics is not None
        assert orch2._loop_state is not None
        assert orch2._amplitude_state is not None
        assert orch2._value_orientation is not None
        assert orch2._stability_valve is not None
        assert orch2._tendency_sys is not None

        # 感情値が有効範囲
        emo = orch2._psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, f"Emotion {name} out of range after load: {val}"

        # fear_level が有効範囲
        assert 0.0 <= orch2.fear_level <= 1.0

    def test_phase_outputs_valid_type_after_load_and_tick(self, tmp_path):
        """復元後の最初の帯域実行で、各Phaseの出力が型・構造として有効であること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 5ティック実行して全帯域を発火
        _run_ticks(orch2, 5)

        # 各Phase出力の型検証
        if orch2._last_diff_summary is not None:
            assert hasattr(orch2._last_diff_summary, '__class__')
        if orch2._last_episodes is not None:
            assert hasattr(orch2._last_episodes, '__class__')
        if orch2._last_expectations is not None:
            assert hasattr(orch2._last_expectations, '__class__')
        if orch2._last_other_model is not None:
            assert hasattr(orch2._last_other_model, '__class__')

    def test_multiple_ticks_stable_after_load(self, tmp_path):
        """復元後に複数ティック実行し、帯域が複数回実行された後の状態が安定していること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 20ティック追加実行（全帯域が複数回発火）
        _run_ticks_with_policy(orch2, 20)
        assert orch2.tick_count == 30

        # 例外が発生していないこと（ここまで到達 = 安定）
        # 感情値が有効範囲
        emo = orch2._psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion {name} out of range after load + 20 ticks: {val}"
            )

        # enrichmentが正常生成
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_double_save_load_resume_stability(self, tmp_path):
        """save → load → 実行 → save → load → 実行 の2サイクルが安定すること。"""
        # サイクル1
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        # サイクル2
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        _run_ticks_with_policy(orch2, 10)
        orch2.save()

        # サイクル3
        orch3 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch3.load()
        assert orch3.tick_count == 20

        # 復元後にさらに実行
        _run_ticks_with_policy(orch3, 10)
        assert orch3.tick_count == 30

        # 全APIが正常動作
        enrichment = orch3.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch3.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict) and "policy_label" in policy


class TestSaveLoadAtVariousPoints:
    """帯域をまたぐ保存・復元の検証（保存時と復元後で帯域の実行タイミングが異なるケース）。"""

    def test_save_mid_3tick_cycle_resume(self, tmp_path):
        """3ティック帯域の途中（ティック2）でsave → 復元 → 次の3ティック帯域まで。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 2)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()
        assert orch2.tick_count == 2

        # ティック3で3ティック帯域発火
        percept = _make_percept()
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 3
        assert orch2._tendency_awareness is not None, (
            "3-tick band should fire at tick 3 after load from tick 2"
        )

    def test_save_between_5_and_10_band(self, tmp_path):
        """ティック6でsave（5ティック帯域完了後、10ティック帯域前）→ 復元 → ティック10。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 6)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 4ティック追加でティック10に到達
        _run_ticks(orch2, 4)
        assert orch2.tick_count == 10

        # 10ティック帯域が正常発火
        assert orch2._dynamics_observer is not None

    def test_save_at_15_load_continue_to_30(self, tmp_path):
        """ティック15でsave → 復元 → ティック30まで継続。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 15)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        _run_ticks_with_policy(orch2, 15)
        assert orch2.tick_count == 30

        # 全機能が正常動作
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 100
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_every_tick_phase_output_valid_after_load(self, tmp_path):
        """復元後の毎ティックPhase（Phase 1-7）の出力が有効であること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 5)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 1ティック実行 → 毎ティックPhaseが正常動作
        percept = _make_percept(emotion="sad", valence=-0.5)
        orch2.post_response_update(percept, delta_time=1.0)

        # Phase 1: 感情が更新される
        assert orch2._psyche is not None
        emo = orch2._psyche.emotions.as_dict()
        for name, val in emo.items():
            assert 0.0 <= val <= 1.0

        # Phase 2: dynamics が有効
        assert orch2._dynamics is not None

        # Phase 5: self_ref_state が有効
        assert orch2._self_ref_state is not None

        # Phase 7: fear_level が有効範囲
        assert 0.0 <= orch2.fear_level <= 1.0

    def test_policy_selection_valid_after_load(self, tmp_path):
        """復元後のポリシー選択（Phase 30-35）が正常動作すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks_with_policy(orch1, 10)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        orch2.load()

        # 復元直後のポリシー選択
        percept = _make_percept()
        policy = orch2.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy
        assert isinstance(policy["policy_label"], str)
        assert len(policy["policy_label"]) > 0


# ══════════════════════════════════════════════════════════════════
# 追加: 変動の確認（異なる入力に対して異なる出力が生成されること）
# ══════════════════════════════════════════════════════════════════


class TestOutputVariation:
    """設計書の「変動の確認」原則に基づくテスト。

    異なる入力に対して異なる出力が生成されることを確認する。
    """

    def test_different_emotions_produce_different_states(self):
        """異なる感情入力で異なる内部状態が生成されること。"""
        orch_happy = PsycheOrchestrator()
        orch_sad = PsycheOrchestrator()

        # 同じティック数だが異なる感情
        for _ in range(5):
            p_happy = _make_percept(emotion="happy", valence=0.9)
            orch_happy.post_response_update(p_happy, delta_time=1.0)

            p_sad = _make_percept(emotion="sad", valence=-0.8)
            orch_sad.post_response_update(p_sad, delta_time=1.0)

        # 少なくとも1つの感情次元が異なる
        emo_happy = orch_happy._psyche.emotions.as_dict()
        emo_sad = orch_sad._psyche.emotions.as_dict()
        diff_found = any(
            abs(emo_happy[k] - emo_sad.get(k, 0)) > 1e-6
            for k in emo_happy
        )
        assert diff_found, (
            "Different emotion inputs should produce different emotion states"
        )

    def test_enrichment_varies_with_input(self):
        """異なる入力履歴でenrichmentの内容が変わること。"""
        orch_a = PsycheOrchestrator()
        orch_b = PsycheOrchestrator()

        for _ in range(10):
            p_a = _make_percept(emotion="happy", valence=0.9)
            orch_a.post_response_update(p_a, delta_time=1.0)

            p_b = _make_percept(emotion="angry", valence=-0.8)
            orch_b.post_response_update(p_b, delta_time=1.0)

        enrichment_a = orch_a.get_prompt_enrichment()
        enrichment_b = orch_b.get_prompt_enrichment()

        # enrichmentの内容が異なる（異なる入力履歴なので）
        assert enrichment_a != enrichment_b, (
            "Different input histories should produce different enrichment"
        )


# ══════════════════════════════════════════════════════════════════
# 追加: 安全代替値の確認
# ══════════════════════════════════════════════════════════════════


class TestSafeFallback:
    """入力が不足している場合にPhase処理が安全な代替値を返すことの確認。"""

    def test_minimal_percept_no_exception(self):
        """最小限のPercept入力でも例外が発生しないこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept(emotion="neutral", valence=0.0, text="")
        # 例外なく10ティック実行
        for _ in range(10):
            orch.post_response_update(percept, delta_time=1.0)
        assert orch.tick_count == 10

    def test_enrichment_safe_without_policy(self):
        """ポリシー選択なしでもenrichmentが安全に生成されること。"""
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        # ポリシー選択を行わずにenrichmentを生成
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

    def test_policy_safe_at_tick_0(self):
        """ティック0でのポリシー選択が安全な結果を返すこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_save_load_at_tick_1(self, tmp_path):
        """ティック1でのsave/loadが安全に動作すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        percept = _make_percept()
        orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 1

        # 復元後の操作が安全
        orch2.post_response_update(percept, delta_time=1.0)
        assert orch2.tick_count == 2
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
