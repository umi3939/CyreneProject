"""
tests/test_phase_dependency_update.py - Cycle 9-10 Phase依存関係更新の検証テスト

Phase宣言定義(phase_declaration.py)と外部入力変数(phase_dependency_validator.py)が
Cycle 9-10の構造変更を正しく反映していることを検証する。

検証対象:
1. 新規追加Phase (21g, 21h, 26-exp) の宣言存在と属性正確性
2. PHASE_1のreads拡張（Cycle 9-10で追加された参照先）
3. EXTERNAL_INPUT_VARIABLESのCycle 9-10追加分
4. band_orderの連続性（5ティック帯域内で重複・欠番がないこと）
5. 検証ツール実行結果の整合性（unproduced_readsがCycle 9-10関連でゼロ）
6. 全Phase IDの一意性
7. 既存Phase定義の保存確認（Cycle 9-10以前のPhaseが消失していないこと）
"""

from __future__ import annotations

import pytest

from psyche.phase_declaration import (
    ALL_PHASES,
    ALL_BANDS,
    Band,
    PhaseDefinition,
    PHASE_BY_ID,
    PHASE_1,
    PHASE_21G,
    PHASE_21H,
    PHASE_26_EXP,
    PHASE_21F,
    PHASE_22,
    PHASE_26,
    PHASE_26B,
    compute_data_dependencies,
    get_intra_band_dependencies,
    get_cross_band_dependencies,
    get_all_persisted_fields,
    get_all_enrichment_items,
)

from tools.phase_dependency_validator import (
    EXTERNAL_INPUT_VARIABLES,
    validate_dependencies,
    report_to_dict,
)


# ── 新規Phase定義の存在と属性 ────────────────────────────────────


class TestPhase21GDeclaration:
    """Phase 21g (記憶想起→感情帰還) の宣言的定義の正確性を検証する。"""

    def test_phase_21g_exists_in_all_phases(self):
        """Phase 21gがALL_PHASESに含まれていること。"""
        phase_ids = [p.phase_id for p in ALL_PHASES]
        assert "21g" in phase_ids

    def test_phase_21g_exists_in_phase_by_id(self):
        """Phase 21gがPHASE_BY_IDから引けること。"""
        assert "21g" in PHASE_BY_ID

    def test_phase_21g_band(self):
        """Phase 21gが5ティック帯域に所属すること。"""
        assert PHASE_21G.band == Band.EVERY_5_TICKS

    def test_phase_21g_modules(self):
        """Phase 21gのモジュール名が正しいこと。"""
        assert "memory_emotion_return" in PHASE_21G.modules

    def test_phase_21g_reads_memory_emotion_return(self):
        """Phase 21gが_memory_emotion_returnを読み取ること。"""
        assert "_memory_emotion_return" in PHASE_21G.reads

    def test_phase_21g_reads_multi_path_recall(self):
        """Phase 21gが_multi_path_recallを読み取ること。"""
        assert "_multi_path_recall" in PHASE_21G.reads

    def test_phase_21g_reads_spontaneous_recall(self):
        """Phase 21gが_spontaneous_recallを読み取ること。"""
        assert "_spontaneous_recall" in PHASE_21G.reads

    def test_phase_21g_reads_last_bindings(self):
        """Phase 21gが_last_bindingsを読み取ること。"""
        assert "_last_bindings" in PHASE_21G.reads

    def test_phase_21g_reads_psyche(self):
        """Phase 21gが_psycheを読み取ること。"""
        assert "_psyche" in PHASE_21G.reads

    def test_phase_21g_writes_psyche(self):
        """Phase 21gが_psycheに書き込むこと。"""
        assert "_psyche" in PHASE_21G.writes

    def test_phase_21g_persisted_fields(self):
        """Phase 21gの永続化フィールドが正しいこと。"""
        assert "memory_emotion_return_state" in PHASE_21G.persisted_fields

    def test_phase_21g_error_absorbed(self):
        """Phase 21gがエラー吸収境界で囲まれていること。"""
        assert PHASE_21G.error_absorbed is True

    def test_phase_21g_after_21f(self):
        """Phase 21gが21fより後に実行されること。"""
        assert PHASE_21G.band_order > PHASE_21F.band_order

    def test_phase_21g_before_22(self):
        """Phase 21gが22より前に実行されること。"""
        assert PHASE_21G.band_order < PHASE_22.band_order


class TestPhase21HDeclaration:
    """Phase 21h (他者仮説→感情帰還) の宣言的定義の正確性を検証する。"""

    def test_phase_21h_exists_in_all_phases(self):
        """Phase 21hがALL_PHASESに含まれていること。"""
        phase_ids = [p.phase_id for p in ALL_PHASES]
        assert "21h" in phase_ids

    def test_phase_21h_exists_in_phase_by_id(self):
        """Phase 21hがPHASE_BY_IDから引けること。"""
        assert "21h" in PHASE_BY_ID

    def test_phase_21h_band(self):
        """Phase 21hが5ティック帯域に所属すること。"""
        assert PHASE_21H.band == Band.EVERY_5_TICKS

    def test_phase_21h_modules(self):
        """Phase 21hのモジュール名が正しいこと。"""
        assert "other_hypothesis_emotion_return" in PHASE_21H.modules

    def test_phase_21h_reads_other_hypothesis_emotion_return(self):
        """Phase 21hが_other_hypothesis_emotion_returnを読み取ること。"""
        assert "_other_hypothesis_emotion_return" in PHASE_21H.reads

    def test_phase_21h_reads_other_model_sys(self):
        """Phase 21hが_other_model_sysを読み取ること。"""
        assert "_other_model_sys" in PHASE_21H.reads

    def test_phase_21h_reads_psyche(self):
        """Phase 21hが_psycheを読み取ること。"""
        assert "_psyche" in PHASE_21H.reads

    def test_phase_21h_writes_psyche(self):
        """Phase 21hが_psycheに書き込むこと。"""
        assert "_psyche" in PHASE_21H.writes

    def test_phase_21h_persisted_fields(self):
        """Phase 21hの永続化フィールドが正しいこと。"""
        assert "other_hypothesis_emotion_return_state" in PHASE_21H.persisted_fields

    def test_phase_21h_error_absorbed(self):
        """Phase 21hがエラー吸収境界で囲まれていること。"""
        assert PHASE_21H.error_absorbed is True

    def test_phase_21h_after_21g(self):
        """Phase 21hが21gより後に実行されること。"""
        assert PHASE_21H.band_order > PHASE_21G.band_order

    def test_phase_21h_before_22(self):
        """Phase 21hが22より前に実行されること。"""
        assert PHASE_21H.band_order < PHASE_22.band_order


class TestPhase26EXPDeclaration:
    """Phase 26-EXP (経験強度帯域拡大) の宣言的定義の正確性を検証する。"""

    def test_phase_26_exp_exists_in_all_phases(self):
        """Phase 26-expがALL_PHASESに含まれていること。"""
        phase_ids = [p.phase_id for p in ALL_PHASES]
        assert "26-exp" in phase_ids

    def test_phase_26_exp_exists_in_phase_by_id(self):
        """Phase 26-expがPHASE_BY_IDから引けること。"""
        assert "26-exp" in PHASE_BY_ID

    def test_phase_26_exp_band(self):
        """Phase 26-expが5ティック帯域に所属すること。"""
        assert PHASE_26_EXP.band == Band.EVERY_5_TICKS

    def test_phase_26_exp_modules(self):
        """Phase 26-expのモジュール名が正しいこと。"""
        assert "value_orientation" in PHASE_26_EXP.modules

    def test_phase_26_exp_reads_psyche(self):
        """Phase 26-expが_psycheを読み取ること。"""
        assert "_psyche" in PHASE_26_EXP.reads

    def test_phase_26_exp_reads_value_orientation(self):
        """Phase 26-expが_value_orientationを読み取ること。"""
        assert "_value_orientation" in PHASE_26_EXP.reads

    def test_phase_26_exp_reads_last_episodes(self):
        """Phase 26-expが_last_episodesを読み取ること。"""
        assert "_last_episodes" in PHASE_26_EXP.reads

    def test_phase_26_exp_writes_value_orientation(self):
        """Phase 26-expが_value_orientationに書き込むこと。"""
        assert "_value_orientation" in PHASE_26_EXP.writes

    def test_phase_26_exp_no_persisted_fields(self):
        """Phase 26-expは永続化フィールドを持たないこと（非永続属性のみ使用）。"""
        assert len(PHASE_26_EXP.persisted_fields) == 0

    def test_phase_26_exp_error_absorbed(self):
        """Phase 26-expがエラー吸収境界で囲まれていること。"""
        assert PHASE_26_EXP.error_absorbed is True

    def test_phase_26_exp_after_26(self):
        """Phase 26-expが26より後に実行されること。"""
        assert PHASE_26_EXP.band_order > PHASE_26.band_order

    def test_phase_26_exp_before_26b(self):
        """Phase 26-expが26bより前に実行されること。"""
        assert PHASE_26_EXP.band_order < PHASE_26B.band_order


# ── PHASE_1のreads拡張 ─────────────────────────────────────────────


class TestPhase1ReadsExtension:
    """Cycle 9-10でPHASE_1に追加されたreadsの検証。"""

    def test_phase1_reads_behavioral_diversity_state(self):
        """PHASE_1が_behavioral_diversity_stateを読み取ること（build_drive_context経由）。"""
        assert "_behavioral_diversity_state" in PHASE_1.reads

    def test_phase1_reads_contradiction_processor(self):
        """PHASE_1が_contradiction_processorを読み取ること（build_drive_context経由）。"""
        assert "_contradiction_processor" in PHASE_1.reads

    def test_phase1_reads_memory_emotion_return(self):
        """PHASE_1が_memory_emotion_returnを読み取ること（build_mood_context経由）。"""
        assert "_memory_emotion_return" in PHASE_1.reads

    def test_phase1_reads_temporal_cognition(self):
        """PHASE_1が_temporal_cognitionを読み取ること（build_drive/mood_context経由）。"""
        assert "_temporal_cognition" in PHASE_1.reads

    def test_phase1_reads_transient_goal_mgr(self):
        """PHASE_1が_transient_goal_mgrを読み取ること（build_drive/mood_context経由）。"""
        assert "_transient_goal_mgr" in PHASE_1.reads

    def test_phase1_reads_persistent_commitment(self):
        """PHASE_1が_persistent_commitmentを読み取ること（build_drive/mood_context経由）。"""
        assert "_persistent_commitment" in PHASE_1.reads

    def test_phase1_reads_scoped_goal_sys(self):
        """PHASE_1が_scoped_goal_sysを読み取ること（build_drive/mood_context経由）。"""
        assert "_scoped_goal_sys" in PHASE_1.reads

    def test_phase1_still_reads_psyche(self):
        """PHASE_1が依然として_psycheを読み取ること。"""
        assert "_psyche" in PHASE_1.reads

    def test_phase1_still_reads_loop_state(self):
        """PHASE_1が依然として_loop_stateを読み取ること。"""
        assert "_loop_state" in PHASE_1.reads

    def test_phase1_still_writes_psyche(self):
        """PHASE_1が依然として_psycheに書き込むこと。"""
        assert "_psyche" in PHASE_1.writes

    def test_phase1_still_writes_loop_state(self):
        """PHASE_1が依然として_loop_stateに書き込むこと。"""
        assert "_loop_state" in PHASE_1.writes


# ── EXTERNAL_INPUT_VARIABLES拡張 ──────────────────────────────────


class TestExternalInputVariablesExtension:
    """Cycle 9-10で追加されたEXTERNAL_INPUT_VARIABLESの検証。"""

    def test_memory_emotion_return_in_external(self):
        """_memory_emotion_returnがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_memory_emotion_return" in EXTERNAL_INPUT_VARIABLES

    def test_other_hypothesis_emotion_return_in_external(self):
        """_other_hypothesis_emotion_returnがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_other_hypothesis_emotion_return" in EXTERNAL_INPUT_VARIABLES

    def test_other_model_sys_in_external(self):
        """_other_model_sysがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_other_model_sys" in EXTERNAL_INPUT_VARIABLES

    def test_behavioral_diversity_state_in_external(self):
        """_behavioral_diversity_stateがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_behavioral_diversity_state" in EXTERNAL_INPUT_VARIABLES

    def test_contradiction_processor_in_external(self):
        """_contradiction_processorがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_contradiction_processor" in EXTERNAL_INPUT_VARIABLES

    def test_return_pathway_monitor_in_external(self):
        """_return_pathway_monitorがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_return_pathway_monitor" in EXTERNAL_INPUT_VARIABLES

    def test_selection_attribution_recorder_in_external(self):
        """_selection_attribution_recorderがEXTERNAL_INPUT_VARIABLESに含まれること。"""
        assert "_selection_attribution_recorder" in EXTERNAL_INPUT_VARIABLES

    def test_preexisting_variables_preserved(self):
        """既存の外部入力変数が保存されていること。"""
        preexisting = [
            "_last_percept", "_last_activation_result", "_psyche", "_loop_state",
            "_text_dialogue_processor", "_spontaneous_processor",
            "_last_recalled_memories", "_responsibility_mgr", "_dispersion_state",
            "_action_result_observer", "_dialogue_learning_processor",
            "_forgetting_fixation_processor", "_multi_path_recall",
            "_spontaneous_recall", "_real_feed_processor", "_self_action_recorder",
            "_last_other_model", "_input_supply", "_meta_emotion_processor",
            "_introspection_cross_section", "_temporal_cognition",
            "_last_selected_policy_label", "_last_selected_policy_axis",
        ]
        for var in preexisting:
            assert var in EXTERNAL_INPUT_VARIABLES, f"{var} missing from EXTERNAL_INPUT_VARIABLES"


# ── band_orderの連続性 ─────────────────────────────────────────────


class TestBandOrderConsistency:
    """5ティック帯域内のband_orderの連続性と一意性を検証する。"""

    def test_5tick_band_orders_unique(self):
        """5ティック帯域内のband_orderに重複がないこと。"""
        five_tick_phases = [p for p in ALL_PHASES if p.band == Band.EVERY_5_TICKS]
        orders = [p.band_order for p in five_tick_phases]
        assert len(orders) == len(set(orders)), (
            f"Duplicate band_orders found: {sorted(orders)}"
        )

    def test_5tick_band_orders_contiguous(self):
        """5ティック帯域内のband_orderが0から連続していること。"""
        five_tick_phases = [p for p in ALL_PHASES if p.band == Band.EVERY_5_TICKS]
        orders = sorted(p.band_order for p in five_tick_phases)
        expected = list(range(len(orders)))
        assert orders == expected, (
            f"Non-contiguous band_orders: got {orders}, expected {expected}"
        )

    def test_every_tick_band_orders_unique(self):
        """毎ティック帯域内のband_orderに重複がないこと。"""
        tick_phases = [p for p in ALL_PHASES if p.band == Band.EVERY_TICK]
        orders = [p.band_order for p in tick_phases]
        assert len(orders) == len(set(orders))

    def test_every_3tick_band_orders_unique(self):
        """3ティック帯域内のband_orderに重複がないこと。"""
        tick3_phases = [p for p in ALL_PHASES if p.band == Band.EVERY_3_TICKS]
        orders = [p.band_order for p in tick3_phases]
        assert len(orders) == len(set(orders))

    def test_all_bands_orders_unique(self):
        """全帯域内でband_orderに重複がないこと。"""
        for band in Band:
            phases = [p for p in ALL_PHASES if p.band == band]
            orders = [p.band_order for p in phases]
            assert len(orders) == len(set(orders)), (
                f"Duplicate band_orders in {band.value}: {sorted(orders)}"
            )


# ── Phase IDの一意性 ──────────────────────────────────────────────


class TestPhaseIdUniqueness:
    """全Phase IDの一意性を検証する。"""

    def test_all_phase_ids_unique(self):
        """全PhaseのIDに重複がないこと。"""
        ids = [p.phase_id for p in ALL_PHASES]
        assert len(ids) == len(set(ids)), f"Duplicate phase IDs: {sorted(ids)}"

    def test_phase_by_id_count_matches(self):
        """PHASE_BY_IDのエントリ数がALL_PHASESと一致すること。"""
        assert len(PHASE_BY_ID) == len(ALL_PHASES)


# ── 既存Phase保存確認 ────────────────────────────────────────────


class TestPreexistingPhasesPreserved:
    """Cycle 9-10以前に存在したPhaseが消失していないことを検証する。"""

    @pytest.mark.parametrize("phase_id", [
        "1", "2", "2a", "2b", "2c", "3", "4", "5", "6", "7",
        "7a", "7b", "7c", "7d", "7e", "7f",
        "8", "9", "10", "11", "12", "12b", "13", "14", "14b",
        "14c", "14d", "14e", "14f", "14g", "14h", "14i", "14j",
        "15", "15b", "16", "17", "18", "19", "20",
        "21", "21b", "21c", "21d", "21e", "21f",
        "22", "23", "24", "24b",
        "25a", "25c", "25d", "25e", "25f", "25",
        "26", "26b", "26c", "26c2", "26d", "26e", "26f", "26g", "26h",
        "27", "28", "29",
        "31", "30", "30b", "32", "33", "34", "35", "35b", "35b2", "35c",
        "ps-1", "ps-2",
    ])
    def test_preexisting_phase_preserved(self, phase_id):
        """既存のPhase IDが引き続き存在すること。"""
        assert phase_id in PHASE_BY_ID, f"Phase {phase_id} missing from PHASE_BY_ID"


# ── 検証ツール結果の整合性 ───────────────────────────────────────


class TestValidatorResults:
    """検証ツールの実行結果がCycle 9-10関連で問題ないことを検証する。"""

    def test_no_unproduced_reads(self):
        """未生産読み取りがゼロであること。"""
        report = validate_dependencies()
        assert len(report.validation.unproduced_reads) == 0, (
            f"Unproduced reads found: {report.validation.unproduced_reads}"
        )

    def test_no_unexpected_order_violations_in_new_phases(self):
        """新規追加Phase (21g, 21h, 26-exp) に関する予期しない順序不整合がないこと。
        _psyche変数は複数Phaseで読み書きされるため、この変数に関する
        同一帯域内の順序報告は既知の正当なパターンとして許容する。"""
        report = validate_dependencies()
        new_phase_ids = {"21g", "21h", "26-exp"}
        # _psycheは多数のPhaseで読み書きされる共有変数。
        # 同一帯域内で後続Phaseが_psycheを書き込む場合、先行Phaseが
        # 前ティックの値を読むことは設計上正当な構造であり、不整合ではない。
        known_shared_variables = {"_psyche", "_value_orientation"}
        for v in report.validation.intra_band_order_violations:
            if v.reader_phase_id in new_phase_ids and v.writer_phase_id in new_phase_ids:
                assert v.variable in known_shared_variables, (
                    f"Unexpected order violation between new phases: "
                    f"reader={v.reader_phase_id}, writer={v.writer_phase_id}, var={v.variable}"
                )

    def test_total_phases_count(self):
        """総Phase数が83であること（既存80 + 新規3）。"""
        assert len(ALL_PHASES) == 83

    def test_report_to_dict_succeeds(self):
        """report_to_dictがエラーなく完了すること。"""
        report = validate_dependencies()
        d = report_to_dict(report)
        assert "validation" in d
        assert "summary" in d
        assert d["summary"]["statistics"]["total_phases"] == 83

    def test_behavioral_diversity_state_no_longer_unreferenced(self):
        """_behavioral_diversity_stateがunreferenced writesに含まれないこと。
        PHASE_1がこの変数を読み取るようになったため。"""
        report = validate_dependencies()
        unreferenced_vars = [uw.variable for uw in report.validation.unreferenced_writes]
        assert "_behavioral_diversity_state" not in unreferenced_vars


# ── 依存関係グラフの整合性 ──────────────────────────────────────


class TestDependencyGraph:
    """データ依存グラフの整合性を検証する。"""

    def test_compute_data_dependencies_includes_new_phases(self):
        """新規Phaseが依存グラフに参加していること。"""
        deps = compute_data_dependencies()
        participating_phases = set()
        for d in deps:
            participating_phases.add(d.consumer_phase_id)
            participating_phases.add(d.producer_phase_id)
        # Phase 21g reads _last_bindings (written by Phase 21)
        assert "21g" in participating_phases
        # Phase 26-exp reads _value_orientation (written by Phase 26)
        assert "26-exp" in participating_phases

    def test_21g_depends_on_21_for_bindings(self):
        """Phase 21gがPhase 21の_last_bindingsに依存すること。"""
        deps = compute_data_dependencies()
        found = False
        for d in deps:
            if (d.consumer_phase_id == "21g"
                    and d.producer_phase_id == "21"
                    and d.intermediate_state == "_last_bindings"):
                found = True
                break
        assert found, "Phase 21g -> Phase 21 via _last_bindings not found"

    def test_26_exp_depends_on_26_for_value_orientation(self):
        """Phase 26-expがPhase 26の_value_orientationに依存すること。"""
        deps = compute_data_dependencies()
        found = False
        for d in deps:
            if (d.consumer_phase_id == "26-exp"
                    and d.producer_phase_id == "26"
                    and d.intermediate_state == "_value_orientation"):
                found = True
                break
        assert found, "Phase 26-exp -> Phase 26 via _value_orientation not found"

    def test_intra_band_dependencies_dict_has_all_bands(self):
        """get_intra_band_dependenciesが全帯域を含むこと。"""
        result = get_intra_band_dependencies()
        for band in Band:
            assert band in result

    def test_cross_band_dependencies_not_empty(self):
        """帯域間依存が存在すること。"""
        result = get_cross_band_dependencies()
        assert len(result) > 0


# ── 新規Phase定義間の順序関係 ──────────────────────────────────


class TestNewPhaseOrdering:
    """新規Phaseの実行順序が設計通りであることを検証する。"""

    def test_21f_before_21g_before_21h_before_22(self):
        """Phase 21f → 21g → 21h → 22 の順序が保証されること。"""
        orders = {
            "21f": PHASE_BY_ID["21f"].band_order,
            "21g": PHASE_BY_ID["21g"].band_order,
            "21h": PHASE_BY_ID["21h"].band_order,
            "22": PHASE_BY_ID["22"].band_order,
        }
        assert orders["21f"] < orders["21g"] < orders["21h"] < orders["22"]

    def test_26_before_26exp_before_26b(self):
        """Phase 26 → 26-exp → 26b の順序が保証されること。"""
        orders = {
            "26": PHASE_BY_ID["26"].band_order,
            "26-exp": PHASE_BY_ID["26-exp"].band_order,
            "26b": PHASE_BY_ID["26b"].band_order,
        }
        assert orders["26"] < orders["26-exp"] < orders["26b"]

    def test_new_phases_same_band_as_neighbors(self):
        """新規Phaseが隣接Phaseと同一帯域であること。"""
        assert PHASE_21G.band == PHASE_21F.band
        assert PHASE_21H.band == PHASE_21G.band
        assert PHASE_26_EXP.band == PHASE_26.band
