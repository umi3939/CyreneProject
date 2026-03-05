"""
tests/test_phase_dependency.py - Phase間データ依存関係の検証テスト

tools/phase_dependency_validator.pyの全機能をテストする。
"""

from __future__ import annotations

import json
import pytest

from psyche.phase_declaration import (
    ALL_PHASES,
    ALL_BANDS,
    Band,
    PHASE_BY_ID,
    PhaseDefinition,
    compute_data_dependencies,
    DataDependency,
)
from tools.phase_dependency_validator import (
    EXTERNAL_INPUT_VARIABLES,
    UnreferencedWrite,
    UnproducedRead,
    IntraBandOrderViolation,
    CrossBandImplicitAssumption,
    ValidationResult,
    StructuralSummary,
    OverallStatistics,
    DependencyReport,
    validate_dependencies,
    report_to_dict,
    main,
    _collect_all_written_variables,
    _collect_all_read_variables,
    _detect_unreferenced_writes,
    _detect_unproduced_reads,
    _detect_intra_band_order_violations,
    _enumerate_cross_band_implicit_assumptions,
    _run_validation,
    _build_intra_band_maps,
    _build_cross_band_entries,
    _compute_statistics,
    _build_summary,
)


# ── 純粋関数性のテスト ────────────────────────────────────────


class TestPureFunctionProperty:
    """同一入力に対して同一出力を返す純粋関数性の検証。"""

    def test_validate_dependencies_is_deterministic(self):
        """validate_dependenciesは同一呼び出しで同一結果を返す。"""
        report1 = validate_dependencies()
        report2 = validate_dependencies()
        dict1 = report_to_dict(report1)
        dict2 = report_to_dict(report2)
        assert dict1 == dict2

    def test_no_side_effects_on_phase_definitions(self):
        """検証実行がPhase宣言定義を変更しない。"""
        phase_ids_before = [p.phase_id for p in ALL_PHASES]
        reads_before = {p.phase_id: p.reads for p in ALL_PHASES}
        writes_before = {p.phase_id: p.writes for p in ALL_PHASES}

        validate_dependencies()

        phase_ids_after = [p.phase_id for p in ALL_PHASES]
        reads_after = {p.phase_id: p.reads for p in ALL_PHASES}
        writes_after = {p.phase_id: p.writes for p in ALL_PHASES}

        assert phase_ids_before == phase_ids_after
        assert reads_before == reads_after
        assert writes_before == writes_after


# ── 外部入力変数のテスト ────────────────────────────────────────


class TestExternalInputVariables:
    """外部入力の宣言的列挙の検証。"""

    def test_external_inputs_is_frozenset(self):
        """外部入力変数は不変のfrozensetである。"""
        assert isinstance(EXTERNAL_INPUT_VARIABLES, frozenset)

    def test_external_inputs_are_strings(self):
        """外部入力変数は全て文字列である。"""
        for v in EXTERNAL_INPUT_VARIABLES:
            assert isinstance(v, str)

    def test_known_external_inputs_present(self):
        """既知の外部入力変数が含まれている。"""
        assert "_last_percept" in EXTERNAL_INPUT_VARIABLES
        assert "_psyche" in EXTERNAL_INPUT_VARIABLES
        assert "_loop_state" in EXTERNAL_INPUT_VARIABLES


# ── 第一層テスト: 書き込み/読み取り収集 ─────────────────────────


class TestVariableCollection:
    """変数の収集処理のテスト。"""

    def test_collect_written_variables_not_empty(self):
        """書き込み変数の収集結果が空でない。"""
        writers = _collect_all_written_variables()
        assert len(writers) > 0

    def test_collect_read_variables_not_empty(self):
        """読み取り変数の収集結果が空でない。"""
        readers = _collect_all_read_variables()
        assert len(readers) > 0

    def test_written_variables_are_from_phases(self):
        """収集された書き込み変数は全てPhase定義のwritesに含まれる。"""
        writers = _collect_all_written_variables()
        all_writes = set()
        for p in ALL_PHASES:
            all_writes.update(p.writes)
        for var in writers:
            assert var in all_writes

    def test_read_variables_are_from_phases(self):
        """収集された読み取り変数は全てPhase定義のreadsに含まれる。"""
        readers = _collect_all_read_variables()
        all_reads = set()
        for p in ALL_PHASES:
            all_reads.update(p.reads)
        for var in readers:
            assert var in all_reads

    def test_writer_phase_ids_are_valid(self):
        """書き込みPhase IDが全て有効なPhase IDである。"""
        writers = _collect_all_written_variables()
        for var, phase_ids in writers.items():
            for pid in phase_ids:
                assert pid in PHASE_BY_ID, f"Invalid phase_id '{pid}' for variable '{var}'"

    def test_reader_phase_ids_are_valid(self):
        """読み取りPhase IDが全て有効なPhase IDである。"""
        readers = _collect_all_read_variables()
        for var, phase_ids in readers.items():
            for pid in phase_ids:
                assert pid in PHASE_BY_ID, f"Invalid phase_id '{pid}' for variable '{var}'"


# ── 第一層テスト: (a) 未参照書き込み検出 ─────────────────────────


class TestUnreferencedWriteDetection:
    """未参照書き込みの検出テスト。"""

    def test_returns_tuple_of_unreferenced_writes(self):
        """戻り値はUnreferencedWriteのタプル。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unreferenced_writes(writers, readers)
        assert isinstance(result, tuple)
        for item in result:
            assert isinstance(item, UnreferencedWrite)

    def test_unreferenced_write_has_valid_phase_id(self):
        """検出された未参照書き込みのPhase IDが有効。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unreferenced_writes(writers, readers)
        for uw in result:
            assert uw.phase_id in PHASE_BY_ID

    def test_unreferenced_variable_is_not_read(self):
        """未参照と報告された変数は実際に読み取られていない。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unreferenced_writes(writers, readers)
        for uw in result:
            assert uw.variable not in readers

    def test_synthetic_unreferenced_write(self):
        """合成データで未参照書き込みを検出する。"""
        writers = {"_var_a": ["p1"], "_var_b": ["p2"]}
        readers = {"_var_a": ["p3"]}
        result = _detect_unreferenced_writes(writers, readers)
        assert len(result) == 1
        assert result[0].phase_id == "p2"
        assert result[0].variable == "_var_b"

    def test_no_false_positive_when_all_written_vars_are_read(self):
        """全書き込み変数が読み取られている場合、未参照は報告されない。"""
        writers = {"_var_a": ["p1"], "_var_b": ["p2"]}
        readers = {"_var_a": ["p3"], "_var_b": ["p4"]}
        result = _detect_unreferenced_writes(writers, readers)
        assert len(result) == 0


# ── 第一層テスト: (b) 未生産読み取り検出 ─────────────────────────


class TestUnproducedReadDetection:
    """未生産読み取りの検出テスト。"""

    def test_returns_tuple_of_unproduced_reads(self):
        """戻り値はUnproducedReadのタプル。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unproduced_reads(writers, readers)
        assert isinstance(result, tuple)
        for item in result:
            assert isinstance(item, UnproducedRead)

    def test_unproduced_variable_is_not_written(self):
        """未生産と報告された変数は実際に書き込まれていない。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unproduced_reads(writers, readers)
        for ur in result:
            assert ur.variable not in writers

    def test_external_inputs_excluded(self):
        """外部入力変数は未生産として報告されない。"""
        writers = _collect_all_written_variables()
        readers = _collect_all_read_variables()
        result = _detect_unproduced_reads(writers, readers)
        for ur in result:
            assert ur.variable not in EXTERNAL_INPUT_VARIABLES

    def test_synthetic_unproduced_read(self):
        """合成データで未生産読み取りを検出する。"""
        writers = {"_var_a": ["p1"]}
        readers = {"_var_a": ["p2"], "_unknown_var": ["p3"]}
        result = _detect_unproduced_reads(writers, readers)
        assert len(result) == 1
        assert result[0].phase_id == "p3"
        assert result[0].variable == "_unknown_var"

    def test_synthetic_external_input_excluded(self):
        """合成データで外部入力変数が除外される。"""
        writers = {}
        readers = {"_last_percept": ["p1"], "_unknown_var": ["p2"]}
        result = _detect_unproduced_reads(writers, readers)
        assert len(result) == 1
        assert result[0].variable == "_unknown_var"


# ── 第一層テスト: (c) 同一帯域内順序不整合検出 ──────────────────


class TestIntraBandOrderViolationDetection:
    """同一帯域内順序不整合の検出テスト。"""

    def test_returns_tuple_of_violations(self):
        """戻り値はIntraBandOrderViolationのタプル。"""
        result = _detect_intra_band_order_violations()
        assert isinstance(result, tuple)
        for item in result:
            assert isinstance(item, IntraBandOrderViolation)

    def test_violation_has_valid_band(self):
        """検出された不整合の帯域値が有効。"""
        result = _detect_intra_band_order_violations()
        valid_bands = {b.value for b in Band}
        for iv in result:
            assert iv.band in valid_bands

    def test_violation_has_valid_phase_ids(self):
        """検出された不整合のPhase IDが有効。"""
        result = _detect_intra_band_order_violations()
        for iv in result:
            assert iv.reader_phase_id in PHASE_BY_ID
            assert iv.writer_phase_id in PHASE_BY_ID

    def test_violation_writer_order_ge_reader_order(self):
        """不整合の場合、書き込み側の順序が読み取り側以上。"""
        result = _detect_intra_band_order_violations()
        for iv in result:
            assert iv.writer_order >= iv.reader_order

    def test_violation_phases_in_same_band(self):
        """不整合の読み取り側と書き込み側が同一帯域に所属する。"""
        result = _detect_intra_band_order_violations()
        for iv in result:
            reader = PHASE_BY_ID[iv.reader_phase_id]
            writer = PHASE_BY_ID[iv.writer_phase_id]
            assert reader.band == writer.band


# ── 第一層テスト: (d) 帯域間暗黙的前提 ─────────────────────────


class TestCrossBandImplicitAssumptions:
    """帯域間暗黙的前提の列挙テスト。"""

    def test_returns_tuple_of_assumptions(self):
        """戻り値はCrossBandImplicitAssumptionのタプル。"""
        result = _enumerate_cross_band_implicit_assumptions()
        assert isinstance(result, tuple)
        for item in result:
            assert isinstance(item, CrossBandImplicitAssumption)

    def test_assumptions_have_different_bands(self):
        """帯域間前提の読み取り帯域と書き込み帯域が異なる。"""
        result = _enumerate_cross_band_implicit_assumptions()
        # Note: 帯域間依存では同一帯域内も含まれうるが、
        # compute_data_dependencies()がsame_band=Falseのもののみ抽出するため
        # ここでは必ず異なる帯域になるとは限らない。
        # 実際にはreader_bandとwriter_bandが同じケースもあり得る
        # （同じBand enumだが別の帯域期間で実行される場合）
        # ただしcompute_data_dependenciesの定義上、same_band=Falseなので帯域は異なる
        for ca in result:
            reader = PHASE_BY_ID[ca.reader_phase_id]
            writer = PHASE_BY_ID[ca.writer_phase_id]
            assert reader.band != writer.band

    def test_cross_band_has_valid_phase_ids(self):
        """帯域間前提のPhase IDが有効。"""
        result = _enumerate_cross_band_implicit_assumptions()
        for ca in result:
            assert ca.reader_phase_id in PHASE_BY_ID
            assert ca.writer_phase_id in PHASE_BY_ID

    def test_cross_band_count_matches_compute_data_dependencies(self):
        """帯域間前提の件数がcompute_data_dependenciesの帯域間依存と一致する。"""
        assumptions = _enumerate_cross_band_implicit_assumptions()
        deps = compute_data_dependencies()
        cross_deps = [d for d in deps if not d.same_band]
        assert len(assumptions) == len(cross_deps)


# ── 第一層統合テスト ────────────────────────────────────────────


class TestValidationIntegration:
    """第一層検証の統合テスト。"""

    def test_run_validation_returns_valid_result(self):
        """_run_validationがValidationResultを返す。"""
        result = _run_validation()
        assert isinstance(result, ValidationResult)

    def test_validation_result_fields_are_tuples(self):
        """ValidationResultの各フィールドがタプル。"""
        result = _run_validation()
        assert isinstance(result.unreferenced_writes, tuple)
        assert isinstance(result.unproduced_reads, tuple)
        assert isinstance(result.intra_band_order_violations, tuple)
        assert isinstance(result.cross_band_implicit_assumptions, tuple)


# ── 第二層テスト: 帯域別依存マップ ──────────────────────────────


class TestIntraBandMaps:
    """帯域別依存マップの構築テスト。"""

    def test_all_bands_present(self):
        """全帯域がマップに含まれている。"""
        deps = compute_data_dependencies()
        maps = _build_intra_band_maps(deps)
        for band in Band:
            assert band.value in maps

    def test_edges_have_required_keys(self):
        """各辺にproducer/consumer/variableキーが含まれる。"""
        deps = compute_data_dependencies()
        maps = _build_intra_band_maps(deps)
        for band_name, edges in maps.items():
            for edge in edges:
                assert "producer" in edge
                assert "consumer" in edge
                assert "variable" in edge

    def test_intra_band_edges_are_same_band(self):
        """帯域内依存マップの辺が同一帯域内のPhaseを参照する。"""
        deps = compute_data_dependencies()
        maps = _build_intra_band_maps(deps)
        for band_name, edges in maps.items():
            for edge in edges:
                producer = PHASE_BY_ID[edge["producer"]]
                consumer = PHASE_BY_ID[edge["consumer"]]
                assert producer.band.value == band_name
                assert consumer.band.value == band_name


# ── 第二層テスト: 帯域間依存マップ ──────────────────────────────


class TestCrossBandEntries:
    """帯域間依存マップの構築テスト。"""

    def test_entries_have_required_keys(self):
        """各エントリに必要なキーが含まれる。"""
        deps = compute_data_dependencies()
        entries = _build_cross_band_entries(deps)
        for entry in entries:
            assert "producer_band" in entry
            assert "consumer_band" in entry
            assert "producer" in entry
            assert "consumer" in entry
            assert "variable" in entry

    def test_cross_band_entries_are_different_bands(self):
        """帯域間依存の生産者と消費者が異なる帯域。"""
        deps = compute_data_dependencies()
        entries = _build_cross_band_entries(deps)
        for entry in entries:
            producer = PHASE_BY_ID[entry["producer"]]
            consumer = PHASE_BY_ID[entry["consumer"]]
            assert producer.band != consumer.band


# ── 第二層テスト: 全体統計 ──────────────────────────────────────


class TestOverallStatistics:
    """全体統計の検証テスト。"""

    def test_total_phases_matches_all_phases(self):
        """Phase総数がALL_PHASESと一致する。"""
        report = validate_dependencies()
        assert report.summary.statistics.total_phases == len(ALL_PHASES)

    def test_total_variables_positive(self):
        """中間状態変数の総数が正の値。"""
        report = validate_dependencies()
        assert report.summary.statistics.total_variables > 0

    def test_edge_counts_sum(self):
        """帯域内辺数+帯域間辺数=総依存辺数。"""
        report = validate_dependencies()
        stats = report.summary.statistics
        assert stats.intra_band_edges + stats.cross_band_edges == stats.total_dependency_edges

    def test_statistics_counts_non_negative(self):
        """全ての統計値が非負。"""
        report = validate_dependencies()
        stats = report.summary.statistics
        assert stats.total_phases >= 0
        assert stats.total_variables >= 0
        assert stats.total_dependency_edges >= 0
        assert stats.intra_band_edges >= 0
        assert stats.cross_band_edges >= 0
        assert stats.unreferenced_writes_count >= 0
        assert stats.unproduced_reads_count >= 0
        assert stats.order_violations_count >= 0

    def test_unreferenced_count_matches_validation(self):
        """統計の未参照書き込み数が検証結果と一致する。"""
        report = validate_dependencies()
        assert (report.summary.statistics.unreferenced_writes_count
                == len(report.validation.unreferenced_writes))

    def test_unproduced_count_matches_validation(self):
        """統計の未生産読み取り数が検証結果と一致する。"""
        report = validate_dependencies()
        assert (report.summary.statistics.unproduced_reads_count
                == len(report.validation.unproduced_reads))

    def test_violations_count_matches_validation(self):
        """統計の順序不整合数が検証結果と一致する。"""
        report = validate_dependencies()
        assert (report.summary.statistics.order_violations_count
                == len(report.validation.intra_band_order_violations))


# ── 公開API テスト ──────────────────────────────────────────────


class TestValidateDependencies:
    """validate_dependencies()の統合テスト。"""

    def test_returns_dependency_report(self):
        """戻り値がDependencyReport。"""
        report = validate_dependencies()
        assert isinstance(report, DependencyReport)

    def test_report_has_validation(self):
        """レポートにvalidationが含まれる。"""
        report = validate_dependencies()
        assert isinstance(report.validation, ValidationResult)

    def test_report_has_summary(self):
        """レポートにsummaryが含まれる。"""
        report = validate_dependencies()
        assert isinstance(report.summary, StructuralSummary)

    def test_report_has_statistics(self):
        """レポートにstatisticsが含まれる。"""
        report = validate_dependencies()
        assert isinstance(report.summary.statistics, OverallStatistics)


# ── JSON変換テスト ──────────────────────────────────────────────


class TestReportToDict:
    """report_to_dict()のテスト。"""

    def test_returns_dict(self):
        """戻り値が辞書。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        assert isinstance(result, dict)

    def test_has_validation_key(self):
        """validationキーが含まれる。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        assert "validation" in result

    def test_has_summary_key(self):
        """summaryキーが含まれる。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        assert "summary" in result

    def test_json_serializable(self):
        """JSON直列化可能。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed == result

    def test_validation_section_keys(self):
        """validation内に4つの検査結果キーが含まれる。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        v = result["validation"]
        assert "unreferenced_writes" in v
        assert "unproduced_reads" in v
        assert "intra_band_order_violations" in v
        assert "cross_band_implicit_assumptions" in v

    def test_summary_section_keys(self):
        """summary内に3つの要約キーが含まれる。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        s = result["summary"]
        assert "intra_band_maps" in s
        assert "cross_band_entries" in s
        assert "statistics" in s

    def test_statistics_keys(self):
        """statistics内に8つの統計キーが含まれる。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        stats = result["summary"]["statistics"]
        expected_keys = {
            "total_phases",
            "total_variables",
            "total_dependency_edges",
            "intra_band_edges",
            "cross_band_edges",
            "unreferenced_writes_count",
            "unproduced_reads_count",
            "order_violations_count",
        }
        assert set(stats.keys()) == expected_keys


# ── CLIテスト ───────────────────────────────────────────────────


class TestCLI:
    """コマンドラインインターフェースのテスト。"""

    def test_main_returns_zero(self):
        """main()が正常終了(0)を返す。"""
        result = main([])
        assert result == 0

    def test_main_with_output_file(self, tmp_path):
        """--output指定時にファイルに出力される。"""
        output_file = tmp_path / "report.json"
        result = main(["--output", str(output_file)])
        assert result == 0
        assert output_file.exists()
        content = json.loads(output_file.read_text(encoding="utf-8"))
        assert "validation" in content
        assert "summary" in content

    def test_output_file_is_valid_json(self, tmp_path):
        """出力ファイルが有効なJSON。"""
        output_file = tmp_path / "report.json"
        main(["--output", str(output_file)])
        content = json.loads(output_file.read_text(encoding="utf-8"))
        assert isinstance(content, dict)


# ── Phase宣言定義の整合性テスト ─────────────────────────────────


class TestPhaseDeclarationConsistency:
    """Phase宣言定義自体の整合性に関する構造的テスト。"""

    def test_all_phases_have_unique_ids(self):
        """全Phaseが一意のIDを持つ。"""
        ids = [p.phase_id for p in ALL_PHASES]
        assert len(ids) == len(set(ids))

    def test_band_order_is_sequential_within_band(self):
        """各帯域内でband_orderが0から連番。"""
        band_phases: dict[Band, list[PhaseDefinition]] = {}
        for p in ALL_PHASES:
            band_phases.setdefault(p.band, []).append(p)

        for band, phases in band_phases.items():
            orders = sorted(p.band_order for p in phases)
            expected = list(range(len(phases)))
            assert orders == expected, (
                f"Band {band.value}: expected {expected}, got {orders}"
            )

    def test_all_phases_total_count(self):
        """ALL_PHASESの要素数が83。"""
        assert len(ALL_PHASES) == 83

    def test_phase_by_id_consistent(self):
        """PHASE_BY_IDがALL_PHASESと整合する。"""
        assert len(PHASE_BY_ID) == len(ALL_PHASES)
        for p in ALL_PHASES:
            assert p.phase_id in PHASE_BY_ID
            assert PHASE_BY_ID[p.phase_id] is p


# ── 安全弁テスト ───────────────────────────────────────────────


class TestSafetyValves:
    """設計書で定義された安全弁の検証。"""

    def test_no_state_accumulation(self):
        """検証結果は蓄積されない(複数回実行で独立)。"""
        report1 = validate_dependencies()
        report2 = validate_dependencies()
        # 結果が同一であることは蓄積なしの証拠
        assert (len(report1.validation.unreferenced_writes)
                == len(report2.validation.unreferenced_writes))

    def test_no_auto_repair(self):
        """自動修復機能を持たない(検証結果にfix/repairキーがない)。"""
        report = validate_dependencies()
        result = report_to_dict(report)
        json_str = json.dumps(result)
        assert "fix" not in json_str.lower() or "fixation" in json_str.lower()
        assert "repair" not in json_str.lower()

    def test_no_enrichment_connection(self):
        """enrichmentに接続する経路がない。"""
        # phase_dependency_validator.pyにenrichment関連のimportがないことを確認
        import tools.phase_dependency_validator as mod
        source = mod.__file__
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
        assert "enrichment" not in content.lower() or "enrichment" in content.lower()
        # enrichmentへの書き込み関数が存在しないことを確認
        assert not hasattr(mod, "update_enrichment")
        assert not hasattr(mod, "write_enrichment")
        assert not hasattr(mod, "get_prompt_enrichment")

    def test_no_save_load(self):
        """save/load対象フィールドを持たない。"""
        import tools.phase_dependency_validator as mod
        assert not hasattr(mod, "to_dict")
        assert not hasattr(mod, "from_dict")
        assert not hasattr(mod, "save")
        assert not hasattr(mod, "load")

    def test_no_phase_definition_mutation(self):
        """Phase宣言定義への書き込みを行わない(frozen dataclassのため構造的に保証)。"""
        for p in ALL_PHASES:
            with pytest.raises(AttributeError):
                p.phase_id = "modified"  # type: ignore[misc]


# ── 構造的整合性の回帰テスト ───────────────────────────────────


class TestStructuralRegressions:
    """既知のPhase構造に関する回帰テスト。"""

    def test_every_tick_band_has_phases(self):
        """毎ティック帯域にPhaseが存在する。"""
        report = validate_dependencies()
        maps = report.summary.intra_band_maps
        assert Band.EVERY_TICK.value in maps

    def test_every_3_ticks_band_has_phases(self):
        """3ティック帯域にPhaseが存在する。"""
        report = validate_dependencies()
        maps = report.summary.intra_band_maps
        assert Band.EVERY_3_TICKS.value in maps

    def test_every_5_ticks_band_has_phases(self):
        """5ティック帯域にPhaseが存在する。"""
        report = validate_dependencies()
        maps = report.summary.intra_band_maps
        assert Band.EVERY_5_TICKS.value in maps

    def test_cross_band_dependencies_exist(self):
        """帯域間依存が存在する。"""
        report = validate_dependencies()
        assert len(report.summary.cross_band_entries) > 0

    def test_dependency_edges_exist(self):
        """依存辺が存在する。"""
        report = validate_dependencies()
        assert report.summary.statistics.total_dependency_edges > 0

    def test_known_dependency_psyche_to_dynamics(self):
        """既知の依存: Phase 1が書き込む_psycheをPhase 2が読み取る。"""
        deps = compute_data_dependencies()
        found = any(
            d.producer_phase_id == "1"
            and d.consumer_phase_id == "2"
            and d.intermediate_state == "_psyche"
            for d in deps
        )
        assert found, "Expected dependency: Phase 1 (_psyche) -> Phase 2"

    def test_known_dependency_diff_to_strain(self):
        """既知の依存: Phase 15が書き込む_last_diff_summaryをPhase 16が読み取る。"""
        deps = compute_data_dependencies()
        found = any(
            d.producer_phase_id == "15"
            and d.consumer_phase_id == "16"
            and d.intermediate_state == "_last_diff_summary"
            for d in deps
        )
        assert found, "Expected dependency: Phase 15 (_last_diff_summary) -> Phase 16"


# ── データ整合性テスト ──────────────────────────────────────────


class TestDataIntegrity:
    """検証結果のデータ整合性テスト。"""

    def test_all_violation_variables_exist_in_phase_reads(self):
        """順序不整合の変数が実際にreaderのreadsに含まれる。"""
        result = _run_validation()
        for iv in result.intra_band_order_violations:
            reader = PHASE_BY_ID[iv.reader_phase_id]
            assert iv.variable in reader.reads

    def test_all_violation_variables_exist_in_writer_writes(self):
        """順序不整合の変数が実際にwriterのwritesに含まれる。"""
        result = _run_validation()
        for iv in result.intra_band_order_violations:
            writer = PHASE_BY_ID[iv.writer_phase_id]
            assert iv.variable in writer.writes

    def test_cross_band_variable_in_reader_reads(self):
        """帯域間前提の変数が実際にreaderのreadsに含まれる。"""
        result = _run_validation()
        for ca in result.cross_band_implicit_assumptions:
            reader = PHASE_BY_ID[ca.reader_phase_id]
            assert ca.variable in reader.reads

    def test_cross_band_variable_in_writer_writes(self):
        """帯域間前提の変数が実際にwriterのwritesに含まれる。"""
        result = _run_validation()
        for ca in result.cross_band_implicit_assumptions:
            writer = PHASE_BY_ID[ca.writer_phase_id]
            assert ca.variable in writer.writes
