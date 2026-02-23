"""
tests/test_phase_declaration.py - Phase宣言的定義の整合性検証テスト

宣言的定義内部の構造整合性を検証する。
既存orchestratorの手続き的コードは変更しない。
"""

from __future__ import annotations

import pytest

from psyche.phase_declaration import (
    ALL_BANDS,
    ALL_PHASES,
    PHASE_BY_ID,
    Band,
    BandDefinition,
    DataDependency,
    PhaseDefinition,
    compute_data_dependencies,
    get_all_enrichment_items,
    get_all_persisted_fields,
    get_cross_band_dependencies,
    get_intra_band_dependencies,
)


# ── Phase定義の基本整合性 ────────────────────────────────────


class TestPhaseDefinitionBasics:
    """Phase定義レコードの基本的な整合性を検証する。"""

    def test_all_phases_count(self):
        """設計書§3.3に記載された全Phaseが定義されていること。"""
        # EVERY_TICK: 16, EVERY_3_TICKS: 17, EVERY_5_TICKS: 32,
        # EVERY_10_TICKS: 3, CANDIDATE_GENERATION: 10, POST_SELECTION: 2
        assert len(ALL_PHASES) == 80

    def test_phase_ids_unique(self):
        """全Phase IDが一意であること。"""
        ids = [p.phase_id for p in ALL_PHASES]
        assert len(ids) == len(set(ids)), f"Duplicate phase IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_phase_by_id_complete(self):
        """PHASE_BY_ID索引が全Phaseを含むこと。"""
        assert len(PHASE_BY_ID) == len(ALL_PHASES)
        for p in ALL_PHASES:
            assert p.phase_id in PHASE_BY_ID
            assert PHASE_BY_ID[p.phase_id] is p

    def test_all_phases_have_display_name(self):
        """全Phaseに表示名が設定されていること。"""
        for p in ALL_PHASES:
            assert p.display_name, f"Phase {p.phase_id} has no display_name"

    def test_all_phases_have_valid_band(self):
        """全Phaseの帯域が有効な列挙型値であること。"""
        for p in ALL_PHASES:
            assert isinstance(p.band, Band), f"Phase {p.phase_id} has invalid band"

    def test_all_phases_have_method_name(self):
        """全Phaseにメソッド名が設定されていること。"""
        for p in ALL_PHASES:
            assert p.method_name, f"Phase {p.phase_id} has no method_name"

    def test_phase_definitions_are_frozen(self):
        """PhaseDefinitionがfrozenであること (不変性保証)。"""
        p = ALL_PHASES[0]
        with pytest.raises(AttributeError):
            p.phase_id = "modified"  # type: ignore[misc]

    def test_band_definitions_are_frozen(self):
        """BandDefinitionがfrozenであること。"""
        b = ALL_BANDS[0]
        with pytest.raises(AttributeError):
            b.band = Band.EVERY_3_TICKS  # type: ignore[misc]


# ── 帯域内順序の整合性 ──────────────────────────────────────


class TestBandOrderConsistency:
    """同一帯域内の実行順序が連続的かつ一意であることを検証する。"""

    @pytest.fixture
    def phases_by_band(self) -> dict[Band, list[PhaseDefinition]]:
        result: dict[Band, list[PhaseDefinition]] = {b: [] for b in Band}
        for p in ALL_PHASES:
            result[p.band].append(p)
        return result

    def test_band_orders_are_zero_based_consecutive(self, phases_by_band):
        """各帯域内でband_orderが0始まり連番であること。"""
        for band, phases in phases_by_band.items():
            orders = sorted(p.band_order for p in phases)
            expected = list(range(len(phases)))
            assert orders == expected, (
                f"Band {band.name}: expected orders {expected}, got {orders}"
            )

    def test_band_orders_unique_within_band(self, phases_by_band):
        """各帯域内でband_orderが一意であること。"""
        for band, phases in phases_by_band.items():
            orders = [p.band_order for p in phases]
            assert len(orders) == len(set(orders)), (
                f"Band {band.name}: duplicate band_orders"
            )

    def test_every_tick_phase_count(self, phases_by_band):
        """EVERY_TICK帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.EVERY_TICK]) == 16

    def test_every_3_ticks_phase_count(self, phases_by_band):
        """EVERY_3_TICKS帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.EVERY_3_TICKS]) == 17

    def test_every_5_ticks_phase_count(self, phases_by_band):
        """EVERY_5_TICKS帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.EVERY_5_TICKS]) == 32

    def test_every_10_ticks_phase_count(self, phases_by_band):
        """EVERY_10_TICKS帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.EVERY_10_TICKS]) == 3

    def test_candidate_generation_phase_count(self, phases_by_band):
        """CANDIDATE_GENERATION帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.CANDIDATE_GENERATION]) == 10

    def test_post_selection_phase_count(self, phases_by_band):
        """POST_SELECTION帯域のPhase数が設計書と一致すること。"""
        assert len(phases_by_band[Band.POST_SELECTION]) == 2


# ── 帯域定義の整合性 ────────────────────────────────────────


class TestBandDefinitionConsistency:
    """帯域定義レコードとPhase定義の整合性を検証する。"""

    def test_all_bands_count(self):
        """6帯域が定義されていること。"""
        assert len(ALL_BANDS) == 6

    def test_all_bands_cover_all_band_types(self):
        """全Band列挙型値がALL_BANDSに含まれること。"""
        defined_bands = {b.band for b in ALL_BANDS}
        for band in Band:
            assert band in defined_bands, f"Band {band.name} not in ALL_BANDS"

    def test_band_phase_ids_match_phase_definitions(self):
        """帯域定義のphase_idsが実際のPhase定義と一致すること。"""
        for bd in ALL_BANDS:
            expected_ids = tuple(
                p.phase_id
                for p in sorted(
                    (p for p in ALL_PHASES if p.band == bd.band),
                    key=lambda p: p.band_order,
                )
            )
            assert bd.phase_ids == expected_ids, (
                f"Band {bd.band.name}: phase_ids mismatch. "
                f"Expected {expected_ids}, got {bd.phase_ids}"
            )

    def test_band_execution_methods_consistent(self):
        """帯域定義のexecution_methodが所属Phase群のmethod_nameと一致すること。"""
        for bd in ALL_BANDS:
            for pid in bd.phase_ids:
                phase = PHASE_BY_ID[pid]
                assert phase.method_name == bd.execution_method, (
                    f"Phase {pid} method_name={phase.method_name} "
                    f"doesn't match band method={bd.execution_method}"
                )

    def test_all_phases_belong_to_exactly_one_band(self):
        """全Phaseがちょうど1つの帯域に所属すること。"""
        all_ids_from_bands: list[str] = []
        for bd in ALL_BANDS:
            all_ids_from_bands.extend(bd.phase_ids)
        # 重複なし
        assert len(all_ids_from_bands) == len(set(all_ids_from_bands))
        # 全Phaseカバー
        assert set(all_ids_from_bands) == {p.phase_id for p in ALL_PHASES}


# ── データ依存グラフの整合性 ──────────────────────────────────


class TestDataDependencyGraph:
    """データ依存グラフの導出結果を検証する。"""

    @pytest.fixture
    def dependencies(self) -> tuple[DataDependency, ...]:
        return compute_data_dependencies()

    def test_dependencies_are_non_empty(self, dependencies):
        """データ依存が存在すること。"""
        assert len(dependencies) > 0

    def test_all_dependency_phases_exist(self, dependencies):
        """依存関係に登場する全Phase IDが定義済みであること。"""
        for d in dependencies:
            assert d.consumer_phase_id in PHASE_BY_ID, (
                f"Consumer {d.consumer_phase_id} not in PHASE_BY_ID"
            )
            assert d.producer_phase_id in PHASE_BY_ID, (
                f"Producer {d.producer_phase_id} not in PHASE_BY_ID"
            )

    def test_no_self_dependencies(self, dependencies):
        """自己依存が存在しないこと。"""
        for d in dependencies:
            assert d.consumer_phase_id != d.producer_phase_id, (
                f"Self-dependency found: {d.consumer_phase_id}"
            )

    def test_same_band_flag_correct(self, dependencies):
        """same_bandフラグが実際の帯域関係と一致すること。"""
        for d in dependencies:
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            producer = PHASE_BY_ID[d.producer_phase_id]
            expected = (consumer.band == producer.band)
            assert d.same_band == expected, (
                f"Dependency {d.consumer_phase_id}->{d.producer_phase_id}: "
                f"same_band={d.same_band}, expected={expected}"
            )

    def test_intra_band_order_respected(self, dependencies):
        """同一帯域内の依存で、消費者が生産者より後に実行されること。"""
        for d in dependencies:
            if not d.same_band:
                continue
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            producer = PHASE_BY_ID[d.producer_phase_id]
            assert consumer.band_order > producer.band_order, (
                f"Intra-band dependency violation: "
                f"Phase {d.consumer_phase_id} (order={consumer.band_order}) "
                f"reads from Phase {d.producer_phase_id} (order={producer.band_order}) "
                f"in band {consumer.band.name}, "
                f"intermediate={d.intermediate_state}"
            )

    def test_intermediate_states_reference_valid_writes(self, dependencies):
        """依存関係の中間状態が生産者Phaseの書き込み対象に含まれること。"""
        for d in dependencies:
            producer = PHASE_BY_ID[d.producer_phase_id]
            assert d.intermediate_state in producer.writes, (
                f"Intermediate state '{d.intermediate_state}' not in writes "
                f"of producer Phase {d.producer_phase_id}: {producer.writes}"
            )

    def test_intermediate_states_reference_valid_reads(self, dependencies):
        """依存関係の中間状態が消費者Phaseの読み取り対象に含まれること。"""
        for d in dependencies:
            consumer = PHASE_BY_ID[d.consumer_phase_id]
            assert d.intermediate_state in consumer.reads, (
                f"Intermediate state '{d.intermediate_state}' not in reads "
                f"of consumer Phase {d.consumer_phase_id}: {consumer.reads}"
            )


# ── 帯域間依存の検証 ──────────────────────────────────────────


class TestCrossBandDependencies:
    """帯域間データ依存の検証。"""

    def test_cross_band_dependencies_exist(self):
        """帯域間依存が存在すること。"""
        cross = get_cross_band_dependencies()
        assert len(cross) > 0

    def test_intra_band_dependencies_exist(self):
        """帯域内依存が存在すること。"""
        intra = get_intra_band_dependencies()
        has_any = any(len(deps) > 0 for deps in intra.values())
        assert has_any

    def test_known_cross_band_dependency_tendency(self):
        """既知の帯域間依存: Phase 8(every_3) reads _tendency_sys from Phase 6(every_tick)。"""
        cross = get_cross_band_dependencies()
        found = any(
            d.consumer_phase_id == "8"
            and d.producer_phase_id == "6"
            and d.intermediate_state == "_tendency_sys"
            for d in cross
        )
        assert found, "Expected cross-band dependency: 8 -> 6 via _tendency_sys"

    def test_known_cross_band_dependency_self_view(self):
        """既知の帯域間依存: Phase 15(every_5) reads _last_self_view from Phase 9(every_3)。"""
        cross = get_cross_band_dependencies()
        found = any(
            d.consumer_phase_id == "15"
            and d.producer_phase_id == "9"
            and d.intermediate_state == "_last_self_view"
            for d in cross
        )
        assert found, "Expected cross-band dependency: 15 -> 9 via _last_self_view"


# ── 永続化フィールドの検証 ──────────────────────────────────


class TestPersistenceFields:
    """永続化対象フィールドの整合性を検証する。"""

    def test_persisted_fields_non_empty(self):
        """永続化対象フィールドが存在すること。"""
        fields = get_all_persisted_fields()
        assert len(fields) > 0

    def test_persisted_fields_count(self):
        """永続化対象フィールドの総数が設計書と整合すること。

        save/load v42で66フィールドだが、tick_count等の管理フィールドを除き、
        Phase定義からの導出では一部重複除去される。
        """
        fields = get_all_persisted_fields()
        # 設計書の66フィールドのうち、Phase定義で宣言されるもの
        # (tick_count, psyche, loop_state等の基本フィールドも含まれるが、
        #  重複した名前は1つにまとめられる)
        assert len(fields) >= 40, f"Too few persisted fields: {len(fields)}"

    def test_known_persisted_fields_present(self):
        """既知の永続化フィールドが含まれること。"""
        fields = get_all_persisted_fields()
        known = [
            "psyche", "loop_state", "dynamics", "amplitude",
            "value_orientation", "tendency_state", "stability_valve",
            "last_diff_summary", "last_episodes", "last_bindings",
            "forgetting_fixation_state", "action_result_state",
            "meta_emotion_state", "temporal_cognition_state",
        ]
        for f in known:
            assert f in fields, f"Expected persisted field '{f}' not found"

    def test_persisted_fields_are_strings(self):
        """全永続化フィールド名が文字列であること。"""
        for p in ALL_PHASES:
            for f in p.persisted_fields:
                assert isinstance(f, str), (
                    f"Phase {p.phase_id}: persisted field {f!r} is not str"
                )


# ── enrichment項目の検証 ───────────────────────────────────


class TestEnrichmentItems:
    """enrichment項目宣言の整合性を検証する。"""

    def test_enrichment_items_non_empty(self):
        """enrichment項目が存在すること。"""
        items = get_all_enrichment_items()
        assert len(items) > 0

    def test_known_enrichment_items_present(self):
        """既知のenrichment項目番号が含まれること。"""
        items = get_all_enrichment_items()
        known = ["25", "26", "27", "28", "29", "30", "31", "33",
                 "34", "35", "36", "38", "39", "40", "41", "42",
                 "43", "44", "45", "46", "48"]
        for i in known:
            assert i in items, f"Expected enrichment item #{i} not found"

    def test_enrichment_items_are_strings(self):
        """全enrichment項目番号が文字列であること。"""
        for p in ALL_PHASES:
            for e in p.enrichment_items:
                assert isinstance(e, str), (
                    f"Phase {p.phase_id}: enrichment item {e!r} is not str"
                )

    def test_no_duplicate_enrichment_across_phases(self):
        """同一enrichment項目が複数Phaseで重複宣言されていないこと。"""
        seen: dict[str, str] = {}
        for p in ALL_PHASES:
            for e in p.enrichment_items:
                if e in seen:
                    pytest.fail(
                        f"Enrichment item #{e} declared in both "
                        f"Phase {seen[e]} and Phase {p.phase_id}"
                    )
                seen[e] = p.phase_id


# ── 特定Phase定義の内容検証 ──────────────────────────────────


class TestSpecificPhaseContent:
    """設計書で明示された特定Phaseの属性値を検証する。"""

    def test_phase_1_attributes(self):
        """Phase 1: 感情更新+STM残留の属性が正しいこと。"""
        p = PHASE_BY_ID["1"]
        assert p.display_name == "感情更新+STM残留"
        assert p.band == Band.EVERY_TICK
        assert p.band_order == 0
        assert "react_with_stm" in p.modules
        assert not p.error_absorbed

    def test_phase_30_attributes(self):
        """Phase 30: 候補ポリシー生成の属性が正しいこと。"""
        p = PHASE_BY_ID["30"]
        assert p.display_name == "候補ポリシー生成"
        assert p.band == Band.CANDIDATE_GENERATION
        assert p.band_order == 1
        assert "thought" in p.modules
        assert not p.error_absorbed

    def test_phase_ps1_attributes(self):
        """Phase ps-1: 価値軸フィードバックの属性が正しいこと。"""
        p = PHASE_BY_ID["ps-1"]
        assert p.display_name == "価値軸フィードバック"
        assert p.band == Band.POST_SELECTION
        assert p.band_order == 0
        assert "value_orientation" in p.modules
        assert p.error_absorbed

    def test_phase_ps2_attributes(self):
        """Phase ps-2: 選択帰属記録の属性が正しいこと。"""
        p = PHASE_BY_ID["ps-2"]
        assert p.display_name == "選択帰属記録"
        assert p.band == Band.POST_SELECTION
        assert p.band_order == 1
        assert "selection_attribution" in p.modules
        assert "31" in p.enrichment_items

    def test_phase_26d_is_orchestrator_internal(self):
        """Phase 26d: orchestrator内部処理(モジュールなし)であること。"""
        p = PHASE_BY_ID["26d"]
        assert p.display_name == "予期差分照合"
        assert len(p.modules) == 0

    def test_phase_29_snapshot(self):
        """Phase 29: スナップショットが空のreads/writesであること。"""
        p = PHASE_BY_ID["29"]
        assert p.display_name == "スナップショット"
        assert len(p.reads) == 0
        assert len(p.writes) == 0
        assert not p.error_absorbed

    def test_phase_7f_reads_many_states(self):
        """Phase 7f: 注意配分が多数の中間状態を読み取ること。"""
        p = PHASE_BY_ID["7f"]
        assert len(p.reads) >= 10
        assert "_psyche" in p.reads
        assert "_last_bindings" in p.reads

    def test_phase_24b_reads_many_states(self):
        """Phase 24b: 参照頻度記述が多数の中間状態を読み取ること。"""
        p = PHASE_BY_ID["24b"]
        assert len(p.reads) >= 13
        assert "_last_episodes" in p.reads
        assert "_spontaneous_recall" in p.reads


# ── 循環依存の検出 ────────────────────────────────────────────


class TestNoCyclicDependencies:
    """同一帯域内で循環依存が存在しないことを検証する。"""

    def test_no_intra_band_cycles(self):
        """同一帯域内でデータ依存の循環が存在しないこと。

        同一帯域内では band_order で実行順序が保証されるため、
        消費者の band_order > 生産者の band_order が成立すれば
        循環は構造的に不可能。
        """
        intra = get_intra_band_dependencies()
        for band, deps in intra.items():
            for d in deps:
                consumer = PHASE_BY_ID[d.consumer_phase_id]
                producer = PHASE_BY_ID[d.producer_phase_id]
                assert consumer.band_order > producer.band_order, (
                    f"Potential cycle in band {band.name}: "
                    f"{d.consumer_phase_id}(order={consumer.band_order}) "
                    f"depends on {d.producer_phase_id}(order={producer.band_order})"
                )


# ── 実行非介入の構造検証 ──────────────────────────────────────


class TestExecutionNonIntervention:
    """宣言的定義が実行コードから構造的に分離されていることを検証する。"""

    def test_phase_declaration_not_imported_by_orchestrator(self):
        """orchestrator.pyがphase_declarationをインポートしていないこと。

        安全弁1: 実行非介入保証 — 宣言的定義は実行時に参照されない。
        """
        import importlib
        import inspect

        orchestrator = importlib.import_module("psyche.orchestrator")
        source = inspect.getsource(orchestrator)
        assert "phase_declaration" not in source, (
            "orchestrator.py must NOT import phase_declaration "
            "(execution non-intervention guarantee)"
        )

    def test_phase_definitions_are_static(self):
        """PhaseDefinitionがfrozenで実行時変更不可であること。"""
        for p in ALL_PHASES:
            with pytest.raises(AttributeError):
                p.band_order = 999  # type: ignore[misc]

    def test_band_definitions_are_static(self):
        """BandDefinitionがfrozenで実行時変更不可であること。"""
        for b in ALL_BANDS:
            with pytest.raises(AttributeError):
                b.execution_method = "modified"  # type: ignore[misc]


# ── ALL_PHASES順序の検証 ──────────────────────────────────────


class TestAllPhasesOrdering:
    """ALL_PHASESタプル内の順序が帯域順→帯域内順序順であることを検証する。"""

    def test_all_phases_ordered_by_band_then_order(self):
        """ALL_PHASES内のPhaseが帯域定義順→帯域内順序で並んでいること。"""
        band_order = [b.band for b in ALL_BANDS]
        prev_band_idx = -1
        prev_band_order = -1
        for p in ALL_PHASES:
            band_idx = band_order.index(p.band)
            if band_idx > prev_band_idx:
                # 新しい帯域に移行
                prev_band_idx = band_idx
                prev_band_order = p.band_order
            elif band_idx == prev_band_idx:
                # 同一帯域内で順序が昇順
                assert p.band_order > prev_band_order, (
                    f"Phase {p.phase_id}: band_order {p.band_order} "
                    f"not greater than previous {prev_band_order} "
                    f"in band {p.band.name}"
                )
                prev_band_order = p.band_order
            else:
                pytest.fail(
                    f"Phase {p.phase_id}: band {p.band.name} appears "
                    f"after a later band in ALL_PHASES"
                )
