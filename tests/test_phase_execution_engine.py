"""
tests/test_phase_execution_engine.py - Phase実行エンジンのテスト

テスト項目:
- 実行エンジンの初期化と登録
- 10ティック帯域限定の検証
- Phase実行順序の宣言的定義との一致
- エラー吸収境界の挙動
- 有効/無効フラグによる切替
- 等価性テスト（エンジン有効/無効で同一出力）
- フォールバック経路の検証
- 帯域限定保証
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from psyche.phase_execution_engine import (
    PhaseExecutionEngine,
    PhaseExecutionLog,
    PhaseStatus,
)
from psyche.phase_declaration import (
    Band,
    BAND_EVERY_10_TICKS,
    PHASE_BY_ID,
    PHASE_27,
    PHASE_28,
    PHASE_29,
)
from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept


# ── テストヘルパー ──────────────────────────────────────────────


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
) -> Percept:
    """テスト用Perceptを生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent="expression",
        emotion_valence=valence,
    )


# ── 実行エンジン初期化テスト ──────────────────────────────────


class TestPhaseExecutionEngineInit:
    """PhaseExecutionEngineの初期化テスト。"""

    def test_engine_created_enabled(self):
        """初期状態で有効であること。"""
        engine = PhaseExecutionEngine()
        assert engine.enabled is True

    def test_engine_band_phases(self):
        """10ティック帯域のPhase一覧が宣言的定義と一致すること。"""
        engine = PhaseExecutionEngine()
        assert engine._band_phase_ids == BAND_EVERY_10_TICKS.phase_ids
        assert engine._band_phase_ids == ("27", "28", "29")

    def test_engine_initially_not_fully_registered(self):
        """初期状態ではハンドラ未登録であること。"""
        engine = PhaseExecutionEngine()
        assert not engine.is_fully_registered()
        assert engine.get_registered_phase_ids() == ()

    def test_engine_initial_log_empty(self):
        """初期状態で実行ログが空であること。"""
        engine = PhaseExecutionEngine()
        assert engine.last_log.phase_results == {}


# ── ハンドラ登録テスト ──────────────────────────────────────────


class TestHandlerRegistration:
    """Phase処理関数の登録テスト。"""

    def test_register_valid_phase(self):
        """10ティック帯域のPhaseに処理関数を登録できること。"""
        engine = PhaseExecutionEngine()
        handler = MagicMock()
        engine.register_handler("27", handler)
        assert "27" in engine.get_registered_phase_ids()

    def test_register_all_phases(self):
        """全3 Phaseを登録するとfully_registeredになること。"""
        engine = PhaseExecutionEngine()
        for pid in ("27", "28", "29"):
            engine.register_handler(pid, MagicMock())
        assert engine.is_fully_registered()

    def test_register_invalid_phase_raises(self):
        """未サポート帯域のPhase登録がValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError, match="not in any supported band"):
            # Phase 15 はEVERY_5_TICKS帯域で未サポート
            engine.register_handler("15", MagicMock())

    def test_register_nonexistent_phase_raises(self):
        """存在しないPhase ID登録がValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError, match="not in any supported band"):
            engine.register_handler("999", MagicMock())


# ── 有効フラグテスト ──────────────────────────────────────────


class TestEnabledFlag:
    """実行エンジン有効/無効フラグのテスト。"""

    def test_disable_engine(self):
        """エンジンを無効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_enabled(False)
        assert engine.enabled is False

    def test_re_enable_engine(self):
        """エンジンを再有効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_enabled(False)
        engine.set_enabled(True)
        assert engine.enabled is True


# ── 帯域実行テスト ──────────────────────────────────────────────


class TestBandExecution:
    """帯域実行の基本テスト。"""

    def test_execute_calls_handlers_in_order(self):
        """ハンドラが宣言的定義の帯域内順序で呼ばれること。"""
        engine = PhaseExecutionEngine()
        call_order = []

        def handler27(orch, uid):
            call_order.append("27")

        def handler28(orch, uid):
            call_order.append("28")

        def handler29(orch, uid):
            call_order.append("29")

        engine.register_handler("27", handler27)
        engine.register_handler("28", handler28)
        engine.register_handler("29", handler29)

        mock_orch = MagicMock()
        log = engine.execute_band(mock_orch, "viewer")

        assert call_order == ["27", "28", "29"]
        assert log.phase_results["27"] == PhaseStatus.SUCCESS
        assert log.phase_results["28"] == PhaseStatus.SUCCESS
        assert log.phase_results["29"] == PhaseStatus.SUCCESS

    def test_execute_with_unregistered_handler(self):
        """未登録のPhaseがSKIPPEDとなること。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("27", MagicMock())
        # 28, 29 は未登録

        log = engine.execute_band(MagicMock(), "viewer")
        assert log.phase_results["27"] == PhaseStatus.SUCCESS
        assert log.phase_results["28"] == PhaseStatus.SKIPPED
        assert log.phase_results["29"] == PhaseStatus.SKIPPED

    def test_execute_passes_orchestrator_and_user_id(self):
        """ハンドラにorchestratorとuser_idが渡されること。"""
        engine = PhaseExecutionEngine()
        handler = MagicMock()
        engine.register_handler("27", handler)
        engine.register_handler("28", MagicMock())
        engine.register_handler("29", MagicMock())

        mock_orch = MagicMock()
        engine.execute_band(mock_orch, "test_user")

        handler.assert_called_once_with(mock_orch, "test_user")


# ── エラー吸収テスト ──────────────────────────────────────────


class TestErrorAbsorption:
    """エラー吸収境界の挙動テスト。"""

    def test_error_absorbed_phase_continues(self):
        """error_absorbed=TrueのPhaseでエラーが吸収されること。"""
        # Phase 27 (error_absorbed=True), Phase 28 (error_absorbed=True)
        assert PHASE_27.error_absorbed is True
        assert PHASE_28.error_absorbed is True

        engine = PhaseExecutionEngine()

        def failing_handler(orch, uid):
            raise RuntimeError("Test error")

        handler29 = MagicMock()
        engine.register_handler("27", failing_handler)
        engine.register_handler("28", MagicMock())
        engine.register_handler("29", handler29)

        log = engine.execute_band(MagicMock(), "viewer")

        # Phase 27 fails but is absorbed
        assert log.phase_results["27"] == PhaseStatus.FAILED
        # Phase 28 still executes
        assert log.phase_results["28"] == PhaseStatus.SUCCESS
        # Phase 29 still executes
        assert log.phase_results["29"] == PhaseStatus.SUCCESS

    def test_non_absorbed_phase_propagates_error(self):
        """error_absorbed=FalseのPhaseでエラーが伝播すること。"""
        # Phase 29 (error_absorbed=False)
        assert PHASE_29.error_absorbed is False

        engine = PhaseExecutionEngine()
        engine.register_handler("27", MagicMock())
        engine.register_handler("28", MagicMock())
        engine.register_handler("29", lambda o, u: (_ for _ in ()).throw(RuntimeError("Test")))

        with pytest.raises(RuntimeError, match="Test"):
            engine.execute_band(MagicMock(), "viewer")

    def test_error_absorption_matches_declaration(self):
        """エラー吸収挙動が宣言的定義と一致すること。"""
        for pid in ("27", "28", "29"):
            phase_def = PHASE_BY_ID[pid]
            if phase_def.error_absorbed:
                # error_absorbedなPhaseは例外が伝播しないことを確認
                engine = PhaseExecutionEngine()
                for p in ("27", "28", "29"):
                    if p == pid:
                        engine.register_handler(p, lambda o, u: (_ for _ in ()).throw(RuntimeError()))
                    else:
                        engine.register_handler(p, MagicMock())
                # エラーが吸収されるので例外は発生しない
                log = engine.execute_band(MagicMock(), "viewer")
                assert log.phase_results[pid] == PhaseStatus.FAILED


# ── 実行ログテスト ──────────────────────────────────────────────


class TestExecutionLog:
    """実行ログのテスト。"""

    def test_log_overwrites_previous(self):
        """実行ログが上書きされ蓄積されないこと。"""
        engine = PhaseExecutionEngine()
        for pid in ("27", "28", "29"):
            engine.register_handler(pid, MagicMock())

        # 1回目の実行
        log1 = engine.execute_band(MagicMock(), "viewer")
        assert all(v == PhaseStatus.SUCCESS for v in log1.phase_results.values())

        # 2回目の実行（Phase 27を失敗させる）
        engine.register_handler("27", lambda o, u: (_ for _ in ()).throw(RuntimeError()))
        log2 = engine.execute_band(MagicMock(), "viewer")

        # last_logは2回目の結果
        assert engine.last_log is log2
        assert log2.phase_results["27"] == PhaseStatus.FAILED

    def test_log_structure(self):
        """ログ構造が仕様通りであること。"""
        log = PhaseExecutionLog()
        assert isinstance(log.phase_results, dict)
        log.phase_results["27"] = PhaseStatus.SUCCESS
        assert log.phase_results["27"] == "success"


# ── Phase定義の検証テスト ──────────────────────────────────────


class TestPhaseDefinitions:
    """10ティック帯域のPhase定義の検証。"""

    def test_10tick_band_has_3_phases(self):
        """10ティック帯域が3 Phaseであること。"""
        assert len(BAND_EVERY_10_TICKS.phase_ids) == 3

    def test_10tick_band_phase_order(self):
        """10ティック帯域のPhase順序が27→28→29であること。"""
        assert BAND_EVERY_10_TICKS.phase_ids == ("27", "28", "29")

    def test_phase27_definition(self):
        """Phase 27の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["27"]
        assert p.band == Band.EVERY_10_TICKS
        assert p.band_order == 0
        assert p.error_absorbed is True
        assert "stability_valve" in p.modules

    def test_phase28_definition(self):
        """Phase 28の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["28"]
        assert p.band == Band.EVERY_10_TICKS
        assert p.band_order == 1
        assert p.error_absorbed is True
        assert "long_term_dynamics" in p.modules

    def test_phase29_definition(self):
        """Phase 29の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["29"]
        assert p.band == Band.EVERY_10_TICKS
        assert p.band_order == 2
        assert p.error_absorbed is False

    def test_get_band_phase_definitions(self):
        """get_band_phase_definitionsが正しいPhase定義を返すこと。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions()
        assert len(defs) == 3
        assert defs[0].phase_id == "27"
        assert defs[1].phase_id == "28"
        assert defs[2].phase_id == "29"


# ── 帯域限定保証テスト ──────────────────────────────────────────


class TestBandIsolation:
    """実行エンジンが10ティック帯域に限定されていることの検証。"""

    def test_can_register_every_tick_phase(self):
        """段階4: EVERY_TICK帯域のPhaseを登録できること。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("1", MagicMock())
        assert "1" in engine.get_band_registered_phase_ids(Band.EVERY_TICK)

    def test_can_register_3tick_phase(self):
        """段階3: EVERY_3_TICKS帯域のPhaseを登録できること。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        assert "8" in engine.get_band_registered_phase_ids(Band.EVERY_3_TICKS)

    def test_cannot_register_5tick_phase(self):
        """EVERY_5_TICKS帯域のPhaseを登録できないこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.register_handler("15", MagicMock())

    def test_cannot_register_candidate_gen_phase(self):
        """CANDIDATE_GENERATION帯域のPhaseを登録できないこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.register_handler("30", MagicMock())

    def test_cannot_register_post_selection_phase(self):
        """POST_SELECTION帯域のPhaseを登録できないこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.register_handler("ps-1", MagicMock())


# ── Orchestrator統合テスト ──────────────────────────────────────


class TestOrchestratorIntegration:
    """PsycheOrchestratorとの統合テスト。"""

    def test_orchestrator_has_engine(self):
        """Orchestratorが実行エンジンを保持していること。"""
        orch = PsycheOrchestrator()
        assert hasattr(orch, "_phase_engine")
        assert isinstance(orch._phase_engine, PhaseExecutionEngine)

    def test_orchestrator_engine_enabled_by_default(self):
        """Orchestratorの実行エンジンがデフォルト有効であること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.enabled is True

    def test_orchestrator_engine_fully_registered(self):
        """Orchestratorの全10ティックPhaseが登録されていること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_fully_registered()

    def test_orchestrator_engine_registered_phases(self):
        """登録されたPhase IDが27/28/29であること。"""
        orch = PsycheOrchestrator()
        registered = set(orch._phase_engine.get_registered_phase_ids())
        assert registered == {"27", "28", "29"}


# ── 等価性テスト ──────────────────────────────────────────────


class TestEquivalence:
    """実行エンジン有効/無効で同一の中間状態変化が生じることの検証。"""

    def _run_10_ticks(self, orch: PsycheOrchestrator) -> None:
        """10ティック目まで実行してPhase 27-29を発火させる。"""
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

    def _get_stability_state(self, orch: PsycheOrchestrator) -> dict:
        """stability_valveの状態を取得する。"""
        return orch._stability_valve.to_dict()

    def test_engine_enabled_produces_valid_output(self):
        """エンジン有効時に正常実行されること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.enabled is True

        self._run_10_ticks(orch)

        # 実行ログを確認
        log = orch._phase_engine.last_log
        # Phase 27, 28はerror_absorbed=Trueなので成功か失敗
        assert log.phase_results.get("27") in (PhaseStatus.SUCCESS, PhaseStatus.FAILED)
        assert log.phase_results.get("28") in (PhaseStatus.SUCCESS, PhaseStatus.FAILED)
        assert log.phase_results.get("29") in (PhaseStatus.SUCCESS, PhaseStatus.FAILED, PhaseStatus.SKIPPED)

    def test_engine_disabled_produces_valid_output(self):
        """エンジン無効時（フォールバック）に正常実行されること。"""
        orch = PsycheOrchestrator()
        orch._phase_engine.set_enabled(False)

        # フォールバックパスが実行されること
        self._run_10_ticks(orch)

        # フォールバック時はエンジンのログは更新されない
        log = orch._phase_engine.last_log
        assert log.phase_results == {}

    def test_equivalence_stability_valve(self):
        """エンジン有効/無効でstability_valveの状態が等価であること。"""
        # エンジン有効
        orch_enabled = PsycheOrchestrator()
        assert orch_enabled._phase_engine.enabled is True
        self._run_10_ticks(orch_enabled)
        state_enabled = self._get_stability_state(orch_enabled)

        # エンジン無効（フォールバック）
        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_enabled(False)
        self._run_10_ticks(orch_disabled)
        state_disabled = self._get_stability_state(orch_disabled)

        # stability_valveの観測結果が等価であること
        # timestamp は wall-clock time なので除外して比較
        def _strip_timestamp(d):
            d = dict(d)
            if "last_indicators" in d and d["last_indicators"] is not None:
                d["last_indicators"] = {
                    k: v for k, v in d["last_indicators"].items()
                    if k != "timestamp"
                }
            return d

        assert _strip_timestamp(state_enabled) == _strip_timestamp(state_disabled)

    def test_equivalence_dynamics_observer(self):
        """エンジン有効/無効でdynamics_observerの状態が等価であること。"""
        # エンジン有効
        orch_enabled = PsycheOrchestrator()
        assert orch_enabled._phase_engine.enabled is True
        self._run_10_ticks(orch_enabled)
        dynamics_enabled = orch_enabled._dynamics_observer

        # エンジン無効
        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_enabled(False)
        self._run_10_ticks(orch_disabled)
        dynamics_disabled = orch_disabled._dynamics_observer

        # 記録件数が等しいこと（get_total_turnsで直接比較）
        assert dynamics_enabled.get_total_turns() == dynamics_disabled.get_total_turns()
        assert dynamics_enabled.get_entry_count() == dynamics_disabled.get_entry_count()


# ── フォールバック切替テスト ──────────────────────────────────


class TestFallbackSwitching:
    """フォールバック経路の切替テスト。"""

    def test_switch_during_execution(self):
        """エンジンの有効/無効を切り替えても正常動作すること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()

        # 最初の10ティック: エンジン有効
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        assert orch._phase_engine.last_log.phase_results != {}

        # エンジン無効に切替
        orch._phase_engine.set_enabled(False)

        # 次の10ティック: フォールバック
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # 再有効化
        orch._phase_engine.set_enabled(True)

        # 次の10ティック: エンジン有効
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        assert orch._phase_engine.last_log.phase_results != {}


# ── Phase実行順序テスト ──────────────────────────────────────────


class TestExecutionOrder:
    """Phase実行順序が宣言的定義と一致することのテスト。"""

    def test_execution_order_matches_band_order(self):
        """実行順序が宣言的定義のband_orderと一致すること。"""
        engine = PhaseExecutionEngine()
        execution_order = []

        def make_handler(pid):
            def handler(orch, uid):
                execution_order.append(pid)
            return handler

        for pid in ("27", "28", "29"):
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer")

        # 実行順序がPhase定義のband_order順と一致
        expected_order = list(BAND_EVERY_10_TICKS.phase_ids)
        assert execution_order == expected_order

    def test_execution_order_not_reversed(self):
        """ハンドラ登録順序に関わらず宣言的定義順で実行されること。"""
        engine = PhaseExecutionEngine()
        execution_order = []

        def make_handler(pid):
            def handler(orch, uid):
                execution_order.append(pid)
            return handler

        # 逆順に登録
        for pid in ("29", "28", "27"):
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer")

        # 宣言的定義順（27→28→29）で実行される
        assert execution_order == ["27", "28", "29"]


# ── save/load非影響テスト ──────────────────────────────────────


class TestSaveLoadNonImpact:
    """実行エンジンがsave/loadに影響しないことの検証。"""

    def test_engine_state_not_in_save(self, tmp_path):
        """エンジン状態がsave()の出力に含まれないこと。"""
        import json

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        # 10ティック実行してPhase 27-29を発火
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        save_path = tmp_path / "psyche_snapshot.json"
        orch.save(save_path)

        data = json.loads(save_path.read_text(encoding="utf-8"))

        # エンジン関連のキーが存在しないこと
        assert "phase_engine" not in data
        assert "phase_execution_engine" not in data
        assert "_phase_engine" not in data
        assert "engine_enabled" not in data
        assert "engine_log" not in data

    def test_load_does_not_affect_engine(self, tmp_path):
        """load()でエンジン状態が変化しないこと。"""
        import json

        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(10):
            orch1.post_response_update(percept, delta_time=0.5, user_id="viewer")

        save_path = tmp_path / "psyche_snapshot.json"
        orch1.save(save_path)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load(save_path)

        # エンジンはデフォルト状態のまま
        assert orch2._phase_engine.enabled is True
        assert orch2._phase_engine.is_fully_registered()


# ── orchestrator.pyの構造検証 ──────────────────────────────────


class TestOrchestratorStructure:
    """orchestrator.pyの構造がphase_declarationを直接参照しないことの検証。"""

    def test_orchestrator_does_not_import_phase_declaration(self):
        """orchestrator.pyがphase_declarationを直接インポートしていないこと。

        orchestratorはphase_execution_engineのみを参照し、
        phase_declarationへの直接参照は持たない。
        """
        import importlib
        import inspect

        orchestrator = importlib.import_module("psyche.orchestrator")
        source = inspect.getsource(orchestrator)
        # phase_execution_engine は含まれるが phase_declaration は含まれない
        assert "phase_execution_engine" in source
        assert "phase_declaration" not in source


# ── enrichment非接続テスト ──────────────────────────────────────


class TestEnrichmentNonConnection:
    """実行エンジンの内部状態がenrichmentに接続されていないことの検証。"""

    def test_engine_log_not_in_enrichment(self):
        """エンジンログがenrichmentに含まれないこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        enrichment = orch.get_prompt_enrichment()
        enrichment_str = str(enrichment)

        # エンジン関連の用語が含まれないこと
        assert "phase_engine" not in enrichment_str.lower()
        assert "PhaseExecutionEngine" not in enrichment_str
        assert "PhaseExecutionLog" not in enrichment_str


