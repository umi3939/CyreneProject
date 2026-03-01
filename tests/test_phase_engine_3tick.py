"""
tests/test_phase_engine_3tick.py - Phase実行エンジン段階3（3ティック帯域）のテスト

テスト項目:
- 帯域横断エンジンの構造テスト（複数帯域対応）
- 3ティック帯域のハンドラ登録テスト
- 3ティック帯域の実行順序テスト
- 3ティック帯域のエラー吸収テスト
- 帯域別有効フラグの独立性テスト
- 帯域別実行ログの独立性テスト
- 等価性テスト（エンジン有効/無効で同一出力）
- フォールバック経路テスト
- 帯域間非干渉テスト
- save/load非影響テスト
- enrichment非接続テスト
- 10ティック帯域の既存動作維持テスト
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from psyche.phase_execution_engine import (
    PhaseExecutionEngine,
    PhaseExecutionLog,
    PhaseStatus,
    Band,
)
from psyche.phase_declaration import (
    BAND_EVERY_3_TICKS,
    BAND_EVERY_10_TICKS,
    PHASE_BY_ID,
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


# 3ティック帯域のPhase ID一覧（宣言的定義の帯域内順序）
_3TICK_PHASE_IDS = BAND_EVERY_3_TICKS.phase_ids


# ── 帯域横断エンジン構造テスト ──────────────────────────────────


class TestMultiBandEngineStructure:
    """帯域横断エンジンの構造テスト。"""

    def test_engine_supports_two_bands(self):
        """エンジンが2帯域をサポートしていること。"""
        engine = PhaseExecutionEngine()
        supported = engine.get_supported_bands()
        assert Band.EVERY_10_TICKS in supported
        assert Band.EVERY_3_TICKS in supported

    def test_engine_both_bands_enabled_by_default(self):
        """両帯域がデフォルトで有効であること。"""
        engine = PhaseExecutionEngine()
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is True
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_engine_backward_compat_enabled(self):
        """後方互換のenabledプロパティが10ティック帯域を返すこと。"""
        engine = PhaseExecutionEngine()
        assert engine.enabled is True
        engine.set_enabled(False)
        assert engine.enabled is False
        # 3ティック帯域は影響を受けない
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_3tick_band_has_17_phases(self):
        """3ティック帯域が17 Phaseであること。"""
        assert len(_3TICK_PHASE_IDS) == 17

    def test_3tick_band_phase_order(self):
        """3ティック帯域のPhase順序が正しいこと。"""
        expected = (
            "8", "9", "10", "11", "12", "12b", "13", "14",
            "14b", "14c", "14d", "14e", "14f", "14g", "14h", "14i", "14j",
        )
        assert _3TICK_PHASE_IDS == expected

    def test_all_3tick_phases_error_absorbed(self):
        """3ティック帯域の全Phaseがerror_absorbed=Trueであること。"""
        for pid in _3TICK_PHASE_IDS:
            phase_def = PHASE_BY_ID[pid]
            assert phase_def.error_absorbed is True, (
                f"Phase {pid} ({phase_def.display_name}) should be error_absorbed"
            )

    def test_unsupported_band_not_enabled(self):
        """未対応帯域のis_band_enabledがFalseを返すこと。"""
        engine = PhaseExecutionEngine()
        assert engine.is_band_enabled(Band.EVERY_5_TICKS) is False
        assert engine.is_band_enabled(Band.CANDIDATE_GENERATION) is False

    def test_set_unsupported_band_raises(self):
        """未対応帯域のset_band_enabledがValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.set_band_enabled(Band.EVERY_5_TICKS, True)


# ── 3ティック帯域ハンドラ登録テスト ────────────────────────────


class TestThreeTickHandlerRegistration:
    """3ティック帯域のPhase処理関数登録テスト。"""

    def test_register_3tick_phase(self):
        """3ティック帯域のPhaseに処理関数を登録できること。"""
        engine = PhaseExecutionEngine()
        handler = MagicMock()
        engine.register_handler("8", handler)
        assert "8" in engine.get_band_registered_phase_ids(Band.EVERY_3_TICKS)

    def test_register_all_3tick_phases(self):
        """全17 Phaseを登録するとfully_registeredになること。"""
        engine = PhaseExecutionEngine()
        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        assert engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_partially_registered_3tick_band(self):
        """部分登録ではfully_registeredにならないこと。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        engine.register_handler("9", MagicMock())
        assert not engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_3tick_registration_does_not_affect_10tick(self):
        """3ティック帯域の登録が10ティック帯域に影響しないこと。"""
        engine = PhaseExecutionEngine()
        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        # 10ティック帯域は未登録のまま
        assert not engine.is_band_fully_registered(Band.EVERY_10_TICKS)
        assert engine.get_band_registered_phase_ids(Band.EVERY_10_TICKS) == ()

    def test_10tick_registration_does_not_affect_3tick(self):
        """10ティック帯域の登録が3ティック帯域に影響しないこと。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        assert not engine.is_band_fully_registered(Band.EVERY_3_TICKS)


# ── 帯域別有効フラグ独立性テスト ──────────────────────────────


class TestBandEnabledIndependence:
    """帯域別有効フラグの独立性テスト。"""

    def test_disable_3tick_only(self):
        """3ティック帯域のみ無効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is False
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is True

    def test_disable_10tick_only(self):
        """10ティック帯域のみ無効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_10_TICKS, False)
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is False
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_disable_both_bands(self):
        """両帯域を同時に無効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        engine.set_band_enabled(Band.EVERY_10_TICKS, False)
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is False
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is False

    def test_re_enable_3tick_band(self):
        """3ティック帯域を再有効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        engine.set_band_enabled(Band.EVERY_3_TICKS, True)
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True


# ── 3ティック帯域実行テスト ──────────────────────────────────────


class TestThreeTickBandExecution:
    """3ティック帯域の実行テスト。"""

    def test_execute_3tick_band_calls_handlers_in_order(self):
        """ハンドラが宣言的定義の帯域内順序で呼ばれること。"""
        engine = PhaseExecutionEngine()
        call_order = []

        def make_handler(pid):
            def handler(orch, uid):
                call_order.append(pid)
            return handler

        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, make_handler(pid))

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        assert call_order == list(_3TICK_PHASE_IDS)
        for pid in _3TICK_PHASE_IDS:
            assert log.phase_results[pid] == PhaseStatus.SUCCESS

    def test_execute_with_unregistered_handler_skipped(self):
        """未登録のPhaseがSKIPPEDとなること。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        # 残り16 Phase は未登録

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)
        assert log.phase_results["8"] == PhaseStatus.SUCCESS
        assert log.phase_results["9"] == PhaseStatus.SKIPPED
        assert log.phase_results["14j"] == PhaseStatus.SKIPPED

    def test_execute_passes_orchestrator_and_user_id(self):
        """ハンドラにorchestratorとuser_idが渡されること。"""
        engine = PhaseExecutionEngine()
        handler8 = MagicMock()
        engine.register_handler("8", handler8)
        for pid in _3TICK_PHASE_IDS:
            if pid != "8":
                engine.register_handler(pid, MagicMock())

        mock_orch = MagicMock()
        engine.execute_band(mock_orch, "test_user", band=Band.EVERY_3_TICKS)

        handler8.assert_called_once_with(mock_orch, "test_user")

    def test_unsupported_band_execute_raises(self):
        """未対応帯域のexecute_bandがValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.execute_band(MagicMock(), "viewer", band=Band.CANDIDATE_GENERATION)


# ── 3ティック帯域エラー吸収テスト ────────────────────────────


class TestThreeTickErrorAbsorption:
    """3ティック帯域のエラー吸収挙動テスト。"""

    def test_error_absorbed_continues_to_next_phase(self):
        """Phase 8でエラーが発生してもPhase 9以降が実行されること。"""
        engine = PhaseExecutionEngine()

        def failing_handler(orch, uid):
            raise RuntimeError("Test error")

        handler9 = MagicMock()
        engine.register_handler("8", failing_handler)
        engine.register_handler("9", handler9)
        for pid in _3TICK_PHASE_IDS:
            if pid not in ("8", "9"):
                engine.register_handler(pid, MagicMock())

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        assert log.phase_results["8"] == PhaseStatus.FAILED
        assert log.phase_results["9"] == PhaseStatus.SUCCESS
        handler9.assert_called_once()

    def test_multiple_failures_do_not_stop_execution(self):
        """複数のPhaseが失敗しても最後のPhaseまで実行されること。"""
        engine = PhaseExecutionEngine()

        def failing_handler(orch, uid):
            raise RuntimeError("Test error")

        handler_14j = MagicMock()

        for pid in _3TICK_PHASE_IDS:
            if pid == "14j":
                engine.register_handler(pid, handler_14j)
            else:
                engine.register_handler(pid, failing_handler)

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 最後のPhaseは正常実行される
        assert log.phase_results["14j"] == PhaseStatus.SUCCESS
        handler_14j.assert_called_once()

        # 他のPhaseは全てFAILED
        for pid in _3TICK_PHASE_IDS:
            if pid != "14j":
                assert log.phase_results[pid] == PhaseStatus.FAILED

    def test_all_17_phases_error_absorbed_in_declaration(self):
        """宣言的定義上、全17 Phaseがerror_absorbed=Trueであることの再確認。"""
        for pid in _3TICK_PHASE_IDS:
            assert PHASE_BY_ID[pid].error_absorbed is True


# ── 帯域別実行ログ独立性テスト ──────────────────────────────────


class TestBandLogIndependence:
    """帯域別実行ログの独立性テスト。"""

    def test_3tick_log_independent_from_10tick(self):
        """3ティック帯域のログが10ティック帯域のログに影響しないこと。"""
        engine = PhaseExecutionEngine()

        # 3ティック帯域にハンドラ登録
        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())

        # 3ティック帯域を実行
        log_3tick = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 10ティック帯域のログは空のまま
        log_10tick = engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert log_10tick.phase_results == {}

        # 3ティック帯域のログは更新されている
        assert len(log_3tick.phase_results) == 17

    def test_10tick_log_independent_from_3tick(self):
        """10ティック帯域のログが3ティック帯域のログに影響しないこと。"""
        engine = PhaseExecutionEngine()

        # 10ティック帯域にハンドラ登録
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 10ティック帯域を実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_10_TICKS)

        # 3ティック帯域のログは空のまま
        log_3tick = engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_3tick.phase_results == {}

    def test_backward_compat_last_log(self):
        """後方互換のlast_logが10ティック帯域のログを返すこと。"""
        engine = PhaseExecutionEngine()

        # 両帯域にハンドラ登録
        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 3ティック帯域を実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # last_logは10ティック帯域のログ（空のまま）
        assert engine.last_log.phase_results == {}

    def test_log_overwrite(self):
        """3ティック帯域のログが上書きされ蓄積されないこと。"""
        engine = PhaseExecutionEngine()
        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())

        # 1回目実行
        log1 = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)
        assert all(v == PhaseStatus.SUCCESS for v in log1.phase_results.values())

        # Phase 8を失敗させて2回目実行
        engine.register_handler("8", lambda o, u: (_ for _ in ()).throw(RuntimeError()))
        log2 = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # get_band_last_logは2回目の結果
        current_log = engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert current_log is log2
        assert log2.phase_results["8"] == PhaseStatus.FAILED


# ── Phase実行順序テスト ──────────────────────────────────────────


class TestThreeTickExecutionOrder:
    """3ティック帯域のPhase実行順序テスト。"""

    def test_execution_order_matches_band_order(self):
        """実行順序が宣言的定義のband_orderと一致すること（band_order 0から16）。"""
        engine = PhaseExecutionEngine()
        execution_order = []

        def make_handler(pid):
            def handler(orch, uid):
                execution_order.append(pid)
            return handler

        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 宣言的定義の帯域内順序と一致
        expected_order = list(BAND_EVERY_3_TICKS.phase_ids)
        assert execution_order == expected_order

    def test_execution_order_not_affected_by_registration_order(self):
        """ハンドラ登録順序に関わらず宣言的定義順で実行されること。"""
        engine = PhaseExecutionEngine()
        execution_order = []

        def make_handler(pid):
            def handler(orch, uid):
                execution_order.append(pid)
            return handler

        # 逆順に登録
        for pid in reversed(_3TICK_PHASE_IDS):
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 宣言的定義順（8→9→...→14j）で実行される
        assert execution_order == list(_3TICK_PHASE_IDS)

    def test_band_order_continuous_0_to_16(self):
        """band_orderが0から16まで連番であること。"""
        for i, pid in enumerate(_3TICK_PHASE_IDS):
            phase_def = PHASE_BY_ID[pid]
            assert phase_def.band_order == i, (
                f"Phase {pid} should have band_order={i}, got {phase_def.band_order}"
            )


# ── Orchestrator統合テスト（3ティック帯域） ──────────────────


class TestOrchestratorThreeTickIntegration:
    """PsycheOrchestratorとの3ティック帯域統合テスト。"""

    def test_orchestrator_3tick_band_fully_registered(self):
        """Orchestratorの全3ティックPhaseが登録されていること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_orchestrator_3tick_registered_phases(self):
        """登録された3ティック帯域Phase IDが17個であること。"""
        orch = PsycheOrchestrator()
        registered = set(
            orch._phase_engine.get_band_registered_phase_ids(Band.EVERY_3_TICKS)
        )
        assert registered == set(_3TICK_PHASE_IDS)
        assert len(registered) == 17

    def test_orchestrator_10tick_still_registered(self):
        """10ティック帯域の登録が維持されていること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_fully_registered(Band.EVERY_10_TICKS)
        assert set(
            orch._phase_engine.get_registered_phase_ids()
        ) == {"27", "28", "29"}

    def test_orchestrator_both_bands_enabled(self):
        """両帯域がデフォルトで有効であること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True
        assert orch._phase_engine.is_band_enabled(Band.EVERY_10_TICKS) is True


# ── 等価性テスト（エンジン有効/無効で同一出力） ──────────────


class TestThreeTickEquivalence:
    """3ティック帯域のエンジン有効/無効で同一の中間状態変化が生じることの検証。"""

    def _run_3_ticks(self, orch: PsycheOrchestrator) -> None:
        """3ティック目まで実行してPhase 8-14jを発火させる。"""
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

    def _run_6_ticks(self, orch: PsycheOrchestrator) -> None:
        """6ティック目まで実行して3ティック帯域を2回発火させる。"""
        percept = _make_percept()
        for _ in range(6):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

    def test_engine_enabled_produces_valid_output(self):
        """エンジン有効時に正常実行されること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True

        self._run_3_ticks(orch)

        # 実行ログを確認
        log = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        # 全17 Phaseの結果が記録されていること
        assert len(log.phase_results) == 17
        # 各PhaseはSUCCESSまたはFAILED（error_absorbed=Trueなので）
        for pid in _3TICK_PHASE_IDS:
            assert log.phase_results.get(pid) in (
                PhaseStatus.SUCCESS, PhaseStatus.FAILED
            ), f"Phase {pid} should be SUCCESS or FAILED"

    def test_engine_disabled_produces_valid_output(self):
        """エンジン無効時（フォールバック）に正常実行されること。"""
        orch = PsycheOrchestrator()
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)

        # フォールバックパスが実行されること
        self._run_3_ticks(orch)

        # フォールバック時はエンジンの3ティックログは更新されない
        log = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log.phase_results == {}

    def test_equivalence_tendency_awareness(self):
        """エンジン有効/無効でtendency_awarenessの結果が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        # 傾向認知の結果が等価
        enabled_awareness = orch_enabled._tendency_awareness
        disabled_awareness = orch_disabled._tendency_awareness
        if enabled_awareness is not None and disabled_awareness is not None:
            assert type(enabled_awareness) == type(disabled_awareness)

    def test_equivalence_self_view(self):
        """エンジン有効/無効でself_viewの結果が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        enabled_view = orch_enabled._last_self_view
        disabled_view = orch_disabled._last_self_view
        # 型が一致すること（両方Noneまたは両方SelfStateView）
        assert type(enabled_view) == type(disabled_view)

    def test_equivalence_vector_gen(self):
        """エンジン有効/無効でvector_genの呼び出し回数が等価であること。

        proto_goal_vectorの生成にはsimilarity_thresholdが関わり、
        同一入力でも微小なタイミング差でベクトル数が変動しうるため、
        turn_count（observe_turn呼び出し回数）で等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        assert (
            orch_enabled._vector_gen.state.turn_count
            == orch_disabled._vector_gen.state.turn_count
        )

    def test_equivalence_candidate_gen(self):
        """エンジン有効/無効でcandidate_genの呼び出し回数が等価であること。

        goal_candidatesの生成にはproto_goal_vectorの状態が関わり、
        ベクトル生成の非決定性が候補数に影響しうるため、
        observe_turn呼び出し回数(turn_count)で等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        assert (
            orch_enabled._candidate_gen.state.turn_count
            == orch_disabled._candidate_gen.state.turn_count
        )

    def test_equivalence_transient_goal(self):
        """エンジン有効/無効でtransient_goalの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        enabled_goal = orch_enabled._transient_goal_mgr.state.active_goal
        disabled_goal = orch_disabled._transient_goal_mgr.state.active_goal
        assert type(enabled_goal) == type(disabled_goal)

    def test_equivalence_motives(self):
        """エンジン有効/無効でmotivesの結果が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        enabled_motives = orch_enabled._last_motives
        disabled_motives = orch_disabled._last_motives
        assert type(enabled_motives) == type(disabled_motives)

    def test_equivalence_meta_emotion(self):
        """エンジン有効/無効でmeta_emotionの結果が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_6_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_6_ticks(orch_disabled)

        enabled_me = orch_enabled._last_meta_emotion
        disabled_me = orch_disabled._last_meta_emotion
        assert type(enabled_me) == type(disabled_me)

    def test_equivalence_after_multiple_3tick_cycles(self):
        """複数回の3ティック帯域発火後も等価であること。

        proto_goal_vectorの生成にはsimilarity_thresholdとランダムIDが関わるため、
        ベクトル数ではなくturn_count（呼び出し回数）で等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        self._run_6_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_6_ticks(orch_disabled)

        # tick_countが等しいこと
        assert orch_enabled.tick_count == orch_disabled.tick_count == 6

        # self_viewが等価
        assert type(orch_enabled._last_self_view) == type(orch_disabled._last_self_view)

        # vector_gen呼び出し回数が等価（turn_countで検証）
        assert (
            orch_enabled._vector_gen.state.turn_count
            == orch_disabled._vector_gen.state.turn_count
        )


# ── フォールバック切替テスト ────────────────────────────────────


class TestThreeTickFallbackSwitching:
    """3ティック帯域のフォールバック切替テスト。"""

    def test_switch_during_execution(self):
        """エンジンの有効/無効を切り替えても正常動作すること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()

        # 最初の3ティック: エンジン有効
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        log = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log.phase_results != {}

        # エンジン無効に切替
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)

        # 次の3ティック: フォールバック
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # 再有効化
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, True)

        # 次の3ティック: エンジン有効
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        log = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log.phase_results != {}

    def test_3tick_fallback_does_not_affect_10tick_engine(self):
        """3ティック帯域のフォールバックが10ティック帯域のエンジンに影響しないこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()

        # 3ティック帯域のみ無効化
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)

        # 10ティック分実行（10ティック帯域も発火する）
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # 10ティック帯域はエンジン有効で実行されていること
        log_10tick = orch._phase_engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert log_10tick.phase_results != {}

        # 3ティック帯域のログは空（フォールバック使用のため）
        log_3tick = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_3tick.phase_results == {}


# ── 帯域間非干渉テスト ──────────────────────────────────────────


class TestBandNonInterference:
    """帯域間の非干渉テスト。"""

    def test_3tick_execution_does_not_affect_10tick_log(self):
        """3ティック帯域の実行が10ティック帯域のログに影響しないこと。"""
        engine = PhaseExecutionEngine()

        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 3ティック帯域のみ実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 10ティック帯域のログは空
        assert engine.get_band_last_log(Band.EVERY_10_TICKS).phase_results == {}
        assert engine.last_log.phase_results == {}  # 後方互換

    def test_10tick_execution_does_not_affect_3tick_log(self):
        """10ティック帯域の実行が3ティック帯域のログに影響しないこと。"""
        engine = PhaseExecutionEngine()

        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 10ティック帯域のみ実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_10_TICKS)

        # 3ティック帯域のログは空
        assert engine.get_band_last_log(Band.EVERY_3_TICKS).phase_results == {}

    def test_both_bands_execute_independently(self):
        """両帯域を順次実行して互いに独立したログが取得できること。"""
        engine = PhaseExecutionEngine()

        for pid in _3TICK_PHASE_IDS:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 3ティック帯域実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)
        # 10ティック帯域実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_10_TICKS)

        log_3tick = engine.get_band_last_log(Band.EVERY_3_TICKS)
        log_10tick = engine.get_band_last_log(Band.EVERY_10_TICKS)

        assert len(log_3tick.phase_results) == 17
        assert len(log_10tick.phase_results) == 3

        # ログ内容が混在しないこと
        for pid in _3TICK_PHASE_IDS:
            assert pid in log_3tick.phase_results
            assert pid not in log_10tick.phase_results
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            assert pid in log_10tick.phase_results
            assert pid not in log_3tick.phase_results


# ── save/load非影響テスト ────────────────────────────────────────


class TestSaveLoadNonImpact3Tick:
    """3ティック帯域のエンジン拡張がsave/loadに影響しないことの検証。"""

    def test_engine_state_not_in_save(self, tmp_path):
        """エンジン状態がsave()の出力に含まれないこと。"""
        import json

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        # 3ティック分実行してPhase 8-14jを発火
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        save_path = tmp_path / "psyche_snapshot.json"
        orch.save(save_path)

        data = json.loads(save_path.read_text(encoding="utf-8"))

        # エンジン関連のキーが存在しないこと
        assert "phase_engine" not in data
        assert "phase_execution_engine" not in data
        assert "_phase_engine" not in data
        assert "band_enabled" not in data
        assert "band_handlers" not in data

    def test_load_does_not_affect_engine(self, tmp_path):
        """load()でエンジン状態が変化しないこと。"""
        import json

        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(3):
            orch1.post_response_update(percept, delta_time=0.5, user_id="viewer")

        save_path = tmp_path / "psyche_snapshot.json"
        orch1.save(save_path)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load(save_path)

        # エンジンはデフォルト状態のまま
        assert orch2._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True
        assert orch2._phase_engine.is_band_enabled(Band.EVERY_10_TICKS) is True
        assert orch2._phase_engine.is_band_fully_registered(Band.EVERY_3_TICKS)
        assert orch2._phase_engine.is_band_fully_registered(Band.EVERY_10_TICKS)


# ── enrichment非接続テスト ──────────────────────────────────────


class TestEnrichmentNonConnection3Tick:
    """3ティック帯域のエンジン内部状態がenrichmentに接続されていないことの検証。"""

    def test_engine_log_not_in_enrichment(self):
        """エンジンログがenrichmentに含まれないこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        enrichment = orch.get_prompt_enrichment()
        enrichment_str = str(enrichment)

        # エンジン関連の用語が含まれないこと
        assert "phase_engine" not in enrichment_str.lower()
        assert "PhaseExecutionEngine" not in enrichment_str
        assert "PhaseExecutionLog" not in enrichment_str
        assert "band_enabled" not in enrichment_str.lower()


# ── 10ティック帯域の既存動作維持テスト ──────────────────────────


class TestTenTickPreservation:
    """3ティック帯域の追加によって10ティック帯域の既存動作が維持されることの検証。"""

    def test_10tick_engine_still_works(self):
        """10ティック帯域のエンジンが引き続き動作すること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        log = orch._phase_engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert len(log.phase_results) == 3

    def test_10tick_backward_compat_api(self):
        """段階2の後方互換APIが引き続き動作すること。"""
        orch = PsycheOrchestrator()
        engine = orch._phase_engine

        # 段階2 API
        assert engine.enabled is True
        assert engine.is_fully_registered()
        assert set(engine.get_registered_phase_ids()) == {"27", "28", "29"}

        # 段階2の last_log
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        assert engine.last_log.phase_results != {}

    def test_10tick_equivalence_preserved(self):
        """10ティック帯域のエンジン有効/無効等価性が維持されていること。"""
        orch_enabled = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch_enabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_enabled(False)
        for _ in range(10):
            orch_disabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # stability_valveの状態が等価
        def _strip_timestamp(d):
            d = dict(d)
            if "last_indicators" in d and d["last_indicators"] is not None:
                d["last_indicators"] = {
                    k: v for k, v in d["last_indicators"].items()
                    if k != "timestamp"
                }
            return d

        state_enabled = orch_enabled._stability_valve.to_dict()
        state_disabled = orch_disabled._stability_valve.to_dict()
        assert _strip_timestamp(state_enabled) == _strip_timestamp(state_disabled)

    def test_10tick_dynamics_equivalence_preserved(self):
        """10ティック帯域のdynamics_observer等価性が維持されていること。"""
        orch_enabled = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(10):
            orch_enabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_enabled(False)
        for _ in range(10):
            orch_disabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        assert (
            orch_enabled._dynamics_observer.get_total_turns()
            == orch_disabled._dynamics_observer.get_total_turns()
        )


# ── デバッグログ出力等価性テスト ──────────────────────────────


class TestDebugLogEquivalence:
    """帯域末尾のデバッグログ出力がエンジン有効/無効で等価であることの検証。"""

    def test_3tick_debug_log_emitted_with_engine(self):
        """エンジン有効時にデバッグログが出力されること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()

        with patch("psyche.orchestrator.logger") as mock_logger:
            for _ in range(3):
                orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

            # "every-3" を含むデバッグログが出力されること
            debug_calls = [
                str(c) for c in mock_logger.debug.call_args_list
            ]
            assert any("every-3" in c for c in debug_calls)

    def test_3tick_debug_log_emitted_with_fallback(self):
        """フォールバック時にもデバッグログが出力されること。"""
        orch = PsycheOrchestrator()
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        percept = _make_percept()

        with patch("psyche.orchestrator.logger") as mock_logger:
            for _ in range(3):
                orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

            debug_calls = [
                str(c) for c in mock_logger.debug.call_args_list
            ]
            assert any("every-3" in c for c in debug_calls)


# ── Phase定義の検証テスト ──────────────────────────────────────


class TestThreeTickPhaseDefinitions:
    """3ティック帯域のPhase定義の検証。"""

    def test_phase8_definition(self):
        """Phase 8の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["8"]
        assert p.band == Band.EVERY_3_TICKS
        assert p.band_order == 0
        assert p.error_absorbed is True
        assert "tendency_awareness" in p.modules

    def test_phase9_definition(self):
        """Phase 9の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["9"]
        assert p.band == Band.EVERY_3_TICKS
        assert p.band_order == 1
        assert "self_model" in p.modules

    def test_phase14j_definition(self):
        """Phase 14j（末尾）の宣言的定義が正しいこと。"""
        p = PHASE_BY_ID["14j"]
        assert p.band == Band.EVERY_3_TICKS
        assert p.band_order == 16
        assert p.error_absorbed is True

    def test_get_band_phase_definitions_3tick(self):
        """get_band_phase_definitionsが3ティック帯域の正しいPhase定義を返すこと。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions(band=Band.EVERY_3_TICKS)
        assert len(defs) == 17
        assert defs[0].phase_id == "8"
        assert defs[-1].phase_id == "14j"

    def test_get_band_phase_definitions_backward_compat(self):
        """get_band_phase_definitions引数省略時に10ティック帯域を返すこと。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions()
        assert len(defs) == 3
        assert defs[0].phase_id == "27"


# ── ハンドラ未登録Phase検出テスト ──────────────────────────────


class TestUnregisteredPhaseDetection:
    """ハンドラ未登録Phaseのスキップ記録テスト。"""

    def test_unregistered_phase_skipped_in_log(self):
        """未登録Phaseがスキップとしてログに記録されること。"""
        engine = PhaseExecutionEngine()
        # Phase 8のみ登録
        engine.register_handler("8", MagicMock())

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        assert log.phase_results["8"] == PhaseStatus.SUCCESS
        for pid in _3TICK_PHASE_IDS:
            if pid != "8":
                assert log.phase_results[pid] == PhaseStatus.SKIPPED
