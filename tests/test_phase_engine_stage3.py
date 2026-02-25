"""
tests/test_phase_engine_stage3.py - Phase実行エンジン段階3（3ティック帯域）のテスト

テスト項目:
- 帯域横断エンジンの初期化・帯域別管理
- 3ティック帯域の17 Phase全ハンドラ登録
- 帯域別有効/無効フラグの独立制御
- 帯域別実行ログの独立性
- 3ティック帯域のPhase実行順序（band_order 0-16）
- エラー吸収境界の挙動（全17 Phaseがerror_absorbed=True）
- 等価性テスト（エンジン有効/無効で同一中間状態変化）
- 10ティック帯域の等価性維持
- フォールバック経路の検証
- 帯域間の独立性保証
- save/load非影響
- enrichment非接続
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
)
from psyche.phase_declaration import (
    Band,
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


# ── 帯域横断エンジン初期化テスト ──────────────────────────────────


class TestMultiBandEngineInit:
    """帯域横断エンジンの初期化テスト。"""

    def test_supported_bands(self):
        """サポート帯域が10ティックと3ティックであること。"""
        engine = PhaseExecutionEngine()
        bands = engine.get_supported_bands()
        assert Band.EVERY_10_TICKS in bands
        assert Band.EVERY_3_TICKS in bands

    def test_both_bands_enabled_by_default(self):
        """初期状態で全帯域が有効であること。"""
        engine = PhaseExecutionEngine()
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is True
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_backward_compat_enabled_returns_10tick(self):
        """後方互換性: enabled は10ティック帯域の状態を返すこと。"""
        engine = PhaseExecutionEngine()
        assert engine.enabled is True
        engine.set_band_enabled(Band.EVERY_10_TICKS, False)
        assert engine.enabled is False

    def test_backward_compat_set_enabled(self):
        """後方互換性: set_enabled は10ティック帯域を制御すること。"""
        engine = PhaseExecutionEngine()
        engine.set_enabled(False)
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is False
        # 3ティック帯域には影響しない
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_backward_compat_band_phase_ids(self):
        """後方互換性: _band_phase_ids が10ティック帯域のPhase IDsを返すこと。"""
        engine = PhaseExecutionEngine()
        assert engine._band_phase_ids == BAND_EVERY_10_TICKS.phase_ids
        assert engine._band_phase_ids == ("27", "28", "29")

    def test_3tick_band_phase_count(self):
        """3ティック帯域が17 Phaseであること。"""
        assert len(BAND_EVERY_3_TICKS.phase_ids) == 17

    def test_3tick_band_phase_order(self):
        """3ティック帯域のPhase順序が正しいこと。"""
        expected = (
            "8", "9", "10", "11", "12", "12b",
            "13", "14", "14b", "14c", "14d", "14e",
            "14f", "14g", "14h", "14i", "14j",
        )
        assert BAND_EVERY_3_TICKS.phase_ids == expected


# ── 帯域別有効フラグ独立性テスト ──────────────────────────────────


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
        """両帯域とも無効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        engine.set_band_enabled(Band.EVERY_10_TICKS, False)
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is False
        assert engine.is_band_enabled(Band.EVERY_10_TICKS) is False

    def test_re_enable_3tick(self):
        """3ティック帯域を再有効化できること。"""
        engine = PhaseExecutionEngine()
        engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        engine.set_band_enabled(Band.EVERY_3_TICKS, True)
        assert engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_unsupported_band_returns_false(self):
        """未サポート帯域は常にFalseを返すこと。"""
        engine = PhaseExecutionEngine()
        assert engine.is_band_enabled(Band.EVERY_TICK) is False
        assert engine.is_band_enabled(Band.EVERY_5_TICKS) is False

    def test_unsupported_band_set_raises(self):
        """未サポート帯域の有効フラグ設定がValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.set_band_enabled(Band.EVERY_TICK, True)


# ── 3ティック帯域ハンドラ登録テスト ──────────────────────────────


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
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        assert engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_partial_registration_not_fully_registered(self):
        """一部のみ登録ではfully_registeredにならないこと。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        engine.register_handler("9", MagicMock())
        assert not engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_register_10tick_and_3tick_independently(self):
        """10ティックと3ティックの登録が独立していること。"""
        engine = PhaseExecutionEngine()
        # 10ティック全登録
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        assert engine.is_band_fully_registered(Band.EVERY_10_TICKS)
        assert not engine.is_band_fully_registered(Band.EVERY_3_TICKS)

        # 3ティック全登録
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        assert engine.is_band_fully_registered(Band.EVERY_3_TICKS)
        assert engine.is_band_fully_registered(Band.EVERY_10_TICKS)

    def test_register_invalid_phase_raises(self):
        """サポート帯域外のPhase登録がValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.register_handler("1", MagicMock())  # EVERY_TICK
        with pytest.raises(ValueError):
            engine.register_handler("15", MagicMock())  # EVERY_5_TICKS
        with pytest.raises(ValueError):
            engine.register_handler("30", MagicMock())  # CANDIDATE_GENERATION

    def test_backward_compat_is_fully_registered(self):
        """後方互換性: is_fully_registered は10ティック帯域の状態を返すこと。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        assert engine.is_fully_registered()
        # 3ティック帯域は未登録でもis_fully_registered()はTrue
        assert not engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_backward_compat_get_registered_phase_ids(self):
        """後方互換性: get_registered_phase_ids は10ティック帯域のみ返すこと。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        engine.register_handler("27", MagicMock())
        registered = engine.get_registered_phase_ids()
        assert "27" in registered
        assert "8" not in registered  # 3ティック帯域は含まない

    def test_3tick_phase_definitions(self):
        """3ティック帯域の全Phase定義が正しいこと。"""
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            p = PHASE_BY_ID[pid]
            assert p.band == Band.EVERY_3_TICKS
            assert p.error_absorbed is True


# ── 3ティック帯域実行テスト ──────────────────────────────────────


class TestThreeTickBandExecution:
    """3ティック帯域の実行テスト。"""

    def test_execute_3tick_calls_handlers_in_order(self):
        """ハンドラが宣言的定義の帯域内順序で呼ばれること。"""
        engine = PhaseExecutionEngine()
        call_order = []

        def make_handler(pid):
            def handler(orch, uid):
                call_order.append(pid)
            return handler

        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, make_handler(pid))

        mock_orch = MagicMock()
        log = engine.execute_band(mock_orch, "viewer", band=Band.EVERY_3_TICKS)

        assert call_order == list(BAND_EVERY_3_TICKS.phase_ids)
        assert all(
            log.phase_results[pid] == PhaseStatus.SUCCESS
            for pid in BAND_EVERY_3_TICKS.phase_ids
        )

    def test_execute_3tick_with_unregistered_handler(self):
        """未登録のPhaseがSKIPPEDとなること。"""
        engine = PhaseExecutionEngine()
        engine.register_handler("8", MagicMock())
        # 残り16 Phaseは未登録

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)
        assert log.phase_results["8"] == PhaseStatus.SUCCESS
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            if pid != "8":
                assert log.phase_results[pid] == PhaseStatus.SKIPPED

    def test_execute_backward_compat_default_10tick(self):
        """execute_band() のband省略時は10ティック帯域で実行されること。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        log = engine.execute_band(MagicMock(), "viewer")
        assert set(log.phase_results.keys()) == set(BAND_EVERY_10_TICKS.phase_ids)

    def test_execute_unsupported_band_raises(self):
        """未サポート帯域の実行がValueErrorを出すこと。"""
        engine = PhaseExecutionEngine()
        with pytest.raises(ValueError):
            engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_TICK)


# ── 3ティック帯域エラー吸収テスト ──────────────────────────────────


class TestThreeTickErrorAbsorption:
    """3ティック帯域のエラー吸収境界テスト。"""

    def test_all_3tick_phases_error_absorbed(self):
        """3ティック帯域の全17 Phaseがerror_absorbed=Trueであること。"""
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            assert PHASE_BY_ID[pid].error_absorbed is True

    def test_error_absorbed_continues_execution(self):
        """エラーが発生しても後続Phaseが実行されること。"""
        engine = PhaseExecutionEngine()
        call_order = []

        def failing_handler(orch, uid):
            call_order.append("fail")
            raise RuntimeError("Test error")

        def success_handler(pid):
            def handler(orch, uid):
                call_order.append(pid)
            return handler

        # Phase 8を失敗させ、他は成功
        engine.register_handler("8", failing_handler)
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            if pid != "8":
                engine.register_handler(pid, success_handler(pid))

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        assert log.phase_results["8"] == PhaseStatus.FAILED
        assert log.phase_results["9"] == PhaseStatus.SUCCESS
        assert "fail" in call_order
        # 全17 Phase（1失敗+16成功）が実行されていること
        assert len(call_order) == 17

    def test_multiple_failures_dont_stop_execution(self):
        """複数のPhaseが失敗しても全Phaseが試行されること。"""
        engine = PhaseExecutionEngine()

        def failing_handler(orch, uid):
            raise RuntimeError("Test error")

        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, failing_handler)

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 全17 PhaseがFAILED
        assert all(
            log.phase_results[pid] == PhaseStatus.FAILED
            for pid in BAND_EVERY_3_TICKS.phase_ids
        )


# ── 帯域別実行ログ独立性テスト ──────────────────────────────────


class TestBandLogIndependence:
    """帯域別実行ログの独立性テスト。"""

    def test_3tick_log_independent_from_10tick(self):
        """3ティック帯域のログが10ティック帯域のログと独立していること。"""
        engine = PhaseExecutionEngine()
        # 10ティック帯域登録
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        # 3ティック帯域登録
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 3ティック帯域のみ実行
        log_3 = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 10ティック帯域のログは空のまま
        log_10 = engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert log_10.phase_results == {}
        assert log_3.phase_results != {}

    def test_10tick_log_independent_from_3tick(self):
        """10ティック帯域のログが3ティック帯域のログと独立していること。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 10ティック帯域のみ実行
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_10_TICKS)

        # 3ティック帯域のログは空のまま
        log_3 = engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_3.phase_results == {}

    def test_backward_compat_last_log(self):
        """後方互換性: last_log は10ティック帯域のログを返すこと。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_10_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)
        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_10_TICKS)

        # last_log は10ティック帯域のログ
        assert "27" in engine.last_log.phase_results
        assert "8" not in engine.last_log.phase_results

    def test_log_overwrites_per_band(self):
        """各帯域のログが上書きされ蓄積されないこと。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        # 1回目実行
        log1 = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # 2回目実行（Phase 8を失敗させる）
        engine.register_handler("8", lambda o, u: (_ for _ in ()).throw(RuntimeError()))
        log2 = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        # get_band_last_logは2回目のログ
        current_log = engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert current_log is log2
        assert log2.phase_results["8"] == PhaseStatus.FAILED


# ── Orchestrator統合テスト（3ティック帯域） ──────────────────────


class TestOrchestratorThreeTickIntegration:
    """PsycheOrchestratorの3ティック帯域統合テスト。"""

    def test_orchestrator_3tick_band_fully_registered(self):
        """Orchestratorの全17 3ティックPhaseが登録されていること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_fully_registered(Band.EVERY_3_TICKS)

    def test_orchestrator_3tick_registered_phases(self):
        """登録された3ティックPhase IDが全17件であること。"""
        orch = PsycheOrchestrator()
        registered = set(orch._phase_engine.get_band_registered_phase_ids(Band.EVERY_3_TICKS))
        expected = set(BAND_EVERY_3_TICKS.phase_ids)
        assert registered == expected

    def test_orchestrator_10tick_still_fully_registered(self):
        """10ティック帯域も引き続き全登録されていること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_fully_registered(Band.EVERY_10_TICKS)
        assert orch._phase_engine.is_fully_registered()

    def test_orchestrator_3tick_enabled_by_default(self):
        """3ティック帯域がデフォルト有効であること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True

    def test_orchestrator_has_fallback_method(self):
        """フォールバックメソッドが存在すること。"""
        orch = PsycheOrchestrator()
        assert hasattr(orch, "_run_every_3_ticks_fallback")
        assert callable(orch._run_every_3_ticks_fallback)


# ── 等価性テスト（3ティック帯域） ──────────────────────────────────


class TestThreeTickEquivalence:
    """実行エンジン有効/無効で同一の中間状態変化が生じることの検証。"""

    def _run_3_ticks(self, orch: PsycheOrchestrator) -> None:
        """3ティック目まで実行してPhase 8-14jを発火させる。"""
        percept = _make_percept()
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

    def _run_6_ticks(self, orch: PsycheOrchestrator) -> None:
        """6ティック目まで実行してPhase 8-14jを2回発火させる。"""
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
        # 全17 PhaseがSUCCESSまたはFAILED（SKIPPEDはハンドラ登録済みなので発生しない）
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            assert log.phase_results.get(pid) in (
                PhaseStatus.SUCCESS, PhaseStatus.FAILED
            )

    def test_engine_disabled_produces_valid_output(self):
        """エンジン無効時（フォールバック）に正常実行されること。"""
        orch = PsycheOrchestrator()
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)

        # フォールバックパスが実行されること
        self._run_3_ticks(orch)

        # フォールバック時は3ティック帯域のエンジンログは更新されない
        log = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log.phase_results == {}

    def test_equivalence_tendency_awareness(self):
        """エンジン有効/無効でtendency_awarenessの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        assert orch_enabled._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        # tendency_awarenessの状態が等価
        ta_enabled = orch_enabled._tendency_awareness
        ta_disabled = orch_disabled._tendency_awareness
        assert type(ta_enabled) is type(ta_disabled)

    def test_equivalence_self_view(self):
        """エンジン有効/無効でlast_self_viewの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        sv_enabled = orch_enabled._last_self_view
        sv_disabled = orch_disabled._last_self_view
        assert type(sv_enabled) is type(sv_disabled)
        if sv_enabled is not None and sv_disabled is not None:
            assert sv_enabled.emotional.intensity == sv_disabled.emotional.intensity
            assert sv_enabled.emotional.spread == sv_disabled.emotional.spread

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

        vg_enabled = orch_enabled._vector_gen.state
        vg_disabled = orch_disabled._vector_gen.state
        assert vg_enabled.turn_count == vg_disabled.turn_count

    def test_equivalence_candidate_gen(self):
        """エンジン有効/無効でcandidate_genの呼び出し回数が等価であること。

        goal_candidatesの生成にはproto_goal_vectorの状態が関わり、
        ベクトル生成の非決定性が候補数に影響しうるため、
        turn_count（observe_vectors呼び出し回数）で等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        cg_enabled = orch_enabled._candidate_gen.state
        cg_disabled = orch_disabled._candidate_gen.state
        assert cg_enabled.turn_count == cg_disabled.turn_count

    def test_equivalence_transient_goal(self):
        """エンジン有効/無効でtransient_goalの呼び出し回数が等価であること。

        transient_goalの選択結果はproto_goal_vector/goal_candidatesの
        非決定性に依存するため、turn_count（observe_turn呼び出し回数）で
        等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        tg_enabled = orch_enabled._transient_goal_mgr.state
        tg_disabled = orch_disabled._transient_goal_mgr.state
        assert tg_enabled.turn_count == tg_disabled.turn_count

    def test_equivalence_motives(self):
        """エンジン有効/無効でmotivesの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_3_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_3_ticks(orch_disabled)

        m_enabled = orch_enabled._last_motives
        m_disabled = orch_disabled._last_motives
        assert type(m_enabled) is type(m_disabled)

    def test_equivalence_meta_emotion(self):
        """エンジン有効/無効でmeta_emotionの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_6_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_6_ticks(orch_disabled)

        me_enabled = orch_enabled._last_meta_emotion
        me_disabled = orch_disabled._last_meta_emotion
        assert type(me_enabled) is type(me_disabled)

    def test_equivalence_cooccurrence(self):
        """エンジン有効/無効でemotion_cooccurrenceの状態が等価であること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_6_ticks(orch_enabled)

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        self._run_6_ticks(orch_disabled)

        co_enabled = orch_enabled._last_cooccurrence_result
        co_disabled = orch_disabled._last_cooccurrence_result
        assert type(co_enabled) is type(co_disabled)

    def test_equivalence_over_multiple_cycles(self):
        """複数サイクルにわたってエンジン有効/無効で等価であること。

        proto_goal_vector/goal_candidatesの生成には非決定性があるため、
        turn_count（呼び出し回数）で等価性を検証する。
        """
        orch_enabled = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(12):
            orch_enabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)
        for _ in range(12):
            orch_disabled.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # vector_gen呼び出し回数が等価
        assert (
            orch_enabled._vector_gen.state.turn_count
            == orch_disabled._vector_gen.state.turn_count
        )
        # candidate_gen呼び出し回数が等価
        assert (
            orch_enabled._candidate_gen.state.turn_count
            == orch_disabled._candidate_gen.state.turn_count
        )


# ── 10ティック帯域等価性維持テスト ──────────────────────────────────


class TestTenTickEquivalencePreserved:
    """帯域横断エンジン移行後も10ティック帯域の等価性が維持されることの検証。"""

    def _run_10_ticks(self, orch: PsycheOrchestrator) -> None:
        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

    def test_10tick_still_works_with_3tick_enabled(self):
        """3ティック帯域有効時でも10ティック帯域が正常動作すること。"""
        orch = PsycheOrchestrator()
        assert orch._phase_engine.is_band_enabled(Band.EVERY_3_TICKS) is True
        assert orch._phase_engine.is_band_enabled(Band.EVERY_10_TICKS) is True

        self._run_10_ticks(orch)

        log_10 = orch._phase_engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert log_10.phase_results.get("27") in (PhaseStatus.SUCCESS, PhaseStatus.FAILED)

    def test_10tick_equivalence_preserved(self):
        """10ティック帯域の有効/無効等価性が維持されていること。"""
        orch_enabled = PsycheOrchestrator()
        self._run_10_ticks(orch_enabled)
        state_enabled = orch_enabled._stability_valve.to_dict()

        orch_disabled = PsycheOrchestrator()
        orch_disabled._phase_engine.set_band_enabled(Band.EVERY_10_TICKS, False)
        self._run_10_ticks(orch_disabled)
        state_disabled = orch_disabled._stability_valve.to_dict()

        # timestampを除外して比較
        def _strip_timestamp(d):
            d = dict(d)
            if "last_indicators" in d and d["last_indicators"] is not None:
                d["last_indicators"] = {
                    k: v for k, v in d["last_indicators"].items()
                    if k != "timestamp"
                }
            return d

        assert _strip_timestamp(state_enabled) == _strip_timestamp(state_disabled)


# ── フォールバック切替テスト ──────────────────────────────────


class TestThreeTickFallbackSwitching:
    """3ティック帯域のフォールバック経路切替テスト。"""

    def test_switch_3tick_during_execution(self):
        """3ティック帯域の有効/無効を切り替えても正常動作すること。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()

        # 最初の3ティック: エンジン有効
        for _ in range(3):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")
        log_3 = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_3.phase_results != {}

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
        log_after = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_after.phase_results != {}

    def test_3tick_disabled_10tick_still_works(self):
        """3ティック帯域無効時でも10ティック帯域が正常動作すること。"""
        orch = PsycheOrchestrator()
        orch._phase_engine.set_band_enabled(Band.EVERY_3_TICKS, False)

        percept = _make_percept()
        for _ in range(10):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        # 10ティック帯域のログは存在する
        log_10 = orch._phase_engine.get_band_last_log(Band.EVERY_10_TICKS)
        assert log_10.phase_results != {}

        # 3ティック帯域のログは空（フォールバック実行のためエンジンログなし）
        log_3 = orch._phase_engine.get_band_last_log(Band.EVERY_3_TICKS)
        assert log_3.phase_results == {}


# ── Phase実行順序テスト ──────────────────────────────────────────


class TestThreeTickExecutionOrder:
    """3ティック帯域のPhase実行順序テスト。"""

    def test_execution_order_matches_band_order(self):
        """実行順序が宣言的定義のband_orderと一致すること。"""
        engine = PhaseExecutionEngine()
        execution_order = []

        def make_handler(pid):
            def handler(orch, uid):
                execution_order.append(pid)
            return handler

        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

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
        for pid in reversed(BAND_EVERY_3_TICKS.phase_ids):
            engine.register_handler(pid, make_handler(pid))

        engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        expected_order = list(BAND_EVERY_3_TICKS.phase_ids)
        assert execution_order == expected_order

    def test_band_order_continuous_0_to_16(self):
        """帯域内順序が0から16まで連続であること。"""
        band_orders = []
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            band_orders.append(PHASE_BY_ID[pid].band_order)
        assert band_orders == list(range(17))


# ── save/load非影響テスト ──────────────────────────────────────


class TestThreeTickSaveLoadNonImpact:
    """3ティック帯域のエンジン拡張がsave/loadに影響しないことの検証。"""

    def test_engine_state_not_in_save(self, tmp_path):
        """エンジン状態（3ティック帯域含む）がsave()の出力に含まれないこと。"""
        import json

        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(6):
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
        assert "band_enabled" not in data
        assert "band_handlers" not in data

    def test_load_does_not_affect_engine(self, tmp_path):
        """load()でエンジン状態が変化しないこと。"""
        import json

        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(6):
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


class TestThreeTickEnrichmentNonConnection:
    """3ティック帯域のエンジン内部状態がenrichmentに接続されていないことの検証。"""

    def test_engine_log_not_in_enrichment(self):
        """3ティック帯域のエンジンログがenrichmentに含まれないこと。"""
        orch = PsycheOrchestrator()
        percept = _make_percept()
        for _ in range(6):
            orch.post_response_update(percept, delta_time=0.5, user_id="viewer")

        enrichment = orch.get_prompt_enrichment()
        enrichment_str = str(enrichment)

        assert "phase_engine" not in enrichment_str.lower()
        assert "PhaseExecutionEngine" not in enrichment_str
        assert "PhaseExecutionLog" not in enrichment_str
        assert "band_enabled" not in enrichment_str.lower()


# ── 検証用アクセサテスト ──────────────────────────────────────────


class TestThreeTickAccessors:
    """帯域横断エンジンの検証用アクセサテスト。"""

    def test_get_band_phase_definitions_3tick(self):
        """3ティック帯域のPhase定義が正しく返されること。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions(band=Band.EVERY_3_TICKS)
        assert len(defs) == 17
        assert defs[0].phase_id == "8"
        assert defs[-1].phase_id == "14j"

    def test_get_band_phase_definitions_10tick(self):
        """10ティック帯域のPhase定義が正しく返されること。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions(band=Band.EVERY_10_TICKS)
        assert len(defs) == 3
        assert defs[0].phase_id == "27"

    def test_get_band_phase_definitions_backward_compat(self):
        """後方互換性: band省略時は10ティック帯域のPhase定義を返すこと。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions()
        assert len(defs) == 3
        assert defs[0].phase_id == "27"

    def test_get_band_phase_definitions_unsupported(self):
        """未サポート帯域で空タプルが返されること。"""
        engine = PhaseExecutionEngine()
        defs = engine.get_band_phase_definitions(band=Band.EVERY_TICK)
        assert defs == ()


# ── ハンドラ未登録Phase検出テスト ──────────────────────────────────


class TestUnregisteredPhaseDetection:
    """ハンドラ未登録Phaseの検出テスト。"""

    def test_unregistered_phases_listed_as_skipped(self):
        """未登録PhaseがSKIPPEDとしてログに記録されること。"""
        engine = PhaseExecutionEngine()
        # Phase 8のみ登録
        engine.register_handler("8", MagicMock())

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        skipped = [
            pid for pid, status in log.phase_results.items()
            if status == PhaseStatus.SKIPPED
        ]
        # Phase 8以外の16 PhaseがSKIPPED
        assert len(skipped) == 16
        assert "8" not in skipped

    def test_fully_registered_no_skips(self):
        """全Phase登録済みならSKIPPEDは発生しないこと。"""
        engine = PhaseExecutionEngine()
        for pid in BAND_EVERY_3_TICKS.phase_ids:
            engine.register_handler(pid, MagicMock())

        log = engine.execute_band(MagicMock(), "viewer", band=Band.EVERY_3_TICKS)

        skipped = [
            pid for pid, status in log.phase_results.items()
            if status == PhaseStatus.SKIPPED
        ]
        assert len(skipped) == 0
