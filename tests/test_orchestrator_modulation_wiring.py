"""
tests/test_orchestrator_modulation_wiring.py

orchestrator.py への経験依存変調3モジュールの接続テスト。
各モジュールの内部ロジックは既存テストでカバー済みのため、
ここでは「orchestrator経由で正しく呼び出されること」を検証する。

対象:
1. decay_rate_modulation — セッション起動時の感情減衰速度変調
2. window_size_modulation — セッション起動時のウィンドウサイズ変調
3. firing_frequency_modulation — セッション起動時の帯域上限変調
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from psyche.orchestrator import PsycheOrchestrator

# ── Helpers ──────────────────────────────────────────────────────────

def _create_orchestrator(tmp_path: Path) -> PsycheOrchestrator:
    """テスト用のPsycheOrchestratorインスタンスを作成する。"""
    return PsycheOrchestrator(data_dir=tmp_path)


def _save_and_load(orch: PsycheOrchestrator, tmp_path: Path) -> PsycheOrchestrator:
    """save → 新インスタンス → load のサイクルを実行する。"""
    orch.save(tmp_path / "psyche_snapshot.json")
    orch2 = PsycheOrchestrator(data_dir=tmp_path)
    orch2.load(tmp_path / "psyche_snapshot.json")
    return orch2


# ── 1. Decay rate modulation tests ──────────────────────────────────

class TestDecayRateModulationWiring:
    """decay_rate_modulation がload時に呼び出されることを検証。"""

    def test_load_calls_apply_decay_rate_modulation(self, tmp_path):
        """load()内でapply_decay_rate_modulationが呼び出される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_decay_rate_modulation",
            return_value=(0.95, None),
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            mock_apply.assert_called_once()

    def test_decay_modulation_result_saved(self, tmp_path):
        """変調結果がsave/loadで永続化される。"""
        orch = _create_orchestrator(tmp_path)
        # 変調結果を手動設定
        orch._decay_rate_modulated = 0.9475
        orch.save(tmp_path / "psyche_snapshot.json")

        # ファイル内に保存されていることを確認
        data = json.loads((tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8"))
        assert "decay_rate_modulated" in data
        assert abs(data["decay_rate_modulated"] - 0.9475) < 1e-6

    def test_decay_modulation_previous_rate_passed(self, tmp_path):
        """前回の変調結果がprevious_modulated_rateとして渡される。"""
        orch = _create_orchestrator(tmp_path)
        orch._decay_rate_modulated = 0.9475
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_decay_rate_modulation",
            return_value=(0.948, None),
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            # previous_modulated_rateとして0.9475が渡されることを確認
            call_args = mock_apply.call_args
            assert call_args is not None
            # 第3引数 previous_modulated_rate
            assert abs(call_args[1].get("previous_modulated_rate", call_args[0][2]) - 0.9475) < 1e-6

    def test_decay_modulation_no_previous_rate(self, tmp_path):
        """初回load時はprevious_modulated_rateがNone。"""
        orch = _create_orchestrator(tmp_path)
        # _decay_rate_modulated を設定しない（初回）
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_decay_rate_modulation",
            return_value=(0.95, None),
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            call_args = mock_apply.call_args
            assert call_args is not None
            # previous_modulated_rate が None
            assert call_args[0][2] is None


# ── 2. Window size modulation tests ─────────────────────────────────

class TestWindowSizeModulationWiring:
    """window_size_modulation がload時に呼び出されることを検証。"""

    def test_load_calls_apply_window_size_modulation(self, tmp_path):
        """load()内でapply_window_size_modulationが呼び出される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_window_size_modulation",
            return_value={"window_size_25": 25, "window_size_30": 30, "window_size_50": 50},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            mock_apply.assert_called_once()

    def test_window_modulation_result_saved(self, tmp_path):
        """変調結果がsave/loadで永続化される。"""
        orch = _create_orchestrator(tmp_path)
        orch._window_size_modulated = {"window_size_25": 28, "window_size_30": 33}
        orch.save(tmp_path / "psyche_snapshot.json")

        data = json.loads((tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8"))
        assert "window_size_modulated" in data
        assert data["window_size_modulated"]["window_size_25"] == 28

    def test_window_modulation_previous_values_passed(self, tmp_path):
        """前回の変調結果がprevious_modulated_valuesとして渡される。"""
        orch = _create_orchestrator(tmp_path)
        orch._window_size_modulated = {"window_size_25": 28, "window_size_30": 33}
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_window_size_modulation",
            return_value={"window_size_25": 27},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            call_args = mock_apply.call_args
            assert call_args is not None
            prev = call_args[1].get("previous_modulated_values", call_args[0][1])
            assert prev["window_size_25"] == 28

    def test_window_modulation_frb_state_passed(self, tmp_path):
        """frb_stateが正しく渡される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_window_size_modulation",
            return_value={},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            call_args = mock_apply.call_args
            assert call_args is not None
            from psyche.forgetting_recall_balance import ForgettingRecallBalanceState
            # 第1引数がForgettingRecallBalanceState型であること
            assert isinstance(call_args[0][0], ForgettingRecallBalanceState)


# ── 3. Firing frequency modulation tests ────────────────────────────

class TestFiringFrequencyModulationWiring:
    """firing_frequency_modulation がload時に呼び出されることを検証。"""

    def test_load_calls_apply_firing_frequency_modulation(self, tmp_path):
        """load()内でapply_firing_frequency_modulationが呼び出される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_firing_frequency_modulation",
            return_value={},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            mock_apply.assert_called_once()

    def test_firing_frequency_receives_fire_counts(self, tmp_path):
        """pathway_fire_countsが正しく渡される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_firing_frequency_modulation",
            return_value={},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            call_args = mock_apply.call_args
            assert call_args is not None
            fire_counts = call_args[0][0]
            assert isinstance(fire_counts, dict)

    def test_firing_frequency_receives_configs(self, tmp_path):
        """memory_emotion_return_configとother_hypothesis_configが渡される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch(
            "psyche.orchestrator.apply_firing_frequency_modulation",
            return_value={},
        ) as mock_apply:
            orch2.load(tmp_path / "psyche_snapshot.json")
            call_args = mock_apply.call_args
            assert call_args is not None
            # memory_emotion_return_config (2nd arg)
            mer_config = call_args[0][1]
            assert hasattr(mer_config, "per_candidate_max_delta")
            # other_hypothesis_emotion_return_config (3rd arg)
            oher_config = call_args[0][2]
            assert hasattr(oher_config, "per_candidate_max_delta")

    def test_firing_frequency_no_save_needed(self, tmp_path):
        """firing_frequency_modulationの結果はsave不要（再計算で復元）。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        data = json.loads((tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8"))
        # firing_frequency関連の保存フィールドが存在しないことを確認
        assert "firing_frequency_modulation" not in data


# ── 4. Integration tests ────────────────────────────────────────────

class TestModulationIntegration:
    """3つの変調モジュールが統合的に動作することを検証。"""

    def test_all_modulations_called_on_load(self, tmp_path):
        """load()で3つの変調が全て呼び出される。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch("psyche.orchestrator.apply_decay_rate_modulation", return_value=(0.95, None)) as m1, \
             patch("psyche.orchestrator.apply_window_size_modulation", return_value={}) as m2, \
             patch("psyche.orchestrator.apply_firing_frequency_modulation", return_value={}) as m3:
            orch2.load(tmp_path / "psyche_snapshot.json")
            m1.assert_called_once()
            m2.assert_called_once()
            m3.assert_called_once()

    def test_modulation_failure_does_not_block_load(self, tmp_path):
        """変調処理が失敗してもload全体は成功する。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        with patch("psyche.orchestrator.apply_decay_rate_modulation", side_effect=RuntimeError("test")), \
             patch("psyche.orchestrator.apply_window_size_modulation", side_effect=RuntimeError("test")), \
             patch("psyche.orchestrator.apply_firing_frequency_modulation", side_effect=RuntimeError("test")):
            result = orch2.load(tmp_path / "psyche_snapshot.json")
            assert result is True

    def test_save_load_cycle_preserves_modulation_state(self, tmp_path):
        """save → load サイクルで変調状態が保持される。"""
        orch = _create_orchestrator(tmp_path)
        orch._decay_rate_modulated = 0.9475
        orch._window_size_modulated = {"window_size_25": 28}
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = _create_orchestrator(tmp_path)
        # パッチで変調結果を固定
        with patch("psyche.orchestrator.apply_decay_rate_modulation", return_value=(0.948, 0.6)), \
             patch("psyche.orchestrator.apply_window_size_modulation", return_value={"window_size_25": 27}), \
             patch("psyche.orchestrator.apply_firing_frequency_modulation", return_value={}):
            orch2.load(tmp_path / "psyche_snapshot.json")

        assert abs(orch2._decay_rate_modulated - 0.948) < 1e-6
        assert orch2._window_size_modulated["window_size_25"] == 27


# ── 5. Edge cases ───────────────────────────────────────────────────

class TestModulationEdgeCases:
    """エッジケース。"""

    def test_init_has_default_modulation_values(self):
        """初期化時のデフォルト値が設定されている。"""
        with tempfile.TemporaryDirectory() as td:
            orch = PsycheOrchestrator(data_dir=Path(td))
            assert orch._decay_rate_modulated is None
            assert orch._window_size_modulated is None

    def test_modulation_with_empty_backdrop(self, tmp_path):
        """BackdropStateが空の場合でも変調は安全に動作する。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        # 実際のモジュールを使って呼び出し（モックなし）
        result = orch2.load(tmp_path / "psyche_snapshot.json")
        assert result is True
        # 空のBackdropStateの場合、decay_rateは基準値のまま
        # (amplitude_score=Noneで変調なし)

    def test_modulation_with_empty_frb_state(self, tmp_path):
        """ForgettingRecallBalanceStateが空の場合でも変調は安全に動作する。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        result = orch2.load(tmp_path / "psyche_snapshot.json")
        assert result is True

    def test_modulation_with_zero_fire_counts(self, tmp_path):
        """発火回数が0の場合でもfiring_frequency変調は安全に動作する。"""
        orch = _create_orchestrator(tmp_path)
        orch.save(tmp_path / "psyche_snapshot.json")

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        result = orch2.load(tmp_path / "psyche_snapshot.json")
        assert result is True
