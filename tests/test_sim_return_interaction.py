"""
tests/test_sim_return_interaction.py - 帰還経路相互作用分析シナリオのテスト

帰還経路の相互作用を分析する3シナリオの実行結果の構造的整合性、
記録の完全性、分析情報の生成を検証する。
"""

from __future__ import annotations

import pytest

from tools.long_term_sim import (
    INPUT_PATTERNS,
    RETURN_PATHWAY_IDS,
    SCENARIOS,
    _RETURN_INTERACTION_SCENARIOS,
    _compute_return_interaction_analysis,
    _read_return_interaction_tick,
    compute_statistics,
    run_simulation,
)
from tools.return_pathway_monitor import (
    PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E,
)


# ══════════════════════════════════════════════════════════════
# 第1段: シナリオ定義の検証
# ══════════════════════════════════════════════════════════════


class TestScenarioDefinitions:
    """3シナリオの定義が正しいことを検証する。"""

    def test_simultaneous_scenario_exists(self):
        """return_interaction_simultaneousシナリオが存在する。"""
        assert "return_interaction_simultaneous" in SCENARIOS

    def test_neutral_scenario_exists(self):
        """return_interaction_neutralシナリオが存在する。"""
        assert "return_interaction_neutral" in SCENARIOS

    def test_residual_scenario_exists(self):
        """return_interaction_residualシナリオが存在する。"""
        assert "return_interaction_residual" in SCENARIOS

    def test_simultaneous_scenario_length(self):
        """simultaneousシナリオのターン数が50。"""
        assert len(SCENARIOS["return_interaction_simultaneous"]) == 50

    def test_neutral_scenario_length(self):
        """neutralシナリオのターン数が50。"""
        assert len(SCENARIOS["return_interaction_neutral"]) == 50

    def test_residual_scenario_length(self):
        """residualシナリオのターン数が50。"""
        assert len(SCENARIOS["return_interaction_residual"]) == 50

    def test_simultaneous_uses_high_emotion_patterns(self):
        """simultaneousシナリオが高感情入力パターンを使用する。"""
        seq = SCENARIOS["return_interaction_simultaneous"]
        high_emotion_patterns = {"high_positive", "high_negative", "loving", "angry", "fearful"}
        used = set(seq)
        assert used.issubset(high_emotion_patterns), (
            f"Unexpected patterns: {used - high_emotion_patterns}"
        )

    def test_neutral_uses_only_neutral(self):
        """neutralシナリオが中立入力のみを使用する。"""
        seq = SCENARIOS["return_interaction_neutral"]
        assert all(p == "neutral" for p in seq)

    def test_residual_has_both_phases(self):
        """residualシナリオが高変動フェーズと中立フェーズを持つ。"""
        seq = SCENARIOS["return_interaction_residual"]
        has_high = any(p != "neutral" for p in seq)
        has_neutral = any(p == "neutral" for p in seq)
        assert has_high and has_neutral

    def test_all_pattern_keys_valid(self):
        """全シナリオのパターンキーが INPUT_PATTERNS に存在する。"""
        for scenario_name in _RETURN_INTERACTION_SCENARIOS:
            seq = SCENARIOS[scenario_name]
            for key in seq:
                assert key in INPUT_PATTERNS, (
                    f"{scenario_name}: invalid key {key}"
                )

    def test_interaction_scenario_set_complete(self):
        """_RETURN_INTERACTION_SCENARIOS が3シナリオ全てを含む。"""
        expected = {
            "return_interaction_simultaneous",
            "return_interaction_neutral",
            "return_interaction_residual",
        }
        assert _RETURN_INTERACTION_SCENARIOS == expected


class TestNewInputPatterns:
    """新規追加された入力パターンの検証。"""

    def test_high_positive_exists(self):
        """high_positiveパターンが存在する。"""
        assert "high_positive" in INPUT_PATTERNS

    def test_high_negative_exists(self):
        """high_negativeパターンが存在する。"""
        assert "high_negative" in INPUT_PATTERNS

    def test_high_positive_has_high_valence(self):
        """high_positiveが高valenceを持つ。"""
        assert INPUT_PATTERNS["high_positive"]["emotion_valence"] >= 0.8

    def test_high_negative_has_low_valence(self):
        """high_negativeが低valenceを持つ。"""
        assert INPUT_PATTERNS["high_negative"]["emotion_valence"] <= -0.8

    def test_high_positive_has_required_fields(self):
        """high_positiveに必須フィールドがある。"""
        required = {"text", "emotion", "emotion_valence", "intent"}
        assert required.issubset(INPUT_PATTERNS["high_positive"].keys())

    def test_high_negative_has_required_fields(self):
        """high_negativeに必須フィールドがある。"""
        required = {"text", "emotion", "emotion_valence", "intent"}
        assert required.issubset(INPUT_PATTERNS["high_negative"].keys())


class TestReturnPathwayIds:
    """RETURN_PATHWAY_IDSが全5経路を含むことの検証。"""

    def test_contains_all_5_pathways(self):
        """RETURN_PATHWAY_IDSが5経路全てを含む。"""
        assert len(RETURN_PATHWAY_IDS) == 5

    def test_contains_pathway_a(self):
        assert PATHWAY_A in RETURN_PATHWAY_IDS

    def test_contains_pathway_b(self):
        assert PATHWAY_B in RETURN_PATHWAY_IDS

    def test_contains_pathway_c(self):
        assert PATHWAY_C in RETURN_PATHWAY_IDS

    def test_contains_pathway_d(self):
        assert PATHWAY_D in RETURN_PATHWAY_IDS

    def test_contains_pathway_e(self):
        assert PATHWAY_E in RETURN_PATHWAY_IDS


# ══════════════════════════════════════════════════════════════
# 第2段: シナリオ1(全5本同時発火促進)のテスト
# ══════════════════════════════════════════════════════════════


class TestSimultaneousScenario:
    """シナリオ1: 高感情入力による帰還経路同時発火の記録検証。"""

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return run_simulation(scenario_name="return_interaction_simultaneous")

    def test_all_turns_have_return_interaction(self, result):
        """全ターンにreturn_interactionフィールドが存在する。"""
        for rec in result["turns"]:
            assert "return_interaction" in rec, (
                f"Turn {rec['turn']}: missing return_interaction"
            )

    def test_return_interaction_has_required_keys(self, result):
        """return_interactionに必須キーが存在する。"""
        required = {
            "fired_pathways", "fire_count",
            "combined_emotion_deltas", "combined_drive_deltas",
            "combined_mood_speed_deltas",
            "emotion_variation", "emotion_total_variation",
            "drive_variation", "drive_total_variation",
            "mood_variation", "mood_total_variation",
        }
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            missing = required - set(ri.keys())
            assert not missing, (
                f"Turn {rec['turn']}: missing keys {missing}"
            )

    def test_fired_pathways_is_list(self, result):
        """fired_pathwaysがリストである。"""
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            assert isinstance(ri["fired_pathways"], list)

    def test_fire_count_is_non_negative_int(self, result):
        """fire_countが0以上の整数である。"""
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            assert isinstance(ri["fire_count"], int)
            assert ri["fire_count"] >= 0

    def test_emotion_variation_is_dict(self, result):
        """emotion_variationが辞書である。"""
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            assert isinstance(ri["emotion_variation"], dict)

    def test_drive_variation_is_dict(self, result):
        """drive_variationが辞書である。"""
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            assert isinstance(ri["drive_variation"], dict)

    def test_mood_variation_is_dict(self, result):
        """mood_variationが辞書である。"""
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            assert isinstance(ri["mood_variation"], dict)
            assert "valence" in ri["mood_variation"]
            assert "arousal" in ri["mood_variation"]

    def test_total_turns_is_50(self, result):
        """ターン数が50である。"""
        assert result["metadata"]["total_turns"] == 50

    def test_analysis_present(self, result):
        """return_interaction_analysisが存在する。"""
        assert "return_interaction_analysis" in result

    def test_analysis_has_scenario(self, result):
        """分析にscenarioフィールドがある。"""
        analysis = result["return_interaction_analysis"]
        assert analysis["scenario"] == "return_interaction_simultaneous"

    def test_analysis_has_fire_counts(self, result):
        """分析に帰還経路別の発火カウントがある。"""
        analysis = result["return_interaction_analysis"]
        assert "pathway_fire_counts" in analysis
        assert "pathway_fire_ratios" in analysis

    def test_analysis_has_simultaneous_count(self, result):
        """分析に同時発火カウントがある。"""
        analysis = result["return_interaction_analysis"]
        assert "simultaneous_fire_count" in analysis
        assert "non_simultaneous_fire_count" in analysis
        total = (
            analysis["simultaneous_fire_count"]
            + analysis["non_simultaneous_fire_count"]
        )
        assert total == 50

    def test_existing_fields_preserved(self, result):
        """既存フィールドが維持されている。"""
        required = {
            "turn", "tick", "input_pattern", "input",
            "psyche_state", "responsibility", "responsibility_influence",
            "policy", "outcome_applied", "enrichment_chars",
            "return_pathway",
        }
        for rec in result["turns"]:
            missing = required - set(rec.keys())
            assert not missing, f"Turn {rec['turn']}: missing {missing}"


# ══════════════════════════════════════════════════════════════
# 第3段: シナリオ2(入力中立化による帰還経路効果の間接観測)のテスト
# ══════════════════════════════════════════════════════════════


class TestNeutralScenario:
    """シナリオ2: 中立入力による帰還経路のみの動態観測。"""

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return run_simulation(scenario_name="return_interaction_neutral")

    def test_all_turns_have_return_interaction(self, result):
        """全ターンにreturn_interactionフィールドが存在する。"""
        for rec in result["turns"]:
            assert "return_interaction" in rec

    def test_all_inputs_are_neutral(self, result):
        """全入力が中立パターンである。"""
        for rec in result["turns"]:
            assert rec["input_pattern"] == "neutral"

    def test_return_interaction_has_required_keys(self, result):
        """return_interactionに必須キーが存在する。"""
        required = {
            "fired_pathways", "fire_count",
            "emotion_variation", "emotion_total_variation",
            "drive_variation", "drive_total_variation",
            "mood_variation", "mood_total_variation",
        }
        for rec in result["turns"]:
            ri = rec["return_interaction"]
            missing = required - set(ri.keys())
            assert not missing

    def test_analysis_present(self, result):
        """return_interaction_analysisが存在する。"""
        assert "return_interaction_analysis" in result

    def test_total_turns_is_50(self, result):
        """ターン数が50である。"""
        assert result["metadata"]["total_turns"] == 50

    def test_analysis_scenario(self, result):
        """分析のシナリオ名が正しい。"""
        analysis = result["return_interaction_analysis"]
        assert analysis["scenario"] == "return_interaction_neutral"

    def test_existing_fields_preserved(self, result):
        """既存フィールドが維持されている。"""
        required = {"turn", "tick", "input_pattern", "psyche_state", "policy"}
        for rec in result["turns"]:
            missing = required - set(rec.keys())
            assert not missing


# ══════════════════════════════════════════════════════════════
# 第4段: シナリオ3(高変動入力後の沈静化観測)のテスト
# ══════════════════════════════════════════════════════════════


class TestResidualScenario:
    """シナリオ3: 帰還経路の残響効果の記録検証。"""

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return run_simulation(scenario_name="return_interaction_residual")

    def test_all_turns_have_return_interaction(self, result):
        """全ターンにreturn_interactionフィールドが存在する。"""
        for rec in result["turns"]:
            assert "return_interaction" in rec

    def test_has_both_phases(self, result):
        """高変動フェーズと中立フェーズが存在する。"""
        non_neutral = [r for r in result["turns"] if r["input_pattern"] != "neutral"]
        neutral = [r for r in result["turns"] if r["input_pattern"] == "neutral"]
        assert len(non_neutral) > 0
        assert len(neutral) > 0

    def test_analysis_present(self, result):
        """return_interaction_analysisが存在する。"""
        assert "return_interaction_analysis" in result

    def test_analysis_has_residual_info(self, result):
        """分析に残響情報がある。"""
        analysis = result["return_interaction_analysis"]
        assert "neutral_start_turn" in analysis
        assert "residual_ticks" in analysis
        assert "converged" in analysis

    def test_residual_ticks_is_non_negative(self, result):
        """残響ティック数が0以上。"""
        analysis = result["return_interaction_analysis"]
        assert analysis["residual_ticks"] >= 0

    def test_converged_is_bool(self, result):
        """convergedがブール値。"""
        analysis = result["return_interaction_analysis"]
        assert isinstance(analysis["converged"], bool)

    def test_residual_variation_series_present(self, result):
        """残響期間の変動量推移が記録されている。"""
        analysis = result["return_interaction_analysis"]
        assert "residual_variation_series" in analysis
        series = analysis["residual_variation_series"]
        assert isinstance(series, list)
        if series:
            entry = series[0]
            assert "turn" in entry
            assert "emotion_total" in entry
            assert "drive_total" in entry
            assert "mood_total" in entry

    def test_neutral_start_turn_is_positive(self, result):
        """neutral_start_turnが正の整数。"""
        analysis = result["return_interaction_analysis"]
        assert analysis["neutral_start_turn"] > 0

    def test_total_turns_is_50(self, result):
        """ターン数が50である。"""
        assert result["metadata"]["total_turns"] == 50

    def test_existing_fields_preserved(self, result):
        """既存フィールドが維持されている。"""
        required = {"turn", "tick", "input_pattern", "psyche_state", "policy"}
        for rec in result["turns"]:
            missing = required - set(rec.keys())
            assert not missing


# ══════════════════════════════════════════════════════════════
# 第5段: 非対象シナリオでは return_interaction が記録されないことの検証
# ══════════════════════════════════════════════════════════════


class TestNonInteractionScenarios:
    """非対象シナリオでreturn_interactionが生成されないことの検証。"""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        return run_simulation(scenario_name="smoke")

    def test_no_return_interaction_in_turns(self, smoke_result):
        """smokeシナリオのターンにreturn_interactionが存在しない。"""
        for rec in smoke_result["turns"]:
            assert "return_interaction" not in rec

    def test_no_return_interaction_analysis(self, smoke_result):
        """smokeシナリオにreturn_interaction_analysisが存在しない。"""
        assert "return_interaction_analysis" not in smoke_result


# ══════════════════════════════════════════════════════════════
# 第6段: 分析関数の単体テスト
# ══════════════════════════════════════════════════════════════


class TestComputeReturnInteractionAnalysis:
    """_compute_return_interaction_analysis の単体テスト。"""

    def _make_turn(
        self,
        turn: int,
        pattern: str,
        fire_count: int = 0,
        fired: list[str] | None = None,
        emo_var: float = 0.0,
        drv_var: float = 0.0,
        mood_var: float = 0.0,
    ) -> dict:
        return {
            "turn": turn,
            "input_pattern": pattern,
            "return_interaction": {
                "fired_pathways": fired or [],
                "fire_count": fire_count,
                "combined_emotion_deltas": {},
                "combined_drive_deltas": {},
                "combined_mood_speed_deltas": {},
                "emotion_variation": {},
                "emotion_total_variation": emo_var,
                "drive_variation": {},
                "drive_total_variation": drv_var,
                "mood_variation": {"valence": 0.0, "arousal": 0.0},
                "mood_total_variation": mood_var,
            },
        }

    def test_empty_turns(self):
        """空のターンリストで空の分析を返す。"""
        analysis = _compute_return_interaction_analysis([], "test")
        assert analysis["scenario"] == "test"

    def test_no_return_interaction_field(self):
        """return_interactionなしのターンで空の分析を返す。"""
        turns = [{"turn": 1, "input_pattern": "neutral"}]
        analysis = _compute_return_interaction_analysis(turns, "test")
        assert "simultaneous_fire_count" not in analysis

    def test_simultaneous_and_non_simultaneous_counts(self):
        """同時発火と非同時発火のカウントが正しい。"""
        turns = [
            self._make_turn(1, "positive", fire_count=3, fired=["a", "b", "c"],
                            emo_var=0.1, drv_var=0.05),
            self._make_turn(2, "neutral", fire_count=1, fired=["a"],
                            emo_var=0.01),
            self._make_turn(3, "positive", fire_count=2, fired=["a", "b"],
                            emo_var=0.08, drv_var=0.03),
        ]
        analysis = _compute_return_interaction_analysis(turns, "test")
        assert analysis["simultaneous_fire_count"] == 2
        assert analysis["non_simultaneous_fire_count"] == 1

    def test_simultaneous_variation_stats(self):
        """同時発火ティックの変動量統計が正しい。"""
        turns = [
            self._make_turn(1, "p", fire_count=2, fired=["a", "b"],
                            emo_var=0.1, drv_var=0.05, mood_var=0.02),
            self._make_turn(2, "p", fire_count=2, fired=["a", "c"],
                            emo_var=0.2, drv_var=0.1, mood_var=0.04),
        ]
        analysis = _compute_return_interaction_analysis(turns, "test")
        sim_var = analysis["simultaneous_variation"]
        assert sim_var["emotion_total"]["min"] == pytest.approx(0.1, abs=1e-5)
        assert sim_var["emotion_total"]["max"] == pytest.approx(0.2, abs=1e-5)

    def test_pathway_fire_counts_and_ratios(self):
        """帰還経路別の発火カウントと比率が正しい。"""
        turns = [
            self._make_turn(1, "p", fire_count=2, fired=["a", "b"]),
            self._make_turn(2, "p", fire_count=1, fired=["a"]),
            self._make_turn(3, "p", fire_count=1, fired=["b"]),
        ]
        analysis = _compute_return_interaction_analysis(turns, "test")
        assert analysis["pathway_fire_counts"]["a"] == 2
        assert analysis["pathway_fire_counts"]["b"] == 2
        assert analysis["pathway_fire_ratios"]["a"] == pytest.approx(2 / 3, abs=1e-5)

    def test_residual_analysis_for_residual_scenario(self):
        """残響分析がresidualシナリオで生成される。"""
        turns = [
            self._make_turn(1, "high_positive", fire_count=2, fired=["a", "b"],
                            emo_var=0.3, drv_var=0.1, mood_var=0.05),
            self._make_turn(2, "neutral", fire_count=0,
                            emo_var=0.005, drv_var=0.002, mood_var=0.001),
        ]
        analysis = _compute_return_interaction_analysis(
            turns, "return_interaction_residual",
        )
        assert "neutral_start_turn" in analysis
        assert analysis["neutral_start_turn"] == 2

    def test_residual_converged(self):
        """残響が収束する場合converged=True。"""
        turns = [
            self._make_turn(1, "high_positive", emo_var=0.3, drv_var=0.1, mood_var=0.05),
            self._make_turn(2, "neutral", emo_var=0.005, drv_var=0.002, mood_var=0.001),
        ]
        analysis = _compute_return_interaction_analysis(
            turns, "return_interaction_residual",
        )
        assert analysis["converged"] is True

    def test_residual_not_converged(self):
        """残響が収束しない場合converged=False。"""
        turns = [
            self._make_turn(1, "high_positive", emo_var=0.3),
            self._make_turn(2, "neutral", emo_var=0.05, drv_var=0.02, mood_var=0.01),
            self._make_turn(3, "neutral", emo_var=0.04, drv_var=0.02, mood_var=0.01),
        ]
        analysis = _compute_return_interaction_analysis(
            turns, "return_interaction_residual",
        )
        assert analysis["converged"] is False
        assert analysis["residual_ticks"] == 2

    def test_non_residual_scenario_no_residual_fields(self):
        """非residualシナリオでは残響分析フィールドが存在しない。"""
        turns = [
            self._make_turn(1, "positive", fire_count=1, fired=["a"]),
        ]
        analysis = _compute_return_interaction_analysis(
            turns, "return_interaction_simultaneous",
        )
        assert "neutral_start_turn" not in analysis
        assert "residual_ticks" not in analysis


# ══════════════════════════════════════════════════════════════
# 第7段: 統計サマリーとの互換性テスト
# ══════════════════════════════════════════════════════════════


class TestStatisticsCompatibility:
    """既存のcompute_statisticsが新シナリオで正常に動作することの検証。"""

    @pytest.fixture(scope="class")
    def simultaneous_result(self) -> dict:
        return run_simulation(scenario_name="return_interaction_simultaneous")

    def test_compute_statistics_works(self, simultaneous_result):
        """compute_statisticsがエラーなく実行される。"""
        stats = compute_statistics(simultaneous_result)
        assert "scenario" in stats
        assert stats["scenario"] == "return_interaction_simultaneous"

    def test_statistics_has_emotions(self, simultaneous_result):
        """統計にemotionsが含まれる。"""
        stats = compute_statistics(simultaneous_result)
        assert "emotions" in stats

    def test_statistics_has_mood(self, simultaneous_result):
        """統計にmoodが含まれる。"""
        stats = compute_statistics(simultaneous_result)
        assert "mood" in stats

    def test_statistics_has_drives(self, simultaneous_result):
        """統計にdrivesが含まれる。"""
        stats = compute_statistics(simultaneous_result)
        assert "drives" in stats

    def test_statistics_has_return_pathway(self, simultaneous_result):
        """統計にreturn_pathwayが含まれる。"""
        stats = compute_statistics(simultaneous_result)
        assert "return_pathway" in stats


# ══════════════════════════════════════════════════════════════
# 第8段: psycheコード非変更の安全弁テスト
# ══════════════════════════════════════════════════════════════


class TestSafetyValves:
    """設計書の安全弁が守られていることの検証。"""

    def test_sim_uses_temp_directory(self):
        """simが一時ディレクトリで実行され、永続化に影響しない。"""
        # smokeシナリオで2回実行し、結果が独立していることを確認
        result1 = run_simulation(scenario_name="smoke")
        result2 = run_simulation(scenario_name="smoke")
        # 両方とも正常に完了する(独立したorchestratorインスタンス)
        assert result1["metadata"]["total_turns"] == result2["metadata"]["total_turns"]

    def test_interaction_scenarios_independent(self):
        """3シナリオが独立して実行可能。"""
        for scenario in _RETURN_INTERACTION_SCENARIOS:
            result = run_simulation(scenario_name=scenario)
            assert result["metadata"]["scenario"] == scenario
            assert len(result["turns"]) == 50

    def test_no_return_interaction_analysis_for_non_target(self):
        """非対象シナリオでreturn_interaction_analysisが生成されない。"""
        result = run_simulation(scenario_name="smoke")
        assert "return_interaction_analysis" not in result
