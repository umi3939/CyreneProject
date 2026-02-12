"""
tests/test_long_term_sim.py - 長期シミュレータのテスト (~14件)
"""

from __future__ import annotations

import pytest

from psyche.state import Percept
from tools.long_term_sim import (
    INPUT_PATTERNS,
    SCENARIOS,
    _pattern_to_outcome,
    run_simulation,
)


# ── Pattern / Scenario validation ─────────────────────────────

class TestPatternDefinitions:
    """入力パターン定義の検証"""

    def test_all_patterns_create_valid_percept(self):
        """全パターンがPercept生成可能"""
        for key, pat in INPUT_PATTERNS.items():
            p = Percept(**pat)
            assert p.text, f"Pattern '{key}' has empty text"
            assert p.emotion, f"Pattern '{key}' has empty emotion"
            assert -1.0 <= p.emotion_valence <= 1.0, (
                f"Pattern '{key}' valence out of range"
            )

    def test_all_scenarios_use_valid_patterns(self):
        """全シナリオのキーがINPUT_PATTERNSに存在"""
        for name, seq in SCENARIOS.items():
            for key in seq:
                assert key in INPUT_PATTERNS, (
                    f"Scenario '{name}' uses unknown pattern '{key}'"
                )


class TestOutcomeMapping:
    """outcome変換の検証"""

    def test_pattern_to_outcome_all_keys(self):
        """全パターンキーでoutcome変換が動作"""
        for key in INPUT_PATTERNS:
            outcome = _pattern_to_outcome(key)
            assert "user_reaction" in outcome
            assert "relationship_delta" in outcome
            assert "expectation_gap" in outcome

    def test_pattern_to_outcome_unknown_key(self):
        """未知キーでデフォルト値が返される"""
        outcome = _pattern_to_outcome("nonexistent_key")
        assert outcome["user_reaction"] == "neutral"
        assert outcome["relationship_delta"] == 0.0
        assert outcome["expectation_gap"] == 0.0


# ── Simulation execution ──────────────────────────────────────

class TestSimulationExecution:
    """シミュレーション実行テスト"""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        """smoke シナリオの結果（クラス内で共有）"""
        return run_simulation(scenario_name="smoke")

    def test_smoke_scenario_runs(self, smoke_result):
        """smoke 5ターン実行 → JSON出力"""
        assert "metadata" in smoke_result
        assert "turns" in smoke_result
        assert smoke_result["metadata"]["scenario"] == "smoke"

    def test_turn_count_matches(self, smoke_result):
        """turns配列の長さ = シナリオ長"""
        expected = len(SCENARIOS["smoke"])
        assert len(smoke_result["turns"]) == expected
        assert smoke_result["metadata"]["total_turns"] == expected

    def test_turn_records_have_all_fields(self, smoke_result):
        """全レコードに必須キーあり"""
        required = {
            "turn", "tick", "input_pattern", "input",
            "psyche_state", "responsibility", "responsibility_influence",
            "policy", "outcome_applied",
        }
        for rec in smoke_result["turns"]:
            missing = required - set(rec.keys())
            assert not missing, f"Turn {rec.get('turn')}: missing {missing}"

    def test_psyche_values_in_range(self, smoke_result):
        """emotions 0-1, mood.valence -1..1"""
        for rec in smoke_result["turns"]:
            ps = rec["psyche_state"]
            for emo_name, val in ps["emotions"].items():
                assert 0.0 <= val <= 1.0, (
                    f"Turn {rec['turn']}: emotion {emo_name}={val} out of range"
                )
            assert -1.0 <= ps["mood"]["valence"] <= 1.0
            assert 0.0 <= ps["mood"]["arousal"] <= 1.0

    def test_responsibility_values_in_range(self, smoke_result):
        """total_weight 0-1"""
        for rec in smoke_result["turns"]:
            tw = rec["responsibility"]["total_weight"]
            assert 0.0 <= tw <= 1.0, (
                f"Turn {rec['turn']}: total_weight={tw} out of range"
            )

    def test_policy_always_selected(self, smoke_result):
        """全ターンで policy_label 非空"""
        for rec in smoke_result["turns"]:
            assert rec["policy"]["policy_label"], (
                f"Turn {rec['turn']}: empty policy_label"
            )

    def test_tick_increments(self, smoke_result):
        """tick値が単調増加"""
        ticks = [rec["tick"] for rec in smoke_result["turns"]]
        for i in range(1, len(ticks)):
            assert ticks[i] > ticks[i - 1], (
                f"Tick did not increase at turn {i + 1}: {ticks[i]} <= {ticks[i - 1]}"
            )


class TestScenarioBehaviour:
    """シナリオ別の挙動テスト"""

    def test_repeated_failure_increases_harm(self):
        """repeated_failure後 harm > 0"""
        result = run_simulation(scenario_name="repeated_failure")
        last = result["turns"][-1]
        assert last["responsibility"]["accumulated_harm"] > 0.0

    def test_custom_sequence(self):
        """カスタムリスト実行"""
        seq = ["positive", "negative", "neutral"]
        result = run_simulation(custom_sequence=seq)
        assert len(result["turns"]) == 3
        assert result["metadata"]["scenario"] == "custom"
        patterns = [t["input_pattern"] for t in result["turns"]]
        assert patterns == seq


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_invalid_pattern_raises(self):
        """不正パターンで ValueError"""
        with pytest.raises(ValueError, match="Invalid pattern key"):
            run_simulation(custom_sequence=["nonexistent_pattern"])

    def test_invalid_scenario_raises(self):
        """不正シナリオ名で ValueError"""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_simulation(scenario_name="nonexistent_scenario")
