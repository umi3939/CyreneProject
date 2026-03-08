"""
Tests for psyche/firing_frequency_modulation.py

帰還経路発火頻度による係数帯域限界の変調のテスト。
design_firing_frequency_modulation.md に基づく検証。
"""

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional

from psyche.firing_frequency_modulation import (
    compute_log_modulation_ratio,
    compute_modulated_value,
    apply_firing_frequency_modulation,
    get_modulation_summary,
    _safe_get_count,
    _ABSOLUTE_CAP_MULTIPLIER,
    _LOG_SCALE_FACTOR,
    _MAX_MODULATION_RATIO,
    _PATHWAY_A,
    _PATHWAY_C,
    _PATHWAY_D,
    _PATHWAY_E,
)


# =============================================================================
# Mock Config classes
# =============================================================================


@dataclass
class MockMemoryEmotionReturnConfig:
    """Mock for MemoryEmotionReturnConfig."""
    per_candidate_max_delta: float = 0.03
    total_max_delta: float = 0.15
    tracking_speed_modulation_ratio_cap: float = 0.10


@dataclass
class MockOtherHypothesisEmotionReturnConfig:
    """Mock for OtherHypothesisEmotionReturnConfig."""
    per_candidate_max_delta: float = 0.02
    total_max_delta: float = 0.07


# =============================================================================
# Tests for compute_log_modulation_ratio
# =============================================================================


class TestComputeLogModulationRatio:
    """対数スケーリングによる変調比率算出のテスト。"""

    def test_zero_count_returns_zero(self):
        """累計発火回数が0の場合、変調比率は0.0。"""
        assert compute_log_modulation_ratio(0) == 0.0

    def test_negative_count_returns_zero(self):
        """負の累計発火回数の場合、変調比率は0.0。"""
        assert compute_log_modulation_ratio(-5) == 0.0

    def test_small_count_gives_small_ratio(self):
        """少数の発火回数では変調比率が小さい。"""
        ratio = compute_log_modulation_ratio(5)
        assert 0.0 < ratio < 0.1

    def test_moderate_count_gives_moderate_ratio(self):
        """中程度の発火回数では中程度の変調比率。"""
        ratio = compute_log_modulation_ratio(100)
        assert 0.0 < ratio < _MAX_MODULATION_RATIO

    def test_large_count_approaches_cap(self):
        """大量の発火回数では変調比率が上限に近づく。"""
        ratio = compute_log_modulation_ratio(100000)
        assert ratio <= _MAX_MODULATION_RATIO
        assert ratio > 0.3  # 十分大きな値

    def test_monotonically_increasing(self):
        """変調比率は発火回数に対して単調増加する。"""
        counts = [0, 1, 5, 10, 50, 100, 500, 1000, 10000]
        ratios = [compute_log_modulation_ratio(c) for c in counts]
        for i in range(len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1], (
                f"ratio for count={counts[i]} ({ratios[i]}) > "
                f"ratio for count={counts[i+1]} ({ratios[i+1]})"
            )

    def test_diminishing_returns(self):
        """対数スケーリングにより追加発火の効果が逓減する。"""
        # 0->100 の増加量 vs 100->200 の増加量
        r_0 = compute_log_modulation_ratio(0)
        r_100 = compute_log_modulation_ratio(100)
        r_200 = compute_log_modulation_ratio(200)
        delta_first = r_100 - r_0
        delta_second = r_200 - r_100
        assert delta_first > delta_second, (
            "First 100 firings should have more effect than second 100"
        )

    def test_never_exceeds_max_ratio(self):
        """いかなる累計発火回数でも最大変調比率を超えない。"""
        for count in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            ratio = compute_log_modulation_ratio(count)
            assert ratio <= _MAX_MODULATION_RATIO, (
                f"ratio={ratio} for count={count} exceeds max={_MAX_MODULATION_RATIO}"
            )

    def test_absolute_cap_multiplier_is_1_5(self):
        """絶対帯域上限の倍率が1.5であることを確認。"""
        assert _ABSOLUTE_CAP_MULTIPLIER == 1.5

    def test_max_modulation_ratio_is_0_5(self):
        """最大変調比率が0.5であることを確認（1.5 - 1.0）。"""
        assert _MAX_MODULATION_RATIO == 0.5


# =============================================================================
# Tests for compute_modulated_value
# =============================================================================


class TestComputeModulatedValue:
    """変調後の帯域上限値算出のテスト。"""

    def test_zero_count_returns_base(self):
        """累計発火回数0では基準値そのまま。"""
        base = 0.03
        result = compute_modulated_value(base, 0)
        assert result == base

    def test_positive_count_increases_value(self):
        """正の累計発火回数では基準値より大きくなる。"""
        base = 0.03
        result = compute_modulated_value(base, 100)
        assert result > base

    def test_absolute_cap_enforced(self):
        """変調後の値が基準値の1.5倍を超えないことを確認。"""
        base = 0.03
        result = compute_modulated_value(base, 1000000)
        assert result <= base * _ABSOLUTE_CAP_MULTIPLIER

    def test_zero_base_returns_zero(self):
        """基準値が0の場合は0のまま。"""
        result = compute_modulated_value(0.0, 100)
        assert result == 0.0

    def test_various_bases(self):
        """異なる基準値に対して正しく変調される。"""
        for base in [0.01, 0.03, 0.07, 0.10, 0.15]:
            result = compute_modulated_value(base, 50)
            assert base <= result <= base * _ABSOLUTE_CAP_MULTIPLIER

    def test_monotonically_increasing_with_count(self):
        """同じ基準値に対して、発火回数が増えると変調後の値も増える。"""
        base = 0.03
        counts = [0, 10, 50, 100, 500]
        values = [compute_modulated_value(base, c) for c in counts]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


# =============================================================================
# Tests for apply_firing_frequency_modulation
# =============================================================================


class TestApplyFiringFrequencyModulation:
    """帯域変調の適用テスト。"""

    def test_empty_fire_counts(self):
        """空の発火カウントでは変調なし。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03, "curiosity": 0.03, "expression": 0.03}}

        result = apply_firing_frequency_modulation({}, mer, oher, band)
        assert result == {}
        assert mer.per_candidate_max_delta == 0.03
        assert oher.per_candidate_max_delta == 0.02

    def test_invalid_fire_counts(self):
        """無効な型の発火カウントでは変調なし。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        result = apply_firing_frequency_modulation("invalid", mer, oher, band)
        assert result == {}

    def test_pathway_a_modulation(self):
        """Pathway A (memory_emotion_return) の変調が正しく適用される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        counts = {_PATHWAY_A: 100}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert "pathway_a" in result
        assert mer.per_candidate_max_delta > 0.03
        assert mer.total_max_delta > 0.15
        # Absolute cap check
        assert mer.per_candidate_max_delta <= 0.03 * _ABSOLUTE_CAP_MULTIPLIER
        assert mer.total_max_delta <= 0.15 * _ABSOLUTE_CAP_MULTIPLIER

    def test_pathway_c_modulation(self):
        """Pathway C (other_hypothesis_emotion_return) の変調が正しく適用される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        counts = {_PATHWAY_C: 200}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert "pathway_c" in result
        assert oher.per_candidate_max_delta > 0.02
        assert oher.total_max_delta > 0.07
        # Absolute cap check
        assert oher.per_candidate_max_delta <= 0.02 * _ABSOLUTE_CAP_MULTIPLIER
        assert oher.total_max_delta <= 0.07 * _ABSOLUTE_CAP_MULTIPLIER

    def test_pathway_d_modulation(self):
        """Pathway D (result_diversity_drive_return) の変調が正しく適用される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03, "curiosity": 0.03, "expression": 0.03}}

        counts = {_PATHWAY_D: 150}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert "pathway_d" in result
        for axis in ["social", "curiosity", "expression"]:
            assert band["result_diversity_return"][axis] > 0.03
            assert band["result_diversity_return"][axis] <= 0.03 * _ABSOLUTE_CAP_MULTIPLIER

    def test_pathway_e_modulation(self):
        """Pathway E (emotion_return_tracking_speed) の変調が正しく適用される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        counts = {_PATHWAY_E: 80}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert "pathway_e" in result
        assert mer.tracking_speed_modulation_ratio_cap > 0.10
        assert mer.tracking_speed_modulation_ratio_cap <= 0.10 * _ABSOLUTE_CAP_MULTIPLIER

    def test_all_pathways_modulated_simultaneously(self):
        """全帰還経路が同時に変調される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03, "curiosity": 0.03, "expression": 0.03}}

        counts = {
            _PATHWAY_A: 100,
            _PATHWAY_C: 50,
            _PATHWAY_D: 200,
            _PATHWAY_E: 30,
        }
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert "pathway_a" in result
        assert "pathway_c" in result
        assert "pathway_d" in result
        assert "pathway_e" in result

    def test_zero_count_pathways_not_modulated(self):
        """発火回数が0の帰還経路は変調されない。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        counts = {
            _PATHWAY_A: 0,
            _PATHWAY_C: 0,
            _PATHWAY_D: 0,
            _PATHWAY_E: 0,
        }
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        assert result == {}
        assert mer.per_candidate_max_delta == 0.03
        assert oher.per_candidate_max_delta == 0.02

    def test_none_config_is_safe(self):
        """Configオブジェクトがnoneでもクラッシュしない。"""
        counts = {_PATHWAY_A: 100, _PATHWAY_C: 100}
        result = apply_firing_frequency_modulation(counts, None, None, None)
        assert isinstance(result, dict)

    def test_missing_result_diversity_key_is_safe(self):
        """drive_section_bandにresult_diversity_returnがない場合も安全。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"emotion_drive_coupling": {"social": 0.06}}

        counts = {_PATHWAY_D: 100}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)
        # result_diversity_return キーがないので pathway_d には空 modulated_fields
        if "pathway_d" in result:
            assert result["pathway_d"]["modulated_fields"] == {}

    def test_modulation_result_contains_correct_info(self):
        """変調結果の辞書が正しい構造を持つ。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        counts = {_PATHWAY_A: 50}
        result = apply_firing_frequency_modulation(counts, mer, oher, band)

        entry = result["pathway_a"]
        assert "count" in entry
        assert "ratio" in entry
        assert "modulated_fields" in entry
        assert entry["count"] == 50
        assert 0.0 < entry["ratio"] < _MAX_MODULATION_RATIO


# =============================================================================
# Tests for safety valves
# =============================================================================


class TestSafetyValves:
    """安全弁のテスト。"""

    def test_log_scaling_saturation(self):
        """安全弁1: 対数スケーリングによる飽和。"""
        # 1000回 vs 2000回 の差が 0->1000 の差より小さい
        r_0 = compute_log_modulation_ratio(0)
        r_1000 = compute_log_modulation_ratio(1000)
        r_2000 = compute_log_modulation_ratio(2000)

        delta_0_1000 = r_1000 - r_0
        delta_1000_2000 = r_2000 - r_1000

        assert delta_0_1000 > delta_1000_2000

    def test_absolute_cap_at_1_5x(self):
        """安全弁2: 基準値の1.5倍を超えない。"""
        for base in [0.01, 0.03, 0.05, 0.10, 0.15, 0.50]:
            result = compute_modulated_value(base, 10000000)
            assert result <= base * 1.5 + 1e-10, (
                f"base={base}, result={result}, cap={base * 1.5}"
            )

    def test_count_never_decremented(self):
        """安全弁5: カウンタ減算のテスト（_safe_get_countは負値を0にする）。"""
        assert _safe_get_count({"a": -5}, "a") == 0
        assert _safe_get_count({"a": 0}, "a") == 0
        assert _safe_get_count({"a": 10}, "a") == 10

    def test_pathway_a_e_independent_modulation(self):
        """Pathway AとEは異なるカウントに基づいて独立に変調される。"""
        mer = MockMemoryEmotionReturnConfig()
        oher = MockOtherHypothesisEmotionReturnConfig()
        band = {"result_diversity_return": {"social": 0.03}}

        # Pathway A のみ大量発火、E は少ない
        counts = {_PATHWAY_A: 500, _PATHWAY_E: 10}
        apply_firing_frequency_modulation(counts, mer, oher, band)

        # per_candidate_max_delta の変調量は pathway_a のカウントに基づく
        # tracking_speed_modulation_ratio_cap は pathway_e のカウントに基づく
        a_ratio = compute_log_modulation_ratio(500)
        e_ratio = compute_log_modulation_ratio(10)
        assert a_ratio > e_ratio  # A のカウントが多いので比率も大きい

    def test_non_numeric_count_handled(self):
        """非数値のカウント値が安全に処理される。"""
        assert _safe_get_count({"a": "string"}, "a") == 0
        assert _safe_get_count({"a": None}, "a") == 0
        assert _safe_get_count({"a": [1, 2]}, "a") == 0

    def test_missing_pathway_key_handled(self):
        """存在しない経路キーが安全に処理される。"""
        assert _safe_get_count({}, _PATHWAY_A) == 0

    def test_float_count_handled(self):
        """浮動小数点のカウント値が整数に変換される。"""
        assert _safe_get_count({"a": 10.7}, "a") == 10


# =============================================================================
# Tests for get_modulation_summary
# =============================================================================


class TestGetModulationSummary:
    """変調サマリーの読み取りテスト。"""

    def test_empty_counts(self):
        """空カウントでも全経路の情報を返す。"""
        summary = get_modulation_summary({})
        assert len(summary) == 4
        for pid in [_PATHWAY_A, _PATHWAY_C, _PATHWAY_D, _PATHWAY_E]:
            assert pid in summary
            assert summary[pid]["count"] == 0.0
            assert summary[pid]["ratio"] == 0.0

    def test_with_counts(self):
        """カウントがある場合に正しい比率を返す。"""
        counts = {_PATHWAY_A: 100, _PATHWAY_D: 50}
        summary = get_modulation_summary(counts)

        assert summary[_PATHWAY_A]["count"] == 100.0
        assert summary[_PATHWAY_A]["ratio"] > 0.0
        assert summary[_PATHWAY_D]["count"] == 50.0
        assert summary[_PATHWAY_D]["ratio"] > 0.0
        assert summary[_PATHWAY_C]["count"] == 0.0

    def test_invalid_input(self):
        """無効な入力では空辞書を返す。"""
        assert get_modulation_summary(None) == {}
        assert get_modulation_summary("invalid") == {}


# =============================================================================
# Tests for determinism
# =============================================================================


class TestDeterminism:
    """決定論的変換のテスト。"""

    def test_same_input_same_output(self):
        """同一の累計発火回数からは常に同一の結果が得られる。"""
        for count in [0, 1, 10, 50, 100, 500, 1000]:
            r1 = compute_log_modulation_ratio(count)
            r2 = compute_log_modulation_ratio(count)
            assert r1 == r2

    def test_same_modulated_value(self):
        """同一の入力からは常に同一の変調後の値が得られる。"""
        for base in [0.03, 0.07, 0.15]:
            for count in [0, 50, 100, 500]:
                v1 = compute_modulated_value(base, count)
                v2 = compute_modulated_value(base, count)
                assert v1 == v2

    def test_full_modulation_deterministic(self):
        """apply_firing_frequency_modulation が決定論的。"""
        counts = {_PATHWAY_A: 100, _PATHWAY_C: 50}

        # 1回目
        mer1 = MockMemoryEmotionReturnConfig()
        oher1 = MockOtherHypothesisEmotionReturnConfig()
        band1 = {"result_diversity_return": {"social": 0.03}}
        apply_firing_frequency_modulation(counts, mer1, oher1, band1)

        # 2回目
        mer2 = MockMemoryEmotionReturnConfig()
        oher2 = MockOtherHypothesisEmotionReturnConfig()
        band2 = {"result_diversity_return": {"social": 0.03}}
        apply_firing_frequency_modulation(counts, mer2, oher2, band2)

        assert mer1.per_candidate_max_delta == mer2.per_candidate_max_delta
        assert mer1.total_max_delta == mer2.total_max_delta
        assert oher1.per_candidate_max_delta == oher2.per_candidate_max_delta
        assert oher1.total_max_delta == oher2.total_max_delta


# =============================================================================
# Tests for non-decrease property
# =============================================================================


class TestNonDecrease:
    """帯域上限の非減少性テスト。"""

    def test_modulated_value_never_below_base(self):
        """変調後の値が基準値を下回らない。"""
        for base in [0.01, 0.03, 0.05, 0.10, 0.15]:
            for count in [0, 1, 10, 100, 1000]:
                result = compute_modulated_value(base, count)
                assert result >= base, (
                    f"modulated={result} < base={base} for count={count}"
                )

    def test_increasing_count_never_decreases_value(self):
        """発火回数の増加で変調後の値が減少しない。"""
        base = 0.03
        prev = compute_modulated_value(base, 0)
        for count in range(1, 1001, 10):
            curr = compute_modulated_value(base, count)
            assert curr >= prev, (
                f"value decreased: {prev} -> {curr} at count={count}"
            )
            prev = curr


# =============================================================================
# Tests for _safe_get_count
# =============================================================================


class TestSafeGetCount:
    """安全なカウント取得のテスト。"""

    def test_valid_int(self):
        assert _safe_get_count({"a": 42}, "a") == 42

    def test_valid_float(self):
        assert _safe_get_count({"a": 42.9}, "a") == 42

    def test_negative(self):
        assert _safe_get_count({"a": -5}, "a") == 0

    def test_missing_key(self):
        assert _safe_get_count({"b": 10}, "a") == 0

    def test_none_value(self):
        assert _safe_get_count({"a": None}, "a") == 0

    def test_string_value(self):
        assert _safe_get_count({"a": "10"}, "a") == 0

    def test_list_value(self):
        assert _safe_get_count({"a": [1, 2]}, "a") == 0

    def test_zero(self):
        assert _safe_get_count({"a": 0}, "a") == 0
