"""
tests/test_window_size_modulation.py - ウィンドウサイズ経験依存化のテスト

design_window_size_modulation.md の実装に対するテスト。
"""

import pytest
from unittest.mock import patch

from psyche import coefficient_registry
from psyche.forgetting_recall_balance import (
    ForgettingRecallBalanceState,
    JuxtapositionEntry,
    ForgettingSectionSnapshot,
    ExternalRecallSectionSnapshot,
    SpontaneousRecallSectionSnapshot,
)
from psyche.window_size_modulation import (
    extract_recall_frequency,
    classify_recall_frequency,
    compute_modulation,
    apply_window_size_modulation,
    save_modulated_values,
    load_modulated_values,
    _apply_to_registry,
    _MODULATION_BAND_RATIO,
    _REGRESSION_PRESSURE,
    _RECALL_FREQUENCY_STAGES,
    _RECALL_FREQUENCY_VERY_ACTIVE_FACTOR,
    _WINDOW_SIZE_KEYS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_registry():
    """各テスト前後にcoefficient_registryをリセットする。"""
    coefficient_registry.reset()
    coefficient_registry.load()
    yield
    coefficient_registry.reset()


def _make_frb_state(entries: list[tuple[int, int]]) -> ForgettingRecallBalanceState:
    """テスト用のForgettingRecallBalanceStateを作成する。

    Args:
        entries: (external_recall_count, spontaneous_recall_count) のリスト

    Returns:
        ForgettingRecallBalanceState
    """
    history = []
    for ext_count, spont_count in entries:
        entry = JuxtapositionEntry(
            timestamp=1000.0,
            forgetting=ForgettingSectionSnapshot(),
            external_recall=ExternalRecallSectionSnapshot(total_count=ext_count),
            spontaneous_recall=SpontaneousRecallSectionSnapshot(total_count=spont_count),
        )
        history.append(entry)
    return ForgettingRecallBalanceState(history=history, cycle_count=len(history))


# =============================================================================
# extract_recall_frequency tests
# =============================================================================

class TestExtractRecallFrequency:
    """想起頻度の読み取りテスト。"""

    def test_empty_history(self):
        """空の履歴では0を返す。"""
        state = ForgettingRecallBalanceState()
        assert extract_recall_frequency(state) == 0

    def test_single_entry(self):
        """単一エントリの件数合算。"""
        state = _make_frb_state([(3, 2)])
        assert extract_recall_frequency(state) == 5

    def test_multiple_entries(self):
        """複数エントリの件数合算。"""
        state = _make_frb_state([(3, 2), (5, 1), (0, 4)])
        assert extract_recall_frequency(state) == 15

    def test_all_zero(self):
        """全てゼロのエントリ。"""
        state = _make_frb_state([(0, 0), (0, 0)])
        assert extract_recall_frequency(state) == 0

    def test_read_only(self):
        """蓄積データへの書き込みがないことの確認。"""
        state = _make_frb_state([(3, 2)])
        original_count = state.history[0].external_recall.total_count
        extract_recall_frequency(state)
        assert state.history[0].external_recall.total_count == original_count


# =============================================================================
# classify_recall_frequency tests
# =============================================================================

class TestClassifyRecallFrequency:
    """想起頻度の段階値変換テスト。"""

    def test_zero_recall(self):
        """想起なし: 縮小方向。"""
        factor = classify_recall_frequency(0)
        assert factor == _RECALL_FREQUENCY_STAGES[0][1]
        assert factor < 0

    def test_low_recall(self):
        """不活発: 縮小方向。"""
        factor = classify_recall_frequency(2)
        assert factor == _RECALL_FREQUENCY_STAGES[1][1]
        assert factor < 0

    def test_normal_recall(self):
        """普通: 変調なし。"""
        factor = classify_recall_frequency(8)
        assert factor == _RECALL_FREQUENCY_STAGES[2][1]
        assert factor == 0.0

    def test_active_recall(self):
        """活発: 拡大方向。"""
        factor = classify_recall_frequency(15)
        assert factor == _RECALL_FREQUENCY_STAGES[3][1]
        assert factor > 0

    def test_very_active_recall(self):
        """非常に活発: 拡大方向（最大）。"""
        factor = classify_recall_frequency(100)
        assert factor == _RECALL_FREQUENCY_VERY_ACTIVE_FACTOR
        assert factor > 0

    def test_boundary_values(self):
        """境界値のテスト。"""
        # 各段階の上限値で分類
        for upper, expected_factor in _RECALL_FREQUENCY_STAGES:
            assert classify_recall_frequency(upper) == expected_factor


# =============================================================================
# compute_modulation tests
# =============================================================================

class TestComputeModulation:
    """変調値算出テスト。"""

    def test_no_modulation(self):
        """変調方向係数0.0の場合、基準値と一致。"""
        result = compute_modulation(base_value=30, direction_factor=0.0)
        assert result == 30

    def test_positive_modulation(self):
        """拡大方向の変調。"""
        result = compute_modulation(base_value=30, direction_factor=0.8)
        assert result > 30

    def test_negative_modulation(self):
        """縮小方向の変調。"""
        result = compute_modulation(base_value=30, direction_factor=-0.8)
        assert result < 30

    def test_band_limit_upper(self):
        """帯域上限の遵守（拡大方向）。"""
        result = compute_modulation(base_value=30, direction_factor=1.0)
        max_allowed = round(30 * (1.0 + _MODULATION_BAND_RATIO))
        assert result <= max_allowed

    def test_band_limit_lower(self):
        """帯域上限の遵守（縮小方向）。"""
        result = compute_modulation(base_value=30, direction_factor=-1.0)
        min_allowed = round(30 * (1.0 - _MODULATION_BAND_RATIO))
        assert result >= min_allowed

    def test_integer_result(self):
        """結果が整数であること。"""
        result = compute_modulation(base_value=25, direction_factor=0.3)
        assert isinstance(result, int)

    def test_minimum_value_one(self):
        """結果が最小1であること。"""
        result = compute_modulation(base_value=1, direction_factor=-1.0)
        assert result >= 1

    def test_regression_pressure_reduces_deviation(self):
        """回帰圧力が前回偏差を縮小する方向に作用すること。"""
        # 前回: 基準値30に対して36（正方向偏差+6）
        # 新規変調: 方向係数0（変調なし）
        # 回帰圧力により基準値方向に引き戻される
        result = compute_modulation(
            base_value=30,
            direction_factor=0.0,
            previous_modulated=36,
        )
        # 回帰圧力により30未満にはならない（正方向偏差の引き戻しなので30に近づく）
        assert result < 36
        assert result >= round(30 * (1.0 - _MODULATION_BAND_RATIO))

    def test_regression_pressure_with_same_direction(self):
        """回帰圧力と新規変調が同じ方向の場合も帯域内。"""
        result = compute_modulation(
            base_value=30,
            direction_factor=0.8,
            previous_modulated=36,
        )
        max_allowed = round(30 * (1.0 + _MODULATION_BAND_RATIO))
        assert result <= max_allowed

    def test_no_previous_no_regression(self):
        """前回変調値なしの場合、回帰圧力なし。"""
        result_with_none = compute_modulation(
            base_value=30,
            direction_factor=0.5,
            previous_modulated=None,
        )
        # 回帰圧力なし、純粋な方向係数*帯域
        expected_raw = 30 + 0.5 * 30 * _MODULATION_BAND_RATIO
        assert result_with_none == round(expected_raw)


# =============================================================================
# apply_window_size_modulation tests
# =============================================================================

class TestApplyWindowSizeModulation:
    """統合テスト: 変調の適用。"""

    def test_empty_frb_state(self):
        """空のFRB状態: 想起なし→縮小方向。"""
        state = ForgettingRecallBalanceState()
        result = apply_window_size_modulation(state)
        # 全キーが結果に含まれる
        for key in _WINDOW_SIZE_KEYS:
            assert key in result
        # 想起なしは縮小方向
        base_25 = coefficient_registry.get_defaults()["description_common"]["window_size_25"]
        assert result["window_size_25"] <= base_25

    def test_active_recall_expands_windows(self):
        """活発な想起: ウィンドウ拡大。"""
        state = _make_frb_state([(10, 5), (8, 7)])  # total=30, very active
        result = apply_window_size_modulation(state)
        base_30 = coefficient_registry.get_defaults()["description_common"]["window_size_30"]
        assert result["window_size_30"] > base_30

    def test_inactive_recall_shrinks_windows(self):
        """不活発な想起: ウィンドウ縮小。"""
        state = _make_frb_state([(0, 0)])  # total=0, inactive
        result = apply_window_size_modulation(state)
        base_50 = coefficient_registry.get_defaults()["description_common"]["window_size_50"]
        assert result["window_size_50"] < base_50

    def test_modulation_within_band(self):
        """変調後の値が帯域内であること。"""
        state = _make_frb_state([(20, 20)])  # very active
        result = apply_window_size_modulation(state)
        defaults = coefficient_registry.get_defaults()["description_common"]
        for key in _WINDOW_SIZE_KEYS:
            base = defaults[key]
            min_val = round(base * (1.0 - _MODULATION_BAND_RATIO))
            max_val = round(base * (1.0 + _MODULATION_BAND_RATIO))
            assert min_val <= result[key] <= max_val, (
                f"{key}: {result[key]} not in [{min_val}, {max_val}]"
            )

    def test_registry_values_updated(self):
        """変調後の値が係数レジストリに反映されること。"""
        state = _make_frb_state([(20, 20)])  # very active
        result = apply_window_size_modulation(state)
        for key in _WINDOW_SIZE_KEYS:
            reg_val = coefficient_registry.get("description_common", key)
            assert reg_val == result[key]

    def test_with_previous_modulated_values(self):
        """前回変調値ありの場合、回帰圧力が作用。"""
        state = _make_frb_state([(5, 5)])  # total=10, normal → factor=0.0
        prev = {"window_size_25": 30, "window_size_30": 36, "window_size_50": 60}
        result = apply_window_size_modulation(state, prev)
        # factor=0.0 + 回帰圧力 → 基準値方向
        base_30 = coefficient_registry.get_defaults()["description_common"]["window_size_30"]
        # 回帰圧力により、前回値36から基準値30方向に近づく
        assert result["window_size_30"] <= 36

    def test_frb_state_not_modified(self):
        """FRB状態が変更されないこと（READ-ONLY）。"""
        state = _make_frb_state([(5, 3)])
        original_cycle = state.cycle_count
        original_history_len = len(state.history)
        apply_window_size_modulation(state)
        assert state.cycle_count == original_cycle
        assert len(state.history) == original_history_len

    def test_all_integer_results(self):
        """全結果が整数であること。"""
        state = _make_frb_state([(7, 3)])
        result = apply_window_size_modulation(state)
        for key, val in result.items():
            assert isinstance(val, int), f"{key}: {val} is not int"

    def test_bidirectional_modulation(self):
        """双方向の変調が可能であること。"""
        # 縮小方向
        shrink_state = _make_frb_state([(0, 0)])
        shrink_result = apply_window_size_modulation(shrink_state)

        # レジストリをリセットして再ロード
        coefficient_registry.reset()
        coefficient_registry.load()

        # 拡大方向
        expand_state = _make_frb_state([(30, 30)])
        expand_result = apply_window_size_modulation(expand_state)

        for key in _WINDOW_SIZE_KEYS:
            assert shrink_result[key] < expand_result[key], (
                f"{key}: shrink={shrink_result[key]} >= expand={expand_result[key]}"
            )


# =============================================================================
# _apply_to_registry tests
# =============================================================================

class TestApplyToRegistry:
    """レジストリへの値適用テスト。"""

    def test_update_existing_key(self):
        """既存キーの値を更新できること。"""
        _apply_to_registry("window_size_30", 35)
        assert coefficient_registry.get("description_common", "window_size_30") == 35

    def test_update_multiple_keys(self):
        """複数キーを更新できること。"""
        _apply_to_registry("window_size_25", 28)
        _apply_to_registry("window_size_50", 55)
        assert coefficient_registry.get("description_common", "window_size_25") == 28
        assert coefficient_registry.get("description_common", "window_size_50") == 55

    def test_no_side_effects_on_other_keys(self):
        """他のキーに影響しないこと。"""
        original_fifo = coefficient_registry.get("description_common", "fifo_limit_30")
        _apply_to_registry("window_size_30", 35)
        assert coefficient_registry.get("description_common", "fifo_limit_30") == original_fifo


# =============================================================================
# save_modulated_values / load_modulated_values tests
# =============================================================================

class TestSaveLoadModulatedValues:
    """永続化テスト。"""

    def test_save_returns_dict(self):
        """save_modulated_values が辞書を返すこと。"""
        data = {"window_size_25": 28, "window_size_30": 33}
        result = save_modulated_values(data)
        assert isinstance(result, dict)
        assert result == data

    def test_load_from_dict(self):
        """辞書から正しく復元できること。"""
        data = {"window_size_25": 28, "window_size_30": 33}
        result = load_modulated_values(data)
        assert result == data

    def test_load_from_none(self):
        """Noneからは空辞書を返すこと。"""
        result = load_modulated_values(None)
        assert result == {}

    def test_load_from_invalid_type(self):
        """不正な型からは空辞書を返すこと。"""
        result = load_modulated_values("not a dict")
        assert result == {}
        result = load_modulated_values(123)
        assert result == {}

    def test_load_filters_invalid_values(self):
        """不正な値はフィルタされること。"""
        data = {
            "window_size_25": 28,
            "window_size_30": "invalid",
            "window_size_50": None,
        }
        result = load_modulated_values(data)
        assert "window_size_25" in result
        assert "window_size_30" not in result
        assert "window_size_50" not in result

    def test_round_trip(self):
        """save → load のラウンドトリップ。"""
        original = {"window_size_25": 28, "window_size_30": 33, "window_size_50": 55}
        saved = save_modulated_values(original)
        loaded = load_modulated_values(saved)
        assert loaded == original

    def test_load_converts_float_to_int(self):
        """float値はintに変換されること。"""
        data = {"window_size_25": 28.0, "window_size_30": 33.5}
        result = load_modulated_values(data)
        assert result["window_size_25"] == 28
        assert result["window_size_30"] == 33
        for v in result.values():
            assert isinstance(v, int)


# =============================================================================
# Safety valve tests
# =============================================================================

class TestSafetyValves:
    """安全弁のテスト。"""

    def test_band_limit_positive_extreme(self):
        """安全弁1: 最大拡大でも帯域内。"""
        result = compute_modulation(base_value=30, direction_factor=1.0)
        max_allowed = round(30 * (1.0 + _MODULATION_BAND_RATIO))
        assert result <= max_allowed

    def test_band_limit_negative_extreme(self):
        """安全弁1: 最大縮小でも帯域内。"""
        result = compute_modulation(base_value=30, direction_factor=-1.0)
        min_allowed = round(30 * (1.0 - _MODULATION_BAND_RATIO))
        assert result >= min_allowed

    def test_regression_toward_base(self):
        """安全弁2: 回帰圧力が基準値方向に作用。"""
        # 前回: 大きく拡大した状態（36）、今回: 方向係数0
        result = compute_modulation(
            base_value=30,
            direction_factor=0.0,
            previous_modulated=36,
        )
        # 回帰圧力により基準値方向（30）に引き戻される
        assert result < 36

    def test_regression_from_shrunk_state(self):
        """安全弁2: 縮小状態からの回帰。"""
        # 前回: 大きく縮小した状態（24）、今回: 方向係数0
        result = compute_modulation(
            base_value=30,
            direction_factor=0.0,
            previous_modulated=24,
        )
        # 回帰圧力により基準値方向（30）に引き戻される
        assert result > 24

    def test_integer_rounding(self):
        """安全弁3: 整数丸め。"""
        # 小さい基準値で非整数の変調量が発生するケース
        result = compute_modulation(base_value=7, direction_factor=0.3)
        assert isinstance(result, int)

    def test_read_only_frb_state(self):
        """安全弁5: FRB状態が変更されないこと。"""
        state = _make_frb_state([(5, 3), (2, 1)])
        original_data = state.to_dict()
        apply_window_size_modulation(state)
        assert state.to_dict() == original_data


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_very_small_base_value(self):
        """基準値が非常に小さい場合でも最低1を返す。"""
        result = compute_modulation(base_value=1, direction_factor=-1.0)
        assert result >= 1

    def test_large_recall_count(self):
        """非常に大きな想起頻度でも帯域内。"""
        state = _make_frb_state([(1000, 1000)])
        result = apply_window_size_modulation(state)
        defaults = coefficient_registry.get_defaults()["description_common"]
        for key in _WINDOW_SIZE_KEYS:
            base = defaults[key]
            max_val = round(base * (1.0 + _MODULATION_BAND_RATIO))
            assert result[key] <= max_val

    def test_extreme_previous_modulated(self):
        """前回変調値が帯域外の極端な値でも安全に動作。"""
        state = _make_frb_state([(5, 5)])
        prev = {"window_size_30": 100}  # 帯域外
        result = apply_window_size_modulation(state, prev)
        base_30 = coefficient_registry.get_defaults()["description_common"]["window_size_30"]
        max_val = round(base_30 * (1.0 + _MODULATION_BAND_RATIO))
        min_val = round(base_30 * (1.0 - _MODULATION_BAND_RATIO))
        assert min_val <= result["window_size_30"] <= max_val

    def test_many_history_entries(self):
        """大量の履歴エントリでも正常動作。"""
        entries = [(i % 5, i % 3) for i in range(200)]
        state = _make_frb_state(entries)
        result = apply_window_size_modulation(state)
        for key in _WINDOW_SIZE_KEYS:
            assert isinstance(result[key], int)
            assert result[key] >= 1

    def test_previous_modulated_partial_keys(self):
        """前回変調値が一部のキーのみの場合。"""
        state = _make_frb_state([(5, 5)])
        prev = {"window_size_25": 28}  # window_size_30, window_size_50 は欠落
        result = apply_window_size_modulation(state, prev)
        for key in _WINDOW_SIZE_KEYS:
            assert key in result

    def test_normal_recall_no_change(self):
        """普通の想起頻度（方向係数0）で前回変調なし: 基準値のまま。"""
        state = _make_frb_state([(5, 5)])  # total=10, normal
        result = apply_window_size_modulation(state)
        defaults = coefficient_registry.get_defaults()["description_common"]
        for key in _WINDOW_SIZE_KEYS:
            assert result[key] == defaults[key]
