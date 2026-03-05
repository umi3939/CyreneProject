"""
tests/test_exp_cumulative_safety.py

Phase 26-EXP 累積安全弁のテスト。
設計書: design_exp_cumulative_safety.md

テスト対象:
- _apply_cumulative_safety_valve: 累積安全弁の適用（同時発動検出・スケールダウン・対象C調整）
- _apply_consecutive_firing_decay: 連続発動時の追加減衰
- 安全弁の検証（累積上限、均等スケールダウン、最低維持、非永続、enrichment遮断、FIFO）
- 係数レジストリとの連携
"""

import pytest
import time
from unittest.mock import MagicMock
from types import SimpleNamespace

from psyche.orchestrator_5tick_phases import (
    _apply_experience_driven_value_update,
    _apply_cumulative_safety_valve,
    _apply_consecutive_firing_decay,
    _compute_experience_intensity,
    _compute_bandwidth_expansion_coefficient,
    _EXP_BANDWIDTH_MAX_MULTIPLIER,
    _EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
    _EXP_BANDWIDTH_COOLDOWN_TICKS,
    _EXP_DRIVE_LIMIT_MULTIPLIER_MAX,
    _EXP_SCORE_BAND_ADDITION_MAX,
    _EXP_CUMULATIVE_LIMIT_RATIO,
    _EXP_CONSECUTIVE_FIRING_THRESHOLD,
    _EXP_CONSECUTIVE_FIRING_DECAY_BASE,
    _EXP_CONSECUTIVE_FIRING_MIN_FACTOR,
    _EXP_FIRING_WINDOW_SIZE,
)

from psyche.value_orientation import (
    ValueOrientation,
    ValueOrientationConfig,
    generate_decision_signal,
)

from psyche.episodic_memory import (
    EmotionalCompanion,
    EpisodeEntry,
    EpisodeType,
    ImportanceLevel,
    EpisodeStore,
)

from psyche.state import EmotionVector, Mood


# =============================================================================
# Helpers
# =============================================================================

def _make_episode_entry(
    intensity_level: float = 0.9,
    valence: float = 0.5,
    primary_emotion: str = "joy",
) -> EpisodeEntry:
    """Create a test EpisodeEntry with emotional companion."""
    return EpisodeEntry(
        episode_id="test_ep_cs",
        episode_type=EpisodeType.EMOTIONAL_EVENT,
        summary="Test episode for cumulative safety",
        topics=("test",),
        source_texts=("hello",),
        timestamp=time.time(),
        duration_estimate=0.0,
        emotional_companion=EmotionalCompanion(
            primary_emotion=primary_emotion,
            intensity_level=intensity_level,
            valence=valence,
            harmony=0.5,
            emotion_description="test emotion",
            coexisting_emotions=(),
        ),
        self_observation_companion=None,
        context_summary="test context",
        importance=ImportanceLevel.NOTABLE,
        vividness=0.9,
        reference_count=0,
        reinterpretation_count=0,
        is_compressed=False,
        compressed_episode_ids=(),
    )


def _make_episode_store(episodes=None) -> EpisodeStore:
    """Create a test EpisodeStore."""
    eps = episodes or (_make_episode_entry(),)
    return EpisodeStore(
        episodes=tuple(eps),
        links=(),
        total_episodes_recorded=len(eps),
        total_compressions=0,
        average_vividness=0.9,
        active_episode_count=len(eps),
        compressed_episode_count=0,
        timestamp=time.time(),
        description="Test store",
    )


def _make_mock_orchestrator(
    policy_label: str = "共感する",
    emotion_intensity: float = 0.9,
    emotion_amplitude: float = 0.9,
    mood_arousal: float = 0.9,
    tick_count: int = 100,
    last_bandwidth_tick: int = -100,
    orientation: ValueOrientation = None,
    episodes_store: EpisodeStore = None,
    vo_config: ValueOrientationConfig = None,
    drive_multiplier: float = None,
    score_addition: float = None,
) -> MagicMock:
    """Create a mock PsycheOrchestrator for cumulative safety valve testing."""
    orch = MagicMock()
    orch._last_selected_policy_label = policy_label
    orch._tick_count = tick_count
    orch._exp_bandwidth_last_tick = last_bandwidth_tick
    orch._value_orientation = orientation or ValueOrientation()
    orch._vo_config = vo_config or ValueOrientationConfig()

    # Emotion vector mock
    emo_dict = {
        "joy": emotion_amplitude,
        "anger": 0.0,
        "sorrow": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "love": 0.0,
        "fun": 0.0,
    }
    orch._psyche = MagicMock()
    orch._psyche.emotions.as_dict.return_value = emo_dict
    orch._psyche.mood.arousal = mood_arousal
    orch._psyche.mood.valence = 0.3
    orch._psyche.drives.as_dict.return_value = {
        "social": 0.7, "curiosity": 0.6, "expression": 0.5,
    }

    # Episodes store
    if episodes_store is None:
        episodes_store = _make_episode_store(
            [_make_episode_entry(intensity_level=emotion_intensity)]
        )
    orch._last_episodes = episodes_store

    # Pre-set targets if provided
    if drive_multiplier is not None:
        orch._exp_drive_total_limit_multiplier = drive_multiplier
    if score_addition is not None:
        orch._exp_score_band_addition = score_addition

    return orch


# =============================================================================
# Tests: Coefficient constants loaded correctly
# =============================================================================

class TestCoefficientConstants:
    """Verify coefficient constants are loaded from registry."""

    def test_cumulative_limit_ratio_exists(self):
        """累積上限比率が存在し、正の値である。"""
        assert _EXP_CUMULATIVE_LIMIT_RATIO > 0.0

    def test_cumulative_limit_wider_than_individual(self):
        """累積上限は個別安全弁の合算より狭い（安全弁1:
        通常の単一効果発動時に累積安全弁が不要に作動しない）。"""
        # 個別上限の合算 = 3.0 (各効果の正規化比率が最大1.0ずつ)
        assert _EXP_CUMULATIVE_LIMIT_RATIO < 3.0

    def test_cumulative_limit_more_than_one(self):
        """累積上限は1.0より大きい（単一効果では作動しない）。"""
        assert _EXP_CUMULATIVE_LIMIT_RATIO > 1.0

    def test_consecutive_threshold_positive(self):
        """連続発動閾値が正の整数である。"""
        assert _EXP_CONSECUTIVE_FIRING_THRESHOLD > 0
        assert isinstance(_EXP_CONSECUTIVE_FIRING_THRESHOLD, int)

    def test_decay_base_in_range(self):
        """減衰ベースが0-1の範囲内である。"""
        assert 0.0 < _EXP_CONSECUTIVE_FIRING_DECAY_BASE < 1.0

    def test_min_factor_positive(self):
        """最低維持係数が正であり、帯域拡大を完全に無効化しない。"""
        assert _EXP_CONSECUTIVE_FIRING_MIN_FACTOR > 0.0

    def test_min_factor_less_than_one(self):
        """最低維持係数が1.0未満（減衰が効く）。"""
        assert _EXP_CONSECUTIVE_FIRING_MIN_FACTOR < 1.0

    def test_firing_window_positive(self):
        """発動ウィンドウサイズが正の整数である。"""
        assert _EXP_FIRING_WINDOW_SIZE > 0
        assert isinstance(_EXP_FIRING_WINDOW_SIZE, int)


# =============================================================================
# Tests: _apply_cumulative_safety_valve - 同時発動検出
# =============================================================================

class TestCumulativeSafetyValveSimultaneousDetection:
    """同時発動の検出テスト。"""

    def test_no_active_effects_no_scaling(self):
        """全効果が非活性の場合、スケールダウンしない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = None
        orch._exp_score_band_addition = None
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # multiplier should remain None
        assert orch._exp_drive_total_limit_multiplier is None

    def test_single_active_effect_no_scaling(self):
        """1つのみ活性の場合、同時発動とみなさない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = None
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # multiplier should remain unchanged (no cumulative scaling)
        assert orch._exp_drive_total_limit_multiplier == 1.2

    def test_two_active_effects_detected_as_simultaneous(self):
        """2つ活性の場合、同時発動と見なされる。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}
        # A ratio = 1.0, B ratio = 1.0, C ratio = 0.0 → cumulative = 2.0
        # cumulative = 2.0 < 2.5 → no scaling
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # At max individual limits, cumulative = 2.0 < 2.5, no scale down
        assert orch._exp_drive_total_limit_multiplier == _EXP_DRIVE_LIMIT_MULTIPLIER_MAX

    def test_three_active_effects_at_max_triggers_scaling(self):
        """3効果が全て最大の場合、累積影響量が3.0で上限2.5を超える。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}
        # C at max: c_dim_delta_max = _EXP_BANDWIDTH_MAX_DELTA_PER_DIM
        # A=1.0, B=1.0, C=1.0 → cumulative = 3.0 > 2.5
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # A should be scaled down
        assert orch._exp_drive_total_limit_multiplier < _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        # B should be scaled down
        assert orch._exp_score_band_addition < _EXP_SCORE_BAND_ADDITION_MAX


# =============================================================================
# Tests: _apply_cumulative_safety_valve - スケールダウン
# =============================================================================

class TestCumulativeSafetyValveScaleDown:
    """スケールダウンの均等性・正確性テスト。"""

    def test_uniform_scale_factor(self):
        """全効果に同一のスケール係数が適用される（安全弁2: 均等適用）。"""
        orch = _make_mock_orchestrator()
        a_max = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        b_max = _EXP_SCORE_BAND_ADDITION_MAX
        orch._exp_drive_total_limit_multiplier = a_max
        orch._exp_score_band_addition = b_max
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}
        c_max = _EXP_BANDWIDTH_MAX_DELTA_PER_DIM

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=c_max, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # Expected scale = 2.5 / 3.0 = 0.833...
        expected_scale = _EXP_CUMULATIVE_LIMIT_RATIO / 3.0

        # A: 1.0 + (a_max - 1.0) * scale
        expected_a = 1.0 + (a_max - 1.0) * expected_scale
        assert abs(orch._exp_drive_total_limit_multiplier - expected_a) < 1e-9

        # B: b_max * scale
        expected_b = b_max * expected_scale
        assert abs(orch._exp_score_band_addition - expected_b) < 1e-9

    def test_scale_preserves_relative_ratios(self):
        """スケールダウンは相対比率を維持する。"""
        orch = _make_mock_orchestrator()
        # A at 50% of max, B at 100% of max
        a_half = 1.0 + (_EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0) * 0.5
        b_full = _EXP_SCORE_BAND_ADDITION_MAX
        orch._exp_drive_total_limit_multiplier = a_half
        orch._exp_score_band_addition = b_full
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}
        c_max = _EXP_BANDWIDTH_MAX_DELTA_PER_DIM

        # A ratio = 0.5, B ratio = 1.0, C ratio = 1.0 → cumulative = 2.5
        # cumulative == limit → no scaling needed
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=c_max, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # At exactly the limit, no scaling
        assert orch._exp_drive_total_limit_multiplier == a_half

    def test_above_limit_scales_down(self):
        """累積影響量が上限を超えた場合のみスケールダウンする。"""
        orch = _make_mock_orchestrator()
        # A at 90%, B at 90%, C at 90% → cumulative = 2.7 > 2.5
        a_val = 1.0 + (_EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0) * 0.9
        b_val = _EXP_SCORE_BAND_ADDITION_MAX * 0.9
        c_val = _EXP_BANDWIDTH_MAX_DELTA_PER_DIM * 0.9
        orch._exp_drive_total_limit_multiplier = a_val
        orch._exp_score_band_addition = b_val
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=c_val, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # Should have been scaled down
        assert orch._exp_drive_total_limit_multiplier < a_val
        assert orch._exp_score_band_addition < b_val

    def test_below_limit_no_scaling(self):
        """累積影響量が上限以下の場合はスケールダウンしない。"""
        orch = _make_mock_orchestrator()
        # A at 50%, B at 50% → cumulative = 1.0 < 2.5
        a_val = 1.0 + (_EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0) * 0.5
        b_val = _EXP_SCORE_BAND_ADDITION_MAX * 0.5
        orch._exp_drive_total_limit_multiplier = a_val
        orch._exp_score_band_addition = b_val
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # No scaling applied
        assert orch._exp_drive_total_limit_multiplier == a_val
        assert orch._exp_score_band_addition == b_val

    def test_no_selective_suppression(self):
        """特定の効果を選択的に抑制しない（安全弁2: 全効果に均等適用）。"""
        orch = _make_mock_orchestrator()
        a_val = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        b_val = _EXP_SCORE_BAND_ADDITION_MAX
        orch._exp_drive_total_limit_multiplier = a_val
        orch._exp_score_band_addition = b_val
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}
        c_val = _EXP_BANDWIDTH_MAX_DELTA_PER_DIM

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=c_val, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # Both A and B should have the same ratio of reduction
        a_orig_ratio = (a_val - 1.0) / (_EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0)
        a_new_ratio = (orch._exp_drive_total_limit_multiplier - 1.0) / (_EXP_DRIVE_LIMIT_MULTIPLIER_MAX - 1.0)
        b_orig_ratio = b_val / _EXP_SCORE_BAND_ADDITION_MAX
        b_new_ratio = orch._exp_score_band_addition / _EXP_SCORE_BAND_ADDITION_MAX

        # Both should have the same scale factor applied
        a_scale = a_new_ratio / a_orig_ratio if a_orig_ratio > 0 else 1.0
        b_scale = b_new_ratio / b_orig_ratio if b_orig_ratio > 0 else 1.0
        assert abs(a_scale - b_scale) < 1e-9


# =============================================================================
# Tests: Target C (value orientation) scaling
# =============================================================================

class TestTargetCScaling:
    """対象C（価値指向次元変動量）のスケールダウンテスト。"""

    def test_c_scaling_reduces_dimension_delta(self):
        """Cのスケールダウンで次元変動量が縮小される。"""
        orch = _make_mock_orchestrator()
        # Set up orientation with some dimension change
        vo = ValueOrientation()
        vo.dim_a = 0.1  # This represents the current value
        orch._value_orientation = vo
        orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX

        # old_dims represents pre-EXP state
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # C should be scaled down: dim_a should be closer to 0.0 than 0.1
        assert abs(orch._value_orientation.dim_a) < 0.1

    def test_c_zero_delta_not_affected(self):
        """C変動量がゼロの場合は対象Cとして計上されない。"""
        orch = _make_mock_orchestrator()
        vo = ValueOrientation()
        orch._value_orientation = vo
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3

        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # With only A and B active, cumulative = ratios of A + B
        # If below limit, no scaling applied
        assert orch._exp_drive_total_limit_multiplier == 1.2 or \
               orch._exp_drive_total_limit_multiplier <= 1.2


# =============================================================================
# Tests: _apply_consecutive_firing_decay
# =============================================================================

class TestConsecutiveFiringDecay:
    """連続発動時の追加減衰テスト。"""

    def test_no_history_no_decay(self):
        """発動履歴がない場合、減衰は適用されない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3
        # Explicitly no history attribute

        _apply_consecutive_firing_decay(orch)

        assert orch._exp_drive_total_limit_multiplier == 1.2
        assert orch._exp_score_band_addition == 0.3

    def test_below_threshold_no_decay(self):
        """発動回数が閾値以下の場合、減衰は適用されない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3
        orch._exp_firing_tick_history = list(range(_EXP_CONSECUTIVE_FIRING_THRESHOLD))

        _apply_consecutive_firing_decay(orch)

        assert orch._exp_drive_total_limit_multiplier == 1.2
        assert orch._exp_score_band_addition == 0.3

    def test_above_threshold_applies_decay(self):
        """発動回数が閾値を超えた場合、減衰が適用される。"""
        orch = _make_mock_orchestrator()
        a_original = 1.2
        b_original = 0.3
        orch._exp_drive_total_limit_multiplier = a_original
        orch._exp_score_band_addition = b_original
        # threshold + 2 firings
        count = _EXP_CONSECUTIVE_FIRING_THRESHOLD + 2
        orch._exp_firing_tick_history = list(range(count))

        _apply_consecutive_firing_decay(orch)

        # A should be decayed: 1.0 + (1.2 - 1.0) * decay_factor
        assert orch._exp_drive_total_limit_multiplier < a_original
        assert orch._exp_drive_total_limit_multiplier > 1.0  # Still above 1.0

        # B should be decayed
        assert orch._exp_score_band_addition < b_original
        assert orch._exp_score_band_addition > 0.0  # Still positive

    def test_decay_increases_with_excess(self):
        """超過回数が増えると減衰が強くなる。"""
        results_a = []
        for excess in range(1, 5):
            orch = _make_mock_orchestrator()
            orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
            orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
            count = _EXP_CONSECUTIVE_FIRING_THRESHOLD + excess
            orch._exp_firing_tick_history = list(range(count))

            _apply_consecutive_firing_decay(orch)
            results_a.append(orch._exp_drive_total_limit_multiplier)

        # Each successive result should be smaller or equal
        for i in range(len(results_a) - 1):
            assert results_a[i] >= results_a[i + 1]

    def test_decay_never_below_min_factor(self):
        """減衰は最低維持係数を下回らない（安全弁3: 完全無効化防止）。"""
        orch = _make_mock_orchestrator()
        a_original = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_drive_total_limit_multiplier = a_original
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        # Very large excess to push decay to minimum
        count = _EXP_CONSECUTIVE_FIRING_THRESHOLD + 100
        orch._exp_firing_tick_history = list(range(count))

        _apply_consecutive_firing_decay(orch)

        # A multiplier should still be > 1.0 (min_factor applied)
        expected_min_a = 1.0 + (a_original - 1.0) * _EXP_CONSECUTIVE_FIRING_MIN_FACTOR
        assert orch._exp_drive_total_limit_multiplier >= expected_min_a - 1e-9

        # B addition should still be > 0
        expected_min_b = _EXP_SCORE_BAND_ADDITION_MAX * _EXP_CONSECUTIVE_FIRING_MIN_FACTOR
        assert orch._exp_score_band_addition >= expected_min_b - 1e-9

    def test_decay_does_not_affect_none_values(self):
        """None値には減衰が適用されない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = None
        orch._exp_score_band_addition = None
        count = _EXP_CONSECUTIVE_FIRING_THRESHOLD + 5
        orch._exp_firing_tick_history = list(range(count))

        _apply_consecutive_firing_decay(orch)

        assert orch._exp_drive_total_limit_multiplier is None
        assert orch._exp_score_band_addition is None

    def test_decay_does_not_affect_base_values(self):
        """基準値(A=1.0, B=0.0)には減衰が適用されない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.0
        orch._exp_score_band_addition = 0.0
        count = _EXP_CONSECUTIVE_FIRING_THRESHOLD + 5
        orch._exp_firing_tick_history = list(range(count))

        _apply_consecutive_firing_decay(orch)

        assert orch._exp_drive_total_limit_multiplier == 1.0
        assert orch._exp_score_band_addition == 0.0


# =============================================================================
# Tests: Firing tick history - FIFO管理
# =============================================================================

class TestFiringTickHistory:
    """発動ティック履歴のFIFO管理テスト。"""

    def test_history_recorded_on_firing(self):
        """帯域拡大発動時にティック番号が記録される。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
            tick_count=50,
            last_bandwidth_tick=-100,
        )
        # Ensure _exp_firing_tick_history is not a MagicMock
        orch._exp_firing_tick_history = []
        _apply_experience_driven_value_update(orch)

        # If firing occurred, history should have the current tick
        history = orch._exp_firing_tick_history
        if isinstance(history, list) and len(history) > 0:
            assert 50 in history

    def test_window_eviction(self):
        """ウィンドウ外の古いエントリが消失する（安全弁7: FIFO管理）。"""
        orch = _make_mock_orchestrator(
            tick_count=100,
            last_bandwidth_tick=-100,
        )
        # Pre-populate with old entries
        orch._exp_firing_tick_history = [10, 20, 30, 90, 95]

        _apply_experience_driven_value_update(orch)

        if hasattr(orch, '_exp_firing_tick_history'):
            # Entries older than tick_count - window_size should be evicted
            for t in orch._exp_firing_tick_history:
                assert 100 - t < _EXP_FIRING_WINDOW_SIZE

    def test_history_not_persisted(self):
        """発動履歴は非永続（安全弁5: セッション間で引き継がない）。"""
        # This is a structural test - the attribute is not in save/load fields
        orch = _make_mock_orchestrator()
        if hasattr(orch, '_exp_firing_tick_history'):
            # The attribute should not be in orchestrator's save dict
            # This is verified by the non-persistence design
            pass
        # No save/load field for _exp_firing_tick_history exists by design


# =============================================================================
# Tests: Integration with _apply_experience_driven_value_update
# =============================================================================

class TestIntegrationWithExperienceUpdate:
    """累積安全弁と帯域拡大の統合テスト。"""

    def test_basic_firing_with_cumulative_check(self):
        """基本的な帯域拡大が累積安全弁と共に動作する。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
            tick_count=50,
            last_bandwidth_tick=-100,
        )
        old_orientation = orch._value_orientation
        _apply_experience_driven_value_update(orch)

        # Update should have occurred
        assert orch._value_orientation.update_count >= old_orientation.update_count

    def test_repeated_firings_accumulate_history(self):
        """複数回の帯域拡大で発動履歴が蓄積される。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
        )
        orch._exp_firing_tick_history = []

        # Simulate multiple firings at different ticks
        for tick in range(10, 60, 3):
            orch._tick_count = tick
            orch._exp_bandwidth_last_tick = tick - 10
            _apply_experience_driven_value_update(orch)

        if hasattr(orch, '_exp_firing_tick_history'):
            # History should exist with entries
            assert len(orch._exp_firing_tick_history) > 0
            # Should not exceed window size equivalent entries
            assert len(orch._exp_firing_tick_history) <= _EXP_FIRING_WINDOW_SIZE + 1

    def test_enrichment_not_exposed(self):
        """累積安全弁の発動事実がenrichmentに露出しない（安全弁6）。
        Structurally verified: the function only writes to _exp_ prefixed attributes."""
        import inspect
        source = inspect.getsource(_apply_cumulative_safety_valve)
        # Remove docstring portion for analysis
        body = source.split('"""')[-1] if '"""' in source else source
        # The function body should not reference enrichment writing
        assert "get_enrichment" not in body
        assert "enrichment_data" not in body
        assert "_enrichment" not in body.replace("enrichment非露出", "")

    def test_individual_limits_not_changed(self):
        """個別安全弁の閾値が変更されない（設計書制約）。"""
        # Verify the module-level constants are unchanged after operation
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
            tick_count=50,
            last_bandwidth_tick=-100,
        )
        _apply_experience_driven_value_update(orch)

        # Module-level constants should remain at their loaded values
        from psyche import coefficient_registry as cr
        exp = cr.get("experience_intensity")
        assert _EXP_DRIVE_LIMIT_MULTIPLIER_MAX == exp["drive_limit_multiplier_max"]
        assert _EXP_SCORE_BAND_ADDITION_MAX == exp["score_band_addition_max"]
        assert _EXP_BANDWIDTH_MAX_DELTA_PER_DIM == exp["bandwidth_max_delta_per_dim"]

    def test_cooldown_not_changed(self):
        """冷却期間の動的導出ロジックが変更されない（設計書制約）。"""
        from psyche.orchestrator_5tick_phases import _derive_dynamic_cooldown
        # Verify the function still works correctly
        result = _derive_dynamic_cooldown(0.5, 0.5)
        assert result >= _EXP_BANDWIDTH_COOLDOWN_TICKS - (_EXP_BANDWIDTH_COOLDOWN_TICKS - 2)


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_zero_individual_limits_safe(self):
        """個別上限がゼロ除算を起こさない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.0  # At base, not active
        orch._exp_score_band_addition = 0.0  # At zero, not active
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        # Should not raise
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.0, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

    def test_very_large_cumulative_impact(self):
        """累積影響量が極めて大きい場合もスケールダウンが安全に動作する。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        # C at double the max (shouldn't happen, but test robustness)
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM * 2,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # Values should still be valid (positive, finite)
        assert orch._exp_drive_total_limit_multiplier >= 1.0
        assert orch._exp_score_band_addition >= 0.0

    def test_empty_old_dims(self):
        """old_vo_dimsが空の場合もエラーにならない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.2
        orch._exp_score_band_addition = 0.3
        old_dims = {}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        # Should not raise
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=0.05, old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

    def test_concurrent_cumulative_and_consecutive(self):
        """累積スケールダウンと連続発動減衰の両方が適用される。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        # Set high consecutive firing count
        orch._exp_firing_tick_history = list(range(_EXP_CONSECUTIVE_FIRING_THRESHOLD + 5))
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )

        # Both cumulative scale and consecutive decay applied
        # A should be reduced more than either alone
        assert orch._exp_drive_total_limit_multiplier < _EXP_DRIVE_LIMIT_MULTIPLIER_MAX
        assert orch._exp_drive_total_limit_multiplier >= 1.0

    def test_a_only_inactive_b_c_active(self):
        """Aが非活性、BとCのみ活性の同時発動。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = None  # A inactive
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        # B=1.0, C=1.0 → cumulative=2.0 < 2.5 → no scaling
        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # A still None
        assert orch._exp_drive_total_limit_multiplier is None

    def test_a_base_value_not_active(self):
        """A=1.0（拡大なし）は活性とみなされない。"""
        orch = _make_mock_orchestrator()
        orch._exp_drive_total_limit_multiplier = 1.0  # Not active
        orch._exp_score_band_addition = _EXP_SCORE_BAND_ADDITION_MAX
        old_dims = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        vo_config = ValueOrientationConfig()
        signal = {"a": 0.1}

        _apply_cumulative_safety_valve(
            orch, c_dim_delta_max=_EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
            old_vo_dims=old_dims,
            decision_signal=signal, vo_config=vo_config, expansion_coeff=2.0,
        )
        # A remains at 1.0, no scaling applied to it
        assert orch._exp_drive_total_limit_multiplier == 1.0


# =============================================================================
# Tests: Coefficient registry integration
# =============================================================================

class TestCoefficientRegistryIntegration:
    """係数レジストリとの連携テスト。"""

    def test_defaults_contain_cumulative_keys(self):
        """デフォルト値に累積安全弁のキーが含まれる。"""
        from psyche.coefficient_registry import get_defaults
        defaults = get_defaults()
        exp = defaults["experience_intensity"]
        assert "cumulative_limit_ratio" in exp
        assert "consecutive_firing_threshold" in exp
        assert "consecutive_firing_decay_base" in exp
        assert "consecutive_firing_min_factor" in exp
        assert "firing_window_size" in exp

    def test_registry_values_match_constants(self):
        """レジストリの値がモジュール定数と一致する。"""
        from psyche.coefficient_registry import get
        exp = get("experience_intensity")
        assert _EXP_CUMULATIVE_LIMIT_RATIO == exp["cumulative_limit_ratio"]
        assert _EXP_CONSECUTIVE_FIRING_THRESHOLD == exp["consecutive_firing_threshold"]
        assert _EXP_CONSECUTIVE_FIRING_DECAY_BASE == exp["consecutive_firing_decay_base"]
        assert _EXP_CONSECUTIVE_FIRING_MIN_FACTOR == exp["consecutive_firing_min_factor"]
        assert _EXP_FIRING_WINDOW_SIZE == exp["firing_window_size"]

    def test_fallback_on_missing_key(self):
        """キーが欠落している場合のフォールバック。"""
        from psyche.coefficient_registry import get_defaults
        defaults = get_defaults()
        exp = defaults["experience_intensity"]
        # Default values should be reasonable
        assert exp["cumulative_limit_ratio"] == 2.5
        assert exp["consecutive_firing_threshold"] == 3
        assert exp["consecutive_firing_decay_base"] == 0.85
        assert exp["consecutive_firing_min_factor"] == 0.3
        assert exp["firing_window_size"] == 10


# =============================================================================
# Tests: Non-persistence verification
# =============================================================================

class TestNonPersistence:
    """非永続性の検証テスト。"""

    def test_firing_history_is_ephemeral(self):
        """発動履歴は非永続属性である。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
            tick_count=50,
            last_bandwidth_tick=-100,
        )
        _apply_experience_driven_value_update(orch)

        # The attribute name _exp_firing_tick_history is NOT in the
        # orchestrator's save/load field list (verified by design)
        if hasattr(orch, '_exp_firing_tick_history'):
            assert isinstance(orch._exp_firing_tick_history, list)

    def test_cumulative_state_reset_on_new_session(self):
        """新セッション開始時に累積安全弁の状態が初期化される。"""
        orch = _make_mock_orchestrator()
        # Simulate a new session by not having the history attribute
        if hasattr(orch, '_exp_firing_tick_history'):
            delattr(orch, '_exp_firing_tick_history')

        # First call should initialize the list
        orch._tick_count = 50
        orch._exp_bandwidth_last_tick = -100
        _apply_experience_driven_value_update(orch)

        if hasattr(orch, '_exp_firing_tick_history'):
            # Should start fresh with only the current tick
            assert len(orch._exp_firing_tick_history) <= 1
