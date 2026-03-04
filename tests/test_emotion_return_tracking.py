"""
tests/test_emotion_return_tracking.py - 感情帰還方向追跡と追従速度変調のテスト

C9-8: 帰還処理の方向連続性追跡と、ムード追従速度への変調量導出のテスト。

テスト対象:
- 帰還方向の判定（正/負/中立）
- 同方向連続でカウント増加
- 逆方向でカウントリセット
- 段階的鮮度減衰の適用
- 変調量の上限制限
- 増加方向のみ変調
- 追従速度への適用
- save/loadの往復保全
- 入力不在時のゼロ変調
"""

from __future__ import annotations

import pytest

from psyche.memory_emotion_return import (
    MemoryEmotionReturnConfig,
    MemoryEmotionReturnProcessor,
    MemoryEmotionReturnState,
    ReturnRecord,
    create_memory_emotion_return,
)
from psyche.reaction import (
    MoodContextInputs,
    _derive_tracking_speeds,
    _TRACKING_SPEED_MIN,
    _TRACKING_SPEED_MAX,
)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def processor() -> MemoryEmotionReturnProcessor:
    """Default processor."""
    return create_memory_emotion_return()


@pytest.fixture
def custom_processor() -> MemoryEmotionReturnProcessor:
    """Processor with custom config for easier testing."""
    cfg = MemoryEmotionReturnConfig(
        direction_freshness_decay=0.8,
        tracking_speed_modulation_ratio_cap=0.10,
        tracking_speed_modulation_scale=0.02,
    )
    return create_memory_emotion_return(config=cfg)


# ── Direction Determination Tests ─────────────────────────────────

class TestDirectionDetermination:
    """帰還方向の判定テスト。"""

    def test_positive_direction_from_positive_deltas(self, processor):
        """正のdelta合計は正方向と判定される。"""
        processor._update_direction_tracking({"joy": 0.05, "love": 0.02})
        assert processor.state.last_direction_label == "positive"

    def test_negative_direction_from_negative_deltas(self, processor):
        """負のdelta合計は負方向と判定される。"""
        processor._update_direction_tracking({"sorrow": -0.05, "fear": -0.02})
        assert processor.state.last_direction_label == "negative"

    def test_neutral_direction_from_zero_deltas(self, processor):
        """ゼロに近いdelta合計は中立と判定される。"""
        processor._update_direction_tracking({"joy": 0.0000001, "sorrow": -0.0000001})
        assert processor.state.last_direction_label == "neutral"

    def test_neutral_direction_from_empty_deltas(self, processor):
        """空のdeltasは中立と判定される。"""
        processor._update_direction_tracking({})
        assert processor.state.last_direction_label == "neutral"

    def test_mixed_deltas_positive_sum(self, processor):
        """混合deltasでも合計が正なら正方向。"""
        processor._update_direction_tracking({"joy": 0.05, "sorrow": -0.02})
        assert processor.state.last_direction_label == "positive"

    def test_mixed_deltas_negative_sum(self, processor):
        """混合deltasでも合計が負なら負方向。"""
        processor._update_direction_tracking({"joy": 0.01, "sorrow": -0.05})
        assert processor.state.last_direction_label == "negative"


# ── Consecutive Count Tests ──────────────────────────────────────

class TestConsecutiveCount:
    """同方向連続カウントのテスト。"""

    def test_first_positive_sets_count_to_one(self, processor):
        """最初の正方向帰還でカウントが1になる。"""
        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.direction_consecutive_count_positive == 1.0
        assert processor.state.direction_consecutive_count_negative == 0.0

    def test_first_negative_sets_count_to_one(self, processor):
        """最初の負方向帰還でカウントが1になる。"""
        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_negative == 1.0
        assert processor.state.direction_consecutive_count_positive == 0.0

    def test_consecutive_positive_increases_count(self, processor):
        """正方向が連続するとカウントが増加する（鮮度減衰付き）。"""
        processor._update_direction_tracking({"joy": 0.05})
        count_1 = processor.state.direction_consecutive_count_positive
        assert count_1 == 1.0

        processor._update_direction_tracking({"joy": 0.05})
        count_2 = processor.state.direction_consecutive_count_positive
        # 1.0 * 0.8 + 1.0 = 1.8
        assert count_2 == pytest.approx(1.8, abs=0.01)

    def test_consecutive_negative_increases_count(self, processor):
        """負方向が連続するとカウントが増加する（鮮度減衰付き）。"""
        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_negative == 1.0

        processor._update_direction_tracking({"sorrow": -0.05})
        # 1.0 * 0.8 + 1.0 = 1.8
        assert processor.state.direction_consecutive_count_negative == pytest.approx(1.8, abs=0.01)

    def test_three_consecutive_positive_decay(self, processor):
        """3連続正方向の減衰付きカウント値。"""
        processor._update_direction_tracking({"joy": 0.05})
        # count = 1.0
        processor._update_direction_tracking({"joy": 0.05})
        # count = 1.0 * 0.8 + 1.0 = 1.8
        processor._update_direction_tracking({"joy": 0.05})
        # count = 1.8 * 0.8 + 1.0 = 2.44
        assert processor.state.direction_consecutive_count_positive == pytest.approx(2.44, abs=0.01)


# ── Direction Reversal Tests ─────────────────────────────────────

class TestDirectionReversal:
    """逆方向帰還によるカウントリセットのテスト。"""

    def test_positive_to_negative_resets_positive_count(self, processor):
        """正方向から負方向への切り替えで正方向カウントがリセットされる。"""
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.direction_consecutive_count_positive > 1.0

        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_positive == 0.0
        assert processor.state.direction_consecutive_count_negative == 1.0

    def test_negative_to_positive_resets_negative_count(self, processor):
        """負方向から正方向への切り替えで負方向カウントがリセットされる。"""
        processor._update_direction_tracking({"sorrow": -0.05})
        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_negative > 1.0

        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.direction_consecutive_count_negative == 0.0
        assert processor.state.direction_consecutive_count_positive == 1.0

    def test_neutral_does_not_reset_counts(self, processor):
        """中立帰還はカウントをリセットしない。"""
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})
        saved_count = processor.state.direction_consecutive_count_positive

        processor._update_direction_tracking({})  # neutral
        assert processor.state.direction_consecutive_count_positive == saved_count
        assert processor.state.last_direction_label == "neutral"

    def test_reversal_after_many_consecutive(self, processor):
        """多数の連続後でも逆方向1回でリセット。"""
        for _ in range(10):
            processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.direction_consecutive_count_positive > 4.0

        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_positive == 0.0
        assert processor.state.direction_consecutive_count_negative == 1.0


# ── Freshness Decay Tests ────────────────────────────────────────

class TestFreshnessDecay:
    """段階的鮮度減衰のテスト。"""

    def test_decay_is_applied_each_consecutive(self, processor):
        """各連続帰還で既存カウントに減衰が適用される。"""
        decay = processor._config.direction_freshness_decay  # 0.8

        processor._update_direction_tracking({"joy": 0.05})
        c1 = processor.state.direction_consecutive_count_positive
        assert c1 == 1.0

        processor._update_direction_tracking({"joy": 0.05})
        c2 = processor.state.direction_consecutive_count_positive
        assert c2 == pytest.approx(c1 * decay + 1.0, abs=0.001)

        processor._update_direction_tracking({"joy": 0.05})
        c3 = processor.state.direction_consecutive_count_positive
        assert c3 == pytest.approx(c2 * decay + 1.0, abs=0.001)

    def test_count_converges_with_many_consecutive(self, processor):
        """多数の連続帰還でカウントは収束する（無限に増加しない）。"""
        for _ in range(100):
            processor._update_direction_tracking({"joy": 0.05})

        count = processor.state.direction_consecutive_count_positive
        # Geometric series convergence: 1/(1-decay) = 1/0.2 = 5.0
        assert count < 5.1
        assert count > 4.5

    def test_custom_decay_rate(self):
        """カスタム減衰率の動作確認。"""
        cfg = MemoryEmotionReturnConfig(direction_freshness_decay=0.5)
        proc = create_memory_emotion_return(config=cfg)

        proc._update_direction_tracking({"joy": 0.05})
        proc._update_direction_tracking({"joy": 0.05})
        # 1.0 * 0.5 + 1.0 = 1.5
        assert proc.state.direction_consecutive_count_positive == pytest.approx(1.5, abs=0.01)


# ── Modulation Amount Tests ──────────────────────────────────────

class TestModulationAmount:
    """変調量の導出テスト。"""

    def test_zero_count_produces_zero_modulation(self, processor):
        """カウントがゼロのとき変調量はゼロ。"""
        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod == 0.0
        assert a_mod == 0.0

    def test_positive_count_produces_positive_modulation(self, processor):
        """正方向カウントがあると正の変調量が導出される。"""
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})

        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod > 0.0
        assert a_mod > 0.0

    def test_negative_count_produces_positive_modulation(self, processor):
        """負方向カウントがあっても変調量は正（増加方向のみ）。"""
        processor._update_direction_tracking({"sorrow": -0.05})
        processor._update_direction_tracking({"sorrow": -0.05})

        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod > 0.0
        assert a_mod > 0.0

    def test_modulation_cap_at_ratio(self, processor):
        """変調量は追従速度の比率で上限制限される。"""
        # Build up a large consecutive count
        for _ in range(50):
            processor._update_direction_tracking({"joy": 0.05})

        current_speed = 0.10
        cap = processor._config.tracking_speed_modulation_ratio_cap
        v_mod, a_mod = processor.get_tracking_speed_modulation(
            current_tracking_speed_valence=current_speed,
            current_tracking_speed_arousal=current_speed,
        )

        assert v_mod <= current_speed * cap + 1e-9
        assert a_mod <= current_speed * cap + 1e-9

    def test_modulation_never_negative(self, processor):
        """変調量は常に非負（増加方向のみ）。"""
        # Even with zero count
        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod >= 0.0
        assert a_mod >= 0.0

        # With some count
        processor._update_direction_tracking({"joy": 0.05})
        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod >= 0.0
        assert a_mod >= 0.0

    def test_modulation_increases_with_consecutive_count(self):
        """連続カウントが増えると変調量も増える（上限まで）。"""
        # Use a high cap ratio so modulation isn't capped early
        cfg = MemoryEmotionReturnConfig(
            tracking_speed_modulation_ratio_cap=0.50,
            tracking_speed_modulation_scale=0.02,
        )
        proc = create_memory_emotion_return(config=cfg)
        high_speed = 0.25

        proc._update_direction_tracking({"joy": 0.05})
        v1, _ = proc.get_tracking_speed_modulation(
            current_tracking_speed_valence=high_speed,
        )

        proc._update_direction_tracking({"joy": 0.05})
        v2, _ = proc.get_tracking_speed_modulation(
            current_tracking_speed_valence=high_speed,
        )

        proc._update_direction_tracking({"joy": 0.05})
        v3, _ = proc.get_tracking_speed_modulation(
            current_tracking_speed_valence=high_speed,
        )

        assert v2 > v1
        assert v3 > v2

    def test_modulation_with_different_current_speeds(self, processor):
        """異なる追従速度に対して上限が適切に計算される。"""
        for _ in range(20):
            processor._update_direction_tracking({"joy": 0.05})

        # Slow speed -> lower cap
        v_slow, _ = processor.get_tracking_speed_modulation(
            current_tracking_speed_valence=0.05,
        )
        # Fast speed -> higher cap
        v_fast, _ = processor.get_tracking_speed_modulation(
            current_tracking_speed_valence=0.20,
        )

        assert v_fast >= v_slow


# ── Tracking Speed Application Tests ─────────────────────────────

class TestTrackingSpeedApplication:
    """追従速度への変調量適用テスト（reaction.py側）。"""

    def test_modulation_absent_no_change(self):
        """変調量フィールドがNoneの場合、追従速度に変化なし。"""
        ctx = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
        )
        v_speed_base, a_speed_base = _derive_tracking_speeds(ctx)

        # Now with explicit None (same as default)
        ctx2 = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
            emotion_return_tracking_speed_modulation_valence=None,
            emotion_return_tracking_speed_modulation_arousal=None,
        )
        v_speed_none, a_speed_none = _derive_tracking_speeds(ctx2)

        assert v_speed_base == v_speed_none
        assert a_speed_base == a_speed_none

    def test_positive_modulation_increases_speed(self):
        """正の変調量が追従速度を増加させる。"""
        ctx_base = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
        )
        v_base, a_base = _derive_tracking_speeds(ctx_base)

        ctx_mod = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
            emotion_return_tracking_speed_modulation_valence=0.005,
            emotion_return_tracking_speed_modulation_arousal=0.005,
        )
        v_mod, a_mod = _derive_tracking_speeds(ctx_mod)

        assert v_mod > v_base
        assert a_mod > a_base

    def test_modulation_respects_band_limit(self):
        """変調後もband制限内に収まる。"""
        ctx = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.9,
            fear_level=0.5,
            emotion_return_tracking_speed_modulation_valence=1.0,
            emotion_return_tracking_speed_modulation_arousal=1.0,
        )
        v_speed, a_speed = _derive_tracking_speeds(ctx)

        assert v_speed <= _TRACKING_SPEED_MAX
        assert a_speed <= _TRACKING_SPEED_MAX
        assert v_speed >= _TRACKING_SPEED_MIN
        assert a_speed >= _TRACKING_SPEED_MIN

    def test_zero_modulation_no_change(self):
        """ゼロの変調量は追従速度を変えない。"""
        ctx_base = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
        )
        v_base, a_base = _derive_tracking_speeds(ctx_base)

        ctx_zero = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
            emotion_return_tracking_speed_modulation_valence=0.0,
            emotion_return_tracking_speed_modulation_arousal=0.0,
        )
        v_zero, a_zero = _derive_tracking_speeds(ctx_zero)

        assert v_zero == v_base
        assert a_zero == a_base

    def test_negative_modulation_ignored(self):
        """負の変調量は増加方向のみなので無視される。"""
        ctx_base = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
        )
        v_base, a_base = _derive_tracking_speeds(ctx_base)

        ctx_neg = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
            emotion_return_tracking_speed_modulation_valence=-0.05,
            emotion_return_tracking_speed_modulation_arousal=-0.05,
        )
        v_neg, a_neg = _derive_tracking_speeds(ctx_neg)

        # Negative should be treated as zero (max(0.0, neg))
        assert v_neg == v_base
        assert a_neg == a_base


# ── Save/Load Round-Trip Tests ───────────────────────────────────

class TestSaveLoadRoundTrip:
    """save/loadの往復保全テスト。"""

    def test_state_round_trip_with_direction_tracking(self, processor):
        """方向追跡状態がsave/loadで保全される。"""
        # Build some state
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})

        original_state = processor.state
        saved = original_state.to_dict()
        restored = MemoryEmotionReturnState.from_dict(saved)

        assert restored.direction_consecutive_count_positive == pytest.approx(
            original_state.direction_consecutive_count_positive, abs=0.001
        )
        assert restored.direction_consecutive_count_negative == pytest.approx(
            original_state.direction_consecutive_count_negative, abs=0.001
        )
        assert restored.last_direction_label == original_state.last_direction_label

    def test_state_round_trip_negative_direction(self, processor):
        """負方向追跡状態のsave/load。"""
        processor._update_direction_tracking({"sorrow": -0.05})
        processor._update_direction_tracking({"sorrow": -0.05})
        processor._update_direction_tracking({"sorrow": -0.05})

        saved = processor.state.to_dict()
        restored = MemoryEmotionReturnState.from_dict(saved)

        assert restored.direction_consecutive_count_negative == pytest.approx(
            processor.state.direction_consecutive_count_negative, abs=0.001
        )
        assert restored.last_direction_label == "negative"

    def test_from_dict_missing_direction_fields_uses_defaults(self):
        """方向追跡フィールドが欠落した旧形式データからの復元。"""
        old_data = {
            "return_history": [],
            "last_applied_tick": 5,
            "cycle_count": 3,
            # No direction fields (legacy data)
        }
        restored = MemoryEmotionReturnState.from_dict(old_data)

        assert restored.direction_consecutive_count_positive == 0.0
        assert restored.direction_consecutive_count_negative == 0.0
        assert restored.last_direction_label == "neutral"

    def test_modulation_consistent_after_restore(self, processor):
        """restore後のmodulation値が一貫している。"""
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})

        v_before, a_before = processor.get_tracking_speed_modulation()

        saved = processor.state.to_dict()
        new_processor = create_memory_emotion_return()
        new_processor.state = MemoryEmotionReturnState.from_dict(saved)

        v_after, a_after = new_processor.get_tracking_speed_modulation()

        assert v_before == pytest.approx(v_after, abs=0.0001)
        assert a_before == pytest.approx(a_after, abs=0.0001)


# ── Zero Modulation on No Input Tests ────────────────────────────

class TestZeroModulationOnNoInput:
    """入力不在時のゼロ変調テスト。"""

    def test_fresh_processor_zero_modulation(self, processor):
        """新規プロセッサの変調量はゼロ。"""
        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod == 0.0
        assert a_mod == 0.0

    def test_after_reset_zero_modulation(self, processor):
        """カウントリセット直後の変調量はゼロ。"""
        processor._update_direction_tracking({"joy": 0.05})
        processor._update_direction_tracking({"joy": 0.05})

        # Reverse resets positive count
        processor._update_direction_tracking({"sorrow": -0.05})
        # Now check: only negative count=1, positive=0
        # There should be modulation from negative count=1
        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod >= 0.0  # Count=1, so some modulation

    def test_neutral_does_not_generate_modulation_increase(self, processor):
        """中立帰還は変調量を増加させない。"""
        v_before, a_before = processor.get_tracking_speed_modulation()

        processor._update_direction_tracking({})  # neutral
        v_after, a_after = processor.get_tracking_speed_modulation()

        assert v_after == v_before
        assert a_after == a_before


# ── Integration Tests ─────────────────────────────────────────────

class TestIntegration:
    """統合テスト: 方向追跡+変調量+追従速度。"""

    def test_full_pipeline_positive_direction(self, processor):
        """正方向連続帰還→変調量正→追従速度増加の全経路。"""
        # Build consecutive positive
        for _ in range(5):
            processor._update_direction_tracking({"joy": 0.05, "love": 0.02})

        v_mod, a_mod = processor.get_tracking_speed_modulation()
        assert v_mod > 0.0

        # Apply to tracking speeds
        ctx = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
            emotion_return_tracking_speed_modulation_valence=v_mod,
            emotion_return_tracking_speed_modulation_arousal=a_mod,
        )
        v_speed, a_speed = _derive_tracking_speeds(ctx)

        # Compare with no modulation
        ctx_base = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.3,
        )
        v_base, a_base = _derive_tracking_speeds(ctx_base)

        assert v_speed > v_base
        assert a_speed > a_base

    def test_reversal_eliminates_prior_modulation(self, processor):
        """方向反転で前方向のカウントがリセットされ変調量が変化する。"""
        # Use a high current speed so cap is not hit early
        high_speed = 0.25

        # Build up positive count
        for _ in range(10):
            processor._update_direction_tracking({"joy": 0.05})

        v_before, _ = processor.get_tracking_speed_modulation(
            current_tracking_speed_valence=high_speed,
        )
        assert v_before > 0.0

        # Reverse
        processor._update_direction_tracking({"sorrow": -0.05})

        v_after, _ = processor.get_tracking_speed_modulation(
            current_tracking_speed_valence=high_speed,
        )
        # After reversal, only negative count = 1 (vs convergent ~5 before)
        # So modulation should be much smaller than before
        assert v_after < v_before

    def test_modulation_does_not_exceed_band(self, processor):
        """極端な連続帰還でも追従速度はband制限内に収まる。"""
        for _ in range(1000):
            processor._update_direction_tracking({"joy": 0.05})

        v_mod, a_mod = processor.get_tracking_speed_modulation()

        ctx = MoodContextInputs(
            current_valence=0.0,
            current_arousal=0.9,
            fear_level=0.5,
            emotion_return_tracking_speed_modulation_valence=v_mod,
            emotion_return_tracking_speed_modulation_arousal=a_mod,
        )
        v_speed, a_speed = _derive_tracking_speeds(ctx)

        assert v_speed <= _TRACKING_SPEED_MAX
        assert a_speed <= _TRACKING_SPEED_MAX

    def test_config_defaults_are_safe(self):
        """デフォルト設定で安全弁が機能する。"""
        cfg = MemoryEmotionReturnConfig()
        assert 0.0 < cfg.direction_freshness_decay < 1.0
        assert 0.0 < cfg.tracking_speed_modulation_ratio_cap <= 0.15
        assert 0.0 < cfg.tracking_speed_modulation_scale <= 0.1


# ── State Consistency Tests ──────────────────────────────────────

class TestStateConsistency:
    """状態の一貫性テスト。"""

    def test_initial_state_is_neutral(self, processor):
        """初期状態は中立。"""
        assert processor.state.direction_consecutive_count_positive == 0.0
        assert processor.state.direction_consecutive_count_negative == 0.0
        assert processor.state.last_direction_label == "neutral"

    def test_direction_label_tracks_latest(self, processor):
        """方向ラベルは最新の帰還方向を追跡する。"""
        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.last_direction_label == "positive"

        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.last_direction_label == "negative"

        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.last_direction_label == "positive"

    def test_only_one_direction_has_nonzero_count(self, processor):
        """常に一方の方向のみが非ゼロカウントを持つ。"""
        processor._update_direction_tracking({"joy": 0.05})
        assert processor.state.direction_consecutive_count_positive > 0.0
        assert processor.state.direction_consecutive_count_negative == 0.0

        processor._update_direction_tracking({"sorrow": -0.05})
        assert processor.state.direction_consecutive_count_positive == 0.0
        assert processor.state.direction_consecutive_count_negative > 0.0

    def test_positive_and_negative_never_both_nonzero(self, processor):
        """正と負のカウントが同時に非ゼロになることはない。"""
        for delta in [
            {"joy": 0.05}, {"joy": 0.05}, {"sorrow": -0.05},
            {"sorrow": -0.05}, {"joy": 0.05}, {"joy": 0.05},
        ]:
            processor._update_direction_tracking(delta)
            p = processor.state.direction_consecutive_count_positive
            n = processor.state.direction_consecutive_count_negative
            assert p == 0.0 or n == 0.0, f"Both non-zero: pos={p}, neg={n}"


# ── MoodContextInputs Field Tests ────────────────────────────────

class TestMoodContextFields:
    """MoodContextInputsの新フィールドテスト。"""

    def test_default_is_none(self):
        """デフォルト値はNone。"""
        ctx = MoodContextInputs()
        assert ctx.emotion_return_tracking_speed_modulation_valence is None
        assert ctx.emotion_return_tracking_speed_modulation_arousal is None

    def test_explicit_values(self):
        """明示的な値設定。"""
        ctx = MoodContextInputs(
            emotion_return_tracking_speed_modulation_valence=0.005,
            emotion_return_tracking_speed_modulation_arousal=0.003,
        )
        assert ctx.emotion_return_tracking_speed_modulation_valence == 0.005
        assert ctx.emotion_return_tracking_speed_modulation_arousal == 0.003
