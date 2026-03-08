"""
Tests for psyche/decay_rate_modulation.py

感情蓄積パターンによる感情減衰速度の変調のテスト。
"""

import pytest
from psyche.decay_rate_modulation import (
    compute_emotion_amplitude_score,
    compute_decay_rate_modulation,
    apply_decay_rate_modulation,
    _MODULATION_BAND_RATIO,
    _REGRESSION_PRESSURE_FACTOR,
    _AMPLITUDE_NEUTRAL_POINT,
    _clamp,
)


# =============================================================================
# Helper: composition record builders
# =============================================================================

def _make_record(
    emotion_series: dict[str, list[float]] | None = None,
    valence_series: list[float] | None = None,
    freshness: float = 1.0,
) -> dict:
    """テスト用の構成記述レコードを生成する。"""
    return {
        "record_id": "test",
        "tick": 10,
        "timestamp": 0.0,
        "window_size": 10,
        "tick_range": 10,
        "time_range": 10.0,
        "emotion_series": emotion_series or {},
        "valence_series": valence_series or [],
        "arousal_series": [],
        "phase_series": [],
        "low_variability_noted": False,
        "freshness": freshness,
        "freshness_stage": "active" if freshness >= 0.8 else "weakening",
    }


def _make_high_variance_records(n: int = 5) -> list[dict]:
    """高い感情変動を持つレコード群を生成する。"""
    records = []
    for i in range(n):
        # 大きな変動: 0.0 - 1.0 を交互に
        vals = [0.0, 1.0, 0.0, 1.0, 0.0] if i % 2 == 0 else [1.0, 0.0, 1.0, 0.0, 1.0]
        records.append(_make_record(
            emotion_series={"joy": vals, "sorrow": vals},
            valence_series=vals,
            freshness=0.9,
        ))
    return records


def _make_low_variance_records(n: int = 5) -> list[dict]:
    """低い感情変動を持つレコード群を生成する。"""
    records = []
    for i in range(n):
        # ほぼ一定の値
        vals = [0.5, 0.5, 0.5, 0.5, 0.5]
        records.append(_make_record(
            emotion_series={"joy": vals, "sorrow": vals},
            valence_series=vals,
            freshness=0.9,
        ))
    return records


def _make_medium_variance_records(n: int = 5) -> list[dict]:
    """中程度の感情変動を持つレコード群を生成する。"""
    records = []
    for i in range(n):
        # 中程度の変動
        vals = [0.3, 0.5, 0.4, 0.6, 0.35]
        records.append(_make_record(
            emotion_series={"joy": vals, "fear": vals},
            valence_series=vals,
            freshness=0.9,
        ))
    return records


# =============================================================================
# Tests: compute_emotion_amplitude_score
# =============================================================================

class TestComputeEmotionAmplitudeScore:
    """感情変動振幅スコアの算出テスト。"""

    def test_empty_records_returns_none(self):
        """空のレコードリストはNoneを返す。"""
        assert compute_emotion_amplitude_score([]) is None

    def test_single_record_returns_none(self):
        """1件のみの場合はNoneを返す（visible >= 2 が必要）。"""
        records = [_make_record(freshness=0.9)]
        assert compute_emotion_amplitude_score(records) is None

    def test_low_freshness_records_filtered(self):
        """鮮度が低いレコードは除外される。"""
        records = [
            _make_record(freshness=0.1),  # invisible
            _make_record(freshness=0.2),  # near_invisible
            _make_record(freshness=0.3),  # fading (below 0.6 threshold)
        ]
        assert compute_emotion_amplitude_score(records) is None

    def test_high_variance_returns_high_score(self):
        """高い感情変動は高いスコアを返す。"""
        records = _make_high_variance_records(3)
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score > 0.5  # 高い変動 → 高いスコア

    def test_low_variance_returns_low_score(self):
        """低い感情変動は低いスコアを返す。"""
        records = _make_low_variance_records(3)
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score < 0.1  # 低い変動 → 低いスコア

    def test_medium_variance_returns_medium_score(self):
        """中程度の感情変動は中程度のスコアを返す。"""
        records = _make_medium_variance_records(3)
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert 0.0 < score < 1.0

    def test_score_range_0_to_1(self):
        """スコアは 0.0-1.0 の範囲内。"""
        for records in [
            _make_high_variance_records(5),
            _make_low_variance_records(5),
            _make_medium_variance_records(5),
        ]:
            score = compute_emotion_amplitude_score(records)
            assert score is not None
            assert 0.0 <= score <= 1.0

    def test_no_emotion_series_returns_none_if_only_short_series(self):
        """感情次元のvaluesが短すぎる場合はNoneを返す（最低3値必要）。"""
        records = [
            _make_record(
                emotion_series={"joy": [0.5, 0.5]},  # Only 2 values
                valence_series=[0.5, 0.5],  # Only 2 values
                freshness=0.9,
            )
            for _ in range(3)
        ]
        assert compute_emotion_amplitude_score(records) is None

    def test_valence_series_contributes(self):
        """valence_seriesも振幅算出に寄与する。"""
        # emotion_seriesなし、valence_seriesのみ
        records = [
            _make_record(
                emotion_series={},
                valence_series=[0.0, 1.0, 0.0, 1.0, 0.0],
                freshness=0.9,
            )
            for _ in range(3)
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score > 0.0

    def test_mixed_freshness(self):
        """高鮮度と低鮮度が混在する場合、高鮮度のみが使用される。"""
        records = [
            _make_record(
                emotion_series={"joy": [0.0, 1.0, 0.0, 1.0, 0.0]},
                freshness=0.9,  # active (included)
            ),
            _make_record(
                emotion_series={"joy": [0.0, 1.0, 0.0, 1.0, 0.0]},
                freshness=0.7,  # weakening (included)
            ),
            _make_record(
                emotion_series={"joy": [0.5, 0.5, 0.5, 0.5, 0.5]},
                freshness=0.3,  # fading (excluded)
            ),
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        # Only the first 2 (high variance) records contribute
        assert score > 0.5


# =============================================================================
# Tests: compute_decay_rate_modulation
# =============================================================================

class TestComputeDecayRateModulation:
    """減衰速度変調の算出テスト。"""

    BASE_RATE = 0.95

    def test_none_amplitude_returns_base(self):
        """amplitude_scoreがNoneの場合は基準値を返す。"""
        result = compute_decay_rate_modulation(self.BASE_RATE, None, None)
        assert result == self.BASE_RATE

    def test_neutral_amplitude_no_previous(self):
        """中立振幅・前回値なしの場合は基準値に近い。"""
        result = compute_decay_rate_modulation(
            self.BASE_RATE, _AMPLITUDE_NEUTRAL_POINT, None
        )
        # 中立点 → 変調なし
        assert abs(result - self.BASE_RATE) < 1e-10

    def test_high_amplitude_decreases_rate(self):
        """高い振幅は減衰速度を下げる（減衰が速くなる方向）。"""
        result = compute_decay_rate_modulation(self.BASE_RATE, 1.0, None)
        assert result < self.BASE_RATE

    def test_low_amplitude_increases_rate(self):
        """低い振幅は減衰速度を上げる（減衰が遅くなる方向）。"""
        result = compute_decay_rate_modulation(self.BASE_RATE, 0.0, None)
        assert result > self.BASE_RATE

    def test_band_limit_upper(self):
        """変調後の値は基準値 + 5% を超えない。"""
        max_rate = self.BASE_RATE * (1.0 + _MODULATION_BAND_RATIO)
        result = compute_decay_rate_modulation(self.BASE_RATE, 0.0, None)
        assert result <= max_rate + 1e-10

    def test_band_limit_lower(self):
        """変調後の値は基準値 - 5% を下回らない。"""
        min_rate = self.BASE_RATE * (1.0 - _MODULATION_BAND_RATIO)
        result = compute_decay_rate_modulation(self.BASE_RATE, 1.0, None)
        assert result >= min_rate - 1e-10

    def test_regression_pressure_toward_base(self):
        """前回の値が基準値から離れていると、基準値方向に引き戻される。"""
        # 前回は基準値より高かった
        prev_high = self.BASE_RATE + 0.04  # 基準値より高い
        result = compute_decay_rate_modulation(
            self.BASE_RATE, _AMPLITUDE_NEUTRAL_POINT, prev_high
        )
        # 回帰圧力で基準値に近づく
        assert result < prev_high

        # 前回は基準値より低かった
        prev_low = self.BASE_RATE - 0.04  # 基準値より低い
        result = compute_decay_rate_modulation(
            self.BASE_RATE, _AMPLITUDE_NEUTRAL_POINT, prev_low
        )
        # 回帰圧力で基準値に近づく
        assert result > prev_low

    def test_regression_pressure_with_amplitude(self):
        """回帰圧力と振幅変調が共存する場合。"""
        # 前回高く、現在高振幅（低下方向）→ 両方が低下方向
        prev_high = self.BASE_RATE + 0.03
        result = compute_decay_rate_modulation(self.BASE_RATE, 0.9, prev_high)
        # 基準値以下になりうる（振幅と回帰圧力の両方が低下方向）
        assert result < prev_high

    def test_band_limit_with_regression(self):
        """回帰圧力を含む場合でも帯域上限内。"""
        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        max_rate = self.BASE_RATE + band
        min_rate = self.BASE_RATE - band

        # 極端な前回値
        prev_extreme = self.BASE_RATE + 0.1  # 帯域外（異常値）
        result = compute_decay_rate_modulation(self.BASE_RATE, 0.0, prev_extreme)
        assert min_rate - 1e-10 <= result <= max_rate + 1e-10

    def test_symmetric_modulation(self):
        """中立点を境に対称的な変調。"""
        high = compute_decay_rate_modulation(self.BASE_RATE, 0.8, None)
        low = compute_decay_rate_modulation(self.BASE_RATE, 0.2, None)
        # high amplitude → 低い rate、low amplitude → 高い rate
        assert high < self.BASE_RATE
        assert low > self.BASE_RATE
        # 中立点からの距離が等しいので、偏差の絶対値も等しい
        assert abs(abs(high - self.BASE_RATE) - abs(low - self.BASE_RATE)) < 1e-10

    def test_multiple_sessions_with_regression(self):
        """複数セッションにわたる回帰圧力のシミュレーション。"""
        prev = None
        rates = []
        for _i in range(10):
            rate = compute_decay_rate_modulation(
                self.BASE_RATE, 0.8, prev  # 常に高振幅
            )
            rates.append(rate)
            prev = rate

        # 回帰圧力があるため、値は帯域下限に漸近する
        # しかし帯域内に留まる
        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        for r in rates:
            assert self.BASE_RATE - band - 1e-10 <= r <= self.BASE_RATE + band + 1e-10

        # 回帰圧力により、完全に帯域下限に到達しない
        # (振幅が高い方向の変調と回帰圧力が均衡する)
        assert rates[-1] < self.BASE_RATE  # 基準値より低い

    def test_alternating_amplitude_oscillation(self):
        """振幅が交互に変わる場合の振動挙動。"""
        prev = None
        rates = []
        for i in range(6):
            amp = 1.0 if i % 2 == 0 else 0.0  # 高低交互
            rate = compute_decay_rate_modulation(self.BASE_RATE, amp, prev)
            rates.append(rate)
            prev = rate

        # 振動するが帯域内に留まる
        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        for r in rates:
            assert self.BASE_RATE - band - 1e-10 <= r <= self.BASE_RATE + band + 1e-10


# =============================================================================
# Tests: apply_decay_rate_modulation
# =============================================================================

class TestApplyDecayRateModulation:
    """統合的な変調適用テスト。"""

    BASE_RATE = 0.95

    def test_empty_backdrop_state(self):
        """空のバックドロップ状態では変調なし。"""
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert rate == self.BASE_RATE
        assert score is None

    def test_empty_composition_records(self):
        """構成記述が空の場合は変調なし。"""
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": []},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert rate == self.BASE_RATE
        assert score is None

    def test_high_variance_backdrop(self):
        """高い感情変動を持つバックドロップ状態での変調。"""
        records = _make_high_variance_records(3)
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert score is not None
        assert score > 0.5
        assert rate < self.BASE_RATE  # 高振幅 → 減衰速く

    def test_low_variance_backdrop(self):
        """低い感情変動を持つバックドロップ状態での変調。"""
        records = _make_low_variance_records(3)
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert score is not None
        assert score < 0.1
        assert rate > self.BASE_RATE  # 低振幅 → 減衰遅く

    def test_with_previous_modulated_rate(self):
        """前回の変調後の値がある場合、回帰圧力が作用する。"""
        records = _make_medium_variance_records(3)
        prev_rate = self.BASE_RATE + 0.03  # 前回は高め

        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=prev_rate,
        )
        assert score is not None
        # 回帰圧力で前回値より基準値に近い
        assert abs(rate - self.BASE_RATE) <= abs(prev_rate - self.BASE_RATE) + 1e-10

    def test_band_limit_enforced(self):
        """帯域上限が適用される。"""
        records = _make_high_variance_records(5)
        band = self.BASE_RATE * _MODULATION_BAND_RATIO

        rate, _score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert rate >= self.BASE_RATE - band - 1e-10
        assert rate <= self.BASE_RATE + band + 1e-10

    def test_returns_tuple(self):
        """返り値がタプル (rate, score) であること。"""
        result = apply_decay_rate_modulation(
            backdrop_state_dict={},
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_full_backdrop_state_dict(self):
        """BackdropState.to_dict()形式の完全な辞書で動作する。"""
        from psyche.emotional_backdrop_cognition import (
            BackdropState,
            BackdropConfig,
            EmotionalBackdropProcessor,
            BackdropInputs,
        )
        # プロセッサを作成して数回tickさせる
        processor = EmotionalBackdropProcessor()
        for i in range(10):
            inputs = BackdropInputs(
                emotion_values={"joy": 0.3 + 0.1 * (i % 3), "sorrow": 0.2},
                mood_valence=0.5 + 0.1 * (i % 2),
                mood_arousal=0.4,
                current_tick=i,
            )
            processor.tick(inputs)

        state_dict = processor.save()
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict=state_dict,
            base_decay_rate=self.BASE_RATE,
            previous_modulated_rate=None,
        )
        # Should not crash and return valid values
        assert isinstance(rate, float)
        assert self.BASE_RATE * (1 - _MODULATION_BAND_RATIO) - 1e-10 <= rate
        assert rate <= self.BASE_RATE * (1 + _MODULATION_BAND_RATIO) + 1e-10


# =============================================================================
# Tests: Safety valve verification
# =============================================================================

class TestSafetyValves:
    """安全弁のテスト。"""

    BASE_RATE = 0.95

    def test_safety_valve_1_band_limit(self):
        """安全弁1: 帯域上限 (+-5%) が厳密に守られる。"""
        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        min_rate = self.BASE_RATE - band
        max_rate = self.BASE_RATE + band

        # 極端な振幅値でテスト
        for amp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for prev in [None, self.BASE_RATE - 0.1, self.BASE_RATE + 0.1]:
                rate = compute_decay_rate_modulation(
                    self.BASE_RATE, amp, prev
                )
                assert min_rate - 1e-10 <= rate <= max_rate + 1e-10, (
                    f"Band limit violated: amp={amp}, prev={prev}, "
                    f"rate={rate}, band=[{min_rate}, {max_rate}]"
                )

    def test_safety_valve_2_regression_pressure(self):
        """安全弁2: 基準値回帰圧力が作用する。"""
        # 中立振幅で、前回値が基準値から離れている場合、
        # 回帰圧力により基準値に近づく
        prev_high = self.BASE_RATE + 0.04
        rate = compute_decay_rate_modulation(
            self.BASE_RATE, _AMPLITUDE_NEUTRAL_POINT, prev_high
        )
        assert abs(rate - self.BASE_RATE) < abs(prev_high - self.BASE_RATE)

    def test_safety_valve_3_startup_only(self):
        """安全弁3: 関数は起動時のみの呼び出しを想定（構造的制約）。

        本テストは、関数が冪等であり、同じ入力に対して同じ出力を返すことを確認する。
        """
        records = _make_medium_variance_records(3)
        state_dict = {"composition_records": records}

        rate1, score1 = apply_decay_rate_modulation(
            state_dict, self.BASE_RATE, None
        )
        rate2, score2 = apply_decay_rate_modulation(
            state_dict, self.BASE_RATE, None
        )
        assert rate1 == rate2
        assert score1 == score2

    def test_safety_valve_4_read_only_reference(self):
        """安全弁4: 蓄積データへの書き込みがないことを確認。"""
        import copy
        records = _make_high_variance_records(3)
        state_dict = {"composition_records": copy.deepcopy(records)}
        original_records = copy.deepcopy(records)

        apply_decay_rate_modulation(
            state_dict, self.BASE_RATE, None
        )

        # 元のレコードが変更されていないことを確認
        for i, rec in enumerate(state_dict["composition_records"]):
            assert rec == original_records[i]

    def test_no_single_direction_drift(self):
        """単方向ドリフトの防止: 同一振幅を繰り返しても帯域内で収束する。"""
        prev = None
        for _ in range(50):
            rate = compute_decay_rate_modulation(self.BASE_RATE, 0.9, prev)
            prev = rate

        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        assert rate >= self.BASE_RATE - band - 1e-10
        assert rate <= self.BASE_RATE + band + 1e-10


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestEdgeCases:
    """エッジケースのテスト。"""

    BASE_RATE = 0.95

    def test_all_zero_emotions(self):
        """全感情が0の場合。"""
        records = [
            _make_record(
                emotion_series={"joy": [0.0, 0.0, 0.0, 0.0, 0.0]},
                valence_series=[0.0, 0.0, 0.0, 0.0, 0.0],
                freshness=0.9,
            )
            for _ in range(3)
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score == 0.0  # 変動なし

    def test_all_max_emotions(self):
        """全感情が1.0の場合。"""
        records = [
            _make_record(
                emotion_series={"joy": [1.0, 1.0, 1.0, 1.0, 1.0]},
                valence_series=[1.0, 1.0, 1.0, 1.0, 1.0],
                freshness=0.9,
            )
            for _ in range(3)
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score == 0.0  # 変動なし

    def test_single_emotion_dimension(self):
        """1つの感情次元のみの場合。"""
        records = [
            _make_record(
                emotion_series={"joy": [0.1, 0.9, 0.2, 0.8, 0.3]},
                freshness=0.9,
            )
            for _ in range(3)
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert score > 0.0

    def test_many_emotion_dimensions(self):
        """多数の感情次元がある場合。"""
        records = [
            _make_record(
                emotion_series={
                    f"emotion_{j}": [0.1 * ((i + j) % 10) for i in range(5)]
                    for j in range(7)
                },
                freshness=0.9,
            )
            for _ in range(3)
        ]
        score = compute_emotion_amplitude_score(records)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_base_rate_zero(self):
        """基準値が0の場合でもクラッシュしない。"""
        rate = compute_decay_rate_modulation(0.0, 0.5, None)
        assert rate == 0.0

    def test_base_rate_one(self):
        """基準値が1の場合でもクラッシュしない。"""
        rate = compute_decay_rate_modulation(1.0, 0.5, None)
        assert rate == 1.0

    def test_previous_rate_far_from_base(self):
        """前回の値が基準値から大きく離れている場合でも帯域内。"""
        rate = compute_decay_rate_modulation(
            self.BASE_RATE, 0.5, self.BASE_RATE + 1.0
        )
        band = self.BASE_RATE * _MODULATION_BAND_RATIO
        assert self.BASE_RATE - band - 1e-10 <= rate <= self.BASE_RATE + band + 1e-10


# =============================================================================
# Tests: Orchestrator integration (mock)
# =============================================================================

class TestOrchestratorIntegration:
    """オーケストレータ統合のテスト（モック使用）。"""

    def test_persistence_round_trip(self):
        """変調後の値がsave/loadで往復できる。"""
        # 変調を実行
        records = _make_high_variance_records(3)
        rate, score = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=0.95,
            previous_modulated_rate=None,
        )

        # 値が永続化可能な型であること
        assert isinstance(rate, float)
        assert rate != 0.95  # 変調が適用されている

        # 永続化→復元のシミュレーション
        saved_data = {"decay_rate_modulated_value": rate}
        restored = saved_data["decay_rate_modulated_value"]
        assert isinstance(restored, float)
        assert restored == rate

    def test_second_session_with_previous_rate(self):
        """2回目のセッションで前回の値が回帰圧力に使用される。"""
        records = _make_high_variance_records(3)

        # 1st session
        rate1, _ = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=0.95,
            previous_modulated_rate=None,
        )

        # 2nd session (same data but with previous rate)
        rate2, _ = apply_decay_rate_modulation(
            backdrop_state_dict={"composition_records": records},
            base_decay_rate=0.95,
            previous_modulated_rate=rate1,
        )

        # Both should be within band
        band = 0.95 * _MODULATION_BAND_RATIO
        assert 0.95 - band - 1e-10 <= rate1 <= 0.95 + band + 1e-10
        assert 0.95 - band - 1e-10 <= rate2 <= 0.95 + band + 1e-10

    def test_reaction_decay_rate_type(self):
        """reaction.DECAY_RATE が float であり、上書き可能であること。"""
        from psyche import reaction
        original = reaction.DECAY_RATE
        assert isinstance(original, float)

        # 書き換えテスト
        reaction.DECAY_RATE = 0.94
        assert reaction.DECAY_RATE == 0.94

        # 元に戻す
        reaction.DECAY_RATE = original
        assert reaction.DECAY_RATE == original
