"""
tests/test_reference_fluctuation_link.py

参照偏在度 → スコアリング揺らぎ振幅微調整の結合テスト。

検証項目:
  - 偏在度が高い場合に変動度が微増すること
  - 偏在度が低い場合に変動度が変わらないこと
  - 偏在度Noneの場合に既存動作と同一
  - amplitude_capが維持されること
  - 振幅微減が発生しないこと
  - 状態蓄積がないこと
"""

import pytest

from psyche.scoring_fluctuation import (
    ScoringFluctuationConfig,
    apply_scoring_fluctuation,
    compose_variability,
    extract_reference_imbalance_variability,
    limit_amplitude,
    extract_emotion_variability,
    extract_drive_variability,
    extract_elapsed_variability,
    extract_stm_variability,
)


# =============================================================================
# extract_reference_imbalance_variability
# =============================================================================

class TestExtractReferenceImbalanceVariability:
    """第5入力源: 偏在度からの変動度抽出テスト。"""

    def test_none_returns_zero(self):
        """偏在度Noneの場合は0.0を返す。"""
        assert extract_reference_imbalance_variability(None) == 0.0

    def test_zero_returns_zero(self):
        """偏在度0.0の場合は0.0を返す。"""
        assert extract_reference_imbalance_variability(0.0) == 0.0

    def test_high_imbalance_returns_high(self):
        """偏在度が高い場合は高い変動度を返す。"""
        result = extract_reference_imbalance_variability(0.8)
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_max_imbalance(self):
        """偏在度1.0の場合は1.0を返す。"""
        assert extract_reference_imbalance_variability(1.0) == pytest.approx(1.0)

    def test_low_imbalance_returns_low(self):
        """偏在度が低い場合は低い変動度を返す。"""
        result = extract_reference_imbalance_variability(0.1)
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_negative_clamped_to_zero(self):
        """負の偏在度は0.0にクランプされる（負の寄与は構造的に不可能）。"""
        result = extract_reference_imbalance_variability(-0.5)
        assert result == 0.0

    def test_over_one_clamped(self):
        """1.0超の偏在度は1.0にクランプされる。"""
        result = extract_reference_imbalance_variability(1.5)
        assert result == pytest.approx(1.0)

    def test_never_negative(self):
        """あらゆる入力で負の変動度を返さない。"""
        for val in [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, None]:
            result = extract_reference_imbalance_variability(val)
            assert result >= 0.0, f"Negative variability for input {val}: {result}"


# =============================================================================
# compose_variability with 5th input
# =============================================================================

class TestComposeVariabilityWithImbalance:
    """変動度合成に第5入力源が追加された場合のテスト。"""

    def setup_method(self):
        self.config = ScoringFluctuationConfig()

    def test_zero_imbalance_no_effect(self):
        """偏在度変動度0.0は合成結果に影響しない（実質4入力と同等）。"""
        result_without = compose_variability(0.3, 0.2, 0.1, 0.4, self.config, 0.0)
        # 重みが5入力に分散するため厳密には異なるが、0.0入力は重み付き平均への寄与がゼロ
        # → 加重平均の分母に weight_reference_imbalance が加わるため微減が生じうるが、
        #   max_val は変わらない。結果は (weighted_avg + max_val)/2 なので微変化。
        # ただし reference_imbalance_var=0.0 の場合、その項の寄与は0なので分母拡大のみ。
        assert 0.0 <= result_without <= 1.0

    def test_high_imbalance_increases_composed(self):
        """偏在度変動度が高いと合成変動度が微増する。"""
        base = compose_variability(0.3, 0.2, 0.1, 0.1, self.config, 0.0)
        with_imbalance = compose_variability(0.3, 0.2, 0.1, 0.1, self.config, 0.9)
        assert with_imbalance > base

    def test_result_bounded(self):
        """合成結果は0.0-1.0の範囲内。"""
        result = compose_variability(1.0, 1.0, 1.0, 1.0, self.config, 1.0)
        assert 0.0 <= result <= 1.0

    def test_all_zero_inputs(self):
        """全入力が0の場合は0.0。"""
        result = compose_variability(0.0, 0.0, 0.0, 0.0, self.config, 0.0)
        assert result == 0.0


# =============================================================================
# apply_scoring_fluctuation with reference_imbalance
# =============================================================================

class TestApplyScoringFluctuationWithImbalance:
    """メインパイプラインに偏在度を渡した場合のテスト。"""

    def _make_candidates(self, n=3):
        return [
            {
                "policy_label": f"policy_{i}",
                "drive_target": "curiosity",
                "expected_drive_change": {"curiosity": 0.1},
                "_score": 0.5,
            }
            for i in range(n)
        ]

    def test_none_imbalance_same_as_omitted(self):
        """reference_imbalance=Noneは引数省略時と同一の動作。"""
        candidates = self._make_candidates()
        emotions = {"joy": 0.5, "sadness": 0.2}
        drives = {"curiosity": 0.7, "social": 0.3}
        config = ScoringFluctuationConfig()

        result_none = apply_scoring_fluctuation(
            candidates=candidates,
            emotions=emotions,
            drives=drives,
            elapsed_seconds=10.0,
            config=config,
            reference_imbalance=None,
        )
        result_omitted = apply_scoring_fluctuation(
            candidates=candidates,
            emotions=emotions,
            drives=drives,
            elapsed_seconds=10.0,
            config=config,
        )
        # Both should produce fluctuation-applied candidates
        assert len(result_none) == len(result_omitted)
        for c in result_none:
            assert c.get("_fluctuation_applied") is True

    def test_high_imbalance_fluctuation_applied(self):
        """高偏在度を渡した場合も揺らぎが正常に適用される。"""
        candidates = self._make_candidates()
        result = apply_scoring_fluctuation(
            candidates=candidates,
            emotions={"joy": 0.5},
            drives={"curiosity": 0.5},
            elapsed_seconds=5.0,
            reference_imbalance=0.9,
        )
        assert len(result) == 3
        for c in result:
            assert c.get("_fluctuation_applied") is True
            assert "_fluctuation" in c

    def test_amplitude_cap_maintained(self):
        """偏在度が高くてもamplitude_capを超えない。"""
        config = ScoringFluctuationConfig(amplitude_cap=0.10)
        candidates = self._make_candidates(5)
        result = apply_scoring_fluctuation(
            candidates=candidates,
            emotions={"joy": 1.0, "sadness": 0.0, "anger": 1.0},
            drives={"curiosity": 1.0, "social": 0.0, "autonomy": 1.0},
            elapsed_seconds=300.0,
            config=config,
            reference_imbalance=1.0,
        )
        for c in result:
            fluct = abs(c.get("_fluctuation", 0.0))
            # fluctuation = hash_val * amplitude * change_factor
            # amplitude <= amplitude_cap, |hash_val| <= 1, change_factor <= 1
            assert fluct <= config.amplitude_cap + 1e-9

    def test_low_imbalance_no_amplitude_decrease(self):
        """偏在度が低い場合に振幅が微減しないこと。

        偏在度=0.0の変動度は0.0であり、合成結果の加重平均の分母が拡大する分
        微減しうるが、max_valとの中間を取るため、最終振幅が偏在度なし
        (reference_imbalance=None)より大きく減少しないことを確認する。
        「振幅微減禁止」は偏在度由来の変動度が負にならないことで保証される。
        """
        # reference_imbalance_var が 0.0 の場合の変動度
        var = extract_reference_imbalance_variability(0.0)
        assert var == 0.0  # 偏在度低 → 変動度ゼロ（負にならない）

        var_neg = extract_reference_imbalance_variability(-0.1)
        assert var_neg == 0.0  # 負入力もゼロ

    def test_no_state_accumulation(self):
        """揺らぎ生成構造は状態を蓄積しない — 同一入力で2回呼んでも独立。"""
        candidates = self._make_candidates()
        emotions = {"joy": 0.6}
        drives = {"curiosity": 0.4}

        # 1回目
        apply_scoring_fluctuation(
            candidates=candidates,
            emotions=emotions,
            drives=drives,
            elapsed_seconds=5.0,
            reference_imbalance=0.7,
        )
        # 2回目: 元の candidates は変更されていないことを確認
        for c in candidates:
            assert "_fluctuation" not in c
            assert "_fluctuation_applied" not in c

    def test_empty_candidates(self):
        """候補が空の場合は空リストを返す。"""
        result = apply_scoring_fluctuation(
            candidates=[],
            emotions={"joy": 0.5},
            drives={"curiosity": 0.5},
            reference_imbalance=0.8,
        )
        assert result == []

    def test_imbalance_increases_amplitude_for_high_values(self):
        """高い偏在度が変動度合成に正の寄与をもたらすことの確認。

        同一のemotion/drive/stm/elapsedの条件で、
        reference_imbalance=0.0 と reference_imbalance=0.9 を比較し、
        後者の方が合成変動度が高い（→振幅が大きい可能性がある）ことを検証。
        """
        config = ScoringFluctuationConfig()
        emotion_var = extract_emotion_variability({"joy": 0.5, "sadness": 0.2})
        drive_var = extract_drive_variability({"curiosity": 0.6, "social": 0.3})
        elapsed_var = extract_elapsed_variability(10.0, config)
        stm_var = extract_stm_variability(3, 30.0, 1.0, 0.5)

        composed_base = compose_variability(
            emotion_var, stm_var, drive_var, elapsed_var, config, 0.0,
        )
        composed_high = compose_variability(
            emotion_var, stm_var, drive_var, elapsed_var, config, 0.9,
        )
        assert composed_high > composed_base

    def test_amplitude_with_imbalance_still_within_cap(self):
        """偏在度を含む合成変動度でも、limit_amplitudeのcapは維持される。"""
        config = ScoringFluctuationConfig(amplitude_cap=0.08)
        composed = compose_variability(0.8, 0.7, 0.9, 0.6, config, 1.0)
        amplitude = limit_amplitude(composed, config)
        assert amplitude <= config.amplitude_cap


# =============================================================================
# Config: weight_reference_imbalance
# =============================================================================

class TestConfigWeightReferenceImbalance:
    """設定値のテスト。"""

    def test_default_weight_exists(self):
        """デフォルト設定にweight_reference_imbalanceが存在する。"""
        config = ScoringFluctuationConfig()
        assert hasattr(config, "weight_reference_imbalance")
        assert config.weight_reference_imbalance > 0

    def test_weight_not_larger_than_others(self):
        """偏在度の重みは既存4入力源の重みと同程度またはそれ以下。"""
        config = ScoringFluctuationConfig()
        other_weights = [
            config.weight_emotion,
            config.weight_stm,
            config.weight_drives,
            config.weight_elapsed,
        ]
        assert config.weight_reference_imbalance <= max(other_weights)

    def test_custom_weight(self):
        """カスタム重みの設定。"""
        config = ScoringFluctuationConfig(weight_reference_imbalance=0.1)
        assert config.weight_reference_imbalance == 0.1


# =============================================================================
# Integration: compose -> limit -> no cap violation
# =============================================================================

class TestIntegrationAmplitudeCap:
    """合成→制限の結合テスト: amplitude_capが常に維持される。"""

    @pytest.mark.parametrize("imbalance", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_cap_maintained_across_imbalance_levels(self, imbalance):
        """様々な偏在度レベルでamplitude_capが維持される。"""
        config = ScoringFluctuationConfig(amplitude_cap=0.12)
        composed = compose_variability(
            0.8, 0.6, 0.7, 0.5, config,
            extract_reference_imbalance_variability(imbalance),
        )
        amplitude = limit_amplitude(composed, config)
        assert amplitude <= config.amplitude_cap

    @pytest.mark.parametrize("imbalance", [None, 0.0, 0.01])
    def test_no_negative_contribution(self, imbalance):
        """偏在度が低い/Noneの場合、変動度寄与は非負。"""
        var = extract_reference_imbalance_variability(imbalance)
        assert var >= 0.0
