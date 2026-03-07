"""
tests/test_contradiction_amplitude_modulation.py - 内部矛盾蓄積→振幅上限値変調のテスト

テスト対象: psyche/scoring_fluctuation.py の変調機能
設計書: design_contradiction_amplitude_modulation.md

テスト項目:
- 変調量の算出（段階値テーブル、帯域上限、ゼロ入力）
- 変調の適用（configへの反映、絶対上限、下限維持）
- 安全弁6: 帯域制限（初期値の±10%以内）
- 安全弁7: 絶対上限（vo_max_bias_strength未満）
- 安全弁8: 矛盾記述構造への非介入（READ-ONLY）
- 既存C9-5参照偏在変調との独立性
- セッション起動時一回のみの特性
- エッジケース
"""

import pytest

from psyche.scoring_fluctuation import (
    ScoringFluctuationConfig,
    compute_contradiction_modulation,
    apply_contradiction_amplitude_modulation,
    apply_scoring_fluctuation,
    _CONTRADICTION_MODULATION_STEPS,
    _CONTRADICTION_MODULATION_BAND_LIMIT,
)
from psyche.internal_contradiction_description import (
    InternalContradictionProcessor,
    ContradictionConfig,
    ContradictionInputs,
    ContradictionRecord,
    ContradictionState,
    create_contradiction_processor,
)
from psyche import coefficient_registry


# =============================================================================
# Helper
# =============================================================================

def _make_config(
    amplitude_cap: float = 0.12,
    amplitude_floor: float = 0.005,
    vo_max_bias_strength: float = 0.15,
) -> ScoringFluctuationConfig:
    return ScoringFluctuationConfig(
        amplitude_cap=amplitude_cap,
        amplitude_floor=amplitude_floor,
        vo_max_bias_strength=vo_max_bias_strength,
    )


def _make_contradiction_state_with_count(count: int) -> ContradictionState:
    """指定件数の矛盾対レコードを持つContradictionStateを作成する。"""
    records = []
    for i in range(count):
        records.append(ContradictionRecord(
            pair_name=f"pair_{i % 6}",
            section_a=f"section_a_{i}",
            section_b=f"section_b_{i}",
            direction_a=f"dir_a={0.1 * i:.2f}",
            direction_b=f"dir_b={0.9 - 0.1 * i:.2f}",
            tick=i,
            freshness=max(0.2, 1.0 - 0.02 * i),
        ))
    state = ContradictionState()
    state.contradiction_window = records
    return state


# =============================================================================
# Tests: compute_contradiction_modulation
# =============================================================================

class TestComputeContradictionModulation:
    """変調量の算出テスト。"""

    def test_zero_count_returns_zero(self):
        """蓄積件数ゼロでは変調量ゼロ。"""
        result = compute_contradiction_modulation(0, 0.12)
        assert result == 0.0

    def test_negative_count_returns_zero(self):
        """蓄積件数が負の場合は変調量ゼロ。"""
        result = compute_contradiction_modulation(-5, 0.12)
        assert result == 0.0

    def test_zero_base_cap_returns_zero(self):
        """基準振幅上限値がゼロの場合は変調量ゼロ。"""
        result = compute_contradiction_modulation(30, 0.0)
        assert result == 0.0

    def test_negative_base_cap_returns_zero(self):
        result = compute_contradiction_modulation(30, -0.1)
        assert result == 0.0

    def test_below_first_threshold_returns_zero(self):
        """最低閾値未満では変調量ゼロ。"""
        first_threshold = _CONTRADICTION_MODULATION_STEPS[0][0]
        result = compute_contradiction_modulation(first_threshold - 1, 0.12)
        assert result == 0.0

    def test_at_first_threshold_returns_nonzero(self):
        """最低閾値に達すると変調量が非ゼロ。"""
        first_threshold, first_ratio = _CONTRADICTION_MODULATION_STEPS[0]
        result = compute_contradiction_modulation(first_threshold, 0.12)
        expected = 0.12 * first_ratio
        assert result == pytest.approx(expected, abs=1e-8)

    def test_monotonic_increase_with_count(self):
        """蓄積件数が増えると変調量が増える（単調非減少）。"""
        base_cap = 0.12
        prev = 0.0
        for count in range(0, 60, 5):
            current = compute_contradiction_modulation(count, base_cap)
            assert current >= prev
            prev = current

    def test_capped_at_band_limit(self):
        """帯域上限（初期値の10%）で頭打ち。"""
        base_cap = 0.12
        max_expected = base_cap * _CONTRADICTION_MODULATION_BAND_LIMIT
        # 非常に大きな件数でも帯域上限を超えない
        result = compute_contradiction_modulation(1000, base_cap)
        assert result <= max_expected + 1e-10

    def test_deterministic(self):
        """同一入力に対し同一出力（決定論的）。"""
        r1 = compute_contradiction_modulation(30, 0.12)
        r2 = compute_contradiction_modulation(30, 0.12)
        assert r1 == r2

    def test_each_threshold_step(self):
        """各段階値閾値で期待される変調量を確認。"""
        base_cap = 0.12
        for threshold, ratio in _CONTRADICTION_MODULATION_STEPS:
            result = compute_contradiction_modulation(threshold, base_cap)
            expected = base_cap * ratio
            assert result == pytest.approx(expected, abs=1e-8), \
                f"At threshold {threshold}: expected {expected}, got {result}"

    def test_between_thresholds_uses_lower(self):
        """閾値間の件数は直前の閾値の変調量を使用。"""
        base_cap = 0.12
        if len(_CONTRADICTION_MODULATION_STEPS) >= 2:
            t1, r1 = _CONTRADICTION_MODULATION_STEPS[0]
            t2, _ = _CONTRADICTION_MODULATION_STEPS[1]
            mid_count = (t1 + t2) // 2
            if mid_count > t1:
                result = compute_contradiction_modulation(mid_count, base_cap)
                expected = base_cap * r1
                assert result == pytest.approx(expected, abs=1e-8)


# =============================================================================
# Tests: apply_contradiction_amplitude_modulation
# =============================================================================

class TestApplyContradictionAmplitudeModulation:
    """変調の適用テスト。"""

    def test_no_contradictions_no_change(self):
        """矛盾蓄積ゼロではamplitude_cap不変。"""
        cfg = _make_config()
        original_cap = cfg.amplitude_cap
        modulation = apply_contradiction_amplitude_modulation(cfg, 0)
        assert modulation == 0.0
        assert cfg.amplitude_cap == original_cap

    def test_modulation_increases_cap(self):
        """矛盾蓄積ありではamplitude_capが増加する。"""
        cfg = _make_config()
        original_cap = cfg.amplitude_cap
        modulation = apply_contradiction_amplitude_modulation(cfg, 30)
        assert modulation > 0.0
        assert cfg.amplitude_cap > original_cap

    def test_cap_stays_below_vo_max(self):
        """安全弁7: 変調後もvo_max_bias_strength未満。"""
        cfg = _make_config(amplitude_cap=0.14, vo_max_bias_strength=0.15)
        apply_contradiction_amplitude_modulation(cfg, 50)
        assert cfg.amplitude_cap < cfg.vo_max_bias_strength

    def test_cap_stays_below_vo_max_extreme(self):
        """安全弁7: 極端な蓄積件数でもvo_max_bias_strength未満。"""
        cfg = _make_config(amplitude_cap=0.12, vo_max_bias_strength=0.15)
        apply_contradiction_amplitude_modulation(cfg, 10000)
        assert cfg.amplitude_cap < cfg.vo_max_bias_strength

    def test_band_limit_enforced(self):
        """安全弁6: 変調後のcapが初期値の+10%を超えない。"""
        cfg = _make_config(amplitude_cap=0.12)
        base_cap = coefficient_registry.get("fluctuation", "amplitude_cap")
        apply_contradiction_amplitude_modulation(cfg, 50)
        max_allowed = base_cap * (1.0 + _CONTRADICTION_MODULATION_BAND_LIMIT)
        # Also must be less than vo_max
        assert cfg.amplitude_cap <= max_allowed + 1e-10

    def test_floor_maintained(self):
        """変調後もamplitude_floorがamplitude_cap未満であることを保証。"""
        cfg = _make_config()
        apply_contradiction_amplitude_modulation(cfg, 30)
        assert cfg.amplitude_floor < cfg.amplitude_cap

    def test_returns_actual_modulation(self):
        """返り値が実際に適用された変調量。"""
        cfg = _make_config()
        original_cap = cfg.amplitude_cap
        modulation = apply_contradiction_amplitude_modulation(cfg, 30)
        assert cfg.amplitude_cap == pytest.approx(
            original_cap + modulation, abs=1e-8
        )

    def test_idempotent_result_with_same_input(self):
        """同一入力で2回呼ぶと2回目も加算される（累積防止はセッション起動1回のみの呼び出しで担保）。"""
        cfg1 = _make_config()
        cfg2 = _make_config()
        m1 = apply_contradiction_amplitude_modulation(cfg1, 30)
        m2 = apply_contradiction_amplitude_modulation(cfg2, 30)
        assert m1 == m2
        assert cfg1.amplitude_cap == cfg2.amplitude_cap

    def test_small_count_small_modulation(self):
        """少数の矛盾蓄積では微小な変調量。"""
        cfg = _make_config()
        original_cap = cfg.amplitude_cap
        first_threshold = _CONTRADICTION_MODULATION_STEPS[0][0]
        apply_contradiction_amplitude_modulation(cfg, first_threshold)
        diff = cfg.amplitude_cap - original_cap
        assert diff > 0
        assert diff < original_cap * 0.05  # 5%未満

    def test_large_count_larger_modulation(self):
        """大量の矛盾蓄積ではより大きな変調量。"""
        cfg_small = _make_config()
        cfg_large = _make_config()
        m_small = apply_contradiction_amplitude_modulation(cfg_small, 10)
        m_large = apply_contradiction_amplitude_modulation(cfg_large, 50)
        assert m_large >= m_small


# =============================================================================
# Tests: Safety valves
# =============================================================================

class TestSafetyValve6BandLimit:
    """安全弁6: 変調帯域の制限。"""

    def test_band_limit_is_10_percent(self):
        """帯域上限が10%であることを確認。"""
        assert _CONTRADICTION_MODULATION_BAND_LIMIT == 0.10

    def test_modulation_steps_respect_band(self):
        """全段階値テーブルの変調割合が帯域上限以内。"""
        for _, ratio in _CONTRADICTION_MODULATION_STEPS:
            assert ratio <= _CONTRADICTION_MODULATION_BAND_LIMIT + 1e-10

    def test_compute_never_exceeds_band(self):
        """compute_contradiction_modulation の出力が帯域上限を超えない。"""
        base_cap = 0.12
        max_modulation = base_cap * _CONTRADICTION_MODULATION_BAND_LIMIT
        for count in range(0, 200):
            result = compute_contradiction_modulation(count, base_cap)
            assert result <= max_modulation + 1e-10


class TestSafetyValve7AbsoluteLimit:
    """安全弁7: 振幅上限値の絶対上限。"""

    def test_never_exceeds_vo_max(self):
        """どんな入力でもvo_max_bias_strengthを超えない。"""
        for cap in [0.10, 0.12, 0.14]:
            for vo_max in [0.15, 0.13]:
                if cap >= vo_max:
                    continue
                cfg = _make_config(amplitude_cap=cap, vo_max_bias_strength=vo_max)
                apply_contradiction_amplitude_modulation(cfg, 50)
                assert cfg.amplitude_cap < vo_max

    def test_close_to_vo_max_still_safe(self):
        """amplitude_capがvo_maxに近い場合でも安全。"""
        cfg = _make_config(amplitude_cap=0.149, vo_max_bias_strength=0.15)
        # __post_init__ will cap it
        assert cfg.amplitude_cap < 0.15
        apply_contradiction_amplitude_modulation(cfg, 50)
        assert cfg.amplitude_cap < 0.15


class TestSafetyValve8ReadOnly:
    """安全弁8: 矛盾記述構造への非介入。"""

    def test_contradiction_state_not_modified(self):
        """変調適用前後で矛盾記述構造の状態が変化しない。"""
        processor = create_contradiction_processor()
        # いくつかの矛盾対を蓄積
        for i in range(10):
            inputs = ContradictionInputs(
                self_model_emotion_intensity=0.9,
                meta_emotion_dominant_stability=0.1,
                current_tick=i,
            )
            processor.process(inputs)

        # 状態のスナップショットを取る
        state_before = processor.save()
        window_count_before = len(processor.state.contradiction_window)

        # 変調を適用（件数のみ参照、プロセッサには触らない）
        cfg = _make_config()
        apply_contradiction_amplitude_modulation(cfg, window_count_before)

        # 状態が変化していないことを確認
        state_after = processor.save()
        assert state_before == state_after
        assert len(processor.state.contradiction_window) == window_count_before


# =============================================================================
# Tests: C9-5 independence
# =============================================================================

class TestC95Independence:
    """既存のC9-5参照偏在→振幅変調リンクとの独立性。"""

    def test_contradiction_modulation_changes_cap_not_variability(self):
        """矛盾変調はamplitude_capを変更し、変動度合成には介入しない。"""
        cfg = _make_config()
        original_floor = cfg.amplitude_floor
        apply_contradiction_amplitude_modulation(cfg, 30)

        # 変調はcapのみを変更
        # reference_imbalanceは5段パイプラインの変動度合成に入力される
        # この2つは独立
        # 確認: floorは変更されていない（capが十分大きければ）
        assert cfg.amplitude_floor == original_floor or cfg.amplitude_floor < cfg.amplitude_cap

    def test_both_modulations_can_coexist(self):
        """矛盾変調と参照偏在変調が共存できることを確認。"""
        cfg = _make_config()
        apply_contradiction_amplitude_modulation(cfg, 30)

        # apply_scoring_fluctuation にreference_imbalanceを渡す
        candidates = [
            {"policy_label": "A", "_score": 5.0, "drive_target": "social",
             "expected_drive_change": {"social": -0.05}},
        ]
        result = apply_scoring_fluctuation(
            candidates=candidates,
            emotions={"joy": 0.5, "anger": 0.3},
            drives={"social": 0.5, "curiosity": 0.5},
            config=cfg,
            reference_imbalance=0.8,
        )
        assert len(result) == 1
        assert result[0]["_fluctuation_applied"] is True


# =============================================================================
# Tests: Session startup-only characteristic
# =============================================================================

class TestSessionStartupOnly:
    """セッション起動時一回のみの特性。"""

    def test_modulation_does_not_persist(self):
        """変調結果自体は永続化されない（次回起動時に再算出）。

        これはconfigがnon-persistent（save/load対象外）であることで保証される。
        新しいconfigを作成して同じ件数で呼べば同じ結果になる。
        """
        cfg1 = _make_config()
        cfg2 = _make_config()
        m1 = apply_contradiction_amplitude_modulation(cfg1, 25)
        m2 = apply_contradiction_amplitude_modulation(cfg2, 25)
        assert m1 == m2
        assert cfg1.amplitude_cap == cfg2.amplitude_cap

    def test_different_counts_different_modulations(self):
        """異なる蓄積件数では異なる変調量（双方向変化の可能性）。"""
        cfg1 = _make_config()
        cfg2 = _make_config()
        apply_contradiction_amplitude_modulation(cfg1, 10)
        apply_contradiction_amplitude_modulation(cfg2, 40)
        # 40件の方が変調量が大きいのでcapも大きい
        assert cfg2.amplitude_cap >= cfg1.amplitude_cap


# =============================================================================
# Tests: Bidirectionality
# =============================================================================

class TestBidirectionality:
    """双方向変化の可能性（FIFOにより件数が増減する）。"""

    def test_fewer_contradictions_less_modulation(self):
        """矛盾が少なければ変調量も小さくなる。"""
        cfg_few = _make_config()
        cfg_many = _make_config()
        m_few = apply_contradiction_amplitude_modulation(cfg_few, 3)
        m_many = apply_contradiction_amplitude_modulation(cfg_many, 30)
        assert m_many >= m_few

    def test_reversibility_across_sessions(self):
        """セッション間で蓄積件数が減れば変調量も減る。

        これは毎セッション再算出により自然に実現される。
        """
        # Session 1: 多くの矛盾
        cfg1 = _make_config()
        m1 = apply_contradiction_amplitude_modulation(cfg1, 40)

        # Session 2: 矛盾が減少（FIFOの自然消失）
        cfg2 = _make_config()
        m2 = apply_contradiction_amplitude_modulation(cfg2, 10)

        assert m1 >= m2


# =============================================================================
# Tests: Integration with InternalContradictionProcessor
# =============================================================================

class TestContradictionProcessorIntegration:
    """InternalContradictionProcessor との統合テスト。"""

    def test_read_count_from_processor(self):
        """プロセッサから蓄積件数を読み取って変調に使用できる。"""
        processor = create_contradiction_processor()
        # 矛盾を蓄積
        for i in range(20):
            inputs = ContradictionInputs(
                self_model_emotion_intensity=0.9,
                meta_emotion_dominant_stability=0.1,
                self_image_stability=0.9,
                temporal_diff_magnitude=0.9,
                current_tick=i,
            )
            processor.process(inputs)

        count = len(processor.state.contradiction_window)
        assert count > 0

        cfg = _make_config()
        modulation = apply_contradiction_amplitude_modulation(cfg, count)
        # 十分な矛盾が蓄積されていれば変調が発生
        if count >= _CONTRADICTION_MODULATION_STEPS[0][0]:
            assert modulation > 0

    def test_with_loaded_state(self):
        """永続化から復元された状態でも正しく動作する。"""
        # 状態を作成して保存
        processor = create_contradiction_processor()
        state = _make_contradiction_state_with_count(30)
        processor.load(state.to_dict())

        count = len(processor.state.contradiction_window)
        assert count == 30

        cfg = _make_config()
        modulation = apply_contradiction_amplitude_modulation(cfg, count)
        assert modulation > 0

    def test_empty_processor_no_modulation(self):
        """空のプロセッサでは変調なし。"""
        processor = create_contradiction_processor()
        count = len(processor.state.contradiction_window)
        assert count == 0

        cfg = _make_config()
        modulation = apply_contradiction_amplitude_modulation(cfg, count)
        assert modulation == 0.0


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_count_1_below_threshold(self):
        """蓄積件数1（最低閾値未満）。"""
        result = compute_contradiction_modulation(1, 0.12)
        assert result == 0.0

    def test_count_exactly_at_max_threshold(self):
        """蓄積件数が最大閾値ちょうど。"""
        max_threshold = _CONTRADICTION_MODULATION_STEPS[-1][0]
        max_ratio = _CONTRADICTION_MODULATION_STEPS[-1][1]
        result = compute_contradiction_modulation(max_threshold, 0.12)
        expected = 0.12 * max_ratio
        assert result == pytest.approx(expected, abs=1e-8)

    def test_very_large_count(self):
        """非常に大きな蓄積件数でも帯域上限内。"""
        result = compute_contradiction_modulation(100000, 0.12)
        max_expected = 0.12 * _CONTRADICTION_MODULATION_BAND_LIMIT
        assert result <= max_expected + 1e-10

    def test_very_small_base_cap(self):
        """非常に小さな基準振幅でも正常動作。"""
        result = compute_contradiction_modulation(30, 0.001)
        assert result >= 0
        assert result <= 0.001 * _CONTRADICTION_MODULATION_BAND_LIMIT + 1e-10

    def test_very_large_base_cap(self):
        """非常に大きな基準振幅でも変調量は割合で制限。"""
        result = compute_contradiction_modulation(30, 10.0)
        assert result <= 10.0 * _CONTRADICTION_MODULATION_BAND_LIMIT + 1e-10

    def test_config_with_floor_near_cap(self):
        """floorがcapに近い場合でも安全。"""
        cfg = _make_config(amplitude_cap=0.01, amplitude_floor=0.009,
                           vo_max_bias_strength=0.15)
        apply_contradiction_amplitude_modulation(cfg, 30)
        assert cfg.amplitude_floor < cfg.amplitude_cap


# =============================================================================
# Tests: Pipeline integration
# =============================================================================

class TestPipelineIntegration:
    """変調後のスコアリング揺らぎパイプライン統合テスト。"""

    def test_modulated_config_works_in_pipeline(self):
        """変調後のconfigでapply_scoring_fluctuationが正常動作する。"""
        cfg = _make_config()
        apply_contradiction_amplitude_modulation(cfg, 30)

        candidates = [
            {"policy_label": "A", "_score": 5.0, "drive_target": "social",
             "expected_drive_change": {"social": -0.05}},
            {"policy_label": "B", "_score": 4.5, "drive_target": "curiosity",
             "expected_drive_change": {"curiosity": -0.10}},
        ]
        result = apply_scoring_fluctuation(
            candidates=candidates,
            emotions={"joy": 0.8, "anger": 0.1},
            drives={"social": 0.7, "curiosity": 0.3, "expression": 0.5},
            elapsed_seconds=30.0,
            config=cfg,
        )
        assert len(result) == 2
        for c in result:
            assert c["_fluctuation_applied"] is True
            # 揺らぎが変調後のcapを超えない
            assert abs(c["_fluctuation"]) <= cfg.amplitude_cap * 1.01

    def test_modulated_larger_cap_allows_larger_fluctuation(self):
        """変調後のcapが大きいほど、より大きな揺らぎが可能。"""
        # 変調なし
        cfg_no_mod = _make_config()
        cap_no_mod = cfg_no_mod.amplitude_cap

        # 変調あり
        cfg_mod = _make_config()
        apply_contradiction_amplitude_modulation(cfg_mod, 50)
        cap_mod = cfg_mod.amplitude_cap

        # 変調ありの方がcapが大きい
        assert cap_mod > cap_no_mod

    def test_unmodulated_config_unchanged(self):
        """変調を適用しないconfigは元のまま。"""
        cfg = _make_config()
        original_cap = cfg.amplitude_cap
        original_floor = cfg.amplitude_floor

        # 変調なし（0件）
        apply_contradiction_amplitude_modulation(cfg, 0)

        assert cfg.amplitude_cap == original_cap
        assert cfg.amplitude_floor == original_floor
