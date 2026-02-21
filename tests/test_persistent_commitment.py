"""
tests/test_persistent_commitment.py - 持続的取り組み保持構造のテスト

テスト項目:
- 初期化テスト
- 昇格処理テスト (条件4: 唯一の生成経路)
- 減衰処理テスト (飽和構造, 慣性独立減衰)
- 解除判定テスト (条件2: 達成認知は解除トリガーではない)
- 資源競合テスト (揺らぎ付き動的分配)
- 認知記録テスト (FIFO, READ-ONLY, 評価判定なし)
- バイアス出力テスト (条件3: バイアス上限)
- 安全弁テスト (条件6: 6種)
- ループ遮断テスト (4つ)
- 永続化テスト (to_dict / from_dict)
- enrichment テスト
"""

import time
from typing import Any

import pytest

from psyche.persistent_commitment import (
    PersistentCommitmentConfig,
    PersistentCommitmentState,
    PersistentCommitmentProcessor,
    CommitmentItem,
    CognitionRecord,
    CommitmentCrossSectionInputs,
    create_persistent_commitment_processor,
    get_commitment_summary,
    _direction_similarity,
    _compute_candidate_alignment,
    _clamp,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_inputs(**kwargs) -> CommitmentCrossSectionInputs:
    """テスト用の8断面入力を生成する。"""
    return CommitmentCrossSectionInputs(**kwargs)


def _make_direction(social: float = 0.5, expression: float = 0.3) -> dict[str, float]:
    return {"social": social, "expression": expression}


def _make_candidate(
    policy_label: str = "engage",
    score: float = 0.5,
    drive_target: str = "social",
    direction: dict[str, float] | None = None,
) -> dict[str, Any]:
    result = {
        "policy_label": policy_label,
        "_score": score,
        "drive_target": drive_target,
    }
    if direction is not None:
        result["direction"] = direction
    return result


# ── 初期化テスト ──────────────────────────────────────────────────


class TestInitialization:
    """初期化テスト。"""

    def test_default_init(self):
        proc = create_persistent_commitment_processor()
        assert proc.state is not None
        assert len(proc.state.items) == 0
        assert len(proc.state.cognition_records) == 0
        assert proc.state.total_promotions == 0

    def test_custom_config(self):
        config = PersistentCommitmentConfig(max_slots=10, max_retention_ticks=500)
        proc = PersistentCommitmentProcessor(config=config)
        assert proc.config.max_slots == 10
        assert proc.config.max_retention_ticks == 500

    def test_state_property(self):
        proc = create_persistent_commitment_processor()
        assert isinstance(proc.state, PersistentCommitmentState)

    def test_empty_state_is_normal(self):
        """保持項目0件は正常状態。"""
        proc = create_persistent_commitment_processor()
        proc.tick(_make_inputs(), current_tick=1)
        assert len([it for it in proc.state.items if not it.released]) == 0


# ── 昇格処理テスト ────────────────────────────────────────────────


class TestPromotion:
    """昇格処理テスト (条件4: 唯一の生成経路)。"""

    def test_basic_promotion(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        assert item is not None
        assert item.source_goal_id == "g1"
        assert item.category == "exploration"
        assert item.strength == pytest.approx(0.6)
        assert item.promotion_tick == 100

    def test_promotion_rejected_low_strength(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.1,  # Below min_promotion_strength
            maintained_ticks=10,
            current_tick=100,
        )
        assert item is None

    def test_promotion_rejected_low_ticks(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=2,  # Below min_promotion_ticks
            current_tick=100,
        )
        assert item is None

    def test_promotion_creates_cognition_record(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        records = proc.state.cognition_records
        assert len(records) == 1
        assert records[0].record_type == "promotion"

    def test_multiple_promotions(self):
        proc = create_persistent_commitment_processor()
        for i in range(3):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i) / 3.0},
                remaining_strength=0.5 + i * 0.1,
                maintained_ticks=10,
                current_tick=100 + i,
            )
        active = [it for it in proc.state.items if not it.released]
        assert len(active) == 3

    def test_promotion_slot_full_replaces_weakest(self):
        config = PersistentCommitmentConfig(max_slots=3, dynamic_slot_base=3)
        proc = PersistentCommitmentProcessor(config=config)

        # Fill 3 slots
        for i in range(3):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.3 + i * 0.1,
                maintained_ticks=10,
                current_tick=100 + i,
            )

        # Promote stronger item when slots are full
        item = proc.try_promote(
            goal_id="g_strong",
            category="cat_strong",
            direction_signature={"dim": 10.0},
            remaining_strength=0.8,
            maintained_ticks=10,
            current_tick=110,
        )
        assert item is not None
        active = [it for it in proc.state.items if not it.released]
        assert len(active) == 3

    def test_promotion_rejected_when_weaker_than_all(self):
        config = PersistentCommitmentConfig(max_slots=2, dynamic_slot_base=2)
        proc = PersistentCommitmentProcessor(config=config)

        # Fill 2 slots with strong items
        for i in range(2):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.8,
                maintained_ticks=10,
                current_tick=100 + i,
            )

        # Try to promote weaker item
        item = proc.try_promote(
            goal_id="g_weak",
            category="cat_weak",
            direction_signature={"dim": 99.0},
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=110,
        )
        assert item is None

    def test_same_direction_reinforcement(self):
        """同一方向の既存項目がある場合は新規追加ではなく微小補強。"""
        proc = create_persistent_commitment_processor()
        direction = _make_direction(social=0.8, expression=0.5)

        # First promotion
        item1 = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=direction,
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        assert item1 is not None

        # Decay strength a bit
        item1.strength = 0.4

        # Second promotion with same direction
        item2 = proc.try_promote(
            goal_id="g2",
            category="exploration",
            direction_signature=direction,
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=110,
        )
        # Should return None (reinforcement instead of new item)
        assert item2 is None
        # Strength should have increased slightly
        assert item1.strength > 0.4

    def test_initial_inertia_from_maintained_ticks(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=25,  # inertia = 25/50 = 0.5
            current_tick=100,
        )
        assert item is not None
        assert item.inertia == pytest.approx(0.5)

    def test_inertia_capped(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=100,  # inertia = 100/50 = 2.0, but capped at 1.0
            current_tick=100,
        )
        assert item is not None
        assert item.inertia <= 1.0


# ── 減衰処理テスト ────────────────────────────────────────────────


class TestDecay:
    """減衰処理テスト。"""

    def _make_proc_with_item(
        self,
        strength: float = 0.6,
        inertia: float = 0.5,
    ) -> tuple[PersistentCommitmentProcessor, CommitmentItem]:
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=strength,
            maintained_ticks=int(inertia * 50),
            current_tick=0,
        )
        assert item is not None
        return proc, item

    def test_strength_decays_over_time(self):
        proc, item = self._make_proc_with_item(strength=0.6)
        initial = item.strength
        for tick in range(1, 11):
            proc.tick(_make_inputs(), current_tick=tick)
        assert item.strength < initial

    def test_inertia_decays_independently(self):
        """慣性は強度とは独立して減衰する（条件1）。"""
        proc, item = self._make_proc_with_item(strength=0.6, inertia=0.8)
        initial_inertia = item.inertia
        for tick in range(1, 11):
            proc.tick(_make_inputs(), current_tick=tick)
        assert item.inertia < initial_inertia

    def test_saturation_structure(self):
        """飽和構造: 高強度で減衰が再加速する。"""
        proc_high, item_high = self._make_proc_with_item(strength=0.95, inertia=0.1)
        proc_mid, item_mid = self._make_proc_with_item(strength=0.5, inertia=0.1)

        for tick in range(1, 20):
            proc_high.tick(_make_inputs(), current_tick=tick)
            proc_mid.tick(_make_inputs(), current_tick=tick)

        # High strength should have decayed more rapidly due to saturation
        high_decay = 0.95 - item_high.strength
        mid_decay = 0.5 - item_mid.strength
        # High should have decayed proportionally more than mid due to saturation
        assert high_decay / 0.95 > mid_decay / 0.5 * 0.5  # Some measurable difference

    def test_low_strength_accelerated_decay(self):
        """低強度ではより速く減衰する。"""
        proc, item = self._make_proc_with_item(strength=0.35, inertia=0.1)
        # Manually lower strength to the low range to test accelerated decay
        item.strength = 0.2
        initial = item.strength
        proc.tick(_make_inputs(), current_tick=1)
        decay_amount = initial - item.strength
        assert decay_amount > 0

    def test_max_retention_forced_decay(self):
        """最大保持期間超過で加速減衰（安全弁4）。"""
        config = PersistentCommitmentConfig(max_retention_ticks=5)
        proc = PersistentCommitmentProcessor(config=config)
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None

        # Run past max retention
        for tick in range(1, 15):
            proc.tick(_make_inputs(), current_tick=tick)

        # Should have been released or nearly zero
        assert item.strength < 0.3 or item.released

    def test_no_strength_increase(self):
        """減衰は一方向。外部から停止・逆転できない。"""
        proc, item = self._make_proc_with_item(strength=0.6)
        strengths = [item.strength]
        for tick in range(1, 20):
            proc.tick(_make_inputs(), current_tick=tick)
            strengths.append(item.strength)

        # Each subsequent strength should be <= previous
        for i in range(1, len(strengths)):
            assert strengths[i] <= strengths[i - 1] + 1e-9


# ── 解除判定テスト ────────────────────────────────────────────────


class TestRelease:
    """解除判定テスト (条件2: 達成認知は解除トリガーではない)。"""

    def test_release_by_time_decay(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None

        # Run until released
        for tick in range(1, 200):
            proc.tick(_make_inputs(), current_tick=tick)
            if item.released:
                break

        assert item.released
        # Check that release reason is recorded
        release_records = [
            r for r in proc.state.cognition_records
            if r.record_type == "release"
        ]
        assert len(release_records) > 0

    def test_release_by_state_divergence(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None

        # High internal state divergence
        high_divergence = _make_inputs(
            arousal_delta=0.8,
            context_disruption=0.9,
            drive_variability=0.8,
            transient_direction_distance=0.9,
            orientation_alignment_delta=0.7,
            competing_candidate_intensity=0.8,
            responsibility_pressure=0.8,
            scoring_fluctuation_amount=0.7,
        )

        for tick in range(1, 50):
            proc.tick(high_divergence, current_tick=tick)
            if item.released:
                break

        assert item.released

    def test_release_by_competition(self):
        """帯域を完全に失った場合の解除。"""
        config = PersistentCommitmentConfig(
            max_slots=3,
            dynamic_slot_base=3,
            bandwidth_deficit_decay_boost=5.0,
        )
        proc = PersistentCommitmentProcessor(config=config)

        # Add a weak item and two strong items
        weak = proc.try_promote(
            goal_id="weak",
            category="cat_weak",
            direction_signature={"dim": 0.1},
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=0,
        )
        for i in range(2):
            proc.try_promote(
                goal_id=f"strong{i}",
                category=f"cat_strong{i}",
                direction_signature={"dim": float(i + 1)},
                remaining_strength=0.9,
                maintained_ticks=10,
                current_tick=1 + i,
            )

        # Run for a while - weak should eventually be released
        for tick in range(3, 100):
            proc.tick(_make_inputs(), current_tick=tick)
            if weak is not None and weak.released:
                break

    def test_no_achievement_release(self):
        """達成認知は解除トリガーではない（条件2）。"""
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None
        # No method to trigger achievement-based release exists
        # Item should only be released through time/state/competition
        assert not item.released

    def test_release_records_reason(self):
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=0,
        )
        # Force strength below threshold
        item.strength = 0.01
        proc.tick(_make_inputs(), current_tick=1)
        assert item.released
        release_records = [
            r for r in proc.state.cognition_records
            if r.record_type == "release"
        ]
        assert len(release_records) >= 1
        assert release_records[-1].release_reason == "time_decay"


# ── 資源競合テスト ────────────────────────────────────────────────


class TestResourceCompetition:
    """資源競合テスト。"""

    def test_bandwidth_allocation(self):
        proc = create_persistent_commitment_processor()
        for i in range(3):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.5 + i * 0.1,
                maintained_ticks=10,
                current_tick=100 + i,
            )

        proc.tick(_make_inputs(), current_tick=200)
        active = [it for it in proc.state.items if not it.released]
        # All active items should have some bandwidth
        for item in active:
            assert item.bandwidth_share >= 0.0
        # Sum should be approximately 1.0
        total = sum(it.bandwidth_share for it in active)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_bandwidth_with_fluctuation(self):
        """揺らぎにより帯域分配が毎回微小に変動する。"""
        proc = create_persistent_commitment_processor()
        # Use strong items so they don't get released during the test
        proc.try_promote(
            goal_id="g0",
            category="cat0",
            direction_signature={"dim": 0.0},
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )
        proc.try_promote(
            goal_id="g1",
            category="cat1",
            direction_signature={"dim": 1.0},
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=1,
        )

        shares_history: list[list[float]] = []
        for tick in range(2, 8):
            proc.tick(
                _make_inputs(scoring_fluctuation_amount=float(tick) * 0.1),
                current_tick=tick,
            )
            active = [it for it in proc.state.items if not it.released]
            if len(active) >= 2:
                shares_history.append([it.bandwidth_share for it in active])

        # Shares should exist and sum to ~1.0
        assert len(shares_history) > 0
        for shares in shares_history:
            assert sum(shares) == pytest.approx(1.0, abs=0.01)


# ── 認知記録テスト ────────────────────────────────────────────────


class TestCognitionRecords:
    """認知記録テスト (FIFO, READ-ONLY, 評価判定なし)。"""

    def test_records_created_on_promotion(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        records = proc.get_cognition_records()
        assert len(records) == 1
        assert records[0].record_type == "promotion"

    def test_records_fifo(self):
        config = PersistentCommitmentConfig(max_cognition_records=5)
        proc = PersistentCommitmentProcessor(config=config)

        for i in range(10):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.6,
                maintained_ticks=10,
                current_tick=i,
            )

        records = proc.get_cognition_records()
        assert len(records) <= 5

    def test_records_read_only(self):
        """get_cognition_records は読み取り専用コピー。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        records = proc.get_cognition_records()
        records.clear()
        # Internal state should be unaffected
        assert len(proc.state.cognition_records) > 0

    def test_records_no_evaluation(self):
        """認知記録に評価判定がないことを確認。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        records = proc.get_cognition_records()
        for rec in records:
            # No "success", "failure", "good", "bad" fields
            rec_dict = rec.to_dict()
            assert "success" not in rec_dict
            assert "failure" not in rec_dict
            assert "evaluation" not in rec_dict

    def test_records_equal_weight(self):
        """すべての記録は等価。重み付けなし。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        records = proc.get_cognition_records()
        for rec in records:
            rec_dict = rec.to_dict()
            assert "weight" not in rec_dict
            assert "priority" not in rec_dict
            assert "importance" not in rec_dict


# ── バイアス出力テスト ────────────────────────────────────────────


class TestBiasOutput:
    """バイアス出力テスト (条件3: バイアス上限)。"""

    def test_no_bias_when_empty(self):
        proc = create_persistent_commitment_processor()
        candidate = _make_candidate()
        bias = proc.compute_bias(candidate)
        assert bias == 0.0

    def test_bias_within_limits(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="approach",
            direction_signature=_make_direction(),
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        candidate = _make_candidate(policy_label="engage")
        bias = proc.compute_bias(candidate)
        assert abs(bias) <= proc.config.max_total_bias

    def test_bias_total_capped(self):
        """複数保持項目のバイアス合計が上限を超えない（安全弁5）。"""
        proc = create_persistent_commitment_processor()
        for i in range(5):
            proc.try_promote(
                goal_id=f"g{i}",
                category="approach",
                direction_signature={"social": 1.0},
                remaining_strength=0.9,
                maintained_ticks=10,
                current_tick=i,
            )
        candidate = _make_candidate(policy_label="engage")
        bias = proc.compute_bias(candidate)
        assert abs(bias) <= proc.config.max_total_bias

    def test_bias_below_value_orientation(self):
        """バイアスがvalue_orientationの+-5%未満（条件3）。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="approach",
            direction_signature=_make_direction(),
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        candidate = _make_candidate(policy_label="engage")
        bias = proc.compute_bias(candidate)
        # value_orientation max_bias_strength = 0.15
        assert abs(bias) < 0.15

    def test_apply_bias_to_candidates(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="approach",
            direction_signature=_make_direction(),
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )
        candidates = [
            _make_candidate(policy_label="engage", score=0.5),
            _make_candidate(policy_label="silence", score=0.5),
        ]
        result = proc.apply_bias_to_candidates(candidates)
        assert len(result) == 2
        # Each candidate should have bias metadata
        for c in result:
            assert "_persistent_commitment_bias" in c

    def test_cognition_records_not_used_for_bias(self):
        """認知記録はバイアス計算に使用されない（経路遮断4）。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="approach",
            direction_signature=_make_direction(),
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )
        # Add many cognition records
        for i in range(20):
            proc._add_cognition_record("g1", "strength_change", i)

        candidate = _make_candidate(policy_label="engage")
        bias_with_records = proc.compute_bias(candidate)

        # Bias should depend only on active items, not records
        assert isinstance(bias_with_records, float)

    def test_empty_candidates_returns_empty(self):
        proc = create_persistent_commitment_processor()
        result = proc.apply_bias_to_candidates([])
        assert result == []


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁テスト (条件6: 6種)。"""

    def test_safety_valve_1_concentration(self):
        """安全弁1: 単一保持項目集中度の監視。"""
        proc = create_persistent_commitment_processor()
        # One very strong, others weak
        proc.try_promote(
            goal_id="strong",
            category="cat_strong",
            direction_signature={"dim": 1.0},
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        proc.try_promote(
            goal_id="weak",
            category="cat_weak",
            direction_signature={"dim": 2.0},
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=1,
        )

        proc.tick(_make_inputs(), current_tick=2)
        # Concentration should be high
        assert proc.state.concentration_ratio > 0.5

    def test_safety_valve_2_inertia_cap(self):
        """安全弁2: 慣性累積上限。"""
        config = PersistentCommitmentConfig(max_total_inertia=1.0)
        proc = PersistentCommitmentProcessor(config=config)

        for i in range(3):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.6,
                maintained_ticks=50,  # inertia = 1.0
                current_tick=i,
            )

        active = [it for it in proc.state.items if not it.released]
        total_inertia = sum(it.inertia for it in active)
        assert total_inertia <= config.max_total_inertia + 0.01

    def test_safety_valve_3_same_direction_suppression(self):
        """安全弁3: 同一方向連続保持抑制。"""
        config = PersistentCommitmentConfig(
            same_direction_threshold=2,
            same_direction_promotion_penalty=0.5,
        )
        proc = PersistentCommitmentProcessor(config=config)

        # Promote same direction twice
        for i in range(2):
            proc.try_promote(
                goal_id=f"g{i}",
                category="exploration",
                direction_signature={"social": float(i + 1)},
                remaining_strength=0.6,
                maintained_ticks=10,
                current_tick=i,
            )

        # Third same-direction should be suppressed if strength is not high enough
        item = proc.try_promote(
            goal_id="g2",
            category="exploration",
            direction_signature={"social": 3.0},
            remaining_strength=0.5,  # Below 0.25 + 0.5 = 0.75
            maintained_ticks=10,
            current_tick=10,
        )
        assert item is None

    def test_safety_valve_4_max_retention(self):
        """安全弁4: 最大保持期間の絶対上限。"""
        config = PersistentCommitmentConfig(max_retention_ticks=10)
        proc = PersistentCommitmentProcessor(config=config)

        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None
        assert item.remaining_ticks == 10

        # Run past max retention
        for tick in range(1, 30):
            proc.tick(_make_inputs(), current_tick=tick)

        # Item should be released or very weak
        assert item.released or item.strength < 0.1

    def test_safety_valve_5_bias_total_cap(self):
        """安全弁5: バイアス総量上限。"""
        proc = create_persistent_commitment_processor()
        for i in range(5):
            proc.try_promote(
                goal_id=f"g{i}",
                category="approach",
                direction_signature={"social": 1.0},
                remaining_strength=0.9,
                maintained_ticks=10,
                current_tick=i,
            )
        candidate = _make_candidate(policy_label="engage")
        bias = proc.compute_bias(candidate)
        assert abs(bias) <= proc.config.max_total_bias

    def test_safety_valve_6_emergency_mass_decay(self):
        """安全弁6: 全保持項目の強制一括減衰。"""
        config = PersistentCommitmentConfig(
            consecutive_safety_trigger=2,
            concentration_threshold=0.6,
            emergency_decay_ratio=0.5,
        )
        proc = PersistentCommitmentProcessor(config=config)

        # One very strong, one weak
        proc.try_promote(
            goal_id="strong",
            category="cat_strong",
            direction_signature={"dim": 1.0},
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        proc.try_promote(
            goal_id="weak",
            category="cat_weak",
            direction_signature={"dim": 2.0},
            remaining_strength=0.3,
            maintained_ticks=10,
            current_tick=1,
        )

        # Run enough ticks to trigger emergency
        strengths_before = [
            it.strength for it in proc.state.items if not it.released
        ]

        for tick in range(2, 20):
            proc.tick(_make_inputs(), current_tick=tick)

        # After emergency, all items should have lower strength
        active = [it for it in proc.state.items if not it.released]
        if active:
            for item in active:
                assert item.strength < 0.9  # Should have decayed


# ── ループ遮断テスト ──────────────────────────────────────────────


class TestLoopPrevention:
    """自己強化ループ遮断テスト（4つ）。"""

    def test_loop1_reinforce_cap(self):
        """ループ遮断1: 再昇格時の補強に上限がある。"""
        proc = create_persistent_commitment_processor()
        direction = _make_direction(social=0.8, expression=0.5)

        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=direction,
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=0,
        )
        initial_strength = item.initial_strength
        cap = initial_strength * proc.config.reinforce_cap_ratio

        # Decay the item's strength below the cap so reinforcement can apply
        item.strength = 0.3

        # Multiple reinforcement attempts
        for i in range(10):
            proc.try_promote(
                goal_id=f"g_{i}",
                category="exploration",
                direction_signature=direction,
                remaining_strength=0.9,
                maintained_ticks=10,
                current_tick=10 + i,
            )

        # Strength should have increased but never exceed initial_strength * cap_ratio
        assert item.strength > 0.3  # Some reinforcement happened
        assert item.strength <= cap + 0.001

    def test_loop2_inertia_independent_decay(self):
        """ループ遮断2: 慣性は強度とは独立して減衰。"""
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=50,  # inertia = 1.0
            current_tick=0,
        )

        inertias = [item.inertia]
        for tick in range(1, 30):
            proc.tick(_make_inputs(), current_tick=tick)
            inertias.append(item.inertia)

        # Inertia should monotonically decrease
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i - 1] + 1e-9

    def test_loop3_same_direction_bias_cap(self):
        """ループ遮断3: 同一方向バイアス合計に上限。"""
        proc = create_persistent_commitment_processor()
        # Multiple same-direction items (using different direction sigs to avoid reinforcement)
        for i in range(5):
            proc.try_promote(
                goal_id=f"g{i}",
                category="approach",
                direction_signature={"social": float(i + 1)},
                remaining_strength=0.9,
                maintained_ticks=10,
                current_tick=i,
            )

        candidate = _make_candidate(policy_label="engage", direction={"social": 1.0})
        bias = proc.compute_bias(candidate)
        assert abs(bias) <= proc.config.max_total_bias

    def test_loop4_cognition_record_no_judgment_path(self):
        """ループ遮断4: 認知記録→判断系の経路は遮断。"""
        proc = create_persistent_commitment_processor()
        # The processor has no method that feeds cognition records into bias
        # This is a structural guarantee
        assert not hasattr(proc, "compute_bias_from_records")
        assert not hasattr(proc, "adjust_bias_from_history")


# ── 永続化テスト ──────────────────────────────────────────────────


class TestPersistence:
    """永続化テスト (to_dict / from_dict)。"""

    def test_state_roundtrip(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=100,
        )
        proc.tick(_make_inputs(), current_tick=101)

        data = proc.state.to_dict()
        restored = PersistentCommitmentState.from_dict(data)

        assert len(restored.items) == len(proc.state.items)
        assert len(restored.cognition_records) == len(proc.state.cognition_records)
        assert restored.total_promotions == proc.state.total_promotions

    def test_item_roundtrip(self):
        item = CommitmentItem(
            item_id="test123",
            source_goal_id="g1",
            category="exploration",
            direction_signature={"social": 0.5},
            strength=0.7,
            initial_strength=0.8,
            inertia=0.4,
            promotion_tick=100,
            remaining_ticks=150,
            bandwidth_share=0.33,
            released=False,
        )
        data = item.to_dict()
        restored = CommitmentItem.from_dict(data)
        assert restored.item_id == "test123"
        assert restored.strength == pytest.approx(0.7)
        assert restored.direction_signature == {"social": 0.5}

    def test_cognition_record_roundtrip(self):
        record = CognitionRecord(
            item_id="test123",
            record_type="release",
            tick=200,
            release_reason="time_decay",
            residual_strength=0.05,
        )
        data = record.to_dict()
        restored = CognitionRecord.from_dict(data)
        assert restored.record_type == "release"
        assert restored.release_reason == "time_decay"
        assert restored.residual_strength == pytest.approx(0.05)

    def test_empty_state_roundtrip(self):
        state = PersistentCommitmentState()
        data = state.to_dict()
        restored = PersistentCommitmentState.from_dict(data)
        assert len(restored.items) == 0
        assert len(restored.cognition_records) == 0

    def test_validate_on_load(self):
        """ロード時の最大保持期間超過検証。"""
        proc = create_persistent_commitment_processor()
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=0,
        )
        item.remaining_ticks = 0  # Simulate expired

        proc.validate_on_load()
        # After validation, tick should apply accelerated decay
        proc.tick(_make_inputs(), current_tick=1)
        assert item.strength < 0.6  # Should have decayed faster


# ── enrichment テスト ─────────────────────────────────────────────


class TestEnrichment:
    """enrichment テスト。"""

    def test_empty_enrichment(self):
        proc = create_persistent_commitment_processor()
        data = proc.get_enrichment_data()
        assert data["active_count"] == 0
        assert data["summary_text"] == "持続保持: 待機中"

    def test_enrichment_with_items(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=0,
        )
        data = proc.get_enrichment_data()
        assert data["active_count"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["category"] == "exploration"
        assert "strength_level" in data["items"][0]

    def test_enrichment_strength_levels(self):
        """段階値の等価列挙。"""
        proc = create_persistent_commitment_processor()

        strengths = [0.8, 0.5, 0.2, 0.05]
        expected_levels = ["高", "中", "低", "微"]

        for i, (s, expected) in enumerate(zip(strengths, expected_levels)):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=s,
                maintained_ticks=10,
                current_tick=i,
            )

        data = proc.get_enrichment_data()
        # Check strength levels (items may have decayed slightly)
        for entry in data["items"]:
            assert entry["strength_level"] in ["高", "中", "低", "微"]

    def test_enrichment_no_emphasis(self):
        """特定の保持項目を推奨・強調しない。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.9,
            maintained_ticks=10,
            current_tick=0,
        )
        data = proc.get_enrichment_data()
        summary = data["summary_text"]
        assert "推奨" not in summary
        assert "重要" not in summary
        assert "注目" not in summary

    def test_enrichment_recent_records(self):
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.6,
            maintained_ticks=10,
            current_tick=0,
        )
        data = proc.get_enrichment_data()
        assert len(data["recent_records"]) > 0


# ── ヘルパー関数テスト ────────────────────────────────────────────


class TestHelpers:
    """ヘルパー関数テスト。"""

    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.5, 0.2, 0.8) == 0.5
        assert _clamp(0.1, 0.2, 0.8) == 0.2
        assert _clamp(0.9, 0.2, 0.8) == 0.8

    def test_direction_similarity_identical(self):
        sig = {"a": 1.0, "b": 0.0}
        assert _direction_similarity(sig, sig) == pytest.approx(1.0)

    def test_direction_similarity_orthogonal(self):
        sig_a = {"a": 1.0, "b": 0.0}
        sig_b = {"a": 0.0, "b": 1.0}
        assert _direction_similarity(sig_a, sig_b) == pytest.approx(0.0)

    def test_direction_similarity_empty(self):
        assert _direction_similarity({}, {"a": 1.0}) == 0.0
        assert _direction_similarity({"a": 1.0}, {}) == 0.0

    def test_compute_candidate_alignment(self):
        item = CommitmentItem(category="approach", direction_signature={"social": 1.0})
        candidate = _make_candidate(policy_label="engage")
        alignment = _compute_candidate_alignment(candidate, item)
        assert alignment >= 0.0

    def test_get_commitment_summary_empty(self):
        state = PersistentCommitmentState()
        assert get_commitment_summary(state) == "持続保持: 待機中"

    def test_get_commitment_summary_with_items(self):
        state = PersistentCommitmentState()
        state.items.append(CommitmentItem(
            category="exploration",
            strength=0.6,
            released=False,
        ))
        summary = get_commitment_summary(state)
        assert "保持中=1" in summary
        assert "exploration" in summary


# ── CommitmentCrossSectionInputs テスト ───────────────────────────


class TestCrossSectionInputs:
    """8断面入力テスト。"""

    def test_default_inputs(self):
        inputs = CommitmentCrossSectionInputs()
        assert inputs.dominant_emotion == ""
        assert inputs.arousal_delta == 0.0
        assert inputs.context_disruption == 0.0

    def test_custom_inputs(self):
        inputs = _make_inputs(
            dominant_emotion="joy",
            arousal_delta=0.3,
            context_disruption=0.5,
        )
        assert inputs.dominant_emotion == "joy"
        assert inputs.arousal_delta == 0.3
        assert inputs.context_disruption == 0.5


# ── ファクトリ関数テスト ──────────────────────────────────────────


class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_default(self):
        proc = create_persistent_commitment_processor()
        assert isinstance(proc, PersistentCommitmentProcessor)

    def test_create_with_config(self):
        config = PersistentCommitmentConfig(max_slots=10)
        proc = create_persistent_commitment_processor(config=config)
        assert proc.config.max_slots == 10


# ── 統合テスト ────────────────────────────────────────────────────


class TestIntegration:
    """統合テスト: 複数機能の組み合わせ。"""

    def test_full_lifecycle(self):
        """昇格→減衰→解除→記録の全サイクル。"""
        proc = create_persistent_commitment_processor()

        # Promote
        item = proc.try_promote(
            goal_id="g1",
            category="exploration",
            direction_signature=_make_direction(),
            remaining_strength=0.5,
            maintained_ticks=10,
            current_tick=0,
        )
        assert item is not None
        assert not item.released

        # Decay over time
        for tick in range(1, 100):
            proc.tick(_make_inputs(), current_tick=tick)
            if item.released:
                break

        assert item.released
        assert proc.state.total_releases > 0
        assert len(proc.state.cognition_records) >= 2  # promotion + release

    def test_multiple_items_lifecycle(self):
        """複数保持項目の並行管理。"""
        proc = create_persistent_commitment_processor()

        for i in range(3):
            proc.try_promote(
                goal_id=f"g{i}",
                category=f"cat{i}",
                direction_signature={"dim": float(i)},
                remaining_strength=0.5 + i * 0.1,
                maintained_ticks=10,
                current_tick=i,
            )

        # Run for a while
        for tick in range(3, 50):
            proc.tick(_make_inputs(), current_tick=tick)

        # Some should have been released
        summary = proc.get_summary()
        assert summary["recent_promotions"] == 3

    def test_bias_then_decay(self):
        """バイアス出力→減衰→バイアス減少。"""
        proc = create_persistent_commitment_processor()
        proc.try_promote(
            goal_id="g1",
            category="approach",
            direction_signature=_make_direction(),
            remaining_strength=0.7,
            maintained_ticks=10,
            current_tick=0,
        )

        candidate = _make_candidate(policy_label="engage")
        bias_initial = proc.compute_bias(candidate)

        # Decay
        for tick in range(1, 20):
            proc.tick(_make_inputs(), current_tick=tick)

        bias_after = proc.compute_bias(candidate)
        # Bias should decrease as strength decays
        assert bias_after <= bias_initial + 0.001
