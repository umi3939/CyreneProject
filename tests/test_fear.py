"""Tests for psyche/fear.py - Fear Index Computation & Emotion/Drive Boost."""

from __future__ import annotations

import pytest

from psyche.fear import (
    _clamp01,
    compute_fear_index,
    fear_drive_boost,
    fear_emotion_boost,
)
from psyche.pillars import FearIndex
from psyche.state import DriveVector, EmotionVector


# ── helpers ───────────────────────────────────────────────────


def _emo(**kw: float) -> EmotionVector:
    return EmotionVector(**kw)


def _drv(**kw: float) -> DriveVector:
    return DriveVector(**kw)


# ── 1. compute_fear_index with default values (all zero) ─────


class TestComputeFearIndexDefaults:
    def test_all_defaults_zero(self) -> None:
        fi = compute_fear_index()
        assert fi.identity_risk == 0.0
        assert fi.attachment_risk == 0.0
        assert fi.continuity_risk == 0.0
        assert fi.projection_risk == 0.0

    def test_all_defaults_value_zero(self) -> None:
        fi = compute_fear_index()
        assert fi.value == pytest.approx(0.0)


# ── 2. compute_fear_index with various risk levels ───────────


class TestComputeFearIndexVariousRisks:
    def test_single_identity(self) -> None:
        fi = compute_fear_index(identity_risk=0.5)
        assert fi.identity_risk == 0.5
        assert fi.attachment_risk == 0.0

    def test_single_attachment(self) -> None:
        fi = compute_fear_index(attachment_risk=0.8)
        assert fi.attachment_risk == 0.8

    def test_all_max(self) -> None:
        fi = compute_fear_index(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        assert fi.value == pytest.approx(1.0)

    def test_mixed_risks(self) -> None:
        fi = compute_fear_index(
            identity_risk=0.4,
            attachment_risk=0.6,
            continuity_risk=0.1,
            projection_risk=0.9,
        )
        assert fi.identity_risk == 0.4
        assert fi.attachment_risk == 0.6
        assert fi.continuity_risk == 0.1
        assert fi.projection_risk == 0.9

    def test_returns_fear_index_type(self) -> None:
        fi = compute_fear_index(identity_risk=0.2)
        assert isinstance(fi, FearIndex)


# ── 3. FearIndex.value calculation (weighted) ────────────────


class TestFearIndexValue:
    def test_identity_weight_03(self) -> None:
        fi = FearIndex(identity_risk=1.0)
        assert fi.value == pytest.approx(0.3)

    def test_attachment_weight_03(self) -> None:
        fi = FearIndex(attachment_risk=1.0)
        assert fi.value == pytest.approx(0.3)

    def test_continuity_weight_02(self) -> None:
        fi = FearIndex(continuity_risk=1.0)
        assert fi.value == pytest.approx(0.2)

    def test_projection_weight_02(self) -> None:
        fi = FearIndex(projection_risk=1.0)
        assert fi.value == pytest.approx(0.2)

    def test_weighted_sum_example(self) -> None:
        # 0.5*0.3 + 0.5*0.3 + 0.5*0.2 + 0.5*0.2 = 0.5
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        assert fi.value == pytest.approx(0.5)

    def test_weighted_sum_asymmetric(self) -> None:
        # 0.2*0.3 + 0.4*0.3 + 0.6*0.2 + 0.8*0.2
        #   = 0.06 + 0.12 + 0.12 + 0.16 = 0.46
        fi = FearIndex(
            identity_risk=0.2,
            attachment_risk=0.4,
            continuity_risk=0.6,
            projection_risk=0.8,
        )
        assert fi.value == pytest.approx(0.46)


# ── 4. fear_emotion_boost below threshold (0.2) -> no change ─


class TestFearEmotionBoostBelowThreshold:
    def test_value_zero_unchanged(self) -> None:
        fi = FearIndex()  # all zero => value = 0.0
        emo = _emo(joy=0.5, fear=0.1, sorrow=0.1)
        result = fear_emotion_boost(fi, emo)
        assert result.joy == pytest.approx(0.5)
        assert result.fear == pytest.approx(0.1)
        assert result.sorrow == pytest.approx(0.1)

    def test_value_below_threshold_unchanged(self) -> None:
        # value = 0.1*0.3 + 0.1*0.3 + 0.1*0.2 + 0.1*0.2 = 0.1
        fi = FearIndex(
            identity_risk=0.1,
            attachment_risk=0.1,
            continuity_risk=0.1,
            projection_risk=0.1,
        )
        assert fi.value == pytest.approx(0.1)
        emo = _emo(fear=0.3, sorrow=0.2)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(0.3)
        assert result.sorrow == pytest.approx(0.2)

    def test_returns_same_object_when_below_threshold(self) -> None:
        fi = FearIndex()
        emo = _emo(joy=0.5)
        result = fear_emotion_boost(fi, emo)
        # When value <= 0.2, the function returns the same object
        assert result is emo


# ── 5. fear_emotion_boost at threshold boundary ──────────────


class TestFearEmotionBoostAtBoundary:
    def test_exactly_at_02_no_boost(self) -> None:
        # value = 0.2 exactly => no boost (condition is <= 0.2)
        fi = FearIndex(
            identity_risk=0.2,
            attachment_risk=0.2,
            continuity_risk=0.2,
            projection_risk=0.2,
        )
        assert fi.value == pytest.approx(0.2)
        emo = _emo(fear=0.1, sorrow=0.1)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(0.1)
        assert result.sorrow == pytest.approx(0.1)

    def test_barely_above_02_gets_boost(self) -> None:
        # value slightly above 0.2
        # 0.3*0.3 + 0.2*0.3 + 0.2*0.2 + 0.2*0.2 = 0.09+0.06+0.04+0.04 = 0.23
        fi = FearIndex(
            identity_risk=0.3,
            attachment_risk=0.2,
            continuity_risk=0.2,
            projection_risk=0.2,
        )
        assert fi.value > 0.2
        emo = _emo(fear=0.0, sorrow=0.0)
        result = fear_emotion_boost(fi, emo)
        # boost = (0.23 - 0.2) * 0.5 = 0.015
        expected_boost = (fi.value - 0.2) * 0.5
        assert result.fear == pytest.approx(expected_boost, abs=1e-6)
        assert result.sorrow == pytest.approx(expected_boost * 0.5, abs=1e-6)


# ── 6. fear_emotion_boost above threshold ────────────────────


class TestFearEmotionBoostAboveThreshold:
    def test_moderate_fear_boosted(self) -> None:
        # value = 0.5 => boost = (0.5 - 0.2) * 0.5 = 0.15
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        assert fi.value == pytest.approx(0.5)
        emo = _emo(fear=0.1, sorrow=0.1)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(0.1 + 0.15)
        assert result.sorrow == pytest.approx(0.1 + 0.075)

    def test_high_fear_boosted(self) -> None:
        # value = 1.0 => boost = (1.0 - 0.2) * 0.5 = 0.4
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        assert fi.value == pytest.approx(1.0)
        emo = _emo(fear=0.0, sorrow=0.0)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(0.4)
        assert result.sorrow == pytest.approx(0.2)

    def test_new_object_returned(self) -> None:
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        emo = _emo(fear=0.1)
        result = fear_emotion_boost(fi, emo)
        assert result is not emo


# ── 7. fear_emotion_boost clamping at 1.0 ────────────────────


class TestFearEmotionBoostClamping:
    def test_fear_clamped_at_1(self) -> None:
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        # boost = 0.4, fear starts at 0.9 => 1.3 => clamp to 1.0
        emo = _emo(fear=0.9, sorrow=0.0)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(1.0)

    def test_sorrow_clamped_at_1(self) -> None:
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        # boost * 0.5 = 0.2, sorrow starts at 0.95 => 1.15 => clamp to 1.0
        emo = _emo(sorrow=0.95)
        result = fear_emotion_boost(fi, emo)
        assert result.sorrow == pytest.approx(1.0)

    def test_both_clamped(self) -> None:
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        emo = _emo(fear=0.95, sorrow=0.95)
        result = fear_emotion_boost(fi, emo)
        assert result.fear == pytest.approx(1.0)
        assert result.sorrow == pytest.approx(1.0)


# ── 8. fear_emotion_boost other emotions untouched ───────────


class TestFearEmotionBoostOtherEmotions:
    def test_joy_unchanged(self) -> None:
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        emo = _emo(joy=0.7, anger=0.3, surprise=0.4, love=0.6, fun=0.2)
        result = fear_emotion_boost(fi, emo)
        assert result.joy == pytest.approx(0.7)
        assert result.anger == pytest.approx(0.3)
        assert result.surprise == pytest.approx(0.4)
        assert result.love == pytest.approx(0.6)
        assert result.fun == pytest.approx(0.2)

    def test_all_non_fear_sorrow_preserved(self) -> None:
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        emo = _emo(joy=0.1, anger=0.2, surprise=0.3, love=0.4, fun=0.5,
                    fear=0.0, sorrow=0.0)
        result = fear_emotion_boost(fi, emo)
        assert result.joy == pytest.approx(0.1)
        assert result.anger == pytest.approx(0.2)
        assert result.surprise == pytest.approx(0.3)
        assert result.love == pytest.approx(0.4)
        assert result.fun == pytest.approx(0.5)


# ── 9. fear_drive_boost below attachment threshold ───────────


class TestFearDriveBoostBelowAttachmentThreshold:
    def test_attachment_zero_no_social_change(self) -> None:
        fi = FearIndex(attachment_risk=0.0)
        drv = _drv(social=0.5)
        result = fear_drive_boost(fi, drv)
        assert result.social == pytest.approx(0.5)

    def test_attachment_at_03_no_social_change(self) -> None:
        fi = FearIndex(attachment_risk=0.3)
        drv = _drv(social=0.5)
        result = fear_drive_boost(fi, drv)
        assert result.social == pytest.approx(0.5)

    def test_attachment_below_03_no_social_change(self) -> None:
        fi = FearIndex(attachment_risk=0.2)
        drv = _drv(social=0.4)
        result = fear_drive_boost(fi, drv)
        assert result.social == pytest.approx(0.4)


# ── 10. fear_drive_boost above attachment threshold ──────────


class TestFearDriveBoostAboveAttachmentThreshold:
    def test_attachment_above_03_social_boosted(self) -> None:
        fi = FearIndex(attachment_risk=0.6)
        drv = _drv(social=0.5)
        result = fear_drive_boost(fi, drv)
        # social += (0.6 - 0.3) * 0.3 = 0.09
        assert result.social == pytest.approx(0.5 + 0.09)

    def test_attachment_max_social_boost(self) -> None:
        fi = FearIndex(attachment_risk=1.0)
        drv = _drv(social=0.5)
        result = fear_drive_boost(fi, drv)
        # social += (1.0 - 0.3) * 0.3 = 0.21
        assert result.social == pytest.approx(0.5 + 0.21)

    def test_attachment_barely_above_03(self) -> None:
        fi = FearIndex(attachment_risk=0.31)
        drv = _drv(social=0.5)
        result = fear_drive_boost(fi, drv)
        # social += (0.31 - 0.3) * 0.3 = 0.003
        assert result.social == pytest.approx(0.5 + 0.003, abs=1e-6)


# ── 11. fear_drive_boost below identity threshold ────────────


class TestFearDriveBoostBelowIdentityThreshold:
    def test_identity_zero_no_expression_change(self) -> None:
        fi = FearIndex(identity_risk=0.0)
        drv = _drv(expression=0.5)
        result = fear_drive_boost(fi, drv)
        assert result.expression == pytest.approx(0.5)

    def test_identity_at_03_no_expression_change(self) -> None:
        fi = FearIndex(identity_risk=0.3)
        drv = _drv(expression=0.5)
        result = fear_drive_boost(fi, drv)
        assert result.expression == pytest.approx(0.5)

    def test_identity_below_03_no_expression_change(self) -> None:
        fi = FearIndex(identity_risk=0.15)
        drv = _drv(expression=0.3)
        result = fear_drive_boost(fi, drv)
        assert result.expression == pytest.approx(0.3)


# ── 12. fear_drive_boost above identity threshold ────────────


class TestFearDriveBoostAboveIdentityThreshold:
    def test_identity_above_03_expression_boosted(self) -> None:
        fi = FearIndex(identity_risk=0.7)
        drv = _drv(expression=0.5)
        result = fear_drive_boost(fi, drv)
        # expression += (0.7 - 0.3) * 0.3 = 0.12
        assert result.expression == pytest.approx(0.5 + 0.12)

    def test_identity_max_expression_boost(self) -> None:
        fi = FearIndex(identity_risk=1.0)
        drv = _drv(expression=0.5)
        result = fear_drive_boost(fi, drv)
        # expression += (1.0 - 0.3) * 0.3 = 0.21
        assert result.expression == pytest.approx(0.5 + 0.21)


# ── 13. fear_drive_boost both active simultaneously ──────────


class TestFearDriveBoostBothActive:
    def test_both_thresholds_exceeded(self) -> None:
        fi = FearIndex(identity_risk=0.8, attachment_risk=0.6)
        drv = _drv(social=0.4, expression=0.3, curiosity=0.5)
        result = fear_drive_boost(fi, drv)
        # social += (0.6 - 0.3) * 0.3 = 0.09
        # expression += (0.8 - 0.3) * 0.3 = 0.15
        assert result.social == pytest.approx(0.4 + 0.09)
        assert result.expression == pytest.approx(0.3 + 0.15)
        assert result.curiosity == pytest.approx(0.5)

    def test_both_max(self) -> None:
        fi = FearIndex(identity_risk=1.0, attachment_risk=1.0)
        drv = _drv(social=0.5, expression=0.5, curiosity=0.5)
        result = fear_drive_boost(fi, drv)
        assert result.social == pytest.approx(0.5 + 0.21)
        assert result.expression == pytest.approx(0.5 + 0.21)
        assert result.curiosity == pytest.approx(0.5)


# ── 14. fear_drive_boost clamping at 1.0 ─────────────────────


class TestFearDriveBoostClamping:
    def test_social_clamped(self) -> None:
        fi = FearIndex(attachment_risk=1.0)
        drv = _drv(social=0.95)
        result = fear_drive_boost(fi, drv)
        # social += 0.21 => 1.16 => clamp to 1.0
        assert result.social == pytest.approx(1.0)

    def test_expression_clamped(self) -> None:
        fi = FearIndex(identity_risk=1.0)
        drv = _drv(expression=0.95)
        result = fear_drive_boost(fi, drv)
        # expression += 0.21 => 1.16 => clamp to 1.0
        assert result.expression == pytest.approx(1.0)

    def test_both_clamped(self) -> None:
        fi = FearIndex(identity_risk=1.0, attachment_risk=1.0)
        drv = _drv(social=0.9, expression=0.9)
        result = fear_drive_boost(fi, drv)
        assert result.social == pytest.approx(1.0)
        assert result.expression == pytest.approx(1.0)


# ── 15. Immutability: input EmotionVector/DriveVector unchanged


class TestImmutability:
    def test_emotion_vector_input_unchanged(self) -> None:
        fi = FearIndex(
            identity_risk=1.0,
            attachment_risk=1.0,
            continuity_risk=1.0,
            projection_risk=1.0,
        )
        emo = _emo(fear=0.1, sorrow=0.1, joy=0.5)
        original_fear = emo.fear
        original_sorrow = emo.sorrow
        original_joy = emo.joy
        _ = fear_emotion_boost(fi, emo)
        assert emo.fear == pytest.approx(original_fear)
        assert emo.sorrow == pytest.approx(original_sorrow)
        assert emo.joy == pytest.approx(original_joy)

    def test_drive_vector_input_unchanged(self) -> None:
        fi = FearIndex(identity_risk=0.8, attachment_risk=0.8)
        drv = _drv(social=0.4, expression=0.3, curiosity=0.5)
        original_social = drv.social
        original_expression = drv.expression
        original_curiosity = drv.curiosity
        _ = fear_drive_boost(fi, drv)
        assert drv.social == pytest.approx(original_social)
        assert drv.expression == pytest.approx(original_expression)
        assert drv.curiosity == pytest.approx(original_curiosity)

    def test_emotion_boost_returns_new_instance(self) -> None:
        fi = FearIndex(
            identity_risk=0.5,
            attachment_risk=0.5,
            continuity_risk=0.5,
            projection_risk=0.5,
        )
        emo = _emo(fear=0.1, sorrow=0.1)
        result = fear_emotion_boost(fi, emo)
        assert result is not emo

    def test_drive_boost_returns_new_instance(self) -> None:
        fi = FearIndex(identity_risk=0.5, attachment_risk=0.5)
        drv = _drv(social=0.4, expression=0.3)
        result = fear_drive_boost(fi, drv)
        assert result is not drv


# ── Bonus: _clamp01 unit tests ───────────────────────────────


class TestClamp01:
    def test_within_range(self) -> None:
        assert _clamp01(0.5) == pytest.approx(0.5)

    def test_at_zero(self) -> None:
        assert _clamp01(0.0) == pytest.approx(0.0)

    def test_at_one(self) -> None:
        assert _clamp01(1.0) == pytest.approx(1.0)

    def test_below_zero(self) -> None:
        assert _clamp01(-0.5) == pytest.approx(0.0)

    def test_above_one(self) -> None:
        assert _clamp01(1.5) == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        assert _clamp01(-100.0) == pytest.approx(0.0)

    def test_large_positive(self) -> None:
        assert _clamp01(100.0) == pytest.approx(1.0)
