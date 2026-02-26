"""
tests/test_mood_autonomy.py - ムード自律化のテスト

ムードの自律化機能（reaction.py内の状態依存的ムード更新）をテストする。

テスト項目:
1. MoodContextInputs のデフォルト値
2. 段階1: 多入力源からの目標値導出
   - 感情由来の目標値
   - ドライブ由来の目標値
   - 目的階層由来の目標値
   - 恐怖由来のarousal目標値
   - 入力不在時のゼロ寄与
   - 帯域制限の遵守
3. 段階2: 追従速度の導出
   - valence追従速度の覚醒度依存性
   - arousal追従速度の恐怖指数依存性
   - 追従速度の帯域制限
   - 時間密度による修正
4. 段階3: compute_autonomous_mood 統合テスト
   - valence/arousal独立更新
   - 変動量上限の遵守
   - 入力不在時のムード維持
5. react()統合テスト
   - 既存のムード更新が置き換わっていること
   - mood_context渡し時の動作
   - mood_context省略時のデフォルト構成
   - 既存テスト (test_reaction.py) との互換性
6. 安全弁テスト
   - 安全弁1: 入力源別帯域制限
   - 安全弁2: 追従速度帯域制限
   - 安全弁3: 変動量上限
   - 安全弁4: 入力不在時中立化
   - 安全弁5: 非蓄積 (純粋関数)
   - 安全弁6: valence/arousal独立性
"""

import copy

import pytest

from psyche.reaction import (
    MoodContextInputs,
    _MOOD_BAND,
    _MOOD_DELTA_LIMIT,
    _TRACKING_SPEED_MAX,
    _TRACKING_SPEED_MIN,
    _derive_mood_targets,
    _derive_tracking_speeds,
    compute_autonomous_mood,
    react,
)
from psyche.responsibility import ResponsibilityInfluence
from psyche.state import DriveVector, EmotionVector, Mood, Percept, PsycheState


# ── Helpers ───────────────────────────────────────────────────

def _default_ctx(**overrides) -> MoodContextInputs:
    """Create a MoodContextInputs with sensible defaults."""
    kw = dict(
        emotions={"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
                  "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
        drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        current_valence=0.0,
        current_arousal=0.3,
        fear_level=0.0,
        delta_time=1.0,
    )
    kw.update(overrides)
    return MoodContextInputs(**kw)


def _zero_emotion_state(**overrides) -> PsycheState:
    """Create a PsycheState with all emotions at 0.0 and default drives/mood."""
    emo_kw = {k: 0.0 for k in EmotionVector.model_fields}
    emo_kw.update(overrides.pop("emotions", {}))
    return PsycheState(
        emotions=EmotionVector(**emo_kw),
        drives=overrides.pop("drives", DriveVector(social=0.5, curiosity=0.5, expression=0.5)),
        mood=overrides.pop("mood", Mood(valence=0.0, arousal=0.3)),
        **overrides,
    )


def _neutral_percept(**overrides) -> Percept:
    """Create a neutral percept with optional overrides."""
    kw = dict(text="", meaning="", emotion="neutral", intent="unknown",
              topics=[], sentiment=0.0, emotion_valence=0.0)
    kw.update(overrides)
    return Percept(**kw)


# =============================================================================
# 1. MoodContextInputs defaults
# =============================================================================

class TestMoodContextInputsDefaults:
    def test_default_emotions_none(self):
        ctx = MoodContextInputs()
        assert ctx.emotions is None

    def test_default_drives_none(self):
        ctx = MoodContextInputs()
        assert ctx.drives is None

    def test_default_valence_zero(self):
        ctx = MoodContextInputs()
        assert ctx.current_valence == 0.0

    def test_default_arousal(self):
        ctx = MoodContextInputs()
        assert ctx.current_arousal == 0.3

    def test_default_fear_zero(self):
        ctx = MoodContextInputs()
        assert ctx.fear_level == 0.0

    def test_default_goals_absent(self):
        ctx = MoodContextInputs()
        assert ctx.has_transient_goal is False
        assert ctx.persistent_commitment_count == 0
        assert ctx.has_scoped_goal is False

    def test_default_time_density_none(self):
        ctx = MoodContextInputs()
        assert ctx.time_density_label is None

    def test_default_responsibility_zero(self):
        ctx = MoodContextInputs()
        assert ctx.responsibility_anxiety == 0.0


# =============================================================================
# 2. Stage 1: Target derivation
# =============================================================================

class TestMoodTargetDerivation:
    """段階1: 多入力源からの目標値導出テスト。"""

    def test_zero_emotions_zero_target_valence(self):
        """全感情ゼロの場合、感情由来のvalence目標もゼロ。"""
        ctx = _default_ctx()
        tv, ta = _derive_mood_targets(ctx)
        # 感情由来は0, ドライブ由来は中立(0.5なので0), 目的なし, 恐怖なし
        assert tv == pytest.approx(0.0, abs=1e-6)

    def test_positive_emotions_positive_valence_target(self):
        """正の感情はvalence目標を正に押す。"""
        ctx = _default_ctx(emotions={"joy": 0.8, "love": 0.3, "fun": 0.2,
                                     "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                     "surprise": 0.0})
        tv, _ = _derive_mood_targets(ctx)
        assert tv > 0.0

    def test_negative_emotions_negative_valence_target(self):
        """負の感情はvalence目標を負に押す。"""
        ctx = _default_ctx(emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                                     "sorrow": 0.7, "anger": 0.3, "fear": 0.2,
                                     "surprise": 0.0})
        tv, _ = _derive_mood_targets(ctx)
        assert tv < 0.0

    def test_high_emotion_high_arousal_target(self):
        """高い感情強度はarousal目標を上げる。"""
        ctx_zero = _default_ctx(emotions={"joy": 0.0, "anger": 0.0, "sorrow": 0.0,
                                          "fear": 0.0, "surprise": 0.0, "love": 0.0,
                                          "fun": 0.0})
        ctx_high = _default_ctx(emotions={"joy": 0.9, "anger": 0.0, "sorrow": 0.0,
                                          "fear": 0.0, "surprise": 0.0, "love": 0.0,
                                          "fun": 0.0})
        _, ta_zero = _derive_mood_targets(ctx_zero)
        _, ta_high = _derive_mood_targets(ctx_high)
        assert ta_high > ta_zero

    def test_drives_affect_valence_target(self):
        """ドライブの充足度がvalence目標に影響する。"""
        ctx_low = _default_ctx(drives={"social": 0.1, "curiosity": 0.1, "expression": 0.5})
        ctx_high = _default_ctx(drives={"social": 0.9, "curiosity": 0.9, "expression": 0.5})
        tv_low, _ = _derive_mood_targets(ctx_low)
        tv_high, _ = _derive_mood_targets(ctx_high)
        assert tv_high > tv_low

    def test_drives_affect_arousal_target(self):
        """ドライブ（表出）がarousal目標に影響する。"""
        ctx_low = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.1})
        ctx_high = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.9})
        _, ta_low = _derive_mood_targets(ctx_low)
        _, ta_high = _derive_mood_targets(ctx_high)
        assert ta_high > ta_low

    def test_goal_presence_positive_valence_contribution(self):
        """目的の存在はvalence目標に微弱な正の寄与。"""
        ctx_no_goal = _default_ctx()
        ctx_with_goals = _default_ctx(has_transient_goal=True,
                                      persistent_commitment_count=2,
                                      has_scoped_goal=True)
        tv_no, _ = _derive_mood_targets(ctx_no_goal)
        tv_with, _ = _derive_mood_targets(ctx_with_goals)
        assert tv_with > tv_no

    def test_goal_presence_small_arousal_contribution(self):
        """目的の存在はarousal目標にわずかな正の寄与。"""
        ctx_no_goal = _default_ctx()
        ctx_with_goals = _default_ctx(has_transient_goal=True,
                                      persistent_commitment_count=1)
        _, ta_no = _derive_mood_targets(ctx_no_goal)
        _, ta_with = _derive_mood_targets(ctx_with_goals)
        assert ta_with > ta_no

    def test_fear_increases_arousal_target_only(self):
        """恐怖指数はarousal目標のみを上げる（valenceには直接寄与しない）。"""
        ctx_no_fear = _default_ctx(fear_level=0.0)
        ctx_fear = _default_ctx(fear_level=0.6)
        tv_no, ta_no = _derive_mood_targets(ctx_no_fear)
        tv_fear, ta_fear = _derive_mood_targets(ctx_fear)
        # arousalが上がる
        assert ta_fear > ta_no
        # valenceは同じ（恐怖由来のvalence帯域は0）
        assert tv_fear == pytest.approx(tv_no, abs=1e-9)

    def test_all_inputs_absent_zero_targets(self):
        """全入力が不在（None/ゼロ）の場合、目標はゼロ。"""
        ctx = MoodContextInputs()  # all defaults
        tv, ta = _derive_mood_targets(ctx)
        assert tv == pytest.approx(0.0, abs=1e-9)
        assert ta == pytest.approx(0.0, abs=1e-9)

    def test_emotion_valence_band_limit(self):
        """感情由来のvalence寄与は帯域上限内に収まる。"""
        ctx = _default_ctx(emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                                     "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                                     "surprise": 0.0})
        tv, _ = _derive_mood_targets(ctx)
        band = _MOOD_BAND["emotion"]["valence"]
        # 感情由来のvalence寄与のみ（ドライブは中立）なので帯域内
        assert abs(tv) <= band + 1e-9

    def test_emotion_arousal_band_limit(self):
        """感情由来のarousal寄与は帯域上限内に収まる。"""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        )
        _, ta = _derive_mood_targets(ctx)
        # arousal寄与は感情帯域内（ドライブは中立なので0）
        assert ta <= _MOOD_BAND["emotion"]["arousal"] + 1e-9

    def test_drive_band_limit(self):
        """ドライブ由来の寄与は帯域上限内に収まる。"""
        ctx = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
        )
        tv, ta = _derive_mood_targets(ctx)
        assert abs(tv) <= _MOOD_BAND["drive"]["valence"] + 1e-9
        assert abs(ta) <= _MOOD_BAND["drive"]["arousal"] + 1e-9

    def test_goal_band_limit(self):
        """目的階層由来の寄与は帯域上限内に収まる。"""
        ctx = _default_ctx(
            has_transient_goal=True,
            persistent_commitment_count=10,  # extreme
            has_scoped_goal=True,
        )
        # Remove emotion and drive contributions for isolation
        ctx.emotions = {"joy": 0.0, "love": 0.0, "fun": 0.0,
                        "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                        "surprise": 0.0}
        ctx.drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        tv, ta = _derive_mood_targets(ctx)
        assert abs(tv) <= _MOOD_BAND["goal"]["valence"] + 1e-9
        assert abs(ta) <= _MOOD_BAND["goal"]["arousal"] + 1e-9

    def test_fear_arousal_band_limit(self):
        """恐怖由来のarousal寄与は帯域上限内に収まる。"""
        ctx = _default_ctx(fear_level=1.0)
        ctx.emotions = {"joy": 0.0, "love": 0.0, "fun": 0.0,
                        "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                        "surprise": 0.0}
        ctx.drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        _, ta = _derive_mood_targets(ctx)
        assert ta <= _MOOD_BAND["fear"]["arousal"] + 1e-9

    def test_goal_count_capped_at_3(self):
        """目的カウントは3で上限（それ以上でも寄与は増えない）。"""
        ctx3 = _default_ctx(persistent_commitment_count=3)
        ctx10 = _default_ctx(persistent_commitment_count=10)
        ctx3.emotions = {"joy": 0.0, "love": 0.0, "fun": 0.0,
                         "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                         "surprise": 0.0}
        ctx10.emotions = {"joy": 0.0, "love": 0.0, "fun": 0.0,
                          "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                          "surprise": 0.0}
        ctx3.drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        ctx10.drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.5}
        tv3, ta3 = _derive_mood_targets(ctx3)
        tv10, ta10 = _derive_mood_targets(ctx10)
        assert tv3 == pytest.approx(tv10, abs=1e-9)
        assert ta3 == pytest.approx(ta10, abs=1e-9)


# =============================================================================
# 3. Stage 2: Tracking speed derivation
# =============================================================================

class TestTrackingSpeedDerivation:
    """段階2: 追従速度の導出テスト。"""

    def test_default_speed_near_010(self):
        """デフォルト条件での追従速度は0.10付近。"""
        ctx = _default_ctx()
        v_speed, a_speed = _derive_tracking_speeds(ctx)
        assert v_speed == pytest.approx(0.10, abs=0.02)
        assert a_speed == pytest.approx(0.10, abs=0.02)

    def test_high_arousal_increases_valence_speed(self):
        """高覚醒時はvalence追従速度が速い。"""
        ctx_low = _default_ctx(current_arousal=0.2)
        ctx_high = _default_ctx(current_arousal=0.8)
        v_low, _ = _derive_tracking_speeds(ctx_low)
        v_high, _ = _derive_tracking_speeds(ctx_high)
        assert v_high > v_low

    def test_low_arousal_decreases_valence_speed(self):
        """低覚醒時はvalence追従速度が遅い。"""
        ctx_mid = _default_ctx(current_arousal=0.4)
        ctx_low = _default_ctx(current_arousal=0.1)
        v_mid, _ = _derive_tracking_speeds(ctx_mid)
        v_low, _ = _derive_tracking_speeds(ctx_low)
        assert v_low < v_mid

    def test_high_fear_increases_arousal_speed(self):
        """高恐怖時はarousal追従速度が速い。"""
        ctx_no_fear = _default_ctx(fear_level=0.0)
        ctx_fear = _default_ctx(fear_level=0.7)
        _, a_no = _derive_tracking_speeds(ctx_no_fear)
        _, a_fear = _derive_tracking_speeds(ctx_fear)
        assert a_fear > a_no

    def test_emotion_range_affects_arousal_speed(self):
        """感情変動幅が大きいとarousal追従が速い。"""
        ctx_flat = _default_ctx(emotions={"joy": 0.3, "anger": 0.3, "sorrow": 0.3,
                                          "fear": 0.3, "surprise": 0.3, "love": 0.3,
                                          "fun": 0.3})
        ctx_varied = _default_ctx(emotions={"joy": 0.9, "anger": 0.0, "sorrow": 0.0,
                                            "fear": 0.0, "surprise": 0.0, "love": 0.0,
                                            "fun": 0.0})
        _, a_flat = _derive_tracking_speeds(ctx_flat)
        _, a_varied = _derive_tracking_speeds(ctx_varied)
        assert a_varied > a_flat

    def test_speed_min_bound(self):
        """追従速度は下限以上。"""
        # 極端に低い条件
        ctx = _default_ctx(current_arousal=0.0, fear_level=0.0,
                           time_density_label="sparse")
        ctx.drives = {"social": 0.5, "curiosity": 0.5, "expression": 0.1}
        v_speed, a_speed = _derive_tracking_speeds(ctx)
        assert v_speed >= _TRACKING_SPEED_MIN
        assert a_speed >= _TRACKING_SPEED_MIN

    def test_speed_max_bound(self):
        """追従速度は上限以下。"""
        # 極端に高い条件
        ctx = _default_ctx(current_arousal=1.0, fear_level=1.0,
                           time_density_label="dense")
        ctx.drives = {"social": 0.5, "curiosity": 0.5, "expression": 1.0}
        ctx.emotions = {"joy": 1.0, "anger": 0.0, "sorrow": 0.0,
                        "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0}
        v_speed, a_speed = _derive_tracking_speeds(ctx)
        assert v_speed <= _TRACKING_SPEED_MAX
        assert a_speed <= _TRACKING_SPEED_MAX

    def test_sparse_time_slows_valence(self):
        """sparse時間密度はvalence追従を遅くする。"""
        ctx_normal = _default_ctx(time_density_label=None)
        ctx_sparse = _default_ctx(time_density_label="sparse")
        v_normal, _ = _derive_tracking_speeds(ctx_normal)
        v_sparse, _ = _derive_tracking_speeds(ctx_sparse)
        assert v_sparse < v_normal

    def test_dense_time_speeds_valence(self):
        """dense時間密度はvalence追従を速くする。"""
        ctx_normal = _default_ctx(time_density_label=None)
        ctx_dense = _default_ctx(time_density_label="dense")
        v_normal, _ = _derive_tracking_speeds(ctx_normal)
        v_dense, _ = _derive_tracking_speeds(ctx_dense)
        assert v_dense > v_normal

    def test_expression_drive_affects_valence_speed(self):
        """表出ドライブがvalence追従速度に影響する。"""
        ctx_low = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.1})
        ctx_high = _default_ctx(drives={"social": 0.5, "curiosity": 0.5, "expression": 0.9})
        v_low, _ = _derive_tracking_speeds(ctx_low)
        v_high, _ = _derive_tracking_speeds(ctx_high)
        assert v_high > v_low


# =============================================================================
# 4. Stage 3: compute_autonomous_mood integration
# =============================================================================

class TestComputeAutonomousMood:
    """段階3: compute_autonomous_mood 統合テスト。"""

    def test_positive_emotions_increase_valence(self):
        """正の感情でvalenceが正方向に移動する。"""
        ctx = _default_ctx(
            emotions={"joy": 0.8, "love": 0.3, "fun": 0.2,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=0.0,
        )
        new_v, _ = compute_autonomous_mood(ctx)
        assert new_v > 0.0

    def test_negative_emotions_decrease_valence(self):
        """負の感情でvalenceが負方向に移動する。"""
        ctx = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.7, "anger": 0.3, "fear": 0.2,
                      "surprise": 0.0},
            current_valence=0.0,
        )
        new_v, _ = compute_autonomous_mood(ctx)
        assert new_v < 0.0

    def test_high_emotion_increases_arousal(self):
        """高い感情強度でarousalが上がる（ゼロからの上昇）。"""
        ctx = _default_ctx(
            emotions={"joy": 0.9, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_arousal=0.0,
        )
        _, new_a = compute_autonomous_mood(ctx)
        assert new_a > 0.0

    def test_zero_input_maintains_mood(self):
        """全入力がゼロの場合、ムードは前回値に近い。"""
        ctx = _default_ctx(current_valence=0.3, current_arousal=0.5)
        new_v, new_a = compute_autonomous_mood(ctx)
        # 目標がほぼ0なので前回値からゼロ方向に少し動く
        assert new_v < 0.3  # 目標は0付近なのでvalenceは減少
        assert new_a < 0.5  # 目標は0付近なのでarousalは減少

    def test_all_inputs_none_stable(self):
        """全入力がNone/デフォルトの場合もクラッシュしない。"""
        ctx = MoodContextInputs()
        new_v, new_a = compute_autonomous_mood(ctx)
        # 目標=0, current_valence=0 -> delta=0 -> 変化なし
        assert new_v == pytest.approx(0.0, abs=1e-9)
        # current_arousal=0.3, 目標=0 -> arousalは下がる
        assert new_a < 0.3

    def test_delta_limit_valence(self):
        """valence変動量が上限内に収まる。"""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=-1.0,
            current_arousal=1.0,  # 高覚醒で追従が速い
        )
        new_v, _ = compute_autonomous_mood(ctx)
        delta = abs(new_v - (-1.0))
        assert delta <= _MOOD_DELTA_LIMIT + 1e-9

    def test_delta_limit_arousal(self):
        """arousal変動量が上限内に収まる。"""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_arousal=0.0,
            fear_level=1.0,  # 高恐怖で追従が速い
        )
        _, new_a = compute_autonomous_mood(ctx)
        delta = abs(new_a - 0.0)
        assert delta <= _MOOD_DELTA_LIMIT + 1e-9

    def test_valence_arousal_independent(self):
        """valenceとarousalは独立に更新される（安全弁6）。"""
        # 感情でvalenceだけを動かす条件
        ctx = _default_ctx(
            emotions={"joy": 0.8, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=0.0,
            current_arousal=0.0,
        )
        new_v, new_a = compute_autonomous_mood(ctx)
        # 両方動くが独立に（valenceの変化がarousalを直接支配しない）
        assert new_v > 0.0  # 正の感情でvalence上昇
        assert new_a > 0.0  # 感情強度でarousal上昇
        # 変化量は独立の追従速度で導出される
        # (これはもっと詳細にテストするが、ここでは独立性の確認)

    def test_different_drive_different_mood(self):
        """同じ感情でもドライブが異なればムード変化が異なる。"""
        ctx1 = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.1, "curiosity": 0.1, "expression": 0.1},
        )
        ctx2 = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.9, "curiosity": 0.9, "expression": 0.9},
        )
        v1, a1 = compute_autonomous_mood(ctx1)
        v2, a2 = compute_autonomous_mood(ctx2)
        # ドライブが異なるのでvalenceとarousalの変化が異なる
        assert v1 != pytest.approx(v2, abs=1e-6)

    def test_pure_function_no_side_effects(self):
        """compute_autonomous_moodは純粋関数（入力を変更しない）。"""
        ctx = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=0.3,
            current_arousal=0.5,
        )
        original_valence = ctx.current_valence
        original_arousal = ctx.current_arousal
        original_emotions = ctx.emotions.copy()

        compute_autonomous_mood(ctx)

        assert ctx.current_valence == original_valence
        assert ctx.current_arousal == original_arousal
        assert ctx.emotions == original_emotions


# =============================================================================
# 5. react() integration tests
# =============================================================================

class TestReactMoodIntegration:
    """react()統合テスト: ムード自律更新が正しく統合されていること。"""

    def test_positive_emotions_drift_valence_up(self):
        """Joy pushes mood valence toward positive (existing behavior preserved)."""
        state = _zero_emotion_state(emotions={"joy": 0.8}, mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        assert result.mood.valence > 0.0

    def test_negative_emotions_drift_valence_down(self):
        """Sorrow pushes mood valence toward negative (existing behavior preserved)."""
        state = _zero_emotion_state(emotions={"sorrow": 0.8}, mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        assert result.mood.valence < 0.0

    def test_arousal_tracks_emotion_intensity(self):
        """Arousal moves toward emotion intensity (existing behavior preserved)."""
        state = _zero_emotion_state(emotions={"joy": 0.9}, mood=Mood(valence=0.0, arousal=0.0))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        assert result.mood.arousal > 0.0

    def test_mood_inertia(self):
        """Mood changes slowly, not instantly tracking inputs."""
        state = _zero_emotion_state(emotions={"joy": 1.0}, mood=Mood(valence=0.0, arousal=0.0))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0)
        # Should not instantly jump to the target
        assert result.mood.valence < 0.2

    def test_explicit_mood_context_used(self):
        """When mood_context is explicitly provided, it is used."""
        state = _zero_emotion_state(mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()

        # Provide mood context with goals and fear
        ctx = MoodContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=2,
            fear_level=0.5,
        )
        result = react(percept, state, delta_time=1.0, mood_context=ctx)
        # Should not crash
        assert isinstance(result, PsycheState)
        assert -1.0 <= result.mood.valence <= 1.0
        assert 0.0 <= result.mood.arousal <= 1.0

    def test_mood_context_none_default_construction(self):
        """When mood_context is None, a default is constructed internally."""
        state = _zero_emotion_state(emotions={"joy": 0.5}, mood=Mood(valence=0.0, arousal=0.3))
        percept = _neutral_percept()
        result = react(percept, state, delta_time=1.0, mood_context=None)
        assert isinstance(result, PsycheState)
        assert result.mood.valence > 0.0  # Joy should push valence up

    def test_responsibility_mood_penalty_still_applied(self):
        """Responsibility mood penalty is still applied on top of autonomous mood."""
        state = _zero_emotion_state(mood=Mood(valence=0.5, arousal=0.3))
        percept = _neutral_percept()
        influence = ResponsibilityInfluence(anxiety_baseline=0.2, fear_amplification=0.0)

        result_with = react(percept, state, delta_time=1.0, responsibility_influence=influence)
        result_without = react(percept, state, delta_time=1.0)

        # With responsibility, valence should be lower
        assert result_with.mood.valence < result_without.mood.valence

    def test_valence_clamped(self):
        """Mood valence stays within [-1, 1]."""
        state = _zero_emotion_state(
            emotions={"sorrow": 1.0, "anger": 1.0, "fear": 1.0},
            mood=Mood(valence=-0.9, arousal=0.3),
        )
        percept = _neutral_percept(emotion="sad", emotion_valence=-1.0)
        influence = ResponsibilityInfluence(anxiety_baseline=0.3, fear_amplification=0.5)
        result = react(percept, state, delta_time=1.0,
                       responsibility_influence=influence, amplitude_modifier=2.0)
        assert -1.0 <= result.mood.valence <= 1.0

    def test_arousal_clamped(self):
        """Mood arousal stays within [0, 1]."""
        state = _zero_emotion_state(
            emotions={"joy": 1.0},
            mood=Mood(valence=0.0, arousal=0.99),
        )
        percept = _neutral_percept(emotion="happy", emotion_valence=1.0)
        result = react(percept, state, delta_time=1.0, amplitude_modifier=2.0)
        assert 0.0 <= result.mood.arousal <= 1.0

    def test_immutability_of_input_state(self):
        """react() must not mutate the input PsycheState."""
        state = _zero_emotion_state(
            emotions={"joy": 0.5, "sorrow": 0.3},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.0, arousal=0.3),
        )
        original_valence = state.mood.valence
        original_arousal = state.mood.arousal

        percept = _neutral_percept(emotion="happy", emotion_valence=0.5)
        _ = react(percept, state, delta_time=1.0)

        assert state.mood.valence == original_valence
        assert state.mood.arousal == original_arousal

    def test_full_pipeline_with_all_features(self):
        """Smoke test: react with mood_context containing all fields."""
        from psyche.pillars import FearIndex
        fi = FearIndex(identity_risk=0.3, attachment_risk=0.3,
                       continuity_risk=0.2, projection_risk=0.2)
        state = _zero_emotion_state(
            emotions={"joy": 0.3, "fear": 0.2},
            drives=DriveVector(social=0.5, curiosity=0.5, expression=0.5),
            mood=Mood(valence=0.1, arousal=0.3),
            fear_index=fi,
        )
        percept = _neutral_percept(emotion="happy", emotion_valence=0.5, intent="sharing")
        influence = ResponsibilityInfluence(anxiety_baseline=0.1, fear_amplification=0.2)
        mood_ctx = MoodContextInputs(
            has_transient_goal=True,
            persistent_commitment_count=1,
            has_scoped_goal=True,
            time_density_label="dense",
        )
        result = react(
            percept, state, delta_time=2.0,
            responsibility_influence=influence,
            amplitude_modifier=1.3,
            mood_context=mood_ctx,
        )
        assert isinstance(result, PsycheState)
        for field in EmotionVector.model_fields:
            val = getattr(result.emotions, field)
            assert 0.0 <= val <= 1.0, f"emotion {field} = {val}"
        assert -1.0 <= result.mood.valence <= 1.0
        assert 0.0 <= result.mood.arousal <= 1.0


# =============================================================================
# 6. Safety valve tests
# =============================================================================

class TestSafetyValves:
    """安全弁の検証。"""

    def test_sv1_emotion_band_limit(self):
        """安全弁1: 感情由来の寄与が帯域上限内。"""
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        )
        tv, ta = _derive_mood_targets(ctx)
        # Total should be within sum of all band limits
        total_v_max = sum(b["valence"] for b in _MOOD_BAND.values())
        total_a_max = sum(b["arousal"] for b in _MOOD_BAND.values())
        assert abs(tv) <= total_v_max + 1e-9
        assert abs(ta) <= total_a_max + 1e-9

    def test_sv2_speed_always_in_range(self):
        """安全弁2: 追従速度が常に帯域内。"""
        test_cases = [
            _default_ctx(current_arousal=0.0, fear_level=0.0, time_density_label="sparse"),
            _default_ctx(current_arousal=1.0, fear_level=1.0, time_density_label="dense"),
            _default_ctx(current_arousal=0.5, fear_level=0.5),
            _default_ctx(),
        ]
        for ctx in test_cases:
            v, a = _derive_tracking_speeds(ctx)
            assert _TRACKING_SPEED_MIN <= v <= _TRACKING_SPEED_MAX, f"valence speed {v}"
            assert _TRACKING_SPEED_MIN <= a <= _TRACKING_SPEED_MAX, f"arousal speed {a}"

    def test_sv3_delta_limit(self):
        """安全弁3: 1ティックあたりの変動量が上限内。"""
        # Extreme conditions
        ctx = _default_ctx(
            emotions={"joy": 1.0, "love": 1.0, "fun": 1.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=-1.0,
            current_arousal=0.0,
            fear_level=1.0,
        )
        new_v, new_a = compute_autonomous_mood(ctx)
        assert abs(new_v - (-1.0)) <= _MOOD_DELTA_LIMIT + 1e-9
        assert abs(new_a - 0.0) <= _MOOD_DELTA_LIMIT + 1e-9

    def test_sv4_null_inputs_neutral(self):
        """安全弁4: 全入力がNoneの場合、寄与ゼロ。"""
        ctx = MoodContextInputs(current_valence=0.0, current_arousal=0.0)
        new_v, new_a = compute_autonomous_mood(ctx)
        # 目標=0, current=0 -> 変化なし
        assert new_v == pytest.approx(0.0, abs=1e-9)
        assert new_a == pytest.approx(0.0, abs=1e-9)

    def test_sv5_no_state_accumulation(self):
        """安全弁5: 同じ入力に対して常に同じ出力（非蓄積）。"""
        ctx = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=0.2,
            current_arousal=0.4,
        )
        result1 = compute_autonomous_mood(ctx)
        result2 = compute_autonomous_mood(ctx)
        result3 = compute_autonomous_mood(ctx)
        assert result1[0] == pytest.approx(result2[0], abs=1e-12)
        assert result1[1] == pytest.approx(result2[1], abs=1e-12)
        assert result2[0] == pytest.approx(result3[0], abs=1e-12)
        assert result2[1] == pytest.approx(result3[1], abs=1e-12)

    def test_sv6_valence_arousal_independent_update(self):
        """安全弁6: valenceの変化がarousalに直接影響しない。"""
        # 同じ入力で、current_valenceだけ変えた場合、arousalの変化は同じ
        ctx1 = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=-0.5,
            current_arousal=0.3,
        )
        ctx2 = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            current_valence=0.5,
            current_arousal=0.3,
        )
        _, a1 = compute_autonomous_mood(ctx1)
        _, a2 = compute_autonomous_mood(ctx2)
        # arousalの目標値と追従速度は同じ（valenceは入力に含まれない）
        assert a1 == pytest.approx(a2, abs=1e-12)

    def test_band_upper_limit_below_value_orientation(self):
        """帯域上限が価値方向性のmax_bias_strength(0.15)と同水準以下。"""
        max_bias_strength = 0.15
        for source, bands in _MOOD_BAND.items():
            for axis, limit in bands.items():
                assert limit <= max_bias_strength, (
                    f"Band {source}.{axis} = {limit} exceeds max_bias_strength {max_bias_strength}"
                )


# =============================================================================
# 7. Multi-input source interaction tests
# =============================================================================

class TestMultiInputInteraction:
    """複数入力源の相互作用テスト。"""

    def test_emotion_dominates_over_drive(self):
        """感情の寄与がドライブの寄与より大きい。"""
        # 感情のみ
        ctx_emo = _default_ctx(
            emotions={"joy": 0.8, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        )
        # ドライブのみ
        ctx_drv = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 1.0, "curiosity": 1.0, "expression": 1.0},
        )
        tv_emo, _ = _derive_mood_targets(ctx_emo)
        tv_drv, _ = _derive_mood_targets(ctx_drv)
        # 感情の最大寄与帯域は0.12、ドライブの最大寄与帯域は0.05
        assert abs(tv_emo) > abs(tv_drv)

    def test_additive_composition(self):
        """各入力源の寄与は加算的に合成される。"""
        ctx_emo_only = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.5, "curiosity": 0.5, "expression": 0.5},
        )
        ctx_both = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            drives={"social": 0.9, "curiosity": 0.9, "expression": 0.5},
            has_transient_goal=True,
        )
        tv_emo, _ = _derive_mood_targets(ctx_emo_only)
        tv_both, _ = _derive_mood_targets(ctx_both)
        # ドライブと目的が加算されているので、bothの方が大きい
        assert tv_both > tv_emo

    def test_same_emotion_different_goals_different_mood(self):
        """同じ感情でも目的の有無でムードが異なる。"""
        ctx_no_goal = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
        )
        ctx_with_goal = _default_ctx(
            emotions={"joy": 0.5, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.0,
                      "surprise": 0.0},
            has_transient_goal=True,
            persistent_commitment_count=2,
        )
        v_no, _ = compute_autonomous_mood(ctx_no_goal)
        v_with, _ = compute_autonomous_mood(ctx_with_goal)
        assert v_with > v_no

    def test_fear_and_high_emotion_high_arousal(self):
        """恐怖+高感情でarousalが高い。"""
        ctx = _default_ctx(
            emotions={"joy": 0.0, "love": 0.0, "fun": 0.0,
                      "sorrow": 0.0, "anger": 0.0, "fear": 0.8,
                      "surprise": 0.0},
            fear_level=0.6,
            current_arousal=0.1,
        )
        _, new_a = compute_autonomous_mood(ctx)
        # 恐怖指数と高い感情でarousalが上昇
        assert new_a > 0.1
