"""
tests/test_experience_driven_value_update.py

経験強度による価値更新帯域拡大のテスト。
設計書: design_experience_driven_value_update.md

テスト対象:
- _compute_experience_intensity: 経験強度係数の算出
- _compute_bandwidth_expansion_coefficient: 帯域拡大係数の算出
- _apply_experience_driven_value_update: 統合テスト（orchestratorモック経由）
- 安全弁の検証（絶対上限、冷却期間、confidence damping維持、enrichment遮断、蓄積禁止）
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from psyche.orchestrator_5tick_phases import (
    _compute_experience_intensity,
    _compute_bandwidth_expansion_coefficient,
    _apply_experience_driven_value_update,
    _EXP_BANDWIDTH_MAX_MULTIPLIER,
    _EXP_BANDWIDTH_MAX_DELTA_PER_DIM,
    _EXP_BANDWIDTH_COOLDOWN_TICKS,
)

from psyche.value_orientation import (
    ValueOrientation,
    ValueOrientationConfig,
    update_from_decision,
    generate_decision_signal,
    compute_effective_learning_rate,
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
    intensity_level: float = 0.8,
    valence: float = 0.5,
    primary_emotion: str = "joy",
    vividness: float = 0.9,
) -> EpisodeEntry:
    """Create a test EpisodeEntry with emotional companion."""
    return EpisodeEntry(
        episode_id="test_ep_001",
        episode_type=EpisodeType.EMOTIONAL_EVENT,
        summary="Test episode",
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
        vividness=vividness,
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
    emotion_intensity: float = 0.8,
    emotion_amplitude: float = 0.7,
    mood_arousal: float = 0.6,
    tick_count: int = 10,
    last_bandwidth_tick: int = -10,
    orientation: ValueOrientation = None,
    episodes_store: EpisodeStore = None,
    vo_config: ValueOrientationConfig = None,
) -> MagicMock:
    """Create a mock PsycheOrchestrator for testing."""
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

    # Episodes store
    if episodes_store is None:
        episodes_store = _make_episode_store(
            [_make_episode_entry(intensity_level=emotion_intensity)]
        )
    orch._last_episodes = episodes_store

    return orch


# =============================================================================
# Tests: _compute_experience_intensity (段階1)
# =============================================================================

class TestComputeExperienceIntensity:
    """経験強度係数の算出テスト。"""

    def test_all_high_returns_near_one(self):
        """全断面が高い場合、係数は1.0に近い。"""
        result = _compute_experience_intensity(1.0, 1.0, 1.0)
        assert result == 1.0

    def test_all_zero_returns_zero(self):
        """全断面がゼロの場合、係数はゼロ。"""
        result = _compute_experience_intensity(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_one_zero_makes_product_zero(self):
        """いずれかの断面がゼロの場合、乗算的結合によりゼロになる。"""
        assert _compute_experience_intensity(0.0, 0.8, 0.6) == 0.0
        assert _compute_experience_intensity(0.8, 0.0, 0.6) == 0.0
        assert _compute_experience_intensity(0.8, 0.6, 0.0) == 0.0

    def test_moderate_values(self):
        """中程度の値の乗算結合。"""
        result = _compute_experience_intensity(0.5, 0.5, 0.5)
        assert abs(result - 0.125) < 0.001

    def test_values_clamped_to_valid_range(self):
        """範囲外の値はクランプされる。"""
        result = _compute_experience_intensity(1.5, -0.5, 2.0)
        # clamped to (1.0, 0.0, 1.0) → 0.0
        assert result == 0.0

    def test_asymmetric_values(self):
        """非対称な入力でも正しく乗算される。"""
        result = _compute_experience_intensity(0.9, 0.3, 0.8)
        expected = 0.9 * 0.3 * 0.8
        assert abs(result - expected) < 0.001

    def test_low_intensity_low_result(self):
        """低い断面値は低い結果をもたらす。"""
        result = _compute_experience_intensity(0.1, 0.1, 0.1)
        assert result < 0.01


# =============================================================================
# Tests: _compute_bandwidth_expansion_coefficient (段階2)
# =============================================================================

class TestComputeBandwidthExpansionCoefficient:
    """帯域拡大係数の算出テスト。"""

    def test_zero_intensity_no_expansion(self):
        """経験強度ゼロの場合、拡大なし (1.0)。"""
        result = _compute_bandwidth_expansion_coefficient(0.0, 0.01)
        assert result == 1.0

    def test_very_low_intensity_no_expansion(self):
        """極めて低い経験強度の場合、拡大なし。"""
        result = _compute_bandwidth_expansion_coefficient(0.005, 0.01)
        assert result == 1.0

    def test_max_intensity_capped(self):
        """最大経験強度でも上限以下。"""
        result = _compute_bandwidth_expansion_coefficient(1.0, 0.01)
        assert result <= _EXP_BANDWIDTH_MAX_MULTIPLIER
        assert result == pytest.approx(_EXP_BANDWIDTH_MAX_MULTIPLIER, abs=0.01)

    def test_moderate_intensity_moderate_expansion(self):
        """中程度の経験強度で中程度の拡大。"""
        result = _compute_bandwidth_expansion_coefficient(0.25, 0.01)
        assert 1.0 < result < _EXP_BANDWIDTH_MAX_MULTIPLIER

    def test_expansion_monotonically_increases(self):
        """経験強度が増加すると拡大係数も増加する。"""
        vals = [_compute_bandwidth_expansion_coefficient(i * 0.1, 0.01)
                for i in range(1, 11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    def test_expansion_state_dependent(self):
        """帯域拡大係数は入力依存であり、固定定数ではない（設計書制約）。"""
        r1 = _compute_bandwidth_expansion_coefficient(0.3, 0.01)
        r2 = _compute_bandwidth_expansion_coefficient(0.7, 0.01)
        assert r1 != r2


# =============================================================================
# Tests: _apply_experience_driven_value_update (統合テスト)
# =============================================================================

class TestApplyExperienceDrivenValueUpdate:
    """経験強度帯域拡大の統合テスト。"""

    def test_basic_application(self):
        """基本的な帯域拡大の適用。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        old_orientation = orch._value_orientation
        _apply_experience_driven_value_update(orch)
        # 更新されていることを確認
        new_orientation = orch._value_orientation
        # "共感する" は dim_b: 0.2, dim_d: -0.1 のシグナル
        assert new_orientation.update_count > old_orientation.update_count

    def test_no_update_without_policy(self):
        """ポリシー未選択時は更新されない。"""
        orch = _make_mock_orchestrator(policy_label="")
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == old_count

    def test_no_update_without_episodes(self):
        """エピソード記憶がない場合は更新されない。"""
        orch = _make_mock_orchestrator()
        orch._last_episodes = None
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == old_count

    def test_no_update_without_emotional_companion(self):
        """エピソードに感情随伴情報がない場合は更新されない。"""
        ep = EpisodeEntry(
            episode_id="no_emo",
            episode_type=EpisodeType.OBSERVATION,
            summary="No emotion",
            topics=(),
            source_texts=(),
            timestamp=time.time(),
            duration_estimate=0.0,
            emotional_companion=None,
            self_observation_companion=None,
            context_summary="",
            importance=ImportanceLevel.TRIVIAL,
            vividness=0.9,
            reference_count=0,
            reinterpretation_count=0,
            is_compressed=False,
            compressed_episode_ids=(),
        )
        store = _make_episode_store([ep])
        orch = _make_mock_orchestrator(episodes_store=store)
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == old_count

    def test_low_intensity_no_effective_change(self):
        """低い経験強度では実質的な帯域拡大が発生しない。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.05,
            emotion_amplitude=0.05,
            mood_arousal=0.05,
        )
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        # 極めて低い強度なので更新されないはず（intensity < 0.01）
        assert orch._value_orientation.update_count == old_count

    def test_stronger_experience_larger_update(self):
        """強い経験のほうが大きな更新量を生じる。"""
        # 弱い経験
        orch_weak = _make_mock_orchestrator(
            emotion_intensity=0.3,
            emotion_amplitude=0.3,
            mood_arousal=0.3,
        )
        _apply_experience_driven_value_update(orch_weak)
        weak_dim_b = orch_weak._value_orientation.dim_b

        # 強い経験
        orch_strong = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.9,
            mood_arousal=0.9,
        )
        _apply_experience_driven_value_update(orch_strong)
        strong_dim_b = orch_strong._value_orientation.dim_b

        # "共感する" は dim_b: 0.2 の正シグナル → 強い方が大きく変動
        assert abs(strong_dim_b) > abs(weak_dim_b)


# =============================================================================
# Tests: 安全弁 (Safety Valves)
# =============================================================================

class TestSafetyValves:
    """安全弁の検証テスト。"""

    def test_absolute_upper_limit_on_expansion(self):
        """安全弁1: 帯域拡大係数の絶対上限。"""
        # 最大強度でも上限以下
        coeff = _compute_bandwidth_expansion_coefficient(1.0, 0.01)
        assert coeff <= _EXP_BANDWIDTH_MAX_MULTIPLIER

    def test_per_update_delta_limit(self):
        """安全弁2: 1回の更新で次元が変動できる量の絶対上限。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=1.0,
            emotion_amplitude=1.0,
            mood_arousal=1.0,
        )
        old_dims = orch._value_orientation.get_all_dimensions()
        _apply_experience_driven_value_update(orch)
        new_dims = orch._value_orientation.get_all_dimensions()

        for dim_key in old_dims:
            delta = abs(new_dims.get(dim_key, 0.0) - old_dims.get(dim_key, 0.0))
            assert delta <= _EXP_BANDWIDTH_MAX_DELTA_PER_DIM + 0.001, \
                f"Dimension {dim_key} delta {delta} exceeds limit"

    def test_cooldown_period(self):
        """安全弁3: 冷却期間中は帯域拡大が適用されない。"""
        orch = _make_mock_orchestrator(
            tick_count=10,
            last_bandwidth_tick=9,  # 1ティック前
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == old_count

    def test_cooldown_expired_allows_update(self):
        """冷却期間経過後は帯域拡大が再適用される。"""
        orch = _make_mock_orchestrator(
            tick_count=10,
            last_bandwidth_tick=10 - _EXP_BANDWIDTH_COOLDOWN_TICKS - 1,
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count > old_count

    def test_confidence_damping_maintained(self):
        """安全弁4: confidence dampingが帯域拡大後にも適用される。"""
        # 高い確信度の次元は帯域拡大されても変動しにくい
        high_conf_orientation = ValueOrientation(
            dim_b=0.5,
            confidence_b=0.9,  # 高い確信度
        )
        orch_high_conf = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
            orientation=high_conf_orientation,
        )
        _apply_experience_driven_value_update(orch_high_conf)
        high_conf_delta = abs(orch_high_conf._value_orientation.dim_b - 0.5)

        low_conf_orientation = ValueOrientation(
            dim_b=0.5,
            confidence_b=0.0,  # 低い確信度
        )
        orch_low_conf = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
            orientation=low_conf_orientation,
        )
        _apply_experience_driven_value_update(orch_low_conf)
        low_conf_delta = abs(orch_low_conf._value_orientation.dim_b - 0.5)

        # 高い確信度の方が変動量が小さい
        assert high_conf_delta < low_conf_delta

    def test_episodic_memory_not_written(self):
        """安全弁: エピソード記憶への書き込みが行われない。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        store_before = orch._last_episodes
        episodes_before = store_before.episodes
        _apply_experience_driven_value_update(orch)
        # エピソード記憶は変更されていない
        assert orch._last_episodes.episodes == episodes_before

    def test_no_enrichment_exposure(self):
        """安全弁6: 帯域拡大の情報がenrichmentに露出しない。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)
        # 実装コード内でenrichment関連のメソッド呼び出しや属性設定が
        # 行われていないことを確認する。
        # 関数本体（docstring除外）にenrichmentへの書き込みパターンがない
        import inspect
        source = inspect.getsource(_apply_experience_driven_value_update)
        # docstring以降のコード部分のみを検査
        # enrichment辞書への代入・追加がないことを確認
        assert "get_prompt_enrichment" not in source
        assert "_enrichment" not in source.split('"""', 2)[-1] if '"""' in source else True
        # 実行時: orchに対してenrichment関連の属性セットが行われていない
        # MagicMockは任意属性にアクセスできるため、代わりに
        # value_orientation以外の代入がないことを間接的に確認
        # (関数が変更するのは _value_orientation と _exp_bandwidth_last_tick のみ)

    def test_no_persistent_state(self):
        """安全弁7: 帯域拡大係数・適用履歴は永続化されない。"""
        # 実装コード内でログ記録や永続化が行われていないことをソースコード検査で確認
        import inspect
        source = inspect.getsource(_apply_experience_driven_value_update)
        # 永続化用のログ蓄積パターンが存在しない
        assert "_coefficient_log" not in source
        # コメント行を除外してから検索（コメント中の "save" 誤検出を防止）
        source_no_comments = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )
        assert "save" not in source_no_comments.lower()
        assert "persist" not in source_no_comments.lower()
        # _exp_bandwidth_last_tick が設定される（冷却管理用、非永続）
        assert "_exp_bandwidth_last_tick" in source
        # _exp_firing_tick_history は非永続の発動ティック履歴（累積安全弁用）
        # save/load対象外であることが設計上保証されている


# =============================================================================
# Tests: 既存経路との統合
# =============================================================================

class TestExistingPathIntegration:
    """既存の更新経路との統合テスト。"""

    def test_uses_existing_policy_dimension_map(self):
        """帯域拡大は既存のpolicy_dimension_mapを経由する。"""
        # "からかう" は dim_a: 0.3, dim_c: 0.2
        orch = _make_mock_orchestrator(
            policy_label="からかう",
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)
        new_orientation = orch._value_orientation
        # dim_a と dim_c が変動しているはず
        assert abs(new_orientation.dim_a) > 0.0 or abs(new_orientation.dim_c) > 0.0

    def test_unmapped_policy_no_update(self):
        """マッピングされていないポリシーでは帯域拡大が発生しない。"""
        orch = _make_mock_orchestrator(
            policy_label="unknown_policy_xyz",
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        # default_policy_influence が空なので帯域拡大なし
        assert orch._value_orientation.update_count == old_count

    def test_orientation_remains_in_valid_range(self):
        """帯域拡大後も次元値は [-1.0, 1.0] の範囲内。"""
        # 極端に高い値の状態から更新
        extreme_orientation = ValueOrientation(
            dim_a=0.95,
            dim_b=0.95,
            dim_c=0.95,
            dim_d=0.95,
            dim_e=0.95,
        )
        orch = _make_mock_orchestrator(
            emotion_intensity=1.0,
            emotion_amplitude=1.0,
            mood_arousal=1.0,
            orientation=extreme_orientation,
        )
        _apply_experience_driven_value_update(orch)
        dims = orch._value_orientation.get_all_dimensions()
        for dim_key, val in dims.items():
            assert -1.0 <= val <= 1.0, f"Dimension {dim_key} out of range: {val}"

    def test_cooldown_tick_recorded(self):
        """帯域拡大適用後、冷却期間のティック番号が記録される。"""
        orch = _make_mock_orchestrator(
            tick_count=42,
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)
        assert orch._exp_bandwidth_last_tick == 42

    def test_multiple_consecutive_calls_cooldown(self):
        """連続呼び出しで冷却期間が機能する。"""
        orch = _make_mock_orchestrator(
            tick_count=10,
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)
        first_count = orch._value_orientation.update_count

        # 同一ティックで再呼び出し — 冷却期間中なので更新されない
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == first_count

    def test_no_new_emotion_value_mapping(self):
        """新たな感情→価値次元の固定対応関係が導入されていない。"""
        # 帯域拡大は既存のポリシー→次元マップのみを使用する
        config = ValueOrientationConfig()
        original_map = dict(config.policy_dimension_map)

        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)

        # マップが変更されていない
        current_config = orch._vo_config
        assert current_config.policy_dimension_map == original_map

    def test_different_internal_states_different_expansion(self):
        """同一の経験でも内部状態が異なれば異なる帯域拡大が適用される（非固定性）。"""
        orch1 = _make_mock_orchestrator(
            emotion_intensity=0.8,
            emotion_amplitude=0.9,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch1)
        delta1 = abs(orch1._value_orientation.dim_b)  # "共感する" → dim_b

        orch2 = _make_mock_orchestrator(
            emotion_intensity=0.8,
            emotion_amplitude=0.3,  # 異なる振幅
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch2)
        delta2 = abs(orch2._value_orientation.dim_b)

        assert delta1 != delta2


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_hasattr_initialization(self):
        """_exp_bandwidth_last_tick が未設定の場合でも正常動作する。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        # 属性を明示的に削除
        if hasattr(orch, '_exp_bandwidth_last_tick'):
            delattr(orch, '_exp_bandwidth_last_tick')
        # 正常に動作する
        _apply_experience_driven_value_update(orch)
        assert hasattr(orch, '_exp_bandwidth_last_tick')

    def test_empty_episodes_tuple(self):
        """空のエピソードタプルの場合は更新されない。"""
        empty_store = EpisodeStore(
            episodes=(),
            links=(),
            total_episodes_recorded=0,
            total_compressions=0,
            average_vividness=0.0,
            active_episode_count=0,
            compressed_episode_count=0,
            timestamp=time.time(),
            description="Empty",
        )
        orch = _make_mock_orchestrator(episodes_store=empty_store)
        old_count = orch._value_orientation.update_count
        _apply_experience_driven_value_update(orch)
        assert orch._value_orientation.update_count == old_count

    def test_vo_config_none_uses_default(self):
        """_vo_config が None の場合はデフォルト設定が使用される。"""
        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        orch._vo_config = None
        # エラーなく動作する
        _apply_experience_driven_value_update(orch)

    def test_neutral_decay_preserved(self):
        """帯域拡大が既存のneutral_decay機構を変更しない。"""
        config = ValueOrientationConfig()
        assert config.neutral_decay_rate == 0.0001  # デフォルト値

        orch = _make_mock_orchestrator(
            emotion_intensity=0.9,
            emotion_amplitude=0.8,
            mood_arousal=0.7,
        )
        _apply_experience_driven_value_update(orch)
        # 設定が変更されていない
        assert orch._vo_config.neutral_decay_rate == 0.0001
