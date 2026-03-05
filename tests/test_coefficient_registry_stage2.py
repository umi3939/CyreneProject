"""
tests/test_coefficient_registry_stage2.py - 記述層モジュール群のcoefficient_registry段階的統合（第2段階）テスト

等価変換の検証:
- 新名前空間 description_common が正しく読み込まれること
- 各モジュールのConfig デフォルト値がハードコード値と一致すること
- 外部ファイル不在時にデフォルト値が使用されること
- 外部ファイルでの値変更が反映されること
- 全対象モジュールのConfig生成が正常動作すること
"""

from __future__ import annotations

import json
import os
import pytest
from typing import Any

# Reset registry before each test to ensure clean state
from psyche import coefficient_registry


@pytest.fixture(autouse=True)
def reset_registry():
    coefficient_registry.reset()
    yield
    coefficient_registry.reset()


# =============================================================================
# Test: description_common namespace existence and values
# =============================================================================

class TestDescriptionCommonNamespace:
    """description_common名前空間の基本検証。"""

    def test_category_accessible(self):
        """description_common カテゴリがアクセス可能。"""
        coefficient_registry.load("/nonexistent.json")
        result = coefficient_registry.get("description_common")
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        """全キーが存在する。"""
        coefficient_registry.load("/nonexistent.json")
        result = coefficient_registry.get("description_common")
        expected_keys = {
            "fifo_limit_30",
            "fifo_limit_50",
            "fifo_limit_100",
            "fifo_limit_200",
            "window_size_25",
            "window_size_30",
            "window_size_50",
            "freshness_decay_rate_002",
        }
        assert set(result.keys()) == expected_keys

    def test_fifo_limit_30(self):
        assert coefficient_registry.get("description_common", "fifo_limit_30") == 30

    def test_fifo_limit_50(self):
        assert coefficient_registry.get("description_common", "fifo_limit_50") == 50

    def test_fifo_limit_100(self):
        assert coefficient_registry.get("description_common", "fifo_limit_100") == 100

    def test_fifo_limit_200(self):
        assert coefficient_registry.get("description_common", "fifo_limit_200") == 200

    def test_window_size_25(self):
        assert coefficient_registry.get("description_common", "window_size_25") == 25

    def test_window_size_30(self):
        assert coefficient_registry.get("description_common", "window_size_30") == 30

    def test_window_size_50(self):
        assert coefficient_registry.get("description_common", "window_size_50") == 50

    def test_freshness_decay_rate_002(self):
        assert coefficient_registry.get("description_common", "freshness_decay_rate_002") == 0.02

    def test_values_are_read_only_copies(self):
        """返り値がディープコピーであることの確認。"""
        d1 = coefficient_registry.get("description_common")
        d1["fifo_limit_30"] = 999
        d2 = coefficient_registry.get("description_common")
        assert d2["fifo_limit_30"] == 30


# =============================================================================
# Test: External file override
# =============================================================================

class TestExternalFileOverride:
    """外部ファイルでの値変更反映テスト。"""

    def test_override_fifo_limit_30(self, tmp_path):
        """fifo_limit_30のオーバーライドが反映される。"""
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"fifo_limit_30": 42}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)
        assert coefficient_registry.get("description_common", "fifo_limit_30") == 42

    def test_override_freshness_decay(self, tmp_path):
        """freshness_decay_rate_002のオーバーライドが反映される。"""
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"freshness_decay_rate_002": 0.05}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)
        assert coefficient_registry.get("description_common", "freshness_decay_rate_002") == 0.05

    def test_partial_override_preserves_other_defaults(self, tmp_path):
        """部分オーバーライド時に他のデフォルト値が保持される。"""
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"fifo_limit_30": 42}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)
        assert coefficient_registry.get("description_common", "fifo_limit_30") == 42
        assert coefficient_registry.get("description_common", "fifo_limit_50") == 50
        assert coefficient_registry.get("description_common", "window_size_30") == 30

    def test_file_absent_uses_all_defaults(self):
        """ファイル不在時に全デフォルト値が使用される。"""
        coefficient_registry.load("/nonexistent.json")
        assert coefficient_registry.get("description_common", "fifo_limit_30") == 30
        assert coefficient_registry.get("description_common", "fifo_limit_50") == 50
        assert coefficient_registry.get("description_common", "fifo_limit_100") == 100
        assert coefficient_registry.get("description_common", "fifo_limit_200") == 200
        assert coefficient_registry.get("description_common", "window_size_25") == 25
        assert coefficient_registry.get("description_common", "window_size_30") == 30
        assert coefficient_registry.get("description_common", "window_size_50") == 50
        assert coefficient_registry.get("description_common", "freshness_decay_rate_002") == 0.02


# =============================================================================
# Test: Module Config equivalence (FIFO limits)
# =============================================================================

class TestModuleConfigFIFOLimits:
    """各モジュールのConfig FIFO上限がハードコード値と一致することの検証。"""

    def test_stabilization_description_max_history(self):
        from psyche.stabilization_description import StabilizationDescriptionConfig
        config = StabilizationDescriptionConfig()
        assert config.max_history == 30

    def test_behavioral_diversity_max_history(self):
        from psyche.behavioral_diversity_description import BehavioralDiversityConfig
        config = BehavioralDiversityConfig()
        assert config.max_history == 30

    def test_forgetting_recall_balance_max_history(self):
        from psyche.forgetting_recall_balance import ForgettingRecallBalanceConfig
        config = ForgettingRecallBalanceConfig()
        assert config.max_history == 30

    def test_reference_frequency_max_snapshot_history(self):
        from psyche.reference_frequency_description import ReferenceFrequencyConfig
        config = ReferenceFrequencyConfig()
        assert config.max_snapshot_history == 30

    def test_attention_distribution_max_snapshot_history(self):
        from psyche.attention_distribution_description import AttentionDistributionConfig
        config = AttentionDistributionConfig()
        assert config.max_snapshot_history == 30

    def test_input_pathway_balance_max_snapshot_history(self):
        from psyche.input_pathway_balance import InputPathwayBalanceConfig
        config = InputPathwayBalanceConfig()
        assert config.max_snapshot_history == 30

    def test_selection_attribution_max_records(self):
        from psyche.selection_attribution import SelectionAttributionConfig
        config = SelectionAttributionConfig()
        assert config.max_records == 50

    def test_self_action_perception_max_records(self):
        from psyche.self_action_perception import SelfActionPerceptionConfig
        config = SelfActionPerceptionConfig()
        assert config.max_records == 50

    def test_intent_action_gap_max_records(self):
        from psyche.intent_action_gap import IntentActionGapConfig
        config = IntentActionGapConfig()
        assert config.max_records == 50

    def test_emotion_cooccurrence_max_records(self):
        from psyche.emotion_cooccurrence_description import CooccurrenceConfig
        config = CooccurrenceConfig()
        assert config.max_records == 50

    def test_perceptual_context_max_summaries(self):
        from psyche.perceptual_context import PerceptualContextConfig
        config = PerceptualContextConfig()
        assert config.max_summaries == 50

    def test_interaction_accumulation_max_pairs(self):
        from psyche.interaction_accumulation import InteractionAccumulationConfig
        config = InteractionAccumulationConfig()
        assert config.max_pairs == 100

    def test_temporal_cognition_max_elapsed_records(self):
        from psyche.temporal_cognition import TemporalCognitionConfig
        config = TemporalCognitionConfig()
        assert config.max_elapsed_records == 100

    def test_temporal_cognition_max_external_input_records(self):
        from psyche.temporal_cognition import TemporalCognitionConfig
        config = TemporalCognitionConfig()
        assert config.max_external_input_records == 100

    def test_responsibility_temporal_trace_max_snapshots(self):
        from psyche.responsibility_temporal_trace import ResponsibilityTemporalTraceConfig
        config = ResponsibilityTemporalTraceConfig()
        assert config.max_snapshots == 100

    def test_expectation_lifecycle_max_records(self):
        from psyche.expectation_lifecycle_description import ExpectationLifecycleConfig
        config = ExpectationLifecycleConfig()
        assert config.max_records == 200

    def test_goal_hierarchy_propagation_max_records(self):
        from psyche.goal_hierarchy_propagation import GoalHierarchyPropagationConfig
        config = GoalHierarchyPropagationConfig()
        assert config.max_records == 200

    def test_input_pathway_balance_max_usage_facts(self):
        from psyche.input_pathway_balance import InputPathwayBalanceConfig
        config = InputPathwayBalanceConfig()
        assert config.max_usage_facts == 200

    def test_other_boundary_max_records_total(self):
        from psyche.other_boundary_accumulation import OtherBoundaryAccumulationConfig
        config = OtherBoundaryAccumulationConfig()
        assert config.max_records_total == 200

    def test_situational_self_presentation_max_records_total(self):
        from psyche.situational_self_presentation import SituationalSelfPresentationConfig
        config = SituationalSelfPresentationConfig()
        assert config.max_records_total == 200

    def test_hypothesis_observation_max_total_pairs(self):
        from psyche.hypothesis_observation_pairing import HypothesisObservationPairingConfig
        config = HypothesisObservationPairingConfig()
        assert config.max_total_pairs == 200


# =============================================================================
# Test: Module Config equivalence (Window sizes)
# =============================================================================

class TestModuleConfigWindowSizes:
    """各モジュールのConfig ウィンドウサイズがハードコード値と一致することの検証。"""

    def test_introspection_cross_section_max_snapshots(self):
        from psyche.introspection_cross_section import IntrospectionCrossSectionConfig
        config = IntrospectionCrossSectionConfig()
        assert config.max_snapshots == 25

    def test_emotional_backdrop_max_window_size(self):
        from psyche.emotional_backdrop_cognition import BackdropConfig
        config = BackdropConfig()
        assert config.max_window_size == 30

    def test_multi_path_recall_rumination_window_size(self):
        from psyche.multi_path_recall import MultiPathRecallConfig
        config = MultiPathRecallConfig()
        assert config.rumination_window_size == 30

    def test_spontaneous_recall_rumination_window_size(self):
        from psyche.spontaneous_recall import SpontaneousRecallConfig
        config = SpontaneousRecallConfig()
        assert config.rumination_window_size == 30

    def test_internal_contradiction_max_window_size(self):
        from psyche.internal_contradiction_description import ContradictionConfig
        config = ContradictionConfig()
        assert config.max_window_size == 50

    def test_drive_variation_max_window_size(self):
        from psyche.drive_variation_description import DriveVariationConfig
        config = DriveVariationConfig()
        assert config.max_window_size == 50

    def test_input_pathway_balance_sliding_window_size(self):
        from psyche.input_pathway_balance import InputPathwayBalanceConfig
        config = InputPathwayBalanceConfig()
        assert config.sliding_window_size == 50


# =============================================================================
# Test: Module Config equivalence (Freshness decay rate)
# =============================================================================

class TestModuleConfigFreshnessDecay:
    """各モジュールのConfig 鮮度減衰速度がハードコード値と一致することの検証。"""

    def test_emotion_cooccurrence_freshness_decay(self):
        from psyche.emotion_cooccurrence_description import CooccurrenceConfig
        config = CooccurrenceConfig()
        assert config.freshness_decay_rate == 0.02

    def test_drive_variation_freshness_decay(self):
        from psyche.drive_variation_description import DriveVariationConfig
        config = DriveVariationConfig()
        assert config.freshness_decay_rate == 0.02

    def test_emotional_backdrop_freshness_decay(self):
        from psyche.emotional_backdrop_cognition import BackdropConfig
        config = BackdropConfig()
        assert config.freshness_decay_rate == 0.02

    def test_other_boundary_freshness_decay(self):
        from psyche.other_boundary_accumulation import OtherBoundaryAccumulationConfig
        config = OtherBoundaryAccumulationConfig()
        assert config.freshness_decay_rate == 0.02

    def test_situational_self_presentation_freshness_decay(self):
        from psyche.situational_self_presentation import SituationalSelfPresentationConfig
        config = SituationalSelfPresentationConfig()
        assert config.freshness_decay_rate == 0.02

    def test_hypothesis_observation_freshness_decay(self):
        from psyche.hypothesis_observation_pairing import HypothesisObservationPairingConfig
        config = HypothesisObservationPairingConfig()
        assert config.freshness_decay_rate == 0.02

    def test_goal_hierarchy_freshness_decay(self):
        from psyche.goal_hierarchy_propagation import GoalHierarchyPropagationConfig
        config = GoalHierarchyPropagationConfig()
        assert config.record_freshness_decay_rate == 0.02

    def test_expectation_lifecycle_freshness_decay(self):
        from psyche.expectation_lifecycle_description import ExpectationLifecycleConfig
        config = ExpectationLifecycleConfig()
        assert config.record_freshness_decay_rate == 0.02


# =============================================================================
# Test: Override propagation to modules
# =============================================================================

class TestOverridePropagation:
    """レジストリのオーバーライドがモジュールConfigに伝播することの検証。"""

    def test_fifo_30_override_to_stabilization(self, tmp_path):
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"fifo_limit_30": 42}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)

        from psyche.stabilization_description import StabilizationDescriptionConfig
        config = StabilizationDescriptionConfig()
        assert config.max_history == 42

    def test_fifo_50_override_to_selection_attribution(self, tmp_path):
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"fifo_limit_50": 77}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)

        from psyche.selection_attribution import SelectionAttributionConfig
        config = SelectionAttributionConfig()
        assert config.max_records == 77

    def test_window_50_override_to_internal_contradiction(self, tmp_path):
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"window_size_50": 99}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)

        from psyche.internal_contradiction_description import ContradictionConfig
        config = ContradictionConfig()
        assert config.max_window_size == 99

    def test_freshness_override_to_emotion_cooccurrence(self, tmp_path):
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"freshness_decay_rate_002": 0.05}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)

        from psyche.emotion_cooccurrence_description import CooccurrenceConfig
        config = CooccurrenceConfig()
        assert config.freshness_decay_rate == 0.05

    def test_explicit_constructor_overrides_registry(self, tmp_path):
        """Configコンストラクタに明示的に値を渡した場合はレジストリ値を上書きする。"""
        coeff_path = str(tmp_path / "coefficients.json")
        data = {"description_common": {"fifo_limit_30": 42}}
        with open(coeff_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        coefficient_registry.load(coeff_path)

        from psyche.stabilization_description import StabilizationDescriptionConfig
        config = StabilizationDescriptionConfig(max_history=10)
        assert config.max_history == 10


# =============================================================================
# Test: Factory function equivalence
# =============================================================================

class TestFactoryFunctions:
    """ファクトリ関数がレジストリデフォルトと明示値の両方に対応することの検証。"""

    def test_stabilization_factory_default(self):
        from psyche.stabilization_description import create_stabilization_description_config
        config = create_stabilization_description_config()
        assert config.max_history == 30

    def test_stabilization_factory_explicit(self):
        from psyche.stabilization_description import create_stabilization_description_config
        config = create_stabilization_description_config(max_history=15)
        assert config.max_history == 15

    def test_behavioral_diversity_factory_default(self):
        from psyche.behavioral_diversity_description import create_behavioral_diversity_config
        config = create_behavioral_diversity_config()
        assert config.max_history == 30

    def test_behavioral_diversity_factory_explicit(self):
        from psyche.behavioral_diversity_description import create_behavioral_diversity_config
        config = create_behavioral_diversity_config(max_history=15)
        assert config.max_history == 15

    def test_forgetting_recall_factory_default(self):
        from psyche.forgetting_recall_balance import create_forgetting_recall_balance_config
        config = create_forgetting_recall_balance_config()
        assert config.max_history == 30

    def test_forgetting_recall_factory_explicit(self):
        from psyche.forgetting_recall_balance import create_forgetting_recall_balance_config
        config = create_forgetting_recall_balance_config(max_history=15)
        assert config.max_history == 15

    def test_reference_frequency_factory_default(self):
        from psyche.reference_frequency_description import create_reference_frequency_config
        config = create_reference_frequency_config()
        assert config.max_snapshot_history == 30

    def test_reference_frequency_factory_explicit(self):
        from psyche.reference_frequency_description import create_reference_frequency_config
        config = create_reference_frequency_config(max_snapshot_history=15)
        assert config.max_snapshot_history == 15

    def test_attention_distribution_factory_default(self):
        from psyche.attention_distribution_description import create_attention_distribution_config
        config = create_attention_distribution_config()
        assert config.max_snapshot_history == 30

    def test_attention_distribution_factory_explicit(self):
        from psyche.attention_distribution_description import create_attention_distribution_config
        config = create_attention_distribution_config(max_snapshot_history=15)
        assert config.max_snapshot_history == 15


# =============================================================================
# Test: coefficients.json contains the new category
# =============================================================================

class TestCoefficientsJSON:
    """data/coefficients.json に description_common が含まれることの検証。"""

    def test_json_contains_description_common(self):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "coefficients.json",
        )
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "description_common" in data
        dc = data["description_common"]
        assert dc["fifo_limit_30"] == 30
        assert dc["fifo_limit_50"] == 50
        assert dc["fifo_limit_100"] == 100
        assert dc["fifo_limit_200"] == 200
        assert dc["window_size_25"] == 25
        assert dc["window_size_30"] == 30
        assert dc["window_size_50"] == 50
        assert dc["freshness_decay_rate_002"] == 0.02


# =============================================================================
# Test: Non-integrated constants remain untouched
# =============================================================================

class TestNonIntegratedConstants:
    """統合対象外の定数が変更されていないことの検証。"""

    def test_internal_contradiction_freshness_decay_not_integrated(self):
        """internal_contradiction の freshness_decay_rate=0.03 は統合対象外。"""
        from psyche.internal_contradiction_description import ContradictionConfig
        config = ContradictionConfig()
        assert config.freshness_decay_rate == 0.03

    def test_other_boundary_per_user_limit_not_integrated(self):
        """other_boundary の max_records_per_user=50 はモジュール固有。"""
        from psyche.other_boundary_accumulation import OtherBoundaryAccumulationConfig
        config = OtherBoundaryAccumulationConfig()
        assert config.max_records_per_user == 50

    def test_situational_self_presentation_per_user_limit_not_integrated(self):
        """situational_self_presentation の max_records_per_user=50 はモジュール固有。"""
        from psyche.situational_self_presentation import SituationalSelfPresentationConfig
        config = SituationalSelfPresentationConfig()
        assert config.max_records_per_user == 50

    def test_hypothesis_per_user_limit_not_integrated(self):
        """hypothesis_observation の max_pairs_per_user=50 はモジュール固有。"""
        from psyche.hypothesis_observation_pairing import HypothesisObservationPairingConfig
        config = HypothesisObservationPairingConfig()
        assert config.max_pairs_per_user == 50
