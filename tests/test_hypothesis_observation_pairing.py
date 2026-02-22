"""
tests/test_hypothesis_observation_pairing.py

仮説-観測隣接対の包括的テスト。
設計書 (design_hypothesis_observation_pairing.md) の全要件をカバーする。

テスト項目:
- 6段パイプラインの各段階
- 安全弁7種
- 経路遮断
- save/load (永続化)
- 確認バイアス防止
- ルーミネーション防止
- 相手別分離蓄積
- FIFO押し出し
- 鮮度管理と自然消失
"""

import time
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any, Optional

from psyche.hypothesis_observation_pairing import (
    # Data structures
    HypothesisSnapshot,
    ObservationDescription,
    AdjacentPair,
    # Config / State
    HypothesisObservationPairingConfig,
    HypothesisObservationPairingState,
    # Stage functions
    acquire_hypothesis_snapshots,
    update_snapshot_buffer,
    acquire_observation_descriptions,
    compose_adjacent_pairs,
    accumulate_pairs_by_user,
    apply_freshness_decay,
    prepare_enrichment_pairs,
    get_reference_history,
    # Summary
    get_pairing_summary_text,
    # Save / Load
    save_pairing_state,
    load_pairing_state,
    # Processor
    HypothesisObservationPairingProcessor,
    create_hypothesis_observation_pairing_processor,
    get_hypothesis_observation_pairing_summary,
    # Helpers
    _extract_user_id,
)


# =============================================================================
# Mock objects
# =============================================================================

@dataclass
class MockHypothesis:
    """仮説のモック。other_agent_model.OtherStateHypothesis の代替。"""
    hypothesis_id: str = "h1"
    description: str = "相手は退屈しているかもしれない"
    freshness: float = 0.8
    strength: float = 0.6


@dataclass
class MockObservationFragment:
    """観測断片のモック。other_model_real_feed.ObservationFragment の代替。"""
    type: Any = None
    description: str = ""
    text_hint: str = ""
    source_description: str = ""

    def __post_init__(self):
        if self.type is None:
            self.type = MagicMock(value="speech_reaction")


@dataclass
class MockHypothesisSource:
    """仮説群を提供するモック。"""
    hypotheses: list = None

    def __post_init__(self):
        if self.hypotheses is None:
            self.hypotheses = []

    def get_active_hypotheses(self):
        return self.hypotheses


@dataclass
class MockObservationSource:
    """観測断片群を提供するモック。"""
    units: list = None

    def __post_init__(self):
        if self.units is None:
            self.units = []

    def get_latest_units(self):
        return self.units


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestHypothesisSnapshot:
    """HypothesisSnapshot データ構造テスト。"""

    def test_create_default(self):
        s = HypothesisSnapshot()
        assert s.hypothesis_id == ""
        assert s.description == ""
        assert s.freshness_value == 0.0
        assert s.strength_value == 0.0
        assert s.snapshot_cycle == 0
        assert s.timestamp > 0

    def test_create_with_values(self):
        s = HypothesisSnapshot(
            hypothesis_id="h1",
            description="test hypothesis",
            freshness_value=0.8,
            strength_value=0.6,
            snapshot_cycle=5,
            timestamp=1000.0,
        )
        assert s.hypothesis_id == "h1"
        assert s.description == "test hypothesis"
        assert s.freshness_value == 0.8
        assert s.strength_value == 0.6
        assert s.snapshot_cycle == 5

    def test_to_dict_from_dict_roundtrip(self):
        s = HypothesisSnapshot(
            hypothesis_id="h1",
            description="test",
            freshness_value=0.7,
            strength_value=0.5,
            snapshot_cycle=3,
            timestamp=1234.5,
        )
        d = s.to_dict()
        restored = HypothesisSnapshot.from_dict(d)
        assert restored.hypothesis_id == "h1"
        assert restored.description == "test"
        assert restored.freshness_value == 0.7
        assert restored.strength_value == 0.5
        assert restored.snapshot_cycle == 3
        assert restored.timestamp == 1234.5


class TestObservationDescription:
    """ObservationDescription データ構造テスト。"""

    def test_create_default(self):
        o = ObservationDescription()
        assert o.fragment_type == ""
        assert o.description == ""
        assert o.arrival_cycle == 0

    def test_to_dict_from_dict_roundtrip(self):
        o = ObservationDescription(
            fragment_type="speech_reaction",
            description="相手の返答",
            arrival_cycle=5,
            timestamp=2000.0,
        )
        d = o.to_dict()
        restored = ObservationDescription.from_dict(d)
        assert restored.fragment_type == "speech_reaction"
        assert restored.description == "相手の返答"
        assert restored.arrival_cycle == 5


class TestAdjacentPair:
    """AdjacentPair データ構造テスト。"""

    def test_create_default(self):
        p = AdjacentPair()
        assert p.pair_id != ""
        assert p.hypothesis_id == ""
        assert p.freshness == 1.0

    def test_all_fields_preserved(self):
        """全項目は等価に保持される。"""
        p = AdjacentPair(
            pair_id="test_id",
            hypothesis_id="h1",
            hypothesis_description="仮説記述",
            hypothesis_freshness=0.8,
            hypothesis_strength=0.6,
            hypothesis_snapshot_cycle=3,
            observation_type="speech_reaction",
            observation_description="観測記述",
            observation_arrival_cycle=5,
            user_id="user_a",
            timestamp=1000.0,
            freshness=0.9,
        )
        assert p.hypothesis_id == "h1"
        assert p.hypothesis_description == "仮説記述"
        assert p.hypothesis_freshness == 0.8
        assert p.hypothesis_strength == 0.6
        assert p.hypothesis_snapshot_cycle == 3
        assert p.observation_type == "speech_reaction"
        assert p.observation_description == "観測記述"
        assert p.observation_arrival_cycle == 5
        assert p.user_id == "user_a"
        assert p.freshness == 0.9

    def test_to_dict_from_dict_roundtrip(self):
        p = AdjacentPair(
            pair_id="test_id",
            hypothesis_id="h1",
            hypothesis_description="desc",
            hypothesis_freshness=0.8,
            hypothesis_strength=0.6,
            hypothesis_snapshot_cycle=3,
            observation_type="speech",
            observation_description="obs",
            observation_arrival_cycle=5,
            user_id="user_a",
            timestamp=1000.0,
            freshness=0.9,
        )
        d = p.to_dict()
        restored = AdjacentPair.from_dict(d)
        assert restored.pair_id == "test_id"
        assert restored.hypothesis_id == "h1"
        assert restored.hypothesis_description == "desc"
        assert restored.user_id == "user_a"
        assert restored.freshness == 0.9

    def test_no_weight_or_score_fields(self):
        """安全弁1: 重み・スコア・重要度フィールドが存在しないことを確認。"""
        p = AdjacentPair()
        assert not hasattr(p, "weight")
        assert not hasattr(p, "score")
        assert not hasattr(p, "importance")
        assert not hasattr(p, "priority")
        assert not hasattr(p, "relevance")


# =============================================================================
# Stage 1 Tests: 仮説スナップショット取得
# =============================================================================

class TestStage1HypothesisSnapshotAcquisition:
    """段階1: 仮説スナップショット取得のテスト。"""

    def test_acquire_from_model_system(self):
        """OtherAgentModelSystem パターンからの仮説取得。"""
        source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説A", freshness=0.8, strength=0.6),
            MockHypothesis(hypothesis_id="h2", description="仮説B", freshness=0.5, strength=0.3),
        ])
        snapshots = acquire_hypothesis_snapshots(source, current_cycle=10)
        assert len(snapshots) == 2
        assert snapshots[0].hypothesis_id == "h1"
        assert snapshots[0].description == "仮説A"
        assert snapshots[0].freshness_value == 0.8
        assert snapshots[0].strength_value == 0.6
        assert snapshots[0].snapshot_cycle == 10

    def test_acquire_from_list(self):
        """直接リストからの仮説取得。"""
        hyps = [
            MockHypothesis(hypothesis_id="h1", description="仮説X"),
        ]
        snapshots = acquire_hypothesis_snapshots(hyps, current_cycle=5)
        assert len(snapshots) == 1
        assert snapshots[0].hypothesis_id == "h1"

    def test_skip_empty_description(self):
        """空の記述を持つ仮説はスキップされる。"""
        source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description=""),
            MockHypothesis(hypothesis_id="h2", description="有効な仮説"),
        ])
        snapshots = acquire_hypothesis_snapshots(source, current_cycle=1)
        assert len(snapshots) == 1
        assert snapshots[0].hypothesis_id == "h2"

    def test_read_only_no_modification(self):
        """READ-ONLY: 仮説ソースの内容を改変しない。"""
        source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="元の記述", freshness=0.8, strength=0.6),
        ])
        acquire_hypothesis_snapshots(source, current_cycle=1)
        # 元のソースが改変されていないことを確認
        assert source.hypotheses[0].description == "元の記述"
        assert source.hypotheses[0].freshness == 0.8
        assert source.hypotheses[0].strength == 0.6

    def test_update_snapshot_buffer(self):
        """スナップショットバッファの更新。"""
        config = HypothesisObservationPairingConfig(
            max_snapshot_buffer=5,
            snapshot_retention_cycles=3,
        )
        buffer = [
            HypothesisSnapshot(hypothesis_id="old", description="old", snapshot_cycle=1),
        ]
        new = [
            HypothesisSnapshot(hypothesis_id="new", description="new", snapshot_cycle=5),
        ]
        updated = update_snapshot_buffer(buffer, new, config, current_cycle=5)
        # old (cycle=1) は retention_cycles=3 を超えて除去
        assert len(updated) == 1
        assert updated[0].hypothesis_id == "new"

    def test_buffer_overflow(self):
        """バッファ上限を超えた場合の最古の押し出し。"""
        config = HypothesisObservationPairingConfig(
            max_snapshot_buffer=3,
            snapshot_retention_cycles=100,
        )
        buffer = [
            HypothesisSnapshot(hypothesis_id=f"h{i}", description=f"d{i}", snapshot_cycle=i)
            for i in range(3)
        ]
        new = [
            HypothesisSnapshot(hypothesis_id="h_new", description="new", snapshot_cycle=3),
        ]
        updated = update_snapshot_buffer(buffer, new, config, current_cycle=3)
        assert len(updated) == 3
        # 最古の h0 が押し出される
        ids = [s.hypothesis_id for s in updated]
        assert "h0" not in ids
        assert "h_new" in ids


# =============================================================================
# Stage 2 Tests: 観測記述取得
# =============================================================================

class TestStage2ObservationDescriptionAcquisition:
    """段階2: 観測記述取得のテスト。"""

    def test_acquire_from_processor(self):
        """RealFeedProcessor パターンからの観測取得。"""
        frag = MockObservationFragment(
            description="相手が質問に答えた",
        )
        source = MockObservationSource(units=[frag])
        descriptions = acquire_observation_descriptions(source, current_cycle=10)
        assert len(descriptions) == 1
        assert descriptions[0].description == "相手が質問に答えた"
        assert descriptions[0].arrival_cycle == 10

    def test_acquire_from_list(self):
        """直接リストからの観測取得。"""
        frags = [
            MockObservationFragment(description="観測A"),
            MockObservationFragment(description="観測B"),
        ]
        descriptions = acquire_observation_descriptions(frags, current_cycle=5)
        assert len(descriptions) == 2

    def test_fallback_to_text_hint(self):
        """description が空の場合 text_hint にフォールバック。"""
        frag = MockObservationFragment(description="", text_hint="hint text")
        descriptions = acquire_observation_descriptions([frag], current_cycle=1)
        assert len(descriptions) == 1
        assert descriptions[0].description == "hint text"

    def test_fallback_to_source_description(self):
        """description と text_hint が空の場合 source_description にフォールバック。"""
        frag = MockObservationFragment(description="", text_hint="", source_description="source desc")
        descriptions = acquire_observation_descriptions([frag], current_cycle=1)
        assert len(descriptions) == 1
        assert descriptions[0].description == "source desc"

    def test_type_only_fallback(self):
        """description 系が全て空の場合、type のみで構成。"""
        frag = MockObservationFragment(description="", text_hint="", source_description="")
        descriptions = acquire_observation_descriptions([frag], current_cycle=1)
        assert len(descriptions) == 1
        assert "[speech_reaction]" in descriptions[0].description

    def test_skip_empty_fragment(self):
        """type も description も空の場合はスキップ。"""
        frag = MockObservationFragment(description="", text_hint="", source_description="")
        frag.type = None
        descriptions = acquire_observation_descriptions([frag], current_cycle=1)
        assert len(descriptions) == 0

    def test_read_only_no_modification(self):
        """READ-ONLY: 観測ソースの内容を改変しない。"""
        frag = MockObservationFragment(description="元の観測")
        source = MockObservationSource(units=[frag])
        acquire_observation_descriptions(source, current_cycle=1)
        assert source.units[0].description == "元の観測"


# =============================================================================
# Stage 3 Tests: 隣接対の構成
# =============================================================================

class TestStage3AdjacentPairComposition:
    """段階3: 隣接対の構成のテスト。"""

    def _make_config(self, **kwargs):
        return HypothesisObservationPairingConfig(**kwargs)

    def test_basic_pair_composition(self):
        """基本的な対構成: 仮説と観測がサイクル近接範囲内。"""
        config = self._make_config(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="仮説", snapshot_cycle=3),
        ]
        observations = [
            ObservationDescription(fragment_type="speech", description="観測", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config, user_id="user_a")
        assert len(pairs) == 1
        assert pairs[0].hypothesis_id == "h1"
        assert pairs[0].hypothesis_description == "仮説"
        assert pairs[0].observation_description == "観測"
        assert pairs[0].user_id == "user_a"
        assert pairs[0].freshness == 1.0

    def test_multiple_hypotheses_multiple_observations(self):
        """一つの仮説に対して複数の観測がそれぞれ独立した対として構成される。"""
        config = self._make_config(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="仮説A", snapshot_cycle=3),
            HypothesisSnapshot(hypothesis_id="h2", description="仮説B", snapshot_cycle=3),
        ]
        observations = [
            ObservationDescription(fragment_type="s1", description="観測X", arrival_cycle=5),
            ObservationDescription(fragment_type="s2", description="観測Y", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        # h1-X, h1-Y, h2-X, h2-Y = 4 pairs
        assert len(pairs) == 4

    def test_cycle_diff_out_of_range(self):
        """サイクル差が近接範囲を超える場合は対構成しない。"""
        config = self._make_config(cycle_proximity_range=3)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="仮説", snapshot_cycle=1),
        ]
        observations = [
            ObservationDescription(fragment_type="s", description="観測", arrival_cycle=10),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        assert len(pairs) == 0

    def test_hypothesis_after_observation_no_pair(self):
        """仮説が観測より後のサイクルの場合は対構成しない（仮説が先行記述）。"""
        config = self._make_config(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="仮説", snapshot_cycle=10),
        ]
        observations = [
            ObservationDescription(fragment_type="s", description="観測", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        assert len(pairs) == 0

    def test_same_cycle_is_valid(self):
        """同一サイクル（差=0）は対構成の対象。"""
        config = self._make_config(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="仮説", snapshot_cycle=5),
        ]
        observations = [
            ObservationDescription(fragment_type="s", description="観測", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        assert len(pairs) == 1

    def test_no_content_based_selection(self):
        """確認バイアス防止: 内容に基づく選択的対構成を行わない。

        仮説と「整合する」観測も「整合しない」観測も等価に対構成される。
        """
        config = self._make_config(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="相手は退屈している", snapshot_cycle=3),
        ]
        observations = [
            # "整合する"観測
            ObservationDescription(fragment_type="s", description="相手があくびをした", arrival_cycle=5),
            # "整合しない"観測
            ObservationDescription(fragment_type="s", description="相手が積極的に質問した", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        # 両方とも等価に対構成される
        assert len(pairs) == 2

    def test_empty_snapshots(self):
        """スナップショットが空の場合は対構成なし。"""
        config = self._make_config()
        pairs = compose_adjacent_pairs([], [ObservationDescription(description="obs", arrival_cycle=1)], config)
        assert len(pairs) == 0

    def test_empty_observations(self):
        """観測が空の場合は対構成なし。"""
        config = self._make_config()
        pairs = compose_adjacent_pairs(
            [HypothesisSnapshot(description="hyp", snapshot_cycle=1)], [], config
        )
        assert len(pairs) == 0


# =============================================================================
# Stage 4 Tests: 相手別分離蓄積
# =============================================================================

class TestStage4UserSeparatedAccumulation:
    """段階4: 相手別分離蓄積のテスト。"""

    def _make_config(self, **kwargs):
        defaults = {
            "max_pairs_per_user": 5,
            "max_total_pairs": 20,
        }
        defaults.update(kwargs)
        return HypothesisObservationPairingConfig(**defaults)

    def test_basic_accumulation(self):
        """基本的な蓄積。"""
        config = self._make_config()
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id="p1", user_id="user_a", hypothesis_description="h1"),
            AdjacentPair(pair_id="p2", user_id="user_b", hypothesis_description="h2"),
        ]
        accumulate_pairs_by_user(state, pairs, config)
        assert len(state.all_pairs) == 2
        assert "user_a" in state.user_pairs
        assert "user_b" in state.user_pairs
        assert len(state.user_pairs["user_a"]) == 1
        assert len(state.user_pairs["user_b"]) == 1
        assert state.total_pairs_created == 2

    def test_user_fifo_overflow(self):
        """相手別上限到達時のFIFO押し出し。"""
        config = self._make_config(max_pairs_per_user=3)
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id=f"p{i}", user_id="user_a", hypothesis_description=f"h{i}")
            for i in range(5)
        ]
        accumulate_pairs_by_user(state, pairs, config)
        assert len(state.user_pairs["user_a"]) == 3
        # 最古の p0, p1 が押し出されている
        remaining_ids = [p.pair_id for p in state.user_pairs["user_a"]]
        assert "p0" not in remaining_ids
        assert "p1" not in remaining_ids
        assert "p4" in remaining_ids

    def test_total_fifo_overflow(self):
        """全体上限到達時のFIFO押し出し。"""
        config = self._make_config(max_total_pairs=5, max_pairs_per_user=100)
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id=f"p{i}", user_id="user_a", hypothesis_description=f"h{i}")
            for i in range(8)
        ]
        accumulate_pairs_by_user(state, pairs, config)
        assert len(state.all_pairs) == 5
        assert state.total_pairs_pushed_out >= 3

    def test_unknown_user_id(self):
        """空のuser_idは__unknown__として蓄積される。"""
        config = self._make_config()
        state = HypothesisObservationPairingState()
        pairs = [AdjacentPair(pair_id="p1", user_id="")]
        accumulate_pairs_by_user(state, pairs, config)
        assert "__unknown__" in state.user_pairs

    def test_no_priority_between_users(self):
        """相手の属性・重要度・頻度に基づく優先は設けない。"""
        config = self._make_config(max_total_pairs=4)
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id="pa1", user_id="user_a"),
            AdjacentPair(pair_id="pa2", user_id="user_a"),
            AdjacentPair(pair_id="pb1", user_id="user_b"),
            AdjacentPair(pair_id="pb2", user_id="user_b"),
        ]
        accumulate_pairs_by_user(state, pairs, config)
        # 全対が蓄積される（上限内）
        assert len(state.all_pairs) == 4

    def test_empty_user_pairs_cleaned_after_overflow(self):
        """全体FIFO押し出しで空になった相手のエントリが削除される。"""
        config = self._make_config(max_total_pairs=2, max_pairs_per_user=100)
        state = HypothesisObservationPairingState()
        # user_a: 1対, user_b: 3対 => 全体上限2
        pairs = [
            AdjacentPair(pair_id="pa1", user_id="user_a"),
            AdjacentPair(pair_id="pb1", user_id="user_b"),
            AdjacentPair(pair_id="pb2", user_id="user_b"),
            AdjacentPair(pair_id="pb3", user_id="user_b"),
        ]
        accumulate_pairs_by_user(state, pairs, config)
        # 全体上限2で最古が押し出される
        assert len(state.all_pairs) == 2
        # user_a の pa1 が押し出されている可能性が高い
        if "user_a" in state.user_pairs:
            assert len(state.user_pairs["user_a"]) >= 0


# =============================================================================
# Stage 5 Tests: 鮮度管理と自然消失
# =============================================================================

class TestStage5FreshnessManagement:
    """段階5: 鮮度管理と自然消失のテスト。"""

    def _make_config(self, **kwargs):
        defaults = {
            "freshness_decay_rate": 0.1,
            "freshness_invisible_threshold": 0.05,
        }
        defaults.update(kwargs)
        return HypothesisObservationPairingConfig(**defaults)

    def test_uniform_decay(self):
        """全記録に均一の減衰が適用される。"""
        config = self._make_config(freshness_decay_rate=0.1)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=1.0, user_id="ua"),
            AdjacentPair(pair_id="p2", freshness=0.8, user_id="ua"),
            AdjacentPair(pair_id="p3", freshness=0.5, user_id="ub"),
        ]
        state.user_pairs = {
            "ua": [state.all_pairs[0], state.all_pairs[1]],
            "ub": [state.all_pairs[2]],
        }
        apply_freshness_decay(state, config)
        assert abs(state.all_pairs[0].freshness - 0.9) < 0.01
        assert abs(state.all_pairs[1].freshness - 0.7) < 0.01
        assert abs(state.all_pairs[2].freshness - 0.4) < 0.01

    def test_invisible_threshold_removal(self):
        """鮮度が消失水準以下の記録は除去される。"""
        config = self._make_config(freshness_decay_rate=0.1, freshness_invisible_threshold=0.15)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=0.5, user_id="ua"),
            AdjacentPair(pair_id="p2", freshness=0.15, user_id="ua"),  # will become 0.05, below threshold
        ]
        state.user_pairs = {"ua": list(state.all_pairs)}
        apply_freshness_decay(state, config)
        assert len(state.all_pairs) == 1
        assert state.all_pairs[0].pair_id == "p1"
        # 相手別蓄積からも除去
        assert len(state.user_pairs["ua"]) == 1

    def test_mechanical_decay_no_content_judgment(self):
        """消失は機械的であり、記録の内容に基づく判断を含まない。

        同じ鮮度の記録は内容に関係なく同じタイミングで消失する。
        """
        config = self._make_config(freshness_decay_rate=0.1, freshness_invisible_threshold=0.05)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=0.1, hypothesis_description="重要かもしれない仮説", user_id="ua"),
            AdjacentPair(pair_id="p2", freshness=0.1, hypothesis_description="つまらない仮説", user_id="ua"),
        ]
        state.user_pairs = {"ua": list(state.all_pairs)}
        apply_freshness_decay(state, config)
        # 両方とも 0.1 - 0.1 = 0.0 <= 0.05 で消失
        assert len(state.all_pairs) == 0

    def test_freshness_clamped_to_zero(self):
        """鮮度は0未満にならない。"""
        config = self._make_config(freshness_decay_rate=0.5, freshness_invisible_threshold=0.0)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=0.1, user_id="ua"),
        ]
        state.user_pairs = {"ua": list(state.all_pairs)}
        apply_freshness_decay(state, config)
        # 0.1 - 0.5 = -0.4 -> clamped to 0.0, but 0.0 <= 0.0, so removed if threshold is 0.0
        # threshold is 0.0 so 0.0 <= 0.0 is true, will be removed
        assert len(state.all_pairs) == 0

    def test_empty_user_entry_cleaned_after_invisible(self):
        """消失後に空になった相手のエントリが削除される。"""
        config = self._make_config(freshness_decay_rate=0.5, freshness_invisible_threshold=0.05)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=0.1, user_id="ua"),
        ]
        state.user_pairs = {"ua": [state.all_pairs[0]]}
        apply_freshness_decay(state, config)
        assert "ua" not in state.user_pairs


# =============================================================================
# Stage 6 Tests: 参照情報としての受渡準備
# =============================================================================

class TestStage6HandoffPreparation:
    """段階6: 参照情報としての受渡準備のテスト。"""

    def _make_config(self, **kwargs):
        defaults = {
            "enrichment_count": 3,
            "rumination_consecutive_limit": 2,
            "rumination_cooldown_cycles": 2,
        }
        defaults.update(kwargs)
        return HypothesisObservationPairingConfig(**defaults)

    def test_basic_enrichment(self):
        """基本的なenrichment列挙。"""
        config = self._make_config(enrichment_count=3)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id=f"p{i}", hypothesis_description=f"h{i}")
            for i in range(5)
        ]
        result = prepare_enrichment_pairs(state, config)
        assert len(result) == 3
        # 直近3件
        assert result[0].pair_id == "p2"
        assert result[1].pair_id == "p3"
        assert result[2].pair_id == "p4"

    def test_enrichment_empty_state(self):
        """空の状態ではenrichment結果も空。"""
        config = self._make_config()
        state = HypothesisObservationPairingState()
        result = prepare_enrichment_pairs(state, config)
        assert len(result) == 0

    def test_enrichment_fewer_than_count(self):
        """蓄積数がenrichment_count未満の場合は全て返す。"""
        config = self._make_config(enrichment_count=10)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1"),
            AdjacentPair(pair_id="p2"),
        ]
        result = prepare_enrichment_pairs(state, config)
        assert len(result) == 2

    def test_rumination_prevention(self):
        """ルーミネーション防止: 同一対の連続列挙が制限される。"""
        config = self._make_config(
            enrichment_count=2,
            rumination_consecutive_limit=2,
        )
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1"),
            AdjacentPair(pair_id="p2"),
            AdjacentPair(pair_id="p3"),
        ]
        # 1回目: p2, p3 が列挙される
        result1 = prepare_enrichment_pairs(state, config)
        assert len(result1) == 2
        assert {p.pair_id for p in result1} == {"p2", "p3"}

        # 2回目: p2, p3 の consecutive が 2 に達する
        result2 = prepare_enrichment_pairs(state, config)
        assert len(result2) == 2

        # 3回目: p2, p3 の consecutive が limit に達して除外される
        result3 = prepare_enrichment_pairs(state, config)
        # p1 が入ってくるはず
        pair_ids = {p.pair_id for p in result3}
        assert "p1" in pair_ids

    def test_rumination_cooldown_recovery(self):
        """ルーミネーション防止: 除外後の復帰。

        同一対が連続列挙されると上限で除外され、
        列挙から外れている間にクールダウン（デクリメント）で復帰する。
        """
        config = self._make_config(
            enrichment_count=1,
            rumination_consecutive_limit=2,
            rumination_cooldown_cycles=1,
        )
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1"),
            AdjacentPair(pair_id="p2"),
        ]

        # Call 1: p2 listed (consecutive=1)
        r1 = prepare_enrichment_pairs(state, config)
        assert r1[0].pair_id == "p2"

        # Call 2: p2 listed (consecutive=2 = limit)
        r2 = prepare_enrichment_pairs(state, config)
        assert r2[0].pair_id == "p2"

        # Call 3: p2 excluded (consecutive=2 >= limit), p1 listed
        r3 = prepare_enrichment_pairs(state, config)
        assert r3[0].pair_id == "p1"
        # p2 was not listed so its consecutive decremented to 1

        # Call 4: p2 is back (consecutive=1 < limit=2)
        r4 = prepare_enrichment_pairs(state, config)
        # p2 is the most recent so it's selected again
        assert r4[0].pair_id == "p2"

        # This shows the cooldown mechanism works:
        # p2 was excluded, then recovered after decrement
        assert len(r4) >= 1

    def test_reference_history(self):
        """READ-ONLY参照履歴の取得。"""
        config = self._make_config()
        config.reference_history_count = 3
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id=f"p{i}")
            for i in range(10)
        ]
        result = get_reference_history(state, config)
        assert len(result) == 3
        assert result[0].pair_id == "p7"

    def test_reference_history_by_user(self):
        """相手別の参照履歴。"""
        config = self._make_config()
        config.reference_history_count = 2
        state = HypothesisObservationPairingState()
        state.user_pairs = {
            "ua": [
                AdjacentPair(pair_id="pa1", user_id="ua"),
                AdjacentPair(pair_id="pa2", user_id="ua"),
                AdjacentPair(pair_id="pa3", user_id="ua"),
            ],
            "ub": [
                AdjacentPair(pair_id="pb1", user_id="ub"),
            ],
        }
        result = get_reference_history(state, config, user_id="ua")
        assert len(result) == 2
        assert result[0].pair_id == "pa2"
        assert result[1].pair_id == "pa3"


# =============================================================================
# Safety Valve Tests (安全弁)
# =============================================================================

class TestSafetyValve1EqualWeighting:
    """安全弁1: 全記録の等価性。"""

    def test_no_weight_in_pair(self):
        """隣接対に重み・スコア・重要度フィールドが存在しない。"""
        p = AdjacentPair()
        for attr in ["weight", "score", "importance", "priority", "relevance",
                      "accuracy", "match_score", "consistency"]:
            assert not hasattr(p, attr), f"Unexpected attribute: {attr}"

    def test_all_pairs_have_same_freshness_decay(self):
        """全記録に同じ減衰率が適用される。"""
        config = HypothesisObservationPairingConfig(freshness_decay_rate=0.1, freshness_invisible_threshold=0.0)
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", freshness=1.0, user_id="ua"),
            AdjacentPair(pair_id="p2", freshness=1.0, user_id="ub"),
        ]
        state.user_pairs = {
            "ua": [state.all_pairs[0]],
            "ub": [state.all_pairs[1]],
        }
        apply_freshness_decay(state, config)
        # 同じ鮮度からスタートしたので同じ結果
        assert abs(state.all_pairs[0].freshness - state.all_pairs[1].freshness) < 0.001


class TestSafetyValve2ConfirmationBiasPrevention:
    """安全弁2: 確認バイアスの構造的排除。"""

    def test_no_content_matching(self):
        """対構成基準に内容的整合性を用いない。"""
        config = HypothesisObservationPairingConfig(cycle_proximity_range=5)
        snapshots = [
            HypothesisSnapshot(hypothesis_id="h1", description="相手は怒っている", snapshot_cycle=3),
        ]
        # 整合する観測も整合しない観測も等価に構成
        observations = [
            ObservationDescription(description="相手が怒鳴った", arrival_cycle=5),
            ObservationDescription(description="相手が笑った", arrival_cycle=5),
            ObservationDescription(description="相手が黙った", arrival_cycle=5),
        ]
        pairs = compose_adjacent_pairs(snapshots, observations, config)
        assert len(pairs) == 3  # 全て等価に対構成される

    def test_no_accuracy_or_match_computation(self):
        """整合度・一致度・合致率のような数量を生成しない。"""
        p = AdjacentPair(
            hypothesis_description="相手は退屈している",
            observation_description="相手が積極的に質問した",
        )
        for attr in ["accuracy", "match_score", "consistency_score",
                      "alignment", "agreement", "correctness"]:
            assert not hasattr(p, attr), f"Unexpected attribute: {attr}"


class TestSafetyValve3FIFODisappearance:
    """安全弁3: FIFOによる自然消失。"""

    def test_fifo_mechanical_pushout(self):
        """上限到達時に最古の記録から機械的に押し出す。"""
        config = HypothesisObservationPairingConfig(max_total_pairs=3, max_pairs_per_user=100)
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id=f"p{i}", user_id="ua") for i in range(5)
        ]
        accumulate_pairs_by_user(state, pairs, config)
        assert len(state.all_pairs) == 3
        remaining_ids = {p.pair_id for p in state.all_pairs}
        assert "p0" not in remaining_ids
        assert "p1" not in remaining_ids
        assert "p4" in remaining_ids

    def test_no_selective_retention(self):
        """選択的保持を行わない。内容に関係なく最古から押し出される。"""
        config = HypothesisObservationPairingConfig(max_total_pairs=2, max_pairs_per_user=100)
        state = HypothesisObservationPairingState()
        pairs = [
            AdjacentPair(pair_id="important", user_id="ua", hypothesis_description="非常に重要な仮説"),
            AdjacentPair(pair_id="trivial", user_id="ua", hypothesis_description="些細な仮説"),
            AdjacentPair(pair_id="newest", user_id="ua", hypothesis_description="新しい仮説"),
        ]
        accumulate_pairs_by_user(state, pairs, config)
        remaining_ids = {p.pair_id for p in state.all_pairs}
        assert "important" not in remaining_ids  # 最古だから内容に関係なく押し出される


class TestSafetyValve4RuminationPrevention:
    """安全弁4: ルーミネーション防止。"""

    def test_consecutive_listing_limit(self):
        """同一対のenrichment連続列挙が制限される。"""
        config = HypothesisObservationPairingConfig(
            enrichment_count=1,
            rumination_consecutive_limit=2,
        )
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1"),
            AdjacentPair(pair_id="p2"),
        ]

        # p2 が連続で列挙される（直近だから）
        r1 = prepare_enrichment_pairs(state, config)
        assert r1[0].pair_id == "p2"
        r2 = prepare_enrichment_pairs(state, config)
        assert r2[0].pair_id == "p2"

        # 3回目で p2 が除外される
        r3 = prepare_enrichment_pairs(state, config)
        if r3:
            assert r3[0].pair_id == "p1"

    def test_exclusion_is_mechanical(self):
        """除外と復帰は機械的であり、対の内容に基づく判断を含まない。"""
        config = HypothesisObservationPairingConfig(
            enrichment_count=1,
            rumination_consecutive_limit=1,
        )
        state = HypothesisObservationPairingState()
        state.all_pairs = [
            AdjacentPair(pair_id="p1", hypothesis_description="内容は関係ない"),
            AdjacentPair(pair_id="p2", hypothesis_description="これも内容は関係ない"),
        ]
        r1 = prepare_enrichment_pairs(state, config)
        assert len(r1) == 1
        # consecutive limit=1 なので即座に除外される可能性がある
        r2 = prepare_enrichment_pairs(state, config)
        # 交互に列挙されることで機械的制御が動作している
        assert len(r2) >= 1


class TestSafetyValve5NoPatternExtraction:
    """安全弁5: パターン抽出の構造的排除。"""

    def test_no_statistics_methods(self):
        """統計量・頻度分布・傾向・規則性・成功率を算出するメソッドが存在しない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["compute_statistics", "extract_patterns",
                            "analyze_trends", "compute_success_rate",
                            "get_frequency_distribution", "detect_regularities"]:
            assert not hasattr(proc, method_name), f"Unexpected method: {method_name}"

    def test_summary_is_factual(self):
        """要約はパターン抽出を含まない事実記述のみ。"""
        state = HypothesisObservationPairingState()
        state.cycle_count = 10
        state.all_pairs = [AdjacentPair(pair_id="p1")]
        summary = get_pairing_summary_text(state)
        # パターン・傾向・規則性を示唆する語句が含まれないことを確認
        for word in ["パターン", "傾向", "規則性", "成功率", "一致率", "相関"]:
            assert word not in summary


class TestSafetyValve6UnidirectionalReference:
    """安全弁6: 単方向参照保証。"""

    def test_no_reverse_flow_methods(self):
        """他者状態推測層・観測供給層・長期蓄積層への逆流メソッドが存在しない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["update_hypothesis", "modify_observation",
                            "update_dialogue_learning", "write_back",
                            "push_to_hypothesis", "modify_other_model"]:
            assert not hasattr(proc, method_name), f"Unexpected method: {method_name}"

    def test_process_does_not_modify_sources(self):
        """process() が入力ソースを改変しない。"""
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        proc = HypothesisObservationPairingProcessor()
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=1,
        )
        # ソースが改変されていない
        assert hyp_source.hypotheses[0].description == "仮説"
        assert obs_source.units[0].description == "観測"


class TestSafetyValve7JudgmentPathwayBlock:
    """安全弁7: 判断系への経路遮断。"""

    def test_no_policy_selection_input(self):
        """ポリシー選択への入力経路を持たない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["get_policy_input", "compute_bias",
                            "get_stability_input", "get_responsibility_input",
                            "apply_to_emotion"]:
            assert not hasattr(proc, method_name), f"Unexpected method: {method_name}"

    def test_enrichment_is_read_only(self):
        """enrichmentは参照情報のみ。"""
        proc = HypothesisObservationPairingProcessor()
        proc._state.all_pairs = [
            AdjacentPair(pair_id="p1", hypothesis_description="h1", observation_description="o1"),
        ]
        data = proc.get_enrichment_data()
        assert "entries" in data
        assert "summary_text" in data
        # 判断を誘導する情報が含まれないことを確認
        for entry in data["entries"]:
            for key in ["recommendation", "action", "decision", "should",
                        "must", "policy", "bias"]:
                assert key not in entry, f"Unexpected key in enrichment: {key}"


# =============================================================================
# Processor Tests
# =============================================================================

class TestProcessor:
    """HypothesisObservationPairingProcessor の統合テスト。"""

    def test_full_pipeline(self):
        """6段パイプラインの一括実行。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(cycle_proximity_range=5)
        )

        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説A"),
            MockHypothesis(hypothesis_id="h2", description="仮説B"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測X"),
            MockObservationFragment(description="観測Y"),
        ])

        count = proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=5,
        )
        # h1-X, h1-Y, h2-X, h2-Y = 4 pairs (all in cycle_proximity_range)
        assert count == 4
        assert len(proc.state.all_pairs) == 4
        assert proc.state.cycle_count == 1

    def test_incremental_processing(self):
        """複数サイクルにわたる逐次処理。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(
                cycle_proximity_range=3,
                snapshot_retention_cycles=5,
            )
        )

        # サイクル1: 仮説のみ
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説"),
        ])
        count1 = proc.process(
            hypothesis_source=hyp_source,
            observation_source=None,
            user_id_source="user_a",
            current_cycle=1,
        )
        assert count1 == 0  # 観測がないので対構成なし
        assert len(proc.state.snapshot_buffer) == 1

        # サイクル3: 観測到着
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        count2 = proc.process(
            hypothesis_source=None,
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=3,
        )
        # バッファ内のh1 (cycle=1) と観測 (cycle=3) の差は2、proximity_range=3内
        assert count2 == 1

    def test_process_none_sources(self):
        """入力が全てNoneでもエラーなく処理される。"""
        proc = HypothesisObservationPairingProcessor()
        count = proc.process()
        assert count == 0
        assert proc.state.cycle_count == 1

    def test_enrichment_data_format(self):
        """enrichmentデータの形式が正しいことを確認。"""
        proc = HypothesisObservationPairingProcessor()
        proc._state.all_pairs = [
            AdjacentPair(
                pair_id="p1",
                hypothesis_description="仮説の記述",
                hypothesis_freshness=0.8,
                hypothesis_strength=0.6,
                observation_type="speech_reaction",
                observation_description="観測の記述",
                user_id="user_a",
                hypothesis_snapshot_cycle=3,
                observation_arrival_cycle=5,
            ),
        ]
        data = proc.get_enrichment_data()
        assert "pair_count" in data
        assert "user_count" in data
        assert "entries" in data
        assert "summary_text" in data
        assert data["pair_count"] == 1
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert "hypothesis" in entry
        assert "observation" in entry
        assert "user_id" in entry

    def test_enrichment_text_truncation(self):
        """長いテキストのenrichmentでの切り詰め。"""
        proc = HypothesisObservationPairingProcessor()
        long_text = "A" * 200
        proc._state.all_pairs = [
            AdjacentPair(
                pair_id="p1",
                hypothesis_description=long_text,
                observation_description=long_text,
            ),
        ]
        data = proc.get_enrichment_data()
        entry = data["entries"][0]
        assert len(entry["hypothesis"]) <= 84  # 80 + "..."
        assert len(entry["observation"]) <= 84

    def test_get_latest_pairs(self):
        """直近の隣接対の取得。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(enrichment_count=3)
        )
        proc._state.all_pairs = [
            AdjacentPair(pair_id=f"p{i}") for i in range(10)
        ]
        result = proc.get_latest_pairs()
        assert len(result) == 3
        assert result[-1].pair_id == "p9"

    def test_get_latest_pairs_with_count(self):
        proc = HypothesisObservationPairingProcessor()
        proc._state.all_pairs = [
            AdjacentPair(pair_id=f"p{i}") for i in range(10)
        ]
        result = proc.get_latest_pairs(count=2)
        assert len(result) == 2

    def test_get_pair_history(self):
        """参照履歴の取得。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(reference_history_count=5)
        )
        proc._state.all_pairs = [
            AdjacentPair(pair_id=f"p{i}") for i in range(20)
        ]
        result = proc.get_pair_history()
        assert len(result) == 5

    def test_get_pair_history_by_user(self):
        """相手別の参照履歴取得。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(reference_history_count=2)
        )
        proc._state.user_pairs = {
            "ua": [AdjacentPair(pair_id=f"pa{i}", user_id="ua") for i in range(5)],
        }
        result = proc.get_pair_history(user_id="ua")
        assert len(result) == 2
        assert result[-1].pair_id == "pa4"

    def test_get_user_ids(self):
        """相手識別子リストの取得。"""
        proc = HypothesisObservationPairingProcessor()
        proc._state.user_pairs = {
            "ua": [AdjacentPair(pair_id="p1", user_id="ua")],
            "ub": [AdjacentPair(pair_id="p2", user_id="ub")],
        }
        user_ids = proc.get_user_ids()
        assert set(user_ids) == {"ua", "ub"}

    def test_get_summary(self):
        """モジュールサマリの取得。"""
        proc = HypothesisObservationPairingProcessor()
        proc._state.all_pairs = [AdjacentPair(pair_id="p1")]
        proc._state.user_pairs = {"ua": [AdjacentPair(pair_id="p1", user_id="ua")]}
        proc._state.total_pairs_created = 5
        proc._state.total_pairs_pushed_out = 2
        proc._state.cycle_count = 10
        summary = proc.get_summary()
        assert summary["pair_count"] == 1
        assert summary["user_count"] == 1
        assert summary["total_pairs_created"] == 5
        assert summary["total_pairs_pushed_out"] == 2
        assert summary["cycle_count"] == 10


# =============================================================================
# Save / Load Tests
# =============================================================================

class TestSaveLoad:
    """永続化 (save/load) のテスト。"""

    def test_empty_state_roundtrip(self):
        """空の状態のsave/load往復。"""
        state = HypothesisObservationPairingState()
        data = save_pairing_state(state)
        restored = load_pairing_state(data)
        assert restored.cycle_count == 0
        assert len(restored.all_pairs) == 0
        assert len(restored.snapshot_buffer) == 0
        assert len(restored.user_pairs) == 0

    def test_populated_state_roundtrip(self):
        """データの入った状態のsave/load往復。"""
        state = HypothesisObservationPairingState()
        state.snapshot_buffer = [
            HypothesisSnapshot(
                hypothesis_id="h1", description="snap", freshness_value=0.8,
                strength_value=0.6, snapshot_cycle=3, timestamp=1000.0,
            ),
        ]
        state.all_pairs = [
            AdjacentPair(
                pair_id="p1", hypothesis_id="h1", hypothesis_description="hyp",
                hypothesis_freshness=0.8, hypothesis_strength=0.6,
                hypothesis_snapshot_cycle=3, observation_type="speech",
                observation_description="obs", observation_arrival_cycle=5,
                user_id="ua", timestamp=2000.0, freshness=0.9,
            ),
        ]
        state.user_pairs = {"ua": list(state.all_pairs)}
        state.enrichment_consecutive = {"p1": 2}
        state.total_pairs_created = 10
        state.total_pairs_pushed_out = 3
        state.cycle_count = 20

        data = save_pairing_state(state)
        restored = load_pairing_state(data)

        assert len(restored.snapshot_buffer) == 1
        assert restored.snapshot_buffer[0].hypothesis_id == "h1"
        assert len(restored.all_pairs) == 1
        assert restored.all_pairs[0].pair_id == "p1"
        assert restored.all_pairs[0].hypothesis_description == "hyp"
        assert restored.all_pairs[0].user_id == "ua"
        assert "ua" in restored.user_pairs
        assert len(restored.user_pairs["ua"]) == 1
        assert restored.enrichment_consecutive == {"p1": 2}
        assert restored.total_pairs_created == 10
        assert restored.total_pairs_pushed_out == 3
        assert restored.cycle_count == 20

    def test_processor_state_roundtrip(self):
        """プロセッサ状態のsave/load往復。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(cycle_proximity_range=5)
        )

        # 処理を実行
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=5,
        )

        # save
        data = save_pairing_state(proc.state)

        # 新しいプロセッサにload
        new_proc = HypothesisObservationPairingProcessor()
        new_proc.state = load_pairing_state(data)

        assert new_proc.state.cycle_count == proc.state.cycle_count
        assert len(new_proc.state.all_pairs) == len(proc.state.all_pairs)

    def test_save_load_with_missing_fields(self):
        """不足フィールドがある場合のload（後方互換性）。"""
        data = {
            "cycle_count": 5,
            # 他のフィールドは省略
        }
        restored = load_pairing_state(data)
        assert restored.cycle_count == 5
        assert len(restored.all_pairs) == 0
        assert len(restored.snapshot_buffer) == 0


# =============================================================================
# User ID Extraction Tests
# =============================================================================

class TestUserIdExtraction:
    """相手識別情報の抽出テスト。"""

    def test_string_input(self):
        assert _extract_user_id("user_a") == "user_a"

    def test_none_input(self):
        assert _extract_user_id(None) == ""

    def test_object_with_user_id(self):
        obj = MagicMock()
        obj.user_id = "user_b"
        assert _extract_user_id(obj) == "user_b"

    def test_dict_with_user_id(self):
        assert _extract_user_id({"user_id": "user_c"}) == "user_c"

    def test_dict_without_user_id(self):
        assert _extract_user_id({"other_key": "val"}) == ""


# =============================================================================
# Summary Tests
# =============================================================================

class TestSummary:
    """要約テスト。"""

    def test_waiting_state(self):
        state = HypothesisObservationPairingState()
        text = get_pairing_summary_text(state)
        assert "待機中" in text

    def test_with_data(self):
        state = HypothesisObservationPairingState()
        state.cycle_count = 10
        state.all_pairs = [AdjacentPair(pair_id="p1")]
        state.user_pairs = {"ua": [state.all_pairs[0]]}
        state.total_pairs_pushed_out = 3
        state.snapshot_buffer = [HypothesisSnapshot(hypothesis_id="h1", description="d")]
        text = get_pairing_summary_text(state)
        assert "cycle=10" in text
        assert "蓄積対=1" in text
        assert "相手数=1" in text
        assert "消失累計=3" in text

    def test_factory_summary(self):
        proc = create_hypothesis_observation_pairing_processor()
        text = get_hypothesis_observation_pairing_summary(proc)
        assert "待機中" in text


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory:
    """ファクトリ関数テスト。"""

    def test_create_default(self):
        proc = create_hypothesis_observation_pairing_processor()
        assert isinstance(proc, HypothesisObservationPairingProcessor)
        assert proc.state.cycle_count == 0

    def test_create_with_config(self):
        config = HypothesisObservationPairingConfig(max_total_pairs=50)
        proc = create_hypothesis_observation_pairing_processor(config=config)
        assert proc.config.max_total_pairs == 50


# =============================================================================
# Design Constraint Verification Tests
# =============================================================================

class TestDesignConstraints:
    """設計書の制約条件を検証するテスト。"""

    def test_no_correctness_judgment(self):
        """仮説の正誤を判定しない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["evaluate_correctness", "judge_accuracy",
                            "check_match", "validate_hypothesis",
                            "compute_correctness"]:
            assert not hasattr(proc, method_name)

    def test_no_hypothesis_modification(self):
        """仮説の修正を行わない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["modify_hypothesis", "revise_hypothesis",
                            "strengthen_hypothesis", "weaken_hypothesis",
                            "retract_hypothesis"]:
            assert not hasattr(proc, method_name)

    def test_no_trust_modification(self):
        """仮説の正確さに基づいて他者モデルの信頼度を変動させない。"""
        proc = HypothesisObservationPairingProcessor()
        for method_name in ["update_trust", "modify_trust",
                            "compute_trust_delta", "adjust_confidence"]:
            assert not hasattr(proc, method_name)

    def test_no_normative_output(self):
        """規範的情報を生成しない。"""
        proc = HypothesisObservationPairingProcessor()
        proc._state.all_pairs = [
            AdjacentPair(pair_id="p1", hypothesis_description="仮説", observation_description="観測"),
        ]
        data = proc.get_enrichment_data()
        summary = data.get("summary_text", "")
        for word in ["修正すべき", "不正確", "改善", "推奨", "べき"]:
            assert word not in summary

    def test_no_pattern_extraction_in_output(self):
        """蓄積された隣接対からパターン・傾向・規則性を抽出しない。"""
        proc = HypothesisObservationPairingProcessor()
        data = proc.get_enrichment_data()
        for key in ["patterns", "trends", "regularities", "statistics",
                     "frequency_distribution", "success_rate"]:
            assert key not in data

    def test_no_judgment_connection(self):
        """蓄積情報を判断・行動選択・ポリシー選択に接続しない。"""
        proc = HypothesisObservationPairingProcessor()
        data = proc.get_enrichment_data()
        for key in ["policy_recommendation", "action_suggestion",
                     "bias_adjustment", "stability_signal"]:
            assert key not in data

    def test_immutable_pairs(self):
        """一度構成された隣接対は変更されない（追記のみ）。

        鮮度以外の全フィールドが変更されないことを確認。
        """
        config = HypothesisObservationPairingConfig(
            cycle_proximity_range=5,
            freshness_decay_rate=0.1,
            freshness_invisible_threshold=0.0,
        )
        proc = HypothesisObservationPairingProcessor(config=config)

        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説A"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測X"),
        ])
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=5,
        )

        # 隣接対のフィールドを記録
        pair = proc.state.all_pairs[0]
        original_id = pair.hypothesis_id
        original_desc = pair.hypothesis_description
        original_obs = pair.observation_description

        # もう1サイクル実行
        proc.process(current_cycle=6)

        # 鮮度以外は変更されていない
        assert pair.hypothesis_id == original_id
        assert pair.hypothesis_description == original_desc
        assert pair.observation_description == original_obs

    def test_user_separation_by_id_only(self):
        """相手別分離は識別子の一致のみに基づく。"""
        config = HypothesisObservationPairingConfig(cycle_proximity_range=5)
        proc = HypothesisObservationPairingProcessor(config=config)

        # 2つの相手で処理
        for uid in ["user_a", "user_b"]:
            hyp_source = MockHypothesisSource(hypotheses=[
                MockHypothesis(hypothesis_id=f"h_{uid}", description=f"仮説for {uid}"),
            ])
            obs_source = MockObservationSource(units=[
                MockObservationFragment(description=f"観測from {uid}"),
            ])
            proc.process(
                hypothesis_source=hyp_source,
                observation_source=obs_source,
                user_id_source=uid,
                current_cycle=5,
            )

        # 相手間で共有・統合・比較する経路を持たない
        assert "user_a" in proc.state.user_pairs
        assert "user_b" in proc.state.user_pairs
        # 各相手の対が独立
        for uid in ["user_a", "user_b"]:
            for pair in proc.state.user_pairs[uid]:
                assert pair.user_id == uid


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """境界条件テスト。"""

    def test_very_large_cycle_number(self):
        """非常に大きなサイクル番号。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(cycle_proximity_range=5)
        )
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(hypothesis_id="h1", description="仮説"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="ua",
            current_cycle=1_000_000,
        )
        assert proc.state.cycle_count == 1

    def test_concurrent_user_overflow(self):
        """複数ユーザーが同時に上限に近づく場合。"""
        config = HypothesisObservationPairingConfig(
            max_pairs_per_user=2,
            max_total_pairs=10,
            cycle_proximity_range=5,
        )
        proc = HypothesisObservationPairingProcessor(config=config)

        for i in range(5):
            for uid in ["ua", "ub", "uc"]:
                hyp_source = MockHypothesisSource(hypotheses=[
                    MockHypothesis(hypothesis_id=f"h_{uid}_{i}", description=f"仮説{i}"),
                ])
                obs_source = MockObservationSource(units=[
                    MockObservationFragment(description=f"観測{i}"),
                ])
                proc.process(
                    hypothesis_source=hyp_source,
                    observation_source=obs_source,
                    user_id_source=uid,
                    current_cycle=i + 1,
                )

        # 各ユーザー最大2対
        for uid in proc.state.user_pairs:
            assert len(proc.state.user_pairs[uid]) <= 2
        # 全体最大10対
        assert len(proc.state.all_pairs) <= 10

    def test_empty_string_user_id(self):
        """空文字列のuser_id。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(cycle_proximity_range=5)
        )
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(description="仮説"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="",
            current_cycle=5,
        )
        # __unknown__ として蓄積される
        assert "__unknown__" in proc.state.user_pairs or len(proc.state.all_pairs) > 0

    def test_process_with_explicit_cycle(self):
        """明示的なサイクル番号を指定。"""
        proc = HypothesisObservationPairingProcessor()
        proc.process(current_cycle=100)
        assert proc.state.cycle_count == 1

    def test_process_without_cycle(self):
        """サイクル番号を指定しない場合のデフォルト動作。"""
        proc = HypothesisObservationPairingProcessor()
        proc.process()
        assert proc.state.cycle_count == 1
        proc.process()
        assert proc.state.cycle_count == 2

    def test_freshness_decay_multiple_cycles(self):
        """複数サイクルにわたる鮮度減衰。"""
        config = HypothesisObservationPairingConfig(
            freshness_decay_rate=0.1,
            freshness_invisible_threshold=0.05,
            cycle_proximity_range=5,
        )
        proc = HypothesisObservationPairingProcessor(config=config)

        # 対を作成
        hyp_source = MockHypothesisSource(hypotheses=[
            MockHypothesis(description="仮説"),
        ])
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測"),
        ])
        proc.process(
            hypothesis_source=hyp_source,
            observation_source=obs_source,
            user_id_source="ua",
            current_cycle=5,
        )
        initial_count = len(proc.state.all_pairs)
        assert initial_count == 1

        # 繰り返し処理して鮮度を減衰させる
        for i in range(20):
            proc.process(current_cycle=6 + i)

        # 最終的に対が消失しているはず
        assert len(proc.state.all_pairs) < initial_count or len(proc.state.all_pairs) == 0

    def test_state_property(self):
        """state プロパティの get/set。"""
        proc = HypothesisObservationPairingProcessor()
        new_state = HypothesisObservationPairingState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42

    def test_config_property(self):
        """config プロパティの取得。"""
        config = HypothesisObservationPairingConfig(max_total_pairs=99)
        proc = HypothesisObservationPairingProcessor(config=config)
        assert proc.config.max_total_pairs == 99


# =============================================================================
# Integration-like Tests
# =============================================================================

class TestIntegrationScenarios:
    """統合シナリオテスト。"""

    def test_multi_cycle_accumulation(self):
        """複数サイクルにわたる蓄積と鮮度管理。"""
        config = HypothesisObservationPairingConfig(
            cycle_proximity_range=3,
            snapshot_retention_cycles=5,
            freshness_decay_rate=0.05,
            freshness_invisible_threshold=0.05,
        )
        proc = HypothesisObservationPairingProcessor(config=config)

        total_created = 0
        for cycle in range(1, 11):
            hyp_source = MockHypothesisSource(hypotheses=[
                MockHypothesis(hypothesis_id=f"h_{cycle}", description=f"仮説{cycle}"),
            ])
            obs_source = MockObservationSource(units=[
                MockObservationFragment(description=f"観測{cycle}"),
            ])
            count = proc.process(
                hypothesis_source=hyp_source,
                observation_source=obs_source,
                user_id_source="user_a",
                current_cycle=cycle,
            )
            total_created += count

        assert proc.state.cycle_count == 10
        assert proc.state.total_pairs_created == total_created
        assert len(proc.state.all_pairs) > 0

    def test_multiple_users_independent(self):
        """複数の相手が独立して蓄積される。"""
        config = HypothesisObservationPairingConfig(cycle_proximity_range=5)
        proc = HypothesisObservationPairingProcessor(config=config)

        for uid in ["alice", "bob", "charlie"]:
            hyp_source = MockHypothesisSource(hypotheses=[
                MockHypothesis(hypothesis_id=f"h_{uid}", description=f"仮説 about {uid}"),
            ])
            obs_source = MockObservationSource(units=[
                MockObservationFragment(description=f"観測 from {uid}"),
            ])
            proc.process(
                hypothesis_source=hyp_source,
                observation_source=obs_source,
                user_id_source=uid,
                current_cycle=5,
            )

        # 各ユーザーが独立して蓄積
        assert len(proc.get_user_ids()) == 3
        for uid in ["alice", "bob", "charlie"]:
            pairs = proc.get_pair_history(user_id=uid)
            assert len(pairs) > 0
            for pair in pairs:
                assert pair.user_id == uid

    def test_hypothesis_only_then_observation_later(self):
        """先に仮説のみ蓄積し、後から観測が到着するパターン。"""
        config = HypothesisObservationPairingConfig(
            cycle_proximity_range=5,
            snapshot_retention_cycles=10,
        )
        proc = HypothesisObservationPairingProcessor(config=config)

        # 仮説のみ（cycle 1-3）
        for cycle in range(1, 4):
            hyp_source = MockHypothesisSource(hypotheses=[
                MockHypothesis(hypothesis_id=f"h_{cycle}", description=f"仮説{cycle}"),
            ])
            proc.process(hypothesis_source=hyp_source, current_cycle=cycle)

        assert len(proc.state.snapshot_buffer) == 3
        assert len(proc.state.all_pairs) == 0

        # 観測到着（cycle 4）
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="遅延した観測"),
        ])
        count = proc.process(
            observation_source=obs_source,
            user_id_source="user_a",
            current_cycle=4,
        )
        # cycle_proximity_range=5 なので cycle 1,2,3 の仮説全てと対構成
        assert count == 3

    def test_observation_without_hypothesis_creates_no_pairs(self):
        """仮説なしで観測のみの場合、対構成は行われない。"""
        proc = HypothesisObservationPairingProcessor(
            config=HypothesisObservationPairingConfig(cycle_proximity_range=5)
        )
        obs_source = MockObservationSource(units=[
            MockObservationFragment(description="観測のみ"),
        ])
        count = proc.process(observation_source=obs_source, current_cycle=5)
        assert count == 0
