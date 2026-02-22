"""
tests/test_interaction_accumulation.py - 相互作用の蓄積記述テスト

全機能のテスト:
- 4段パイプラインの各段階
- save/load往復テスト
- 安全弁テスト（全記録等価性、ルーミネーション防止、パターン抽出不在など）
- エッジケーステスト
"""

import time
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional

from psyche.interaction_accumulation import (
    AdjacentPair,
    BufferEntry,
    InteractionAccumulationState,
    InteractionAccumulationConfig,
    InteractionAccumulationProcessor,
    compose_pairs,
    accumulate_pairs,
    prepare_enrichment_pairs,
    get_interaction_summary,
    create_interaction_accumulation_processor,
    _extract_reaction_description,
)


# =============================================================================
# Test Helpers
# =============================================================================

@dataclass
class MockSelfRecord:
    """自己行動知覚の記録モック。"""
    response_text: str = "テスト出力"
    policy_label: str = "policy_a"
    tick: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MockOtherUnit:
    """他者モデルリアルフィードの観測ユニットモック。"""
    tick: int = 0
    text_hint: str = "相手の反応"
    source_description: str = ""
    summary: str = ""
    fragments: Optional[list] = None


@dataclass
class MockFragment:
    """観測断片モック。"""
    type: Any = None
    text_hint: str = ""
    source_description: str = ""


@dataclass
class MockFragmentType:
    """フラグメントタイプモック。"""
    value: str = "speech_reaction"


# =============================================================================
# Stage 1: 隣接対の構成テスト
# =============================================================================

class TestComposePairs:
    """compose_pairs のテスト。"""

    def test_basic_pair_composition(self):
        """基本的な対構成。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        self_records = [MockSelfRecord(tick=10, response_text="自分の発言")]
        other_units = [MockOtherUnit(tick=12, text_hint="相手の反応")]

        new_pairs, buffer = compose_pairs(
            self_records=self_records,
            other_units=other_units,
            buffer=[],
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 1
        assert new_pairs[0].self_text == "自分の発言"
        assert new_pairs[0].other_reaction == "相手の反応"
        assert new_pairs[0].self_tick == 10
        assert new_pairs[0].other_tick == 12

    def test_no_pair_when_tick_gap_too_small(self):
        """最低経過ティック未満では対構成しない。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        self_records = [MockSelfRecord(tick=10)]
        other_units = [MockOtherUnit(tick=10)]  # 同一ティック

        new_pairs, _ = compose_pairs(
            self_records=self_records,
            other_units=other_units,
            buffer=[],
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 0

    def test_no_pair_when_tick_gap_too_large(self):
        """ティック差が範囲外では対構成しない。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        self_records = [MockSelfRecord(tick=10)]
        other_units = [MockOtherUnit(tick=20)]  # 差が10

        new_pairs, _ = compose_pairs(
            self_records=self_records,
            other_units=other_units,
            buffer=[],
            config=config,
            current_tick=25,
        )

        assert len(new_pairs) == 0

    def test_multiple_other_reactions_for_one_self(self):
        """一つの自己表出に対して複数の他者反応が隣接する場合、それぞれ独立した対。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        self_records = [MockSelfRecord(tick=10, response_text="自分の発言")]
        other_units = [
            MockOtherUnit(tick=12, text_hint="反応A"),
            MockOtherUnit(tick=13, text_hint="反応B"),
        ]

        new_pairs, _ = compose_pairs(
            self_records=self_records,
            other_units=other_units,
            buffer=[],
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 2
        reactions = {p.other_reaction for p in new_pairs}
        assert "反応A" in reactions
        assert "反応B" in reactions

    def test_multiple_self_for_one_other(self):
        """一つの他者反応に対して複数の自己表出が隣接する場合、それぞれ独立した対。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        self_records = [
            MockSelfRecord(tick=10, response_text="発言A"),
            MockSelfRecord(tick=11, response_text="発言B"),
        ]
        other_units = [MockOtherUnit(tick=13, text_hint="相手の反応")]

        new_pairs, _ = compose_pairs(
            self_records=self_records,
            other_units=other_units,
            buffer=[],
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 2
        texts = {p.self_text for p in new_pairs}
        assert "発言A" in texts
        assert "発言B" in texts

    def test_buffer_timeout(self):
        """タイムアウトしたバッファエントリは保留状態に移行する。"""
        config = InteractionAccumulationConfig(buffer_timeout_ticks=5)
        self_records = [MockSelfRecord(tick=1, response_text="古い発言")]

        _, buffer = compose_pairs(
            self_records=self_records,
            other_units=[],
            buffer=[],
            config=config,
            current_tick=100,  # 十分経過
        )

        assert len(buffer) == 1
        assert buffer[0].is_pending is True

    def test_buffer_entry_removed_on_match(self):
        """対構成に成功したバッファエントリは除去される。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        buf = [BufferEntry(self_text="バッファ発言", self_tick=10)]
        other_units = [MockOtherUnit(tick=12, text_hint="反応")]

        new_pairs, updated_buffer = compose_pairs(
            self_records=[],
            other_units=other_units,
            buffer=buf,
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 1
        assert len(updated_buffer) == 0

    def test_empty_text_not_buffered(self):
        """空テキストはバッファに追加しない。"""
        config = InteractionAccumulationConfig()
        self_records = [MockSelfRecord(tick=1, response_text="")]

        _, buffer = compose_pairs(
            self_records=self_records,
            other_units=[],
            buffer=[],
            config=config,
            current_tick=5,
        )

        assert len(buffer) == 0

    def test_duplicate_tick_not_re_buffered(self):
        """既にバッファにあるティック番号は再追加しない。"""
        config = InteractionAccumulationConfig()
        existing_buf = [BufferEntry(self_text="既存", self_tick=10)]
        self_records = [MockSelfRecord(tick=10, response_text="重複")]

        _, buffer = compose_pairs(
            self_records=self_records,
            other_units=[],
            buffer=existing_buf,
            config=config,
            current_tick=15,
        )

        assert len(buffer) == 1
        assert buffer[0].self_text == "既存"

    def test_buffer_limit(self):
        """バッファの上限管理。"""
        config = InteractionAccumulationConfig(max_buffer=3)
        self_records = [
            MockSelfRecord(tick=i, response_text=f"発言{i}")
            for i in range(10)
        ]

        _, buffer = compose_pairs(
            self_records=self_records,
            other_units=[],
            buffer=[],
            config=config,
            current_tick=100,
        )

        assert len(buffer) <= config.max_buffer

    def test_pending_entries_not_matched(self):
        """保留状態のバッファエントリは対構成に参加しない。"""
        config = InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        pending_buf = [BufferEntry(self_text="保留", self_tick=10, is_pending=True)]
        other_units = [MockOtherUnit(tick=12, text_hint="反応")]

        new_pairs, _ = compose_pairs(
            self_records=[],
            other_units=other_units,
            buffer=pending_buf,
            config=config,
            current_tick=15,
        )

        assert len(new_pairs) == 0


# =============================================================================
# Stage 2: 対の記述テスト
# =============================================================================

class TestPairDescription:
    """AdjacentPair の記述テスト。"""

    def test_pair_fields_complete(self):
        """全フィールドが設定される。"""
        pair = AdjacentPair(
            self_text="テスト出力",
            self_policy_label="policy_a",
            self_tick=10,
            other_reaction="相手の反応",
            other_tick=12,
        )

        assert pair.self_text == "テスト出力"
        assert pair.self_policy_label == "policy_a"
        assert pair.self_tick == 10
        assert pair.other_reaction == "相手の反応"
        assert pair.other_tick == 12
        assert pair.pair_id  # IDが生成されている
        assert pair.timestamp > 0

    def test_pair_to_dict_roundtrip(self):
        """to_dict/from_dict の往復。"""
        pair = AdjacentPair(
            self_text="出力テキスト",
            self_policy_label="label_x",
            self_tick=5,
            other_reaction="反応テキスト",
            other_tick=7,
        )

        data = pair.to_dict()
        restored = AdjacentPair.from_dict(data)

        assert restored.self_text == pair.self_text
        assert restored.self_policy_label == pair.self_policy_label
        assert restored.self_tick == pair.self_tick
        assert restored.other_reaction == pair.other_reaction
        assert restored.other_tick == pair.other_tick

    def test_pair_no_weight_or_score(self):
        """対に重み・スコア・重要度の属性が存在しない。"""
        pair = AdjacentPair()
        d = pair.to_dict()

        # 重み・スコア系の属性がないことを確認
        for key in d:
            assert "weight" not in key.lower()
            assert "score" not in key.lower()
            assert "importance" not in key.lower()
            assert "priority" not in key.lower()


# =============================================================================
# Stage 3: 蓄積と消失テスト
# =============================================================================

class TestAccumulatePairs:
    """accumulate_pairs のテスト。"""

    def test_basic_accumulation(self):
        """基本的な蓄積。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=100)
        new_pairs = [
            AdjacentPair(self_text="A", other_reaction="B"),
            AdjacentPair(self_text="C", other_reaction="D"),
        ]

        accumulate_pairs(state, new_pairs, config)

        assert len(state.pairs) == 2
        assert state.total_pairs_created == 2

    def test_fifo_pushout(self):
        """上限到達時のFIFO押し出し。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=3)

        for i in range(5):
            accumulate_pairs(
                state,
                [AdjacentPair(self_text=f"text_{i}", self_tick=i)],
                config,
            )

        assert len(state.pairs) == 3
        assert state.total_pairs_created == 5
        assert state.total_pairs_pushed_out == 2
        # 最古のものが押し出されている
        assert state.pairs[0].self_text == "text_2"

    def test_fifo_is_mechanical(self):
        """FIFO押し出しは内容に基づかず機械的。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=2)

        # 異なる内容の対を蓄積
        pairs = [
            AdjacentPair(self_text="重要そう", other_reaction="重要な反応"),
            AdjacentPair(self_text="普通", other_reaction="普通の反応"),
            AdjacentPair(self_text="最新", other_reaction="最新の反応"),
        ]

        accumulate_pairs(state, pairs, config)

        # 最古のものが押し出されている（内容に関係なく）
        assert len(state.pairs) == 2
        assert state.pairs[0].self_text == "普通"
        assert state.pairs[1].self_text == "最新"

    def test_no_retroactive_modification(self):
        """既存の対は遡及的に変更されない。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=100)

        pair1 = AdjacentPair(self_text="first", self_tick=1)
        accumulate_pairs(state, [pair1], config)

        original_text = state.pairs[0].self_text
        original_tick = state.pairs[0].self_tick

        # 追加の蓄積
        pair2 = AdjacentPair(self_text="second", self_tick=2)
        accumulate_pairs(state, [pair2], config)

        # 最初の対は変更されていない
        assert state.pairs[0].self_text == original_text
        assert state.pairs[0].self_tick == original_tick


# =============================================================================
# Stage 4: 参照情報の提供テスト
# =============================================================================

class TestEnrichmentPreparation:
    """参照情報の提供テスト。"""

    def test_enrichment_returns_recent_pairs(self):
        """enrichmentは直近の対を返す。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(enrichment_count=3, max_pairs=100)

        for i in range(10):
            state.pairs.append(
                AdjacentPair(self_text=f"text_{i}", self_tick=i)
            )

        result = prepare_enrichment_pairs(state, config)

        assert len(result) <= 3
        # 直近のものが含まれる
        ticks = [p.self_tick for p in result]
        assert max(ticks) == 9

    def test_enrichment_respects_count_limit(self):
        """enrichmentの件数上限。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(enrichment_count=2)

        for i in range(10):
            state.pairs.append(AdjacentPair(self_text=f"text_{i}"))

        result = prepare_enrichment_pairs(state, config)
        assert len(result) <= 2

    def test_enrichment_empty_when_no_pairs(self):
        """対がない場合は空リスト。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig()

        result = prepare_enrichment_pairs(state, config)
        assert result == []

    def test_rumination_prevention(self):
        """ルーミネーション防止: 連続列挙上限超過で除外。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(
            enrichment_count=3,
            rumination_consecutive_limit=2,
        )

        pair = AdjacentPair(self_text="repeated")
        state.pairs.append(pair)

        # 連続列挙回数を上限に設定
        state.enrichment_consecutive[pair.pair_id] = 2

        result = prepare_enrichment_pairs(state, config)

        # 上限に達した対は除外される
        assert pair.pair_id not in {p.pair_id for p in result}

    def test_rumination_cooldown(self):
        """ルーミネーション: 列挙から外れるとカウントが減算される。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(
            enrichment_count=2,
            rumination_consecutive_limit=3,
        )

        # 対を追加
        for i in range(5):
            state.pairs.append(AdjacentPair(self_text=f"text_{i}", self_tick=i))

        # あるIDの連続カウントを設定
        old_pair_id = state.pairs[0].pair_id
        state.enrichment_consecutive[old_pair_id] = 2

        # enrichment取得（old_pair_idは直近ではないので列挙されない）
        prepare_enrichment_pairs(state, config)

        # 列挙されなかった場合カウントが減算される
        if old_pair_id in state.enrichment_consecutive:
            assert state.enrichment_consecutive[old_pair_id] < 2


# =============================================================================
# Processor テスト
# =============================================================================

class TestInteractionAccumulationProcessor:
    """プロセッサの統合テスト。"""

    def test_create_processor(self):
        """プロセッサの生成。"""
        proc = create_interaction_accumulation_processor()
        assert isinstance(proc, InteractionAccumulationProcessor)
        assert proc.state.cycle_count == 0

    def test_process_creates_pairs(self):
        """processで対が構成される。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )

        self_records = [MockSelfRecord(tick=10, response_text="出力")]
        other_units = [MockOtherUnit(tick=12, text_hint="反応")]

        count = proc.process(
            self_records=self_records,
            other_units=other_units,
            current_tick=15,
        )

        assert count == 1
        assert len(proc.state.pairs) == 1
        assert proc.state.cycle_count == 1

    def test_process_with_no_input(self):
        """入力なしでもエラーにならない。"""
        proc = create_interaction_accumulation_processor()

        count = proc.process(current_tick=1)

        assert count == 0
        assert proc.state.cycle_count == 1

    def test_process_increments_cycle(self):
        """processがサイクルカウントを増加させる。"""
        proc = create_interaction_accumulation_processor()

        proc.process(current_tick=1)
        proc.process(current_tick=2)
        proc.process(current_tick=3)

        assert proc.state.cycle_count == 3

    def test_get_enrichment_data(self):
        """enrichmentデータの構造。"""
        proc = create_interaction_accumulation_processor()

        # 対を追加
        proc.state.pairs.append(
            AdjacentPair(
                self_text="テスト出力テキスト",
                self_policy_label="policy_a",
                other_reaction="相手の反応テキスト",
                self_tick=10,
                other_tick=12,
            )
        )

        data = proc.get_enrichment_data()

        assert "pair_count" in data
        assert "entries" in data
        assert "summary_text" in data
        assert data["pair_count"] == 1
        assert len(data["entries"]) == 1

        entry = data["entries"][0]
        assert "self_text" in entry
        assert "self_policy" in entry
        assert "other_reaction" in entry
        assert "self_tick" in entry
        assert "other_tick" in entry

    def test_get_enrichment_data_text_truncation(self):
        """enrichmentのテキスト切り詰め。"""
        proc = create_interaction_accumulation_processor()

        long_text = "あ" * 200
        proc.state.pairs.append(
            AdjacentPair(self_text=long_text, other_reaction=long_text)
        )

        data = proc.get_enrichment_data()
        entry = data["entries"][0]

        assert len(entry["self_text"]) <= 84  # 80 + "..."
        assert len(entry["other_reaction"]) <= 84

    def test_get_latest_pairs(self):
        """直近対のREAD-ONLY取得。"""
        proc = create_interaction_accumulation_processor()

        for i in range(10):
            proc.state.pairs.append(
                AdjacentPair(self_text=f"text_{i}", self_tick=i)
            )

        result = proc.get_latest_pairs(count=3)

        assert len(result) == 3
        assert result[-1].self_text == "text_9"

    def test_get_latest_pairs_default_count(self):
        """get_latest_pairsのデフォルト件数。"""
        config = InteractionAccumulationConfig(enrichment_count=5)
        proc = InteractionAccumulationProcessor(config=config)

        for i in range(20):
            proc.state.pairs.append(
                AdjacentPair(self_text=f"text_{i}")
            )

        result = proc.get_latest_pairs()
        assert len(result) == 5

    def test_get_pair_history(self):
        """対履歴のREAD-ONLY取得。"""
        config = InteractionAccumulationConfig(reference_history_count=10)
        proc = InteractionAccumulationProcessor(config=config)

        for i in range(20):
            proc.state.pairs.append(
                AdjacentPair(self_text=f"text_{i}", self_tick=i)
            )

        result = proc.get_pair_history()

        assert len(result) == 10
        assert result[0].self_tick == 10

    def test_get_summary(self):
        """サマリの構造。"""
        proc = create_interaction_accumulation_processor()
        proc.state.total_pairs_created = 5
        proc.state.total_pairs_pushed_out = 2
        proc.state.cycle_count = 3

        summary = proc.get_summary()

        assert summary["total_pairs_created"] == 5
        assert summary["total_pairs_pushed_out"] == 2
        assert summary["cycle_count"] == 3

    def test_state_property_getter_setter(self):
        """stateプロパティのgetter/setter。"""
        proc = create_interaction_accumulation_processor()

        new_state = InteractionAccumulationState(cycle_count=42)
        proc.state = new_state

        assert proc.state.cycle_count == 42


# =============================================================================
# Save/Load テスト
# =============================================================================

class TestSaveLoad:
    """永続化テスト。"""

    def test_state_to_dict_from_dict_roundtrip(self):
        """state の to_dict/from_dict 往復。"""
        state = InteractionAccumulationState(
            pairs=[
                AdjacentPair(
                    self_text="出力A",
                    self_policy_label="p1",
                    self_tick=5,
                    other_reaction="反応A",
                    other_tick=7,
                ),
                AdjacentPair(
                    self_text="出力B",
                    self_policy_label="p2",
                    self_tick=10,
                    other_reaction="反応B",
                    other_tick=12,
                ),
            ],
            buffer=[
                BufferEntry(
                    self_text="バッファ",
                    self_policy_label="p3",
                    self_tick=15,
                    is_pending=True,
                ),
            ],
            enrichment_consecutive={"abc": 2, "def": 1},
            total_pairs_created=10,
            total_pairs_pushed_out=3,
            cycle_count=5,
        )

        data = state.to_dict()
        restored = InteractionAccumulationState.from_dict(data)

        assert len(restored.pairs) == 2
        assert restored.pairs[0].self_text == "出力A"
        assert restored.pairs[1].other_reaction == "反応B"
        assert len(restored.buffer) == 1
        assert restored.buffer[0].is_pending is True
        assert restored.enrichment_consecutive == {"abc": 2, "def": 1}
        assert restored.total_pairs_created == 10
        assert restored.total_pairs_pushed_out == 3
        assert restored.cycle_count == 5

    def test_empty_state_roundtrip(self):
        """空状態の往復。"""
        state = InteractionAccumulationState()
        data = state.to_dict()
        restored = InteractionAccumulationState.from_dict(data)

        assert len(restored.pairs) == 0
        assert len(restored.buffer) == 0
        assert restored.total_pairs_created == 0

    def test_buffer_entry_roundtrip(self):
        """BufferEntry の往復。"""
        entry = BufferEntry(
            self_text="テスト",
            self_policy_label="p1",
            self_tick=5,
            is_pending=True,
        )

        data = entry.to_dict()
        restored = BufferEntry.from_dict(data)

        assert restored.self_text == entry.self_text
        assert restored.self_policy_label == entry.self_policy_label
        assert restored.self_tick == entry.self_tick
        assert restored.is_pending == entry.is_pending

    def test_from_dict_with_empty_data(self):
        """空dictからの復元。"""
        restored = InteractionAccumulationState.from_dict({})

        assert len(restored.pairs) == 0
        assert len(restored.buffer) == 0
        assert restored.total_pairs_created == 0

    def test_from_dict_with_partial_data(self):
        """部分的なdictからの復元。"""
        data = {
            "total_pairs_created": 42,
            "cycle_count": 7,
        }
        restored = InteractionAccumulationState.from_dict(data)

        assert restored.total_pairs_created == 42
        assert restored.cycle_count == 7
        assert len(restored.pairs) == 0


# =============================================================================
# 安全弁テスト
# =============================================================================

class TestSafetyValves:
    """安全弁テスト。"""

    def test_all_records_equal_no_weights(self):
        """安全弁1: 全記録の等価性 - 重み・スコア・重要度が存在しない。"""
        pair = AdjacentPair(self_text="test", other_reaction="reaction")
        d = pair.to_dict()

        forbidden_keys = ["weight", "score", "importance", "priority", "rank"]
        for key in d:
            for forbidden in forbidden_keys:
                assert forbidden not in key.lower(), \
                    f"Forbidden key pattern '{forbidden}' found in '{key}'"

    def test_fifo_pushout_no_selective_retention(self):
        """安全弁2: FIFO自然消失 - 選択的保持を行わない。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=2)

        # 「重要そうな」対と「普通の」対を交互に蓄積
        for i in range(5):
            accumulate_pairs(
                state,
                [AdjacentPair(self_text=f"item_{i}", self_tick=i)],
                config,
            )

        # 最古のものが押し出されている（内容に依存しない）
        assert len(state.pairs) == 2
        assert state.pairs[0].self_text == "item_3"
        assert state.pairs[1].self_text == "item_4"

    def test_no_pushout_recovery(self):
        """安全弁2: 押し出された対の復帰経路がない。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(max_pairs=2)

        # 3対蓄積して最初のものを押し出す
        pair1 = AdjacentPair(pair_id="pushed_out", self_text="old")
        pair2 = AdjacentPair(self_text="mid")
        pair3 = AdjacentPair(self_text="new")

        accumulate_pairs(state, [pair1, pair2, pair3], config)

        # 押し出された対のIDは存在しない
        pair_ids = {p.pair_id for p in state.pairs}
        assert "pushed_out" not in pair_ids

    def test_rumination_prevention_enrichment(self):
        """安全弁3: ルーミネーション防止。"""
        state = InteractionAccumulationState()
        config = InteractionAccumulationConfig(
            enrichment_count=3,
            rumination_consecutive_limit=2,
        )

        # 少数の対を追加
        for i in range(3):
            state.pairs.append(AdjacentPair(self_text=f"text_{i}"))

        # 全対の連続カウントを上限に設定
        for pair in state.pairs:
            state.enrichment_consecutive[pair.pair_id] = 2

        result = prepare_enrichment_pairs(state, config)

        # 全て除外される
        assert len(result) == 0

    def test_no_pattern_extraction(self):
        """安全弁4: パターン抽出の構造的排除。

        InteractionAccumulationProcessor に統計量・頻度分布・傾向・規則性を
        算出するメソッドが存在しないことを確認する。
        """
        proc = create_interaction_accumulation_processor()

        # パターン抽出系のメソッドが存在しないことを確認
        forbidden_methods = [
            "extract_pattern", "analyze_pattern", "get_tendency",
            "get_frequency", "get_statistics", "compute_trend",
            "detect_pattern", "classify", "categorize",
            "get_distribution", "compute_correlation",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(proc, method_name), \
                f"Forbidden method '{method_name}' found on processor"

    def test_no_judgment_system_connection(self):
        """安全弁5: 判断系への経路遮断。

        プロセッサの出力メソッドがenrichmentとREAD-ONLY参照のみであることを確認。
        """
        proc = create_interaction_accumulation_processor()

        # 判断系接続を示唆するメソッドが存在しないことを確認
        forbidden_methods = [
            "apply_to_policy", "apply_to_bias", "update_policy",
            "modify_stability", "adjust_value", "influence_decision",
            "get_signal", "compute_bias", "update_orientation",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(proc, method_name), \
                f"Forbidden method '{method_name}' found on processor"

    def test_no_causal_attribution(self):
        """因果帰属を行わない。対に因果関係の属性がない。"""
        pair = AdjacentPair(self_text="X", other_reaction="Y")
        d = pair.to_dict()

        forbidden_keys = ["cause", "effect", "because", "result_of", "reason"]
        for key in d:
            for forbidden in forbidden_keys:
                assert forbidden not in key.lower(), \
                    f"Causal attribution key '{forbidden}' found in '{key}'"

    def test_no_evaluation_attributes(self):
        """相互作用の良否判定を行わない。"""
        pair = AdjacentPair(self_text="X", other_reaction="Y")
        d = pair.to_dict()

        forbidden_keys = ["good", "bad", "quality", "rating", "evaluation"]
        for key in d:
            for forbidden in forbidden_keys:
                assert forbidden not in key.lower(), \
                    f"Evaluation key '{forbidden}' found in '{key}'"

    def test_no_prediction_structure(self):
        """反応の予測を行わない。"""
        proc = create_interaction_accumulation_processor()

        forbidden_methods = [
            "predict", "forecast", "estimate_next", "expected_reaction",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(proc, method_name), \
                f"Prediction method '{method_name}' found on processor"


# =============================================================================
# エッジケーステスト
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_empty_inputs(self):
        """空入力。"""
        proc = create_interaction_accumulation_processor()
        count = proc.process(self_records=[], other_units=[], current_tick=0)
        assert count == 0

    def test_none_inputs(self):
        """None入力。"""
        proc = create_interaction_accumulation_processor()
        count = proc.process(self_records=None, other_units=None, current_tick=0)
        assert count == 0

    def test_very_long_text(self):
        """非常に長いテキスト。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )
        long_text = "あ" * 10000
        self_records = [MockSelfRecord(tick=10, response_text=long_text)]
        other_units = [MockOtherUnit(tick=12, text_hint="短い反応")]

        count = proc.process(self_records=self_records, other_units=other_units, current_tick=15)
        assert count == 1
        assert len(proc.state.pairs[0].self_text) == 10000

    def test_zero_tick(self):
        """ティック0での処理。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )
        self_records = [MockSelfRecord(tick=0, response_text="初期出力")]
        other_units = [MockOtherUnit(tick=2, text_hint="初期反応")]

        count = proc.process(self_records=self_records, other_units=other_units, current_tick=5)
        assert count == 1

    def test_large_tick_numbers(self):
        """大きなティック番号。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )
        self_records = [MockSelfRecord(tick=1000000, response_text="大きいティック")]
        other_units = [MockOtherUnit(tick=1000003, text_hint="大きい反応")]

        count = proc.process(
            self_records=self_records,
            other_units=other_units,
            current_tick=1000010,
        )
        assert count == 1

    def test_negative_tick_gap(self):
        """他者反応が自己表出より前のティック（逆順）。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )
        self_records = [MockSelfRecord(tick=10, response_text="自分")]
        other_units = [MockOtherUnit(tick=8, text_hint="先の反応")]  # 自分より前

        count = proc.process(self_records=self_records, other_units=other_units, current_tick=15)
        assert count == 0  # 負のtick_diffは対構成しない

    def test_many_pairs_performance(self):
        """多数の対の蓄積。"""
        config = InteractionAccumulationConfig(max_pairs=1000)
        proc = InteractionAccumulationProcessor(config=config)

        for i in range(500):
            proc.state.pairs.append(
                AdjacentPair(self_text=f"t_{i}", other_reaction=f"r_{i}")
            )

        # enrichmentが正常に動作する
        data = proc.get_enrichment_data()
        assert data["pair_count"] == 500

    def test_extract_reaction_with_fragments(self):
        """フラグメントを持つユニットからの反応記述抽出。"""
        frag_type = MockFragmentType(value="speech_reaction")
        fragments = [
            MockFragment(type=frag_type, text_hint="言語反応"),
        ]
        unit = MockOtherUnit(fragments=fragments)

        desc = _extract_reaction_description(unit)
        assert "speech_reaction" in desc
        assert "言語反応" in desc

    def test_extract_reaction_with_summary(self):
        """summary属性からの反応記述抽出。"""
        unit = MockOtherUnit(text_hint="", summary="要約テキスト")

        desc = _extract_reaction_description(unit)
        assert desc == "要約テキスト"

    def test_extract_reaction_empty(self):
        """反応記述がない場合。"""
        unit = MockOtherUnit(text_hint="", source_description="", summary="")

        desc = _extract_reaction_description(unit)
        assert desc == ""

    def test_multiple_cycles(self):
        """複数サイクルにわたる処理。"""
        proc = create_interaction_accumulation_processor(
            InteractionAccumulationConfig(tick_proximity_range=5, min_tick_gap=1)
        )

        for cycle in range(10):
            base_tick = cycle * 10
            self_records = [
                MockSelfRecord(tick=base_tick, response_text=f"出力{cycle}")
            ]
            other_units = [
                MockOtherUnit(tick=base_tick + 2, text_hint=f"反応{cycle}")
            ]
            proc.process(
                self_records=self_records,
                other_units=other_units,
                current_tick=base_tick + 5,
            )

        assert proc.state.cycle_count == 10
        assert len(proc.state.pairs) == 10

    def test_buffered_self_matched_later(self):
        """バッファに入った自己表出が後のサイクルで他者反応と対構成される。"""
        config = InteractionAccumulationConfig(
            tick_proximity_range=5,
            min_tick_gap=1,
            buffer_timeout_ticks=20,
        )
        proc = InteractionAccumulationProcessor(config=config)

        # サイクル1: 自己表出のみ（バッファに入る）
        self_records = [MockSelfRecord(tick=10, response_text="先行発言")]
        proc.process(self_records=self_records, other_units=[], current_tick=10)

        assert len(proc.state.buffer) == 1
        assert len(proc.state.pairs) == 0

        # サイクル2: 他者反応が到着
        other_units = [MockOtherUnit(tick=13, text_hint="遅延反応")]
        count = proc.process(self_records=[], other_units=other_units, current_tick=15)

        assert count == 1
        assert len(proc.state.pairs) == 1
        assert proc.state.pairs[0].self_text == "先行発言"
        assert proc.state.pairs[0].other_reaction == "遅延反応"


# =============================================================================
# Summary テスト
# =============================================================================

class TestSummary:
    """summary関数のテスト。"""

    def test_summary_waiting(self):
        """初期状態のサマリ。"""
        state = InteractionAccumulationState()
        text = get_interaction_summary(state)
        assert "待機中" in text

    def test_summary_with_pairs(self):
        """対がある状態のサマリ。"""
        state = InteractionAccumulationState(
            cycle_count=5,
            total_pairs_pushed_out=2,
        )
        state.pairs.append(AdjacentPair(self_text="test"))

        text = get_interaction_summary(state)
        assert "cycle=5" in text
        assert "蓄積対=1" in text
        assert "押出累計=2" in text

    def test_summary_with_buffer(self):
        """バッファがある状態のサマリ。"""
        state = InteractionAccumulationState(cycle_count=1)
        state.buffer.append(BufferEntry(self_text="buf"))

        text = get_interaction_summary(state)
        assert "バッファ=1" in text

    def test_summary_no_evaluation(self):
        """サマリに評価的語彙が含まれない。"""
        state = InteractionAccumulationState(cycle_count=10)
        for i in range(5):
            state.pairs.append(AdjacentPair(self_text=f"t_{i}"))

        text = get_interaction_summary(state)

        forbidden = ["良い", "悪い", "改善", "異常", "正常", "望ましい"]
        for word in forbidden:
            assert word not in text
