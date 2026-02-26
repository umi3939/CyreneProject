"""
tests/test_memory_emotion_return.py - 記憶想起から感情への帰還経路テスト

設計書 (design_memory_emotion_return.md) に基づく検証:
- 4段パイプラインの正常動作
- 安全弁7種の適切な適用
- enrichment非露出
- 想起モジュールへの逆流なし
- 忘却処理への経路なし
- ルーミネーション減衰
- 感情値有効範囲クランプ
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from psyche.memory_emotion_return import (
    MemoryEmotionReturnProcessor,
    MemoryEmotionReturnState,
    MemoryEmotionReturnConfig,
    ReturnRecord,
    ReturnResult,
    create_memory_emotion_return,
    _collect_trace_data,
    _derive_return_amounts,
    _apply_safety_valves,
    _apply_and_record,
    _clamp,
)


# =============================================================================
# Mock Objects
# =============================================================================

@dataclass
class MockTrace:
    """Mock emotional trace."""
    emotion_label: str = "joy"
    intensity: float = 0.5
    valence: float = 0.3
    freshness: float = 0.8


@dataclass
class MockBinding:
    """Mock memory binding."""
    memory_key: str = "unit_001"
    traces: tuple = ()


@dataclass
class MockBindingStore:
    """Mock binding store."""
    bindings: tuple = ()


@dataclass
class MockCandidate:
    """Mock recall candidate."""
    unit_id: str = "unit_001"
    source_id: str = "src_001"
    summary: str = "test memory"
    path_label: str = "emotional"


# =============================================================================
# Helper Functions
# =============================================================================

def _make_binding_store(entries: list[tuple[str, list[MockTrace]]]) -> MockBindingStore:
    """Create a mock binding store from a list of (memory_key, traces) pairs."""
    bindings = []
    for memory_key, traces in entries:
        bindings.append(MockBinding(memory_key=memory_key, traces=tuple(traces)))
    return MockBindingStore(bindings=tuple(bindings))


def _make_candidates(unit_ids: list[str]) -> list[MockCandidate]:
    """Create a list of mock candidates."""
    return [MockCandidate(unit_id=uid) for uid in unit_ids]


# =============================================================================
# Test: Factory and Initialization
# =============================================================================

class TestFactory:
    def test_create_default(self):
        proc = create_memory_emotion_return()
        assert isinstance(proc, MemoryEmotionReturnProcessor)
        assert proc.state.cycle_count == 0
        assert proc.state.last_applied_tick == -1
        assert proc.state.return_history == []

    def test_create_with_config(self):
        cfg = MemoryEmotionReturnConfig(history_window_size=100)
        proc = create_memory_emotion_return(config=cfg)
        assert proc._config.history_window_size == 100

    def test_state_serialization(self):
        state = MemoryEmotionReturnState(
            return_history=[
                ReturnRecord(unit_id="u1", recall_system_label="multi_path")
            ],
            last_applied_tick=5,
            cycle_count=3,
        )
        d = state.to_dict()
        restored = MemoryEmotionReturnState.from_dict(d)
        assert restored.cycle_count == 3
        assert restored.last_applied_tick == 5
        assert len(restored.return_history) == 1
        assert restored.return_history[0].unit_id == "u1"

    def test_state_empty_serialization(self):
        state = MemoryEmotionReturnState()
        d = state.to_dict()
        restored = MemoryEmotionReturnState.from_dict(d)
        assert restored.cycle_count == 0
        assert restored.last_applied_tick == -1
        assert restored.return_history == []


# =============================================================================
# Test: ReturnRecord
# =============================================================================

class TestReturnRecord:
    def test_serialization(self):
        rec = ReturnRecord(
            unit_id="u1",
            recall_system_label="spontaneous",
            emotion_labels=["joy", "sorrow"],
            emotion_deltas={"joy": 0.01, "sorrow": -0.005},
            mood_direction=0.2,
            arousal_level=0.5,
            rumination_decay_applied=True,
            timestamp=12345.0,
            tick_number=10,
        )
        d = rec.to_dict()
        restored = ReturnRecord.from_dict(d)
        assert restored.unit_id == "u1"
        assert restored.recall_system_label == "spontaneous"
        assert restored.emotion_labels == ["joy", "sorrow"]
        assert restored.emotion_deltas["joy"] == 0.01
        assert restored.rumination_decay_applied is True
        assert restored.tick_number == 10


# =============================================================================
# Test: ReturnResult
# =============================================================================

class TestReturnResult:
    def test_default(self):
        result = ReturnResult()
        assert result.emotion_deltas == {}
        assert result.records_created == 0
        assert result.total_candidates_processed == 0
        assert result.candidates_with_traces == 0

    def test_to_dict(self):
        result = ReturnResult(
            emotion_deltas={"joy": 0.02},
            records_created=1,
            total_candidates_processed=5,
            candidates_with_traces=2,
        )
        d = result.to_dict()
        assert d["emotion_deltas"]["joy"] == 0.02
        assert d["records_created"] == 1


# =============================================================================
# Test: Stage 1 - Trace Collection
# =============================================================================

class TestStage1:
    def test_empty_candidates(self):
        store = _make_binding_store([])
        result = _collect_trace_data([], store, "multi_path")
        assert result == []

    def test_none_binding_store(self):
        candidates = _make_candidates(["u1"])
        result = _collect_trace_data(candidates, None, "multi_path")
        assert result == []

    def test_no_matching_traces(self):
        store = _make_binding_store([
            ("unrelated_key", [MockTrace()])
        ])
        candidates = _make_candidates(["u1"])
        result = _collect_trace_data(candidates, store, "multi_path")
        assert result == []

    def test_matching_by_unit_id(self):
        store = _make_binding_store([
            ("unit_001", [MockTrace(emotion_label="joy", intensity=0.6, valence=0.4, freshness=0.9)])
        ])
        candidates = _make_candidates(["unit_001"])
        result = _collect_trace_data(candidates, store, "multi_path")
        assert len(result) == 1
        assert result[0]["unit_id"] == "unit_001"
        assert result[0]["system_label"] == "multi_path"
        assert len(result[0]["traces"]) == 1
        assert result[0]["traces"][0]["emotion_label"] == "joy"

    def test_matching_by_source_id(self):
        store = _make_binding_store([
            ("src_001", [MockTrace(emotion_label="sorrow", intensity=0.4)])
        ])
        candidates = [MockCandidate(unit_id="no_match", source_id="src_001")]
        result = _collect_trace_data(candidates, store, "spontaneous")
        assert len(result) == 1
        assert result[0]["system_label"] == "spontaneous"

    def test_multiple_traces_per_binding(self):
        store = _make_binding_store([
            ("unit_001", [
                MockTrace(emotion_label="joy", intensity=0.6, valence=0.3),
                MockTrace(emotion_label="love", intensity=0.4, valence=0.5),
            ])
        ])
        candidates = _make_candidates(["unit_001"])
        result = _collect_trace_data(candidates, store, "multi_path")
        assert len(result[0]["traces"]) == 2

    def test_two_systems_equal(self):
        """2系統の候補は等価に扱われる（系統間の優先順位なし）"""
        store = _make_binding_store([
            ("unit_mpr", [MockTrace(emotion_label="joy", intensity=0.5)]),
            ("unit_sr", [MockTrace(emotion_label="sorrow", intensity=0.5)]),
        ])
        mpr_candidates = _make_candidates(["unit_mpr"])
        sr_candidates = _make_candidates(["unit_sr"])

        mpr_traces = _collect_trace_data(mpr_candidates, store, "multi_path")
        sr_traces = _collect_trace_data(sr_candidates, store, "spontaneous")

        all_traces = mpr_traces + sr_traces
        assert len(all_traces) == 2

    def test_candidate_without_trace_excluded(self):
        """紐づけが存在しない候補は帰還対象から除外される"""
        store = _make_binding_store([
            ("unit_001", [MockTrace()])
        ])
        candidates = _make_candidates(["unit_001", "unit_002_no_trace"])
        result = _collect_trace_data(candidates, store, "multi_path")
        assert len(result) == 1
        assert result[0]["unit_id"] == "unit_001"


# =============================================================================
# Test: Stage 2 - Return Amount Derivation
# =============================================================================

class TestStage2:
    def test_basic_derivation(self):
        trace_entries = [{
            "unit_id": "u1",
            "system_label": "multi_path",
            "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ],
        }]
        config = MemoryEmotionReturnConfig()
        result = _derive_return_amounts(
            trace_entries,
            current_emotions={"joy": 0.2, "sorrow": 0.0},
            mood_valence=0.1,
            mood_arousal=0.5,
            config=config,
        )
        assert len(result) == 1
        assert "joy" in result[0]["deltas"]
        assert result[0]["deltas"]["joy"] > 0

    def test_intensity_scales_amount(self):
        """痕跡が強いほど帰還量が大きい"""
        config = MemoryEmotionReturnConfig()
        weak = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.1, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, 0.0, 0.5, config,
        )
        strong = _derive_return_amounts(
            [{"unit_id": "u2", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.9, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, 0.0, 0.5, config,
        )
        assert abs(strong[0]["deltas"]["joy"]) > abs(weak[0]["deltas"]["joy"])

    def test_freshness_scales_amount(self):
        """鮮度が低いほど帰還量が縮小する"""
        config = MemoryEmotionReturnConfig()
        fresh = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 1.0}
            ]}],
            {"joy": 0.0}, 0.0, 0.5, config,
        )
        stale = _derive_return_amounts(
            [{"unit_id": "u2", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.1}
            ]}],
            {"joy": 0.0}, 0.0, 0.5, config,
        )
        assert abs(fresh[0]["deltas"]["joy"]) > abs(stale[0]["deltas"]["joy"])

    def test_convergence_reduces_return(self):
        """既に高い感情への帰還は収束的に縮小する"""
        config = MemoryEmotionReturnConfig()
        low_current = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.1}, 0.0, 0.5, config,
        )
        high_current = _derive_return_amounts(
            [{"unit_id": "u2", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.8}, 0.0, 0.5, config,
        )
        assert abs(low_current[0]["deltas"]["joy"]) > abs(high_current[0]["deltas"]["joy"])

    def test_low_arousal_dampens(self):
        """低覚醒時は帰還量全体が鈍化する"""
        config = MemoryEmotionReturnConfig()
        high_arousal = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, 0.0, 0.7, config,
        )
        low_arousal = _derive_return_amounts(
            [{"unit_id": "u2", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, 0.0, 0.05, config,
        )
        assert abs(high_arousal[0]["deltas"]["joy"]) > abs(low_arousal[0]["deltas"]["joy"])

    def test_negative_valence_negative_delta(self):
        """負の感情価は負方向の帰還量を生成する"""
        config = MemoryEmotionReturnConfig()
        result = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "sorrow", "intensity": 0.5, "valence": -0.4, "freshness": 0.8}
            ]}],
            {"sorrow": 0.0}, 0.0, 0.5, config,
        )
        assert result[0]["deltas"]["sorrow"] < 0

    def test_direction_not_fixed(self):
        """帰還量の方向は固定値ではなく、ムード方向との関係で都度異なりうる"""
        config = MemoryEmotionReturnConfig()
        # Same trace, different mood directions
        r1 = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, mood_valence=0.5, mood_arousal=0.5, config=config,
        )
        r2 = _derive_return_amounts(
            [{"unit_id": "u2", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.8}
            ]}],
            {"joy": 0.0}, mood_valence=-0.5, mood_arousal=0.5, config=config,
        )
        # The amounts should differ due to mood alignment
        assert abs(r1[0]["deltas"]["joy"]) != abs(r2[0]["deltas"]["joy"])

    def test_zero_freshness_zero_return(self):
        """鮮度ゼロの痕跡は帰還量ゼロ"""
        config = MemoryEmotionReturnConfig()
        result = _derive_return_amounts(
            [{"unit_id": "u1", "system_label": "m", "traces": [
                {"emotion_label": "joy", "intensity": 0.5, "valence": 0.3, "freshness": 0.0}
            ]}],
            {"joy": 0.0}, 0.0, 0.5, config,
        )
        # With freshness=0.0, base_amount *= 0.0 = 0.0
        assert abs(result[0]["deltas"].get("joy", 0.0)) < 1e-9


# =============================================================================
# Test: Stage 3 - Safety Valves
# =============================================================================

class TestStage3:
    def test_per_candidate_max_delta(self):
        """安全弁1: 候補別帰還量上限"""
        config = MemoryEmotionReturnConfig(per_candidate_max_delta=0.02)
        entries = [{
            "unit_id": "u1", "system_label": "m",
            "deltas": {"joy": 0.1, "sorrow": -0.1},
        }]
        modified, total = _apply_safety_valves(entries, [], config, 0.15)
        assert modified[0]["deltas"]["joy"] <= 0.02
        assert modified[0]["deltas"]["sorrow"] >= -0.02

    def test_total_max_delta(self):
        """安全弁2: 合成後総帰還量上限"""
        config = MemoryEmotionReturnConfig(per_candidate_max_delta=0.05)
        entries = [
            {"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.04}},
            {"unit_id": "u2", "system_label": "m", "deltas": {"joy": 0.04}},
            {"unit_id": "u3", "system_label": "m", "deltas": {"joy": 0.04}},
        ]
        # total_max_delta should be min(config.total_max_delta, max_bias_strength)
        modified, total = _apply_safety_valves(entries, [], config, max_bias_strength=0.10)
        assert total["joy"] <= 0.10

    def test_total_max_uses_max_bias_strength(self):
        """安全弁2: max_bias_strength以下に制限"""
        config = MemoryEmotionReturnConfig(
            per_candidate_max_delta=0.1, total_max_delta=0.5,
        )
        entries = [
            {"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.08}},
            {"unit_id": "u2", "system_label": "m", "deltas": {"joy": 0.08}},
        ]
        modified, total = _apply_safety_valves(entries, [], config, max_bias_strength=0.15)
        assert total["joy"] <= 0.15

    def test_rumination_decay(self):
        """安全弁3: ルーミネーション減衰"""
        config = MemoryEmotionReturnConfig(
            rumination_threshold=2, rumination_decay_factor=0.5,
        )
        # u1 appears 3 times in history (over threshold of 2)
        history = [
            ReturnRecord(unit_id="u1"),
            ReturnRecord(unit_id="u1"),
            ReturnRecord(unit_id="u1"),
        ]
        entries = [
            {"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.03}},
            {"unit_id": "u2", "system_label": "m", "deltas": {"joy": 0.03}},
        ]
        modified, total = _apply_safety_valves(entries, history, config, 0.15)
        # u1 should have decay applied
        assert modified[0]["rumination_applied"] is True
        # u2 should not have decay
        assert modified[1]["rumination_applied"] is False
        # u1's delta should be smaller than u2's
        assert abs(modified[0]["deltas"]["joy"]) < abs(modified[1]["deltas"]["joy"])

    def test_rumination_convergence_to_zero(self):
        """安全弁3: 十分な回数を超えると帰還量がゼロに収束"""
        config = MemoryEmotionReturnConfig(
            rumination_threshold=2, rumination_decay_factor=0.5,
        )
        # u1 appears many times
        history = [ReturnRecord(unit_id="u1") for _ in range(10)]
        entries = [
            {"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.03}},
        ]
        modified, total = _apply_safety_valves(entries, history, config, 0.15)
        # With 10 occurrences, decay_multiplier = 1 - (10 - 2 + 1) * 0.5 = 1 - 4.5 = -3.5 -> 0.0
        assert abs(modified[0]["deltas"]["joy"]) < 1e-9


# =============================================================================
# Test: Stage 4 - Apply and Record
# =============================================================================

class TestStage4:
    def test_basic_apply(self):
        config = MemoryEmotionReturnConfig()
        state = MemoryEmotionReturnState()
        total_deltas = {"joy": 0.02, "sorrow": -0.01}
        entries = [{
            "unit_id": "u1", "system_label": "multi_path",
            "deltas": {"joy": 0.02, "sorrow": -0.01},
            "rumination_applied": False,
        }]
        clamped, records = _apply_and_record(
            total_deltas, entries,
            {"joy": 0.3, "sorrow": 0.1},
            0.1, 0.5, 10, state, config,
        )
        assert "joy" in clamped
        assert abs(clamped["joy"] - 0.02) < 1e-6
        assert len(records) == 1
        assert records[0].unit_id == "u1"

    def test_clamp_upper(self):
        """安全弁5: 感情値有効範囲クランプ (上限)"""
        config = MemoryEmotionReturnConfig()
        state = MemoryEmotionReturnState()
        total_deltas = {"joy": 0.1}
        entries = [{"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.1}, "rumination_applied": False}]
        clamped, records = _apply_and_record(
            total_deltas, entries,
            {"joy": 0.95}, 0.0, 0.5, 10, state, config,
        )
        # joy = 0.95 + 0.1 = 1.05 -> clamped to 1.0 -> actual delta = 0.05
        assert clamped["joy"] <= 0.05 + 1e-6

    def test_clamp_lower(self):
        """安全弁5: 感情値有効範囲クランプ (下限)"""
        config = MemoryEmotionReturnConfig()
        state = MemoryEmotionReturnState()
        total_deltas = {"sorrow": -0.1}
        entries = [{"unit_id": "u1", "system_label": "m", "deltas": {"sorrow": -0.1}, "rumination_applied": False}]
        clamped, records = _apply_and_record(
            total_deltas, entries,
            {"sorrow": 0.03}, 0.0, 0.5, 10, state, config,
        )
        # sorrow = 0.03 + (-0.1) = -0.07 -> clamped to 0.0 -> actual delta = -0.03
        assert clamped["sorrow"] >= -0.03 - 1e-6

    def test_zero_delta_not_recorded(self):
        """帰還量がゼロの場合は記録しない"""
        config = MemoryEmotionReturnConfig()
        state = MemoryEmotionReturnState()
        total_deltas = {"joy": 0.0}
        entries = [{"unit_id": "u1", "system_label": "m", "deltas": {"joy": 0.0}, "rumination_applied": False}]
        clamped, records = _apply_and_record(
            total_deltas, entries,
            {"joy": 0.5}, 0.0, 0.5, 10, state, config,
        )
        assert len(clamped) == 0
        assert len(records) == 0


# =============================================================================
# Test: Full Pipeline (Processor.process)
# =============================================================================

class TestProcessorPipeline:
    def test_empty_input(self):
        proc = create_memory_emotion_return()
        result = proc.process(tick_number=1)
        assert result.emotion_deltas == {}
        assert result.records_created == 0

    def test_no_binding_store(self):
        proc = create_memory_emotion_return()
        result = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=None,
            tick_number=1,
        )
        assert result.emotion_deltas == {}
        assert result.total_candidates_processed == 1

    def test_no_matching_traces(self):
        proc = create_memory_emotion_return()
        store = _make_binding_store([("unrelated", [MockTrace()])])
        result = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            tick_number=1,
        )
        assert result.emotion_deltas == {}

    def test_basic_return(self):
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("unit_001", [
                MockTrace(emotion_label="joy", intensity=0.6, valence=0.4, freshness=0.9)
            ])
        ])
        result = proc.process(
            multi_path_candidates=_make_candidates(["unit_001"]),
            binding_store=store,
            current_emotions={"joy": 0.2, "sorrow": 0.0, "anger": 0.0,
                              "fear": 0.0, "surprise": 0.0, "love": 0.0, "fun": 0.0},
            mood_valence=0.1,
            mood_arousal=0.5,
            tick_number=1,
        )
        assert "joy" in result.emotion_deltas
        assert result.emotion_deltas["joy"] > 0
        assert result.records_created > 0
        assert result.candidates_with_traces == 1

    def test_both_systems_contribute(self):
        """2系統の候補は等価に扱われる"""
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("unit_mpr", [MockTrace(emotion_label="joy", intensity=0.5, freshness=0.8, valence=0.3)]),
            ("unit_sr", [MockTrace(emotion_label="sorrow", intensity=0.5, freshness=0.8, valence=-0.3)]),
        ])
        result = proc.process(
            multi_path_candidates=_make_candidates(["unit_mpr"]),
            spontaneous_candidates=_make_candidates(["unit_sr"]),
            binding_store=store,
            current_emotions={"joy": 0.0, "sorrow": 0.0},
            mood_arousal=0.5,
            tick_number=1,
        )
        # Both systems should contribute
        assert result.candidates_with_traces == 2

    def test_cycle_count_increments(self):
        proc = create_memory_emotion_return()
        proc.process(tick_number=1)
        assert proc.state.cycle_count == 1
        proc.process(tick_number=2)
        assert proc.state.cycle_count == 2


# =============================================================================
# Test: Safety Valve 4 - Same Tick Prevention
# =============================================================================

class TestSameTickPrevention:
    def test_same_tick_blocked(self):
        """安全弁4: 同一ティック内での二重適用を防止"""
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.6, valence=0.4, freshness=0.9)])
        ])
        # First call at tick 5
        r1 = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            tick_number=5,
        )
        assert r1.emotion_deltas.get("joy", 0) > 0

        # Second call at same tick 5 should be blocked
        r2 = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            tick_number=5,
        )
        assert r2.emotion_deltas == {}

    def test_different_tick_allowed(self):
        """異なるティックでは帰還が許可される"""
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.6, valence=0.4, freshness=0.9)])
        ])
        r1 = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            tick_number=5,
        )
        r2 = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            tick_number=6,
        )
        # Both should produce results (though rumination may reduce second)
        assert r1.candidates_with_traces > 0
        assert r2.candidates_with_traces > 0


# =============================================================================
# Test: Safety Valve 6 - Existing Decay Cooperation
# =============================================================================

class TestDecayCooperation:
    def test_return_does_not_block_decay(self):
        """安全弁6: 帰還処理は減衰を阻害しない（構造的）

        本モジュールは感情ベクトルへの加算量のみを出力する。
        既存の指数減衰処理は反応処理側にあり、本モジュールはそれに一切関与しない。
        """
        proc = create_memory_emotion_return()
        # No "block decay" or "inhibit decay" methods exist
        assert not hasattr(proc, "block_decay")
        assert not hasattr(proc, "inhibit_decay")
        assert not hasattr(proc, "modify_decay_rate")


# =============================================================================
# Test: Safety Valve 7 - Enrichment Non-Exposure
# =============================================================================

class TestEnrichmentNonExposure:
    def test_no_enrichment_method(self):
        """安全弁7: enrichmentへの直接露出を行わない"""
        proc = create_memory_emotion_return()
        assert not hasattr(proc, "get_enrichment_data")
        assert not hasattr(proc, "get_enrichment")
        assert not hasattr(proc, "enrichment")

    def test_no_enrichment_in_result(self):
        """ReturnResultにenrichmentデータが含まれない"""
        result = ReturnResult(emotion_deltas={"joy": 0.02})
        d = result.to_dict()
        assert "enrichment" not in d


# =============================================================================
# Test: No Reverse Flow
# =============================================================================

class TestNoReverseFlow:
    def test_no_recall_modification_methods(self):
        """想起モジュールへの逆流経路を持たない"""
        proc = create_memory_emotion_return()
        assert not hasattr(proc, "modify_recall")
        assert not hasattr(proc, "update_recall")
        assert not hasattr(proc, "influence_recall")

    def test_no_forgetting_modification_methods(self):
        """忘却処理への経路を持たない"""
        proc = create_memory_emotion_return()
        assert not hasattr(proc, "modify_forgetting")
        assert not hasattr(proc, "update_forgetting")
        assert not hasattr(proc, "influence_forgetting")

    def test_no_decision_methods(self):
        """判断・行動・方針を確定する経路を持たない"""
        proc = create_memory_emotion_return()
        assert not hasattr(proc, "make_decision")
        assert not hasattr(proc, "select_policy")
        assert not hasattr(proc, "set_goal")
        assert not hasattr(proc, "evaluate")


# =============================================================================
# Test: FIFO Window
# =============================================================================

class TestFIFOWindow:
    def test_history_trimmed(self):
        """帰還事実履歴がFIFOウィンドウサイズを超えた場合にトリミングされる"""
        cfg = MemoryEmotionReturnConfig(history_window_size=5)
        proc = create_memory_emotion_return(config=cfg)
        store = _make_binding_store([
            (f"u{i}", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
            for i in range(10)
        ])

        for tick in range(10):
            proc.process(
                multi_path_candidates=_make_candidates([f"u{tick}"]),
                binding_store=store,
                current_emotions={"joy": 0.1},
                mood_arousal=0.5,
                tick_number=tick,
            )

        assert len(proc.state.return_history) <= 5

    def test_old_records_naturally_expire(self):
        """古い記録はFIFOウィンドウの移動で自然に消失する"""
        cfg = MemoryEmotionReturnConfig(history_window_size=3)
        proc = create_memory_emotion_return(config=cfg)
        store = _make_binding_store([
            ("u0", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)]),
            ("u1", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)]),
            ("u2", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)]),
            ("u3", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)]),
        ])

        # Process 4 ticks, window size 3
        for tick in range(4):
            proc.process(
                multi_path_candidates=_make_candidates([f"u{tick}"]),
                binding_store=store,
                current_emotions={"joy": 0.1},
                mood_arousal=0.5,
                tick_number=tick,
            )

        # u0 should have been naturally expired from history
        unit_ids = [r.unit_id for r in proc.state.return_history]
        assert "u0" not in unit_ids


# =============================================================================
# Test: Rumination Prevention Full Pipeline
# =============================================================================

class TestRuminationPrevention:
    def test_repeated_recall_diminishes(self):
        """同一記憶からの連続帰還は収束的に縮小する"""
        cfg = MemoryEmotionReturnConfig(
            rumination_threshold=2,
            rumination_decay_factor=0.5,
            history_window_size=100,
        )
        proc = create_memory_emotion_return(config=cfg)
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.6, valence=0.4, freshness=0.9)])
        ])

        deltas_over_time = []
        for tick in range(6):
            result = proc.process(
                multi_path_candidates=_make_candidates(["u1"]),
                binding_store=store,
                current_emotions={"joy": 0.1},
                mood_arousal=0.5,
                tick_number=tick,
            )
            joy_delta = result.emotion_deltas.get("joy", 0.0)
            deltas_over_time.append(joy_delta)

        # First calls should have positive deltas
        assert deltas_over_time[0] > 0

        # After threshold, deltas should decrease
        # Eventually they should converge toward zero
        if len(deltas_over_time) >= 4:
            assert deltas_over_time[3] < deltas_over_time[0]


# =============================================================================
# Test: State Persistence (save/load)
# =============================================================================

class TestStatePersistence:
    def test_state_setter(self):
        proc = create_memory_emotion_return()
        new_state = MemoryEmotionReturnState(cycle_count=10, last_applied_tick=5)
        proc.state = new_state
        assert proc.state.cycle_count == 10
        assert proc.state.last_applied_tick == 5

    def test_roundtrip(self):
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
        ])
        proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            mood_arousal=0.5,
            tick_number=1,
        )

        # Save state
        saved = proc.state.to_dict()

        # Create new processor and restore
        proc2 = create_memory_emotion_return()
        proc2.state = MemoryEmotionReturnState.from_dict(saved)

        assert proc2.state.cycle_count == proc.state.cycle_count
        assert proc2.state.last_applied_tick == proc.state.last_applied_tick
        assert len(proc2.state.return_history) == len(proc.state.return_history)


# =============================================================================
# Test: Summary
# =============================================================================

class TestSummary:
    def test_get_summary(self):
        proc = create_memory_emotion_return()
        summary = proc.get_summary()
        assert "cycle_count" in summary
        assert "history_length" in summary
        assert "last_applied_tick" in summary

    def test_summary_after_processing(self):
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
        ])
        proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.2},
            mood_arousal=0.5,
            tick_number=1,
        )
        summary = proc.get_summary()
        assert summary["cycle_count"] == 1
        assert summary["last_applied_tick"] == 1
        assert summary["history_length"] > 0


# =============================================================================
# Test: No Fixed Mapping
# =============================================================================

class TestNoFixedMapping:
    def test_same_memory_different_mood_different_result(self):
        """特定の記憶が特定の感情を「必ず」引き起こすマッピングではない"""
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
        ])

        proc1 = create_memory_emotion_return()
        r1 = proc1.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.0},
            mood_valence=0.8,
            mood_arousal=0.8,
            tick_number=1,
        )

        proc2 = create_memory_emotion_return()
        r2 = proc2.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.0},
            mood_valence=-0.8,
            mood_arousal=0.1,
            tick_number=1,
        )

        # Results should differ due to different mood states
        d1 = r1.emotion_deltas.get("joy", 0.0)
        d2 = r2.emotion_deltas.get("joy", 0.0)
        assert d1 != d2


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_emotions(self):
        proc = create_memory_emotion_return()
        store = _make_binding_store([
            ("u1", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
        ])
        result = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={},
            tick_number=1,
        )
        # Should still process (joy not in current_emotions defaults to 0.0)
        assert result.candidates_with_traces > 0

    def test_negative_tick_number(self):
        """Negative tick numbers should work (last_applied_tick starts at -1)"""
        proc = create_memory_emotion_return()
        result = proc.process(tick_number=-1)
        assert result.emotion_deltas == {}
        # -1 == -1 (default), so same-tick guard fires
        # But the first call should still be allowed since cycle_count starts at 0
        # Actually, default last_applied_tick is -1, and tick_number=-1 -> blocked
        # This is correct behavior: negative ticks are guarded

    def test_large_number_of_candidates(self):
        """Many candidates should be processed correctly with safety valves"""
        store_entries = [
            (f"u{i}", [MockTrace(emotion_label="joy", intensity=0.5, valence=0.3, freshness=0.8)])
            for i in range(50)
        ]
        store = _make_binding_store(store_entries)
        candidates = _make_candidates([f"u{i}" for i in range(50)])

        proc = create_memory_emotion_return()
        result = proc.process(
            multi_path_candidates=candidates,
            binding_store=store,
            current_emotions={"joy": 0.1},
            mood_arousal=0.5,
            max_bias_strength=0.15,
            tick_number=1,
        )
        # Total delta should be capped by max_bias_strength
        if "joy" in result.emotion_deltas:
            assert result.emotion_deltas["joy"] <= 0.15

    def test_clamp_helper(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.5, -1.0, 1.0) == 0.5

    def test_multiple_traces_same_emotion(self):
        """Multiple traces for the same emotion label should accumulate"""
        store = _make_binding_store([
            ("u1", [
                MockTrace(emotion_label="joy", intensity=0.3, valence=0.2, freshness=0.7),
                MockTrace(emotion_label="joy", intensity=0.4, valence=0.3, freshness=0.8),
            ])
        ])
        proc = create_memory_emotion_return()
        result = proc.process(
            multi_path_candidates=_make_candidates(["u1"]),
            binding_store=store,
            current_emotions={"joy": 0.0},
            mood_arousal=0.5,
            tick_number=1,
        )
        # Should have a single joy delta (accumulated from 2 traces)
        assert "joy" in result.emotion_deltas
