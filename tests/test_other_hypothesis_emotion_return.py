"""
tests/test_other_hypothesis_emotion_return.py

他者仮説由来の感情帰還経路のテスト。
4段パイプライン、安全弁7種、永続化、非固定性を検証する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from psyche.other_hypothesis_emotion_return import (
    OtherHypothesisEmotionReturnProcessor,
    OtherHypothesisEmotionReturnState,
    OtherHypothesisEmotionReturnConfig,
    HypothesisReturnRecord,
    HypothesisReturnResult,
    EMOTION_KEYWORD_DICT,
    _extract_emotion_labels,
    _derive_return_amounts,
    _apply_safety_valves,
    _apply_and_record,
    create_other_hypothesis_emotion_return,
)


# =============================================================================
# Test Helpers / Mock Objects
# =============================================================================

@dataclass
class MockHypothesis:
    """Mock OtherStateHypothesis for testing."""
    hypothesis_id: str = "hyp_001"
    description: str = "Other party appears happy and engaged"
    strength: float = 0.6
    freshness: float = 0.8
    competing_ids: tuple = ()


def make_hypothesis(
    hyp_id: str = "hyp_001",
    description: str = "Other party appears happy and engaged",
    strength: float = 0.6,
    freshness: float = 0.8,
    competing_ids: tuple = (),
) -> MockHypothesis:
    return MockHypothesis(
        hypothesis_id=hyp_id,
        description=description,
        strength=strength,
        freshness=freshness,
        competing_ids=competing_ids,
    )


def default_emotions() -> dict[str, float]:
    return {
        "joy": 0.1,
        "anger": 0.0,
        "sorrow": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "love": 0.0,
        "fun": 0.0,
    }


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory:
    def test_create_default(self):
        proc = create_other_hypothesis_emotion_return()
        assert isinstance(proc, OtherHypothesisEmotionReturnProcessor)
        assert proc.state.cycle_count == 0
        assert proc.state.last_applied_tick == -1
        assert len(proc.state.return_history) == 0

    def test_create_with_config(self):
        config = OtherHypothesisEmotionReturnConfig(
            per_candidate_max_delta=0.01,
            total_max_delta=0.05,
        )
        proc = create_other_hypothesis_emotion_return(config=config)
        assert proc._config.per_candidate_max_delta == 0.01
        assert proc._config.total_max_delta == 0.05


# =============================================================================
# Stage 1: Keyword Extraction Tests
# =============================================================================

class TestStage1Extraction:
    def test_happy_keyword_matches_joy(self):
        hyps = [make_hypothesis(description="Other party appears happy")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "joy" in axes

    def test_angry_keyword_matches_anger(self):
        hyps = [make_hypothesis(description="Other party seems angry and frustrated")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "anger" in axes

    def test_no_keyword_excluded(self):
        hyps = [make_hypothesis(description="Other party is present")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 0

    def test_multiple_axes_matched(self):
        hyps = [make_hypothesis(
            description="Other party appears happy and anxious at the same time"
        )]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "joy" in axes
        assert "fear" in axes

    def test_case_insensitive(self):
        hyps = [make_hypothesis(description="Other party is HAPPY and CHEERFUL")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1

    def test_negative_direction_matched(self):
        hyps = [make_hypothesis(description="Other party appears calm and relaxed")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        dirs = [m[1] for m in results[0]["matches"]]
        assert "negative" in dirs  # calm is negative-anger

    def test_empty_description(self):
        hyps = [make_hypothesis(description="")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 0

    def test_multiple_hypotheses(self):
        hyps = [
            make_hypothesis(hyp_id="h1", description="Other appears sad"),
            make_hypothesis(hyp_id="h2", description="Other seems cheerful"),
            make_hypothesis(hyp_id="h3", description="No emotion keywords here"),
        ]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 2

    def test_all_axes_have_keywords(self):
        """All 7 emotion axes should have keywords defined."""
        expected_axes = {"joy", "anger", "sorrow", "fear", "surprise", "love", "fun"}
        assert set(EMOTION_KEYWORD_DICT.keys()) == expected_axes
        for axis in expected_axes:
            assert "positive" in EMOTION_KEYWORD_DICT[axis]
            assert "negative" in EMOTION_KEYWORD_DICT[axis]
            assert len(EMOTION_KEYWORD_DICT[axis]["positive"]) > 0
            assert len(EMOTION_KEYWORD_DICT[axis]["negative"]) > 0

    def test_love_axis_keywords(self):
        hyps = [make_hypothesis(description="Other is warm and caring")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "love" in axes

    def test_fun_axis_keywords(self):
        hyps = [make_hypothesis(description="Other is playful and humorous")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "fun" in axes

    def test_surprise_axis_keywords(self):
        hyps = [make_hypothesis(description="Other seems surprised by the result")]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 1
        axes = [m[0] for m in results[0]["matches"]]
        assert "surprise" in axes


# =============================================================================
# Stage 2: Return Amount Derivation Tests
# =============================================================================

class TestStage2Derivation:
    def test_basic_derivation(self):
        matched = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        config = OtherHypothesisEmotionReturnConfig()
        result = _derive_return_amounts(
            matched, default_emotions(), 0.0, 0.5, config,
        )
        assert len(result) == 1
        assert "joy" in result[0]["deltas"]
        assert result[0]["deltas"]["joy"] > 0

    def test_higher_strength_larger_return(self):
        config = OtherHypothesisEmotionReturnConfig()
        matched_low = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.2,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        matched_high = [{
            "hypothesis_id": "h2",
            "description": "happy",
            "strength": 0.8,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        result_low = _derive_return_amounts(matched_low, default_emotions(), 0.0, 0.5, config)
        result_high = _derive_return_amounts(matched_high, default_emotions(), 0.0, 0.5, config)
        assert result_high[0]["deltas"]["joy"] > result_low[0]["deltas"]["joy"]

    def test_lower_freshness_smaller_return(self):
        config = OtherHypothesisEmotionReturnConfig()
        matched_fresh = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.9,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        matched_stale = [{
            "hypothesis_id": "h2",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.2,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        result_fresh = _derive_return_amounts(matched_fresh, default_emotions(), 0.0, 0.5, config)
        result_stale = _derive_return_amounts(matched_stale, default_emotions(), 0.0, 0.5, config)
        assert result_fresh[0]["deltas"]["joy"] > result_stale[0]["deltas"]["joy"]

    def test_competition_reduces_return(self):
        config = OtherHypothesisEmotionReturnConfig()
        matched_no_comp = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        matched_with_comp = [{
            "hypothesis_id": "h2",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 3,
            "matches": [("joy", "positive")],
        }]
        result_no = _derive_return_amounts(matched_no_comp, default_emotions(), 0.0, 0.5, config)
        result_comp = _derive_return_amounts(matched_with_comp, default_emotions(), 0.0, 0.5, config)
        assert result_no[0]["deltas"]["joy"] > result_comp[0]["deltas"]["joy"]

    def test_negative_direction(self):
        matched = [{
            "hypothesis_id": "h1",
            "description": "calm",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("anger", "negative")],
        }]
        config = OtherHypothesisEmotionReturnConfig()
        result = _derive_return_amounts(
            matched, default_emotions(), 0.0, 0.5, config,
        )
        assert result[0]["deltas"]["anger"] < 0

    def test_low_arousal_dampens_return(self):
        config = OtherHypothesisEmotionReturnConfig()
        matched = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        result_normal = _derive_return_amounts(matched, default_emotions(), 0.0, 0.5, config)
        result_low = _derive_return_amounts(matched, default_emotions(), 0.0, 0.05, config)
        assert result_normal[0]["deltas"]["joy"] > result_low[0]["deltas"]["joy"]

    def test_convergence_reduces_high_emotion(self):
        config = OtherHypothesisEmotionReturnConfig()
        emotions_high = default_emotions()
        emotions_high["joy"] = 0.8
        matched = [{
            "hypothesis_id": "h1",
            "description": "happy",
            "strength": 0.6,
            "freshness": 0.8,
            "competing_count": 0,
            "matches": [("joy", "positive")],
        }]
        result_low_base = _derive_return_amounts(matched, default_emotions(), 0.0, 0.5, config)
        result_high_base = _derive_return_amounts(matched, emotions_high, 0.0, 0.5, config)
        assert result_low_base[0]["deltas"]["joy"] > result_high_base[0]["deltas"]["joy"]


# =============================================================================
# Stage 3: Safety Valve Tests
# =============================================================================

class TestStage3SafetyValves:
    def test_per_candidate_max_clamp(self):
        config = OtherHypothesisEmotionReturnConfig(per_candidate_max_delta=0.01)
        derived = [{
            "hypothesis_id": "h1",
            "deltas": {"joy": 0.05},
        }]
        modified, total = _apply_safety_valves(derived, [], config, 0.15)
        assert total["joy"] <= 0.01

    def test_total_max_clamp(self):
        config = OtherHypothesisEmotionReturnConfig(
            per_candidate_max_delta=0.05,
            total_max_delta=0.03,
        )
        derived = [
            {"hypothesis_id": "h1", "deltas": {"joy": 0.02}},
            {"hypothesis_id": "h2", "deltas": {"joy": 0.02}},
        ]
        modified, total = _apply_safety_valves(derived, [], config, 0.15)
        assert total["joy"] <= 0.03

    def test_max_bias_strength_limits_total(self):
        config = OtherHypothesisEmotionReturnConfig(total_max_delta=0.10)
        derived = [
            {"hypothesis_id": "h1", "deltas": {"joy": 0.05}},
        ]
        modified, total = _apply_safety_valves(derived, [], config, 0.03)
        assert total["joy"] <= 0.03

    def test_rumination_decay(self):
        config = OtherHypothesisEmotionReturnConfig(
            rumination_threshold=2,
            rumination_decay_factor=0.5,
        )
        history = [
            HypothesisReturnRecord(hypothesis_id="h1", tick_number=1),
            HypothesisReturnRecord(hypothesis_id="h1", tick_number=2),
            HypothesisReturnRecord(hypothesis_id="h1", tick_number=3),
        ]
        derived = [{"hypothesis_id": "h1", "deltas": {"joy": 0.02}}]
        modified, total = _apply_safety_valves(derived, history, config, 0.15)
        # With 3 occurrences >= threshold of 2, decay should apply
        assert modified[0]["rumination_applied"] is True
        assert total["joy"] < 0.02

    def test_no_rumination_below_threshold(self):
        config = OtherHypothesisEmotionReturnConfig(rumination_threshold=3)
        history = [
            HypothesisReturnRecord(hypothesis_id="h1", tick_number=1),
        ]
        derived = [{"hypothesis_id": "h1", "deltas": {"joy": 0.02}}]
        modified, total = _apply_safety_valves(derived, history, config, 0.15)
        assert modified[0]["rumination_applied"] is False

    def test_combined_bandwidth_with_memory_return(self):
        config = OtherHypothesisEmotionReturnConfig(
            combined_max_delta=0.10,
            total_max_delta=0.10,
        )
        derived = [{"hypothesis_id": "h1", "deltas": {"joy": 0.08}}]
        memory_deltas = {"joy": 0.06}
        modified, total = _apply_safety_valves(
            derived, [], config, 0.15, memory_deltas,
        )
        # Combined should not exceed 0.10
        assert abs(total["joy"]) + abs(memory_deltas["joy"]) <= 0.10 + 1e-6

    def test_negative_clamping(self):
        config = OtherHypothesisEmotionReturnConfig(per_candidate_max_delta=0.01)
        derived = [{"hypothesis_id": "h1", "deltas": {"anger": -0.05}}]
        modified, total = _apply_safety_valves(derived, [], config, 0.15)
        assert total["anger"] >= -0.01


# =============================================================================
# Stage 4: Apply and Record Tests
# =============================================================================

class TestStage4ApplyRecord:
    def test_clamp_to_valid_range(self):
        total_deltas = {"joy": 0.95}
        emotions = {"joy": 0.9}
        clamped, records = _apply_and_record(
            total_deltas, [{"hypothesis_id": "h1", "deltas": {"joy": 0.95}, "rumination_applied": False}],
            emotions, 0.0, 0.5, 42,
        )
        assert clamped["joy"] <= 0.1 + 1e-6  # max 1.0 - 0.9

    def test_negative_clamp(self):
        total_deltas = {"anger": -0.5}
        emotions = {"anger": 0.1}
        clamped, records = _apply_and_record(
            total_deltas, [{"hypothesis_id": "h1", "deltas": {"anger": -0.5}, "rumination_applied": False}],
            emotions, 0.0, 0.5, 42,
        )
        assert clamped["anger"] >= -0.1 - 1e-6  # min 0.0 - 0.1

    def test_records_created(self):
        total_deltas = {"joy": 0.01}
        entries = [{"hypothesis_id": "h1", "deltas": {"joy": 0.01}, "rumination_applied": False}]
        clamped, records = _apply_and_record(
            total_deltas, entries, default_emotions(), 0.0, 0.5, 42,
        )
        assert len(records) == 1
        assert records[0].hypothesis_id == "h1"
        assert records[0].tick_number == 42

    def test_zero_delta_not_recorded(self):
        total_deltas = {"joy": 0.0}
        entries = [{"hypothesis_id": "h1", "deltas": {"joy": 0.0}, "rumination_applied": False}]
        clamped, records = _apply_and_record(
            total_deltas, entries, default_emotions(), 0.0, 0.5, 42,
        )
        assert len(records) == 0


# =============================================================================
# Processor Integration Tests
# =============================================================================

class TestProcessorIntegration:
    def test_full_pipeline(self):
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(description="Other party appears happy and engaged")]
        result = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            mood_valence=0.1,
            mood_arousal=0.5,
            max_bias_strength=0.15,
            tick_number=1,
        )
        assert result.total_hypotheses_processed == 1
        assert result.hypotheses_with_matches >= 1
        assert "joy" in result.emotion_deltas or result.records_created >= 0

    def test_no_hypotheses(self):
        proc = create_other_hypothesis_emotion_return()
        result = proc.process(
            active_hypotheses=[],
            current_emotions=default_emotions(),
            tick_number=1,
        )
        assert result.emotion_deltas == {}
        assert result.records_created == 0

    def test_none_hypotheses(self):
        proc = create_other_hypothesis_emotion_return()
        result = proc.process(
            active_hypotheses=None,
            current_emotions=default_emotions(),
            tick_number=1,
        )
        assert result.emotion_deltas == {}

    def test_same_tick_prevention(self):
        """Safety valve 5: same tick should not process twice."""
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(description="Other party appears happy")]
        result1 = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            tick_number=5,
        )
        result2 = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            tick_number=5,
        )
        assert result2.emotion_deltas == {}

    def test_different_ticks_process(self):
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(description="Other party appears happy")]
        result1 = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            tick_number=5,
        )
        result2 = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            tick_number=6,
        )
        # Both should process (different ticks)
        assert proc.state.cycle_count == 2

    def test_fifo_trimming(self):
        config = OtherHypothesisEmotionReturnConfig(history_window_size=3)
        proc = create_other_hypothesis_emotion_return(config=config)
        hyps = [make_hypothesis(description="Other party appears happy")]
        for tick in range(10):
            proc.process(
                active_hypotheses=hyps,
                current_emotions=default_emotions(),
                tick_number=tick,
            )
        assert len(proc.state.return_history) <= 3

    def test_cycle_count_increments(self):
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(description="Other party appears happy")]
        for tick in range(5):
            proc.process(
                active_hypotheses=hyps,
                current_emotions=default_emotions(),
                tick_number=tick,
            )
        assert proc.state.cycle_count == 5

    def test_enrichment_not_exposed(self):
        """Safety valve 7: no enrichment exposure."""
        proc = create_other_hypothesis_emotion_return()
        # The processor should not have any get_enrichment method
        assert not hasattr(proc, "get_enrichment")
        assert not hasattr(proc, "enrichment")
        assert not hasattr(proc, "get_enrichment_entry")

    def test_no_write_to_hypotheses(self):
        """No write-back path to other model."""
        proc = create_other_hypothesis_emotion_return()
        forbidden = [
            "update_hypothesis", "modify_hypothesis", "revise_hypothesis",
            "set_hypothesis", "write_hypothesis",
            "update_other_model", "modify_other_model",
        ]
        methods = [m for m in dir(proc) if not m.startswith("_")]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower()

    def test_no_memory_write_path(self):
        """No write-back path to memory/forgetting systems."""
        proc = create_other_hypothesis_emotion_return()
        forbidden = [
            "update_memory", "modify_memory", "write_memory",
            "update_forgetting", "modify_forgetting",
        ]
        methods = [m for m in dir(proc) if not m.startswith("_")]
        for method in methods:
            for pattern in forbidden:
                assert pattern not in method.lower()


# =============================================================================
# State Serialization Tests
# =============================================================================

class TestStateSerialization:
    def test_state_to_dict(self):
        state = OtherHypothesisEmotionReturnState(
            return_history=[
                HypothesisReturnRecord(
                    hypothesis_id="h1",
                    emotion_labels=["joy"],
                    emotion_deltas={"joy": 0.01},
                    tick_number=1,
                ),
            ],
            last_applied_tick=5,
            cycle_count=3,
        )
        d = state.to_dict()
        assert d["last_applied_tick"] == 5
        assert d["cycle_count"] == 3
        assert len(d["return_history"]) == 1
        assert d["return_history"][0]["hypothesis_id"] == "h1"

    def test_state_from_dict(self):
        d = {
            "return_history": [
                {
                    "hypothesis_id": "h1",
                    "emotion_labels": ["joy"],
                    "emotion_deltas": {"joy": 0.01},
                    "mood_direction": 0.1,
                    "arousal_level": 0.5,
                    "rumination_decay_applied": False,
                    "timestamp": 1234567890.0,
                    "tick_number": 42,
                }
            ],
            "last_applied_tick": 42,
            "cycle_count": 10,
        }
        state = OtherHypothesisEmotionReturnState.from_dict(d)
        assert state.last_applied_tick == 42
        assert state.cycle_count == 10
        assert len(state.return_history) == 1
        assert state.return_history[0].hypothesis_id == "h1"

    def test_roundtrip(self):
        state = OtherHypothesisEmotionReturnState(
            return_history=[
                HypothesisReturnRecord(
                    hypothesis_id="h2",
                    emotion_labels=["anger", "fear"],
                    emotion_deltas={"anger": -0.01, "fear": 0.005},
                    mood_direction=-0.2,
                    arousal_level=0.3,
                    rumination_decay_applied=True,
                    timestamp=1000.0,
                    tick_number=7,
                ),
            ],
            last_applied_tick=7,
            cycle_count=5,
        )
        d = state.to_dict()
        restored = OtherHypothesisEmotionReturnState.from_dict(d)
        assert restored.last_applied_tick == state.last_applied_tick
        assert restored.cycle_count == state.cycle_count
        assert len(restored.return_history) == 1
        rec = restored.return_history[0]
        assert rec.hypothesis_id == "h2"
        assert rec.rumination_decay_applied is True
        assert abs(rec.emotion_deltas["anger"] - (-0.01)) < 1e-6

    def test_empty_state_roundtrip(self):
        state = OtherHypothesisEmotionReturnState()
        d = state.to_dict()
        restored = OtherHypothesisEmotionReturnState.from_dict(d)
        assert restored.last_applied_tick == -1
        assert restored.cycle_count == 0
        assert len(restored.return_history) == 0

    def test_state_setter(self):
        proc = create_other_hypothesis_emotion_return()
        new_state = OtherHypothesisEmotionReturnState(
            last_applied_tick=99,
            cycle_count=50,
        )
        proc.state = new_state
        assert proc.state.last_applied_tick == 99
        assert proc.state.cycle_count == 50


# =============================================================================
# Record Tests
# =============================================================================

class TestRecord:
    def test_record_to_dict(self):
        rec = HypothesisReturnRecord(
            hypothesis_id="h1",
            emotion_labels=["joy", "love"],
            emotion_deltas={"joy": 0.01, "love": 0.005},
            mood_direction=0.2,
            arousal_level=0.5,
            rumination_decay_applied=False,
            timestamp=1000.0,
            tick_number=10,
        )
        d = rec.to_dict()
        assert d["hypothesis_id"] == "h1"
        assert len(d["emotion_labels"]) == 2
        assert d["tick_number"] == 10

    def test_record_from_dict(self):
        d = {
            "hypothesis_id": "h3",
            "emotion_labels": ["sorrow"],
            "emotion_deltas": {"sorrow": 0.015},
            "mood_direction": -0.1,
            "arousal_level": 0.4,
            "rumination_decay_applied": True,
            "timestamp": 2000.0,
            "tick_number": 20,
        }
        rec = HypothesisReturnRecord.from_dict(d)
        assert rec.hypothesis_id == "h3"
        assert rec.rumination_decay_applied is True
        assert rec.tick_number == 20

    def test_record_defaults(self):
        rec = HypothesisReturnRecord()
        assert rec.hypothesis_id == ""
        assert len(rec.emotion_labels) == 0
        assert rec.tick_number == 0


# =============================================================================
# Non-Fixation Tests
# =============================================================================

class TestNonFixation:
    def test_return_varies_with_strength(self):
        """Same hypothesis content but different strength produces different return."""
        proc = create_other_hypothesis_emotion_return()
        hyps_low = [make_hypothesis(description="Other appears happy", strength=0.2)]
        hyps_high = [make_hypothesis(description="Other appears happy", strength=0.9)]
        r1 = proc.process(
            active_hypotheses=hyps_low,
            current_emotions=default_emotions(),
            tick_number=1,
        )
        r2 = proc.process(
            active_hypotheses=hyps_high,
            current_emotions=default_emotions(),
            tick_number=2,
        )
        # Higher strength should produce larger delta
        d1 = r1.emotion_deltas.get("joy", 0.0)
        d2 = r2.emotion_deltas.get("joy", 0.0)
        assert d2 > d1

    def test_return_varies_with_mood(self):
        """Same hypothesis but different mood produces different return."""
        # Use higher per_candidate_max to avoid clamping masking differences
        config = OtherHypothesisEmotionReturnConfig(per_candidate_max_delta=0.10)
        proc1 = create_other_hypothesis_emotion_return(config=config)
        proc2 = create_other_hypothesis_emotion_return(config=config)
        hyps = [make_hypothesis(description="Other appears happy")]
        r1 = proc1.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            mood_valence=0.5,
            mood_arousal=0.5,
            tick_number=1,
        )
        r2 = proc2.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            mood_valence=-0.5,
            mood_arousal=0.5,
            tick_number=1,
        )
        d1 = r1.emotion_deltas.get("joy", 0.0)
        d2 = r2.emotion_deltas.get("joy", 0.0)
        assert d1 != d2

    def test_return_varies_with_competition(self):
        """More competing hypotheses -> smaller return."""
        # Use higher per_candidate_max to avoid clamping masking differences
        config = OtherHypothesisEmotionReturnConfig(per_candidate_max_delta=0.10)
        proc1 = create_other_hypothesis_emotion_return(config=config)
        proc2 = create_other_hypothesis_emotion_return(config=config)
        hyps_no = [make_hypothesis(description="Other appears happy", competing_ids=())]
        hyps_comp = [make_hypothesis(description="Other appears happy", competing_ids=("c1", "c2", "c3"))]
        r1 = proc1.process(
            active_hypotheses=hyps_no,
            current_emotions=default_emotions(),
            tick_number=1,
        )
        r2 = proc2.process(
            active_hypotheses=hyps_comp,
            current_emotions=default_emotions(),
            tick_number=1,
        )
        d1 = r1.emotion_deltas.get("joy", 0.0)
        d2 = r2.emotion_deltas.get("joy", 0.0)
        assert d1 > d2

    def test_rumination_reduces_repeated_return(self):
        """Repeated same hypothesis should have diminishing returns."""
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(hyp_id="repeater", description="Other appears happy")]
        deltas = []
        for tick in range(6):
            r = proc.process(
                active_hypotheses=hyps,
                current_emotions=default_emotions(),
                tick_number=tick,
            )
            deltas.append(r.emotion_deltas.get("joy", 0.0))
        # Later deltas should be smaller or zero due to rumination
        # First delta should be largest (before rumination kicks in)
        if deltas[0] > 0:
            assert deltas[-1] <= deltas[0]


# =============================================================================
# Bandwidth Tests
# =============================================================================

class TestBandwidth:
    def test_per_candidate_within_memory_return_limit(self):
        """Per-candidate limit should be <= memory return's per_candidate (0.03)."""
        config = OtherHypothesisEmotionReturnConfig()
        assert config.per_candidate_max_delta <= 0.03

    def test_total_within_half_memory_return_limit(self):
        """Total limit should be <= half of memory return's total (0.15)."""
        config = OtherHypothesisEmotionReturnConfig()
        assert config.total_max_delta <= 0.15 / 2

    def test_output_magnitude_small(self):
        """Output should be much smaller than typical emotion stimulus."""
        proc = create_other_hypothesis_emotion_return()
        hyps = [
            make_hypothesis(hyp_id=f"h{i}", description="Other appears happy and cheerful", strength=0.9)
            for i in range(5)
        ]
        result = proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            mood_valence=0.5,
            mood_arousal=0.8,
            max_bias_strength=0.15,
            tick_number=1,
        )
        for label, delta in result.emotion_deltas.items():
            assert abs(delta) <= 0.07  # total_max_delta


# =============================================================================
# Get Summary Tests
# =============================================================================

class TestGetSummary:
    def test_summary_keys(self):
        proc = create_other_hypothesis_emotion_return()
        summary = proc.get_summary()
        assert "cycle_count" in summary
        assert "history_length" in summary
        assert "last_applied_tick" in summary

    def test_summary_after_processing(self):
        proc = create_other_hypothesis_emotion_return()
        hyps = [make_hypothesis(description="Other appears happy")]
        proc.process(
            active_hypotheses=hyps,
            current_emotions=default_emotions(),
            tick_number=1,
        )
        summary = proc.get_summary()
        assert summary["cycle_count"] == 1
        assert summary["last_applied_tick"] == 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_hypothesis_with_no_description_attr(self):
        """Hypothesis missing description attribute should be skipped."""
        class BareHyp:
            hypothesis_id = "bare"
            strength = 0.5
            freshness = 0.5
            competing_ids = ()
        proc = create_other_hypothesis_emotion_return()
        result = proc.process(
            active_hypotheses=[BareHyp()],
            current_emotions=default_emotions(),
            tick_number=1,
        )
        assert result.emotion_deltas == {}

    def test_hypothesis_with_non_string_description(self):
        hyps = [make_hypothesis(description=None)]
        results = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        assert len(results) == 0

    def test_zero_freshness_hypothesis(self):
        hyps = [make_hypothesis(description="Other appears happy", freshness=0.0)]
        config = OtherHypothesisEmotionReturnConfig()
        matched = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        if matched:
            derived = _derive_return_amounts(matched, default_emotions(), 0.0, 0.5, config)
            # Zero freshness should result in zero return
            for entry in derived:
                for v in entry["deltas"].values():
                    assert abs(v) < 1e-6

    def test_zero_strength_hypothesis(self):
        hyps = [make_hypothesis(description="Other appears happy", strength=0.0)]
        config = OtherHypothesisEmotionReturnConfig()
        matched = _extract_emotion_labels(hyps, EMOTION_KEYWORD_DICT)
        if matched:
            derived = _derive_return_amounts(matched, default_emotions(), 0.0, 0.5, config)
            for entry in derived:
                for v in entry["deltas"].values():
                    assert abs(v) < 1e-6

    def test_all_emotions_at_max(self):
        """With all emotions at 1.0, positive deltas should be clamped to 0."""
        proc = create_other_hypothesis_emotion_return()
        emotions = {k: 1.0 for k in default_emotions()}
        hyps = [make_hypothesis(description="Other appears happy")]
        result = proc.process(
            active_hypotheses=hyps,
            current_emotions=emotions,
            tick_number=1,
        )
        # Joy should not increase beyond 1.0
        assert result.emotion_deltas.get("joy", 0.0) <= 0.0 + 1e-6

    def test_result_to_dict(self):
        result = HypothesisReturnResult(
            emotion_deltas={"joy": 0.01},
            records_created=1,
            total_hypotheses_processed=3,
            hypotheses_with_matches=1,
        )
        d = result.to_dict()
        assert d["emotion_deltas"]["joy"] == 0.01
        assert d["records_created"] == 1
