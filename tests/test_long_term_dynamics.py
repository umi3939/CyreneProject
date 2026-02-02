"""
Tests for psyche/long_term_dynamics.py - Long-Term Dynamics Logging (長期挙動ログ)

Verifies:
- PASSIVE observation only (no behavior changes)
- Aggregated stats (averages, variance, counts)
- Window-based collection
- Lightweight (no raw log storage)
- JSON file persistence
"""

import pytest
import tempfile
import os
import json
import time

from psyche.long_term_dynamics import (
    EmotionStats,
    DecisionStats,
    ValueOrientationStats,
    ResponsibilityStats,
    StabilityValveStats,
    WindowStats,
    LongTermEntry,
    DynamicsObserverConfig,
    DynamicsObserver,
    create_observer,
    create_config,
    get_observer_summary,
    entries_to_json,
    entries_from_json,
    compute_mean,
    compute_variance,
    compute_std,
)

from psyche.state import EmotionVector
from psyche.value_orientation import ValueOrientation


# ── Statistics Helpers Tests ────────────────────────────────────────


class TestStatisticsHelpers:
    """Tests for statistics helper functions."""

    def test_compute_mean(self):
        """Mean computed correctly."""
        assert compute_mean([1, 2, 3, 4, 5]) == 3.0
        assert compute_mean([]) == 0.0

    def test_compute_variance(self):
        """Variance computed correctly."""
        # [1, 3] has mean 2, variance = ((1-2)^2 + (3-2)^2) / 2 = 1
        assert compute_variance([1, 3]) == 1.0
        assert compute_variance([5]) == 0.0

    def test_compute_std(self):
        """Standard deviation computed correctly."""
        assert compute_std([1, 3]) == 1.0


# ── Stats Dataclass Tests ───────────────────────────────────────────


class TestEmotionStats:
    """Tests for EmotionStats."""

    def test_default_values(self):
        """Default stats are zero."""
        stats = EmotionStats()
        assert stats.joy_mean == 0.0
        assert stats.fear_std == 0.0
        assert stats.dominant_emotion == ""

    def test_serialization(self):
        """Stats can be serialized and deserialized."""
        original = EmotionStats(
            joy_mean=0.5,
            joy_std=0.1,
            dominant_emotion="joy",
            peak_count=5,
        )
        data = original.to_dict()
        restored = EmotionStats.from_dict(data)

        assert restored.joy_mean == original.joy_mean
        assert restored.dominant_emotion == original.dominant_emotion
        assert restored.peak_count == original.peak_count


class TestDecisionStats:
    """Tests for DecisionStats."""

    def test_serialization(self):
        """Stats can be serialized and deserialized."""
        original = DecisionStats(
            total_decisions=10,
            silence_count=3,
            silence_rate=0.3,
            policy_counts={"共感する": 5, "沈黙する": 3},
            most_common_policy="共感する",
        )
        data = original.to_dict()
        restored = DecisionStats.from_dict(data)

        assert restored.total_decisions == 10
        assert restored.silence_rate == 0.3
        assert restored.most_common_policy == "共感する"


class TestValueOrientationStats:
    """Tests for ValueOrientationStats."""

    def test_serialization(self):
        """Stats can be serialized and deserialized."""
        original = ValueOrientationStats(
            dim_a_mean=0.3,
            dim_a_delta=0.1,
            most_changed_dim="a",
            overall_stability=0.8,
        )
        data = original.to_dict()
        restored = ValueOrientationStats.from_dict(data)

        assert restored.dim_a_mean == original.dim_a_mean
        assert restored.dim_a_delta == original.dim_a_delta
        assert restored.most_changed_dim == "a"


class TestWindowStats:
    """Tests for WindowStats."""

    def test_contains_all_stat_types(self):
        """WindowStats contains all stat types."""
        stats = WindowStats()
        assert isinstance(stats.emotion, EmotionStats)
        assert isinstance(stats.decision, DecisionStats)
        assert isinstance(stats.value_orientation, ValueOrientationStats)
        assert isinstance(stats.responsibility, ResponsibilityStats)
        assert isinstance(stats.stability_valve, StabilityValveStats)

    def test_serialization(self):
        """WindowStats can be serialized and deserialized."""
        original = WindowStats(
            emotion=EmotionStats(joy_mean=0.5),
            decision=DecisionStats(total_decisions=10),
        )
        data = original.to_dict()
        restored = WindowStats.from_dict(data)

        assert restored.emotion.joy_mean == 0.5
        assert restored.decision.total_decisions == 10


# ── LongTermEntry Tests ─────────────────────────────────────────────


class TestLongTermEntry:
    """Tests for LongTermEntry."""

    def test_creates_with_metadata(self):
        """Entry has metadata fields."""
        entry = LongTermEntry(
            entry_id=1,
            window_start_turn=1,
            window_end_turn=10,
            window_size=10,
        )

        assert entry.entry_id == 1
        assert entry.window_start_turn == 1
        assert entry.window_end_turn == 10

    def test_serialization(self):
        """Entry can be serialized and deserialized."""
        original = LongTermEntry(
            entry_id=5,
            window_start_turn=41,
            window_end_turn=50,
            window_size=10,
            stats=WindowStats(
                emotion=EmotionStats(joy_mean=0.6)
            ),
            has_delta=True,
            delta_summary={"emotion_intensity_delta": 0.1},
        )

        data = original.to_dict()
        restored = LongTermEntry.from_dict(data)

        assert restored.entry_id == 5
        assert restored.window_start_turn == 41
        assert restored.stats.emotion.joy_mean == 0.6
        assert restored.has_delta is True
        assert "emotion_intensity_delta" in restored.delta_summary


# ── DynamicsObserver Tests ──────────────────────────────────────────


class TestDynamicsObserver:
    """Tests for DynamicsObserver."""

    def test_creates_with_default_config(self):
        """Observer creates with default config."""
        observer = DynamicsObserver()
        assert observer.config is not None
        assert observer.config.window_size == 10

    def test_creates_with_custom_config(self):
        """Observer creates with custom config."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)
        assert observer.config.window_size == 5

    def test_records_turns(self):
        """Observer records turns."""
        observer = DynamicsObserver()

        for i in range(5):
            observer.record_turn(decision_label=f"decision_{i}")

        assert observer.get_total_turns() == 5

    def test_completes_window(self):
        """Observer completes window after window_size turns."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)

        # Record 5 turns
        for i in range(4):
            result = observer.record_turn(decision_label="test")
            assert result is None  # Not complete yet

        # 5th turn completes window
        result = observer.record_turn(decision_label="test")
        assert result is not None
        assert isinstance(result, LongTermEntry)

    def test_entry_has_correct_turn_range(self):
        """Completed entry has correct turn range."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)

        for _ in range(5):
            result = observer.record_turn(decision_label="test")

        assert result.window_start_turn == 1
        assert result.window_end_turn == 5
        assert result.window_size == 5


class TestPassiveObservation:
    """Tests verifying PASSIVE observation (no behavior changes)."""

    def test_does_not_modify_emotion_state(self):
        """Observer does not modify emotion state."""
        observer = DynamicsObserver()
        emotion = EmotionVector(joy=0.5, fear=0.3)

        original_joy = emotion.joy
        original_fear = emotion.fear

        observer.record_turn(emotion_state=emotion)

        assert emotion.joy == original_joy
        assert emotion.fear == original_fear

    def test_does_not_modify_value_orientation(self):
        """Observer does not modify value orientation."""
        observer = DynamicsObserver()
        orientation = ValueOrientation(dim_a=0.5)

        original_a = orientation.dim_a

        observer.record_turn(value_orientation=orientation)

        assert orientation.dim_a == original_a

    def test_record_returns_none_or_entry_only(self):
        """record_turn only returns None or LongTermEntry."""
        observer = DynamicsObserver()

        result = observer.record_turn(decision_label="test")

        assert result is None or isinstance(result, LongTermEntry)


class TestAggregation:
    """Tests for data aggregation."""

    def test_aggregates_emotion_stats(self):
        """Aggregates emotion statistics correctly."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        emotions = [
            EmotionVector(joy=0.2),
            EmotionVector(joy=0.4),
            EmotionVector(joy=0.6),
        ]

        for emotion in emotions:
            result = observer.record_turn(emotion_state=emotion)

        # Mean should be 0.4
        assert result is not None
        assert abs(result.stats.emotion.joy_mean - 0.4) < 0.01

    def test_aggregates_decision_counts(self):
        """Aggregates decision counts correctly."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)

        decisions = ["共感する", "沈黙する", "共感する", "からかう", "共感する"]

        for d in decisions:
            result = observer.record_turn(
                decision_label=d,
                is_silence=(d == "沈黙する"),
            )

        assert result is not None
        assert result.stats.decision.total_decisions == 5
        assert result.stats.decision.silence_count == 1
        assert result.stats.decision.silence_rate == 0.2
        assert result.stats.decision.most_common_policy == "共感する"

    def test_aggregates_value_orientation_delta(self):
        """Tracks value orientation delta within window."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        # Values increase over window
        orientations = [
            ValueOrientation(dim_a=0.1),
            ValueOrientation(dim_a=0.2),
            ValueOrientation(dim_a=0.4),
        ]

        for o in orientations:
            result = observer.record_turn(value_orientation=o)

        assert result is not None
        # Delta should be 0.4 - 0.1 = 0.3
        assert abs(result.stats.value_orientation.dim_a_delta - 0.3) < 0.01

    def test_does_not_store_raw_samples(self):
        """Observer does not store raw samples in entries."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)

        for _ in range(5):
            observer.record_turn(
                emotion_state=EmotionVector(joy=0.5),
                decision_label="test",
            )

        entry = observer.get_latest_entry()
        data = entry.to_dict()

        # Should not contain raw sample arrays
        assert "joy_samples" not in str(data)
        assert "decision_labels" not in str(data)


class TestDeltaComputation:
    """Tests for delta computation between windows."""

    def test_first_entry_has_no_delta(self):
        """First entry has no delta (no previous window)."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        for _ in range(3):
            result = observer.record_turn(decision_label="test")

        assert result.has_delta is False
        assert result.delta_summary == {}

    def test_second_entry_has_delta(self):
        """Second entry has delta from first."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        # First window
        for _ in range(3):
            observer.record_turn(decision_label="共感する", is_silence=False)

        # Second window with different data
        for _ in range(3):
            result = observer.record_turn(decision_label="沈黙する", is_silence=True)

        assert result.has_delta is True
        assert "silence_rate_delta" in result.delta_summary
        # Silence rate went from 0 to 1
        assert result.delta_summary["silence_rate_delta"] > 0


# ── Persistence Tests ───────────────────────────────────────────────


class TestPersistence:
    """Tests for file persistence."""

    def test_save_to_file(self):
        """Can save entries to JSON file."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        for _ in range(6):  # 2 windows
            observer.record_turn(decision_label="test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            observer.save_to_file(file_path)

            # Verify file exists and contains valid JSON
            with open(file_path, "r") as f:
                data = json.load(f)

            assert data["entry_count"] == 2
            assert len(data["entries"]) == 2
        finally:
            os.unlink(file_path)

    def test_load_from_file(self):
        """Can load entries from JSON file."""
        config = DynamicsObserverConfig(window_size=3)
        observer1 = DynamicsObserver(config)

        for _ in range(6):  # 2 windows
            observer1.record_turn(decision_label="test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            observer1.save_to_file(file_path)

            # Load into new observer
            observer2 = DynamicsObserver(config)
            loaded = observer2.load_from_file(file_path)

            assert loaded == 2
            assert observer2.get_entry_count() == 2
            assert observer2.get_total_turns() == 6
        finally:
            os.unlink(file_path)

    def test_continuity_after_reload(self):
        """Can continue observation after reload."""
        config = DynamicsObserverConfig(window_size=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            # First session
            observer1 = DynamicsObserver(config)
            for _ in range(3):
                observer1.record_turn(decision_label="test")
            observer1.save_to_file(file_path)

            # Second session
            observer2 = DynamicsObserver(config)
            observer2.load_from_file(file_path)

            # Continue observing
            for _ in range(3):
                result = observer2.record_turn(decision_label="test2")

            # Should have 2 entries now
            assert observer2.get_entry_count() == 2
        finally:
            os.unlink(file_path)


# ── Enable/Disable Tests ────────────────────────────────────────────


class TestEnableDisable:
    """Tests for enabling/disabling observation."""

    def test_disabled_observer_does_not_record(self):
        """Disabled observer does not record turns."""
        config = DynamicsObserverConfig(enabled=False)
        observer = DynamicsObserver(config)

        for _ in range(10):
            observer.record_turn(decision_label="test")

        assert observer.get_total_turns() == 0

    def test_can_toggle_enabled(self):
        """Can toggle enabled state."""
        observer = DynamicsObserver()

        observer.record_turn(decision_label="test")
        assert observer.get_total_turns() == 1

        observer.set_enabled(False)
        observer.record_turn(decision_label="test")
        assert observer.get_total_turns() == 1  # Did not increase

        observer.set_enabled(True)
        observer.record_turn(decision_label="test")
        assert observer.get_total_turns() == 2


# ── Utility Function Tests ──────────────────────────────────────────


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_observer(self):
        """create_observer creates with custom settings."""
        observer = create_observer(window_size=20, enabled=True)
        assert observer.config.window_size == 20
        assert observer.is_enabled()

    def test_create_config(self):
        """create_config creates config."""
        config = create_config(window_size=15, max_entries=50)
        assert config.window_size == 15
        assert config.max_entries_in_memory == 50

    def test_get_observer_summary(self):
        """get_observer_summary returns readable summary."""
        config = DynamicsObserverConfig(window_size=3)
        observer = DynamicsObserver(config)

        for _ in range(3):
            observer.record_turn(
                emotion_state=EmotionVector(joy=0.5),
                decision_label="test",
            )

        summary = get_observer_summary(observer)

        assert "DynamicsObserver" in summary
        assert "1 entries" in summary
        assert "3 turns" in summary

    def test_entries_to_json(self):
        """entries_to_json converts entries to JSON format."""
        entries = [
            LongTermEntry(entry_id=1),
            LongTermEntry(entry_id=2),
        ]

        data = entries_to_json(entries)

        assert len(data) == 2
        assert data[0]["entry_id"] == 1

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_entries_from_json(self):
        """entries_from_json restores entries from JSON format."""
        data = [
            {"entry_id": 1, "window_start_turn": 1, "window_end_turn": 10},
            {"entry_id": 2, "window_start_turn": 11, "window_end_turn": 20},
        ]

        entries = entries_from_json(data)

        assert len(entries) == 2
        assert entries[0].entry_id == 1
        assert entries[1].window_start_turn == 11


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    """Integration tests."""

    def test_full_observation_workflow(self):
        """Complete workflow from observation to analysis."""
        config = DynamicsObserverConfig(window_size=5)
        observer = DynamicsObserver(config)

        # Simulate 15 turns (3 windows)
        for i in range(15):
            emotion = EmotionVector(joy=0.3 + i * 0.02)
            orientation = ValueOrientation(dim_a=0.1 * (i % 5))
            is_silence = (i % 4 == 0)

            result = observer.record_turn(
                emotion_state=emotion,
                decision_label="沈黙する" if is_silence else "共感する",
                is_silence=is_silence,
                value_orientation=orientation,
            )

        # Should have 3 entries
        assert observer.get_entry_count() == 3
        assert observer.get_total_turns() == 15

        # Get entries and verify
        entries = observer.get_entries()
        assert len(entries) == 3

        # Later entries should have delta
        assert entries[0].has_delta is False
        assert entries[1].has_delta is True
        assert entries[2].has_delta is True

    def test_lightweight_storage(self):
        """Verifies storage is lightweight (aggregated, not raw)."""
        config = DynamicsObserverConfig(window_size=100)
        observer = DynamicsObserver(config)

        # Record 100 turns with data
        for i in range(100):
            observer.record_turn(
                emotion_state=EmotionVector(joy=i / 100),
                decision_label=f"decision_{i % 10}",
                value_orientation=ValueOrientation(dim_a=i / 200),
            )

        entry = observer.get_latest_entry()
        data = entry.to_dict()

        # Entry should be small (no raw data)
        json_str = json.dumps(data)
        # A reasonable size for aggregated data (not 100 raw samples)
        assert len(json_str) < 5000
