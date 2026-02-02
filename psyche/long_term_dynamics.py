"""
psyche/long_term_dynamics.py - Long-Term Dynamics Logging (長期挙動ログ)

Implements passive observation of long-term behavioral patterns,
aggregating data over windows without affecting system behavior.

Design principles (from design_long_term_dynamics.md):
- PASSIVE observation only (does NOT trigger behavior changes)
- Aggregated stats (averages, variance, counts) not raw logs
- Window-based collection (every N turns)
- Lightweight (don't save every interaction)
- Append-only logs, never modified

Usage::

    from psyche.long_term_dynamics import (
        DynamicsObserver,
        WindowStats,
        LongTermEntry,
        create_observer,
    )

    # Create observer
    observer = DynamicsObserver(window_size=10)

    # Record observations passively (call each turn)
    observer.record_turn(
        emotion_state=emotion,
        decision_label="共感する",
        value_orientation=orientation,
    )

    # Automatically aggregates when window completes
    # Get recent entries
    entries = observer.get_recent_entries(5)

    # Save to file
    observer.save_to_file("long_term_log.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
import json
import time
import math
import threading


# ── Statistics Helpers ──────────────────────────────────────────────


def compute_mean(values: list[float]) -> float:
    """Compute mean of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_variance(values: list[float]) -> float:
    """Compute variance of values."""
    if len(values) < 2:
        return 0.0
    mean = compute_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def compute_std(values: list[float]) -> float:
    """Compute standard deviation."""
    return math.sqrt(compute_variance(values))


def compute_min_max(values: list[float]) -> tuple[float, float]:
    """Compute min and max."""
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


# ── Window Statistics ───────────────────────────────────────────────


@dataclass
class EmotionStats:
    """Aggregated emotion statistics for a window."""
    joy_mean: float = 0.0
    joy_std: float = 0.0
    anger_mean: float = 0.0
    anger_std: float = 0.0
    sadness_mean: float = 0.0
    sadness_std: float = 0.0
    fear_mean: float = 0.0
    fear_std: float = 0.0
    disgust_mean: float = 0.0
    disgust_std: float = 0.0
    surprise_mean: float = 0.0
    surprise_std: float = 0.0

    # Derived metrics
    dominant_emotion: str = ""
    peak_count: int = 0  # Times any emotion exceeded threshold
    average_intensity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "joy": {"mean": round(self.joy_mean, 4), "std": round(self.joy_std, 4)},
            "anger": {"mean": round(self.anger_mean, 4), "std": round(self.anger_std, 4)},
            "sadness": {"mean": round(self.sadness_mean, 4), "std": round(self.sadness_std, 4)},
            "fear": {"mean": round(self.fear_mean, 4), "std": round(self.fear_std, 4)},
            "disgust": {"mean": round(self.disgust_mean, 4), "std": round(self.disgust_std, 4)},
            "surprise": {"mean": round(self.surprise_mean, 4), "std": round(self.surprise_std, 4)},
            "dominant_emotion": self.dominant_emotion,
            "peak_count": self.peak_count,
            "average_intensity": round(self.average_intensity, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionStats":
        return cls(
            joy_mean=data.get("joy", {}).get("mean", 0.0),
            joy_std=data.get("joy", {}).get("std", 0.0),
            anger_mean=data.get("anger", {}).get("mean", 0.0),
            anger_std=data.get("anger", {}).get("std", 0.0),
            sadness_mean=data.get("sadness", {}).get("mean", 0.0),
            sadness_std=data.get("sadness", {}).get("std", 0.0),
            fear_mean=data.get("fear", {}).get("mean", 0.0),
            fear_std=data.get("fear", {}).get("std", 0.0),
            disgust_mean=data.get("disgust", {}).get("mean", 0.0),
            disgust_std=data.get("disgust", {}).get("std", 0.0),
            surprise_mean=data.get("surprise", {}).get("mean", 0.0),
            surprise_std=data.get("surprise", {}).get("std", 0.0),
            dominant_emotion=data.get("dominant_emotion", ""),
            peak_count=data.get("peak_count", 0),
            average_intensity=data.get("average_intensity", 0.0),
        )


@dataclass
class DecisionStats:
    """Aggregated decision statistics for a window."""
    total_decisions: int = 0
    silence_count: int = 0
    silence_rate: float = 0.0
    light_tone_count: int = 0
    light_tone_rate: float = 0.0
    serious_tone_count: int = 0
    serious_tone_rate: float = 0.0
    normal_count: int = 0
    normal_rate: float = 0.0

    # Policy distribution
    policy_counts: dict[str, int] = field(default_factory=dict)
    most_common_policy: str = ""
    unique_policies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "silence": {"count": self.silence_count, "rate": round(self.silence_rate, 4)},
            "light_tone": {"count": self.light_tone_count, "rate": round(self.light_tone_rate, 4)},
            "serious_tone": {"count": self.serious_tone_count, "rate": round(self.serious_tone_rate, 4)},
            "normal": {"count": self.normal_count, "rate": round(self.normal_rate, 4)},
            "policy_counts": self.policy_counts.copy(),
            "most_common_policy": self.most_common_policy,
            "unique_policies": self.unique_policies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionStats":
        return cls(
            total_decisions=data.get("total_decisions", 0),
            silence_count=data.get("silence", {}).get("count", 0),
            silence_rate=data.get("silence", {}).get("rate", 0.0),
            light_tone_count=data.get("light_tone", {}).get("count", 0),
            light_tone_rate=data.get("light_tone", {}).get("rate", 0.0),
            serious_tone_count=data.get("serious_tone", {}).get("count", 0),
            serious_tone_rate=data.get("serious_tone", {}).get("rate", 0.0),
            normal_count=data.get("normal", {}).get("count", 0),
            normal_rate=data.get("normal", {}).get("rate", 0.0),
            policy_counts=data.get("policy_counts", {}),
            most_common_policy=data.get("most_common_policy", ""),
            unique_policies=data.get("unique_policies", 0),
        )


@dataclass
class ValueOrientationStats:
    """Aggregated value orientation statistics for a window."""
    dim_a_mean: float = 0.0
    dim_a_std: float = 0.0
    dim_a_delta: float = 0.0  # Change from window start to end
    dim_b_mean: float = 0.0
    dim_b_std: float = 0.0
    dim_b_delta: float = 0.0
    dim_c_mean: float = 0.0
    dim_c_std: float = 0.0
    dim_c_delta: float = 0.0
    dim_d_mean: float = 0.0
    dim_d_std: float = 0.0
    dim_d_delta: float = 0.0
    dim_e_mean: float = 0.0
    dim_e_std: float = 0.0
    dim_e_delta: float = 0.0

    overall_stability: float = 0.0
    most_changed_dim: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_a": {"mean": round(self.dim_a_mean, 4), "std": round(self.dim_a_std, 4), "delta": round(self.dim_a_delta, 4)},
            "dim_b": {"mean": round(self.dim_b_mean, 4), "std": round(self.dim_b_std, 4), "delta": round(self.dim_b_delta, 4)},
            "dim_c": {"mean": round(self.dim_c_mean, 4), "std": round(self.dim_c_std, 4), "delta": round(self.dim_c_delta, 4)},
            "dim_d": {"mean": round(self.dim_d_mean, 4), "std": round(self.dim_d_std, 4), "delta": round(self.dim_d_delta, 4)},
            "dim_e": {"mean": round(self.dim_e_mean, 4), "std": round(self.dim_e_std, 4), "delta": round(self.dim_e_delta, 4)},
            "overall_stability": round(self.overall_stability, 4),
            "most_changed_dim": self.most_changed_dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueOrientationStats":
        return cls(
            dim_a_mean=data.get("dim_a", {}).get("mean", 0.0),
            dim_a_std=data.get("dim_a", {}).get("std", 0.0),
            dim_a_delta=data.get("dim_a", {}).get("delta", 0.0),
            dim_b_mean=data.get("dim_b", {}).get("mean", 0.0),
            dim_b_std=data.get("dim_b", {}).get("std", 0.0),
            dim_b_delta=data.get("dim_b", {}).get("delta", 0.0),
            dim_c_mean=data.get("dim_c", {}).get("mean", 0.0),
            dim_c_std=data.get("dim_c", {}).get("std", 0.0),
            dim_c_delta=data.get("dim_c", {}).get("delta", 0.0),
            dim_d_mean=data.get("dim_d", {}).get("mean", 0.0),
            dim_d_std=data.get("dim_d", {}).get("std", 0.0),
            dim_d_delta=data.get("dim_d", {}).get("delta", 0.0),
            dim_e_mean=data.get("dim_e", {}).get("mean", 0.0),
            dim_e_std=data.get("dim_e", {}).get("std", 0.0),
            dim_e_delta=data.get("dim_e", {}).get("delta", 0.0),
            overall_stability=data.get("overall_stability", 0.0),
            most_changed_dim=data.get("most_changed_dim", ""),
        )


@dataclass
class ResponsibilityStats:
    """Aggregated responsibility statistics for a window."""
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_max: float = 0.0
    caution_mean: float = 0.0
    caution_std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": {"mean": round(self.weight_mean, 4), "std": round(self.weight_std, 4), "max": round(self.weight_max, 4)},
            "caution": {"mean": round(self.caution_mean, 4), "std": round(self.caution_std, 4)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponsibilityStats":
        return cls(
            weight_mean=data.get("weight", {}).get("mean", 0.0),
            weight_std=data.get("weight", {}).get("std", 0.0),
            weight_max=data.get("weight", {}).get("max", 0.0),
            caution_mean=data.get("caution", {}).get("mean", 0.0),
            caution_std=data.get("caution", {}).get("std", 0.0),
        )


@dataclass
class StabilityValveStats:
    """Aggregated stability valve statistics for a window."""
    activation_count: int = 0
    activation_rate: float = 0.0
    average_activation_level: float = 0.0
    max_activation_level: float = 0.0
    dominant_trigger: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_count": self.activation_count,
            "activation_rate": round(self.activation_rate, 4),
            "average_activation_level": round(self.average_activation_level, 4),
            "max_activation_level": round(self.max_activation_level, 4),
            "dominant_trigger": self.dominant_trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StabilityValveStats":
        return cls(
            activation_count=data.get("activation_count", 0),
            activation_rate=data.get("activation_rate", 0.0),
            average_activation_level=data.get("average_activation_level", 0.0),
            max_activation_level=data.get("max_activation_level", 0.0),
            dominant_trigger=data.get("dominant_trigger", ""),
        )


@dataclass
class WindowStats:
    """Complete statistics for a single observation window."""
    emotion: EmotionStats = field(default_factory=EmotionStats)
    decision: DecisionStats = field(default_factory=DecisionStats)
    value_orientation: ValueOrientationStats = field(default_factory=ValueOrientationStats)
    responsibility: ResponsibilityStats = field(default_factory=ResponsibilityStats)
    stability_valve: StabilityValveStats = field(default_factory=StabilityValveStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "decision": self.decision.to_dict(),
            "value_orientation": self.value_orientation.to_dict(),
            "responsibility": self.responsibility.to_dict(),
            "stability_valve": self.stability_valve.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowStats":
        return cls(
            emotion=EmotionStats.from_dict(data.get("emotion", {})),
            decision=DecisionStats.from_dict(data.get("decision", {})),
            value_orientation=ValueOrientationStats.from_dict(data.get("value_orientation", {})),
            responsibility=ResponsibilityStats.from_dict(data.get("responsibility", {})),
            stability_valve=StabilityValveStats.from_dict(data.get("stability_valve", {})),
        )


# ── Long-Term Entry ─────────────────────────────────────────────────


@dataclass
class LongTermEntry:
    """A single entry in the long-term log."""
    entry_id: int = 0
    window_start_turn: int = 0
    window_end_turn: int = 0
    window_size: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    stats: WindowStats = field(default_factory=WindowStats)

    # Delta from previous window (if available)
    has_delta: bool = False
    delta_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "window_start_turn": self.window_start_turn,
            "window_end_turn": self.window_end_turn,
            "window_size": self.window_size,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "stats": self.stats.to_dict(),
            "has_delta": self.has_delta,
            "delta_summary": {k: round(v, 4) for k, v in self.delta_summary.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTermEntry":
        return cls(
            entry_id=data.get("entry_id", 0),
            window_start_turn=data.get("window_start_turn", 0),
            window_end_turn=data.get("window_end_turn", 0),
            window_size=data.get("window_size", 0),
            timestamp_start=data.get("timestamp_start", 0.0),
            timestamp_end=data.get("timestamp_end", 0.0),
            stats=WindowStats.from_dict(data.get("stats", {})),
            has_delta=data.get("has_delta", False),
            delta_summary=data.get("delta_summary", {}),
        )


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class DynamicsObserverConfig:
    """Configuration for dynamics observer."""
    # Window size (number of turns per window)
    window_size: int = 10

    # Maximum entries to keep in memory
    max_entries_in_memory: int = 100

    # Emotion peak threshold
    emotion_peak_threshold: float = 0.5

    # Auto-save interval (entries)
    auto_save_interval: int = 5

    # Log file path (None = no auto-save)
    log_file_path: Optional[str] = None

    # Enable/disable observation
    enabled: bool = True


# ── Raw Data Collector ──────────────────────────────────────────────


@dataclass
class WindowRawData:
    """Raw data collected within a window (not saved, only for aggregation)."""
    turn_count: int = 0
    start_turn: int = 0
    start_time: float = field(default_factory=time.time)

    # Emotion samples
    joy_samples: list[float] = field(default_factory=list)
    anger_samples: list[float] = field(default_factory=list)
    sadness_samples: list[float] = field(default_factory=list)
    fear_samples: list[float] = field(default_factory=list)
    disgust_samples: list[float] = field(default_factory=list)
    surprise_samples: list[float] = field(default_factory=list)
    peak_count: int = 0

    # Decision samples
    decision_labels: list[str] = field(default_factory=list)
    silence_count: int = 0
    light_tone_count: int = 0
    serious_tone_count: int = 0

    # Value orientation samples
    dim_a_samples: list[float] = field(default_factory=list)
    dim_b_samples: list[float] = field(default_factory=list)
    dim_c_samples: list[float] = field(default_factory=list)
    dim_d_samples: list[float] = field(default_factory=list)
    dim_e_samples: list[float] = field(default_factory=list)

    # Responsibility samples
    weight_samples: list[float] = field(default_factory=list)
    caution_samples: list[float] = field(default_factory=list)

    # Stability valve samples
    activation_levels: list[float] = field(default_factory=list)
    trigger_sources: list[str] = field(default_factory=list)


# ── Dynamics Observer ───────────────────────────────────────────────


class DynamicsObserver:
    """
    Passive observer that collects long-term dynamics statistics.

    IMPORTANT: This is PASSIVE observation only.
    It does NOT trigger any behavior changes.
    It runs quietly without blocking the main loop.
    """

    def __init__(self, config: Optional[DynamicsObserverConfig] = None):
        self.config = config or DynamicsObserverConfig()
        self._entries: list[LongTermEntry] = []
        self._current_window: WindowRawData = WindowRawData()
        self._total_turns: int = 0
        self._entry_counter: int = 0
        self._last_save_entry: int = 0
        self._lock = threading.Lock()

    def record_turn(
        self,
        emotion_state: Optional[Any] = None,
        decision_label: str = "",
        is_silence: bool = False,
        tone: str = "neutral",
        value_orientation: Optional[Any] = None,
        responsibility_weight: float = 0.0,
        responsibility_caution: float = 0.0,
        stability_activation: float = 0.0,
        stability_trigger: str = "",
    ) -> Optional[LongTermEntry]:
        """
        Record a single turn's observations.

        This is PASSIVE - it only collects data.
        Returns a LongTermEntry if a window was completed, None otherwise.
        """
        if not self.config.enabled:
            return None

        with self._lock:
            self._total_turns += 1

            # Initialize window if needed
            if self._current_window.turn_count == 0:
                self._current_window.start_turn = self._total_turns
                self._current_window.start_time = time.time()

            self._current_window.turn_count += 1

            # Collect emotion samples
            if emotion_state is not None:
                self._collect_emotion_sample(emotion_state)

            # Collect decision samples
            if decision_label:
                self._current_window.decision_labels.append(decision_label)
            if is_silence:
                self._current_window.silence_count += 1
            if tone == "light":
                self._current_window.light_tone_count += 1
            elif tone == "serious":
                self._current_window.serious_tone_count += 1

            # Collect value orientation samples
            if value_orientation is not None:
                self._collect_value_sample(value_orientation)

            # Collect responsibility samples
            if responsibility_weight > 0 or responsibility_caution > 0:
                self._current_window.weight_samples.append(responsibility_weight)
                self._current_window.caution_samples.append(responsibility_caution)

            # Collect stability valve samples
            if stability_activation > 0:
                self._current_window.activation_levels.append(stability_activation)
                if stability_trigger:
                    self._current_window.trigger_sources.append(stability_trigger)

            # Check if window is complete
            if self._current_window.turn_count >= self.config.window_size:
                return self._complete_window()

            return None

    def _collect_emotion_sample(self, emotion_state: Any) -> None:
        """Collect emotion sample from state."""
        joy = getattr(emotion_state, "joy", 0.0)
        anger = getattr(emotion_state, "anger", 0.0)
        sadness = getattr(emotion_state, "sadness", 0.0)
        fear = getattr(emotion_state, "fear", 0.0)
        disgust = getattr(emotion_state, "disgust", 0.0)
        surprise = getattr(emotion_state, "surprise", 0.0)

        self._current_window.joy_samples.append(joy)
        self._current_window.anger_samples.append(anger)
        self._current_window.sadness_samples.append(sadness)
        self._current_window.fear_samples.append(fear)
        self._current_window.disgust_samples.append(disgust)
        self._current_window.surprise_samples.append(surprise)

        # Count peaks
        threshold = self.config.emotion_peak_threshold
        if max(joy, anger, sadness, fear, disgust, surprise) > threshold:
            self._current_window.peak_count += 1

    def _collect_value_sample(self, value_orientation: Any) -> None:
        """Collect value orientation sample."""
        self._current_window.dim_a_samples.append(getattr(value_orientation, "dim_a", 0.0))
        self._current_window.dim_b_samples.append(getattr(value_orientation, "dim_b", 0.0))
        self._current_window.dim_c_samples.append(getattr(value_orientation, "dim_c", 0.0))
        self._current_window.dim_d_samples.append(getattr(value_orientation, "dim_d", 0.0))
        self._current_window.dim_e_samples.append(getattr(value_orientation, "dim_e", 0.0))

    def _complete_window(self) -> LongTermEntry:
        """Complete current window and create entry."""
        self._entry_counter += 1

        # Aggregate statistics
        stats = self._aggregate_stats()

        # Compute delta from previous entry
        has_delta = False
        delta_summary: dict[str, float] = {}

        if self._entries:
            prev = self._entries[-1]
            has_delta = True
            delta_summary = self._compute_delta(prev.stats, stats)

        # Create entry
        entry = LongTermEntry(
            entry_id=self._entry_counter,
            window_start_turn=self._current_window.start_turn,
            window_end_turn=self._total_turns,
            window_size=self._current_window.turn_count,
            timestamp_start=self._current_window.start_time,
            timestamp_end=time.time(),
            stats=stats,
            has_delta=has_delta,
            delta_summary=delta_summary,
        )

        # Store entry (trim if needed)
        self._entries.append(entry)
        if len(self._entries) > self.config.max_entries_in_memory:
            self._entries = self._entries[-self.config.max_entries_in_memory:]

        # Reset window
        self._current_window = WindowRawData()

        # Auto-save if configured
        if (self.config.log_file_path and
            self._entry_counter - self._last_save_entry >= self.config.auto_save_interval):
            self._auto_save()

        return entry

    def _aggregate_stats(self) -> WindowStats:
        """Aggregate raw data into statistics."""
        w = self._current_window

        # Emotion stats
        emotion = EmotionStats(
            joy_mean=compute_mean(w.joy_samples),
            joy_std=compute_std(w.joy_samples),
            anger_mean=compute_mean(w.anger_samples),
            anger_std=compute_std(w.anger_samples),
            sadness_mean=compute_mean(w.sadness_samples),
            sadness_std=compute_std(w.sadness_samples),
            fear_mean=compute_mean(w.fear_samples),
            fear_std=compute_std(w.fear_samples),
            disgust_mean=compute_mean(w.disgust_samples),
            disgust_std=compute_std(w.disgust_samples),
            surprise_mean=compute_mean(w.surprise_samples),
            surprise_std=compute_std(w.surprise_samples),
            peak_count=w.peak_count,
        )

        # Find dominant emotion
        means = {
            "joy": emotion.joy_mean,
            "anger": emotion.anger_mean,
            "sadness": emotion.sadness_mean,
            "fear": emotion.fear_mean,
            "disgust": emotion.disgust_mean,
            "surprise": emotion.surprise_mean,
        }
        if any(means.values()):
            emotion.dominant_emotion = max(means, key=means.get)
        emotion.average_intensity = sum(means.values())

        # Decision stats
        total = len(w.decision_labels)
        decision = DecisionStats(
            total_decisions=total,
            silence_count=w.silence_count,
            silence_rate=w.silence_count / total if total > 0 else 0.0,
            light_tone_count=w.light_tone_count,
            light_tone_rate=w.light_tone_count / total if total > 0 else 0.0,
            serious_tone_count=w.serious_tone_count,
            serious_tone_rate=w.serious_tone_count / total if total > 0 else 0.0,
            normal_count=total - w.silence_count,
            normal_rate=(total - w.silence_count) / total if total > 0 else 0.0,
        )

        # Policy counts
        from collections import Counter
        if w.decision_labels:
            counts = Counter(w.decision_labels)
            decision.policy_counts = dict(counts)
            decision.most_common_policy = counts.most_common(1)[0][0]
            decision.unique_policies = len(counts)

        # Value orientation stats
        value = ValueOrientationStats(
            dim_a_mean=compute_mean(w.dim_a_samples),
            dim_a_std=compute_std(w.dim_a_samples),
            dim_a_delta=w.dim_a_samples[-1] - w.dim_a_samples[0] if len(w.dim_a_samples) >= 2 else 0.0,
            dim_b_mean=compute_mean(w.dim_b_samples),
            dim_b_std=compute_std(w.dim_b_samples),
            dim_b_delta=w.dim_b_samples[-1] - w.dim_b_samples[0] if len(w.dim_b_samples) >= 2 else 0.0,
            dim_c_mean=compute_mean(w.dim_c_samples),
            dim_c_std=compute_std(w.dim_c_samples),
            dim_c_delta=w.dim_c_samples[-1] - w.dim_c_samples[0] if len(w.dim_c_samples) >= 2 else 0.0,
            dim_d_mean=compute_mean(w.dim_d_samples),
            dim_d_std=compute_std(w.dim_d_samples),
            dim_d_delta=w.dim_d_samples[-1] - w.dim_d_samples[0] if len(w.dim_d_samples) >= 2 else 0.0,
            dim_e_mean=compute_mean(w.dim_e_samples),
            dim_e_std=compute_std(w.dim_e_samples),
            dim_e_delta=w.dim_e_samples[-1] - w.dim_e_samples[0] if len(w.dim_e_samples) >= 2 else 0.0,
        )

        # Find most changed dimension
        deltas = {
            "a": abs(value.dim_a_delta),
            "b": abs(value.dim_b_delta),
            "c": abs(value.dim_c_delta),
            "d": abs(value.dim_d_delta),
            "e": abs(value.dim_e_delta),
        }
        if any(deltas.values()):
            value.most_changed_dim = max(deltas, key=deltas.get)

        # Overall stability (low variance = stable)
        stds = [value.dim_a_std, value.dim_b_std, value.dim_c_std, value.dim_d_std, value.dim_e_std]
        value.overall_stability = 1.0 - min(1.0, sum(stds) / 5.0)

        # Responsibility stats
        responsibility = ResponsibilityStats(
            weight_mean=compute_mean(w.weight_samples),
            weight_std=compute_std(w.weight_samples),
            weight_max=max(w.weight_samples) if w.weight_samples else 0.0,
            caution_mean=compute_mean(w.caution_samples),
            caution_std=compute_std(w.caution_samples),
        )

        # Stability valve stats
        activation_count = sum(1 for a in w.activation_levels if a > 0.1)
        stability = StabilityValveStats(
            activation_count=activation_count,
            activation_rate=activation_count / w.turn_count if w.turn_count > 0 else 0.0,
            average_activation_level=compute_mean(w.activation_levels),
            max_activation_level=max(w.activation_levels) if w.activation_levels else 0.0,
        )
        if w.trigger_sources:
            trigger_counts = Counter(w.trigger_sources)
            stability.dominant_trigger = trigger_counts.most_common(1)[0][0]

        return WindowStats(
            emotion=emotion,
            decision=decision,
            value_orientation=value,
            responsibility=responsibility,
            stability_valve=stability,
        )

    def _compute_delta(
        self,
        prev: WindowStats,
        curr: WindowStats,
    ) -> dict[str, float]:
        """Compute delta between two window stats."""
        return {
            "emotion_intensity_delta": curr.emotion.average_intensity - prev.emotion.average_intensity,
            "silence_rate_delta": curr.decision.silence_rate - prev.decision.silence_rate,
            "light_tone_delta": curr.decision.light_tone_rate - prev.decision.light_tone_rate,
            "value_stability_delta": curr.value_orientation.overall_stability - prev.value_orientation.overall_stability,
            "responsibility_weight_delta": curr.responsibility.weight_mean - prev.responsibility.weight_mean,
            "stability_activation_delta": curr.stability_valve.activation_rate - prev.stability_valve.activation_rate,
        }

    def _auto_save(self) -> None:
        """Auto-save entries to file (runs in background)."""
        if not self.config.log_file_path:
            return

        self._last_save_entry = self._entry_counter

        # Save in a separate thread to not block
        def save():
            try:
                self.save_to_file(self.config.log_file_path)
            except Exception:
                pass  # Silently fail to not affect main loop

        thread = threading.Thread(target=save, daemon=True)
        thread.start()

    def get_entries(self) -> list[LongTermEntry]:
        """Get all entries in memory."""
        with self._lock:
            return list(self._entries)

    def get_recent_entries(self, n: int = 5) -> list[LongTermEntry]:
        """Get N most recent entries."""
        with self._lock:
            return self._entries[-n:]

    def get_latest_entry(self) -> Optional[LongTermEntry]:
        """Get the latest entry."""
        with self._lock:
            return self._entries[-1] if self._entries else None

    def get_total_turns(self) -> int:
        """Get total turns observed."""
        return self._total_turns

    def get_entry_count(self) -> int:
        """Get total entries created."""
        return self._entry_counter

    def save_to_file(self, file_path: str) -> None:
        """Save entries to JSON file."""
        with self._lock:
            data = {
                "version": 1,
                "total_turns": self._total_turns,
                "entry_count": self._entry_counter,
                "window_size": self.config.window_size,
                "entries": [e.to_dict() for e in self._entries],
            }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, file_path: str) -> int:
        """Load entries from JSON file. Returns number of entries loaded."""
        path = Path(file_path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._total_turns = data.get("total_turns", 0)
            self._entry_counter = data.get("entry_count", 0)
            self._entries = [
                LongTermEntry.from_dict(e)
                for e in data.get("entries", [])
            ]

        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries and reset."""
        with self._lock:
            self._entries.clear()
            self._current_window = WindowRawData()
            self._total_turns = 0
            self._entry_counter = 0

    def is_enabled(self) -> bool:
        """Check if observation is enabled."""
        return self.config.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable observation."""
        self.config.enabled = enabled


# ── Convenience Functions ───────────────────────────────────────────


def create_observer(
    window_size: int = 10,
    log_file_path: Optional[str] = None,
    enabled: bool = True,
) -> DynamicsObserver:
    """Create a dynamics observer with custom settings."""
    config = DynamicsObserverConfig(
        window_size=window_size,
        log_file_path=log_file_path,
        enabled=enabled,
    )
    return DynamicsObserver(config)


def create_config(
    window_size: int = 10,
    max_entries: int = 100,
    auto_save_interval: int = 5,
) -> DynamicsObserverConfig:
    """Create observer configuration."""
    return DynamicsObserverConfig(
        window_size=window_size,
        max_entries_in_memory=max_entries,
        auto_save_interval=auto_save_interval,
    )


def get_observer_summary(observer: DynamicsObserver) -> str:
    """Get human-readable summary of observer state."""
    entries = observer.get_entry_count()
    turns = observer.get_total_turns()
    latest = observer.get_latest_entry()

    summary = f"DynamicsObserver: {entries} entries from {turns} turns"

    if latest:
        summary += f"\n  Latest window: turns {latest.window_start_turn}-{latest.window_end_turn}"
        summary += f"\n  Emotion intensity: {latest.stats.emotion.average_intensity:.2f}"
        summary += f"\n  Silence rate: {latest.stats.decision.silence_rate:.1%}"
        summary += f"\n  Value stability: {latest.stats.value_orientation.overall_stability:.2f}"

    return summary


def entries_to_json(entries: list[LongTermEntry]) -> list[dict[str, Any]]:
    """Convert entries to JSON-compatible format."""
    return [e.to_dict() for e in entries]


def entries_from_json(data: list[dict[str, Any]]) -> list[LongTermEntry]:
    """Load entries from JSON-compatible format."""
    return [LongTermEntry.from_dict(d) for d in data]
