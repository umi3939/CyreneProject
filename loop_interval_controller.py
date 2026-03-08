"""
loop_interval_controller.py - Main Loop 3-Path Interval Controller

Controls the timing intervals for the three input pathways in the main loop:
A. Screen capture interval: adaptive based on capture results
B. Spontaneous activation check interval: linked to check results
C. Text input - screen capture coordination
D. Arousal-based interval modulation (post-stage micro-adjustment)

This module only controls timing intervals. It does NOT reference
psyche module states or outputs. All decisions are based solely on
observable facts within the main loop itself:
- Capture result (frame present or None)
- Text queue state (empty or not)
- Spontaneous activation result (activated or not)
- Arousal value (passed as argument by caller, not read from psyche)

No state is persisted across sessions.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Counter overflow prevention: halve all counters when any exceeds this
_TELEMETRY_COUNTER_MAX: int = 2_000_000_000


# --- Interval Stage Definitions ---

# Screen capture interval stages (seconds)
# Stages expand gradually when no change is detected
CAPTURE_INTERVAL_STAGES: list[float] = [
    0.1,   # Stage 0: base interval (immediate re-check)
    0.5,   # Stage 1: short delay
    1.0,   # Stage 2: moderate delay
    2.0,   # Stage 3: longer delay
    5.0,   # Stage 4: max delay (upper bound)
]

# Spontaneous check additional wait stages (seconds)
# Added on top of the idle threshold when check returns "not activated"
SPONTANEOUS_WAIT_STAGES: list[float] = [
    0.0,   # Stage 0: no additional wait (base)
    5.0,   # Stage 1: short additional wait
    10.0,  # Stage 2: moderate additional wait
    15.0,  # Stage 3: max additional wait (upper bound)
]

# Text input - capture coordination
# After this many consecutive text inputs without a screen capture,
# force a screen capture opportunity
TEXT_CONSECUTIVE_LIMIT: int = 5

# --- Arousal Modulation Constants ---

# Maximum modulation ratio (+-10% of base interval)
_AROUSAL_MOD_MAX_RATIO: float = 0.1

# Arousal neutral point (center of arousal range 0.0-1.0)
_AROUSAL_NEUTRAL: float = 0.5

# Number of recent arousal values to keep for safety valve detection
_AROUSAL_HISTORY_SIZE: int = 5

# How many consecutive same-direction changes trigger the safety valve
_AROUSAL_MONOTONIC_THRESHOLD: int = 4

# How many direction reversals needed to release the safety valve (hysteresis)
_AROUSAL_HYSTERESIS_REVERSALS: int = 2

# Maximum modulation effect records to keep (FIFO diagnostic)
_AROUSAL_EFFECT_RECORD_SIZE: int = 20


@dataclass
class ArousalModulationState:
    """State for arousal-based loop interval modulation.

    All fields are ephemeral (not persisted across sessions).
    """
    # Recent arousal values for monotonic change detection
    arousal_history: deque = field(default_factory=lambda: deque(maxlen=_AROUSAL_HISTORY_SIZE))
    # Safety valve: True = modulation disabled (monotonic change detected)
    safety_valve_active: bool = False
    # Counter for direction reversals needed to release safety valve
    reversal_count: int = 0

    def reset(self) -> None:
        """Reset to initial state (no modulation, safety valve off)."""
        self.arousal_history.clear()
        self.safety_valve_active = False
        self.reversal_count = 0


@dataclass
class ArousalModulationRecord:
    """Single record of arousal modulation effect (diagnostic only)."""
    arousal_value: float = 0.0
    modulation_coefficient: float = 1.0
    base_interval: float = 0.0
    effective_interval: float = 0.0
    safety_valve_active: bool = False


@dataclass
class CaptureIntervalState:
    """State for screen capture interval control."""
    current_stage: int = 0
    last_capture_time: float = 0.0
    consecutive_no_change: int = 0

    def reset(self, now: Optional[float] = None) -> None:
        """Reset to base stage."""
        self.current_stage = 0
        self.consecutive_no_change = 0
        if now is not None:
            self.last_capture_time = now


@dataclass
class SpontaneousIntervalState:
    """State for spontaneous activation check interval control."""
    last_check_result_activated: bool = False
    last_check_time: float = 0.0
    additional_wait_stage: int = 0

    def reset(self, now: Optional[float] = None) -> None:
        """Reset to base stage."""
        self.additional_wait_stage = 0
        self.last_check_result_activated = False
        if now is not None:
            self.last_check_time = now


@dataclass
class TextCaptureCoordState:
    """State for text input - screen capture coordination."""
    text_just_processed: bool = False
    consecutive_text_count: int = 0

    def reset(self) -> None:
        """Reset coordination state."""
        self.text_just_processed = False
        self.consecutive_text_count = 0


@dataclass
class TelemetryCounters:
    """
    Read-only event counters for pathway utilization recording.
    These counters are NEVER referenced by control logic.
    They exist solely for diagnostic snapshot exposure.
    """
    text_processed_count: int = 0
    capture_attempt_count: int = 0
    capture_change_detected_count: int = 0
    spontaneous_check_count: int = 0
    spontaneous_activated_count: int = 0
    all_pathway_reset_count: int = 0

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.text_processed_count = 0
        self.capture_attempt_count = 0
        self.capture_change_detected_count = 0
        self.spontaneous_check_count = 0
        self.spontaneous_activated_count = 0
        self.all_pathway_reset_count = 0

    def _halve_all(self) -> None:
        """Halve all counters to prevent integer overflow while preserving ratios."""
        self.text_processed_count //= 2
        self.capture_attempt_count //= 2
        self.capture_change_detected_count //= 2
        self.spontaneous_check_count //= 2
        self.spontaneous_activated_count //= 2
        self.all_pathway_reset_count //= 2

    def _check_overflow(self) -> None:
        """If any counter exceeds the max, halve all counters."""
        if (
            self.text_processed_count > _TELEMETRY_COUNTER_MAX
            or self.capture_attempt_count > _TELEMETRY_COUNTER_MAX
            or self.capture_change_detected_count > _TELEMETRY_COUNTER_MAX
            or self.spontaneous_check_count > _TELEMETRY_COUNTER_MAX
            or self.spontaneous_activated_count > _TELEMETRY_COUNTER_MAX
            or self.all_pathway_reset_count > _TELEMETRY_COUNTER_MAX
        ):
            self._halve_all()

    def to_dict(self) -> dict:
        """Return counter values as a dictionary."""
        return {
            "text_processed_count": self.text_processed_count,
            "capture_attempt_count": self.capture_attempt_count,
            "capture_change_detected_count": self.capture_change_detected_count,
            "spontaneous_check_count": self.spontaneous_check_count,
            "spontaneous_activated_count": self.spontaneous_activated_count,
            "all_pathway_reset_count": self.all_pathway_reset_count,
        }


class LoopIntervalController:
    """
    Controls timing intervals for the main loop's three input pathways.

    All decisions are based solely on observable loop-level facts:
    - Capture layer return value (frame or None)
    - Text queue state (empty or not)
    - Spontaneous check result (activated or not)

    No psyche module states or outputs are referenced.
    No state is persisted across sessions.
    """

    def __init__(
        self,
        capture_stages: Optional[list[float]] = None,
        spontaneous_stages: Optional[list[float]] = None,
        text_consecutive_limit: Optional[int] = None,
        base_idle_threshold: float = 30.0,
    ):
        """
        Initialize the controller.

        Args:
            capture_stages: Custom capture interval stages (seconds).
                Defaults to CAPTURE_INTERVAL_STAGES.
            spontaneous_stages: Custom spontaneous wait stages (seconds).
                Defaults to SPONTANEOUS_WAIT_STAGES.
            text_consecutive_limit: Max consecutive text inputs before
                forcing a capture opportunity. Defaults to TEXT_CONSECUTIVE_LIMIT.
            base_idle_threshold: Base idle seconds threshold for spontaneous
                activation. Defaults to 30.0.
        """
        self._capture_stages = (
            list(capture_stages) if capture_stages is not None
            else list(CAPTURE_INTERVAL_STAGES)
        )
        self._spontaneous_stages = (
            list(spontaneous_stages) if spontaneous_stages is not None
            else list(SPONTANEOUS_WAIT_STAGES)
        )
        self._text_consecutive_limit = (
            text_consecutive_limit if text_consecutive_limit is not None
            else TEXT_CONSECUTIVE_LIMIT
        )
        self._base_idle_threshold = base_idle_threshold

        # Validate stages
        if len(self._capture_stages) == 0:
            self._capture_stages = [0.1]
        if len(self._spontaneous_stages) == 0:
            self._spontaneous_stages = [0.0]
        if self._text_consecutive_limit < 1:
            self._text_consecutive_limit = 1

        # Internal states
        self._capture_state = CaptureIntervalState()
        self._spontaneous_state = SpontaneousIntervalState()
        self._text_capture_coord = TextCaptureCoordState()

        # Telemetry counters (diagnostic-only, never referenced by control logic)
        self._telemetry = TelemetryCounters()

        # Arousal modulation state (ephemeral, not persisted)
        self._arousal_state = ArousalModulationState()
        self._arousal_effect_log: deque[ArousalModulationRecord] = deque(
            maxlen=_AROUSAL_EFFECT_RECORD_SIZE,
        )

        # Initialize times
        now = time.monotonic()
        self._capture_state.last_capture_time = now
        self._spontaneous_state.last_check_time = now

    # --- Properties for inspection ---

    @property
    def capture_state(self) -> CaptureIntervalState:
        """Read-only access to capture interval state."""
        return self._capture_state

    @property
    def spontaneous_state(self) -> SpontaneousIntervalState:
        """Read-only access to spontaneous interval state."""
        return self._spontaneous_state

    @property
    def text_capture_coord(self) -> TextCaptureCoordState:
        """Read-only access to text-capture coordination state."""
        return self._text_capture_coord

    @property
    def capture_stages(self) -> list[float]:
        """The configured capture interval stages."""
        return list(self._capture_stages)

    @property
    def spontaneous_stages(self) -> list[float]:
        """The configured spontaneous wait stages."""
        return list(self._spontaneous_stages)

    @property
    def text_consecutive_limit(self) -> int:
        """Max consecutive text inputs before forced capture."""
        return self._text_consecutive_limit

    @property
    def base_idle_threshold(self) -> float:
        """Base idle seconds threshold for spontaneous activation."""
        return self._base_idle_threshold

    @property
    def telemetry(self) -> TelemetryCounters:
        """Read-only access to telemetry counters."""
        return self._telemetry

    @property
    def arousal_state(self) -> ArousalModulationState:
        """Read-only access to arousal modulation state."""
        return self._arousal_state

    @property
    def arousal_effect_log(self) -> deque:
        """Read-only access to arousal modulation effect records."""
        return self._arousal_effect_log

    # --- A. Screen Capture Interval Control ---

    def should_attempt_capture(self, now: Optional[float] = None) -> bool:
        """
        Determine whether a screen capture should be attempted at this moment.

        Based on:
        - Current interval stage
        - Elapsed time since last capture attempt

        Args:
            now: Current monotonic time. Defaults to time.monotonic().

        Returns:
            True if enough time has elapsed to attempt capture.
        """
        if now is None:
            now = time.monotonic()

        stage_idx = min(
            self._capture_state.current_stage,
            len(self._capture_stages) - 1,
        )
        required_interval = self._capture_stages[stage_idx]
        elapsed = now - self._capture_state.last_capture_time
        return elapsed >= required_interval

    def on_capture_result(
        self, frame_present: bool, now: Optional[float] = None,
    ) -> None:
        """
        Update capture interval state based on capture result.

        Args:
            frame_present: True if capture returned a frame (change detected),
                False if capture returned None (no change).
            now: Current monotonic time.
        """
        if now is None:
            now = time.monotonic()

        self._capture_state.last_capture_time = now

        # Telemetry: record capture attempt
        self._telemetry.capture_attempt_count += 1
        if frame_present:
            self._telemetry.capture_change_detected_count += 1
        self._telemetry._check_overflow()

        if frame_present:
            # Change detected: reset to base stage
            self._capture_state.current_stage = 0
            self._capture_state.consecutive_no_change = 0
        else:
            # No change: advance stage (extend interval)
            self._capture_state.consecutive_no_change += 1
            max_stage = len(self._capture_stages) - 1
            if self._capture_state.current_stage < max_stage:
                self._capture_state.current_stage += 1

    def get_current_capture_interval(self) -> float:
        """Return the current capture interval in seconds."""
        stage_idx = min(
            self._capture_state.current_stage,
            len(self._capture_stages) - 1,
        )
        return self._capture_stages[stage_idx]

    # --- B. Spontaneous Activation Check Interval Control ---

    def should_check_spontaneous(
        self, idle_seconds: float, now: Optional[float] = None,
    ) -> bool:
        """
        Determine whether a spontaneous activation check should be performed.

        Args:
            idle_seconds: Seconds since last activity.
            now: Current monotonic time.

        Returns:
            True if idle threshold (with additional wait) is exceeded
            and enough time has passed since last check.
        """
        if now is None:
            now = time.monotonic()

        # Calculate effective threshold: base + additional wait from stage
        stage_idx = min(
            self._spontaneous_state.additional_wait_stage,
            len(self._spontaneous_stages) - 1,
        )
        additional_wait = self._spontaneous_stages[stage_idx]
        effective_threshold = self._base_idle_threshold + additional_wait

        # Also ensure minimum time since last check
        time_since_last_check = now - self._spontaneous_state.last_check_time
        min_recheck_interval = additional_wait if additional_wait > 0 else 0.0

        return (
            idle_seconds >= effective_threshold
            and time_since_last_check >= min_recheck_interval
        )

    def on_spontaneous_result(
        self, activated: bool, now: Optional[float] = None,
    ) -> None:
        """
        Update spontaneous check interval state based on check result.

        Args:
            activated: True if spontaneous activation occurred (spoke),
                False if check returned "do not activate".
            now: Current monotonic time.
        """
        if now is None:
            now = time.monotonic()

        self._spontaneous_state.last_check_time = now
        self._spontaneous_state.last_check_result_activated = activated

        # Telemetry: record spontaneous check
        self._telemetry.spontaneous_check_count += 1
        if activated:
            self._telemetry.spontaneous_activated_count += 1
        self._telemetry._check_overflow()

        if activated:
            # Activation occurred: reset additional wait
            self._spontaneous_state.additional_wait_stage = 0
        else:
            # No activation: increase additional wait stage
            max_stage = len(self._spontaneous_stages) - 1
            if self._spontaneous_state.additional_wait_stage < max_stage:
                self._spontaneous_state.additional_wait_stage += 1

    def get_effective_idle_threshold(self) -> float:
        """Return the current effective idle threshold including additional wait."""
        stage_idx = min(
            self._spontaneous_state.additional_wait_stage,
            len(self._spontaneous_stages) - 1,
        )
        return self._base_idle_threshold + self._spontaneous_stages[stage_idx]

    # --- C. Text Input - Screen Capture Coordination ---

    def on_text_processed(self) -> None:
        """
        Notify the controller that a text input was processed.
        Increments consecutive text count and sets the flag for
        post-text capture reset.
        """
        self._text_capture_coord.text_just_processed = True
        self._text_capture_coord.consecutive_text_count += 1

        # Telemetry: record text processing
        self._telemetry.text_processed_count += 1
        self._telemetry._check_overflow()

    def should_force_capture_after_text(self) -> bool:
        """
        Check if a screen capture should be forced due to consecutive
        text inputs reaching the limit.

        Returns:
            True if consecutive text count >= limit.
        """
        return (
            self._text_capture_coord.consecutive_text_count
            >= self._text_consecutive_limit
        )

    def on_capture_after_text(self, now: Optional[float] = None) -> None:
        """
        Called after a forced capture due to text limit.
        Resets the consecutive text counter.
        """
        self._text_capture_coord.consecutive_text_count = 0
        if now is not None:
            self._capture_state.last_capture_time = now

    def check_and_apply_text_capture_reset(
        self, now: Optional[float] = None,
    ) -> bool:
        """
        If text was processed in the previous loop iteration,
        reset capture interval to base stage. This ensures
        post-text screen changes are detected promptly.

        Returns:
            True if a reset was applied.
        """
        if self._text_capture_coord.text_just_processed:
            self._text_capture_coord.text_just_processed = False
            self._capture_state.reset(now=now)
            return True
        return False

    # --- Cross-pathway reset (safety valve) ---

    def on_any_activity(self, now: Optional[float] = None) -> None:
        """
        Called when any pathway produces actual output (text response,
        screen perception, or spontaneous speech). Resets all interval
        states to base stage.

        This is the multi-pathway reset safety valve: activity on any
        pathway restores detection opportunity on all other pathways.

        Args:
            now: Current monotonic time.
        """
        if now is None:
            now = time.monotonic()

        self._capture_state.reset(now=now)
        self._spontaneous_state.reset(now=now)
        self._text_capture_coord.reset()

        # Telemetry: record all-pathway reset
        self._telemetry.all_pathway_reset_count += 1
        self._telemetry._check_overflow()

    # --- D. Arousal-Based Interval Modulation ---

    def _detect_monotonic_arousal(self) -> bool:
        """Check if recent arousal values show monotonic (same-direction) change.

        Returns True if the last _AROUSAL_MONOTONIC_THRESHOLD consecutive
        changes are all in the same direction (all increasing or all decreasing).
        """
        history = self._arousal_state.arousal_history
        if len(history) < _AROUSAL_MONOTONIC_THRESHOLD + 1:
            return False

        # Check the last THRESHOLD consecutive differences
        recent = list(history)[-(_AROUSAL_MONOTONIC_THRESHOLD + 1):]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]

        # All positive or all negative (ignore zero as non-directional)
        all_positive = all(d > 0 for d in diffs)
        all_negative = all(d < 0 for d in diffs)
        return all_positive or all_negative

    def _detect_reversal(self) -> bool:
        """Check if the most recent arousal change is a direction reversal
        compared to the one before it.

        Returns True if the last two consecutive differences have opposite signs.
        """
        history = self._arousal_state.arousal_history
        if len(history) < 3:
            return False

        recent = list(history)[-3:]
        d1 = recent[1] - recent[0]
        d2 = recent[2] - recent[1]

        # Reversal: signs differ (and neither is zero)
        if d1 == 0 or d2 == 0:
            return False
        return (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)

    def _compute_arousal_coefficient(self, arousal: float) -> float:
        """Compute the modulation coefficient from arousal value.

        Higher arousal -> coefficient < 1.0 (shorter interval)
        Lower arousal -> coefficient > 1.0 (longer interval)

        Coefficient is clamped to [1 - MAX_RATIO, 1 + MAX_RATIO].
        """
        # Linear mapping: deviation from neutral scaled to +-MAX_RATIO
        deviation = arousal - _AROUSAL_NEUTRAL
        # Normalize: deviation range is [-0.5, 0.5], map to [-MAX_RATIO, MAX_RATIO]
        raw_mod = -(deviation / 0.5) * _AROUSAL_MOD_MAX_RATIO
        # Clamp
        clamped = max(-_AROUSAL_MOD_MAX_RATIO, min(_AROUSAL_MOD_MAX_RATIO, raw_mod))
        return 1.0 + clamped

    def apply_arousal_modulation(
        self, base_interval: float, arousal: float,
    ) -> float:
        """Apply arousal-based modulation to a base interval.

        This is the post-stage modulation layer. The base_interval is the
        value already determined by the 3-path control (stage-based).
        The arousal value is passed by the caller (not read from psyche).

        Args:
            base_interval: The interval determined by existing 3-path control.
            arousal: Current arousal value (typically 0.0-1.0), passed by caller.

        Returns:
            The effective interval after arousal modulation.
            If the safety valve is active, returns base_interval unchanged.
        """
        # Clamp arousal to valid range
        arousal = max(0.0, min(1.0, arousal))

        # Record arousal in history
        self._arousal_state.arousal_history.append(arousal)

        # Safety valve logic
        if self._arousal_state.safety_valve_active:
            # Check for direction reversal to count toward release
            if self._detect_reversal():
                self._arousal_state.reversal_count += 1
            else:
                # Non-reversal does not reset the counter (accumulative)
                pass

            # Release safety valve after enough reversals
            if self._arousal_state.reversal_count >= _AROUSAL_HYSTERESIS_REVERSALS:
                self._arousal_state.safety_valve_active = False
                self._arousal_state.reversal_count = 0
            else:
                # Safety valve still active: no modulation
                record = ArousalModulationRecord(
                    arousal_value=arousal,
                    modulation_coefficient=1.0,
                    base_interval=base_interval,
                    effective_interval=base_interval,
                    safety_valve_active=True,
                )
                self._arousal_effect_log.append(record)
                return base_interval
        else:
            # Check if monotonic change triggers safety valve
            if self._detect_monotonic_arousal():
                self._arousal_state.safety_valve_active = True
                self._arousal_state.reversal_count = 0
                # Immediately disable modulation
                record = ArousalModulationRecord(
                    arousal_value=arousal,
                    modulation_coefficient=1.0,
                    base_interval=base_interval,
                    effective_interval=base_interval,
                    safety_valve_active=True,
                )
                self._arousal_effect_log.append(record)
                return base_interval

        # Compute and apply modulation
        coefficient = self._compute_arousal_coefficient(arousal)
        effective_interval = base_interval * coefficient

        # Record effect
        record = ArousalModulationRecord(
            arousal_value=arousal,
            modulation_coefficient=coefficient,
            base_interval=base_interval,
            effective_interval=effective_interval,
            safety_valve_active=False,
        )
        self._arousal_effect_log.append(record)

        return effective_interval

    def get_arousal_diagnostics(self) -> dict:
        """Return arousal modulation diagnostic snapshot.

        For debugging/logging purposes only. Not referenced by control logic.
        """
        history = list(self._arousal_state.arousal_history)
        recent_effects = [
            {
                "arousal": r.arousal_value,
                "coefficient": r.modulation_coefficient,
                "base_interval": r.base_interval,
                "effective_interval": r.effective_interval,
                "safety_valve_active": r.safety_valve_active,
            }
            for r in self._arousal_effect_log
        ]
        return {
            "arousal_history": history,
            "safety_valve_active": self._arousal_state.safety_valve_active,
            "reversal_count": self._arousal_state.reversal_count,
            "effect_log_size": len(self._arousal_effect_log),
            "recent_effects": recent_effects[-5:] if recent_effects else [],
        }

    # --- Snapshot for diagnostics ---

    def get_diagnostics(self) -> dict:
        """
        Return a diagnostic snapshot of all internal states.
        For debugging/logging purposes only.
        """
        return {
            "capture": {
                "current_stage": self._capture_state.current_stage,
                "current_interval": self.get_current_capture_interval(),
                "consecutive_no_change": self._capture_state.consecutive_no_change,
            },
            "spontaneous": {
                "additional_wait_stage": self._spontaneous_state.additional_wait_stage,
                "effective_threshold": self.get_effective_idle_threshold(),
                "last_result_activated": self._spontaneous_state.last_check_result_activated,
            },
            "text_capture_coord": {
                "text_just_processed": self._text_capture_coord.text_just_processed,
                "consecutive_text_count": self._text_capture_coord.consecutive_text_count,
                "force_capture_needed": self.should_force_capture_after_text(),
            },
            "telemetry": self._telemetry.to_dict(),
            "arousal_modulation": {
                "safety_valve_active": self._arousal_state.safety_valve_active,
                "history_length": len(self._arousal_state.arousal_history),
                "effect_log_size": len(self._arousal_effect_log),
            },
        }
