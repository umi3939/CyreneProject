"""
psyche/silence_hesitation.py - Silence / Hesitation as a Choice

Implements silence and hesitation as valid decision outcomes, not errors.
Silence is a scorable candidate in the decision process, treated equally
with other response options.

Design principles (from design_silence_hesitation.md):
- Silence is an explicit choice, NOT "doing nothing"
- Silence is temporary, NOT permanent stoppage
- Silence references emotion/responsibility/self-reference states (read-only)
- Silence does NOT cause algorithm branching
- Internal state continues updating during silence

Usage::

    from psyche.silence_hesitation import (
        SilenceConfig,
        SilenceCandidate,
        SilenceResult,
        generate_silence_candidate,
        evaluate_silence_score,
        create_silence_result,
        is_silence_result,
    )

    # Generate silence as a candidate alongside other policies
    silence = generate_silence_candidate(state, percept, config)

    # Check if selected policy is silence
    if is_silence_result(policy):
        result = create_silence_result(policy, state)
        # result.duration indicates suggested pause duration
        # System continues to next cycle without blocking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time

from .state import PsycheState, Percept
from .responsibility import ResponsibilityInfluence
from .decision_bias import DecisionBias


class SilenceType(Enum):
    """Types of silence/hesitation."""

    # Emotional hesitation - feelings are conflicted
    EMOTIONAL_HESITATION = "emotional_hesitation"

    # Processing pause - need time to process
    PROCESSING_PAUSE = "processing_pause"

    # Respectful silence - listening, not interrupting
    RESPECTFUL_SILENCE = "respectful_silence"

    # Uncertain pause - unsure how to respond
    UNCERTAIN_PAUSE = "uncertain_pause"

    # Deliberate silence - choosing not to speak
    DELIBERATE_SILENCE = "deliberate_silence"


@dataclass
class SilenceConfig:
    """
    Configuration for silence/hesitation behavior.

    All parameters are externally configurable.
    No hardcoded conditions for when to be silent.
    """

    # Base score for silence (can be adjusted)
    base_silence_score: float = 0.5

    # Minimum score threshold for silence to be considered
    minimum_score_threshold: float = 0.3

    # Duration range for silence (in seconds)
    min_duration: float = 0.5
    max_duration: float = 3.0
    default_duration: float = 1.0

    # Score modifiers based on state conditions
    # (These are multipliers, not absolute values)
    emotional_conflict_bonus: float = 2.0  # Bonus when emotions conflict
    high_uncertainty_bonus: float = 1.5    # Bonus when uncertain
    low_arousal_bonus: float = 1.0         # Bonus when calm/quiet
    responsibility_weight_bonus: float = 1.5  # Bonus when carrying responsibility
    negative_mood_bonus: float = 1.2       # Bonus when mood is negative

    # Penalties
    high_expression_drive_penalty: float = 2.0  # Penalty when need to express
    high_social_drive_penalty: float = 1.5      # Penalty when need social contact
    positive_mood_penalty: float = 0.8          # Less silence when happy

    # Whether silence can be selected as top choice
    allow_as_top_choice: bool = True

    # Maximum consecutive silences (0 = no limit)
    max_consecutive_silences: int = 2


@dataclass
class SilenceCandidate:
    """
    Silence as a policy candidate.

    This structure allows silence to be scored and selected
    just like any other response policy.
    """

    # Policy identification (matches other policy format)
    policy_label: str = "沈黙する"
    rationale: str = "今は言葉にしない"

    # Silence-specific attributes
    silence_type: SilenceType = SilenceType.DELIBERATE_SILENCE
    suggested_duration: float = 1.0

    # Score (computed during evaluation)
    score: float = 0.0

    # Expected drive changes (silence has different effects)
    expected_drive_change: dict[str, float] = field(default_factory=lambda: {
        "social": 0.02,      # Social need slightly increases
        "curiosity": -0.01,  # Curiosity slightly satisfied (observing)
        "expression": 0.03,  # Expression need builds up
    })

    # Fallback text (for internal logging, not spoken)
    text: str = "..."

    # Metadata
    timestamp: float = field(default_factory=time.time)
    state_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class SilenceResult:
    """
    Result when silence is chosen.

    This is NOT an error. It's a valid decision result with:
    - A suggested duration (NOT blocking sleep)
    - State continuation information
    - Tracking data for consecutive silence detection
    """

    # Was silence chosen?
    is_silence: bool = True

    # Type of silence
    silence_type: SilenceType = SilenceType.DELIBERATE_SILENCE

    # Suggested duration (in seconds) - NOT a blocking sleep
    duration: float = 1.0

    # Internal state continues updating
    state_continues: bool = True

    # Counter for consecutive silences
    consecutive_count: int = 1

    # Reason (for logging/debugging)
    reason: str = ""

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # The candidate that was selected
    candidate: Optional[SilenceCandidate] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "is_silence": self.is_silence,
            "silence_type": self.silence_type.value,
            "duration": self.duration,
            "state_continues": self.state_continues,
            "consecutive_count": self.consecutive_count,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SilenceResult":
        """Deserialize from dict."""
        return cls(
            is_silence=data.get("is_silence", True),
            silence_type=SilenceType(data.get("silence_type", "deliberate_silence")),
            duration=data.get("duration", 1.0),
            state_continues=data.get("state_continues", True),
            consecutive_count=data.get("consecutive_count", 1),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# ── Silence State Tracking ─────────────────────────────────────────


@dataclass
class SilenceState:
    """
    Tracks silence-related state across turns.

    This allows the system to:
    - Count consecutive silences
    - Naturally return to speech after silence
    - Not get stuck in permanent silence
    """

    # Number of consecutive silent turns
    consecutive_silences: int = 0

    # Last silence timestamp
    last_silence_time: Optional[float] = None

    # Total silences in session
    total_silences: int = 0

    # Last non-silence timestamp
    last_speech_time: Optional[float] = None

    def record_silence(self) -> "SilenceState":
        """Record that silence was chosen."""
        return SilenceState(
            consecutive_silences=self.consecutive_silences + 1,
            last_silence_time=time.time(),
            total_silences=self.total_silences + 1,
            last_speech_time=self.last_speech_time,
        )

    def record_speech(self) -> "SilenceState":
        """Record that speech was chosen (resets consecutive count)."""
        return SilenceState(
            consecutive_silences=0,
            last_silence_time=self.last_silence_time,
            total_silences=self.total_silences,
            last_speech_time=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "consecutive_silences": self.consecutive_silences,
            "last_silence_time": self.last_silence_time,
            "total_silences": self.total_silences,
            "last_speech_time": self.last_speech_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SilenceState":
        """Deserialize from dict."""
        return cls(
            consecutive_silences=data.get("consecutive_silences", 0),
            last_silence_time=data.get("last_silence_time"),
            total_silences=data.get("total_silences", 0),
            last_speech_time=data.get("last_speech_time"),
        )


# ── Core Functions ─────────────────────────────────────────────────


def generate_silence_candidate(
    state: PsycheState,
    percept: Optional[Percept] = None,
    config: Optional[SilenceConfig] = None,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    silence_state: Optional[SilenceState] = None,
) -> SilenceCandidate:
    """
    Generate a silence candidate for the decision process.

    This candidate is scored and can compete with other response options.
    Silence is NOT automatically selected - it must win the scoring.

    Args:
        state: Current psychological state (read-only reference)
        percept: Current percept (optional)
        config: Silence configuration
        responsibility_influence: Responsibility influence (optional)
        decision_bias: Decision bias from STM/dynamics (optional)
        silence_state: Current silence tracking state (optional)

    Returns:
        SilenceCandidate that can be included in policy candidates
    """
    cfg = config or SilenceConfig()
    ss = silence_state or SilenceState()

    # Determine silence type based on state
    silence_type = _determine_silence_type(state, percept)

    # Compute score
    score = evaluate_silence_score(
        state, percept, cfg, responsibility_influence, decision_bias, ss
    )

    # Compute suggested duration
    duration = _compute_duration(state, cfg)

    # Determine rationale
    rationale = _get_silence_rationale(silence_type)

    return SilenceCandidate(
        policy_label="沈黙する",
        rationale=rationale,
        silence_type=silence_type,
        suggested_duration=duration,
        score=score,
        text="...",
        state_snapshot={
            "dominant_emotion": state.dominant_emotion,
            "mood_valence": state.mood.valence,
            "fear_level": state.fear_level,
        },
    )


def evaluate_silence_score(
    state: PsycheState,
    percept: Optional[Percept] = None,
    config: Optional[SilenceConfig] = None,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    silence_state: Optional[SilenceState] = None,
) -> float:
    """
    Evaluate the score for choosing silence.

    This uses the same scoring approach as other policies,
    allowing silence to compete fairly.

    Returns:
        Score for silence (higher = more likely to be chosen)
    """
    cfg = config or SilenceConfig()
    ss = silence_state or SilenceState()

    score = cfg.base_silence_score

    # ── State-based scoring ──

    # 1. Emotional conflict bonus
    # When multiple conflicting emotions are active, hesitation is natural
    emotions = state.emotions.as_dict()
    active_count = sum(1 for v in emotions.values() if v > 0.2)
    if active_count >= 3:
        score += cfg.emotional_conflict_bonus * 0.5

    # Check for opposing emotions
    joy_sorrow_conflict = min(emotions.get("joy", 0), emotions.get("sorrow", 0))
    if joy_sorrow_conflict > 0.2:
        score += cfg.emotional_conflict_bonus

    # 2. Mood-based scoring
    mood_val = state.mood.valence
    if mood_val < -0.3:
        # Negative mood → more likely to be silent
        score += cfg.negative_mood_bonus * abs(mood_val)
    elif mood_val > 0.3:
        # Positive mood → less likely to be silent
        score -= cfg.positive_mood_penalty * mood_val

    # 3. Arousal-based scoring
    arousal = state.mood.arousal
    if arousal < 0.3:
        # Low arousal → more contemplative, silence is natural
        score += cfg.low_arousal_bonus * (0.3 - arousal)
    elif arousal > 0.7:
        # High arousal → less likely to be silent
        score -= (arousal - 0.7) * 1.5

    # 4. Drive-based penalties
    drives = state.drives
    if drives.expression > 0.6:
        score -= cfg.high_expression_drive_penalty * (drives.expression - 0.6)
    if drives.social > 0.6:
        score -= cfg.high_social_drive_penalty * (drives.social - 0.6)

    # 5. Fear-based scoring
    if state.fear_index:
        # High fear can cause hesitation
        total_fear = state.fear_level
        if total_fear > 0.4:
            score += cfg.high_uncertainty_bonus * (total_fear - 0.4)

    # ── Responsibility influence ──

    if responsibility_influence:
        # Carrying responsibility weight → more contemplative
        if responsibility_influence.anxiety_baseline > 0.2:
            score += cfg.responsibility_weight_bonus * responsibility_influence.anxiety_baseline

        # High caution → pause before speaking
        if responsibility_influence.caution_bias > 0.3:
            score += responsibility_influence.caution_bias * 0.5

    # ── Percept-based scoring ──

    if percept:
        # Negative valence input → silence as processing time
        if percept.emotion_valence < -0.5:
            score += 0.5

        # If user is sharing deeply, silence can be respectful
        if percept.intent == "sharing":
            score += 0.3

    # ── Consecutive silence penalty ──

    # Prevent getting stuck in silence
    if ss.consecutive_silences >= cfg.max_consecutive_silences and cfg.max_consecutive_silences > 0:
        score -= 3.0  # Strong penalty to break silence streak

    # Slight penalty for each consecutive silence
    score -= ss.consecutive_silences * 0.5

    # ── Minimum threshold ──

    if score < cfg.minimum_score_threshold:
        score = cfg.minimum_score_threshold * 0.5  # Very low but not zero

    return max(0.0, score)


def _determine_silence_type(
    state: PsycheState,
    percept: Optional[Percept] = None,
) -> SilenceType:
    """Determine the type of silence based on state."""
    emotions = state.emotions.as_dict()

    # Check for emotional conflict
    joy = emotions.get("joy", 0)
    sorrow = emotions.get("sorrow", 0)
    fear = emotions.get("fear", 0)

    if min(joy, sorrow) > 0.2 or min(joy, fear) > 0.2:
        return SilenceType.EMOTIONAL_HESITATION

    # Check for uncertainty
    if state.fear_level > 0.5:
        return SilenceType.UNCERTAIN_PAUSE

    # Check percept context
    if percept and percept.intent == "sharing":
        return SilenceType.RESPECTFUL_SILENCE

    # Low arousal = processing
    if state.mood.arousal < 0.3:
        return SilenceType.PROCESSING_PAUSE

    # Default
    return SilenceType.DELIBERATE_SILENCE


def _compute_duration(
    state: PsycheState,
    config: SilenceConfig,
) -> float:
    """Compute suggested silence duration based on state."""
    base = config.default_duration

    # Longer silence when more conflicted
    emotions = state.emotions.as_dict()
    active_count = sum(1 for v in emotions.values() if v > 0.2)
    if active_count >= 4:
        base += 0.5

    # Longer when mood is very negative
    if state.mood.valence < -0.5:
        base += 0.5

    # Shorter when arousal is high
    if state.mood.arousal > 0.6:
        base -= 0.3

    return max(config.min_duration, min(config.max_duration, base))


def _get_silence_rationale(silence_type: SilenceType) -> str:
    """Get rationale text for silence type."""
    rationales = {
        SilenceType.EMOTIONAL_HESITATION: "感情が揺れている",
        SilenceType.PROCESSING_PAUSE: "考えを整理している",
        SilenceType.RESPECTFUL_SILENCE: "静かに聴いている",
        SilenceType.UNCERTAIN_PAUSE: "言葉を探している",
        SilenceType.DELIBERATE_SILENCE: "今は言葉にしない",
    }
    return rationales.get(silence_type, "沈黙")


# ── Result Creation ────────────────────────────────────────────────


def create_silence_result(
    candidate: SilenceCandidate,
    state: PsycheState,
    silence_state: Optional[SilenceState] = None,
) -> SilenceResult:
    """
    Create a silence result from a selected silence candidate.

    This result indicates:
    - Silence was chosen (not an error)
    - Suggested duration (NOT blocking)
    - State continues updating

    Args:
        candidate: The selected silence candidate
        state: Current state (for context)
        silence_state: Tracking state (optional)

    Returns:
        SilenceResult that the system can act on
    """
    ss = silence_state or SilenceState()

    return SilenceResult(
        is_silence=True,
        silence_type=candidate.silence_type,
        duration=candidate.suggested_duration,
        state_continues=True,  # Always true - silence doesn't stop the system
        consecutive_count=ss.consecutive_silences + 1,
        reason=candidate.rationale,
        candidate=candidate,
    )


def create_speech_result() -> SilenceResult:
    """Create a result indicating speech (not silence)."""
    return SilenceResult(
        is_silence=False,
        silence_type=SilenceType.DELIBERATE_SILENCE,
        duration=0.0,
        state_continues=True,
        consecutive_count=0,
        reason="speaking",
    )


# ── Policy Integration ─────────────────────────────────────────────


def silence_candidate_to_policy(candidate: SilenceCandidate) -> dict[str, Any]:
    """
    Convert a SilenceCandidate to the standard policy dict format.

    This allows silence to be included in the candidates list
    alongside other policies.
    """
    return {
        "policy_label": candidate.policy_label,
        "rationale": candidate.rationale,
        "expected_drive_change": candidate.expected_drive_change,
        "text": candidate.text,
        "_score": candidate.score,
        "_is_silence": True,
        "_silence_type": candidate.silence_type.value,
        "_silence_duration": candidate.suggested_duration,
    }


def is_silence_policy(policy: dict[str, Any]) -> bool:
    """Check if a policy dict represents silence."""
    return policy.get("_is_silence", False) or policy.get("policy_label") == "沈黙する"


def is_silence_result(result: Any) -> bool:
    """Check if a result is a silence result."""
    if isinstance(result, SilenceResult):
        return result.is_silence
    if isinstance(result, dict):
        return result.get("_is_silence", False) or result.get("policy_label") == "沈黙する"
    return False


def get_silence_duration(policy: dict[str, Any]) -> float:
    """Get the suggested silence duration from a policy."""
    return policy.get("_silence_duration", 1.0)


# ── Extended Policy Generation ─────────────────────────────────────


def generate_candidates_with_silence(
    state: PsycheState,
    percept: Percept,
    recalled: list[dict],
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    silence_config: Optional[SilenceConfig] = None,
    silence_state: Optional[SilenceState] = None,
    base_candidates: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Generate policy candidates including silence option.

    This wraps the standard candidate generation and adds
    silence as a scorable option.

    Args:
        state: Current psychological state
        percept: Current percept
        recalled: Retrieved memories
        responsibility_influence: Optional responsibility influence
        decision_bias: Optional decision bias
        silence_config: Optional silence configuration
        silence_state: Optional silence tracking state
        base_candidates: Pre-generated base candidates (if available)

    Returns:
        List of policy candidates including silence
    """
    # Generate silence candidate
    silence = generate_silence_candidate(
        state=state,
        percept=percept,
        config=silence_config,
        responsibility_influence=responsibility_influence,
        decision_bias=decision_bias,
        silence_state=silence_state,
    )

    # Convert to policy format
    silence_policy = silence_candidate_to_policy(silence)

    # Combine with base candidates
    if base_candidates is None:
        # Import here to avoid circular dependency
        from .thought import generate_thought_candidates
        base_candidates = generate_thought_candidates(
            state, percept, recalled, responsibility_influence, decision_bias
        )

    # Add silence to candidates
    all_candidates = list(base_candidates) + [silence_policy]

    # Re-sort by score
    all_candidates.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return all_candidates


# ── Utility Functions ──────────────────────────────────────────────


def get_silence_summary(result: SilenceResult) -> str:
    """Get human-readable summary of silence result."""
    if not result.is_silence:
        return "Not silence - speaking"

    return (
        f"Silence ({result.silence_type.value}): "
        f"duration={result.duration:.1f}s, "
        f"consecutive={result.consecutive_count}, "
        f"reason=\"{result.reason}\""
    )


def create_silence_config(
    base_score: float = 0.5,
    min_duration: float = 0.5,
    max_duration: float = 3.0,
    max_consecutive: int = 2,
) -> SilenceConfig:
    """Create a silence configuration with custom parameters."""
    return SilenceConfig(
        base_silence_score=base_score,
        min_duration=min_duration,
        max_duration=max_duration,
        max_consecutive_silences=max_consecutive,
    )


def to_dict(config: SilenceConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "base_silence_score": config.base_silence_score,
        "minimum_score_threshold": config.minimum_score_threshold,
        "min_duration": config.min_duration,
        "max_duration": config.max_duration,
        "default_duration": config.default_duration,
        "emotional_conflict_bonus": config.emotional_conflict_bonus,
        "high_uncertainty_bonus": config.high_uncertainty_bonus,
        "low_arousal_bonus": config.low_arousal_bonus,
        "responsibility_weight_bonus": config.responsibility_weight_bonus,
        "negative_mood_bonus": config.negative_mood_bonus,
        "high_expression_drive_penalty": config.high_expression_drive_penalty,
        "high_social_drive_penalty": config.high_social_drive_penalty,
        "positive_mood_penalty": config.positive_mood_penalty,
        "allow_as_top_choice": config.allow_as_top_choice,
        "max_consecutive_silences": config.max_consecutive_silences,
    }


def from_dict(data: dict[str, Any]) -> SilenceConfig:
    """Deserialize config from dict."""
    return SilenceConfig(
        base_silence_score=data.get("base_silence_score", 0.5),
        minimum_score_threshold=data.get("minimum_score_threshold", 0.3),
        min_duration=data.get("min_duration", 0.5),
        max_duration=data.get("max_duration", 3.0),
        default_duration=data.get("default_duration", 1.0),
        emotional_conflict_bonus=data.get("emotional_conflict_bonus", 2.0),
        high_uncertainty_bonus=data.get("high_uncertainty_bonus", 1.5),
        low_arousal_bonus=data.get("low_arousal_bonus", 1.0),
        responsibility_weight_bonus=data.get("responsibility_weight_bonus", 1.5),
        negative_mood_bonus=data.get("negative_mood_bonus", 1.2),
        high_expression_drive_penalty=data.get("high_expression_drive_penalty", 2.0),
        high_social_drive_penalty=data.get("high_social_drive_penalty", 1.5),
        positive_mood_penalty=data.get("positive_mood_penalty", 0.8),
        allow_as_top_choice=data.get("allow_as_top_choice", True),
        max_consecutive_silences=data.get("max_consecutive_silences", 2),
    )
